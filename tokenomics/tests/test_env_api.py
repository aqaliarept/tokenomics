"""PettingZoo API compliance + random rollout (ADR-0025 model).

phase1.json is re-derived in B7; until then the env tests run on the composite
supply-chain fixture (eligibility holds: every product's required capacity types
are owned by a single producer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pettingzoo.test import api_test, seed_test

from tokenomics.domain import build_system
from tokenomics.domain.common import from_producer_id
from tokenomics.domain.config import load_config
from tokenomics.domain.producing_agent import ProducingAgentCommand
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.spaces import build_spec

COMPOSITE = Path(__file__).resolve().parent / "fixtures" / "composite.json"
PHASE1 = Path(__file__).resolve().parents[1] / "tokenomics" / "config" / "phase1.json"


def _cfg():
    return load_config(COMPOSITE)


def _env(horizon: int) -> TokenomicsAECEnv:
    return TokenomicsAECEnv(config=_cfg(), horizon=horizon)


def test_pettingzoo_api():
    api_test(_env(horizon=40), num_cycles=20, verbose_progress=False)


def test_seed_determinism():
    # env is deterministic (scripted consumers); same seed -> identical run
    seed_test(lambda: _env(horizon=30))


def test_random_rollout_to_termination():
    e = _env(horizon=50)
    e.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    for agent in e.agent_iter(max_iter=10000):
        obs, reward, term, trunc, info = e.last()
        if term or trunc:
            action = None
        else:
            space = e.action_space(agent)
            space.seed(int(rng.integers(0, 2**31)))
            action = space.sample()
        e.step(action)
        steps += 1
    assert steps > 0
    assert len(e.agents) == 0  # all agents reached done


def test_observation_in_space():
    e = _env(horizon=10)
    e.reset()
    for agent in e.possible_agents:
        obs = e.observe(agent)
        assert e.observation_space(agent).contains(obs), agent


def test_k_slots_cover_worst_case_producer_orders():
    # A producer's ordering agent can hold, in one cycle, 3 investment orders
    # (restoration/upgrade/expansion) plus one replenishment order per relevant
    # input. K must cover that so no order is dropped to an unobservable,
    # uncontrollable even-split fallback in decode (ADR-0025).
    cfg = load_config(PHASE1)  # shipped config: costs fund every order type
    sp = build_spec(cfg)
    sysm = build_system(cfg)
    full = ProducingAgentCommand(True, True, True, {20: 3, 11: 3})  # all 3 investments + 2 replenish
    for pid in sysm.producing_agents:
        sysm.producing_agent_action(pid, full if pid == "p4" else ProducingAgentCommand(False, False))
    sysm.start_ordering()
    counts = {aid: len(oa.incoming) for aid, oa in sysm.ordering_agents.items()}
    assert counts[from_producer_id("p4")] > 3  # the scenario really exceeds the old K=3
    for aid, n in counts.items():
        assert n <= sp.k, (aid, n, sp.k)


def test_config_fingerprint_is_stable_and_shape_sensitive():
    # #7: a checkpoint stamps the fingerprint of the config it was trained on, so
    # evaluating it on a shape-incompatible config can be caught instead of crashing
    # deep in the actor (the B9 wrong-config gotcha). Same config -> same fingerprint;
    # a config with different spec dims -> different fingerprint.
    from tokenomics.domain.config import load_config
    from tokenomics.env.spaces import config_fingerprint

    cfg = load_config(PHASE1)
    fp = config_fingerprint(cfg)
    assert isinstance(fp, str) and fp
    assert config_fingerprint(load_config(PHASE1)) == fp  # deterministic across loads

    # add a producer -> producer count (and thus obs/global dims) changes -> new fp
    from dataclasses import replace
    extra = replace(cfg.producer_configs[0], id="p_extra")
    wider = replace(cfg, producer_configs=[*cfg.producer_configs, extra])
    assert config_fingerprint(wider) != fp


def test_config_fingerprint_separates_same_shape_different_economy():
    # [P1 review 2026-06-15]: a shape-only fingerprint collides for configs with
    # identical IDs/dims but different DYNAMICS, so signal_check reports a valid match
    # for an off-config comparison. The fingerprint must capture the economy too:
    #  - competition.json vs complex.json: same topology, different demand distribution
    #  - random_demand.json vs composite.json: different recipes + capacities
    from tokenomics.env.spaces import build_spec, config_fingerprint

    fx = Path(__file__).resolve().parent / "fixtures"
    for a, b in [("competition.json", "complex.json"),
                 ("random_demand.json", "composite.json")]:
        ca, cb = load_config(fx / a), load_config(fx / b)
        sa, sb = build_spec(ca), build_spec(cb)
        # same tensor shape -> P2's dim check alone would wave these through as "compatible"
        assert sa.producing_obs_dim == sb.producing_obs_dim
        assert sa.ordering_obs_dim == sb.ordering_obs_dim
        # ...but different economy -> fingerprints must differ
        assert config_fingerprint(ca) != config_fingerprint(cb), f"{a} vs {b} collide"
    # stability preserved: same economy -> same fingerprint across loads
    assert config_fingerprint(load_config(fx / "complex.json")) == config_fingerprint(
        load_config(fx / "complex.json")
    )


def test_global_state_shape():
    e = _env(horizon=10)
    e.reset()
    gs = e.global_state()
    assert gs.shape == (e.spec_.global_state_dim,)


def test_producing_obs_exposes_open_consumer_demand():
    # #2: consumer orders exist before producers act, but the producing obs used to
    # carry no demand. Producers now see aggregate open consumer demand per product
    # (quantity then tokens, indexed by order_product_index) just before the
    # cycle-progress scalar, normalized by inv_ref / token_ref.
    import pytest
    from collections import defaultdict

    e = _env(horizon=10)
    e.reset()
    sp = e.spec_
    open_qty: dict[int, float] = defaultdict(float)
    open_tok: dict[int, float] = defaultdict(float)
    for o in e.system.orders.values():
        if o.consumer_request is not None:
            open_qty[o.product] += o.unresolved_quantity()
            open_tok[o.product] += o.available_tokens()
    assert open_qty  # phase1 places a consumer order each cycle

    prod_agent = next(a for a in e.possible_agents if a.startswith("prod_"))
    obs = e.observe(prod_agent)
    p = sp.num_order_products
    qty_block = obs[-(2 * p + 1):-(p + 1)]  # first P of the demand block = quantities
    tok_block = obs[-(p + 1):-1]            # next P = backing tokens
    for product, q in open_qty.items():
        idx = sp.order_product_index[product]
        assert qty_block[idx] == pytest.approx(q / sp.inv_ref)
        assert tok_block[idx] == pytest.approx(open_tok[product] / sp.token_ref)


def test_producing_obs_exposes_outstanding_inbound():
    # Finding A (deep-chain viability): priming a multi-hop pipeline requires the
    # policy to see what it has ALREADY ordered (outstanding inbound). Without it a
    # producer re-orders in-flight stock every cycle and a deep chain never primes
    # (depth-4 fulfilment collapses). The producing obs exposes a per-input inbound
    # block right after the per-input inventory-stock block, normalized by inv_ref.
    import pytest

    e = _env(horizon=10)
    e.reset()
    sp = e.spec_
    pid = next(p for p in sp.producer_ids if sp.producer_inputs[p])
    g = sp.producer_inputs[pid][0]
    e.system.producing_agents[pid].outstanding_inbound[g] = 7

    obs = e.observe(f"prod_{pid}")
    t = sp.num_types
    inv_start = 2 * t + 4               # caps(2T) + requested(1) + 3 running flags
    inbound_start = inv_start + sp.num_inputs  # after the per-input stock block
    assert obs[inbound_start + sp.input_index[g]] == pytest.approx(7 / sp.inv_ref)


def test_global_state_exposes_open_consumer_demand():
    # #2: the centralized critic (ADR-0004) also needs demand. global_state appends
    # the same 2P open-consumer-demand block (quantity then tokens) before the
    # cycle-progress scalar.
    import pytest
    from collections import defaultdict

    e = _env(horizon=10)
    e.reset()
    sp = e.spec_
    open_qty: dict[int, float] = defaultdict(float)
    for o in e.system.orders.values():
        if o.consumer_request is not None:
            open_qty[o.product] += o.unresolved_quantity()
    assert open_qty

    gs = e.global_state()
    p = sp.num_order_products
    qty_block = gs[-(2 * p + 1):-(p + 1)]
    for product, q in open_qty.items():
        idx = sp.order_product_index[product]
        assert qty_block[idx] == pytest.approx(q / sp.inv_ref)


def test_reward_shape():
    e = _env(horizon=5)
    e.reset()
    rng = np.random.default_rng(2)
    reward = 0.0
    for agent in e.agent_iter(max_iter=200):
        obs, reward, term, trunc, info = e.last()
        if term or trunc:
            e.step(None)
            continue
        sp = e.action_space(agent)
        sp.seed(int(rng.integers(0, 2**31)))
        e.step(sp.sample())
        if e.current_cycle >= 1:
            break
    assert isinstance(reward, float)


# --- Finding B (ADR-0042): reward = consumer fulfilment − per-producer holding ---
def test_reward_is_fulfilment_minus_per_producer_holding_only():
    # At producer-scale, instrumental producer-order churn (investment/replenishment
    # completed/rejected, in-progress) and per-unit consumer rejection swamped the
    # +10/unit fulfilment objective — do-nothing became the return-maximizer. The
    # training reward is now consumer fulfilment minus a holding penalty normalized
    # per producer (so it does not grow with producer count).
    import pytest
    from tokenomics.domain.system import CycleResult
    from tokenomics.env.aec_env import R_CONSUMER_PER_UNIT, R_HOLDING_PER_UNIT, cycle_reward

    r = CycleResult(score=0, consumer_completed_units=4, consumer_rejected_units=9,
                    in_progress=50, investment_rejected=30, replenishment_rejected=20,
                    investment_completed=7, replenishment_completed=8)
    expected = R_CONSUMER_PER_UNIT * 4 - R_HOLDING_PER_UNIT * 100 / 10
    assert cycle_reward(r, held=100, num_producers=10) == pytest.approx(expected)
    # only consumer completions and held inventory move the reward; all the producer-
    # order churn and consumer rejections above are ignored.
    churn = CycleResult(score=0, consumer_completed_units=4, consumer_rejected_units=999,
                        in_progress=999, investment_rejected=999, replenishment_rejected=999,
                        investment_completed=999, replenishment_completed=999)
    assert cycle_reward(churn, held=100, num_producers=10) == pytest.approx(expected)


# --- B6 reward: inventory holding penalty ---
def test_holding_penalty_scales_with_units_held():
    from tokenomics.domain.system import CycleResult
    from tokenomics.env.aec_env import R_HOLDING_PER_UNIT, cycle_reward

    base = CycleResult(score=0)  # no completions/rejections
    r0 = cycle_reward(base, held=0)
    r5 = cycle_reward(base, held=5)
    r10 = cycle_reward(base, held=10)
    assert r0 == 0.0
    assert r5 == -5 * R_HOLDING_PER_UNIT
    # penalty is linear in units held
    assert (r0 - r5) == (r5 - r10)


def test_consumer_reward_scales_with_fulfilled_units():
    # #3: a qty-3 order consumes ~3x the capacity of a qty-1 order, so it must earn
    # ~3x the reward. Reward weights consumer UNITS, not order counts. One completed
    # order of 3 units beats one completed order of 1 unit by 2 per-unit rewards.
    import pytest
    from tokenomics.domain.system import CycleResult
    from tokenomics.env.aec_env import R_CONSUMER_PER_UNIT, cycle_reward

    one = CycleResult(score=0, consumer_completed=1, consumer_completed_units=1)
    three = CycleResult(score=0, consumer_completed=1, consumer_completed_units=3)
    assert cycle_reward(three, 0) - cycle_reward(one, 0) == pytest.approx(2 * R_CONSUMER_PER_UNIT)


def test_consumer_rejection_does_not_penalize_reward():
    # Finding B (ADR-0042): consumer rejection is no longer a separate penalty — at
    # scale the −5/unit term swamped fulfilment (demand outstrips a deep chain's
    # throughput, so rejection volume is policy-independent). The forgone +10/unit is
    # the only rejection signal, so a cycle that rejects more is reward-unchanged.
    from tokenomics.domain.system import CycleResult
    from tokenomics.env.aec_env import cycle_reward

    one = CycleResult(score=0, consumer_rejected=1, consumer_rejected_units=1)
    three = CycleResult(score=0, consumer_rejected=1, consumer_rejected_units=3)
    assert cycle_reward(three, 0) == cycle_reward(one, 0)


def test_holding_penalty_independent_of_completions():
    from tokenomics.domain.system import CycleResult
    from tokenomics.env.aec_env import R_HOLDING_PER_UNIT, cycle_reward

    import pytest

    res = CycleResult(score=0, consumer_completed=2)
    delta = cycle_reward(res, held=3) - cycle_reward(res, held=0)
    assert delta == pytest.approx(-3 * R_HOLDING_PER_UNIT)


# --- B9 step 2: seeded consumer demand (product + quantity) ---
def _pool_consumer_env(seed: int, horizon: int = 20) -> TokenomicsAECEnv:
    from tokenomics.domain.config import (
        Configuration,
        ConsumerConfig,
        ProcessSheet,
        ProducingAgentConfig,
        Restoration,
        Upgrade,
    )

    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(p, {f"capacity-{p}": 2}) for p in (1, 2, 3)],
        producer_configs=[
            ProducingAgentConfig(
                f"p{p}", {f"capacity-{p}": 100}, 5,
                Restoration(p, 50, f"capacity-{p}"), Upgrade(p, 40, f"capacity-{p}"),
                produces=frozenset({p}),
            )
            for p in (1, 2, 3)
        ],
        consumers=[ConsumerConfig("c1", product=1, products=(1, 2, 3),
                                  quantity_min=1, quantity_max=4)],
    )
    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=seed)
    return env


def test_seeded_demand_replays_and_varies_by_seed():
    # Same seed -> identical episode demand schedule (deterministic replay); a
    # different seed -> different demand. The env owns the seeded RNG; the domain
    # consumer just replays the injected schedule (CLAUDE.md domain purity).
    sa = _pool_consumer_env(seed=7).system.consumers["c1"]._schedule
    sb = _pool_consumer_env(seed=7).system.consumers["c1"]._schedule
    sc = _pool_consumer_env(seed=8).system.consumers["c1"]._schedule
    assert sa == sb
    assert sa != sc
    assert len(sa) == 20
    assert all(p in {1, 2, 3} and 1 <= q <= 4 for p, q in sa)
    assert len({p for p, _ in sa}) > 1  # product actually varies within the episode
    assert len({q for _, q in sa}) > 1  # quantity actually varies within the episode


def test_unrandomized_consumer_demand_stays_fixed():
    # Absent randomization bounds, demand is the canonical fixed single product at
    # quantity 1 regardless of seed (canonical phase1.json must stay deterministic).
    from tokenomics.domain.consumer import ScriptedConsumer

    env = TokenomicsAECEnv(horizon=10)
    env.reset(seed=123)
    c = env.system.consumers["c1"]
    assert isinstance(c, ScriptedConsumer)


def _fixed_consumer_env(consumer_kw: dict, seed: int = 0) -> TokenomicsAECEnv:
    from tokenomics.domain.config import (
        Configuration,
        ConsumerConfig,
        ProcessSheet,
        ProducingAgentConfig,
        Restoration,
        Upgrade,
    )

    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(1, {"capacity-1": 2})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
        ],
        consumers=[ConsumerConfig("c1", **consumer_kw)],
    )
    env = TokenomicsAECEnv(config=cfg, horizon=10)
    env.reset(seed=seed)
    return env


def test_fixed_nondefault_quantity_is_honored():
    # quantityMin == quantityMax == 2 is deterministic but NOT the legacy default;
    # the consumer must demand quantity 2, never silently revert to 1.
    from tokenomics.domain.consumer import ScheduledConsumer

    c = _fixed_consumer_env(dict(product=1, quantity_min=2, quantity_max=2)).system.consumers["c1"]
    assert isinstance(c, ScheduledConsumer)
    assert c._schedule == [(1, 2)]


def test_episode_tracks_consumer_units_distinct_from_orders():
    # #3: metrics must record consumer UNITS, not just order counts. With every
    # consumer order at quantity 2, delivered units = 2x completed orders — so a
    # randomized-demand suite can no longer hide large orders behind an order count.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import run_episode

    env = _fixed_consumer_env(dict(product=1, quantity_min=2, quantity_max=2))
    s = run_episode(env, PRODUCING_BASELINES["always_restore"], ORDERING_BASELINES["proportional"], seed=0)
    assert s["consumer_completed"] > 0
    assert s["consumer_completed_units"] == 2 * s["consumer_completed"]


def test_singleton_pool_ignores_unrelated_default_product():
    # A singleton `products` pool is the source of truth; the (unvalidated) default
    # `product` field must not leak in and must not crash reset on a missing sheet.
    from tokenomics.domain.consumer import ScheduledConsumer

    c = _fixed_consumer_env(dict(product=999, products=(1,))).system.consumers["c1"]
    assert isinstance(c, ScheduledConsumer)
    assert c._schedule == [(1, 1)]  # product 1 from the pool, not default 999
