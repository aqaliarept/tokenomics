"""EXP-0007A: stable product-stream ordering agents (ADR-0052).

Behaviour-preservation tests. Splitting a source's orders across per-product
ordering agents must not change per-order routing, so deterministic baselines
must produce bit-identical episode metrics under the product-stream contract.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pettingzoo.test import api_test

from tokenomics.domain.config import load_config
from tokenomics.env.aec_env import TokenomicsAECEnv, DEFAULT_CONFIG
from tokenomics.env.spaces import (
    STREAM_PRODUCER_FEATURES,
    STREAM_PRODUCT_FEATURES,
    STREAM_SLOT_SCALARS,
    build_spec,
    config_fingerprint,
)
from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
from tokenomics.eval.runner import run_episode


def _stream_obs_dim(P: int, K: int) -> int:
    return STREAM_PRODUCT_FEATURES + K * STREAM_SLOT_SCALARS + P * STREAM_PRODUCER_FEATURES + 1


def _episode(order_policy: str, *, product_stream: bool) -> dict:
    env = TokenomicsAECEnv(horizon=100, product_stream_ordering=product_stream)
    return run_episode(
        env,
        PRODUCING_BASELINES["stock_aware"],
        ORDERING_BASELINES[order_policy],
        seed=0,
    )


def test_proportional_parity() -> None:
    # Tracer bullet: per-order routing is independent of agent grouping, so the
    # product-stream contract must reproduce the legacy metrics exactly.
    legacy = _episode("proportional", product_stream=False)
    stream = _episode("proportional", product_stream=True)
    assert stream == legacy


def test_even_split_parity() -> None:
    legacy = _episode("even_split", product_stream=False)
    stream = _episode("even_split", product_stream=True)
    assert stream == legacy


def test_stream_agents_are_restricted_and_deterministic() -> None:
    # Restricted coverage (ADR-0052): consumer pool products + producer
    # replenishment/restoration/upgrade/inventory requirement products only.
    env = TokenomicsAECEnv(horizon=100, product_stream_ordering=True)
    order_agents = [a for a in env.possible_agents if a.startswith("order_")]
    assert order_agents == [
        "order_c1__30",
        "order_p1__11",
        "order_p2__10",
        "order_p3__10",
        "order_p3__11",
        "order_p4__10",
        "order_p4__11",
        "order_p4__20",
    ]
    # stable across construction
    again = TokenomicsAECEnv(horizon=100, product_stream_ordering=True)
    assert again.possible_agents == env.possible_agents


def test_stream_agent_holds_only_its_product() -> None:
    # The whole point of ADR-0052: a stream agent never multiplexes products.
    env = TokenomicsAECEnv(horizon=100, product_stream_ordering=True)
    seen_nonempty = False

    def checking_orderer(e: TokenomicsAECEnv, agent: str):
        _, did = e._role[agent]
        product = int(did.split("__")[1])
        incoming = e.system.ordering_agents[did].incoming
        nonlocal seen_nonempty
        seen_nonempty = seen_nonempty or bool(incoming)
        assert all(oi.product == product for oi in incoming.values())
        return ORDERING_BASELINES["proportional"](e, agent)

    run_episode(env, PRODUCING_BASELINES["stock_aware"], checking_orderer, seed=0)
    assert seen_nonempty  # the invariant was actually exercised


def test_pettingzoo_api_stream_mode() -> None:
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    api_test(env, num_cycles=20, verbose_progress=False)


# --- product_stream_v1 observation contract (orderer_obs_version 3) ---


def test_stream_mode_uses_obs_version_three() -> None:
    # Tracer: product-stream mode selects the simplified per-stream obs contract,
    # whose width scales with producers (P rows) + backlog (K), never with the
    # product vocabulary (no Q one-hot).
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    assert env.spec_.orderer_obs_version == 3
    expected = _stream_obs_dim(env.spec_.num_producers, env.spec_.k)
    assert env.spec_.ordering_obs_dim == expected

    env.reset(seed=0)
    order_agent = next(a for a in env.possible_agents if a.startswith("order_"))
    assert env.observe(order_agent).shape == (expected,)


def _relabel(config, f):
    """Apply a product-id bijection f to every product reference in a config."""
    sheets = [
        replace(ps, product=f(ps.product), inputs={f(g): n for g, n in ps.inputs.items()})
        for ps in config.process_sheets
    ]
    producers = []
    for p in config.producer_configs:
        ie = p.inventory_expansion
        producers.append(replace(
            p,
            produces=frozenset(f(x) for x in p.produces),
            restoration=replace(p.restoration, require=f(p.restoration.require)),
            upgrade=replace(p.upgrade, require=f(p.upgrade.require)),
            inventory_expansion=replace(ie, require=f(ie.require)) if ie is not None else None,
        ))
    consumers = [
        replace(c, product=f(c.product), products=tuple(f(x) for x in c.products))
        for c in config.consumers
    ]
    return replace(config, process_sheets=sheets, producer_configs=producers, consumers=consumers)


def _all_observations(config, seed: int) -> list[np.ndarray]:
    """Drive a deterministic baseline episode, recording each acting agent's
    observation in order."""
    env = TokenomicsAECEnv(config=config, horizon=40, product_stream_ordering=True)
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    obs_log: list[np.ndarray] = []
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    for agent in env.agent_iter(max_iter=max_iter):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        obs_log.append(env.observe(agent).copy())
        role = env._role[agent][0]
        policy = PRODUCING_BASELINES["stock_aware"] if role == "prod" else ORDERING_BASELINES["proportional"]
        env.step(policy(env, agent))
    return obs_log


def test_stream_obs_dim_independent_of_product_count() -> None:
    # Scales with producers, not products: adding an orderable product (no new
    # producer, no new replenish stream) leaves the v3 width unchanged, while the
    # product-multiplexing v2 width grows.
    base = load_config(DEFAULT_CONFIG)
    extra_sheet = replace(base.process_sheets[0], product=999, require={"capacity-4": 2}, inputs={})
    p4 = next(p for p in base.producer_configs if p.id == "p4")
    grown_p4 = replace(p4, produces=p4.produces | {999})
    grown = replace(
        base,
        process_sheets=[*base.process_sheets, extra_sheet],
        producer_configs=[grown_p4 if p.id == "p4" else p for p in base.producer_configs],
    )
    v3_base = build_spec(base, orderer_obs_version=3)
    v3_grown = build_spec(grown, orderer_obs_version=3)
    assert v3_grown.num_producers == v3_base.num_producers
    assert v3_grown.ordering_obs_dim == v3_base.ordering_obs_dim  # no Q dependence
    # sanity: the product-multiplexing v2 contract DOES grow with product count.
    assert (
        build_spec(grown, orderer_obs_version=2).ordering_obs_dim
        > build_spec(base, orderer_obs_version=2).ordering_obs_dim
    )


def test_fingerprint_separates_stream_from_legacy_contracts() -> None:
    # A v3 (stream) checkpoint must not be mis-scored against a v1/v2 economy.
    cfg = load_config(DEFAULT_CONFIG)
    fps = {config_fingerprint(cfg, v) for v in (1, 2, 3)}
    assert len(fps) == 3


def test_stream_obs_invariant_to_product_relabeling() -> None:
    # Decisions depend on economy structure, not product identity: a monotonic
    # relabel of every product id yields bit-identical observations throughout the
    # episode (and hence identical routing), since the v3 obs carries no product id.
    base = load_config(DEFAULT_CONFIG)
    relabeled = _relabel(base, lambda x: x + 100)
    base_obs = _all_observations(base, seed=0)
    relabeled_obs = _all_observations(relabeled, seed=0)
    assert len(base_obs) == len(relabeled_obs)
    for a, b in zip(base_obs, relabeled_obs):
        assert np.array_equal(a, b)


def test_stream_producer_rows_are_keyed_by_producer() -> None:
    # Producer-row equivariance: permuting the producer order relocates each
    # producer's row but never changes its contents, so a permutation of producers
    # maps cleanly onto a permutation of the action columns.
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    env.reset(seed=0)
    from tokenomics.env.encode import encode_ordering_stream

    spec = env.spec_
    view = env.system.product_stream_view("c1__30")
    obs = encode_ordering_stream(spec, view, 0.0)

    permuted = replace(spec, producer_ids=list(reversed(spec.producer_ids)))
    obs_perm = encode_ordering_stream(permuted, view, 0.0)

    head = STREAM_PRODUCT_FEATURES + spec.k * STREAM_SLOT_SCALARS
    F = STREAM_PRODUCER_FEATURES

    def row(vec, ordering, pid):
        j = ordering.index(pid)
        return vec[head + j * F : head + (j + 1) * F]

    for pid in spec.producer_ids:
        assert np.array_equal(row(obs, spec.producer_ids, pid), row(obs_perm, permuted.producer_ids, pid))


def test_decode_routes_by_producer_order() -> None:
    # Action-column equivariance: a logit peaked at a producer routes the order to
    # that producer regardless of where it sits in the producer ordering.
    from tokenomics.domain.order import OrderInfo
    from tokenomics.domain.ordering_agent import OrderingAgentView
    from tokenomics.env.decode import decode_ordering

    spec = build_spec(load_config(DEFAULT_CONFIG), orderer_obs_version=3)
    infos = {pid: None for pid in spec.producer_ids}
    oi = OrderInfo("o1", tokens=10, product=30, quantity=4)
    # pretend two producers are eligible for this order
    p_a, p_b = spec.producer_ids[0], spec.producer_ids[2]
    view = OrderingAgentView(incoming={"o1": oi}, eligible={"o1": {p_a: infos[p_a], p_b: infos[p_b]}})

    for target in (p_a, p_b):
        action = np.zeros((spec.k, spec.num_producers), dtype=np.float32)
        action[0, spec.producer_ids.index(target)] = 10.0  # peak on the target column
        cmd = decode_ordering(spec, action.flatten(), view)
        routed = {s.producer_id: s.quantity for s in cmd.orders["o1"] if s.quantity > 0}
        assert max(routed, key=routed.get) == target


def test_stream_view_is_single_product_with_full_producer_rows() -> None:
    # Domain owns the stream-local facts (ADR-0052): the view fixes one product,
    # carries a row for every producer, and its backlog holds only that product.
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    env.reset(seed=0)
    view = env.system.product_stream_view("c1__30")
    assert view.product == 30
    assert set(view.producers) == set(env.spec_.producer_ids)
    assert set(view.producer_products) == set(env.spec_.producer_ids)
    assert all(slot.product == 30 for slot in view.backlog)


def test_stream_encoder_length_matches_spec() -> None:
    # Lockstep guard: encoder width == spec.ordering_obs_dim for both an active and
    # an idle stream (an idle producer stream still emits product + producer rows).
    from tokenomics.env.encode import encode_ordering_stream

    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True)
    env.reset(seed=0)
    dim = env.spec_.ordering_obs_dim
    for aid in ("c1__30", "p1__11"):  # active (consumer order placed) and idle
        vec = encode_ordering_stream(env.spec_, env.system.product_stream_view(aid), 0.0)
        assert vec.shape == (dim,)


# --- eval bars must be scored in the candidate's agent layout (review P2) ---


def test_random_envelope_is_layout_sensitive() -> None:
    # The premise behind the eval-bar fix: `random` ordering draws action-space RNG
    # over the agent set, and the (source, product) stream split changes that set —
    # so the random-inclusive envelope is a *different bar* under each layout, even
    # though the deterministic baselines are grouping-invariant (parity). A stream-mode
    # candidate must therefore be scored against a stream-mode envelope.
    from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
    from tokenomics.eval.runner import evaluate
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(8, depth=3)  # s8 envelope-best is stock_aware+random
    sa, rnd = PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["random"]
    legacy = evaluate(sa, rnd, seeds=range(3), horizon=100, config=cfg, product_stream=False)
    stream = evaluate(sa, rnd, seeds=range(3), horizon=100, config=cfg, product_stream=True)
    assert legacy["consumer_completed_units"][0] != stream["consumer_completed_units"][0]


def test_eval_m2_scores_envelope_in_candidate_layout() -> None:
    # Regression for review P2: eval_m2 must report an env_cu computed in the SAME
    # layout as the (stream-mode) M2 candidate — i.e. the stream-mode envelope, not a
    # legacy-mode one. The M2 policy quality is irrelevant here (random init); env_cu is
    # baseline-only, so we only assert the bar's layout.
    pytest.importorskip("torch")
    from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
    from tokenomics.eval.orderer_oracle import ENVELOPE_COMBOS
    from tokenomics.eval.runner import evaluate
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.orderer_improve import eval_m2
    from tokenomics.train.smoke_ppo import GaussianActor, save_policies

    import tempfile, os

    depth, seed, eval_seeds = 3, 8, 3
    cfg = build_scale_config(seed, depth=depth)
    probe = TokenomicsAECEnv(config=cfg, horizon=1, product_stream_ordering=True)
    prod = GaussianActor(probe.observation_space(probe._prod_agents[0]).shape[0],
                         probe.spec_.producing_action_dim)
    order = GaussianActor(probe.observation_space(probe._order_agents[0]).shape[0],
                          probe.spec_.ordering_action_dim)
    with tempfile.TemporaryDirectory() as d:
        save = os.path.join(d, "m2.pt")
        save_policies(prod, order, 5.0, save, config=cfg, orderer_obs_version=3)
        rows = eval_m2(save, depth=depth, seeds=[seed], eval_seeds=eval_seeds,
                       horizon=100, product_stream=True, log=lambda *a, **k: None)

    row = rows[0]
    pp, oo = ENVELOPE_COMBOS[row["env_name"]]
    direct = evaluate(PRODUCING_BASELINES[pp], ORDERING_BASELINES[oo], seeds=range(eval_seeds),
                      horizon=100, config=cfg, product_stream=True)
    assert row["env_cu"] == round(direct["consumer_completed_units"][0], 1)
