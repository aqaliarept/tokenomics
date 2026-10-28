"""Demand patterns in the env layer (add-m3-development-audit task 4.3, design D11;
stochastic-environment delta: "Demand patterns", "Patterns replace drift on randomized consumers
only", "Patterned orders keep their tokens", "Demand patterns draw on their own streams",
"Demand patterns are unannounced", "Demand patterns are paired and digested", "No pattern, no
change")."""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.domain.config import (PATTERN_KINDS, ConsumerConfig, Configuration, DemandDrift,
                                      DemandPattern, load_config)
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.demand_digest import demand_digest
from tokenomics.env.demand_patterns import pattern_schedules, seasonal_factor
from tokenomics.env.spaces import _canonical_economy, config_fingerprint
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.contract import ContractMonitor
from tokenomics.eval.families import build_family_config
from tokenomics.env.shock_schedule import shock_digest
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.registry import REGISTRY_V1, REGISTRY_V2, load_registry
from tokenomics.eval.resilience import resilience_rows, shock_free_twin
from tokenomics.eval.routing_diagnostics import DIAGNOSTIC_ROUTERS
from tokenomics.eval.runner import run_episode
from tokenomics.eval.scale import config_to_dict

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "tokenomics" / "config" / "phase1.json"
SHOCKED_12 = "dev-sparse_wide-12-shocked-00"

PATTERNS = {
    "burst": DemandPattern("burst", burst_p=0.3),
    "hit": DemandPattern("hit", hit_alpha=2.0, hit_period=10),
    "seasonal": DemandPattern("seasonal", amplitude=0.6, period=12, phase=3.0),
    "switch": DemandPattern("switch", switch_count=2, switch_window_min=5,
                            switch_window_max=18, switch_min_gap=4, low_weight=0.0),
}


def _base(copies: int = 1, *, family: str = "sparse_wide",
          quantity: tuple[int, int] | None = None) -> Configuration:
    """A 12-producer family world with B9-randomized consumers (shocked regime without a shock
    profile: no shocks, no drift), its consumers copied `copies` times, optionally with a fixed
    quantity range."""
    cfg = build_family_config(0, family=family, producers=12, regime="shocked")
    consumers = []
    for k in range(copies):
        for c in cfg.consumers:
            lo, hi = quantity or (c.quantity_min, c.quantity_max)
            consumers.append(replace(c, id=f"{c.id}_{k}", quantity_min=lo, quantity_max=hi))
    out = replace(cfg, consumers=consumers)
    out.validate()
    return out


def _with(cfg: Configuration, pattern: DemandPattern | None) -> Configuration:
    out = replace(cfg, demand_pattern=pattern)
    out.validate()
    return out


def _schedules(cfg: Configuration, seed: int, horizon: int = 20) -> dict[str, tuple]:
    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=seed)
    return dict(env._demand_schedules)


# --- the record ---


def test_pattern_record_validates() -> None:
    for kind in PATTERN_KINDS:
        DemandPattern(kind).validate()
    bad = [DemandPattern("surge"), DemandPattern("burst", burst_p=1.5),
           DemandPattern("burst", burst_factor_min=6, burst_factor_max=5),
           DemandPattern("hit", hit_alpha=0.0), DemandPattern("seasonal", period=0),
           DemandPattern("seasonal", amplitude=float("nan")),
           DemandPattern("switch", switch_window_min=1),
           DemandPattern("switch", switch_count=5, switch_window_min=11, switch_window_max=20,
                         switch_min_gap=5)]
    for p in bad:
        with pytest.raises(ValueError):
            p.validate()
    with pytest.raises(ValueError, match="at most one demand process"):
        replace(_base(), demand_pattern=DemandPattern("burst"), demand_drift=DemandDrift(0.3)).validate()


def test_pattern_round_trips_through_the_config_json(tmp_path: Path) -> None:
    cfg = _with(_base(), PATTERNS["switch"])
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config_to_dict(cfg)))
    loaded = load_config(path)
    assert loaded.demand_pattern == cfg.demand_pattern
    assert config_fingerprint(loaded) == config_fingerprint(cfg)


# --- "Demand patterns" ---


def test_burst_factor() -> None:
    """Scenario "Burst factor": every order of the p = 0.3 burst world is its unpatterned
    quantity (the p = 0 world on the same stream) or that times an integer in [5, 10]; some
    orders burst. Tokens: test_patterned_orders_keep_their_tokens."""
    base = _base(4)
    plain = _schedules(_with(base, DemandPattern("burst", burst_p=0.0)), seed=3)
    burst = _schedules(_with(base, PATTERNS["burst"]), seed=3)
    bursts = 0
    for cid, entries in burst.items():
        for (p, q), (p0, q0) in zip(entries, plain[cid]):
            assert p == p0
            if q != q0:
                assert q % q0 == 0 and 5 <= q // q0 <= 10
                bursts += 1
    total = sum(len(e) for e in burst.values())
    assert 0.15 * total < bursts < 0.45 * total


def _modal_product(schedules: dict[str, tuple], cycles: range) -> int:
    counts = Counter(entries[t - 1][0] for entries in schedules.values() for t in cycles)
    return counts.most_common(1)[0][0]


def test_hit_moves() -> None:
    """Scenario "Hit moves": with period 10 the most popular product of cycles 1-10 differs from
    that of cycles 11-20 (dense world, every consumer may order every top product)."""
    sched = _schedules(_with(_base(30, family="dense"), PATTERNS["hit"]), seed=1)
    assert _modal_product(sched, range(1, 11)) != _modal_product(sched, range(11, 21))


def test_seasonal_envelope() -> None:
    """Scenario "Seasonal envelope": two episode seeds, fixed quantity 4: every order is 4 * s(t)
    rounded down or up (never below quantity_min), so both seeds follow the same envelope and
    differ only by the seeded rounding."""
    cfg = _with(_base(5, quantity=(4, 4)), PATTERNS["seasonal"])
    a, b = _schedules(cfg, seed=1), _schedules(cfg, seed=2)
    assert a != b
    for sched in (a, b):
        for entries in sched.values():
            for t, (_, q) in enumerate(entries, start=1):
                x = 4 * seasonal_factor(cfg.demand_pattern, t)
                assert q in {max(4, math.floor(x)), max(4, math.ceil(x))}
    peak = max(range(1, 21), key=lambda t: seasonal_factor(cfg.demand_pattern, t))
    assert all(e[peak - 1][1] > 4 for e in a.values())


def test_regime_switch() -> None:
    """Scenario "Regime switch": one switch at cycle 40 (window [40, 40]); with the unfavoured
    weight 0 the products ordered before cycle 40 and from cycle 40 on are disjoint halves."""
    cfg = _with(_base(30, family="dense"),
                DemandPattern("switch", switch_count=1, switch_window_min=40,
                              switch_window_max=40, low_weight=0.0))
    sched = _schedules(cfg, seed=4, horizon=60)
    before = {e[t][0] for e in sched.values() for t in range(39)}
    after = {e[t][0] for e in sched.values() for t in range(39, 60)}
    assert before and after and not before & after
    assert len(before | after) == 4


# --- "Patterns replace drift on randomized consumers only" ---


def test_cyclic_consumer_untouched() -> None:
    """Scenario "Cyclic consumer untouched": a consumer with an explicit schedule keeps it in a
    burst world, whatever the seed."""
    base = _base()
    cyclic = ConsumerConfig(id="cyc", product=base.consumers[0].product,
                            schedule=((base.consumers[0].product, 2), (base.consumers[0].product, 1)))
    cfg = _with(replace(base, consumers=[*base.consumers, cyclic]), PATTERNS["burst"])
    for seed in (0, 5):
        assert _schedules(cfg, seed)["cyc"] == cyclic.schedule


# --- "Patterned orders keep their tokens" ---


def _requests(cfg: Configuration, seed: int, horizon: int = 12) -> dict[tuple[str, int], tuple]:
    """(consumer, cycle) -> (product, quantity, tokens) of every consumer request of one
    entity_planner / round_robin episode."""
    env = joint_stack_env(cfg, horizon)
    seen: dict[tuple[str, int], tuple] = {}
    original_reset = env.reset

    def reset(*args: object, **kwargs: object) -> object:
        out = original_reset(*args, **kwargs)
        for cid, consumer in env.system.consumers.items():
            order = consumer.order

            def recorded(_order=order, _cid=cid):  # noqa: ANN001, ANN202
                reqs = _order()
                for r in reqs:
                    seen[(_cid, env.system.cycle_counter)] = (r.product, r.quantity, r.tokens)
                return reqs

            consumer.order = recorded  # type: ignore[method-assign]
        return out

    env.reset = reset  # type: ignore[method-assign]
    run_episode(env, PRODUCING_BASELINES["entity_planner"], DIAGNOSTIC_ROUTERS["round_robin"],
                seed=seed)
    return seen


@pytest.mark.parametrize("pattern,plain", [
    (PATTERNS["burst"], DemandPattern("burst", burst_p=0.0)),
    (PATTERNS["seasonal"], DemandPattern("seasonal", amplitude=0.0, period=12, phase=3.0)),
])
def test_patterned_orders_keep_their_tokens(pattern: DemandPattern, plain: DemandPattern) -> None:
    """Scenario "Seasonal peak" (and the burst token clause): the same consumer and cycle is
    backed by the same tokens with and without the quantity change."""
    base = _base(2)
    shaped, flat = _requests(_with(base, pattern), 2), _requests(_with(base, plain), 2)
    assert shaped.keys() == flat.keys()
    changed = [k for k in shaped if shaped[k][1] != flat[k][1]]
    assert changed
    assert all(shaped[k][2] == flat[k][2] for k in shaped)


# --- "Demand patterns draw on their own streams" ---


def test_shocks_unchanged_by_a_pattern() -> None:
    """Scenario "Shocks unchanged by a pattern": a shocked registry world (drift removed, one
    demand process) has the same shock schedule with and without a pattern."""
    world = replace(load_registry(REGISTRY_V2).entry(SHOCKED_12).config(), demand_drift=None)
    seed = load_registry(REGISTRY_V2).entry(SHOCKED_12).episode_seeds[0]
    plain = TokenomicsAECEnv(config=world, horizon=100)
    plain.reset(seed=seed)
    for pattern in PATTERNS.values():
        env = TokenomicsAECEnv(config=_with(world, pattern), horizon=100)
        env.reset(seed=seed)
        assert env._shock_schedule == plain._shock_schedule and env._shock_schedule
        assert env._demand_schedules != plain._demand_schedules


@pytest.mark.parametrize("kind", sorted(PATTERNS))
def test_prefix(kind: str) -> None:
    """Scenario "Prefix": horizons 10 and 20, same seed: cycles 1-10 identical."""
    cfg = _with(_base(3, family="dense"), PATTERNS[kind])
    short, long = _schedules(cfg, 7, horizon=10), _schedules(cfg, 7, horizon=20)
    assert all(tuple(long[cid][:10]) == tuple(short[cid]) for cid in short)


def test_pattern_never_reads_the_legacy_stream() -> None:
    """The patterned consumers' schedules equal `pattern_schedules` on the pattern's stream alone,
    and a non-randomized consumer is not passed to it."""
    cfg = _with(_base(2), PATTERNS["hit"])
    targets = [c for c in cfg.consumers if c.is_randomized]
    assert {cid: tuple(s) for cid, s in pattern_schedules(targets, cfg.demand_pattern, 9, 20).items()} \
        == {cid: tuple(s) for cid, s in _schedules(cfg, 9).items()}


# --- "Demand patterns are unannounced" ---


def test_spaces_with_and_without_a_pattern() -> None:
    """Scenario "Spaces with and without a pattern": every agent's observation and action space
    is identical, for the default and the joint-stack env."""
    base = _base(2)
    for make in (lambda c: TokenomicsAECEnv(config=c, horizon=12), lambda c: joint_stack_env(c, 12)):
        plain = make(base)
        for pattern in PATTERNS.values():
            env = make(_with(base, pattern))
            assert env.possible_agents == plain.possible_agents
            for a in env.possible_agents:
                assert env.observation_space(a) == plain.observation_space(a)
                assert env.action_space(a) == plain.action_space(a)


def test_no_observation_module_reads_the_pattern() -> None:
    """Only the env's demand build reads `demand_pattern`: no encoder or space module names it."""
    for name in ("encode.py", "spaces.py", "decode.py", "allocator.py"):
        src = (ROOT / "tokenomics" / "env" / name).read_text()
        names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
        assert "demand_pattern" not in names, name


# --- "Demand patterns are paired and digested" ---


def test_twin_keeps_the_pattern() -> None:
    """Scenario "Twin keeps the pattern": a shocked burst world and its shock-free twin (the
    gate's `shock_free_twin`) get the same bursts and equal demand digests."""
    entry = load_registry(REGISTRY_V2).entry(SHOCKED_12)
    shocked = _with(replace(entry.config(), demand_drift=None), PATTERNS["burst"])
    twin = shock_free_twin(shocked)
    assert twin.demand_pattern == shocked.demand_pattern and twin.shocks is None
    a = TokenomicsAECEnv(config=shocked, horizon=100)
    b = TokenomicsAECEnv(config=twin, horizon=100)
    a.reset(seed=entry.episode_seeds[0])
    b.reset(seed=entry.episode_seeds[0])
    assert a._demand_schedules == b._demand_schedules
    assert a.episode_stats["demand_digest"] == b.episode_stats["demand_digest"] \
        == demand_digest(a._demand_schedules)


def test_burst_is_demand_not_a_shock() -> None:
    """Scenario "Burst is demand, not a shock": resilience rows of a shocked burst episode are
    exactly the shock schedule's events; no burst appears."""
    from tokenomics.eval.shock_profiles import SHOCKS_I_SMOKE
    from tokenomics.eval.routing_audit import fanout_world

    world = fanout_world("sparse_wide", 12, 2, "shocked", 0, 1, SHOCKS_I_SMOKE)
    shocked = _with(replace(world.config(), demand_drift=None), PATTERNS["burst"])
    traces = []
    for cfg in (shocked, shock_free_twin(shocked)):
        env = joint_stack_env(cfg, 12)
        stats = run_episode(env, PRODUCING_BASELINES["entity_planner"],
                            DIAGNOSTIC_ROUTERS["round_robin"], seed=world.episode_seeds[0])
        traces.append((env, stats))
    (env, stats), (_, twin_stats) = traces
    schedule = env._shock_schedule
    assert schedule and stats["shock_digest"] == shock_digest(schedule)
    rows = resilience_rows(stats["consumer_fulfilled_units_trace"],
                           twin_stats["consumer_fulfilled_units_trace"], schedule,
                           shocked_dead=bool(stats["dead"]))
    assert [(r.kind, r.start) for r in rows] == [(e.kind.value, e.start_cycle) for e in schedule]


def test_pattern_is_in_the_fingerprint() -> None:
    base = _base()
    prints = {config_fingerprint(base)} | {config_fingerprint(_with(base, p)) for p in PATTERNS.values()}
    assert len(prints) == 1 + len(PATTERNS)


# --- "No pattern, no change" ---


def test_legacy_world() -> None:
    """Scenario "Legacy world": a configuration without a pattern leaves the field out of its
    canonical economy (so its fingerprint is what it was); phase1, every registry v1 / v2 world
    and a drift world. Episode statistics: the goldens (test_legacy_invariance)."""
    configs = [load_config(PHASE1)]
    for path in (REGISTRY_V1, REGISTRY_V2):
        configs += [e.config() for e in load_registry(path).entries]
    assert any(c.demand_drift is not None for c in configs)
    for cfg in configs:
        assert cfg.demand_pattern is None
        assert "demand_pattern" not in json.loads(_canonical_economy(cfg))


# --- contract ---


def test_burst_x10_episode_keeps_the_contract() -> None:
    """A horizon-12 episode with every order bursting by 10 (entity_planner / round_robin):
    zero contract failures."""
    cfg = _with(_base(2), DemandPattern("burst", burst_p=1.0, burst_factor_min=10,
                                        burst_factor_max=10))
    env = joint_stack_env(cfg, 12)
    monitor = ContractMonitor(env)
    stats = run_episode(env, PRODUCING_BASELINES["entity_planner"],
                        DIAGNOSTIC_ROUTERS["round_robin"], seed=0, monitor=monitor)
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=12)  # raises on any
    assert max(q for s in env._demand_schedules.values() for _, q in s) >= 20


def test_future_fork_refuses_a_patterned_world() -> None:
    """`fork_future_demand` would redraw the tail i.i.d. from B9 and lose the pattern, so it
    refuses a patterned config loudly (the pattern-aware fork is task 6.1's)."""
    from tokenomics.env.future_fork import fork_future_demand

    for pattern in PATTERNS.values():
        env = TokenomicsAECEnv(config=_with(_base(), pattern), horizon=12)
        env.reset(seed=0)
        with pytest.raises(ValueError, match="does not support demand patterns"):
            fork_future_demand(env, task_id="t", decision_digest=1, continuation_id=0)



# --- review of 4.3: the stream, the hit's timing, the scope, the switch gaps ---


def test_burst_draws_on_its_own_stream() -> None:
    """The burst schedule replays by hand from `derived_rng(seed, "pattern-burst")` (product,
    quantity, burst test, factor per order, cycle-major), not from the legacy B9 stream; and
    every kind has its own stream name."""
    from tokenomics.env.demand_patterns import pattern_stream
    from tokenomics.env.shock_schedule import derived_rng

    cfg = _with(_base(2), PATTERNS["burst"])
    got = _schedules(cfg, seed=11, horizon=6)
    rng = derived_rng(11, "pattern-burst")
    want: dict[str, list] = {c.id: [] for c in cfg.consumers}
    for _ in range(6):
        for c in cfg.consumers:
            product = c.product_pool[int(rng.integers(len(c.product_pool)))]
            quantity = int(rng.integers(c.quantity_min, c.quantity_max + 1))
            burst = rng.random() < 0.3
            factor = int(rng.integers(5, 11))
            want[c.id].append((product, quantity * factor if burst else quantity))
    assert {k: list(v) for k, v in got.items()} == want
    assert len({pattern_stream(k) for k in PATTERN_KINDS}) == len(PATTERN_KINDS)


def test_hit_moves_exactly_after_its_period() -> None:
    """With alpha 50 every order is the current hit (dense pools hold every product): the hit is
    the seeded ranking's first product for cycles 1..T and its second from cycle T + 1 on."""
    from tokenomics.env.demand_patterns import pattern_products
    from tokenomics.env.shock_schedule import derived_rng

    period = 5
    cfg = _with(_base(3, family="dense"), DemandPattern("hit", hit_alpha=50.0, hit_period=period))
    sched = _schedules(cfg, seed=2, horizon=2 * period)
    products = pattern_products(cfg.consumers)
    ranking = [products[int(i)] for i in derived_rng(2, "pattern-hit").permutation(len(products))]
    for t in range(1, 2 * period + 1):
        ordered = {e[t - 1][0] for e in sched.values()}
        assert ordered == {ranking[0] if t <= period else ranking[1]}, t


def test_non_randomized_consumers_are_never_patterned() -> None:
    """Scenario "Patterns replace drift on randomized consumers only", beyond explicit schedules:
    a fixed-product consumer (quantity 1) and a singleton-pool fixed-quantity consumer keep their
    one-entry schedules in every patterned world."""
    base = _base()
    p = base.consumers[0].product
    fixed = ConsumerConfig(id="fixed", product=p)
    single = ConsumerConfig(id="single", product=p, products=(p,), quantity_min=3, quantity_max=3)
    assert not fixed.is_randomized and not single.is_randomized
    world = replace(base, consumers=[*base.consumers, fixed, single])
    for pattern in PATTERNS.values():
        sched = _schedules(_with(world, pattern), seed=1)
        assert tuple(sched["fixed"]) == ((p, 1),)
        assert tuple(sched["single"]) == ((p, 3),)


@pytest.mark.parametrize("count,lo,hi,gap", [(2, 5, 18, 4), (3, 11, 90, 10), (4, 2, 11, 3),
                                             (3, 10, 30, 10)])
def test_switch_cycles_lie_in_the_window_with_gaps(count: int, lo: int, hi: int, gap: int) -> None:
    """Over many seeds every draw is strictly increasing, inside [lo, hi], with consecutive gaps
    >= the minimum; (4, 2, 11, 3) and (3, 10, 30, 10) are tight fits (one legal set)."""
    from tokenomics.env.demand_patterns import switch_cycles
    from tokenomics.env.shock_schedule import derived_rng

    pattern = DemandPattern("switch", switch_count=count, switch_window_min=lo,
                            switch_window_max=hi, switch_min_gap=gap)
    pattern.validate()
    seen = set()
    for seed in range(200):
        cycles = switch_cycles(derived_rng(seed, "pattern-switch"), pattern)
        assert len(cycles) == count and lo <= cycles[0] and cycles[-1] <= hi
        assert all(b - a >= gap for a, b in zip(cycles, cycles[1:]))
        seen.add(tuple(cycles))
    tight = hi - lo + 1 == count + (count - 1) * (gap - 1)
    assert (len(seen) == 1) == tight
