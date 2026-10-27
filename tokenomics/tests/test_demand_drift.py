"""Smooth demand drift (ADR-0055; add-shocks-i-drift design D4; stochastic-environment
"Smooth demand drift", "Seeded event schedule").

Between consecutive cycles at most floor(j*N) of the N B9-randomized consumers change their
order (product or quantity). Drift draws on its own `drift` stream, so B9 sampling without
drift and every shock schedule are untouched.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tokenomics.domain.config import Configuration, ConsumerConfig, DemandDrift, load_config
from tokenomics.domain.shocks import ShockGenerator, ShockKind, ShockModel
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.demand_drift import draw_schedule_entry, drift_budget
from tokenomics.env.future_fork import fork_future_demand
from tokenomics.env.spaces import config_fingerprint
from tokenomics.eval.scale import config_to_dict

PHASE1 = Path(__file__).resolve().parents[1] / "tokenomics" / "config" / "phase1.json"
POOL = (10, 20, 30)  # phase1 products with a process sheet


def _config(n: int, j: float | None, *, pool: tuple[int, ...] = POOL, qmax: int = 2,
            fixed: int = 0) -> Configuration:
    """phase1 economy with `n` B9-randomized consumers (+ `fixed` fixed-product ones)."""
    base = load_config(PHASE1)
    consumers = [ConsumerConfig(id=f"c{i}", product=pool[0], products=pool, quantity_min=1,
                                quantity_max=qmax) for i in range(n)]
    consumers += [ConsumerConfig(id=f"f{i}", product=30) for i in range(fixed)]
    cfg = replace(base, consumers=consumers,
                  demand_drift=None if j is None else DemandDrift(j))
    cfg.validate()
    return cfg


def _schedules(cfg: Configuration, seed: int, horizon: int) -> dict[str, tuple]:
    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=seed)
    return dict(env._demand_schedules)


def _changes(schedules: dict[str, tuple], ids: list[str]) -> list[int]:
    """Changed orders per transition among `ids`."""
    horizon = len(schedules[ids[0]])
    return [sum(schedules[c][t] != schedules[c][t - 1] for c in ids) for t in range(1, horizon)]


# --- Smooth demand drift ---


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_drift_bound_respected(seed: int) -> None:
    """Scenario "Drift bound respected": N = 10, j = 0.2 -> at most 2 changes per transition."""
    cfg = _config(10, 0.2)
    s = _schedules(cfg, seed, 200)
    changes = _changes(s, [f"c{i}" for i in range(10)])
    assert len(changes) == 199
    assert max(changes) <= 2
    assert sum(changes) > 0.8 * 2 * 199  # nearly every transition spends its budget


def test_product_switch_is_a_whole_order_change() -> None:
    """Scenario "Product switch is a whole-order change": with quantity fixed at 1 every
    change is a product switch, and each counts as one change against the budget
    (N = 5, j = 0.2 -> k = 1)."""
    cfg = _config(5, 0.2, pool=(10, 30), qmax=1)
    for seed in range(5):
        s = _schedules(cfg, seed, 100)
        ids = [f"c{i}" for i in range(5)]
        for t in range(1, 100):
            switched = [c for c in ids if s[c][t][0] != s[c][t - 1][0]]
            assert len(switched) <= 1
            assert all(s[c][t][1] == 1 for c in ids)
        assert max(_changes(s, ids)) == 1


def test_budget_rounds_down_to_zero() -> None:
    """Scenario "Budget rounds down to zero": N = 3, j = 0.2 -> constant demand."""
    assert drift_budget(0.2, 3) == 0
    cfg = _config(3, 0.2)
    for seed in range(5):
        s = _schedules(cfg, seed, 50)
        for c in ("c0", "c1", "c2"):
            assert len(set(s[c])) == 1


def test_budget_at_the_gate_sizes() -> None:
    """j = 0.34 (the shocks-i profile) gives k = 1 of 3 at 12 producers, 3 of 11 at 42."""
    assert drift_budget(0.34, 3) == 1
    assert drift_budget(0.34, 11) == 3
    assert drift_budget(0.29, 100) == 29  # exact: a float product would floor to 28
    for n, k in ((3, 1), (11, 3)):
        s = _schedules(_config(n, 0.34), 4, 120)
        changes = _changes(s, [f"c{i}" for i in range(n)])
        assert max(changes) == k


def test_full_budget_and_zero_j() -> None:
    s = _schedules(_config(3, 1.0), 0, 30)
    assert set(_changes(s, ["c0", "c1", "c2"])) == {3}  # every order changes every cycle
    s = _schedules(_config(4, 0.0), 0, 30)
    assert set(_changes(s, ["c0", "c1", "c2", "c3"])) == {0}


def test_fixed_consumers_neither_drift_nor_count() -> None:
    """N counts only the B9-randomized consumers: 3 randomized + 2 fixed at j = 0.34 gives
    k = 1, and the fixed consumers keep their canonical demand."""
    cfg = _config(3, 0.34, fixed=2)
    s = _schedules(cfg, 2, 60)
    assert max(_changes(s, ["c0", "c1", "c2"])) == 1
    assert s["f0"] == s["f1"] == ((30, 1),)


def test_drift_is_prefix_stable_across_horizons() -> None:
    cfg = _config(10, 0.2)
    for seed in range(4):
        short, long = _schedules(cfg, seed, 10), _schedules(cfg, seed, 20)
        assert all(long[c][:10] == short[c] for c in short)


def test_drift_unchanged_when_shocks_toggled() -> None:
    cfg = _config(10, 0.2)
    gens = (ShockGenerator(ShockKind.CAPACITY_HIT, 3, (2, 20), (1, 1), (0.3, 0.6)),
            ShockGenerator(ShockKind.PRODUCER_OUTAGE, 2, (2, 20), (1, 3), (1.0, 1.0)))
    shocked = replace(cfg, shocks=ShockModel(generators=gens))
    for seed in range(4):
        a, b = TokenomicsAECEnv(config=cfg, horizon=30), TokenomicsAECEnv(config=shocked, horizon=30)
        a.reset(seed=seed)
        b.reset(seed=seed)
        assert a._demand_schedules == b._demand_schedules
        assert a.episode_stats["demand_digest"] == b.episode_stats["demand_digest"]
        assert b.episode_stats["shock_events"] == 5


def test_drift_is_seeded_and_feeds_the_demand_digest() -> None:
    cfg = _config(10, 0.2)
    digests = set()
    for seed in (0, 0, 1, 2):
        env = TokenomicsAECEnv(config=cfg, horizon=30)
        env.reset(seed=seed)
        digests.add(env.episode_stats["demand_digest"])
    assert len(digests) == 3


def test_b9_sampling_unchanged_without_drift() -> None:
    """Without drift every randomized consumer keeps its legacy schedule: horizon draws per
    consumer, consumer-major, on `default_rng(seed)` (the goldens pin the numbers)."""
    cfg = _config(4, None)
    for seed in (0, 3):
        s = _schedules(cfg, seed, 25)
        rng = np.random.default_rng(seed)
        for c in cfg.consumers:
            want = tuple(draw_schedule_entry(rng, c.product_pool, c.quantity_min, c.quantity_max)
                         for _ in range(25))
            assert s[c.id] == want


def test_stationary_marginal_is_close_to_uniform() -> None:
    """Redraw-from-the-others keeps the B9 marginal (uniform over pool x quantities)."""
    cfg = _config(10, 0.2)
    counts: Counter = Counter()
    for seed in range(5):
        for sched in _schedules(cfg, seed, 400).values():
            counts.update(sched[50:])
    total = sum(counts.values())
    assert len(counts) == 6
    assert all(abs(v / total - 1 / 6) < 0.02 for v in counts.values())


# --- config and seeding ---


@pytest.mark.parametrize("j", [-0.1, 1.1, float("nan"), True, "0.3"])
def test_invalid_j_rejected(j: object) -> None:
    with pytest.raises(ValueError, match=r"demand drift j must be a number in \[0, 1\]"):
        replace(_config(3, None), demand_drift=DemandDrift(j)).validate()  # type: ignore[arg-type]


def test_json_round_trip_and_fingerprint(tmp_path: Path) -> None:
    cfg = _config(3, 0.34)
    d = config_to_dict(cfg)
    assert d["demandDrift"] == {"j": 0.34}
    assert "demandDrift" not in config_to_dict(_config(3, None))
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(d))
    loaded = load_config(path)
    assert loaded.demand_drift == DemandDrift(0.34) and config_to_dict(loaded) == d
    d["demandDrift"] = {"j": 1}
    path.write_text(json.dumps(d))
    assert load_config(path).demand_drift == DemandDrift(1.0)
    assert config_fingerprint(cfg) != config_fingerprint(_config(3, None))


def test_drift_needs_a_seed() -> None:
    env = TokenomicsAECEnv(config=_config(3, 0.34), horizon=10)
    with pytest.raises(ValueError, match=r"demand drift needs an integer episode seed"):
        env.reset(seed=None)
    TokenomicsAECEnv(config=_config(3, None), horizon=10).reset(seed=None)  # B9 unchanged


def test_future_fork_refuses_drift() -> None:
    env = TokenomicsAECEnv(config=_config(3, 0.34), horizon=10)
    env.reset(seed=1)
    with pytest.raises(ValueError, match=r"does not support demand drift"):
        fork_future_demand(env, task_id="t", decision_digest=0, continuation_id=0)


# --- numeric forms keep one fingerprint (review concern 162) ---


@pytest.mark.parametrize("j", [1, 1.0, 0, 0.0])
def test_integer_and_float_j_share_a_fingerprint_through_json(tmp_path: Path, j: float) -> None:
    """Equal drift configs fingerprint identically whatever the numeric form of j, and keep
    that fingerprint through a real config_to_dict -> JSON -> load_config round trip."""
    as_given, as_float = _config(3, j), _config(3, float(j))
    assert as_given == as_float
    assert type(as_given.demand_drift.j) is float
    assert config_fingerprint(as_given) == config_fingerprint(as_float)
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(config_to_dict(as_given)))
    assert config_fingerprint(load_config(path)) == config_fingerprint(as_given)


def test_integer_shock_magnitudes_share_a_fingerprint_through_json(tmp_path: Path) -> None:
    """The same rule for shock magnitudes (events and generator bounds)."""
    from tokenomics.domain.shocks import ShockEvent, ShockTarget

    def shocked(m: float) -> Configuration:
        ev = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p2"), 3, 2, m)
        gen = ShockGenerator(ShockKind.PRODUCER_OUTAGE, 1, (2, 9), (1, 2), (m, m))
        cfg = replace(load_config(PHASE1), shocks=ShockModel(events=(ev,), generators=(gen,)))
        cfg.validate()
        return cfg

    a, b = shocked(1), shocked(1.0)
    assert a == b and config_fingerprint(a) == config_fingerprint(b)
    path = tmp_path / "shocked.json"
    path.write_text(json.dumps(config_to_dict(a)))
    assert config_fingerprint(load_config(path)) == config_fingerprint(a)


def test_bool_j_still_rejected_after_normalization() -> None:
    assert DemandDrift(True).j is True  # bool is not normalized ...
    with pytest.raises(ValueError, match=r"demand drift j must be a number"):
        replace(_config(3, None), demand_drift=DemandDrift(True)).validate()  # ... and rejected


# --- unseeded drift with N = 0 (review concern 163) ---


def test_drift_needs_a_seed_even_with_no_randomized_consumer() -> None:
    cfg = _config(0, 0.34, fixed=3)
    assert cfg.demand_drift is not None and not any(c.is_randomized for c in cfg.consumers)
    with pytest.raises(ValueError, match=r"demand drift needs an integer episode seed"):
        TokenomicsAECEnv(config=cfg, horizon=5).reset(seed=None)
    TokenomicsAECEnv(config=cfg, horizon=5).reset(seed=0)  # seeded: fine, nothing drifts
    TokenomicsAECEnv(config=_config(0, None, fixed=3), horizon=5).reset(seed=None)


# --- bounded redraw (review concern 164) ---


def _brute_redraw(rng: np.random.Generator, c: ConsumerConfig, current: tuple) -> tuple:
    """Reference: enumerate the domain (position-major, every pool position) minus current."""
    options = [(p, q) for p in c.product_pool for q in range(c.quantity_min, c.quantity_max + 1)
               if (p, q) != current]
    return options[int(rng.integers(len(options)))] if options else current


@pytest.mark.parametrize("pool, qmin, qmax", [
    ((10, 20, 30), 1, 2), ((10, 10, 30), 1, 3), ((30, 10, 30, 10), 2, 2), ((10, 30), 1, 1),
    ((20,), 1, 4), ((10, 10), 1, 1),
])
def test_rank_redraw_equals_enumeration(pool: tuple[int, ...], qmin: int, qmax: int) -> None:
    """Same entries and same RNG stream as choosing uniformly among the enumerated
    alternatives, duplicates in the pool included (every occurrence of current excluded)."""
    from tokenomics.env.demand_drift import redraw

    c = ConsumerConfig(id="c", product=pool[0], products=pool, quantity_min=qmin, quantity_max=qmax)
    domain = [(p, q) for p in pool for q in range(qmin, qmax + 1)]
    for seed in range(30):
        for current in set(domain):
            a, b = np.random.default_rng(seed), np.random.default_rng(seed)
            got, want = redraw(a, c, current), _brute_redraw(b, c, current)
            assert got == want
            assert a.integers(1 << 30) == b.integers(1 << 30)  # same draws consumed
            if any(e != current for e in domain):
                assert got != current


def test_huge_quantity_range_redraw_is_bounded() -> None:
    """A valid config with a 1e9-wide quantity range drifts without enumerating its domain."""
    import tracemalloc

    cfg = _config(2, 1.0, qmax=1_000_000_000)
    tracemalloc.start()
    s = _schedules(cfg, 3, 40)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 20_000_000  # well under one materialized domain
    for cid in ("c0", "c1"):
        sched = s[cid]
        assert all(p in POOL and 1 <= q <= 1_000_000_000 for p, q in sched)
        assert all(sched[t] != sched[t - 1] for t in range(1, 40))  # j = 1: every order changes


# --- independent drift-stream reconstruction (review concern 165) ---


def _independent_drift(consumers: list[ConsumerConfig], j_num: int, j_den: int, seed: int,
                       horizon: int) -> dict[str, list[tuple]]:
    """D4 rebuilt from hashlib + numpy only (no production stream or drift code): stream
    sha256("tokenomics|drift|<seed>"), cycle 1 = per consumer in config order a uniform pool
    position then a uniform quantity, then k = floor(j*N) consumers without replacement each
    redrawing uniformly among the enumerated alternatives."""
    import hashlib

    digest = hashlib.sha256(f"tokenomics|drift|{seed}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    n = len(consumers)
    k = (j_num * n) // j_den
    cur = [(c.product_pool[int(rng.integers(len(c.product_pool)))],
            int(rng.integers(c.quantity_min, c.quantity_max + 1))) for c in consumers]
    out = [[e] for e in cur]
    for _ in range(1, horizon):
        if k:
            for i in sorted(int(x) for x in rng.choice(n, size=k, replace=False)):
                cur[i] = _brute_redraw(rng, consumers[i], cur[i])
        for i, e in enumerate(cur):
            out[i].append(e)
    return {c.id: o for c, o in zip(consumers, out)}


def _heterogeneous() -> Configuration:
    base = load_config(PHASE1)
    consumers = [
        ConsumerConfig(id="a", product=10, products=(10, 20, 30), quantity_min=1, quantity_max=2),
        ConsumerConfig(id="b", product=30, products=(30,), quantity_min=2, quantity_max=5),
        ConsumerConfig(id="c", product=10, products=(10, 10, 20), quantity_min=1, quantity_max=1),
        ConsumerConfig(id="d", product=20, products=(20, 30), quantity_min=3, quantity_max=4),
        ConsumerConfig(id="fixed", product=30),
    ]
    cfg = replace(base, consumers=consumers, demand_drift=DemandDrift(0.5))
    cfg.validate()
    return cfg


@pytest.mark.parametrize("seed", [0, 5, 123])
def test_drift_matches_an_independent_reconstruction(seed: int) -> None:
    cfg = _heterogeneous()
    drifting = [c for c in cfg.consumers if c.is_randomized]
    assert [c.id for c in drifting] == ["a", "b", "c", "d"]
    want = _independent_drift(drifting, 1, 2, seed, 30)  # j = 0.5 -> k = 2 of 4
    got = _schedules(cfg, seed, 30)
    for cid, sched in want.items():
        assert list(got[cid]) == sched
    assert got["fixed"] == ((30, 1),)


def test_drift_seed0_literal_cycles() -> None:
    """Literal pins of the first cycles (seed 0), independent of any helper."""
    got = _schedules(_heterogeneous(), 0, 4)
    assert {cid: list(s) for cid, s in got.items() if cid != "fixed"} == PINNED_DRIFT_SEED0


PINNED_DRIFT_SEED0 = {
    "a": [(10, 1), (20, 2), (20, 2), (20, 2)],
    "b": [(30, 4), (30, 4), (30, 4), (30, 5)],
    "c": [(20, 1), (20, 1), (10, 1), (10, 1)],
    "d": [(30, 4), (30, 3), (20, 4), (30, 4)],
}
