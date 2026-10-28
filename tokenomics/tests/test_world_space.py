"""World-space generator (add-m3-development-audit task 4.4, design D10; world-space spec;
ADR-0057): levers, records, the neutral point's byte identity, topology invariance, sizes, the
lever corners' contract, balanced categorical levels and the tightness corner's ceiling."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from tokenomics.domain.config import Configuration
from tokenomics.env.spaces import config_fingerprint
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.ceiling_bound import B_BASE, episode_demand, world_ceiling
from tokenomics.eval.contract import ContractMonitor
from tokenomics.eval.families import (build_family_config, demand_units_per_cycle,
                                      num_consumers, top_layer_throughput)
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.routing_audit import config_digest
from tokenomics.eval.routing_diagnostics import DIAGNOSTIC_ROUTERS
from tokenomics.eval.runner import Policy, run_episode
from tokenomics.eval.shock_profiles import SMOKE_START_WINDOW, load_shock_profile
from tokenomics.eval.world_space import (CHOICES, CYCLIC, DRIFT, GENERATOR_VERSION, IID, LEVERS,
                                         PATTERN_RANGES, PATTERNS, SHOCK_LEVERS, Levers,
                                         WorldPoint, build_world, config_from_record,
                                         episode_seeds, lhs_columns, measure, point_from_record,
                                         sample_levers, sample_worlds, world_record, world_seed)

ATLAS = "atlas-test"
ORDERER_CKPT = Path(__file__).resolve().parents[2] / (
    "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt")
COORDINATES = {"rho", "sigma", "cv_capacity", "cv_degradation", "cv_cost",
               "shock_events_per_producer", "shock_hit_magnitude", "shock_outage_duration",
               "demand_concentration", "demand_volatility"}
# (family, fanout) per neutral-point cell; dense is the whole layer
CELLS = (("sparse_wide", 2, "sparse_f2"), ("sparse_wide", 4, "sparse_f4"),
         ("dense", None, "dense"))


def _episode_stats(cfg: Configuration, order: Policy | None = None, horizon: int = 12,
                   seed: int = 3) -> dict:
    """One entity_planner episode (round_robin unless `order`) with the contract monitor on;
    raises on any contract violation."""
    env = joint_stack_env(cfg, horizon)
    monitor = ContractMonitor(env)
    stats = run_episode(env, PRODUCING_BASELINES["entity_planner"],
                        order or DIAGNOSTIC_ROUTERS["round_robin"], seed=seed, monitor=monitor)
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=horizon)
    return env.episode_stats


def _family(seed: int, size: int, family: str, fanout: int | None, regime: str,
            profile: str | None = None) -> Configuration:
    return build_family_config(seed, family=family, producers=size, regime=regime,
                               fanout=fanout, shock_profile=profile)


# --- "One lever per axis", "World record", "Same record, same world" ---


def test_unset_levers_are_neutral_and_recorded() -> None:
    """Scenario "One lever per axis": a world requested with only k_c set builds and records
    every other lever at its neutral value."""
    cfg = build_world(11, producers=12, levers={"k_c": 2.0})
    assert cfg == build_world(11, producers=12, levers=Levers(k_c=2.0))
    rec = world_record(WorldPoint(ATLAS, 12, 0, 11, Levers(k_c=2.0)))
    neutral = Levers().to_dict()
    assert rec["levers"] == {**neutral, "k_c": 2.0}
    assert {f.name for f in fields(Levers)} == set(rec["levers"])
    for name, rng in LEVERS.items():
        assert getattr(Levers(), name) == rng.neutral
    assert build_world(11, producers=12) == build_world(11, producers=12, levers=Levers())


def test_world_record_holds_every_field() -> None:
    """Scenario "World record": levers, seed, generator version, every measured coordinate,
    declarers per product and the config hash."""
    for point in sample_worlds(ATLAS, 12, 16)[:16:5] + sample_worlds(ATLAS, 42, 8)[:2]:
        rec = world_record(point)
        cfg = point.config()
        assert rec["levers"] == point.levers.to_dict()
        assert rec["seed"] == point.seed == world_seed(ATLAS, point.size, point.index)
        assert rec["generator_version"] == GENERATOR_VERSION
        assert set(rec["coordinates"]) == COORDINATES
        assert all(np.isfinite(v) and v >= 0 for v in rec["coordinates"].values())
        assert rec["config_sha256"] == config_digest(cfg)
        declared = Counter(x for p in cfg.producer_configs for x in p.produces)
        assert rec["declarers_per_product"] == {str(k): v for k, v in sorted(declared.items())}
        assert len(rec["declarers_per_product"]) == len(cfg.process_sheets)


def test_rebuilt_record_has_the_recorded_hash() -> None:
    """Scenario "Same record, same world": a record round-tripped through JSON rebuilds the
    same configuration; a tampered record is refused."""
    points = sample_worlds(ATLAS, 12, 64)[::9] + sample_worlds(ATLAS, 42, 16)[::7]
    points.append(WorldPoint(ATLAS, 12, 99, 5, Levers(shocks=True), SMOKE_START_WINDOW))
    assert {p.levers.pattern for p in points} >= {CYCLIC, IID}
    for point in points:
        rec = json.loads(json.dumps(world_record(point)))
        assert point_from_record(rec) == point
        assert config_digest(config_from_record(rec)) == rec["config_sha256"]
    tampered = {**rec, "levers": {**rec["levers"], "m_q": 2.0}}
    with pytest.raises(ValueError, match="sha256"):
        config_from_record(tampered)
    with pytest.raises(ValueError, match="generator"):
        point_from_record({**rec, "generator_version": "world-space-v0"})


# --- "Neutral point" ---


@pytest.mark.parametrize("size", [12, 42])
@pytest.mark.parametrize("family,fanout,choice", CELLS)
def test_neutral_point_reproduces_family_worlds(size: int, family: str, fanout: int | None,
                                                choice: str) -> None:
    """Scenario "Neutral point": with every lever neutral the config, its fingerprint and a
    horizon-12 episode's statistics equal `build_family_config`'s, calm (cyclic) and shocked
    (shocks-i with drift; and without a profile, the iid pattern), on sample seeds."""
    cases = (
        (Levers(choice=choice), "calm", None),
        (Levers(choice=choice, shocks=True, pattern=DRIFT), "shocked", "shocks-i"),
        (Levers(choice=choice, pattern=IID), "shocked", None),
    )
    for levers, regime, profile in cases:
        for seed in (1040, 77, 123457):
            ours = build_world(seed, producers=size, levers=levers)
            theirs = _family(seed, size, family, fanout, regime, profile)
            assert ours == theirs
            assert config_digest(ours) == config_digest(theirs)
            assert config_fingerprint(ours) == config_fingerprint(theirs)
        assert _episode_stats(ours) == _episode_stats(theirs)


def test_neutral_point_matches_registry_v2_worlds() -> None:
    """Scenario "Neutral point" on the gate registry: the config hash of 2 registry v2
    development worlds (12 calm, 42 shocked with shocks-i) is reproduced."""
    reg = load_registry(REGISTRY_V2)
    entries = {e.id: e for e in reg.entries}
    for wid, levers in (("dev-sparse_wide-12-calm-00", Levers()),
                        ("dev-sparse_wide-42-shocked-00", Levers(shocks=True, pattern=DRIFT))):
        e = entries[wid]
        ours = build_world(e.world_seed, producers=e.size, levers=levers)
        assert config_digest(ours) == config_digest(e.config())
        assert config_fingerprint(ours) == config_fingerprint(e.config())


EXP0050_SPEC = Path(__file__).resolve().parents[2] / (
    "experiments/EXP-0050-routing-vs-fanout/spec.json")
# fan-sparse_wide-12-f2-calm-00, archived in EXP-0050's spec.json (independent of this tree)
ARCHIVED_DIGEST = (1194150781, "48a4ac7412efaa8cb6688a2632c41fe91165a002c07f285dd324494312050d44")


def test_neutral_point_reproduces_archived_config_hashes() -> None:
    """Scenario "Neutral point" against archived values: every world pinned in EXP-0050's
    spec.json (12/42 x every choice x calm / shocked with shocks-i) built neutral by
    `build_world` has its archived config sha256, so record hashes survive later optional
    config fields (`routing_audit.DIGEST_OMIT_WHEN_UNSET`)."""
    seed, digest = ARCHIVED_DIGEST
    assert config_digest(build_world(seed, producers=12)) == digest
    pins = json.loads(EXP0050_SPEC.read_text())["pins"]["worlds"]
    assert len(pins) == 64
    for w in pins:
        fanout = w["generator"]["params"]["fanout"]
        choice = "dense" if w["family"] == "dense" else f"sparse_f{fanout}"
        levers = (Levers(choice=choice) if w["regime"] == "calm"
                  else Levers(choice=choice, shocks=True, pattern=DRIFT))
        cfg = build_world(w["world_seed"], producers=w["size"], levers=levers)
        assert config_digest(cfg) == w["config_sha256"], w["id"]


def test_new_optional_config_fields_leave_the_digest() -> None:
    """Every optional (None-default) Configuration field beyond the pre-pin `shocks` and
    `demand_drift` is dropped from `config_digest` when unset; a field added without joining
    `DIGEST_OMIT_WHEN_UNSET` would move every recorded config hash."""
    from tokenomics.eval.routing_audit import DIGEST_OMIT_WHEN_UNSET

    optional = {f.name for f in fields(Configuration) if f.default is None}
    assert optional - {"shocks", "demand_drift"} == set(DIGEST_OMIT_WHEN_UNSET)


def test_neutral_shock_profile_is_shocks_i_without_its_drift() -> None:
    """The code-built profile at (r, m, d) = (1, 0.45, 3) has the file's shocks-i kinds; the
    profile's drift is never applied with it (shocks and demand vary independently)."""
    from tokenomics.eval.families import shock_profile_from_levers

    assert shock_profile_from_levers().kinds == load_shock_profile("shocks-i").kinds
    iid = build_world(5, producers=12, levers=Levers(shocks=True, pattern=IID))
    assert iid.shocks is not None and iid.demand_drift is None and iid.demand_pattern is None
    calm = build_world(5, producers=12, levers=Levers(shocks=True))
    assert calm.shocks == iid.shocks and all(c.schedule for c in calm.consumers)
    assert calm.process_sheets == build_world(5, producers=12).process_sheets


def test_neutral_rho_is_the_world_demand_ratio() -> None:
    """The neutral point's measured rho equals the world-level
    `demand_units_per_cycle / top_layer_throughput`."""
    for size in (12, 42):
        for levers, regime in ((Levers(), "calm"), (Levers(pattern=IID), "shocked")):
            for seed in (1040, 9):
                cfg = _family(seed, size, "sparse_wide", 2, regime)
                rec = world_record(WorldPoint(ATLAS, size, 0, seed, levers))
                assert rec["coordinates"]["rho"] == (demand_units_per_cycle(cfg)
                                                     / top_layer_throughput(cfg))
                assert rec["config_sha256"] == config_digest(cfg)


# --- "Heterogeneity changed" and topology invariance ---


def _topology(cfg: Configuration) -> tuple:
    return (cfg.process_sheets, [(p.id, p.produces) for p in cfg.producer_configs])


def test_heterogeneity_changes_values_not_topology() -> None:
    """Scenario "Heterogeneity changed": h_cap = 0 against neutral: identical recipes and
    declared products, and every producer of a layer has the same capacity; other draws
    unchanged. h_deg and h_cost = 0 likewise flatten their values only."""
    for size, choice in ((12, "sparse_f2"), (42, "sparse_f7"), (42, "dense")):
        base = build_world(31, producers=size, levers=Levers(choice=choice))
        layer = size // 3
        for lever, read in (("h_cap", lambda p: next(iter(p.capacities.values()))),
                            ("h_deg", lambda p: p.degradation),
                            ("h_cost", lambda p: (p.restoration.cost, p.upgrade.cost))):
            flat = build_world(31, producers=size, levers=Levers(choice=choice, **{lever: 0.0}))
            assert _topology(flat) == _topology(base)
            assert flat.consumers == base.consumers
            for level in range(3):
                ps = flat.producer_configs[level * layer:(level + 1) * layer]
                assert len({read(p) for p in ps}) == 1, (lever, level)
            spread = build_world(31, producers=size, levers=Levers(choice=choice, **{lever: 2.0}))
            assert _topology(spread) == _topology(base)
            for a, b in zip(base.producer_configs, spread.producer_configs):
                assert a.restoration.restores == b.restoration.restores
                assert a.upgrade.increases == b.upgrade.increases
                assert a.inventory_limit == b.inventory_limit
                if lever != "h_cap":
                    assert a.capacities == b.capacities
        caps = [next(iter(p.capacities.values())) for p in base.producer_configs]
        spread = build_world(31, producers=size, levers=Levers(choice=choice, h_cap=2.0))
        wide = [next(iter(p.capacities.values())) for p in spread.producer_configs]
        assert np.std(wide) > np.std(caps)


def test_heterogeneity_centres_on_the_layer_median() -> None:
    """D10: at h = 0 every value of a layer is round_half_up(median of the layer's drawn
    values), not the mean; at h = 2 each is round_half_up(med + 2·(v − med)), clamped."""
    import statistics

    centred_off_mean = 0
    for seed in range(1040, 1046):
        for size in (12, 42):
            base = build_world(seed, producers=size)
            flat = build_world(seed, producers=size, levers=Levers(h_cap=0.0, h_cost=0.0))
            wide = build_world(seed, producers=size, levers=Levers(h_cap=2.0))
            layer = size // 3
            for level in range(3):
                sl = slice(level * layer, (level + 1) * layer)
                for read in (lambda p: next(iter(p.capacities.values())),
                             lambda p: p.restoration.cost, lambda p: p.upgrade.cost):
                    drawn = [read(p) for p in base.producer_configs[sl]]
                    med = statistics.median(drawn)
                    assert {read(p) for p in flat.producer_configs[sl]} == {int(med + 0.5)}
                    centred_off_mean += int(med + 0.5) != int(statistics.fmean(drawn) + 0.5)
                caps = [next(iter(p.capacities.values())) for p in base.producer_configs[sl]]
                med = statistics.median(caps)
                assert [next(iter(p.capacities.values())) for p in wide.producer_configs[sl]] \
                    == [max(1, int(med + 2 * (v - med) + 0.5)) for v in caps]
    assert centred_off_mean > 0  # the median and the mean differ somewhere, so this pins it


@pytest.mark.parametrize("lever", sorted(LEVERS))
def test_levers_do_not_move_the_topology(lever: str) -> None:
    """Requirement "Levers do not move the topology": every capacity, heterogeneity, quantity
    and shock lever at both ends keeps the recipes and declared products, and the first
    consumers' pools."""
    base = build_world(8, producers=42, levers=Levers(choice="sparse_f4", pattern=IID))
    for end in (LEVERS[lever].lo, LEVERS[lever].hi):
        cfg = build_world(8, producers=42, levers=Levers(
            choice="sparse_f4", pattern=IID, shocks=lever in SHOCK_LEVERS, **{lever: end}))
        assert _topology(cfg) == _topology(base)
        for a, b in zip(base.consumers, cfg.consumers):
            assert a.products == b.products


def test_consumer_and_quantity_multipliers() -> None:
    """k_c sets max(1, round_half_up(k_c·3P/12)) consumers; extra consumers draw on their own
    streams (a smaller k_c's consumers are a prefix of a larger one's) with the family's pool
    rule; m_q scales quantity ranges and calm schedules, rounded half up."""
    for size in (12, 42):
        for regime_levers in (Levers(), Levers(pattern=IID)):
            base = build_world(4, producers=size, levers=regime_levers)
            assert len(base.consumers) == num_consumers(size)
            big = build_world(4, producers=size, levers=replace(regime_levers, k_c=8.0))
            mid = build_world(4, producers=size, levers=replace(regime_levers, k_c=3.0))
            assert len(big.consumers) == 8 * 3 * size // 12
            assert big.consumers[:len(mid.consumers)] == mid.consumers
            assert big.consumers[:len(base.consumers)] == base.consumers
            pools = Counter(c.products for c in big.consumers)
            assert set(pools) == {c.products for c in base.consumers}
            small = build_world(4, producers=size, levers=replace(regime_levers, k_c=0.5))
            assert small.consumers == base.consumers[:max(1, int(0.5 * 3 * size / 12 + 0.5))]
            q = build_world(4, producers=size, levers=replace(regime_levers, m_q=2.5))
            for a, b in zip(base.consumers, q.consumers):
                assert (b.quantity_min, b.quantity_max) == (3, int(2.5 * a.quantity_max + 0.5))
                assert [p for p, _ in b.schedule] == [p for p, _ in a.schedule]
                assert [x for _, x in b.schedule] == [int(2.5 * x + 0.5) for _, x in a.schedule]


def test_product_patterns_widen_the_pools() -> None:
    """Design D11 "Pools": hit and switch widen each B9 consumer's pool to min(4, W)
    consecutive top products from its start; other patterns keep the family's pools."""
    for size, choice in ((12, "sparse_f2"), (42, "sparse_f2"), (42, "dense")):
        plain = build_world(6, producers=size, levers=Levers(choice=choice, pattern=IID))
        for kind in PATTERNS:
            cfg = build_world(6, producers=size, levers=Levers(choice=choice, pattern=kind))
            assert _topology(cfg) == _topology(plain)
            for a, b in zip(plain.consumers, cfg.consumers):
                if kind in ("hit", "switch"):
                    assert len(b.products) == 4 and b.products[:len(a.products)] == a.products
                elif kind != CYCLIC:
                    assert b.products == a.products


def test_larger_size_refused() -> None:
    """Scenario "Larger size requested": 120 producers (and any size but 12 or 42) is refused,
    naming the size."""
    for size in (120, 24, 9):
        with pytest.raises(ValueError, match=f"size {size}"):
            build_world(1, producers=size)
        with pytest.raises(ValueError, match=f"size {size}"):
            sample_levers(ATLAS, size, 4)


def test_out_of_range_levers_refused() -> None:
    with pytest.raises(ValueError, match="k_c"):
        build_world(1, producers=12, levers=Levers(k_c=9.0))
    with pytest.raises(ValueError, match="outage_duration"):
        build_world(1, producers=12, levers=Levers(outage_duration=2.5))
    with pytest.raises(ValueError, match="not available at 12"):
        build_world(1, producers=12, levers=Levers(choice="sparse_f7"))
    with pytest.raises(ValueError, match="takes no parameter"):
        build_world(1, producers=12, levers=Levers(pattern=IID, pattern_params=(("burst_p", 0.1),)))


# --- sampling ---


def test_balanced_categorical_levels() -> None:
    """Scenario "Balanced categorical levels": a 1,024-world sample at 12 gives each of the 7
    demand patterns 146 or 147 worlds (choice 341/342, shocks 512 each); 128 at 42 likewise."""
    for size, n in ((12, 1024), (42, 128)):
        sample = sample_levers(ATLAS, size, n)
        for axis, levels in (("pattern", PATTERNS), ("choice", CHOICES[size]),
                             ("shocks", (False, True))):
            counts = Counter(getattr(lv, axis) for lv in sample)
            assert set(counts) == set(levels)
            assert set(counts.values()) <= {n // len(levels), -(-n // len(levels))}, (axis, counts)
    counts = Counter(lv.pattern for lv in sample_levers(ATLAS, 12, 1024))
    assert sorted(counts.values()) == [146] * 5 + [147] * 2


def test_latin_hypercube_strata_and_ranges() -> None:
    """One point per stratum in every continuous column; values in range on their scale; the
    sample is a function of (atlas, size, n); pattern parameters within the D11 ranges."""
    n = 256
    cols = lhs_columns(ATLAS, 42, n)
    for name in LEVERS:
        assert sorted(np.floor(cols[name] * n).astype(int)) == list(range(n))
    sample = sample_levers(ATLAS, 42, n)
    assert sample == sample_levers(ATLAS, 42, n)
    assert sample != sample_levers("atlas-other", 42, n)
    for name, rng in LEVERS.items():
        values = [getattr(lv, name) for lv in sample]
        assert rng.lo <= min(values) and max(values) <= rng.hi
    log_k = np.log([lv.k_c for lv in sample])
    assert abs(np.median(log_k) - (np.log(0.5) + np.log(8.0)) / 2) < 0.15
    assert {lv.outage_duration for lv in sample} == set(range(2, 11))
    for lv in sample:
        params = dict(lv.pattern_params)
        for pname, lo, hi, integer in PATTERN_RANGES.get(lv.pattern, ()):
            assert lo <= params[pname] <= hi and isinstance(params[pname], int) == integer
        if lv.pattern == "seasonal":
            assert 0 <= params["phase"] < params["period"]
    points = sample_worlds(ATLAS, 12, 4)
    assert len({p.seed for p in points}) == 4
    assert set(episode_seeds(points[0].world_id, 3)).isdisjoint(
        episode_seeds(points[0].world_id, 3, stage="confirm"))


def test_measured_coordinates_follow_the_levers() -> None:
    """rho grows with k_c and m_q and falls with c; the within-layer heterogeneity CVs are 0
    at h = 0 and grow at h = 2; shock
    intensity is 0 without shocks and scales with r; a burst raises volatility; a hit
    concentrates demand."""
    def coords(**kw: object) -> dict:
        lv = Levers(**kw)
        return measure(build_world(21, producers=12, levers=lv), 21)

    base = coords(pattern=IID)
    assert coords(pattern=IID, k_c=4.0)["rho"] > base["rho"]
    assert coords(pattern=IID, m_q=4.0)["sigma"] > base["sigma"]
    assert coords(pattern=IID, capacity_scale=0.25)["rho"] > base["rho"]
    flat = coords(pattern=IID, h_cap=0.0, h_deg=0.0, h_cost=0.0)
    for cv, lever in (("cv_capacity", "h_cap"), ("cv_degradation", "h_deg"),
                      ("cv_cost", "h_cost")):
        assert flat[cv] == 0.0 < base[cv]
        assert coords(pattern=IID, **{lever: 2.0})[cv] > base[cv]
    assert base["shock_events_per_producer"] == 0.0
    on = coords(pattern=IID, shocks=True)
    assert on["shock_events_per_producer"] == 3 / 12
    assert on["shock_hit_magnitude"] == pytest.approx(0.45)
    assert on["shock_outage_duration"] == 4.5
    assert coords(pattern=IID, shocks=True, shock_rate=3.0)["shock_events_per_producer"] == 9 / 12
    burst = coords(pattern="burst", pattern_params=(("burst_p", 0.1),))
    assert burst["demand_volatility"] > base["demand_volatility"]
    sharp = coords(pattern="hit", pattern_params=(("hit_alpha", 2.0), ("hit_period", 30)))
    soft = coords(pattern="hit", pattern_params=(("hit_alpha", 0.8), ("hit_period", 30)))
    assert sharp["demand_concentration"] > soft["demand_concentration"]


# --- "Lever corner" ---


def _corners() -> list[tuple[str, Levers]]:
    out = [("neutral", Levers())]
    for name, rng in LEVERS.items():
        out += [(f"{name}={end}", Levers(shocks=name in SHOCK_LEVERS, **{name: end}))
                for end in (rng.lo, rng.hi)]
    lo = {n: r.lo for n, r in LEVERS.items()}
    hi = {n: r.hi for n, r in LEVERS.items()}
    out.append(("all-low", Levers(shocks=True, pattern="burst",
                                  pattern_params=(("burst_p", 0.02),), **lo)))
    out.append(("all-high", Levers(shocks=True, pattern="burst",
                                   pattern_params=(("burst_p", 0.10),), **hi)))
    for kind, ranges in PATTERN_RANGES.items():
        for end in ("low", "high"):  # every parameter of the kind at that end
            params = {p: lo_ if end == "low" else hi_ for p, lo_, hi_, _ in ranges}
            out.append((f"{kind}-{end}",
                        Levers(pattern=kind, pattern_params=tuple(sorted(params.items())))))
    return out


def _stacks() -> list[tuple[str, Policy]]:
    stacks = [("round_robin", DIAGNOSTIC_ROUTERS["round_robin"])]
    if ORDERER_CKPT.exists() and ORDERER_CKPT.read_bytes()[:7] != b"version":  # git-lfs pointer
        pytest.importorskip("torch")
        from tokenomics.eval.routing_audit import load_trained_policy

        stacks.append(("trained_orderer", load_trained_policy("orderer", str(ORDERER_CKPT))))
    return stacks


def test_lever_corners_keep_the_contract() -> None:
    """Scenario "Lever corner": each lever at both ends with the others neutral, the all-low and
    all-high corners and every pattern kind at its parameter ends, horizon 12, shocks inside the
    horizon (smoke window), `entity_planner/round_robin` and `entity_planner/trained_orderer`
    (when the git-lfs checkpoint is present): no contract failure at 12 producers; at 42
    (dense, the widest choice) the all-low and all-high corners."""
    stacks = _stacks()
    corners = _corners()
    assert len(corners) == 1 + 2 * len(LEVERS) + 2 + 2 * len(PATTERN_RANGES)
    played = [(12, "sparse_f2", lv) for _, lv in corners]
    played += [(42, "dense", lv) for name, lv in corners if name.startswith("all-")]
    for size, choice, levers in played:
        cfg = build_world(7, producers=size, levers=replace(levers, choice=choice),
                          shock_window=SMOKE_START_WINDOW)
        for _, order in stacks:
            stats = _episode_stats(cfg, order)  # raises on any contract violation
            assert stats["cycles"] >= 1


def test_tight_corner_ceiling_sees_capacity() -> None:
    """At the high-tightness lever corner (k_c = 8, m_q = 8, c = 0.25) B_base is below the
    scheduled demand at horizon 100 (the base-capacity ceiling sees capacity), while B2 stays
    at the scheduled units (a demand cap)."""
    for size, levers in ((12, Levers(k_c=8.0, m_q=8.0, capacity_scale=0.25)),
                         (42, Levers(k_c=8.0, m_q=8.0, capacity_scale=0.25, pattern=IID))):
        cfg = build_world(1040, producers=size, levers=levers)
        demand = episode_demand(cfg, 100, 7)
        bounds = world_ceiling(cfg, 100, demand)
        scheduled = sum(demand.values())
        assert bounds[B_BASE].units < scheduled, (size, bounds, scheduled)
        assert bounds[B_BASE].binding_constraint.startswith("capacity")
        assert bounds["B2"].units == scheduled
