"""Gate world families (design D8, D9, add-measurement-gate-harness; gate-harness "Scale
families", "Constant economy ratios across sizes"; stochastic-environment "Deterministic cyclic
demand")."""

from __future__ import annotations

from collections import Counter

import pytest

from tokenomics.domain.config import Configuration
from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.families import (
    CALM,
    DENSE,
    DENSE_WIDTH,
    FAMILIES,
    SHOCKED,
    SPARSE_WIDE,
    build_dense_config,
    build_family_config,
    build_sparse_wide_config,
    cycle_emission,
    demand_throughput_ratio,
    demand_units_per_cycle,
    emission_per_producer,
    family_demand_ratio,
    num_consumers,
    top_layer_throughput,
)
from tokenomics.eval.scale import build_scale_config
from tokenomics.eval.task_archive import REPRODUCERS, TaskRecord, replay

SIZES = (12, 42)  # gate sizes (task 8.7)
BUILDABLE = (12, 42, 120)  # 120: future, non-gating, still built and checked
FAMS = (DENSE, SPARSE_WIDE)


def _layer_of(cfg: Configuration, product: int) -> int:
    return next(int(t.split("-")[1]) for t in cfg.process_sheet(product).require)


def _producer_layer(producer_id: str) -> int:
    return int(producer_id[1:].split("_")[0])


def _declarers(cfg: Configuration) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {ps.product: [] for ps in cfg.process_sheets}
    for p in cfg.producer_configs:
        for x in p.produces:
            out[x].append(p.id)
    return out


# --- Scale families ---


@pytest.mark.parametrize("producers", (42, 120))
def test_sparse_wide_fanout_2(producers: int) -> None:
    """gate-harness "Sparse-wide fanout": every product declared by exactly 2 producers."""
    cfg = build_family_config(1000, family=SPARSE_WIDE, producers=producers, regime=SHOCKED,
                              fanout=2)
    decl = _declarers(cfg)
    assert len(decl) == producers  # W = P/3 products per layer, 3 layers
    assert all(len(ps) == 2 for ps in decl.values())
    # each producer declares `fanout` products of its own layer only
    for p in cfg.producer_configs:
        assert len(p.produces) == 2
        assert {_layer_of(cfg, x) for x in p.produces} == {_producer_layer(p.id)}


@pytest.mark.parametrize("fanout", [1, 3, 5])
def test_sparse_wide_explicit_fanout(fanout: int) -> None:
    cfg = build_sparse_wide_config(7, producers=42, regime=CALM, fanout=fanout)
    assert set(Counter(len(v) for v in _declarers(cfg).values())) == {fanout}


@pytest.mark.parametrize("producers", (42, 120))
def test_dense_every_layer_producer_declares_every_layer_product(producers: int) -> None:
    """gate-harness "Dense fanout"."""
    cfg = build_family_config(1000, family=DENSE, producers=producers, regime=SHOCKED)
    decl = _declarers(cfg)
    assert len(decl) == 3 * DENSE_WIDTH
    for product, ps in decl.items():
        layer = _layer_of(cfg, product)
        assert sorted(ps) == sorted(p.id for p in cfg.producer_configs
                                    if _producer_layer(p.id) == layer)
        assert len(ps) == producers // 3


@pytest.mark.parametrize("producers", BUILDABLE)
def test_scaling_rules_at_every_size(producers: int) -> None:
    """Module docstring rules at 12, 42 and 120: forced-strong every m-th producer (m =
    min(fanout, 4)) and round-robin maintenance; a sparse layer-0 product carries exactly 3
    producers' maintenance at every size."""
    n = producers // 3
    for family, m in ((DENSE, 4), (SPARSE_WIDE, 2)):
        cfg = build_family_config(1000, family=family, producers=producers, regime=SHOCKED)
        for p in cfg.producer_configs:
            if int(p.id.split("_")[1]) % m == 0:
                assert sum(p.capacities.values()) >= 80, p.id
        load = Counter(p.restoration.require for p in cfg.producer_configs)
        assert all(p.restoration.require == p.upgrade.require for p in cfg.producer_configs)
        if family == SPARSE_WIDE:
            assert len(load) == n and set(load.values()) == {3}
        else:
            assert len(load) == DENSE_WIDTH and max(load.values()) - min(load.values()) <= 1


def test_family_fanout_is_explicit() -> None:
    assert FAMILIES[DENSE].fanout(40, None) == 40
    assert FAMILIES[SPARSE_WIDE].fanout(40, None) == 2
    with pytest.raises(ValueError, match="dense fanout"):
        build_dense_config(0, producers=42, regime=CALM, fanout=2)
    with pytest.raises(ValueError, match="sparse fanout"):
        build_sparse_wide_config(0, producers=12, regime=CALM, fanout=5)


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("producers", BUILDABLE)
@pytest.mark.parametrize("regime", (CALM, SHOCKED))
def test_worlds_validate_adr_0028(family: str, producers: int, regime: str) -> None:
    """Config validation passes and every declared product's recipe types are owned (ADR-0028)."""
    cfg = build_family_config(1003, family=family, producers=producers, regime=regime)
    cfg.validate()
    assert len(cfg.producer_configs) == producers
    assert len(cfg.consumers) == num_consumers(producers)
    for p in cfg.producer_configs:
        for x in p.produces:
            assert set(cfg.process_sheet(x).require) <= set(p.capacities)
    # every product has a strong declarer (capacity >= 80): no product is only marginally makeable
    caps = {p.id: sum(p.capacities.values()) for p in cfg.producer_configs}
    assert all(max(caps[d] for d in ds) >= 80 for ds in _declarers(cfg).values())


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("producers", SIZES)
def test_consumer_pools_cover_every_top_product(family: str, producers: int) -> None:
    cfg = build_family_config(1000, family=family, producers=producers, regime=SHOCKED)
    top = {ps.product for ps in cfg.process_sheets if _layer_of(cfg, ps.product) == 2}
    pools = [set(c.product_pool) for c in cfg.consumers]
    assert set().union(*pools) == top
    if family == DENSE:
        assert all(p == top for p in pools)


# --- Constant economy ratios across sizes ---


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("regime", (CALM, SHOCKED))
def test_economy_ratios_match_between_12_and_42(family: str, regime: str) -> None:
    """gate-harness "More producers, same pressure" (family-level contract): the family's demand
    per unit of initial top-layer throughput, as a ratio of sums over its recorded sampling
    protocol, agrees within the recorded tolerance between 12 and 42 (gating) and, as a
    non-gating check, 120. Emission per producer is exact in every sampled world; emission per
    consumer is exact where 3·P/12 is whole (12, 120) and 4400·42/12 / 11 = 1400 at 42."""
    fam = FAMILIES[family]
    assert fam.ratio_tolerance == 0.10
    assert fam.ratio_sample == ((12, 1000, 100), (42, 1000, 40), (120, 1000, 20))
    r12 = family_demand_ratio(family, 12, regime)
    for size in (42, 120):
        assert abs(family_demand_ratio(family, size, regime) / r12 - 1) <= fam.ratio_tolerance
    assert r12 == demand_throughput_ratio([build_family_config(
        s, family=family, producers=12, regime=regime) for s in range(1000, 1100)])
    for size, per_consumer in ((12, 4400 / 3), (42, 1400), (120, 4400 / 3)):
        for s in fam.sample_seeds(size):
            cfg = build_family_config(s, family=family, producers=size, regime=regime)
            assert emission_per_producer(cfg) == 4400 / 12
            assert cfg.cycle_emission / len(cfg.consumers) == per_consumer
    assert (cycle_emission(12), cycle_emission(42), cycle_emission(120)) == (4400, 15400, 44000)
    # 3·42/12 = 10.5 rounds half up to 11 (Python's round() would give 10; module docstring)
    assert (num_consumers(12), num_consumers(42), num_consumers(120)) == (3, 11, 30)


def test_sample_protocol_refuses_unrecorded_size() -> None:
    with pytest.raises(ValueError, match="no ratio sample at 24"):
        FAMILIES[DENSE].sample_seeds(24)


def test_throughput_and_demand_definitions() -> None:
    """Hand check on one world: throughput = sum over top producers of capacity / mean recipe
    requirement of their declared products; demand = midpoint of each B9 quantity range."""
    cfg = build_family_config(1000, family=SPARSE_WIDE, producers=12, regime=SHOCKED)
    expect = 0.0
    for p in cfg.producer_configs:
        if p.id.startswith("p2_"):
            reqs = [cfg.process_sheet(x).require["capacity-2"] for x in p.produces]
            expect += p.capacities["capacity-2"] / (sum(reqs) / len(reqs))
    assert top_layer_throughput(cfg) == pytest.approx(expect)
    assert demand_units_per_cycle(cfg) == sum((1 + c.quantity_max) / 2 for c in cfg.consumers)


# --- determinism and regimes ---


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("producers", SIZES)
def test_builder_is_deterministic(family: str, producers: int) -> None:
    a = build_family_config(1001, family=family, producers=producers, regime=CALM)
    assert a == build_family_config(1001, family=family, producers=producers, regime=CALM)
    assert a != build_family_config(1002, family=family, producers=producers, regime=CALM)


@pytest.mark.parametrize("family", FAMS)
def test_calm_and_shocked_share_topology(family: str) -> None:
    """Schedules are drawn last, so the regimes differ only in the consumers' schedules."""
    calm = build_family_config(1000, family=family, producers=42, regime=CALM)
    shocked = build_family_config(1000, family=family, producers=42, regime=SHOCKED)
    assert calm.producer_configs == shocked.producer_configs
    assert calm.process_sheets == shocked.process_sheets
    assert [c.products for c in calm.consumers] == [c.products for c in shocked.consumers]
    assert all(not c.schedule for c in shocked.consumers)
    for c in calm.consumers:
        assert len(c.schedule) == 7
        assert all(p in c.products and c.quantity_min <= q <= c.quantity_max
                   for p, q in c.schedule)


def _digest(cfg: Configuration, seed: int) -> str:
    env = TokenomicsAECEnv(config=cfg, horizon=12)
    env.reset(seed=seed)
    return env.episode_stats["demand_digest"]


@pytest.mark.parametrize("family", FAMS)
@pytest.mark.parametrize("producers", SIZES)
def test_calm_demand_identical_under_seeds_0_and_9(family: str, producers: int) -> None:
    """stochastic-environment "Episode seed has no effect": the calm world's demand digest is
    the same under episode seeds 0 and 9; the shocked world's is not (B9 random demand)."""
    calm = build_family_config(1000, family=family, producers=producers, regime=CALM)
    assert _digest(calm, 0) == _digest(calm, 9)
    shocked = build_family_config(1000, family=family, producers=producers, regime=SHOCKED)
    assert _digest(shocked, 0) != _digest(shocked, 9)


def test_dense_12_shocked_keeps_build_scale_config_draws() -> None:
    """At 12 producers the dense shocked world is build_scale_config's economy except that
    maintenance is spread over the layer-0 products (module docstring)."""
    fam = build_family_config(1000, family=DENSE, producers=12, regime=SHOCKED)
    base = build_scale_config(1000)
    assert fam.process_sheets == base.process_sheets
    assert fam.consumers == base.consumers
    assert fam.cycle_emission == base.cycle_emission
    for a, b in zip(fam.producer_configs, base.producer_configs, strict=True):
        assert (a.id, a.capacities, a.degradation, a.produces) == (
            b.id, b.capacities, b.degradation, b.produces)
        assert (a.restoration.restores, a.upgrade.increases) == (
            b.restoration.restores, b.upgrade.increases)


# --- generator refs ---


@pytest.mark.parametrize("generator,params", [
    ("dense_family", {"producers": 42, "regime": CALM}),
    ("sparse_wide_family", {"producers": 42, "regime": SHOCKED, "fanout": 2}),
])
def test_families_replay_from_generator_ref(generator: str, params: dict) -> None:
    assert generator in REPRODUCERS
    rec = TaskRecord(task_id="t", task_family="scale", generator_version="v1",
                     serializer_version="v1", world_seed=1000, episode_seed=1000,
                     world_spec={"generator": generator, "params": params})
    cfg = replay(TaskRecord.from_json_line(rec.to_json_line()))
    family = DENSE if generator == "dense_family" else SPARSE_WIDE
    assert cfg == build_family_config(1000, family=family, **params)
