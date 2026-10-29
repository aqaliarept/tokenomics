"""World space: a world as a point on named axes (add-m3-development-audit design D10, ADR-0057;
world-space spec). The headroom atlas (design D12) samples it; nothing here plays an episode.

A world is a size (12 or 42 producers, the scale target; anything else is refused), a world
seed and one value per **lever**; `build_world` turns them into a configuration through
`families.build_family_config` and its world-space levers (families docstring), and
`world_record` records the world with its **measured coordinates**, which are what coverage and
every reading use (several levers move one coordinate, and the realized value depends on the
world's draws).

Axes and levers (D10; task 4.6 calibrates the ranges; neutral values reproduce today's worlds):
- choice: `sparse_f<E>` (sparse_wide, E declarers per product) or `dense` (the whole layer);
  12: sparse_f2, sparse_f4, dense; 42: sparse_f2, sparse_f4, sparse_f7, sparse_f14, dense.
  Neutral sparse_f2;
- capacity tightness and order size: k_c in [0.5, 8], m_q in [1, 8], capacity scale c in
  [0.25, 1], each log-uniform. Neutral 1, 1, 1;
- producer heterogeneity: h_cap, h_deg, h_cost in [0, 2], uniform. Neutral 1 (no transform);
- shocks: off / on; on = the `shocks-i` kinds with counts x r (r in [0.5, 3]), hit magnitudes
  centred at m (m in [0.2, 0.8], width 0.3) and outage durations [d, d + 3] (d in [2, 10],
  integer), a profile object built in code (`families.shock_profile_from_levers`), never its
  drift. Neutral off, or on at (1, 0.45, 3);
- demand pattern: cyclic (the calm regime's schedules), iid (B9 per-cycle sampling), drift
  (j = 0.34, the registered `shocks-i` value, not sampled), burst, hit, seasonal, switch (design
  D11 parameters, drawn per world from the world seed: burst p in [0.02, 0.10] with factors 5-10;
  hit alpha in [0.8, 2], period in [5, 30]; seasonal amplitude in [0.3, 0.8], period in
  [10, 40], phase in [0, period); switch low weight in [0.05, 0.3], 1-4 switches in [11, 90]
  at least 10 apart). Product patterns (hit, switch) widen each pool to min(4, W) products.
  Neutral cyclic (calm) or drift (shocked).

Sampling (`sample_levers`): a seeded Latin hypercube per size, numpy only: one column per
continuous lever (stratum i of n gets one uniform point, mapped to the lever's scale) and one per
categorical axis (choice, shocks, pattern), whose levels are equal-width strata of the stratum
index, so every level gets floor or ceil of n / levels worlds. World seed =
sha256(atlas, size, index); episode seeds hash the world id, as `routing_audit.fanout_worlds`.

Measured coordinates (`measure`), on the demand schedule the env builds for the measurement seed
(the world seed) over 100 cycles, so drift and patterns are covered:
- rho: expected demand units per cycle / initial top-layer throughput
  (`families.demand_units_per_cycle` / `families.top_layer_throughput`; the B9 range midpoint,
  without the burst or seasonal factor, which the realized volatility carries);
- sigma: mean over the schedule's orders of quantity x the product's capacity units per unit,
  divided by the median capacity of the product's declarers;
- cv_capacity, cv_degradation, cv_cost: within-layer spread, the mean over layers of the
  coefficient of variation (population) among the layer's producers of capacity, of degradation,
  and of restoration and upgrade cost (each its own CV, averaged), because the h levers act
  within a layer (around its median) and the layers differ by construction; 0 at h = 0;
- shock_events_per_producer (expected events per producer per 100 cycles: every generator's
  start window lies in the 100-cycle horizon), shock_hit_magnitude and shock_outage_duration
  (range midpoints); 0 without shocks;
- demand_concentration (Herfindahl of demanded units over products) and demand_volatility
  (coefficient of variation of per-cycle demanded units).
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..domain.config import (PATTERN_BURST, PATTERN_HIT, PATTERN_SEASONAL, PATTERN_SWITCH,
                             Configuration, DemandDrift, DemandPattern)
from ..domain.shocks import ShockKind
from .families import (CALM, DENSE, SHOCKED, SHOCKS_I_DRIFT_J, SHOCKS_I_WINDOW, SPARSE_WIDE,
                       build_family_config, demand_units_per_cycle, shock_profile_from_levers,
                       top_layer_throughput)
from .routing_audit import config_digest, grid_seed

GENERATOR_VERSION = "world-space-v1"
SIZES = (12, 42)
MEASURE_HORIZON = 100

DENSE_CHOICE = "dense"
NEUTRAL_CHOICE = "sparse_f2"
CHOICES: dict[int, tuple[str, ...]] = {
    12: ("sparse_f2", "sparse_f4", DENSE_CHOICE),
    42: ("sparse_f2", "sparse_f4", "sparse_f7", "sparse_f14", DENSE_CHOICE),
}

CYCLIC = "cyclic"
IID = "iid"
DRIFT = "drift"
PATTERNS = (CYCLIC, IID, DRIFT, PATTERN_BURST, PATTERN_HIT, PATTERN_SEASONAL, PATTERN_SWITCH)
PRODUCT_PATTERNS = frozenset({PATTERN_HIT, PATTERN_SWITCH})
PRODUCT_PATTERN_POOL = 4  # pool width min(4, W) for product patterns (design D11 "Pools")
DRIFT_J = SHOCKS_I_DRIFT_J

LOG = "log"
LINEAR = "linear"
INTEGER = "integer"


@dataclass(frozen=True)
class LeverRange:
    """A continuous lever's range, sampling scale and neutral value."""

    lo: float
    hi: float
    scale: str
    neutral: float

    def value(self, u: float) -> float | int:
        """The lever value at unit coordinate u in [0, 1)."""
        if self.scale == LOG:
            return float(self.lo * (self.hi / self.lo) ** u)
        if self.scale == INTEGER:
            return int(min(self.hi, math.floor(self.lo + u * (self.hi - self.lo + 1))))
        return float(self.lo + u * (self.hi - self.lo))


# Continuous levers in Latin-hypercube column order (D10; ADR-0057 lever table).
LEVERS: dict[str, LeverRange] = {
    "k_c": LeverRange(0.5, 8.0, LOG, 1.0),
    "m_q": LeverRange(1.0, 8.0, LOG, 1.0),
    # c's upper end calibrated by task 4.6 (D10's initial range [0.25, 1]): with c <= 1 the
    # screen sample's smallest sigma was 0.058 / 0.071 (12 / 42), short of the required 0.05;
    # [0.25, 2] (families.check_capacity_scale's own limit) gives 0.032 / 0.038 (ADR-0057).
    "capacity_scale": LeverRange(0.25, 2.0, LOG, 1.0),
    "h_cap": LeverRange(0.0, 2.0, LINEAR, 1.0),
    "h_deg": LeverRange(0.0, 2.0, LINEAR, 1.0),
    "h_cost": LeverRange(0.0, 2.0, LINEAR, 1.0),
    "shock_rate": LeverRange(0.5, 3.0, LINEAR, 1.0),
    "hit_magnitude": LeverRange(0.2, 0.8, LINEAR, 0.45),
    "outage_duration": LeverRange(2, 10, INTEGER, 3),
}
SHOCK_LEVERS = ("shock_rate", "hit_magnitude", "outage_duration")
CATEGORICAL = ("choice", "shocks", "pattern")  # after the continuous columns

# D11 parameter ranges per pattern kind: (name, lo, hi, integer). Seasonal phase is drawn in
# [0, period) after the period. Fields not listed keep DemandPattern's defaults (burst factors
# 5-10; switch window [11, 90], gap 10).
PATTERN_RANGES: dict[str, tuple[tuple[str, float, float, bool], ...]] = {
    PATTERN_BURST: (("burst_p", 0.02, 0.10, False),),
    PATTERN_HIT: (("hit_alpha", 0.8, 2.0, False), ("hit_period", 5, 30, True)),
    PATTERN_SEASONAL: (("amplitude", 0.3, 0.8, False), ("period", 10, 40, True)),
    PATTERN_SWITCH: (("low_weight", 0.05, 0.3, False), ("switch_count", 1, 4, True)),
}


@dataclass(frozen=True)
class Levers:
    """One value per lever (D10); every default is the neutral value, so an unset lever is
    neutral and recorded as such (spec "One lever per axis"). `pattern_params` holds the D11
    parameters of `pattern` as sorted (name, value) pairs; unset ones keep DemandPattern's
    defaults."""

    choice: str = NEUTRAL_CHOICE
    k_c: float = 1.0
    m_q: float = 1.0
    capacity_scale: float = 1.0
    h_cap: float = 1.0
    h_deg: float = 1.0
    h_cost: float = 1.0
    shocks: bool = False
    shock_rate: float = 1.0
    hit_magnitude: float = 0.45
    outage_duration: int = 3
    pattern: str = CYCLIC
    pattern_params: tuple[tuple[str, float | int], ...] = field(default=())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern_params"] = dict(self.pattern_params)
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> Levers:
        d = dict(d)
        d["pattern_params"] = tuple(sorted(dict(d.get("pattern_params", {})).items()))
        return cls(**d)


@dataclass(frozen=True)
class WorldPoint:
    """One sampled world: where it sits in the sample and its levers."""

    atlas: str
    size: int
    index: int
    seed: int
    levers: Levers
    shock_window: tuple[int, int] = SHOCKS_I_WINDOW

    @property
    def world_id(self) -> str:
        return world_id(self.atlas, self.size, self.index)

    def config(self) -> Configuration:
        return build_world(self.seed, producers=self.size, levers=self.levers,
                           shock_window=self.shock_window)


def check_size(size: int) -> int:
    if size not in SIZES:
        raise ValueError(f"the world space builds {SIZES[0]} and {SIZES[1]} producers only "
                         f"(scale target), got size {size}")
    return size


def world_seed(atlas: str, size: int, index: int) -> int:
    """sha256(atlas, size, index), 31 bits (D10 "Seeds")."""
    return grid_seed(atlas, size, index)


def world_id(atlas: str, size: int, index: int) -> str:
    return f"ws-{atlas}-{size}-{index:04d}"


def episode_seeds(wid: str, count: int, stage: str = "") -> tuple[int, ...]:
    """Episode seeds hash the world id (as `routing_audit.fanout_world`); a `stage` salt gives
    a disjoint fresh set (e.g. the atlas confirm stage)."""
    parts = (wid, stage) if stage else (wid,)
    return tuple(grid_seed(*parts, j) for j in range(count))


def _stream(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def check_levers(size: int, levers: Levers) -> Levers:
    """Refuses a choice the size does not have, a lever outside its D10 range, an unknown
    pattern or a parameter its kind does not take, naming it."""
    if levers.choice not in CHOICES[check_size(size)]:
        raise ValueError(f"choice {levers.choice!r} not available at {size} producers; "
                         f"known: {CHOICES[size]}")
    for name, rng in LEVERS.items():
        v = getattr(levers, name)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not rng.lo <= v <= rng.hi:
            raise ValueError(f"lever {name} must be in [{rng.lo}, {rng.hi}], got {v!r}")
        if rng.scale == INTEGER and not isinstance(v, int):
            raise ValueError(f"lever {name} must be an integer, got {v!r}")
    if not isinstance(levers.shocks, bool):
        raise ValueError(f"lever shocks must be a bool, got {levers.shocks!r}")
    if levers.pattern not in PATTERNS:
        raise ValueError(f"unknown demand pattern {levers.pattern!r}; known: {PATTERNS}")
    allowed = {n for n, *_ in PATTERN_RANGES.get(levers.pattern, ())}
    if levers.pattern == PATTERN_SEASONAL:
        allowed.add("phase")
    unknown = sorted(set(dict(levers.pattern_params)) - allowed)
    if unknown:
        raise ValueError(f"pattern {levers.pattern} takes no parameter {unknown}")
    return levers


def _choice(size: int, choice: str) -> tuple[str, int | None]:
    if choice == DENSE_CHOICE:
        return DENSE, None
    return SPARSE_WIDE, int(choice.removeprefix("sparse_f"))


def _layer_width(size: int, choice: str) -> int:
    return 4 if choice == DENSE_CHOICE else size // 3


def pool_width(size: int, levers: Levers) -> int | None:
    """The product-pattern pool width min(4, W), else None (pools unchanged)."""
    if levers.pattern in PRODUCT_PATTERNS:
        return min(PRODUCT_PATTERN_POOL, _layer_width(size, levers.choice))
    return None


def build_world(seed: int, *, producers: int, levers: Levers | Mapping | None = None,
                shock_window: Sequence[int] = SHOCKS_I_WINDOW) -> Configuration:
    """The world `levers` describe at `producers` with world seed `seed` (module docstring).
    Neutral levers return `build_family_config`'s world unchanged: calm for the cyclic pattern,
    shocked (with `shocks-i` when shocks are on, `shock_profile` included) otherwise.
    `shock_window` moves every generator's start window (a horizon-12 smoke uses
    `shock_profiles.SMOKE_START_WINDOW`); it is recorded with the world."""
    if levers is None:
        levers = Levers()
    elif isinstance(levers, Mapping):
        levers = Levers.from_dict(levers)
    check_levers(producers, levers)
    family, fanout = _choice(producers, levers.choice)
    shocks = (shock_profile_from_levers(levers.shock_rate, levers.hit_magnitude,
                                        levers.outage_duration, tuple(shock_window))
              if levers.shocks else None)
    cfg = build_family_config(
        seed, family=family, producers=producers,
        regime=CALM if levers.pattern == CYCLIC else SHOCKED, fanout=fanout,
        capacity_scale=levers.capacity_scale, consumer_multiplier=levers.k_c,
        quantity_multiplier=levers.m_q, h_cap=levers.h_cap, h_deg=levers.h_deg,
        h_cost=levers.h_cost, pool_width=pool_width(producers, levers), shocks=shocks)
    if levers.pattern == DRIFT:
        cfg = replace(cfg, demand_drift=DemandDrift(DRIFT_J))
    elif levers.pattern not in (CYCLIC, IID):
        cfg = replace(cfg, demand_pattern=DemandPattern(levers.pattern,
                                                        **dict(levers.pattern_params)))
    cfg.validate()
    return cfg


# --- sampling ---


def pattern_params(kind: str, seed: int) -> tuple[tuple[str, float | int], ...]:
    """The D11 parameters of `kind` for the world with seed `seed`, uniform in their ranges on a
    stream of the world seed (the seasonal phase uniform in [0, period))."""
    if kind not in PATTERN_RANGES:
        return ()
    rng = _stream(seed, "pattern-params", kind)
    out: dict[str, float | int] = {}
    for name, lo, hi, integer in PATTERN_RANGES[kind]:
        out[name] = int(rng.integers(lo, hi + 1)) if integer else float(rng.uniform(lo, hi))
    if kind == PATTERN_SEASONAL:
        out["phase"] = float(rng.uniform(0, out["period"]))
    return tuple(sorted(out.items()))


def lhs_columns(atlas: str, size: int, n: int) -> dict[str, np.ndarray]:
    """The Latin hypercube for one size: per continuous lever, the unit coordinate
    (stratum + uniform) / n; per categorical axis, its level index, (stratum · levels) // n,
    so every level gets floor or ceil of n / levels rows. Columns in a fixed order, each its own
    permutation of the strata."""
    if n < 1:
        raise ValueError(f"sample size must be >= 1, got {n}")
    check_size(size)
    rng = _stream(atlas, size, "lhs", GENERATOR_VERSION)
    levels = {"choice": len(CHOICES[size]), "shocks": 2, "pattern": len(PATTERNS)}
    cols: dict[str, np.ndarray] = {}
    for name in LEVERS:
        cols[name] = (rng.permutation(n) + rng.random(n)) / n
    for name in CATEGORICAL:
        cols[name] = rng.permutation(n) * levels[name] // n
    return cols


def sample_levers(atlas: str, size: int, n: int) -> list[Levers]:
    """`n` lever settings for `size` from its Latin hypercube; pattern parameters come from
    each world's own seed, so the sample is a function of (atlas, size, n)."""
    cols = lhs_columns(atlas, size, n)
    out = []
    for i in range(n):
        values = {name: rng.value(float(cols[name][i])) for name, rng in LEVERS.items()}
        kind = PATTERNS[int(cols["pattern"][i])]
        out.append(Levers(choice=CHOICES[size][int(cols["choice"][i])],
                          shocks=bool(cols["shocks"][i]), pattern=kind,
                          pattern_params=pattern_params(kind, world_seed(atlas, size, i)),
                          **values))
    return out


def sample_worlds(atlas: str, size: int, n: int,
                  shock_window: Sequence[int] = SHOCKS_I_WINDOW) -> list[WorldPoint]:
    """The atlas sample for one size: world i has seed sha256(atlas, size, i)."""
    return [WorldPoint(atlas, size, i, world_seed(atlas, size, i), lv, tuple(shock_window))
            for i, lv in enumerate(sample_levers(atlas, size, n))]


# --- measured coordinates and the record ---


def _cv(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _layer_cv(producers: Sequence, read: Callable[[object], float]) -> float:
    """Mean over layers (producers grouped by their capacity types) of the CV of `read`."""
    layers: dict[tuple[str, ...], list[float]] = {}
    for p in producers:
        layers.setdefault(tuple(sorted(p.capacities)), []).append(read(p))
    return statistics.fmean(_cv(v) for v in layers.values())


def declarers(cfg: Configuration) -> dict:
    """Declaring producers per product (ADR-0028: `produces` is authoritative)."""
    count: Counter = Counter(x for p in cfg.producer_configs for x in p.produces)
    return {str(k): count[k] for k in sorted(count)}


def realized_demand(cfg: Configuration, seed: int,
                    horizon: int = MEASURE_HORIZON) -> list[list[tuple[int, int]]]:
    """Per cycle, every consumer's (product, quantity) from the schedules the env builds at
    reset (`_demand_schedules`, the demand digest input), replayed as `ScheduledConsumer` does."""
    from ..env import TokenomicsAECEnv

    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=seed)
    schedules = env._demand_schedules
    return [[tuple(s[t % len(s)]) for s in schedules.values()] for t in range(horizon)]


def measure(cfg: Configuration, seed: int, horizon: int = MEASURE_HORIZON) -> dict[str, float]:
    """The measured coordinates of D10 (module docstring)."""
    producers = cfg.producer_configs
    orders = realized_demand(cfg, seed, horizon)
    caps: dict = {}
    for p in producers:
        for x in p.produces:
            caps.setdefault(x, []).append(sum(p.capacities.values()))
    med_cap = {x: statistics.median(v) for x, v in caps.items()}
    units_per_unit = {s.product: sum(s.require.values()) for s in cfg.process_sheets}
    flat = [o for cycle in orders for o in cycle]
    sigma = statistics.fmean(q * units_per_unit[x] / med_cap[x] for x, q in flat) if flat else 0.0
    by_product: Counter = Counter()
    for x, q in flat:
        by_product[x] += q
    total = sum(by_product.values())
    per_cycle = [sum(q for _, q in cycle) for cycle in orders]
    gens = cfg.shocks.generators if cfg.shocks else ()
    hits = [g for g in gens if g.kind == ShockKind.CAPACITY_HIT]
    outages = [g for g in gens if g.kind == ShockKind.PRODUCER_OUTAGE]
    return {
        "rho": demand_units_per_cycle(cfg) / top_layer_throughput(cfg),
        "sigma": sigma,
        "cv_capacity": _layer_cv(producers, lambda p: sum(p.capacities.values())),
        "cv_degradation": _layer_cv(producers, lambda p: p.degradation),
        "cv_cost": (_layer_cv(producers, lambda p: p.restoration.cost)
                    + _layer_cv(producers, lambda p: p.upgrade.cost)) / 2,
        "shock_events_per_producer": sum(g.count for g in gens) / len(producers),
        "shock_hit_magnitude": (statistics.fmean(sum(g.magnitude) / 2 for g in hits)
                                if hits else 0.0),
        "shock_outage_duration": (statistics.fmean(sum(g.duration) / 2 for g in outages)
                                  if outages else 0.0),
        "demand_concentration": (sum((u / total) ** 2 for u in by_product.values())
                                 if total else 0.0),
        "demand_volatility": _cv(per_cycle) if per_cycle else 0.0,
    }


def world_record(point: WorldPoint) -> dict:
    """The D10 record: levers, seed, generator version, measured coordinates (measurement seed
    = the world seed, 100 cycles), declarers per product and the config's sha256 (as
    `routing_audit.world_record`)."""
    cfg = point.config()
    return {
        "world_id": point.world_id, "atlas": point.atlas, "size": point.size,
        "index": point.index, "seed": point.seed, "generator_version": GENERATOR_VERSION,
        "levers": point.levers.to_dict(), "shock_window": list(point.shock_window),
        "pool_width": pool_width(point.size, point.levers),
        "measure_seed": point.seed, "measure_horizon": MEASURE_HORIZON,
        "coordinates": measure(cfg, point.seed),
        "declarers_per_product": declarers(cfg), "config_sha256": config_digest(cfg),
    }


def point_from_record(record: Mapping) -> WorldPoint:
    """The world a record describes; refuses another generator version."""
    if record.get("generator_version") != GENERATOR_VERSION:
        raise ValueError(f"record of generator {record.get('generator_version')!r}, this is "
                         f"{GENERATOR_VERSION}")
    return WorldPoint(record["atlas"], record["size"], record["index"], record["seed"],
                      Levers.from_dict(record["levers"]), tuple(record["shock_window"]))


def config_from_record(record: Mapping) -> Configuration:
    """Rebuild a recorded world; its config sha256 must equal the recorded one."""
    cfg = point_from_record(record).config()
    digest = config_digest(cfg)
    if digest != record["config_sha256"]:
        raise ValueError(f"{record['world_id']}: rebuilt config sha256 {digest} != recorded "
                         f"{record['config_sha256']}")
    return cfg
