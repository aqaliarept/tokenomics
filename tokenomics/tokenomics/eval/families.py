"""Gate world families: dense and sparse-wide layered economies at the gate sizes, ~12 and ~42
producers (design D8, D9, "Economy sizes", add-measurement-gate-harness; gate-harness "Scale
families", "Constant economy ratios across sizes"; stochastic-environment "Deterministic cyclic
demand"). Any positive multiple of 3 builds; ~120 is a future, non-gating size.

A family world is a depth-3 layered supply chain with P/3 producers per layer, built from the
same draws as `scale.build_scale_config` (recipe requirements, strong/marginal capacities,
degradation, investment amounts and costs, inventory, consumer quantity ranges) and its helpers.
What differs is how wide a layer is and who declares what (ADR-0028: `produces` is authoritative):

- **dense**: W = 4 products per layer, every layer producer declares all of them
  (fanout = P/3, the whole layer).
- **sparse_wide**: W = P/3 products per layer, each declared by `fanout` producers (default 2),
  round-robin: producer k declares products k, k+1, ..., k+fanout-1 (mod W), so product j is
  declared by a contiguous window of `fanout` producers.

Scaling rules, chosen so a 42-producer (or 120-producer) world puts the same pressure on its
economy as a 12-producer one (the ratios `family_demand_ratio` and `emission_per_producer` measure). Both
refinements of `build_scale_config` below are recorded in design D9:
- consumers = 3·P/12 rounded half up; emission = 4400·P/12 (4400 = `build_scale_config`'s
  12-producer emission), so emission per producer is exact and, when 3·P/12 is whole (12, 120),
  emission per consumer too. At 42, 3·P/12 = 10.5: half up gives 11 consumers, not Python's
  banker's 10, because 10 leaves the 42-producer demand ratio 7–9% under the 12-producer one
  (close to the 10% tolerance) while 11 keeps it within ±3% (task 8.7);
- the forced-strong producers (the "first producer is always strong" rule of
  `build_scale_config`) become every m-th producer of a layer, m = min(fanout, 4): with m =
  fanout every sparse product has a strong declarer, and for dense m = 4 reproduces the
  12-producer rule while keeping the forced fraction near 1/4 (4/14 at 42, 10/40 at 120);
- maintenance (restoration / upgrade / inventory-expansion input) is spread round-robin over the
  layer-0 products instead of all on product 0: at 42 (120) producers a single maintenance
  product made by 2 sparse declarers would carry 3.5x (10x) the 12-producer load. Round-robin
  gives every sparse layer-0 product the maintenance of exactly 3 producers (P / W = 3) at any
  size; dense spreads P producers over its 4 layer-0 products (3 each at 12; 10-11 at 42).

The demand ratio is a FAMILY quantity (gate-harness "Constant economy ratios across sizes"): it
is compared as summed demand / summed throughput over the sampling protocol recorded in the
`Family` (`ratio_sample`), never world by world. A single 12-producer world has only 4 top-layer
producers, each strong (80-140) or marginal (8-20) by draw, so its own ratio can sit far (tens of
percent) from the family's; that world-to-world spread is kept on purpose, since development
worlds should span headroom. Emission per producer is exact in every world.

Capacity scale (task 10.2, design D10): a family may carry a `capacity_scale` that multiplies
every capacity-unit quantity a producer draws (initial capacity, restoration `restores`, upgrade
`increases`), rounded half up and floored at 1, after the draw, so the random stream and every
other value are unchanged. It applies at every size, so the demand/throughput ratio test still
compares like with like. The neutral value 1.0 skips the transform, so a 1.0 world is byte for
byte the unscaled one. It is the retuning lever for a family that fails the headroom check.

Regimes (design D8): **shocked** keeps the B9 seeded random demand (product uniform over the
consumer's pool, quantity uniform in its range, drawn per episode from the episode seed).
**calm** draws each consumer's cyclic `schedule` once here, from the world seed, within the same
pool and quantity range, so every episode sees identical demand. Schedules are drawn after every
other value, so a world's calm and shocked variants share the exact same topology.

Shock profile (add-shocks-i-drift D7): a shocked world may name a profile from
`eval/registry/shock_profiles_v1.json`; it adds seeded shock generators (counts per producer)
and demand drift after every other value is drawn, so topology and B9 bounds are unchanged.
Without one, a world is byte for byte what it was before profiles existed.

World-space levers (add-m3-development-audit design D10, ADR-0057; the atlas generator
`eval/world_space.py` sets them): a consumer multiplier, a quantity multiplier, three
heterogeneity spreads, a consumer-pool width for product patterns and a shock profile object.
Each runs after every draw above, on values already drawn, so the main stream, the recipes and
the `produces` sets never move when a lever moves, and each neutral value skips its transform,
so an all-neutral world is byte for byte the world without levers. In order:
- heterogeneity h_cap / h_deg / h_cost: each drawn capacity, degradation, restoration cost and
  upgrade cost v becomes med + h·(v − med), med the median of that draw over the world's layer,
  rounded half up and clamped to its valid range (capacity and cost ≥ 1, degradation in
  [0, 100]); h = 0 gives every producer of a layer the same value;
- consumer multiplier k_c: consumers = max(1, round_half_up(k_c · 3P/12)). Fewer keeps the
  first ones as drawn; extra consumer i draws its quantity bound (and, calm, its schedule) on a
  stream of its own derived from the world seed and i, with the family's pool rule at the base
  consumer count (the pool of consumer i mod C), so per-product load stays balanced and an
  extra consumer's draws do not depend on how many others exist;
- pool width (product patterns, design D11): every B9 consumer's pool becomes the window of
  max(its length, `pool_width`) consecutive top-layer products from its start;
- quantity multiplier m_q: quantity range [1, q_max] becomes [round_half_up(m_q),
  round_half_up(m_q · q_max)], and every calm schedule quantity q becomes round_half_up(m_q · q);
- shocks: a `ShockProfile` object (`shock_profile_from_levers`: the `shocks-i` kinds and window
  with counts × r, hit magnitudes centred at m, outage durations [d, d + 3]) adds its generators
  in either regime and never its drift, which the atlas sets as one value of its demand-pattern
  axis; `shock_profiles_v1.json` is not read or changed.

Pure and deterministic in (family, producers, world seed, regime, fanout, schedule_length,
capacity_scale, shock_profile, the world-space levers).
`build_scale_config` is not touched.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING

from ..domain.config import (Configuration, ConsumerConfig, InventoryExpansion, ProcessSheet,
                             ProducingAgentConfig, Restoration, Upgrade)
from ..domain.shocks import ShockKind, ShockModel
from .scale import _pid, _stride_for_width

if TYPE_CHECKING:
    from .shock_profiles import ShockProfile

DENSE = "dense"
SPARSE_WIDE = "sparse_wide"
CALM = "calm"
SHOCKED = "shocked"
REGIMES = (CALM, SHOCKED)

DEPTH = 3
BASE_PRODUCERS = 12  # the "train small" size every ratio is anchored to
BASE_EMISSION = 4400  # build_scale_config's emission at 12 producers (2000 + 200·3·4)
DENSE_WIDTH = 4
DEFAULT_SPARSE_FANOUT = 2
DEFAULT_SCHEDULE_LENGTH = 7
STRONG_EVERY_MAX = 4
NEUTRAL_CAPACITY_SCALE = 1.0
NEUTRAL_MULTIPLIER = 1.0  # k_c, m_q (module docstring "World-space levers")
NEUTRAL_SPREAD = 1.0  # h_cap, h_deg, h_cost
DEGRADATION_RANGE = (0, 100)  # percent of max capacity per cycle
# The shocks-i values a lever-built profile shifts (add-shocks-i-drift D7; ADR-0057). r = 1,
# m = 0.45, d = 3 reproduce shock_profiles_v1.json's shocks-i kinds exactly.
SHOCKS_I_WINDOW = (21, 90)
SHOCKS_I_HIT_RATE = (1, 6)
SHOCKS_I_OUTAGE_RATE = (1, 12)
SHOCKS_I_HIT_WIDTH = 0.3
SHOCKS_I_OUTAGE_SPAN = 3
SHOCKS_I_DRIFT_J = 0.34
SHOCKS_I_RESILIENCE = (5, 0.05)  # window, tolerance


@dataclass(frozen=True)
class Family:
    """A family definition: its declaration rule, the tolerance within which its 12- and
    42-producer demand ratios must agree (gate-harness "Constant economy ratios across sizes",
    default 10%), and the sampling protocol that ratio is computed under: per size, the world
    seeds `range(first, first + count)` as (size, first, count) triples. Each sample holds
    400–800 top-layer producers (100 × 4 at 12, 40 × 14 at 42, 20 × 40 at 120); the 120 sample
    is kept for the future, non-gating size. `capacity_scale` is the family default the
    registry generator records in each world's ref (module docstring; neutral 1.0)."""

    name: str
    default_fanout: int | None  # None = the whole layer (dense)
    ratio_tolerance: float = 0.10
    ratio_sample: tuple[tuple[int, int, int], ...] = ((12, 1000, 100), (42, 1000, 40),
                                                      (120, 1000, 20))
    capacity_scale: float = NEUTRAL_CAPACITY_SCALE

    def sample_seeds(self, producers: int) -> range:
        for size, first, count in self.ratio_sample:
            if size == producers:
                return range(first, first + count)
        raise ValueError(f"family {self.name} records no ratio sample at {producers} producers")

    def width(self, per_layer: int) -> int:
        """Products per layer."""
        return DENSE_WIDTH if self.name == DENSE else per_layer

    def fanout(self, per_layer: int, fanout: int | None) -> int:
        """Declarers per product. Dense is always the whole layer; an explicit dense fanout must
        say so. Sparse defaults to `default_fanout` and must fit in the layer."""
        if self.name == DENSE:
            if fanout is not None and fanout != per_layer:
                raise ValueError(f"dense fanout is the whole layer ({per_layer}), got {fanout}")
            return per_layer
        f = self.default_fanout if fanout is None else fanout
        if f is None or not 1 <= f <= per_layer:
            raise ValueError(f"sparse fanout must be in [1, {per_layer}], got {f}")
        return f


FAMILIES: dict[str, Family] = {
    DENSE: Family(DENSE, default_fanout=None),
    SPARSE_WIDE: Family(SPARSE_WIDE, default_fanout=DEFAULT_SPARSE_FANOUT),
}


def num_consumers(producers: int) -> int:
    """3·P/12 rounded half up (module docstring): 3 at 12, 11 at 42, 30 at 120."""
    return (6 * producers + BASE_PRODUCERS) // (2 * BASE_PRODUCERS)


def cycle_emission(producers: int) -> int:
    """Linear in P from the 12-producer value; exact for P a multiple of 3."""
    return BASE_EMISSION * producers // BASE_PRODUCERS


def check_capacity_scale(scale: float) -> float:
    """A capacity scale must be a finite number in (0, 2]; bools are refused."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 2:
        raise ValueError(f"capacity_scale must be a number in (0, 2], got {scale!r}")
    return float(scale)


def _scaled(units: int, scale: float) -> int:
    """`units` × `scale`, rounded half up, at least 1; untouched at the neutral scale."""
    if scale == NEUTRAL_CAPACITY_SCALE:
        return units
    return max(1, int(units * scale + 0.5))


def _declared(k: int, width: int, fanout: int, dense: bool) -> list[int]:
    """Within-layer product indices producer k declares."""
    if dense:
        return list(range(width))
    return sorted((k + t) % width for t in range(fanout))


def _pool(i: int, consumers: int, width: int, dense: bool) -> list[int]:
    """Top-layer product indices consumer i may demand. Dense: the whole top layer. Sparse: a
    contiguous window of ceil(W/C) products starting at floor(i·W/C); consecutive windows overlap
    or touch, so together they cover every top product, hence every chain."""
    if dense:
        return list(range(width))
    length = math.ceil(width / consumers)
    start = i * width // consumers
    return [(start + t) % width for t in range(length)]


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def _lever_seed(seed: int, *parts: object) -> int:
    """A process-independent stream seed derived from the world seed (sha256)."""
    digest = hashlib.sha256("|".join(str(p) for p in (seed, *parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def check_multiplier(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 64:
        raise ValueError(f"{name} must be a number in (0, 64], got {value!r}")
    return float(value)


def check_spread(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 4:
        raise ValueError(f"{name} must be a number in [0, 4], got {value!r}")
    return float(value)


def consumers_for(producers: int, multiplier: float) -> int:
    """max(1, round_half_up(k_c · 3P/12)); `num_consumers(P)` at k_c = 1."""
    if multiplier == NEUTRAL_MULTIPLIER:
        return num_consumers(producers)
    return max(1, _round_half_up(multiplier * 3 * producers / BASE_PRODUCERS))


def _spread(values: list[int], h: float, lo: int, hi: int | None) -> list[int]:
    """med + h·(v − med) per value, med the median of `values`, rounded half up, clamped."""
    med = statistics.median(values)
    out = []
    for v in values:
        x = _round_half_up(med + h * (v - med))
        out.append(max(lo, x if hi is None else min(hi, x)))
    return out


def _apply_heterogeneity(configs: list[ProducingAgentConfig], per_layer: int, h_cap: float,
                         h_deg: float, h_cost: float) -> list[ProducingAgentConfig]:
    """The heterogeneity transforms, layer by layer (module docstring)."""
    out: list[ProducingAgentConfig] = []
    for level in range(DEPTH):
        layer = configs[level * per_layer:(level + 1) * per_layer]
        cap_type = f"capacity-{level}"
        caps = [p.capacities[cap_type] for p in layer]
        degs = [p.degradation for p in layer]
        r_costs = [p.restoration.cost for p in layer]
        u_costs = [p.upgrade.cost for p in layer]
        if h_cap != NEUTRAL_SPREAD:
            caps = _spread(caps, h_cap, 1, None)
        if h_deg != NEUTRAL_SPREAD:
            degs = _spread(degs, h_deg, *DEGRADATION_RANGE)
        if h_cost != NEUTRAL_SPREAD:
            r_costs = _spread(r_costs, h_cost, 1, None)
            u_costs = _spread(u_costs, h_cost, 1, None)
        out += [replace(p, capacities={cap_type: c}, degradation=d,
                        restoration=replace(p.restoration, cost=rc),
                        upgrade=replace(p.upgrade, cost=uc))
                for p, c, d, rc, uc in zip(layer, caps, degs, r_costs, u_costs)]
    return out


def shock_profile_from_levers(rate: float = 1.0, hit_magnitude: float = 0.45,
                              outage_duration: int = 3,
                              start_window: tuple[int, int] = SHOCKS_I_WINDOW,
                              name: str = "shocks-i-levers") -> ShockProfile:
    """The parametric shock profile of design D10, built in code: the `shocks-i` kinds with
    counts × `rate` (kept exact as a fraction), hit magnitudes [m − 0.15, m + 0.15] and outage
    durations [d, d + 3]. Its drift_j is `shocks-i`'s, carried but never applied by
    `build_family_config(shocks=...)`. At (1, 0.45, 3) its kinds equal the file's `shocks-i`."""
    from .resilience import ResilienceParams
    from .shock_profiles import KindProfile, ShockProfile

    if isinstance(outage_duration, bool) or not isinstance(outage_duration, int) \
            or outage_duration < 1:
        raise ValueError(f"outage_duration must be an integer >= 1, got {outage_duration!r}")
    half = SHOCKS_I_HIT_WIDTH / 2
    lo, hi = round(hit_magnitude - half, 9), round(hit_magnitude + half, 9)
    if not 0 < lo <= hi < 1:
        raise ValueError(f"hit magnitude range [{lo}, {hi}] must lie in (0, 1)")
    r = Fraction(rate).limit_denominator(10_000)
    if r <= 0:
        raise ValueError(f"shock rate must be positive, got {rate!r}")

    def scaled(base: tuple[int, int]) -> tuple[int, int]:
        f = r * Fraction(*base)
        return (f.numerator, f.denominator)

    window, resilience = SHOCKS_I_RESILIENCE
    return ShockProfile(
        name=name,
        kinds=(KindProfile(ShockKind.CAPACITY_HIT, scaled(SHOCKS_I_HIT_RATE), start_window,
                           (1, 1), (lo, hi)),
               KindProfile(ShockKind.PRODUCER_OUTAGE, scaled(SHOCKS_I_OUTAGE_RATE), start_window,
                           (outage_duration, outage_duration + SHOCKS_I_OUTAGE_SPAN), (1.0, 1.0))),
        drift_j=SHOCKS_I_DRIFT_J, resilience=ResilienceParams(window=window, tolerance=resilience))


def build_family_config(
    seed: int,
    *,
    family: str,
    producers: int,
    regime: str,
    fanout: int | None = None,
    schedule_length: int = DEFAULT_SCHEDULE_LENGTH,
    capacity_scale: float = NEUTRAL_CAPACITY_SCALE,
    shock_profile: str | None = None,
    consumer_multiplier: float = NEUTRAL_MULTIPLIER,
    quantity_multiplier: float = NEUTRAL_MULTIPLIER,
    h_cap: float = NEUTRAL_SPREAD,
    h_deg: float = NEUTRAL_SPREAD,
    h_cost: float = NEUTRAL_SPREAD,
    pool_width: int | None = None,
    shocks: ShockProfile | None = None,
) -> Configuration:
    """One family world (module docstring). `seed` is the world seed. `capacity_scale` is
    explicit (neutral by default) so a generator ref alone fixes the world; the family default
    enters through the registry generator, not here. `shock_profile` (shocked regime only,
    add-shocks-i-drift D7) names a profile in `eval/registry/shock_profiles_v1.json` that adds
    seeded shocks and drift after every other value is drawn; None leaves the world exactly as
    before. The keywords after it are the world-space levers (module docstring, ADR-0057), each
    neutral by default; `shocks` is a profile object whose generators are added in either
    regime, without its drift, and excludes `shock_profile`."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; known: {REGIMES}")
    if producers < DEPTH or producers % DEPTH:
        raise ValueError(f"producers must be a positive multiple of {DEPTH}, got {producers}")
    if schedule_length < 1:
        raise ValueError(f"schedule_length must be >= 1, got {schedule_length}")
    if shock_profile is not None and regime != SHOCKED:
        raise ValueError(f"a {regime} world carries no shock profile, got {shock_profile!r}")
    if shock_profile is not None and shocks is not None:
        raise ValueError("pass a named shock_profile or a shocks object, not both")
    if pool_width is not None and (isinstance(pool_width, bool) or not isinstance(pool_width, int)
                                   or pool_width < 1):
        raise ValueError(f"pool_width must be an integer >= 1, got {pool_width!r}")
    scale = check_capacity_scale(capacity_scale)
    k_c = check_multiplier("consumer_multiplier", consumer_multiplier)
    m_q = check_multiplier("quantity_multiplier", quantity_multiplier)
    if m_q < 1:
        raise ValueError(f"quantity_multiplier must be >= 1, got {quantity_multiplier!r}")
    spreads = [check_spread(n, v) for n, v in (("h_cap", h_cap), ("h_deg", h_deg),
                                                ("h_cost", h_cost))]
    fam = FAMILIES[family]
    dense = family == DENSE
    n = producers // DEPTH
    W = fam.width(n)
    f = fam.fanout(n, fanout)
    strong_every = min(f, STRONG_EVERY_MAX)
    stride = _stride_for_width(W)
    rng = random.Random(seed)

    # --- recipes: index-aligned chains, same draws as build_scale_config ---
    sheets: list[ProcessSheet] = []
    for level in range(DEPTH):
        cap_type = f"capacity-{level}"
        for j in range(W):
            require = {cap_type: rng.randint(2, 6) if level else rng.randint(2, 4)}
            if level == 0:
                sheets.append(ProcessSheet(product=_pid(0, j, stride), require=require))
            else:
                sheets.append(ProcessSheet(product=_pid(level, j, stride), require=require,
                                           inputs={_pid(level - 1, j, stride): 1}))

    # --- producers: n per layer, declaring per the family rule ---
    configs: list[ProducingAgentConfig] = []
    for level in range(DEPTH):
        cap_type = f"capacity-{level}"
        for k in range(n):
            strong = (k % strong_every == 0) or rng.random() < 0.6
            cap = _scaled(rng.randint(80, 140) if strong else rng.randint(8, 20), scale)
            maint = _pid(0, (level * n + k) % W, stride)
            common = dict(
                id=f"p{level}_{k}",
                capacities={cap_type: cap},
                degradation=rng.randint(3, 7),
                restoration=Restoration(require=maint, restores=_scaled(rng.randint(40, 70), scale),
                                        type=cap_type, cost=rng.randint(20, 50)),
                upgrade=Upgrade(require=maint, increases=_scaled(rng.randint(30, 50), scale),
                                type=cap_type, cost=rng.randint(20, 50)),
                produces=frozenset(_pid(level, j, stride) for j in _declared(k, W, f, dense)),
            )
            if level == 0:
                configs.append(ProducingAgentConfig(**common))  # primal: no inventory
            else:
                configs.append(ProducingAgentConfig(
                    **common,
                    inventory_limit=rng.randint(20, 40),
                    max_replenish=rng.randint(8, 14),
                    replenish_cost_per_unit=rng.randint(2, 6),
                    inventory_expansion=InventoryExpansion(
                        require=maint, increases=rng.randint(15, 25), cost=rng.randint(20, 40)),
                ))

    # --- consumers: B9 bounds; calm adds a schedule drawn last from the same stream ---
    top = DEPTH - 1
    C = num_consumers(producers)
    bounds = []
    for i in range(C):
        pool = tuple(_pid(top, j, stride) for j in _pool(i, C, W, dense))
        bounds.append((pool, rng.randint(2, 3)))
    consumers = []
    for i, (pool, qmax) in enumerate(bounds):
        schedule: tuple[tuple[int, int], ...] = ()
        if regime == CALM:
            schedule = tuple((rng.choice(pool), rng.randint(1, qmax))
                             for _ in range(schedule_length))
        consumers.append(ConsumerConfig(id=f"c{i}", product=pool[0], products=pool,
                                        quantity_min=1, quantity_max=qmax, schedule=schedule))

    # --- world-space levers: after every draw above (module docstring) ---
    if spreads != [NEUTRAL_SPREAD] * 3:
        configs = _apply_heterogeneity(configs, n, *spreads)
    target = consumers_for(producers, k_c)
    if target < C:
        consumers = consumers[:target]
    for i in range(C, target):
        extra = random.Random(_lever_seed(seed, "extra-consumer", i))
        pool = tuple(_pid(top, j, stride) for j in _pool(i % C, C, W, dense))
        qmax = extra.randint(2, 3)
        schedule = ()
        if regime == CALM:
            schedule = tuple((extra.choice(pool), extra.randint(1, qmax))
                             for _ in range(schedule_length))
        consumers.append(ConsumerConfig(id=f"c{i}", product=pool[0], products=pool,
                                        quantity_min=1, quantity_max=qmax, schedule=schedule))
    if pool_width is not None:
        consumers = [_widened(c, pool_width, W, stride) for c in consumers]
    if m_q != NEUTRAL_MULTIPLIER:
        consumers = [_multiplied(c, m_q) for c in consumers]

    cfg = Configuration(cycle_emission=cycle_emission(producers), process_sheets=sheets,
                        producer_configs=configs, consumers=consumers)
    cfg.validate()
    if shock_profile is not None:
        from .shock_profiles import apply_shock_profile, load_shock_profile

        cfg = apply_shock_profile(cfg, load_shock_profile(shock_profile), producers)
    if shocks is not None:
        cfg = replace(cfg, shocks=ShockModel(
            generators=tuple(k.generator(producers) for k in shocks.kinds)))
        cfg.validate()
    return cfg


def _widened(c: ConsumerConfig, width: int, layer_width: int, stride: int) -> ConsumerConfig:
    """A B9 consumer's pool as the window of max(len, `width`) consecutive top products from
    its start (capped at the layer); a scheduled consumer is never patterned, so unchanged."""
    if c.schedule:
        return c
    top = DEPTH - 1
    index = {_pid(top, j, stride): j for j in range(layer_width)}
    start = index[c.products[0]]
    length = min(layer_width, max(len(c.products), width))
    pool = tuple(_pid(top, (start + t) % layer_width, stride) for t in range(length))
    return replace(c, product=pool[0], products=pool)


def _multiplied(c: ConsumerConfig, m_q: float) -> ConsumerConfig:
    """Quantity range and schedule quantities × m_q, rounded half up (module docstring)."""
    return replace(c, quantity_min=_round_half_up(m_q * c.quantity_min),
                   quantity_max=_round_half_up(m_q * c.quantity_max),
                   schedule=tuple((p, _round_half_up(m_q * q)) for p, q in c.schedule))


def build_dense_config(seed: int, *, producers: int, regime: str,
                       fanout: int | None = None,
                       schedule_length: int = DEFAULT_SCHEDULE_LENGTH,
                       capacity_scale: float = NEUTRAL_CAPACITY_SCALE,
                       shock_profile: str | None = None) -> Configuration:
    """Generator-ref target for the dense family (`task_archive.REPRODUCERS`)."""
    return build_family_config(seed, family=DENSE, producers=producers, regime=regime,
                               fanout=fanout, schedule_length=schedule_length,
                               capacity_scale=capacity_scale, shock_profile=shock_profile)


def build_sparse_wide_config(seed: int, *, producers: int, regime: str,
                             fanout: int = DEFAULT_SPARSE_FANOUT,
                             schedule_length: int = DEFAULT_SCHEDULE_LENGTH,
                             capacity_scale: float = NEUTRAL_CAPACITY_SCALE,
                             shock_profile: str | None = None) -> Configuration:
    """Generator-ref target for the sparse-wide family (`task_archive.REPRODUCERS`)."""
    return build_family_config(seed, family=SPARSE_WIDE, producers=producers, regime=regime,
                               fanout=fanout, schedule_length=schedule_length,
                               capacity_scale=capacity_scale, shock_profile=shock_profile)


# --- economy ratios (gate-harness "Constant economy ratios across sizes") ---


def top_layer_throughput(cfg: Configuration) -> float:
    """Initial top-layer throughput: top-layer product units per cycle the top layer could
    assemble at its initial capacity with inputs free, each producer splitting its capacity
    evenly over its declared products. Sum over top-layer producers of
    `capacity / mean recipe requirement of its declared products`. The top layer is the layer of
    the consumers' products; its producers are those declaring any of them."""
    top_products = {p for c in cfg.consumers for p in c.product_pool}
    total = 0.0
    for p in cfg.producer_configs:
        declared = [x for x in p.produces if x in top_products]
        if not declared:
            continue
        req = sum(sum(cfg.process_sheet(x).require.values()) for x in declared) / len(declared)
        total += sum(p.capacities.values()) / req
    return total


def demand_units_per_cycle(cfg: Configuration) -> float:
    """Expected consumer units ordered per cycle: a schedule's mean quantity (calm), else the
    midpoint of the B9 quantity range (shocked)."""
    total = 0.0
    for c in cfg.consumers:
        if c.schedule:
            total += sum(q for _, q in c.schedule) / len(c.schedule)
        else:
            total += (c.quantity_min + c.quantity_max) / 2
    return total


def demand_throughput_ratio(cfgs: list[Configuration]) -> float:
    """Family-level demand per unit of initial top-layer throughput over a sample of worlds:
    total demand / total throughput (ratio of sums). A single 12-producer world has only 4 top
    producers, so its own ratio swings with the strong/marginal draw; the family law is what the
    builder fixes, and the ratio of sums estimates it without the 1/T bias of a mean of ratios."""
    return sum(map(demand_units_per_cycle, cfgs)) / sum(map(top_layer_throughput, cfgs))


def family_demand_ratio(family: str, producers: int, regime: str,
                        fanout: int | None = None, capacity_scale: float | None = None,
                        shock_profile: str | None = None) -> float:
    """The family's demand ratio at one size, under its recorded sampling protocol and (unless
    overridden) its recorded capacity scale; `shock_profile` builds the shocked worlds with it."""
    fam = FAMILIES[family]
    scale = fam.capacity_scale if capacity_scale is None else capacity_scale
    return demand_throughput_ratio([
        build_family_config(s, family=family, producers=producers, regime=regime, fanout=fanout,
                            capacity_scale=scale, shock_profile=shock_profile)
        for s in fam.sample_seeds(producers)
    ])


def emission_per_producer(cfg: Configuration) -> float:
    return cfg.cycle_emission / len(cfg.producer_configs)
