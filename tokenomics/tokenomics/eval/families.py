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

Pure and deterministic in (family, producers, world seed, regime, fanout, schedule_length,
capacity_scale, shock_profile).
`build_scale_config` is not touched.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..domain.config import (Configuration, ConsumerConfig, InventoryExpansion, ProcessSheet,
                             ProducingAgentConfig, Restoration, Upgrade)
from .scale import _pid, _stride_for_width

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
) -> Configuration:
    """One family world (module docstring). `seed` is the world seed. `capacity_scale` is
    explicit (neutral by default) so a generator ref alone fixes the world; the family default
    enters through the registry generator, not here. `shock_profile` (shocked regime only,
    add-shocks-i-drift D7) names a profile in `eval/registry/shock_profiles_v1.json` that adds
    seeded shocks and drift after every other value is drawn; None leaves the world exactly as
    before."""
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
    scale = check_capacity_scale(capacity_scale)
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

    cfg = Configuration(cycle_emission=cycle_emission(producers), process_sheets=sheets,
                        producer_configs=configs, consumers=consumers)
    cfg.validate()
    if shock_profile is not None:
        from .shock_profiles import apply_shock_profile, load_shock_profile

        cfg = apply_shock_profile(cfg, load_shock_profile(shock_profile), producers)
    return cfg


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
