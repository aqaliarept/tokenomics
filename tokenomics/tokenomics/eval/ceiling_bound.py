"""Offline achievable-units upper bounds per world-episode (add-m3-development-audit task 4.2,
design D12 "Bound and ceiling"; development-audit "Upper bound stays above every played arm",
"Room is read against base capacity").

Restored from commit b96ecc0 (the parent of cleanup 18c717d, an ancestor of branch `archive-1`),
where EXP-0033 used it. Stated reason for the restore: the headroom atlas needs a cheap per-world
ceiling to tell a saturated world from a world with room, and EXP-0033 showed the bound sound
(bound >= every observed arm on all 26 archived tasks). Ported, not restored: the EXP-0033
archive/curriculum CLI and its `eval/curriculum.py` imports are gone, and demand now comes from
the schedule the env built for the episode seed (`TokenomicsAECEnv._demand_schedules`), so
explicit schedules, drift and demand patterns are covered instead of a pre-M2 mirror.

Tiers, in consumer-completed units over `horizon` cycles, each optimistic:
- B0: the scheduled consumer demand;
- B1: B0 capped by a direct recipe capacity-throughput cap;
- B2: B1 with each demanded product's recipe expanded transitively through its inputs.
Capacity is aggregated over producers and pinned at each producer's initial maximum every cycle
(degradation ignored); TTL, inventory, funding and routing are ignored.

Upgrade slack. With `include_upgrade_slack=True` (EXP-0033's setting) upgrades are free and
instant, at most one per producer per cycle, adding `upgrade.increases * horizon**2` capacity
units per producer. That keeps B2 a guaranteed bound for policies that upgrade, but on family
worlds the slack is 33-74x the base capacity per type (world seed 0 of sparse 12 f2, sparse 42
f7, dense 12, horizon 100), so B2's capacity term essentially never binds there: **B2 is a demand
cap** (EXP-0033's "98.3% of the bound" was 98.3% of exact demand). With the slack off the same
tiers use base capacity only: `B_base` (B2 without slack) is the base-capacity ceiling. It is NOT
guaranteed, since an upgrading policy can exceed it, but it sees capacity. The atlas (D12) reads
room and saturation against the ceiling C = max(min(B2, B_base), best played arm), so C never sits
below what was delivered, and counts every arm that exceeds B_base.

Pure: no RNG, IO or env state beyond reading a built demand schedule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..domain.common import CapacityType, ConsumerId, Product
from ..domain.config import Configuration
from ..domain.consumer import ScheduleEntry

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

DEMAND = "demand"
B_BASE = "B_base"  # B2 without upgrade slack: the base-capacity ceiling (not guaranteed)


@dataclass(frozen=True)
class Bound:
    """One upper-bound tier in consumer-completed units; `binding_constraint` is `demand` or
    `capacity:<type>`."""

    tier: str
    units: float
    binding_constraint: str


# --- demand ---


def schedule_demand(schedules: Mapping[ConsumerId, Sequence[ScheduleEntry]],
                    horizon: int) -> dict[Product, int]:
    """Units each product is ordered over `horizon` cycles when every consumer replays its
    schedule as `ScheduledConsumer` does (one entry per cycle, wrapping when exhausted). An upper
    bound on what is ordered: a consumer with no balance skips a cycle."""
    demand: dict[Product, int] = {}
    for schedule in schedules.values():
        for t in range(horizon):
            product, quantity = schedule[t % len(schedule)]
            demand[product] = demand.get(product, 0) + int(quantity)
    return demand


def env_demand(env: "TokenomicsAECEnv") -> dict[Product, int]:
    """Demand of the episode `env` was last reset for, from the schedules it built (the
    `demand_digest` input), over its horizon."""
    return schedule_demand(env._demand_schedules, env.horizon)


def episode_demand(cfg: Configuration, horizon: int, seed: int) -> dict[Product, int]:
    """Demand of `cfg`'s episode `seed` over `horizon` cycles, read from a fresh env's reset."""
    from ..env import TokenomicsAECEnv

    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=seed)
    return env_demand(env)


# --- capacity ---


def capacity_cycles_by_type(
    cfg: Configuration,
    horizon: int,
    *,
    include_upgrade_slack: bool = True,
) -> dict[CapacityType, float]:
    """Optimistic aggregate capacity-cycles available by type.

    Base capacity is pinned at each producer's initial max every cycle.  When
    ``include_upgrade_slack`` is true, upgrades are relaxed to be free and instant, with
    at most one upgrade per producer per cycle.  This is deliberately loose, but it keeps
    the tier an upper bound even for policies that expand max capacity during the run.
    """
    caps: dict[CapacityType, float] = {}
    for producer in cfg.producer_configs:
        for cap_type, cap in producer.capacities.items():
            caps[cap_type] = caps.get(cap_type, 0.0) + float(cap) * horizon
        if include_upgrade_slack and producer.upgrade.type in producer.capacities:
            caps[producer.upgrade.type] = caps.get(producer.upgrade.type, 0.0) + (
                float(producer.upgrade.increases) * horizon * horizon
            )
    return caps


def upgrade_slack_ratio(cfg: Configuration, horizon: int) -> dict[CapacityType, float]:
    """Per capacity type, upgrade slack / base capacity-cycles: how much the free-upgrade
    relaxation adds over base capacity (module docstring)."""
    base = capacity_cycles_by_type(cfg, horizon, include_upgrade_slack=False)
    full = capacity_cycles_by_type(cfg, horizon, include_upgrade_slack=True)
    return {t: (full[t] - base[t]) / base[t] for t in base if base[t] > 0}


def recipe_capacity_bundle(
    cfg: Configuration,
    product: Product,
    *,
    transitive: bool,
) -> dict[CapacityType, float]:
    """Capacity required to deliver one unit of ``product`` under a relaxed recipe model.

    Direct mode returns only the product's own recipe requirements (B1).  Transitive mode
    also expands input products and charges their capacity bundles (B2).  If a cyclic
    recipe is encountered, the repeated edge is ignored; that is an optimistic relaxation,
    so it stays on the upper-bound side.
    """
    visiting: set[Product] = set()

    def expand(p: Product) -> dict[CapacityType, float]:
        if p in visiting:
            return {}
        visiting.add(p)
        sheet = cfg.process_sheet(p)
        bundle = {t: float(v) for t, v in sheet.require.items()}
        if transitive:
            for input_product, qty in sheet.inputs.items():
                for cap_type, amount in expand(input_product).items():
                    bundle[cap_type] = bundle.get(cap_type, 0.0) + amount * qty
        visiting.discard(p)
        return bundle

    return expand(product)


def _type_unit_bound(
    available: float,
    costs_by_product: dict[Product, float],
    demand: dict[Product, int],
) -> float:
    """Max delivered units allowed by one capacity type under fixed product demand.

    This is a fractional knapsack relaxation: serve zero/cheap-cost demanded products
    first.  It can overstate the true mix-constrained optimum, but not understate it.
    """
    delivered = 0.0
    remaining = available
    for product, cost in sorted(costs_by_product.items(), key=lambda item: item[1]):
        qty = float(demand.get(product, 0))
        if qty <= 0.0:
            continue
        if cost <= 0.0:
            delivered += qty
            continue
        take = min(qty, remaining / cost)
        delivered += take
        remaining -= take * cost
        if remaining <= 1e-12:
            break
    return delivered


def _capacity_bound_for_demand(
    cfg: Configuration,
    horizon: int,
    *,
    transitive: bool,
    demand: dict[Product, int],
    include_upgrade_slack: bool = True,
) -> Bound:
    caps = capacity_cycles_by_type(cfg, horizon, include_upgrade_slack=include_upgrade_slack)
    products = set(demand)
    bundles = {
        product: recipe_capacity_bundle(cfg, product, transitive=transitive)
        for product in products
    }
    demand_cap = float(sum(demand.values()))

    best_units = math.inf
    binding = "capacity:none"
    for cap_type, available in caps.items():
        costs = {product: bundle.get(cap_type, 0.0) for product, bundle in bundles.items()}
        positive = [c for c in costs.values() if c > 0.0]
        if not positive:
            continue
        units = _type_unit_bound(available, costs, demand)
        if units < best_units:
            best_units = units
            binding = f"capacity:{cap_type}"

    if demand_cap <= best_units:
        return Bound("B2" if transitive else "B1", demand_cap, DEMAND)
    return Bound("B2" if transitive else "B1", best_units, binding)


# --- tiers ---


def ceiling_bounds(
    cfg: Configuration,
    horizon: int,
    demand: Mapping[Product, int],
    *,
    include_upgrade_slack: bool = True,
) -> dict[str, Bound]:
    """B0/B1/B2 for one episode's `demand` (from `schedule_demand` / `env_demand`) over
    `horizon` cycles. With the default upgrade slack every tier is a guaranteed upper bound on
    consumer-completed units; without it the tiers count base capacity only (module docstring)."""
    d = {p: int(q) for p, q in demand.items()}
    return {
        "B0": Bound("B0", float(sum(d.values())), DEMAND),
        "B1": _capacity_bound_for_demand(cfg, horizon, transitive=False, demand=d,
                                         include_upgrade_slack=include_upgrade_slack),
        "B2": _capacity_bound_for_demand(cfg, horizon, transitive=True, demand=d,
                                         include_upgrade_slack=include_upgrade_slack),
    }


def world_ceiling(cfg: Configuration, horizon: int,
                  demand: Mapping[Product, int]) -> dict[str, Bound]:
    """The two bounds the atlas reads per world-episode (D12): `B2`, the guaranteed bound (with
    upgrade slack; on family worlds a demand cap), and `B_base`, B2 without upgrade slack (the
    base-capacity ceiling; not guaranteed). These are the inputs, not the ceiling: room is read
    against C = max(min(B2, B_base), best played arm), which needs the episode's arm units."""
    base = ceiling_bounds(cfg, horizon, demand, include_upgrade_slack=False)["B2"]
    return {"B2": ceiling_bounds(cfg, horizon, demand)["B2"],
            B_BASE: Bound(B_BASE, base.units, base.binding_constraint)}
