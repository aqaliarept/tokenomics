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

Input-output tier B_lp (task 4.7, design D12 "Bound"). One linear program over every producer,
capacity type and product, in the spirit of a dynamic Leontief model: the guaranteed bound that
sees capacity, which B2 (a demand cap) and B_base (not guaranteed) do not. Each relaxation keeps it
an upper bound on the consumer units of every policy (soundness argument in `lp_bound`):
- per-producer `produces` eligibility and per-producer capacity: no pooling across non-declarers;
- one product mix across all capacity types (not a per-type minimum);
- recipes exactly, cycles included: a product balance per product (inputs, restoration and upgrade
  products are consumed like any other demand);
- capacity per producer x type x cycle, degrading geometrically at the configured rate and
  raised only by restorations and upgrades, each at most one completion per producer per cycle and
  each costing one unit of its product through that product's recipe (upgrades are a growth bound,
  not free slack);
- consumer service in the TTL window: an order placed in cycle t is served only by production in
  the auctions of t..t + ORDER_TTL, out of that cycle's capacity;
- other product flows (inputs, investment products) aggregated over the horizon: a product may be
  used before it is made; lead time, inventory limits, funding, routing and shocks are ignored.
A time-expanded flow for inputs too would multiply the program by the horizon again, and on 24
sampled EXP-0051 worlds the consumer window itself moved B_lp on one world, by 0.1% (task 4.7),
so it is not added.
`upgrades=False` gives B_lp_base: capacity-aware, but NOT guaranteed (an upgrading policy can
exceed it). Needs scipy (the `lp` extra, HiGHS).

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
from ..domain.order import ORDER_TTL

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

DEMAND = "demand"
B_BASE = "B_base"  # B2 without upgrade slack: the base-capacity ceiling (not guaranteed)
B_LP = "B_lp"  # the input-output LP: guaranteed and capacity-aware (task 4.7)
B_LP_BASE = "B_lp_base"  # B_lp without upgrades: capacity-aware, NOT guaranteed
LP_CAPACITY = "lp"  # B_lp's binding label when the LP sits below demand


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
    """The two analytic bounds the atlas reads per world-episode (D12): `B2`, the guaranteed
    bound (with upgrade slack; on family worlds a demand cap), and `B_base`, B2 without upgrade
    slack (the base-capacity ceiling; not guaranteed). These are inputs, not the ceiling: room is
    read against C = max(min(B2, B_lp, B_lp_base), best played arm) where the LP ran (task 4.7),
    else max(min(B2, B_base), best played arm), which needs the episode's arm units."""
    base = ceiling_bounds(cfg, horizon, demand, include_upgrade_slack=False)["B2"]
    return {"B2": ceiling_bounds(cfg, horizon, demand)["B2"],
            B_BASE: Bound(B_BASE, base.units, base.binding_constraint)}


# --- input-output LP (task 4.7) ---


CycleDemand = Mapping[Product, Sequence[int]]  # product -> units ordered per placement cycle


def schedule_cycle_demand(schedules: Mapping[ConsumerId, Sequence[ScheduleEntry]],
                          horizon: int) -> dict[Product, list[int]]:
    """`schedule_demand` per placement cycle: units of each product ordered in cycles
    0..horizon-1 (the same replay, so it sums to `schedule_demand`)."""
    out: dict[Product, list[int]] = {}
    for schedule in schedules.values():
        for t in range(horizon):
            product, quantity = schedule[t % len(schedule)]
            out.setdefault(product, [0] * horizon)[t] += int(quantity)
    return out


def env_cycle_demand(env: "TokenomicsAECEnv") -> dict[Product, list[int]]:
    """Per-cycle demand of the episode `env` was last reset for (B_lp's input)."""
    return schedule_cycle_demand(env._demand_schedules, env.horizon)


def episode_cycle_demand(cfg: Configuration, horizon: int, seed: int) -> dict[Product, list[int]]:
    """`env_cycle_demand` of a fresh env reset for episode `seed`."""
    from ..env import TokenomicsAECEnv

    env = TokenomicsAECEnv(config=cfg, horizon=horizon)
    env.reset(seed=seed)
    return env_cycle_demand(env)


@dataclass(frozen=True)
class LpSize:
    """Shape of one B_lp program (sizing and tests)."""

    variables: int
    constraints: int
    nonzeros: int


@dataclass
class _Program:
    cost: list[float]
    bounds: list[tuple[float, float | None]]
    ub: list[tuple[int, int, float]]
    n_ub: int
    rhs: dict[int, float]  # nonzero A_ub right-hand sides
    eq: list[tuple[int, int, float]]
    n_eq: int
    total: float


def _lp_program(cfg: Configuration, horizon: int, demand: CycleDemand, upgrades: bool,
                window: int | None) -> _Program:
    """B_lp's columns and rows (`lp_bound`). Consumer service: with `window` = w, an order
    placed in cycle t is served by consumer production in cycles t..t+w (time-expanded,
    per-cycle capacity); with None, deliveries are capped by total demand only (aggregated)."""
    cost: list[float] = []
    bounds: list[tuple[float, float | None]] = []
    ub: list[tuple[int, int, float]] = []
    eq: list[tuple[int, int, float]] = []
    ub_rhs: dict[int, float] = {}
    n = {"ub": 0, "eq": 0}

    def col(c: float, lo: float, hi: float | None) -> int:
        cost.append(c)
        bounds.append((lo, hi))
        return len(cost) - 1

    def row(kind: str = "ub") -> int:
        n[kind] += 1
        return n[kind] - 1

    sheets = {sh.product: sh for sh in cfg.process_sheets}
    want = {j: [max(0, int(q)) for q in qs[:horizon]] for j, qs in demand.items()
            if j in sheets and sum(qs[:horizon]) > 0}
    total = float(sum(sum(qs) for qs in want.values()))
    balance = {j: row() for j in sheets}  # uses - production <= 0, over the horizon
    serve: dict[tuple[Product, int], int] = {}  # (j, s): consumer service from cycle-s output
    for j, qs in sorted(want.items()):
        if window is None:
            ub.append((balance[j], col(-1.0, 0.0, float(sum(qs))), 1.0))
            continue
        for s in range(horizon):
            serve[(j, s)] = row()
        for t, q in enumerate(qs):
            if q <= 0:
                continue
            cap = row()  # served units of cycle-t orders <= ordered
            for s in range(t, min(horizon, t + window + 1)):
                w = col(-1.0, 0.0, None)
                ub += [(cap, w, 1.0), (serve[(j, s)], w, 1.0), (balance[j], w, 1.0)]
            ub_rhs[cap] = float(q)
    late = horizon - 1  # completions at the end of the last cycle cannot be used
    for p in cfg.producer_configs:
        keep = 1.0 - p.degradation / 100.0  # ceil(d M / 100) >= (d / 100) x current
        cap_row = {k: row() for k in p.capacities}  # all production <= sum of a over cycles
        rest, upg = p.restoration, p.upgrade
        a: dict[CapacityType, list[int]] = {}
        r = [col(0.0, 0.0, 1.0) for _ in range(late)] if rest.restores > 0 else []
        u = [col(0.0, 0.0, 1.0) for _ in range(late)] if (
            upgrades and upg.increases > 0 and upg.type in p.capacities) else []
        m: list[int] = []
        if u:
            top = float(p.capacities[upg.type])
            m = [col(0.0, top, top)] + [col(0.0, top, None) for _ in range(late)]
            for t in range(late):
                e = row("eq")
                eq += [(e, m[t + 1], 1.0), (e, m[t], -1.0), (e, u[t], -float(upg.increases))]
        for k, c0 in p.capacities.items():
            grows = bool(m) and k == upg.type
            a[k] = [col(0.0, 0.0, float(c0))] + [
                col(0.0, 0.0, None if grows else float(c0)) for _ in range(late)]
            for t in range(horizon):
                ub.append((cap_row[k], a[k][t], -1.0))
                if grows:
                    g = row()
                    ub += [(g, a[k][t], 1.0), (g, m[t], -1.0)]
            for t in range(late):
                d = row()
                ub += [(d, a[k][t + 1], 1.0), (d, a[k][t], -keep)]
                if r and k == rest.type:
                    ub.append((d, r[t], -float(rest.restores)))
                if u and k == upg.type:
                    ub.append((d, u[t], -float(upg.increases)))
        for v in r:
            ub.append((balance[rest.require], v, 1.0))
        for v in u:
            ub.append((balance[upg.require], v, 1.0))
        per_cycle = {(k, t): row() for k in p.capacities for t in range(horizon)} if (
            window is not None) else {}
        for k, t in per_cycle:  # consumer production in cycle t <= a_pkt
            ub.append((per_cycle[(k, t)], a[k][t], -1.0))
        for j in sorted(p.produces):
            sheet = sheets[j]
            outputs = [col(0.0, 0.0, None)]  # y: production for any use, aggregated
            if window is not None and j in want:
                outputs += [col(0.0, 0.0, None) for _ in range(horizon)]  # f_pjs
            for idx, y in enumerate(outputs):
                ub.append((balance[j], y, -1.0))
                for k, need in sheet.require.items():
                    ub.append((cap_row[k], y, float(need)))
                    if idx:
                        ub.append((per_cycle[(k, idx - 1)], y, float(need)))
                for i, qty in sheet.inputs.items():
                    ub.append((balance[i], y, float(qty)))
                if idx:
                    ub.append((serve[(j, idx - 1)], y, -1.0))
    return _Program(cost, bounds, ub, n["ub"], ub_rhs, eq, n["eq"], total)


def lp_available() -> bool:
    """Whether B_lp can be solved here (scipy, the `lp` extra)."""
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        return False
    return True


def lp_size(cfg: Configuration, horizon: int, demand: CycleDemand, *, upgrades: bool = True,
            window: int | None = ORDER_TTL) -> LpSize:
    pr = _lp_program(cfg, horizon, demand, upgrades, window)
    return LpSize(len(pr.cost), pr.n_ub + pr.n_eq, len(pr.ub) + len(pr.eq))


def lp_bound(cfg: Configuration, horizon: int, demand: CycleDemand, *, upgrades: bool = True,
             window: int | None = ORDER_TTL) -> Bound:
    """B_lp: the most consumer units over `horizon` cycles the input-output LP allows for this
    episode's per-cycle `demand` (module docstring). Sound because every played trajectory is a
    feasible point. With its own deliveries, productions, completed investments and capacity at
    each auction: (i) an order placed in cycle t completes only by the auction of cycle
    t + ORDER_TTL (domain `Order.complete_cycle`) and its units were accepted, so their
    capacity was allocated, in one of those auctions; (ii) every unit delivered, used as an input
    or consumed by an investment was produced by a declarer (aggregated balance, no initial
    stock); (iii) a producer's capacity use in a cycle is at most its capacity at that auction,
    and over the episode at most their sum; (iv) the capacity at the next auction is at most
    min(max, max(0, a - ceil(d M / 100)) + restored + upgraded) <= (1 - d / 100) a + restored +
    upgraded, because a <= M; capacity hits and outages only lower it; (v) restorations and
    upgrades each run one at a time, so at most one completes per producer per cycle, and one
    completed at the end of the last cycle is never used. `upgrades=False` drops (v)'s upgrades:
    B_lp_base, capacity-aware but NOT guaranteed for a policy that upgrades. `window=None`
    serves orders from production in any cycle (aggregated; looser). Raises if scipy is missing
    or HiGHS does not report an optimum."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("B_lp needs scipy: pip install -e .[lp]") from exc
    pr = _lp_program(cfg, horizon, demand, upgrades, window)
    tier = B_LP if upgrades else B_LP_BASE
    if pr.total <= 0:
        return Bound(tier, 0.0, DEMAND)

    def mat(trip: list[tuple[int, int, float]], n_rows: int) -> object:
        rr, cc, vv = zip(*trip)
        return coo_matrix((vv, (rr, cc)), shape=(n_rows, len(pr.cost))).tocsr()

    b_ub = [0.0] * pr.n_ub
    for i, v in pr.rhs.items():
        b_ub[i] = v
    res = linprog(pr.cost, A_ub=mat(pr.ub, pr.n_ub), b_ub=b_ub,
                  A_eq=mat(pr.eq, pr.n_eq) if pr.n_eq else None,
                  b_eq=[0.0] * pr.n_eq if pr.n_eq else None, bounds=pr.bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"{tier}: HiGHS did not solve ({res.status}: {res.message})")
    units = min(pr.total, -float(res.fun))
    return Bound(tier, units, DEMAND if units >= pr.total - 1e-6 else LP_CAPACITY)
