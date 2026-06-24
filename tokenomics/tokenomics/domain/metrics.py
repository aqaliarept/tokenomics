"""Pure snapshot-derived economy metrics (EXP-0006A, no persistent state).

The authoritative current-state formulas behind the enriched orderer observation
(ALPHAZERO_EXPERT_ITERATION_APPROACH.md, "Feature calculation guide"). Kept in the
domain so they are testable outside RL and reusable by white-box
diagnostics/search; the env layer only normalizes them. Every function is a pure
read of a `ProducingAgent` snapshot and its process sheets — no history, no IO.
"""

from __future__ import annotations

from collections.abc import Iterable

from .common import Product
from .config import ProcessSheet
from .producing_agent import ProducingAgent


def producer_total_current_capacity(p: ProducingAgent) -> int:
    """sum_t C[p,t] — total live capacity across owned types."""
    return sum(p.capacities.values())


def producer_total_max_capacity(p: ProducingAgent) -> int:
    """sum_t M[p,t]."""
    return sum(p.max_capacities.values())


def producer_capacity_headroom(p: ProducingAgent) -> int:
    """sum_t max(0, M[p,t] - C[p,t]) — restorable capacity gap, never negative."""
    return sum(max(0, p.max_capacities[t] - p.capacities[t]) for t in p.capacities)


def _capacity_ratios(p: ProducingAgent) -> list[float]:
    """capacity_ratio[p,t] = C[p,t] / max(1, M[p,t]) over owned types."""
    return [p.capacities[t] / max(1, p.max_capacities[t]) for t in p.capacities]


def producer_min_capacity_ratio(p: ProducingAgent) -> float:
    """min_t capacity_ratio — the worst-degraded owned type (0.0 if none owned)."""
    ratios = _capacity_ratios(p)
    return min(ratios) if ratios else 0.0


def producer_mean_capacity_ratio(p: ProducingAgent) -> float:
    """mean_t capacity_ratio over owned types (0.0 if none owned)."""
    ratios = _capacity_ratios(p)
    return sum(ratios) / len(ratios) if ratios else 0.0


def producer_requested_load(p: ProducingAgent) -> int:
    """producer[p].requested_capacity — capacity bid for last cycle."""
    return p.requested_capacity


def producer_reserved_load(p: ProducingAgent) -> int:
    """sum over live reservations of bid.total_capacity — capacity currently held
    by composites waiting for their inputs (unavailable to new bids)."""
    return sum(r.bid.total_capacity() for r in p.reservations)


def producer_total_load(p: ProducingAgent) -> int:
    """requested_load + reserved_load."""
    return producer_requested_load(p) + producer_reserved_load(p)


def producer_outstanding_inbound_units(p: ProducingAgent) -> int:
    """sum_g outstanding_inbound[p,g] — replenishment units ordered but not yet
    delivered (in-flight inbound stock)."""
    return p.outstanding_inbound_units()


def producer_active_investments(p: ProducingAgent) -> tuple[float, float, float]:
    """one_hot(restoration_running, upgrade_running, inventory_expansion_running) —
    which investment streams are in flight (each gated to one at a time)."""
    return (
        1.0 if p.restoration_running else 0.0,
        1.0 if p.upgrade_running else 0.0,
        1.0 if p.inventory_expansion_running else 0.0,
    )


# --- producer-product makeability (eligibility is authoritative, ADR-0028) ---
def _available_capacity(p: ProducingAgent) -> dict[str, int]:
    """Capacity free for NEW bids, per type = current minus capacity held by live
    composite reservations. The auction subtracts reservations before accepting new
    bids (ADR-0025), so routing-decision metrics must use this, not raw current."""
    reserved = p.reserved_capacity()
    return {t: max(0, c - reserved.get(t, 0)) for t, c in p.capacities.items()}


def capacity_makeable_qty(p: ProducingAgent, x: Product) -> int:
    """0 if x not in p.produces; else min_t floor(avail[p,t] / R[x,t]) over the
    capacity types x's recipe requires — how many units can clear the auction now,
    counting only capacity not already held by live reservations."""
    if x not in p.produces:
        return 0
    require = p.sheets[x].require
    avail = _available_capacity(p)
    return min(avail.get(t, 0) // c for t, c in require.items())


def input_makeable_qty(p: ProducingAgent, x: Product) -> float:
    """inf if x is primal (no inputs); else min_g floor(Inv[p,g] / I[x,g]) over the
    input products x consumes — how many units current stock can assemble now."""
    if x not in p.produces:
        return 0
    inputs = p.sheets[x].inputs
    if not inputs:
        return float("inf")
    return min(p.inventory.get(g, 0) // n for g, n in inputs.items())


def accept_qty_now(p: ProducingAgent, x: Product) -> int:
    """How much can clear the capacity auction now (= capacity_makeable_qty), even if
    composite inputs are not in stock and the bid would become a reservation."""
    return capacity_makeable_qty(p, x)


def complete_qty_now(p: ProducingAgent, x: Product) -> int:
    """How much can complete immediately with current capacity AND input inventory =
    min(capacity_makeable, input_makeable)."""
    return int(min(capacity_makeable_qty(p, x), input_makeable_qty(p, x)))


def makeable_qty_now(p: ProducingAgent, x: Product) -> int:
    """The recommended default meaning of 'makeable now' = complete_qty_now."""
    return complete_qty_now(p, x)


def capacity_bottleneck_ratio(p: ProducingAgent, x: Product, q: int) -> float:
    """min_t avail[p,t] / max(1, q*R[x,t]) — capacity available for new bids vs the
    capacity q units of x demand. >1 means over-supplied; 0.0 if x not declared."""
    if x not in p.produces:
        return 0.0
    require = p.sheets[x].require
    avail = _available_capacity(p)
    return min(avail.get(t, 0) / max(1, q * c) for t, c in require.items())


def input_bottleneck_ratio(p: ProducingAgent, x: Product, q: int) -> float:
    """1.0 if x is primal; else min_g Inv[p,g] / max(1, q*I[x,g]) — input stock vs
    the inputs q units of x demand. 0.0 if x not declared."""
    if x not in p.produces:
        return 0.0
    inputs = p.sheets[x].inputs
    if not inputs:
        return 1.0
    return min(p.inventory.get(g, 0) / max(1, q * n) for g, n in inputs.items())


def producer_product_bottleneck_ratio(p: ProducingAgent, x: Product, q: int) -> float:
    """min(capacity_bottleneck_ratio, input_bottleneck_ratio) — the binding
    constraint on making q units of x now."""
    if x not in p.produces:
        return 0.0
    return min(
        capacity_bottleneck_ratio(p, x, q), input_bottleneck_ratio(p, x, q)
    )


def slot_makeability(p: ProducingAgent, x: Product, q: int) -> float:
    """min(q, makeable_qty_now[p,x]) / max(1, q) — the fraction of a q-unit order x
    this producer can deliver now, in [0, 1]."""
    return min(q, makeable_qty_now(p, x)) / max(1, q)


# --- static recipe-graph facts (exogenous, config-derived) ---
def product_depth(
    x: Product, sheets: dict[Product, ProcessSheet], _visiting: frozenset[Product] = frozenset()
) -> int:
    """0 for a primal product; else 1 + max product_depth over its inputs — the
    product's layer in the economy graph. Cycle-safe: the config model supports
    circular recipes (inventory breaks the dependency), so a back edge to a product
    already on the current path is cut (contributes 0) instead of recursing forever."""
    inputs = sheets[x].inputs
    if not inputs:
        return 0
    visiting = _visiting | {x}
    sub = [product_depth(g, sheets, visiting) for g in inputs if g not in visiting]
    return 1 + (max(sub) if sub else 0)


def recipe_input_count(sheet: ProcessSheet) -> int:
    """len(I[x,*]) — number of distinct input products the recipe consumes."""
    return len(sheet.inputs)


def recipe_input_units(sheet: ProcessSheet) -> int:
    """sum_g I[x,g] — total input units per unit of the product."""
    return sum(sheet.inputs.values())


def recipe_capacity_type_count(sheet: ProcessSheet) -> int:
    """len(R[x,*]) — number of distinct capacity types the recipe requires."""
    return len(sheet.require)


def eligible_producer_count(x: Product, producers: Iterable[ProducingAgent]) -> int:
    """sum_p eligible(p, x) — how many producers declare x in `produces` (ADR-0028)."""
    return sum(1 for p in producers if x in p.produces)
