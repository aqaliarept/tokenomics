"""ProducerEntityView: the domain's authoritative entity-factored feature bundle for
one producing agent's decision (ADR-0053, EXP-0010A).

The flat producer obs (`encode_producing`) is indexed by arbitrary product/capacity
ids and slot positions, so producer training memorizes one topology's vocabulary. This
view replaces it with rows: one per owned capacity type, one per declared product
(`produces`), one per input product those recipes consume, plus recipe-edge relations
keyed by ROW POSITION (so the env encoder can permute rows and the relation planes
follow). The learned part later depends on row features + relations, not slot identity.

Mirrors `orderer_view.py`: pure domain (no RL, no numpy, no IO); the env layer only
normalizes it. Rows are returned as ordered lists (default: sorted id) — the order is a
property of the returned lists, and the encoder honors whatever order they carry. Per
ADR the view carries NO cycle progress: that is an env-time scalar the encoder appends.
Reservation rows and requestor-side replenishment EMAs are deferred (EXP-0010A scope).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import metrics
from .common import CapacityType, CapacityUnitPrice, OrderId, Product, ProducerId
from .config import ProcessSheet
from .order import Order
from .producing_agent import ProducingAgent
from .telemetry import EconomyTelemetry


@dataclass
class ProducerGlobalStats:
    """Producer-wide scalars (the doc's global features). No cycle progress — that is
    an env-time scalar the encoder appends last (matching the orderer views)."""

    cut_off_price: CapacityUnitPrice
    inventory_units: int
    inventory_limit: int
    inventory_headroom: int
    outstanding_inbound_units: int
    active_investments: tuple[float, float, float]  # restore/upgrade/inv-expansion running


@dataclass
class CapacityRow:
    """Per owned capacity type t (ADR-0053). `previous_requested_capacity` is ReqPrev[p,t]
    — last cycle's post-auction per-type pressure (current bids are not yet routed at the
    decision point). All quantities are raw; the env normalizes."""

    capacity_type: CapacityType
    owned: float
    current: int
    max: int
    headroom: int
    ratio: float
    degradation: int
    reserved: int
    previous_requested_capacity: int
    free: int
    reserved_ratio: float
    restore_effect: int
    restore_cost: int
    upgrade_effect: int
    upgrade_cost: int
    demand_pressure: int


@dataclass
class ProductRow:
    """Per declared product x in `p.produces` (ADR-0053). `eligible` is 1.0 on every real
    row (the universe is `produces`); kept explicit so the makeability eligible-guard
    stays visible and so padded rows read 0. Makeability/bottleneck use the product's open
    demand quantity, and are 0 for non-`produces` products (metric eligible-guard)."""

    product: Product
    eligible: float
    open_demand_qty: int
    open_demand_tokens: int
    product_depth: int
    recipe_input_count: int
    recipe_input_units: int
    recipe_capacity_type_count: int
    recent_routed_qty: float
    recent_completed_qty: float
    recent_rejected_qty: float
    recent_completion_latency: float
    active_reservation_qty: int
    capacity_makeable_qty: int
    input_makeable_qty: float
    complete_qty_now: int
    capacity_bottleneck_ratio: float
    input_bottleneck_ratio: float
    bottleneck_ratio: float


@dataclass
class InputRow:
    """Per input product g this producer's recipes consume (ADR-0053). Requestor-side
    replenishment EMAs are deferred (no telemetry source yet); only current-state facts."""

    input_product: Product
    inventory: int
    outstanding_inbound: int
    blocked_reservation_qty: int
    downstream_demand_pressure: int


@dataclass
class Relations:
    """Recipe edges keyed by ROW POSITION (not raw id), so they stay aligned when the
    encoder permutes rows (ADR-0053). `product_capacity_required[(xi, ti)] = R[x,t]`;
    `product_input_required[(xi, gi)] = I[x,g]`. Absent keys = 0 (no edge)."""

    product_capacity_required: dict[tuple[int, int], int] = field(default_factory=dict)
    product_input_required: dict[tuple[int, int], int] = field(default_factory=dict)


@dataclass
class ProducerEntityView:
    producer_id: ProducerId
    globals: ProducerGlobalStats
    capacity_rows: list[CapacityRow]
    product_rows: list[ProductRow]
    input_rows: list[InputRow]
    relations: Relations


def _open_consumer_qty(all_orders: dict[OrderId, Order]) -> dict[Product, int]:
    """System-wide open consumer demand per product — the demand signal behind the
    capacity/input pressure features (consumer orders exist before routing)."""
    out: dict[Product, int] = {}
    for o in all_orders.values():
        if o.consumer_request is not None:
            out[o.product] = out.get(o.product, 0) + o.unresolved_quantity()
    return out


def _global_stats(p: ProducingAgent) -> ProducerGlobalStats:
    return ProducerGlobalStats(
        cut_off_price=p.cut_off_price,
        inventory_units=p.inventory_units(),
        inventory_limit=p.inventory_limit,
        inventory_headroom=p.headroom(),
        outstanding_inbound_units=p.outstanding_inbound_units(),
        active_investments=metrics.producer_active_investments(p),
    )


def _capacity_row(p: ProducingAgent, t: CapacityType, open_qty: dict[Product, int]) -> CapacityRow:
    current = p.capacities[t]
    mx = p.max_capacities[t]
    reserved = p.reserved_capacity().get(t, 0)
    return CapacityRow(
        capacity_type=t,
        owned=1.0,
        current=current,
        max=mx,
        headroom=max(0, mx - current),
        ratio=current / max(1, mx),
        degradation=p.degradation_for(t),
        reserved=reserved,
        previous_requested_capacity=p.requested_capacity_by_type.get(t, 0),
        free=max(0, current - reserved),
        # ADR-0053: blocked fraction of CURRENT usable capacity (Res/C), not max — a
        # degraded producer's reservations block more of what it can actually use.
        reserved_ratio=reserved / max(1, current),
        restore_effect=metrics.restore_effect(p, t),
        restore_cost=metrics.restore_cost(p, t),
        upgrade_effect=metrics.upgrade_effect(p, t),
        upgrade_cost=metrics.upgrade_cost(p, t),
        demand_pressure=metrics.capacity_demand_pressure(p, t, open_qty),
    )


def _product_row(
    p: ProducingAgent,
    x: Product,
    sheets: dict[Product, ProcessSheet],
    telemetry: EconomyTelemetry,
    open_qty: dict[Product, int],
) -> ProductRow:
    sheet = sheets[x]
    q = open_qty.get(x, 0)
    return ProductRow(
        product=x,
        eligible=1.0 if x in p.produces else 0.0,
        open_demand_qty=q,
        open_demand_tokens=0,  # filled by the builder from the all_orders token sweep
        product_depth=metrics.product_depth(x, sheets),
        recipe_input_count=metrics.recipe_input_count(sheet),
        recipe_input_units=metrics.recipe_input_units(sheet),
        recipe_capacity_type_count=metrics.recipe_capacity_type_count(sheet),
        recent_routed_qty=telemetry.routed_ema(p.id, x),
        recent_completed_qty=telemetry.completed_ema(p.id, x),
        recent_rejected_qty=telemetry.rejected_or_failed_ema(p.id, x),
        recent_completion_latency=telemetry.completion_latency(p.id, x),
        active_reservation_qty=metrics.active_reservation_qty(p, x),
        capacity_makeable_qty=metrics.capacity_makeable_qty(p, x),
        input_makeable_qty=metrics.input_makeable_qty(p, x),
        complete_qty_now=metrics.complete_qty_now(p, x),
        capacity_bottleneck_ratio=metrics.capacity_bottleneck_ratio(p, x, q),
        input_bottleneck_ratio=metrics.input_bottleneck_ratio(p, x, q),
        bottleneck_ratio=metrics.producer_product_bottleneck_ratio(p, x, q),
    )


def _input_row(p: ProducingAgent, g: Product, open_qty: dict[Product, int]) -> InputRow:
    return InputRow(
        input_product=g,
        inventory=p.inventory.get(g, 0),
        outstanding_inbound=p.outstanding_inbound.get(g, 0),
        blocked_reservation_qty=metrics.blocked_reservation_qty(p, g),
        downstream_demand_pressure=metrics.downstream_demand_pressure(p, g, open_qty),
    )


def _input_products(p: ProducingAgent, sheets: dict[Product, ProcessSheet]) -> list[Product]:
    """Input products the producer's declared recipes consume, sorted by id."""
    inputs: set[Product] = set()
    for x in p.produces:
        inputs.update(sheets[x].inputs)
    return sorted(inputs)


def build_producer_entity_view(
    producer: ProducingAgent,
    telemetry: EconomyTelemetry,
    sheets: dict[Product, ProcessSheet],
    all_orders: dict[OrderId, Order],
) -> ProducerEntityView:
    """Assemble the entity view for one producing agent at its decision point (ADR-0053).
    Rows default to sorted-id order for determinism; relations are keyed by the resulting
    row positions so a later reorder keeps the recipe planes aligned."""
    open_qty = _open_consumer_qty(all_orders)
    open_tokens = _open_consumer_tokens_map(all_orders)

    cap_types = sorted(producer.capacities)
    products = sorted(producer.produces)
    inputs = _input_products(producer, sheets)

    capacity_rows = [_capacity_row(producer, t, open_qty) for t in cap_types]
    product_rows = [
        _product_row(producer, x, sheets, telemetry, open_qty) for x in products
    ]
    for row in product_rows:
        row.open_demand_tokens = open_tokens.get(row.product, 0)
    input_rows = [_input_row(producer, g, open_qty) for g in inputs]

    cap_index = {t: i for i, t in enumerate(cap_types)}
    input_index = {g: i for i, g in enumerate(inputs)}
    relations = Relations()
    for xi, x in enumerate(products):
        sheet = sheets[x]
        for t, c in sheet.require.items():
            if t in cap_index:
                relations.product_capacity_required[(xi, cap_index[t])] = c
        for g, n in sheet.inputs.items():
            if g in input_index:
                relations.product_input_required[(xi, input_index[g])] = n

    return ProducerEntityView(
        producer_id=producer.id,
        globals=_global_stats(producer),
        capacity_rows=capacity_rows,
        product_rows=product_rows,
        input_rows=input_rows,
        relations=relations,
    )


def _open_consumer_tokens_map(all_orders: dict[OrderId, Order]) -> dict[Product, int]:
    out: dict[Product, int] = {}
    for o in all_orders.values():
        if o.consumer_request is not None:
            out[o.product] = out.get(o.product, 0) + o.available_tokens()
    return out
