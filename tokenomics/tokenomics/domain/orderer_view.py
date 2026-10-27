"""OrdererEconomyView: the domain's authoritative enriched feature bundle for one
ordering agent (EXP-0006A).

Assembled from live producer/product/order state plus EconomyTelemetry, it is the
single place that owns "what the orderer is allowed to know" — so the env encoder
only normalizes it and never becomes a second hidden simulation model. Pure domain:
no RL, no numpy, no IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import metrics
from .common import CapacityUnitPrice, OrderId, Product, ProducerId
from .config import ProcessSheet
from .order import ORDER_TTL, Order
from .producing_agent import ProducingAgent
from .telemetry import EconomyTelemetry


@dataclass
class OrderSlotStats:
    """Per open order this agent must distribute (the doc's order-slot features)."""

    order_id: OrderId
    product: Product
    unresolved_quantity: int
    available_tokens: int
    tokens_per_unit: float
    order_age: int
    ttl_remaining: int
    order_type: tuple[float, float, float]  # one_hot(consumer, replenishment, investment)
    eligible_producer_count: int
    eligible_mask: dict[ProducerId, float]  # per producer: 1.0 if eligible else 0.0


@dataclass(frozen=True)
class ProducerStats:
    """Per producer: live capacity / load / in-flight state + recent flow rates
    (the doc's producer features). Capacity-blind routing (ADR-0029) is the gap the
    enriched obs closes — this is exactly the load/headroom the search teacher uses.
    Frozen: one instance is shared by every stream view built from one
    `StreamViewCache`."""

    total_current_capacity: int
    total_max_capacity: int
    capacity_headroom: int
    min_capacity_ratio: float
    mean_capacity_ratio: float
    requested_load: int
    reserved_load: int
    total_load: int
    cut_off_price: CapacityUnitPrice
    outstanding_inbound_units: int
    active_investments: tuple[float, float, float]
    acceptance_rate: float
    completion_rate: float
    rejection_rate: float


@dataclass
class ProductStats:
    """Per product appearing in this agent's orders: system-wide open consumer
    demand + recent demand history + static recipe-graph shape (the doc's product
    features)."""

    open_demand_quantity: int
    open_demand_tokens: int
    recent_average_cycle_demand: float
    recent_rejected_quantity: float
    product_depth: int
    recipe_input_count: int
    recipe_input_units: int
    recipe_capacity_type_count: int
    eligible_producer_count: int


@dataclass(frozen=True)
class ProducerProductStats:
    """Per (order, producer): can this producer make this order's product now, and
    how reliably has it lately (the doc's producer-product features). Quantity-
    dependent fields use the order's unresolved quantity. Frozen: shared by the stream
    views built from one `StreamViewCache`."""

    eligible: float
    accept_qty_now: int
    complete_qty_now: int
    makeable_qty_now: int
    capacity_bottleneck_ratio: float
    input_bottleneck_ratio: float
    bottleneck_ratio: float
    slot_makeability: float
    recent_routed_qty: float
    recent_completed_qty: float
    recent_rejected_qty: float
    recent_success_rate: float
    recent_throughput: float
    recent_completion_latency: float
    recent_failure_latency: float


@dataclass
class OrdererEconomyView:
    order_slots: list[OrderSlotStats]
    producers: dict[ProducerId, ProducerStats]
    products: dict[Product, ProductStats]
    # per (order_id, producer_id) over the full slot x producer grid.
    producer_products: dict[tuple[OrderId, ProducerId], ProducerProductStats]


@dataclass
class ProductStreamView:
    """Authoritative feature bundle for one *product-stream* ordering agent (EXP-0007A,
    ADR-0052). The agent holds a single fixed product, so unlike OrdererEconomyView the
    product appears once (not per slot) and the producer-product facts collapse from a
    slot x producer grid to one row per producer (computed against the stream's total
    open backlog). Pure domain: the env only normalizes it."""

    product: Product
    product_stats: ProductStats
    backlog: list[OrderSlotStats]
    producers: dict[ProducerId, ProducerStats]
    # per producer, for the stream's fixed product.
    producer_products: dict[ProducerId, ProducerProductStats]


@dataclass
class StreamViewCache:
    """Memo of the producer rows shared by product-stream views while no producer row can
    change (task 8.4; docs/DIAGNOSTICS.md entry 17).

    Producer rows read capacities, reservations, requested load, cut-off price, inventory,
    inbound and investment flags, and telemetry EMAs. `System` creates a fresh memo at
    `start_cycle` (after production and the telemetry fold) and at `start_ordering`. It
    drops the memo at every call that can change those: `producing_agent_action` and
    `complete_cycle`. In between, the only mutations are emission and consumer orders
    (neither touches a producer or an EMA) and routing. Routing (`place_sub_orders`)
    only appends to `p.bids`, which no row reads, and adds telemetry accumulators, which
    fold only in `complete_cycle`. So a row computed once is the row every later stream
    would compute, and each stream reuses it instead of recomputing all producers."""

    producers: dict[ProducerId, ProducerStats] = field(default_factory=dict)
    # (producer, product, backlog quantity) -> row. The quantity is part of the key
    # because four fields depend on it, except for a producer that does not declare
    # the product. There every quantity-dependent `metrics` function returns its
    # constant (0 / 0.0) before reading q, so the key carries None and one row serves
    # every quantity.
    producer_products: dict[tuple[ProducerId, Product, int | None], ProducerProductStats] = (
        field(default_factory=dict)
    )
    # The single row of every (producer, product) pair where the producer does not
    # declare the product AND telemetry holds no EMA entry for the pair. Each field is
    # either a `metrics` early-return constant or a missing-key telemetry 0.0, so the
    # row is the same for every such pair and quantity. Computed once per phase by the
    # real `_producer_product_stats` (not hand-written), then shared.
    undeclared_no_history: ProducerProductStats | None = None


def _order_type(order: Order) -> tuple[float, float, float]:
    if order.consumer_request is not None:
        return (1.0, 0.0, 0.0)
    if order.replenishment_request is not None:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)  # investment


def _order_slot(
    order: Order, producer_ids: list[ProducerId], producers: dict[ProducerId, ProducingAgent]
) -> OrderSlotStats:
    q = order.unresolved_quantity()
    tokens = order.available_tokens()
    mask = {pid: (1.0 if order.product in producers[pid].produces else 0.0) for pid in producer_ids}
    return OrderSlotStats(
        order_id=order.id,
        product=order.product,
        unresolved_quantity=q,
        available_tokens=tokens,
        tokens_per_unit=tokens / max(1, q),
        order_age=order.cycle_counter,
        ttl_remaining=max(0, ORDER_TTL - order.cycle_counter),
        order_type=_order_type(order),
        eligible_producer_count=sum(1 for m in mask.values() if m == 1.0),
        eligible_mask=mask,
    )


def _producer_stats(p: ProducingAgent, telemetry: EconomyTelemetry) -> ProducerStats:
    return ProducerStats(
        total_current_capacity=metrics.producer_total_current_capacity(p),
        total_max_capacity=metrics.producer_total_max_capacity(p),
        capacity_headroom=metrics.producer_capacity_headroom(p),
        min_capacity_ratio=metrics.producer_min_capacity_ratio(p),
        mean_capacity_ratio=metrics.producer_mean_capacity_ratio(p),
        requested_load=metrics.producer_requested_load(p),
        reserved_load=metrics.producer_reserved_load(p),
        total_load=metrics.producer_total_load(p),
        cut_off_price=p.cut_off_price,
        outstanding_inbound_units=metrics.producer_outstanding_inbound_units(p),
        active_investments=metrics.producer_active_investments(p),
        acceptance_rate=telemetry.producer_acceptance_rate(p.id),
        completion_rate=telemetry.producer_completion_rate(p.id),
        rejection_rate=telemetry.producer_rejection_rate(p.id),
    )


def _open_consumer_demand(
    product: Product, all_orders: dict[OrderId, Order]
) -> tuple[int, int]:
    """System-wide open consumer demand for a product (qty, tokens) — the real
    demand signal, since consumer orders exist before routing."""
    qty = tokens = 0
    for o in all_orders.values():
        if o.consumer_request is not None and o.product == product:
            qty += o.unresolved_quantity()
            tokens += o.available_tokens()
    return qty, tokens


def _product_stats(
    x: Product,
    producers: dict[ProducerId, ProducingAgent],
    telemetry: EconomyTelemetry,
    sheets: dict[Product, ProcessSheet],
    all_orders: dict[OrderId, Order],
) -> ProductStats:
    qty, tokens = _open_consumer_demand(x, all_orders)
    sheet = sheets[x]
    return ProductStats(
        open_demand_quantity=qty,
        open_demand_tokens=tokens,
        recent_average_cycle_demand=telemetry.demand_ema(x),
        recent_rejected_quantity=telemetry.rejected_demand_ema(x),
        product_depth=metrics.product_depth(x, sheets),
        recipe_input_count=metrics.recipe_input_count(sheet),
        recipe_input_units=metrics.recipe_input_units(sheet),
        recipe_capacity_type_count=metrics.recipe_capacity_type_count(sheet),
        eligible_producer_count=metrics.eligible_producer_count(x, producers.values()),
    )


def _producer_product_stats(
    p: ProducingAgent, x: Product, q: int, telemetry: EconomyTelemetry
) -> ProducerProductStats:
    return ProducerProductStats(
        eligible=1.0 if x in p.produces else 0.0,
        accept_qty_now=metrics.accept_qty_now(p, x),
        complete_qty_now=metrics.complete_qty_now(p, x),
        makeable_qty_now=metrics.makeable_qty_now(p, x),
        capacity_bottleneck_ratio=metrics.capacity_bottleneck_ratio(p, x, q),
        input_bottleneck_ratio=metrics.input_bottleneck_ratio(p, x, q),
        bottleneck_ratio=metrics.producer_product_bottleneck_ratio(p, x, q),
        slot_makeability=metrics.slot_makeability(p, x, q),
        recent_routed_qty=telemetry.routed_ema(p.id, x),
        recent_completed_qty=telemetry.completed_ema(p.id, x),
        recent_rejected_qty=telemetry.rejected_or_failed_ema(p.id, x),
        recent_success_rate=telemetry.success_rate(p.id, x),
        recent_throughput=telemetry.throughput(p.id, x),
        recent_completion_latency=telemetry.completion_latency(p.id, x),
        recent_failure_latency=telemetry.failure_latency(p.id, x),
    )


def build_orderer_economy_view(
    incoming: dict[OrderId, Order],
    producers: dict[ProducerId, ProducingAgent],
    telemetry: EconomyTelemetry,
    sheets: dict[Product, ProcessSheet],
    all_orders: dict[OrderId, Order],
) -> OrdererEconomyView:
    """Assemble the view for one ordering agent. `incoming` are the agent's open
    orders (sorted by id for a stable layout, matching the env's slot ordering);
    `all_orders` is the whole system's open orders (for system-wide demand)."""
    producer_ids = sorted(producers)
    ordered = [incoming[oid] for oid in sorted(incoming)]
    slots = [_order_slot(o, producer_ids, producers) for o in ordered]
    producer_stats = {
        pid: _producer_stats(producers[pid], telemetry) for pid in producer_ids
    }
    product_stats = {
        o.product: _product_stats(o.product, producers, telemetry, sheets, all_orders)
        for o in ordered
    }
    producer_products = {
        (o.id, pid): _producer_product_stats(
            producers[pid], o.product, o.unresolved_quantity(), telemetry
        )
        for o in ordered
        for pid in producer_ids
    }
    return OrdererEconomyView(
        order_slots=slots,
        producers=producer_stats,
        products=product_stats,
        producer_products=producer_products,
    )


def build_product_stream_view(
    product: Product,
    incoming: dict[OrderId, Order],
    producers: dict[ProducerId, ProducingAgent],
    telemetry: EconomyTelemetry,
    sheets: dict[Product, ProcessSheet],
    all_orders: dict[OrderId, Order],
    cache: StreamViewCache | None = None,
) -> ProductStreamView:
    """Assemble the view for one product-stream ordering agent (ADR-0052). All
    `incoming` orders share `product`. Quantity-dependent producer-product facts use
    the stream's total open backlog quantity, so the view has one row per producer
    rather than a per-slot grid. An idle stream (empty `incoming`) still reports every
    producer row so the obs is well-defined when the agent's turn comes up empty.

    `cache` (see `StreamViewCache`: valid while no producer row can change) reuses
    producer rows across streams; without it every row is computed fresh. Both give
    identical views (tests/test_orderer_view_perf_equivalence.py)."""
    producer_ids = sorted(producers)
    ordered = [incoming[oid] for oid in sorted(incoming)]
    backlog = [_order_slot(o, producer_ids, producers) for o in ordered]
    backlog_qty = sum(o.unresolved_quantity() for o in ordered)
    product_stats = _product_stats(product, producers, telemetry, sheets, all_orders)
    if cache is None:
        producer_stats = {pid: _producer_stats(producers[pid], telemetry) for pid in producer_ids}
        producer_products = {
            pid: _producer_product_stats(producers[pid], product, backlog_qty, telemetry)
            for pid in producer_ids
        }
    else:
        producer_stats = {}
        producer_products = {}
        for pid in producer_ids:
            row = cache.producers.get(pid)
            if row is None:
                row = cache.producers[pid] = _producer_stats(producers[pid], telemetry)
            producer_stats[pid] = row
            p = producers[pid]
            declared = product in p.produces
            if not declared and not telemetry.has_pair_history(pid, product):
                pp = cache.undeclared_no_history
                if pp is None:
                    pp = cache.undeclared_no_history = _producer_product_stats(
                        p, product, backlog_qty, telemetry
                    )
                producer_products[pid] = pp
                continue
            key = (pid, product, backlog_qty if declared else None)
            pp = cache.producer_products.get(key)
            if pp is None:
                pp = cache.producer_products[key] = _producer_product_stats(
                    p, product, backlog_qty, telemetry
                )
            producer_products[pid] = pp
    return ProductStreamView(
        product=product,
        product_stats=product_stats,
        backlog=backlog,
        producers=producer_stats,
        producer_products=producer_products,
    )
