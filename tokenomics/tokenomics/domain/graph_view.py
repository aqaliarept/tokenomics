"""EconomyGraphView: the domain's authoritative *relational* decision-time bundle for
one routing decision (EXP-0039, graph_primitives_v1 in ARCH.md).

The counterpart of OrdererEconomyView (EXP-0006A): instead of hand-compressed per-
producer metrics, it exposes primitive, typed graph facts — producers, products,
capacity types, orderers, live orders, and the recipe / capacity / inventory /
telemetry / commitment relations among them — and lets a GNN learn relational
summaries. It is the single place that owns "what the router is allowed to know" as a
graph, so the eval encoder only assigns local indices + normalizes and never becomes a
second hidden simulation.

Pure domain: no numpy, no torch, no serialization, no labels, no IDs as model features
(IDs live here only as join/audit keys). Mirrors the ownership boundary of
`build_orderer_economy_view`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .common import (
    CapacityType,
    OrderId,
    OrderingAgentId,
    Product,
    ProducerId,
)
from .order import ORDER_TTL, Order


# --- decision anchor ------------------------------------------------------------------
@dataclass
class DecisionNode:
    """The single anchor for the routing decision being scored. Carries no domain
    scalar features (ARCH.md): current context is expressed structurally through the
    `current_order`, `current_orderer`, and `candidate` relations."""


# --- nodes ----------------------------------------------------------------------------
@dataclass
class OrderNode:
    """One live order (every order in system.orders, not just the current one). Current-
    order / same-orderer context is structural (the `current_order` / `owns` relations),
    never duplicated as identity-like scalar flags (ARCH.md)."""

    order_id: OrderId
    original_quantity: int
    unresolved_quantity: int
    accepted_quantity: int
    completed_quantity: int
    failed_quantity: int
    available_tokens: int
    burnt_tokens: int
    order_age: int
    ttl_remaining: int
    funded: float
    order_kind: tuple[float, float, float]  # one-hot(consumer, replenishment, investment)


@dataclass
class OrdererNode:
    """One product-stream ordering agent. Minimal features: a source-role one-hot
    distinguishing consumer-funded and producer-funded streams (ARCH.md). Its fixed
    stream product / producer source are relations, not scalars."""

    orderer_id: OrderingAgentId
    source_role: tuple[float, float]  # one-hot(consumer_funded, producer_funded)


@dataclass
class ProducerNode:
    """One producer (including currently ineligible / zero-capacity producers).
    Producer-wide primitives only; per-capacity / per-product / historical state lives on
    relations."""

    producer_id: ProducerId
    cut_off_price: float
    cutoff_is_defined: float
    funds: int
    inventory_limit: int
    upgrade_running: float
    restoration_running: float
    inventory_expansion_running: float
    has_inventory_expansion: float


@dataclass
class ProductNode:
    """One configured product. No scalar product ID and no learned per-product embedding
    (ARCH.md) — products are distinguishable only through graph structure. Allowed
    historical primitives: folded demand / rejected-demand EMAs. Current open demand is
    live order nodes, never pre-summed here."""

    product: Product
    demand_ema: float
    rejected_demand_ema: float


@dataclass
class CapacityTypeNode:
    """One configured capacity type. No identity or domain scalar features (ARCH.md); its
    only signal is its schema-role embedding and graph structure."""

    capacity_type: CapacityType


# --- candidate edge (decision -> producer) --------------------------------------------
@dataclass
class CandidateEdge:
    """One eligible output candidate for the current order. No edge attributes; the
    stream index is audit-only metadata, never a model feature (ARCH.md)."""

    producer_id: ProducerId


# --- anchor edges (decision -> ...) ---------------------------------------------------
@dataclass
class CurrentOrderEdge:
    """decision -> order: the order being scored. No attributes."""

    order_id: OrderId


@dataclass
class CurrentOrdererEdge:
    """decision -> orderer: the acting product-stream ordering agent. No attributes."""

    orderer_id: OrderingAgentId


# --- static topology relations --------------------------------------------------------
@dataclass
class RequiresCapacityEdge:
    """product -> capacity_type: recipe capacity requirement, attr units per output unit."""

    product: Product
    capacity_type: CapacityType
    units: int


@dataclass
class RequiresInputEdge:
    """product -> product: direct recipe input, attr input units per output unit."""

    product: Product
    input_product: Product
    units: int


@dataclass
class CanMakeEdge:
    """producer -> product: authoritative eligibility (ADR-0028 `produces`)."""

    producer_id: ProducerId
    product: Product


@dataclass
class StreamForEdge:
    """orderer -> product: the fixed product a product-stream orderer serves."""

    orderer_id: OrderingAgentId
    product: Product


@dataclass
class SourceOfEdge:
    """producer -> orderer: the producer that funds a producer-funded ordering stream."""

    producer_id: ProducerId
    orderer_id: OrderingAgentId


@dataclass
class OwnsEdge:
    """orderer -> order: queue / source ownership of a live order."""

    orderer_id: OrderingAgentId
    order_id: OrderId


@dataclass
class RequestsEdge:
    """order -> product: the requested product."""

    order_id: OrderId
    product: Product


# --- capacity_state edge (producer -> capacity_type) ----------------------------------
@dataclass
class CapacityStateEdge:
    """Owned per-type capacity state for one producer. Primitive per-type facts, not
    min/mean/max summaries. `free = current - reserved` and `headroom = max - current`
    are local per-relation arithmetic, permitted by ARCH.md."""

    producer_id: ProducerId
    capacity_type: CapacityType
    current: int
    max: int
    reserved: int
    free: int
    headroom: int
    prior_requested_load: int
    degradation: int


# --- dynamic overlay relations (structural zeros: present even when values are 0) -----
@dataclass
class InventoryStateEdge:
    """producer -> product: structurally relevant input stock. Present for every storable
    input product of what the producer can make, even at zero inventory/inbound."""

    producer_id: ProducerId
    input_product: Product
    inventory_units: int
    outstanding_inbound_units: int


@dataclass
class TelemetryStateEdge:
    """producer -> product: folded past-cycle performance for every can_make pair. Strictly
    past-cycle EMAs (deployable at decision time), present even when all zero."""

    producer_id: ProducerId
    product: Product
    routed_ema: float
    completed_ema: float
    rejected_ema: float
    failed_ema: float
    completion_latency: float
    failure_latency: float


@dataclass
class PendingBidEdge:
    """order -> producer: already routed into the producer's current-cycle auction by an
    earlier ordering agent. Causal (exists before this action), not future leakage."""

    order_id: OrderId
    producer_id: ProducerId
    quantity: int
    tokens: int
    routed_age: int


@dataclass
class ReservationEdge:
    """order -> producer: an accepted composite bundle holding capacity while it waits for
    its inputs. Product-specific consequences total reserved capacity does not preserve."""

    order_id: OrderId
    producer_id: ProducerId
    quantity: int
    tokens: int
    cycles_waited: int


@dataclass
class EconomyGraphView:
    """One graph per routing decision. Node collections carry domain IDs only as join /
    audit keys (the eval encoder assigns local indices). Relations are typed edge lists;
    relations that share endpoint node types but differ semantically stay distinct."""

    decision: DecisionNode
    orders: list[OrderNode]
    orderers: list[OrdererNode]
    producers: list[ProducerNode]
    products: list[ProductNode]
    capacity_types: list[CapacityTypeNode]
    # anchors
    current_order: CurrentOrderEdge
    current_orderer: CurrentOrdererEdge
    candidate: list[CandidateEdge]
    # static topology relations
    requires_capacity: list[RequiresCapacityEdge]
    requires_input: list[RequiresInputEdge]
    can_make: list[CanMakeEdge]
    stream_for: list[StreamForEdge]
    source_of: list[SourceOfEdge]
    # dynamic relations
    owns: list[OwnsEdge]
    requests: list[RequestsEdge]
    capacity_state: list[CapacityStateEdge]
    inventory_state: list[InventoryStateEdge]
    telemetry_state: list[TelemetryStateEdge]
    pending_bid: list[PendingBidEdge]
    reservation: list[ReservationEdge]


def build_economy_graph_view(
    system,
    current_orderer_id: OrderingAgentId,
    current_order_id: OrderId,
    eligible_producer_ids: list[ProducerId],
) -> EconomyGraphView:
    """Assemble the graph_primitives_v1 view for one routing decision from live `system`
    state, at the acting ordering agent's turn (before its action, before the teacher).

    `eligible_producer_ids` is the decision's candidate set, passed in so candidate edges
    align one-to-one with the legacy eligible columns EXP-0038 dumped (alignment
    contract). Ineligible / zero-capacity producers still appear as producer nodes and in
    the static relations; only the `candidate` edges are restricted to the eligible set.
    """
    producers = system.producing_agents

    order_nodes = [_order_node(o) for o in system.orders.values()]
    orderer_nodes = [
        OrdererNode(orderer_id=aid, source_role=_source_role(aid, prod, producers))
        for aid, prod in system.stream_products.items()
    ]
    producer_nodes = [_producer_node(pid, p) for pid, p in producers.items()]
    product_nodes = [
        ProductNode(product=x, demand_ema=system.telemetry.demand_ema(x),
                    rejected_demand_ema=system.telemetry.rejected_demand_ema(x))
        for x in system.process_sheets
    ]
    capacity_type_nodes = [
        CapacityTypeNode(capacity_type=t)
        for t in sorted({t for p in producers.values() for t in p.capacities})
    ]

    candidate = [CandidateEdge(producer_id=pid) for pid in eligible_producer_ids]

    # static topology relations (recipe + eligibility + stream/source)
    requires_capacity: list[RequiresCapacityEdge] = []
    requires_input: list[RequiresInputEdge] = []
    for x, ps in system.process_sheets.items():
        for t, c in ps.require.items():
            requires_capacity.append(RequiresCapacityEdge(product=x, capacity_type=t, units=c))
        for i, n in ps.inputs.items():
            requires_input.append(RequiresInputEdge(product=x, input_product=i, units=n))
    can_make = [CanMakeEdge(producer_id=pid, product=x)
                for pid, p in producers.items() for x in p.produces]
    stream_for = [StreamForEdge(orderer_id=aid, product=prod)
                  for aid, prod in system.stream_products.items()]
    source_of: list[SourceOfEdge] = []
    for aid, prod in system.stream_products.items():
        source = aid[: -(len(f"__{prod}"))]
        if source in producers:
            source_of.append(SourceOfEdge(producer_id=source, orderer_id=aid))

    # dynamic ownership / request relations
    owns: list[OwnsEdge] = []
    requests: list[RequestsEdge] = []
    for o in system.orders.values():
        requests.append(RequestsEdge(order_id=o.id, product=o.product))
        if not o.requires_funding():  # an unfunded (expiring) order has no orderer this cycle
            owns.append(OwnsEdge(orderer_id=o.stream_agent_id(), order_id=o.id))

    cycle = system.cycle_counter
    tel = system.telemetry
    capacity_state: list[CapacityStateEdge] = []
    inventory_state: list[InventoryStateEdge] = []
    telemetry_state: list[TelemetryStateEdge] = []
    pending_bid: list[PendingBidEdge] = []
    reservation: list[ReservationEdge] = []
    for pid, p in producers.items():
        reserved = p.reserved_capacity()
        for t, cur in p.capacities.items():
            mx = p.max_capacities.get(t, 0)
            resv = reserved.get(t, 0)
            capacity_state.append(CapacityStateEdge(
                producer_id=pid,
                capacity_type=t,
                current=cur,
                max=mx,
                reserved=resv,
                free=cur - resv,
                headroom=mx - cur,
                prior_requested_load=p.requested_capacity_by_type.get(t, 0),
                degradation=p.degradation_for(t),
            ))
        # structural zeros: every storable input product / every can_make pair
        storable = {i for made in p.produces for i in system.process_sheets[made].inputs}
        for i in sorted(storable):
            inventory_state.append(InventoryStateEdge(
                producer_id=pid, input_product=i,
                inventory_units=p.inventory.get(i, 0),
                outstanding_inbound_units=p.outstanding_inbound.get(i, 0)))
        for x in sorted(p.produces):
            telemetry_state.append(TelemetryStateEdge(
                producer_id=pid, product=x,
                routed_ema=tel.routed_ema(pid, x),
                completed_ema=tel.completed_ema(pid, x),
                rejected_ema=tel.rejected_ema(pid, x),
                failed_ema=tel.failed_ema(pid, x),
                completion_latency=tel.completion_latency(pid, x),
                failure_latency=tel.failure_latency(pid, x)))
        for b in p.bids:
            pending_bid.append(PendingBidEdge(
                order_id=b.order_id, producer_id=pid, quantity=b.quantity,
                tokens=b.tokens, routed_age=max(0, cycle - b.routed_cycle)))
        for r in p.reservations:
            reservation.append(ReservationEdge(
                order_id=r.bid.order_id, producer_id=pid, quantity=r.bid.quantity,
                tokens=r.bid.tokens, cycles_waited=r.cycles_waited))

    return EconomyGraphView(
        decision=DecisionNode(),
        orders=order_nodes,
        orderers=orderer_nodes,
        producers=producer_nodes,
        products=product_nodes,
        capacity_types=capacity_type_nodes,
        current_order=CurrentOrderEdge(order_id=current_order_id),
        current_orderer=CurrentOrdererEdge(orderer_id=current_orderer_id),
        candidate=candidate,
        requires_capacity=requires_capacity,
        requires_input=requires_input,
        can_make=can_make,
        stream_for=stream_for,
        source_of=source_of,
        owns=owns,
        requests=requests,
        capacity_state=capacity_state,
        inventory_state=inventory_state,
        telemetry_state=telemetry_state,
        pending_bid=pending_bid,
        reservation=reservation,
    )


def _order_kind(order: Order) -> tuple[float, float, float]:
    if order.consumer_request is not None:
        return (1.0, 0.0, 0.0)
    if order.replenishment_request is not None:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)  # investment


def _order_node(o: Order) -> OrderNode:
    return OrderNode(
        order_id=o.id,
        original_quantity=o.quantity,
        unresolved_quantity=o.unresolved_quantity(),
        accepted_quantity=o.accepted_qty,
        completed_quantity=o.completed_qty,
        failed_quantity=o.failed_qty,
        available_tokens=o.available_tokens(),
        burnt_tokens=o.burnt_tokens,
        order_age=o.cycle_counter,
        ttl_remaining=max(0, ORDER_TTL - o.cycle_counter),
        funded=1.0 if o.funded else 0.0,
        order_kind=_order_kind(o),
    )


def _source_role(
    orderer_id: OrderingAgentId, product: Product, producers: dict
) -> tuple[float, float]:
    """A product-stream orderer is producer-funded iff its source is a producer (ADR-0052:
    orderer id = f"{source}__{product}"). Source is recovered by stripping the known
    product suffix, then membership-tested against producers — an id join, not a feature."""
    source = orderer_id[: -(len(f"__{product}"))]
    is_producer = source in producers
    return (0.0, 1.0) if is_producer else (1.0, 0.0)


def _producer_node(pid: ProducerId, p) -> ProducerNode:
    # UNDEFINED_PRICE is NaN, so test definedness with isnan, never `!= UNDEFINED_PRICE`
    # (NaN != NaN is always True) — an undefined cut-off must read as masked, not defined.
    defined = not math.isnan(p.cut_off_price)
    return ProducerNode(
        producer_id=pid,
        cut_off_price=float(p.cut_off_price) if defined else 0.0,
        cutoff_is_defined=1.0 if defined else 0.0,
        funds=p.funds,
        inventory_limit=p.inventory_limit,
        upgrade_running=1.0 if p.upgrade_running else 0.0,
        restoration_running=1.0 if p.restoration_running else 0.0,
        inventory_expansion_running=1.0 if p.inventory_expansion_running else 0.0,
        has_inventory_expansion=1.0 if p.inventory_expansion is not None else 0.0,
    )
