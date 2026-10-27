"""System: the cycle orchestrator. Port of emulation/domain/system.go.

Divergence from Go (intentional): Go computes `producerInfos` once in NewSystem
and never refreshes it, so ordering agents always saw the *initial* producer
state. That is a latent bug — for RL the ordering agent must see live capacity
and cut-off price (ADR-0002). Here producer_infos is refreshed each cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from .common import (
    Capacity,
    CapacityType,
    ConsumerId,
    OrderId,
    OrderingAgentId,
    ProducerId,
    Product,
    Tokens,
    WrongStateError,
    f32,
    from_consumer_id,
    from_producer_id,
    is_nan,
    must_get,
    stream_id,
)
from .config import Configuration
from .shocks import ProducerFacts, ShockEvent, ShockKind, ShockModel, validate_shock_model
from .consumer import Consumer
from .order import (
    ConsumerRequestCompleted,
    ConsumerRequestRejected,
    InvestmentRequestCompleted,
    InvestmentRequestRejected,
    Order,
    OrderStillProcessing,
    ReplenishmentRequestCompleted,
    ReplenishmentRequestRejected,
    Score,
)
from .orderer_view import (
    OrdererEconomyView,
    ProductStreamView,
    StreamViewCache,
    build_orderer_economy_view,
    build_product_stream_view,
)
from .producer_view import ProducerEntityView, build_producer_entity_view
from .ordering_agent import OrderingAgent, OrderingAgentCommand, OrderingAgentView
from .requests import ConsumerRequest, InvestmentType, ReplenishmentRequest
from .telemetry import EconomyTelemetry
from .producing_agent import (
    Bid,
    ProducerInfo,
    ProducingAgent,
    ProducingAgentCommand,
    ProducingAgentView,
    _Reservation,
)


class SystemState:
    ORDERS_PLACEMENT = 0
    ORDERING = 1


class OrderIdGenerator(Protocol):
    def new(self) -> OrderId: ...


class SequentialOrderIdGenerator:
    def __init__(self) -> None:
        self._n = 0

    def new(self) -> OrderId:
        self._n += 1
        return f"o{self._n}"


@dataclass(frozen=True)
class ConsumerOrderOutcome:
    """One consumer order resolved this cycle (design D2, add-measurement-gate-harness).
    Pure data so the env can attribute ordered/fulfilled units per consumer and audit
    token priority without re-deriving them from the aggregate counters.
    `placement_cycle` is the cycle the order was first played in (`Order.placement_cycle`)."""

    consumer_id: ConsumerId
    product: Product
    quantity: int
    tokens: Tokens  # backing tokens at placement (ConsumerRequest.tokens)
    placement_cycle: int
    completed: bool  # False = rejected (failed or TTL expired)


@dataclass(frozen=True)
class ProducerAuctionRecord:
    """One producer's auction this cycle (design D3, add-measurement-gate-harness): the
    raw inputs of the env's spare-fraction metric, pure data. `capacity` is the total
    current capacity over owned types when `produce()` runs, before this cycle's
    degradation; `accepted_units` is the summed capacity bundle of the bids accepted
    THIS cycle. Capacity still held by reservations carried from earlier cycles is not
    counted as accepted, so a spare fraction built from this can be overstated."""

    producer_id: ProducerId
    capacity: Capacity
    accepted_units: Capacity


@dataclass(frozen=True)
class CapacitySnapshot:
    """One producer's (current, max) capacity at the end of a played cycle, taken before the
    trailing `start_cycle` (add-shocks-i-drift D2), so a shock belonging to the next,
    possibly unplayed, cycle never reaches end-of-cycle metrics such as `r_end`. Satisfies
    the env's `HasCapacities` protocol."""

    producer_id: ProducerId
    capacities: dict[CapacityType, Capacity]
    max_capacities: dict[CapacityType, Capacity]


def capacity_hit_removal(capacity: Capacity, magnitude: float) -> Capacity:
    """Units a capacity hit removes: floor(m * c), exact (D2). `Fraction(str(m))` reads the
    magnitude as its decimal, so e.g. 0.29 * 100 removes 29, not the 28 a float product gives."""
    return math.floor(Fraction(str(magnitude)) * capacity)


@dataclass
class CycleResult:
    score: Score  # Go penalty score (parity only)
    consumer_completed: int = 0  # consumer orders fulfilled (the terminal objective)
    investment_completed: int = 0  # restoration/upgrade orders fulfilled (instrumental)
    in_progress: int = 0  # orders still processing
    consumer_rejected: int = 0  # consumer orders rejected or timed out
    investment_rejected: int = 0  # investment orders rejected or timed out
    replenishment_completed: int = 0  # input-purchase orders delivered (instrumental)
    replenishment_rejected: int = 0  # input-purchase orders rejected or timed out
    # Quantity-weighted consumer accounting (#3): a whole order completes/rejects as
    # a unit (ADR-0025), but a qty-3 order represents 3x the demand of a qty-1 one.
    # The env weights its training reward + metrics by these, not by order counts.
    consumer_completed_units: int = 0  # consumer units delivered
    consumer_rejected_units: int = 0  # consumer units demanded but not delivered
    # Per-order record of every consumer order completed or rejected this cycle
    # (design D2); additive — the counters above are unchanged.
    consumer_outcomes: tuple[ConsumerOrderOutcome, ...] = ()
    # One auction record per producer, in producer order (design D3); additive.
    producer_auctions: tuple[ProducerAuctionRecord, ...] = ()
    # Every producer's capacity at the end of this played cycle, in producer order, taken
    # before the next cycle's shocks apply (add-shocks-i-drift D2); additive.
    end_capacities: tuple[CapacitySnapshot, ...] = ()

    @property
    def completed(self) -> int:
        return self.consumer_completed + self.investment_completed + self.replenishment_completed

    @property
    def rejected(self) -> int:
        return self.consumer_rejected + self.investment_rejected + self.replenishment_rejected


# Survival-first funding priority within a producer's budget (ADR-0025):
# restoration → replenishment → upgrade → inventory expansion (lower = first).
# Replenishment (input-purchase) orders are not investment orders → priority 1.
_INVEST_PRIORITY: dict[InvestmentType, int] = {
    InvestmentType.RESTORATION: 0,
    InvestmentType.UPGRADE: 2,
    InvestmentType.INVENTORY_EXPANSION: 3,
}


def _consumer_outcome(order: Order, request: ConsumerRequest, completed: bool) -> ConsumerOrderOutcome:
    # System._place_consumer_orders tags every consumer order; an untagged one would
    # silently corrupt played-cycle accounting (design D2), so fail loudly.
    if order.placement_cycle is None:
        raise RuntimeError(f"consumer order [{order.id}] has no placement cycle")
    return ConsumerOrderOutcome(
        request.consumer_id, order.product, order.quantity, request.tokens,
        order.placement_cycle, completed,
    )


def _order_priority(order: Order) -> int:
    if order.investment_request is not None:
        return _INVEST_PRIORITY.get(order.investment_request.type, 1)
    return 1


def distribute_investment_fund(orders: dict[OrderId, Order], investment_fund: Tokens) -> None:
    """Two-stage producer-first allocation (ADR-0025). Stage 1 splits the fund
    across *producers* proportional to each producer's scalar cut-off price (NaN
    producers split the remainder equally) — so issuing more orders cannot raise a
    producer's share. Stage 2 distributes a producer's whole budget across its own
    orders in survival-first priority order."""
    groups: dict[ProducerId, list[Order]] = {}
    for order in orders.values():
        if not order.requires_funding():
            continue
        pid = order.producer_id()
        assert pid is not None  # only producer-funded orders require funding
        groups.setdefault(pid, []).append(order)
    if not groups:
        return

    # producer cut-off = the (shared) cut-off its orders were issued at
    priced: dict[ProducerId, float] = {}
    nan_pids: list[ProducerId] = []
    for pid, os in groups.items():
        c = os[0].cut_off_price()
        if is_nan(c):
            nan_pids.append(pid)
        else:
            priced[pid] = c

    # stage 1: per-producer budgets
    total = sum(priced.values())
    remains = investment_fund
    budgets: dict[ProducerId, Tokens] = {}
    for pid, c in priced.items():
        b = 0 if total == 0 else int(f32(c) * f32(investment_fund) / f32(total))
        budgets[pid] = b
        remains -= b
    if nan_pids:
        share = remains // len(nan_pids)
        for pid in nan_pids:
            budgets[pid] = share
            remains -= share

    # stage 2: spend each producer's budget greedily, survival-first (ADR-0027).
    # Walk orders by fixed priority; fill each up to its funding target (cost),
    # advancing to the next only with budget left over. The boundary order takes
    # whatever remains (partial); lower-priority orders get nothing and expire
    # unfunded. None target = unbounded (take the whole remaining budget).
    for pid, os in groups.items():
        remaining = budgets.get(pid, 0)
        for order in sorted(os, key=_order_priority):
            if remaining <= 0:
                break  # rest expire (rejected) this cycle
            target = order.funding_target()
            grant = remaining if target is None else min(target, remaining)
            if grant <= 0:
                continue  # zero-cost order claims nothing, never funded for free
            order.fund(grant)
            remaining -= grant


def product_stream_agents(config: Configuration) -> list[tuple[str, int]]:
    """Restricted product-stream coverage (ADR-0052): the (source, product) pairs a
    source can actually order, in deterministic order. Single source of truth for
    System (agent creation + placement) and the env (`possible_agents`).

    - consumer source: each product in its pool (the products it requests).
    - producer source: replenishment input products for everything it `produces`,
      plus its restoration / upgrade / inventory-expansion requirement products.
    """
    streams: list[tuple[str, int]] = []
    for c in config.consumers:
        for product in sorted(set(c.product_pool)):
            streams.append((c.id, product))
    for p in config.producer_configs:
        products: set[int] = set()
        for made in p.produces:
            products.update(config.inputs_of(made))
        products.add(p.restoration.require)
        products.add(p.upgrade.require)
        if p.inventory_expansion is not None:
            products.add(p.inventory_expansion.require)
        for product in sorted(products):
            streams.append((p.id, product))
    return streams


class System:
    def __init__(
        self,
        id_gen: OrderIdGenerator,
        config: Configuration,
        consumers: dict[ConsumerId, Consumer],
        product_stream_ordering: bool = False,
        shock_schedule: tuple[ShockEvent, ...] = (),
    ) -> None:
        self.state = SystemState.ORDERS_PLACEMENT
        self.id_gen = id_gen
        self.cycle_emission: Tokens = config.cycle_emission
        self.investment_fund: Tokens = 0
        self.process_sheets = {ps.product: ps for ps in config.process_sheets}
        # each producer owns the recipes for the products it makes (ADR-0029), so it
        # can expand routed sub-orders into capacity bundles itself.
        self.producing_agents: dict[ProducerId, ProducingAgent] = {
            p.id: ProducingAgent(p, {prod: self.process_sheets[prod] for prod in p.produces})
            for p in config.producer_configs
        }
        self.ordering_agents: dict[OrderingAgentId, OrderingAgent] = {}
        self.orders: dict[OrderId, Order] = {}
        self.consumers = consumers
        self.cycle_counter = 0

        # EXP-0006A: domain-owned economy history (recent routed/produced/demand
        # EMAs) behind the enriched orderer observation. Each bid carries its own
        # routed cycle (see ProducingAgent.place_sub_orders), so production events are
        # attributed a completion/failure latency per bid.
        self.telemetry = EconomyTelemetry()
        # Producer rows shared by product-stream views while no producer row can change
        # (task 8.4, StreamViewCache): created at start_cycle / start_ordering, dropped at
        # every producer mutation (producing_agent_action, complete_cycle). None = build
        # views fresh.
        self._stream_view_cache: StreamViewCache | None = None

        self.producer_infos: dict[ProducerId, ProducerInfo] = self._snapshot_producer_infos()
        # Injected shock schedule (ADR-0055, add-shocks-i-drift D2): the env draws it, the
        # domain only replays it and never mutates it. Private: no view reads it, so agents
        # see a shock only through its consequences.
        self._shock_schedule: tuple[ShockEvent, ...] = self._check_shock_schedule(
            config, tuple(shock_schedule))
        self._hits_by_cycle: dict[int, list[ShockEvent]] = {}
        # played cycle -> producers in an outage that cycle (overlapping outages = union)
        self._outages_by_cycle: dict[int, set[ProducerId]] = {}
        for e in self._shock_schedule:
            if e.kind is ShockKind.CAPACITY_HIT:
                self._hits_by_cycle.setdefault(e.start_cycle, []).append(e)
            elif e.kind is ShockKind.PRODUCER_OUTAGE:
                for c in range(e.start_cycle, e.start_cycle + e.duration):
                    self._outages_by_cycle.setdefault(c, set()).add(e.target.producer)
        # ADR-0052: product-stream mode keys ordering agents by (source, product) so
        # one agent never multiplexes products; legacy mode keys them by source.
        self.product_stream_ordering = product_stream_ordering
        # aid -> the fixed product a stream agent serves (ADR-0052); lets an idle
        # stream still build a well-defined product-stream view.
        self.stream_products: dict[OrderingAgentId, int] = {}
        if product_stream_ordering:
            for source, product in product_stream_agents(config):
                aid = stream_id(source, product)
                self.ordering_agents[aid] = OrderingAgent(aid)
                self.stream_products[aid] = product
        else:
            for c in consumers.values():
                aid = from_consumer_id(c.id())
                self.ordering_agents[aid] = OrderingAgent(aid)
            for prod_id in self.producing_agents:
                aid = from_producer_id(prod_id)
                self.ordering_agents[aid] = OrderingAgent(aid)

        self.start_cycle()

    @staticmethod
    def _check_shock_schedule(
        config: Configuration, schedule: tuple[ShockEvent, ...]
    ) -> tuple[ShockEvent, ...]:
        """Validate the injected schedule against the config with the config-level rules
        (design D1); a kind without domain mechanics is refused rather than ignored."""
        if not schedule:
            return ()
        validate_shock_model(ShockModel(events=schedule), {
            p.id: ProducerFacts(frozenset(p.capacities), p.produces) for p in config.producer_configs
        })
        for e in schedule:
            if e.kind not in (ShockKind.CAPACITY_HIT, ShockKind.PRODUCER_OUTAGE):
                raise ValueError(f"shock kind {e.kind.value!r} has no domain mechanics yet")
        return schedule

    def _apply_capacity_hits(self) -> None:
        """Capacity hits starting this cycle (D2): after the previous cycle's degradation,
        before this cycle's observations and auction. Max capacity is untouched; m < 1 keeps
        a positive capacity positive, so a hit alone cannot cause system death."""
        for e in self._hits_by_cycle.get(self.cycle_counter, ()):
            p = must_get(self.producing_agents, e.target.producer)
            for t in e.target.capacity_types:
                p.capacities[t] -= capacity_hit_removal(p.capacities[t], e.magnitude)

    def _snapshot_end_capacities(self) -> tuple[CapacitySnapshot, ...]:
        return tuple(
            CapacitySnapshot(pid, dict(p.capacities), dict(p.max_capacities))
            for pid, p in self.producing_agents.items()
        )

    def _snapshot_producer_infos(self) -> dict[ProducerId, ProducerInfo]:
        return {pid: p.info() for pid, p in self.producing_agents.items()}

    def _record_production(self, kind: str, pid: ProducerId, bid: Bid) -> None:
        """Feed one production-result bid into the orderer telemetry (EXP-0006A).
        Latency is dated from when this bundle was routed (a same-cycle complete is
        latency 0; a reservation that assembles later ages from its routing)."""
        rc = bid.routed_cycle
        record = {
            "accepted": self.telemetry.record_accepted,
            "completed": self.telemetry.record_completed,
            "failed": self.telemetry.record_failed,
            "rejected": self.telemetry.record_rejected,
        }[kind]
        record(pid, bid.product, bid.quantity, rc, self.cycle_counter)

    # --- producing agent phase ---
    def producing_agent_view(self, pid: ProducerId) -> ProducingAgentView:
        if self.state != SystemState.ORDERS_PLACEMENT:
            raise WrongStateError()
        return must_get(self.producing_agents, pid).view()

    def producer_entity_view(self, pid: ProducerId) -> ProducerEntityView:
        """The entity-factored, authoritative feature bundle for one producing agent's
        decision (ADR-0053, EXP-0010A): capacity/product/input rows + recipe relations.
        The env layer normalizes this into the entity producer observation."""
        if self.state != SystemState.ORDERS_PLACEMENT:
            raise WrongStateError()
        return build_producer_entity_view(
            must_get(self.producing_agents, pid), self.telemetry, self.process_sheets, self.orders
        )

    def producing_agent_action(self, pid: ProducerId, cmd: ProducingAgentCommand) -> None:
        if self.state != SystemState.ORDERS_PLACEMENT:
            raise WrongStateError()
        self._stream_view_cache = None  # the producer's investments/inbound may change
        p = must_get(self.producing_agents, pid)
        for r in p.handle_cmd(cmd):
            oid = self.id_gen.new()
            ps = must_get(self.process_sheets, r.product)
            if isinstance(r, ReplenishmentRequest):
                self.orders[oid] = Order.new_replenishment_order(oid, ps, r)
            else:
                self.orders[oid] = Order.new_investment_order(oid, ps, r)

    # --- ordering agent phase ---
    def ordering_agent_view(self, aid: OrderingAgentId) -> OrderingAgentView:
        if self.state != SystemState.ORDERING:
            raise WrongStateError()
        return must_get(self.ordering_agents, aid).view(self.producer_infos)

    def orderer_economy_view(self, aid: OrderingAgentId) -> OrdererEconomyView:
        """The enriched, authoritative feature bundle for one ordering agent's open
        orders (EXP-0006A). The env layer normalizes this into the observation."""
        agent = must_get(self.ordering_agents, aid)
        incoming = {oid: self.orders[oid] for oid in agent.incoming}
        return build_orderer_economy_view(
            incoming, self.producing_agents, self.telemetry, self.process_sheets, self.orders
        )

    def product_stream_view(self, aid: OrderingAgentId) -> ProductStreamView:
        """The product-stream feature bundle for one stream ordering agent (ADR-0052).
        The agent's fixed product comes from `stream_products`, so an idle stream still
        yields a well-defined view. The env layer normalizes this into the obs."""
        agent = must_get(self.ordering_agents, aid)
        incoming = {oid: self.orders[oid] for oid in agent.incoming}
        return build_product_stream_view(
            self.stream_products[aid], incoming, self.producing_agents,
            self.telemetry, self.process_sheets, self.orders, self._stream_view_cache,
        )

    def ordering_agent_action(self, aid: OrderingAgentId, cmd: OrderingAgentCommand) -> None:
        if self.state != SystemState.ORDERING:
            raise WrongStateError()
        agent = must_get(self.ordering_agents, aid)
        routed = agent.handle_cmd(cmd, self.producer_infos)
        for prod_id, subs in routed.items():
            must_get(self.producing_agents, prod_id).place_sub_orders(subs, self.cycle_counter)
            for s in subs:  # record realized routing for the orderer telemetry
                self.telemetry.record_routed(prod_id, s.product, s.quantity, self.cycle_counter)

    # --- cycle lifecycle ---
    def _emit(self) -> None:
        self.investment_fund = self.cycle_emission // 2
        if not self.consumers:
            return
        consumer_tokens = self.cycle_emission // 2 // len(self.consumers)
        for c in self.consumers.values():
            c.emit(consumer_tokens)

    def _place_consumer_orders(self) -> None:
        # Placed before start_cycle advances the counter, so these orders are first
        # played in cycle `cycle_counter + 1` (the first complete_cycle plays cycle 1).
        placement_cycle = self.cycle_counter + 1
        for c in self.consumers.values():
            for request in c.order():
                oid = self.id_gen.new()
                self.orders[oid] = Order.new_consumer_order(
                    oid, must_get(self.process_sheets, request.product), request,
                    placement_cycle=placement_cycle,
                )
                # product-level demand history (true demand, not routed load).
                self.telemetry.record_demand(request.product, request.quantity)

    def start_cycle(self) -> None:
        self.state = SystemState.ORDERS_PLACEMENT
        # Production and the telemetry fold are done; until a producer acts, no producer
        # row can change (emission and consumer orders touch neither producers nor EMAs).
        self._stream_view_cache = StreamViewCache()
        self._emit()
        self._place_consumer_orders()
        self.cycle_counter += 1
        self._apply_capacity_hits()

    def start_ordering(self) -> None:
        if self.state != SystemState.ORDERS_PLACEMENT:
            raise WrongStateError()
        distribute_investment_fund(self.orders, self.investment_fund)
        # Refresh live producer state for ordering agents (see module docstring).
        self.producer_infos = self._snapshot_producer_infos()
        for order in self.orders.values():
            if order.requires_funding():
                continue  # expired this cycle (unfunded) -> rejected in complete_cycle
            aid = order.stream_agent_id() if self.product_stream_ordering else order.agent_id()
            must_get(self.ordering_agents, aid).place_order(order.info())
        self.state = SystemState.ORDERING
        # Fresh memo for this ordering phase (see StreamViewCache for why its rows
        # cannot change before complete_cycle).
        self._stream_view_cache = StreamViewCache()

    def complete_cycle(self) -> CycleResult:
        if self.state != SystemState.ORDERING:
            raise WrongStateError()
        self._stream_view_cache = None  # production and the telemetry fold follow

        for oa in self.ordering_agents.values():
            oa.complete_cycle()

        producer_auctions: list[ProducerAuctionRecord] = []
        # producers in an outage during this played cycle (D2)
        out = self._outages_by_cycle.get(self.cycle_counter, set())
        for pid, p in self.producing_agents.items():
            capacity_at_auction = sum(p.capacities.values())  # produce() degrades after
            result = p.produce(outage=pid in out)
            producer_auctions.append(ProducerAuctionRecord(
                pid, capacity_at_auction, sum(bid.total_capacity() for bid in result.accepted),
            ))
            # attribute each production event to the cycle the bundle was routed, so
            # the orderer telemetry can age completion/failure latency (EXP-0006A).
            for bid in result.accepted:
                must_get(self.orders, bid.order_id).sub_accepted(bid.quantity, bid.tokens)
                self._record_production("accepted", pid, bid)
            for bid in result.completed:
                must_get(self.orders, bid.order_id).sub_completed(bid.quantity)
                self._record_production("completed", pid, bid)
            for bid in result.failed:
                must_get(self.orders, bid.order_id).sub_failed(bid.quantity)
                self._record_production("failed", pid, bid)
            for bid in result.rejected:
                must_get(self.orders, bid.order_id).sub_rejected(bid.quantity)
                self._record_production("rejected", pid, bid)

        cycle_score: Score = 0
        completed_ids: list[OrderId] = []
        n_cons_done = n_inv_done = n_in_progress = n_cons_rej = n_inv_rej = 0
        n_repl_done = n_repl_rej = 0
        units_cons_done = units_cons_rej = 0
        consumer_outcomes: list[ConsumerOrderOutcome] = []
        for oid, order in self.orders.items():
            # an unfunded producer order expired this cycle (budget exhausted by
            # higher-priority orders) — reject it without an auction (ADR-0025).
            score, event = order.expire() if order.requires_funding() else order.complete_cycle()
            cycle_score += score
            keep = False
            if isinstance(event, ConsumerRequestCompleted):
                n_cons_done += 1
                units_cons_done += order.quantity  # whole order delivered (ADR-0025)
                consumer_outcomes.append(_consumer_outcome(order, event.request, completed=True))
            elif isinstance(event, InvestmentRequestCompleted):
                n_inv_done += 1
                self.producing_agents[event.request.producer_id].investment_completed(event.request)
            elif isinstance(event, ReplenishmentRequestCompleted):
                n_repl_done += 1
                self.producing_agents[event.request.producer_id].replenishment_completed(event.request)
            elif isinstance(event, ConsumerRequestRejected):
                n_cons_rej += 1
                units_cons_rej += order.quantity  # whole order's demand went unmet
                consumer_outcomes.append(_consumer_outcome(order, event.request, completed=False))
                self.telemetry.record_consumer_rejected(order.product, order.quantity)
                self.consumers[event.request.consumer_id].emit(event.remaining)
            elif isinstance(event, InvestmentRequestRejected):
                n_inv_rej += 1
                self.producing_agents[event.request.producer_id].investment_rejected(event.request)
            elif isinstance(event, ReplenishmentRequestRejected):
                n_repl_rej += 1
                self.producing_agents[event.request.producer_id].replenishment_rejected(event.request)
            elif isinstance(event, OrderStillProcessing):
                n_in_progress += 1
                keep = True
            else:
                raise RuntimeError("unsupported order event")
            if not keep:
                completed_ids.append(oid)

        removed = set(completed_ids)
        for oid in completed_ids:
            del self.orders[oid]

        # A composite reservation can outlive its parent order: the order's TTL may
        # fire while the reservation still waits for inputs. That accepted capacity
        # will never deliver, so surface it as failed production (the failure
        # pressure the enriched orderer needs, EXP-0006A) before dropping it —
        # otherwise the next produce() surfaces a bid for a deleted order and
        # must_get raises (ADR-0025). Record before the fold so it lands this cycle.
        if removed:
            for pid, p in self.producing_agents.items():
                survivors: list[_Reservation] = []
                for r in p.reservations:
                    if r.bid.order_id in removed:
                        self._record_production("failed", pid, r.bid)
                    else:
                        survivors.append(r)
                p.reservations = survivors

        # fold this cycle's telemetry accumulators into their EMAs (once per cycle).
        self.telemetry.end_cycle()

        end_capacities = self._snapshot_end_capacities()  # before the next cycle's shocks
        self.start_cycle()
        return CycleResult(
            cycle_score, n_cons_done, n_inv_done, n_in_progress, n_cons_rej, n_inv_rej,
            n_repl_done, n_repl_rej,
            consumer_completed_units=units_cons_done,
            consumer_rejected_units=units_cons_rej,
            consumer_outcomes=tuple(consumer_outcomes),
            producer_auctions=tuple(producer_auctions),
            end_capacities=end_capacities,
        )

    def is_dead(self) -> bool:
        """System death (ADR-0025/0028/0030): a product is permanently unmakeable
        when no producer that *declares* it (`produces`) still holds every capacity
        type its recipe requires. Since `produces` is authoritative, capacity is not
        interchangeable across owners — a type held by a producer that does not
        declare the product cannot supply it. The episode terminates the moment any
        process-sheet product becomes unmakeable."""
        for product, ps in self.process_sheets.items():
            makeable = any(
                all(p.capacities.get(t, 0) > 0 for t in ps.require)
                for p in self.producing_agents.values()
                if product in p.produces
            )
            if not makeable:
                return True
        return False

    # --- introspection ---
    def get_producer_infos(self) -> dict[ProducerId, ProducerInfo]:
        return self.producer_infos

    def get_ordering_agent_ids(self) -> list[OrderingAgentId]:
        return list(self.ordering_agents.keys())
