"""Producing agent: owns one or more capacity types, runs the atomic bundle
auction, handles investment. Track B redesign (ADR-0025).

The auction is **atomic and single-cycle**: a bundle bid clears only if every
required capacity type fits *this* cycle. There is no multi-cycle carry-over (the
old single-type `in_progress` booking is removed); a bundle that does not fit is
rejected. Cut-off price stays a single scalar = the lowest accepted bid price.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .common import (
    Capacity,
    CapacityType,
    CapacityUnitPrice,
    OrderId,
    Product,
    ProducerId,
    Tokens,
    UNDEFINED_PRICE,
    f32,
)
from .config import ProcessSheet, ProducingAgentConfig
from .requests import InvestmentRequest, InvestmentType, ReplenishmentRequest

# A composite reservation waits at most this many cycles for its inputs before
# failing (rejected, −5). The default; a producer's `reservation_ttl` follows the world's
# order TTL (`Configuration.ttl`, task 7.16), so the two lifetimes stay aligned (ADR-0025).
RESERVATION_TTL = 2

# Why a bid was rejected or a reservation failed (add-m3-development-audit D2, "Telemetry").
# DIAGNOSTIC ONLY: `outage` is derived from hidden shock state that no agent observes, so no
# policy and no teacher fork may read these (design D4); they exist for the routing trace.
REJECT_OUTAGE = "outage"  # the producer was in an outage this cycle
REJECT_NO_FIT = "no_fit"  # needs more of some type than the auction-start budget holds
REJECT_DISPLACED = "displaced"  # fits the auction-start budget, not what higher bids left
FAIL_DROPPED = "dropped"  # reservation lost re-validation against degraded capacity
FAIL_TTL = "ttl"  # reservation (or its order) expired waiting for inputs
REJECT_DECLINED = "declined"  # the producer's acceptance policy declined it (ADR-0061)


@dataclass
class Bid:
    """A whole-bundle bid: the capacity of every type the sub-order requires, the
    tokens backing it, and the order it belongs to (ADR-0025). Composite bids also
    carry the product and the total input-product units assembly consumes."""

    capacities: dict[CapacityType, Capacity]
    tokens: Tokens
    order_id: OrderId
    product: Product = 0
    inputs: dict[Product, int] = field(default_factory=dict)
    quantity: int = 1  # units of `product` this sub-order fulfils (ADR-0025)
    # cycle this bundle was routed (EXP-0006A). Travels on the bid so production
    # telemetry can age completion/failure latency per bid — a same-order remainder
    # re-routed in a later cycle gets its own bid and cannot overwrite this one.
    routed_cycle: int = 0
    # the parent order's remaining lifetime when routed (SubOrder.remaining_ttl; ADR-0061):
    # cycles it survives after the routing cycle; read by no existing path
    remaining_ttl: int | None = None

    def is_composite(self) -> bool:
        return bool(self.inputs)

    def total_capacity(self) -> Capacity:
        return sum(self.capacities.values())

    def capacity_unit_price(self) -> CapacityUnitPrice:
        return f32(f32(self.tokens) / f32(self.total_capacity()))


@dataclass
class SubOrder:
    """A routed slice of an order handed to one producer (ADR-0029): the product,
    how many units, the tokens backing it, and the order it belongs to. It carries
    **no capacity** — the producer expands it into an internal `Bid` using its own
    process sheet (`place_sub_orders`). The ordering side never sees capacity."""

    order_id: OrderId
    product: Product
    quantity: int
    tokens: Tokens
    # the parent order's remaining lifetime at routing (`OrderInfo.remaining_ttl`; ADR-0061),
    # own-order information the producer may see; additive, None when not set
    remaining_ttl: int | None = None


@dataclass
class _Reservation:
    """A composite bid that cleared the auction and reserved its capacity, waiting
    for its inputs to arrive in stock (ADR-0025). Under an acceptance policy (ADR-0061) an
    accepted bid that does not fit yet is a *pending* claim of the same shape, held in
    `ProducingAgent.pending` without capacity until it fits; `priority` orders execution."""

    bid: Bid
    cycles_waited: int = 0
    priority: float = 0.0


@dataclass(frozen=True)
class AcceptanceDecision:
    """One routed bid's acceptance (ADR-0061): accept or decline, an execution priority
    (higher first; ties by price) and, for a decline, its recorded reason."""

    accept: bool
    priority: float = 0.0
    reason: str = REJECT_DECLINED


@dataclass(frozen=True)
class AcceptanceView:
    """What an acceptance policy sees at the decision point (ADR-0061): the producer's own
    state only. `bids` are this cycle's routed bids, price-descending (the order decisions
    are returned in); `available` is the capacity left after carried reservations."""

    producer_id: ProducerId
    bids: tuple[Bid, ...]
    available: dict[CapacityType, Capacity]
    capacities: dict[CapacityType, Capacity]
    held: tuple[Bid, ...]
    pending: tuple[Bid, ...]
    inventory: dict[Product, int]
    outstanding_inbound: dict[Product, int]
    ttl: int
    cut_off_price: CapacityUnitPrice


AcceptanceRule = Callable[[AcceptanceView], Sequence[AcceptanceDecision]]


@dataclass
class ProducerInfo:
    id: ProducerId
    capacities: dict[CapacityType, Capacity]  # current, per type
    max_capacities: dict[CapacityType, Capacity]
    cut_off_price: CapacityUnitPrice
    # products this producer is eligible to make (ADR-0028) — the ordering side's
    # authoritative eligibility signal, no longer derived from capacity ownership.
    produces: frozenset[Product] = frozenset()

    def owns(self, t: CapacityType) -> bool:
        return t in self.capacities


@dataclass
class ProducingAgentCommand:
    do_restoration: bool
    do_upgrade: bool
    do_inventory_expansion: bool = False
    # input product -> units to buy this cycle (ADR-0025). Zero/absent = no order.
    replenishments: dict[Product, int] = field(default_factory=dict)


@dataclass
class ProductionResult:
    accepted: list[Bid]  # cleared the auction THIS cycle (tokens burnt now)
    completed: list[Bid]  # delivered THIS cycle (same-cycle or reservation assembled)
    failed: list[Bid]  # a prior acceptance whose reservation dropped/TTL'd this cycle
    rejected: list[Bid]  # bid that never cleared the auction (tokens stay available)
    # Diagnostic-only reasons aligned index for index with `rejected` / `failed` (REJECT_* /
    # FAIL_* above); additive, never read by the simulation or by any agent.
    rejection_reasons: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class ProducingAgentView:
    id: ProducerId
    capacities: dict[CapacityType, Capacity]
    max_capacities: dict[CapacityType, Capacity]
    requested_capacity: Capacity
    degradation: dict[CapacityType, Capacity]  # per-type degradation this cycle
    upgrade: Capacity
    restoration: Capacity
    upgrade_running: bool
    restoration_running: bool
    # inventory subsystem (ADR-0025): per-input stock + the joint headroom the env
    # decode apportions a cycle's replenishment within.
    inventory: dict[Product, int]
    inventory_limit: int
    inventory_units: int
    headroom: int
    inventory_expansion_running: bool
    has_inventory_expansion: bool
    # units ordered but not yet delivered, per input product (ADR-0041): the obs
    # signal a policy needs to avoid re-ordering in-flight stock when priming a
    # multi-hop pipeline. Defaulted so existing view constructors stay valid.
    outstanding_inbound: dict[Product, int] = field(default_factory=dict)


class ProducingAgent:
    def __init__(
        self, config: ProducingAgentConfig, sheets: dict[Product, ProcessSheet] | None = None,
        reservation_ttl: int = RESERVATION_TTL,
    ) -> None:
        self.id = config.id
        self.reservation_ttl = reservation_ttl  # = the world's order TTL (task 7.16)
        self.capacities: dict[CapacityType, Capacity] = dict(config.capacities)
        self.max_capacities: dict[CapacityType, Capacity] = dict(config.capacities)
        self.produces: frozenset[Product] = config.produces  # ADR-0028
        # process sheets for the products this producer makes (ADR-0029) — the
        # producer owns recipe knowledge and expands sub-orders into bids itself.
        self.sheets: dict[Product, ProcessSheet] = sheets or {}
        self.degradation = config.degradation
        self.restoration = config.restoration
        self.upgrade = config.upgrade
        self.inventory_expansion = config.inventory_expansion
        self.replenish_cost_per_unit = config.replenish_cost_per_unit
        # producer state
        self.bids: list[Bid] = []
        self.requested_capacity: Capacity = 0
        # per-type requested capacity (ReqPrev, ADR-0053): the entity producer view
        # needs per-type pressure, not just the scalar total. Committed post-auction.
        self.requested_capacity_by_type: dict[CapacityType, Capacity] = {}
        self.funds: Tokens = 0
        self.cut_off_price: CapacityUnitPrice = UNDEFINED_PRICE
        # inventory subsystem (ADR-0025): input-product stock + unit-count limit,
        # and capacity reserved by composites waiting for their inputs.
        self.inventory: dict[Product, int] = {}
        self.inventory_limit: int = config.inventory_limit
        self.reservations: list[_Reservation] = []
        # ADR-0061: accepted claims not yet holding capacity (empty without an acceptance rule)
        # and the acceptance rule itself (None = the ADR-0025 auction, unchanged)
        self.pending: list[_Reservation] = []
        self.acceptance: AcceptanceRule | None = None
        # diagnostic counters of the acceptance path (reported, never observed)
        self.acceptance_stats: dict[str, int] = {}
        # units ordered via replenishment but not yet delivered — claim inventory
        # headroom so the env's joint headroom H stays accurate (ADR-0025).
        self.outstanding_inbound: dict[Product, int] = {}
        # consumer (investment) state
        self.upgrade_running: bool = False
        self.restoration_running: bool = False
        self.inventory_expansion_running: bool = False
        self.cmd_handled: bool = False

    def owns(self, t: CapacityType) -> bool:
        return t in self.capacities

    def info(self) -> ProducerInfo:
        return ProducerInfo(
            self.id, dict(self.capacities), dict(self.max_capacities), self.cut_off_price, self.produces
        )

    def degradation_for(self, t: CapacityType) -> Capacity:
        return int(math.ceil(self.degradation * self.max_capacities[t] / 100))

    def view(self) -> ProducingAgentView:
        return ProducingAgentView(
            self.id,
            dict(self.capacities),
            dict(self.max_capacities),
            self.requested_capacity,
            {t: self.degradation_for(t) for t in self.capacities},
            self.upgrade.increases,
            self.restoration.restores,
            self.upgrade_running,
            self.restoration_running,
            dict(self.inventory),
            self.inventory_limit,
            self.inventory_units(),
            self.headroom(),
            self.inventory_expansion_running,
            self.inventory_expansion is not None,
            dict(self.outstanding_inbound),
        )

    def place_bids(self, bids: list[Bid]) -> None:
        for b in bids:
            for t in b.capacities:
                if t not in self.capacities:
                    raise RuntimeError(f"producer {self.id} does not own capacity type {t}")
        self.bids.extend(bids)

    def place_sub_orders(self, subs: list[SubOrder], routed_cycle: int = 0) -> None:
        """Accept routed sub-orders and expand each into an internal bid using this
        producer's own process sheet (ADR-0029). Capacity and input requirements
        scale with the sub-order quantity — recipe knowledge stays producer-side.
        `routed_cycle` stamps each bid for latency telemetry (EXP-0006A)."""
        self.place_bids([self._to_bid(s, routed_cycle) for s in subs])

    def _to_bid(self, sub: SubOrder, routed_cycle: int = 0) -> Bid:
        if sub.product not in self.sheets:
            raise RuntimeError(f"producer {self.id} has no process sheet for product {sub.product}")
        ps = self.sheets[sub.product]
        return Bid(
            capacities={t: c * sub.quantity for t, c in ps.require.items()},
            tokens=sub.tokens,
            order_id=sub.order_id,
            product=sub.product,
            inputs={p: n * sub.quantity for p, n in ps.inputs.items()},
            quantity=sub.quantity,
            routed_cycle=routed_cycle,
            remaining_ttl=sub.remaining_ttl,
        )

    def inventory_units(self) -> int:
        return sum(self.inventory.values())

    def outstanding_inbound_units(self) -> int:
        return sum(self.outstanding_inbound.values())

    def reserved_capacity(self) -> dict[CapacityType, Capacity]:
        """Capacity held by live composite reservations, per type — the amount
        unavailable to new bids this cycle (ADR-0025). Available-for-bids capacity
        is `capacities[t] − reserved_capacity()[t]`."""
        out: dict[CapacityType, Capacity] = {}
        for r in self.reservations:
            for t, c in r.bid.capacities.items():
                out[t] = out.get(t, 0) + c
        return out

    def headroom(self) -> int:
        """Inventory units still orderable: limit − held − outstanding inbound
        (ADR-0025). The env apportions a cycle's replenishment jointly within it."""
        return self.inventory_limit - self.inventory_units() - self.outstanding_inbound_units()

    def handle_cmd(
        self, cmd: ProducingAgentCommand
    ) -> list[InvestmentRequest | ReplenishmentRequest]:
        if self.cmd_handled:
            raise RuntimeError("command is handled already")
        if self.upgrade_running and cmd.do_upgrade:
            raise RuntimeError("upgrade is running")
        if self.restoration_running and cmd.do_restoration:
            raise RuntimeError("restorations is running")
        if self.inventory_expansion_running and cmd.do_inventory_expansion:
            raise RuntimeError("expansion is running")
        if cmd.do_inventory_expansion and self.inventory_expansion is None:
            # validate before any state mutation: a mixed command must not leak
            # outstanding-inbound headroom when a later toggle is invalid (ADR-0025).
            raise RuntimeError("producer has no inventory expansion configured")
        requests: list[InvestmentRequest | ReplenishmentRequest] = []
        for product, qty in cmd.replenishments.items():
            if qty <= 0:
                continue
            cost = None if self.replenish_cost_per_unit is None else self.replenish_cost_per_unit * qty
            requests.append(ReplenishmentRequest(self.id, product, qty, self.cut_off_price, cost))
            self.outstanding_inbound[product] = self.outstanding_inbound.get(product, 0) + qty
        if cmd.do_upgrade:
            requests.append(
                InvestmentRequest(
                    self.id, InvestmentType.UPGRADE, self.upgrade.require, self.cut_off_price, self.upgrade.cost
                )
            )
            self.upgrade_running = True
        if cmd.do_restoration:
            requests.append(
                InvestmentRequest(
                    self.id, InvestmentType.RESTORATION, self.restoration.require, self.cut_off_price,
                    self.restoration.cost,
                )
            )
            self.restoration_running = True
        if cmd.do_inventory_expansion:
            assert self.inventory_expansion is not None  # validated above
            requests.append(
                InvestmentRequest(
                    self.id, InvestmentType.INVENTORY_EXPANSION, self.inventory_expansion.require,
                    self.cut_off_price, self.inventory_expansion.cost,
                )
            )
            self.inventory_expansion_running = True
        self.cmd_handled = True
        return requests

    def investment_completed(self, request: InvestmentRequest) -> None:
        if request.producer_id != self.id:
            raise RuntimeError("not found")
        if request.type == InvestmentType.RESTORATION:
            if not self.restoration_running:
                raise RuntimeError("no restoration running")
            self.restoration_running = False
            t = self.restoration.type
            self.capacities[t] = min(self.max_capacities[t], self.capacities[t] + self.restoration.restores)
        elif request.type == InvestmentType.UPGRADE:
            if not self.upgrade_running:
                raise RuntimeError("no upgrades running")
            self.upgrade_running = False
            t = self.upgrade.type
            self.max_capacities[t] += self.upgrade.increases
            self.capacities[t] += self.upgrade.increases
        elif request.type == InvestmentType.INVENTORY_EXPANSION:
            assert self.inventory_expansion is not None
            if not self.inventory_expansion_running:
                raise RuntimeError("no expansion running")
            self.inventory_expansion_running = False
            self.inventory_limit += self.inventory_expansion.increases
        else:
            raise RuntimeError("unsupported")

    def investment_rejected(self, request: InvestmentRequest) -> None:
        if request.producer_id != self.id:
            raise RuntimeError("not found")
        if request.type == InvestmentType.RESTORATION:
            if not self.restoration_running:
                raise RuntimeError("no restoration running")
            self.restoration_running = False
        elif request.type == InvestmentType.UPGRADE:
            if not self.upgrade_running:
                raise RuntimeError("no upgrades running")
            self.upgrade_running = False
        elif request.type == InvestmentType.INVENTORY_EXPANSION:
            if not self.inventory_expansion_running:
                raise RuntimeError("no expansion running")
            self.inventory_expansion_running = False
        else:
            raise RuntimeError("unsupported")

    def replenishment_completed(self, request: ReplenishmentRequest) -> None:
        if request.producer_id != self.id:
            raise RuntimeError("not found")
        self.inventory[request.product] = self.inventory.get(request.product, 0) + request.quantity
        self._clear_inbound(request)

    def replenishment_rejected(self, request: ReplenishmentRequest) -> None:
        if request.producer_id != self.id:
            raise RuntimeError("not found")
        self._clear_inbound(request)

    def _clear_inbound(self, request: ReplenishmentRequest) -> None:
        remaining = self.outstanding_inbound.get(request.product, 0) - request.quantity
        if remaining > 0:
            self.outstanding_inbound[request.product] = remaining
        else:
            self.outstanding_inbound.pop(request.product, None)

    # --- inventory / assembly helpers ---
    def _produce_by_policy(self, outage: bool, available: dict[CapacityType, Capacity],
                           still: list[_Reservation], accepted: list[Bid], completed: list[Bid],
                           failed: list[Bid], rejected: list[Bid], failure_reasons: list[str],
                           rejection_reasons: list[str],
                           cut_off_price: CapacityUnitPrice) -> ProductionResult:
        """ADR-0061 steps 3b-5: pending claims take capacity in priority order (or age and
        fail at TTL); the acceptance rule decides this cycle's bids; accepted bids execute in
        priority order as the budget allows, the rest wait as pending claims. Tokens burn at
        acceptance; the cut-off is the lowest accepted price this cycle, carried otherwise."""
        stats = self.acceptance_stats

        def bump(key: str, n: int = 1) -> None:
            stats[key] = stats.get(key, 0) + n

        def execute(claims: list[_Reservation], keep: list[_Reservation], ran: str) -> None:
            """Claims that fit take capacity (complete, or hold it waiting for inputs); `ran`
            counts them (pending claims from earlier cycles vs this cycle's accepted bids)."""
            for r in sorted(claims, key=lambda r: (-r.priority, -r.bid.capacity_unit_price())):
                if not outage and self._fits(available, r.bid.capacities):
                    self._subtract(available, r.bid.capacities)
                    bump(ran)
                    if r.bid.is_composite() and not self._has_inputs(r.bid):
                        still.append(r)
                    else:
                        if r.bid.is_composite():
                            self._consume_inputs(r.bid)
                        completed.append(r.bid)
                else:
                    keep.append(r)

        # 3b. claims accepted in earlier cycles (no capacity held yet)
        waiting: list[_Reservation] = []
        execute(self.pending, waiting, "pending_ran_later")
        pending: list[_Reservation] = []
        for r in waiting:
            r.cycles_waited += 1
            if r.cycles_waited >= self.reservation_ttl:
                failed.append(r.bid)
                failure_reasons.append(FAIL_TTL)
                bump("pending_failed_ttl")
            else:
                pending.append(r)

        # 4. the acceptance decision over this cycle's bids (price-descending)
        requested: Capacity = 0
        requested_by_type: dict[CapacityType, Capacity] = {}
        self.bids.sort(key=lambda b: b.capacity_unit_price(), reverse=True)
        for bid in self.bids:
            requested += bid.total_capacity()
            for t, c in bid.capacities.items():
                requested_by_type[t] = requested_by_type.get(t, 0) + c
        decisions = list(self.acceptance(AcceptanceView(
            self.id, tuple(self.bids), dict(available), dict(self.capacities),
            tuple(r.bid for r in still), tuple(r.bid for r in pending), dict(self.inventory),
            dict(self.outstanding_inbound), self.reservation_ttl, self.cut_off_price)))
        if len(decisions) != len(self.bids):
            raise ValueError(f"acceptance returned {len(decisions)} decisions for "
                             f"{len(self.bids)} bids")
        funds: Tokens = 0
        new: list[_Reservation] = []
        declined: list[Bid] = []
        lowest: CapacityUnitPrice | None = None
        for bid, d in zip(self.bids, decisions):
            if outage or not d.accept:
                rejected.append(bid)
                rejection_reasons.append(REJECT_OUTAGE if outage else d.reason)
                if not outage:
                    declined.append(bid)
                continue
            funds += bid.tokens
            price = bid.capacity_unit_price()
            lowest = price if lowest is None or price < lowest else lowest
            accepted.append(bid)
            new.append(_Reservation(bid, 0, float(d.priority)))
        if lowest is not None:
            cut_off_price = lowest
        bump("declined", len(declined))
        bump("declined_below_cutoff",
             sum(1 for b in declined if lowest is not None and b.capacity_unit_price() < lowest))
        bump("accepted", len(new))
        before = len(pending)
        execute(new, pending, "ran_same_cycle")
        bump("accepted_pending", len(pending) - before)

        # 5. commit next state, as the auction path
        for t in self.capacities:
            self.capacities[t] = max(0, self.capacities[t] - self.degradation_for(t))
        self.bids = []
        self.reservations = still
        self.pending = pending
        self.requested_capacity = requested
        self.requested_capacity_by_type = requested_by_type
        self.funds = funds
        self.cut_off_price = cut_off_price
        self.cmd_handled = False
        return ProductionResult(
            accepted, completed, failed, rejected,
            rejection_reasons=rejection_reasons, failure_reasons=failure_reasons,
        )

    def _has_inputs(self, bid: Bid) -> bool:
        return all(self.inventory.get(p, 0) >= q for p, q in bid.inputs.items())

    def _consume_inputs(self, bid: Bid) -> None:
        for p, q in bid.inputs.items():
            self.inventory[p] -= q

    @staticmethod
    def _fits(budget: dict[CapacityType, Capacity], caps: dict[CapacityType, Capacity]) -> bool:
        return all(budget.get(t, 0) >= c for t, c in caps.items())

    @staticmethod
    def _subtract(budget: dict[CapacityType, Capacity], caps: dict[CapacityType, Capacity]) -> None:
        for t, c in caps.items():
            budget[t] -= c

    def _revalidate_reservations(self) -> tuple[list[_Reservation], list[_Reservation]]:
        """Re-run the atomic price-greedy auction over live reservations against
        current (post-degradation) capacity. Pricey-first acceptance means a
        reservation competing for a deficient type drops cheapest-first; one not
        touching the deficient type always survives (ADR-0025)."""
        ordered = sorted(self.reservations, key=lambda r: r.bid.capacity_unit_price(), reverse=True)
        budget: dict[CapacityType, Capacity] = dict(self.capacities)
        survivors: list[_Reservation] = []
        dropped: list[_Reservation] = []
        for r in ordered:
            if self._fits(budget, r.bid.capacities):
                self._subtract(budget, r.bid.capacities)
                survivors.append(r)
            else:
                dropped.append(r)
        return survivors, dropped

    def produce(self, outage: bool = False) -> ProductionResult:
        """One cycle of production (ADR-0025). Order: re-validate live reservations
        against post-degradation capacity (drop → fail); reserved capacity blocks
        the per-cycle budget; assemble reservations whose inputs arrived; fail
        those that hit TTL; then run the atomic price-greedy auction over the
        remaining budget — composites without inputs reserve and wait, everything
        else completes.

        `outage` (producer outage, ADR-0055, add-shocks-i-drift D2): the producer
        assembles no reservation (each ages toward its TTL as if its inputs were
        missing) and accepts no bid (all rejected, requested capacity still recorded,
        cut-off price kept); re-validation and degradation run as usual. Its own
        purchases and deliveries are untouched: they are resolved by the system."""
        accepted: list[Bid] = []
        completed: list[Bid] = []
        failed: list[Bid] = []
        rejected: list[Bid] = []
        failure_reasons: list[str] = []
        rejection_reasons: list[str] = []
        cut_off_price = self.cut_off_price
        funds: Tokens = 0
        requested: Capacity = 0

        # 1. re-validate carried reservations; dropped ones fail (already accepted
        # in an earlier cycle — surfaced as failed, not accepted again).
        survivors, dropped = self._revalidate_reservations()
        for r in dropped:
            failed.append(r.bid)
            failure_reasons.append(FAIL_DROPPED)

        # 2. survivors hold the per-cycle budget (the intended blocking pressure)
        available: dict[CapacityType, Capacity] = dict(self.capacities)
        for r in survivors:
            self._subtract(available, r.bid.capacities)

        # 3. assemble survivors whose inputs are now in stock; else age / TTL-fail.
        # Survivors were accepted in a prior cycle, so they never re-enter
        # `accepted`; they only surface here when they resolve (completed/failed).
        still: list[_Reservation] = []
        for r in survivors:
            if not outage and self._has_inputs(r.bid):
                self._consume_inputs(r.bid)
                completed.append(r.bid)
            else:
                r.cycles_waited += 1
                if r.cycles_waited >= self.reservation_ttl:
                    # TTL hit: the reservation fails and frees its held capacity
                    # back to this cycle's budget for the new-bid auction below.
                    for t, c in r.bid.capacities.items():
                        available[t] += c
                    failed.append(r.bid)
                    failure_reasons.append(FAIL_TTL)
                else:
                    still.append(r)

        if self.acceptance is not None:
            return self._produce_by_policy(
                outage, available, still, accepted, completed, failed, rejected,
                failure_reasons, rejection_reasons, cut_off_price)

        # 4. atomic price-greedy auction over the remaining budget. A bid that
        # clears is `accepted` (tokens burnt) this cycle, whether it completes now
        # or reserves and waits.
        requested_by_type: dict[CapacityType, Capacity] = {}
        auction_budget = dict(available)  # diagnostic: no-fit vs displaced (D2)
        self.bids.sort(key=lambda b: b.capacity_unit_price(), reverse=True)
        for bid in self.bids:
            requested += bid.total_capacity()
            for t, c in bid.capacities.items():
                requested_by_type[t] = requested_by_type.get(t, 0) + c
            if outage or not self._fits(available, bid.capacities):
                rejected.append(bid)
                rejection_reasons.append(
                    REJECT_OUTAGE if outage
                    else REJECT_NO_FIT if not self._fits(auction_budget, bid.capacities)
                    else REJECT_DISPLACED
                )
                continue
            self._subtract(available, bid.capacities)
            funds += bid.tokens
            cut_off_price = bid.capacity_unit_price()
            accepted.append(bid)
            if bid.is_composite() and not self._has_inputs(bid):
                still.append(_Reservation(bid))
            else:
                if bid.is_composite():
                    self._consume_inputs(bid)
                completed.append(bid)

        # 5. commit next state: degrade each owned type toward next cycle
        for t in self.capacities:
            self.capacities[t] = max(0, self.capacities[t] - self.degradation_for(t))
        self.bids = []
        self.reservations = still
        self.requested_capacity = requested
        self.requested_capacity_by_type = requested_by_type
        self.funds = funds
        self.cut_off_price = cut_off_price
        self.cmd_handled = False
        return ProductionResult(
            accepted, completed, failed, rejected,
            rejection_reasons=rejection_reasons, failure_reasons=failure_reasons,
        )
