"""Order lifecycle — whole-product orders (ADR-0025).

An order demands a *quantity* of a product as a whole. An ordering agent splits
that quantity across eligible producers as sub-orders; each producer reports its
sub-order completed / rejected / still-processing by unit quantity. The order is
the scoring unit — completed only when its whole quantity completes.

Scores returned by complete_cycle (0/3/1/5) are the Go emulator's penalty scores,
kept for domain parity. The RL reward is computed separately in the env layer from
the returned OrderEvent (see ADR-0002), NOT from these scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import (
    CapacityUnitPrice,
    OrderId,
    OrderingAgentId,
    Product,
    Tokens,
    from_consumer_id,
    from_producer_id,
    stream_id,
)
from .config import ProcessSheet
from .requests import ConsumerRequest, InvestmentRequest, ReplenishmentRequest

Score = int

# Maximum cycles an order stays in progress before it is rejected (ADR-0025). The
# order is placed at cycle_counter 0 and rejected once it reaches this age. Named so
# the enriched orderer observation can expose order TTL/age (EXP-0006A).
ORDER_TTL = 2


# --- Order events (domain.OrderEvent) ---
@dataclass
class ConsumerRequestCompleted:
    request: ConsumerRequest


@dataclass
class InvestmentRequestCompleted:
    request: InvestmentRequest


@dataclass
class ConsumerRequestRejected:
    remaining: Tokens
    request: ConsumerRequest


@dataclass
class InvestmentRequestRejected:
    request: InvestmentRequest


@dataclass
class ReplenishmentRequestCompleted:
    request: ReplenishmentRequest


@dataclass
class ReplenishmentRequestRejected:
    request: ReplenishmentRequest


@dataclass
class OrderStillProcessing:
    pass


OrderEvent = object


@dataclass
class OrderInfo:
    """What an ordering agent sees for an order it must distribute: the whole
    product, its quantity, and the backing tokens — no capacity (ADR-0029). The
    ordering side reasons about products and quantities only; the per-unit capacity
    bundle and input requirements are private to the producer, which expands them
    from its own process sheet at bid time. Eligibility is decided against
    `product` — a producer is eligible iff it declares it in `produces` (ADR-0028)."""

    id: OrderId
    tokens: Tokens
    product: Product
    quantity: int


class Order:
    def __init__(
        self,
        id: OrderId,
        tokens: Tokens,
        product: Product,
        quantity: int,
        consumer_request: ConsumerRequest | None,
        investment_request: InvestmentRequest | None,
        funded: bool,
        replenishment_request: ReplenishmentRequest | None = None,
    ) -> None:
        self.id = id
        self.tokens = tokens
        self.product = product
        self.quantity = quantity
        self.consumer_request = consumer_request
        self.investment_request = investment_request
        self.replenishment_request = replenishment_request
        self.funded = funded
        self.cycle_counter = 0
        # Cumulative sub-order accounting (ADR-0025). A sub-order's tokens are
        # burnt the cycle it clears the auction (accepted), never refunded — even
        # if its reservation later fails. Only never-accepted quantity (and its
        # still-available tokens) is redistributed next cycle.
        self.accepted_qty = 0  # quantity that ever cleared the auction
        self.completed_qty = 0  # quantity delivered
        self.failed_qty = 0  # accepted quantity whose reservation then failed
        self.burnt_tokens: Tokens = 0  # tokens of accepted sub-orders (out of circulation)

    @classmethod
    def new_investment_order(cls, id: OrderId, ps: ProcessSheet, request: InvestmentRequest) -> "Order":
        # Investment orders are for a single means-of-production unit.
        return cls(id, 0, ps.product, 1, None, request, False)

    @classmethod
    def new_replenishment_order(cls, id: OrderId, ps: ProcessSheet, request: ReplenishmentRequest) -> "Order":
        # A producer buying `quantity` units of an input product; unfunded until
        # the central fund backs it (priority 1), like an investment order.
        return cls(
            id, 0, ps.product, request.quantity, None, None, False,
            replenishment_request=request,
        )

    @classmethod
    def new_consumer_order(cls, id: OrderId, ps: ProcessSheet, request: ConsumerRequest) -> "Order":
        return cls(id, request.tokens, ps.product, request.quantity, request, None, True)

    # --- funding (unchanged contract; used by distribute_investment_fund) ---
    def _must_be_funded(self) -> None:
        if self.requires_funding():
            raise RuntimeError("funding required")

    def requires_funding(self) -> bool:
        return not self.funded

    def producer_id(self):
        """The producer this order funds (investment or replenishment), else None
        — the unit `distribute_investment_fund` groups by (ADR-0025)."""
        if self.investment_request is not None:
            return self.investment_request.producer_id
        if self.replenishment_request is not None:
            return self.replenishment_request.producer_id
        return None

    def funding_target(self) -> Tokens | None:
        """Tokens the fund fills this order to before advancing to the next
        priority (ADR-0027). None = unbounded (take the whole remaining budget)."""
        if self.investment_request is not None:
            return self.investment_request.cost
        if self.replenishment_request is not None:
            return self.replenishment_request.cost
        return None

    def cut_off_price(self) -> CapacityUnitPrice:
        if self.investment_request is not None:
            return self.investment_request.cut_off_price
        if self.replenishment_request is not None:
            return self.replenishment_request.cut_off_price
        raise RuntimeError("not a producer-funded order")

    def fund(self, t: Tokens) -> None:
        if not self.requires_funding():
            raise RuntimeError("not in funding status")
        if self.investment_request is None and self.replenishment_request is None:
            raise RuntimeError("not a producer-funded order")
        self.tokens = t
        self.funded = True

    # --- distribution-facing view ---
    def unresolved_quantity(self) -> int:
        """Quantity never accepted by any producer — the only part re-biddable
        next cycle (accepted-but-failed quantity keeps no tokens, ADR-0025)."""
        return self.quantity - self.accepted_qty

    def available_tokens(self) -> Tokens:
        return self.tokens - self.burnt_tokens

    def info(self) -> OrderInfo:
        self._must_be_funded()
        return OrderInfo(
            self.id, self.available_tokens(), self.product, self.unresolved_quantity()
        )

    def agent_id(self) -> OrderingAgentId:
        self._must_be_funded()
        if self.investment_request is not None:
            return from_producer_id(self.investment_request.producer_id)
        if self.replenishment_request is not None:
            return from_producer_id(self.replenishment_request.producer_id)
        assert self.consumer_request is not None
        return from_consumer_id(self.consumer_request.consumer_id)

    def stream_agent_id(self) -> OrderingAgentId:
        """Product-stream ordering-agent id (ADR-0052): the source's per-product
        stream this order routes to. Source is the same as `agent_id`; the product
        disambiguates which stream agent holds it."""
        return stream_id(self.agent_id(), self.product)

    # --- sub-order outcomes reported by producers ---
    def sub_accepted(self, quantity: int, tokens: Tokens) -> None:
        """A sub-order cleared the auction this cycle: its tokens are burnt now
        (whether it completes immediately or reserves and waits)."""
        self._must_be_funded()
        self.accepted_qty += quantity
        self.burnt_tokens += tokens

    def sub_completed(self, quantity: int) -> None:
        self._must_be_funded()
        self.completed_qty += quantity

    def sub_failed(self, quantity: int) -> None:
        """An accepted reservation dropped/TTL'd. Tokens stay burnt (already
        spent at acceptance); the quantity will not be re-bid."""
        self._must_be_funded()
        self.failed_qty += quantity

    def sub_rejected(self, quantity: int) -> None:
        # A bid that never cleared the auction: nothing burnt, nothing accepted —
        # the quantity stays unresolved and is re-bid next cycle.
        self._must_be_funded()

    def complete_cycle(self) -> tuple[Score, OrderEvent]:
        self._must_be_funded()
        if self.completed_qty == self.quantity:
            if self.consumer_request is not None:
                return 0, ConsumerRequestCompleted(self.consumer_request)
            if self.replenishment_request is not None:
                return 0, ReplenishmentRequestCompleted(self.replenishment_request)
            assert self.investment_request is not None
            return 0, InvestmentRequestCompleted(self.investment_request)

        reserved_pending = self.accepted_qty - self.completed_qty - self.failed_qty
        # all quantity accounted (none left to bid, none still reserved) but not
        # all delivered -> the failed portion sinks the whole order.
        if reserved_pending == 0 and self.unresolved_quantity() == 0:
            return self._reject(3)

        if self.cycle_counter == ORDER_TTL:
            return self._reject(5)

        self.cycle_counter += 1
        return 1, OrderStillProcessing()

    def expire(self) -> tuple[Score, OrderEvent]:
        """A producer order that never won funding this cycle (budget exhausted by
        higher-priority orders). Rejected without going through the auction — it was
        never placed (ADR-0025)."""
        return self._reject(5)

    def _reject(self, score: Score) -> tuple[Score, OrderEvent]:
        if self.consumer_request is not None:
            return score, ConsumerRequestRejected(self.available_tokens(), self.consumer_request)
        if self.replenishment_request is not None:
            return score, ReplenishmentRequestRejected(self.replenishment_request)
        assert self.investment_request is not None
        return score, InvestmentRequestRejected(self.investment_request)
