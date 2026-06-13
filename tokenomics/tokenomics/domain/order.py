"""Order lifecycle. Port of emulation/domain/order.go.

Scores returned by complete_cycle (0/3/1/5) are the Go emulator's penalty
scores, kept for domain parity. The RL reward is computed separately in the env
layer from the returned OrderEvent (see ADR-0002), NOT from these scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .common import (
    Capacity,
    CapacityType,
    CapacityUnitPrice,
    OrderId,
    OrderingAgentId,
    Tokens,
    WrongStateError,
    from_consumer_id,
    from_producer_id,
)
from .config import ProcessSheet
from .requests import ConsumerRequest, InvestmentRequest


class PartStatus(IntEnum):
    UNKNOWN = 0
    PROCESSING = 1
    REJECTED = 2
    COMPLETED = 3


@dataclass
class Part:
    capacity: Capacity
    status: PartStatus = PartStatus.UNKNOWN


Score = int


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
class OrderStillProcessing:
    pass


OrderEvent = object


@dataclass
class OrderInfo:
    id: OrderId
    tokens: Tokens
    required: dict[CapacityType, Capacity]

    def fulfilled(self) -> bool:
        return len(self.required) == 0


class Order:
    def __init__(
        self,
        id: OrderId,
        tokens: Tokens,
        parts: dict[CapacityType, Part],
        consumer_request: ConsumerRequest | None,
        investment_request: InvestmentRequest | None,
        funded: bool,
    ) -> None:
        self.id = id
        self.tokens = tokens
        self.parts = parts
        self.consumer_request = consumer_request
        self.investment_request = investment_request
        self.cycle_counter = 0
        self.funded = funded

    @classmethod
    def new_investment_order(cls, id: OrderId, ps: ProcessSheet, request: InvestmentRequest) -> "Order":
        parts = {t: Part(capacity) for t, capacity in ps.require.items()}
        return cls(id, 0, parts, None, request, False)

    @classmethod
    def new_consumer_order(cls, id: OrderId, ps: ProcessSheet, request: ConsumerRequest) -> "Order":
        parts = {t: Part(capacity) for t, capacity in ps.require.items()}
        return cls(id, request.tokens, parts, request, None, True)

    def _must_be_funded(self) -> None:
        if self.requires_funding():
            raise RuntimeError("funding required")

    def info(self) -> OrderInfo:
        self._must_be_funded()
        required = {
            k: v.capacity
            for k, v in self.parts.items()
            if v.status not in (PartStatus.PROCESSING, PartStatus.COMPLETED)
        }
        return OrderInfo(self.id, self.tokens, required)

    def agent_id(self) -> OrderingAgentId:
        self._must_be_funded()
        if self.investment_request is not None:
            return from_producer_id(self.investment_request.producer_id)
        return from_consumer_id(self.consumer_request.consumer_id)

    def requires_funding(self) -> bool:
        return not self.funded

    def cut_off_price(self) -> CapacityUnitPrice:
        if self.investment_request is None:
            raise RuntimeError("not an investment order")
        return self.investment_request.cut_off_price

    def fund(self, t: Tokens) -> None:
        if not self.requires_funding():
            raise RuntimeError("not in funding status")
        if self.investment_request is None:
            raise RuntimeError("not an investment order")
        self.tokens = t
        self.funded = True

    def _get_part_status(self, ct: CapacityType) -> PartStatus:
        if ct not in self.parts:
            raise KeyError(ct)
        return self.parts[ct].status

    def _spend_tokens(self, t: Tokens) -> None:
        if self.tokens < t:
            raise RuntimeError("too few tokens left")
        self.tokens -= t

    def processing(self, ct: CapacityType, t: Tokens) -> None:
        self._must_be_funded()
        status = self._get_part_status(ct)
        if status == PartStatus.PROCESSING:
            return
        if status == PartStatus.COMPLETED:
            raise WrongStateError()
        self._spend_tokens(t)
        self.parts[ct].status = PartStatus.PROCESSING

    def completed(self, ct: CapacityType, t: Tokens) -> None:
        self._must_be_funded()
        status = self._get_part_status(ct)
        if status == PartStatus.COMPLETED:
            raise WrongStateError()
        if status != PartStatus.PROCESSING:
            self._spend_tokens(t)
        self.parts[ct].status = PartStatus.COMPLETED

    def rejected(self, ct: CapacityType) -> None:
        self._must_be_funded()
        status = self._get_part_status(ct)
        if status in (PartStatus.PROCESSING, PartStatus.COMPLETED):
            raise WrongStateError()
        self.parts[ct].status = PartStatus.REJECTED

    def complete_cycle(self) -> tuple[Score, OrderEvent]:
        self._must_be_funded()
        rejected_count = sum(1 for p in self.parts.values() if p.status == PartStatus.REJECTED)
        completed_count = sum(1 for p in self.parts.values() if p.status == PartStatus.COMPLETED)

        if completed_count == len(self.parts):
            if self.consumer_request is not None:
                return 0, ConsumerRequestCompleted(self.consumer_request)
            return 0, InvestmentRequestCompleted(self.investment_request)

        if rejected_count == len(self.parts):
            if self.consumer_request is not None:
                return 3, ConsumerRequestRejected(self.tokens, self.consumer_request)
            return 3, InvestmentRequestRejected(self.investment_request)

        if self.cycle_counter == 2:
            if self.consumer_request is not None:
                return 5, ConsumerRequestRejected(self.tokens, self.consumer_request)
            return 5, InvestmentRequestRejected(self.investment_request)

        self.cycle_counter += 1
        return 1, OrderStillProcessing()
