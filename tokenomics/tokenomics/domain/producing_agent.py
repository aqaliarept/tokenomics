"""Producing agent: owns one capacity type, runs the greedy auction, handles
investment. Port of emulation/domain/producing_agent.go.

Auction subtlety: `remaining_capacity` starts at the *current* capacity (signed
int, may go negative) and degradation is applied only to the capacity stored for
the next cycle. The in-progress booking carries an unfinished bid across cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .common import (
    Capacity,
    CapacityType,
    CapacityUnitPrice,
    OrderId,
    ProducerId,
    Tokens,
    UNDEFINED_PRICE,
    f32,
)
from .config import ProducingAgentConfig
from .requests import InvestmentRequest, InvestmentType


@dataclass
class Bid:
    capacity_type: CapacityType
    capacity: Capacity
    tokens: Tokens
    order_id: OrderId

    def capacity_unit_price(self) -> CapacityUnitPrice:
        return f32(f32(self.tokens) / f32(self.capacity))


@dataclass
class ProducerInfo:
    id: ProducerId
    capacity_type: CapacityType
    max_capacity: Capacity
    capacity: Capacity
    cut_off_price: CapacityUnitPrice


@dataclass
class _Booking:
    order_id: OrderId
    booked: Capacity
    bid: Bid


@dataclass
class ProducingAgentCommand:
    do_restoration: bool
    do_upgrade: bool


@dataclass
class ProductionResult:
    processing: list[Bid]
    completed: list[Bid]
    rejected: list[Bid]


@dataclass
class ProducingAgentView:
    id: ProducerId
    max_capacity: Capacity
    capacity: Capacity
    requested_capacity: Capacity
    degradation: Capacity
    upgrade: Capacity
    restoration: Capacity
    upgrade_running: bool
    restoration_running: bool


class ProducingAgent:
    def __init__(self, config: ProducingAgentConfig) -> None:
        self.id = config.id
        self.capacity_type = config.type
        self.degradation = config.degradation
        self.restoration = config.restoration
        self.upgrade = config.upgrade
        # producer state
        self.capacity: Capacity = config.capacity
        self.max_capacity: Capacity = config.capacity
        self.bids: list[Bid] = []
        self.in_progress: _Booking | None = None
        self.requested_capacity: Capacity = 0
        self.funds: Tokens = 0
        self.cut_off_price: CapacityUnitPrice = UNDEFINED_PRICE
        # consumer state
        self.upgrade_running: bool = False
        self.restoration_running: bool = False
        self.cmd_handled: bool = False

    def info(self) -> ProducerInfo:
        return ProducerInfo(self.id, self.capacity_type, self.max_capacity, self.capacity, self.cut_off_price)

    def capacity_degradation(self) -> Capacity:
        return int(math.ceil(self.degradation * self.max_capacity / 100))

    def view(self) -> ProducingAgentView:
        return ProducingAgentView(
            self.id,
            self.max_capacity,
            self.capacity,
            self.requested_capacity,
            self.capacity_degradation(),
            self.upgrade.increases,
            self.restoration.restores,
            self.upgrade_running,
            self.restoration_running,
        )

    def place_bids(self, bids: list[Bid]) -> None:
        for b in bids:
            if b.capacity_type != self.capacity_type:
                raise RuntimeError("wrong capacity type")
        self.bids.extend(bids)

    def handle_cmd(self, cmd: ProducingAgentCommand) -> list[InvestmentRequest]:
        if self.cmd_handled:
            raise RuntimeError("command is handled already")
        if self.upgrade_running and cmd.do_upgrade:
            raise RuntimeError("upgrade is running")
        if self.restoration_running and cmd.do_restoration:
            raise RuntimeError("restorations is running")
        requests: list[InvestmentRequest] = []
        if cmd.do_upgrade:
            requests.append(
                InvestmentRequest(self.id, InvestmentType.UPGRADE, self.upgrade.require, self.cut_off_price)
            )
            self.upgrade_running = True
        if cmd.do_restoration:
            requests.append(
                InvestmentRequest(self.id, InvestmentType.RESTORATION, self.restoration.require, self.cut_off_price)
            )
            self.restoration_running = True
        self.cmd_handled = True
        return requests

    def investment_completed(self, request: InvestmentRequest) -> None:
        if request.producer_id != self.id:
            raise RuntimeError("not found")
        if request.type == InvestmentType.RESTORATION:
            if not self.restoration_running:
                raise RuntimeError("no restoration running")
            self.restoration_running = False
            self.capacity = min(self.max_capacity, self.capacity + self.restoration.restores)
        elif request.type == InvestmentType.UPGRADE:
            if not self.upgrade_running:
                raise RuntimeError("no upgrades running")
            self.upgrade_running = False
            self.max_capacity += self.upgrade.increases
            self.capacity += self.upgrade.increases
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
        else:
            raise RuntimeError("unsupported")

    def produce(self) -> ProductionResult:
        # Sort bids by capacity unit price descending (most valuable first).
        # Go's comparator returns 1 when a<b else -1 (a strict desc order).
        self.bids.sort(key=lambda b: b.capacity_unit_price(), reverse=True)
        requested_capacity = sum(b.capacity for b in self.bids)
        remaining_capacity = int(self.capacity)
        completed: list[Bid] = []
        rejected: list[Bid] = []
        processing: list[Bid] = []
        capacity_next = max(0, self.capacity - self.capacity_degradation())

        cut_off_price = self.cut_off_price
        funds: Tokens = 0
        in_progress = self.in_progress
        if in_progress is not None:
            remaining_capacity -= int(in_progress.booked)
            if remaining_capacity >= 0:
                completed.append(in_progress.bid)
                in_progress = None
            else:
                in_progress = _Booking(in_progress.order_id, in_progress.booked - self.capacity, in_progress.bid)

        for bid in self.bids:
            if remaining_capacity > 0:
                cut_off_price = bid.capacity_unit_price()
                funds += bid.tokens
                remaining_capacity -= int(bid.capacity)
                if remaining_capacity >= 0:
                    completed.append(bid)
                else:
                    in_progress = _Booking(bid.order_id, -remaining_capacity, bid)
                continue
            rejected.append(bid)

        # commit next state
        self.capacity = capacity_next
        self.bids = []
        self.in_progress = in_progress
        self.requested_capacity = requested_capacity
        self.funds = funds
        self.cut_off_price = cut_off_price
        if in_progress is not None:
            processing.append(in_progress.bid)
        self.cmd_handled = False
        return ProductionResult(processing, completed, rejected)
