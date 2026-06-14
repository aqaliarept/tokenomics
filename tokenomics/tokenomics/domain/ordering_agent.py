"""Ordering agent: splits an order's *quantity* across eligible producers as
whole-bundle sub-orders (ADR-0025).

It never decomposes a product into recipe components — that is the producing
agent's job. For each order it holds, it hands each chosen producer a sub-order
(quantity + tokens) the producer must fulfil as a whole; the bundle's capacity and
input-product requirements scale with the sub-order quantity. A producer is
*eligible* only if it owns every capacity type the recipe requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import (
    CapacityType,
    NotFoundError,
    OrderId,
    OrderingAgentId,
    ProducerId,
    Tokens,
)
from .order import OrderInfo
from .producing_agent import Bid, ProducerInfo


@dataclass
class SubOrder:
    """One slice of an order handed to a single producer (ADR-0025)."""

    producer_id: ProducerId
    quantity: int
    tokens: Tokens


@dataclass
class OrderingAgentView:
    incoming: dict[OrderId, OrderInfo]
    # eligible producers per order: those owning every required capacity type
    eligible: dict[OrderId, dict[ProducerId, ProducerInfo]]


@dataclass
class OrderingAgentCommand:
    orders: dict[OrderId, list[SubOrder]] = field(default_factory=dict)


def _eligible(p: ProducerInfo, required: dict[CapacityType, int]) -> bool:
    return all(p.owns(t) for t in required)


class OrderingAgent:
    def __init__(self, id: OrderingAgentId) -> None:
        self.id = id
        self.incoming: dict[OrderId, OrderInfo] = {}
        self.cmd_handled = False

    def place_order(self, order_info: OrderInfo) -> None:
        # Nothing left to distribute once every unit has been accepted (delivered
        # or reserved) — only the unresolved remainder is re-bid (ADR-0025).
        if order_info.quantity == 0:
            return
        self.incoming[order_info.id] = order_info

    def view(self, producers: dict[ProducerId, ProducerInfo]) -> OrderingAgentView:
        eligible: dict[OrderId, dict[ProducerId, ProducerInfo]] = {}
        for oid, oi in self.incoming.items():
            eligible[oid] = {
                pid: p for pid, p in producers.items() if _eligible(p, oi.unit_capacity)
            }
        return OrderingAgentView(dict(self.incoming), eligible)

    def complete_cycle(self) -> None:
        # Divergence from Go: Go set cmdHandled=True here, which would make
        # handle_cmd raise on every cycle after the first (a latent bug — HTTP runs
        # were single-cycle). Reset so the agent can handle next cycle's command.
        self.incoming = {}
        self.cmd_handled = False

    def handle_cmd(
        self, cmd: OrderingAgentCommand, producers: dict[ProducerId, ProducerInfo]
    ) -> dict[ProducerId, list[Bid]]:
        if self.cmd_handled:
            raise RuntimeError("command is handled already")
        if len(cmd.orders) != len(self.incoming):
            raise RuntimeError(
                f"too few orders passed. Incoming [{len(self.incoming)}] passed [{len(cmd.orders)}]"
            )

        result: dict[ProducerId, list[Bid]] = {}
        for order_id, subs in cmd.orders.items():
            if order_id not in self.incoming:
                raise NotFoundError(f"order id [{order_id}] not found for agent [{self.id}]")
            order = self.incoming[order_id]
            self._check_conservation(order, subs)
            for sub in subs:
                p = producers.get(sub.producer_id)
                if p is None or not _eligible(p, order.unit_capacity):
                    raise RuntimeError(
                        f"order [{order_id}] sub-order to ineligible producer [{sub.producer_id}]"
                    )
                bid = Bid(
                    capacities={t: c * sub.quantity for t, c in order.unit_capacity.items()},
                    tokens=sub.tokens,
                    order_id=order_id,
                    product=order.product,
                    inputs={pp: q * sub.quantity for pp, q in order.unit_inputs.items()},
                    quantity=sub.quantity,
                )
                result.setdefault(sub.producer_id, []).append(bid)
        self.cmd_handled = True
        return result

    @staticmethod
    def _check_conservation(order: OrderInfo, subs: list[SubOrder]) -> None:
        for s in subs:
            if s.quantity <= 0:
                raise RuntimeError(f"order [{order.id}] has a zero-quantity sub-order")
        if sum(s.quantity for s in subs) != order.quantity:
            raise RuntimeError(
                f"order [{order.id}] sub-quantities do not sum to {order.quantity}"
            )
        if sum(s.tokens for s in subs) != order.tokens:
            raise RuntimeError(
                f"order [{order.id}] sub-tokens do not sum to {order.tokens}"
            )
