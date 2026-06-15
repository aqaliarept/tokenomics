"""Ordering agent: splits an order's *quantity* across eligible producers as
whole-bundle sub-orders (ADR-0025).

It never decomposes a product into recipe components — that is the producing
agent's job. For each order it holds, it hands each chosen producer a sub-order
(quantity + tokens) the producer must fulfil as a whole; the bundle's capacity and
input-product requirements scale with the sub-order quantity. A producer is
*eligible* for a product iff it declares the product in `produces` (ADR-0028);
config validation guarantees it owns the capacity types that product's recipe needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import (
    NotFoundError,
    OrderId,
    OrderingAgentId,
    Product,
    ProducerId,
    Tokens,
)
from .order import OrderInfo
from .producing_agent import ProducerInfo
from .producing_agent import SubOrder as RoutedSubOrder


@dataclass
class SubOrder:
    """One slice of an order as the policy specifies it (ADR-0025): which producer
    gets how much quantity and how many tokens. The order id and product are
    contextual (the command keys by order id). `handle_cmd` turns these into
    producer-routed `RoutedSubOrder`s — neither carries capacity (ADR-0029)."""

    producer_id: ProducerId
    quantity: int
    tokens: Tokens


@dataclass
class OrderingAgentView:
    incoming: dict[OrderId, OrderInfo]
    # eligible producers per order: those declaring the order's product (ADR-0028)
    eligible: dict[OrderId, dict[ProducerId, ProducerInfo]]


@dataclass
class OrderingAgentCommand:
    orders: dict[OrderId, list[SubOrder]] = field(default_factory=dict)


def _eligible(p: ProducerInfo, product: Product) -> bool:
    # Authoritative eligibility (ADR-0028): the producer declares what it makes.
    return product in p.produces


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
                pid: p for pid, p in producers.items() if _eligible(p, oi.product)
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
    ) -> dict[ProducerId, list[RoutedSubOrder]]:
        """Validate the policy's split and route each slice to its producer as a
        capacity-free `RoutedSubOrder` (product + quantity + tokens). The producer
        expands it into a capacity bundle from its own sheet (ADR-0029)."""
        if self.cmd_handled:
            raise RuntimeError("command is handled already")
        if len(cmd.orders) != len(self.incoming):
            raise RuntimeError(
                f"too few orders passed. Incoming [{len(self.incoming)}] passed [{len(cmd.orders)}]"
            )

        result: dict[ProducerId, list[RoutedSubOrder]] = {}
        for order_id, subs in cmd.orders.items():
            if order_id not in self.incoming:
                raise NotFoundError(f"order id [{order_id}] not found for agent [{self.id}]")
            order = self.incoming[order_id]
            self._check_conservation(order, subs)
            for sub in subs:
                p = producers.get(sub.producer_id)
                if p is None or not _eligible(p, order.product):
                    raise RuntimeError(
                        f"order [{order_id}] sub-order to ineligible producer [{sub.producer_id}]"
                    )
                routed = RoutedSubOrder(
                    order_id=order_id, product=order.product, quantity=sub.quantity, tokens=sub.tokens
                )
                result.setdefault(sub.producer_id, []).append(routed)
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
