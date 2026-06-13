"""Ordering agent: distributes an order's tokens across producing agents as
bids. Port of emulation/domain/ordering_agent.go.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .common import (
    Capacity,
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
class OrderingAgentView:
    incoming: dict[OrderId, dict[CapacityType, Capacity]]
    producers: dict[CapacityType, dict[ProducerId, ProducerInfo]]


@dataclass
class OrderingAgentCommand:
    orders: dict[OrderId, dict[ProducerId, Tokens]] = field(default_factory=dict)


class OrderingAgent:
    def __init__(self, id: OrderingAgentId) -> None:
        self.id = id
        self.incoming: dict[OrderId, OrderInfo] = {}
        self.cmd_handled = False

    def place_order(self, order_info: OrderInfo) -> None:
        if order_info.fulfilled():
            return
        self.incoming[order_info.id] = order_info

    def view(self, producers: dict[ProducerId, ProducerInfo]) -> OrderingAgentView:
        capacity_types: set[CapacityType] = set()
        incoming: dict[OrderId, dict[CapacityType, Capacity]] = {}
        for oid, oi in self.incoming.items():
            for ct in oi.required:
                capacity_types.add(ct)
            incoming[oid] = oi.required
        result_producers: dict[CapacityType, dict[ProducerId, ProducerInfo]] = {}
        for producer_id, p in producers.items():
            if p.capacity_type not in capacity_types:
                continue
            result_producers.setdefault(p.capacity_type, {})[producer_id] = p
        return OrderingAgentView(incoming, result_producers)

    def complete_cycle(self) -> None:
        # Divergence from Go: Go set cmdHandled=True here, which would make
        # handle_cmd raise on every cycle after the first. That is a latent bug
        # (HTTP runs were single-cycle). Reset to False so the agent can handle
        # next cycle's command, matching ProducingAgent.produce semantics.
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
        for order_id, bids in cmd.orders.items():
            if order_id not in self.incoming:
                raise NotFoundError(f"order id [{order_id}] not found for agent [{self.id}]")
            order = self.incoming[order_id]
            if len(order.required) != len(bids):
                raise RuntimeError(f"too few bids passed for order [{order_id}]")
            agent_bids = sum(bids.values())
            if order.tokens != agent_bids:
                raise RuntimeError(
                    f"order-id: [{order_id}] agent bids sum [{agent_bids}] not equal to order tokens [{order.tokens}] "
                )
            for producer_id, tokens in bids.items():
                cap_type = producers[producer_id].capacity_type
                if cap_type not in order.required:
                    raise RuntimeError(
                        f"order [{order_id}] doesn't contain capacity type for producer "
                        f"[{producer_id}] with capacity type [{cap_type}]"
                    )
                required = order.required[cap_type]
                result.setdefault(producer_id, []).append(Bid(cap_type, required, tokens, order_id))
        self.cmd_handled = True
        return result
