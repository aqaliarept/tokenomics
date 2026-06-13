"""System: the cycle orchestrator. Port of emulation/domain/system.go.

Divergence from Go (intentional): Go computes `producerInfos` once in NewSystem
and never refreshes it, so ordering agents always saw the *initial* producer
state. That is a latent bug — for RL the ordering agent must see live capacity
and cut-off price (ADR-0002). Here producer_infos is refreshed each cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .common import (
    ConsumerId,
    OrderId,
    OrderingAgentId,
    ProducerId,
    Tokens,
    WrongStateError,
    f32,
    from_consumer_id,
    from_producer_id,
    is_nan,
    must_get,
)
from .config import Configuration
from .consumer import Consumer
from .order import (
    ConsumerRequestCompleted,
    ConsumerRequestRejected,
    InvestmentRequestCompleted,
    InvestmentRequestRejected,
    Order,
    OrderStillProcessing,
    Score,
)
from .ordering_agent import OrderingAgent, OrderingAgentCommand, OrderingAgentView
from .producing_agent import (
    ProducerInfo,
    ProducingAgent,
    ProducingAgentCommand,
    ProducingAgentView,
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


@dataclass
class CycleResult:
    score: Score  # Go penalty score (parity only)
    consumer_completed: int = 0  # consumer orders fulfilled (the terminal objective)
    investment_completed: int = 0  # restoration/upgrade orders fulfilled (instrumental)
    in_progress: int = 0  # orders still processing
    consumer_rejected: int = 0  # consumer orders rejected or timed out
    investment_rejected: int = 0  # investment orders rejected or timed out

    @property
    def completed(self) -> int:
        return self.consumer_completed + self.investment_completed

    @property
    def rejected(self) -> int:
        return self.consumer_rejected + self.investment_rejected


def distribute_investment_fund(orders: dict[OrderId, Order], investment_fund: Tokens) -> None:
    """Port of distibuteInvestmentFund. Distributes the fund proportionally to
    each unfunded investment order's cut-off price; orders with a NaN cut-off
    price split the remainder equally."""
    total_cut_off_sum = 0.0
    nan_count = 0

    for order in orders.values():
        if not order.requires_funding():
            continue
        if not is_nan(order.cut_off_price()):
            total_cut_off_sum += order.cut_off_price()
        else:
            nan_count += 1

    remains = investment_fund
    for order in orders.values():
        if not order.requires_funding() or is_nan(order.cut_off_price()):
            continue
        # Divergence from Go: Go does a raw float32 division here; if all priced
        # orders have a zero cut-off price total_cut_off_sum is 0 and Go yields
        # inf/garbage (never panics). Guard it — zero price means no capacity
        # deficit, so fund 0 and let NaN orders split the remainder.
        if total_cut_off_sum == 0:
            funds = 0
        else:
            funds = int(f32(order.cut_off_price()) * f32(investment_fund) / f32(total_cut_off_sum))
        order.fund(funds)
        remains -= funds

    if nan_count == 0:
        return

    funds = remains // nan_count
    for order in orders.values():
        if not order.requires_funding():
            continue
        order.fund(funds)
        remains -= funds


class System:
    def __init__(
        self, id_gen: OrderIdGenerator, config: Configuration, consumers: dict[ConsumerId, Consumer]
    ) -> None:
        self.state = SystemState.ORDERS_PLACEMENT
        self.id_gen = id_gen
        self.cycle_emission: Tokens = config.cycle_emission
        self.investment_fund: Tokens = 0
        self.process_sheets = {ps.product: ps for ps in config.process_sheets}
        self.producing_agents: dict[ProducerId, ProducingAgent] = {
            p.id: ProducingAgent(p) for p in config.producer_configs
        }
        self.ordering_agents: dict[OrderingAgentId, OrderingAgent] = {}
        self.orders: dict[OrderId, Order] = {}
        self.consumers = consumers
        self.cycle_counter = 0

        self.producer_infos: dict[ProducerId, ProducerInfo] = self._snapshot_producer_infos()
        for c in consumers.values():
            aid = from_consumer_id(c.id())
            self.ordering_agents[aid] = OrderingAgent(aid)
        for prod_id in self.producing_agents:
            aid = from_producer_id(prod_id)
            self.ordering_agents[aid] = OrderingAgent(aid)

        self.start_cycle()

    def _snapshot_producer_infos(self) -> dict[ProducerId, ProducerInfo]:
        return {pid: p.info() for pid, p in self.producing_agents.items()}

    # --- producing agent phase ---
    def producing_agent_view(self, pid: ProducerId) -> ProducingAgentView:
        if self.state != SystemState.ORDERS_PLACEMENT:
            raise WrongStateError()
        return must_get(self.producing_agents, pid).view()

    def producing_agent_action(self, pid: ProducerId, cmd: ProducingAgentCommand) -> None:
        if self.state != SystemState.ORDERS_PLACEMENT:
            raise WrongStateError()
        p = must_get(self.producing_agents, pid)
        for r in p.handle_cmd(cmd):
            oid = self.id_gen.new()
            self.orders[oid] = Order.new_investment_order(oid, must_get(self.process_sheets, r.product), r)

    # --- ordering agent phase ---
    def ordering_agent_view(self, aid: OrderingAgentId) -> OrderingAgentView:
        if self.state != SystemState.ORDERING:
            raise WrongStateError()
        return must_get(self.ordering_agents, aid).view(self.producer_infos)

    def ordering_agent_action(self, aid: OrderingAgentId, cmd: OrderingAgentCommand) -> None:
        if self.state != SystemState.ORDERING:
            raise WrongStateError()
        agent = must_get(self.ordering_agents, aid)
        bids = agent.handle_cmd(cmd, self.producer_infos)
        for prod_id, prod_bids in bids.items():
            must_get(self.producing_agents, prod_id).place_bids(prod_bids)

    # --- cycle lifecycle ---
    def _emit(self) -> None:
        self.investment_fund = self.cycle_emission // 2
        if not self.consumers:
            return
        consumer_tokens = self.cycle_emission // 2 // len(self.consumers)
        for c in self.consumers.values():
            c.emit(consumer_tokens)

    def _place_consumer_orders(self) -> None:
        for c in self.consumers.values():
            for request in c.order():
                oid = self.id_gen.new()
                self.orders[oid] = Order.new_consumer_order(
                    oid, must_get(self.process_sheets, request.product), request
                )

    def start_cycle(self) -> None:
        self.state = SystemState.ORDERS_PLACEMENT
        self._emit()
        self._place_consumer_orders()
        self.cycle_counter += 1

    def start_ordering(self) -> None:
        if self.state != SystemState.ORDERS_PLACEMENT:
            raise WrongStateError()
        distribute_investment_fund(self.orders, self.investment_fund)
        # Refresh live producer state for ordering agents (see module docstring).
        self.producer_infos = self._snapshot_producer_infos()
        for order in self.orders.values():
            aid = order.agent_id()
            must_get(self.ordering_agents, aid).place_order(order.info())
        self.state = SystemState.ORDERING

    def complete_cycle(self) -> CycleResult:
        if self.state != SystemState.ORDERING:
            raise WrongStateError()

        for oa in self.ordering_agents.values():
            oa.complete_cycle()

        for p in self.producing_agents.values():
            result = p.produce()
            for bid in result.processing:
                must_get(self.orders, bid.order_id).processing(bid.capacity_type, bid.tokens)
            for bid in result.completed:
                must_get(self.orders, bid.order_id).completed(bid.capacity_type, bid.tokens)
            for bid in result.rejected:
                must_get(self.orders, bid.order_id).rejected(bid.capacity_type)

        cycle_score: Score = 0
        completed_ids: list[OrderId] = []
        n_cons_done = n_inv_done = n_in_progress = n_cons_rej = n_inv_rej = 0
        for oid, order in self.orders.items():
            score, event = order.complete_cycle()
            cycle_score += score
            keep = False
            if isinstance(event, ConsumerRequestCompleted):
                n_cons_done += 1
            elif isinstance(event, InvestmentRequestCompleted):
                n_inv_done += 1
                self.producing_agents[event.request.producer_id].investment_completed(event.request)
            elif isinstance(event, ConsumerRequestRejected):
                n_cons_rej += 1
                self.consumers[event.request.consumer_id].emit(event.remaining)
            elif isinstance(event, InvestmentRequestRejected):
                n_inv_rej += 1
                self.producing_agents[event.request.producer_id].investment_rejected(event.request)
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

        # Divergence from Go: a timed-out/rejected order can be removed while a
        # producer still carries it as an in-progress booking. Go never hit this
        # (single-cycle HTTP runs); left unhandled it crashes next produce() with
        # a bid for a missing order. Release dangling bookings here.
        for p in self.producing_agents.values():
            if p.in_progress is not None and p.in_progress.order_id in removed:
                p.in_progress = None

        self.start_cycle()
        return CycleResult(cycle_score, n_cons_done, n_inv_done, n_in_progress, n_cons_rej, n_inv_rej)

    def is_dead(self) -> bool:
        """System death: all producers at zero capacity (ADR-0002)."""
        return all(p.capacity == 0 for p in self.producing_agents.values())

    # --- introspection ---
    def get_producer_infos(self) -> dict[ProducerId, ProducerInfo]:
        return self.producer_infos

    def get_ordering_agent_ids(self) -> list[OrderingAgentId]:
        return list(self.ordering_agents.keys())
