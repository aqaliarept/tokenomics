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
    ReplenishmentRequestCompleted,
    ReplenishmentRequestRejected,
    Score,
)
from .ordering_agent import OrderingAgent, OrderingAgentCommand, OrderingAgentView
from .requests import InvestmentType, ReplenishmentRequest
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
    replenishment_completed: int = 0  # input-purchase orders delivered (instrumental)
    replenishment_rejected: int = 0  # input-purchase orders rejected or timed out

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


class System:
    def __init__(
        self, id_gen: OrderIdGenerator, config: Configuration, consumers: dict[ConsumerId, Consumer]
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

    def ordering_agent_action(self, aid: OrderingAgentId, cmd: OrderingAgentCommand) -> None:
        if self.state != SystemState.ORDERING:
            raise WrongStateError()
        agent = must_get(self.ordering_agents, aid)
        routed = agent.handle_cmd(cmd, self.producer_infos)
        for prod_id, subs in routed.items():
            must_get(self.producing_agents, prod_id).place_sub_orders(subs)

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
            if order.requires_funding():
                continue  # expired this cycle (unfunded) -> rejected in complete_cycle
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
            for bid in result.accepted:
                must_get(self.orders, bid.order_id).sub_accepted(bid.quantity, bid.tokens)
            for bid in result.completed:
                must_get(self.orders, bid.order_id).sub_completed(bid.quantity)
            for bid in result.failed:
                must_get(self.orders, bid.order_id).sub_failed(bid.quantity)
            for bid in result.rejected:
                must_get(self.orders, bid.order_id).sub_rejected(bid.quantity)

        cycle_score: Score = 0
        completed_ids: list[OrderId] = []
        n_cons_done = n_inv_done = n_in_progress = n_cons_rej = n_inv_rej = 0
        n_repl_done = n_repl_rej = 0
        for oid, order in self.orders.items():
            # an unfunded producer order expired this cycle (budget exhausted by
            # higher-priority orders) — reject it without an auction (ADR-0025).
            score, event = order.expire() if order.requires_funding() else order.complete_cycle()
            cycle_score += score
            keep = False
            if isinstance(event, ConsumerRequestCompleted):
                n_cons_done += 1
            elif isinstance(event, InvestmentRequestCompleted):
                n_inv_done += 1
                self.producing_agents[event.request.producer_id].investment_completed(event.request)
            elif isinstance(event, ReplenishmentRequestCompleted):
                n_repl_done += 1
                self.producing_agents[event.request.producer_id].replenishment_completed(event.request)
            elif isinstance(event, ConsumerRequestRejected):
                n_cons_rej += 1
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
        # fire while the reservation still waits for inputs. Drop those orphaned
        # reservations now — otherwise the next produce() surfaces a bid for a
        # deleted order and must_get raises (ADR-0025).
        if removed:
            for p in self.producing_agents.values():
                p.reservations = [r for r in p.reservations if r.bid.order_id not in removed]

        self.start_cycle()
        return CycleResult(
            cycle_score, n_cons_done, n_inv_done, n_in_progress, n_cons_rej, n_inv_rej,
            n_repl_done, n_repl_rej,
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
