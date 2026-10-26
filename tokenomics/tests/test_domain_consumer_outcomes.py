"""Per-order consumer outcome records on CycleResult (design D2,
add-measurement-gate-harness).

Each consumer order is tagged with the cycle it is first played in; when it
completes or is rejected, CycleResult carries one ConsumerOrderOutcome for it. The
env (task 3.1) counts ordered units from played cycles only, so the tag must make
the final, never-played placement distinguishable.
"""

from __future__ import annotations

from tokenomics.domain import ConsumerOrderOutcome, build_system
import pytest

from tokenomics.domain.common import ConsumerId, Product, Tokens
from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.order import ORDER_TTL, Order
from tokenomics.domain.ordering_agent import OrderingAgentCommand, SubOrder
from tokenomics.domain.producing_agent import ProducingAgentCommand
from tokenomics.domain.requests import ConsumerRequest
from tokenomics.domain.system import CycleResult, System


class _OneShotConsumer:
    """Orders `quantity` x `product` once, in the first cycle, with its whole balance."""

    def __init__(self, consumer_id: ConsumerId, product: Product, quantity: int) -> None:
        self._id = consumer_id
        self._product = product
        self._quantity = quantity
        self._balance: Tokens = 0
        self._done = False

    def id(self) -> ConsumerId:
        return self._id

    def emit(self, tokens: Tokens) -> None:
        self._balance += tokens

    def order(self) -> list[ConsumerRequest]:
        if self._done:
            return []
        self._done = True
        req = ConsumerRequest(self._id, self._product, self._balance, quantity=self._quantity)
        self._balance = 0
        return [req]


def _producer(pid: str, ctype: str, cap: int, product: Product) -> ProducingAgentConfig:
    return ProducingAgentConfig(
        id=pid,
        capacities={ctype: cap},
        degradation=0,
        restoration=Restoration(require=10, restores=10, type=ctype),
        upgrade=Upgrade(require=10, increases=10, type=ctype),
        produces=frozenset({product}),
    )


def _economy() -> System:
    """c1 wants 2 x product 10 (cheap, p1 has ample capacity-1) -> completes in cycle 1.
    c2 wants 3 x product 20 (50 capacity-2 per unit): p2 (100) takes 1 unit, p3 (10)
    can never take the other 2 -> rejected at TTL with 100 of its 250 tokens burnt."""
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=10, require={"capacity-1": 5}),
            ProcessSheet(product=20, require={"capacity-2": 50}),
        ],
        producer_configs=[
            _producer("p1", "capacity-1", 100, 10),
            _producer("p2", "capacity-2", 100, 20),
            _producer("p3", "capacity-2", 10, 20),
        ],
        consumers=[ConsumerConfig(id="c1", product=10), ConsumerConfig(id="c2", product=20)],
    )
    consumers = {"c1": _OneShotConsumer("c1", 10, 2), "c2": _OneShotConsumer("c2", 20, 3)}
    return build_system(cfg, consumers=consumers)


def _play_cycle(system: System, first: bool) -> CycleResult:
    for pid in system.producing_agents:
        system.producing_agent_action(pid, ProducingAgentCommand(False, False))
    system.start_ordering()
    for aid, agent in system.ordering_agents.items():
        view = agent.view(system.producer_infos)
        orders: dict[str, list[SubOrder]] = {}
        for oid, oi in view.incoming.items():
            if oi.product == 10:
                orders[oid] = [SubOrder("p1", oi.quantity, oi.tokens)]
            elif first:  # split c2's order: 1 unit (100 tokens) to p2, the rest to p3
                orders[oid] = [SubOrder("p2", 1, 100), SubOrder("p3", oi.quantity - 1, oi.tokens - 100)]
            else:
                orders[oid] = [SubOrder("p3", oi.quantity, oi.tokens)]
        system.ordering_agent_action(aid, OrderingAgentCommand(orders))
    return system.complete_cycle()


def test_initial_consumer_orders_are_tagged_with_first_played_cycle() -> None:
    system = _economy()
    assert sorted(o.placement_cycle for o in system.orders.values()) == [1, 1]


def test_one_completed_and_one_rejected_order_produce_outcome_records() -> None:
    system = _economy()
    results = [_play_cycle(system, first=True)]
    (open_c2,) = system.orders.values()
    assert (open_c2.accepted_qty, open_c2.available_tokens()) == (1, 150)  # p2 took 1 unit
    results += [_play_cycle(system, first=False) for _ in range(ORDER_TTL)]

    assert results[0].consumer_outcomes == (
        ConsumerOrderOutcome("c1", 10, 2, 250, 1, True),
    )
    for r in results[1:ORDER_TTL]:
        assert r.consumer_outcomes == ()
    # tokens are those at placement (250), not the 150 still available at rejection.
    assert results[ORDER_TTL].consumer_outcomes == (
        ConsumerOrderOutcome("c2", 20, 3, 250, 1, False),
    )
    assert not system.orders  # one-shot consumers place nothing after cycle 1


def test_outcomes_agree_with_existing_counters() -> None:
    system = _economy()
    for i in range(ORDER_TTL + 1):
        r = _play_cycle(system, first=(i == 0))
        done = [o for o in r.consumer_outcomes if o.completed]
        rej = [o for o in r.consumer_outcomes if not o.completed]
        assert (len(done), len(rej)) == (r.consumer_completed, r.consumer_rejected)
        assert sum(o.quantity for o in done) == r.consumer_completed_units
        assert sum(o.quantity for o in rej) == r.consumer_rejected_units


def test_trailing_placement_is_tagged_with_the_unplayed_next_cycle() -> None:
    """complete_cycle ends by placing the next cycle's orders; after k played cycles
    every open order placed then carries placement_cycle k + 1, and resolved orders
    carry a cycle <= k."""
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=10, require={"capacity-1": 5})],
        producer_configs=[_producer("p1", "capacity-1", 100, 10)],
        consumers=[ConsumerConfig(id="c1", product=10)],
    )
    system = build_system(cfg)
    seen: list[int] = []
    for k in range(1, 4):
        r = _play_cycle(system, first=False)
        seen.extend(o.placement_cycle for o in r.consumer_outcomes)
        assert all(pc <= k for pc in seen)
        assert [o.placement_cycle for o in system.orders.values()] == [k + 1]
    assert seen == [1, 2, 3]  # scripted c1 orders once per cycle, each completes in-cycle


def test_untagged_consumer_order_fails_loudly_when_it_resolves() -> None:
    """An outcome with no placement cycle would silently corrupt the env's
    played-cycle accounting (design D2), so complete_cycle must refuse it."""
    system = _economy()
    system.start_ordering()
    ghost = Order.new_consumer_order(
        "ghost", system.process_sheets[10], ConsumerRequest("c1", 10, tokens=5, quantity=1)
    )
    ghost.completed_qty = 1  # resolves (completed) this cycle
    system.orders["ghost"] = ghost
    with pytest.raises(RuntimeError, match="no placement cycle"):
        system.complete_cycle()
