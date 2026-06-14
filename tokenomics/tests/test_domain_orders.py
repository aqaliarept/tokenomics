"""B4-integration: whole-product orders + sub-order aggregation (ADR-0025).

An order is for a quantity of a product as a whole. An ordering agent splits the
quantity across eligible producers as sub-orders; the producer reports each
sub-order completed / rejected / still-processing by its unit quantity. The order
is the scoring unit: completed only when its whole quantity completes.
"""

from __future__ import annotations

from math import nan

import pytest

from tokenomics.domain import build_system
from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.order import (
    ConsumerRequestCompleted,
    ConsumerRequestRejected,
    Order,
    OrderStillProcessing,
)
from tokenomics.domain.ordering_agent import OrderingAgent, OrderingAgentCommand, SubOrder
from tokenomics.domain.producing_agent import ProducerInfo, ProducingAgentCommand
from tokenomics.domain.requests import ConsumerRequest


def _drive_ordering(system):
    """Hand each order's whole quantity+tokens to its first eligible producer."""
    for aid, agent in system.ordering_agents.items():
        view = agent.view(system.producer_infos)
        cmd_orders = {}
        for oid, oi in view.incoming.items():
            pid = next(iter(view.eligible[oid]))
            cmd_orders[oid] = [SubOrder(pid, oi.quantity, oi.tokens)]
        system.ordering_agent_action(aid, OrderingAgentCommand(cmd_orders))


def test_full_cycle_consumer_order_completes():
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=10, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig(
                id="p1",
                capacities={"capacity-1": 100},
                degradation=0,
                restoration=Restoration(require=1, restores=10, type="capacity-1"),
                upgrade=Upgrade(require=1, increases=10, type="capacity-1"),
            )
        ],
        consumers=[ConsumerConfig(id="c1", product=10)],
    )
    system = build_system(cfg)
    system.producing_agent_action("p1", ProducingAgentCommand(False, False))
    system.start_ordering()
    _drive_ordering(system)
    result = system.complete_cycle()
    assert result.consumer_completed == 1


def test_reservation_cleared_when_parent_order_closes():
    """A composite reservation can outlive its parent order (order TTL fires while
    the reservation still waits for inputs). If the closed order is deleted without
    dropping that reservation, the next produce() surfaces a bid for a missing
    order and crashes. Running a contended composite for many cycles must not
    raise, and no producer may hold a reservation for a closed order (ADR-0025)."""
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=1, require={"capacity-1": 1}),  # primal input, never stocked
            ProcessSheet(product=30, require={"capacity-1": 5}, inputs={1: 1}),  # composite
        ],
        producer_configs=[
            ProducingAgentConfig(
                id="p1",
                capacities={"capacity-1": 5},  # fits only one bundle -> heavy contention
                degradation=0,
                restoration=Restoration(require=1, restores=5, type="capacity-1"),
                upgrade=Upgrade(require=1, increases=5, type="capacity-1"),
            )
        ],
        consumers=[ConsumerConfig(id="c1", product=30), ConsumerConfig(id="c2", product=30)],
    )
    system = build_system(cfg)
    for _ in range(8):
        _run_cycle(system)  # must not raise
    live_order_ids = set(system.orders)
    for p in system.producing_agents.values():
        for r in p.reservations:
            assert r.bid.order_id in live_order_ids  # no reservation for a closed order


def _simple_config(degradation):
    return Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=10, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig(
                id="p1",
                capacities={"capacity-1": 100},
                degradation=degradation,
                restoration=Restoration(require=10, restores=30, type="capacity-1"),
                upgrade=Upgrade(require=10, increases=20, type="capacity-1"),
            )
        ],
        consumers=[ConsumerConfig(id="c1", product=10)],
    )


def _run_cycle(system, *, restore=False, upgrade=False):
    for pid, p in system.producing_agents.items():
        system.producing_agent_action(
            pid,
            ProducingAgentCommand(
                do_restoration=restore and not p.restoration_running,
                do_upgrade=upgrade and not p.upgrade_running,
            ),
        )
    system.start_ordering()
    _drive_ordering(system)
    return system.complete_cycle()


def test_degradation_without_restoration_kills_system():
    system = build_system(_simple_config(degradation=10))
    assert not system.is_dead()
    for _ in range(15):
        if system.is_dead():
            break
        _run_cycle(system)
    assert system.is_dead()
    assert system.producing_agents["p1"].capacities["capacity-1"] == 0


def test_restoration_recovers_capacity():
    system = build_system(_simple_config(degradation=10))
    p1 = system.producing_agents["p1"]
    for _ in range(4):
        _run_cycle(system)
    degraded = p1.capacities["capacity-1"]
    assert degraded < 100  # degraded from start
    recovered = degraded
    for _ in range(12):
        _run_cycle(system, restore=True)
        recovered = max(recovered, p1.capacities["capacity-1"])
    assert recovered > degraded  # restoration lifted capacity above the low point


def test_order_completes_only_when_all_quantity_completed():
    ps = ProcessSheet(product=10, require={"capacity-1": 2})
    req = ConsumerRequest("c1", product=10, tokens=100, quantity=3)
    order = Order.new_consumer_order("o1", ps, req)

    order.sub_accepted(quantity=2, tokens=66)
    order.sub_completed(quantity=2)
    _, event = order.complete_cycle()
    assert isinstance(event, OrderStillProcessing)  # 2 of 3 -> not whole yet

    order.sub_accepted(quantity=1, tokens=34)
    order.sub_completed(quantity=1)
    _, event = order.complete_cycle()
    assert isinstance(event, ConsumerRequestCompleted)


def test_info_exposes_only_unresolved_quantity_and_tokens():
    ps = ProcessSheet(product=10, require={"capacity-1": 2})
    order = Order.new_consumer_order("o1", ps, ConsumerRequest("c1", 10, 100, quantity=4))
    order.sub_accepted(quantity=1, tokens=25)  # cleared the auction -> tokens burnt
    order.sub_completed(quantity=1)
    info = order.info()
    assert info.quantity == 3  # 4 - 1 accepted; reserved/done not re-distributed
    assert info.tokens == 75  # 100 - 25 burnt


def test_auction_rejected_quantity_stays_unresolved():
    ps = ProcessSheet(product=10, require={"capacity-1": 2})
    order = Order.new_consumer_order("o1", ps, ConsumerRequest("c1", 10, 100, quantity=4))
    order.sub_accepted(quantity=2, tokens=50)
    order.sub_completed(quantity=2)
    order.sub_rejected(quantity=2)  # never cleared the auction -> re-biddable
    info = order.info()
    assert info.quantity == 2
    assert info.tokens == 50
    _, event = order.complete_cycle()
    assert isinstance(event, OrderStillProcessing)  # still open, will re-bid


def test_failed_reservation_tokens_are_not_refunded():
    ps = ProcessSheet(product=20, require={"capacity-1": 5}, inputs={10: 1})
    order = Order.new_consumer_order("o1", ps, ConsumerRequest("c1", 20, 100, quantity=1))
    order.sub_accepted(quantity=1, tokens=100)  # cleared, tokens burnt at acceptance
    order.sub_failed(quantity=1)  # reservation TTL'd / dropped
    _, event = order.complete_cycle()
    assert isinstance(event, ConsumerRequestRejected)
    assert event.remaining == 0  # burnt tokens are not refunded


def _producer(pid, types):
    caps = {t: 999 for t in types}
    return ProducerInfo(pid, caps, dict(caps), nan)


def test_ordering_agent_splits_quantity_into_bundle_bids():
    ps = ProcessSheet(product=30, require={"capacity-1": 2, "capacity-2": 3}, inputs={20: 1})
    req = ConsumerRequest("c1", product=30, tokens=100, quantity=4)
    order = Order.new_consumer_order("o1", ps, req)

    agent = OrderingAgent("c1")
    agent.place_order(order.info())
    producers = {
        "p1": _producer("p1", ["capacity-1", "capacity-2"]),
        "p2": _producer("p2", ["capacity-1", "capacity-2"]),
    }
    cmd = OrderingAgentCommand({"o1": [SubOrder("p1", 1, 25), SubOrder("p2", 3, 75)]})
    bids = agent.handle_cmd(cmd, producers)

    [b1] = bids["p1"]
    assert b1.capacities == {"capacity-1": 2, "capacity-2": 3}
    assert b1.inputs == {20: 1}
    assert b1.quantity == 1
    assert b1.tokens == 25
    assert b1.product == 30

    [b2] = bids["p2"]
    assert b2.capacities == {"capacity-1": 6, "capacity-2": 9}
    assert b2.inputs == {20: 3}
    assert b2.quantity == 3
    assert b2.tokens == 75


def test_ordering_agent_rejects_ineligible_producer():
    # product needs both capacity types; p2 owns only capacity-1 -> ineligible
    ps = ProcessSheet(product=30, require={"capacity-1": 2, "capacity-2": 3})
    order = Order.new_consumer_order("o1", ps, ConsumerRequest("c1", 30, 50, quantity=2))
    agent = OrderingAgent("c1")
    agent.place_order(order.info())
    producers = {
        "p1": _producer("p1", ["capacity-1", "capacity-2"]),
        "p2": _producer("p2", ["capacity-1"]),
    }
    cmd = OrderingAgentCommand({"o1": [SubOrder("p1", 1, 25), SubOrder("p2", 1, 25)]})
    with pytest.raises(RuntimeError, match="ineligible"):
        agent.handle_cmd(cmd, producers)


def test_ordering_agent_enforces_quantity_and_token_conservation():
    ps = ProcessSheet(product=10, require={"capacity-1": 1})
    order = Order.new_consumer_order("o1", ps, ConsumerRequest("c1", 10, 100, quantity=4))
    agent = OrderingAgent("c1")
    agent.place_order(order.info())
    producers = {"p1": _producer("p1", ["capacity-1"]), "p2": _producer("p2", ["capacity-1"])}

    # sub-quantities 1+1 != 4
    bad_qty = OrderingAgentCommand({"o1": [SubOrder("p1", 1, 50), SubOrder("p2", 1, 50)]})
    with pytest.raises(RuntimeError, match="sum to 4"):
        agent.handle_cmd(bad_qty, producers)

    # quantities ok but tokens 40+50 != 100
    agent.cmd_handled = False
    bad_tok = OrderingAgentCommand({"o1": [SubOrder("p1", 2, 40), SubOrder("p2", 2, 50)]})
    with pytest.raises(RuntimeError, match="sub-tokens"):
        agent.handle_cmd(bad_tok, producers)

    # zero-quantity sub-order dropped at decode -> domain rejects if present
    agent.cmd_handled = False
    zero = OrderingAgentCommand({"o1": [SubOrder("p1", 0, 0), SubOrder("p2", 4, 100)]})
    with pytest.raises(RuntimeError, match="zero-quantity"):
        agent.handle_cmd(zero, producers)
