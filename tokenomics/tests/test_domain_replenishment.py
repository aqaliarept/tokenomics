"""B5-domain: replenishment order path — producers buy input products into
inventory, funded by the central fund at survival priority 1 (ADR-0025).

Replenishment is its OWN request/order kind, not an investment (separate type so
the fund's restoration → replenishment → upgrade → expansion priority holds).
"""

from __future__ import annotations

from tokenomics.domain import build_system
from tokenomics.domain.config import (
    Configuration,
    InventoryExpansion,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
import pytest

from tokenomics.domain.ordering_agent import OrderingAgentCommand, SubOrder
from tokenomics.domain.requests import InvestmentRequest, InvestmentType
from tokenomics.domain.system import distribute_investment_fund
from tokenomics.domain.order import (
    Order,
    ReplenishmentRequestCompleted,
    ReplenishmentRequestRejected,
)
from tokenomics.domain.producing_agent import ProducingAgent, ProducingAgentCommand
from tokenomics.domain.requests import ReplenishmentRequest


def make_producer(capacities=None, inv_limit=100) -> ProducingAgent:
    capacities = capacities or {"capacity-1": 100}
    t = next(iter(capacities))
    return ProducingAgent(
        ProducingAgentConfig(
            id="p",
            capacities=capacities,
            degradation=0,
            restoration=Restoration(require=1, restores=10, type=t),
            upgrade=Upgrade(require=1, increases=10, type=t),
            inventory_limit=inv_limit,
        )
    )


def test_handle_cmd_emits_replenishment_per_positive_input():
    p = make_producer()
    reqs = p.handle_cmd(
        ProducingAgentCommand(False, False, replenishments={10: 5, 11: 0, 12: 3})
    )
    by_product = {r.product: r for r in reqs if isinstance(r, ReplenishmentRequest)}
    # zero-quantity input emits nothing
    assert set(by_product) == {10, 12}
    assert by_product[10].quantity == 5
    assert by_product[12].quantity == 3
    assert by_product[10].producer_id == "p"


def test_outstanding_inbound_claims_headroom():
    p = make_producer(inv_limit=100)
    assert p.headroom() == 100
    p.handle_cmd(ProducingAgentCommand(False, False, replenishments={10: 30}))
    assert p.outstanding_inbound_units() == 30
    assert p.headroom() == 70  # ordered-but-undelivered claims headroom


def test_completed_replenishment_deposits_and_clears_outstanding():
    p = make_producer(inv_limit=100)
    [req] = p.handle_cmd(ProducingAgentCommand(False, False, replenishments={10: 30}))
    p.replenishment_completed(req)
    assert p.inventory[10] == 30
    assert p.outstanding_inbound_units() == 0
    assert p.headroom() == 70  # now held, not inbound — still claims the same room


def test_rejected_replenishment_clears_outstanding_no_deposit():
    p = make_producer(inv_limit=100)
    [req] = p.handle_cmd(ProducingAgentCommand(False, False, replenishments={10: 30}))
    p.replenishment_rejected(req)
    assert p.inventory.get(10, 0) == 0
    assert p.outstanding_inbound_units() == 0
    assert p.headroom() == 100


def _repl_order(oid="r1", qty=3, price=5.0):
    # product 10 is a primal input: capacity-1 only, no further inputs
    ps = ProcessSheet(product=10, require={"capacity-1": 4})
    req = ReplenishmentRequest("p", product=10, quantity=qty, cut_off_price=price)
    return Order.new_replenishment_order(oid, ps, req)


def test_replenishment_order_is_producer_funded_and_routes_to_producer():
    o = _repl_order()
    assert o.requires_funding()  # needs the central fund, no consumer tokens
    assert o.product == 10
    assert o.quantity == 3
    assert o.unit_capacity == {"capacity-1": 4}
    o.fund(120)
    assert o.agent_id() == "p"  # the requesting producer's ordering agent


def test_replenishment_order_completes_and_rejects():
    o = _repl_order(qty=2)
    o.fund(100)
    o.sub_accepted(2, 100)
    o.sub_completed(2)
    _, ev = o.complete_cycle()
    assert isinstance(ev, ReplenishmentRequestCompleted)
    assert ev.request.product == 10

    o2 = _repl_order(qty=2)
    o2.fund(100)
    o2.cycle_counter = 2  # TTL
    _, ev2 = o2.complete_cycle()
    assert isinstance(ev2, ReplenishmentRequestRejected)


def test_view_exposes_inventory_state():
    p = make_producer(inv_limit=40)
    p.handle_cmd(ProducingAgentCommand(False, False, replenishments={10: 12}))  # 12 inbound
    p.inventory = {10: 7}  # 7 held
    v = p.view()  # must not raise (P1a: all view fields populated)
    assert v.inventory_limit == 40
    assert v.inventory_units == 7
    assert v.headroom == 40 - 7 - 12  # held + outstanding both claim headroom


def test_mixed_command_with_invalid_expansion_does_not_leak_headroom():
    # no inventory_expansion configured -> expansion toggle invalid; the bundled
    # replenishment must not have already claimed headroom (P2).
    p = make_producer(inv_limit=40)
    assert p.inventory_expansion is None
    with pytest.raises(RuntimeError, match="no inventory expansion"):
        p.handle_cmd(ProducingAgentCommand(False, False, do_inventory_expansion=True, replenishments={10: 5}))
    assert p.outstanding_inbound_units() == 0  # no ghost inbound
    assert p.headroom() == 40


def _inv_order(oid, itype, price=10.0, cost=None):
    ps = ProcessSheet(product=10, require={"capacity-1": 4})
    return Order.new_investment_order(oid, ps, InvestmentRequest("p", itype, 10, price, cost))


def _repl_order_for(oid, price=10.0, qty=2, cost=None):
    ps = ProcessSheet(product=10, require={"capacity-1": 4})
    return Order.new_replenishment_order(oid, ps, ReplenishmentRequest("p", 10, qty, price, cost))


def _priority_orders(cost=4):
    # one producer "p", same cut-off price -> stage-1 hands it the whole fund;
    # stage-2 fills survival-first restoration(0) < replenishment(1) < upgrade(2)
    # < expansion(3), each up to its per-order cost (ADR-0025/0027).
    return {
        "rest": _inv_order("rest", InvestmentType.RESTORATION, cost=cost),
        "repl": _repl_order_for("repl", cost=cost),
        "upg": _inv_order("upg", InvestmentType.UPGRADE, cost=cost),
        "exp": _inv_order("exp", InvestmentType.INVENTORY_EXPANSION, cost=cost),
    }


def test_greedy_fills_by_priority_to_each_cost_then_expires_rest():
    orders = _priority_orders(cost=4)
    distribute_investment_fund(orders, 10)  # 4 + 4 + remainder 2; expansion starves
    assert orders["rest"].tokens == 4  # filled to cost
    assert orders["repl"].tokens == 4  # filled to cost
    assert orders["upg"].tokens == 2  # boundary order: only the remainder
    assert orders["exp"].requires_funding()  # nothing left -> expires


def test_ample_budget_caps_each_order_at_its_cost():
    orders = _priority_orders(cost=4)
    distribute_investment_fund(orders, 100)
    assert all(o.tokens == 4 for o in orders.values())  # capped at cost, no waste
    assert all(not o.requires_funding() for o in orders.values())


def test_scarce_budget_funds_only_top_priority():
    orders = _priority_orders(cost=4)
    distribute_investment_fund(orders, 4)  # exactly restoration's cost
    assert orders["rest"].tokens == 4 and not orders["rest"].requires_funding()
    for k in ("repl", "upg", "exp"):
        assert orders[k].requires_funding()  # survival-first: lower priorities starve


def test_no_zero_token_order_is_marked_funded():
    orders = _priority_orders(cost=4)
    distribute_investment_fund(orders, 5)  # rest=4, repl=1 (partial); upg/exp expire
    for o in orders.values():
        if not o.requires_funding():
            assert o.tokens >= 1  # funded => has tokens; no free execution


def test_expired_unfunded_order_is_rejected_without_crashing_the_cycle():
    """Under a budget too small for every request, the lowest-priority order
    expires unfunded. The cycle must still run: the expired order is rejected and
    its running flag cleared, not crash on an unfunded info() (ADR-0025)."""
    cfg = Configuration(
        cycle_emission=4,  # fund = 2 < the 3 producer orders p1 raises
        process_sheets=[ProcessSheet(product=10, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig(
                id="p1", capacities={"capacity-1": 100}, degradation=0,
                restoration=Restoration(require=10, restores=10, type="capacity-1"),
                upgrade=Upgrade(require=10, increases=10, type="capacity-1"),
                inventory_limit=20,
                inventory_expansion=InventoryExpansion(require=10, increases=5),
            )
        ],
        consumers=[],
    )
    system = build_system(cfg)
    system.producing_agent_action("p1", ProducingAgentCommand(True, True, do_inventory_expansion=True))
    system.start_ordering()  # must not raise on the unfunded (expired) order
    _drive_ordering(system)
    result = system.complete_cycle()  # must not raise
    assert result.investment_rejected >= 1  # the expired order counts as rejected
    # inventory-expansion (lowest priority) lost the budget -> flag cleared so it
    # can be retried next cycle
    assert not system.producing_agents["p1"].inventory_expansion_running


def _two_producer_config():
    # p1 makes primal product 10; p2 assembles composite 20 from input 10.
    return Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=10, require={"capacity-1": 2}),
            ProcessSheet(product=20, require={"capacity-2": 3}, inputs={10: 1}),
        ],
        producer_configs=[
            ProducingAgentConfig(
                id="p1", capacities={"capacity-1": 100}, degradation=0,
                restoration=Restoration(require=10, restores=10, type="capacity-1"),
                upgrade=Upgrade(require=10, increases=10, type="capacity-1"),
            ),
            ProducingAgentConfig(
                id="p2", capacities={"capacity-2": 100}, degradation=0,
                restoration=Restoration(require=20, restores=10, type="capacity-2"),
                upgrade=Upgrade(require=20, increases=10, type="capacity-2"),
                inventory_limit=50,
            ),
        ],
        consumers=[],
    )


def _drive_ordering(system):
    for aid, agent in system.ordering_agents.items():
        view = agent.view(system.producer_infos)
        cmd_orders = {}
        for oid, oi in view.incoming.items():
            pid = next(iter(view.eligible[oid]))
            cmd_orders[oid] = [SubOrder(pid, oi.quantity, oi.tokens)]
        system.ordering_agent_action(aid, OrderingAgentCommand(cmd_orders))


def test_replenishment_order_deposits_input_into_requester_inventory():
    """End-to-end: p2 orders input product 10; p1 makes it; it lands in p2's
    inventory, funded by the central fund (ADR-0025)."""
    system = build_system(_two_producer_config())
    system.producing_agent_action("p1", ProducingAgentCommand(False, False))
    system.producing_agent_action("p2", ProducingAgentCommand(False, False, replenishments={10: 5}))
    system.start_ordering()
    _drive_ordering(system)
    system.complete_cycle()
    assert system.producing_agents["p2"].inventory.get(10, 0) == 5
    assert system.producing_agents["p2"].outstanding_inbound_units() == 0
