"""B4: inventory-expansion investment, new system death, two-stage producer-first
fund (ADR-0025)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.domain import build_system, load_config
from tokenomics.domain.config import InventoryExpansion, ProcessSheet, ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.order import Order
from tokenomics.domain.producing_agent import ProducingAgent, ProducingAgentCommand
from tokenomics.domain.requests import InvestmentRequest, InvestmentType
from tokenomics.domain.system import distribute_investment_fund

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "composite.json"


def make_producer(capacities=None, inv_limit=10, expansion=8):
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
            inventory_expansion=InventoryExpansion(require=2, increases=expansion),
        )
    )


def test_loader_parses_inventory_expansion():
    cfg = load_config(FIXTURE)
    p1 = next(p for p in cfg.producer_configs if p.id == "p1")
    assert p1.inventory_limit == 40
    assert p1.inventory_expansion is not None
    assert p1.inventory_expansion.require == 11
    assert p1.inventory_expansion.increases == 25


def test_inventory_expansion_request_emitted():
    p = make_producer()
    reqs = p.handle_cmd(ProducingAgentCommand(do_restoration=False, do_upgrade=False, do_inventory_expansion=True))
    assert len(reqs) == 1
    assert reqs[0].type == InvestmentType.INVENTORY_EXPANSION
    assert reqs[0].product == 2  # the means-of-production product it requires


def test_completed_inventory_expansion_raises_limit():
    p = make_producer(inv_limit=10, expansion=8)
    [req] = p.handle_cmd(ProducingAgentCommand(False, False, do_inventory_expansion=True))
    p.investment_completed(req)
    assert p.inventory_limit == 18
    assert not p.inventory_expansion_running  # cleared on completion


def test_inventory_expansion_gated_while_running():
    p = make_producer()
    p.handle_cmd(ProducingAgentCommand(False, False, do_inventory_expansion=True))
    p.cmd_handled = False  # next cycle
    with pytest.raises(RuntimeError, match="expansion is running"):
        p.handle_cmd(ProducingAgentCommand(False, False, do_inventory_expansion=True))


def test_inventory_expansion_rejection_clears_running():
    p = make_producer()
    [req] = p.handle_cmd(ProducingAgentCommand(False, False, do_inventory_expansion=True))
    p.investment_rejected(req)  # must not raise "unsupported"
    assert not p.inventory_expansion_running
    p.cmd_handled = False
    reqs = p.handle_cmd(ProducingAgentCommand(False, False, do_inventory_expansion=True))
    assert len(reqs) == 1  # can re-request after rejection


def test_system_dead_when_a_capacity_type_has_no_living_owner():
    system = build_system(load_config(FIXTURE))
    assert not system.is_dead()
    # p1 is the sole owner of capacity-1; zero it -> that type is unmakeable
    system.producing_agents["p1"].capacities["capacity-1"] = 0
    assert system.is_dead()


def _inv_order(oid, pid, price):
    ps = ProcessSheet(product=1, require={"capacity-1": 10})
    req = InvestmentRequest(pid, InvestmentType.RESTORATION, 1, price)
    return Order.new_investment_order(oid, ps, req)


def test_fund_is_producer_first_not_order_count_weighted():
    """Two producers, equal cut-off price; B floods with more orders. Producer-first
    allocation gives each producer an equal total — order count must not raise B's
    share (ADR-0025)."""
    orders = {
        "a1": _inv_order("a1", "A", 10.0),
        "b1": _inv_order("b1", "B", 10.0),
        "b2": _inv_order("b2", "B", 10.0),
        "b3": _inv_order("b3", "B", 10.0),
    }
    distribute_investment_fund(orders, 1000)
    a_total = orders["a1"].tokens
    b_total = orders["b1"].tokens + orders["b2"].tokens + orders["b3"].tokens
    assert a_total == b_total
