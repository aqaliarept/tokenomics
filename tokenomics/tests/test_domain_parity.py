"""Domain port sanity + parity tests.

These drive the pure-Python System through full cycles using a trivial scripted
policy and assert invariants that must hold for a faithful port of the Go logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.domain import (
    OrderingAgentCommand,
    ProducingAgentCommand,
    SystemState,
    build_system,
    load_config,
)
from tokenomics.domain.producing_agent import Bid

CONFIG = Path(__file__).resolve().parents[1] / "tokenomics" / "config" / "phase1.json"


def load():
    return load_config(CONFIG)


def run_cycle(system, *, restore=False, upgrade=False):
    """Drive one full cycle: producers (no/optional invest) -> ordering agents
    (split each order's budget evenly across required producers) -> complete."""
    assert system.state == SystemState.ORDERS_PLACEMENT
    for pid in list(system.producing_agents):
        p = system.producing_agents[pid]
        # can't request an investment that's already in flight
        want_restore = restore and not p.restoration_running
        want_upgrade = upgrade and not p.upgrade_running
        system.producing_agent_action(
            pid, ProducingAgentCommand(do_restoration=want_restore, do_upgrade=want_upgrade)
        )
    system.start_ordering()

    producer_by_type: dict[str, str] = {}
    for pid, p in system.producing_agents.items():
        producer_by_type[p.capacity_type] = pid

    for aid, agent in system.ordering_agents.items():
        orders_cmd: dict[str, dict[str, int]] = {}
        for oid, oi in agent.incoming.items():
            req = oi.required
            n = len(req)
            budget = oi.tokens
            # even split, last producer absorbs the rounding residual
            alloc: dict[str, int] = {}
            types = list(req.keys())
            running = 0
            for i, ct in enumerate(types):
                pid = producer_by_type[ct]
                if i == n - 1:
                    alloc[pid] = budget - running
                else:
                    share = budget // n
                    alloc[pid] = share
                    running += share
            orders_cmd[oid] = alloc
        system.ordering_agent_action(aid, OrderingAgentCommand(orders_cmd))
    return system.complete_cycle()


def test_config_loads_and_validates():
    cfg = load()
    assert cfg.cycle_emission == 1000
    assert len(cfg.producer_configs) == 4
    assert len(cfg.consumers) == 4
    assert len(cfg.process_sheets) == 4


def test_emission_split():
    cfg = load()
    system = build_system(cfg)
    # half to investment fund
    assert system.investment_fund == 500
    # half split across 4 consumers = 125 each -> each placed a consumer order
    consumer_orders = [o for o in system.orders.values() if o.consumer_request is not None]
    assert len(consumer_orders) == 4
    assert all(o.tokens == 125 for o in consumer_orders)


def test_bid_unit_price_float32():
    # 100 tokens / 8 capacity, computed in float32 like Go
    b = Bid("capacity-1", 8, 100, "o1")
    assert b.capacity_unit_price() == pytest.approx(12.5, rel=1e-6)


def test_single_cycle_runs_and_scores():
    cfg = load()
    system = build_system(cfg)
    result = run_cycle(system)
    # score is a non-negative penalty sum; cycle advanced
    assert result.score >= 0
    assert system.cycle_counter == 2  # started at 1 after init, +1 after complete
    assert system.state == SystemState.ORDERS_PLACEMENT


def test_degradation_without_investment_kills_system():
    """No restoration/upgrade ever -> capacity degrades monotonically toward 0."""
    cfg = load()
    system = build_system(cfg)
    start_caps = {pid: p.capacity for pid, p in system.producing_agents.items()}
    for _ in range(60):
        run_cycle(system)
    end_caps = {pid: p.capacity for pid, p in system.producing_agents.items()}
    for pid in start_caps:
        assert end_caps[pid] <= start_caps[pid]
    # at least one producer should have hit zero given continuous degradation
    assert any(c == 0 for c in end_caps.values())


def test_restoration_recovers_capacity():
    cfg = load()
    system = build_system(cfg)
    # degrade for a few cycles
    for _ in range(5):
        run_cycle(system)
    degraded = {pid: p.capacity for pid, p in system.producing_agents.items()}
    # request restoration; run enough cycles for investment orders to flow
    for _ in range(10):
        run_cycle(system, restore=True)
    recovered = {pid: p.capacity for pid, p in system.producing_agents.items()}
    # at least one producer recovered above its degraded snapshot at some point
    assert any(recovered[pid] >= degraded[pid] for pid in degraded)


def test_long_run_no_crash():
    cfg = load()
    system = build_system(cfg)
    for _ in range(200):
        run_cycle(system, restore=True)
    # system survived the full loop without raising
    assert system.cycle_counter >= 200
