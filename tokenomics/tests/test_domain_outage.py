"""Producer-outage mechanics in the domain (ADR-0055; add-shocks-i-drift design D2;
stochastic-environment "Producer outage").

During an outage a producer accepts no new bid and assembles no reserved (in-flight)
production, which keeps aging toward its TTL as if its inputs were missing. Re-validation
and degradation run as usual, the cut-off price is kept, and the producer's own purchases
and deliveries are untouched. It accepts bids again in the cycle after the outage.
"""

from __future__ import annotations

from pathlib import Path

from tokenomics.domain import build_system
from tokenomics.domain.config import (Configuration, ProcessSheet, ProducingAgentConfig,
                                      Restoration, Upgrade, load_config)
from tokenomics.domain.ordering_agent import OrderingAgentCommand, SubOrder
from tokenomics.domain.producing_agent import (RESERVATION_TTL, ProducingAgent,
                                               ProducingAgentCommand)
from tokenomics.domain.producing_agent import SubOrder as RoutedSubOrder
from tokenomics.domain.shocks import ShockEvent, ShockKind, ShockTarget
from tokenomics.domain.system import CycleResult, System

PHASE1 = Path(__file__).resolve().parents[1] / "tokenomics" / "config" / "phase1.json"

# --- producer-level fixture: one primal (1) and one composite (2, needs one unit of 1) ---

PRIMAL, COMPOSITE, CAP = 1, 2, "c"
SHEETS = {
    PRIMAL: ProcessSheet(PRIMAL, {CAP: 2}),
    COMPOSITE: ProcessSheet(COMPOSITE, {CAP: 3}, inputs={PRIMAL: 1}),
}


def _producer(capacity: int = 100, degradation: int = 10) -> ProducingAgent:
    return ProducingAgent(ProducingAgentConfig(
        id="p", capacities={CAP: capacity}, degradation=degradation,
        restoration=Restoration(require=PRIMAL, restores=10, type=CAP),
        upgrade=Upgrade(require=PRIMAL, increases=10, type=CAP),
        produces=frozenset({PRIMAL, COMPOSITE}), inventory_limit=10,
    ), dict(SHEETS))


def _reserve_composite(p: ProducingAgent) -> None:
    """Cycle t-1: a composite bid is accepted with no inputs in stock, so it reserves."""
    p.place_sub_orders([RoutedSubOrder("o1", COMPOSITE, 1, 30)], routed_cycle=1)
    r = p.produce()
    assert [b.order_id for b in r.accepted] == ["o1"] and not r.completed
    assert len(p.reservations) == 1 and p.reservations[0].cycles_waited == 0


def test_outage_rejects_every_new_bid_and_keeps_the_cut_off_price() -> None:
    p = _producer()
    p.place_sub_orders([RoutedSubOrder("o0", PRIMAL, 1, 50)])
    p.produce()
    cut_off = p.cut_off_price
    p.place_sub_orders([RoutedSubOrder("o1", PRIMAL, 2, 90), RoutedSubOrder("o2", COMPOSITE, 1, 10)])
    r = p.produce(outage=True)
    assert not r.accepted and not r.completed and not r.failed
    assert sorted(b.order_id for b in r.rejected) == ["o1", "o2"]
    assert p.cut_off_price == cut_off and p.funds == 0
    assert p.requested_capacity == 2 * 2 + 3  # demand pressure is still recorded
    assert not p.reservations  # a rejected composite never reserves


def test_in_flight_production_during_an_outage() -> None:
    """Scenario "In-flight production during an outage": a composite reserved at t-1 whose
    inputs arrive is not completed at t, keeps aging, and fails at its reservation TTL."""
    assert RESERVATION_TTL == 2
    p = _producer()
    _reserve_composite(p)
    p.inventory[PRIMAL] = 1  # the inputs arrive for cycle t
    r_t = p.produce(outage=True)
    assert not r_t.completed and not r_t.failed
    assert p.reservations[0].cycles_waited == 1 and p.inventory[PRIMAL] == 1  # inputs untouched
    r_t1 = p.produce(outage=True)  # outage continues: TTL reached
    assert [b.order_id for b in r_t1.failed] == ["o1"] and not r_t1.completed
    assert not p.reservations and p.inventory[PRIMAL] == 1


def test_in_flight_production_resumes_after_a_short_outage() -> None:
    """The same reservation with a one-cycle outage completes when the producer resumes."""
    p = _producer()
    _reserve_composite(p)
    p.inventory[PRIMAL] = 1
    assert not p.produce(outage=True).completed
    r = p.produce()
    assert [b.order_id for b in r.completed] == ["o1"] and p.inventory[PRIMAL] == 0


def test_control_without_outage_completes_at_t() -> None:
    p = _producer()
    _reserve_composite(p)
    p.inventory[PRIMAL] = 1
    assert [b.order_id for b in p.produce().completed] == ["o1"]


def test_capacity_degrades_during_an_outage_like_idle_production() -> None:
    """Scenario "Capacity degrades during an outage" (producer level): capacity after each
    outage cycle equals pure degradation, even with bids offered."""
    out, idle = _producer(), _producer()
    for k in range(1, 6):
        out.place_sub_orders([RoutedSubOrder(f"o{k}", PRIMAL, 1, 40)])
        out.produce(outage=True)
        idle.produce()
        assert out.capacities[CAP] == idle.capacities[CAP] == 100 - 10 * k


def test_reservations_still_revalidated_during_an_outage() -> None:
    """Re-validation runs as usual: a reservation whose capacity degraded away fails."""
    p = _producer(capacity=4, degradation=50)  # 4 -> 2 after the reserving cycle
    _reserve_composite(p)  # needs 3 units of CAP
    r = p.produce(outage=True)
    assert [b.order_id for b in r.failed] == ["o1"]


# --- system level: phase1 (consumer c1 orders product 30 from p4) ---


def _outage(producer: str, start: int, duration: int) -> ShockEvent:
    return ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer=producer), start, duration,
                      1.0)


def _play(system: System, restore: frozenset[str] = frozenset()) -> CycleResult:
    for pid, p in system.producing_agents.items():
        system.producing_agent_action(
            pid, ProducingAgentCommand(pid in restore and not p.restoration_running, False))
    system.start_ordering()
    for aid, agent in system.ordering_agents.items():
        view = agent.view(system.producer_infos)
        orders = {oid: [SubOrder(next(iter(view.eligible[oid])), oi.quantity, oi.tokens)]
                  for oid, oi in view.incoming.items()}
        system.ordering_agent_action(aid, OrderingAgentCommand(orders))
    return system.complete_cycle()


def _auction(result: CycleResult, pid: str) -> tuple[int, int]:
    rec = next(r for r in result.producer_auctions if r.producer_id == pid)
    return rec.capacity, rec.accepted_units


def _phase1() -> Configuration:
    return load_config(PHASE1)


def test_orders_during_an_outage() -> None:
    """Scenario "Orders during an outage": sub-orders routed to P in cycles t..t+N-1 are not
    accepted, and P accepts bids again from cycle t+N."""
    t, n = 3, 3
    system = build_system(_phase1(), shock_schedule=(_outage("p4", t, n),))
    accepted: dict[int, int] = {}
    requested: dict[int, int] = {}
    for cycle in range(1, t + n + 2):
        assert system.cycle_counter == cycle
        result = _play(system)
        accepted[cycle] = _auction(result, "p4")[1]
        requested[cycle] = system.producing_agents["p4"].requested_capacity
    for cycle in range(t, t + n):
        assert requested[cycle] > 0, cycle  # a sub-order was routed to p4 ...
        assert accepted[cycle] == 0, cycle  # ... and not accepted
    assert accepted[t - 1] > 0
    assert accepted[t + n] > 0  # accepts bids again from t+N


def test_capacity_degrades_and_restoration_completes_during_an_outage() -> None:
    """Scenario "Capacity degrades during an outage" (system level): capacity follows normal
    degradation, and a restoration the producer bought completes inside the outage."""
    degradation = 6  # units per cycle at max 120 (5% in phase1)
    # outage covers cycles 2..10; p4 restores whenever possible
    shocked = build_system(_phase1(), shock_schedule=(_outage("p4", 2, 9),))
    restored_inside = False
    for cycle in range(1, 11):
        before = shocked.producing_agents["p4"].capacities["capacity-4"]
        result = _play(shocked, restore=frozenset({"p4"}))
        after = shocked.producing_agents["p4"].capacities["capacity-4"]
        if cycle >= 2:
            assert _auction(result, "p4")[1] == 0  # in outage: nothing accepted
            expected_decay = max(0, before - degradation)
            if after > expected_decay:
                restored_inside = True  # an investment completion lifted it
            else:
                assert after == expected_decay, cycle
    assert restored_inside
    assert not shocked.producing_agents["p4"].reservations


def test_outage_without_restoration_matches_pure_degradation() -> None:
    system = build_system(_phase1(), shock_schedule=(_outage("p4", 2, 6),))
    for cycle in range(1, 8):
        _play(system)
    # 7 played cycles of 6-unit decay from 120, no investment requested
    assert system.producing_agents["p4"].capacities["capacity-4"] == 120 - 7 * 6


def test_overlapping_outages_are_a_union_and_others_unaffected() -> None:
    system = build_system(_phase1(), shock_schedule=(_outage("p4", 2, 2), _outage("p4", 3, 3)))
    calm = build_system(_phase1())
    for cycle in range(1, 8):
        r, c = _play(system), _play(calm)
        if 2 <= cycle <= 5:
            assert _auction(r, "p4")[1] == 0, cycle
        # other producers' auctions are untouched until p4's rejections propagate upstream
        if cycle == 2:
            assert _auction(r, "p2") == _auction(c, "p2")
