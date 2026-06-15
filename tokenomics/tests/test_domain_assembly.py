"""B3: inventory + composite assembly with capacity reserved at acceptance
(ADR-0025). Drives ProducingAgent.produce() directly with composite bundle bids.
"""

from __future__ import annotations

from tokenomics.domain.config import ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.producing_agent import Bid, ProducingAgent


def make_producer(capacities: dict[str, int], degradation: int = 0, inv_limit: int = 1000) -> ProducingAgent:
    t = next(iter(capacities))
    cfg = ProducingAgentConfig(
        id="p",
        capacities=capacities,
        degradation=degradation,
        restoration=Restoration(require=1, restores=10, type=t),
        upgrade=Upgrade(require=1, increases=10, type=t),
        inventory_limit=inv_limit,
    )
    return ProducingAgent(cfg)


def composite(caps, tokens, oid, inputs):
    return Bid(capacities=caps, tokens=tokens, order_id=oid, product=20, inputs=inputs)


def test_composite_assembles_same_cycle_when_inputs_in_stock():
    p = make_producer({"capacity-1": 10})
    p.inventory = {10: 2}
    bid = composite({"capacity-1": 5}, 50, "o", {10: 2})
    p.place_bids([bid])
    result = p.produce()
    assert [b.order_id for b in result.completed] == ["o"]
    assert [b.order_id for b in result.accepted] == ["o"]  # cleared + delivered same cycle
    assert result.failed == []
    assert p.inventory[10] == 0  # inputs consumed
    assert p.reservations == []


def test_composite_without_inputs_reserves_and_blocks_competing_bid():
    """Accepted composite holds its capacity while waiting for inputs, blocking a
    cheaper bid that would otherwise fit — the intended over-acceptance pressure."""
    p = make_producer({"capacity-1": 10})
    waiting = composite({"capacity-1": 8}, 80, "wait", {10: 1})  # price 10, no stock
    other = Bid({"capacity-1": 5}, 20, "other")  # price 4, primal, would fit alone
    p.place_bids([waiting, other])
    result = p.produce()
    assert [b.order_id for b in result.accepted] == ["wait"]  # cleared, now reserved
    assert result.completed == []  # not delivered (no inputs yet)
    assert [b.order_id for b in result.rejected] == ["other"]
    assert len(p.reservations) == 1


def test_reserved_composite_fails_on_ttl():
    p = make_producer({"capacity-1": 10})
    p.place_bids([composite({"capacity-1": 5}, 50, "o", {10: 1})])
    r1 = p.produce()  # accepted, reserved
    assert [b.order_id for b in r1.accepted] == ["o"]
    r2 = p.produce()  # still waiting (no stock ever arrives) -> not re-surfaced
    assert r2.accepted == [] and r2.completed == [] and r2.failed == []
    assert [r.bid.order_id for r in p.reservations] == ["o"]
    r3 = p.produce()  # TTL hit
    assert [b.order_id for b in r3.failed] == ["o"]
    assert p.reservations == []


def test_no_overcommit_second_composite_rejected_when_first_reserves():
    p = make_producer({"capacity-1": 10})
    first = composite({"capacity-1": 6}, 120, "first", {10: 1})  # price 20
    second = composite({"capacity-1": 6}, 60, "second", {10: 1})  # price 10, no room
    p.place_bids([first, second])
    result = p.produce()
    assert [b.order_id for b in result.accepted] == ["first"]
    assert [b.order_id for b in result.rejected] == ["second"]


def test_capacity_freed_when_reservation_expires_same_cycle():
    """A reservation TTL-failing this cycle must return its capacity before the
    new-bid auction, so a bid that now fits is accepted, not rejected (ADR-0025)."""
    p = make_producer({"capacity-1": 10})
    p.place_bids([composite({"capacity-1": 8}, 80, "wait", {10: 1})])  # reserves, no stock
    p.produce()  # cycle 1: reserved
    p.produce()  # cycle 2: still waiting
    fresh = Bid({"capacity-1": 8}, 80, "fresh")  # primal, needs the soon-freed capacity
    p.place_bids([fresh])
    r3 = p.produce()  # cycle 3: reservation TTLs and frees its 8 units
    assert [b.order_id for b in r3.failed] == ["wait"]
    assert [b.order_id for b in r3.completed] == ["fresh"]
    assert [b.order_id for b in r3.accepted] == ["fresh"]


def test_reserved_capacity_reports_held_units_per_type():
    p = make_producer({"capacity-1": 10, "capacity-2": 10})
    assert p.reserved_capacity() == {}  # nothing reserved yet
    p.place_bids([composite({"capacity-1": 6}, 60, "o", {10: 1})])  # no stock -> reserves
    p.produce()
    assert p.reserved_capacity() == {"capacity-1": 6}


def test_degradation_drops_cheapest_reservation_and_spares_disjoint_type():
    p = make_producer({"capacity-1": 10, "capacity-2": 10}, degradation=20)
    pricey = composite({"capacity-1": 5}, 100, "pricey", {10: 1})  # price 20, c1
    cheap = composite({"capacity-1": 5}, 50, "cheap", {10: 1})  # price 10, c1
    disjoint = composite({"capacity-2": 5}, 50, "disjoint", {10: 1})  # price 10, c2
    p.place_bids([pricey, cheap, disjoint])
    r1 = p.produce()  # all reserved; then c1,c2 degrade 10 -> 8
    assert {b.order_id for b in r1.accepted} == {"pricey", "cheap", "disjoint"}
    r2 = p.produce()  # revalidate against 8: c1 fits only one 5-unit reservation
    assert [b.order_id for b in r2.failed] == ["cheap"]  # cheapest competing for c1
    assert {r.bid.order_id for r in p.reservations} == {"pricey", "disjoint"}  # spared
