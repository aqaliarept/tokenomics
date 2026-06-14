"""B2: multi-capacity producers + atomic single-cycle bundle auction (ADR-0025).

Drives ProducingAgent.produce() directly with synthetic bundle bids — no System
wiring — to pin the auction semantics: atomic multi-type acceptance, single-cycle
rejection of an over-large bundle (no carry-over), and per-type degradation.
"""

from __future__ import annotations

from tokenomics.domain.config import ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.producing_agent import Bid, ProducingAgent


def make_producer(capacities: dict[str, int], degradation: int = 0) -> ProducingAgent:
    t = next(iter(capacities))
    cfg = ProducingAgentConfig(
        id="p",
        capacities=capacities,
        degradation=degradation,
        restoration=Restoration(require=1, restores=10, type=t),
        upgrade=Upgrade(require=1, increases=10, type=t),
    )
    return ProducingAgent(cfg)


def test_bundle_unit_price_is_tokens_over_total_capacity():
    b = Bid({"capacity-1": 6, "capacity-2": 4}, tokens=100, order_id="o1")
    assert b.total_capacity() == 10
    assert b.capacity_unit_price() == 100 / 10


def test_atomic_multi_type_acceptance():
    """A bundle clears only when every required type fits; price-greedy order."""
    p = make_producer({"capacity-1": 10, "capacity-2": 10})
    # cheap bundle (price 5) and pricey bundle (price 10); both fit together.
    cheap = Bid({"capacity-1": 4, "capacity-2": 4}, tokens=40, order_id="cheap")
    pricey = Bid({"capacity-1": 5, "capacity-2": 5}, tokens=100, order_id="pricey")
    p.place_bids([cheap, pricey])
    result = p.produce()
    ids = {b.order_id for b in result.completed}
    assert ids == {"cheap", "pricey"}
    assert result.rejected == []
    # cut-off = lowest accepted price
    assert p.cut_off_price == 5.0


def test_over_large_bundle_rejected_not_carried_over():
    """A bundle exceeding remaining capacity is rejected this cycle (no carry-over);
    a smaller bundle still clears."""
    p = make_producer({"capacity-1": 10})
    big = Bid({"capacity-1": 15}, tokens=300, order_id="big")  # price 20, but too big
    small = Bid({"capacity-1": 6}, tokens=60, order_id="small")  # price 10, fits
    p.place_bids([big, small])
    result = p.produce()
    assert [b.order_id for b in result.completed] == ["small"]
    assert [b.order_id for b in result.rejected] == ["big"]
    assert p.reservations == []  # nothing booked across cycles
    # next cycle starts clean: no residual booking re-applied
    assert p.bids == []


def test_acceptance_is_atomic_across_types():
    """A bundle whose one type overflows is rejected whole, even if the other fits."""
    p = make_producer({"capacity-1": 100, "capacity-2": 10})
    bid = Bid({"capacity-1": 8, "capacity-2": 20}, tokens=280, order_id="o")
    p.place_bids([bid])
    result = p.produce()
    assert result.completed == []
    assert [b.order_id for b in result.rejected] == ["o"]


def test_per_type_degradation():
    p = make_producer({"capacity-1": 100, "capacity-2": 200}, degradation=10)
    p.produce()  # no bids; degradation still applies
    assert p.capacities == {"capacity-1": 90, "capacity-2": 180}
