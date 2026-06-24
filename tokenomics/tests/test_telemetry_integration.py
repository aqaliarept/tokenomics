"""EXP-0006A: the System feeds its EconomyTelemetry from the real cycle, so the
enriched orderer observation is backed by realized routing/production history (not a
second hidden model). Driven through a full episode with baseline policies.
"""

from __future__ import annotations

import pytest

from tokenomics.domain import build_system, load_config
from tokenomics.domain.order import Order
from tokenomics.domain.producing_agent import Bid, _Reservation
from tokenomics.domain.requests import ConsumerRequest
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.aec_env import DEFAULT_CONFIG
from tokenomics.eval.baselines import (
    ordering_proportional,
    producing_always_both,
    producing_noop,
)
from tokenomics.eval.runner import run_episode

# phase1 consumer demand: product 30 at quantity 1 each cycle (only orderable
# product that is consumer demand; the rest are investment/replenishment inputs).
CONSUMER_PRODUCT = 30


def _run(horizon: int = 12) -> TokenomicsAECEnv:
    env = TokenomicsAECEnv(horizon=horizon)
    run_episode(env, producing_always_both, ordering_proportional, seed=0)
    return env


def test_system_records_routing_and_production_into_telemetry():
    env = _run()
    tele = env.system.telemetry
    spec = env.spec_

    routed = sum(
        tele.routed_ema(pid, x)
        for pid in spec.producer_ids
        for x in spec.order_products
    )
    completed = sum(
        tele.completed_ema(pid, x)
        for pid in spec.producer_ids
        for x in spec.order_products
    )
    assert routed > 0.0  # consumer orders were routed to producers
    assert completed > 0.0  # and some of them were produced


def test_telemetry_records_consumer_demand_only_for_demanded_product():
    env = _run()
    tele = env.system.telemetry
    spec = env.spec_
    # consumer demand history is recorded for the demanded product...
    assert tele.demand_ema(CONSUMER_PRODUCT) > 0.0
    # ...and not for products that are never consumer-demanded (inputs/investments).
    for x in spec.order_products:
        if x != CONSUMER_PRODUCT:
            assert tele.demand_ema(x) == 0.0


def test_telemetry_records_rejected_consumer_demand_on_collapse():
    """A noop producer lets capacity degrade to death, so consumer orders get
    rejected — the rejected-demand EMA must reflect it."""
    env = TokenomicsAECEnv(horizon=60)
    stats = run_episode(env, producing_noop, ordering_proportional, seed=0)
    assert stats["consumer_rejected_units"] > 0  # precondition: collapse rejected demand
    assert env.system.telemetry.rejected_demand_ema(CONSUMER_PRODUCT) > 0.0


def test_production_latency_dates_from_per_bid_route_cycle():
    """A reservation accepted at its route cycle but completed cycles later must age
    its true latency — the route cycle travels on the bid, so a same-order remainder
    re-routed in between cannot overwrite it."""
    system = build_system(load_config(DEFAULT_CONFIG))
    system.cycle_counter = 3
    bid = Bid({"capacity-4": 25}, tokens=10, order_id="o", product=CONSUMER_PRODUCT,
              quantity=2, routed_cycle=1)
    system._record_production("completed", "p4", bid)
    system.telemetry.end_cycle()
    # latency = 3 - 1 = 2; single quantity-weighted event at default alpha 0.3.
    assert system.telemetry.completion_latency("p4", CONSUMER_PRODUCT) == pytest.approx(0.3 * 2)


def test_orphaned_reservation_recorded_as_failed():
    """A composite reservation that outlives its parent order (the order's TTL fires
    while the reservation still waits) is accepted capacity that never delivers — it
    must surface as failed production (failure pressure the enriched orderer needs),
    not be silently dropped."""
    system = build_system(load_config(DEFAULT_CONFIG))
    system.start_ordering()  # state -> ORDERING so complete_cycle is callable
    p4 = system.producing_agents["p4"]

    # plant a live composite reservation for product 30: its inputs (20, 11) are not
    # in stock, so produce() leaves it waiting (ages, not TTL'd) this cycle.
    bid = Bid({"capacity-4": 25}, tokens=5, order_id="ghost", product=CONSUMER_PRODUCT,
              inputs={20: 1, 11: 1}, quantity=1, routed_cycle=0)
    p4.reservations = [_Reservation(bid)]

    # plant the parent order so it rejects THIS cycle (at ORDER_TTL) -> lands in
    # `removed`, orphaning the still-live reservation.
    order = Order.new_consumer_order(
        "ghost", system.process_sheets[CONSUMER_PRODUCT],
        ConsumerRequest("c1", CONSUMER_PRODUCT, tokens=5, quantity=1),
    )
    order.cycle_counter = 2  # == ORDER_TTL
    order.accepted_qty = 1   # fully accepted as the reservation (reserved still pending)
    order.burnt_tokens = 5
    system.orders["ghost"] = order

    system.complete_cycle()

    assert system.telemetry.failed_ema("p4", CONSUMER_PRODUCT) > 0.0
    assert system.telemetry.rejected_or_failed_ema("p4", CONSUMER_PRODUCT) > 0.0
    assert all(r.bid.order_id != "ghost" for r in p4.reservations)  # dropped
