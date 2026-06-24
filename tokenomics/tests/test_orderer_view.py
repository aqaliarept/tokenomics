"""EXP-0006A: OrdererEconomyView — the domain's authoritative feature bundle for
one ordering agent's orders, assembled from live producer/product/order state +
telemetry. The env layer normalizes this into the enriched observation; here we pin
the domain facts directly (testable outside RL).

Harness: build a System and advance it to the ORDERING phase (consumer orders are
placed and funded in start_cycle/start_ordering), then read the view for the
consumer's ordering agent.
"""

from __future__ import annotations

from tokenomics.domain import build_system, load_config
from tokenomics.domain.common import from_consumer_id, is_nan
from tokenomics.domain.config import ProcessSheet, ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.order import Order
from tokenomics.domain.orderer_view import build_orderer_economy_view
from tokenomics.domain.producing_agent import ProducingAgent
from tokenomics.domain.requests import ConsumerRequest
from tokenomics.domain.telemetry import EconomyTelemetry
from tokenomics.env.aec_env import DEFAULT_CONFIG

CONSUMER_PRODUCT = 30


def _system_at_ordering():
    config = load_config(DEFAULT_CONFIG)
    system = build_system(config)
    system.start_ordering()  # distribute fund, place funded orders into orderers
    return system, config


def test_view_order_slot_features_for_a_fresh_consumer_order():
    system, config = _system_at_ordering()
    aid = from_consumer_id("c1")
    view = system.orderer_economy_view(aid)

    assert len(view.order_slots) == 1
    slot = view.order_slots[0]
    assert slot.product == CONSUMER_PRODUCT
    assert slot.unresolved_quantity == 1
    assert slot.available_tokens > 0
    assert slot.order_age == 0  # placed this cycle
    assert slot.ttl_remaining == 2  # ORDER_TTL - age
    assert slot.order_type == (1.0, 0.0, 0.0)  # consumer

    # eligibility is authoritative (ADR-0028): count + per-producer mask agree.
    expected_eligible = {
        p.id for p in config.producer_configs if CONSUMER_PRODUCT in p.produces
    }
    assert slot.eligible_producer_count == len(expected_eligible)
    assert {pid for pid, m in slot.eligible_mask.items() if m == 1.0} == expected_eligible


def test_view_producer_features_at_fresh_start():
    system, config = _system_at_ordering()
    view = system.orderer_economy_view(from_consumer_id("c1"))

    cfg = config.producer_configs[0]
    ps = view.producers[cfg.id]
    total = sum(cfg.capacities.values())
    # at a fresh start capacity is full: current == max, no headroom, ratios 1.0.
    assert ps.total_current_capacity == total
    assert ps.total_max_capacity == total
    assert ps.capacity_headroom == 0
    assert ps.min_capacity_ratio == 1.0
    assert ps.mean_capacity_ratio == 1.0
    # no load, no in-flight investments, no production history yet.
    assert ps.requested_load == 0
    assert ps.reserved_load == 0
    assert ps.total_load == 0
    assert ps.outstanding_inbound_units == 0
    assert ps.active_investments == (0.0, 0.0, 0.0)
    assert ps.acceptance_rate == 0.0
    assert ps.completion_rate == 0.0
    assert ps.rejection_rate == 0.0
    assert is_nan(ps.cut_off_price)  # no auction has cleared yet


def test_view_product_features_for_consumer_product():
    system, _ = _system_at_ordering()
    view = system.orderer_economy_view(from_consumer_id("c1"))

    px = view.products[CONSUMER_PRODUCT]
    # current open consumer demand (1 unit, backed by tokens) for product 30.
    assert px.open_demand_quantity == 1
    assert px.open_demand_tokens > 0
    # static recipe facts: 30 = capacity-4 + inputs {20, 11}, two layers deep.
    assert px.product_depth == 2
    assert px.recipe_input_count == 2
    assert px.recipe_input_units == 2  # 20:1 + 11:1
    assert px.recipe_capacity_type_count == 1
    assert px.eligible_producer_count == 1  # only p4 declares 30
    # no folded history yet at the first ordering phase.
    assert px.recent_average_cycle_demand == 0.0
    assert px.recent_rejected_quantity == 0.0


def test_view_producer_product_makeability_grid():
    system, _ = _system_at_ordering()
    view = system.orderer_economy_view(from_consumer_id("c1"))
    oid = view.order_slots[0].order_id

    # p4 makes 30 with capacity-4: capacity affords 4 units, but inputs (20, 11) are
    # not yet in stock, so it can ACCEPT 4 but COMPLETE none right now.
    pp = view.producer_products[(oid, "p4")]
    assert pp.eligible == 1.0
    assert pp.accept_qty_now == 4  # 120 // 25
    assert pp.complete_qty_now == 0  # no input stock
    assert pp.makeable_qty_now == 0
    assert pp.slot_makeability == 0.0  # can deliver none of the 1-unit order now
    assert pp.capacity_bottleneck_ratio > 0.0  # capacity is not the binding limit
    assert pp.bottleneck_ratio == 0.0  # inputs are

    # an ineligible producer contributes nothing for this product.
    other = view.producer_products[(oid, "p1")]
    assert other.eligible == 0.0
    assert other.makeable_qty_now == 0
    assert other.bottleneck_ratio == 0.0


def test_recent_rejected_qty_includes_failed_bids():
    """The catalog defines recent_rejected_qty as rejected-OR-failed quantity, so a
    composite reservation that fails after acceptance (TTL/degradation) still shows
    recent rejection pressure to the orderer — not just auction rejections."""
    sheet = ProcessSheet(product=99, require={"capacity-1": 1})
    cfg = ProducingAgentConfig(
        id="p1", capacities={"capacity-1": 10}, degradation=0,
        restoration=Restoration("r", 10, "capacity-1"), upgrade=Upgrade("u", 10, "capacity-1"),
        produces=frozenset({99}),
    )
    producer = ProducingAgent(cfg, sheets={99: sheet})

    telemetry = EconomyTelemetry(alpha=0.5)
    telemetry.record_failed("p1", 99, 4, routed_cycle=1, cycle=3)  # failed, not rejected
    telemetry.end_cycle()
    assert telemetry.rejected_ema("p1", 99) == 0.0  # nothing auction-rejected

    order = Order.new_consumer_order("o1", sheet, ConsumerRequest("c1", 99, tokens=10, quantity=1))
    view = build_orderer_economy_view(
        {"o1": order}, {"p1": producer}, telemetry, {99: sheet}, {"o1": order}
    )
    pp = view.producer_products[("o1", "p1")]
    assert pp.recent_rejected_qty == telemetry.failed_ema("p1", 99) > 0.0
