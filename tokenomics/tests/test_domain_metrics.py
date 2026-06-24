"""EXP-0006A: pure snapshot-derived economy metrics (no persistent state).

These formulas back the enriched orderer observation. They live in the domain so
they are testable outside RL and reusable by white-box diagnostics/search
(ALPHAZERO_EXPERT_ITERATION_APPROACH.md, "Implementation boundary"). One test per
formula; each asserts the canonical formula from the doc's "Feature calculation
guide", not an implementation detail.
"""

from __future__ import annotations

from tokenomics.domain import metrics
from tokenomics.domain.config import (
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.producing_agent import Bid, ProducingAgent, _Reservation


def make_producer(
    capacities: dict[str, int],
    *,
    produces: frozenset[str] | None = None,
    sheets: dict[str, ProcessSheet] | None = None,
    degradation: int = 0,
    inventory_limit: int = 0,
) -> ProducingAgent:
    t = next(iter(capacities))
    cfg = ProducingAgentConfig(
        id="p",
        capacities=capacities,
        degradation=degradation,
        restoration=Restoration(require="r", restores=10, type=t),
        upgrade=Upgrade(require="u", increases=10, type=t),
        produces=produces if produces is not None else frozenset(),
        inventory_limit=inventory_limit,
    )
    return ProducingAgent(cfg, sheets)


def test_producer_capacity_summary_sums_over_types():
    """total current / total max / headroom = the doc's producer capacity block.
    Headroom is the per-type max-minus-current gap summed (never negative)."""
    p = make_producer({"capacity-1": 100, "capacity-2": 60})
    # degrade current below max so headroom is a real gap, not zero.
    p.capacities["capacity-1"] = 40  # max stays 100 -> headroom 60
    p.capacities["capacity-2"] = 60  # max 60 -> headroom 0

    assert metrics.producer_total_current_capacity(p) == 100
    assert metrics.producer_total_max_capacity(p) == 160
    assert metrics.producer_capacity_headroom(p) == 60


def test_capacity_ratio_min_and_mean_over_owned_types():
    """capacity_ratio[p,t] = C/max(1,M); the producer summary reduces it to the
    min (worst-degraded type) and mean across owned types."""
    p = make_producer({"capacity-1": 100, "capacity-2": 200})
    p.capacities["capacity-1"] = 25  # ratio 0.25
    p.capacities["capacity-2"] = 150  # ratio 0.75

    assert metrics.producer_min_capacity_ratio(p) == 0.25
    assert metrics.producer_mean_capacity_ratio(p) == 0.5


def test_producer_load_is_requested_plus_reserved_capacity():
    """requested_load = last cycle's requested capacity; reserved_load = capacity
    held by live composite reservations; total = their sum (the doc's load block)."""
    p = make_producer({"capacity-1": 100})
    p.requested_capacity = 30
    p.reservations = [
        _Reservation(Bid({"capacity-1": 5, "capacity-2": 3}, tokens=10, order_id="a")),
        _Reservation(Bid({"capacity-1": 4}, tokens=8, order_id="b")),
    ]
    assert metrics.producer_requested_load(p) == 30
    assert metrics.producer_reserved_load(p) == 12  # (5+3) + 4
    assert metrics.producer_total_load(p) == 42


def test_producer_in_flight_inbound_units_and_active_investment_onehot():
    """outstanding-inbound = units ordered but not delivered (per the doc); active
    investments one-hot is (restoration, upgrade, inventory_expansion)."""
    p = make_producer({"capacity-1": 100})
    p.outstanding_inbound = {"g1": 4, "g2": 3}
    p.restoration_running = True
    p.inventory_expansion_running = True
    assert metrics.producer_outstanding_inbound_units(p) == 7
    assert metrics.producer_active_investments(p) == (1.0, 0.0, 1.0)


def test_capacity_makeable_qty_is_min_over_required_types():
    """capacity_makeable_qty[p,x] = min_t floor(C[p,t]/R[x,t]); 0 if x not declared
    (eligibility is authoritative, ADR-0028, not derived from capacity)."""
    sheet = ProcessSheet(product="x", require={"capacity-1": 10, "capacity-2": 30})
    p = make_producer(
        {"capacity-1": 100, "capacity-2": 60}, produces=frozenset({"x"}), sheets={"x": sheet}
    )
    # capacity-1 affords 10 units, capacity-2 only 2 -> the binding type wins.
    assert metrics.capacity_makeable_qty(p, "x") == 2

    # not declared -> 0 even though capacity would suffice for some other product.
    other = make_producer({"capacity-1": 100}, produces=frozenset(), sheets={})
    assert metrics.capacity_makeable_qty(other, "x") == 0


def test_makeability_and_bottleneck_exclude_reserved_capacity():
    """accept/bottleneck describe NEW routing, and the auction subtracts live
    reservations before accepting new bids — so these use available capacity
    (current minus reserved), not raw current."""
    sheet = ProcessSheet(product="x", require={"capacity-1": 10})
    p = make_producer({"capacity-1": 100}, produces=frozenset({"x"}), sheets={"x": sheet})
    assert metrics.accept_qty_now(p, "x") == 10  # all capacity free

    # a live reservation holds 60 of capacity-1; only 40 is available to new bids.
    p.reservations = [_Reservation(Bid({"capacity-1": 60}, tokens=1, order_id="r"))]
    assert metrics.accept_qty_now(p, "x") == 4  # 40 // 10
    assert metrics.capacity_makeable_qty(p, "x") == 4
    assert metrics.capacity_bottleneck_ratio(p, "x", 2) == 40 / (2 * 10)  # available / demand
    assert metrics.slot_makeability(p, "x", q=8) == 0.5  # min(8, 4)/8


def test_input_makeable_qty_is_min_over_inputs_infinite_when_primal():
    """input_makeable_qty[p,x] = min_g floor(Inv[p,g]/I[x,g]); a primal product (no
    inputs) is unconstrained by inventory -> infinity."""
    composite = ProcessSheet(
        product="c", require={"capacity-1": 1}, inputs={"g1": 2, "g2": 1}
    )
    primal = ProcessSheet(product="x", require={"capacity-1": 1})
    p = make_producer(
        {"capacity-1": 100},
        produces=frozenset({"c", "x"}),
        sheets={"c": composite, "x": primal},
    )
    p.inventory = {"g1": 10, "g2": 3}
    assert metrics.input_makeable_qty(p, "c") == 3  # min(10//2, 3//1)
    assert metrics.input_makeable_qty(p, "x") == float("inf")


def test_accept_vs_complete_qty_now_split_on_input_stock():
    """accept_qty_now = capacity alone clears the auction (composite may reserve);
    complete_qty_now = min(capacity, input) delivers immediately. makeable_qty_now
    defaults to complete_qty_now (the doc's recommended meaning)."""
    composite = ProcessSheet(
        product="c", require={"capacity-1": 10}, inputs={"g1": 2}
    )
    p = make_producer(
        {"capacity-1": 100}, produces=frozenset({"c"}), sheets={"c": composite}
    )
    p.inventory = {"g1": 6}  # inputs only afford 3, capacity affords 10
    assert metrics.accept_qty_now(p, "c") == 10
    assert metrics.complete_qty_now(p, "c") == 3
    assert metrics.makeable_qty_now(p, "c") == 3


def test_bottleneck_ratios_for_quantity_q():
    """capacity/input bottleneck ratios = supply / demand for qty q (can exceed 1
    when over-supplied); combined = the binding min. Primal input ratio = 1.0."""
    composite = ProcessSheet(
        product="c", require={"capacity-1": 10}, inputs={"g1": 2}
    )
    primal = ProcessSheet(product="x", require={"capacity-1": 10})
    p = make_producer(
        {"capacity-1": 100},
        produces=frozenset({"c", "x"}),
        sheets={"c": composite, "x": primal},
    )
    p.inventory = {"g1": 6}
    q = 4
    assert metrics.capacity_bottleneck_ratio(p, "c", q) == 100 / (4 * 10)  # 2.5
    assert metrics.input_bottleneck_ratio(p, "c", q) == 6 / (4 * 2)  # 0.75
    assert metrics.input_bottleneck_ratio(p, "x", q) == 1.0  # primal
    assert metrics.producer_product_bottleneck_ratio(p, "c", q) == 0.75  # binding min


def test_slot_makeability_is_fraction_of_order_makeable_now():
    """slot_makeability = min(q, makeable_qty_now[p,x]) / max(1,q): the fraction of
    a q-unit order this producer can deliver now, capped at 1.0."""
    composite = ProcessSheet(
        product="c", require={"capacity-1": 10}, inputs={"g1": 2}
    )
    p = make_producer(
        {"capacity-1": 100}, produces=frozenset({"c"}), sheets={"c": composite}
    )
    p.inventory = {"g1": 6}  # makeable_qty_now = 3
    assert metrics.slot_makeability(p, "c", q=4) == 0.75  # min(4,3)/4
    assert metrics.slot_makeability(p, "c", q=2) == 1.0  # min(2,3)/2, capped
    # ineligible producer can make none of it.
    other = make_producer({"capacity-1": 100}, produces=frozenset(), sheets={})
    assert metrics.slot_makeability(other, "c", q=4) == 0.0


def test_product_depth_is_one_plus_deepest_input():
    """product_depth[x] = 0 for primal; else 1 + max depth over its inputs (the
    product's layer in the economy graph)."""
    sheets = {
        "x": ProcessSheet(product="x", require={"capacity-1": 1}),  # primal
        "c1": ProcessSheet(product="c1", require={"capacity-1": 1}, inputs={"x": 1}),
        "c2": ProcessSheet(
            product="c2", require={"capacity-1": 1}, inputs={"c1": 1, "x": 1}
        ),
    }
    assert metrics.product_depth("x", sheets) == 0
    assert metrics.product_depth("c1", sheets) == 1
    assert metrics.product_depth("c2", sheets) == 2  # 1 + max(depth c1=1, depth x=0)


def test_product_depth_terminates_on_circular_recipes():
    """The config model supports cycle-safe recipe closure (inventory breaks the
    circular dependency), so depth must break cycles, not recurse forever. A back
    edge to a product already on the path is cut (counts 0)."""
    sheets = {
        "a": ProcessSheet(product="a", require={"capacity-1": 1}, inputs={"b": 1}),
        "b": ProcessSheet(product="b", require={"capacity-1": 1}, inputs={"a": 1}),
    }
    # a -> b -> (a is a back edge, cut): b contributes depth 1, so a is 1 + 1.
    assert metrics.product_depth("a", sheets) == 2
    assert metrics.product_depth("b", sheets) == 2


def test_recipe_complexity_and_eligible_producer_count():
    """Static recipe shape: input count / total input units / capacity-type count,
    plus how many producers declare the product (authoritative eligibility)."""
    sheet = ProcessSheet(
        product="c",
        require={"capacity-1": 1, "capacity-2": 1},
        inputs={"g1": 2, "g2": 3},
    )
    assert metrics.recipe_input_count(sheet) == 2
    assert metrics.recipe_input_units(sheet) == 5
    assert metrics.recipe_capacity_type_count(sheet) == 2

    maker = make_producer({"capacity-1": 1}, produces=frozenset({"c"}))
    bystander = make_producer({"capacity-1": 1}, produces=frozenset({"other"}))
    assert metrics.eligible_producer_count("c", [maker, bystander]) == 1
