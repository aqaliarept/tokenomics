"""EXP-0038B per-producer/product observation ladder: pure feature-block builders.

Torch-free. Each block is a pure read of a `ProducingAgent` snapshot + its process sheets
(the deployable, decision-time facts the current 7-feature row collapses). One behavior per test
(HANDOVER ladder P1-P4, P6). These are the builders the pod re-dump rides into producers.csv.
"""

from __future__ import annotations

from tokenomics.domain.config import (
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.producing_agent import ProducingAgent
from tokenomics.eval import producer_product_features as ppf


def _agent(
    caps: dict[str, int],
    produces: set[int],
    sheets: dict[int, ProcessSheet],
    *,
    max_caps: dict[str, int] | None = None,
    inventory: dict[int, int] | None = None,
    inbound: dict[int, int] | None = None,
) -> ProducingAgent:
    """A minimal live producer: owns `caps`, declares `produces`, holds the given input stock /
    inbound. `max_caps` defaults to current (fully restored). No reservations."""
    t0 = next(iter(caps))
    cfg = ProducingAgentConfig(
        id="p", capacities=dict(caps), degradation=0,
        restoration=Restoration(require=0, restores=0, type=t0),
        upgrade=Upgrade(require=0, increases=0, type=t0),
        produces=frozenset(produces), inventory_limit=9999,
    )
    p = ProducingAgent(cfg, sheets)
    if max_caps is not None:
        p.max_capacities = dict(max_caps)
    if inventory is not None:
        p.inventory = dict(inventory)
    if inbound is not None:
        p.outstanding_inbound = dict(inbound)
    return p


def test_pp_throughput_dispersion_separates_producers_tied_on_binding_min() -> None:
    # Product 1 needs two capacity types (c0: 2/unit, c1: 1/unit). Two producers BIND at the same
    # min throughput (5 units of product 1, capacity c0-limited) but differ on the NON-binding type.
    # A block that only logged the binding min would tie them; the dispersion stats (max, std) must
    # separate them so a within-decision re-ranker can prefer the one with recovery slack.
    sheets = {1: ProcessSheet(product=1, require={"c0": 2, "c1": 1})}
    tight = _agent({"c0": 10, "c1": 5}, {1}, sheets)   # c0->5, c1->5 : both bind at 5
    loose = _agent({"c0": 10, "c1": 50}, {1}, sheets)  # c0->5, c1->50: binds at 5, huge slack

    bt = ppf.pp_throughput_block(tight, product=1, quantity=3)
    bl = ppf.pp_throughput_block(loose, product=1, quantity=3)

    assert bt["pp_throughput_req_type_count"] == 2.0
    assert bt["pp_throughput_current_min"] == bl["pp_throughput_current_min"] == 5.0
    # non-binding dispersion distinguishes them
    assert bl["pp_throughput_current_max"] > bt["pp_throughput_current_max"]
    assert bl["pp_throughput_current_std"] > bt["pp_throughput_current_std"]
    assert bt["pp_throughput_current_std"] == 0.0
    # order of 3 units fits inside the binding min (5) -> slack 2, not zero
    assert bt["pp_throughput_binding_slack"] == 2.0
    assert bt["pp_throughput_binding_is_zero"] == 0.0
    assert bt["pp_throughput_qty_over_free"] == 3.0 / 5.0


def test_pp_dep_cover_is_in_units_of_target_not_raw_inventory() -> None:
    # Composite product 2 consumes 2x product 10 and 1x product 11 per unit. Producer holds lots of
    # product 10 (20) but little product 11 (5). Raw inventory total (25) is ambiguous — it says
    # nothing about how many units of product 2 it can cover. Cover in units of 2 is min(20/2, 5/1)
    # = 5, and the binding (bottleneck) input is product 11. For an 8-unit order it is short 8*1-5=3
    # units of product 11. That target-product cover, not the raw count, is the deployable signal.
    sheets = {2: ProcessSheet(product=2, require={"c0": 1}, inputs={10: 2, 11: 1})}
    p = _agent({"c0": 100}, {2}, sheets, inventory={10: 20, 11: 5})
    b = ppf.pp_dep_block(p, product=2, quantity=8)
    assert b["pp_dep_is_primal"] == 0.0
    assert b["pp_dep_input_count"] == 2.0
    assert b["pp_dep_cover_min"] == 5.0
    assert b["pp_dep_cover_max"] == 10.0
    assert b["pp_dep_bottleneck_cover"] == 5.0
    assert b["pp_dep_missing_units"] == 3.0


def test_pp_dep_primal_sentinel_distinguishes_primal_from_starved_composite() -> None:
    # A primal product (no inputs) has no dependency to cover; a composite producer that simply holds
    # no relevant input stock has zero cover. Both would read cover 0 if primal collapsed to an
    # accidental zero. The explicit is_primal flag (and a high cover sentinel) must separate them.
    prim_sheets = {1: ProcessSheet(product=1, require={"c0": 1})}
    primal = _agent({"c0": 10}, {1}, prim_sheets)
    comp_sheets = {2: ProcessSheet(product=2, require={"c0": 1}, inputs={10: 1})}
    starved = _agent({"c0": 10}, {2}, comp_sheets, inventory={10: 0})

    bp = ppf.pp_dep_block(primal, product=1, quantity=4)
    bs = ppf.pp_dep_block(starved, product=2, quantity=4)
    assert bp["pp_dep_is_primal"] == 1.0 and bs["pp_dep_is_primal"] == 0.0
    assert bp["pp_dep_cover_min"] > bs["pp_dep_cover_min"]
    assert bs["pp_dep_cover_min"] == 0.0
    assert bp["pp_dep_missing_units"] == 0.0
    assert bs["pp_dep_missing_units"] == 4.0


def test_pp_opp_cost_rises_only_when_alternate_shares_the_target_capacity() -> None:
    # Target product 1 needs capacity c0. Three producers all eligible for product 1 with equal c0.
    #  - solo: makes only product 1 -> no alternative use, zero opportunity cost.
    #  - shared: also makes product 2 which ALSO needs c0 and is in high demand -> using its c0 for
    #    product 1 forgoes product-2 throughput; real opportunity cost.
    #  - other: also makes product 3 which needs a DIFFERENT capacity c1 -> having an alternative
    #    does not compete for c0, so no opportunity cost. Capacity SHARING, not mere alternatives.
    sheets = {1: ProcessSheet(product=1, require={"c0": 1}),
              2: ProcessSheet(product=2, require={"c0": 1}),
              3: ProcessSheet(product=3, require={"c1": 1})}
    open_qty = {2: 20, 3: 20}  # both alternates equally demanded
    solo = _agent({"c0": 10}, {1}, sheets)
    shared = _agent({"c0": 10}, {1, 2}, sheets)
    other = _agent({"c0": 10, "c1": 10}, {1, 3}, sheets)

    bs = ppf.pp_opp_block(solo, product=1, quantity=4, open_qty=open_qty)
    bh = ppf.pp_opp_block(shared, product=1, quantity=4, open_qty=open_qty)
    bo = ppf.pp_opp_block(other, product=1, quantity=4, open_qty=open_qty)

    assert bs["pp_opp_alt_product_count"] == 0.0
    assert bh["pp_opp_alt_product_count"] == 1.0 and bo["pp_opp_alt_product_count"] == 1.0
    # shared alternate uses the target's capacity -> overlap 1, real opportunity cost
    assert bh["pp_opp_same_cap_overlap_max"] == 1.0
    assert bh["pp_opp_capacity_opportunity_cost"] > 0.0
    # different-capacity alternate does not compete for c0 -> zero overlap, zero opportunity cost
    assert bo["pp_opp_same_cap_overlap_max"] == 0.0
    assert bo["pp_opp_capacity_opportunity_cost"] == 0.0
    assert bs["pp_opp_capacity_opportunity_cost"] == 0.0
    assert bh["pp_opp_capacity_opportunity_cost"] > bo["pp_opp_capacity_opportunity_cost"]


def test_pp_base_ranks_producers_by_base_logit_with_margins_and_softmax() -> None:
    # The base router (blend_0.5) already assigns each eligible producer a logit. P4 exposes how far
    # the teacher would have to rerank against it: the top producer, its margin, and where each sits.
    # Logits a=2, b=1, c=1: a is the unique best; b and c tie one step below.
    import math
    base = ppf.pp_base_block({"a": 2.0, "b": 1.0, "c": 1.0})
    assert base["a"]["pp_base_rank"] == 0.0
    assert base["a"]["pp_base_margin_to_best"] == 0.0
    assert base["a"]["pp_base_margin_to_next"] == 0.0
    # b and c each have exactly one strictly-better producer (a) -> rank 1, margin -1 to it
    assert base["b"]["pp_base_rank"] == 1.0 and base["c"]["pp_base_rank"] == 1.0
    assert base["b"]["pp_base_margin_to_best"] == -1.0
    assert base["b"]["pp_base_margin_to_next"] == -1.0
    # softmax weights: highest logit gets the largest weight and the row sums to 1
    w = {k: base[k]["pp_base_softmax_weight"] for k in base}
    assert w["a"] > w["b"] and math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)


def test_pp_fulfill_estimates_ttl_feasibility_from_capacity_and_cover() -> None:
    from tokenomics.domain.telemetry import EconomyTelemetry
    sheets = {1: ProcessSheet(product=1, require={"c0": 1})}
    # p can make 10 units/cycle of product 1; a 4-unit order fits in one cycle, well inside ttl=2.
    p = _agent({"c0": 10}, {1}, sheets)
    tel = EconomyTelemetry()
    b = ppf.pp_fulfill_block(p, product=1, quantity=4, ttl_remaining=2, telemetry=tel)
    assert b["pp_fulfill_est_cycles_to_produce"] <= 1.0
    assert b["pp_fulfill_feasible_before_ttl"] == 1.0
    assert b["pp_fulfill_ttl_minus_est"] >= 1.0

    # A 40-unit order needs > ttl cycles at 10 units/cycle -> not feasible before ttl.
    b2 = ppf.pp_fulfill_block(p, product=1, quantity=40, ttl_remaining=2, telemetry=tel)
    assert b2["pp_fulfill_est_cycles_to_produce"] > 2.0
    assert b2["pp_fulfill_feasible_before_ttl"] == 0.0


def test_pp_fulfill_history_uses_only_past_folded_cycles_not_current() -> None:
    # Telemetry EMAs are deployable only if they reflect PAST cycles: an event recorded in the
    # current (not-yet-folded) cycle must not leak into the block. Fold one past cycle, snapshot the
    # block, then record a huge current-cycle completion WITHOUT folding — the block must not move.
    from tokenomics.domain.telemetry import EconomyTelemetry
    sheets = {1: ProcessSheet(product=1, require={"c0": 1})}
    p = _agent({"c0": 10}, {1}, sheets)
    tel = EconomyTelemetry()
    tel.record_completed("p", 1, 10, routed_cycle=0, cycle=1)
    tel.end_cycle()  # fold: this is now past history
    before = ppf.pp_fulfill_block(p, product=1, quantity=4, ttl_remaining=2, telemetry=tel)
    assert before["pp_fulfill_recent_completion_rate"] > 0.0

    tel.record_completed("p", 1, 1000, routed_cycle=1, cycle=2)  # current cycle, not folded
    after = ppf.pp_fulfill_block(p, product=1, quantity=4, ttl_remaining=2, telemetry=tel)
    assert after == before  # no current-cycle leakage -> deployable at decision time


# --- EXP-0038B Phase 2: system-wide (sys_*) blocks -----------------------------------
from tokenomics.domain.order import Order  # noqa: E402


def _order(oid: str, product: int, quantity: int, *, age: int = 0) -> Order:
    """A minimal live order for demand tests: only product/quantity/age are read."""
    o = Order(oid, 0, product, quantity, None, None, True)
    o.cycle_counter = age
    return o


def test_sys_capacity_aggregates_throughput_and_concentration() -> None:
    # Two eligible producers for target product 1 (c0: 1/unit): free throughputs 10 and 30. The block
    # sums economy-wide free capacity (40) and reports its concentration (HHI) — one producer holding
    # most free capacity makes the routing choice nearly forced. Global scarcity of the required type
    # = open-demand load / global supply.
    sheets = {1: ProcessSheet(product=1, require={"c0": 1})}
    a = _agent({"c0": 10}, {1}, sheets)
    b = _agent({"c0": 30}, {1}, sheets)
    blk = ppf.sys_capacity_block([a, b], product=1, sheets=sheets, all_agents=[a, b],
                                 open_qty={1: 5})
    assert blk["sys_capacity_target_total_free_units"] == 40.0
    assert blk["sys_capacity_target_eligible_producer_count"] == 2.0
    assert abs(blk["sys_capacity_target_free_hhi"] - ((10 / 40) ** 2 + (30 / 40) ** 2)) < 1e-9
    # scarcity of c0 = demand(5 units * 1/unit) / supply(40) = 0.125
    assert abs(blk["sys_capacity_required_type_scarcity_min"] - 0.125) < 1e-9


def test_sys_demand_counts_target_and_only_capacity_competing_products() -> None:
    # Target product 1 needs c0. Product 2 also needs c0 (competes for the same capacity); product 3
    # needs c1 (a different type — no competition). Only capacity-sharing demand counts as competing,
    # and the producer-varying field counts only the competing products THIS producer declares.
    sheets = {1: ProcessSheet(product=1, require={"c0": 1}),
              2: ProcessSheet(product=2, require={"c0": 1}),
              3: ProcessSheet(product=3, require={"c1": 1})}
    p = _agent({"c0": 10, "c1": 10}, {1, 2, 3}, sheets)
    open_qty = {1: 6, 2: 4, 3: 9}
    orders = [_order("o1", 1, 4, age=0), _order("o2", 1, 2, age=1)]
    blk = ppf.sys_demand_block(p, product=1, sheets=sheets, open_qty=open_qty,
                               target_orders=orders, this_orderer_target_qty=4)
    assert blk["sys_demand_target_queued_qty"] == 6.0
    assert blk["sys_demand_target_order_count"] == 2.0
    assert blk["sys_demand_target_oldest_age"] == 1.0
    assert blk["sys_demand_target_min_ttl"] == 1.0            # ORDER_TTL(2) - oldest age 1
    assert blk["sys_demand_same_product_other_orderers"] == 2.0   # 6 total - 4 this orderer
    assert blk["sys_demand_competing_products_qty"] == 4.0    # only product 2 shares c0
    assert blk["sys_demand_competing_for_p_capacity_qty"] == 4.0  # p makes 2 (c0), not 3 (c1)


def test_sys_recipe_depth_downstream_unlocked_and_transitive_inputs() -> None:
    # Chain 1(primal) <- 2 <- 3. depth(3)=2, its transitive inputs are {2,1}. Demand for product 2
    # (5 units, 2x product-1 each) pulls 10 units of product 1 as an input -> downstream unlocked.
    # A primal target has no upstream to cover -> high sentinel, not an accidental 0.
    sheets = {1: ProcessSheet(product=1, require={"c0": 1}),
              2: ProcessSheet(product=2, require={"c0": 1}, inputs={1: 2}),
              3: ProcessSheet(product=3, require={"c0": 1}, inputs={2: 1})}
    a = _agent({"c0": 100}, {1, 2, 3}, sheets)
    open_qty = {2: 5, 3: 4}
    b1 = ppf.sys_recipe_block(product=1, sheets=sheets, open_qty=open_qty, all_agents=[a])
    assert b1["sys_recipe_product_depth"] == 0.0
    assert b1["sys_recipe_input_count"] == 0.0
    assert b1["sys_recipe_downstream_demand_unlocked"] == 10.0
    assert b1["sys_recipe_upstream_capacity_cover_min"] == ppf._SYS_SENTINEL
    b3 = ppf.sys_recipe_block(product=3, sheets=sheets, open_qty=open_qty, all_agents=[a])
    assert b3["sys_recipe_product_depth"] == 2.0
    assert b3["sys_recipe_input_count"] == 1.0
    assert b3["sys_recipe_transitive_input_count"] == 2.0     # {2, 1}


def test_sys_cutoff_urgency_rises_as_ttl_falls_and_logs_no_faked_price() -> None:
    import math
    hi_ttl = ppf.sys_cutoff_block(quantity=5, ttl_remaining=2, oldest_age=0)
    lo_ttl = ppf.sys_cutoff_block(quantity=5, ttl_remaining=0, oldest_age=1)
    assert lo_ttl["sys_cutoff_order_urgency"] > hi_ttl["sys_cutoff_order_urgency"]
    assert lo_ttl["sys_cutoff_expected_lost_units_if_wait"] == 5.0   # ttl 0 -> whole order at risk
    assert hi_ttl["sys_cutoff_expected_lost_units_if_wait"] < 5.0
    assert hi_ttl["sys_cutoff_price_available"] == 0.0               # documented absence, not a price
    assert all(not math.isnan(v) for v in hi_ttl.values())


def test_sys_actor_recent_flows_sum_over_producers_and_competing_is_producer_specific() -> None:
    from tokenomics.domain.telemetry import EconomyTelemetry
    sheets = {1: ProcessSheet(product=1, require={"c0": 1}),
              2: ProcessSheet(product=2, require={"c0": 1})}
    a = _agent({"c0": 10}, {1, 2}, sheets)   # also makes product 2
    b = _agent({"c0": 10}, {1}, sheets)      # makes only the target
    a.id, b.id = "pa", "pb"
    tel = EconomyTelemetry()
    tel.record_completed("pa", 1, 4, routed_cycle=0, cycle=1)
    tel.record_completed("pb", 1, 6, routed_cycle=0, cycle=1)
    tel.end_cycle()                          # fold: alpha 0.3 -> ema 1.2 + 1.8 = 3.0
    open_qty = {1: 8, 2: 3}
    ba = ppf.sys_actor_block(a, product=1, open_qty=open_qty, telemetry=tel,
                             all_agents=[a, b], this_orderer_target_qty=5)
    assert abs(ba["sys_actor_recent_completed"] - 3.0) < 1e-9
    assert ba["sys_actor_other_orderer_queued_qty"] == 3.0    # 8 total - 5 this orderer
    assert ba["sys_actor_competing_qty_for_p"] == 3.0         # a also makes product 2 (qty 3)
    bb = ppf.sys_actor_block(b, product=1, open_qty=open_qty, telemetry=tel,
                             all_agents=[a, b], this_orderer_target_qty=5)
    assert bb["sys_actor_competing_qty_for_p"] == 0.0         # b makes only the target
