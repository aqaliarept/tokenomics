"""EXP-0038B per-producer/product observation ladder: deployable feature-block builders.

EXP-0038/0038A proved the current 7-feature producer row aliases the marginal-value teacher and that
the deployable blocks the old dump already logged (order/queue context) shift the label *level* but
carry no *within-decision* contrast (they are constant across a decision's eligible producers). This
module builds the missing class — **per-producer, per-target-product** metrics that DO vary across
candidate producers inside one decision and can therefore re-rank them (HANDOVER Phase 1, P1-P4/P6).

Every builder is a pure read of a `ProducingAgent` snapshot + its process sheets (like `domain.metrics`
and the ordering baselines): white-box, torch-free, deployable at decision time (no teacher rollout,
no future/oracle facts). `value_router_errors` orchestrates them into the `producer_product_v1` dump;
`observation_ladder` scores each `pp_*` prefix as a cumulative ladder block.

Naming: each block owns one stable column prefix so the ladder can split blocks by prefix —
`pp_throughput_` (P1), `pp_opp_` (P2), `pp_dep_` (P3), `pp_base_` (P4), `pp_fulfill_` (P6). P5 history
reuses the existing `h_*` telemetry-EMA block; this env has no exact 5-cycle window (telemetry keeps
folded EMAs only), so P5 stays on the EMA block per the HANDOVER fallback.
"""

from __future__ import annotations

import numpy as np

from ..domain import metrics as dmetrics
from ..domain.config import ProcessSheet
from ..domain.order import ORDER_TTL, Order
from ..domain.producing_agent import ProducingAgent

# Cover sentinel for a primal product: no input dependency == unbounded cover. Explicit, so a primal
# producer (is_primal=1, cover=SENTINEL) never collapses into a starved composite (is_primal=0,
# cover=0) — the HANDOVER "primal sentinel, not accidental zeros" requirement.
_DEP_SENTINEL = 10.0

# Cover sentinel for an absent system-wide upstream/recipe constraint (e.g. a primal target has no
# upstream to cover): explicit high value so "no constraint" never collapses into "starved" (0).
_SYS_SENTINEL = 10.0


def _required(p: ProducingAgent, product: int) -> dict[str, int]:
    """The capacity types product's recipe requires (positive per-unit cost), read from the
    producer's own process sheet. Empty if the producer does not declare product."""
    if product not in p.sheets:
        return {}
    return {t: c for t, c in p.sheets[product].require.items() if c > 0}


def _stats(vals: list[float], prefix: str) -> dict[str, float]:
    """min/mean/max/std of a per-type unit vector (dispersion the binding-min feature hides).
    Empty vector -> all zeros (a producer with no required types cannot make the product)."""
    a = np.array(vals, dtype=np.float64) if vals else np.zeros(1)
    if not vals:
        return {f"{prefix}_min": 0.0, f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0,
                f"{prefix}_std": 0.0}
    return {f"{prefix}_min": float(a.min()), f"{prefix}_mean": float(a.mean()),
            f"{prefix}_max": float(a.max()), f"{prefix}_std": float(a.std())}


def pp_throughput_block(p: ProducingAgent, *, product: int, quantity: int) -> dict[str, float]:
    """P1 — dedicated throughput of `p` for the target `product`, per required capacity type.

    For each required type t with per-unit cost req[t], the units of `product` the producer could
    make if it dedicated that type: current cap[t]/req[t], free (cap-reserved)/req, reserved/req,
    max maxcap/req, headroom (maxcap-cap)/req. Logged as min/mean/max/std over the required types —
    the dispersion two producers tied on the binding min still differ on (HANDOVER: "do not log only
    a single min"; non-binding capacity affects downstream recovery). Plus the order's demand
    pressure against the binding throughput (qty over free/current/max) and the binding slack."""
    need = _required(p, product)
    reserved = p.reserved_capacity()
    cap = p.capacities
    mx = p.max_capacities

    current = [cap.get(t, 0) / c for t, c in need.items()]
    free = [(cap.get(t, 0) - reserved.get(t, 0)) / c for t, c in need.items()]
    resv = [reserved.get(t, 0) / c for t, c in need.items()]
    maxu = [mx.get(t, 0) / c for t, c in need.items()]
    head = [(mx.get(t, 0) - cap.get(t, 0)) / c for t, c in need.items()]

    free_min = min(free) if free else 0.0
    current_min = min(current) if current else 0.0
    max_min = min(maxu) if maxu else 0.0

    row: dict[str, float] = {"pp_throughput_req_type_count": float(len(need))}
    row.update(_stats(current, "pp_throughput_current"))
    row.update(_stats(free, "pp_throughput_free"))
    row.update(_stats(resv, "pp_throughput_reserved"))
    row.update(_stats(maxu, "pp_throughput_max"))
    row.update(_stats(head, "pp_throughput_headroom"))
    row["pp_throughput_qty_over_free"] = quantity / free_min if free_min > 0 else float(quantity)
    row["pp_throughput_qty_over_current"] = quantity / current_min if current_min > 0 else float(quantity)
    row["pp_throughput_qty_over_max"] = quantity / max_min if max_min > 0 else float(quantity)
    row["pp_throughput_binding_slack"] = free_min - quantity
    row["pp_throughput_binding_is_zero"] = 1.0 if free_min <= 0 else 0.0
    return row


def pp_dep_block(p: ProducingAgent, *, product: int, quantity: int) -> dict[str, float]:
    """P3 — dependency (input) cover for the target `product`, converted into units of the target.

    Raw input inventory is ambiguous: the same stock can serve several products, so a raw count says
    nothing about *this* order (HANDOVER: "do not promote raw inventory count"). For each input i of
    `product` needing need[i] per unit, the cover in units of the target is
    (inventory[i] + inbound[i]) / need[i]; the binding (min) cover is how many units current stock +
    inbound can assemble. `missing_units` is the input shortfall for the whole order. A **primal**
    product has no dependency: `is_primal` = 1 and cover reads an explicit high sentinel (never an
    accidental 0), so a primal producer is distinguishable from a starved composite."""
    inputs = p.sheets[product].inputs if product in p.sheets else {}
    if not inputs:  # primal: no dependency to cover
        return {"pp_dep_is_primal": 1.0, "pp_dep_input_count": 0.0,
                "pp_dep_cover_min": _DEP_SENTINEL, "pp_dep_cover_mean": _DEP_SENTINEL,
                "pp_dep_cover_max": _DEP_SENTINEL, "pp_dep_cover_std": 0.0,
                "pp_dep_inventory_cover_min": _DEP_SENTINEL, "pp_dep_inbound_cover_min": _DEP_SENTINEL,
                "pp_dep_bottleneck_cover": _DEP_SENTINEL, "pp_dep_missing_units": 0.0}

    inv = p.inventory
    inbound = p.outstanding_inbound
    cover = [(inv.get(i, 0) + inbound.get(i, 0)) / n for i, n in inputs.items()]
    inv_cover = [inv.get(i, 0) / n for i, n in inputs.items()]
    inb_cover = [inbound.get(i, 0) / n for i, n in inputs.items()]
    missing = sum(max(0, quantity * n - (inv.get(i, 0) + inbound.get(i, 0))) for i, n in inputs.items())

    row = {"pp_dep_is_primal": 0.0, "pp_dep_input_count": float(len(inputs))}
    row.update(_stats(cover, "pp_dep_cover"))
    row["pp_dep_inventory_cover_min"] = float(min(inv_cover))
    row["pp_dep_inbound_cover_min"] = float(min(inb_cover))
    row["pp_dep_bottleneck_cover"] = float(min(cover))
    row["pp_dep_missing_units"] = float(missing)
    return row


def _free_units(p: ProducingAgent, product: int) -> float:
    """Dedicated free throughput of `p` for `product`: min over required types of
    (capacity - reserved) / per-unit cost — how many units it could make now if it dedicated its
    free capacity. 0.0 if the producer does not declare the product / owns none of a required type."""
    need = _required(p, product)
    if not need:
        return 0.0
    reserved = p.reserved_capacity()
    return min((p.capacities.get(t, 0) - reserved.get(t, 0)) / c for t, c in need.items())


def pp_opp_block(
    p: ProducingAgent, *, product: int, quantity: int, open_qty: dict[int, int]
) -> dict[str, float]:
    """P2 — opportunity cost of spending `p`'s capacity on the target `product`, vs the other products
    it also declares that COMPETE for the same capacity types (HANDOVER: current capacity is not
    enough, opportunity cost matters).

    For every alternate product B in `produces` (B != target), its dedicated free throughput and how
    much its recipe overlaps the target's required capacity types (share of the target's types B also
    uses). Alternates are demand-weighted by decision-time queued quantity (`open_qty`). The
    opportunity cost is the best alternative throughput the capacity could serve, scaled by capacity
    overlap — a highly demanded alternate on a DIFFERENT capacity type does not compete and costs
    nothing; one on the SAME type does. Zero alternatives -> all zeros (no competition)."""
    target_types = set(_required(p, product))
    target_free = _free_units(p, product)
    alts = [b for b in p.produces if b != product and _required(p, b)]

    row: dict[str, float] = {"pp_opp_alt_product_count": float(len(alts))}
    if not alts or not target_types:
        for c in ("alt_best_free", "alt_best_current", "alt_best_max", "alt_mean_free",
                  "alt_demand_weighted_free", "same_cap_overlap_max", "same_cap_overlap_mean",
                  "capacity_opportunity_cost", "scarcity_weighted_cost"):
            row[f"pp_opp_{c}"] = 0.0
        row["pp_opp_target_vs_best_margin"] = target_free
        row["pp_opp_target_vs_demand_weighted_margin"] = target_free
        return row

    free = {b: _free_units(p, b) for b in alts}
    current = {b: (min(p.capacities.get(t, 0) / c for t, c in _required(p, b).items())) for b in alts}
    maxu = {b: (min(p.max_capacities.get(t, 0) / c for t, c in _required(p, b).items())) for b in alts}
    overlap = {b: len(target_types & set(_required(p, b))) / len(target_types) for b in alts}
    weight = {b: float(open_qty.get(b, 0)) for b in alts}
    wsum = sum(weight.values())

    best_b = max(alts, key=lambda b: free[b])
    dw_free = (sum(weight[b] * free[b] for b in alts) / wsum) if wsum > 0 else float(np.mean(list(free.values())))
    ov_max = max(overlap.values())
    ov_mean = float(np.mean(list(overlap.values())))

    row["pp_opp_alt_best_free"] = float(free[best_b])
    row["pp_opp_alt_best_current"] = float(max(current.values()))
    row["pp_opp_alt_best_max"] = float(max(maxu.values()))
    row["pp_opp_alt_mean_free"] = float(np.mean(list(free.values())))
    row["pp_opp_alt_demand_weighted_free"] = float(dw_free)
    row["pp_opp_same_cap_overlap_max"] = float(ov_max)
    row["pp_opp_same_cap_overlap_mean"] = ov_mean
    # opportunity cost = best alternative throughput actually competing for the target's capacity
    row["pp_opp_capacity_opportunity_cost"] = float(max(free[b] * overlap[b] for b in alts))
    row["pp_opp_scarcity_weighted_cost"] = float(dw_free * ov_mean)
    row["pp_opp_target_vs_best_margin"] = float(target_free - free[best_b])
    row["pp_opp_target_vs_demand_weighted_margin"] = float(target_free - dw_free)
    return row


def pp_base_block(logits_by_pid: dict[str, float]) -> dict[str, dict[str, float]]:
    """P4 — base-routing context, one sub-row per eligible producer, from the base policy's
    (blend_0.5) per-producer logits for this decision.

    Purpose (HANDOVER): tell "the teacher disagrees because the base already nearly chose this
    producer" apart from "the teacher wants a large rerank against the base". So per producer: its
    base logit, softmax weight (over the eligible set), rank (# producers strictly better), margin to
    the best logit (<= 0), and margin to the next strictly-higher logit (0 for the top). This env has
    no cut-off / market price the ordering side sees at decision time, so those HANDOVER price columns
    are omitted (documented absence) — the base logit is the routing proxy."""
    pids = list(logits_by_pid)
    vals = np.array([logits_by_pid[p] for p in pids], dtype=np.float64)
    ex = np.exp(vals - vals.max())
    soft = ex / ex.sum()
    best = float(vals.max())

    out: dict[str, dict[str, float]] = {}
    for i, pid in enumerate(pids):
        v = float(vals[i])
        higher = vals[vals > v]
        rank = int((vals > v).sum())
        margin_to_next = float(v - higher.min()) if higher.size else 0.0
        out[pid] = {
            "pp_base_logit": v,
            "pp_base_softmax_weight": float(soft[i]),
            "pp_base_rank": float(rank),
            "pp_base_margin_to_best": v - best,
            "pp_base_margin_to_next": margin_to_next,
        }
    return out


def pp_fulfill_block(
    p: ProducingAgent, *, product: int, quantity: int, ttl_remaining: int, telemetry
) -> dict[str, float]:
    """P6 — a producer-specific "can this order finish before its TTL?" estimate, folding the P1
    capacity throughput and P3 input cover into a causal, deployable speed read (HANDOVER).

    Capacity-limited cycles to make the order = quantity / free throughput. Inputs add at least one
    cycle if current stock + inbound cannot cover the order (no upstream-speed model exists at the
    producer level, so this is a coarse >=1-cycle proxy, not a rollout). `est_cycles_to_complete` is
    the binding of the two; `feasible_before_ttl` compares it to the order's remaining TTL. History
    (recent completion rate, cycles-at-recent-rate) reads the telemetry EMA, which is folded per
    cycle — strictly past-cycle, so it never leaks the current cycle (deployable at decision time)."""
    free_min = _free_units(p, product)
    dep = pp_dep_block(p, product=product, quantity=quantity)
    missing = dep["pp_dep_missing_units"]

    cyc_produce = quantity / free_min if free_min > 0 else float(quantity)
    cyc_inputs = 0.0 if missing <= 0 else 1.0
    est_complete = max(cyc_produce, cyc_inputs)

    completed_ema = float(telemetry.completed_ema(p.id, product))
    return {
        "pp_fulfill_qty_over_hist_rate": quantity / completed_ema if completed_ema > 0 else float(quantity),
        "pp_fulfill_est_cycles_to_produce": float(cyc_produce),
        "pp_fulfill_est_cycles_to_inputs_ready": float(cyc_inputs),
        "pp_fulfill_est_cycles_to_complete": float(est_complete),
        "pp_fulfill_ttl_minus_est": float(ttl_remaining - est_complete),
        "pp_fulfill_feasible_before_ttl": 1.0 if est_complete <= ttl_remaining else 0.0,
        "pp_fulfill_recent_completion_rate": float(telemetry.success_rate(p.id, product)),
    }


# =====================================================================================
# EXP-0038B Phase 2 — system-wide (`sys_*`) blocks.
#
# EXP-0038B Phase 1 closed: the per-producer/product blocks (pp_*) do not explain the teacher's
# within-decision producer choice. Phase 2 tests the last deployable class — SYSTEM-WIDE facts
# (economy demand / global capacity scarcity / recipe-graph pressure / order urgency / other-actor
# presence) the teacher sees through its 8-cycle rollout but the current row throws away. CRUCIAL
# (HANDOVER): most `sys_*` fields are DECISION-CONSTANT — they take one value across a decision's
# eligible producers, so they cannot re-rank producers ADDITIVELY. The ladder scores them as
# interaction *context* for the producer-varying columns (observation_ladder._split_cols); the few
# producer-varying `sys_*` fields (`*_competing_for_p_capacity_qty`) re-rank through those same
# interactions. Every builder stays a pure read of live domain snapshots (torch-free, deployable at
# decision time — folded telemetry only, no teacher rollout / future facts).
# =====================================================================================


def system_open_qty(all_orders: dict[str, Order]) -> dict[int, int]:
    """Total unresolved quantity per product over every live order — the economy's queued demand at
    decision time. Includes consumer, replenishment and investment orders, since all of them pull on
    producer capacity (the competition `sys_*` measures). Zero-remainder orders contribute nothing."""
    out: dict[int, int] = {}
    for o in all_orders.values():
        q = o.unresolved_quantity()
        if q > 0:
            out[o.product] = out.get(o.product, 0) + q
    return out


def transitive_inputs(product: int, sheets: dict[int, ProcessSheet]) -> set[int]:
    """The set of all products reachable as (transitive) inputs of `product`'s recipe. Cycle-safe
    (the config model allows circular recipes broken by inventory): a back-edge to a product already
    on the current path is cut, matching `metrics.product_depth`."""
    seen: set[int] = set()

    def walk(x: int) -> None:
        for g in sheets[x].inputs:
            if g not in seen and g in sheets:
                seen.add(g)
                walk(g)

    walk(product)
    return seen


def _global_free_units(product: int, agents: list[ProducingAgent]) -> float:
    """System-wide dedicated FREE throughput for `product`: sum over every agent declaring it of its
    dedicated free throughput (`_free_units`). How many units the whole economy could make now."""
    return float(sum(_free_units(a, product) for a in agents if product in a.produces))


def _shares_capacity(p: ProducingAgent, target_types: set[str], other: int) -> bool:
    """Does product `other` (in p's sheets) require any capacity type the target also requires?
    Capacity SHARING, not mere co-production, is what makes `other` a real competitor for the
    target's capacity (mirrors pp_opp's overlap test)."""
    return bool(target_types & set(_required(p, other)))


def sys_demand_block(
    p: ProducingAgent, *, product: int, sheets: dict[int, ProcessSheet],
    open_qty: dict[int, int], target_orders: list[Order], this_orderer_target_qty: int,
) -> dict[str, float]:
    """S1 — target and competing demand across the whole economy. Decision-constant target-demand
    facts (queued qty / order count / oldest age / min TTL for the target product; capacity-competing
    demand from OTHER products) plus one PRODUCER-VARYING interaction-ready field:
    `sys_demand_competing_for_p_capacity_qty` — the demand for the OTHER products *this* producer
    declares that compete for the target's capacity types. That field varies across a decision's
    producers, so (unlike the constant fields) it can help re-rank once interacted."""
    target_types = {t for x in (product,) for t in sheets[x].require}
    ages = [o.cycle_counter for o in target_orders]
    ttls = [max(0, ORDER_TTL - o.cycle_counter) for o in target_orders]
    queued = float(open_qty.get(product, 0))
    competing = float(sum(q for x, q in open_qty.items()
                          if x != product and (target_types & set(sheets[x].require))))
    # producer-specific: other products p makes that compete for the target's capacity types
    competing_for_p = float(sum(open_qty.get(b, 0) for b in p.produces
                                if b != product and _shares_capacity(p, target_types, b)))
    return {
        "sys_demand_target_queued_qty": queued,
        "sys_demand_target_order_count": float(len(target_orders)),
        "sys_demand_target_oldest_age": float(max(ages)) if ages else 0.0,
        "sys_demand_target_min_ttl": float(min(ttls)) if ttls else float(ORDER_TTL),
        "sys_demand_same_product_other_orderers": max(0.0, queued - float(this_orderer_target_qty)),
        "sys_demand_competing_products_qty": competing,
        "sys_demand_competing_for_p_capacity_qty": competing_for_p,
    }


def sys_capacity_block(
    eligible_agents: list[ProducingAgent], *, product: int, sheets: dict[int, ProcessSheet],
    all_agents: list[ProducingAgent], open_qty: dict[int, int],
) -> dict[str, float]:
    """S2 — global capacity scarcity for the target product (decision-constant). Sum over the
    eligible producers of dedicated throughput (current/free/max/reserved/headroom units) — the total
    the economy can throw at the product — plus the free-throughput concentration (HHI: 1.0 = one
    producer holds all free capacity, → the routing choice is nearly forced). Per required capacity
    type, global scarcity = open-demand capacity load on the type / global supply of the type,
    summarised min/mean over the target's required types (a scarce shared type makes the target hard
    for everyone; the interaction with a producer's own slack is what should re-rank)."""
    reserved = [a.reserved_capacity() for a in eligible_agents]
    need = {t: c for t, c in sheets[product].require.items() if c > 0}

    def ded(a: ProducingAgent, kind: str, rc: dict[str, int]) -> float:
        if not need:
            return 0.0
        if kind == "current":
            return min(a.capacities.get(t, 0) / c for t, c in need.items())
        if kind == "free":
            return min((a.capacities.get(t, 0) - rc.get(t, 0)) / c for t, c in need.items())
        if kind == "max":
            return min(a.max_capacities.get(t, 0) / c for t, c in need.items())
        if kind == "reserved":
            return min(rc.get(t, 0) / c for t, c in need.items())
        return min((a.max_capacities.get(t, 0) - a.capacities.get(t, 0)) / c for t, c in need.items())

    free = [ded(a, "free", rc) for a, rc in zip(eligible_agents, reserved)]
    total_free = float(sum(free))
    hhi = float(sum((f / total_free) ** 2 for f in free)) if total_free > 0 else 0.0

    scar: list[float] = []
    for t in need:
        demand_t = sum(open_qty.get(x, 0) * sheets[x].require.get(t, 0) for x in open_qty)
        supply_t = sum(a.capacities.get(t, 0) for a in all_agents)
        scar.append(demand_t / max(1.0, supply_t))

    return {
        "sys_capacity_target_total_free_units": total_free,
        "sys_capacity_target_total_current_units": float(sum(ded(a, "current", rc) for a, rc in zip(eligible_agents, reserved))),
        "sys_capacity_target_total_max_units": float(sum(ded(a, "max", rc) for a, rc in zip(eligible_agents, reserved))),
        "sys_capacity_target_total_reserved_units": float(sum(ded(a, "reserved", rc) for a, rc in zip(eligible_agents, reserved))),
        "sys_capacity_target_total_headroom_units": float(sum(ded(a, "head", rc) for a, rc in zip(eligible_agents, reserved))),
        "sys_capacity_target_eligible_producer_count": float(len(eligible_agents)),
        "sys_capacity_target_free_hhi": hhi,
        "sys_capacity_required_type_scarcity_min": float(min(scar)) if scar else 0.0,
        "sys_capacity_required_type_scarcity_mean": float(np.mean(scar)) if scar else 0.0,
    }


def sys_recipe_block(
    *, product: int, sheets: dict[int, ProcessSheet], open_qty: dict[int, int],
    all_agents: list[ProducingAgent],
) -> dict[str, float]:
    """S3 — recipe-graph position and up/downstream bottleneck pressure (decision-constant, mostly
    static). Static shape: product depth, direct/transitive input counts, required capacity-type
    count. Demand-coupled: `downstream_demand_unlocked` = how much OTHER products' open demand pulls
    on the target *as an input* (a deep target that many downstream orders need is worth routing even
    when its own order is small). `upstream_capacity_cover_min` = the binding ratio of global free
    throughput of the target's inputs to the demand on them — a starved upstream caps the target for
    every producer (primal target → high sentinel, no upstream to cover)."""
    inputs = sheets[product].inputs
    downstream = float(sum(open_qty.get(b, 0) * sheets[b].inputs.get(product, 0) for b in open_qty))

    if not inputs:
        cover_min = _SYS_SENTINEL
    else:
        covers = []
        for g, n in inputs.items():
            supply = _global_free_units(g, all_agents)
            demand = open_qty.get(g, 0) + n  # standing demand for g plus this target's per-unit need
            covers.append(supply / max(1.0, demand))
        cover_min = float(min(covers))

    return {
        "sys_recipe_product_depth": float(dmetrics.product_depth(product, sheets)),
        "sys_recipe_input_count": float(dmetrics.recipe_input_count(sheets[product])),
        "sys_recipe_capacity_type_count": float(dmetrics.recipe_capacity_type_count(sheets[product])),
        "sys_recipe_transitive_input_count": float(len(transitive_inputs(product, sheets))),
        "sys_recipe_downstream_demand_unlocked": downstream,
        "sys_recipe_upstream_capacity_cover_min": cover_min,
    }


def sys_cutoff_block(*, quantity: int, ttl_remaining: int, oldest_age: int) -> dict[str, float]:
    """S4 — order urgency / cut-off context (decision-constant). This env exposes NO market/cut-off
    price the ordering side sees for a NEW order at decision time (the producer cut-off price is an
    internal post-auction scalar and defaults to NaN before any auction — HANDOVER documented
    absence). So S4 logs deployable URGENCY proxies only, never a faked price: urgency rises as TTL
    falls, a TTL-pressure level, the oldest same-product order's age, and a proxy for units at risk
    if the order waits a cycle. `sys_cutoff_price_available` = 0 flags the absence explicitly."""
    return {
        "sys_cutoff_order_urgency": 1.0 / (ttl_remaining + 1.0),
        "sys_cutoff_ttl_pressure_bucket": float(min(ttl_remaining, ORDER_TTL)),
        "sys_cutoff_oldest_age": float(oldest_age),
        "sys_cutoff_expected_lost_units_if_wait": float(quantity) if ttl_remaining <= 0
        else float(quantity) / (ttl_remaining + 1.0),
        "sys_cutoff_price_available": 0.0,
    }


def sys_actor_block(
    p: ProducingAgent, *, product: int, open_qty: dict[int, int], telemetry,
    all_agents: list[ProducingAgent], this_orderer_target_qty: int,
) -> dict[str, float]:
    """S5 — other-actor presence and recent system flow (decision-constant + one producer-varying).
    Constant: the target demand held by OTHER orderers (system total minus this orderer's), and the
    economy-wide recent routed/completed/rejected flow for the target product (folded EMAs summed
    over producers — how busy the target's supply already is). PRODUCER-VARYING:
    `sys_actor_competing_qty_for_p` — the total OTHER-product demand aimed at *this* producer (every
    other product it declares), the interaction-ready 'this producer is already contended' signal."""
    queued = float(open_qty.get(product, 0))
    routed = float(sum(telemetry.routed_ema(a.id, product) for a in all_agents))
    completed = float(sum(telemetry.completed_ema(a.id, product) for a in all_agents))
    rejected = float(sum(telemetry.rejected_or_failed_ema(a.id, product) for a in all_agents))
    competing_for_p = float(sum(open_qty.get(b, 0) for b in p.produces if b != product))
    return {
        "sys_actor_other_orderer_queued_qty": max(0.0, queued - float(this_orderer_target_qty)),
        "sys_actor_recent_routed": routed,
        "sys_actor_recent_completed": completed,
        "sys_actor_recent_rejected": rejected,
        "sys_actor_competing_qty_for_p": competing_for_p,
    }
