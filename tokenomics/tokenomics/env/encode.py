"""Domain state -> fixed-size normalized float32 observation vectors (ADR-0025).

All features are non-negative and clipped to [0, OBS_CLIP]. A NaN cut-off price
(UNDEFINED_PRICE, no production yet) maps to 0.0. Layouts match spaces.EnvSpec.
"""

from __future__ import annotations

import numpy as np

from ..domain.common import Product, Tokens, is_nan
from ..domain.order import ORDER_TTL, OrderInfo
from ..domain.orderer_view import OrdererEconomyView, ProductStreamView
from ..domain.producing_agent import ProducerInfo, ProducingAgentView
from .spaces import (
    OBS_CLIP,
    ORDERER_PP_FEATURES,
    ORDERER_PRODUCT_FEATURES,
    ORDERER_SLOT_SCALARS,
    STREAM_SLOT_SCALARS,
    EnvSpec,
)


def _price(spec: EnvSpec, p: float) -> float:
    if is_nan(p):
        return 0.0
    return p / spec.price_ref


def _producer_summary(spec: EnvSpec, producer_infos: dict[str, ProducerInfo]) -> list[float]:
    """Per producer: total current capacity + cut-off price. Producing obs only —
    the ordering obs gets prices alone (ADR-0029)."""
    out: list[float] = []
    for pid in spec.producer_ids:
        info = producer_infos[pid]
        out.append(sum(info.capacities.values()) / spec.cap_ref)
        out.append(_price(spec, info.cut_off_price))
    return out


def _producer_prices(spec: EnvSpec, producer_infos: dict[str, ProducerInfo]) -> list[float]:
    """Per producer: cut-off price only. The ordering side is blind to capacity
    (ADR-0029) — it competes on price, not on producers' capacity."""
    return [_price(spec, producer_infos[pid].cut_off_price) for pid in spec.producer_ids]


def _open_demand_features(
    spec: EnvSpec, open_demand: dict[Product, tuple[int, Tokens]]
) -> list[float]:
    """Aggregate open consumer demand per orderable product (#2). First P features
    = normalized unresolved quantity, next P = normalized backing tokens, indexed by
    spec.order_product_index. Consumer orders exist before producers act, so this
    lets a producer anticipate what to make instead of inferring it from prices."""
    qty = [0.0] * spec.num_order_products
    tok = [0.0] * spec.num_order_products
    for product, (q, t) in open_demand.items():
        i = spec.order_product_index[product]
        qty[i] = q / spec.inv_ref
        tok[i] = t / spec.token_ref
    return qty + tok


def encode_producing(
    spec: EnvSpec,
    view: ProducingAgentView,
    producer_infos: dict[str, ProducerInfo],
    open_demand: dict[Product, tuple[int, Tokens]],
    cycle_progress: float,
) -> np.ndarray:
    feats: list[float] = []
    for t in spec.capacity_types:
        feats.append(view.capacities.get(t, 0) / spec.cap_ref)
    for t in spec.capacity_types:
        feats.append(view.max_capacities.get(t, 0) / spec.cap_ref)
    feats.append(view.requested_capacity / spec.cap_ref)
    feats.append(1.0 if view.restoration_running else 0.0)
    feats.append(1.0 if view.upgrade_running else 0.0)
    feats.append(1.0 if view.inventory_expansion_running else 0.0)
    for g in spec.input_products:
        feats.append(view.inventory.get(g, 0) / spec.inv_ref)
    # in-flight replenishment per input (ADR-0041): lets a policy avoid re-ordering
    # stock already on the way, which is what blocks priming a deep pipeline.
    for g in spec.input_products:
        feats.append(view.outstanding_inbound.get(g, 0) / spec.inv_ref)
    feats.append(view.inventory_units / spec.inv_ref)
    feats.append(view.inventory_limit / spec.inv_ref)
    feats.append(max(0, view.headroom) / spec.inv_ref)
    feats.extend(_producer_summary(spec, producer_infos))
    feats.extend(_open_demand_features(spec, open_demand))
    feats.extend(_producer_mechanics(spec, view))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def _producer_mechanics(spec: EnvSpec, view: ProducingAgentView) -> list[float]:
    """The agent's own producer-mechanics (ADR-0044): the per-seed-varying action
    semantics it must adapt to. Without it the producing obs has the same shape
    across scale topologies but a different decision problem, so the policy
    memorizes one operating point. Per-type degradation units this cycle, then how
    much one restoration / upgrade buys (capacity recovered / max gained), then the
    investment economics (max-replenish, replenish cost, restoration / upgrade cost)
    that set what each toggle is worth — all per-seed-varying in the scale suite."""
    out: list[float] = []
    for t in spec.capacity_types:
        out.append(view.degradation.get(t, 0) / spec.cap_ref)
    out.append(view.restoration / spec.cap_ref)
    out.append(view.upgrade / spec.cap_ref)
    m = spec.producer_mechanics[view.id]
    out.append(spec.max_replenish[view.id] / spec.inv_ref)
    out.append(m.replenish_cost_per_unit / spec.token_ref)
    out.append(m.restoration_cost / spec.token_ref)
    out.append(m.upgrade_cost / spec.token_ref)
    out.append(m.inventory_expansion_increases / spec.inv_ref)
    out.append(m.inventory_expansion_cost / spec.token_ref)
    return out


def _entity_global_block(spec: EnvSpec, g) -> list[float]:
    """Producer-wide scalars (PRODUCER_GLOBAL_FEATURES). Cut-off price normalized like
    the flat obs; inventory by inv_ref; the three investment-running flags raw."""
    return [
        _price(spec, g.cut_off_price),
        g.inventory_units / spec.inv_ref,
        g.inventory_limit / spec.inv_ref,
        max(0, g.inventory_headroom) / spec.inv_ref,
        g.outstanding_inbound_units / spec.inv_ref,
        *g.active_investments,
    ]


def _entity_producer_mechanics(spec: EnvSpec, pid: str) -> list[float]:
    """Producer-global action economics the capacity rows don't carry per-type
    (ADR-0044): replenishment ceiling + per-unit cost, and inventory-expansion effect +
    cost. Mirrors the flat `_producer_mechanics` tail. Without these, two scale seeds
    with identical row shape but different replenish/expansion costs encode the SAME
    producer obs while `decode_producing`/funding interpret the action differently, so an
    entity producer can't learn the correct replenishment/expansion choices. Restoration
    and upgrade cost are omitted here — already per targeted type in the capacity rows."""
    m = spec.producer_mechanics[pid]
    return [
        spec.max_replenish[pid] / spec.inv_ref,
        m.replenish_cost_per_unit / spec.token_ref,
        m.inventory_expansion_increases / spec.inv_ref,
        m.inventory_expansion_cost / spec.token_ref,
    ]


def _entity_capacity_row(spec: EnvSpec, r) -> list[float]:
    """One capacity-type row (PRODUCER_CAP_ROW_FEATURES), led by its active mask."""
    return [
        1.0,  # active mask
        r.owned,
        r.current / spec.cap_ref,
        r.max / spec.cap_ref,
        r.headroom / spec.cap_ref,
        r.ratio,
        r.degradation / spec.cap_ref,
        r.reserved / spec.cap_ref,
        r.previous_requested_capacity / spec.cap_ref,
        r.free / spec.cap_ref,
        r.reserved_ratio,
        r.restore_effect / spec.cap_ref,
        r.restore_cost / spec.token_ref,
        r.upgrade_effect / spec.cap_ref,
        r.upgrade_cost / spec.token_ref,
        r.demand_pressure / spec.cap_ref,
    ]


def _entity_product_row(spec: EnvSpec, r) -> list[float]:
    """One declared-product row (PRODUCER_PRODUCT_ROW_FEATURES), led by its active mask.
    Recipe-shape ints stay raw (clipped to OBS_CLIP), like the orderer product block;
    makeable quantities normalize by inv_ref (input_makeable inf -> clipped)."""
    return [
        1.0,  # active mask
        r.eligible,
        r.open_demand_qty / spec.inv_ref,
        r.open_demand_tokens / spec.token_ref,
        float(r.product_depth),
        float(r.recipe_input_count),
        float(r.recipe_input_units),
        float(r.recipe_capacity_type_count),
        r.recent_routed_qty / spec.inv_ref,
        r.recent_completed_qty / spec.inv_ref,
        r.recent_rejected_qty / spec.inv_ref,
        r.recent_completion_latency / ORDER_TTL,
        r.active_reservation_qty / spec.inv_ref,
        r.capacity_makeable_qty / spec.inv_ref,
        r.input_makeable_qty / spec.inv_ref,
        r.complete_qty_now / spec.inv_ref,
        r.capacity_bottleneck_ratio,
        r.input_bottleneck_ratio,
        r.bottleneck_ratio,
    ]


def _entity_input_row(spec: EnvSpec, r) -> list[float]:
    """One input-product row (PRODUCER_INPUT_ROW_FEATURES), led by its active mask."""
    return [
        1.0,  # active mask
        r.inventory / spec.inv_ref,
        r.outstanding_inbound / spec.inv_ref,
        r.blocked_reservation_qty / spec.inv_ref,
        r.downstream_demand_pressure / spec.inv_ref,
    ]


def encode_producing_entity(spec: EnvSpec, view, cycle_progress: float) -> np.ndarray:
    """Entity-factored producer obs (ADR-0053, producer_obs_version 2). Emits
    [globals, capacity rows, product rows, input rows, relation planes, cycle_progress].

    It consumes the view's row-list order VERBATIM (does not re-sort by EnvSpec): a
    permuted row list yields correspondingly permuted row slots, and the relation
    planes — keyed by row position — permute with them. That is what makes the encoder
    order-equivariant. Rows are zero-padded (mask 0) to the spec's fixed max counts so
    the Box stays fixed width; relation planes are Q x T and Q x G, zero outside real
    rows. cycle_progress is appended here, last, as the single trailing scalar."""
    from .spaces import (
        PRODUCER_CAP_ROW_FEATURES,
        PRODUCER_INPUT_ROW_FEATURES,
        PRODUCER_PRODUCT_ROW_FEATURES,
    )

    feats: list[float] = _entity_global_block(spec, view.globals)
    feats.extend(_entity_producer_mechanics(spec, view.producer_id))

    cap_rows = [_entity_capacity_row(spec, r) for r in view.capacity_rows]
    _pad(feats, cap_rows, spec.num_types, PRODUCER_CAP_ROW_FEATURES)

    prod_rows = [_entity_product_row(spec, r) for r in view.product_rows]
    _pad(feats, prod_rows, spec.num_order_products, PRODUCER_PRODUCT_ROW_FEATURES)

    in_rows = [_entity_input_row(spec, r) for r in view.input_rows]
    _pad(feats, in_rows, spec.num_inputs, PRODUCER_INPUT_ROW_FEATURES)

    # recipe-edge relation planes, indexed by row position; raw counts (clipped).
    Q, T, G = spec.num_order_products, spec.num_types, spec.num_inputs
    cap_plane = [0.0] * (Q * T)
    for (xi, ti), c in view.relations.product_capacity_required.items():
        cap_plane[xi * T + ti] = float(c)
    feats.extend(cap_plane)
    in_plane = [0.0] * (Q * G)
    for (xi, gi), n in view.relations.product_input_required.items():
        in_plane[xi * G + gi] = float(n)
    feats.extend(in_plane)

    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def _pad(feats: list[float], rows: list[list[float]], max_rows: int, width: int) -> None:
    """Append real rows verbatim then zero-pad (mask 0) to max_rows x width."""
    for row in rows:
        feats.extend(row)
    for _ in range(max_rows - len(rows)):
        feats.extend([0.0] * width)


def encode_ordering(
    spec: EnvSpec,
    incoming: dict[str, OrderInfo],
    producer_infos: dict[str, ProducerInfo],
    cycle_progress: float,
) -> np.ndarray:
    feats: list[float] = []
    ordered = [incoming[oid] for oid in sorted(incoming.keys())]
    for k in range(spec.k):
        if k < len(ordered):
            oi = ordered[k]
            feats.append(1.0)  # active
            feats.append(oi.tokens / spec.token_ref)
            feats.append(oi.quantity / spec.inv_ref)
            # product one-hot (B9): make the slot's product observable so seeded
            # demand-product randomization is not hidden state for a trained policy.
            onehot = [0.0] * spec.num_order_products
            onehot[spec.order_product_index[oi.product]] = 1.0
            feats.extend(onehot)
        else:
            feats.extend([0.0] * (3 + spec.num_order_products))
    feats.extend(_producer_prices(spec, producer_infos))  # no capacity (ADR-0029)
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def _product_features(spec: EnvSpec, pf) -> list[float]:
    """Product features (ORDERER_PRODUCT_FEATURES): open demand + recent history +
    static recipe-graph shape. No product id/one-hot — relabel-invariant by design."""
    return [
        pf.open_demand_quantity / spec.inv_ref,
        pf.open_demand_tokens / spec.token_ref,
        pf.recent_average_cycle_demand / spec.inv_ref,
        pf.recent_rejected_quantity / spec.inv_ref,
        float(pf.product_depth),
        float(pf.recipe_input_count),
        float(pf.recipe_input_units),
        float(pf.recipe_capacity_type_count),
        pf.eligible_producer_count / spec.num_producers,
    ]


def _pp_features(spec: EnvSpec, pp) -> list[float]:
    """One (order/stream, producer) cell (ORDERER_PP_FEATURES): makeability /
    bottleneck / recent throughput / latency for the product against this producer."""
    return [
        pp.eligible,
        pp.accept_qty_now / spec.inv_ref,
        pp.complete_qty_now / spec.inv_ref,
        pp.makeable_qty_now / spec.inv_ref,
        pp.capacity_bottleneck_ratio,
        pp.input_bottleneck_ratio,
        pp.bottleneck_ratio,
        pp.slot_makeability,
        pp.recent_routed_qty / spec.inv_ref,
        pp.recent_completed_qty / spec.inv_ref,
        pp.recent_rejected_qty / spec.inv_ref,
        pp.recent_success_rate,
        pp.recent_throughput / spec.inv_ref,
        pp.recent_completion_latency / ORDER_TTL,
        pp.recent_failure_latency / ORDER_TTL,
    ]


def _enriched_slot_block(spec: EnvSpec, slot, view: OrdererEconomyView) -> list[float]:
    """One order slot: scalar header + product one-hot + product features + the
    producer-product grid. Widths mirror spaces.ORDERER_* (test D2 ties them)."""
    out: list[float] = []
    # scalar header (ORDERER_SLOT_SCALARS)
    out.append(1.0)  # active
    out.append(slot.available_tokens / spec.token_ref)
    out.append(slot.unresolved_quantity / spec.inv_ref)
    out.append(slot.tokens_per_unit / spec.token_ref)
    out.append(slot.order_age / ORDER_TTL)
    out.append(slot.ttl_remaining / ORDER_TTL)
    out.extend(slot.order_type)  # one_hot(consumer, replenishment, investment)
    out.append(slot.eligible_producer_count / spec.num_producers)
    # slot product one-hot (Q)
    onehot = [0.0] * spec.num_order_products
    onehot[spec.order_product_index[slot.product]] = 1.0
    out.extend(onehot)
    # product features (ORDERER_PRODUCT_FEATURES)
    out.extend(_product_features(spec, view.products[slot.product]))
    # producer-product grid (num_producers * ORDERER_PP_FEATURES)
    for pid in spec.producer_ids:
        out.extend(_pp_features(spec, view.producer_products[(slot.order_id, pid)]))
    return out


def _enriched_producer_block(spec: EnvSpec, ps) -> list[float]:
    """One producer's global state (ORDERER_PRODUCER_FEATURES): capacity / load /
    cut-off / in-flight / recent rates — the load the search teacher routes by."""
    return [
        ps.total_current_capacity / spec.cap_ref,
        ps.total_max_capacity / spec.cap_ref,
        ps.capacity_headroom / spec.cap_ref,
        ps.min_capacity_ratio,
        ps.mean_capacity_ratio,
        ps.requested_load / spec.cap_ref,
        ps.reserved_load / spec.cap_ref,
        ps.total_load / spec.cap_ref,
        _price(spec, ps.cut_off_price),
        ps.outstanding_inbound_units / spec.inv_ref,
        *ps.active_investments,
        ps.acceptance_rate,
        ps.completion_rate,
        ps.rejection_rate,
    ]


def encode_ordering_enriched(
    spec: EnvSpec, view: OrdererEconomyView, cycle_progress: float
) -> np.ndarray:
    """Enriched orderer obs (EXP-0006A, orderer_obs_version 2): normalize the domain
    OrdererEconomyView into a fixed-width vector. Producer/product/order state the
    capacity-blind v1 obs (ADR-0029) hid — added so the search teacher is
    distillable. The domain owns the facts; this only normalizes + clips."""
    P, Q = spec.num_producers, spec.num_order_products
    slot_width = (
        ORDERER_SLOT_SCALARS + Q + ORDERER_PRODUCT_FEATURES + P * ORDERER_PP_FEATURES
    )
    feats: list[float] = []
    for k in range(spec.k):
        if k < len(view.order_slots):
            feats.extend(_enriched_slot_block(spec, view.order_slots[k], view))
        else:
            feats.extend([0.0] * slot_width)
    for pid in spec.producer_ids:
        feats.extend(_enriched_producer_block(spec, view.producers[pid]))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def _stream_slot_block(spec: EnvSpec, slot) -> list[float]:
    """One backlog slot of a product stream (STREAM_SLOT_SCALARS). Same scalar header
    as the enriched slot but WITHOUT the product one-hot or the eligible-producer
    count — the product is fixed for the stream and eligibility lives in producer
    rows, so neither belongs per slot (ADR-0052)."""
    return [
        1.0,  # active
        slot.available_tokens / spec.token_ref,
        slot.unresolved_quantity / spec.inv_ref,
        slot.tokens_per_unit / spec.token_ref,
        slot.order_age / ORDER_TTL,
        slot.ttl_remaining / ORDER_TTL,
        *slot.order_type,  # one_hot(consumer, replenishment, investment)
    ]


def encode_ordering_stream(
    spec: EnvSpec, view: ProductStreamView, cycle_progress: float
) -> np.ndarray:
    """Product-stream orderer obs (EXP-0007A, orderer_obs_version 3, ADR-0052).

    The agent serves a single fixed product, so the vector carries the product
    features once, then K backlog slots (no product one-hot), then one row per
    producer (global producer block + producer-product block for the fixed product),
    then cycle progress. Width scales with producers and backlog, never with the
    product vocabulary. The domain owns the facts; this only normalizes + clips."""
    feats: list[float] = []
    feats.extend(_product_features(spec, view.product_stats))
    for k in range(spec.k):
        if k < len(view.backlog):
            feats.extend(_stream_slot_block(spec, view.backlog[k]))
        else:
            feats.extend([0.0] * STREAM_SLOT_SCALARS)
    for pid in spec.producer_ids:
        feats.extend(_enriched_producer_block(spec, view.producers[pid]))
        feats.extend(_pp_features(spec, view.producer_products[pid]))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def _global_mechanics(spec: EnvSpec, view: ProducingAgentView) -> list[float]:
    """Per-producer producer-mechanics rolled up for the centralized critic (plan
    item 1) — the critic counterpart of the per-agent obs block (ADR-0044). The
    obs gives the *acting* producer its own mechanics; the critic needs every
    producer's so two scale seeds with identical global-vector shape no longer
    collapse to the same value input despite different return dynamics. Degradation
    is summed over types (the global state already sums capacity per producer);
    investment amounts/economics mirror `_producer_mechanics`."""
    m = spec.producer_mechanics[view.id]
    return [
        sum(view.degradation.values()) / spec.cap_ref,
        view.restoration / spec.cap_ref,
        view.upgrade / spec.cap_ref,
        spec.max_replenish[view.id] / spec.inv_ref,
        m.replenish_cost_per_unit / spec.token_ref,
        m.restoration_cost / spec.token_ref,
        m.upgrade_cost / spec.token_ref,
        m.inventory_expansion_increases / spec.inv_ref,
        m.inventory_expansion_cost / spec.token_ref,
    ]


def encode_global(
    spec: EnvSpec,
    views: dict[str, ProducingAgentView],
    producer_infos: dict[str, ProducerInfo],
    open_demand: dict[Product, tuple[int, Tokens]],
    cycle_progress: float,
) -> np.ndarray:
    """Centralized-critic global state (ADR-0004): every producer's rolled-up
    state + cut-off price, then every producer's mechanics block (plan item 1),
    then open consumer demand + cycle progress. Producers are the system state."""
    feats: list[float] = []
    for pid in spec.producer_ids:
        v = views[pid]
        feats.append(sum(v.capacities.values()) / spec.cap_ref)
        feats.append(sum(v.max_capacities.values()) / spec.cap_ref)
        feats.append(v.requested_capacity / spec.cap_ref)
        feats.append(v.inventory_units / spec.inv_ref)
        feats.append(_price(spec, producer_infos[pid].cut_off_price))
    for pid in spec.producer_ids:
        feats.extend(_global_mechanics(spec, views[pid]))
    feats.extend(_open_demand_features(spec, open_demand))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)
