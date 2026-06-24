"""Domain state -> fixed-size normalized float32 observation vectors (ADR-0025).

All features are non-negative and clipped to [0, OBS_CLIP]. A NaN cut-off price
(UNDEFINED_PRICE, no production yet) maps to 0.0. Layouts match spaces.EnvSpec.
"""

from __future__ import annotations

import numpy as np

from ..domain.common import Product, Tokens, is_nan
from ..domain.order import ORDER_TTL, OrderInfo
from ..domain.orderer_view import OrdererEconomyView
from ..domain.producing_agent import ProducerInfo, ProducingAgentView
from .spaces import (
    OBS_CLIP,
    ORDERER_PP_FEATURES,
    ORDERER_PRODUCT_FEATURES,
    ORDERER_SLOT_SCALARS,
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
    pf = view.products[slot.product]
    out.append(pf.open_demand_quantity / spec.inv_ref)
    out.append(pf.open_demand_tokens / spec.token_ref)
    out.append(pf.recent_average_cycle_demand / spec.inv_ref)
    out.append(pf.recent_rejected_quantity / spec.inv_ref)
    out.append(float(pf.product_depth))
    out.append(float(pf.recipe_input_count))
    out.append(float(pf.recipe_input_units))
    out.append(float(pf.recipe_capacity_type_count))
    out.append(pf.eligible_producer_count / spec.num_producers)
    # producer-product grid (num_producers * ORDERER_PP_FEATURES)
    for pid in spec.producer_ids:
        pp = view.producer_products[(slot.order_id, pid)]
        out.append(pp.eligible)
        out.append(pp.accept_qty_now / spec.inv_ref)
        out.append(pp.complete_qty_now / spec.inv_ref)
        out.append(pp.makeable_qty_now / spec.inv_ref)
        out.append(pp.capacity_bottleneck_ratio)
        out.append(pp.input_bottleneck_ratio)
        out.append(pp.bottleneck_ratio)
        out.append(pp.slot_makeability)
        out.append(pp.recent_routed_qty / spec.inv_ref)
        out.append(pp.recent_completed_qty / spec.inv_ref)
        out.append(pp.recent_rejected_qty / spec.inv_ref)
        out.append(pp.recent_success_rate)
        out.append(pp.recent_throughput / spec.inv_ref)
        out.append(pp.recent_completion_latency / ORDER_TTL)
        out.append(pp.recent_failure_latency / ORDER_TTL)
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
