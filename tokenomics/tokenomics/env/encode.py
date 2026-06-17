"""Domain state -> fixed-size normalized float32 observation vectors (ADR-0025).

All features are non-negative and clipped to [0, OBS_CLIP]. A NaN cut-off price
(UNDEFINED_PRICE, no production yet) maps to 0.0. Layouts match spaces.EnvSpec.
"""

from __future__ import annotations

import numpy as np

from ..domain.common import Product, Tokens, is_nan
from ..domain.order import OrderInfo
from ..domain.producing_agent import ProducerInfo, ProducingAgentView
from .spaces import OBS_CLIP, EnvSpec


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


def encode_global(
    spec: EnvSpec,
    views: dict[str, ProducingAgentView],
    producer_infos: dict[str, ProducerInfo],
    open_demand: dict[Product, tuple[int, Tokens]],
    cycle_progress: float,
) -> np.ndarray:
    """Centralized-critic global state (ADR-0004): every producer's rolled-up
    state + cut-off price + cycle progress. Producers are the system state."""
    feats: list[float] = []
    for pid in spec.producer_ids:
        v = views[pid]
        feats.append(sum(v.capacities.values()) / spec.cap_ref)
        feats.append(sum(v.max_capacities.values()) / spec.cap_ref)
        feats.append(v.requested_capacity / spec.cap_ref)
        feats.append(v.inventory_units / spec.inv_ref)
        feats.append(_price(spec, producer_infos[pid].cut_off_price))
    feats.extend(_open_demand_features(spec, open_demand))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)
