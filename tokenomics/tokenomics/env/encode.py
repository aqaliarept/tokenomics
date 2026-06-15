"""Domain state -> fixed-size normalized float32 observation vectors (ADR-0025).

All features are non-negative and clipped to [0, OBS_CLIP]. A NaN cut-off price
(UNDEFINED_PRICE, no production yet) maps to 0.0. Layouts match spaces.EnvSpec.
"""

from __future__ import annotations

import numpy as np

from ..domain.common import is_nan
from ..domain.order import OrderInfo
from ..domain.producing_agent import ProducerInfo, ProducingAgentView
from .spaces import OBS_CLIP, EnvSpec


def _price(spec: EnvSpec, p: float) -> float:
    if is_nan(p):
        return 0.0
    return p / spec.price_ref


def _producer_summary(spec: EnvSpec, producer_infos: dict[str, ProducerInfo]) -> list[float]:
    """Per producer: total current capacity + cut-off price."""
    out: list[float] = []
    for pid in spec.producer_ids:
        info = producer_infos[pid]
        out.append(sum(info.capacities.values()) / spec.cap_ref)
        out.append(_price(spec, info.cut_off_price))
    return out


def encode_producing(
    spec: EnvSpec,
    view: ProducingAgentView,
    producer_infos: dict[str, ProducerInfo],
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
    feats.append(view.inventory_units / spec.inv_ref)
    feats.append(view.inventory_limit / spec.inv_ref)
    feats.append(max(0, view.headroom) / spec.inv_ref)
    feats.extend(_producer_summary(spec, producer_infos))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


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
            for t in spec.capacity_types:
                feats.append(oi.unit_capacity.get(t, 0) / spec.cap_ref)
        else:
            feats.extend([0.0] * (3 + spec.num_types))
    feats.extend(_producer_summary(spec, producer_infos))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def encode_global(
    spec: EnvSpec,
    views: dict[str, ProducingAgentView],
    producer_infos: dict[str, ProducerInfo],
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
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)
