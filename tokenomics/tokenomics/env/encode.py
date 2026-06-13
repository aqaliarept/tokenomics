"""Domain state -> fixed-size normalized float32 observation vectors.

All features are non-negative and clipped to [0, OBS_CLIP]. A NaN cut-off price
(UNDEFINED_PRICE, no production yet) maps to 0.0.
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
    out: list[float] = []
    for pid in spec.producer_ids:
        info = producer_infos[pid]
        out.append(info.capacity / spec.cap_ref)
        out.append(_price(spec, info.cut_off_price))
    return out


def encode_producing(
    spec: EnvSpec,
    view: ProducingAgentView,
    producer_infos: dict[str, ProducerInfo],
    cycle_progress: float,
) -> np.ndarray:
    feats: list[float] = [
        view.capacity / spec.cap_ref,
        view.max_capacity / spec.cap_ref,
        view.requested_capacity / spec.cap_ref,
        view.degradation / spec.cap_ref,
        1.0 if view.upgrade_running else 0.0,
        1.0 if view.restoration_running else 0.0,
    ]
    feats.extend(_producer_summary(spec, producer_infos))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def encode_global(
    spec: EnvSpec,
    views: dict[str, ProducingAgentView],
    producer_infos: dict[str, ProducerInfo],
    cycle_progress: float,
) -> np.ndarray:
    """Centralized-critic global state (ADR-0004): every producer's full state +
    cut-off price + cycle progress. In Phase 1 producers are the system state."""
    feats: list[float] = []
    for pid in spec.producer_ids:
        v = views[pid]
        feats.append(v.capacity / spec.cap_ref)
        feats.append(v.max_capacity / spec.cap_ref)
        feats.append(v.requested_capacity / spec.cap_ref)
        feats.append(v.degradation / spec.cap_ref)
        feats.append(1.0 if v.upgrade_running else 0.0)
        feats.append(1.0 if v.restoration_running else 0.0)
        feats.append(_price(spec, producer_infos[pid].cut_off_price))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)


def encode_ordering(
    spec: EnvSpec,
    incoming: dict[str, OrderInfo],
    producer_infos: dict[str, ProducerInfo],
    cycle_progress: float,
) -> np.ndarray:
    feats: list[float] = []
    # K order slots, filled in sorted order-id order; pad the rest with zeros.
    ordered = [incoming[oid] for oid in sorted(incoming.keys())]
    for k in range(spec.k):
        if k < len(ordered):
            oi = ordered[k]
            feats.append(1.0)  # active
            feats.append(oi.tokens / spec.token_ref)  # budget
            for ct in spec.capacity_types:
                feats.append(oi.required.get(ct, 0) / spec.cap_ref)
        else:
            feats.extend([0.0] * (2 + spec.num_types))
    feats.extend(_producer_summary(spec, producer_infos))
    feats.append(cycle_progress)
    return np.clip(np.asarray(feats, dtype=np.float32), 0.0, OBS_CLIP)
