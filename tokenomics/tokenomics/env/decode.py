"""Raw policy actions -> domain commands (ADR-0025/0026).

Ordering action: per order, a raw logit per producer. Decode restricts to the
order's *eligible* producers, softmaxes, apportions the order's **quantity** by
the largest-remainder rule, then apportions tokens following the realized integer
quantities (one allocation, not two — ADR-0025). Zero-quantity sub-orders are
dropped. Σ sub-quantities = order quantity, Σ sub-tokens = order tokens.

Producing action: a Box of raw reals — G replenishment quantities then 3
investment toggles (restoration, upgrade, inventory expansion). Decode owns all
gating: joint replenishment over shared inventory headroom, irrelevant dims are
no-ops, and a toggle for an already-running / unconfigured investment is ignored.
"""

from __future__ import annotations

import numpy as np

from ..domain.ordering_agent import OrderingAgentCommand, OrderingAgentView, SubOrder
from ..domain.producing_agent import ProducingAgentCommand, ProducingAgentView
from .allocator import allocate_suborders
from .spaces import ACTION_LOGIT_BOUND, EnvSpec


def _allocate_int(proportions: np.ndarray, budget: int) -> list[int]:
    """Split `budget` into non-negative ints proportional to `proportions`,
    summing exactly to budget. Largest fractional remainders get the leftover."""
    if budget <= 0:
        return [0] * len(proportions)
    raw = proportions * budget
    floors = np.floor(raw).astype(int)
    remainder = budget - int(floors.sum())
    if remainder > 0:
        frac = raw - floors
        order = np.argsort(-frac)  # largest fractional part first
        for i in range(remainder):
            floors[order[i % len(floors)]] += 1
    return floors.tolist()


def decode_ordering(
    spec: EnvSpec, action: np.ndarray, view: OrderingAgentView
) -> OrderingAgentCommand:
    # k_stream == k for v1/v2/v3; v4 narrows the backlog bound to per-stream (O0).
    action = np.asarray(action, dtype=np.float64).reshape(spec.k_stream, spec.num_producers)
    ids = spec.producer_ids
    orders: dict[str, list[SubOrder]] = {}
    for k, oid in enumerate(sorted(view.incoming)):
        oi = view.incoming[oid]
        mask = np.array([pid in view.eligible[oid] for pid in ids])
        # The slot's per-producer logits are the candidate scores; orders beyond the
        # backlog bound have no slot, so score uniformly (-> even split) to stay
        # addressed. The masked allocator (O1) owns eligibility + conservation.
        scores = action[k] if k < spec.k_stream else np.zeros(len(ids))
        orders[oid] = allocate_suborders(ids, scores, mask, oi.quantity, oi.tokens)
    return OrderingAgentCommand(orders)


def decode_producing(
    spec: EnvSpec, action: np.ndarray, view: ProducingAgentView
) -> ProducingAgentCommand:
    action = np.asarray(action, dtype=np.float64)
    g = spec.num_inputs
    replen_raw = action[:g]
    toggles = action[g : g + 3]  # [restoration, upgrade, inventory_expansion]

    relevant = spec.producer_inputs[view.id]
    max_r = spec.max_replenish[view.id]
    headroom = max(0, view.headroom)

    # per-relevant-input desired quantity from the raw action (raw <= 0 -> 0).
    desired: list[int] = []
    for prod in relevant:
        raw = replen_raw[spec.input_index[prod]]
        d = 0 if raw <= 0 else int(round(min(raw, ACTION_LOGIT_BOUND) / ACTION_LOGIT_BOUND * max_r))
        desired.append(d)

    # joint over shared headroom: use as-is when the sum fits, else scale to
    # exactly H by the largest-remainder rule (ADR-0025).
    total = sum(desired)
    if total > headroom:
        props = np.asarray(desired, dtype=np.float64) / total
        desired = _allocate_int(props, headroom)
    replenishments = {prod: q for prod, q in zip(relevant, desired) if q > 0}

    # toggles: raw > 0 fires, gated on already-running / unconfigured (decode owns
    # all gating per ADR-0026 — the actor emits raw reals only).
    do_restoration = bool(toggles[0] > 0) and not view.restoration_running
    do_upgrade = bool(toggles[1] > 0) and not view.upgrade_running
    do_inventory_expansion = (
        bool(toggles[2] > 0) and view.has_inventory_expansion and not view.inventory_expansion_running
    )
    return ProducingAgentCommand(
        do_restoration=do_restoration,
        do_upgrade=do_upgrade,
        do_inventory_expansion=do_inventory_expansion,
        replenishments=replenishments,
    )
