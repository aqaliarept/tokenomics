"""Deterministic masked candidate allocator (PRODUCT_AGNOSTIC O1).

Portable orderer decode: one runtime score per producer **row** (variable P, no
fixed `K*P` action head) + an eligibility mask -> a conserved split of an order's
quantity and tokens across the *eligible* producers. The learned part only supplies
per-candidate scores; this decoder owns eligibility masking, the softmax, integer
rounding, token apportionment, and the conservation invariants — none of which a
model should have to rediscover (ADR-0025 conservation, ADR-0028 eligibility).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..domain.common import ProducerId
from ..domain.ordering_agent import SubOrder


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _allocate_int(proportions: np.ndarray, budget: int) -> list[int]:
    """Split `budget` into non-negative ints proportional to `proportions`, summing
    exactly to budget. Largest fractional remainders take the leftover (ADR-0025)."""
    if budget <= 0:
        return [0] * len(proportions)
    raw = proportions * budget
    floors = np.floor(raw).astype(int)
    remainder = budget - int(floors.sum())
    if remainder > 0:
        frac = raw - floors
        order = np.argsort(-frac)
        for i in range(remainder):
            floors[order[i % len(floors)]] += 1
    return floors.tolist()


def allocate_suborders(
    producer_ids: Sequence[ProducerId],
    scores: np.ndarray,
    eligible_mask: np.ndarray,
    quantity: int,
    tokens: int,
) -> list[SubOrder]:
    """Allocate one order across its eligible producers by candidate score.

    `producer_ids`, `scores`, `eligible_mask` are aligned, one entry per runtime
    producer row. Only masked-eligible rows receive quantity/tokens. Σ sub-quantities
    == quantity, Σ sub-tokens == tokens; zero-quantity sub-orders are dropped."""
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(eligible_mask, dtype=bool)
    elig_idx = np.flatnonzero(mask)
    if elig_idx.size == 0:
        return []  # no eligible producer: the order cannot be routed this cycle
    props = _softmax(scores[elig_idx])
    quantities = _allocate_int(props, quantity)
    q_props = np.asarray(quantities, dtype=np.float64) / quantity if quantity > 0 else props
    toks = _allocate_int(q_props, tokens)
    return [
        SubOrder(producer_ids[elig_idx[j]], q, t)
        for j, (q, t) in enumerate(zip(quantities, toks))
        if q > 0
    ]
