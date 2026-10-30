"""Deterministic masked candidate allocator (PRODUCT_AGNOSTIC O1) and the `choice-v1`
validity layer (`validate_choice`, ADR-0059).

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


# Tolerance on Σ shares (ADR-0059): a float32 share vector that sums to one within it is valid.
SHARE_SUM_TOL = 1e-6


class InvalidChoice(ValueError):
    """A `choice-v1` action that is not a valid choice (ADR-0059). Never repaired: the episode
    stops, and the gate's contract path (`eval/contract.py`) records a `job` row under `check`."""

    check = "choice_validity"


def validate_choice(
    producer_ids: Sequence[ProducerId],
    shares: np.ndarray,
    eligible_mask: np.ndarray,
    quantity: int,
    tokens: int,
) -> list[SubOrder]:
    """Decode one order's `choice-v1` shares (ADR-0059, design D4): check, never repair.

    Valid = finite, every share ≥ 0, no positive share on an ineligible row, and
    |Σ − 1| ≤ `SHARE_SUM_TOL`. An order with no eligible producer is valid only as an all-zero
    row and is not routed, as in `allocate_suborders`. Quantities are `_allocate_int` of the
    shares (normalised by their sum, so float32 rounding inside the tolerance cannot break
    conservation) × quantity; tokens follow the integer quantities. A whole choice (a one-hot
    row) gives one sub-order with the whole quantity and every token. Anything else raises
    `InvalidChoice`."""
    shares = np.asarray(shares, dtype=np.float64)
    mask = np.asarray(eligible_mask, dtype=bool)
    if shares.shape != mask.shape or shares.shape != (len(producer_ids),):
        raise InvalidChoice(f"share row shape {shares.shape} != producers ({len(producer_ids)},)")
    if not np.isfinite(shares).all():
        raise InvalidChoice(f"non-finite shares {shares.tolist()}")
    if (shares < 0).any():
        i = int(np.flatnonzero(shares < 0)[0])
        raise InvalidChoice(f"negative share {float(shares[i])!r} on producer {producer_ids[i]}")
    bad = np.flatnonzero((shares > 0) & ~mask)
    if bad.size:
        raise InvalidChoice(
            f"share {float(shares[bad[0]])!r} on ineligible producer {producer_ids[bad[0]]} "
            f"(eligible: {[producer_ids[i] for i in np.flatnonzero(mask)]})")
    elig_idx = np.flatnonzero(mask)
    total = float(shares.sum())
    if elig_idx.size == 0:
        if total != 0.0:  # unreachable: positive shares off the (empty) support raised above
            raise InvalidChoice(f"shares sum to {total!r} on an order with no eligible producer")
        return []
    if abs(total - 1.0) > SHARE_SUM_TOL:
        raise InvalidChoice(f"shares sum to {total!r}, not 1 within {SHARE_SUM_TOL}")
    props = shares[elig_idx] / total
    quantities = _allocate_int(props, quantity)
    if sum(quantities) != max(quantity, 0):  # internal invariant of the largest-remainder rule
        raise InvalidChoice(f"quantities {quantities} do not sum to the order's {quantity}")
    q_props = np.asarray(quantities, dtype=np.float64) / quantity if quantity > 0 else props
    toks = _allocate_int(q_props, tokens)
    return [
        SubOrder(producer_ids[elig_idx[j]], q, t)
        for j, (q, t) in enumerate(zip(quantities, toks))
        if q > 0
    ]
