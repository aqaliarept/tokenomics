"""PRODUCT_AGNOSTIC O1: deterministic masked candidate allocator.

The portable orderer decode: given one runtime score per producer row (variable P,
no fixed K*P action head) plus an eligibility mask, split an order's quantity and
tokens across eligible producers, owning eligibility + conservation. Future learned
scorers feed this; decode_ordering refactors onto it.
"""

from __future__ import annotations

import numpy as np

from tokenomics.domain.ordering_agent import SubOrder
from tokenomics.env.allocator import allocate_suborders


def test_two_eligible_conserve_and_rank() -> None:
    # Tracer: both producers eligible, the higher score takes the larger share, and
    # quantity/tokens are conserved exactly.
    ids = ["p1", "p2"]
    subs = allocate_suborders(
        ids, scores=np.array([2.0, 0.0]), eligible_mask=np.array([True, True]),
        quantity=10, tokens=100,
    )
    by = {s.producer_id: s for s in subs}
    assert sum(s.quantity for s in subs) == 10
    assert sum(s.tokens for s in subs) == 100
    assert by["p1"].quantity > by["p2"].quantity


def test_ineligible_receive_nothing() -> None:
    # Eligibility mask gate: a high score on an ineligible row never routes quantity
    # or tokens to it.
    ids = ["p1", "p2", "p3"]
    subs = allocate_suborders(
        ids, scores=np.array([9.0, 1.0, 1.0]), eligible_mask=np.array([False, True, True]),
        quantity=8, tokens=80,
    )
    routed = {s.producer_id for s in subs}
    assert "p1" not in routed
    assert sum(s.quantity for s in subs) == 8 and sum(s.tokens for s in subs) == 80


def test_single_eligible_takes_all() -> None:
    # Single-eligible gate: the only valid route gets the whole order, no learned
    # distinction required.
    ids = ["p1", "p2"]
    subs = allocate_suborders(
        ids, scores=np.array([0.0, 5.0]), eligible_mask=np.array([True, False]),
        quantity=7, tokens=70,
    )
    assert subs == [SubOrder("p1", 7, 70)]


def test_zero_quantity_suborders_dropped() -> None:
    # A producer whose largest-remainder share rounds to 0 emits no sub-order.
    ids = ["p1", "p2"]
    subs = allocate_suborders(
        ids, scores=np.array([100.0, -100.0]), eligible_mask=np.array([True, True]),
        quantity=1, tokens=10,
    )
    assert subs == [SubOrder("p1", 1, 10)]


def test_row_permutation_equivariance() -> None:
    # Producer-row permutation moves each producer's allocation with its row; totals
    # and per-producer amounts are unchanged.
    ids = ["p1", "p2", "p3"]
    scores = np.array([2.0, 0.5, 1.0])
    mask = np.array([True, True, True])
    base = {s.producer_id: (s.quantity, s.tokens)
            for s in allocate_suborders(ids, scores, mask, 11, 110)}
    perm = [2, 0, 1]
    pids = [ids[i] for i in perm]
    permuted = {s.producer_id: (s.quantity, s.tokens)
                for s in allocate_suborders(pids, scores[perm], mask[perm], 11, 110)}
    assert base == permuted


def test_conservation_under_random_scores() -> None:
    # Random scores are good at finding decoder bugs (doc): conservation must hold for
    # any score tensor and any eligibility pattern with >=1 eligible row. Orders
    # always demand >=1 unit (tokens follow the realized integer quantities).
    rng = np.random.default_rng(0)
    ids = [f"p{i}" for i in range(6)]
    for _ in range(200):
        scores = rng.normal(size=6) * 10
        mask = rng.random(6) > 0.5
        if not mask.any():
            mask[rng.integers(6)] = True
        q, tok = int(rng.integers(1, 20)), int(rng.integers(0, 200))
        subs = allocate_suborders(ids, scores, mask, q, tok)
        assert sum(s.quantity for s in subs) == q
        assert sum(s.tokens for s in subs) == tok
        assert all(s.quantity > 0 for s in subs)
        assert all(mask[ids.index(s.producer_id)] for s in subs)


def test_no_eligible_producer_emits_nothing() -> None:
    # Degenerate gate: an order with zero eligible producers cannot be routed — the
    # allocator emits no sub-orders (and no NaN), rather than softmaxing an empty set.
    ids = ["p1", "p2"]
    subs = allocate_suborders(
        ids, scores=np.array([1.0, 2.0]), eligible_mask=np.array([False, False]),
        quantity=5, tokens=50,
    )
    assert subs == []
