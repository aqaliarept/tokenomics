"""Reachable producer hard-case mining (ADR-0049 Phase 3).

The decode-aware clone passes d3_p00 and holds across trained topologies, but held-out
topologies vary (seed 3 was the lowest). Phase 3 mines REACHABLE producer states where
the learner diverges from `obs_stock_aware` and the rollout underperforms, tagged by the
EXP-0001 failure surface, to retrain on (Phase 4). These tests pin the pure tagging +
acceptance logic; the rollout miner is covered by a write/read smoke."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_classify_failures_tags_the_exp0001_surface():
    # The classifier maps a producer decision's diagnostics to the accepted failure
    # tags (EXP-0001 menu). A healthy middle-layer decision matches nothing; a starved
    # one matches the relevant tags.
    from tokenomics.train.producer_hard_cases import DecisionDiag, classify_failures

    healthy = DecisionDiag(
        layer=1, depth=3, cap_ratio=0.9, blocked=0, reserved_cap=0,
        min_input_inv=5, total_inbound=4, has_inputs=True,
        layer_cap_ratio=0.9, other_layer_cap_ratio=0.9, open_demand_units=2,
    )
    assert classify_failures(healthy) == set()

    starved = DecisionDiag(
        layer=1, depth=3, cap_ratio=0.1, blocked=20, reserved_cap=120,
        min_input_inv=0, total_inbound=0, has_inputs=True,
        layer_cap_ratio=0.1, other_layer_cap_ratio=0.9, open_demand_units=10,
    )
    tags = classify_failures(starved)
    assert "middle_capacity_low" in tags          # middle layer, low cap ratio
    assert "blocked_reservation_high" in tags      # many waiting reservations
    assert "missing_input_inventory" in tags       # a needed input at 0 stock
    assert "inbound_missing_or_stalled" in tags    # ... and nothing on the way
    assert "layer_capacity_imbalance" in tags      # this layer far below another
    assert "downstream_demand_unserved" in tags    # open demand while starved


def test_classify_failures_inbound_present_is_not_stalled():
    # Missing input inventory but replenishment IS in flight -> the input tag fires but
    # NOT the stalled tag (the producer is correctly ordering; just not landed yet).
    from tokenomics.train.producer_hard_cases import DecisionDiag, classify_failures

    diag = DecisionDiag(
        layer=1, depth=3, cap_ratio=0.9, blocked=0, reserved_cap=0,
        min_input_inv=0, total_inbound=8, has_inputs=True,
        layer_cap_ratio=0.9, other_layer_cap_ratio=0.9, open_demand_units=0,
    )
    tags = classify_failures(diag)
    assert "missing_input_inventory" in tags
    assert "inbound_missing_or_stalled" not in tags
