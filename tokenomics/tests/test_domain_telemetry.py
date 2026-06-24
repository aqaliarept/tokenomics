"""EXP-0006A: persistent per-cycle EMA telemetry (domain-owned economy history).

The enriched orderer observation needs recent (not lifetime) flow statistics so
routing can react to the current bottleneck (ALPHAZERO_EXPERT_ITERATION_APPROACH.md,
"Use recent EMA/windowed statistics"). The domain owns these so they are testable
outside RL. EMA recurrence: EMA_next = alpha*current + (1-alpha)*EMA_prev, folded
once per cycle; a cycle with no event decays the EMA toward zero.
"""

from __future__ import annotations

import pytest

from tokenomics.domain.telemetry import EconomyTelemetry


def test_routed_qty_ema_follows_recurrence_and_decays():
    t = EconomyTelemetry(alpha=0.5)
    assert t.routed_ema("p", "x") == 0.0  # nothing seen yet

    t.record_routed("p", "x", 4, cycle=1)
    t.end_cycle()
    assert t.routed_ema("p", "x") == 2.0  # 0.5*4 + 0.5*0

    t.record_routed("p", "x", 8, cycle=2)
    t.end_cycle()
    assert t.routed_ema("p", "x") == 5.0  # 0.5*8 + 0.5*2

    t.end_cycle()  # no routing this cycle -> decays toward 0
    assert t.routed_ema("p", "x") == 2.5  # 0.5*0 + 0.5*5


def test_completed_rejected_failed_emas_per_producer_product():
    t = EconomyTelemetry(alpha=0.5)
    t.record_completed("p", "x", 6, routed_cycle=1, cycle=1)
    t.record_rejected("p", "x", 2, routed_cycle=1, cycle=1)
    t.record_failed("p", "x", 1, routed_cycle=1, cycle=1)
    t.end_cycle()
    assert t.completed_ema("p", "x") == 3.0  # 0.5*6
    assert t.rejected_ema("p", "x") == 1.0  # 0.5*2
    assert t.failed_ema("p", "x") == 0.5  # 0.5*1


def test_derived_rates_per_product_and_aggregated_per_producer():
    """success_rate/throughput are per-(p,x); producer accept/complete/reject rates
    aggregate the per-product EMAs over the producer (EMA is linear, so summing
    per-product EMAs equals the producer total)."""
    t = EconomyTelemetry(alpha=0.5)
    # product x
    t.record_routed("p", "x", 10, cycle=1)
    t.record_accepted("p", "x", 8, routed_cycle=1, cycle=1)
    t.record_completed("p", "x", 6, routed_cycle=1, cycle=1)
    t.record_rejected("p", "x", 2, routed_cycle=1, cycle=1)
    t.record_failed("p", "x", 1, routed_cycle=1, cycle=1)
    # product y
    t.record_routed("p", "y", 4, cycle=1)
    t.record_accepted("p", "y", 4, routed_cycle=1, cycle=1)
    t.record_completed("p", "y", 4, routed_cycle=1, cycle=1)
    t.end_cycle()

    # per-(p,x): EMAs are completed=3, rejected=1, failed=0.5
    assert t.success_rate("p", "x") == pytest.approx(3.0 / 4.5)
    assert t.throughput("p", "x") == 3.0
    # producer totals: routed=5+2=7, accepted=4+2=6, completed=3+2=5, rej+fail=1.5
    assert t.producer_acceptance_rate("p") == pytest.approx(6.0 / 7.0)
    assert t.producer_completion_rate("p") == pytest.approx(5.0 / 7.0)
    assert t.producer_rejection_rate("p") == pytest.approx(1.5 / 7.0)


def test_completion_latency_is_quantity_weighted_ema_without_idle_decay():
    """Latency = cycle - routed_cycle, averaged within a cycle weighted by quantity,
    then EMA-folded. A cycle with no completion leaves the latency EMA unchanged
    (decaying it toward zero would falsely imply faster delivery)."""
    t = EconomyTelemetry(alpha=0.5)
    # cycle 3: two completions, latencies 2 (qty 3) and 1 (qty 1) -> weighted 1.75
    t.record_completed("p", "x", 3, routed_cycle=1, cycle=3)
    t.record_completed("p", "x", 1, routed_cycle=2, cycle=3)
    t.end_cycle()
    assert t.completion_latency("p", "x") == pytest.approx(0.875)  # 0.5*1.75

    t.end_cycle()  # no completions -> latency EMA held, not decayed
    assert t.completion_latency("p", "x") == pytest.approx(0.875)

    # next completion: latency 4 (cycle 6, routed 2) folds in
    t.record_completed("p", "x", 2, routed_cycle=2, cycle=6)
    t.end_cycle()
    assert t.completion_latency("p", "x") == pytest.approx(0.5 * 4 + 0.5 * 0.875)


def test_failure_latency_tracks_failed_events():
    t = EconomyTelemetry(alpha=0.5)
    t.record_failed("p", "x", 2, routed_cycle=1, cycle=3)  # latency 2
    t.end_cycle()
    assert t.failure_latency("p", "x") == pytest.approx(1.0)  # 0.5*2


def test_product_demand_and_rejected_demand_emas():
    """Per-product consumer demand history: new demand placed this cycle and demand
    rejected this cycle, each as an EMA (true product-level demand, not routed load)."""
    t = EconomyTelemetry(alpha=0.5)
    t.record_demand("x", 10)
    t.record_consumer_rejected("x", 3)
    t.end_cycle()
    assert t.demand_ema("x") == 5.0  # 0.5*10
    assert t.rejected_demand_ema("x") == 1.5  # 0.5*3
