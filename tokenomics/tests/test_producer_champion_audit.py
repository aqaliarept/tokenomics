"""EXP-0011A: deterministic producer champion audit.

Before replacing the v4 producer in the champion stack, compare the deterministic
entity planner against the current stochastic product-stream champion orderer on the
same held-out topology/eval seeds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")


_STREAM_CHAMPION = Path(__file__).resolve().parents[1] / "m2_inputs" / "m2_stream_stochastic.pt"
requires_stream_champion = pytest.mark.skipif(
    not _STREAM_CHAMPION.exists(), reason="product-stream champion checkpoint absent"
)


@requires_stream_champion
def test_producer_champion_audit_runs_one_tiny_cell():
    from tokenomics.eval.producer_champion_audit import audit_entity_planner_vs_champion

    rows = audit_entity_planner_vs_champion(
        str(_STREAM_CHAMPION),
        topology_seeds=[6],
        eval_seeds=1,
        horizon=5,
        include_envelope=False,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.topology_seed == 6
    assert row.champion_completed_units >= 0.0
    assert row.entity_completed_units >= 0.0
    assert row.champion_dead_rate == 0.0
    assert row.entity_dead_rate == 0.0


def test_producer_champion_summary_requires_no_death_and_no_mean_regression():
    from tokenomics.eval.producer_champion_audit import (
        ProducerChampionAuditRow,
        summarize_producer_champion_audit,
    )

    good = [
        ProducerChampionAuditRow(6, 40, 100, 10.0, 1.0, 11.0, 1.0, 0.0, 0.0),
        ProducerChampionAuditRow(7, 40, 100, 20.0, 1.0, 20.0, 1.0, 0.0, 0.0),
    ]
    assert summarize_producer_champion_audit(good).promotion_recommended

    regressed = [
        ProducerChampionAuditRow(6, 40, 100, 10.0, 1.0, 9.9, 1.0, 0.0, 0.0),
    ]
    assert not summarize_producer_champion_audit(regressed).promotion_recommended

    dead = [
        ProducerChampionAuditRow(6, 40, 100, 10.0, 1.0, 11.0, 1.0, 0.0, 0.1),
    ]
    assert not summarize_producer_champion_audit(dead).promotion_recommended


@requires_stream_champion
def test_producer_champion_audit_reports_size_transfer_compatibility():
    from tokenomics.eval.producer_champion_audit import audit_producer_size_transfer

    rows = audit_producer_size_transfer(
        str(_STREAM_CHAMPION),
        sizes=[4, 8],
        topology_seed=0,
        horizon=3,
    )

    assert rows[0].products_per_layer == 4
    assert rows[0].checkpoint_compatible
    assert rows[1].products_per_layer == 8
    assert not rows[1].checkpoint_compatible
    assert rows[1].entity_completed_units >= 0.0
    assert rows[1].entity_dead_rate == 0.0
