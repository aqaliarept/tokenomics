"""EXP-0010D: producer size-holdout audit.

The audit is separate from reward optimization: it reports reward/death quality and
runtime/memory scaling as distinct columns, while checking row-order/product/capacity
relabel gates at each generated size.
"""

from __future__ import annotations


def test_size_holdout_audit_reports_reward_and_scaling_separately():
    from tokenomics.eval.producer_size_holdout import audit_entity_planner_size_holdout

    rows = audit_entity_planner_size_holdout(
        sizes=[2, 3],
        seeds=[0],
        depth=2,
        producers_per_layer=2,
        horizon=3,
    )

    assert [r.products_per_layer for r in rows] == [2, 3]
    for row in rows:
        assert row.consumer_completed_units >= 0
        assert row.dead in (True, False)
        assert row.elapsed_sec >= 0.0
        assert row.peak_kib > 0.0


def test_scale_generator_supports_widths_above_single_digit_for_size_holdouts():
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(
        0,
        depth=2,
        products_per_layer=16,
        producers_per_layer=2,
        num_consumers=2,
    )

    assert len({s.product for s in cfg.process_sheets}) == 32
    cfg.validate()


def test_size_holdout_gates_pass_for_generated_topology():
    from tokenomics.eval.producer_size_holdout import producer_size_gates
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(
        0,
        depth=3,
        products_per_layer=4,
        producers_per_layer=3,
        num_consumers=2,
    )

    gates = producer_size_gates(cfg, products_per_layer=4, producers_per_layer=3)

    assert gates.row_order
    assert gates.product_relabel
    assert gates.capacity_relabel


def test_size_holdout_product_relabel_passes_when_headroom_is_scarce():
    """At larger widths, a composite producer may not have enough inventory headroom
    to replenish every equally valued input. The planner must not pick that subset by
    product id."""
    from tokenomics.eval.producer_size_holdout import producer_size_gates
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(
        0,
        depth=3,
        products_per_layer=16,
        producers_per_layer=4,
        num_consumers=3,
    )

    gates = producer_size_gates(cfg, products_per_layer=16, producers_per_layer=4)

    assert gates.product_relabel


def test_size_holdout_summary_csv_uses_separate_scaling_columns(tmp_path):
    from tokenomics.eval.producer_size_holdout import (
        audit_entity_planner_size_holdout,
        write_size_audit_csv,
    )

    rows = audit_entity_planner_size_holdout(
        sizes=[2],
        seeds=[0],
        depth=2,
        producers_per_layer=2,
        horizon=2,
    )
    out = tmp_path / "summary.csv"

    write_size_audit_csv(rows, out)

    text = out.read_text()
    assert "consumer_completed_units" in text
    assert "elapsed_sec" in text
    assert "peak_kib" in text
