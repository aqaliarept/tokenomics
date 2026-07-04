"""Pure surface of the single-task overfit discriminator (no torch, no distills)."""

from __future__ import annotations

from tokenomics.eval.overfit_probe import probe_verdict, read_probe_units


def test_verdict_teacher_matched_within_fidelity_floor() -> None:
    v = probe_verdict(clone_units=99.0, teacher_units=100.0, champion_units=80.0)
    assert v["teacher_matched"]                 # 0.99 >= 0.98 floor
    assert v["fidelity"] == 0.99
    assert v["closure"] == 0.95                 # closed 19 of 20


def test_verdict_not_matched_below_floor() -> None:
    v = probe_verdict(clone_units=90.0, teacher_units=100.0, champion_units=80.0)
    assert not v["teacher_matched"]
    assert v["closure"] == 0.5


def test_verdict_tiny_gap_nulls_closure_but_fidelity_stands() -> None:
    v = probe_verdict(clone_units=100.0, teacher_units=100.5, champion_units=100.0)
    assert v["closure"] is None
    assert v["teacher_matched"]


def test_read_probe_units_picks_teacher_stack_column(tmp_path) -> None:
    p = tmp_path / "gap_attribution.csv"
    p.write_text("task_id,champion_units,full_bar_units,bar_producer_champ_orderer_units\n"
                 "t1,878.0,884.67,956.67\n")
    assert read_probe_units(p) == {"t1": (956.67, 878.0)}
