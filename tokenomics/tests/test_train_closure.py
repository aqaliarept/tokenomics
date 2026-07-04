"""Pure surface of the train-split closure diagnostic (no torch, no rollouts)."""

from __future__ import annotations

from tokenomics.eval.train_closure import (
    closure_summary, join_closure_rows, read_attribution_units,
)


def _joined(cand: float, champ: float, bar: float, task_id: str = "t") -> dict:
    return join_closure_rows([{"task_id": task_id, "candidate_units": cand}],
                             {task_id: (champ, bar)})[0]


def test_join_computes_closure_against_attribution_units() -> None:
    row = _joined(cand=50.0, champ=40.0, bar=60.0)
    assert row["gap_units"] == 20.0
    assert row["candidate_minus_champion"] == 10.0
    assert row["closure"] == 0.5


def test_tiny_gap_tasks_get_no_per_task_ratio() -> None:
    # a sub-1u mined gap makes the ratio noise; keep the row, null the ratio.
    row = _joined(cand=41.0, champ=40.0, bar=40.5)
    assert row["closure"] is None


def test_summary_aggregates_units_not_ratios() -> None:
    rows = [_joined(50.0, 40.0, 60.0, "a"),      # closed 10 of 20
            _joined(40.0, 40.0, 80.0, "b")]      # closed 0 of 40
    s = closure_summary(rows)
    assert s["n_tasks"] == 2
    assert s["aggregate_closure"] == round(10.0 / 60.0, 4)   # unit sums, not mean of ratios
    assert s["median_task_closure"] in (0.0, 0.5)


def test_read_attribution_units_joins_on_task_id(tmp_path) -> None:
    p = tmp_path / "gap_attribution.csv"
    p.write_text("task_id,champion_units,full_bar_units,extra\n"
                 "t1,40.0,60.0,x\nt2,10.0,12.5,y\n")
    assert read_attribution_units(p) == {"t1": (40.0, 60.0), "t2": (10.0, 12.5)}
