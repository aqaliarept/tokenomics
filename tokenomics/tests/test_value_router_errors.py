"""EXP-0038 actor/teacher error microscope: per-decision teacher-vs-actor comparison.

Not a trainer. For every real blend_0.5 routing decision it asks "teacher wants weight on
producer X, actor wants weight on producer Y — what state separates X and Y?", so the
conclusion (add order/backlog/TTL features, change the teacher, or stop) is grounded in the
mistake dump, not a hypothesis (HANDOVER Definition Of Done).

Kept tiny (small scale configs, 1 task, seconds-long) so it stays local per the compute rule;
the real s29/s77/s45 dump runs on a pod only if needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenomics.eval.value_router_errors import (
    ErrorRecord,
    _alias_item,
    alias_groups,
    decision_metrics,
    summarize,
    top_mistakes,
)


def test_decision_metrics_scores_a_disagreement_by_teacher_regret():
    """Tracer bullet: over the eligible producers, the teacher's best is the max-label producer
    and the actor's best is the max-score producer. When they disagree, `decision_regret` is the
    teacher value the actor's pick leaves on the table (teacher-best label minus label at the
    actor pick) — always >= 0 — and top-1 does not match."""
    # 3 producers eligible; producer 2 has the best teacher label, producer 0 the best actor score.
    labels = np.array([0.1, 0.4, 0.9], dtype=np.float32)
    scores = np.array([1.0, 0.2, -0.5], dtype=np.float32)
    m = decision_metrics(labels, scores, eligible_cols=[0, 1, 2])

    assert m["teacher_best_producer"] == 2
    assert m["actor_best_producer"] == 0
    assert m["teacher_value_at_teacher_pick"] == pytest.approx(0.9, abs=1e-6)
    assert m["teacher_value_at_actor_pick"] == pytest.approx(0.1, abs=1e-6)
    assert m["decision_regret"] == pytest.approx(0.8, abs=1e-6)
    assert m["top1_match"] is False


def test_rank_correlation_captures_whole_ordering_agreement_not_just_top1():
    """A top-1 match can still hide a badly-ordered actor (and vice versa), so the microscope also
    reports the rank correlation between teacher labels and actor scores over the eligible set.
    Perfectly-aligned orderings give +1, exactly-reversed give -1."""
    labels = np.array([0.1, 0.4, 0.9], dtype=np.float32)
    aligned = np.array([0.0, 0.5, 1.0], dtype=np.float32)   # same ranking as labels
    reversed_ = np.array([1.0, 0.5, 0.0], dtype=np.float32)  # opposite ranking
    cols = [0, 1, 2]

    assert decision_metrics(labels, aligned, cols)["rank_correlation"] == pytest.approx(1.0)
    assert decision_metrics(labels, reversed_, cols)["rank_correlation"] == pytest.approx(-1.0)


def test_rank_correlation_is_defined_for_a_single_eligible_producer():
    """With only one eligible producer there is no ordering to correlate (and no real routing
    choice). The metric must stay finite — reported as 0.0 — so a decision row never carries a NaN
    into the CSVs / summary."""
    labels = np.array([0.5, 0.0], dtype=np.float32)
    scores = np.array([0.3, 0.0], dtype=np.float32)
    m = decision_metrics(labels, scores, eligible_cols=[0])
    assert m["rank_correlation"] == 0.0
    assert m["top1_match"] is True


def test_rank_correlation_is_zero_when_a_side_is_flat():
    """A flat teacher label vector (all producers equal) is exactly the failure mode this microscope
    diagnoses — there is no ordering to correlate. It must report 0.0, never an ordinal-rank
    artifact (argsort of a constant still yields distinct positions, so ranking must not decide
    constancy). Guards against a spurious +1 polluting rank_correlation in decisions.csv."""
    flat = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    varied = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    cols = [0, 1, 2]
    assert decision_metrics(flat, varied, cols)["rank_correlation"] == 0.0
    assert decision_metrics(varied, flat, cols)["rank_correlation"] == 0.0


def test_rank_correlation_is_tie_aware():
    """With a tie in one side the correlation must reflect averaged ranks, not the arbitrary order
    argsort breaks ties in. Two producers sharing a label, one clearly better: perfect monotone
    scores give a strong-but-not-degenerate positive correlation, and ordering-consistent."""
    labels = np.array([0.0, 0.0, 1.0], dtype=np.float32)   # first two tied
    aligned = np.array([0.2, 0.3, 0.9], dtype=np.float32)  # respects labels' only real gap
    r = decision_metrics(labels, aligned, [0, 1, 2])["rank_correlation"]
    assert 0.0 < r <= 1.0


def _mk_record(rows: np.ndarray, labels: np.ndarray) -> ErrorRecord:
    p = rows.shape[0]
    return ErrorRecord(
        task_id="t", seed=0, cycle=0, agent="a", order_id="o", stream_index=0, product=1,
        quantity=1, order_age=0, ttl_remaining=2, backlog_size=1, same_product_backlog=1,
        eligible_cols=list(range(p)), producer_ids=[f"p{i}" for i in range(p)],
        labels=labels, scores=np.zeros(p, dtype=np.float32), rows=rows)


def test_alias_hash_separates_near_but_not_identical_feature_rows():
    """Aliasing evidence must be exact: the actor receives full-precision floats, so two decisions
    whose feature rows differ in the 3rd/4th decimal are DISTINCT to the actor and must NOT be
    reported as the same alias (that would manufacture the "missing observation" signal). Only
    byte-identical actor features share a feature_hash."""
    base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    near = base.copy()
    near[0, 0] += 1e-3  # sub-rounding perturbation, still a distinct float to the model
    labels = np.array([1.0, 0.0], dtype=np.float32)

    h_base = _alias_item(_mk_record(base, labels))["feature_hash"]
    h_near = _alias_item(_mk_record(near, labels))["feature_hash"]
    h_same = _alias_item(_mk_record(base.copy(), labels))["feature_hash"]
    assert h_base != h_near      # near-alias must not collapse
    assert h_base == h_same      # identical features still alias


def test_summarize_reports_top1_match_rate_and_mean_regret():
    """The DoD headline (summary.csv): over all decisions, how often the actor's top-1 matches the
    teacher's, and the mean teacher regret the actor's picks incur. Built from decision rows, so a
    reviewer can recompute it from decisions.csv."""
    rows = [
        {"top1_match": True, "decision_regret": 0.0},
        {"top1_match": False, "decision_regret": 0.6},
        {"top1_match": False, "decision_regret": 0.2},
        {"top1_match": True, "decision_regret": 0.0},
    ]
    s = summarize(rows)
    assert s["n_decisions"] == 4
    assert s["top1_match_rate"] == pytest.approx(0.5)
    assert s["mean_regret"] == pytest.approx(0.2)
    assert s["max_regret"] == pytest.approx(0.6)


def test_summarize_handles_no_decisions():
    """An empty run (no contested decision) must not divide by zero — rates report 0.0 so the
    smoke can still write a summary.csv."""
    s = summarize([])
    assert s["n_decisions"] == 0
    assert s["top1_match_rate"] == 0.0
    assert s["mean_regret"] == 0.0


def test_top_mistakes_are_the_highest_regret_decisions_in_order():
    """top_mistakes.csv makes the largest failures inspectable without rerunning: the decisions
    ranked by `decision_regret` descending, truncated to `n`. Matched top-1 picks (regret 0) rank
    last and are dropped by the truncation."""
    rows = [
        {"order_id": "o_lo", "decision_regret": 0.1, "top1_match": False},
        {"order_id": "o_hi", "decision_regret": 0.9, "top1_match": False},
        {"order_id": "o_mid", "decision_regret": 0.5, "top1_match": False},
        {"order_id": "o_zero", "decision_regret": 0.0, "top1_match": True},
    ]
    top = top_mistakes(rows, n=2)
    assert [r["order_id"] for r in top] == ["o_hi", "o_mid"]


def test_alias_groups_flags_identical_features_with_conflicting_teacher_picks():
    """The aliasing check (alias_groups.csv): decisions where the actor sees the SAME feature rows
    (`feature_hash`) but the teacher prefers a DIFFERENT producer position mean the current
    features cannot separate what the teacher separates — the core "missing observation" evidence.
    Such a group has `label_variance` > 0; a group where the teacher always picks the same position
    is consistent (variance 0)."""
    items = [
        {"feature_hash": "A", "teacher_pos": 0, "order_id": "o1"},
        {"feature_hash": "A", "teacher_pos": 1, "order_id": "o2"},  # same features, other pick
        {"feature_hash": "B", "teacher_pos": 0, "order_id": "o3"},
        {"feature_hash": "B", "teacher_pos": 0, "order_id": "o4"},  # consistent
    ]
    groups = {g["feature_hash"]: g for g in alias_groups(items, min_group=2)}

    assert groups["A"]["n_decisions"] == 2
    assert groups["A"]["label_variance"] > 0.0
    assert groups["A"]["teacher_best_distribution"] == {"0": 1, "1": 1}
    assert set(groups["A"]["examples"]) == {"o1", "o2"}

    assert groups["B"]["label_variance"] == 0.0


def test_alias_groups_drops_singleton_feature_hashes():
    """A feature signature seen once carries no aliasing evidence (nothing to conflict with), so
    `min_group` filters it out — the CSV only shows repeated signatures worth inspecting."""
    items = [
        {"feature_hash": "A", "teacher_pos": 0, "order_id": "o1"},
        {"feature_hash": "solo", "teacher_pos": 0, "order_id": "o2"},
        {"feature_hash": "A", "teacher_pos": 1, "order_id": "o3"},
    ]
    hashes = {g["feature_hash"] for g in alias_groups(items, min_group=2)}
    assert hashes == {"A"}
