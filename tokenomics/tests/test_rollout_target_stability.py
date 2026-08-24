"""EXP-0040 collector/analyzer contracts.

The expensive run is intentionally absent.  These tests prove deterministic authority
sampling, exact raw-row coverage, artifact-only gate decisions, and a tiny real-simulator
collector path before the registered CPU-pod run buys labels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tokenomics.eval import rollout_target_stability as rs

from test_graph_view import _scale_task


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _candidate(producer: str, label: float, score: float) -> rs.CandidateAuthority:
    return rs.CandidateAuthority(producer, label, score)


def _decision(*, task: str = "task", seed: int = 1, cycle: int = 0, order: str = "o1",
              flat: bool = False, rank: str = "0") -> rs.SelectedDecision:
    labels = (0.0, 0.0) if flat else (1.0, 0.0)
    return rs.SelectedDecision(
        decision_index=cycle, task_id=task, seed=seed, cycle=cycle, agent="order_c0__x",
        order_id=order, stream_index=0,
        decision_digest=rs.decision_digest(task, seed, cycle, order),
        cycle_quartile=min(3, cycle // 25), sample_class="flat" if flat else "signal",
        actor_best_producer="p0",
        candidates=(_candidate("p0", labels[0], 1.0), _candidate("p1", labels[1], 0.0)),
        selection_rank=rank,
    )


def test_sampling_is_lowest_hash_per_stratum_without_backfill() -> None:
    decisions = tuple([
        _decision(cycle=1, order="o1", rank="c"),
        _decision(cycle=2, order="o2", rank="a"),
        _decision(cycle=3, order="o3", rank="b"),
        _decision(cycle=4, order="o4", flat=True, rank="d"),
    ])
    selected, manifest = rs.select_sample(
        rs.Authority(decisions, {}), signal_per_stratum=2, flat_per_stratum=3)
    assert {d.order_id for d in selected} == {"o2", "o3", "o4"}
    flat = next(s for s in manifest["strata"] if s["sample_class"] == "flat")
    assert flat["selected"] == 1 and flat["shortfall"] == 2


def test_authority_join_requires_exact_decision_candidate_and_actor_alignment() -> None:
    d = {"task_id": "t", "seed": "1", "cycle": "2", "agent": "order_c0__x",
         "order_id": "o", "stream_index": "0", "n_eligible": "2"}
    c = [
        {"decision_index": "0", "task_id": "t", "seed": "1", "cycle": "2",
         "order_id": "o", "candidate_index": "0", "producer_id": "p0",
         "teacher_label": "2"},
        {"decision_index": "0", "task_id": "t", "seed": "1", "cycle": "2",
         "order_id": "o", "candidate_index": "1", "producer_id": "p1",
         "teacher_label": "0"},
    ]
    ad = [{**d, "actor_best_producer": "p1"}]
    ap = [
        {**{k: row[k] for k in ("task_id", "seed", "cycle", "order_id", "producer_id")},
         "eligible": "True", "actor_score": str(i)}
        for i, row in enumerate(c)
    ]
    joined = rs._join_authorities([d], c, ad, ap)
    assert joined[0].actor_best_producer == "p1"
    with pytest.raises(ValueError, match="missing canonical actor candidate"):
        rs._join_authorities([d], c, ad, ap[:1])
    with pytest.raises(ValueError, match="actor_best_producer"):
        rs._join_authorities([d], c, [{**d, "actor_best_producer": "p0"}], ap)


def _stable_rows(*, original_matches: bool, actor_good: bool = False) -> list[dict]:
    rows: list[dict] = []
    for ti, task in enumerate(("a", "b", "c")):
        for seed in (1, 2, 3):
            digest = 1000 * ti + seed
            original = {"p0": 2.0, "p1": 0.0} if original_matches else {"p0": 0.0, "p1": 2.0}
            actor = "p0" if actor_good else "p1"
            for p in ("p0", "p1"):
                rows.append({
                    "decision_index": digest, "task_id": task, "seed": seed, "cycle": 0,
                    "agent": "order", "order_id": f"o{seed}", "stream_index": 0,
                    "decision_digest": digest, "cycle_quartile": 0,
                    "sample_class": "signal", "actor_best_producer": actor,
                    "producer_id": p, "continuation_id": -1, "label": original[p],
                })
            for cid in range(16):
                # Non-zero within-candidate variance, but an invariant ranking/winner.
                values = {"p0": 10.0 + cid * 0.01, "p1": -cid * 0.01}
                for p in ("p0", "p1"):
                    rows.append({
                        "decision_index": digest, "task_id": task, "seed": seed, "cycle": 0,
                        "agent": "order", "order_id": f"o{seed}", "stream_index": 0,
                        "decision_digest": digest, "cycle_quartile": 0,
                        "sample_class": "signal", "actor_best_producer": actor,
                        "producer_id": p, "continuation_id": cid, "label": values[p],
                    })
    return rows


def test_analyzer_licenses_mean_target_when_future_average_stable_useful_and_different() -> None:
    result = rs.analyze_rows(_stable_rows(original_matches=False), repeats=10)
    assert result["gate1_convergence"] and result["gate2_usefulness"]
    assert result["classification"] == "MEAN_TARGET_LICENSED"
    assert result["convergence_by_k"] == {"4": True, "8": True}
    assert len(result["passing_usefulness_tasks"]) == 3


def test_analyzer_licenses_coverage_when_original_and_expected_preferences_agree() -> None:
    result = rs.analyze_rows(_stable_rows(original_matches=True), repeats=10)
    assert result["classification"] == "COVERAGE_PATH_LICENSED"


def test_analyzer_returns_search_only_when_expected_selector_cannot_beat_actor() -> None:
    result = rs.analyze_rows(_stable_rows(original_matches=True, actor_good=True), repeats=10)
    assert result["gate1_convergence"]
    assert not result["gate2_usefulness"]
    assert result["classification"] == "SEARCH_ONLY"


def test_analyzer_refuses_partial_or_duplicated_raw_coverage() -> None:
    rows = _stable_rows(original_matches=True)
    with pytest.raises(ValueError, match="registered task/seed scope"):
        rs.analyze_rows(rows, repeats=2, require_registered_scope=True)
    with pytest.raises(ValueError, match="duplicate raw label"):
        rs.analyze_rows([*rows, rows[-1]], repeats=2)
    missing = [r for r in rows if not (
        r["task_id"] == "a" and r["seed"] == 1 and r["producer_id"] == "p1" and
        r["continuation_id"] == 7)]
    with pytest.raises(ValueError, match="candidate coverage mismatch"):
        rs.analyze_rows(missing, repeats=2)


def test_bootstrap_accepts_signed_decision_digest_seed() -> None:
    lo, hi = rs._bootstrap_interval((1.0, 2.0, 3.0), -7)
    assert 1.0 <= lo <= hi <= 3.0


def test_convergence_reports_fixed_ensemble_selector_regret_change() -> None:
    continuations = {}
    for cid in range(8):
        continuations[cid] = {"p0": 1.0, "p1": 10.0 if cid < 4 else -20.0}
    d = {
        "task_id": "t", "seed": 1, "cycle": 0, "order_id": "o",
        "sample_class": "signal", "actor_best_producer": "p0",
        "original": {"p0": 1.0, "p1": 0.0}, "continuations": continuations,
    }
    metrics = rs._convergence_metrics([d], list(range(8)), size=4, repeats=20)
    assert metrics["mean_abs_ensemble_regret_change"] > 0.0


def test_tiny_collector_replays_authority_and_persists_raw_continuations(tmp_path: Path) -> None:
    """Real simulator path, pinned smoke-size: one topology/seed/decision, K=2, lookahead=1."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.graph_value_router_dump import run_graph_dump

    task = _scale_task(0, depth=3)
    dump = tmp_path / "dump"
    run_graph_dump(
        [task], prod=PRODUCING_BASELINES["stock_aware"], eval_seeds=1,
        delta=1.0, lookahead=1, horizon=8, out_dir=str(dump), log=lambda *a, **k: None)
    with (dump / "decisions.csv").open() as f:
        decisions = list(csv.DictReader(f))
    with (dump / "candidates.csv").open() as f:
        candidates = list(csv.DictReader(f))
    actor_decisions = [{**d} for d in decisions]
    actor_producers = [{
        "task_id": c["task_id"], "seed": c["seed"], "cycle": c["cycle"],
        "order_id": c["order_id"], "producer_id": c["producer_id"], "eligible": "True",
        "actor_score": c["actor_score"],
    } for c in candidates]
    joined = rs._join_authorities(decisions, candidates, actor_decisions, actor_producers)
    selected, _ = rs.select_sample(rs.Authority(joined, {}), max_decisions=1)
    raw = tmp_path / "continuation_labels.csv"
    rows = rs.collect_labels(
        [task], prod=PRODUCING_BASELINES["stock_aware"], selected=selected,
        continuation_ids=(0, 1), out_csv=str(raw), delta=1.0, lookahead=1, horizon=8,
        log=lambda *a, **k: None)
    assert raw.is_file()
    assert {int(r["continuation_id"]) for r in rows} == {-1, 0, 1}
    assert len(rows) == 3 * len(selected[0].candidates)


def test_registered_input_hash_mismatch_stops_before_collection(tmp_path: Path) -> None:
    paths = {}
    for name in rs.REGISTERED_INPUT_SHA256:
        p = tmp_path / name
        p.write_text("wrong")
        paths[name] = str(p)
    with pytest.raises(ValueError, match="stop before rollout spend"):
        rs._input_manifest(paths, require_registered=True)
