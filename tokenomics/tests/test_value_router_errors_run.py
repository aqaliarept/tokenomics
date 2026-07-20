"""EXP-0038 actor/teacher error microscope: the collection + end-to-end dump.

torch-guarded (the actor is the EXP-0037 `ResidualScoreModel`). Kept tiny — 1 scale task,
horizon 12, 1 seed, lookahead 2 — the local smoke the HANDOVER prescribes to prove the file
schema + mistake extraction before any pod spend.
"""

from __future__ import annotations

import csv
import json

import pytest

pytest.importorskip("torch")

import torch

from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.curriculum import TaskRecord
from tokenomics.eval.value_router_errors import (
    collect_error_records,
    decision_row,
    fit_reproducible_actor,
    producer_rows,
    run_value_router_errors,
)
from tokenomics.train.value_router import PRODUCER_FEATURES, ResidualScoreModel


def _scale_task(seed: int, depth: int, episode_seed: int = 0) -> TaskRecord:
    return TaskRecord(
        task_id=f"scale-d{depth}-s{seed}", task_family="scale",
        generator_version="test", serializer_version="test",
        world_seed=seed, episode_seed=episode_seed,
        world_spec={"generator": "build_scale_config", "params": {"depth": depth}},
    )


def test_collect_error_records_captures_contested_decisions_with_both_signals():
    """The microscope's raw material: driving blend_0.5, every decision with >= 2 eligible
    producers is captured with the teacher labels AND the actor scores over the SAME eligible set,
    plus the order identity (cycle, order, product, quantity) so a mistake is inspectable. Records
    are only made where routing is a real choice."""
    torch.manual_seed(0)
    model = ResidualScoreModel()
    recs = collect_error_records(
        [_scale_task(0, depth=3)], PRODUCING_BASELINES["stock_aware"], model,
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12,
    )
    assert recs, "a depth-3 economy must present at least one contested routing decision"
    for r in recs:
        assert len(r.eligible_cols) >= 2
        assert r.labels.shape == r.scores.shape == (len(r.producer_ids),)
        assert r.rows.shape == (len(r.producer_ids), PRODUCER_FEATURES)
        assert isinstance(r.product, int) and r.quantity >= 1
        assert r.ttl_remaining >= 0


def test_decision_and_producer_rows_are_derivable_from_a_record():
    """A record renders to the two base CSVs: a decision-level row (teacher/actor picks, regret,
    identity) and one producer-level row per eligible producer (its label, score, and the facts the
    actor sees). Everything downstream (top_mistakes, summary) derives from these."""
    torch.manual_seed(0)
    model = ResidualScoreModel()
    recs = collect_error_records(
        [_scale_task(0, depth=3)], PRODUCING_BASELINES["stock_aware"], model,
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12,
    )
    d = decision_row(recs[0])
    assert d["order_id"] == recs[0].order_id
    assert d["teacher_best_producer"] in recs[0].producer_ids
    assert d["actor_best_producer"] in recs[0].producer_ids
    assert d["decision_regret"] >= 0.0

    prows = producer_rows(recs[0])
    assert len(prows) == len(recs[0].eligible_cols)
    assert {"producer_id", "teacher_label", "actor_score", "eligible"} <= set(prows[0])


def test_run_value_router_errors_smoke_writes_the_dump_layout(tmp_path):
    """HANDOVER first pass: local-smoke sized (1 task, horizon 12, seed 1, lookahead 2) end to end,
    proving the file schema before any pod spend. Emits the decision/producer base CSVs and the
    derived top_mistakes / alias_groups / summary, plus spec.json."""
    tasks = [_scale_task(0, depth=3)]
    prod = PRODUCING_BASELINES["stock_aware"]
    # Microscope the REAL (seed-reproduced) EXP-0037 actor, not a random-init model.
    model = fit_reproducible_actor(
        tasks, prod, delta=1.0, lookahead=2, eval_seeds=1, horizon=12, epochs=5, seed=1,
    )
    out = run_value_router_errors(
        tasks, prod=prod, model=model,
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, seed=1,
        out_dir=str(tmp_path), log=lambda *a, **k: None,
    )
    for name in ("decisions.csv", "producers.csv", "top_mistakes.csv",
                 "alias_groups.csv", "summary.csv", "spec.json"):
        assert (tmp_path / name).exists(), name

    srows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert len(srows) == 1
    s = srows[0]
    assert int(s["n_decisions"]) == out["n_decisions"]
    assert 0.0 <= float(s["top1_match_rate"]) <= 1.0
    assert float(s["mean_regret"]) >= 0.0

    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["seed"] == 1 and spec["lookahead"] == 2


def test_cli_resolves_the_producer_scorer_and_archive_loaders():
    """The pod dump's CLI (`main`) loads the EXP-0035 producer scorer and the task archive by
    deferred import — a wrong module path (scorer lives under train, not eval) would fail only at
    invocation, after arg parsing, so it must be pinned. Import exactly what main() imports."""
    from tokenomics.eval.curriculum_replay import _load_archive  # noqa: F401
    from tokenomics.train.producer_scorer import (  # noqa: F401
        load_producer_score_model,
        scorer_producing_policy,
    )
    from tokenomics.train.value_router import ResidualScoreModel  # noqa: F401
