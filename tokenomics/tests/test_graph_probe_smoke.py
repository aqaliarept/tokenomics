"""EXP-0039 Phase-6 local smoke: graph dump -> GNN probe -> gate table, end to end.

Tiny (depth-3 scale, 2 seeds, horizon 8, lookahead 2, 2 epochs, 1 model seed) — proves the
whole pipeline (topology compile, snapshot dump, NPZ reload, one training pass, leave-one-
seed-out gate, floors, metrics file) runs and writes finite values before any pod spend. The
numbers are NOT the experimental read (that is the s29/s77/s45 pod run).
"""

from __future__ import annotations

import csv
import os

import pytest

pytest.importorskip("torch")

from tokenomics.eval import graph_value_router_dump as gd
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.train import graph_router_probe as gp

from test_graph_view import _scale_task


def test_dump_then_probe_end_to_end(tmp_path):
    dump = str(tmp_path / "dump_graph")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=2, horizon=8, out_dir=dump, log=lambda *a, **k: None)

    # the pre-registered shape guard rejects a partial (1-task, <3-seed) dump by default
    with pytest.raises(ValueError):
        gp.run_probe(dump, str(tmp_path / "reject"), model_seeds=(0, 1, 2), epochs=1,
                     log=lambda *a, **k: None)

    task0 = next(csv.DictReader(open(f"{dump}/candidates.csv")))["task_id"]
    probe = str(tmp_path / "probe")
    # a high-precision override that would flip PASS->FAIL if the floor CSV rounded to 4 dp (1.000049
    # -> 1.0), so the smoke also guards Finding-5.3 (floor_by_task.csv full precision).
    override_val = 1.000049
    result = gp.run_probe(dump, probe, model_seeds=(0,), epochs=2, require_shape=False,
                          actor_floor_override={task0: override_val},
                          actor_floor_override_source={"path": "floor.json", "sha256": "abc123"},
                          source_commit="smoke-commit", log=lambda *a, **k: None)
    # Finding-6.1/7.1: the probe result binds to the immutable source identity (whole-tree bytes +
    # commit), matching the dump's fingerprint fn, so a training/metric/verdict code change is
    # detectable (and cross-checked against the dump on a real run).
    assert result["source"]["tree_sha256"] == gd.source_fingerprint()["tree_sha256"]
    assert result["source"]["commit"] == "smoke-commit"
    # device + determinism preflight are recorded; CPU is bitwise-exact (HANDOVER.md:381).
    # Finding-4.4: the preflight certifies a REAL leave-one-seed-out fold at the run's FULL epochs
    # (two identical full runs -> identical predictions), not a truncated 40x10 subset.
    det = result["determinism"]
    assert result["device"] == "cpu" and det["max_abs_diff"] == 0.0
    assert det["epochs"] == 2 and det["n_train_decisions"] >= 1 and det["fold"]
    # the override REPLACES the derived floor for the verdict, and both are recorded
    assert result["verdict_floor"] == "override"
    assert result["actor_floor_override"] == {task0: override_val}
    assert result["actor_floor_override_source"]["sha256"] == "abc123"

    with open(f"{probe}/metrics_by_task.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows, "the gate must score at least one task"
    for r in rows:
        assert float(r["gnn_mean_regret"]) >= 0.0
        assert int(r["n_decisions"]) >= 1
    # Finding-4.3: the table must report the floor that DECIDES the verdict, with its source, so it
    # never contradicts the override the verdict actually used. The derived actor floor is still
    # shown alongside (it is a reported baseline). The display table rounds (4 dp).
    m0 = next(r for r in rows if r["task_id"] == task0)
    assert m0["floor_source"] == "override" and float(m0["verdict_floor_regret"]) == round(override_val, 4)
    assert float(m0["actor_mean_regret"]) != override_val  # derived floor still reported
    # Finding-5.3: floor_by_task.csv is the reproduction artifact — it must carry the verdict floor at
    # FULL precision (not the display rounding) and record the override's source, so the verdict is
    # reproducible from this CSV alone.
    with open(f"{probe}/floor_by_task.csv") as f:
        floors = {r["task_id"]: r for r in csv.DictReader(f)}
    assert floors[task0]["floor_source"] == "override"
    assert float(floors[task0]["verdict_floor_regret"]) == override_val  # exact, not 1.0
    assert floors[task0]["override_source"] == "floor.json@sha256:abc123"
    assert result["epochs"] == 2 and result["model_seeds"] == [0]
    # provenance: the read is stamped with the schema it was scored against and a computed
    # pass/fail from the pre-registered conditions (not a human reinterpretation)
    assert result["schema_hash"] == gd.schema_hash()
    assert "pass" in result["verdict"] and set(result["verdict"]["conditions"]) >= {
        "beats_actor_2of3", "no_task_worse_5pct", "stable_ge3_seeds"}

    # audit artifacts (HANDOVER.md:379/405/408): per-fold normalization actually used for
    # training, loss history, a checkpoint per fold, and predictions.csv from which the gate is
    # recomputable without a checkpoint or the simulator.
    assert result["folds"], "per-fold audit metadata must be written"
    for fold in result["folds"]:
        assert fold["normalization"] and fold["loss_history"]
        assert os.path.exists(os.path.join(probe, fold["checkpoint"]))
    with open(f"{probe}/predictions.csv") as f:
        preds = list(csv.DictReader(f))
    assert preds and len(preds) == result["n_predictions"]
    assert set(preds[0]) >= {"task_id", "held_seed", "model_seed", "decision_index",
                             "candidate_index", "pred", "teacher_label", "actor_score"}
    # recompute a pooled per-task regret from predictions.csv alone and match the reported metric
    from tokenomics.eval.observation_ladder import _routing_metrics
    task0 = rows[0]["task_id"]
    prows = [{"decision_key": (p["task_id"], int(p["held_seed"]), int(p["decision_index"])),
              "teacher_label": float(p["teacher_label"]), "task_id": p["task_id"],
              "actor_score": float(p["actor_score"])}
             for p in preds if p["task_id"] == task0 and int(p["model_seed"]) == 0]
    recomputed = _routing_metrics(prows, [float(p["pred"]) for p in preds
                                          if p["task_id"] == task0 and int(p["model_seed"]) == 0])
    assert abs(recomputed["mean_regret"] - result["per_seed"][0][task0]["mean_regret"]) < 1e-6
