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


def test_fullstate_rows_add_per_type_dispersion_and_telemetry_history():
    """EXP-0038A blocks 4+7: with full=True, each record carries per-eligible-producer block-4
    dispersion stats (per-required-type cap/reserved/max/headroom min<=max, input stats, output inv,
    makeable, counts) and block-7 telemetry history (routed/completed/rejected EMAs + producer
    acceptance/completion/rejection rates). All deployable at decision time (past-cycle folded EMA).
    """
    torch.manual_seed(0)
    model = ResidualScoreModel()
    recs = collect_error_records(
        [_scale_task(0, depth=3)], PRODUCING_BASELINES["stock_aware"], model,
        delta=1.0, lookahead=2, eval_seeds=1, horizon=16, full=True,
    )
    assert recs
    r = recs[0]
    assert r.full_rows is not None
    pid = r.producer_ids[r.eligible_cols[0]]
    fr = r.full_rows[pid]
    # block-4 dispersion: min <= max for every collapsed ratio family
    for fam in ("b4_cap", "b4_resv", "b4_max", "b4_head", "b4_inp"):
        assert fr[f"{fam}_min"] <= fr[f"{fam}_max"] + 1e-6
    assert fr["b4_n_req_types"] >= 1
    assert fr["b4_makeable"] >= 0
    # block-7 history: all present and finite floats
    for col in ("h_routed_ema", "h_completed_ema", "h_rejfail_ema", "h_success_rate",
                "h_prod_acceptance_rate", "h_prod_completion_rate", "h_prod_rejection_rate"):
        assert isinstance(fr[col], float)
        assert fr[col] >= 0.0


def test_full_dump_writes_block4_and_history_columns(tmp_path):
    """EXP-0038A re-dump smoke: run_value_router_errors(full=True) rides the block-4/block-7 columns
    into producers.csv alongside feat_0..6, ready for observation_ladder's +full_producer_state /
    +history blocks. Tiny (1 task, horizon 12)."""
    tasks = [_scale_task(0, depth=3)]
    prod = PRODUCING_BASELINES["stock_aware"]
    model = fit_reproducible_actor(
        tasks, prod, delta=1.0, lookahead=2, eval_seeds=1, horizon=12, epochs=5, seed=1,
    )
    run_value_router_errors(
        tasks, prod=prod, model=model, delta=1.0, lookahead=2, eval_seeds=1, horizon=12, seed=1,
        out_dir=str(tmp_path), full=True, log=lambda *a, **k: None,
    )
    prows = list(csv.DictReader((tmp_path / "producers.csv").open()))
    assert prows
    cols = set(prows[0])
    assert {f"feat_{i}" for i in range(7)} <= cols          # base row still present
    assert {"b4_cap_min", "b4_cap_std", "b4_head_mean", "b4_makeable", "b4_n_req_types"} <= cols
    assert {"h_routed_ema", "h_prod_completion_rate", "h_success_rate"} <= cols
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["full"] is True


def test_producer_product_v1_rows_carry_all_phase1_blocks():
    """EXP-0038B feature_set='producer_product_v1': each eligible producer row carries the deployable
    per-producer/product Phase-1 blocks (P1 pp_throughput_, P2 pp_opp_, P3 pp_dep_, P4 pp_base_,
    P6 pp_fulfill_) plus the reused P5 history (h_*). Sanity: throughput dispersion is ordered
    (min<=max) and the base softmax weights over a decision's eligible producers sum to ~1."""
    import numpy as np
    torch.manual_seed(0)
    model = ResidualScoreModel()
    recs = collect_error_records(
        [_scale_task(0, depth=3)], PRODUCING_BASELINES["stock_aware"], model,
        delta=1.0, lookahead=2, eval_seeds=1, horizon=16, feature_set="producer_product_v1",
    )
    assert recs
    r = recs[0]
    assert r.full_rows is not None
    pid = r.producer_ids[r.eligible_cols[0]]
    fr = r.full_rows[pid]
    for col in ("pp_throughput_req_type_count", "pp_throughput_current_min", "pp_throughput_current_max",
                "pp_opp_alt_product_count", "pp_dep_is_primal", "pp_base_logit", "pp_base_rank",
                "pp_fulfill_feasible_before_ttl", "h_routed_ema"):
        assert col in fr, col
    assert fr["pp_throughput_current_min"] <= fr["pp_throughput_current_max"] + 1e-6
    # softmax weights across the decision's eligible producers form a distribution
    w = sum(r.full_rows[r.producer_ids[c]]["pp_base_softmax_weight"] for c in r.eligible_cols)
    assert abs(w - 1.0) < 1e-5


def test_producer_product_v1_dump_writes_pp_columns(tmp_path):
    """EXP-0038B re-dump smoke: run_value_router_errors(feature_set='producer_product_v1') rides the
    pp_* + h_* columns into producers.csv and records the feature_set in spec.json."""
    tasks = [_scale_task(0, depth=3)]
    prod = PRODUCING_BASELINES["stock_aware"]
    model = fit_reproducible_actor(
        tasks, prod, delta=1.0, lookahead=2, eval_seeds=1, horizon=12, epochs=5, seed=1,
    )
    run_value_router_errors(
        tasks, prod=prod, model=model, delta=1.0, lookahead=2, eval_seeds=1, horizon=12, seed=1,
        out_dir=str(tmp_path), feature_set="producer_product_v1", log=lambda *a, **k: None,
    )
    prows = list(csv.DictReader((tmp_path / "producers.csv").open()))
    assert prows
    cols = set(prows[0])
    assert {f"feat_{i}" for i in range(7)} <= cols
    assert {"pp_throughput_current_min", "pp_opp_capacity_opportunity_cost", "pp_dep_bottleneck_cover",
            "pp_base_margin_to_best", "pp_fulfill_est_cycles_to_complete"} <= cols
    assert "h_routed_ema" in cols
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["feature_set"] == "producer_product_v1"


def test_producer_product_system_v1_rows_carry_phase1_and_sys_blocks():
    """EXP-0038B Phase 2 feature_set='producer_product_system_v1': each eligible producer row carries
    the Phase-1 pp_* blocks AND the system-wide sys_* blocks (S1 demand, S2 capacity, S3 recipe,
    S4 cutoff, S5 actor). The decision-constant sys fields (e.g. capacity totals, recipe depth) are
    identical across a decision's producers; the producer-varying ones (*_competing_for_p_*) may
    differ. All finite."""
    import math
    torch.manual_seed(0)
    model = ResidualScoreModel()
    recs = collect_error_records(
        [_scale_task(0, depth=3)], PRODUCING_BASELINES["stock_aware"], model,
        delta=1.0, lookahead=2, eval_seeds=1, horizon=16, feature_set="producer_product_system_v1",
    )
    assert recs
    r = recs[0]
    assert r.full_rows is not None
    pid = r.producer_ids[r.eligible_cols[0]]
    fr = r.full_rows[pid]
    for col in ("pp_throughput_current_min", "pp_base_logit",           # Phase-1 still present
                "sys_demand_target_queued_qty", "sys_capacity_target_total_free_units",
                "sys_recipe_product_depth", "sys_cutoff_order_urgency",
                "sys_actor_recent_completed", "sys_demand_competing_for_p_capacity_qty"):
        assert col in fr, col
        assert math.isfinite(fr[col])
    # a decision-constant sys field takes ONE value across the decision's eligible producers
    caps = {r.full_rows[r.producer_ids[c]]["sys_capacity_target_total_free_units"]
            for c in r.eligible_cols}
    assert len(caps) == 1


def test_producer_product_system_v1_dump_writes_sys_columns(tmp_path):
    """EXP-0038B Phase-2 re-dump smoke: run_value_router_errors(feature_set='producer_product_system_v1')
    rides the sys_* columns (alongside pp_*/h_*) into producers.csv and records the feature_set."""
    tasks = [_scale_task(0, depth=3)]
    prod = PRODUCING_BASELINES["stock_aware"]
    model = fit_reproducible_actor(
        tasks, prod, delta=1.0, lookahead=2, eval_seeds=1, horizon=12, epochs=5, seed=1,
    )
    run_value_router_errors(
        tasks, prod=prod, model=model, delta=1.0, lookahead=2, eval_seeds=1, horizon=12, seed=1,
        out_dir=str(tmp_path), feature_set="producer_product_system_v1", log=lambda *a, **k: None,
    )
    prows = list(csv.DictReader((tmp_path / "producers.csv").open()))
    assert prows
    cols = set(prows[0])
    assert {"pp_throughput_current_min", "h_routed_ema"} <= cols       # Phase-1 blocks still there
    assert {"sys_demand_target_queued_qty", "sys_capacity_target_free_hhi",
            "sys_recipe_downstream_demand_unlocked", "sys_cutoff_ttl_pressure_bucket",
            "sys_actor_competing_qty_for_p"} <= cols
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["feature_set"] == "producer_product_system_v1"
