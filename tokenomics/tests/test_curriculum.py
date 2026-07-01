"""Curriculum game-master scaffold: TaskRecord serialize/replay, stress suite,
failure classification (CURRICULUM_GAME_MASTER_APPROACH.md Stage 0 + Stage 1).

Tests exercise the public curriculum surface: a task serializes to a jsonl line and
back, replays to the exact economy via its generator-ref (no hidden process state),
converts existing fixture families into stable-id records, and classifies a replay
outcome into the doc's failure taxonomy. Torch/champion smoke is guarded.
"""

from __future__ import annotations

from tokenomics.eval.curriculum import TaskRecord, replay
from tokenomics.eval.scale import build_scale_config


def test_task_record_roundtrips_through_json_line() -> None:
    rec = TaskRecord(
        task_id="stress-scale-d3-s6",
        task_family="scale",
        generator_version="v1",
        serializer_version="v1",
        world_seed=6,
        episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 3}},
        initial_state_spec={},
        event_trace=[],
        demand_trace=[],
        shock_trace=[],
        legal=True,
        replayable=True,
        reference_status="not_run",
        failure_class="none",
        expected_claim_boundary="in_claim",
        notes="",
    )
    line = rec.to_json_line()
    assert "\n" not in line
    assert TaskRecord.from_json_line(line) == rec


def test_replay_rebuilds_the_exact_economy_from_generator_ref() -> None:
    rec = TaskRecord(
        task_id="stress-scale-d3-s6",
        task_family="scale",
        generator_version="v1",
        serializer_version="v1",
        world_seed=6,
        episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 3}},
    )
    cfg = replay(rec)
    # Replay must reproduce the exact pure builder output (no hidden process state).
    assert cfg == build_scale_config(6, depth=3)
    cfg.validate()  # a legal, runnable economy


import pytest
from tokenomics.eval.curriculum import ReplayOutcome, classify_failure


@pytest.mark.parametrize("outcome,boundary,expected", [
    # contract gate first: a checkpoint that won't load is never a policy score.
    (ReplayOutcome(None, 8.0, False, False, contract_ok=False), "in_claim", "contract_failure"),
    # a crash/timeout/NaN during replay is an implementation failure, not a hard case.
    (ReplayOutcome(None, 8.0, False, False, contract_ok=True, crashed=True), "in_claim", "implementation_failure"),
    # reference beats learner on the same in-claim task -> a real policy weakness.
    (ReplayOutcome(6.0, 10.0, False, False, contract_ok=True), "in_claim", "policy_weakness"),
    # learner >= reference, no death -> nothing wrong.
    (ReplayOutcome(10.0, 6.0, False, False, contract_ok=True), "in_claim", "none"),
    (ReplayOutcome(6.0, 6.0, False, False, contract_ok=True), "in_claim", "none"),
    # out-of-claim gap is NOT a promoted-claim policy weakness (doc: don't miscount it).
    (ReplayOutcome(6.0, 10.0, False, False, contract_ok=True), "out_of_claim", "distribution_mismatch"),
    # a labeled wall where every policy dies is a feasibility wall, not a policy miss.
    (ReplayOutcome(0.0, 0.0, True, True, contract_ok=True), "feasibility_wall", "feasibility_wall"),
    # a "wall" the learner actually solves was mislabeled -> fall through, not a wall.
    (ReplayOutcome(10.0, 8.0, False, False, contract_ok=True), "feasibility_wall", "none"),
])
def test_classify_failure_taxonomy(outcome, boundary, expected) -> None:
    assert classify_failure(outcome, boundary=boundary) == expected


from tokenomics.eval.curriculum import stress_suite, FAILURE_CLASSES


def test_stress_suite_is_legal_replayable_and_multi_family() -> None:
    suite = stress_suite()
    assert len(suite) >= 8

    ids = [r.task_id for r in suite]
    assert len(ids) == len(set(ids)), "task ids must be unique and stable"

    # Independent stress suite must span more than one hard-case family (doc Stage 1),
    # and carry the known feasibility wall labeled separately from policy failures.
    assert {"scale", "fixture", "stress"} <= {r.task_family for r in suite}
    boundaries = {r.expected_claim_boundary for r in suite}
    assert {"in_claim", "out_of_claim", "feasibility_wall"} <= boundaries

    for r in suite:
        assert r.legal and r.replayable
        assert r.failure_class in FAILURE_CLASSES
        replay(r).validate()  # every stress task reproduces a legal, runnable economy


from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PRODUCER_CKPT = _REPO / "experiments/EXP-0021-producer-broaden-training/artifacts/producer_scorer_broad.pt"
_ORDERER_CKPT = _REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"


def test_replay_smoke_runs_champion_and_reference_and_classifies() -> None:
    pytest.importorskip("torch")
    if not (_PRODUCER_CKPT.exists() and _ORDERER_CKPT.exists()):
        pytest.skip("portable stack checkpoints not present")
    from tokenomics.eval.curriculum import run_replay_suite
    from tokenomics.train.producer_scorer import load_producer_score_model
    from tokenomics.train.orderer_scorer import load_scorer

    prod = load_producer_score_model(str(_PRODUCER_CKPT))
    order = load_scorer(str(_ORDERER_CKPT))

    subset = [r for r in stress_suite()
              if r.task_id in ("stress-scale-d3-s6", "stress-degenerate-primal_only-s0")]
    rows = run_replay_suite(subset, prod, order, eval_seeds=1, horizon=40)

    assert len(rows) == 2
    for row in rows:
        assert row["failure_class"] in FAILURE_CLASSES
        # reference bar (entity_planner+proportional) is deterministic and always runs.
        assert row["reference_units"] is not None
        assert row["reference_status"] in ("solved", "partial", "failed", "not_run")
        # target_depth authoritative (recipe chain), never blank; Stage 1 telemetry present.
        assert isinstance(row["target_depth"], int) and row["target_depth"] >= 1
        assert row["reference_model_id"] == "entity_planner+proportional"
        assert row["reference_rejected_units"] is not None
        for col in ("task_id", "contract_ok", "learner_dead", "learner_rejected_units"):
            assert col in row


from tokenomics.eval.curriculum import _episode_seeds


def test_episode_seeds_start_at_recorded_seed() -> None:
    # A one-row archive with episode_seed=7 must replay seed 7, not 0.
    assert _episode_seeds(7, 1) == [7]
    # --eval-seeds > 1 averages seeds anchored at the recorded seed, not [0..n).
    assert _episode_seeds(7, 3) == [7, 8, 9]
    assert _episode_seeds(0, 2) == [0, 1]


from tokenomics.eval.curriculum import _chain_depth
from tokenomics.eval.fixtures import build_deep_join_config, build_forkjoin_config


def test_chain_depth_reflects_recipe_chain_not_capacity_types() -> None:
    assert _chain_depth(build_scale_config(0, depth=3)) == 3
    assert _chain_depth(build_forkjoin_config(0)) == 3      # default depth 3
    assert _chain_depth(build_deep_join_config(0)) == 5     # default depth 5 (the wall)


def test_stress_records_serialize_depth_so_matrix_depth_is_never_blank() -> None:
    # Every record must carry depth in its generator params (self-describing archive),
    # incl. fixture/scarcity/degenerate rows that used to rely on builder defaults.
    for r in stress_suite():
        assert "depth" in r.world_spec["params"], r.task_id
    # and the deep_join wall must record depth 5, not blank.
    wall = next(r for r in stress_suite() if r.task_id == "stress-deep_join-s0")
    assert wall.world_spec["params"]["depth"] == 5
    assert _chain_depth(replay(wall)) == 5


from tokenomics.eval.curriculum import _MATRIX_COLUMNS, REFERENCE_MODEL_ID


def test_matrix_schema_carries_rejected_units_and_reference_identity() -> None:
    # Stage 1 stress-row contract: per-row rejected units (learner + reference) and the
    # reference identity, so a low-completed-units row is diagnosable and a future mixed
    # reference bar is attributable.
    for col in ("learner_rejected_units", "reference_rejected_units", "reference_model_id"):
        assert col in _MATRIX_COLUMNS
    assert REFERENCE_MODEL_ID == "entity_planner+proportional"


# --- stronger reference bar (Stage 3 enabler) ---------------------------------------
from tokenomics.eval.curriculum import REFERENCE_BARS, DEFAULT_REFERENCE_BARS


def test_reference_bar_registry_has_practical_and_search_bars() -> None:
    # The practical fixed bar (EXP-0023) plus a stronger bounded short-rollout / directed
    # search bar (doc Stage 0: "local routing search / bounded short-rollout where
    # available") — so Stage 3 mining can compare the learner against a bar that isn't a
    # single fixed orderer.
    assert "entity_planner+proportional" in REFERENCE_BARS
    assert "entity_planner+mpc_search" in REFERENCE_BARS
    assert DEFAULT_REFERENCE_BARS == ("entity_planner+proportional",)  # non-breaking default


def test_reference_bars_run_and_return_units_on_a_replayable_task() -> None:
    # Every registered bar is a checkpoint-free, deterministic-given-seed policy that runs
    # on a replayed economy and reports the stats the matrix reads (units/rejected/dead).
    cfg = build_scale_config(6, depth=3)
    for name, bar in REFERENCE_BARS.items():
        stats = bar(cfg, seed=0, horizon=30)
        for k in ("consumer_completed_units", "consumer_rejected_units", "dead"):
            assert k in stats, (name, k)
        assert stats["consumer_completed_units"] >= 0.0


from tokenomics.eval.curriculum import _best_reference


def test_best_reference_picks_highest_units_bar_and_is_attributable() -> None:
    # The reference the learner is scored against is the STRONGEST available bar, and the
    # winning bar id is retained so the row's ratio is attributable (doc: reference
    # identity per row; never read a ratio against an unknown bar).
    scored = {
        "entity_planner+proportional": {"consumer_completed_units": 453.0, "dead": 0.0},
        "entity_planner+mpc_search": {"consumer_completed_units": 501.0, "dead": 0.0},
    }
    name, stats = _best_reference(scored)
    assert name == "entity_planner+mpc_search"
    assert stats["consumer_completed_units"] == 501.0
    # tie -> the REGISTRY-first bar wins, independent of the caller's dict order. Build the
    # tie dict with mpc_search FIRST to prove attribution follows REFERENCE_BARS order, not
    # the --reference-bars ordering (reversing the flag must not flip the tie winner).
    tie = {
        "entity_planner+mpc_search": {"consumer_completed_units": 400.0, "dead": 0.0},
        "entity_planner+proportional": {"consumer_completed_units": 400.0, "dead": 0.0},
    }
    assert _best_reference(tie)[0] == "entity_planner+proportional"
