"""Stage 4 prioritized-replay pure half (CURRICULUM_GAME_MASTER_APPROACH.md Stage 4).

Tests the torch-free curriculum-replay primitives — the archive split, priority weights,
and priority sampling — so the replay/relabel/train driver's curation logic is verified
without loading checkpoints. The torch-heavy driver has its own guarded micro-smoke.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.eval.curriculum import TaskRecord
from tokenomics.eval.curriculum_replay import (
    replay_priorities, sample_replay, score_from_notes, split_archive,
)

_REPO = Path(__file__).resolve().parents[2]
_CHAMP_PROD = _REPO / "experiments/EXP-0021-producer-broaden-training/artifacts/producer_scorer_broad.pt"
_CHAMP_ORD = _REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"


def _rec(i: int, score: float) -> TaskRecord:
    """A minimal mined TaskRecord carrying a mining score in the archived `notes` form."""
    return TaskRecord(
        task_id=f"mine-t{i}",
        task_family="random",
        generator_version="mine-v1",
        serializer_version="v1",
        world_seed=i,
        episode_seed=i,
        world_spec={"generator": "build_scale_config", "params": {"depth": 3}},
        notes=f"mined: score={score} gap={score * 10}u win=1.0",
    )


def _archive(n: int) -> list[TaskRecord]:
    return [_rec(i, round(0.1 + i * 0.05, 3)) for i in range(n)]


# --- split_archive -------------------------------------------------------------------

def test_split_is_disjoint_and_total() -> None:
    recs = _archive(12)
    train, val, blind = split_archive(recs, val_frac=0.25, blind_frac=0.25, seed=0)
    ids = lambda rs: {r.task_id for r in rs}
    assert ids(train) | ids(val) | ids(blind) == ids(recs)
    assert not (ids(train) & ids(val))
    assert not (ids(train) & ids(blind))
    assert not (ids(val) & ids(blind))
    assert len(train) + len(val) + len(blind) == 12


def test_split_honors_fractions() -> None:
    recs = _archive(12)
    train, val, blind = split_archive(recs, val_frac=0.25, blind_frac=0.25, seed=0)
    assert len(val) == 3 and len(blind) == 3  # 25% each of 12
    assert len(train) == 6  # remainder to train


def test_split_is_deterministic_given_seed() -> None:
    recs = _archive(12)
    a = split_archive(recs, seed=7)
    b = split_archive(recs, seed=7)
    assert [r.task_id for r in a[0]] == [r.task_id for r in b[0]]
    assert [r.task_id for r in a[1]] == [r.task_id for r in b[1]]
    assert [r.task_id for r in a[2]] == [r.task_id for r in b[2]]


def test_split_seed_changes_partition() -> None:
    recs = _archive(20)
    a = split_archive(recs, seed=0)
    b = split_archive(recs, seed=1)
    # different shuffle -> different val membership (overwhelmingly likely for n=20)
    assert {r.task_id for r in a[1]} != {r.task_id for r in b[1]}


def test_split_tiny_archive_keeps_a_train_task() -> None:
    # 2 tasks, 25/25 split rounds val/blind to 0 -> both go to train (never empty train).
    train, val, blind = split_archive(_archive(2), val_frac=0.25, blind_frac=0.25, seed=0)
    assert len(train) == 2 and not val and not blind


# --- score_from_notes ----------------------------------------------------------------

def test_score_from_notes_parses_mining_score() -> None:
    assert score_from_notes("mined: score=0.5153 gap=7.667u win=1.0") == 0.5153


def test_score_from_notes_falls_back_to_gap() -> None:
    # no score= present -> use the gap magnitude so an un-scored record still ranks.
    assert score_from_notes("mined: gap=7.667u win=1.0") == 7.667


def test_score_from_notes_defaults_zero() -> None:
    assert score_from_notes("") == 0.0


# --- replay_priorities ---------------------------------------------------------------

def test_priorities_sum_to_one() -> None:
    p = replay_priorities([0.1, 0.2, 0.7])
    assert abs(sum(p) - 1.0) < 1e-9


def test_priorities_monotone_in_score() -> None:
    p = replay_priorities([0.1, 0.2, 0.7])
    assert p[0] < p[1] < p[2]


def test_priorities_eta_sharpens() -> None:
    lo = replay_priorities([0.1, 0.9], eta=1.0)
    hi = replay_priorities([0.1, 0.9], eta=3.0)
    # higher eta concentrates weight on the top task.
    assert hi[1] > lo[1]


def test_priorities_epsilon_floors_zero_score() -> None:
    # a 0-score task still gets positive weight via epsilon (never starves).
    p = replay_priorities([0.0, 0.0, 1.0], epsilon=0.01)
    assert p[0] > 0.0 and p[1] > 0.0


def test_priorities_empty() -> None:
    assert replay_priorities([]) == []


# --- sample_replay -------------------------------------------------------------------

def test_sample_is_deterministic_given_seed() -> None:
    recs = _archive(10)
    pri = replay_priorities([score_from_notes(r.notes) for r in recs])
    a = sample_replay(recs, pri, 4, seed=3)
    b = sample_replay(recs, pri, 4, seed=3)
    assert [r.task_id for r in a] == [r.task_id for r in b]


def test_sample_has_no_duplicates() -> None:
    recs = _archive(10)
    pri = replay_priorities([score_from_notes(r.notes) for r in recs])
    got = sample_replay(recs, pri, 6, seed=1)
    assert len({r.task_id for r in got}) == 6


def test_sample_k_capped_at_len() -> None:
    recs = _archive(5)
    pri = replay_priorities([score_from_notes(r.notes) for r in recs])
    got = sample_replay(recs, pri, 99, seed=0)
    assert len(got) == 5


def test_sample_favors_high_priority() -> None:
    # over many seeds, the top-score task is drawn far more often than the bottom one.
    recs = _archive(10)
    pri = replay_priorities([score_from_notes(r.notes) for r in recs])
    top, bot = recs[-1].task_id, recs[0].task_id
    top_hits = bot_hits = 0
    for s in range(200):
        picked = {r.task_id for r in sample_replay(recs, pri, 3, seed=s)}
        top_hits += top in picked
        bot_hits += bot in picked
    assert top_hits > bot_hits


def test_sample_with_replacement_prioritizes_full_budget() -> None:
    # the default Stage-4 draw: k == len WITH replacement. Priorities must bite (not just
    # reorder) -> aggregated over seeds the top task is replayed strictly more than the bottom,
    # and each batch keeps the full training budget. Without replacement it'd be one each.
    recs = _archive(10)
    pri = replay_priorities([score_from_notes(r.notes) for r in recs])
    top = bot = 0
    for s in range(200):
        got = sample_replay(recs, pri, len(recs), seed=s, replace=True)
        assert len(got) == len(recs)  # full budget every draw
        ids = [g.task_id for g in got]
        top += ids.count(recs[-1].task_id)
        bot += ids.count(recs[0].task_id)
    assert top > bot


def test_sample_with_replacement_may_exceed_pool_and_duplicate() -> None:
    recs = _archive(4)
    pri = replay_priorities([score_from_notes(r.notes) for r in recs])
    got = sample_replay(recs, pri, 12, seed=0, replace=True)  # k > pool
    assert len(got) == 12  # not capped
    assert len({r.task_id for r in got}) < 12  # duplicates present


def test_sample_with_replacement_deterministic() -> None:
    recs = _archive(8)
    pri = replay_priorities([score_from_notes(r.notes) for r in recs])
    a = sample_replay(recs, pri, 8, seed=5, replace=True)
    b = sample_replay(recs, pri, 8, seed=5, replace=True)
    assert [r.task_id for r in a] == [r.task_id for r in b]


# --- driver micro-smoke (torch) ------------------------------------------------------

def _mined_record(seed: int, depth: int, score: float) -> TaskRecord:
    """A single-cell mined task (build_scale_config depth), as Stage 3 archives them."""
    return TaskRecord(
        task_id=f"mine-build_scale_config-depth{depth}-s{seed}",
        task_family="random",
        generator_version="mine-v1",
        serializer_version="v1",
        world_seed=seed,
        episode_seed=seed,
        world_spec={"generator": "build_scale_config", "params": {"depth": depth}},
        failure_class="policy_weakness",
        reference_status="solved",
        notes=f"mined: score={score} gap={score * 10}u win=1.0",
    )


def test_run_stage4_replay_smoke(tmp_path) -> None:
    """End-to-end wiring on a tiny single-cell archive: relabel+train candidate score models,
    eval vs champion, emit the EXP layout + an advisory gate. Fast (depth-2, short horizon,
    2 fit steps, fast reference bar, no fixtures) — proves the pipeline, not the numbers."""
    pytest.importorskip("torch")
    if not (_CHAMP_PROD.exists() and _CHAMP_ORD.exists()):
        pytest.skip("champion checkpoints not present")
    from tokenomics.eval.curriculum import write_archive
    from tokenomics.eval.curriculum_replay import run_stage4_replay

    depth = 2
    archive = tmp_path / "task_archive.jsonl"
    write_archive([_mined_record(s, depth, 0.2 + 0.1 * i) for i, s in enumerate((0, 1, 2, 3))],
                  archive)
    out = tmp_path / "exp"
    captured: list[str] = []
    gate = run_stage4_replay(
        archive, str(_CHAMP_PROD), str(_CHAMP_ORD), out,
        depth=depth, epochs=2, iters=2, held_out_seeds=(6,), eval_seeds=1, horizon=20,
        reference_bar="entity_planner+proportional", fixtures=False, log=captured.append)

    # ordered [phase i/7] progress markers for the detached-run heartbeat.
    phases = [ln for ln in captured if ln.startswith("[phase ")]
    assert phases[0] == "[phase 1/7] split + prioritize"
    assert any(p.startswith("[phase 7/7]") for p in phases)

    assert (out / "artifacts" / "producer_candidate.pt").exists()
    assert (out / "artifacts" / "orderer_candidate.pt").exists()
    assert (out / "artifacts" / "replay_validation.csv").exists()
    assert (out / "artifacts" / "replay_blind.csv").exists()
    assert (out / "artifacts" / "transfer_matrix.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "spec.json").exists()
    assert (out / "RESULT.md").exists()
    # advisory gate is present and never self-promotes.
    assert gate["promote_recommended"] is False
    for key in ("val_improves_or_ties", "blind_tail_not_worse", "new_contract_failures",
                "no_transfer_regression"):
        assert key in gate
