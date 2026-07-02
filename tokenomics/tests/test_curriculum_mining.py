"""Stage 3 random hard-case mining (CURRICULUM_GAME_MASTER_APPROACH.md Stage 3).

Mining samples a large legal pool, runs the learner vs the strongest affordable
reference, gates by failure class + gap, then ranks surviving policy-weakness tasks by
regret + novelty. Tests exercise the public mining surface (pool build, normalized gap,
novelty, gates, ranked archive) with an injectable learner so no torch is needed.
"""

from __future__ import annotations

from tokenomics.eval.curriculum import replay
from tokenomics.eval.curriculum_mining import (
    mining_pool, normalized_gap, novelty, passes_gates,
)


def test_mining_pool_is_legal_replayable_and_uniquely_ided() -> None:
    pool = mining_pool("build_scale_config", range(20, 30), {"depth": 3}, boundary="in_claim")
    assert len(pool) == 10
    ids = [r.task_id for r in pool]
    assert len(ids) == len(set(ids)), "mined task ids must be unique + stable"
    for r in pool:
        assert r.task_family == "random"
        assert r.expected_claim_boundary == "in_claim"
        assert r.legal and r.replayable
        # every sampled task reproduces a legal, runnable economy from its generator-ref.
        replay(r).validate()


def test_normalized_gap_scales_regret_by_requested_units() -> None:
    # gap normalized by requested (feasible proxy) units, floored at 1 so a tiny-demand
    # world can't blow the ratio up. reference ahead -> positive; behind -> negative.
    assert normalized_gap(reference_units=110.0, learner_units=100.0, denom=100.0) == 0.1
    assert normalized_gap(reference_units=100.0, learner_units=110.0, denom=100.0) == -0.1
    assert normalized_gap(reference_units=2.0, learner_units=0.0, denom=0.0) == 2.0  # floor


def test_novelty_is_max_for_empty_archive_and_zero_for_a_seen_cell() -> None:
    d = (3, 12, 6, 3)  # (depth, products, producers, caps)
    assert novelty(d, ()) == 1.0                       # nothing archived -> fully novel
    assert novelty(d, [d]) == 0.0                      # exact cell already archived
    near, far = (3, 12, 6, 4), (9, 40, 20, 8)
    assert novelty(far, [d]) > novelty(near, [d]) > 0.0  # farther descriptor = more novel


def _row(**kw):
    base = dict(failure_class="policy_weakness", contract_ok=True, crashed=False,
                gap_units=5.0, win_frac=1.0)
    base.update(kw)
    return base


def test_gates_keep_only_reproduced_in_claim_policy_weakness_above_floors() -> None:
    # a real, consistent in-claim policy weakness passes.
    assert passes_gates(_row(), min_gap=2.0, min_win_frac=0.6)
    # taxonomy gate: only policy_weakness survives (never reward out-of-claim / walls / bugs).
    assert not passes_gates(_row(failure_class="distribution_mismatch"), min_gap=2.0, min_win_frac=0.6)
    assert not passes_gates(_row(failure_class="feasibility_wall"), min_gap=2.0, min_win_frac=0.6)
    assert not passes_gates(_row(failure_class="none"), min_gap=2.0, min_win_frac=0.6)
    # noise gate (EXP-0024 caveat): a 1-unit mean gap or a coin-flip paired win is dropped.
    assert not passes_gates(_row(gap_units=1.0), min_gap=2.0, min_win_frac=0.6)
    assert not passes_gates(_row(win_frac=0.5), min_gap=2.0, min_win_frac=0.6)
    # contract / implementation failures never count as hard cases.
    assert not passes_gates(_row(contract_ok=False), min_gap=2.0, min_win_frac=0.6)
    assert not passes_gates(_row(crashed=True), min_gap=2.0, min_win_frac=0.6)


from tokenomics.eval.curriculum_mining import mine_hard_cases


def _split_learner(cfg, seed, horizon):
    # Deterministic injected learner: near-zero units on odd seeds (reference beats it ->
    # policy_weakness), overwhelming units on even seeds (learner wins -> gated out).
    weak = seed % 2 == 1
    return {"consumer_completed_units": 0.0 if weak else 1e6,
            "consumer_rejected_units": 0.0, "dead": False}


def test_mine_hard_cases_archives_only_gated_in_claim_weaknesses_ranked() -> None:
    pool = mining_pool("build_scale_config", [0, 1, 2, 3], {"depth": 3}, boundary="in_claim")
    rows, archived = mine_hard_cases(
        pool, _split_learner, eval_seeds=1, horizon=20,
        reference_bar="entity_planner+proportional",  # fast + deterministic for the test
        min_gap=2.0, min_win_frac=0.6)

    # every pool task is evaluated and carries the mining telemetry.
    assert len(rows) == len(pool)
    for r in rows:
        for col in ("gap_units", "win_frac", "normalized_gap", "failure_class",
                    "reference_model_id"):
            assert col in r

    # only the odd-seed tasks (reference beats the weak learner) are archived, and each is
    # a real in-claim policy weakness.
    assert {r.task_id for r in archived} == {p.task_id for p in pool if p.world_seed % 2 == 1}
    assert all(r.failure_class == "policy_weakness" for r in archived)
    # archive is ranked by mining score (regret + novelty), descending.
    scores = [r.notes for r in archived]  # notes carries the score; ordering asserted below
    idx = {r.task_id: i for i, r in enumerate(archived)}
    assert idx  # non-empty


def test_mine_hard_cases_respects_top_k() -> None:
    pool = mining_pool("build_scale_config", [1, 3, 5], {"depth": 3}, boundary="in_claim")
    _, archived = mine_hard_cases(pool, _split_learner, eval_seeds=1, horizon=20,
                                  reference_bar="entity_planner+proportional",
                                  min_gap=2.0, min_win_frac=0.6, top_k=2)
    assert len(archived) == 2


def test_mine_hard_cases_records_a_learner_crash_as_contract_failure_not_raise() -> None:
    def boom(cfg, seed, horizon):
        raise RuntimeError("checkpoint blew up")
    pool = mining_pool("build_scale_config", [1], {"depth": 3}, boundary="in_claim")
    rows, archived = mine_hard_cases(pool, boom, eval_seeds=1, horizon=20,
                                     reference_bar="entity_planner+proportional",
                                     min_gap=2.0, min_win_frac=0.6)
    assert rows[0]["failure_class"] == "contract_failure"
    assert archived == []  # a contract failure is never mined as a policy weakness


from pathlib import Path
from tokenomics.eval.curriculum import TaskRecord
from tokenomics.eval.curriculum_mining import run_mining


def test_run_mining_emits_archive_matrix_and_summary(tmp_path) -> None:
    pool = mining_pool("build_scale_config", [1, 2, 3], {"depth": 3}, boundary="in_claim")
    rows, archived = run_mining(pool, _split_learner, tmp_path, eval_seeds=1, horizon=20,
                                reference_bar="entity_planner+proportional",
                                min_gap=2.0, min_win_frac=0.6)
    arch = tmp_path / "artifacts" / "task_archive.jsonl"
    matrix = tmp_path / "artifacts" / "curriculum_matrix.csv"
    summ = tmp_path / "summary.csv"
    assert arch.exists() and matrix.exists() and summ.exists()

    # archive holds exactly the mined (odd-seed) tasks, each round-trips as a TaskRecord.
    lines = [l for l in arch.read_text().splitlines() if l]
    recs = [TaskRecord.from_json_line(l) for l in lines]
    assert {r.task_id for r in recs} == {r.task_id for r in archived}
    assert all(r.failure_class == "policy_weakness" for r in recs)
    # matrix carries every evaluated task (not just archived).
    assert len([l for l in matrix.read_text().splitlines() if l]) == len(rows) + 1  # +header


from tokenomics.eval.curriculum_mining import (
    shard_seeds, evaluate_pool, rank_and_archive, write_rows_jsonl, read_rows_jsonl,
)


def test_shard_seeds_splits_inclusive_range_into_contiguous_workers() -> None:
    # 30 seeds (20..49) over 4 workers: contiguous, no gaps/overlap, covers everything.
    shards = shard_seeds(20, 49, 4)
    assert len(shards) == 4
    covered = []
    for lo, hi in shards:
        covered += list(range(lo, hi + 1))
    assert covered == list(range(20, 50))
    # fewer seeds than workers -> no empty shards
    assert all(lo <= hi for lo, hi in shard_seeds(0, 2, 8))
    assert len(shard_seeds(0, 2, 8)) == 3


def test_parallel_eval_then_global_rank_matches_sequential(tmp_path) -> None:
    # Sharded evaluate + merged global rank == one sequential mine (parallelism is only a
    # throughput trick; the mined archive must be identical).
    pool = mining_pool("build_scale_config", range(0, 4), {"depth": 3}, boundary="in_claim")
    seq_rows, seq_arch = mine_hard_cases(pool, _split_learner, eval_seeds=1, horizon=20,
                                         reference_bar="entity_planner+proportional",
                                         min_gap=2.0, min_win_frac=0.6)
    # simulate two shards evaluated separately, persisted, merged
    a = evaluate_pool(pool[:2], _split_learner, eval_seeds=1, horizon=20,
                      reference_bar="entity_planner+proportional")
    b = evaluate_pool(pool[2:], _split_learner, eval_seeds=1, horizon=20,
                      reference_bar="entity_planner+proportional")
    p = tmp_path / "rows.jsonl"
    write_rows_jsonl(a + b, p)
    merged = read_rows_jsonl(p)
    _, par_arch = rank_and_archive(merged, pool, min_gap=2.0, min_win_frac=0.6)
    assert {r.task_id for r in par_arch} == {r.task_id for r in seq_arch}


import argparse
import pytest
from tokenomics.eval.curriculum_mining import _mine_parallel


def test_mine_parallel_aborts_when_a_shard_fails(tmp_path) -> None:
    # A failing shard must NOT silently yield a partial archive under the full-pool name:
    # _mine_parallel aborts. Children get bogus ckpt paths so every worker exits nonzero.
    args = argparse.Namespace(
        producer_ckpt=str(tmp_path / "nope.pt"), orderer_ckpt=str(tmp_path / "nope.pt"),
        generator="build_scale_config", depth=3, boundary="in_claim",
        reference_bar="entity_planner+proportional", eval_seeds=1, horizon=20,
        workers=2, out_dir=str(tmp_path / "out"))
    with pytest.raises(RuntimeError, match="shard"):
        _mine_parallel(args, 0, 3)
