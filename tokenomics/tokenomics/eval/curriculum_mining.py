"""Stage 3 random hard-case mining (CURRICULUM_GAME_MASTER_APPROACH.md Stage 3).

Find useful *policy weaknesses* without a learned generator: sample a large legal pool,
run the portable stack (learner) and the strongest affordable reference on each replayed
world, gate by the failure taxonomy (keep only in-claim `policy_weakness`, drop
feasibility walls / contract / implementation failures / out-of-claim), then rank the
survivors by regret + novelty and archive the top cells.

Stage 2's dedicated random world generator is not built yet, so the pool is sampled over
the *existing* vetted generator-ref families (`REPRODUCERS`) with wide seed / param
sweeps — a legal, replayable broad sample, not a new inline `world_spec`. The learner is
injected (a `(cfg, seed, horizon) -> stats` callable) so the ranking logic is testable
without torch; the driver wires the real portable stack. The reference is a single
checkpoint-free bar (default the `mpc_search` short-rollout bar), the "strongest
affordable reference method" the doc asks for.

Design note (doc): everything here is deterministic-given-seed, so a mined ranking
reproduces exactly on re-run (the doc's "blind subset reproduces the original ranking").
Because a narrow gap on one seed can be search-sampling noise (EXP-0024 caveat), a task
is only kept when the reference beats the learner on a *paired* fraction of seeds
(`min_win_frac`) AND by an absolute unit floor (`min_gap`), not just a mean tie-break.
"""

from __future__ import annotations

from typing import Callable

from ..domain.config import Configuration
from .curriculum import (
    REFERENCE_BARS, REPRODUCERS, ReplayOutcome, TaskRecord,
    _chain_depth, _dims, classify_failure, replay, write_archive, write_matrix_csv,
)

# A learner runner: replayed economy + episode seed + horizon -> episode_stats dict
# (same shape `run_joint_stack` / a baseline `run_episode` returns). Injected so the
# mining/ranking logic is unit-testable without loading torch checkpoints.
LearnerRun = Callable[[Configuration, int, int], dict]

_MINE_GENERATOR_VERSION = "mine-v1"


def mining_pool(generator: str, seeds: "range | list[int]", params: dict, *,
                boundary: str = "in_claim", task_family: str = "random",
                id_prefix: str = "mine") -> list[TaskRecord]:
    """A legal, replayable pool sampled over one vetted generator-ref family + seed sweep.

    `generator` must be a whitelisted `REPRODUCERS` builder; `params` its non-seed args
    (e.g. `{"depth": 3}`). Each seed becomes a self-describing `TaskRecord` (world_seed ==
    episode_seed, so the row replays the exact condition it serialized). `boundary` labels
    whether the sampled family is inside the promoted claim — only in-claim tasks can score
    as `policy_weakness`, so mining for real weaknesses samples the in-claim family."""
    if generator not in REPRODUCERS:
        raise KeyError(f"unknown generator {generator!r} (not a vetted reproducer)")
    pstr = "-".join(f"{k}{params[k]}" for k in sorted(params))
    pool: list[TaskRecord] = []
    for s in seeds:
        pool.append(TaskRecord(
            task_id=f"{id_prefix}-{generator}-{pstr}-s{s}",
            task_family=task_family,
            generator_version=_MINE_GENERATOR_VERSION,
            serializer_version="v1",
            world_seed=s,
            episode_seed=s,
            world_spec={"generator": generator, "params": params},
            expected_claim_boundary=boundary,
        ))
    return pool


def shard_seeds(lo: int, hi: int, workers: int) -> list[tuple[int, int]]:
    """Split the inclusive seed range [lo, hi] into ≤`workers` contiguous (lo, hi) shards.

    Contiguous (not round-robin) so each shard is a plain `--seeds LO-HI` a child process
    re-parses; no empty shards when there are fewer seeds than workers."""
    n = hi - lo + 1
    workers = max(1, min(workers, n))
    base, extra = divmod(n, workers)
    shards: list[tuple[int, int]] = []
    start = lo
    for i in range(workers):
        size = base + (1 if i < extra else 0)
        shards.append((start, start + size - 1))
        start += size
    return shards


def write_rows_jsonl(rows: list[dict], path) -> None:
    """Persist evaluated matrix rows as jsonl (one shard's output for the parent to merge)."""
    import json as _json
    from pathlib import Path
    with Path(path).open("w") as fh:
        for r in rows:
            fh.write(_json.dumps(r) + "\n")


def read_rows_jsonl(path) -> list[dict]:
    """Read evaluated rows back; `descriptor` round-trips as a list, cast to tuple on use."""
    import json as _json
    from pathlib import Path
    with Path(path).open() as fh:
        return [_json.loads(line) for line in fh if line.strip()]


# --- ranking primitives --------------------------------------------------------------
# The doc's rank (surviving tasks only): score = bandpass(gap) * (alpha*normalized_gap +
# beta*novelty + ...). These are the pure pieces; `mine_hard_cases` composes them.

Descriptor = tuple[int, int, int, int]  # (depth, products, producers, capacity_types)


def normalized_gap(reference_units: float, learner_units: float, denom: float) -> float:
    """Regret normalized by requested/feasible units (doc default), denom floored at 1.

    `denom` is the feasible-or-requested upper bound (completed + rejected units of the
    reference run); flooring at 1 keeps a tiny-demand world from inflating the ratio."""
    return (reference_units - learner_units) / max(1.0, denom)


def novelty(desc: Descriptor, archive_descs: "list[Descriptor] | tuple") -> float:
    """Bounded [0,1) novelty of a task descriptor vs the already-archived cells.

    Empty archive -> 1.0 (fully novel). Otherwise `d/(1+d)` where `d` is the L1 distance
    to the nearest archived descriptor: an exact cell is 0, farther cells approach 1. Used
    with `beta` so mining doesn't collapse the archive onto one topology cell."""
    if not archive_descs:
        return 1.0
    d = min(sum(abs(a - b) for a, b in zip(desc, other)) for other in archive_descs)
    return d / (1.0 + d)


def passes_gates(row: dict, *, min_gap: float, min_win_frac: float) -> bool:
    """Hard gate before ranking: keep only a *reproduced, in-claim* policy weakness.

    The failure taxonomy already folds in legality, contract/implementation health,
    feasibility walls, and claim boundary — so `policy_weakness` is the one class worth
    training on (never reward raw failure, out-of-claim gaps, or walls). On top of the
    class, two anti-noise floors (EXP-0024 caveat): a mean unit gap `>= min_gap` and a
    paired per-seed win fraction `>= min_win_frac`, so a 2-unit one-seed fluke from search
    sampling is dropped rather than mined."""
    return (
        row["contract_ok"]
        and not row["crashed"]
        and row["failure_class"] == "policy_weakness"
        and row["gap_units"] >= min_gap
        and row["win_frac"] >= min_win_frac
    )


def _descriptor(cfg: Configuration) -> Descriptor:
    products, producers, caps = _dims(cfg)
    return (_chain_depth(cfg), products, producers, caps)


def _evaluate(record: TaskRecord, learner_run: LearnerRun, reference_bar: str,
              eval_seeds: int, horizon: int) -> dict:
    """Replay one task under the injected learner + the reference bar on the SAME seeds.

    Returns a matrix row with paired-seed telemetry (`win_frac`), mean regret
    (`gap_units`, `normalized_gap`) and the taxonomy `failure_class`. A learner exception
    is caught as a `contract_failure` (never propagated), so one bad checkpoint cell can't
    sink a large mining pool."""
    cfg = replay(record)
    desc = _descriptor(cfg)
    bar = REFERENCE_BARS[reference_bar]
    seeds = list(range(record.episode_seed, record.episode_seed + eval_seeds))

    ref_u = ref_rej = ref_dead = 0.0
    ref_units: list[float] = []
    for es in seeds:
        rs = bar(cfg, es, horizon)
        ref_units.append(rs["consumer_completed_units"])
        ref_u += rs["consumer_completed_units"]
        ref_rej += rs["consumer_rejected_units"]
        ref_dead += float(rs["dead"])
    n = len(seeds)
    ref_u /= n; ref_rej /= n

    contract_ok, crashed = True, False
    learner_u = learner_dead = 0.0
    wins = 0
    notes = record.notes
    try:
        lu = ld = 0.0
        for i, es in enumerate(seeds):
            ls = learner_run(cfg, es, horizon)
            u = ls["consumer_completed_units"]
            if u != u:  # NaN -> implementation failure, not contract
                crashed = True
            lu += u; ld += float(ls["dead"])
            if ref_units[i] > u:
                wins += 1
        learner_u = lu / n; learner_dead = ld / n
    except Exception as e:  # noqa: BLE001 — learner raise = contract failure per taxonomy
        contract_ok = False
        notes = f"contract: {type(e).__name__}: {e}"

    outcome = ReplayOutcome(
        learner_units=None if not contract_ok else round(learner_u, 2),
        reference_units=round(ref_u, 2),
        learner_dead=bool(learner_dead),
        reference_dead=bool(ref_dead),
        contract_ok=contract_ok,
        crashed=crashed,
    )
    failure_class = classify_failure(outcome, boundary=record.expected_claim_boundary)
    gap_units = (ref_u - learner_u) if contract_ok else 0.0
    denom = ref_u + ref_rej  # requested/feasible proxy (reference throughput)

    return {
        "task_id": record.task_id,
        "task_family": record.task_family,
        "expected_claim_boundary": record.expected_claim_boundary,
        "descriptor": desc,
        "target_depth": desc[0],
        "target_products": desc[1],
        "target_producers": desc[2],
        "target_capacity_types": desc[3],
        "learner_units": outcome.learner_units,
        "learner_dead": outcome.learner_dead,
        "reference_model_id": reference_bar,
        "reference_units": outcome.reference_units,
        "reference_dead": outcome.reference_dead,
        "gap_units": round(gap_units, 3),
        "win_frac": round(wins / n, 3) if contract_ok else 0.0,
        "normalized_gap": round(normalized_gap(ref_u, learner_u, denom), 4) if contract_ok else 0.0,
        "contract_ok": contract_ok,
        "crashed": crashed,
        "failure_class": failure_class,
        "notes": notes,
    }


def mine_hard_cases(pool: list[TaskRecord], learner_run: LearnerRun, *,
                    eval_seeds: int = 1, horizon: int = 100,
                    reference_bar: str = "entity_planner+mpc_search",
                    alpha: float = 1.0, beta: float = 0.5,
                    min_gap: float = 2.0, min_win_frac: float = 0.6,
                    top_k: "int | None" = None,
                    archive_descriptors: "list[Descriptor] | tuple" = (),
                    log: Callable[[str], None] = lambda *_: None,
                    ) -> tuple[list[dict], list[TaskRecord]]:
    """Mine a legal pool for reproduced in-claim policy weaknesses (doc Stage 3).

    Evaluate every task (learner vs `reference_bar` on paired seeds), gate to real
    in-claim `policy_weakness` above the noise floors, then greedily select the archive by
    `alpha*normalized_gap + beta*novelty` — novelty measured against the growing archive
    (seeded with `archive_descriptors`) so the archive spreads across topology cells
    instead of collapsing onto one. Returns (all rows, ranked archived TaskRecords).

    Deterministic given seeds, so the ranking reproduces on re-run (doc's blind-subset
    check). No promotion, no training — this only curates tasks."""
    rows = evaluate_pool(pool, learner_run, eval_seeds=eval_seeds, horizon=horizon,
                         reference_bar=reference_bar, log=log)
    return rank_and_archive(rows, pool, alpha=alpha, beta=beta, min_gap=min_gap,
                            min_win_frac=min_win_frac, top_k=top_k,
                            archive_descriptors=archive_descriptors)


def evaluate_pool(pool: list[TaskRecord], learner_run: LearnerRun, *,
                  eval_seeds: int = 1, horizon: int = 100,
                  reference_bar: str = "entity_planner+mpc_search",
                  log: Callable[[str], None] = lambda *_: None) -> list[dict]:
    """Evaluate every task (learner vs `reference_bar` on paired seeds) → matrix rows.

    The parallelizable half of mining: each row is independent, so a shard of the pool can
    be evaluated in its own process (`--workers` fan-out) and the rows merged. Ranking is
    deliberately NOT done here — novelty is global across the whole pool, so selection lives
    in `rank_and_archive` on the merged rows."""
    if reference_bar not in REFERENCE_BARS:
        raise KeyError(f"unknown reference bar {reference_bar!r}; "
                       f"available: {','.join(REFERENCE_BARS)}")
    rows = []
    for rec in pool:
        r = _evaluate(rec, learner_run, reference_bar, eval_seeds, horizon)
        rows.append(r)
        log(f"{r['task_id']}: learner={r['learner_units']} ref={r['reference_units']} "
            f"gap={r['gap_units']} win={r['win_frac']} class={r['failure_class']}")
    return rows


def rank_and_archive(rows: list[dict], pool: list[TaskRecord], *,
                     alpha: float = 1.0, beta: float = 0.5,
                     min_gap: float = 2.0, min_win_frac: float = 0.6,
                     top_k: "int | None" = None,
                     archive_descriptors: "list[Descriptor] | tuple" = (),
                     ) -> tuple[list[dict], list[TaskRecord]]:
    """Gate rows to reproduced in-claim policy weaknesses, then greedily select the archive.

    Global selection over ALL rows (so a `--workers` parallel run ranks identically to a
    sequential one): repeatedly take the highest `alpha*normalized_gap + beta*novelty`
    survivor, then treat its cell as archived so the next pick prefers a fresh topology
    cell. Returns (rows, ranked archived TaskRecords)."""
    by_id = {rec.task_id: rec for rec in pool}
    survivors = [r for r in rows if passes_gates(r, min_gap=min_gap, min_win_frac=min_win_frac)]

    archive_descs = list(archive_descriptors)
    archived: list[TaskRecord] = []
    remaining = survivors[:]
    while remaining and (top_k is None or len(archived) < top_k):
        def score(r: dict) -> float:
            return alpha * r["normalized_gap"] + beta * novelty(tuple(r["descriptor"]), archive_descs)
        best = max(remaining, key=score)
        best["mining_score"] = round(score(best), 4)
        best["novelty"] = round(novelty(tuple(best["descriptor"]), archive_descs), 4)
        rec = by_id[best["task_id"]]
        archived.append(TaskRecord(**{**rec.__dict__,
                                      "failure_class": "policy_weakness",
                                      "reference_status": "solved",
                                      "notes": f"mined: score={best['mining_score']} "
                                               f"gap={best['gap_units']}u win={best['win_frac']}"}))
        archive_descs.append(tuple(best["descriptor"]))
        remaining.remove(best)

    return rows, archived


def _class_histogram(rows: list[dict]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for r in rows:
        hist[r["failure_class"]] = hist.get(r["failure_class"], 0) + 1
    return hist


def run_mining(pool: list[TaskRecord], learner_run: LearnerRun, out_dir, *,
               eval_seeds: int = 1, horizon: int = 100,
               reference_bar: str = "entity_planner+mpc_search",
               alpha: float = 1.0, beta: float = 0.5,
               min_gap: float = 2.0, min_win_frac: float = 0.6,
               top_k: "int | None" = None,
               log: Callable[[str], None] = lambda *_: None,
               ) -> tuple[list[dict], list[TaskRecord]]:
    """Mine `pool` and emit the EXP-NNNN curriculum layout (doc Experiment Record).

    Writes `artifacts/task_archive.jsonl` (the mined, replayable high-regret tasks),
    `artifacts/curriculum_matrix.csv` + `summary.csv` (every evaluated row with per-seed
    win frac, regret, novelty, class). Returns (rows, archived) for the caller to report
    diversity / histogram in RESULT.md. No promotion, no training."""
    from pathlib import Path

    rows, archived = mine_hard_cases(
        pool, learner_run, eval_seeds=eval_seeds, horizon=horizon,
        reference_bar=reference_bar, alpha=alpha, beta=beta,
        min_gap=min_gap, min_win_frac=min_win_frac, top_k=top_k, log=log)

    out = Path(out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    write_archive(archived, out / "artifacts" / "task_archive.jsonl")
    # `descriptor` (a tuple) is redundant with the target_* columns and ugly in CSV.
    csv_rows = [{k: v for k, v in r.items() if k != "descriptor"} for r in rows]
    write_matrix_csv(csv_rows, out / "artifacts" / "curriculum_matrix.csv")
    write_matrix_csv(csv_rows, out / "summary.csv")
    return rows, archived


def _joint_stack_learner(producer_model: object, orderer_model: object) -> LearnerRun:
    """Wrap the portable stack as a `LearnerRun` for the driver (imported lazily = no torch
    in the pure mining/ranking path the tests exercise)."""
    from .joint_stack import run_joint_stack

    def run(cfg: Configuration, seed: int, horizon: int) -> dict:
        return run_joint_stack(producer_model, orderer_model, cfg, seed=seed, horizon=horizon)

    return run


def _mine_parallel(args: "argparse.Namespace", lo: int, hi: int) -> list[dict]:
    """Fan the seed range out across `args.workers` child processes, one per shard, then
    merge their evaluated rows. Mirrors `tokenomics.train.parallel`: mining per task is
    single-threaded, so throughput comes from independent processes, one per core, each with
    BLAS pinned to 1 thread (no oversubscription). The parent ranks the merged rows."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    shards = shard_seeds(lo, hi, args.workers)
    out = Path(args.out_dir)
    logs = out / "artifacts" / "logs"
    rowdir = out / "artifacts" / "shard_rows"
    logs.mkdir(parents=True, exist_ok=True)
    rowdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    procs = []
    for i, (slo, shi) in enumerate(shards):
        rowpath = rowdir / f"shard_{i}_{slo}-{shi}.jsonl"
        cmd = [sys.executable, "-u", "-m", "tokenomics.eval.curriculum_mining",
               "--producer-ckpt", args.producer_ckpt, "--orderer-ckpt", args.orderer_ckpt,
               "--generator", args.generator, "--depth", str(args.depth),
               "--boundary", args.boundary, "--reference-bar", args.reference_bar,
               "--eval-seeds", str(args.eval_seeds), "--horizon", str(args.horizon),
               "--seeds", f"{slo}-{shi}", "--emit-rows", str(rowpath), "--workers", "1"]
        lf = (logs / f"shard_{i}.log").open("w")
        print(f"[parallel] worker {i}: seeds {slo}-{shi} -> {rowpath.name} (log {lf.name})")
        procs.append((subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env),
                      lf, rowpath))

    rows: list[dict] = []
    failed = []
    for i, (p, lf, rowpath) in enumerate(procs):  # wait on ALL before deciding (no zombies)
        rc = p.wait()
        lf.close()
        if rc != 0 or not rowpath.exists():
            failed.append((i, rc))
            print(f"[parallel] worker {i} exit {rc}; see {lf.name}")
            continue
        rows += read_rows_jsonl(rowpath)
    if failed:
        # Any shard missing its seeds means the pool is incomplete: ranking the survivors
        # would silently emit a BIASED archive under the full-pool name. Abort instead so a
        # transient crash/OOM never masquerades as a complete mining result — inspect the
        # shard logs and re-run.
        raise RuntimeError(
            f"{len(failed)} of {len(procs)} mining shards failed (worker,exit)={failed}; "
            f"the archive would cover only {len(rows)} of the requested tasks — aborting. "
            f"See shard logs under {logs}/ and re-run.")
    return rows


def main() -> None:
    """Stage 3 mining run (cloud per CLAUDE.md HARD RULE; local only for --dry-run/tiny).

    Sample a legal pool over a vetted generator-ref family + seed sweep, run the portable
    stack vs the strongest affordable reference bar, gate to reproduced in-claim policy
    weaknesses, rank by regret + novelty, and emit the EXP layout."""
    import argparse
    from pathlib import Path

    from ..train.orderer_scorer import load_scorer
    from ..train.producer_scorer import load_producer_score_model

    ap = argparse.ArgumentParser(description="Curriculum Stage 3: random hard-case mining")
    ap.add_argument("--producer-ckpt", required=True)
    ap.add_argument("--orderer-ckpt", required=True)
    ap.add_argument("--out-dir", default=None, help="experiment dir (writes artifacts/ + summary.csv)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (fan out the pool across cores; mining "
                         "per task is single-threaded, so throughput = one process per core)")
    ap.add_argument("--emit-rows", default=None,
                    help="INTERNAL child mode: evaluate this seed shard and write rows jsonl "
                         "here (no ranking, no EXP dir); the parent merges + ranks globally")
    ap.add_argument("--generator", default="build_scale_config",
                    help=f"vetted reproducer to sample (available: {','.join(REPRODUCERS)})")
    ap.add_argument("--depth", type=int, default=3, help="chain depth param for the generator")
    ap.add_argument("--seeds", default="20-119", help="inclusive seed range LO-HI for the pool")
    ap.add_argument("--boundary", default="in_claim",
                    choices=("in_claim", "out_of_claim", "feasibility_wall"))
    ap.add_argument("--reference-bar", default="entity_planner+mpc_search",
                    help=f"strongest affordable reference (available: {','.join(REFERENCE_BARS)})")
    ap.add_argument("--eval-seeds", type=int, default=3, help="paired seeds per task (win-frac)")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.0, help="regret weight")
    ap.add_argument("--beta", type=float, default=0.5, help="novelty weight")
    ap.add_argument("--min-gap", type=float, default=2.0, help="min mean unit gap (noise floor)")
    ap.add_argument("--min-win-frac", type=float, default=0.6, help="min paired-seed win frac")
    ap.add_argument("--top-k", type=int, default=None, help="cap the archived count")
    ap.add_argument("--dry-run", action="store_true", help="print the pool + config, don't run")
    args = ap.parse_args()

    if args.generator not in REPRODUCERS:
        ap.error(f"unknown generator {args.generator!r}; available: {','.join(REPRODUCERS)}")
    if args.reference_bar not in REFERENCE_BARS:
        ap.error(f"unknown reference bar {args.reference_bar!r}; available: {','.join(REFERENCE_BARS)}")
    lo, hi = (int(x) for x in args.seeds.split("-"))
    pool = mining_pool(args.generator, range(lo, hi + 1), {"depth": args.depth},
                       boundary=args.boundary)

    if args.dry_run:
        shards = shard_seeds(lo, hi, args.workers)
        print(f"[dry-run] {len(pool)} tasks {args.generator} depth={args.depth} "
              f"seeds={lo}-{hi} ref={args.reference_bar} eval_seeds={args.eval_seeds} "
              f"workers={args.workers} shards={shards}")
        for r in pool[:5]:
            print(f"  {r.task_id}")
        return

    prod = load_producer_score_model(args.producer_ckpt)
    order = load_scorer(args.orderer_ckpt)
    learner = _joint_stack_learner(prod, order)

    # child mode: just evaluate this shard's pool and persist rows for the parent to merge.
    if args.emit_rows:
        rows = evaluate_pool(pool, learner, eval_seeds=args.eval_seeds, horizon=args.horizon,
                             reference_bar=args.reference_bar, log=print)
        write_rows_jsonl(rows, args.emit_rows)
        print(f"emitted {len(rows)} rows -> {args.emit_rows}")
        return

    if not args.out_dir:
        ap.error("--out-dir is required (except in --emit-rows child mode)")

    if args.workers > 1:
        rows = _mine_parallel(args, lo, hi)
    else:
        rows = evaluate_pool(pool, learner, eval_seeds=args.eval_seeds, horizon=args.horizon,
                             reference_bar=args.reference_bar, log=print)

    _, archived = rank_and_archive(rows, pool, alpha=args.alpha, beta=args.beta,
                                   min_gap=args.min_gap, min_win_frac=args.min_win_frac,
                                   top_k=args.top_k)
    out = Path(args.out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    write_archive(archived, out / "artifacts" / "task_archive.jsonl")
    csv_rows = [{k: v for k, v in r.items() if k != "descriptor"} for r in rows]
    write_matrix_csv(csv_rows, out / "artifacts" / "curriculum_matrix.csv")
    write_matrix_csv(csv_rows, out / "summary.csv")

    cells = {tuple(r["descriptor"]) for r in rows}
    arch_cells = {(r.world_spec["params"].get("depth"),) for r in archived}
    print(f"\n{len(pool)} sampled; failure classes: {_class_histogram(rows)}")
    print(f"archived {len(archived)} policy_weakness tasks; "
          f"pool topology cells={len(cells)}; archived depth cells={len(arch_cells)}")


if __name__ == "__main__":
    main()
