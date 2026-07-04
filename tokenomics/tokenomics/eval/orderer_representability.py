"""Orderer oracle-relabel representability smoke (EXP-0032 v4, step 1).

Answers the standing EXP-0026 question BEFORE spending a pod run: *can the portable orderer
scorer head represent the mining bar's per-cycle search routing at all?* The Stage-4 orderer
distilled only from the `proportional` throughput teacher and could never absorb the
search-oracle headroom the mined gap used; v4 relabels the orderer from the bar's own search
(`collect_oracle_orderer_samples`). If a fresh scorer distilled on those oracle targets cannot
even reproduce the search on the SAME task it was trained on, the limit is architectural (no
run needed); if it recovers a large share, the pod run is green-lit.

Per task, with the producer held at the champion (deployment-matched), it reports four on-task
consumer-unit numbers:
  - `champ_cu`  : champion producer + champion orderer          (the stack being improved)
  - `search_cu` : champion producer + the bar's search orderer  (the teacher on-task = ceiling)
  - `oracle_cu` : champion producer + oracle-relabeled orderer  (the distilled candidate)
  - `bar_cu`    : entity_planner + mpc_search (the archived reference gap was measured against)
and two recovery ratios:
  - `recovered_vs_teacher = (oracle_cu - champ_cu) / (search_cu - champ_cu)`  — the
    representability answer (can the head reproduce the search it was trained on?),
  - `recovered_vs_bar     = (oracle_cu - champ_cu) / (bar_cu   - champ_cu)`   — recovery of the
    archived gap (bounded below 1 by the producer-side share entity_planner keeps).

CLOUD run (CLAUDE.md HARD RULE): this is a **decision-gating verification** — full-horizon
per-cycle search, distill to convergence, `eval_seeds > 1`, multiple tasks (~14 heavy search
episodes). It is NOT a local smoke despite touching only 1-2 tasks; classify by compute profile,
not task count. Launch it wrapped with `run_pod_training.py`. The full v4 promotion candidate is
the follow-on pod `run_stage4_replay` step.
"""

from __future__ import annotations

import argparse
import statistics

from .curriculum import TaskRecord, _episode_seeds, replay
from ..train.orderer_scorer import (
    OrdererScoreModel,
    collect_oracle_orderer_samples,
    fit_scorer,
    scorer_ordering_policy,
)

# The bar's search config (mirrors `curriculum._bar_mpc_search` / the collector defaults) so the
# oracle teacher, the on-task search ceiling, and the archived bar all use one search.
_SEARCH = dict(candidates=("even_split", "proportional"), n_samples=8, sample_scale=1.0,
               lookahead=6)


def _mean_units(run) -> float:
    return run["consumer_completed_units"]


def _champ_search_units(cfg, prod_pol, champ_order_model, *, horizon: int, seeds: list[int]
                        ) -> tuple[float, float]:
    """Mean on-task units for the two producer-frozen crosses that bound the orderer headroom:
    `champ_prod + champ_ord` (the deployed stack) and `champ_prod + per-cycle search` (the
    teacher ceiling). Shared by the headroom scan and the representability row so the two never
    drift. `seeds` MUST be the archive-anchored episode seeds (`_episode_seeds`), never
    `range(eval_seeds)` — the mined gap lives at the record's demand trace, and evaluating any
    other seed measures a different task (EXP-0036 probe bug: champ alive at seeds 0-2 while
    the archived gap had it dead at seeds 77-79)."""
    from .joint_stack import joint_stack_env
    from .orderer_oracle import search_orderer_episode
    from .baselines import ORDERING_BASELINES
    from .runner import run_episode

    cand = {n: ORDERING_BASELINES[n] for n in _SEARCH["candidates"]}
    champ_order = scorer_ordering_policy(champ_order_model)
    champ, search = [], []
    for es in seeds:
        champ.append(_mean_units(run_episode(joint_stack_env(cfg, horizon), prod_pol, champ_order, seed=es)))
        stats, _ = search_orderer_episode(joint_stack_env(cfg, horizon), prod_pol, cand,
                                          _SEARCH["n_samples"], _SEARCH["sample_scale"],
                                          _SEARCH["lookahead"], es)
        search.append(stats["consumer_completed_units"])
    return statistics.mean(champ), statistics.mean(search)


def headroom_row(record: TaskRecord, prod_pol, champ_order_model, *, horizon: int,
                 eval_seeds: int, log=lambda *a, **k: None) -> dict:
    """Cheap per-task orderer-headroom probe (no distill/oracle/bar): `teacher_headroom =
    champ_prod+search − champ_prod+champ_ord`, under the FROZEN producer. EXP-0032 showed the
    archived mining `gap`/`score` does not survive the augmented EXP-0030 producer (it already
    closed most of it), so probe tasks must be re-selected by *measured* headroom, not archived
    score. `headroom_frac` normalizes by `champ_cu` for a topology-agnostic floor."""
    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    champ_cu, search_cu = _champ_search_units(cfg, prod_pol, champ_order_model,
                                              horizon=horizon, seeds=seeds)
    head = search_cu - champ_cu
    return {
        "task_id": record.task_id, "world_seed": record.world_seed,
        "champ_cu": round(champ_cu, 2), "search_cu": round(search_cu, 2),
        "teacher_headroom": round(head, 2),
        "headroom_frac": round(head / champ_cu, 4) if champ_cu > 1e-9 else None,
    }


def representability_row(
    record: TaskRecord, prod_pol, champ_order_model, *, iters: int, horizon: int,
    eval_seeds: int, log=lambda *a, **k: None,
) -> dict:
    """One task's representability row: distill an orderer on this task's oracle-relabel
    targets (champion producer drives collection), then eval champion / search-teacher /
    oracle / bar on-task. Producer fixed at champion throughout (isolates the orderer)."""
    from .curriculum import REFERENCE_BARS
    from .joint_stack import joint_stack_env
    from .runner import run_episode

    cfg = replay(record)
    seeds = _episode_seeds(record.episode_seed, eval_seeds)

    # 1. collect oracle-relabel samples on this task AT THE ARCHIVED EPISODE SEEDS (the same
    # demand traces every eval below averages — collect/deploy seed mismatch craters the oracle).
    samples = collect_oracle_orderer_samples(
        [cfg] * len(seeds), seeds, prod=prod_pol, horizon=horizon,
        candidates=_SEARCH["candidates"], n_samples=_SEARCH["n_samples"],
        sample_scale=_SEARCH["sample_scale"], lookahead=_SEARCH["lookahead"], log=log)
    log(f"{record.task_id}: {len(samples)} oracle samples")
    oracle_model = fit_scorer(OrdererScoreModel(), samples, iters=iters, log=log)

    oracle_order = scorer_ordering_policy(oracle_model)
    bar = REFERENCE_BARS["entity_planner+mpc_search"]

    # champ + search (the frozen-producer bounds) share one code path with the headroom scan.
    champ_cu, search_cu = _champ_search_units(cfg, prod_pol, champ_order_model,
                                              horizon=horizon, seeds=seeds)
    oracle, barcu = [], []
    for es in seeds:
        oracle.append(_mean_units(run_episode(joint_stack_env(cfg, horizon), prod_pol, oracle_order, seed=es)))
        barcu.append(_mean_units(bar(cfg, es, horizon)))

    oracle_cu = statistics.mean(oracle)
    bar_cu = statistics.mean(barcu)
    teacher_head = search_cu - champ_cu
    bar_head = bar_cu - champ_cu
    recovered = oracle_cu - champ_cu
    return {
        "task_id": record.task_id, "world_seed": record.world_seed,
        "champ_cu": round(champ_cu, 2), "search_cu": round(search_cu, 2),
        "oracle_cu": round(oracle_cu, 2), "bar_cu": round(bar_cu, 2),
        "teacher_headroom": round(teacher_head, 2), "bar_headroom": round(bar_head, 2),
        "recovered": round(recovered, 2),
        "recovered_vs_teacher": round(recovered / teacher_head, 3) if teacher_head > 1e-9 else None,
        "recovered_vs_bar": round(recovered / bar_head, 3) if bar_head > 1e-9 else None,
        "n_samples": len(samples),
    }


def run_representability_verification(archive_path: str, producer_ckpt: str, orderer_ckpt: str,
                                      task_ids: list[str], *, iters: int, horizon: int,
                                      eval_seeds: int, out_dir=None, log=print) -> list[dict]:
    """Load champion producer + orderer, pick `task_ids` from the mined archive, and emit one
    representability row per task. The producer is held at champion (frozen at EXP-0030).

    This is a **decision-gating verification run** (full-horizon per-cycle search, distill to
    convergence, `eval_seeds > 1`, multi-task) — a cloud run per CLAUDE.md, not a local smoke.
    When `out_dir` is given, writes the EXP layout (`summary.csv`, `spec.json`, `RESULT.md`) so a
    pod run can be pulled back into a fresh `EXP-NNNN` dir."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import load_scorer

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champ_order_model = load_scorer(orderer_ckpt)
    by_id = {r.task_id: r for r in _load_archive(archive_path)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise ValueError(f"task ids not in archive: {missing}")

    rows = []
    for tid in task_ids:
        row = representability_row(by_id[tid], prod_pol, champ_order_model,
                                   iters=iters, horizon=horizon, eval_seeds=eval_seeds, log=log)
        log(f"[repr] {row}")
        rows.append(row)
    if out_dir is not None:
        _emit_representability(out_dir, rows, meta={
            "producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt,
            "task_ids": task_ids, "iters": iters, "horizon": horizon,
            "eval_seeds": eval_seeds, "search": _SEARCH})
    return rows


def _emit_representability(out_dir, rows: list[dict], meta: dict) -> None:
    """Write the EXP layout for a pod pull: `summary.csv` (per-task rows), `spec.json` (run
    knobs), and a `RESULT.md` skeleton whose gate line is the mean `recovered_vs_teacher`
    (>= ~0.5 => the scorer head can represent the search => green-light the full v4 pod run)."""
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    tvals = [r["recovered_vs_teacher"] for r in rows if r["recovered_vs_teacher"] is not None]
    mean_t = statistics.mean(tvals) if tvals else None
    verdict = ("green-light full v4 pod run" if mean_t is not None and mean_t >= 0.5 else
               "architecture limit likely real — escalate (richer stream obs / deploy-time search)")
    lines = [
        "# EXP-0032 v4 orderer oracle-relabel representability", "",
        "Decision gate for v4 (EXP-0032): can the portable orderer scorer head represent the",
        "mining bar's per-cycle search routing? Distill a fresh orderer on each task's",
        "oracle-relabel targets (champion producer drives collection), then compare on-task",
        "champion / search-teacher / oracle / bar consumer units. Producer frozen at champion.", "",
        f"**Gate:** mean `recovered_vs_teacher` = "
        f"{mean_t:.3f} (>= ~0.5 => head represents the search)." if mean_t is not None
        else "**Gate:** no teacher headroom on these tasks (degenerate).", "",
        f"**Verdict:** {verdict}.", "",
        "## Per-task", "", "```json", json.dumps(rows, indent=2), "```", "",
        "## Conclusions", "", "_TODO (human): confirm go/no-go on the full v4 `run_stage4_replay` run._", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def _qualifies(row: dict, floor_frac: float) -> bool:
    """A task is an informative representability probe iff its measured orderer headroom clears
    the floor (a fraction of `champ_cu`). Below the floor the recovery ratio is noise (EXP-0032:
    a 0.33u denominator turned a −10u miss into −31)."""
    f = row.get("headroom_frac")
    return f is not None and f >= floor_frac


def scan_headroom(archive_path: str, producer_ckpt: str, orderer_ckpt: str, *, horizon: int,
                  eval_seeds: int, floor_frac: float = 0.02, out_dir=None, log=print) -> list[dict]:
    """Measure per-task orderer headroom over the WHOLE archive under the frozen producer, rank by
    it, and flag the tasks that clear `floor_frac` — the valid representability probes. If few/none
    clear the floor, the augmented producer has already closed the mined gap and the orderer lever
    is largely spent (STOP, don't escalate). Cloud run (per-cycle search × archive)."""
    from .curriculum_replay import _load_archive
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.orderer_scorer import load_scorer

    prod_pol = scorer_producing_policy(load_producer_score_model(producer_ckpt))
    champ_order_model = load_scorer(orderer_ckpt)
    records = _load_archive(archive_path)
    rows = []
    for rec in records:
        row = headroom_row(rec, prod_pol, champ_order_model, horizon=horizon,
                           eval_seeds=eval_seeds, log=log)
        row["qualifies"] = _qualifies(row, floor_frac)
        log(f"[headroom] {row}")
        rows.append(row)
    rows.sort(key=lambda r: r["teacher_headroom"], reverse=True)
    if out_dir is not None:
        _emit_headroom(out_dir, rows, floor_frac, meta={
            "producer_ckpt": producer_ckpt, "orderer_ckpt": orderer_ckpt,
            "horizon": horizon, "eval_seeds": eval_seeds, "floor_frac": floor_frac,
            "search": _SEARCH})
    return rows


def _emit_headroom(out_dir, rows: list[dict], floor_frac: float, meta: dict) -> None:
    """Write the headroom-scan EXP layout: `summary.csv` (per-task, headroom-sorted), `spec.json`,
    and a `RESULT.md` naming the qualifying probe task_ids (or, if few, the STOP signal)."""
    import json
    from pathlib import Path

    from .curriculum import write_matrix_csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rows, out / "summary.csv")
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")

    qual = [r for r in rows if r.get("qualifies")]
    verdict = (f"{len(qual)}/{len(rows)} tasks clear the {floor_frac:.0%} headroom floor — "
               "re-run representability on these" if qual else
               f"NO task clears the {floor_frac:.0%} floor — the augmented producer already "
               "closed the mined orderer gap; the orderer lever is spent (STOP, don't escalate)")
    lines = [
        "# EXP-0032b orderer-headroom scan (probe re-selection)", "",
        "EXP-0032 was an invalid probe (near-zero headroom). This scan measures `teacher_headroom",
        "= champ_prod+search − champ_prod+champ_ord` over the whole archive under the FROZEN",
        f"EXP-0030 producer, and keeps tasks with `headroom_frac >= {floor_frac:.0%}`.", "",
        f"**Result:** {verdict}.", "",
        "## Qualifying probe task_ids", "",
        ("```\n" + "\n".join(r["task_id"] for r in qual) + "\n```") if qual else "_none_", "",
        "## Per-task (headroom-sorted)", "", "```json", json.dumps(rows, indent=2), "```", "",
        "## Conclusions", "",
        "_TODO (human): if qualifying tasks exist, re-run representability on them; else STOP._", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="headroom-scan mode: measure teacher_headroom over the whole archive and "
                         "flag tasks clearing --floor-frac (probe re-selection); ignores --task-ids")
    ap.add_argument("--floor-frac", type=float, default=0.02,
                    help="with --scan: min headroom_frac (of champ_cu) for a task to qualify")
    ap.add_argument("--archive", required=True, help="mined task_archive.jsonl (Stage 3 output)")
    ap.add_argument("--producer-ckpt", required=True, help="champion producer score model (frozen)")
    ap.add_argument("--orderer-ckpt", required=True, help="champion orderer score model")
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated archived task ids (required unless --scan)")
    ap.add_argument("--iters", type=int, default=300, help="orderer distill iters")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--out-dir", default=None, help="EXP dir to write summary.csv + RESULT.md (pod pull)")
    args = ap.parse_args()

    if args.scan:
        rows = scan_headroom(args.archive, args.producer_ckpt, args.orderer_ckpt,
                             horizon=args.horizon, eval_seeds=args.eval_seeds,
                             floor_frac=args.floor_frac, out_dir=args.out_dir)
        qual = [r for r in rows if r.get("qualifies")]
        print(f"\n{len(qual)}/{len(rows)} tasks clear the {args.floor_frac:.0%} headroom floor")
        for r in qual:
            print(f"  {r['teacher_headroom']:>6} ({r['headroom_frac']:.1%})  {r['task_id']}")
        return

    if not args.task_ids:
        ap.error("--task-ids is required unless --scan")
    rows = run_representability_verification(
        args.archive, args.producer_ckpt, args.orderer_ckpt,
        [t for t in args.task_ids.split(",")], iters=args.iters,
        horizon=args.horizon, eval_seeds=args.eval_seeds, out_dir=args.out_dir)
    print("\ntask_id,champ_cu,search_cu,oracle_cu,bar_cu,recovered_vs_teacher,recovered_vs_bar")
    for r in rows:
        print(f"{r['task_id']},{r['champ_cu']},{r['search_cu']},{r['oracle_cu']},{r['bar_cu']},"
              f"{r['recovered_vs_teacher']},{r['recovered_vs_bar']}")
    tvals = [r["recovered_vs_teacher"] for r in rows if r["recovered_vs_teacher"] is not None]
    if tvals:
        print(f"\nmean recovered_vs_teacher: {statistics.mean(tvals):.3f} "
              f"(>= ~0.5 => head can represent the search; green-light pod run)")


if __name__ == "__main__":
    main()
