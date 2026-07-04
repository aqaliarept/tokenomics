"""Train-split closure diagnostic (EXP-0034 follow-up; CLAUDE.md root-cause rule).

Discriminates WHY a distilled candidate captures little of the mined headroom:
run the candidate stack on the tasks it was TRAINED on and measure gap closure
there. Low closure on train == distill-fidelity/representability wall (the scorer
cannot express the search teacher; fix training or accept the ceiling). High
closure on train but low on val/blind == generalization gap (mine denser; more
Track-A rounds pay). Champion + bar units are NOT re-rolled: they are joined from
the gap_attribution CSV, which used the same paired-seed convention
(episode_seed..episode_seed+eval_seeds-1, same horizon).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
from typing import Callable

from .curriculum import TaskRecord

Log = Callable[[str], None]

_WORKER_PROD = None
_WORKER_ORD = None

# per-task closure is noise-dominated when the mined gap is tiny; aggregate
# closure (unit sums) is the headline number, per-task rows are for digging.
_MIN_GAP_FOR_RATIO = 1.0


def closure_summary(rows: list[dict]) -> dict:
    """Aggregate closure over joined rows: (Σcand − Σchamp) / (Σbar − Σchamp)."""
    n = len(rows)
    cand = sum(r["candidate_units"] for r in rows)
    champ = sum(r["champion_units"] for r in rows)
    bar = sum(r["bar_units"] for r in rows)
    ratios = sorted(r["closure"] for r in rows
                    if r["closure"] is not None)
    return {
        "n_tasks": n,
        "mean_candidate_units": round(cand / n, 2) if n else 0.0,
        "mean_champion_units": round(champ / n, 2) if n else 0.0,
        "mean_bar_units": round(bar / n, 2) if n else 0.0,
        "aggregate_closure": round((cand - champ) / (bar - champ), 4)
                             if bar > champ else None,
        "median_task_closure": round(ratios[len(ratios) // 2], 4) if ratios else None,
    }


def join_closure_rows(evaluated: list[dict], attribution: dict[str, tuple[float, float]],
                      ) -> list[dict]:
    """Join candidate rollout units against the attribution's champion/bar units."""
    rows: list[dict] = []
    for e in evaluated:
        champ, bar = attribution[e["task_id"]]
        gap = bar - champ
        rows.append({
            **e,
            "champion_units": champ,
            "bar_units": bar,
            "gap_units": round(gap, 2),
            "candidate_minus_champion": round(e["candidate_units"] - champ, 2),
            "closure": round((e["candidate_units"] - champ) / gap, 4)
                       if gap >= _MIN_GAP_FOR_RATIO else None,
        })
    return rows


def read_attribution_units(path: str | Path) -> dict[str, tuple[float, float]]:
    with Path(path).open() as fh:
        return {r["task_id"]: (float(r["champion_units"]), float(r["full_bar_units"]))
                for r in csv.DictReader(fh)}


def _init_worker(producer_ckpt: str, orderer_ckpt: str) -> None:
    global _WORKER_PROD, _WORKER_ORD
    # pin BLAS/torch per worker (gap_attribution lesson: 16 workers × default
    # intra-op threads oversubscribed the box, load 157, nothing finished).
    import os as _os
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _os.environ[_v] = "1"
    try:
        import torch as _torch
        _torch.set_num_threads(1)
    except ImportError:
        pass
    from ..train.producer_scorer import load_producer_score_model
    from ..train.orderer_scorer import load_scorer

    _WORKER_PROD = load_producer_score_model(str(producer_ckpt))
    _WORKER_ORD = load_scorer(str(orderer_ckpt))


def _worker_eval(payload: dict) -> dict:
    if _WORKER_PROD is None or _WORKER_ORD is None:
        raise RuntimeError("train_closure worker was not initialized")
    from .curriculum import replay
    from .joint_stack import run_joint_stack

    rec = TaskRecord.from_json_line(payload["record"])
    cfg = replay(rec)
    seeds = range(rec.episode_seed, rec.episode_seed + payload["eval_seeds"])
    units = [run_joint_stack(_WORKER_PROD, _WORKER_ORD, cfg, seed=s,
                             horizon=payload["horizon"])["consumer_completed_units"]
             for s in seeds]
    return {"task_id": rec.task_id, "candidate_units": round(sum(units) / len(units), 2)}


def run_train_closure(archive: str | Path, attribution: str | Path, producer_ckpt: str,
                      orderer_ckpt: str, out_dir: str | Path, *, split: str = "train",
                      split_seed: int = 0, val_frac: float = 0.25, blind_frac: float = 0.25,
                      eval_seeds: int = 3, horizon: int = 100, workers: int = 1,
                      max_tasks: int | None = None, log: Log = print) -> dict:
    from .curriculum_replay import _load_archive, split_archive

    records = _load_archive(archive)
    train, val, blind = split_archive(records, val_frac=val_frac, blind_frac=blind_frac,
                                      seed=split_seed)
    chosen = {"train": train, "val": val, "blind": blind}[split]
    if max_tasks is not None:
        chosen = chosen[:max_tasks]
    units = read_attribution_units(attribution)
    missing = [r.task_id for r in chosen if r.task_id not in units]
    if missing:
        raise KeyError(f"{len(missing)} {split} tasks missing from attribution CSV: "
                       f"{missing[:3]}...")

    log(f"train_closure: {len(chosen)} {split} tasks over {workers} workers "
        f"(eval_seeds={eval_seeds}, horizon={horizon}); starting...")
    payloads = [{"record": r.to_json_line(), "eval_seeds": eval_seeds, "horizon": horizon}
                for r in chosen]
    evaluated: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(str(producer_ckpt), str(orderer_ckpt))) as pool:
        futures = [pool.submit(_worker_eval, p) for p in payloads]
        for future in concurrent.futures.as_completed(futures):
            e = future.result()
            evaluated.append(e)
            log(f"({len(evaluated)}/{len(payloads)} done) {e['task_id']}: "
                f"cand={e['candidate_units']}")

    rows = join_closure_rows(evaluated, units)
    rows.sort(key=lambda r: r["gap_units"], reverse=True)
    summary = {"split": split, **closure_summary(rows)}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{split}_closure.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / f"{split}_closure_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"train_closure summary: {json.dumps(summary)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, help="mined archive the replay trained from")
    ap.add_argument("--attribution", required=True,
                    help="gap_attribution.csv with champion_units/full_bar_units per task")
    ap.add_argument("--producer-ckpt", required=True, help="CANDIDATE producer score model")
    ap.add_argument("--orderer-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="train", choices=("train", "val", "blind"))
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--blind-frac", type=float, default=0.25)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-tasks", type=int, default=None)
    args = ap.parse_args()
    run_train_closure(args.archive, args.attribution, args.producer_ckpt, args.orderer_ckpt,
                      args.out_dir, split=args.split, split_seed=args.split_seed,
                      val_frac=args.val_frac, blind_frac=args.blind_frac,
                      eval_seeds=args.eval_seeds, horizon=args.horizon,
                      workers=args.workers, max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
