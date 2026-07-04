"""Single-task overfit discriminator (EXP-0034 follow-up; CLAUDE.md root-cause rule).

The train-closure diagnostic showed a cloning-fidelity wall: the replay candidate
captures ~13% of its own teacher's gap ON ITS OWN TRAIN SPLIT. Two causes demand
opposite fixes: an optimization/capacity limit at scale (fix: bigger scorer, staged
DAgger) vs representability (the scorer's observation/architecture cannot express the
planner's behavior at all). Discriminator: distill the producer scorer on ONE mined
task with a generous epoch budget — the recipe exactly as Stage-4 replay runs it
(entity_planner teacher, proportional-orderer collect at seed 0) — then evaluate the
clone with the frozen champion orderer on the task's paired eval seeds. Teacher
matched (fidelity ≥ ~0.98 of `bar_producer_champ_orderer_units`) → the class can
express it; the wall is optimization/data at scale. Not matched even here →
representability (or the recipe's collect-vs-deploy covariate shift: samples come
from proportional-orderer seed-0 episodes, deployment pairs the champion orderer on
other seeds — see `distill_producer_scorer`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

Log = Callable[[str], None]

_MATCH_FIDELITY = 0.98


def probe_verdict(clone_units: float, teacher_units: float, champion_units: float) -> dict:
    """Fidelity vs own teacher + gap closure for one (task, arm) distill."""
    gap = teacher_units - champion_units
    return {
        "clone_units": round(clone_units, 2),
        "teacher_units": round(teacher_units, 2),
        "champion_units": round(champion_units, 2),
        "fidelity": round(clone_units / teacher_units, 4) if teacher_units else None,
        "closure": round((clone_units - champion_units) / gap, 4) if gap >= 1.0 else None,
        "teacher_matched": teacher_units > 0
                           and clone_units >= _MATCH_FIDELITY * teacher_units,
    }


def read_probe_units(attribution: str | Path) -> dict[str, tuple[float, float]]:
    """task_id -> (teacher stack units, champion units) from the attribution CSV.

    Teacher = `bar_producer_champ_orderer_units`: the exact producer the replay
    distills from, paired with the same frozen orderer the clone deploys with."""
    with Path(attribution).open() as fh:
        return {r["task_id"]: (float(r["bar_producer_champ_orderer_units"]),
                               float(r["champion_units"]))
                for r in csv.DictReader(fh)}


def run_overfit_probe(archive: str | Path, attribution: str | Path, orderer_ckpt: str,
                      out_dir: str | Path, *, task_ids: list[str],
                      hiddens: tuple[int, ...] = (64,), epochs: int = 2000,
                      lr: float = 3e-3, eval_seeds: int = 3, horizon: int = 100,
                      log: Log = print) -> list[dict]:
    from .curriculum import replay
    from .curriculum_replay import _load_archive
    from .joint_stack import run_joint_stack
    from ..train.orderer_scorer import load_scorer
    from ..train.producer_scorer import distill_producer_scorer

    by_id = {r.task_id: r for r in _load_archive(archive)}
    missing = [t for t in task_ids if t not in by_id]
    if missing:
        raise KeyError(f"tasks not in archive: {missing}")
    units = read_probe_units(attribution)
    orderer = load_scorer(str(orderer_ckpt))

    rows: list[dict] = []
    for task_id in task_ids:
        rec = by_id[task_id]
        cfg = replay(rec)
        teacher_u, champ_u = units[task_id]
        for hidden in hiddens:
            log(f"[probe] {task_id} hidden={hidden} epochs={epochs}: distilling...")
            model, _ = distill_producer_scorer(
                depth=3, train_seeds=[], eval_seeds_list=[], epochs=epochs, lr=lr,
                eval_seeds=eval_seeds, horizon=horizon, hidden=hidden,
                train_configs=[cfg], log=log)
            seeds = range(rec.episode_seed, rec.episode_seed + eval_seeds)
            clone = [run_joint_stack(model, orderer, cfg, seed=s, horizon=horizon)
                     ["consumer_completed_units"] for s in seeds]
            row = {"task_id": task_id, "hidden": hidden, "epochs": epochs,
                   **probe_verdict(sum(clone) / len(clone), teacher_u, champ_u)}
            rows.append(row)
            log(f"[probe] {task_id} hidden={hidden}: clone={row['clone_units']} "
                f"teacher={teacher_u} champ={champ_u} fidelity={row['fidelity']} "
                f"closure={row['closure']} matched={row['teacher_matched']}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "overfit_probe.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    matched = sum(1 for r in rows if r["teacher_matched"])
    summary = {"arms": len(rows), "teacher_matched": matched,
               "verdict": "optimization_or_data_at_scale" if matched == len(rows)
                          else "representability_or_recipe" if matched == 0
                          else "mixed"}
    (out / "overfit_probe_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[probe] summary: {json.dumps(summary)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--orderer-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-ids", required=True, help="comma-separated archive task ids")
    ap.add_argument("--hiddens", default="64", help="comma-separated scorer widths (arms)")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    args = ap.parse_args()
    run_overfit_probe(args.archive, args.attribution, args.orderer_ckpt, args.out_dir,
                      task_ids=args.task_ids.split(","),
                      hiddens=tuple(int(h) for h in args.hiddens.split(",")),
                      epochs=args.epochs, lr=args.lr, eval_seeds=args.eval_seeds,
                      horizon=args.horizon)


if __name__ == "__main__":
    main()
