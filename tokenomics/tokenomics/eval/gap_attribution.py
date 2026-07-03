"""Archived-task gap attribution between producer and orderer channels.

This diagnostic is evaluation-only.  It replays mined archive tasks under four stacks:

* champion producer + champion orderer
* bar producer + search/MPC orderer
* bar producer + champion orderer
* champion producer + search/MPC orderer

The cross stacks show whether the mining-bar gap can be recovered by changing the
producer, the orderer, or only their interaction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .curriculum import TaskRecord, _dims, _episode_seeds, replay
from .curriculum_replay import split_archive
from .joint_stack import joint_stack_env
from .orderer_oracle import search_orderer_episode
from .runner import run_episode

_WORKER_CHAMP_PROD: Policy | None = None
_WORKER_CHAMP_ORD: Policy | None = None
_WORKER_BAR_PROD: Policy | None = None


@dataclass(frozen=True)
class EpisodeAggregate:
    units: float
    rejected: float
    dead_rate: float


_NOTE_RE = re.compile(
    r"score=(?P<score>[-+0-9.]+)\s+gap=(?P<gap>[-+0-9.]+)u\s+win=(?P<win>[-+0-9.]+)"
)


def _parse_notes(notes: str) -> dict[str, float | None]:
    m = _NOTE_RE.search(notes or "")
    if not m:
        return {"archive_score": None, "archive_gap": None, "archive_win": None}
    return {
        "archive_score": float(m.group("score")),
        "archive_gap": float(m.group("gap")),
        "archive_win": float(m.group("win")),
    }


def _parse_ints(spec: str) -> list[int]:
    if not spec:
        return []
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _load_archive(path: str | Path) -> list[TaskRecord]:
    with Path(path).open() as fh:
        return [TaskRecord.from_json_line(line) for line in fh if line.strip()]


def _load_matrix_baselines(path: str | Path | None) -> dict[str, dict[str, EpisodeAggregate]]:
    if path is None:
        return {}
    out: dict[str, dict[str, EpisodeAggregate]] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                champion = EpisodeAggregate(
                    units=round(float(row["learner_units"]), 2),
                    rejected=round(float(row.get("learner_rejected_units") or 0.0), 2),
                    dead_rate=round(float(str(row.get("learner_dead", "False")).lower() == "true"), 3),
                )
                full_bar = EpisodeAggregate(
                    units=round(float(row["reference_units"]), 2),
                    rejected=round(float(row.get("reference_rejected_units") or 0.0), 2),
                    dead_rate=round(float(str(row.get("reference_dead", "False")).lower() == "true"), 3),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid curriculum matrix row for {row.get('task_id')!r}: {exc}") from exc
            out[str(row["task_id"])] = {"champion": champion, "full_bar": full_bar}
    return out


def _aggregate(
    run_one: Callable[[int], dict],
    seeds: Iterable[int],
) -> EpisodeAggregate:
    n = 0
    units = rejected = dead = 0.0
    for seed in seeds:
        stats = run_one(seed)
        units += float(stats["consumer_completed_units"])
        rejected += float(stats.get("consumer_rejected_units", 0.0))
        dead += float(stats["dead"])
        n += 1
    if n == 0:
        raise ValueError("attribution requires at least one episode seed")
    return EpisodeAggregate(
        units=round(units / n, 2),
        rejected=round(rejected / n, 2),
        dead_rate=round(dead / n, 3),
    )


def _search_stack(
    cfg,
    prod: Policy,
    seed: int,
    horizon: int,
    *,
    lookahead: int,
    n_samples: int,
    sample_scale: float,
    candidates: tuple[str, ...],
) -> dict:
    env = joint_stack_env(cfg, horizon)
    cand = {name: ORDERING_BASELINES[name] for name in candidates}
    stats, _ = search_orderer_episode(env, prod, cand, n_samples, sample_scale, lookahead, seed)
    return dict(stats)


def _split_labels(
    records: list[TaskRecord],
    *,
    split_seed: int,
    val_frac: float,
    blind_frac: float,
    held_out_seeds: set[int],
) -> dict[str, str]:
    train, val, blind = split_archive(
        records, seed=split_seed, val_frac=val_frac, blind_frac=blind_frac
    )
    labels: dict[str, str] = {r.task_id: "train" for r in train}
    labels.update({r.task_id: "val" for r in val})
    labels.update({r.task_id: "blind" for r in blind})
    for r in train:
        if r.world_seed in held_out_seeds:
            labels[r.task_id] = "train_leakage_dropped"
    return labels


def _compute_row(
    idx: int,
    total: int,
    rec: TaskRecord,
    *,
    split_label: str,
    champ_prod: Policy,
    champ_ord: Policy,
    bar_prod: Policy,
    baseline: dict[str, EpisodeAggregate] | None,
    eval_seeds: int,
    horizon: int,
    lookahead: int,
    n_samples: int,
    sample_scale: float,
    candidates: tuple[str, ...],
    orderer_cross: bool,
) -> tuple[dict, str]:
    cfg = replay(rec)
    seeds = _episode_seeds(rec.episode_seed, eval_seeds)
    params = dict(rec.world_spec.get("params", {}))
    generator = rec.world_spec.get("generator", "")

    if baseline is None:
        champ = _aggregate(
            lambda es: run_episode(joint_stack_env(cfg, horizon), champ_prod, champ_ord, seed=es),
            seeds,
        )
        full_bar = _aggregate(
            lambda es: _search_stack(
                cfg,
                bar_prod,
                es,
                horizon,
                lookahead=lookahead,
                n_samples=n_samples,
                sample_scale=sample_scale,
                candidates=candidates,
            ),
            seeds,
        )
    else:
        champ = baseline["champion"]
        full_bar = baseline["full_bar"]

    bar_prod_champ_ord = _aggregate(
        lambda es: run_episode(joint_stack_env(cfg, horizon), bar_prod, champ_ord, seed=es),
        seeds,
    )
    champ_prod_bar_ord = None
    if orderer_cross:
        champ_prod_bar_ord = _aggregate(
            lambda es: _search_stack(
                cfg,
                champ_prod,
                es,
                horizon,
                lookahead=lookahead,
                n_samples=n_samples,
                sample_scale=sample_scale,
                candidates=candidates,
            ),
            seeds,
        )

    products, producers, caps = _dims(cfg)
    gap = round(full_bar.units - champ.units, 3)
    producer_only_gain = round(bar_prod_champ_ord.units - champ.units, 3)
    orderer_only_gain = (
        round(champ_prod_bar_ord.units - champ.units, 3)
        if champ_prod_bar_ord is not None else None
    )
    interaction_residual = (
        round(gap - producer_only_gain - orderer_only_gain, 3)
        if orderer_only_gain is not None else None
    )
    pos_prod = max(0.0, producer_only_gain)
    pos_ord = max(0.0, orderer_only_gain or 0.0)
    pos_total = pos_prod + pos_ord
    if gap <= 0.0:
        dominant = "no_positive_gap"
    elif not orderer_cross:
        if producer_only_gain >= 0.5 * gap:
            dominant = "producer"
        elif producer_only_gain > 0.0:
            dominant = "producer_partial"
        else:
            dominant = "not_producer"
    elif pos_total == 0.0:
        dominant = "interaction_or_noise"
    elif pos_ord >= 0.5 * gap and pos_ord >= pos_prod:
        dominant = "orderer"
    elif pos_prod >= 0.5 * gap and pos_prod > pos_ord:
        dominant = "producer"
    else:
        dominant = "mixed"

    notes = _parse_notes(rec.notes)
    row = {
        "archive_index": idx,
        "task_id": rec.task_id,
        "split": split_label,
        "generator": generator,
        "world_seed": rec.world_seed,
        "episode_seed": rec.episode_seed,
        "depth": params.get("depth"),
        "producers_per_layer": params.get("producers_per_layer"),
        "products_per_layer": params.get("products_per_layer"),
        "target_products": products,
        "target_producers": producers,
        "target_capacity_types": caps,
        **notes,
        "champion_units": champ.units,
        "full_bar_units": full_bar.units,
        "bar_producer_champ_orderer_units": bar_prod_champ_ord.units,
        "champ_producer_bar_orderer_units": (
            champ_prod_bar_ord.units if champ_prod_bar_ord is not None else None
        ),
        "gap_units": gap,
        "producer_only_gain": producer_only_gain,
        "orderer_only_gain": orderer_only_gain,
        "remaining_gap_if_orderer_fixed": round(full_bar.units - bar_prod_champ_ord.units, 3),
        "remaining_gap_if_producer_fixed": (
            round(full_bar.units - champ_prod_bar_ord.units, 3)
            if champ_prod_bar_ord is not None else None
        ),
        "interaction_residual": interaction_residual,
        "producer_positive_share": round(pos_prod / pos_total, 4) if pos_total else 0.0,
        "orderer_positive_share": round(pos_ord / pos_total, 4) if pos_total else 0.0,
        "dominant_channel": dominant,
        "champion_dead_rate": champ.dead_rate,
        "full_bar_dead_rate": full_bar.dead_rate,
        "bar_producer_champ_orderer_dead_rate": bar_prod_champ_ord.dead_rate,
        "champ_producer_bar_orderer_dead_rate": (
            champ_prod_bar_ord.dead_rate if champ_prod_bar_ord is not None else None
        ),
    }
    if orderer_only_gain is None:
        msg = (
            f"[{idx}/{total}] {rec.task_id}: gap={gap:+.2f}, "
            f"P-only={producer_only_gain:+.2f}, O-only=skipped, {dominant}"
        )
    else:
        msg = (
            f"[{idx}/{total}] {rec.task_id}: gap={gap:+.2f}, "
            f"P-only={producer_only_gain:+.2f}, O-only={orderer_only_gain:+.2f}, "
            f"{dominant}"
        )
    return row, msg


def _init_worker(producer_ckpt: str, orderer_ckpt: str) -> None:
    global _WORKER_BAR_PROD, _WORKER_CHAMP_ORD, _WORKER_CHAMP_PROD
    from ..train.orderer_scorer import load_scorer, scorer_ordering_policy
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

    champ_prod_model = load_producer_score_model(str(producer_ckpt))
    champ_ord_model = load_scorer(str(orderer_ckpt))
    _WORKER_CHAMP_PROD = scorer_producing_policy(champ_prod_model)
    _WORKER_CHAMP_ORD = scorer_ordering_policy(champ_ord_model)
    _WORKER_BAR_PROD = PRODUCING_BASELINES["entity_planner"]


def _worker_row(payload: dict) -> tuple[dict, str]:
    if _WORKER_CHAMP_PROD is None or _WORKER_CHAMP_ORD is None or _WORKER_BAR_PROD is None:
        raise RuntimeError("gap attribution worker was not initialized")
    rec = TaskRecord.from_json_line(payload["record"])
    return _compute_row(
        payload["idx"],
        payload["total"],
        rec,
        split_label=payload["split_label"],
        champ_prod=_WORKER_CHAMP_PROD,
        champ_ord=_WORKER_CHAMP_ORD,
        bar_prod=_WORKER_BAR_PROD,
        baseline=payload["baseline"],
        eval_seeds=payload["eval_seeds"],
        horizon=payload["horizon"],
        lookahead=payload["lookahead"],
        n_samples=payload["n_samples"],
        sample_scale=payload["sample_scale"],
        candidates=payload["candidates"],
        orderer_cross=payload["orderer_cross"],
    )


def run_gap_attribution(
    archive_path: str | Path,
    producer_ckpt: str | Path,
    orderer_ckpt: str | Path,
    out_dir: str | Path,
    *,
    eval_seeds: int = 3,
    horizon: int = 100,
    lookahead: int = 6,
    n_samples: int = 8,
    sample_scale: float = 1.0,
    candidates: tuple[str, ...] = ("even_split", "proportional"),
    split_seed: int = 0,
    val_frac: float = 0.25,
    blind_frac: float = 0.25,
    held_out_seeds: Iterable[int] = (6, 7, 8, 9),
    curriculum_matrix: str | Path | None = None,
    max_tasks: int | None = None,
    workers: int = 1,
    orderer_cross: bool = True,
    log=print,
) -> dict:
    """Replay archive tasks and write attribution artifacts.

    Returns the same summary dictionary written to ``summary.json``.
    """
    from ..train.orderer_scorer import load_scorer, scorer_ordering_policy
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

    records = _load_archive(archive_path)
    if max_tasks is not None:
        records = records[:max_tasks]
    if not records:
        raise ValueError(f"archive has no records: {archive_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    champ_prod_model = load_producer_score_model(str(producer_ckpt))
    champ_ord_model = load_scorer(str(orderer_ckpt))
    champ_prod = scorer_producing_policy(champ_prod_model)
    champ_ord = scorer_ordering_policy(champ_ord_model)
    bar_prod = PRODUCING_BASELINES["entity_planner"]

    split_by_id = _split_labels(
        records,
        split_seed=split_seed,
        val_frac=val_frac,
        blind_frac=blind_frac,
        held_out_seeds=set(held_out_seeds),
    )
    matrix_baselines = _load_matrix_baselines(curriculum_matrix)

    rows: list[dict] = []
    if workers <= 1:
        for idx, rec in enumerate(records, start=1):
            row, msg = _compute_row(
                idx,
                len(records),
                rec,
                split_label=split_by_id.get(rec.task_id, ""),
                champ_prod=champ_prod,
                champ_ord=champ_ord,
                bar_prod=bar_prod,
                baseline=matrix_baselines.get(rec.task_id),
                eval_seeds=eval_seeds,
                horizon=horizon,
                lookahead=lookahead,
                n_samples=n_samples,
                sample_scale=sample_scale,
                candidates=candidates,
                orderer_cross=orderer_cross,
            )
            rows.append(row)
            log(msg)
    else:
        payloads = [
            {
                "idx": idx,
                "total": len(records),
                "record": rec.to_json_line(),
                "split_label": split_by_id.get(rec.task_id, ""),
                "baseline": matrix_baselines.get(rec.task_id),
                "eval_seeds": eval_seeds,
                "horizon": horizon,
                "lookahead": lookahead,
                "n_samples": n_samples,
                "sample_scale": sample_scale,
                "candidates": candidates,
                "orderer_cross": orderer_cross,
            }
            for idx, rec in enumerate(records, start=1)
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(producer_ckpt), str(orderer_ckpt)),
        ) as pool:
            futures = [pool.submit(_worker_row, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                row, msg = future.result()
                rows.append(row)
                log(msg)
        rows.sort(key=lambda r: int(r["archive_index"]))

    csv_path = out_path / "gap_attribution.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positives = [r for r in rows if float(r["gap_units"]) > 0.0]
    sum_gap = sum(float(r["gap_units"]) for r in positives)
    sum_prod = sum(max(0.0, float(r["producer_only_gain"])) for r in positives)
    orderer_values = [
        float(r["orderer_only_gain"]) for r in positives if r["orderer_only_gain"] is not None
    ]
    sum_ord = sum(max(0.0, v) for v in orderer_values)
    by_channel: dict[str, int] = {}
    for r in positives:
        by_channel[str(r["dominant_channel"])] = by_channel.get(str(r["dominant_channel"]), 0) + 1

    summary = {
        "archive": str(archive_path),
        "producer_ckpt": str(producer_ckpt),
        "orderer_ckpt": str(orderer_ckpt),
        "eval_seeds": eval_seeds,
        "horizon": horizon,
        "lookahead": lookahead,
        "n_samples": n_samples,
        "sample_scale": sample_scale,
        "candidates": candidates,
        "curriculum_matrix": str(curriculum_matrix) if curriculum_matrix is not None else None,
        "workers": workers,
        "orderer_cross": orderer_cross,
        "rows": len(rows),
        "positive_gap_rows": len(positives),
        "sum_positive_gap_units": round(sum_gap, 3),
        "sum_positive_producer_only_gain": round(sum_prod, 3),
        "sum_positive_orderer_only_gain": round(sum_ord, 3) if orderer_cross else None,
        "producer_closure_ratio": round(sum_prod / sum_gap, 4) if sum_gap else 0.0,
        "orderer_closure_ratio": (
            round(sum_ord / sum_gap, 4) if orderer_cross and sum_gap else None
        ),
        "dominant_channel_counts": by_channel,
        "csv": str(csv_path),
    }
    with (out_path / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--producer-ckpt", required=True)
    ap.add_argument("--orderer-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--sample-scale", type=float, default=1.0)
    ap.add_argument("--candidates", default="even_split,proportional")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--blind-frac", type=float, default=0.25)
    ap.add_argument("--held-out-seeds", default="6,7,8,9")
    ap.add_argument("--curriculum-matrix", default=None,
                    help="optional EXP-0028 curriculum_matrix.csv; reuses its champion/full-bar units")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--skip-orderer-cross", action="store_true",
                    help="skip expensive champion-producer + search-orderer cross")
    args = ap.parse_args()

    summary = run_gap_attribution(
        args.archive,
        args.producer_ckpt,
        args.orderer_ckpt,
        args.out_dir,
        eval_seeds=args.eval_seeds,
        horizon=args.horizon,
        lookahead=args.lookahead,
        n_samples=args.n_samples,
        sample_scale=args.sample_scale,
        candidates=tuple(x.strip() for x in args.candidates.split(",") if x.strip()),
        split_seed=args.split_seed,
        val_frac=args.val_frac,
        blind_frac=args.blind_frac,
        held_out_seeds=_parse_ints(args.held_out_seeds),
        curriculum_matrix=args.curriculum_matrix,
        max_tasks=args.max_tasks,
        workers=args.workers,
        orderer_cross=not args.skip_orderer_cross,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
