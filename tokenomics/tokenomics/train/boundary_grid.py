"""Boundary-mapping grid (VERIFICATION.md): locate the first learning wall in the
depth × quantity plane.

The open question (VERIFICATION §"What we do not yet know"): does from-scratch PPO
fail because of chain *depth* (depth-3 producer credit assignment) or because of
*variable quantity* (the demand rung, `--max-diff`)? This driver runs the clean
experiment that maps the boundary:

    depth 1, p=0.00
    depth 2, p=0.00
    depth 3, p=0.00
    depth 3, p=0.10
    depth 3, p=0.24
    depth 3, p=0.25

For each cell it (1) materializes the FIXED scale topology of that depth to a JSON on
disk — the only clean way to pin a depth-D base (`--topology-depth` alone is inert
without `--topology-random`); (2) trains from-scratch PPO holding the curriculum at the
cell's quantity rung (`--max-diff`), streaming stdout live to the cell's log file; and
(3) runs the policy-swap probe (`eval.swap_eval`) on the trained checkpoint to isolate
whether the producer or the orderer is the blocker.

Everything lands under one run dir: `configs/scale_dD.json` (the materialized
topologies), `<cell>.log` (live training log), `<cell>.pt` (checkpoint), `<cell>.csv`
(per-iter curve), and `<cell>.swap.txt` (the policy-swap table). All child processes
run with PYTHONUNBUFFERED so logs flush to file with no buffering delay.

Run (local smoke — prove the plumbing):
    python -m tokenomics.train.boundary_grid --iters 5 --workers 2 --cells d3_p00,d3_p25

Run (full grid):
    python -m tokenomics.train.boundary_grid --iters 300 --steps 1024 \\
        --held-out-seeds 5 --swap-seeds 5 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .parallel import Job, run_jobs
from ..eval.scale import build_scale_config, config_to_dict


@dataclass(frozen=True)
class Cell:
    """One grid cell: a `name` (log/ckpt stem), the supply-chain `depth` of its fixed
    scale topology, and the curriculum quantity rung `max_diff` it is held at."""

    name: str
    depth: int
    max_diff: float


# The VERIFICATION boundary grid: depth sweep at the easiest rung, then the quantity
# sweep at the contested depth-3 topology.
DEFAULT_CELLS: tuple[Cell, ...] = (
    Cell("d1_p00", 1, 0.00),
    Cell("d2_p00", 2, 0.00),
    Cell("d3_p00", 3, 0.00),
    Cell("d3_p10", 3, 0.10),
    Cell("d3_p24", 3, 0.24),
    Cell("d3_p25", 3, 0.25),
)


def log(msg: str) -> None:
    print(f"[boundary] {msg}", flush=True)


def artifacts_dir(out_dir: Path) -> Path:
    """Heavy per-cell files (checkpoints, raw logs, per-iter curves, swap tables) live
    under `<out_dir>/artifacts/`; the tracked experiment records (spec.json, summary.csv,
    RESULT.md) + materialized `configs/` sit at `<out_dir>/` root. This split is the
    experiments-repo layout (experiments/README.md): point `--out-dir` at an EXP dir and
    the run drops straight into the convention (artifacts/ is the git-lfs tier)."""
    return out_dir / "artifacts"


def materialize_configs(cells: list[Cell], cfg_dir: Path) -> dict[int, Path]:
    """Write the FIXED scale topology JSON for every distinct depth in `cells`.

    Returns depth -> path. The file is the authoritative record of exactly what each
    cell trained on (so the experiment is reproducible / inspectable), and the same
    file is what `smoke_ppo --config` reads — train and the swap probe see one config."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for depth in sorted({c.depth for c in cells}):
        cfg = build_scale_config(0, depth=depth)
        path = cfg_dir / f"scale_d{depth}.json"
        path.write_text(json.dumps(config_to_dict(cfg), indent=2))
        n_prod = len(cfg.producer_configs)
        log(f"materialized depth {depth} -> {path} ({n_prod} producers, {len(cfg.process_sheets)} sheets)")
        paths[depth] = path
    return paths


def build_grid_jobs(
    cells: list[Cell],
    cfg_paths: dict[int, Path],
    out_dir: Path,
    *,
    iters: int,
    steps: int,
    seed: int,
    held_out_seeds: int,
    eval_every: int,
    eval_seeds: int,
) -> list[Job]:
    """Expand each cell into a from-scratch smoke_ppo job (one fixed-topology PPO run).

    `--config` pins the cell's depth topology; `--max-diff` holds the curriculum at the
    cell's quantity rung; `--save`/`--out` write the checkpoint + curve the swap probe
    and the summary read back. `--threads 1` (set here, not left to parallel's default)
    keeps every fanned-out run single-BLAS so N cells share the cores cleanly."""
    art = artifacts_dir(out_dir)
    jobs: list[Job] = []
    for c in cells:
        ckpt = art / f"{c.name}.pt"
        csv = art / f"{c.name}.csv"
        args = [
            "--config", str(cfg_paths[c.depth]),
            "--max-diff", f"{c.max_diff}",
            "--iters", str(iters),
            "--steps", str(steps),
            "--seed", str(seed),
            "--held-out-seeds", str(held_out_seeds),
            "--eval-every", str(eval_every),
            "--eval-seeds", str(eval_seeds),
            "--save", str(ckpt),
            "--out", str(csv),
            "--threads", "1",
        ]
        jobs.append(Job(c.name, args))
    return jobs


def run_swaps(
    cells: list[Cell],
    out_dir: Path,
    *,
    swap_seeds: int,
    horizon: int,
    workers: int,
) -> None:
    """Run the policy-swap probe on every trained checkpoint, fanned out across cores
    via `parallel.run_jobs` — the SAME executor as the training stage, pointed at the
    `eval.swap_eval` module (Job.module). Each table streams to `<cell>.swap.log`
    (run_jobs writes `<name>.log`). A cell whose checkpoint is missing (its training job
    failed) is skipped."""
    art = artifacts_dir(out_dir)
    jobs: list[Job] = []
    for c in cells:
        ckpt = art / f"{c.name}.pt"
        if not ckpt.exists():
            log(f"swap skip {c.name}: no checkpoint at {ckpt}")
            continue
        args = [
            "--pt", str(ckpt), "--depth", str(c.depth),
            "--seeds", str(swap_seeds), "--horizon", str(horizon),
        ]
        jobs.append(Job(f"{c.name}.swap", args, module="tokenomics.eval.swap_eval"))
    if jobs:
        run_jobs(jobs, workers=workers, log_dir=art)


def _parse_held_out(log_path: Path) -> dict[str, float]:
    """Pull the metrics off a cell's `held-out (...)  cyc .. dead .. cons✓ .. cu✓ .. ret ..`
    line (smoke_ppo.train). Empty dict if the run produced no held-out line."""
    keys = {"cyc": "cyc", "dead": "dead", "cu✓": "cu", "ret": "ret"}
    if not log_path.exists():
        return {}
    for line in log_path.read_text().splitlines():
        if not line.startswith("held-out"):
            continue
        toks = line.split()
        out: dict[str, float] = {}
        for i, t in enumerate(toks):
            if t in keys and i + 1 < len(toks):
                try:
                    out[keys[t]] = float(toks[i + 1])
                except ValueError:
                    pass
        return out
    return {}


def _parse_swap(swap_path: Path) -> dict[str, dict[str, float]]:
    """Parse a `swap_eval` table into {combo -> {units, dead, cons, ret}}. Combo keys are
    normalized to `bb` (base+base), `tt`, `tb`, `bt`."""
    norm = {"base  + base": "bb", "train + train": "tt", "train + base": "tb", "base  + train": "bt"}
    rows: dict[str, dict[str, float]] = {}
    if not swap_path.exists():
        return rows
    for line in swap_path.read_text().splitlines():
        for label, key in norm.items():
            if line.startswith(label):
                nums = line[len(label):].split()
                if len(nums) >= 4:
                    rows[key] = dict(units=float(nums[0]), dead=float(nums[1]),
                                     cons=float(nums[2]), ret=float(nums[3]))
    return rows


def collect_rows(cells: list[Cell], out_dir: Path) -> list[dict]:
    """Parse every cell's held-out line + swap table into one structured row list —
    the shared source for the console summary, summary.csv, and RESULT.md."""
    art = artifacts_dir(out_dir)
    rows: list[dict] = []
    for c in cells:
        ho = _parse_held_out(art / f"{c.name}.log")
        sw = _parse_swap(art / f"{c.name}.swap.log")
        bb = sw.get("bb", {}).get("units", float("nan"))
        tt = sw.get("tt", {}).get("units", float("nan"))
        ratio = (tt / bb) if bb else float("nan")
        rows.append({
            "cell": c.name, "depth": c.depth, "max_diff": c.max_diff,
            "held_cu": ho.get("cu", float("nan")), "held_dead": ho.get("dead", float("nan")),
            "held_cyc": ho.get("cyc", float("nan")), "held_ret": ho.get("ret", float("nan")),
            "swap_bb": bb, "swap_tt": tt, "swap_ratio": ratio,
            "swap_tb": sw.get("tb", {}).get("units", float("nan")),
            "swap_bt": sw.get("bt", {}).get("units", float("nan")),
            "swap_tb_dead": sw.get("tb", {}).get("dead", float("nan")),
        })
    return rows


def write_summary_csv(rows: list[dict], out_dir: Path) -> None:
    """Tracked, diff-able per-cell metrics table — the experiment's machine-readable result."""
    import csv as _csv
    path = out_dir / "summary.csv"
    cols = ["cell", "depth", "max_diff", "held_cu", "held_dead", "held_cyc", "held_ret",
            "swap_bb", "swap_tt", "swap_ratio", "swap_bt", "swap_tb", "swap_tb_dead"]
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    log(f"wrote {path}")


def write_spec(out_dir: Path, cells: list[Cell], cfg_paths: dict[int, Path], args) -> None:
    """Declarative reproducer (experiments/README.md convention): the driver + every knob
    needed to re-run this grid, plus the materialized config filenames."""
    spec = {
        "version": "boundary-grid-v1",
        "date": time.strftime("%Y-%m-%d"),
        "driver": "tokenomics.train.boundary_grid",
        "params": {
            "iters": args.iters, "steps": args.steps, "seed": args.seed,
            "held_out_seeds": args.held_out_seeds, "eval_every": args.eval_every,
            "eval_seeds": args.eval_seeds, "swap_seeds": args.swap_seeds,
            "swap_horizon": args.swap_horizon,
        },
        "cells": [{"name": c.name, "depth": c.depth, "max_diff": c.max_diff} for c in cells],
        "configs": {str(d): f"configs/{p.name}" for d, p in cfg_paths.items()},
    }
    path = out_dir / "spec.json"
    path.write_text(json.dumps(spec, indent=2))
    log(f"wrote {path}")


def write_result_md(rows: list[dict], out_dir: Path) -> None:
    """Auto-generated RESULT.md skeleton: the metrics table + a Conclusions stub for the
    human to fill. (For a curated finding, edit this file after the run — EXP-0001's was
    hand-written.)"""
    lines = [
        f"# {out_dir.name} — boundary grid", "",
        f"_Generated {time.strftime('%Y-%m-%d')} by `tokenomics.train.boundary_grid`. See `spec.json` to reproduce._", "",
        "| cell | depth | p | held cu✓ | held dead | swap base+base | swap train+train | train/base | base+train | train+base |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cell']} | {r['depth']} | {r['max_diff']:.2f} | {r['held_cu']:.1f} | "
            f"{r['held_dead']:.2f} | {r['swap_bb']:.1f} | {r['swap_tt']:.1f} | "
            f"{r['swap_ratio']:.2f} | {r['swap_bt']:.1f} | {r['swap_tb']:.1f} |"
        )
    lines += ["", "## Conclusions", "", "_TODO: fill in._", ""]
    path = out_dir / "RESULT.md"
    path.write_text("\n".join(lines))
    log(f"wrote {path}")


def summarize(cells: list[Cell], out_dir: Path) -> None:
    """Print the held-out line from each cell's training log and its swap table, so the
    whole boundary is readable from one place."""
    art = artifacts_dir(out_dir)
    print("\n" + "=" * 70, flush=True)
    print("BOUNDARY GRID SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for c in cells:
        log_path = art / f"{c.name}.log"
        held = "(no held-out line)"
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.startswith("held-out"):
                    held = line.strip()
        print(f"\n--- {c.name}  (depth {c.depth}, max-diff {c.max_diff}) ---", flush=True)
        print(f"  {held}", flush=True)
        swap_path = art / f"{c.name}.swap.log"
        if swap_path.exists():
            for line in swap_path.read_text().splitlines():
                if line.strip() and not line.startswith(("===", "#")):
                    print(f"  {line}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Boundary-mapping grid: from-scratch PPO + policy-swap across depth × quantity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--iters", type=int, default=300, help="PPO iterations per cell")
    ap.add_argument("--steps", type=int, default=1024, help="rollout decisions per iter")
    ap.add_argument("--seed", type=int, default=0, help="model seed for every cell")
    ap.add_argument("--held-out-seeds", type=int, default=5, help="disjoint held-out eval seeds reported per cell")
    ap.add_argument("--eval-every", type=int, default=10, help="greedy-eval cadence (iters) for checkpoint selection")
    ap.add_argument("--eval-seeds", type=int, default=3, help="validation seeds per greedy eval")
    ap.add_argument("--swap-seeds", type=int, default=5, help="seeds for the policy-swap probe")
    ap.add_argument("--swap-horizon", type=int, default=100, help="horizon for the policy-swap probe")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                    help="max concurrent cell runs (one BLAS thread each)")
    ap.add_argument("--cells", type=str, default=None,
                    help="comma-separated cell names to run (default: the full grid). Names: "
                         + ",".join(c.name for c in DEFAULT_CELLS))
    ap.add_argument("--out-dir", type=str, default=None,
                    help="run dir (default: ./boundary_runs/<timestamp>)")
    ap.add_argument("--no-swap", action="store_true", help="skip the policy-swap probe")
    ap.add_argument("--dry-run", action="store_true", help="print the cell commands without running")
    args = ap.parse_args()

    cells = list(DEFAULT_CELLS)
    if args.cells:
        want = {n.strip() for n in args.cells.split(",")}
        by_name = {c.name: c for c in DEFAULT_CELLS}
        unknown = want - by_name.keys()
        if unknown:
            ap.error(f"unknown cell(s): {sorted(unknown)}; known: {sorted(by_name)}")
        cells = [by_name[n] for n in by_name if n in want]

    out_dir = Path(args.out_dir) if args.out_dir else Path("boundary_runs") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir(out_dir).mkdir(parents=True, exist_ok=True)
    log(f"run dir: {out_dir.resolve()}")
    log(f"cells: {', '.join(c.name for c in cells)}")

    cfg_paths = materialize_configs(cells, out_dir / "configs")
    write_spec(out_dir, cells, cfg_paths, args)
    jobs = build_grid_jobs(
        cells, cfg_paths, out_dir,
        iters=args.iters, steps=args.steps, seed=args.seed,
        held_out_seeds=args.held_out_seeds, eval_every=args.eval_every,
        eval_seeds=args.eval_seeds,
    )
    results = run_jobs(jobs, workers=args.workers, log_dir=artifacts_dir(out_dir), dry_run=args.dry_run)
    if args.dry_run:
        return

    if not args.no_swap:
        run_swaps(cells, out_dir, swap_seeds=args.swap_seeds,
                  horizon=args.swap_horizon, workers=args.workers)
    rows = collect_rows(cells, out_dir)
    write_summary_csv(rows, out_dir)
    write_result_md(rows, out_dir)
    summarize(cells, out_dir)
    log(f"done. records + artifacts in {out_dir.resolve()}")
    if any(r.returncode != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
