"""Run many independent `smoke_ppo` training runs concurrently, one OS process per
core.

Why this exists (not in-process): the PPO rollout in `smoke_ppo.collect` is a
single-threaded Python round-robin over `num_envs` (ADR-0005) — extra cores do
nothing for one run. The throughput win on an N-core box is N *independent* runs
at once: the boundary-mapping grid (VERIFICATION: depth/p sweep), the `--num-seeds`
robustness sweep, or a hyperparameter sweep. In-process parallelism is the wrong
tool here: `train_seeds` is coupled to monkeypatch/shared-state test contracts and
the domain-randomization closure is unpicklable. Subprocess fan-out sidesteps both
and lets each child pin its own BLAS threads.

Each job is `python -m tokenomics.train.smoke_ppo <common> <job args> --threads T`
(T defaults to 1 — see smoke_ppo `--threads`: batch-1 inference makes multithreaded
BLAS pure overhead, and one thread per process is what keeps N parallel runs from
oversubscribing the cores). Stdout+stderr stream to `<log-dir>/<name>.log`.

Example — the VERIFICATION boundary grid (fixed depth-3, climbing demand rung):

    python -m tokenomics.train.parallel --workers 8 \\
        --common "--iters 400 --steps 1024 --topology-depth 3 --held-out-seeds 5" \\
        --job d3_p00 "--max-diff 0.00" \\
        --job d3_p10 "--max-diff 0.10" \\
        --job d3_p24 "--max-diff 0.24" \\
        --job d3_p25 "--max-diff 0.25"

Example — an N-seed sweep fanned across cores (one process per seed):

    python -m tokenomics.train.parallel --workers 8 --seeds 0-7 \\
        --common "--iters 400 --steps 1024 --config config/scale_d3.json"
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Job:
    """One training run: a `name` (used for the log file) and the smoke_ppo CLI
    args (already split into argv tokens) that distinguish it from the others."""

    name: str
    args: list[str]


@dataclass(frozen=True)
class JobResult:
    """Outcome of one finished job — exit code, wall time, and where its log went."""

    name: str
    returncode: int
    seconds: float
    log_path: Path


def build_jobs(
    common: Sequence[str],
    named: Sequence[tuple[str, str]],
    seeds: range | None,
    threads: int,
) -> list[Job]:
    """Expand the CLI spec into concrete jobs. `common` tokens prefix every job;
    each `--job NAME "ARGS"` becomes one job; `--seeds LO-HI` fans out one job per
    seed (`--seed i`). A job's args carry `--threads T` unless the caller already
    pinned threads in `common`/the job args (so an explicit override wins)."""
    pins_threads = any(t == "--threads" for t in common)
    jobs: list[Job] = []

    def with_threads(args: list[str]) -> list[str]:
        if pins_threads or "--threads" in args:
            return args
        return args + ["--threads", str(threads)]

    for name, arg_str in named:
        jobs.append(Job(name, with_threads([*common, *shlex.split(arg_str)])))
    if seeds is not None:
        for s in seeds:
            jobs.append(Job(f"seed{s}", with_threads([*common, "--seed", str(s)])))
    return jobs


def _run_one(job: Job, log_dir: Path, dry_run: bool) -> JobResult:
    """Launch one smoke_ppo subprocess, streaming combined output to its log file.
    Uses the current interpreter (`sys.executable -m`) so the active venv is kept."""
    cmd = [sys.executable, "-m", "tokenomics.train.smoke_ppo", *job.args]
    log_path = log_dir / f"{job.name}.log"
    if dry_run:
        print(f"[dry-run] {job.name}: {shlex.join(cmd)}")
        return JobResult(job.name, 0, 0.0, log_path)
    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"# {shlex.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    print(f"[done] {job.name:16}  rc={proc.returncode}  {elapsed:7.1f}s  -> {log_path}")
    return JobResult(job.name, proc.returncode, elapsed, log_path)


def run_jobs(
    jobs: Sequence[Job],
    *,
    workers: int,
    log_dir: Path,
    dry_run: bool = False,
) -> list[JobResult]:
    """Run `jobs` with at most `workers` concurrent subprocesses. The pool is
    threads, not processes: each worker only blocks on a child process, so the GIL
    is irrelevant and the real CPU work runs in the children, one BLAS thread each."""
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"running {len(jobs)} job(s), {workers} at a time, logs in {log_dir}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda j: _run_one(j, log_dir, dry_run), jobs))
    failed = [r for r in results if r.returncode != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} ok"
          + (f"; FAILED: {', '.join(r.name for r in failed)}" if failed else ""))
    return results


def _parse_seeds(spec: str) -> range:
    """`LO-HI` (inclusive) -> range. A bare `N` means just seed N."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return range(int(lo), int(hi) + 1)
    n = int(spec)
    return range(n, n + 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fan out independent smoke_ppo runs across CPU cores.",
    )
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="max concurrent runs (default: CPU count). Keep ~= cores when each run is --threads 1.")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads passed to each child run (default 1; see smoke_ppo --threads). Ignored for a job that sets --threads itself.")
    ap.add_argument("--log-dir", type=str, default=None,
                    help="directory for per-job logs (default: ./parallel_runs/<timestamp>)")
    ap.add_argument("--common", action="append", default=[],
                    help="smoke_ppo args shared by every job (repeatable; each is shell-split)")
    ap.add_argument("--job", nargs=2, action="append", default=[], metavar=("NAME", "ARGS"),
                    help="one variant: a log name + its distinguishing smoke_ppo args (repeatable)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="convenience: LO-HI (inclusive) -> one job per seed (--seed i)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    common: list[str] = []
    for chunk in args.common:
        common.extend(shlex.split(chunk))
    seeds = _parse_seeds(args.seeds) if args.seeds else None
    jobs = build_jobs(common, args.job, seeds, args.threads)
    if not jobs:
        ap.error("no jobs: pass at least one --job NAME ARGS or --seeds LO-HI")

    log_dir = Path(args.log_dir) if args.log_dir else Path("parallel_runs") / time.strftime("%Y%m%d-%H%M%S")
    results = run_jobs(jobs, workers=args.workers, log_dir=log_dir, dry_run=args.dry_run)
    if any(r.returncode != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
