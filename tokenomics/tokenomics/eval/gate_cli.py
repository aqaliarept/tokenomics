"""Gate driver: `python -m tokenomics.eval.gate` (design D6, D11, add-measurement-gate-harness;
gate-harness "Identical demand across observation contracts", "Gate report contents", "Verdict is
reproducible from rows", "Provenance binding", "Sealed worlds need a pre-registration").

Modes:
- `report`: run the subject (`CANDIDATE` rows) and every roster member (R replicates for a random
  member, action seed `seeding.replicate_seed`) on the registry's gate worlds of one split; write the
  EXP-dir layout (`spec.json`, `episodes.csv`, `worlds.csv`, `regimes.csv`, `verdict.json`, a
  `RESULT.md` skeleton). All arithmetic is `gate.compute_report`; this module only produces rows.
- `recompute`: rebuild the report from `episodes.csv` plus the roster / parameter payloads embedded
  in `spec.json` (hash-checked against the named versions) and compare it with `verdict.json`.
- `screen`: roster only, on the development gate worlds and their reserves. In order: zero-baseline
  replacement (registry screening, development split) -> headroom per family × size on the final
  world set (D10; any failure stops with exit 3 and writes no frozen files) -> optional pruning
  (`--prune`, roster.prune_roster) -> X/Y calibration (D4) over the final worlds' episodes with the
  (pruned) roster -> candidate frozen files `frozen/worlds_v1.json` (screened, headroom recorded),
  `frozen/roster_v1.json` (frozen), `frozen/side_effect_params_v1.json`, plus `screening.json`.

Jobs fan out over a spawn-context `ProcessPoolExecutor`, one process per worker, BLAS pinned to one
thread: inference is batch-1 and rollouts are sequential Python (CLAUDE.md "Parallel work").
`pin_blas` forces (not defaults) the BLAS env vars to 1 in the parent before the pool spawns, so a
fresh worker's numpy/torch start single-threaded whatever the shell exported; it also resets an
already-loaded OpenBLAS (numpy's bundled build, via ctypes) and torch in-process, which covers the
`--workers 1` path where numpy is imported before any flag is parsed. Each finished row is appended to `episodes.csv` at once; a rerun on the same `--out-dir`
checks the run pins in `spec.json`, drops a torn last line, skips rows already present and runs the
rest. Demand digests are compared as rows arrive, so a contract mismatch stops the run early; the
pure layer re-checks them.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from statistics import fmean
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .gate import (
    CALIBRATION_RULE,
    CANDIDATE,
    CSV_COLUMNS,
    DigestMismatch,
    EpisodeRow,
    GateReport,
    Calibration,
    Headroom,
    SideEffectParams,
    calibrate_x_y,
    compute_report,
    episode_bests,
    headroom,
    load_side_effect_params,
    params_from_dict,
    row_from_dict,
    row_from_stats,
    row_to_dict,
    side_effect_params_json,
)
from .provenance import environment_manifest, sha256_file, source_fingerprint
from .registry import (
    DEFAULT_REGISTRY,
    DEVELOPMENT,
    RESERVE,
    SEALED,
    SEALED_ACCESS_LOG,
    SPLITS,
    Registry,
    WorldEntry,
    check_sealed_access,
    load_preregistration,
    load_registry,
    record_headroom,
    save_registry,
    screen_zero_baselines,
)
from .roster import (
    DEFAULT_ROSTER,
    GATE_SIZES,
    JOINT_STACK,
    RANDOM_POLICY,
    DevRow,
    Roster,
    contract_for,
    freeze_roster,
    load_roster,
    prune_roster,
    roster_from_dict,
    save_roster,
)
from .seeding import replicate_seed

DRIVER = "tokenomics.eval.gate"
PORTABLE = "portable"
BASELINE_PREFIX = "baseline:"
BLAS_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
Job = dict  # one (world, episode, subject, replicate) run; plain data so it pickles


# --- subject ---


def parse_subject(subject: str) -> tuple[str, str] | None:
    """`baseline:<producing>/<ordering>` -> the pair; `portable` -> None."""
    if subject == PORTABLE:
        return None
    if subject.startswith(BASELINE_PREFIX) and "/" in subject:
        prod, order = subject[len(BASELINE_PREFIX):].split("/", 1)
        if prod not in PRODUCING_BASELINES or order not in ORDERING_BASELINES:
            raise ValueError(f"unknown baseline subject {subject!r}")
        return prod, order
    raise ValueError(f"subject must be {PORTABLE!r} or 'baseline:<producing>/<ordering>', "
                     f"got {subject!r}")


# --- jobs ---


def select_worlds(registry: Registry, split: str, *, sizes: Sequence[int] | None = None,
                  families: Sequence[str] | None = None,
                  max_worlds: int | None = None) -> list[WorldEntry]:
    worlds = [e for e in registry.gate_worlds(split)
              if (not sizes or e.size in sizes) and (not families or e.family in families)]
    return worlds[:max_worlds] if max_worlds else worlds


def build_jobs(worlds: Sequence[WorldEntry], roster: Roster, subject: str | None, *,
               episodes_per_world: int | None = None) -> list[Job]:
    """Every (world, episode, subject, replicate) the run needs: the candidate once per episode
    (R times when a baseline subject is random; none when `subject` is None, the roster-only screen),
    each member `roster.replicates_for` times."""
    pair = parse_subject(subject) if subject is not None else None
    jobs: list[Job] = []
    for e in worlds:
        episodes = e.episode_seeds[:episodes_per_world] if episodes_per_world else e.episode_seeds
        for ep in episodes:
            base = {"world": e.id, "size": e.size, "regime": e.regime, "world_seed": e.world_seed,
                    "generator": e.generator, "episode": ep}
            if subject is None:
                pass
            elif pair is None:
                jobs.append({**base, "subject": CANDIDATE, "replicate": 0, "producing": PORTABLE,
                             "ordering": PORTABLE, "contract": JOINT_STACK, "action_seed": None})
            else:
                random = RANDOM_POLICY in pair
                for r in range(roster.replicates if random else 1):
                    jobs.append({**base, "subject": CANDIDATE, "replicate": r,
                                 "producing": pair[0], "ordering": pair[1],
                                 "contract": contract_for(*pair),
                                 "action_seed": replicate_seed(e.id, ep, r) if random else None})
            for m in roster.members:
                for r in range(roster.replicates_for(m)):
                    jobs.append({**base, "subject": m.id, "replicate": r,
                                 "producing": m.producing, "ordering": m.ordering,
                                 "contract": m.contract,
                                 "action_seed": replicate_seed(e.id, ep, r) if m.random else None})
    return jobs


def job_key(world: str, episode: int, subject: str, replicate: int) -> tuple[str, int, str, int]:
    return (world, episode, subject, replicate)


# --- workers ---

_W: dict = {}
_OPENBLAS_SET = ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_",
                 "openblas_set_num_threads")
_OPENBLAS_GET = ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_",
                 "openblas_get_num_threads")


def _openblas() -> ctypes.CDLL | None:
    """numpy's bundled OpenBLAS, if numpy is loaded and ships one (best effort)."""
    np = sys.modules.get("numpy")
    if np is None:
        return None
    libs = glob.glob(os.path.join(os.path.dirname(np.__file__), os.pardir, "numpy.libs",
                                  "*openblas*"))
    for path in libs:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def blas_threads() -> int | None:
    """Thread count of the loaded OpenBLAS, or None when it cannot be read."""
    lib = _openblas()
    for name in _OPENBLAS_GET if lib is not None else ():
        fn = getattr(lib, name, None)
        if fn is not None:
            return int(fn())
    return None


def pin_blas() -> None:
    """Force one BLAS thread: env vars for libraries not yet loaded (and spawned children), the
    loaded OpenBLAS via its setter, torch when imported."""
    for k in BLAS_ENV:
        os.environ[k] = "1"
    lib = _openblas()
    for name in _OPENBLAS_SET if lib is not None else ():
        fn = getattr(lib, name, None)
        if fn is not None:
            fn(1)
            break
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def _worker_init(horizon: int, prod_ckpt: str | None, orderer_ckpt: str | None) -> None:
    """Per-process state: horizon, a config cache, and the portable policies when the subject
    needs them (torch imported lazily, pinned to one thread)."""
    pin_blas()
    _W.clear()
    _W.update(horizon=horizon, configs={})
    if prod_ckpt and orderer_ckpt:
        import torch

        from ..train.orderer_scorer import load_scorer, scorer_ordering_policy
        from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

        torch.set_num_threads(1)
        _W["portable"] = (scorer_producing_policy(load_producer_score_model(prod_ckpt)),
                          scorer_ordering_policy(load_scorer(orderer_ckpt)))


def _policies(job: Job) -> tuple[Policy, Policy]:
    if job["producing"] == PORTABLE:
        if "portable" not in _W:
            raise RuntimeError("portable subject requested but no checkpoints were loaded")
        return _W["portable"]
    return PRODUCING_BASELINES[job["producing"]], ORDERING_BASELINES[job["ordering"]]


def run_job(job: Job) -> dict[str, str]:
    """Run one episode and return its row as CSV strings (`gate.row_to_dict`)."""
    from ..env import TokenomicsAECEnv
    from .joint_stack import joint_stack_env
    from .runner import run_episode
    from .task_archive import REPRODUCERS

    configs = _W["configs"]
    if job["world"] not in configs:
        g = job["generator"]
        cfg = REPRODUCERS[g["generator"]](job["world_seed"], **g.get("params", {}))
        cfg.validate()
        configs[job["world"]] = cfg
    cfg = configs[job["world"]]
    horizon = _W["horizon"]
    env = (joint_stack_env(cfg, horizon) if job["contract"] == JOINT_STACK
           else TokenomicsAECEnv(config=cfg, horizon=horizon))
    prod, order = _policies(job)
    run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"])
    row = row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                         job["subject"], job["replicate"], env.episode_stats)
    return row_to_dict(row)


# --- output ---


def _append_row(path: Path, row: Mapping[str, str]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def read_rows(path: Path) -> list[EpisodeRow]:
    """Rows already on disk. A torn last line (a crash mid-write) is cut off the file first, so a
    resumed run appends after the last complete row."""
    if not path.exists():
        return []
    text = path.read_text()
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        path.write_text(text)
    if not text.strip():
        return []
    return [row_from_dict(d) for d in csv.DictReader(text.splitlines())]


def _write_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _flat(v: object) -> object:
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return "" if v is None else v


def world_table(report: GateReport) -> list[dict[str, object]]:
    out = []
    for w in report.worlds:
        d = w.to_dict()
        monitored = d.pop("monitored")
        row = {k: _flat(v) for k, v in d.items()}
        row.update({f"monitored_{k}": _flat(v) for k, v in monitored.items()})
        out.append(row)
    return out


def regime_table(report: GateReport) -> list[dict[str, object]]:
    return [{k: _flat(v) for k, v in r.to_dict().items()} for r in report.regimes]


def write_report_files(out: Path, report: GateReport) -> None:
    _write_table(out / "worlds.csv", world_table(report))
    _write_table(out / "regimes.csv", regime_table(report))
    (out / "verdict.json").write_text(report.to_json())


RESULT_SKELETON = """# {title}

{banner}**Verdict: {overall}** (failing clauses: {failing})

Roster {roster}, registry {registry}, side-effect parameters {params}; split {split}, subject
`{subject}`. Produced by `python -m {driver} --mode report` (see `spec.json`).

## Question

TODO

## Result

TODO: plain-English reading of `verdict.json`, `worlds.csv` and `regimes.csv`.

## Conclusions

TODO
"""
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered worlds/episodes/members, an "
              "unfrozen roster, an unscreened registry or no source commit; see `spec.json`.\n\n")


# --- the expected job grid (what a complete episodes.csv must hold) ---

Key = tuple[str, int, str, int]


def job_grid(worlds: Sequence[WorldEntry], episodes_per_world: int | None) -> dict[str, dict]:
    """world -> {size, regime, episodes}: the episodes a report runs, pinned in `spec.json` so a
    resume and a recompute can check the CSV against it without the registry."""
    return {w.id: {"size": w.size, "regime": w.regime,
                   "episodes": list(w.episode_seeds[:episodes_per_world] if episodes_per_world
                                    else w.episode_seeds)}
            for w in worlds}


def expected_keys(grid: Mapping[str, Mapping], roster: Roster, candidate_replicates: int) -> set[Key]:
    keys: set[Key] = set()
    for world, g in grid.items():
        for ep in g["episodes"]:
            keys.update((world, ep, CANDIDATE, r) for r in range(candidate_replicates))
            for m in roster.members:
                keys.update((world, ep, m.id, r) for r in range(roster.replicates_for(m)))
    return keys


def check_rows(rows: Sequence[EpisodeRow], grid: Mapping[str, Mapping], expected: set[Key], *,
               complete: bool) -> None:
    """Every row must be an expected (world, episode, subject, replicate) key, appear once, and
    carry its world's pinned size and regime; with `complete`, every expected key must be present.
    A missing world, episode or replicate is corruption, not a smaller report."""
    seen: set[Key] = set()
    for r in rows:
        k = job_key(r.world, r.episode, r.subject, r.replicate)
        if k not in expected:
            raise ValueError(f"row {k} is not in the pinned job grid")
        if k in seen:
            raise ValueError(f"duplicate row {k}")
        seen.add(k)
        g = grid[r.world]
        if (r.size, r.regime) != (g["size"], g["regime"]):
            raise ValueError(f"row {k}: (size, regime) {(r.size, r.regime)} != pinned "
                             f"{(g['size'], g['regime'])}")
    if complete and seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"episodes.csv is incomplete: {len(missing)} of {len(expected)} pinned "
                         f"rows missing, e.g. {missing[:3]}")


# --- shared run machinery (report and screen) ---


def _write_spec(out: Path, spec: Mapping[str, object]) -> None:
    (out / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")


def _prepare_resume(out: Path, pins: Mapping[str, object], grid: Mapping[str, Mapping],
                    expected: set[Key]) -> tuple[dict | None, list[EpisodeRow]]:
    """Refuse anything a resume cannot honestly continue, before any write: rows without a spec
    (orphans), a spec with different pins (inputs, source or grid), rows outside the grid."""
    spec_path, episodes = out / "spec.json", out / "episodes.csv"
    old_spec = json.loads(spec_path.read_text()) if spec_path.exists() else None
    if old_spec is None and episodes.exists() and episodes.stat().st_size:
        raise ValueError(f"{episodes} exists without spec.json: orphan rows, refusing to resume")
    if old_spec is not None and old_spec.get("pins") != pins:
        raise ValueError(f"{out} holds a run with different pins (inputs, source or grid); "
                         f"refusing to resume:\n  recorded {old_spec.get('pins')}\n  now      {pins}")
    done = read_rows(episodes) if old_spec is not None else []
    check_rows(done, grid, expected, complete=False)
    return old_spec, done


def _open_spec(out: Path, old_spec: dict | None, workers: int, fresh: dict) -> dict:
    """A new spec (provenance taken now) or, on a resume with matching pins, the original one with
    the resume appended: rows keep the provenance they were produced under."""
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    if old_spec is None:
        spec = {**fresh, "date": now[:10], "started": now, "environment": environment_manifest(),
                "blas_threads": None, "workers": workers, "resumes": [], "status": "running"}
    else:
        spec = old_spec
        spec["resumes"] = spec.get("resumes", []) + [
            {"time": now, "workers": workers, "environment": environment_manifest()}]
        spec["status"] = "running"
    _write_spec(out, spec)
    return spec


def _execute(out: Path, spec: dict, done: Sequence[EpisodeRow], jobs: Sequence[Job],
             expected: set[Key], workers: int, init: tuple,
             run_job_fn: Callable[[Job], dict[str, str]], log: Callable[[str], None]) -> None:
    """Run every job not already in `done`, appending each row as it lands and checking demand
    digests per (world, episode) as rows arrive."""
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    digests: dict[tuple[str, int], tuple[str, str]] = {}

    def note_digest(row: EpisodeRow) -> None:
        ref = digests.setdefault((row.world, row.episode), (row.demand_digest, row.subject))
        if row.demand_digest != ref[0]:
            raise DigestMismatch(
                f"world {row.world} episode {row.episode}: {row.subject} replicate "
                f"{row.replicate} saw demand digest {row.demand_digest}, {ref[1]} saw {ref[0]}")

    for r in done:
        note_digest(r)
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    todo = [j for j in jobs
            if job_key(j["world"], j["episode"], j["subject"], j["replicate"]) not in done_keys]
    log(f"[gate] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    def finish(d: dict[str, str]) -> None:
        row = row_from_dict(d)
        note_digest(row)
        _append_row(episodes, d)
        log(f"[gate] {row.world} ep={row.episode} {row.subject} r{row.replicate} "
            f"units={row.units:g} dead={row.dead}")

    pin_blas()  # before the pool spawns: children inherit the forced env
    if spec.get("blas_threads") is None:
        spec["blas_threads"] = blas_threads()
    if workers <= 1 or len(todo) <= 1:
        _worker_init(*init)
        for job in todo:
            finish(run_job_fn(job))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=_worker_init, initargs=init) as pool:
            futures = [pool.submit(run_job_fn, j) for j in todo]
            try:
                for fut in as_completed(futures):
                    finish(fut.result())
            except BaseException:
                for f in futures:
                    f.cancel()
                raise


# --- report ---


def run_report(
    *,
    out_dir: str | Path,
    subject: str,
    split: str,
    horizon: int,
    workers: int,
    registry: Registry,
    roster: Roster,
    params: SideEffectParams,
    params_payload: dict,
    prod_ckpt: str | None = None,
    orderer_ckpt: str | None = None,
    preregistration: str | None = None,
    access_log: str | Path = SEALED_ACCESS_LOG,
    sizes: Sequence[int] | None = None,
    families: Sequence[str] | None = None,
    max_worlds: int | None = None,
    episodes_per_world: int | None = None,
    source_commit: str = "",
    dry_run: bool = False,
    run_job_fn: Callable[[Job], dict[str, str]] = run_job,
    log: Callable[[str], None] = print,
) -> tuple[GateReport, dict[str, int]]:
    """Run (or resume) a report into `out_dir`; return the report and job counts.

    A real report (`dry_run=False`) needs a frozen roster, a screened registry, a full 40-hex
    source commit and no world/episode filters; anything less must be declared a dry run, which
    `spec.json` and `RESULT.md` mark as not an acceptance read. Everything that can refuse runs
    before any file in `out_dir` is touched."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    pair = parse_subject(subject)
    if pair is None and not (prod_ckpt and orderer_ckpt):
        raise ValueError("the portable subject needs --prod-ckpt and --orderer-ckpt")
    if not dry_run:
        problems = [msg for bad, msg in (
            (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
            (bool(registry.admission_problems(split)),
             "; ".join(registry.admission_problems(split))),
            (len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit),
             "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
            (any(f is not None for f in (sizes, families, max_worlds, episodes_per_world)),
             "world/episode filters are set"),
        ) if bad]
        if problems:
            raise ValueError("not a valid real report (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    worlds = select_worlds(registry, split, sizes=sizes, families=families, max_worlds=max_worlds)
    if not worlds:
        raise ValueError("no gate worlds selected")
    checkpoints = ({"producer": prod_ckpt, "orderer": orderer_ckpt} if pair is None else {})
    candidate_replicates = roster.replicates if pair and RANDOM_POLICY in pair else 1
    grid = job_grid(worlds, episodes_per_world)
    source = source_fingerprint(source_commit or None)
    pins = {
        "subject": subject, "split": split, "horizon": horizon, "dry_run": dry_run,
        "registry_version": registry.version, "roster_version": roster.version,
        "params_version": params.version,
        "checkpoints": {k: sha256_file(v) for k, v in checkpoints.items()},
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "candidate_replicates": candidate_replicates, "grid": grid,
    }
    out = Path(out_dir)
    expected = expected_keys(grid, roster, candidate_replicates)
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    if split == SEALED:
        # Before any episode: refuses without a matching pre-registration, logs the access.
        check_sealed_access(
            registry, [w.id for w in worlds],
            preregistration=load_preregistration(preregistration) if preregistration else None,
            checkpoints=checkpoints, roster_version=roster.version,
            params_version=params.version, log_path=access_log)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "mode": "report",
        "kind": "dry-run" if dry_run else "report", "pins": pins,
        "roster": roster.payload() | {"content_hash": roster.version},
        "side_effect_params": params_payload,
        "inputs": {k: {"path": str(v), "sha256": pins["checkpoints"][k]}
                   for k, v in checkpoints.items()},
        "source": source,
    })
    jobs = build_jobs(worlds, roster, subject, episodes_per_world=episodes_per_world)
    init = (horizon, prod_ckpt if pair is None else None, orderer_ckpt if pair is None else None)
    episodes = out / "episodes.csv"
    _execute(out, spec, done, jobs, expected, workers, init, run_job_fn, log)
    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=True)
    report = compute_report(rows, roster, params, registry.version)
    write_report_files(out, report)
    spec["status"] = "complete"
    spec["result"] = {"overall": report.overall, "failing_clauses": list(report.failing_clauses),
                      "non_gating_worlds": list(report.non_gating_worlds)}
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "",
            overall=report.overall.upper() + (" (dry run)" if dry_run else ""),
            failing=", ".join(report.failing_clauses) or "none", roster=roster.version[:12],
            registry=registry.version[:12], params=params.version[:12], split=split,
            subject=subject, driver=DRIVER))
    log(f"[gate] {'DRY RUN ' if dry_run else ''}verdict {report.overall}; failing clauses: "
        f"{', '.join(report.failing_clauses) or 'none'}")
    return report, {"jobs": len(jobs), "skipped": len(done), "run": len(jobs) - len(done)}


# --- recompute ---


def recompute(out_dir: str | Path) -> tuple[GateReport, bool]:
    """Rebuild the report from `episodes.csv` and the payloads embedded in `spec.json` (hash-checked
    against the pinned versions), after checking the CSV holds exactly the pinned job grid; compare
    it with the recorded `verdict.json`. Returns (report, matches)."""
    out = Path(out_dir)
    spec = json.loads((out / "spec.json").read_text())
    pins = spec["pins"]
    roster = roster_from_dict(spec["roster"])
    params = params_from_dict(spec["side_effect_params"])
    for what, got, want in (("roster", roster.version, pins["roster_version"]),
                            ("params", params.version, pins["params_version"])):
        if got != want:
            raise ValueError(f"embedded {what} hashes to {got}, spec pins {want}")
    rows = read_rows(out / "episodes.csv")
    grid = pins["grid"]
    check_rows(rows, grid, expected_keys(grid, roster, pins["candidate_replicates"]),
               complete=True)
    report = compute_report(rows, roster, params, pins["registry_version"])
    recorded = (out / "verdict.json").read_text()
    return report, report.to_json() == recorded


# --- screen (tasks 8.2 / 10.1) ---

HEADROOM_FAILED = 3  # exit status: a family × size failed D10; no frozen files written


@dataclass(frozen=True)
class ScreenOutcome:
    """Everything screening decides, a pure function of roster-only rows (`screen_outcome`)."""

    registry: Registry  # screened, headroom recorded
    headroom: tuple[Headroom, ...]
    passed: bool
    roster: Roster | None  # frozen (and pruned if asked); None when headroom failed
    calibration: Calibration | None
    params_json: str | None

    def report(self) -> dict[str, object]:
        sc = self.registry.screening
        assert sc is not None
        cal = self.calibration
        return {
            "registry_version": self.registry.version,
            "passed_headroom": self.passed,
            "headroom": [h.to_dict() for h in self.headroom],
            "replacements": [r.to_dict() for r in sc.replacements],
            "baseline_means": dict(sc.baseline_means),
            "roster_version": None if self.roster is None else self.roster.version,
            "pruned": (list(self.roster.pruning.dropped)
                       if self.roster is not None and self.roster.pruning else []),
            "x": None if cal is None else cal.x, "y": None if cal is None else cal.y,
            "q": None if cal is None else cal.q,
            "calibration_evidence": None if cal is None else cal.evidence(),
        }


def screen_outcome(scope: Registry, roster: Roster, rows: Sequence[EpisodeRow], *, q: float,
                   prune: bool, prune_sizes: Sequence[int] = GATE_SIZES) -> ScreenOutcome:
    """Zero-baseline replacement -> headroom -> pruning -> X/Y, in that order (module docstring).
    Headroom and X/Y read only the final development gate worlds (after replacement); reserves
    feed replacement only."""
    screened = screen_zero_baselines(scope, roster, rows, splits=(DEVELOPMENT,))
    final = {e.id: e for e in screened.gate_worlds(DEVELOPMENT)}
    final_rows = [r for r in rows if r.world in final]
    bests = episode_bests(final_rows, roster)
    hr = tuple(headroom(bests, {w: e.family for w, e in final.items()},
                        screened.saturation_ceiling))
    screened = record_headroom(screened, hr)
    if not all(h.passed for h in hr):
        return ScreenOutcome(screened, hr, False, None, None, None)
    kept = roster
    if prune:
        dev = {}
        for r in final_rows:
            dev.setdefault((r.world, r.size, r.episode, r.subject), []).append(r.units)
        kept = prune_roster(roster, [DevRow(w, DEVELOPMENT, size, ep, m, fmean(v))
                                     for (w, size, ep, m), v in sorted(dev.items())],
                            sizes=prune_sizes)
        final_rows = [r for r in final_rows if r.subject in set(kept.member_ids)]
    frozen = freeze_roster(kept)
    cal = calibrate_x_y(final_rows, kept, q)
    params = side_effect_params_json(
        cal.x, cal.y, q, name="side-effect-params-v1", rule=CALIBRATION_RULE, method=cal.method,
        evidence=cal.evidence() | {"roster_version": frozen.version,
                                   "registry_version": screened.version})
    return ScreenOutcome(screened, hr, True, frozen, cal, params)


def run_screen(
    *,
    out_dir: str | Path,
    horizon: int,
    workers: int,
    registry: Registry,
    roster: Roster,
    q: float = 0.05,
    prune: bool = False,
    reserves: bool = True,
    sizes: Sequence[int] | None = None,
    families: Sequence[str] | None = None,
    max_worlds: int | None = None,
    episodes_per_world: int | None = None,
    source_commit: str = "",
    dry_run: bool = False,
    run_job_fn: Callable[[Job], dict[str, str]] = run_job,
    log: Callable[[str], None] = print,
) -> tuple[ScreenOutcome, int]:
    """Run (or resume) the roster on the development gate worlds and their reserves, then
    `screen_outcome`; write the screening report and, if headroom passes, `frozen/`. Returns the
    outcome and the exit status (0, or `HEADROOM_FAILED`). A real screen needs an unscreened
    registry, a 40-hex source commit, reserves and no filters; a dry run screens the selected
    worlds as a sub-registry."""
    if registry.screening is not None:
        raise ValueError(f"registry {registry.version[:12]} is already screened")
    if not dry_run:
        problems = [msg for bad, msg in (
            (len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit),
             "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
            (any(f is not None for f in (sizes, families, max_worlds, episodes_per_world)),
             "world/episode filters are set"),
            (not reserves, "reserves are skipped"),
            (bool(registry.coverage_problems()), "; ".join(registry.coverage_problems())),
        ) if bad]
        if problems:
            raise ValueError("not a valid real screen (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    gate = select_worlds(registry, DEVELOPMENT, sizes=sizes, families=families,
                         max_worlds=max_worlds)
    if not gate:
        raise ValueError("no development gate worlds selected")
    groups = {e.group for e in gate}
    extra = [e for e in registry.entries
             if reserves and e.status == RESERVE and e.group in groups]
    if dry_run:  # a sub-registry of exactly what runs (selected worlds, truncated episodes)
        def cut(e: WorldEntry) -> WorldEntry:
            return replace(e, episode_seeds=e.episode_seeds[:episodes_per_world]
                           if episodes_per_world else e.episode_seeds)

        scope = Registry(name=f"{registry.name}-dry-run", seed_ranges=registry.seed_ranges,
                         entries=tuple(cut(e) for e in (*gate, *extra)),
                         saturation_ceiling=registry.saturation_ceiling)
    else:
        scope = registry
    worlds = list(gate) + extra
    grid = job_grid(worlds, episodes_per_world)
    source = source_fingerprint(source_commit or None)
    pins = {
        "mode": "screen", "horizon": horizon, "dry_run": dry_run, "q": q, "prune": prune,
        "registry_version": registry.version, "scope_version": scope.version,
        "roster_version": roster.version,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "candidate_replicates": 0, "grid": grid,
    }
    out = Path(out_dir)
    expected = expected_keys(grid, roster, 0)
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "mode": "screen",
        "kind": "dry-run" if dry_run else "screen", "pins": pins,
        "roster": roster.payload() | {"content_hash": roster.version}, "source": source,
    })
    jobs = build_jobs(worlds, roster, None, episodes_per_world=episodes_per_world)
    _execute(out, spec, done, jobs, expected, workers, (horizon, None, None), run_job_fn, log)

    rows = read_rows(out / "episodes.csv")
    check_rows(rows, grid, expected, complete=True)
    prune_sizes = GATE_SIZES if not dry_run else sorted({e.size for e in gate})
    outcome = screen_outcome(scope, roster, rows, q=q, prune=prune, prune_sizes=prune_sizes)
    report = outcome.report()
    (out / "screening.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_table(out / "headroom.csv", [h.to_dict() for h in outcome.headroom])
    if outcome.passed:
        frozen = out / "frozen"
        frozen.mkdir(exist_ok=True)
        save_registry(outcome.registry, frozen / "worlds_v1.json")
        assert outcome.roster is not None and outcome.params_json is not None
        save_roster(outcome.roster, frozen / "roster_v1.json")
        (frozen / "side_effect_params_v1.json").write_text(outcome.params_json)
        params_hash = json.loads(outcome.params_json)["content_hash"]
        spec["frozen"] = {"worlds_v1.json": outcome.registry.version,
                          "roster_v1.json": outcome.roster.version,
                          "side_effect_params_v1.json": params_hash}
    spec["status"] = "complete" if outcome.passed else "headroom-failed"
    spec["result"] = report
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(SCREEN_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "",
            status="HEADROOM PASSED" if outcome.passed else "HEADROOM FAILED (no frozen files)",
            driver=DRIVER))
    log(f"[gate] screen {'(dry run) ' if dry_run else ''}"
        f"{'passed' if outcome.passed else 'FAILED'} headroom; "
        + ("" if outcome.calibration is None else
           f"X={outcome.calibration.x!r} Y={outcome.calibration.y!r}"))
    return outcome, (0 if outcome.passed else HEADROOM_FAILED)


SCREEN_SKELETON = """# {title}

{banner}**Screening: {status}** (see `screening.json`, `headroom.csv`, `frozen/`)

Produced by `python -m {driver} --mode screen` (see `spec.json`).

## Result

TODO: admitted families, replacements, pruned members, X and Y, in plain English.

## Conclusions

TODO
"""


# --- CLI ---


def _csv_ints(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",")] if s else None


def _csv_strs(s: str | None) -> list[str] | None:
    return s.split(",") if s else None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tokenomics.eval.gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("report", "recompute", "screen"), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subject", default=PORTABLE,
                    help="'portable' (needs --prod-ckpt, --orderer-ckpt) or "
                         "'baseline:<producing>/<ordering>'")
    ap.add_argument("--prod-ckpt")
    ap.add_argument("--orderer-ckpt")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--roster", default=str(DEFAULT_ROSTER))
    ap.add_argument("--params", help="side-effect parameter file (X, Y, q, content hash)")
    ap.add_argument("--split", choices=SPLITS, default="development")
    ap.add_argument("--preregistration", help="manifest required for the sealed split")
    ap.add_argument("--access-log", default=str(SEALED_ACCESS_LOG))
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sizes", help="dry-run filter, e.g. 12")
    ap.add_argument("--families", help="dry-run filter, e.g. dense,sparse_wide")
    ap.add_argument("--max-worlds", type=int)
    ap.add_argument("--max-members", type=int, help="dry run: keep the first N roster members")
    ap.add_argument("--episodes-per-world", type=int)
    ap.add_argument("--replicates", type=int, help="dry run: override roster R")
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--verify-source-commit", action="store_true")
    ap.add_argument("--q", type=float, default=0.05, help="screen: X/Y calibration quantile (D4)")
    ap.add_argument("--prune", action="store_true", help="screen: apply the never-argmax rule")
    ap.add_argument("--no-reserves", action="store_true", help="screen dry run: skip reserves")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke run: allows filters, an unfrozen roster, an unscreened registry and "
                         "no source commit; marked as not an acceptance read")
    args = ap.parse_args(argv)

    if args.mode == "recompute":
        report, ok = recompute(args.out_dir)
        print(f"[gate] recomputed verdict {report.overall}; "
              f"{'MATCH' if ok else 'MISMATCH'} with {Path(args.out_dir) / 'verdict.json'}")
        return 0 if ok else 1

    if args.mode == "report" and not args.params:
        ap.error("--mode report needs --params (the side-effect parameter file)")
    if args.no_reserves and not args.dry_run:
        ap.error("--no-reserves is a dry-run option")
    source_commit = args.source_commit
    if args.verify_source_commit:
        from .provenance import resolve_verified_commit

        source_commit = resolve_verified_commit()
    if (args.max_members or args.replicates) and not args.dry_run:
        ap.error("--max-members / --replicates change the roster: pass --dry-run")
    roster = load_roster(args.roster)
    if args.max_members or args.replicates:
        members = roster.members[: args.max_members] if args.max_members else roster.members
        roster = replace(roster, name=f"{roster.name}-dry-run", members=members,
                         replicates=args.replicates or roster.replicates, frozen=False,
                         pruning=None)
    if args.mode == "screen":
        _, status = run_screen(
            out_dir=args.out_dir, horizon=args.horizon, workers=args.workers,
            registry=load_registry(args.registry), roster=roster, q=args.q, prune=args.prune,
            reserves=not args.no_reserves, sizes=_csv_ints(args.sizes),
            families=_csv_strs(args.families), max_worlds=args.max_worlds,
            episodes_per_world=args.episodes_per_world, source_commit=source_commit,
            dry_run=args.dry_run)
        return status
    params_payload = json.loads(Path(args.params).read_text())
    params = load_side_effect_params(args.params)
    run_report(
        out_dir=args.out_dir, subject=args.subject, split=args.split, horizon=args.horizon,
        workers=args.workers, registry=load_registry(args.registry), roster=roster,
        params=params, params_payload=params_payload, prod_ckpt=args.prod_ckpt,
        orderer_ckpt=args.orderer_ckpt, preregistration=args.preregistration,
        access_log=args.access_log, sizes=_csv_ints(args.sizes),
        families=_csv_strs(args.families), max_worlds=args.max_worlds,
        episodes_per_world=args.episodes_per_world, source_commit=source_commit,
        dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
