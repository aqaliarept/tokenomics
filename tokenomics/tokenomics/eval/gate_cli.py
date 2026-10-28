"""Gate driver: `python -m tokenomics.eval.gate` (design D6, D11, add-measurement-gate-harness;
gate-harness "Identical demand across observation contracts", "Gate report contents", "Verdict is
reproducible from rows", "Provenance binding", "Sealed worlds need a pre-registration").

Modes:
- `report`: run the subject (`CANDIDATE` rows) and every roster member (R replicates for a random
  member, action seed `seeding.replicate_seed`) on the registry's gate worlds of one split; write the
  EXP-dir layout (`spec.json`, `episodes.csv`, `worlds.csv`, `regimes.csv`, `verdict.json`, a
  `RESULT.md` skeleton). All arithmetic is `gate.compute_report`; this module only produces rows.
  On a world whose config has shocks, each subject episode also runs its shock-free twin
  (subject `gate.TWIN`, add-shocks-i-drift D10); twin rows never enter the verdict and feed only
  `resilience.csv` (`gate.resilience_table`), written with the report.
- `recompute`: rebuild the report from `episodes.csv` plus the roster / parameter payloads embedded
  in `spec.json` (hash-checked against the named versions) and compare it with `verdict.json`;
  rebuild `resilience.csv` too and compare it with the recorded file (a report from before it
  existed, e.g. EXP-0045, matches when the rebuilt table is empty).
- `screen`: roster only, on the development gate worlds, plus the reserves they need. By
  default (`--reserves on-demand`, task 8.6) reserves run in waves only when a gate world has a
  zero baseline. `--reserves eager` runs every reserve up front; both give identical frozen
  files. In order: zero-baseline replacement (registry screening, development split) -> headroom per family × size on the final
  world set (D10; any failure stops with exit 3 and writes no frozen files) -> optional pruning
  (`--prune`, roster.prune_roster) -> X/Y calibration (D4) over the final worlds' episodes with the
  (pruned) roster -> candidate frozen files `frozen/worlds_v<n>.json` (screened, headroom
  recorded), `frozen/roster_v<n>.json` (frozen), `frozen/side_effect_params_v<n>.json`, plus
  `screening.json`. The tag `v<n>` comes from the input registry's name (`worlds-v2` -> v2;
  add-shocks-i-drift D10). From v2 on, all three files record `screened_on` = the draft registry
  hash, and a real report admits a v2+ set only when the three agree (`version_set_problem`).
- `probe` (task 10.2): baseline-only headroom of the development gate worlds of `--families`
  (default dense) at each `--scales` capacity scale, with an optional `--members` subset; writes
  `scale-<s>/` per scale and `probe_headroom.csv`. Calibration evidence for retuning a family
  (D10), never an acceptance read.

Jobs fan out over a spawn-context `ProcessPoolExecutor`, one process per worker, BLAS pinned to one
thread: inference is batch-1 and rollouts are sequential Python (CLAUDE.md "Parallel work").
`pin_blas` forces (not defaults) the BLAS env vars to 1 in the parent before the pool spawns, so a
fresh worker's numpy/torch start single-threaded whatever the shell exported; it also resets an
already-loaded OpenBLAS (numpy's bundled build, via ctypes) and torch in-process, which covers the
`--workers 1` path where numpy is imported before any flag is parsed. Each finished row is appended to `episodes.csv` at once; a rerun on the same `--out-dir`
checks the run pins in `spec.json`, drops a torn last line, skips rows already present and runs the
rest. Demand and shock digests are compared as rows arrive, so a contract mismatch stops the run
early; the pure layer re-checks them. A world naming the test-only `shocks-i-smoke` profile (D7)
runs only in a dry run. Contract failures (`eval/contract.py`, D10) are recorded in the
append-only `contract_failures.csv`; any row, or a missing episode row, makes the run `invalid`
(exit `CONTRACT_INVALID`) with no verdict.
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
from typing import Callable, Collection, Mapping, Sequence

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .gate import (
    CALIBRATION_RULE,
    CANDIDATE,
    CSV_COLUMNS,
    DEFAULT_PARAMS,
    EMPTY_SHOCK_DIGEST,
    TWIN,
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
    is_twin,
    params_from_dict,
    resilience_table,
    resilience_to_csv,
    row_from_dict,
    row_from_stats,
    row_to_dict,
    side_effect_params_json,
)
from .resilience import DEFAULT_PARAMS as DEFAULT_RESILIENCE, ResilienceParams, shock_free_twin
from .shock_profiles import load_shock_profile, names_test_only_profile
from .contract import (
    CHECKPOINT,
    DIGEST,
    FAILURES_FILE,
    ContractMonitor,
    ContractRunError,
    append_failure,
    ensure_failures_file,
    failure_from_dict,
    is_failure,
    job_failure,
    read_failures,
    run_failure,
)
from .provenance import environment_manifest, sha256_file, source_fingerprint
from .families import DENSE, NEUTRAL_CAPACITY_SCALE, check_capacity_scale
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
    plan_replacements,
    record_headroom,
    save_registry,
    screen_zero_baselines,
    world_baseline_means,
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
    name_version_tag,
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
                  families: Sequence[str] | None = None, regimes: Sequence[str] | None = None,
                  max_worlds: int | None = None) -> list[WorldEntry]:
    worlds = [e for e in registry.gate_worlds(split)
              if (not sizes or e.size in sizes) and (not families or e.family in families)
              and (not regimes or e.regime in regimes)]
    return worlds[:max_worlds] if max_worlds else worlds


def test_only_problem(worlds: Sequence[WorldEntry]) -> str | None:
    """D7: the smoke profile is for dry runs only; a recorded run must use the registered one."""
    bad = sorted(e.id for e in worlds if names_test_only_profile(e.generator))
    return (f"worlds {bad} name a test-only shock profile, which runs only with --dry-run"
            if bad else None)


def twin_params(e: WorldEntry) -> ResilienceParams | None:
    """The resilience window/tolerance when `e`'s config has shocks (its subject episodes get
    twins), else None. From the world's named profile; the D7 defaults when it names none."""
    if e.config().shocks is None:
        return None
    name = e.generator["params"].get("shock_profile")
    return load_shock_profile(name).resilience if name else DEFAULT_RESILIENCE


def build_jobs(worlds: Sequence[WorldEntry], roster: Roster, subject: str | None, *,
               episodes_per_world: int | None = None,
               twin_worlds: Collection[str] = ()) -> list[Job]:
    """Every (world, episode, subject, replicate) the run needs: the candidate once per episode
    (R times when a baseline subject is random; none when `subject` is None, the roster-only screen),
    each member `roster.replicates_for` times; on `twin_worlds`, each candidate job's shock-free
    twin (subject `TWIN`, same policies and seeds)."""
    pair = parse_subject(subject) if subject is not None else None
    jobs: list[Job] = []
    for e in worlds:
        episodes = e.episode_seeds[:episodes_per_world] if episodes_per_world else e.episode_seeds
        for ep in episodes:
            base = {"world": e.id, "size": e.size, "regime": e.regime, "world_seed": e.world_seed,
                    "generator": e.generator, "episode": ep}
            cand: list[Job] = []
            if subject is None:
                pass
            elif pair is None:
                cand.append({**base, "subject": CANDIDATE, "replicate": 0, "producing": PORTABLE,
                             "ordering": PORTABLE, "contract": JOINT_STACK, "action_seed": None})
            else:
                random = RANDOM_POLICY in pair
                for r in range(roster.replicates if random else 1):
                    cand.append({**base, "subject": CANDIDATE, "replicate": r,
                                 "producing": pair[0], "ordering": pair[1],
                                 "contract": contract_for(*pair),
                                 "action_seed": replicate_seed(e.id, ep, r) if random else None})
            jobs += cand
            if e.id in twin_worlds:
                jobs += [{**j, "subject": TWIN} for j in cand]
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
    """Run one episode and return its row as CSV strings (`gate.row_to_dict`), or, when anything
    in the job raises or a contract check fails (`eval/contract.py`), its failure row
    (`contract.Failure.to_dict`): the job stops and the other jobs go on."""
    try:
        return _run_job(job)
    except Exception as exc:  # noqa: BLE001 - every job failure is recorded, never fatal
        return job_failure(job, exc).to_dict()


def _run_job(job: Job) -> dict[str, str]:
    from ..env import TokenomicsAECEnv
    from .joint_stack import joint_stack_env
    from .runner import run_episode
    from .task_archive import REPRODUCERS

    from ..env.shock_schedule import build_schedule, canonical_events, shock_digest

    configs = _W["configs"]
    twin = is_twin(job["subject"])
    key = (job["world"], twin)
    if key not in configs:
        g = job["generator"]
        cfg = REPRODUCERS[g["generator"]](job["world_seed"], **g.get("params", {}))
        cfg.validate()
        configs[key] = shock_free_twin(cfg) if twin else cfg
    cfg = configs[key]
    horizon = _W["horizon"]
    env = (joint_stack_env(cfg, horizon) if job["contract"] == JOINT_STACK
           else TokenomicsAECEnv(config=cfg, horizon=horizon))
    prod, order = _policies(job)
    monitor = ContractMonitor(env)
    run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"],
                monitor=monitor)
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=horizon)
    schedule = None
    if job["subject"] == CANDIDATE or twin:  # resilience inputs: candidate and twin rows only
        events = build_schedule(cfg, job["episode"], horizon)
        if shock_digest(events) != env.episode_stats["shock_digest"]:
            raise AssertionError(f"{job['world']}/{job['episode']}: rebuilt shock schedule "
                                 "differs from the episode's")  # internal invariant
        schedule = json.dumps(canonical_events(events), sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True)
    row = row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                         job["subject"], job["replicate"], env.episode_stats,
                         shock_schedule=schedule)
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


def write_report_files(out: Path, report: GateReport, resilience: str) -> None:
    _write_table(out / "worlds.csv", world_table(report))
    _write_table(out / "regimes.csv", regime_table(report))
    (out / "verdict.json").write_text(report.to_json())
    (out / "resilience.csv").write_text(resilience)


def grid_resilience(grid: Mapping[str, Mapping]) -> dict[str, ResilienceParams]:
    """world -> resilience params, for the pinned worlds with twins."""
    return {w: ResilienceParams(window=g["twin"]["window"], tolerance=g["twin"]["tolerance"])
            for w, g in grid.items() if "twin" in g}


RESULT_SKELETON = """# {title}

{banner}**Verdict: {overall}** (failing clauses: {failing})

Roster {roster}, registry {registry}, side-effect parameters {params}; split {split}, subject
`{subject}`. Produced by `python -m {driver} --mode report` (see `spec.json`).

## Question

TODO

## Result

TODO: plain-English reading of `verdict.json`, `worlds.csv` and `regimes.csv`; per-shock
resilience (not an acceptance clause) in `resilience.csv`.

## Conclusions

TODO
"""
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered worlds/episodes/members, an "
              "unfrozen roster, an unscreened registry or no source commit; see `spec.json`.\n\n")


# --- the expected job grid (what a complete episodes.csv must hold) ---

Key = tuple[str, int, str, int]


def job_grid(worlds: Sequence[WorldEntry], episodes_per_world: int | None, *,
             twins: bool = False) -> dict[str, dict]:
    """world -> {size, regime, episodes}: the episodes a report runs, pinned in `spec.json` so a
    resume and a recompute can check the CSV against it without the registry. With `twins` (a
    report), a world whose config has shocks also pins `twin` = its resilience window and
    tolerance: its candidate episodes get twins and resilience rows. The key is absent
    otherwise, so a grid without shocked configs is unchanged."""
    grid = {}
    for w in worlds:
        g: dict = {"size": w.size, "regime": w.regime,
                   "episodes": list(w.episode_seeds[:episodes_per_world] if episodes_per_world
                                    else w.episode_seeds)}
        tp = twin_params(w) if twins else None
        if tp is not None:
            g["twin"] = {"window": tp.window, "tolerance": tp.tolerance}
        grid[w.id] = g
    return grid


def expected_keys(grid: Mapping[str, Mapping], roster: Roster, candidate_replicates: int) -> set[Key]:
    keys: set[Key] = set()
    for world, g in grid.items():
        for ep in g["episodes"]:
            keys.update((world, ep, CANDIDATE, r) for r in range(candidate_replicates))
            if "twin" in g:
                keys.update((world, ep, TWIN, r) for r in range(candidate_replicates))
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
    if old_spec is not None and "pins" not in old_spec and old_spec.get("status") == INVALID:
        # the spec of a run that failed before any job (a checkpoint, D10): nothing was pinned
        # or run, so this run pins afresh; contract_failures.csv keeps the failure, so the
        # run still ends invalid
        old_spec = None
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
    and shock digests per (world, episode) as rows arrive. A twin pairs on demand only and must
    have an empty shock schedule. A failed job (`run_job` returns its failure row: every
    exception inside a job is caught there) is appended to `contract_failures.csv` and the
    other jobs go on; a digest mismatch of any kind is recorded as one `run` row, sets
    `spec.json` invalid and still stops the run. An
    exception out of `run_job_fn` itself or a dead worker is a crash of the run, not of a job:
    it propagates and the run resumes as before."""
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    demand: dict[tuple[str, int], tuple[str, str]] = {}
    shocks: dict[tuple[str, int], tuple[str, str]] = {}

    def note_digest(row: EpisodeRow) -> None:
        where = (f"world {row.world} episode {row.episode}: {row.subject} replicate "
                 f"{row.replicate}")
        checks = [("demand digest", demand, row.demand_digest)]
        if is_twin(row.subject):
            if row.shock_events != 0 or row.shock_digest != EMPTY_SHOCK_DIGEST:
                raise DigestMismatch(f"{where}: the shock-free twin has {row.shock_events} "
                                     f"shock events (shock digest {row.shock_digest})")
        else:
            checks.append(("shock digest", shocks, row.shock_digest))
        for what, seen, value in checks:
            ref = seen.setdefault((row.world, row.episode), (value, row.subject))
            if value != ref[0]:
                raise DigestMismatch(f"{where} saw {what} {value}, {ref[1]} saw {ref[0]}")

    def pair(row: EpisodeRow) -> None:
        """Every pairing failure, whatever its variant: one `run` row, status invalid, raise."""
        try:
            note_digest(row)
        except DigestMismatch as exc:
            append_failure(out, run_failure(DIGEST, exc))
            spec["status"] = INVALID
            spec["result"] = {"status": INVALID, "contract_failures": len(read_failures(out))}
            _write_spec(out, spec)
            raise

    for r in done:
        pair(r)
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    todo = [j for j in jobs
            if job_key(j["world"], j["episode"], j["subject"], j["replicate"]) not in done_keys]
    log(f"[gate] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    def finish(d: dict[str, str]) -> None:
        if is_failure(d):
            f = failure_from_dict(d)
            append_failure(out, f)
            log(f"[gate] CONTRACT FAILURE {f.world} ep={f.episode} {f.subject} "
                f"r{f.replicate}: {f.check} {f.error_type}")
            return
        row = row_from_dict(d)
        pair(row)
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


def version_set_problem(registry: Registry, roster: Roster,
                        params: SideEffectParams) -> str | None:
    """Why a real report may not pair these three files (add-shocks-i-drift D8 "Mixed versions",
    gate-harness "Mixed versions refused"); None = admitted. A registry v2+ set is admitted only
    when the registry, roster and parameters record the same `screened_on`, i.e. come from one
    screen of one draft. v1 files record none and pair only with each other, so a v1 registry
    with a v2+ roster or parameter file is refused too."""
    files = (("registry", registry.name, registry.version, registry.screened_on),
             ("roster", roster.name, roster.version, roster.screened_on),
             ("parameters", params.name, params.version, params.screened_on))
    tags = (registry.version_tag, roster.version_tag, name_version_tag(params.name))
    for (what, name, version, on), tag in zip(files, tags):
        # parsers refuse this on files; in-memory objects reach here too
        if tag >= 2 and on is None:
            return (f"mixed versions: {what} {name!r} ({version[:12]}) is v{tag} but records no "
                    "screened_on")
    bad = [f for f in files[1:] if f[3] != registry.screened_on]
    if not bad:
        return None

    def show(f: tuple[str, str, str, str | None]) -> str:
        on = "none" if f[3] is None else f[3][:12]
        return f"{f[0]} {f[1]!r} ({f[2][:12]}, screened_on {on})"

    return (f"mixed versions: {', '.join(show(f) for f in bad)} do not pair with "
            f"{show(files[0])}")


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
    regimes: Sequence[str] | None = None,
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
    before any file in `out_dir` is touched.

    Contract failures (`eval/contract.py`) are recorded in `contract_failures.csv`, never
    refused: a checkpoint that does not hash or load is a `run` row and raises
    `ContractRunError` before any job; a failed job is a `job` row and the others finish; a
    digest mismatch is a `run` row and still raises. A run with any failure row, now or from
    an earlier session in the same `out_dir`, or with missing rows, ends `invalid` (spec.json and
    verdict.json) and returns no report."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    pair = parse_subject(subject)
    if pair is None and not (prod_ckpt and orderer_ckpt):
        raise ValueError("the portable subject needs --prod-ckpt and --orderer-ckpt")
    worlds = select_worlds(registry, split, sizes=sizes, families=families, regimes=regimes,
                           max_worlds=max_worlds)
    if not dry_run:
        smoke = test_only_problem(worlds)
        problems = [msg for bad, msg in (
            (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
            (bool(registry.admission_problems(split)),
             "; ".join(registry.admission_problems(split))),
            (len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit),
             "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
            (any(f is not None for f in (sizes, families, regimes, max_worlds,
                                         episodes_per_world)),
             "world/episode filters are set"),
            (smoke is not None, smoke or ""),
        ) if bad]
        mixed = version_set_problem(registry, roster, params)
        if mixed is not None:
            problems.append(mixed)
        if problems:
            raise ValueError("not a valid real report (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    if not worlds:
        raise ValueError("no gate worlds selected")
    checkpoints = ({"producer": prod_ckpt, "orderer": orderer_ckpt} if pair is None else {})
    out = Path(out_dir)
    ckpt_hashes = _checked_checkpoints(out, checkpoints, dry_run, log)
    candidate_replicates = roster.replicates if pair and RANDOM_POLICY in pair else 1
    grid = job_grid(worlds, episodes_per_world, twins=True)
    source = source_fingerprint(source_commit or None)
    pins = {
        "subject": subject, "split": split, "horizon": horizon, "dry_run": dry_run,
        "registry_version": registry.version, "roster_version": roster.version,
        "params_version": params.version,
        "checkpoints": ckpt_hashes,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "candidate_replicates": candidate_replicates, "grid": grid,
    }
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
    jobs = build_jobs(worlds, roster, subject, episodes_per_world=episodes_per_world,
                      twin_worlds={w for w, g in grid.items() if "twin" in g})
    init = (horizon, prod_ckpt if pair is None else None, orderer_ckpt if pair is None else None)
    episodes = out / "episodes.csv"
    counts = {"jobs": len(jobs), "skipped": len(done), "run": len(jobs) - len(done)}
    ensure_failures_file(out)
    try:
        _execute(out, spec, done, jobs, expected, workers, init, run_job_fn, log)
        rows = read_rows(episodes)
        check_rows(rows, grid, expected, complete=False)
        missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
        failures = read_failures(out)
        if failures or missing:
            _finish_invalid(out, spec, len(failures), len(missing), dry_run, log)
            return None, counts
        try:
            report = compute_report(rows, roster, params, registry.version)
        except DigestMismatch as exc:  # the pure re-check; the live one should have caught it
            append_failure(out, run_failure(DIGEST, exc))
            raise
    except DigestMismatch:
        _finish_invalid(out, spec, len(read_failures(out)), None, dry_run, log)
        raise
    resilience = resilience_table(rows, grid_resilience(grid))
    write_report_files(out, report, resilience_to_csv(resilience))
    spec["status"] = "complete"
    spec["result"] = {"overall": report.overall, "failing_clauses": list(report.failing_clauses),
                      "non_gating_worlds": list(report.non_gating_worlds),
                      "resilience_rows": len(resilience), "contract_failures": 0}
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "",
            overall=report.overall.upper() + (" (dry run)" if dry_run else ""),
            failing=", ".join(report.failing_clauses) or "none", roster=roster.version[:12],
            registry=registry.version[:12], params=params.version[:12], split=split,
            subject=subject, driver=DRIVER))
    log(f"[gate] {'DRY RUN ' if dry_run else ''}verdict {report.overall}; failing clauses: "
        f"{', '.join(report.failing_clauses) or 'none'}; contract failures: 0")
    return report, counts


INVALID = "invalid"


def _checked_checkpoints(out: Path, checkpoints: Mapping[str, str | None], dry_run: bool,
                         log: Callable[[str], None]) -> dict[str, str]:
    """sha256 of each checkpoint, then a trial load in the parent (D10 run-level failures):
    a checkpoint that does not hash or load is a `run` row and `ContractRunError`, before any
    job runs. `spec.json` and `verdict.json` then say `invalid`: an existing spec (an earlier
    session in this `out_dir`) keeps its pins and provenance; otherwise a minimal one names
    the inputs."""
    try:
        hashes = {k: sha256_file(v) for k, v in checkpoints.items()}
        if checkpoints:
            from ..train.orderer_scorer import load_scorer
            from ..train.producer_scorer import load_producer_score_model

            load_producer_score_model(checkpoints["producer"])
            load_scorer(checkpoints["orderer"])
    except Exception as exc:  # noqa: BLE001 - recorded, then the run ends invalid
        append_failure(out, run_failure(CHECKPOINT, exc))
        spec_path = out / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {
            "experiment": out.name, "driver": DRIVER, "mode": "report",
            "kind": "dry-run" if dry_run else "report",
            "inputs": {k: {"path": str(v), "sha256": None} for k, v in checkpoints.items()},
            "date": datetime.now(timezone.utc).isoformat()[:10]}
        _finish_invalid(out, spec, len(read_failures(out)), None, dry_run, log)
        raise ContractRunError(f"checkpoint failed to hash or load ({type(exc).__name__}: "
                               f"{exc}); recorded in {out / FAILURES_FILE}") from exc
    return hashes


def invalid_verdict(failures: int, missing: int | None) -> dict[str, object]:
    return {"status": INVALID, "contract_failures": failures, "missing_rows": missing,
            "acceptance_read": False,
            "reason": "contract failures recorded or episode rows missing; no verdict is built "
                      "from partial rows (add-shocks-i-drift D10)"}


def _finish_invalid(out: Path, spec: dict, failures: int, missing: int | None, dry_run: bool,
                    log: Callable[[str], None]) -> None:
    doc = invalid_verdict(failures, missing)
    (out / "verdict.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    spec["status"] = INVALID
    spec["result"] = doc
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(INVALID_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", failures=failures,
            missing="unknown" if missing is None else missing, driver=DRIVER))
    log(f"[gate] INVALID run: {failures} contract failure(s), "
        f"{'unknown' if missing is None else missing} missing row(s); see {FAILURES_FILE}")


INVALID_SKELETON = """# {title}

{banner}**INVALID: not an acceptance read.** {failures} contract failure(s) in
`contract_failures.csv`, {missing} episode row(s) missing. No verdict is built from partial rows.

Produced by `python -m {driver} --mode report` (see `spec.json`).

## Diagnosis

TODO
"""


# --- recompute ---


@dataclass(frozen=True)
class Recomputed:
    """`--mode recompute` outcome. `status` is "valid" (a report was rebuilt and compared) or
    `INVALID` (contract failure rows, or rows of the pinned grid missing: no report is built)."""

    status: str
    report: GateReport | None
    verdict_ok: bool
    resilience_ok: bool
    contract_failures: int
    missing_rows: int

    @property
    def ok(self) -> bool:
        return self.status != INVALID and self.verdict_ok and self.resilience_ok


def recompute(out_dir: str | Path) -> tuple[GateReport | None, bool]:
    """(report, valid and both verdict and resilience match): `recompute_details` folded."""
    r = recompute_details(out_dir)
    return r.report, r.ok


def recompute_details(out_dir: str | Path) -> Recomputed:
    """Rebuild the report from `episodes.csv` and the payloads embedded in `spec.json` (hash-checked
    against the pinned versions), after checking the CSV rows against the pinned job grid; compare
    it with the recorded `verdict.json`, and the rebuilt `resilience.csv` with the recorded one (a
    run without that file, from before it existed, matches only an empty table). A run with any
    `contract_failures.csv` row, or with pinned rows missing, is `INVALID`: never a verdict
    from partial rows (D10)."""
    out = Path(out_dir)
    failures = len(read_failures(out))
    spec = json.loads((out / "spec.json").read_text())
    if "pins" not in spec:
        if failures:  # a run-level failure before any job: nothing was pinned
            return Recomputed(INVALID, None, False, False, failures, 0)
        raise ValueError(f"{out / 'spec.json'} pins nothing")
    pins = spec["pins"]
    roster = roster_from_dict(spec["roster"])
    if roster.version != pins["roster_version"]:
        raise ValueError(f"embedded roster hashes to {roster.version}, spec pins "
                         f"{pins['roster_version']}")
    rows = read_rows(out / "episodes.csv")
    grid = pins["grid"]
    expected = expected_keys(grid, roster, pins["candidate_replicates"])
    check_rows(rows, grid, expected, complete=False)
    missing = len(expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows})
    if failures or missing:  # decided before anything only a verdict needs is read
        return Recomputed(INVALID, None, False, False, failures, missing)
    params = params_from_dict(spec["side_effect_params"])
    if params.version != pins["params_version"]:
        raise ValueError(f"embedded params hashes to {params.version}, spec pins "
                         f"{pins['params_version']}")
    report = compute_report(rows, roster, params, pins["registry_version"])
    recorded = (out / "verdict.json").read_text()
    table = resilience_table(rows, grid_resilience(grid))
    res_path = out / "resilience.csv"
    res_ok = (res_path.read_text() == resilience_to_csv(table) if res_path.exists()
              else not table)
    return Recomputed("valid", report, report.to_json() == recorded, res_ok, 0, 0)


# --- screen (tasks 8.2 / 10.1) ---

HEADROOM_FAILED = 3  # exit status: a family × size failed D10; no frozen files written
CONTRACT_INVALID = 2  # exit status: contract failures recorded (or rows missing); run invalid
RESERVE_MODES = ("on-demand", "eager", "none")


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
                   prune: bool, prune_sizes: Sequence[int] = GATE_SIZES, tag: int = 1,
                   screened_on: str | None = None) -> ScreenOutcome:
    """Zero-baseline replacement -> headroom -> pruning -> X/Y, in that order (module docstring).
    Headroom and X/Y read only the final development gate worlds (after replacement); reserves
    feed replacement only.

    `tag` is the output generation (D10 "Versioned screen outputs"): the parameters are named
    `side-effect-params-v<tag>`. From tag 2 on, `screened_on` (the draft registry hash) is
    recorded in the frozen registry, roster and parameters, and the frozen roster is named
    `roster-v<tag>`; tag 1 records nothing new, so v1 outputs are unchanged."""
    if (tag >= 2) != (screened_on is not None):
        raise ValueError(f"screened_on is recorded from tag v2 on, exactly (tag v{tag}, "
                         f"screened_on {screened_on!r})")
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
    if screened_on is not None:
        screened = replace(screened, screened_on=screened_on)
        frozen = replace(frozen, name=f"roster-v{tag}", screened_on=screened_on)
    cal = calibrate_x_y(final_rows, kept, q)
    params = side_effect_params_json(
        cal.x, cal.y, q, name=f"side-effect-params-v{tag}", rule=CALIBRATION_RULE,
        method=cal.method, screened_on=screened_on,
        evidence=cal.evidence() | {"roster_version": frozen.version,
                                   "registry_version": screened.version})
    return ScreenOutcome(screened, hr, True, frozen, cal, params)


def _check_waves(waves: Sequence[object], *, reserves: str, scope: Registry, registry: Registry,
                 roster: Roster, rows: Sequence[EpisodeRow], grid: Mapping[str, Mapping],
                 episodes_per_world: int | None) -> None:
    """Refuse recorded reserve waves the planner would not have scheduled (review concern 92).
    `spec.json` "waves" sits outside the pins, so an edited or corrupt list would otherwise widen
    the job grid and turn arbitrary rows into expected ones. Wave n (numbered from 2) is valid only
    if every wave before it is complete in the rows and `plan_replacements` over exactly those rows
    gives n's pending reserves with their canonical grid. Only the last wave may be incomplete:
    that is a mid-wave resume."""
    if reserves != "on-demand":
        raise ValueError(f"spec.json records reserve waves but reserves={reserves!r}; "
                         f"refusing to resume")
    cum = dict(grid)
    for i, wave in enumerate(waves):
        n = i + 2
        if not isinstance(wave, dict) or set(wave) != {"wave", "worlds"} or wave["wave"] != n:
            raise ValueError(f"recorded reserve wave #{n} is malformed: {wave!r}; "
                             f"refusing to resume")
        prefix = [r for r in rows if r.world in cum]
        try:
            check_rows(prefix, cum, expected_keys(cum, roster, 0), complete=True)
        except ValueError as exc:
            raise ValueError(f"recorded reserve wave {n} but the waves before it are incomplete "
                             f"({exc}); refusing to resume") from None
        plan = plan_replacements(scope, world_baseline_means(scope, roster, prefix))
        want = job_grid([registry.entry(w) for w in plan.pending], episodes_per_world)
        if not want or wave["worlds"] != want:
            raise ValueError(f"recorded reserve wave {n} {sorted(wave['worlds'])} is not the "
                             f"planned wave {sorted(want)}; refusing to resume")
        cum.update(want)


def run_screen(
    *,
    out_dir: str | Path,
    horizon: int,
    workers: int,
    registry: Registry,
    roster: Roster,
    q: float = 0.05,
    prune: bool = False,
    reserves: str = "on-demand",
    sizes: Sequence[int] | None = None,
    families: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    max_worlds: int | None = None,
    episodes_per_world: int | None = None,
    source_commit: str = "",
    dry_run: bool = False,
    run_job_fn: Callable[[Job], dict[str, str]] = run_job,
    log: Callable[[str], None] = print,
) -> tuple[ScreenOutcome, int]:
    """Run (or resume) the roster on the development gate worlds and the reserves they need, then
    `screen_outcome`; write the screening report and, if headroom passes, `frozen/`. Returns the
    outcome and the exit status (0, or `HEADROOM_FAILED`). A real screen needs an unscreened
    registry, a 40-hex source commit, reserves and no filters; a dry run screens the selected
    worlds as a sub-registry.

    `reserves` (user decision 2026-10-27, task 8.6):
    - "on-demand" (default): wave 1 runs the gate worlds only. Each later wave runs exactly the
      reserves `registry.plan_replacements` reports pending, i.e. the reserves the fixed order is
      certain to consult. Each wave is decided only after the previous one has completed, so worker
      timing cannot change it. The waves stop when the plan has no pending reserve, or raise when
      a group runs out.
    - "eager": every reserve of the selected groups runs up front (the 8.2 behaviour).
    - "none": dry runs only.
    Both modes feed the same `screen_outcome`, whose record keeps only consulted reserves, so they
    give identical frozen files. The waves are recorded in `spec.json` ("waves"); a resume reruns
    the recorded waves first and then continues planning from the rows."""
    if reserves not in RESERVE_MODES:
        raise ValueError(f"reserves must be one of {RESERVE_MODES}, got {reserves!r}")
    if registry.screening is not None:
        raise ValueError(f"registry {registry.version[:12]} is already screened")
    # Output generation from the input registry's name (D10 "Versioned screen outputs").
    tag = registry.version_tag
    screened_on = registry.version if tag >= 2 else None
    gate = select_worlds(registry, DEVELOPMENT, sizes=sizes, families=families, regimes=regimes,
                         max_worlds=max_worlds)
    groups = {e.group for e in gate}
    extra = [e for e in registry.entries
             if reserves != "none" and e.status == RESERVE and e.group in groups]
    if not dry_run:
        smoke = test_only_problem([*gate, *extra])
        problems = [msg for bad, msg in (
            (len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit),
             "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
            (any(f is not None for f in (sizes, families, regimes, max_worlds,
                                         episodes_per_world)),
             "world/episode filters are set"),
            (reserves == "none", "reserves are skipped"),
            (bool(registry.coverage_problems()), "; ".join(registry.coverage_problems())),
            (tag >= 2 and roster.version_tag != tag,
             f"registry {registry.name!r} is v{tag} but roster {roster.name!r} is "
             f"v{roster.version_tag}"),
            (smoke is not None, smoke or ""),
        ) if bad]
        if problems:
            raise ValueError("not a valid real screen (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    if not gate:
        raise ValueError("no development gate worlds selected")
    if dry_run:  # a sub-registry of exactly what runs (selected worlds, truncated episodes)
        def cut(e: WorldEntry) -> WorldEntry:
            return replace(e, episode_seeds=e.episode_seeds[:episodes_per_world]
                           if episodes_per_world else e.episode_seeds)

        scope = Registry(name=f"{registry.name}-dry-run", seed_ranges=registry.seed_ranges,
                         entries=tuple(cut(e) for e in (*gate, *extra)),
                         saturation_ceiling=registry.saturation_ceiling,
                         shock_profiles=registry.shock_profiles)
    else:
        scope = registry
    # wave 1: gate worlds (plus every reserve when eager); the pinned grid is wave 1's.
    worlds = list(gate) + (extra if reserves == "eager" else [])
    grid = job_grid(worlds, episodes_per_world)
    source = source_fingerprint(source_commit or None)
    pins = {
        "mode": "screen", "horizon": horizon, "dry_run": dry_run, "q": q, "prune": prune,
        "reserves": reserves,
        "registry_version": registry.version, "scope_version": scope.version,
        "roster_version": roster.version,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "candidate_replicates": 0, "grid": dict(grid),  # a copy: later waves extend `grid`
    }
    out = Path(out_dir)
    # Reserve waves already scheduled by an earlier run of this screen: their worlds join the
    # grid before the resume check, so their rows are not orphans. A pin mismatch is refused by
    # _prepare_resume anyway, so the waves of a different run are never read. The waves are not
    # pinned, so each one is replayed from the rows before any job runs (`_check_waves`).
    prior = (json.loads((out / "spec.json").read_text())
             if (out / "spec.json").exists() else {})
    waves: list[dict] = prior.get("waves", []) if prior.get("pins") == pins else []
    if waves:
        _check_waves(waves, reserves=reserves, scope=scope, registry=registry, roster=roster,
                     rows=read_rows(out / "episodes.csv"), grid=grid,
                     episodes_per_world=episodes_per_world)
    for wave in waves:
        worlds += [registry.entry(w) for w in wave["worlds"]]
        grid.update(wave["worlds"])
    expected = expected_keys(grid, roster, 0)
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "mode": "screen",
        "kind": "dry-run" if dry_run else "screen", "pins": pins,
        "roster": roster.payload() | {"content_hash": roster.version}, "source": source,
        "waves": [],
    })
    while True:
        jobs = build_jobs(worlds, roster, None, episodes_per_world=episodes_per_world)
        _execute(out, spec, done, jobs, expected, workers, (horizon, None, None), run_job_fn,
                 log)
        _stop_if_failed(out, spec)
        rows = read_rows(out / "episodes.csv")
        check_rows(rows, grid, expected, complete=True)
        if reserves != "on-demand":
            break
        # Decided only now, after every scheduled job finished: deterministic in the rows.
        plan = plan_replacements(scope, world_baseline_means(scope, roster, rows))
        if not plan.pending:
            break
        new = [registry.entry(w) for w in plan.pending]
        wave_grid = job_grid(new, episodes_per_world)
        spec["waves"] = spec.get("waves", []) + [
            {"wave": len(spec.get("waves", [])) + 2, "worlds": wave_grid}]
        _write_spec(out, spec)
        log(f"[gate] reserve wave {spec['waves'][-1]['wave']}: {sorted(wave_grid)}")
        worlds += new
        grid.update(wave_grid)
        expected = expected_keys(grid, roster, 0)
        done = rows
    prune_sizes = GATE_SIZES if not dry_run else sorted({e.size for e in gate})
    outcome = screen_outcome(scope, roster, rows, q=q, prune=prune, prune_sizes=prune_sizes,
                             tag=tag, screened_on=screened_on)
    report = outcome.report()
    (out / "screening.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_table(out / "headroom.csv", [h.to_dict() for h in outcome.headroom])
    if outcome.passed:
        frozen = out / "frozen"
        frozen.mkdir(exist_ok=True)
        names = {k: f"{k}_v{tag}.json" for k in ("worlds", "roster", "side_effect_params")}
        save_registry(outcome.registry, frozen / names["worlds"])
        assert outcome.roster is not None and outcome.params_json is not None
        save_roster(outcome.roster, frozen / names["roster"])
        (frozen / names["side_effect_params"]).write_text(outcome.params_json)
        params_hash = json.loads(outcome.params_json)["content_hash"]
        spec["frozen"] = {names["worlds"]: outcome.registry.version,
                          names["roster"]: outcome.roster.version,
                          names["side_effect_params"]: params_hash}
        if screened_on is not None:
            spec["screened_on"] = screened_on
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


# --- dense-scale probe (task 10.2) ---


def scaled_entry(e: WorldEntry, scale: float) -> WorldEntry:
    """`e` with its generator ref rewritten to `scale` (canonical form: omitted when neutral)."""
    params = {k: v for k, v in e.generator["params"].items() if k != "capacity_scale"}
    if check_capacity_scale(scale) != NEUTRAL_CAPACITY_SCALE:
        params["capacity_scale"] = scale
    return replace(e, generator={"generator": e.generator["generator"], "params": params})


def baseline_setters(rows: Sequence[EpisodeRow], roster: Roster, family: str) -> list[str]:
    """Members that set the per-episode hindsight baseline on `family`'s worlds in a finished
    screen (`episode_bests` argmax), sorted. 10.3 probes with exactly these members: on the
    screen's own worlds and scale they reproduce its baseline, because the max over a subset that
    holds every argmax member equals the max over the full roster."""
    return sorted({b.member for b in episode_bests(rows, roster)
                   if f"-{family}-" in b.world})


def run_probe(
    *,
    out_dir: str | Path,
    horizon: int,
    workers: int,
    registry: Registry,
    roster: Roster,
    scales: Sequence[float],
    families: Sequence[str] = (DENSE,),
    sizes: Sequence[int] | None = None,
    max_worlds: int | None = None,
    episodes_per_world: int | None = None,
    source_commit: str = "",
    run_job_fn: Callable[[Job], dict[str, str]] = run_job,
    log: Callable[[str], None] = print,
) -> list[Headroom]:
    """Measure headroom at several capacity scales in one session (design D10, task 10.2): for
    each scale, rewrite the selected development gate worlds' generator refs to that scale, run
    the (sub)roster, and apply the headroom rule. Baseline-only calibration evidence, never an
    acceptance read: no replacement, pruning, X/Y or frozen files. One subdir per scale
    (`scale-<s>/`: spec.json, episodes.csv, headroom.csv) resumes on its own pins; the per-scale
    table is `probe_headroom.csv`. A probe sees fewer members, replicates or episodes than the
    gate, so its medians estimate the screen's; the rescreen stays the binding check."""
    if not scales or len(set(scales)) != len(scales):
        raise ValueError(f"scales must be non-empty and distinct, got {list(scales)}")
    for sc in scales:
        check_capacity_scale(sc)
    gate = select_worlds(registry, DEVELOPMENT, sizes=sizes, families=families,
                         max_worlds=max_worlds)
    if not gate:
        raise ValueError("no development gate worlds selected for the probe")
    smoke = test_only_problem(gate)
    if smoke is not None:
        raise ValueError(f"not a valid probe ({smoke}; a probe has no dry run)")
    out = Path(out_dir)
    source = source_fingerprint(source_commit or None)
    config = {
        "families": sorted({e.family for e in gate}), "sizes": sorted({e.size for e in gate}),
        "worlds": [e.id for e in gate], "scales": list(scales),
        "members": list(roster.member_ids), "replicates": roster.replicates,
        "episodes_per_world": episodes_per_world, "horizon": horizon,
        "registry_version": registry.version, "roster_version": roster.version,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
    }
    _bind_probe_root(out, config)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(PROBE_SKELETON.format(
            title=out.name, driver=DRIVER, members=", ".join(roster.member_ids),
            replicates=roster.replicates,
            episodes="all" if episodes_per_world is None else f"≤ {episodes_per_world}"))
    table: list[Headroom] = []
    rows_out: list[dict[str, object]] = []
    for sc in scales:
        worlds = [scaled_entry(e, sc) for e in gate]
        scope = Registry(name=f"{registry.name}-probe", seed_ranges=registry.seed_ranges,
                         entries=tuple(worlds), saturation_ceiling=registry.saturation_ceiling,
                         shock_profiles=registry.shock_profiles)
        sub = out / f"scale-{sc:g}"
        grid = job_grid(worlds, episodes_per_world)
        expected = expected_keys(grid, roster, 0)
        pins = {
            "mode": "probe", "capacity_scale": sc, "horizon": horizon,
            "registry_version": registry.version, "scope_version": scope.version,
            "roster_version": roster.version,
            "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
            "candidate_replicates": 0, "grid": grid,
        }
        old_spec, done = _prepare_resume(sub, pins, grid, expected)
        spec = _open_spec(sub, old_spec, workers, {
            "experiment": out.name, "driver": DRIVER, "mode": "probe", "kind": "probe",
            "pins": pins, "roster": roster.payload() | {"content_hash": roster.version},
            "source": source,
        })
        log(f"[gate] probe capacity_scale={sc:g}")
        _execute(sub, spec, done, build_jobs(worlds, roster, None,
                                             episodes_per_world=episodes_per_world),
                 expected, workers, (horizon, None, None), run_job_fn, log)
        _stop_if_failed(sub, spec)
        rows = read_rows(sub / "episodes.csv")
        check_rows(rows, grid, expected, complete=True)
        hr = headroom(episode_bests(rows, roster), {e.id: e.family for e in worlds},
                      registry.saturation_ceiling)
        _write_table(sub / "headroom.csv", [h.to_dict() for h in hr])
        spec["status"] = "complete"
        spec["result"] = [h.to_dict() for h in hr]
        _write_spec(sub, spec)
        table += hr
        rows_out += [{"capacity_scale": sc, **h.to_dict()} for h in hr]
        log("[gate] probe " + "; ".join(f"{h.family}-{h.size} median {h.median_fill:.4f}"
                                        for h in hr))
    _write_table(out / "probe_headroom.csv", rows_out)
    return table


def _stop_if_failed(out: Path, spec: dict) -> None:
    """Screen and probe: any contract failure row ends the run invalid before any outcome."""
    failures = read_failures(out)
    if failures:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures)}
        _write_spec(out, spec)
        raise ContractRunError(f"{len(failures)} contract failure(s) recorded in "
                               f"{out / FAILURES_FILE}; the run is invalid")


RUN_ARTIFACTS = ("spec.json", "episodes.csv", "screening.json", "verdict.json", "frozen")


def _bind_probe_root(out: Path, config: Mapping[str, object]) -> None:
    """Before any write (review concern 113): refuse a root that holds a screen or report run,
    so a probe can never add files to (or be mistaken for) one; and bind the root to its probe
    configuration in `probe.json`, so a resume with other scales, members, R, episodes, worlds
    or source is refused and the root summary, RESULT approximation and subdirs never disagree."""
    clash = sorted(name for name in RUN_ARTIFACTS if (out / name).exists())
    if clash:
        raise ValueError(f"{out} holds a screen/report run ({', '.join(clash)}); a probe needs "
                         "its own --out-dir")
    probe_json = out / "probe.json"
    if probe_json.exists():
        recorded = json.loads(probe_json.read_text())
        if recorded != json.loads(json.dumps(config)):
            raise ValueError(f"{out} holds a probe with a different configuration; refusing to "
                             f"resume:\n  recorded {recorded}\n  now      {dict(config)}")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty and holds no probe.json; a probe needs a fresh "
                         "--out-dir")
    out.mkdir(parents=True, exist_ok=True)
    probe_json.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


PROBE_SKELETON = """# {title}

> **PROBE — calibration evidence, not an acceptance read.** Baseline-only headroom at several
> capacity scales (design D10); no frozen files.

See `probe_headroom.csv` (one row per scale × family × size), `probe.json` and `scale-*/`.
Produced by `python -m {driver} --mode probe` (see each `scale-*/spec.json`).

**Approximation.** The hindsight baseline here is the per-episode best of only these members:
{members}; random members ran R = {replicates} replicate(s), on {episodes} episodes per world.
The gate screen uses the full roster, R = 5 and every registered episode, so these medians
estimate the screen's; the rescreen is the binding check.

## Result

TODO

## Conclusions

TODO
"""


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
    ap.add_argument("--mode", choices=("report", "recompute", "screen", "probe"), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subject", default=PORTABLE,
                    help="'portable' (needs --prod-ckpt, --orderer-ckpt) or "
                         "'baseline:<producing>/<ordering>'")
    ap.add_argument("--prod-ckpt")
    ap.add_argument("--orderer-ckpt")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--roster", default=str(DEFAULT_ROSTER))
    ap.add_argument("--params", default=str(DEFAULT_PARAMS),
                    help="side-effect parameter file (X, Y, q, content hash); default: the "
                         "packaged frozen file (task 11.1)")
    ap.add_argument("--split", choices=SPLITS, default="development")
    ap.add_argument("--preregistration", help="manifest required for the sealed split")
    ap.add_argument("--access-log", default=str(SEALED_ACCESS_LOG))
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sizes", help="dry-run filter, e.g. 12")
    ap.add_argument("--families", help="dry-run filter, e.g. dense,sparse_wide")
    ap.add_argument("--regimes", help="report/screen dry-run filter, e.g. shocked")
    ap.add_argument("--max-worlds", type=int)
    ap.add_argument("--max-members", type=int, help="dry run: keep the first N roster members")
    ap.add_argument("--episodes-per-world", type=int)
    ap.add_argument("--replicates", type=int, help="dry run: override roster R")
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--verify-source-commit", action="store_true")
    ap.add_argument("--q", type=float, default=0.05, help="screen: X/Y calibration quantile (D4)")
    ap.add_argument("--prune", action="store_true", help="screen: apply the never-argmax rule")
    ap.add_argument("--reserves", choices=("on-demand", "eager", "none"), default="on-demand",
                    help="screen: run reserve worlds only when a gate world needs replacing "
                         "(default), all up front (eager), or not at all (dry runs only)")
    ap.add_argument("--scales", help="probe: capacity scales, e.g. 1.0,0.95,0.9,0.85,0.8")
    ap.add_argument("--members", help="probe or dry run: keep only these roster members "
                                      "(comma list of member ids)")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke run: allows filters, an unfrozen roster, an unscreened registry and "
                         "no source commit; marked as not an acceptance read")
    args = ap.parse_args(argv)

    if args.mode == "recompute":
        r = recompute_details(args.out_dir)
        if r.status == INVALID:
            print(f"[gate] recomputed status INVALID: {r.contract_failures} contract failure(s), "
                  f"{r.missing_rows} missing row(s); no verdict from partial rows")
            return CONTRACT_INVALID
        assert r.report is not None
        print(f"[gate] recomputed verdict {r.report.overall}; "
              f"{'MATCH' if r.ok else 'MISMATCH'} with {Path(args.out_dir) / 'verdict.json'} "
              f"(verdict {'equal' if r.verdict_ok else 'differs'}, resilience.csv "
              f"{'equal' if r.resilience_ok else 'differs'})")
        return 0 if r.ok else 1

    if args.regimes and args.mode not in ("report", "screen"):
        ap.error("--regimes filters a report or screen dry run")
    if args.regimes and set(_csv_strs(args.regimes) or []) - {"calm", "shocked"}:
        ap.error("--regimes takes calm and/or shocked")
    if args.mode == "report" and not args.params:
        ap.error("--mode report needs --params (the side-effect parameter file)")
    if args.reserves == "none" and not args.dry_run:
        ap.error("--reserves none is a dry-run option")
    source_commit = args.source_commit
    if args.verify_source_commit:
        from .provenance import resolve_verified_commit

        source_commit = resolve_verified_commit()
    probe = args.mode == "probe"
    if (args.max_members or args.replicates or args.members) and not (args.dry_run or probe):
        ap.error("--max-members / --replicates / --members change the roster: pass --dry-run")
    if probe and not args.scales:
        ap.error("--mode probe needs --scales")
    roster = load_roster(args.roster)
    if args.max_members or args.replicates or args.members:
        members = roster.members[: args.max_members] if args.max_members else roster.members
        if args.members:
            want = _csv_strs(args.members) or []
            unknown = sorted(set(want) - {m.id for m in roster.members})
            if unknown:
                ap.error(f"--members not in roster {roster.version[:12]}: {unknown}")
            members = tuple(m for m in members if m.id in set(want))
        roster = replace(roster, name=f"{roster.name}-{'probe' if probe else 'dry-run'}",
                         members=members,
                         replicates=args.replicates or roster.replicates, frozen=False,
                         pruning=None, screened_on=None)  # a derived smoke roster: no provenance
    try:
        return _run_mode(args, roster, source_commit, probe)
    except (ContractRunError, DigestMismatch) as exc:
        print(f"[gate] INVALID run: {exc}", file=sys.stderr)
        return CONTRACT_INVALID


def _run_mode(args: argparse.Namespace, roster: Roster, source_commit: str, probe: bool) -> int:
    if probe:
        run_probe(out_dir=args.out_dir, horizon=args.horizon, workers=args.workers,
                  registry=load_registry(args.registry), roster=roster,
                  scales=[float(x) for x in args.scales.split(",")],
                  families=_csv_strs(args.families) or [DENSE], sizes=_csv_ints(args.sizes),
                  max_worlds=args.max_worlds, episodes_per_world=args.episodes_per_world,
                  source_commit=source_commit)
        return 0
    if args.mode == "screen":
        _, status = run_screen(
            out_dir=args.out_dir, horizon=args.horizon, workers=args.workers,
            registry=load_registry(args.registry), roster=roster, q=args.q, prune=args.prune,
            reserves=args.reserves, sizes=_csv_ints(args.sizes),
            families=_csv_strs(args.families), regimes=_csv_strs(args.regimes),
            max_worlds=args.max_worlds,
            episodes_per_world=args.episodes_per_world, source_commit=source_commit,
            dry_run=args.dry_run)
        return status
    params_payload = json.loads(Path(args.params).read_text())
    params = load_side_effect_params(args.params)
    try:
        report, _ = run_report(
            out_dir=args.out_dir, subject=args.subject, split=args.split, horizon=args.horizon,
            workers=args.workers, registry=load_registry(args.registry), roster=roster,
            params=params, params_payload=params_payload, prod_ckpt=args.prod_ckpt,
            orderer_ckpt=args.orderer_ckpt, preregistration=args.preregistration,
            access_log=args.access_log, sizes=_csv_ints(args.sizes),
            families=_csv_strs(args.families), regimes=_csv_strs(args.regimes),
            max_worlds=args.max_worlds,
            episodes_per_world=args.episodes_per_world, source_commit=source_commit,
            dry_run=args.dry_run)
    except (ContractRunError, DigestMismatch) as exc:
        print(f"[gate] INVALID run: {exc}", file=sys.stderr)
        return CONTRACT_INVALID
    return 0 if report is not None else CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
