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
    MARGINS,
    PARAMS_V2,
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
    calibrate_families,
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
    side_effect_params_json_v3,
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
from .families import SHOCKED as SHOCKED_REGIME
from .families import SPARSE_WIDE as SPARSE_WIDE_FAMILY
from .registry import (
    DEFAULT_REGISTRY,
    DEVELOPMENT,
    EXCLUDED,
    GATE,
    REGION_FAMILIES,
    REPLACED,
    RESERVE,
    SEALED,
    SEALED_ACCESS_LOG,
    SPLITS,
    WS_WIDE,
    Registry,
    ReplacementPlan,
    ScreeningRecord,
    WorldEntry,
    check_sealed_access,
    load_preregistration,
    load_registry,
    plan_replacements,
    record_exclusions,
    record_headroom,
    smoke_variant,
    save_registry,
    screen_zero_baselines,
    world_baseline_means,
)
from .roster import (
    DEFAULT_ROSTER,
    GATE_SIZES,
    JOINT_STACK,
    RANDOM_POLICY,
    PILOT_V3_NAME,
    ROSTER_V2,
    WAVE2_PRODUCING,
    DevRow,
    Roster,
    contract_for,
    freeze_roster,
    load_roster,
    name_version_tag,
    prune_roster,
    prune_two_wave,
    roster_from_dict,
    save_roster,
    wave2_members,
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
             twins: bool = False, families: bool = False) -> dict[str, dict]:
    """world -> {size, regime, episodes}: the episodes a report runs, pinned in `spec.json` so a
    resume and a recompute can check the CSV against it without the registry. With `twins` (a
    report), a world whose config has shocks also pins `twin` = its resilience window and
    tolerance: its candidate episodes get twins and resilience rows. The key is absent
    otherwise, so a grid without shocked configs is unchanged. With `families` (a report under
    v3+ per-family side-effect parameters, add-gate-v3 D4) each world also pins its `family`,
    which a recompute needs; absent otherwise, so v1 / v2 grids are unchanged."""
    grid = {}
    for w in worlds:
        g: dict = {"size": w.size, "regime": w.regime,
                   "episodes": list(w.episode_seeds[:episodes_per_world] if episodes_per_world
                                    else w.episode_seeds)}
        if families:
            g["family"] = w.family
        tp = twin_params(w) if twins else None
        if tp is not None:
            g["twin"] = {"window": tp.window, "tolerance": tp.tolerance}
        grid[w.id] = g
    return grid


def grid_families(grid: Mapping[str, Mapping]) -> dict[str, str]:
    """world -> family from a grid pinned with `families` (v3+ parameters)."""
    missing = sorted(w for w, g in grid.items() if "family" not in g)
    if missing:
        raise ValueError(f"the pinned grid records no family for worlds {missing[:3]}; per-family "
                         "side-effect parameters need it")
    return {w: g["family"] for w, g in grid.items()}


def params_family_problem(registry: Registry, params: SideEffectParams) -> str | None:
    """Why per-family (v3+) parameters cannot check this registry (add-gate-v3 D4): a family
    with any non-excluded entry and no X / Y in the file. None for v1 / v2 parameters."""
    missing = params.missing_families(e.family for e in registry.entries if e.status != EXCLUDED)
    if not missing:
        return None
    return (f"side-effect parameters {params.name!r} ({params.version[:12]}) record no X / Y for "
            f"families {missing} present in registry {registry.name!r} ({registry.version[:12]})")


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
    family_problem = params_family_problem(registry, params)
    if family_problem is not None:  # also in a dry run: no check could use the missing values
        raise ValueError(family_problem)
    checkpoints = ({"producer": prod_ckpt, "orderer": orderer_ckpt} if pair is None else {})
    out = Path(out_dir)
    ckpt_hashes = _checked_checkpoints(out, checkpoints, dry_run, log)
    candidate_replicates = roster.replicates if pair and RANDOM_POLICY in pair else 1
    grid = job_grid(worlds, episodes_per_world, twins=True, families=params.per_family)
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
            report = compute_report(rows, roster, params, registry.version,
                                    world_family=grid_families(grid) if params.per_family
                                    else None)
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
    report = compute_report(rows, roster, params, pins["registry_version"],
                            world_family=grid_families(grid) if params.per_family else None)
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
    member_waves: str | None = None,
    worlds_per_family: int | None = None,
    region_shocked_episodes: int | None = None,
    reuse_rows: str | None = None,
    shock_profile: str | None = None,
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
    if not dry_run and roster.name == PILOT_V3_NAME:  # carry-forward from the 1.1 review
        raise ValueError(f"roster {roster.name!r} is the pilot roster: a dry run only, never "
                         "frozen as a gate roster")
    if member_waves is not None and member_waves != MEMBER_WAVES_V3:
        raise ValueError(f"--member-waves takes {MEMBER_WAVES_V3!r}, got {member_waves!r}")
    if (member_waves == MEMBER_WAVES_V3) != (registry.version_tag >= 3):
        raise ValueError(f"registry {registry.name!r} (v{registry.version_tag}) "
                         + ("needs --member-waves v3 (design D3)" if registry.version_tag >= 3
                            else "is screened in one wave; --member-waves v3 is for v3+"))
    if member_waves == MEMBER_WAVES_V3:
        return run_screen_v3(  # type: ignore[return-value]
            out_dir=out_dir, horizon=horizon, workers=workers, registry=registry, roster=roster,
            q=q, reserves=reserves, sizes=sizes, families=families, regimes=regimes,
            max_worlds=max_worlds, worlds_per_family=worlds_per_family,
            episodes_per_world=episodes_per_world, source_commit=source_commit,
            dry_run=dry_run, region_shocked_episodes=region_shocked_episodes,
            reuse_rows=reuse_rows, shock_profile=shock_profile, run_job_fn=run_job_fn, log=log)
    if any(v is not None for v in (worlds_per_family, region_shocked_episodes, reuse_rows,
                                   shock_profile)):
        raise ValueError("--worlds-per-family / --region-shocked-episodes / --reuse-rows / "
                         "--shock-profile belong to the two-wave screen (--member-waves v3)")
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


# --- the two-wave v3 screen (add-gate-v3 task 3.1, design D3 / D4) ---

MEMBER_WAVES_V3 = "v3"
# The per-world routing reading (D3 step 4): the input-aware routers against the blind ones, all
# with the entity_planner producer; a world counts at a relative gain >= 0.05 above its floor; a
# family reads `routing_matters` when >= 1/3 of its admitted worlds count. Reported, never gating.
INPUT_AWARE_ROUTERS = ("makeable_weighted_whole", "makeable_split")
BLIND_ROUTERS = ("blend_0.5", "even_split", "obs_price", "proportional", "random", "shock_reactive",
                 "round_robin", "even_split_rotating", "random_whole")
ROUTING_PRODUCER = WAVE2_PRODUCING
ROUTING_GAIN = 0.05
ROUTING_SHARE = 1 / 3
CALIBRATION_RULE_V3 = (
    "add-gate-v3 D4: X and Y per gate family (sparse_wide pooled over its admitted sizes), each by "
    "the D4 rule (`gate.CALIBRATION_RULE`) over that family's admitted final development episodes, "
    "with the pruned two-wave roster; excluded family x size episodes enter no family."
)
DRY_ONLY_V3 = "not an acceptance read"


def _rows_by(rows: Sequence[EpisodeRow], worlds: Collection[str],
             members: Collection[str] | None = None) -> list[EpisodeRow]:
    return [r for r in rows if r.world in worlds and (members is None or r.subject in members)]


def _complete(rows: Sequence[EpisodeRow], scheduled: Sequence[tuple[WorldEntry, Roster]],
              episodes_per_world: int | None) -> bool:
    have = {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    return all(job_key(j["world"], j["episode"], j["subject"], j["replicate"]) in have
               for e, ros in scheduled
               for j in build_jobs([e], ros, None, episodes_per_world=episodes_per_world))


def episode_argmax(rows: Sequence[EpisodeRow], roster: Roster) -> dict[tuple[str, int], str]:
    """(world, episode) -> the member with the largest replicate-mean units, ties to the smallest
    member id (`gate.episode_bests`' pick)."""
    vals: dict[tuple[str, int], dict[str, list[float]]] = {}
    for r in rows:
        vals.setdefault((r.world, r.episode), {}).setdefault(r.subject, []).append(r.units)
    out = {}
    for k, by_m in vals.items():
        means = {m: fmean(v) for m, v in by_m.items() if m in set(roster.member_ids)}
        best = max(means.values())
        out[k] = min(m for m, v in means.items() if v == best)
    return out


def bar_dead_worlds(scope: Registry, roster: Roster, rows: Sequence[EpisodeRow]
                    ) -> dict[str, tuple[int, str]]:
    """Region worlds whose per-episode argmax member reached system death in some episode (any
    replicate: a death has no mean), with the first such (episode, member) in registered episode
    order (D3 step 3, gate-harness "Region worlds whose best baseline dies are replaced").
    sparse_wide keeps its v2 rule and is never listed."""
    best = episode_argmax(rows, roster)
    dead = {(r.world, r.episode, r.subject) for r in rows if r.dead}
    out: dict[str, tuple[int, str]] = {}
    for e in scope.entries:
        if e.family not in REGION_FAMILIES:
            continue
        for ep in e.episode_seeds:
            m = best.get((e.id, ep))
            if m is not None and (e.id, ep, m) in dead:
                out[e.id] = (ep, m)
                break
    return out


def _sub_registry(scope: Registry, keep: Callable[[WorldEntry], bool]) -> Registry | None:
    entries = tuple(e for e in scope.entries if keep(e))
    if not any(e.status == GATE and e.split == DEVELOPMENT for e in entries):
        return None
    return Registry(name=scope.name, seed_ranges=scope.seed_ranges, entries=entries,
                    saturation_ceiling=scope.saturation_ceiling,
                    shock_profiles=scope.shock_profiles)


def _apply(scope: Registry, plan: ReplacementPlan) -> dict[str, WorldEntry]:
    """The development gate worlds after a replacement plan (statuses as `screen_zero_baselines`
    sets them)."""
    final = {e.id: e for e in scope.gate_worlds(DEVELOPMENT)}
    for rep_ in plan.replacements:
        final.pop(rep_.replaced, None)
        final[rep_.by] = scope.entry(rep_.by)
    return final


@dataclass(frozen=True)
class WaveState:
    """One member wave as far as the rows decide it: the worlds scheduled (gate worlds, then the
    reserve sub-waves the replacement plan named), and, once complete, the final plan."""

    roster: Roster
    scheduled: tuple[WorldEntry, ...]
    sub_waves: tuple[tuple[str, ...], ...]
    complete: bool
    plan: ReplacementPlan | None = None
    means: tuple[tuple[str, float], ...] = ()
    bar_dead: tuple[tuple[str, int, str], ...] = ()


def _wave(scope: Registry | None, roster: Roster, rows: Sequence[EpisodeRow],
          reserves: str, episodes_per_world: int | None) -> WaveState | None:
    """Plan one wave from the rows: gate worlds first; while complete, the replacement plan
    (zero-baseline, plus bar-death for region worlds) names the next reserve sub-wave."""
    if scope is None:
        return None
    scheduled = list(scope.gate_worlds(DEVELOPMENT))
    sub_waves: list[tuple[str, ...]] = []
    while True:
        pairs = [(e, roster) for e in scheduled]
        if not _complete(rows, pairs, episodes_per_world):
            return WaveState(roster, tuple(scheduled), tuple(sub_waves), False)
        ids = {e.id for e in scheduled}
        mine = _rows_by(rows, ids, set(roster.member_ids))
        means = world_baseline_means(scope, roster, mine)
        dead = bar_dead_worlds(scope, roster, mine)
        plan = plan_replacements(scope, means, (DEVELOPMENT,), bad=set(dead))
        if not plan.pending or reserves != "on-demand":
            kept = {e.id for e in scope.gate_worlds(DEVELOPMENT)} | set(plan.consulted)
            return WaveState(roster, tuple(scheduled), tuple(sub_waves), True, plan,
                             tuple(sorted((w, m) for w, m in means.items() if w in kept)),
                             tuple(sorted((w, ep, m) for w, (ep, m) in dead.items()
                                          if w in kept)))
        new = tuple(w for w in plan.pending if w not in ids)
        if not new:
            raise AssertionError("replacement plan names pending reserves already run")
        sub_waves.append(new)
        scheduled += [scope.entry(w) for w in new]


def _dev_rows(rows: Sequence[EpisodeRow]) -> list[DevRow]:
    acc: dict[tuple[str, int, int, str], list[float]] = {}
    for r in rows:
        acc.setdefault((r.world, r.size, r.episode, r.subject), []).append(r.units)
    return [DevRow(w, DEVELOPMENT, z, ep, m, fmean(v)) for (w, z, ep, m), v in sorted(acc.items())]


@dataclass(frozen=True)
class V3Plan:
    """The two-wave screen as far as the rows decide it (`plan_v3`). `jobs` is every (world,
    roster) scheduled so far, wave 1 then wave 2; `complete` says whether nothing more needs to
    run; `stop` names a size left with no admitted family (exit 3)."""

    wave1: WaveState
    wave2: WaveState | None
    w2: tuple[str, ...] | None
    headroom1: tuple[Headroom, ...]
    headroom2: tuple[Headroom, ...]
    excluded: frozenset[tuple[str, int]]
    stop: str | None
    complete: bool

    @property
    def jobs(self) -> list[tuple[WorldEntry, Roster]]:
        out = [(e, self.wave1.roster) for e in self.wave1.scheduled]
        if self.wave2 is not None:
            out += [(e, self.wave2.roster) for e in self.wave2.scheduled]
        return out


def _headroom(scope: Registry, wave: WaveState, rows: Sequence[EpisodeRow]
              ) -> tuple[dict[str, WorldEntry], tuple[Headroom, ...]]:
    final = _apply(scope, wave.plan)  # type: ignore[arg-type]
    mine = _rows_by(rows, final, set(wave.roster.member_ids))
    bests = episode_bests(mine, wave.roster)
    return final, tuple(headroom(bests, {w: e.family for w, e in final.items()},
                                 scope.saturation_ceiling))


def _size_problem(scope: Registry, excluded: Collection[tuple[str, int]],
                  pending: Collection[tuple[str, int]] = ()) -> str | None:
    """A size whose every family is excluded (pending families still count as admitted)."""
    fams: dict[int, set[str]] = {}
    for e in scope.gate_worlds(DEVELOPMENT):
        fams.setdefault(e.size, set()).add(e.family)
    for size, fs in sorted(fams.items()):
        left = {f for f in fs if (f, size) not in excluded or (f, size) in pending}
        if not left:
            return (f"no admitted family at {size} producers: "
                    f"{', '.join(sorted(f'{f}/{size}' for f in fs))} failed headroom")
    return None


def plan_v3(scope: Registry, roster: Roster, rows: Sequence[EpisodeRow], *,
            reserves: str = "on-demand", episodes_per_world: int | None = None) -> V3Plan:
    """D3 waves 1-2 decided from the rows alone, deterministically (a resume replays it):
    wave 1 = every member on the development gate worlds except `ws_wide`, then its reserve
    sub-waves; headroom per family x size of wave 1 (a failure is excluded; a size with no
    admitted family stops here, before wave 2); W2 = the wave-1 argmax members on admitted
    episodes plus every entity_planner member (`roster.wave2_members`); wave 2 = W2 on `ws_wide`
    and its reserves; headroom of `ws_wide`; a size with no admitted family stops."""
    scope1 = _sub_registry(scope, lambda e: e.family != WS_WIDE)
    scope2 = _sub_registry(scope, lambda e: e.family == WS_WIDE)
    if scope1 is None:
        raise ValueError("the two-wave screen needs wave-1 development gate worlds")
    wave1 = _wave(scope1, roster, rows, reserves, episodes_per_world)
    assert wave1 is not None
    if not wave1.complete:
        return V3Plan(wave1, None, None, (), (), frozenset(), None, False)
    final1, hr1 = _headroom(scope1, wave1, rows)
    excluded = {(h.family, h.size) for h in hr1 if not h.passed}
    pending = ({(WS_WIDE, e.size) for e in scope2.gate_worlds(DEVELOPMENT)}
               if scope2 is not None else set())
    stop = _size_problem(scope, excluded, pending)
    if stop is not None:
        return V3Plan(wave1, None, None, hr1, (), frozenset(excluded), stop, True)
    admitted1 = [w for w, e in final1.items() if (e.family, e.size) not in excluded]
    w2 = wave2_members(roster, _dev_rows(_rows_by(rows, admitted1, set(roster.member_ids))))
    if scope2 is None:
        return V3Plan(wave1, None, w2, hr1, (), frozenset(excluded), None, True)
    w2_roster = Roster(name=f"{roster.name}-w2", replicates=roster.replicates,
                       members=tuple(m for m in roster.members if m.id in set(w2)))
    wave2 = _wave(scope2, w2_roster, rows, reserves, episodes_per_world)
    assert wave2 is not None
    if not wave2.complete:
        return V3Plan(wave1, wave2, w2, hr1, (), frozenset(excluded), None, False)
    _, hr2 = _headroom(scope2, wave2, rows)
    excluded |= {(h.family, h.size) for h in hr2 if not h.passed}
    stop = _size_problem(scope, excluded)
    return V3Plan(wave1, wave2, w2, hr1, hr2, frozenset(excluded), stop, True)


def attainability(scope: Registry, worlds: Mapping[str, tuple[WorldEntry, Roster]],
                  rows: Sequence[EpisodeRow], excluded: Collection[tuple[str, int]]
                  ) -> dict[str, list[dict[str, object]]]:
    """Perfect-fill ratios (D3 step 4, gate-harness "Attainable margins are reported"): per final
    world, 1 / (mean over its episodes of the argmax member's fill); per (size, regime) over the
    admitted worlds and per (family, size, regime) over all final worlds, the mean against the
    regime's margin, flagged `unattainable` below it. Reported, never gating."""
    per_world: dict[str, float | None] = {}
    for w, (e, ros) in worlds.items():
        fills = [b.fill for b in episode_bests(_rows_by(rows, {w}, set(ros.member_ids)), ros)]
        per_world[w] = (None if not fills or any(f is None or f == 0 for f in fills)
                        else 1.0 / fmean(fills))  # type: ignore[arg-type]

    def cell(ids: list[str], regime: str) -> dict[str, object]:
        vals = [per_world[w] for w in ids if per_world[w] is not None]
        mean = fmean(vals) if vals else None  # type: ignore[arg-type]
        margin = MARGINS[regime]
        return {"worlds": len(ids), "measured": len(vals), "perfect_fill_ratio": mean,
                "margin": margin, "unattainable": mean is not None and mean < margin}

    groups: dict[tuple[int, str], list[str]] = {}
    fam_groups: dict[tuple[str, int, str], list[str]] = {}
    for w, (e, _) in sorted(worlds.items()):
        fam_groups.setdefault((e.family, e.size, e.regime), []).append(w)
        if (e.family, e.size) not in excluded:
            groups.setdefault((e.size, e.regime), []).append(w)
    return {
        "by_size_regime": [{"size": z, "regime": g, **cell(ids, g)}
                           for (z, g), ids in sorted(groups.items())],
        "by_family": [{"family": f, "size": z, "regime": g,
                       "excluded": (f, z) in excluded, **cell(ids, g)}
                      for (f, z, g), ids in sorted(fam_groups.items())],
        "by_world": [{"world": w, "perfect_fill_ratio": per_world[w]} for w in sorted(worlds)],
    }


def _member_runs(rows: Sequence[EpisodeRow], member: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for r in rows:
        if r.subject == member:
            out.setdefault(r.episode, []).append(r.units)
    return out


def routing_world(e: WorldEntry, roster: Roster, rows: Sequence[EpisodeRow]) -> dict[str, object]:
    """One world's routing reading (D3 step 4, revised rule): `reach_gain_w` = (best input-aware
    - best blind) / best blind on world means, all with entity_planner; the world counts when the
    gain is >= 0.05 and the absolute gain is above the world's floor. Floors, pre-registered by
    world type: shocked, 3·sd(d_e)/√n over per-episode differences of the two sides (replicate
    means; undefined with one episode, then the world does not count); calm with a stochastic
    side, 3·√(Σ s²/n) over each stochastic side's episode × replicate runs; calm with both sides
    deterministic, 0. `router_spread_w` (max - min) / max over the 11 routers is a column only."""
    import statistics

    ids = {m.id: m for m in roster.members}
    want = {o: f"{ROUTING_PRODUCER}/{o}" for o in INPUT_AWARE_ROUTERS + BLIND_ROUTERS}
    out: dict[str, object] = {"world": e.id, "family": e.family, "size": e.size,
                              "regime": e.regime}
    if any(mid not in ids for mid in want.values()):
        return out | {"reading": "insufficient_members", "counts": False}
    mine = _rows_by(rows, {e.id}, set(want.values()))
    runs = {o: _member_runs(mine, mid) for o, mid in want.items()}
    means = {o: fmean(fmean(v) for v in runs[o].values()) for o in want}

    def best(group: Sequence[str]) -> str:
        top = max(means[o] for o in group)
        return min(o for o in group if means[o] == top)

    ia, bl = best(INPUT_AWARE_ROUTERS), best(BLIND_ROUTERS)
    diff = means[ia] - means[bl]
    gain = diff / means[bl] if means[bl] > 0 else None
    sides = (ia, bl)
    if e.regime == SHOCKED_REGIME:
        d = [fmean(runs[ia][ep]) - fmean(runs[bl][ep]) for ep in sorted(runs[ia])]
        floor = 3 * statistics.stdev(d) / len(d) ** 0.5 if len(d) >= 2 else None
        floor_type = "shocked"
    elif any(ids[want[o]].random for o in sides):
        terms = []
        for o in sides:
            if ids[want[o]].random:
                vals = [u for v in runs[o].values() for u in v]
                terms.append(statistics.stdev(vals) ** 2 / len(vals))
        floor = 3 * sum(terms) ** 0.5
        floor_type = "calm_stochastic"
    else:
        floor, floor_type = 0.0, "calm_deterministic"
    counts = gain is not None and floor is not None and gain >= ROUTING_GAIN and diff > floor
    top = max(means.values())
    return out | {"reading": "computed", "input_aware": ia, "blind": bl,
                  "input_aware_units": means[ia], "blind_units": means[bl],
                  "reach_gain": gain, "floor": floor, "floor_type": floor_type,
                  "counts": counts,
                  "router_spread": (top - min(means.values())) / top if top > 0 else None}


def routing_reading(worlds: Mapping[str, tuple[WorldEntry, Roster]], rows: Sequence[EpisodeRow],
                    excluded: Collection[tuple[str, int]]) -> dict[str, list[dict[str, object]]]:
    """Per-world readings and the family rule: `routing_matters` iff the counting worlds are at
    least a third of the family's admitted worlds, else `routing_flat`. Reported, never gating."""
    per_world = [routing_world(e, ros, rows) for _, (e, ros) in sorted(worlds.items())]
    fams: dict[tuple[str, int], list[dict[str, object]]] = {}
    for r in per_world:
        fams.setdefault((r["family"], r["size"]), []).append(r)  # type: ignore[index]
    families = []
    for (f, z), rs in sorted(fams.items()):
        n = len(rs)
        k = sum(1 for r in rs if r["counts"])
        missing = sum(1 for r in rs if r["reading"] != "computed")
        families.append({
            "family": f, "size": z, "admitted_worlds": n, "counting_worlds": k,
            "insufficient": missing, "excluded": (f, z) in excluded,
            "reading": ("excluded" if (f, z) in excluded
                        else "insufficient_members" if missing == n
                        else "routing_matters" if k / n >= ROUTING_SHARE
                        else "routing_flat")})
    return {"worlds": per_world, "families": families}


def v2_bars(worlds: Mapping[str, tuple[WorldEntry, Roster]], rows: Sequence[EpisodeRow]
            ) -> list[dict[str, object]]:
    """On each final v2 (sparse_wide) world: the hindsight baseline of the frozen v2 roster
    beside the v3 bar (wave-1 roster). Skipped when a v2 member did not run (a filtered dry run).
    Reported, never gating."""
    from .gate import episode_baseline, world_baseline

    v2 = load_roster(ROSTER_V2).member_ids
    out = []
    for w, (e, ros) in sorted(worlds.items()):
        if e.family != SPARSE_WIDE_FAMILY or not set(v2) <= set(ros.member_ids):
            continue
        mine = _rows_by(rows, {w})
        eps = sorted({r.episode for r in mine})
        val = {(ep, m): fmean(r.units for r in mine if r.episode == ep and r.subject == m)
               for ep in eps for m in ros.member_ids}
        bar3 = world_baseline([episode_baseline({m: val[(ep, m)] for m in ros.member_ids})
                               for ep in eps])
        bar2 = world_baseline([episode_baseline({m: val[(ep, m)] for m in v2}) for ep in eps])
        out.append({"world": w, "size": e.size, "regime": e.regime, "bar_v3": bar3,
                    "bar_v2_roster": bar2, "ratio": bar3 / bar2 if bar2 else None})
    return out


@dataclass(frozen=True)
class V3Outcome:
    """Everything the two-wave screen decides (`screen_outcome_v3`)."""

    plan: V3Plan
    registry: Registry  # screened: replacements, bar deaths, headroom, exclusions
    passed: bool  # every size kept an admitted family
    roster: Roster | None
    calibrations: tuple | None
    params_json: str | None
    tables: dict[str, object]


def screen_outcome_v3(scope: Registry, roster: Roster, rows: Sequence[EpisodeRow], *, q: float,
                      tag: int, screened_on: str, dry_run: bool, reserves: str = "on-demand",
                      episodes_per_world: int | None = None) -> V3Outcome:
    """The two-wave screen's outcome, a pure function of the rows (D3 steps 3-7): the screened
    registry (both waves' replacements and bar deaths, headroom of every family x size,
    exclusions in both splits), and, unless a size lost every family, the two-wave pruned
    roster and per-family X / Y (D4) over the admitted final worlds. Excluded worlds' rows enter
    no W2 winner set, no pruning evidence and no X / Y; they are reported."""
    plan = plan_v3(scope, roster, rows, reserves=reserves, episodes_per_world=episodes_per_world)
    if not plan.complete:
        raise ValueError("the two-wave screen's rows are incomplete")
    waves = [plan.wave1] + ([plan.wave2] if plan.wave2 is not None and plan.wave2.plan else [])
    replacements = tuple(r for w in waves for r in w.plan.replacements)  # type: ignore[union-attr]
    means = tuple(sorted(m for w in waves for m in w.means))
    bar_dead = tuple(sorted(b for w in waves for b in w.bar_dead))
    updated = {e.id: e for e in scope.entries}
    for rep_ in replacements:
        updated[rep_.replaced] = replace(updated[rep_.replaced], status=REPLACED)
        for sk in rep_.skipped:
            updated[sk] = replace(updated[sk], status=REPLACED)
        updated[rep_.by] = replace(updated[rep_.by], status=GATE)
    record = ScreeningRecord(roster_version=roster.version, baseline_means=means,
                             replacements=replacements, splits=(DEVELOPMENT,),
                             bar_deaths=bar_dead)
    screened = Registry(name=scope.name, seed_ranges=scope.seed_ranges,
                        entries=tuple(updated.values()), screening=record,
                        saturation_ceiling=scope.saturation_ceiling,
                        shock_profiles=scope.shock_profiles)
    screened = record_headroom(screened, plan.headroom1 + plan.headroom2)
    screened = record_exclusions(screened)
    screened = replace(screened, screened_on=screened_on)
    finals: dict[str, tuple[WorldEntry, Roster]] = {}
    for w in waves:
        sub = _sub_registry(scope, (lambda e: e.family != WS_WIDE) if w is plan.wave1
                            else (lambda e: e.family == WS_WIDE))
        finals.update({wid: (e, w.roster) for wid, e in _apply(sub, w.plan).items()})  # type: ignore[arg-type]
    admitted = {w: v for w, v in finals.items() if (v[0].family, v[0].size) not in plan.excluded}
    tables: dict[str, object] = {
        "attainability": attainability(scope, finals, rows, plan.excluded),
        "routing": routing_reading(admitted, rows, plan.excluded),
        "v2_bars": v2_bars(finals, rows),
    }
    if plan.stop is not None:
        return V3Outcome(plan, screened, False, None, None, None, tables)
    w1_ids = [w for w, (e, ros) in admitted.items() if ros is plan.wave1.roster]
    w2_ids = [w for w, (e, ros) in admitted.items() if ros is not plan.wave1.roster]
    pruned = prune_two_wave(
        roster, _dev_rows(_rows_by(rows, w1_ids, set(roster.member_ids))),
        _dev_rows(_rows_by(rows, w2_ids, set(plan.w2 or ()))), plan.w2 or ())
    suffix = "-dry-run" if dry_run else ""
    frozen = replace(freeze_roster(pruned), name=f"roster-v{tag}{suffix}",
                     screened_on=screened_on)
    kept = set(frozen.member_ids)
    cal_rows = _rows_by(rows, admitted, kept)
    cals = calibrate_families(cal_rows, frozen, q, {w: e.family for w, (e, _) in admitted.items()})
    params = side_effect_params_json_v3(
        {c.family: c.payload() for c in cals}, q, name=f"side-effect-params-v{tag}{suffix}",
        rule=CALIBRATION_RULE_V3, method=cals[0].calibration.method,
        evidence={"roster_version": frozen.version, "registry_version": screened.version,
                  "excluded": sorted(f"{f}/{z}" for f, z in plan.excluded)},
        screened_on=screened_on)
    v2 = load_side_effect_params(PARAMS_V2)
    tables["xy_compare"] = [{"family": c.family, "x_v3": c.calibration.x,
                             "y_v3": c.calibration.y, "x_v2": v2.x, "y_v2": v2.y,
                             "small_sample": c.small_sample} for c in cals]
    return V3Outcome(plan, screened, True, frozen, cals, params, tables)


def _smoke_scope_entry(e: WorldEntry) -> WorldEntry:
    """The horizon-12 smoke form of a shocked entry (dry runs only): a legacy family names
    `shocks-i-smoke`; a region entry takes `registry.smoke_variant` (its own record)."""
    from .shock_profiles import SHOCKS_I_SMOKE

    if e.regime != SHOCKED_REGIME:
        return e
    if e.family in REGION_FAMILIES:
        return smoke_variant(e)
    g = json.loads(json.dumps(e.generator))
    g["params"]["shock_profile"] = SHOCKS_I_SMOKE
    return replace(e, generator=g)


def _reuse_v2_rows(out: Path, scope: Registry, roster: Roster, src: Path, *, spec: dict,
                   workers: int, horizon: int, run_job_fn: Callable[[Job], dict[str, str]],
                   log: Callable[[str], None]) -> dict[str, object]:
    """D3 reduction 2: wave 1's v2-world rows of the v2-draft members come from EXP-0047, after
    a reproduction check: one world-episode per (size, regime), the first gate world by id and
    its first episode, run here for every v2-draft member and replicate, must equal EXP-0047's
    rows exactly (every CSV field). Then every other such row is appended from EXP-0047."""
    from .registry import build_worlds_v2
    from .roster import build_roster_v2

    sspec = json.loads((src / "spec.json").read_text())
    spins = sspec.get("pins", {})
    want = {"registry_version": build_worlds_v2().version,
            "roster_version": build_roster_v2().version, "horizon": 100, "dry_run": False}
    bad = {k: spins.get(k) for k, v in want.items() if spins.get(k) != v}
    if bad:
        raise ValueError(f"{src}: not a real horizon-100 screen of the registry v2 / roster v2 "
                         f"drafts (pins {bad}); refusing to reuse its rows")
    v2_ids = set(build_roster_v2().member_ids)
    if not v2_ids <= set(roster.member_ids) or roster.replicates != 5:
        raise ValueError("reuse needs every v2-draft member in the roster at R = 5 (EXP-0047's R)")
    sub = Roster(name=f"{roster.name}-v2-members", replicates=roster.replicates,
                 members=tuple(m for m in roster.members if m.id in v2_ids))
    v2w = sorted((e for e in scope.gate_worlds(DEVELOPMENT) if e.family == SPARSE_WIDE_FAMILY),
                 key=lambda e: e.id)
    ids = {e.id for e in v2w}
    src_rows = {job_key(r.world, r.episode, r.subject, r.replicate): r
                for r in read_rows(src / "episodes.csv") if r.world in ids and r.subject in v2_ids}
    repro: dict[tuple[int, str], WorldEntry] = {}
    for e in v2w:
        repro.setdefault((e.size, e.regime), e)
    jobs = [j for e in repro.values() for j in build_jobs([e], sub, None)
            if j["episode"] == e.episode_seeds[0]]
    keys = {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs}
    episodes = out / "episodes.csv"
    done = read_rows(episodes) if episodes.exists() and episodes.stat().st_size else []
    _execute(out, spec, done, jobs, keys, workers, (horizon, None, None), run_job_fn, log)
    _stop_if_failed(out, spec)
    have = {job_key(r.world, r.episode, r.subject, r.replicate): r for r in read_rows(episodes)}
    bad = sorted(k for k in keys if k not in src_rows
                 or row_to_dict(have[k]) != row_to_dict(src_rows[k]))
    if bad:
        raise ValueError(f"reuse reproduction check failed on {len(bad)} of {len(keys)} rows, "
                         f"e.g. {bad[:3]}: EXP-0047's rows are not reused")
    added = 0
    for e in v2w:
        for j in build_jobs([e], sub, None):
            k = job_key(j["world"], j["episode"], j["subject"], j["replicate"])
            if k in have:
                continue
            if k not in src_rows:
                raise ValueError(f"{src}: no row for {k}; refusing a partial reuse")
            _append_row(episodes, row_to_dict(src_rows[k]))
            added += 1
    log(f"[gate] reuse: reproduction matched {len(keys)} rows; {added} rows reused from {src}")
    return {"source": str(src), "episodes_sha256": sha256_file(str(src / "episodes.csv")),
            "reproduction": sorted(f"{e.id}/{e.episode_seeds[0]}" for e in repro.values()),
            "reproduced_rows": len(keys), "reused_rows": added, "status": "match"}


def _v3_wave_summary(plan: V3Plan) -> list[dict[str, object]]:
    out = []
    for n, w in ((1, plan.wave1), (2, plan.wave2)):
        if w is None:
            continue
        gate = [e.id for e in w.scheduled if e.status == GATE]
        out.append({"wave": n, "members": list(w.roster.member_ids),
                    "gate_worlds": gate, "reserve_sub_waves": [list(x) for x in w.sub_waves],
                    "complete": w.complete})
    return out


def run_screen_v3(
    *, out_dir: str | Path, horizon: int, workers: int, registry: Registry, roster: Roster,
    q: float = 0.05, reserves: str = "on-demand", sizes: Sequence[int] | None = None,
    families: Sequence[str] | None = None, regimes: Sequence[str] | None = None,
    max_worlds: int | None = None, worlds_per_family: int | None = None,
    episodes_per_world: int | None = None, source_commit: str = "", dry_run: bool = False,
    region_shocked_episodes: int | None = None, reuse_rows: str | None = None,
    shock_profile: str | None = None,
    run_job_fn: Callable[[Job], dict[str, str]] = run_job, log: Callable[[str], None] = print,
) -> tuple[V3Outcome, int]:
    """The two-wave v3 screen (`--member-waves v3`, design D3): run the waves `plan_v3` decides
    from the rows, then `screen_outcome_v3`; write `screening.json` (both waves, exclusions,
    attainability, routing, v2 / v3 X / Y, v2-roster bars), its tables and, when every size keeps
    an admitted family, `frozen/{worlds,roster,side_effect_params}_v3.json` with `screened_on` =
    the draft's hash. Exit 0 or `HEADROOM_FAILED` (a size with no admitted family). The run
    always prunes (`never-argmax-two-wave-v1`): W2's members alone have rows on wave 2.

    D3's reductions are flags pinned in `spec.json`: `region_shocked_episodes` (shocked region
    worlds keep their first n episode seeds, in both splits, in the screened registry) and
    `reuse_rows` (`_reuse_v2_rows`). A dry run may pass `shock_profile="shocks-i-smoke"` (every
    shocked entry in its smoke form) and `worlds_per_family` (per family the first 2n calm and n
    shocked gate worlds, so the scope keeps the gate's regime ratio). A resume replays `plan_v3` from the rows; rows outside its plan are refused."""
    from .shock_profiles import SHOCKS_I_SMOKE

    tag = registry.version_tag
    if tag < 3:
        raise ValueError(f"--member-waves v3 screens a v3+ registry, got {registry.name!r}")
    if reserves == "eager":
        raise ValueError("the two-wave screen plans reserves on demand per wave; eager is refused")
    if shock_profile is not None and (not dry_run or shock_profile != SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes only {SHOCKS_I_SMOKE!r}, in a dry run")
    if region_shocked_episodes is not None and not 1 <= region_shocked_episodes < 10:
        raise ValueError("--region-shocked-episodes must be in 1..9 (D3 reduction 1 uses 5)")
    screened_on = registry.version
    reg = registry
    if region_shocked_episodes is not None:
        n = region_shocked_episodes
        reg = replace(registry, entries=tuple(
            replace(e, episode_seeds=e.episode_seeds[:n])
            if e.family in REGION_FAMILIES and e.regime == SHOCKED_REGIME else e
            for e in registry.entries))
    gate = select_worlds(reg, DEVELOPMENT, sizes=sizes, families=families, regimes=regimes,
                         max_worlds=max_worlds)
    if worlds_per_family is not None:  # per family: the first 2N calm and N shocked gate worlds,
        # the gate's 2 : 1 regime ratio, so the dry-run scope keeps "Regimes" (calm >= 2x shocked)
        seen: dict[tuple[str, str], int] = {}
        kept_gate = []
        for e in gate:
            cap = worlds_per_family * (2 if e.regime == "calm" else 1)
            if seen.get((e.family, e.regime), 0) < cap:
                kept_gate.append(e)
                seen[(e.family, e.regime)] = seen.get((e.family, e.regime), 0) + 1
        gate = kept_gate
    groups = {e.group for e in gate}
    extra = [e for e in reg.entries
             if reserves != "none" and e.status == RESERVE and e.group in groups]
    if not dry_run:
        smoke = test_only_problem([*gate, *extra])
        problems = [msg for bad, msg in (
            (len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit),
             "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
            (any(f is not None for f in (sizes, families, regimes, max_worlds,
                                         worlds_per_family, episodes_per_world)),
             "world/episode filters are set"),
            (reserves == "none", "reserves are skipped"),
            (bool(reg.coverage_problems()), "; ".join(reg.coverage_problems())),
            (roster.version_tag != tag,
             f"registry {registry.name!r} is v{tag} but roster {roster.name!r} is "
             f"v{roster.version_tag}"),
            (smoke is not None, smoke or ""),
        ) if bad]
        if problems:
            raise ValueError("not a valid real screen (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    if not gate:
        raise ValueError("no development gate worlds selected")
    if dry_run:
        def cut(e: WorldEntry) -> WorldEntry:
            e = replace(e, episode_seeds=e.episode_seeds[:episodes_per_world]
                        if episodes_per_world else e.episode_seeds)
            return _smoke_scope_entry(e) if shock_profile else e

        scope = Registry(name=f"{reg.name}-dry-run", seed_ranges=reg.seed_ranges,
                         entries=tuple(cut(e) for e in (*gate, *extra)),
                         saturation_ceiling=reg.saturation_ceiling,
                         shock_profiles=reg.shock_profiles)
        epw = None  # the scope already holds the cut episodes
    else:
        scope, epw = reg, None
    src = Path(reuse_rows) if reuse_rows else None
    if src is not None and (horizon != 100 or episodes_per_world is not None):
        raise ValueError("--reuse-rows needs horizon 100 and every episode (EXP-0047's rows)")
    source = source_fingerprint(source_commit or None)
    pins = {
        "mode": "screen", "member_waves": MEMBER_WAVES_V3, "horizon": horizon,
        "dry_run": dry_run, "q": q, "prune": True, "reserves": reserves,
        "registry_version": registry.version, "scope_version": scope.version,
        "roster_version": roster.version,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "candidate_replicates": 0,
        "grid": job_grid(list(scope.gate_worlds(DEVELOPMENT)), epw),
        "region_shocked_episodes": region_shocked_episodes,
        "reuse_rows": None if src is None else {
            "path": str(src.resolve()), "episodes_sha256": sha256_file(str(src / "episodes.csv"))},
        "shock_profile": shock_profile, "worlds_per_family": worlds_per_family,
    }
    out = Path(out_dir)
    spec_path, episodes = out / "spec.json", out / "episodes.csv"
    old_spec = json.loads(spec_path.read_text()) if spec_path.exists() else None
    if old_spec is None and episodes.exists() and episodes.stat().st_size:
        raise ValueError(f"{episodes} exists without spec.json: orphan rows, refusing to resume")
    if old_spec is not None and old_spec.get("pins") != pins:
        raise ValueError(f"{out} holds a run with different pins; refusing to resume")
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "mode": "screen",
        "kind": "dry-run" if dry_run else "screen", "pins": pins,
        "roster": roster.payload() | {"content_hash": roster.version}, "source": source,
        "member_waves": [],
    })
    if src is not None and not (spec.get("reuse") or {}).get("status") == "match":
        spec["reuse"] = _reuse_v2_rows(out, scope, roster, src, spec=spec, workers=workers,
                                       horizon=horizon, run_job_fn=run_job_fn, log=log)
        _write_spec(out, spec)
    while True:
        rows = read_rows(episodes) if episodes.exists() and episodes.stat().st_size else []
        plan = plan_v3(scope, roster, rows, reserves=reserves, episodes_per_world=epw)
        pairs = plan.jobs
        jobs = [j for e, ros in pairs for j in build_jobs([e], ros, None, episodes_per_world=epw)]
        expected = {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs}
        check_rows(rows, job_grid([e for e, _ in pairs], epw), expected, complete=False)
        spec["member_waves"] = _v3_wave_summary(plan)
        _write_spec(out, spec)
        if plan.complete:
            break
        _execute(out, spec, rows, jobs, expected, workers, (horizon, None, None), run_job_fn,
                 log)
        _stop_if_failed(out, spec)
    outcome = screen_outcome_v3(scope, roster, rows, q=q, tag=tag, screened_on=screened_on,
                                dry_run=dry_run, reserves=reserves, episodes_per_world=epw)
    p = outcome.plan
    report: dict[str, object] = {
        "member_waves": MEMBER_WAVES_V3, "acceptance_read": not dry_run,
        "note": DRY_ONLY_V3 if dry_run else "",
        "registry_version": outcome.registry.version, "screened_on": screened_on,
        "waves": _v3_wave_summary(p), "w2": list(p.w2 or ()),
        "not_carried": [m for m in roster.member_ids if p.w2 is not None and m not in p.w2],
        "headroom": [h.to_dict() for h in p.headroom1 + p.headroom2],
        "exclusions": sorted(f"{f}/{z}" for f, z in p.excluded), "stop": p.stop,
        "replacements": [r.to_dict() for r in outcome.registry.screening.replacements],  # type: ignore[union-attr]
        "bar_deaths": [{"world": w, "episode": ep, "member": m}
                       for w, ep, m in outcome.registry.screening.bar_deaths or ()],  # type: ignore[union-attr]
        "roster_version": None if outcome.roster is None else outcome.roster.version,
        "pruned": ([] if outcome.roster is None or outcome.roster.pruning is None
                   else list(outcome.roster.pruning.dropped)),
        "families": (None if outcome.calibrations is None else
                     {c.family: c.payload() for c in outcome.calibrations}),
        "q": q, **outcome.tables,
    }
    (out / "screening.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_table(out / "headroom.csv", [h.to_dict() for h in p.headroom1 + p.headroom2])
    att = outcome.tables["attainability"]
    _write_table(out / "attainability.csv",
                 [{"cell": "size_regime", "family": "", "excluded": "", **r}
                  for r in att["by_size_regime"]]  # type: ignore[index]
                 + [{"cell": "family", **r} for r in att["by_family"]])  # type: ignore[index]
    _write_table(out / "routing.csv", outcome.tables["routing"]["worlds"])  # type: ignore[index]
    if outcome.passed:
        frozen = out / "frozen"
        frozen.mkdir(exist_ok=True)
        names = {k: f"{k}_v{tag}.json" for k in ("worlds", "roster", "side_effect_params")}
        save_registry(outcome.registry, frozen / names["worlds"])
        assert outcome.roster is not None and outcome.params_json is not None
        save_roster(outcome.roster, frozen / names["roster"])
        (frozen / names["side_effect_params"]).write_text(outcome.params_json)
        spec["frozen"] = {names["worlds"]: outcome.registry.version,
                          names["roster"]: outcome.roster.version,
                          names["side_effect_params"]: json.loads(outcome.params_json)[
                              "content_hash"]}
        spec["screened_on"] = screened_on
    spec["status"] = "complete" if outcome.passed else "headroom-failed"
    spec["result"] = report
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(SCREEN_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "",
            status=("HEADROOM PASSED" if outcome.passed
                    else f"HEADROOM FAILED (no frozen files): {p.stop}"),
            driver=DRIVER))
    log(f"[gate] two-wave screen {'(dry run) ' if dry_run else ''}"
        f"{'passed' if outcome.passed else 'FAILED'}; excluded "
        f"{sorted(f'{f}/{z}' for f, z in p.excluded) or 'none'}")
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
    # Defaults: the frozen v2 set (add-shocks-i-drift D8, task 10.1). The v1 set stays
    # selectable by passing all three v1 paths; recompute reads only spec.json.
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                    help="default: the packaged frozen worlds_v2.json; a screen needs a draft")
    ap.add_argument("--roster", default=str(DEFAULT_ROSTER),
                    help="default: the packaged frozen roster_v2.json; a screen needs a draft")
    ap.add_argument("--params", default=str(DEFAULT_PARAMS),
                    help="side-effect parameter file (X, Y, q, content hash); default: the "
                         "packaged frozen side_effect_params_v2.json")
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
    ap.add_argument("--member-waves", choices=(MEMBER_WAVES_V3,),
                    help="screen: the two-wave v3 screen (design D3); needed for a v3+ registry")
    ap.add_argument("--worlds-per-family", type=int,
                    help="two-wave screen dry run: per family the first 2N calm and N shocked "
                         "gate worlds (the gate's regime ratio)")
    ap.add_argument("--region-shocked-episodes", type=int,
                    help="two-wave screen: D3 reduction 1, shocked region worlds at N episodes")
    ap.add_argument("--reuse-rows", help="two-wave screen: D3 reduction 2, reuse the v2-draft "
                                         "members' v2-world rows of this EXP-0047 dir")
    ap.add_argument("--shock-profile", help="two-wave screen dry run: shocks-i-smoke")
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
            dry_run=args.dry_run, member_waves=args.member_waves,
            worlds_per_family=args.worlds_per_family,
            region_shocked_episodes=args.region_shocked_episodes, reuse_rows=args.reuse_rows,
            shock_profile=args.shock_profile)
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
