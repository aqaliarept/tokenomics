"""EXP-0048 / EXP-0049 driver: `python -m tokenomics.eval.routing_audit` (add-m3-development-audit
design D2, "EXP-0048 grid" and "Informed spreading routers and trained-agent trace";
development-audit "Baseline anomaly diagnosed before the teacher audit").

`--plan` picks the run's router set. `exp0048` (default): the six roster members and the 15
diagnostic routers of EXP-0048. `exp0049` (task 3.2): the six roster members, the references
`round_robin`, `even_split_rotating`, `proportional_whole`, `random_whole`, the informed
spreading routers `two_choice_spare`, `spare_weighted_whole`, `spare_rotating_whole`, and two
trained subjects (`--prod-ckpt`, `--orderer-ckpt`): `entity_planner/trained_orderer` (the
portable orderer scorer with the routers' producer) and `portable` (the full portable stack,
its own producer). Both trained subjects are deterministic (scorer logits, decoded without
sampling), so they run once, like the deterministic routers. Their checkpoints are hashed and
trial-loaded before any job (a failure is a `run` row, as in the gate) and every worker checks
the hash of the file it loads.

EXP-0048's question: why does the `random` ordering baseline set the hindsight baseline on most v2
development world-episodes? Every router runs with the `entity_planner` producer (the producer
of every roster v2 member) on every registry development gate world-episode, horizon 100,
routing tracer on (`eval/routing_trace.py`):
- the six roster members, each under the contract its roster entry records (roster v2:
  `entity_planner/obs_price` legacy, the rest joint_stack), as references whose rows must
  reproduce EXP-0047; `random` runs R replicates with the gate's replicate seeds;
- the plan's diagnostic routers of `eval/routing_diagnostics.py` under joint_stack, the
  deterministic ones once, the stochastic ones R times with the same replicate seeds.

Rows use the gate row format (`gate.row_from_stats`) plus `trace_*` columns, and are paired as
they arrive: every run of a (world, episode) must see the same demand and shock digests
(`gate.DigestMismatch`, a `run` row in `contract_failures.csv`, run invalid). Job failures and
the step-loop contract checks are `eval/contract.py`'s. Resume, spec and grid pinning reuse the
gate's machinery (`gate_cli._prepare_resume` / `_open_spec` / `check_rows`). Output: the EXP
layout `spec.json`, `episodes.csv`, `routing_metrics.csv`, `edge.csv`,
`contract_failures.csv`, a `RESULT.md` skeleton.

The edge arithmetic is pure (`edge_table`, `metrics_table`) and pre-registered in D2: a
world-episode's edge is random's replicate-mean units minus the best deterministic roster
member's units; a router recovers the edge in a size x regime cell when its equal-weight world
mean gain over that best member is at least 0.8 of random's equal-weight world mean edge; it
recovers it in "most cells" at 2 of the 3 random-dominated cells (12-shocked, 42-calm,
42-shocked). `edge.csv` also holds the (f) whole/split x deterministic/random 2x2 at matched
split degree (tolerance `MATCHED_SPLIT_TOLERANCE`, chosen in this driver, not pre-registered)
and the (e) replicate check.

A real run needs the frozen roster, a screened registry, a 40-hex source commit, horizon 100,
every router of the plan, the roster's R and no filters; anything else needs `--dry-run`
(marked as not an acceptance read). DIAGNOSTIC ONLY: the trace reads hidden shock state
(outage reasons); nothing here feeds a policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .contract import (
    CHECKPOINT,
    DIGEST,
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
from .gate import CSV_COLUMNS, DigestMismatch, row_from_dict, row_from_stats, row_to_dict
from .gate_cli import (
    CONTRACT_INVALID,
    INVALID,
    Job,
    Key,
    _open_spec,
    _prepare_resume,
    _write_spec,
    blas_threads,
    check_rows,
    job_grid,
    job_key,
    pin_blas,
    read_rows,
    test_only_problem,
)
from .provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .registry import DEFAULT_REGISTRY, DEVELOPMENT, Registry, WorldEntry, load_registry
from .roster import DEFAULT_ROSTER, JOINT_STACK, RANDOM_POLICY, Roster, load_roster, member_id
from .routing_diagnostics import DETERMINISTIC, DIAGNOSTIC_ROUTERS, ROUTER_KIND, STOCHASTIC
from .routing_trace import RoutingTracer
from .seeding import replicate_seed

DRIVER = "tokenomics.eval.routing_audit"
PRODUCER = "entity_planner"  # the producer of every roster v2 member (D2)
HORIZON = 100

# D2, pre-registered: recovery threshold, the random-dominated cells, "most" of them.
RECOVERY_THRESHOLD = 0.8
DOMINANT_CELLS = ((12, "shocked"), (42, "calm"), (42, "shocked"))
MOST_CELLS = 2
# Recovered fractions are ratios of float means; "at least 0.8" allows this much rounding.
FLOAT_TOLERANCE = 1e-9
# (f) 2x2: routers per (split class, kind). "Matched split degree": within a split class the
# deterministic and random routers' mean consumer split degree differ by at most
# MATCHED_SPLIT_TOLERANCE. A driver choice (task 1.2), NOT pre-registered in D2: spec.json lists
# it under `driver_choices` and edge.csv reports the gap itself, so a reader can apply another.
SPLIT_CLASSES: dict[tuple[str, str], tuple[str, ...]] = {
    ("whole", DETERMINISTIC): ("proportional_whole", "round_robin"),
    ("whole", STOCHASTIC): ("random_whole",),
    ("split", DETERMINISTIC): ("even_split", "even_split_rotating"),
    ("split", STOCHASTIC): ("even_split_random_leftover",),
}
MATCHED_SPLIT_TOLERANCE = 0.05

# Router sets per plan (module docstring). EXP-0048's 15 diagnostic routers are listed, not
# taken from DIAGNOSTIC_ROUTERS, so routers added later never change that grid.
EXP0048 = "exp0048"
EXP0049 = "exp0049"
TRAINED_ORDERER = "trained_orderer"
PORTABLE = "portable"
PLANS: dict[str, dict[str, tuple[str, ...]]] = {
    EXP0048: {"diagnostic": (
        "round_robin", "even_split_rotating", "proportional_load_aware", "proportional_whole",
        "random_scaled_0.5", "random_scaled_1", "random_scaled_2", "random_scaled_5",
        "random_fixed_episode", "blend_noise_0.15", "blend_noise_0.6", "random_consumer_only",
        "random_producer_only", "random_whole", "even_split_random_leftover"),
        "trained": ()},
    EXP0049: {"diagnostic": (
        "round_robin", "even_split_rotating", "proportional_whole", "random_whole",
        "two_choice_spare", "spare_weighted_whole", "spare_rotating_whole"),
        "trained": (TRAINED_ORDERER, PORTABLE)},
}
PLAN_EXPERIMENT = {EXP0048: "EXP-0048", EXP0049: "EXP-0049"}
DEFAULT_PLAN = EXP0048

TRACE_PREFIX = "trace_"
TRACE_SCALARS = tuple(RoutingTracer().summary())
TRACE_SERIES = ("fund_share_by_producer", "cut_off_price_by_producer", "capacity_ratio_trace")
TRACE_COLUMNS = tuple(TRACE_PREFIX + k for k in TRACE_SCALARS + TRACE_SERIES)
EPISODE_COLUMNS = CSV_COLUMNS + TRACE_COLUMNS


# --- routers and jobs ---


def _trained_entry(name: str) -> dict:
    """`trained_orderer`: entity_planner + the portable orderer; `portable`: the full portable
    stack. Both joint_stack (the scorer reads the product-stream view) and deterministic."""
    producing = PRODUCER if name == TRAINED_ORDERER else PORTABLE
    return {"router": name, "subject": member_id(PRODUCER, name) if name == TRAINED_ORDERER
            else PORTABLE, "producing": producing, "ordering": TRAINED_ORDERER,
            "contract": JOINT_STACK, "random": False, "roster": False, "kind": DETERMINISTIC,
            "trained": True}


def router_table(roster: Roster, routers: Sequence[str] | None = None,
                 plan: str = DEFAULT_PLAN) -> list[dict]:
    """Every router of the run, in order: the roster members (their recorded contract and
    random flag), then the plan's diagnostic routers (joint_stack; random = stochastic), then
    its trained subjects. `routers` keeps a subset by router name (dry run)."""
    if plan not in PLANS:
        raise ValueError(f"unknown plan {plan!r}; known: {sorted(PLANS)}")
    table = [{"router": m.ordering, "subject": m.id, "producing": m.producing,
              "ordering": m.ordering, "contract": m.contract, "random": m.random,
              "roster": True, "kind": STOCHASTIC if m.random else DETERMINISTIC,
              "trained": False}
             for m in roster.members]
    table += [{"router": name, "subject": member_id(PRODUCER, name), "producing": PRODUCER,
               "ordering": name, "contract": JOINT_STACK,
               "random": ROUTER_KIND[name] == STOCHASTIC, "roster": False,
               "kind": ROUTER_KIND[name], "trained": False}
              for name in PLANS[plan]["diagnostic"]]
    table += [_trained_entry(name) for name in PLANS[plan]["trained"]]
    if routers is not None:
        known = {r["router"] for r in table}
        unknown = sorted(set(routers) - known)
        if unknown:
            raise ValueError(f"unknown routers {unknown}; known: {sorted(known)}")
        table = [r for r in table if r["router"] in set(routers)]
    return table


def select_worlds(registry: Registry, *, sizes: Sequence[int] | None = None,
                  regimes: Sequence[str] | None = None, max_worlds: int | None = None,
                  shock_profile: str | None = None) -> list[WorldEntry]:
    """Development gate worlds in registry order; `max_worlds` keeps the first N of each
    size x regime cell. `shock_profile` rewrites the shocked worlds' profile (a dry run's
    `shocks-i-smoke`)."""
    out: list[WorldEntry] = []
    per_cell: dict[tuple[int, str], int] = defaultdict(int)
    for e in registry.gate_worlds(DEVELOPMENT):
        if (sizes and e.size not in sizes) or (regimes and e.regime not in regimes):
            continue
        if max_worlds is not None and per_cell[(e.size, e.regime)] >= max_worlds:
            continue
        per_cell[(e.size, e.regime)] += 1
        params = e.generator.get("params", {})
        if shock_profile and "shock_profile" in params:
            e = replace(e, generator={**e.generator,
                                      "params": {**params, "shock_profile": shock_profile}})
        out.append(e)
    return out


def replicates_of(router: Mapping[str, object], replicates: int) -> int:
    return replicates if router["random"] else 1


def build_jobs(worlds: Sequence[WorldEntry], routers: Sequence[Mapping[str, object]],
               replicates: int, horizon: int,
               episodes_per_world: int | None = None,
               checkpoints: Mapping[str, Mapping[str, str]] | None = None) -> list[Job]:
    """One job per (world, episode, router, replicate); a random router's replicate r gets the
    gate's action seed `replicate_seed(world, episode, r)`, a deterministic one keeps the
    legacy coupling (None), exactly as the gate runs roster members. A trained subject's job
    carries `checkpoints` (role -> {path, sha256})."""
    jobs: list[Job] = []
    for e in worlds:
        episodes = e.episode_seeds[:episodes_per_world] if episodes_per_world else e.episode_seeds
        for ep in episodes:
            for r in routers:
                for rep in range(replicates_of(r, replicates)):
                    jobs.append({
                        "world": e.id, "size": e.size, "regime": e.regime,
                        "world_seed": e.world_seed, "generator": e.generator, "episode": ep,
                        "subject": r["subject"], "replicate": rep, "producing": r["producing"],
                        "ordering": r["ordering"], "contract": r["contract"], "horizon": horizon,
                        "action_seed": replicate_seed(e.id, ep, rep) if r["random"] else None,
                        "checkpoints": dict(checkpoints or {}) if r["trained"] else None})
    return jobs


def expected_keys(grid: Mapping[str, Mapping], routers: Sequence[Mapping[str, object]],
                  replicates: int) -> set[Key]:
    return {(w, ep, r["subject"], rep) for w, g in grid.items() for ep in g["episodes"]
            for r in routers for rep in range(replicates_of(r, replicates))}


# --- worker ---

_CONFIGS: dict[str, object] = {}
_TRAINED: dict[tuple[str, str, str], Policy] = {}
ROLES = ("producer", "orderer")


def ordering_policy(name: str) -> Policy:
    return ORDERING_BASELINES[name] if name in ORDERING_BASELINES else DIAGNOSTIC_ROUTERS[name]


def load_trained_policy(role: str, path: str) -> Policy:
    """The portable producer (`producer`) or orderer (`orderer`) as a `Policy`; torch is
    imported here only, pinned to one thread."""
    import torch

    torch.set_num_threads(1)
    if role == "producer":
        from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

        return scorer_producing_policy(load_producer_score_model(path))
    from ..train.orderer_scorer import load_scorer, scorer_ordering_policy

    return scorer_ordering_policy(load_scorer(path))


def trained_policy(job: Job, role: str) -> Policy:
    """The job's checkpoint for `role`, loaded once per process after checking its sha256
    against the parent's pin."""
    ckpt = job["checkpoints"][role]
    key = (role, ckpt["path"], ckpt["sha256"])
    if key not in _TRAINED:
        digest = sha256_file(ckpt["path"])
        if digest != ckpt["sha256"]:
            raise ValueError(f"{role} checkpoint {ckpt['path']} has sha256 {digest}, "
                             f"pinned {ckpt['sha256']}")
        _TRAINED[key] = load_trained_policy(role, ckpt["path"])
    return _TRAINED[key]


def job_policies(job: Job) -> tuple[Policy, Policy]:
    prod = (trained_policy(job, "producer") if job["producing"] == PORTABLE
            else PRODUCING_BASELINES[job["producing"]])
    order = (trained_policy(job, "orderer") if job["ordering"] == TRAINED_ORDERER
             else ordering_policy(job["ordering"]))
    return prod, order


def checked_checkpoints(out: Path, paths: Mapping[str, str], dry_run: bool,
                        log: Callable[[str], None]) -> dict[str, dict[str, str]]:
    """role -> {path, sha256} after a trial load in the parent (the gate's run-level check):
    a checkpoint that does not hash or load is a `run` row in `contract_failures.csv`,
    `spec.json` invalid, and `ContractRunError` before any job runs."""
    try:
        pins = {role: {"path": str(paths[role]), "sha256": sha256_file(paths[role])}
                for role in ROLES}
        for role in ROLES:
            load_trained_policy(role, paths[role])
    except Exception as exc:  # noqa: BLE001 - recorded, then the run ends invalid
        out.mkdir(parents=True, exist_ok=True)
        append_failure(out, run_failure(CHECKPOINT, exc))
        spec_path = out / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {
            "experiment": out.name, "driver": DRIVER,
            "kind": "dry-run" if dry_run else "audit",
            "inputs": {role: {"path": str(paths.get(role)), "sha256": None} for role in ROLES}}
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(read_failures(out))}
        _write_spec(out, spec)
        log(f"[routing_audit] INVALID: checkpoint failed to hash or load ({exc})")
        raise ContractRunError(f"checkpoint failed to hash or load ({type(exc).__name__}: "
                               f"{exc})") from exc
    return pins


def run_job(job: Job) -> dict[str, str]:
    """One traced episode's row (gate columns + `trace_*`), or its failure row when anything
    in the job raises or a contract check fails (as `gate_cli.run_job`)."""
    try:
        return _run_job(job)
    except Exception as exc:  # noqa: BLE001 - every job failure is recorded, never fatal
        return job_failure(job, exc).to_dict()


def _run_job(job: Job) -> dict[str, str]:
    from ..env import TokenomicsAECEnv
    from .joint_stack import joint_stack_env
    from .runner import run_episode
    from .task_archive import REPRODUCERS

    key = json.dumps([job["world"], job["world_seed"], job["generator"]], sort_keys=True)
    if key not in _CONFIGS:
        g = job["generator"]
        cfg = REPRODUCERS[g["generator"]](job["world_seed"], **g.get("params", {}))
        cfg.validate()
        _CONFIGS[key] = cfg
    cfg = _CONFIGS[key]
    horizon = job["horizon"]
    env = (joint_stack_env(cfg, horizon) if job["contract"] == JOINT_STACK
           else TokenomicsAECEnv(config=cfg, horizon=horizon))
    tracer = RoutingTracer()
    tracer.attach(env)
    monitor = ContractMonitor(env)
    prod, order = job_policies(job)
    run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"],
                monitor=monitor)
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=horizon)
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     job["subject"], job["replicate"], stats))
    row.update(trace_cells(tracer))
    return row


def trace_cells(tracer: RoutingTracer) -> dict[str, str]:
    cells = {TRACE_PREFIX + k: repr(float(v)) for k, v in tracer.summary().items()}
    series = tracer.series()
    cells.update({TRACE_PREFIX + k: json.dumps(series[k], sort_keys=True, separators=(",", ":"))
                  for k in TRACE_SERIES})
    return cells


# --- executor (the gate's `_execute`, with trace columns) ---


def _append(path: Path, row: Mapping[str, str]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EPISODE_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def _init_worker() -> None:
    pin_blas()


def execute(out: Path, spec: dict, done_keys: set[Key], done_rows: Sequence[Mapping[str, str]],
            jobs: Sequence[Job], workers: int, run_job_fn: Callable[[Job], dict[str, str]],
            log: Callable[[str], None]) -> None:
    """Run every job not in `done_keys`, appending each row as it lands. Rows of one (world,
    episode) are paired on demand and shock digests as they arrive; a mismatch is one `run`
    row, `spec.json` invalid, and raises. A failed job is a `job` row; the others go on."""
    episodes = out / "episodes.csv"
    seen: dict[tuple[str, int], dict[str, tuple[str, str]]] = {}

    def pair(d: Mapping[str, str]) -> None:
        where = (d["world"], int(d["episode"]))
        ref = seen.setdefault(where, {})
        try:
            for what in ("demand_digest", "shock_digest"):
                first = ref.setdefault(what, (d[what], d["subject"]))
                if d[what] != first[0]:
                    raise DigestMismatch(
                        f"world {where[0]} episode {where[1]}: {d['subject']} replicate "
                        f"{d['replicate']} saw {what.replace('_', ' ')} {d[what]}, {first[1]} "
                        f"saw {first[0]}")
        except DigestMismatch as exc:
            append_failure(out, run_failure(DIGEST, exc))
            spec["status"] = INVALID
            spec["result"] = {"status": INVALID, "contract_failures": len(read_failures(out))}
            _write_spec(out, spec)
            raise

    for d in done_rows:
        pair(d)
    todo = [j for j in jobs
            if job_key(j["world"], j["episode"], j["subject"], j["replicate"]) not in done_keys]
    log(f"[routing_audit] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, "
        f"{len(todo)} to run")

    def finish(d: dict[str, str]) -> None:
        if is_failure(d):
            f = failure_from_dict(d)
            append_failure(out, f)
            log(f"[routing_audit] CONTRACT FAILURE {f.world} ep={f.episode} {f.subject} "
                f"r{f.replicate}: {f.check} {f.error_type}")
            return
        row_from_dict(d)  # the gate's row validation
        pair(d)
        _append(episodes, d)
        log(f"[routing_audit] {d['world']} ep={d['episode']} {d['subject']} r{d['replicate']} "
            f"units={float(d['units']):g} dead={d['dead']}")

    pin_blas()
    if spec.get("blas_threads") is None:
        spec["blas_threads"] = blas_threads()
    if workers <= 1 or len(todo) <= 1:
        for job in todo:
            finish(run_job_fn(job))
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=_init_worker) as pool:
        futures = [pool.submit(run_job_fn, j) for j in todo]
        try:
            for fut in as_completed(futures):
                finish(fut.result())
        except BaseException:
            for f in futures:
                f.cancel()
            raise


def read_episode_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --- pure analysis ---


def _num(s: str) -> float | None:
    return None if s in ("", "None") else float(s)


def _episode_values(rows: Sequence[Mapping[str, str]], routers: Sequence[Mapping[str, object]],
                    ) -> dict[tuple[str, int], dict[str, dict[str, object]]]:
    """(world, episode) -> router -> {size, regime, units (replicate mean), replicates (units
    by replicate), dead, metrics (replicate-mean trace scalars)}."""
    by_subject = {r["subject"]: r["router"] for r in routers}
    grouped: dict[tuple[str, int], dict[str, list[Mapping[str, str]]]] = defaultdict(
        lambda: defaultdict(list))
    for d in rows:
        name = by_subject.get(d["subject"])
        if name is not None:
            grouped[(d["world"], int(d["episode"]))][str(name)].append(d)
    out: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for key, per_router in grouped.items():
        out[key] = {}
        for name, ds in per_router.items():
            ds = sorted(ds, key=lambda d: int(d["replicate"]))
            metrics = {k: fmean(float(d[TRACE_PREFIX + k]) for d in ds) for k in TRACE_SCALARS}
            out[key][name] = {
                "size": int(ds[0]["size"]), "regime": ds[0]["regime"],
                "units": fmean(float(d["units"]) for d in ds),
                "replicates": [float(d["units"]) for d in ds],
                "dead": fmean(1.0 if d["dead"] == "true" else 0.0 for d in ds),
                "metrics": metrics}
    return out


def _world_mean(values: Mapping[str, Sequence[float]]) -> float:
    """Equal-weight world mean: mean over worlds of each world's episode mean."""
    return fmean(fmean(v) for v in values.values())


EDGE_COLUMNS = (
    "level", "size", "regime", "world", "episode", "router", "kind", "split_class",
    "units", "best_det_units", "best_det_member", "random_units", "random_r0_units",
    "random_min_units", "edge", "edge_r1", "random_worst_beats_best_det", "gain",
    "recovered_fraction", "recovers", "n_worlds", "n_episodes", "worst_beats_count",
    "edge_positive", "dominant_cells_present", "cells_recovered", "recovers_most",
    "split_degree_consumer", "split_degree_producer", "split_share_consumer",
    "wasted_units_consumer", "wasted_units_producer", "split_degree_gap", "matched_split_degree",
)


def _fraction(gain: float, edge: float) -> float | None:
    return gain / edge if edge > 0 else None


def recovers(fraction: float | None) -> bool:
    """D2: a recovered fraction of at least `RECOVERY_THRESHOLD` (float rounding allowed)."""
    return fraction is not None and fraction >= RECOVERY_THRESHOLD - FLOAT_TOLERANCE


def edge_table(rows: Sequence[Mapping[str, str]], routers: Sequence[Mapping[str, object]],
               ) -> list[dict[str, object]]:
    """`edge.csv` (module docstring). Levels: `episode` (per world-episode and router: gain over
    the best deterministic roster member, recovered fraction of random's edge, the (e) replicate
    check), `cell` (per size x regime and router, equal-weight world means), `rule` (per router:
    cells of `DOMINANT_CELLS` recovered, "most" = `MOST_CELLS`), `split_2x2` (per cell, the (f)
    routers with their split degree and the deterministic-random gap in their split class).
    Needs the random roster member and at least one deterministic roster member."""
    info = {r["router"]: r for r in routers}
    det_members = [r["router"] for r in routers if r["roster"] and not r["random"]]
    if RANDOM_POLICY not in info or not det_members:
        raise ValueError("the edge needs the random roster member and a deterministic one")
    values = _episode_values(rows, routers)
    out: list[dict[str, object]] = []
    # cell -> router -> world -> per-episode values
    gains: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    edges: dict = defaultdict(lambda: defaultdict(list))
    units: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metric: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    worst_beats: dict = defaultdict(int)
    for (world, ep), per in sorted(values.items()):
        some = next(iter(per.values()))
        cell = (some["size"], some["regime"])
        best = max(per[m]["units"] for m in det_members)
        best_names = ";".join(m for m in det_members if per[m]["units"] == best)
        rnd = per[RANDOM_POLICY]
        reps = rnd["replicates"]
        edge = rnd["units"] - best
        beats = min(reps) > best
        worst_beats[cell] += int(beats)
        edges[cell][world].append(edge)
        for name, v in per.items():
            gain = v["units"] - best
            gains[cell][name][world].append(gain)
            units[cell][name][world].append(v["units"])
            for k in ("split_degree_consumer", "split_degree_producer", "split_share_consumer",
                      "wasted_units_consumer", "wasted_units_producer"):
                metric[cell][name][k][world].append(v["metrics"][k])
            out.append({
                "level": "episode", "size": cell[0], "regime": cell[1], "world": world,
                "episode": ep, "router": name, "kind": info[name]["kind"],
                "units": v["units"], "best_det_units": best, "best_det_member": best_names,
                "random_units": rnd["units"], "random_r0_units": reps[0],
                "random_min_units": min(reps), "edge": edge, "edge_r1": reps[0] - best,
                "random_worst_beats_best_det": beats, "gain": gain,
                "recovered_fraction": gain / edge if edge != 0 else None,
                **{k: v["metrics"][k] for k in ("split_degree_consumer", "split_degree_producer",
                                                "split_share_consumer", "wasted_units_consumer",
                                                "wasted_units_producer")}})
    recovered: dict[tuple[tuple[int, str], str], float | None] = {}
    cell_metrics: dict = {}
    for cell in sorted(edges):
        edge_mean = _world_mean(edges[cell])
        n_episodes = sum(len(v) for v in edges[cell].values())
        for name in info:
            if name not in gains[cell]:
                continue
            gain_mean = _world_mean(gains[cell][name])
            frac = _fraction(gain_mean, edge_mean)
            recovered[(cell, name)] = frac
            m = {k: _world_mean(metric[cell][name][k]) for k in metric[cell][name]}
            cell_metrics[(cell, name)] = m
            out.append({
                "level": "cell", "size": cell[0], "regime": cell[1], "router": name,
                "kind": info[name]["kind"], "units": _world_mean(units[cell][name]),
                "edge": edge_mean, "gain": gain_mean, "recovered_fraction": frac,
                "recovers": recovers(frac),
                "edge_positive": edge_mean > 0, "n_worlds": len(edges[cell]),
                "n_episodes": n_episodes,
                "worst_beats_count": worst_beats[cell] if name == RANDOM_POLICY else None, **m})
    present = [c for c in DOMINANT_CELLS if c in edges]
    for name in info:
        fracs = [recovered.get((c, name)) for c in present]
        got = [f for f in fracs if recovers(f)]
        out.append({
            "level": "rule", "router": name, "kind": info[name]["kind"],
            "dominant_cells_present": ";".join(f"{s}-{r}" for s, r in present),
            "cells_recovered": len(got), "recovers_most": len(got) >= MOST_CELLS})
    for cell in sorted(edges):
        for split in ("whole", "split"):
            degree = {}
            for kind in (DETERMINISTIC, STOCHASTIC):
                names = [n for n in SPLIT_CLASSES[(split, kind)] if (cell, n) in cell_metrics]
                if names:
                    degree[kind] = fmean(cell_metrics[(cell, n)]["split_degree_consumer"]
                                         for n in names)
            gap = (abs(degree[DETERMINISTIC] - degree[STOCHASTIC])
                   if len(degree) == 2 else None)
            for kind in (DETERMINISTIC, STOCHASTIC):
                for name in SPLIT_CLASSES[(split, kind)]:
                    if (cell, name) not in cell_metrics:
                        continue
                    frac = recovered[(cell, name)]
                    out.append({
                        "level": "split_2x2", "size": cell[0], "regime": cell[1],
                        "router": name, "kind": kind, "split_class": split,
                        "units": _world_mean(units[cell][name]), "recovered_fraction": frac,
                        "recovers": recovers(frac),
                        **cell_metrics[(cell, name)], "split_degree_gap": gap,
                        "matched_split_degree": (None if gap is None
                                                 else gap <= MATCHED_SPLIT_TOLERANCE)})
    return out


METRIC_COLUMNS = ("size", "regime", "router", "kind", "contract", "n_worlds", "n_episodes",
                  "units", "dead_rate") + TRACE_SCALARS


def metrics_table(rows: Sequence[Mapping[str, str]], routers: Sequence[Mapping[str, object]],
                  ) -> list[dict[str, object]]:
    """`routing_metrics.csv`: per size x regime and router, equal-weight world means of units,
    death rate and every tracer scalar (replicate means first)."""
    values = _episode_values(rows, routers)
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for (world, _), per in values.items():
        for name, v in per.items():
            a = acc[(v["size"], v["regime"])][name]
            a["units"][world].append(v["units"])
            a["dead_rate"][world].append(v["dead"])
            for k, x in v["metrics"].items():
                a[k][world].append(x)
    out = []
    for cell in sorted(acc):
        for r in routers:
            a = acc[cell].get(r["router"])
            if a is None:
                continue
            out.append({"size": cell[0], "regime": cell[1], "router": r["router"],
                        "kind": r["kind"], "contract": r["contract"],
                        "n_worlds": len(a["units"]),
                        "n_episodes": sum(len(v) for v in a["units"].values()),
                        **{k: _world_mean(a[k]) for k in ("units", "dead_rate") + TRACE_SCALARS}})
    return out


def _cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _cell(r.get(c)) for c in columns})


# --- run ---

QUESTIONS = {
    EXP0048: """\
Why does the `random` ordering baseline set the hindsight baseline on most registry v2
development world-episodes (design D2)? Each router isolates one explanation: (a) packing under
atomic acceptance, (b) temporal variation versus randomness, (c) rule defects, (d) the
investment-fund channel, (e) the scoring of random replicates, (f) the split penalty under
whole-order completion.""",
    EXP0049: """\
Can a router that reads producer data (spare capacity) spread whole orders better than the
blind `round_robin` / `random_whole`, and did the trained portable orderer learn random's trick
(whole orders, a receiver that varies over time, no herding) or `proportional`'s (design D2,
"Informed spreading routers and trained-agent trace")?""",
}

RESULT_SKELETON = """# {title}

{banner}**Status: {status}.** {n_rows} episode rows over {n_world_episodes} world-episodes,
{n_routers} routers (R = {replicates}), horizon {horizon}; registry {registry}, roster {roster}.
Produced by `python -m {driver} --plan {plan}` (see `spec.json`).

## Question

{question}

## Result

TODO: plain-English reading of `edge.csv` (per size x regime cell: random's edge, each
router's recovered fraction, the 0.8 / 2-of-3-cells rule, the (f) 2x2 at matched split degree,
the (e) replicate check; the matched-split tolerance {tolerance} is a driver choice, not
pre-registered in D2) and `routing_metrics.csv` (rejection reasons, fund channel, wasted
completed units).

## Conclusions

TODO: the diagnosed cause(s) with evidence, and what changes per design D2.
"""
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered worlds, episodes, routers or "
              "replicates, another horizon or shock profile, or no source commit; see "
              "`spec.json`.\n\n")


def _is_commit(s: str) -> bool:
    return len(s) == 40 and all(c in "0123456789abcdef" for c in s)


def version_pair_problem(registry: Registry, roster: Roster) -> str | None:
    """Why a real run may not pair this registry and roster (the gate's "Mixed versions
    refused", `gate_cli.version_set_problem`, without side-effect parameters, which this driver
    does not read); None = admitted. A v2+ file must record `screened_on`, and the two must
    record the same one (one screen of one draft); v1 files record none and pair only with
    each other."""
    files = (("registry", registry.name, registry.version_tag, registry.screened_on),
             ("roster", roster.name, roster.version_tag, roster.screened_on))
    for what, name, tag, on in files:
        if tag >= 2 and on is None:
            return f"mixed versions: {what} {name!r} is v{tag} but records no screened_on"
    if registry.screened_on != roster.screened_on:
        return (f"mixed versions: roster {roster.name!r} (screened_on {roster.screened_on}) does "
                f"not pair with registry {registry.name!r} (screened_on {registry.screened_on})")
    return None


def real_run_problems(*, registry: Registry, roster: Roster, worlds: Sequence[WorldEntry],
                      horizon: int, replicates: int, source_commit: str,
                      filters: Mapping[str, object]) -> list[str]:
    problems = [msg for bad, msg in (
        (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
        (bool(registry.admission_problems(DEVELOPMENT)),
         "; ".join(registry.admission_problems(DEVELOPMENT))),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (replicates != roster.replicates, f"R = {replicates} != roster R {roster.replicates}"),
    ) if bad]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    smoke = test_only_problem(worlds)
    if smoke:
        problems.append(smoke)
    mixed = version_pair_problem(registry, roster)
    if mixed:
        problems.append(mixed)
    return problems


def run_audit(*, out_dir: str | Path, registry: Registry, roster: Roster, workers: int = 1,
              horizon: int = HORIZON, replicates: int | None = None,
              sizes: Sequence[int] | None = None, regimes: Sequence[str] | None = None,
              max_worlds: int | None = None, episodes_per_world: int | None = None,
              routers: Sequence[str] | None = None, shock_profile: str | None = None,
              source_commit: str = "", dry_run: bool = False, plan: str = DEFAULT_PLAN,
              prod_ckpt: str | None = None, orderer_ckpt: str | None = None,
              run_job_fn: Callable[[Job], dict[str, str]] = run_job,
              log: Callable[[str], None] = print) -> int:
    """Run (or resume) the plan's experiment (EXP-0048 / EXP-0049) into `out_dir`; return the
    exit status (0, or `CONTRACT_INVALID` when contract failures were recorded or rows are
    missing)."""
    replicates = roster.replicates if replicates is None else replicates
    worlds = select_worlds(registry, sizes=sizes, regimes=regimes, max_worlds=max_worlds,
                           shock_profile=shock_profile)
    if not worlds:
        raise ValueError("no development gate worlds selected")
    table = router_table(roster, routers, plan)
    trained = any(r["trained"] for r in table)
    if trained and not (prod_ckpt and orderer_ckpt):
        raise ValueError("the trained subjects need --prod-ckpt and --orderer-ckpt")
    if not dry_run:
        problems = real_run_problems(
            registry=registry, roster=roster, worlds=worlds, horizon=horizon,
            replicates=replicates, source_commit=source_commit,
            filters={"sizes": sizes, "regimes": regimes, "max_worlds": max_worlds,
                     "episodes_per_world": episodes_per_world, "routers": routers,
                     "shock_profile": shock_profile})
        if problems:
            raise ValueError(f"not a valid {PLAN_EXPERIMENT[plan]} run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    out = Path(out_dir)
    checkpoints = (checked_checkpoints(out, {"producer": prod_ckpt, "orderer": orderer_ckpt},
                                       dry_run, log) if trained else {})
    grid = job_grid(worlds, episodes_per_world)
    source = source_fingerprint(source_commit or None)
    pins = {
        "driver": DRIVER, "split": DEVELOPMENT, "horizon": horizon, "dry_run": dry_run,
        "registry_version": registry.version, "roster_version": roster.version,
        "registry_name": registry.name, "roster_name": roster.name,
        "registry_screened_on": registry.screened_on, "roster_screened_on": roster.screened_on,
        "plan": plan, "checkpoints": {k: v["sha256"] for k, v in checkpoints.items()},
        "producer": PRODUCER, "routers": table, "replicates": replicates,
        "shock_profile_override": shock_profile,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    expected = expected_keys(grid, table, replicates)
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "audit",
        "pins": pins, "roster": roster.payload() | {"content_hash": roster.version},
        "inputs": checkpoints, "source": source,
        "pre_registered": {"recovery_threshold": RECOVERY_THRESHOLD,
                           "dominant_cells": [list(c) for c in DOMINANT_CELLS],
                           "most_cells": MOST_CELLS,
                           "split_classes": {f"{s}/{k}": list(v)
                                             for (s, k), v in SPLIT_CLASSES.items()}},
        "driver_choices": {"matched_split_tolerance": MATCHED_SPLIT_TOLERANCE,
                           "float_tolerance": FLOAT_TOLERANCE},
    })
    ensure_failures_file(out)
    jobs = build_jobs(worlds, table, replicates, horizon, episodes_per_world, checkpoints)
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    execute(out, spec, done_keys, read_episode_dicts(episodes), jobs, workers, run_job_fn, log)

    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    failures = read_failures(out)
    banner = DRY_BANNER if dry_run else ""
    if failures or missing:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing)}
        _write_spec(out, spec)
        status = f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing row(s))"
        code = CONTRACT_INVALID
    else:
        dicts = read_episode_dicts(episodes)
        write_csv(out / "routing_metrics.csv", METRIC_COLUMNS, metrics_table(dicts, table))
        has_edge = any(r["router"] == RANDOM_POLICY for r in table) and any(
            r["roster"] and not r["random"] for r in table)
        edge = edge_table(dicts, table) if has_edge else []
        write_csv(out / "edge.csv", EDGE_COLUMNS, edge)
        spec["status"] = "complete"
        spec["result"] = {"status": "complete", "rows": len(rows), "contract_failures": 0,
                          "edge_rows": len(edge),
                          "edge_note": None if has_edge else
                          "edge.csv is header-only: the router subset lacks random or a "
                          "deterministic roster member"}
        _write_spec(out, spec)
        status = "complete"
        code = 0
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(RESULT_SKELETON.format(
            title=out.name, banner=banner, status=status, n_rows=len(rows),
            n_world_episodes=sum(len(g["episodes"]) for g in grid.values()),
            n_routers=len(table), replicates=replicates, horizon=horizon,
            registry=registry.version[:12], roster=roster.version[:12], driver=DRIVER,
            tolerance=MATCHED_SPLIT_TOLERANCE, plan=plan, question=QUESTIONS[plan]))
    log(f"[routing_audit] {'DRY RUN ' if dry_run else ''}{status}")
    return code


def _csv_ints(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",")] if s else None


def _csv_strs(s: str | None) -> list[str] | None:
    return [x for x in s.split(",") if x] if s else None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=f"python -m {DRIVER}", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--roster", default=str(DEFAULT_ROSTER))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=HORIZON, help="dry run: another horizon")
    ap.add_argument("--replicates", type=int, help="dry run: override R")
    ap.add_argument("--sizes", help="dry run, e.g. 12")
    ap.add_argument("--regimes", help="dry run: calm and/or shocked")
    ap.add_argument("--max-worlds", type=int,
                    help="dry run: the first N worlds of each size x regime cell")
    ap.add_argument("--episodes-per-world", type=int, help="dry run")
    ap.add_argument("--plan", choices=sorted(PLANS), default=DEFAULT_PLAN,
                    help="router set: exp0048 (default) or exp0049 (needs the checkpoints)")
    ap.add_argument("--prod-ckpt", help="exp0049: portable producer checkpoint "
                                        "(EXP-0035 producer_candidate.pt)")
    ap.add_argument("--orderer-ckpt", help="exp0049: portable orderer checkpoint "
                                           "(EXP-0012 orderer_scorer.pt)")
    ap.add_argument("--routers", help="dry run: comma list of router names")
    ap.add_argument("--shock-profile", help="dry run: rewrite shocked worlds' profile, e.g. "
                                            "shocks-i-smoke")
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--verify-source-commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke run: filters allowed, marked as not an acceptance read")
    args = ap.parse_args(argv)
    if args.regimes and set(_csv_strs(args.regimes) or []) - {"calm", "shocked"}:
        ap.error("--regimes takes calm and/or shocked")
    source_commit = resolve_verified_commit() if args.verify_source_commit else args.source_commit
    try:
        return run_audit(
            out_dir=args.out_dir, registry=load_registry(args.registry),
            roster=load_roster(args.roster), workers=args.workers, horizon=args.horizon,
            replicates=args.replicates, sizes=_csv_ints(args.sizes),
            regimes=_csv_strs(args.regimes), max_worlds=args.max_worlds,
            episodes_per_world=args.episodes_per_world, routers=_csv_strs(args.routers),
            shock_profile=args.shock_profile, source_commit=source_commit,
            dry_run=args.dry_run, plan=args.plan, prod_ckpt=args.prod_ckpt,
            orderer_ckpt=args.orderer_ckpt)
    except (DigestMismatch, ContractRunError) as exc:
        print(f"[routing_audit] INVALID run: {exc}", file=sys.stderr)
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
