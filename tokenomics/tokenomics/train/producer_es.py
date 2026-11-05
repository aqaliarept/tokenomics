"""M5 ES driver for the producer residual: `python -m tokenomics.train.producer_es`
(add-m5-potential-game-gnn task 2.2, design D3 / D5 / D6 / D7).

Training set (D7, supervisor decision 2026-11-05): M4's pinned ES training worlds and schedule
(EXP-0058 `bar/`: 64 worlds at 12 producers, 128 world-episodes with fresh episode seeds, the
16-world-episode monitor set, per-seed minibatches, the probe's minibatches), hash-checked. Every
job plays the producer under test with the frozen MWWk orderer (`mww_keyed`) at DrawKey(world,
episode, replicate), joint_stack, horizon 100; replicate 0 for the bar, the members and the monitor
(common random numbers: the key never names the member), 1..4 for the noise floor.

Modes:
- `bar` (L1): the base `entity_planner_buyahead_k4_rf3` on every training world-episode at r0 ->
  `bar.csv` (order_es's bar format: B_we = the base's consumer units, its guards and digests) and
  `base_rows.csv`. With `--atlas` and `--exp0079-episodes`, rf3 is also replayed on 2 EXP-0057
  world-episodes through the supply-trace job path that wrote EXP-0079, and must equal its rows
  (`crosscheck.csv`; design D7).
- `noise-floor` (L1, D5): the base re-keyed at r1..r4 plus r0 again on every world-episode. r0 must
  reproduce the bar's rows (`reproduction.csv`). The margin is order_es's v4 formula (c x sqrt 2 x
  the pooled within-world-episode SD of the run-down ratio and of each consumer's fill), with
  the base's self-flag rate against its own r0 under it (target <= 5%) -> `margins.json`.
- `probe` (L2): the ES estimate at theta = 0 on the two disjoint probe minibatches (24
  world-episodes each, M4's recorded probe stream) with one common noise draw; their cosine reads
  `signal` (>= 0.3) / `weak` (>= 0.1) / `no_signal` -> `probe.json`.
- `train` (L3 / L4): antithetic ES (PAIRS pairs, sigma per parameter), centered ranks over the 2n
  members, a disqualified member (death, or run-down / starvation beyond the margin where the
  base has none: order_es's relative rule against rf3) ranks worst, Adam ascent. The minibatch
  per generation comes from M4's recorded stream for the optimiser seed. One checkpoint per
  generation; resume needs every pin to match. The monitor (the centre and the bar on the
  16-world-episode monitor set) every MONITOR_EVERY generations.

Fitness (D3, supervisor decision 2026-11-05): consumer units only, relative to the bar,
units / B_we (order_es `we_fitness` with h = 0); no holding term. Holding units per cycle and the
investment rejection rate are reported per member, generation and read as diagnostics, never in the
fitness. The `stock_heavy` sanity check belongs to the reads (task 2.4).

Constants marked "registered by 2.3" are frozen in the pre-registration (task 2.3); a real run
will need it. A real run needs a 40-hex source commit, horizon 100 and no filters; anything else
needs `--dry-run`. Contract failures, missing rows or a digest / reproduction mismatch: exit 2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from ..eval.contract import (ContractMonitor, append_failure, ensure_failures_file,
                             failure_from_dict, is_failure, job_failure, read_failures,
                             run_failure)
from ..eval.gate import CSV_COLUMNS, row_from_stats, row_to_dict
from ..eval.gate_cli import CONTRACT_INVALID, Job
from ..eval.provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .producer_residual import BASE_PRODUCER, THETA_DIM, producer_residual_policy

DRIVER = "tokenomics.train.producer_es"
RESIDUAL = "m5_residual"
ORDERING = "mww_keyed"
HORIZON = 100
TRAINING_SIZE = 12
# registered by 2.3 (the pre-registration freezes these values)
PAIRS = 12  # 24 members
MINIBATCH = 24  # DIAGNOSTICS 25
SIGMA = 0.1  # per parameter
ADAM_LR = 0.02  # order_es's Adam (beta1 0.9, beta2 0.999, eps 1e-8), weight decay 0.005
GENERATIONS = {"pilot": 20, "main": 40}  # DIAGNOSTICS 27
MONITOR_EVERY = 10
NOISE_REPLICATES = (1, 2, 3, 4)
SIGNAL, WEAK = 0.3, 0.1
SELF_FLAG_TARGET = 0.05
NOISE_STREAM_PURPOSE = {"train": "m5-train", "probe": "m5-probe"}
BAR, NOISE, PROBE, TRAIN = "bar", "noise-floor", "probe", "train"
MODES = (BAR, NOISE, PROBE, TRAIN)
ES_COLUMNS = ("job_id", "kind", "generation", "member", "sign", "pair", "batch", "theta_sha256",
              "producing", "nonfinite", "episode_seconds", "job_seconds")
ROW_COLUMNS = CSV_COLUMNS + ES_COLUMNS
CENTER = "c"
BASE_LABEL = "base"
ROW_FILES = {BAR: "bar_episodes.csv", NOISE: "noise_episodes.csv", PROBE: "probe_episodes.csv",
             TRAIN: "members.csv"}
CHECKPOINT_FORMAT = "m5-es-checkpoint-v1"
# The frozen pre-registration (task 2.3) every real probe / train run needs; None until 2.3 pins it,
# so a real L2 / L3 run is refused (review m5-2-2).
PREREG_PATH = Path(__file__).resolve().parent / "registry" / "es_prereg_m5_v1.json"
PREREG_SHA256: str | None = None


def prereg_problems() -> list[str]:
    if PREREG_SHA256 is None:
        return ["no pinned pre-registration (task 2.3 freezes es_prereg_m5_v1.json)"]
    if not PREREG_PATH.exists() or sha256_file(str(PREREG_PATH)) != PREREG_SHA256:
        return [f"{PREREG_PATH.name} does not match the pinned SHA-256 {PREREG_SHA256[:12]}"]
    return []


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def theta_sha256(theta: np.ndarray) -> str:
    return _sha([float(x) for x in np.asarray(theta, np.float64)])


# --- the training set (EXP-0058 `bar/`) ---


@dataclass
class TrainingSet:
    dir: Path
    training: dict
    schedule: dict
    entries: dict
    costs: dict
    world_episodes: list[tuple[str, int]]
    pins: dict


def load_training(training_dir: str | Path) -> TrainingSet:
    """M4's pinned training worlds and schedule, each checked against its content hash and the
    recorded streams (order_es.load_inputs' checks, without M4's orderer bar or normaliser)."""
    from . import order_es_worlds as ow
    from ..eval.world_space import point_from_record

    d = Path(training_dir)
    training = json.loads((d / "training_worlds.json").read_text())
    if ow.content_hash(training) != training.get("content_hash"):
        raise ValueError("training_worlds.json does not match its content hash")
    schedule = json.loads((d / "schedule.json").read_text())
    if ow.content_hash(schedule) != schedule.get("content_hash"):
        raise ValueError("schedule.json does not match its content hash")
    if int(training["size"]) != TRAINING_SIZE:
        raise ValueError(f"training size {training['size']} != {TRAINING_SIZE}")
    entries, costs = {}, {}
    for w in training["worlds"]:
        point = replace(point_from_record(w["record"]), shock_window=tuple(w["shock_window"]))
        if point.world_id != w["world"]:
            raise ValueError(f"record of {w['world']} describes {point.world_id}")
        c = ow.Candidate(int(w[ow.ORDER_INDEX]), str(w["source"]), point,
                         tuple(int(e) for e in w["episodes"]), float(w["rho"]))
        entries[c.world] = c.entry()
        cs = w.get("cost_s") or {}
        costs[c.world] = float(max(cs.values(), default=1.0))
    we = [(w["world"], int(ep)) for w in training["worlds"] for ep in w["episodes"]]
    mon = _fmt_we(schedule["monitor"]["world_episodes"])
    if mon != ow.monitor_set(we):
        raise ValueError("schedule.json monitor set differs from the recorded stream")
    return TrainingSet(d, training, schedule, entries, costs, we, {
        "training_worlds_sha256": training["content_hash"],
        "schedule_sha256": schedule["content_hash"]})


def _fmt_we(items: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    """schedule.json's world-episode entries ({world, episode}) as tuples (order_es's format)."""
    from .order_es import _fmt_we as fmt

    return fmt(items)


def monitor_set(ts: TrainingSet) -> list[tuple[str, int]]:
    keep = set(ts.world_episodes)
    return [k for k in _fmt_we(ts.schedule["monitor"]["world_episodes"]) if k in keep]


def matched_to_bar(ts: TrainingSet, bar: Mapping, dry_run: bool) -> TrainingSet:
    """A real run's bar covers every training world-episode; a dry run's (fewer) bar restricts the
    training set to its world-episodes, so the streams draw only what the bar holds."""
    if set(bar) == set(ts.world_episodes):
        return ts
    if not dry_run or not set(bar) <= set(ts.world_episodes):
        raise ValueError(f"the bar holds {len(bar)} of the {len(ts.world_episodes)} training "
                         "world-episodes")
    return replace(ts, world_episodes=sorted(bar))


def minibatch(ts: TrainingSet, opt_seed: int, generation: int, size: int) -> list[tuple[str, int]]:
    """M4's recorded CRN stream; checked against schedule.json where it holds this entry."""
    from . import order_es_worlds as ow

    want = ow.minibatch(ts.world_episodes, opt_seed, generation, size)
    mbs = ts.schedule["minibatches"].get(str(opt_seed), [])
    if size == int(ts.schedule["minibatch_size"]) and generation <= len(mbs):
        got = _fmt_we(mbs[generation - 1])
        if got != want:
            raise ValueError(f"schedule.json minibatch {opt_seed}/{generation} differs from the "
                             "recorded stream")
    return want


def probe_batches(ts: TrainingSet, size: int) -> tuple[list, list]:
    from . import order_es_worlds as ow

    a, b = ow.probe_minibatches(ts.world_episodes, size)
    if len(a) != size or len(b) != size or set(a) & set(b):
        raise ValueError(f"no two disjoint probe minibatches of {size}")
    return a, b


# --- jobs ---


def job_id(kind: str, generation: object, member: str, world: str, episode: int,
           replicate: int) -> str:
    return f"{kind}|{generation}|{member}|{world}|{episode}|{replicate}"


def make_job(ts: TrainingSet, *, kind: str, producing: str, theta: np.ndarray | None,
             generation: object, member: str, world: str, episode: int, replicate: int,
             horizon: int, sign: int = 0, pair: int = -1, batch: int = -1) -> Job:
    e = ts.entries[world]
    t = None if theta is None else [float(x) for x in np.asarray(theta, np.float64)]
    return {"kind": kind, "producing": producing, "theta": t,
            "theta_sha256": "" if t is None else theta_sha256(np.asarray(t)),
            "generation": str(generation), "member": member, "sign": sign, "pair": pair,
            "batch": batch, "world": world, "size": e.size, "regime": e.regime,
            "world_seed": e.world_seed, "generator": e.generator, "episode": int(episode),
            "replicate": int(replicate), "horizon": horizon, "ordering": ORDERING,
            "subject": f"{kind}/{member}", "cost": ts.costs[world],
            "job_id": job_id(kind, generation, member, world, episode, replicate)}


def run_producer_job(job: Job) -> dict[str, str]:
    """One keyed episode of the job's producer with MWWk -> its row, or its failure row."""
    from ..eval.baselines import producing_policy
    from ..eval.joint_stack import joint_stack_env
    from ..eval.routing_audit import job_config, ordering_policy
    from ..eval.routing_diagnostics import DrawKey, set_draw_key
    from ..eval.runner import run_episode

    started = time.perf_counter()
    try:
        env = joint_stack_env(job_config(job), job["horizon"])
        set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
        monitor = ContractMonitor(env)
        if job["producing"] == RESIDUAL:
            prod = producer_residual_policy(np.asarray(job["theta"], np.float64))
            if theta_sha256(prod.theta) != job["theta_sha256"]:
                raise AssertionError("job theta does not match its sha256")
        else:
            prod = producing_policy(job["producing"])
        t0 = time.perf_counter()
        run_episode(env, prod, ordering_policy(job["ordering"]), seed=job["episode"],
                    action_seed=None, monitor=monitor)
        secs = time.perf_counter() - t0
        stats = env.episode_stats
        monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"],
                                         job["episode"], job["subject"], job["replicate"],
                                         stats))
        row.update({"job_id": job["job_id"], "kind": job["kind"], "generation": job["generation"],
                    "member": job["member"], "sign": str(job["sign"]), "pair": str(job["pair"]),
                    "batch": str(job["batch"]), "theta_sha256": job["theta_sha256"],
                    "producing": job["producing"], "nonfinite": "false",
                    "episode_seconds": repr(secs),
                    "job_seconds": repr(time.perf_counter() - started)})
        return {c: row.get(c, "") for c in ROW_COLUMNS}
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


def _init_worker() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


# --- csv / execution ---


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _append(path: Path, row: Mapping[str, str]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROW_COLUMNS), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


class DigestMismatch(RuntimeError):
    pass


def play(out: Path, rows_file: str, jobs: Sequence[Job], workers: int,
         run_job_fn: Callable[[Job], dict[str, str]], bar: Mapping | None,
         log: Callable[[str], None]) -> tuple[dict[str, dict[str, str]], int]:
    """Run every job whose row is not yet in `rows_file` (longest first), appending rows as they
    land. A row of an earlier session must carry this job's theta; with a `bar`, every row must see
    the bar's demand / shock digests (CRN). Returns (rows by job id, new contract failures)."""
    path = out / rows_file
    want = {j["job_id"]: j for j in jobs}
    have = {d["job_id"]: d for d in read_csv(path) if d["job_id"] in want}
    for jid, d in have.items():
        if d["theta_sha256"] != want[jid]["theta_sha256"]:
            raise ValueError(f"{rows_file} row {jid} has another theta than this run's job")
    todo = sorted((j for j in jobs if j["job_id"] not in have),
                  key=lambda j: (-float(j["cost"]), j["job_id"]))
    failures = 0

    def finish(d: dict[str, str]) -> None:
        nonlocal failures
        if is_failure(d):
            failures += 1
            append_failure(out, failure_from_dict(d))
            log(f"[producer_es] CONTRACT FAILURE {d.get('world')} ep={d.get('episode')} "
                f"{d.get('subject')} r{d.get('replicate')}: {d.get('check')} {d.get('error_type')}")
            return
        if bar is not None:
            b = bar[(d["world"], int(d["episode"]))]
            for what, ref in (("demand_digest", b.demand_digest),
                              ("shock_digest", b.shock_digest)):
                if d[what] != ref:
                    from ..eval.contract import DIGEST
                    exc = DigestMismatch(f"{d['job_id']} saw {what} {d[what]}, the bar saw {ref}")
                    append_failure(out, run_failure(DIGEST, exc))
                    raise exc
        _append(path, d)
        have[d["job_id"]] = d

    if workers <= 1 or len(todo) <= 1:
        for j in todo:
            finish(run_job_fn(j))
    else:
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
    return have, failures


# --- bar rows (order_es's bar format) ---

BAR_COLUMNS = ("world", "episode", "bar", "demand_per_cycle", "a0_units", "a0_dead",
               "a0_rundown", "a0_starved_consumers", "demand_digest", "shock_digest")


def bar_row(d: Mapping[str, str], horizon: int) -> dict[str, str]:
    from . import order_es_worlds as ow

    units = float(d["units"])
    rs = float(d["r_start"]) if d["r_start"] not in ("", "None") else 0.0
    re_ = float(d["r_end"]) if d["r_end"] not in ("", "None") else 0.0
    fills = json.loads(d["consumer_fill_rate"])
    starved = sorted(c for c, v in fills.items() if v is not None and v < ow.Y_STARVATION)
    return {"world": d["world"], "episode": d["episode"], "bar": repr(max(units, 1.0)),
            "demand_per_cycle": repr(max(float(d["ordered_units"]), 1.0) / horizon),
            "a0_units": repr(units), "a0_dead": d["dead"],
            "a0_rundown": "true" if re_ < ow.X_RUNDOWN * rs else "false",
            "a0_starved_consumers": json.dumps(starved), "demand_digest": d["demand_digest"],
            "shock_digest": d["shock_digest"]}


def load_bar(bar_dir: str | Path) -> tuple[dict, list[dict[str, str]], dict]:
    """(bar rows by (world, episode) as order_es.BarRow, the base's r0 rows, pins)."""
    from .order_es import read_bar

    bd = Path(bar_dir)
    spec = json.loads((bd / "spec.json").read_text())
    if spec.get("mode") != BAR or spec.get("status") != "complete":
        raise ValueError(f"{bd} is not a complete producer_es bar run")
    return (read_bar(bd / "bar.csv"), read_csv(bd / "base_rows.csv"),
            {"bar_csv_sha256": sha256_file(str(bd / "bar.csv")),
             "bar_spec_pins_sha256": _sha(spec.get("pins"))})


# --- spec / pins ---


def _open_run(out: Path, mode: str, pins: dict, workers: int, log: Callable[[str], None]
              ) -> dict:
    """A new spec, or (same pins) the recorded one with a resume noted; other pins are refused
    before any write."""
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "spec.json"
    pins = json.loads(json.dumps(pins, sort_keys=True, default=str))
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        if old.get("driver") != DRIVER or old.get("mode") != mode:
            raise ValueError(f"{out}/spec.json belongs to another run "
                             f"({old.get('driver')} {old.get('mode')})")
        if old.get("pins") != pins:
            diff = sorted(k for k in set(old.get("pins") or {}) | set(pins)
                          if (old.get("pins") or {}).get(k) != pins.get(k))
            raise ValueError(f"{out} holds a run with different pins; refusing to resume "
                             f"(differs at {diff})")
        old.setdefault("resumes", []).append({"workers": workers, "at": time.time()})
        spec = old
        log(f"[producer_es] resuming {out} ({mode})")
    else:
        spec = {"driver": DRIVER, "mode": mode, "pins": pins, "status": "running",
                "acceptance_read": False, "started": time.time(), "workers": workers}
    ensure_failures_file(out)
    _write_spec(out, spec)
    return spec


def _write_spec(out: Path, spec: dict) -> None:
    tmp = out / "spec.json.tmp"
    tmp.write_text(json.dumps(spec, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, out / "spec.json")


def real_run_problems(*, horizon: int, source_commit: str, filters: Mapping[str, object]
                      ) -> list[str]:
    import re

    problems = []
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
        problems.append("no full 40-hex source commit (--source-commit or --verify-source-commit)")
    if horizon != HORIZON:
        problems.append(f"horizon {horizon} != {HORIZON}")
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


def _finish(out: Path, spec: dict, expected: int, rows: int, extra_bad: list[str],
            result: dict, log: Callable[[str], None]) -> int:
    failures = read_failures(out)
    missing = expected - rows
    bad = bool(failures or missing > 0 or extra_bad)
    spec["status"] = "invalid" if bad else "complete"
    spec["result"] = {"status": spec["status"], "rows": rows, "expected_rows": expected,
                      "contract_failures": len(failures), "problems": extra_bad, **result}
    _write_spec(out, spec)
    log(f"[producer_es] {spec['mode']} {spec['status']}"
        + (f" ({len(failures)} failure(s), {missing} missing, {extra_bad})" if bad else ""))
    return CONTRACT_INVALID if bad else 0


def _diag(rows: Sequence[Mapping[str, str]]) -> dict[str, float | None]:
    """Mean holding units per cycle and investment rejection rate (diagnostics, never fitness)."""
    def m(col: str) -> float | None:
        xs = [float(r[col]) for r in rows if r.get(col) not in (None, "", "None")]
        return fmean(xs) if xs else None
    return {"holding_units_per_cycle": m("holding_units_per_cycle"),
            "investment_rejection_rate": m("investment_rejection_rate"),
            "units": m("units")}


# --- modes ---


def _select(ts: TrainingSet, max_world_episodes: int | None) -> list[tuple[str, int]]:
    we = sorted(ts.world_episodes)
    return we if max_world_episodes is None else we[:max_world_episodes]


def run_bar(*, training_dir: str | Path, out_dir: str | Path, workers: int = 1,
            horizon: int = HORIZON, max_world_episodes: int | None = None,
            atlas: str | Path | None = None, exp0079_episodes: str | Path | None = None,
            source_commit: str = "", dry_run: bool = False,
            run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
            log: Callable[[str], None] = _log) -> int:
    ts = load_training(training_dir)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit,
                              filters={"max_world_episodes": max_world_episodes})
        if (atlas is None) != (exp0079_episodes is None) or atlas is None:
            p.append("a real bar run needs --atlas and --exp0079-episodes (the cross-check)")
        if p:
            raise ValueError("not a valid bar run (" + "; ".join(p) + "); pass --dry-run")
    we = _select(ts, max_world_episodes)
    pins = {"driver": DRIVER, "mode": BAR, "base": BASE_PRODUCER, "ordering": ORDERING,
            "horizon": horizon, "dry_run": dry_run, "world_episodes": we, **ts.pins,
            "exp0079_episodes_sha256": (sha256_file(str(exp0079_episodes))
                                        if exp0079_episodes else None),
            "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, BAR, pins, workers, log)
    jobs = [make_job(ts, kind=BAR, producing=BASE_PRODUCER, theta=None, generation=0,
                     member=BASE_LABEL, world=w, episode=e, replicate=0, horizon=horizon)
            for w, e in we]
    rows, _ = play(out, ROW_FILES[BAR], jobs, workers, run_job_fn, None, log)
    ordered = [rows[j["job_id"]] for j in jobs if j["job_id"] in rows]
    write_csv(out / "base_rows.csv", ROW_COLUMNS, ordered)
    write_csv(out / "bar.csv", BAR_COLUMNS, [bar_row(d, horizon) for d in ordered])
    bad: list[str] = []
    cross: dict = {}
    if atlas is not None and exp0079_episodes is not None:
        cross = crosscheck_exp0079(atlas, exp0079_episodes, out, log)
        if not cross["equal"]:
            bad.append(f"rf3 replay differs from EXP-0079 on {cross['differing']}")
    return _finish(out, spec, len(jobs), len(ordered), bad,
                   {"diagnostics": _diag(ordered), "crosscheck": cross}, log)


CROSSCHECK_N = 2


def crosscheck_exp0079(atlas: str | Path, exp0079_episodes: str | Path, out: Path,
                       log: Callable[[str], None]) -> dict:
    """rf3 replayed through the supply-trace job path on the first CROSSCHECK_N MWWk r0
    world-episodes of EXP-0079 (sorted): every cell EXP-0079 wrote must be equal (D7)."""
    from ..eval import supply_trace as st
    from ..eval.decoder_check import MWWK, decoder_arm, select_views, world_entry
    from ..eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from ..eval.routing_audit import build_jobs
    from ..eval.teacher_audit import load_atlas

    csv.field_size_limit(1 << 30)
    ref = {(r["world"], r["episode"], r["replicate"]): r for r in read_csv(Path(exp0079_episodes))
           if r["subject"].endswith("/mww_keyed") and r["replicate"] == "0"}
    keys = sorted(ref)[:CROSSCHECK_N]
    views = select_views(load_atlas(atlas), sorted({k[0] for k in keys}))
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    arm = {**arm, "producing": BASE_PRODUCER, "subject": member_id(BASE_PRODUCER, arm["ordering"])}
    jobs = [j for j in build_jobs([world_entry(v) for v in views], [arm], 1, HORIZON, None, {})
            if (j["world"], str(j["episode"]), str(j["replicate"])) in set(keys)]
    for j in jobs:
        j["arm"] = MWWK
        j["order_ttl"] = 2
    rows, differing = [], []
    for j in jobs:
        d = st.run_supply_job(j)
        r = ref[(j["world"], str(j["episode"]), str(j["replicate"]))]
        diff = [c for c in r if d.get(c) != r[c]]
        rows.append({"world": j["world"], "episode": j["episode"], "compared": len(r),
                     "differing": ";".join(diff[:8]), "equal": not diff})
        if diff:
            differing.append(f"{j['world']}:{j['episode']}")
    write_csv(out / "crosscheck.csv", ("world", "episode", "compared", "differing", "equal"), rows)
    log(f"[producer_es] EXP-0079 cross-check: {sum(r['equal'] for r in rows)} / {len(rows)} equal")
    return {"rows": len(rows), "equal": len(rows) == CROSSCHECK_N and not differing,
            "differing": differing}


def run_noise_floor(*, training_dir: str | Path, bar_dir: str | Path, out_dir: str | Path,
                    workers: int = 1, horizon: int = HORIZON,
                    replicates: Sequence[int] = NOISE_REPLICATES, source_commit: str = "",
                    dry_run: bool = False,
                    run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
                    log: Callable[[str], None] = _log) -> int:
    from .order_es import NOISE_MARGIN_C, NoiseMargin, noise_margins, self_flags

    ts = load_training(training_dir)
    bar, base_rows, bar_pins = load_bar(bar_dir)
    ts = matched_to_bar(ts, bar, dry_run)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit,
                              filters={"replicates": None if tuple(replicates) == NOISE_REPLICATES
                                       else replicates})
        if p:
            raise ValueError("not a valid noise-floor run (" + "; ".join(p) + "); pass --dry-run")
    we = sorted(bar)
    pins = {"driver": DRIVER, "mode": NOISE, "base": BASE_PRODUCER, "horizon": horizon,
            "dry_run": dry_run, "replicates": list(replicates), **ts.pins, **bar_pins,
            "c": NOISE_MARGIN_C, "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, NOISE, pins, workers, log)
    jobs = [make_job(ts, kind=NOISE, producing=BASE_PRODUCER, theta=None, generation=0,
                     member=f"r{r}", world=w, episode=e, replicate=r, horizon=horizon)
            for w, e in we for r in (0, *replicates)]
    rows, _ = play(out, ROW_FILES[NOISE], jobs, workers, run_job_fn, bar, log)
    got = [rows[j["job_id"]] for j in jobs if j["job_id"] in rows]
    base = {(d["world"], d["episode"]): d for d in base_rows}
    repro, bad = [], []
    for d in got:
        if d["replicate"] != "0":
            continue
        b = base[(d["world"], d["episode"])]
        diff = [c for c in CSV_COLUMNS if c != "subject" and d.get(c) != b.get(c)]
        repro.append({"world": d["world"], "episode": d["episode"], "differing": ";".join(diff),
                      "equal": not diff})
    write_csv(out / "reproduction.csv", ("world", "episode", "differing", "equal"), repro)
    if not all(r["equal"] for r in repro) or len(repro) != len(we):
        bad.append(f"r0 reproduces {sum(r['equal'] for r in repro)} / {len(we)} bar rows")
    margins = {}
    if got and not bad:
        m = noise_margins(got)
        margin = NoiseMargin(m["rundown"], m["starvation"])
        flagged, n = self_flags(got, bar, margin)
        raw, _ = self_flags(got, bar, None)
        margins = {**m, "self_flags": flagged, "self_flags_without_margin": raw, "episodes": n,
                   "self_flag_rate": flagged / n if n else None,
                   "target_rate": SELF_FLAG_TARGET,
                   "target_met": bool(n) and flagged / n <= SELF_FLAG_TARGET}
        doc = {"format": "m5-noise-margin-v1", "base": BASE_PRODUCER, **margins,
               "rows_sha256": _sha(sorted((d["job_id"], d["units"], d["r_end"]) for d in got))}
        (out / "margins.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return _finish(out, spec, len(jobs), len(got), bad,
                   {"margins": margins, "diagnostics": _diag(got)}, log)


def load_margin(margins_path: str | Path) -> tuple[Any, dict]:
    from .order_es import NoiseMargin

    doc = json.loads(Path(margins_path).read_text())
    if doc.get("format") != "m5-noise-margin-v1":
        raise ValueError(f"{margins_path} is not an M5 noise-margin file")
    return NoiseMargin(float(doc["rundown"]), float(doc["starvation"])), {
        "margins_sha256": sha256_file(str(margins_path))}


def _scores(rows_by_member: Mapping[str, Sequence[Mapping[str, str]]], bar: Mapping,
            margin: Any, refs: Mapping) -> dict:
    """order_es.score_member with h = 0: units-only fitness, the relative rule against the base."""
    from .order_es import score_member

    return {lab: score_member(rs, bar, 0.0, margin, refs) for lab, rs in rows_by_member.items()}


def _member_jobs(ts: TrainingSet, kind: str, thetas: Mapping[str, np.ndarray], world_eps,
                 generation: object, horizon: int, batch: int = -1) -> list[Job]:
    return [make_job(ts, kind=kind, producing=RESIDUAL, theta=th, generation=generation,
                     member=lab, world=w, episode=e, replicate=0, horizon=horizon,
                     sign=0 if lab == CENTER else (1 if lab[0] == "+" else -1),
                     pair=-1 if lab == CENTER else int(lab[1:]), batch=batch)
            for lab, th in thetas.items() for w, e in world_eps]


def _by_member(rows: Mapping[str, dict], jobs: Sequence[Job]) -> dict[str, list]:
    out: dict[str, list] = {}
    for j in jobs:
        if j["job_id"] in rows:
            out.setdefault(j["member"], []).append(rows[j["job_id"]])
    return out


def run_probe(*, training_dir: str | Path, bar_dir: str | Path, margins: str | Path,
              out_dir: str | Path, workers: int = 1, horizon: int = HORIZON,
              size: int = MINIBATCH, pairs: int = PAIRS, sigma: float = SIGMA,
              source_commit: str = "", dry_run: bool = False,
              run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
              log: Callable[[str], None] = _log) -> int:
    from .order_es import a0_values, cosine, member_thetas, noise, rank_and_estimate

    ts = load_training(training_dir)
    bar, base_rows, bar_pins = load_bar(bar_dir)
    margin, m_pins = load_margin(margins)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit, filters={
            "size": None if size == MINIBATCH else size,
            "pairs": None if pairs == PAIRS else pairs,
            "sigma": None if sigma == SIGMA else sigma}) + prereg_problems()
        if p:
            raise ValueError("not a valid probe run (" + "; ".join(p) + "); pass --dry-run")
    ts = matched_to_bar(ts, bar, dry_run)  # after the real-run refusals
    a, b = probe_batches(ts, size)
    sig = np.full(THETA_DIM, sigma)
    eps = noise(0, 0, pairs, THETA_DIM, NOISE_STREAM_PURPOSE["probe"])
    thetas = {k: v for k, v in member_thetas(np.zeros(THETA_DIM), eps, sig).items() if k != CENTER}
    pins = {"driver": DRIVER, "mode": PROBE, "horizon": horizon, "dry_run": dry_run,
            "size": size, "pairs": pairs, "sigma": sigma, "theta_dim": THETA_DIM, **ts.pins,
            **bar_pins, **m_pins, "batches": [a, b], "eps_sha256": _sha(eps.tolist()),
            "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, PROBE, pins, workers, log)
    jobs = (_member_jobs(ts, PROBE, thetas, a, 0, horizon, 0)
            + _member_jobs(ts, PROBE, thetas, b, 0, horizon, 1))
    rows, _ = play(out, ROW_FILES[PROBE], jobs, workers, run_job_fn, bar, log)
    refs = {k: v[0] for k, v in a0_values(base_rows).items()}
    est, per_batch = [], []
    for bi in (0, 1):
        bj = [j for j in jobs if j["batch"] == bi]
        sc = _scores(_by_member(rows, bj), bar, margin, refs)
        if len(sc) < 2 * pairs:
            est.append(None)
            continue
        ranked = rank_and_estimate(eps, sig, {k: s.fitness for k, s in sc.items()},
                                   {k: s.disqualified for k, s in sc.items()})
        est.append(ranked.estimate)
        per_batch.append({"batch": bi, "disqualified": sum(s.disqualified for s in sc.values()),
                          "fitness_mean": fmean(s.fitness for s in sc.values()
                                                if s.fitness is not None),
                          "diagnostics": _diag([r for j in bj if (r := rows.get(j["job_id"]))])})
    cos = cosine(est[0], est[1]) if all(e is not None for e in est) else None
    reading = (None if cos is None else "signal" if cos >= SIGNAL else "weak" if cos >= WEAK
               else "no_signal")
    doc = {"cosine": cos, "reading": reading, "thresholds": {"signal": SIGNAL, "weak": WEAK},
           "batches": per_batch,
           "estimates": [None if e is None else [float(x) for x in e] for e in est]}
    (out / "probe.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return _finish(out, spec, len(jobs), sum(j["job_id"] in rows for j in jobs), [],
                   {"reading": reading, "cosine": cos}, log)


def checkpoint_path(out: Path, generation: int) -> Path:
    return out / "checkpoints" / f"gen-{generation:03d}.json"


def save_checkpoint(path: Path, doc: dict) -> None:
    doc = dict(doc)
    doc["content_hash"] = _sha({k: v for k, v in doc.items() if k != "content_hash"})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if _sha({k: v for k, v in doc.items() if k != "content_hash"}) != doc.get("content_hash"):
        raise ValueError(f"{path} does not match its content hash")
    if theta_sha256(np.asarray(doc["theta"])) != doc["theta_sha256"]:
        raise ValueError(f"{path} theta does not match its sha256")
    return doc


GEN_COLUMNS = ("generation", "theta_sha256", "minibatch_sha256", "fitness_mean", "fitness_best",
               "disqualified", "grad_norm", "theta_norm", "holding_units_per_cycle",
               "investment_rejection_rate", "units")
MON_COLUMNS = ("generation", "theta_sha256", "world_episodes", "g_tr", "center_units",
               "bar_units", "center_holding", "bar_holding", "center_investment_rejection",
               "bar_investment_rejection")


def run_train(*, training_dir: str | Path, bar_dir: str | Path, margins: str | Path,
              out_dir: str | Path, generations: int, opt_seed: int = 0, workers: int = 1,
              horizon: int = HORIZON, size: int = MINIBATCH, pairs: int = PAIRS,
              sigma: float = SIGMA, lr: float = ADAM_LR, monitor_every: int = MONITOR_EVERY,
              source_commit: str = "", dry_run: bool = False,
              run_job_fn: Callable[[Job], dict[str, str]] = run_producer_job,
              log: Callable[[str], None] = _log) -> int:
    """`generations` = the target generation count; a resume continues from the latest
    checkpoint of the same pins (the target may grow: pilot 20 -> main 40)."""
    from .order_es import AdamState, a0_values, adam_ascend, member_thetas, noise, \
        rank_and_estimate

    ts = load_training(training_dir)
    bar, base_rows, bar_pins = load_bar(bar_dir)
    margin, m_pins = load_margin(margins)
    out = Path(out_dir)
    if not dry_run:
        p = real_run_problems(horizon=horizon, source_commit=source_commit, filters={
            "size": None if size == MINIBATCH else size,
            "pairs": None if pairs == PAIRS else pairs,
            "sigma": None if sigma == SIGMA else sigma, "lr": None if lr == ADAM_LR else lr,
            "monitor_every": None if monitor_every == MONITOR_EVERY else monitor_every}
        ) + prereg_problems()
        if p:
            raise ValueError("not a valid train run (" + "; ".join(p) + "); pass --dry-run")
    ts = matched_to_bar(ts, bar, dry_run)  # after the real-run refusals
    pins = {"driver": DRIVER, "mode": TRAIN, "horizon": horizon, "dry_run": dry_run,
            "opt_seed": opt_seed, "size": size, "pairs": pairs, "sigma": sigma, "lr": lr,
            "monitor_every": monitor_every, "theta_dim": THETA_DIM, **ts.pins, **bar_pins,
            **m_pins, "source": source_fingerprint(source_commit or None)}
    spec = _open_run(out, TRAIN, pins, workers, log)
    refs = {k: v[0] for k, v in a0_values(base_rows).items()}
    sig = np.full(THETA_DIM, sigma)
    done = sorted((out / "checkpoints").glob("gen-*.json")) if (out / "checkpoints").exists() else []
    if done:
        ck = load_checkpoint(done[-1])
        if ck["pins_sha256"] != _sha(spec["pins"]):
            raise ValueError(f"{done[-1]} was written under other pins")
        theta, adam, start = np.asarray(ck["theta"]), AdamState.from_dict(ck["adam"]), ck["generation"]
    else:
        theta, adam, start = np.zeros(THETA_DIM), AdamState.zeros(THETA_DIM), 0
    # rows beyond the checkpoint (a crash between generations.csv and the checkpoint) are dropped,
    # one row per generation (review m5-2-2)
    gens = list({int(d["generation"]): d for d in read_csv(out / "generations.csv")
                 if int(d["generation"]) <= start}.values())
    gens.sort(key=lambda d: int(d["generation"]))
    mon = list({int(d["generation"]): d for d in read_csv(out / "monitor.csv")}.values())
    mon.sort(key=lambda d: int(d["generation"]))
    if len(gens) != start:
        raise ValueError(f"generations.csv holds {len(gens)} rows for checkpoint generation {start}")
    write_csv(out / "generations.csv", GEN_COLUMNS, gens)

    def monitor(g: int, th: np.ndarray) -> None:
        mws = monitor_set(ts)
        mjobs = _member_jobs(ts, "monitor", {CENTER: th}, mws, g, horizon)
        mrows, _ = play(out, "monitor_episodes.csv", mjobs, workers, run_job_fn, bar, log)
        got = [mrows[j["job_id"]] for j in mjobs if j["job_id"] in mrows]
        if len(got) != len(mjobs):
            raise RuntimeError(f"monitor at generation {g}: {len(got)} of {len(mjobs)} rows")
        bars = [bar[(d["world"], int(d["episode"]))] for d in got]
        base_by = {(d["world"], d["episode"]): d for d in base_rows}
        bd = [base_by[(d["world"], d["episode"])] for d in got]
        dc, db = _diag(got), _diag(bd)
        mon[:] = [m for m in mon if int(m["generation"]) != g]
        mon.append({"generation": g, "theta_sha256": theta_sha256(th),
                    "world_episodes": len(got),
                    "g_tr": fmean(float(d["units"]) - b.a0_units for d, b in zip(got, bars)),
                    "center_units": dc["units"], "bar_units": db["units"],
                    "center_holding": dc["holding_units_per_cycle"],
                    "bar_holding": db["holding_units_per_cycle"],
                    "center_investment_rejection": dc["investment_rejection_rate"],
                    "bar_investment_rejection": db["investment_rejection_rate"]})
        mon.sort(key=lambda d: int(d["generation"]))
        write_csv(out / "monitor.csv", MON_COLUMNS, mon)

    # a monitor that died after its generation's checkpoint is run on resume (review m5-2-2)
    if start and start % monitor_every == 0 and not any(int(m["generation"]) == start
                                                         for m in mon):
        log(f"[producer_es] resume: running the missing monitor of generation {start}")
        monitor(start, theta)
    expected = rows_seen = 0
    for g in range(start + 1, generations + 1):
        mb = minibatch(ts, opt_seed, g, size)
        eps = noise(opt_seed, g, pairs, THETA_DIM, NOISE_STREAM_PURPOSE["train"])
        thetas = {k: v for k, v in member_thetas(theta, eps, sig).items() if k != CENTER}
        jobs = _member_jobs(ts, TRAIN, thetas, mb, g, horizon)
        rows, _ = play(out, ROW_FILES[TRAIN], jobs, workers, run_job_fn, bar, log)
        expected += len(jobs)
        rows_seen += sum(j["job_id"] in rows for j in jobs)
        sc = _scores(_by_member(rows, jobs), bar, margin, refs)
        if len(sc) < 2 * pairs or any(len(v) != len(mb) for v in _by_member(rows, jobs).values()):
            log(f"[producer_es] generation {g}: missing member rows; stopping")
            break
        ranked = rank_and_estimate(eps, sig, {k: s.fitness for k, s in sc.items()},
                                   {k: s.disqualified for k, s in sc.items()})
        theta, adam = adam_ascend(theta, ranked.estimate, adam, lr=lr)
        member_rows = [r for j in jobs if (r := rows.get(j["job_id"]))]
        fits = [s.fitness for s in sc.values() if s.fitness is not None]
        gens.append({"generation": g, "theta_sha256": theta_sha256(theta),
                     "minibatch_sha256": _sha(mb), "fitness_mean": fmean(fits),
                     "fitness_best": max(fits),
                     "disqualified": sum(s.disqualified for s in sc.values()),
                     "grad_norm": float(np.linalg.norm(ranked.estimate)),
                     "theta_norm": float(np.linalg.norm(theta)), **_diag(member_rows)})
        write_csv(out / "generations.csv", GEN_COLUMNS, gens)
        save_checkpoint(checkpoint_path(out, g), {
            "format": CHECKPOINT_FORMAT, "generation": g, "theta": [float(x) for x in theta],
            "theta_sha256": theta_sha256(theta), "adam": adam.to_dict(),
            "pins_sha256": _sha(spec["pins"])})
        if g % monitor_every == 0:
            monitor(g, theta)
        log(f"[producer_es] generation {g}: fitness {fmean(fits):.4f}, "
            f"disqualified {gens[-1]['disqualified']}, |grad| {gens[-1]['grad_norm']:.4f}")
    last = int(gens[-1]["generation"]) if gens else start
    due = [g for g in range(monitor_every, last + 1, monitor_every)
           if not any(int(m["generation"]) == g for m in mon)]
    return _finish(out, spec, expected, rows_seen,
                   ([] if last >= generations else [f"stopped at generation {last}"])
                   + ([f"monitor missing at generations {due}"] if due else []),
                   {"generation": last, "theta_sha256": theta_sha256(theta)}, log)


# --- cli ---


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--training-dir", required=True, help="EXP-0058 bar/ (training set, schedule)")
    p.add_argument("--bar-dir", help="noise-floor / probe / train: this driver's bar run")
    p.add_argument("--margins", help="probe / train: the noise-floor run's margins.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-world-episodes", type=int, help="bar dry run: the first N")
    p.add_argument("--atlas", help="bar: atlas_v2.json (the EXP-0079 cross-check)")
    p.add_argument("--exp0079-episodes", help="bar: EXP-0079's rf3 episodes.csv")
    p.add_argument("--replicates", help="noise-floor dry run: comma-separated (default 1,2,3,4)")
    p.add_argument("--generations", type=int, help="train: target generation count")
    p.add_argument("--opt-seed", type=int, default=0)
    p.add_argument("--size", type=int, default=MINIBATCH, help="dry run: minibatch size")
    p.add_argument("--pairs", type=int, default=PAIRS, help="dry run: antithetic pairs")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--lr", type=float, default=ADAM_LR)
    p.add_argument("--monitor-every", type=int, default=MONITOR_EVERY)
    a = p.parse_args(argv)
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    common = dict(training_dir=a.training_dir, out_dir=a.out_dir, workers=a.workers,
                  horizon=a.horizon, source_commit=commit, dry_run=a.dry_run)
    if a.mode == BAR:
        return run_bar(**common, max_world_episodes=a.max_world_episodes, atlas=a.atlas,
                       exp0079_episodes=a.exp0079_episodes)
    if a.bar_dir is None:
        p.error(f"--mode {a.mode} needs --bar-dir")
    if a.mode == NOISE:
        reps = NOISE_REPLICATES if a.replicates is None else tuple(
            int(x) for x in a.replicates.split(","))
        return run_noise_floor(**common, bar_dir=a.bar_dir, replicates=reps)
    if a.margins is None:
        p.error(f"--mode {a.mode} needs --margins")
    if a.mode == PROBE:
        return run_probe(**common, bar_dir=a.bar_dir, margins=a.margins, size=a.size,
                         pairs=a.pairs, sigma=a.sigma)
    if a.generations is None:
        p.error("--mode train needs --generations")
    return run_train(**common, bar_dir=a.bar_dir, margins=a.margins, generations=a.generations,
                     opt_seed=a.opt_seed, size=a.size, pairs=a.pairs, sigma=a.sigma, lr=a.lr,
                     monitor_every=a.monitor_every)


if __name__ == "__main__":
    sys.exit(main())
