"""PoC-A teacher audit driver: `python -m tokenomics.eval.teacher_audit` (add-m3-development-audit
design D5, ADR-0056; development-audit "Teacher audit reading", "Clairvoyant bound is a reference
only", "Expensive teacher only on a mined sample", "Held-out sealed").

Question (EXP-0053): on atlas worlds where the bound leaves room, does a lookahead search reach
more than the best cheap arm?

Input is an atlas manifest (`atlas_v2.json`; EXP-0052's `reclassified/atlas_v2.json`, task 5.3).
The pre-registered sample (`pre_registered_sample`, D5): atlas v2's train split, 12 producers,
demand pattern cyclic / iid / drift / burst / seasonal (the moving hit and the regime switch
carry hidden pattern state ADR-0056 defines no posterior for), 6 hard worlds by confirm gain and
6 open worlds by room (the other class fills a shortfall), per class the best n // 2 shocked and
n - n // 2 unshocked (a shortfall on either side filled by the class's best remaining worlds);
1 episode per world, the confirm stage's first episode seed. It equals the "new" sample of
`probes/exp0052_check.py --reclassified`.

Arms, all with the `entity_planner` producer and the joint-stack contract:
- `floor`: `round_robin` (the teacher's floor and unresolved fallback);
- `d2`: the EXP-0042 bounded rollout search at depth 2 over `round_robin`, candidates = the
  whole order to each eligible producer plus the trained orderer's row (`--orderer-ckpt`),
  scenarios = ADR-0056 posterior forks, or one scenario with the zero-variance rule on a
  deterministic-future world (cyclic demand, no shocks); unresolved cycles act as the floor;
- `ub`: the clairvoyant bound, the same search over the true future (one scenario). Only on worlds
  with shocks or stochastic demand (elsewhere it equals `d2` by construction). It reads hidden
  information: an upper reference, listed apart from the teacher arms, never a label source;
- the world's blind and reach arms of D12 (the atlas confirm's arms without `portable`), so the
  bar, the reach arm and the digests are checked in the same run (`routing_audit.execute`: a
  demand or shock digest that differs between two arms of a world-episode fails the run, naming
  world, episode and arm).

Rows are the atlas's (gate columns, `trace_*`, bounds) plus the teacher columns; teacher rows
run without the routing tracer (its hook is a closure a fork's deepcopy would share with the live
system), so their `trace_*` cells are empty. Writes
`spec.json`, `episodes.csv`, `ratios.csv` (per world: `d2` and `ub` against the bar, the reach
arm, the ceiling C and B2; readings), `teacher_decisions.csv` (one row per teacher decision:
posterior draws / accepted / acceptance rate / unresolved, searched streams, candidates,
scenarios, rollouts and seconds per searched stream), `contract_failures.csv` and a `RESULT.md`
skeleton. Never an acceptance read: no verdict, the gate margins are reference ratios only.

Refuses any world above 42 producers, any world outside the manifest's train split and any
moving-hit or regime-switch world. A real run needs the registered atlas (content hash), the full
sample or a pre-registered reduction of it (`--reduction step2`: D5 step 2, 8 world-episodes with
`ub` kept, the EXP-0053 sample of sizing.md §EXP-0053), every arm, horizon 100, the registered shock window, the frozen roster, B_lp (scipy) and
a 40-hex source commit; anything else needs `--dry-run` (marked "not an acceptance read").
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence

from .atlas import (
    ATLAS_COLUMNS,
    ATLAS_FORMAT,
    BLIND,
    BOUND_CHECK,
    CONFIRM,
    CONFIRM_STAGE,
    REACH,
    REPORTED,
    TRAIN,
    arm_table,
    bound_cells,
    bound_violations,
    confirm_episode_count,
    content_hash,
    episode_values,
    flag_lp_jobs,
    measures,
    run_atlas_job,
    space_entry,
    space_info,
)
from .ceiling_bound import lp_available
from .contract import (
    RUN,
    ContractMonitor,
    ContractRunError,
    Failure,
    append_failure,
    ensure_failures_file,
    is_failure,
    job_failure,
    read_failures,
    run_failure,
)
from .gate_cli import (
    CONTRACT_INVALID,
    INVALID,
    Job,
    Key,
    _open_spec,
    _prepare_resume,
    _write_spec,
    check_rows,
    job_grid,
    job_key,
    read_rows,
)
from .provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .roster import DEFAULT_ROSTER, JOINT_STACK, Roster, load_roster
from .routing_audit import (
    PRODUCER,
    _is_commit,
    build_jobs,
    execute,
    expected_keys,
    job_config,
    load_trained_policy,
    ordering_policy,
    read_episode_dicts,
    trained_policy,
    write_csv,
)
from .routing_diagnostics import DETERMINISTIC
from .shock_profiles import SHOCKS_I_SMOKE, SMOKE_START_WINDOW
from .world_space import (
    DRIFT,
    IID,
    CYCLIC,
    config_from_record,
    episode_seeds,
    point_from_record,
)

DRIVER = "tokenomics.eval.teacher_audit"
EXPERIMENT = "EXP-0053"
HORIZON = 100
DEPTH = 2  # D5: EXP-0042 promoted depth 2; depth 3 costs about twice as much
MAX_PRODUCERS = 42  # spec "Expensive teacher only on a mined sample"
SAMPLE_PRODUCERS = 12
SAMPLE_SIZE = 12
SAMPLE_HARD = 6
# Pre-registered reductions (D5 "Sizing"; sizing.md §EXP-0053): name -> (size, hard target, note).
# A reduction is the same rule at a smaller size, so it is allowed in a real run.
STEP2 = "step2"
REDUCTIONS: dict[str, tuple[int, int, str]] = {
    STEP2: (8, 4, "D5 reduction step 2 (8 world-episodes, 4 hard + 4 open by the same rule) with "
                  "`ub` kept: a deviation from the written order (step 1, drop `ub`, first), "
                  "decided by the supervisor on cost alone before any result was seen "
                  "(sizing.md §EXP-0053): with step 2 the run fits the 10 vCPU-h stop with `ub`, "
                  "the only measure of what the hidden information is worth"),
}
HARD, OPEN = "hard", "open"
POC_PATTERNS = (CYCLIC, IID, DRIFT, "burst", "seasonal")  # D4: hit / switch carry hidden state
FLOOR_ROUTER = "round_robin"
FLOOR, D2, UB = "floor", "d2", "ub"
TEACHER_ARMS = (FLOOR, D2, UB)
TEACHER = "teacher"  # arm side of floor / d2
BOUND = "bound"  # arm side of ub: hidden information, never a teacher
SUBJECT = {FLOOR: "teacher/floor", D2: "teacher/d2", UB: "bound/ub"}
GATE_MARGINS = {"calm": 1.05, "shocked": 1.10}  # reference ratios only (D5)
FLOOR_CHECK = "floor_reproduction"
# EXP-0052's reclassified atlas (task 5.3); a real run reads exactly this manifest
REGISTERED_ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"

TEACHER_COLUMNS = ("arm", "side", "hidden_information", "scenario_source", "decisions",
                   "unresolved_decisions", "unresolved_fraction", "posterior_cycles",
                   "unresolved_cycles", "mean_acceptance_rate", "searched_streams",
                   "override_streams", "rollouts", "search_seconds")
COLUMNS = ATLAS_COLUMNS + TEACHER_COLUMNS
DECISION_COLUMNS = ("world", "episode", "arm", "hidden_information", "scenario_source", "cycle",
                    "agent", "posterior_draws", "posterior_accepted", "acceptance_rate",
                    "unresolved", "slots", "contested", "searched_streams", "candidates",
                    "scenarios", "rollouts", "overrides", "confident_base",
                    "deterministic_zero", "unresolved_streams", "capped", "seconds",
                    "base_seconds", "stream_seconds", "seconds_per_searched_stream")
RATIO_COLUMNS = (
    "world", "class", "size", "choice", "pattern", "shocks", "episode", "noise_floor",
    "bar_arm", "bar", "reach_arm", "reach", "ceiling", "b2", "floor_units", "d2_units",
    "d2_minus_bar", "d2_over_bar", "d2_minus_reach", "d2_over_ceiling", "d2_room_closed",
    "d2_beats_bar", "d2_ge_reach", "d2_below_bar", "gate_margin_ref", "d2_clears_margin_ref",
    "d2_unresolved_fraction", "d2_mean_acceptance_rate", "floor_dead", "d2_dead",
    "bar_dead", "reading",
    "clairvoyant_ub_units", "clairvoyant_ub_minus_bar", "clairvoyant_ub_over_bar",
    "clairvoyant_ub_beats_bar", "clairvoyant_ub_dead", "clairvoyant_ub_hidden_information")


# --- atlas manifest and the pre-registered sample ---


def load_atlas(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != ATLAS_FORMAT:
        raise ValueError(f"{path}: format {doc.get('format')!r}, the teacher audit reads "
                         f"{ATLAS_FORMAT!r} (task 5.3's reclassified atlas)")
    if content_hash(doc) != doc.get("content_hash"):
        raise ValueError(f"{path}: content hash does not match its body")
    return doc


def world_view(entry: Mapping) -> dict:
    """What the sample and the readings need from one atlas world entry."""
    rec = entry["record"]
    lv = rec["levers"]
    return {"world": rec["world_id"], "split": entry["split"], "size": int(rec["size"]),
            "pattern": lv["pattern"], "shocks": bool(lv["shocks"]), "choice": lv["choice"],
            "class": entry["confirm_class"], "gain": entry.get("gain"),
            "room": entry.get("room"), "noise_floor": entry.get("noise_floor"),
            "bar_arm": entry.get("bar_arm"), "reach_arm": entry.get("reach_arm"),
            "record": rec}


def refusal(view: Mapping) -> str | None:
    """Why the teacher must not run on this world (None = it may)."""
    if view["size"] > MAX_PRODUCERS:
        return f"{view['world']} has {view['size']} producers (> {MAX_PRODUCERS})"
    if view["split"] != TRAIN:
        return f"{view['world']} is in the {view['split']} split; the teacher reads train only"
    if view["pattern"] not in POC_PATTERNS:
        return (f"{view['world']} has the {view['pattern']} demand pattern, whose hidden "
                "state ADR-0056 defines no posterior for")
    return None


def pre_registered_sample(atlas: Mapping, reduction: str | None = None
                          ) -> list[tuple[str, str]]:
    """(class, world) of D5's sample, hard first, each class in rank order; `reduction` = a
    pre-registered smaller size of the same rule (`REDUCTIONS`)."""
    if reduction is not None and reduction not in REDUCTIONS:
        raise ValueError(f"unknown reduction {reduction!r}; known: {sorted(REDUCTIONS)}")
    size, hard = (REDUCTIONS[reduction][:2] if reduction is not None
                  else (SAMPLE_SIZE, SAMPLE_HARD))
    pool = {v["world"]: v for v in map(world_view, atlas["worlds"])
            if v["split"] == TRAIN and v["size"] == SAMPLE_PRODUCERS
            and v["pattern"] in POC_PATTERNS}
    ranked = {c: sorted((w for w, v in pool.items() if v["class"] == c),
                        key=lambda w: -(pool[w]["gain"] if c == HARD else pool[w]["room"]))
              for c in (HARD, OPEN)}
    n = {HARD: min(hard, len(ranked[HARD]))}
    n[OPEN] = min(size - n[HARD], len(ranked[OPEN]))
    n[HARD] = min(size - n[OPEN], len(ranked[HARD]))
    out: list[tuple[str, str]] = []
    for c in (HARD, OPEN):
        want = {True: n[c] // 2, False: n[c] - n[c] // 2}
        pick: list[str] = []
        for shocked in (True, False):
            pick += [w for w in ranked[c] if pool[w]["shocks"] == shocked][:want[shocked]]
        pick += [w for w in ranked[c] if w not in pick][:n[c] - len(pick)]
        out += [(c, w) for w in sorted(pick, key=ranked[c].index)]
    return out


def first_confirm_seed(view: Mapping) -> int:
    """The confirm stage's first episode seed of the world (atlas `--mode confirm`)."""
    info = space_info(view["record"])
    return episode_seeds(info.id, confirm_episode_count(info), CONFIRM_STAGE)[0]


def select_worlds(atlas: Mapping, worlds: Sequence[str] | None = None,
                  max_worlds: int | None = None, reduction: str | None = None) -> list[dict]:
    """The sampled world views (`worlds`: a dry run's explicit ids, each refused like the
    sample would be). Any refused world raises."""
    views = {v["world"]: v for v in map(world_view, atlas["worlds"])}
    if worlds and reduction is not None:
        raise ValueError("--worlds and --reduction are exclusive: a reduction is a sample")
    if worlds:
        unknown = sorted(set(worlds) - set(views))
        if unknown:
            raise ValueError(f"worlds {unknown} are not in the atlas")
        chosen = [views[w] for w in worlds]
    else:
        chosen = [views[w] for _, w in pre_registered_sample(atlas, reduction)]
    for v in chosen:
        why = refusal(v)
        if why:
            raise ValueError(f"refused: {why}")
    return chosen[:max_worlds] if max_worlds is not None else chosen


def world_entry(view: Mapping, shock_profile: str | None = None):  # -> WorldEntry
    """The job-path world of one atlas world: its recorded config (hash-checked), or with
    `shocks-i-smoke` the start window moved to cycles 2-8 (dry run)."""
    rec = view["record"]
    point = point_from_record(rec)
    if shock_profile is None:
        config_from_record(rec)
    else:
        point = replace(point, shock_window=tuple(SMOKE_START_WINDOW))
    return space_entry(point, (first_confirm_seed(view),), split=TRAIN)


# --- arms ---


def teacher_arm(arm: str) -> dict:
    return {"router": arm, "subject": SUBJECT[arm], "producing": PRODUCER,
            "ordering": FLOOR_ROUTER, "contract": JOINT_STACK, "random": False,
            "roster": False, "kind": DETERMINISTIC, "trained": arm != FLOOR,
            "side": BOUND if arm == UB else TEACHER, "teacher": arm,
            "hidden_information": arm == UB}


def baseline_arms(roster: Roster, arms: Sequence[str] | None = None,
                  replicates: int | None = None) -> list[dict]:
    """D12's blind and reach arms with the confirm's R (`portable`, a reported arm, is not
    rerun)."""
    return [a for a in arm_table(roster, CONFIRM, arms, replicates) if a["side"] != REPORTED]


def world_arms(view: Mapping, baseline: Sequence[dict], teachers: Sequence[str]) -> list[dict]:
    """`ub` only where the future is hidden (shocks or stochastic demand)."""
    differing = space_info(view["record"]).differing
    return list(baseline) + [teacher_arm(a) for a in teachers if a != UB or differing]


# --- worker: one teacher episode ---


def _decision_row(job: Job, arm: str, source: str, rec: object | None,
                  sample: object | None, cycle: int, agent: str) -> dict:
    d = {"world": job["world"], "episode": job["episode"], "arm": arm,
         "hidden_information": arm == UB, "scenario_source": source, "cycle": cycle,
         "agent": agent}
    if sample is not None:
        d.update(posterior_draws=sample.draws, posterior_accepted=sample.accepted,
                 acceptance_rate=sample.rate, unresolved=sample.unresolved)
    else:
        d["unresolved"] = False
    if rec is not None:
        r = asdict(rec)
        stream = r.pop("stream_seconds")
        d.update({k: v for k, v in r.items() if k not in ("cycle", "agent", "searched")})
        d["searched_streams"] = rec.searched
        d["stream_seconds"] = json.dumps({str(k): v for k, v in sorted(stream.items())})
        d["seconds_per_searched_stream"] = (sum(stream.values()) / len(stream)
                                            if stream else None)
    return d


def run_teacher_episode(job: Job, *, max_search_actions: int | None = None
                        ) -> tuple[dict[str, str], list[dict]]:
    """One `floor` / `d2` / `ub` episode: (row, teacher decision rows)."""
    from ..env.shock_posterior import (
        HistoryRecorder,
        PosteriorScenarioSource,
        ShockPosterior,
        with_floor,
    )
    from .baselines import PRODUCING_BASELINES
    from .bounded_rollout_orderer import (
        BoundedScenarioOrderer,
        ClairvoyantSource,
        DeterministicSource,
        PosteriorSource,
        WholeOrderCandidates,
        deterministic_future,
    )
    from .gate import row_from_stats, row_to_dict
    from .joint_stack import joint_stack_env
    from .runner import run_episode

    arm = job["teacher"]
    cfg = job_config(job)
    horizon = job["horizon"]
    env = joint_stack_env(cfg, horizon)
    # No routing tracer: its hooks replace `system.complete_cycle` by a closure over the live
    # system, which `deepcopy` shares, so every fork (and the recorder's stripped copies) would
    # complete the live system's cycle. Teacher rows leave the `trace_*` cells empty.
    monitor = ContractMonitor(env)
    prod = PRODUCING_BASELINES[PRODUCER]
    floor = ordering_policy(FLOOR_ROUTER)
    decisions: list[dict] = []
    posterior = None
    search = None
    source_name = ""
    started = time.perf_counter()
    if arm == FLOOR:
        prod_policy, order_policy = prod, floor
    else:
        candidates = WholeOrderCandidates(trained_policy(job, "orderer"))
        recorder = None
        if arm == UB:
            scenarios: object = ClairvoyantSource()
        elif deterministic_future(cfg):
            scenarios = DeterministicSource()
        else:
            recorder = HistoryRecorder(keep_replay=False)
            posterior = PosteriorScenarioSource(recorder, ShockPosterior(cfg),
                                                task_id=job["world"], episode_seed=job["episode"])
            scenarios = PosteriorSource(posterior)
        source_name = scenarios.name  # type: ignore[attr-defined]
        search = BoundedScenarioOrderer(
            task_id=job["world"], episode_seed=job["episode"], prod=prod, depth=DEPTH,
            base=floor, scenarios=scenarios, candidates=candidates,  # type: ignore[arg-type]
            record_decisions=True, max_search_actions=max_search_actions)
        teacher = with_floor(search, floor, posterior) if posterior is not None else search

        def logged(env_: object, agent: str):  # noqa: ANN202 - a Policy
            n = len(search.decisions)
            action = teacher(env_, agent)
            rec = search.decisions[n] if len(search.decisions) > n else None
            sample = posterior.sample(env_) if posterior is not None else None
            decisions.append(_decision_row(job, arm, source_name, rec, sample,
                                           int(env_.current_cycle), agent))
            return action

        prod_policy, order_policy = prod, logged
        if recorder is not None:  # every agent wrapped from reset (ADR-0056)
            prod_policy, order_policy = recorder.wrap(prod), recorder.wrap(logged)
    run_episode(env, prod_policy, order_policy, seed=job["episode"], action_seed=None,
                monitor=monitor)
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=horizon)
    row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"], job["episode"],
                                     job["subject"], job["replicate"], stats))
    row.update(bound_cells(job))
    unresolved = sum(bool(d["unresolved"]) for d in decisions)
    log = posterior.log if posterior is not None else []
    tel = search.telemetry if search is not None else None
    row.update({
        "arm": arm, "side": BOUND if arm == UB else TEACHER,
        "hidden_information": str(arm == UB), "scenario_source": source_name,
        "decisions": str(len(decisions)), "unresolved_decisions": str(unresolved),
        "unresolved_fraction": repr(unresolved / len(decisions)) if decisions else "",
        "posterior_cycles": str(len(log)) if posterior is not None else "",
        "unresolved_cycles": str(sum(r.unresolved for r in log)) if posterior is not None
        else "",
        "mean_acceptance_rate": repr(fmean(r.rate for r in log)) if log else "",
        "searched_streams": str(tel.searched_streams) if tel else "",
        "override_streams": str(tel.override_streams) if tel else "",
        "rollouts": str(tel.rollout_calls) if tel else "",
        "search_seconds": repr(time.perf_counter() - started) if tel else "",
    })
    return row, decisions


def decisions_path(out: Path, job: Job) -> Path:
    safe = job["subject"].replace("/", "_")
    return out / "artifacts" / "decisions" / f"{job['world']}__{job['episode']}__{safe}.csv"


def run_audit_job(job: Job) -> dict[str, str]:
    """A teacher job (row + its decision file) or the atlas job of a baseline arm; a failure
    stays a failure row."""
    if not job.get("teacher"):
        return run_atlas_job(job)
    try:
        row, decisions = run_teacher_episode(job,
                                             max_search_actions=job.get("max_search_actions"))
        if job.get("decisions_file"):
            path = Path(job["decisions_file"])
            path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(path, DECISION_COLUMNS, decisions)
        return row
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


def checked_orderer(out: Path, path: str, log: Callable[[str], None]) -> dict[str, str]:
    """{path, sha256} of the trained orderer after a trial load in the parent; a checkpoint that
    does not hash or load is a `run` row and `ContractRunError` before any job runs."""
    from .contract import CHECKPOINT

    try:
        pin = {"path": str(path), "sha256": sha256_file(path)}
        load_trained_policy("orderer", path)
    except Exception as exc:  # noqa: BLE001 - recorded, then the run ends invalid
        out.mkdir(parents=True, exist_ok=True)
        append_failure(out, run_failure(CHECKPOINT, exc))
        spec_path = out / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {
            "experiment": out.name, "driver": DRIVER}
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(read_failures(out))}
        _write_spec(out, spec)
        log(f"[teacher_audit] INVALID: orderer checkpoint failed to hash or load ({exc})")
        raise ContractRunError(f"orderer checkpoint failed to hash or load "
                               f"({type(exc).__name__}: {exc})") from exc
    return pin


# --- pure analysis ---


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def world_ratios(dicts: Sequence[Mapping[str, str]], views: Mapping[str, Mapping],
                 baseline: Sequence[Mapping]) -> list[dict]:
    """Per sampled world-episode: the teacher's units against the atlas bar (best blind arm on
    this world-episode), the best reach arm, the ceiling C and B2 (from the baseline arms only,
    as the atlas defines them), and its reading (D5). `ub` columns are the clairvoyant bound:
    hidden information, a reference only."""
    base_subjects = {a["subject"] for a in baseline}
    base_vals = episode_values([d for d in dicts if d["subject"] in base_subjects])
    all_vals = episode_values(dicts)
    out = []
    for w, view in views.items():
        if w not in all_vals:
            continue
        for ep in sorted(all_vals[w]):
            arms = all_vals[w][ep].arms
            m = measures(w, {ep: base_vals[w][ep]}, baseline) if w in base_vals else None
            units = {a: (fmean(arms[SUBJECT[a]].units) if SUBJECT[a] in arms else None)
                     for a in TEACHER_ARMS}
            dead = {a: (any(arms[SUBJECT[a]].dead) if SUBJECT[a] in arms else None)
                    for a in TEACHER_ARMS}
            bar = m.bar if m else None
            reach = m.reach if m else None
            ceiling = m.ceiling if m else None
            nf = view.get("noise_floor")
            d2, ub = units[D2], units[UB]
            margin = GATE_MARGINS["shocked" if view["shocks"] else "calm"]
            d2_beats = (d2 is not None and bar is not None and nf is not None
                        and d2 - bar > nf)
            ub_beats = (ub is not None and bar is not None and nf is not None
                        and ub - bar > nf)
            if d2 is None or bar is None:
                reading = None
            elif view["class"] == OPEN:
                # `ub` absent = a deterministic-future world, where it equals `d2` (and
                # `ub_beats` is False, as `d2_beats` is here)
                reading = ("search_headroom" if d2_beats else "fork_limited" if ub_beats
                           else "ceiling_limited")
            elif view["class"] == HARD:
                reading = ("beyond_trained" if reach is not None and d2 >= reach
                           else "candidate_or_depth_limited")
            else:
                reading = "unclassified"
            d2_rows = [d for d in dicts if d["world"] == w and int(d["episode"]) == ep
                       and d["subject"] == SUBJECT[D2]]
            uf = d2_rows[0].get("unresolved_fraction") if d2_rows else None
            ar = d2_rows[0].get("mean_acceptance_rate") if d2_rows else None
            out.append({
                "world": w, "class": view["class"], "size": view["size"],
                "choice": view["choice"], "pattern": view["pattern"],
                "shocks": view["shocks"], "episode": ep, "noise_floor": nf,
                "bar_arm": m.bar_arm if m else None, "bar": bar,
                "reach_arm": m.reach_arm if m else None, "reach": reach,
                "ceiling": ceiling, "b2": m.b2 if m else None,
                "floor_units": units[FLOOR], "d2_units": d2,
                "d2_minus_bar": d2 - bar if d2 is not None and bar is not None else None,
                "d2_over_bar": _ratio(d2, bar),
                "d2_minus_reach": d2 - reach if d2 is not None and reach is not None else None,
                "d2_over_ceiling": _ratio(d2, ceiling),
                "d2_room_closed": ((d2 - bar) / (ceiling - bar)
                                   if d2 is not None and bar is not None and ceiling
                                   and ceiling > bar else None),
                "d2_beats_bar": d2_beats if d2 is not None else None,
                "d2_ge_reach": (d2 >= reach) if d2 is not None and reach is not None else None,
                "d2_below_bar": (d2 < bar) if d2 is not None and bar is not None else None,
                "gate_margin_ref": margin,
                "d2_clears_margin_ref": (d2 >= margin * bar)
                if d2 is not None and bar is not None else None,
                "d2_unresolved_fraction": float(uf) if uf not in (None, "") else None,
                "d2_mean_acceptance_rate": float(ar) if ar not in (None, "") else None,
                "floor_dead": dead[FLOOR], "d2_dead": dead[D2],
                "bar_dead": (any(base_vals[w][ep].arms[m.bar_arm].dead)
                             if m and m.bar_arm else None),
                "reading": reading,
                "clairvoyant_ub_units": ub,
                "clairvoyant_ub_minus_bar": ub - bar if ub is not None and bar is not None
                else None,
                "clairvoyant_ub_over_bar": _ratio(ub, bar),
                "clairvoyant_ub_beats_bar": ub_beats if ub is not None else None,
                "clairvoyant_ub_dead": dead[UB],
                "clairvoyant_ub_hidden_information": True if ub is not None else None,
            })
    return out


def class_readings(ratios: Sequence[Mapping]) -> dict:
    """D5's per-class readings over the sampled worlds (no verdict, no margin)."""
    out: dict[str, dict] = {}
    opens = [r for r in ratios if r["class"] == OPEN and r["d2_units"] is not None]
    if opens:
        beats = sum(bool(r["d2_beats_bar"]) for r in opens)
        out[OPEN] = {
            "worlds": len(opens), "d2_beats_bar": beats,
            "search_headroom": beats * 2 >= len(opens),
            "readings": {k: sum(r["reading"] == k for r in opens)
                         for k in ("search_headroom", "fork_limited", "ceiling_limited")},
            "teacher_label_candidates": [r["world"] for r in opens if r["d2_beats_bar"]]}
    hards = [r for r in ratios if r["class"] == HARD and r["d2_units"] is not None]
    if hards:
        out[HARD] = {
            "worlds": len(hards),
            "beyond_trained": sum(r["reading"] == "beyond_trained" for r in hards),
            "candidate_or_depth_limited": [r["world"] for r in hards
                                           if r["reading"] == "candidate_or_depth_limited"]}
    d2 = [r for r in ratios if r["d2_units"] is not None]
    out["all"] = {
        "worlds": len(d2), "below_bar": [r["world"] for r in d2 if r["d2_below_bar"]],
        "d2_deaths": sum(bool(r["d2_dead"]) for r in d2),
        "floor_deaths": sum(bool(r["floor_dead"]) for r in ratios if r["floor_dead"]),
        "mean_unresolved_fraction": _mean([r["d2_unresolved_fraction"] for r in d2
                                           if r["d2_unresolved_fraction"] is not None])}
    return out


def floor_failures(dicts: Sequence[Mapping[str, str]]) -> list[Failure]:
    """`floor` is `round_robin` through the teacher's wrappers: its units must equal the
    `entity_planner/round_robin` blind arm's on the same world-episode."""
    rr = f"{PRODUCER}/{FLOOR_ROUTER}"
    by = {(d["world"], d["episode"], d["subject"]): d for d in dicts}
    out = []
    for (w, ep, s), d in sorted(by.items()):
        if s != SUBJECT[FLOOR] or (w, ep, rr) not in by:
            continue
        a, b = float(d["units"]), float(by[(w, ep, rr)]["units"])
        if a != b:
            out.append(Failure(scope=RUN, check=FLOOR_CHECK, error_type="FloorMismatch",
                               message=f"world {w} episode {ep}: {SUBJECT[FLOOR]} units {a:g} "
                                       f"!= {rr} units {b:g}"))
    return out


# --- driver ---


QUESTION = ("On atlas worlds where the bound leaves room, does a lookahead search (EXP-0042's "
            "bounded rollout at depth 2 over `round_robin`, posterior forks) reach more than the "
            "best cheap arm? (design D5)")
RESULT_SKELETON = """# {title}

{banner}**Status: {status}.** Not an acceptance read: no verdict, the gate margins are reference
ratios only. {n_rows} episode rows over {n_worlds} sampled world-episodes, teacher arms
{teacher_arms}, the clairvoyant bound `ub` on {n_ub} world(s) (hidden information: a reference,
never a teacher), {n_baseline} blind and reach arms, horizon {horizon}. Produced by
`python -m {driver}` (see `spec.json`).

## Question

{question}

## Result

TODO: plain-English reading of `ratios.csv` per class (open: `d2` beats the bar by more than the
noise floor on at least half the worlds → search finds headroom; else fork-limited or
ceiling-limited; hard: `d2` against the best reach arm), worlds below the bar, deaths, the
unresolved fraction and acceptance rates from `teacher_decisions.csv`. The posterior's acceptance
rate is binary by construction (ADR-0056 "Floor"): 1.0 on a feasible plan (16 of 16 draws), 0 and
unresolved on an infeasible one.

## Conclusions

TODO.
"""
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered worlds, arms or replicates, "
              "another horizon or shock window, a search cap, or no source commit; see "
              "`spec.json`.\n\n")


def real_run_problems(*, atlas: Mapping, roster: Roster, horizon: int, source_commit: str,
                      filters: Mapping[str, object]) -> list[str]:
    problems = [msg for bad, msg in (
        (atlas.get("content_hash") != REGISTERED_ATLAS_HASH,
         f"atlas {str(atlas.get('content_hash'))[:12]} is not the registered EXP-0052 "
         f"reclassified atlas {REGISTERED_ATLAS_HASH[:12]}"),
        (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (not lp_available(), "the ceiling's B_lp needs scipy: pip install -e .[lp]"),
    ) if bad]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


def run_teacher_audit(*, atlas_path: str | Path, out_dir: str | Path, orderer_ckpt: str,
                      roster: Roster, workers: int = 1, horizon: int = HORIZON,
                      worlds: Sequence[str] | None = None, max_worlds: int | None = None,
                      arms: Sequence[str] | None = None,
                      baseline: Sequence[str] | None = None, replicates: int | None = None,
                      shock_profile: str | None = None, max_search_actions: int | None = None,
                      reduction: str | None = None,
                      source_commit: str = "", dry_run: bool = False,
                      run_job_fn: Callable[[Job], dict[str, str]] = run_audit_job,
                      log: Callable[[str], None] = print) -> int:
    """Run (or resume) the teacher audit into `out_dir`; return the exit status (0, or
    `CONTRACT_INVALID` on contract failures, a digest mismatch, a bound violation, a floor
    mismatch or missing rows). `baseline` = blind / reach routers to keep (dry run; `[]` = none)."""
    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    teachers = tuple(arms) if arms is not None else TEACHER_ARMS
    if set(teachers) - set(TEACHER_ARMS):
        raise ValueError(f"unknown teacher arms {sorted(set(teachers) - set(TEACHER_ARMS))}; "
                         f"known: {TEACHER_ARMS}")
    atlas_path = Path(atlas_path)
    atlas = load_atlas(atlas_path)
    views = select_worlds(atlas, worlds, max_worlds, reduction)
    if not dry_run:
        problems = real_run_problems(
            atlas=atlas, roster=roster, horizon=horizon, source_commit=source_commit,
            filters={"worlds": worlds, "max_worlds": max_worlds, "arms": arms,
                     "baseline": baseline, "replicates": replicates,
                     "shock_profile": shock_profile, "max_search_actions": max_search_actions})
        if problems:
            raise ValueError("not a valid teacher audit run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    base_table = ([] if baseline is not None and not list(baseline)
                  else baseline_arms(roster, baseline or None, replicates))
    out = Path(out_dir)
    needs_orderer = any(a["trained"] for a in base_table) or any(t != FLOOR for t in teachers)
    pin = checked_orderer(out, orderer_ckpt, log) if needs_orderer else None
    checkpoints = {"orderer": pin} if pin else {}
    entries = [world_entry(v, shock_profile) for v in views]
    tables = {v["world"]: world_arms(v, base_table, teachers) for v in views}
    grid = job_grid(entries, None)
    expected: set[Key] = set()
    for e in entries:
        expected |= expected_keys({e.id: grid[e.id]}, tables[e.id], 1)
    source = source_fingerprint(source_commit or None)
    classes = {w: c for c, w in pre_registered_sample(atlas, reduction)}
    pins = {
        "driver": DRIVER, "horizon": horizon, "dry_run": dry_run, "depth": DEPTH,
        "atlas": {"path": str(atlas_path), "sha256": sha256_file(atlas_path),
                  "content_hash": atlas["content_hash"]},
        "roster_version": roster.version, "checkpoints": {"orderer": pin["sha256"]} if pin
        else {}, "producer": PRODUCER, "floor": FLOOR_ROUTER, "teacher_arms": list(teachers),
        "baseline_arms": base_table, "shock_profile": shock_profile,
        "max_search_actions": max_search_actions, "reduction": reduction,
        "sample": [{"world": v["world"], "class": v["class"], "shocks": v["shocks"],
                    "pattern": v["pattern"], "pre_registered": v["world"] in classes,
                    "arms": [a["subject"] for a in tables[v["world"]]]} for v in views],
        "worlds": [v["record"] for v in views],
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "audit",
        "acceptance_read": False, "pins": pins, "inputs": checkpoints, "source": source,
        "pre_registered": {
            "sample": "D5: train split, 12 producers, 6 hard by gain + 6 open by room, per "
                      "class n // 2 shocked, patterns " + "/".join(POC_PATTERNS)
                      + ", the confirm's first episode seed",
            "depth": DEPTH, "floor": FLOOR_ROUTER,
            "candidates": "whole order to each eligible producer + the trained orderer's row",
            "scenarios": "ADR-0056 posterior forks; one scenario (zero-variance rule) on a "
                         "deterministic-future world",
            "teacher_arms": [FLOOR, D2],
            "reference_arms": {UB: {"hidden_information": True, "teacher": False,
                                    "label_source": False}},
            "gate_margins_reference_only": GATE_MARGINS,
            "reduction": ({"name": reduction, "size": REDUCTIONS[reduction][0],
                           "hard": REDUCTIONS[reduction][1], "note": REDUCTIONS[reduction][2]}
                          if reduction is not None else None)},
        "driver_choices": {
            "ambiguity_gate": "none: every stream with >= 2 eligible producers is searched "
                              "(a whole-order floor is never ambiguous)",
            "candidate_dedup": "a candidate equal to the floor's row or an earlier candidate "
                               "is dropped",
            "bar": "best blind arm on the same world-episode (this run's rows)",
            "floor_check": f"{SUBJECT[FLOOR]} units == {PRODUCER}/{FLOOR_ROUTER} units"},
    })
    ensure_failures_file(out)
    jobs: list[Job] = []
    for e in entries:
        jobs += build_jobs([e], tables[e.id], 1, horizon, None, checkpoints)
    by_subject = {a["subject"]: a for t in tables.values() for a in t}
    for j in jobs:
        arm = by_subject[j["subject"]]
        if arm.get("teacher"):
            j["teacher"] = arm["teacher"]
            j["max_search_actions"] = max_search_actions
            j["decisions_file"] = str(decisions_path(out, j))
    if lp_available():
        flag_lp_jobs(jobs)
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    execute(out, spec, done_keys, read_episode_dicts(episodes), jobs, workers, run_job_fn, log,
            columns=COLUMNS)

    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    dicts = read_episode_dicts(episodes)
    recorded = {f["message"] for f in read_failures(out)}
    for f in bound_violations(dicts) + floor_failures(dicts):
        if f.message not in recorded:
            append_failure(out, f)
            log(f"[teacher_audit] {f.check} {f.message}")
    merged: list[dict] = []
    for j in jobs:
        p = Path(j["decisions_file"]) if j.get("decisions_file") else None
        if p is not None and p.exists() and p.stat().st_size:
            with p.open(newline="") as fh:
                merged += list(csv.DictReader(fh))
    write_csv(out / "teacher_decisions.csv", DECISION_COLUMNS, merged)
    failures = read_failures(out)
    ratios = world_ratios(dicts, {v["world"]: v for v in views}, base_table)
    write_csv(out / "ratios.csv", RATIO_COLUMNS, ratios)
    if failures or missing:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing),
                          "bound_violations": [f["message"] for f in failures
                                               if f["check"] == BOUND_CHECK]}
        status = f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing row(s))"
        code = CONTRACT_INVALID
    else:
        spec["status"] = "complete"
        spec["result"] = {"status": "complete", "rows": len(rows), "contract_failures": 0,
                          "acceptance_read": False, "readings": class_readings(ratios),
                          "teacher_decisions": len(merged)}
        status = "complete"
        code = 0
    _write_spec(out, spec)
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", status=status,
            n_rows=len(rows), n_worlds=sum(len(g["episodes"]) for g in grid.values()),
            teacher_arms=", ".join(f"`{t}`" for t in teachers if t != UB),
            n_ub=sum(any(a["subject"] == SUBJECT[UB] for a in t) for t in tables.values()),
            n_baseline=len(base_table), horizon=horizon, driver=DRIVER, question=QUESTION))
    log(f"[teacher_audit] {'DRY RUN ' if dry_run else ''}{status}")
    return code


def _csv_strs(s: str | None) -> list[str] | None:
    return None if s is None else [x for x in s.split(",") if x]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--atlas", required=True, help="atlas_v2.json (EXP-0052 reclassified)")
    p.add_argument("--orderer-ckpt", required=True, help="the trained orderer (EXP-0012)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--roster", default=str(DEFAULT_ROSTER))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worlds", help="dry run: comma-separated atlas world ids")
    p.add_argument("--max-worlds", type=int, help="dry run: the first N sampled worlds")
    p.add_argument("--arms", help=f"dry run: teacher arms out of {','.join(TEACHER_ARMS)}")
    p.add_argument("--baseline", help="dry run: blind / reach routers to keep ('' = none)")
    p.add_argument("--replicates", type=int, help="dry run: R of every stochastic arm")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--shock-profile", choices=(SHOCKS_I_SMOKE,))
    p.add_argument("--reduction", choices=sorted(REDUCTIONS),
                   help="a pre-registered smaller sample (allowed in a real run)")
    p.add_argument("--max-search-actions", type=int,
                   help="dry run: search at most N decisions per teacher episode")
    a = p.parse_args(argv)
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    try:
        return run_teacher_audit(
            atlas_path=a.atlas, out_dir=a.out_dir, orderer_ckpt=a.orderer_ckpt,
            roster=load_roster(a.roster), workers=a.workers, horizon=a.horizon,
            worlds=_csv_strs(a.worlds), max_worlds=a.max_worlds, arms=_csv_strs(a.arms),
            baseline=_csv_strs(a.baseline), replicates=a.replicates,
            shock_profile=a.shock_profile, max_search_actions=a.max_search_actions,
            reduction=a.reduction,
            source_commit=commit, dry_run=a.dry_run)
    except ContractRunError:
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
