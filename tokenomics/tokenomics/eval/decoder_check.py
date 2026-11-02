"""EXP-0057 driver: `python -m tokenomics.eval.decoder_check` (add-m4-orderer-redesign design D3,
learned-orderer "Decoder check precedes training").

Question: does the trained orderer (EXP-0012 scores) lose to whole-order `round_robin` because of
its decode (a softmax split of every order) rather than its scores? Every arm plays with the
`entity_planner` producer on `routing_audit`'s job path, routing tracer on:
- TR  = the trained orderer as deployed;
- TS  = `trained_sampled_whole`: one eligible producer per order drawn from softmax(score) at
  T = 1 by the keyed uniform, the order routed whole (R = 2);
- MWWk = `mww_keyed`: MWW's weights (makeable_qty_now) drawn with the same keyed uniform, so
  TS - MWWk is paired (R = 2);
- MWW = the atlas arm `makeable_weighted_whole` (`action_space.np_random`, R = 2), for
  reproduction of EXP-0052;
- TW  = `trained_whole` (argmax whole);
- RR  = `round_robin`.
Every job sets its `DrawKey(world, episode seed, replicate)` (`set_draw_key`); a keyed router
without one raises, so the job is a contract failure, never a default key.

Sample (D3): atlas v2 train split, rho > 0.28, v2 class not `collapse` (23 worlds at 12
producers, 14 at 42), at the confirm's episode seeds (2 per world, 1 on cyclic shock-free
worlds). Only the 12-producer worlds enter the readings; the 42-producer table is reported.

Readings are pure functions of the episode rows (`world_table`, `primary_reading`,
`fidelity_reading`, `dispersion_reading`), each with D3's precedence; the licence comes from the
primary reading alone. Floors are per comparison: max(atlas `noise_floor`, 3 SE of the run's own
paired differences), the own SE only with >= 2 paired observations (the n = 1 rule).

Validity (exit 2): contract failures, missing rows, digest differences across arms, and the
reproduction of EXP-0056 (TR / TW / RR, the three shocked worlds, first episode) and EXP-0052
(TR / RR / MWW, the confirm rows): units, death and digests must be equal.

`--mode arm-compare --arm utwf` (task 5.5, EXP-0067) replays every EXP-0057 world-episode with
one more keyed router (uTWF, R = 2, EXP-0057's draw keys) and compares it with
EXP-0057's recorded RR, MWWk and TS rows (`run_arm_compare`); RR is replayed once per
world-episode only to check neutrality against EXP-0057. The check mode is unchanged.

Outputs: `spec.json`, `episodes.csv` (gate rows + `trace_*` + `arm` + score-fidelity cells),
`decoder_check.csv`, `fidelity.csv`, `dispersion.csv`, `reading.json`, `reproduction.csv`,
`routing_metrics.csv`, `contract_failures.csv`, a `RESULT.md` skeleton. A real run needs the
registered atlas, checkpoint and reference hashes, horizon 100, a 40-hex source commit and no
filters; anything else needs `--dry-run`. Resumes on the same `--out-dir` when every pin matches
(the gate's `_prepare_resume`).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .atlas import (
    COLLAPSE,
    CONFIRM,
    CONFIRM_STAGE,
    TRAIN,
    arm_table,
    confirm_episode_count,
    quartile_cuts,
    space_entry,
    space_info,
)
from .baselines import PRODUCING_BASELINES, Policy
from .contract import (
    DIGEST,
    ContractMonitor,
    ContractRunError,
    append_failure,
    ensure_failures_file,
    job_failure,
    read_failures,
    run_failure,
)
from .gate import DigestMismatch, row_from_stats, row_to_dict
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
from .registry import WorldEntry
from .provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .roster import DEFAULT_ROSTER, JOINT_STACK, Roster, load_roster, member_id
from .routing_audit import (
    EPISODE_COLUMNS,
    METRIC_COLUMNS,
    PRODUCER,
    TRAINED_ORDERER,
    _is_commit,
    _trained_entry,
    build_jobs,
    execute,
    expected_keys,
    job_config,
    metrics_table,
    ordering_policy,
    read_episode_dicts,
    trace_cells,
    trained_policy,
    write_csv,
)
from .routing_diagnostics import (
    DETERMINISTIC,
    ROUTER_KIND,
    STOCHASTIC,
    TRAINED_WRAPPERS,
    DrawKey,
    set_draw_key,
)
from .routing_trace import RoutingTracer, ScoreFidelity
from .shock_profiles import SHOCKS_I_SMOKE, SMOKE_START_WINDOW
from .teacher_audit import (
    REGISTERED_ATLAS_HASH,
    ROLE_CHECKPOINTS,
    checked_orderer,
    load_atlas,
    world_view,
)
from .world_space import config_from_record, episode_seeds, point_from_record

DRIVER = "tokenomics.eval.decoder_check"
EXPERIMENT = "EXP-0057"
HORIZON = 100
READING_SIZE = 12  # D3: only the 12-producer worlds enter the readings
REPORTED_SIZE = 42
SIZES = (READING_SIZE, REPORTED_SIZE)
RHO_LOADED = 0.28
EXPECTED_WORLDS = {READING_SIZE: 23, REPORTED_SIZE: 14}
EXP0056_WORLDS = ("ws-atlas-v1-12-0060", "ws-atlas-v1-12-0111", "ws-atlas-v1-12-0344")
KEYED_R = 2  # D3: TS and MWWk replicates

# registered inputs (a real run refuses anything else)
EXP0052_EPISODES_SHA = "0e22ddd0546da404b098045d87737550ce06d9574c66ceb58762fc14ce9a5b12"
EXP0056_EPISODES_SHA = "6963ee42af61e2db9a5fb3273ce4eee18943d23f626371e8eb278d6f6d82c62a"
EXP0052, EXP0056 = "exp0052", "exp0056"

TR, TS, MWWK, MWW, TW, RR = "TR", "TS", "MWWk", "MWW", "TW", "RR"
ARMS = (TR, TS, MWWK, MWW, TW, RR)
ROUTER = {TR: TRAINED_ORDERER, TS: "trained_sampled_whole", MWWK: "mww_keyed",
          MWW: "makeable_weighted_whole", TW: "trained_whole", RR: "round_robin"}
SUBJECT = {a: member_id(PRODUCER, r) for a, r in ROUTER.items()}
# replicates per arm (D3; MWW = the atlas confirm's R): a world's arm with any other replicate set
# is incomplete, so a lost replicate never pairs silently
ARM_REPLICATES = {TR: 1, TS: KEYED_R, MWWK: KEYED_R, MWW: 2, TW: 1, RR: 1}
FIDELITY_ARMS = (TR, TS, TW)  # the scorer is wrapped by ScoreFidelity (reads only)
REFERENCE_ARMS = {EXP0056: (TR, TW, RR), EXP0052: (TR, RR, MWW)}
REPRODUCTION = "reference_reproduction"

# D3 thresholds
NO_GAP_MIN = 5
REC_STRONG = 2 / 3
SHARE_STRONG = 2 / 3
REC_PARTIAL = 1 / 3
FAITHFUL_SHARE = 2 / 3
FIDELITY_SIDE_SHARE = 1 / 3
FLOOR_SE = 3.0
MIN_OWN_PAIRS = 2
HISTORY_LINE = 0.05
SHARE_LINE = 0.10
DISPERSION_SHARE = 2 / 3
BORDERLINE_BAND = 0.02

NO_GAP, DECODER, DECODER_PARTIAL, NOT_DECODER = "no_gap", "decoder", "decoder_partial", "not_decoder"
FAITHFUL, MIXED, FIDELITY_LOSS, BEYOND_TARGET = "faithful", "mixed", "fidelity_loss", "beyond_target"
HISTORY_DIFFERS, HISTORY_FLAT = "history_differs", "history_flat"
INCOMPLETE = "incomplete"
EQUIVALENT, BELOW, ABOVE = "equivalent", "below", "above"
LICENCE = {NO_GAP: "none: diagnose", DECODER: "step 2",
           DECODER_PARTIAL: "step 2 (the unexplained share is recorded)",
           NOT_DECODER: "none: diagnose", INCOMPLETE: "none: incomplete run"}
DISPERSION_NEXT = {HISTORY_DIFFERS: "none",
                   HISTORY_FLAT: "recorded; the supervisor asks the user before group 4, since "
                                 "agent-local history would carry little contrast"}

# D3 interpretation table: primary (licensed rows merged) x fidelity
_LICENSED = "decoder / decoder_partial (licensed)"
INTERPRETATION = {
    _LICENSED: {
        FAITHFUL: "decode is the bottleneck; scores carry their target",
        FIDELITY_LOSS: "decode is the bottleneck; MWWk is the stronger reference (noted)",
        BEYOND_TARGET: "decode is the bottleneck; scores add to their target (noted)",
        MIXED: "decode is the bottleneck; fidelity varies by world (noted)"},
    f"{NOT_DECODER} (diagnose)": {
        FAITHFUL: "whole input-aware sampling does not close the gap: start from rotation and "
                  "per-agent offsets vs sampling",
        FIDELITY_LOSS: "start from score fidelity (a design update would change the base)",
        BEYOND_TARGET: "start from what TS's extra signal costs against RR",
        MIXED: "start world by world"},
    f"{NO_GAP} (diagnose)": {
        f: "start from why the addendum's loss did not reproduce"
        for f in (FAITHFUL, FIDELITY_LOSS, BEYOND_TARGET, MIXED)},
}
KL_TARGET = ("softmax(makeable_logits(m)): log makeable_qty_now clipped to +-10 "
             "(env.spaces.ACTION_LOGIT_BOUND), a -10 logit for a producer with nothing makeable, "
             "uniform when nothing is makeable (routing_trace.makeable_logits, the EXP-0012 "
             "scorer's distillation target); not D3's plain makeable share, whose KL is infinite "
             "wherever a producer has m = 0")

# task 1.2's offline prediction (notes/1.2.md, pre_pod_checks.md §d): MWW in TS's place
PREDICTION = {"reading": DECODER, "worlds": 23, "gap_worlds": 17, "rec_strong_worlds": 16,
              "share_rec_ge_2_3": 16 / 17, "median_rec": 0.908,
              "flips_if": "TS reaches less than about 0.78 of MWW's gain over TR",
              "source": "openspec/changes/add-m4-orderer-redesign/notes/1.2.md, task 1.2 (§d)"}

HISTORY = "trace_agent_history_dispersion"
HISTORY_N = "trace_agent_history_cells"
SHARE = "trace_routed_share_dispersion"
SHARE_N = "trace_routed_share_pairs"
ALTERNATIVES = {  # reported only (1.3 review, carry-forward): measure -> its count column
    "producer_streams": ("trace_routed_share_dispersion_producer_streams",
                         "trace_routed_share_pairs_producer_streams"),
    "vs_rest": ("trace_routed_share_dispersion_vs_rest", SHARE_N),
    "unit_weighted": ("trace_routed_share_dispersion_unit_weighted", SHARE_N),
}
FIDELITY_COLUMNS = tuple(ScoreFidelity(lambda env, agent: None).cells())
COLUMNS = EPISODE_COLUMNS + ("arm",) + FIDELITY_COLUMNS


# --- sample and arms ---


def sample_views(atlas: Mapping) -> list[dict]:
    """D3's worlds: train split, rho > 0.28, v2 class not collapse, 12 and 42 producers, sorted
    by (size, world)."""
    out = []
    for entry in atlas["worlds"]:
        v = world_view(entry)
        v["rho"] = float(entry["record"]["coordinates"]["rho"])
        if (v["split"] == TRAIN and v["size"] in SIZES and v["rho"] > RHO_LOADED
                and v["class"] != COLLAPSE):
            out.append(v)
    return sorted(out, key=lambda v: (v["size"], v["world"]))


def select_views(atlas: Mapping, worlds: Sequence[str] | None = None,
                 max_worlds: int | None = None) -> list[dict]:
    """The sample, or a dry run's explicit train worlds (others are refused)."""
    if not worlds:
        chosen = sample_views(atlas)
    else:
        views = {}
        for entry in atlas["worlds"]:
            v = world_view(entry)
            v["rho"] = float(entry["record"]["coordinates"]["rho"])
            views[v["world"]] = v
        unknown = sorted(set(worlds) - set(views))
        if unknown:
            raise ValueError(f"worlds {unknown} are not in the atlas")
        chosen = [views[w] for w in worlds]
        for v in chosen:
            if v["split"] != TRAIN:
                raise ValueError(f"refused: {v['world']} is in the {v['split']} split; the "
                                 "decoder check reads train only")
            if v["size"] not in SIZES:
                raise ValueError(f"refused: {v['world']} has {v['size']} producers")
    return chosen[:max_worlds] if max_worlds is not None else chosen


def confirm_seeds(view: Mapping) -> tuple[int, ...]:
    """The confirm stage's episode seeds in generation order (EXP-0052's row order)."""
    info = space_info(view["record"])
    return tuple(episode_seeds(info.id, confirm_episode_count(info), CONFIRM_STAGE))


def world_entry(view: Mapping, shock_profile: str | None = None) -> WorldEntry:
    """The job-path world: its recorded config (hash-checked) at every confirm seed, or with
    `shocks-i-smoke` the shock window moved to cycles 2-8 (dry run)."""
    rec = view["record"]
    point = point_from_record(rec)
    if shock_profile is None:
        config_from_record(rec)
    else:
        point = replace(point, shock_window=tuple(SMOKE_START_WINDOW))
    return space_entry(point, confirm_seeds(view), split=TRAIN)


def decoder_arm(arm: str, roster: Roster) -> dict:
    """The arm's routing_audit router entry plus `arm`."""
    if arm == TR:
        entry = _trained_entry(TRAINED_ORDERER)
    elif arm in (TS, TW):
        stochastic = arm == TS
        entry = {"router": ROUTER[arm], "subject": SUBJECT[arm], "producing": PRODUCER,
                 "ordering": ROUTER[arm], "contract": JOINT_STACK, "random": stochastic,
                 "roster": False, "kind": STOCHASTIC if stochastic else DETERMINISTIC,
                 "trained": True}
        if stochastic:
            entry["replicates"] = KEYED_R
    elif arm == MWWK:
        entry = {"router": ROUTER[arm], "subject": SUBJECT[arm], "producing": PRODUCER,
                 "ordering": ROUTER[arm], "contract": JOINT_STACK, "random": True,
                 "roster": False, "kind": ROUTER_KIND[ROUTER[arm]], "trained": False,
                 "replicates": KEYED_R}
    elif arm in (MWW, RR):
        entry = arm_table(roster, CONFIRM, [ROUTER[arm]])[0]
    else:
        raise ValueError(f"unknown decoder-check arm {arm!r}; known: {ARMS}")
    if entry["subject"] != SUBJECT[arm]:
        raise AssertionError(f"{arm}: subject {entry['subject']} != {SUBJECT[arm]}")
    return {**entry, "arm": arm}


def job_cost(job: Job) -> float:
    """Relative wall-clock estimate for longest-first scheduling (task 1.3 timings: a
    42-producer episode about 8x a 12-producer one; a torch scorer about 3x a rule)."""
    return (job["size"] / READING_SIZE) ** 2 * (3.0 if job.get("checkpoints") else 1.0)


def longest_first(jobs: Sequence[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: -job_cost(j))


# --- worker ---


def decoder_policies(job: Job) -> tuple[Policy, Policy, ScoreFidelity | None]:
    """entity_planner and the arm's orderer; a trained arm's scorer is wrapped by ScoreFidelity
    (inside TS's / TW's wrapper, so the scores it sees are the ones decoded)."""
    prod = PRODUCING_BASELINES[job["producing"]]
    ordering = job["ordering"]
    if ordering == TRAINED_ORDERER:
        fid = ScoreFidelity(trained_policy(job, "orderer"))
        return prod, fid, fid
    if ordering in TRAINED_WRAPPERS:
        fid = ScoreFidelity(trained_policy(job, "orderer"))
        return prod, TRAINED_WRAPPERS[ordering](fid), fid
    return prod, ordering_policy(ordering), None


def job_draw_key(job: Job) -> DrawKey:
    """(world, episode seed, replicate), ints coerced (a CSV-parsed value would be refused)."""
    return DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"]))


def run_decoder_job(job: Job) -> dict[str, str]:
    """One traced episode's row, or its failure row (anything raising, contract checks)."""
    try:
        from .joint_stack import joint_stack_env
        from .runner import run_episode

        if job["contract"] != JOINT_STACK:
            raise ValueError(f"{job['subject']}: contract {job['contract']} (D3: joint_stack)")
        env = joint_stack_env(job_config(job), job["horizon"])
        set_draw_key(env, job_draw_key(job))
        tracer = RoutingTracer()
        tracer.attach(env)
        monitor = ContractMonitor(env)
        prod, order, fid = decoder_policies(job)
        run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"],
                    monitor=monitor)
        stats = env.episode_stats
        monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"],
                                         job["episode"], job["subject"], job["replicate"],
                                         stats))
        row.update(trace_cells(tracer))
        row["arm"] = job["arm"]
        row.update(fid.cells() if fid is not None else {c: "" for c in FIDELITY_COLUMNS})
        return row
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


# --- pure arithmetic ---

Units = dict[int, dict[int, float]]  # episode seed -> replicate -> value


def arm_units(dicts: Sequence[Mapping[str, str]], arm_of: Mapping[str, str] | None = None,
              column: str = "units") -> dict[str, dict[str, Units]]:
    """world -> arm -> episode -> replicate -> float(column). The arm is the row's `arm`, or
    `arm_of[subject]` (rows of another run, e.g. EXP-0052's); rows of other subjects are
    skipped."""
    out: dict[str, dict[str, Units]] = {}
    for d in dicts:
        arm = arm_of.get(d["subject"]) if arm_of is not None else d.get("arm")
        if not arm:
            continue
        out.setdefault(d["world"], {}).setdefault(arm, {}).setdefault(
            int(d["episode"]), {})[int(d["replicate"])] = float(d[column])
    return out


def world_mean(u: Units) -> float:
    """Equal-weight mean over episodes of the replicate means (atlas D12)."""
    return statistics.fmean(statistics.fmean(r.values()) for _, r in sorted(u.items()))


def paired_diffs(a: Units, b: Units) -> list[float]:
    """a - b per paired observation: per episode, replicate by replicate when both arms have the
    same replicates (TS - MWWk share keys), else every replicate of the stochastic arm against
    the deterministic arm's one row."""
    if set(a) != set(b):
        raise ValueError(f"episodes differ: {sorted(a)} vs {sorted(b)}")
    out: list[float] = []
    for ep in sorted(a):
        ra, rb = a[ep], b[ep]
        if set(ra) == set(rb):
            out += [ra[r] - rb[r] for r in sorted(ra)]
        elif len(rb) == 1:
            (y,) = rb.values()
            out += [ra[r] - y for r in sorted(ra)]
        elif len(ra) == 1:
            (x,) = ra.values()
            out += [x - rb[r] for r in sorted(rb)]
        else:
            raise ValueError(f"episode {ep}: replicates {sorted(ra)} vs {sorted(rb)} do not pair")
    return out


def own_3se(diffs: Sequence[float]) -> float | None:
    """3 x SE of the paired differences; None below 2 paired observations (the n = 1 rule)."""
    if len(diffs) < MIN_OWN_PAIRS:
        return None
    return FLOOR_SE * statistics.stdev(diffs) / math.sqrt(len(diffs))


@dataclass(frozen=True)
class Comparison:
    """One compared pair of arms on one world (D3 floors: each comparison its own)."""

    a: float
    b: float
    diff: float  # a_w - b_w
    pairs: int
    atlas_floor: float
    own3se: float | None

    @property
    def floor(self) -> float:
        return max(self.atlas_floor, self.own3se) if self.own3se is not None else self.atlas_floor


def compare(a: Units, b: Units, atlas_floor: float) -> Comparison:
    diffs = paired_diffs(a, b)
    return Comparison(world_mean(a), world_mean(b), world_mean(a) - world_mean(b), len(diffs),
                      float(atlas_floor), own_3se(diffs))


@dataclass(frozen=True)
class WorldResult:
    world: str
    size: int
    choice: str
    rho: float
    episodes: int
    means: dict[str, float]
    gap: Comparison | None  # RR vs TR
    fidelity: Comparison | None  # TS vs MWWk
    ts_tw: Comparison | None  # TS vs TW (secondary)

    @property
    def is_gap(self) -> bool:
        """A gap world: G_w = RR_w - TR_w > floor_w (a gap exactly at the floor is none)."""
        return self.gap is not None and self.gap.diff > self.gap.floor

    @property
    def rec(self) -> float | None:
        """rec_w = (TS_w - TR_w) / G_w (no floor on rec_w; D3 gives none)."""
        if self.gap is None or not self.gap.diff or TS not in self.means:
            return None
        return (self.means[TS] - self.means[TR]) / self.gap.diff

    @property
    def relation(self) -> str | None:
        """TS against MWWk: equivalent within the floor, else below / above it."""
        f = self.fidelity
        if f is None:
            return None
        if abs(f.diff) <= f.floor:
            return EQUIVALENT
        return BELOW if f.diff < 0 else ABOVE


def complete_arm(arm: str, u: Units, replicates: Mapping[str, int] = ARM_REPLICATES) -> bool:
    """Every episode of the arm holds exactly replicates 0 .. R - 1."""
    want = set(range(replicates.get(arm, 1)))
    return bool(u) and all(set(reps) == want for reps in u.values())


def world_table(units: Mapping[str, Mapping[str, Units]], views: Sequence[Mapping],
                replicates: Mapping[str, int] = ARM_REPLICATES) -> list[WorldResult]:
    """Per sampled world, the three comparisons (None where an arm has no rows or lacks a
    replicate: never paired from partial rows)."""
    out = []
    for v in views:
        per = {a: u for a, u in units.get(v["world"], {}).items()
               if complete_arm(a, u, replicates)}
        floor = float(v["noise_floor"])

        def cmp(a: str, b: str) -> Comparison | None:
            if a not in per or b not in per:
                return None
            try:
                return compare(per[a], per[b], floor)
            except ValueError:  # an episode or replicate missing (the run is invalid anyway)
                return None

        eps = {e for u in per.values() for e in u}
        out.append(WorldResult(v["world"], int(v["size"]), v["choice"], float(v["rho"]),
                               len(eps), {a: world_mean(u) for a, u in sorted(per.items())},
                               cmp(RR, TR), cmp(TS, MWWK), cmp(TS, TW)))
    return out


def _at_least(n: int, total: int, share: float) -> bool:
    """n / total >= share, exactly (shares are 1/3 or 2/3)."""
    return total > 0 and n * 3 >= round(share * 3) * total


def primary_reading(results: Sequence[WorldResult], size: int = READING_SIZE) -> dict:
    """D3 primary reading on `size`'s worlds, first match wins: `no_gap` (< 5 gap worlds) ->
    `decoder` (rec_w >= 2/3 on >= 2/3 of gap worlds) -> `decoder_partial` (median rec_w >= 1/3)
    -> `not_decoder`."""
    ws = [r for r in results if r.size == size]
    if not ws or any(r.gap is None or TS not in r.means for r in ws):
        return {"reading": INCOMPLETE, "licence": LICENCE[INCOMPLETE], "worlds": len(ws)}
    gw = [r for r in ws if r.is_gap]
    recs = [r.rec for r in gw]
    strong = sum(1 for x in recs if x >= REC_STRONG)
    med = statistics.median(recs) if recs else None
    if len(gw) < NO_GAP_MIN:
        reading = NO_GAP
    elif _at_least(strong, len(gw), SHARE_STRONG):
        reading = DECODER
    elif med >= REC_PARTIAL:
        reading = DECODER_PARTIAL
    else:
        reading = NOT_DECODER
    return {"reading": reading, "licence": LICENCE[reading], "worlds": len(ws),
            "gap_worlds": len(gw), "rec_strong_worlds": strong,
            "share_rec_ge_2_3": strong / len(gw) if gw else None, "median_rec": med,
            "gap_world_ids": [r.world for r in gw]}


def fidelity_reading(results: Sequence[WorldResult], size: int = READING_SIZE) -> dict:
    """D3 fidelity reading on `size`'s worlds, first match wins: `faithful` (equivalent on >= 2/3)
    -> `mixed` (both or neither of the next two) -> `fidelity_loss` (TS below MWWk beyond the
    floor on >= 1/3) -> `beyond_target` (above, >= 1/3)."""
    ws = [r for r in results if r.size == size]
    if not ws or any(r.fidelity is None for r in ws):
        return {"reading": INCOMPLETE, "worlds": len(ws)}
    n = len(ws)
    counts = {k: sum(1 for r in ws if r.relation == k) for k in (EQUIVALENT, BELOW, ABOVE)}
    loss = _at_least(counts[BELOW], n, FIDELITY_SIDE_SHARE)
    beyond = _at_least(counts[ABOVE], n, FIDELITY_SIDE_SHARE)
    if _at_least(counts[EQUIVALENT], n, FAITHFUL_SHARE):
        reading = FAITHFUL
    elif loss == beyond:
        reading = MIXED
    elif loss:
        reading = FIDELITY_LOSS
    else:
        reading = BEYOND_TARGET
    return {"reading": reading, "worlds": n, **counts, "loss_rule": loss,
            "beyond_rule": beyond}


def interpretation(primary: str, fidelity: str) -> str | None:
    """The D3 table's cell (None when either reading is incomplete)."""
    if primary in (DECODER, DECODER_PARTIAL):
        row = _LICENSED
    elif primary in (NOT_DECODER, NO_GAP):
        row = f"{primary} (diagnose)"
    else:
        return None
    return INTERPRETATION[row].get(fidelity)


def _row_values(dicts: Sequence[Mapping[str, str]], arm: str, value: str, count: str
                ) -> dict[str, Units]:
    """world -> episode -> replicate -> value, only the rows whose count column is > 0 (an
    empty measure is undefined, never 0)."""
    out: dict[str, Units] = {}
    for d in dicts:
        if (d.get("arm") != arm or not d.get(count) or float(d[count]) <= 0
                or d.get(value) in (None, "")):
            continue
        out.setdefault(d["world"], {}).setdefault(int(d["episode"]), {})[
            int(d["replicate"])] = float(d[value])
    return out


def dispersion_values(dicts: Sequence[Mapping[str, str]], arm: str
                      ) -> dict[str, dict[str, float | None]]:
    """world -> measure -> the world value (equal-weight over episodes of the replicate means of
    the defined rows) or None when no row of the world defines it. Measures: `history`,
    `share` (the pre-registered two) and the reported alternatives."""
    cols = {"history": (HISTORY, HISTORY_N), "share": (SHARE, SHARE_N), **ALTERNATIVES}
    worlds = {d["world"] for d in dicts if d.get("arm") == arm}
    out: dict[str, dict[str, float | None]] = {w: {} for w in worlds}
    for name, (value, count) in cols.items():
        vals = _row_values(dicts, arm, value, count)
        for w in worlds:
            out[w][name] = world_mean(vals[w]) if w in vals else None
    return out


def _dispersion_rule(values: Mapping[str, Mapping[str, float | None]], share_key: str,
                     moved: str | None = None) -> dict:
    """The D3 dispersion rule over worlds whose two measures are defined; `moved` = "pass" /
    "fail" forces every value within the band to pass / fail (the borderline check)."""
    defined = sorted(w for w, v in values.items()
                     if v["history"] is not None and v[share_key] is not None)
    skipped = sorted(set(values) - set(defined))

    def ok(x: float, line: float) -> bool:
        if moved is not None and abs(x - line) <= BORDERLINE_BAND:
            return moved == "pass"
        return x >= line

    passing = [w for w in defined if ok(values[w]["history"], HISTORY_LINE)
               and ok(values[w][share_key], SHARE_LINE)]
    reading = (HISTORY_DIFFERS if _at_least(len(passing), len(defined), DISPERSION_SHARE)
               else HISTORY_FLAT)
    return {"reading": reading, "defined_worlds": len(defined), "passing_worlds": len(passing),
            "skipped_worlds": skipped, "passing": passing}


def dispersion_reading(dicts: Sequence[Mapping[str, str]], views: Sequence[Mapping],
                       size: int = READING_SIZE) -> dict:
    """D3 dispersion reading under TS on `size`'s worlds: `history_differs` when both measures
    reach their lines (>= 0.05, >= 0.10) on >= 2/3 of the worlds, else `history_flat`. A world
    whose count column is 0 for a measure is undefined for it and skipped (never below the
    line); with no defined world the reading is `history_flat`. `borderline`: a pre-registered
    value within +-0.02 of its line on a deciding world (a world that enters the reading);
    `borderline_changes_reading`: moving every in-band value to pass vs to fail changes the
    reading. The alternatives (producer streams, vs rest, unit-weighted) replace the share
    measure, reported only."""
    ws = {v["world"] for v in views if int(v["size"]) == size}
    values = {w: v for w, v in dispersion_values(dicts, TS).items() if w in ws}
    if not values:
        return {"reading": INCOMPLETE, "defined_worlds": 0}
    base = _dispersion_rule(values, "share")
    in_band = sorted(
        w for w in values if w not in base["skipped_worlds"]
        and (abs(values[w]["history"] - HISTORY_LINE) <= BORDERLINE_BAND
             or abs(values[w]["share"] - SHARE_LINE) <= BORDERLINE_BAND))
    up = _dispersion_rule(values, "share", "pass")["reading"]
    down = _dispersion_rule(values, "share", "fail")["reading"]
    alts = {name: {k: v for k, v in _dispersion_rule(values, name).items() if k != "passing"}
            for name in ALTERNATIVES}
    return {**{k: v for k, v in base.items() if k != "passing"},
            "next": DISPERSION_NEXT[base["reading"]],
            "thresholds": {"agent_history_dispersion": HISTORY_LINE,
                           "routed_share_dispersion": SHARE_LINE, "share": DISPERSION_SHARE},
            "skipped_reason": "a count column (trace_agent_history_cells / "
                              "trace_routed_share_pairs) is 0 on every row of the world: the "
                              "measure is undefined there, never below the line",
            "borderline": bool(in_band), "borderline_worlds": in_band,
            "borderline_band": BORDERLINE_BAND,
            "borderline_changes_reading": up != down,
            "alternatives_reported_only": alts}


def _median_cells(dicts: Sequence[Mapping[str, str]], arm: str, column: str) -> float | None:
    xs = [float(d[column]) for d in dicts if d.get("arm") == arm and d.get(column)]
    return statistics.median(xs) if xs else None


def score_fidelity(dicts: Sequence[Mapping[str, str]], worlds: set[str]) -> dict:
    """Per trained arm on `worlds`: the median over episode rows of the per-row medians
    (Spearman vs log makeable, KL) and the decision-weighted top-1 agreement."""
    out = {}
    for arm in FIDELITY_ARMS:
        rows = [d for d in dicts if d.get("arm") == arm and d["world"] in worlds]
        if not rows:
            continue
        t1 = [(float(d["score_top1_agreement"]), int(d["score_top1_decisions"]))
              for d in rows if d.get("score_top1_agreement")]
        n1 = sum(n for _, n in t1)
        out[arm] = {
            "spearman_log_makeable": _median_cells(rows, arm, "score_spearman_log_makeable"),
            "top1_agreement": sum(x * n for x, n in t1) / n1 if n1 else None,
            "kl_to_makeable_logits": _median_cells(rows, arm, "score_kl_to_makeable"),
            "decisions": sum(int(d.get("score_decisions") or 0) for d in rows),
            "rows": len(rows)}
    return out


def rho_quartiles(results: Sequence[WorldResult]) -> dict[str, str]:
    """world -> q1..q4 by nearest-rank quartile cuts of rho within its size."""
    out = {}
    for size in sorted({r.size for r in results}):
        rs = [r for r in results if r.size == size]
        cuts = quartile_cuts([r.rho for r in rs])
        for r in rs:
            out[r.world] = f"q{1 + sum(1 for c in cuts if r.rho > c)}"
    return out


DECODER_COLUMNS = (
    "scope", "size", "world", "cell", "rho", "rho_quartile", "worlds", "episodes",
    "enters_reading", *ARMS, "G_w", "gap_atlas_floor", "gap_own_3se", "gap_pairs", "gap_floor",
    "gap", "rec_w", "ts_minus_mwwk", "fidelity_floor", "relation", "ts_minus_tw", "ts_tw_floor",
    "ts_tw_pairs", "ts_tw_beyond_floor")


def decoder_rows(results: Sequence[WorldResult]) -> list[dict]:
    """`decoder_check.csv`: per world, then the secondary reads (never gating) per size x choice
    cell and per rho quartile: mean TS - TW, worlds with TS - TW beyond its floor, the gap
    worlds' median rec_w."""
    q = rho_quartiles(results)
    rows = []
    for r in results:
        g, f, t = r.gap, r.fidelity, r.ts_tw
        rows.append({
            "scope": "world", "size": r.size, "world": r.world, "cell": f"{r.size}/{r.choice}",
            "rho": r.rho, "rho_quartile": q[r.world], "worlds": 1, "episodes": r.episodes,
            "enters_reading": r.size == READING_SIZE, **{a: r.means.get(a) for a in ARMS},
            "G_w": g.diff if g else None, "gap_atlas_floor": g.atlas_floor if g else None,
            "gap_own_3se": g.own3se if g else None, "gap_pairs": g.pairs if g else None,
            "gap_floor": g.floor if g else None, "gap": r.is_gap if g else None, "rec_w": r.rec,
            "ts_minus_mwwk": f.diff if f else None, "fidelity_floor": f.floor if f else None,
            "relation": r.relation, "ts_minus_tw": t.diff if t else None,
            "ts_tw_floor": t.floor if t else None, "ts_tw_pairs": t.pairs if t else None,
            "ts_tw_beyond_floor": (t.diff > t.floor) if t else None})
    for scope, key in (("cell", lambda r: f"{r.size}/{r.choice}"),
                       ("rho_quartile", lambda r: q[r.world])):
        groups: dict[tuple[int, str], list[WorldResult]] = {}
        for r in results:
            groups.setdefault((r.size, key(r)), []).append(r)
        for size, name in sorted(groups):
            rs = groups[(size, name)]
            tt = [r.ts_tw for r in rs if r.ts_tw is not None]
            recs = [r.rec for r in rs if r.is_gap and r.rec is not None]
            rows.append({
                "scope": scope, "size": size, scope: name, "worlds": len(rs),
                "enters_reading": False,
                "ts_minus_tw": statistics.fmean(t.diff for t in tt) if tt else None,
                "ts_tw_beyond_floor": sum(t.diff > t.floor for t in tt) if tt else None,
                "gap": sum(r.is_gap for r in rs), "rec_w": statistics.median(recs) if recs
                else None})
    return rows


FIDELITY_TABLE_COLUMNS = (
    "size", "world", "enters_reading", "episodes", "TS", "MWWk", "ts_minus_mwwk", "pairs",
    "atlas_floor", "own_3se", "floor", "relation", *(f"{a}_{m}" for a in FIDELITY_ARMS for m in (
        "spearman_log_makeable", "top1_agreement", "kl_to_makeable_logits")))


def fidelity_rows(results: Sequence[WorldResult], dicts: Sequence[Mapping[str, str]]
                  ) -> list[dict]:
    rows = []
    for r in results:
        f = r.fidelity
        fid = score_fidelity(dicts, {r.world})
        rows.append({
            "size": r.size, "world": r.world, "enters_reading": r.size == READING_SIZE,
            "episodes": r.episodes, "TS": r.means.get(TS), "MWWk": r.means.get(MWWK),
            "ts_minus_mwwk": f.diff if f else None, "pairs": f.pairs if f else None,
            "atlas_floor": f.atlas_floor if f else None, "own_3se": f.own3se if f else None,
            "floor": f.floor if f else None, "relation": r.relation,
            **{f"{a}_{m}": fid.get(a, {}).get(m) for a in FIDELITY_ARMS for m in (
                "spearman_log_makeable", "top1_agreement", "kl_to_makeable_logits")}})
    return rows


DISPERSION_TABLE_COLUMNS = (
    "size", "world", "arm", "enters_reading", "agent_history_dispersion", "history_defined",
    "history_passes", "history_in_band", "routed_share_dispersion", "share_defined",
    "share_passes", "share_in_band", *(f"{k}_reported" for k in ALTERNATIVES))


def dispersion_rows(dicts: Sequence[Mapping[str, str]], views: Sequence[Mapping]) -> list[dict]:
    """`dispersion.csv`: per world and arm (TS is the reading's arm; MWW, RR and the others are
    reported alongside), the two pre-registered measures and the reported alternatives."""
    size = {v["world"]: int(v["size"]) for v in views}
    rows = []
    for arm in ARMS:
        vals = dispersion_values(dicts, arm)
        for w in sorted(vals, key=lambda w: (size.get(w, 0), w)):
            v = vals[w]
            h, s = v["history"], v["share"]
            rows.append({
                "size": size.get(w), "world": w, "arm": arm,
                "enters_reading": arm == TS and size.get(w) == READING_SIZE,
                "agent_history_dispersion": h, "history_defined": h is not None,
                "history_passes": None if h is None else h >= HISTORY_LINE,
                "history_in_band": None if h is None else abs(h - HISTORY_LINE) <= BORDERLINE_BAND,
                "routed_share_dispersion": s, "share_defined": s is not None,
                "share_passes": None if s is None else s >= SHARE_LINE,
                "share_in_band": None if s is None else abs(s - SHARE_LINE) <= BORDERLINE_BAND,
                **{f"{k}_reported": v[k] for k in ALTERNATIVES}})
    return rows


def readings(results: Sequence[WorldResult], dicts: Sequence[Mapping[str, str]],
             views: Sequence[Mapping]) -> dict:
    """`reading.json`: the three 12-producer readings, the D3 interpretation table and its cell,
    the 42-producer table (reported), score fidelity, the 1.2 prediction."""
    out = {}
    for size, label in ((READING_SIZE, "readings"), (REPORTED_SIZE, "reported_42")):
        p, f = primary_reading(results, size), fidelity_reading(results, size)
        d = dispersion_reading(dicts, views, size)
        ws = {r.world for r in results if r.size == size}
        out[label] = {"size": size, "primary": p, "fidelity": f, "dispersion": d,
                      "interpretation": interpretation(p["reading"], f["reading"]),
                      "score_fidelity": score_fidelity(dicts, ws)}
    out["reported_42"]["reported"] = True
    out["reported_42"]["note"] = ("42-producer worlds are reported only (D3): they never grant, "
                                  "withhold or change a reading")
    main = out["readings"]
    return {
        "experiment": EXPERIMENT,
        "licence": main["primary"].get("licence"),
        "licence_rule": "the licence comes from the primary reading alone; fidelity and "
                        "dispersion never grant or withhold one",
        "primary": main["primary"], "fidelity": main["fidelity"],
        "dispersion": main["dispersion"], "interpretation": main["interpretation"],
        "score_fidelity": main["score_fidelity"], "kl_target": KL_TARGET,
        "interpretation_table": INTERPRETATION, "reported_42": out["reported_42"],
        "prediction_task_1_2": PREDICTION,
        "rules": {
            "primary": "G_w = RR_w - TR_w; gap world: G_w > floor_w; rec_w = (TS_w - TR_w) / "
                       "G_w; first match: no_gap (< 5 gap worlds) -> decoder (rec_w >= 2/3 on "
                       ">= 2/3 of gap worlds) -> decoder_partial (median rec_w >= 1/3) -> "
                       "not_decoder",
            "fidelity": "equivalent: |TS_w - MWWk_w| <= floor_w; first match: faithful "
                        "(equivalent on >= 2/3) -> mixed (both or neither of the next two) -> "
                        "fidelity_loss (TS < MWWk beyond the floor on >= 1/3) -> beyond_target "
                        "(TS > MWWk beyond the floor on >= 1/3)",
            "dispersion": "under TS: history_differs when agent_history_dispersion >= 0.05 and "
                          "routed_share_dispersion >= 0.10 on >= 2/3 of the defined worlds, "
                          "else history_flat",
            "floor": "per comparison: max(atlas noise_floor, 3 SE of the run's own paired "
                     "differences); the own SE only with >= 2 paired observations (episodes x "
                     "replicates), else the atlas floor (n = 1 rule)",
            "means": "per world: equal-weight mean over episodes of the replicate means",
            "atlas_floor": "the atlas v2 world entry's recorded noise_floor (the confirm's "
                           "counts, which are the run's)",
            "score_fidelity": "per arm: median over episode rows of the per-row medians "
                              "(Spearman, KL); top-1 decision-weighted; top-1 ties: first argmax",
        },
    }


WITHHELD_KEEP = ("experiment", "dry_run", "licence_rule", "kl_target", "interpretation_table",
                 "prediction_task_1_2", "rules")


def withheld_reading(full: Mapping, failures: int, missing: int) -> dict:
    """`reading.json` of an invalid run: no reading and no licence from partial rows (the
    contract rule); only the pre-registered rules, the table and the prediction."""
    return {**{k: full[k] for k in WITHHELD_KEEP if k in full}, "status": INVALID,
            "readings_withheld": True, "licence": None, "contract_failures": failures,
            "missing_rows": missing,
            "reason": "contract failures or missing rows (contract_failures.csv, spec.json): "
                      "readings are never computed from partial rows"}


# --- reproduction ---

REPRODUCTION_COLUMNS = ("reference", "world", "episode", "arm", "subject", "replicate", "units",
                        "reference_units", "dead", "reference_dead", "demand_digest_equal",
                        "shock_digest_equal", "equal")


def read_reference(path: Path) -> list[dict[str, str]]:
    import csv

    csv.field_size_limit(1 << 30)
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def reproduction(dicts: Sequence[Mapping[str, str]], name: str,
                 ref: Sequence[Mapping[str, str]]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Rows of `REFERENCE_ARMS[name]` against the reference's rows of the same (world, episode,
    subject, replicate): units, death and digests must be equal. EXP-0052 must hold every such
    row of this run; every EXP-0056 row of a world-episode this run played must be in this run
    (when its arm ran). Returns (rows, problems as (check, message))."""
    subjects = {SUBJECT[a]: a for a in REFERENCE_ARMS[name]}
    ran = {d["arm"] for d in dicts}
    mine = {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])): d
            for d in dicts if d["subject"] in subjects}
    theirs = {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])): d
              for d in ref if d["subject"] in subjects}
    played = {(d["world"], int(d["episode"])) for d in dicts}
    if name == EXP0052:
        keys = sorted(mine)
    else:
        keys = sorted(k for k in theirs if k[:2] in played and subjects[k[2]] in ran)
    rows, problems = [], []
    for k in keys:
        a, b = mine.get(k), theirs.get(k)
        where = f"{name}: {k[0]} episode {k[1]} {k[2]} r{k[3]}"
        if b is None:
            problems.append((REPRODUCTION, f"{where}: no reference row"))
            continue
        if a is None:
            problems.append((REPRODUCTION, f"{where}: no row in this run"))
            continue
        dd = a["demand_digest"] == b["demand_digest"]
        sd = a.get("shock_digest", "") == b.get("shock_digest", "")
        eq = (float(a["units"]) == float(b["units"]) and a["dead"] == b["dead"] and dd and sd)
        rows.append({"reference": name, "world": k[0], "episode": k[1], "arm": subjects[k[2]],
                     "subject": k[2], "replicate": k[3], "units": float(a["units"]),
                     "reference_units": float(b["units"]), "dead": a["dead"],
                     "reference_dead": b["dead"], "demand_digest_equal": dd,
                     "shock_digest_equal": sd, "equal": eq})
        if not (dd and sd):
            problems.append((DIGEST, f"{where}: digests differ from the reference's"))
        elif not eq:
            problems.append((REPRODUCTION, f"{where}: units {a['units']} / dead {a['dead']} vs "
                                           f"the reference's {b['units']} / {b['dead']}"))
    return rows, problems


# --- driver ---

QUESTION = ("Does the trained orderer lose to whole-order `round_robin` because of its decode "
            "(a softmax split of every order) rather than its scores? With one producer drawn "
            "per order from softmax(score) and the order routed whole (TS), how much of the "
            "RR - TR gap is recovered on the pre-registered 12-producer worlds? (design D3)")
RESULT_SKELETON = """# {title}

{banner}**Status: {status}.** Not an acceptance read. {n_rows} episode rows over {n_worlds}
worlds ({n_episodes} world-episodes), arms {arms}, `entity_planner` producer, horizon {horizon}.
Produced by `python -m {driver}` (see `spec.json`); readings in `reading.json`.

## Question

{question}

## Prediction (task 1.2, recorded before the run)

Task 1.2 computed the primary reading offline from EXP-0052's confirm rows with MWW in TS's
place: **`{p_reading}`**: {p_gap} gap worlds of {p_worlds}; rec_w >= 2/3 on {p_strong} of
{p_gap} ({p_share:.3f}); median rec_w {p_median:.3f}. It is the prediction for TS under
`faithful`; it flips only if {p_flips}.

## Result

TODO (plain English first): the primary reading (`decoder_check.csv`: G_w, its floor, rec_w per
world; the licence comes from this reading alone), the fidelity reading (`fidelity.csv`: TS -
MWWk against its floor; score fidelity medians, the KL target being {kl_target}), the D3
interpretation cell, the dispersion reading under TS (`dispersion.csv`) beside the reported
alternatives (producer streams, vs rest, unit-weighted) and the borderline flag, the secondary
reads (TS - TW per size x choice cell and per rho quartile), the 42-producer table (reported),
the reproduction of EXP-0056 / EXP-0052 (`reproduction.csv`) and the concentration and herding
columns (`routing_metrics.csv`).

## Conclusions

TODO.
"""
SKELETON_END = "## Conclusions\n\nTODO.\n"  # an unedited skeleton is rewritten on every run
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered worlds or arms, another horizon "
              "or shock window, unregistered inputs or no source commit; see `spec.json`.\n\n")


def real_run_problems(*, atlas: Mapping, roster: Roster, horizon: int, source_commit: str,
                      orderer_sha: str, references: Mapping[str, str | None],
                      filters: Mapping[str, object]) -> list[str]:
    problems = [msg for bad, msg in (
        (atlas.get("content_hash") != REGISTERED_ATLAS_HASH,
         f"atlas {str(atlas.get('content_hash'))[:12]} is not the registered EXP-0052 "
         f"reclassified atlas {REGISTERED_ATLAS_HASH[:12]}"),
        (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (orderer_sha != ROLE_CHECKPOINTS["orderer"],
         f"orderer checkpoint {orderer_sha[:12]} is not the registered "
         f"{ROLE_CHECKPOINTS['orderer'][:12]}"),
        (references.get(EXP0052) != EXP0052_EPISODES_SHA,
         f"--exp0052-episodes {str(references.get(EXP0052))[:12]} is not the registered "
         f"EXP-0052 episodes.csv {EXP0052_EPISODES_SHA[:12]}"),
        (references.get(EXP0056) != EXP0056_EPISODES_SHA,
         f"--exp0056-episodes {str(references.get(EXP0056))[:12]} is not the registered "
         f"EXP-0056 episodes.csv {EXP0056_EPISODES_SHA[:12]}"),
    ) if bad]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


def run_decoder_check(*, atlas_path: str | Path, out_dir: str | Path, orderer_ckpt: str,
                      roster: Roster, exp0052_episodes: str | Path | None = None,
                      exp0056_episodes: str | Path | None = None, workers: int = 1,
                      horizon: int = HORIZON, worlds: Sequence[str] | None = None,
                      max_worlds: int | None = None, arms: Sequence[str] | None = None,
                      shock_profile: str | None = None, source_commit: str = "",
                      dry_run: bool = False,
                      run_job_fn: Callable[[Job], dict[str, str]] = run_decoder_job,
                      log: Callable[[str], None] = print) -> int:
    """Run (or resume) D3's decoder check into `out_dir`; return the exit status (0, or
    `CONTRACT_INVALID` on contract failures, missing rows, digest differences or a reference
    mismatch)."""
    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    chosen_arms = tuple(arms) if arms is not None else ARMS
    if set(chosen_arms) - set(ARMS):
        raise ValueError(f"unknown decoder-check arms {sorted(set(chosen_arms) - set(ARMS))}; "
                         f"known: {ARMS}")
    atlas_path = Path(atlas_path)
    atlas = load_atlas(atlas_path)
    views = select_views(atlas, worlds, max_worlds)
    ref_paths = {n: Path(p) for n, p in ((EXP0052, exp0052_episodes), (EXP0056, exp0056_episodes))
                 if p is not None}
    ref_sha = {n: sha256_file(p) for n, p in ref_paths.items()}
    out = Path(out_dir)
    pin = checked_orderer(out, orderer_ckpt, log)
    if not dry_run:
        problems = real_run_problems(
            atlas=atlas, roster=roster, horizon=horizon, source_commit=source_commit,
            orderer_sha=pin["sha256"], references=ref_sha,
            filters={"worlds": worlds, "max_worlds": max_worlds, "arms": arms,
                     "shock_profile": shock_profile})
        if problems:
            raise ValueError("not a valid decoder-check run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
        counts = {s: sum(1 for v in views if v["size"] == s) for s in SIZES}
        if counts != EXPECTED_WORLDS or not set(EXP0056_WORLDS) <= {v["world"] for v in views}:
            raise ValueError(f"the sample has {counts} worlds, D3 registers {EXPECTED_WORLDS} "
                             f"including {EXP0056_WORLDS}")
    checkpoints = {"orderer": pin}
    entries = [world_entry(v, shock_profile) for v in views]
    table = [decoder_arm(a, roster) for a in chosen_arms]
    grid = job_grid(entries, None)
    expected: set[Key] = expected_keys(grid, table, 1)
    source = source_fingerprint(source_commit or None)
    pins = {
        "driver": DRIVER, "horizon": horizon, "dry_run": dry_run,
        "atlas": {"path": str(atlas_path), "sha256": sha256_file(atlas_path),
                  "content_hash": atlas["content_hash"]},
        "roster_version": roster.version, "checkpoints": {"orderer": pin["sha256"]},
        "references": {n: {"path": str(ref_paths[n]), "sha256": ref_sha[n]} for n in ref_paths},
        "producer": PRODUCER, "arms": list(chosen_arms),
        "subjects": {a: SUBJECT[a] for a in chosen_arms}, "arm_table": table,
        "shock_profile": shock_profile,
        "sample": [{"world": v["world"], "size": v["size"], "class": v["class"],
                    "rho": v["rho"], "choice": v["choice"], "pattern": v["pattern"],
                    "shocks": v["shocks"], "noise_floor": v["noise_floor"]} for v in views],
        "worlds": [v["record"] for v in views],
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "audit",
        "acceptance_read": False, "pins": pins, "inputs": checkpoints, "source": source,
        "pre_registered": {
            "design": "add-m4-orderer-redesign D3",
            "sample": "atlas v2 train split, rho > 0.28, v2 class not collapse (23 at 12, 14 at "
                      "42), the confirm's episode seeds; readings on 12 producers only",
            "arms": {TR: "entity_planner + trained orderer (as deployed)",
                     TS: "trained_sampled_whole: softmax(score) at T = 1, keyed inverse-CDF "
                         "draw per slot order, whole; R = 2",
                     MWWK: "mww_keyed: makeable_qty_now weights, the same keyed draw; R = 2",
                     MWW: "makeable_weighted_whole, action_space.np_random (atlas arm); R = 2",
                     TW: "trained_whole: the scores' argmax, whole",
                     RR: "round_robin"},
            "draw_key": "SHA-256(world, episode seed, replicate, agent, cycle, slot); "
                        "set_draw_key per job",
            "kl_target": KL_TARGET,
            "prediction": PREDICTION,
        },
    })
    ensure_failures_file(out)
    jobs = build_jobs(entries, table, 1, horizon, None, checkpoints)
    by_subject = {a["subject"]: a for a in table}
    for j in jobs:
        j["arm"] = by_subject[j["subject"]]["arm"]
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    try:
        execute(out, spec, done_keys, read_episode_dicts(episodes), longest_first(jobs),
                workers, run_job_fn, log, columns=COLUMNS)
    except DigestMismatch as exc:  # recorded as a run row by execute; the run ends invalid
        log(f"[decoder_check] INVALID: {exc}")
    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    dicts = read_episode_dicts(episodes)

    repro: list[dict] = []
    recorded = {f["message"] for f in read_failures(out)}
    for name, path in sorted(ref_paths.items()):
        got, problems = reproduction(dicts, name, read_reference(path))
        repro += got
        for check, msg in problems:
            if msg not in recorded:
                append_failure(out, run_failure(check, ValueError(msg)))
                log(f"[decoder_check] {check} {msg}")
    write_csv(out / "reproduction.csv", REPRODUCTION_COLUMNS, repro)

    failures = read_failures(out)
    invalid = bool(failures or missing)
    results = world_table(arm_units(dicts), views)
    reading = readings(results, dicts, views)
    reading["dry_run"] = dry_run
    if invalid:
        reading = withheld_reading(reading, len(failures), len(missing))
    else:
        reading.update({"status": "valid", "readings_withheld": False})
    write_csv(out / "decoder_check.csv", DECODER_COLUMNS, decoder_rows(results))
    write_csv(out / "fidelity.csv", FIDELITY_TABLE_COLUMNS, fidelity_rows(results, dicts))
    write_csv(out / "dispersion.csv", DISPERSION_TABLE_COLUMNS, dispersion_rows(dicts, views))
    write_csv(out / "routing_metrics.csv", METRIC_COLUMNS, metrics_table(dicts, table))
    (out / "reading.json").write_text(json.dumps(reading, indent=2, sort_keys=True,
                                                 default=str) + "\n")
    if invalid:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing)}
        status = f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing row(s))"
        code = CONTRACT_INVALID
    else:
        spec["status"] = "complete"
        spec["result"] = {"status": "complete", "rows": len(rows), "contract_failures": 0,
                          "acceptance_read": False, "primary": reading["primary"]["reading"],
                          "fidelity": reading["fidelity"]["reading"],
                          "dispersion": reading["dispersion"]["reading"],
                          "reproduced": sum(r["equal"] for r in repro), "compared": len(repro)}
        status = "complete"
        code = 0
    _write_spec(out, spec)
    result_md = out / "RESULT.md"
    if not result_md.exists() or result_md.read_text().endswith(SKELETON_END):
        result_md.write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", status=status,
            n_rows=len(rows), n_worlds=len(grid),
            n_episodes=sum(len(g["episodes"]) for g in grid.values()),
            arms=", ".join(chosen_arms), horizon=horizon, driver=DRIVER, question=QUESTION,
            p_reading=PREDICTION["reading"], p_gap=PREDICTION["gap_worlds"],
            p_worlds=PREDICTION["worlds"], p_strong=PREDICTION["rec_strong_worlds"],
            p_share=PREDICTION["share_rec_ge_2_3"], p_median=PREDICTION["median_rec"],
            p_flips=PREDICTION["flips_if"], kl_target=KL_TARGET))
    log(f"[decoder_check] {'DRY RUN ' if dry_run else ''}{status}"
        + (f", primary {reading['primary']['reading']}, fidelity "
           f"{reading['fidelity']['reading']}, dispersion {reading['dispersion']['reading']}"
           if code == 0 else ""))
    return code


# --- arm-compare mode (add-m4-orderer-redesign task 5.5, EXP-0067) ---
#
# One more keyed whole-order router (`--arm`, uTWF = `utwf`) replayed on every EXP-0057
# world-episode (R = 2, EXP-0057's draw keys) and compared with EXP-0057's recorded RR, MWWk and
# TS rows; RR is played once more per world-episode only as the neutrality check (it must equal
# EXP-0057's row). The check mode above is unchanged.

CHECK, ARM_COMPARE = "check", "arm-compare"
MODES = (CHECK, ARM_COMPARE)
COMPARE_EXPERIMENT = "EXP-0067"
UTWF = "uTWF"
COMPARE_ROUTERS = {"utwf": UTWF}  # --arm -> arm label (keyed diagnostic routers)
EXP0057 = "exp0057"
COMPARE_REFERENCE_ARMS = (RR, MWWK, TS)  # read from EXP-0057's rows, never rerun
COMPARE_REPLICATES = {UTWF: KEYED_R, RR: 1, MWWK: KEYED_R, TS: KEYED_R}
LOSS_SHARE_DEN = 6  # pre-registered: uTWF loses to RR beyond floor_w on <= 1/6 of the worlds
UTWF_STRONGEST, UTWF_NOT_BETTER = "utwf_strongest", "utwf_not_better"
COMPARE_NEXT = {
    UTWF_STRONGEST: "the supervisor brings two questions to the user before group 6: uTWF into "
                  "roster v3 (the strongest-policy rule), and whether step 2's frozen base "
                  "becomes uTWF's logits (or uTWF's level a phi feature)",
    UTWF_NOT_BETTER: "recorded; step 2 unchanged"}
COMPARE_RULES = {
    "reading": "12 producers decide (42 reported): utwf_strongest when uTWF - RR and uTWF - "
               "MWWk both exceed their per-comparison floor on the 23-world loaded mean, i.e. "
               "mean_w(X_w) > mean_w(floor_w) for each comparison X, AND uTWF loses to RR "
               "(RR_w - uTWF_w > floor_w) on <= 1/6 of the worlds (6 n <= worlds, exact); "
               "utwf_not_better otherwise",
    "floor": "per comparison and world (D3): floor_w = max(atlas noise_floor, 3 SE of the run's "
             "own paired differences), the own SE only with >= 2 paired observations (the n = 1 "
             "rule)",
    "pairing": "uTWF - MWWk / uTWF - TS: per episode, replicate by replicate (the same keys); "
               "uTWF - RR: each uTWF replicate against RR's one row",
    "means": "per world: equal-weight mean over episodes of the replicate means",
    "reference": "RR, MWWk and TS are EXP-0057's recorded rows (episodes.csv SHA-256 "
                 "registered); the reproduced RR row of every world-episode must equal EXP-0057's "
                 "(exp0057_neutrality), and every row's digests must equal EXP-0057's",
    "mean_floor_alternative": "reported only: the mean difference against max(mean atlas "
                              "floor, 3 SE of the per-world differences)",
}


def _exp0057_sha() -> str:
    from .split_need import EXP0057_EPISODES_SHA  # split_need imports this module

    return EXP0057_EPISODES_SHA


def compare_arm(router: str) -> dict:
    """The `--arm` router's routing_audit entry: entity_planner + the keyed router, R = 2."""
    if router not in COMPARE_ROUTERS:
        raise ValueError(f"--arm {router!r}: arm-compare takes {sorted(COMPARE_ROUTERS)}")
    if ROUTER_KIND.get(router) != STOCHASTIC:
        raise AssertionError(f"{router} is not a keyed (stochastic) diagnostic router")
    return {"router": router, "subject": member_id(PRODUCER, router), "producing": PRODUCER,
            "ordering": router, "contract": JOINT_STACK, "random": True, "roster": False,
            "kind": STOCHASTIC, "trained": False, "replicates": KEYED_R,
            "arm": COMPARE_ROUTERS[router]}


@dataclass(frozen=True)
class CompareWorld:
    world: str
    size: int
    choice: str
    rho: float
    episodes: int
    means: dict[str, float]
    utwf_rr: Comparison | None
    utwf_mwwk: Comparison | None
    utwf_ts: Comparison | None  # reported

    @property
    def loses_to_rr(self) -> bool | None:
        """RR_w - uTWF_w > floor_w (uTWF - RR's floor)."""
        c = self.utwf_rr
        return None if c is None else -c.diff > c.floor


def compare_table(units: Mapping[str, Mapping[str, Units]], views: Sequence[Mapping],
                  replicates: Mapping[str, int] = COMPARE_REPLICATES) -> list[CompareWorld]:
    """Per sampled world, uTWF against RR, MWWk (and TS, reported); None where an arm has no
    complete rows (never paired from partial rows)."""
    out = []
    for v in views:
        per = {a: u for a, u in units.get(v["world"], {}).items()
               if complete_arm(a, u, replicates)}
        floor = float(v["noise_floor"])

        def cmp(b: str) -> Comparison | None:
            if UTWF not in per or b not in per:
                return None
            try:
                return compare(per[UTWF], per[b], floor)
            except ValueError:  # episodes differ (the run is invalid anyway)
                return None

        eps = {e for u in per.values() for e in u}
        out.append(CompareWorld(v["world"], int(v["size"]), v["choice"], float(v["rho"]),
                                len(eps), {a: world_mean(u) for a, u in sorted(per.items())},
                                cmp(RR), cmp(MWWK), cmp(TS)))
    return out


def _mean_rule(cs: Sequence[Comparison]) -> dict:
    """The pre-registered mean rule for one comparison over a size's worlds, plus the reported
    alternative floor."""
    diffs = [c.diff for c in cs]
    mean_diff, mean_floor = statistics.fmean(diffs), statistics.fmean(c.floor for c in cs)
    se3 = own_3se(diffs)
    alt = max(statistics.fmean(c.atlas_floor for c in cs), se3 if se3 is not None else 0.0)
    return {"mean_diff": mean_diff, "mean_floor": mean_floor, "exceeds": mean_diff > mean_floor,
            "worlds_beyond_floor": sum(c.diff > c.floor for c in cs),
            "worlds_below_floor": sum(-c.diff > c.floor for c in cs),
            "alternative_floor_reported_only": alt,
            "alternative_exceeds_reported_only": mean_diff > alt}


def compare_reading(results: Sequence[CompareWorld], size: int = READING_SIZE) -> dict:
    """task 5.5's pre-registered reading on `size`'s worlds: `utwf_strongest` when uTWF - RR and
    uTWF - MWWk both exceed their floor on the mean and uTWF loses to RR beyond floor_w on <= 1/6
    of the worlds; else `utwf_not_better` (`incomplete` when a comparison is missing)."""
    ws = [r for r in results if r.size == size]
    if not ws or any(r.utwf_rr is None or r.utwf_mwwk is None for r in ws):
        return {"reading": INCOMPLETE, "worlds": len(ws), "next": None}
    rr = _mean_rule([r.utwf_rr for r in ws])
    mwwk = _mean_rule([r.utwf_mwwk for r in ws])
    ts = (_mean_rule([r.utwf_ts for r in ws]) if all(r.utwf_ts is not None for r in ws) else None)
    losses = [r.world for r in ws if r.loses_to_rr]
    losses_ok = len(losses) * LOSS_SHARE_DEN <= len(ws)
    reading = UTWF_STRONGEST if rr["exceeds"] and mwwk["exceeds"] and losses_ok else UTWF_NOT_BETTER
    return {"reading": reading, "next": COMPARE_NEXT[reading], "worlds": len(ws),
            "utwf_minus_rr": rr, "utwf_minus_mwwk": mwwk, "utwf_minus_ts_reported": ts,
            "losses_to_rr": len(losses), "loss_worlds": losses,
            "max_losses": len(ws) // LOSS_SHARE_DEN, "losses_rule": losses_ok}


COMPARE_COLUMNS = (
    "scope", "size", "world", "cell", "rho", "enters_reading", "worlds", "episodes", UTWF, RR,
    "RR_reproduced", MWWK, TS,
    *(f"{p}_{m}" for p in ("utwf_rr", "utwf_mwwk", "utwf_ts")
      for m in ("diff", "atlas_floor", "own_3se", "pairs", "floor", "beyond_floor",
                "below_floor")),
    "loses_to_rr")


def compare_rows(results: Sequence[CompareWorld], reproduced: Mapping[str, float]
                 ) -> list[dict]:
    """`compare.csv`: per world, then one mean row per size (the reading's two sides)."""
    rows = []
    for r in results:
        row = {"scope": "world", "size": r.size, "world": r.world, "cell": f"{r.size}/{r.choice}",
               "rho": r.rho, "enters_reading": r.size == READING_SIZE, "worlds": 1,
               "episodes": r.episodes, **{a: r.means.get(a) for a in (UTWF, RR, MWWK, TS)},
               "RR_reproduced": reproduced.get(r.world), "loses_to_rr": r.loses_to_rr}
        for p, c in (("utwf_rr", r.utwf_rr), ("utwf_mwwk", r.utwf_mwwk), ("utwf_ts", r.utwf_ts)):
            if c is not None:
                row.update({f"{p}_diff": c.diff, f"{p}_atlas_floor": c.atlas_floor,
                            f"{p}_own_3se": c.own3se, f"{p}_pairs": c.pairs,
                            f"{p}_floor": c.floor, f"{p}_beyond_floor": c.diff > c.floor,
                            f"{p}_below_floor": -c.diff > c.floor})
        rows.append(row)
    for size in sorted({r.size for r in results}):
        ws = [r for r in results if r.size == size]
        row = {"scope": "mean", "size": size, "enters_reading": size == READING_SIZE,
               "worlds": len(ws), "episodes": sum(r.episodes for r in ws),
               "loses_to_rr": sum(bool(r.loses_to_rr) for r in ws)}
        for a in (UTWF, RR, MWWK, TS):
            xs = [r.means[a] for r in ws if a in r.means]
            row[a] = statistics.fmean(xs) if len(xs) == len(ws) else None
        for p in ("utwf_rr", "utwf_mwwk", "utwf_ts"):
            cs = [getattr(r, p) for r in ws]
            if all(c is not None for c in cs):
                row.update({f"{p}_diff": statistics.fmean(c.diff for c in cs),
                            f"{p}_atlas_floor": statistics.fmean(c.atlas_floor for c in cs),
                            f"{p}_floor": statistics.fmean(c.floor for c in cs),
                            f"{p}_beyond_floor": sum(c.diff > c.floor for c in cs),
                            f"{p}_below_floor": sum(-c.diff > c.floor for c in cs)})
        rows.append(row)
    return rows


def reference_digest_problems(dicts: Sequence[Mapping[str, str]],
                              ref: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    """Every row of this run against every EXP-0057 row of its world-episode: demand and shock
    digests equal (digest pairing across the two runs)."""
    theirs: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for d in ref:
        theirs.setdefault((d["world"], int(d["episode"])), set()).add(
            (d["demand_digest"], d.get("shock_digest", "")))
    problems = []
    for d in dicts:
        k = (d["world"], int(d["episode"]))
        mine = (d["demand_digest"], d.get("shock_digest", ""))
        if k in theirs and theirs[k] != {mine}:
            problems.append((DIGEST, f"{k[0]} episode {k[1]} {d['subject']} r{d['replicate']}: "
                                     "digests differ from EXP-0057's"))
    return problems


def compare_run_problems(*, atlas: Mapping, roster: Roster, horizon: int, source_commit: str,
                         orderer_sha: str | None, reference_sha: str | None,
                         filters: Mapping[str, object]) -> list[str]:
    problems = [msg for bad, msg in (
        (atlas.get("content_hash") != REGISTERED_ATLAS_HASH,
         f"atlas {str(atlas.get('content_hash'))[:12]} is not the registered EXP-0052 "
         f"reclassified atlas {REGISTERED_ATLAS_HASH[:12]}"),
        (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (orderer_sha != ROLE_CHECKPOINTS["orderer"],
         f"orderer checkpoint {str(orderer_sha)[:12]} is not the registered "
         f"{ROLE_CHECKPOINTS['orderer'][:12]} (EXP-0057's TS rows)"),
        (reference_sha != _exp0057_sha(),
         f"--exp0057-episodes {str(reference_sha)[:12]} is not the registered EXP-0057 "
         f"episodes.csv {_exp0057_sha()[:12]}"),
    ) if bad]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


COMPARE_QUESTION = (
    "With many ordering agents routing the same product from the same start-of-cycle snapshot, "
    "a deterministic 'pick the best' herds, and drawing in proportion to spare capacity (MWWk) "
    "only partly avoids it. Does the paper's unsplittable Tidal Water Filling (uTWF: each agent, "
    "from its own order size a, the product's number of ordering agents M and the producers' "
    "spare capacity, draws one producer with Definition 2's p_n, so that M agents of its size "
    "would level the spare capacity together) beat both `round_robin` and MWWk on the 23 loaded "
    "12-producer worlds of EXP-0057? (task 5.5; Goren / Vargaftik / Moses, DISC 2020)")
COMPARE_SKELETON = """# {title}

{banner}**Status: {status}.** Not an acceptance read. {n_rows} episode rows over {n_worlds}
worlds ({n_episodes} world-episodes): arm {arm} (`{router}`, R = 2, EXP-0057's draw keys) and
the RR neutrality replay, `entity_planner` producer, horizon {horizon}; RR, MWWk and TS are
EXP-0057's recorded rows. Produced by `python -m {driver} --mode arm-compare` (see
`spec.json`); the reading in `reading.json`.

## Question

{question}

## Pre-registered reading (task 5.5)

{rule}

## Result

TODO (plain English first): what uTWF does and why (the herding literature), the reading and its
two mean comparisons against their floors (`compare.csv` mean rows), the worlds where uTWF loses
to RR beyond the floor, uTWF - TS (reported), the 42-producer table (reported), the neutrality
replay (`neutrality.csv`) and the herding columns (`routing_metrics.csv`).

## Conclusions

TODO.
"""


def run_arm_compare(*, atlas_path: str | Path, out_dir: str | Path, arm: str,
                    exp0057_episodes: str | Path | None = None,
                    orderer_ckpt: str | Path | None = None, roster: Roster | None = None,
                    workers: int = 1, horizon: int = HORIZON,
                    worlds: Sequence[str] | None = None, max_worlds: int | None = None,
                    shock_profile: str | None = None, source_commit: str = "",
                    dry_run: bool = False,
                    run_job_fn: Callable[[Job], dict[str, str]] = run_decoder_job,
                    log: Callable[[str], None] = print) -> int:
    """Run (or resume) `--mode arm-compare` into `out_dir`; return the exit status (0, or
    `CONTRACT_INVALID` on contract failures, missing rows, digest differences or a neutrality
    mismatch). The orderer checkpoint is pinned by SHA-256 only (no arm plays it: it is the
    provenance of EXP-0057's TS rows)."""
    from .split_need import NEUTRALITY_COLUMNS, neutrality

    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    roster = roster if roster is not None else load_roster(DEFAULT_ROSTER)
    arm_entry = compare_arm(arm)
    atlas_path = Path(atlas_path)
    atlas = load_atlas(atlas_path)
    views = select_views(atlas, worlds, max_worlds)
    ref_path = Path(exp0057_episodes) if exp0057_episodes is not None else None
    ref_sha = sha256_file(ref_path) if ref_path is not None else None
    orderer = ({"path": str(orderer_ckpt), "sha256": sha256_file(orderer_ckpt)}
               if orderer_ckpt is not None else None)
    out = Path(out_dir)
    if not dry_run:
        problems = compare_run_problems(
            atlas=atlas, roster=roster, horizon=horizon, source_commit=source_commit,
            orderer_sha=orderer["sha256"] if orderer else None, reference_sha=ref_sha,
            filters={"worlds": worlds, "max_worlds": max_worlds, "shock_profile": shock_profile})
        if problems:
            raise ValueError("not a valid arm-compare run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
        counts = {s: sum(1 for v in views if v["size"] == s) for s in SIZES}
        if counts != EXPECTED_WORLDS:
            raise ValueError(f"the sample has {counts} worlds, D3 registers {EXPECTED_WORLDS}")
    entries = [world_entry(v, shock_profile) for v in views]
    table = [arm_entry, decoder_arm(RR, roster)]
    grid = job_grid(entries, None)
    expected: set[Key] = expected_keys(grid, table, 1)
    played = {(w, int(ep)) for w, g in grid.items() for ep in g["episodes"]}
    ref_subjects = {SUBJECT[a]: a for a in COMPARE_REFERENCE_ARMS}
    reference = None
    if ref_path is not None:  # fail before any job: the compared rows must all be there
        reference = [d for d in read_reference(ref_path)
                     if (d["world"], int(d["episode"])) in played and d["subject"] in ref_subjects]
        have = {(d["world"], int(d["episode"]), ref_subjects[d["subject"]], int(d["replicate"]))
                for d in reference}
        need_arms = COMPARE_REFERENCE_ARMS if not dry_run else (RR,)
        need = {(w, ep, a, r) for w, ep in played for a in need_arms
                for r in range(COMPARE_REPLICATES[a])}
        absent = sorted(need - have)
        if absent:
            raise ValueError(f"--exp0057-episodes lacks {len(absent)} of the {len(need)} "
                             f"compared rows, e.g. {absent[:2]}")
    source = source_fingerprint(source_commit or None)
    pins = {
        "driver": DRIVER, "mode": ARM_COMPARE, "arm": arm, "horizon": horizon,
        "dry_run": dry_run,
        "atlas": {"path": str(atlas_path), "sha256": sha256_file(atlas_path),
                  "content_hash": atlas["content_hash"]},
        "roster_version": roster.version,
        "checkpoints": {"orderer": orderer["sha256"]} if orderer else {},
        "reference": ({EXP0057: {"path": str(ref_path), "sha256": ref_sha}}
                      if ref_path is not None else None),
        "producer": PRODUCER, "arms": [UTWF, RR], "reference_arms": list(COMPARE_REFERENCE_ARMS),
        "subjects": {UTWF: arm_entry["subject"], **{a: SUBJECT[a] for a in COMPARE_REFERENCE_ARMS}},
        "arm_table": table, "shock_profile": shock_profile,
        "sample": [{"world": v["world"], "size": v["size"], "class": v["class"],
                    "rho": v["rho"], "choice": v["choice"], "pattern": v["pattern"],
                    "shocks": v["shocks"], "noise_floor": v["noise_floor"]} for v in views],
        "worlds": [v["record"] for v in views],
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "mode": ARM_COMPARE,
        "kind": "dry-run" if dry_run else "audit", "acceptance_read": False, "pins": pins,
        "inputs": {"orderer": orderer} if orderer else {}, "source": source,
        "pre_registered": {
            "task": "add-m4-orderer-redesign 5.5 (inserted 2026-11-02)",
            "sample": "EXP-0057's: atlas v2 train split, rho > 0.28, not collapse (23 at 12, 14 "
                      "at 42), every confirm episode seed; 12 producers decide, 42 reported",
            "arm": f"{UTWF} = {arm}: the paper's uTWF (Definition 2, Eq. (29)) from local "
                   "information (own order quantity a, static stream count M_p, the producers' "
                   "makeable_qty_now as queues); one keyed inverse-CDF draw (EXP-0057's keys) "
                   "per agent and product, its orders whole to the drawn producer "
                   "(notes/5.5.md); R = 2",
            "rules": COMPARE_RULES, "next": COMPARE_NEXT,
        },
    })
    ensure_failures_file(out)
    jobs = build_jobs(entries, table, 1, horizon, None, None)
    by_subject = {a["subject"]: a for a in table}
    for j in jobs:
        j["arm"] = by_subject[j["subject"]]["arm"]
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    try:
        execute(out, spec, done_keys, read_episode_dicts(episodes), longest_first(jobs),
                workers, run_job_fn, log, columns=COLUMNS)
    except DigestMismatch as exc:  # recorded as a run row by execute; the run ends invalid
        log(f"[decoder_check arm-compare] INVALID: {exc}")
    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    dicts = read_episode_dicts(episodes)

    recorded = {f["message"] for f in read_failures(out)}
    problems: list[tuple[str, str]] = []
    neutral: list[dict] = []
    if reference is not None:
        rr_mine = [d for d in dicts if d.get("arm") == RR]
        neutral, more = neutrality(rr_mine, [d for d in reference if d["subject"] == SUBJECT[RR]])
        problems += more + reference_digest_problems(dicts, reference)
    for check, msg in problems:
        if msg not in recorded:
            append_failure(out, run_failure(check, ValueError(msg)))
            recorded.add(msg)
            log(f"[decoder_check arm-compare] {check} {msg}")
    write_csv(out / "neutrality.csv", NEUTRALITY_COLUMNS, neutral)

    failures = read_failures(out)
    invalid = bool(failures or missing)
    mine = arm_units([d for d in dicts if d.get("arm") == UTWF])
    reproduced = {w: world_mean(u[RR]) for w, u in arm_units(
        [d for d in dicts if d.get("arm") == RR]).items() if RR in u}
    theirs = (arm_units(reference, arm_of=ref_subjects) if reference is not None else
              arm_units([d for d in dicts if d.get("arm") == RR]))  # dry run: own RR only
    units = {w: {**theirs.get(w, {}), **mine.get(w, {})} for w in set(mine) | set(theirs)}
    results = compare_table(units, views)
    reading = {"experiment": COMPARE_EXPERIMENT, "mode": ARM_COMPARE, "arm": arm,
               "dry_run": dry_run, "rules": COMPARE_RULES, "next_by_reading": COMPARE_NEXT,
               "reference": EXP0057 if reference is not None else None,
               "neutrality_checked": reference is not None,
               "neutral_rows": sum(r["equal"] for r in neutral), "compared_rows": len(neutral),
               "reading": compare_reading(results, READING_SIZE),
               "reported_42": {**compare_reading(results, REPORTED_SIZE), "reported": True}}
    if invalid:
        reading = {**withheld_reading(reading, len(failures), len(missing)), "reading": None,
                   "mode": ARM_COMPARE, "arm": arm, "next_by_reading": COMPARE_NEXT}
    else:
        reading.update({"status": "valid", "readings_withheld": False})
    write_csv(out / "compare.csv", COMPARE_COLUMNS, compare_rows(results, reproduced))
    write_csv(out / "routing_metrics.csv", METRIC_COLUMNS, metrics_table(dicts, table))
    (out / "reading.json").write_text(json.dumps(reading, indent=2, sort_keys=True,
                                                 default=str) + "\n")
    if invalid:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing)}
        status = f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing row(s))"
        code = CONTRACT_INVALID
    else:
        spec["status"] = "complete"
        spec["result"] = {"status": "complete", "rows": len(rows), "contract_failures": 0,
                          "acceptance_read": False, "reading": reading["reading"]["reading"],
                          "neutral_rows": reading["neutral_rows"],
                          "compared_rows": reading["compared_rows"]}
        status = "complete"
        code = 0
    _write_spec(out, spec)
    result_md = out / "RESULT.md"
    if not result_md.exists() or result_md.read_text().endswith(SKELETON_END):
        result_md.write_text(COMPARE_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", status=status,
            n_rows=len(rows), n_worlds=len(grid),
            n_episodes=sum(len(g["episodes"]) for g in grid.values()), arm=UTWF, router=arm,
            horizon=horizon, driver=DRIVER, question=COMPARE_QUESTION,
            rule=COMPARE_RULES["reading"][0].upper() + COMPARE_RULES["reading"][1:] + "."))
    log(f"[decoder_check arm-compare] {'DRY RUN ' if dry_run else ''}{status}"
        + (f", reading {reading['reading']['reading']}" if code == 0 else ""))
    return code


def _csv_strs(s: str | None) -> list[str] | None:
    return None if s is None else [x for x in s.split(",") if x]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=MODES, default=CHECK,
                   help="check: EXP-0057's decoder check (default); arm-compare: one keyed "
                        "router (--arm) against EXP-0057's recorded rows (task 5.5, EXP-0067)")
    p.add_argument("--atlas", required=True, help="atlas_v2.json (EXP-0052 reclassified)")
    p.add_argument("--orderer-ckpt", help="the trained orderer (EXP-0012); required by check, "
                                          "pinned by SHA-256 in a real arm-compare run")
    p.add_argument("--arm", choices=sorted(COMPARE_ROUTERS), help="arm-compare: the router (utwf)")
    p.add_argument("--exp0057-episodes", help="arm-compare: EXP-0057's episodes.csv (the "
                                              "compared rows; needed in a real run)")
    p.add_argument("--exp0052-episodes", help="EXP-0052's episodes.csv (TR / RR / MWW "
                                              "reproduction; needed in a real run)")
    p.add_argument("--exp0056-episodes", help="EXP-0056's episodes.csv (TR / TW / RR "
                                              "reproduction; needed in a real run)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--roster", default=str(DEFAULT_ROSTER))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worlds", help="dry run: comma-separated atlas train world ids")
    p.add_argument("--max-worlds", type=int, help="dry run: the first N sampled worlds")
    p.add_argument("--arms", help=f"dry run: arms out of {','.join(ARMS)}")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--shock-profile", choices=(SHOCKS_I_SMOKE,))
    a = p.parse_args(argv)
    if a.mode == ARM_COMPARE:
        wrong = [f for f, v in (("--arms", a.arms), ("--exp0052-episodes", a.exp0052_episodes),
                                ("--exp0056-episodes", a.exp0056_episodes)) if v is not None]
        if wrong or a.arm is None:
            p.error(f"--mode arm-compare takes --arm and --exp0057-episodes, not {wrong}"
                    if wrong else "--mode arm-compare needs --arm")
    else:
        wrong = [f for f, v in (("--arm", a.arm), ("--exp0057-episodes", a.exp0057_episodes))
                 if v is not None]
        if wrong:
            p.error(f"{wrong} belong to --mode arm-compare")
        if a.orderer_ckpt is None:
            p.error("the following arguments are required: --orderer-ckpt")
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    if a.mode == ARM_COMPARE:
        try:
            return run_arm_compare(
                atlas_path=a.atlas, out_dir=a.out_dir, arm=a.arm,
                exp0057_episodes=a.exp0057_episodes, orderer_ckpt=a.orderer_ckpt,
                roster=load_roster(a.roster), workers=a.workers, horizon=a.horizon,
                worlds=_csv_strs(a.worlds), max_worlds=a.max_worlds,
                shock_profile=a.shock_profile, source_commit=commit, dry_run=a.dry_run)
        except (ContractRunError, DigestMismatch):  # recorded in contract_failures.csv
            return CONTRACT_INVALID
    try:
        return run_decoder_check(
            atlas_path=a.atlas, out_dir=a.out_dir, orderer_ckpt=a.orderer_ckpt,
            roster=load_roster(a.roster), exp0052_episodes=a.exp0052_episodes,
            exp0056_episodes=a.exp0056_episodes, workers=a.workers, horizon=a.horizon,
            worlds=_csv_strs(a.worlds), max_worlds=a.max_worlds, arms=_csv_strs(a.arms),
            shock_profile=a.shock_profile, source_commit=commit, dry_run=a.dry_run)
    except (ContractRunError, DigestMismatch):  # recorded in contract_failures.csv
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
