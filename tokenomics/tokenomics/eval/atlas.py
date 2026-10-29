"""Headroom atlas driver: `python -m tokenomics.eval.atlas` (add-m3-development-audit design D12,
ADR-0057; development-audit "Per-world headroom" ... "Atlas evidence does not change the gate").

Two modes, one EXP directory each:
- `--mode screen` (EXP-0051): a world-space Latin hypercube sample per size (`world_space`,
  512 worlds at 12 producers, 128 at 42), 1 episode per world, plus the 8 EXP-0050 anchor
  world-episodes (world 0 of each fanout cell, its first shocked episode, rebuilt by
  `routing_audit.fanout_world`). Arms: the blind and hand-written arms (the roster v2 members
  with `random` at R = 1, `round_robin`, `random_whole` at R = 2) and the reach arms without
  `portable` (`makeable_split`, `makeable_weighted_whole` at R = 1,
  `entity_planner/trained_orderer`). Every world gets a provisional class (noise floor from
  EXP-0050's spreads, 2 x SE), then the deterministic selection and the split are recorded
  (`selection.csv`, `coverage.csv`).
- `--mode confirm` (EXP-0052): the screen's selected worlds on fresh episode and replicate seeds
  (`world_space.episode_seeds(..., stage="confirm")`; replicate seeds hash the episode seed),
  2 episodes per world (1 for cyclic demand without shocks), `random`, `random_whole` and
  `makeable_weighted_whole` at R = 2, `portable` added (reported, never an argmax). The bar and
  reach argmax arms are fixed from the screen; gain, room and the pooled noise floor (3 x SE)
  are re-estimated and every world reclassified (task 5.3 rules: the spreads are the
  normal-consistent median absolute residual, `mad_sigma`, and a world whose fixed bar or reach
  arm dies is collapse). Writes `deciding_check.csv` (EXP-0050's deciding check) and freezes
  `atlas_v2.json` (content hash) into its own out dir only, never into the package registry.
- `--mode reclassify --confirm-dir <confirm> --out-dir <derived>` (task 5.3): a recorded
  confirm's rows classified again under the current rules, offline, into a derived directory
  (`worlds.csv` with the recorded `class_v1`, `deciding_check.csv`, `atlas_v2.json`,
  `reclassify.json`); the confirm's own files are never written. EXP-0052's `atlas_v1.json`
  was frozen under the pre-registered pooled SD without the dead-arm rule.

Jobs run through `routing_audit`'s job path (tracer and contract monitor on, the gate's
replicate seeds, demand/shock digest pairing per world-episode). Each row also carries the
restored bounds of `ceiling_bound` for its world-episode (`bound_b2`, `bound_base`, from the
demand schedule the env built for the episode seed). B2 >= every played arm's units (tolerance
`BOUND_TOLERANCE`) is an invariant: a violation is a `run` row naming world, episode and arm, and
the run is invalid.

The analysis is pure and pre-registered in D12 (`measures`, `pooled_spreads`, `noise_floor`,
`classify`, `select`, `split_worlds`): bar = best blind arm, reach = best reach arm, gain from the
paired per-episode differences of the two argmax arms, ceiling C = max(min(B2, B_base), best
played arm), room = (C - bar) / C; classes collapse > saturated > hard > open > flat. One driver
choice, recorded in `spec.json`: when both argmax arms are stochastic, the replicate term of
SE_w is sigma_rep^2 / R per stochastic arm (D12 writes one term, for one stochastic arm), and
"positive in each replicate" pairs replicates by index (replicate seeds are shared).

A real run needs the frozen roster, both checkpoints, a 40-hex source commit, horizon 100, every
arm, the stage's R, the full sample and no test-only shock window; the screen also needs EXP-0050's
`episodes.csv` (`--exp0050-episodes`: provisional spreads + anchor reproduction); a real confirm
needs a complete real screen. Anything else needs `--dry-run` (marked "not an acceptance read").
The atlas never writes registry, roster or parameter files (development-audit "Atlas evidence
does not change the gate").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Iterable, Mapping, Sequence

from .ceiling_bound import (
    B_BASE,
    episode_cycle_demand,
    episode_demand,
    lp_available,
    lp_bound,
    world_ceiling,
)
from .contract import (
    RUN,
    ContractRunError,
    Failure,
    append_failure,
    ensure_failures_file,
    is_failure,
    job_failure,
    read_failures,
)
from .families import CALM, DENSE, SHOCKED, SHOCKS_I_WINDOW
from .gate import DigestMismatch
from .gate_cli import (
    CONTRACT_INVALID,
    INVALID,
    Job,
    _open_spec,
    _prepare_resume,
    _write_spec,
    check_rows,
    job_grid,
    job_key,
    read_rows,
    test_only_problem,
)
from .provenance import resolve_verified_commit, sha256_file, source_fingerprint
from .registry import DEFAULT_SATURATION_CEILING, WorldEntry
from .roster import DEFAULT_ROSTER, JOINT_STACK, Roster, load_roster, member_id
from .routing_audit import (
    EPISODE_COLUMNS,
    FANOUT_CELLS,
    FANOUT_EPISODES,
    PORTABLE,
    PRODUCER,
    TRAINED_ORDERER,
    _is_commit,
    _trained_entry,
    build_jobs,
    checked_checkpoints,
    execute,
    expected_keys,
    fanout_world,
    job_config,
    read_episode_dicts,
    run_job,
    write_csv,
)
from .routing_audit import world_record as fanout_record
from .routing_diagnostics import DETERMINISTIC, ROUTER_KIND, STOCHASTIC
from .shock_profiles import SHOCKS_I, SHOCKS_I_SMOKE, SMOKE_START_WINDOW
from .world_space import (
    CHOICES,
    CYCLIC,
    PATTERNS,
    SIZES,
    WorldPoint,
    episode_seeds,
    point_from_record,
    sample_worlds,
)
from .world_space import config_from_record
from .world_space import world_record as space_record
from ..domain.config import PATTERN_SWITCH

DRIVER = "tokenomics.eval.atlas"
SCREEN = "screen"
CONFIRM = "confirm"
RECLASSIFY = "reclassify"  # offline: a recorded confirm's rows under the current rules (5.3)
MODES = (SCREEN, CONFIRM)
ATLAS = "atlas-v1"  # world seeds and ids hash this name (world_space.world_seed)
ATLAS_FILE = "atlas_v2.json"  # the confirm rules of task 5.3; EXP-0052's atlas_v1.json predates
ATLAS_FORMAT = "atlas-v2"
HORIZON = 100
# D12 planned 1,024 at 12; task 4.6's probe put that at 10.8 pod vCPU-h, above the 10 stop,
# so D12's reduction step 1 applies (sizing.md §EXP-0051): a 512-point Latin hypercube at 12.
SCREEN_WORLDS = {12: 512, 42: 128}
K = {12: 64, 42: 32}
OPEN_SHARE = 0.25  # at least K/4 open worlds when that many exist
SCREEN_EPISODES = 1
CONFIRM_EPISODES = 2
CONFIRM_EPISODES_EXACT = 1  # cyclic demand, no shocks: every episode replays identically
CONFIRM_STAGE = "confirm"
SPLIT_FAMILY = "world_space"
ANCHOR_SPLIT = "atlas-anchor"
SCREEN_SPLIT = "atlas-screen"

# arm sides (D12 "Arms per world")
BLIND = "blind"
REACH = "reach"
REPORTED = "reported"  # portable: confirm only, never an argmax
BLIND_DIAGNOSTIC = ("round_robin", "random_whole")
REACH_DIAGNOSTIC = ("makeable_split", "makeable_weighted_whole")
REPLICATES: dict[str, dict[str, int]] = {
    SCREEN: {"random": 1, "random_whole": 2, "makeable_weighted_whole": 1},
    CONFIRM: {"random": 2, "random_whole": 2, "makeable_weighted_whole": 2},
}

# pre-registered thresholds (D12 "Bound and ceiling", "Measures per world", "Classes")
BOUND_TOLERANCE = 0.01
SATURATION_FILL = DEFAULT_SATURATION_CEILING  # 0.95, the registry's saturation ceiling
SATURATION_SHARE = 0.97  # bar >= 0.97 x C
OPEN_ROOM = 0.10
COLLAPSE_CYCLE = 50
FLOOR_SE = {SCREEN: 2.0, CONFIRM: 3.0}
FLOOR_BAR_SHARE = 0.01
MIN_POOLED_DF = 30
# spread estimators (D12 "noise floor", task 5.3): the screen keeps EXP-0050's pooled SD as
# recorded in EXP-0051; the confirm's own spreads use the normal-consistent median absolute
# residual, robust to the few heavy-tailed worlds that carried EXP-0052's sums of squares
POOLED_SD = "pooled_sd"
POOLED_MAD = "pooled_mad"
SPREAD_ESTIMATOR = {SCREEN: POOLED_SD, CONFIRM: POOLED_MAD}
NORMAL_MAD = 0.6744897501960817  # Phi^-1(0.75): median |X| of a standard normal
DECIDING_SHARE = 2 / 3
DECIDING_MIN_CHOICE = 4  # hard worlds with E >= 4 producers per product
SPLIT_SHARES = (0.2, 0.2)  # held-out, validation; the rest train

COLLAPSE = "collapse"
SATURATED = "saturated"
HARD = "hard"
OPEN = "open"
FLAT = "flat"
CLASSES = (COLLAPSE, SATURATED, HARD, OPEN, FLAT)

HELDOUT_REGION = "held-out-region"
HELDOUT = "held-out"
VALIDATION = "validation"
TRAIN = "train"

COORDINATES = ("rho", "sigma", "cv_capacity", "cv_degradation", "cv_cost",
               "shock_events_per_producer", "demand_concentration", "demand_volatility")
CATEGORICAL_AXES = ("choice", "shocks", "pattern")
BOUND_CHECK = "bound_below_arm"
ANCHOR_CHECK = "anchor_reproduction"
BOUND_COLUMNS = ("bound_b2", "bound_base", "bound_b2_binding", "bound_base_binding",
                 "bound_lp", "bound_lp_base", "bound_lp_binding", "bound_lp_base_binding")
LP_JOB = "solve_lp"  # job flag: this job also solves its world-episode's B_lp (task 4.7)
ATLAS_COLUMNS = EPISODE_COLUMNS + BOUND_COLUMNS
TTL_PREFIX = "trace_ttl_lost_units_"

Stratum = tuple[int, str]


# --- arms ---


def arm_table(roster: Roster, mode: str, arms: Sequence[str] | None = None,
              replicates: int | None = None) -> list[dict]:
    """Every arm of `mode` in order (D12): the roster members and the blind diagnostic routers
    (`blind`), the reach routers and the trained orderer (`reach`), and in confirm `portable`
    (`reported`). Each entry has routing_audit's router fields plus `side` and, for a
    stochastic arm, its R (`replicates` overrides every R: a dry run). `arms` keeps a subset
    by router name."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {MODES}")

    def r_of(name: str) -> int:
        return replicates if replicates is not None else REPLICATES[mode].get(name, 1)

    table = [{"router": m.ordering, "subject": m.id, "producing": m.producing,
              "ordering": m.ordering, "contract": m.contract, "random": m.random,
              "roster": True, "kind": STOCHASTIC if m.random else DETERMINISTIC,
              "trained": False, "side": BLIND}
             for m in roster.members]
    for side, names in ((BLIND, BLIND_DIAGNOSTIC), (REACH, REACH_DIAGNOSTIC)):
        table += [{"router": n, "subject": member_id(PRODUCER, n), "producing": PRODUCER,
                   "ordering": n, "contract": JOINT_STACK,
                   "random": ROUTER_KIND[n] == STOCHASTIC, "roster": False,
                   "kind": ROUTER_KIND[n], "trained": False, "side": side} for n in names]
    table.append({**_trained_entry(TRAINED_ORDERER), "side": REACH})
    if mode == CONFIRM:
        table.append({**_trained_entry(PORTABLE), "side": REPORTED})
    for a in table:
        if a["random"]:
            a["replicates"] = r_of(a["router"])
    if arms is not None:
        known = {a["router"] for a in table}
        unknown = sorted(set(arms) - known)
        if unknown:
            raise ValueError(f"unknown arms {unknown}; known: {sorted(known)}")
        table = [a for a in table if a["router"] in set(arms)]
    return table


# --- worlds ---


@dataclass(frozen=True)
class WorldInfo:
    """What the analysis needs to know about a world besides its rows."""

    id: str
    size: int
    choice: str
    pattern: str
    shocks: bool
    differing: bool  # its episodes can differ (stochastic demand or shocks)
    anchor: bool = False
    record: Mapping | None = None

    @property
    def stratum(self) -> Stratum:
        return (self.size, self.choice)

    @property
    def eligible(self) -> int:
        """Producers per product E (D10 "Choice"): dense = the whole layer."""
        return self.size // 3 if self.choice == DENSE else int(self.choice.removeprefix("sparse_f"))


def space_entry(point: WorldPoint, episodes: Sequence[int], split: str = SCREEN_SPLIT
                ) -> WorldEntry:
    """A world-space world as a job-path `WorldEntry` (generator `world_space`)."""
    return WorldEntry(
        id=point.world_id, split=split, family=SPLIT_FAMILY, size=point.size,
        regime=SHOCKED if point.levers.shocks else CALM, world_seed=point.seed,
        episode_seeds=tuple(episodes),
        generator={"generator": "world_space",
                   "params": {"producers": point.size, "levers": point.levers.to_dict(),
                              "shock_window": list(point.shock_window)}})


def space_info(record: Mapping) -> WorldInfo:
    lv = record["levers"]
    return WorldInfo(id=record["world_id"], size=record["size"], choice=lv["choice"],
                     pattern=lv["pattern"], shocks=bool(lv["shocks"]),
                     differing=not (lv["pattern"] == CYCLIC and not lv["shocks"]),
                     record=record)


def anchor_worlds(shock_profile: str = SHOCKS_I, sizes: Sequence[int] | None = None
                  ) -> list[WorldEntry]:
    """The 8 EXP-0050 anchors (D12 "Screen"): world 0 of each fanout cell, shocked, its first
    episode. Ids and seeds are EXP-0050's, so rows pair with its `episodes.csv`."""
    out = []
    for family, producers, fanout in FANOUT_CELLS:
        if sizes and producers not in sizes:
            continue
        e = fanout_world(family, producers, fanout, SHOCKED, 0, FANOUT_EPISODES[SHOCKED],
                         shock_profile)
        out.append(replace(e, episode_seeds=e.episode_seeds[:1], split=ANCHOR_SPLIT))
    return out


def anchor_info(e: WorldEntry) -> WorldInfo:
    fanout = e.generator["params"]["fanout"]
    return WorldInfo(id=e.id, size=e.size, choice=DENSE if e.family == DENSE else
                     f"sparse_f{fanout}", pattern="anchor", shocks=True, differing=True,
                     anchor=True)


def screen_points(sizes: Sequence[int], worlds_per_size: int | None,
                  shock_window: Sequence[int]) -> list[WorldPoint]:
    """The screen sample: the first `worlds_per_size` (dry run) of each size's full sample, so a
    dry run's worlds are the real run's."""
    out: list[WorldPoint] = []
    for size in sizes:
        pts = sample_worlds(ATLAS, size, SCREEN_WORLDS[size], shock_window)
        out += pts[:worlds_per_size] if worlds_per_size is not None else pts
    return out


def confirm_episode_count(info: WorldInfo) -> int:
    return CONFIRM_EPISODES if info.differing else CONFIRM_EPISODES_EXACT


# --- worker: the routing_audit job plus the world-episode's bounds ---

_BOUNDS: dict[tuple, dict[str, str]] = {}


def bound_cells(job: Job) -> dict[str, str]:
    """B2 and B_base of the job's world-episode (cached per process); on the one job per
    world-episode flagged `LP_JOB`, also B_lp and B_lp_base (task 4.7: an LP is 0.1-10 s, so it
    is solved once per world-episode, not once per arm; the other rows leave the cells empty)."""
    key = (job["world"], job["world_seed"], json.dumps(job["generator"], sort_keys=True),
           job["episode"], job["horizon"])
    if key not in _BOUNDS:
        cfg = job_config(job)
        b = world_ceiling(cfg, job["horizon"], episode_demand(cfg, job["horizon"], job["episode"]))
        _BOUNDS[key] = {"bound_b2": repr(float(b["B2"].units)),
                        "bound_base": repr(float(b[B_BASE].units)),
                        "bound_b2_binding": b["B2"].binding_constraint,
                        "bound_base_binding": b[B_BASE].binding_constraint}
    cells = dict(_BOUNDS[key])
    if job.get(LP_JOB):
        cfg = job_config(job)
        demand = episode_cycle_demand(cfg, job["horizon"], job["episode"])
        for tier, bound in (("bound_lp", lp_bound(cfg, job["horizon"], demand)),
                            ("bound_lp_base", lp_bound(cfg, job["horizon"], demand,
                                                       upgrades=False))):
            cells[tier] = repr(float(bound.units))
            cells[f"{tier}_binding"] = bound.binding_constraint
    return cells


def flag_lp_jobs(jobs: Sequence[Job]) -> None:
    """Flag the first job of every world-episode (`LP_JOB`), in job order."""
    seen: set[tuple[str, int]] = set()
    for j in jobs:
        k = (j["world"], int(j["episode"]))
        if k not in seen:
            seen.add(k)
            j[LP_JOB] = True


def run_atlas_job(job: Job) -> dict[str, str]:
    """`routing_audit.run_job`'s row plus the bound columns; a failure stays a failure row."""
    row = run_job(job)
    if is_failure(row):
        return row
    try:
        row.update(bound_cells(job))
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()
    return row


# --- pure analysis ---


def _num(s: object) -> float | None:
    if s is None or s in ("", "None"):
        return None
    return float(s)  # type: ignore[arg-type]


@dataclass
class ArmEpisode:
    """One arm on one world-episode, by replicate."""

    units: list[float] = field(default_factory=list)
    fills: list[float | None] = field(default_factory=list)
    collapsed: list[bool] = field(default_factory=list)
    dead: list[bool] = field(default_factory=list)  # system death at any cycle
    ttl_short: float = 0.0
    ttl_ok: float = 0.0

    @property
    def mean(self) -> float:
        return fmean(self.units)


@dataclass
class Episode:
    b2: float | None = None
    base: float | None = None
    lp: float | None = None  # B_lp (guaranteed; task 4.7), from the world-episode's LP row
    lp_base: float | None = None  # B_lp_base (capacity-aware, not guaranteed)
    arms: dict[str, ArmEpisode] = field(default_factory=dict)


def _cycles(d: Mapping[str, str]) -> int | None:
    trace = d.get("trace_capacity_ratio_trace")
    return len(json.loads(trace)) if trace else None


def episode_values(dicts: Iterable[Mapping[str, str]]) -> dict[str, dict[int, Episode]]:
    """world -> episode -> Episode (bounds and every arm's replicate values, replicate order)."""
    out: dict[str, dict[int, Episode]] = defaultdict(dict)
    for d in sorted(dicts, key=lambda d: (d["world"], int(d["episode"]), d["subject"],
                                          int(d["replicate"]))):
        ep = out[d["world"]].setdefault(int(d["episode"]), Episode())
        ep.b2 = _num(d.get("bound_b2")) if ep.b2 is None else ep.b2
        ep.base = _num(d.get("bound_base")) if ep.base is None else ep.base
        ep.lp = _num(d.get("bound_lp")) if ep.lp is None else ep.lp
        ep.lp_base = _num(d.get("bound_lp_base")) if ep.lp_base is None else ep.lp_base
        a = ep.arms.setdefault(d["subject"], ArmEpisode())
        a.units.append(float(d["units"]))
        ordered, fulfilled = _num(d.get("ordered_units")), _num(d.get("fulfilled_units"))
        a.fills.append(fulfilled / ordered if ordered and fulfilled is not None else None)
        cycles = _cycles(d)
        dead = str(d["dead"]).lower() == "true"
        a.dead.append(dead)
        a.collapsed.append(dead and cycles is not None and cycles < COLLAPSE_CYCLE)
        for k, v in d.items():
            if k.startswith(TTL_PREFIX) and v not in ("", None):
                if k.endswith("_inputs_short"):
                    a.ttl_short += float(v)
                elif k.endswith("_inputs_ok"):
                    a.ttl_ok += float(v)
    return dict(out)


@dataclass
class Measures:
    """D12's per-world measures (equal-weight means over the world's episodes)."""

    world: str
    episodes: int
    bar_arm: str | None
    bar: float | None
    reach_arm: str | None
    reach: float | None
    gain: float | None
    d: list[float]  # paired per-episode gain (replicate means)
    d_reps: list[list[float]]  # per episode, per replicate (paired by index)
    bar_fill: float | None
    b2: float | None
    base: float | None
    ceiling: float
    room: float
    collapse: bool
    above_base: dict[str, int]  # arm -> world-episodes on which it exceeded B_base
    stochastic: dict[str, int]  # stochastic argmax arm -> its R (min over episodes)
    lp: float | None = None  # mean B_lp (task 4.7)
    lp_base: float | None = None  # mean B_lp_base
    rep_ss: float = 0.0  # sum of squared replicate deviations of the stochastic argmax arms
    rep_df: int = 0
    rep_resid: list[float] = field(default_factory=list)  # their scaled residuals (residuals())
    argmax_died: bool = False  # the bar or reach argmax arm died in some episode / replicate
    arm_means: dict[str, float] = field(default_factory=dict)


def _argmax(means: Mapping[str, float], order: Sequence[str]) -> str | None:
    best = None
    for s in order:
        if s in means and (best is None or means[s] > means[best]):
            best = s
    return best


def measures(world: str, episodes: Mapping[int, Episode], arms: Sequence[Mapping],
             fixed: tuple[str | None, str | None] | None = None) -> Measures:
    """bar, reach, gain, ceiling, room and the inputs of the noise floor (D12). `fixed` = the
    (bar, reach) argmax arms from the screen (confirm), else they are this run's argmax."""
    eps = sorted(episodes)
    present = [a for a in arms if all(a["subject"] in episodes[e].arms for e in eps)]
    means = {a["subject"]: fmean(episodes[e].arms[a["subject"]].mean for e in eps)
             for a in present}
    order = [a["subject"] for a in present]
    side = {a["subject"]: a["side"] for a in present}
    stoch = {a["subject"]: bool(a["random"]) for a in present}
    if fixed is not None:
        bar_arm = fixed[0] if fixed[0] in means else None
        reach_arm = fixed[1] if fixed[1] in means else None
    else:
        bar_arm = _argmax({s: v for s, v in means.items() if side[s] == BLIND}, order)
        reach_arm = _argmax({s: v for s, v in means.items() if side[s] == REACH}, order)
    bar = means.get(bar_arm) if bar_arm else None
    reach = means.get(reach_arm) if reach_arm else None
    d: list[float] = []
    d_reps: list[list[float]] = []
    stochastic: dict[str, int] = {}
    rep_ss, rep_df = 0.0, 0
    rep_resid: list[float] = []
    if bar_arm and reach_arm:
        for e in eps:
            r, b = episodes[e].arms[reach_arm], episodes[e].arms[bar_arm]
            d.append(r.mean - b.mean)
            n = max(len(r.units), len(b.units))
            d_reps.append([r.units[i % len(r.units)] - b.units[i % len(b.units)]
                           for i in range(n)])
        for s in (bar_arm, reach_arm):
            if not stoch[s]:
                continue
            stochastic[s] = min(len(episodes[e].arms[s].units) for e in eps)
            for e in eps:
                u = episodes[e].arms[s].units
                if len(u) >= 2:
                    m = fmean(u)
                    rep_ss += sum((x - m) ** 2 for x in u)
                    rep_df += len(u) - 1
                    rep_resid += residuals(u)
    ceilings, fills = [], []
    above: dict[str, int] = defaultdict(int)
    for e in eps:
        ep = episodes[e]
        best = max(x for a in ep.arms.values() for x in a.units)
        # the base-capacity ceiling is B_lp_base when the LP ran (task 4.7), else B_base
        cap = ep.lp_base if ep.lp_base is not None else ep.base
        bounds = [b for b in (ep.b2, ep.lp, cap) if b is not None]
        ceilings.append(max(min(bounds), best) if bounds else best)
        if cap is not None:
            for s, a in ep.arms.items():
                if max(a.units) > cap + BOUND_TOLERANCE:
                    above[s] += 1
        if bar_arm:
            fills += [f for f in ep.arms[bar_arm].fills if f is not None]
    ceiling = fmean(ceilings) if ceilings else 0.0
    room = (ceiling - bar) / ceiling if bar is not None and ceiling > 0 else 0.0
    b2s = [episodes[e].b2 for e in eps if episodes[e].b2 is not None]
    bases = [episodes[e].base for e in eps if episodes[e].base is not None]
    lps = [episodes[e].lp for e in eps if episodes[e].lp is not None]
    lp_bases = [episodes[e].lp_base for e in eps if episodes[e].lp_base is not None]
    collapse = bool(eps) and all(c for e in eps for a in episodes[e].arms.values()
                                 for c in a.collapsed)
    died = any(x for s in (bar_arm, reach_arm) if s for e in eps
               for x in episodes[e].arms[s].dead)
    return Measures(
        world=world, episodes=len(eps), bar_arm=bar_arm, bar=bar, reach_arm=reach_arm,
        reach=reach, gain=fmean(d) if d else None, d=d, d_reps=d_reps,
        bar_fill=fmean(fills) if fills else None, b2=fmean(b2s) if b2s else None,
        base=fmean(bases) if bases else None, ceiling=ceiling, room=room, collapse=collapse,
        above_base=dict(above), stochastic=stochastic, lp=fmean(lps) if lps else None,
        lp_base=fmean(lp_bases) if lp_bases else None, rep_ss=rep_ss, rep_df=rep_df,
        rep_resid=rep_resid, argmax_died=died, arm_means=means)


@dataclass(frozen=True)
class Spread:
    """Pooled spreads of one stratum (D12 "noise floor"); `merged_*` = pooled within size
    because the stratum had fewer than `MIN_POOLED_DF` degrees of freedom. `sigma_*` are the
    `estimator`'s and enter the floor; `sd_*` are the pooled SDs of the same pools, reported."""

    sigma_ep: float | None
    df_ep: int
    sigma_rep: float | None
    df_rep: int
    merged_ep: bool = False
    merged_rep: bool = False
    estimator: str = POOLED_SD
    sd_ep: float | None = None
    sd_rep: float | None = None

    def to_dict(self) -> dict:
        return {"sigma_ep": self.sigma_ep, "df_ep": self.df_ep, "sigma_rep": self.sigma_rep,
                "df_rep": self.df_rep, "merged_ep": self.merged_ep,
                "merged_rep": self.merged_rep, "estimator": self.estimator,
                "sd_ep": self.sd_ep, "sd_rep": self.sd_rep}

    @classmethod
    def from_dict(cls, d: Mapping) -> Spread:
        return cls(**{k: d[k] for k in ("sigma_ep", "df_ep", "sigma_rep", "df_rep",
                                         "merged_ep", "merged_rep")},
                   **{k: d[k] for k in ("estimator", "sd_ep", "sd_rep") if k in d})


def _sigma(ss: float, df: int) -> float | None:
    return math.sqrt(ss / df) if df > 0 else None


def residuals(xs: Sequence[float]) -> list[float]:
    """(x - mean) * sqrt(n / (n - 1)) per value: each has the variance of one value, so a pool of
    them estimates the within-group sigma (n = 2: +-(x1 - x2) / sqrt 2)."""
    n = len(xs)
    if n < 2:
        return []
    m, k = fmean(xs), math.sqrt(n / (n - 1))
    return [(x - m) * k for x in xs]


def mad_sigma(resid: Sequence[float]) -> float | None:
    """Normal-consistent median absolute residual: median |r| / Phi^-1(0.75). Its breakdown
    point is 50%, so fewer than half the residuals, however large, cannot move it past the
    others (task 5.3); at the normal it is about 37% efficient against the pooled SD."""
    return median(abs(r) for r in resid) / NORMAL_MAD if resid else None


def pooled_spreads(ms: Sequence[Measures], infos: Mapping[str, WorldInfo], *,
                   merge_below: int = MIN_POOLED_DF,
                   estimator: str = POOLED_MAD) -> dict[Stratum, Spread]:
    """sigma_ep(s): pooled within-world spread of the paired per-episode gain over the worlds of
    stratum s = size x choice with >= 2 differing episodes, df = sum(n_w - 1); sigma_rep(s):
    pooled within-episode spread between the replicates of the stochastic argmax arms. A stratum
    below `merge_below` df takes its size's pool (recorded): `MIN_POOLED_DF` in confirm (D12);
    the screen's EXP-0050 spreads stay per size x fanout and merge only with no df at all.
    `estimator`: `POOLED_MAD` (the confirm's, task 5.3) = `mad_sigma` of the pool's residuals;
    `POOLED_SD` (the screen's, as recorded in EXP-0051) = sqrt(sum of squares / df)."""
    if estimator not in (POOLED_SD, POOLED_MAD):
        raise ValueError(f"unknown spread estimator {estimator!r}")
    ep: dict[Stratum, list] = {}
    rep: dict[Stratum, list] = {}
    for m in ms:
        info = infos[m.world]
        s = info.stratum
        e_, r_ = ep.setdefault(s, [0.0, 0, []]), rep.setdefault(s, [0.0, 0, []])
        if info.differing and len(m.d) >= 2:
            mean = fmean(m.d)
            e_[0] += sum((x - mean) ** 2 for x in m.d)
            e_[1] += len(m.d) - 1
            e_[2] += residuals(m.d)
        r_[0] += m.rep_ss
        r_[1] += m.rep_df
        r_[2] += m.rep_resid

    def by_size(pool: Mapping[Stratum, list], size: int) -> list:
        cells = [v for k, v in pool.items() if k[0] == size]
        return [sum(v[0] for v in cells), int(sum(v[1] for v in cells)),
                [x for v in cells for x in v[2]]]

    def spread(cell: list) -> tuple[float | None, float | None]:
        sd = _sigma(cell[0], int(cell[1]))
        return (sd if estimator == POOLED_SD or sd is None else mad_sigma(cell[2])), sd

    out: dict[Stratum, Spread] = {}
    for s in sorted(set(ep) | set(rep)):
        e_, r_ = ep[s], rep[s]
        merged_ep = e_[1] < merge_below
        merged_rep = r_[1] < merge_below
        if merged_ep:
            e_ = by_size(ep, s[0])
        if merged_rep:
            r_ = by_size(rep, s[0])
        (se_, sde), (sr_, sdr) = spread(e_), spread(r_)
        out[s] = Spread(se_, int(e_[1]), sr_, int(r_[1]), merged_ep, merged_rep, estimator,
                        sde, sdr)
    return out


def standard_error(n: int, differing: bool, stochastic: Mapping[str, int],
                   spread: Spread | None) -> float:
    """SE_w = sqrt(sigma_ep^2 [differing] + sum over stochastic argmax arms of
    sigma_rep^2 / R) / sqrt(n_w); infinite when a needed spread is unknown (no pooled df), so
    such a world is never hard."""
    var = 0.0
    if differing:
        if spread is None or spread.sigma_ep is None:
            return math.inf
        var += spread.sigma_ep ** 2
    for r in stochastic.values():
        if spread is None or spread.sigma_rep is None:
            return math.inf
        var += spread.sigma_rep ** 2 / r
    return math.sqrt(var) / math.sqrt(n) if n else math.inf


def noise_floor(m: Measures, info: WorldInfo, spread: Spread | None,
                multiplier: float) -> tuple[float, float]:
    """(SE_w, floor_w = max(multiplier x SE_w, 0.01 x bar))."""
    se = standard_error(m.episodes, info.differing, m.stochastic, spread)
    return se, max(multiplier * se, FLOOR_BAR_SHARE * (m.bar or 0.0))


def sign_consistent(m: Measures) -> bool:
    """The gain is positive in every episode and every replicate pairing."""
    return bool(m.d_reps) and all(x > 0 for reps in m.d_reps for x in reps)


def classify(m: Measures, floor: float, *, dead_arm: bool = False) -> str:
    """D12 classes, in order: collapse, saturated, hard, open, flat. `dead_arm` (the confirm's
    rule, task 5.3): a world whose fixed bar or reach arm reaches system death in any episode or
    replicate is collapse, so a gain over a dying reference arm is never hard."""
    if m.collapse or (dead_arm and m.argmax_died):
        return COLLAPSE
    if (m.bar_fill is not None and m.bar_fill >= SATURATION_FILL) or (
            m.bar is not None and m.bar >= SATURATION_SHARE * m.ceiling):
        return SATURATED
    if m.gain is not None and m.gain > floor and sign_consistent(m):
        return HARD
    if m.room >= OPEN_ROOM:
        return OPEN
    return FLAT


def bound_violations(dicts: Iterable[Mapping[str, str]]) -> list[Failure]:
    """B2 and, where the LP ran, B_lp >= every played arm's units (tolerance
    `BOUND_TOLERANCE`): one `run` row per row above either, naming world, episode and arm."""
    dicts = list(dicts)
    out = []
    for d in dicts:
        b2 = _num(d.get("bound_b2"))
        if b2 is not None and float(d["units"]) > b2 + BOUND_TOLERANCE:
            out.append(Failure(
                scope=RUN, check=BOUND_CHECK, error_type="BoundBelowArm",
                message=(f"world {d['world']} episode {d['episode']}: arm {d['subject']} "
                         f"replicate {d['replicate']} delivered {float(d['units']):g} units, "
                         f"above B2 = {b2:g}"),
                world=d["world"], episode=int(d["episode"]), subject=d["subject"],
                replicate=int(d["replicate"])))
    lp: dict[tuple[str, int], float] = {}
    for d in dicts:
        v = _num(d.get("bound_lp"))
        if v is not None:
            lp.setdefault((d["world"], int(d["episode"])), v)
    for d in dicts:
        v = lp.get((d["world"], int(d["episode"])))
        if v is not None and float(d["units"]) > v + BOUND_TOLERANCE:
            out.append(Failure(
                scope=RUN, check=BOUND_CHECK, error_type="BoundBelowArm",
                message=(f"world {d['world']} episode {d['episode']}: arm {d['subject']} "
                         f"replicate {d['replicate']} delivered {float(d['units']):g} units, "
                         f"above B_lp = {v:g}"),
                world=d["world"], episode=int(d["episode"]), subject=d["subject"],
                replicate=int(d["replicate"])))
    return out


# --- selection and split ---


@dataclass(frozen=True)
class Bin:
    size: int
    axis: str
    label: str
    lo: float | None = None
    hi: float | None = None

    def holds(self, record: Mapping) -> bool:
        if self.axis in CATEGORICAL_AXES:
            return str(record["levers"][self.axis]) == self.label
        v = record["coordinates"][self.axis]
        return (self.lo is None or v > self.lo) and (self.hi is None or v <= self.hi)


def quartile_cuts(values: Sequence[float]) -> list[float]:
    """Nearest-rank quartile cut points (25/50/75%)."""
    v = sorted(values)
    return [v[max(0, math.ceil(q * len(v)) - 1)] for q in (0.25, 0.5, 0.75)] if v else []


def coverage_bins(size: int, records: Sequence[Mapping]) -> list[Bin]:
    """The coverage bins of one size in their fixed order: every level of every categorical
    axis, then the quartiles of every measured coordinate over the size's screened worlds."""
    bins = [Bin(size, "choice", c) for c in CHOICES[size]]
    bins += [Bin(size, "shocks", str(b)) for b in (False, True)]
    bins += [Bin(size, "pattern", p) for p in PATTERNS]
    for coord in COORDINATES:
        cuts = quartile_cuts([r["coordinates"][coord] for r in records])
        if not cuts:
            continue
        edges = [None, *cuts, None]
        bins += [Bin(size, coord, f"Q{i + 1}", edges[i], edges[i + 1]) for i in range(4)]
    return bins


@dataclass(frozen=True)
class Candidate:
    world: str
    cls: str
    gain: float | None
    bar: float | None
    room: float
    record: Mapping


def score_key(c: Candidate) -> tuple:
    """Hard worlds by gain / bar, then open worlds by room (both descending); ties by id."""
    if c.cls == HARD:
        ratio = (c.gain or 0.0) / c.bar if c.bar else math.inf
        return (0, -ratio, c.world)
    return (1, -c.room, c.world)


def select(candidates: Sequence[Candidate], k: int, bins: Sequence[Bin]
           ) -> list[tuple[Candidate, str]]:
    """The deterministic selection (D12): pass 1 adds, for each uncovered bin in order, its
    best-scored candidate; pass 2 fills to k by score, keeping room for at least k/4 open
    worlds when that many exist. Only hard and open worlds are candidates. Returns
    (candidate, pass) in selection order."""
    pool = sorted((c for c in candidates if c.cls in (HARD, OPEN)), key=score_key)
    chosen: list[tuple[Candidate, str]] = []
    ids: set[str] = set()
    for b in bins:
        if len(chosen) >= k:
            break
        if any(b.holds(c.record) for c, _ in chosen):
            continue
        for c in pool:
            if c.world not in ids and b.holds(c.record):
                chosen.append((c, f"coverage:{b.axis}={b.label}"))
                ids.add(c.world)
                break
    need = min(int(k * OPEN_SHARE), sum(c.cls == OPEN for c in pool))
    for c in pool:
        if len(chosen) >= k:
            break
        if c.world in ids:
            continue
        open_now = sum(x.cls == OPEN for x, _ in chosen)
        if c.cls == HARD and k - len(chosen) <= need - open_now:
            continue
        chosen.append((c, "score"))
        ids.add(c.world)
    return chosen


def coverage_rows(bins: Sequence[Bin], records: Sequence[Mapping],
                  candidates: Sequence[Candidate], chosen: Sequence[str]) -> list[dict]:
    picked = set(chosen)
    sel = {c.world for c in candidates if c.world in picked}
    cand = [c for c in candidates if c.cls in (HARD, OPEN)]
    return [{"size": b.size, "axis": b.axis, "bin": b.label, "lo": b.lo, "hi": b.hi,
             "worlds": sum(b.holds(r) for r in records),
             "candidates": sum(b.holds(c.record) for c in cand),
             "selected": sum(b.holds(c.record) for c in candidates if c.world in sel),
             "covered": any(b.holds(c.record) for c in candidates if c.world in sel)}
            for b in bins]


def _hash_rank(world: str) -> str:
    return hashlib.sha256(world.encode()).hexdigest()


def split_worlds(selected: Sequence[WorldInfo]) -> dict[str, str]:
    """The split (D12): every selected 42-producer regime-switch world is held out out of region;
    the rest, within size x choice x pattern strata in sha256(world id) order, 20% held-out,
    20% validation, 60% train (counts rounded half up: held-out round(0.2 n), validation
    round(0.4 n) - held-out)."""
    out: dict[str, str] = {}
    strata: dict[tuple, list[str]] = defaultdict(list)
    for w in selected:
        if w.size == 42 and w.pattern == PATTERN_SWITCH:
            out[w.id] = HELDOUT_REGION
        else:
            strata[(w.size, w.choice, w.pattern)].append(w.id)
    for ids in strata.values():
        ids = sorted(ids, key=_hash_rank)
        n = len(ids)
        n_h = math.floor(SPLIT_SHARES[0] * n + 0.5)
        n_v = math.floor((SPLIT_SHARES[0] + SPLIT_SHARES[1]) * n + 0.5) - n_h
        for i, w in enumerate(ids):
            out[w] = HELDOUT if i < n_h else VALIDATION if i < n_h + n_v else TRAIN
    return out


def train_worlds(atlas: Mapping) -> list[dict]:
    """The worlds a teacher sample may draw from (spec "Held-out sealed"): train split only."""
    return [w for w in atlas["worlds"] if w["split"] == TRAIN]


# --- EXP-0050 links: provisional spreads and anchor reproduction ---


def exp0050_infos() -> dict[str, WorldInfo]:
    from .routing_audit import fanout_worlds

    infos = {}
    for e in fanout_worlds():
        fanout = e.generator["params"]["fanout"]
        infos[e.id] = WorldInfo(id=e.id, size=e.size,
                                choice=DENSE if e.family == DENSE else f"sparse_f{fanout}",
                                pattern="family", shocks=e.regime == SHOCKED,
                                differing=e.regime == SHOCKED)
    return infos


def provisional_spreads(dicts: Sequence[Mapping[str, str]], arms: Sequence[Mapping]
                        ) -> dict[Stratum, Spread]:
    """The screen's spreads (D12 "Screen"): sigma_ep and sigma_rep per size x fanout (dense its
    own) from EXP-0050's rows, with the arms both runs have (bar over the blind arms, reach =
    the trained orderer); a stratum without df takes its size's pool."""
    infos = exp0050_infos()
    values = episode_values(d for d in dicts if d["world"] in infos)
    ms = [measures(w, eps, arms) for w, eps in sorted(values.items())]
    return pooled_spreads(ms, infos, merge_below=1, estimator=SPREAD_ESTIMATOR[SCREEN])


def anchor_rows(dicts: Sequence[Mapping[str, str]], reference: Sequence[Mapping[str, str]],
                anchors: Sequence[WorldEntry]) -> list[dict]:
    """Per anchor row whose (world, episode, subject, replicate) EXP-0050 also ran: both units
    and whether they match (|diff| <= 1e-9)."""
    keys = {(e.id, e.episode_seeds[0]) for e in anchors}
    ref = {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])): float(d["units"])
           for d in reference if (d["world"], int(d["episode"])) in keys}
    out = []
    for d in sorted(dicts, key=lambda d: (d["world"], d["subject"], int(d["replicate"]))):
        k = (d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))
        if k in ref:
            u = float(d["units"])
            out.append({"world": k[0], "episode": k[1], "subject": k[2], "replicate": k[3],
                        "units": u, "exp0050_units": ref[k], "match": abs(u - ref[k]) <= 1e-9})
    return out


# --- deciding check (confirm) ---


def deciding_rows(values: Mapping[str, Mapping[int, Episode]], infos: Mapping[str, WorldInfo],
                  classes: Mapping[str, str], ms: Mapping[str, Measures],
                  spreads: Mapping[Stratum, Spread], arms: Sequence[Mapping]) -> list[dict]:
    """EXP-0050's deciding check on the confirm's hard worlds with E >= 4 (D12): per world
    makeable_split - trained orderer and makeable_weighted_whole - makeable_split against the
    pair's noise floor; the three branches at 2/3; and per arm the share of TTL-lost units whose
    receiver's inputs could not cover the order."""
    ms_s, mww_s = member_id(PRODUCER, "makeable_split"), member_id(PRODUCER,
                                                                   "makeable_weighted_whole")
    to_s = member_id(PRODUCER, TRAINED_ORDERER)
    stoch = {a["subject"]: bool(a["random"]) for a in arms}
    rows: list[dict] = []
    flags: dict[str, list[bool]] = defaultdict(list)
    for w in sorted(values):
        info, eps = infos[w], values[w]
        if classes.get(w) != HARD or info.eligible < DECIDING_MIN_CHOICE:
            continue
        if not all(s in eps[e].arms for e in eps for s in (ms_s, mww_s, to_s)):
            continue
        row: dict = {"kind": "world", "world": w, "size": info.size, "choice": info.choice,
                     "eligible": info.eligible}
        bar = ms[w].bar or 0.0
        for name, (a, b) in (("split_vs_trained", (ms_s, to_s)),
                             ("whole_vs_split", (mww_s, ms_s))):
            d = fmean(fmean(eps[e].arms[a].units) - fmean(eps[e].arms[b].units) for e in eps)
            reps = {s: min(len(eps[e].arms[s].units) for e in eps) for s in (a, b) if stoch[s]}
            se = standard_error(len(eps), info.differing, reps, spreads.get(info.stratum))
            floor = max(FLOOR_SE[CONFIRM] * se, FLOOR_BAR_SHARE * bar)
            row[f"{name}_gain"], row[f"{name}_floor"] = d, floor
        within = abs(row["split_vs_trained_gain"]) <= row["split_vs_trained_floor"]
        whole_above = row["whole_vs_split_gain"] > row["whole_vs_split_floor"]
        split_below = row["split_vs_trained_gain"] < -row["split_vs_trained_floor"]
        row.update(split_within_trained=within, whole_above_split=whole_above,
                   split_below_trained=split_below)
        for k, v in (("input_aware_weighting", within), ("splitting_costs", whole_above),
                     ("scorer_beyond_target", split_below)):
            flags[k].append(v)
        rows.append(row)
    for k in ("input_aware_weighting", "splitting_costs", "scorer_beyond_target"):
        n, hits = len(flags[k]), sum(flags[k])
        rows.append({"kind": "branch", "branch": k, "worlds": n, "count": hits,
                     "share": hits / n if n else None,
                     "holds": bool(n) and hits / n >= DECIDING_SHARE})
    for scope, ws in (("all", sorted(values)),
                      ("hard_e4", [r["world"] for r in rows if r["kind"] == "world"])):
        for a in arms:
            s = a["subject"]
            short = sum(eps_.arms[s].ttl_short for w in ws for eps_ in values[w].values()
                        if s in eps_.arms)
            ok = sum(eps_.arms[s].ttl_ok for w in ws for eps_ in values[w].values()
                     if s in eps_.arms)
            rows.append({"kind": "ttl", "scope": scope, "arm": s, "ttl_lost_inputs_short": short,
                         "ttl_lost_inputs_ok": ok,
                         "share_inputs_short": short / (short + ok) if short + ok else None})
    return rows


DECIDING_COLUMNS = ("kind", "world", "size", "choice", "eligible", "split_vs_trained_gain",
                    "split_vs_trained_floor", "whole_vs_split_gain", "whole_vs_split_floor",
                    "split_within_trained", "whole_above_split", "split_below_trained",
                    "branch", "worlds", "count", "share", "holds", "scope", "arm",
                    "ttl_lost_inputs_short", "ttl_lost_inputs_ok", "share_inputs_short")


# --- world table ---

WORLD_COLUMNS = ("world", "size", "choice", "pattern", "shocks", "anchor", "episodes",
                 "bar_arm", "bar", "bar_fill", "reach_arm", "reach", "gain", "b2", "b_base",
                 "b_lp", "b_lp_base",
                 "ceiling", "room", "sigma_ep", "df_ep", "merged_ep", "sigma_rep", "df_rep",
                 "merged_rep", "estimator", "sd_ep", "sd_rep", "se", "noise_floor",
                 "sign_consistent", "argmax_died", "class",
                 "arms_above_base", "screen_class", "class_changed", "split", "arm_units",
                 *COORDINATES)
SELECTION_COLUMNS = ("size", "rank", "world", "class", "score_gain_over_bar", "room", "pass",
                     "split", "bar_arm", "reach_arm", "choice", "pattern", "shocks")
COVERAGE_COLUMNS = ("size", "axis", "bin", "lo", "hi", "worlds", "candidates", "selected",
                    "covered")
ANCHOR_COLUMNS = ("world", "episode", "subject", "replicate", "units", "exp0050_units", "match")


def analyse(dicts: Sequence[Mapping[str, str]], infos: Mapping[str, WorldInfo],
            arms: Sequence[Mapping], mode: str, spreads: Mapping[Stratum, Spread] | None = None,
            fixed: Mapping[str, tuple[str | None, str | None]] | None = None
            ) -> tuple[dict[str, Measures], dict[Stratum, Spread], list[dict]]:
    """Measures, the spreads used (the screen's provisional ones, or the confirm's pooled
    ones, `SPREAD_ESTIMATOR[mode]`) and one `worlds.csv` row per world with its floor and class
    (the dead-arm rule in confirm)."""
    values = episode_values(dicts)
    ms = {w: measures(w, eps, arms, (fixed or {}).get(w)) for w, eps in sorted(values.items())}
    if spreads is None:
        spreads = pooled_spreads(list(ms.values()), infos, estimator=SPREAD_ESTIMATOR[mode])
    rows = []
    for w, m in ms.items():
        info = infos[w]
        sp = spreads.get(info.stratum)
        se, floor = noise_floor(m, info, sp, FLOOR_SE[mode])
        coords = (info.record or {}).get("coordinates", {})
        rows.append({
            "world": w, "size": info.size, "choice": info.choice, "pattern": info.pattern,
            "shocks": info.shocks, "anchor": info.anchor, "episodes": m.episodes,
            "bar_arm": m.bar_arm, "bar": m.bar, "bar_fill": m.bar_fill,
            "reach_arm": m.reach_arm, "reach": m.reach, "gain": m.gain, "b2": m.b2,
            "b_base": m.base, "b_lp": m.lp, "b_lp_base": m.lp_base, "ceiling": m.ceiling,
            "room": m.room,
            **({k: v for k, v in sp.to_dict().items()} if sp else {}),
            "se": se, "noise_floor": floor, "sign_consistent": sign_consistent(m),
            "argmax_died": m.argmax_died,
            "class": classify(m, floor, dead_arm=mode == CONFIRM), "arms_above_base": sum(m.above_base.values()),
            "arm_units": json.dumps(m.arm_means, sort_keys=True),
            **{c: coords.get(c) for c in COORDINATES}})
    return ms, dict(spreads), rows


def above_base_summary(ms: Mapping[str, Measures], infos: Mapping[str, WorldInfo]) -> dict:
    """Arm -> size x choice -> world-episodes on which the arm exceeded B_base."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w, m in ms.items():
        for s, n in m.above_base.items():
            out[s][f"{infos[w].size}/{infos[w].choice}"] += n
    return {s: dict(v) for s, v in sorted(out.items())}


# --- run ---

QUESTIONS = {
    SCREEN: """\
Where in the world space does a better routing policy exist that is cheap to find (design D12)?
Every sampled world is played by the blind and hand-written arms (the bar) and the reach arms
(`makeable_split`, `makeable_weighted_whole`, the trained orderer); the restored bounds give each
world's ceiling. Each world gets a provisional class (collapse, saturated, hard, open, flat) and
the hard and open worlds are selected and split for the confirm stage.""",
    CONFIRM: """\
Do the screen's selected worlds keep their class on fresh episode and replicate seeds, with the
pooled noise floor (design D12)? Plus EXP-0050's deciding check on the hard worlds with E >= 4
and the freeze of the atlas (`atlas_v2.json`).""",
}

RESULT_SKELETON = """# {title}

{banner}**Status: {status}.** `--mode {mode}`: {n_rows} episode rows over {n_world_episodes}
world-episodes, {n_arms} arms, horizon {horizon}; roster {roster}. Produced by
`python -m {driver} --mode {mode}` (see `spec.json`).

## Question

{question}

## Result

TODO: plain-English reading of `worlds.csv` (classes per size x choice x pattern, gains against
the noise floor, room against the base-capacity ceiling, arms above B_base){extra}.

## Conclusions

TODO: what the classes say about where headroom is (DIAGNOSTICS entry 20) and what changes.
"""
DRY_BANNER = ("> **DRY RUN — not an acceptance read.** Filtered sizes, worlds, arms or "
              "replicates, another horizon or shock window, or no source commit; see "
              "`spec.json`.\n\n")


def real_run_problems(*, roster: Roster, worlds: Sequence[WorldEntry], horizon: int,
                      source_commit: str, filters: Mapping[str, object],
                      shock_window: Sequence[int]) -> list[str]:
    """Why this is not a valid recorded run (empty = valid)."""
    problems = [msg for bad, msg in (
        (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (not lp_available(), "B_lp needs scipy: install the lp extra (pip install -e .[lp])"),
        (tuple(shock_window) != tuple(SHOCKS_I_WINDOW),
         f"shock window {tuple(shock_window)} is not {SHOCKS_I_WINDOW} (test-only)"),
    ) if bad]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    smoke = test_only_problem(worlds)
    if smoke:
        problems.append(smoke)
    return problems


def _canonical(o: object) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(doc: Mapping) -> str:
    """sha256 of the canonical JSON of `doc` without its `content_hash` field."""
    return hashlib.sha256(_canonical({k: v for k, v in doc.items()
                                      if k != "content_hash"}).encode()).hexdigest()


def _jsonable_float(v: object) -> object:
    return None if isinstance(v, float) and not math.isfinite(v) else v


def run_atlas(*, mode: str, out_dir: str | Path, roster: Roster, workers: int = 1,
              horizon: int = HORIZON, sizes: Sequence[int] | None = None,
              worlds_per_size: int | None = None, arms: Sequence[str] | None = None,
              replicates: int | None = None, shock_profile: str | None = None,
              screen_dir: str | Path | None = None, exp0050_episodes: str | None = None,
              prod_ckpt: str | None = None, orderer_ckpt: str | None = None,
              source_commit: str = "", dry_run: bool = False,
              run_job_fn: Callable[[Job], dict[str, str]] = run_atlas_job,
              log: Callable[[str], None] = print) -> int:
    """Run (or resume) the atlas screen or confirm into `out_dir`; return the exit status (0, or
    `CONTRACT_INVALID` on contract failures, a bound violation, an anchor mismatch or missing
    rows)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {MODES}")
    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    if sizes and set(sizes) - set(SIZES):
        raise ValueError(f"--sizes takes {SIZES}, got {sizes}")
    size_list = list(sizes) if sizes else list(SIZES)
    table = arm_table(roster, mode, arms, replicates)
    trained = any(a["trained"] for a in table)
    if trained and not (prod_ckpt and orderer_ckpt):
        raise ValueError("the trained arms need --prod-ckpt and --orderer-ckpt")
    out = Path(out_dir)
    infos: dict[str, WorldInfo] = {}
    fixed: dict[str, tuple[str | None, str | None]] = {}
    splits: dict[str, str] = {}
    screen_classes: dict[str, str] = {}
    screen_pins: dict = {}
    if mode == SCREEN:
        if screen_dir:
            raise ValueError("--screen-dir is for --mode confirm")
        if not dry_run and not exp0050_episodes:
            raise ValueError("a real screen needs --exp0050-episodes (provisional spreads and "
                             "anchor reproduction)")
        window = SMOKE_START_WINDOW if shock_profile else SHOCKS_I_WINDOW
        points = screen_points(size_list, worlds_per_size, window)
        records = [space_record(p) for p in points]
        worlds = [space_entry(p, episode_seeds(p.world_id, SCREEN_EPISODES)) for p in points]
        anchors = anchor_worlds(shock_profile or SHOCKS_I, size_list)
        if worlds_per_size is not None:  # a dry run keeps the first anchors of each size
            anchors = [e for e in anchors
                       if [a.size for a in anchors[:anchors.index(e)]].count(e.size)
                       < worlds_per_size]
        infos.update({r["world_id"]: space_info(r) for r in records})
        infos.update({e.id: anchor_info(e) for e in anchors})
        world_pins = records + [fanout_record(e) for e in anchors]
        worlds += anchors
    else:
        if not screen_dir:
            raise ValueError("--mode confirm needs --screen-dir")
        if shock_profile:
            raise ValueError("confirm runs the screen's worlds as recorded; --shock-profile is "
                             "for the screen")
        sdir = Path(screen_dir)
        sspec = json.loads((sdir / "spec.json").read_text())
        if sspec.get("pins", {}).get("mode") != SCREEN:
            raise ValueError(f"{sdir} is not an atlas screen")
        if not dry_run and (sspec.get("kind") != "screen" or sspec.get("status") != "complete"):
            raise ValueError(f"a real confirm needs a complete real screen; {sdir} is "
                             f"{sspec.get('kind')} / {sspec.get('status')}")
        with (sdir / "selection.csv").open(newline="") as fh:
            selection = list(csv.DictReader(fh))
        with (sdir / "worlds.csv").open(newline="") as fh:
            screened = list(csv.DictReader(fh))
        recs = {r["world_id"]: r for r in sspec["pins"]["worlds"] if "world_id" in r}
        chosen = [s for s in selection if int(s["size"]) in size_list]
        stand_ins: list[str] = []
        if worlds_per_size is not None:  # dry run: the first N per size, padded if needed
            picked: list[dict] = []
            for size in size_list:
                mine = [s for s in chosen if int(s["size"]) == size][:worlds_per_size]
                have = {s["world"] for s in mine}
                for r in screened:
                    if len(mine) >= worlds_per_size:
                        break
                    if (r["world"] in recs and int(r["size"]) == size
                            and r["world"] not in have):
                        mine.append({"world": r["world"], "size": r["size"],
                                     "split": TRAIN, "bar_arm": r["bar_arm"],
                                     "reach_arm": r["reach_arm"], "class": r["class"]})
                        stand_ins.append(r["world"])
                picked += mine
            chosen = picked
        if not chosen:
            raise ValueError(f"{sdir} selected no worlds")
        world_pins, worlds = [], []
        for s in chosen:
            rec = recs[s["world"]]
            config_from_record(rec)  # the rebuilt config must hash to the recorded one
            info = space_info(rec)
            infos[info.id] = info
            point = point_from_record(rec)
            worlds.append(space_entry(point, episode_seeds(
                info.id, confirm_episode_count(info), CONFIRM_STAGE), split=s["split"]))
            world_pins.append(rec)
            fixed[info.id] = (s["bar_arm"] or None, s["reach_arm"] or None)
            splits[info.id] = s["split"]
            screen_classes[info.id] = s["class"]
        screen_pins = {"dir": str(sdir), "experiment": sspec.get("experiment"),
                       "spec_kind": sspec.get("kind"),
                       "source_commit": sspec.get("pins", {}).get("source", {}).get("commit"),
                       "selection_sha256": sha256_file(sdir / "selection.csv"),
                       "worlds_sha256": sha256_file(sdir / "worlds.csv"),
                       "stand_ins": stand_ins}
        window = tuple(worlds[0].generator["params"]["shock_window"])
    if not dry_run:
        problems = real_run_problems(
            roster=roster, worlds=worlds, horizon=horizon, source_commit=source_commit,
            filters={"sizes": sizes, "worlds_per_size": worlds_per_size, "arms": arms,
                     "replicates": replicates, "shock_profile": shock_profile},
            shock_window=window)
        if problems:
            raise ValueError(f"not a valid atlas {mode} run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    checkpoints = (checked_checkpoints(out, {"producer": prod_ckpt, "orderer": orderer_ckpt},
                                       dry_run, log) if trained else {})
    grid = job_grid(worlds, None)
    source = source_fingerprint(source_commit or None)
    pins = {
        "driver": DRIVER, "mode": mode, "atlas": ATLAS, "horizon": horizon, "dry_run": dry_run,
        "roster_version": roster.version, "roster_name": roster.name,
        "checkpoints": {k: v["sha256"] for k, v in checkpoints.items()},
        "producer": PRODUCER, "arms": table, "shock_window": list(window),
        "exp0050_episodes_sha256": (sha256_file(exp0050_episodes) if exp0050_episodes
                                    else None),
        "screen": screen_pins or None, "worlds": world_pins,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "grid": grid,
    }
    expected = expected_keys(grid, table, 1)
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else mode,
        "pins": pins, "roster": roster.payload() | {"content_hash": roster.version},
        "inputs": checkpoints, "source": source,
        "pre_registered": {
            "screen_worlds": SCREEN_WORLDS, "k": K, "open_share": OPEN_SHARE,
            "replicates": REPLICATES, "bound_tolerance": BOUND_TOLERANCE,
            "saturation_fill": SATURATION_FILL, "saturation_share": SATURATION_SHARE,
            "open_room": OPEN_ROOM, "collapse_cycle": COLLAPSE_CYCLE, "floor_se": FLOOR_SE,
            "floor_bar_share": FLOOR_BAR_SHARE, "min_pooled_df": MIN_POOLED_DF,
            "spread_estimator": SPREAD_ESTIMATOR, "dead_arm_rule": CONFIRM_RULES["dead_arm"],
            "deciding_share": DECIDING_SHARE, "split_shares": SPLIT_SHARES},
        "driver_choices": {
            "two_stochastic_argmax_arms": "SE_w replicate term = sum of sigma_rep^2 / R over "
                                          "the stochastic argmax arms",
            "replicate_sign_pairing": "replicate i of one arm with replicate i mod R of the "
                                      "other",
            "quartiles": "nearest rank over the size's screened worlds",
            "screen_spread_merge": "an EXP-0050 stratum with no df (e.g. no stochastic argmax "
                                   "arm) takes its size's pool",
            "split_rounding": "held-out round_half_up(0.2 n), validation "
                              "round_half_up(0.4 n) - held-out"},
    })
    ensure_failures_file(out)
    jobs = build_jobs(worlds, table, 1, horizon, None, checkpoints)
    if lp_available():  # a real run requires it (real_run_problems); a dry run may skip it
        flag_lp_jobs(jobs)
    if {job_key(j["world"], j["episode"], j["subject"], j["replicate"]) for j in jobs} != expected:
        raise AssertionError("job list and pinned grid disagree")  # internal invariant
    episodes = out / "episodes.csv"
    done_keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in done}
    execute(out, spec, done_keys, read_episode_dicts(episodes), jobs, workers, run_job_fn, log,
            columns=ATLAS_COLUMNS)

    rows = read_rows(episodes)
    check_rows(rows, grid, expected, complete=False)
    missing = expected - {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    dicts = read_episode_dicts(episodes)
    recorded = {f["message"] for f in read_failures(out)}
    for f in bound_violations(dicts):
        if f.message not in recorded:
            append_failure(out, f)
            log(f"[atlas] BOUND VIOLATION {f.message}")
    anchors_table: list[dict] = []
    anchor_note = None
    if mode == SCREEN:
        anchor_entries = [e for e in worlds if e.split == ANCHOR_SPLIT]
        if exp0050_episodes and horizon == HORIZON and not shock_profile:
            anchors_table = anchor_rows(dicts, read_episode_dicts(Path(exp0050_episodes)),
                                        anchor_entries)
            bad = [r for r in anchors_table if not r["match"]]
            if bad and not any(f["check"] == ANCHOR_CHECK for f in read_failures(out)):
                append_failure(out, Failure(
                    scope=RUN, check=ANCHOR_CHECK, error_type="AnchorMismatch",
                    message=f"{len(bad)} anchor rows differ from EXP-0050, e.g. {bad[0]}"))
        else:
            anchor_note = ("not compared: needs --exp0050-episodes, horizon 100 and the "
                           "registered shock window")
        write_csv(out / "anchors.csv", ANCHOR_COLUMNS, anchors_table)
    failures = read_failures(out)
    banner = DRY_BANNER if dry_run else ""
    extra = ""
    if failures or missing:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing),
                          "bound_violations": [f["message"] for f in failures
                                               if f["check"] == BOUND_CHECK]}
        _write_spec(out, spec)
        status = f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing row(s))"
        code = CONTRACT_INVALID
    else:
        result = _finish(mode, out, dicts, infos, table, fixed, splits, screen_classes,
                         exp0050_episodes, spec, screen_pins, source)
        result["anchors"] = {"rows": len(anchors_table), "note": anchor_note} \
            if mode == SCREEN else None
        spec["result"] = {"status": "complete", "rows": len(rows), "contract_failures": 0,
                          **result}
        spec["status"] = "complete"
        _write_spec(out, spec)
        status = "complete"
        code = 0
        extra = ("; `selection.csv` / `coverage.csv` (the selection and split), `anchors.csv` "
                 "(EXP-0050 reproduction)" if mode == SCREEN else
                 "; class changes against the screen; `deciding_check.csv` (EXP-0050's "
                 "deciding check, for DIAGNOSTICS entry 19); `atlas_v2.json`")
    if not (out / "RESULT.md").exists():
        (out / "RESULT.md").write_text(RESULT_SKELETON.format(
            title=out.name, banner=banner, status=status, mode=mode, n_rows=len(rows),
            n_world_episodes=sum(len(g["episodes"]) for g in grid.values()),
            n_arms=len(table), horizon=horizon, roster=roster.version[:12], driver=DRIVER,
            question=QUESTIONS[mode], extra=extra))
    log(f"[atlas] {mode} {'DRY RUN ' if dry_run else ''}{status}")
    return code


def _finish(mode: str, out: Path, dicts: Sequence[Mapping[str, str]],
            infos: Mapping[str, WorldInfo], table: Sequence[Mapping],
            fixed: Mapping[str, tuple[str | None, str | None]], splits: Mapping[str, str],
            screen_classes: Mapping[str, str], exp0050_episodes: str | None, spec: Mapping,
            screen_pins: Mapping, source: Mapping) -> dict:
    """The analysis files of a complete run; returns the spec's result summary."""
    if mode == SCREEN:
        spreads = (provisional_spreads(read_episode_dicts(Path(exp0050_episodes)), table)
                   if exp0050_episodes else {})
        ms, spreads, wrows = analyse(dicts, infos, table, SCREEN, spreads)
        by_world = {r["world"]: r for r in wrows}
        sel_rows, cov_rows = [], []
        selected_infos: list[WorldInfo] = []
        passes: dict[str, tuple[int, str, Candidate]] = {}
        for size in sorted({i.size for i in infos.values()}):
            space = [r for r in wrows if not infos[r["world"]].anchor and r["size"] == size]
            records = [infos[r["world"]].record for r in space]
            cands = [Candidate(r["world"], r["class"], r["gain"], r["bar"], r["room"],
                               infos[r["world"]].record) for r in space]
            bins = coverage_bins(size, records)
            chosen = select(cands, K[size], bins)
            cov_rows += coverage_rows(bins, records, cands, [c.world for c, _ in chosen])
            for rank, (c, how) in enumerate(chosen):
                passes[c.world] = (rank, how, c)
                selected_infos.append(infos[c.world])
        split = split_worlds(selected_infos)
        for w, (rank, how, c) in passes.items():
            info, r = infos[w], by_world[w]
            sel_rows.append({"size": info.size, "rank": rank, "world": w, "class": c.cls,
                             "score_gain_over_bar": (c.gain / c.bar if c.cls == HARD and c.bar
                                                     else None),
                             "room": c.room, "pass": how, "split": split[w],
                             "bar_arm": r["bar_arm"], "reach_arm": r["reach_arm"],
                             "choice": info.choice, "pattern": info.pattern,
                             "shocks": info.shocks})
        for r in wrows:
            r["split"] = split.get(r["world"])
        write_csv(out / "worlds.csv", WORLD_COLUMNS, wrows)
        write_csv(out / "selection.csv", SELECTION_COLUMNS, sel_rows)
        write_csv(out / "coverage.csv", COVERAGE_COLUMNS, cov_rows)
        return {"classes": _class_counts(wrows), "selected": len(sel_rows),
                "splits": _counts(split.values()),
                "provisional_spreads": {f"{s[0]}/{s[1]}": v.to_dict()
                                        for s, v in spreads.items()},
                "arms_above_base": above_base_summary(ms, infos)}
    header = {
        "screen": {k: screen_pins.get(k) for k in ("experiment", "source_commit", "spec_kind",
                                                   "selection_sha256")},
        "confirm": {"experiment": spec.get("experiment"), "source_commit": source["commit"],
                    "spec_kind": spec.get("kind")}}
    return confirm_outputs(out, dicts, infos, table, fixed, splits, screen_classes, header)


CONFIRM_RULES = {
    "spread_estimator": SPREAD_ESTIMATOR[CONFIRM], "floor_se": FLOOR_SE[CONFIRM],
    "floor_bar_share": FLOOR_BAR_SHARE, "min_pooled_df": MIN_POOLED_DF,
    "dead_arm": "collapse when the fixed bar or reach arm dies in any episode or replicate",
}


def confirm_outputs(out: Path, dicts: Sequence[Mapping[str, str]],
                    infos: Mapping[str, WorldInfo], table: Sequence[Mapping],
                    fixed: Mapping[str, tuple[str | None, str | None]],
                    splits: Mapping[str, str], screen_classes: Mapping[str, str],
                    header: Mapping, previous: Mapping[str, str] | None = None) -> dict:
    """The confirm's analysis files (`worlds.csv`, `deciding_check.csv`, the atlas file) under
    the current rules; `previous` = each world's class from an earlier classification of the
    same rows (reclassify), written beside the new one."""
    ms, spreads, wrows = analyse(dicts, infos, table, CONFIRM, None, fixed)
    for r in wrows:
        r["screen_class"] = screen_classes.get(r["world"])
        r["class_changed"] = r["screen_class"] != r["class"]
        r["split"] = splits.get(r["world"])
        if previous is not None:
            r["class_v1"] = previous.get(r["world"])
    columns = WORLD_COLUMNS + (("class_v1",) if previous is not None else ())
    write_csv(out / "worlds.csv", columns, wrows)
    classes = {r["world"]: r["class"] for r in wrows}
    write_csv(out / "deciding_check.csv", DECIDING_COLUMNS,
              deciding_rows(episode_values(dicts), infos, classes, ms, spreads, table))
    atlas = {
        "name": ATLAS, "format": ATLAS_FORMAT, **header, "rules": CONFIRM_RULES,
        "spreads": {f"{s[0]}/{s[1]}": v.to_dict() for s, v in spreads.items()},
        "worlds": [{"record": infos[r["world"]].record, "screen_class": r["screen_class"],
                    "confirm_class": r["class"], "gain": r["gain"], "room": r["room"],
                    "noise_floor": _jsonable_float(r["noise_floor"]), "split": r["split"],
                    "bar_arm": r["bar_arm"], "reach_arm": r["reach_arm"],
                    "argmax_died": r["argmax_died"],
                    **({"class_v1": r["class_v1"]} if previous is not None else {})}
                   for r in wrows],
    }
    atlas["content_hash"] = content_hash(atlas)
    (out / ATLAS_FILE).write_text(json.dumps(atlas, indent=1, sort_keys=True) + "\n")
    result = {"classes": _class_counts(wrows),
              "class_changed": sum(bool(r["class_changed"]) for r in wrows),
              "atlas_content_hash": atlas["content_hash"],
              "spreads": atlas["spreads"], "arms_above_base": above_base_summary(ms, infos)}
    if previous is not None:
        result["classes_v1"] = _class_counts([{"size": r["size"], "anchor": r["anchor"], "class": r["class_v1"]}
                                              for r in wrows])
        result["flipped"] = [{"world": r["world"], "class_v1": r["class_v1"],
                              "class": r["class"], "argmax_died": r["argmax_died"]}
                             for r in wrows if r["class_v1"] != r["class"]]
    return result


def _episodes_file(d: Path) -> Path:
    p = d / "artifacts" / "episodes.csv"
    p = p if p.exists() else d / "episodes.csv"
    if p.read_bytes()[:40].startswith(b"version https://git-lfs"):
        raise ValueError(f"{p} is a git-lfs pointer; run git lfs pull first")
    return p


def reclassify(confirm_dir: str | Path, out_dir: str | Path, source_commit: str | None = None,
               log: Callable[[str], None] = print) -> int:
    """Classify a recorded confirm's rows again under the current rules (task 5.3: the robust
    spread estimator and the dead-arm rule), offline: reads its `spec.json`, `worlds.csv` (fixed
    arms, split, screen and recorded classes) and `episodes.csv`, plays nothing and never writes
    into `confirm_dir`. Writes `worlds.csv` (with `class_v1`), `deciding_check.csv`,
    `atlas_v2.json` and `reclassify.json` (inputs' hashes, rules, counts, flips)."""
    cdir, out = Path(confirm_dir), Path(out_dir)
    if out.resolve() == cdir.resolve():
        raise ValueError("reclassify writes a derived directory, never the confirm's own")
    spec = json.loads((cdir / "spec.json").read_text())
    pins = spec.get("pins", {})
    if pins.get("mode") != CONFIRM or spec.get("status") != "complete":
        raise ValueError(f"{cdir} is not a complete atlas confirm "
                         f"({pins.get('mode')} / {spec.get('status')})")
    if read_failures(cdir):
        raise ValueError(f"{cdir} has contract failures; an invalid run is not reclassified")
    episodes = _episodes_file(cdir)
    with (cdir / "worlds.csv").open(newline="") as fh:
        recorded = list(csv.DictReader(fh))
    infos = {r["world_id"]: space_info(r) for r in pins["worlds"] if "world_id" in r}
    fixed = {r["world"]: (r["bar_arm"] or None, r["reach_arm"] or None) for r in recorded}
    splits = {r["world"]: r["split"] for r in recorded}
    screen_classes = {r["world"]: r["screen_class"] for r in recorded}
    previous = {r["world"]: r["class"] for r in recorded}
    dicts = read_episode_dicts(episodes)
    if {d["world"] for d in dicts} != set(previous) or set(previous) - set(infos):
        raise ValueError("episodes.csv, worlds.csv and the pinned world records disagree")
    out.mkdir(parents=True, exist_ok=True)
    screen = pins.get("screen") or {}
    old_atlas = cdir / "atlas_v1.json"
    header = {
        "screen": {k: screen.get(k) for k in ("experiment", "source_commit", "spec_kind",
                                              "selection_sha256")},
        "confirm": {"experiment": spec.get("experiment"),
                    "source_commit": pins.get("source", {}).get("commit"),
                    "spec_kind": spec.get("kind")},
        "reclassified_from": {
            "episodes_sha256": sha256_file(episodes),
            "worlds_sha256": sha256_file(cdir / "worlds.csv"),
            "atlas_v1_content_hash": (json.loads(old_atlas.read_text()).get("content_hash")
                                      if old_atlas.exists() else None)}}
    result = confirm_outputs(out, dicts, infos, pins["arms"], fixed, splits, screen_classes,
                             header, previous)
    doc = {"experiment": out.name, "driver": DRIVER, "kind": RECLASSIFY,
           "not_an_acceptance_read": spec.get("kind") != CONFIRM,
           **header, "rules": CONFIRM_RULES,
           "source": {k: v for k, v in source_fingerprint(source_commit).items()
                      if k in ("tree_sha256", "commit")},
           "result": result}
    (out / "reclassify.json").write_text(json.dumps(doc, indent=1, sort_keys=True,
                                                    default=str) + "\n")
    log(f"[atlas] reclassified {len(previous)} worlds of {cdir.name}: "
        f"{result['classes_v1']} -> {result['classes']}")
    return 0


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for v in values:
        out[v] += 1
    return dict(sorted(out.items()))


def _class_counts(rows: Sequence[Mapping]) -> dict[str, dict[str, int]]:
    """size -> class -> worlds (anchors apart)."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        key = "anchor" if r["anchor"] else str(r["size"])
        out.setdefault(key, {c: 0 for c in CLASSES})[r["class"]] += 1
    return out


def _csv_ints(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",")] if s else None


def _csv_strs(s: str | None) -> list[str] | None:
    return [x for x in s.split(",") if x] if s else None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=f"python -m {DRIVER}", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES + (RECLASSIFY,), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--roster", default=str(DEFAULT_ROSTER))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--screen-dir", help="confirm: the screen's EXP dir (its selection)")
    ap.add_argument("--confirm-dir", help="reclassify: a recorded confirm's EXP dir (read only)")
    ap.add_argument("--exp0050-episodes",
                    help="screen: EXP-0050's artifacts/episodes.csv (provisional spreads, "
                         "anchor reproduction)")
    ap.add_argument("--prod-ckpt", help="portable producer checkpoint (EXP-0035 "
                                        "producer_candidate.pt)")
    ap.add_argument("--orderer-ckpt", help="portable orderer checkpoint (EXP-0012 "
                                           "orderer_scorer.pt)")
    ap.add_argument("--horizon", type=int, default=HORIZON, help="dry run: another horizon")
    ap.add_argument("--sizes", help="dry run, e.g. 12")
    ap.add_argument("--worlds-per-size", type=int,
                    help="dry run: the first N sampled (screen) or selected (confirm) worlds "
                         "per size; the screen keeps the first N anchors per size")
    ap.add_argument("--arms", help="dry run: comma list of arm (router) names")
    ap.add_argument("--replicates", type=int, help="dry run: R of every stochastic arm")
    ap.add_argument("--shock-profile", choices=(SHOCKS_I_SMOKE,),
                    help="dry run, screen: the smoke shock window (onsets at cycles 2-8)")
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--verify-source-commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="smoke run: filters allowed, marked as not an acceptance read")
    args = ap.parse_args(argv)
    source_commit = resolve_verified_commit() if args.verify_source_commit else args.source_commit
    if args.mode == RECLASSIFY:
        if not args.confirm_dir:
            ap.error("--mode reclassify needs --confirm-dir")
        return reclassify(args.confirm_dir, args.out_dir, source_commit or None)
    try:
        return run_atlas(
            mode=args.mode, out_dir=args.out_dir, roster=load_roster(args.roster),
            workers=args.workers, horizon=args.horizon, sizes=_csv_ints(args.sizes),
            worlds_per_size=args.worlds_per_size, arms=_csv_strs(args.arms),
            replicates=args.replicates, shock_profile=args.shock_profile,
            screen_dir=args.screen_dir, exp0050_episodes=args.exp0050_episodes,
            prod_ckpt=args.prod_ckpt, orderer_ckpt=args.orderer_ckpt,
            source_commit=source_commit, dry_run=args.dry_run)
    except (DigestMismatch, ContractRunError) as exc:
        print(f"[atlas] INVALID run: {exc}", file=sys.stderr)
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
