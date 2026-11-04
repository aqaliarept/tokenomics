"""Training worlds and the bar precompute for step-2 ES: `python -m tokenomics.train.order_es_worlds`
(add-m4-orderer-redesign design D6, task 5.1; learned-orderer "Relative disqualification").

Worlds (D6 "Training worlds"): the 23 loaded (rho > 0.28) non-collapse 12-producer atlas v2 train
worlds, plus fresh `eval/world_space` draws at 12 producers, 64 in all.
- Fresh draws are the Latin hypercube of the world-space generator under the atlas name
  `FRESH_ATLAS` (`FRESH_POOL` points, the atlas lever distribution: every lever range, shocks
  off / on and the seven demand patterns). The redraw order is the index order.
- A draw whose world seed or training episode seed is a reserved seed (`reserved_seeds`: every
  atlas v1 / v2 world seed at 12 and 42 producers, the screen's and confirm's episode seeds,
  hence the validation and held-out splits, EXP-0050's fanout worlds and registry v1 / v2) is
  skipped, so the stream is disjoint by construction. Atlas train worlds get fresh episode
  seeds too (stage `TRAIN_STAGE`).
- Filters on a draw (in this order): measured rho <= 0.28 (offline), a bar arm dies on a
  training episode, the world's bar fill >= 0.95 (the atlas's saturation fill: the fill rate of
  the arm with the best mean units).
- "A0 infeasible" (D6): a candidate training world (an atlas world, or a draw that passed the
  filters) on which A0 dies is replaced by the next draw. More than 1/4 of the target (16 of
  64) replaced, or fewer than 50 feasible worlds once the pool is exhausted, stops the run as
  `no_feasible_start` (exit 3), which goes to the user.

`--mode bar` plays, on every candidate world-episode (2 training episodes per world), with the
`entity_planner` producer (D2, the user's decision): RR (`round_robin`), MWW
(`makeable_weighted_whole`, R = 2), TR (the trained orderer as deployed), the roster v2
deterministic members (none needs a checkpoint) under `logits-v1`, and A0 (the zero-residual
per-order policy, = TS) under `choice-v1` through `per_order_env`, keyed by
DrawKey(world, episode, 0). Worlds are played in waves in the fixed order: a wave holds every
candidate whose outcome is still unknown, so the selection is the same as playing the whole order
and taking the first feasible worlds, and a resume replays the waves from the rows.

Per-job timings (`episode_seconds`, `cpu_seconds`, `job_seconds`) are columns of `episodes.csv`
and `timings.csv`; each training world records its mean cost per arm (`cost_s`), which task 5.2
uses for longest-first submission.

Outputs: `spec.json`, `episodes.csv`, `contract_failures.csv`, `candidates.csv` (every world in
the order, its status), `bar.csv` (one row per training world-episode: B_we, every arm's units, A0's
guards, D_we), `training_worlds.json` (the pinned worlds, hashed), `schedule.json` (the 16-world-
episode monitor set, the minibatch schedule per optimiser seed and the probe's two minibatches,
from the recorded `SCHEDULE_STREAM`), `fit_rows/` (A0's raw normaliser rows per world-episode),
`normaliser.json` + `normaliser_report.csv` (D5's normaliser fitted on the training world-episodes'
A0 rows only), `bar_summary.json`, a `RESULT.md` skeleton.
`--mode fit-normaliser --bar-dir <bar> --out-dir <derived>` refits it offline from a bar run's
rows, compares the hash with the bar's, and never writes the bar dir.

A real run needs the registered atlas and EXP-0012 checkpoint, the frozen roster, horizon 100, a
40-hex source commit and no filters; anything else needs `--dry-run`. Resume needs every pin to
match (including each arm's decode). Contract failures or missing rows: exit 2.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from ..env.spaces import CHOICE_V1, LOGITS_V1, decode_pins
from ..eval.atlas import (
    CONFIRM_EPISODES,
    CONFIRM_STAGE,
    SATURATION_FILL,
    SCREEN_EPISODES,
    SCREEN_WORLDS,
    space_entry,
)
from ..eval.atlas import ATLAS as ATLAS_V1_NAME
from ..eval.baselines import PRODUCING_BASELINES
from ..eval.contract import ContractMonitor, ContractRunError, ensure_failures_file, job_failure
from ..eval.contract import read_failures
from ..eval.decoder_check import sample_views
from ..eval.gate import DigestMismatch, load_side_effect_params, row_from_stats, row_to_dict
from ..eval.gate_cli import (CONTRACT_INVALID, INVALID, Job, Key, _open_spec, _prepare_resume,
                             _write_spec, job_key, read_rows)
from ..eval.provenance import resolve_verified_commit, sha256_file, source_fingerprint
from ..eval.registry import REGISTRY_V1, REGISTRY_V2, WorldEntry, load_registry
from ..eval.roster import (DEFAULT_ROSTER, JOINT_STACK, ROSTER_V2, Roster, load_roster,
                           member_id)
from ..eval.routing_audit import (EPISODE_COLUMNS, PRODUCER, TRAINED_ORDERER, _is_commit,
                                  _trained_entry, build_jobs, execute, expected_keys,
                                  fanout_worlds, job_config, job_policies, read_episode_dicts,
                                  trace_cells, write_csv)
from ..eval.routing_diagnostics import (DETERMINISTIC, ROUTER_KIND, STOCHASTIC, DrawKey,
                                        set_draw_key)
from ..eval.routing_trace import RoutingTracer
from ..eval.shock_profiles import SHOCKS_I_SMOKE, SMOKE_START_WINDOW
from ..eval.teacher_audit import (REGISTERED_ATLAS_HASH, ROLE_CHECKPOINTS, checked_orderer,
                                  load_atlas)
from ..eval.world_space import (WorldPoint, declarers, episode_seeds, point_from_record,
                                realized_demand, sample_worlds, world_id, world_record,
                                world_seed)
from ..eval.world_space import config_from_record
from .per_order_scorer import (CLIP, N_PHI, N_PSI, NORMALISER_VERSION, PHI_FEATURES,
                               PSI_FEATURES, SCALE_FLOORS, FeatureNormaliser, RawRows,
                               input_transforms, transform_psi)

DRIVER = "tokenomics.train.order_es_worlds"
WORLDS_FORMAT = "m4-es-training-worlds-v1"
SCHEDULE_FORMAT = "m4-es-schedule-v1"
BAR, FIT = "bar", "fit-normaliser"
HORIZON = 100
SIZE = 12  # D6: training at 12 producers only
ATLAS_WORLDS = 23  # loaded non-collapse 12-producer atlas v2 train worlds (D9)
TARGET_WORLDS = 64  # 23 + 41 fresh draws
MIN_FEASIBLE = 50
MAX_REPLACED_SHARE = 0.25  # "more than 1/4 ... need replacement" -> no_feasible_start
RHO_LOADED = 0.28
FRESH_ATLAS = "m4-es-train-v1"  # world seeds / ids of the fresh draws hash this name
FRESH_POOL = 256  # the draw order; ~2/3 pass rho (172 of 256), so 41 fit with room
TRAIN_STAGE = "m4-es-train"  # episode-seed salt of every training world-episode
EPISODES_PER_WORLD = 2  # D10: 64 worlds x 2 episodes
MWW_R = 2
# D6's relative guards: X / Y of the frozen side-effect params v2 (0.7516 / 0.7302 rounded)
SIDE_EFFECT_PARAMS = load_side_effect_params()
X_RUNDOWN = SIDE_EFFECT_PARAMS.x
Y_STARVATION = SIDE_EFFECT_PARAMS.y
# schedule (D6 "Training-monitor set", "Common random numbers")
SCHEDULE_STREAM = "m4-es-schedule-v1"
MONITOR_WORLD_EPISODES = 16
MONITOR_REPLICATES = 2
MINIBATCH = 12
GENERATIONS = 40
OPT_SEEDS = (0, 1)
PROBE_MINIBATCH = 6
TRAIN_REPLICATE = 0  # every training job's draw key replicate (CRN with the members)

ATLAS_SOURCE, FRESH_SOURCE = "atlas", "fresh"
TRAINING = "training"
DROPPED_RHO = "dropped_rho"
SKIPPED_SEED = "skipped_reserved_seed"
DROPPED_BAR_DEAD = "dropped_bar_dead"
DROPPED_SATURATED = "dropped_saturated"
REPLACED_A0 = "replaced_a0_dead"
INCOMPLETE = "incomplete"
UNPLAYED = "unplayed"
COMPLETE, NO_FEASIBLE_START = "complete", "no_feasible_start"
NO_FEASIBLE_EXIT = 3

RR, MWW, TR, A0 = "RR", "MWW", "TR", "A0"
A0_ORDERING = "per_order_a0"
A0_CONTRACT = "per_order_choice_v1"
A0_SUBJECT = member_id(PRODUCER, A0_ORDERING)
BAR_ROUTER = {RR: "round_robin", MWW: "makeable_weighted_whole"}
TIMING_COLUMNS = ("episode_seconds", "cpu_seconds", "job_seconds")
FIT_COLUMNS = ("fit_phi_rows", "fit_psi_rows", "fit_rows_sha256")
COLUMNS = EPISODE_COLUMNS + ("arm", "ordering_decode") + TIMING_COLUMNS + FIT_COLUMNS
FIT_DIR = "fit_rows"

# The normaliser's fitting code (carry-forward "Task 5.3", the 5.1 review; review round 1 of
# 5.3): the source text, decorators included, of these top-level definitions — the fit and apply
# (normaliser, input transforms, scale floors, clip, fit-row loading) and the code that builds and
# feeds its rows (the per-order policy's `decide`, which transforms the rows for both the fit sink
# and the normaliser; the shared / history / order / product row builders; the agent-local
# history) — hashed by `normaliser_code_sha256` and recorded in `NORMALISER_PREREG`. Every bar
# run writes that block into `normaliser.json`, and every ES / validation run requires it equal
# to its own, so a fit made, or rows built, by other code than the frozen pre-registration's is
# refused. Not covered: the domain / env state those builders read (`product_stream_view`, the
# outcome kinds); each run's package tree hash records that code.
NORMALISER_CODE_DEFS: dict[str, tuple[str, ...]] = {
    "train/per_order_scorer.py": (
        "CLIP", "HAS_COMPLETED", "OWN_TURN_FEATURES", "ORDER_FEATURES", "PHI_FEATURES",
        "PSI_FEATURES", "N_PHI", "N_PSI", "LOG1P_PHI", "LOG1P_PSI", "SCALE_FLOORS",
        "_LOG1P_PHI_COLS", "_LOG1P_PSI_COLS", "_SHARED", "_HIST", "transform_phi",
        "transform_psi", "input_transforms", "apply_scale_floors", "FeatureNormaliser",
        "RawRows", "shared_rows", "history_rows", "order_features", "PerOrderPolicy"),
    "train/orderer_scorer.py": ("_BOUND", "_ORDERER_PRODUCT_FEATURES",
                                "_ORDERER_PRODUCER_ROW_FEATURES", "_product_features",
                                "_producer_row_features"),
    "env/agent_history.py": ("FEATURES", "N_FEATURES", "KINDS", "OUTCOME_KINDS", "PRIOR_UNITS",
                             "PRIOR_RATE", "NO_LATENCY", "_PairState", "_EMPTY", "AgentHistory",
                             "stream_producers", "_FoldHook", "_ResetHook",
                             "attach_agent_history", "env_agent_id"),
    "train/order_es_worlds.py": ("rows_digest", "load_fit_rows", "fit_normaliser",
                                 "fit_rows_sha256"),
}


def _top_level_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def normaliser_code_source(source: str, names: Sequence[str]) -> str:
    """The source lines (not `ast.dump`, so independent of the Python version) of `names'`
    top-level definitions in `source`, decorators included, in `names` order; an absent name is
    marked."""
    tree = ast.parse(source)
    lines = source.splitlines()
    seg = {}
    for node in tree.body:
        for n in _top_level_names(node):
            if n in names:
                decorators = getattr(node, "decorator_list", [])
                start = min([d.lineno for d in decorators] + [node.lineno])
                seg[n] = "\n".join(lines[start - 1:node.end_lineno])
    return "\n".join(f"# {n}\n{seg.get(n, '<absent>')}" for n in names)


def normaliser_code_sha256() -> str:
    """SHA-256 over `NORMALISER_CODE_DEFS`' source text, file by file (package-relative)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel, names in sorted(NORMALISER_CODE_DEFS.items()):
        text = normaliser_code_source((root / rel).read_text(), names)
        h.update(rel.encode())
        h.update(b"\0")
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


# D5 / carry-forward "Task 5.1": the normaliser rule as fitted here; task 5.3 freezes this entry
# into es_prereg_m4_v1.json (its `normaliser` section; `readings.normaliser_pins` holds its hash)
HEAVY_CLIP_SHARE = 0.25  # a column clipping on >= 1/4 of its rows is flagged
NORMALISER_PREREG = {
    "version": NORMALISER_VERSION, "centre": "median (numpy percentile 50, linear)",
    "scale": "IQR (75th - 25th, numpy linear)",
    "zero_spread_rule": "IQR = 0 -> the column's population std; std = 0 -> 1.0 (all-equal "
                        "values have std 0 exactly)",
    "clip": CLIP, "clip_rule": "normalised values clipped to +-clip",
    "fit_rows": "A0's unnormalised rows (after the input transforms) on the training "
                "world-episodes of the bar precompute only (horizon 100, agent-history mode, "
                "after the infeasible-world replacement): one phi row per (decision, eligible "
                "producer), one psi row per decision",
    "phi_features": list(PHI_FEATURES), "psi_features": list(PSI_FEATURES),
    "transforms": input_transforms(),
    "transform_rule": "the listed columns are log1p-transformed before the fit and before every "
                      "apply (policy and fit rows alike); the rest enter raw",
    "scale_floors": SCALE_FLOORS,
    "scale_floor_rule": "a floored column's scale = max(the rule's scale, its floor); floor of "
                        "psi.eligible_producer_count = log1p(4) - log1p(2), the spread of its "
                        "two 12-producer values",
    "fitting_code": {"defs": {k: list(v) for k, v in sorted(NORMALISER_CODE_DEFS.items())},
                     "sha256": normaliser_code_sha256(),
                     "rule": "SHA-256 over, per package-relative file in sorted order, the "
                             "file name, NUL, the source lines of the listed top-level "
                             "definitions, decorators included, in list order (each '# name' "
                             "then its lines), NUL",
                     "covers": "the normaliser fit and apply, the input transforms, scale "
                               "floors and clip, fit-row loading, and the code that builds and "
                               "feeds the rows (PerOrderPolicy, the shared / history / order / "
                               "product row builders, the agent-local history)",
                     "not_covered": "the domain / env state the row builders read "
                                    "(product_stream_view, bid outcome kinds): recorded by each "
                                    "run's package tree hash only"},
}
# psi columns that are config facts of the order's product, checked at 42 producers
STATIC_PSI = ("product_depth", "recipe_input_count", "recipe_input_units",
              "recipe_capacity_type_count", "eligible_producer_count")

ORDER_INDEX = "order"


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_hash(doc: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in doc.items() if k != "content_hash"})
                          ).hexdigest()


# --- seeds ---


def train_episode_seeds(wid: str, count: int = EPISODES_PER_WORLD) -> tuple[int, ...]:
    return episode_seeds(wid, count, TRAIN_STAGE)


def reserved_seeds() -> dict[str, set[int]]:
    """Every seed a training world or episode must avoid, by source: the atlas v1 sample (the
    worlds of atlas v1 and v2, so also its validation and held-out splits) at 12 and 42
    producers with its screen and confirm episode seeds, EXP-0050's fanout grid, and the gate
    registries v1 / v2."""
    out: dict[str, set[int]] = {"atlas_world": set(), "atlas_screen_episode": set(),
                                "atlas_confirm_episode": set(), "exp0050": set(),
                                "registry": set()}
    for size, n in SCREEN_WORLDS.items():
        for i in range(n):
            wid = world_id(ATLAS_V1_NAME, size, i)
            out["atlas_world"].add(world_seed(ATLAS_V1_NAME, size, i))
            out["atlas_screen_episode"].update(episode_seeds(wid, SCREEN_EPISODES))
            out["atlas_confirm_episode"].update(episode_seeds(wid, CONFIRM_EPISODES,
                                                              CONFIRM_STAGE))
    for e in fanout_worlds():
        out["exp0050"].add(e.world_seed)
        out["exp0050"].update(e.episode_seeds)
    for path in (REGISTRY_V1, REGISTRY_V2):
        for e in load_registry(path).entries:
            out["registry"].add(e.world_seed)
            out["registry"].update(e.episode_seeds)
    return out


def reserved_union() -> set[int]:
    return set().union(*reserved_seeds().values())


# --- candidates ---


@dataclass(frozen=True)
class Candidate:
    """A world in the fixed order: its point, source, training episodes and (fresh draws)
    the offline rho / seed screening."""

    order: int
    source: str
    point: WorldPoint
    episodes: tuple[int, ...]
    rho: float
    record: Mapping[str, Any] | None = None  # the atlas's record (atlas worlds)
    screened: str | None = None  # DROPPED_RHO / SKIPPED_SEED, else None (playable)

    @property
    def world(self) -> str:
        return self.point.world_id

    def entry(self) -> WorldEntry:
        return space_entry(self.point, self.episodes, split=TRAIN_STAGE)


def _rho(point: WorldPoint) -> float:
    from ..eval.families import demand_units_per_cycle, top_layer_throughput

    cfg = point.config()
    return demand_units_per_cycle(cfg) / top_layer_throughput(cfg)  # world_space.measure's rho


def atlas_candidates(atlas: Mapping, max_worlds: int | None = None,
                     shock_profile: str | None = None) -> list[Candidate]:
    """The loaded non-collapse 12-producer atlas train worlds (decoder check D3's sample at 12),
    sorted by world id, at fresh training episode seeds."""
    views = [v for v in sample_views(atlas) if v["size"] == SIZE]
    views = views[:max_worlds] if max_worlds is not None else views
    out = []
    for i, v in enumerate(views):
        rec = v["record"]
        point = point_from_record(rec)
        if shock_profile is None:
            config_from_record(rec)  # the recorded config hash must rebuild
        else:
            point = replace(point, shock_window=tuple(SMOKE_START_WINDOW))
        out.append(Candidate(i, ATLAS_SOURCE, point, train_episode_seeds(v["world"]),
                             float(v["rho"]), rec))
    return out


def fresh_candidates(start: int, pool: int = FRESH_POOL, shock_profile: str | None = None,
                     reserved: set[int] | None = None,
                     episodes_per_world: int = EPISODES_PER_WORLD) -> list[Candidate]:
    """The fresh draws in index order, each screened offline: a reserved world or episode seed
    -> skipped, rho <= 0.28 -> dropped."""
    reserved = reserved_union() if reserved is None else reserved
    window = tuple(SMOKE_START_WINDOW) if shock_profile is not None else None
    out = []
    for k, point in enumerate(sample_worlds(FRESH_ATLAS, SIZE, pool)):
        if window is not None:
            point = replace(point, shock_window=window)
        eps = train_episode_seeds(point.world_id, episodes_per_world)
        rho = _rho(point)
        screened = (SKIPPED_SEED if point.seed in reserved or reserved & set(eps)
                    else DROPPED_RHO if not rho > RHO_LOADED else None)
        out.append(Candidate(start + k, FRESH_SOURCE, point, eps, rho, None, screened))
    return out


# --- arms ---


def bar_arms(roster: Roster) -> list[dict]:
    """The bar arms (D6: RR, MWW R = 2, TR, the roster v2 deterministic members) plus A0, as
    routing_audit router entries with `arm` (label) and `decode`."""
    table = [
        {"router": BAR_ROUTER[RR], "subject": member_id(PRODUCER, BAR_ROUTER[RR]),
         "producing": PRODUCER, "ordering": BAR_ROUTER[RR], "contract": JOINT_STACK,
         "random": False, "roster": False, "kind": ROUTER_KIND[BAR_ROUTER[RR]],
         "trained": False, "arm": RR},
        {"router": BAR_ROUTER[MWW], "subject": member_id(PRODUCER, BAR_ROUTER[MWW]),
         "producing": PRODUCER, "ordering": BAR_ROUTER[MWW], "contract": JOINT_STACK,
         "random": True, "roster": False, "kind": ROUTER_KIND[BAR_ROUTER[MWW]],
         "trained": False, "replicates": MWW_R, "arm": MWW},
        {**_trained_entry(TRAINED_ORDERER), "arm": TR},
    ]
    for m in roster.members:
        if m.random:
            continue
        if m.producing != PRODUCER:
            raise ValueError(f"roster member {m.id}: producer {m.producing} != {PRODUCER}")
        table.append({"router": m.ordering, "subject": m.id, "producing": m.producing,
                      "ordering": m.ordering, "contract": m.contract, "random": False,
                      "roster": True, "kind": DETERMINISTIC, "trained": False,
                      "arm": m.ordering})
    table.append({"router": A0_ORDERING, "subject": A0_SUBJECT, "producing": PRODUCER,
                  "ordering": A0_ORDERING, "contract": A0_CONTRACT, "random": False,
                  "roster": False, "kind": STOCHASTIC, "trained": True, "arm": A0})
    for a in table:
        a["decode"] = CHOICE_V1 if a["arm"] == A0 else LOGITS_V1
    if ROUTER_KIND[BAR_ROUTER[MWW]] != STOCHASTIC or ROUTER_KIND[BAR_ROUTER[RR]] != DETERMINISTIC:
        raise AssertionError("bar router kinds changed")  # MWW R = 2, RR R = 1
    return table


def bar_labels(table: Sequence[Mapping]) -> list[str]:
    """The arms whose max is the bar (every arm except A0)."""
    return [a["arm"] for a in table if a["arm"] != A0]


def arm_replicates(table: Sequence[Mapping]) -> dict[str, int]:
    return {a["arm"]: int(a.get("replicates", 1)) if a["random"] else 1 for a in table}


# --- worker ---

_BASE: dict[tuple[str, str], Any] = {}


def _base_scorer(job: Job) -> Any:
    from .per_order_scorer import BaseScorer

    ckpt = job["checkpoints"]["orderer"]
    key = (ckpt["path"], ckpt["sha256"])
    if key not in _BASE:
        import torch

        torch.set_num_threads(1)
        base = BaseScorer.load(ckpt["path"])
        if base.sha256 != ckpt["sha256"]:
            raise ValueError(f"orderer checkpoint {ckpt['path']} has sha256 {base.sha256}, "
                             f"pinned {ckpt['sha256']}")
        _BASE[key] = base
    return _BASE[key]


def fit_rows_path(fit_dir: str | Path, world: str, episode: int) -> Path:
    return Path(fit_dir) / f"{world}__{int(episode)}.npz"


def rows_digest(phi: np.ndarray, psi: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (phi, psi):
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _a0_episode(job: Job) -> tuple[Any, RoutingTracer, float, float, RawRows]:
    from ..eval.runner import run_episode
    from .per_order_scorer import make_policy, per_order_env

    env = per_order_env(job_config(job), job["horizon"])
    set_draw_key(env, DrawKey(str(job["world"]), int(job["episode"]), int(job["replicate"])))
    tracer = RoutingTracer()
    tracer.attach(env)
    monitor = ContractMonitor(env)
    from .per_order_scorer import BASE_TS

    policy = make_policy(_base_scorer(job), A0, base_kind=job.get("base_kind", BASE_TS))
    policy.sink = RawRows()
    t0, c0 = time.perf_counter(), time.process_time()
    run_episode(env, PRODUCING_BASELINES[job["producing"]], policy, seed=job["episode"],
                action_seed=job["action_seed"], monitor=monitor)
    secs, cpu = time.perf_counter() - t0, time.process_time() - c0
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
    return env, tracer, secs, cpu, policy.sink


def _bar_episode(job: Job) -> tuple[Any, RoutingTracer, float, float]:
    """routing_audit's job path (its env per contract, tracer, contract monitor), timed around
    the episode only."""
    from ..env import TokenomicsAECEnv
    from ..eval.joint_stack import joint_stack_env
    from ..eval.runner import run_episode

    cfg = job_config(job)
    env = (joint_stack_env(cfg, job["horizon"]) if job["contract"] == JOINT_STACK
           else TokenomicsAECEnv(config=cfg, horizon=job["horizon"]))
    if getattr(env, "ordering_decode", LOGITS_V1) != LOGITS_V1:
        raise AssertionError("a bar arm must run under logits-v1")
    tracer = RoutingTracer()
    tracer.attach(env)
    monitor = ContractMonitor(env)
    prod, order = job_policies(job)
    t0, c0 = time.perf_counter(), time.process_time()
    run_episode(env, prod, order, seed=job["episode"], action_seed=job["action_seed"],
                monitor=monitor)
    secs, cpu = time.perf_counter() - t0, time.process_time() - c0
    stats = env.episode_stats
    monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=job["horizon"])
    return env, tracer, secs, cpu


def run_bar_job(job: Job) -> dict[str, str]:
    """One traced episode's row with its timings (and, for A0, its fit rows written to
    `fit_dir`), or its failure row."""
    try:
        started = time.perf_counter()
        fit = {c: "" for c in FIT_COLUMNS}
        if job["arm"] == A0:
            env, tracer, secs, cpu, sink = _a0_episode(job)
            phi, psi = sink.arrays()
            path = fit_rows_path(job["fit_dir"], job["world"], job["episode"])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, phi=phi, psi=psi)
            fit = {"fit_phi_rows": str(len(phi)), "fit_psi_rows": str(len(psi)),
                   "fit_rows_sha256": rows_digest(phi, psi)}
        else:
            env, tracer, secs, cpu = _bar_episode(job)
        stats = env.episode_stats
        row = row_to_dict(row_from_stats(job["world"], job["size"], job["regime"],
                                         job["episode"], job["subject"], job["replicate"],
                                         stats))
        row.update(trace_cells(tracer))
        row.update({"arm": job["arm"], "ordering_decode": env.ordering_decode
                    if hasattr(env, "ordering_decode") else LOGITS_V1,
                    "episode_seconds": repr(secs), "cpu_seconds": repr(cpu),
                    "job_seconds": repr(time.perf_counter() - started), **fit})
        if row["ordering_decode"] != job["decode"]:
            raise AssertionError(f"{job['arm']} ran under {row['ordering_decode']}, pinned "
                                 f"{job['decode']}")
        return row
    except Exception as exc:  # noqa: BLE001 - recorded as the job's failure, never fatal
        return job_failure(job, exc).to_dict()


def job_cost(job: Job) -> float:
    """Static longest-first estimate (one size): a torch-scored arm about 3x a rule."""
    return 3.0 if job.get("checkpoints") else 1.0


def longest_first(jobs: Sequence[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: -job_cost(j))


# --- pure: outcomes, waves, the bar table ---


@dataclass(frozen=True)
class WorldOutcome:
    world: str
    complete: bool
    bar_dead_arms: tuple[str, ...] = ()
    a0_dead: bool = False
    bar_arm: str | None = None
    bar_fill: float | None = None


def _f(s: str) -> float | None:
    return None if s in ("", "None") else float(s)


def world_outcome(world: str, episodes: Sequence[int], rows: Sequence[Mapping[str, str]],
                  replicates: Mapping[str, int], bar: Sequence[str]) -> WorldOutcome:
    """A world's filters from its rows: complete only when every (episode, arm, replicate) has
    a row; the bar arm is the arm with the best mean units over the world's episodes (MWW: its
    replicate mean; first in arm order on a tie) and the bar fill its mean fill rate."""
    have = {(int(d["episode"]), d["arm"], int(d["replicate"])): d for d in rows
            if d["world"] == world}
    want = {(ep, a, r) for ep in episodes for a, n in replicates.items() for r in range(n)}
    if not want <= set(have):
        return WorldOutcome(world, False)
    dead = tuple(a for a in bar if any(have[(ep, a, r)]["dead"] == "true" for ep in episodes
                                       for r in range(replicates[a])))
    a0_dead = any(have[(ep, A0, 0)]["dead"] == "true" for ep in episodes)
    means = {a: fmean(fmean(float(have[(ep, a, r)]["units"]) for r in range(replicates[a]))
                      for ep in episodes) for a in bar}
    best = max(bar, key=lambda a: (means[a], -list(bar).index(a)))
    fills = []
    for ep in episodes:
        for r in range(replicates[best]):
            d = have[(ep, best, r)]
            o, f = _f(d["ordered_units"]), _f(d["fulfilled_units"])
            if o and f is not None:
                fills.append(f / o)
    return WorldOutcome(world, True, dead, a0_dead, best, fmean(fills) if fills else None)


def world_status(c: Candidate, o: WorldOutcome | None) -> str:
    if c.screened is not None:
        return c.screened
    if o is None:
        return UNPLAYED
    if not o.complete:
        return INCOMPLETE
    if c.source == FRESH_SOURCE:
        if o.bar_dead_arms:
            return DROPPED_BAR_DEAD
        if o.bar_fill is not None and o.bar_fill >= SATURATION_FILL:
            return DROPPED_SATURATED
    return REPLACED_A0 if o.a0_dead else TRAINING


@dataclass
class Selection:
    """Where the fixed order stands: `wave` = candidates still to play; `status` is None while
    waves remain, else complete / no_feasible_start / invalid."""

    status: str | None
    training: list[Candidate] = field(default_factory=list)
    wave: list[Candidate] = field(default_factory=list)
    replaced: int = 0
    considered: list[Candidate] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)
    reason: str = ""


def select(order: Sequence[Candidate], outcomes: Mapping[str, WorldOutcome], *,
           target: int = TARGET_WORLDS, min_feasible: int = MIN_FEASIBLE,
           max_replaced: int | None = None) -> Selection:
    """Walk the fixed order (D6 "A0 infeasible"): every atlas candidate, then draws while
    accepted + pending < target. A world on which A0 dies is replaced by the next draw; more
    than `max_replaced` (1/4 of the target) replaced, or fewer than `min_feasible` training
    worlds once the order is exhausted -> no_feasible_start. A world with missing rows (a
    contract failure) -> invalid."""
    cap = int(math.floor(target * MAX_REPLACED_SHARE)) if max_replaced is None else max_replaced
    sel = Selection(None)
    pending: list[Candidate] = []
    for c in order:
        if c.screened is not None:
            sel.statuses[c.world] = c.screened
            continue
        if c.source == FRESH_SOURCE and len(sel.training) + len(pending) >= target:
            break
        st = world_status(c, outcomes.get(c.world))
        sel.considered.append(c)
        sel.statuses[c.world] = st
        if st == UNPLAYED:
            pending.append(c)
        elif st == TRAINING:
            sel.training.append(c)
        elif st == REPLACED_A0:
            sel.replaced += 1
        elif st == INCOMPLETE:
            sel.status, sel.reason = INVALID, f"{c.world} has missing rows"
            return sel
        if sel.replaced > cap:
            sel.status = NO_FEASIBLE_START
            sel.reason = (f"{sel.replaced} candidate training worlds need replacement because "
                          f"A0 dies (> {cap} = 1/4 of {target})")
            return sel
    if pending:
        sel.wave = pending
        return sel
    if len(sel.training) >= target:
        sel.status = COMPLETE
        sel.training = sel.training[:target]
    elif len(sel.training) >= min_feasible:
        sel.status = COMPLETE
        sel.reason = f"draw order exhausted at {len(sel.training)} feasible worlds (>= {min_feasible})"
    else:
        sel.status = NO_FEASIBLE_START
        sel.reason = (f"only {len(sel.training)} feasible worlds can be drawn (< {min_feasible})")
    return sel


def scheduled_demand_per_cycle(c: Candidate, episode: int, horizon: int) -> float:
    """D_we: mean scheduled consumer demand (units) per cycle over the horizon, from the
    schedule the env builds for the episode seed."""
    cycles = realized_demand(c.point.config(), episode, horizon)
    return fmean(sum(q for _, q in cyc) for cyc in cycles)


def bar_rows(training: Sequence[Candidate], dicts: Sequence[Mapping[str, str]],
             table: Sequence[Mapping], horizon: int,
             monitor: set[tuple[str, int]] | None = None,
             demand: Callable[[Candidate, int, int], float] = scheduled_demand_per_cycle
             ) -> list[dict]:
    """One row per training world-episode (`BAR_COLUMNS`): B_we = max over the bar arms (MWW:
    replicate mean), each arm's units, A0's units and guards (dead, run-down r_end < X r_start,
    the consumers it starves below Y), its holding and cost, and D_we."""
    reps = arm_replicates(table)
    bar = bar_labels(table)
    by = {(d["world"], int(d["episode"]), d["arm"], int(d["replicate"])): d for d in dicts}
    out = []
    for c in training:
        for ep in c.episodes:
            units = {a: fmean(float(by[(c.world, ep, a, r)]["units"]) for r in range(reps[a]))
                     for a in bar}
            best = max(bar, key=lambda a: (units[a], -bar.index(a)))
            a0 = by[(c.world, ep, A0, TRAIN_REPLICATE)]
            fills = json.loads(a0["consumer_fill_rate"])
            starved = sorted(k for k, v in fills.items() if v is not None and v < Y_STARVATION)
            rs, re_ = _f(a0["r_start"]), _f(a0["r_end"])
            row = {"world": c.world, "size": c.point.size, "source": c.source,
                   ORDER_INDEX: c.order, "episode": ep, "regime": a0["regime"], "rho": c.rho,
                   "bar": units[best], "bar_arm": best,
                   **{f"units_{a}": units[a] for a in bar},
                   **{f"units_{MWW}_r{r}": float(by[(c.world, ep, MWW, r)]["units"])
                      for r in range(reps[MWW])},
                   "a0_units": float(a0["units"]), "a0_dead": a0["dead"] == "true",
                   "a0_r_start": rs, "a0_r_end": re_,
                   "a0_rundown": bool(rs is not None and re_ is not None
                                      and re_ < X_RUNDOWN * rs),
                   "a0_starved_consumers": json.dumps(starved),
                   "a0_min_fill": min((v for v in fills.values() if v is not None), default=None),
                   "a0_holding_units_per_cycle": _f(a0["holding_units_per_cycle"]),
                   "a0_episode_seconds": float(a0["episode_seconds"]),
                   "demand_per_cycle": demand(c, ep, horizon),
                   "demand_digest": a0["demand_digest"], "shock_digest": a0["shock_digest"],
                   "monitor": (c.world, ep) in (monitor or set())}
            out.append(row)
    return out


def bar_columns(table: Sequence[Mapping]) -> tuple[str, ...]:
    bar = bar_labels(table)
    return (("world", "size", "source", ORDER_INDEX, "episode", "regime", "rho", "bar",
             "bar_arm") + tuple(f"units_{a}" for a in bar)
            + tuple(f"units_{MWW}_r{r}" for r in range(arm_replicates(table)[MWW]))
            + ("a0_units", "a0_dead", "a0_r_start", "a0_r_end", "a0_rundown",
               "a0_starved_consumers", "a0_min_fill", "a0_holding_units_per_cycle",
               "a0_episode_seconds", "demand_per_cycle", "demand_digest", "shock_digest",
               "monitor"))


def timing_rows(dicts: Sequence[Mapping[str, str]]) -> list[dict]:
    return [{"world": d["world"], "episode": d["episode"], "arm": d["arm"],
             "subject": d["subject"], "replicate": d["replicate"],
             "ordering_decode": d["ordering_decode"], **{c: d[c] for c in TIMING_COLUMNS},
             "dead": d["dead"]} for d in dicts]


TIMING_TABLE = ("world", "episode", "arm", "subject", "replicate", "ordering_decode",
                *TIMING_COLUMNS, "dead")


def world_costs(world: str, dicts: Sequence[Mapping[str, str]]) -> dict[str, float]:
    """Mean episode seconds per arm on `world` (task 5.2's longest-first input)."""
    by: dict[str, list[float]] = {}
    for d in dicts:
        if d["world"] == world:
            by.setdefault(d["arm"], []).append(float(d["episode_seconds"]))
    return {a: fmean(v) for a, v in sorted(by.items())}


# --- schedule (recorded seed stream) ---


def stream_seed(*parts: object) -> int:
    """A 64-bit seed of `SCHEDULE_STREAM` and `parts` (sha256)."""
    digest = hashlib.sha256("|".join(str(p) for p in (SCHEDULE_STREAM, *parts)).encode()
                            ).hexdigest()
    return int(digest[:16], 16)


def _pick(world_episodes: Sequence[tuple[str, int]], k: int, seed: int) -> list[tuple[str, int]]:
    """k world-episodes on k distinct worlds (k <= worlds), one episode each, drawn with
    `seed`; in draw order."""
    worlds = sorted({w for w, _ in world_episodes}, key=[w for w, _ in world_episodes].index)
    if k > len(worlds):
        raise ValueError(f"{k} distinct worlds asked of {len(worlds)}")
    eps = {w: [e for x, e in world_episodes if x == w] for w in worlds}
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(worlds), size=k, replace=False):
        w = worlds[int(i)]
        out.append((w, int(eps[w][int(rng.integers(len(eps[w])))])))
    return out


def monitor_set(world_episodes: Sequence[tuple[str, int]],
                size: int = MONITOR_WORLD_EPISODES) -> list[tuple[str, int]]:
    """D6's training-monitor set: `size` fixed training world-episodes on distinct worlds."""
    return _pick(world_episodes, min(size, len({w for w, _ in world_episodes})),
                 stream_seed("monitor"))


def minibatch(world_episodes: Sequence[tuple[str, int]], opt_seed: int, generation: int,
              size: int = MINIBATCH) -> list[tuple[str, int]]:
    """The CRN minibatch of generation `generation` (1-based) of optimiser seed `opt_seed`: a pure
    function of the training world-episodes, so a resume (and task 5.2) recomputes it."""
    if generation < 1:
        raise ValueError("generations are 1-based")
    return _pick(world_episodes, size, stream_seed("minibatch", size, opt_seed, generation))


def probe_minibatches(world_episodes: Sequence[tuple[str, int]], size: int = PROBE_MINIBATCH
                      ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """D7's gradient probe: two disjoint minibatches of `size` (on 2·size distinct worlds)."""
    both = _pick(world_episodes, 2 * size, stream_seed("probe", size))
    return both[:size], both[size:]


def schedule_doc(world_episodes: Sequence[tuple[str, int]], *, generations: int = GENERATIONS,
                 size: int = MINIBATCH) -> dict:
    def fmt(we: Iterable[tuple[str, int]]) -> list[dict]:
        return [{"world": w, "episode": e} for w, e in we]

    k = min(size, len({w for w, _ in world_episodes}))
    pk = min(PROBE_MINIBATCH, len({w for w, _ in world_episodes}) // 2)
    doc = {
        "format": SCHEDULE_FORMAT, "stream": SCHEDULE_STREAM,
        "seed_rule": "int(sha256('|'.join([stream, purpose, ...]))[:16], 16) -> "
                     "numpy default_rng; k distinct worlds without replacement, then one of the "
                     "world's training episodes",
        "training_world_episodes": len(world_episodes),
        "monitor": {"replicates": MONITOR_REPLICATES, "seed": stream_seed("monitor"),
                    "world_episodes": fmt(monitor_set(world_episodes))},
        "minibatch_size": k, "generations": generations,
        "train_replicate": TRAIN_REPLICATE,
        "minibatches": {str(s): [fmt(minibatch(world_episodes, s, g, k))
                                 for g in range(1, generations + 1)] for s in OPT_SEEDS},
        "probe": {"size": pk, "seed": stream_seed("probe", pk),
                  "minibatches": [fmt(m) for m in probe_minibatches(world_episodes, pk)]
                  if pk else [[], []]},
    }
    doc["content_hash"] = content_hash(doc)
    return doc


# --- normaliser fit ---


def load_fit_rows(fit_dir: Path, world_episodes: Sequence[tuple[str, int]],
                  digests: Mapping[tuple[str, int], str] | None = None) -> RawRows:
    """A0's raw rows of exactly these world-episodes (each file's arrays re-hashed against the
    recorded digest when given)."""
    rows = RawRows()
    for w, ep in world_episodes:
        with np.load(fit_rows_path(fit_dir, w, ep)) as z:
            phi, psi = np.asarray(z["phi"]), np.asarray(z["psi"])
        if digests is not None and rows_digest(phi, psi) != digests[(w, ep)]:
            raise ValueError(f"fit rows of {w} episode {ep} do not match their recorded digest")
        rows.phi.append(phi.reshape(-1, N_PHI))
        rows.psi.append(psi.reshape(-1, N_PSI))
    return rows


def column_report(rows: RawRows, nm: FeatureNormaliser) -> list[dict]:
    """Per column: centre, quantiles, std, scale, the zero-spread rule used, and the share of
    rows clipped at +-clip (the D5 / carry-forward "Task 5.1" report)."""
    phi, psi = rows.arrays()
    out = []
    for block, x, names, med, scale in (("phi", phi, PHI_FEATURES, nm.phi_median, nm.phi_scale),
                                        ("psi", psi, PSI_FEATURES, nm.psi_median, nm.psi_scale)):
        for j, name in enumerate(names):
            col = x[:, j]
            q25, q75 = (float(v) for v in np.percentile(col, [25.0, 75.0]))
            sd = 0.0 if np.ptp(col) == 0 else float(np.std(col))  # as FeatureNormaliser.fit
            rule = "iqr" if q75 - q25 > 0 else ("std" if sd > 0 else "unit")
            if scale[j] != (q75 - q25 if rule == "iqr" else sd if rule == "std" else 1.0):
                rule = "floor"  # the registered scale floor of a transformed count
            z = (col - med[j]) / scale[j]
            lo, hi = float(np.mean(z < -nm.clip)), float(np.mean(z > nm.clip))
            flags = []
            if lo + hi >= HEAVY_CLIP_SHARE:
                flags.append("heavy_clip")
            if rule == "unit":
                flags.append("constant_raw_units")
            out.append({"block": block, "index": j, "feature": name, "median": med[j],
                        "q25": q25, "q75": q75, "iqr": q75 - q25, "std": sd,
                        "scale": scale[j], "rule": rule, "clip_low": lo, "clip_high": hi,
                        "clip_fraction": lo + hi, "rows": int(len(col)),
                        "flags": ";".join(flags)})
    return out


REPORT_COLUMNS = ("block", "index", "feature", "median", "q25", "q75", "iqr", "std", "scale",
                  "rule", "clip_low", "clip_high", "clip_fraction", "rows", "flags")


def static_psi_values(cfg: Any) -> dict[str, list[float]]:
    """The config-fact psi columns for every product of the world's process sheets (the values
    the product-stream view reports; `domain/metrics`)."""
    from ..domain import metrics

    sheets = {s.product: s for s in cfg.process_sheets}
    elig = declarers(cfg)
    out: dict[str, list[float]] = {k: [] for k in STATIC_PSI}
    for x, s in sorted(sheets.items()):
        out["product_depth"].append(float(metrics.product_depth(x, sheets)))
        out["recipe_input_count"].append(float(metrics.recipe_input_count(s)))
        out["recipe_input_units"].append(float(metrics.recipe_input_units(s)))
        out["recipe_capacity_type_count"].append(float(metrics.recipe_capacity_type_count(s)))
        out["eligible_producer_count"].append(float(elig.get(str(x), 0)))
    return out


def static_psi_check(nm: FeatureNormaliser, configs: Mapping[str, Any]) -> list[dict]:
    """Per static psi column and world: the normalised values and the share that clips, e.g. on
    the 42-producer reads (the 12-producer-trained policy runs there unretrained). The raw
    config counts go through the policy's input transform (`transform_psi`: log1p on
    `eligible_producer_count`) before the normaliser, as the policy's rows do; `raw_*` stay
    untransformed. `at_clip` = the share with |z| >= clip (pinned at the clip)."""
    out = []
    transforms = input_transforms()
    for wid, cfg in sorted(configs.items()):
        vals = static_psi_values(cfg)
        n = len(vals[STATIC_PSI[0]])
        psi = np.zeros((n, N_PSI))
        for name in STATIC_PSI:
            psi[:, PSI_FEATURES.index(name)] = vals[name]
        tpsi = transform_psi(psi)
        for name in STATIC_PSI:
            j = PSI_FEATURES.index(name)
            raw = psi[:, j]
            z = (tpsi[:, j] - nm.psi_median[j]) / nm.psi_scale[j]
            out.append({"world": wid, "feature": name,
                        "transform": transforms.get(f"psi.{name}", ""),
                        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                        "z_min": float(z.min()), "z_max": float(z.max()),
                        "clip_fraction": float(np.mean(np.abs(z) > nm.clip)),
                        "at_clip": float(np.mean(np.abs(z) >= nm.clip))})
    return out


STATIC_COLUMNS = ("world", "feature", "transform", "raw_min", "raw_max", "z_min", "z_max",
                  "clip_fraction", "at_clip")


def reads_42_configs(atlas: Mapping) -> dict[str, Any]:
    """The 42-producer loaded non-collapse atlas worlds of every split (the reads' 42-producer
    worlds and the train ones), rebuilt from their records."""
    from ..eval.atlas import COLLAPSE
    from ..eval.teacher_audit import world_view

    out = {}
    for entry in atlas["worlds"]:
        v = world_view(entry)
        if v["size"] == 42 and float(v["record"]["coordinates"]["rho"]) > RHO_LOADED \
                and v["class"] != COLLAPSE:
            out[v["world"]] = config_from_record(v["record"])
    return out


def fit_normaliser(rows: RawRows) -> tuple[FeatureNormaliser, list[dict]]:
    nm = FeatureNormaliser.fit(rows)
    return nm, column_report(rows, nm)


def fit_rows_sha256(world_episodes: Sequence[tuple[str, int]],
                    digests: Mapping[tuple[str, int], str]) -> str:
    """The fit's inputs: SHA-256 over the ordered (world, episode, per-file row digest) list."""
    return hashlib.sha256(_canonical([[w, int(ep), digests[(w, ep)]]
                                      for w, ep in world_episodes])).hexdigest()


def normaliser_doc(nm: FeatureNormaliser, report: Sequence[Mapping], static: Sequence[Mapping],
                   world_episodes: Sequence[tuple[str, int]], dry_run: bool,
                   inputs: Mapping[str, str] | None = None) -> dict:
    flags = {
        "heavy_clip": [f"{r['block']}[{r['index']}] {r['feature']} ({r['clip_fraction']:.3f})"
                       for r in report if "heavy_clip" in r["flags"]],
        "constant_raw_units": [f"{r['block']}[{r['index']}] {r['feature']}" for r in report
                               if "constant_raw_units" in r["flags"]],
        "history_completion_rate_clip": next(
            r["clip_fraction"] for r in report
            if r["block"] == "phi" and r["index"] == PHI_FEATURES.index("completion_rate", 31)),
        "static_psi_clip_42": sorted({f"{s['feature']} on {s['world']} ({s['at_clip']:.3f})"
                                      for s in static if s["at_clip"] > 0}),
    }
    return {"normaliser": nm.to_dict(), "sha256": nm.sha256, "prereg": NORMALISER_PREREG,
            "rules": {f"{r['block']}[{r['index']}] {r['feature']}": r["rule"] for r in report},
            "fit_world_episodes": len(world_episodes), "phi_rows": report[0]["rows"],
            **dict(inputs or {}),
            "psi_rows": report[-1]["rows"], "flags": flags, "dry_run": dry_run,
            "note": "dry-run fit (not the pre-registered horizon-100 fit)" if dry_run else ""}


def write_normaliser(out: Path, fit_dir: Path, world_episodes: Sequence[tuple[str, int]],
                     digests: Mapping[tuple[str, int], str], training_sha256: str,
                     configs_42: Mapping[str, Any], dry_run: bool) -> dict:
    """Fit on exactly `world_episodes`' A0 rows (each re-hashed) and write the normaliser files;
    `normaliser.json` names its inputs: `fit_rows_sha256` and `training_worlds_sha256`."""
    rows = load_fit_rows(fit_dir, world_episodes, digests)
    nm, report = fit_normaliser(rows)
    static = static_psi_check(nm, configs_42)
    doc = normaliser_doc(nm, report, static, world_episodes, dry_run, {
        "fit_rows_sha256": fit_rows_sha256(world_episodes, digests),
        "training_worlds_sha256": training_sha256})
    (out / "normaliser.json").write_text(json.dumps(doc, indent=1, sort_keys=True,
                                                    allow_nan=False) + "\n")
    write_csv(out / "normaliser_report.csv", REPORT_COLUMNS, report)
    write_csv(out / "normaliser_static_42.csv", STATIC_COLUMNS, static)
    return doc


# --- driver ---


def candidate_rows(order: Sequence[Candidate], sel: Selection,
                   outcomes: Mapping[str, WorldOutcome]) -> list[dict]:
    out = []
    training = {c.world for c in sel.training}
    for c in order:
        if c.world not in sel.statuses:
            continue
        o = outcomes.get(c.world)
        out.append({ORDER_INDEX: c.order, "world": c.world, "source": c.source,
                    "seed": c.point.seed, "rho": c.rho, "status": sel.statuses[c.world],
                    "training": c.world in training,
                    "episodes": json.dumps(list(c.episodes)),
                    "bar_arm": o.bar_arm if o else "", "bar_fill": o.bar_fill if o else "",
                    "bar_dead_arms": json.dumps(list(o.bar_dead_arms)) if o else "",
                    "a0_dead": o.a0_dead if o else ""})
    return out


def drawn_records(order: Sequence[Candidate], sel: Selection) -> dict[str, dict]:
    """The world record (levers, seed, measured coordinates, config hash) of every world the
    walk reached, whatever its status: atlas worlds keep the atlas's record."""
    return {c.world: dict(c.record) if c.record is not None else world_record(c.point)
            for c in order if c.world in sel.statuses}


CANDIDATE_COLUMNS = (ORDER_INDEX, "world", "source", "seed", "rho", "status", "training",
                     "episodes", "bar_arm", "bar_fill", "bar_dead_arms", "a0_dead")


def training_doc(training: Sequence[Candidate], dicts: Sequence[Mapping[str, str]],
                 pins: Mapping[str, Any]) -> dict:
    worlds = []
    for c in training:
        rec = dict(c.record) if c.record is not None else world_record(c.point)
        worlds.append({"world": c.world, "source": c.source, ORDER_INDEX: c.order,
                       "seed": c.point.seed, "episodes": list(c.episodes), "rho": c.rho,
                       "shock_window": list(c.point.shock_window), "record": rec,
                       "cost_s": world_costs(c.world, dicts)})
    doc = {"format": WORLDS_FORMAT, "size": SIZE, "producer": PRODUCER,
           "horizon": pins["horizon"], "dry_run": pins["dry_run"],
           "fresh_atlas": FRESH_ATLAS, "train_stage": TRAIN_STAGE,
           "train_replicate": TRAIN_REPLICATE, "worlds": worlds}
    doc["content_hash"] = content_hash(doc)
    return doc


def real_run_problems(*, atlas: Mapping, roster: Roster, horizon: int, source_commit: str,
                      orderer_sha: str, filters: Mapping[str, object]) -> list[str]:
    v2 = load_roster(ROSTER_V2)
    problems = [msg for bad, msg in (
        (atlas.get("content_hash") != REGISTERED_ATLAS_HASH,
         f"atlas {str(atlas.get('content_hash'))[:12]} is not the registered EXP-0052 "
         f"reclassified atlas {REGISTERED_ATLAS_HASH[:12]}"),
        (not roster.frozen, f"roster {roster.version[:12]} is not frozen"),
        ((roster.name, roster.version) != (v2.name, v2.version),
         f"roster {roster.name} {roster.version[:12]} is not the packaged frozen roster v2 "
         f"{v2.version[:12]} (D6: B_we over the roster v2 deterministic members)"),
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
        (orderer_sha != ROLE_CHECKPOINTS["orderer"],
         f"orderer checkpoint {orderer_sha[:12]} is not the registered "
         f"{ROLE_CHECKPOINTS['orderer'][:12]}"),
    ) if bad]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


def _grid(cands: Sequence[Candidate]) -> dict[str, dict]:
    return {c.world: {"size": c.point.size, "regime": c.entry().regime,
                      "episodes": list(c.episodes)} for c in cands}


def _order_digest(order: Sequence[Candidate]) -> str:
    return hashlib.sha256(_canonical([
        [c.order, c.source, c.world, c.point.seed, c.point.levers.to_dict(),
         list(c.point.shock_window), list(c.episodes), c.screened] for c in order])).hexdigest()


def run_bar(*, atlas_path: str | Path, out_dir: str | Path, orderer_ckpt: str, roster: Roster,
            workers: int = 1, horizon: int = HORIZON, atlas_worlds: int | None = None,
            target: int | None = None, min_feasible: int | None = None,
            pool: int | None = None, episodes_per_world: int | None = None,
            shock_profile: str | None = None, source_commit: str = "", dry_run: bool = False,
            base_kind: str = "ts",
            run_job_fn: Callable[[Job], dict[str, str]] = run_bar_job,
            log: Callable[[str], None] = print) -> int:
    """Run (or resume) the bar precompute into `out_dir`; return the exit status (0,
    `CONTRACT_INVALID`, or `NO_FEASIBLE_EXIT` on no_feasible_start). `base_kind` = the base A0
    plays (`ts`, EXP-0058; `mww`, the re-based step 2 of task 7.6); pinned in `per_order`."""
    if shock_profile not in (None, SHOCKS_I_SMOKE):
        raise ValueError(f"--shock-profile takes {SHOCKS_I_SMOKE} only (a dry run's window)")
    atlas_path = Path(atlas_path)
    atlas = load_atlas(atlas_path)
    out = Path(out_dir)
    pin = checked_orderer(out, orderer_ckpt, log)
    filters = {"atlas_worlds": atlas_worlds, "target": target, "min_feasible": min_feasible,
               "pool": pool, "episodes_per_world": episodes_per_world,
               "shock_profile": shock_profile}
    if not dry_run:
        problems = real_run_problems(atlas=atlas, roster=roster, horizon=horizon,
                                     source_commit=source_commit, orderer_sha=pin["sha256"],
                                     filters=filters)
        if problems:
            raise ValueError("not a valid bar-precompute run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    n_target = TARGET_WORLDS if target is None else target
    n_min = MIN_FEASIBLE if min_feasible is None else min_feasible
    n_eps = EPISODES_PER_WORLD if episodes_per_world is None else episodes_per_world
    atl = atlas_candidates(atlas, atlas_worlds, shock_profile)
    if n_eps != EPISODES_PER_WORLD:
        atl = [replace(c, episodes=train_episode_seeds(c.world, n_eps)) for c in atl]
    if not dry_run and len(atl) != ATLAS_WORLDS:
        raise ValueError(f"the atlas holds {len(atl)} loaded non-collapse 12-producer train "
                         f"worlds, D6 registers {ATLAS_WORLDS}")
    reserved = reserved_union()
    clash = sorted(c.world for c in atl if reserved & set(c.episodes))
    if clash:
        raise ValueError(f"training episode seeds of {clash} are reserved seeds")
    order = atl + fresh_candidates(len(atl), FRESH_POOL if pool is None else pool,
                                   shock_profile, reserved, n_eps)
    table = bar_arms(roster)
    reps = arm_replicates(table)
    bar = bar_labels(table)
    playable = [c for c in order if c.screened is None]
    grid = _grid(playable)
    expected: set[Key] = expected_keys(grid, table, 1)
    from .per_order_scorer import BaseScorer, make_policy, per_order_pins

    a0_pins = per_order_pins(make_policy(BaseScorer.load(pin["path"]), A0,
                                            base_kind=base_kind))
    source = source_fingerprint(source_commit or None)
    pins = {
        "driver": DRIVER, "mode": BAR, "horizon": horizon, "dry_run": dry_run,
        "atlas": {"path": str(atlas_path), "sha256": sha256_file(atlas_path),
                  "content_hash": atlas["content_hash"]},
        "roster_version": roster.version, "checkpoints": {"orderer": pin["sha256"]},
        "producer": PRODUCER, "arms": [a["arm"] for a in table], "arm_table": table,
        **decode_pins(CHOICE_V1), "decode_by_arm": {a["arm"]: a["decode"] for a in table},
        "per_order": a0_pins["per_order"], "shock_profile": shock_profile,
        "selection": {"target": n_target, "min_feasible": n_min,
                      "max_replaced_share": MAX_REPLACED_SHARE, "rho_loaded": RHO_LOADED,
                      "saturation_fill": SATURATION_FILL, "fresh_atlas": FRESH_ATLAS,
                      "fresh_pool": FRESH_POOL if pool is None else pool,
                      "train_stage": TRAIN_STAGE, "episodes_per_world": n_eps,
                      "atlas_worlds": len(atl)},
        "order_sha256": _order_digest(order),
        "guards": {"x_rundown": X_RUNDOWN, "y_starvation": Y_STARVATION,
                   "side_effect_params": SIDE_EFFECT_PARAMS.version},
        "schedule": {"stream": SCHEDULE_STREAM, "monitor": MONITOR_WORLD_EPISODES,
                     "monitor_replicates": MONITOR_REPLICATES, "minibatch": MINIBATCH,
                     "generations": GENERATIONS, "opt_seeds": list(OPT_SEEDS),
                     "probe_minibatch": PROBE_MINIBATCH},
        "normaliser": NORMALISER_PREREG,
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
    }
    old_spec, done = _prepare_resume(out, pins, grid, expected)
    spec = _open_spec(out, old_spec, workers, {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "bar",
        "acceptance_read": False, "pins": pins, "inputs": {"orderer": pin}, "source": source,
        "pre_registered": {
            "design": "add-m4-orderer-redesign D6 (training worlds, bar, A0 infeasible), "
                      "D5 (normaliser), D2 (producer entity_planner, user decision 2026-10-30)",
            "bar": "B_we = max over RR, MWW (R = 2 mean), TR and the roster v2 deterministic "
                   "members of consumer units on the world-episode",
            "a0": "zero-residual per-order policy (= TS), choice-v1, DrawKey(world, episode, 0)",
            "normaliser": NORMALISER_PREREG,
            "prereg_file": "es_prereg_m4_v1.json is frozen by task 5.3; it copies "
                           "spec.pins.normaliser and the fitted normaliser.json sha256",
        }})
    ensure_failures_file(out)
    episodes = out / "episodes.csv"
    fit_dir = out / FIT_DIR
    by_world = {c.world: c for c in playable}
    if done:
        log(f"[order_es_worlds] resume: {len(done)} rows already recorded")
    while True:
        dicts = read_episode_dicts(episodes)
        rows_by_world: dict[str, list[dict[str, str]]] = {}
        for d in dicts:
            rows_by_world.setdefault(d["world"], []).append(d)
        recorded_failures = read_failures(out)
        n_failures = len(recorded_failures)
        failed = {f["world"] for f in recorded_failures if f.get("world")}
        outcomes = {}
        for w, rows in rows_by_world.items():
            outcomes[w] = world_outcome(w, by_world[w].episodes, rows, reps, bar)
        for w in failed:
            if w in by_world and not (w in outcomes and outcomes[w].complete):
                outcomes[w] = WorldOutcome(w, False)
        sel = select(order, outcomes, target=n_target, min_feasible=n_min)
        if sel.status is not None or not sel.wave:
            break
        wave = [c.entry() for c in sel.wave]
        jobs = build_jobs(wave, table, 1, horizon, None, {"orderer": pin})
        by_subject = {a["subject"]: a for a in table}
        for j in jobs:
            a = by_subject[j["subject"]]
            j.update({"arm": a["arm"], "decode": a["decode"]})
            if a["arm"] == A0:
                j["fit_dir"] = str(fit_dir)
                if base_kind != "ts":
                    j["base_kind"] = base_kind
        log(f"[order_es_worlds] wave: {len(sel.wave)} world(s) from order "
            f"{sel.wave[0].order}, {len(sel.training)} training so far, {sel.replaced} replaced")
        done_keys = {job_key(d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))
                     for d in dicts}
        try:
            execute(out, spec, done_keys, dicts, longest_first(jobs), workers, run_job_fn, log,
                    columns=COLUMNS)
        except DigestMismatch as exc:  # recorded as a run row by execute; the run ends invalid
            log(f"[order_es_worlds] INVALID: {exc}")
            sel = Selection(INVALID, reason=str(exc))
            break
        if (len(read_episode_dicts(episodes)) == len(dicts)
                and len(read_failures(out)) == n_failures):
            sel = Selection(INVALID, reason="a wave made no progress")  # internal invariant
            break

    dicts = read_episode_dicts(episodes)
    rows = read_rows(episodes)
    keys = {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
    if not keys <= expected:  # _prepare_resume checked the old rows; a new one outside = bug
        raise AssertionError("rows outside the candidate grid")
    failures = read_failures(out)
    played = {c.world for c in sel.considered if sel.statuses.get(c.world) != UNPLAYED}
    want = {k for k in expected if k[0] in played}
    missing = want - keys
    write_csv(out / "candidates.csv", CANDIDATE_COLUMNS, candidate_rows(order, sel, outcomes))
    (out / "world_records.json").write_text(json.dumps(drawn_records(order, sel), indent=1,
                                                       sort_keys=True) + "\n")
    write_csv(out / "timings.csv", TIMING_TABLE, timing_rows(dicts))
    summary: dict[str, Any] = {
        "status": sel.status, "reason": sel.reason, "training_worlds": len(sel.training),
        "replaced_a0_dead": sel.replaced, "considered": len(sel.considered),
        "by_status": {s: sum(1 for v in sel.statuses.values() if v == s)
                      for s in sorted(set(sel.statuses.values()))},
        "contract_failures": len(failures), "missing_rows": len(missing), "dry_run": dry_run}
    invalid = bool(failures or missing) or sel.status == INVALID
    code = 0
    if invalid:
        spec["status"] = INVALID
        spec["result"] = {"status": INVALID, "contract_failures": len(failures),
                          "missing_rows": len(missing), "reason": sel.reason}
        status_txt = (f"INVALID ({len(failures)} contract failure(s), {len(missing)} missing "
                      f"row(s))")
        code = CONTRACT_INVALID
    elif sel.status == NO_FEASIBLE_START:
        spec["status"] = NO_FEASIBLE_START
        spec["result"] = {"status": NO_FEASIBLE_START, "reason": sel.reason,
                          "replaced_a0_dead": sel.replaced,
                          "training_worlds": len(sel.training)}
        status_txt = f"NO FEASIBLE START: {sel.reason} (to the user)"
        code = NO_FEASIBLE_EXIT
    else:
        we = [(c.world, ep) for c in sel.training for ep in c.episodes]
        sched = schedule_doc(we)
        mon = {(d["world"], d["episode"]) for d in sched["monitor"]["world_episodes"]}
        brows = bar_rows(sel.training, dicts, table, horizon, mon)
        write_csv(out / "bar.csv", bar_columns(table), brows)
        tdoc = training_doc(sel.training, dicts, pins)
        (out / "training_worlds.json").write_text(json.dumps(tdoc, indent=1, sort_keys=True)
                                                  + "\n")
        (out / "schedule.json").write_text(json.dumps(sched, indent=1, sort_keys=True) + "\n")
        digests = {(d["world"], int(d["episode"])): d["fit_rows_sha256"] for d in dicts
                   if d["arm"] == A0}
        ndoc = write_normaliser(out, fit_dir, we, digests, tdoc["content_hash"],
                                reads_42_configs(atlas), dry_run)
        summary.update({"training_world_episodes": len(we),
                        "training_worlds_sha256": tdoc["content_hash"],
                        "schedule_sha256": sched["content_hash"],
                        "normaliser_sha256": ndoc["sha256"], "normaliser_flags": ndoc["flags"],
                        "mean_bar": fmean(r["bar"] for r in brows),
                        "a0_rundown_share": fmean(r["a0_rundown"] for r in brows),
                        "a0_starved_share": fmean(r["a0_starved_consumers"] != "[]"
                                                  for r in brows)})
        spec["status"] = COMPLETE
        spec["result"] = {"status": COMPLETE, "rows": len(rows), "contract_failures": 0,
                          "training_worlds": len(sel.training),
                          "training_worlds_sha256": tdoc["content_hash"],
                          "schedule_sha256": sched["content_hash"],
                          "normaliser_sha256": ndoc["sha256"], "acceptance_read": False}
        status_txt = (f"complete: {len(sel.training)} training worlds, {len(we)} world-episodes, "
                      f"{sel.replaced} replaced; normaliser {ndoc['sha256'][:12]}")
    (out / "bar_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True,
                                                     default=str) + "\n")
    _write_spec(out, spec)
    result_md = out / "RESULT.md"
    if not result_md.exists() or result_md.read_text().endswith(SKELETON_END):
        result_md.write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", status=status_txt,
            n_rows=len(rows), horizon=horizon, driver=DRIVER))
    log(f"[order_es_worlds] {'DRY RUN ' if dry_run else ''}{status_txt}")
    return code


def run_fit(*, bar_dir: str | Path, out_dir: str | Path, atlas_path: str | Path | None = None,
            log: Callable[[str], None] = print) -> int:
    """Refit the normaliser from a complete bar run's A0 rows (its training world-episodes only)
    into `out_dir`; exit 0 when the hash equals the bar's, 2 otherwise. The bar dir is read
    only."""
    bar = Path(bar_dir)
    out = Path(out_dir)
    if out.resolve() == bar.resolve():
        raise ValueError("--out-dir must differ from --bar-dir (the bar dir is never written)")
    spec = json.loads((bar / "spec.json").read_text())
    if spec.get("status") != COMPLETE:
        raise ValueError(f"{bar} is not a complete bar run (status {spec.get('status')!r})")
    tdoc = json.loads((bar / "training_worlds.json").read_text())
    if content_hash(tdoc) != tdoc["content_hash"]:
        raise ValueError("training_worlds.json does not match its content hash")
    we = [(w["world"], int(ep)) for w in tdoc["worlds"] for ep in w["episodes"]]
    dicts = read_episode_dicts(bar / "episodes.csv")
    digests = {(d["world"], int(d["episode"])): d["fit_rows_sha256"] for d in dicts
               if d["arm"] == A0}
    atlas = load_atlas(atlas_path or spec["pins"]["atlas"]["path"])
    if atlas["content_hash"] != spec["pins"]["atlas"]["content_hash"]:
        raise ValueError(f"atlas {atlas['content_hash'][:12]} is not the bar run's "
                         f"{spec['pins']['atlas']['content_hash'][:12]}")
    out.mkdir(parents=True, exist_ok=True)
    doc = write_normaliser(out, bar / FIT_DIR, we, digests, tdoc["content_hash"],
                           reads_42_configs(atlas), bool(tdoc["dry_run"]))
    recorded = json.loads((bar / "normaliser.json").read_text())
    keys = ("sha256", "fit_rows_sha256", "training_worlds_sha256")
    compared = {k: {"refit": doc[k], "bar": recorded.get(k), "match": doc[k] == recorded.get(k)}
                for k in keys}
    match = all(c["match"] for c in compared.values())
    (out / "fit.json").write_text(json.dumps({"bar_dir": str(bar), "compared": compared,
                                              "match": match}, indent=1, sort_keys=True) + "\n")
    log(f"[order_es_worlds] normaliser {doc['sha256'][:12]} "
        f"{'MATCH' if match else 'MISMATCH'} vs the bar's "
        + ", ".join(f"{k} {'=' if c['match'] else '!='}" for k, c in compared.items()))
    return 0 if match else CONTRACT_INVALID


DRY_BANNER = ("> **Dry run: not the training-world set.** Filters or a test-only profile were "
              "used; nothing here is pinned for ES.\n\n")
SKELETON_END = "Conclusions\n\nTODO.\n"
RESULT_SKELETON = """# {title}

{banner}Bar precompute for step-2 ES (add-m4-orderer-redesign D6, task 5.1), driver `{driver}`,
horizon {horizon}. Status: {status}. Rows: {n_rows}.

Outputs: `bar.csv` (B_we per training world-episode), `training_worlds.json`, `schedule.json`,
`normaliser.json` / `normaliser_report.csv`, `candidates.csv`, `timings.csv`, `bar_summary.json`.

## Conclusions

TODO.
"""


def _opt_int(s: str | None) -> int | None:
    return None if s is None else int(s)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=(BAR, FIT), default=BAR)
    p.add_argument("--atlas", help="atlas_v2.json (EXP-0052 reclassified)")
    p.add_argument("--orderer-ckpt", help="the trained orderer (EXP-0012), A0's base and TR")
    p.add_argument("--bar-dir", help="fit-normaliser: a complete bar run (read only)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--roster", default=str(DEFAULT_ROSTER))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--atlas-worlds", help="dry run: the first N atlas candidates")
    p.add_argument("--target", help=f"dry run: training worlds (real {TARGET_WORLDS})")
    p.add_argument("--min-feasible", help=f"dry run: (real {MIN_FEASIBLE})")
    p.add_argument("--pool", help=f"dry run: fresh draw pool (real {FRESH_POOL})")
    p.add_argument("--episodes-per-world", help=f"dry run: (real {EPISODES_PER_WORLD})")
    p.add_argument("--shock-profile", choices=(SHOCKS_I_SMOKE,))
    p.add_argument("--base", choices=("ts", "mww"), default="ts",
                   help="the base A0 plays (mww: the re-based step 2, task 7.6)")
    a = p.parse_args(argv)
    if a.mode == FIT:
        if not a.bar_dir:
            p.error("--mode fit-normaliser needs --bar-dir")
        return run_fit(bar_dir=a.bar_dir, out_dir=a.out_dir, atlas_path=a.atlas)
    if not a.atlas or not a.orderer_ckpt:
        p.error("--mode bar needs --atlas and --orderer-ckpt")
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    try:
        return run_bar(atlas_path=a.atlas, out_dir=a.out_dir, orderer_ckpt=a.orderer_ckpt,
                       roster=load_roster(a.roster), workers=a.workers, horizon=a.horizon,
                       atlas_worlds=_opt_int(a.atlas_worlds), target=_opt_int(a.target),
                       min_feasible=_opt_int(a.min_feasible), pool=_opt_int(a.pool),
                       episodes_per_world=_opt_int(a.episodes_per_world),
                       shock_profile=a.shock_profile, source_commit=commit,
                       dry_run=a.dry_run, base_kind=a.base)
    except (ContractRunError, DigestMismatch):  # recorded in contract_failures.csv
        return CONTRACT_INVALID


if __name__ == "__main__":
    sys.exit(main())
