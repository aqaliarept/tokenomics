"""PoC-B: deployed-path cost driver (EXP-0055; add-m3-development-audit D7).

Times the deployed portable stack (EXP-0035 producer + EXP-0012 orderer under
`joint_stack_env`) on calm `sparse_wide` family worlds, stage by stage, per decision, and fits
a log-log OLS slope per stage on three independent axes:

- producers P in {12, 24, 42} at fanout E = 2;
- fanout E in {2, 4, 7, 14} at P = 42 (14 = the whole layer);
- consumers per producer, consumer multiplier k_c in {1, 2, 4} (the world-space consumer
  lever, D10) at P = 42, E = 4.

No economy above 42 producers is ever timed (user decision, 2026-10-28): the grid builder
refuses one, naming the size, and no flag allows it.

Stages (exclusive wall time, `perf_counter`; the timers wrap calls from this driver, never
code in `domain/`):

1. `obs` - observation build: the env's `observe` (via `env.last()`) and the white-box views
   the policies read (`System.producer_entity_view`, `System.product_stream_view`);
2. `forward` - the scorers' `forward_view` (row features, tensors, MLP);
3. `decode` - decode / allocate: the policy's remaining work after its forward (the producer
   planner, action packing), the env's `decode_producing` / `decode_ordering` (with
   `allocate_suborders`), and the System applying the command (`producing_agent_action`,
   `ordering_agent_action`);
4. `domain` - the domain cycle: `System.start_ordering` and `System.complete_cycle`;
5. `harness` - everything else in the episode (PettingZoo bookkeeping, rewards, statistics,
   `reset`).

The five sum to the timed episode's wall time by construction (nested calls charge the
innermost stage only). The policy-side stages (`obs`, `forward`, `decode`) are also charged
per role (producer / orderer), because the two roles' decision counts grow differently with
fanout: pooled, one role's growth is diluted by the other's. Measures (`MEASURES`): seconds per
decision over all decisions, per role over that role's decisions, and per cycle. A
per-decision slope (pooled or per role) above 1.1 on the fanout or the consumer axis flags the
stage ("Superlinear stage"; D2's threshold; a warning, not proof); per-cycle slopes are
reported, never flagged.

Protocol: horizon 20, 3 world seeds per cell, 3 repeats (median per world), BLAS pinned to one
thread, one process per worker, workers <= usable vCPU - 2 (at least 1; affinity mask capped by
the cgroup quota, since a container may report the host's CPUs).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping, Sequence

from .baselines import Policy
from .families import CALM, SPARSE_WIDE, build_family_config, consumers_for
from .gate_cli import pin_blas
from .joint_stack import joint_stack_env
from .provenance import environment_manifest, sha256_file, source_fingerprint
from .runner import run_episode

DRIVER = "tokenomics.eval.deployed_cost"
MAX_PRODUCERS = 42  # user decision, 2026-10-28: nothing above the 12 -> 42 target
HORIZON = 20
WORLD_SEEDS = (1000, 1001, 1002)  # the family ratio sample's first three world seeds
REPEATS = 3
EPISODE_SEED = 0  # calm worlds play a fixed schedule; the seed only fixes the action spaces
SUPERLINEAR = 1.1  # D2 threshold on per-decision time
SUPERLINEAR_LABEL = "Superlinear stage"
FLAGGED_AXES = ("fanout", "consumers_per_producer")
# EXP-0035 `producer_candidate.pt`, EXP-0012 `orderer_scorer.pt` (same pins as PoC-C)
CHECKPOINTS = {
    "producer": "8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea",
    "orderer": "d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c"}

OBS, FORWARD, DECODE, DOMAIN, HARNESS = "obs", "forward", "decode", "domain", "harness"
STAGES = (OBS, FORWARD, DECODE, DOMAIN, HARNESS)


# --- grid ---


@dataclass(frozen=True)
class Cell:
    """One world shape: P producers, fanout E, consumer multiplier k_c."""

    producers: int
    fanout: int
    consumer_multiplier: int

    @property
    def id(self) -> str:
        return f"P{self.producers}-E{self.fanout}-k{self.consumer_multiplier}"

    @property
    def consumers(self) -> int:
        return consumers_for(self.producers, self.consumer_multiplier)

    @property
    def consumers_per_producer(self) -> float:
        return self.consumers / self.producers


def check_size(producers: int) -> int:
    """Refuse any economy above 42 producers, naming the size (no override exists)."""
    if producers > MAX_PRODUCERS:
        raise ValueError(f"Larger economy requested: {producers} producers; the cost audit "
                         f"times nothing above {MAX_PRODUCERS} (user decision, 2026-10-28)")
    return producers


def make_cell(producers: int, fanout: int, consumer_multiplier: int = 1) -> Cell:
    check_size(producers)
    per_layer = producers // 3
    if producers < 3 or producers % 3:
        raise ValueError(f"producers must be a positive multiple of 3, got {producers}")
    if not 1 <= fanout <= per_layer:
        raise ValueError(f"fanout must be in [1, {per_layer}] at {producers} producers, "
                         f"got {fanout}")
    if consumer_multiplier < 1:
        raise ValueError(f"consumer multiplier must be >= 1, got {consumer_multiplier}")
    return Cell(producers, fanout, consumer_multiplier)


def parse_cell(spec: str) -> Cell:
    """`P42-E4-k2` -> Cell (dry-run `--cells`); P > 42 is refused like in the grid."""
    try:
        p, e, k = spec.split("-")
        if p[0] != "P" or e[0] != "E" or k[0] != "k":
            raise ValueError
        producers, fanout, mult = int(p[1:]), int(e[1:]), int(k[1:])
    except (ValueError, IndexError):
        raise ValueError(f"cell {spec!r} is not P<producers>-E<fanout>-k<multiplier>") from None
    return make_cell(producers, fanout, mult)


def axes() -> dict[str, list[Cell]]:
    """The three D7 axes, each a list of cells in increasing order of its variable."""
    return {
        "producers": [make_cell(p, 2) for p in (12, 24, 42)],
        "fanout": [make_cell(42, e) for e in (2, 4, 7, 14)],
        "consumers_per_producer": [make_cell(42, 4, k) for k in (1, 2, 4)],
    }


def grid() -> list[Cell]:
    """The distinct cells of the three axes (8), in first-seen order."""
    out: list[Cell] = []
    for cells in axes().values():
        out += [c for c in cells if c not in out]
    return out


def axis_value(axis: str, cell: Cell) -> float:
    return {"producers": float(cell.producers), "fanout": float(cell.fanout),
            "consumers_per_producer": cell.consumers_per_producer}[axis]


# --- slope ---


def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """OLS slope of log(y) on log(x). Needs >= 2 distinct positive x and positive y."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("a slope needs two or more (x, y) points of equal count")
    if min(xs) <= 0 or min(ys) <= 0:
        raise ValueError("log-log slope needs positive x and y")
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = fmean(lx), fmean(ly)
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        raise ValueError("a slope needs two or more distinct x values")
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx


def superlinear(axis: str, slope: float, measure: str = "decision") -> bool:
    """D7: a per-decision slope > 1.1 on the fanout or the consumer axis flags the stage
    (pooled or per role; per-cycle rows are reported, never flagged)."""
    return axis in FLAGGED_AXES and measure in FLAGGED_MEASURES and slope > SUPERLINEAR


# --- stage timers ---

PROD, ORDER = "prod", "order"
ROLES = (PROD, ORDER)
POLICY_STAGES = (OBS, FORWARD, DECODE)  # charged per role too; domain / harness are per cycle


class StageTimer:
    """Exclusive wall time per stage: a stack of active (stage, role) frames; the time between
    two timer events is charged to the frame on top, so nested calls never count twice and the
    stages sum to stop - start. A frame with a role (`prod` / `order`) is also charged to
    `by_role`, so a policy-side stage's cost per producer and per orderer decision stays
    separable. Outside `start`/`stop` the wrappers only pass through."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = dict.fromkeys(STAGES, 0.0)
        self.by_role: dict[tuple[str, str], float] = {(r, s): 0.0 for r in ROLES
                                                      for s in POLICY_STAGES}
        self.calls: dict[str, int] = dict.fromkeys(STAGES, 0)
        self._stack: list[tuple[str, str | None]] = []
        self._last = 0.0

    def start(self) -> None:
        self._stack = [(HARNESS, None)]
        self._last = perf_counter()

    def stop(self) -> None:
        self._charge(perf_counter())
        self._stack = []

    def _charge(self, now: float) -> None:
        stage, role = self._stack[-1]
        self.totals[stage] += now - self._last
        if role is not None:
            self.by_role[(role, stage)] += now - self._last
        self._last = now

    def wrap(self, stage: str, fn: Callable[..., Any],
             role: str | Callable[..., str] | None = None) -> Callable[..., Any]:
        """`role` is fixed, a function of the call's arguments, or None (per-cycle work)."""

        def timed(*args: Any, **kwargs: Any) -> Any:
            if not self._stack:
                return fn(*args, **kwargs)
            self._charge(perf_counter())
            self._stack.append((stage, role(*args, **kwargs) if callable(role) else role))
            self.calls[stage] += 1
            try:
                return fn(*args, **kwargs)
            finally:
                self._charge(perf_counter())
                self._stack.pop()
        return timed


# System method -> (stage, role); every policy-side call has a fixed role
_SYSTEM_STAGES = {"producer_entity_view": (OBS, PROD), "product_stream_view": (OBS, ORDER),
                  "producing_agent_action": (DECODE, PROD),
                  "ordering_agent_action": (DECODE, ORDER),
                  "start_ordering": (DOMAIN, None), "complete_cycle": (DOMAIN, None)}


@contextmanager
def instrumented(env: Any, timer: StageTimer, models: Mapping[str, Any] | None = None
                 ) -> Iterator[None]:
    """Install the stage wrappers on this env (instance attributes), its System after every
    reset, the env module's decode functions and the models' `forward_view` (`models` = role ->
    model); restore all on exit, also after a partial install. Behaviour is unchanged: each
    wrapper calls the original with the same arguments."""
    from ..env import aec_env

    saved_decode = (aec_env.decode_producing, aec_env.decode_ordering)
    orig_reset = env.reset
    models = dict(models or {})

    def reset(*args: Any, **kwargs: Any) -> Any:
        out = orig_reset(*args, **kwargs)
        for name, (stage, role) in _SYSTEM_STAGES.items():
            setattr(env.system, name, timer.wrap(
                stage, getattr(type(env.system), name).__get__(env.system), role))
        return out

    try:
        env.reset = reset
        env.observe = timer.wrap(OBS, type(env).observe.__get__(env),
                                 lambda agent: env._role[agent][0])
        aec_env.decode_producing = timer.wrap(DECODE, saved_decode[0], PROD)
        aec_env.decode_ordering = timer.wrap(DECODE, saved_decode[1], ORDER)
        for role, m in models.items():
            m.forward_view = timer.wrap(FORWARD, type(m).forward_view.__get__(m), role)
        yield
    finally:
        aec_env.decode_producing, aec_env.decode_ordering = saved_decode
        for attr in ("reset", "observe"):
            env.__dict__.pop(attr, None)
        if env.system is not None:
            for name in _SYSTEM_STAGES:
                env.system.__dict__.pop(name, None)
        for m in models.values():
            getattr(m, "__dict__", {}).pop("forward_view", None)


def timed_episode(env: Any, prod_policy: Policy, order_policy: Policy, seed: int,
                  models: Mapping[str, Any] | None = None
                  ) -> tuple[dict, dict[str, float], dict[str, int]]:
    """One episode under the stage timers -> (episode stats, seconds, counts). The policies'
    own time beyond their timed views and forwards is `decode` (planner, action packing).
    `seconds` holds `<stage>_s` and, for the policy-side stages, `<role>_<stage>_s`."""
    timer = StageTimer()
    counts = {f"{PROD}_decisions": 0, f"{ORDER}_decisions": 0}

    def counted(fn: Policy, role: str) -> Policy:
        def policy(e: Any, agent: str) -> Any:
            counts[f"{role}_decisions"] += 1
            return fn(e, agent)
        return timer.wrap(DECODE, policy, role)

    with instrumented(env, timer, models):
        prod, order = counted(prod_policy, PROD), counted(order_policy, ORDER)
        t0 = perf_counter()
        timer.start()
        stats = run_episode(env, prod, order, seed)
        timer.stop()
        wall = perf_counter() - t0
    seconds = {f"{s}_s": timer.totals[s] for s in STAGES}
    seconds.update({f"{r}_{s}_s": timer.by_role[(r, s)] for r in ROLES for s in POLICY_STAGES})
    seconds["wall_s"] = wall
    return stats, seconds, counts


# --- jobs ---

_W: dict[str, Any] = {}


def _load_models(prod_ckpt: str, orderer_ckpt: str) -> tuple[Any, Any]:
    import torch

    from ..train.orderer_scorer import load_scorer
    from ..train.producer_scorer import load_producer_score_model

    torch.set_num_threads(1)
    return load_producer_score_model(prod_ckpt), load_scorer(orderer_ckpt)


def _worker_init(prod_ckpt: str, orderer_ckpt: str) -> None:
    pin_blas()
    _W.clear()
    _W["models"] = _load_models(prod_ckpt, orderer_ckpt)


def run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """One timed episode of one (cell, world seed, repeat); models from the worker."""
    from ..train.orderer_scorer import scorer_ordering_policy
    from ..train.producer_scorer import scorer_producing_policy

    prod_model, order_model = _W["models"]
    cell = make_cell(job["producers"], job["fanout"], job["consumer_multiplier"])
    t0 = perf_counter()
    cfg = build_family_config(job["world_seed"], family=SPARSE_WIDE, producers=cell.producers,
                              regime=CALM, fanout=cell.fanout,
                              consumer_multiplier=cell.consumer_multiplier)
    env = joint_stack_env(cfg, job["horizon"])
    setup = perf_counter() - t0
    stats, seconds, counts = timed_episode(
        env, scorer_producing_policy(prod_model), scorer_ordering_policy(order_model),
        job["episode_seed"], models={PROD: prod_model, ORDER: order_model})
    decisions = counts["prod_decisions"] + counts["order_decisions"]
    return {"cell": cell.id, "producers": cell.producers, "fanout": cell.fanout,
            "consumer_multiplier": cell.consumer_multiplier, "consumers": cell.consumers,
            "world_seed": job["world_seed"], "repeat": job["repeat"],
            "episode_seed": job["episode_seed"], "horizon": job["horizon"],
            "cycles": stats["cycles"], "dead": int(bool(stats["dead"])),
            "units": stats["consumer_completed_units"], "decisions": decisions, **counts,
            "setup_s": setup, **seconds}


def build_jobs(cells: Sequence[Cell], seeds: Sequence[int], repeats: int,
               horizon: int) -> list[dict[str, Any]]:
    """Repeat-major order: a transient load spike cannot hit all repeats of one cell."""
    return [{"producers": c.producers, "fanout": c.fanout,
             "consumer_multiplier": c.consumer_multiplier, "world_seed": s, "repeat": r,
             "episode_seed": EPISODE_SEED, "horizon": horizon}
            for r in range(repeats) for c in cells for s in seeds]


def run_jobs(jobs: Sequence[Mapping[str, Any]], workers: int, prod_ckpt: str,
             orderer_ckpt: str, log: Callable[[str], None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    done = 0

    def note(row: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        log(f"[deployed_cost] {done}/{len(jobs)} {row['cell']} seed {row['world_seed']} "
            f"rep {row['repeat']}: {row['wall_s']:.2f} s, {row['decisions']} decisions")

    if workers == 1:
        _worker_init(prod_ckpt, orderer_ckpt)
        for j in jobs:
            rows.append(run_job(j))
            note(rows[-1])
        return rows
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    pin_blas()  # env vars reach the spawned children before they import numpy / torch
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=_worker_init,
                             initargs=(prod_ckpt, orderer_ckpt)) as pool:
        for fut in as_completed([pool.submit(run_job, j) for j in jobs]):
            rows.append(fut.result())
            note(rows[-1])
    return rows


# --- aggregation ---

# measure -> (denominator column, prefix of the role's stage columns or "", stages, flagged).
# `decision` is D7's per-decision time over all decisions; `prod_decision` / `order_decision`
# charge the policy-side stages to the role's own decisions, so a stage that grows on one role
# is not diluted by the other role's (differently growing) count; `cycle` is per cycle, the
# natural unit of `domain` / `harness`, reported but never flagged.
MEASURES: dict[str, tuple[str, str, tuple[str, ...], bool]] = {
    "decision": ("decisions", "", STAGES, True),
    "prod_decision": ("prod_decisions", "prod_", POLICY_STAGES, True),
    "order_decision": ("order_decisions", "order_", POLICY_STAGES, True),
    "cycle": ("cycles", "", STAGES, False),
}
# the one source of the flag setting: `superlinear` and spec.json both read MEASURES
FLAGGED_MEASURES = tuple(m for m, (_, _, _, flagged) in MEASURES.items() if flagged)
TIMING_COLUMNS = ("cell", "producers", "fanout", "consumer_multiplier", "consumers",
                  "world_seed", "repeat", "episode_seed", "horizon", "cycles", "dead", "units",
                  "decisions", "prod_decisions", "order_decisions", "setup_s", "wall_s",
                  *(f"{s}_s" for s in STAGES),
                  *(f"{r}_{s}_s" for r in ROLES for s in POLICY_STAGES))


def measure_column(measure: str, stage: str) -> str:
    """cells.csv column: `obs_per_decision_s`, `prod_obs_per_decision_s`, `obs_per_cycle_s`."""
    _, prefix, _, _ = MEASURES[measure]
    unit = "cycle" if measure == "cycle" else "decision"
    return f"{prefix}{stage}_per_{unit}_s"


def _measure_values(r: Mapping[str, Any]) -> dict[str, float]:
    """One timings row -> every cells.csv measure column ("total" = the wall, or for a role
    the sum of its policy-side stages)."""
    out: dict[str, float] = {}
    for measure, (denom, prefix, stages, _) in MEASURES.items():
        n = int(r[denom])
        parts = {s: float(r[f"{prefix}{s}_s"]) for s in stages}
        total = float(r["wall_s"]) if not prefix else sum(parts.values())
        for s, v in (*parts.items(), ("total", total)):
            out[measure_column(measure, s)] = v / n if n else 0.0
    return out


CELL_COLUMNS = ("cell", "producers", "fanout", "consumer_multiplier", "consumers",
                "consumers_per_producer", "worlds", "decisions", "prod_decisions",
                "order_decisions", "cycles", "wall_s",
                *(measure_column(m, s) for m, (_, _, stages, _) in MEASURES.items()
                  for s in (*stages, "total")))
SLOPE_COLUMNS = ("axis", "measure", "stage", "slope", "points", "x", "seconds", "flagged")
COUNT_KEYS = ("decisions", "prod_decisions", "order_decisions", "cycles", "wall_s")


def _row_sort_key(r: Mapping[str, Any]) -> tuple:
    return (int(r["producers"]), int(r["fanout"]), int(r["consumer_multiplier"]),
            int(r["world_seed"]), int(r["repeat"]))


def cell_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per cell: per world, the median over repeats of each measure (stage seconds / its
    denominator); then the mean over worlds. Counts and wall are means of the world medians."""
    by_world: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for r in rows:
        by_world.setdefault((r["cell"], int(r["world_seed"])), []).append(r)
    per_cell: dict[str, list[dict[str, float]]] = {}
    for (cell, _), reps in sorted(by_world.items()):
        values = [_measure_values(r) for r in reps]
        w = {k: median(v[k] for v in values) for k in values[0]}
        w.update({k: median(float(r[k]) for r in reps) for k in COUNT_KEYS})
        per_cell.setdefault(cell, []).append(w)
    out: dict[str, dict[str, Any]] = {}
    for cell_id, worlds in per_cell.items():
        c = parse_cell(cell_id)
        row: dict[str, Any] = {"cell": cell_id, "producers": c.producers, "fanout": c.fanout,
                               "consumer_multiplier": c.consumer_multiplier,
                               "consumers": c.consumers,
                               "consumers_per_producer": c.consumers_per_producer,
                               "worlds": len(worlds)}
        row.update({k: fmean(w[k] for w in worlds) for k in worlds[0]})
        out[cell_id] = row
    return out


def slope_rows(cells: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per (axis, measure, stage) whose axis has >= 2 measured cells."""
    out: list[dict[str, Any]] = []
    for axis, members in axes().items():
        present = [c for c in members if c.id in cells]
        if len({axis_value(axis, c) for c in present}) < 2:
            continue
        xs = [axis_value(axis, c) for c in present]
        for measure, (_, _, stages, _) in MEASURES.items():
            for s in (*stages, "total"):
                ys = [float(cells[c.id][measure_column(measure, s)]) for c in present]
                if min(ys) <= 0:
                    continue  # a stage that never ran on some cell has no slope
                slope = loglog_slope(xs, ys)
                out.append({"axis": axis, "measure": measure, "stage": s,
                            "slope": round(slope, 4), "points": len(xs),
                            "x": " ".join(f"{x:.4g}" for x in xs),
                            "seconds": " ".join(f"{y:.4g}" for y in ys),
                            "flagged": SUPERLINEAR_LABEL if superlinear(axis, slope, measure)
                            else ""})
    return out


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


# --- run ---


CGROUP_ROOT = Path("/sys/fs/cgroup")


PROC_CGROUP = Path("/proc/self/cgroup")


def _own_cgroup(root: Path, proc: Path) -> Path:
    """This process's cgroup v2 directory (`0::<path>` in `proc`), else `root`."""
    try:
        for line in proc.read_text().splitlines():
            if line.startswith("0::"):
                own = root / line[3:].lstrip("/")
                if (own / "cpu.max").exists():
                    return own
    except OSError:
        pass
    return root


def cgroup_cpu_limit(root: Path | None = None,
                     proc: Path | None = None) -> tuple[float | None, str | None]:
    """The CPU quota in CPUs (cgroup v2 `cpu.max` of this process's cgroup, else v1
    `cpu.cfs_quota_us` / `cpu.cfs_period_us`) and the raw text read; (None, raw) when
    unlimited, (None, None) when unreadable."""
    root = CGROUP_ROOT if root is None else root
    try:
        raw = (_own_cgroup(root, PROC_CGROUP if proc is None else proc)
               / "cpu.max").read_text().strip()
        quota, period = raw.split()[:2]
        return (None if quota == "max" else int(quota) / int(period)), raw
    except (OSError, ValueError):
        pass
    try:
        q = (root / "cpu" / "cpu.cfs_quota_us").read_text().strip()
        per = (root / "cpu" / "cpu.cfs_period_us").read_text().strip()
        raw = f"{q} {per}"
        return (None if int(q) <= 0 else int(q) / int(per)), raw
    except (OSError, ValueError):
        return None, None


def cpu_counts(root: Path | None = None, proc: Path | None = None) -> dict[str, Any]:
    """Raw CPU counts and the usable vCPU: the affinity mask (else `os.cpu_count`), capped by
    the cgroup quota rounded down (at least 1). A container may report the host's CPUs in
    `os.cpu_count`; the affinity mask and the quota are what this process can actually use."""
    logical = os.cpu_count() or 1
    try:
        affinity: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    quota, raw = cgroup_cpu_limit(root, proc)
    vcpu = affinity if affinity is not None else logical
    if quota is not None:
        vcpu = min(vcpu, max(1, math.floor(quota)))
    return {"os_cpu_count": logical, "affinity": affinity, "cgroup_cpu_quota": quota,
            "cgroup_cpu_raw": raw, "vcpu": vcpu}


def host_info() -> dict[str, Any]:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"node": platform.node(), "cpu_model": cpu, **cpu_counts(),
            "loadavg_start": list(os.getloadavg()), **environment_manifest()}


def max_workers() -> int:
    """D7: at least 2 usable cores idle, at least one worker."""
    return max(1, cpu_counts()["vcpu"] - 2)


def _is_commit(s: str) -> bool:
    return len(s) == 40 and all(c in "0123456789abcdef" for c in s)


def real_run_problems(*, source_commit: str, horizon: int, pins: Mapping[str, str],
                      filters: Mapping[str, object]) -> list[str]:
    problems = [msg for bad, msg in (
        (not _is_commit(source_commit),
         "no full 40-hex source commit (--source-commit or --verify-source-commit)"),
        (horizon != HORIZON, f"horizon {horizon} != {HORIZON}"),
    ) if bad]
    problems += [f"{role} checkpoint {pins[role][:12]} is not the registered {want[:12]}"
                 for role, want in CHECKPOINTS.items() if pins[role] != want]
    problems += [f"--{k.replace('_', '-')} is set" for k, v in filters.items() if v is not None]
    return problems


QUESTION = ("Does any stage of the deployed portable path (observation build, policy forward, "
            "decode / allocate, domain cycle, harness) cost more per decision as fanout or "
            "consumers per producer grow, within the 12 -> 42 producer target (D7)?")
DRY_BANNER = "> **DRY RUN** - a smoke of the driver, not a measurement (D7 protocol not met).\n\n"
RESULT_SKELETON = """# {title}

{banner}**Question.** {question}

**Driver.** `python -m {driver}` (design D7). Portable stack (EXP-0035 producer, EXP-0012
orderer) under `joint_stack_env`, calm `sparse_wide` worlds, horizon {horizon}, world seeds
{seeds}, {repeats} repeat(s), {workers} worker(s) on {vcpu} vCPU. Status: {status}.

Seconds per stage are divided by each measure's unit: all decisions (`decision`), the role's
own decisions for the policy-side stages (`prod_decision`, `order_decision`), or cycles
(`cycle`). Per world the median over repeats, then the mean over worlds (`cells.csv`); slopes
are log-log OLS over an axis's cells (`slopes.csv`). A per-decision slope (pooled or per role)
above {thr} on the fanout or consumer axis is flagged "{label}" (a warning, not proof);
per-cycle slopes are reported, never flagged.

## Slopes

| axis | measure | stage | slope | flag |
|---|---|---|---|---|
{slope_table}

## Conclusions

TODO.
"""


def run_deployed_cost(*, out_dir: str | Path, prod_ckpt: str, orderer_ckpt: str,
                      workers: int = 1, horizon: int = HORIZON,
                      cells: Sequence[str] | None = None, seeds: int | None = None,
                      repeats: int | None = None, source_commit: str = "",
                      dry_run: bool = False, log: Callable[[str], None] = print) -> int:
    """Run the D7 grid (or a dry-run subset) into `out_dir`; return the exit status."""
    if workers < 1 or workers > max_workers():
        raise ValueError(f"--workers must be in [1, {max_workers()}] (vCPU "
                         f"{cpu_counts()['vcpu']} - 2 idle, D7), got {workers}")
    chosen = [parse_cell(c) for c in cells] if cells else grid()
    if seeds is not None and not 1 <= seeds <= len(WORLD_SEEDS):
        raise ValueError(f"--seeds must be in [1, {len(WORLD_SEEDS)}], got {seeds}")
    world_seeds = list(WORLD_SEEDS[:seeds] if seeds is not None else WORLD_SEEDS)
    n_rep = REPEATS if repeats is None else repeats
    if n_rep < 1:
        raise ValueError(f"--repeats must be >= 1, got {n_rep}")
    pins = {"producer": sha256_file(prod_ckpt), "orderer": sha256_file(orderer_ckpt)}
    if not dry_run:
        problems = real_run_problems(source_commit=source_commit, horizon=horizon, pins=pins,
                                     filters={"cells": cells, "seeds": seeds,
                                              "repeats": repeats})
        if problems:
            raise ValueError("not a valid PoC-B run (" + "; ".join(problems)
                             + "); pass --dry-run for a smoke run")
    out = Path(out_dir)
    if (out / "spec.json").exists():
        raise ValueError(f"{out} already holds a run (spec.json); use a fresh --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    pin_blas()
    host = host_info()
    source = source_fingerprint(source_commit or None)
    jobs = build_jobs(chosen, world_seeds, n_rep, horizon)
    log(f"[deployed_cost] {'DRY RUN ' if dry_run else ''}{len(chosen)} cell(s) x "
        f"{len(world_seeds)} world(s) x {n_rep} repeat(s) = {len(jobs)} episode(s), "
        f"{workers} worker(s)")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = perf_counter()
    rows = sorted(run_jobs(jobs, workers, prod_ckpt, orderer_ckpt, log), key=_row_sort_key)
    elapsed = perf_counter() - t0
    table = cell_table(rows)
    slopes = slope_rows(table)
    flagged = [f"{s['stage']} per {s['measure']} on {s['axis']} ({s['slope']})"
               for s in slopes if s["flagged"]]
    _write_csv(out / "timings.csv", TIMING_COLUMNS, rows)
    _write_csv(out / "cells.csv", CELL_COLUMNS, [table[c.id] for c in chosen if c.id in table])
    _write_csv(out / "slopes.csv", SLOPE_COLUMNS, slopes)
    host["loadavg_end"] = list(os.getloadavg())
    spec = {
        "experiment": out.name, "driver": DRIVER, "kind": "dry-run" if dry_run else "measurement",
        "acceptance_read": False, "started": started, "elapsed_s": round(elapsed, 3),
        "pins": {"checkpoints": pins, "registered_checkpoints": CHECKPOINTS,
                 "family": SPARSE_WIDE, "regime": CALM, "horizon": horizon,
                 "world_seeds": world_seeds, "repeats": n_rep, "episode_seed": EPISODE_SEED,
                 "cells": [c.id for c in chosen], "workers": workers},
        "inputs": {"producer": {"path": str(prod_ckpt), "sha256": pins["producer"]},
                   "orderer": {"path": str(orderer_ckpt), "sha256": pins["orderer"]}},
        "source": {"tree_sha256": source["tree_sha256"], "commit": source["commit"]},
        "host": host,
        "pre_registered": {
            "axes": {a: [c.id for c in cs] for a, cs in axes().items()},
            "stages": list(STAGES), "max_producers": MAX_PRODUCERS,
            "measures": {m: {"denominator": d, "stages": [*st, "total"], "flagged": f}
                         for m, (d, _, st, f) in MEASURES.items()},
            "aggregate": "per world the median over repeats of stage seconds / the measure's "
                         "denominator; per cell the mean over worlds; log-log OLS over an "
                         "axis's cells",
            "flag": f"slope > {SUPERLINEAR} on {', '.join(FLAGGED_AXES)} for the measures "
                    f"{', '.join(FLAGGED_MEASURES)} (pooled and per-role per-decision; "
                    f"per-cycle rows reported, never flagged): '{SUPERLINEAR_LABEL}'"},
        "status": "complete",
        "result": {"episodes": len(rows), "flagged": flagged},
    }
    (out / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    if not (out / "RESULT.md").exists():
        table_md = "\n".join(f"| {s['axis']} | {s['measure']} | {s['stage']} | {s['slope']} | "
                             f"{s['flagged']} |" for s in slopes) or "| - | - | - | - | - |"
        (out / "RESULT.md").write_text(RESULT_SKELETON.format(
            title=out.name, banner=DRY_BANNER if dry_run else "", question=QUESTION,
            driver=DRIVER, horizon=horizon, seeds=", ".join(map(str, world_seeds)),
            repeats=n_rep, workers=workers, vcpu=host["vcpu"], status="complete",
            thr=SUPERLINEAR, label=SUPERLINEAR_LABEL, slope_table=table_md))
    log(f"[deployed_cost] done in {elapsed:.1f} s; flagged: {', '.join(flagged) or 'none'}")
    return 0


def _csv_strs(s: str | None) -> list[str] | None:
    return None if s is None else [x for x in s.split(",") if x]


def main(argv: Sequence[str] | None = None) -> int:
    from .provenance import resolve_verified_commit

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prod-ckpt", required=True, help="EXP-0035 producer_candidate.pt")
    p.add_argument("--orderer-ckpt", required=True, help="EXP-0012 orderer_scorer.pt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", type=int, default=1, help="at most vCPU - 2 (D7)")
    p.add_argument("--source-commit", default="")
    p.add_argument("--verify-source-commit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cells", help="dry run: comma-separated P<p>-E<e>-k<k> cells (P <= 42)")
    p.add_argument("--seeds", type=int, help="dry run: the first N of the 3 world seeds")
    p.add_argument("--repeats", type=int, help="dry run: repeats per world")
    p.add_argument("--horizon", type=int, default=HORIZON)
    a = p.parse_args(argv)
    commit = resolve_verified_commit() if a.verify_source_commit else a.source_commit
    return run_deployed_cost(out_dir=a.out_dir, prod_ckpt=a.prod_ckpt,
                             orderer_ckpt=a.orderer_ckpt, workers=a.workers, horizon=a.horizon,
                             cells=_csv_strs(a.cells), seeds=a.seeds, repeats=a.repeats,
                             source_commit=commit, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
