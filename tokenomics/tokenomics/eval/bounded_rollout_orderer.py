"""EXP-0042: confidence-bounded online rollout orderer over ``blend_0.5``.

EXP-0040 found that producer preferences change materially with the future demand
suffix: the expected selector is useful, but its winner does not converge affordably
enough to train a standalone actor.  This module keeps that signal where it is valid —
inside an online model-based planner — and asks how little rollout depth is sufficient.

At each ordering-agent action the planner:

* starts from the fixed ``blend_0.5`` action;
* considers only contested streams whose top-two base routing probabilities are close;
* scores one-logit producer bumps under common resampled future-demand scenarios;
* checks confidence after 4, 8, and 16 scenarios; and
* applies only confidence-qualified bumps, falling back to ``blend_0.5`` otherwise.

The experiment compares rollout depths 1, 2, and 3 on fresh paired episode seeds.  It
is an evaluator and model-based planner, not a trainer or an observation-only actor.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ..env import TokenomicsAECEnv
from ..env.future_fork import fork_future_demand
from .baselines import Policy, ordering_blend
from .curriculum import TaskRecord, replay
from .graph_generalization_dump import TASK_IDS, TASK_S29, TASK_S45, TASK_S77
from .graph_value_router_dump import (
    assert_inputs_match,
    decision_digest,
    environment_manifest,
    sha256_file,
    source_fingerprint,
)
from .joint_stack import joint_stack_env
from .orderer_oracle import _rollout_units
from .runner import run_episode
from .value_router import _one_cycle_bump


EXPERIMENT = "EXP-0042"
FLOOR = "blend_0.5"
DEPTHS = (1, 2, 3)
SCENARIO_CHECKPOINTS = (4, 8, 16)
DELTA = 1.0
AMBIGUITY_PROB_GAP = 0.25
CONFIDENCE_Z = 1.645
MIN_EXPECTED_GAIN = 0.25
HORIZON = 100
REGISTERED_SEEDS: dict[str, tuple[int, ...]] = {
    TASK_S29: tuple(range(3301, 3311)),
    TASK_S77: tuple(range(3400, 3410)),
    TASK_S45: tuple(range(3500, 3510)),
}
REGISTERED_INPUT_SHA256 = {
    "archive": "bc6ca4ebacff6835571453cf7c0bf10de5c366cc208674f44cf3a5860b778ceb",
    "producer_ckpt": "8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea",
}
SMOKE_SEED = 3299


@dataclass(frozen=True)
class StreamCandidate:
    stream_index: int
    order_id: str
    eligible_cols: tuple[int, ...]


@dataclass(frozen=True)
class Resolution:
    """Confidence read for one stream; ``choice`` is a producer column or ``None``."""

    resolved: bool
    choice: int | None
    reason: str
    scenarios: int
    mean_gain: float
    min_lcb: float


@dataclass
class SearchTelemetry:
    order_actions: int = 0
    contested_streams: int = 0
    ambiguity_gated_streams: int = 0
    searched_streams: int = 0
    search_actions: int = 0
    scenario_forks: int = 0
    stream_scenarios: int = 0
    rollout_calls: int = 0
    override_streams: int = 0
    confident_base_streams: int = 0
    deterministic_zero_streams: int = 0
    unresolved_fallback_streams: int = 0
    search_seconds: float = 0.0


ScenarioScorer = Callable[
    [TokenomicsAECEnv, str, Sequence[StreamCandidate], int],
    Mapping[int, np.ndarray],
]


def _softmax_gap(logits: Sequence[float]) -> float:
    """Top-two probability gap for an eligible-producer logit slice."""
    x = np.asarray(logits, dtype=np.float64)
    if x.size < 2:
        return 1.0
    x = x - float(np.max(x))
    probs = np.exp(x)
    probs /= float(probs.sum())
    top = np.sort(probs)[-2:]
    return float(top[-1] - top[-2])


def resolve_stream(
    gains: np.ndarray,
    eligible_cols: Sequence[int],
    *,
    z: float = CONFIDENCE_Z,
    min_expected_gain: float = MIN_EXPECTED_GAIN,
    tol: float = 1e-12,
) -> Resolution:
    """Resolve a stream from ``[scenario, candidate]`` gains versus the base action.

    Base is an explicit zero-valued candidate.  A unique top action is accepted only
    when its paired lower confidence bound beats every alternative.  A producer bump
    must additionally clear ``min_expected_gain``. Exact all-zero matrices are a safe
    early base stop; every other inconclusive read remains unresolved.
    """
    arr = np.asarray(gains, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(eligible_cols) or arr.shape[0] < 2:
        raise ValueError(
            f"gains must have shape [K,{len(eligible_cols)}] with K>=2, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("gains contain a non-finite value")
    n = int(arr.shape[0])
    if bool(np.all(np.abs(arr) <= tol)):
        return Resolution(True, None, "deterministic_zero", n, 0.0, 0.0)

    scores = np.column_stack([np.zeros(n, dtype=np.float64), arr])
    means = scores.mean(axis=0)
    best = int(np.argmax(means))
    tied = np.flatnonzero(np.abs(means - means[best]) <= tol)
    if len(tied) != 1:
        return Resolution(False, None, "mean_tie", n, float(means[best]), float("-inf"))

    lcbs: list[float] = []
    for other in range(scores.shape[1]):
        if other == best:
            continue
        diff = scores[:, best] - scores[:, other]
        se = float(np.std(diff, ddof=1) / math.sqrt(n))
        lcbs.append(float(np.mean(diff) - z * se))
    min_lcb = min(lcbs)
    if min_lcb <= 0.0:
        return Resolution(False, None, "confidence", n, float(means[best]), min_lcb)
    if best == 0:
        return Resolution(True, None, "confident_base", n, 0.0, min_lcb)
    if float(means[best]) < min_expected_gain:
        return Resolution(False, None, "minimum_gain", n, float(means[best]), min_lcb)
    return Resolution(
        True, int(eligible_cols[best - 1]), "confident_override", n,
        float(means[best]), min_lcb)


class BoundedScenarioOrderer:
    """Callable order policy implementing the registered adaptive rollout rule."""

    def __init__(
        self,
        *,
        task_id: str,
        episode_seed: int,
        prod: Policy,
        depth: int,
        base: Policy | None = None,
        delta: float = DELTA,
        ambiguity_prob_gap: float = AMBIGUITY_PROB_GAP,
        checkpoints: Sequence[int] = SCENARIO_CHECKPOINTS,
        z: float = CONFIDENCE_Z,
        min_expected_gain: float = MIN_EXPECTED_GAIN,
        scenario_scorer: ScenarioScorer | None = None,
        max_search_actions: int | None = None,
    ) -> None:
        if depth not in DEPTHS:
            raise ValueError(f"depth {depth} not in registered depths {DEPTHS}")
        checkpoints = tuple(int(k) for k in checkpoints)
        if not checkpoints or checkpoints[0] < 2 or any(
                b <= a for a, b in zip(checkpoints, checkpoints[1:])):
            raise ValueError("scenario checkpoints must be strictly increasing and start at K>=2")
        self.task_id = task_id
        self.episode_seed = int(episode_seed)
        self.prod = prod
        self.depth = int(depth)
        self.base = base if base is not None else ordering_blend(0.5)
        self.delta = float(delta)
        self.ambiguity_prob_gap = float(ambiguity_prob_gap)
        self.checkpoints = checkpoints
        self.z = float(z)
        self.min_expected_gain = float(min_expected_gain)
        self.scenario_scorer = scenario_scorer or self._score_scenario
        self.max_search_actions = max_search_actions
        self.telemetry = SearchTelemetry()

    def _score_scenario(
        self,
        env: TokenomicsAECEnv,
        agent: str,
        streams: Sequence[StreamCandidate],
        continuation_id: int,
    ) -> Mapping[int, np.ndarray]:
        digest = decision_digest(
            self.task_id, self.episode_seed, env.current_cycle, f"agent:{agent}")
        forked = fork_future_demand(
            env, task_id=self.task_id, decision_digest=digest,
            continuation_id=continuation_id, experiment=EXPERIMENT)
        base_units = _rollout_units(copy.deepcopy(forked), self.prod, self.base, self.depth)
        out: dict[int, np.ndarray] = {}
        self.telemetry.rollout_calls += 1
        for stream in streams:
            values = []
            for col in stream.eligible_cols:
                bumped = _one_cycle_bump(
                    self.base, agent, stream.stream_index, col, self.delta,
                    env.current_cycle)
                units = _rollout_units(copy.deepcopy(forked), self.prod, bumped, self.depth)
                values.append(float(units - base_units))
                self.telemetry.rollout_calls += 1
            out[stream.stream_index] = np.asarray(values, dtype=np.float64)
        return out

    def __call__(self, env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        started = time.perf_counter()
        action = np.asarray(self.base(env, agent), dtype=np.float32).copy()
        self.telemetry.order_actions += 1
        _, aid = env._role[agent]
        view = env.system.ordering_agents[aid].view(env.system.producer_infos)
        mat = action.reshape(env.spec_.k_stream, env.spec_.num_producers)
        streams: list[StreamCandidate] = []
        for k, oid in enumerate(sorted(view.incoming)[:env.spec_.k_stream]):
            elig = tuple(env.spec_.producer_ids.index(p) for p in view.eligible[oid])
            if len(elig) < 2:
                continue
            self.telemetry.contested_streams += 1
            if _softmax_gap([float(mat[k, c]) for c in elig]) > self.ambiguity_prob_gap:
                self.telemetry.ambiguity_gated_streams += 1
                continue
            streams.append(StreamCandidate(k, oid, elig))
        self.telemetry.searched_streams += len(streams)
        if not streams:
            return action
        if (self.max_search_actions is not None and
                self.telemetry.search_actions >= self.max_search_actions):
            self.telemetry.unresolved_fallback_streams += len(streams)
            return action

        self.telemetry.search_actions += 1
        active = {s.stream_index: s for s in streams}
        rows: dict[int, list[np.ndarray]] = {s.stream_index: [] for s in streams}
        choices: dict[int, int] = {}
        previous = 0
        for checkpoint in self.checkpoints:
            if not active:
                break
            active_list = list(active.values())
            for cid in range(previous, checkpoint):
                scored = self.scenario_scorer(env, agent, active_list, cid)
                if set(scored) != set(active):
                    raise ValueError(
                        f"scenario scorer returned streams {sorted(scored)} != {sorted(active)}")
                for k, vals in scored.items():
                    arr = np.asarray(vals, dtype=np.float64)
                    if arr.shape != (len(active[k].eligible_cols),) or not np.isfinite(arr).all():
                        raise ValueError(f"invalid scenario gains for stream {k}: {arr}")
                    rows[k].append(arr)
                self.telemetry.scenario_forks += 1
            for k, stream in list(active.items()):
                resolution = resolve_stream(
                    np.stack(rows[k]), stream.eligible_cols,
                    z=self.z, min_expected_gain=self.min_expected_gain)
                if not resolution.resolved:
                    continue
                self.telemetry.stream_scenarios += resolution.scenarios
                if resolution.reason == "deterministic_zero":
                    self.telemetry.deterministic_zero_streams += 1
                elif resolution.choice is None:
                    self.telemetry.confident_base_streams += 1
                else:
                    choices[k] = resolution.choice
                    self.telemetry.override_streams += 1
                del active[k]
            previous = checkpoint

        for k in active:
            self.telemetry.stream_scenarios += len(rows[k])
            self.telemetry.unresolved_fallback_streams += 1
        for k, col in choices.items():
            mat[k, col] += self.delta
        if not np.isfinite(action).all():
            raise ValueError("bounded rollout orderer emitted a non-finite action")
        self.telemetry.search_seconds += time.perf_counter() - started
        return action


def run_search_episode(
    record: TaskRecord,
    prod: Policy,
    *,
    seed: int,
    depth: int,
    horizon: int = HORIZON,
    max_search_actions: int | None = None,
) -> dict:
    planner = BoundedScenarioOrderer(
        task_id=record.task_id, episode_seed=seed, prod=prod, depth=depth,
        max_search_actions=max_search_actions)
    started = time.perf_counter()
    stats = run_episode(joint_stack_env(replay(record), horizon), prod, planner, seed)
    elapsed = time.perf_counter() - started
    return {
        "task_id": record.task_id, "seed": seed, "arm": f"search_d{depth}",
        "depth": depth, "consumer_units": float(stats["consumer_completed_units"]),
        "consumer_rejected_units": float(stats["consumer_rejected_units"]),
        "dead": int(bool(stats["dead"])), "cycles": int(stats["cycles"]),
        "wall_seconds": elapsed, **asdict(planner.telemetry),
    }


def run_floor_episode(
    record: TaskRecord, prod: Policy, *, seed: int, horizon: int = HORIZON,
) -> dict:
    started = time.perf_counter()
    stats = run_episode(
        joint_stack_env(replay(record), horizon), prod, ordering_blend(0.5), seed)
    return {
        "task_id": record.task_id, "seed": seed, "arm": FLOOR, "depth": 0,
        "consumer_units": float(stats["consumer_completed_units"]),
        "consumer_rejected_units": float(stats["consumer_rejected_units"]),
        "dead": int(bool(stats["dead"])), "cycles": int(stats["cycles"]),
        "wall_seconds": time.perf_counter() - started,
        **asdict(SearchTelemetry()),
    }


def analyze_episode_rows(rows: Sequence[Mapping]) -> dict:
    """Artifact-only paired gate and shallowest-passing-depth selection."""
    if not rows:
        raise ValueError("no episode rows")
    keyed: dict[tuple[str, int, str], Mapping] = {}
    for row in rows:
        key = (str(row["task_id"]), int(row["seed"]), str(row["arm"]))
        if key in keyed:
            raise ValueError(f"duplicate episode row {key}")
        if not math.isfinite(float(row["consumer_units"])):
            raise ValueError(f"non-finite consumer units at {key}")
        keyed[key] = row
    tasks = sorted({str(r["task_id"]) for r in rows})
    depths = sorted({int(r["depth"]) for r in rows if int(r["depth"]) > 0})
    task_rows: list[dict] = []
    arm_rows: list[dict] = []
    passing: list[int] = []
    for depth in depths:
        arm = f"search_d{depth}"
        depth_tasks = []
        episode_rel: list[float] = []
        for task in tasks:
            seeds = sorted({int(r["seed"]) for r in rows if str(r["task_id"]) == task})
            pairs = []
            for seed in seeds:
                fk, sk = (task, seed, FLOOR), (task, seed, arm)
                if fk not in keyed or sk not in keyed:
                    raise ValueError(f"missing paired row for task={task}, seed={seed}, arm={arm}")
                floor, search = keyed[fk], keyed[sk]
                fu, su = float(floor["consumer_units"]), float(search["consumer_units"])
                rel = (su - fu) / fu if fu > 0 else 0.0
                episode_rel.append(rel)
                pairs.append((floor, search, rel))
            floor_mean = statistics.mean(float(p[0]["consumer_units"]) for p in pairs)
            search_mean = statistics.mean(float(p[1]["consumer_units"]) for p in pairs)
            relative = (search_mean - floor_mean) / floor_mean if floor_mean > 0 else 0.0
            floor_dead = statistics.mean(int(p[0]["dead"]) for p in pairs)
            search_dead = statistics.mean(int(p[1]["dead"]) for p in pairs)
            tr = {
                "task_id": task, "depth": depth, "n_episodes": len(pairs),
                "floor_units": floor_mean, "search_units": search_mean,
                "relative_improvement": relative,
                "floor_dead_rate": floor_dead, "search_dead_rate": search_dead,
                "worst_episode_relative": min(p[2] for p in pairs),
                "mean_override_streams": statistics.mean(
                    float(p[1]["override_streams"]) for p in pairs),
                "mean_rollout_calls": statistics.mean(
                    float(p[1]["rollout_calls"]) for p in pairs),
                "mean_wall_seconds": statistics.mean(
                    float(p[1]["wall_seconds"]) for p in pairs),
            }
            task_rows.append(tr)
            depth_tasks.append(tr)
        beats = sum(r["search_units"] > r["floor_units"] for r in depth_tasks)
        mean_relative = statistics.mean(r["relative_improvement"] for r in depth_tasks)
        gate = (
            beats >= 2 and mean_relative >= 0.05 and
            all(r["relative_improvement"] >= -0.05 for r in depth_tasks) and
            min(episode_rel) >= -0.10 and
            all(r["search_dead_rate"] <= r["floor_dead_rate"] for r in depth_tasks)
        )
        if gate:
            passing.append(depth)
        arm_rows.append({
            "depth": depth, "tasks_beating_floor": beats,
            "mean_relative_improvement": mean_relative,
            "worst_task_relative": min(r["relative_improvement"] for r in depth_tasks),
            "worst_episode_relative": min(episode_rel),
            "death_safe": all(
                r["search_dead_rate"] <= r["floor_dead_rate"] for r in depth_tasks),
            "gate_pass": gate,
        })
    promoted = min(passing) if passing else None
    return {
        "experiment": EXPERIMENT,
        "verdict": "PASS_BOUNDED_SEARCH" if promoted is not None else "NO_BOUNDED_SEARCH_GAIN",
        "promoted_depth": promoted,
        "task_metrics": task_rows,
        "arm_metrics": arm_rows,
    }


_WORKER_PROD: Policy | None = None


def _worker_init(prod_ckpt: str) -> None:
    global _WORKER_PROD
    import torch
    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy

    torch.set_num_threads(1)
    _WORKER_PROD = scorer_producing_policy(load_producer_score_model(prod_ckpt))


def _run_job(job: tuple[TaskRecord, int, int, int | None]) -> dict:
    record, seed, depth, max_search_actions = job
    if _WORKER_PROD is None:
        raise RuntimeError("worker producer was not initialized")
    if depth == 0:
        return run_floor_episode(record, _WORKER_PROD, seed=seed)
    return run_search_episode(
        record, _WORKER_PROD, seed=seed, depth=depth,
        max_search_actions=max_search_actions)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_registered_experiment(
    records: Sequence[TaskRecord],
    *,
    archive: str,
    prod_ckpt: str,
    out_dir: str,
    source_commit: str,
    workers: int,
    smoke: bool = False,
) -> dict:
    by_id = {r.task_id: r for r in records}
    missing = set(TASK_IDS) - set(by_id)
    if missing:
        raise ValueError(f"archive missing registered tasks: {sorted(missing)}")
    manifest = {
        "archive": {"path": archive, "sha256": sha256_file(archive)},
        "producer_ckpt": {"path": prod_ckpt, "sha256": sha256_file(prod_ckpt)},
    }
    if not smoke:
        assert_inputs_match(manifest, REGISTERED_INPUT_SHA256)
        if len(source_commit) != 40:
            raise ValueError("real run requires a full 40-character source commit")
    seeds = ({TASK_S29: (SMOKE_SEED,)} if smoke else REGISTERED_SEEDS)
    depths = (1,) if smoke else DEPTHS
    max_actions = 1 if smoke else None
    jobs = [
        (by_id[task], seed, depth, max_actions)
        for task, task_seeds in seeds.items()
        for seed in task_seeds
        for depth in (0, *depths)
    ]
    rows: list[dict] = []
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    def checkpoint(row: dict) -> None:
        rows.append(row)
        ordered = sorted(
            rows, key=lambda r: (r["task_id"], int(r["seed"]), int(r["depth"])))
        # Persist every completed episode. If a worker fails, the successful paired work is
        # still retrievable from the pod and can diagnose or seed an exact resume.
        _write_csv(output / "episodes.csv", ordered)
        print(f"[{EXPERIMENT}] {row['task_id']} seed={row['seed']} {row['arm']} "
              f"units={row['consumer_units']} wall={row['wall_seconds']:.2f}s", flush=True)

    if workers == 1:
        _worker_init(prod_ckpt)
        for job in jobs:
            checkpoint(_run_job(job))
    else:
        with ProcessPoolExecutor(
                max_workers=workers, initializer=_worker_init, initargs=(prod_ckpt,)) as pool:
            futures = {pool.submit(_run_job, job): job for job in jobs}
            for future in as_completed(futures):
                checkpoint(future.result())
    rows.sort(key=lambda r: (r["task_id"], int(r["seed"]), int(r["depth"])))
    analysis = analyze_episode_rows(rows)
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "metrics_by_task.csv", analysis["task_metrics"])
    _write_csv(output / "metrics_by_arm.csv", analysis["arm_metrics"])
    spec = {
        "experiment": EXPERIMENT, "smoke": smoke,
        "params": {
            "depths": list(depths), "scenario_checkpoints": list(SCENARIO_CHECKPOINTS),
            "delta": DELTA, "ambiguity_probability_gap": AMBIGUITY_PROB_GAP,
            "confidence_z": CONFIDENCE_Z, "minimum_expected_gain": MIN_EXPECTED_GAIN,
            "horizon": HORIZON, "seeds": {k: list(v) for k, v in seeds.items()},
        },
        "inputs": manifest, "source": source_fingerprint(source_commit),
        "environment": environment_manifest(), "workers": workers,
        "result": {k: v for k, v in analysis.items()
                   if k not in ("task_metrics", "arm_metrics")},
    }
    (output / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (output / "result.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    return analysis


def main() -> None:
    from .curriculum_replay import _load_archive
    from .graph_value_router_dump import resolve_verified_commit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--prod-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--verify-source-commit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    source_commit = args.source_commit
    if args.verify_source_commit:
        source_commit = resolve_verified_commit()
    run_registered_experiment(
        _load_archive(args.archive), archive=args.archive, prod_ckpt=args.prod_ckpt,
        out_dir=args.out_dir, source_commit=source_commit, workers=args.workers,
        smoke=args.smoke)


if __name__ == "__main__":
    main()
