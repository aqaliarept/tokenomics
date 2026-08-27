"""EXP-0043: scenario-conditioned GNN surrogate for EXP-0042 depth-2 search.

The learned model is deliberately not an observation-only orderer.  It sees the same
sampled future-demand suffix as the online planner and predicts the paired two-cycle
gain of each one-logit producer bump.  Deployment is conservative: the ensemble may
only certify an all-zero gain vector; every other stream/scenario is evaluated by the
unchanged real depth-2 rollout.

The CLI implements the registered, staged protocol:

1. collect train + validation labels on fresh whole episodes;
2. fit row, 0-hop, 2-layer and 4-layer controls and select on validation only;
3. collect/read the untouched test labels and repeat the frozen offline gate; and
4. only after both offline gates pass, compare blend, real depth-2 and hybrid depth-2
   on fresh closed-loop episodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..domain.consumer import ScheduledConsumer
from ..domain.graph_view import build_economy_graph_view
from ..env import TokenomicsAECEnv
from ..env.future_fork import fork_future_demand
from ..train.graph_router_probe import Batch, FoldNormalizer, RelationalGNN
from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
from .baselines import Policy, ordering_blend
from .bounded_rollout_orderer import (
    FLOOR,
    HORIZON,
    REGISTERED_SEEDS as EXP0042_SEEDS,
    SCENARIO_CHECKPOINTS,
    BoundedScenarioOrderer,
    Resolution,
    StreamCandidate,
    resolve_stream,
    run_floor_episode,
    run_search_episode,
)
from .curriculum import TaskRecord, replay
from .graph_generalization_dump import (
    ROLLOUT_REGISTRY,
    SPENT_SEEDS,
    SPLIT_REGISTRY,
    TASK_IDS,
    TASK_S29,
    TASK_S45,
    TASK_S77,
)
from .graph_value_router_dump import (
    GraphArrays,
    assert_inputs_match,
    decision_digest,
    encode_graph,
    environment_manifest,
    load_shard,
    schema_hash,
    sha256_file,
    source_fingerprint,
    write_shard,
)
from .joint_stack import joint_stack_env
from .orderer_oracle import _rollout_units
from .runner import run_episode
from .value_router import _one_cycle_bump


EXPERIMENT = "EXP-0043"
PLANNER_EXPERIMENT = "EXP-0042"
DEPTH = 2
FUTURE_STEPS = 2
FUTURE_FEATURES = 2 * FUTURE_STEPS  # log1p(quantity), count at each step
TRAIN_ZERO_KEEP = 0.10
MODEL_SEEDS = (0, 1, 2)
ARMS: dict[str, dict] = {
    "row_mlp_h64": {"kind": "row", "hidden": 64, "layers": 0},
    "graph_0hop_h64": {"kind": "graph", "hidden": 64, "layers": 0},
    "mpnn_2x64": {"kind": "graph", "hidden": 64, "layers": 2},
    "mpnn_4x64": {"kind": "graph", "hidden": 64, "layers": 4},
}
SELECTABLE_ARMS = ("mpnn_2x64", "mpnn_4x64")
ZERO_ABS_GRID = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
SPREAD_QUANTILES = (0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
MAX_EPOCHS = 60
MIN_EPOCHS = 12
PATIENCE = 8
BATCH_SCENARIOS = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_SAMPLES = 4096

SPLIT_SEEDS: dict[str, dict[str, tuple[int, ...]]] = {
    "train": {
        TASK_S29: tuple(range(4300, 4304)),
        TASK_S77: tuple(range(4400, 4404)),
        TASK_S45: tuple(range(4500, 4504)),
    },
    "validation": {
        TASK_S29: (5300, 5301), TASK_S77: (5400, 5401), TASK_S45: (5500, 5501),
    },
    "test": {
        TASK_S29: (6300, 6301), TASK_S77: (6400, 6401), TASK_S45: (6500, 6501),
    },
    "closed_loop_test": {
        TASK_S29: tuple(range(7300, 7310)),
        TASK_S77: tuple(range(7400, 7410)),
        TASK_S45: tuple(range(7500, 7510)),
    },
}
REGISTERED_INPUT_SHA256 = {
    "archive": "bc6ca4ebacff6835571453cf7c0bf10de5c366cc208674f44cf3a5860b778ceb",
    "producer_ckpt": "8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea",
}


def validate_registry(registry: Mapping[str, Mapping[str, Sequence[int]]] = SPLIT_SEEDS) -> None:
    """Reject split overlap or a malformed registered task set before label compute."""
    expected = set(TASK_IDS)
    seen: dict[str, set[int]] = {t: set() for t in expected}
    for split, task_map in registry.items():
        if set(task_map) != expected:
            raise ValueError(f"{split} task set differs from the registered three tasks")
        for task, seeds in task_map.items():
            vals = [int(s) for s in seeds]
            if len(vals) != len(set(vals)):
                raise ValueError(f"duplicate {split} seed for {task}")
            overlap = seen[task].intersection(vals)
            if overlap:
                raise ValueError(f"{task} seeds {sorted(overlap)} appear in multiple splits")
            seen[task].update(vals)
    for task in expected:
        prior = set(SPENT_SEEDS[task]) | set(EXP0042_SEEDS[task])
        prior.update(seed for seeds in SPLIT_REGISTRY[task].values() for seed in seeds)
        prior.update(seed for seeds in ROLLOUT_REGISTRY[task].values() for seed in seeds)
        overlap = seen[task] & prior
        if overlap:
            raise ValueError(
                f"{task} seeds {sorted(overlap)} were already spent by EXP-0037..0042")


def future_product_features(
    forked: TokenomicsAECEnv,
    product_ids: Sequence,
    *,
    steps: int = FUTURE_STEPS,
) -> np.ndarray:
    """Encode the explicit sampled suffix as identity-free per-product time features.

    Product identity is represented only by alignment with graph product nodes.  For
    every future position we aggregate scheduled quantity and request count across the
    randomized consumers, then log-compress quantity.  ``ScheduledConsumer.order`` wraps
    schedules, so this encoder mirrors the same modulo rule at the horizon boundary.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    index = {p: i for i, p in enumerate(product_ids)}
    qty = np.zeros((len(product_ids), steps), dtype=np.float64)
    count = np.zeros((len(product_ids), steps), dtype=np.float64)
    for cid in sorted(forked.system.consumers):
        consumer = forked.system.consumers[cid]
        if not isinstance(consumer, ScheduledConsumer):
            continue
        for offset in range(steps):
            product, quantity = consumer._schedule[
                (consumer._cycle + offset) % len(consumer._schedule)]
            if product not in index:
                raise ValueError(f"future product {product!r} is absent from graph products")
            pi = index[product]
            qty[pi, offset] += float(quantity)
            count[pi, offset] += 1.0
    out = np.zeros((len(product_ids), 2 * steps), dtype=np.float32)
    for offset in range(steps):
        out[:, 2 * offset] = np.log1p(qty[:, offset]).astype(np.float32)
        out[:, 2 * offset + 1] = count[:, offset].astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("non-finite future demand feature")
    return out


@dataclass
class _CollectedGroup:
    stream: StreamCandidate
    graph: GraphArrays
    cycle: int
    agent: str
    future: list[np.ndarray] = field(default_factory=list)
    gains: list[np.ndarray] = field(default_factory=list)
    continuation_ids: list[int] = field(default_factory=list)
    resolution: Resolution | None = None


class ScenarioLabelCollectorOrderer(BoundedScenarioOrderer):
    """Depth-2 planner that persists its per-scenario bump labels.

    Training follows adaptive EXP-0042 stopping.  Validation/test additionally fill the
    unobserved suffix to K=16 *after* the real action is fixed, allowing exact offline
    replay of a hybrid whose stopping checkpoint differs from the authority planner.
    """

    def __init__(self, *args, full_scenarios: bool = False, **kwargs) -> None:
        super().__init__(*args, depth=DEPTH, **kwargs)
        self.full_scenarios = bool(full_scenarios)
        self.groups: list[_CollectedGroup] = []
        self._action_groups: dict[int, _CollectedGroup] = {}
        self._postfill = False
        self.postfill_rollout_calls = 0

    def _new_group(
        self, env: TokenomicsAECEnv, agent: str, stream: StreamCandidate
    ) -> _CollectedGroup:
        _, aid = env._role[agent]
        eligible = [env.spec_.producer_ids[c] for c in stream.eligible_cols]
        graph = encode_graph(build_economy_graph_view(
            env.system, aid, stream.order_id, eligible))
        graph.decision_digest = decision_digest(
            self.task_id, self.episode_seed, env.current_cycle,
            f"{agent}|{stream.order_id}")
        group = _CollectedGroup(
            stream=stream, graph=graph, cycle=int(env.current_cycle), agent=agent)
        self._action_groups[stream.stream_index] = group
        self.groups.append(group)
        return group

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
            continuation_id=continuation_id, experiment=PLANNER_EXPERIMENT)
        base_units = _rollout_units(copy.deepcopy(forked), self.prod, self.base, DEPTH)
        self.telemetry.rollout_calls += 1
        if self._postfill:
            self.postfill_rollout_calls += 1
        out: dict[int, np.ndarray] = {}
        for stream in streams:
            group = self._action_groups.get(stream.stream_index)
            if group is None:
                group = self._new_group(env, agent, stream)
            values = []
            for col in stream.eligible_cols:
                bumped = _one_cycle_bump(
                    self.base, agent, stream.stream_index, col, self.delta,
                    env.current_cycle)
                units = _rollout_units(copy.deepcopy(forked), self.prod, bumped, DEPTH)
                values.append(float(units - base_units))
                self.telemetry.rollout_calls += 1
                if self._postfill:
                    self.postfill_rollout_calls += 1
            gains = np.asarray(values, dtype=np.float64)
            group.future.append(future_product_features(forked, group.graph.id_maps["product"]))
            group.gains.append(gains)
            group.continuation_ids.append(int(continuation_id))
            out[stream.stream_index] = gains
        return out

    @staticmethod
    def _resolve_prefix(group: _CollectedGroup) -> Resolution:
        by_cid = {cid: gain for cid, gain in zip(group.continuation_ids, group.gains)}
        for checkpoint in SCENARIO_CHECKPOINTS:
            arr = np.stack([by_cid[cid] for cid in range(checkpoint)])
            resolution = resolve_stream(arr, group.stream.eligible_cols)
            if resolution.resolved or checkpoint == SCENARIO_CHECKPOINTS[-1]:
                return resolution
        raise AssertionError("unreachable")

    def __call__(self, env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        self._action_groups = {}
        action = super().__call__(env, agent)
        action_groups = list(self._action_groups.values())
        if self.full_scenarios and action_groups:
            have = {g.stream.stream_index: set(g.continuation_ids) for g in action_groups}
            self._postfill = True
            try:
                for cid in range(SCENARIO_CHECKPOINTS[-1]):
                    missing = [g.stream for g in action_groups
                               if cid not in have[g.stream.stream_index]]
                    if missing:
                        self._score_scenario(env, agent, missing, cid)
            finally:
                self._postfill = False
        for group in action_groups:
            order = np.argsort(np.asarray(group.continuation_ids))
            group.continuation_ids = [group.continuation_ids[i] for i in order]
            group.future = [group.future[i] for i in order]
            group.gains = [group.gains[i] for i in order]
            group.resolution = self._resolve_prefix(group)
        return action


def _group_key(task_id: str, seed: int, group: _CollectedGroup) -> str:
    return f"{task_id}|{seed}|{group.cycle}|{group.agent}|{group.stream.order_id}"


def deterministic_zero_keep(task_id: str, seed: int, group: _CollectedGroup) -> bool:
    digest = hashlib.sha256(_group_key(task_id, seed, group).encode()).digest()
    return int.from_bytes(digest[:8], "big") < int(TRAIN_ZERO_KEEP * (1 << 64))


def _episode_dir(root: str | Path, split: str, task_id: str, seed: int) -> Path:
    suffix = task_id.rsplit("-", 1)[-1]
    return Path(root) / "episodes" / f"{split}_{suffix}_{seed}"


def write_collected_episode(
    root: str | Path,
    *,
    split: str,
    task_id: str,
    seed: int,
    planner: ScenarioLabelCollectorOrderer,
    stats: Mapping,
    elapsed: float,
) -> dict:
    out = _episode_dir(root, split, task_id, seed)
    out.mkdir(parents=True, exist_ok=True)
    write_shard(str(out / "graphs.npz"), [g.graph for g in planner.groups])
    sample_graph: list[int] = []
    continuation: list[int] = []
    future: list[np.ndarray] = []
    gain_ptr = [0]
    group_sample_ptr = [0]
    gains: list[float] = []
    group_rows: list[dict] = []
    for gi, group in enumerate(planner.groups):
        if group.resolution is None:
            raise ValueError("collector group has no resolution")
        for cid, fut, gain in zip(group.continuation_ids, group.future, group.gains):
            sample_graph.append(gi)
            continuation.append(cid)
            future.append(fut)
            gains.extend(float(v) for v in gain)
            gain_ptr.append(len(gains))
        choice = group.resolution.choice
        group_rows.append({
            "graph_index": gi,
            "cycle": group.cycle,
            "agent": group.agent,
            "stream_index": group.stream.stream_index,
            "order_id": group.stream.order_id,
            "eligible_cols": list(group.stream.eligible_cols),
            "n_scenarios": len(group.gains),
            "resolution_reason": group.resolution.reason,
            "resolution_choice": -1 if choice is None else int(choice),
            "resolution_mean_gain": float(group.resolution.mean_gain),
            "train_keep": (group.resolution.reason != "deterministic_zero" or
                           deterministic_zero_keep(task_id, seed, group)),
        })
        group_sample_ptr.append(len(sample_graph))
    n_products = (planner.groups[0].future[0].shape[0] if planner.groups else 0)
    np.savez_compressed(
        out / "samples.npz",
        sample_graph=np.asarray(sample_graph, dtype=np.int64),
        continuation_id=np.asarray(continuation, dtype=np.int64),
        future=(np.stack(future).astype(np.float32) if future else
                np.zeros((0, n_products, FUTURE_FEATURES), dtype=np.float32)),
        group_sample_ptr=np.asarray(group_sample_ptr, dtype=np.int64),
        gain_ptr=np.asarray(gain_ptr, dtype=np.int64),
        gains=np.asarray(gains, dtype=np.float32),
    )
    manifest = {
        "experiment": EXPERIMENT,
        "split": split,
        "task_id": task_id,
        "seed": int(seed),
        "full_scenarios": planner.full_scenarios,
        "groups": group_rows,
        "episode": {
            "consumer_units": float(stats["consumer_completed_units"]),
            "consumer_rejected_units": float(stats["consumer_rejected_units"]),
            "dead": int(bool(stats["dead"])),
            "cycles": int(stats["cycles"]),
            "wall_seconds": float(elapsed),
            "postfill_rollout_calls": planner.postfill_rollout_calls,
            **asdict(planner.telemetry),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "split": split, "task_id": task_id, "seed": int(seed),
        "groups": len(group_rows), "samples": len(sample_graph),
        **manifest["episode"],
    }


def collect_episode(
    record: TaskRecord,
    prod: Policy,
    *,
    split: str,
    seed: int,
    out_dir: str,
    horizon: int = HORIZON,
    max_search_actions: int | None = None,
) -> dict:
    full = split in ("validation", "test")
    planner = ScenarioLabelCollectorOrderer(
        task_id=record.task_id, episode_seed=seed, prod=prod,
        full_scenarios=full, max_search_actions=max_search_actions)
    started = time.perf_counter()
    stats = run_episode(joint_stack_env(replay(record), horizon), prod, planner, seed)
    return write_collected_episode(
        out_dir, split=split, task_id=record.task_id, seed=seed,
        planner=planner, stats=stats, elapsed=time.perf_counter() - started)


@dataclass
class ScenarioSample:
    continuation_id: int
    future: np.ndarray
    gains: np.ndarray


@dataclass
class ScenarioGroup:
    graph: GraphArrays
    task_id: str
    split: str
    seed: int
    cycle: int
    agent: str
    stream_index: int
    order_id: str
    eligible_cols: tuple[int, ...]
    authority_reason: str
    authority_choice: int | None
    authority_mean_gain: float
    train_keep: bool
    samples: list[ScenarioSample]

    @property
    def action_key(self) -> tuple[str, int, int, str]:
        return (self.task_id, self.seed, self.cycle, self.agent)


def load_collected(root: str | Path, splits: Sequence[str]) -> list[ScenarioGroup]:
    """Load and fully validate collected episodes for the requested registered splits."""
    wanted = set(splits)
    groups: list[ScenarioGroup] = []
    found: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for path in sorted((Path(root) / "episodes").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        split = manifest["split"]
        if split not in wanted:
            continue
        task_id, seed = manifest["task_id"], int(manifest["seed"])
        if task_id not in TASK_IDS or seed not in SPLIT_SEEDS[split][task_id]:
            raise ValueError(f"unregistered collected episode {split}/{task_id}/{seed}")
        found[split][task_id].add(seed)
        episode_dir = path.parent
        graphs = load_shard(str(episode_dir / "graphs.npz"))
        z = np.load(episode_dir / "samples.npz", allow_pickle=False)
        sample_graph = z["sample_graph"]
        cids = z["continuation_id"]
        futures = z["future"]
        group_ptr = z["group_sample_ptr"]
        ptr = z["gain_ptr"]
        values = z["gains"]
        rows = manifest["groups"]
        if (len(graphs) != len(rows) or len(sample_graph) + 1 != len(ptr) or
                len(group_ptr) != len(rows) + 1):
            raise ValueError(f"misaligned episode artifact {episode_dir}")
        for gi, (graph, row) in enumerate(zip(graphs, rows)):
            indices = range(int(group_ptr[gi]), int(group_ptr[gi + 1]))
            if any(int(sample_graph[si]) != gi for si in indices):
                raise ValueError(f"non-contiguous sample group in {episode_dir}, group {gi}")
            samples: list[ScenarioSample] = []
            expected_k = int(graph.candidate_producers.shape[0])
            for si in indices:
                gain = values[ptr[si]:ptr[si + 1]].astype(np.float64)
                future = futures[si].astype(np.float32)
                if len(gain) != expected_k or future.shape != (
                        graph.node_features["product"].shape[0], FUTURE_FEATURES):
                    raise ValueError(f"bad sample shape in {episode_dir}, group {gi}")
                if not np.isfinite(gain).all() or not np.isfinite(future).all():
                    raise ValueError(f"non-finite sample in {episode_dir}, group {gi}")
                samples.append(ScenarioSample(int(cids[si]), future, gain))
            samples.sort(key=lambda s: s.continuation_id)
            expected_n = SCENARIO_CHECKPOINTS[-1] if split in ("validation", "test") \
                else int(row["n_scenarios"])
            if [s.continuation_id for s in samples] != list(range(expected_n)):
                raise ValueError(f"scenario coverage is not 0..{expected_n - 1} in {episode_dir}")
            choice = int(row["resolution_choice"])
            groups.append(ScenarioGroup(
                graph=graph, task_id=task_id, split=split, seed=seed,
                cycle=int(row["cycle"]), agent=row["agent"],
                stream_index=int(row["stream_index"]), order_id=row["order_id"],
                eligible_cols=tuple(int(v) for v in row["eligible_cols"]),
                authority_reason=row["resolution_reason"],
                authority_choice=None if choice < 0 else choice,
                authority_mean_gain=float(row["resolution_mean_gain"]),
                train_keep=bool(row["train_keep"]), samples=samples))
    for split in wanted:
        for task, seeds in SPLIT_SEEDS[split].items():
            if found[split][task] != set(seeds):
                raise ValueError(
                    f"{split}/{task} collected seeds {sorted(found[split][task])} "
                    f"!= registered {list(seeds)}")
    return groups


@dataclass
class ScenarioNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, groups: Sequence[ScenarioGroup]) -> "ScenarioNormalizer":
        rows = [s.future for g in groups for s in g.samples]
        if not rows:
            raise ValueError("cannot fit scenario normalizer on no rows")
        stacked = np.concatenate(rows, axis=0).astype(np.float64)
        return cls(stacked.mean(axis=0).astype(np.float32),
                   np.clip(stacked.std(axis=0), 1e-6, None).astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, raw: Mapping) -> "ScenarioNormalizer":
        return cls(np.asarray(raw["mean"], dtype=np.float32),
                   np.asarray(raw["std"], dtype=np.float32))


class ScenarioBatch(Batch):
    """The established relational batch plus sampled-future rows aligned to products."""

    def __init__(self, graphs: Sequence[GraphArrays], futures: Sequence[np.ndarray]) -> None:
        if len(graphs) != len(futures):
            raise ValueError("one future tensor is required per graph")
        super().__init__(graphs)
        for graph, future in zip(graphs, futures):
            if future.shape != (graph.node_features["product"].shape[0], FUTURE_FEATURES):
                raise ValueError("future tensor does not align with graph product nodes")
        self.future_product = __import__("torch").tensor(
            np.concatenate(futures, axis=0) if futures else
            np.zeros((0, FUTURE_FEATURES), dtype=np.float32), dtype=__import__("torch").float32)
        off_orderer = np.cumsum(
            [0] + [g.node_features["orderer"].shape[0] for g in graphs])
        graph_orderer = __import__("torch").tensor(
            [off_orderer[i] + g.current_orderer for i, g in enumerate(graphs)],
            dtype=__import__("torch").long)
        self.cand_orderer = graph_orderer[self.cand_graph]

    def to(self, device):
        super().to(device)
        self.future_product = self.future_product.to(device)
        self.cand_orderer = self.cand_orderer.to(device)
        return self


class ScenarioRelationalGNN(RelationalGNN):
    """Relational scorer with sampled future demand injected into product nodes."""

    def __init__(self, *, hidden: int, layers: int, dropout: float = 0.1) -> None:
        import torch.nn as nn

        super().__init__(hidden=hidden, layers=layers, dropout=dropout)
        self.future_enc = nn.Linear(FUTURE_FEATURES, hidden)

    def forward(self, b: ScenarioBatch):
        import torch
        import torch.nn as nn

        h = {}
        for node_type in self.node_role:
            role = self.node_role[node_type].expand(b.node_counts[node_type], self.hidden)
            h[node_type] = (role + self.node_enc[node_type](b.x[node_type])
                            if node_type in self.node_enc else role)
        h["product"] = h["product"] + self.future_enc(b.future_product)
        dev = next(self.parameters()).device
        from .graph_value_router_dump import NODE_TYPES, RELATION_TYPES, RELATIONS
        for li in range(self.layers):
            agg = {t: torch.zeros(b.node_counts[t], self.hidden, device=dev) for t in NODE_TYPES}
            for relation in RELATION_TYPES:
                src_t, dst_t, _ = RELATIONS[relation]
                src, dst, attr = b.edges[relation]
                if src.shape[0] == 0:
                    continue
                fwd = torch.relu(self.msg[li][relation](
                    h[src_t][src] + self._edge_term(relation, attr, False)))
                agg[dst_t].index_add_(0, dst, fwd)
                rev = torch.relu(self.msg_rev[li][relation](
                    h[dst_t][dst] + self._edge_term(relation, attr, True)))
                agg[src_t].index_add_(0, src, rev)
            for node_type in NODE_TYPES:
                h[node_type] = self.norm[li][node_type](
                    h[node_type] + torch.relu(self.update[li][node_type](agg[node_type])))
                if self.dropout:
                    h[node_type] = nn.functional.dropout(
                        h[node_type], self.dropout, self.training)
        feat = torch.cat([
            h["decision"][b.cand_decision], h["order"][b.cand_order],
            h["product"][b.cand_product], h["producer"][b.cand_producer],
            self.cand_rel.expand(b.cand_producer.shape[0], self.hidden)], dim=1)
        return self.head(feat).squeeze(-1)


def row_features(batch: ScenarioBatch):
    """Identity-free local control: current anchors, candidate and target future only."""
    import torch

    return torch.cat([
        batch.x["producer"][batch.cand_producer],
        batch.x["product"][batch.cand_product],
        batch.x["order"][batch.cand_order],
        batch.x["orderer"][batch.cand_orderer],
        batch.future_product[batch.cand_product],
    ], dim=1)


class ScenarioRowMLP(__import__("torch").nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        import torch.nn as nn

        super().__init__()
        # producer 8 + product 2 + order 13 + orderer 2 + future 4
        self.net = nn.Sequential(
            nn.Linear(29, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch: ScenarioBatch):
        return self.net(row_features(batch)).squeeze(-1)


def build_model(arm: str):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    cfg = ARMS[arm]
    if cfg["kind"] == "row":
        return ScenarioRowMLP(cfg["hidden"])
    return ScenarioRelationalGNN(
        hidden=cfg["hidden"], layers=cfg["layers"], dropout=0.1)


def _fold_normalizer_dict(normalizer: FoldNormalizer) -> dict:
    return normalizer.groups()


def _fold_normalizer_from_dict(raw: Mapping) -> FoldNormalizer:
    def load_group(group: Mapping) -> dict:
        return {
            key: (
                np.asarray(value["cols"], dtype=np.int64),
                np.asarray(value["mean"], dtype=np.float32),
                np.asarray(value["std"], dtype=np.float32),
            )
            for key, value in group.items()
        }
    return FoldNormalizer(load_group(raw["node"]), load_group(raw["edge"]))


def normalize_groups(
    groups: Sequence[ScenarioGroup],
    graph_normalizer: FoldNormalizer,
    scenario_normalizer: ScenarioNormalizer,
) -> list[ScenarioGroup]:
    normalized_graphs = graph_normalizer.apply([g.graph for g in groups])
    out: list[ScenarioGroup] = []
    for group, graph in zip(groups, normalized_graphs):
        samples = [replace(s, future=scenario_normalizer.apply(s.future))
                   for s in group.samples]
        out.append(replace(group, graph=graph, samples=samples))
    return out


def flat_samples(groups: Sequence[ScenarioGroup]) -> list[tuple[ScenarioGroup, ScenarioSample]]:
    return [(group, sample) for group in groups for sample in group.samples]


def _batch_from_items(items: Sequence[tuple[ScenarioGroup, ScenarioSample]], device: str):
    graphs = [group.graph for group, _ in items]
    futures = [sample.future for _, sample in items]
    gains = np.concatenate([sample.gains for _, sample in items]).astype(np.float32)
    return ScenarioBatch(graphs, futures).to(device), __import__("torch").tensor(
        gains, dtype=__import__("torch").float32, device=device)


def _scenario_weighted_loss(pred, target, candidate_graph, n_graphs: int):
    import torch
    import torch.nn.functional as functional

    raw = functional.smooth_l1_loss(pred, target, reduction="none")
    per = torch.zeros(n_graphs, device=pred.device).index_add_(0, candidate_graph, raw)
    count = torch.zeros(n_graphs, device=pred.device).index_add_(
        0, candidate_graph, torch.ones_like(raw))
    return (per / count.clamp(min=1)).mean()


@__import__("torch").no_grad()
def _dataset_loss(model, items, *, device: str, batch_size: int = BATCH_SCENARIOS) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch, target = _batch_from_items(chunk, device)
        loss = _scenario_weighted_loss(
            model(batch), target, batch.cand_graph, len(chunk))
        weighted += float(loss.item()) * len(chunk)
        count += len(chunk)
    return weighted / max(count, 1)


def train_model(
    arm: str,
    train_items: Sequence[tuple[ScenarioGroup, ScenarioSample]],
    validation_items: Sequence[tuple[ScenarioGroup, ScenarioSample]],
    *,
    model_seed: int,
    device: str,
) -> tuple[object, dict]:
    import torch

    torch.manual_seed(model_seed)
    if device == "cpu":
        torch.set_num_threads(1)
    model = build_model(arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(model_seed)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history: list[dict] = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(len(train_items))
        epoch_loss = 0.0
        epoch_count = 0
        for start in range(0, len(order), BATCH_SCENARIOS):
            chunk = [train_items[int(i)] for i in order[start:start + BATCH_SCENARIOS]]
            batch, target = _batch_from_items(chunk, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = _scenario_weighted_loss(
                pred, target, batch.cand_graph, len(chunk))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(chunk)
            epoch_count += len(chunk)
        val_loss = _dataset_loss(model, validation_items, device=device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(epoch_count, 1),
            "validation_loss": val_loss,
        })
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "arm": arm, "model_seed": model_seed, "best_validation_loss": best_loss,
        "epochs": len(history), "history": history,
    }


@__import__("torch").no_grad()
def predict_samples(
    model,
    groups: Sequence[ScenarioGroup],
    *,
    device: str,
    batch_size: int = BATCH_SCENARIOS,
) -> list[np.ndarray]:
    model.eval()
    items = flat_samples(groups)
    out: list[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = ScenarioBatch(
            [group.graph for group, _ in chunk],
            [sample.future for _, sample in chunk]).to(device)
        values = model(batch).detach().cpu().numpy().astype(np.float64)
        ptr = 0
        for group, _ in chunk:
            width = len(group.eligible_cols)
            out.append(values[ptr:ptr + width])
            ptr += width
        if ptr != len(values):
            raise ValueError("prediction/candidate alignment mismatch")
    return out


def _resolution_at_checkpoints(
    gains: Sequence[np.ndarray], eligible_cols: Sequence[int]
) -> tuple[Resolution, int]:
    if len(gains) < SCENARIO_CHECKPOINTS[-1]:
        raise ValueError("offline adapter replay requires all K=16 scenarios")
    resolution: Resolution | None = None
    for checkpoint in SCENARIO_CHECKPOINTS:
        resolution = resolve_stream(
            np.stack(gains[:checkpoint]), eligible_cols)
        if resolution.resolved:
            return resolution, checkpoint
    assert resolution is not None
    return resolution, SCENARIO_CHECKPOINTS[-1]


def simulate_adapter(
    groups: Sequence[ScenarioGroup],
    pred_mean: Sequence[np.ndarray],
    pred_spread: Sequence[np.ndarray],
    *,
    zero_abs: float,
    spread: float,
    _with_task_metrics: bool = True,
) -> dict:
    """Artifact-only replay of authority and hybrid stopping plus exact rollout counts."""
    items = flat_samples(groups)
    if len(items) != len(pred_mean) or len(items) != len(pred_spread):
        raise ValueError("prediction rows do not align with scenario rows")
    offset = 0
    authority_candidate_calls: dict[tuple, int] = defaultdict(int)
    hybrid_candidate_calls: dict[tuple, int] = defaultdict(int)
    authority_base: set[tuple] = set()
    hybrid_base: set[tuple] = set()
    override_total = 0
    override_match = 0
    value_total = 0.0
    value_match = 0.0
    trusted = 0
    total_scenarios = 0
    mismatches = 0
    for group in groups:
        n = len(group.samples)
        means = pred_mean[offset:offset + n]
        spreads = pred_spread[offset:offset + n]
        offset += n
        truth = [sample.gains for sample in group.samples]
        hybrid: list[np.ndarray] = []
        trust_flags: list[bool] = []
        for mean, std, sample in zip(means, spreads, group.samples):
            if mean.shape != sample.gains.shape or std.shape != sample.gains.shape:
                raise ValueError("candidate prediction shape mismatch")
            is_trusted = (
                float(np.max(np.abs(mean))) <= zero_abs and
                float(np.max(std)) <= spread)
            trust_flags.append(is_trusted)
            hybrid.append(np.zeros_like(sample.gains) if is_trusted else sample.gains)
        authority, authority_k = _resolution_at_checkpoints(truth, group.eligible_cols)
        deployed, hybrid_k = _resolution_at_checkpoints(hybrid, group.eligible_cols)
        if (authority.choice != group.authority_choice or
                authority.reason != group.authority_reason):
            raise ValueError("persisted authority outcome does not recompute from labels")
        if authority.choice is not None:
            override_total += 1
            weight = max(float(authority.mean_gain), 1e-9)
            value_total += weight
            if deployed.choice == authority.choice:
                override_match += 1
                value_match += weight
        mismatches += int(deployed.choice != authority.choice)
        width = len(group.eligible_cols)
        for cid in range(authority_k):
            key = (*group.action_key, cid)
            authority_base.add(key)
            authority_candidate_calls[key] += width
        for cid in range(hybrid_k):
            total_scenarios += 1
            if trust_flags[cid]:
                trusted += 1
                continue
            key = (*group.action_key, cid)
            hybrid_base.add(key)
            hybrid_candidate_calls[key] += width
    authority_calls = len(authority_base) + sum(authority_candidate_calls.values())
    hybrid_calls = len(hybrid_base) + sum(hybrid_candidate_calls.values())
    exact_recall = override_match / max(override_total, 1)
    value_recall = value_match / max(value_total, 1e-12)
    rollout_reduction = 1.0 - hybrid_calls / max(authority_calls, 1)
    task_metrics: list[dict] = []
    if _with_task_metrics:
        cursor = 0
        grouped: dict[str, tuple[list[ScenarioGroup], list[np.ndarray], list[np.ndarray]]] = {}
        for group in groups:
            n = len(group.samples)
            if group.task_id not in grouped:
                grouped[group.task_id] = ([], [], [])
            grouped[group.task_id][0].append(group)
            grouped[group.task_id][1].extend(pred_mean[cursor:cursor + n])
            grouped[group.task_id][2].extend(pred_spread[cursor:cursor + n])
            cursor += n
        for task, (task_groups, task_means, task_spreads) in sorted(grouped.items()):
            task_metrics.append({"task_id": task, **simulate_adapter(
                task_groups, task_means, task_spreads,
                zero_abs=zero_abs, spread=spread, _with_task_metrics=False)})
    task_recall_safe = all(
        row["exact_override_recall"] >= 0.95 and
        row["value_weighted_override_recall"] >= 0.99
        for row in task_metrics) if _with_task_metrics else True
    return {
        "zero_abs": float(zero_abs), "spread": float(spread),
        "groups": len(groups), "authority_overrides": override_total,
        "exact_override_recall": exact_recall,
        "value_weighted_override_recall": value_recall,
        "choice_mismatches": mismatches,
        "authority_rollout_calls": authority_calls,
        "hybrid_rollout_calls": hybrid_calls,
        "real_rollout_reduction": rollout_reduction,
        "trusted_scenario_stream_fraction": trusted / max(total_scenarios, 1),
        "task_metrics": task_metrics,
        "gate_pass": (
            exact_recall >= 0.95 and value_recall >= 0.99 and
            rollout_reduction >= 0.70 and task_recall_safe),
    }


def _prediction_errors(
    groups: Sequence[ScenarioGroup], means: Sequence[np.ndarray]
) -> dict:
    truth = [sample.gains for _, sample in flat_samples(groups)]
    if len(truth) != len(means):
        raise ValueError("prediction count mismatch")
    y = np.concatenate(truth)
    p = np.concatenate(means)
    return {
        "candidate_mae": float(np.mean(np.abs(p - y))),
        "candidate_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "candidate_zero_fraction": float(np.mean(np.abs(y) <= 1e-12)),
        "n_scenarios": len(truth), "n_candidates": len(y),
    }


def adapter_grid(
    groups: Sequence[ScenarioGroup],
    means: Sequence[np.ndarray],
    spreads: Sequence[np.ndarray],
) -> list[dict]:
    max_spread = np.asarray([float(np.max(v)) for v in spreads], dtype=np.float64)
    thresholds = sorted(set(float(np.quantile(max_spread, q)) for q in SPREAD_QUANTILES))
    return [simulate_adapter(
        groups, means, spreads, zero_abs=zero_abs, spread=threshold)
        for zero_abs in ZERO_ABS_GRID for threshold in thresholds]


def _adapter_rank(row: Mapping) -> tuple:
    return (
        float(row["real_rollout_reduction"]),
        float(row["value_weighted_override_recall"]),
        -float(row["zero_abs"]),
        -float(row["spread"]),
    )


def _early_stop_view(groups: Sequence[ScenarioGroup]) -> list[tuple[ScenarioGroup, ScenarioSample]]:
    balanced = flat_samples([g for g in groups if g.train_keep])
    balanced.sort(key=lambda item: hashlib.sha256(
        f"{item[0].task_id}|{item[0].seed}|{item[0].cycle}|{item[0].agent}|"
        f"{item[0].order_id}|{item[1].continuation_id}".encode()).digest())
    return balanced[:EARLY_STOP_SAMPLES]


def fit_and_select(
    data_dir: str,
    fit_dir: str,
    *,
    device: str | None = None,
) -> dict:
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_raw = load_collected(data_dir, ("train",))
    validation_raw = load_collected(data_dir, ("validation",))
    retained_train = [group for group in train_raw if group.train_keep]
    graph_normalizer = FoldNormalizer.fit([g.graph for g in retained_train])
    scenario_normalizer = ScenarioNormalizer.fit(retained_train)
    train = normalize_groups(retained_train, graph_normalizer, scenario_normalizer)
    validation = normalize_groups(validation_raw, graph_normalizer, scenario_normalizer)
    train_items = flat_samples(train)
    validation_early = _early_stop_view(validation)
    if not train_items or not validation_early:
        raise ValueError("empty training or early-stop population")
    output = Path(fit_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalizers.json").write_text(json.dumps({
        "graph": _fold_normalizer_dict(graph_normalizer),
        "scenario": scenario_normalizer.as_dict(),
    }, indent=2, sort_keys=True) + "\n")

    arm_predictions: dict[str, list[list[np.ndarray]]] = {}
    training_runs: list[dict] = []
    for arm in ARMS:
        arm_predictions[arm] = []
        for model_seed in MODEL_SEEDS:
            model, run = train_model(
                arm, train_items, validation_early,
                model_seed=model_seed, device=dev)
            checkpoint = output / f"{arm}_seed{model_seed}.pt"
            torch.save({
                "experiment": EXPERIMENT, "arm": arm, "model_seed": model_seed,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }, checkpoint)
            preds = predict_samples(model, validation, device=dev)
            arm_predictions[arm].append(preds)
            training_runs.append({**run, "checkpoint": checkpoint.name})
            print(f"[{EXPERIMENT}] fit {arm} seed={model_seed} "
                  f"epochs={run['epochs']} val={run['best_validation_loss']:.6f}", flush=True)

    arm_rows: list[dict] = []
    grid_rows: list[dict] = []
    selected: dict | None = None
    prediction_dump: dict[str, np.ndarray] = {}
    for arm, per_seed in arm_predictions.items():
        n_samples = len(per_seed[0])
        means: list[np.ndarray] = []
        spreads: list[np.ndarray] = []
        for i in range(n_samples):
            stack = np.stack([seed_pred[i] for seed_pred in per_seed])
            means.append(stack.mean(axis=0))
            spreads.append(stack.std(axis=0))
        errors = _prediction_errors(validation, means)
        grid = adapter_grid(validation, means, spreads)
        for row in grid:
            grid_rows.append({"arm": arm, **row})
        passing = [row for row in grid if row["gate_pass"]]
        best = max(passing, key=_adapter_rank) if passing else max(grid, key=_adapter_rank)
        arm_rows.append({"arm": arm, **errors, **{f"adapter_{k}": v for k, v in best.items()}})
        prediction_dump[f"{arm}__mean"] = np.concatenate(means).astype(np.float32)
        prediction_dump[f"{arm}__spread"] = np.concatenate(spreads).astype(np.float32)
        if arm in SELECTABLE_ARMS and passing:
            candidate = {"arm": arm, **max(passing, key=_adapter_rank)}
            if selected is None or _adapter_rank(candidate) > _adapter_rank(selected):
                selected = candidate

    np.savez_compressed(output / "validation_predictions.npz", **prediction_dump)
    (output / "training_runs.json").write_text(
        json.dumps(training_runs, indent=2, sort_keys=True) + "\n")
    (output / "validation_grid.json").write_text(
        json.dumps(grid_rows, indent=2, sort_keys=True) + "\n")
    result = {
        "experiment": EXPERIMENT,
        "verdict": "VALIDATION_PASS" if selected is not None else "NO_SAFE_ACCELERATION",
        "selected": selected,
        "arm_metrics": arm_rows,
        "population": {
            "train_groups_total": len(train_raw),
            "train_groups_retained": len(retained_train),
            "train_scenarios": len(train_items),
            "validation_groups": len(validation),
            "validation_scenarios": len(flat_samples(validation)),
            "early_stop_scenarios": len(validation_early),
        },
        "device": dev,
    }
    (output / "validation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def load_fit(fit_dir: str, arm: str, *, device: str = "cpu"):
    import torch

    raw = json.loads((Path(fit_dir) / "normalizers.json").read_text())
    graph_normalizer = _fold_normalizer_from_dict(raw["graph"])
    scenario_normalizer = ScenarioNormalizer.from_dict(raw["scenario"])
    models = []
    for seed in MODEL_SEEDS:
        payload = torch.load(
            Path(fit_dir) / f"{arm}_seed{seed}.pt", map_location=device,
            weights_only=False)
        if payload.get("arm") != arm or int(payload.get("model_seed", -1)) != seed:
            raise ValueError("checkpoint identity mismatch")
        model = build_model(arm).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models, graph_normalizer, scenario_normalizer


def ensemble_predictions(
    fit_dir: str,
    arm: str,
    groups_raw: Sequence[ScenarioGroup],
    *,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    models, graph_normalizer, scenario_normalizer = load_fit(
        fit_dir, arm, device=device)
    groups = normalize_groups(groups_raw, graph_normalizer, scenario_normalizer)
    per_seed = [predict_samples(model, groups, device=device) for model in models]
    means, spreads = [], []
    for i in range(len(per_seed[0])):
        stack = np.stack([pred[i] for pred in per_seed])
        means.append(stack.mean(axis=0))
        spreads.append(stack.std(axis=0))
    return means, spreads


def run_offline_test(
    data_dir: str,
    fit_dir: str,
    out_dir: str,
    selection: Mapping,
    *,
    device: str | None = None,
) -> dict:
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    test = load_collected(data_dir, ("test",))
    means, spreads = ensemble_predictions(
        fit_dir, str(selection["arm"]), test, device=dev)
    metrics = simulate_adapter(
        test, means, spreads,
        zero_abs=float(selection["zero_abs"]), spread=float(selection["spread"]))
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "test_predictions.npz",
        mean=np.concatenate(means).astype(np.float32),
        spread=np.concatenate(spreads).astype(np.float32))
    result = {
        "experiment": EXPERIMENT,
        "verdict": "OFFLINE_TEST_PASS" if metrics["gate_pass"] else "NO_SAFE_ACCELERATION",
        "selection": dict(selection), "metrics": metrics,
        "prediction_errors": _prediction_errors(test, means),
        "device": dev,
    }
    (output / "offline_test_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


class HybridScenarioOrderer(BoundedScenarioOrderer):
    """EXP-0042 depth-2 planner with a GNN zero certificate and real fallback."""

    def __init__(
        self,
        *args,
        models: Sequence,
        graph_normalizer: FoldNormalizer,
        scenario_normalizer: ScenarioNormalizer,
        zero_abs: float,
        spread: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, depth=DEPTH, **kwargs)
        if len(models) != len(MODEL_SEEDS):
            raise ValueError("hybrid requires the registered three-model ensemble")
        self.models = list(models)
        self.graph_normalizer = graph_normalizer
        self.scenario_normalizer = scenario_normalizer
        self.zero_abs = float(zero_abs)
        self.spread = float(spread)
        self.surrogate_stream_scenarios = 0
        self.trusted_zero_stream_scenarios = 0
        self.real_stream_scenarios = 0

    def _score_scenario(
        self,
        env: TokenomicsAECEnv,
        agent: str,
        streams: Sequence[StreamCandidate],
        continuation_id: int,
    ) -> Mapping[int, np.ndarray]:
        import torch

        digest = decision_digest(
            self.task_id, self.episode_seed, env.current_cycle, f"agent:{agent}")
        forked = fork_future_demand(
            env, task_id=self.task_id, decision_digest=digest,
            continuation_id=continuation_id, experiment=PLANNER_EXPERIMENT)
        _, aid = env._role[agent]
        graphs: list[GraphArrays] = []
        futures: list[np.ndarray] = []
        for stream in streams:
            eligible = [env.spec_.producer_ids[col] for col in stream.eligible_cols]
            graph = encode_graph(build_economy_graph_view(
                env.system, aid, stream.order_id, eligible))
            graphs.append(graph)
            futures.append(future_product_features(forked, graph.id_maps["product"]))
        normalized_graphs = self.graph_normalizer.apply(graphs)
        normalized_futures = [self.scenario_normalizer.apply(future) for future in futures]
        batch = ScenarioBatch(normalized_graphs, normalized_futures)
        per_model = []
        with torch.no_grad():
            for model in self.models:
                per_model.append(model(batch).cpu().numpy().astype(np.float64))
        means_flat = np.stack(per_model).mean(axis=0)
        spreads_flat = np.stack(per_model).std(axis=0)
        means: dict[int, np.ndarray] = {}
        spreads: dict[int, np.ndarray] = {}
        ptr = 0
        for stream in streams:
            width = len(stream.eligible_cols)
            means[stream.stream_index] = means_flat[ptr:ptr + width]
            spreads[stream.stream_index] = spreads_flat[ptr:ptr + width]
            ptr += width
        if ptr != len(means_flat):
            raise ValueError("hybrid prediction alignment mismatch")

        out: dict[int, np.ndarray] = {}
        uncertain: list[StreamCandidate] = []
        self.surrogate_stream_scenarios += len(streams)
        for stream in streams:
            mean = means[stream.stream_index]
            std = spreads[stream.stream_index]
            trusted = (
                float(np.max(np.abs(mean))) <= self.zero_abs and
                float(np.max(std)) <= self.spread)
            if trusted:
                out[stream.stream_index] = np.zeros(len(stream.eligible_cols), dtype=np.float64)
                self.trusted_zero_stream_scenarios += 1
            else:
                uncertain.append(stream)
                self.real_stream_scenarios += 1
        if uncertain:
            base_units = _rollout_units(copy.deepcopy(forked), self.prod, self.base, DEPTH)
            self.telemetry.rollout_calls += 1
            for stream in uncertain:
                values = []
                for col in stream.eligible_cols:
                    bumped = _one_cycle_bump(
                        self.base, agent, stream.stream_index, col, self.delta,
                        env.current_cycle)
                    units = _rollout_units(copy.deepcopy(forked), self.prod, bumped, DEPTH)
                    values.append(float(units - base_units))
                    self.telemetry.rollout_calls += 1
                out[stream.stream_index] = np.asarray(values, dtype=np.float64)
        return out


def run_hybrid_episode(
    record: TaskRecord,
    prod: Policy,
    *,
    seed: int,
    models: Sequence,
    graph_normalizer: FoldNormalizer,
    scenario_normalizer: ScenarioNormalizer,
    zero_abs: float,
    spread: float,
    horizon: int = HORIZON,
) -> dict:
    planner = HybridScenarioOrderer(
        task_id=record.task_id, episode_seed=seed, prod=prod,
        models=models, graph_normalizer=graph_normalizer,
        scenario_normalizer=scenario_normalizer,
        zero_abs=zero_abs, spread=spread)
    started = time.perf_counter()
    stats = run_episode(joint_stack_env(replay(record), horizon), prod, planner, seed)
    return {
        "task_id": record.task_id, "seed": seed, "arm": "hybrid_gnn_d2", "depth": DEPTH,
        "consumer_units": float(stats["consumer_completed_units"]),
        "consumer_rejected_units": float(stats["consumer_rejected_units"]),
        "dead": int(bool(stats["dead"])), "cycles": int(stats["cycles"]),
        "wall_seconds": time.perf_counter() - started,
        **asdict(planner.telemetry),
        "surrogate_stream_scenarios": planner.surrogate_stream_scenarios,
        "trusted_zero_stream_scenarios": planner.trusted_zero_stream_scenarios,
        "real_stream_scenarios": planner.real_stream_scenarios,
    }


def analyze_closed_loop(rows: Sequence[Mapping]) -> dict:
    keyed: dict[tuple[str, int, str], Mapping] = {}
    for row in rows:
        key = (str(row["task_id"]), int(row["seed"]), str(row["arm"]))
        if key in keyed:
            raise ValueError(f"duplicate closed-loop row {key}")
        if not math.isfinite(float(row["consumer_units"])):
            raise ValueError(f"non-finite closed-loop units {key}")
        keyed[key] = row
    task_rows: list[dict] = []
    episode_floor_relative: list[float] = []
    reference_calls = 0
    hybrid_calls = 0
    for task in TASK_IDS:
        seeds = SPLIT_SEEDS["closed_loop_test"][task]
        triplets = []
        for seed in seeds:
            keys = [(task, seed, arm) for arm in (FLOOR, "search_d2", "hybrid_gnn_d2")]
            if any(key not in keyed for key in keys):
                raise ValueError(f"missing paired closed-loop row for {task}/{seed}")
            floor, depth2, hybrid = (keyed[key] for key in keys)
            triplets.append((floor, depth2, hybrid))
            fu = float(floor["consumer_units"])
            hu = float(hybrid["consumer_units"])
            episode_floor_relative.append((hu - fu) / fu if fu > 0 else 0.0)
            reference_calls += int(depth2["rollout_calls"])
            hybrid_calls += int(hybrid["rollout_calls"])
        floor_mean = statistics.mean(float(row[0]["consumer_units"]) for row in triplets)
        depth2_mean = statistics.mean(float(row[1]["consumer_units"]) for row in triplets)
        hybrid_mean = statistics.mean(float(row[2]["consumer_units"]) for row in triplets)
        depth_gain = depth2_mean - floor_mean
        retention = ((hybrid_mean - floor_mean) / depth_gain
                     if depth_gain > 0 else float("-inf"))
        task_rows.append({
            "task_id": task, "n_episodes": len(triplets),
            "floor_units": floor_mean, "depth2_units": depth2_mean,
            "hybrid_units": hybrid_mean, "depth2_gain": depth_gain,
            "depth2_gain_retention": retention,
            "depth2_dead_rate": statistics.mean(int(row[1]["dead"]) for row in triplets),
            "hybrid_dead_rate": statistics.mean(int(row[2]["dead"]) for row in triplets),
            "mean_hybrid_wall_seconds": statistics.mean(
                float(row[2]["wall_seconds"]) for row in triplets),
        })
    mean_retention = statistics.mean(row["depth2_gain_retention"] for row in task_rows)
    rollout_reduction = 1.0 - hybrid_calls / max(reference_calls, 1)
    death_safe = all(row["hybrid_dead_rate"] <= row["depth2_dead_rate"] for row in task_rows)
    gate = (
        mean_retention >= 0.95 and
        all(row["depth2_gain_retention"] >= 0.90 for row in task_rows) and
        rollout_reduction >= 0.70 and
        min(episode_floor_relative) >= -0.10 and death_safe)
    return {
        "experiment": EXPERIMENT,
        "verdict": "PASS_SURROGATE" if gate else "OFFLINE_ONLY",
        "gate_pass": gate,
        "mean_depth2_gain_retention": mean_retention,
        "real_rollout_reduction": rollout_reduction,
        "reference_rollout_calls": reference_calls,
        "hybrid_rollout_calls": hybrid_calls,
        "worst_episode_relative_vs_floor": min(episode_floor_relative),
        "death_safe": death_safe,
        "task_metrics": task_rows,
    }


_WORKER_PROD: Policy | None = None
_WORKER_MODELS: Sequence | None = None
_WORKER_GRAPH_NORMALIZER: FoldNormalizer | None = None
_WORKER_SCENARIO_NORMALIZER: ScenarioNormalizer | None = None
_WORKER_SELECTION: dict | None = None


def _worker_init(prod_ckpt: str, fit_dir: str = "", selection_json: str = "") -> None:
    global _WORKER_PROD, _WORKER_MODELS, _WORKER_GRAPH_NORMALIZER
    global _WORKER_SCENARIO_NORMALIZER, _WORKER_SELECTION
    import torch

    torch.set_num_threads(1)
    _WORKER_PROD = scorer_producing_policy(load_producer_score_model(prod_ckpt))
    if fit_dir and selection_json:
        _WORKER_SELECTION = json.loads(selection_json)
        (_WORKER_MODELS, _WORKER_GRAPH_NORMALIZER,
         _WORKER_SCENARIO_NORMALIZER) = load_fit(
            fit_dir, _WORKER_SELECTION["arm"], device="cpu")


def _collect_job(job: tuple[TaskRecord, str, int, str]) -> dict:
    record, split, seed, out_dir = job
    if _WORKER_PROD is None:
        raise RuntimeError("worker producer is not initialized")
    return collect_episode(
        record, _WORKER_PROD, split=split, seed=seed, out_dir=out_dir)


def _closed_loop_job(job: tuple[TaskRecord, int, str]) -> dict:
    record, seed, arm = job
    if _WORKER_PROD is None:
        raise RuntimeError("worker producer is not initialized")
    if arm == FLOOR:
        return run_floor_episode(record, _WORKER_PROD, seed=seed)
    if arm == "search_d2":
        return run_search_episode(record, _WORKER_PROD, seed=seed, depth=DEPTH)
    if (_WORKER_MODELS is None or _WORKER_GRAPH_NORMALIZER is None or
            _WORKER_SCENARIO_NORMALIZER is None or _WORKER_SELECTION is None):
        raise RuntimeError("worker surrogate is not initialized")
    return run_hybrid_episode(
        record, _WORKER_PROD, seed=seed, models=_WORKER_MODELS,
        graph_normalizer=_WORKER_GRAPH_NORMALIZER,
        scenario_normalizer=_WORKER_SCENARIO_NORMALIZER,
        zero_abs=float(_WORKER_SELECTION["zero_abs"]),
        spread=float(_WORKER_SELECTION["spread"]))


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    import csv

    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_collection(
    records: Sequence[TaskRecord],
    *,
    splits: Sequence[str],
    out_dir: str,
    prod_ckpt: str,
    workers: int,
) -> list[dict]:
    by_id = {record.task_id: record for record in records}
    if set(TASK_IDS) - set(by_id):
        raise ValueError("archive is missing a registered task")
    jobs = [
        (by_id[task], split, seed, out_dir)
        for split in splits
        for task, seeds in SPLIT_SEEDS[split].items()
        for seed in seeds
    ]
    rows: list[dict] = []
    summary_path = Path(out_dir) / f"collection_{'_'.join(splits)}.csv"
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init, initargs=(prod_ckpt,)) as pool:
        futures = {pool.submit(_collect_job, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r["split"], r["task_id"], int(r["seed"])))
            _write_csv(summary_path, rows)
            print(f"[{EXPERIMENT}] collected {row['split']} {row['task_id'][-4:]} "
                  f"seed={row['seed']} groups={row['groups']} "
                  f"wall={row['wall_seconds']:.1f}s", flush=True)
    return rows


def run_closed_loop(
    records: Sequence[TaskRecord],
    *,
    prod_ckpt: str,
    fit_dir: str,
    selection: Mapping,
    out_dir: str,
    workers: int,
) -> dict:
    by_id = {record.task_id: record for record in records}
    jobs = [
        (by_id[task], seed, arm)
        for task, seeds in SPLIT_SEEDS["closed_loop_test"].items()
        for seed in seeds for arm in (FLOOR, "search_d2", "hybrid_gnn_d2")
    ]
    rows: list[dict] = []
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
            initargs=(prod_ckpt, fit_dir, json.dumps(dict(selection)))) as pool:
        futures = {pool.submit(_closed_loop_job, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r["task_id"], int(r["seed"]), r["arm"]))
            _write_csv(output / "closed_loop_episodes.csv", rows)
            print(f"[{EXPERIMENT}] closed-loop {row['task_id'][-4:]} seed={row['seed']} "
                  f"{row['arm']} units={row['consumer_units']} "
                  f"calls={row.get('rollout_calls', 0)} wall={row['wall_seconds']:.1f}s",
                  flush=True)
    result = analyze_closed_loop(rows)
    _write_csv(output / "closed_loop_episodes.csv", rows)
    _write_csv(output / "closed_loop_by_task.csv", result["task_metrics"])
    (output / "closed_loop_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_registered_experiment(
    records: Sequence[TaskRecord],
    *,
    archive: str,
    prod_ckpt: str,
    out_dir: str,
    source_commit: str,
    workers: int,
    device: str | None = None,
) -> dict:
    validate_registry()
    manifest = {
        "archive": {"path": archive, "sha256": sha256_file(archive)},
        "producer_ckpt": {"path": prod_ckpt, "sha256": sha256_file(prod_ckpt)},
    }
    assert_inputs_match(manifest, REGISTERED_INPUT_SHA256)
    if len(source_commit) != 40:
        raise ValueError("registered run requires a full 40-character source commit")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_spec = {
        "experiment": EXPERIMENT,
        "inputs": manifest,
        "source": source_fingerprint(source_commit),
        "environment": environment_manifest(),
        "schema_hash": schema_hash(),
        "split_seeds": {split: {task: list(seeds) for task, seeds in task_map.items()}
                        for split, task_map in SPLIT_SEEDS.items()},
        "workers": workers,
        "models": {"arms": ARMS, "model_seeds": list(MODEL_SEEDS)},
    }
    (output / "run_spec.json").write_text(
        json.dumps(run_spec, indent=2, sort_keys=True) + "\n")

    data_dir = str(output / "data")
    fit_dir = str(output / "fit")
    run_collection(
        records, splits=("train", "validation"), out_dir=data_dir,
        prod_ckpt=prod_ckpt, workers=workers)
    validation = fit_and_select(data_dir, fit_dir, device=device)
    if validation["selected"] is None:
        final = {"experiment": EXPERIMENT, "verdict": "NO_SAFE_ACCELERATION",
                 "validation": validation, "offline_test": None, "closed_loop": None}
        (output / "result.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
        return final

    # The test tranche is intentionally absent until validation has locked the adapter.
    run_collection(
        records, splits=("test",), out_dir=data_dir,
        prod_ckpt=prod_ckpt, workers=workers)
    offline_test = run_offline_test(
        data_dir, fit_dir, str(output / "offline_test"), validation["selected"],
        device=device)
    if not offline_test["metrics"]["gate_pass"]:
        final = {"experiment": EXPERIMENT, "verdict": "OFFLINE_ONLY",
                 "validation": validation, "offline_test": offline_test, "closed_loop": None}
        (output / "result.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
        return final

    closed_loop = run_closed_loop(
        records, prod_ckpt=prod_ckpt, fit_dir=fit_dir,
        selection=validation["selected"], out_dir=str(output / "closed_loop"),
        workers=workers)
    final = {
        "experiment": EXPERIMENT, "verdict": closed_loop["verdict"],
        "validation": validation, "offline_test": offline_test,
        "closed_loop": closed_loop,
    }
    (output / "result.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def run_smoke(
    records: Sequence[TaskRecord], *, prod_ckpt: str, out_dir: str
) -> dict:
    by_id = {record.task_id: record for record in records}
    _worker_init(prod_ckpt)
    assert _WORKER_PROD is not None
    return collect_episode(
        by_id[TASK_S29], _WORKER_PROD, split="train", seed=4299,
        out_dir=out_dir, horizon=20, max_search_actions=1)


def main() -> None:
    from .curriculum_replay import _load_archive

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--prod-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    records = _load_archive(args.archive)
    if args.smoke:
        result = run_smoke(records, prod_ckpt=args.prod_ckpt, out_dir=args.out_dir)
    else:
        result = run_registered_experiment(
            records, archive=args.archive, prod_ckpt=args.prod_ckpt,
            out_dir=args.out_dir, source_commit=args.source_commit,
            workers=args.workers, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
