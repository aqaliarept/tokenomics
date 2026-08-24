"""EXP-0040: audit marginal-value labels across counterfactual demand futures.

EXP-0039's GNN fitted its training trajectories and fell to random routing on held-out
trajectories.  Before buying another supervised dataset, this module asks whether the
teacher target is stable after future exogenous demand is marginalized out.  It has two
deliberately separate halves:

* the collector replays an authority-selected subset of EXP-0039 states and persists the
  original label plus raw labels from counterfactual future-demand continuations;
* the analyzer reads only those rows, computes the pre-registered convergence/usefulness
  gates, and writes the EXP-0040 target licence consumed by EXP-0041.

The raw CSV is the audit authority.  No simulator or checkpoint is needed to recompute the
verdict, and no GNN is trained here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .graph_generalization_dump import (
    TASK_IDS,
    build_view,
    continuation_label_matrix,
)
from .graph_value_router_dump import (
    _assert_finite,
    decision_digest,
    encode_graph,
    environment_manifest,
    schema_hash,
    sha256_file,
    source_fingerprint,
)

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv
    from .baselines import Policy
    from .curriculum import TaskRecord


EXPERIMENT = "EXP-0040"
PRIMARY_CONTINUATIONS = 16
ESCALATED_CONTINUATIONS = 32
CONTINUATION_COUNTS = (4, 8, 16)
SIGNAL_PER_STRATUM = 10
FLAT_PER_STRATUM = 3
SPLIT_REPEATS = 100
FLAT_TOL = 1e-9
REGISTERED_SEEDS: dict[str, frozenset[int]] = {
    TASK_IDS[0]: frozenset({29, 30, 31}),
    TASK_IDS[1]: frozenset({77, 78, 79}),
    TASK_IDS[2]: frozenset({45, 46, 47}),
}

# Frozen by experiments/EXP-0040-rollout-target-stability/spec.json.  The collector checks
# every byte before replaying an episode; a historical artifact silently regenerated under
# different code is not the registered experiment.
REGISTERED_INPUT_SHA256: dict[str, str] = {
    "archive": "bc6ca4ebacff6835571453cf7c0bf10de5c366cc208674f44cf3a5860b778ceb",
    "producer_ckpt": "8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea",
    "exp0039_decisions": "09771dca6d6ee014374b196bf6ae08074ebf21e3d2bbac76d87affd4f9262bc1",
    "exp0039_candidates": "d2a82c7471bf983e5b9d20b844a73f73de1c265ee6d969f61af54b644886dfd2",
    "actor_decisions": "b22af97ed201c316144f6ed4305841e760858284412faa7f810b2f81adb9bd02",
    "actor_producers": "06c5cd55a6a70abef25288248a42657ee8eef14a266acd64c32427bb6d17daf2",
}


DecisionKey = tuple[str, int, int, str]
CandidateKey = tuple[str, int, int, str, str]


@dataclass(frozen=True)
class CandidateAuthority:
    """One candidate's original label and deployed-actor score."""

    producer_id: str
    original_label: float
    actor_score: float


@dataclass(frozen=True)
class SelectedDecision:
    """One authority-bound pre-action decision selected for the target audit."""

    decision_index: int
    task_id: str
    seed: int
    cycle: int
    agent: str
    order_id: str
    stream_index: int
    decision_digest: int
    cycle_quartile: int
    sample_class: str
    actor_best_producer: str
    candidates: tuple[CandidateAuthority, ...]
    selection_rank: str

    @property
    def key(self) -> DecisionKey:
        return (self.task_id, self.seed, self.cycle, self.order_id)


@dataclass(frozen=True)
class Authority:
    """Validated EXP-0039 label rows joined to the canonical deployed-actor rows."""

    decisions: tuple[SelectedDecision, ...]
    inputs: dict[str, dict[str, str | int]]


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _input_manifest(paths: Mapping[str, str], *, require_registered: bool) -> dict[str, dict]:
    missing = sorted(set(REGISTERED_INPUT_SHA256) - set(paths))
    if missing:
        raise ValueError(f"missing authority inputs: {missing}")
    out: dict[str, dict] = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = REGISTERED_INPUT_SHA256.get(name)
        if require_registered and actual != expected:
            raise ValueError(
                f"{name} SHA-256 {actual} != registered {expected}; stop before rollout spend")
        out[name] = {"path": os.path.abspath(path), "sha256": actual,
                     "size": os.path.getsize(path)}
    return out


def _decision_key(row: Mapping[str, str]) -> DecisionKey:
    return (row["task_id"], int(row["seed"]), int(row["cycle"]), row["order_id"])


def _candidate_key(row: Mapping[str, str]) -> CandidateKey:
    return (*_decision_key(row), row["producer_id"])


def _winner_set(values: Mapping[str, float], *, tol: float = FLAT_TOL) -> frozenset[str]:
    best = max(values.values())
    return frozenset(k for k, v in values.items() if best - v <= tol)


def _join_authorities(
    exp_decisions: Sequence[Mapping[str, str]],
    exp_candidates: Sequence[Mapping[str, str]],
    actor_decisions: Sequence[Mapping[str, str]],
    actor_producers: Sequence[Mapping[str, str]],
) -> tuple[SelectedDecision, ...]:
    """Exact key/set join; authority drift is detected before sampling can hide it."""
    actor_dec_by_key = {_decision_key(r): r for r in actor_decisions}
    if len(actor_dec_by_key) != len(actor_decisions):
        raise ValueError("canonical actor decisions contain duplicate decision keys")
    exp_keys = [_decision_key(r) for r in exp_decisions]
    if len(set(exp_keys)) != len(exp_keys):
        raise ValueError("EXP-0039 decisions contain duplicate decision keys")
    if set(exp_keys) != set(actor_dec_by_key):
        missing = set(exp_keys) - set(actor_dec_by_key)
        extra = set(actor_dec_by_key) - set(exp_keys)
        raise ValueError(f"actor/EXP-0039 decision coverage mismatch: missing={len(missing)}, "
                         f"extra={len(extra)}")

    exp_by_dec: dict[int, list[Mapping[str, str]]] = {}
    for row in exp_candidates:
        exp_by_dec.setdefault(int(row["decision_index"]), []).append(row)
    actor_by_key = {_candidate_key(r): r for r in actor_producers if _truthy(r["eligible"])}
    if len(actor_by_key) != sum(_truthy(r["eligible"]) for r in actor_producers):
        raise ValueError("canonical actor producers contain duplicate eligible candidate keys")

    out: list[SelectedDecision] = []
    seen_candidates: set[CandidateKey] = set()
    for di, drow in enumerate(exp_decisions):
        key = _decision_key(drow)
        crows = sorted(exp_by_dec.get(di, ()), key=lambda r: int(r["candidate_index"]))
        if len(crows) != int(drow["n_eligible"]) or len(crows) < 2:
            raise ValueError(f"{key}: candidate count {len(crows)} != n_eligible "
                             f"{drow['n_eligible']}")
        candidates: list[CandidateAuthority] = []
        for row in crows:
            ck = _candidate_key(row)
            arow = actor_by_key.get(ck)
            if arow is None:
                raise ValueError(f"{ck}: missing canonical actor candidate")
            seen_candidates.add(ck)
            candidates.append(CandidateAuthority(
                producer_id=row["producer_id"], original_label=float(row["teacher_label"]),
                actor_score=float(arow["actor_score"])))
        actor_scores = {c.producer_id: c.actor_score for c in candidates}
        actor_winners = _winner_set(actor_scores)
        recorded_actor = actor_dec_by_key[key]["actor_best_producer"]
        if recorded_actor not in actor_winners:
            raise ValueError(f"{key}: actor_best_producer {recorded_actor} not in score winners "
                             f"{sorted(actor_winners)}")
        labels = [c.original_label for c in candidates]
        flat = max(labels) - min(labels) <= FLAT_TOL
        digest = decision_digest(key[0], key[1], key[2], key[3])
        rank = hashlib.sha256(str(digest).encode()).hexdigest()
        out.append(SelectedDecision(
            decision_index=di, task_id=key[0], seed=key[1], cycle=key[2],
            agent=drow["agent"], order_id=key[3], stream_index=int(drow["stream_index"]),
            decision_digest=digest, cycle_quartile=min(3, key[2] // 25),
            sample_class="flat" if flat else "signal",
            actor_best_producer=recorded_actor, candidates=tuple(candidates),
            selection_rank=rank))
    if seen_candidates != set(actor_by_key):
        raise ValueError(f"actor candidate coverage has {len(set(actor_by_key) - seen_candidates)} "
                         "rows outside EXP-0039")
    return tuple(out)


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_authority(
    *, exp0039_decisions: str, exp0039_candidates: str, actor_decisions: str,
    actor_producers: str, archive: str, producer_ckpt: str,
    require_registered: bool = True,
) -> Authority:
    """Load and fully align all frozen inputs before any simulator work begins."""
    paths = {
        "archive": archive, "producer_ckpt": producer_ckpt,
        "exp0039_decisions": exp0039_decisions, "exp0039_candidates": exp0039_candidates,
        "actor_decisions": actor_decisions, "actor_producers": actor_producers,
    }
    manifest = _input_manifest(paths, require_registered=require_registered)
    decisions = _join_authorities(
        _read_csv(exp0039_decisions), _read_csv(exp0039_candidates),
        _read_csv(actor_decisions), _read_csv(actor_producers))
    if require_registered:
        tasks = {d.task_id for d in decisions}
        if tasks != set(TASK_IDS):
            raise ValueError(f"authority tasks {sorted(tasks)} != registered {sorted(TASK_IDS)}")
        expected_seeds = {t: set(seeds) for t, seeds in REGISTERED_SEEDS.items()}
        actual_seeds = {t: {d.seed for d in decisions if d.task_id == t} for t in tasks}
        if actual_seeds != expected_seeds:
            raise ValueError(f"authority seed sets {actual_seeds} != registered {expected_seeds}")
    return Authority(decisions, manifest)


def select_sample(
    authority: Authority, *, signal_per_stratum: int = SIGNAL_PER_STRATUM,
    flat_per_stratum: int = FLAT_PER_STRATUM, max_decisions: int | None = None,
) -> tuple[list[SelectedDecision], dict]:
    """Lowest-SHA selection within `(task, seed, quartile, label class)`, never backfilled."""
    if signal_per_stratum < 0 or flat_per_stratum < 0:
        raise ValueError("per-stratum limits must be non-negative")
    strata: dict[tuple[str, int, int, str], list[SelectedDecision]] = {}
    for d in authority.decisions:
        strata.setdefault((d.task_id, d.seed, d.cycle_quartile, d.sample_class), []).append(d)
    chosen: list[SelectedDecision] = []
    stratum_rows: list[dict] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda d: (d.selection_rank, d.decision_digest))
        limit = signal_per_stratum if key[3] == "signal" else flat_per_stratum
        take = rows[:limit]
        chosen.extend(take)
        stratum_rows.append({
            "task_id": key[0], "seed": key[1], "cycle_quartile": key[2],
            "sample_class": key[3], "available": len(rows), "limit": limit,
            "selected": len(take), "shortfall": max(0, limit - len(take)),
        })
    chosen.sort(key=lambda d: (d.task_id, d.seed, d.cycle, d.order_id))
    if max_decisions is not None:
        chosen = sorted(chosen, key=lambda d: (d.selection_rank, d.decision_digest))[:max_decisions]
        chosen.sort(key=lambda d: (d.task_id, d.seed, d.cycle, d.order_id))
    manifest = {
        "experiment": EXPERIMENT, "method": "lowest_sha256_rank_within_stratum",
        "strata": stratum_rows, "n_selected": len(chosen),
        "n_signal": sum(d.sample_class == "signal" for d in chosen),
        "n_flat": sum(d.sample_class == "flat" for d in chosen),
        "max_decisions": max_decisions,
    }
    return chosen, manifest


def sample_rows(selected: Sequence[SelectedDecision]) -> list[dict]:
    return [{
        "decision_index": d.decision_index, "task_id": d.task_id, "seed": d.seed,
        "cycle": d.cycle, "agent": d.agent, "order_id": d.order_id,
        "stream_index": d.stream_index, "decision_digest": d.decision_digest,
        "cycle_quartile": d.cycle_quartile, "sample_class": d.sample_class,
        "n_candidates": len(d.candidates), "actor_best_producer": d.actor_best_producer,
        "selection_rank": d.selection_rank,
    } for d in selected]


def _graphs_equal(a: object, b: object) -> bool:
    for name in ("node_features", "edge_index", "edge_attr"):
        ad = getattr(a, name)
        bd = getattr(b, name)
        if set(ad) != set(bd) or any(not np.array_equal(ad[k], bd[k]) for k in ad):
            return False
    return getattr(a, "id_maps") == getattr(b, "id_maps")


def collect_labels(
    records: Sequence["TaskRecord"], *, prod: "Policy", selected: Sequence[SelectedDecision],
    continuation_ids: Sequence[int], out_csv: str, base: "Policy | None" = None,
    delta: float = 1.0, lookahead: int = 8, horizon: int = 100,
    append: bool = False, log: "Callable[..., None]" = print,
) -> list[dict]:
    """Replay selected states and persist original + raw continuation labels.

    On append, original rows and existing continuation IDs are retained and only missing IDs
    are simulated.  This implements the registered 16 -> 32 escalation without re-buying the
    first tranche.
    """
    from ..env.future_fork import fork_future_demand
    from .baselines import ordering_blend
    from .curriculum import replay
    from .joint_stack import joint_stack_env
    from .value_router import marginal_value_labels

    ids = tuple(sorted(set(int(c) for c in continuation_ids)))
    if any(c < 0 for c in ids):
        raise ValueError("continuation IDs must be non-negative")
    base = base if base is not None else ordering_blend(0.5)
    existing = _read_csv(out_csv) if append and os.path.isfile(out_csv) else []
    existing_keys = {(int(r["decision_digest"]), int(r["continuation_id"]), r["producer_id"])
                     for r in existing if int(r["continuation_id"]) >= 0}
    original_keys = {(int(r["decision_digest"]), r["producer_id"]) for r in existing
                     if int(r["continuation_id"]) == -1}
    rows: list[dict] = [dict(r) for r in existing]
    by_key = {d.key: d for d in selected}
    by_task_seed: dict[tuple[str, int], set[DecisionKey]] = {}
    for d in selected:
        by_task_seed.setdefault((d.task_id, d.seed), set()).add(d.key)
    records_by_id = {r.task_id: r for r in records}
    visited: set[DecisionKey] = set()

    for (task_id, seed), wanted in sorted(by_task_seed.items()):
        cfg = replay(records_by_id[task_id])
        env = joint_stack_env(cfg, horizon)
        env.reset(seed=seed)
        before = len(rows)
        for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
            # A smoke can select early states from a horizon-100 authority. Keep horizon=100 so
            # randomized consumers draw the identical schedules, but stop as soon as this
            # episode's selected states are collected instead of playing an irrelevant suffix.
            if wanted <= visited:
                break
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            role, aid = env._role[agent]
            if role == "prod":
                env.step(prod(env, agent))
                continue
            view = env.system.ordering_agents[aid].view(env.system.producer_infos)
            incoming = sorted(view.incoming)
            for k, oid in enumerate(incoming[:env.spec_.k_stream]):
                key = (task_id, seed, env.current_cycle, oid)
                if key not in wanted:
                    continue
                d = by_key[key]
                if agent != d.agent or k != d.stream_index:
                    raise ValueError(f"{key}: replay boundary drift (agent={agent}, k={k})")
                elig = list(view.eligible[oid])
                expected_ids = [c.producer_id for c in d.candidates]
                if set(elig) != set(expected_ids):
                    raise ValueError(f"{key}: eligible producers {elig} != authority {expected_ids}")
                cols = [env.spec_.producer_ids.index(p) for p in expected_ids]
                graph = encode_graph(build_view(env, aid, oid, expected_ids))
                _assert_finite(graph)
                probe_fork = fork_future_demand(
                    env, task_id=task_id, decision_digest=d.decision_digest,
                    continuation_id=ids[0] if ids else 0)
                fork_graph = encode_graph(build_view(probe_fork, aid, oid, expected_ids))
                if not _graphs_equal(graph, fork_graph):
                    raise ValueError(f"{key}: future fork changed the deployable pre-action graph")
                original = marginal_value_labels(env, prod, base, agent, k, cols, delta, lookahead)
                for c, col in zip(d.candidates, cols):
                    if not math.isclose(float(original[col]), c.original_label,
                                        rel_tol=0.0, abs_tol=1e-6):
                        raise ValueError(f"{key}/{c.producer_id}: original-label parity failed: "
                                         f"{original[col]} != {c.original_label}")
                    ok = (d.decision_digest, c.producer_id)
                    if ok not in original_keys:
                        rows.append(_raw_row(d, c.producer_id, -1, float(original[col])))
                        original_keys.add(ok)
                missing_ids = [cid for cid in ids if any(
                    (d.decision_digest, cid, p) not in existing_keys for p in expected_ids)]
                if missing_ids:
                    matrix = continuation_label_matrix(
                        env, prod, base, agent, k, cols, delta=delta, lookahead=lookahead,
                        task_id=task_id, digest=d.decision_digest, continuation_ids=missing_ids)
                    for ri, cid in enumerate(missing_ids):
                        for p, col in zip(expected_ids, cols):
                            ek = (d.decision_digest, cid, p)
                            if ek not in existing_keys:
                                rows.append(_raw_row(d, p, cid, float(matrix[ri, col])))
                                existing_keys.add(ek)
                visited.add(key)
            env.step(base(env, agent))
        log(f"[exp0040] {task_id} seed {seed}: +{len(rows) - before} raw rows")
    if visited != set(by_key):
        missing = sorted(set(by_key) - visited)
        raise ValueError(f"replay did not visit {len(missing)} selected decisions: {missing[:3]}")
    rows.sort(key=lambda r: (r["task_id"], int(r["seed"]), int(r["cycle"]),
                             r["order_id"], int(r["continuation_id"]), r["producer_id"]))
    _write_csv(out_csv, rows)
    return rows


def _raw_row(d: SelectedDecision, producer_id: str, continuation_id: int,
             label: float) -> dict:
    return {
        "decision_index": d.decision_index, "task_id": d.task_id, "seed": d.seed,
        "cycle": d.cycle, "agent": d.agent, "order_id": d.order_id,
        "stream_index": d.stream_index, "decision_digest": d.decision_digest,
        "cycle_quartile": d.cycle_quartile, "sample_class": d.sample_class,
        "actor_best_producer": d.actor_best_producer, "producer_id": producer_id,
        "continuation_id": continuation_id, "label": repr(float(label)),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return 0.0
    ar, br = _rankdata(a), _rankdata(b)
    if float(np.std(ar)) == 0.0 or float(np.std(br)) == 0.0:
        return 1.0 if np.array_equal(ar, br) else 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _bootstrap_interval(values: Sequence[float], seed: int) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    # `decision_digest` is stored as a signed int64, so a valid digest may be negative.
    rng = np.random.default_rng(seed % (1 << 128))
    arr = np.asarray(values, dtype=np.float64)
    means = np.mean(rng.choice(arr, size=(1000, len(arr)), replace=True), axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return float(lo), float(hi)


def _raw_cube(rows: Sequence[Mapping[str, str]]) -> tuple[dict[int, dict], list[int]]:
    by_dec: dict[int, dict] = {}
    cont_ids: set[int] = set()
    seen: set[tuple[int, int, str]] = set()
    for row in rows:
        digest = int(row["decision_digest"])
        cid = int(row["continuation_id"])
        producer = row["producer_id"]
        key = (digest, cid, producer)
        if key in seen:
            raise ValueError(f"duplicate raw label row {key}")
        seen.add(key)
        d = by_dec.setdefault(digest, {
            "task_id": row["task_id"], "seed": int(row["seed"]),
            "cycle": int(row["cycle"]), "order_id": row["order_id"],
            "sample_class": row["sample_class"],
            "actor_best_producer": row["actor_best_producer"],
            "original": {}, "continuations": {},
        })
        if (d["task_id"], d["seed"], d["cycle"], d["order_id"]) != (
                row["task_id"], int(row["seed"]), int(row["cycle"]), row["order_id"]):
            raise ValueError(f"decision digest collision/drift {digest}")
        value = float(row["label"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite label at {key}")
        if cid == -1:
            d["original"][producer] = value
        elif cid >= 0:
            cont_ids.add(cid)
            d["continuations"].setdefault(cid, {})[producer] = value
        else:
            raise ValueError(f"invalid continuation_id {cid}")
    ids = sorted(cont_ids)
    if ids != list(range(len(ids))):
        raise ValueError(f"continuation IDs must be a contiguous 0..K-1 block, got {ids}")
    for digest, d in by_dec.items():
        producers = set(d["original"])
        if len(producers) < 2:
            raise ValueError(f"{digest}: fewer than two candidates")
        if d["actor_best_producer"] not in producers:
            raise ValueError(f"{digest}: actor pick is not an eligible producer")
        if set(d["continuations"]) != set(ids):
            raise ValueError(f"{digest}: continuation coverage mismatch")
        for cid in ids:
            if set(d["continuations"][cid]) != producers:
                raise ValueError(f"{digest}/{cid}: candidate coverage mismatch")
    return by_dec, ids


def _balanced_splits(ids: Sequence[int], *, repeats: int, seed_blob: str,
                     size: int) -> Iterable[tuple[list[int], list[int]]]:
    if 2 * size > len(ids):
        return
    seed = int.from_bytes(hashlib.sha256(seed_blob.encode()).digest(), "big")
    rng = np.random.default_rng(seed)
    arr = np.asarray(ids, dtype=np.int64)
    for _ in range(repeats):
        p = rng.permutation(arr)
        yield p[:size].tolist(), p[size:2 * size].tolist()


def _convergence_metrics(decisions: Sequence[dict], ids: Sequence[int], *, size: int,
                         repeats: int = SPLIT_REPEATS) -> dict:
    spearman: list[float] = []
    overlaps: list[float] = []
    margin_changes: list[float] = []
    actor_regret_changes: list[float] = []
    ensemble_regret_changes: list[float] = []
    for d in decisions:
        producers = sorted(d["original"])
        for a_ids, b_ids in _balanced_splits(
                ids, repeats=repeats,
                seed_blob=f"{EXPERIMENT}|split|{d['task_id']}|{d['seed']}|{d['cycle']}|"
                          f"{d['order_id']}|{size}", size=size):
            av = {p: statistics.mean(d["continuations"][cid][p] for cid in a_ids)
                  for p in producers}
            bv = {p: statistics.mean(d["continuations"][cid][p] for cid in b_ids)
                  for p in producers}
            spearman.append(_spearman([av[p] for p in producers], [bv[p] for p in producers]))
            aw, bw = _winner_set(av), _winner_set(bv)
            overlaps.append(_jaccard(aw, bw))
            margin_changes.append(abs(_top_two_margin(av) - _top_two_margin(bv)))
            actor = d["actor_best_producer"]
            actor_regret_changes.append(abs((max(av.values()) - av[actor]) -
                                             (max(bv.values()) - bv[actor])))
            ensemble_regret_changes.append(abs(
                _selector_regret(av, aw) - _selector_regret(bv, bw)))
    return {
        "estimator_k": size,
        "median_split_half_spearman": float(statistics.median(spearman)),
        "mean_split_half_winner_overlap": float(statistics.mean(overlaps)),
        "mean_abs_margin_change": float(statistics.mean(margin_changes)),
        "mean_abs_actor_regret_change": float(statistics.mean(actor_regret_changes)),
        "mean_abs_ensemble_regret_change": float(statistics.mean(ensemble_regret_changes)),
        "converged": (statistics.median(spearman) >= 0.70 and
                      statistics.mean(overlaps) >= 0.80),
    }


def _top_two_margin(values: Mapping[str, float]) -> float:
    ranked = sorted(values.values(), reverse=True)
    return float(ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0


def _selector_regret(values: Mapping[str, float], selected: frozenset[str]) -> float:
    return max(values.values()) - statistics.mean(values[p] for p in selected)


def _decision_summary(digest: int, d: dict, ids: Sequence[int]) -> tuple[dict, list[dict]]:
    producers = sorted(d["original"])
    means = {p: statistics.mean(d["continuations"][cid][p] for cid in ids) for p in producers}
    mean_winners = _winner_set(means)
    original_winners = _winner_set(d["original"])
    actor = d["actor_best_producer"]
    actor_regrets: list[float] = []
    ensemble_regrets: list[float] = []
    random_regrets: list[float] = []
    perfect_info: list[float] = []
    candidate_rows: list[dict] = []
    for p in producers:
        vals = [d["continuations"][cid][p] for cid in ids]
        lo, hi = _bootstrap_interval(vals, digest ^ int.from_bytes(hashlib.sha256(p.encode()).digest()[:8], "big"))
        candidate_rows.append({
            "decision_digest": digest, "task_id": d["task_id"], "seed": d["seed"],
            "cycle": d["cycle"], "order_id": d["order_id"], "sample_class": d["sample_class"],
            "producer_id": p, "original_label": d["original"][p],
            "continuation_mean": means[p], "continuation_std": statistics.pstdev(vals),
            "continuation_stderr": statistics.pstdev(vals) / math.sqrt(len(vals)),
            "bootstrap_lo": lo, "bootstrap_hi": hi,
            "winner_probability": statistics.mean(
                p in _winner_set(d["continuations"][cid]) for cid in ids),
            "mean_winner": int(p in mean_winners), "actor_selected": int(p == actor),
        })
    for cid in ids:
        values = d["continuations"][cid]
        oracle = max(values.values())
        actor_regrets.append(oracle - values[actor])
        ensemble_value = statistics.mean(values[p] for p in mean_winners)
        ensemble_regrets.append(oracle - ensemble_value)
        random_regrets.append(oracle - statistics.mean(values.values()))
        perfect_info.append(oracle - ensemble_value)
    candidate_means = list(means.values())
    within_variances = [statistics.pvariance(
        [d["continuations"][cid][p] for cid in ids]) for p in producers]
    denom = statistics.mean(within_variances)
    # Persisted metrics must stay finite. A deterministic target has zero within-candidate
    # variance; the epsilon preserves its intended "very high SNR" meaning without JSON Infinity.
    snr = statistics.pvariance(candidate_means) / max(denom, 1e-12)
    summary = {
        "decision_digest": digest, "task_id": d["task_id"], "seed": d["seed"],
        "cycle": d["cycle"], "order_id": d["order_id"], "sample_class": d["sample_class"],
        "n_candidates": len(producers), "original_mean_winner_overlap":
            _jaccard(original_winners, mean_winners),
        "original_mean_rank_correlation": _spearman(
            [d["original"][p] for p in producers], [means[p] for p in producers]),
        "top_two_expected_margin": _top_two_margin(means), "signal_to_noise": snr,
        "ensemble_regret": statistics.mean(ensemble_regrets),
        "actor_regret": statistics.mean(actor_regrets),
        "random_regret": statistics.mean(random_regrets),
        "perfect_future_information": statistics.mean(perfect_info),
    }
    return summary, candidate_rows


def _aggregate_scope(decisions: Sequence[tuple[int, dict]], ids: Sequence[int], *,
                     estimator_k: int, repeats: int = SPLIT_REPEATS) -> dict:
    raw = [d for _, d in decisions]
    conv = _convergence_metrics(raw, ids, size=estimator_k, repeats=repeats)
    summaries = [_decision_summary(digest, d, ids)[0] for digest, d in decisions]
    signal = [s for s in summaries if s["sample_class"] == "signal"]
    use = signal or summaries
    actor = statistics.mean(s["actor_regret"] for s in use)
    ensemble = statistics.mean(s["ensemble_regret"] for s in use)
    reduction = ((actor - ensemble) / actor) if actor > 0 else 0.0
    return {
        **conv, "n_decisions": len(use),
        "original_mean_winner_overlap": statistics.mean(
            s["original_mean_winner_overlap"] for s in use),
        "original_mean_rank_correlation": statistics.mean(
            s["original_mean_rank_correlation"] for s in use),
        "ensemble_regret": ensemble, "actor_regret": actor,
        "random_regret": statistics.mean(s["random_regret"] for s in use),
        "regret_reduction_vs_actor": reduction,
    }


def analyze_rows(rows: Sequence[Mapping[str, str]], *, repeats: int = SPLIT_REPEATS,
                 require_registered_scope: bool = False) -> dict:
    """Pure artifact read: compute every metric and the mechanical EXP-0040 classification."""
    cube, ids = _raw_cube(rows)
    if len(ids) not in (PRIMARY_CONTINUATIONS, ESCALATED_CONTINUATIONS):
        raise ValueError(f"analysis requires 16 or 32 continuations, got {len(ids)}")
    signal = [(digest, d) for digest, d in cube.items() if d["sample_class"] == "signal"]
    if not signal:
        raise ValueError("no signal decisions in raw rows")
    estimator_k = len(ids) // 2
    tasks = sorted({d["task_id"] for _, d in signal})
    if require_registered_scope:
        actual = {t: {d["seed"] for _, d in signal if d["task_id"] == t} for t in tasks}
        expected = {t: set(seeds) for t, seeds in REGISTERED_SEEDS.items()}
        if actual != expected:
            raise ValueError(f"analysis scope {actual} != registered task/seed scope {expected}")
    task_metrics: list[dict] = []
    seed_metrics: list[dict] = []
    for task in tasks:
        td = [(digest, d) for digest, d in signal if d["task_id"] == task]
        task_metrics.append({
            "task_id": task,
            **_aggregate_scope(td, ids, estimator_k=estimator_k, repeats=repeats)})
        for seed in sorted({d["seed"] for _, d in td}):
            sd = [(digest, d) for digest, d in td if d["seed"] == seed]
            seed_metrics.append({"task_id": task, "seed": seed,
                                 **_aggregate_scope(sd, ids, estimator_k=estimator_k,
                                                    repeats=repeats)})

    gate1 = all(m["converged"] for m in [*task_metrics, *seed_metrics])
    passing_tasks: list[str] = []
    for tm in task_metrics:
        per_seed = [m for m in seed_metrics if m["task_id"] == tm["task_id"]]
        if (tm["regret_reduction_vs_actor"] >= 0.10 and
                all(m["regret_reduction_vs_actor"] >= 0.10 for m in per_seed)):
            passing_tasks.append(tm["task_id"])
    remaining_safe = all(
        m["ensemble_regret"] <= 1.05 * m["actor_regret"]
        for m in task_metrics if m["task_id"] not in passing_tasks)
    gate2 = len(passing_tasks) >= 2 and remaining_safe

    convergence_by_k: dict[str, bool] = {}
    convergence_detail: dict[str, dict] = {}
    for k in CONTINUATION_COUNTS:
        if 2 * k <= len(ids):
            scopes = []
            for task in tasks:
                td = [d for _, d in signal if d["task_id"] == task]
                scopes.append(_convergence_metrics(td, ids, size=k, repeats=repeats))
                for seed in sorted({d["seed"] for d in td}):
                    scopes.append(_convergence_metrics(
                        [d for d in td if d["seed"] == seed], ids, size=k, repeats=repeats))
            convergence_by_k[str(k)] = all(s["converged"] for s in scopes)
            convergence_detail[str(k)] = {"pass": convergence_by_k[str(k)], "scopes": scopes}

    if not gate1 or not gate2:
        classification = "SEARCH_ONLY"
    else:
        mean_pattern = sum(m["original_mean_winner_overlap"] < 0.70 for m in task_metrics) >= 2
        coverage_pattern = sum(
            m["original_mean_winner_overlap"] >= 0.80 and
            m["original_mean_rank_correlation"] >= 0.70 for m in task_metrics) >= 2
        classification = ("MEAN_TARGET_LICENSED" if mean_pattern else
                          "COVERAGE_PATH_LICENSED" if coverage_pattern else "AMBIGUOUS")

    decision_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for digest, d in sorted(cube.items()):
        summary, candidates = _decision_summary(digest, d, ids)
        decision_rows.append(summary)
        candidate_rows.extend(candidates)
    return {
        "experiment": EXPERIMENT, "classification": classification,
        "continuation_count": len(ids), "estimator_k": estimator_k,
        "gate1_convergence": gate1, "gate2_usefulness": gate2,
        "passing_usefulness_tasks": passing_tasks,
        "convergence_by_k": convergence_by_k,
        "convergence_detail": convergence_detail,
        "task_metrics": task_metrics, "seed_metrics": seed_metrics,
        "decision_rows": decision_rows, "candidate_rows": candidate_rows,
    }


def _assert_sample_coverage(raw_rows: Sequence[Mapping[str, str]], sample_csv: str) -> None:
    """Raw rows must cover exactly the preselected decisions; a partial run cannot self-certify."""
    samples = _read_csv(sample_csv)
    expected = {int(r["decision_digest"]): (
        r["task_id"], int(r["seed"]), int(r["cycle"]), r["order_id"], r["sample_class"])
        for r in samples}
    if len(expected) != len(samples):
        raise ValueError("sample.csv contains duplicate decision digests")
    actual: dict[int, tuple] = {}
    for r in raw_rows:
        digest = int(r["decision_digest"])
        actual[digest] = (r["task_id"], int(r["seed"]), int(r["cycle"]),
                          r["order_id"], r["sample_class"])
    if actual != expected:
        raise ValueError(f"raw/sample decision coverage mismatch: raw={len(actual)}, "
                         f"sample={len(expected)}")


def analyze_file(raw_csv: str, *, out_dir: str, sample_csv: str | None = None,
                 inputs: Mapping[str, object] | None = None,
                 source_commit: str | None = None,
                 require_registered_scope: bool = False) -> dict:
    raw_rows = _read_csv(raw_csv)
    if sample_csv is not None:
        _assert_sample_coverage(raw_rows, sample_csv)
    result = analyze_rows(raw_rows, require_registered_scope=require_registered_scope)
    artifacts = os.path.join(out_dir, "artifacts")
    _write_csv(os.path.join(artifacts, "decision_stability.csv"), result.pop("decision_rows"))
    _write_csv(os.path.join(artifacts, "candidate_stability.csv"), result.pop("candidate_rows"))
    _write_csv(os.path.join(artifacts, "metrics_by_task.csv"), result["task_metrics"])
    _write_csv(os.path.join(artifacts, "metrics_by_seed.csv"), result["seed_metrics"])
    _write_csv(os.path.join(out_dir, "summary.csv"), result["task_metrics"])
    manifest = {
        **result, "inputs": inputs, "raw_labels": {
            "path": os.path.abspath(raw_csv), "sha256": sha256_file(raw_csv)},
        "sample": ({"path": os.path.abspath(sample_csv), "sha256": sha256_file(sample_csv)}
                   if sample_csv is not None else None),
        "source": source_fingerprint(source_commit), "env": environment_manifest(),
    }
    _write_json(os.path.join(artifacts, "selection_manifest.json"), manifest)
    return manifest


def run_protocol(
    records: Sequence["TaskRecord"], *, prod: "Policy", authority: Authority,
    out_dir: str, delta: float = 1.0, lookahead: int = 8, horizon: int = 100,
    signal_per_stratum: int = SIGNAL_PER_STRATUM, flat_per_stratum: int = FLAT_PER_STRATUM,
    max_decisions: int | None = None, source_commit: str | None = None,
    require_registered_scope: bool = True,
    log: "Callable[..., None]" = print,
) -> dict:
    """Registered two-tranche protocol: K=16, then append IDs 16..31 only if Gate 1 fails."""
    selected, sampling = select_sample(
        authority, signal_per_stratum=signal_per_stratum,
        flat_per_stratum=flat_per_stratum, max_decisions=max_decisions)
    if not selected:
        raise ValueError("authority selection produced no decisions")
    os.makedirs(out_dir, exist_ok=True)
    artifacts = os.path.join(out_dir, "artifacts")
    os.makedirs(artifacts, exist_ok=True)
    run_log = os.path.join(artifacts, "run.log")
    Path(run_log).write_text("")

    def emit(*parts: object, **kwargs: object) -> None:
        message = " ".join(str(p) for p in parts)
        log(message, **kwargs)
        with open(run_log, "a") as f:
            f.write(message + "\n")

    sample_csv = os.path.join(out_dir, "sample.csv")
    _write_csv(sample_csv, sample_rows(selected))
    emit(f"[exp0040] selected {len(selected)} decisions: "
         f"{sampling['n_signal']} signal, {sampling['n_flat']} flat")
    raw = os.path.join(artifacts, "continuation_labels.csv")
    collect_labels(records, prod=prod, selected=selected,
                   continuation_ids=range(PRIMARY_CONTINUATIONS), out_csv=raw,
                   delta=delta, lookahead=lookahead, horizon=horizon, log=emit)
    result = analyze_file(raw, out_dir=out_dir, sample_csv=sample_csv, inputs=authority.inputs,
                          source_commit=source_commit,
                          require_registered_scope=require_registered_scope)
    if not result["gate1_convergence"]:
        emit("[exp0040] Gate 1 failed at 16; appending the single registered escalation to 32")
        collect_labels(records, prod=prod, selected=selected,
                       continuation_ids=range(PRIMARY_CONTINUATIONS, ESCALATED_CONTINUATIONS),
                       out_csv=raw, delta=delta, lookahead=lookahead, horizon=horizon,
                       append=True, log=emit)
        result = analyze_file(raw, out_dir=out_dir, sample_csv=sample_csv,
                              inputs=authority.inputs,
                              source_commit=source_commit,
                              require_registered_scope=require_registered_scope)
    spec = {
        "experiment": EXPERIMENT, "version": "rollout-target-stability-v1",
        "status": "COMPLETE", "delta": delta, "lookahead": lookahead, "horizon": horizon,
        "floor": "blend_0.5", "sampling": sampling,
        "schema": "graph_primitives_v1", "schema_hash": schema_hash(),
        "continuation_count": result["continuation_count"],
        "classification": result["classification"],
        "convergence_by_k": result["convergence_by_k"], "inputs": authority.inputs,
        "source": source_fingerprint(source_commit), "env": environment_manifest(),
    }
    _write_json(os.path.join(out_dir, "spec.json"), spec)
    _write_json(os.path.join(artifacts, "spec.json"), spec)
    emit(f"[exp0040] classification={result['classification']} K="
         f"{result['continuation_count']}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--task-ids", default=",".join(TASK_IDS))
    ap.add_argument("--prod-ckpt", required=True)
    ap.add_argument("--exp0039-decisions", required=True)
    ap.add_argument("--exp0039-candidates", required=True)
    ap.add_argument("--actor-decisions", required=True)
    ap.add_argument("--actor-producers", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--signal-per-stratum", type=int, default=SIGNAL_PER_STRATUM)
    ap.add_argument("--flat-per-stratum", type=int, default=FLAT_PER_STRATUM)
    ap.add_argument("--max-decisions", type=int)
    ap.add_argument("--source-commit")
    ap.add_argument("--no-require-registered", action="store_true",
                    help="smoke only: permit non-registered inputs/tasks")
    args = ap.parse_args()

    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from .curriculum_replay import _load_archive

    require = not args.no_require_registered
    if require and (args.delta, args.lookahead, args.horizon) != (1.0, 8, 100):
        raise ValueError("registered run requires delta=1, lookahead=8, horizon=100")
    if require and args.max_decisions is not None:
        raise ValueError("--max-decisions is smoke-only")
    if require and (args.source_commit is None or len(args.source_commit) != 40 or
                    any(c not in "0123456789abcdef" for c in args.source_commit.lower())):
        raise ValueError("registered run requires a full launcher-verified --source-commit")
    authority = load_authority(
        exp0039_decisions=args.exp0039_decisions,
        exp0039_candidates=args.exp0039_candidates,
        actor_decisions=args.actor_decisions, actor_producers=args.actor_producers,
        archive=args.archive, producer_ckpt=args.prod_ckpt, require_registered=require)
    by_id = {r.task_id: r for r in _load_archive(args.archive)}
    task_ids = args.task_ids.split(",")
    if require and tuple(task_ids) != TASK_IDS:
        raise ValueError(f"registered task order must be {TASK_IDS}")
    records = [by_id[t] for t in task_ids]
    prod = scorer_producing_policy(load_producer_score_model(args.prod_ckpt))
    run_protocol(
        records, prod=prod, authority=authority, out_dir=args.out_dir,
        delta=args.delta, lookahead=args.lookahead, horizon=args.horizon,
        signal_per_stratum=args.signal_per_stratum,
        flat_per_stratum=args.flat_per_stratum, max_decisions=args.max_decisions,
        source_commit=args.source_commit, require_registered_scope=require)


if __name__ == "__main__":
    main()
