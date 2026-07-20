"""EXP-0037 value-routed orderer: residual router over a fixed blend_0.5 base.

EXP-0036 closed the *blend-selection* orderer: a learned switch over
even_split/blend/proportional could not beat fixed blend_0.5. Those are all
capacity-shaped — they only tune how sharply routing follows current throughput.
This module tests the genuinely different action space (HANDOVER): a learned
per-(stream, producer) score that *bends* routing away from throughput,

    final_logits = blend_0.5_logits + clip(score(env, agent), -r, +r)
    route        = softmax(final_logits) over each order's eligible producers

so a lower-capacity producer can win weight when inventory / inbound / TTL / recipe
position / scarcity make it strategically better. blend_0.5 stays the zero-cost
floor: a zero score is byte-identical to it, and the clip bounds how far a
mistrained score can move away.

Measurement, not a trainer — the torch model + fit live in `train.value_router`.
"""

from __future__ import annotations

import copy
import statistics
from typing import Callable, Sequence

import numpy as np

from ..env import TokenomicsAECEnv
from .baselines import Policy, ordering_blend
from .curriculum import TaskRecord, _episode_seeds, replay
from .joint_stack import joint_stack_env
from .orderer_oracle import _rollout_units
from .runner import run_episode

# A per-(stream, producer) residual score in the same flat k_stream*P layout as an
# ordering action (see decode.decode_ordering). Added to the base logits, then clipped.
ScoreFn = Callable[[TokenomicsAECEnv, str], np.ndarray]


def residual_router(base: Policy, score_fn: ScoreFn, clip: float) -> Policy:
    """A routing policy = `base` logits + a clipped learned residual. A zero `score_fn`
    reduces exactly to `base` (the blend_0.5 floor); `clip` bounds how far the residual
    can override the base per producer, so a mistrained score degrades gracefully."""

    def policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        base_logits = base(env, agent)
        residual = np.clip(score_fn(env, agent), -clip, clip)
        return (base_logits + residual).astype(np.float32)

    return policy


def _one_cycle_bump(
    base: Policy, target_agent: str, k: int, col: int, delta: float, at_cycle: int
) -> Policy:
    """`base` routing, but with `delta` added to stream `k` / producer column `col` **only**
    for `target_agent` at cycle `at_cycle` (base everywhere else). Gating on the agent is
    essential: many product-stream ordering agents act each cycle sharing the same stream
    indices/producer columns, so a cycle-only bump would leak into every other agent and
    confound the label — this isolates the single labeled decision."""

    def pol(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        logits = np.asarray(base(env, agent), dtype=np.float32).copy()
        if agent == target_agent and env.current_cycle == at_cycle:
            mat = logits.reshape(env.spec_.k_stream, env.spec_.num_producers)
            mat[k, col] += delta
            logits = mat.flatten()
        return logits

    return pol


def marginal_value_labels(
    env: TokenomicsAECEnv,
    prod: Policy,
    base: Policy,
    agent: str,
    k: int,
    eligible_cols: Sequence[int],
    delta: float,
    lookahead: int,
) -> np.ndarray:
    """The teacher label for one ordering decision by `agent` (HANDOVER: "marginal value of
    increasing a producer's routing logit around blend_0.5"). At the CURRENT cycle, for each
    eligible producer column, nudge `agent`'s stream-`k` logit by `+delta`, follow `base` after
    (for `agent` AND every other ordering agent), and roll `lookahead` cycles under fixed `prod`;
    the label is consumer units gained vs the unperturbed `base` rollout. Only `agent`'s decision
    is perturbed, so the label is not confounded by other agents. Ineligible columns stay 0.
    Deterministic (domain is deterministic; `env` is cloned per candidate, never mutated).
    Requires `env` to be at `agent`'s turn (its next action is the labeled decision)."""
    at_cycle = env.current_cycle
    base_units = _rollout_units(copy.deepcopy(env), prod, base, lookahead)
    labels = np.zeros(env.spec_.num_producers, dtype=np.float32)
    for col in eligible_cols:
        bumped = _one_cycle_bump(base, agent, k, int(col), delta, at_cycle)
        units = _rollout_units(copy.deepcopy(env), prod, bumped, lookahead)
        labels[col] = float(units - base_units)
    return labels


def _per_seed_cu_dead(
    cfg, prod: Policy, order: Policy, seeds: Sequence[int], horizon: int
) -> list[tuple[int, float, float]]:
    """Per-seed `(seed, consumer_completed_units, dead)` for `(prod, order)` over anchored seeds,
    on the SAME env contract EXP-0036 measured its blend floor with (`joint_stack_env`: entity
    producer obs + corrected v4 product-stream ordering) so `blend_cu` is comparable. Raw rows,
    not yet aggregated — the caller means them for the gate verdict but the per-seed detail is
    persisted so a reviewer can check whether one anchored seed drove the mean / a death count."""
    out: list[tuple[int, float, float]] = []
    for s in seeds:
        stats = run_episode(joint_stack_env(cfg, horizon), prod, order, seed=s)
        out.append((s, stats["consumer_completed_units"], 1.0 if stats["dead"] else 0.0))
    return out


def router_gate(
    records: Sequence[TaskRecord],
    prod: Policy,
    score_fn: ScoreFn,
    *,
    clip: float,
    eval_seeds: int,
    horizon: int,
    base: "Policy | None" = None,
) -> tuple[list[dict], list[dict]]:
    """The DoD instrument (HANDOVER): deploy the residual router by ROLLOUT and compare it to the
    fixed `blend_0.5` floor on each task, at the archive-anchored episode seeds. Rolls each policy
    out once per (task, seed) and returns `(aggregate_rows, per_seed_rows)`:

    - aggregate_rows: per task `learned_cu` vs `blend_cu`, `delta_vs_blend`, both death rates,
      `beats_floor` — the gate verdict, judged on deployed units (never table error).
    - per_seed_rows: one `{task_id, seed, policy, cu, dead}` per (task, seed, policy) so the
      aggregate's mean / death counts are auditable and reproducible from raw rows.

    The residual base defaults to `blend_0.5` (the floor), so a zero `score_fn` scores exactly at
    the floor (delta 0, `beats_floor` False)."""
    base = base if base is not None else ordering_blend(0.5)
    learned = residual_router(base, score_fn, clip)
    agg: list[dict] = []
    per_seed: list[dict] = []
    for rec in records:
        cfg = replay(rec)
        seeds = _episode_seeds(rec.episode_seed, eval_seeds)
        detail = {"learned": _per_seed_cu_dead(cfg, prod, learned, seeds, horizon),
                  "blend_0.5": _per_seed_cu_dead(cfg, prod, base, seeds, horizon)}
        for policy, rows_ in detail.items():
            for s, cu, dead in rows_:
                per_seed.append({"task_id": rec.task_id, "seed": s, "policy": policy,
                                 "cu": round(cu, 2), "dead": round(dead, 3)})
        l_cu = statistics.mean(cu for _, cu, _ in detail["learned"])
        l_dead = statistics.mean(d for _, _, d in detail["learned"])
        b_cu = statistics.mean(cu for _, cu, _ in detail["blend_0.5"])
        b_dead = statistics.mean(d for _, _, d in detail["blend_0.5"])
        agg.append({
            "task_id": rec.task_id,
            "learned_cu": round(l_cu, 2), "blend_cu": round(b_cu, 2),
            "delta_vs_blend": round(l_cu - b_cu, 2),
            "learned_dead": round(l_dead, 3), "blend_dead": round(b_dead, 3),
            "beats_floor": l_cu > b_cu,
        })
    return agg, per_seed


def evaluate_router_gate(
    records: Sequence[TaskRecord],
    prod: Policy,
    score_fn: ScoreFn,
    *,
    clip: float,
    eval_seeds: int,
    horizon: int,
    base: "Policy | None" = None,
) -> list[dict]:
    """Aggregate-only view of `router_gate` (kept for callers that only need the per-task verdict)."""
    agg, _ = router_gate(records, prod, score_fn, clip=clip, eval_seeds=eval_seeds,
                         horizon=horizon, base=base)
    return agg
