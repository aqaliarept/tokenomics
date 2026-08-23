"""EXP-0041 bounded confidence-gated residual router + closed-loop Gate B.

A passing offline score licenses only a CONSERVATIVE deployment: the ensemble never routes by
`argmax`. Per decision it centers each model seed's candidate scores, requires the seeds to agree
on a winner and the top-two margin to clear a threshold, and only then adds a clipped, scaled
residual to the `blend_0.5` logits:

```text
r_m       = center(score from model seed m over eligible producers)
r_mean    = mean_m r_m
unanimous = the seeds' top winner sets intersect
confident = unanimous and top_two_margin(r_mean) >= margin_threshold
residual  = alpha * clip(r_mean, -1, 1) if confident else 0
final     = blend_0.5_logits + residual        # decoded by the existing masked allocator
```

When the seeds disagree or the margin is small the policy is byte-identical to `blend_0.5` for
that decision, and `alpha = 0` is byte-identical everywhere — so the floor is always reachable
and a mistrained score degrades gracefully instead of hard-switching a producer.

`(alpha, margin_threshold)` is selected on VALIDATION rollouts only and frozen before any
final-test episode runs. Gate B then compares the frozen policy to `blend_0.5` on paired test
episodes, including a per-episode floor, a death check, and per-decision route-integrity audits
(conservation, eligibility masking, and preserved multi-producer routing).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv
    from .baselines import Policy
    from .curriculum import TaskRecord

ALPHA_GRID: tuple[float, ...] = (0.10, 0.25, 0.50)
MARGIN_GRID: tuple[float, ...] = (0.0, 0.5, 1.0)
CLIP = 1.0
TIE_TOL = 1e-12


@dataclass(frozen=True)
class Adapter:
    """The frozen routing adapter. `clip` is fixed at 1.0 by the pre-registration, so `alpha` is
    the only bound on how far the residual can move a logit."""

    alpha: float
    margin_threshold: float
    clip: float = CLIP

    def label(self) -> str:
        return f"alpha{self.alpha:g}_margin{self.margin_threshold:g}"


@dataclass
class RouteAudit:
    """Per-decision integrity counters collected while a residual policy plays (Gate B
    condition 6). Every check compares the FINAL routing against the base routing at the SAME
    state, so the multi-producer property is a paired comparison, not a guess."""

    n_decisions: int = 0
    n_confident: int = 0
    nonfinite: int = 0
    conservation_violations: int = 0
    eligibility_violations: int = 0
    multi_producer_violations: int = 0

    def as_row(self) -> dict:
        return {"route_decisions": self.n_decisions, "route_confident": self.n_confident,
                "route_nonfinite": self.nonfinite,
                "route_conservation_violations": self.conservation_violations,
                "route_eligibility_violations": self.eligibility_violations,
                "route_multi_producer_violations": self.multi_producer_violations}

    def violations(self) -> int:
        return (self.nonfinite + self.conservation_violations + self.eligibility_violations
                + self.multi_producer_violations)


# --- the residual rule (pure) ------------------------------------------------------------
def ensemble_residual(scores: Sequence[np.ndarray], adapter: Adapter) -> tuple[np.ndarray, dict]:
    """The confidence-gated residual for ONE decision from the per-model-seed candidate scores
    (each `[k]`, candidate order). Returns `(residual, info)`.

    Centering per seed removes each model's arbitrary additive level, so only the within-decision
    contrast survives; unanimity is set INTERSECTION of the seeds' top (tie-inclusive) winners,
    which means a shared tie still counts as agreement; the margin is read off the ensemble mean.
    Not confident -> an exactly-zero residual, i.e. the blend action byte-for-byte."""
    k = len(scores[0]) if scores else 0
    zeros = np.zeros(k, dtype=np.float64)
    if not scores or k == 0:
        return zeros, {"unanimous": False, "margin": 0.0, "confident": False}
    raw = [np.asarray(s, dtype=np.float64) for s in scores]
    if not all(np.isfinite(r).all() for r in raw):  # checked BEFORE centering: inf - inf = nan
        return zeros, {"unanimous": False, "margin": float("nan"), "confident": False,
                       "nonfinite": True}
    centered = [r - float(r.mean()) for r in raw]
    winners = [set(np.flatnonzero(c >= c.max() - TIE_TOL).tolist()) for c in centered]
    unanimous = bool(set.intersection(*winners)) if winners else False
    r_mean = np.mean(np.stack(centered), axis=0)
    order = np.sort(r_mean)[::-1]
    margin = float(order[0] - order[1]) if k >= 2 else float("inf")
    confident = bool(unanimous and margin >= adapter.margin_threshold)
    residual = (adapter.alpha * np.clip(r_mean, -adapter.clip, adapter.clip) if confident
                else zeros)
    return residual.astype(np.float64), {"unanimous": unanimous, "margin": margin,
                                         "confident": confident}


# --- deploy-time scorer --------------------------------------------------------------------
class EnsembleRouter:
    """Scores a live decision with the frozen ensemble and emits the flat `k_stream x P`
    residual an ordering action expects. Ineligible producers and uncontested streams always get
    exactly 0, so eligibility masking and the blend floor are structural, not learned."""

    def __init__(self, arm: str, models: Sequence, normalizer=None, *, adapter: Adapter,
                 audit: "RouteAudit | None" = None) -> None:
        from ..train.graph_generalization_probe import ARMS

        spec = ARMS[arm]
        if not spec.get("pooled", True) or spec.get("diagnostic_only"):
            raise ValueError(f"arm {arm!r} is a diagnostic (per-task fit or control target) and is "
                             "not deployable")
        if not models:
            raise ValueError("the ensemble needs at least one model")
        self.arm = arm
        self.kind = spec["kind"]
        self.models = list(models)
        self.normalizer = normalizer
        self.adapter = adapter
        self.audit = audit

    def _candidate_scores(self, env: "TokenomicsAECEnv", aid: str, oid: str,
                          elig: Sequence[str], view) -> list[np.ndarray]:
        import torch

        if self.kind == "row":
            from ..train.value_router import producer_row_features

            rows = producer_row_features(env, oid, view)
            cols = [env.spec_.producer_ids.index(p) for p in elig]
            x = torch.as_tensor(rows[cols], dtype=torch.float32)
            with torch.no_grad():
                return [m(x).cpu().numpy().astype(np.float64) for m in self.models]
        from ..domain.graph_view import build_economy_graph_view
        from ..train.graph_router_probe import Batch
        from .graph_value_router_dump import encode_graph

        g = encode_graph(build_economy_graph_view(env.system, aid, oid, list(elig)))
        gs = self.normalizer.apply([g]) if self.normalizer is not None else [g]
        batch = Batch(gs)
        with torch.no_grad():
            return [m(batch).cpu().numpy().astype(np.float64) for m in self.models]

    def residual(self, env: "TokenomicsAECEnv", agent: str,
                 base_logits: "np.ndarray | None" = None) -> np.ndarray:
        sp = env.spec_
        _, aid = env._role[agent]
        view = env.system.ordering_agents[aid].view(env.system.producer_infos)
        mat = np.zeros((sp.k_stream, sp.num_producers), dtype=np.float64)
        base = (np.asarray(base_logits, dtype=np.float64).reshape(sp.k_stream, sp.num_producers)
                if base_logits is not None else None)
        for k, oid in enumerate(sorted(view.incoming)):
            if k >= sp.k_stream:
                break
            elig = list(view.eligible[oid])
            if len(elig) < 2:
                continue  # uncontested: nothing to prefer, stay exactly at the blend
            cols = [sp.producer_ids.index(p) for p in elig]
            res, info = ensemble_residual(self._candidate_scores(env, aid, oid, elig, view),
                                          self.adapter)
            for ci, col in enumerate(cols):
                mat[k, col] = res[ci]
            if self.audit is not None:
                self._record(env, view, oid, k, mat, base, info)
        return mat.flatten()

    def _record(self, env, view, oid, k, mat, base, info) -> None:
        """Audit this decision's decoded routing against the base routing at the same state."""
        from ..env.allocator import allocate_suborders

        a = self.audit
        assert a is not None
        a.n_decisions += 1
        a.n_confident += int(bool(info.get("confident")))
        if info.get("nonfinite") or not np.isfinite(mat[k]).all():
            a.nonfinite += 1
            return
        if base is None:
            return
        sp = env.spec_
        oi = view.incoming[oid]
        mask = np.array([pid in view.eligible[oid] for pid in sp.producer_ids])
        final = allocate_suborders(sp.producer_ids, base[k] + mat[k], mask, oi.quantity, oi.tokens)
        blend = allocate_suborders(sp.producer_ids, base[k], mask, oi.quantity, oi.tokens)
        if sum(s.quantity for s in final) != oi.quantity:
            a.conservation_violations += 1
        if any(s.producer_id not in view.eligible[oid] for s in final):
            a.eligibility_violations += 1
        if len([s for s in blend if s.quantity > 0]) > 1 and \
                len([s for s in final if s.quantity > 0]) < 2:
            a.multi_producer_violations += 1

    def score_fn(self) -> Callable[["TokenomicsAECEnv", str], np.ndarray]:
        def fn(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
            return self.residual(env, agent).astype(np.float32)

        return fn


def adapter_policy(base: "Policy", router: EnsembleRouter) -> "Policy":
    """`blend_0.5` logits plus the confidence-gated residual, decoded by the unchanged masked
    allocator. With `alpha = 0` — or on any decision the ensemble is not confident about — the
    returned logits are the base logits byte-for-byte."""
    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        base_logits = np.asarray(base(env, agent), dtype=np.float32)
        residual = router.residual(env, agent, base_logits=base_logits)
        return (base_logits + residual).astype(np.float32)

    return policy


# --- rollouts ------------------------------------------------------------------------------
@dataclass
class PolicyBuild:
    """A named policy plus the audit it fills (None for the plain blend floor)."""

    name: str
    make: Callable[[], tuple]  # () -> (Policy, RouteAudit | None)
    meta: dict = field(default_factory=dict)


def rollout_rows(records: "Sequence[TaskRecord]", builds: Sequence[PolicyBuild], *,
                 prod: "Policy", seeds_by_task: dict, horizon: int,
                 log=lambda *a, **k: None) -> list[dict]:
    """One row per (task, episode seed, policy): consumer completed units, death flag, and the
    route-integrity counters. Rows are raw — every aggregate and both gates are recomputed from
    them, so a reader never has to trust a summary."""
    from .curriculum import replay
    from .joint_stack import joint_stack_env
    from .runner import run_episode

    rows: list[dict] = []
    for rec in records:
        cfg = replay(rec)
        for seed in sorted(seeds_by_task.get(rec.task_id, ())):
            for build in builds:
                policy, audit = build.make()
                stats = run_episode(joint_stack_env(cfg, horizon), prod, policy, seed=int(seed))
                row = {"task_id": rec.task_id, "trajectory_seed": int(seed), "policy": build.name,
                       "cu": float(stats["consumer_completed_units"]),
                       "dead": 1 if stats["dead"] else 0, **build.meta}
                row.update((audit or RouteAudit()).as_row())
                rows.append(row)
                log(f"[rollout] {rec.task_id} seed {seed} {build.name}: cu {row['cu']:.1f} "
                    f"dead {row['dead']}")
    return rows


def _by_task(rows: Sequence[dict], policy: str) -> dict:
    out: dict[str, list[dict]] = {}
    for r in rows:
        if r["policy"] == policy:
            out.setdefault(r["task_id"], []).append(r)
    return out


def per_task_summary(rows: Sequence[dict], policy: str) -> dict:
    return {task: {"cu": float(np.mean([r["cu"] for r in rs])),
                   "dead": float(np.mean([r["dead"] for r in rs])),
                   "n_episodes": len(rs),
                   "violations": int(sum(r.get("route_conservation_violations", 0)
                                         + r.get("route_eligibility_violations", 0)
                                         + r.get("route_multi_producer_violations", 0)
                                         + r.get("route_nonfinite", 0) for r in rs))}
            for task, rs in _by_task(rows, policy).items()}


# --- validation-only adapter selection ---------------------------------------------------
def select_adapter(rows: Sequence[dict], *, blend_policy: str = "blend_0.5",
                   candidates: "Sequence[str] | None" = None,
                   meta: "dict[str, dict] | None" = None) -> dict:
    """Pick `(alpha, margin_threshold)` from VALIDATION rollout rows with the pre-registered
    rule: (1) reject a pair that loses more than 2% consumer units on ANY task or raises deaths;
    (2) among survivors take the highest mean validation consumer units; (3) break a 0.5% tie
    toward the smaller alpha, then the larger margin; (4) if no pair beats blend on at least 2 of
    3 tasks, stop before any final-test rollout."""
    blend = per_task_summary(rows, blend_policy)
    if not blend:
        raise ValueError(f"no {blend_policy} rows: the floor must be rolled out on the same seeds")
    names = list(candidates) if candidates is not None else sorted(
        {r["policy"] for r in rows if r["policy"] != blend_policy})
    row_meta = {r["policy"]: r for r in rows}
    evaluated: dict[str, dict] = {}
    for name in names:
        summary = per_task_summary(rows, name)
        if set(summary) != set(blend):
            continue
        safe = all(summary[t]["cu"] >= 0.98 * blend[t]["cu"]
                   and summary[t]["dead"] <= blend[t]["dead"] for t in blend)
        wins = sum(1 for t in blend if summary[t]["cu"] > blend[t]["cu"])
        knobs = (meta or {}).get(name) or row_meta.get(name, {})
        evaluated[name] = {
            "mean_cu": float(np.mean([summary[t]["cu"] for t in summary])),
            "safe": bool(safe), "wins": wins,
            "alpha": float(knobs.get("alpha", float("inf"))),
            "margin_threshold": float(knobs.get("margin_threshold", float("-inf"))),
            "by_task": {t: {"cu": summary[t]["cu"], "blend_cu": blend[t]["cu"],
                            "dead": summary[t]["dead"], "blend_dead": blend[t]["dead"],
                            "rel": (summary[t]["cu"] - blend[t]["cu"]) / blend[t]["cu"]
                            if blend[t]["cu"] else float("nan")} for t in summary}}
    survivors = {n: v for n, v in evaluated.items() if v["safe"] and v["wins"] >= 2}
    if not survivors:
        return {"selected": None, "reason": "no (alpha, margin) pair is safe on every task and "
                                            "beats blend on >=2 tasks", "evaluated": evaluated}
    best = max(v["mean_cu"] for v in survivors.values())
    close = [n for n, v in survivors.items() if v["mean_cu"] >= best * 0.995]
    chosen = min(close, key=lambda n: (survivors[n]["alpha"], -survivors[n]["margin_threshold"], n))
    return {"selected": chosen, "reason": f"highest mean validation cu {best:.2f}"
            + (f"; 0.5% tie among {sorted(close)} resolved to the smaller alpha, then the "
               "larger margin" if len(close) > 1 else ""),
            "evaluated": evaluated}


# --- closed-loop Gate B --------------------------------------------------------------------
def _pairing(rows: Sequence[dict], *, policy: str, blend_policy: str,
             expected_tasks: "Sequence[str] | None",
             expected_seeds_by_task: "dict | None") -> dict:
    """Is this the WHOLE registered final-test rollout, played episode-for-episode against blend?

    Checking only that the two policies name the same tasks (and that there are three of them)
    let a run pass on a subset of the registered episodes, on tasks that are not the registered
    ones, or with a duplicated episode standing in for a missing one (review finding 2). Coverage
    now means: exactly the registered task ids; exactly the registered episode seeds per task; one
    row per (policy, task, seed) with no duplicates; and a blend partner for every learned
    episode."""
    seen: dict[tuple[str, str, int], int] = {}
    for r in rows:
        if r["policy"] not in (policy, blend_policy):
            continue
        key = (r["policy"], r["task_id"], int(r["trajectory_seed"]))
        seen[key] = seen.get(key, 0) + 1
    duplicates = sum(n - 1 for n in seen.values() if n > 1)
    tasks_seen = sorted({k[1] for k in seen})
    want_tasks = sorted(expected_tasks) if expected_tasks else tasks_seen
    seeds = {t: {k[2] for k in seen if k[1] == t and k[0] == policy} for t in tasks_seen}
    blend_seeds = {t: {k[2] for k in seen if k[1] == t and k[0] == blend_policy}
                   for t in tasks_seen}
    seeds_ok = (all(seeds.get(t, set()) == set(expected_seeds_by_task[t]) for t in want_tasks)
                if expected_seeds_by_task else all(seeds.get(t) for t in want_tasks))
    return {"ok": bool(tasks_seen == want_tasks and len(want_tasks) == 3 and seeds_ok
                       and not duplicates and seeds == blend_seeds),
            "tasks": tasks_seen, "episode_seeds": {t: sorted(s) for t, s in seeds.items()},
            "paired_with_blend": seeds == blend_seeds, "duplicate_rows": duplicates,
            "expected_episode_seeds": ({t: sorted(v) for t, v in expected_seeds_by_task.items()}
                                       if expected_seeds_by_task else None)}


def registered_test_scope() -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    """The pre-registered Gate B scope: the three tasks and their final-test episodes."""
    from .graph_generalization_dump import SPLIT_REGISTRY, TASK_IDS

    return TASK_IDS, {t: tuple(SPLIT_REGISTRY[t]["test"]) for t in TASK_IDS}


def gate_b(rows: Sequence[dict], *, policy: str, blend_policy: str = "blend_0.5",
           expected_tasks: "Sequence[str] | None" = None,
           expected_seeds_by_task: "dict | None" = None,
           require_registered: bool = True) -> dict:
    """Compute Gate B from paired final-test rollout ROWS alone.

    (1) mean consumer units beat blend on >= 2/3 tasks; (2) the average relative improvement
    across tasks is >= 5%; (3) no task is worse than blend by more than 5%; (4) no individual
    episode is worse than its PAIRED blend episode by more than 10%; (5) deaths never exceed
    blend on a task; (6) every routed decision conserved quantity, masked ineligible producers,
    kept multi-producer mass wherever blend did, stayed finite — and the rollout covers exactly
    the registered tasks and episode seeds, paired one-for-one with blend.

    `require_registered` (default True) supplies that registered scope. Leaving the expectations
    to the caller meant an omitted map was filled in from the rows themselves, so a two-episode
    rollout certified itself as the whole final test (review finding 1). A smoke passes
    `require_registered=False`; nothing that produces a verdict may."""
    if require_registered:
        reg_tasks, reg_seeds = registered_test_scope()
        expected_tasks = list(reg_tasks) if expected_tasks is None else expected_tasks
        expected_seeds_by_task = (reg_seeds if expected_seeds_by_task is None
                                  else expected_seeds_by_task)
        got = {t: sorted(int(s) for s in (expected_seeds_by_task or {}).get(t, ()))
               for t in reg_seeds}
        want = {t: sorted(int(s) for s in reg_seeds[t]) for t in reg_seeds}
        deviation = (f"gate scope tasks {sorted(expected_tasks)} != registered {sorted(reg_tasks)}"
                     if sorted(expected_tasks) != sorted(reg_tasks) else
                     f"gate scope episodes {got} != registered {want}" if got != want else None)
        if deviation:
            return {"pass": False, "conditions": {"registered_scope": False}, "reason": deviation,
                    "coverage": {"ok": False, "reason": deviation}}
    learned = per_task_summary(rows, policy)
    blend = per_task_summary(rows, blend_policy)
    tasks = sorted(set(learned) & set(blend))
    if not tasks or set(learned) != set(blend):
        return {"pass": False, "conditions": {"coverage": False},
                "reason": "learned and blend rows do not cover the same tasks"}
    pairing = _pairing(rows, policy=policy, blend_policy=blend_policy,
                       expected_tasks=expected_tasks,
                       expected_seeds_by_task=expected_seeds_by_task)
    rel = {t: ((learned[t]["cu"] - blend[t]["cu"]) / blend[t]["cu"]) if blend[t]["cu"] else
           float("nan") for t in tasks}
    paired = {}
    for t in tasks:
        b = {r["trajectory_seed"]: r["cu"] for r in rows
             if r["policy"] == blend_policy and r["task_id"] == t}
        for r in rows:
            if r["policy"] == policy and r["task_id"] == t and r["trajectory_seed"] in b:
                base_cu = b[r["trajectory_seed"]]
                paired[(t, r["trajectory_seed"])] = ((r["cu"] - base_cu) / base_cu
                                                     if base_cu else float("nan"))
    c1 = sum(1 for t in tasks if learned[t]["cu"] > blend[t]["cu"]) >= 2
    c2 = float(np.nanmean([rel[t] for t in tasks])) >= 0.05
    c3 = all(rel[t] >= -0.05 for t in tasks)
    c4 = bool(paired) and all(v >= -0.10 for v in paired.values())
    c5 = all(learned[t]["dead"] <= blend[t]["dead"] for t in tasks)
    c6 = all(learned[t]["violations"] == 0 for t in tasks) and bool(pairing["ok"])
    conditions = {"beats_blend_2of3": bool(c1), "avg_rel_improvement_5pct": bool(c2),
                  "no_task_worse_5pct": bool(c3), "no_episode_worse_10pct": bool(c4),
                  "no_extra_deaths": bool(c5), "route_integrity_and_three_tasks": bool(c6)}
    return {"pass": all(conditions.values()), "conditions": conditions, "coverage": pairing,
            "avg_rel_improvement": float(np.nanmean([rel[t] for t in tasks])),
            "per_task": {t: {"cu": learned[t]["cu"], "blend_cu": blend[t]["cu"], "rel": rel[t],
                             "dead": learned[t]["dead"], "blend_dead": blend[t]["dead"],
                             "violations": learned[t]["violations"]} for t in tasks},
            "worst_paired_episode": (min(paired.values()) if paired else float("nan"))}


# --- stages ------------------------------------------------------------------------------
def assert_rollout_shape(fit_spec: dict, *, records: "Sequence[TaskRecord]", seeds_by_task: dict,
                         horizon: int, model_seeds: Sequence[int], split: str,
                         input_manifest: "dict | None") -> None:
    """Refuse a rollout that is not the pre-registered one.

    The rollout stages used to accept any archive, producer, task records, seeds, and horizon
    (review finding 3), which meant the closed-loop number could come from a different economy or
    a different producer than the labels did. This pins all of it to the dump the fit was
    certified against: exactly the registered tasks, exactly that split's registered episode
    seeds, the dump's horizon, the REGISTERED model seeds (not merely whatever the fit recorded),
    a fit that itself satisfies the pre-registered fit contract, the same code and runtime, and
    the same archive + producer checkpoint by SHA-256."""
    from ..train.graph_generalization_probe import MODEL_SEEDS, certify_fit_spec
    from ..train.graph_router_probe import assert_env_matches, assert_source_matches

    dump_spec = fit_spec.get("dump_spec") or {}
    registry = (dump_spec.get("split_manifest") or {}).get("registry") or {}
    if not registry:
        raise ValueError("fit spec carries no dump split registry; it cannot license a rollout")
    certify_fit_spec(fit_spec)
    if sorted(int(s) for s in model_seeds) != sorted(MODEL_SEEDS):
        raise ValueError(f"rollout model seeds {sorted(model_seeds)} != the pre-registered "
                         f"{list(MODEL_SEEDS)}")
    # the rollout must be played by the code and runtime that produced the labels and the fit
    assert_source_matches(dump_spec.get("source"),
                          expected_commit=(fit_spec.get("source") or {}).get("commit"))
    assert_env_matches(dump_spec.get("env"))
    want_tasks = sorted(registry)
    got_tasks = sorted(r.task_id for r in records)
    if got_tasks != want_tasks:
        raise ValueError(f"rollout tasks {got_tasks} != the dump's registered {want_tasks}")
    for task in want_tasks:
        got = sorted(int(s) for s in seeds_by_task.get(task, ()))
        want = sorted(int(s) for s in registry[task][split])
        if got != want:
            raise ValueError(f"{task}: rollout {split} episodes {got} != registered {want}")
    if int(horizon) != int(dump_spec.get("horizon", -1)):
        raise ValueError(f"rollout horizon {horizon} != the dump's {dump_spec.get('horizon')}")
    if sorted(int(s) for s in model_seeds) != sorted(int(s) for s in fit_spec.get("model_seeds", ())):
        raise ValueError(f"rollout model seeds {sorted(model_seeds)} != the fit's "
                         f"{sorted(fit_spec.get('model_seeds', ()))}")
    want_inputs = dump_spec.get("inputs") or {}
    if input_manifest is None:
        raise ValueError("a pre-registered rollout must fingerprint its archive + producer "
                         "checkpoint so they can be compared to the dump's inputs")
    for role, want in want_inputs.items():
        got = (input_manifest.get(role) or {}).get("sha256") if want else None
        if (want or {}).get("sha256") != got:
            raise ValueError(f"rollout input '{role}' sha256 {got} != the dump's "
                             f"{(want or {}).get('sha256')}: the closed-loop run must be played "
                             "with the exact archive and producer that produced the labels")


def rollout_spec(stage: str, *, fit_spec: dict, seeds_by_task: dict, horizon: int,
                 model_seeds: Sequence[int], adapter: dict, input_manifest: "dict | None",
                 source_commit: "str | None", extra: "dict | None" = None) -> dict:
    """The rollout stage's own provenance record (required by the handover's artifact layout)."""
    from .graph_value_router_dump import environment_manifest, source_fingerprint

    return {"experiment": "EXP-0041", "stage": stage,
            "seeds_by_task": {t: list(v) for t, v in seeds_by_task.items()}, "horizon": horizon,
            "model_seeds": list(model_seeds), "adapter": adapter, "inputs": input_manifest,
            "alpha_grid": list(ALPHA_GRID), "margin_grid": list(MARGIN_GRID), "clip": CLIP,
            "dump_spec": fit_spec.get("dump_spec"), "fit_selection": fit_spec.get("selection"),
            "source": source_fingerprint(source_commit), "env": environment_manifest(),
            **(extra or {})}


def build_routers(arm: str, fit_dir: str, *, model_seeds: Sequence[int], tasks: Sequence[str],
                  adapter: Adapter, device: "str | None" = None
                  ) -> tuple[EnsembleRouter, RouteAudit]:
    """Load the frozen ensemble for one adapter setting (fresh audit per build)."""
    from ..train.graph_generalization_probe import load_arm

    arms = [load_arm(arm, fit_dir, model_seed=ms, tasks=tasks, device=device) for ms in model_seeds]
    audit = RouteAudit()
    models = [a.models[None] for a in arms]
    return EnsembleRouter(arm, models, arms[0].normalizer, adapter=adapter, audit=audit), audit


def run_validation_selection(records: "Sequence[TaskRecord]", fit_dir: str, out_dir: str, *,
                             prod: "Policy", arm: "str | None" = None,
                             baseline_arm: str = "row_mlp_h32",
                             model_seeds: Sequence[int] = (0, 1, 2),
                             seeds_by_task: "dict | None" = None, horizon: int = 100,
                             alphas: Sequence[float] = ALPHA_GRID,
                             margins: Sequence[float] = MARGIN_GRID,
                             base: "Policy | None" = None, device: "str | None" = None,
                             require_shape: bool = True, input_manifest: "dict | None" = None,
                             source_commit: "str | None" = None, log=print) -> dict:
    """Stage 3+4: run the `(alpha, margin)` grid on VALIDATION episodes, select and freeze the
    adapter, and lock the model/normalizer/arm-selection/adapter set by content hash. No
    final-test episode is touched here — Gate A and Gate B both come later, against this frozen
    manifest."""
    from ..train.graph_generalization_probe import lock_manifest, locked_arm
    from .baselines import ordering_blend

    fit_spec = json.load(open(os.path.join(fit_dir, "spec.json")))
    selected = locked_arm(fit_dir, {})
    if arm is not None and selected is not None and arm != selected:
        raise ValueError(f"arm {arm!r} != the fit's selection {selected!r}")
    arm = selected or arm
    if arm is None:
        raise ValueError("validation selected no graph arm; there is nothing to deploy")
    base = base if base is not None else ordering_blend(0.5)
    tasks = [r.task_id for r in records]
    registry = fit_spec.get("dump_spec", {}).get("split_manifest", {}).get("registry", {})
    seeds_by_task = seeds_by_task or {t: tuple(registry[t]["validation"]) for t in registry}
    if require_shape:
        assert_rollout_shape(fit_spec, records=records, seeds_by_task=seeds_by_task,
                             horizon=horizon, model_seeds=model_seeds, split="validation",
                             input_manifest=input_manifest)

    builds = [PolicyBuild("blend_0.5", lambda: (base, None))]
    for name, a in (("gnn", arm), ("row", baseline_arm)):
        for alpha in alphas:
            for margin in margins:
                adapter = Adapter(alpha=float(alpha), margin_threshold=float(margin))

                def make(a=a, adapter=adapter):
                    router, audit = build_routers(a, fit_dir, model_seeds=model_seeds, tasks=tasks,
                                                  adapter=adapter, device=device)
                    return adapter_policy(base, router), audit

                builds.append(PolicyBuild(f"{name}_{adapter.label()}", make,
                                          meta={"arm": a, "alpha": adapter.alpha,
                                                "margin_threshold": adapter.margin_threshold}))
    rows = rollout_rows(records, builds, prod=prod, seeds_by_task=seeds_by_task, horizon=horizon,
                        log=log)
    knobs = {b.name: b.meta for b in builds if b.meta}
    gnn_choice = select_adapter(rows, meta=knobs,
                                candidates=[b.name for b in builds if b.name.startswith("gnn_")])
    row_choice = select_adapter(rows, meta=knobs,
                                candidates=[b.name for b in builds if b.name.startswith("row_")])

    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "validation_grid.csv"), rows)
    adapter_doc = {
        "arm": arm, "baseline_arm": baseline_arm, "model_seeds": list(model_seeds),
        "clip": CLIP, "alpha_grid": list(alphas), "margin_grid": list(margins),
        "validation_seeds": {t: list(v) for t, v in seeds_by_task.items()},
        "gnn": gnn_choice, "row": row_choice,
        "selected": gnn_choice["selected"],
        "selected_alpha": (gnn_choice["evaluated"][gnn_choice["selected"]]["alpha"]
                           if gnn_choice["selected"] else None),
        "selected_margin": (gnn_choice["evaluated"][gnn_choice["selected"]]["margin_threshold"]
                            if gnn_choice["selected"] else None),
    }
    adapter_path = os.path.join(out_dir, "adapter.json")
    with open(adapter_path, "w") as f:
        json.dump(adapter_doc, f, indent=2)
    import glob

    # adapter.json is locked as a REAL manifest entry (its own root), not as a free-form field the
    # verifier ignores — otherwise the selected (alpha, margin) stays editable after the freeze.
    # fit/spec.json is locked too: Gate B reads the registry seeds, input hashes, model seeds,
    # source and require_shape flag OUT OF IT, so an unlocked copy is an editable rollout contract
    # (review finding 3).
    manifest = lock_manifest([*glob.glob(os.path.join(fit_dir, "checkpoints", "*.pt")),
                              os.path.join(fit_dir, "normalization.json"),
                              os.path.join(fit_dir, "spec.json"),
                              os.path.join(fit_dir, "architecture_selection.json")],
                             os.path.join(out_dir, "manifest.json"), base_dir=fit_dir,
                             roots={"adapter": (out_dir, [adapter_path])},
                             extra={"arm": arm})
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(rollout_spec("select", fit_spec=fit_spec, seeds_by_task=seeds_by_task,
                               horizon=horizon, model_seeds=model_seeds, adapter=adapter_doc,
                               input_manifest=input_manifest, source_commit=source_commit,
                               extra={"require_shape": bool(require_shape),
                                      "n_validation_rows": len(rows)}),
                  f, indent=2, default=str)
    log(f"[adapter] selected {adapter_doc['selected']} ({gnn_choice['reason']})")
    return {"adapter": adapter_doc, "manifest": manifest, "rows": rows}


def run_test_rollout(records: "Sequence[TaskRecord]", fit_dir: str, adapter_dir: str,
                     out_dir: str, *, prod: "Policy", gate_a_path: str,
                     model_seeds: Sequence[int] = (0, 1, 2),
                     seeds_by_task: "dict | None" = None, horizon: int = 100,
                     base: "Policy | None" = None, device: "str | None" = None,
                     require_shape: bool = True, input_manifest: "dict | None" = None,
                     source_commit: "str | None" = None, log=print) -> dict:
    """Stage 6: execute the ALREADY-LOCKED adapter on the final-test episodes, once.

    Gate A is RECOMPUTED here from the prediction rows its verdict names, and those rows are
    checked against the SHA-256 the Gate A stage recorded — a `gate_a.json` that merely contains
    `pass: true` proves nothing on its own (review finding 1). The manifest (checkpoints,
    normalization, arm selection, adapter) must still verify, and the arm named by Gate A, by the
    manifest, and by the adapter document must be the same arm."""
    from ..train.graph_generalization_probe import gate_a as recompute_gate_a
    from ..train.graph_generalization_probe import verify_manifest
    from .baselines import ordering_blend

    manifest_path = os.path.join(adapter_dir, "manifest.json")
    gate = json.load(open(gate_a_path))
    verdict = _verified_gate_a(gate, gate_a_path, manifest_path=manifest_path,
                               recompute=recompute_gate_a, require_binding=require_shape)
    if not verdict.get("pass"):
        raise ValueError("Offline Gate A did not pass; Gate B must not be run (HANDOVER stop rule)")
    manifest = verify_manifest(manifest_path, base_dir=fit_dir, roots={"adapter": adapter_dir})
    doc = json.load(open(os.path.join(adapter_dir, "adapter.json")))
    if doc["selected"] is None:
        raise ValueError("no adapter cleared the validation safety rule; no test rollout is allowed")
    for name, value in (("manifest", manifest.get("arm")), ("gate_a", gate.get("arm"))):
        if value is not None and value != doc["arm"]:
            raise ValueError(f"{name} arm {value!r} != the locked adapter's {doc['arm']!r}")
    base = base if base is not None else ordering_blend(0.5)
    tasks = [r.task_id for r in records]
    fit_spec = json.load(open(os.path.join(fit_dir, "spec.json")))
    registry = fit_spec.get("dump_spec", {}).get("split_manifest", {}).get("registry", {})
    seeds_by_task = seeds_by_task or {t: tuple(registry[t]["test"]) for t in registry}
    if require_shape:
        assert_rollout_shape(fit_spec, records=records, seeds_by_task=seeds_by_task,
                             horizon=horizon, model_seeds=model_seeds, split="test",
                             input_manifest=input_manifest)
        # Gate A must have scored the SAME dump this fit was built from
        want = ((fit_spec.get("dump_spec") or {}).get("source") or {}).get("tree_sha256")
        got = ((gate.get("dump") or {}).get("source") or {}).get("tree_sha256")
        if not want or want != got:
            raise ValueError(f"gate_a.json scored a dump with source {got} but the fit was built "
                             f"from {want}")

    def _build(name: str, arm: str, alpha: float, margin: float) -> PolicyBuild:
        adapter = Adapter(alpha=float(alpha), margin_threshold=float(margin))

        def make():
            router, audit = build_routers(arm, fit_dir, model_seeds=model_seeds, tasks=tasks,
                                          adapter=adapter, device=device)
            return adapter_policy(base, router), audit

        return PolicyBuild(name, make, meta={"arm": arm, "alpha": adapter.alpha,
                                             "margin_threshold": adapter.margin_threshold})

    builds = [PolicyBuild("blend_0.5", lambda: (base, None)),
              _build("gnn_residual", doc["arm"], doc["selected_alpha"], doc["selected_margin"])]
    row_sel = doc.get("row", {}).get("selected")
    if row_sel:
        row_entry = doc["row"]["evaluated"][row_sel]
        builds.append(_build("row_residual", doc["baseline_arm"], row_entry["alpha"],
                             row_entry["margin_threshold"]))
    rows = rollout_rows(records, builds, prod=prod, seeds_by_task=seeds_by_task, horizon=horizon,
                        log=log)
    verdict_b = gate_b(rows, policy="gnn_residual", expected_tasks=sorted(seeds_by_task),
                       expected_seeds_by_task=seeds_by_task, require_registered=require_shape)
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "test_rows.csv")
    _write_csv(rows_path, rows)
    summary = [{"policy": p, "task_id": t, **s} for p in sorted({r["policy"] for r in rows})
               for t, s in per_task_summary(rows, p).items()]
    _write_csv(os.path.join(out_dir, "summary.csv"), summary)
    # bound exactly like gate_a.json: the verdict names the rows it came from and their SHA-256,
    # so the standalone verdict stage recomputes it instead of trusting `pass` (review finding 3)
    with open(os.path.join(out_dir, "gate_b.json"), "w") as f:
        json.dump({"policy": "gnn_residual", "blend_policy": "blend_0.5", "adapter": doc,
                   "gate_a": gate_a_path, "gate_a_sha256": _sha256(gate_a_path),
                   "rows": os.path.basename(rows_path), "rows_sha256": _sha256(rows_path),
                   "tasks": sorted(seeds_by_task), "require_shape": bool(require_shape),
                   "test_seeds_by_task": {t: list(v) for t, v in seeds_by_task.items()},
                   "verdict": verdict_b}, f, indent=2)
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(rollout_spec("test", fit_spec=fit_spec, seeds_by_task=seeds_by_task,
                               horizon=horizon, model_seeds=model_seeds, adapter=doc,
                               input_manifest=input_manifest, source_commit=source_commit,
                               extra={"require_shape": bool(require_shape),
                                      "gate_a": gate_a_path, "n_test_rows": len(rows)}),
                  f, indent=2, default=str)
    write_verdict(out_dir, gate_a=verdict, adapter=doc, gate_b=verdict_b)
    log(f"[gate-b] pass={verdict_b['pass']} conditions={verdict_b['conditions']}")
    return verdict_b


def _verified_gate_a(gate: dict, gate_a_path: str, *, manifest_path: str, recompute: Callable,
                     require_binding: bool) -> dict:
    """Return Gate A's verdict only after re-deriving it from the artifacts it names.

    A JSON file saying `pass: true` is not evidence: the verdict is recomputed from the very
    prediction CSV whose SHA-256 the Gate A stage recorded, and the manifest hash it was computed
    against must still be the manifest the adapter dir holds. With `require_binding` (the real
    run) an unbound `gate_a.json` is rejected outright, the recomputation is held to the
    REGISTERED test scope, and a document that admits it was produced with `require_shape: false`
    (a smoke) cannot license a rollout."""
    verdict = gate.get("verdict", gate)
    name, want_sha = gate.get("predictions"), gate.get("predictions_sha256")
    if not name or not want_sha:
        if require_binding:
            raise ValueError(f"{gate_a_path} is not bound to its prediction rows "
                             "(no predictions/predictions_sha256); its verdict cannot be verified")
        return verdict
    if require_binding:
        if gate.get("require_shape") is False:
            raise ValueError(f"{gate_a_path} was produced with require_shape=False (a smoke); it "
                             "cannot authorize a pre-registered rollout")
        missing = [k for k in ("arm", "tasks", "model_seeds", "test_seeds_by_task")
                   if not gate.get(k)]
        if missing:
            raise ValueError(f"{gate_a_path} does not record the scope it certified {missing}; "
                             "an unscoped Gate A cannot be re-derived")
    pred_path = os.path.join(os.path.dirname(os.path.abspath(gate_a_path)), name)
    got_sha = _sha256(pred_path)
    if got_sha != want_sha:
        raise ValueError(f"{name} sha256 {got_sha} != the {want_sha} Gate A recorded: the test "
                         "predictions changed after the offline verdict")
    want_manifest = gate.get("manifest_sha256")
    if want_manifest:
        if not os.path.exists(manifest_path):
            if require_binding:
                raise ValueError(f"gate_a.json is bound to a locked manifest but {manifest_path} "
                                 "was not supplied; its verdict cannot be verified")
        elif want_manifest != _sha256(manifest_path):
            raise ValueError("gate_a.json was computed against a different locked manifest than "
                             "the adapter dir holds")
    import csv

    with open(pred_path) as f:
        rows = list(csv.DictReader(f))
    seeds = gate.get("test_seeds_by_task") or None
    fresh = recompute(rows, arm=gate["arm"], tasks=gate.get("tasks"),
                      model_seeds=[int(s) for s in gate.get("model_seeds", ())] or None,
                      seeds_by_task=seeds, require_registered=require_binding)
    if bool(fresh["pass"]) != bool(verdict.get("pass")):
        raise ValueError(f"recomputing Gate A from {name} gives pass={fresh['pass']}, but "
                         f"{gate_a_path} claims {verdict.get('pass')}")
    return fresh


def _verified_gate_b(gate: dict, gate_b_path: str, *, require_binding: bool) -> dict:
    """The same treatment for Gate B: recompute the closed-loop verdict from the rollout CSV whose
    SHA-256 `gate_b.json` recorded, against the registered scope, and refuse a document that only
    asserts its own result (review finding 3)."""
    verdict = gate.get("verdict", gate)
    name, want_sha = gate.get("rows"), gate.get("rows_sha256")
    if not name or not want_sha:
        if require_binding:
            raise ValueError(f"{gate_b_path} is not bound to its rollout rows "
                             "(no rows/rows_sha256); its verdict cannot be verified")
        return verdict
    if require_binding and gate.get("require_shape") is False:
        raise ValueError(f"{gate_b_path} was produced with require_shape=False (a smoke); it "
                         "cannot stand as the pre-registered closed-loop result")
    rows_path = os.path.join(os.path.dirname(os.path.abspath(gate_b_path)), name)
    got_sha = _sha256(rows_path)
    if got_sha != want_sha:
        raise ValueError(f"{name} sha256 {got_sha} != the {want_sha} Gate B recorded: the rollout "
                         "rows changed after the closed-loop verdict")
    fresh = gate_b(_read_rollout_rows(rows_path), policy=gate.get("policy", "gnn_residual"),
                   blend_policy=gate.get("blend_policy", "blend_0.5"),
                   expected_tasks=gate.get("tasks"),
                   expected_seeds_by_task=gate.get("test_seeds_by_task"),
                   require_registered=require_binding)
    if bool(fresh["pass"]) != bool(verdict.get("pass")):
        raise ValueError(f"recomputing Gate B from {name} gives pass={fresh['pass']}, but "
                         f"{gate_b_path} claims {verdict.get('pass')}")
    return fresh


def _read_rollout_rows(path: str) -> list[dict]:
    """Read `test_rows.csv` back with the numeric types the gate computes on (a CSV reader hands
    back strings, and `"0" < "1"` is not the comparison Gate B means)."""
    import csv

    with open(path) as f:
        raw = list(csv.DictReader(f))
    rows: list[dict] = []
    for r in raw:
        row = dict(r)
        row["trajectory_seed"] = int(float(r["trajectory_seed"]))
        row["cu"] = float(r["cu"])
        row["dead"] = float(r["dead"])
        for k, v in list(row.items()):
            if k.startswith("route_"):
                row[k] = int(float(v)) if str(v).strip() else 0
        rows.append(row)
    return rows


def write_verdict(out_dir: str, *, gate_a: "dict | None", adapter: "dict | None",
                  gate_b: "dict | None", cancelled_by_exp0040: bool = False) -> dict:
    """Emit the single mechanical EXP-0041 result field as an artifact (`verdict.json`).

    The handover requires exactly one result string; leaving it implicit in three separate gate
    files invites a prose reading of a partial pass (review finding 5)."""
    doc = {"experiment": "EXP-0041",
           "verdict": verdict_from_gates(bool(gate_a and gate_a.get("pass")),
                                         bool(adapter and adapter.get("selected")),
                                         None if gate_b is None else bool(gate_b.get("pass")),
                                         cancelled_by_exp0040=cancelled_by_exp0040),
           "gate_a_pass": None if gate_a is None else bool(gate_a.get("pass")),
           "adapter_selected": None if adapter is None else adapter.get("selected"),
           "gate_b_pass": None if gate_b is None else bool(gate_b.get("pass"))}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "verdict.json"), "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def verdict_from_gates(gate_a_pass: bool, adapter_selected: bool, gate_b_pass: "bool | None",
                       *, cancelled_by_exp0040: bool = False) -> str:
    """The single mechanical EXP-0041 result field (HANDOVER §Mechanical Verdict). No prose may
    override it: Gate A alone is never a deployment license, and a failed validation adapter
    selection is `OFFLINE_ONLY`, not a partial pass."""
    if cancelled_by_exp0040:
        return "CANCELLED_BY_EXP0040"
    if not gate_a_pass:
        return "NO_GENERALIZATION"
    if not adapter_selected or not gate_b_pass:
        return "OFFLINE_ONLY"
    return "PASS_RESIDUAL_WITHIN_TOPOLOGY"


def _sha256(path: str) -> str:
    from .graph_value_router_dump import sha256_file

    return sha256_file(path)


def _write_csv(path: str, rows: Sequence[dict]) -> None:
    import csv

    if not rows:
        open(path, "w").close()
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("select", "test", "verdict"), required=True)
    # the verdict stage is artifact-only: it needs no archive, no task records, and no simulator
    ap.add_argument("--archive", default=None)
    ap.add_argument("--task-ids", default=None)
    ap.add_argument("--fit-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--adapter-dir", default=None, help="test stage: dir holding adapter.json")
    ap.add_argument("--gate-a", default=None, help="test stage: gate_a.json (must have passed)")
    ap.add_argument("--prod", default="stock_aware")
    ap.add_argument("--prod-ckpt", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--model-seeds", default="0,1,2")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--source-commit", default=None)
    ap.add_argument("--gate-b", default=None, help="verdict stage: gate_b.json")
    ap.add_argument("--allow-unregistered-rollout", action="store_true",
                    help="skip the pre-registered task/seed/horizon/input checks (smoke only)")
    args = ap.parse_args()

    if args.stage == "verdict":
        # artifact-only, but NOT trust-only: both gates are re-derived from the CSVs they name
        from ..train.graph_generalization_probe import gate_a as recompute_gate_a

        require_binding = not args.allow_unregistered_rollout
        if require_binding and not (args.gate_a and args.adapter_dir):
            raise SystemExit("the verdict stage recomputes its gates: pass --gate-a and "
                             "--adapter-dir (or --allow-unregistered-rollout for a smoke)")
        adapter = (json.load(open(os.path.join(args.adapter_dir, "adapter.json")))
                   if args.adapter_dir else None)
        gate_a_verdict = None
        if args.gate_a:
            gate_a_verdict = _verified_gate_a(
                json.load(open(args.gate_a)), args.gate_a,
                manifest_path=os.path.join(args.adapter_dir or "", "manifest.json"),
                recompute=recompute_gate_a, require_binding=require_binding)
        gate_b_verdict = (_verified_gate_b(json.load(open(args.gate_b)), args.gate_b,
                                           require_binding=require_binding)
                          if args.gate_b else None)
        doc = write_verdict(args.out_dir, gate_a=gate_a_verdict, adapter=adapter,
                            gate_b=gate_b_verdict)
        print(doc["verdict"])
        return
    if not (args.archive and args.task_ids and args.fit_dir):
        raise SystemExit(f"stage {args.stage} requires --archive, --task-ids and --fit-dir")

    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from .baselines import PRODUCING_BASELINES
    from .curriculum_replay import _load_archive
    from .graph_value_router_dump import sha256_file

    by_id = {r.task_id: r for r in _load_archive(args.archive)}
    records = [by_id[t] for t in args.task_ids.split(",")]
    prod = (scorer_producing_policy(load_producer_score_model(args.prod_ckpt))
            if args.prod_ckpt else PRODUCING_BASELINES[args.prod])
    seeds = tuple(int(s) for s in args.model_seeds.split(","))
    require_shape = not args.allow_unregistered_rollout
    # the rollout must be played with the SAME archive + producer the labels came from
    inputs = {"archive": {"path": args.archive, "sha256": sha256_file(args.archive)},
              "producer_ckpt": ({"path": args.prod_ckpt, "sha256": sha256_file(args.prod_ckpt)}
                                if args.prod_ckpt else None)}
    if args.stage == "select":
        run_validation_selection(records, args.fit_dir, args.out_dir, prod=prod, arm=args.arm,
                                 model_seeds=seeds, horizon=args.horizon, device=args.device,
                                 require_shape=require_shape, input_manifest=inputs,
                                 source_commit=args.source_commit)
    else:
        if not args.adapter_dir or not args.gate_a:
            raise SystemExit("test stage requires --adapter-dir and --gate-a")
        run_test_rollout(records, args.fit_dir, args.adapter_dir, args.out_dir, prod=prod,
                         gate_a_path=args.gate_a, model_seeds=seeds, horizon=args.horizon,
                         device=args.device, require_shape=require_shape, input_manifest=inputs,
                         source_commit=args.source_commit)


if __name__ == "__main__":
    main()
