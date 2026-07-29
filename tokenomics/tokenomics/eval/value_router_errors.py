"""EXP-0038 actor/teacher error microscope: explain *why* the EXP-0037 residual actor learns
the marginal-value teacher badly, before training any new orderer (HANDOVER).

For every real blend_0.5 routing decision we already have two signals over the same eligible
producers: the teacher's marginal-value labels (`eval.value_router.marginal_value_labels`) and
the actor's per-producer residual scores (`train.value_router` model). This module compares them
decision by decision — where they disagree, and what state separates the teacher's pick from the
actor's — so the conclusion is grounded in a mistake dump, not a hypothesis (HARD RULE: root cause
before next strategy).

Measurement only; reuses the EXP-0037 teacher + actor so labels match the gate exactly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from ..domain.order import ORDER_TTL
from .baselines import Policy, ordering_blend
from .curriculum import TaskRecord, _episode_seeds, replay
from .joint_stack import joint_stack_env
from .value_router import marginal_value_labels

if TYPE_CHECKING:  # torch only needed to hold the actor; keep the metric API torch-free
    from ..env.aec_env import TokenomicsAECEnv
    from ..train.value_router import ResidualScoreModel


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Tie-aware ranks (ties share their average rank), so a partial tie in labels/scores does not
    get an arbitrary order from argsort's tie-breaking — the Spearman correlation reflects the true
    partial ordering, not the input's incidental layout."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(x.shape[0], dtype=np.float64)
    i = 0
    n = x.shape[0]
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two 1-D arrays (Pearson over tie-aware ranks). Undefined for < 2 points
    or a constant *original* side — a flat label/score vector is exactly a failure mode this
    microscope diagnoses, and argsort of a constant still yields distinct ordinal ranks, so
    constancy must be judged on the raw values, not the ranks. Undefined -> 0.0, so a decision row
    never carries a NaN (or a spurious +1) into decisions.csv / summary."""
    if a.shape[0] < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def summarize(decision_rows: Sequence[dict]) -> dict:
    """The DoD headline (summary.csv): over all decision rows, how often the actor's top-1 matches
    the teacher's and the mean/max teacher regret the actor's picks incur. Empty run -> zeros (no
    division by an empty count) so the smoke still writes a summary."""
    n = len(decision_rows)
    if n == 0:
        return {"n_decisions": 0, "top1_match_rate": 0.0, "mean_regret": 0.0, "max_regret": 0.0}
    regrets = [float(r["decision_regret"]) for r in decision_rows]
    return {
        "n_decisions": n,
        "top1_match_rate": sum(1 for r in decision_rows if r["top1_match"]) / n,
        "mean_regret": sum(regrets) / n,
        "max_regret": max(regrets),
    }


def alias_groups(
    items: Sequence[dict], *, min_group: int = 2, max_examples: int = 5
) -> list[dict]:
    """Group decisions by `feature_hash` (a signature of the actor-visible feature rows) and, per
    group, report how the teacher's preferred producer *position* varies. Identical features with a
    conflicting teacher pick (`label_variance` > 0) means the current features cannot separate what
    the teacher separates — the concrete "missing observation" evidence (HANDOVER DoD). Singletons
    carry no conflict and are dropped by `min_group`. Sorted by conflict then size."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_hash[it["feature_hash"]].append(it)

    groups: list[dict] = []
    for fh, its in by_hash.items():
        if len(its) < min_group:
            continue
        positions = [int(it["teacher_pos"]) for it in its]
        dist = Counter(positions)
        groups.append({
            "feature_hash": fh,
            "n_decisions": len(its),
            "label_variance": float(np.var(positions)),
            "teacher_best_distribution": {str(p): c for p, c in sorted(dist.items())},
            "examples": [it["order_id"] for it in its[:max_examples]],
        })
    groups.sort(key=lambda g: (g["label_variance"], g["n_decisions"]), reverse=True)
    return groups


def top_mistakes(decision_rows: Sequence[dict], n: int) -> list[dict]:
    """The `n` decisions with the highest teacher regret (descending). High-margin misses matter
    most (HANDOVER analysis 2); matched picks (regret ~0) sort last and fall off the truncation."""
    return sorted(decision_rows, key=lambda r: float(r["decision_regret"]), reverse=True)[:n]


def decision_metrics(
    labels: np.ndarray, scores: np.ndarray, eligible_cols: Sequence[int]
) -> dict:
    """Score one routing decision by teacher regret. Over the eligible producer columns, the
    teacher-best is the max marginal-value label and the actor-best is the max residual score;
    `decision_regret` is the teacher value the actor's pick forgoes (teacher-best label minus the
    label at the actor's pick, always >= 0). `top1_match` is whether the two picks agree."""
    cols = list(eligible_cols)
    lab = labels[cols]
    sc = scores[cols]
    t_best = cols[int(np.argmax(lab))]
    a_best = cols[int(np.argmax(sc))]
    t_val_t = float(labels[t_best])
    t_val_a = float(labels[a_best])
    return {
        "teacher_best_producer": t_best,
        "actor_best_producer": a_best,
        "teacher_value_at_teacher_pick": t_val_t,
        "teacher_value_at_actor_pick": t_val_a,
        "decision_regret": t_val_t - t_val_a,
        "top1_match": t_best == a_best,
        "rank_correlation": _spearman(lab, sc),
    }


# --- per-decision record: teacher + actor over one contested routing decision --------
@dataclass
class ErrorRecord:
    """One blend_0.5 routing decision with >= 2 eligible producers, holding both signals over the
    same eligible set (teacher marginal-value `labels`, actor residual `scores`), the actor-visible
    feature `rows`, and the order identity — enough to render every downstream CSV without rerun."""
    task_id: str
    seed: int
    cycle: int
    agent: str
    order_id: str
    stream_index: int
    product: int
    quantity: int
    order_age: int
    ttl_remaining: int
    backlog_size: int
    same_product_backlog: int
    eligible_cols: list[int]
    producer_ids: list[str]
    labels: np.ndarray   # [P] teacher marginal values (0 for ineligible)
    scores: np.ndarray   # [P] actor residual scores (0 for ineligible)
    rows: np.ndarray     # [P, F] actor-visible producer feature rows
    full_rows: "dict | None" = None  # EXP-0038A: {pid -> block-4/block-7 named cols} when full=True


_FS_BOUND = 10.0  # match producer_row_features' _BOUND clip


def producer_fullstate_row(env: TokenomicsAECEnv, oid: str, view, pid: str) -> dict:
    """EXP-0038A blocks 4+7 for one eligible producer — the deployable per-producer facts the
    current 7-feature row throws away (HANDOVER: "do not collapse to min ratios only").

    Block 4 (full producer state): for the order product's required capacity types and inputs, the
    per-type ratios (capacity/reserved/max/headroom over the per-unit requirement; input
    inventory+inbound over the per-unit need) summarised as min/max/mean/std — dispersion the
    binding-min feature hides, which CAN re-rank two producers a decision otherwise ties. Block 7
    (history): `env.system.telemetry` folded EMAs — routed/completed/rejected recent throughput and
    the producer's acceptance/completion/rejection rates. Folded = strictly past-cycle, so deployable
    at decision time (no current-cycle leakage)."""
    from ..domain import metrics as dmetrics

    oi = view.incoming[oid]
    product = oi.product
    sheet = env.system.process_sheets[product]
    need = {t: c for t, c in sheet.require.items() if c > 0}
    p = env.system.producing_agents[pid]
    reserved = p.reserved_capacity()

    def clip(x: float) -> float:
        return float(np.clip(x, -_FS_BOUND, _FS_BOUND))

    cap = [p.capacities.get(t, 0) / c for t, c in need.items()]
    resv = [reserved.get(t, 0) / c for t, c in need.items()]
    mx = [p.max_capacities.get(t, 0) / c for t, c in need.items()]
    head = [(p.max_capacities.get(t, 0) - p.capacities.get(t, 0)) / c for t, c in need.items()]
    inp = [(p.inventory.get(i, 0) + p.outstanding_inbound.get(i, 0)) / n
           for i, n in sheet.inputs.items()]

    def stats(vals: list[float], prefix: str, *, with_std: bool = True) -> dict:
        a = np.array(vals, dtype=np.float64) if vals else np.zeros(1)
        out = {f"{prefix}_min": clip(float(a.min())), f"{prefix}_max": clip(float(a.max())),
               f"{prefix}_mean": clip(float(a.mean()))}
        if with_std:
            out[f"{prefix}_std"] = clip(float(a.std()))
        return out

    row: dict = {}
    for vals, pre in ((cap, "b4_cap"), (resv, "b4_resv"), (mx, "b4_max"), (head, "b4_head")):
        row.update(stats(vals, pre))
    row.update(stats(inp if inp else [_FS_BOUND], "b4_inp", with_std=False))  # primal: no inputs
    row["b4_n_req_types"] = float(len(need))
    row["b4_n_inputs"] = float(len(sheet.inputs))
    row["b4_out_inv"] = clip(p.inventory.get(product, 0))
    row["b4_total_inv"] = clip(sum(p.inventory.values()))
    row["b4_makeable"] = clip(dmetrics.makeable_qty_now(p, product))
    row.update(history_block(env.system.telemetry, pid, product))
    return row


def history_block(telemetry, pid: str, product: int) -> dict:
    """Block-7 / P5 telemetry history for one producer/product (EXP-0038A block 7, reused as the
    EXP-0038B P5 block): routed/completed/rejected recent throughput EMAs + producer
    acceptance/completion/rejection rates. Folded EMAs = strictly past-cycle, so deployable at
    decision time (no current-cycle leakage). This env keeps no exact 5-cycle window, so P5 rides the
    EMA per the HANDOVER fallback."""
    def clip(x: float) -> float:
        return float(np.clip(x, -_FS_BOUND, _FS_BOUND))

    t = telemetry
    return {
        "h_routed_ema": clip(t.routed_ema(pid, product)),
        "h_completed_ema": clip(t.completed_ema(pid, product)),
        "h_rejfail_ema": clip(t.rejected_or_failed_ema(pid, product)),
        "h_success_rate": clip(t.success_rate(pid, product)),
        "h_completion_latency": clip(t.completion_latency(pid, product)),
        "h_prod_acceptance_rate": clip(t.producer_acceptance_rate(pid)),
        "h_prod_completion_rate": clip(t.producer_completion_rate(pid)),
        "h_prod_rejection_rate": clip(t.producer_rejection_rate(pid)),
    }


_PP_BOUND = 50.0  # generous clip for pp_* unit features: guards outliers without flattening the
# throughput/cover dispersion the EXP-0038B ladder is meant to measure (the ladder standardizes).
_SYS_CLIP = 1.0e6  # sys_* counts/qtys are economy-scale (can far exceed _PP_BOUND at s29/s77/s45),
# so clip only to guard non-finite values, never to flatten the demand/capacity magnitudes; the
# ladder standardizes every column anyway.


def _sanitize(x: float, bound: float) -> float:
    """Finite, bounded float for a CSV feature cell — a non-finite value (should never occur; the
    sys_* builders guard every division) becomes 0.0 rather than poisoning the ladder's standardizer."""
    v = float(x)
    if not np.isfinite(v):
        return 0.0
    return float(np.clip(v, -bound, bound))


def producer_product_rows(env: TokenomicsAECEnv, oid: str, view, eligible, *,
                          base_logits: dict, ttl_remaining: int, system: bool = False) -> dict:
    """EXP-0038B `producer_product_v1` observation: the deployable per-producer/product Phase-1
    ladder for one decision, one merged column dict per eligible producer. Unlike order/queue context
    (constant across a decision, EXP-0038A), these blocks vary across candidate producers and can
    re-rank within a decision — the whole point of the re-dump.

    P1 pp_throughput_ (dedicated throughput dispersion), P2 pp_opp_ (opportunity cost vs competing
    products), P3 pp_dep_ (input cover in target units), P4 pp_base_ (base-routing margin, from the
    base policy's per-producer logits), P6 pp_fulfill_ (TTL feasibility), plus the reused P5 h_
    history. Every value clipped to `_PP_BOUND`. `base_logits` maps eligible pid -> base-policy logit
    for this order-stream; `open_qty` for P2 is the orderer's queued quantity by product (deployable
    decision-time demand).

    `system=True` (EXP-0038B Phase 2, `producer_product_system_v1`) also appends the SYSTEM-WIDE
    `sys_*` blocks (S1 demand / S2 capacity / S3 recipe / S4 cutoff-urgency / S5 other-actor). Most
    sys_* values are decision-CONSTANT (one value across the decision's producers) — the ladder scores
    them as interaction context, not additive terms (observation_ladder); only the few producer-
    varying sys_* fields (`*_competing_for_p_capacity_qty`) differ across producers. Computed once per
    decision from live economy state (all orders, all producers, folded telemetry)."""
    from . import producer_product_features as ppf

    oi = view.incoming[oid]
    product, quantity = oi.product, oi.quantity
    open_qty: dict[int, int] = {}
    for o in view.incoming.values():
        open_qty[o.product] = open_qty.get(o.product, 0) + o.quantity
    tel = env.system.telemetry
    elig = list(eligible)
    base_rows = ppf.pp_base_block({pid: base_logits[pid] for pid in elig})

    def clip(x: float) -> float:
        return float(np.clip(x, -_PP_BOUND, _PP_BOUND))

    sys_ctx = _system_context(env, view, elig, product=product, ttl_remaining=ttl_remaining) if system else None

    out: dict = {}
    for pid in elig:
        p = env.system.producing_agents[pid]
        row: dict = {}
        row.update(ppf.pp_throughput_block(p, product=product, quantity=quantity))
        row.update(ppf.pp_opp_block(p, product=product, quantity=quantity, open_qty=open_qty))
        row.update(ppf.pp_dep_block(p, product=product, quantity=quantity))
        row.update(ppf.pp_fulfill_block(p, product=product, quantity=quantity,
                                        ttl_remaining=ttl_remaining, telemetry=tel))
        row.update(base_rows[pid])
        row = {k: clip(v) for k, v in row.items()}
        row.update(history_block(tel, pid, product))  # h_ already clipped to _FS_BOUND
        if sys_ctx is not None:
            row.update({k: _sanitize(v, _SYS_CLIP) for k, v in _system_rows(env, p, sys_ctx).items()})
        out[pid] = row
    return out


def _system_context(env: TokenomicsAECEnv, view, elig, *, product: int, ttl_remaining: int) -> dict:
    """Precompute the per-decision system-wide facts the `sys_*` blocks share (EXP-0038B Phase 2), so
    the per-producer loop only fills in the producer-varying fields. Reads all live orders (economy
    queued demand), the eligible producers' full state, and folded telemetry — all deployable at
    decision time."""
    from . import producer_product_features as ppf

    sheets = env.system.process_sheets
    all_orders = env.system.orders
    open_qty = ppf.system_open_qty(all_orders)
    target_orders = [o for o in all_orders.values()
                     if o.product == product and o.unresolved_quantity() > 0]
    this_qty = sum(o.quantity for o in view.incoming.values() if o.product == product)
    eligible_agents = [env.system.producing_agents[pid] for pid in elig]
    all_agents = list(env.system.producing_agents.values())
    oldest_age = max((o.cycle_counter for o in target_orders), default=0)
    return {
        "sheets": sheets, "open_qty": open_qty, "target_orders": target_orders,
        "this_qty": this_qty, "eligible_agents": eligible_agents, "all_agents": all_agents,
        "product": product, "ttl_remaining": ttl_remaining, "oldest_age": oldest_age,
        "telemetry": env.system.telemetry,
        # capacity/recipe blocks are pure decision-constant — build once, reuse for every producer
        "capacity": ppf.sys_capacity_block(eligible_agents, product=product, sheets=sheets,
                                            all_agents=all_agents, open_qty=open_qty),
        "recipe": ppf.sys_recipe_block(product=product, sheets=sheets, open_qty=open_qty,
                                       all_agents=all_agents),
    }


def _system_rows(env: TokenomicsAECEnv, p, ctx: dict) -> dict:
    """The `sys_*` columns for one eligible producer: the decision-constant capacity/recipe/cutoff
    blocks (shared) plus the demand/actor blocks (which carry the producer-varying
    `*_competing_for_p_capacity_qty` fields)."""
    from . import producer_product_features as ppf

    # quantity for the cutoff urgency proxy: the target's total queued qty this decision
    qty = int(ctx["open_qty"].get(ctx["product"], 0))
    row: dict = {}
    row.update(ppf.sys_demand_block(p, product=ctx["product"], sheets=ctx["sheets"],
                                    open_qty=ctx["open_qty"], target_orders=ctx["target_orders"],
                                    this_orderer_target_qty=ctx["this_qty"]))
    row.update(ctx["capacity"])
    row.update(ctx["recipe"])
    row.update(ppf.sys_cutoff_block(quantity=qty, ttl_remaining=ctx["ttl_remaining"],
                                    oldest_age=ctx["oldest_age"]))
    row.update(ppf.sys_actor_block(p, product=ctx["product"], open_qty=ctx["open_qty"],
                                   telemetry=ctx["telemetry"], all_agents=ctx["all_agents"],
                                   this_orderer_target_qty=ctx["this_qty"]))
    return row


def _actor_scores(model: "ResidualScoreModel", rows: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """The actor's per-producer residual score for one order's rows, zeroed on ineligible producers
    — exactly what `train.value_router.to_score_fn` places into the deployed residual, so the
    microscope scores the same signal the gate deployed."""
    import torch

    with torch.no_grad():
        sc = model(torch.as_tensor(rows, dtype=torch.float32)).numpy()
    return (sc * eligible).astype(np.float32)


def collect_error_records(
    records: Sequence[TaskRecord], prod: Policy, model: "ResidualScoreModel", *,
    delta: float, lookahead: int, eval_seeds: int, horizon: int,
    base: "Policy | None" = None, full: bool = False, feature_set: "str | None" = None,
    log: Callable[..., None] = lambda *a, **k: None,
) -> list[ErrorRecord]:
    """Drive the fixed blend_0.5 trajectory on each task/seed (the states the deployed router
    visits) and, at every ordering decision with >= 2 eligible producers, record the teacher labels
    (`marginal_value_labels`), the actor scores (`model`), the actor-visible feature rows, and the
    order identity. Only contested decisions are kept — where routing weight is a real choice."""
    from ..train.value_router import producer_row_features

    base = base if base is not None else ordering_blend(0.5)
    out: list[ErrorRecord] = []
    for rec in records:
        cfg = replay(rec)
        for s in _episode_seeds(rec.episode_seed, eval_seeds):
            env = joint_stack_env(cfg, horizon)
            env.reset(seed=s)
            for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
                if env.terminations[agent] or env.truncations[agent]:
                    env.step(None)
                    continue
                role, aid = env._role[agent]
                if role == "prod":
                    env.step(prod(env, agent))
                    continue
                view = env.system.ordering_agents[aid].view(env.system.producer_infos)
                incoming = sorted(view.incoming)
                for k, oid in enumerate(incoming):
                    if k >= env.spec_.k_stream:
                        break
                    elig = view.eligible[oid]
                    if len(elig) < 2:
                        continue
                    cols = [env.spec_.producer_ids.index(p) for p in elig]
                    labels = marginal_value_labels(env, prod, base, agent, k, cols, delta, lookahead)
                    rows = producer_row_features(env, oid, view)
                    elig_mask = np.array([pid in elig for pid in env.spec_.producer_ids])
                    scores = _actor_scores(model, rows, elig_mask)
                    oi = view.incoming[oid]
                    order = env.system.orders.get(oid)
                    age = order.cycle_counter if order is not None else -1
                    ttl = max(0, ORDER_TTL - age) if age >= 0 else -1
                    same_prod = sum(1 for o in incoming if view.incoming[o].product == oi.product)
                    if feature_set in ("producer_product_v1", "producer_product_system_v1"):
                        base_mat = np.asarray(base(env, agent), dtype=np.float64).reshape(
                            env.spec_.k_stream, env.spec_.num_producers)
                        base_logits = {p: float(base_mat[k, env.spec_.producer_ids.index(p)])
                                       for p in elig}
                        full_rows = producer_product_rows(
                            env, oid, view, elig, base_logits=base_logits, ttl_remaining=ttl,
                            system=(feature_set == "producer_product_system_v1"))
                    elif full:
                        full_rows = {p: producer_fullstate_row(env, oid, view, p) for p in elig}
                    else:
                        full_rows = None
                    out.append(ErrorRecord(
                        task_id=rec.task_id, seed=s, cycle=env.current_cycle, agent=agent,
                        order_id=oid, stream_index=k, product=oi.product, quantity=oi.quantity,
                        order_age=age, ttl_remaining=ttl, backlog_size=len(incoming),
                        same_product_backlog=same_prod, eligible_cols=cols,
                        producer_ids=list(env.spec_.producer_ids), labels=labels, scores=scores,
                        rows=rows, full_rows=full_rows))
                env.step(base(env, agent))
            log(f"[errors] {rec.task_id} seed {s}: {len(out)} decisions so far")
    return out


# --- render one record into the base CSV rows ----------------------------------------
def decision_row(rec: ErrorRecord) -> dict:
    """The decision-level CSV row: order identity + the teacher/actor comparison
    (`decision_metrics`) with best-producer columns mapped back to producer ids."""
    m = decision_metrics(rec.labels, rec.scores, rec.eligible_cols)
    return {
        "task_id": rec.task_id, "seed": rec.seed, "cycle": rec.cycle, "agent": rec.agent,
        "order_id": rec.order_id, "stream_index": rec.stream_index, "product": rec.product,
        "quantity": rec.quantity, "order_age": rec.order_age, "ttl_remaining": rec.ttl_remaining,
        "backlog_size": rec.backlog_size, "same_product_backlog": rec.same_product_backlog,
        "n_eligible": len(rec.eligible_cols),
        "teacher_best_producer": rec.producer_ids[m["teacher_best_producer"]],
        "actor_best_producer": rec.producer_ids[m["actor_best_producer"]],
        "teacher_value_at_teacher_pick": round(m["teacher_value_at_teacher_pick"], 4),
        "teacher_value_at_actor_pick": round(m["teacher_value_at_actor_pick"], 4),
        "decision_regret": round(m["decision_regret"], 4),
        "top1_match": m["top1_match"],
        "rank_correlation": round(m["rank_correlation"], 4),
    }


def producer_rows(rec: ErrorRecord) -> list[dict]:
    """One row per eligible producer: its teacher label, actor score, and the actor-visible feature
    facts (feature columns, HANDOVER producer-level list) — the delta between the teacher pick and
    the actor pick is read off these rows."""
    rows: list[dict] = []
    for col in rec.eligible_cols:
        r = {
            "task_id": rec.task_id, "seed": rec.seed, "cycle": rec.cycle, "order_id": rec.order_id,
            "producer_id": rec.producer_ids[col], "eligible": True,
            "teacher_label": round(float(rec.labels[col]), 4),
            "actor_score": round(float(rec.scores[col]), 4),
        }
        for fi, val in enumerate(rec.rows[col]):
            r[f"feat_{fi}"] = round(float(val), 4)
        if rec.full_rows is not None:  # EXP-0038A blocks 4+7 ride along when full=True
            for name, val in rec.full_rows[rec.producer_ids[col]].items():
                r[name] = round(float(val), 4)
        rows.append(r)
    return rows


def _alias_item(rec: ErrorRecord) -> dict:
    """Reduce a decision to an aliasing probe: a hash of the actor-visible eligible feature rows
    (order-canonicalised so producer order does not matter) and the teacher-preferred producer's
    POSITION within that canonical order. Two decisions with the same hash but different position
    = identical actor features, conflicting teacher pick.

    The hash is over the EXACT float rows (no rounding): the actor receives full-precision floats,
    so rows differing in any bit are distinguishable to it and must not collapse into a false alias
    — that would manufacture the "missing observation" signal the EXP depends on."""
    m = decision_metrics(rec.labels, rec.scores, rec.eligible_cols)
    elig_rows = [tuple(rec.rows[c].astype(np.float32).tolist()) for c in rec.eligible_cols]
    order = sorted(range(len(rec.eligible_cols)), key=lambda i: elig_rows[i])
    canon = tuple(elig_rows[i] for i in order)
    fh = hashlib.sha1(repr(canon).encode()).hexdigest()[:12]
    t_local = rec.eligible_cols.index(m["teacher_best_producer"])
    teacher_pos = order.index(t_local)
    return {"feature_hash": fh, "teacher_pos": teacher_pos, "order_id": rec.order_id}


# --- end-to-end dump -----------------------------------------------------------------
def _write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fit_reproducible_actor(
    records: Sequence[TaskRecord], prod: Policy, *, delta: float, lookahead: int,
    eval_seeds: int, horizon: int, epochs: int, seed: int,
    log: Callable[..., None] = lambda *a, **k: None,
) -> "ResidualScoreModel":
    """Reproduce the EXP-0037 actor via the seeded fit path (HANDOVER "prefer a seeded rerun for
    auditability" — the checked-in `gate/` artifact is pre-P2). Reuses the exact EXP-0037 collect +
    fit so the microscoped actor IS the gate actor: same labels, same masked-MSE regression."""
    import torch

    from ..train.value_router import (
        ResidualScoreModel,
        collect_router_samples,
        fit_residual_scorer,
    )

    samples = collect_router_samples(records, prod, delta=delta, lookahead=lookahead,
                                     eval_seeds=eval_seeds, horizon=horizon, log=log)
    torch.manual_seed(seed)
    model = ResidualScoreModel()
    fit_residual_scorer(model, samples, epochs=epochs, log=log)
    return model


def run_value_router_errors(
    records: Sequence[TaskRecord], *, prod: Policy, model: "ResidualScoreModel",
    delta: float, lookahead: int, eval_seeds: int, horizon: int, out_dir: str,
    seed: int = 0, top_n: int = 50, base: "Policy | None" = None, full: bool = False,
    feature_set: "str | None" = None,
    log: Callable[..., None] = print,
) -> dict:
    """The microscope dump (HANDOVER first pass): microscope the ALREADY-TRAINED EXP-0037 actor
    `model` against the teacher — collect every contested blend_0.5 decision with both signals and
    write the base CSVs (`decisions`, `producers`) plus the derived `top_mistakes`, `alias_groups`,
    `summary`, and a `spec.json` (`seed` records the actor's fit provenance). Returns the summary
    dict. Local-smoke sized by default; the s29/s77/s45 dump runs on a pod only if needed.

    `full=True` (EXP-0038A) additionally logs per-producer block-4 (full per-type state) + block-7
    (telemetry history) columns into `producers.csv` for the observation-ladder re-dump."""
    from ..train.value_router import PRODUCER_FEATURES

    recs = collect_error_records(records, prod, model, delta=delta, lookahead=lookahead,
                                 eval_seeds=eval_seeds, horizon=horizon, base=base, full=full,
                                 feature_set=feature_set, log=log)
    d_rows = [decision_row(r) for r in recs]
    p_rows = [pr for r in recs for pr in producer_rows(r)]
    mistakes = top_mistakes(d_rows, top_n)
    aliases = alias_groups([_alias_item(r) for r in recs], min_group=2)
    summary = summarize(d_rows)

    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "decisions.csv"), d_rows)
    _write_csv(os.path.join(out_dir, "producers.csv"), p_rows)
    _write_csv(os.path.join(out_dir, "top_mistakes.csv"), mistakes)
    _write_csv(os.path.join(out_dir, "alias_groups.csv"),
               [{**g, "examples": "|".join(g["examples"]),
                 "teacher_best_distribution": json.dumps(g["teacher_best_distribution"])}
                for g in aliases])
    _write_csv(os.path.join(out_dir, "summary.csv"), [summary])
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump({"task_ids": [r.task_id for r in records], "delta": delta,
                   "lookahead": lookahead, "eval_seeds": eval_seeds, "horizon": horizon,
                   "seed": seed, "feat_dim": PRODUCER_FEATURES, "floor": "blend_0.5",
                   "full": full, "feature_set": feature_set,
                   "n_decisions": summary["n_decisions"], "n_alias_groups": len(aliases)},
                  f, indent=2)
    log(f"{summary['n_decisions']} decisions, top1 {summary['top1_match_rate']:.2f}, "
        f"mean regret {summary['mean_regret']:.3f} -> {out_dir}")
    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, help="task archive jsonl")
    ap.add_argument("--task-ids", required=True, help="comma-separated gate task ids (s29,s77,s45)")
    ap.add_argument("--prod", default="stock_aware", help="producer baseline held fixed")
    ap.add_argument("--prod-ckpt", default=None,
                    help="producer scorer checkpoint held fixed instead of --prod (EXP-0035)")
    ap.add_argument("--actor-ckpt", default=None,
                    help="pre-P2 gate/residual_scorer.pt to microscope; omit to seed-rerun the fit")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=300, help="fit epochs for the seeded actor rerun")
    ap.add_argument("--seed", type=int, default=0, help="torch seed for the actor fit (provenance)")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--full", action="store_true",
                    help="EXP-0038A: also log per-producer block-4 (full state) + block-7 (history)")
    ap.add_argument("--feature-set", default=None,
                    choices=["producer_product_v1", "producer_product_system_v1"],
                    help="EXP-0038B: 'producer_product_v1' logs the deployable per-producer/product "
                         "Phase-1 blocks (pp_throughput_/pp_opp_/pp_dep_/pp_base_/pp_fulfill_) + h_ "
                         "history into producers.csv, superseding --full for the 0038B re-dump; "
                         "'producer_product_system_v1' additionally logs the Phase-2 system-wide "
                         "sys_* blocks (demand/capacity/recipe/cutoff/actor)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import torch

    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from ..train.value_router import ResidualScoreModel
    from .baselines import PRODUCING_BASELINES
    from .curriculum_replay import _load_archive

    by_id = {r.task_id: r for r in _load_archive(args.archive)}
    records = [by_id[t] for t in args.task_ids.split(",")]
    prod = (scorer_producing_policy(load_producer_score_model(args.prod_ckpt))
            if args.prod_ckpt else PRODUCING_BASELINES[args.prod])

    if args.actor_ckpt:
        ckpt = torch.load(args.actor_ckpt, map_location="cpu")
        model = ResidualScoreModel()
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = fit_reproducible_actor(
            records, prod, delta=args.delta, lookahead=args.lookahead,
            eval_seeds=args.eval_seeds, horizon=args.horizon, epochs=args.epochs, seed=args.seed,
            log=print)

    run_value_router_errors(
        records, prod=prod, model=model, delta=args.delta, lookahead=args.lookahead,
        eval_seeds=args.eval_seeds, horizon=args.horizon, out_dir=args.out_dir, seed=args.seed,
        top_n=args.top_n, full=args.full, feature_set=args.feature_set)


if __name__ == "__main__":
    main()
