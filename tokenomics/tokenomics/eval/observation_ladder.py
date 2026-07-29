"""EXP-0038A missing-observation audit: does adding a *deployable* observation block make the
EXP-0037 marginal-value teacher label a stable function of the actor's input?

EXP-0038 proved the 7-feature producer row aliases the teacher badly (median label range 10 over
rows seen >= 5x). This module extends that into a cumulative *observation ladder*: for each block
prefix it measures how much the exact-alias / near-neighbor label conflict drops AND whether a
held-out probe actually re-ranks producers better on real routing choices (HANDOVER decision gate).
Alias-spread alone is not enough: order-context fields are constant across the eligible producers
of one decision, so they only shift the between-decision label *level* — the licensing metric is
held-out within-decision regret.

Torch-free measurement; reuses the EXP-0038 `dump/` (no re-simulation) for the blocks it already
logged (current row, order, local queue). Heavier blocks (full producer state, system demand,
recipe bottleneck, history, oracle) need a live re-dump and are out of this module's scope.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Sequence

import numpy as np

# Order-context columns logged per DECISION (constant across a decision's eligible producers),
# joined onto each producer row by the decision key. Integer-typed in the dump.
ORDER_COLS = ("stream_index", "product", "quantity", "order_age", "ttl_remaining")
QUEUE_COLS = ("backlog_size", "same_product_backlog")
CURRENT_COLS = tuple(f"feat_{i}" for i in range(7))

# The cumulative deployable-observation ladder over the blocks the EXP-0038 dump already logged
# (HANDOVER blocks 1-3). Each entry is (name, columns-of-this-block); the ladder is prefix-summed.
BLOCK_LADDER = (
    ("current_row", CURRENT_COLS),
    ("+order", ORDER_COLS),
    ("+local_queue", QUEUE_COLS),
)


def block_columns(up_to: str) -> list[str]:
    """The cumulative feature columns for the (base) ladder prefix ending at block `up_to`."""
    cols: list[str] = []
    for name, block in BLOCK_LADDER:
        cols.extend(block)
        if name == up_to:
            return cols
    raise KeyError(up_to)


# EXP-0038B per-producer/product blocks (Phase 1). Each owns one column prefix so the ladder can
# split it out; per-producer, so unlike order/queue context they CAN re-rank within a decision.
PP_BLOCKS = (
    ("+pp_throughput", "pp_throughput_"),   # P1 dedicated throughput dispersion
    ("+pp_opportunity", "pp_opp_"),         # P2 opportunity cost vs competing products
    ("+pp_dependency", "pp_dep_"),          # P3 input cover in target units
    ("+pp_base", "pp_base_"),               # P4 base-routing margin
    ("+pp_fulfilment", "pp_fulfill_"),      # P6 TTL feasibility
)

# EXP-0038B system-wide blocks (Phase 2). Each owns one column prefix. Most sys_* columns are
# DECISION-CONSTANT (one value across a decision's eligible producers), so unlike the pp_ blocks they
# cannot re-rank producers additively — `_split_cols` routes them into the ridge as interaction
# *context* for the producer-varying columns (feat_/pp_/h_), and kNN reads them as distance dims. The
# few producer-varying sys_ fields (`*_competing_for_p_capacity_qty`) re-rank through those same
# interactions. Appended AFTER +history (see `ladder_for`).
SYS_BLOCKS = (
    ("+sys_demand", "sys_demand_"),       # S1 target + competing demand
    ("+sys_capacity", "sys_capacity_"),   # S2 global capacity scarcity for the target
    ("+sys_recipe", "sys_recipe_"),       # S3 recipe-graph up/downstream pressure
    ("+sys_cutoff", "sys_cutoff_"),       # S4 order urgency (env has no decision-time price)
    ("+sys_actor", "sys_actor_"),         # S5 other-actor presence + recent system flow
)


def ladder_for(rows: Sequence[dict]) -> list[tuple[str, tuple[str, ...]]]:
    """The block ladder actually present in a dump: the base 3 blocks always, plus the per-producer
    blocks a live re-dump logged. An EXP-0038A `full=True` dump adds +full_producer_state (`b4_*`);
    an EXP-0038B `producer_product_v1` dump adds the Phase-1 pp_* blocks (throughput/opportunity/
    dependency/base/fulfilment). Both add +history (`h_*`). Unlike order/queue context, every one of
    these is per-producer, so it CAN re-rank within a decision — the reason for the re-dump."""
    present = set(rows[0]) if rows else set()
    ladder = list(BLOCK_LADDER)
    for name, prefix in PP_BLOCKS:
        cols = tuple(sorted(c for c in present if c.startswith(prefix)))
        if cols:
            ladder.append((name, cols))
    b4 = tuple(sorted(c for c in present if c.startswith("b4_")))
    hist = tuple(sorted(c for c in present if c.startswith("h_")))
    if b4:
        ladder.append(("+full_producer_state", b4))
    if hist:
        ladder.append(("+history", hist))
    # EXP-0038B Phase 2: system-wide blocks ride LAST (after the deployable per-producer ladder), so
    # each sys_ block's held-out probe is read on top of the full Phase-1 observation.
    for name, prefix in SYS_BLOCKS:
        cols = tuple(sorted(c for c in present if c.startswith(prefix)))
        if cols:
            ladder.append((name, cols))
    return ladder


def load_dump(dump_dir: str) -> list[dict]:
    """Join the EXP-0038 `dump/` into one flat record per eligible producer row (HANDOVER: reuse the
    existing dump). `producers.csv` holds the per-producer 7-feature row + teacher label; the order
    context (product, quantity, age, ttl, backlog) lives once per decision in `decisions.csv`. Each
    producer row inherits its decision's context by the (task_id, seed, cycle, order_id) key, so a
    block can key on producer *and* order fields. Numeric fields are coerced; `decision_key` groups
    a decision's producer rows for the within-decision routing probe.

    A `full=True` re-dump (EXP-0038A) adds per-producer block-4 (`b4_*`) and block-7 (`h_*`) columns
    to `producers.csv`; they are discovered by prefix and carried through as float features so the
    +full_producer_state / +history ladder blocks can key/regress on them."""
    with open(os.path.join(dump_dir, "decisions.csv")) as f:
        ctx = {(d["task_id"], d["seed"], d["cycle"], d["order_id"]): d for d in csv.DictReader(f)}

    rows: list[dict] = []
    with open(os.path.join(dump_dir, "producers.csv")) as f:
        reader = csv.DictReader(f)
        extra = [c for c in (reader.fieldnames or []) if c.startswith(("b4_", "h_", "pp_", "sys_"))]
        for d in reader:
            key = (d["task_id"], d["seed"], d["cycle"], d["order_id"])
            c = ctx[key]
            row: dict = {
                "task_id": d["task_id"], "seed": d["seed"], "cycle": d["cycle"],
                "order_id": d["order_id"], "producer_id": d["producer_id"],
                "decision_key": key, "teacher_label": float(d["teacher_label"]),
                "actor_score": float(d["actor_score"]),
            }
            for col in CURRENT_COLS:
                row[col] = float(d[col])
            for col in ORDER_COLS + QUEUE_COLS:
                row[col] = int(c[col])
            for col in extra:
                row[col] = float(d[col])
            rows.append(row)
    return rows


def _split_cols(columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Partition a block's columns into per-producer `current` feats and decision-`context`. Context
    = order/queue fields (constant across a decision, EXP-0038A) PLUS the EXP-0038B Phase-2 `sys_*`
    system-wide fields, which are (mostly) decision-constant too. Per-producer feats (feat_*, and the
    b4_*/h_*/pp_* full-state/history/product columns) enter the ridge directly AND get current x
    context interaction terms; only the interactions let a decision-constant context field re-rank
    producers within one decision (HANDOVER within-decision routing note). A `sys_*` block therefore
    can only license enrichment through its interaction with a producer-varying column, never as a
    standalone additive term — the Phase-2 gate premise. (The few producer-varying sys_ fields still
    re-rank through those interactions; kNN, which reads every column as a distance dim, is the
    additive cross-check.)"""
    context = [c for c in columns if c in ORDER_COLS + QUEUE_COLS or c.startswith("sys_")]
    current = [c for c in columns if c not in context]
    return current, context


def _standardizer(rows: Sequence[dict], columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[r[c] for c in columns] for r in rows], dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _design(rows: Sequence[dict], columns: Sequence[str], mean: np.ndarray, std: np.ndarray,
            *, interactions: bool) -> np.ndarray:
    """Standardized block features, optionally augmented with current x context interaction terms,
    with a leading intercept column (ridge)."""
    cols = list(columns)
    X = (np.array([[r[c] for c in cols] for r in rows], dtype=np.float64) - mean) / std
    if interactions:
        current, context = _split_cols(cols)
        ci = [cols.index(c) for c in current]
        oi = [cols.index(c) for c in context]
        extra = [X[:, i] * X[:, j] for i in ci for j in oi]
        if extra:
            X = np.column_stack([X, *extra])
    return np.column_stack([np.ones(len(rows)), X])


def _decision_centered(rows: Sequence[dict]) -> np.ndarray:
    """Per-decision-centered teacher labels: each label minus the mean over its decision's rows.
    The teacher label carries a large between-decision *level* (bigger orders → uniformly bigger
    marginal values) that order context predicts but that says nothing about *which* producer to
    route to. Centering strips that level so the probe is fit and scored on the within-decision
    *contrast* the routing choice actually needs (the metric RESULT calls 'centered')."""
    by_dec: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        by_dec[r["decision_key"]].append(float(r["teacher_label"]))
    mean = {k: float(np.mean(v)) for k, v in by_dec.items()}
    return np.array([float(r["teacher_label"]) - mean[r["decision_key"]] for r in rows],
                    dtype=np.float64)


def _ridge_predict(train: Sequence[dict], test: Sequence[dict], columns: Sequence[str],
                   lam: float = 1.0) -> np.ndarray:
    mean, std = _standardizer(train, columns)
    Xtr = _design(train, columns, mean, std, interactions=True)
    Xte = _design(test, columns, mean, std, interactions=True)
    ytr = _decision_centered(train)  # fit within-decision contrast, not the between-decision level
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    A[0, 0] -= lam  # do not regularize the intercept
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ w


def _knn_predict(train: Sequence[dict], test: Sequence[dict], columns: Sequence[str],
                 k: int = 10) -> np.ndarray:
    """Nonparametric probe: predict each held-out row's centered label from the mean of its k nearest
    train rows in standardized block space. Because order context enters the distance, matched
    neighbors must share the decision's context too, so kNN can re-rank within a decision without an
    explicit interaction term (the cross-check against ridge, HANDOVER 'both probes agree')."""
    mean, std = _standardizer(train, columns)
    cols = list(columns)
    Xtr = (np.array([[r[c] for c in cols] for r in train], dtype=np.float64) - mean) / std
    Xte = (np.array([[r[c] for c in cols] for r in test], dtype=np.float64) - mean) / std
    ytr = _decision_centered(train)  # neighbors vote on within-decision contrast, not level
    kk = min(k, len(train))
    preds = np.empty(len(test), dtype=np.float64)
    for i, x in enumerate(Xte):
        d = ((Xtr - x) ** 2).sum(axis=1)
        idx = np.argpartition(d, kk - 1)[:kk]
        preds[i] = ytr[idx].mean()
    return preds


def _routing_metrics(test: Sequence[dict], pred: "Sequence[float] | np.ndarray") -> dict:
    """Score a routing signal `pred` (one value per producer row) against the teacher, over the
    REAL-CHOICE decisions only — decisions with >= 2 eligible producers AND a non-flat teacher
    (some producer strictly preferred). Flat decisions carry no preference, so including them just
    dilutes every method equally and hides the signal (HANDOVER 'real-choice regret').

    Every metric is DECISION-WEIGHTED (each real-choice decision counts once) so the probe and the
    floors, which see the same decision set, are directly comparable. `mean_regret` = teacher value
    the argmax(pred) pick forgoes; `top1_match` is TIE-AWARE (a pick whose label ties the best
    counts, matching the floors); `r2` is variance explained on the per-decision-centered labels
    (the within-decision contrast) versus the zero predictor."""
    by_dec: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r, p in zip(test, pred):
        by_dec[r["decision_key"]].append((float(p), float(r["teacher_label"])))

    regrets, matches = [], []
    c_res = c_tot = 0.0
    for dec in by_dec.values():
        if len(dec) < 2:
            continue
        ls = np.array([d[1] for d in dec], dtype=np.float64)
        if np.ptp(ls) == 0.0:  # flat teacher -> not a real choice
            continue
        ps = np.array([d[0] for d in dec], dtype=np.float64)
        best = float(ls.max())
        pick = int(np.argmax(ps))
        regrets.append(best - float(ls[pick]))
        matches.append(int(float(ls[pick]) == best))  # tie-aware, consistent with floor_metrics
        cl = ls - ls.mean()
        cp = ps - ps.mean()
        c_res += float(((cl - cp) ** 2).sum())
        c_tot += float((cl ** 2).sum())
    return {
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "top1_match": float(np.mean(matches)) if matches else 0.0,
        "r2": 1.0 - c_res / c_tot if c_tot > 0 else 0.0,
        "n_decisions": len(regrets),
    }


def probe_fold(train: Sequence[dict], test: Sequence[dict], columns: Sequence[str],
               *, kind: str = "ridge") -> dict:
    """Fit one probe on `train` rows and score its held-out routing on `test` rows for a block.
    `kind` selects the predictor (`ridge` = standardized feats + current x context interactions;
    `knn` = nonparametric). Both fit decision-centered labels; scored by `_routing_metrics`."""
    if kind == "ridge":
        pred = _ridge_predict(train, test, columns)
    elif kind == "knn":
        pred = _knn_predict(train, test, columns)
    else:
        raise ValueError(f"unknown probe kind {kind!r}")
    return _routing_metrics(test, pred)


def floor_metrics(rows: Sequence[dict], *, kind: str = "random") -> dict:
    """Non-learning routing floors the probe must beat (HANDOVER 'compare every block to simple
    floors'), on the SAME real-choice decisions and the SAME decision-weighting as the probe so the
    comparison is apples-to-apples. `actor` = the deployed EXP-0037 residual scorer (`actor_score`
    argmax) — the incumbent — scored through `_routing_metrics`. `random` = the no-information
    ceiling: per real-choice decision, expected regret of a uniform eligible pick (best minus mean
    label) and top-1 = P(uniform pick ties best). No fit, so scored over all of each task's rows."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    out: dict[str, dict] = {}
    for task, task_rows in by_task.items():
        if kind == "actor":
            m = _routing_metrics(task_rows, [float(r["actor_score"]) for r in task_rows])
            if m["n_decisions"]:
                out[task] = {"kind": kind, **m}
        elif kind == "random":
            by_dec: dict[tuple, list[float]] = defaultdict(list)
            for r in task_rows:
                by_dec[r["decision_key"]].append(float(r["teacher_label"]))
            regrets, matches = [], []
            for ls in by_dec.values():
                a = np.array(ls, dtype=np.float64)
                if len(a) < 2 or np.ptp(a) == 0.0:  # real-choice only, like the probe
                    continue
                best = float(a.max())
                regrets.append(best - float(a.mean()))
                matches.append(float((a == best).mean()))  # P(uniform pick ties best)
            if regrets:
                out[task] = {"kind": kind, "mean_regret": float(np.mean(regrets)),
                             "top1_match": float(np.mean(matches)), "n_decisions": len(regrets)}
        else:
            raise ValueError(f"unknown floor kind {kind!r}")
    return out


def heldout_probe(rows: Sequence[dict], columns: Sequence[str], *, kind: str = "ridge") -> dict:
    """Leave-one-seed-out per task (HANDOVER: 'train on two anchored seeds, evaluate on the held-out
    anchored seed'). For each task, hold each seed out once and fit the probe on the others; then
    POOL every held-out (prediction, decision) across folds — since each decision is held out in
    exactly one fold, the pool covers every decision once — and score the pool ONCE with
    `_routing_metrics`. Pooling makes the metric decision-weighted over the whole task, matching the
    floors; averaging fold-means (the earlier bug) weighted small and large folds equally and was
    not comparable to the decision-weighted actor floor. Per-task dict — the gate reads 'improves on
    >= 2 of 3 tasks'."""
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task[r["task_id"]][r["seed"]].append(r)

    out: dict[str, dict] = {}
    for task, by_seed in by_task.items():
        seeds = sorted(by_seed)
        pooled_test: list[dict] = []
        pooled_pred: list[float] = []
        n_folds = 0
        for held in seeds:
            train = [r for s in seeds if s != held for r in by_seed[s]]
            test = by_seed[held]
            if not train or not test:
                continue
            pred = (_ridge_predict(train, test, columns) if kind == "ridge"
                    else _knn_predict(train, test, columns))
            pooled_test.extend(test)
            pooled_pred.extend(float(x) for x in pred)
            n_folds += 1
        if not pooled_test:
            continue
        out[task] = {"n_folds": n_folds, **_routing_metrics(pooled_test, pooled_pred)}
    return out


def alias_spread(rows: Sequence[dict], columns: Sequence[str], min_count: int = 5) -> dict:
    """Exact-alias label spread for a block (HANDOVER metric). Group the producer rows by the tuple
    of `columns` values; for every group seen at least `min_count` times, the label *range*
    (max - min teacher_label) is how much the teacher disagrees given identical block features. A
    block that resolves the aliasing drives these ranges toward 0. Report median/mean/max range and
    the fraction of groups whose range exceeds 1 unit, plus coverage (how many rows / groups the
    repeated-group evidence rests on)."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = tuple(r[c] for c in columns)
        groups[key].append(float(r["teacher_label"]))

    ranges = [max(v) - min(v) for v in groups.values() if len(v) >= min_count]
    n_rows_in_groups = sum(len(v) for v in groups.values() if len(v) >= min_count)
    if not ranges:
        return {"n_groups": 0, "n_rows_in_groups": 0, "median_range": 0.0,
                "mean_range": 0.0, "max_range": 0.0, "frac_gt1": 0.0}
    arr = np.array(ranges)
    return {
        "n_groups": len(ranges),
        "n_rows_in_groups": n_rows_in_groups,
        "median_range": float(np.median(arr)),
        "mean_range": float(np.mean(arr)),
        "max_range": float(np.max(arr)),
        "frac_gt1": float(np.mean(arr > 1.0)),
    }


def actor_mistake_rows(rows: Sequence[dict], n: int) -> list[dict]:
    """Missingness attribution (HANDOVER): the `n` decisions where the deployed actor (`actor_score`
    argmax) forgoes the most teacher value, each rendered with its order context. Because order
    context is constant across a decision's producers, this table shows WHERE the hard decisions
    live in context space (e.g. do they cluster at large quantity / low ttl?), not a within-decision
    separator — that constancy is itself the finding. `teacher_vs_actor_current_differs` flags
    whether the teacher-best and actor-best producers even differ on the current row."""
    by_dec: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_dec[r["decision_key"]].append(r)

    scored: list[dict] = []
    for dec in by_dec.values():
        if len(dec) < 2:
            continue
        labels = np.array([r["teacher_label"] for r in dec], dtype=np.float64)
        a_pick = int(np.argmax([r["actor_score"] for r in dec]))
        t_pick = int(np.argmax(labels))
        regret = float(labels.max() - labels[a_pick])
        cur_a = tuple(dec[a_pick][c] for c in CURRENT_COLS)
        cur_t = tuple(dec[t_pick][c] for c in CURRENT_COLS)
        ex = dec[0]
        scored.append({
            "task_id": ex["task_id"], "seed": ex["seed"], "order_id": ex["order_id"],
            "n_eligible": len(dec), "actor_regret": round(regret, 4),
            "quantity": ex["quantity"], "product": ex["product"], "order_age": ex["order_age"],
            "ttl_remaining": ex["ttl_remaining"], "backlog_size": ex["backlog_size"],
            "same_product_backlog": ex["same_product_backlog"],
            "teacher_vs_actor_current_differs": int(cur_a != cur_t),
        })
    return sorted(scored, key=lambda r: r["actor_regret"], reverse=True)[:n]


def _write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_audit(dump_dir: str, out_dir: str, *, min_count: int = 5, top_n: int = 50,
              log=print) -> dict:
    """The EXP-0038A audit (HANDOVER outputs). Over the deployable block ladder the EXP-0038 dump
    already logged (current_row, +order, +local_queue), emit for each block the exact-alias label
    conflict AND the held-out routing quality of both probes (ridge, knn) per task, benchmarked
    against the random and deployed-actor floors. The gate reads two conditions per block: alias
    conflict collapses AND both probes beat the deployed-actor floor on >= 2/3 tasks. Writes
    block_summary / alias_by_block / probe_by_block / top_mistakes_by_block CSVs + spec.json."""
    rows = load_dump(dump_dir)
    tasks = sorted({r["task_id"] for r in rows})
    floors = {k: floor_metrics(rows, kind=k) for k in ("random", "actor")}
    log(f"[audit] {len(rows)} rows, {len(tasks)} tasks")

    ladder = ladder_for(rows)
    log(f"[audit] blocks: {[b for b, _ in ladder]}")

    alias_rows, probe_rows, block_rows = [], [], []
    cols: list[str] = []
    for block, block_cols in ladder:
        cols = cols + list(block_cols)  # cumulative prefix
        al = alias_spread(rows, cols, min_count)
        alias_rows.append({"block": block, **al})

        probes = {kind: heldout_probe(rows, cols, kind=kind) for kind in ("ridge", "knn")}
        summary = {"block": block, "median_alias_range": al["median_range"],
                   "frac_gt1": round(al["frac_gt1"], 3), "n_alias_groups": al["n_groups"]}
        for kind, hp in probes.items():
            # a task with < 2 seeds yields no leave-one-seed-out fold -> absent from hp; skip it so
            # the audit still runs (the 3-seed gate dump has all tasks; a 1-seed smoke would not).
            ptasks = [t for t in tasks if t in hp and t in floors["actor"]]
            for t in ptasks:
                m = hp[t]
                a_reg = floors["actor"][t]["mean_regret"]
                probe_rows.append({
                    "block": block, "probe": kind, "task": t[-3:],
                    "mean_regret": round(m["mean_regret"], 4), "top1": round(m["top1_match"], 4),
                    "r2": round(m["r2"], 4), "n_decisions": m["n_decisions"],
                    "actor_floor_regret": round(a_reg, 4),
                    "random_floor_regret": round(floors["random"][t]["mean_regret"], 4),
                    "beats_actor_floor": int(m["mean_regret"] < a_reg)})
            summary[f"{kind}_regret"] = (round(float(np.mean([hp[t]["mean_regret"] for t in ptasks])), 4)
                                         if ptasks else None)
            summary[f"{kind}_beats_actor_ntasks"] = sum(
                1 for t in ptasks if hp[t]["mean_regret"] < floors["actor"][t]["mean_regret"])
            summary[f"{kind}_r2"] = (round(float(np.mean([hp[t]["r2"] for t in ptasks])), 4)
                                     if ptasks else None)
        block_rows.append(summary)
        log(f"[audit] {block}: alias med={al['median_range']} frac>1={al['frac_gt1']:.3f} "
            f"ridge beats-actor {summary['ridge_beats_actor_ntasks']}/{len(tasks)} "
            f"knn {summary['knn_beats_actor_ntasks']}/{len(tasks)}")

    mistakes = actor_mistake_rows(rows, top_n)

    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "block_summary.csv"), block_rows)
    _write_csv(os.path.join(out_dir, "alias_by_block.csv"), alias_rows)
    _write_csv(os.path.join(out_dir, "probe_by_block.csv"), probe_rows)
    _write_csv(os.path.join(out_dir, "top_mistakes_by_block.csv"), mistakes)
    spec = {"source_dump": dump_dir, "tasks": tasks, "min_count": min_count,
            "blocks": [b for b, _ in ladder],
            "floors": {k: {t: floors[k][t] for t in tasks} for k in floors},
            "gate": "deployable block licenses enrichment only if alias conflict collapses AND "
                    "BOTH probes beat the deployed-actor floor on >= 2/3 tasks"}
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    return {"block_summary": block_rows, "mistakes": mistakes}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, help="EXP-0038 dump/ (producers.csv + decisions.csv)")
    ap.add_argument("--out-dir", required=True, help="EXP-0038A audit output dir")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()
    run_audit(args.dump_dir, args.out_dir, min_count=args.min_count, top_n=args.top_n)


if __name__ == "__main__":
    main()
