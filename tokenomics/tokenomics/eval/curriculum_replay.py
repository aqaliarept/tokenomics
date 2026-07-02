"""Stage 4 prioritized replay + relabeling (CURRICULUM_GAME_MASTER_APPROACH.md Stage 4).

Turn the Stage-3 mined weakness archive into a curriculum: split it into train / validation
/ blind-holdout replay, priority-sample the train tasks by their mining score, replay each
under the portable stack's state distribution, **relabel** the visited producer/orderer
states with the same teachers the mining reference used, **train** fresh portable scorers
(no schema change), and evaluate the candidate against the champion on the full matrix
(mined validation, blind holdout, transfer, stress, producer-alone + joint). This is a
curriculum/eval tool — it never promotes; promotion stays human-approved (doc).

The pure half (`split_archive`, `score_from_notes`, `replay_priorities`, `sample_replay`)
carries the curation logic and is torch-free/unit-tested; `run_stage4_replay` is the
torch-guarded driver that composes the existing DAgger relabel/train machinery
(`clone_fidelity.dagger_producer_first`, `orderer_improve.dagger_orderer`) over the replayed
configs. Priority sampling follows the doc: `priority_i ∝ (epsilon + score_i) ** eta`.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from .curriculum import TaskRecord

# The archived mining note is self-describing: "mined: score=<s> gap=<g>u win=<w>".
_SCORE_RE = re.compile(r"score=([0-9.]+)")
_GAP_RE = re.compile(r"gap=([0-9.]+)")


def score_from_notes(note: str) -> float:
    """The task's priority score, read from its self-contained archived `notes`.

    Prefers the mining `score=` (the `alpha*normalized_gap + beta*novelty` the archive was
    ranked by); falls back to the `gap=` magnitude so an un-scored record still ranks; 0.0
    if neither is present (never negative — priority weighting assumes a non-negative score)."""
    m = _SCORE_RE.search(note or "")
    if m:
        return float(m.group(1))
    m = _GAP_RE.search(note or "")
    return float(m.group(1)) if m else 0.0


def split_archive(records: list[TaskRecord], *, val_frac: float = 0.25,
                  blind_frac: float = 0.25, seed: int = 0,
                  ) -> tuple[list[TaskRecord], list[TaskRecord], list[TaskRecord]]:
    """Deterministically partition the archive into (train, validation, blind) replay sets.

    The doc keeps three disjoint sets: train replay (the trainer optimizes on), validation
    replay (candidate-vs-champion gate), and a blind holdout the trainer never sees (tail
    check). A fixed seeded shuffle makes the split reproducible; `val`/`blind` take their
    floor'd fractions and the remainder goes to train (so train is never starved — a tiny
    archive keeps every task in train). Independent of the mined ranking, so a lucky
    top-scored task is as likely to land in blind as in train."""
    order = list(records)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_val = int(n * val_frac)
    n_blind = int(n * blind_frac)
    val = order[:n_val]
    blind = order[n_val:n_val + n_blind]
    train = order[n_val + n_blind:]
    return train, val, blind


def replay_priorities(scores: list[float], *, epsilon: float = 0.01,
                      eta: float = 1.0) -> list[float]:
    """Prioritized-replay weights over task scores: `w_i ∝ (epsilon + score_i) ** eta`.

    `epsilon` floors a 0-score task so it never starves; `eta` sharpens (>1 concentrates on
    the highest-regret tasks, =1 is proportional). Returns weights normalized to sum 1 (empty
    in -> empty out). Scores are assumed non-negative (see `score_from_notes`)."""
    if not scores:
        return []
    raw = [(epsilon + max(0.0, s)) ** eta for s in scores]
    total = sum(raw)
    if total <= 0.0:  # degenerate (all-zero, eta huge) -> uniform
        return [1.0 / len(raw)] * len(raw)
    return [r / total for r in raw]


def sample_replay(records: list[TaskRecord], priorities: list[float], k: int, *,
                  seed: int = 0, replace: bool = False) -> list[TaskRecord]:
    """Priority-weighted sample of `k` tasks (deterministic given seed).

    Picks which mined tasks to replay for training this round; higher-priority (higher-regret)
    tasks are favored but the draw is stochastic so the train set isn't frozen to the top-k.

    `replace=False` (default) draws WITHOUT replacement — a distinct subset, at most one copy
    of each task (`k` capped at the pool size). Use this to select a strict subset via an
    explicit `k < len(records)`; note that `k == len(records)` returns the whole pool and
    priorities only reorder it (no actual prioritization).

    `replace=True` draws WITH replacement, so priorities genuinely bite even at `k ==
    len(records)`: high-regret tasks are duplicated and thus replayed more (the intended
    prioritized-replay semantics), and `k` may exceed the pool size."""
    rng = random.Random(seed)
    if replace:
        total = sum(max(0.0, w) for w in priorities)
        if total <= 0.0:  # degenerate -> uniform with replacement
            return [rng.choice(records) for _ in range(k)] if records else []
        return list(rng.choices(records, weights=priorities, k=k)) if records else []
    k = min(k, len(records))
    pool = list(zip(records, priorities))
    picked: list[TaskRecord] = []
    for _ in range(k):
        total = sum(w for _, w in pool)
        if total <= 0.0:
            picked.extend(rec for rec, _ in pool[:k - len(picked)])
            break
        r = rng.random() * total
        acc = 0.0
        for idx, (rec, w) in enumerate(pool):
            acc += w
            if r <= acc:
                picked.append(rec)
                pool.pop(idx)
                break
    return picked


# --- driver (torch-guarded) ----------------------------------------------------------
# Composes the EXISTING score-model relabel/train entry points over the replayed mined
# configs, then evaluates candidate-vs-champion on the shared portable-stack harness. The
# candidate is trained in the SAME portable score-model format as the champion (producer:
# `distill_producer_scorer`, entity_planner teacher == the mining bar's producer; orderer:
# `distill_orderer_scorer`, proportional teacher), so `transfer_matrix` / `eval_joint_stack`
# compare the two apples-to-apples. (Design note: the score-model orderer distills only from
# the proportional teacher, so it cannot absorb the search-oracle headroom the mining
# reference used — retraining on the mined seeds is expected to reproduce the champion; that
# limit is itself the Stage-4 orderer finding, reported, not hidden.)

# The strongest affordable reference the mined gap was measured against (Stage 3 bar).
_MINING_REFERENCE_BAR = "entity_planner+mpc_search"

# The pipeline's ordered phases — emitted as greppable `[phase i/N] name` markers so a
# detached run's local heartbeat (`run_pod_training.write_run_status`) can report progress,
# not just raw log tail. The format is the contract; keep it stable.
_PHASES = ("split + prioritize", "producer distill", "orderer distill",
           "mined validation replay", "blind holdout replay", "transfer + stress",
           "producer-alone + joint")


def _phase(i: int, name: str, log) -> None:
    """Emit an ordered `[phase i/N] name` progress marker (see `_PHASES`)."""
    log(f"[phase {i}/{len(_PHASES)}] {name}")


def _load_archive(path: str | Path) -> list[TaskRecord]:
    """Read a mined `task_archive.jsonl` back into TaskRecords (self-contained generator-refs)."""
    with Path(path).open() as fh:
        return [TaskRecord.from_json_line(line) for line in fh if line.strip()]


def _replay_rows(records: list[TaskRecord], learner_run, stack: str, *,
                 eval_seeds: int, horizon: int, reference_bar: str, log) -> list[dict]:
    """Evaluate one stack (`learner_run`) over a replay split vs the mining reference bar.

    Reuses the Stage-3 `evaluate_pool` (learner-vs-reference on paired seeds), tags each row
    with the `stack` id so champion and candidate rows coexist in one CSV, and keeps the
    regret/win-frac telemetry the mined gap was defined by."""
    from .curriculum_mining import evaluate_pool

    if not records:
        return []
    rows = evaluate_pool(records, learner_run, eval_seeds=eval_seeds, horizon=horizon,
                         reference_bar=reference_bar, log=log)
    for r in rows:
        r["stack"] = stack
        r.pop("descriptor", None)  # tuple, redundant with target_* cols and ugly in CSV
    return rows


def _mean_units(rows: list[dict], stack: str) -> float:
    """Mean learner units for one stack's replay rows (contract failures score 0)."""
    us = [r["learner_units"] or 0.0 for r in rows if r["stack"] == stack]
    return sum(us) / len(us) if us else 0.0


def _min_units(rows: list[dict], stack: str) -> float:
    """Worst-case (tail) learner units for one stack — the blind-holdout tail metric."""
    us = [r["learner_units"] or 0.0 for r in rows if r["stack"] == stack]
    return min(us) if us else 0.0


def _transfer_rows(champ_prod, champ_ord, cand_prod, cand_ord, targets, *,
                   eval_seeds: int, horizon: int, log) -> list[dict]:
    """Champion and candidate portable stacks over the same transfer+stress targets.

    `transfer_matrix` binds one producer+orderer model per call, so it runs twice (champion
    models, then candidate models) on an identical target list; rows are tagged by stack. The
    per-row `consumer_completed_units` + contract flags feed the no-regression gate."""
    from .transfer_matrix import StackSpec, transfer_matrix

    rows: list[dict] = []
    for stack_id, prod, ordr in (("champion", champ_prod, champ_ord),
                                 ("candidate", cand_prod, cand_ord)):
        spec = StackSpec(stack_id, "portable_scorer", "portable_scorer")
        tm = transfer_matrix([spec], targets, eval_seeds=eval_seeds, horizon=horizon,
                             producer_model=prod, orderer_model=ordr, log=log)
        for r in tm:
            r["stack"] = stack_id
        rows += tm
    return rows


def _transfer_regressions(rows: list[dict], *, rel_tol: float = 0.02) -> list[dict]:
    """Targets where the candidate loses value or contract vs champion on the same cell.

    A regression = candidate contract-incompatible where champion loaded, OR candidate units
    below champion by more than `rel_tol` on a cell both ran. These are the Stage-4 reject
    conditions (no promoted-row regression); the caller reports them, never auto-promotes."""
    champ = {r["target_topology"]: r for r in rows if r["stack"] == "champion"}
    cand = {r["target_topology"]: r for r in rows if r["stack"] == "candidate"}
    out: list[dict] = []
    for name, cr in cand.items():
        hr = champ.get(name)
        if hr is None:
            continue
        cand_ok = cr.get("producer_contract_compatible") and cr.get("orderer_contract_compatible")
        champ_ok = hr.get("producer_contract_compatible") and hr.get("orderer_contract_compatible")
        cu_c, cu_h = cr.get("consumer_completed_units", 0.0), hr.get("consumer_completed_units", 0.0)
        contract_reg = champ_ok and not cand_ok
        value_reg = champ_ok and cand_ok and cu_c < cu_h * (1.0 - rel_tol)
        if contract_reg or value_reg:
            out.append({"target": name, "champion_units": round(cu_h, 2),
                        "candidate_units": round(cu_c, 2),
                        "kind": "contract" if contract_reg else "value"})
    return out


def run_stage4_replay(archive_path: str | Path, producer_ckpt: str, orderer_ckpt: str,
                      out_dir: str | Path, *, val_frac: float = 0.25, blind_frac: float = 0.25,
                      split_seed: int = 0, priority_epsilon: float = 0.01,
                      priority_eta: float = 1.0, sample_k: int | None = None, depth: int = 3,
                      epochs: int = 400, iters: int = 400, lr: float = 3e-3, hidden: int = 64,
                      held_out_seeds: tuple[int, ...] = (6, 7, 8, 9), eval_seeds: int = 3,
                      horizon: int = 100, reference_bar: str = _MINING_REFERENCE_BAR,
                      fixtures: bool = True, log=print) -> dict:
    """Stage 4 driver: split the mined archive, relabel+train candidate score models on the
    prioritized train replay, evaluate candidate-vs-champion on the full matrix, emit the EXP
    layout, and return a machine-readable gate. NEVER promotes (doc): the gate is advisory and
    `promote_recommended` is always False here — the single-cell archive caps this at
    'promising', and promotion is human-approved from the written experiment record.

    Composes existing entry points: `distill_producer_scorer` (entity_planner teacher, the
    mining bar's producer) over the replayed train configs, `distill_orderer_scorer`
    (proportional teacher) over the train seeds; `evaluate_pool` for mined-val + blind replay;
    `transfer_matrix` (twice: champion, candidate) for transfer+stress; `eval_joint_stack` for
    producer-alone + joint held-out rows."""
    from .curriculum import replay
    from .curriculum_mining import _joint_stack_learner
    from .joint_stack import eval_joint_stack
    from .transfer_matrix import default_targets, fixture_matrix_targets
    from ..train.orderer_scorer import distill_orderer_scorer, load_scorer
    from ..train.producer_scorer import distill_producer_scorer, load_producer_score_model

    out = Path(out_dir)
    art = out / "artifacts"
    (art / "logs").mkdir(parents=True, exist_ok=True)

    # 1. split + prioritize the mined archive.
    _phase(1, _PHASES[0], log)
    records = _load_archive(archive_path)
    train, val, blind = split_archive(records, val_frac=val_frac, blind_frac=blind_frac,
                                      seed=split_seed)
    scores = [score_from_notes(r.notes) for r in train]
    priorities = replay_priorities(scores, epsilon=priority_epsilon, eta=priority_eta)
    # Default (sample_k=None): draw a full-budget batch WITH replacement so priorities actually
    # bite — high-regret tasks are duplicated and replayed more (true prioritized replay).
    # Without replacement + k==len(train) would select every task once, so priorities would only
    # reorder, not prioritize. An explicit --sample-k picks a distinct subset (no replacement).
    if sample_k is not None:
        sampled = sample_replay(train, priorities, sample_k, seed=split_seed, replace=False)
    else:
        sampled = sample_replay(train, priorities, len(train), seed=split_seed, replace=True)
    train_seeds = [r.world_seed for r in sampled]
    train_configs = [replay(r) for r in sampled]
    held = list(held_out_seeds)
    _seed_counts = {s: train_seeds.count(s) for s in dict.fromkeys(train_seeds)}
    log(f"[stage4] archive={len(records)} train={len(train)} val={len(val)} blind={len(blind)}"
        f" | prioritized train seeds (count)={_seed_counts}")

    # 2. relabel + train candidate score models (champion format).
    prod_path = str(art / "producer_candidate.pt")
    ord_path = str(art / "orderer_candidate.pt")
    _phase(2, _PHASES[1], log)
    log("[stage4] distilling producer candidate (entity_planner teacher over mined configs)...")
    distill_producer_scorer(depth, train_seeds, held, epochs, lr, eval_seeds, horizon,
                            save=prod_path, hidden=hidden, train_configs=train_configs, log=log)
    _phase(3, _PHASES[2], log)
    log("[stage4] distilling orderer candidate (proportional teacher over mined configs)...")
    distill_orderer_scorer(depth, train_seeds, held, iters, eval_seeds, horizon,
                           save=ord_path, hidden=hidden, log=log, train_configs=train_configs)

    # 3. load champion + candidate score models.
    champ_prod, champ_ord = load_producer_score_model(producer_ckpt), load_scorer(orderer_ckpt)
    cand_prod, cand_ord = load_producer_score_model(prod_path), load_scorer(ord_path)
    champ_run = _joint_stack_learner(champ_prod, champ_ord)
    cand_run = _joint_stack_learner(cand_prod, cand_ord)

    # 4. mined validation + blind holdout replay (candidate AND champion vs the mining bar).
    _phase(4, _PHASES[3], log)
    log("[stage4] mined validation replay...")
    val_rows = (_replay_rows(val, champ_run, "champion", eval_seeds=eval_seeds, horizon=horizon,
                             reference_bar=reference_bar, log=log)
                + _replay_rows(val, cand_run, "candidate", eval_seeds=eval_seeds, horizon=horizon,
                               reference_bar=reference_bar, log=log))
    _phase(5, _PHASES[4], log)
    log("[stage4] blind holdout replay...")
    blind_rows = (_replay_rows(blind, champ_run, "champion", eval_seeds=eval_seeds, horizon=horizon,
                               reference_bar=reference_bar, log=log)
                  + _replay_rows(blind, cand_run, "candidate", eval_seeds=eval_seeds,
                                 horizon=horizon, reference_bar=reference_bar, log=log))

    # 5. transfer + stress (no-regression gate).
    _phase(6, _PHASES[5], log)
    log("[stage4] transfer + stress matrix (champion, candidate)...")
    targets = default_targets(depth, held)
    if fixtures:
        targets = targets + fixture_matrix_targets(held, depth=depth)
    tm_rows = _transfer_rows(champ_prod, champ_ord, cand_prod, cand_ord, targets,
                             eval_seeds=eval_seeds, horizon=horizon, log=log)
    regressions = _transfer_regressions(tm_rows)

    # 6. producer-alone + joint held-out (attribution: which side moved).
    _phase(7, _PHASES[6], log)
    log("[stage4] producer-alone + joint held-out...")
    joint_rows: list[dict] = []
    for tag, prod, ordr in (("champion", champ_prod, champ_ord),
                            ("candidate_joint", cand_prod, cand_ord),
                            ("candidate_producer_alone", cand_prod, champ_ord),
                            ("candidate_orderer_alone", champ_prod, cand_ord)):
        for row in eval_joint_stack(prod, ordr, depth, held, eval_seeds, horizon, log=log):
            row["stack"] = tag
            joint_rows.append(row)

    # 7. gate (advisory; never promotes) + emit.
    gate = _stage4_gate(val_rows, blind_rows, tm_rows, regressions)
    _emit_stage4(out, gate, val_rows, blind_rows, tm_rows, joint_rows, regressions,
                 producer_ckpt=producer_ckpt, orderer_ckpt=orderer_ckpt,
                 train_seeds=train_seeds, val_ids=[r.task_id for r in val],
                 blind_ids=[r.task_id for r in blind], depth=depth, reference_bar=reference_bar,
                 priority=(priority_epsilon, priority_eta), split=(val_frac, blind_frac, split_seed))
    log(f"[stage4] gate={gate}")
    return gate


def _stage4_gate(val_rows, blind_rows, tm_rows, regressions) -> dict:
    """The Stage-4 acceptance checks as booleans (kept distinct — never one scalar).

    A row's `contract_ok`/`crashed` come from the mining evaluate. Candidate must: improve or
    tie champion mean units on mined validation, not drop the blind-holdout tail, add zero new
    contract failures, and cause no transfer/stress regression. `promote_recommended` is always
    False (single-cell archive + human-approved promotion)."""
    val_c, val_h = _mean_units(val_rows, "candidate"), _mean_units(val_rows, "champion")
    blind_c, blind_h = _min_units(blind_rows, "candidate"), _min_units(blind_rows, "champion")
    new_contract_failures = sum(
        1 for r in tm_rows if r["stack"] == "candidate"
        and not (r.get("producer_contract_compatible") and r.get("orderer_contract_compatible")))
    tol = 1e-6
    return {
        "val_candidate_units": round(val_c, 3),
        "val_champion_units": round(val_h, 3),
        "val_improves_or_ties": val_c >= val_h - tol,
        "blind_candidate_tail": round(blind_c, 3),
        "blind_champion_tail": round(blind_h, 3),
        "blind_tail_not_worse": blind_c >= blind_h - tol,
        "new_contract_failures": int(new_contract_failures),
        "transfer_regressions": len(regressions),
        "no_transfer_regression": not regressions,
        "promote_recommended": False,  # single-cell archive; promotion is human-approved
    }


def _emit_stage4(out: Path, gate, val_rows, blind_rows, tm_rows, joint_rows, regressions, *,
                 producer_ckpt, orderer_ckpt, train_seeds, val_ids, blind_ids, depth,
                 reference_bar, priority, split) -> None:
    """Write the EXP layout: spec.json, per-split CSVs, summary.csv, and a RESULT.md skeleton
    (conclusions filled by a human from the written artifacts — the driver never promotes)."""
    import json

    from .curriculum import write_matrix_csv

    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(val_rows, art / "replay_validation.csv")
    write_matrix_csv(blind_rows, art / "replay_blind.csv")
    write_matrix_csv(tm_rows, art / "transfer_matrix.csv")
    write_matrix_csv(joint_rows, out / "summary.csv")

    spec = {
        "stage": 4,
        "title": "prioritized replay + relabeling",
        "generator_version": "stage4-v1",
        "champion_producer_ckpt": producer_ckpt,
        "champion_orderer_ckpt": orderer_ckpt,
        "producer_teacher": "entity_planner",
        "orderer_teacher": "proportional",
        "reference_bar": reference_bar,
        "depth": depth,
        "priority_epsilon": priority[0], "priority_eta": priority[1],
        "val_frac": split[0], "blind_frac": split[1], "split_seed": split[2],
        "train_seeds": train_seeds, "val_task_ids": val_ids, "blind_task_ids": blind_ids,
        "gate": gate,
    }
    (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")

    lines = [
        "# EXP-0026 — Stage 4: Prioritized Replay & Relabeling", "",
        "Stage 4 of CURRICULUM_GAME_MASTER_APPROACH.md: replay the Stage-3 mined archive,",
        "relabel + train candidate portable score models, evaluate vs champion. No promotion.", "",
        "## Gate (advisory — promotion is human-approved)", "",
        "```json", json.dumps(gate, indent=2), "```", "",
        "## Caveat", "",
        "Archive is **single-cell** (all mined tasks `build_scale_config depth=3`), so the best",
        "outcome is *promising, not promoted*. The orderer score model distills only from the",
        "proportional teacher and cannot absorb the search-oracle headroom the mined gap used —",
        "the orderer half is expected to reproduce the champion (a scorer-architecture limit, not",
        "a training-data gap). `PROMOTE_RECOMMENDED=false`.", "",
        "## Transfer/stress regressions vs champion", "",
        ("none" if not regressions else "```json\n" + json.dumps(regressions, indent=2) + "\n```"), "",
        "## Conclusions", "", "_TODO (human): fill from the artifacts above._", "",
    ]
    (out / "RESULT.md").write_text("\n".join(lines))


def main() -> None:
    """Stage 4 run. Real train+eval = cloud per CLAUDE.md HARD RULE; local only for --dry-run
    (prints the split + priorities + sampled ids) or a tiny smoke."""
    import argparse

    ap = argparse.ArgumentParser(description="Curriculum Stage 4: prioritized replay + relabeling")
    ap.add_argument("--archive", required=True, help="mined task_archive.jsonl (Stage 3 output)")
    ap.add_argument("--producer-ckpt", required=True, help="champion producer score model")
    ap.add_argument("--orderer-ckpt", required=True, help="champion orderer score model")
    ap.add_argument("--out-dir", default=None, help="experiment dir (writes artifacts/ + summary.csv)")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--blind-frac", type=float, default=0.25)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--priority-epsilon", type=float, default=0.01)
    ap.add_argument("--priority-eta", type=float, default=1.0)
    ap.add_argument("--sample-k", type=int, default=None, help="train tasks to replay (default: all)")
    ap.add_argument("--depth", type=int, default=3, help="chain depth of the mined family")
    ap.add_argument("--epochs", type=int, default=400, help="producer scorer fit epochs")
    ap.add_argument("--iters", type=int, default=400, help="orderer scorer fit iters")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--held-out-seeds", default="6,7,8,9", help="held-out scale seeds for transfer/joint")
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--reference-bar", default=_MINING_REFERENCE_BAR)
    ap.add_argument("--no-fixtures", action="store_true", help="skip the fixture stress targets (faster)")
    ap.add_argument("--dry-run", action="store_true", help="print split + priorities, train nothing")
    args = ap.parse_args()

    records = _load_archive(args.archive)
    train, val, blind = split_archive(records, val_frac=args.val_frac, blind_frac=args.blind_frac,
                                      seed=args.split_seed)
    scores = [score_from_notes(r.notes) for r in train]
    priorities = replay_priorities(scores, epsilon=args.priority_epsilon, eta=args.priority_eta)
    # Mirror run_stage4_replay: explicit --sample-k = distinct subset (no replacement); default
    # = full-budget WITH replacement so priorities duplicate high-regret tasks (real prioritized
    # replay), rather than selecting every task once.
    if args.sample_k is not None:
        sampled = sample_replay(train, priorities, args.sample_k, seed=args.split_seed, replace=False)
    else:
        sampled = sample_replay(train, priorities, len(train), seed=args.split_seed, replace=True)

    if args.dry_run:
        counts = {r.task_id: [s.task_id for s in sampled].count(r.task_id) for r in train}
        print(f"[dry-run] archive={len(records)} -> train={len(train)} val={len(val)} "
              f"blind={len(blind)} (split_seed={args.split_seed})")
        print(f"[dry-run] prioritized train ({len(sampled)} draws, eta={args.priority_eta}); "
              f"'xN' = replay count:")
        ranked = sorted(zip(train, priorities, scores), key=lambda t: -t[1])
        for rec, p, s in ranked:
            n = counts[rec.task_id]
            mark = f"x{n}" if n else "  "
            print(f"  {mark} p={p:.4f} score={s:.4f} {rec.task_id}")
        print(f"[dry-run] val ids: {[r.task_id for r in val]}")
        print(f"[dry-run] blind ids: {[r.task_id for r in blind]}")
        return

    if not args.out_dir:
        ap.error("--out-dir is required (except --dry-run)")
    run_stage4_replay(
        args.archive, args.producer_ckpt, args.orderer_ckpt, args.out_dir,
        val_frac=args.val_frac, blind_frac=args.blind_frac, split_seed=args.split_seed,
        priority_epsilon=args.priority_epsilon, priority_eta=args.priority_eta,
        sample_k=args.sample_k, depth=args.depth, epochs=args.epochs, iters=args.iters,
        lr=args.lr, hidden=args.hidden,
        held_out_seeds=tuple(int(x) for x in args.held_out_seeds.split(",")),
        eval_seeds=args.eval_seeds, horizon=args.horizon, reference_bar=args.reference_bar,
        fixtures=not args.no_fixtures, log=print)


if __name__ == "__main__":
    main()

