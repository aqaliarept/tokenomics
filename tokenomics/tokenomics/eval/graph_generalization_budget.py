"""EXP-0041 compute budget: time a two-decision label smoke and project the full run.

The HANDOVER forbids a pod launch until the implementing agent has measured what labels cost in
BOTH target modes and written a projected CPU-core-hour and artifact-size budget. Label
generation dominates: every contested decision pays one short rollout for the base plus one per
eligible candidate, and `continuation_mean` repeats that whole cost once per continuation (plus
the original-suffix control), so K is a multiplier on the most expensive part of the experiment.

The projection scales a measured per-decision cost by the EXP-0041 episode count, using the
EXP-0039 dump's MEASURED population (13,940 decisions / 37,804 candidates over 9 episodes of the
same three topologies at horizon 100) as the decisions-per-episode estimate. Nothing here runs a
real dump: `max_decisions` stops the traversal after a couple of labeled decisions.
"""

from __future__ import annotations

import json
import time
from typing import Sequence, TYPE_CHECKING

from .graph_generalization_dump import (
    FLAT_TOL as _FLAT_TOL,
    NESTED_TRAIN_SIZES,
    SPLIT_REGISTRY,
    TASK_IDS,
    run_generalization_dump,
)

if TYPE_CHECKING:
    from .baselines import Policy
    from .curriculum import TaskRecord

# Measured EXP-0039 dump population (experiments/EXP-0039-.../artifacts/dump_graph/spec.json):
# 3 tasks x 3 anchored trajectory seeds at horizon 100.
EXP0039 = {"episodes": 9, "decisions": 13940, "candidates": 37804, "shard_bytes": 124 * 1024**2}
DECISIONS_PER_EPISODE = EXP0039["decisions"] / EXP0039["episodes"]
CANDIDATES_PER_DECISION = EXP0039["candidates"] / EXP0039["decisions"]
SHARD_BYTES_PER_DECISION = EXP0039["shard_bytes"] / EXP0039["decisions"]
# candidates.csv is the widest CSV: identity + 2 labels + 7 row features, ~160 bytes/row measured
# on the smoke dumps; continuation_labels.csv adds ~90 bytes per (candidate, continuation).
CSV_BYTES_PER_CANDIDATE = 160
CSV_BYTES_PER_CONTINUATION_ROW = 90
# rollout shape (pre-registered grid + split sizes), used for the episode-count projection
ALPHA_MARGIN_PAIRS = tuple((a, m) for a in (0.10, 0.25, 0.50) for m in (0.0, 0.5, 1.0))
VALIDATION_EPISODES = 9   # 3 tasks x 3 validation seeds
TEST_EPISODES = 9         # 3 tasks x 3 final-test seeds


def episodes_in_registry(registry: "dict | None" = None,
                         splits: Sequence[str] = ("train", "validation", "test")) -> int:
    reg = SPLIT_REGISTRY if registry is None else registry
    return sum(len(reg[t][s]) for t in reg for s in splits)


def measure_label_cost(records: "Sequence[TaskRecord]", *, prod: "Policy", target_mode: str,
                       continuations: int = 0, out_dir: str, delta: float = 1.0,
                       lookahead: int = 8, horizon: int = 100, max_decisions: int = 2,
                       registry: "dict | None" = None, strict_splits: bool = True,
                       log=lambda *a, **k: None) -> dict:
    """Wall-clock seconds per labeled decision, measured on the REAL topology at the real horizon
    and lookahead — a couple of decisions is enough because the per-decision cost is dominated by
    a fixed number of short rollouts, not by trajectory position."""
    t0 = time.time()
    spec = run_generalization_dump(
        records, prod=prod, target_mode=target_mode, continuations=continuations, delta=delta,
        lookahead=lookahead, horizon=horizon, out_dir=out_dir, registry=registry,
        strict_splits=strict_splits, max_decisions=max_decisions,
        # a timing measurement, explicitly not a dataset: it stops after N decisions and its
        # output is never fitted, so it needs no EXP-0040 license (and cannot become one)
        require_license=False, log=log)
    elapsed = time.time() - t0
    n = max(1, spec["n_decisions"])
    return {"target_mode": target_mode, "continuations": continuations,
            "n_decisions": spec["n_decisions"], "n_candidates": spec["n_candidates"],
            "seconds": elapsed, "seconds_per_decision": elapsed / n,
            "seconds_per_candidate": elapsed / max(1, spec["n_candidates"])}


def project(single: dict, mean: "dict | None" = None, *, episodes: "int | None" = None,
            ks: Sequence[int] = (4, 8, 16), curve_sizes: Sequence[int] = NESTED_TRAIN_SIZES
            ) -> dict:
    """Project the full-run label budget from the measured per-decision costs.

    The continuation cost is separated into a fixed part (the original-suffix control, which is
    always paid) and a per-continuation part, so a K that was not measured is extrapolated
    linearly rather than guessed: `cost(K) = cost_single + K * per_continuation`.

    Also reports the probe's extra fits: the learning curve refits the current architecture on the
    smaller nested train sets, so the fit cost is roughly proportional to the summed episode
    counts, not to the primary fit alone."""
    n_eps = episodes if episodes is not None else episodes_in_registry()
    decisions = DECISIONS_PER_EPISODE * n_eps
    candidates = decisions * CANDIDATES_PER_DECISION
    per_dec_single = single["seconds_per_decision"]
    per_cont = None
    if mean is not None and mean.get("continuations"):
        per_cont = max(0.0, (mean["seconds_per_decision"] - per_dec_single)
                       / mean["continuations"])
    modes = {"single_suffix": {"seconds_per_decision": per_dec_single,
                               "core_hours": decisions * per_dec_single / 3600}}
    if per_cont is not None:
        for k in ks:
            cost = per_dec_single + k * per_cont
            modes[f"continuation_mean_k{k}"] = {
                "seconds_per_decision": cost, "core_hours": decisions * cost / 3600,
                "multiple_of_single": cost / per_dec_single if per_dec_single else float("nan")}
    artifacts = {
        "graph_shards_mb": decisions * SHARD_BYTES_PER_DECISION / 1024**2,
        "candidates_csv_mb": candidates * CSV_BYTES_PER_CANDIDATE / 1024**2,
        "continuation_csv_mb_per_k": candidates * CSV_BYTES_PER_CONTINUATION_ROW / 1024**2,
    }
    return {
        "episodes": n_eps, "projected_decisions": decisions, "projected_candidates": candidates,
        "decisions_per_episode": DECISIONS_PER_EPISODE, "modes": modes, "artifacts_mb": artifacts,
        "learning_curve_fit_multiplier": sum(curve_sizes) / max(curve_sizes),
        "basis": "EXP-0039 measured population (13940 decisions / 37804 candidates / 9 episodes)",
        # the second-largest cost after labels: the validation grid rolls out one episode per
        # (policy, task, validation seed), and every contested decision inside costs a graph
        # encode plus one forward per model seed. Stated as episode counts (not seconds) because
        # the per-decision inference cost is measured by the fit stage, not by this smoke.
        "rollout_episodes": {
            "validation_grid": (len(ALPHA_MARGIN_PAIRS) * 2 + 1) * VALIDATION_EPISODES,
            "final_test": 3 * TEST_EPISODES,
            "note": "grid = (|alpha| x |margin| pairs for the GNN and the row baseline) + blend; "
                    "test = blend + gnn_residual + row_residual",
        },
    }


# The fit stage's optimizer runs, as {arm: [(n_model_seeds, fraction of the train-8 data)]}. The
# learning curve refits the current architecture on the nested 2/4/8 train sets, so it is charged
# at its data fraction. `mpnn_4x64_per_task` is one model PER TASK per seed: 3x the optimizer runs
# at 1/3 the data each, i.e. the same total cost as the pooled arm but 9 runs, not 3 — which is
# what makes the stage 27 runs (30 in mean mode) rather than 21.
FIT_RUNS: dict[str, list[tuple[int, float]]] = {
    "row_mlp_h32": [(3, 1.0)],
    "graph_0hop_h32": [(3, 1.0)],
    "mpnn_2x32": [(3, 1.0)],
    "mpnn_4x64": [(3, 1.0)],
    "mpnn_4x64_per_task": [(3, 1.0)],       # 3 task-models per seed over the same total data
    "mpnn_4x64_curve": [(3, 0.25), (3, 0.5)],
    "mpnn_4x64_single_suffix_control": [(3, 1.0)],   # mean mode only
}
FIT_TIMED_ARMS: tuple[str, ...] = ("mpnn_4x64", "mpnn_2x32", "graph_0hop_h32", "row_mlp_h32")
# arms whose per-batch cost is that of another arm (same architecture, different data split)
FIT_COST_ALIAS: dict[str, str] = {"mpnn_4x64_per_task": "mpnn_4x64",
                                  "mpnn_4x64_curve": "mpnn_4x64",
                                  "mpnn_4x64_single_suffix_control": "mpnn_4x64"}
# arms fitted once per TASK instead of once pooled (same total data, more optimizer runs)
FIT_MODELS_PER_SEED: dict[str, int] = {"mpnn_4x64_per_task": len(TASK_IDS)}


def nonflat_fraction(dump_dir: str) -> dict:
    """Share of decisions that are NOT primary-target-flat, per task and overall.

    Training drops flat decisions (`select(..., drop_flat=True)`), so the fit population is this
    fraction of the collected one — on the EXP-0039 dump only ~31% of decisions survive, and
    budgeting the fit on the collected count overstates it by 3x. Reads `primary_flat` when the
    dump records it (EXP-0041) and otherwise re-derives flatness from the candidate labels."""
    import collections
    import csv
    import os

    with open(os.path.join(dump_dir, "decisions.csv")) as f:
        drows = list(csv.DictReader(f))
    flat_col = "primary_flat" if drows and "primary_flat" in drows[0] else None
    by_dec: dict[int, list[float]] = collections.defaultdict(list)
    if flat_col is None:
        with open(os.path.join(dump_dir, "candidates.csv")) as f:
            for r in csv.DictReader(f):
                by_dec[int(r["decision_index"])].append(float(r["teacher_label"]))
    total: dict[str, int] = collections.Counter()
    nonflat: dict[str, int] = collections.Counter()
    episodes: dict[str, set] = collections.defaultdict(set)
    for i, r in enumerate(drows):
        task = r["task_id"]
        total[task] += 1
        episodes[task].add(r.get("trajectory_seed", r.get("seed")))
        if flat_col is not None:
            flat = bool(int(r[flat_col]))
        else:
            vals = by_dec[i]
            flat = len(vals) < 2 or (max(vals) - min(vals)) <= _FLAT_TOL
        if not flat:
            nonflat[task] += 1
    per_task = {t: {"total": total[t], "nonflat": nonflat[t], "episodes": len(episodes[t]),
                    "nonflat_per_episode": nonflat[t] / max(1, len(episodes[t]))}
                for t in sorted(total)}
    n_tot, n_nf = sum(total.values()), sum(nonflat.values())
    return {"per_task": per_task, "total": n_tot, "nonflat": n_nf,
            "nonflat_fraction": n_nf / max(1, n_tot),
            "nonflat_per_episode": sum(v["nonflat_per_episode"] for v in per_task.values())}


def measure_fit_cost(dump_dir: str, *, arms: Sequence[str] = FIT_TIMED_ARMS,
                     batch_decisions: int = 96, reps: int = 6, device: str = "cpu",
                     log=lambda *a, **k: None) -> dict:
    """Wall-clock seconds per training batch, per arm, on REAL graphs from an existing dump.

    Labels are only half the bill and the handover budgeted only that half: the fit is 27 optimizer
    runs (30 in mean mode) of up to 300 epochs, and `mpnn_4x64` is ~87% of it. This times the exact
    step `train_arm` runs — `Batch` construction, forward, loss, backward, clip, `opt.step()` — at
    the pinned single-thread setting, and needs no labels, no license and no teacher: any dump of
    the same schema (EXP-0039's included) measures it."""
    import glob
    import os
    import time

    import numpy as np
    import torch

    from ..train.graph_generalization_probe import ARMS, build_model
    from ..train.graph_router_probe import Batch, FoldNormalizer
    from .graph_generalization_dump import ROW_FEATURES
    from .graph_value_router_dump import load_shard

    torch.set_num_threads(1)  # what train_arm pins; the fan-out is over runs, not threads
    shards = sorted(glob.glob(os.path.join(dump_dir, "graph_shard_*.npz")))
    if not shards:
        raise ValueError(f"no graph shards in {dump_dir}")
    graphs = []
    for shard in shards:
        graphs.extend(load_shard(shard))
        if len(graphs) >= batch_decisions:
            break
    graphs = graphs[:batch_decisions]
    if len(graphs) < batch_decisions:
        raise ValueError(f"{dump_dir} holds {len(graphs)} graphs < batch of {batch_decisions}")
    normed = FoldNormalizer.fit(graphs).apply(graphs)

    out: dict[str, float] = {}
    for arm in arms:
        model = build_model(arm, model_seed=0, dropout=0.10, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        # the row arm consumes candidate rows, not graphs; only the SHAPE drives its cost
        n_cand = int(round(batch_decisions * CANDIDATES_PER_DECISION))
        rows = torch.zeros(n_cand, ROW_FEATURES, device=device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            if ARMS[arm]["kind"] == "row":
                pred = model(rows)
            else:
                pred = model(Batch(normed).to(device))
            loss = (pred ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            times.append(time.perf_counter() - t0)
        out[arm] = float(np.median(times[1:]))  # drop rep 0: allocator/autograd warm-up
        log(f"[fit-cost] {arm}: {out[arm]:.3f} s / {batch_decisions}-decision batch")
    return {"seconds_per_batch": out, "batch_decisions": batch_decisions, "device": device,
            "threads": 1, "n_graphs": len(graphs)}


def project_fit(per_batch: "dict[str, float]", *, train_decisions: int, batch_decisions: int = 96,
                epochs: Sequence[int] = (50, 150, 300), mean_mode: bool = False,
                include_optional: bool = False, cores: int = 16) -> dict:
    """Core-hours for the whole fit stage from measured per-batch costs.

    `train_decisions` must be the NON-FLAT train-8 population (see `nonflat_fraction`). The result
    is a LOWER bound: it excludes the every-`validation_every`-epochs validation pass, and in mean
    mode the averaged target breaks ties a single suffix left flat, which raises `train_decisions`
    itself."""
    from ..train.graph_generalization_probe import OPTIONAL_DIAGNOSTIC_ARMS

    batches = train_decisions / batch_decisions
    runs = {a: s for a, s in FIT_RUNS.items()
            if (mean_mode or a != "mpnn_4x64_single_suffix_control")
            # protocol v2: the per-task arm is an opt-in post-mortem, not part of the mandatory
            # run. The 2/4/8 curve stays mandatory and is always billed.
            and (include_optional or a not in OPTIONAL_DIAGNOSTIC_ARMS)}
    per_arm, total = {}, 0.0
    for arm, specs in runs.items():
        cost = per_batch.get(FIT_COST_ALIAS.get(arm, arm))
        if cost is None:
            raise ValueError(f"no measured per-batch cost for {arm!r} "
                             f"(have {sorted(per_batch)})")
        s = sum(n * frac * cost * batches for n, frac in specs)
        per_arm[arm] = s
        total += s
    return {
        "train_decisions": train_decisions, "batches_per_epoch": batches,
        "n_optimizer_runs": sum(n * FIT_MODELS_PER_SEED.get(arm, 1)
                                for arm, specs in runs.items() for n, _ in specs),
        "seconds_per_epoch_by_arm": per_arm,
        "core_hours_per_epoch": total / 3600,
        "core_hours": {f"epochs_{e}": total * e / 3600 for e in epochs},
        "wall_clock_hours_at_cores": {f"epochs_{e}": total * e / 3600 / cores for e in epochs},
        "dominant_arm_share": max(per_arm.values()) / total if total else float("nan"),
        "note": "LOWER bound: excludes the periodic validation pass; mean mode also enlarges the "
                "non-flat train population by breaking single-suffix ties.",
    }


def load_foreign_dump_for_timing(dump_dir: str, *, val_seed_per_task: "dict | None" = None):
    """An EXP-0039-shaped dump re-read as a `LoadedDump`, FOR TIMING ONLY.

    The epochs-to-early-stop of an `mpnn_4x64` fit is the single largest unknown in the budget (a
    6x swing between `min_epochs` and `max_epochs`), and it can be measured on labels that already
    exist — EXP-0039's — instead of paying for new ones. That dump predates the EXP-0041 schema:
    it has no `split` / `trajectory_seed` / `primary_flat` columns and no deployable row features,
    so this reconstructs the first three and zero-fills the last.

    Deliberately NOT in the probe module and deliberately not certifiable: nothing loaded this way
    may produce a fitted arm, a gate read, or an adapter. It exists to answer "how many epochs"
    and "how many seconds per epoch". Because the row features are zeros, `row_mlp_h32` is not
    measurable from such a dump — time the graph arms only."""
    import collections
    import csv
    import glob
    import json
    import os

    import numpy as np

    from ..train.graph_generalization_probe import Decision, LoadedDump
    from .graph_generalization_dump import ROW_FEATURES
    from .graph_value_router_dump import load_shard

    graphs: list = []
    for shard in sorted(glob.glob(os.path.join(dump_dir, "graph_shard_*.npz"))):
        graphs.extend(load_shard(shard))
    with open(os.path.join(dump_dir, "decisions.csv")) as f:
        drows = list(csv.DictReader(f))
    if len(graphs) != len(drows):
        raise ValueError(f"{len(graphs)} graphs but {len(drows)} decision rows in {dump_dir}")
    by_dec: dict[int, list[dict]] = collections.defaultdict(list)
    with open(os.path.join(dump_dir, "candidates.csv")) as f:
        for r in csv.DictReader(f):
            by_dec[int(r["decision_index"])].append(r)

    # hold out one episode per task as validation so early stopping has something to stop on
    seeds_by_task: dict[str, set] = collections.defaultdict(set)
    for r in drows:
        seeds_by_task[r["task_id"]].add(int(r.get("trajectory_seed", r["seed"])))
    val_seed = dict(val_seed_per_task or {})
    for task, seeds in seeds_by_task.items():
        val_seed.setdefault(task, max(seeds))

    decisions: list[Decision] = []
    for i, r in enumerate(drows):
        cands = sorted(by_dec[i], key=lambda c: int(c["candidate_index"]))
        labels = np.asarray([float(c["teacher_label"]) for c in cands], dtype=np.float64)
        seed = int(r.get("trajectory_seed", r["seed"]))
        flat = len(labels) < 2 or float(labels.max() - labels.min()) <= _FLAT_TOL
        decisions.append(Decision(
            index=i, task_id=r["task_id"],
            split="validation" if seed == val_seed[r["task_id"]] else "train",
            seed=seed, cycle=int(r["cycle"]), order_id=r["order_id"], flat=flat,
            producer_ids=[c["producer_id"] for c in cands], labels=labels,
            rows=np.zeros((len(cands), ROW_FEATURES), dtype=np.float32),
            original_labels=labels, original_flat=flat))
    spec = json.load(open(os.path.join(dump_dir, "spec.json")))
    return LoadedDump(graphs=graphs, decisions=decisions, spec=spec, manifest={})


def measure_fit_epochs(dump_dir: str, *, arm: str = "mpnn_4x64", model_seed: int = 0,
                       max_epochs: "int | None" = None, threads: int = 1,
                       training: "dict | None" = None, out_dir: "str | None" = None,
                       log=lambda *a, **k: None) -> dict:
    """Run ONE real fit and report how many epochs early stopping actually uses.

    This is the measurement the budget is most sensitive to and the one no projection can supply:
    `min_epochs=50` and `max_epochs=300` differ by 6x in cost, and the recipe cannot tell you where
    between them a real fit lands. It needs no new labels.

    `threads` is the knob that makes it affordable. The pre-registered recipe pins one thread so
    reduction order — and therefore the result — is reproducible; the epoch COUNT does not need
    that guarantee, so a timing probe may use many threads and still answer the question. The
    per-epoch seconds it reports are then NOT the single-thread cost to budget with: take those
    from `measure_fit_cost`, and take only the epoch count from here."""
    import json
    import os
    import time

    import torch

    from ..train.graph_generalization_probe import TRAINING, select, train_arm

    dump = load_foreign_dump_for_timing(dump_dir)
    cfg = {**TRAINING, **(training or {})}
    if max_epochs is not None:
        cfg["max_epochs"] = int(max_epochs)
    train_idx = select(dump.decisions, split="train", drop_flat=True)
    val_idx = select(dump.decisions, split="validation", drop_flat=True)
    if not train_idx or not val_idx:
        raise ValueError("timing dump has no non-flat train or validation decisions")
    torch.set_num_threads(max(1, int(threads)))
    log(f"[fit-epochs] {arm} seed {model_seed}: {len(train_idx)} train / {len(val_idx)} "
        f"validation decisions, max_epochs={cfg['max_epochs']}, threads={threads}")
    t0 = time.perf_counter()
    trained = train_arm(arm, dump, train_idx, val_idx, model_seed=model_seed, training=cfg,
                        log=log)
    elapsed = time.perf_counter() - t0
    curve = [h for h in trained.history if h["val_regret"] != ""]
    out = {
        "arm": arm, "model_seed": model_seed, "threads": threads,
        "max_epochs": cfg["max_epochs"], "min_epochs": cfg["min_epochs"],
        "n_train_decisions": len(train_idx), "n_validation_decisions": len(val_idx),
        "epochs_run": trained.epochs_run,
        "best_epoch": trained.best_epochs.get("pooled"),
        "stopped_early": trained.epochs_run < cfg["max_epochs"],
        "seconds": elapsed, "seconds_per_epoch": elapsed / max(1, trained.epochs_run),
        "best_val_regret": trained.best_val_regret,
        "validation_curve": [{"epoch": h["epoch"], "val_regret": h["val_regret"],
                              "train_loss": h["train_loss"]} for h in curve],
        "caveat": "epoch count only; per-epoch seconds are at `threads`, not the pinned "
                  "single-thread cost the budget uses.",
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "fit_epochs.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default=None, help="required unless --fit-only")
    ap.add_argument("--task-ids", default=None,
                    help="one task id is enough for a timing smoke; required unless --fit-only")
    ap.add_argument("--prod", default="stock_aware")
    ap.add_argument("--prod-ckpt", default=None)
    ap.add_argument("--decisions", type=int, default=2, help="labeled decisions per mode")
    ap.add_argument("--continuations", type=int, default=2,
                    help="K used for the mean-mode measurement (extrapolated to 4/8/16)")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fit-dump-dir", default=None,
                    help="measure the FIT stage too (27/30 optimizer runs), on the real graphs of "
                         "an EXISTING dump — EXP-0039's works, no labels or license needed. "
                         "Without it only the label half of the budget is projected.")
    ap.add_argument("--fit-only", action="store_true",
                    help="skip the label smoke (no archive rollouts): fit budget only")
    ap.add_argument("--train-episodes", type=int, default=8,
                    help="train-8 is the pre-registered primary fit size")
    ap.add_argument("--cores", type=int, default=16, help="pod vCPUs, for the wall-clock estimate")
    ap.add_argument("--measure-epochs", action="store_true",
                    help="also run ONE real fit on --fit-dump-dir to measure how many epochs early "
                         "stopping actually uses (the 6x swing the projection cannot supply). "
                         "Needs no new labels.")
    ap.add_argument("--epochs-arm", default="mpnn_4x64")
    ap.add_argument("--epochs-max", type=int, default=None,
                    help="cap the probe below the frozen max_epochs=300 (a capped probe still "
                         "shows where validation regret plateaus)")
    ap.add_argument("--epochs-threads", type=int, default=1,
                    help="threads for the epoch probe ONLY. The frozen recipe pins 1 for "
                         "reproducibility; the epoch COUNT does not need that, so raising this "
                         "buys wall-clock. Budget per-epoch seconds from --fit-only, not from here.")
    args = ap.parse_args()
    if not args.fit_only and not (args.archive and args.task_ids):
        ap.error("--archive and --task-ids are required for the label smoke (or pass --fit-only)")
    if args.fit_only and not args.fit_dump_dir:
        ap.error("--fit-only needs --fit-dump-dir")

    import os

    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from .baselines import PRODUCING_BASELINES
    from .curriculum_replay import _load_archive

    os.makedirs(args.out_dir, exist_ok=True)
    budget: dict = {}
    if args.fit_dump_dir:
        flat = nonflat_fraction(args.fit_dump_dir)
        # the fit trains on non-flat decisions only, scaled from this dump's per-episode rate to
        # the pre-registered train-8 population across all three tasks
        train_decisions = int(round(flat["nonflat_per_episode"] * args.train_episodes))
        fit = measure_fit_cost(args.fit_dump_dir, log=print)
        budget["fit"] = {
            "population": flat, "timing": fit,
            "single_suffix": project_fit(fit["seconds_per_batch"],
                                         train_decisions=train_decisions,
                                         batch_decisions=fit["batch_decisions"],
                                         cores=args.cores),
            "continuation_mean": project_fit(fit["seconds_per_batch"],
                                             train_decisions=train_decisions,
                                             batch_decisions=fit["batch_decisions"],
                                             mean_mode=True, cores=args.cores),
        }
        print(json.dumps(budget["fit"]["single_suffix"], indent=2))
    if args.measure_epochs:
        if not args.fit_dump_dir:
            ap.error("--measure-epochs needs --fit-dump-dir")
        budget["fit_epochs"] = measure_fit_epochs(
            args.fit_dump_dir, arm=args.epochs_arm, max_epochs=args.epochs_max,
            threads=args.epochs_threads, out_dir=args.out_dir, log=print)
        print(json.dumps({k: v for k, v in budget["fit_epochs"].items()
                          if k != "validation_curve"}, indent=2))
    if args.fit_only:
        with open(os.path.join(args.out_dir, "budget.json"), "w") as f:
            json.dump(budget, f, indent=2)
        return

    by_id = {r.task_id: r for r in _load_archive(args.archive)}
    records = [by_id[t] for t in args.task_ids.split(",")]
    prod = (scorer_producing_policy(load_producer_score_model(args.prod_ckpt))
            if args.prod_ckpt else PRODUCING_BASELINES[args.prod])
    single = measure_label_cost(
        records, prod=prod, target_mode="single_suffix", delta=args.delta,
        lookahead=args.lookahead, horizon=args.horizon, max_decisions=args.decisions,
        out_dir=os.path.join(args.out_dir, "timing_single"), log=print)
    mean = measure_label_cost(
        records, prod=prod, target_mode="continuation_mean", continuations=args.continuations,
        delta=args.delta, lookahead=args.lookahead, horizon=args.horizon,
        max_decisions=args.decisions, out_dir=os.path.join(args.out_dir, "timing_mean"), log=print)
    budget.update({"single": single, "mean": mean, "projection": project(single, mean)})
    with open(os.path.join(args.out_dir, "budget.json"), "w") as f:
        json.dump(budget, f, indent=2)
    print(json.dumps(budget["projection"], indent=2))


if __name__ == "__main__":
    main()
