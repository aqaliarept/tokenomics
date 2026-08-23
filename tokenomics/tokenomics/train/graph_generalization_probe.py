"""EXP-0041 pooled generalization probe: model arms, validation-only selection, Offline Gate A.

EXP-0039 fitted one MPNN per task on two trajectories per fold and its held-out scores fell to
the random floor. This probe changes exactly three things and keeps everything else frozen:

1. many independent training episodes (nested 2 -> 4 -> 8) instead of two;
2. one POOLED identity-free fit over the three topologies (no task id, no per-task normalizer);
3. an untouched validation split that drives early stopping, checkpoint choice, and arm choice —
   final-test rows are never loaded during the fit.

The arm ladder (`row_mlp_h32`, `graph_0hop_h32`, `mpnn_2x32`, `mpnn_4x64`, and the per-task
diagnostic) shares one dataset, one loss, one optimizer recipe, and one validation protocol, so a
difference between arms is a difference in the model, not in the pipeline. `row_mlp_h32` is the
FAIR baseline: EXP-0039's actor floor was fitted on the trajectories it was later scored on,
which could not show whether the row actor itself generalizes.

Run order is mechanical (HANDOVER §Immutable run order): `fit` (train + validation only) ->
adapter selection + manifest lock (`eval.graph_residual_router`) -> `gate-a` (the first and only
read of final-test labels). The `gate-a` stage refuses to run without a verified manifest, so no
test result can flow back into a fit, an arm choice, or an adapter choice.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..eval.graph_generalization_dump import (
    NESTED_TRAIN_SIZES,
    SPLIT_REGISTRY,
    TARGET_MODES,
    TASK_IDS,
    validate_split_registry,
)
from ..eval.graph_value_router_dump import GraphArrays, schema_hash
from .graph_router_probe import Batch, FoldNormalizer, RelationalGNN, _pick_device

# --- pre-registered arms + recipe -------------------------------------------------------
ARMS: dict[str, dict] = {
    "row_mlp_h32": {"kind": "row", "hidden": 32, "pooled": True},
    "graph_0hop_h32": {"kind": "graph", "layers": 0, "hidden": 32, "pooled": True},
    "mpnn_2x32": {"kind": "graph", "layers": 2, "hidden": 32, "pooled": True},
    "mpnn_4x64": {"kind": "graph", "layers": 4, "hidden": 64, "pooled": True},
    "mpnn_4x64_per_task": {"kind": "graph", "layers": 4, "hidden": 64, "pooled": False,
                           "diagnostic_only": True},
    # HANDOVER: "If EXP-0040 selects continuation_mean, train one pooled mpnn_4x64 single-suffix
    # control on the same train-8 graph rows." Same architecture, same episodes, same recipe —
    # only the target differs, so it isolates the averaging from everything else. Diagnostic: it
    # can never become the primary target after test results are visible.
    "mpnn_4x64_single_suffix_control": {"kind": "graph", "layers": 4, "hidden": 64, "pooled": True,
                                        "diagnostic_only": True, "target": "original_suffix"},
}
CONTROL_ARM = "mpnn_4x64_single_suffix_control"
# The MANDATORY ladder (protocol v2). `mpnn_4x64_per_task` attributes a pooling effect and does not
# feed any verdict, so it left the mandatory run and is available as an opt-in post-mortem; the
# 2/4/8 learning curve stayed IN, because it is the one instrument that separates "more independent
# trajectories help" from "trajectory coverage is not the missing ingredient" — the question the
# EXP-0039 dump pilot raised when pooling alone failed to close the generalization gap.
MANDATORY_ARMS: tuple[str, ...] = ("row_mlp_h32", "graph_0hop_h32", "mpnn_2x32", "mpnn_4x64")
OPTIONAL_DIAGNOSTIC_ARMS: tuple[str, ...] = ("mpnn_4x64_per_task",)
# The control is fitted only when the licensed target is the mean, so it is not a default arm.
DEFAULT_ARMS: tuple[str, ...] = MANDATORY_ARMS
# Only a POOLED graph arm may win the selection and license the residual adapter: the row MLP is
# the baseline the winner must beat, and the per-task arm exists only to attribute a pooling
# effect (HANDOVER §Model Arms).
SELECTABLE_ARMS: tuple[str, ...] = ("graph_0hop_h32", "mpnn_2x32", "mpnn_4x64")
CURVE_ARM = "mpnn_4x64"
MODEL_SEEDS: tuple[int, ...] = (0, 1, 2)

TRAINING: dict = {
    "max_epochs": 300, "min_epochs": 50, "validation_every": 5, "early_stop_patience": 25,
    "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.10, "gradient_clip_norm": 1.0,
    "batch_decisions_per_task": 32,
    # How the pooled loss shares gradient weight between topologies. `epoch_balanced` equalizes
    # each task's TOTAL weight per epoch (`epoch_task_weights`); `per_batch` is the older
    # within-batch-only balance, which silently favours the largest task through the tail batches
    # it owns alone. Recorded in the recipe so the fit spec says which one produced a number.
    "task_weighting": "epoch_balanced",
}
FLAT_TOL = 1e-9

# The pre-registered FIT shape. `require_shape` certifies the dataset; this certifies the RUN.
# Without it a one-seed, one-epoch, one-arm fit could still be stamped certified and go on to
# license an adapter and a rollout (review finding 2).
FIT_CONTRACT: dict = {"arms": DEFAULT_ARMS, "model_seeds": MODEL_SEEDS, "train_size": 8,
                      "curve_sizes": tuple(NESTED_TRAIN_SIZES), "training": TRAINING,
                      # protocol v2: the curve is a REQUIRED diagnostic of the mandatory run, but
                      # it is only ever a diagnostic — it does not select an architecture, does not
                      # reach the adapter, and neither gate reads it. A flat curve cannot cancel a
                      # test that train-8 earned by passing validation arm-selection, and a rising
                      # curve cannot rescue a train-8 that failed it.
                      "curve_arm": CURVE_ARM, "curve_required": True,
                      "curve_is_gate": False}


def certify_fit_contract(*, arms: Sequence[str], model_seeds: Sequence[int], train_size: int,
                         curve_sizes: Sequence[int], training: "dict | None",
                         source_commit: "str | None") -> None:
    """Refuse a fit that is not the pre-registered one, BEFORE it spends training compute.

    The control arm is added by `run_fit` itself in mean mode, so it is accepted here but cannot
    be requested as a substitute for a registered arm."""
    got_arms = sorted(a for a in arms if a != CONTROL_ARM)
    if got_arms != sorted(FIT_CONTRACT["arms"]):
        raise ValueError(f"fit arms {got_arms} != the pre-registered "
                         f"{sorted(FIT_CONTRACT['arms'])}")
    if sorted(int(s) for s in model_seeds) != sorted(FIT_CONTRACT["model_seeds"]):
        raise ValueError(f"fit model seeds {sorted(model_seeds)} != the pre-registered "
                         f"{list(FIT_CONTRACT['model_seeds'])}")
    if int(train_size) != int(FIT_CONTRACT["train_size"]):
        raise ValueError(f"fit train size {train_size} != the pre-registered "
                         f"{FIT_CONTRACT['train_size']}")
    if tuple(int(s) for s in curve_sizes) != tuple(FIT_CONTRACT["curve_sizes"]):
        raise ValueError(f"learning-curve sizes {tuple(curve_sizes)} != the pre-registered "
                         f"{FIT_CONTRACT['curve_sizes']}")
    changed = {k: v for k, v in (training or {}).items() if TRAINING.get(k) != v}
    if changed:
        raise ValueError(f"fit overrides pre-registered training hyperparameters {changed}; the "
                         "recipe is frozen (a shortened --max-epochs is a smoke, not the run)")
    if len(str(source_commit or "")) != 40:
        raise ValueError(f"fit source commit {source_commit!r} is not a full 40-char revision; a "
                         "real fit must be launched from a committed tree")


def certify_fit_spec(fit_spec: dict) -> None:
    """The same contract, re-checked from a WRITTEN fit spec — used by the rollout stages so a fit
    dir cannot claim `require_shape: true` while recording an off-contract run."""
    if not fit_spec.get("require_shape", False):
        raise ValueError("the fit was produced with require_shape=False (an uncertified dump); "
                         "a pre-registered rollout may not be built on it")
    certify_fit_contract(arms=list(fit_spec.get("arms") or ()),
                         model_seeds=list(fit_spec.get("model_seeds") or ()),
                         train_size=int(fit_spec.get("train_size", -1)),
                         curve_sizes=tuple(fit_spec.get("curve_sizes") or ()),
                         training={k: v for k, v in (fit_spec.get("training") or {}).items()
                                   if TRAINING.get(k) != v},
                         source_commit=(fit_spec.get("source") or {}).get("commit"))


# --- loaded decisions --------------------------------------------------------------------
@dataclass
class Decision:
    """One contested routing decision as the probe consumes it: split keys and audit ids
    (never model tensors), the primary target per candidate, and the deployable 7-feature
    actor rows so the fair row baseline trains on the identical population."""

    index: int              # the dump's global decision_index (audit join key)
    task_id: str
    split: str
    seed: int
    cycle: int
    order_id: str
    flat: bool              # flatness of the PRIMARY target (never of a prediction)
    producer_ids: list[str]
    labels: np.ndarray      # [k] primary labels, candidate_index order
    rows: np.ndarray        # [k, 7] deployable row features, candidate_index order
    # the single-suffix label always rides along: in continuation_mean mode it is the diagnostic
    # CONTROL target the handover requires (same graph rows, same seeds, unaveraged label). In
    # single_suffix mode it is the primary label itself.
    original_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    original_flat: bool = False


@dataclass
class LoadedDump:
    graphs: list[GraphArrays]
    decisions: list[Decision]
    spec: dict
    manifest: dict = field(default_factory=dict)


def control_view(dump: LoadedDump) -> LoadedDump:
    """The same graphs and episodes with the SINGLE-SUFFIX label as the target — the control the
    handover requires in `continuation_mean` mode. Flatness follows the control target, because a
    decision can be flat under the averaged label and contested under the raw one."""
    from dataclasses import replace

    return LoadedDump(graphs=dump.graphs, spec=dump.spec, manifest=dump.manifest,
                      decisions=[replace(d, labels=d.original_labels, flat=d.original_flat)
                                 for d in dump.decisions])


# Candidate columns that are pure identity/audit — no label, no feature. A row outside the
# requested splits is reduced to exactly these, so an excluded label is never even parsed.
IDENTITY_COLS: tuple[str, ...] = ("decision_index", "task_id", "split", "trajectory_seed",
                                  "cycle", "order_id", "decision_digest", "candidate_index",
                                  "producer_id")


def load_generalization_dump(dump_dir: str, *, splits: "Sequence[str] | None" = None,
                             expected_target_mode: "str | None" = None,
                             expected_schema_hash: "str | None" = None,
                             verify_continuation_mean: bool = True,
                             certify: bool = False,
                             expected_registry: "dict | None" = None,
                             expected_splits: "Sequence[str] | None" = None) -> LoadedDump:
    """Load the EXP-0041 dump and validate every alignment the verdict depends on.

    `splits` PHYSICALLY restricts what is READ, not just what is returned: a decision outside the
    requested splits is reduced to its identity columns before anything parses a number, so the
    fit stage (`("train","validation")`) never touches a final-test label, target, continuation
    row, or actor feature — not in a normalizer, not in an early-stopping decision, not in an arm
    choice, not in an adapter choice. Structural alignment is still verified for every row.

    `certify=True` additionally asserts the dump is THE pre-registered dataset
    (`certify_dump_contract`: frozen config, registered tasks, exact collected seeds, pinned
    inputs, committed source, EXP-0040 license). It is off by default because unit fixtures and
    smokes are legitimately not that dump; the real stages turn it on via `require_shape`.

    Rejections (all raise ValueError): missing/mismatched schema hash; a target mode other than
    the expected one; graphs and decisions.csv out of step; candidate_index not exactly 0..k-1;
    candidate count disagreeing with the graph or `n_eligible`; producer ids not decoding from
    the graph via topologies.jsonl; a candidate row whose identity disagrees with its decision
    row; a graph payload whose digest is not the decision's; a non-finite feature or label; an
    episode that appears in more than one split; and — in `continuation_mean` mode — a primary
    label that is not the mean of the persisted continuation labels."""
    import glob

    from ..eval.graph_generalization_dump import certify_dump_contract
    from ..eval.graph_value_router_dump import decision_digest, load_shard

    spec_path = os.path.join(dump_dir, "spec.json")
    if not os.path.exists(spec_path):
        raise ValueError("dump has no spec.json; cannot certify it")
    spec = json.load(open(spec_path))
    schema_path = os.path.join(dump_dir, "graph_schema.json")
    if not os.path.exists(schema_path):
        raise ValueError("dump is missing graph_schema.json")
    stamped = json.load(open(schema_path)).get("schema_hash")
    want_schema = expected_schema_hash or schema_hash()
    if stamped != want_schema or spec.get("schema_hash") != want_schema:
        raise ValueError(f"dump schema hash {stamped}/{spec.get('schema_hash')} != {want_schema}: "
                         "the probe would score against a different observation")
    mode = spec.get("target_mode")
    if mode not in TARGET_MODES:
        raise ValueError(f"dump target_mode {mode!r} is not one of {TARGET_MODES}")
    if expected_target_mode is not None and mode != expected_target_mode:
        raise ValueError(f"dump target_mode {mode!r} != expected {expected_target_mode!r}")

    graphs_all = [g for s in sorted(glob.glob(os.path.join(dump_dir, "graph_shard_*.npz")))
                  for g in load_shard(s)]
    with open(os.path.join(dump_dir, "decisions.csv")) as f:
        drows = list(csv.DictReader(f))
    if len(graphs_all) != len(drows):
        raise ValueError(f"{len(graphs_all)} graphs but {len(drows)} decisions.csv rows")
    # WHICH decisions may be read at all — decided from the split column alone, before any label
    # or feature is parsed anywhere below (review finding 4).
    included = {di for di, r in enumerate(drows) if splits is None or r["split"] in splits}
    topo = {}
    with open(os.path.join(dump_dir, "topologies.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            topo[rec["task_id"]] = rec["producer_ids"]

    by_dec: dict[int, list[dict]] = defaultdict(list)
    n_candidate_rows = 0
    with open(os.path.join(dump_dir, "candidates.csv")) as f:
        for r in csv.DictReader(f):
            n_candidate_rows += 1
            di = int(r["decision_index"])
            # an excluded decision keeps ONLY its identity columns: its label/features are never
            # parsed, never normalized, never validated, never in memory as a number
            by_dec[di].append(r if di in included else {k: r[k] for k in IDENTITY_COLS})
    if set(by_dec) != set(range(len(drows))):
        raise ValueError("candidates.csv decision_index is not exactly one group per decision")

    cont_by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    cont_path = os.path.join(dump_dir, "continuation_labels.csv")
    if mode == "continuation_mean":
        if not os.path.exists(cont_path):
            raise ValueError("continuation_mean dump is missing continuation_labels.csv")
        with open(cont_path) as f:
            for r in csv.DictReader(f):
                di = int(r["decision_index"])
                if di not in included:
                    continue  # an excluded decision's raw continuation labels stay unread too
                cont_by_key[(di, int(r["candidate_index"]))].append(float(r["label"]))
    if certify:
        from ..eval.graph_generalization_dump import SPLITS as _ALL_SPLITS

        certify_dump_contract(spec, expected_target_mode=expected_target_mode,
                              expected_registry=expected_registry, n_decisions=len(drows),
                              n_candidates=n_candidate_rows,
                              expected_splits=(_ALL_SPLITS if expected_splits is None
                                               else expected_splits))

    split_of_episode: dict[tuple[str, int], str] = {}
    graphs: list[GraphArrays] = []
    decisions: list[Decision] = []
    for di, drow in enumerate(drows):
        group = sorted(by_dec[di], key=lambda r: int(r["candidate_index"]))
        k = len(group)
        if [int(r["candidate_index"]) for r in group] != list(range(k)):
            raise ValueError(f"decision {di}: candidate_index is not 0..{k - 1}")
        if k != graphs_all[di].candidate_producers.shape[0] or k != int(drow["n_eligible"]):
            raise ValueError(f"decision {di}: candidate count {k} != graph/n_eligible")
        pids = [topo[drow["task_id"]][int(local)] for local in graphs_all[di].candidate_producers]
        if pids != [r["producer_id"] for r in group]:
            raise ValueError(f"decision {di}: producer alignment mismatch vs candidates.csv")
        key = (drow["task_id"], drow["trajectory_seed"], drow["cycle"], drow["order_id"])
        for r in group:
            if (r["task_id"], r["trajectory_seed"], r["cycle"], r["order_id"]) != key:
                raise ValueError(f"decision {di}: candidate metadata disagrees with decisions.csv")
            if r["split"] != drow["split"]:
                raise ValueError(f"decision {di}: candidate split {r['split']} != {drow['split']}")
        expect = decision_digest(*key)
        if graphs_all[di].decision_digest != expect or int(drow["decision_digest"]) != expect:
            raise ValueError(f"decision {di}: graph payload not bound to its decisions.csv row")

        episode = (drow["task_id"], int(drow["trajectory_seed"]))
        prior = split_of_episode.setdefault(episode, drow["split"])
        if prior != drow["split"]:
            raise ValueError(f"episode {episode} appears in both {prior} and {drow['split']} "
                             "— splits must be whole episodes")
        if di not in included:
            continue  # structurally verified above; nothing numeric about it is read
        labels = np.array([float(r["primary_label"]) for r in group], dtype=np.float64)
        original = np.array([float(r["original_suffix_label"]) for r in group], dtype=np.float64)
        rows = np.array([[float(r[f"feat_{i}"]) for i in range(7)] for r in group],
                        dtype=np.float32)
        if not np.isfinite(labels).all() or not np.isfinite(rows).all() \
                or not np.isfinite(original).all():
            raise ValueError(f"decision {di}: non-finite label or row feature")
        if mode == "continuation_mean" and verify_continuation_mean:
            for ci in range(k):
                got = cont_by_key.get((di, ci))
                if not got:
                    raise ValueError(f"decision {di} candidate {ci}: no continuation labels")
                if len(got) != int(drow["continuation_count"]):
                    raise ValueError(f"decision {di} candidate {ci}: {len(got)} continuation rows "
                                     f"!= declared {drow['continuation_count']}")
                if abs(float(np.mean(got)) - labels[ci]) > 1e-5:
                    raise ValueError(
                        f"decision {di} candidate {ci}: primary_label {labels[ci]} is not the mean "
                        f"of its persisted continuation labels ({np.mean(got)})")
        graphs.append(graphs_all[di])
        decisions.append(Decision(
            index=di, task_id=drow["task_id"], split=drow["split"],
            seed=int(drow["trajectory_seed"]), cycle=int(drow["cycle"]),
            order_id=drow["order_id"], flat=bool(int(drow["primary_flat"])),
            producer_ids=[r["producer_id"] for r in group], labels=labels, rows=rows,
            original_labels=original, original_flat=bool(int(drow["original_flat"]))))
    for g in graphs:
        for arr in list(g.node_features.values()) + list(g.edge_attr.values()):
            if arr.size and not np.isfinite(arr).all():
                raise ValueError("non-finite feature in loaded dump")
    manifest = spec.get("split_manifest", {})
    if manifest.get("registry"):
        validate_split_registry({t: {n: tuple(v) for n, v in s.items()}
                                 for t, s in manifest["registry"].items()}, strict=False)
    return LoadedDump(graphs=graphs, decisions=decisions, spec=spec, manifest=manifest)


# --- selection over loaded decisions -------------------------------------------------------
def select(decisions: Sequence[Decision], *, split: "str | None" = None,
           tasks: "Sequence[str] | None" = None, seeds: "Sequence[int] | None" = None,
           seeds_by_task: "Mapping[str, Sequence[int]] | None" = None,
           drop_flat: bool = False) -> list[int]:
    """Positions (into `decisions`) matching a split / task / episode-seed filter. `drop_flat`
    removes primary-target-flat decisions — done for TRAINING and for the primary offline
    metrics, never for the population counts, which are reported in full."""
    out: list[int] = []
    for i, d in enumerate(decisions):
        if split is not None and d.split != split:
            continue
        if tasks is not None and d.task_id not in tasks:
            continue
        if seeds is not None and d.seed not in set(seeds):
            continue
        if seeds_by_task is not None and d.seed not in set(seeds_by_task.get(d.task_id, ())):
            continue
        if drop_flat and d.flat:
            continue
        out.append(i)
    return out


def flatten(decisions: Sequence[Decision], indices: Sequence[int]) -> tuple:
    """Per-candidate arrays for a decision subset, concatenated in (decision, candidate_index)
    order — the canonical prediction order every artifact uses. Returns
    `(labels, group, task_of_group, rows)`."""
    labels, group, rows, task_of_group = [], [], [], []
    for local, i in enumerate(indices):
        d = decisions[i]
        labels.extend(d.labels.tolist())
        group.extend([local] * len(d.labels))
        rows.append(d.rows)
        task_of_group.append(d.task_id)
    return (np.asarray(labels, dtype=np.float64), np.asarray(group, dtype=np.int64),
            task_of_group, (np.concatenate(rows, axis=0) if rows
                            else np.zeros((0, 7), dtype=np.float32)))


def task_balanced_batches(decisions: Sequence[Decision], indices: Sequence[int], *,
                          model_seed: int, epoch: int, per_task: int = 32) -> list[list[int]]:
    """Minibatches that keep every decision INTACT and give each task equal weight: shuffle each
    task's decisions with a deterministic `(model_seed, epoch)` stream, cut them into chunks of
    `per_task`, and zip the chunks across tasks (so a full batch is `per_task x n_tasks`
    decisions). A task that runs out contributes nothing to the later batches rather than being
    resampled, so no decision is seen twice in one epoch."""
    by_task: dict[str, list[int]] = defaultdict(list)
    for i in indices:
        by_task[decisions[i].task_id].append(i)
    chunks: dict[str, list[list[int]]] = {}
    for task in sorted(by_task):
        order = list(by_task[task])
        rng = np.random.default_rng([model_seed, epoch, len(order)])
        perm = rng.permutation(len(order))
        shuffled = [order[p] for p in perm]
        chunks[task] = [shuffled[s:s + per_task] for s in range(0, len(shuffled), per_task)]
    n_batches = max((len(c) for c in chunks.values()), default=0)
    return [[i for task in sorted(chunks) if b < len(chunks[task]) for i in chunks[task][b]]
            for b in range(n_batches)]


def epoch_task_weights(decisions: Sequence[Decision], indices: Sequence[int], *,
                       per_task: int = 32) -> dict[str, float]:
    """Per-task loss weights that make each task's TOTAL gradient weight per epoch equal.

    `task_balanced_batches` gives every task an equal share of each batch, but only while it still
    has decisions left: the tasks with more decisions own the tail batches alone, and a per-batch
    equal-weight loss then hands them full weight there. On the registered topologies that is a
    real skew — s77 holds 1.43x the decisions of s29 but would take 2.16x the epoch gradient
    weight — so the pooled fit would be defined by the largest topology, which is exactly what
    pooling is supposed to prevent.

    The fix keeps sampling WITHOUT replacement (no decision is seen twice, no task is truncated)
    and corrects the weight instead: a task appearing in `n_t` batches is weighted `1 / n_t` per
    batch, so it sums to the same total over the epoch whichever batches it appears in. The
    `mean_chunks / n_tasks` factor rescales so the objective the frozen learning rate and clip
    norm were calibrated against is not silently inflated — a uniform rescale of the loss is
    indistinguishable from a changed learning rate. It is preserved EXACTLY for equal-sized tasks
    (the weights are then the old uniform `1 / n_tasks`) and to +2.3% on the registered profile:
    the weights sum to >= 1 by AM-HM, with the excess set by the spread of the per-task batch
    counts. The real change is confined to the tail batches, where the largest task used to sit
    alone at full weight and now carries 0.27 — 3.7x less. That is the correction itself, not a
    side effect of the rescale. Pinned by
    `test_epoch_weights_preserve_loss_scale_on_full_batches_and_only_shrink_the_tail`."""
    chunk_count: dict[str, int] = {}
    counts: dict[str, int] = defaultdict(int)
    for i in indices:
        counts[decisions[i].task_id] += 1
    for task, n in counts.items():
        chunk_count[task] = max(1, math.ceil(n / max(1, per_task)))
    if not chunk_count:
        return {}
    mean_chunks = sum(chunk_count.values()) / len(chunk_count)
    return {t: mean_chunks / (len(chunk_count) * n) for t, n in chunk_count.items()}


# --- loss --------------------------------------------------------------------------------
def _center(v: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    sums = torch.zeros(n_groups, device=v.device).index_add_(0, group, v)
    counts = torch.zeros(n_groups, device=v.device).index_add_(0, group, torch.ones_like(v))
    return v - (sums / counts.clamp(min=1))[group]


def task_balanced_loss(pred: torch.Tensor, labels: torch.Tensor, group: torch.Tensor,
                       task_index: torch.Tensor, n_groups: int, n_tasks: int,
                       task_weight: "torch.Tensor | None" = None) -> torch.Tensor:
    """Mean candidate squared error within a decision (on decision-CENTERED predictions and
    labels), mean over decisions within a task, then a weighted sum across tasks. Task balancing
    matters because the pooled batch mixes topologies with different decision counts and
    candidate counts: without it the largest task would define the fit.

    `task_weight` (one weight per task column, from `epoch_task_weights`) is what makes the
    balance hold over the EPOCH rather than only within a batch — see that function. Omitting it
    weights the tasks PRESENT in this batch equally, which is the per-batch-only behaviour."""
    se = (_center(pred, group, n_groups) - _center(labels, group, n_groups)) ** 2
    per_dec = torch.zeros(n_groups, device=pred.device).index_add_(0, group, se)
    counts = torch.zeros(n_groups, device=pred.device).index_add_(0, group, torch.ones_like(se))
    dec_loss = per_dec / counts.clamp(min=1)
    task_sum = torch.zeros(n_tasks, device=pred.device).index_add_(0, task_index, dec_loss)
    task_cnt = torch.zeros(n_tasks, device=pred.device).index_add_(
        0, task_index, torch.ones_like(dec_loss))
    present = task_cnt > 0
    per_task_loss = task_sum[present] / task_cnt[present]
    if task_weight is None:
        return per_task_loss.mean()
    return (task_weight[present] * per_task_loss).sum()


# --- arms ----------------------------------------------------------------------------------
class ArmModel:
    """A trained arm plus everything needed to score new decisions: the fitted normalizer (graph
    arms) and either one pooled model or one model per task (the diagnostic arm). Prediction
    order is always (decision, candidate_index)."""

    def __init__(self, arm: str, models: dict, normalizer: "FoldNormalizer | None") -> None:
        self.arm = arm
        self.spec = ARMS[arm]
        self.models = models  # {None: model} pooled, or {task_id: model}
        self.normalizer = normalizer

    def _model_for(self, task_id: str) -> nn.Module:
        return self.models[None] if None in self.models else self.models[task_id]

    @torch.no_grad()
    def predict(self, graphs: Sequence[GraphArrays], decisions: Sequence[Decision],
                indices: Sequence[int]) -> np.ndarray:
        out = np.zeros(sum(len(decisions[i].labels) for i in indices), dtype=np.float64)
        if not len(indices):
            return out
        offs: list[int] = []
        pos = 0
        for i in indices:
            offs.append(pos)
            pos += len(decisions[i].labels)
        by_model: dict[str, list[int]] = defaultdict(list)
        for j, i in enumerate(indices):
            by_model[decisions[i].task_id if None not in self.models else ""].append(j)
        for key, js in by_model.items():
            model = self._model_for(decisions[indices[js[0]]].task_id) if key else self.models[None]
            model.eval()
            sub = [indices[j] for j in js]
            dev = next(model.parameters()).device
            if self.spec["kind"] == "row":
                rows = np.concatenate([decisions[i].rows for i in sub], axis=0)
                pred = model(torch.as_tensor(rows, dtype=torch.float32, device=dev)).cpu().numpy()
            else:
                gs = [graphs[i] for i in sub]
                gs = self.normalizer.apply(gs) if self.normalizer is not None else gs
                pred = model(Batch(gs).to(dev)).cpu().numpy()
            cursor = 0
            for j in js:
                n = len(decisions[indices[j]].labels)
                out[offs[j]:offs[j] + n] = pred[cursor:cursor + n]
                cursor += n
        return out


def build_model(arm: str, *, model_seed: int, dropout: float, device: str) -> nn.Module:
    """Instantiate one arm under a fixed seed. Graph arms are the single parameterized
    `RelationalGNN` (0-hop is `layers=0`); the row arm is the deployed EXP-0037
    `ResidualScoreModel`, unchanged, so the baseline really is the incumbent architecture."""
    torch.manual_seed(model_seed)
    spec = ARMS[arm]
    if spec["kind"] == "row":
        from .value_router import ResidualScoreModel

        return ResidualScoreModel(hidden=spec["hidden"]).to(device)
    return RelationalGNN(hidden=spec["hidden"], layers=spec["layers"], dropout=dropout).to(device)


def _forward(arm_kind: str, model: nn.Module, graphs: Sequence[GraphArrays],
             rows: np.ndarray, device: str) -> torch.Tensor:
    if arm_kind == "row":
        return model(torch.as_tensor(rows, dtype=torch.float32, device=device))
    return model(Batch(graphs).to(device))


@dataclass
class TrainedArm:
    arm: str
    model: ArmModel
    history: list[dict]
    best_val_regret: float          # validation regret of the SELECTED checkpoint set
    best_epochs: dict[str, int]     # group ("pooled" or task id) -> selected epoch
    epochs_run: int


def train_arm(arm: str, dump: LoadedDump, train_idx: Sequence[int], val_idx: Sequence[int], *,
              model_seed: int, training: "dict | None" = None, device: "str | None" = None,
              log=lambda *a, **k: None) -> TrainedArm:
    """Fit one arm on `train_idx` and select its checkpoint on `val_idx`.

    Checkpoint selection is the validation decision-weighted mean regret averaged over tasks (the
    same task-balanced weighting as the loss and the same instrument as the gate) — validation r2,
    top-1, and loss are recorded but never decide. Early stopping needs `min_epochs` first and
    then `early_stop_patience` consecutive validation checks with no improvement. The normalizer
    is fitted ONCE on the pooled training graphs; the diagnostic per-task arm reuses that same
    global normalizer so the only difference between it and the pooled arm is which decisions
    the weights saw."""
    cfg = {**TRAINING, **(training or {})}
    dev = _pick_device(device)
    torch.set_num_threads(1)  # single-thread BLAS: no multithreaded reduction-order jitter
    spec = ARMS[arm]
    graphs, decisions = dump.graphs, dump.decisions
    normalizer = None
    if spec["kind"] == "graph":
        normalizer = FoldNormalizer.fit([graphs[i] for i in train_idx])
        norm_graphs = {i: g for i, g in zip(train_idx, normalizer.apply([graphs[i]
                                                                        for i in train_idx]))}
    else:
        norm_graphs = {}

    groups = ([None] if spec.get("pooled", True)
              else sorted({decisions[i].task_id for i in train_idx}))
    models: dict = {}
    history: list[dict] = []
    best_epochs: dict[str, int] = {}
    epochs_run = 0
    for key in groups:
        idx = ([i for i in train_idx if key is None or decisions[i].task_id == key])
        model = build_model(arm, model_seed=model_seed, dropout=cfg["dropout"], device=dev)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                               weight_decay=cfg["weight_decay"])
        models[key] = model
        tasks_seen = sorted({decisions[i].task_id for i in idx})
        task_col = {t: j for j, t in enumerate(tasks_seen)}
        # Equal gradient weight per task PER EPOCH, not just per batch (the tail batches of an
        # epoch belong to the largest task alone). Counts are fixed across epochs, so this is
        # computed once. `per_batch` reproduces the old within-batch-only balance.
        task_w = None
        if cfg.get("task_weighting", "epoch_balanced") == "epoch_balanced":
            weights = epoch_task_weights(decisions, idx, per_task=cfg["batch_decisions_per_task"])
            task_w = torch.tensor([weights[t] for t in tasks_seen], dtype=torch.float32,
                                  device=dev)
        best_local, best_local_state, best_local_epoch, stale = float("inf"), None, -1, 0
        for epoch in range(cfg["max_epochs"]):
            model.train()
            batches = task_balanced_batches(decisions, idx, model_seed=model_seed, epoch=epoch,
                                            per_task=cfg["batch_decisions_per_task"])
            ep_loss = 0.0
            for batch in batches:
                labels, group, task_of_group, rows = flatten(decisions, batch)
                gs = [norm_graphs[i] for i in batch] if spec["kind"] == "graph" else []
                pred = _forward(spec["kind"], model, gs, rows, dev)
                loss = task_balanced_loss(
                    pred, torch.as_tensor(labels, dtype=torch.float32, device=dev),
                    torch.as_tensor(group, device=dev),
                    torch.as_tensor([task_col[t] for t in task_of_group], device=dev),
                    len(batch), len(tasks_seen), task_weight=task_w)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
                opt.step()
                ep_loss += float(loss.item())
            mean_loss = ep_loss / max(1, len(batches))
            epochs_run = max(epochs_run, epoch + 1)
            if (epoch + 1) % cfg["validation_every"]:
                history.append({"arm": arm, "group": key or "pooled", "model_seed": model_seed,
                                "epoch": epoch, "train_loss": mean_loss, "val_regret": "",
                                "val_r2": ""})
                continue
            probe = ArmModel(arm, {key: model}, normalizer)
            vidx = [i for i in val_idx if key is None or decisions[i].task_id == key]
            vm = arm_metrics(probe, dump, vidx)
            regret = vm["mean_regret_over_tasks"]
            history.append({"arm": arm, "group": key or "pooled", "model_seed": model_seed,
                            "epoch": epoch, "train_loss": mean_loss, "val_regret": regret,
                            "val_r2": vm["r2_over_tasks"]})
            if regret < best_local - 1e-12:
                best_local, best_local_epoch, stale = regret, epoch, 0
                best_local_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch + 1 >= cfg["min_epochs"] and stale >= cfg["early_stop_patience"]:
                log(f"[{arm}/{key or 'pooled'}/m{model_seed}] early stop at epoch {epoch} "
                    f"(best val regret {best_local:.4f})")
                break
        if best_local_state is not None:  # restore the VALIDATION-selected checkpoint, not the last
            model.load_state_dict(best_local_state)
        best_epochs[str(key or "pooled")] = best_local_epoch
    trained = ArmModel(arm, models, normalizer)
    final = arm_metrics(trained, dump, list(val_idx))
    return TrainedArm(arm=arm, model=trained, history=history,
                      best_val_regret=float(final["mean_regret_over_tasks"]),
                      best_epochs=best_epochs, epochs_run=epochs_run)


# --- metrics ---------------------------------------------------------------------------------
def decision_metrics(labels: Sequence[np.ndarray], preds: Sequence[np.ndarray]) -> dict:
    """Routing metrics over REAL-CHOICE decisions (>=2 candidates and a non-flat target), each
    decision counted once — the same decision-weighted, tie-aware convention EXP-0038B and
    EXP-0039 were scored with, extended with the two extra reads EXP-0041 requires: centered MSE
    (fit quality on the within-decision contrast) and winner-set overlap (Jaccard of the
    predicted and teacher argmax sets, which credits a correct tie without crediting a lucky
    argmax tiebreak)."""
    regrets, top1, overlap, mses = [], [], [], []
    rand_regret = []
    c_res = c_tot = 0.0
    n_flat = 0
    for ls, ps in zip(labels, preds):
        ls = np.asarray(ls, dtype=np.float64)
        ps = np.asarray(ps, dtype=np.float64)
        if ls.size < 2 or float(np.ptp(ls)) <= FLAT_TOL:
            n_flat += 1
            continue
        best = float(ls.max())
        pick = int(np.argmax(ps))
        regrets.append(best - float(ls[pick]))
        top1.append(int(float(ls[pick]) == best))
        pw = set(np.flatnonzero(ps >= ps.max() - 1e-12).tolist())
        tw = set(np.flatnonzero(ls >= best - 1e-12).tolist())
        overlap.append(len(pw & tw) / len(pw | tw))
        rand_regret.append(best - float(ls.mean()))
        cl, cp = ls - ls.mean(), ps - ps.mean()
        mses.append(float(((cl - cp) ** 2).mean()))
        c_res += float(((cl - cp) ** 2).sum())
        c_tot += float((cl ** 2).sum())
    return {
        "n_decisions": len(regrets), "n_flat": n_flat,
        "mean_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "random_regret": float(np.mean(rand_regret)) if rand_regret else float("nan"),
        "top1_match": float(np.mean(top1)) if top1 else float("nan"),
        "winner_overlap": float(np.mean(overlap)) if overlap else float("nan"),
        "centered_mse": float(np.mean(mses)) if mses else float("nan"),
        "r2": 1.0 - c_res / c_tot if c_tot > 0 else float("nan"),
    }


def _grouped(dump: LoadedDump, indices: Sequence[int], pred: np.ndarray):
    """Split a flat prediction vector back into per-decision arrays (canonical order)."""
    labels, preds, pos = [], [], 0
    for i in indices:
        n = len(dump.decisions[i].labels)
        labels.append(dump.decisions[i].labels)
        preds.append(pred[pos:pos + n])
        pos += n
    return labels, preds


def arm_metrics(model: ArmModel, dump: LoadedDump, indices: Sequence[int]) -> dict:
    """Task-balanced summary used for checkpoint and arm selection: the per-task decision-weighted
    regret / r2, then an equal mean across the tasks present."""
    if not indices:
        return {"mean_regret_over_tasks": float("inf"), "r2_over_tasks": float("nan"), "by_task": {}}
    pred = model.predict(dump.graphs, dump.decisions, indices)
    labels, preds = _grouped(dump, indices, pred)
    by_task: dict[str, dict] = {}
    for task in sorted({dump.decisions[i].task_id for i in indices}):
        sel = [j for j, i in enumerate(indices) if dump.decisions[i].task_id == task]
        m = decision_metrics([labels[j] for j in sel], [preds[j] for j in sel])
        if m["n_decisions"]:
            by_task[task] = m
    regrets = [m["mean_regret"] for m in by_task.values()]
    r2s = [m["r2"] for m in by_task.values()]
    return {"mean_regret_over_tasks": float(np.mean(regrets)) if regrets else float("inf"),
            "r2_over_tasks": float(np.mean(r2s)) if r2s else float("nan"), "by_task": by_task}


def ensemble_predictions(preds_by_seed: dict) -> np.ndarray:
    """Ensemble score = the mean of the per-seed predictions. Callers center within a decision
    before ranking, so a per-seed additive offset cannot tilt the ensemble."""
    return np.mean(np.stack([preds_by_seed[s] for s in sorted(preds_by_seed)]), axis=0)


def centered_by_decision(dump: LoadedDump, indices: Sequence[int], pred: np.ndarray) -> np.ndarray:
    out = np.array(pred, dtype=np.float64, copy=True)
    pos = 0
    for i in indices:
        n = len(dump.decisions[i].labels)
        if n:
            out[pos:pos + n] -= out[pos:pos + n].mean()
        pos += n
    return out


# --- artifact-only Offline Gate A ---------------------------------------------------------
def _coverage(rows: Sequence[dict], *, arm: str, baseline_arm: str, want_tasks: Sequence[str],
              want_seeds: Sequence[int],
              seeds_by_task: "Mapping[str, Sequence[int]] | None") -> dict:
    """Is this prediction set the WHOLE registered final test, scored identically by every arm and
    model seed?

    "At least one trajectory seed per task" was the old check, which a two-of-three-episode set
    (or an arm scored on a different decision subset) passed (review finding 2). A partial test
    set is not a weaker pass, it is a different experiment, so coverage now means: exactly the
    registered tasks; exactly the registered episode seeds per task; every (arm, model seed) group
    present; all of them covering the byte-identical decision/candidate key set; and no duplicated
    prediction row."""
    keysets: dict[tuple[str, str], set] = {}
    duplicates = 0
    for r in rows:
        if r["arm"] not in (arm, baseline_arm):
            continue
        key = (r["task_id"], int(r["trajectory_seed"]), int(r["decision_index"]),
               int(r["candidate_index"]))
        ks = keysets.setdefault((r["arm"], str(r["model_seed"])), set())
        duplicates += int(key in ks)
        ks.add(key)
    expected_groups = {(a, s) for a in {arm, baseline_arm}
                       for s in [*(str(x) for x in want_seeds), "ensemble"]}
    missing_groups = sorted(expected_groups - set(keysets))
    identical = len({frozenset(v) for v in keysets.values()}) == 1 if keysets else False
    seen_tasks = {k[0] for v in keysets.values() for k in v}
    seeds_seen = {t: {k[1] for v in keysets.values() for k in v if k[0] == t} for t in seen_tasks}
    seeds_ok = (all(seeds_seen.get(t, set()) == set(seeds_by_task[t]) for t in want_tasks)
                if seeds_by_task else all(seeds_seen.get(t) for t in want_tasks))
    return {"ok": bool(seen_tasks == set(want_tasks) and seeds_ok and identical
                       and not missing_groups and not duplicates and len(want_tasks) == 3),
            "tasks": sorted(seen_tasks), "episode_seeds": {t: sorted(s)
                                                           for t, s in seeds_seen.items()},
            "missing_groups": missing_groups, "duplicate_rows": duplicates,
            "identical_decision_coverage": bool(identical),
            "expected_episode_seeds": ({t: sorted(seeds_by_task[t]) for t in seeds_by_task}
                                       if seeds_by_task else None)}


def registered_test_scope() -> tuple[tuple[str, ...], dict[str, tuple[int, ...]], tuple[int, ...]]:
    """The pre-registered Gate A scope: the three tasks, their final-test episodes, the 3 model
    seeds. This is what "the whole test set" MEANS; it is not inferred from the rows on hand."""
    return (TASK_IDS, {t: tuple(SPLIT_REGISTRY[t]["test"]) for t in TASK_IDS}, MODEL_SEEDS)


def scope_mismatch(tasks: Sequence[str], model_seeds: Sequence[int],
                   seeds_by_task: "Mapping[str, Sequence[int]] | None",
                   want: "Mapping[str, Sequence[int]]") -> "str | None":
    """Is the shape a gate is being asked to certify the pre-registered one? Returns the first
    deviation as a sentence, or None."""
    if sorted(tasks) != sorted(want):
        return f"gate scope tasks {sorted(tasks)} != registered {sorted(want)}"
    if sorted(int(s) for s in model_seeds) != sorted(MODEL_SEEDS):
        return f"gate scope model seeds {sorted(model_seeds)} != registered {list(MODEL_SEEDS)}"
    got = {t: sorted(int(s) for s in (seeds_by_task or {}).get(t, ())) for t in want}
    exp = {t: sorted(int(s) for s in want[t]) for t in want}
    if got != exp:
        return f"gate scope episodes {got} != registered {exp}"
    return None


def gate_a(pred_rows: Sequence[dict], *, arm: str, baseline_arm: str = "row_mlp_h32",
           tasks: "Sequence[str] | None" = None,
           model_seeds: "Sequence[int] | None" = None,
           seeds_by_task: "Mapping[str, Sequence[int]] | None" = None,
           require_registered: bool = True) -> dict:
    """Compute Offline Gate A from prediction ROWS alone — no checkpoint, no simulator, no model
    object (HANDOVER: the offline verdict must be recomputable from CSV).

    Conditions (all on final-test episodes): (1) centered r2 > 0 on >= 2/3 tasks; (2) regret at
    least 10% below the uniform-random floor on >= 2/3 tasks; (3) regret beats the fairly
    held-out row baseline on >= 2/3 tasks; (4) no task worse than the row baseline by more than
    5%; (5) conditions 1 and 2 hold for the ensemble mean AND independently for every model seed;
    (6) coverage is EXACT — the registered tasks, the registered episode seeds per task, every
    (arm, model seed) group present over an identical decision/candidate key set, no duplicated
    row — and every prediction is finite.

    `require_registered` (default True) is what makes condition 6 mean anything: without it, an
    omitted `tasks`/`seeds_by_task` was filled in FROM THE ROWS, so a one-episode-per-task
    prediction file certified itself as complete (review finding 1). The registered scope is now
    the default, and a caller that passes a different one is told so instead of being believed. A
    smoke passes `require_registered=False`; nothing that produces a verdict may."""
    rows = [r for r in pred_rows if r["split"] == "test"]
    if require_registered:
        reg_tasks, reg_seeds, reg_model_seeds = registered_test_scope()
        tasks = reg_tasks if tasks is None else tasks
        model_seeds = reg_model_seeds if model_seeds is None else model_seeds
        seeds_by_task = reg_seeds if seeds_by_task is None else seeds_by_task
        deviation = scope_mismatch(tasks, model_seeds, seeds_by_task, reg_seeds)
        if deviation:
            return {"pass": False, "conditions": {"registered_scope": False}, "arm": arm,
                    "reason": deviation, "coverage": {"ok": False, "reason": deviation}}
    if not rows:
        return {"pass": False, "conditions": {"coverage": False},
                "reason": "no final-test prediction rows"}
    want_tasks = sorted(set(tasks) if tasks else {r["task_id"] for r in rows})
    want_seeds = sorted(set(model_seeds) if model_seeds else
                        {int(r["model_seed"]) for r in rows if r["model_seed"] != "ensemble"})
    coverage = _coverage(rows, arm=arm, baseline_arm=baseline_arm, want_tasks=want_tasks,
                         want_seeds=want_seeds, seeds_by_task=seeds_by_task)
    if coverage["duplicate_rows"]:
        # a duplicated candidate row would also make the per-decision metrics ill-defined; the
        # gate must still RETURN a verdict (it is recomputed from CSV by readers), not explode
        return {"pass": False, "conditions": {"coverage_and_finite": False}, "arm": arm,
                "reason": f"{coverage['duplicate_rows']} duplicate prediction rows",
                "coverage": coverage}

    def _by_decision(arm_name: str, seed_key: str) -> dict:
        out: dict[str, dict[tuple, list[tuple[int, float, float]]]] = defaultdict(
            lambda: defaultdict(list))
        for r in rows:
            if r["arm"] != arm_name or str(r["model_seed"]) != seed_key:
                continue
            out[r["task_id"]][(r["task_id"], int(r["trajectory_seed"]),
                               int(r["decision_index"]))].append(
                (int(r["candidate_index"]), float(r["pred"]), float(r["primary_label"])))
        return out

    def _metrics(arm_name: str, seed_key: str) -> dict:
        per_task = {}
        for task, decs in _by_decision(arm_name, seed_key).items():
            labels, preds = [], []
            for cand in decs.values():
                cand = sorted(cand)
                if [c[0] for c in cand] != list(range(len(cand))):
                    raise ValueError(f"{arm_name}/{seed_key}/{task}: candidate_index not 0..k-1")
                labels.append(np.array([c[2] for c in cand]))
                preds.append(np.array([c[1] for c in cand]))
            per_task[task] = decision_metrics(labels, preds)
        return per_task

    ens = _metrics(arm, "ensemble")
    base = _metrics(baseline_arm, "ensemble")
    finite = all(np.isfinite(float(r["pred"])) for r in rows)
    covered = bool(coverage["ok"] and set(ens) >= set(want_tasks) and set(base) >= set(want_tasks))

    def _pair(per_task: dict) -> tuple[dict, dict]:
        return ({t: per_task[t]["r2"] for t in want_tasks if t in per_task},
                {t: per_task[t]["mean_regret"] for t in want_tasks if t in per_task})

    r2, regret = _pair(ens)
    _, base_regret = _pair(base)
    rand = {t: ens[t]["random_regret"] for t in want_tasks if t in ens}
    c1 = sum(1 for t in want_tasks if r2.get(t, -np.inf) > 0) >= 2
    c2 = sum(1 for t in want_tasks
             if regret.get(t, np.inf) <= 0.90 * rand.get(t, 0.0)) >= 2
    c3 = sum(1 for t in want_tasks
             if regret.get(t, np.inf) < base_regret.get(t, -np.inf)) >= 2
    c4 = all(regret.get(t, np.inf) <= base_regret.get(t, 0.0) * 1.05 for t in want_tasks)
    per_seed = {}
    for s in want_seeds:
        m = _metrics(arm, str(s))
        sr2 = {t: m[t]["r2"] for t in want_tasks if t in m}
        sreg = {t: m[t]["mean_regret"] for t in want_tasks if t in m}
        srand = {t: m[t]["random_regret"] for t in want_tasks if t in m}
        per_seed[s] = {
            "r2_pos_2of3": sum(1 for t in want_tasks if sr2.get(t, -np.inf) > 0) >= 2,
            "beats_random_10pct_2of3": sum(
                1 for t in want_tasks if sreg.get(t, np.inf) <= 0.90 * srand.get(t, 0.0)) >= 2,
            "r2": sr2, "mean_regret": sreg}
    c5 = bool(per_seed) and all(v["r2_pos_2of3"] and v["beats_random_10pct_2of3"]
                                for v in per_seed.values())
    c6 = bool(finite and covered and len(want_tasks) == 3)
    conditions = {"r2_pos_2of3": bool(c1), "beats_random_10pct_2of3": bool(c2),
                  "beats_row_baseline_2of3": bool(c3), "no_task_worse_than_row_5pct": bool(c4),
                  "holds_for_every_model_seed": bool(c5), "coverage_and_finite": bool(c6)}
    return {"pass": all(conditions.values()), "conditions": conditions, "arm": arm,
            "per_task": {t: {"r2": r2.get(t), "mean_regret": regret.get(t),
                             "random_regret": rand.get(t), "row_regret": base_regret.get(t)}
                         for t in want_tasks},
            "per_model_seed": per_seed, "coverage": coverage}


# --- artifact writing -----------------------------------------------------------------------
PRED_FIELDS = ["arm", "model_seed", "split", "task_id", "trajectory_seed", "decision_index",
               "candidate_index", "producer_id", "pred", "primary_label", "flat"]
METRIC_FIELDS = ["arm", "model_seed", "split", "task_id", "trajectory_seed", "n_decisions",
                 "n_flat", "mean_regret", "random_regret", "row_regret", "top1_match",
                 "winner_overlap", "centered_mse", "r2", "pred_dispersion"]


def prediction_rows(dump: LoadedDump, indices: Sequence[int], pred: np.ndarray, *, arm: str,
                    model_seed: str) -> list[dict]:
    rows, pos = [], 0
    for i in indices:
        d = dump.decisions[i]
        for ci, pid in enumerate(d.producer_ids):
            rows.append({"arm": arm, "model_seed": model_seed, "split": d.split,
                         "task_id": d.task_id, "trajectory_seed": d.seed,
                         "decision_index": d.index, "candidate_index": ci, "producer_id": pid,
                         "pred": repr(float(pred[pos + ci])),
                         "primary_label": repr(float(d.labels[ci])), "flat": int(d.flat)})
        pos += len(d.labels)
    return rows


def write_csv(path: str, rows: Sequence[dict], fields: "Sequence[str] | None" = None) -> None:
    if not rows and not fields:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields or rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha256_of(path: str) -> str:
    from ..eval.graph_value_router_dump import sha256_file

    return sha256_file(path)


def lock_manifest(paths: Sequence[str], out_path: str, *, base_dir: str,
                  extra: "dict | None" = None,
                  roots: "Mapping[str, tuple[str, Sequence[str]]] | None" = None) -> dict:
    """Freeze the model/normalizer/arm-selection/adapter set by content hash (HANDOVER step 4).

    Artifacts live in two directories (checkpoints in the fit dir, `adapter.json` in the rollout
    dir), so the lock is keyed by ROOT: `{"fit": {rel: sha}, "adapter": {rel: sha}}`. The earlier
    shape put the adapter hash in a free-form `extra` field that `verify_manifest` never looked
    at, which made the adapter editable after the lock (review finding 1) — every hash a manifest
    records is now verified, and a root the verifier has no directory for is a hard error rather
    than a silent skip."""
    groups: dict[str, tuple[str, Sequence[str]]] = {"fit": (base_dir, list(paths)),
                                                    **dict(roots or {})}
    hashes = {name: {os.path.relpath(p, d): sha256_of(p) for p in sorted(ps)}
              for name, (d, ps) in groups.items()}
    manifest = {"base_dir": os.path.basename(base_dir.rstrip("/")),
                "roots": hashes, "sha256": hashes["fit"], **(extra or {})}
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_manifest(manifest_path: str, *, base_dir: str,
                    roots: "Mapping[str, str] | None" = None) -> dict:
    """Recompute EVERY hash the manifest records. Returns the manifest so a caller can bind to
    its `arm` field. A recorded root with no directory supplied raises: an unverifiable entry
    must never read as a pass."""
    manifest = json.load(open(manifest_path))
    dirs = {"fit": base_dir, **dict(roots or {})}
    recorded = manifest.get("roots") or {"fit": manifest.get("sha256", {})}
    for name, entries in recorded.items():
        root = dirs.get(name)
        if root is None:
            raise ValueError(f"manifest locks root {name!r} but no directory was supplied to "
                             "verify it; refusing to treat an unchecked artifact set as frozen")
        for rel, want in entries.items():
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                raise ValueError(f"locked artifact {name}/{rel} is missing")
            got = sha256_of(path)
            if got != want:
                raise ValueError(f"locked artifact {name}/{rel} sha256 {got} != {want}: the "
                                 "frozen model/adapter set changed after selection")
    return manifest


# --- fit stage --------------------------------------------------------------------------------
def run_fit(dump_dir: str, out_dir: str, *, arms: Sequence[str] = DEFAULT_ARMS,
            model_seeds: Sequence[int] = MODEL_SEEDS, training: "dict | None" = None,
            train_size: int = 8, curve_sizes: Sequence[int] = NESTED_TRAIN_SIZES,
            expected_target_mode: "str | None" = None, device: "str | None" = None,
            source_commit: "str | None" = None, require_shape: bool = True,
            dump_tranche: "Sequence[str] | None" = None, log=print) -> dict:
    """Stage 1+2: fit every arm on the primary train split, select checkpoints and the graph arm
    from VALIDATION only, and persist predictions/metrics/checkpoints. Final-test rows are not
    even loaded here (`splits=("train","validation")`), which is the mechanical guarantee that no
    test row can enter a normalizer, an early-stopping decision, or the arm choice.

    `require_shape` (default True, as in the EXP-0039 probe) certifies that the dump really is the
    pre-registered dataset — frozen config, registered tasks, exact collected seeds, pinned
    inputs, committed source, EXP-0040 license — AND that this fit is the pre-registered run:
    the registered arms, the three model seeds, train-8, the 2/4/8 curve, an unmodified training
    recipe, and a committed revision that matches the dump's (review finding 2). The dump's source
    tree and runtime are compared against the live ones exactly as EXP-0039 does, so a fit cannot
    score a dump built by other code. A smoke passes `require_shape=False`; nothing else may.

    In `continuation_mean` mode the pooled single-suffix CONTROL arm is fitted automatically on
    the same train-8 graph rows, since the handover requires it and an optional control is one
    that quietly goes missing."""
    from ..eval.graph_value_router_dump import environment_manifest, source_fingerprint
    from .graph_router_probe import assert_env_matches, assert_source_matches

    if require_shape:
        certify_fit_contract(arms=arms, model_seeds=model_seeds, train_size=train_size,
                             curve_sizes=curve_sizes, training=training,
                             source_commit=source_commit)
    # `splits` restricts what is READ; `dump_tranche` declares what the dump on disk should
    # CONTAIN. Under staged collection (protocol v2) the fit runs against a train+validation
    # tranche and the final-test labels do not exist yet; a complete dump is still accepted.
    dump = load_generalization_dump(dump_dir, splits=("train", "validation"),
                                    expected_target_mode=expected_target_mode,
                                    certify=require_shape, expected_splits=dump_tranche)
    if require_shape:  # cross-stage: the dump's code and runtime are the ones now fitting it
        assert_source_matches(dump.spec.get("source"), expected_commit=source_commit)
        assert_env_matches(dump.spec.get("env"))
    registry = dump.manifest.get("registry", {})
    seeds_by_task = {t: tuple(registry[t]["train"])[:train_size] for t in registry} or None
    train_idx = select(dump.decisions, split="train", seeds_by_task=seeds_by_task, drop_flat=True)
    # reporting uses the FULL train split (flat decisions included) so the emitted train rows carry
    # the flat counts and flat-candidate predictions the handover asks for; training itself still
    # sees only the non-flat decisions.
    train_all = select(dump.decisions, split="train", seeds_by_task=seeds_by_task)
    val_idx = select(dump.decisions, split="validation")
    val_real = select(dump.decisions, split="validation", drop_flat=True)
    if not train_idx or not val_real:
        raise ValueError("fit needs non-flat training and validation decisions")
    arms = list(arms)
    if dump.spec.get("target_mode") == "continuation_mean" and CONTROL_ARM not in arms:
        arms.append(CONTROL_ARM)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    pred_rows: list[dict] = []
    metric_rows: list[dict] = []
    history_rows: list[dict] = []
    curve_rows: list[dict] = []
    norm_dump: dict = {}
    trained: dict[str, dict[int, TrainedArm]] = {}
    ckpt_paths: list[str] = []

    for arm in arms:
        # the control arm is scored against the SINGLE-SUFFIX label everywhere (fit, checkpoint
        # selection, metrics); every other arm sees the licensed primary target
        view = control_view(dump) if ARMS[arm].get("target") == "original_suffix" else dump
        fit_idx = (select(view.decisions, split="train", seeds_by_task=seeds_by_task,
                          drop_flat=True) if view is not dump else train_idx)
        sel_idx = (select(view.decisions, split="validation", drop_flat=True)
                   if view is not dump else val_real)
        trained[arm] = {}
        for ms in model_seeds:
            t = train_arm(arm, view, fit_idx, sel_idx, model_seed=ms, training=training,
                          device=device, log=log)
            trained[arm][ms] = t
            history_rows.extend(t.history)
            for key, model in t.model.models.items():
                path = os.path.join(ckpt_dir, f"{arm}_m{ms}_{key or 'pooled'}.pt")
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
                ckpt_paths.append(path)
            if t.model.normalizer is not None and arm not in norm_dump:
                norm_dump[arm] = t.model.normalizer.groups()
            log(f"[fit] {arm} seed {ms}: val regret {t.best_val_regret:.4f} "
                f"({t.epochs_run} epochs)")
        # per-seed + ensemble predictions/metrics on the FULL train and validation splits
        for split, idx in (("train", train_all), ("validation", val_idx)):
            preds = {ms: trained[arm][ms].model.predict(view.graphs, view.decisions, idx)
                     for ms in model_seeds}
            centered = {ms: centered_by_decision(view, idx, preds[ms]) for ms in model_seeds}
            ens = ensemble_predictions(centered)
            disp = float(np.mean(np.std(np.stack([centered[ms] for ms in model_seeds]), axis=0)))
            for ms in model_seeds:
                pred_rows.extend(prediction_rows(view, idx, preds[ms], arm=arm,
                                                 model_seed=str(ms)))
                metric_rows.extend(_metric_rows(view, idx, preds[ms], arm=arm, model_seed=str(ms),
                                                split=split, dispersion=""))
            pred_rows.extend(prediction_rows(view, idx, ens, arm=arm, model_seed="ensemble"))
            metric_rows.extend(_metric_rows(view, idx, ens, arm=arm, model_seed="ensemble",
                                            split=split, dispersion=disp))

    # nested 2 -> 4 -> 8 learning curve for the current architecture. REQUIRED by protocol v2 and
    # still strictly diagnostic: it is written to its own CSV, is never passed to
    # `select_graph_arm`, and no gate reads it. Every size gets its OWN train-fold normalizer
    # (fitted inside `train_arm` on that size's training graphs) and the SAME validation split, so
    # the only thing that varies along the curve is the number of independent train episodes.
    if CURVE_ARM in arms and registry:
        for size in curve_sizes:
            if size == train_size:  # reuse the primary fit rather than refitting it
                models = {ms: trained[CURVE_ARM][ms].model for ms in model_seeds}
            else:
                sub = {t: tuple(registry[t]["train"])[:size] for t in registry}
                sub_idx = select(dump.decisions, split="train", seeds_by_task=sub, drop_flat=True)
                models = {}
                for ms in model_seeds:
                    t_ = train_arm(CURVE_ARM, dump, sub_idx, val_real, model_seed=ms,
                                   training=training, device=device, log=log)
                    models[ms] = t_.model
                    log(f"[curve] {CURVE_ARM} train-{size} seed {ms}: "
                        f"val regret {t_.best_val_regret:.4f}")
            # dispersion across model seeds, per task: how much the three seeds disagree at this
            # much data. A curve that improves only by seeds converging is a different story from
            # one that improves by every seed getting better.
            centered = {ms: centered_by_decision(dump, val_real,
                                                 models[ms].predict(dump.graphs, dump.decisions,
                                                                    val_real))
                        for ms in model_seeds}
            disp = _per_task_dispersion(dump, val_real, centered)
            for ms in model_seeds:
                curve_rows.extend(_curve_rows(CURVE_ARM, size, ms,
                                              arm_metrics(models[ms], dump, val_real),
                                              dispersion=disp))

    _fill_row_baseline(metric_rows)
    selection = select_graph_arm(metric_rows, arms=[a for a in arms if a in SELECTABLE_ARMS])
    write_csv(os.path.join(out_dir, "predictions.csv"), pred_rows, PRED_FIELDS)
    write_csv(os.path.join(out_dir, "metrics_by_split_task_seed.csv"), metric_rows, METRIC_FIELDS)
    write_csv(os.path.join(out_dir, "loss_history.csv"), history_rows,
              ["arm", "group", "model_seed", "epoch", "train_loss", "val_regret", "val_r2"])
    write_csv(os.path.join(out_dir, "learning_curve.csv"), curve_rows,
              ["arm", "n_train_episodes", "model_seed", "task_id", "mean_regret", "r2",
               "random_regret", "regret_over_random", "dispersion"])
    with open(os.path.join(out_dir, "architecture_selection.json"), "w") as f:
        json.dump(selection, f, indent=2)
    with open(os.path.join(out_dir, "normalization.json"), "w") as f:
        json.dump(norm_dump, f, indent=2)
    spec = {
        "experiment": "EXP-0041", "stage": "fit", "dump_spec": dump.spec,
        "arms": {a: ARMS[a] for a in arms}, "model_seeds": list(model_seeds),
        "training": {**TRAINING, **(training or {})}, "train_size": train_size,
        "curve_sizes": list(curve_sizes),
        # what the dump on disk contained when this fit certified it: under staged collection the
        # final-test labels did not exist yet, which Gate A re-checks before pairing a tranche
        "dump_tranche": list(dump_tranche) if dump_tranche else None,
        "selection": selection, "schema_hash": schema_hash(),
        "require_shape": bool(require_shape),
        "control_arm": CONTROL_ARM if CONTROL_ARM in arms else None,
        "best_epochs": {a: {ms: trained[a][ms].best_epochs for ms in model_seeds} for a in arms},
        "generalization_gap": generalization_gaps(metric_rows),
        "n_train_decisions": len(train_idx), "n_train_reported": len(train_all),
        "n_validation_decisions": len(val_idx),
        "n_validation_real_choice": len(val_real),
        "checkpoints": sorted(os.path.relpath(p, out_dir) for p in ckpt_paths),
        "source": source_fingerprint(source_commit), "env": environment_manifest(),
        "device": _pick_device(device),
    }
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2, default=str)
    log(f"[fit] selected graph arm: {selection.get('selected')} ({selection.get('reason')})")
    return spec


def _metric_rows(dump: LoadedDump, indices: Sequence[int], pred: np.ndarray, *, arm: str,
                 model_seed: str, split: str, dispersion) -> list[dict]:
    """Per (task, trajectory seed) rows plus a per-task aggregate row (trajectory_seed='all')."""
    labels, preds = _grouped(dump, indices, pred)
    rows: list[dict] = []
    tasks = sorted({dump.decisions[i].task_id for i in indices})
    for task in tasks:
        seeds = sorted({dump.decisions[i].seed for i in indices
                        if dump.decisions[i].task_id == task})
        for seed in [*seeds, "all"]:
            sel = [j for j, i in enumerate(indices)
                   if dump.decisions[i].task_id == task
                   and (seed == "all" or dump.decisions[i].seed == seed)]
            m = decision_metrics([labels[j] for j in sel], [preds[j] for j in sel])
            rows.append({"arm": arm, "model_seed": model_seed, "split": split, "task_id": task,
                         "trajectory_seed": seed, "row_regret": "", "pred_dispersion": dispersion,
                         **{k: m[k] for k in ("n_decisions", "n_flat", "mean_regret",
                                              "random_regret", "top1_match", "winner_overlap",
                                              "centered_mse", "r2")}})
    return rows


def _fill_row_baseline(metric_rows: list[dict], baseline_arm: str = "row_mlp_h32") -> None:
    """Stamp the fair row-baseline regret onto every metric row with the same
    (split, task, trajectory seed, model seed), so each arm's row is directly comparable to the
    incumbent architecture without a join (HANDOVER §Required Offline Metrics)."""
    base = {(r["split"], r["task_id"], str(r["trajectory_seed"]), str(r["model_seed"])):
            r["mean_regret"] for r in metric_rows if r["arm"] == baseline_arm}
    for r in metric_rows:
        if ARMS.get(r["arm"], {}).get("target"):
            r["row_regret"] = ""  # scored against a different target: not comparable to the floor
            continue
        r["row_regret"] = base.get(
            (r["split"], r["task_id"], str(r["trajectory_seed"]), str(r["model_seed"])), "")


def generalization_gaps(metric_rows: Sequence[dict], *, model_seed: str = "ensemble") -> dict:
    """Per arm and task: validation-minus-train regret on the ensemble prediction. A large
    positive gap is the EXP-0039 failure signature (fits the training episodes, does not
    transfer); a small gap with a high regret means the target is hard everywhere."""
    idx = {(r["arm"], r["split"], r["task_id"]): float(r["mean_regret"]) for r in metric_rows
           if str(r["model_seed"]) == model_seed and str(r["trajectory_seed"]) == "all"}
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (arm, split, task), value in idx.items():
        if split != "train":
            continue
        for other in ("validation", "test"):
            if (arm, other, task) in idx:
                out[arm][f"{task}|{other}_minus_train"] = idx[(arm, other, task)] - value
    return dict(out)


def _per_task_dispersion(dump: LoadedDump, indices: Sequence[int],
                         centered: "Mapping[int, np.ndarray]") -> dict[str, float]:
    """Mean per-candidate std across model seeds, grouped by task, over decision-CENTERED
    predictions (the same quantity the arm rows report, split per task so a curve row can carry
    it). Centering matters: only within-decision contrasts are ever deployed, so raw-level
    disagreement between seeds is not disagreement about a routing choice."""
    task_of_candidate: list[str] = []
    for i in indices:
        task_of_candidate.extend([dump.decisions[i].task_id] * len(dump.decisions[i].labels))
    tasks = np.asarray(task_of_candidate)
    stacked = np.stack([centered[ms] for ms in sorted(centered)])
    spread = np.std(stacked, axis=0)
    return {t: float(np.mean(spread[tasks == t])) for t in sorted(set(task_of_candidate))}


def _curve_rows(arm: str, size: int, model_seed: int, metrics: dict,
                dispersion: "Mapping[str, float] | None" = None) -> list[dict]:
    """One row per (curve size, model seed, task). `regret_over_random` rides along rather than
    being left to the reader: the tasks have different random floors (EXP-0039 measured 5.24 /
    5.29 / 6.23), so raw regret is not comparable across them and a curve read on raw regret can
    invert."""
    out = []
    for task, m in metrics["by_task"].items():
        rand = m["random_regret"]
        out.append({"arm": arm, "n_train_episodes": size, "model_seed": model_seed,
                    "task_id": task, "mean_regret": m["mean_regret"], "r2": m["r2"],
                    "random_regret": rand,
                    "regret_over_random": (m["mean_regret"] / rand) if rand else "",
                    "dispersion": (dispersion or {}).get(task, "")})
    return out


def select_graph_arm(metric_rows: Sequence[dict], *, arms: Sequence[str] = SELECTABLE_ARMS) -> dict:
    """Pick the graph arm from VALIDATION ensemble metrics with the pre-registered rule:
    (1) drop any arm without positive centered r2 on >= 2 of 3 validation tasks; (2) keep the
    lowest mean relative regret versus the validation random floor; (3) break a <1% tie toward
    the SMALLER arm; (4) if nothing survives step 1, offline Gate A fails without reading a
    single test label."""
    order = {a: i for i, a in enumerate(SELECTABLE_ARMS)}  # smaller arm first
    summary: dict[str, dict] = {}
    for arm in arms:
        rows = [r for r in metric_rows if r["arm"] == arm and r["split"] == "validation"
                and r["model_seed"] == "ensemble" and r["trajectory_seed"] == "all"]
        if not rows:
            continue
        r2_pos = sum(1 for r in rows if float(r["r2"]) > 0)
        rel = [float(r["mean_regret"]) / float(r["random_regret"])
               for r in rows if float(r["random_regret"]) > 0]
        summary[arm] = {"tasks": len(rows), "r2_positive_tasks": r2_pos,
                        "mean_relative_regret": float(np.mean(rel)) if rel else float("inf"),
                        "eligible": r2_pos >= 2}
    eligible = {a: v for a, v in summary.items() if v["eligible"]}
    if not eligible:
        return {"selected": None, "reason": "no graph arm has positive r2 on >=2 validation tasks",
                "summary": summary}
    best = min(eligible.values(), key=lambda v: v["mean_relative_regret"])["mean_relative_regret"]
    close = [a for a, v in eligible.items() if v["mean_relative_regret"] <= best * 1.01]
    chosen = min(close, key=lambda a: (order.get(a, 99), a))
    return {"selected": chosen,
            "reason": (f"lowest mean relative regret {eligible[chosen]['mean_relative_regret']:.4f}"
                       + (f"; tie<=1% resolved to the smaller arm among {sorted(close)}"
                          if len(close) > 1 else "")),
            "summary": summary}


# --- gate-a stage --------------------------------------------------------------------------
def locked_arm(fit_dir: str, manifest: dict) -> "str | None":
    """The graph arm that validation selected, read from the LOCKED `architecture_selection.json`
    and cross-checked against the manifest's own `arm` field.

    Gate A used to take the arm from the fit dir's unlocked `spec.json`, which meant the one
    decision the whole test read is about could be edited after the freeze (review finding 1).
    `architecture_selection.json` is inside the content-hash lock, so it cannot."""
    selection = json.load(open(os.path.join(fit_dir, "architecture_selection.json")))
    arm = selection.get("selected")
    if manifest.get("arm") not in (None, arm):
        raise ValueError(f"manifest arm {manifest.get('arm')!r} != locked selection {arm!r}")
    return arm


def run_gate_a(dump_dir: str, fit_dir: str, out_dir: str, *, manifest_path: str,
               arm: "str | None" = None, model_seeds: Sequence[int] = MODEL_SEEDS,
               expected_target_mode: "str | None" = None, device: "str | None" = None,
               require_shape: bool = True, adapter_dir: "str | None" = None,
               test_dump_dir: "str | None" = None, log=print) -> dict:
    """Stage 5: the FIRST read of final-test labels. Requires the locked manifest to verify, so
    the checkpoints, arm selection, and adapter scored here are exactly the ones validation froze.

    Predictions are retained for EVERY fitted arm (the handover asks for candidate-level
    predictions per arm and split); only the locked selected arm is judged. The emitted
    `gate_a.json` binds the verdict to the dump, the manifest, the arm, and the SHA-256 of the
    prediction rows it was computed from, so a later stage can recompute it instead of trusting
    a `pass` field."""
    manifest = verify_manifest(manifest_path, base_dir=fit_dir,
                               roots={"adapter": adapter_dir} if adapter_dir else None)
    fit_spec = json.load(open(os.path.join(fit_dir, "spec.json")))
    selected = locked_arm(fit_dir, manifest)
    if arm is not None and selected is not None and arm != selected:
        raise ValueError(f"--arm {arm!r} != the locked validation selection {selected!r}; the "
                         "test read may only score the arm validation chose")
    arm = selected or arm
    if arm is None:
        raise ValueError("no graph arm was selected on validation; Gate A fails without a test read")
    # Staged collection (protocol v2): the final-test labels may live in their own tranche,
    # collected only after the arm and adapter were locked. Gate A then reads test labels from
    # there and cross-certifies the two tranches belong to one experiment — same frozen contract,
    # pinned inputs, licence, target/K and registry — since two directories collected weeks apart
    # could otherwise score a model against another experiment's labels.
    from ..eval.graph_generalization_dump import TEST_TRANCHE, assert_tranches_agree

    staged = test_dump_dir is not None and os.path.abspath(test_dump_dir) != os.path.abspath(
        dump_dir)
    dump = load_generalization_dump(test_dump_dir or dump_dir,
                                    expected_target_mode=expected_target_mode,
                                    certify=require_shape,
                                    expected_splits=TEST_TRANCHE if staged else None)
    if staged and require_shape:
        assert_tranches_agree(fit_spec.get("dump_spec") or {}, dump.spec)
    if require_shape:
        from .graph_router_probe import assert_env_matches, assert_source_matches

        certify_fit_spec(fit_spec)
        # the test read must happen under the SAME code and runtime that produced the dump and fit
        assert_source_matches(dump.spec.get("source"),
                              expected_commit=(fit_spec.get("source") or {}).get("commit"))
        assert_env_matches(dump.spec.get("env"))
    test_idx = select(dump.decisions, split="test")
    if not test_idx:
        raise ValueError("dump has no final-test decisions")
    tasks = sorted({d.task_id for d in dump.decisions})
    registry = dump.manifest.get("registry", {})
    seeds_by_task = {t: tuple(registry[t]["test"]) for t in registry} or None

    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict] = []
    metric_rows: list[dict] = []
    fitted = [a for a in fit_spec.get("arms", {}) if a in ARMS]
    for name in sorted({arm, "row_mlp_h32", *fitted}):
        if not _has_checkpoints(fit_dir, name, model_seeds, tasks):
            continue
        view = control_view(dump) if ARMS[name].get("target") == "original_suffix" else dump
        preds = {}
        for ms in model_seeds:
            model = load_arm(name, fit_dir, model_seed=ms, tasks=tasks, device=device)
            preds[ms] = model.predict(view.graphs, view.decisions, test_idx)
            rows.extend(prediction_rows(view, test_idx, preds[ms], arm=name, model_seed=str(ms)))
            metric_rows.extend(_metric_rows(view, test_idx, preds[ms], arm=name,
                                            model_seed=str(ms), split="test", dispersion=""))
        centered = {ms: centered_by_decision(view, test_idx, preds[ms]) for ms in model_seeds}
        ens = ensemble_predictions(centered)
        disp = float(np.mean(np.std(np.stack([centered[ms] for ms in model_seeds]), axis=0)))
        rows.extend(prediction_rows(view, test_idx, ens, arm=name, model_seed="ensemble"))
        metric_rows.extend(_metric_rows(view, test_idx, ens, arm=name, model_seed="ensemble",
                                        split="test", dispersion=disp))

    _fill_row_baseline(metric_rows)
    verdict = gate_a(rows, arm=arm, tasks=tasks, model_seeds=model_seeds,
                     seeds_by_task=seeds_by_task, require_registered=require_shape)
    pred_path = os.path.join(out_dir, "test_predictions.csv")
    write_csv(pred_path, rows, PRED_FIELDS)
    write_csv(os.path.join(out_dir, "test_metrics_by_split_task_seed.csv"), metric_rows,
              METRIC_FIELDS)
    # train->test gaps need both halves: the frozen fit's train rows plus this test read
    fit_metrics = _read_csv(os.path.join(fit_dir, "metrics_by_split_task_seed.csv"))
    with open(os.path.join(out_dir, "gate_a.json"), "w") as f:
        json.dump({
            "arm": arm, "manifest": os.path.abspath(manifest_path),
            "manifest_sha256": sha256_of(manifest_path),
            "predictions": os.path.basename(pred_path),
            "predictions_sha256": sha256_of(pred_path),
            "dump": {"target_mode": dump.spec.get("target_mode"),
                     "schema_hash": dump.spec.get("schema_hash"),
                     "source": dump.spec.get("source"), "n_decisions": dump.spec.get("n_decisions")},
            "tasks": tasks, "model_seeds": list(model_seeds), "require_shape": bool(require_shape),
            "test_seeds_by_task": {t: list(v) for t, v in (seeds_by_task or {}).items()},
            "generalization_gap": generalization_gaps([*fit_metrics, *metric_rows]),
            "verdict": verdict}, f, indent=2, default=str)
    log(f"[gate-a] arm={arm} pass={verdict['pass']} conditions={verdict['conditions']}")
    return verdict


def _has_checkpoints(fit_dir: str, arm: str, model_seeds: Sequence[int],
                     tasks: Sequence[str]) -> bool:
    keys = ["pooled"] if ARMS[arm].get("pooled", True) else list(tasks)
    return all(os.path.exists(os.path.join(fit_dir, "checkpoints", f"{arm}_m{ms}_{k}.pt"))
               for ms in model_seeds for k in keys)


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_arm(arm: str, fit_dir: str, *, model_seed: int, tasks: Sequence[str],
             device: "str | None" = None) -> ArmModel:
    """Rebuild a trained arm from its frozen checkpoint(s) + the persisted normalizer stats."""
    dev = _pick_device(device)
    ckpt_dir = os.path.join(fit_dir, "checkpoints")
    spec = ARMS[arm]
    keys = [None] if spec.get("pooled", True) else list(tasks)
    models = {}
    for key in keys:
        model = build_model(arm, model_seed=model_seed, dropout=0.0, device=dev)
        state = torch.load(os.path.join(ckpt_dir, f"{arm}_m{model_seed}_{key or 'pooled'}.pt"),
                           map_location=dev)
        model.load_state_dict(state)
        model.eval()
        models[key] = model
    normalizer = None
    if spec["kind"] == "graph":
        groups = json.load(open(os.path.join(fit_dir, "normalization.json")))[arm]
        normalizer = normalizer_from_groups(groups)
    return ArmModel(arm, models, normalizer)


def normalizer_from_groups(groups: dict) -> FoldNormalizer:
    """Rebuild a `FoldNormalizer` from its persisted audit view, so a deployed/reloaded model
    standardizes with the EXACT training statistics (never a refit on the data being scored)."""
    def _stats(d: dict) -> dict:
        return {k: (np.asarray(v["cols"], dtype=np.int64),
                    np.asarray(v["mean"], dtype=np.float32),
                    np.asarray(v["std"], dtype=np.float32)) for k, v in d.items()}

    return FoldNormalizer(_stats(groups.get("node", {})), _stats(groups.get("edge", {})))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("fit", "gate-a"), required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fit-dir", default=None, help="gate-a: the frozen fit stage dir")
    ap.add_argument("--manifest", default=None, help="gate-a: locked model/adapter manifest")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--adapter-dir", default=None,
                    help="gate-a: dir holding the locked adapter.json (verified with the manifest)")
    ap.add_argument("--allow-uncertified-dump", action="store_true",
                    help="skip the pre-registered dump contract check (smoke only)")
    ap.add_argument("--model-seeds", default=",".join(str(s) for s in MODEL_SEEDS))
    ap.add_argument("--train-size", type=int, default=8)
    ap.add_argument("--curve-sizes", default=",".join(str(s) for s in NESTED_TRAIN_SIZES))
    ap.add_argument("--target-mode", default=None, choices=[*TARGET_MODES, None],
                    help="assert the dump was built under this EXP-0040-licensed target")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.model_seeds.split(","))
    training = {"max_epochs": args.max_epochs} if args.max_epochs else None
    require_shape = not args.allow_uncertified_dump
    if args.stage == "fit":
        run_fit(args.dump_dir, args.out_dir, arms=tuple(args.arms.split(",")), model_seeds=seeds,
                training=training, train_size=args.train_size,
                curve_sizes=tuple(int(s) for s in args.curve_sizes.split(",")),
                expected_target_mode=args.target_mode, device=args.device,
                source_commit=args.source_commit, require_shape=require_shape)
    else:
        if not args.fit_dir or not args.manifest:
            raise SystemExit("gate-a requires --fit-dir and --manifest (the locked artifact set)")
        run_gate_a(args.dump_dir, args.fit_dir, args.out_dir, manifest_path=args.manifest,
                   model_seeds=seeds, expected_target_mode=args.target_mode, device=args.device,
                   adapter_dir=args.adapter_dir, require_shape=require_shape)


if __name__ == "__main__":
    main()
