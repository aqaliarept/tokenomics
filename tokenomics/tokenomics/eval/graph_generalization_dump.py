"""EXP-0041 generalization dump: collect `graph_primitives_v1` decision graphs over EXPLICIT
whole-episode train/validation/test seed sets, with a conditional label target.

EXP-0039 fitted one MPNN per task on two trajectories per fold and fell to the random floor on
a held-out trajectory. EXP-0041 asks whether that is repairable with more independent episodes,
pooled tasks, and honest splits. This module owns the DATA half:

- SPLIT_REGISTRY: the pre-registered whole-episode splits (nested train 2/4/8, validation, test)
  and the mechanical checks that they are disjoint, nested, task-complete, and never reuse a
  trajectory seed already spent by EXP-0037..EXP-0040 (the repeatedly-inspected anchor set).
- target modes: `single_suffix` (the EXP-0039 label) and `continuation_mean` (the average over K
  EXP-0040 future-demand continuations). The mode is a REQUIRED argument — EXP-0041's HANDOVER
  forbids choosing one before EXP-0040 classifies the target, so nothing here defaults to a mode.
  The original single-suffix label is ALWAYS persisted, as the mean mode's diagnostic control.
- artifacts: decisions.csv, candidates.csv (+ the 7 deployable actor row features, so the fair
  `row_mlp_h32` baseline retrains on the exact same splits/target), continuation_labels.csv,
  split_manifest.json, graph shards, topologies.jsonl, graph_schema.json, spec.json.

The graph schema, encoder, shard format, digest binding, and provenance helpers are reused
UNCHANGED from EXP-0039 (`graph_value_router_dump`) — this experiment changes which episodes are
collected and how a candidate is labeled, nothing about the observation. Test-enforced parity:
in `single_suffix` mode the collected population and labels are identical to the EXP-0039 dump.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

from .graph_value_router_dump import (
    GraphArrays,
    _assert_finite,
    decision_digest,
    encode_graph,
    environment_manifest,
    schema_dict,
    schema_hash,
    source_fingerprint,
    write_shard,
)

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv
    from .baselines import Policy
    from .curriculum import TaskRecord


# --- frozen task set (identical to the EXP-0039 gate tasks; asserted in tests) ---------
TASK_S29 = ("mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-"
            "producers_per_layer2-quantity_max14-s29")
TASK_S77 = ("mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-"
            "producers_per_layer3-quantity_max14-s77")
TASK_S45 = ("mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-"
            "producers_per_layer3-quantity_max14-s45")
TASK_IDS: tuple[str, ...] = (TASK_S29, TASK_S77, TASK_S45)

SPLITS: tuple[str, ...] = ("train", "validation", "test")
NESTED_TRAIN_SIZES: tuple[int, ...] = (2, 4, 8)

# Protocol v2 — STAGED label collection. The expensive labels are bought in two tranches: the fit
# tranche first, and the final-test tranche only after the arm and adapter are chosen and locked.
# This is not a weakening of the pre-registration: the FULL registry (test seeds included), the
# target mode, the stop rule and the commit are all fixed before the first label is paid for, and
# `certify_dump_contract` requires every tranche to record that same full registry. What changes
# is only WHEN the money is spent — and, usefully, the fit cannot read a final-test label because
# at fit time it does not yet exist. Saves the 9 final-test episodes (~21% of the label budget)
# whenever validation is not passed.
FIT_TRANCHE: tuple[str, ...] = ("train", "validation")
TEST_TRANCHE: tuple[str, ...] = ("test",)

# Protocol v2 — ROLLOUT-ONLY episodes. The labeled splits above stay at 3 episodes per task,
# because a teacher label costs a lookahead search per candidate. A closed-loop ROLLOUT costs no
# teacher at all — it just plays the policy — so the adapter grid and Gate B are not obliged to
# live on 3 episodes each. They were: nine (alpha, margin) configurations were being selected on
# nine validation episodes and Gate B read nine test episodes, where one lucky seed can carry the
# result. These seed blocks are pre-registered exactly like the labeled ones, disjoint from them
# and from every seed EXP-0037..0040 already spent.
ROLLOUT_VALIDATION_SEEDS_PER_TASK = 10
ROLLOUT_TEST_SEEDS_PER_TASK = 20
ROLLOUT_REGISTRY: dict[str, dict[str, tuple[int, ...]]] = {
    TASK_S29: {"validation": tuple(range(1300, 1310)), "test": tuple(range(2300, 2320))},
    TASK_S77: {"validation": tuple(range(1400, 1410)), "test": tuple(range(2400, 2420))},
    TASK_S45: {"validation": tuple(range(1500, 1510)), "test": tuple(range(2500, 2520))},
}

# Pre-registered whole-episode splits (HANDOVER §Episode Splits). Train seeds are NESTED: the
# 2- and 4-episode learning-curve fits are prefixes of the 8-episode primary fit, so the curve
# varies only the number of independent episodes. Validation and final test are separate,
# never-inspected seed blocks.
SPLIT_REGISTRY: dict[str, dict[str, tuple[int, ...]]] = {
    TASK_S29: {"train": (32, 33, 34, 35, 36, 37, 38, 39),
               "validation": (129, 130, 131), "test": (229, 230, 231)},
    TASK_S77: {"train": (80, 81, 82, 83, 84, 85, 86, 87),
               "validation": (177, 178, 179), "test": (277, 278, 279)},
    TASK_S45: {"train": (48, 49, 50, 51, 52, 53, 54, 55),
               "validation": (145, 146, 147), "test": (245, 246, 247)},
}

# Trajectory seeds already spent (and repeatedly inspected) by EXP-0037..EXP-0040. Reusing one
# in ANY EXP-0041 split would recycle the anchor set the earlier reads were tuned against, so
# the registry validator rejects it outright.
SPENT_SEEDS: dict[str, frozenset[int]] = {
    TASK_S29: frozenset({29, 30, 31}),
    TASK_S77: frozenset({77, 78, 79}),
    TASK_S45: frozenset({45, 46, 47}),
}

TARGET_MODES: tuple[str, ...] = ("single_suffix", "continuation_mean")
FLAT_TOL = 1e-9  # same tolerance the EXP-0039 probe uses to call a decision flat
ROW_FEATURES = 7  # producer_row_features width (train.value_router.PRODUCER_FEATURES)

# The frozen EXP-0039 environment contract every real EXP-0041 dump must satisfy
# (HANDOVER §Frozen Inputs). Checked by `certify_dump_contract`, not merely documented.
FROZEN_CONTRACT: dict = {"delta": 1.0, "lookahead": 8, "horizon": 100, "floor": "blend_0.5",
                         "schema_hash": "9918b12f69760bcd"}


# --- EXP-0040 target license -----------------------------------------------------------
# EXP-0041's target is NOT a free parameter: EXP-0040's mechanical classification selects it,
# and two of the four classifications forbid the experiment outright (HANDOVER §Hard Dependency).
EXP0040_TARGET_LICENSE: dict[str, "str | None"] = {
    "MEAN_TARGET_LICENSED": "continuation_mean",
    "COVERAGE_PATH_LICENSED": "single_suffix",
    "AMBIGUOUS": None,
    "SEARCH_ONLY": None,
}
# The only continuation counts EXP-0040 can license; the run must take the SMALLEST that met
# EXP-0040 convergence, never an arbitrary K.
LICENSED_K: tuple[int, ...] = (4, 8, 16)
EXP0040_ID = "EXP-0040"


class Exp0040Blocked(ValueError):
    """EXP-0040 returned `AMBIGUOUS` — EXP-0041 is blocked pending a narrower target audit."""


class Exp0040Cancelled(ValueError):
    """EXP-0040 returned `SEARCH_ONLY` — the mechanical verdict is `CANCELLED_BY_EXP0040`."""


@dataclass(frozen=True)
class TargetLicense:
    """The ONE (target_mode, K) pair EXP-0040 licenses, plus the artifact it was read from.

    A dump cannot be produced without one: `target_mode` alone is a free string, so leaving the
    choice to a CLI flag would let an unlicensed target buy a full label run (review finding 3)."""

    classification: str
    target_mode: str
    continuations: int
    source: str
    source_sha256: "str | None" = None
    # EXP-0040's own convergence map {K: converged?}. Carried into the dump spec so certification
    # can RE-DERIVE K from the evidence instead of trusting the copied number (review finding 4).
    evidence: "dict | None" = None

    def as_dict(self) -> dict:
        return {"classification": self.classification, "target_mode": self.target_mode,
                "continuations": self.continuations, "source": self.source,
                "source_sha256": self.source_sha256, "evidence": self.evidence}


def target_license(classification: str, *, continuations: "int | None" = None,
                   convergence_by_k: "dict | None" = None, source: str = "inline",
                   source_sha256: "str | None" = None) -> TargetLicense:
    """Turn an EXP-0040 classification into the licensed target, or refuse.

    `MEAN_TARGET_LICENSED` additionally needs the converged continuation count: either an explicit
    `continuations` in {4,8,16} or a `convergence_by_k` map from which the SMALLEST passing K is
    taken (HANDOVER: "the smallest value in {4,8,16} that met EXP-0040 convergence"). Without that
    evidence there is no licensed K and the dump must not run."""
    if classification not in EXP0040_TARGET_LICENSE:
        raise ValueError(f"unknown EXP-0040 classification {classification!r} "
                         f"(expected one of {sorted(EXP0040_TARGET_LICENSE)})")
    if classification == "AMBIGUOUS":
        raise Exp0040Blocked("EXP-0040 classified the target AMBIGUOUS: EXP-0041 is blocked until "
                             "a narrower target audit is pre-registered")
    if classification == "SEARCH_ONLY":
        raise Exp0040Cancelled("EXP-0040 classified the target SEARCH_ONLY: EXP-0041 is cancelled "
                               "(verdict CANCELLED_BY_EXP0040); no data or training run is allowed")
    mode = EXP0040_TARGET_LICENSE[classification]
    assert mode is not None
    evidence = ({str(k): bool(v) for k, v in convergence_by_k.items()}
                if convergence_by_k else None)
    if mode == "single_suffix":
        if continuations:
            raise ValueError("COVERAGE_PATH_LICENSED licenses the single-suffix label; averaging "
                             f"{continuations} continuations is not licensed")
        return TargetLicense(classification, mode, 0, source, source_sha256, evidence)
    passing = sorted(k for k in LICENSED_K
                     if bool((convergence_by_k or {}).get(str(k), (convergence_by_k or {}).get(k))))
    if continuations is None:
        if not passing:
            raise ValueError("MEAN_TARGET_LICENSED needs the converged continuation count: pass "
                             f"continuations in {LICENSED_K} or a convergence_by_k map")
        continuations = passing[0]
    if continuations not in LICENSED_K:
        raise ValueError(f"continuations {continuations} is not one of the licensed {LICENSED_K}")
    if passing and continuations != passing[0]:
        raise ValueError(f"K={continuations} is not the SMALLEST converged continuation count "
                         f"{passing[0]} EXP-0040 licensed (passing: {passing})")
    return TargetLicense(classification, mode, int(continuations), source, source_sha256, evidence)


def read_exp0040_license(path: str) -> TargetLicense:
    """Read EXP-0040's artifact-derived classification from its result JSON and license a target.

    Accepts the classification at the top level or under `verdict`/`classification`, so either the
    experiment's `spec.json` result field or its `artifacts/selection_manifest.json` can be pointed
    at directly. Anything else — including a missing classification — raises rather than defaulting
    to a mode.

    Two things the artifact must supply ITSELF (review finding 4). First, its own identity: a file
    that merely contains the string `MEAN_TARGET_LICENSED` could be any document, so the JSON has
    to stamp `experiment: EXP-0040`. Second, the continuation count — which is why there is no
    `continuations` argument here. K is derived from the artifact's `convergence_by_k` map (the
    smallest converged K in {4,8,16}); a caller-supplied K would let an artifact that converged at
    8 buy a 16-continuation label run, and a bare `converged_continuations` field with no map
    behind it is a claim rather than evidence."""
    from .graph_value_router_dump import sha256_file

    doc = json.load(open(path))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not an {EXP0040_ID} result document")
    node: dict = doc
    for key in ("verdict", "result", "classification"):
        value = doc.get(key)
        if isinstance(value, dict):
            node = value
            break
    stamped = {str(d.get("experiment", "")) for d in (doc, node)}
    if EXP0040_ID not in stamped:
        raise ValueError(f"{path} does not identify itself as {EXP0040_ID} (experiment="
                         f"{sorted(s for s in stamped if s)}); only {EXP0040_ID}'s own result "
                         "artifact may license EXP-0041's target")
    label = None
    for key in ("classification", "verdict", "result"):
        value = node.get(key) if isinstance(node, dict) else None
        if isinstance(value, str):
            label = value
            break
    if label is None:
        raise ValueError(f"{path} carries no EXP-0040 classification field; EXP-0041 may not pick "
                         "a target mode without one")
    convergence = node.get("convergence_by_k", doc.get("convergence_by_k"))
    license = target_license(label, convergence_by_k=convergence, source=path,
                             source_sha256=sha256_file(path))
    declared = node.get("converged_continuations", doc.get("converged_continuations"))
    if declared is not None and int(declared) != license.continuations:
        raise ValueError(f"{path} declares converged_continuations={declared} but its "
                         f"convergence_by_k evidence licenses K={license.continuations}")
    return license


# --- split registry ------------------------------------------------------------------
def validate_rollout_registry(rollout: "dict | None" = None,
                              labeled: "dict | None" = None, *, strict: bool = True) -> None:
    """Reject a rollout-only seed table that could contaminate a labeled split (protocol v2).

    Rollout episodes are cheap because they never call the teacher, which is exactly why it would
    be easy to be careless with them — and a Gate B rollout played on an episode the model was
    fitted or checkpoint-selected on is not a held-out number at all. So the same discipline as the
    labeled registry applies: pre-registered, unique, disjoint from EVERY labeled split of the same
    task (not merely the same-named one), disjoint between validation and test, and never a seed
    EXP-0037..0040 already spent. Under `strict` the block sizes are the registered 10 and 20."""
    reg = ROLLOUT_REGISTRY if rollout is None else rollout
    lab = SPLIT_REGISTRY if labeled is None else labeled
    if strict and set(reg) != set(TASK_IDS):
        raise ValueError(f"rollout registry tasks {sorted(reg)} != registered {sorted(TASK_IDS)}")
    want = {"validation": ROLLOUT_VALIDATION_SEEDS_PER_TASK, "test": ROLLOUT_TEST_SEEDS_PER_TASK}
    for task, splits in reg.items():
        for name in ("validation", "test"):
            if name not in splits or not splits[name]:
                raise ValueError(f"{task}: rollout split {name!r} missing or empty")
            seeds = list(splits[name])
            if any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
                raise ValueError(f"{task}/rollout-{name}: seeds must be ints, got {seeds}")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{task}/rollout-{name}: duplicate seeds {seeds}")
            if strict and len(seeds) != want[name]:
                raise ValueError(f"{task}/rollout-{name}: {len(seeds)} seeds != the registered "
                                 f"{want[name]}")
            spent = set(seeds) & set(SPENT_SEEDS.get(task, ()))
            if strict and spent:
                raise ValueError(f"{task}/rollout-{name}: seeds {sorted(spent)} were already spent "
                                 "by EXP-0037..0040")
            # against EVERY labeled split, not just the same-named one: a Gate B rollout on a
            # TRAIN episode is contaminated no matter which block it was filed under
            for lab_name in SPLITS:
                clash = set(seeds) & set((lab.get(task) or {}).get(lab_name, ()))
                if clash:
                    raise ValueError(f"{task}: rollout-{name} reuses labeled {lab_name} episodes "
                                     f"{sorted(clash)}")
        overlap = set(splits["validation"]) & set(splits["test"])
        if overlap:
            raise ValueError(f"{task}: rollout validation and test share seeds {sorted(overlap)}")


def validate_split_registry(registry: "dict[str, dict[str, tuple[int, ...]]] | None" = None, *,
                            strict: bool = True) -> None:
    """Reject any split table that could leak an episode between train/validation/test.

    Checked always: the three split keys exist and are non-empty; seeds are unique ints; the
    three splits are pairwise DISJOINT within a task; the nested train sizes are available and
    are true prefixes of the train list. Checked under `strict` (the pre-registered run, off for
    a tiny smoke): exactly the three registered tasks, the exact 8/3/3 split sizes, and no
    overlap with the EXP-0037..0040 spent seeds. Split leakage cannot be detected downstream —
    a contaminated validation set simply reports an optimistic number — so it fails here."""
    reg = SPLIT_REGISTRY if registry is None else registry
    if strict and set(reg) != set(TASK_IDS):
        raise ValueError(f"split registry tasks {sorted(reg)} != registered {sorted(TASK_IDS)}")
    for task, splits in reg.items():
        missing = [s for s in SPLITS if s not in splits or not splits[s]]
        if missing:
            raise ValueError(f"{task}: split(s) {missing} missing or empty")
        for name in SPLITS:
            seeds = list(splits[name])
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{task}/{name}: duplicate trajectory seeds {seeds}")
            if any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
                raise ValueError(f"{task}/{name}: trajectory seeds must be ints, got {seeds}")
        for a in range(len(SPLITS)):
            for b in range(a + 1, len(SPLITS)):
                overlap = set(splits[SPLITS[a]]) & set(splits[SPLITS[b]])
                if overlap:
                    raise ValueError(
                        f"{task}: {SPLITS[a]} and {SPLITS[b]} share episode seeds {sorted(overlap)}")
        train = list(splits["train"])
        for size in NESTED_TRAIN_SIZES:
            if strict and size > len(train):
                raise ValueError(f"{task}: nested train size {size} > {len(train)} train episodes")
            if list(train[:size]) != list(train[:size]):  # pragma: no cover - prefix by construction
                raise ValueError(f"{task}: nested train {size} is not a prefix of train")
        if strict:
            if (len(train), len(splits["validation"]), len(splits["test"])) != (8, 3, 3):
                raise ValueError(
                    f"{task}: split sizes {(len(train), len(splits['validation']), len(splits['test']))}"
                    " != pre-registered (8, 3, 3)")
            spent = SPENT_SEEDS.get(task, frozenset())
            reused = {n: sorted(set(splits[n]) & spent) for n in SPLITS if set(splits[n]) & spent}
            if reused:
                raise ValueError(
                    f"{task}: split(s) reuse EXP-0037..0040 trajectory seeds {reused}; EXP-0041 "
                    "requires episodes that were never part of the inspected anchor set")


def nested_train_seeds(task_id: str, size: int, *,
                       registry: "dict[str, dict[str, tuple[int, ...]]] | None" = None
                       ) -> tuple[int, ...]:
    """The first `size` train episodes for `task_id` — the 2 -> 4 -> 8 learning-curve rung. A
    prefix (not a resample) so the curve changes ONLY the independent-episode count."""
    reg = SPLIT_REGISTRY if registry is None else registry
    train = tuple(reg[task_id]["train"])
    if size > len(train):
        raise ValueError(f"{task_id}: nested train size {size} > {len(train)} available episodes")
    return train[:size]


def split_of(task_id: str, seed: int, *,
             registry: "dict[str, dict[str, tuple[int, ...]]] | None" = None) -> str:
    """Which split an episode belongs to. Raises for an unregistered (task, seed) so a stray
    episode can never be silently folded into training."""
    reg = SPLIT_REGISTRY if registry is None else registry
    for name in SPLITS:
        if seed in tuple(reg.get(task_id, {}).get(name, ())):
            return name
    raise KeyError(f"({task_id}, seed {seed}) is not in the split registry")


def episode_plan(*, registry: "dict[str, dict[str, tuple[int, ...]]] | None" = None,
                 splits: Sequence[str] = SPLITS,
                 task_ids: "Sequence[str] | None" = None) -> list[tuple[str, str, int]]:
    """Deterministic (task, split, seed) traversal order for a dump — task order as given,
    then split order as given, then ascending seed."""
    reg = SPLIT_REGISTRY if registry is None else registry
    tasks = list(task_ids) if task_ids is not None else list(reg)
    return [(t, name, int(s)) for t in tasks for name in splits
            for s in sorted(reg[t][name])]


# --- target labels --------------------------------------------------------------------
def continuation_label_matrix(env: "TokenomicsAECEnv", prod: "Policy", base: "Policy", agent: str,
                              k: int, eligible_cols: Sequence[int], *, delta: float,
                              lookahead: int, task_id: str, digest: int,
                              continuation_ids: Sequence[int]) -> np.ndarray:
    """`[len(continuation_ids), P]` marginal-value labels, one row per counterfactual future.

    For each continuation the EXP-0040 fork replaces ONLY the unconsumed future demand suffix and
    holds the exact pre-action state fixed; `marginal_value_labels` then clones that forked env
    for its unperturbed base rollout AND for every candidate bump, so within one continuation the
    base and all candidates see the SAME future (common random numbers — the comparison stays a
    contrast between candidates, not between futures)."""
    from ..env.future_fork import fork_future_demand
    from .value_router import marginal_value_labels

    rows = []
    for cid in continuation_ids:
        forked = fork_future_demand(env, task_id=task_id, decision_digest=digest,
                                    continuation_id=int(cid))
        rows.append(marginal_value_labels(forked, prod, base, agent, k, eligible_cols, delta,
                                          lookahead))
    return (np.stack(rows) if rows else np.zeros((0, env.spec_.num_producers), dtype=np.float32))


def primary_label_vector(target_mode: str, original: np.ndarray,
                         continuations: "np.ndarray | None") -> np.ndarray:
    """The label the model is fitted on. `single_suffix` -> the original realized-future label
    (identical to EXP-0039). `continuation_mean` -> the mean over the K continuation rows, which
    marginalizes the future out; the raw rows are persisted so the mean is recomputable."""
    if target_mode == "single_suffix":
        return np.asarray(original, dtype=np.float32)
    if target_mode == "continuation_mean":
        if continuations is None or continuations.shape[0] == 0:
            raise ValueError("continuation_mean requires at least one continuation label row")
        return continuations.mean(axis=0).astype(np.float32)
    raise ValueError(f"unknown target_mode {target_mode!r} (expected one of {TARGET_MODES})")


def is_flat(labels: np.ndarray, eligible_cols: Sequence[int], tol: float = FLAT_TOL) -> bool:
    """A decision is flat when the PRIMARY target gives every eligible candidate the same value —
    there is no preference to learn. Flatness is a property of the target, never of a prediction."""
    vals = np.asarray([float(labels[c]) for c in eligible_cols], dtype=np.float64)
    return bool(vals.size < 2 or (vals.max() - vals.min()) <= tol)


# --- dump -------------------------------------------------------------------------------
def run_generalization_dump(
    records: "Sequence[TaskRecord]", *, prod: "Policy", target_mode: str,
    continuations: int = 0, base: "Policy | None" = None, delta: float = 1.0, lookahead: int = 8,
    horizon: int = 100, out_dir: str, registry: "dict[str, dict[str, tuple[int, ...]]] | None" = None,
    splits: Sequence[str] = SPLITS, shard_size: int = 2000, strict_splits: bool = True,
    max_decisions: "int | None" = None, input_manifest: "dict | None" = None,
    source_commit: "str | None" = None, license: "TargetLicense | None" = None,
    require_license: bool = True, log: "Callable[..., None]" = print,
) -> dict:
    """Drive `blend_0.5` over every registered episode and materialize one graph per contested
    routing decision (>= 2 eligible producers), labeled under `target_mode`.

    Same causal boundary as EXP-0039: the graph is encoded BEFORE the teacher call and BEFORE
    `env.step`, and no label, actor score, split name, seed, cycle, or raw id ever enters a graph
    tensor — identities live in the CSVs and in the shard's audit digest only.

    `target_mode` is required (EXP-0040 licenses it; nothing here may pick a default) and, unless
    `require_license=False` (a smoke or the timing budget), must MATCH a `TargetLicense` read from
    EXP-0040's result — otherwise an unlicensed target could buy a full label run.
    `continuations` is the K of `continuation_mean` and must be 0 in `single_suffix` mode, so a
    mis-set K cannot silently pay for forks that are then unused. `max_decisions` stops the
    traversal after N labeled decisions — the timing smoke that produces the compute budget
    (`graph_generalization_budget`), never a real dump."""
    from ..domain.order import ORDER_TTL
    from ..train.value_router import producer_row_features
    from .baselines import ordering_blend
    from .curriculum import replay
    from .joint_stack import joint_stack_env
    from .value_router import marginal_value_labels

    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown target_mode {target_mode!r} (expected one of {TARGET_MODES})")
    if target_mode == "single_suffix" and continuations:
        raise ValueError("single_suffix mode takes no continuations (K must be 0)")
    if target_mode == "continuation_mean" and continuations < 1:
        raise ValueError("continuation_mean mode requires continuations >= 1")
    if require_license and license is None:
        raise ValueError(
            "no EXP-0040 TargetLicense: EXP-0041 may not spend label compute on a target mode it "
            "was not licensed to use. Read one with `read_exp0040_license(<EXP-0040 result>)`, or "
            "pass require_license=False for a smoke / timing measurement (never a real dump).")
    if license is not None:
        if (license.target_mode, license.continuations) != (target_mode, continuations):
            raise ValueError(
                f"requested target ({target_mode}, K={continuations}) != the EXP-0040 licensed "
                f"({license.target_mode}, K={license.continuations}) from {license.source}")
    reg = SPLIT_REGISTRY if registry is None else registry
    validate_split_registry(reg, strict=strict_splits)

    base = base if base is not None else ordering_blend(0.5)
    cont_ids = tuple(range(continuations))
    by_id = {r.task_id: r for r in records}
    plan = episode_plan(registry=reg, splits=splits, task_ids=[r.task_id for r in records])

    graphs: list[GraphArrays] = []
    dec_rows: list[dict] = []
    cand_rows: list[dict] = []
    cont_rows: list[dict] = []
    topologies: dict[str, dict] = {}

    for task_id, split, s in plan:
        if max_decisions is not None and len(dec_rows) >= max_decisions:
            break
        rec = by_id[task_id]
        cfg = replay(rec)
        env = joint_stack_env(cfg, horizon)
        env.reset(seed=s)
        n_before = len(dec_rows)
        for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
            if max_decisions is not None and len(dec_rows) >= max_decisions:
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
            for k, oid in enumerate(incoming):
                if k >= env.spec_.k_stream:
                    break
                elig = view.eligible[oid]
                if len(elig) < 2:
                    continue
                g = encode_graph(build_view(env, aid, oid, list(elig)))
                _assert_finite(g)
                cols = [env.spec_.producer_ids.index(p) for p in elig]
                digest = decision_digest(task_id, s, env.current_cycle, oid)
                original = marginal_value_labels(env, prod, base, agent, k, cols, delta, lookahead)
                cont = (continuation_label_matrix(
                    env, prod, base, agent, k, cols, delta=delta, lookahead=lookahead,
                    task_id=task_id, digest=digest, continuation_ids=cont_ids)
                    if target_mode == "continuation_mean" else None)
                primary = primary_label_vector(target_mode, original, cont)
                rows = producer_row_features(env, oid, view)
                oi = view.incoming[oid]
                order = env.system.orders.get(oid)
                age = order.cycle_counter if order is not None else -1
                ttl = max(0, ORDER_TTL - age) if age >= 0 else -1
                same_prod = sum(1 for o in incoming if view.incoming[o].product == oi.product)
                di = len(dec_rows)
                g.decision_digest = digest
                graphs.append(g)
                best = max(cols, key=lambda c: float(primary[c]))
                dec_rows.append({
                    "decision_index": di, "task_id": task_id, "split": split,
                    "trajectory_seed": s, "cycle": env.current_cycle, "agent": agent,
                    "order_id": oid, "stream_index": k, "product": oi.product,
                    "quantity": oi.quantity, "order_age": age, "ttl_remaining": ttl,
                    "backlog_size": len(incoming), "same_product_backlog": same_prod,
                    "n_eligible": len(cols), "decision_digest": digest,
                    "target_mode": target_mode, "continuation_count": len(cont_ids),
                    "primary_flat": int(is_flat(primary, cols)),
                    "original_flat": int(is_flat(original, cols)),
                    "teacher_best_producer": env.spec_.producer_ids[best],
                    "primary_best_value": round(float(primary[best]), 6),
                })
                for ci, col in enumerate(cols):
                    row = {
                        "decision_index": di, "task_id": task_id, "split": split,
                        "trajectory_seed": s, "cycle": env.current_cycle, "order_id": oid,
                        "decision_digest": digest, "candidate_index": ci,
                        "producer_id": env.spec_.producer_ids[col],
                        "original_suffix_label": round(float(original[col]), 6),
                        "primary_label": round(float(primary[col]), 6),
                        "target_mode": target_mode, "continuation_count": len(cont_ids),
                    }
                    # the deployable 7-feature actor row travels WITH the candidate so the fair
                    # `row_mlp_h32` baseline is retrained on exactly these splits + this target
                    # (EXP-0039's floor was fitted on its own evaluation trajectories).
                    for fi in range(ROW_FEATURES):
                        row[f"feat_{fi}"] = round(float(rows[col][fi]), 6)
                    cand_rows.append(row)
                    if cont is not None:
                        for cid in cont_ids:
                            cont_rows.append({
                                "decision_index": di, "task_id": task_id, "split": split,
                                "trajectory_seed": s, "candidate_index": ci,
                                "producer_id": env.spec_.producer_ids[col],
                                "continuation_id": int(cid),
                                "label": round(float(cont[cid][col]), 6)})
                topologies.setdefault(task_id, {
                    "task_id": task_id,
                    "producer_ids": g.id_maps["producer"],
                    "product_ids": g.id_maps["product"],
                    "capacity_type_ids": g.id_maps["capacity_type"],
                    "orderer_ids": g.id_maps["orderer"]})
            env.step(base(env, agent))
        log(f"[gen-dump] {task_id} {split} seed {s}: +{len(dec_rows) - n_before} decisions "
            f"({len(dec_rows)} total)")

    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, "decisions.csv"), dec_rows)
    _write_csv(os.path.join(out_dir, "candidates.csv"), cand_rows)
    if target_mode == "continuation_mean":
        _write_csv(os.path.join(out_dir, "continuation_labels.csv"), cont_rows)
    with open(os.path.join(out_dir, "topologies.jsonl"), "w") as f:
        for t in topologies.values():
            f.write(json.dumps(t) + "\n")
    with open(os.path.join(out_dir, "graph_schema.json"), "w") as f:
        json.dump({**schema_dict(), "schema_hash": schema_hash()}, f, indent=2)
    n_shards = 0
    for start in range(0, len(graphs), shard_size):
        write_shard(os.path.join(out_dir, f"graph_shard_{n_shards:03d}.npz"),
                    graphs[start:start + shard_size])
        n_shards += 1

    manifest = split_manifest(reg, dec_rows, cand_rows, splits=splits,
                              task_ids=[r.task_id for r in records])
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    spec = {
        "experiment": "EXP-0041", "schema_name": schema_dict()["schema_name"],
        "schema_hash": schema_hash(), "task_ids": [r.task_id for r in records],
        "target_mode": target_mode, "continuation_count": len(cont_ids),
        "delta": delta, "lookahead": lookahead, "horizon": horizon, "floor": "blend_0.5",
        "splits": list(splits), "n_decisions": len(dec_rows), "n_candidates": len(cand_rows),
        "n_continuation_rows": len(cont_rows), "n_shards": n_shards, "shard_size": shard_size,
        "split_manifest": manifest, "inputs": input_manifest,
        "license": license.as_dict() if license is not None else None,
        "source": source_fingerprint(source_commit), "env": environment_manifest(),
    }
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    log(f"{len(dec_rows)} decisions, {len(cand_rows)} candidates, {n_shards} shards -> {out_dir}")
    return spec


def build_view(env: "TokenomicsAECEnv", aid: str, oid: str, eligible: Sequence[str]):
    """The EXP-0039 pre-action graph view for one decision (kept as a one-liner so the dump's
    causal snapshot boundary is a single named call, identical to EXP-0039's)."""
    from ..domain.graph_view import build_economy_graph_view

    return build_economy_graph_view(env.system, aid, oid, list(eligible))


def split_manifest(registry: dict, dec_rows: Sequence[dict], cand_rows: Sequence[dict], *,
                   splits: Sequence[str] = SPLITS,
                   task_ids: "Sequence[str] | None" = None) -> dict:
    """The exact split table used + what was actually collected per (task, split, seed):
    decisions, candidates, and flat-decision counts under the primary target. A reader can
    verify from this alone that no episode crossed a split and that every registered episode
    contributed rows (HANDOVER: report every metric by task and episode seed).

    `registry` is recorded IN FULL (all three splits) even when a dump materializes a subset —
    a downstream reader validates the whole table for disjointness, and a partial table would
    look like a malformed registry rather than a partial collection. `collected_splits` records
    what this dump actually walked."""
    tasks = list(task_ids) if task_ids is not None else list(registry)
    counts: dict[str, dict] = {}
    for r in dec_rows:
        key = f"{r['task_id']}|{r['split']}|{r['trajectory_seed']}"
        c = counts.setdefault(key, {"task_id": r["task_id"], "split": r["split"],
                                    "trajectory_seed": int(r["trajectory_seed"]),
                                    "n_decisions": 0, "n_flat_primary": 0, "n_candidates": 0})
        c["n_decisions"] += 1
        c["n_flat_primary"] += int(r["primary_flat"])
    for r in cand_rows:
        counts[f"{r['task_id']}|{r['split']}|{r['trajectory_seed']}"]["n_candidates"] += 1
    return {
        "registry": {t: {name: list(registry[t][name]) for name in SPLITS} for t in tasks},
        "collected_splits": list(splits),
        "nested_train": {t: {str(n): list(nested_train_seeds(t, n, registry=registry))
                             for n in NESTED_TRAIN_SIZES if n <= len(registry[t]["train"])}
                         for t in tasks},
        "collected": sorted(counts.values(),
                            key=lambda c: (c["task_id"], c["split"], c["trajectory_seed"])),
    }


def certify_dump_contract(spec: dict, *, expected_target_mode: "str | None" = None,
                          expected_tasks: Sequence[str] = TASK_IDS,
                          expected_registry: "dict | None" = None,
                          expected_inputs: "dict | None" = None,
                          expected_splits: Sequence[str] = SPLITS,
                          n_decisions: "int | None" = None,
                          n_candidates: "int | None" = None,
                          require_commit: bool = True) -> None:
    """Assert a dump really is THE pre-registered EXP-0041 dataset before anything reads it.

    A downstream stage that only checks the schema hash and the target-mode string will happily
    fit, gate, and publish a verdict from a dump built on the wrong topologies, the wrong seeds,
    an unpinned producer checkpoint, a different lookahead, or an unlicensed target (review
    finding 3). Everything the verdict silently depends on is checked here, from the dump's own
    spec.json plus the collected-episode table inside it:

    experiment id; frozen delta/lookahead/horizon/floor/schema hash; exactly the three registered
    tasks; the exact split registry AND the exact collected seed set per (task, split); the pinned
    archive/producer SHA-256; a whole-tree source fingerprint with a real commit; the EXP-0040
    license and its agreement with the dump's target mode and K; and the declared row counts.

    Callers that are genuinely not the pre-registered run (unit fixtures, the timing budget) skip
    this by passing `require_shape=False` to their stage — never by weakening a check here."""
    from .graph_value_router_dump import assert_inputs_match, schema_hash

    if spec.get("experiment") != "EXP-0041":
        raise ValueError(f"dump experiment {spec.get('experiment')!r} != 'EXP-0041'")
    for key, want in FROZEN_CONTRACT.items():
        got = spec.get(key)
        if got != want:
            raise ValueError(f"dump {key}={got!r} != frozen EXP-0039 contract {want!r}")
    if schema_hash() != FROZEN_CONTRACT["schema_hash"]:
        raise ValueError(f"live schema hash {schema_hash()} != frozen "
                         f"{FROZEN_CONTRACT['schema_hash']}: the observation contract moved")
    tasks = list(spec.get("task_ids") or ())
    if sorted(tasks) != sorted(expected_tasks):
        raise ValueError(f"dump tasks {sorted(tasks)} != registered {sorted(expected_tasks)}")
    if list(spec.get("splits") or ()) != list(expected_splits):
        raise ValueError(f"dump collected splits {spec.get('splits')} != the expected tranche "
                         f"{list(expected_splits)}")

    reg_want = SPLIT_REGISTRY if expected_registry is None else expected_registry
    manifest = spec.get("split_manifest") or {}
    reg_got = manifest.get("registry") or {}
    # The FULL registry — final-test seeds included — must be recorded even by a train+validation
    # tranche. That is what makes staged collection safe: the test episodes are pre-registered
    # before the fit runs, so a later tranche cannot pick its test seeds with the fit in hand.
    if {t: {n: sorted(v) for n, v in s.items()} for t, s in reg_got.items()} != \
            {t: {n: sorted(reg_want[t][n]) for n in SPLITS} for t in reg_want}:
        raise ValueError("dump split registry != the pre-registered SPLIT_REGISTRY")
    collected: dict[tuple[str, str], set[int]] = {}
    for row in manifest.get("collected") or ():
        collected.setdefault((row["task_id"], row["split"]), set()).add(int(row["trajectory_seed"]))
    for task in reg_want:
        for name in SPLITS:
            got = collected.get((task, name), set())
            # a tranche must have collected its own splits EXACTLY, and nothing outside them
            want = set(reg_want[task][name]) if name in expected_splits else set()
            if got != want:
                raise ValueError(
                    f"{task}/{name}: collected episodes {sorted(got)} != "
                    f"{sorted(want)} for tranche {list(expected_splits)}"
                    + ("" if name in expected_splits else
                       " (a tranche must not collect labels outside its own splits)"))

    assert_inputs_match(spec.get("inputs") or {},
                        expected_inputs if expected_inputs is not None else _pinned_inputs())
    source = spec.get("source") or {}
    if len(str(source.get("tree_sha256") or "")) != 64:
        raise ValueError("dump has no whole-tree source fingerprint")
    if require_commit and len(str(source.get("commit") or "")) != 40:
        raise ValueError(f"dump source commit {source.get('commit')!r} is not a full 40-char "
                         "revision; a real run must be launched from a committed tree")

    lic = spec.get("license")
    if not lic:
        raise ValueError("dump carries no EXP-0040 license; its target mode was never licensed")
    if str(lic.get("source") or "inline") == "inline" or len(str(lic.get("source_sha256") or "")) != 64:
        raise ValueError("dump license was not read from an EXP-0040 artifact (it records no "
                         "source file + sha256); an inline classification is a claim, not a "
                         "license")
    # K is RE-DERIVED from the recorded convergence evidence, never taken from the recorded number:
    # a copied `continuations: 16` next to evidence that converged at 8 must not certify.
    try:
        licensed = target_license(lic.get("classification", ""),
                                  convergence_by_k=lic.get("evidence"),
                                  source=str(lic.get("source")),
                                  source_sha256=str(lic.get("source_sha256")))
    except (Exp0040Blocked, Exp0040Cancelled):
        raise
    except ValueError as exc:
        raise ValueError("dump license cannot be re-derived from its own EXP-0040 evidence "
                         f"({dict(lic)}): {exc}") from exc
    if licensed.continuations != int(lic.get("continuations") or 0):
        raise ValueError(f"dump license records K={lic.get('continuations')} but its EXP-0040 "
                         f"evidence licenses K={licensed.continuations}")
    if (licensed.target_mode, licensed.continuations) != (spec.get("target_mode"),
                                                          spec.get("continuation_count")):
        raise ValueError(
            f"dump target ({spec.get('target_mode')}, K={spec.get('continuation_count')}) != the "
            f"licensed ({licensed.target_mode}, K={licensed.continuations})")
    if expected_target_mode is not None and spec.get("target_mode") != expected_target_mode:
        raise ValueError(f"dump target_mode {spec.get('target_mode')!r} != expected "
                         f"{expected_target_mode!r}")
    for name, got in (("n_decisions", n_decisions), ("n_candidates", n_candidates)):
        if got is not None and int(spec.get(name, -1)) != int(got):
            raise ValueError(f"dump declares {name}={spec.get(name)} but {got} rows are present")


def assert_tranches_agree(fit_spec: dict, test_spec: dict) -> None:
    """Assert a final-test tranche belongs to the same experiment as the fit tranche it will be
    scored against (protocol v2, staged collection).

    Two dumps produced weeks apart could differ in target mode, K, pinned producer, teacher window,
    schema, or registry, and a Gate A that simply read whichever directory it was pointed at would
    silently score a model against labels from a different experiment. Everything the verdict
    depends on must therefore match exactly; the source COMMIT deliberately need not, since the
    fit legitimately happens between the two collections and the tree may move — the frozen
    contract, inputs, licence and registry are what pin the science."""
    for key in (*FROZEN_CONTRACT, "experiment", "target_mode", "continuation_count", "task_ids"):
        a, b = fit_spec.get(key), test_spec.get(key)
        if a != b:
            raise ValueError(f"tranche mismatch on {key!r}: fit={a!r} test={b!r}")
    if (fit_spec.get("inputs") or {}) != (test_spec.get("inputs") or {}):
        raise ValueError("tranche mismatch on pinned inputs (archive / producer checkpoint)")
    lic_a, lic_b = fit_spec.get("license") or {}, test_spec.get("license") or {}
    for key in ("classification", "continuations", "source_sha256"):
        if lic_a.get(key) != lic_b.get(key):
            raise ValueError(f"tranche mismatch on EXP-0040 license {key!r}: "
                             f"fit={lic_a.get(key)!r} test={lic_b.get(key)!r}")
    reg_a = (fit_spec.get("split_manifest") or {}).get("registry") or {}
    reg_b = (test_spec.get("split_manifest") or {}).get("registry") or {}
    if {t: {n: sorted(v) for n, v in s.items()} for t, s in reg_a.items()} != \
            {t: {n: sorted(v) for n, v in s.items()} for t, s in reg_b.items()}:
        raise ValueError("tranche mismatch on the split registry: the final-test tranche was not "
                         "collected against the registry the fit was pre-registered under")
    if list(fit_spec.get("splits") or ()) != list(FIT_TRANCHE):
        raise ValueError(f"fit tranche collected {fit_spec.get('splits')} != {list(FIT_TRANCHE)}")
    if list(test_spec.get("splits") or ()) != list(TEST_TRANCHE):
        raise ValueError(f"test tranche collected {test_spec.get('splits')} != {list(TEST_TRANCHE)}")


def _pinned_inputs() -> dict:
    """The EXP-0039 archive + EXP-0035 producer SHA-256 pins (no actor: EXP-0041 retrains it)."""
    from ..train.graph_router_probe import GATE_INPUT_SHA256

    return {k: v for k, v in GATE_INPUT_SHA256.items() if k != "actor_ckpt"}


def _write_csv(path: str, rows: Sequence[dict]) -> None:
    import csv

    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    import argparse

    from .graph_value_router_dump import assert_inputs_match, resolve_verified_commit, sha256_file

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, help="task archive jsonl")
    ap.add_argument("--task-ids", default=",".join(TASK_IDS),
                    help="comma-separated task ids (default: the three registered tasks)")
    ap.add_argument("--prod", default="stock_aware", help="producer baseline held fixed")
    ap.add_argument("--prod-ckpt", default=None, help="producer scorer checkpoint (EXP-0035)")
    ap.add_argument("--exp0040-result", required=True,
                    help="REQUIRED — EXP-0040's result JSON; its mechanical classification is what "
                         "licenses the target mode and K (AMBIGUOUS blocks, SEARCH_ONLY cancels)")
    ap.add_argument("--target-mode", default=None, choices=[*TARGET_MODES, None],
                    help="optional assertion: must equal the mode EXP-0040 licensed")
    # there is deliberately NO --continuations flag: K is EXP-0040's to license, read from the
    # convergence evidence in its result artifact (review finding 4)
    ap.add_argument("--splits", default=",".join(SPLITS),
                    help="which registered splits to collect (default: all three)")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-commit", default=None,
                    help="full 40-char git commit of the producing tree (the pod has no .git)")
    ap.add_argument("--verify-source-commit", action="store_true",
                    help="resolve+verify the commit from the LOCAL git tree and use it")
    # provenance is ON by default: the pinned-input and commit checks are what make a pulled dump
    # attributable, and an opt-in check is one nobody remembers to pass on the real launch.
    ap.add_argument("--allow-unpinned-inputs", action="store_true",
                    help="skip the archive/producer SHA-256 pin check (smoke only)")
    ap.add_argument("--allow-uncommitted-source", action="store_true",
                    help="skip the source-commit requirement (smoke only)")
    args = ap.parse_args()

    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from .baselines import PRODUCING_BASELINES
    from .curriculum_replay import _load_archive

    by_id = {r.task_id: r for r in _load_archive(args.archive)}
    records = [by_id[t] for t in args.task_ids.split(",")]
    prod = (scorer_producing_policy(load_producer_score_model(args.prod_ckpt))
            if args.prod_ckpt else PRODUCING_BASELINES[args.prod])

    def _fp(path: "str | None") -> "dict | None":
        return {"path": path, "sha256": sha256_file(path)} if path else None

    input_manifest = {"archive": _fp(args.archive), "producer_ckpt": _fp(args.prod_ckpt)}
    if not args.allow_unpinned_inputs:
        assert_inputs_match(input_manifest, _pinned_inputs())
    source_commit = args.source_commit
    if args.verify_source_commit:
        source_commit = resolve_verified_commit()
    if not args.allow_uncommitted_source and len(str(source_commit or "")) != 40:
        raise SystemExit("a real dump must record its revision: pass --verify-source-commit "
                         "(local git) or --source-commit <40-char sha>")

    license = read_exp0040_license(args.exp0040_result)
    if args.target_mode is not None and args.target_mode != license.target_mode:
        raise SystemExit(f"--target-mode {args.target_mode} != the EXP-0040 licensed "
                         f"{license.target_mode} ({license.classification})")

    run_generalization_dump(
        records, prod=prod, target_mode=license.target_mode,
        continuations=license.continuations, license=license,
        delta=args.delta, lookahead=args.lookahead, horizon=args.horizon, out_dir=args.out_dir,
        splits=tuple(args.splits.split(",")), shard_size=args.shard_size,
        input_manifest=input_manifest, source_commit=source_commit)


if __name__ == "__main__":
    main()
