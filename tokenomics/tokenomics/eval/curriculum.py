"""Curriculum game-master scaffold (CURRICULUM_GAME_MASTER_APPROACH.md Stage 0 + 1).

This is the *curriculum and evaluation* tool, NOT part of the promoted system stack.
It gives the portable producer/orderer stack a replayable task archive and an
independent stress suite so failures can be classified — contract vs policy vs
feasibility — instead of collapsed into one scalar.

Design choice (Stage 0/1): a task is reproduced from its **generator-ref**, not an
inlined world dump. `world_spec = {"generator": <name>, "params": {...}}` names a pure,
seed-deterministic builder in `REPRODUCERS`; `replay()` re-calls it. The existing
fixtures (`scale`, `fixtures`) are already pure functions of `(seed, **params)`, so the
record reproduces the exact economy with no hidden process state — the doc's
replayability requirement. Stage 2's random generator can add an inline world_spec form
later without changing this contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from ..domain.config import Configuration
from . import fixtures, scale


@dataclass(frozen=True)
class TaskRecord:
    """One curriculum/stress task, serialized per the doc's Task Object.

    `world_spec` is a generator-ref (`{"generator", "params"}`) that `replay()` turns
    back into the exact `Configuration`. `expected_claim_boundary` (Stage 1 metadata)
    labels whether the task is inside the promoted claim, out of it, or a known
    feasibility wall — the bandpass/classification inputs, kept off the value scalar."""

    task_id: str
    task_family: str  # scale|fixture|stress|random|mutated|learned
    generator_version: str
    serializer_version: str
    world_seed: int
    episode_seed: int
    world_spec: dict  # {"generator": name, "params": {...}} -> reproduces Configuration
    initial_state_spec: dict = field(default_factory=dict)
    event_trace: list = field(default_factory=list)
    demand_trace: list = field(default_factory=list)
    shock_trace: list = field(default_factory=list)
    legal: bool = True
    replayable: bool = True
    reference_status: str = "not_run"  # solved|partial|failed|not_run
    failure_class: str = "none"
    expected_claim_boundary: str = "in_claim"  # in_claim|out_of_claim|feasibility_wall
    notes: str = ""

    def to_json_line(self) -> str:
        """One archive line (jsonl). Compact, newline-free so it round-trips 1:1."""
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json_line(cls, line: str) -> "TaskRecord":
        return cls(**json.loads(line))


# --- generator-ref reproducers -------------------------------------------------------
# Each maps a `world_spec["generator"]` name to a pure `(seed, **params) -> Configuration`
# builder. Keeping this an explicit whitelist (not getattr) means a record can only
# reproduce vetted, legal, seed-deterministic economies — an illegal generator name is a
# replay failure, not arbitrary code.

Reproducer = Callable[..., Configuration]


def _scarcity_variant(seed: int, *, variant: str, depth: int = 3, width: int = 4) -> Configuration:
    """A single scarcity-suite stress variant off a base scale economy (the suite keys on
    a base Configuration, so the reproducer rebuilds the base then selects the axis)."""
    base = scale.build_scale_config(seed, depth=depth, products_per_layer=width)
    return fixtures.scarcity_suite(base)[variant]


def _degenerate_variant(seed: int, *, variant: str, depth: int = 3, width: int = 4) -> Configuration:
    """A single degenerate edge-economy variant (same two-step: the suite is keyed by name)."""
    return fixtures.degenerate_suite(seed, depth=depth, width=width)[variant]


REPRODUCERS: dict[str, Reproducer] = {
    "build_scale_config": scale.build_scale_config,
    "build_forkjoin_config": fixtures.build_forkjoin_config,
    "build_deep_join_config": fixtures.build_deep_join_config,
    "build_shared_capacity_config": fixtures.build_shared_capacity_config,
    "build_uneven_strength_config": fixtures.build_uneven_strength_config,
    "scarcity_variant": _scarcity_variant,
    "degenerate_variant": _degenerate_variant,
}


def replay(record: TaskRecord) -> Configuration:
    """Rebuild the exact economy a record describes, from its generator-ref alone.

    Pure + deterministic: `world_seed` + `world_spec["params"]` fully determine the
    Configuration, so champion, candidate, and reference bars all replay the same world."""
    spec = record.world_spec
    name = spec["generator"]
    if name not in REPRODUCERS:
        raise KeyError(f"unknown task generator {name!r} (not a vetted reproducer)")
    cfg = REPRODUCERS[name](record.world_seed, **spec.get("params", {}))
    cfg.validate()
    return cfg


# --- failure classification ----------------------------------------------------------
# The doc's non-negotiable: keep contract / implementation / feasibility / policy /
# distribution failures DISTINCT — never collapsed into one scalar. Precedence below is
# most-specific-first so a row lands in exactly one class.

FAILURE_CLASSES = (
    "none", "policy_weakness", "feasibility_wall",
    "contract_failure", "implementation_failure", "distribution_mismatch",
)


@dataclass(frozen=True)
class ReplayOutcome:
    """One replayed task's learner-vs-reference result on the same world.

    `contract_ok` = both checkpoints loaded and ran the target's shapes; `crashed` = the
    episode raised / NaN'd / timed out. Units are mean consumer-completed units; `*_dead`
    is the death flag (system death). All the inputs `classify_failure` needs, no more."""

    learner_units: float | None
    reference_units: float | None
    learner_dead: bool
    reference_dead: bool
    contract_ok: bool
    crashed: bool = False


def _failed(units: float | None, dead: bool) -> bool:
    """A policy 'failed' the task if it died or delivered no units."""
    return dead or units is None or units <= 0.0


def classify_failure(outcome: ReplayOutcome, *, boundary: str = "in_claim",
                     gap_eps: float = 1e-9) -> str:
    """Map a replay outcome + claim-boundary label to exactly one failure class.

    Precedence (most specific first): contract can't-run beats everything; a crash is an
    implementation failure; a labeled wall where BOTH policies fail is a feasibility wall;
    an out-of-claim task is a distribution mismatch (never miscounted as a promoted-claim
    policy weakness); a reference that beats the learner on an in-claim task is a policy
    weakness; otherwise nothing is wrong."""
    if not outcome.contract_ok:
        return "contract_failure"
    if outcome.crashed:
        return "implementation_failure"
    if boundary == "feasibility_wall" and _failed(outcome.learner_units, outcome.learner_dead) \
            and _failed(outcome.reference_units, outcome.reference_dead):
        return "feasibility_wall"
    if boundary == "out_of_claim":
        return "distribution_mismatch"
    ref = outcome.reference_units or 0.0
    learner = outcome.learner_units or 0.0
    if ref > learner + gap_eps:
        return "policy_weakness"
    return "none"


# --- Stage 1: independent stress suite ------------------------------------------------
# Hand-scripted, evaluation-only tasks converted from the already-accepted fixture
# families (scale held-out, out-of-family recipe/capacity, scarcity, degenerate edges).
# Kept SEPARATE from any training replay buffer (doc Stage 1 reject rule). Each row is a
# generator-ref TaskRecord so champion, candidate, and reference bars replay the same
# world, with the claim boundary labeled so a gap is scored, not miscounted.

_STRESS_GENERATOR_VERSION = "v1"


def _stress_record(task_id: str, family: str, generator: str, seed: int, params: dict,
                   boundary: str, notes: str = "") -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        task_family=family,
        generator_version=_STRESS_GENERATOR_VERSION,
        serializer_version="v1",
        world_seed=seed,
        episode_seed=0,
        world_spec={"generator": generator, "params": params},
        legal=True,
        replayable=True,
        reference_status="not_run",
        failure_class="none",
        expected_claim_boundary=boundary,
        notes=notes,
    )


# Fixture-derived variant names (mirror `fixtures.scarcity_suite` / `degenerate_suite` keys).
_SCARCITY_VARIANTS = ("low_funding", "low_inventory", "high_backlog", "degraded", "running_investment")
_DEGENERATE_VARIANTS = ("single_producer", "primal_only", "no_inventory_expansion", "no_backlog", "minimal")


def stress_suite() -> list[TaskRecord]:
    """The independent stress suite: one legal, replayable, labeled task per accepted
    fixture family. Held-out scale seeds are the in-claim rows; out-of-family recipe/
    capacity fixtures are out-of-claim; the deep multi-input join (depth 5) is the known
    TTL feasibility wall; scarcity and degenerate edges are single-axis stress."""
    suite: list[TaskRecord] = []

    # in-claim: the champion's train-size held-out scale seeds (depth 3).
    for s in (6, 7, 8, 9):
        suite.append(_stress_record(
            f"stress-scale-d3-s{s}", "scale", "build_scale_config", s, {"depth": 3},
            "in_claim"))

    # out-of-family recipe / capacity (the C3/P3 transfer surface). Depth is serialized
    # explicitly (matching each builder's default) so the archive is self-describing and
    # the matrix's target_depth is never blank — incl. the depth-5 wall below.
    suite.append(_stress_record(
        "stress-forkjoin-s0", "fixture", "build_forkjoin_config", 0, {"depth": 3}, "out_of_claim",
        "multi-input join (recipe DAG out of family)"))
    suite.append(_stress_record(
        "stress-shared_cap-s0", "fixture", "build_shared_capacity_config", 0, {"depth": 3},
        "out_of_claim", "levels share capacity types"))
    suite.append(_stress_record(
        "stress-uneven-s0", "fixture", "build_uneven_strength_config", 0, {"depth": 3},
        "out_of_claim", "one dominant producer per level"))
    # known feasibility wall: depth-5 chain vs TTL=2 (see scale reward calibration).
    suite.append(_stress_record(
        "stress-deep_join-s0", "fixture", "build_deep_join_config", 0, {"depth": 5},
        "feasibility_wall", "depth-5 chain exceeds TTL=2 reach (known wall)"))

    # single-axis scarcity stress off the base scale economy.
    for v in _SCARCITY_VARIANTS:
        suite.append(_stress_record(
            f"stress-scarcity-{v}-s0", "stress", "scarcity_variant", 0, {"variant": v, "depth": 3},
            "out_of_claim", f"scarcity axis: {v}"))

    # degenerate edge economies (contract boundaries).
    for v in _DEGENERATE_VARIANTS:
        suite.append(_stress_record(
            f"stress-degenerate-{v}-s0", "stress", "degenerate_variant", 0, {"variant": v, "depth": 3},
            "out_of_claim", f"degenerate edge: {v}"))

    return suite


# --- replay smoke: run learner + reference on a suite, classify each row --------------
# The learner is the portable stack (producer scorer/planner + orderer scorer/allocator);
# the reference bar is the deterministic `entity_planner + proportional` practical
# baseline (no checkpoint, runs anywhere) — the same bar the transfer matrix reads
# against. Both replay the SAME world, so contract/policy/feasibility stay separable.

# Reference identity recorded per row so a future mixed/stronger reference bar (Stage 3
# search) stays attributable and a row's ratio is never read against an unknown bar.
REFERENCE_MODEL_ID = "entity_planner+proportional"


def _dims(cfg: Configuration) -> tuple[int, int, int]:
    products = len(cfg.process_sheets)
    producers = len(cfg.producer_configs)
    caps = len({t for s in cfg.process_sheets for t in s.require})
    return products, producers, caps


def _chain_depth(cfg: Configuration) -> int:
    """Longest recipe chain (top product -> ... -> primal) = the task's true depth.

    Derived from the recipe DAG, NOT the capacity-type count: shared-capacity fixtures
    reuse types across levels, so counting types undercounts depth. Cycle-safe (recipes
    may be circular across producers) — a back-edge contributes no extra depth."""
    memo: dict[int, int] = {}
    on_stack: set[int] = set()

    def depth(p: int) -> int:
        if p in memo:
            return memo[p]
        if p in on_stack:
            return 1  # cycle guard: don't recurse through a back-edge
        on_stack.add(p)
        d = 1 + max((depth(i) for i in cfg.inputs_of(p)), default=0)
        on_stack.discard(p)
        memo[p] = d
        return d

    return max((depth(s.product) for s in cfg.process_sheets), default=0)


def _reference_status(units: float, dead: bool) -> str:
    if dead or units <= 0.0:
        return "failed"
    return "solved"


def _episode_seeds(episode_seed: int, eval_seeds: int) -> list[int]:
    """The episode seeds a replay evaluates, ANCHORED at the record's `episode_seed`.

    A one-row archive must reproduce the exact condition it serialized (seed
    `episode_seed`, not 0); `--eval-seeds > 1` averages the contiguous block starting
    there so every averaged seed is derivable from the record."""
    return list(range(episode_seed, episode_seed + eval_seeds))


def run_replay_task(record: TaskRecord, producer_model: object, orderer_model: object,
                    *, eval_seeds: int = 1, horizon: int = 100) -> dict:
    """Replay one task under the portable stack (learner) and the deterministic reference
    bar on the same world, then classify the outcome. A learner exception is recorded as a
    contract failure (matrix convention) — never propagated, so one bad cell can't sink the
    run. Returns a curriculum_matrix row."""
    from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from .joint_stack import joint_stack_env, run_joint_stack
    from .runner import run_episode

    cfg = replay(record)
    products, producers, caps = _dims(cfg)
    bar_prod = PRODUCING_BASELINES["entity_planner"]
    bar_order = ORDERING_BASELINES["proportional"]

    # Episodes ANCHORED at the record's serialized seed (not 0), so the row reproduces
    # the exact condition(s) the archive records.
    seeds = _episode_seeds(record.episode_seed, eval_seeds)
    n = len(seeds)

    # reference bar — deterministic, always runs.
    ref_u = ref_d = ref_rej = 0.0
    for es in seeds:
        rs = run_episode(joint_stack_env(cfg, horizon), bar_prod, bar_order, seed=es)
        ref_u += rs["consumer_completed_units"]; ref_d += float(rs["dead"])
        ref_rej += rs["consumer_rejected_units"]
    ref_u /= n; ref_d /= n; ref_rej /= n

    # learner — the portable stack; any episode-time failure = contract-incompatible cell.
    contract_ok = True
    crashed = False
    notes = record.notes
    learner_u = learner_d = learner_rej = 0.0
    try:
        lu = ld = lr = 0.0
        for es in seeds:
            ls = run_joint_stack(producer_model, orderer_model, cfg, seed=es, horizon=horizon)
            u = ls["consumer_completed_units"]
            if u != u:  # NaN -> implementation failure, not contract
                crashed = True
            lu += u; ld += float(ls["dead"]); lr += ls["consumer_rejected_units"]
        learner_u = lu / n; learner_d = ld / n; learner_rej = lr / n
    except Exception as e:  # noqa: BLE001 — episode-time raise = contract failure per taxonomy
        contract_ok = False
        notes = f"contract: {type(e).__name__}: {e}"

    outcome = ReplayOutcome(
        learner_units=None if not contract_ok else round(learner_u, 2),
        reference_units=round(ref_u, 2),
        learner_dead=bool(learner_d),
        reference_dead=bool(ref_d),
        contract_ok=contract_ok,
        crashed=crashed,
    )
    failure_class = classify_failure(outcome, boundary=record.expected_claim_boundary)

    lu_val = outcome.learner_units
    return {
        "task_id": record.task_id,
        "task_family": record.task_family,
        "expected_claim_boundary": record.expected_claim_boundary,
        "target_products": products,
        "target_producers": producers,
        "target_capacity_types": caps,
        "target_depth": _chain_depth(cfg),  # authoritative (recipe chain), never blank
        "learner_units": lu_val,
        "learner_rejected_units": None if not contract_ok else round(learner_rej, 2),
        "learner_dead": outcome.learner_dead,
        "reference_model_id": REFERENCE_MODEL_ID,
        "reference_units": outcome.reference_units,
        "reference_rejected_units": round(ref_rej, 2),
        "reference_dead": outcome.reference_dead,
        "reference_status": _reference_status(ref_u, outcome.reference_dead),
        "ratio_vs_reference": (round(lu_val / ref_u, 4)
                               if (lu_val is not None and ref_u) else None),
        "contract_ok": contract_ok,
        "failure_class": failure_class,
        "notes": notes,
    }


def run_replay_suite(records: list[TaskRecord], producer_model: object, orderer_model: object,
                     *, eval_seeds: int = 1, horizon: int = 100,
                     log=lambda *a, **k: None) -> list[dict]:
    """Run every task's learner-vs-reference replay + classification (Stage 0/1 smoke)."""
    rows: list[dict] = []
    for rec in records:
        row = run_replay_task(rec, producer_model, orderer_model,
                              eval_seeds=eval_seeds, horizon=horizon)
        rows.append(row)
        log(f"{row['task_id']}: learner={row['learner_units']} ref={row['reference_units']} "
            f"class={row['failure_class']}")
    return rows


# --- experiment emission (EXP-0023 layout) -------------------------------------------

_MATRIX_COLUMNS = [
    "task_id", "task_family", "expected_claim_boundary",
    "target_products", "target_producers", "target_capacity_types", "target_depth",
    "learner_units", "learner_rejected_units", "learner_dead",
    "reference_model_id", "reference_units", "reference_rejected_units", "reference_dead",
    "reference_status", "ratio_vs_reference", "contract_ok", "failure_class", "notes",
]


def write_archive(records: list[TaskRecord], path) -> None:
    """The task archive as jsonl — one reproducible generator-ref record per line."""
    from pathlib import Path
    with Path(path).open("w") as fh:
        for r in records:
            fh.write(r.to_json_line() + "\n")


def write_matrix_csv(rows: list[dict], path) -> None:
    """The curriculum matrix CSV (stable columns; extras append, never silently dropped)."""
    import csv
    from pathlib import Path
    extra = [k for r in rows for k in r if k not in _MATRIX_COLUMNS]
    fields = _MATRIX_COLUMNS + list(dict.fromkeys(extra))
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """Stage 0/1 scaffold run (local eval/smoke): serialize the stress suite, replay it
    under the portable champion + deterministic reference bar, classify each row, and emit
    the archive + matrix into an EXP-NNNN experiment layout."""
    import argparse
    from pathlib import Path

    from ..train.orderer_scorer import load_scorer
    from ..train.producer_scorer import load_producer_score_model

    ap = argparse.ArgumentParser(description="Curriculum Stage 0/1: task archive + stress replay")
    ap.add_argument("--producer-ckpt", required=True)
    ap.add_argument("--orderer-ckpt", required=True)
    ap.add_argument("--out-dir", required=True, help="experiment dir (writes artifacts/ + summary.csv)")
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    suite = stress_suite()
    write_archive(suite, out / "artifacts" / "task_archive.jsonl")

    prod = load_producer_score_model(args.producer_ckpt)
    order = load_scorer(args.orderer_ckpt)
    rows = run_replay_suite(suite, prod, order, eval_seeds=args.eval_seeds,
                            horizon=args.horizon, log=print)
    write_matrix_csv(rows, out / "artifacts" / "curriculum_matrix.csv")
    write_matrix_csv(rows, out / "summary.csv")

    by_class: dict[str, int] = {}
    for r in rows:
        by_class[r["failure_class"]] = by_class.get(r["failure_class"], 0) + 1
    print(f"\n{len(rows)} tasks; failure classes: {by_class}")


if __name__ == "__main__":
    main()
