"""C-1 verification / transfer matrix (PRODUCT_AGNOSTIC "Evaluation Output").

The doc's stated first deliverable: a compatibility-and-behavior matrix that
separates four failure modes — contract, equivariance, policy, scaling — so a
missing number reads as "checkpoint can't run here" not "policy is bad".

One row = one (stack x target topology) cell. A stack is a (producer_kind,
orderer_kind) pair; deterministic kinds come from the baseline dicts, `portable_scorer`
needs a distilled checkpoint. Everything runs on the same joint env
(`joint_stack_env`) so deterministic bars and portable modules are apples-to-apples.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from ..domain.config import Configuration
from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .joint_stack import joint_stack_env, relabel_capacity_types
from .runner import run_episode
from .scale import build_scale_config


@dataclass
class StackSpec:
    """A producer x orderer pairing under test. `*_kind` is a baseline-dict key or
    `portable_scorer` (which then needs the matching checkpoint model)."""

    model_id: str
    producer_kind: str
    orderer_kind: str
    train_topology: str = "none"


@dataclass
class TargetTopology:
    """A topology to transfer onto. Dims are derived from `config`; `depth` is carried
    because the config doesn't record it."""

    name: str
    config: Configuration
    depth: int


def _target_dims(cfg: Configuration) -> tuple[int, int, int]:
    products = len(cfg.process_sheets)
    producers = len(cfg.producer_configs)
    caps = len({t for s in cfg.process_sheets for t in s.require})
    return products, producers, caps


def _resolve_producer(kind: str, model: object | None) -> tuple[Policy | None, bool]:
    if kind in PRODUCING_BASELINES:
        return PRODUCING_BASELINES[kind], True
    if kind == "portable_scorer":
        if model is None:
            return None, False
        from ..train.producer_scorer import scorer_producing_policy
        return scorer_producing_policy(model), True
    return None, False


def _resolve_orderer(kind: str, model: object | None) -> tuple[Policy | None, bool]:
    if kind in ORDERING_BASELINES:
        return ORDERING_BASELINES[kind], True
    if kind == "portable_scorer":
        if model is None:
            return None, False
        from ..train.orderer_scorer import scorer_ordering_policy
        return scorer_ordering_policy(model), True
    return None, False


def evaluate_stack_on_target(stack: StackSpec, target: TargetTopology, eval_seeds: int = 5,
                             horizon: int = 100, *, producer_model: object | None = None,
                             orderer_model: object | None = None) -> dict:
    """One matrix row: resolve the stack's policies, record schema/contract flags, and
    (if both sides load) average value over `eval_seeds` episodes on the joint env."""
    products, producers, caps = _target_dims(target.config)
    prod_pol, prod_loads = _resolve_producer(stack.producer_kind, producer_model)
    order_pol, order_loads = _resolve_orderer(stack.orderer_kind, orderer_model)

    row: dict = {
        "model_id": stack.model_id,
        "producer_kind": stack.producer_kind,
        "orderer_kind": stack.orderer_kind,
        "train_topology": stack.train_topology,
        "target_topology": target.name,
        "target_products": products,
        "target_producers": producers,
        "target_capacity_types": caps,
        "target_depth": target.depth,
        "producer_schema_loads": prod_loads,
        "orderer_schema_loads": order_loads,
        "producer_contract_compatible": prod_loads,
        "orderer_contract_compatible": order_loads,
    }

    def _contract_failure(note: str) -> dict:
        # The portable side(s) under test are the ones that could be contract-incompatible
        # on this target; deterministic baselines run anywhere. Mark only those False.
        if stack.producer_kind == "portable_scorer":
            row["producer_contract_compatible"] = False
        if stack.orderer_kind == "portable_scorer":
            row["orderer_contract_compatible"] = False
        if not (stack.producer_kind == "portable_scorer"
                or stack.orderer_kind == "portable_scorer"):
            row["producer_contract_compatible"] = False
            row["orderer_contract_compatible"] = False
        row.update(runs_episode=False, failure_class="contract",
                   consumer_completed_units=None, consumer_rejected_units=None,
                   dead_rate=None, elapsed_sec=0.0, peak_memory_kib=0.0, notes=note)
        return row

    if not (prod_loads and order_loads):
        return _contract_failure("contract: checkpoint did not load")

    cu = rj = dr = 0.0
    t0 = time.perf_counter()
    try:
        # A loaded checkpoint may still be incompatible with this target's tensor/action
        # shapes (legacy fixed actors, C2/C3 off-size targets). A raise here is a CONTRACT
        # failure per the doc's taxonomy, recorded as a row — never propagated out so one
        # bad cell can't sink the whole matrix.
        for es in range(eval_seeds):
            s = run_episode(joint_stack_env(target.config, horizon), prod_pol, order_pol, seed=es)
            cu += s["consumer_completed_units"]
            rj += s["consumer_rejected_units"]
            dr += float(s["dead"])
    except Exception as e:  # noqa: BLE001 — any episode-time failure = contract-incompatible
        return _contract_failure(f"contract: {type(e).__name__}: {e}")
    elapsed = time.perf_counter() - t0

    row.update(runs_episode=True, failure_class="none",
               consumer_completed_units=round(cu / eval_seeds, 2),
               consumer_rejected_units=round(rj / eval_seeds, 2),
               dead_rate=round(dr / eval_seeds, 3),
               elapsed_sec=round(elapsed, 3), peak_memory_kib=0.0, notes="")
    return row


# The practical baseline every value ratio is read against — a deterministic stack that
# runs on any target with no checkpoint (PRODUCT_AGNOSTIC "Baseline Requirements").
PRACTICAL_BASELINE = StackSpec("entity_planner+proportional",
                               "entity_planner", "proportional")


def transfer_matrix(stacks: list[StackSpec], targets: list[TargetTopology],
                    eval_seeds: int = 5, horizon: int = 100, *,
                    producer_model: object | None = None,
                    orderer_model: object | None = None,
                    log=lambda *a, **k: None) -> list[dict]:
    """The full matrix: every stack x target cell, each row carrying its value ratio vs
    the practical `entity_planner + proportional` baseline measured on the SAME target
    (so contract failures and value failures stay separable per row)."""
    rows: list[dict] = []
    for target in targets:
        base = evaluate_stack_on_target(PRACTICAL_BASELINE, target, eval_seeds, horizon)
        base_cu = base["consumer_completed_units"]
        for stack in stacks:
            row = evaluate_stack_on_target(stack, target, eval_seeds, horizon,
                                           producer_model=producer_model,
                                           orderer_model=orderer_model)
            row["baseline_family"] = "practical"
            row["baseline_model_id"] = PRACTICAL_BASELINE.model_id
            row["baseline_consumer_completed_units"] = base_cu
            row["baseline_dead_rate"] = base["dead_rate"]
            row["baseline_contract_compatible"] = base["runs_episode"]
            cu = row["consumer_completed_units"]
            row["ratio_vs_practical_baseline"] = (
                round(cu / base_cu, 4) if (cu is not None and base_cu) else None)
            # No distinct best-available bar (champion/search teacher) is wired into this
            # matrix yet, so the column stays blank rather than aliasing the practical bar.
            row["ratio_vs_best_available_bar"] = None
            rows.append(row)
            log(f"{target.name} | {stack.model_id}: "
                f"runs={row['runs_episode']} cu={cu} "
                f"ratio={row['ratio_vs_practical_baseline']}")
    return rows


# Canonical column order = PRODUCT_AGNOSTIC "Evaluation Output" + "Baseline columns".
CSV_COLUMNS = [
    "model_id", "producer_kind", "orderer_kind", "train_topology", "target_topology",
    "target_products", "target_producers", "target_capacity_types", "target_depth",
    "producer_schema_loads", "orderer_schema_loads",
    "producer_contract_compatible", "orderer_contract_compatible",
    "runs_episode", "failure_class",
    "consumer_completed_units", "consumer_rejected_units", "dead_rate",
    "elapsed_sec", "peak_memory_kib",
    "baseline_family", "baseline_model_id", "baseline_consumer_completed_units",
    "baseline_dead_rate", "ratio_vs_practical_baseline", "ratio_vs_best_available_bar",
    "baseline_contract_compatible", "notes",
]


def write_matrix_csv(rows: list[dict], path: str | Path) -> None:
    """Emit the matrix as the canonical CSV (stable column order; any extra keys append
    after the known columns so the file never silently drops data)."""
    extra = [k for r in rows for k in r if k not in CSV_COLUMNS]
    fields = CSV_COLUMNS + list(dict.fromkeys(extra))
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


# Stacks populated from already-available artifacts (PRODUCT_AGNOSTIC step 2): floor,
# practical bar, each portable module solo against a deterministic counterpart (C0), and
# the full portable stack (C1/EXP-0014). Portable stacks no-op gracefully if a checkpoint
# is absent (recorded as a contract failure, not skipped).
DEFAULT_STACKS = [
    StackSpec("noop+proportional", "noop", "proportional", "none"),
    StackSpec("entity_planner+proportional", "entity_planner", "proportional", "none"),
    StackSpec("stock_aware+proportional", "stock_aware", "proportional", "none"),
    StackSpec("portable_producer+proportional", "portable_scorer", "proportional",
              "scale d3 P12"),
    StackSpec("stock_aware+portable_orderer", "stock_aware", "portable_scorer",
              "scale d3 P12"),
    StackSpec("portable_producer+portable_orderer", "portable_scorer", "portable_scorer",
              "scale d3 P12"),
]


def default_targets(depth: int, seeds: list[int]) -> list[TargetTopology]:
    """Train-size held-out scale seeds + one capacity-relabel target (the same-shape
    relabel axis). Size/out-of-family targets belong to later ladder rungs (C2/C3)."""
    targets = [TargetTopology(f"scale-d{depth}-s{s}", build_scale_config(s, depth=depth), depth)
               for s in seeds]
    s0 = seeds[0]
    targets.append(TargetTopology(
        f"scale-d{depth}-s{s0}-relabel",
        relabel_capacity_types(build_scale_config(s0, depth=depth)), depth))
    return targets


def main() -> None:
    """Emit the C-1 verification matrix CSV from available artifacts (local eval)."""
    import argparse

    ap = argparse.ArgumentParser(description="C-1 transfer/verification matrix")
    ap.add_argument("--producer-ckpt", default=None)
    ap.add_argument("--orderer-ckpt", default=None)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seeds", default="6-9")
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--out", default="transfer_matrix.csv")
    args = ap.parse_args()

    prod = ord_ = None
    if args.producer_ckpt:
        from ..train.producer_scorer import load_producer_score_model
        prod = load_producer_score_model(args.producer_ckpt)
    if args.orderer_ckpt:
        from ..train.orderer_scorer import load_scorer
        ord_ = load_scorer(args.orderer_ckpt)

    seeds = _parse_seeds(args.seeds)
    targets = default_targets(args.depth, seeds)
    rows = transfer_matrix(DEFAULT_STACKS, targets, args.eval_seeds, args.horizon,
                           producer_model=prod, orderer_model=ord_, log=print)
    write_matrix_csv(rows, args.out)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


if __name__ == "__main__":
    main()
