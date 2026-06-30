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


# Deployment default for the planner-gated producer mode. Planner-owned investment gates
# are the validated EXP-0017 fix; the replenishment tie tolerance is NOT shipped on
# (default 0.0 = exact ties) because it is non-monotonic across fork/join seeds — it
# helps s9 but hurts s8, with the s8 loss outweighing the s9 gain by mean ratio. It stays
# a tunable ablation knob (CLI `--tie-rel-tol`, labeled per row), not a default behavior.
PLANNER_GATE_TIE_REL_TOL = 0.0


def producer_policy_mode(kind: str) -> str:
    """Which investment-gate ownership a producer kind deploys, for the matrix CSV.

    Same checkpoint + schema for both portable kinds; only the policy adapter differs
    (`learned_gates` = learned BCE gates; `planner_gates` = planner-owned gates + tie
    tolerance, the EXP-0017 P2/C3 deployment-safe mode). Deterministic/legacy kinds are
    `n/a`."""
    if kind == "portable_scorer":
        return "learned_gates"
    if kind == "portable_scorer_planner_gates":
        return "planner_gates"
    return "n/a"


def _resolve_producer(
    kind: str, model: object | None, tie_rel_tol: float = PLANNER_GATE_TIE_REL_TOL
) -> tuple[Policy | None, bool]:
    if kind in PRODUCING_BASELINES:
        return PRODUCING_BASELINES[kind], True
    if kind in ("portable_scorer", "portable_scorer_planner_gates"):
        if model is None:
            return None, False
        from ..train.producer_scorer import scorer_producing_policy
        planner_gates = kind == "portable_scorer_planner_gates"
        return scorer_producing_policy(
            model,
            planner_gates=planner_gates,
            replenish_tie_rel_tol=tie_rel_tol if planner_gates else 0.0,
        ), True
    return None, False


def _row_tie_rel_tol(kind: str, tie_rel_tol: float) -> float:
    """The replenishment tie tolerance a row ran with (0.0 unless planner-gated)."""
    return tie_rel_tol if kind == "portable_scorer_planner_gates" else 0.0


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
                             orderer_model: object | None = None,
                             planner_gate_tie_rel_tol: float = PLANNER_GATE_TIE_REL_TOL) -> dict:
    """One matrix row: resolve the stack's policies, record schema/contract flags, and
    (if both sides load) average value over `eval_seeds` episodes on the joint env.

    `planner_gate_tie_rel_tol` tunes the replenishment tie tolerance for the planner-gated
    producer kind only (an ablation knob; the row records what it ran with)."""
    products, producers, caps = _target_dims(target.config)
    prod_pol, prod_loads = _resolve_producer(stack.producer_kind, producer_model,
                                             planner_gate_tie_rel_tol)
    order_pol, order_loads = _resolve_orderer(stack.orderer_kind, orderer_model)

    row: dict = {
        "model_id": stack.model_id,
        "producer_kind": stack.producer_kind,
        "producer_policy_mode": producer_policy_mode(stack.producer_kind),
        "producer_tie_rel_tol": _row_tie_rel_tol(stack.producer_kind, planner_gate_tie_rel_tol),
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
                    planner_gate_tie_rel_tol: float = PLANNER_GATE_TIE_REL_TOL,
                    log=lambda *a, **k: None) -> list[dict]:
    """The full matrix: every stack x target cell, each row carrying its value ratio vs
    the practical `entity_planner + proportional` baseline measured on the SAME target
    (so contract failures and value failures stay separable per row).

    `planner_gate_tie_rel_tol` tunes the planner-gated producer's replenishment tie
    tolerance (ablation knob; each row is labeled with the value it ran)."""
    rows: list[dict] = []
    for target in targets:
        base = evaluate_stack_on_target(PRACTICAL_BASELINE, target, eval_seeds, horizon)
        base_cu = base["consumer_completed_units"]
        for stack in stacks:
            row = evaluate_stack_on_target(stack, target, eval_seeds, horizon,
                                           producer_model=producer_model,
                                           orderer_model=orderer_model,
                                           planner_gate_tie_rel_tol=planner_gate_tie_rel_tol)
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
    "model_id", "producer_kind", "producer_policy_mode", "producer_tie_rel_tol",
    "orderer_kind", "train_topology", "target_topology",
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
    # Planner-gated producer mode (EXP-0017 fix): same checkpoint, planner-owned
    # investment gates + tie-tolerant replenishment. Kept alongside the learned-gate rows
    # so the original behavior stays a regression/diagnosis bar, not overwritten.
    StackSpec("portable_producer_planner_gates+proportional",
              "portable_scorer_planner_gates", "proportional", "scale d3 P12"),
    StackSpec("stock_aware+portable_orderer", "stock_aware", "portable_scorer",
              "scale d3 P12"),
    StackSpec("portable_producer+portable_orderer", "portable_scorer", "portable_scorer",
              "scale d3 P12"),
    StackSpec("portable_producer_planner_gates+portable_orderer",
              "portable_scorer_planner_gates", "portable_scorer", "scale d3 P12"),
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


def size_transfer_targets(seed: int, *, depth: int = 3,
                          product_width: int = 8, producer_width: int = 8) -> list[TargetTopology]:
    """C2 size-transfer axis (ordered plan step 4): off ONE base seed, widen the product
    count, the producer count, then both — relative to the train family (width 4). Depth
    (and so capacity-type count) is held fixed so a contract/scaling failure on a wider
    target stays separable from the train-size rows. The portable scorers are id-free, so
    the SAME checkpoints are expected to load and run unchanged on every entry here."""
    return [
        TargetTopology(
            f"scale-d{depth}-s{seed}-prodW{product_width}",
            build_scale_config(seed, depth=depth, products_per_layer=product_width), depth),
        TargetTopology(
            f"scale-d{depth}-s{seed}-agentW{producer_width}",
            build_scale_config(seed, depth=depth, producers_per_layer=producer_width), depth),
        TargetTopology(
            f"scale-d{depth}-s{seed}-prodW{product_width}-agentW{producer_width}",
            build_scale_config(seed, depth=depth, products_per_layer=product_width,
                               producers_per_layer=producer_width), depth),
    ]


def out_of_family_targets(seeds: list[int], *, depth: int = 3, width: int = 4,
                          inputs_per_product: int = 2) -> list[TargetTopology]:
    """C3 out-of-family axis (ordered plan step 6): one fork/join, multi-input target per
    seed. These leave the layered single-input scale family (every composite product joins
    `inputs_per_product` lower products), so a contract failure here is attributable to the
    recipe DAG, not product/producer count. The id-free scorers are expected to load and
    run unchanged — that is exactly the dominant open transfer claim under test."""
    from .fixtures import build_forkjoin_config
    return [
        TargetTopology(
            f"forkjoin-d{depth}-w{width}-in{inputs_per_product}-s{s}",
            build_forkjoin_config(s, depth=depth, width=width,
                                  inputs_per_product=inputs_per_product), depth)
        for s in seeds
    ]


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
    ap.add_argument("--size-transfer", action="store_true",
                    help="append C2 size-transfer targets (larger product/producer/both) "
                         "off the first seed (ordered plan step 4)")
    ap.add_argument("--product-width", type=int, default=8)
    ap.add_argument("--producer-width", type=int, default=8)
    ap.add_argument("--out-of-family", action="store_true",
                    help="use the C3 fork/join multi-input targets (one per seed) instead "
                         "of the scale-family targets (ordered plan step 6)")
    ap.add_argument("--inputs-per-product", type=int, default=2)
    ap.add_argument("--tie-rel-tol", type=float, default=PLANNER_GATE_TIE_REL_TOL,
                    help="planner-gated producer's replenishment tie tolerance (ablation "
                         "knob; default 0.0 = exact ties. Non-monotonic across fork/join "
                         "seeds, so it ships off; rows are labeled with the value used).")
    args = ap.parse_args()

    prod = ord_ = None
    if args.producer_ckpt:
        from ..train.producer_scorer import load_producer_score_model
        prod = load_producer_score_model(args.producer_ckpt)
    if args.orderer_ckpt:
        from ..train.orderer_scorer import load_scorer
        ord_ = load_scorer(args.orderer_ckpt)

    seeds = _parse_seeds(args.seeds)
    if args.out_of_family:
        targets = out_of_family_targets(seeds, depth=args.depth,
                                        inputs_per_product=args.inputs_per_product)
    else:
        targets = default_targets(args.depth, seeds)
    if args.size_transfer:
        targets += size_transfer_targets(seeds[0], depth=args.depth,
                                         product_width=args.product_width,
                                         producer_width=args.producer_width)
    rows = transfer_matrix(DEFAULT_STACKS, targets, args.eval_seeds, args.horizon,
                           producer_model=prod, orderer_model=ord_,
                           planner_gate_tie_rel_tol=args.tie_rel_tol, log=print)
    write_matrix_csv(rows, args.out)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


if __name__ == "__main__":
    main()
