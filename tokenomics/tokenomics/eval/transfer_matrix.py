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


def _producer_arch(kind: str, model: object | None) -> str:
    """Architecture id of the loaded portable producer checkpoint. Empty for
    deterministic-baseline producer kinds and when no checkpoint is loaded."""
    if kind not in ("portable_scorer", "portable_scorer_planner_gates") or model is None:
        return ""
    from ..train.producer_scorer import producer_portable_schema

    return producer_portable_schema().architecture_version


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
        "producer_arch": _producer_arch(stack.producer_kind, producer_model),
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
    "model_id", "producer_kind", "producer_arch", "producer_policy_mode", "producer_tie_rel_tol",
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


# P-1 producer audit (ordered plan step 8): hold the orderer fixed at the deterministic
# `proportional` baseline and vary ONLY the producer — floor, the practical `entity_planner`
# bar, the deterministic `stock_aware` bar, and BOTH portable producer policy modes (learned
# vs planner gates). So any value spread on a target is attributable to the producer.
# Shape-compatible legacy flat producers are unwired (no checkpoint) and contract-
# incompatible off-size, so they are recorded as omitted in the writeup, not run here.
PRODUCER_AUDIT_STACKS = [
    StackSpec("noop+proportional", "noop", "proportional", "none"),
    StackSpec("entity_planner+proportional", "entity_planner", "proportional", "none"),
    StackSpec("stock_aware+proportional", "stock_aware", "proportional", "none"),
    StackSpec("portable_producer+proportional", "portable_scorer", "proportional",
              "scale d3 P12"),
    StackSpec("portable_producer_planner_gates+proportional",
              "portable_scorer_planner_gates", "proportional", "scale d3 P12"),
]

# O-1 orderer audit (ordered plan step 8): hold the producer fixed at the deterministic
# `stock_aware` baseline and vary ONLY the orderer across the deterministic orderer
# baselines and the portable scorer/allocator. The `noop` floor row keeps the nontriviality
# check; legacy product-stream orderers are same-shape references only (unwired here).
ORDERER_AUDIT_STACKS = [
    StackSpec("noop+proportional", "noop", "proportional", "none"),
    StackSpec("stock_aware+even_split", "stock_aware", "even_split", "none"),
    StackSpec("stock_aware+proportional", "stock_aware", "proportional", "none"),
    StackSpec("stock_aware+obs_price", "stock_aware", "obs_price", "none"),
    StackSpec("stock_aware+portable_orderer", "stock_aware", "portable_scorer",
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


def numeric_perturbation_targets(seed: int, *, depth: int = 3) -> list[TargetTopology]:
    """P-1/O-1 same-shape NUMERIC perturbation axis (ordered plan step 8): single-axis,
    dims-preserving transforms off `build_scale_config(seed)` — harsher degradation,
    unseen demand quantity, leaner producers, dearer investment costs (reusing
    `eval.generalization`). None adds or removes a product, producer, or capacity type, so
    a value change on these targets is a POLICY-quality signal, never a contract one (every
    portable cell stays loadable). One numeric axis per target keeps a failure attributable."""
    from .generalization import (perturb_costs, perturb_degradation,
                                 perturb_producer_capacity, perturb_quantity)
    base = build_scale_config(seed, depth=depth)
    qmax = max(c.quantity_max for c in base.consumers) + 3
    variants = {
        "degr2x": perturb_degradation(base, 2.0),
        f"qty{qmax}": perturb_quantity(base, qmax),
        "lean0.7": perturb_producer_capacity(base, 0.7),
        "cost1.5x": perturb_costs(base, 1.5),
    }
    return [TargetTopology(f"scale-d{depth}-s{seed}-{tag}", cfg, depth)
            for tag, cfg in variants.items()]


def producer_permutation_targets(seed: int, *, depth: int = 3, width: int = 4,
                                 n_perms: int = 1, perm_seed: int = 0) -> list[TargetTopology]:
    """Same-shape PRODUCER-ROW permutation axis (O-1 equivariance check, ordered plan
    step 8): relabel which sorted producer slot holds which config without changing the
    economy (`permute_scale_config`, product slots fixed). Dims, capacity-type set, and
    `produces` are preserved, so a correct row-equivariant orderer/producer must reproduce
    the base seed's value up to deterministic tolerance — a mismatch is an equivariance
    failure, never a contract one."""
    import random

    from .scale import permute_scale_config
    base = build_scale_config(seed, depth=depth, products_per_layer=width,
                              producers_per_layer=width)
    rng = random.Random(perm_seed)
    out: list[TargetTopology] = []
    for i in range(n_perms):
        qp = list(range(width))
        rng.shuffle(qp)
        out.append(TargetTopology(
            f"scale-d{depth}-s{seed}-permP{i}",
            permute_scale_config(base, product_perm=list(range(width)), producer_perm=qp),
            depth))
    return out


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


def shared_capacity_targets(seeds: list[int], *, depth: int = 3,
                            width: int = 4) -> list[TargetTopology]:
    """Step-10 broad C3 axis: one shared-capacity target per seed. Levels SHARE capacity
    types (`max(1,depth-1)` types < depth levels) while staying single-input, so a failure
    here isolates the capacity-sharing axis from the recipe DAG (covered by fork/join)."""
    from .fixtures import build_shared_capacity_config
    return [
        TargetTopology(f"sharedcap-d{depth}-w{width}-s{s}",
                       build_shared_capacity_config(s, depth=depth, width=width), depth)
        for s in seeds
    ]


def deep_join_targets(seeds: list[int], *, depth: int = 5, width: int = 4,
                      inputs_per_product: int = 3) -> list[TargetTopology]:
    """Step-10 broad C3 axis: one deeper/wider fork/join target per seed (depth 5, 3-input
    joins vs the step-6 probe's depth 3, 2-input). Pushes the recipe-graph depth and join
    width together, so a failure here isolates recipe-graph scale, not a new mechanism."""
    from .fixtures import build_deep_join_config
    return [
        TargetTopology(f"deepjoin-d{depth}-w{width}-in{inputs_per_product}-s{s}",
                       build_deep_join_config(s, depth=depth, width=width,
                                              inputs_per_product=inputs_per_product), depth)
        for s in seeds
    ]


def uneven_strength_targets(seeds: list[int], *, depth: int = 3,
                            width: int = 4) -> list[TargetTopology]:
    """Step-10 broad P3 axis: one uneven-strength target per seed — every level has one
    dominant producer and the rest weak (guaranteed >= 12x spread), single-input and
    one-type-per-level otherwise. Strength dispersion is the only out-of-family axis, so a
    value drop here reads as routing/investment-concentration quality, not a contract one."""
    from .fixtures import build_uneven_strength_config
    return [
        TargetTopology(f"uneven-d{depth}-w{width}-s{s}",
                       build_uneven_strength_config(s, depth=depth, width=width), depth)
        for s in seeds
    ]


def scarcity_targets(seed: int, *, depth: int = 3) -> list[TargetTopology]:
    """Step-10 scarcity axis: the five single-axis stress variants of `build_scale_config
    (seed)` from `fixtures.scarcity_suite` (lean fund, tight inventory, heavy backlog,
    harsh decay, lean-and-decaying running-investment start). Each is dims-preserving, so a
    portable checkpoint loads unchanged and a value drop reads as scarcity behavior."""
    from .fixtures import scarcity_suite
    base = build_scale_config(seed, depth=depth)
    return [TargetTopology(f"scarcity-d{depth}-s{seed}-{axis}", cfg, depth)
            for axis, cfg in scarcity_suite(base).items()]


def degenerate_targets(seed: int, *, depth: int = 3,
                       width: int = 4) -> list[TargetTopology]:
    """Step-10 degenerate axis: the five edge economies from `fixtures.degenerate_suite`
    (single eligible producer, all-primal, no inventory expansion, no backlog, minimal).
    `primal_only` collapses to a depth-1 chain, so its carried depth is 1; the rest keep
    `depth`. Each probes a boundary of the producer/orderer contract, not its typical point."""
    from .fixtures import degenerate_suite
    suite = degenerate_suite(seed, depth=depth, width=width)
    return [TargetTopology(f"degen-s{seed}-{name}", cfg,
                           1 if name == "primal_only" else depth)
            for name, cfg in suite.items()]


def fixture_matrix_targets(seeds: list[int], *, depth: int = 3,
                           width: int = 4) -> list[TargetTopology]:
    """Step-10 broad C3/P3 sweep: every reusable fixture family in one target list. The
    per-seed builder families (shared-capacity, deeper-join, uneven-strength) span all
    `seeds`; the dims-preserving suite families (scarcity, degenerate) ride the FIRST seed
    only, matching the size-transfer convention (one base economy, many single-axis
    variants). Names are family-prefixed and unique so no two cells collide in the CSV."""
    s0 = seeds[0]
    return (shared_capacity_targets(seeds, depth=depth, width=width)
            + deep_join_targets(seeds, width=width)
            + uneven_strength_targets(seeds, depth=depth, width=width)
            + scarcity_targets(s0, depth=depth)
            + degenerate_targets(s0, depth=depth, width=width))


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
    ap.add_argument("--audit", choices=["producer", "orderer"], default=None,
                    help="run the P-1 (producer) or O-1 (orderer) dynamic baseline audit "
                         "(ordered plan step 8): pick the matching audit stack set and "
                         "assemble the step-8 target sweep — held-out seeds + same-shape "
                         "relabel + numeric perturbations + larger product/producer (plus "
                         "a producer-row permutation for the orderer audit).")
    ap.add_argument("--out-of-family", action="store_true",
                    help="use the C3 fork/join multi-input targets (one per seed) instead "
                         "of the scale-family targets (ordered plan step 6)")
    ap.add_argument("--fixtures", action="store_true",
                    help="use the broad C3/P3 fixture-family sweep (shared-capacity, "
                         "deeper-join, uneven-strength across all seeds + scarcity and "
                         "degenerate suites off the first seed) as the targets, in either "
                         "the audit or joint stack mode (ordered plan step 10)")
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
    if args.audit:
        # P-1/O-1 step-8 sweep: held-out seeds + same-shape relabel (default_targets),
        # same-shape numeric perturbations, and larger product/producer (size transfer),
        # off the first seed. The orderer audit adds a producer-row permutation
        # (equivariance). One self-contained sweep so contract vs policy stays separable.
        stacks = PRODUCER_AUDIT_STACKS if args.audit == "producer" else ORDERER_AUDIT_STACKS
        s0 = seeds[0]
        if args.fixtures:
            # Step 10: run the chosen audit stack set on the broad fixture-family sweep
            # instead of the step-8 same-shape sweep, so producer-only / orderer-only rows
            # are classified on the out-of-family / scarcity / degenerate fixtures too.
            targets = fixture_matrix_targets(seeds, depth=args.depth)
        else:
            targets = default_targets(args.depth, seeds)
            targets += numeric_perturbation_targets(s0, depth=args.depth)
            targets += size_transfer_targets(s0, depth=args.depth,
                                             product_width=args.product_width,
                                             producer_width=args.producer_width)
            if args.audit == "orderer":
                targets += producer_permutation_targets(s0, depth=args.depth)
        rows = transfer_matrix(stacks, targets, args.eval_seeds, args.horizon,
                               producer_model=prod, orderer_model=ord_,
                               planner_gate_tie_rel_tol=args.tie_rel_tol, log=print)
        write_matrix_csv(rows, args.out)
        print(f"\nwrote {len(rows)} rows -> {args.out}")
        return

    if args.fixtures:
        targets = fixture_matrix_targets(seeds, depth=args.depth)
    elif args.out_of_family:
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
