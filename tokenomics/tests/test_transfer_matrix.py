"""C-1 verification matrix gates (PRODUCT_AGNOSTIC ordered plan step 2).

The matrix must separate contract failure (model/checkpoint can't run on a target)
from policy quality. These tests pin that separation through the public runner.
"""

from __future__ import annotations

from tokenomics.eval.scale import build_scale_config
import csv

from pytest import approx

from tokenomics.eval.transfer_matrix import (
    StackSpec,
    TargetTopology,
    evaluate_stack_on_target,
    numeric_perturbation_targets,
    out_of_family_targets,
    producer_permutation_targets,
    size_transfer_targets,
    transfer_matrix,
    write_matrix_csv,
)


def _target(seed: int = 6, depth: int = 2) -> TargetTopology:
    return TargetTopology(name=f"scale-d{depth}-s{seed}",
                          config=build_scale_config(seed, depth=depth), depth=depth)


def test_deterministic_stack_row_reports_value_and_target_dims() -> None:
    stack = StackSpec(model_id="entity_planner+proportional",
                      producer_kind="entity_planner", orderer_kind="proportional")
    row = evaluate_stack_on_target(stack, _target(), eval_seeds=1, horizon=20)

    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"
    assert row["consumer_completed_units"] > 0
    # depth-2 default scale family: 4 products/layer, 4 producers/layer, 1 cap/layer
    assert row["target_products"] == 8
    assert row["target_producers"] == 8
    assert row["target_capacity_types"] == 2
    assert row["target_depth"] == 2


def test_portable_stack_without_checkpoint_is_contract_failure_not_policy() -> None:
    # No checkpoint supplied -> the portable scorer can't load. That is a CONTRACT
    # failure, recorded separately from policy quality, with no value reported.
    stack = StackSpec(model_id="portable+portable",
                      producer_kind="portable_scorer", orderer_kind="portable_scorer")
    row = evaluate_stack_on_target(stack, _target(), eval_seeds=1, horizon=20)

    assert row["runs_episode"] is False
    assert row["failure_class"] == "contract"
    assert row["producer_schema_loads"] is False
    assert row["orderer_schema_loads"] is False
    assert row["consumer_completed_units"] is None


def test_planner_gated_producer_is_a_distinct_policy_mode_not_a_new_schema() -> None:
    """The planner-gated producer (EXP-0017 P2/C3 deployment) is the SAME checkpoint and
    schema as `portable_scorer` — only the producer policy adapter differs (learned
    continuous scores, planner-owned investment gates). The matrix must expose it as a
    separate kind with a `producer_policy_mode` flag, never by silently flipping
    `portable_scorer`, and its contract flags must stay identical (it loads/runs wherever
    the plain portable producer does)."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import ProducerScoreModel

    model = ProducerScoreModel(hidden=16)
    target = _target()

    learned = evaluate_stack_on_target(
        StackSpec("portable_producer+proportional", "portable_scorer", "proportional"),
        target, eval_seeds=1, horizon=20, producer_model=model)
    gated = evaluate_stack_on_target(
        StackSpec("portable_producer_planner_gates+proportional",
                  "portable_scorer_planner_gates", "proportional"),
        target, eval_seeds=1, horizon=20, producer_model=model)

    assert learned["producer_policy_mode"] == "learned_gates"
    assert gated["producer_policy_mode"] == "planner_gates"
    # policy-mode distinction, NOT a load/schema distinction: same contract flags.
    assert gated["producer_schema_loads"] is True
    assert gated["producer_contract_compatible"] is True
    assert gated["runs_episode"] is True
    assert gated["failure_class"] == "none"


def test_planner_gated_tie_tolerance_defaults_to_zero_and_is_tunable_per_row() -> None:
    """Tie tolerance is an ablation knob, not a shipped behavior: the planner-gated
    producer deploys at tol 0.0 by default (planner gates are the validated fix; tie
    tolerance is non-monotonic across seeds). The runner records the tolerance each row
    ran with so EXP-0018 rows can be labeled by it, and it is overridable per run."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import ProducerScoreModel

    model = ProducerScoreModel(hidden=16)
    target = _target()
    gated = StackSpec("portable_producer_planner_gates+proportional",
                      "portable_scorer_planner_gates", "proportional")

    default = evaluate_stack_on_target(gated, target, eval_seeds=1, horizon=20,
                                       producer_model=model)
    tuned = evaluate_stack_on_target(gated, target, eval_seeds=1, horizon=20,
                                     producer_model=model, planner_gate_tie_rel_tol=0.05)
    learned = evaluate_stack_on_target(
        StackSpec("portable_producer+proportional", "portable_scorer", "proportional"),
        target, eval_seeds=1, horizon=20, producer_model=model)

    assert default["producer_tie_rel_tol"] == 0.0     # shipped default
    assert tuned["producer_tie_rel_tol"] == 0.05      # ablation override flows into the row
    assert learned["producer_tie_rel_tol"] == 0.0     # learned-gate mode keeps exact ties


def test_loaded_but_runtime_incompatible_model_is_a_contract_failure_row() -> None:
    # A checkpoint that "loads" (model object exists) but blows up against this target
    # at episode time must NOT raise out of the runner — it must become an explicit
    # contract-failure row so the matrix keeps the contract/policy split honest (the
    # case that arises once legacy fixed checkpoints / C2-C3 off-size targets land).
    class _Incompatible:  # resolves to a policy, raises when actually run
        pass

    stack = StackSpec(model_id="broken_portable+proportional",
                      producer_kind="portable_scorer", orderer_kind="proportional")
    row = evaluate_stack_on_target(stack, _target(), eval_seeds=1, horizon=20,
                                   producer_model=_Incompatible())

    assert row["producer_schema_loads"] is True       # the model object existed
    assert row["runs_episode"] is False               # but it could not run here
    assert row["failure_class"] == "contract"
    assert row["producer_contract_compatible"] is False
    assert row["consumer_completed_units"] is None
    assert "contract" in row["notes"].lower()


def test_transfer_matrix_attaches_practical_baseline_ratio() -> None:
    stacks = [
        StackSpec("entity_planner+proportional", "entity_planner", "proportional"),
        StackSpec("stock_aware+proportional", "stock_aware", "proportional"),
    ]
    rows = transfer_matrix(stacks, [_target()], eval_seeds=1, horizon=20)
    assert len(rows) == 2

    ep = next(r for r in rows if r["model_id"] == "entity_planner+proportional")
    # the practical baseline IS entity_planner+proportional -> ratio is exactly 1.0
    assert ep["ratio_vs_practical_baseline"] == approx(1.0)
    assert ep["baseline_consumer_completed_units"] > 0

    sa = next(r for r in rows if r["model_id"] == "stock_aware+proportional")
    assert sa["ratio_vs_practical_baseline"] == approx(
        sa["consumer_completed_units"] / ep["baseline_consumer_completed_units"], rel=1e-6)


def test_size_transfer_targets_widen_product_and_producer_axes() -> None:
    # C2 size transfer (ordered plan step 4): off ONE base seed, build larger-product,
    # larger-producer, and larger-both targets. Dims must scale on the widened axis only
    # (depth fixed) so a contract/scaling failure stays separable from the train-size rows.
    targets = size_transfer_targets(seed=6, depth=3, product_width=8, producer_width=8)
    by_name = {t.name: t for t in targets}
    assert len(targets) == 3

    prod = by_name["scale-d3-s6-prodW8"]
    # product-width 8 at depth 3 -> 24 products, producers stay 12 (4/level), caps = depth
    pp, ppr, pc = (len(prod.config.process_sheets),
                   len(prod.config.producer_configs),
                   len({t for s in prod.config.process_sheets for t in s.require}))
    assert (pp, ppr, pc, prod.depth) == (24, 12, 3, 3)

    agent = by_name["scale-d3-s6-agentW8"]
    ap, apr, ac = (len(agent.config.process_sheets),
                   len(agent.config.producer_configs),
                   len({t for s in agent.config.process_sheets for t in s.require}))
    assert (ap, apr, ac) == (12, 24, 3)  # products fixed, producers doubled

    both = by_name["scale-d3-s6-prodW8-agentW8"]
    bp, bpr = len(both.config.process_sheets), len(both.config.producer_configs)
    assert (bp, bpr) == (24, 24)

    # a deterministic stack must actually run on a widened target (contract-compatible)
    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, prod, eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"
    assert row["target_products"] == 24
    assert row["consumer_completed_units"] > 0


def test_numeric_perturbation_targets_preserve_dims_and_run() -> None:
    # P-1/O-1 step-8 "same-shape numeric perturbations": each target changes only numeric
    # values (degradation, demand quantity, capacity, cost) off one base seed, so dims are
    # identical to the base and a value change is a POLICY signal, never a contract one.
    base = build_scale_config(6, depth=3)
    base_dims = (len(base.process_sheets), len(base.producer_configs),
                 len({t for s in base.process_sheets for t in s.require}))

    targets = numeric_perturbation_targets(seed=6, depth=3)
    assert len(targets) >= 3  # at least degradation, quantity, capacity axes
    names = {t.name for t in targets}
    assert all("s6" in n for n in names)

    for t in targets:
        dims = (len(t.config.process_sheets), len(t.config.producer_configs),
                len({s_t for s in t.config.process_sheets for s_t in s.require}))
        assert dims == base_dims  # dims-preserving: same shape, only numbers moved

    # a deterministic stack must actually run on a perturbed target (contract-compatible)
    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, targets[0], eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"


def test_producer_permutation_targets_relabel_rows_without_changing_economy() -> None:
    # O-1 step-8 producer-row equivariance axis: permuting which sorted producer slot holds
    # which config must not change dims (same-shape relabel) and must stay runnable, so a
    # value mismatch vs the base seed reads as an equivariance failure, not a contract one.
    base = build_scale_config(6, depth=3)
    base_dims = (len(base.process_sheets), len(base.producer_configs),
                 len({t for s in base.process_sheets for t in s.require}))

    targets = producer_permutation_targets(seed=6, depth=3, n_perms=2)
    assert len(targets) == 2
    assert all("perm" in t.name for t in targets)

    for t in targets:
        dims = (len(t.config.process_sheets), len(t.config.producer_configs),
                len({s_t for s in t.config.process_sheets for s_t in s.require}))
        assert dims == base_dims

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, targets[0], eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"


def test_out_of_family_targets_are_multi_input_and_runnable() -> None:
    # C3 out-of-family (ordered plan step 6): the fork/join targets must leave the layered
    # single-input scale family (>=1 multi-input product) AND still run a deterministic
    # stack, so a contract failure here is attributable to the recipe axis, not a broken
    # fixture.
    targets = out_of_family_targets(seeds=[6, 7], depth=3, width=4, inputs_per_product=2)
    assert len(targets) == 2
    t = targets[0]
    assert "forkjoin" in t.name
    assert max(len(ps.inputs) for ps in t.config.process_sheets) == 2  # genuinely join

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, t, eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"
    assert row["consumer_completed_units"] > 0


def test_shared_capacity_targets_share_types_across_levels_and_run() -> None:
    # Step 10 broad C3 axis: the shared-capacity fixture leaves the scale family on the
    # CAPACITY axis (levels share types -> fewer cap types than depth), staying single-input.
    # A contract failure here is attributable to capacity sharing, not the recipe DAG.
    from tokenomics.eval.transfer_matrix import shared_capacity_targets

    targets = shared_capacity_targets(seeds=[6, 7], depth=3, width=4)
    assert len(targets) == 2
    t = targets[0]
    assert "sharedcap" in t.name
    caps = len({ct for s in t.config.process_sheets for ct in s.require})
    assert caps < 3  # fewer capacity types than levels: levels share

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, t, eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"
    assert row["consumer_completed_units"] > 0


def test_deep_join_targets_are_deeper_and_wider_than_forkjoin_and_run() -> None:
    # Step 10 broad C3 axis: deeper joins push BOTH recipe axes past the step-6 fork/join
    # probe (depth 5, 3-input joins). A failure here exercises recipe-graph depth/width.
    from tokenomics.eval.transfer_matrix import deep_join_targets

    targets = deep_join_targets(seeds=[6], depth=5, width=4, inputs_per_product=3)
    assert len(targets) == 1
    t = targets[0]
    assert "deepjoin" in t.name
    assert t.depth == 5
    assert max(len(ps.inputs) for ps in t.config.process_sheets) == 3  # wider join

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, t, eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"


def test_uneven_strength_targets_force_concentration_and_run() -> None:
    # Step 10 broad P3 axis: one dominant producer per level, the rest weak (guaranteed
    # extreme within-level capacity spread), so routing/investment must concentrate. A
    # value drop here is a producer/orderer policy signal, never a contract one.
    from tokenomics.eval.transfer_matrix import uneven_strength_targets

    targets = uneven_strength_targets(seeds=[6, 7], depth=3, width=4)
    assert len(targets) == 2
    t = targets[0]
    assert "uneven" in t.name
    caps = [next(iter(p.capacities.values())) for p in t.config.producer_configs]
    assert max(caps) / min(caps) >= 8  # genuinely uneven strengths

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, t, eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"


def test_scarcity_targets_are_dims_preserving_stress_axes_and_run() -> None:
    # Step 10 scarcity axis: the five single-axis stress variants off one base seed, each
    # dims-preserving (so a portable checkpoint loads unchanged) -> a value drop is a
    # scarcity/stress-behavior signal, not a contract one.
    from tokenomics.eval.transfer_matrix import scarcity_targets

    base = build_scale_config(6, depth=3)
    base_dims = (len(base.process_sheets), len(base.producer_configs),
                 len({t for s in base.process_sheets for t in s.require}))

    targets = scarcity_targets(seed=6, depth=3)
    assert len(targets) == 5
    names = {t.name for t in targets}
    for axis in ("low_funding", "low_inventory", "high_backlog", "degraded",
                 "running_investment"):
        assert any(axis in n for n in names), axis
    for t in targets:
        dims = (len(t.config.process_sheets), len(t.config.producer_configs),
                len({s_t for s in t.config.process_sheets for s_t in s.require}))
        assert dims == base_dims  # dims-preserving

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, targets[0], eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True
    assert row["failure_class"] == "none"


def test_degenerate_targets_cover_edge_economies_and_run() -> None:
    # Step 10 degenerate axis: edge economies at the boundary of the id-free contract
    # (single eligible producer, all-primal, no inventory expansion, no backlog, minimal).
    # Each must still RUN a deterministic stack so a later portable failure is attributable.
    from tokenomics.eval.transfer_matrix import degenerate_targets

    targets = degenerate_targets(seed=6, depth=3, width=4)
    assert len(targets) == 5
    names = {t.name for t in targets}
    for edge in ("single_producer", "primal_only", "no_inventory_expansion",
                 "no_backlog", "minimal"):
        assert any(edge in n for n in names), edge
    # primal_only is a depth-1 economy: its carried depth must reflect that, not the default
    primal = next(t for t in targets if "primal_only" in t.name)
    assert primal.depth == 1

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    for t in targets:  # every degenerate edge must be a runnable, valid economy
        row = evaluate_stack_on_target(stack, t, eval_seeds=1, horizon=20)
        assert row["runs_episode"] is True, t.name
        assert row["failure_class"] == "none", t.name


def test_fixture_matrix_targets_assemble_all_families_with_unique_names() -> None:
    # Step 10 broad C3/P3 sweep: one umbrella that gathers every reusable fixture family —
    # the per-seed builder families (shared-capacity, deep-join, uneven-strength) across all
    # seeds, plus the dims-preserving suite families (scarcity, degenerate) off the first
    # seed (like size-transfer). Names must be unique so no two cells collide in the CSV.
    from tokenomics.eval.transfer_matrix import fixture_matrix_targets

    targets = fixture_matrix_targets(seeds=[6, 7], depth=3, width=4)
    names = [t.name for t in targets]
    assert len(names) == len(set(names))  # no collisions

    families = ("sharedcap", "deepjoin", "uneven", "scarcity", "degen")
    for fam in families:
        assert any(fam in n for n in names), fam
    # builder families span both seeds; suite families ride the first seed only
    assert sum("sharedcap" in n for n in names) == 2
    assert sum("scarcity" in n for n in names) == 5   # one base seed, five axes
    assert sum("degen" in n for n in names) == 5

    stack = StackSpec("entity_planner+proportional", "entity_planner", "proportional")
    row = evaluate_stack_on_target(stack, targets[0], eval_seeds=1, horizon=20)
    assert row["runs_episode"] is True


def test_producer_audit_stacks_hold_orderer_fixed_and_vary_producer() -> None:
    # P-1 (step 8): the producer-side audit must hold the orderer fixed (a deterministic
    # baseline) and vary ONLY the producer across entity_planner, the deterministic
    # stock_aware bar, and both portable producer policy modes (learned vs planner gates),
    # so any value spread is attributable to the producer.
    from tokenomics.eval.transfer_matrix import PRODUCER_AUDIT_STACKS

    assert {s.orderer_kind for s in PRODUCER_AUDIT_STACKS} == {"proportional"}
    producer_kinds = {s.producer_kind for s in PRODUCER_AUDIT_STACKS}
    assert {"entity_planner", "portable_scorer", "portable_scorer_planner_gates"} <= producer_kinds
    assert "noop" in producer_kinds  # nontriviality floor


def test_orderer_audit_stacks_hold_producer_fixed_and_vary_orderer() -> None:
    # O-1 (step 8): the orderer-side audit must hold the producer fixed (a deterministic
    # baseline) and vary ONLY the orderer across the deterministic orderer baselines and
    # the portable scorer/allocator.
    from tokenomics.eval.transfer_matrix import ORDERER_AUDIT_STACKS

    varied = [s for s in ORDERER_AUDIT_STACKS if s.producer_kind != "noop"]
    assert {s.producer_kind for s in varied} == {"stock_aware"}
    orderer_kinds = {s.orderer_kind for s in varied}
    assert {"proportional", "portable_scorer"} <= orderer_kinds


def test_write_matrix_csv_round_trips_columns_and_values(tmp_path) -> None:
    stacks = [StackSpec("entity_planner+proportional", "entity_planner", "proportional")]
    rows = transfer_matrix(stacks, [_target()], eval_seeds=1, horizon=20)
    out = tmp_path / "matrix.csv"
    write_matrix_csv(rows, out)

    back = list(csv.DictReader(out.open()))
    assert len(back) == 1
    r = back[0]
    # the doc's contract/value/baseline columns must all be present as headers
    for col in ("model_id", "producer_kind", "orderer_kind", "target_topology",
                "target_products", "target_producers", "target_capacity_types",
                "target_depth", "producer_schema_loads", "orderer_schema_loads",
                "runs_episode", "failure_class", "consumer_completed_units",
                "dead_rate", "ratio_vs_practical_baseline",
                # full baseline schema (PRODUCT_AGNOSTIC "Baseline columns")
                "baseline_family", "baseline_model_id",
                "baseline_consumer_completed_units", "baseline_dead_rate",
                "ratio_vs_best_available_bar", "baseline_contract_compatible"):
        assert col in r
    assert r["model_id"] == "entity_planner+proportional"
    assert float(r["consumer_completed_units"]) > 0
    # the practical baseline ran on this target, so it is contract-compatible
    assert r["baseline_contract_compatible"] == "True"
