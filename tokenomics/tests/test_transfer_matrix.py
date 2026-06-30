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
    out_of_family_targets,
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
