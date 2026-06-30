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
