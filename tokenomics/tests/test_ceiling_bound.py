"""EXP-0033 offline ceiling-bound audit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.eval.ceiling_bound import (
    ceiling_bounds,
    demand_cap_units,
    episode_demand,
    load_observed_units,
    recipe_capacity_bundle,
    run_ceiling_audit,
)
from tokenomics.eval.curriculum import _episode_seeds, replay
from tokenomics.eval.curriculum_replay import _load_archive


def _producer(pid: str, cap_type: str, cap: int, produces: frozenset[int]) -> ProducingAgentConfig:
    return ProducingAgentConfig(
        id=pid,
        capacities={cap_type: cap},
        degradation=0,
        restoration=Restoration(require=1, restores=0, type=cap_type),
        upgrade=Upgrade(require=1, increases=0, type=cap_type),
        produces=produces,
    )


def _toy_config() -> Configuration:
    """A hand-checkable two-level economy.

    Top product 2 uses 5 units of c1 and two input units of product 1.  Product 1 uses
    2 units of c0, so B2 charges c1=5 and c0=4 per delivered top unit.
    """
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=1, require={"c0": 2}),
            ProcessSheet(product=2, require={"c1": 5}, inputs={1: 2}),
        ],
        producer_configs=[
            _producer("p0", "c0", 10, frozenset({1})),
            _producer("p1", "c1", 20, frozenset({2})),
        ],
        consumers=[ConsumerConfig(id="c", product=2, quantity_min=100, quantity_max=100)],
    )
    cfg.validate()
    return cfg


def test_b0_demand_cap_uses_scripted_consumer_quantity_max() -> None:
    assert demand_cap_units(_toy_config(), horizon=4) == 400.0
    assert episode_demand(_toy_config(), horizon=4, seed=123) == {2: 400}


def test_transitive_recipe_bundle_charges_input_products() -> None:
    cfg = _toy_config()
    assert recipe_capacity_bundle(cfg, 2, transitive=False) == {"c1": 5.0}
    assert recipe_capacity_bundle(cfg, 2, transitive=True) == {"c1": 5.0, "c0": 4.0}


def test_b0_b1_b2_hand_checked_toy_bounds() -> None:
    bounds = ceiling_bounds(_toy_config(), horizon=4)
    # Capacity cycles: c0 = 10*4 = 40; c1 = 20*4 = 80.  Direct top recipe binds at
    # 80/5 = 16 units; transitive input coupling binds at 40/4 = 10 units.
    assert bounds["B0"].units == 400.0
    assert bounds["B0"].binding_constraint == "demand"
    assert bounds["B1"].units == 16.0
    assert bounds["B1"].binding_constraint == "capacity:c1"
    assert bounds["B2"].units == 10.0
    assert bounds["B2"].binding_constraint == "capacity:c0"


def test_run_ceiling_audit_writes_exp_layout(tmp_path) -> None:
    archive = tmp_path / "archive.jsonl"
    matrix = tmp_path / "matrix.csv"
    rec_line = (
        '{"demand_trace":[],"episode_seed":0,"event_trace":[],"expected_claim_boundary":"in_claim",'
        '"failure_class":"policy_weakness","generator_version":"mine-v1","initial_state_spec":{},'
        '"legal":true,"notes":"","reference_status":"solved","replayable":true,'
        '"serializer_version":"v1","shock_trace":[],"task_family":"random",'
        '"task_id":"mine-build_scale_config-depth3-s0","world_seed":0,'
        '"world_spec":{"generator":"build_scale_config","params":{"depth":3}}}'
    )
    archive.write_text(rec_line + "\n")
    matrix.write_text(
        "task_id,learner_units,reference_units\n"
        "mine-build_scale_config-depth3-s0,10,11\n"
    )

    summary = run_ceiling_audit(archive, matrix, tmp_path / "EXP-0033-smoke", horizon=20)
    out = tmp_path / "EXP-0033-smoke"
    assert summary["tasks"] == 1
    assert summary["rows"] == 3
    assert (out / "summary.csv").exists()
    assert (out / "spec.json").exists()
    assert "B2 mean champion/bound ratio" in (out / "RESULT.md").read_text()


def test_exp0028_archive_bounds_dominate_observed_units() -> None:
    repo = Path(__file__).resolve().parents[2]
    archive = repo / "experiments/EXP-0028-topology-grid-mining/artifacts/task_archive.jsonl"
    matrix = repo / "experiments/EXP-0028-topology-grid-mining/artifacts/curriculum_matrix.csv"
    if not archive.exists() or not matrix.exists():
        pytest.skip("EXP-0028 archive artifacts not present")

    observed = load_observed_units(matrix)
    for record in _load_archive(archive):
        cfg = replay(record)
        best = max(observed[record.task_id]["champion_units"], observed[record.task_id]["bar_units"])
        bounds = ceiling_bounds(cfg, horizon=100, episode_seeds=_episode_seeds(record.episode_seed, 3))
        for tier, bound in bounds.items():
            assert bound.units + 0.01 >= best, (record.task_id, tier, bound.units, best)
