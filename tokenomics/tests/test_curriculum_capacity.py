"""Stage 4b Track A: capacity-binding mining (CURRICULUM_GAME_MASTER_APPROACH.md).

Track A reruns Stage-3 mining where CAPACITY, not demand, is the ceiling — the only
regime with headroom by construction. The pure, torch-free surface here is the ceiling
pre-flight cell filter: keep only grid cells whose no-slack capacity ceiling sits below
demand (a policy can fail because base capacity cannot meet demand without investment
skill), drop the structurally demand-bound cells the EXP-0033 audit already showed are
saturated. Tests exercise it without loading any checkpoint.
"""

from __future__ import annotations

import csv

from tokenomics.eval.curriculum import TaskRecord, write_archive
from tokenomics.eval.curriculum_capacity import (
    capacity_tight_cells, filter_archive_recoverable, is_recoverable,
    preflight_cells, recoverable_task_ids, write_preflight_report,
)
from tokenomics.eval.curriculum_replay import _load_archive


def _attr(task_id: str, channel: str, prod: float, ordr) -> dict:
    return {"task_id": task_id, "dominant_channel": channel,
            "producer_only_gain": prod, "orderer_only_gain": ordr}


def test_preflight_keeps_only_capacity_bound_cells() -> None:
    demand_bound = {"depth": 3, "producers_per_layer": 3}
    capacity_bound = {"depth": 3, "producers_per_layer": 3,
                      "quantity_max": 16, "capacity_scale": 0.25}
    kept, rows = preflight_cells(
        "build_scale_config", [demand_bound, capacity_bound],
        world_seeds=range(20, 30), eval_seeds=3, horizon=60)

    assert kept == [capacity_bound]              # demand-bound cell dropped (saturated)
    assert len(rows) == 2                        # every candidate cell reported
    by_binds = {r["capacity_binds"]: r for r in rows}
    assert by_binds[True]["params"] == capacity_bound
    assert by_binds[False]["params"] == demand_bound
    # the kept cell's capacity ceiling is below demand at every sampled world seed
    assert by_binds[True]["b2_over_b0_worst"] < 1.0
    assert by_binds[True]["binds_frac"] == 1.0


def test_preflight_drops_a_cell_that_only_binds_at_some_world_seeds() -> None:
    # THE LEAK: build_scale_config draws capacities per world seed, so a cell can be
    # capacity-bound at seed 0 yet demand-bound (saturated) at seed 1.  Mining sweeps the
    # whole range, so pre-flight must certify EVERY world seed, not just the first.
    leaker = {"depth": 3, "producers_per_layer": 3, "quantity_max": 12, "capacity_scale": 0.40}
    single = preflight_cells("build_scale_config", [leaker],
                             world_seeds=[0], eval_seeds=3, horizon=100)[0]
    swept = preflight_cells("build_scale_config", [leaker],
                            world_seeds=range(0, 10), eval_seeds=3, horizon=100)
    assert single == [leaker]                    # binds at world seed 0 alone
    assert swept[0] == []                         # but NOT across 0..9 -> dropped
    assert 0.0 < swept[1][0]["binds_frac"] < 1.0  # reported as a partial-bind cell


def test_preflight_margin_requires_meaningful_capacity_gap() -> None:
    # a cell only mildly below demand is dropped when a margin is demanded.
    cell = {"depth": 3, "producers_per_layer": 3, "quantity_max": 16, "capacity_scale": 0.25}
    loose, _ = preflight_cells("build_scale_config", [cell],
                               world_seeds=[20], eval_seeds=1, horizon=60, margin=0.0)
    strict, _ = preflight_cells("build_scale_config", [cell],
                                world_seeds=[20], eval_seeds=1, horizon=60, margin=0.99)
    assert loose == [cell]
    assert strict == []                          # 99% headroom demand not met -> dropped


def test_capacity_tight_cells_are_all_capacity_bound() -> None:
    # the shipped Track A grid must survive its own pre-flight ACROSS the mining seed
    # range (no saturated cells leak in at any world seed the mine will sample).
    cells = capacity_tight_cells()
    assert len(cells) >= 3
    kept, rows = preflight_cells("build_scale_config", cells,
                                 world_seeds=range(20, 120), eval_seeds=3, horizon=100)
    assert kept == cells
    assert all(r["binds_frac"] == 1.0 for r in rows)


def test_recoverability_gate_keeps_component_attributable_gaps() -> None:
    # a gap a producer- or orderer-swap closes is trainable -> recoverable.
    assert is_recoverable(_attr("p", "producer", prod=6.0, ordr=0.0))
    assert is_recoverable(_attr("o", "orderer", prod=0.0, ordr=4.0))
    assert is_recoverable(_attr("m", "mixed", prod=3.0, ordr=3.0))


def test_recoverability_gate_drops_joint_mismatch_gaps() -> None:
    # no single component closes it -> unlearnable by component distill (EXP-0031/0032b).
    assert not is_recoverable(_attr("noop", "no_positive_gap", prod=0.0, ordr=0.0))
    assert not is_recoverable(_attr("resid", "interaction_or_noise", prod=0.1, ordr=0.1))
    # right channel but the component barely moves it -> below the noise floor.
    assert not is_recoverable(_attr("tiny", "producer", prod=0.2, ordr=0.0),
                              min_component_gain=0.5)


def test_recoverable_task_ids_filters_an_attribution_matrix() -> None:
    rows = [
        _attr("keep-prod", "producer", 5.0, 0.0),
        _attr("keep-ord", "orderer", 0.0, 3.0),
        _attr("drop-mismatch", "interaction_or_noise", 0.1, 0.1),
        _attr("drop-nogap", "no_positive_gap", 0.0, 0.0),
    ]
    assert recoverable_task_ids(rows) == ["keep-prod", "keep-ord"]


def _mined_record(task_id: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id, task_family="random", generator_version="mine-v1",
        serializer_version="v1", world_seed=0, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": 3}},
        failure_class="policy_weakness", reference_status="solved")


def test_filter_archive_recoverable_writes_the_recoverable_subset(tmp_path) -> None:
    archive = tmp_path / "task_archive.jsonl"
    write_archive([_mined_record("keep-prod"), _mined_record("drop-mismatch"),
                   _mined_record("keep-ord")], archive)

    attr = tmp_path / "gap_attribution.csv"
    with attr.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "task_id", "dominant_channel", "producer_only_gain", "orderer_only_gain"])
        w.writeheader()
        w.writerow({"task_id": "keep-prod", "dominant_channel": "producer",
                    "producer_only_gain": "5.0", "orderer_only_gain": "0.0"})
        w.writerow({"task_id": "drop-mismatch", "dominant_channel": "interaction_or_noise",
                    "producer_only_gain": "0.1", "orderer_only_gain": ""})
        w.writerow({"task_id": "keep-ord", "dominant_channel": "orderer",
                    "producer_only_gain": "0.0", "orderer_only_gain": "3.0"})

    out = tmp_path / "recoverable_archive.jsonl"
    total, kept = filter_archive_recoverable(archive, attr, out)
    assert (total, kept) == (3, 2)
    kept_ids = [r.task_id for r in _load_archive(out)]
    assert kept_ids == ["keep-prod", "keep-ord"]      # mismatch task dropped


def test_write_preflight_report_tables_every_candidate_cell(tmp_path) -> None:
    cells = [{"depth": 3, "producers_per_layer": 3},                    # demand-bound
             {"depth": 3, "producers_per_layer": 3, "quantity_max": 16,
              "capacity_scale": 0.25}]                                  # capacity-bound
    out = tmp_path / "preflight.csv"
    kept = write_preflight_report("build_scale_config", cells, out,
                                  world_seeds=range(20, 30), eval_seeds=3, horizon=60)
    assert kept == [cells[1]]
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {r["capacity_binds"] for r in rows} == {"True", "False"}
