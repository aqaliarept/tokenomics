"""Replayable world records: TaskRecord serialize/replay and archive IO.

A record serializes to one jsonl line and back, and replays to the exact economy via
its generator-ref (no hidden process state) — the world format the EXP-0042 teacher
replays from the EXP-0036 archive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.eval.scale import build_scale_config
from tokenomics.eval.task_archive import REPRODUCERS, TaskRecord, load_archive, replay


def _record(task_id: str = "stress-scale-d3-s6", generator: str = "build_scale_config") -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        task_family="scale",
        generator_version="v1",
        serializer_version="v1",
        world_seed=6,
        episode_seed=0,
        world_spec={"generator": generator, "params": {"depth": 3}},
    )


def test_task_record_roundtrips_through_json_line() -> None:
    rec = _record()
    line = rec.to_json_line()
    assert "\n" not in line
    assert TaskRecord.from_json_line(line) == rec


def test_replay_rebuilds_the_exact_economy_from_generator_ref() -> None:
    cfg = replay(_record())
    # Replay must reproduce the exact pure builder output (no hidden process state).
    assert cfg == build_scale_config(6, depth=3)
    cfg.validate()  # a legal, runnable economy


def test_replay_rejects_an_unvetted_generator() -> None:
    with pytest.raises(KeyError, match="not a vetted reproducer"):
        replay(_record(generator="os.system"))


def test_load_archive_reads_every_nonblank_line(tmp_path: Path) -> None:
    a, b = _record("a"), _record("b")
    path = tmp_path / "task_archive.jsonl"
    path.write_text(a.to_json_line() + "\n\n" + b.to_json_line() + "\n")
    assert load_archive(path) == [a, b]


# One vetted params set per whitelisted generator (the former curriculum stress suite).
_REPRODUCER_CASES = [
    ("build_scale_config", {"depth": 3}),
    ("build_forkjoin_config", {"depth": 3}),
    ("build_shared_capacity_config", {"depth": 3}),
    ("build_uneven_strength_config", {"depth": 3}),
    ("build_deep_join_config", {"depth": 5}),
    *[("scarcity_variant", {"variant": v, "depth": 3})
      for v in ("low_funding", "low_inventory", "high_backlog", "degraded", "running_investment")],
    *[("degenerate_variant", {"variant": v, "depth": 3})
      for v in ("single_producer", "primal_only", "no_inventory_expansion", "no_backlog", "minimal")],
]


def test_every_reproducer_has_a_replay_case() -> None:
    assert {g for g, _ in _REPRODUCER_CASES} == set(REPRODUCERS)


@pytest.mark.parametrize("generator,params", _REPRODUCER_CASES)
def test_every_reproducer_replays_a_legal_deterministic_economy(generator: str, params: dict) -> None:
    rec = TaskRecord(
        task_id=f"{generator}-{params}", task_family="stress", generator_version="v1",
        serializer_version="v1", world_seed=0, episode_seed=0,
        world_spec={"generator": generator, "params": params},
    )
    cfg = replay(rec)  # replay() validates the economy
    assert cfg == replay(TaskRecord.from_json_line(rec.to_json_line()))
