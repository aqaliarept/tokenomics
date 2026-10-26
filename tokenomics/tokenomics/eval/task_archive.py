"""Replayable world records: `TaskRecord`, its vetted generator-refs, and archive IO.

A record reproduces an economy from its **generator-ref** (`world_spec = {"generator",
"params"}`), not an inlined world dump: `replay()` re-calls a pure, seed-deterministic
builder from the `REPRODUCERS` whitelist, so champion, candidate and baselines all replay
the identical world. This is the world format the EXP-0036 archive uses and the EXP-0042
teacher replays.

Extracted from the closed curriculum game-master (experiments/HISTORY.md §7); the mining,
reference-bar and replay-training machinery is preserved on the `archive-1` branch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..domain.config import Configuration
from . import fixtures, scale


@dataclass(frozen=True)
class TaskRecord:
    """One replayable world, serialized as a self-contained generator-ref.

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


def load_archive(path: str | Path) -> list[TaskRecord]:
    """Read a `task_archive.jsonl` back into TaskRecords (self-contained generator-refs)."""
    with Path(path).open() as fh:
        return [TaskRecord.from_json_line(line) for line in fh if line.strip()]
