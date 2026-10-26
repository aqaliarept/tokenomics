"""Legacy measurement invariance (gate-harness "Legacy measurement unchanged").

The M1 gate harness adds per-order accounting, side-effect metrics and an `action_seed`
to episode runs (design D2, D5). None of that may move a number the project already
reports. This test replays a fixed grid of (fixture world, env contract, baseline pair,
episode seed) cells and compares every value recorded in `data/legacy_goldens.json`,
captured on the tree before the harness landed (commit 7c34122), plus the
`build_scale_config(seed)` output for seeds 0-3.

Comparison is exact (floats round-trip through JSON repr) and covers only what the
goldens record: keys added later (new `episode_stats` fields, new dataclass fields) pass,
while a changed value or a missing recorded key fails. Plain data dicts inside a record
(`capacities`, `require`, `inputs`, ...) are encoded as closed `[key, value]` pair lists, so
an added entry there is a changed value, not a tolerated new field. Random baselines run with fixed
episode seeds, so the default seeding of `run_episode` is pinned too.

Regenerate ONLY on a tree whose legacy behaviour is known-good:
    ../.venv/bin/python tests/test_legacy_invariance.py --write
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Callable, Collection

import pytest

from tokenomics.domain.config import Configuration, load_config
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.aec_env import DEFAULT_CONFIG
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.fixtures import build_forkjoin_config
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.runner import run_episode
from tokenomics.eval.scale import build_scale_config

GOLDENS = Path(__file__).resolve().parent / "data" / "legacy_goldens.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
HORIZON = 30
SCALE_SEEDS = (0, 1, 2, 3)

# Every `env.episode_stats` key on the capture tree, plus the two `run_episode` adds.
LEGACY_EPISODE_STATS_KEYS = (
    "cycles", "dead", "consumer_completed", "consumer_completed_units",
    "investment_completed", "replenishment_completed", "consumer_rejected",
    "consumer_rejected_units", "investment_rejected", "replenishment_rejected",
    "in_progress", "holding_units", "return",
)
RUN_EPISODE_EXTRA_KEYS = ("cap_final", "cap_min")

WorldFn = Callable[[], Configuration]
ContractFn = Callable[[Configuration, int], TokenomicsAECEnv]

WORLDS: dict[str, WorldFn] = {
    "phase1": lambda: load_config(DEFAULT_CONFIG),
    "competition": lambda: load_config(FIXTURES / "competition.json"),
    "complex": lambda: load_config(FIXTURES / "complex.json"),
    "composite": lambda: load_config(FIXTURES / "composite.json"),
    "random_demand": lambda: load_config(FIXTURES / "random_demand.json"),
    "scale0": lambda: build_scale_config(0),
    "forkjoin0": lambda: build_forkjoin_config(0),
}
SMALL_WORLDS = ("phase1", "competition", "complex", "composite", "random_demand")
LARGE_WORLDS = ("scale0", "forkjoin0")


def _legacy_env(config: Configuration, horizon: int) -> TokenomicsAECEnv:
    return TokenomicsAECEnv(config=config, horizon=horizon)


CONTRACTS: dict[str, ContractFn] = {"legacy": _legacy_env, "joint": joint_stack_env}

# The baseline names on the capture tree. The grid is pinned to these, not read from the live
# dicts: a baseline added later (e.g. `blend_0.5`, design D5) has no legacy number to protect,
# and must not demand golden cells the capture tree never had.
LEGACY_PRODUCING = (
    "noop", "always_restore", "always_both", "restore_when_low", "stock_aware",
    "entity_planner", "obs_stock_aware", "random",
)
LEGACY_ORDERING = ("even_split", "proportional", "obs_price", "random")

# The joint contract is 2-10x slower per episode than legacy, so (to stay well under the
# 60 s budget) it runs a pinned pair subset instead of the full 8x4 cross product: every
# producing baseline against `proportional`, plus every ordering baseline at least once.
JOINT_PAIRS = tuple((p, "proportional") for p in LEGACY_PRODUCING) + (
    ("entity_planner", "even_split"), ("entity_planner", "obs_price"),
    ("obs_stock_aware", "obs_price"), ("stock_aware", "random"), ("random", "random"),
)
# 12-producer worlds: the planner bar, obs-only stack, early death (noop) and random pairs.
LARGE_JOINT_PAIRS = (
    ("entity_planner", "proportional"), ("obs_stock_aware", "obs_price"),
    ("noop", "even_split"), ("stock_aware", "proportional"),
    ("entity_planner", "random"), ("random", "random"),
)

Cell = tuple[str, str, str, str, int]  # (world, contract, producing, ordering, seed)


def _all_pairs() -> list[tuple[str, str]]:
    return [(p, o) for p in LEGACY_PRODUCING for o in LEGACY_ORDERING]


def _random_pairs() -> list[tuple[str, str]]:
    return [(p, o) for p, o in _all_pairs() if "random" in (p, o)]


def capture_cells() -> list[Cell]:
    """The cell grid captured into the goldens. Seed 0 everywhere; seed 7 on the
    randomized-demand worlds pins the demand RNG and random-policy action seeding."""
    cells: list[Cell] = []
    for w in SMALL_WORLDS:
        cells += [(w, "legacy", p, o, 0) for p, o in _all_pairs()]
        cells += [(w, "joint", p, o, 0) for p, o in JOINT_PAIRS]
    for w in LARGE_WORLDS:
        cells += [(w, "legacy", p, o, 0) for p, o in _all_pairs()]
        cells += [(w, "joint", p, o, 0) for p, o in LARGE_JOINT_PAIRS]
    cells += [("random_demand", "legacy", p, o, 7) for p, o in _random_pairs()]
    cells += [("random_demand", "joint", p, o, 7) for p, o in JOINT_PAIRS if "random" in (p, o)]
    cells += [("scale0", "legacy", p, o, 7) for p, o in _random_pairs()]
    cells += [("scale0", "legacy", "entity_planner", "proportional", 7),
              ("scale0", "joint", "entity_planner", "proportional", 7),
              ("scale0", "joint", "random", "random", 7)]
    return cells


def cell_id(cell: Cell) -> str:
    return "|".join(str(x) for x in cell)


def parse_cell_id(cid: str) -> Cell:
    w, c, p, o, s = cid.split("|")
    return (w, c, p, o, int(s))


def _to_json(value: Any) -> Any:
    """JSON-normal form: numpy scalars -> Python, dataclass -> field dict (open: fields may
    be added), plain dict -> sorted `[key, value]` pair list (closed: an added entry is a
    change; keys keep their type), set -> sorted list, tuple -> list. A json round-trip
    keeps floats exact (repr)."""
    def default(o: Any) -> Any:
        if hasattr(o, "item") and callable(o.item):
            return o.item()
        raise TypeError(f"not JSON-serializable: {type(o).__name__}")
    return json.loads(json.dumps(_plain(value), default=default))


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return [[_plain(k), _plain(v)] for k, v in sorted(value.items(), key=_key_order)]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _key_order(item: tuple[Any, Any]) -> tuple[str, str]:
    return (type(item[0]).__name__, repr(item[0]))


def run_cell(cell: Cell, stats_keys: Collection[str] | None = None) -> dict[str, Any]:
    """One episode, exactly as legacy callers run it: `run_episode(env, prod, order, seed)`.
    `stats_keys` keeps only those stats before serializing, so a key added later with a
    value `_to_json` cannot encode never reaches it; a missing recorded key stays missing."""
    world, contract, prod, order, seed = cell
    env = CONTRACTS[contract](WORLDS[world](), HORIZON)
    stats = run_episode(env, PRODUCING_BASELINES[prod], ORDERING_BASELINES[order], seed)
    if stats_keys is not None:
        stats = {k: v for k, v in stats.items() if k in stats_keys}
    # The record and the stats dict stay open dicts: later tasks add `episode_stats` keys.
    return {"stats": {str(k): _to_json(v) for k, v in stats.items()},
            "capacity_trace": _to_json(list(env.capacity_trace))}


def scale_config_record(seed: int) -> Any:
    return _to_json(build_scale_config(seed))


def assert_recorded_match(golden: Any, actual: Any, path: str) -> None:
    """Every value the golden records must be present and identical (type included, so
    True != 1 and 3 != 3.0). Dict keys the golden lacks are allowed: later tasks ADD keys.
    Only the stats dict and dataclass-field dicts reach here as dicts (see `_to_json`)."""
    if isinstance(golden, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        for k, v in golden.items():
            assert k in actual, f"{path}: recorded key {k!r} disappeared"
            assert_recorded_match(v, actual[k], f"{path}.{k}")
    elif isinstance(golden, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(actual) == len(golden), f"{path}: length {len(actual)} != golden {len(golden)}"
        for i, (g, a) in enumerate(zip(golden, actual)):
            assert_recorded_match(g, a, f"{path}[{i}]")
    else:
        assert type(actual) is type(golden) and actual == golden, (
            f"{path}: {actual!r} ({type(actual).__name__}) != golden {golden!r} "
            f"({type(golden).__name__})")


def _load_goldens() -> dict[str, Any]:
    with GOLDENS.open() as f:
        return json.load(f)


_GOLDENS = _load_goldens() if GOLDENS.exists() else {"cells": {}, "scale_config": {}}


def test_goldens_cover_the_capture_grid_and_every_legacy_key() -> None:
    assert _GOLDENS.get("horizon") == HORIZON
    assert set(_GOLDENS["cells"]) == {cell_id(c) for c in capture_cells()}
    assert set(_GOLDENS["scale_config"]) == {str(s) for s in SCALE_SEEDS}
    want = set(LEGACY_EPISODE_STATS_KEYS) | set(RUN_EPISODE_EXTRA_KEYS)
    for cid, rec in _GOLDENS["cells"].items():
        assert set(rec["stats"]) == want, cid
        assert len(rec["capacity_trace"]) == rec["stats"]["cycles"], cid


@pytest.mark.parametrize("cid", sorted(_GOLDENS["cells"]))
def test_legacy_episode_unchanged(cid: str) -> None:
    golden = _GOLDENS["cells"][cid]
    actual = run_cell(parse_cell_id(cid), stats_keys=set(golden["stats"]))
    assert_recorded_match(golden, actual, cid)


@pytest.mark.parametrize("seed", SCALE_SEEDS)
def test_scale_generator_unchanged(seed: int) -> None:
    assert_recorded_match(_GOLDENS["scale_config"][str(seed)], scale_config_record(seed),
                          f"build_scale_config({seed})")


def test_assert_recorded_match_allows_new_keys_but_not_changes() -> None:
    golden = {"a": 1, "b": [0.5, True]}
    assert_recorded_match(golden, {"a": 1, "b": [0.5, True], "new": 3}, "x")
    for bad in ({"b": [0.5, True]}, {"a": 2, "b": [0.5, True]}, {"a": 1.0, "b": [0.5, True]},
                {"a": 1, "b": [0.5, 1]}, {"a": 1, "b": [0.5]}):
        with pytest.raises(AssertionError):
            assert_recorded_match(golden, bad, "x")


@dataclasses.dataclass
class _Sheet:
    product: int
    require: dict[str, int]
    inputs: dict[int, int]


def test_plain_dict_entries_are_closed_but_dataclass_fields_open() -> None:
    golden = _to_json(_Sheet(10, {"capacity-1": 6}, {0: 1}))
    assert_recorded_match(golden, {**golden, "new_field": 1}, "x")
    for bad in (_Sheet(10, {"capacity-1": 6, "capacity-extra": 5}, {0: 1}),
                _Sheet(10, {"capacity-1": 6}, {0: 1, 1: 1}),
                _Sheet(10, {"capacity-1": 6}, {}),
                _Sheet(10, {"capacity-1": 6}, {"0": 1})):
        with pytest.raises(AssertionError):
            assert_recorded_match(golden, _to_json(bad), "x")


def _one_record_per_line(out: dict[str, Any]) -> str:
    """Valid JSON with one cell / scale config per line: compact, and a diff names the cell."""
    def block(d: dict[str, Any]) -> str:
        rows = [f"  {json.dumps(k)}: {json.dumps(d[k], sort_keys=True)}" for k in sorted(d)]
        return "{\n" + ",\n".join(rows) + "\n }"
    return (f'{{\n "horizon": {json.dumps(out["horizon"])},\n'
            f' "cells": {block(out["cells"])},\n'
            f' "scale_config": {block(out["scale_config"])}\n}}\n')


def write_goldens() -> None:
    """Capture the goldens from the current tree. Refuses if `episode_stats` has keys
    outside the pinned legacy set, so the file records exactly the pre-harness keys."""
    cells: dict[str, Any] = {}
    for cell in capture_cells():
        rec = run_cell(cell)
        extra = set(rec["stats"]) - set(LEGACY_EPISODE_STATS_KEYS) - set(RUN_EPISODE_EXTRA_KEYS)
        missing = (set(LEGACY_EPISODE_STATS_KEYS) | set(RUN_EPISODE_EXTRA_KEYS)) - set(rec["stats"])
        if extra or missing:
            raise SystemExit(f"{cell_id(cell)}: stats keys differ from legacy set "
                             f"(extra={sorted(extra)}, missing={sorted(missing)})")
        cells[cell_id(cell)] = rec
    out = {
        "horizon": HORIZON,
        "cells": cells,
        "scale_config": {str(s): scale_config_record(s) for s in SCALE_SEEDS},
    }
    GOLDENS.parent.mkdir(parents=True, exist_ok=True)
    GOLDENS.write_text(_one_record_per_line(out))
    assert _load_goldens() == out, "goldens did not round-trip exactly"
    print(f"wrote {len(cells)} cells + {len(SCALE_SEEDS)} scale configs to {GOLDENS}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python tests/test_legacy_invariance.py --write")
    write_goldens()
