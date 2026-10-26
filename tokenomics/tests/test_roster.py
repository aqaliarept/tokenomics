"""Gate baseline roster (design D5, add-measurement-gate-harness; gate-harness "Frozen baseline
roster", "Roster pruning uses development evidence only")."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.domain import load_config
from tokenomics.env.aec_env import DEFAULT_CONFIG
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.roster import (
    DEFAULT_ROSTER,
    DEVELOPMENT,
    JOINT_STACK,
    LEGACY,
    NEVER_ARGMAX_RULE,
    DevRow,
    EpisodeKey,
    Roster,
    RosterHashMismatch,
    build_roster_v1,
    contract_for,
    freeze_roster,
    load_roster,
    member_env,
    member_policies,
    prune_roster,
    roster_from_dict,
    rows_digest,
    save_roster,
)
from tokenomics.eval.runner import run_episode

SIZES = (12, 120)


def _rows(roster: Roster, values: dict[str, float], *, world: str = "w", size: int = 12,
          episode: int = 1000, default: float = 0.0, split: str = DEVELOPMENT) -> list[DevRow]:
    """One full episode: every member gets a row, `values` overrides `default`."""
    return [
        DevRow(world, split, size, episode, mid, values.get(mid, default))
        for mid in roster.member_ids
    ]


def _both_sizes(roster: Roster, small: dict[str, float], large: dict[str, float]) -> list[DevRow]:
    return _rows(roster, small, size=12) + _rows(roster, large, size=120, world="w2")


# --- Frozen baseline roster ---


def test_roster_v1_is_producing_cross_ordering_40_with_12_random() -> None:
    r = load_roster()
    assert len(r.members) == len(PRODUCING_BASELINES) * len(ORDERING_BASELINES) == 40
    assert sum(m.random for m in r.members) == 12
    assert {(m.producing, m.ordering) for m in r.members} == {
        (p, o) for p in PRODUCING_BASELINES for o in ORDERING_BASELINES
    }
    assert r.replicates == 5 and not r.frozen and r.pruning is None


def test_blend_0_5_is_a_roster_member() -> None:
    # Scenario "blend_0.5 is a roster member".
    r = load_roster()
    assert any(m.ordering == "blend_0.5" for m in r.members)


def test_committed_draft_equals_builder() -> None:
    assert DEFAULT_ROSTER.read_text() == build_roster_v1().to_json()
    assert load_roster().version == build_roster_v1().version


def test_contract_rule() -> None:
    # obs_* reads the legacy flat obs -> LEGACY; random never forces a contract.
    assert contract_for("obs_stock_aware", "proportional") == LEGACY
    assert contract_for("entity_planner", "obs_price") == LEGACY
    assert contract_for("random", "obs_price") == LEGACY
    assert contract_for("obs_stock_aware", "random") == LEGACY
    assert contract_for("random", "random") == JOINT_STACK
    assert contract_for("entity_planner", "blend_0.5") == JOINT_STACK
    r = load_roster()
    assert sum(m.contract == LEGACY for m in r.members) == 12  # 8 x obs_price + 5 x obs_stock_aware - 1


def test_replicates_only_for_random_members() -> None:
    r = load_roster()
    assert r.replicates_for(r.member("random/proportional")) == 5
    assert r.replicates_for(r.member("entity_planner/proportional")) == 1


def test_member_list_edit_without_new_hash_refused(tmp_path: Path) -> None:
    # Scenario "Roster edited without a new version": load raises, so the gate (which only gets
    # rosters through load_roster) cannot compute a hindsight baseline from it.
    d = json.loads(DEFAULT_ROSTER.read_text())
    d["members"] = d["members"][1:]
    p = tmp_path / "edited.json"
    p.write_text(json.dumps(d))
    with pytest.raises(RosterHashMismatch):
        load_roster(p)


@pytest.mark.parametrize("field,value", [("replicates", 3), ("frozen", True), ("name", "x")])
def test_any_field_edit_without_new_hash_refused(tmp_path: Path, field: str, value: object) -> None:
    d = json.loads(DEFAULT_ROSTER.read_text())
    d[field] = value
    p = tmp_path / "edited.json"
    p.write_text(json.dumps(d))
    with pytest.raises(RosterHashMismatch):
        load_roster(p)


def test_missing_hash_refused(tmp_path: Path) -> None:
    d = json.loads(DEFAULT_ROSTER.read_text())
    del d["content_hash"]
    p = tmp_path / "nohash.json"
    p.write_text(json.dumps(d))
    with pytest.raises(RosterHashMismatch):
        load_roster(p)


def test_save_load_roundtrip_and_version_tracks_content(tmp_path: Path) -> None:
    r = build_roster_v1()
    p = tmp_path / "r.json"
    save_roster(r, p)
    assert load_roster(p) == r
    assert build_roster_v1(replicates=3).version != r.version


def test_member_order_does_not_change_version() -> None:
    r = build_roster_v1()
    assert Roster(r.name, r.replicates, tuple(reversed(r.members))).version == r.version


def test_flag_inconsistent_with_policies_rejected() -> None:
    r = build_roster_v1()
    m = r.member("entity_planner/proportional")
    with pytest.raises(ValueError, match="contract"):
        Roster(r.name, r.replicates, (replace(m, contract=LEGACY),))
    with pytest.raises(ValueError, match="random"):
        Roster(r.name, r.replicates, (replace(m, random=True),))


def test_every_member_runs_under_its_contract() -> None:
    # Smoke (horizon 2, one seed): each member's policies act in its contract's env.
    cfg = load_config(DEFAULT_CONFIG)
    for m in load_roster().members:
        prod, order = member_policies(m)
        stats = run_episode(member_env(m, cfg, horizon=2), prod, order, seed=0)
        assert stats["cycles"] > 0, m.id


# --- Roster pruning uses development evidence only ---


def test_never_winning_stack_dropped_with_record() -> None:
    # Scenario "Never-winning stack dropped": winners on either size survive; the rest are
    # dropped and the file records the rule and the development runs.
    r = build_roster_v1()
    a, b = "entity_planner/proportional", "stock_aware/blend_0.5"
    rows = _both_sizes(r, {a: 10.0}, {b: 7.0})
    pruned = prune_roster(r, rows)
    assert pruned.member_ids == tuple(sorted((a, b)))
    rec = pruned.pruning
    assert rec is not None and rec.rule == NEVER_ARGMAX_RULE
    assert rec.parent_version == r.version
    assert set(rec.dropped) == set(r.member_ids) - {a, b}
    assert rec.episodes == (EpisodeKey("w", 12, 1000), EpisodeKey("w2", 120, 1000))
    assert rec.rows_digest == rows_digest(rows)
    assert pruned.version != r.version and not pruned.frozen
    # the record survives the file round trip and is covered by the hash
    assert roster_from_dict(json.loads(pruned.to_json())) == pruned


def test_ties_never_prune_a_co_winner() -> None:
    r = build_roster_v1()
    a, b = "entity_planner/proportional", "entity_planner/blend_0.5"
    pruned = prune_roster(r, _both_sizes(r, {a: 5.0, b: 5.0}, {a: 3.0}))
    assert set(pruned.member_ids) == {a, b}
    # an all-zero episode makes every member a winner: nothing dropped
    assert prune_roster(r, _both_sizes(r, {}, {})).member_ids == r.member_ids


def test_non_development_rows_rejected() -> None:
    r = build_roster_v1()
    rows = _both_sizes(r, {}, {}) + _rows(r, {}, world="s", split="sealed")
    with pytest.raises(ValueError, match="development"):
        prune_roster(r, rows)


def test_candidate_rows_rejected() -> None:
    r = build_roster_v1()
    rows = _both_sizes(r, {}, {}) + [DevRow("w", DEVELOPMENT, 12, 1000, "candidate", 99.0)]
    with pytest.raises(ValueError, match="non-member"):
        prune_roster(r, rows)


def test_absent_member_rejected() -> None:
    r = build_roster_v1()
    rows = _both_sizes(r, {}, {})
    with pytest.raises(ValueError, match="lacks rows"):
        prune_roster(r, rows[1:])


def test_duplicate_and_nonfinite_rows_rejected() -> None:
    r = build_roster_v1()
    rows = _both_sizes(r, {}, {})
    with pytest.raises(ValueError, match="duplicate"):
        prune_roster(r, rows + rows[:1])
    bad = [DevRow(x.world, x.split, x.size, x.episode, x.member, float("nan")) if i == 0 else x
           for i, x in enumerate(rows)]
    with pytest.raises(ValueError, match="non-finite"):
        prune_roster(r, bad)


def test_evidence_must_cover_both_sizes() -> None:
    r = build_roster_v1()
    with pytest.raises(ValueError, match="sizes"):
        prune_roster(r, _rows(r, {}, size=12))


def test_frozen_roster_refuses_pruning() -> None:
    r = freeze_roster(build_roster_v1())
    assert r.frozen and r.version != build_roster_v1().version
    with pytest.raises(ValueError, match="frozen"):
        prune_roster(r, _both_sizes(r, {}, {}))


def test_already_pruned_roster_refuses_second_pruning() -> None:
    r = build_roster_v1()
    pruned = prune_roster(r, _both_sizes(r, {}, {}))
    with pytest.raises(ValueError, match="already pruned"):
        prune_roster(pruned, _both_sizes(pruned, {}, {}))
