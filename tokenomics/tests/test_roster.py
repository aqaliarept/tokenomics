"""Gate baseline roster (design D5, add-measurement-gate-harness; gate-harness "Frozen baseline
roster", "Roster pruning uses development evidence only")."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.domain import load_config
from tokenomics.env.aec_env import DEFAULT_CONFIG
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, STOCHASTIC_BASELINES
from tokenomics.eval.gate import DEFAULT_PARAMS, PARAMS_V1, PARAMS_V2, load_side_effect_params
from tokenomics.eval.gate_cli import build_jobs
from tokenomics.eval.registry import (DEFAULT_REGISTRY, REGISTRY_V1, REGISTRY_V2, build_worlds_v1,
                                      build_worlds_v2, load_registry)
from tokenomics.eval.roster import (
    DEFAULT_ROSTER,
    PILOT_V3_NAME,
    PROMOTED_V3,
    ROSTER_V1,
    ROSTER_V2,
    ROSTER_V2_ORDERING,
    ROSTER_V3_ORDERING,
    DEVELOPMENT,
    JOINT_STACK,
    LEGACY,
    NEVER_ARGMAX_RULE,
    ROSTER_V1_ORDERING,
    DevRow,
    EpisodeKey,
    Roster,
    RosterHashMismatch,
    build_pilot_roster_v3,
    build_roster_v1,
    build_roster_v2,
    build_roster_v3,
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
from tokenomics.eval.seeding import replicate_seed

SIZES = (12, 42)


def _rows(roster: Roster, values: dict[str, float], *, world: str = "w", size: int = 12,
          episode: int = 1000, default: float = 0.0, split: str = DEVELOPMENT) -> list[DevRow]:
    """One full episode: every member gets a row, `values` overrides `default`."""
    return [
        DevRow(world, split, size, episode, mid, values.get(mid, default))
        for mid in roster.member_ids
    ]


def _both_sizes(roster: Roster, small: dict[str, float], large: dict[str, float]) -> list[DevRow]:
    return _rows(roster, small, size=12) + _rows(roster, large, size=42, world="w2")


# --- Frozen baseline roster ---


def test_roster_v1_is_producing_cross_ordering_40_with_12_random() -> None:
    r = build_roster_v1()  # the draft definition; the packaged file is its pruned, frozen form
    assert len(r.members) == len(PRODUCING_BASELINES) * len(ROSTER_V1_ORDERING) == 40
    assert sum(m.random for m in r.members) == 12
    assert {(m.producing, m.ordering) for m in r.members} == {
        (p, o) for p in PRODUCING_BASELINES for o in ROSTER_V1_ORDERING
    }
    # roster v2's shock-reactive router (add-shocks-i-drift D9) is not a v1 member
    # nor are the routers gate v3 promotes (add-gate-v3 D1)
    assert set(ROSTER_V1_ORDERING) == set(ORDERING_BASELINES) - {"shock_reactive", *PROMOTED_V3}
    assert r.replicates == 5 and not r.frozen and r.pruning is None


def test_blend_0_5_is_a_roster_member() -> None:
    # Scenario "blend_0.5 is a roster member".
    r = load_roster(ROSTER_V1)
    assert any(m.ordering == "blend_0.5" for m in r.members)


def test_packaged_roster_is_the_frozen_pruning_of_the_draft() -> None:
    """Task 11.1: the packaged roster is the EXP-0044 attempt-2 freeze of the 40-member draft:
    frozen, carrying its pruning record, and holding a subset of the draft's members."""
    r, draft = load_roster(ROSTER_V1), build_roster_v1()
    assert r.frozen and r.pruning is not None
    assert set(r.member_ids) < set(draft.member_ids)
    assert r.replicates == draft.replicates


def test_contract_rule() -> None:
    # obs_* reads the legacy flat obs -> LEGACY; random never forces a contract.
    assert contract_for("obs_stock_aware", "proportional") == LEGACY
    assert contract_for("entity_planner", "obs_price") == LEGACY
    assert contract_for("random", "obs_price") == LEGACY
    assert contract_for("obs_stock_aware", "random") == LEGACY
    assert contract_for("random", "random") == JOINT_STACK
    assert contract_for("entity_planner", "blend_0.5") == JOINT_STACK
    r = build_roster_v1()
    assert sum(m.contract == LEGACY for m in r.members) == 12  # 8 x obs_price + 5 x obs_stock_aware - 1


def test_replicates_only_for_random_members() -> None:
    r = build_roster_v1()
    assert r.replicates_for(r.member("random/proportional")) == 5
    assert r.replicates_for(r.member("entity_planner/proportional")) == 1


def test_member_list_edit_without_new_hash_refused(tmp_path: Path) -> None:
    # Scenario "Roster edited without a new version": load raises, so the gate (which only gets
    # rosters through load_roster) cannot compute a hindsight baseline from it.
    d = json.loads(ROSTER_V1.read_text())
    d["members"] = d["members"][1:]
    p = tmp_path / "edited.json"
    p.write_text(json.dumps(d))
    with pytest.raises(RosterHashMismatch):
        load_roster(p)


@pytest.mark.parametrize("field,value", [("replicates", 3), ("frozen", False), ("name", "x")])
def test_any_field_edit_without_new_hash_refused(tmp_path: Path, field: str, value: object) -> None:
    d = json.loads(ROSTER_V1.read_text())
    d[field] = value
    p = tmp_path / "edited.json"
    p.write_text(json.dumps(d))
    with pytest.raises(RosterHashMismatch):
        load_roster(p)


def test_missing_hash_refused(tmp_path: Path) -> None:
    d = json.loads(ROSTER_V1.read_text())
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
    for m in load_roster(ROSTER_V1).members:
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
    assert rec.episodes == (EpisodeKey("w", 12, 1000), EpisodeKey("w2", 42, 1000))
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


# --- roster v3 draft (add-gate-v3 D1) ---

DRAFTS = Path(__file__).resolve().parents[2] / "openspec" / "changes" / "add-gate-v3" / "drafts"
# Recorded at base commit 43f77c6, before the promotion: the v1 / v2 drafts use explicit ordering
# lists, so adding baselines must not move them; the frozen files keep their content hashes.
DRAFT_HASHES = {
    "roster-v1": "8791e24b2760a24bc2b40ef588578ee96b7e8e93502b76c50ef0d2fac6b83239",
    "roster-v2": "a050c99e4960eb422b9eda34fd4a0de56d259629c934ccf28e4ffcb33da816fe",
    "worlds-v2": "3032147f1f17233703f486c66e4c56e006c026399364dd1cbaead5b46bcfceb5",
}
FROZEN_HASHES = {  # tests/test_frozen_v2.py FROZEN_V1 / FROZEN_V2
    "worlds_v1": "566946fe0c0363ec888f9a58bfe7bd480bcc948e049c93e9923088e2cab911e3",
    "roster_v1": "c2832dbb44d26601f279fc04c0c483c543c5a5fcdc8cb2231090fc03f76a3731",
    "params_v1": "e6651b2ebe541b18ade0a8904f27b0a08844e2a4f8ee0e231ca062514472f2c1",
    "worlds_v2": "7b09f536e9880bc31370694f80eaca828587621eb5daa16fdf64e80ed05ee7ff",
    "roster_v2": "5050508ecb25f0eda0ae0a61b0dde4ba948a6e9b0d901cd14cde28e0db161f38",
    "params_v2": "e2aee97ac38524d709862bc956c43940413ac0cb44c846304a4b8c3d324c3a99",
}


def test_roster_v3_draft_is_88_members_32_stochastic() -> None:
    """Scenario "Roster v3 draft": every producing baseline × every v2 ordering baseline and each
    promoted router, no trained stack; 32 stochastic members (11 with the `random` producer, 7
    producers × 3 stochastic routers), so 216 runs per world-episode at R = 5."""
    r = build_roster_v3()
    assert ROSTER_V3_ORDERING == ROSTER_V2_ORDERING + PROMOTED_V3
    assert set(ROSTER_V3_ORDERING) == set(ORDERING_BASELINES)
    assert {(m.producing, m.ordering) for m in r.members} == {
        (p, o) for p in PRODUCING_BASELINES for o in ROSTER_V3_ORDERING}
    assert len(r.members) == 88 and sum(m.random for m in r.members) == 32
    assert sum(r.replicates_for(m) for m in r.members) == 216
    assert r.name == "roster-v3" and r.version_tag == 3 and r.replicates == 5
    assert not r.frozen and r.pruning is None and r.screened_on is None
    assert all(m.random == (m.producing in STOCHASTIC_BASELINES or m.ordering in STOCHASTIC_BASELINES)
               for m in r.members)
    # obs_stock_aware x a promoted router runs LEGACY (D1 "Legacy contract")
    assert {m.ordering for m in r.members if m.producing == "obs_stock_aware"
            and m.contract == LEGACY} == set(ROSTER_V3_ORDERING)


def test_stochastic_flag_refused_when_wrong() -> None:
    m = build_roster_v3().member("entity_planner/makeable_weighted_whole")
    assert m.random
    with pytest.raises(ValueError, match="random"):
        Roster("roster-v3", 5, (replace(m, random=False),))


def test_pilot_roster_is_entity_planner_times_every_ordering_baseline() -> None:
    p = build_pilot_roster_v3()
    assert {(m.producing, m.ordering) for m in p.members} == {
        ("entity_planner", o) for o in ROSTER_V3_ORDERING}
    assert sum(p.replicates_for(m) for m in p.members) == 23  # D3 step 0
    assert p.name == PILOT_V3_NAME and "pilot, not a gate roster" in p.name
    assert p.version_tag == 3 and not p.frozen and p.pruning is None


def test_drafts_on_disk_are_the_builders() -> None:
    """The Drafts files (written by 1.1) are exactly `build_roster_v3()` / the pilot, and load."""
    if not DRAFTS.exists():  # the openspec tree is not shipped to pods
        pytest.skip("Drafts dir not present")
    for name, roster in (("roster_v3_draft.json", build_roster_v3()),
                         ("roster_v3_pilot_ep.json", build_pilot_roster_v3())):
        assert (DRAFTS / name).read_text() == roster.to_json(), name
        assert load_roster(DRAFTS / name).version == roster.version


def test_v1_v2_drafts_frozen_hashes_and_defaults_unchanged() -> None:
    assert build_roster_v1().version == DRAFT_HASHES["roster-v1"]
    assert build_roster_v2().version == DRAFT_HASHES["roster-v2"]
    assert build_worlds_v2().version == DRAFT_HASHES["worlds-v2"]
    build_worlds_v1()  # still builds
    got = {
        "worlds_v1": load_registry(REGISTRY_V1).version, "roster_v1": load_roster(ROSTER_V1).version,
        "params_v1": load_side_effect_params(PARAMS_V1).version,
        "worlds_v2": load_registry(REGISTRY_V2).version, "roster_v2": load_roster(ROSTER_V2).version,
        "params_v2": load_side_effect_params(PARAMS_V2).version,
    }
    assert got == FROZEN_HASHES
    # stored `random` flags of v1 / v2 unchanged: only `random` makes a v1 / v2 member stochastic
    for r in (load_roster(ROSTER_V1), load_roster(ROSTER_V2), build_roster_v2()):
        assert all(m.random == ("random" in (m.producing, m.ordering)) for m in r.members)
    assert (DEFAULT_ROSTER, DEFAULT_REGISTRY, DEFAULT_PARAMS) == (ROSTER_V2, REGISTRY_V2, PARAMS_V2)


def test_stochastic_members_get_replicate_seeds() -> None:
    """gate-harness "Stochastic promoted router": a stochastic v3 member runs R jobs with the
    hashed action seeds (world, episode, replicate); a deterministic one runs once, unseeded."""
    roster = build_roster_v3()
    world = load_registry(REGISTRY_V2).gate_worlds(DEVELOPMENT)[0]
    jobs = build_jobs([world], roster, None, episodes_per_world=1)
    assert len(jobs) == 216
    ep = world.episode_seeds[0]
    seeds: dict[str, list] = {}
    for j in jobs:
        seeds.setdefault(j["subject"], []).append((j["replicate"], j["action_seed"]))
    for m in roster.members:
        want = ([(r, replicate_seed(world.id, ep, r)) for r in range(5)] if m.random
                else [(0, None)])
        assert sorted(seeds[m.id]) == want, m.id
    assert all(s != ep for j in jobs if (s := j["action_seed"]) is not None)


@pytest.mark.parametrize("module", ["baselines", "routing_diagnostics", "roster", "gate_cli",
                                    "rotation_routers"])
def test_fresh_interpreter_imports_module_first(module: str) -> None:
    """D1 "No import cycle": each module imports first in a fresh interpreter."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", f"import tokenomics.eval.{module}"], cwd=root,
                   check=True, capture_output=True, text=True)


def test_rotation_routers_imports_neither_baselines_nor_diagnostics() -> None:
    import tokenomics.eval.rotation_routers as rr

    tree = ast.parse(Path(rr.__file__).read_text())
    names = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(x.split(".")[-1] in ("baselines", "routing_diagnostics") for x in names), names
