"""Headroom atlas driver (add-m3-development-audit task 4.5, design D12; development-audit
"Per-world headroom" ... "Atlas evidence does not change the gate")."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import pytest

from tokenomics.eval import atlas
from tokenomics.eval.atlas import (
    BLIND,
    CONFIRM,
    FLOOR_SE,
    HARD,
    HELDOUT,
    HELDOUT_REGION,
    OPEN,
    REACH,
    SCREEN,
    TRAIN,
    VALIDATION,
    Bin,
    Candidate,
    Spread,
    WorldInfo,
    analyse,
    anchor_worlds,
    arm_table,
    bound_violations,
    classify,
    coverage_bins,
    episode_values,
    measures,
    noise_floor,
    pooled_spreads,
    run_atlas,
    run_atlas_job,
    select,
    split_worlds,
    train_worlds,
)
from tokenomics.eval.gate_cli import CONTRACT_INVALID
from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
from tokenomics.eval.routing_audit import build_jobs
from tokenomics.eval.routing_audit import world_record as fanout_record
from tokenomics.eval.seeding import replicate_seed
from tokenomics.eval.world_space import episode_seeds

REPO = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO / "tokenomics" / "tokenomics" / "eval" / "registry"
EXP0050_SPEC = REPO / "experiments" / "EXP-0050-routing-vs-fanout" / "spec.json"

# synthetic arms: two blind (one stochastic), two reach (one stochastic), one reported
ARMS = [{"subject": "bar_d", "side": BLIND, "random": False},
        {"subject": "bar_r", "side": BLIND, "random": True},
        {"subject": "reach_d", "side": REACH, "random": False},
        {"subject": "reach_r", "side": REACH, "random": True},
        {"subject": "portable", "side": atlas.REPORTED, "random": False}]


def _row(world: str, ep: int, subject: str, units: float, rep: int = 0, *, b2: float = 1e4,
         base: float = 1e4, dead: bool = False, cycles: int = 100, ordered: float = 1e4,
         fulfilled: float | None = None) -> dict[str, str]:
    return {"world": world, "episode": str(ep), "subject": subject, "replicate": str(rep),
            "units": repr(float(units)), "dead": "true" if dead else "false",
            "ordered_units": repr(ordered),
            "fulfilled_units": repr(units if fulfilled is None else fulfilled),
            "trace_capacity_ratio_trace": json.dumps([1.0] * cycles),
            "bound_b2": repr(b2), "bound_base": repr(base)}


def _world(world: str, units: dict[str, list[list[float]]], **kw: object) -> list[dict]:
    """units: subject -> per episode -> per replicate."""
    return [_row(world, e, s, u, r, **kw) for s, eps in units.items()
            for e, reps in enumerate(eps) for r, u in enumerate(reps)]


def _info(world: str, size: int = 12, choice: str = "sparse_f2", differing: bool = True,
          pattern: str = "iid") -> WorldInfo:
    return WorldInfo(id=world, size=size, choice=choice, pattern=pattern, shocks=False,
                     differing=differing)


def _measure(rows: list[dict], world: str = "w", fixed: tuple | None = None) -> atlas.Measures:
    return measures(world, episode_values(rows)[world], ARMS, fixed)


def _class(rows: list[dict], spread: Spread | None, info: WorldInfo | None = None,
           mode: str = CONFIRM) -> str:
    m = _measure(rows)
    _, floor = noise_floor(m, info or _info("w"), spread, FLOOR_SE[mode])
    return classify(m, floor)


# a stratum spread with ample df
SPREAD = Spread(sigma_ep=2.0, df_ep=40, sigma_rep=2.0, df_rep=40)


# --- per-world headroom (spec "Per-world headroom") ---


def test_better_policy_reachable_is_hard() -> None:
    rows = _world("w", {"bar_d": [[100], [102]], "bar_r": [[90, 91], [92, 93]],
                        "reach_d": [[150], [152]], "reach_r": [[120, 121], [119, 118]]},
                  ordered=1000.0)
    m = _measure(rows)
    assert (m.bar_arm, m.reach_arm) == ("bar_d", "reach_d")
    assert m.gain == pytest.approx(50.0) and m.d == [50.0, 50.0]
    assert _class(rows, SPREAD) == HARD


def test_everyone_fails_is_not_hard() -> None:
    rows = _world("w", {"bar_d": [[2], [1]], "bar_r": [[1, 0], [2, 1]],
                        "reach_d": [[3], [2]], "reach_r": [[0, 1], [1, 0]]},
                  b2=500.0, base=500.0, ordered=500.0)
    assert _class(rows, Spread(5.0, 40, 5.0, 40)) != HARD


def test_gain_within_the_noise_is_not_hard() -> None:
    rows = _world("w", {"bar_d": [[100], [104]], "reach_d": [[103], [107]]}, ordered=1000.0)
    m = _measure(rows)
    se, floor = noise_floor(m, _info("w"), Spread(5.0, 40, None, 0), 3.0)
    assert m.gain == pytest.approx(3.0) and floor == pytest.approx(3 * 5 / math.sqrt(2))
    assert classify(m, floor) != HARD


def test_gain_not_consistent_across_episodes_is_not_hard() -> None:
    rows = _world("w", {"bar_d": [[100], [100]], "reach_d": [[200], [95]]}, ordered=1000.0)
    m = _measure(rows)
    _, floor = noise_floor(m, _info("w"), Spread(1.0, 40, None, 0), 3.0)
    assert m.gain == pytest.approx(47.5) and m.gain > floor
    assert classify(m, floor) != HARD


def test_sign_consistency_needs_every_replicate() -> None:
    rows = _world("w", {"bar_r": [[100, 140], [100, 100]], "reach_d": [[130], [130]]},
                  ordered=1000.0)
    m = _measure(rows)
    assert m.d_reps == [[30.0, -10.0], [30.0, 30.0]]
    assert not atlas.sign_consistent(m)


# --- noise floor (spec "Noise floor is pooled") ---


def test_noise_floor_arithmetic() -> None:
    # differing world, n = 2, stochastic bar arm with R = 2
    rows = _world("w", {"bar_r": [[100, 102], [101, 99]], "reach_d": [[200], [300]]},
                  ordered=1e4)
    m = _measure(rows)
    assert m.stochastic == {"bar_r": 2}
    se, floor = noise_floor(m, _info("w"), Spread(6.0, 40, 4.0, 40), 3.0)
    assert se == pytest.approx(math.sqrt(36 + 16 / 2) / math.sqrt(2))
    assert floor == pytest.approx(max(3 * se, 0.01 * m.bar))
    # an exact world (cyclic, no shocks, deterministic argmax arms): only the 1% term
    exact = _world("x", {"bar_d": [[400]], "reach_d": [[405]]}, ordered=1e4)
    mx = _measure(exact, "x")
    se, floor = noise_floor(mx, _info("x", differing=False), None, 3.0)
    assert se == 0.0 and floor == pytest.approx(4.0)
    assert classify(mx, floor) == HARD
    # the screen's multiplier is 2
    assert FLOOR_SE == {SCREEN: 2.0, CONFIRM: 3.0}
    # a needed spread with no df: infinite floor, never hard
    se, floor = noise_floor(m, _info("w"), Spread(None, 0, None, 0), 3.0)
    assert math.isinf(floor)


def test_two_episodes_that_agree_by_chance_use_the_pooled_spread() -> None:
    lucky = _world("lucky", {"bar_d": [[100], [100]], "reach_d": [[110], [110.001]]},
                   ordered=1e4)
    others = []
    infos = {"lucky": _info("lucky", choice="sparse_f4")}
    for i in range(40):  # stratum 12/sparse_f2: 40 worlds x 1 df, d spread 20
        w = f"o{i}"
        others += _world(w, {"bar_d": [[100], [100]], "reach_d": [[100], [120 if i % 2 else 80]]},
                         ordered=1e4)
        infos[w] = _info(w)
    rows = lucky + others
    ms = [measures(w, eps, ARMS) for w, eps in episode_values(rows).items()]
    spreads = pooled_spreads(ms, infos)
    own = spreads[(12, "sparse_f4")]
    assert own.df_ep == 41 and own.merged_ep  # 1 df < 30: merged within size
    assert spreads[(12, "sparse_f2")].df_ep == 40 and not spreads[(12, "sparse_f2")].merged_ep
    m = next(x for x in ms if x.world == "lucky")
    se, floor = noise_floor(m, infos["lucky"], own, 3.0)
    assert se > 1.0  # not the world's own ~0 standard error
    assert classify(m, floor) != HARD
    unmerged = pooled_spreads(ms, infos, merge_below=1)[(12, "sparse_f4")]
    assert unmerged.df_ep == 1 and not unmerged.merged_ep  # the screen's rule: df > 0 kept
    assert unmerged.merged_rep and unmerged.df_rep == 0  # no replicate df anywhere


# --- bound and ceiling ---


def test_bound_below_an_arm_names_world_episode_and_arm() -> None:
    rows = [_row("w1", 7, "bar_d", 50, b2=60.0), _row("w1", 7, "reach_d", 60.005, b2=60.0),
            _row("w2", 3, "reach_r", 61, 1, b2=60.0)]
    bad = bound_violations(rows)
    assert len(bad) == 1  # 60.005 is within the 0.01 tolerance
    f = bad[0]
    assert (f.world, f.episode, f.subject, f.replicate) == ("w2", 3, "reach_r", 1)
    assert "w2" in f.message and "episode 3" in f.message and "reach_r" in f.message


def test_near_full_world_is_saturated_and_not_selected() -> None:
    rows = _world("w", {"bar_d": [[98]], "reach_d": [[200]]}, ordered=100.0, b2=1e4, base=1e4)
    for r in rows:
        r["fulfilled_units"] = repr(98.0)
    m = _measure(rows)
    assert m.bar_fill == pytest.approx(0.98)
    assert _class(rows, SPREAD) == atlas.SATURATED
    rec = _record("w")
    chosen = select([Candidate("w", atlas.SATURATED, m.gain, m.bar, m.room, rec)], 4,
                    coverage_bins(12, [rec]))
    assert chosen == []


def test_demand_beyond_base_capacity_is_saturated_not_open() -> None:
    rows = _world("w", {"bar_d": [[98]], "reach_d": [[90]]}, b2=1000.0, base=100.0,
                  ordered=1000.0)
    m = _measure(rows)
    assert m.ceiling == pytest.approx(100.0) and m.room == pytest.approx(0.02)
    assert _class(rows, SPREAD) == atlas.SATURATED
    # B2 alone (the demand cap) would have read room 0.9
    assert (1000 - 98) / 1000 > atlas.OPEN_ROOM


def test_arm_above_the_base_capacity_ceiling() -> None:
    rows = _world("w", {"bar_d": [[80]], "reach_d": [[120]], "portable": [[130]]},
                  b2=1000.0, base=100.0, ordered=1000.0)
    m = _measure(rows)
    assert m.ceiling == pytest.approx(130.0)  # never below a played arm (portable too)
    assert m.above_base == {"reach_d": 1, "portable": 1}


def test_collapse_open_and_flat_classes() -> None:
    dead = _world("w", {"bar_d": [[5]], "reach_d": [[6]]}, dead=True, cycles=30)
    assert _class(dead, SPREAD) == atlas.COLLAPSE
    late = _world("w", {"bar_d": [[5]], "reach_d": [[6]]}, dead=True, cycles=60,
                  ordered=1000.0)
    assert _class(late, SPREAD) != atlas.COLLAPSE
    roomy = _world("w", {"bar_d": [[100]], "reach_d": [[90]]}, b2=500.0, base=500.0,
                   ordered=1000.0)
    assert _class(roomy, SPREAD) == OPEN
    tight = _world("w", {"bar_d": [[100]], "reach_d": [[90]]}, b2=105.0, base=1000.0,
                   ordered=1000.0)
    assert _class(tight, SPREAD) == atlas.FLAT


def test_fixed_argmax_arms_in_confirm() -> None:
    rows = _world("w", {"bar_d": [[100], [100]], "bar_r": [[150, 150], [150, 150]],
                        "reach_d": [[120], [120]], "reach_r": [[10, 10], [10, 10]]})
    m = _measure(rows, fixed=("bar_d", "reach_r"))
    assert (m.bar_arm, m.reach_arm, m.gain) == ("bar_d", "reach_r", -90.0)


# --- selection, coverage, split ---


def _record(world: str, *, choice: str = "sparse_f2", pattern: str = "iid",
            shocks: bool = False, rho: float = 0.5) -> dict:
    coords = {c: 0.0 for c in atlas.COORDINATES} | {"rho": rho}
    return {"world_id": world, "size": 12, "coordinates": coords,
            "levers": {"choice": choice, "pattern": pattern, "shocks": shocks}}


def test_uncovered_bin_gets_its_best_candidate_first() -> None:
    cands = [Candidate(f"h{i}", HARD, 50.0 - i, 100.0, 0.5, _record(f"h{i}"))
             for i in range(6)]
    cands += [Candidate("sw_low", HARD, 1.0, 100.0, 0.5, _record("sw_low", pattern="switch")),
              Candidate("sw_best", HARD, 2.0, 100.0, 0.5, _record("sw_best",
                                                                   pattern="switch"))]
    chosen = select(cands, 3, [Bin(12, "pattern", "switch")])
    ids = [c.world for c, _ in chosen]
    assert ids[0] == "sw_best" and chosen[0][1] == "coverage:pattern=switch"
    assert ids[1:] == ["h0", "h1"]  # then by score


def test_selection_is_deterministic_and_keeps_a_quarter_open() -> None:
    cands = [Candidate(f"h{i:02d}", HARD, 10.0 + i, 100.0, 0.5, _record(f"h{i:02d}"))
             for i in range(20)]
    cands += [Candidate(f"o{i}", OPEN, None, 100.0, 0.2 + 0.01 * i, _record(f"o{i}"))
              for i in range(5)]
    records = [c.record for c in cands]
    bins = coverage_bins(12, records)
    a = select(cands, 8, bins)
    b = select(list(reversed(cands)), 8, coverage_bins(12, list(reversed(records))))
    assert [(c.world, p) for c, p in a] == [(c.world, p) for c, p in b]
    assert len(a) == 8
    assert sum(c.cls == OPEN for c, _ in a) >= 2  # K/4
    hard = [c.world for c, _ in a if c.cls == HARD]
    assert "h19" in hard  # best gain/bar first


def test_split_region_then_hash_strata() -> None:
    region = [WorldInfo(f"r{i}", 42, "sparse_f2", "switch", False, True) for i in range(3)]
    stratum = [WorldInfo(f"s{i}", 12, "sparse_f2", "burst", False, True) for i in range(10)]
    split = split_worlds(region + stratum)
    assert all(split[w.id] == HELDOUT_REGION for w in region)
    by_hash = sorted((w.id for w in stratum),
                     key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    assert [split[w] for w in by_hash] == [HELDOUT] * 2 + [VALIDATION] * 2 + [TRAIN] * 6
    # a 12-producer switch world is not out of region
    assert split_worlds([WorldInfo("x", 12, "dense", "switch", False, True)])["x"] == TRAIN


def test_held_out_sealed() -> None:
    doc = {"worlds": [{"split": s, "record": {"world_id": s}}
                      for s in (TRAIN, HELDOUT, VALIDATION, HELDOUT_REGION, TRAIN)]}
    assert [w["split"] for w in train_worlds(doc)] == [TRAIN, TRAIN]


def test_confirm_on_fresh_seeds() -> None:
    wid = "ws-atlas-v1-12-0003"
    screen, confirm = set(episode_seeds(wid, 1)), set(episode_seeds(wid, 2, "confirm"))
    assert len(confirm) == 2 and not screen & confirm
    reps_s = {replicate_seed(wid, e, r) for e in screen for r in range(2)}
    reps_c = {replicate_seed(wid, e, r) for e in confirm for r in range(2)}
    assert not reps_s & reps_c


# --- arms and jobs ---


def test_arm_replicates_per_stage() -> None:
    roster = load_roster(DEFAULT_ROSTER)
    screen, confirm = arm_table(roster, SCREEN), arm_table(roster, CONFIRM)
    r = {a["router"]: a.get("replicates") for a in screen if a["random"]}
    assert r == {"random": 1, "random_whole": 2, "makeable_weighted_whole": 1}
    assert {a["router"]: a.get("replicates") for a in confirm if a["random"]} == {
        "random": 2, "random_whole": 2, "makeable_weighted_whole": 2}
    assert "portable" not in {a["router"] for a in screen}
    assert [a["side"] for a in confirm if a["router"] == "portable"] == [atlas.REPORTED]
    assert {a["router"] for a in screen if a["side"] == REACH} == {
        "makeable_split", "makeable_weighted_whole", "trained_orderer"}
    world = anchor_worlds()[0]
    # 12 jobs = 13 router-equivalents (the trained orderer counts 2); confirm 15 jobs
    assert len(build_jobs([world], screen, 99, 12)) == 12
    assert len(build_jobs([world], confirm, 99, 12)) == 15


def test_anchors_rebuild_exp0050_configs() -> None:
    if not EXP0050_SPEC.exists():
        pytest.skip("EXP-0050 spec.json not present")
    pinned = {w["id"]: w for w in json.loads(EXP0050_SPEC.read_text())["pins"]["worlds"]}
    anchors = anchor_worlds()
    assert len(anchors) == 8
    for e in anchors:
        assert e.regime == "shocked" and len(e.episode_seeds) == 1
        p = pinned[e.id]
        assert e.episode_seeds[0] == p["episode_seeds"][0]
        assert fanout_record(e)["config_sha256"] == p["config_sha256"]


# --- dry runs ---

DRY = {"horizon": 12, "sizes": [12], "worlds_per_size": 1,
       "arms": ["round_robin", "makeable_split", "random"], "replicates": 1}


def _snapshot(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def dry_runs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    roster = load_roster(DEFAULT_ROSTER)
    before = _snapshot(REGISTRY_DIR)
    base = tmp_path_factory.mktemp("atlas")
    t0 = time.monotonic()
    code_s = run_atlas(mode=SCREEN, out_dir=base / "screen", roster=roster, dry_run=True,
                       shock_profile="shocks-i-smoke", log=lambda s: None, **DRY)
    t1 = time.monotonic()
    code_c = run_atlas(mode=CONFIRM, out_dir=base / "confirm", roster=roster, dry_run=True,
                       screen_dir=base / "screen", log=lambda s: None,
                       **(DRY | {"replicates": 2}))
    t2 = time.monotonic()
    return {"base": base, "codes": (code_s, code_c), "times": (t1 - t0, t2 - t1),
            "registry_before": before, "registry_after": _snapshot(REGISTRY_DIR)}


def test_dry_runs_write_the_layout(dry_runs: dict) -> None:
    base = dry_runs["base"]
    assert dry_runs["codes"] == (0, 0)
    assert all(t < 120 for t in dry_runs["times"])
    for name in ("spec.json", "episodes.csv", "worlds.csv", "selection.csv", "coverage.csv",
                 "anchors.csv", "contract_failures.csv", "RESULT.md"):
        assert (base / "screen" / name).exists(), name
    for name in ("spec.json", "episodes.csv", "worlds.csv", "deciding_check.csv",
                 "atlas_v1.json", "contract_failures.csv", "RESULT.md"):
        assert (base / "confirm" / name).exists(), name
    head = (base / "screen" / "episodes.csv").read_text().splitlines()[0].split(",")
    assert {"bound_b2", "bound_base", "trace_split_degree_consumer"} <= set(head)
    for stage in ("screen", "confirm"):
        spec = json.loads((base / stage / "spec.json").read_text())
        assert spec["kind"] == "dry-run" and spec["status"] == "complete"
        assert "DRY RUN" in (base / stage / "RESULT.md").read_text()
    with (base / "screen" / "worlds.csv").open(newline="") as fh:
        worlds = list(csv.DictReader(fh))
    assert worlds and all(w["class"] in atlas.CLASSES and json.loads(w["arm_units"])
                          and w["b2"] and w["noise_floor"] for w in worlds)  # "Screen record"
    doc = json.loads((base / "confirm" / "atlas_v1.json").read_text())
    assert doc["content_hash"] == atlas.content_hash(doc)


def test_dry_confirm_runs_fresh_seeds(dry_runs: dict) -> None:
    base = dry_runs["base"]

    def episodes(stage: str) -> set[tuple[str, int]]:
        rows = (base / stage / "episodes.csv").read_text().splitlines()[1:]
        return {(r.split(",")[0], int(r.split(",")[3])) for r in rows}

    s, c = episodes("screen"), episodes("confirm")
    worlds = {w for w, _ in c}
    assert worlds and not {(w, e) for w, e in s if w in worlds} & c


def test_hard_region_found_leaves_the_gate_untouched(dry_runs: dict) -> None:
    assert dry_runs["registry_before"] == dry_runs["registry_after"]
    assert not any("atlas" in name for name in dry_runs["registry_after"])


def test_real_runs_refused(dry_runs: dict, tmp_path: Path) -> None:
    roster = load_roster(DEFAULT_ROSTER)
    with pytest.raises(ValueError, match="not a valid atlas screen run"):
        run_atlas(mode=SCREEN, out_dir=tmp_path / "s", roster=roster, prod_ckpt="p",
                  orderer_ckpt="o", exp0050_episodes=str(tmp_path), log=lambda s: None,
                  **(DRY | {"arms": None, "replicates": None}))
    with pytest.raises(ValueError, match="a real confirm needs a complete real screen"):
        run_atlas(mode=CONFIRM, out_dir=tmp_path / "c", roster=roster,
                  screen_dir=dry_runs["base"] / "screen", source_commit="a" * 40,
                  prod_ckpt="p", orderer_ckpt="o", log=lambda s: None)
    with pytest.raises(ValueError, match="needs --exp0050-episodes"):
        run_atlas(mode=SCREEN, out_dir=tmp_path / "s2", roster=roster,
                  source_commit="a" * 40, prod_ckpt="p", orderer_ckpt="o",
                  log=lambda s: None)


def _inflated(job: dict) -> dict[str, str]:
    row = run_atlas_job(job)
    row["units"] = repr(float(row["bound_b2"]) + 5.0)
    return row


def test_bound_violation_makes_the_run_invalid(tmp_path: Path) -> None:
    code = run_atlas(mode=SCREEN, out_dir=tmp_path / "s", roster=load_roster(DEFAULT_ROSTER),
                     dry_run=True, shock_profile="shocks-i-smoke", run_job_fn=_inflated,
                     log=lambda s: None, **(DRY | {"arms": ["round_robin"]}))
    assert code == CONTRACT_INVALID
    spec = json.loads((tmp_path / "s" / "spec.json").read_text())
    assert spec["status"] == "invalid"
    msgs = spec["result"]["bound_violations"]
    assert msgs and all("round_robin" in m and "episode" in m for m in msgs)
    assert any("ws-atlas-v1-12-0000" in m for m in msgs)


def test_analyse_classes_every_world() -> None:
    rows = (_world("a", {"bar_d": [[100], [100]], "reach_d": [[150], [151]]}, ordered=1e4)
            + _world("b", {"bar_d": [[100], [100]], "reach_d": [[90], [91]]}, ordered=1e4))
    infos = {"a": _info("a"), "b": _info("b")}
    ms, spreads, wrows = analyse(rows, infos, ARMS, CONFIRM)
    assert set(ms) == {"a", "b"} and len(wrows) == 2
    assert {r["world"]: r["class"] for r in wrows}["b"] in (OPEN, atlas.FLAT)


EXP0050_EPISODES = EXP0050_SPEC.parent / "artifacts" / "episodes.csv"


def _exp0050_rows() -> list[dict[str, str]]:
    if not EXP0050_EPISODES.exists() or EXP0050_EPISODES.read_bytes()[:7] == b"version":
        pytest.skip("EXP-0050 episodes.csv not present (git-lfs)")
    from tokenomics.eval.routing_audit import read_episode_dicts

    return read_episode_dicts(EXP0050_EPISODES)


def test_provisional_spreads_from_exp0050() -> None:
    sp = atlas.provisional_spreads(_exp0050_rows(),
                                   arm_table(load_roster(DEFAULT_ROSTER), SCREEN))
    assert set(sp) == {(12, "sparse_f2"), (12, "sparse_f4"), (12, "dense"), (42, "sparse_f2"),
                       (42, "sparse_f4"), (42, "sparse_f7"), (42, "sparse_f14"), (42, "dense")}
    assert all(v.sigma_ep is not None and v.sigma_rep is not None for v in sp.values())
    assert all(not v.merged_ep for v in sp.values())  # 4 shocked worlds x 1 df each


def test_anchor_reproduces_exp0050_units() -> None:
    ref = _exp0050_rows()
    anchor = anchor_worlds()[:1]
    arms = [a for a in arm_table(load_roster(DEFAULT_ROSTER), SCREEN)
            if a["router"] == "round_robin"]
    rows = [run_atlas_job(j) for j in build_jobs(anchor, arms, 1, 100)]
    table = atlas.anchor_rows(rows, ref, anchor)
    assert len(table) == 1 and table[0]["match"], table
    assert float(rows[0]["bound_b2"]) >= float(rows[0]["units"])


# --- guards (review #301) ---


def test_real_run_problems_one_at_a_time() -> None:
    from dataclasses import replace

    roster = load_roster(DEFAULT_ROSTER)
    worlds = anchor_worlds()
    filters = {k: None for k in ("sizes", "worlds_per_size", "arms", "replicates",
                                 "shock_profile")}
    ok = {"roster": roster, "worlds": worlds, "horizon": 100, "source_commit": "a" * 40,
          "filters": filters, "shock_window": (21, 90)}
    assert atlas.real_run_problems(**ok) == []
    cases = [({"horizon": 50}, "horizon 50"), ({"source_commit": "a" * 39}, "40-hex"),
             ({"source_commit": ""}, "40-hex"), ({"source_commit": "g" * 40}, "40-hex"),
             ({"shock_window": (2, 8)}, "shock window"),
             ({"roster": replace(roster, frozen=False, screened_on=None)}, "not frozen"),
             ({"worlds": anchor_worlds("shocks-i-smoke")}, "test-only shock profile")]
    cases += [({"filters": filters | {k: 1}}, f"--{k.replace('_', '-')} is set")
              for k in filters]
    for change, needle in cases:
        problems = atlas.real_run_problems(**(ok | change))
        assert len(problems) == 1 and needle in problems[0], (change, problems)


def _reference_csv(path: Path, world: str, episode: int, units: float) -> Path:
    path.write_text("world,episode,subject,replicate,units,dead\n"
                    f"{world},{episode},entity_planner/round_robin,0,{units},false\n")
    return path


@pytest.mark.parametrize("matching", [True, False])
def test_anchor_mismatch_makes_the_run_invalid(tmp_path: Path, matching: bool) -> None:
    anchor = anchor_worlds()[0]  # fan-sparse_wide-12-f2-shocked-00: round_robin 474 units
    ref = _reference_csv(tmp_path / "ref.csv", anchor.id, anchor.episode_seeds[0],
                         474.0 if matching else 470.0)
    out = tmp_path / "s"
    code = run_atlas(mode=SCREEN, out_dir=out, roster=load_roster(DEFAULT_ROSTER),
                     dry_run=True, exp0050_episodes=str(ref), log=lambda s: None,
                     **(DRY | {"horizon": 100, "arms": ["round_robin"]}))
    rows = list(csv.DictReader((out / "anchors.csv").open()))
    assert len(rows) == 1 and rows[0]["match"] == ("true" if matching else "false")
    failures = list(csv.DictReader((out / "contract_failures.csv").open()))
    spec = json.loads((out / "spec.json").read_text())
    if matching:
        assert code == 0 and failures == [] and spec["status"] == "complete"
    else:
        assert code == CONTRACT_INVALID and spec["status"] == "invalid"
        assert [f["check"] for f in failures] == [atlas.ANCHOR_CHECK]


def test_bound_columns_are_the_world_episodes_ceiling(dry_runs: dict) -> None:
    from tokenomics.eval.ceiling_bound import B_BASE, episode_demand, world_ceiling
    from tokenomics.eval.world_space import config_from_record

    checked = set()
    for stage in ("screen", "confirm"):
        spec = json.loads((dry_runs["base"] / stage / "spec.json").read_text())
        recs = {r.get("world_id", r.get("id")): r for r in spec["pins"]["worlds"]}
        for row in csv.DictReader((dry_runs["base"] / stage / "episodes.csv").open()):
            rec = recs[row["world"]]
            if "world_id" not in rec:
                continue  # an anchor: covered by test_anchor_reproduces_exp0050_units
            cfg = config_from_record(rec)
            assert rec["levers"]["pattern"] not in ("cyclic", "iid")  # a patterned world
            b = world_ceiling(cfg, 12, episode_demand(cfg, 12, int(row["episode"])))
            assert float(row["bound_b2"]) == b["B2"].units
            assert float(row["bound_base"]) == b[B_BASE].units
            assert row["bound_b2_binding"] == b["B2"].binding_constraint
            checked.add((stage, row["world"], row["episode"]))
    assert {s for s, _, _ in checked} == {"screen", "confirm"}
