"""Task 5.1: training worlds and the bar precompute (add-m4-orderer-redesign design D6;
learned-orderer "Relative disqualification" / "No feasible start")."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tokenomics.eval.atlas import COLLAPSE, CONFIRM_EPISODES, CONFIRM_STAGE, TRAIN
from tokenomics.eval.gate import EpisodeRow, row_to_dict
from tokenomics.eval.contract import job_failure
from tokenomics.eval.roster import load_roster
from tokenomics.eval.world_space import (LEVERS, check_levers, episode_seeds, world_record)
from tokenomics.train import order_es_worlds as ow
from tokenomics.train.per_order_scorer import (N_PHI, N_PSI, PHI_FEATURES, PSI_FEATURES,
                                               FeatureNormaliser, RawRows)

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
PARAMS_V2 = REPO / "tokenomics/tokenomics/eval/registry/side_effect_params_v2.json"


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_atlas = pytest.mark.skipif(not _lfs_present(ATLAS_V2), reason="EXP-0052 atlas absent")
needs_ckpt = pytest.mark.skipif(not _lfs_present(ORDERER), reason="git-lfs checkpoint absent")


@pytest.fixture(scope="module")
def atlas() -> dict:
    from tokenomics.eval.teacher_audit import load_atlas

    return load_atlas(ATLAS_V2)


@pytest.fixture(scope="module")
def reserved() -> dict[str, set[int]]:
    return ow.reserved_seeds()


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory: pytest.TempPathFactory) -> str:
    pytest.importorskip("torch")
    from tokenomics.train.orderer_scorer import OrdererScoreModel, save_scorer

    path = str(tmp_path_factory.mktemp("ckpt") / "orderer_scorer.pt")
    save_scorer(OrdererScoreModel(hidden=16), path)
    return path


def test_registered_constants() -> None:
    params = json.loads(PARAMS_V2.read_text())
    assert (ow.X_RUNDOWN, ow.Y_STARVATION) == (params["x"], params["y"])
    assert (round(ow.X_RUNDOWN, 4), round(ow.Y_STARVATION, 4)) == (0.7516, 0.7302)  # D6
    assert ow.TARGET_WORLDS == ow.ATLAS_WORLDS + 41 == 64
    assert ow.MIN_FEASIBLE == 50 and int(ow.TARGET_WORLDS * ow.MAX_REPLACED_SHARE) == 16
    assert ow.HORIZON == 100 and ow.SIZE == 12 and ow.PRODUCER == "entity_planner"


def test_bar_arms_and_decodes() -> None:
    """RR, MWW R = 2, TR, the five deterministic roster v2 members (none trained), A0; only A0
    is choice-v1 and keyed."""
    table = ow.bar_arms(load_roster())
    assert [a["arm"] for a in table] == ["RR", "MWW", "TR", "blend_0.5", "even_split",
                                         "obs_price", "proportional", "shock_reactive", "A0"]
    assert ow.arm_replicates(table) == {a["arm"]: 2 if a["arm"] == "MWW" else 1 for a in table}
    assert {a["arm"]: a["decode"] for a in table} == {
        a["arm"]: "choice-v1" if a["arm"] == "A0" else "logits-v1" for a in table}
    roster = [a for a in table if a["roster"]]
    assert not any(a["trained"] for a in roster) and not any(a["random"] for a in roster)
    assert {a["arm"]: a["trained"] for a in table if a["trained"]} == {"TR": True, "A0": True}
    assert ow.bar_labels(table) == [a["arm"] for a in table][:-1]


# --- worlds ---


@needs_atlas
def test_atlas_candidates(atlas: dict) -> None:
    cands = ow.atlas_candidates(atlas)
    assert len(cands) == ow.ATLAS_WORLDS
    worlds = [c.world for c in cands]
    assert worlds == sorted(worlds)
    by_id = {e["record"]["world_id"]: e for e in atlas["worlds"]}
    for c in cands:
        e = by_id[c.world]
        assert e["split"] == TRAIN and e["record"]["size"] == 12
        assert e["record"]["coordinates"]["rho"] > 0.28 and e["confirm_class"] != COLLAPSE
        assert c.episodes == episode_seeds(c.world, 2, ow.TRAIN_STAGE)


def test_fresh_draws_deterministic_in_range() -> None:
    a = ow.fresh_candidates(23)
    b = ow.fresh_candidates(23)
    assert ow._order_digest(a) == ow._order_digest(b) and len(a) == ow.FRESH_POOL
    assert [c.order for c in a] == list(range(23, 23 + ow.FRESH_POOL))
    playable = [c for c in a if c.screened is None]
    assert len(playable) >= 3 * 41  # room for drops and replacements
    assert all(c.rho > 0.28 for c in playable)
    assert all(c.screened == ow.DROPPED_RHO for c in a if c.screened is not None)
    for c in a:  # the atlas lever distribution (world_space ranges, shocks + patterns)
        check_levers(12, c.point.levers)
    assert {c.point.levers.shocks for c in a} == {False, True}
    assert len({c.point.levers.pattern for c in a}) == 7
    for name, rng in LEVERS.items():
        vals = [getattr(c.point.levers, name) for c in a]
        assert rng.lo <= min(vals) and max(vals) <= rng.hi
    for c in playable[:3]:  # rho as the world record measures it
        rec = world_record(c.point)
        assert rec["coordinates"]["rho"] == pytest.approx(c.rho, abs=1e-12)
        assert rec["levers"] == c.point.levers.to_dict()


@needs_atlas
def test_disjoint_seed_stream(atlas: dict, reserved: dict[str, set[int]]) -> None:
    union = set().union(*reserved.values())
    fresh = [c for c in ow.fresh_candidates(23, reserved=union)]
    assert not any(c.screened == ow.SKIPPED_SEED for c in fresh)  # none collides by chance
    for c in fresh:
        assert c.point.seed not in union and not union & set(c.episodes)
        assert c.world.startswith(f"ws-{ow.FRESH_ATLAS}-12-")
    for c in ow.atlas_candidates(atlas):
        assert not union & set(c.episodes)
    # the reserved set covers the atlas v1 / v2 worlds of every split and the confirm's seeds
    for e in atlas["worlds"]:
        rec = e["record"]
        assert rec["seed"] in reserved["atlas_world"]
        assert set(episode_seeds(rec["world_id"], CONFIRM_EPISODES, CONFIRM_STAGE)) <= \
            reserved["atlas_confirm_episode"]
        assert set(episode_seeds(rec["world_id"], 1)) <= reserved["atlas_screen_episode"]
    assert {e["split"] for e in atlas["worlds"]} >= {"validation", "held-out"}
    assert len(reserved["atlas_world"]) == 512 + 128
    assert reserved["exp0050"] and reserved["registry"]
    # every training seed differs from every other one and from the schedule's stream seeds
    seeds = [s for c in fresh for s in (c.point.seed, *c.episodes)]
    assert len(seeds) == len(set(seeds))
    for s in (ow.stream_seed("monitor"), ow.stream_seed("minibatch", 12, 0, 1),
              ow.stream_seed("probe", 6)):
        assert s not in union
    # a reserved seed is skipped in the fixed order, never perturbed
    first = fresh[0]
    again = ow.fresh_candidates(23, pool=4, reserved={first.point.seed})
    assert again[0].screened == ow.SKIPPED_SEED and again[1].world == fresh[1].world
    again = ow.fresh_candidates(23, pool=4, reserved={first.episodes[1]})
    assert again[0].screened == ow.SKIPPED_SEED


def test_reserved_sources_complete(reserved: dict[str, set[int]]) -> None:
    """Every source's world and episode seeds are in the reserved set (review round 1)."""
    from tokenomics.eval.registry import REGISTRY_V1, REGISTRY_V2, load_registry
    from tokenomics.eval.routing_audit import fanout_worlds

    union = ow.reserved_union()
    for path in (REGISTRY_V1, REGISTRY_V2):
        for e in load_registry(path).entries:
            seeds = {e.world_seed, *e.episode_seeds}
            assert seeds <= reserved["registry"] and seeds <= union, (path.name, e.id)
    grid = fanout_worlds()
    assert grid
    for e in grid:
        seeds = {e.world_seed, *e.episode_seeds}
        assert seeds <= reserved["exp0050"] and seeds <= union, e.id


def test_real_run_requires_roster_v2() -> None:
    """B_we is over the roster v2 deterministic members (D6): a real run refuses another frozen
    roster, e.g. v1 (which lacks shock_reactive)."""
    from tokenomics.eval.roster import ROSTER_V1, ROSTER_V2
    from tokenomics.eval.teacher_audit import REGISTERED_ATLAS_HASH, ROLE_CHECKPOINTS

    ok = dict(atlas={"content_hash": REGISTERED_ATLAS_HASH}, horizon=100,
              source_commit="a" * 40, orderer_sha=ROLE_CHECKPOINTS["orderer"],
              filters={"target": None})
    assert ow.real_run_problems(roster=load_roster(ROSTER_V2), **ok) == []
    v1 = load_roster(ROSTER_V1)
    assert v1.frozen and "shock_reactive" not in [a["arm"] for a in ow.bar_arms(v1)]
    problems = ow.real_run_problems(roster=v1, **ok)
    assert len(problems) == 1 and "roster v2" in problems[0]


# --- selection ---


def _cand(i: int, source: str = ow.FRESH_SOURCE, screened: str | None = None) -> ow.Candidate:
    from tokenomics.eval.world_space import WorldPoint, Levers

    name = "atlas-v1" if source == ow.ATLAS_SOURCE else ow.FRESH_ATLAS
    pt = WorldPoint(name, 12, i, 1000 + i, Levers())
    return ow.Candidate(i, source, pt, (10 * i, 10 * i + 1), 0.5, None, screened)


def _out(world: str, *, a0_dead: bool = False, bar_dead: tuple[str, ...] = (),
         fill: float = 0.5) -> ow.WorldOutcome:
    return ow.WorldOutcome(world, True, bar_dead, a0_dead, "RR", fill)


@needs_atlas
def test_select_64_worlds_in_fixed_order(atlas: dict) -> None:
    order = ow.atlas_candidates(atlas) + ow.fresh_candidates(23)
    playable = [c for c in order if c.screened is None]
    sel = ow.select(order, {})
    assert sel.status is None and len(sel.wave) == 64
    assert sel.wave == playable[:64]  # wave 1: the 23 atlas worlds and the first 41 draws
    done = {c.world: _out(c.world) for c in playable}  # every world feasible
    sel = ow.select(order, done)
    assert sel.status == ow.COMPLETE and len(sel.training) == 64
    assert [c.source for c in sel.training].count(ow.ATLAS_SOURCE) == 23
    assert sel.training == playable[:64]


def test_select_waves_match_playing_everything() -> None:
    """Waves of the walk give the same training set as playing the whole order and taking the
    first feasible worlds."""
    order = [_cand(i, ow.ATLAS_SOURCE) for i in range(3)]
    order += [_cand(i, screened=ow.DROPPED_RHO if i % 5 == 0 else None) for i in range(3, 40)]
    rng = np.random.default_rng(7)
    truth = {}
    for c in order:
        r = rng.random()
        truth[c.world] = _out(c.world, a0_dead=r < 0.08, bar_dead=("RR",) if r > 0.85 else (),
                              fill=0.97 if 0.75 < r <= 0.85 else 0.5)
    known: dict = {}
    for _ in range(50):
        sel = ow.select(order, known, target=10, min_feasible=8, max_replaced=5)
        if sel.status is not None:
            break
        known.update({c.world: truth[c.world] for c in sel.wave})
    expected = [c for c in order if c.screened is None
                and ow.world_status(c, truth[c.world]) == ow.TRAINING][:10]
    assert sel.status == ow.COMPLETE and sel.training == expected
    assert sel.replaced >= 1  # the fixture exercises a replacement
    assert len(known) < len(order)  # the walk never played past its last needed draw


def test_filters_and_statuses() -> None:
    atlas_c, fresh_c = _cand(0, ow.ATLAS_SOURCE), _cand(1)
    assert ow.world_status(_cand(2, screened=ow.DROPPED_RHO), None) == ow.DROPPED_RHO
    assert ow.world_status(fresh_c, None) == ow.UNPLAYED
    assert ow.world_status(fresh_c, ow.WorldOutcome(fresh_c.world, False)) == ow.INCOMPLETE
    assert ow.world_status(fresh_c, _out("x", bar_dead=("TR",))) == ow.DROPPED_BAR_DEAD
    assert ow.world_status(fresh_c, _out("x", fill=0.95)) == ow.DROPPED_SATURATED
    assert ow.world_status(fresh_c, _out("x", fill=0.9499)) == ow.TRAINING
    assert ow.world_status(fresh_c, _out("x", a0_dead=True)) == ow.REPLACED_A0
    # bar filters are draw filters: an atlas world is only replaced when A0 dies
    assert ow.world_status(atlas_c, _out("x", bar_dead=("TR",), fill=0.99)) == ow.TRAINING
    assert ow.world_status(atlas_c, _out("x", a0_dead=True)) == ow.REPLACED_A0
    # a dropped draw on which A0 also dies is a drop, not a replacement
    assert ow.world_status(fresh_c, _out("x", a0_dead=True, fill=0.99)) == ow.DROPPED_SATURATED


def _row(world: str, ep: int, arm: str, rep: int, units: float, *, dead: bool = False,
         ordered: float = 100.0, fills: dict | None = None, r_end: float = 0.9) -> dict:
    return {"world": world, "episode": str(ep), "arm": arm, "replicate": str(rep),
            "units": repr(units), "dead": "true" if dead else "false",
            "ordered_units": repr(ordered), "fulfilled_units": repr(units),
            "consumer_fill_rate": json.dumps(fills or {"c0": 0.9}), "r_start": "1.0",
            "r_end": repr(r_end), "holding_units_per_cycle": "2.0", "episode_seconds": "1.5",
            "regime": "calm", "demand_digest": "d", "shock_digest": "s"}


def test_world_outcome_bar_arm_and_fill() -> None:
    reps = {"RR": 1, "MWW": 2, "TR": 1, "A0": 1}
    bar = ["RR", "MWW", "TR"]
    rows = []
    for ep in (1, 2):
        rows += [_row("w", ep, "RR", 0, 50.0), _row("w", ep, "TR", 0, 40.0),
                 _row("w", ep, "MWW", 0, 70.0), _row("w", ep, "MWW", 1, 40.0),
                 _row("w", ep, "A0", 0, 60.0, dead=ep == 2)]
    o = ow.world_outcome("w", (1, 2), rows, reps, bar)
    assert o.complete and o.bar_arm == "MWW" and o.a0_dead and not o.bar_dead_arms
    assert o.bar_fill == pytest.approx((0.7 + 0.4) / 2)  # MWW mean 55 > RR 50
    rows[0] = _row("w", 1, "RR", 0, 50.0, dead=True)
    assert ow.world_outcome("w", (1, 2), rows, reps, bar).bar_dead_arms == ("RR",)
    rows[1] = _row("w", 1, "TR", 0, 40.0, dead=True)  # any bar arm, not only the first
    rows[3] = _row("w", 1, "MWW", 1, 40.0, dead=True)  # either MWW replicate
    assert ow.world_outcome("w", (1, 2), rows, reps, bar).bar_dead_arms == ("RR", "MWW", "TR")
    assert not ow.world_outcome("w", (1, 2), rows[1:], reps, bar).complete  # a lost row


def test_bar_rows_guards() -> None:
    table = ow.bar_arms(load_roster())
    c = _cand(0, ow.ATLAS_SOURCE)
    rows = []
    for ep in c.episodes:
        rows += [_row(c.world, ep, a["arm"], r, 10.0 + i) for i, a in enumerate(table)
                 for r in range(ow.arm_replicates(table)[a["arm"]]) if a["arm"] != "A0"]
        rows.append(_row(c.world, ep, "A0", 0, 99.0, fills={"c0": 0.73, "c1": ow.Y_STARVATION,
                                                            "c2": 0.5}, r_end=0.75))
    out = ow.bar_rows([c], rows, table, 12, {(c.world, c.episodes[0])},
                      demand=lambda c, ep, h: 3.0)
    assert len(out) == 2 and set(out[0]) == set(ow.bar_columns(table))
    r = out[0]
    assert r["bar_arm"] == "shock_reactive" and r["bar"] == 17.0  # A0's 99 never enters B_we
    assert r["a0_units"] == 99.0 and r["a0_rundown"] is True  # 0.75 < 0.7516 * 1.0
    assert json.loads(r["a0_starved_consumers"]) == ["c0", "c2"]  # fill = Y is not starved
    assert r["monitor"] is True and out[1]["monitor"] is False
    assert r["demand_per_cycle"] == 3.0


def test_no_feasible_start_rules() -> None:
    atlas = [_cand(i, ow.ATLAS_SOURCE) for i in range(4)]
    fresh = [_cand(i) for i in range(4, 12)]
    order = atlas + fresh
    # 4 replaced of target 8 -> cap 2 exceeded
    known = {c.world: _out(c.world, a0_dead=c.order in (0, 1, 4)) for c in order}
    sel = ow.select(order, known, target=8, min_feasible=6)
    assert sel.status == ow.NO_FEASIBLE_START and sel.replaced == 3 and "1/4" in sel.reason
    known = {c.world: _out(c.world, a0_dead=c.order in (0, 4)) for c in order}
    sel = ow.select(order, known, target=8, min_feasible=6)
    assert sel.status == ow.COMPLETE and sel.replaced == 2 and len(sel.training) == 8
    # order exhausted: 7 >= min 6 -> complete with fewer; 7 < 8 -> no_feasible_start
    known = {c.world: _out(c.world, fill=0.99 if c.order in (5, 6, 7) else 0.5)
             for c in order}
    sel = ow.select(order, known, target=10, min_feasible=6)
    assert sel.status == ow.COMPLETE and len(sel.training) == 9 and "exhausted" in sel.reason
    sel = ow.select(order, known, target=10, min_feasible=9)  # exactly the minimum: complete
    assert sel.status == ow.COMPLETE and len(sel.training) == 9
    sel = ow.select(order, known, target=10, min_feasible=10)
    assert sel.status == ow.NO_FEASIBLE_START and "can be drawn" in sel.reason
    known[fresh[0].world] = ow.WorldOutcome(fresh[0].world, False)
    assert ow.select(order, known, target=10, min_feasible=6).status == "invalid"


# --- schedule ---


def test_schedule_stream() -> None:
    we = [(f"w{i:02d}", 100 * i + e) for i in range(64) for e in (0, 1)]
    mon = ow.monitor_set(we)
    assert len(mon) == 16 and len({w for w, _ in mon}) == 16 and set(mon) <= set(we)
    assert mon == ow.monitor_set(list(we))
    mb = ow.minibatch(we, 0, 1)
    assert len(mb) == 12 and len({w for w, _ in mb}) == 12 and set(mb) <= set(we)
    assert mb == ow.minibatch(we, 0, 1) != ow.minibatch(we, 0, 2)
    assert mb != ow.minibatch(we, 1, 1)
    assert len(ow.minibatch(we, 0, 3, size=24)) == 24
    a, b = ow.probe_minibatches(we)
    assert len(a) == len(b) == 6 and not {w for w, _ in a} & {w for w, _ in b}
    doc = ow.schedule_doc(we)
    assert doc["content_hash"] == ow.content_hash(doc) == ow.schedule_doc(we)["content_hash"]
    assert len(doc["minibatches"]["0"]) == 40 and set(doc["minibatches"]) == {"0", "1"}
    assert doc["minibatches"]["1"][4] == [{"world": w, "episode": e}
                                          for w, e in ow.minibatch(we, 1, 5)]
    assert doc["monitor"]["replicates"] == 2 and len(doc["monitor"]["world_episodes"]) == 16
    with pytest.raises(ValueError):
        ow.minibatch(we, 0, 0)


# --- normaliser ---


def _fixture_rows() -> RawRows:
    rng = np.random.default_rng(0)
    n = 400
    phi = rng.normal(size=(n, N_PHI))
    phi[:, 1] = 7.0  # constant -> unit rule
    phi[:, 2] = np.where(np.arange(n) < 360, 0.0, 1.0)  # IQR 0, std > 0 -> std rule
    phi[:, 3] = 0.5 + rng.normal(0, 0.01, n)  # 30 % far out on both sides: heavy clip
    phi[:60, 3], phi[60:120, 3] = 50.0, -50.0
    psi = rng.normal(size=(n // 4, N_PSI))
    psi[:, PSI_FEATURES.index("eligible_producer_count")] = np.log1p(2.0)  # transformed rows
    rows = RawRows()
    for k in range(4):
        rows.phi.append(phi[k * 100:(k + 1) * 100])
        rows.psi.append(psi[k * 25:(k + 1) * 25])
    return rows


def test_normaliser_fit_report_and_flags(tmp_path: Path) -> None:
    rows = _fixture_rows()
    nm, report = ow.fit_normaliser(rows)
    by = {(r["block"], r["index"]): r for r in report}
    assert len(report) == N_PHI + N_PSI
    assert by[("phi", 0)]["rule"] == "iqr" and by[("phi", 1)]["rule"] == "unit"
    assert by[("phi", 2)]["rule"] == "std"
    assert by[("phi", 2)]["scale"] == pytest.approx(float(np.std([0.0] * 360 + [1.0] * 40)))
    assert by[("phi", 1)]["scale"] == 1.0 and by[("phi", 1)]["flags"] == "constant_raw_units"
    assert by[("phi", 3)]["clip_high"] == by[("phi", 3)]["clip_low"] == pytest.approx(0.15)
    assert by[("phi", 3)]["clip_fraction"] == pytest.approx(0.3) and \
        "heavy_clip" in by[("phi", 3)]["flags"]
    assert by[("phi", 0)]["clip_fraction"] == 0.0
    j = PSI_FEATURES.index("eligible_producer_count")
    assert by[("psi", j)]["rule"] == "unit"
    # the 42-producer check (task 5.2: the raw config counts go through the policy's log1p on E
    # first): E = 14 against a median log1p(2) / scale 1 stays inside the clip
    from tokenomics.eval.world_space import Levers, build_world

    cfg42 = build_world(5, producers=42, levers=Levers(choice="sparse_f14"))
    cfg12 = build_world(5, producers=12, levers=Levers(choice="sparse_f2"))
    static = {(s["world"], s["feature"]): s
              for s in ow.static_psi_check(nm, {"w42": cfg42, "w12": cfg12})}
    e42 = static[("w42", "eligible_producer_count")]
    assert e42["raw_max"] == 14.0 and e42["transform"] == "log1p"
    assert e42["z_max"] == pytest.approx(np.log1p(14.0) - np.log1p(2.0)) and e42["at_clip"] == 0
    assert static[("w12", "eligible_producer_count")]["z_max"] == 0.0
    assert static[("w42", "product_depth")]["transform"] == ""
    # a scale narrow enough still pins it at the clip, and the flag names it
    narrow = FeatureNormaliser(nm.phi_median, nm.phi_scale, nm.psi_median,
                               tuple(0.1 if k == j else v for k, v in enumerate(nm.psi_scale)))
    pinned = ow.static_psi_check(narrow, {"w42": cfg42})
    assert next(s for s in pinned if s["feature"] == "eligible_producer_count")["at_clip"] > 0
    assert any("eligible_producer_count on w42" in f for f in ow.normaliser_doc(
        narrow, report, pinned, [("w", 1)], dry_run=False)["flags"]["static_psi_clip_42"])
    doc = ow.normaliser_doc(nm, report, list(static.values()), [("w", 1)], dry_run=False)
    assert doc["sha256"] == nm.sha256 and doc["prereg"]["clip"] == 5.0
    assert doc["prereg"]["zero_spread_rule"] == ("IQR = 0 -> the column's population std; "
                                                 "std = 0 -> 1.0 (all-equal values have std 0 "
                                                 "exactly)")
    assert doc["flags"]["static_psi_clip_42"] == []
    assert doc["prereg"]["transforms"] == {"psi.eligible_producer_count": "log1p"}
    assert doc["flags"]["history_completion_rate_clip"] == by[
        ("phi", PHI_FEATURES.index("completion_rate", 31))]["clip_fraction"]
    assert doc["rules"]["phi[2] " + PHI_FEATURES[2]] == "std"
    # the fit is the 4.3 fit, deterministic, and its hash moves with the rows
    nm2, _ = ow.fit_normaliser(_fixture_rows())
    assert nm2.sha256 == nm.sha256
    rows.phi[0][0, 0] += 100.0
    assert ow.fit_normaliser(rows)[0].sha256 != nm.sha256


def _independent_fit_rows_sha(we: list[tuple[str, int]], digests: dict) -> str:
    """The documented definition, computed here without the module: SHA-256 of the canonical
    JSON list [[world, episode, per-file row digest], ...] in fit order."""
    body = [[w, int(ep), digests[(w, ep)]] for w, ep in we]
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def test_fit_rows_sha_binds_each_row_digest() -> None:
    """Review round 2: `fit_rows_sha256` covers the per-file row digests, not just the list of
    world-episodes: changing any one digest changes it."""
    we = [("w0", 1), ("w0", 2), ("w1", 5)]
    digests = {k: hashlib.sha256(repr(k).encode()).hexdigest() for k in we}
    got = ow.fit_rows_sha256(we, digests)
    assert got == _independent_fit_rows_sha(we, digests)
    for k in we:
        edited = {**digests, k: "0" * 64}
        assert ow.fit_rows_sha256(we, edited) != got, k


def test_fit_rows_round_trip(tmp_path: Path) -> None:
    rows = _fixture_rows()
    digests = {}
    for k, (phi, psi) in enumerate(zip(rows.phi, rows.psi)):
        p = ow.fit_rows_path(tmp_path, "w", k)
        np.savez_compressed(p, phi=phi, psi=psi)
        digests[("w", k)] = ow.rows_digest(phi, psi)
    got = ow.load_fit_rows(tmp_path, [("w", k) for k in range(4)], digests)
    assert ow.fit_normaliser(got)[0].sha256 == ow.fit_normaliser(rows)[0].sha256
    # only the listed world-episodes enter the fit
    part = ow.load_fit_rows(tmp_path, [("w", 0)], digests)
    assert part.arrays()[0].shape == (100, N_PHI)
    digests[("w", 2)] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        ow.load_fit_rows(tmp_path, [("w", k) for k in range(4)], digests)


# --- end to end ---

ARGS = ["--dry-run", "--atlas-worlds", "1", "--target", "2", "--min-feasible", "1",
        "--horizon", "12", "--shock-profile", "shocks-i-smoke", "--workers", "1"]


def _fake_job_fn(a0_dead: set[str], fail: frozenset[str] | set[str] = frozenset()
                 ) -> Callable[[dict], dict[str, str]]:
    """Synthetic rows (no episode) with A0 dead on `a0_dead` worlds, a job failure on `fail`
    worlds' A0 jobs; A0 writes random fit rows."""
    def run(job: dict) -> dict[str, str]:
        if job["arm"] == "A0" and job["world"] in fail:
            return job_failure(job, RuntimeError("forced")).to_dict()
        dead = job["arm"] == "A0" and job["world"] in a0_dead
        row = row_to_dict(EpisodeRow(
            world=job["world"], size=12, regime=job["regime"], episode=job["episode"],
            subject=job["subject"], replicate=job["replicate"], units=10.0, dead=dead,
            demand_digest="d" * 8, r_start=1.0, r_end=0.9, consumer_fill_rate={"c0": 0.5},
            holding_units_per_cycle=1.0, ordered_units=100.0, fulfilled_units=10.0))
        row = {c: row.get(c, "") for c in ow.COLUMNS}
        fit = {c: "" for c in ow.FIT_COLUMNS}
        if job["arm"] == "A0":
            rng = np.random.default_rng(job["episode"] % 1000)
            phi, psi = rng.normal(size=(20, N_PHI)), rng.normal(size=(5, N_PSI))
            p = ow.fit_rows_path(job["fit_dir"], job["world"], job["episode"])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, phi=phi, psi=psi)
            fit = {"fit_phi_rows": "20", "fit_psi_rows": "5",
                   "fit_rows_sha256": ow.rows_digest(phi, psi)}
        row.update({"arm": job["arm"], "ordering_decode": job["decode"],
                    "episode_seconds": "0.1", "cpu_seconds": "0.1", "job_seconds": "0.1", **fit})
        return row
    return run


@needs_atlas
def test_atlas_training_seed_clash_refused(atlas: dict, tiny_ckpt: str, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    clash = ow.atlas_candidates(atlas, 1)[0].episodes[1]
    real = ow.reserved_union
    monkeypatch.setattr(ow, "reserved_union", lambda: real() | {clash})
    with pytest.raises(ValueError, match="reserved seeds"):
        ow.run_bar(atlas_path=ATLAS_V2, out_dir=tmp_path / "x", orderer_ckpt=tiny_ckpt,
                   roster=load_roster(), horizon=12, atlas_worlds=1, target=2, min_feasible=1,
                   pool=8, shock_profile="shocks-i-smoke", dry_run=True, log=lambda m: None)


@needs_atlas
def test_forced_death_no_feasible_start(atlas: dict, tiny_ckpt: str, tmp_path: Path) -> None:
    """"No feasible start" on a forced-death fixture: A0 dies on the atlas world and on the
    first fresh draw; with target 4 the cap is 1, so the second death stops the run (exit 3,
    to the user); with one death the next draw replaces the world."""
    # the draws are the pool's Latin hypercube, so they depend on the pool size (pinned)
    first_fresh = [c for c in ow.fresh_candidates(1, pool=24) if c.screened is None]
    atlas_w = ow.atlas_candidates(atlas, 1)[0].world
    common = dict(atlas_path=ATLAS_V2, orderer_ckpt=tiny_ckpt, roster=load_roster(),
                  horizon=12, atlas_worlds=1, target=4, min_feasible=3, pool=24,
                  shock_profile="shocks-i-smoke", dry_run=True, log=lambda m: None)
    out = tmp_path / "dies"
    code = ow.run_bar(out_dir=out, run_job_fn=_fake_job_fn({atlas_w, first_fresh[0].world}),
                      **common)
    assert code == ow.NO_FEASIBLE_EXIT
    spec = json.loads((out / "spec.json").read_text())
    assert spec["status"] == "no_feasible_start" and spec["result"]["replaced_a0_dead"] == 2
    assert not (out / "bar.csv").exists() and not (out / "normaliser.json").exists()
    cands = {r["world"]: r for r in csv.DictReader((out / "candidates.csv").open())}
    assert cands[atlas_w]["status"] == ow.REPLACED_A0
    assert json.loads((out / "bar_summary.json").read_text())["status"] == "no_feasible_start"

    out = tmp_path / "replaced"
    assert ow.run_bar(out_dir=out, run_job_fn=_fake_job_fn({atlas_w}), **common) == 0
    tw = json.loads((out / "training_worlds.json").read_text())
    assert [w["world"] for w in tw["worlds"]] == [c.world for c in first_fresh[:4]]
    cands = list(csv.DictReader((out / "candidates.csv").open()))
    assert [r["status"] for r in cands if r["world"] == atlas_w] == [ow.REPLACED_A0]
    # the fit uses the training world-episodes' A0 rows only (not the replaced world's)
    nd = json.loads((out / "normaliser.json").read_text())
    assert nd["fit_world_episodes"] == 8 and nd["phi_rows"] == 8 * 20

    out = tmp_path / "failed"
    assert ow.run_bar(out_dir=out, run_job_fn=_fake_job_fn(set(), {first_fresh[1].world}),
                      **common) == ow.CONTRACT_INVALID
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"


@needs_atlas
@needs_ckpt
def test_dry_run_bar_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2 worlds, horizon 12, shocks-i-smoke, --workers 1: every output, the bar table schema,
    decode per arm, resume, the offline refit, and the refusals."""
    out = tmp_path / "EXP-bar"
    base = ["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), *ARGS]
    started = time.perf_counter()
    assert ow.main([*base, "--out-dir", str(out)]) == 0
    assert time.perf_counter() - started < 120
    for f in ("spec.json", "episodes.csv", "contract_failures.csv", "candidates.csv", "bar.csv",
              "training_worlds.json", "schedule.json", "normaliser.json",
              "normaliser_report.csv", "normaliser_static_42.csv", "timings.csv",
              "world_records.json", "bar_summary.json", "RESULT.md"):
        assert (out / f).exists(), f
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    spec = json.loads((out / "spec.json").read_text())
    assert spec["status"] == "complete" and spec["kind"] == "dry-run"
    pins = spec["pins"]
    assert pins["ordering_decode"] == "choice-v1" and pins["producer"] == "entity_planner"
    assert pins["decode_by_arm"]["A0"] == "choice-v1"
    assert {v for k, v in pins["decode_by_arm"].items() if k != "A0"} == {"logits-v1"}
    assert pins["per_order"]["arm"] == "A0" and pins["normaliser"]["clip"] == 5.0
    table = ow.bar_arms(load_roster())
    episodes = list(csv.DictReader((out / "episodes.csv").open()))
    assert len(episodes) == 2 * 2 * 10  # worlds x episodes x (8 arms, MWW twice)
    for e in episodes:
        assert e["ordering_decode"] == ("choice-v1" if e["arm"] == "A0" else "logits-v1")
        assert float(e["episode_seconds"]) > 0 and float(e["job_seconds"]) >= \
            float(e["episode_seconds"])
        assert (e["fit_rows_sha256"] != "") == (e["arm"] == "A0")
    bar = list(csv.DictReader((out / "bar.csv").open()))
    assert list(bar[0]) == list(ow.bar_columns(table)) and len(bar) == 4
    by = {(e["world"], e["episode"], e["arm"], e["replicate"]): e for e in episodes}
    for r in bar:
        units = {a: float(r[f"units_{a}"]) for a in ow.bar_labels(table)}
        assert float(r["bar"]) == max(units.values()) == units[r["bar_arm"]]
        assert float(r["units_MWW"]) == (float(r["units_MWW_r0"]) + float(r["units_MWW_r1"])) / 2
        assert float(r["a0_units"]) == float(by[(r["world"], r["episode"], "A0", "0")]["units"])
        rr = by[(r["world"], r["episode"], "RR", "0")]
        if rr["dead"] == "false":  # D_we x horizon = the units consumers ordered
            assert float(r["demand_per_cycle"]) * 12 == pytest.approx(float(rr["ordered_units"]))
    assert sum(r["monitor"] == "true" for r in bar) == 2
    tw = json.loads((out / "training_worlds.json").read_text())
    assert tw["content_hash"] == ow.content_hash(tw) and len(tw["worlds"]) == 2
    assert [w["source"] for w in tw["worlds"]] == ["atlas", "fresh"]
    for w in tw["worlds"]:
        assert set(w["cost_s"]) == {a["arm"] for a in table} and w["record"]["config_sha256"]
    recs = json.loads((out / "world_records.json").read_text())
    assert set(recs) == {r["world"] for r in csv.DictReader((out / "candidates.csv").open())}
    sched = json.loads((out / "schedule.json").read_text())
    assert sched["content_hash"] == ow.content_hash(sched)
    nd = json.loads((out / "normaliser.json").read_text())
    assert nd["dry_run"] is True and nd["phi_rows"] == sum(
        int(e["fit_phi_rows"]) for e in episodes if e["arm"] == "A0")
    report = list(csv.DictReader((out / "normaliser_report.csv").open()))
    assert len(report) == N_PHI + N_PSI and {r["rule"] for r in report} <= {"iqr", "std", "unit"}

    logs: list[str] = []
    assert ow.run_bar(atlas_path=ATLAS_V2, out_dir=out, orderer_ckpt=str(ORDERER),
                      roster=load_roster(), horizon=12, atlas_worlds=1, target=2,
                      min_feasible=1, shock_profile="shocks-i-smoke", dry_run=True,
                      log=logs.append) == 0
    assert any("resume" in m and "40 rows" in m for m in logs)
    assert not any("wave" in m for m in logs)  # nothing left to play
    assert json.loads((out / "normaliser.json").read_text())["sha256"] == nd["sha256"]

    fit = tmp_path / "refit"
    assert ow.main(["--mode", "fit-normaliser", "--bar-dir", str(out), "--out-dir",
                    str(fit)]) == 0
    fj = json.loads((fit / "fit.json").read_text())
    assert fj["match"] is True and set(fj["compared"]) == {"sha256", "fit_rows_sha256",
                                                           "training_worlds_sha256"}
    # normaliser.json names its inputs: the ordered fit-row digests and the training worlds
    we = [(w["world"], int(ep)) for w in tw["worlds"] for ep in w["episodes"]]
    digests = {(e["world"], int(e["episode"])): e["fit_rows_sha256"] for e in episodes
               if e["arm"] == "A0"}
    assert nd["fit_rows_sha256"] == ow.fit_rows_sha256(we, digests)
    assert nd["training_worlds_sha256"] == tw["content_hash"]
    assert ow.fit_rows_sha256(we[::-1], digests) != nd["fit_rows_sha256"]
    assert nd["fit_rows_sha256"] == _independent_fit_rows_sha(we, digests)  # binds the rows
    import shutil

    for key in ("fit_rows_sha256", "training_worlds_sha256"):  # a recorded input differs
        bad = tmp_path / f"bar-{key}"
        shutil.copytree(out, bad)
        doc = json.loads((bad / "normaliser.json").read_text())
        doc[key] = "0" * 64
        (bad / "normaliser.json").write_text(json.dumps(doc))
        assert ow.main(["--mode", "fit-normaliser", "--bar-dir", str(bad), "--out-dir",
                        str(tmp_path / f"refit-{key}")]) == ow.CONTRACT_INVALID
        got = json.loads((tmp_path / f"refit-{key}" / "fit.json").read_text())
        assert got["compared"][key]["match"] is False and got["compared"]["sha256"]["match"]
    other = tmp_path / "other_atlas.json"  # the refit verifies the atlas against the bar's pin
    doc = json.loads(ATLAS_V2.read_text())
    doc["worlds"] = doc["worlds"][:-1]
    from tokenomics.eval.atlas import content_hash as atlas_hash

    doc["content_hash"] = atlas_hash(doc)
    other.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="not the bar run's"):
        ow.main(["--mode", "fit-normaliser", "--bar-dir", str(out), "--atlas", str(other),
                 "--out-dir", str(tmp_path / "refit-atlas")])
    with pytest.raises(ValueError, match="never written"):
        ow.main(["--mode", "fit-normaliser", "--bar-dir", str(out), "--out-dir", str(out)])

    # a resume under another decode is refused (the pins name each arm's decode)
    real = ow.bar_arms

    def other_decode(roster: object) -> list[dict]:
        return [{**a, "decode": "logits-v1"} for a in real(roster)]

    monkeypatch.setattr(ow, "bar_arms", other_decode)
    with pytest.raises(ValueError, match="different pins"):
        ow.main([*base, "--out-dir", str(out)])
    monkeypatch.setattr(ow, "bar_arms", real)

    # without --dry-run: refused (filters, horizon, commit)
    with pytest.raises(ValueError, match="not a valid bar-precompute run"):
        ow.main(["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--horizon", "12",
                 "--out-dir", str(tmp_path / "real")])


def test_worker_refuses_a_decode_other_than_pinned() -> None:
    """A job pinned to choice-v1 but played under logits-v1 is a job failure."""
    job = {"arm": "RR", "decode": "choice-v1", "world": "w", "episode": 1, "subject": "s",
           "replicate": 0, "size": 12, "regime": "calm", "contract": "joint_stack",
           "horizon": 2, "world_seed": 3, "generator": {"generator": "world_space", "params": {
               "producers": 12, "levers": {}, "shock_window": [10, 60]}},
           "producing": "entity_planner", "ordering": "round_robin", "action_seed": None,
           "checkpoints": None}
    d = ow.run_bar_job(job)
    assert d.get("scope") == "job" and "logits-v1" in d["message"]


def test_a0_job_keys_and_decode(tiny_ckpt: str, monkeypatch: pytest.MonkeyPatch,
                                tmp_path: Path) -> None:
    """A0 runs under choice-v1 keyed by DrawKey(world, episode seed, 0) (the members' CRN key);
    its fit rows are written and hashed."""
    from tokenomics.eval.provenance import sha256_file
    from tokenomics.eval.routing_audit import build_jobs

    c = next(x for x in ow.fresh_candidates(0, pool=8) if x.screened is None)
    table = [a for a in ow.bar_arms(load_roster()) if a["arm"] == "A0"]
    job = build_jobs([c.entry()], table, 1, 2, None,
                     {"orderer": {"path": tiny_ckpt, "sha256": sha256_file(tiny_ckpt)}})[0]
    job.update({"arm": "A0", "decode": "choice-v1", "fit_dir": str(tmp_path)})
    keys = []
    real = ow.set_draw_key
    monkeypatch.setattr(ow, "set_draw_key", lambda env, key: (keys.append(key),
                                                              real(env, key))[1])
    d = ow.run_bar_job(job)
    assert "scope" not in d, d.get("message")
    assert [(k.world, k.episode, k.replicate) for k in keys] == [(c.world, c.episodes[0], 0)]
    assert d["ordering_decode"] == "choice-v1" and int(d["fit_psi_rows"]) > 0
    with np.load(ow.fit_rows_path(tmp_path, c.world, c.episodes[0])) as z:
        assert ow.rows_digest(z["phi"], z["psi"]) == d["fit_rows_sha256"]
