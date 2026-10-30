"""Task 2.2 (add-m4-orderer-redesign design D3): the EXP-0057 decoder-check driver
`eval/decoder_check.py`: sample, arms and keys, the primary / fidelity / dispersion arithmetic
with D3's precedence, the per-comparison floors (n = 1 rule), reproduction, real-run refusal
and a horizon-12 dry run."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from tokenomics.eval import decoder_check as dc
from tokenomics.eval.atlas import content_hash
from tokenomics.eval.contract import DIGEST, job_failure
from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.roster import build_roster_v2, load_roster
from tokenomics.eval.routing_audit import build_jobs
from tokenomics.eval.teacher_audit import load_atlas

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
EXP0052 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/artifacts/episodes.csv"
EXP0056 = REPO / "experiments/EXP-0056-split-vs-rotation/artifacts/episodes.csv"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
PROBE = REPO / "openspec/changes/add-m4-orderer-redesign/probes/pre_pod_offline.py"
SHOCKED = "ws-atlas-v1-12-0060"
# D3 lists 12-0005, 0155, 0176, 0330 and 0464 as one-episode (cyclic, shock-free) worlds;
# 0330 and 0464 are validation worlds, so the train sample holds the other three
ONE_EPISODE = {"ws-atlas-v1-12-0005", "ws-atlas-v1-12-0155", "ws-atlas-v1-12-0176"}
ONE_EPISODE_VALIDATION = {"ws-atlas-v1-12-0330", "ws-atlas-v1-12-0464"}


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_inputs = pytest.mark.skipif(
    not all(_lfs_present(p) for p in (ATLAS_V2, EXP0052, EXP0056)),
    reason="EXP-0052 / EXP-0056 records absent")
needs_ckpt = pytest.mark.skipif(not _lfs_present(ORDERER), reason="git-lfs checkpoint absent")


@pytest.fixture(scope="module")
def atlas() -> dict:
    return load_atlas(ATLAS_V2)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory: pytest.TempPathFactory) -> str:
    pytest.importorskip("torch")
    from tokenomics.train.orderer_scorer import OrdererScoreModel, save_scorer

    path = str(tmp_path_factory.mktemp("ckpt") / "orderer_scorer.pt")
    save_scorer(OrdererScoreModel(hidden=16), path)
    return path


# --- hand-built rows ---


def _view(world: str, size: int = 12, floor: float = 10.0, rho: float = 0.3,
          choice: str = "sparse_f2") -> dict:
    return {"world": world, "size": size, "choice": choice, "rho": rho, "noise_floor": floor}


def _rows(world: str, arm: str, units: dict[int, dict[int, float]], size: int = 12,
          **cells: object) -> list[dict[str, str]]:
    return [{"world": world, "size": str(size), "episode": str(ep), "subject": dc.SUBJECT[arm],
             "replicate": str(rep), "units": repr(float(u)), "arm": arm,
             **{k: str(v) for k, v in cells.items()}}
            for ep, reps in units.items() for rep, u in reps.items()]


def _det_world(world: str, rr: float, tr: float, ts: float, size: int = 12,
               floor: float = 10.0, mwwk: float | None = None) -> tuple[dict, list[dict]]:
    """A one-episode world: RR / TR deterministic, TS (and MWWk) one value at both replicates."""
    rows = (_rows(world, dc.RR, {1: {0: rr}}, size) + _rows(world, dc.TR, {1: {0: tr}}, size)
            + _rows(world, dc.TS, {1: {0: ts, 1: ts}}, size)
            + _rows(world, dc.MWWK, {1: {0: ts if mwwk is None else mwwk,
                                         1: ts if mwwk is None else mwwk}}, size))
    return _view(world, size, floor), rows


def _primary(worlds: list[tuple[dict, list[dict]]], size: int = 12) -> dict:
    views = [v for v, _ in worlds]
    rows = [r for _, rs in worlds for r in rs]
    return dc.primary_reading(dc.world_table(dc.arm_units(rows), views), size)


def _gap_world(i: int, rec: float, g: float = 30.0, size: int = 12) -> tuple[dict, list[dict]]:
    return _det_world(f"w{size}-{i:02d}", 100.0 + g, 100.0, 100.0 + rec * g, size)


def _flat_world(i: int, size: int = 12) -> tuple[dict, list[dict]]:
    return _det_world(f"f{size}-{i:02d}", 100.0, 100.0, 100.0, size)


# --- sample ---


@needs_inputs
def test_sample_selection(atlas: dict) -> None:
    """23 + 14 train worlds with rho > 0.28, not collapse; EXP-0056's three present; no
    validation or held-out world; the confirm's seeds in EXP-0052's row order."""
    views = dc.sample_views(atlas)
    assert sum(v["size"] == 12 for v in views) == 23
    assert sum(v["size"] == 42 for v in views) == 14
    assert set(dc.EXP0056_WORLDS) <= {v["world"] for v in views}
    assert all(v["split"] == "train" and v["rho"] > 0.28 and v["class"] != "collapse"
               for v in views)
    splits = {e["split"] for e in atlas["worlds"]}
    assert {"validation", "held-out"} <= splits  # they exist and are excluded
    rr = {}
    for d in dc.read_reference(EXP0052):
        if d["subject"] == dc.SUBJECT[dc.RR]:
            rr.setdefault(d["world"], []).append(int(d["episode"]))
    for v in views:
        assert list(dc.confirm_seeds(v)) == rr[v["world"]], v["world"]
    one = {v["world"] for v in views if v["size"] == 12 and len(dc.confirm_seeds(v)) == 1}
    assert one == ONE_EPISODE
    split = {e["record"]["world_id"]: e["split"] for e in atlas["worlds"]}
    assert {split[w] for w in ONE_EPISODE_VALIDATION} == {"validation"}
    with pytest.raises(ValueError, match="validation split"):
        val = next(e["record"]["world_id"] for e in atlas["worlds"] if e["split"] == "validation")
        dc.select_views(atlas, [val])


def test_arms_and_replicates() -> None:
    roster = load_roster(dc.DEFAULT_ROSTER)
    table = {a: dc.decoder_arm(a, roster) for a in dc.ARMS}
    reps = {a: (t.get("replicates", 1) if t["random"] else 1) for a, t in table.items()}
    assert reps == {"TR": 1, "TS": 2, "MWWk": 2, "MWW": 2, "TW": 1, "RR": 1}
    assert all(t["producing"] == "entity_planner" and t["contract"] == "joint_stack"
               for t in table.values())
    assert table["TS"]["trained"] and table["TW"]["trained"] and table["TR"]["trained"]
    assert not table["MWWk"]["trained"]
    assert table["MWW"]["subject"] == "entity_planner/makeable_weighted_whole"


def test_longest_first() -> None:
    jobs = [{"size": 12, "checkpoints": None}, {"size": 42, "checkpoints": None},
            {"size": 12, "checkpoints": {"orderer": {}}}, {"size": 42, "checkpoints": {"o": {}}}]
    order = [(j["size"], bool(j["checkpoints"])) for j in dc.longest_first(jobs)]
    assert order == [(42, True), (42, False), (12, True), (12, False)]


# --- primary arithmetic ---


def test_primary_branches() -> None:
    flats = [_flat_world(i) for i in range(4)]
    assert _primary([_gap_world(i, 0.9) for i in range(5)] + flats)["reading"] == "decoder"
    # 3 of 5 strong (< 2/3), median 0.7 >= 1/3 -> decoder_partial
    recs = [0.9, 0.8, 0.7, 0.2, 0.1]
    p = _primary([_gap_world(i, r) for i, r in enumerate(recs)] + flats)
    assert (p["reading"], p["gap_worlds"], p["rec_strong_worlds"]) == ("decoder_partial", 5, 3)
    assert p["median_rec"] == pytest.approx(0.7)
    recs = [0.9, 0.8, 0.2, 0.1, 0.0]
    assert _primary([_gap_world(i, r) for i, r in enumerate(recs)])["reading"] == "not_decoder"
    assert _primary([_gap_world(i, 0.9) for i in range(4)] + flats)["reading"] == "no_gap"
    assert _primary([_gap_world(i, 0.9) for i in range(5)])["licence"] == "step 2"
    assert _primary([_gap_world(i, 0.0) for i in range(5)])["licence"] == "none: diagnose"


def test_primary_edges() -> None:
    # rec_w exactly 2/3 counts as strong; 4 of 6 = exactly 2/3 of gap worlds -> decoder
    recs = [2 / 3, 2 / 3, 1.0, 1.0, 0.0, 0.0]
    p = _primary([_gap_world(i, r) for i, r in enumerate(recs)])
    assert (p["reading"], p["rec_strong_worlds"]) == ("decoder", 4)
    # 2 of 5 strong (< 2/3): median exactly 1/3 -> decoder_partial; just below -> not_decoder
    for mid, reading in ((1 / 3, "decoder_partial"), (0.333, "not_decoder")):
        recs = [1.0, 1.0, mid, 0.0, 0.0]  # 2 of 5 strong; median = mid
        p = _primary([_gap_world(i, r) for i, r in enumerate(recs)])
        assert p["reading"] == reading, (mid, p)
    # a gap exactly at the floor is none: 5 worlds at G = floor -> no_gap
    worlds = [_det_world(f"e{i}", 110.0, 100.0, 110.0, floor=10.0) for i in range(5)]
    assert _primary(worlds)["gap_worlds"] == 0
    worlds[0] = _det_world("e0", 110.5, 100.0, 110.5, floor=10.0)
    assert _primary(worlds)["gap_worlds"] == 1


def test_no_gap_takes_precedence() -> None:
    """Spec "No gap takes precedence": < 5 gap worlds -> no_gap although every gap world
    recovers fully (decoder's rule holds on them)."""
    p = _primary([_gap_world(i, 1.0) for i in range(4)] + [_flat_world(i) for i in range(19)])
    assert p["reading"] == "no_gap" and p["gap_worlds"] == 4 and p["rec_strong_worlds"] == 4


def test_42_producer_rows_never_change_a_reading() -> None:
    base = [_gap_world(i, 0.9) for i in range(4)] + [_flat_world(i) for i in range(3)]
    big = [_gap_world(i, 0.9, g=500.0, size=42) for i in range(10)]
    assert _primary(base)["reading"] == _primary(base + big)["reading"] == "no_gap"
    assert _primary(base + big, size=42)["reading"] == "decoder"
    views = [v for v, _ in base + big]
    rows = [r for _, rs in base + big for r in rs]
    out = dc.readings(dc.world_table(dc.arm_units(rows), views), rows, views)
    assert out["primary"]["reading"] == "no_gap" and out["primary"]["worlds"] == 7
    assert out["reported_42"]["reported"] is True
    assert out["reported_42"]["primary"]["reading"] == "decoder"


# --- floors ---


def test_n1_rule_and_own_spread_floor() -> None:
    """A single-episode world with two deterministic arms uses the atlas floor (one paired
    observation); with two episodes the own 3 SE overrides a smaller atlas floor."""
    view, rows = _det_world("one", 130.0, 100.0, 120.0, floor=10.0)
    (r,) = dc.world_table(dc.arm_units(rows), [view])
    assert r.gap.pairs == 1 and r.gap.own3se is None and r.gap.floor == 10.0 and r.is_gap
    # TS - MWWk on the same world pairs by replicate: 2 observations -> own SE (here 0)
    assert r.fidelity.pairs == 2 and r.fidelity.own3se == 0.0 and r.fidelity.floor == 10.0
    rows = (_rows("two", dc.RR, {1: {0: 200.0}, 2: {0: 100.0}})
            + _rows("two", dc.TR, {1: {0: 100.0}, 2: {0: 100.0}})
            + _rows("two", dc.TS, {1: {0: 150.0, 1: 150.0}, 2: {0: 100.0, 1: 100.0}})
            + _rows("two", dc.MWWK, {1: {0: 150.0, 1: 150.0}, 2: {0: 100.0, 1: 100.0}}))
    (r,) = dc.world_table(dc.arm_units(rows), [_view("two", floor=10.0)])
    # diffs 100, 0: SD 70.71, SE 50, 3 SE 150 > 10; G = 50 is not a gap
    assert r.gap.diff == 50.0 and r.gap.own3se == pytest.approx(150.0)
    assert r.gap.floor == pytest.approx(150.0) and not r.is_gap
    # a larger atlas floor wins over the own 3 SE
    (r,) = dc.world_table(dc.arm_units(rows), [_view("two", floor=400.0)])
    assert r.gap.floor == 400.0
    # stochastic vs deterministic: every TS replicate against TW's one row
    rows += _rows("two", dc.TW, {1: {0: 120.0}, 2: {0: 90.0}})
    (r,) = dc.world_table(dc.arm_units(rows), [_view("two")])
    assert r.ts_tw.pairs == 4 and r.ts_tw.diff == pytest.approx(20.0)


# --- fidelity ---


def _fid_worlds(rel: list[str]) -> list[tuple[dict, list[dict]]]:
    out = []
    for i, k in enumerate(rel):
        mwwk = {"eq": 100.0, "below": 150.0, "above": 50.0}[k]
        out.append(_det_world(f"x{i:02d}", 130.0, 100.0, 100.0, mwwk=mwwk, floor=10.0))
    return out


def _fidelity(rel: list[str]) -> dict:
    worlds = _fid_worlds(rel)
    views = [v for v, _ in worlds]
    rows = [r for _, rs in worlds for r in rs]
    return dc.fidelity_reading(dc.world_table(dc.arm_units(rows), views))


def test_fidelity_both_hold_is_mixed() -> None:
    """Spec "Fidelity readings both hold": 8 below / 8 above / 7 equivalent of 23 -> mixed."""
    f = _fidelity(["below"] * 8 + ["above"] * 8 + ["eq"] * 7)
    assert (f["reading"], f["below"], f["above"], f["equivalent"]) == ("mixed", 8, 8, 7)
    assert f["loss_rule"] and f["beyond_rule"]


def test_fidelity_branches() -> None:
    assert _fidelity(["eq"] * 16 + ["below"] * 7)["reading"] == "faithful"  # 16/23 >= 2/3
    assert _fidelity(["eq"] * 15 + ["below"] * 8)["reading"] == "fidelity_loss"
    assert _fidelity(["eq"] * 15 + ["above"] * 8)["reading"] == "beyond_target"
    # neither side reaches 1/3 and not faithful -> mixed
    assert _fidelity(["eq"] * 15 + ["below"] * 4 + ["above"] * 4)["reading"] == "mixed"
    # exactly 1/3 below (7 of 21) reaches the rule
    assert _fidelity(["eq"] * 13 + ["below"] * 7 + ["above"] * 1)["reading"] == "fidelity_loss"


def test_interpretation_cells() -> None:
    assert dc.interpretation("decoder", "faithful").startswith("decode is the bottleneck")
    assert dc.interpretation("decoder_partial", "mixed").endswith("(noted)")
    assert dc.interpretation("not_decoder", "fidelity_loss").startswith("start from score")
    assert dc.interpretation("no_gap", "beyond_target").startswith("start from why")
    assert dc.interpretation("incomplete", "faithful") is None


# --- dispersion ---


def _disp_rows(world: str, history: float, share: float, cells: int = 5, pairs: int = 5,
               arm: str = dc.TS) -> list[dict]:
    return _rows(world, arm, {1: {0: 0.0, 1: 0.0}}, trace_agent_history_dispersion=history,
                 trace_agent_history_cells=cells, trace_routed_share_dispersion=share,
                 trace_routed_share_pairs=pairs,
                 trace_routed_share_dispersion_producer_streams=0.3,
                 trace_routed_share_pairs_producer_streams=2,
                 trace_routed_share_dispersion_vs_rest=0.3,
                 trace_routed_share_dispersion_unit_weighted=0.01)


def test_agent_histories_do_not_differ() -> None:
    """Spec "Agent histories do not differ": below a line on more than 1/3 of worlds ->
    history_flat (the user is asked); 1 of 3 below -> history_differs."""
    views = [_view(w) for w in ("a", "b", "c")]
    rows = _disp_rows("a", 0.2, 0.3) + _disp_rows("b", 0.01, 0.3) + _disp_rows("c", 0.2, 0.05)
    d = dc.dispersion_reading(rows, views)
    assert (d["reading"], d["passing_worlds"], d["defined_worlds"]) == ("history_flat", 1, 3)
    assert d["next"].startswith("recorded; the supervisor asks the user")
    rows = _disp_rows("a", 0.2, 0.3) + _disp_rows("b", 0.2, 0.3) + _disp_rows("c", 0.2, 0.05)
    assert dc.dispersion_reading(rows, views)["reading"] == "history_differs"
    # other arms never enter the reading
    rows += _disp_rows("a", 0.0, 0.0, arm=dc.MWW) + _disp_rows("b", 0.0, 0.0, arm=dc.RR)
    assert dc.dispersion_reading(rows, views)["reading"] == "history_differs"


def test_dispersion_undefined_borderline_and_alternatives() -> None:
    """A world whose count column is 0 is undefined (skipped, never below the line); an
    in-band value on a deciding world sets `borderline`; alternatives are reported only."""
    views = [_view(w) for w in ("a", "b", "c", "d")]
    rows = (_disp_rows("a", 0.2, 0.3) + _disp_rows("b", 0.2, 0.3)
            + _disp_rows("c", 0.0, 0.0, cells=0, pairs=0) + _disp_rows("d", 0.2, 0.0, pairs=0))
    d = dc.dispersion_reading(rows, views)
    assert d["reading"] == "history_differs" and d["defined_worlds"] == 2
    assert d["skipped_worlds"] == ["c", "d"] and d["borderline"] is False
    assert not d["borderline_changes_reading"]
    # 12-producer share 0.11 (within +-0.02 of 0.10) on a deciding world -> borderline
    rows = _disp_rows("a", 0.2, 0.11) + _disp_rows("b", 0.2, 0.3) + _disp_rows("c", 0.04, 0.3)
    d = dc.dispersion_reading(rows, views[:3])
    assert d["borderline"] is True and d["borderline_worlds"] == ["a", "c"]
    assert d["reading"] == "history_differs" and d["borderline_changes_reading"]
    # alternatives: unit-weighted (0.01) fails everywhere, producer streams pass; reported only
    alt = d["alternatives_reported_only"]
    assert alt["unit_weighted"]["reading"] == "history_flat"
    assert alt["producer_streams"]["passing_worlds"] == 2  # c fails its history line
    # 42-producer worlds never enter the 12-producer reading
    rows42 = _rows("big", dc.TS, {1: {0: 0.0}}, size=42, trace_agent_history_dispersion=0.0,
                   trace_agent_history_cells=3, trace_routed_share_dispersion=0.0,
                   trace_routed_share_pairs=3)
    d2 = dc.dispersion_reading(rows + rows42, views[:3] + [_view("big", size=42)])
    assert d2["defined_worlds"] == 3


# --- prediction cross-check (supervisor item 4) ---


def _probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pre_pod_offline", PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_pod_offline"] = mod  # its dataclasses resolve their module
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@needs_inputs
def test_prediction_cross_check(atlas: dict) -> None:
    """EXP-0052's confirm rows with MWW in TS's place reproduce task 1.2's `decoder` reading
    (17 gap worlds, 16 strong, median rec_w 0.908), world by world equal to
    `probes/pre_pod_offline.py`'s G_w, floor, gap flag and rec_w."""
    views = dc.sample_views(atlas)
    arm_of = {dc.SUBJECT[dc.TR]: dc.TR, dc.SUBJECT[dc.RR]: dc.RR, dc.SUBJECT[dc.MWW]: dc.TS}
    results = dc.world_table(dc.arm_units(dc.read_reference(EXP0052), arm_of), views)
    p = dc.primary_reading(results)
    assert (p["reading"], p["worlds"], p["gap_worlds"], p["rec_strong_worlds"]) == (
        dc.PREDICTION["reading"], 23, 17, 16)
    assert p["median_rec"] == pytest.approx(0.908, abs=5e-4)
    assert p["share_rec_ge_2_3"] == pytest.approx(dc.PREDICTION["share_rec_ge_2_3"])
    probe = _probe()
    patlas, pworlds, _, _ = probe.load()
    ref = {g.world: g for g in probe.gaps(probe.loaded(pworlds, 12, "train"), patlas, probe.MWW)}
    mine = {r.world: r for r in results if r.size == 12}
    assert set(ref) == set(mine)
    for w, g in ref.items():
        r = mine[w]
        assert (r.gap.diff, r.gap.floor, r.is_gap, r.gap.pairs) == (g.g, g.floor, g.is_gap,
                                                                     g.n_pairs), w
        assert r.rec == g.rec, w
    assert dc.primary_reading(results, 42)["reading"] == "decoder"  # 1.2: 11/14, reported


# --- keys ---


def _job(view: dict, arm: str, ckpt: str, horizon: int = 3) -> dict:
    e = dc.world_entry(view, "shocks-i-smoke" if view["shocks"] else None)
    table = [dc.decoder_arm(arm, load_roster(dc.DEFAULT_ROSTER))]
    job = build_jobs([e], table, 1, horizon, None,
                     {"orderer": {"path": ckpt, "sha256": sha256_file(ckpt)}})[0]
    job["arm"] = arm
    return job


@needs_inputs
def test_keyed_job_without_key_is_a_contract_failure(
        atlas: dict, tiny_ckpt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A TS / MWWk job whose key is not set is a job failure row (RuntimeError), never a
    silent default; with the key set (int episode / replicate from the job) it runs."""
    view = next(v for v in dc.sample_views(atlas) if v["world"] == SHOCKED)
    for arm in (dc.TS, dc.MWWK):
        ok = dc.run_decoder_job(_job(view, arm, tiny_ckpt))
        assert "check" not in ok and ok["arm"] == arm, ok
    monkeypatch.setattr(dc, "set_draw_key", lambda env, key: None)
    for arm in (dc.TS, dc.MWWK):
        bad = dc.run_decoder_job(_job(view, arm, tiny_ckpt))
        assert bad["error_type"] == "RuntimeError" and "set_draw_key" in bad["message"], bad
    # unkeyed arms never read the key
    assert "check" not in dc.run_decoder_job(_job(view, dc.RR, tiny_ckpt))


def test_draw_key_from_job() -> None:
    k = dc.job_draw_key({"world": "w", "episode": "17", "replicate": "1"})
    assert (k.world, k.episode, k.replicate) == ("w", 17, 1)


# --- reproduction ---


def _repro_row(world: str, ep: int, arm: str, rep: int = 0, units: float = 10.0,
               dd: str = "d", sd: str = "s") -> dict[str, str]:
    return {"world": world, "episode": str(ep), "subject": dc.SUBJECT[arm], "replicate": str(rep),
            "units": repr(units), "dead": "false", "demand_digest": dd, "shock_digest": sd,
            "arm": arm}


def test_reproduction_checks() -> None:
    mine = [_repro_row("w", 1, a) for a in (dc.TR, dc.RR, dc.TW, dc.TS)] + [
        _repro_row("w", 2, dc.TR)]
    ref56 = [{k: v for k, v in r.items() if k != "arm"} for r in mine[:3]]
    rows, problems = dc.reproduction(mine, dc.EXP0056, ref56)
    assert problems == [] and len(rows) == 3 and all(r["equal"] for r in rows)
    # EXP-0052 must hold every TR / RR / MWW row of the run: episode 2 is missing there
    rows, problems = dc.reproduction(mine, dc.EXP0052, ref56)
    assert [p[0] for p in problems] == [dc.REPRODUCTION] and "no reference row" in problems[0][1]
    bad = [dict(r) for r in ref56]
    bad[0]["units"], bad[1]["shock_digest"] = "11.0", "x"
    _, problems = dc.reproduction(mine, dc.EXP0056, bad)
    assert sorted(p[0] for p in problems) == sorted([dc.REPRODUCTION, DIGEST])
    # a reference row of a world-episode this run played but lacks -> problem
    _, problems = dc.reproduction(mine[1:], dc.EXP0056, ref56)
    assert problems == [(dc.REPRODUCTION, f"exp0056: w episode 1 {dc.SUBJECT[dc.TR]} r0: no row "
                                          "in this run")]


@needs_inputs
@needs_ckpt
def test_job_path_reproduces_exp0056_and_exp0052(atlas: dict) -> None:
    """Horizon 100, 12-0344's first confirm episode: TR / TW / RR equal EXP-0056's rows and TR /
    RR / MWW r0 EXP-0052's (units, death, digests) through `run_decoder_job` (key set,
    ScoreFidelity on the scorer)."""
    view = next(v for v in dc.sample_views(atlas) if v["world"] == "ws-atlas-v1-12-0344")
    e = dc.world_entry(view)
    e = replace(e, episode_seeds=e.episode_seeds[:1])
    roster = load_roster(dc.DEFAULT_ROSTER)
    pin = {"orderer": {"path": str(ORDERER), "sha256": sha256_file(ORDERER)}}
    table = [dc.decoder_arm(a, roster) for a in (dc.TR, dc.TW, dc.RR, dc.MWW)]
    jobs = [j for j in build_jobs([e], table, 1, 100, None, pin) if j["replicate"] == 0]
    arm = {t["subject"]: t["arm"] for t in table}
    rows = []
    for j in jobs:
        j["arm"] = arm[j["subject"]]
        rows.append(dc.run_decoder_job(j))
    assert all("check" not in r for r in rows), rows
    got56, p56 = dc.reproduction(rows, dc.EXP0056, dc.read_reference(EXP0056))
    got52, p52 = dc.reproduction(rows, dc.EXP0052, dc.read_reference(EXP0052))
    assert p56 == [] and p52 == []
    assert sorted(r["arm"] for r in got56) == ["RR", "TR", "TW"]
    assert sorted(r["arm"] for r in got52) == ["MWW", "RR", "TR"]
    assert all(r["equal"] for r in got56 + got52)
    tr = next(r for r in rows if r["arm"] == "TR")
    assert int(tr["score_decisions"]) > 0 and tr["score_kl_to_makeable"] != ""


# --- real run and dry run ---


@needs_inputs
def test_real_run_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real run needs the registered atlas, orderer and references, a 40-hex commit, horizon
    100 and no dry-run filter: each one alone is refused, naming it."""
    monkeypatch.setattr(dc, "checked_orderer", lambda out, path, log: {
        "path": path, "sha256": dc.ROLE_CHECKPOINTS["orderer"]})
    def never(job: dict) -> dict:  # a guard that lets a run through fails here, fast
        raise AssertionError(f"a refused run started a job: {job['world']} {job['subject']}")

    kw = dict(atlas_path=ATLAS_V2, out_dir=tmp_path / "o", orderer_ckpt="o.pt",
              roster=load_roster(dc.DEFAULT_ROSTER), source_commit="a" * 40,
              exp0052_episodes=EXP0052, exp0056_episodes=EXP0056, run_job_fn=never)
    other = tmp_path / "other.csv"
    other.write_text("world\n")
    cases = [({"worlds": [SHOCKED]}, "--worlds is set"),
             ({"max_worlds": 1}, "--max-worlds is set"),
             ({"arms": ["TR"]}, "--arms is set"),
             ({"shock_profile": "shocks-i-smoke"}, "--shock-profile is set"),
             ({"horizon": 12}, "horizon 12 != 100"),
             ({"source_commit": ""}, "no full 40-hex source commit"),
             ({"exp0052_episodes": None}, "--exp0052-episodes None"),
             ({"exp0056_episodes": other}, "is not the registered EXP-0056"),
             ({"roster": build_roster_v2()}, "is not frozen")]
    for edit, msg in cases:
        with pytest.raises(ValueError, match="not a valid decoder-check run") as err:
            dc.run_decoder_check(**{**kw, **edit})
        assert msg in str(err.value), (edit, str(err.value))
        inner = str(err.value).split("(", 1)[1].rsplit("); pass", 1)[0]
        assert ";" not in inner, str(err.value)
    # an atlas that loads (body edited, content hash recomputed) but is not the registered one
    doc = json.loads(ATLAS_V2.read_text())
    doc["name"] = str(doc.get("name")) + "-edited"
    doc["content_hash"] = content_hash(doc)
    edited = tmp_path / "atlas_v2.json"
    edited.write_text(json.dumps(doc))
    assert load_atlas(edited)["content_hash"] != dc.REGISTERED_ATLAS_HASH
    with pytest.raises(ValueError, match="not a valid decoder-check run") as err:
        dc.run_decoder_check(**{**kw, "atlas_path": edited})
    assert "is not the registered EXP-0052 reclassified atlas" in str(err.value)
    # a sample that is not D3's 23 / 14 (with the EXP-0056 worlds) is refused
    full = dc.sample_views
    for drop in (lambda v: v["world"] == "ws-atlas-v1-42-0031",
                 lambda v: v["world"] == dc.EXP0056_WORLDS[1]):
        monkeypatch.setattr(dc, "sample_views",
                            lambda atlas, drop=drop: [v for v in full(atlas) if not drop(v)])
        with pytest.raises(ValueError, match="the sample has .* D3 registers"):
            dc.run_decoder_check(**kw)
    monkeypatch.setattr(dc, "sample_views", full)
    monkeypatch.setattr(dc, "checked_orderer", lambda out, path, log: {
        "path": path, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="orderer checkpoint 000000000000 is not the registered"):
        dc.run_decoder_check(**kw)


@needs_inputs
@needs_ckpt
def test_dry_run_layout(tmp_path: Path) -> None:
    """1 world, horizon 12, shocks-i-smoke, --workers 1: every output, 18 rows (2 episodes x
    TR, TS x 2, MWWk x 2, MWW x 2, TW, RR); a rerun on the same dir resumes; a run paired with
    the first as its references reproduces every row; a tampered reference is invalid."""
    out = tmp_path / "EXP-dry"
    args = ["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run",
            "--worlds", SHOCKED, "--horizon", "12", "--shock-profile", "shocks-i-smoke",
            "--workers", "1"]
    started = time.perf_counter()
    assert dc.main([*args, "--out-dir", str(out)]) == 0
    assert time.perf_counter() - started < 120
    for f in ("spec.json", "episodes.csv", "decoder_check.csv", "fidelity.csv", "dispersion.csv",
              "reading.json", "reproduction.csv", "routing_metrics.csv", "contract_failures.csv",
              "RESULT.md"):
        assert (out / f).exists(), f
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    spec = json.loads((out / "spec.json").read_text())
    assert spec["kind"] == "dry-run" and spec["status"] == "complete"
    episodes = list(csv.DictReader((out / "episodes.csv").open()))
    assert len(episodes) == 18
    assert sorted({(e["arm"], e["replicate"]) for e in episodes}) == [
        ("MWW", "0"), ("MWW", "1"), ("MWWk", "0"), ("MWWk", "1"), ("RR", "0"), ("TR", "0"),
        ("TS", "0"), ("TS", "1"), ("TW", "0")]
    for e in episodes:
        assert e["trace_routed_share_pairs"] != ""
        assert (e["score_decisions"] != "") == (e["arm"] in ("TR", "TS", "TW"))
    reading = json.loads((out / "reading.json").read_text())
    assert reading["kl_target"].startswith("softmax(makeable_logits(m))")
    assert set(reading["interpretation_table"]) == set(dc.INTERPRETATION)
    assert reading["reported_42"]["reported"] is True and reading["dry_run"] is True
    assert reading["prediction_task_1_2"]["reading"] == "decoder"
    for k in ("primary", "fidelity", "dispersion"):
        assert reading[k]["reading"] != "incomplete", k
    assert "borderline" in reading["dispersion"]
    check = list(csv.DictReader((out / "decoder_check.csv").open()))
    assert [r["scope"] for r in check] == ["world", "cell", "rho_quartile"]
    disp = list(csv.DictReader((out / "dispersion.csv").open()))
    assert sorted(r["arm"] for r in disp) == sorted(dc.ARMS)
    assert "producer_streams_reported" in disp[0]
    text = (out / "RESULT.md").read_text()
    assert "not an acceptance read" in text and "17 gap worlds of 23" in text
    assert "median rec_w 0.908" in text and "makeable_logits" in text

    logs: list[str] = []
    assert dc.run_decoder_check(
        atlas_path=ATLAS_V2, out_dir=out, orderer_ckpt=str(ORDERER),
        roster=load_roster(dc.DEFAULT_ROSTER), worlds=[SHOCKED], horizon=12,
        shock_profile="shocks-i-smoke", dry_run=True, log=logs.append) == 0
    assert any("18 already done, 0 to run" in m for m in logs)

    again = tmp_path / "EXP-again"
    ref = out / "episodes.csv"
    assert dc.main([*args, "--out-dir", str(again), "--exp0052-episodes", str(ref),
                    "--exp0056-episodes", str(ref)]) == 0
    repro = list(csv.DictReader((again / "reproduction.csv").open()))
    # EXP-0052 arms TR, RR, MWW x2 and EXP-0056 arms TR, TW, RR, each on 2 episodes
    assert len(repro) == 8 + 6 and all(r["equal"] == "true" for r in repro)

    bad = tmp_path / "bad.csv"
    rows = [{**r, "units": "0.0"} if r["arm"] == "TW" else r
            for r in csv.DictReader(ref.open())]
    with bad.open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert dc.main([*args, "--out-dir", str(tmp_path / "EXP-bad"), "--exp0052-episodes",
                    str(ref), "--exp0056-episodes", str(bad)]) == dc.CONTRACT_INVALID


# --- invalid runs (review round 1) ---

OUTPUTS = ("spec.json", "episodes.csv", "decoder_check.csv", "fidelity.csv", "dispersion.csv",
           "reading.json", "reproduction.csv", "routing_metrics.csv", "contract_failures.csv",
           "RESULT.md")


def _small_run(out: Path, run_job_fn: object) -> int:
    """Dry run without torch: RR + MWWk on 12-0060, horizon 12, smoke (6 jobs)."""
    return dc.run_decoder_check(
        atlas_path=ATLAS_V2, out_dir=out, orderer_ckpt=str(ORDERER),
        roster=load_roster(dc.DEFAULT_ROSTER), worlds=[SHOCKED], arms=["RR", "MWWk"],
        horizon=12, shock_profile="shocks-i-smoke", dry_run=True,
        run_job_fn=run_job_fn, log=lambda m: None)  # type: ignore[arg-type]


def _assert_withheld(out: Path) -> None:
    for f in OUTPUTS:
        assert (out / f).exists(), f
    reading = json.loads((out / "reading.json").read_text())
    assert reading["status"] == "invalid" and reading["readings_withheld"] is True
    assert reading["licence"] is None
    assert not {"primary", "fidelity", "dispersion", "score_fidelity", "reported_42"} & set(reading)
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"
    assert "INVALID" in (out / "RESULT.md").read_text()


@needs_inputs
@needs_ckpt
def test_digest_difference_across_arms_exits_2(tmp_path: Path) -> None:
    """A demand digest that differs across arms of one world-episode is a `digest_pairing` run
    row and exit 2 with every output written and the readings withheld (not a traceback)."""
    def planted(job: dict) -> dict:
        row = dc.run_decoder_job(job)
        if job["arm"] == "MWWk" and job["replicate"] == 1:
            row["demand_digest"] = "0" * 64
        return row

    out = tmp_path / "EXP-digest"
    assert _small_run(out, planted) == dc.CONTRACT_INVALID
    failures = list(csv.DictReader((out / "contract_failures.csv").open()))
    assert [f["check"] for f in failures] == [DIGEST]
    _assert_withheld(out)
    # a resume of the invalid run stays invalid (the failure row is kept)
    assert _small_run(out, dc.run_decoder_job) == dc.CONTRACT_INVALID
    _assert_withheld(out)


@needs_inputs
@needs_ckpt
def test_partial_rows_withhold_readings(tmp_path: Path) -> None:
    """A failed MWWk replicate: exit 2, `reading.json` invalid with no reading or licence."""
    def failing(job: dict) -> dict:
        if job["arm"] == "MWWk" and job["replicate"] == 1:
            return job_failure(job, RuntimeError("planted")).to_dict()
        return dc.run_decoder_job(job)

    out = tmp_path / "EXP-partial"
    assert _small_run(out, failing) == dc.CONTRACT_INVALID
    _assert_withheld(out)


def test_missing_replicate_is_incomplete_not_paired() -> None:
    """A TS arm missing replicate 1 is incomplete: no TS - MWWk pairing from the remaining row
    (the old broadcast against every replicate), so fidelity reads `incomplete`."""
    view, rows = _det_world("w", 130.0, 100.0, 120.0)
    rows = [r for r in rows if not (r["arm"] == dc.TS and r["replicate"] == "1")]
    (r,) = dc.world_table(dc.arm_units(rows), [view])
    assert r.fidelity is None and r.ts_tw is None and dc.TS not in r.means
    assert dc.fidelity_reading([r])["reading"] == "incomplete"
    assert dc.primary_reading([r])["reading"] == "incomplete"
