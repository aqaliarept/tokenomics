"""Task 5.3: step-2 validation readings and the frozen pre-registration (add-m4-orderer-redesign
design D6 / D7 / D9; learned-orderer "Ordered ES readings", "Paired references and noise floors",
"Herding and concentration are measured", "History effect is recorded", "Detectable effect
pre-registered")."""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
import statistics
from collections.abc import Callable
from pathlib import Path

import pytest

from tokenomics.eval.atlas import HELDOUT, HELDOUT_REGION, Spread, TRAIN, VALIDATION
from tokenomics.eval.contract import job_failure
from tokenomics.eval.provenance import sha256_file
from tokenomics.train import order_es as es
from tokenomics.train import order_es_validate as v
from tokenomics.train import order_es_worlds as ow

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ATLAS_WORLDS = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/worlds.csv"
EXP0052 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/artifacts/episodes.csv"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
DRAFT = REPO / "openspec/changes/add-m4-orderer-redesign/es_prereg_m4_v1.draft.json"
VAL_WORLD = "ws-atlas-v1-12-0350"  # a loaded 12-producer validation world


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_inputs = pytest.mark.skipif(not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)),
                                  reason="EXP-0052 atlas / EXP-0012 checkpoint absent")
QUIET: Callable[[str], None] = lambda m: None  # noqa: E731


# --- hand-built numbers ---

ZERO = Spread(0.0, 10, 0.0, 10)  # atlas floor = 0.01 * bar only


def _cells(n: int = 6, spread: Spread | None = ZERO) -> dict[str, v.Cell]:
    out = {f"w{i}": v.Cell(f"w{i}", 12, "dense", True, spread, 0.5 + i / 100, "hard")
           for i in range(n)}
    out["light"] = v.Cell("light", 12, "dense", True, spread, 0.1, "open")
    out["coll"] = v.Cell("coll", 12, "dense", True, spread, 0.9, "collapse")
    out["big"] = v.Cell("big", 42, "dense", True, spread, 0.6, "hard")
    return out


def _row(world: str, arm: str, ep: int, rep: int, units: float, **kw: object) -> dict[str, str]:
    d = {"world": world, "arm": arm, "subject": arm, "episode": str(ep), "replicate": str(rep),
         "units": repr(float(units)), "dead": "false", "r_start": "1.0", "r_end": "1.0",
         "consumer_fill_rate": json.dumps({"c0": 1.0}),
         v.TOP_SHARE: "0.5", v.TOP_SHARE_N: "3", v.SWING: "0.1", v.SWING_N: "5"}
    d.update({k: str(x) for k, x in kw.items()})
    return d


REPS = {v.RR: 1, v.TR: 1, v.MWW: 2, v.A0: 2}


def _dicts(cells: dict[str, v.Cell], units: dict[str, Callable[[str, int, int], float]],
           eps: int = 2, extra: Callable[[str, str, int, int], dict] | None = None
           ) -> list[dict[str, str]]:
    """Rows for every world, arm, episode and replicate (candidates / ablations R = 2)."""
    out = []
    for w in cells:
        for arm, f in units.items():
            for ep in range(1, eps + 1):
                for rep in range(REPS.get(arm, 2)):
                    out.append(_row(w, arm, ep, rep, f(w, ep, rep),
                                    **(extra(w, arm, ep, rep) if extra else {})))
    return out


def _noise(ep: int, rep: int) -> float:
    return (ep - 1.5) * 2 + (rep - 0.5) * 0.5


def _base_units(cand: float = 1100.0, abl: float = 1100.0, a0: float = 1000.0,
                labels: tuple[str, ...] = ("A1-s0",), **over: float
                ) -> dict[str, Callable[[str, int, int], float]]:
    u: dict[str, Callable[[str, int, int], float]] = {
        v.RR: lambda w, e, r: 1000.0, v.TR: lambda w, e, r: 900.0,
        v.MWW: lambda w, e, r: 1000.0 + (r - 0.5), v.A0: lambda w, e, r: a0 + _noise(e, r)}
    for lab in labels:
        c = over.get(lab, cand)
        a = over.get(lab + "/abl", abl)
        u[lab] = lambda w, e, r, c=c: c + _noise(e, r)
        u[v.ablation_label(lab)] = lambda w, e, r, a=a: a + _noise(e, r)
    return u


def _run(label: str, g_tr: float = 200.0, f_tr: float = 50.0, imp: float | None = 0.0
         ) -> v.RunStats:
    arm, seed = label.rsplit("-s", 1)
    return v.RunStats(label, arm, int(seed), 40, g_tr, f_tr, imp)


def _prun(label: str, imp: float | None = 0.0) -> v.RunStats:
    """A run for plateau scenarios: g_tr 9 clears F_tr 5, and a gain of 3 is no kill."""
    return _run(label, g_tr=9.0, f_tr=5.0, imp=imp)


# --- floors and the n = 1 rule ---


def test_floors_per_comparison_and_the_n1_rule() -> None:
    cell = v.Cell("w", 12, "dense", True, Spread(30.0, 10, 40.0, 10), 0.5, "hard")
    # one episode, two deterministic arms: 1 paired observation -> the atlas floor alone
    wc = v.world_comp({7: {0: 1100.0}}, {7: {0: 1000.0}}, cell, [], 1000.0)
    assert wc.pairs == 1 and wc.own3se is None
    assert wc.atlas_floor == pytest.approx(max(3 * 30.0, 10.0)) and wc.floor == wc.atlas_floor
    # the atlas SE scales with the read's counts: n episodes, the stochastic arms' R
    wc4 = v.world_comp({e: {0: 1100.0, 1: 1100.0} for e in range(4)},
                       {e: {0: 1000.0} for e in range(4)}, cell, [2], 1000.0)
    assert wc4.atlas_se == pytest.approx(math.sqrt(30.0 ** 2 + 40.0 ** 2 / 2) / 2)
    assert wc4.pairs == 8 and wc4.own3se == 0.0 and wc4.floor == wc4.atlas_floor
    # the own 3 SE wins when the paired observations are noisy
    noisy = v.world_comp({1: {0: 1300.0, 1: 900.0}, 2: {0: 1300.0, 1: 900.0}},
                         {1: {0: 1000.0, 1: 1000.0}, 2: {0: 1000.0, 1: 1000.0}}, cell, [2, 2],
                         1000.0)
    diffs = [300.0, -100.0, 300.0, -100.0]
    assert noisy.own3se == pytest.approx(3 * statistics.stdev(diffs) / 2)
    assert noisy.floor == noisy.own3se > noisy.atlas_floor
    # a world without a spread (no pooled df) is never cleared (infinite floor)
    lost = v.Cell("x", 12, "dense", True, None, 0.5, "hard")
    assert math.isinf(v.world_comp({1: {0: 1.0}}, {1: {0: 0.0}}, lost, [], 1.0).floor)
    # 0.01 bar is a lower bound of the atlas floor
    flat = v.Cell("f", 12, "dense", False, ZERO, 0.5, "hard")
    assert v.world_comp({1: {0: 5.0}}, {1: {0: 0.0}}, flat, [], 4000.0).floor == 40.0


def test_world_mean_floor_pools_and_keeps_the_n1_rule() -> None:
    cells = {w: v.Cell(w, 12, "dense", True, Spread(20.0, 10, 0.0, 10), 0.5, "hard")
             for w in ("a", "b", "c")}
    units = {"a": {"x": {1: {0: 10.0}, 2: {0: 14.0}}, "y": {1: {0: 0.0}, 2: {0: 0.0}}},
             "b": {"x": {1: {0: 20.0}, 2: {0: 26.0}}, "y": {1: {0: 0.0}, 2: {0: 0.0}}},
             "c": {"x": {1: {0: 5.0}}, "y": {1: {0: 0.0}}}}  # one episode: no own df
    bars = {"a": 100.0, "b": 100.0, "c": 100.0}
    mc = v.mean_comp(units, cells, "x", "y", {}, bars, ["a", "b", "c"])
    assert mc.diff == pytest.approx((12 + 23 + 5) / 3)
    s2 = ((2 ** 2 + 2 ** 2) + (3 ** 2 + 3 ** 2)) / 2  # pooled over a and b, df 2
    assert mc.df == 2
    assert mc.own3se == pytest.approx(3 * math.sqrt(s2 / 2 + s2 / 2 + s2 / 1) / 3)
    se_w = [20.0 / math.sqrt(2), 20.0 / math.sqrt(2), 20.0]
    assert mc.atlas_floor == pytest.approx(3 * math.sqrt(sum(x * x for x in se_w)) / 3)
    assert mc.floor == max(mc.atlas_floor, mc.own3se)
    # no world with >= 2 paired observations: the atlas floor alone
    only_c = v.mean_comp(units, cells, "x", "y", {}, bars, ["c"])
    assert only_c.own3se is None and only_c.df == 0 and only_c.floor == only_c.atlas_floor
    # a missing arm on a world: no comparison from partial rows
    assert v.mean_comp(units, cells, "x", "z", {}, bars, ["a"]) is None
    # the compared stochastic arms enter the atlas term at their R (reps mapping)
    rcell = {"a": v.Cell("a", 12, "dense", False, Spread(0.0, 10, 40.0, 10), 0.5, "hard")}
    ru = {"a": {"s": {1: {0: 10.0, 1: 10.0}}, "d": {1: {0: 0.0}}}}
    det = v.mean_comp(ru, rcell, "s", "d", {}, {"a": 1.0}, ["a"])
    sto = v.mean_comp(ru, rcell, "s", "d", {"s": 2}, {"a": 1.0}, ["a"])
    assert det.atlas_floor == pytest.approx(0.01)
    assert sto.atlas_floor == pytest.approx(3 * 40.0 / math.sqrt(2))
    # "Gain inside the floor": a tie
    tie = v.MeanComp("x", "y", 5.0, 3, 5.0, None, 0, ())
    assert not tie.clears and tie.within


# --- readings ---


def _analyse(units: dict, runs: dict[str, v.RunStats], cells: dict | None = None,
             extra: Callable | None = None, **kw: object) -> dict:
    cells = cells or _cells()
    return v.analyse(_dicts(cells, units, extra=extra), cells, runs, **kw)


def test_success_and_g_val_against_a0() -> None:
    got = _analyse(_base_units(a0=1040.0), {"A1-s0": _run("A1-s0", g_tr=150.0)})
    s0 = got["seed0"]
    assert s0["reading"] == v.SUCCESS and got["final"]["reading"] == v.SUCCESS
    assert s0["selected"] == "A1-s0"
    # g_val = candidate - A0 (not - RR) on the loaded 12-producer worlds only
    assert s0["g_val"] == pytest.approx(1100 - 1040)
    assert got["gating_worlds"] == [f"w{i}" for i in range(6)]
    cand = got["candidates"]["A1-s0"]
    assert cand["comparisons"]["RR"]["diff"] == pytest.approx(100.0)
    # light, collapse and 42-producer worlds do not gate ("Light corner excluded")
    cells = _cells()
    units = _base_units(cand=1000.0, abl=1000.0)
    rows = [d for d in _dicts(cells, units)]
    for d in rows:
        if d["world"] in ("light", "coll", "big") and d["arm"] == "A1-s0":
            d["units"] = "9999.0"
    assert v.analyse(rows, cells, {"A1-s0": _run("A1-s0")})["seed0"]["reading"] != v.SUCCESS


def test_success_clauses() -> None:
    runs = {"A1-s0": _run("A1-s0")}
    # lost to RR beyond floor_w on 2 of 6 worlds (> 1/6): not success
    lose = _base_units()
    lose["A1-s0"] = lambda w, e, r: (800.0 if w in ("w0", "w1") else 1300.0) + _noise(e, r)
    got = _analyse(lose, runs)
    assert got["candidates"]["A1-s0"]["lost_to_rr"] == 2
    assert got["candidates"]["A1-s0"]["success_clauses"]["lost_to_rr"] is False
    assert got["seed0"]["reading"] != v.SUCCESS
    # exactly 1 of 6 lost is allowed
    lose["A1-s0"] = lambda w, e, r: (800.0 if w == "w0" else 1300.0) + _noise(e, r)
    assert _analyse(lose, runs)["seed0"]["reading"] == v.SUCCESS
    # one death of the candidate
    dead = _analyse(_base_units(), runs, extra=lambda w, a, e, r: {"dead": "true"} if (
        a == "A1-s0" and w == "w3" and e == 1 and r == 0) else {})
    assert dead["candidates"]["A1-s0"]["deaths"] == 1 and dead["seed0"]["reading"] != v.SUCCESS
    # run-down share worse than RR's
    rd = _analyse(_base_units(), runs, extra=lambda w, a, e, r: {"r_end": "0.5"} if (
        a == "A1-s0" and w == "w2") else {})
    assert rd["candidates"]["A1-s0"]["success_clauses"]["flags_no_worse"] is False
    # starvation equal to RR's is no worse
    st = _analyse(_base_units(), runs, extra=lambda w, a, e, r: {
        "consumer_fill_rate": json.dumps({"c0": 0.5})} if (
        a in ("A1-s0", v.RR) and w == "w2") else {})
    c = st["candidates"]["A1-s0"]
    assert c["starved_share"] == c["rr_starved_share"] > 0 and st["seed0"]["reading"] == \
        v.SUCCESS
    worse = _analyse(_base_units(), runs, extra=lambda w, a, e, r: {
        "consumer_fill_rate": json.dumps({"c0": 0.5})} if (a == "A1-s0" and w == "w2") else {})
    assert worse["candidates"]["A1-s0"]["success_clauses"]["flags_no_worse"] is False
    assert worse["seed0"]["reading"] != v.SUCCESS
    # a reference within its floor: not success (beats_refs needs every one)
    near = _base_units(cand=1005.0)
    assert _analyse(near, runs)["seed0"]["reading"] != v.SUCCESS


def test_cross_world_failure_is_kill_even_with_success() -> None:
    """Spec "Cross-world failure": g_tr > F_tr but g_val < g_tr / 3 -> kill although success
    holds."""
    got = _analyse(_base_units(), {"A1-s0": _run("A1-s0", g_tr=400.0)})
    assert got["seed0"]["clauses"][v.SUCCESS] and got["seed0"]["clauses"][v.KILL]
    assert got["seed0"]["reading"] == v.KILL
    # exactly g_tr / 3 is no kill
    assert _analyse(_base_units(), {"A1-s0": _run("A1-s0", g_tr=300.0)}
                    )["seed0"]["reading"] == v.SUCCESS


def test_unresolved_takes_precedence() -> None:
    runs = {"A1-s0": _run("A1-s0", g_tr=50.0, f_tr=50.0),
            "A2-s0": _run("A2-s0", g_tr=10.0, f_tr=50.0)}
    got = _analyse(_base_units(labels=("A1-s0", "A2-s0")), runs)
    assert got["seed0"]["reading"] == v.UNRESOLVED
    runs["A2-s0"] = _run("A2-s0", g_tr=60.0)
    assert _analyse(_base_units(labels=("A1-s0", "A2-s0")), runs)["seed0"]["reading"] != \
        v.UNRESOLVED


def _plateau_units() -> dict:
    # candidate within MWW's floor, A2 == A1, history inert
    return _base_units(cand=1003.0, abl=1003.0, labels=("A1-s0", "A2-s0"))


def test_plateau_and_any_arm_still_improving() -> None:
    stopped = {"A1-s0": _prun("A1-s0", imp=1.0), "A2-s0": _prun("A2-s0", imp=4.9)}
    got = _analyse(_plateau_units(), stopped)
    assert got["seed0"]["plateau"] == {"i": True, "ii": True, "iii": True}
    assert got["seed0"]["reading"] == v.PLATEAU and got["seed0"]["selected"] == "A1-s0"
    # spec "Any arm still improving": the non-selected arm improving -> budget_limited
    improving = dict(stopped, **{"A2-s0": _prun("A2-s0", imp=5.0)})
    got = _analyse(_plateau_units(), improving)
    assert got["seed0"]["selected"] == "A1-s0"
    assert got["seed0"]["reading"] == v.BUDGET_LIMITED
    # an improvement unknown (no monitor 10 generations back) counts as still improving
    unknown = dict(stopped, **{"A2-s0": _prun("A2-s0", imp=None)})
    assert _analyse(_plateau_units(), unknown)["seed0"]["reading"] == v.BUDGET_LIMITED
    # clause (ii) fails (A2 well above A1) and nothing improves -> other
    split = _base_units(labels=("A1-s0", "A2-s0"), **{"A1-s0": 990.0, "A1-s0/abl": 990.0,
                                                       "A2-s0": 1009.0, "A2-s0/abl": 1009.0})
    got = _analyse(split, stopped)
    assert got["seed0"]["selected"] == "A2-s0" and not got["seed0"]["plateau"]["ii"]
    assert got["seed0"]["reading"] == v.OTHER


def test_selection_tie_goes_to_a1() -> None:
    runs = {"A1-s0": _run("A1-s0"), "A2-s0": _run("A2-s0")}
    tie = _base_units(labels=("A1-s0", "A2-s0"), **{"A1-s0": 1100.0, "A2-s0": 1105.0})
    got = _analyse(tie, runs)
    assert got["seed0"]["a2_vs_a1"]["diff"] == pytest.approx(5.0)
    assert got["seed0"]["selected"] == "A1-s0"
    clear = _base_units(labels=("A1-s0", "A2-s0"), **{"A1-s0": 1100.0, "A2-s0": 1200.0})
    assert _analyse(clear, runs)["seed0"]["selected"] == "A2-s0"


def test_seed_combination_table() -> None:
    """D7's table, every row (spec "Seed disagreement", "Seed-1 failure carries through")."""
    assert v.combine(v.SUCCESS, v.SUCCESS) == v.SUCCESS
    assert v.combine(v.PLATEAU, v.PLATEAU) == v.PLATEAU
    assert v.combine(v.SUCCESS, v.PLATEAU) == v.SEED_DISAGREEMENT
    assert v.combine(v.PLATEAU, v.SUCCESS) == v.SEED_DISAGREEMENT
    for s0 in (v.SUCCESS, v.PLATEAU):
        for s1 in (v.UNRESOLVED, v.KILL, v.BUDGET_LIMITED, v.OTHER):
            assert v.combine(s0, s1) == s1
    for s0 in (v.UNRESOLVED, v.KILL, v.BUDGET_LIMITED, v.OTHER):
        with pytest.raises(ValueError, match="seed 1 runs only"):
            v.combine(s0, v.SUCCESS)
    reg = v.readings_registration()["seed_combination"]["table"]
    assert all(v.combine(a, b) == c for a, b, c in reg)


def _seed1(units: dict, s0runs: dict, run1: v.RunStats) -> dict:
    return _analyse(units, {run1.label: run1}, seed_combine=True, seed0_runs=s0runs)


def test_seed1_reads_and_combines() -> None:
    s0 = {"A1-s0": _run("A1-s0")}
    both = _base_units(labels=("A1-s0", "A1-s1"))
    got = _seed1(both, s0, _run("A1-s1"))
    assert (got["seed0"]["reading"], got["seed1"]["reading"], got["final"]["reading"]) == (
        v.SUCCESS, v.SUCCESS, v.SUCCESS)
    assert got["final"]["selected"] == "A1-s0"
    # seed-1 failure carries through (kill), with its Next
    got = _seed1(both, s0, _run("A1-s1", g_tr=900.0))
    assert got["seed1"]["reading"] == v.KILL and got["final"]["reading"] == v.KILL
    assert got["final"]["next"] == v.NEXT[v.KILL]
    # success then plateau -> seed_disagreement
    mixed = _base_units(labels=("A1-s0", "A1-s1"), **{"A1-s1": 1003.0, "A1-s1/abl": 1003.0})
    s0p = {"A1-s0": _run("A1-s0", imp=0.0), "A2-s0": _run("A2-s0", imp=0.0)}
    units = _base_units(labels=("A1-s0", "A2-s0", "A1-s1"),
                        **{"A1-s1": 1003.0, "A1-s1/abl": 1003.0, "A2-s0": 1101.0})
    got = _seed1(units, s0p, _prun("A1-s1", imp=0.0))
    assert got["seed0"]["reading"] == v.SUCCESS and got["seed1"]["reading"] == v.PLATEAU
    assert got["final"]["reading"] == v.SEED_DISAGREEMENT
    del mixed
    # clause (ii) comes from seed 0: A2 far below A1 there -> the seed-1 read cannot plateau
    far = _base_units(labels=("A1-s0", "A2-s0", "A1-s1"),
                      **{"A2-s0": 1040.0, "A1-s1": 1003.0, "A1-s1/abl": 1003.0})
    got = _seed1(far, s0p, _prun("A1-s1", imp=0.0))
    assert got["seed0"]["reading"] == v.SUCCESS and got["seed0"]["plateau"]["ii"] is False
    assert got["seed1"]["plateau"] == {"i": True, "ii": False, "iii": True}
    assert got["seed1"]["reading"] == v.OTHER and got["final"]["reading"] == v.OTHER
    # clause (i) includes seed 0's: seed 0 success with both arms still improving, seed 1
    # stopped and plateau-shaped -> seed-1 clause (i) False -> other (review round 1)
    climbing = {"A1-s0": _run("A1-s0", imp=100.0), "A2-s0": _run("A2-s0", imp=100.0)}
    shaped = _base_units(labels=("A1-s0", "A2-s0", "A1-s1"),
                         **{"A1-s1": 1003.0, "A1-s1/abl": 1003.0})
    got = _seed1(shaped, climbing, _prun("A1-s1", imp=0.0))
    assert got["seed0"]["reading"] == v.SUCCESS and got["seed0"]["plateau"]["i"] is False
    assert got["seed1"]["plateau"] == {"i": False, "ii": True, "iii": True}
    assert got["seed1"]["reading"] == v.OTHER and got["final"]["reading"] == v.OTHER
    # a seed-1 read after a seed-0 failure is refused
    with pytest.raises(ValueError, match="--seed-combine after"):
        _seed1(both, {"A1-s0": _run("A1-s0", g_tr=1.0)}, _run("A1-s1"))
    # the seed-1 run must be of the selected arm
    with pytest.raises(ValueError, match="selected arm"):
        _seed1(_base_units(labels=("A1-s0", "A2-s1")), s0, _run("A2-s1"))


def test_seed_1_still_improving() -> None:
    """Spec "Seed 1 still improving": seed-1 inputs meeting every plateau clause except the
    seed-1 run's own improvement >= F_tr read budget_limited, not plateau, while every seed-0
    arm has stopped improving."""
    s0 = {"A1-s0": _prun("A1-s0"), "A2-s0": _prun("A2-s0")}
    units = _base_units(cand=1003.0, abl=1003.0, labels=("A1-s0", "A2-s0", "A1-s1"))
    still = _seed1(units, s0, _prun("A1-s1", imp=5.0))
    assert still["seed0"]["reading"] == v.PLATEAU
    assert all(not r.improving for r in s0.values())
    s1 = still["seed1"]
    assert s1["plateau"] == {"i": False, "ii": True, "iii": True}
    assert s1["reading"] == v.BUDGET_LIMITED and still["final"]["reading"] == v.BUDGET_LIMITED
    # the same inputs with the seed-1 run stopped read plateau
    stopped = _seed1(units, s0, _prun("A1-s1", imp=4.9))
    assert stopped["seed1"]["reading"] == v.PLATEAU and stopped["final"]["reading"] == v.PLATEAU
    # seed-0 arms' improvement does not make a seed-1 read budget_limited: only its own does
    s0_imp = {"A1-s0": _run("A1-s0", imp=100.0), "A2-s0": _run("A2-s0", imp=100.0)}
    win = _base_units(labels=("A1-s0", "A2-s0", "A1-s1"), **{"A2-s0": 1101.0})
    got = _seed1(win, s0_imp, _run("A1-s1", imp=0.0))
    assert got["seed1"]["clauses"][v.BUDGET_LIMITED] is False


# --- labels ---


def _concentrated(cand_top: float) -> Callable:
    def extra(w: str, arm: str, e: int, r: int) -> dict:
        if arm == "A1-s0":
            return {v.TOP_SHARE: repr(cand_top + 0.01 * _noise(e, r))}
        if arm == v.A0:
            return {v.TOP_SHARE: repr(0.6 + 0.01 * _noise(e, r))}
        return {}
    return extra


def test_gain_without_deconcentration() -> None:
    runs = {"A1-s0": _run("A1-s0")}
    got = _analyse(_base_units(abl=1000.0), runs, extra=_concentrated(0.6))
    assert got["seed0"]["reading"] == v.SUCCESS
    assert v.NOT_VIA_DECONCENTRATION in got["final"]["labels"]
    got = _analyse(_base_units(abl=1000.0), runs, extra=_concentrated(0.4))
    assert v.NOT_VIA_DECONCENTRATION not in got["final"]["labels"]
    m = got["label_measures"]["A1-s0"]["deconcentration"]
    assert m["diff"] == pytest.approx(-0.2) and m["se3"] < 0.2
    # slightly below A0 but within 3 SE of its own noisy pairs: still the label
    def noisy(w: str, arm: str, e: int, r: int) -> dict:
        if arm == "A1-s0":
            return {v.TOP_SHARE: repr(0.59 + 0.05 * (1 if (e + r) % 2 else -1))}
        return {v.TOP_SHARE: "0.6"} if arm == v.A0 else {}

    got = _analyse(_base_units(abl=1000.0), runs, extra=noisy)
    m = got["label_measures"]["A1-s0"]["deconcentration"]
    assert m["diff"] == pytest.approx(-0.01) and 0.01 < m["se3"]
    assert v.NOT_VIA_DECONCENTRATION in got["final"]["labels"]
    # undefined rows (no concurrent group) are skipped, never read as 0
    undefined = _analyse(_base_units(abl=1000.0), runs, extra=lambda w, a, e, r: (
        {v.TOP_SHARE: "0.0", v.TOP_SHARE_N: "0"} if a == "A1-s0" and e == 1 else
        _concentrated(0.6)(w, a, e, r)))
    assert undefined["label_measures"]["A1-s0"]["deconcentration"]["diff"] == \
        pytest.approx(0.0, abs=0.02)


def test_not_via_deconcentration_only_under_success() -> None:
    stopped = {"A1-s0": _prun("A1-s0"), "A2-s0": _prun("A2-s0")}
    got = _analyse(_plateau_units(), stopped, extra=_concentrated(0.6))
    assert got["seed0"]["reading"] == v.PLATEAU
    assert got["label_measures"]["A1-s0"][v.NOT_VIA_DECONCENTRATION] is True
    assert v.NOT_VIA_DECONCENTRATION not in got["final"]["labels"]


def test_history_adds_nothing_measurable() -> None:
    """history_inert under success and under plateau; not when the ablation differs."""
    succ = _analyse(_base_units(abl=1100.0), {"A1-s0": _run("A1-s0")},
                    extra=_concentrated(0.4))
    assert succ["seed0"]["reading"] == v.SUCCESS and succ["final"]["labels"] == [v.HISTORY_INERT]
    stopped = {"A1-s0": _prun("A1-s0"), "A2-s0": _prun("A2-s0")}
    plat = _analyse(_plateau_units(), stopped)
    assert plat["seed0"]["reading"] == v.PLATEAU and v.HISTORY_INERT in plat["final"]["labels"]
    near = _analyse(_base_units(abl=1095.0), {"A1-s0": _run("A1-s0")},
                    extra=_concentrated(0.4))  # 5 below the candidate, inside its floor (10)
    assert near["label_measures"]["A1-s0"][v.HISTORY_INERT] is True
    assert near["candidates"]["A1-s0"]["comparisons"]["ablation"]["diff"] == pytest.approx(5.0)
    differs = _analyse(_base_units(abl=900.0), {"A1-s0": _run("A1-s0")},
                       extra=_concentrated(0.4))
    assert differs["final"]["labels"] == []
    # outside success / plateau no label is recorded, though the measures are
    kill = _analyse(_base_units(), {"A1-s0": _run("A1-s0", g_tr=900.0)})
    assert kill["final"]["reading"] == v.KILL and kill["final"]["labels"] == []
    assert kill["label_measures"]["A1-s0"][v.HISTORY_INERT] is True


def _swing(flag_worlds: set[str]) -> Callable:
    def extra(w: str, arm: str, e: int, r: int) -> dict:
        if arm == "A1-s0":
            return {v.SWING: repr((0.5 if w in flag_worlds else 0.1) + 0.001 * _noise(e, r))}
        if arm == v.MWW:
            return {v.SWING: "0.1"}
        return {}
    return extra


def test_agents_switch_together() -> None:
    """herding: candidate swing > MWW's by > 3 SE on >= 1/3 of the loaded worlds."""
    runs = {"A1-s0": _run("A1-s0")}
    got = _analyse(_base_units(abl=1000.0), runs, extra=lambda w, a, e, r: {
        **_swing({"w0", "w1"})(w, a, e, r), **_concentrated(0.4)(w, a, e, r)})
    assert got["label_measures"]["A1-s0"][v.HERDING] is True
    assert got["final"]["labels"] == [v.HERDING]
    flagged = [r for r in got["herding_rows"] if r["flagged"]]
    assert {r["world"] for r in flagged} == {"w0", "w1"}
    one = _analyse(_base_units(abl=1000.0), runs, extra=_swing({"w0"}))
    assert one["label_measures"]["A1-s0"][v.HERDING] is False  # 1 of 6 < 1/3
    # herding is recorded with plateau too
    stopped = {"A1-s0": _prun("A1-s0"), "A2-s0": _prun("A2-s0")}
    plat = _analyse(_plateau_units(), stopped, extra=_swing({"w0", "w1", "w2"}))
    assert plat["final"]["reading"] == v.PLATEAU and v.HERDING in plat["final"]["labels"]


def test_label_table_matches_the_registration() -> None:
    reg = v.readings_registration()["labels"]["table"]
    for lab, readings in reg.items():
        for r in (v.SUCCESS, v.PLATEAU):
            assert (lab in v.LABELS_WITH[r]) == (r in readings)
    assert v.labels_for(v.OTHER, {lab: True for lab in reg}) == []


def test_validation_tables() -> None:
    cells = _cells()
    rows = _dicts(cells, _base_units())
    table = v.validation_rows(rows, cells, ["A1-s0"])
    worlds = [r for r in table if r["scope"] == "world"]
    assert {r["world"] for r in worlds} == set(cells)
    c = next(r for r in worlds if r["world"] == "w0" and r["arm"] == "A1-s0")
    assert c["d_RR"] == pytest.approx(100.0) and c["floor_RR"] > 0
    groups = {(r["size"], r.get("group")) for r in table if r["scope"] == "group"}
    assert {(12, v.LOADED), (12, v.LIGHT), (12, "collapse"), (42, v.LOADED)} <= groups
    assert any(r["scope"] == "rho_quartile" for r in table)
    conc = v.concentration_rows(rows, cells)
    assert {r["arm"] for r in conc} == {"A1-s0", "A1-s0/shared", v.A0, v.MWW, v.RR}


# --- power rule ---


def test_detectable_effect_and_power_rule() -> None:
    diffs = {"a": [10.0, 30.0], "b": [0.0, 4.0], "c": [7.0]}
    got = v.detectable_effect(diffs, 1000.0, 4)
    s = math.sqrt((200 + 8) / 2)
    assert got["s"] == pytest.approx(s) and got["df"] == 2 and got["worlds"] == 3
    assert got["mde"] == pytest.approx(3 * s / 2 / math.sqrt(3))
    assert got["share"] == pytest.approx(got["mde"] / 1000.0)

    def eff(s4: float, s8: float) -> dict:
        return {4: {"share": s4}, 8: {"share": s8}}

    # the deciding pair is the max; exactly 2% at 4 episodes -> 4
    r = v.power_rule({"MWW-RR": eff(0.015, 0.011), "TR-RR": eff(0.020, 0.014)})
    assert r == {"deciding_pair": "TR-RR", "share_4": 0.020, "share_8": 0.014, "episodes": 4,
                 "to_user": False}
    r = v.power_rule({"MWW-RR": eff(0.0201, 0.0142), "TR-RR": eff(0.015, 0.011)})
    assert (r["deciding_pair"], r["episodes"], r["to_user"]) == ("MWW-RR", 8, False)
    assert v.power_rule({"x": eff(0.05, 0.031)})["to_user"] is True
    assert v.power_rule({"x": eff(0.05, 0.030)})["to_user"] is False
    with pytest.raises(ValueError, match="pooled SD"):
        v.detectable_effect({"a": [1.0]}, 1.0, 4)


@pytest.mark.skipif(not (_lfs_present(EXP0052) and _lfs_present(ATLAS_V2)),
                    reason="EXP-0052 rows absent")
def test_power_rows_reproduce_the_first_look(tmp_path: Path) -> None:
    """MWW - RR on the 9 loaded validation worlds at 4 episodes, R = 2: 1.88% of the mean bar
    1831.4 (task 1.2's independent computation). The rule's outcome is task 5.4's."""
    doc = v.run_power(exp0052_episodes=EXP0052, atlas_path=ATLAS_V2,
                      atlas_worlds_csv=ATLAS_WORLDS, out=tmp_path / "p.json", log=QUIET)
    assert len(doc["worlds"]) == 9 and doc["mean_bar"] == pytest.approx(1831.4, abs=0.05)
    assert doc["effects"]["MWW-RR"][4]["share"] == pytest.approx(0.0188, abs=5e-5)
    assert doc["effects"]["MWW-RR"][4]["df"] == 7
    assert set(doc["effects"]) == {"MWW-RR", "TR-RR"} and json.loads(
        (tmp_path / "p.json").read_text())["rule"]["registered"]["four_episodes_max_share"] == 0.02


# --- the frozen pre-registration ---


def test_frozen_prereg_file() -> None:
    """The packaged file is the frozen registration: byte-equal to the module, its hash pinned,
    its `worlds` section equal to `bar_registration()`, status frozen."""
    assert v.PREREG_PATH.read_text() == v.prereg_text(v.frozen_prereg())
    assert sha256_file(str(v.PREREG_PATH)) == v.PREREG_SHA256
    doc = json.loads(v.PREREG_PATH.read_text())
    assert doc["status"] == es.FROZEN and v.prereg_read_problems(doc) == []
    assert es.prereg_problems(doc, mode=es.TRAIN, arm="A1", opt_seed=0, generations=20,
                              minibatch=12) == []
    assert doc["worlds"] == es.bar_registration()
    for sec, want in es.registered_sections().items():
        assert doc[sec] == want, sec
    assert doc["probe"]["theta"].startswith("each arm's registered initial theta")
    n = doc["normaliser"]
    assert n["clip"] == 5.0 and "std = 0 -> 1.0" in n["zero_spread_rule"]
    assert n["scale_floors"] == {"psi.eligible_producer_count": pytest.approx(0.5108256)}
    pins = doc["readings"]["normaliser_pins"]
    assert pins["rule_sha256"] == es._sha(ow.NORMALISER_PREREG)
    assert pins["fitting_code_sha256"] == ow.normaliser_code_sha256()
    pr = doc["readings"]["power_rule"]
    assert pr["pairs"] == [["MWW", "RR"], ["TR", "RR"]] and pr["deciding"] == "max"
    assert pr["replicates"] == 2 and pr["rule"]["four_episodes_max_share"] == 0.02
    assert "outcome" in pr and "never frozen" in pr["outcome"]
    assert "structurally lower" in doc["readings"]["probe_interpretation"]
    assert not DRAFT.exists()  # the draft is superseded by the frozen file
    assert v.run_freeze(v.PREREG_PATH, log=QUIET) == 0  # MATCH, never rewritten


def test_prereg_changes_refused(tmp_path: Path) -> None:
    ok = json.loads(v.PREREG_PATH.read_text())
    for edit, where in (
            (lambda d: d["readings"]["power_rule"]["rule"].update(four_episodes_max_share=0.03),
             "readings.power_rule.rule.four_episodes_max_share"),
            (lambda d: d["readings"].update(kill_ratio=0.5), "readings.kill_ratio"),
            (lambda d: d["readings"]["labels"].update(label_se=2.0), "readings.labels.label_se"),
            (lambda d: d["normaliser"].update(clip=4.0), "normaliser.clip"),
            (lambda d: d["normaliser"]["fitting_code"].update(sha256="0" * 64),
             "normaliser.fitting_code.sha256")):
        d = copy.deepcopy(ok)
        edit(d)
        got = v.prereg_read_problems(d)
        assert len(got) == 1 and where in got[0], (where, got)
    d = copy.deepcopy(ok)
    d["status"] = es.DRAFT
    assert v.prereg_read_problems(d) == [
        "pre-registration status is 'draft', a real run needs 'frozen'"]
    # a different file refuses ES runs at launch too (order_es hook)
    other = tmp_path / "p.json"
    other.write_text(json.dumps(ok))
    assert es.prereg_file_problems(v.PREREG_PATH) == []
    got = es.prereg_file_problems(other)
    assert len(got) == 1 and "not the frozen" in got[0]
    # freeze never overwrites a differing file
    assert v.run_freeze(other, log=QUIET) == es.CONTRACT_INVALID
    assert json.loads(other.read_text()) == ok


def test_normaliser_fitting_code_hash() -> None:
    """The fitting code hash covers every listed definition, decorators included, and the code
    that builds and feeds the rows (review round 1), and changes only with them."""
    root = Path(ow.__file__).resolve().parents[1]
    for rel, names in ow.NORMALISER_CODE_DEFS.items():
        src = ow.normaliser_code_source((root / rel).read_text(), names)
        assert "<absent>" not in src, rel
    assert {"PerOrderPolicy", "shared_rows", "history_rows", "order_features"} <= set(
        ow.NORMALISER_CODE_DEFS["train/per_order_scorer.py"])
    assert "AgentHistory" in ow.NORMALISER_CODE_DEFS["env/agent_history.py"]
    assert "_product_features" in ow.NORMALISER_CODE_DEFS["train/orderer_scorer.py"]
    text = (root / "train/per_order_scorer.py").read_text()
    names = ow.NORMALISER_CODE_DEFS["train/per_order_scorer.py"]
    base = ow.normaliser_code_source(text, names)
    for old, new in (
            ("sd = 0.0 if np.ptp(col) == 0 else float(np.std(col))", "sd = float(np.std(col))"),
            # the call that feeds both the fit sink and the normaliser (review probe)
            ("raw = transform_phi(np.concatenate([shared, hist, own], axis=1)[elig])",
             "raw = np.concatenate([shared, hist, own], axis=1)[elig]"),
            ("@dataclass(frozen=True)\nclass FeatureNormaliser", "@dataclass\nclass "
                                                               "FeatureNormaliser")):
        edited = text.replace(old, new)
        assert edited != text, old
        assert ow.normaliser_code_source(edited, names) != base, old
    unrelated = text.replace('DECODER_VERSION = "per-order-choice-v1"',
                             'DECODER_VERSION = "per-order-choice-vX"')
    assert ow.normaliser_code_source(unrelated, names) == base
    fc = ow.NORMALISER_PREREG["fitting_code"]
    assert fc["sha256"] == ow.normaliser_code_sha256() and "not_covered" in fc


# --- the driver (dry runs on this box: --workers 1, horizon 12) ---


def test_validation_never_reads_held_out_worlds() -> None:
    if not _lfs_present(ATLAS_V2):
        pytest.skip("atlas absent")
    from tokenomics.eval.teacher_audit import load_atlas

    atlas = load_atlas(ATLAS_V2)
    views = v.validation_views(atlas)
    assert views and {x["split"] for x in views} == {VALIDATION}
    cells = v.world_cells(views, atlas["spreads"])
    counts = {s: sum(1 for c in cells.values() if c.size == s and c.group == v.LOADED)
              for s in (12, 42)}
    assert counts == v.EXPECTED_LOADED and len(views) == 16
    by_split = {}
    for e in atlas["worlds"]:
        by_split.setdefault(e["split"], e["record"]["world_id"])
    for split in (HELDOUT, HELDOUT_REGION, TRAIN):
        with pytest.raises(ValueError, match=f"in the {split} split"):
            v.validation_views(atlas, [by_split[split]])


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A 2-world horizon-12 dry-run bar and three fake-trained dry runs (A1-s0, A2-s0, A1-s1)."""
    if not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)):
        pytest.skip("EXP-0052 atlas / EXP-0012 checkpoint absent")
    from test_order_es import _fake

    root = tmp_path_factory.mktemp("val")
    bar = root / "bar"
    assert ow.main(["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run",
                    "--atlas-worlds", "1", "--target", "2", "--min-feasible", "1",
                    "--horizon", "12", "--shock-profile", "shocks-i-smoke", "--workers", "1",
                    "--out-dir", str(bar)]) == 0
    out = {"bar": bar}

    def train(name: str, arm: str, seed: int, gens: int, **kw: object) -> None:
        d = root / name
        assert es.run_train(bar_dir=bar, out_dir=d, orderer_ckpt=str(ORDERER), arm=arm,
                            opt_seed=seed, generations=gens, pairs=2, monitor_every=1,
                            dry_run=True, run_job_fn=_fake(bar), log=QUIET, **kw) == 0
        out[name] = d

    for arm, seed in (("A1", 0), ("A2", 0), ("A1", 1)):
        train(f"{arm}-s{seed}", arm, seed, 2)
    # the main run continues the pilot (2 + 1 generations), and its fresh 3-generation twin
    train("pilot", "A1", 0, 2)
    train("main", "A1", 0, 3, resume_from=str(es.checkpoint_path(root / "pilot", 2)))
    train("fresh3", "A1", 0, 3)
    return out


def _validate(inputs: dict[str, Path], out: Path, runs: tuple[str, ...] = ("A1-s0", "A2-s0"),
              **kw: object) -> int:
    args: dict = dict(runs=[str(inputs[r]) for r in runs], bar_dir=inputs["bar"],
                      atlas_path=ATLAS_V2, orderer_ckpt=str(ORDERER), out_dir=out, episodes=1,
                      worlds=[VAL_WORLD], horizon=12, shock_profile="shocks-i-smoke",
                      dry_run=True, log=QUIET)
    args.update(kw)
    return v.run_validate(**args)


@needs_inputs
def test_dry_run_validate_end_to_end(inputs: dict[str, Path], tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "val"
    seen: list[dict] = []
    modes: list[tuple[str, str]] = []

    class Spy(v.PerOrderPolicy):  # the worker builds each per-order arm in its history mode
        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)  # type: ignore[arg-type]
            modes.append((self.arm.name, self.history_mode))

    monkeypatch.setattr(v, "PerOrderPolicy", Spy)

    def spy(job: dict) -> dict[str, str]:
        seen.append(job)
        return v.run_validate_job(job)

    assert _validate(inputs, out, run_job_fn=spy) == 0
    # 1 world x 1 episode: A0 2 + 2 candidates x (2 + 2 ablation) + RR 1 + MWW 2 + TR 1
    assert len(seen) == 14
    # 2 replicates in the workers, plus one per arm in the parent (the per_order pins)
    assert sorted(modes) == sorted([("A0", "agent")] * 2 + [("A1", "agent"), ("A2", "agent"),
                                                           ("A1", "shared_only"),
                                                           ("A2", "shared_only")] * 3)
    for name in ("spec.json", "episodes.csv", "validation.csv", "concentration.csv",
                 "herding.csv", "reading.json", "contract_failures.csv", "RESULT.md"):
        assert (out / name).exists(), name
    rows = list(csv.DictReader((out / "episodes.csv").open()))
    assert {r["world"] for r in rows} == {VAL_WORLD}
    by_arm = {r["arm"]: r for r in rows}
    assert {by_arm[a]["ordering_decode"] for a in ("RR", "MWW", "TR")} == {"logits-v1"}
    assert {by_arm[a]["ordering_decode"] for a in ("A0", "A1-s0", "A2-s0/shared")} == \
        {"choice-v1"}
    assert by_arm["A1-s0/shared"]["history_mode"] == "shared_only"
    assert by_arm["A1-s0"]["history_mode"] == "agent"
    # candidate, ablation and A0 share keys: one digest per world-episode, tracer on
    assert len({(r["demand_digest"], r["shock_digest"]) for r in rows}) == 1
    assert all(r[v.SWING_N] != "" for r in rows)
    spec = json.loads((out / "spec.json").read_text())
    pins = spec["pins"]
    assert pins["ordering_decode"] == "choice-v1"
    assert pins["decode_by_arm"] == {"RR": "logits-v1", "MWW": "logits-v1", "TR": "logits-v1",
                                     "A0": "choice-v1", "A1-s0": "choice-v1",
                                     "A1-s0/shared": "choice-v1", "A2-s0": "choice-v1",
                                     "A2-s0/shared": "choice-v1"}
    assert pins["per_order"]["A1-s0/shared"]["history_mode"] == "shared_only"
    assert {s["split"] for s in pins["sample"]} == {VALIDATION}
    reading = json.loads((out / "reading.json").read_text())
    assert reading["status"] == "valid" and reading["final"]["reading"] in v.PRECEDENCE
    assert reading["runs"]["A1-s0"]["generation"] == 2
    # the recompute rebuilds the same reading from the rows and the runs' logs
    assert v.run_recompute(out_dir=out, runs=[str(inputs["A1-s0"]), str(inputs["A2-s0"])],
                           bar_dir=inputs["bar"], orderer_ckpt=str(ORDERER), log=QUIET) == 0
    assert json.loads((out / "recompute.json").read_text())["status"] == "MATCH"
    tampered = tmp_path / "tampered"
    shutil.copytree(out, tampered)
    r = json.loads((tampered / "reading.json").read_text())
    r["seed0"]["reading"] = "other" if r["seed0"]["reading"] != "other" else "kill"
    (tampered / "reading.json").write_text(json.dumps(r))
    assert v.run_recompute(out_dir=tampered, runs=[str(inputs["A1-s0"]), str(inputs["A2-s0"])],
                           bar_dir=inputs["bar"], orderer_ckpt=str(ORDERER),
                           log=QUIET) == es.CONTRACT_INVALID
    # resume plays nothing; a decode change is refused before any job
    seen.clear()
    assert _validate(inputs, out, run_job_fn=spy) == 0 and seen == []
    orig = v.legacy_table

    def choice_legacy() -> list[dict]:
        t = orig()
        t[0]["decode"] = "choice-v1"
        return t

    import tokenomics.train.order_es_validate as mod
    mod.legacy_table = choice_legacy
    try:
        with pytest.raises(ValueError, match="different pins"):
            _validate(inputs, out, run_job_fn=spy)
    finally:
        mod.legacy_table = orig
    assert seen == []


@needs_inputs
def test_resumed_run_is_read(inputs: dict[str, Path], tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 1: a main run resumed from the pilot (2 + 1) holds A0's monitor rows in its
    own dir, and validation reads it with the fresh run's g_tr, F_tr and improvement (the
    improvement's earlier center from the checkpoint's monitor history)."""
    rows = list(csv.DictReader((inputs["main"] / "monitor.csv").open()))
    assert {(r["member"], r["generation"]) for r in rows} == {("c", "3"), ("A0", "a0")}
    from tokenomics.eval.teacher_audit import load_atlas

    inp = es.load_inputs(inputs["bar"])
    spreads = load_atlas(ATLAS_V2)["spreads"]
    cells = v.monitor_cells(inp.training, spreads)
    bars = {f"{w}#{ep}": b.bar for (w, ep), b in inp.bar.items()}
    monkeypatch.setattr(v, "IMPROVEMENT_WINDOW", 2)  # generation 1 is only in the pilot's rows
    main = v.load_candidate(inputs["main"], inp, cells, bars, str(ORDERER))
    fresh = v.load_candidate(inputs["fresh3"], inp, cells, bars, str(ORDERER))
    assert main.theta_sha256 == fresh.theta_sha256
    assert main.stats.improvement is not None
    for f in ("generation", "g_tr", "f_tr", "improvement"):
        assert getattr(main.stats, f) == pytest.approx(getattr(fresh.stats, f)), f
    # a history that disagrees with this dir's rows is refused
    with pytest.raises(ValueError, match="monitor history at generation 3"):
        v.run_stats("x", "A1", 0, 3, rows, cells, bars, "",
                    [{"generation": 3, "center_units": -1.0}])
    monkeypatch.undo()
    # and the validation read plays it
    out = tmp_path / "val"
    assert _validate(inputs, out, runs=("main",)) == 0
    reading = json.loads((out / "reading.json").read_text())
    assert reading["runs"]["A1-s0"]["generation"] == 3


@needs_inputs
def test_resumed_a0_must_reproduce(inputs: dict[str, Path], tmp_path: Path) -> None:
    """A resumed dir replays A0's monitor rows; a mean other than the checkpoint's is invalid."""
    from test_order_es import _fake

    fake = _fake(inputs["bar"])

    def drift(job: dict) -> dict[str, str]:
        row = fake(job)
        if job["arm"] == "A0" and job["replicate"] == 1:
            row["units"] = repr(float(row["units"]) + 1.0)
        return row

    code = es.run_train(bar_dir=inputs["bar"], out_dir=tmp_path / "m", orderer_ckpt=str(ORDERER),
                        arm="A1", opt_seed=0, generations=3, pairs=2, monitor_every=1,
                        dry_run=True, run_job_fn=drift, log=QUIET,
                        resume_from=str(es.checkpoint_path(inputs["pilot"], 2)))
    assert code == es.CONTRACT_INVALID
    assert "resumed checkpoint" in (tmp_path / "m" / "contract_failures.csv").read_text()


@needs_inputs
def test_contract_failure_is_invalid(inputs: dict[str, Path], tmp_path: Path) -> None:
    def failing(job: dict) -> dict[str, str]:
        if job["arm"] == "TR":
            return job_failure(job, RuntimeError("forced")).to_dict()
        return v.run_validate_job(job)

    out = tmp_path / "bad"
    assert _validate(inputs, out, runs=("A1-s0",), run_job_fn=failing) == es.CONTRACT_INVALID
    reading = json.loads((out / "reading.json").read_text())
    assert reading["status"] == es.INVALID and reading["readings_withheld"]
    assert "final" not in reading
    assert json.loads((out / "spec.json").read_text())["status"] == es.INVALID


@needs_inputs
def test_real_run_refusals(inputs: dict[str, Path], tmp_path: Path) -> None:
    other = tmp_path / "prereg.json"
    other.write_text(v.PREREG_PATH.read_text().replace('"kill_ratio"', '"kill_ratio" ', 1))
    with pytest.raises(ValueError) as exc:
        _validate(inputs, tmp_path / "real", dry_run=False, prereg_path=other, episodes=5)
    msg = str(exc.value)
    for want in ("not a valid validation read", "is not the frozen", "the bar run is a dry run",
                 "horizon 12", "--episodes 5", "source commit", "--worlds is set",
                 "--shock-profile is set", "ended at generation 2",
                 "trained under pre-registration", "is a 'dry-run' run"):
        assert want in msg, want
    # held-out refused even in a dry run; a seed-1 run in a seed-0 read is refused
    from tokenomics.eval.teacher_audit import load_atlas

    held = next(e["record"]["world_id"] for e in load_atlas(ATLAS_V2)["worlds"]
                if e["split"] == HELDOUT)
    with pytest.raises(ValueError, match="held-out split"):
        _validate(inputs, tmp_path / "h", worlds=[held])
    with pytest.raises(ValueError, match="seed-0 runs only"):
        _validate(inputs, tmp_path / "s", runs=("A1-s0", "A1-s1"))
    with pytest.raises(ValueError, match="go together"):
        _validate(inputs, tmp_path / "t", seed_combine=True)


@needs_inputs
def test_seed_combine_reuses_the_seed0_rows(inputs: dict[str, Path], tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """The seed-1 read plays only the seed-1 candidate and its ablation; every other row comes
    from the seed-0 read (equal digests), and the seed-0 reading must recompute."""
    monkeypatch.setattr(v, "first_match", lambda clauses: v.SUCCESS)
    s0 = tmp_path / "val-seed0"
    assert _validate(inputs, s0, runs=("A1-s0", "A2-s0")) == 0
    r0 = json.loads((s0 / "reading.json").read_text())
    assert r0["seed0"]["reading"] == v.SUCCESS and r0["seed0"]["selected"] == "A1-s0"
    seen: list[dict] = []

    def spy(job: dict) -> dict[str, str]:
        seen.append(job)
        return v.run_validate_job(job)

    s1 = tmp_path / "final"
    assert _validate(inputs, s1, runs=("A1-s0", "A1-s1"), seed_combine=True,
                     reuse_references=s0, run_job_fn=spy) == 0
    assert sorted({j["arm"] for j in seen}) == ["A1-s1", "A1-s1/shared"] and len(seen) == 4
    rows = list(csv.DictReader((s1 / "episodes.csv").open()))
    old = list(csv.DictReader((s0 / "episodes.csv").open()))
    assert len(rows) == len(old) + 4
    assert len({(r["demand_digest"], r["shock_digest"]) for r in rows}) == 1
    final = json.loads((s1 / "reading.json").read_text())
    assert final["seed1"]["candidate"] == "A1-s1"
    assert final["final"]["reading"] == v.SUCCESS
    assert json.loads((s1 / "spec.json").read_text())["pins"]["reuse"]["episodes_sha256"] == \
        sha256_file(str(s0 / "episodes.csv"))
    # a seed-0 read made under other pins is refused (review round 1): another package tree,
    # another episode count
    real_fp = v.source_fingerprint
    monkeypatch.setattr(v, "source_fingerprint",
                        lambda commit=None: {**real_fp(commit), "tree_sha256": "f" * 64})
    with pytest.raises(ValueError, match=r"other pins: \['source_tree_sha256'\]"):
        _validate(inputs, tmp_path / "tree", runs=("A1-s0", "A1-s1"), seed_combine=True,
                  reuse_references=s0)
    monkeypatch.setattr(v, "source_fingerprint", real_fp)
    with pytest.raises(ValueError, match=r"other pins: \['episodes'\]"):
        _validate(inputs, tmp_path / "eps", runs=("A1-s0", "A1-s1"), seed_combine=True,
                  reuse_references=s0, episodes=2)
    # a seed-0 read whose recorded reading does not recompute is refused
    monkeypatch.setattr(v, "first_match", lambda clauses: v.PLATEAU)
    with pytest.raises(ValueError, match="recomputes as plateau"):
        _validate(inputs, tmp_path / "again", runs=("A1-s0", "A1-s1"), seed_combine=True,
                  reuse_references=s0)
    # a seed-1 run with another seed-0 candidate's theta is refused
    with pytest.raises(ValueError, match="not the seed-0 read's candidate"):
        _validate(inputs, tmp_path / "x", runs=("A2-s0", "A1-s1"), seed_combine=True,
                  reuse_references=_with_theta(s0, tmp_path / "s0x"))


def _with_theta(s0: Path, dst: Path) -> Path:
    shutil.copytree(s0, dst)
    r = json.loads((dst / "reading.json").read_text())
    r["runs"]["A2-s0"]["theta_sha256"] = "0" * 64
    (dst / "reading.json").write_text(json.dumps(r))
    return dst
