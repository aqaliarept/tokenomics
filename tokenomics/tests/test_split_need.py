"""Task 4.5 (add-m4-orderer-redesign): the EXP-0066 split-need driver `eval/split_need.py`: the
joint count's competition, the per-order counts, the pre-registered reading's threshold edges,
the observer's class sums against the traced `no_fit` loss (and 4.4's numbers) on a fixture, the
EXP-0057 neutrality check (mismatch -> exit 2), real-run refusal and a horizon-12 dry run."""
from __future__ import annotations

import csv
import json
import time
from fractions import Fraction
from pathlib import Path

import pytest

from tokenomics.eval import split_need as sn
from tokenomics.eval.contract import DIGEST
from tokenomics.eval.roster import load_roster
from tokenomics.eval.routing_audit import build_jobs
from tokenomics.eval.teacher_audit import first_confirm_seed, load_atlas

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
EXP0057 = REPO / "experiments/EXP-0057-decoder-check/artifacts/episodes.csv"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
PROBE_OUT = REPO / "openspec/changes/add-m4-orderer-redesign/probes/split_need_out"
SHOCKED = "ws-atlas-v1-12-0060"
LOSSY = "ws-atlas-v1-12-0111"  # 4.4: MWWk r0 loses 436 consumer units to no_fit, 162 split


def _lfs_present(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_inputs = pytest.mark.skipif(
    not all(_lfs_present(p) for p in (ATLAS_V2, EXP0057, ORDERER)),
    reason="EXP-0052 / EXP-0057 records or the EXP-0012 checkpoint absent")


# --- pure classification ---


def _bid(key: str, chosen: str, need: dict[str, int], price: float,
         producers: tuple[str, ...] = ("A", "B", "C"), q: int = 1) -> sn.LostBid:
    per = {j: {t: c // q for t, c in need.items()} for j in producers}
    return sn.LostBid(key=key, chosen=chosen, quantity=q, price=price,
                      tokens=price * sum(need.values()),
                      need={j: dict(need) for j in producers}, per_unit=per)


def test_joint_count_competition_on_a_two_order_cycle() -> None:
    """Two orders lost to `no_fit` at A in one cycle. Each alone fits B's leftover (choice). Under
    the joint count the higher-priced order takes B; the cheaper one no longer fits any single
    producer but A + C together hold it (split). A third, larger, finds the pool spent
    (unfillable)."""
    leftover = {"A": {"t": 3}, "B": {"t": 10}, "C": {"t": 4}}
    first, second = _bid("o1", "A", {"t": 10}, 2.0), _bid("o2", "A", {"t": 6}, 1.0)
    for b in (first, second):
        assert sn.per_order_classes(b, leftover, leftover, {j: [] for j in leftover})[
            sn.PER_ORDER] == sn.CHOICE
    assert sn.joint_classes([second, first], leftover) == {"o1": sn.CHOICE, "o2": sn.SPLIT}
    third = _bid("o3", "A", {"t": 2}, 0.5)
    assert sn.joint_classes([first, second, third], leftover) == {
        "o1": sn.CHOICE, "o2": sn.SPLIT, "o3": sn.UNFILLABLE}
    # the leftovers given in are not consumed (each cycle starts from the observed ones)
    assert leftover == {"A": {"t": 3}, "B": {"t": 10}, "C": {"t": 4}}
    # every recipe type must fit: a second type short at B makes the first order split
    two = {"A": {"t": 3, "u": 5}, "B": {"t": 10, "u": 0}, "C": {"t": 4, "u": 5}}
    assert sn.joint_classes([_bid("o4", "A", {"t": 6, "u": 5}, 1.0)], two) == {"o4": sn.SPLIT}


def test_per_order_counts() -> None:
    """L: another producer's leftover holds it -> choice; only together -> split; else
    unfillable. units: whole sub-orders each fitting (<= the per-type test). P: inserted at j's
    price, j's cheaper accepted bids give way."""
    q2 = _bid("x", "A", {"t": 4}, 1.0, q=2)  # 2 units of 2 t each
    left = {"A": {"t": 1}, "B": {"t": 3}, "C": {"t": 1}}
    acc = {j: [] for j in left}
    got = sn.per_order_classes(q2, left, left, acc)
    # per type: 1 + 3 + 1 = 5 >= 4 -> split; units: 0 + 1 + 0 = 1 < 2 -> unfillable
    assert got == {sn.PER_ORDER: sn.SPLIT, sn.UNITS: sn.UNFILLABLE, sn.PRICE: sn.SPLIT}
    # B accepted a cheaper bid (price 0.5 < 1.0) holding 3 t: inserted at its price, the lost
    # bid displaces it, so B alone holds 4 -> choice under P only
    budget = {"A": {"t": 1}, "B": {"t": 6}, "C": {"t": 1}}
    cheap = {"A": [], "B": [(0.5, {"t": 3})], "C": []}
    assert sn.per_order_classes(q2, left, budget, cheap)[sn.PRICE] == sn.CHOICE
    dear = {"A": [], "B": [(1.0, {"t": 3})], "C": []}  # ties go first: no displacement
    assert sn.per_order_classes(q2, left, budget, dear)[sn.PRICE] == sn.SPLIT
    # the chosen producer's own leftover never makes it choice-fixable
    own = {"A": {"t": 9}, "B": {"t": 0}, "C": {"t": 0}}
    assert sn.per_order_classes(q2, own, own, acc)[sn.PER_ORDER] == sn.SPLIT
    assert sn.per_order_classes(q2, {j: {"t": 0} for j in left}, left, acc)[
        sn.PER_ORDER] == sn.UNFILLABLE


# --- reading ---


def _row(world: str, arm: str, size: int, units: float, split: float, wasted: float = 0.0,
         failed: float = 0.0, rep: int = 0, ep: int = 1, per_order_split: float | None = None
         ) -> dict[str, str]:
    d = {"world": world, "size": str(size), "episode": str(ep), "subject": sn.SUBJECT[arm],
         "replicate": str(rep), "arm": arm, "units": repr(units),
         "trace_wasted_units_consumer": repr(wasted),
         "trace_failed_bid_units_consumer_dropped": repr(failed),
         "trace_failed_bid_units_consumer_ttl": "0.0",
         "trace_failed_bid_units_consumer_ttl_outage": "0.0",
         "trace_lost_units_consumer_no_fit": repr(split)}
    for c in sn.COUNTS:
        for s in sn.STREAMS:
            for k in sn.CLASSES:
                d[sn._cell(c, s, k)] = "0.0"
    d[sn._cell(sn.JOINT, "consumer", sn.SPLIT)] = repr(split)
    d[sn._cell(sn.PER_ORDER, "consumer", sn.SPLIT)] = repr(
        split if per_order_split is None else per_order_split)
    return d


def test_completion_correction() -> None:
    """corrected = joint split x (units + wasted) / (units + wasted + failed)."""
    d = _row("w", sn.MWWK, 12, units=60.0, split=50.0, wasted=20.0, failed=80.0)
    assert sn.completion(d) == Fraction(80, 160)
    assert sn.corrected_split(d) == 25
    assert sn.completion(_row("w", sn.MWWK, 12, units=0.0, split=5.0)) == 0


def test_reading_threshold_edges() -> None:
    """Pooled over world-episodes and replicates; exactly 5% fires, just below does not; either
    size fires; MWWk decides (TS high alone never fires); a size with no deciding row is
    incomplete unless another size fires."""
    # 12: two rows, pooled corrected (3 + 2) / (60 + 40) = 5% exactly (completion 1)
    at = [_row("a", sn.MWWK, 12, 60.0, 3.0), _row("b", sn.MWWK, 12, 40.0, 2.0, rep=1)]
    r = sn.split_reading(at, [12])
    assert r["reading"] == sn.SPLIT_NEEDED and r["sizes_at_or_above"] == [12]
    assert r["per_size"][12]["MWWk"]["share_corrected"] == 0.05
    below = [_row("a", sn.MWWK, 12, 60.0, 3.0), _row("b", sn.MWWK, 12, 40.0, 1.999, rep=1)]
    assert sn.split_reading(below, [12])["reading"] == sn.WHOLE_SUFFICES
    # the completion correction can take it below: 10% raw x completion 0.4 = 4%
    corr = [_row("a", sn.MWWK, 12, 100.0, 10.0, failed=150.0)]
    r = sn.split_reading(corr, [12])
    assert r["reading"] == sn.WHOLE_SUFFICES
    assert r["per_size"][12]["MWWk"]["share_split_uncorrected"]["joint"] == 0.1
    # the correction is per run, then pooled: a starving run (completion 4 / 400) with all the
    # split units next to a full run reads 1 / 100 = 1%; one pooled completion (100 / 496)
    # would read 100 x 100 / 496 / 100 = 20.2% and fire
    starving = [_row("a", sn.MWWK, 12, 4.0, 100.0, failed=396.0),
                _row("b", sn.MWWK, 12, 96.0, 0.0, rep=1)]
    r = sn.split_reading(starving, [12])
    assert r["reading"] == sn.WHOLE_SUFFICES
    assert r["per_size"][12]["MWWk"]["share_corrected"] == 0.01
    assert r["per_size"][12]["MWWk"]["corrected_split_units"] == 1.0
    # at 42 only
    both = below + [_row("c", sn.MWWK, 42, 100.0, 5.0)]
    r = sn.split_reading(both, [12, 42])
    assert r["reading"] == sn.SPLIT_NEEDED and r["sizes_at_or_above"] == [42]
    assert r["next"] == sn.NEXT[sn.SPLIT_NEEDED]
    # TS reported, never deciding; the per-order count never decides
    ts = below + [_row("a", sn.TS, 12, 10.0, 9.0)] + [
        _row("c", sn.MWWK, 42, 100.0, 1.0, per_order_split=90.0)]
    r = sn.split_reading(ts, [12, 42])
    assert r["reading"] == sn.WHOLE_SUFFICES and r["next"] == sn.NEXT[sn.WHOLE_SUFFICES]
    assert r["per_size"][12]["TS"]["share_corrected"] == 0.9
    assert r["per_size"][42]["MWWk"]["share_split_uncorrected"]["L"] == 0.9
    # a size with no MWWk row: incomplete, unless the other size fires
    assert sn.split_reading(below + [_row("c", sn.TS, 42, 1.0, 1.0)], [12, 42])[
        "reading"] == sn.INCOMPLETE
    assert sn.split_reading(at, [12, 42])["reading"] == sn.SPLIT_NEEDED


# --- validity checks (pure) ---


def test_consistency_problems() -> None:
    ok = _row("w", sn.MWWK, 12, 10.0, 4.0)
    for k in ("sn_lost_total_consumer",):
        ok[k] = "4.0"
    for c in sn.COUNTS:  # every count holds the traced 4 units
        ok[sn._cell(c, "consumer", sn.SPLIT)] = "4.0"
    ok.update({"trace_lost_units_producer_no_fit": "0.0", "sn_lost_total_producer": "0.0",
               **{f"sn_{k}": "0.0" for k in sn.CHECK_COUNTERS}})
    assert sn.consistency([ok]) == []
    short = {**ok, sn._cell(sn.PRICE, "consumer", sn.SPLIT): "3.0"}
    ((check, msg),) = sn.consistency([short])
    assert check == sn.CONSISTENCY and "consumer P classes do not sum" in msg
    # the producer stream is checked too: a short class sum, a wrong observer total
    prod = {**ok, "trace_lost_units_producer_no_fit": "2.0", "sn_lost_total_producer": "2.0",
            **{sn._cell(c, "producer", sn.CHOICE): "2.0" for c in sn.COUNTS}}
    assert sn.consistency([prod]) == []
    ((check, msg),) = sn.consistency([{**prod, sn._cell(sn.JOINT, "producer", sn.CHOICE): "1.0"}])
    assert check == sn.CONSISTENCY and "producer joint classes do not sum" in msg
    ((_, msg),) = sn.consistency([{**prod, "sn_lost_total_producer": "3.0"}])
    assert "producer total 3.0 != traced 2.0" in msg
    replay = {**ok, "sn_replay_mismatch": "1.0"}
    assert "replay_mismatch 1.0" in sn.consistency([replay])[0][1]


def test_neutrality_problems() -> None:
    mine = {"world": "w", "episode": "1", "subject": sn.SUBJECT[sn.TS], "replicate": "0",
            "arm": "TS", "units": "10.0", "dead": "false", "demand_digest": "d",
            "shock_digest": "s", "trace_x": "1.0", "trace_series": "[1]", "sn_joint": "3.0"}
    ref = {k: v for k, v in mine.items() if not k.startswith("sn_")}
    rows, problems = sn.neutrality([mine], [ref])
    assert problems == [] and rows[0]["equal"] and rows[0]["compared"] == 5
    rows, problems = sn.neutrality([mine], [{**ref, "trace_x": "2.0", "trace_series": "[2]"}])
    assert [p[0] for p in problems] == [sn.NEUTRALITY] and rows[0]["differing"] == "trace_x"
    _, problems = sn.neutrality([mine], [{**ref, "demand_digest": "e"}])
    assert [p[0] for p in problems] == [DIGEST]
    # a reference cell this run's row lacks is a difference, not a smaller comparison
    rows, problems = sn.neutrality([{k: v for k, v in mine.items() if k != "trace_x"}], [ref])
    assert [p[0] for p in problems] == [sn.NEUTRALITY] and rows[0]["compared"] == 5
    assert rows[0]["differing"] == "trace_x"
    _, problems = sn.neutrality([mine], [{**ref, "replicate": "1"}])
    assert problems == [(sn.NEUTRALITY, f"w episode 1 {sn.SUBJECT[sn.TS]} r0: no EXP-0057 row")]


# --- job path ---


def _job(world: str, arm: str, horizon: int = sn.HORIZON, rep: int = 0) -> dict:
    atlas = load_atlas(ATLAS_V2)
    (view,) = sn.select_views(atlas, [world])
    from tokenomics.eval.provenance import sha256_file

    table = [sn.decoder_arm(arm, load_roster(sn.DEFAULT_ROSTER))]
    pin = {"orderer": {"path": str(ORDERER), "sha256": sha256_file(ORDERER)}}
    seed = first_confirm_seed(view)
    (job,) = [j for j in build_jobs([sn.world_entry(view)], table, 1, horizon, None, pin)
              if j["episode"] == seed and j["replicate"] == rep]
    return {**job, "arm": arm}


@needs_inputs
def test_classes_sum_to_traced_no_fit_and_reproduce_4_4() -> None:
    """Fixture: 12-0111, first confirm episode, MWWk and TS replicate 0, horizon 100. In both
    streams every count's classes sum to the traced `no_fit` loss, the self-checks are 0, the row
    equals EXP-0057's (79 cells) and the classes equal task 4.4's probe output."""
    ref = sn.read_reference(EXP0057)
    for arm in (sn.MWWK, sn.TS):
        row = sn.run_split_job(_job(LOSSY, arm))
        assert "check" not in row, row
        assert sn.consistency([row]) == []
        assert float(row["trace_lost_units_consumer_no_fit"]) > 0
        rows, problems = sn.neutrality([row], ref)
        assert problems == [] and rows[0]["compared"] == 79, (problems, rows)
        probe = PROBE_OUT / f"{LOSSY}__{arm}__r0.json"
        if probe.exists():
            lost = json.loads(probe.read_text())["probe"]["lost_units"]
            for c in sn.COUNTS:
                for s in sn.STREAMS:
                    assert {k: float(row[sn._cell(c, s, k)]) for k in sn.CLASSES} == {
                        k: float(lost[c][s].get(k, 0)) for k in sn.CLASSES}, (arm, c, s)


@needs_inputs
def test_keyed_job_without_key_is_a_failure_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sn, "set_draw_key", lambda env, key: None)
    bad = sn.run_split_job(_job(SHOCKED, sn.MWWK, horizon=12))
    assert bad["error_type"] == "RuntimeError" and "set_draw_key" in bad["message"], bad


# --- driver ---


def test_real_run_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real run needs the registered atlas, orderer and EXP-0057 episodes, a 40-hex commit,
    horizon 100 and no dry-run filter: each one alone is refused, naming it."""
    if not all(_lfs_present(p) for p in (ATLAS_V2, EXP0057)):
        pytest.skip("EXP-0052 / EXP-0057 records absent")
    monkeypatch.setattr(sn, "checked_orderer", lambda out, path, log: {
        "path": path, "sha256": sn.ROLE_CHECKPOINTS["orderer"]})

    def never(job: dict) -> dict:  # a guard that lets a run through fails here, fast
        raise AssertionError(f"a refused run started a job: {job['world']} {job['subject']}")

    kw = dict(atlas_path=ATLAS_V2, out_dir=tmp_path / "o", orderer_ckpt="o.pt",
              source_commit="a" * 40, exp0057_episodes=EXP0057, run_job_fn=never)
    other = tmp_path / "other.csv"
    other.write_text("world\n")
    cases = [({"worlds": [SHOCKED]}, "--worlds is set"),
             ({"max_worlds": 1}, "--max-worlds is set"),
             ({"arms": ["TS"]}, "--arms is set"),
             ({"shock_profile": "shocks-i-smoke"}, "--shock-profile is set"),
             ({"horizon": 12}, "horizon 12 != 100"),
             ({"source_commit": ""}, "no full 40-hex source commit"),
             ({"exp0057_episodes": None}, "--exp0057-episodes None"),
             ({"exp0057_episodes": other}, "is not the registered EXP-0057")]
    for edit, msg in cases:
        with pytest.raises(ValueError, match="not a valid split-need run") as err:
            sn.run_split_need(**{**kw, **edit})
        assert msg in str(err.value), (edit, str(err.value))
        inner = str(err.value).split("(", 1)[1].rsplit("); pass", 1)[0]
        assert ";" not in inner, str(err.value)
    monkeypatch.setattr(sn, "checked_orderer", lambda out, path, log: {
        "path": path, "sha256": "f" * 64})
    with pytest.raises(ValueError, match="orderer checkpoint ffffffffffff is not the registered"):
        sn.run_split_need(**kw)
    assert not (tmp_path / "o" / "episodes.csv").exists()


@needs_inputs
def test_dry_run_neutrality_and_mismatch_exit_2(tmp_path: Path) -> None:
    """1 world, horizon 12, shocks-i-smoke, --workers 1: every output, 8 rows (2 episodes x TS x
    2, MWWk x 2), a resume runs nothing; a second run checked against the first's episodes.csv is
    neutral; against a tampered copy it is invalid (exit 2), the mismatch recorded and the reading
    withheld."""
    out = tmp_path / "EXP-dry"
    args = ["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run",
            "--worlds", SHOCKED, "--horizon", "12", "--shock-profile", "shocks-i-smoke",
            "--workers", "1"]
    started = time.perf_counter()
    assert sn.main([*args, "--out-dir", str(out)]) == 0
    assert time.perf_counter() - started < 120
    for f in ("spec.json", "episodes.csv", "split_need.csv", "split_need_worlds.csv",
              "neutrality.csv", "reading.json", "contract_failures.csv", "RESULT.md"):
        assert (out / f).exists(), f
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    spec = json.loads((out / "spec.json").read_text())
    assert spec["kind"] == "dry-run" and spec["status"] == "complete"
    assert spec["pre_registered"]["deciding_arm"] == "MWWk"
    episodes = list(csv.DictReader((out / "episodes.csv").open()))
    assert len(episodes) == 8
    assert sorted({(e["arm"], e["replicate"]) for e in episodes}) == [
        ("MWWk", "0"), ("MWWk", "1"), ("TS", "0"), ("TS", "1")]
    assert all(e["sn_joint_consumer_split_units"] != "" for e in episodes)
    reading = json.loads((out / "reading.json").read_text())
    assert reading["status"] == "valid" and reading["dry_run"] is True
    assert reading["reading"] in (sn.SPLIT_NEEDED, sn.WHOLE_SUFFICES)
    assert reading["neutrality_checked"] is False and reading["deciding_arm"] == "MWWk"
    assert list(reading["per_size"]) == ["12"]
    worlds = list(csv.DictReader((out / "split_need_worlds.csv").open()))
    assert sorted(w["arm"] for w in worlds) == ["MWWk", "TS"]
    text = (out / "RESULT.md").read_text()
    assert "not an acceptance read" in text and "Deciding arm MWWk" in text

    logs: list[str] = []
    assert sn.run_split_need(
        atlas_path=ATLAS_V2, out_dir=out, orderer_ckpt=str(ORDERER), worlds=[SHOCKED],
        horizon=12, shock_profile="shocks-i-smoke", dry_run=True, log=logs.append) == 0
    assert any("8 already done, 0 to run" in m for m in logs)

    ref = out / "episodes.csv"
    again = tmp_path / "EXP-again"
    assert sn.main([*args, "--out-dir", str(again), "--exp0057-episodes", str(ref)]) == 0
    neutral = list(csv.DictReader((again / "neutrality.csv").open()))
    assert len(neutral) == 8 and all(r["equal"] == "true" for r in neutral)
    assert json.loads((again / "reading.json").read_text())["neutral_rows"] == 8

    bad = tmp_path / "bad.csv"
    rows = [{**r, "trace_lost_units_consumer_no_fit": "999.0"}
            if (r["arm"], r["replicate"]) == ("MWWk", "1") else r
            for r in csv.DictReader(ref.open())]
    with bad.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    canned = {(r["world"], int(r["episode"]), r["subject"], int(r["replicate"])): r
              for r in csv.DictReader(ref.open())}
    worse = tmp_path / "EXP-bad"
    assert sn.run_split_need(
        atlas_path=ATLAS_V2, out_dir=worse, orderer_ckpt=str(ORDERER), exp0057_episodes=bad,
        worlds=[SHOCKED], horizon=12, shock_profile="shocks-i-smoke", dry_run=True,
        run_job_fn=lambda j: dict(canned[(j["world"], j["episode"], j["subject"],
                                          j["replicate"])]),
        log=lambda m: None) == sn.CONTRACT_INVALID
    partial = tmp_path / "partial.csv"
    with partial.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows[1:])
    with pytest.raises(ValueError, match="lacks 1 of the run's 8 rows"):
        sn.run_split_need(atlas_path=ATLAS_V2, out_dir=tmp_path / "EXP-partial",
                          orderer_ckpt=str(ORDERER), exp0057_episodes=partial, worlds=[SHOCKED],
                          horizon=12, shock_profile="shocks-i-smoke", dry_run=True,
                          run_job_fn=lambda j: {}, log=lambda m: None)
    assert not (tmp_path / "EXP-partial" / "spec.json").exists()
    fails = list(csv.DictReader((worse / "contract_failures.csv").open()))
    assert len(fails) == 2 and {f["check"] for f in fails} == {sn.NEUTRALITY}
    assert all("trace_lost_units_consumer_no_fit" in f["message"] for f in fails)
    reading = json.loads((worse / "reading.json").read_text())
    assert reading["readings_withheld"] is True and reading["reading"] is None
    assert json.loads((worse / "spec.json").read_text())["status"] == "invalid"
