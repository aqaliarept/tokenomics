"""M5 task 2.4: the producer reads. The L3 pilot read (g_tr, F_tr from the monitor rows); the L5
reading table per branch and the selection; dry validate / held-out reads and their recompute; the
held-out ledger (pending, close, refusals); pre-registration v2 pinned, v1 untouched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenomics.train import producer_es as pe
from tokenomics.train import producer_read as rd

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
LOG = lambda s: None  # noqa: E731


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not ((TRAINING / "training_worlds.json").exists() and _lfs(ATLAS_V2)),
                           reason="EXP-0058 training set / EXP-0052 atlas absent")


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if not ((TRAINING / "training_worlds.json").exists() and _lfs(ATLAS_V2)):
        pytest.skip("records absent")
    root = tmp_path_factory.mktemp("prd")
    kw = dict(training_dir=TRAINING, horizon=12, dry_run=True, log=LOG)
    out = {k: root / k for k in ("bar", "noise", "s0", "s1", "val", "held")}
    assert pe.run_bar(out_dir=out["bar"], max_world_episodes=40, **kw) == 0
    assert pe.run_noise_floor(bar_dir=out["bar"], out_dir=out["noise"], replicates=(1,), **kw) == 0
    for seed in (0, 1):
        assert pe.run_train(bar_dir=out["bar"], margins=out["noise"] / "margins.json",
                            out_dir=out[f"s{seed}"], generations=1, opt_seed=seed, size=3,
                            pairs=2, monitor_every=1, **kw) == 0
    return out


def _views(split: str, size: int) -> str:
    from tokenomics.eval.teacher_audit import load_atlas

    vs = rd.read_views(load_atlas(ATLAS_V2), split, None, None)
    return next(v["world"] for v in vs if v["size"] == size and v["rho"] > rd.RHO_LOADED
                and v["class"] != "collapse")


@needs
def test_pilot_reads_g_tr_and_f_tr(chain: dict[str, Path]) -> None:
    doc = rd.run_pilot(run_dir=chain["s0"], training_dir=TRAINING, atlas_path=ATLAS_V2,
                       generation=1, log=LOG)
    assert doc["reading"] in ("feasible", "unresolved") and doc["world_episodes"] >= 1
    assert doc["g_tr"] == pytest.approx(doc["monitor_csv_g_tr"])  # same as the driver's monitor
    assert doc["F_tr"] > 0
    g0 = rd.run_pilot(run_dir=chain["s0"], training_dir=TRAINING, atlas_path=ATLAS_V2,
                      generation=0, log=LOG)
    assert g0["g_tr"] == 0.0  # theta 0 equals rf3 at both keys


@needs
def test_validate_and_heldout_dry_reads_and_recompute(chain: dict[str, Path]) -> None:
    w12 = _views(rd.VALIDATE, 12)  # 42 producers: covered by the reading-table tests
    kw = dict(atlas_path=ATLAS_V2, bar_dir=chain["bar"], horizon=12, episodes=2, dry_run=True,
              log=LOG)
    assert rd.run_read(mode=rd.VALIDATE, out_dir=chain["val"], runs=[str(chain["s0"]),
                       str(chain["s1"])], worlds=[w12], **kw) == 0
    reading = json.loads((chain["val"] / "reading.json").read_text())
    assert reading["reading"] in ("success", "plateau", "kill", "transfer_failure")
    assert reading["selected"] in rd.SEEDS and "stock_heavy" in reading
    assert set(reading["per_size"]) == {"12"}
    assert rd.run_recompute(out_dir=chain["val"], log=LOG) == 0  # MATCH
    rows = pe.read_csv(chain["val"] / "episodes.csv")
    assert {r["arm"] for r in rows} == {*rd.SEEDS, *rd.REFS}
    assert all(r["st_lost_consumer_total"] != "" for r in rows)  # observer on
    h12 = _views(rd.HELDOUT, 12)
    assert rd.run_read(mode=rd.HELDOUT, out_dir=chain["held"], validation_dir=chain["val"],
                       worlds=[h12], **kw) == 0
    held = json.loads((chain["held"] / "reading.json").read_text())
    assert held["reading"] in ("confirmed", "not_confirmed")
    assert rd.run_recompute(out_dir=chain["held"], log=LOG) == 0


@needs
def test_validation_split_only_and_real_reads_refused(chain: dict[str, Path], tmp_path: Path
                                                      ) -> None:
    from tokenomics.eval.teacher_audit import load_atlas

    with pytest.raises(ValueError, match="outside the"):
        rd.read_views(load_atlas(ATLAS_V2), rd.VALIDATE, [_views(rd.HELDOUT, 12)], None)
    with pytest.raises(ValueError, match="not a complete real run|generation"):
        rd.run_read(mode=rd.VALIDATE, out_dir=tmp_path / "v", atlas_path=ATLAS_V2,
                    bar_dir=chain["bar"], runs=[str(chain["s0"]), str(chain["s1"])],
                    source_commit="0" * 40, log=LOG)


def _row(arm: str, world: str, ep: int, r: int, units: float, dead: bool = False,
         hold: float = 10.0) -> dict[str, str]:
    return {"arm": arm, "world": world, "episode": str(ep), "replicate": str(r),
            "units": repr(units), "dead": "true" if dead else "false", "r_start": "1.0",
            "r_end": "0.9", "consumer_fill_rate": json.dumps({"c0": 0.9}),
            "holding_units_per_cycle": repr(hold), "demand_digest": "d", "shock_digest": "s"}


def _table(d12: dict[str, float], d42: dict[str, float] | None = None, dead: str = "",
           hold: float = 10.0):
    """Synthetic rows: 4 worlds at 12 (and 42), 4 episodes x 2 replicates, each arm = rf3 + diff
    with a small paired spread; returns (rows, cells)."""
    from tokenomics.eval.teacher_audit import load_atlas
    from tokenomics.train.order_es_validate import Cell, spread_from_dict

    spreads = load_atlas(ATLAS_V2)["spreads"]
    rows, cells = [], {}
    for size, diffs in ((12, d12), (42, d42)):
        if diffs is None:
            continue
        for wi in range(4):
            w = f"w{size}-{wi}"
            cells[w] = Cell(w, size, "dense", False, spread_from_dict(spreads[f"{size}/dense"]),
                            0.5, "hard")
            for ep in range(4):
                for r in (0, 1):
                    base = 1000.0 + 10 * wi + ep
                    rows.append(_row(rd.BASE, w, ep, r, base))
                    for arm in (*rd.SEEDS, "k4", "ep"):
                        jitter = (ep - 1.5) * 0.5 + r * 0.25
                        rows.append(_row(arm, w, ep, r, base + diffs.get(arm, 0.0) + jitter,
                                         dead=(arm == dead and size == 12 and ep == 0),
                                         hold=hold if arm == "cand-s0" else 10.0))
    return rows, cells


@needs
@pytest.mark.parametrize("d12,d42,dead,want", [
    ({"cand-s0": 500, "cand-s1": 400}, {"cand-s0": 900, "cand-s1": 800}, "", "success"),
    ({"cand-s0": 500, "cand-s1": 0}, None, "", "plateau"),
    ({"cand-s0": 500, "cand-s1": -500}, None, "", "kill"),
    ({"cand-s0": 500, "cand-s1": 400}, None, "cand-s1", "kill"),
    ({"cand-s0": 500, "cand-s1": 400}, {"cand-s0": -3000, "cand-s1": 800}, "",
     "transfer_failure"),
])
def test_validate_reading_table(d12, d42, dead, want) -> None:
    rows, cells = _table(d12, d42, dead)
    assert rd.validate_reading(rows, cells)["reading"] == want


@needs
def test_selection_and_stock_heavy() -> None:
    rows, cells = _table({"cand-s0": 400, "cand-s1": 500}, None)
    assert rd.validate_reading(rows, cells)["selected"] == "cand-s1"
    rows, cells = _table({"cand-s0": 450, "cand-s1": 450}, None)
    assert rd.validate_reading(rows, cells)["selected"] == "cand-s0"  # tie -> seed 0
    rows, cells = _table({"cand-s0": 500, "cand-s1": 400}, None, hold=16.0)
    sh = rd.validate_reading(rows, cells)["stock_heavy"]
    assert sh["flag"] and sh["ratio"] == pytest.approx(1.6)


def test_ledger_pending_close_and_refusals(tmp_path: Path) -> None:
    led = tmp_path / "ledger.json"
    head = lambda: "c0" * 20  # noqa: E731
    assert rd.ledger_problems(None, "t")[0].startswith("a real held-out read needs --ledger")
    assert "not exactly one" in rd.ledger_problems(led, "t")[0]
    e = rd.ledger_pending(led, "theta-a", "atlas", head=head)
    assert e["status"] == "pending" and e["c0"] == "c0" * 20
    assert rd.ledger_problems(led, "theta-a") == []
    assert "another theta" in rd.ledger_problems(led, "theta-b")[0]
    with pytest.raises(ValueError, match="already holds"):
        rd.ledger_pending(led, "theta-a", "atlas", head=head)
    doc = json.loads(led.read_text())
    doc["entries"][0]["package_source_sha256"] = "x"
    led.write_text(json.dumps(doc))
    assert "package tree differs" in rd.ledger_problems(led, "theta-a")[0]
    doc["entries"][0]["package_source_sha256"] = e["package_source_sha256"]
    led.write_text(json.dumps(doc))
    closed = rd.ledger_close(led, "done", "EXP-0084", head=lambda: "c1" * 20)
    assert closed["status"] == "done" and closed["c1"] == "c1" * 20
    assert "never re-run" in rd.ledger_problems(led, "theta-a")[0]
    with pytest.raises(ValueError, match="final"):
        rd.ledger_close(led, "invalid", "EXP-0084")


def test_prereg_v2_untouched_and_v1_untouched() -> None:
    from tokenomics.eval.provenance import sha256_file

    assert sha256_file(str(pe.PREREG_V1_PATH)) == pe.PREREG_V1_SHA256
    assert sha256_file(str(pe.PREREG_V2_PATH)) == pe.PREREG_V2_SHA256
    assert json.loads(pe.PREREG_V2_PATH.read_text()) == pe.registered_prereg(2)
    assert json.loads(pe.PREREG_V1_PATH.read_text()) == pe.registered_prereg(1)
    doc = json.loads(pe.PREREG_V2_PATH.read_text())
    assert doc["version"] == 2 and doc["changes"]["from"]["sha256"] == pe.PREREG_V1_SHA256
    assert "seed 0 on a tie" in doc["readings"]["L6"]["selection"]
    v1 = json.loads(pe.PREREG_V1_PATH.read_text())
    assert {k: v for k, v in doc.items() if k not in ("version", "changes", "readings")} == {
        k: v for k, v in v1.items() if k != "readings"}


@pytest.mark.parametrize("status,prereg,dry,reading,msg", [
    ("invalid", None, False, "success", "not complete"),
    ("complete", "0" * 64, False, "success", "another pre-registration"),
    ("complete", None, True, "success", "real L5 `success`"),
    ("complete", None, False, "plateau", "real L5 `success`"),
])
def test_heldout_needs_a_complete_real_l5_success(tmp_path: Path, status, prereg, dry, reading,
                                                  msg) -> None:
    """Review m5-2-4: a held-out read refuses an L5 read that ended invalid (a reading from partial
    rows), ran under another pre-registration, was a dry run or did not read success."""
    val = tmp_path / "val"
    val.mkdir()
    (val / "reading.json").write_text(json.dumps({"reading": reading, "selected": "cand-s0"}))
    (val / "spec.json").write_text(json.dumps({"status": status, "pins": {
        "dry_run": dry, "prereg": prereg or pe.PREREG_SHA256, "runs": ["s0", "s1"]}}))
    if not _lfs(ATLAS_V2):
        pytest.skip("atlas absent")
    with pytest.raises(ValueError, match=msg):
        rd.run_read(mode=rd.HELDOUT, out_dir=tmp_path / "h", atlas_path=ATLAS_V2,
                    bar_dir=tmp_path, validation_dir=val, source_commit="0" * 40, log=LOG)


@needs
def test_heldout_reading_as_registered() -> None:
    """v2 readings.L6.confirmed: clears at 12, not below at 42, no extra deaths, run-down /
    starvation shares <= rf3's at 12."""
    rows, cells = _table({"cand-s0": 500}, {"cand-s0": 0})
    held = [{**r, "arm": "cand"} if r["arm"] == "cand-s0" else r for r in rows]
    assert rd.heldout_reading(held, cells)["reading"] == "confirmed"
    worse = [{**r, "r_end": "0.5"} if r["arm"] == "cand" and r["world"].startswith("w12")
             and r["episode"] == "0" else r for r in held]
    assert rd.heldout_reading(worse, cells)["reading"] == "not_confirmed"  # run-down share up
    rows, cells = _table({"cand-s0": 500}, {"cand-s0": -3000})
    held = [{**r, "arm": "cand"} if r["arm"] == "cand-s0" else r for r in rows]
    assert rd.heldout_reading(held, cells)["reading"] == "not_confirmed"  # below at 42
    doc = json.loads(pe.PREREG_V2_PATH.read_text())
    assert {"readings.L6.confirmed", "readings.L5.arms"} <= set(doc["changes"]["changed"])
    assert "<= rf3's (L5's clause)" in doc["readings"]["L6"]["confirmed"]


def test_prereg_v3_untouched_changes_only_the_guard() -> None:
    """Task 4.2b: v3 differs from v2 in the guard (and version / changes); never edited."""
    from tokenomics.eval.provenance import sha256_file

    assert sha256_file(str(pe.PREREG_V3_PATH)) == pe.PREREG_V3_SHA256
    assert json.loads(pe.PREREG_V3_PATH.read_text()) == pe.registered_prereg(3)
    v3, v2 = json.loads(pe.PREREG_V3_PATH.read_text()), json.loads(pe.PREREG_V2_PATH.read_text())
    assert v3["version"] == 3 and v3["changes"]["from"]["sha256"] == pe.PREREG_V2_SHA256
    assert sorted(k for k in set(v3) | set(v2) if v3.get(k) != v2.get(k)) == [
        "changes", "guard", "version"]
    assert v3["guard"]["threshold"] == pe.FLAG_THRESHOLD
    assert v3["guard"]["calibration"]["sources"]["noise_episodes_sha256"] == pe.FLAG_SOURCES[
        "noise_episodes_sha256"]


def test_prereg_v4_untouched_changes_only_the_death() -> None:
    """Task 4.2c: vs v3 only the guard's rule / death text and the fitness's death penalty change;
    never edited."""
    from tokenomics.eval.provenance import sha256_file

    assert sha256_file(str(pe.PREREG_V4_PATH)) == pe.PREREG_V4_SHA256
    assert json.loads(pe.PREREG_V4_PATH.read_text()) == pe.registered_prereg(4)
    v4, v3 = json.loads(pe.PREREG_V4_PATH.read_text()), json.loads(pe.PREREG_V3_PATH.read_text())
    assert v4["version"] == 4 and v4["changes"]["from"]["sha256"] == pe.PREREG_V3_SHA256
    assert sorted(k for k in set(v4) | set(v3) if v4.get(k) != v3.get(k)) == [
        "changes", "fitness", "guard", "version"]
    assert {k for k in v4["guard"] if v4["guard"][k] != v3["guard"].get(k)} == {"rule", "death"}
    assert {k for k in v4["fitness"] if v4["fitness"][k] != v3["fitness"].get(k)} == {"death"}
    assert v4["readings"] == v3["readings"]


def test_prereg_v5_untouched_on_rf4() -> None:
    """Task 4.2h / 4.2i: v5 = v4's rules on rf4 with EXP-0085's calibration; never edited."""
    from tokenomics.eval.provenance import sha256_file

    assert sha256_file(str(pe.PREREG_V5_PATH)) == pe.PREREG_V5_SHA256
    v5 = json.loads(pe.PREREG_V5_PATH.read_text())
    assert v5 == pe.registered_prereg(5)
    assert v5["base"]["producer"] == pe.RF4_PRODUCER and v5["version"] == 5
    assert v5["changes"]["from"]["sha256"] == pe.PREREG_V4_SHA256
    assert "rf3" not in json.dumps({k: v5["readings"][k] for k in ("L3", "L5", "L6")})


def test_prereg_v6_is_v5_rebased_on_k5_rf5_and_frozen_only_after_l1(monkeypatch) -> None:
    """Producer finalization: v6 = v5's rules with the final producer k5_rf5 as the base (buffer
    k = 5) and its own L1 calibration; before that run is calibrated, v6 cannot be built and no
    real run is licensed."""
    assert pe.PREREG_PATH.name == "es_prereg_m5_v6.json"
    assert pe.BASE_PRODUCER == pe.K5_RF5_PRODUCER and rd.BASE == "k5_rf5"
    if pe.FLAG_THRESHOLD_V6 is None:
        assert pe.PREREG_SHA256 is None and "no pinned" in pe.prereg_problems()[0]
        with pytest.raises(ValueError, match="needs the k5_rf5 L1 run"):
            pe.registered_prereg(6)
        monkeypatch.setattr(pe, "FLAG_THRESHOLD_V6", 1.0)
        monkeypatch.setattr(pe, "FLAG_SOURCES_V6", {"noise_episodes_sha256": "x"})
        monkeypatch.setattr(pe, "FLAG_CALIBRATION_V6", {"rate": 0.0})
    v6, v5 = pe.registered_prereg(6), json.loads(pe.PREREG_V5_PATH.read_text())
    assert v6["base"]["producer"] == pe.K5_RF5_PRODUCER and v6["version"] == 6
    assert v6["changes"]["from"]["sha256"] == pe.PREREG_V5_SHA256
    assert v6["guard"]["threshold"] == pe.FLAG_THRESHOLD_V6
    assert v6["policy"]["buffer"].startswith("k_g = 5 exp(") and v5["policy"]["buffer"].startswith(
        "k_g = 4 exp(")
    assert v6["es"] == v5["es"] and v6["fitness"]["death"]["rule"] == v5["fitness"]["death"]["rule"]
    text = json.dumps({k: v for k, v in v6.items() if k != "changes"})
    assert "k5_rf5's units" in text and "candidate - k5_rf5" in text
    assert "rf4" not in json.dumps({k: v6["readings"][k] for k in ("L3", "L5", "L6")})
    assert "k4_k5_rf5" not in text  # the full producer name was replaced first