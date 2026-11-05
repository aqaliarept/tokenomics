"""M5 task 2.2: the producer ES driver. Horizon-12 dry runs of bar -> noise-floor -> probe -> train,
resume (and its refusal under other pins), CRN (theta = 0 equals the bar's base rows), real-run
refusals, and the EXP-0079 cross-check through the supply-trace job path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tokenomics.train import producer_es as pe
from tokenomics.train.producer_residual import THETA_DIM

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "experiments/EXP-0058-es-gradient-probe/bar"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
EXP0079 = (REPO / "experiments/EXP-0079-expand-early/entity_planner_buyahead_k4_rf3/artifacts"
           / "episodes.csv")


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_training = pytest.mark.skipif(not (TRAINING / "training_worlds.json").exists(),
                                    reason="EXP-0058 training set absent")
LOG = lambda s: None  # noqa: E731


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if not (TRAINING / "training_worlds.json").exists():
        pytest.skip("EXP-0058 training set absent")
    root = tmp_path_factory.mktemp("pes")
    out = {"bar": root / "bar", "noise": root / "noise", "probe": root / "probe",
           "train": root / "train"}
    kw = dict(training_dir=TRAINING, horizon=12, dry_run=True, log=LOG)
    assert pe.run_bar(out_dir=out["bar"], max_world_episodes=12, **kw) == 0
    assert pe.run_noise_floor(bar_dir=out["bar"], out_dir=out["noise"], replicates=(1, 2),
                              **kw) == 0
    margins = out["noise"] / "margins.json"
    assert pe.run_probe(bar_dir=out["bar"], margins=margins, out_dir=out["probe"], size=3,
                        pairs=2, **kw) == 0
    assert pe.run_train(bar_dir=out["bar"], margins=margins, out_dir=out["train"],
                        generations=2, size=3, pairs=2, monitor_every=2, **kw) == 0
    return out


@needs_training
def test_bar_and_noise_floor(runs: dict[str, Path]) -> None:
    import csv

    bar = list(csv.DictReader((runs["bar"] / "bar.csv").open()))
    assert len(bar) == 12 and all(float(r["bar"]) > 0 for r in bar)
    repro = list(csv.DictReader((runs["noise"] / "reproduction.csv").open()))
    assert len(repro) == 12 and all(r["equal"] == "True" for r in repro)  # r0 == the bar
    m = json.loads((runs["noise"] / "margins.json").read_text())
    assert m["format"] == "m5-noise-margin-v1" and m["rundown"] > 0 and m["episodes"] == 24
    for d in ("bar", "noise", "probe", "train"):
        assert (runs[d] / "contract_failures.csv").read_text().count("\n") == 1, d
        assert json.loads((runs[d] / "spec.json").read_text())["status"] == "complete", d


@needs_training
def test_probe_reads_both_sigma_arms_and_the_step(runs: dict[str, Path]) -> None:
    doc = json.loads((runs["probe"] / "probe.json").read_text())
    assert set(doc["arms"]) == {"S", "S2"}
    assert doc["arms"]["S"]["sigma"] == pytest.approx(pe.SIGMA)
    assert doc["arms"]["S2"]["sigma"] == pytest.approx(2 * pe.SIGMA)
    for arm in doc["arms"].values():
        assert arm["cosine"] is not None and -1 <= arm["cosine"] <= 1
        assert arm["reading"] in ("signal", "weak", "no_signal") and len(arm["batches"]) == 2
    assert doc["reading"] == doc["arms"]["S"]["reading"]  # the pilot's licence is arm S's
    assert doc["step"]["step_reading"] in ("heterogeneity", "flat_at_sigma", "sigma_only",
                                           "no_signal")


@needs_training
def test_train_resumes_and_refuses_other_pins(runs: dict[str, Path]) -> None:
    import csv

    t = runs["train"]
    kw = dict(training_dir=TRAINING, bar_dir=runs["bar"], margins=runs["noise"] / "margins.json",
              out_dir=t, horizon=12, dry_run=True, size=3, pairs=2, monitor_every=2, log=LOG)
    g2 = pe.load_checkpoint(pe.checkpoint_path(t, 2))
    assert pe.run_train(generations=3, **kw) == 0  # resume: one more generation
    gens = list(csv.DictReader((t / "generations.csv").open()))
    assert [g["generation"] for g in gens] == ["1", "2", "3"]
    assert gens[1]["theta_sha256"] == g2["theta_sha256"]
    assert pe.load_checkpoint(pe.checkpoint_path(t, 3))["generation"] == 3
    with pytest.raises(ValueError, match="different pins"):
        pe.run_train(generations=3, **{**kw, "size": 2})
    with pytest.raises(ValueError, match="different pins"):
        pe.run_train(generations=3, sigma=0.2, **kw)


@needs_training
def test_theta_zero_member_equals_the_bar_base(runs: dict[str, Path]) -> None:
    import csv

    ts = pe.load_training(TRAINING)
    base = {(d["world"], d["episode"]): d for d in csv.DictReader(
        (runs["bar"] / "base_rows.csv").open())}
    w, e = sorted(base)[0]
    job = pe.make_job(ts, kind=pe.TRAIN, producing=pe.RESIDUAL, theta=np.zeros(THETA_DIM),
                      generation=1, member="+00", world=w, episode=int(e), replicate=0,
                      horizon=12)
    row = pe.run_producer_job(job)
    for c in ("units", "dead", "r_end", "demand_digest", "shock_digest", "fulfilled_units",
              "holding_units_per_cycle", "consumer_fill_rate"):
        assert row[c] == base[(w, e)][c], c


@needs_training
def test_real_runs_are_refused_without_pins(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a valid bar run"):
        pe.run_bar(training_dir=TRAINING, out_dir=tmp_path / "b", source_commit="0" * 40,
                   log=LOG)  # no cross-check inputs
    with pytest.raises(ValueError, match="not a valid bar run"):
        pe.run_bar(training_dir=TRAINING, out_dir=tmp_path / "b2", max_world_episodes=3,
                   log=LOG)


@needs_training
def test_minibatches_follow_the_recorded_stream() -> None:
    ts = pe.load_training(TRAINING)
    size = int(ts.schedule["minibatch_size"])
    for g in (1, 2):
        assert pe.minibatch(ts, 0, g, size) == pe._fmt_we(ts.schedule["minibatches"]["0"][g - 1])
    assert len(pe.minibatch(ts, 0, 1, pe.MINIBATCH)) == pe.MINIBATCH
    assert len(pe.monitor_set(ts)) == 16 and len(ts.world_episodes) == 128


EXP0084 = (REPO / "experiments/EXP-0084-sustainable-upgrade/entity_planner_buyahead_k4_rf4"
           / "artifacts" / "episodes.csv")


@pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(EXP0084)), reason="EXP-0052 / EXP-0084 absent")
def test_crosscheck_reproduces_the_base_reference_exp0084(tmp_path: Path) -> None:
    """The bar's cross-check replays the base (rf4 from v5) and equals EXP-0084's rf4 rows."""
    assert pe.BASE_PRODUCER == pe.RF4_PRODUCER
    res = pe.crosscheck_reference(ATLAS_V2, EXP0084, tmp_path, LOG)
    assert res["rows"] == pe.CROSSCHECK_N and res["equal"], res


def _train_kw(runs: dict[str, Path], out: Path) -> dict:
    return dict(training_dir=TRAINING, bar_dir=runs["bar"], margins=runs["noise"] / "margins.json",
                out_dir=out, horizon=12, dry_run=True, size=3, pairs=2, monitor_every=2, log=LOG)


@needs_training
def test_a_monitor_lost_in_a_crash_runs_on_resume(runs: dict[str, Path], tmp_path: Path) -> None:
    """Review m5-2-2: the run dies inside generation 2's monitor (after its checkpoint); the resume
    runs that monitor before going on, and the run cannot end complete without it."""
    import csv

    out = tmp_path / "t"

    def dies_in_monitor(job):
        if job["kind"] == "monitor" and job["generation"] == "2":
            raise KeyboardInterrupt("pod lost")
        return pe.run_producer_job(job)

    with pytest.raises(KeyboardInterrupt):
        pe.run_train(generations=2, run_job_fn=dies_in_monitor, **_train_kw(runs, out))
    assert pe.checkpoint_path(out, 2).exists()
    assert [m["generation"] for m in csv.DictReader((out / "monitor.csv").open())] == ["0"]
    assert pe.run_train(generations=3, **_train_kw(runs, out)) == 0
    mon = list(csv.DictReader((out / "monitor.csv").open()))
    assert [m["generation"] for m in mon] == ["0", "2"]
    assert mon[1]["theta_sha256"] == pe.load_checkpoint(pe.checkpoint_path(out, 2))["theta_sha256"]


@needs_training
def test_a_generation_row_without_its_checkpoint_is_not_duplicated(
        runs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import csv

    out = tmp_path / "t"
    real = pe.save_checkpoint

    def dies_at_2(path, doc):
        if doc["generation"] == 2:
            raise KeyboardInterrupt("pod lost")
        real(path, doc)

    monkeypatch.setattr(pe, "save_checkpoint", dies_at_2)
    with pytest.raises(KeyboardInterrupt):
        pe.run_train(generations=2, **_train_kw(runs, out))
    monkeypatch.setattr(pe, "save_checkpoint", real)
    assert [g["generation"] for g in csv.DictReader((out / "generations.csv").open())] == ["1", "2"]
    assert pe.run_train(generations=2, **_train_kw(runs, out)) == 0
    assert [g["generation"] for g in csv.DictReader((out / "generations.csv").open())] == ["1", "2"]


@needs_training
def test_real_probe_and_train_need_the_pinned_preregistration(
        runs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kw = dict(training_dir=TRAINING, bar_dir=runs["bar"], margins=runs["noise"] / "margins.json",
              source_commit="0" * 40, log=LOG)
    if pe.PREREG_SHA256 is None:  # v5 before the rf4 L1 run: every real probe / train is refused
        with pytest.raises(ValueError, match="no pinned pre-registration"):
            pe.run_probe(out_dir=tmp_path / "p0", **kw)
    else:
        assert pe.prereg_problems() == []  # the frozen file, as pinned
    monkeypatch.setattr(pe, "PREREG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="does not match the pinned SHA-256"):
        pe.run_probe(out_dir=tmp_path / "p", **kw)
    with pytest.raises(ValueError, match="does not match the pinned SHA-256"):
        pe.run_train(out_dir=tmp_path / "t", generations=20, **kw)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="--monitor-every is set"):
        pe.run_train(out_dir=tmp_path / "t2", generations=20, monitor_every=7, **kw)


def test_freeze_matches_and_a_changed_section_is_refused(tmp_path: Path) -> None:
    """Task 2.3: the frozen file is the registration (MATCH); a copy with one section changed is
    refused, both by its SHA-256 and by the section comparison."""
    status, sha = pe.freeze_prereg(pe.PREREG_V4_PATH, 4)  # the last frozen file whatever is pinned
    assert status == "MATCH" and sha == pe.PREREG_V4_SHA256
    doc = json.loads(pe.PREREG_V4_PATH.read_text())
    doc["es"]["sigma"] = 0.07
    bad = tmp_path / "prereg.json"
    bad.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    assert pe.freeze_prereg(bad, 4)[0] == "MISMATCH es"
    from tokenomics.eval.provenance import sha256_file

    assert "does not match the pinned" in pe.prereg_problems(bad, pinned=pe.PREREG_V4_SHA256)[0]
    assert "sections differ from the code: ['es']" in pe.prereg_problems(
        bad, pinned=sha256_file(str(bad)), version=4)[0]
    fresh = tmp_path / "new.json"
    assert pe.freeze_prereg(fresh, 4) == ("FROZEN", pe.PREREG_V4_SHA256)  # deterministic bytes


@needs_training
def test_bar_fits_and_pins_the_normaliser(runs: dict[str, Path]) -> None:
    nm, pins = pe.load_normaliser(runs["bar"])
    doc = json.loads((runs["bar"] / "normaliser.json").read_text())
    assert pins["normaliser_sha256"] == nm.sha256 == doc["sha256"]
    assert doc["fit_rows"]["input_rows"] > 0 and doc["clip"] == 5.0
    assert all(sc > 0 for sc in nm.input_scale) and all(sc > 0 for sc in nm.producer_scale)
    spec = json.loads((runs["train"] / "spec.json").read_text())
    assert spec["pins"]["normaliser_sha256"] == nm.sha256
    assert spec["pins"]["prereg"] == pe.PREREG_SHA256


@needs_training
def test_monitor_runs_at_generation_0_at_two_replicates(runs: dict[str, Path]) -> None:
    import csv

    mon = list(csv.DictReader((runs["train"] / "monitor.csv").open()))
    assert [m["generation"] for m in mon][:2] == ["0", "2"]
    eps = list(csv.DictReader((runs["train"] / "monitor_episodes.csv").open()))
    assert {e["replicate"] for e in eps} == {"0", "1"}
    assert {e["kind"] for e in eps} == {"monitor", "monitor_base"}
    g0 = [m for m in mon if m["generation"] == "0"][0]
    assert g0["g_tr"] in ("", "0.0")  # theta 0 equals rf3 at both keys (or no monitor episode)


# --- task 4.2b: the v3 flag-count rule ---

EXP0080 = REPO / "experiments/EXP-0080-producer-bar-noise"
MARGINS = EXP0080 / "noise_floor/margins.json"
needs_exp0080 = pytest.mark.skipif(not (EXP0080 / "noise_floor/noise_episodes.csv").exists()
                                   or not (TRAINING / "training_worlds.json").exists(),
                                   reason="EXP-0080 / EXP-0058 rows absent")


def _v3_inputs() -> tuple:
    bar, base, bar_pins = pe.load_bar(EXP0080 / "bar")
    margin, m_pins = pe.load_margin(MARGINS)
    fref, f_pins = pe.load_flag_reference(MARGINS, bar, base, margin)
    from tokenomics.train.order_es import a0_values

    refs = {k: v[0] for k, v in a0_values(base).items()}
    return bar, base, margin, refs, fref, {**bar_pins, **m_pins, **f_pins}


@needs_exp0080
def test_rf3_rekeys_pass_the_flag_rule_at_the_registered_rate() -> None:
    """The registered threshold is calibrate_flag_threshold on EXP-0080's files, and rf3's own
    re-keys are disqualified on <= 5% of the re-keyed minibatches (the registered rate)."""
    bar, base, margin, refs, fref, pins = _v3_inputs()
    assert pe.flag_source_problems(pins, pe.FLAG_SOURCES) == []  # v3 / v4's EXP-0080 files
    cal = pe.calibrate_flag_threshold(pe.load_training(TRAINING), fref)
    assert cal["threshold"] == pytest.approx(pe.FLAG_THRESHOLD)
    assert {k: cal[k] for k in pe.FLAG_CALIBRATION} == pytest.approx(pe.FLAG_CALIBRATION)
    assert cal["rate"] <= pe.FLAG_TARGET
    # rf3 at r0 scored as a member on every registered minibatch: never disqualified
    rows = {(d["world"], int(d["episode"])): d for d in base}
    for mb in pe.flag_minibatches(pe.load_training(TRAINING)):
        sc = pe.score_member_v3([rows[k] for k in mb], bar, margin, refs, fref)
        assert not sc.disqualified, sc.reasons


@needs_exp0080
def test_an_always_flagged_member_fails_and_deaths_stay_absolute() -> None:
    bar, base, margin, refs, fref, _ = _v3_inputs()
    rows = {(d["world"], int(d["episode"])): d for d in base}
    mb = pe.flag_minibatches(pe.load_training(TRAINING))[1]  # EXP-0081's minibatch B
    starved = [{**rows[k], "consumer_fill_rate": json.dumps(
        {c: 0.0 for c in json.loads(rows[k]["consumer_fill_rate"])})} for k in mb]
    sc = pe.score_member_v3(starved, bar, margin, refs, fref)
    assert sc.disqualified and any(r.startswith("flags:") for r in sc.reasons)
    # one death where rf3 lives at every key disqualifies; where rf3 dies at some key it does not
    live = next(k for k in mb if k not in fref.dead)
    died = [{**rows[k], "dead": "true"} if k == live else rows[k] for k in mb]
    sc = pe.score_member_v3(died, bar, margin, refs, fref)
    assert sc.disqualified and sc.reasons == (f"{live[0]}:{live[1]}:death",)
    (dead_we,) = sorted(fref.dead)
    sc = pe.score_member_v3([{**rows[dead_we], "dead": "true"}], bar, margin, refs, fref)
    assert not sc.disqualified


def test_real_probe_refuses_other_flag_sources() -> None:
    want = pe.FLAG_SOURCES["margins_sha256"][:12]
    assert pe.flag_source_problems({**pe.FLAG_SOURCES, "margins_sha256": "0" * 64},
                                   pe.FLAG_SOURCES) == [
        f"margins_sha256 000000000000 is not the registered {want}"]


@needs_exp0080
def test_a_death_scores_units_minus_the_penalty_and_is_not_auto_worst() -> None:
    """Task 4.2c (v4): a death where rf3 lives at every key scores units / B_we - 1 in the fitness
    and never disqualifies; a death where rf3 dies at some key is not penalised."""
    bar, base, margin, refs, fref, _ = _v3_inputs()
    rows = {(d["world"], int(d["episode"])): d for d in base}
    mb = pe.flag_minibatches(pe.load_training(TRAINING))[1]
    clean = pe.score_member_v4([rows[k] for k in mb], bar, margin, refs, fref)
    live = next(k for k in mb if k not in fref.dead)
    died = pe.score_member_v4([{**rows[k], "dead": "true"} if k == live else rows[k] for k in mb],
                              bar, margin, refs, fref)
    assert not died.disqualified and died.reasons == (f"{live[0]}:{live[1]}:death-penalised",)
    assert died.fitness == pytest.approx(clean.fitness - 1.0 / len(mb))
    # ranked by fitness: a dying member above a worse living one keeps its rank
    from tokenomics.train.order_es import centered_ranks

    worse = clean.fitness - 2.0 / len(mb)
    r = centered_ranks([died.fitness, worse, clean.fitness], [False, False, False])
    assert list(np.argsort(r)) == [1, 0, 2]
    (dead_we,) = sorted(fref.dead)
    one = pe.score_member_v4([{**rows[dead_we], "dead": "true"}], bar, margin, refs, fref)
    assert not one.disqualified and one.fitness == pytest.approx(
        float(rows[dead_we]["units"]) / bar[dead_we].bar)
    # v3 still disqualifies the same death (the recorded rule, unchanged)
    assert pe.score_member_v3([{**rows[k], "dead": "true"} if k == live else rows[k] for k in mb],
                              bar, margin, refs, fref).disqualified


def test_v5_real_runs_are_refused_until_the_rf4_calibration_exists() -> None:
    if pe.FLAG_SOURCES_V5 is None:
        assert pe.flag_source_problems(dict(pe.FLAG_SOURCES)) == [
            "no registered flag calibration for this pre-registration (v5: EXP-0085)"]
    else:
        assert pe.flag_source_problems(dict(pe.FLAG_SOURCES_V5)) == []

