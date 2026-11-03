"""Task 6.1d: pre-registration v2 (user decision "Bigger batches + fix rule"): minibatch 24 for
the probe and the ES estimate, and the noise margin on the relative disqualification rule, with
v1 behaviour byte-identical when v1 is loaded (EXP-0058 / EXP-0068 reproduce)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_order_es import ORDERER, QUIET, _bar, _fake, _row  # noqa: F401
from tokenomics.eval.provenance import sha256_file
from tokenomics.train import order_es as es
from tokenomics.train import order_es_validate as v
from tokenomics.train import order_es_worlds as ow

REPO = Path(__file__).resolve().parents[2]
EXP0058 = REPO / "experiments/EXP-0058-es-gradient-probe"
EXP0068 = REPO / "experiments/EXP-0068-es-noise-floor"


def _real(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs_0058 = pytest.mark.skipif(
    not (_real(EXP0058 / "artifacts/probe_episodes.csv") and _real(EXP0058 / "bar/bar.csv")
         and _real(EXP0058 / "artifacts/bar_episodes.csv") and _real(ORDERER)),
    reason="EXP-0058 records / EXP-0012 checkpoint absent")
needs_0068 = pytest.mark.skipif(not _real(EXP0068 / "noise_floor.csv"),
                                reason="EXP-0068 records absent")


def _rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --- the pre-registration files ---


def test_v1_unchanged_and_v2_frozen() -> None:
    """v1: the module still freezes to the frozen file byte for byte (`b9d449c1…`), version 1 is
    the default. v2: the packaged file is `PREREG_V2_SHA256`, byte-equal to `frozen_prereg(2)`,
    passes its own checks at minibatch 24 and records what changed and why."""
    assert es.registered_sections() == es.registered_sections(1)
    assert v.run_freeze(v.PREREG_PATH, log=QUIET) == 0
    assert sha256_file(str(v.PREREG_PATH)) == v.PREREG_SHA256
    assert v.run_freeze(v.PREREG_V2_PATH, log=QUIET, version=2) == 0
    assert sha256_file(str(v.PREREG_V2_PATH)) == v.PREREG_V2_SHA256
    doc = es.load_prereg(v.PREREG_V2_PATH)
    assert es.prereg_version(doc) == 2 and doc["status"] == es.FROZEN
    assert es.prereg_problems(doc, mode=es.PROBE, arms=list(es.V2_PROBE_ARMS),
                              minibatch=24) == []
    assert any("probe arms" in x for x in es.prereg_problems(  # v2 needs the sigma' arm
        doc, mode=es.PROBE, arms=["A1", "A2"], minibatch=24))
    assert doc["probe"]["sigma_arms"]["A1-S2"]["sigma"] == pytest.approx(0.10)
    why = json.dumps(doc["changes"]["why"])
    assert "not robust" in why and "flat at sigma not excluded" in why
    assert any("probe minibatch 6" in x for x in es.prereg_problems(
        doc, mode=es.PROBE, arms=["A1", "A2"], minibatch=6))
    assert es.prereg_problems(doc, mode=es.TRAIN, arm="A1", opt_seed=0, generations=20,
                              minibatch=24) == []
    assert any("minibatch 12" in x for x in es.prereg_problems(
        doc, mode=es.TRAIN, arm="A1", opt_seed=0, generations=20, minibatch=12))
    assert es.prereg_file_problems(v.PREREG_V2_PATH) == [] == es.prereg_file_problems(
        v.PREREG_PATH)
    m = doc["es"]["disqualification"]["noise_margin"]
    assert (m["rundown"], m["starvation"]) == (es.NOISE_MARGIN_V2["rundown"],
                                               es.NOISE_MARGIN_V2["starvation"])
    assert m["source"]["sha256"] == es.EXP0068_NOISE_FLOOR_SHA256
    assert "EXP-0068" in json.dumps(doc["changes"]) and "EXP-0058" in json.dumps(doc["changes"])
    assert doc["changes"]["from"]["sha256"] == v.PREREG_SHA256
    assert doc["readings"] == v.readings_registration()  # validation rules unchanged
    # the v1 document under the v1 rules still reads clean; v1 content relabelled v2 does not
    v1doc = es.load_prereg(v.PREREG_PATH)
    assert es.prereg_problems(v1doc, mode=es.PROBE, arms=["A1", "A2"], minibatch=6) == []
    relabel = {**v1doc, "format": es.PREREG_V2_FORMAT}
    assert es.prereg_problems(relabel, mode=es.PROBE, arms=["A1", "A2"], minibatch=6)
    assert es.run_rule(v1doc) == (1, None) and es.run_rule(None) == (1, None)
    assert es.run_rule(doc) == (2, es.NoiseMargin(**es.NOISE_MARGIN_V2))


def test_a_file_with_the_wrong_version_for_its_sha_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "p.json"
    other.write_text(v.PREREG_PATH.read_text().replace("m4-es-prereg-v1", "m4-es-prereg-v2"))
    assert "is not the frozen" in "; ".join(es.prereg_file_problems(other))


# --- the margin ---


def test_margin_arithmetic() -> None:
    """Run-down: below X and A0 not run down is flagged only when A0's ratio exceeds the
    member's by more than the margin (strictly); starvation per consumer the same against A0's
    fill; a missing A0 value leaves the v1 test; death has no margin; no margin = v1."""
    x, y = ow.X_RUNDOWN, ow.Y_STARVATION
    m = es.NoiseMargin(rundown=0.1, starvation=0.1)
    member = _row(r_start="1.0", r_end=repr(x - 0.05),
                  consumer_fill_rate=json.dumps({"c0": y - 0.05, "c1": y - 0.2}))
    near = es.A0Ref(ratio=x + 0.02, fills={"c0": y + 0.02, "c1": y + 0.02})
    far = es.A0Ref(ratio=x + 0.2, fills={"c0": y + 0.2, "c1": y + 0.2})
    assert es.disqualifying(member, _bar()) == ["rundown", "starvation:c0,c1"]  # v1
    assert es.disqualifying(member, _bar(), m, near) == ["starvation:c1"]
    assert es.disqualifying(member, _bar(), m, far) == ["rundown", "starvation:c0,c1"]
    edge = es.A0Ref(ratio=(x - 0.05) + 0.1, fills={"c0": (y - 0.05) + 0.1, "c1": 1.0})
    got = es.disqualifying(member, _bar(), es.NoiseMargin(0.1 + 1e-12, 0.1 + 1e-12), edge)
    assert got == ["starvation:c1"]  # A0 - member == margin (to rounding): not beyond it
    missing = es.A0Ref(ratio=None, fills={"c0": None})
    assert es.disqualifying(member, _bar(), m, missing) == ["rundown", "starvation:c0,c1"]
    dead = _row(dead="true")
    assert es.disqualifying(dead, _bar(), m, far) == ["death"]
    with pytest.raises(ValueError, match="reference"):
        es.disqualifying(member, _bar(), m, None)
    bar = {("w", 1): _bar()}
    s = es.score_member([member], bar, margin=m, refs={("w", 1): near})
    assert s.disqualified and s.reasons == ("w:1:starvation:c1",)
    with pytest.raises(ValueError, match="no A0 reference"):
        es.score_member([member], bar, margin=m, refs={})


@needs_0068
@needs_0058
def test_margins_reproduce_from_exp0068_and_meet_the_target() -> None:
    """The registered margins recompute from EXP-0068's pinned noise_floor.csv; without the
    margin re-keyed A0 self-flags 16 / 48 (= EXP-0068 check 2), with it <= 2 / 48 (the target;
    in-sample); the leave-one-world-out refit (reported only) flags each held-out world-episode
    0 times."""
    path = EXP0068 / "noise_floor.csv"
    assert sha256_file(str(path)) == es.EXP0068_NOISE_FLOOR_SHA256
    rows = _rows(path)
    got = es.noise_margins(rows)
    assert (got["rundown"], got["starvation"]) == (es.NOISE_MARGIN_V2["rundown"],
                                                   es.NOISE_MARGIN_V2["starvation"])
    assert {k: got[k] for k in ("df_rundown", "df_starvation")} == {
        "df_rundown": es.NOISE_SIGMA_V2["df_rundown"],
        "df_starvation": es.NOISE_SIGMA_V2["df_starvation"]}
    bar = es.read_bar(EXP0058 / "bar/bar.csv")
    assert es.self_flags(rows, bar, None) == (16, 48)
    hit, n = es.self_flags(rows, bar, es.NoiseMargin(**es.NOISE_MARGIN_V2))
    assert n == es.NOISE_MARGIN_TARGET["episodes"] and hit <= es.NOISE_MARGIN_TARGET[
        "self_flags_max"]
    loo = 0
    for k in sorted({(d["world"], int(d["episode"])) for d in rows}):
        mm = es.noise_margins(rows, exclude=frozenset({k}))
        loo += es.self_flags(rows, bar, es.NoiseMargin(mm["rundown"], mm["starvation"]),
                             only=frozenset({k}))[0]
    assert loo == 0  # reported in notes/6.1d.md, never deciding


# --- v1 byte-identical ---


@needs_0058
def test_v1_rescoring_reproduces_exp0058() -> None:
    """Re-scoring EXP-0058's recorded probe rows through `score_member` (v1, no margin) and the
    estimate gives exactly the recorded probe_fitness.csv and probe.json reading."""
    bar = es.read_bar(EXP0058 / "bar/bar.csv")
    rows = {d["job_id"]: d for d in _rows(EXP0058 / "artifacts/probe_episodes.csv")}
    rec = {(d["arm"], int(d["batch"]), d["member"]): d for d in _rows(
        EXP0058 / "probe_fitness.csv")}
    for arm in ("A1", "A2"):
        for b in (0, 1):
            mine = {j: d for j, d in rows.items() if d["arm"] == arm and int(d["batch"]) == b}
            for lab, _, _ in es.member_labels(es.PAIRS):
                s = es.score_member(es._member_rows(mine, lab), bar)
                r = rec[(arm, b, lab)]
                assert repr(s.fitness) == r["fitness"] and str(s.disqualified) == r[
                    "disqualified"] and ";".join(s.reasons) == r["reasons"]
    doc = json.loads((EXP0058 / "probe.json").read_text())
    again = es.probe_from_rows(_rows(EXP0058 / "probe_fitness.csv"), ["A1", "A2"], es.PAIRS)
    assert again["reading"]["reading"] == doc["reading"]["reading"] == "no_signal"
    assert again["reading"]["minibatch"] == doc["reading"]["minibatch"] == 12
    assert again["arms"]["A1"]["cosine"] == pytest.approx(doc["arms"]["A1"]["cosine"], abs=1e-15)


# --- minibatch 24 from the recorded stream ---


@pytest.fixture(scope="module")
def real_bar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """EXP-0058's bar with its own episodes.csv restored (the v2 margin's A0 references)."""
    if not (_real(EXP0058 / "bar/bar.csv") and _real(EXP0058 / "artifacts/bar_episodes.csv")):
        pytest.skip("EXP-0058 records absent")
    out = tmp_path_factory.mktemp("bar") / "bar"
    shutil.copytree(EXP0058 / "bar", out, ignore=shutil.ignore_patterns("fit_rows"))
    shutil.copy(EXP0058 / "artifacts/bar_episodes.csv", out / "episodes.csv")
    return out


@needs_0058
def test_probe_minibatch_24_comes_from_the_recorded_stream(real_bar: Path) -> None:
    """Size 6 is still the bar's schedule entry (= EXP-0058's probe); size 24 is the recorded
    probe stream, two disjoint minibatches on 48 distinct worlds."""
    inp = es.load_inputs(real_bar)
    six = es.probe_batches(inp)
    rec = json.loads((EXP0058 / "probe.json").read_text())["minibatches"]
    assert [list(map(list, m)) for m in six] == rec
    a, b = es.probe_batches(inp, 24)
    assert (a, b) == ow.probe_minibatches(inp.world_episodes, 24)
    assert len(a) == len(b) == 24 and len({w for w, _ in a + b}) == 48
    assert es.probe_batches(inp, 6) == six


@needs_0058
def test_a0_refs_checked_against_the_bar(real_bar: Path, tmp_path: Path) -> None:
    inp = es.load_inputs(real_bar)
    refs = es.load_a0_refs(inp)
    assert set(refs) == set(inp.bar) and all(r.ratio is not None for r in refs.values())
    bad = tmp_path / "bar"
    shutil.copytree(real_bar, bad)
    rows = _rows(bad / "episodes.csv")
    a0 = next(d for d in rows if d["arm"] == ow.A0)
    a0["units"] = repr(float(a0["units"]) + 1.0)
    with (bad / "episodes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="disagrees with bar.csv"):
        es.load_a0_refs(es.load_inputs(bad))
    (bad / "episodes.csv").unlink()
    with pytest.raises(ValueError, match="missing"):
        es.load_a0_refs(es.load_inputs(bad))


@needs_0058
def test_v2_probe_and_train_dry_runs(real_bar: Path, tmp_path: Path) -> None:
    """Under v2 (dry runs, synthetic rows): the probe plays two 24-world-episode minibatches,
    pins the margin and the A0 references, records the training minibatch 24 in its reading; a
    train generation plays the size-24 stream minibatch. Under no file (v1) neither pin
    appears."""
    out = tmp_path / "probe"
    assert es.run_probe(bar_dir=real_bar, out_dir=out, orderer_ckpt=str(ORDERER), pairs=2,
                        arms=es.V2_PROBE_ARMS, prereg_path=str(v.PREREG_V2_PATH),
                        dry_run=True, run_job_fn=_fake(real_bar), log=QUIET) == 0
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["disqualification"]["noise_margin"] == es.NOISE_MARGIN_V2
    assert all(len(m) == 24 for m in spec["pins"]["minibatches"])
    doc = json.loads((out / "probe.json").read_text())
    assert doc["prereg_version"] == 2 and doc["training_minibatch"] == 24
    assert doc["reading"]["minibatch"] == 24
    again = es.probe_from_rows(_rows(out / "probe_fitness.csv"), es.V2_PROBE_ARMS, 2, 24)
    assert json.loads(json.dumps(again)) == {k: doc[k] for k in ("arms", "reading")}
    assert doc["reading"]["step_reading"] in (es.STEP_FLAT, es.STEP_BOTH, es.STEP_SIGMA_ONLY,
                                              es.NO_SIGNAL)
    assert doc["pre_registered"]["sigma_arms"] == es.SIGMA_ARM_REGISTRATION
    # A1-S2 plays A1's policy on the same world-episodes and keys, at twice A1's deviations
    eps = {d["job_id"]: d for d in _rows(out / "probe_episodes.csv")}
    key = {(d["arm"], d["member"], d["world"], d["episode"], d["replicate"]) for d in
           eps.values()}
    assert {k[1:] for k in key if k[0] == "A1"} == {k[1:] for k in key if k[0] == "A1-S2"}
    v1 = tmp_path / "probe_v1"
    assert es.run_probe(bar_dir=real_bar, out_dir=v1, orderer_ckpt=str(ORDERER), pairs=2,
                        dry_run=True, run_job_fn=_fake(real_bar), log=QUIET) == 0
    spec1 = json.loads((v1 / "spec.json").read_text())
    assert "disqualification" not in spec1["pins"]
    assert all(len(m) == 6 for m in spec1["pins"]["minibatches"])
    assert "prereg_version" not in json.loads((v1 / "probe.json").read_text())
    tr = tmp_path / "train"
    assert es.run_train(bar_dir=real_bar, out_dir=tr, orderer_ckpt=str(ORDERER), arm="A1",
                        opt_seed=0, generations=1, pairs=1, monitor_every=1,
                        prereg_path=str(v.PREREG_V2_PATH), dry_run=True,
                        run_job_fn=_fake(real_bar), log=QUIET) == 0
    inp = es.load_inputs(real_bar)
    played = sorted({(d["world"], int(d["episode"])) for d in _rows(tr / "members.csv")})
    assert played == sorted(es.minibatch_for(inp, 0, 1, 24)) and len(played) == 24
    assert json.loads((tr / "spec.json").read_text())["pins"]["training"][
        "disqualification"]["prereg_version"] == 2


def test_a_v2_run_needs_a_v2_probe() -> None:
    """A v1 probe reading (EXP-0058, or a v1 `weak` at 24) never licenses a v2 training run, and
    a v2 reading never a v1 run; under v2 `signal` and `weak` both train at 24."""
    def doc(cos: float, version: int | None) -> dict:
        r = es.probe_reading({"A1": cos, "A2": 0.5}, 24 if version == 2 else None)
        return {"status": "complete", "dry_run": False, "reading": r,
                **({"prereg_version": version} if version else {})}

    P = es._probe_doc_problems
    assert P(doc(0.5, 2), "A1", 24, 2) == [] == P(doc(0.2, 2), "A1", 24, 2)
    assert "pre-registration v1" in P(doc(0.2, None), "A1", 24, 2)[0]
    assert "pre-registration v2" in P(doc(0.2, 2), "A1", 24)[0]
    assert P(doc(0.5, None), "A1", 12) == []  # v1 unchanged


# --- the boundary of the margin (a `>=` mutant must fail) ---


def test_margin_boundary_is_strict() -> None:
    """A0's value minus the member's exactly equal to the margin (dyadic values, exact in
    binary) is NOT beyond it: not flagged. One ulp beyond is flagged. A `>=` comparison fails
    the first half, so this pins the registered strict `>`."""
    x, y = ow.X_RUNDOWN, ow.Y_STARVATION
    assert 0.625 < x <= 0.875 and 0.5 < y <= 0.75  # member below, A0 not
    member = _row(r_start="1.0", r_end="0.625", consumer_fill_rate=json.dumps({"c0": 0.5}))
    ref = es.A0Ref(ratio=0.875, fills={"c0": 0.75})
    assert 0.875 - 0.625 == 0.25 and 0.75 - 0.5 == 0.25  # exactly the margin
    at = es.NoiseMargin(rundown=0.25, starvation=0.25)
    assert es.disqualifying(member, _bar(), at, ref) == []
    below = es.NoiseMargin(rundown=float(np.nextafter(0.25, 0.0)),
                           starvation=float(np.nextafter(0.25, 0.0)))
    assert es.disqualifying(member, _bar(), below, ref) == ["rundown", "starvation:c0"]


# --- the larger-sigma probe arm (user decision 2026-11-03) ---


def test_sigma_arm_is_a1_at_twice_sigma() -> None:
    """A1-S2: A1's policy, initial theta and eps_i, sigma x 2 (0.10); diagnostic only (never
    trainable, not a per_order_scorer arm); the v1 registration does not know it."""
    assert es.base_arm("A1-S2") == "A1" and es.base_arm("A2") == "A2"
    assert es.arm_spec("A1-S2") is es.ARMS["A1"]
    assert np.array_equal(es.arm_sigma("A1-S2"), 2.0 * es.ARMS["A1"].sigma())
    assert float(es.arm_sigma("A1-S2")[0]) == pytest.approx(0.10)
    n = es.ARMS["A1"].n_params
    assert np.array_equal(es.probe_noise("A1-S2", 12, n), es.probe_noise("A1", 12, n))
    assert np.array_equal(es.probe_noise("A1", 12, n), es.noise(0, 0, 12, n,
                                                                 purpose="probe|A1"))  # v1
    assert "A1-S2" not in es.TRAINABLE_ARMS and "A1-S2" not in es.ARMS
    assert "A1-S2" not in json.dumps(es.registered_sections(1))
    eps = es.probe_noise("A1", 2, n)
    th1 = es.member_thetas(np.zeros(n), eps, es.arm_sigma("A1"))
    th2 = es.member_thetas(np.zeros(n), eps, es.arm_sigma("A1-S2"))
    assert all(np.allclose(th2[k], 2.0 * th1[k]) for k in th1)


@pytest.mark.parametrize("a1,a1s2,step", [
    (0.05, 0.2, "flat_at_sigma"), (0.05, 0.1, "flat_at_sigma"),
    (0.35, 0.4, "heterogeneity"), (0.1, 0.12, "heterogeneity"),
    (0.0, 0.0999, "no_signal"), (None, None, "no_signal"),
    (0.31, 0.05, "sigma_only"), (0.2, None, "sigma_only")])
def test_step_reading_table(a1: float | None, a1s2: float | None, step: str) -> None:
    """The pre-registered sigma / 2 sigma table, D7's thresholds (>= 0.1); A1's own reading,
    the training licence, is unchanged by the extra arm; v1 readings carry no step key."""
    r = es.probe_reading({"A1": a1, "A2": 0.5, "A1-S2": a1s2}, 24)
    assert r["step_reading"] == step and r["step_next"] == es.STEP_NEXT[step]
    plain = es.probe_reading({"A1": a1, "A2": 0.5}, 24)
    assert {k: r[k] for k in plain} == plain and "step_reading" not in plain
