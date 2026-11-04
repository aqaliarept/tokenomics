"""Task 7.6: the re-based step 2 (user decision "Re-base on MWW"): the per-order policy's frozen
base is MWW's keyed whole-order choice (MWWk); A0 == MWWk exactly at theta = 0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_per_order_scorer import _play, base, fit_rows, job, normaliser  # noqa: F401
from tokenomics.env.spaces import CHOICE_V1, LOGITS_V1
from tokenomics.train import per_order_scorer as pos
from tokenomics.train.per_order_scorer import (ARMS, BASE_MWW, BASE_TS, BaseScorer,
                                               FeatureNormaliser, MWWBasePolicy, PerOrderPolicy,
                                               make_policy)


@pytest.fixture(scope="module")
def mwwk(job: dict) -> object:  # noqa: F811
    from tokenomics.eval.routing_diagnostics import ordering_mww_keyed

    return _play(job, order=ordering_mww_keyed, decode=LOGITS_V1)


@pytest.mark.parametrize("arm,theta", [("A0", None), ("A1", "zeros"), ("A1", "initial")])
def test_mww_base_at_zero_residual_equals_mwwk(job: dict, base: BaseScorer,  # noqa: F811
                                               normaliser: FeatureNormaliser,  # noqa: F811
                                               mwwk: object, arm: str, theta: str | None) -> None:
    """A0 (and A1 at theta = 0) on the MWW base under choice-v1 routes exactly as MWWk under
    logits-v1: the same sub-orders, episode statistics, and byte-identical draw weights and
    uniforms (the logit add is computed multiplicatively, so theta = 0 is exact)."""
    th = None if theta is None else (np.zeros(ARMS[arm].n_params) if theta == "zeros"
                                     else ARMS[arm].initial_theta())
    nm = normaliser if ARMS[arm].residual else None
    run = _play(job, policy=make_policy(base, arm, th, nm, base_kind=BASE_MWW))
    assert len(run.cmds) > 50 and run.cmds == mwwk.cmds  # type: ignore[attr-defined]
    assert run.stats == mwwk.stats  # type: ignore[attr-defined]
    assert len(run.draws) > 100 and run.draws == mwwk.draws  # type: ignore[attr-defined]


def test_mww_base_residual_reweights_within_mww_support(job: dict, base: BaseScorer,  # noqa: F811
                                                        normaliser: FeatureNormaliser  # noqa: F811
                                                        ) -> None:
    """A nonzero residual changes the draws but never routes to a producer MWW gives weight 0
    while another eligible producer has weight > 0 (log 0 = -inf is not lifted)."""
    th = np.random.default_rng(3).normal(0.0, 1.0, ARMS["A1"].n_params)
    run = _play(job, policy=make_policy(base, "A1", th, normaliser, base_kind=BASE_MWW))
    decisions = [d for *_, ds in run.log for d in ds]
    assert any(np.any(d.delta != 0.0) for d in decisions)
    mixed = [d for d in decisions if np.isneginf(d.b).any() and np.isfinite(d.b).any()]
    assert mixed, "fixture has no slot with an unmakeable eligible producer"
    for d in mixed:
        assert np.isfinite(d.b[d.eligible.index(d.chosen)])


def test_base_kind_in_pins_and_composite(job: dict, base: BaseScorer,  # noqa: F811
                                         normaliser: FeatureNormaliser,  # noqa: F811
                                         tmp_path: Path) -> None:
    """The MWW base is pinned (pins, composite) and survives a save / load; a TS policy's pins
    and composite are unchanged (no base-kind key)."""
    from types import SimpleNamespace

    th = ARMS["A1"].initial_theta()
    ts = PerOrderPolicy(base, "A1", th, normaliser)
    mw = make_policy(base, "A1", th, normaliser, base_kind=BASE_MWW)
    assert "base_kind" not in pos.per_order_pins(ts)["per_order"]
    assert "kind" not in pos.composite_dict(ts)["base"]
    assert pos.per_order_pins(mw)["per_order"]["base_kind"] == BASE_MWW
    assert pos.composite_dict(mw)["base"]["kind"] == BASE_MWW
    env = SimpleNamespace(ordering_decode=CHOICE_V1)
    for pol, kind in ((ts, BASE_TS), (mw, BASE_MWW)):
        path = tmp_path / f"{kind}.json"
        pos.save_composite(pol, path)
        got = pos.load_composite(path, base.path, env)  # type: ignore[arg-type]
        assert pos.base_kind_of(got) == kind
    bad = json.loads((tmp_path / "mww.json").read_text())
    bad["base"]["kind"] = "other"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="base kind"):
        pos.load_composite(tmp_path / "bad.json", base.path, env)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_kind"):
        make_policy(base, "A0", base_kind="other")
    assert isinstance(mw, MWWBasePolicy) and type(ts) is PerOrderPolicy


# --- pre-registration v3 ---

from tokenomics.train import order_es as es  # noqa: E402
from tokenomics.train import order_es_validate as v  # noqa: E402
from tokenomics.eval.provenance import sha256_file  # noqa: E402


def test_v3_frozen_and_v1_v2_unchanged() -> None:
    """v3 is frozen and pinned; v1 / v2 still freeze to their files (the normaliser
    fitting-code hash, which covers PerOrderPolicy, is unchanged); v3 = v2's training rule on
    the MWW base, A1 only, no probe, the readings identical."""
    for ver, path, sha in ((1, v.PREREG_PATH, v.PREREG_SHA256),
                           (2, v.PREREG_V2_PATH, v.PREREG_V2_SHA256),
                           (3, v.PREREG_V3_PATH, v.PREREG_V3_SHA256)):
        assert v.run_freeze(path, log=lambda m: None, version=ver) == 0
        assert sha256_file(str(path)) == sha and v.FROZEN_PREREGS[sha] == ver
    d1, d2, d3 = (es.load_prereg(p) for p in (v.PREREG_PATH, v.PREREG_V2_PATH, v.PREREG_V3_PATH))
    assert d1["readings"] == d2["readings"] == d3["readings"]
    assert d1["normaliser"] == d3["normaliser"]  # fitting code unchanged
    assert d3["es"]["base"]["kind"] == BASE_MWW and d3["es"]["trainable_arms"] == ["A1"]
    assert d3["es"]["base"]["code"]["sha256"] == es.mww_base_code_sha256()
    assert {k: d3["es"][k] for k in ("minibatch", "pairs", "generations", "adam")} == {
        k: d2["es"][k] for k in ("minibatch", "pairs", "generations", "adam")}
    assert d3["es"]["disqualification"]["noise_margin"] == d2["es"]["disqualification"][
        "noise_margin"]
    assert d3["probe"]["run"] is False and v.prereg_read_problems(d3) == []
    assert es.run_rule(d3)[0] == 3 and es.base_kind_for(3) == BASE_MWW
    assert es.base_kind_for(2) == BASE_TS
    assert es.prereg_problems(d3, mode=es.TRAIN, arm="A1", opt_seed=0, generations=40,
                              minibatch=24) == []
    assert any("trainable arm" in x for x in es.prereg_problems(
        d3, mode=es.TRAIN, arm="A2-R2", opt_seed=0, generations=40, minibatch=24))
    assert any("registers no probe" in x for x in es.prereg_problems(
        d3, mode=es.PROBE, arms=["A1"], minibatch=24))
    rep = d3["validation"]["reported_arms"]["EXP-0060 A1-s0"]
    assert rep["theta_sha256"].startswith("169f4029") and rep["base"] == BASE_TS
    c = v.carried_ii_for(3)
    assert c is not None and c.within and c.diff == -14.125 and v.carried_ii_for(2) is None
    assert d3["changes"]["from"]["sha256"] == v.PREREG_V2_SHA256


# --- v3 end to end (dry runs): bar on the MWW base, A1 training, the read ---

from test_order_es_validate import (ATLAS_V2, ORDERER, VAL_WORLD, _lfs_present,  # noqa: E402
                                    inputs, v2_runs)
from tokenomics.train import order_es_worlds as ow  # noqa: E402

EXP0060_A1 = Path(__file__).resolve().parents[2] / "experiments/EXP-0060-es-main/artifacts/a1-s0"
QUIET = lambda m: None  # noqa: E731
needs_inputs = pytest.mark.skipif(not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)),
                                  reason="EXP-0052 atlas / EXP-0012 checkpoint absent")


@pytest.fixture(scope="module")
def v3_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A 2-world horizon-12 dry-run bar whose A0 plays the MWW base, and a v3 dry run of A1."""
    if not (_lfs_present(ATLAS_V2) and _lfs_present(ORDERER)):
        pytest.skip("EXP-0052 atlas / EXP-0012 checkpoint absent")
    from test_order_es import _fake

    root = tmp_path_factory.mktemp("v3")
    bar = root / "bar"
    assert ow.main(["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run",
                    "--atlas-worlds", "1", "--target", "2", "--min-feasible", "1",
                    "--horizon", "12", "--shock-profile", "shocks-i-smoke", "--workers", "1",
                    "--base", "mww", "--out-dir", str(bar)]) == 0
    size = int(es.load_inputs(bar).schedule["minibatch_size"])
    run = root / "A1-s0"
    assert es.run_train(bar_dir=bar, out_dir=run, orderer_ckpt=str(ORDERER), arm="A1",
                        opt_seed=0, generations=2, pairs=2, monitor_every=1, minibatch=size,
                        prereg_path=str(v.PREREG_V3_PATH), dry_run=True,
                        run_job_fn=_fake(bar), log=QUIET) == 0
    return {"bar": bar, "A1-s0": run, "size": Path(str(size))}


@needs_inputs
def test_v3_bar_and_training_pin_the_mww_base(v3_inputs: dict[str, Path],
                                              inputs: dict[str, Path]) -> None:  # noqa: F811
    """The bar's A0 pins the MWW base; a v3 run pins it, version 3 and its composite's base;
    a v3 run on a TS bar, and a v2 run on the MWW bar, are refused before any job."""
    bar_spec = json.loads((v3_inputs["bar"] / "spec.json").read_text())
    assert bar_spec["pins"]["per_order"]["base_kind"] == BASE_MWW
    ts_spec = json.loads((inputs["bar"] / "spec.json").read_text())
    assert "base_kind" not in ts_spec["pins"]["per_order"]
    spec = json.loads((v3_inputs["A1-s0"] / "spec.json").read_text())
    tp = spec["pins"]["training"]
    assert tp["per_order"]["base_kind"] == BASE_MWW
    assert tp["disqualification"]["prereg_version"] == 3
    comp = json.loads((v3_inputs["A1-s0"] / "candidate.json").read_text())
    assert comp["base"]["kind"] == BASE_MWW
    rows = list(__import__("csv").DictReader((v3_inputs["A1-s0"] / "members.csv").open()))
    assert rows  # the fake job fn played them; the real one is test_mww_base_* above
    kw = dict(orderer_ckpt=str(ORDERER), arm="A1", opt_seed=0, generations=2, pairs=2,
              monitor_every=1, minibatch=int(str(v3_inputs["size"])), dry_run=True, log=QUIET)
    with pytest.raises(ValueError, match="the bar's A0 plays the 'ts' base"):
        es.run_train(bar_dir=inputs["bar"], out_dir=v3_inputs["bar"].parent / "x",
                     prereg_path=str(v.PREREG_V3_PATH), **kw)
    with pytest.raises(ValueError, match="the bar's A0 plays the 'mww' base"):
        es.run_train(bar_dir=v3_inputs["bar"], out_dir=v3_inputs["bar"].parent / "y",
                     prereg_path=str(v.PREREG_V2_PATH), **kw)
    with pytest.raises(ValueError, match="registers no probe"):
        es.run_probe(bar_dir=v3_inputs["bar"], out_dir=v3_inputs["bar"].parent / "z",
                     orderer_ckpt=str(ORDERER), pairs=2, prereg_path=str(v.PREREG_V3_PATH),
                     dry_run=True, log=QUIET)


@needs_inputs
@pytest.mark.skipif(not (EXP0060_A1 / "candidate.json").exists(), reason="EXP-0060 run absent")
def test_v3_read_with_the_reported_arm(  # noqa: F811
        v3_inputs: dict[str, Path], inputs: dict[str, Path], v2_runs: dict[str, Path],
        tmp_path: Path) -> None:
    """A v3 read of the v3 run plays A0 on the MWW base and the reported EXP-0060 A1-s0 (TS
    base), reports the latter's comparisons outside the clauses, carries clause (ii), and
    recomputes; a TS bar, a v2 run, or a reported arm under v2 are refused."""
    rep = {"EXP-0060 A1-s0": str(EXP0060_A1)}
    args = dict(bar_dir=v3_inputs["bar"], atlas_path=ATLAS_V2, orderer_ckpt=str(ORDERER),
                episodes=1, worlds=[VAL_WORLD], horizon=12, shock_profile="shocks-i-smoke",
                dry_run=True, log=QUIET, prereg_path=v.PREREG_V3_PATH)
    out = tmp_path / "read"
    assert v.run_validate(runs=[str(v3_inputs["A1-s0"])], out_dir=out, reported_runs=rep,
                          **args) == 0
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["reported"]["reported:EXP-0060-A1-s0"]["base"] == BASE_TS
    jobs = list(__import__("csv").DictReader((out / "episodes.csv").open()))
    assert {"es/reported:EXP-0060-A1-s0", "es/A1-s0"} <= {d["subject"] for d in jobs}
    r = json.loads((out / "reading.json").read_text())
    if "not_read" not in r:
        assert set(r["reported"]["reported:EXP-0060-A1-s0"]) >= {"A0", "RR", "MWW", "TR",
                                                                  "A1-s0"}
        assert "reported:EXP-0060-A1-s0" not in r["candidates"]
        assert r["seed0"]["plateau"]["ii"] is True
        assert r["seed0"]["a2_vs_a1"]["diff"] == -14.125
        assert v.run_recompute(out_dir=out, runs=[str(v3_inputs["A1-s0"])],
                               bar_dir=v3_inputs["bar"], orderer_ckpt=str(ORDERER),
                               log=QUIET) == 0
    with pytest.raises(ValueError, match="another bar|the bar's A0 plays the 'ts' base"):
        v.run_validate(runs=[str(v3_inputs["A1-s0"])], out_dir=tmp_path / "a",
                       **{**args, "bar_dir": inputs["bar"]})
    with pytest.raises(ValueError, match="pre-registration mix"):
        v.run_validate(runs=[str(v2_runs["A1-s0"])], out_dir=tmp_path / "b",
                       **{**args, "bar_dir": inputs["bar"]})
    with pytest.raises(ValueError, match="registers no reported arm"):
        v.run_validate(runs=[str(v2_runs["A1-s0"]), str(v2_runs["A2-s0"])],
                       out_dir=tmp_path / "c", reported_runs=rep,
                       **{**args, "bar_dir": inputs["bar"], "prereg_path": v.PREREG_V2_PATH})


def test_reported_arms_must_be_the_registered_ones() -> None:
    good = v.Reported("reported:EXP-0060-A1-s0", "A1", (), "169f402915ece660b59411bc7a237d33a5ce5f"
                      "12037b7faf9f3a6e52a262d68d", {}, "", "d19332e1a625a3090d8571b444589604a5"
                      "89f26bef3cbc16592ba01147f579a9", BASE_TS)
    assert v.reported_problems([good], 3) == []
    bad = v.Reported(good.label, "A1", (), "0" * 64, {}, "", good.candidate_sha256, BASE_TS)
    assert any("theta_sha256" in x for x in v.reported_problems([bad], 3))
    assert any("!= the registered" in x for x in v.reported_problems([], 3))
