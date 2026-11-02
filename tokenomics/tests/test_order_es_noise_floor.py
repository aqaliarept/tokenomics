"""Task 6.1c: `order_es --mode noise-floor` (DIAGNOSTICS 25 checks 2 and 3 for EXP-0058's
`no_signal`): A0 at replicate keys r0..rK on the probe's recorded world-episodes, r0 against the
bar's A0 row, the self-disqualification rate and the pair-difference / replicate-spread ratios."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from test_order_es import ORDERER, QUIET, _fake, bar_dir, needs_inputs  # noqa: F401 - fixture
from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.routing_diagnostics import DrawKey, keyed_uniform
from tokenomics.train import order_es as es
from tokenomics.train import order_es_worlds as ow


@pytest.fixture(scope="module")
def probe_dir(bar_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:  # noqa: F811
    """A dry probe on the fixture bar (synthetic rows, 3 pairs): probe.json + probe_fitness.csv
    + probe_episodes.csv, the files a noise-floor run reads."""
    out = tmp_path_factory.mktemp("probe") / "probe"
    assert es.run_probe(bar_dir=bar_dir, out_dir=out, orderer_ckpt=str(ORDERER), pairs=3,
                        dry_run=True, run_job_fn=_fake(bar_dir), log=QUIET) == 0
    return out


def _wes(probe_dir: Path) -> list[tuple[str, int]]:
    return [we for mb in es.probe_world_episodes(json.loads((probe_dir / "probe.json")
                                                            .read_text())) for we in mb]


@needs_inputs
def test_replicate_keys_differ_from_r0_and_each_other(bar_dir: Path,  # noqa: F811
                                                      probe_dir: Path) -> None:
    """Each world-episode gets replicates 0..K once; the draw keys (and the keyed uniforms they
    give) differ from r0's and from each other; the jobs are A0 monitor-path jobs (theta [])."""
    inp = es.load_inputs(bar_dir)
    wes = _wes(probe_dir)
    jobs = es.noise_jobs(inp, {"path": str(ORDERER), "sha256": sha256_file(str(ORDERER))}, wes, 4)
    assert len(jobs) == 5 * len(wes) and len({j["job_id"] for j in jobs}) == len(jobs)
    for w, e in wes:
        mine = [j for j in jobs if (j["world"], j["episode"]) == (w, e)]
        assert sorted(j["replicate"] for j in mine) == [0, 1, 2, 3, 4]
        assert all(j["arm"] == "A0" and j["theta"] == [] and j["action_seed"] is None
                   for j in mine)
        keys = {DrawKey(w, e, j["replicate"]) for j in mine}
        assert len(keys) == 5
        us = {keyed_uniform(k, "order_c0__1", 3, 0) for k in keys}
        assert len(us) == 5
    with pytest.raises(ValueError, match="replicates"):
        es.noise_jobs(inp, {"path": "x", "sha256": "x"}, wes, 0)


@needs_inputs
def test_dry_run_r0_reproduces_the_bar_and_outputs_recompute(bar_dir: Path,  # noqa: F811
                                                             probe_dir: Path,
                                                             tmp_path: Path) -> None:
    """1 world-episode, real A0 episodes at the bar's horizon (12): r0 equals the bar's A0 row,
    every file is written, `--recompute` MATCHes, a tampered check2.json MISMATCHes, and a
    resume plays nothing."""
    out = tmp_path / "nf"
    code = es.run_noise_floor(bar_dir=bar_dir, probe_dir=probe_dir, out_dir=out,
                              orderer_ckpt=str(ORDERER), replicates=2,
                              probe_episodes=probe_dir / "probe_episodes.csv",
                              max_world_episodes=1, dry_run=True, log=QUIET)
    assert code == 0
    for f in ("spec.json", "contract_failures.csv", *es.NOISE_FLOOR_FILES):
        assert (out / f).exists(), f
    rows = list(csv.DictReader((out / "noise_floor.csv").open()))
    assert sorted(int(r["replicate"]) for r in rows) == [0, 1, 2]
    repro = list(csv.DictReader((out / "reproduction.csv").open()))
    assert len(repro) == 1 and repro[0]["equal"] == "True"
    assert json.loads((out / "spec.json").read_text())["result"]["r0_reproduced"] == 1
    c2 = json.loads((out / "check2.json").read_text())
    assert c2["episodes"] == 2  # r1, r2 on one world-episode
    assert es.run_noise_floor(bar_dir=bar_dir, probe_dir=probe_dir, out_dir=out,
                              orderer_ckpt=str(ORDERER), replicates=2,
                              probe_episodes=probe_dir / "probe_episodes.csv",
                              recompute=True, log=QUIET) == 0
    before = (out / "noise_floor.csv").read_text()
    assert es.run_noise_floor(bar_dir=bar_dir, probe_dir=probe_dir, out_dir=out,
                              orderer_ckpt=str(ORDERER), replicates=2,
                              probe_episodes=probe_dir / "probe_episodes.csv",
                              max_world_episodes=1, dry_run=True,
                              run_job_fn=lambda j: pytest.fail("a resume played a job"),
                              log=QUIET) == 0
    assert (out / "noise_floor.csv").read_text() == before
    doc = json.loads((out / "check2.json").read_text())
    doc["rate"] = 0.123
    (out / "check2.json").write_text(json.dumps(doc))
    assert es.run_noise_floor(bar_dir=bar_dir, probe_dir=probe_dir, out_dir=out,
                              orderer_ckpt=str(ORDERER), replicates=2,
                              probe_episodes=probe_dir / "probe_episodes.csv",
                              recompute=True, log=QUIET) == es.CONTRACT_INVALID


@needs_inputs
def test_r0_off_the_bar_is_a_contract_failure(bar_dir: Path, probe_dir: Path,  # noqa: F811
                                              tmp_path: Path) -> None:
    """A replayed r0 that differs from the bar's A0 row makes the run invalid (exit 2)."""
    fake = _fake(bar_dir)

    def off(job: dict) -> dict[str, str]:
        d = fake(job)
        if job["replicate"] == 0:
            d["units"] = repr(float(d["units"]) + 1.0)
        return d

    out = tmp_path / "off"
    assert es.run_noise_floor(bar_dir=bar_dir, probe_dir=probe_dir, out_dir=out,
                              orderer_ckpt=str(ORDERER), replicates=1, dry_run=True,
                              run_job_fn=off, log=QUIET) == es.CONTRACT_INVALID
    fails = list(csv.DictReader((out / "contract_failures.csv").open()))
    assert fails and {f["check"] for f in fails} == {es.A0_REPRODUCTION}
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"


def _faithful(bar_dir: Path):  # noqa: ANN202 - a job function
    """`_fake`, with A0's death and holding taken from the bar row, so r0 reproduces it."""
    fake = _fake(bar_dir)
    with (bar_dir / "bar.csv").open(newline="") as fh:
        ref = {(d["world"], int(d["episode"])): d for d in csv.DictReader(fh)}

    def run(job: dict) -> dict[str, str]:
        d = fake(job)
        b = ref[(job["world"], int(job["episode"]))]
        d.update({"dead": b["a0_dead"].lower(),
                  "holding_units_per_cycle": b["a0_holding_units_per_cycle"]})
        return d
    return run


def _bar(**kw: object) -> es.BarRow:
    d = dict(world="w", episode=1, bar=100.0, demand_per_cycle=10.0, a0_units=60.0,
             a0_dead=False, a0_rundown=False, a0_starved=frozenset(), demand_digest="d",
             shock_digest="s")
    d.update(kw)
    return es.BarRow(**d)  # type: ignore[arg-type]


def _a0(world: str, episode: int, r: int, units: float, *, r_end: float = 0.9,
        fills: dict | None = None, dead: bool = False) -> dict[str, str]:
    return {"world": world, "episode": str(episode), "replicate": str(r), "units": repr(units),
            "dead": "true" if dead else "false", "r_start": "1.0", "r_end": repr(r_end),
            "consumer_fill_rate": json.dumps(fills or {"c0": 0.9}),
            "holding_units_per_cycle": "0.0"}


def test_check2_and_check3_arithmetic() -> None:
    """Hand fixture, 2 world-episodes (one per minibatch), A0 at r0..r2.
    Check 2: r1 on w1 runs down where the bar row (r0) does not -> 1 of 4 replicate episodes.
    Check 3: fitness = units / 100. w1: 0.60, 0.70, 0.80 -> SD 0.1 -> noise sqrt(2) x 0.1;
    w2 (minibatch 2): 0.50, 0.50, 0.56 -> SD 0.034641. A pair difference 0.2 on minibatch 1 has
    ratio 0.2 / (sqrt 2 x 0.1) = 1.414 (above 1x, not 2x)."""
    bar = {("w1", 1): _bar(world="w1"), ("w2", 2): _bar(world="w2", episode=2)}
    y = ow.Y_STARVATION
    rows = [_a0("w1", 1, 0, 60.0), _a0("w1", 1, 1, 70.0, r_end=0.5), _a0("w1", 1, 2, 80.0),
            _a0("w2", 2, 0, 50.0), _a0("w2", 2, 1, 50.0, fills={"c0": y - 0.01}),
            _a0("w2", 2, 2, 56.0)]
    fitness = [{"arm": "A1", "batch": "0", "member": "+00", "fitness": "0.9"},
               {"arm": "A1", "batch": "0", "member": "-00", "fitness": "0.7"},
               {"arm": "A1", "batch": "1", "member": "+00", "fitness": "0.5"},
               {"arm": "A1", "batch": "1", "member": "-00", "fitness": "0.51"}]
    c2, c3rows, c3 = es.noise_floor_readings(rows, bar, [[("w1", 1)], [("w2", 2)]], fitness,
                                             arms=("A1",), pairs=1)
    assert c2["episodes"] == 4 and c2["self_disqualified"] == 2 and c2["rate"] == 0.5
    assert c2["by_flag"] == {"rundown": 1, "starvation": 1}
    assert c2["by_world_episode"] == {"w1:1": {"replicates": 2, "self_disqualified": 1},
                                      "w2:2": {"replicates": 2, "self_disqualified": 1}}
    m1 = next(r for r in c3rows if r["batch"] == 0)
    assert m1["a0_sd"] == pytest.approx(0.1) and m1["a0_replicates"] == 3
    assert m1["diff_noise"] == pytest.approx(math.sqrt(2) * 0.1)
    assert m1["ratio"] == pytest.approx(0.2 / (math.sqrt(2) * 0.1))
    assert m1["above_1x"] is True and m1["above_2x"] is False
    m2 = next(r for r in c3rows if r["batch"] == 1)
    assert m2["a0_sd"] == pytest.approx(float(np.std([0.5, 0.5, 0.56], ddof=1)))
    assert m2["d"] == pytest.approx(-0.01) and m2["above_1x"] is False
    s = c3["summary"]["minibatch/A1"]
    assert s["n"] == 2 and s["share_above_1x"] == 0.5 and s["share_above_2x"] == 0.0
    # the per world-episode level reads probe_episodes rows (member fitness per world-episode)
    pe = [{"arm": "A1", "batch": "0", "member": "+00", "world": "w1", "episode": "1",
           "units": "75.0", "holding_units_per_cycle": "0.0"},
          {"arm": "A1", "batch": "0", "member": "-00", "world": "w1", "episode": "1",
           "units": "70.0", "holding_units_per_cycle": "0.0"}]
    _, rows2, c3b = es.noise_floor_readings(rows, bar, [[("w1", 1)], [("w2", 2)]], fitness, pe,
                                            arms=("A1",), pairs=1)
    we = [r for r in rows2 if r["level"] == "world_episode"]
    assert len(we) == 1 and we[0]["d"] == pytest.approx(0.05)
    assert we[0]["ratio"] == pytest.approx(0.05 / (math.sqrt(2) * 0.1))
    assert c3b["a0_sd"]["world_episode"]["w1:1"] == pytest.approx(0.1)


@needs_inputs
def test_pin_mismatch_exits_2(bar_dir: Path, probe_dir: Path, tmp_path: Path,  # noqa: F811
                              monkeypatch: pytest.MonkeyPatch) -> None:
    """A real run on anything but EXP-0058's bar and probe files exits 2 before any job, each
    pin named; with every pin matched, the registered K and no filter, nothing is refused; a
    resume with other pins also exits 2."""
    never = lambda j: pytest.fail("a refused run played a job")  # noqa: E731
    kw = dict(bar_dir=bar_dir, probe_dir=probe_dir, orderer_ckpt=str(ORDERER),
              probe_episodes=probe_dir / "probe_episodes.csv", run_job_fn=never, log=QUIET)
    assert es.run_noise_floor(out_dir=tmp_path / "a", source_commit="a" * 40,
                              **kw) == es.CONTRACT_INVALID
    inp = es.load_inputs(bar_dir)
    shas = {"probe_fitness_sha256": sha256_file(str(probe_dir / "probe_fitness.csv")),
            "probe_json_sha256": sha256_file(str(probe_dir / "probe.json")),
            "probe_episodes_sha256": sha256_file(str(probe_dir / "probe_episodes.csv"))}
    good = dict(inp=inp, source_commit="a" * 40, orderer_sha=sha256_file(str(ORDERER)),
                probe_shas=shas, replicates=es.NOISE_FLOOR_K,
                filters={"horizon": None, "max_world_episodes": None})
    text = "; ".join(es.noise_floor_problems(**good))
    for needle in ("dry run", "normaliser", "probe_fitness", "probe_json", "probe_episodes"):
        assert needle in text, needle
    # register the fixture's own files: only the dry-run bar (and its knock-ons) remain
    monkeypatch.setattr(es, "EXP0058_PINS", {"normaliser_sha256": inp.normaliser.sha256, **shas})
    text = "; ".join(es.noise_floor_problems(**good))
    assert "normaliser " not in text and "probe_" not in text and "dry run" in text
    assert "--replicates 3 is not the registered 4" in "; ".join(
        es.noise_floor_problems(**{**good, "replicates": 3}))
    assert "--horizon is set" in "; ".join(es.noise_floor_problems(
        **{**good, "filters": {"horizon": 12, "max_world_episodes": None}}))
    assert "40-hex" in "; ".join(es.noise_floor_problems(**{**good, "source_commit": ""}))
    # a resume onto another run's pins
    out = tmp_path / "r"
    assert es.run_noise_floor(out_dir=out, replicates=1, dry_run=True,
                              **{**kw, "run_job_fn": _faithful(bar_dir)}) == 0
    assert es.run_noise_floor(out_dir=out, replicates=2, dry_run=True,
                              **kw) == es.CONTRACT_INVALID


@needs_inputs
def test_cli_mode_and_refusals(bar_dir: Path, probe_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    """The CLI's noise-floor dry run (1 world-episode, horizon 12) exits 0; noise-floor flags
    in another mode are refused; noise-floor without --probe-dir is refused."""
    out = tmp_path / "cli"
    args = ["--mode", "noise-floor", "--bar-dir", str(bar_dir), "--orderer-ckpt", str(ORDERER),
            "--probe-dir", str(probe_dir), "--probe-episodes",
            str(probe_dir / "probe_episodes.csv"), "--replicates", "1",
            "--max-world-episodes", "1", "--horizon", "12", "--dry-run", "--workers", "1",
            "--out-dir", str(out)]
    assert es.main(args) == 0
    assert (out / "check2.json").exists()
    for bad in (["--mode", "train", "--arm", "A1", "--generations", "1", "--bar-dir",
                 str(bar_dir), "--orderer-ckpt", str(ORDERER), "--probe-dir", str(probe_dir),
                 "--out-dir", str(tmp_path / "x")],
                ["--mode", "noise-floor", "--bar-dir", str(bar_dir), "--orderer-ckpt",
                 str(ORDERER), "--out-dir", str(tmp_path / "y")]):
        with pytest.raises(SystemExit):
            es.main(bad)
