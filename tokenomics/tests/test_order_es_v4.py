"""Task 7.8: the noise floor re-measured on a given bar's A0 (`--mode noise-floor --base mww`, no
probe files) and pre-registration v4 = v3 with that margin, computed by code from the run's
rows (frozen at 7.9b after the run); v1-v3 untouched."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from test_order_es_v3 import QUIET, needs_inputs, v3_inputs  # noqa: F401
from test_order_es_validate import ORDERER, inputs  # noqa: F401
from tokenomics.eval.provenance import sha256_file
from tokenomics.train import order_es as es
from tokenomics.train import order_es_validate as v
from tokenomics.train.per_order_scorer import BASE_MWW, BASE_TS

REPO = Path(__file__).resolve().parents[2]
EXP0068_ROWS = REPO / "experiments/EXP-0068-es-noise-floor/noise_floor.csv"
EXP0058_BAR = REPO / "experiments/EXP-0058-es-gradient-probe/bar/bar.csv"
needs_0068 = pytest.mark.skipif(not (EXP0068_ROWS.exists() and EXP0058_BAR.exists()),
                                reason="EXP-0068 / EXP-0058 records absent")


def _rows(p: Path) -> list[dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


@needs_0068
def test_margins_doc_is_the_registered_formula() -> None:
    """On EXP-0068's rows the code reproduces v2's registered margin and its self-flag counts
    (16 -> 0 of 48); A0's guards from its own r0 rows equal the bar's A0 columns there."""
    rows = _rows(EXP0068_ROWS)
    doc = es.margins_doc(rows)
    assert (doc["margin"]["rundown"], doc["margin"]["starvation"]) == (
        es.NOISE_MARGIN_V2["rundown"], es.NOISE_MARGIN_V2["starvation"])
    assert doc["self_flags"] == {"v1_rule": 16, "margin": 0, "episodes": 48}
    assert doc["target"]["self_flags_max"] == 2 and doc["target"]["met"]
    assert doc["lowo"]["flagged"] == 0
    bar = es.read_bar(EXP0058_BAR)
    r0 = es.r0_bar(rows)
    m = es.NoiseMargin(**es.NOISE_MARGIN_V2)
    for margin in (None, m):
        assert es.self_flags(rows, r0, margin) == es.self_flags(rows, bar, margin)
    for k, b in r0.items():
        assert (b.a0_dead, b.a0_rundown, b.a0_starved) == (
            bar[k].a0_dead, bar[k].a0_rundown, bar[k].a0_starved)


@needs_inputs
def test_noise_floor_on_a_given_bar(v3_inputs: dict[str, Path],  # noqa: F811
                                    inputs: dict[str, Path], tmp_path: Path) -> None:  # noqa: F811
    """A0 = MWWk re-keyed on the MWW bar's own world-episodes: r0 reproduces the bar's A0 rows,
    the jobs play the MWW base, margins.json recomputes offline; no probe files are pinned; the
    other base, or a tampered margins.json, is refused."""
    out = tmp_path / "nf"
    seen: list[dict] = []

    def spy(job: dict) -> dict[str, str]:
        seen.append(job)
        return es.run_es_job(job)

    assert es.run_noise_floor_bar(bar_dir=v3_inputs["bar"], out_dir=out,
                                  orderer_ckpt=str(ORDERER), base_kind=BASE_MWW, dry_run=True,
                                  run_job_fn=spy, log=QUIET) == 0
    inp = es.load_inputs(v3_inputs["bar"])
    assert len(seen) == len(inp.world_episodes) * (es.NOISE_FLOOR_K + 1)
    assert {j["base_kind"] for j in seen} == {BASE_MWW} and {j["arm"] for j in seen} == {"A0"}
    repro = _rows(out / "reproduction.csv")
    assert len(repro) == len(inp.world_episodes) and all(r["equal"] == "True" for r in repro)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["set"] == "bar" and spec["pins"]["base_kind"] == BASE_MWW
    assert not any("probe" in k for k in spec["pins"]["inputs"])
    doc = json.loads((out / "margins.json").read_text())
    assert doc["self_flags"]["episodes"] == len(inp.world_episodes) * es.NOISE_FLOOR_K
    assert es.run_noise_floor_bar(bar_dir=v3_inputs["bar"], out_dir=out,
                                  orderer_ckpt=str(ORDERER), base_kind=BASE_MWW,
                                  recompute=True, log=QUIET) == 0
    (out / "margins.json").write_text((out / "margins.json").read_text().replace("1", "2", 1))
    assert es.run_noise_floor_bar(bar_dir=v3_inputs["bar"], out_dir=out,
                                  orderer_ckpt=str(ORDERER), base_kind=BASE_MWW,
                                  recompute=True, log=QUIET) == es.CONTRACT_INVALID
    with pytest.raises(ValueError, match="plays the 'mww' base"):
        es.run_noise_floor_bar(bar_dir=v3_inputs["bar"], out_dir=tmp_path / "x",
                               orderer_ckpt=str(ORDERER), base_kind=BASE_TS, dry_run=True,
                               log=QUIET)
    with pytest.raises(ValueError, match="plays the 'ts' base"):
        es.run_noise_floor_bar(bar_dir=inputs["bar"], out_dir=tmp_path / "y",
                               orderer_ckpt=str(ORDERER), base_kind=BASE_MWW, dry_run=True,
                               log=QUIET)
    # a real run on a dry-run bar is refused before any job
    assert es.run_noise_floor_bar(bar_dir=v3_inputs["bar"], out_dir=tmp_path / "z",
                                  orderer_ckpt=str(ORDERER), base_kind=BASE_MWW,
                                  log=QUIET) == es.CONTRACT_INVALID


@needs_0068
def test_v4_freezes_from_the_staged_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                         ) -> None:
    """v4 cannot be built before the rows exist; the freeze stages the run's CSV into the
    package, computes the margin by code (the file's SHA recorded), never replaces a staged
    file with other rows, and refuses a margin that misses the target; v1-v3 still MATCH."""
    staged = tmp_path / "es_noise_floor_m4_v4.csv"
    monkeypatch.setattr(es, "V4_NOISE_FLOOR_PATH", staged)
    es._V4_CACHE.clear()
    with pytest.raises(ValueError, match="frozen by task 7.9b"):
        es.registered_sections(4)
    out = tmp_path / "es_prereg_m4_v4.json"
    assert v.run_freeze(out, log=QUIET, version=4, noise_floor=EXP0068_ROWS) == 0
    assert sha256_file(str(staged)) == sha256_file(str(EXP0068_ROWS))
    doc = es.load_prereg(out)
    m = doc["es"]["disqualification"]["noise_margin"]
    assert (m["rundown"], m["starvation"]) == (es.NOISE_MARGIN_V2["rundown"],
                                               es.NOISE_MARGIN_V2["starvation"])
    assert m["source"]["sha256"] == sha256_file(str(EXP0068_ROWS)) and m["target"]["met"]
    assert es.prereg_version(doc) == 4 and es.run_rule(doc) == (
        4, es.NoiseMargin(m["rundown"], m["starvation"]))
    assert es.base_kind_for(4) == BASE_MWW
    assert {k: doc[k] for k in es.registered_sections(4)} == es.registered_sections(4)
    d3 = es.load_prereg(v.PREREG_V3_PATH)
    assert doc["readings"] == d3["readings"]
    assert {k: x for k, x in doc["es"].items() if k != "disqualification"} == {
        k: x for k, x in d3["es"].items() if k != "disqualification"}
    assert doc["validation"] == d3["validation"] and doc["probe"] == d3["probe"]
    assert v.run_freeze(out, log=QUIET, version=4) == 0  # MATCH
    assert es.prereg_file_problems(out)  # not pinned yet (PREREG_V4_SHA256 empty until 7.9b)
    other = tmp_path / "other.csv"
    other.write_text(EXP0068_ROWS.read_text().replace("\n", "\n", 1) + "\n")
    assert v.stage_v4_noise_floor(other, log=QUIET) == es.CONTRACT_INVALID
    real = es.v4_margin_registration
    monkeypatch.setattr(es, "v4_margin_registration",
                        lambda path=None: {**real(path), "target": {
                            **real(path)["target"], "met": False}})
    assert v.run_freeze(tmp_path / "v4b.json", log=QUIET, version=4) == es.CONTRACT_INVALID
    for ver, path in ((1, v.PREREG_PATH), (2, v.PREREG_V2_PATH), (3, v.PREREG_V3_PATH)):
        assert v.run_freeze(path, log=QUIET, version=ver) == 0
    es._V4_CACHE.clear()


def test_packaged_v4_is_frozen_from_exp0071() -> None:
    """Task 7.9b: the packaged v4 freezes to MATCH, is pinned, and its margin is EXP-0071's
    (staged rows, SHA recorded), meeting the <= 5% target; v1-v3 MATCH."""
    for ver, path, sha in ((1, v.PREREG_PATH, v.PREREG_SHA256),
                           (2, v.PREREG_V2_PATH, v.PREREG_V2_SHA256),
                           (3, v.PREREG_V3_PATH, v.PREREG_V3_SHA256),
                           (4, v.PREREG_V4_PATH, v.PREREG_V4_SHA256)):
        assert v.run_freeze(path, log=QUIET, version=ver) == 0
        assert sha256_file(str(path)) == sha and v.FROZEN_PREREGS[sha] == ver
    doc = es.load_prereg(v.PREREG_V4_PATH)
    m = doc["es"]["disqualification"]["noise_margin"]
    assert m["source"]["sha256"] == sha256_file(str(es.V4_NOISE_FLOOR_PATH))
    assert m["target"]["met"] and m["target"]["measured"] <= m["target"]["self_flags_max"]
    assert es.prereg_file_problems(v.PREREG_V4_PATH) == []
    assert es.prereg_problems(doc, mode=es.TRAIN, arm="A1", opt_seed=0, generations=40,
                              minibatch=24) == []
    assert es.run_rule(doc) == (4, es.NoiseMargin(m["rundown"], m["starvation"]))
    assert v.prereg_read_problems(doc) == []
