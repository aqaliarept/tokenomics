"""The installed frozen v2 set (add-shocks-i-drift task 10.1, design D8).

EXP-0047 screened the registry v2 / roster v2 drafts and froze `worlds_v2.json`,
`roster_v2.json` and `side_effect_params_v2.json`. They are packaged in place of the drafts and
are the gate CLI default; the v1 set stays selectable by path, and `--mode recompute` reads only
the run's `spec.json`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenomics.env.shock_schedule import build_schedule, shock_digest
from tokenomics.eval.gate import (
    DEFAULT_PARAMS,
    PARAMS_V1,
    PARAMS_V2,
    load_side_effect_params,
)
from tokenomics.eval.gate_cli import read_rows, version_set_problem
from tokenomics.eval.registry import (
    DEFAULT_REGISTRY,
    DEVELOPMENT,
    REGISTRY_V1,
    REGISTRY_V2,
    build_worlds_v2,
    load_registry,
)
from tokenomics.eval.roster import (DEFAULT_ROSTER, ROSTER_V1, ROSTER_V2, build_roster_v2,
                                    load_roster)

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
EXP_0047 = EXPERIMENTS / "EXP-0047-registry-v2-calibration"
EXP_0045 = EXPERIMENTS / "EXP-0045-champion-dev-report-12-42"
FROZEN_V2 = {  # EXP-0047 `spec.json` "frozen", run commit ce6b367
    "worlds_v2.json": "7b09f536e9880bc31370694f80eaca828587621eb5daa16fdf64e80ed05ee7ff",
    "roster_v2.json": "5050508ecb25f0eda0ae0a61b0dde4ba948a6e9b0d901cd14cde28e0db161f38",
    "side_effect_params_v2.json": "e2aee97ac38524d709862bc956c43940413ac0cb44c846304a4b8c3d324c3a99",
}
DRAFT_V2 = "3032147f1f17233703f486c66e4c56e006c026399364dd1cbaead5b46bcfceb5"
FROZEN_V1 = {  # EXP-0044 attempt 2, unchanged by the install
    "worlds_v1.json": "566946fe0c0363ec888f9a58bfe7bd480bcc948e049c93e9923088e2cab911e3",
    "roster_v1.json": "c2832dbb44d26601f279fc04c0c483c543c5a5fcdc8cb2231090fc03f76a3731",
    "side_effect_params_v1.json": "e6651b2ebe541b18ade0a8904f27b0a08844e2a4f8ee0e231ca062514472f2c1",
}
PATHS = {"worlds_v2.json": REGISTRY_V2, "roster_v2.json": ROSTER_V2,
         "side_effect_params_v2.json": PARAMS_V2}


def _versions() -> dict[str, str]:
    return {"worlds_v2.json": load_registry(REGISTRY_V2).version,
            "roster_v2.json": load_roster(ROSTER_V2).version,
            "side_effect_params_v2.json": load_side_effect_params(PARAMS_V2).version}


def test_packaged_v2_files_are_exp0047_frozen() -> None:
    assert _versions() == FROZEN_V2
    if (EXP_0047 / "spec.json").exists():  # the EXP dir is not shipped to pods
        spec = json.loads((EXP_0047 / "spec.json").read_text())
        assert spec["frozen"] == FROZEN_V2
        assert spec["screened_on"] == DRAFT_V2
        assert spec["source"]["commit"].startswith("ce6b367")
        for name, path in PATHS.items():
            assert path.read_bytes() == (EXP_0047 / "frozen" / name).read_bytes(), name


def test_v1_files_keep_their_hashes() -> None:
    assert (load_registry(REGISTRY_V1).version, load_roster(ROSTER_V1).version,
            load_side_effect_params(PARAMS_V1).version) == tuple(FROZEN_V1.values())
    assert version_set_problem(load_registry(REGISTRY_V1), load_roster(ROSTER_V1),
                               load_side_effect_params(PARAMS_V1)) is None


def test_v2_set_is_one_screen_of_the_draft() -> None:
    reg, ros = load_registry(REGISTRY_V2), load_roster(ROSTER_V2)
    params = load_side_effect_params(PARAMS_V2)
    assert build_worlds_v2().version == DRAFT_V2
    assert reg.screened_on == ros.screened_on == params.screened_on == DRAFT_V2
    assert version_set_problem(reg, ros, params) is None
    assert (reg.name, ros.name, params.name) == ("worlds-v2", "roster-v2", "side-effect-params-v2")
    assert ros.frozen and set(ros.member_ids) < set(build_roster_v2().member_ids)


def test_registry_v2_validates_and_replays() -> None:
    """The frozen registry loads (hash and validation), admits its development split, keeps every
    draft entry, and every shocked development gate world replays the shock schedule the screen
    ran (EXP-0047 rows, where present)."""
    reg, draft = load_registry(REGISTRY_V2), build_worlds_v2()
    assert reg.screening is not None and reg.admission_problems(DEVELOPMENT) == []
    assert reg.entries == draft.entries
    worlds = {e.id: e for e in reg.gate_worlds(DEVELOPMENT)}
    for e in reg.entries:
        cfg = e.config()
        assert (cfg.shocks is not None) == (e.regime == "shocked"), e.id
    if not (EXP_0047 / "episodes.csv").exists():
        return
    digests = {(r.world, r.episode): r.shock_digest
               for r in read_rows(EXP_0047 / "episodes.csv")}
    shocked = [w for w in worlds.values() if w.regime == "shocked"]
    assert len(shocked) == 8
    for w in shocked:
        cfg = w.config()
        for ep in w.episode_seeds:
            assert shock_digest(build_schedule(cfg, ep, 100)) == digests[(w.id, ep)], (w.id, ep)


def test_gate_cli_defaults_to_the_v2_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tokenomics.eval.gate_cli as cli

    assert (DEFAULT_REGISTRY, DEFAULT_ROSTER, DEFAULT_PARAMS) == (REGISTRY_V2, ROSTER_V2, PARAMS_V2)
    seen: list[tuple] = []
    monkeypatch.setattr(cli, "_run_mode",
                        lambda args, roster, commit, probe: seen.append((args, roster)) or 0)
    v1 = ["--registry", str(REGISTRY_V1), "--roster", str(ROSTER_V1), "--params", str(PARAMS_V1)]
    for extra in ([], v1):
        assert cli.main(["--mode", "report", "--out-dir", str(tmp_path / "o"), *extra]) == 0
    (default, default_roster), (picked, picked_roster) = seen
    assert (default.registry, default.roster, default.params) == tuple(
        str(p) for p in (REGISTRY_V2, ROSTER_V2, PARAMS_V2))
    assert default_roster.version == FROZEN_V2["roster_v2.json"]
    assert (picked.registry, picked.params) == (str(REGISTRY_V1), str(PARAMS_V1))
    assert picked_roster.version == FROZEN_V1["roster_v1.json"]


@pytest.mark.skipif(not (EXP_0045 / "episodes.csv").exists(), reason="EXP-0045 record absent")
def test_v1_recompute_does_not_depend_on_the_default(capsys: pytest.CaptureFixture[str]) -> None:
    """EXP-0045 (a v1 report) recomputes to MATCH through the CLI with the v2 defaults in place:
    recompute reads only the run's spec.json."""
    import tokenomics.eval.gate_cli as cli

    assert cli.main(["--mode", "recompute", "--out-dir", str(EXP_0045)]) == 0
    out = capsys.readouterr().out
    assert "; MATCH with" in out and "MISMATCH" not in out
    assert not (EXP_0045 / "resilience.csv").exists()  # read-only record, nothing written


def test_write_draft_never_overwrites_a_packaged_screen(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path) -> None:
    from tokenomics.eval import registry

    before = {p: p.read_bytes() for p in (REGISTRY_V1, REGISTRY_V2)}
    for version in ("1", "2"):
        monkeypatch.setattr("sys.argv", ["registry", "--write-draft", "--draft-version", version])
        with pytest.raises(SystemExit) as exc:
            registry.main()
        assert exc.value.code == 2
    assert {p: p.read_bytes() for p in before} == before
    out = tmp_path / "draft.json"
    monkeypatch.setattr("sys.argv", ["registry", "--write-draft", "--draft-version", "2",
                                     "--out", str(out)])
    registry.main()
    assert load_registry(out).version == DRAFT_V2
