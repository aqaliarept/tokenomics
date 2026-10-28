"""Versioned screen outputs and version pairing (add-shocks-i-drift task 5.5, design D8 + D10
"Versioned screen outputs"; gate-harness "Mixed versions refused").

A screen takes its output tag from the input registry's name. From v2 on, the frozen registry,
roster and side-effect parameters all record `screened_on` = the hash of the registry draft the
screen read, and a real report admits a v2+ set only when the three agree. The screens here are
injected-row dry runs: a stub returns synthetic roster rows, no episode is played."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tokenomics.eval.gate import (
    DEFAULT_PARAMS,
    EpisodeRow,
    SideEffectParams,
    load_side_effect_params,
    params_from_dict,
)
from tokenomics.eval.gate_cli import (
    Job,
    row_to_dict,
    run_report,
    run_screen,
    screen_outcome,
    version_set_problem,
)
from tokenomics.eval.registry import (
    DEFAULT_REGISTRY,
    DEVELOPMENT,
    REGISTRY_V2,
    Registry,
    load_registry,
    registry_from_dict,
)
from tokenomics.eval.roster import (
    DEFAULT_ROSTER,
    ROSTER_V2,
    Roster,
    build_roster_v1,
    content_hash,
    freeze_roster,
    load_roster,
    name_version_tag,
    roster_from_dict,
)

# The packaged registry v2 draft (task 5.1) the screen reads; its hash is every v2 file's
# `screened_on`.
DRAFT_V2 = "3032147f1f17233703f486c66e4c56e006c026399364dd1cbaead5b46bcfceb5"
NAMES = ("worlds_v2.json", "roster_v2.json", "side_effect_params_v2.json")
SUBJECT = "baseline:entity_planner/blend_0.5"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _stub(roster: Roster):  # -> Callable[[Job], dict[str, str]]
    """Synthetic roster rows: member k (by sorted id) delivers 100 - k units at fill 0.5."""
    ids = sorted(roster.member_ids)

    def stub(job: Job) -> dict[str, str]:
        k = ids.index(job["subject"])
        return row_to_dict(EpisodeRow(
            world=job["world"], size=job["size"], regime=job["regime"], episode=job["episode"],
            subject=job["subject"], replicate=job["replicate"], units=100.0 - k, dead=False,
            demand_digest=f"d{job['episode']}", r_start=1.0, r_end=0.9 - k / 100,
            consumer_fill_rate={"c0": 0.8 - k / 100}, ordered_units=100, fulfilled_units=50))

    return stub


def _v2_screen(registry: Registry, out: Path) -> Path:
    roster = load_roster(ROSTER_V2)
    _, status = run_screen(out_dir=out, horizon=12, workers=1, registry=registry, roster=roster,
                           episodes_per_world=1, dry_run=True, run_job_fn=_stub(roster),
                           log=lambda s: None)
    assert status == 0
    return out / "frozen"


@pytest.fixture(scope="module")
def v2_screen(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One injected-row v2 screen of the packaged registry v2 / roster v2 drafts."""
    return _v2_screen(load_registry(REGISTRY_V2), tmp_path_factory.mktemp("v2") / "screen")


@pytest.fixture(scope="module")
def other_screen(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A second v2 screen of a different draft: same worlds, another saturation ceiling, hence
    another draft hash (`screened_on`)."""
    draft = dataclasses.replace(load_registry(REGISTRY_V2), saturation_ceiling=0.94)
    assert draft.version != DRAFT_V2
    return _v2_screen(draft, tmp_path_factory.mktemp("v2b") / "screen")


def _frozen_set(frozen: Path) -> tuple[Registry, Roster, SideEffectParams]:
    return (load_registry(frozen / NAMES[0]), load_roster(frozen / NAMES[1]),
            load_side_effect_params(frozen / NAMES[2]))


# --- the v2 screen writes tagged files that record screened_on ---


def test_drafts_record_no_screened_on() -> None:
    for path in (REGISTRY_V2, ROSTER_V2):
        assert "screened_on" not in json.loads(path.read_text()), path
    assert load_registry(REGISTRY_V2).version == DRAFT_V2
    assert load_registry(REGISTRY_V2).screened_on is None
    assert load_roster(ROSTER_V2).screened_on is None


def test_v2_screen_writes_v2_files_with_screened_on(v2_screen: Path) -> None:
    assert sorted(p.name for p in v2_screen.iterdir()) == sorted(NAMES)
    raw = [json.loads((v2_screen / n).read_text()) for n in NAMES]
    assert [d["name"] for d in raw] == ["worlds-v2-dry-run", "roster-v2", "side-effect-params-v2"]
    assert [d["screened_on"] for d in raw] == [DRAFT_V2] * 3
    reg, roster, params = _frozen_set(v2_screen)
    assert (reg.version_tag, roster.version_tag, name_version_tag(params.name)) == (2, 2, 2)
    assert reg.screened_on == roster.screened_on == params.screened_on == DRAFT_V2
    assert reg.screening is not None and roster.frozen
    # the frozen registry's own hash is not the draft's: screening wrote its record into it
    assert reg.version != DRAFT_V2
    spec = json.loads((v2_screen.parent / "spec.json").read_text())
    assert spec["frozen"] == {NAMES[0]: reg.version, NAMES[1]: roster.version,
                              NAMES[2]: params.version}
    assert spec["screened_on"] == DRAFT_V2
    # the parameters' evidence names exactly the frozen files written next to them
    evidence = raw[2]["evidence"]
    assert (evidence["registry_version"], evidence["roster_version"]) == (reg.version,
                                                                          roster.version)


def test_v2_screen_names_an_untagged_roster_roster_v2(tmp_path: Path) -> None:
    """The frozen roster of a v2 screen is named `roster-v2` whatever the input roster's name (a
    dry run may screen an ad-hoc roster), so it can carry `screened_on`."""
    roster = Roster(name="t", replicates=1, members=load_roster(ROSTER_V2).members[:2])
    out = tmp_path / "s"
    _, status = run_screen(out_dir=out, horizon=12, workers=1, registry=load_registry(REGISTRY_V2),
                           roster=roster, episodes_per_world=1, dry_run=True,
                           run_job_fn=_stub(roster), log=lambda s: None)
    assert status == 0
    frozen = load_roster(out / "frozen" / NAMES[1])
    assert (frozen.name, frozen.screened_on, frozen.member_ids) == (
        "roster-v2", DRAFT_V2, roster.member_ids)


def test_screen_outcome_records_screened_on_exactly_from_v2() -> None:
    with pytest.raises(ValueError, match="screened_on is recorded from tag v2 on"):
        screen_outcome(load_registry(REGISTRY_V2), load_roster(ROSTER_V2), [], q=0.05,
                       prune=False, tag=2)
    with pytest.raises(ValueError, match="screened_on is recorded from tag v2 on"):
        screen_outcome(load_registry(), load_roster(), [], q=0.05, prune=False,
                       screened_on=DRAFT_V2)


def test_real_v2_screen_refuses_a_v1_roster(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registry 'worlds-v2' is v2 but roster 'roster-v1' is v1"):
        run_screen(out_dir=tmp_path / "s", horizon=12, workers=1,
                   registry=load_registry(REGISTRY_V2), roster=build_roster_v1(),
                   source_commit=COMMIT, run_job_fn=_stub(build_roster_v1()), log=lambda s: None)
    assert not (tmp_path / "s").exists()


# --- admission: a real report pairs only one screen's files ---


class Admitted(BaseException):
    """Raised by the first job of a real report: everything that can refuse already passed."""


def _first_job(job: Job) -> dict[str, str]:
    raise Admitted(job["world"])


def _real_report(out: Path, registry: Registry, roster: Roster, params_path: Path) -> None:
    run_report(out_dir=out, subject=SUBJECT, split=DEVELOPMENT, horizon=12, workers=1,
               registry=registry, roster=roster, params=load_side_effect_params(params_path),
               params_payload=json.loads(params_path.read_text()), source_commit=COMMIT,
               run_job_fn=_first_job, log=lambda s: None)


def test_real_report_admits_the_set_one_v2_screen_wrote(v2_screen: Path, tmp_path: Path) -> None:
    reg, roster, params = _frozen_set(v2_screen)
    assert version_set_problem(reg, roster, params) is None
    assert reg.admission_problems(DEVELOPMENT) == []
    with pytest.raises(Admitted):
        _real_report(tmp_path / "r", reg, roster, v2_screen / NAMES[2])


def test_v1_set_still_pairs() -> None:
    assert version_set_problem(load_registry(DEFAULT_REGISTRY), load_roster(DEFAULT_ROSTER),
                               load_side_effect_params(DEFAULT_PARAMS)) is None


@pytest.mark.parametrize("swap,named", [
    ("roster-v1", "roster 'roster-v1'"),
    ("params-v1", "parameters 'side-effect-params-v1'"),
    ("other-roster", "roster 'roster-v2'"),
    ("other-params", "parameters 'side-effect-params-v2'"),
])
def test_mixed_versions_refused(v2_screen: Path, other_screen: Path, tmp_path: Path, swap: str,
                                named: str) -> None:
    # Scenario "Mixed versions refused": registry v2 with roster v1, with parameters v1, and with
    # a roster (or parameters) v2 from a second screen of a different draft.
    reg, roster = load_registry(v2_screen / NAMES[0]), load_roster(v2_screen / NAMES[1])
    params_path = v2_screen / NAMES[2]
    if swap == "roster-v1":
        roster = load_roster(DEFAULT_ROSTER)
    elif swap == "params-v1":
        params_path = DEFAULT_PARAMS
    elif swap == "other-roster":
        roster = load_roster(other_screen / NAMES[1])
    else:
        params_path = other_screen / NAMES[2]
    problem = version_set_problem(reg, roster, load_side_effect_params(params_path))
    assert problem is not None and problem.startswith("mixed versions: " + named)
    with pytest.raises(ValueError, match="mixed versions: " + named):
        _real_report(tmp_path / "r", reg, roster, params_path)
    assert not (tmp_path / "r").exists()  # refused before any file is touched


def test_v1_registry_refuses_a_v2_roster(v2_screen: Path) -> None:
    problem = version_set_problem(load_registry(DEFAULT_REGISTRY), load_roster(v2_screen / NAMES[1]),
                                  load_side_effect_params(DEFAULT_PARAMS))
    assert problem is not None and "roster 'roster-v2'" in problem


def test_v2_registry_without_screened_on_refused_at_admission(v2_screen: Path) -> None:
    reg, roster, params = _frozen_set(v2_screen)
    bare = dataclasses.replace(reg, screened_on=None)  # in memory: the parser would refuse it
    problem = version_set_problem(bare, roster, params)
    assert problem is not None and "records no screened_on" in problem


def test_dry_run_report_skips_the_pairing_check(v2_screen: Path, tmp_path: Path) -> None:
    """Pairing gates real reports only; a smoke run may mix versions (marked not an acceptance
    read), exactly as it may use an unfrozen roster."""
    reg = load_registry(v2_screen / NAMES[0])
    with pytest.raises(Admitted):
        run_report(out_dir=tmp_path / "d", subject=SUBJECT, split=DEVELOPMENT, horizon=12,
                   workers=1, registry=reg, roster=load_roster(DEFAULT_ROSTER),
                   params=load_side_effect_params(DEFAULT_PARAMS),
                   params_payload=json.loads(DEFAULT_PARAMS.read_text()), dry_run=True,
                   max_worlds=1, episodes_per_world=1,
                   run_job_fn=_first_job, log=lambda s: None)


# --- parsers: screened_on optional on v1 files and drafts, required on frozen v2+ files ---


def _rehashed(d: dict) -> dict:
    d = {k: v for k, v in d.items() if k != "content_hash"}
    return d | {"content_hash": content_hash(d)}


@pytest.mark.parametrize("name,parse", [
    (NAMES[0], registry_from_dict), (NAMES[1], roster_from_dict), (NAMES[2], params_from_dict),
])
def test_frozen_v2_file_without_screened_on_refused(v2_screen: Path, name: str,
                                                    parse: object) -> None:
    d = json.loads((v2_screen / name).read_text())
    del d["screened_on"]
    with pytest.raises(ValueError, match="hash|screened_on"):
        parse(d)  # type: ignore[operator]  # removed without a new hash: edited
    with pytest.raises(ValueError, match="screened_on"):
        parse(_rehashed(d))  # type: ignore[operator]  # re-hashed: still no pairing key


def test_screened_on_only_where_a_v2_screen_writes_it(v2_screen: Path) -> None:
    reg, roster, _ = _frozen_set(v2_screen)
    with pytest.raises(ValueError, match="only on a screened v2\\+ registry"):
        dataclasses.replace(load_registry(REGISTRY_V2), screened_on=DRAFT_V2)  # unscreened draft
    with pytest.raises(ValueError, match="only on a screened v2\\+ registry"):
        dataclasses.replace(reg, name="worlds-v1", screened_on=DRAFT_V2)
    with pytest.raises(ValueError, match="only on a frozen v2\\+ roster"):
        dataclasses.replace(load_roster(ROSTER_V2), screened_on=DRAFT_V2)  # unfrozen draft
    with pytest.raises(ValueError, match="only on a frozen v2\\+ roster"):
        dataclasses.replace(freeze_roster(build_roster_v1()), screened_on=DRAFT_V2)
    with pytest.raises(ValueError, match="fields"):  # v1 parameters never carry it
        params_from_dict(_rehashed(json.loads(DEFAULT_PARAMS.read_text())
                                   | {"screened_on": DRAFT_V2}))
    assert roster.screened_on == DRAFT_V2  # the frozen v2 roster does


def test_name_version_tag() -> None:
    assert [name_version_tag(n) for n in (
        "worlds-v1", "worlds-v2", "worlds-v2-dry-run", "roster-v2", "side-effect-params-v12",
        "fixture", "fixture-v2", "t", "worlds-v2x")] == [1, 2, 2, 2, 12, 1, 2, 1, 1]


# --- review concern 186: screened_on values, not only keys ---


def test_v2_params_with_null_screened_on_refused() -> None:
    """The repro: v1 parameters renamed v2 with a null screened_on and re-hashed used to load and
    pair with the v1 registry and roster."""
    d = json.loads(DEFAULT_PARAMS.read_text()) | {"name": "side-effect-params-v2",
                                                   "screened_on": None}
    with pytest.raises(ValueError, match="screened_on is recorded exactly from v2 on"):
        params_from_dict(_rehashed(d))
    with pytest.raises(ValueError, match="screened_on is recorded exactly from v2 on"):
        SideEffectParams(x=0.5, y=0.5, version="v", name="side-effect-params-v2")
    with pytest.raises(ValueError, match="screened_on is recorded exactly from v2 on"):
        SideEffectParams(x=0.5, y=0.5, version="v", name="side-effect-params-v1",
                         screened_on=DRAFT_V2)


@pytest.mark.parametrize("bad", [None, "", "x" * 64, DRAFT_V2.upper(), DRAFT_V2[:-1], 7])
@pytest.mark.parametrize("name,parse", [
    (NAMES[0], registry_from_dict), (NAMES[1], roster_from_dict), (NAMES[2], params_from_dict),
])
def test_malformed_screened_on_refused(v2_screen: Path, name: str, parse: object,
                                       bad: object) -> None:
    d = _rehashed(json.loads((v2_screen / name).read_text()) | {"screened_on": bad})
    # an explicit null is also not canonical for registry/roster (their payload omits None)
    with pytest.raises(ValueError, match="screened_on|not in canonical form"):
        parse(d)  # type: ignore[operator]


def test_in_memory_v2_files_without_screened_on_refused_at_admission(v2_screen: Path) -> None:
    reg, roster, params = _frozen_set(v2_screen)
    problem = version_set_problem(reg, dataclasses.replace(roster, screened_on=None), params)
    assert problem is not None and "roster 'roster-v2'" in problem and "no screened_on" in problem
    # a v1 set with a v2+ roster that lost its key is refused too, not paired as "all None"
    problem = version_set_problem(load_registry(DEFAULT_REGISTRY),
                                  dataclasses.replace(roster, screened_on=None),
                                  load_side_effect_params(DEFAULT_PARAMS))
    assert problem is not None and "roster 'roster-v2'" in problem


# --- review concern 187: a smoke/probe roster derived from a frozen v2 roster ---


@pytest.mark.parametrize("argv", [
    ["--mode", "report", "--dry-run", "--max-members", "1"],
    ["--mode", "report", "--dry-run", "--replicates", "1"],
    ["--mode", "probe", "--scales", "1.0", "--members", "always_both/blend_0.5"],
], ids=["report-max-members", "report-replicates", "probe-members"])
def test_cli_derives_a_smoke_roster_from_a_frozen_v2_roster(
        v2_screen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    """`--max-members` / `--replicates` / `--members` derive an unfrozen roster from the loaded
    one; from a frozen v2 roster it must drop `screened_on` (an unfrozen roster carries none)
    and reach the driver."""
    import tokenomics.eval.gate_cli as cli

    frozen = load_roster(v2_screen / NAMES[1])
    assert frozen.frozen and frozen.screened_on == DRAFT_V2
    seen: list[Roster] = []
    monkeypatch.setattr(cli, "_run_mode", lambda args, roster, commit, probe: seen.append(roster)
                        or 0)
    assert cli.main([*argv, "--roster", str(v2_screen / NAMES[1]),
                     "--out-dir", str(tmp_path / "o")]) == 0
    (derived,) = seen
    assert not derived.frozen and derived.screened_on is None and derived.pruning is None
    assert derived.name in ("roster-v2-dry-run", "roster-v2-probe")
    assert set(derived.member_ids) <= set(frozen.member_ids)
