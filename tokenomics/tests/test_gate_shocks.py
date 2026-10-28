"""Gate harness on worlds with shocks (add-shocks-i-drift task 5.2, design D7/D10; gate-harness
"Shock digest mismatch", "Twin shares the demand", "Resilience rows in a report", "Resilience is
not an acceptance clause").

Fixture registry: three development 12-producer worlds, two calm and one shocked (the regime
balance rule needs 2 calm per shocked gate world). The shocked world names the test-only
`shocks-i-smoke` profile (D7: "shocks-i" with start windows at cycles 2-8), so every resolved
event fires inside horizon 12. The report runs 2 of them (`--families sparse_wide`: one calm, one
shocked), 1 episode, horizon 12, 2 workers, a 2-member roster."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tokenomics.domain.shocks import ShockEvent, ShockKind, ShockTarget
from tokenomics.env.shock_schedule import canonical_events, shock_digest
from tokenomics.eval.gate import (
    CANDIDATE,
    CSV_COLUMNS,
    EMPTY_SHOCK_DIGEST,
    RESILIENCE_COLUMNS,
    SHOCK_COLUMNS,
    TWIN,
    DigestMismatch,
    EpisodeRow,
    compute_report,
    load_side_effect_params,
    resilience_table,
    row_from_dict,
    row_to_dict,
    rows_digest,
    side_effect_params_json,
)
from tokenomics.eval.gate_cli import (
    build_jobs,
    main,
    read_rows,
    recompute,
    recompute_details,
    run_probe,
    run_report,
    run_screen,
    select_worlds,
)
from tokenomics.eval.registry import (
    REGISTRY_V2,
    SEED_ORDER,
    Registry,
    build_worlds_v1,
    load_registry,
    save_registry,
)
from tokenomics.eval.resilience import ResilienceParams
from tokenomics.eval.roster import Roster, RosterMember, contract_for, load_roster, save_roster
from tokenomics.eval.shock_profiles import (
    DEFAULT_SHOCK_PROFILES,
    SHOCKS_I,
    SHOCKS_I_SMOKE,
    load_shock_profile,
    load_shock_profiles,
)

HORIZON = 12
SHOCKED_WORLD = "dev-sparse_wide-12-shocked-00"
CALM_WORLD = "dev-sparse_wide-12-calm-00"
RUN_WORLDS = (CALM_WORLD, SHOCKED_WORLD)
MEMBERS = (("entity_planner", "proportional"), ("random", "even_split"))
SUBJECT = "baseline:entity_planner/blend_0.5"
EXP_0045 = Path(__file__).resolve().parents[2] / "experiments" / "EXP-0045-champion-dev-report-12-42"


def _fixture_registry() -> Registry:
    v1 = build_worlds_v1(SEED_ORDER)
    v2 = load_registry(REGISTRY_V2)
    shocked = v2.entry(SHOCKED_WORLD)
    params = {**shocked.generator["params"], "shock_profile": SHOCKS_I_SMOKE}
    shocked = dataclasses.replace(shocked, generator={**shocked.generator, "params": params})
    return Registry(name="fixture-v2", seed_ranges=v2.seed_ranges,
                    entries=(v1.entry("dev-dense-12-calm-00"), v2.entry(CALM_WORLD), shocked),
                    shock_profiles=v2.shock_profiles)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("inputs")
    save_registry(_fixture_registry(), d / "registry.json")
    roster = Roster(name="fixture-roster", replicates=1, members=tuple(
        RosterMember(p, o, contract_for(p, o), "random" in (p, o)) for p, o in MEMBERS))
    save_roster(roster, d / "roster.json")
    (d / "params.json").write_text(side_effect_params_json(0.5, 0.0, 0.05, name="fixture"))
    return {"registry": d / "registry.json", "roster": d / "roster.json",
            "params": d / "params.json"}


def _argv(inputs: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return ["--mode", "report", "--out-dir", str(out), "--subject", SUBJECT,
            "--registry", str(inputs["registry"]), "--roster", str(inputs["roster"]),
            "--params", str(inputs["params"]), "--horizon", str(HORIZON),
            "--episodes-per-world", "1", "--families", "sparse_wide", "--dry-run", *extra]


@pytest.fixture(scope="module")
def report_dir(inputs: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("EXP-gate-shocks")
    assert main(_argv(inputs, out, "--workers", "2")) == 0
    return out


def _schedule(row: EpisodeRow) -> list[dict]:
    assert row.shock_schedule is not None
    return json.loads(row.shock_schedule)


# --- the smoke profile (D7) ---


def test_smoke_profile_is_derived_and_not_in_the_file() -> None:
    base, smoke = load_shock_profile(SHOCKS_I), load_shock_profile(SHOCKS_I_SMOKE)
    assert [k.start_window for k in smoke.kinds] == [(2, 8), (2, 8)]
    assert [dataclasses.replace(k, start_window=(0, 0)) for k in smoke.kinds] == \
        [dataclasses.replace(k, start_window=(0, 0)) for k in base.kinds]
    assert (smoke.drift_j, smoke.resilience) == (base.drift_j, base.resilience)
    assert set(json.loads(DEFAULT_SHOCK_PROFILES.read_text())["profiles"]) == {SHOCKS_I}
    assert load_shock_profiles().version == \
        "c92d066085c000a066d91803b48f4b85ad81db22c4bbd4fa61531d94c267bd6f"


def test_real_runs_refuse_the_smoke_profile(inputs: dict[str, Path], tmp_path: Path) -> None:
    reg, roster = load_registry(inputs["registry"]), load_roster(inputs["roster"])
    params = load_side_effect_params(inputs["params"])
    with pytest.raises(ValueError, match="test-only shock profile"):
        run_report(out_dir=tmp_path / "r", subject=SUBJECT, split="development",
                   horizon=HORIZON, workers=1, registry=reg, roster=roster, params=params,
                   params_payload={}, source_commit="a" * 40)
    with pytest.raises(ValueError, match="test-only shock profile"):
        run_screen(out_dir=tmp_path / "s", horizon=HORIZON, workers=1, registry=reg,
                   roster=roster, source_commit="a" * 40)
    with pytest.raises(ValueError, match="test-only shock profile"):
        run_probe(out_dir=tmp_path / "p", horizon=HORIZON, workers=1, registry=reg,
                  roster=roster, scales=[1.0], families=["sparse_wide"])
    assert not any(tmp_path.iterdir())  # refused before any write


# --- the end-to-end report ---


def test_shocked_episodes_carry_both_kinds_inside_the_horizon(report_dir: Path) -> None:
    rows = read_rows(report_dir / "episodes.csv")
    shocked = [r for r in rows if r.world == SHOCKED_WORLD and r.subject != TWIN]
    assert {r.subject for r in shocked} == {CANDIDATE, "entity_planner/proportional",
                                            "random/even_split"}
    for r in shocked:
        assert r.shock_events > 0 and r.shock_digest != EMPTY_SHOCK_DIGEST
    cand = [r for r in shocked if r.subject == CANDIDATE]
    assert len(cand) == 1
    events = _schedule(cand[0])
    assert len(events) == cand[0].shock_events == 3  # 2 hits + 1 outage at 12 producers
    assert {e["kind"] for e in events} == {"capacity_hit", "producer_outage"}
    assert min(e["startCycle"] for e in events) <= HORIZON
    assert {r.shock_digest for r in shocked} == {shock_digest(_events(events))}
    for r in rows:
        if r.world == CALM_WORLD:
            assert (r.shock_events, r.shock_digest) == (0, EMPTY_SHOCK_DIGEST)


def _events(canonical: list[dict]) -> list[ShockEvent]:
    from tokenomics.domain.shocks import shock_model_from_dict

    return list(shock_model_from_dict({"events": canonical}).events)


def test_twin_shares_the_demand(report_dir: Path) -> None:
    rows = read_rows(report_dir / "episodes.csv")
    twins = [r for r in rows if r.subject == TWIN]
    assert [(t.world, t.replicate) for t in twins] == [(SHOCKED_WORLD, 0)]  # no twin on calm
    cand = next(r for r in rows if r.world == SHOCKED_WORLD and r.subject == CANDIDATE)
    t = twins[0]
    assert t.demand_digest == cand.demand_digest
    assert (t.shock_events, t.shock_digest, _schedule(t)) == (0, EMPTY_SHOCK_DIGEST, [])
    assert len(t.fulfilled_trace or ()) == len(cand.fulfilled_trace or ()) == HORIZON
    # roster rows record no resilience inputs
    assert all(r.shock_schedule is None and r.fulfilled_trace is None for r in rows
               if r.subject not in (CANDIDATE, TWIN))


def test_spec_pins_twin_worlds(report_dir: Path) -> None:
    grid = json.loads((report_dir / "spec.json").read_text())["pins"]["grid"]
    assert grid[SHOCKED_WORLD]["twin"] == {"window": 5, "tolerance": 0.05}
    assert "twin" not in grid[CALM_WORLD]
    assert set(grid) == set(RUN_WORLDS)


def test_resilience_rows_in_a_report(report_dir: Path) -> None:
    text = (report_dir / "resilience.csv").read_text()
    table = list(csv.DictReader(io.StringIO(text)))
    rows = read_rows(report_dir / "episodes.csv")
    cand = next(r for r in rows if r.world == SHOCKED_WORLD and r.subject == CANDIDATE)
    events = _schedule(cand)
    assert tuple(text.splitlines()[0].split(",")) == RESILIENCE_COLUMNS
    assert len(table) == len(events)  # one row per shock: all start <= 12 = played cycles
    assert {t["world"] for t in table} == {SHOCKED_WORLD}
    assert {t["subject"] for t in table} == {CANDIDATE}
    assert sorted(t["kind"] for t in table) == sorted(e["kind"] for e in events)
    assert [int(t["start"]) for t in table] == [e["startCycle"] for e in events]
    for t in table:
        assert t["producer"] and t["overlapped"] in ("true", "false")
        assert t["recovered"] in ("true", "false")
        assert (t["recovery_cycles"] == "") == (t["recovered"] == "false")
        int(t["dip_units"])
    spec = json.loads((report_dir / "spec.json").read_text())
    assert spec["result"]["resilience_rows"] == len(table)


def test_report_then_recompute_identical(report_dir: Path) -> None:
    r = recompute_details(report_dir)
    assert r.status == "valid" and r.verdict_ok and r.resilience_ok and r.report is not None
    assert r.report.to_json() == (report_dir / "verdict.json").read_text()
    res = subprocess.run([sys.executable, "-m", "tokenomics.eval.gate", "--mode", "recompute",
                          "--out-dir", str(report_dir)], capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, res.stderr
    assert "MATCH" in res.stdout and "MISMATCH" not in res.stdout


def test_recompute_detects_an_edited_resilience_file(report_dir: Path, tmp_path: Path) -> None:
    for name in ("spec.json", "verdict.json", "episodes.csv", "resilience.csv"):
        (tmp_path / name).write_text((report_dir / name).read_text())
    lines = (tmp_path / "resilience.csv").read_text().splitlines(keepends=True)
    (tmp_path / "resilience.csv").write_text("".join(lines[:-1]))  # drop one shock row
    r = recompute_details(tmp_path)
    assert r.verdict_ok and not r.resilience_ok
    assert main(["--mode", "recompute", "--out-dir", str(tmp_path)]) == 1
    (tmp_path / "resilience.csv").unlink()  # a missing file matches only an empty table
    assert recompute_details(tmp_path).resilience_ok is False


def test_resilience_is_not_an_acceptance_clause(report_dir: Path, inputs: dict[str, Path]) -> None:
    """Twins that make every shock look unrecovered with a huge dip change the resilience rows,
    and leave the report (every clause, the verdict) byte-identical; so does dropping twins."""
    rows = read_rows(report_dir / "episodes.csv")
    roster = load_roster(inputs["roster"])
    params = load_side_effect_params(inputs["params"])
    grid_res = {SHOCKED_WORLD: ResilienceParams()}
    base = compute_report(rows, roster, params, "r", gate_sizes=(12,)).to_json()
    slow = [dataclasses.replace(r, units=r.units + 10_000.0, fulfilled_trace=tuple(
        v + 1_000 for v in r.fulfilled_trace or ())) if r.subject == TWIN else r for r in rows]
    before, after = resilience_table(rows, grid_res), resilience_table(slow, grid_res)
    assert after != before
    assert all(t["recovered"] == "false" for t in after)
    assert compute_report(slow, roster, params, "r", gate_sizes=(12,)).to_json() == base
    no_twins = [r for r in rows if r.subject != TWIN]
    assert compute_report(no_twins, roster, params, "r", gate_sizes=(12,)).to_json() == base


# --- digest pairing ---


def _copy_run(report_dir: Path, dst: Path) -> list[dict[str, str]]:
    for name in ("spec.json", "verdict.json", "resilience.csv"):
        (dst / name).write_text((report_dir / name).read_text())
    with (report_dir / "episodes.csv").open() as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def test_shock_digest_mismatch_fails_recompute(report_dir: Path, tmp_path: Path) -> None:
    rows = _copy_run(report_dir, tmp_path)
    member = next(r for r in rows if r["world"] == SHOCKED_WORLD and r["subject"] == "random/even_split")
    member["shock_digest"] = "0" * 64
    _write_csv(tmp_path / "episodes.csv", rows)
    with pytest.raises(DigestMismatch,
                       match=rf"{SHOCKED_WORLD} episode {member['episode']}: random/even_split "
                             r"replicate 0 saw shock digest 0{64}"):
        recompute(tmp_path)


@pytest.mark.parametrize("field, value, match", [
    ("demand_digest", "f" * 64, "twin:candidate replicate 0 saw demand digest"),
    ("shock_events", "1", "the shock-free twin has 1 shock events"),
])
def test_twin_pairing_fails_recompute(report_dir: Path, tmp_path: Path, field: str, value: str,
                                      match: str) -> None:
    rows = _copy_run(report_dir, tmp_path)
    twin = next(r for r in rows if r["subject"] == TWIN)
    twin[field] = value
    if field == "shock_events":  # keep the row self-consistent: one recorded event
        cand = next(r for r in rows if r["world"] == SHOCKED_WORLD and r["subject"] == CANDIDATE)
        first = json.loads(cand["shock_schedule"])[:1]
        twin["shock_schedule"] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        twin["shock_digest"] = shock_digest(_events(first))
    _write_csv(tmp_path / "episodes.csv", rows)
    with pytest.raises(DigestMismatch, match=match):
        recompute(tmp_path)


def _replay(report_dir: Path, tamper: str | None, edits: dict[str, str]):  # -> run_job_fn
    """A `run_job_fn` that returns the recorded rows (no simulation), with `edits` applied to
    the `tamper` subject's shocked-world row."""
    with (report_dir / "episodes.csv").open() as fh:
        recorded = {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])): d
                    for d in csv.DictReader(fh)}

    def fn(job: dict) -> dict[str, str]:
        d = dict(recorded[(job["world"], job["episode"], job["subject"], job["replicate"])])
        if job["subject"] == tamper and job["world"] == SHOCKED_WORLD:
            d.update(edits)
        return d
    return fn


def _contaminated_twin(report_dir: Path) -> dict[str, str]:
    """Twin row fields carrying one real shock event, self-consistent (digest = schedule)."""
    with (report_dir / "episodes.csv").open() as fh:
        cand = next(d for d in csv.DictReader(fh)
                    if d["world"] == SHOCKED_WORLD and d["subject"] == CANDIDATE)
    first = json.loads(cand["shock_schedule"])[:1]
    return {"shock_events": "1",
            "shock_schedule": json.dumps(first, sort_keys=True, separators=(",", ":")),
            "shock_digest": shock_digest(_events(first))}


@pytest.mark.parametrize("tamper, edits, match", [
    ("entity_planner/proportional", {"shock_digest": "1" * 64}, "saw shock digest 1{64}"),
    (TWIN, {"demand_digest": "2" * 64}, "twin:candidate replicate 0 saw demand digest 2{64}"),
    (TWIN, None, "the shock-free twin has 1 shock events"),  # contaminated twin
])
def test_digest_mismatch_stops_a_live_run(report_dir: Path, inputs: dict[str, Path],
                                          tmp_path: Path, tamper: str,
                                          edits: dict[str, str] | None, match: str) -> None:
    out = tmp_path / "run"
    kw = dict(out_dir=out, subject=SUBJECT, split="development", horizon=HORIZON, workers=1,
              registry=load_registry(inputs["registry"]), roster=load_roster(inputs["roster"]),
              params=load_side_effect_params(inputs["params"]), params_payload={},
              families=["sparse_wide"], episodes_per_world=1, dry_run=True,
              log=lambda _: None)
    edits = _contaminated_twin(report_dir) if edits is None else edits
    with pytest.raises(DigestMismatch, match=match):
        run_report(run_job_fn=_replay(report_dir, tamper, edits), **kw)  # type: ignore[arg-type]
    # stopped as the row arrived (not by the pure re-check after the run): the bad row was
    # never appended and no report was written; the mismatch is a recorded `run` failure
    assert tamper not in {r.subject for r in read_rows(out / "episodes.csv")
                          if r.world == SHOCKED_WORLD}
    assert json.loads((out / "verdict.json").read_text())["status"] == "invalid"
    assert not (out / "worlds.csv").exists()
    with (out / "contract_failures.csv").open() as fh:
        (failure,) = list(csv.DictReader(fh))
    assert (failure["scope"], failure["check"], failure["error_type"]) == \
        ("run", "digest_pairing", "DigestMismatch")
    assert json.loads((out / "verdict.json").read_text())["contract_failures"] == 1
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"
    # resume with honest rows: every job completes, the recorded mismatch keeps it invalid
    report, _ = run_report(run_job_fn=_replay(report_dir, None, {}), **kw)  # type: ignore[arg-type]
    assert report is None
    assert len(read_rows(out / "episodes.csv")) == len(read_rows(report_dir / "episodes.csv"))
    assert json.loads((out / "spec.json").read_text())["status"] == "invalid"
    r = recompute_details(out)
    assert (r.status, r.contract_failures, r.missing_rows) == ("invalid", 1, 0)


# --- jobs and filters ---


def test_twin_jobs_follow_every_candidate_replicate(inputs: dict[str, Path]) -> None:
    reg, roster = load_registry(inputs["registry"]), load_roster(inputs["roster"])
    roster = dataclasses.replace(roster, replicates=3)
    worlds = select_worlds(reg, "development", families=["sparse_wide"])
    jobs = build_jobs(worlds, roster, "baseline:random/even_split", episodes_per_world=1,
                      twin_worlds={SHOCKED_WORLD})
    cand = {(j["world"], j["replicate"]): j for j in jobs if j["subject"] == CANDIDATE}
    twins = {(j["world"], j["replicate"]): j for j in jobs if j["subject"] == TWIN}
    assert set(twins) == {(SHOCKED_WORLD, r) for r in range(3)}
    for k, t in twins.items():
        assert {**t, "subject": CANDIDATE} == cand[k]  # same policies, seeds, contract
    assert not any(j["subject"] == TWIN for j in build_jobs(
        worlds, roster, None, episodes_per_world=1, twin_worlds={SHOCKED_WORLD}))


def test_regimes_filter(inputs: dict[str, Path], tmp_path: Path) -> None:
    reg = load_registry(inputs["registry"])
    assert [e.id for e in select_worlds(reg, "development", regimes=["shocked"])] == \
        [SHOCKED_WORLD]
    assert len(select_worlds(reg, "development", regimes=["calm"])) == 2
    with pytest.raises(ValueError, match="world/episode filters are set"):
        run_report(out_dir=tmp_path / "r", subject=SUBJECT, split="development",
                   horizon=HORIZON, workers=1, registry=reg, roster=load_roster(inputs["roster"]),
                   params=load_side_effect_params(inputs["params"]), params_payload={},
                   regimes=["calm"], source_commit="a" * 40)
    with pytest.raises(SystemExit):
        main(["--mode", "probe", "--out-dir", str(tmp_path / "p"), "--scales", "1.0",
              "--regimes", "calm"])


# --- rows ---


def test_empty_shock_digest_is_the_env_digest_of_no_shocks() -> None:
    assert EMPTY_SHOCK_DIGEST == shock_digest(())


def _row(**kw: object) -> EpisodeRow:
    base: dict = dict(world="w", size=12, regime="shocked", episode=1, subject=CANDIDATE,
                      replicate=0, units=5.0, dead=False, demand_digest="d", r_start=1.0,
                      r_end=1.0, consumer_fill_rate={"c0": 1.0})
    return EpisodeRow(**{**base, **kw})


def _schedule_json(events: list[ShockEvent]) -> str:
    return json.dumps(canonical_events(events), sort_keys=True, separators=(",", ":"))


HIT = ShockEvent(ShockKind.CAPACITY_HIT, ShockTarget(producer="p0", capacity_types=("c1",)),
                 start_cycle=3, duration=1, magnitude=0.5)


def test_row_checks_its_recorded_schedule() -> None:
    sched = _schedule_json([HIT])
    ok = _row(shock_digest=shock_digest([HIT]), shock_events=1, shock_schedule=sched,
              fulfilled_trace=(1, 2))
    assert row_from_dict(row_to_dict(ok)) == ok
    with pytest.raises(ValueError, match="hashes to"):
        _row(shock_digest="0" * 64, shock_events=1, shock_schedule=sched)
    with pytest.raises(ValueError, match="does not hold 2 events"):
        _row(shock_digest=shock_digest([HIT]), shock_events=2, shock_schedule=sched)
    with pytest.raises(ValueError, match="ints >= 0"):
        _row(fulfilled_trace=(1, -1))


def test_csv_without_shock_columns_reads_as_no_shocks() -> None:
    d = row_to_dict(_row())
    legacy = {k: v for k, v in d.items() if k not in SHOCK_COLUMNS}
    r = row_from_dict(legacy)
    assert (r.shock_digest, r.shock_events, r.shock_schedule, r.fulfilled_trace) == \
        (EMPTY_SHOCK_DIGEST, 0, None, None)
    partial = {k: v for k, v in d.items() if k != "fulfilled_trace"}
    with pytest.raises(ValueError, match=r"lacks columns \['fulfilled_trace'\]"):
        row_from_dict(partial)


def test_rows_digest_unchanged_for_rows_without_shocks() -> None:
    """The pre-5.2 digest form (no shock columns) for rows at their defaults, so a v1 screen's
    calibration evidence is unchanged; a shocked row binds its shock fields."""
    import hashlib

    r = _row()
    old = {k: v for k, v in row_to_dict(r).items() if k not in SHOCK_COLUMNS}
    canon = [json.dumps(old, sort_keys=True)]
    payload = json.dumps({"rows": canon}, sort_keys=True, separators=(",", ":"))
    assert rows_digest([r]) == hashlib.sha256(payload.encode()).hexdigest()
    assert rows_digest([dataclasses.replace(r, shock_digest="1" * 64)]) != rows_digest([r])
    assert rows_digest([dataclasses.replace(r, shock_events=2)]) != rows_digest([r])


def test_twin_that_died_early_is_padded_with_zeros() -> None:
    sched = _schedule_json([HIT])
    kw = dict(shock_digest=shock_digest([HIT]), shock_events=1, shock_schedule=sched)
    cand = _row(fulfilled_trace=(4, 4, 1, 1, 1, 1, 1, 1, 1, 1), **kw)
    twin = _row(subject=TWIN, dead=True, fulfilled_trace=(4, 4, 4, 4, 4, 4),
                shock_schedule="[]")
    params = {"w": ResilienceParams(window=2, tolerance=0.0)}
    (line,) = resilience_table([cand, twin], params)
    # the twin completes 0 from cycle 7 on: window [7, 8] is the first with shocked 2 >= twin 0
    # ([6, 7]: 2 < 4 + 0); the dip sums twin - shocked over cycles 3..6 = 4 * 3
    assert (line["start"], line["recovered"], line["recovery_cycle"], line["recovery_cycles"],
            line["dip_units"]) == ("3", "true", "7", "3", "12")
    with pytest.raises(ValueError, match="no twin row"):
        resilience_table([cand], params)


def test_candidate_death_censors_every_resilience_row() -> None:
    """D6: a shocked episode that ended in system death is unrecovered, with the dip summed to
    its last played cycle, even when a window had already caught up with the twin. The death
    comes from the candidate row's `dead`, not from the trace."""
    sched = _schedule_json([HIT])
    kw = dict(shock_digest=shock_digest([HIT]), shock_events=1, shock_schedule=sched)
    trace = (4, 4, 1, 1, 4, 4, 4, 2, 2, 2)
    twin = _row(subject=TWIN, fulfilled_trace=(4,) * 10, shock_schedule="[]")
    params = {"w": ResilienceParams(window=2, tolerance=0.0)}
    (alive,) = resilience_table([_row(fulfilled_trace=trace, **kw), twin], params)
    # alive: window [5, 6] = 8 >= 8 recovers; the dip is cycles 3..4 = 3 + 3
    assert (alive["recovered"], alive["recovery_cycle"], alive["recovery_cycles"],
            alive["dip_units"]) == ("true", "5", "1", "6")
    (dead,) = resilience_table([_row(dead=True, fulfilled_trace=trace, **kw), twin], params)
    # dead: censored; the dip runs over cycles 3..10 = 3 + 3 + 0 + 0 + 0 + 2 + 2 + 2
    assert (dead["recovered"], dead["recovery_cycle"], dead["recovery_cycles"],
            dead["dip_units"], dead["dip_fraction"]) == ("false", "", "", "12", repr(12 / 32))


def test_twin_shorter_without_dying_is_refused() -> None:
    sched = _schedule_json([HIT])
    kw = dict(shock_digest=shock_digest([HIT]), shock_events=1, shock_schedule=sched)
    cand = _row(fulfilled_trace=(4, 4, 1, 1, 1, 1, 1, 1, 1, 1), **kw)
    twin = _row(subject=TWIN, fulfilled_trace=(4, 4, 4, 4, 4, 4), shock_schedule="[]")
    params = {"w": ResilienceParams(window=2, tolerance=0.0)}
    with pytest.raises(ValueError, match="without dying"):
        resilience_table([cand, twin], params)


# --- v1 reports ---


@pytest.mark.skipif(not (EXP_0045 / "episodes.csv").exists(), reason="EXP-0045 record absent")
def test_exp_0045_recompute_still_matches() -> None:
    r = recompute_details(EXP_0045)
    assert r.status == "valid" and r.verdict_ok and r.resilience_ok
    assert not (EXP_0045 / "resilience.csv").exists()  # read-only record, nothing written
