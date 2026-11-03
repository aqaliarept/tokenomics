"""Gate screening: zero-baseline replacement, headroom, pruning and X/Y calibration (design D4,
D5, D10, add-measurement-gate-harness; gate-harness "Zero-baseline screening", "Development
headroom check", "Side-effect parameter calibration", "Roster pruning uses development evidence
only")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from tokenomics.eval.families import CALM, DENSE, SHOCKED, SPARSE_WIDE
from tokenomics.eval.gate import (
    CANDIDATE,
    QUANTILE_METHOD,
    EpisodeRow,
    calibrate_x_y,
    episode_bests,
    headroom,
    load_side_effect_params,
    params_from_dict,
    quantile_nearest_rank,
    side_effect_params_json,
)
from tokenomics.eval.gate_cli import (
    HEADROOM_FAILED,
    job_grid,
    main,
    row_to_dict,
    run_screen,
    screen_outcome,
)
from tokenomics.eval.registry import (
    DEVELOPMENT,
    GATE,
    REPLACED,
    Registry,
    SEED_ORDER,
    build_worlds_v1,
    load_registry,
    save_registry,
)
from tokenomics.eval.roster import (
    Roster,
    RosterMember,
    contract_for,
    load_roster,
    member_id,
    save_roster,
)

A = member_id("always_restore", "even_split")
B = member_id("restore_when_low", "proportional")
RND = member_id("random", "proportional")


def _roster(*pairs: tuple[str, str], replicates: int = 2) -> Roster:
    return Roster(name="t", replicates=replicates, members=tuple(
        RosterMember(p, o, contract_for(p, o), "random" in (p, o)) for p, o in pairs))


ROSTER = _roster(("always_restore", "even_split"), ("restore_when_low", "proportional"))


def _row(world: str, episode: int, subject: str, units: float, *, replicate: int = 0,
         size: int = 12, regime: str = CALM, r_start: float | None = 1.0,
         r_end: float | None = 0.8, fill: dict | None = None, ordered: float | None = 10.0,
         fulfilled: float | None = 5.0) -> EpisodeRow:
    return EpisodeRow(world=world, size=size, regime=regime, episode=episode, subject=subject,
                      replicate=replicate, units=units, dead=False, demand_digest=f"d{episode}",
                      r_start=r_start, r_end=r_end,
                      consumer_fill_rate=fill if fill is not None else {"c0": 0.5},
                      ordered_units=ordered, fulfilled_units=fulfilled)


# --- quantile (D4: the named, deterministic method) ---


@pytest.mark.parametrize("q,want", [(0.05, 1), (0.1, 2), (0.5, 10), (0.51, 11), (1.0, 20)])
def test_nearest_rank_quantile_known_values(q: float, want: float) -> None:
    assert quantile_nearest_rank([float(v) for v in range(20, 0, -1)], q) == want
    assert QUANTILE_METHOD == "nearest-rank"


def test_nearest_rank_small_sample_is_minimum() -> None:
    assert quantile_nearest_rank([0.9, 0.7, 0.8], 0.05) == 0.7  # k = ceil(0.15) = 1
    with pytest.raises(ValueError):
        quantile_nearest_rank([], 0.05)
    with pytest.raises(ValueError):
        quantile_nearest_rank([1.0], 0.0)


# --- argmax member per episode ---


def test_episode_best_is_argmax_with_smallest_id_tie_break() -> None:
    """A wins episode 1 outright; episode 2 is a tie, broken by the smallest member id, whatever
    the tied members' side-effect values."""
    rows = [_row("w", 1, A, 9, r_end=0.9), _row("w", 1, B, 4, r_end=0.1),
            _row("w", 2, A, 6, r_end=0.95), _row("w", 2, B, 6, r_end=0.2)]
    bests = episode_bests(rows, ROSTER)
    assert [(b.episode, b.member) for b in bests] == [(1, A), (2, A)]
    assert min(A, B) == A


def test_episode_best_uses_replicate_means_and_exclusions() -> None:
    """A random member's value and side effects are replicate means; an unmeasurable run-down
    (r_start 0 or None) and a consumer that ordered nothing (fill None) are excluded."""
    roster = _roster(("always_restore", "even_split"), ("random", "proportional"))
    rows = [
        _row("w", 1, A, 3),
        _row("w", 1, RND, 4, replicate=0, r_start=1.0, r_end=0.6,
             fill={"c0": 0.4, "c1": None}, ordered=10, fulfilled=4),
        _row("w", 1, RND, 6, replicate=1, r_start=0.0, r_end=0.5,
             fill={"c0": 0.8, "c1": None}, ordered=10, fulfilled=8),
    ]
    (b,) = episode_bests(rows, roster)
    assert (b.member, b.units) == (RND, 5.0)
    assert b.run_down == 0.6  # replicate 1 has r_start 0: excluded
    assert b.fill_rates == (pytest.approx(0.6),)  # c1 ordered nothing: excluded
    assert b.fill == pytest.approx(0.6)


def test_episode_best_refuses_candidate_and_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="roster-only"):
        episode_bests([_row("w", 1, A, 1), _row("w", 1, B, 1), _row("w", 1, CANDIDATE, 9)],
                      ROSTER)
    with pytest.raises(ValueError, match="roster wants"):
        episode_bests([_row("w", 1, A, 1)], ROSTER)


# --- X / Y calibration on synthetic rows with known quantiles ---


def _calibration_rows(n: int = 20) -> list[EpisodeRow]:
    """n episodes; A always wins with run-down 0.50, 0.51, ... and fill rates (0.30 + i/100,
    0.90); B loses with terrible side effects that must never enter the bar."""
    rows = []
    for i in range(n):
        rows.append(_row("w", i, A, 10, r_start=1.0, r_end=0.50 + i / 100,
                         fill={"c0": 0.30 + i / 100, "c1": 0.90}))
        rows.append(_row("w", i, B, 1, r_start=1.0, r_end=0.0, fill={"c0": 0.0, "c1": 0.0}))
    return rows


@pytest.mark.parametrize("q,x,y", [(0.05, 0.50, 0.31), (0.10, 0.51, 0.33), (0.5, 0.59, 0.49)])
def test_calibration_known_quantiles(q: float, x: float, y: float) -> None:
    """20 run-down values 0.50..0.69: k = ceil(20q) -> 1, 2, 10. 40 fill values (0.30..0.49 and
    twenty 0.90): k = ceil(40q) -> 2, 4, 20 -> 0.31, 0.33, 0.49."""
    cal = calibrate_x_y(_calibration_rows(), ROSTER, q)
    assert cal.x == pytest.approx(x) and cal.y == pytest.approx(y)
    assert (cal.episodes, cal.run_down_values, cal.fill_values) == (20, 20, 40)
    assert cal.method == "nearest-rank"


def test_calibration_evidence_binds_exact_rows() -> None:
    """rows_digest identifies the exact calibration rows: changes that leave X, Y and the argmax
    summary untouched (scaled r_start/r_end, the winner's units, a demand digest, a losing row, a
    consumer id) still change it; reordering the rows does not."""
    from dataclasses import replace as rep

    rows = _calibration_rows()
    base = calibrate_x_y(rows, ROSTER, 0.05)
    assert calibrate_x_y(list(reversed(rows)), ROSTER, 0.05).rows_digest == base.rows_digest
    edits = [
        lambda r: rep(r, r_start=2.0, r_end=2 * r.r_end),
        lambda r: rep(r, units=11.0),
        lambda r: rep(r, demand_digest="other"),
        lambda r: rep(r, consumer_fill_rate=(("cX", 0.30), ("c1", 0.90))),
    ]
    for edit in edits:
        changed = [edit(rows[0])] + rows[1:]
        cal = calibrate_x_y(changed, ROSTER, 0.05)
        assert cal.rows_digest != base.rows_digest
        assert (cal.x, cal.y) == (base.x, base.y)
    loser = [rows[0], rep(rows[1], units=2.0)] + rows[2:]
    other = calibrate_x_y(loser, ROSTER, 0.05)
    assert other.rows_digest != base.rows_digest and other.bests_digest == base.bests_digest


def test_changing_x_or_y_creates_a_new_params_version() -> None:
    """gate-harness "Parameters changed after a gate run": any change to X or Y is a new version;
    the old file still parses under its own version."""
    v1 = json.loads(side_effect_params_json(0.5, 0.3, 0.05, name="p"))
    v2x = json.loads(side_effect_params_json(0.51, 0.3, 0.05, name="p"))
    v2y = json.loads(side_effect_params_json(0.5, 0.31, 0.05, name="p"))
    versions = {params_from_dict(d).version for d in (v1, v2x, v2y)}
    assert len(versions) == 3
    assert params_from_dict(v1).version == v1["content_hash"]


# --- headroom (D10) ---


def _headroom_rows(world: str, fill: float, size: int = 12) -> list[EpisodeRow]:
    return [_row(world, 1, A, 10, size=size, ordered=100, fulfilled=100 * fill),
            _row(world, 1, B, 1, size=size, ordered=100, fulfilled=0)]


def test_saturated_family_rejected() -> None:
    """gate-harness "Saturated family rejected": a family × size whose median hindsight-baseline
    fill reaches the ceiling fails; exactly at the ceiling fails too (the spec says below)."""
    rows = []
    family = {}
    for i, f in enumerate([0.97, 0.98, 0.99]):
        rows += _headroom_rows(f"sat-{i}", f, size=42)
        family[f"sat-{i}"] = SPARSE_WIDE
    for i, f in enumerate([0.50, 0.95, 0.70]):
        rows += _headroom_rows(f"ok-{i}", f)
        family[f"ok-{i}"] = DENSE
    for i, f in enumerate([0.95, 0.95, 0.96]):
        rows += _headroom_rows(f"edge-{i}", f)
        family[f"edge-{i}"] = SPARSE_WIDE
    res = {(h.family, h.size): h for h in headroom(episode_bests(rows, ROSTER), family, 0.95)}
    assert res[(SPARSE_WIDE, 42)].median_fill == pytest.approx(0.98)
    assert not res[(SPARSE_WIDE, 42)].passed
    assert res[(DENSE, 12)].median_fill == pytest.approx(0.70) and res[(DENSE, 12)].passed
    assert not res[(SPARSE_WIDE, 12)].passed  # median exactly 0.95


# --- screen_outcome on synthetic rows over the real draft registry ---


def _dev_rows(reg: Registry, *, zero: set[str] = frozenset(), fill: float = 0.5,
              roster: Roster = ROSTER) -> list[EpisodeRow]:
    """Roster rows for every development gate world and reserve (1 episode's worth per registered
    episode seed). Worlds in `zero` deliver nothing."""
    rows = []
    for e in reg.entries:
        if e.split != DEVELOPMENT:
            continue
        for ep in e.episode_seeds:
            for k, m in enumerate(roster.members):
                for r in range(roster.replicates_for(m)):
                    units = 0.0 if e.id in zero else 10.0 - k
                    rows.append(_row(e.id, ep, m.id, units, replicate=r, size=e.size,
                                     regime=e.regime, ordered=100, fulfilled=100 * fill,
                                     r_end=0.8 - k / 10, fill={"c0": 0.6 - k / 10}))
    return rows


def test_zero_baseline_world_replaced_by_its_reserve() -> None:
    reg = build_worlds_v1(SEED_ORDER)
    zero = "dev-dense-42-shocked-02"
    out = screen_outcome(reg, ROSTER, _dev_rows(reg, zero={zero}), q=0.05, prune=False)
    assert out.passed
    assert out.registry.entry(zero).status == REPLACED
    assert out.registry.entry("dev-dense-42-shocked-r0").status == GATE
    assert [(r.replaced, r.by) for r in out.registry.screening.replacements] == [
        (zero, "dev-dense-42-shocked-r0")]
    assert out.registry.screening.splits == (DEVELOPMENT,)
    assert out.registry.screening.headroom is not None


def test_saturated_screen_writes_no_calibration() -> None:
    reg = build_worlds_v1(SEED_ORDER)
    out = screen_outcome(reg, ROSTER, _dev_rows(reg, fill=0.99), q=0.05, prune=False)
    assert not out.passed
    assert out.roster is None and out.params_json is None and out.calibration is None
    assert all(not h.passed for h in out.headroom)


def test_screen_outcome_calibrates_and_freezes() -> None:
    """X/Y from the argmax member (A: r_end 0.8, c0 fill 0.6), never from the loser (B)."""
    reg = build_worlds_v1(SEED_ORDER)
    out = screen_outcome(reg, ROSTER, _dev_rows(reg), q=0.05, prune=False)
    assert out.roster is not None and out.roster.frozen
    p = params_from_dict(json.loads(out.params_json))
    assert (p.x, p.y) == (pytest.approx(0.8), pytest.approx(0.6))
    d = json.loads(out.params_json)
    assert d["method"] == "nearest-rank" and d["q"] == 0.05
    assert d["evidence"]["roster_version"] == out.roster.version
    assert d["evidence"]["registry_version"] == out.registry.version


def test_screen_outcome_prunes_never_winning_member() -> None:
    reg = build_worlds_v1(SEED_ORDER)
    out = screen_outcome(reg, ROSTER, _dev_rows(reg), q=0.05, prune=True)
    assert out.roster is not None
    assert out.roster.member_ids == (A,) and out.roster.pruning.dropped == (B,)
    assert out.roster.frozen


# --- e2e: `--mode screen --dry-run` on a fixture registry ---

WORLDS = ("dev-dense-12-calm-00", "dev-sparse_wide-12-calm-00")


@pytest.fixture(scope="module")
def fixture_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("screen-inputs")
    full = build_worlds_v1(SEED_ORDER)
    save_registry(Registry(name="fixture", seed_ranges=full.seed_ranges,
                           entries=tuple(full.entry(w) for w in WORLDS)), d / "registry.json")
    save_roster(_roster(("entity_planner", "proportional"), ("random", "even_split")),
                d / "roster.json")
    return {"registry": d / "registry.json", "roster": d / "roster.json"}


def test_screen_dry_run_writes_frozen_files(fixture_inputs: dict[str, Path],
                                            tmp_path: Path) -> None:
    out = tmp_path / "EXP-screen"
    status = main(["--mode", "screen", "--dry-run", "--out-dir", str(out),
                   "--registry", str(fixture_inputs["registry"]),
                   "--roster", str(fixture_inputs["roster"]), "--horizon", "12",
                   "--episodes-per-world", "1", "--workers", "2"])
    spec = json.loads((out / "spec.json").read_text())
    report = json.loads((out / "screening.json").read_text())
    if status == HEADROOM_FAILED:  # the fixture worlds may be saturated at h=12: still a valid run
        assert not (out / "frozen").exists() and spec["status"] == "headroom-failed"
        pytest.skip(f"fixture worlds saturated: {report['headroom']}")
    assert status == 0 and spec["status"] == "complete" and spec["kind"] == "dry-run"
    reg = load_registry(out / "frozen" / "worlds_v1.json")
    roster = load_roster(out / "frozen" / "roster_v1.json")
    params = load_side_effect_params(out / "frozen" / "side_effect_params_v1.json")
    assert reg.screening is not None and reg.screening.headroom is not None
    assert roster.frozen
    assert spec["frozen"] == {"worlds_v1.json": reg.version, "roster_v1.json": roster.version,
                              "side_effect_params_v1.json": params.version}
    assert (params.x, params.y) == (report["x"], report["y"])
    assert "DRY RUN" in (out / "RESULT.md").read_text()


def _stub_rows(fill: float) -> Callable[[dict], dict]:
    def stub(job: dict) -> dict:
        k = 0 if job["subject"].startswith("entity_planner") else 1
        return row_to_dict(_row(job["world"], job["episode"], job["subject"], 10.0 - k,
                                replicate=job["replicate"], size=job["size"],
                                regime=job["regime"], ordered=100, fulfilled=100 * fill))
    return stub


def test_screen_headroom_failure_stops_without_frozen_files(fixture_inputs: dict[str, Path],
                                                            tmp_path: Path) -> None:
    out = tmp_path / "EXP-sat"
    outcome, status = run_screen(
        out_dir=out, horizon=12, workers=1, registry=load_registry(fixture_inputs["registry"]),
        roster=load_roster(fixture_inputs["roster"]), episodes_per_world=1, dry_run=True,
        run_job_fn=_stub_rows(0.99), log=lambda s: None)
    assert status == HEADROOM_FAILED and not outcome.passed
    assert not (out / "frozen").exists()
    assert json.loads((out / "spec.json").read_text())["status"] == "headroom-failed"
    assert (out / "headroom.csv").exists() and (out / "screening.json").exists()


def test_cli_screen_exit_status_on_headroom_failure(fixture_inputs: dict[str, Path],
                                                    tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import tokenomics.eval.gate_cli as gc

    monkeypatch.setattr(gc, "run_job", _stub_rows(0.99))
    real = gc.run_screen

    def screen(**kw: object) -> tuple:
        return real(**kw, run_job_fn=_stub_rows(0.99))  # type: ignore[arg-type]

    monkeypatch.setattr(gc, "run_screen", screen)
    status = main(["--mode", "screen", "--dry-run", "--out-dir", str(tmp_path / "x"),
                   "--registry", str(fixture_inputs["registry"]),
                   "--roster", str(fixture_inputs["roster"]), "--horizon", "12",
                   "--episodes-per-world", "1"])
    assert status == HEADROOM_FAILED


def test_real_screen_requires_commit_and_no_filters(fixture_inputs: dict[str, Path],
                                                    tmp_path: Path) -> None:
    """Also refused: a registry whose families lack development worlds at a gate size (the
    fixture has 12 producers only; D10 admits a family on evidence at both sizes)."""
    with pytest.raises(ValueError, match=r"source commit.*filters are set.*"
                                         r"no development gate worlds for dense/42"):
        run_screen(out_dir=tmp_path / "r", horizon=12, workers=1,
                   registry=load_registry(fixture_inputs["registry"]),
                   roster=load_roster(fixture_inputs["roster"]), episodes_per_world=1)
    assert not (tmp_path / "r").exists()


def test_screen_refuses_screened_registry(fixture_inputs: dict[str, Path], tmp_path: Path) -> None:
    reg = load_registry(fixture_inputs["registry"])
    out = screen_outcome(reg, _roster(("entity_planner", "proportional"),
                                      ("random", "even_split")),
                         _dev_rows(reg, roster=_roster(("entity_planner", "proportional"),
                                                       ("random", "even_split"))),
                         q=0.05, prune=False)
    with pytest.raises(ValueError, match="already screened"):
        run_screen(out_dir=tmp_path / "s", horizon=12, workers=1, registry=out.registry,
                   roster=load_roster(fixture_inputs["roster"]), dry_run=True)


# --- admission: a real report needs the development headroom recorded and passing (D10) ---


def test_admission_requires_screened_split_and_passing_headroom() -> None:
    from tokenomics.eval.gate import Headroom
    from tokenomics.eval.registry import SEALED, record_headroom, screen_zero_baselines

    reg = build_worlds_v1()  # the M1 registry: sparse-wide only (task 10.4)
    assert "has not screened the development split" in reg.admission_problems(DEVELOPMENT)[0]
    rows = _dev_rows(reg)
    dev = screen_zero_baselines(reg, ROSTER, rows)
    assert "records no development headroom check" in dev.admission_problems(DEVELOPMENT)[0]
    groups = sorted({(e.family, e.size) for e in dev.gate_worlds(DEVELOPMENT)})
    ok = [Headroom(f, z, 8, 0.5, 0.95, True) for f, z in groups]
    assert record_headroom(dev, ok).admission_problems(DEVELOPMENT) == []
    assert "no headroom result for ('sparse_wide', 42)" in record_headroom(
        dev, ok[:-1]).admission_problems(DEVELOPMENT)[0]
    failing = ok[:-1] + [Headroom(*groups[-1], 8, 0.96, 0.95, False)]
    assert "headroom failed" in record_headroom(dev, failing).admission_problems(DEVELOPMENT)[0]
    # sealed split: needs its own screen; a development-only screen does not admit it
    assert "has not screened the sealed split" in record_headroom(
        dev, ok).admission_problems(SEALED)[0]
    sealed_rows = []
    for e in reg.gate_worlds(SEALED):
        for ep in e.episode_seeds:
            for m in ROSTER.members:
                sealed_rows.append(_row(e.id, ep, m.id, 5.0, size=e.size, regime=e.regime))
    sealed_only = screen_zero_baselines(reg, ROSTER, sealed_rows, splits=(SEALED,))
    assert "has not screened the development split" in sealed_only.admission_problems(
        DEVELOPMENT)[0]
    assert "has not screened the development split" in sealed_only.admission_problems(SEALED)[0]
    # a registry without 42-producer development worlds is never admitted
    small = Registry(name="small", seed_ranges=reg.seed_ranges,
                     entries=tuple(e for e in reg.entries if e.size == 12))
    small_dev = screen_zero_baselines(small, ROSTER, [r for r in rows if r.size == 12])
    small_ok = record_headroom(small_dev, [h for h in ok if h.size == 12])
    assert small_ok.admission_problems(DEVELOPMENT) == [
        "no development gate worlds for sparse_wide/42"]
    # dense is not a gate family in M1: a registry holding it is never admitted (task 10.4)
    full = build_worlds_v1(SEED_ORDER)
    full_rows = _dev_rows(full)
    full_dev = screen_zero_baselines(full, ROSTER, full_rows)
    full_ok = record_headroom(full_dev, [
        Headroom(f, z, 8, 0.5, 0.95, True)
        for f, z in sorted({(e.family, e.size) for e in full_dev.gate_worlds(DEVELOPMENT)})])
    assert full_ok.admission_problems(DEVELOPMENT) == [
        "family dense is not a gate family in M1 (gate families: sparse_wide)"]
    # a family present only in the sealed split has no development evidence at all
    sealed_family = Registry(name="sf", seed_ranges=full.seed_ranges, entries=tuple(
        e for e in full.entries if e.family == DENSE or e.split == SEALED))
    sf = screen_zero_baselines(sealed_family, ROSTER,
                               [r for r in full_rows if r.world.startswith("dev-dense")])
    sf_ok = record_headroom(sf, [Headroom(DENSE, z, 8, 0.5, 0.95, True) for z in (12, 42)])
    assert sf_ok.admission_problems(DEVELOPMENT) == [
        "family dense is not a gate family in M1 (gate families: sparse_wide)",
        "no development gate worlds for sparse_wide/12",
        "no development gate worlds for sparse_wide/42"]


def test_inconsistent_headroom_refused() -> None:
    """A pass flag that does not follow from median < ceiling is refused when recorded and when a
    hand-edited file is loaded."""
    from tokenomics.eval.gate import Headroom
    from tokenomics.eval.registry import record_headroom, registry_from_dict, screen_zero_baselines

    reg = build_worlds_v1(SEED_ORDER)
    dev = screen_zero_baselines(reg, ROSTER, _dev_rows(reg))
    with pytest.raises(ValueError, match="inconsistent"):
        record_headroom(dev, [Headroom(DENSE, 12, 8, 0.97, 0.95, True)])
    good = record_headroom(dev, [Headroom(DENSE, 12, 8, 0.5, 0.95, True)])
    d = json.loads(good.to_json())
    d["screening"]["headroom"][0]["median_fill"] = 0.99
    from tokenomics.eval.roster import content_hash
    del d["content_hash"]
    d["content_hash"] = content_hash(d)
    with pytest.raises(ValueError, match="inconsistent"):
        registry_from_dict(d)


# --- on-demand reserves (task 8.6, user decision 2026-10-27) ---

GATE_IDS = ("dev-dense-12-calm-00", "dev-sparse_wide-12-calm-00")
RES_IDS = tuple(f"dev-{f}-12-calm-r{k}" for f in ("dense", "sparse_wide") for k in range(4))


@pytest.fixture(scope="module")
def reserve_registry(tmp_path_factory: pytest.TempPathFactory) -> Registry:
    full = build_worlds_v1(SEED_ORDER)
    return Registry(name="fixture-reserves", seed_ranges=full.seed_ranges,
                    entries=tuple(full.entry(w) for w in (*GATE_IDS, *RES_IDS)))


def _zero_stub(zero: set[str], ran: list[dict] | None = None,
               crash_on: str | None = None) -> Callable[[dict], dict]:
    """Deterministic roster rows: worlds in `zero` deliver nothing, the others 10 - k units."""
    def stub(job: dict) -> dict:
        if crash_on is not None and job["world"] == crash_on:
            raise RuntimeError("simulated crash")
        if ran is not None:
            ran.append(job)
        k = 0 if job["subject"] == A else 1
        units = 0.0 if job["world"] in zero else 10.0 - k
        return row_to_dict(_row(job["world"], job["episode"], job["subject"], units,
                                replicate=job["replicate"], size=job["size"],
                                regime=job["regime"], ordered=100, fulfilled=50))
    return stub


def _screen(reg: Registry, out: Path, stub: Callable[[dict], dict], reserves: str) -> tuple:
    return run_screen(out_dir=out, horizon=12, workers=1, registry=reg, roster=ROSTER,
                      episodes_per_world=1, dry_run=True, reserves=reserves,
                      run_job_fn=stub, log=lambda s: None)


def test_on_demand_runs_no_reserve_when_no_zero_world(reserve_registry: Registry,
                                                      tmp_path: Path) -> None:
    """(a) No gate world has a zero baseline -> not one reserve job is scheduled."""
    ran: list[dict] = []
    outcome, status = _screen(reserve_registry, tmp_path / "a", _zero_stub(set(), ran),
                              "on-demand")
    assert status == 0 and outcome.passed
    assert {j["world"] for j in ran} == set(GATE_IDS)
    assert json.loads((tmp_path / "a" / "spec.json").read_text())["waves"] == []


def test_on_demand_runs_exactly_the_needed_reserves(reserve_registry: Registry,
                                                    tmp_path: Path) -> None:
    """(b) A zero gate world triggers its group's reserves in fixed order, one wave at a time:
    r0 (itself zero, so skipped) then r1 (usable). r2/r3 and the other group's reserves never
    run."""
    ran: list[dict] = []
    zero = {"dev-dense-12-calm-00", "dev-dense-12-calm-r0"}
    outcome, status = _screen(reserve_registry, tmp_path / "b", _zero_stub(zero, ran),
                              "on-demand")
    assert status == 0
    assert {j["world"] for j in ran} == {*GATE_IDS, "dev-dense-12-calm-r0", "dev-dense-12-calm-r1"}
    assert len(ran) == 4 * len(ROSTER.members)  # one episode per world, no repeats
    waves = json.loads((tmp_path / "b" / "spec.json").read_text())["waves"]
    assert [(w["wave"], sorted(w["worlds"])) for w in waves] == [
        (2, ["dev-dense-12-calm-r0"]), (3, ["dev-dense-12-calm-r1"])]
    (rep,) = outcome.registry.screening.replacements
    assert (rep.replaced, rep.by, rep.skipped) == (
        "dev-dense-12-calm-00", "dev-dense-12-calm-r1", ("dev-dense-12-calm-r0",))


def test_eager_and_on_demand_give_identical_frozen_files(reserve_registry: Registry,
                                                         tmp_path: Path) -> None:
    """(c) Same rows, both modes: identical frozen files, screening record and report."""
    zero = {"dev-dense-12-calm-00", "dev-dense-12-calm-r0"}
    eager_ran: list[dict] = []
    e_out, _ = _screen(reserve_registry, tmp_path / "eager", _zero_stub(zero, eager_ran), "eager")
    d_out, _ = _screen(reserve_registry, tmp_path / "ondemand", _zero_stub(zero), "on-demand")
    assert {j["world"] for j in eager_ran} == {*GATE_IDS, *RES_IDS}  # eager ran every reserve
    for name in ("worlds_v1.json", "roster_v1.json", "side_effect_params_v1.json"):
        assert ((tmp_path / "eager" / "frozen" / name).read_bytes()
                == (tmp_path / "ondemand" / "frozen" / name).read_bytes()), name
    assert e_out.registry == d_out.registry
    assert ((tmp_path / "eager" / "screening.json").read_text()
            == (tmp_path / "ondemand" / "screening.json").read_text())


def test_on_demand_resumes_mid_wave(reserve_registry: Registry, tmp_path: Path) -> None:
    """A crash inside wave 3 leaves its world recorded in spec.json; the rerun runs only the
    missing wave-3 jobs and ends with the same frozen files as an uninterrupted screen."""
    zero = {"dev-dense-12-calm-00", "dev-dense-12-calm-r0"}
    out = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="simulated crash"):
        _screen(reserve_registry, out, _zero_stub(zero, crash_on="dev-dense-12-calm-r1"),
                "on-demand")
    spec = json.loads((out / "spec.json").read_text())
    assert [sorted(w["worlds"]) for w in spec["waves"]] == [["dev-dense-12-calm-r0"],
                                                            ["dev-dense-12-calm-r1"]]
    ran: list[dict] = []
    _screen(reserve_registry, out, _zero_stub(zero, ran), "on-demand")
    assert {j["world"] for j in ran} == {"dev-dense-12-calm-r1"}
    _screen(reserve_registry, tmp_path / "clean", _zero_stub(zero), "on-demand")
    for name in ("worlds_v1.json", "roster_v1.json", "side_effect_params_v1.json"):
        assert ((out / "frozen" / name).read_bytes()
                == (tmp_path / "clean" / "frozen" / name).read_bytes()), name


def _tamper(out: Path, edit: Callable[[list[dict]], list[dict]]) -> None:
    spec = json.loads((out / "spec.json").read_text())
    spec["waves"] = edit(spec["waves"])
    (out / "spec.json").write_text(json.dumps(spec))


def _wave(world: str, wave: int) -> dict:
    return {"wave": wave, "worlds": job_grid([build_worlds_v1(SEED_ORDER).entry(world)], 1)}


@pytest.mark.parametrize("case, match", [
    ("unplanned", "not the planned wave"),      # no zero gate world, yet a reserve wave
    ("empty", "not the planned wave"),          # an empty wave the planner never writes
    ("eager", "reserves='eager'"),              # eager screens record no waves
    ("renumbered", "malformed"),
    ("extra_key", "malformed"),
    ("wrong_grid", "not the planned wave"),     # right world, different episodes
    ("non_reserve", "not the planned wave"),    # a gate world smuggled in as a "reserve"
    ("skipped_order", "not the planned wave"),  # r1 recorded where the plan needs r0
])
def test_resume_refuses_waves_the_planner_would_not_schedule(
        reserve_registry: Registry, tmp_path: Path, case: str, match: str) -> None:
    """spec.json "waves" is outside the pins; a resume replays the planner over the rows and
    refuses any recorded wave it would not have scheduled, before a single job runs."""
    no_zero = case in ("unplanned", "empty", "eager")
    zero = set() if no_zero else {"dev-dense-12-calm-00", "dev-dense-12-calm-r0"}
    mode = "eager" if case == "eager" else "on-demand"
    out = tmp_path / case
    _screen(reserve_registry, out, _zero_stub(zero), mode)
    edits: dict[str, Callable[[list[dict]], list[dict]]] = {
        "unplanned": lambda w: w + [_wave("dev-sparse_wide-12-calm-r3", 2)],
        "empty": lambda w: [{"wave": 2, "worlds": {}}],
        "eager": lambda w: [_wave("dev-dense-12-calm-r0", 2)],
        "renumbered": lambda w: [{**w[0], "wave": 5}, *w[1:]],
        "extra_key": lambda w: [{**w[0], "note": "x"}, *w[1:]],
        "wrong_grid": lambda w: [{"wave": 2, "worlds": {
            "dev-dense-12-calm-r0": {**w[0]["worlds"]["dev-dense-12-calm-r0"], "episodes": []}}},
            *w[1:]],
        "non_reserve": lambda w: w + [_wave("dev-sparse_wide-12-calm-00", 4)],
        "skipped_order": lambda w: [_wave("dev-dense-12-calm-r1", 2)],
    }
    _tamper(out, edits[case])
    before = (out / "episodes.csv").read_bytes()
    ran: list[dict] = []
    with pytest.raises(ValueError, match=match):
        _screen(reserve_registry, out, _zero_stub(zero, ran), mode)
    assert ran == [] and (out / "episodes.csv").read_bytes() == before


def test_resume_refuses_wave_after_incomplete_wave(reserve_registry: Registry,
                                                    tmp_path: Path) -> None:
    """A wave may only follow complete predecessors: after a crash inside wave 3, a recorded
    wave 4 cannot have been planned (only the last wave may be partial)."""
    zero = {"dev-dense-12-calm-00", "dev-dense-12-calm-r0"}
    out = tmp_path / "w"
    with pytest.raises(RuntimeError, match="simulated crash"):
        _screen(reserve_registry, out, _zero_stub(zero, crash_on="dev-dense-12-calm-r1"),
                "on-demand")
    _tamper(out, lambda w: w + [_wave("dev-dense-12-calm-r2", 4)])
    ran: list[dict] = []
    with pytest.raises(ValueError, match="waves before it are incomplete"):
        _screen(reserve_registry, out, _zero_stub(zero, ran), "on-demand")
    assert ran == []


def test_mode_change_refuses_resume(reserve_registry: Registry, tmp_path: Path) -> None:
    out = tmp_path / "m"
    _screen(reserve_registry, out, _zero_stub(set()), "on-demand")
    with pytest.raises(ValueError, match="different pins"):
        _screen(reserve_registry, out, _zero_stub(set()), "eager")


def test_plan_replacements_pending_order(reserve_registry: Registry) -> None:
    """Two zero gate slots in one group request the next two reserves at once (both are
    certain to be consulted). Once r0 proves zero and r1 usable, slot 1 takes r1 and slot 2 needs
    r2. All reserves zero -> exhausted."""
    from tokenomics.eval.registry import plan_replacements

    full = build_worlds_v1(SEED_ORDER)
    reg = Registry(name="two-slots", seed_ranges=full.seed_ranges, entries=tuple(
        full.entry(w) for w in ("dev-dense-12-calm-00", "dev-dense-12-calm-01",
                                *(f"dev-dense-12-calm-r{k}" for k in range(4)))))
    zero_gate = {"dev-dense-12-calm-00": 0.0, "dev-dense-12-calm-01": 0.0}
    plan = plan_replacements(reg, zero_gate)
    assert plan.pending == ("dev-dense-12-calm-r0", "dev-dense-12-calm-r1")
    assert plan.replacements == ()
    plan = plan_replacements(reg, {**zero_gate, "dev-dense-12-calm-r0": 0.0,
                                   "dev-dense-12-calm-r1": 3.0})
    assert [(r.replaced, r.by, r.skipped) for r in plan.replacements] == [
        ("dev-dense-12-calm-00", "dev-dense-12-calm-r1", ("dev-dense-12-calm-r0",))]
    assert plan.pending == ("dev-dense-12-calm-r2",)
    assert plan.consulted == ("dev-dense-12-calm-r0", "dev-dense-12-calm-r1")
    with pytest.raises(ValueError, match="no usable reserve left"):
        plan_replacements(reg, {**zero_gate, **{f"dev-dense-12-calm-r{k}": 0.0
                                                for k in range(4)}})


def test_cli_reserves_none_needs_dry_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--mode", "screen", "--reserves", "none", "--out-dir", str(tmp_path / "x")])


# --- dense-scale probe (task 10.2) ---


def test_probe_writes_per_scale_headroom_and_touches_only_selected_family(
        reserve_registry: Registry, tmp_path: Path) -> None:
    """One session, several scales: one subdir per scale with its own rows and headroom, a
    per-scale table, and job generator refs carrying the scale (omitted when neutral) on the
    selected family only."""
    from tokenomics.eval.gate_cli import run_probe

    ran: list[dict] = []
    table = run_probe(out_dir=tmp_path / "p", horizon=12, workers=1, registry=reserve_registry,
                      roster=ROSTER, scales=[1.0, 0.9], families=[DENSE], episodes_per_world=1,
                      run_job_fn=_zero_stub(set(), ran), log=lambda s: None)
    assert [(h.family, h.size) for h in table] == [(DENSE, 12), (DENSE, 12)]
    assert {j["world"] for j in ran} == {"dev-dense-12-calm-00"}  # gate worlds only, dense only
    scales = [j["generator"]["params"].get("capacity_scale") for j in ran]
    assert scales == [None] * len(ROSTER.members) + [0.9] * len(ROSTER.members)
    lines = (tmp_path / "p" / "probe_headroom.csv").read_text().splitlines()
    assert lines[0].startswith("capacity_scale,family,size,worlds,median_fill")
    assert [ln.split(",")[0] for ln in lines[1:]] == ["1.0", "0.9"]
    for sub in ("scale-1", "scale-0.9"):
        assert (tmp_path / "p" / sub / "headroom.csv").exists()
        spec = json.loads((tmp_path / "p" / sub / "spec.json").read_text())
        assert spec["kind"] == "probe" and spec["status"] == "complete"
    # rerun resumes: nothing left to run
    again: list[dict] = []
    run_probe(out_dir=tmp_path / "p", horizon=12, workers=1, registry=reserve_registry,
              roster=ROSTER, scales=[1.0, 0.9], families=[DENSE], episodes_per_world=1,
              run_job_fn=_zero_stub(set(), again), log=lambda s: None)
    assert again == []


@pytest.mark.parametrize("scales,match", [([], "non-empty"), ([0.9, 0.9], "distinct"),
                                          ([0.0], "capacity_scale")])
def test_probe_refuses_bad_scales(reserve_registry: Registry, tmp_path: Path, scales: list,
                                  match: str) -> None:
    from tokenomics.eval.gate_cli import run_probe

    with pytest.raises(ValueError, match=match):
        run_probe(out_dir=tmp_path / "x", horizon=12, workers=1, registry=reserve_registry,
                  roster=ROSTER, scales=scales, run_job_fn=_zero_stub(set()), log=lambda s: None)


def test_baseline_setters_are_the_argmax_members() -> None:
    """Per family: the members that are the per-episode argmax somewhere, sorted."""
    from tokenomics.eval.gate_cli import baseline_setters

    rows = [_row("dev-dense-12-calm-00", 1000, A, 9.0), _row("dev-dense-12-calm-00", 1000, B, 3.0),
            _row("dev-dense-12-calm-00", 1001, A, 2.0), _row("dev-dense-12-calm-00", 1001, B, 4.0),
            _row("dev-sparse_wide-12-calm-00", 1000, A, 1.0),
            _row("dev-sparse_wide-12-calm-00", 1000, B, 5.0)]
    assert baseline_setters(rows, ROSTER, DENSE) == sorted([A, B])
    assert baseline_setters(rows, ROSTER, SPARSE_WIDE) == [B]


def test_cli_probe_needs_scales_and_known_members(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--mode", "probe", "--out-dir", str(tmp_path / "a")])
    with pytest.raises(SystemExit):
        main(["--mode", "probe", "--scales", "0.9", "--members", "nope/none",
              "--out-dir", str(tmp_path / "b")])
    with pytest.raises(SystemExit):
        main(["--mode", "screen", "--members", A, "--out-dir", str(tmp_path / "c")])


def _probe(reg: Registry, out: Path, ran: list[dict], **kw: object) -> list:
    from tokenomics.eval.gate_cli import run_probe

    args: dict = dict(out_dir=out, horizon=12, workers=1, registry=reg, roster=ROSTER,
                      scales=[1.0, 0.9], families=[DENSE], episodes_per_world=1,
                      run_job_fn=_zero_stub(set(), ran), log=lambda s: None)
    args.update(kw)
    return run_probe(**args)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file()}


def test_probe_refuses_a_screen_root(reserve_registry: Registry, tmp_path: Path) -> None:
    """Review concern 113: a probe never writes into a screen/report run directory."""
    out = tmp_path / "screen"
    _screen(reserve_registry, out, _zero_stub(set()), "on-demand")
    before = _snapshot(out)
    ran: list[dict] = []
    with pytest.raises(ValueError, match="holds a screen/report run"):
        _probe(reserve_registry, out, ran)
    assert ran == [] and _snapshot(out) == before


@pytest.mark.parametrize("change", [
    {"scales": [0.8]},
    {"roster": _roster(("always_restore", "even_split"))},
    {"episodes_per_world": None},
    {"horizon": 8},
])
def test_probe_root_is_bound_to_its_configuration(reserve_registry: Registry, tmp_path: Path,
                                                  change: dict) -> None:
    """A rerun with other scales, members, episodes or horizon is refused before any job runs,
    so probe.json, probe_headroom.csv, RESULT.md and the scale subdirs never disagree."""
    out = tmp_path / "p"
    _probe(reserve_registry, out, [])
    before = _snapshot(out)
    ran: list[dict] = []
    with pytest.raises(ValueError, match="different configuration"):
        _probe(reserve_registry, out, ran, **change)
    assert ran == [] and _snapshot(out) == before


def test_probe_refuses_a_foreign_non_empty_root(reserve_registry: Registry,
                                                tmp_path: Path) -> None:
    out = tmp_path / "x"
    out.mkdir()
    (out / "notes.txt").write_text("keep")
    ran: list[dict] = []
    with pytest.raises(ValueError, match="not empty"):
        _probe(reserve_registry, out, ran)
    assert ran == [] and _snapshot(out) == {"notes.txt": b"keep"}


def test_v1_screen_frozen_files_byte_identical_to_c9c7da2(reserve_registry: Registry,
                                                          tmp_path: Path) -> None:
    """add-shocks-i-drift task 5.5 / D10: a v1 screen writes no `screened_on` and its frozen files
    are byte-identical to those base commit c9c7da2 wrote for the same injected rows (the
    on-demand fixture with one zero gate world, pruning on). sha256 taken by running this screen
    at c9c7da2."""
    import hashlib

    out = tmp_path / "v1"
    _, status = run_screen(out_dir=out, horizon=12, workers=1, registry=reserve_registry,
                           roster=ROSTER, episodes_per_world=1, dry_run=True, prune=True,
                           reserves="on-demand",
                           run_job_fn=_zero_stub({"dev-dense-12-calm-00", "dev-dense-12-calm-r0"}),
                           log=lambda s: None)
    assert status == 0
    got = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((out / "frozen").iterdir())}
    assert got == {
        "roster_v1.json": "9c9add0c0aa5fca770c283bf02fb35883fe0c8b7ca93527b9d849b5bb2586d5e",
        "side_effect_params_v1.json":
            "d7b9b50cf925a63a869dcc4cfd71a12208fbadaf9cb88af91922640e8bfba267",
        "worlds_v1.json": "e0557c1c6beb865dadffc285e9b4d04d34a970a7da0636a55679a1a52fbef15f",
    }
    for name in got:
        assert "screened_on" not in json.loads((out / "frozen" / name).read_text()), name
    assert "screened_on" not in json.loads((out / "spec.json").read_text())


# --- the two-wave v3 screen (add-gate-v3 task 3.1, design D3 / D4) ---

import functools  # noqa: E402

from tokenomics.eval.gate import row_from_dict  # noqa: E402
from tokenomics.eval.gate_cli import (  # noqa: E402
    build_jobs,
    job_key,
    plan_v3,
    routing_reading,
    routing_world,
    screen_outcome_v3,
)
from tokenomics.eval.registry import EXCLUDED, SEALED, build_worlds_v3  # noqa: E402
from tokenomics.eval.roster import NEVER_ARGMAX_TWO_WAVE_RULE, is_stochastic  # noqa: E402

EP_RR, EP_MS, EP_MWW = ("entity_planner/round_robin", "entity_planner/makeable_split",
                        "entity_planner/makeable_weighted_whole")
AR, SA = "always_restore/even_split", "stock_aware/proportional"


def _v3_roster(pairs: tuple[tuple[str, str], ...] = (
        ("entity_planner", "round_robin"), ("entity_planner", "makeable_split"),
        ("entity_planner", "makeable_weighted_whole"), ("always_restore", "even_split"),
        ("stock_aware", "proportional"))) -> Roster:
    return Roster(name="roster-v3-test", replicates=2, members=tuple(
        RosterMember(p, o, contract_for(p, o), is_stochastic(p, o)) for p, o in pairs))


@functools.cache
def _draft_v3() -> Registry:
    return build_worlds_v3()


Stub = Callable[[dict], dict]


def _stub(units: Callable[[dict], float] | None = None, *, dead: Callable[[dict], bool] = None,
          fill: Callable[[dict], float] = None) -> Stub:
    """A roster row per job: units / death / fill from the given functions of the job."""
    base = {EP_RR: 100.0, EP_MS: 106.0, EP_MWW: 90.0, AR: 50.0, SA: 101.0}

    def run(job: dict) -> dict:
        u = units(job) if units else base.get(job["subject"], 10.0) + job["replicate"]
        f = fill(job) if fill else 0.5
        return row_to_dict(EpisodeRow(
            world=job["world"], size=job["size"], regime=job["regime"], episode=job["episode"],
            subject=job["subject"], replicate=job["replicate"], units=float(u),
            dead=bool(dead(job)) if dead else False, demand_digest=f"d{job['episode']}",
            r_start=1.0, r_end=0.9, consumer_fill_rate={"c0": 0.7},
            ordered_units=100.0, fulfilled_units=100.0 * f))
    return run


def _drive(scope: Registry, roster: Roster, stub: Stub) -> list[EpisodeRow]:
    """Run exactly the jobs `plan_v3` schedules, as the driver does, until it is complete."""
    rows: list[EpisodeRow] = []
    while True:
        plan = plan_v3(scope, roster, rows)
        have = {job_key(r.world, r.episode, r.subject, r.replicate) for r in rows}
        for e, ros in plan.jobs:
            for j in build_jobs([e], ros, None):
                if job_key(j["world"], j["episode"], j["subject"], j["replicate"]) not in have:
                    rows.append(row_from_dict(run_job_stub(stub, j)))
        if plan.complete:
            return rows


def run_job_stub(stub: Stub, job: dict) -> dict:
    return stub(job)


def _outcome(stub: Stub, roster: Roster | None = None, scope: Registry | None = None):
    scope = scope or _draft_v3()
    roster = roster or _v3_roster()
    rows = _drive(scope, roster, stub)
    return screen_outcome_v3(scope, roster, rows, q=0.05, tag=3, screened_on=scope.version,
                             dry_run=True), rows


def test_v3_wave2_runs_w2_and_carries_a_non_ep_winner() -> None:
    """gate-harness "Wave-1 winner carried to wave 2": stock_aware wins one wave-1 episode, so it
    runs on every ws_wide world and is kept; every entity_planner member runs in wave 2 whether
    or not it won; always_restore never wins and is dropped as not carried."""
    one = {"done": False}

    def units(job: dict) -> float:
        if job["subject"] == SA and job["world"] == "dev-sparse_wide-42-calm-00" \
                and job["episode"] == _draft_v3().entry(job["world"]).episode_seeds[0]:
            return 500.0
        return {EP_RR: 100.0, EP_MS: 106.0, EP_MWW: 90.0, AR: 50.0, SA: 40.0}[job["subject"]]

    out, rows = _outcome(_stub(units))
    assert out.plan.w2 == (EP_MS, EP_MWW, EP_RR, SA)
    wide = {r.subject for r in rows if r.world.startswith("dev-ws_wide-")}
    assert wide == set(out.plan.w2) and AR not in wide
    assert out.roster.member_ids == (EP_MS, SA)  # MS wins elsewhere, SA its one episode
    rec = out.roster.pruning
    assert rec.rule == NEVER_ARGMAX_TWO_WAVE_RULE and rec.not_carried == (AR,)
    assert [w.members for w in rec.waves] == [_v3_roster().member_ids, out.plan.w2]
    assert all(w.rows_digest for w in rec.waves) and rec.waves[1].episodes
    assert {e.world for e in rec.waves[1].episodes} <= {
        e.id for e in out.registry.gate_worlds(DEVELOPMENT) if e.family == "ws_wide"}
    assert one == {"done": False}


def test_v3_bar_death_replaces_region_worlds_in_reserve_order() -> None:
    """gate-harness "Region worlds whose best baseline dies are replaced": the argmax member dies
    on a ws_loaded world and on its first reserve, so the second reserve replaces it (the first
    skipped); the record names world, episode and member. A sparse_wide world whose argmax dies
    keeps its v2 rule (not replaced)."""
    scope = _draft_v3()
    bad = {"dev-ws_loaded-12-calm-00", "dev-ws_loaded-12-calm-r0", "dev-sparse_wide-12-calm-00"}

    def dead(job: dict) -> bool:
        return job["world"] in bad and job["subject"] == EP_MS  # MS is every world's argmax

    out, _ = _outcome(_stub(dead=dead))
    reps = {r.replaced: r for r in out.registry.screening.replacements}
    assert set(reps) == {"dev-ws_loaded-12-calm-00"}
    r = reps["dev-ws_loaded-12-calm-00"]
    assert (r.by, r.skipped) == ("dev-ws_loaded-12-calm-r1", ("dev-ws_loaded-12-calm-r0",))
    first = scope.entry("dev-ws_loaded-12-calm-00").episode_seeds[0]
    assert ("dev-ws_loaded-12-calm-00", first, EP_MS) in out.registry.screening.bar_deaths
    assert out.registry.entry("dev-sparse_wide-12-calm-00").status == GATE
    assert out.plan.wave1.sub_waves == (("dev-ws_loaded-12-calm-r0",),
                                        ("dev-ws_loaded-12-calm-r1",))


def _fill_by_family(fills: dict[tuple[str, int], float]) -> Callable[[dict], float]:
    scope = _draft_v3()
    return lambda job: fills.get((scope.entry(job["world"]).family, job["size"]), 0.5)


def test_v3_one_family_excluded_both_splits_others_proceed() -> None:
    """gate-harness "One family fails under v3", "Exclusion covers both splits", "Recorded
    exclusion passes admission", "Excluded rows do not shape the frozen files"."""

    def units(job: dict) -> float:  # always_restore wins only on the excluded sparse_wide/12
        if job["subject"] == AR and job["size"] == 12 and job["world"].startswith(
                "dev-sparse_wide"):
            return 999.0
        return {EP_RR: 100.0, EP_MS: 106.0, EP_MWW: 90.0, AR: 50.0, SA: 40.0}[job["subject"]]

    out, rows = _outcome(_stub(units, fill=_fill_by_family({("sparse_wide", 12): 0.96})))
    assert out.passed and out.plan.excluded == {("sparse_wide", 12)}
    assert out.plan.wave2 is not None  # wave 2 still runs
    sw12 = [e for e in out.registry.entries if (e.family, e.size) == ("sparse_wide", 12)]
    assert {e.split for e in sw12} == {DEVELOPMENT, SEALED}
    assert all(e.status == EXCLUDED for e in sw12)
    assert out.registry.admission_problems(DEVELOPMENT) == []
    assert out.registry.coverage_problems() == []
    assert AR not in out.plan.w2 and AR not in out.roster.member_ids  # rows of excluded worlds
    fams = {c.family: c for c in out.calibrations}
    assert set(fams) == {"sparse_wide", "ws_loaded", "ws_wide"}
    sw_eps = {(r.world, r.episode) for r in rows if r.world.startswith("dev-sparse_wide-42")}
    assert fams["sparse_wide"].calibration.episodes == len(sw_eps)  # 42 only, no 12-producer
    assert json.loads(out.params_json)["evidence"]["excluded"] == ["sparse_wide/12"]


def test_v3_size_without_an_admitted_family_stops_before_wave_2() -> None:
    """gate-harness "A size without an admitted family": both 12-producer families fail wave 1,
    so the screen stops (exit 3) before wave 2 runs and freezes nothing."""
    out, rows = _outcome(_stub(fill=_fill_by_family({("sparse_wide", 12): 0.96,
                                                      ("ws_loaded", 12): 0.97})))
    assert not out.passed and "12 producers" in out.plan.stop
    assert out.plan.wave2 is None and not any(r.world.startswith("dev-ws_wide") for r in rows)
    assert out.roster is None and out.params_json is None


def test_v3_attainability_flag_is_reported_only() -> None:
    """gate-harness "Shocked margin out of reach": a perfect-fill ratio below the margin is
    flagged unattainable, in the report only."""
    fill = _fill_by_family({("sparse_wide", 12): 0.94})  # 1 / 0.94 = 1.064: calm ok, shocked not
    out, _ = _outcome(_stub(fill=fill))
    cells = {(c["size"], c["regime"]): c for c in out.tables["attainability"]["by_size_regime"]}
    by_fam = {(c["family"], c["size"], c["regime"]): c
              for c in out.tables["attainability"]["by_family"]}
    assert by_fam[("sparse_wide", 12, "shocked")]["unattainable"]
    assert not by_fam[("sparse_wide", 12, "calm")]["unattainable"]
    assert by_fam[("ws_loaded", 12, "calm")]["perfect_fill_ratio"] == pytest.approx(2.0)
    assert not cells[(12, "calm")]["unattainable"]
    assert out.passed  # the flag changes nothing


ROUTERS = ("blend_0.5", "even_split", "obs_price", "proportional", "random", "shock_reactive",
           "round_robin", "even_split_rotating", "random_whole", "makeable_weighted_whole",
           "makeable_split")
EP11 = _v3_roster(tuple(("entity_planner", o) for o in ROUTERS))


def _routing_rows(world: str, regime: str, units: dict[str, list[list[float]]]
                  ) -> list[EpisodeRow]:
    """units[ordering] = per episode, per replicate units (EP11's R = 2 for stochastic ones)."""
    rows = []
    for o in ROUTERS:
        mid = f"entity_planner/{o}"
        for ep, reps in enumerate(units.get(o, [[50.0, 50.0]] * 2)):
            n = 2 if EP11.member(mid).random else 1
            for r in range(n):
                rows.append(_row(world, 100 + ep, mid, reps[r % len(reps)], replicate=r,
                                 regime=regime))
    return rows


def _entry(world: str, regime: str, family: str = "ws_loaded"):
    from tokenomics.eval.registry import WorldEntry

    return WorldEntry(id=world, split=DEVELOPMENT, family=family, size=12, regime=regime,
                      world_seed=1, episode_seeds=(100, 101), generator={})


def test_routing_deterministic_sides_on_a_calm_world_count() -> None:
    """gate-harness "Deterministic sides on a calm world": makeable_split beats round_robin by 6%,
    both deterministic: the floor is 0 and the world counts."""
    u = {"round_robin": [[100.0]] * 2, "makeable_split": [[106.0]] * 2,
         "makeable_weighted_whole": [[60.0, 60.0]] * 2, "random_whole": [[60.0, 60.0]] * 2,
         "random": [[60.0, 60.0]] * 2}
    r = routing_world(_entry("w", CALM), EP11, _routing_rows("w", CALM, u))
    assert (r["input_aware"], r["blind"], r["floor"], r["floor_type"]) == (
        "makeable_split", "round_robin", 0.0, "calm_deterministic")
    assert r["reach_gain"] == pytest.approx(0.06) and r["counts"]


def test_routing_poor_routers_do_not_make_routing_matter() -> None:
    """gate-harness "Poor routers do not make routing matter": even_split at half of round_robin,
    the input-aware routers within 0.02 of it: the family reads routing_flat."""
    worlds = {}
    rows = []
    for i in range(3):
        w = f"w{i}"
        u = {"round_robin": [[100.0]] * 2, "even_split": [[50.0]] * 2,
             "makeable_split": [[101.5]] * 2, "makeable_weighted_whole": [[101.0, 101.0]] * 2}
        rows += _routing_rows(w, CALM, u)
        worlds[w] = (_entry(w, CALM), EP11)
    res = routing_reading(worlds, rows, set())
    assert [f["reading"] for f in res["families"]] == ["routing_flat"]
    assert all(r["router_spread"] >= 0.49 and not r["counts"] for r in res["worlds"])


def test_routing_family_rule_one_third_and_floors() -> None:
    """gate-harness "Flat region family": 1 counting world of 6 reads routing_flat, 2 of 6
    routing_matters. A shocked world's floor is 3·sd(d)/√n; a calm world with a stochastic side
    uses that side's run spread; one shocked episode leaves the floor undefined (no count)."""
    worlds, rows = {}, []
    for i in range(6):
        w = f"w{i}"
        gain = 110.0 if i < 2 else 100.0
        rows += _routing_rows(w, CALM, {"round_robin": [[100.0]] * 2,
                                         "makeable_split": [[gain]] * 2})
        worlds[w] = (_entry(w, CALM), EP11)
    fam = routing_reading(worlds, rows, set())["families"][0]
    assert (fam["counting_worlds"], fam["reading"]) == (2, "routing_matters")
    one = {k: v for k, v in worlds.items() if k != "w1"}
    assert routing_reading(one, rows, set())["families"][0]["reading"] == "routing_flat"
    # shocked floor: d = (10, 30) -> sd 14.14, floor 3 * 14.14 / sqrt(2) = 30 > mean diff 20
    s = _routing_rows("s", SHOCKED, {"round_robin": [[100.0], [100.0]],
                                     "makeable_split": [[110.0], [130.0]]})
    r = routing_world(_entry("s", SHOCKED), EP11, s)
    assert r["floor_type"] == "shocked" and r["floor"] == pytest.approx(30.0)
    assert r["reach_gain"] == pytest.approx(0.2) and not r["counts"]
    # calm, stochastic side: MWW runs 110, 120, 110, 120 -> s 5.77, n 4: floor 3 * 5.77 / 2
    c = _routing_rows("c", CALM, {"round_robin": [[100.0]] * 2,
                                  "makeable_weighted_whole": [[110.0, 120.0]] * 2})
    r = routing_world(_entry("c", CALM), EP11, c)
    assert r["floor_type"] == "calm_stochastic"
    assert r["floor"] == pytest.approx(3 * (100 / 3 / 4) ** 0.5) and r["counts"]
    single = _routing_rows("o", SHOCKED, {"round_robin": [[100.0]],
                                          "makeable_split": [[150.0]]})
    single = [x for x in single if x.episode == 100]
    r = routing_world(_entry("o", SHOCKED), EP11, single)
    assert r["floor"] is None and not r["counts"]


def test_v3_screen_refusals(tmp_path: Path) -> None:
    from tokenomics.eval.registry import build_worlds_v2
    from tokenomics.eval.roster import PILOT_V3_NAME, build_pilot_roster_v3

    v3 = _draft_v3()

    def never(job: dict) -> dict:  # a path that is not refused fails fast instead of playing
        raise AssertionError(f"a refused screen ran a job: {job['world']}")

    with pytest.raises(ValueError, match="pilot roster"):
        run_screen(out_dir=tmp_path / "a", horizon=12, workers=1, registry=v3,
                   roster=build_pilot_roster_v3(), member_waves="v3", run_job_fn=never, log=lambda s: None)
    assert build_pilot_roster_v3().name == PILOT_V3_NAME
    with pytest.raises(ValueError, match="needs --member-waves v3"):
        run_screen(out_dir=tmp_path / "b", horizon=12, workers=1, registry=v3,
                   roster=_v3_roster(), dry_run=True, run_job_fn=never, log=lambda s: None)
    with pytest.raises(ValueError, match="one wave"):
        run_screen(out_dir=tmp_path / "c", horizon=12, workers=1, registry=build_worlds_v2(),
                   roster=_v3_roster(), dry_run=True, member_waves="v3", run_job_fn=never, log=lambda s: None)
    with pytest.raises(ValueError, match="belong to the two-wave screen"):
        run_screen(out_dir=tmp_path / "d", horizon=12, workers=1, registry=build_worlds_v2(),
                   roster=_v3_roster(), dry_run=True, region_shocked_episodes=5,
                   run_job_fn=never, log=lambda s: None)
    with pytest.raises(ValueError, match="in a dry run"):
        run_screen(out_dir=tmp_path / "e", horizon=12, workers=1, registry=v3,
                   roster=_v3_roster(), member_waves="v3", shock_profile="shocks-i-smoke",
                   run_job_fn=never, log=lambda s: None)
    assert not any((tmp_path / x).exists() for x in "abcde")


def test_v3_driver_smoke_profile_reductions_and_outputs(tmp_path: Path) -> None:
    """A dry-run two-wave screen with a stub: shocked entries run in their smoke form, D3
    reduction 1 cuts shocked region worlds to 5 episodes in the screened registry, the waves,
    attainability and routing tables are written, and the frozen v3 files carry one
    screened_on and dry-run names."""
    seen: list[dict] = []
    stub = _stub()

    def run(job: dict) -> dict:
        seen.append(job)
        return stub(job)

    out = tmp_path / "s"
    _, status = run_screen(out_dir=out, horizon=12, workers=1, registry=_draft_v3(),
                           roster=_v3_roster(), dry_run=True, member_waves="v3",
                           worlds_per_family=1, shock_profile="shocks-i-smoke",
                           region_shocked_episodes=5, run_job_fn=run, log=lambda s: None)
    assert status == 0
    shocked = [j for j in seen if j["regime"] == "shocked"]
    assert shocked and {j["generator"]["params"]["shock_profile"] for j in shocked} == {
        "shocks-i-smoke"}
    assert {(j["world"].split("-")[1], j["regime"]) for j in seen} >= {
        ("sparse_wide", "shocked"), ("ws_loaded", "shocked"), ("ws_wide", "shocked")}
    rep = json.loads((out / "screening.json").read_text())
    assert [w["wave"] for w in rep["waves"]] == [1, 2] and rep["acceptance_read"] is False
    assert rep["note"] == "not an acceptance read"
    assert rep["attainability"]["by_size_regime"] and rep["routing"]["families"]
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["region_shocked_episodes"] == 5
    assert spec["pins"]["shock_profile"] == "shocks-i-smoke"
    frozen = {p.name: json.loads(p.read_text()) for p in (out / "frozen").iterdir()}
    assert set(frozen) == {"worlds_v3.json", "roster_v3.json", "side_effect_params_v3.json"}
    assert {d["screened_on"] for d in frozen.values()} == {_draft_v3().version}
    assert frozen["roster_v3.json"]["name"] == "roster-v3-dry-run"
    assert frozen["side_effect_params_v3.json"]["name"] == "side-effect-params-v3-dry-run"
    region_shocked = [e for e in frozen["worlds_v3.json"]["entries"]
                      if e["family"].startswith("ws_") and e["regime"] == "shocked"]
    assert region_shocked and all(len(e["episode_seeds"]) == 5 for e in region_shocked)
    # a rerun on the same out dir resumes from the rows and runs nothing new
    again: list[dict] = []
    run_screen(out_dir=out, horizon=12, workers=1, registry=_draft_v3(), roster=_v3_roster(),
               dry_run=True, member_waves="v3", worlds_per_family=1,
               shock_profile="shocks-i-smoke", region_shocked_episodes=5,
               run_job_fn=lambda j: again.append(j) or stub(j), log=lambda s: None)
    assert again == []


def _source_pins(**edits: object) -> dict:
    """The pins of a real horizon-100 screen of the registry v2 / roster v2 drafts (EXP-0047)."""
    from tokenomics.eval.registry import build_worlds_v2
    from tokenomics.eval.roster import build_roster_v2

    return {"registry_version": build_worlds_v2().version,
            "roster_version": build_roster_v2().version, "horizon": 100, "dry_run": False,
            **edits}


def test_v3_reuse_rows_after_an_exact_reproduction(tmp_path: Path) -> None:
    """D3 reduction 2: v2-world rows of the v2-draft members come from the EXP-0047 dir after its
    reproduction check matches; a single differing row refuses the reuse."""
    from tokenomics.eval.gate_cli import read_rows
    from tokenomics.eval.registry import build_worlds_v2
    from tokenomics.eval.roster import build_roster_v2, build_roster_v3

    roster = build_roster_v3()
    stub = _stub(units=lambda j: 10.0 + len(j["subject"]) % 7 + j["replicate"])
    v2ws = [_draft_v3().entry(w) for w in ("dev-sparse_wide-12-calm-00",
                                           "dev-sparse_wide-12-calm-01",
                                           "dev-sparse_wide-12-shocked-00")]
    v2w = v2ws[2]
    src = tmp_path / "exp47"
    src.mkdir()
    v2_roster = build_roster_v2()
    (src / "spec.json").write_text(json.dumps({"pins": _source_pins()}))
    rows = [stub(j) for j in build_jobs(v2ws, v2_roster, None)]
    with (src / "episodes.csv").open("w") as fh:
        from tokenomics.eval.gate import rows_to_csv

        fh.write(rows_to_csv([row_from_dict(r) for r in rows]))
    seen: list[dict] = []
    out = tmp_path / "r"
    _, status = run_screen(out_dir=out, horizon=100, workers=1, registry=_draft_v3(),
                           roster=roster, dry_run=True, member_waves="v3", worlds_per_family=1,
                           families=["sparse_wide", "ws_wide"],
                           reuse_rows=str(src), run_job_fn=lambda j: seen.append(j) or stub(j),
                           log=lambda s: None)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["reuse"]["status"] == "match" and spec["reuse"]["reproduced_rows"] == 200
    v2_ids = set(v2_roster.member_ids)
    ran = {(j["world"], j["episode"]) for j in seen if j["world"].startswith("dev-sparse_wide")
           and j["subject"] in v2_ids}
    # only the reproduction world-episodes ran: one per (size, regime) = calm-00 and shocked-00
    assert ran == {("dev-sparse_wide-12-calm-00", v2ws[0].episode_seeds[0]),
                   (v2w.id, v2w.episode_seeds[0])}
    assert spec["reuse"]["reused_rows"] == len(rows) - 200
    assert len(read_rows(out / "episodes.csv")) > 0 and status in (0, 3)
    bad = tmp_path / "exp47b"
    bad.mkdir()
    (bad / "spec.json").write_text((src / "spec.json").read_text())
    rows = [dict(r, units="1234.0") if (r["world"], r["subject"], r["replicate"]) == (
        "dev-sparse_wide-12-calm-00", "always_both/even_split", "0") and int(r["episode"]) ==
        v2ws[0].episode_seeds[0] else r for r in rows]
    with (bad / "episodes.csv").open("w") as fh:
        fh.write(rows_to_csv([row_from_dict(r) for r in rows]))
    with pytest.raises(ValueError, match="reproduction check failed"):
        run_screen(out_dir=tmp_path / "r2", horizon=100, workers=1, registry=_draft_v3(),
                   roster=roster, dry_run=True, member_waves="v3", worlds_per_family=1,
                   families=["sparse_wide", "ws_wide"],
                   reuse_rows=str(bad), run_job_fn=stub, log=lambda s: None)


def _reuse_source(tmp_path: Path, name: str, *, pins: dict | None = None,
                  edit: Callable[[dict], dict] | None = None) -> tuple[Path, Stub]:
    """A fake EXP-0047 dir: rows of the v2-draft members on the two 12-producer calm worlds and
    the shocked one `worlds_per_family=1` selects, from a deterministic stub; `edit` changes the
    reproduction row of always_both/even_split on calm-00's first episode."""
    from tokenomics.eval.gate import rows_to_csv
    from tokenomics.eval.roster import build_roster_v2

    stub = _stub(units=lambda j: 10.0 + len(j["subject"]) % 7 + j["replicate"])
    v2ws = [_draft_v3().entry(w) for w in ("dev-sparse_wide-12-calm-00",
                                           "dev-sparse_wide-12-calm-01",
                                           "dev-sparse_wide-12-shocked-00")]
    rows = [stub(j) for j in build_jobs(v2ws, build_roster_v2(), None)]
    target = ("dev-sparse_wide-12-calm-00", "always_both/even_split", "0",
              str(v2ws[0].episode_seeds[0]))
    if edit is not None:
        rows = [edit(dict(r)) if (r["world"], r["subject"], r["replicate"], r["episode"]) ==
                target else r for r in rows]
    src = tmp_path / name
    src.mkdir()
    (src / "spec.json").write_text(json.dumps({"pins": pins or _source_pins()}))
    (src / "episodes.csv").write_text(rows_to_csv([row_from_dict(r) for r in rows]))
    return src, stub


def _screen_with_reuse(tmp_path: Path, name: str, src: Path, stub: Stub) -> None:
    from tokenomics.eval.roster import build_roster_v3

    run_screen(out_dir=tmp_path / name, horizon=100, workers=1, registry=_draft_v3(),
               roster=build_roster_v3(), dry_run=True, member_waves="v3", worlds_per_family=1,
               families=["sparse_wide", "ws_wide"], reuse_rows=str(src), run_job_fn=stub,
               log=lambda s: None)


@pytest.mark.parametrize("field, value", [
    ("dead", "true"),                                # a death
    ("r_end", "0.123"),                              # run-down
    ("consumer_fill_rate", json.dumps({"c0": 0.1})),  # a consumer's fill rate
    ("fulfilled_units", "7.0"),                      # the headroom fill
])
def test_v3_reuse_refused_on_any_differing_field(tmp_path: Path, field: str, value: str) -> None:
    """The reproduction compares whole rows: a reproduction row of EXP-0047 that differs only in
    a non-units field (deaths, run-down, fill rates feed headroom and X / Y) refuses the reuse."""
    src, stub = _reuse_source(tmp_path, "src", edit=lambda r: {**r, field: value})
    with pytest.raises(ValueError, match="reproduction check failed on 1 of 200 rows"):
        _screen_with_reuse(tmp_path, "out", src, stub)


@pytest.mark.parametrize("pins, bad", [
    ({"horizon": 12}, "horizon"), ({"dry_run": True}, "dry_run"),
    ({"roster_version": "f" * 64}, "roster_version"),
    ({"registry_version": "e" * 64}, "registry_version"),
])
def test_v3_reuse_refused_on_a_source_that_is_not_exp0047s_kind(tmp_path: Path, pins: dict,
                                                                bad: str) -> None:
    src, stub = _reuse_source(tmp_path, "src", pins=_source_pins(**pins))
    with pytest.raises(ValueError, match=rf"not a real horizon-100 screen.*{bad}"):
        _screen_with_reuse(tmp_path, "out", src, stub)


def test_v3_bar_death_in_a_later_replicate_of_a_stochastic_argmax() -> None:
    """A death has no mean: the stochastic argmax member (makeable_weighted_whole, R = 2) dies in
    replicate 1 only on a ws_loaded world, and the world is replaced by its first reserve; the
    record names world, episode and member."""
    scope = _draft_v3()
    world = "dev-ws_loaded-12-calm-00"

    def units(job: dict) -> float:
        if job["subject"] == EP_MWW and job["world"] == world:
            return 500.0 + job["replicate"]  # the argmax on this world
        return {EP_RR: 100.0, EP_MS: 106.0, EP_MWW: 90.0, AR: 50.0, SA: 40.0}[job["subject"]]

    first = scope.entry(world).episode_seeds[0]

    def dead(job: dict) -> bool:
        return (job["world"], job["subject"], job["episode"], job["replicate"]) == (
            world, EP_MWW, first, 1)

    out, rows = _outcome(_stub(units, dead=dead))
    assert any(r.dead and r.replicate == 1 for r in rows)
    assert not any(r.dead and r.replicate == 0 for r in rows)
    reps = {r.replaced: r for r in out.registry.screening.replacements}
    assert reps[world].by == "dev-ws_loaded-12-calm-r0"
    assert (world, first, EP_MWW) in out.registry.screening.bar_deaths
