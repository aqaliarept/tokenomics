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
