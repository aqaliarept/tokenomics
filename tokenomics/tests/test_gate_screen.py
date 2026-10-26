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
        rows += _headroom_rows(f"sat-{i}", f, size=120)
        family[f"sat-{i}"] = SPARSE_WIDE
    for i, f in enumerate([0.50, 0.95, 0.70]):
        rows += _headroom_rows(f"ok-{i}", f)
        family[f"ok-{i}"] = DENSE
    for i, f in enumerate([0.95, 0.95, 0.96]):
        rows += _headroom_rows(f"edge-{i}", f)
        family[f"edge-{i}"] = SPARSE_WIDE
    res = {(h.family, h.size): h for h in headroom(episode_bests(rows, ROSTER), family, 0.95)}
    assert res[(SPARSE_WIDE, 120)].median_fill == pytest.approx(0.98)
    assert not res[(SPARSE_WIDE, 120)].passed
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
    reg = build_worlds_v1()
    zero = "dev-dense-120-shocked-02"
    out = screen_outcome(reg, ROSTER, _dev_rows(reg, zero={zero}), q=0.05, prune=False)
    assert out.passed
    assert out.registry.entry(zero).status == REPLACED
    assert out.registry.entry("dev-dense-120-shocked-r0").status == GATE
    assert [(r.replaced, r.by) for r in out.registry.screening.replacements] == [
        (zero, "dev-dense-120-shocked-r0")]
    assert out.registry.screening.splits == (DEVELOPMENT,)
    assert out.registry.screening.headroom is not None


def test_saturated_screen_writes_no_calibration() -> None:
    reg = build_worlds_v1()
    out = screen_outcome(reg, ROSTER, _dev_rows(reg, fill=0.99), q=0.05, prune=False)
    assert not out.passed
    assert out.roster is None and out.params_json is None and out.calibration is None
    assert all(not h.passed for h in out.headroom)


def test_screen_outcome_calibrates_and_freezes() -> None:
    """X/Y from the argmax member (A: r_end 0.8, c0 fill 0.6), never from the loser (B)."""
    reg = build_worlds_v1()
    out = screen_outcome(reg, ROSTER, _dev_rows(reg), q=0.05, prune=False)
    assert out.roster is not None and out.roster.frozen
    p = params_from_dict(json.loads(out.params_json))
    assert (p.x, p.y) == (pytest.approx(0.8), pytest.approx(0.6))
    d = json.loads(out.params_json)
    assert d["method"] == "nearest-rank" and d["q"] == 0.05
    assert d["evidence"]["roster_version"] == out.roster.version
    assert d["evidence"]["registry_version"] == out.registry.version


def test_screen_outcome_prunes_never_winning_member() -> None:
    reg = build_worlds_v1()
    out = screen_outcome(reg, ROSTER, _dev_rows(reg), q=0.05, prune=True)
    assert out.roster is not None
    assert out.roster.member_ids == (A,) and out.roster.pruning.dropped == (B,)
    assert out.roster.frozen


# --- e2e: `--mode screen --dry-run` on a fixture registry ---

WORLDS = ("dev-dense-12-calm-00", "dev-sparse_wide-12-calm-00")


@pytest.fixture(scope="module")
def fixture_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("screen-inputs")
    full = build_worlds_v1()
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
                                         r"no development gate worlds for dense/120"):
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

    reg = build_worlds_v1()
    assert "has not screened the development split" in reg.admission_problems(DEVELOPMENT)[0]
    rows = _dev_rows(reg)
    dev = screen_zero_baselines(reg, ROSTER, rows)
    assert "records no development headroom check" in dev.admission_problems(DEVELOPMENT)[0]
    groups = sorted({(e.family, e.size) for e in dev.gate_worlds(DEVELOPMENT)})
    ok = [Headroom(f, z, 8, 0.5, 0.95, True) for f, z in groups]
    assert record_headroom(dev, ok).admission_problems(DEVELOPMENT) == []
    assert "no headroom result for ('sparse_wide', 120)" in record_headroom(
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
    # a registry without 120-producer development worlds is never admitted
    small = Registry(name="small", seed_ranges=reg.seed_ranges,
                     entries=tuple(e for e in reg.entries if e.size == 12))
    small_dev = screen_zero_baselines(small, ROSTER, [r for r in rows if r.size == 12])
    small_ok = record_headroom(small_dev, [h for h in ok if h.size == 12])
    assert small_ok.admission_problems(DEVELOPMENT) == [
        "no development gate worlds for dense/120", "no development gate worlds for sparse_wide/120"]
    # a family present only in the sealed split has no development evidence at all
    sealed_family = Registry(name="sf", seed_ranges=reg.seed_ranges, entries=tuple(
        e for e in reg.entries if e.family == DENSE or e.split == SEALED))
    sf = screen_zero_baselines(sealed_family, ROSTER,
                               [r for r in rows if r.world.startswith("dev-dense")])
    sf_ok = record_headroom(sf, [h for h in ok if h.family == DENSE])
    assert sf_ok.admission_problems(DEVELOPMENT) == [
        "no development gate worlds for sparse_wide/12",
        "no development gate worlds for sparse_wide/120"]


def test_inconsistent_headroom_refused() -> None:
    """A pass flag that does not follow from median < ceiling is refused when recorded and when a
    hand-edited file is loaded."""
    from tokenomics.eval.gate import Headroom
    from tokenomics.eval.registry import record_headroom, registry_from_dict, screen_zero_baselines

    reg = build_worlds_v1()
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
