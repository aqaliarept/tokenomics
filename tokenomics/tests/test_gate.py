"""Gate arithmetic and verdict (design D6, add-measurement-gate-harness; gate-harness
"Hindsight-baseline arithmetic", "Gate report contents", "Verdict is reproducible from rows";
agent-acceptance clauses). The worked example in ARCHITECTURE.md §7.2 uses exactly the numbers of
`test_per_episode_max_differs_from_max_of_means` and `test_worlds_weigh_equally`."""

from __future__ import annotations

import json

import pytest

from tokenomics.eval.gate import (
    ACCEPT,
    CALM,
    CALM_CLAUSE,
    CANDIDATE,
    CSV_COLUMNS,
    NO_LOSING_CLAUSE,
    REJECT,
    RUN_DOWN_CLAUSE,
    SHOCKED,
    SHOCKED_CLAUSE,
    STARVATION_CLAUSE,
    ZERO_DEATH_CLAUSE,
    DigestMismatch,
    EpisodeRow,
    GateReport,
    SideEffectParams,
    ZeroBaselineWorld,
    compute_report,
    episode_baseline,
    recompute_from_csv,
    regime_ratio,
    row_from_dict,
    row_from_stats,
    row_to_dict,
    rows_from_csv,
    rows_to_csv,
    world_baseline,
    world_ratio,
)
from tokenomics.eval.roster import GATE_SIZES, Roster, RosterMember, contract_for, member_id

A = member_id("always_restore", "even_split")
B = member_id("restore_when_low", "proportional")
RND = member_id("random", "even_split")
PARAMS = SideEffectParams(x=0.5, y=0.25, version="params-test")
REG = "registry-test"


def _member(producing: str, ordering: str) -> RosterMember:
    return RosterMember(producing, ordering, contract_for(producing, ordering),
                        "random" in (producing, ordering))


ROSTER_AB = Roster(name="t", replicates=3, members=(
    _member("always_restore", "even_split"), _member("restore_when_low", "proportional")))
ROSTER_ABR = Roster(name="t", replicates=3, members=(
    *ROSTER_AB.members, _member("random", "even_split")))


def _row(world: str, episode: int, subject: str, units: float, *, replicate: int = 0,
         regime: str = CALM, size: int = 12, dead: bool = False, r_start: float | None = 0.8,
         r_end: float | None = 0.8, fill: dict[str, float | None] | None = None,
         digest: str | None = None, **monitored: float | None) -> EpisodeRow:
    return EpisodeRow(
        world=world, size=size, regime=regime, episode=episode, subject=subject,
        replicate=replicate, units=units, dead=dead,
        demand_digest=digest if digest is not None else f"d-{world}-{episode}",
        r_start=r_start, r_end=r_end,
        consumer_fill_rate=fill if fill is not None else {"c0": 1.0, "c1": 0.5},
        **monitored,
    )


def _world(world: str, a: list[float], b: list[float], cand: list[float], *,
           regime: str = CALM, size: int = 12, first_episode: int = 1000,
           **cand_kw: object) -> list[EpisodeRow]:
    """Rows of a world on which ROSTER_AB members A and B and the candidate deliver the given
    per-episode units."""
    rows: list[EpisodeRow] = []
    for i, (ua, ub, uc) in enumerate(zip(a, b, cand, strict=True)):
        ep = first_episode + i
        rows.append(_row(world, ep, A, ua, regime=regime, size=size))
        rows.append(_row(world, ep, B, ub, regime=regime, size=size))
        rows.append(_row(world, ep, CANDIDATE, uc, regime=regime, size=size, **cand_kw))  # type: ignore[arg-type]
    return rows


SMALL = (12,)  # fixture gate sizes: most tests judge one size; the default is GATE_SIZES


def _report(rows: list[EpisodeRow], roster: Roster = ROSTER_AB,
            params: SideEffectParams = PARAMS,
            gate_sizes: tuple[int, ...] = SMALL) -> GateReport:
    return compute_report(rows, roster, params, REG, gate_sizes)


def _passing(**cand_kw: object) -> list[EpisodeRow]:
    """A calm and a shocked world at size 12, both margins passing (1.2 and 1.2)."""
    return (_world("calm-1", [10, 10], [8, 8], [12, 12], **cand_kw)
            + _world("shock-1", [10, 10], [5, 5], [12, 12], regime=SHOCKED, **cand_kw))


# --- Hindsight-baseline arithmetic ---


def test_per_episode_max_differs_from_max_of_means() -> None:
    """A: 10, 4; B: 6, 8; candidate 9, 9 → per-episode baselines 10, 8 → world baseline 9 (the
    max of means would be max(7, 7) = 7), world ratio 1.0. Same numbers as ARCHITECTURE §7.2."""
    assert [episode_baseline({A: 10, B: 6}), episode_baseline({A: 4, B: 8})] == [10, 8]
    assert world_baseline([10, 8]) == 9
    assert world_ratio(9, 9) == 1.0
    rows = _world("w", [10, 4], [6, 8], [9, 9])
    w = _report(rows).worlds[0]
    assert (w.baseline, w.candidate_mean, w.ratio) == (9.0, 9.0, 1.0)
    assert max(sum([10, 4]) / 2, sum([6, 8]) / 2) == 7  # the retired max-of-means bar


def test_worlds_weigh_equally() -> None:
    """Ratios 1.2 (2 episodes) and 1.0 (3 episodes) → regime ratio 1.1, not the episode-weighted
    1.08. Same numbers as ARCHITECTURE §7.2."""
    assert regime_ratio([1.2, 1.0]) == 1.1
    rows = (_world("w1", [10, 10], [5, 5], [12, 12])
            + _world("w2", [10, 10, 10], [5, 5, 5], [10, 10, 10]))
    rep = _report(rows)
    assert [w.ratio for w in rep.worlds] == [1.2, 1.0]
    calm = next(r for r in rep.regimes if r.regime == CALM)
    assert calm.mean_ratio == 1.1
    assert calm.worlds == ("w1", "w2")


def test_random_member_value_is_replicate_mean() -> None:
    """A random member's per-episode value is its replicate mean (4, 10, 7 → 7), and that mean,
    not its best replicate, enters the per-episode max."""
    rows = [
        _row("w", 1, A, 5), _row("w", 1, B, 6), _row("w", 1, CANDIDATE, 7),
        *(_row("w", 1, RND, u, replicate=i) for i, u in enumerate([4, 10, 7])),
    ]
    assert _report(rows, ROSTER_ABR).worlds[0].baseline == 7.0


def test_candidate_replicates_are_averaged_and_each_is_judged() -> None:
    rows = [_row("w", 1, A, 10), _row("w", 1, B, 10),
            _row("w", 1, CANDIDATE, 8, replicate=0),
            _row("w", 1, CANDIDATE, 14, replicate=1, dead=True)]
    rep = _report(rows)
    assert rep.worlds[0].candidate_mean == 11.0
    (f,) = rep.clause(ZERO_DEATH_CLAUSE).failures
    assert (f.world, f.episode, f.replicate) == ("w", 1, 1)


# --- clauses at and around their thresholds ---


@pytest.mark.parametrize("cand,passed", [(105, True), (106, True), (104, False)])
def test_calm_margin_threshold(cand: float, passed: bool) -> None:
    rows = (_world("c", [100], [50], [cand])
            + _world("s", [100], [50], [120], regime=SHOCKED))
    rep = _report(rows)
    assert rep.clause(CALM_CLAUSE).passed is passed
    assert rep.clause(SHOCKED_CLAUSE).passed
    assert rep.overall == (ACCEPT if passed else REJECT)


@pytest.mark.parametrize("cand,passed", [(110, True), (111, True), (107, False)])
def test_shocked_margin_threshold(cand: float, passed: bool) -> None:
    rows = (_world("c", [100], [50], [120])
            + _world("s", [100], [50], [cand], regime=SHOCKED))
    rep = _report(rows)
    assert rep.clause(SHOCKED_CLAUSE).passed is passed
    assert rep.overall == (ACCEPT if passed else REJECT)
    if not passed:
        (f,) = rep.clause(SHOCKED_CLAUSE).failures
        assert (f.size, f.regime, f.value, f.threshold) == (12, SHOCKED, 1.07, 1.10)


def test_stochastic_world_is_judged_by_shocked_margin() -> None:
    """A shocked-regime world at 1.07 fails the shocked margin even though 1.07 clears the calm
    1.05 (agent-acceptance "Stochastic demand without shocks"): the margin follows the regime."""
    rows = _world("c", [100], [50], [120]) + _world("s", [100], [50], [107], regime=SHOCKED)
    rep = _report(rows)
    assert rep.clause(CALM_CLAUSE).passed and not rep.clause(SHOCKED_CLAUSE).passed


@pytest.mark.parametrize("cand,passed", [(100, True), (97, False)])
def test_no_losing_world_threshold(cand: float, passed: bool) -> None:
    """Mean calm ratio passes (1.2 and the other world average ≥ 1.05) but one world at 0.97
    rejects; exactly 1.0 passes."""
    rows = (_world("c1", [100], [50], [140]) + _world("c2", [100], [50], [cand])
            + _world("s", [100], [50], [120], regime=SHOCKED))
    rep = _report(rows)
    assert rep.clause(CALM_CLAUSE).passed
    assert rep.clause(NO_LOSING_CLAUSE).passed is passed
    if not passed:
        (f,) = rep.clause(NO_LOSING_CLAUSE).failures
        assert (f.world, f.value) == ("c2", 0.97)
        assert not next(r for r in rep.regimes if r.regime == CALM).no_losing_passed


def test_single_death_rejects_regardless_of_margins() -> None:
    rows = _passing()
    rows[2] = _row("calm-1", 1000, CANDIDATE, 12, dead=True)
    rep = _report(rows)
    assert rep.overall == REJECT and rep.failing_clauses == (ZERO_DEATH_CLAUSE,)
    (f,) = rep.clause(ZERO_DEATH_CLAUSE).failures
    assert (f.world, f.episode) == ("calm-1", 1000)
    assert rep.worlds[0].deaths == 1


@pytest.mark.parametrize("r_end,passed", [(0.4, True), (0.41, True), (0.39, False)])
def test_run_down_threshold(r_end: float, passed: bool) -> None:
    """X = 0.5, r_start = 0.8 → bar 0.4; r_end exactly at the bar passes."""
    rep = _report(_passing(r_start=0.8, r_end=r_end))
    assert rep.clause(RUN_DOWN_CLAUSE).passed is passed
    assert rep.overall == (ACCEPT if passed else REJECT)


@pytest.mark.parametrize("r_start,r_end,passed", [
    (None, 0.5, False), (0.8, None, False), (0.0, 0.0, True),
])
def test_run_down_unmeasurable_fails_closed_zero_start_passes(
    r_start: float | None, r_end: float | None, passed: bool,
) -> None:
    rep = _report(_passing(r_start=r_start, r_end=r_end))
    assert rep.clause(RUN_DOWN_CLAUSE).passed is passed


@pytest.mark.parametrize("rate,passed", [(0.25, True), (0.26, True), (0.24, False)])
def test_starvation_threshold(rate: float, passed: bool) -> None:
    rep = _report(_passing(fill={"c0": 1.0, "c1": rate}))
    assert rep.clause(STARVATION_CLAUSE).passed is passed


def test_zero_order_consumer_not_judged_by_starvation() -> None:
    rep = _report(_passing(fill={"c0": 1.0, "c1": None}))
    assert rep.clause(STARVATION_CLAUSE).passed and rep.overall == ACCEPT


def test_roster_rows_are_not_judged() -> None:
    """Only candidate episodes face death / run-down / starvation; a starving, dying baseline
    leaves the verdict unchanged."""
    rows = _passing()
    rows[0] = _row("calm-1", 1000, A, 10, dead=True, r_end=0.0, fill={"c0": 0.0})
    assert _report(rows).overall == ACCEPT


def test_one_clause_fails_names_clause_world_episode() -> None:
    """gate-harness "One clause fails": every margin passes but one episode starves a consumer →
    reject, naming the clause, world and episode."""
    rows = _passing()
    rows[5] = _row("calm-1", 1001, CANDIDATE, 12, fill={"c0": 1.0, "c1": 0.1})
    rep = _report(rows)
    assert rep.clause(CALM_CLAUSE).passed and rep.clause(SHOCKED_CLAUSE).passed
    assert rep.clause(NO_LOSING_CLAUSE).passed
    assert rep.overall == REJECT
    assert rep.failing_clauses == (STARVATION_CLAUSE,)
    (f,) = rep.clause(STARVATION_CLAUSE).failures
    assert (f.clause, f.world, f.episode, f.consumer, f.value) == (
        STARVATION_CLAUSE, "calm-1", 1001, "c1", 0.1)
    assert rep.worlds[0].starvation_failures == 1


def test_all_clauses_pass_accepts() -> None:
    rep = _report(_passing())
    assert rep.overall == ACCEPT and rep.failing_clauses == ()
    assert all(c.passed for c in rep.clauses)


# --- regimes with no worlds, sizes ---


def test_regime_with_no_worlds_fails_its_margin() -> None:
    """A size with calm worlds only fails the shocked margin: no evidence is not a pass."""
    rep = _report(_world("c", [100], [50], [120]))
    assert not rep.clause(SHOCKED_CLAUSE).passed
    (f,) = rep.clause(SHOCKED_CLAUSE).failures
    assert (f.size, f.regime, f.value) == (12, SHOCKED, None)
    assert rep.overall == REJECT


def test_margins_are_per_size() -> None:
    """A 12-producer surplus does not cover a 120-producer shortfall."""
    rows = (_passing()
            + _world("c120", [100], [50], [101], size=120)
            + _world("s120", [100], [50], [140], regime=SHOCKED, size=120))
    rep = _report(rows, gate_sizes=GATE_SIZES)
    (f,) = rep.clause(CALM_CLAUSE).failures
    assert (f.size, f.value) == (120, 1.01)
    assert [(r.size, r.regime) for r in rep.regimes] == [
        (12, CALM), (12, SHOCKED), (120, CALM), (120, SHOCKED)]


def test_missing_gate_size_rejects() -> None:
    """With the default gate sizes (12, 120), a report holding only 12-producer worlds fails both
    margins at 120: acceptance needs every margin at every gate size."""
    rep = compute_report(_passing(), ROSTER_AB, PARAMS, REG)
    assert rep.overall == REJECT
    assert {(f.clause, f.size, f.value) for c in rep.clauses for f in c.failures} == {
        (CALM_CLAUSE, 120, None), (SHOCKED_CLAUSE, 120, None)}


def test_future_scale_rows_refused() -> None:
    """agent-acceptance "Future scale target": a 1000+ producer result SHALL NOT gate, so a row at
    a non-gate size is refused instead of being judged (reported outside the verdict)."""
    future = _world("future1000", [100], [50], [1], size=1200, dead=True, fill={"c0": 0.0})
    with pytest.raises(ValueError, match="size 1200 is not a gate size"):
        compute_report(_passing() + future, ROSTER_AB, PARAMS, REG)
    with pytest.raises(ValueError, match="not a gate size"):
        recompute_from_csv(rows_to_csv(_passing() + future), ROSTER_AB, PARAMS, REG, SMALL)


# --- zero-baseline worlds ---


def test_world_ratio_refuses_zero_baseline() -> None:
    with pytest.raises(ZeroBaselineWorld):
        world_ratio(5, 0)


def test_zero_baseline_world_is_non_gating() -> None:
    """agent-acceptance "Candidate positive on a zero-baseline world": excluded from every clause,
    no infinite ratio in any mean, reported separately."""
    zero = _world("zero", [0, 0], [0, 0], [3, 0], dead=True, fill={"c0": 0.0})
    rep = _report(_passing() + zero)
    assert rep.non_gating_worlds == ("zero",)
    z = next(w for w in rep.worlds if w.world == "zero")
    assert (z.gating, z.ratio, z.baseline, z.candidate_mean, z.deaths) == (False, None, 0.0, 1.5, 2)
    assert "zero" not in next(r for r in rep.regimes if r.regime == CALM).worlds
    assert rep.overall == ACCEPT


# --- row validation ---


def test_missing_member_refused() -> None:
    rows = [r for r in _world("w", [10], [6], [9]) if r.subject != B]
    with pytest.raises(ValueError, match="no rows for roster members"):
        _report(rows)


def test_non_member_subject_refused() -> None:
    rows = _world("w", [10], [6], [9]) + [_row("w", 1000, "noop/random", 3)]
    with pytest.raises(ValueError, match="neither the candidate nor a member"):
        _report(rows)


def test_random_member_needs_all_replicates() -> None:
    rows = [_row("w", 1, A, 5), _row("w", 1, B, 6), _row("w", 1, CANDIDATE, 7),
            _row("w", 1, RND, 4, replicate=0), _row("w", 1, RND, 6, replicate=1)]
    with pytest.raises(ValueError, match="roster wants"):
        _report(rows, ROSTER_ABR)


def test_missing_candidate_refused() -> None:
    rows = [r for r in _world("w", [10], [6], [9]) if r.subject != CANDIDATE]
    with pytest.raises(ValueError, match="no candidate rows"):
        _report(rows)


def test_digest_mismatch_names_world_episode_member() -> None:
    rows = _world("w", [10], [6], [9])
    rows[1] = _row("w", 1000, B, 6, digest="other")
    with pytest.raises(DigestMismatch, match=r"world w episode 1000: restore_when_low/proportional"):
        _report(rows)


@pytest.mark.parametrize("field", ["r_start", "r_end", "fill", "inversion_rate"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_measurement_refused(field: str, bad: float) -> None:
    """A NaN compares False with every bar, so it would pass run-down / starvation silently; the
    env never emits one (None is its "no value"), so a non-finite field is refused on ingestion,
    directly and through the CSV path that recompute uses."""
    kw: dict[str, object] = {"fill": {"c0": bad}} if field == "fill" else {field: bad}
    with pytest.raises(ValueError, match="not finite"):
        _row("w", 1, CANDIDATE, 5, **kw)  # type: ignore[arg-type]
    good = row_to_dict(_row("w", 1, CANDIDATE, 5))
    col = "consumer_fill_rate" if field == "fill" else field
    good[col] = json.dumps({"c0": bad}) if field == "fill" else repr(bad)
    with pytest.raises(ValueError, match="not finite"):
        row_from_dict(good)


def test_params_validated() -> None:
    with pytest.raises(ValueError):
        SideEffectParams(x=float("nan"), y=0.5, version="v")
    with pytest.raises(ValueError):
        SideEffectParams(x=0.5, y=0.5, version="")


# --- Verdict is reproducible from rows ---


def _awkward_rows() -> list[EpisodeRow]:
    """Rows whose floats do not survive naive formatting (0.1 + 0.2, 1/3, tiny/huge values) plus
    None side-effect fields and a failing clause, so the round trip is exercised on real edges."""
    rows = _passing(inversion_rate=0.1 + 0.2, min_spare_fraction=None, holding_units=1e-17)
    rows[2] = _row("calm-1", 1000, CANDIDATE, 12, r_start=1 / 3, r_end=1 / 7,
                   fill={"c0": 2 / 3, "c1": None}, inversion_rate=1 / 3,
                   holding_units_per_cycle=1e300)
    return rows


def test_csv_round_trip_is_exact() -> None:
    rows = _awkward_rows()
    text = rows_to_csv(rows)
    assert text.splitlines()[0].split(",") == list(CSV_COLUMNS)
    assert rows_from_csv(text) == rows
    assert row_from_dict(row_to_dict(rows[2])) == rows[2]


def test_recompute_from_csv_matches_exactly() -> None:
    """gate-harness "Recompute from the CSV": rows → CSV text → rows → report equals the original,
    and the recorded verdict JSON is byte-identical."""
    rows = _awkward_rows()
    original = _report(rows)
    assert original.overall == REJECT  # run-down fails on the awkward row: a non-trivial verdict
    recomputed = recompute_from_csv(rows_to_csv(rows), ROSTER_AB, PARAMS, REG, SMALL)
    assert recomputed == original
    assert recomputed.to_json() == original.to_json()
    assert json.loads(original.to_json()) == original.to_dict()
    d = original.to_dict()
    assert (d["roster_version"], d["registry_version"], d["params_version"]) == (
        ROSTER_AB.version, REG, "params-test")


def test_verdict_independent_of_row_order() -> None:
    rows = _awkward_rows()
    assert _report(list(reversed(rows))).to_json() == _report(rows).to_json()


def test_row_from_stats_maps_env_keys() -> None:
    from tokenomics.env.aec_env import TokenomicsAECEnv
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import run_episode

    env = TokenomicsAECEnv(horizon=3)
    run_episode(env, PRODUCING_BASELINES["always_restore"], ORDERING_BASELINES["even_split"], 0)
    s = env.episode_stats
    r = row_from_stats("w", 12, CALM, 0, CANDIDATE, 0, s)
    assert r.units == float(s["consumer_completed_units"])
    assert r.fill_rates == s["consumer_fill_rate"]
    assert (r.r_start, r.r_end, r.dead) == (s["r_start"], s["r_end"], s["dead"])
    assert rows_from_csv(rows_to_csv([r])) == [r]
