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
    """A 12-producer surplus does not cover a 42-producer shortfall."""
    rows = (_passing()
            + _world("c42", [100], [50], [101], size=42)
            + _world("s42", [100], [50], [140], regime=SHOCKED, size=42))
    rep = _report(rows, gate_sizes=GATE_SIZES)
    (f,) = rep.clause(CALM_CLAUSE).failures
    assert (f.size, f.value) == (42, 1.01)
    assert [(r.size, r.regime) for r in rep.regimes] == [
        (12, CALM), (12, SHOCKED), (42, CALM), (42, SHOCKED)]


def test_missing_gate_size_rejects() -> None:
    """With the default gate sizes (12, 42), a report holding only 12-producer worlds fails both
    margins at 42: acceptance needs every margin at every gate size."""
    rep = compute_report(_passing(), ROSTER_AB, PARAMS, REG)
    assert rep.overall == REJECT
    assert {(f.clause, f.size, f.value) for c in rep.clauses for f in c.failures} == {
        (CALM_CLAUSE, 42, None), (SHOCKED_CLAUSE, 42, None)}


@pytest.mark.parametrize("size", [120, 1200])
def test_future_scale_rows_refused(size: int) -> None:
    """agent-acceptance "Future scale target": a ~120 or 1000+ producer result SHALL NOT gate, so
    a row at a non-gate size is refused instead of being judged (reported outside the verdict)."""
    assert GATE_SIZES == (12, 42)
    future = _world("future", [100], [50], [1], size=size, dead=True, fill={"c0": 0.0})
    with pytest.raises(ValueError, match=f"size {size} is not a gate size"):
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


# --- per-family X / Y and replicated candidates (add-gate-v3 task 2.3, design D4, D6) ---

V3_NAME = "side-effect-params-v3"
ON = "a" * 64  # a screened_on hash (v2+ parameters record one)
SW, WL = "sparse_wide", "ws_wide"
FAMILY_PARAMS = SideEffectParams(x=None, y=None, version="params-v3-test", name=V3_NAME,
                                 screened_on=ON, families=((SW, 0.5, 0.25), (WL, 0.3, 0.05)))
WORLD_FAMILY = {"calm-1": SW, "shock-1": SW, "calm-w": WL, "shock-w": WL}


def _v3_report(rows: list[EpisodeRow], params: SideEffectParams = FAMILY_PARAMS,
               world_family: dict[str, str] | None = None) -> GateReport:
    return compute_report(rows, ROSTER_AB, params, REG, SMALL,
                          world_family=WORLD_FAMILY if world_family is None else world_family)


def _replicated(world: str, cand: list[dict], *, regime: str = CALM,
                episode: int = 1000) -> list[EpisodeRow]:
    """One episode: A 10, B 8 (deterministic) and one candidate row per dict in `cand`
    (replicate r = position), each dict overriding `_row`'s keyword arguments."""
    rows = [_row(world, episode, A, 10, regime=regime), _row(world, episode, B, 8, regime=regime)]
    for r, kw in enumerate(cand):
        rows.append(_row(world, episode, CANDIDATE, kw.pop("units", 12), replicate=r,
                         regime=regime, **kw))
    return rows


def test_family_values_are_applied() -> None:
    """A fill rate of 0.1 is below sparse_wide's Y 0.25 and above ws_wide's Y 0.05: it fails on a
    sparse_wide world only (gate-harness "Family values used")."""
    starving = {"fill": {"c0": 1.0, "c1": 0.1}}
    rows = (_replicated("calm-1", [dict(starving)]) + _replicated("calm-w", [dict(starving)]))
    rep = _v3_report(rows)
    fails = rep.clause(STARVATION_CLAUSE).failures
    assert [(f.world, f.threshold) for f in fails] == [("calm-1", 0.25)]
    assert {w.world: w.family for w in rep.worlds} == {"calm-1": SW, "calm-w": WL}
    assert rep.to_dict()["families"] == {SW: {"x": 0.5, "y": 0.25}, WL: {"x": 0.3, "y": 0.05}}
    assert rep.x is None and rep.y is None
    # run-down: ratio 0.4 is below sparse_wide's X 0.5, above ws_wide's 0.3
    rundown = {"r_start": 1.0, "r_end": 0.4}
    rep = _v3_report(_replicated("calm-1", [dict(rundown)]) + _replicated("calm-w", [dict(rundown)]))
    assert [(f.world, f.threshold) for f in rep.clause(RUN_DOWN_CLAUSE).failures] == [("calm-1", 0.5)]


def test_per_family_params_need_every_family() -> None:
    rows = _replicated("calm-1", [{}]) + _replicated("calm-w", [{}])
    only_sw = SideEffectParams(x=None, y=None, version="v", name=V3_NAME, screened_on=ON,
                               families=((SW, 0.5, 0.25),))
    with pytest.raises(ValueError, match=r"no X / Y for families \['ws_wide'\]"):
        _v3_report(rows, only_sw)
    with pytest.raises(ValueError, match="needs each world's family"):
        compute_report(rows, ROSTER_AB, FAMILY_PARAMS, REG, SMALL)
    with pytest.raises(ValueError, match="no family for worlds"):
        _v3_report(rows, world_family={"calm-1": SW})
    with pytest.raises(ValueError, match="record no X / Y for family 'dense'"):
        FAMILY_PARAMS.values_for("dense")
    assert PARAMS.values_for("anything") == (0.5, 0.25) and PARAMS.missing_families([WL]) == []


def test_one_replicate_starves_a_consumer() -> None:
    """One of five replicates leaves c1 at 0.1 < Y 0.25 while c1's mean over the five is 0.42: the
    starvation clause passes and the replicate's violation is listed as monitored."""
    cand = [{"fill": {"c0": 1.0, "c1": 0.5}} for _ in range(5)]
    cand[2] = {"fill": {"c0": 1.0, "c1": 0.1}}
    rep = _v3_report(_replicated("calm-1", cand))
    assert rep.clause(STARVATION_CLAUSE).passed
    viol = rep.replicate_violations
    assert [(v.clause, v.replicate, v.consumer, v.value) for v in viol] == [
        (STARVATION_CLAUSE, 2, "c1", 0.1)]
    assert rep.worlds[0].replicate_violations == 1 and rep.worlds[0].starvation_failures == 0
    # the same rows under scalar (v2-style) parameters fail per replicate, as before
    v2 = compute_report(_replicated("calm-1", [dict(c) for c in cand]), ROSTER_AB, PARAMS, REG,
                        SMALL)
    assert [f.replicate for f in v2.clause(STARVATION_CLAUSE).failures] == [2]
    assert "replicate_violations" not in v2.to_dict()


def test_mean_below_y_fails_without_a_replicate() -> None:
    cand = [{"fill": {"c0": 1.0, "c1": 0.2}}, {"fill": {"c0": 1.0, "c1": 0.28}}]
    f, = _v3_report(_replicated("calm-1", cand)).clause(STARVATION_CLAUSE).failures
    assert (f.consumer, f.replicate, f.threshold) == ("c1", None, 0.25)
    assert f.value == pytest.approx(0.24)


def test_one_replicate_dies() -> None:
    """One of five replicates dies while the replicate mean of units clears the margin: the
    zero-death clause fails, naming the world, episode and replicate."""
    cand = [{"units": 12} for _ in range(5)]
    cand[3] = {"units": 12, "dead": True}
    rows = (_replicated("calm-1", cand)
            + _replicated("shock-1", [{"units": 12} for _ in range(5)], regime=SHOCKED))
    rep = _v3_report(rows)
    assert rep.clause(CALM_CLAUSE).passed and rep.clause(SHOCKED_CLAUSE).passed
    f, = rep.clause(ZERO_DEATH_CLAUSE).failures
    assert (f.world, f.episode, f.replicate) == ("calm-1", 1000, 3)
    assert rep.overall == REJECT and rep.failing_clauses == (ZERO_DEATH_CLAUSE,)


def test_run_down_on_the_replicate_mean() -> None:
    """Replicate ratios 0.4 and 0.7 (mean 0.55 >= X 0.5) pass, the 0.4 is monitored; a replicate
    with r_end None fails closed; with every r_start 0 there is no ratio and the check passes."""
    rep = _v3_report(_replicated("calm-1", [{"r_start": 1.0, "r_end": 0.4},
                                            {"r_start": 1.0, "r_end": 0.7}]))
    assert rep.clause(RUN_DOWN_CLAUSE).passed
    assert [(v.clause, v.replicate) for v in rep.replicate_violations] == [(RUN_DOWN_CLAUSE, 0)]
    rep = _v3_report(_replicated("calm-1", [{"r_start": 1.0, "r_end": 0.3},
                                            {"r_start": 1.0, "r_end": 0.6}]))
    f, = rep.clause(RUN_DOWN_CLAUSE).failures
    assert (f.replicate, f.value) == (None, pytest.approx(0.45))
    rep = _v3_report(_replicated("calm-1", [{"r_start": 1.0, "r_end": 0.9}, {"r_end": None}]))
    f, = rep.clause(RUN_DOWN_CLAUSE).failures
    assert f.value is None and "unmeasurable" in f.detail
    rep = _v3_report(_replicated("calm-1", [{"r_start": 0.0, "r_end": 0.0}]))
    assert rep.clause(RUN_DOWN_CLAUSE).passed


def test_scalar_reports_serialize_as_before() -> None:
    """v1 / v2 parameters add no v3 key to the report or its worlds (byte-identical verdicts)."""
    d = _report(_passing()).to_dict()
    assert set(d) == {"roster_version", "registry_version", "params_version", "x", "y", "worlds",
                      "regimes", "clauses", "non_gating_worlds", "overall", "failing_clauses"}
    assert all(set(w) == {"world", "size", "regime", "episodes", "candidate_mean", "baseline",
                          "ratio", "gating", "deaths", "run_down_failures",
                          "starvation_failures", "monitored"} for w in d["worlds"])


def test_per_family_params_construction() -> None:
    with pytest.raises(ValueError, match="exist from v3 on"):
        SideEffectParams(x=None, y=None, version="v", name="side-effect-params-v2",
                         screened_on=ON, families=((SW, 0.5, 0.25),))
    with pytest.raises(ValueError, match="record X / Y per family"):
        SideEffectParams(x=0.5, y=0.25, version="v", name=V3_NAME, screened_on=ON)
    with pytest.raises(ValueError, match="no scalar X / Y"):
        SideEffectParams(x=0.5, y=None, version="v", name=V3_NAME, screened_on=ON,
                         families=((SW, 0.5, 0.25),))
    with pytest.raises(ValueError, match="distinct families"):
        SideEffectParams(x=None, y=None, version="v", name=V3_NAME, screened_on=ON,
                         families=((SW, 0.5, 0.25), (SW, 0.4, 0.2)))
    with pytest.raises(ValueError, match="finite"):
        SideEffectParams(x=None, y=None, version="v", name=V3_NAME, screened_on=ON,
                         families=((SW, float("nan"), 0.25),))


def _roster_rows(world: str, episode: int, *, regime: str = CALM, a: float = 10, b: float = 8,
                 r_end: float = 0.8, fill: dict[str, float | None] | None = None
                 ) -> list[EpisodeRow]:
    """Roster-only rows of one episode: A (the argmax) with the given side effects, B lower."""
    return [_row(world, episode, A, a, regime=regime, r_start=1.0, r_end=r_end, fill=fill),
            _row(world, episode, B, b, regime=regime)]


def test_family_calibration_distinct_counts_on_triplicated_calm_rows() -> None:
    """Calm worlds replay: a deterministic argmax member repeats its values in each of a world's
    3 episodes, so value counts triple and distinct counts do not. Families are calibrated apart,
    sparse_wide pooled over sizes."""
    from tokenomics.eval.gate import SMALL_SAMPLE_DISTINCT, calibrate_families

    rows: list[EpisodeRow] = []
    fam: dict[str, str] = {}
    for i in range(10):  # 10 calm ws_wide worlds x 3 identical episodes
        w = f"wl-calm-{i}"
        fam[w] = WL
        for ep in range(3):
            rows += _roster_rows(w, 100 * i + ep, r_end=0.5 + i / 100,
                                 fill={"c0": 0.2 + i / 100, "c1": None})
    for size in (12, 42):  # sparse_wide at both sizes, one pooled sample
        for i in range(25):
            w = f"sw-{size}-{i}"
            fam[w] = SW
            rows += [replace_size(r, size) for r in _roster_rows(
                w, 5000 + size * 100 + i, regime=SHOCKED, r_end=0.7 + i / 1000,
                fill={"c0": 0.6 + i / 1000, "c1": 0.9})]
    cals = {c.family: c for c in calibrate_families(rows, ROSTER_AB, 0.05, fam)}
    wl, sw = cals[WL], cals[SW]
    assert wl.calibration.run_down_values == 30 and wl.run_down_distinct == 10
    assert wl.calibration.fill_values == 30 and wl.fill_distinct == 10
    assert wl.small_sample and wl.x_is_minimum and wl.y_is_minimum  # k = ceil(1.5) = 2nd = min
    assert wl.calibration.x == 0.5 and wl.calibration.y == pytest.approx(0.2)
    assert sw.calibration.episodes == 50 and sw.run_down_distinct == 25
    assert sw.fill_distinct == 26 and not sw.small_sample  # 25 c0 values + c1's 0.9
    assert sw.calibration.x == pytest.approx(0.701) and not sw.x_is_minimum  # 3rd of 50
    ev = wl.evidence()
    assert ev["small_sample"] is True and ev["run_down_distinct"] == 10
    assert min(ev["run_down_distinct"], ev["fill_distinct"]) < SMALL_SAMPLE_DISTINCT
    with pytest.raises(ValueError, match="without a family"):
        calibrate_families(rows, ROSTER_AB, 0.05, {k: v for k, v in fam.items() if k != "wl-calm-0"})


def replace_size(r: EpisodeRow, size: int) -> EpisodeRow:
    from dataclasses import replace

    return replace(r, size=size)


def test_v3_replicate_means_exactly_at_the_bar_pass() -> None:
    """Thresholds stay inclusive under v3: a replicate-mean fill exactly at Y and a replicate-mean
    r_end / r_start exactly at X pass (values exact in binary floating point: 0.25, 0.5, 0.75)."""
    at_y = [{"fill": {"c0": 1.0, "c1": 0.25}}, {"fill": {"c0": 1.0, "c1": 0.25}}]
    rep = _v3_report(_replicated("calm-1", at_y))
    assert rep.clause(STARVATION_CLAUSE).passed and not rep.replicate_violations
    straddle_y = [{"fill": {"c0": 1.0, "c1": 0.125}}, {"fill": {"c0": 1.0, "c1": 0.375}}]
    rep = _v3_report(_replicated("calm-1", straddle_y))  # mean 0.25 exactly
    assert rep.clause(STARVATION_CLAUSE).passed
    assert [v.replicate for v in rep.replicate_violations] == [0]
    at_x = [{"r_start": 1.0, "r_end": 0.25}, {"r_start": 1.0, "r_end": 0.75}]  # mean 0.5 = X
    rep = _v3_report(_replicated("calm-1", at_x))
    assert rep.clause(RUN_DOWN_CLAUSE).passed
    assert _v3_report(_replicated("calm-1", [{"r_start": 0.5, "r_end": 0.25}])).clause(
        RUN_DOWN_CLAUSE).passed  # one replicate, ratio 0.5 = X


def test_v3_run_down_is_the_mean_of_ratios_the_calibration_uses() -> None:
    """Replicates with different r_start: the check uses the mean of r_end / r_start (what
    `episode_bests` calibrates X on), not the ratio of means. Both directions differ at X 0.5."""
    from tokenomics.eval.gate import episode_bests

    passes = [{"r_start": 1.0, "r_end": 0.25}, {"r_start": 0.25, "r_end": 0.25}]
    # mean of ratios (0.25 + 1.0) / 2 = 0.625 >= 0.5; ratio of means 0.5 / 1.25 = 0.4 < 0.5
    assert _v3_report(_replicated("calm-1", [dict(c) for c in passes])).clause(
        RUN_DOWN_CLAUSE).passed
    fails = [{"r_start": 1.0, "r_end": 0.875}, {"r_start": 0.125, "r_end": 0.0078125}]
    # mean of ratios (0.875 + 0.0625) / 2 = 0.46875 < 0.5; ratio of means 0.8828 / 1.125 > 0.5
    f, = _v3_report(_replicated("calm-1", [dict(c) for c in fails])).clause(
        RUN_DOWN_CLAUSE).failures
    assert f.value == 0.46875 and f.replicate is None
    # the calibration reduces the same replicates to the same statistic: a random member with
    # these three replicates as the per-episode argmax gets run_down 0.46875 + its fill means
    reps = [_row("w", 1, RND, 20, replicate=r, r_start=c["r_start"], r_end=c["r_end"],
                 fill=fill)
            for r, (c, fill) in enumerate(zip(fails + [{"r_start": 1.0, "r_end": 0.46875}],
                                              [{"c1": 0.5}, {"c1": None}, {"c1": 0.25}]))]
    best, = episode_bests([_row("w", 1, A, 10), _row("w", 1, B, 8)] + reps, ROSTER_ABR)
    assert best.member == RND and best.run_down == 0.46875 and best.fill_rates == (0.375,)


def test_v3_starvation_skips_replicates_where_the_consumer_did_not_order() -> None:
    """A consumer with fill None in a replicate ordered nothing there: the replicate mean is over
    the replicates where it ordered (None skipped, not counted as 0), as Y is calibrated."""
    skipped = [{"fill": {"c0": 1.0, "c1": 0.375}}, {"fill": {"c0": 1.0, "c1": None}}]
    rep = _v3_report(_replicated("calm-1", skipped))  # 0.375 >= Y; as 0 it would be 0.1875
    assert rep.clause(STARVATION_CLAUSE).passed
    low = [{"fill": {"c0": 1.0, "c1": 0.125}}, {"fill": {"c0": 1.0, "c1": None}},
           {"fill": {"c0": 1.0, "c1": 0.125}}]
    f, = _v3_report(_replicated("calm-1", low)).clause(STARVATION_CLAUSE).failures
    assert (f.consumer, f.value) == ("c1", 0.125)  # as 0 it would be 0.0833...
    none = [{"fill": {"c0": 1.0, "c1": None}}, {"fill": {"c0": 1.0, "c1": None}}]
    assert _v3_report(_replicated("calm-1", none)).clause(STARVATION_CLAUSE).passed
