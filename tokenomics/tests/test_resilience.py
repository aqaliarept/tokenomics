"""Resilience rows against the shock-free twin (ADR-0055; add-shocks-i-drift design D6;
stochastic-environment "Resilience is reported")."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tokenomics.domain.config import DemandDrift, load_config
from tokenomics.domain.shocks import ShockEvent, ShockKind, ShockModel, ShockTarget
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.shock_schedule import shock_digest
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.resilience import (DEFAULT_PARAMS, ResilienceParams, resilience_rows,
                                        shock_free_twin)
from tokenomics.eval.runner import run_episode

RANDOM_DEMAND = Path(__file__).resolve().parent / "fixtures" / "random_demand.json"


def _outage(start: int, duration: int, producer: str = "p4") -> ShockEvent:
    return ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer=producer), start, duration,
                      1.0)


def _hit(start: int, producer: str = "p1") -> ShockEvent:
    return ShockEvent(ShockKind.CAPACITY_HIT,
                      ShockTarget(producer=producer, capacity_types=("capacity-1",)), start, 1, 0.5)


P = ResilienceParams(window=3, tolerance=0.1)  # keep 90% of the twin over 3-cycle windows


def test_recovery_measured() -> None:
    """Scenario "Recovery measured": a known dip and recovery time on hand-built traces."""
    twin = [10] * 20
    shocked = [10, 10, 10, 4, 2, 6, 8, 10] + [10] * 12
    [row] = resilience_rows(shocked, twin, (_outage(4, 2),), P, shocked_dead=False)
    # windows from cycle 6: 6+8+10 = 24 < 27; cycle 7: 8+10+10 = 28 >= 27 -> r = 7
    assert row.recovered and row.recovery_cycle == 7
    assert row.recovery_cycles == 1  # one cycle after the shock ended (e + 1 = 6)
    assert row.dip_units == 6 + 8 + 4  # cycles 4, 5, 6
    assert row.dip_fraction == pytest.approx(18 / 30)
    assert not row.overlapped
    assert (row.kind, row.producer, row.start, row.end) == ("producer_outage", "p4", 4, 5)


def test_immediate_recovery_and_no_dip() -> None:
    twin = shocked = [5] * 12
    [row] = resilience_rows(shocked, twin, (_hit(3),), P, shocked_dead=False)
    assert row.recovered and row.recovery_cycle == 4 and row.recovery_cycles == 0
    assert row.dip_units == 0 and row.dip_fraction == 0.0


def test_signed_dip_and_zero_twin_units() -> None:
    # the shocked run does better than its twin: negative dip
    [row] = resilience_rows([0, 0, 9, 9, 9, 9, 9, 9], [0, 0, 5, 5, 5, 5, 5, 5], (_hit(3),), P, shocked_dead=False)
    assert row.dip_units == -4 and row.recovery_cycles == 0
    # nothing delivered by either: recovered at once, dip fraction undefined
    [row] = resilience_rows([0] * 8, [0] * 8, (_hit(3),), P, shocked_dead=False)
    assert row.recovered and row.dip_units == 0 and row.dip_fraction is None


def test_recovery_never_reached() -> None:
    """Scenario "Recovery never reached": censored, dip summed to the last played cycle."""
    twin = [10] * 12
    shocked = [10, 10, 10, 2, 2, 2, 5, 5, 5, 5, 5, 5]
    [row] = resilience_rows(shocked, twin, (_outage(4, 3),), P, shocked_dead=False)
    assert not row.recovered and row.recovery_cycle is None and row.recovery_cycles is None
    assert row.dip_units == 3 * 8 + 6 * 5  # cycles 4..12
    assert row.dip_fraction == pytest.approx(54 / 90)


def test_recovery_never_reached_after_an_early_death() -> None:
    """Early death: the shocked trace ends at the death cycle; the row is censored there,
    and a shock that would have started after death gets no row."""
    twin = [10] * 12
    shocked = [10, 10, 10, 3, 2, 1, 0]  # died at cycle 7
    rows = resilience_rows(shocked, twin, (_outage(4, 2), _hit(9)), P, shocked_dead=True)
    assert len(rows) == 1
    [row] = rows
    assert not row.recovered and row.dip_units == 7 + 8 + 9 + 10  # cycles 4..7
    assert row.end == 5


@pytest.mark.parametrize("twin_len", [8, 12])
def test_death_censors_a_row_that_looked_recovered(twin_len: int) -> None:
    """D6: a shocked episode that dies censors every row, even one whose windows had caught
    up with the twin, and even when both traces have the same length (death is passed in,
    never inferred from lengths)."""
    shocked, twin = [10] * 8, [10] * twin_len  # dead at cycle 8, no dip at all
    [alive] = resilience_rows(shocked, twin, (_outage(2, 1),), P, shocked_dead=False)
    assert alive.recovered and alive.recovery_cycle == 3 and alive.dip_units == 0
    [dead] = resilience_rows(shocked, twin, (_outage(2, 1),), P, shocked_dead=True)
    assert not dead.recovered and dead.recovery_cycle is None and dead.recovery_cycles is None
    assert dead.dip_units == 0 and dead.dip_fraction == 0.0  # summed through cycle 8
    shocked = [10, 4, 10, 10, 10, 10, 10, 10]
    [dead] = resilience_rows(shocked, twin, (_outage(2, 1),), P, shocked_dead=True)
    assert dead.dip_units == 6 and dead.dip_fraction == pytest.approx(6 / 70)  # cycles 2..8


def test_dead_flag_is_required() -> None:
    with pytest.raises(TypeError):
        resilience_rows([1] * 5, [1] * 5, (), P)  # type: ignore[call-arg]


def test_shock_running_past_the_end_is_unrecovered() -> None:
    [row] = resilience_rows([10] * 10, [10] * 10, (_outage(8, 5),), P, shocked_dead=False)
    assert not row.recovered and row.end == 12 and row.dip_units == 0


def test_overlapping_shocks_flagged() -> None:
    """Scenario "Overlapping shocks flagged": a shock that becomes active between another's
    start and its recovery flags that row. The flag follows the spec's "active" span: the
    second shock's own span (6..7) holds no other active shock (the first ended at 5), so it
    is not flagged even though the first shock's dip still lingers; a later, separate shock
    is not flagged either."""
    twin = [10] * 30
    shocked = [10, 10, 10, 2, 2, 2, 2, 10, 10, 10] + [10] * 8 + [3] + [10] * 11
    schedule = (_outage(4, 2), _outage(6, 1, producer="p3"), _hit(19))
    rows = resilience_rows(shocked, twin, schedule, P, shocked_dead=False)
    assert [r.overlapped for r in rows] == [True, False, False]
    # first shock: ends at 5 but recovery only from cycle 8 -> its span 4..7 holds shock 2
    assert rows[0].recovery_cycle == 8 and rows[0].dip_units == 32
    assert rows[1].recovery_cycle == 8 and rows[1].recovery_cycles == 1
    assert rows[2].recovered and rows[2].dip_units == 7
    # a second shock active inside the first's active span flags both
    both = resilience_rows(shocked, twin, (_outage(4, 3), _outage(5, 1, producer="p3")), P, shocked_dead=False)
    assert [r.overlapped for r in both] == [True, True]


def test_params_and_trace_checks() -> None:
    assert (DEFAULT_PARAMS.window, DEFAULT_PARAMS.tolerance) == (5, 0.05)
    with pytest.raises(ValueError, match=r"window must be an integer >= 1"):
        ResilienceParams(window=0)
    with pytest.raises(ValueError, match=r"tolerance must be in \[0, 1\)"):
        ResilienceParams(tolerance=1.0)
    with pytest.raises(ValueError, match=r"twin trace has 3 cycles"):
        resilience_rows([1] * 5, [1] * 3, (), P, shocked_dead=False)


def test_exact_tolerance_boundary() -> None:
    """tau is read as its exact decimal. At the default tau = 0.05 and twin units 2**60, the
    largest integer below the exact threshold 0.95 * 2**60 is rejected (censored), while the
    float threshold (1 - 0.05) * 2**60 would accept it; one unit more is accepted."""
    from fractions import Fraction
    from math import ceil

    t = 2 ** 60
    below = ceil((1 - Fraction("0.05")) * t) - 1
    assert below >= (1 - 0.05) * t  # the float comparison would call this recovered
    params = ResilienceParams(window=1, tolerance=0.05)
    twin = [t, t, t]
    [row] = resilience_rows([t, 0, below], twin, (_hit(2),), params, shocked_dead=False)
    assert not row.recovered and row.dip_units == t + (t - below)
    [row] = resilience_rows([t, 0, below + 1], twin, (_hit(2),), params, shocked_dead=False)
    assert row.recovered and row.recovery_cycle == 3


def test_recovery_at_the_last_full_window() -> None:
    """The last valid window start is H - w + 1 (here 10 - 3 + 1 = 8): recovery there is
    found, with its dip; a recovery that would need a window running past H is censored."""
    twin = [10] * 10
    shocked = [10, 10, 0, 0, 0, 0, 0, 10, 10, 10]  # back to 10 from cycle 8
    [row] = resilience_rows(shocked, twin, (_hit(3),), P, shocked_dead=False)
    assert row.recovered and row.recovery_cycle == 8 and row.recovery_cycles == 4
    assert row.dip_units == 50  # cycles 3..7
    late = [10, 10, 0, 0, 0, 0, 0, 0, 10, 10]  # back from cycle 9: only a 2-cycle window
    [row] = resilience_rows(late, twin, (_hit(3),), P, shocked_dead=False)
    assert not row.recovered and row.recovery_cycle is None
    assert row.dip_units == 60  # censored: cycles 3..10


# --- env fixture ---


def test_shocked_episode_twin_and_one_row() -> None:
    """A tiny env fixture (horizon 12, one outage, drift on): the twin keeps the demand
    (equal digest) and has no shocks; the pair yields one resilience row."""
    base = load_config(RANDOM_DEMAND)
    cfg = replace(base, demand_drift=DemandDrift(1.0), shocks=ShockModel(events=(_outage(4, 3),)))
    twin_cfg = shock_free_twin(cfg)
    assert twin_cfg.shocks is None and twin_cfg.demand_drift == cfg.demand_drift
    runs = []
    for c in (cfg, twin_cfg):
        env = TokenomicsAECEnv(config=c, horizon=12)
        runs.append(run_episode(env, PRODUCING_BASELINES["stock_aware"],
                                ORDERING_BASELINES["proportional"], seed=3))
        assert sum(env.episode_stats["consumer_fulfilled_units_trace"]) == \
            env.episode_stats["consumer_completed_units"]
    shocked, twin = runs
    assert shocked["demand_digest"] == twin["demand_digest"]
    assert twin["shock_events"] == 0 and twin["shock_digest"] == shock_digest(())
    assert shocked["shock_events"] == 1
    assert len(shocked["consumer_fulfilled_units_trace"]) == shocked["cycles"] == 12
    rows = resilience_rows(shocked["consumer_fulfilled_units_trace"],
                           twin["consumer_fulfilled_units_trace"], (_outage(4, 3),),
                           shocked_dead=shocked["dead"])
    assert len(rows) == 1 and rows[0].kind == "producer_outage" and rows[0].start == 4
    assert shocked["consumer_fulfilled_units_trace"] != twin["consumer_fulfilled_units_trace"]
