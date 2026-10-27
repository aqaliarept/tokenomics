"""Capacity-hit mechanics in the domain (ADR-0055; add-shocks-i-drift design D2;
stochastic-environment "Capacity hit", "Domain stays RNG-free").

The domain replays an injected schedule. A hit removes floor(m*c) units at the start of its
cycle, max capacity is unchanged, and end-of-cycle metrics read a snapshot taken before the
next cycle's shocks, so a hit in an unplayed cycle never reaches `r_end`.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import tokenomics.env.aec_env as aec_env
from tokenomics.domain import build_system
from tokenomics.domain.config import Configuration, load_config
from tokenomics.domain.ordering_agent import OrderingAgentCommand, SubOrder
from tokenomics.domain.producing_agent import ProducingAgentCommand
from tokenomics.domain.shocks import ShockEvent, ShockKind, ShockTarget
from tokenomics.domain.system import CycleResult, System, capacity_hit_removal
from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.runner import run_episode

PKG = Path(__file__).resolve().parents[1] / "tokenomics"
PHASE1 = PKG / "config" / "phase1.json"
CAP = "capacity-1"  # p1's only type in phase1 (120 units, degradation 5% of max)


def _hit(cycle: int, magnitude: float = 0.4, producer: str = "p1",
         types: tuple[str, ...] = (CAP,)) -> ShockEvent:
    return ShockEvent(ShockKind.CAPACITY_HIT, ShockTarget(producer=producer, capacity_types=types),
                      start_cycle=cycle, duration=1, magnitude=magnitude)


def _phase1(*, degradation: int | None = None, p1_capacity: int | None = None) -> Configuration:
    cfg = load_config(PHASE1)
    producers = []
    for p in cfg.producer_configs:
        if degradation is not None:
            p = replace(p, degradation=degradation)
        if p.id == "p1" and p1_capacity is not None:
            p = replace(p, capacities={CAP: p1_capacity})
        producers.append(p)
    return replace(cfg, producer_configs=producers)


def _play(system: System, restore: frozenset[str] = frozenset()) -> CycleResult:
    """One cycle: producers request restoration when asked (and none is running); every
    order goes whole to its first eligible producer."""
    for pid, p in system.producing_agents.items():
        system.producing_agent_action(
            pid, ProducingAgentCommand(pid in restore and not p.restoration_running, False))
    system.start_ordering()
    for aid, agent in system.ordering_agents.items():
        view = agent.view(system.producer_infos)
        orders = {oid: [SubOrder(next(iter(view.eligible[oid])), oi.quantity, oi.tokens)]
                  for oid, oi in view.incoming.items()}
        system.ordering_agent_action(aid, OrderingAgentCommand(orders))
    return system.complete_cycle()


def _cap(system: System, pid: str = "p1") -> int:
    return system.producing_agents[pid].capacities[CAP]


# --- Capacity hit ---


def test_40_percent_hit() -> None:
    """Scenario "40% hit": at cycle t the hit producer holds 60% of its pre-hit capacity of
    the type; max capacity is unchanged."""
    cfg = _phase1(degradation=0)
    shocked, calm = build_system(cfg, shock_schedule=(_hit(3),)), build_system(cfg)
    for _ in range(2):  # play cycles 1 and 2; complete_cycle(2) starts cycle 3
        r_shocked, r_calm = _play(shocked), _play(calm)
    assert shocked.cycle_counter == 3
    pre = _cap(calm)
    assert pre == 120
    assert _cap(shocked) == 72 == 0.6 * pre
    assert shocked.producing_agents["p1"].max_capacities[CAP] == 120
    # the end-of-cycle-2 snapshot predates the hit
    assert r_shocked.end_capacities == r_calm.end_capacities
    snap = {s.producer_id: s for s in r_shocked.end_capacities}
    assert snap["p1"].capacities[CAP] == 120


def test_restoration_after_the_hit() -> None:
    """Scenario "40% hit" (second half): P can restore the hit capacity by investment."""
    cfg = _phase1(degradation=0)
    system = build_system(cfg, shock_schedule=(_hit(3),))
    for _ in range(2):
        _play(system)
    assert _cap(system) == 72
    for _ in range(8):
        _play(system, restore=frozenset({"p1"}))
    assert 72 < _cap(system) <= 120
    assert system.producing_agents["p1"].max_capacities[CAP] == 120


def test_removal_rounds_down() -> None:
    """Scenario "Removal rounds down": magnitude 0.4 on 7 units removes 2, leaving 5."""
    assert capacity_hit_removal(7, 0.4) == 2
    system = build_system(_phase1(degradation=0, p1_capacity=7), shock_schedule=(_hit(2),))
    _play(system)
    assert _cap(system) == 5


@pytest.mark.parametrize("capacity, magnitude, removed", [
    (100, 0.29, 29),  # float product 28.999999999999996 would remove 28
    (100, 0.57, 57),
    (10, 0.3, 3),
    (1, 0.99, 0),     # a hit never empties a positive capacity
    (2, 0.6, 1),
])
def test_removal_is_exact(capacity: int, magnitude: float, removed: int) -> None:
    assert capacity_hit_removal(capacity, magnitude) == removed
    assert capacity - removed >= 1


def test_hit_on_several_types_and_stacked_hits() -> None:
    """A hit naming several owned types reduces each; two hits in one cycle apply in order."""
    cfg = _phase1(degradation=0)
    p1 = cfg.producer_configs[0]
    two_types = replace(p1, capacities={CAP: 100, "capacity-9": 50})
    cfg = replace(cfg, producer_configs=[two_types, *cfg.producer_configs[1:]])
    system = build_system(cfg, shock_schedule=(_hit(2, 0.5, types=(CAP, "capacity-9")),
                                               _hit(2, 0.5)))
    _play(system)
    caps = system.producing_agents["p1"].capacities
    assert caps[CAP] == 25 and caps["capacity-9"] == 25


def test_schedule_checked_against_the_config() -> None:
    cfg = _phase1()
    with pytest.raises(ValueError, match=r"does not own capacity type"):
        build_system(cfg, shock_schedule=(_hit(3, types=("capacity-2",)),))
    with pytest.raises(ValueError, match=r"start cycle must be >= 2"):
        build_system(cfg, shock_schedule=(_hit(1),))
    outage = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p1"), 3, 2, 1.0)
    with pytest.raises(ValueError, match=r"no domain mechanics yet"):
        build_system(cfg, shock_schedule=(outage,))


def test_schedule_is_not_mutated() -> None:
    schedule = (_hit(2), _hit(4, 0.3))
    system = build_system(_phase1(), shock_schedule=schedule)
    for _ in range(5):
        _play(system)
    assert system._shock_schedule == schedule


# --- Domain stays RNG-free ---


def test_domain_run_from_a_schedule() -> None:
    """Scenario "Domain run from a schedule": the same config + schedule give identical
    CycleResults, with no seed anywhere."""
    cfg, schedule = _phase1(), (_hit(2, 0.5), _hit(5, 0.3), _hit(5, 0.45, producer="p2",
                                                                  types=("capacity-2",)))
    runs = []
    for _ in range(2):
        system = build_system(cfg, shock_schedule=schedule)
        runs.append([_play(system, restore=frozenset({"p1", "p2"})) for _ in range(10)])
    assert runs[0] == runs[1]
    calm = build_system(cfg)
    assert [_play(calm, restore=frozenset({"p1", "p2"})) for _ in range(10)] != runs[0]


def test_domain_imports_no_rng() -> None:
    """No module under domain/ imports random, numpy or secrets."""
    banned = {"random", "numpy", "secrets"}
    for path in sorted((PKG / "domain").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            assert not names & banned, f"{path.name} imports {sorted(names & banned)}"


def test_f32_matches_numpy_float32() -> None:
    """domain.common.f32 (stdlib struct) equals np.float32 rounding, including ties,
    subnormals, NaN, ±inf and finite overflow."""
    import math
    import random

    import numpy as np

    from tokenomics.domain.common import f32

    flt_max = float(np.finfo(np.float32).max)
    half_ulp = 2.0 ** 103  # half the float32 ulp at FLT_MAX (ulp = 2**104)
    cases = [0.0, -0.0, 1 / 3, -2 / 3, 0.1, 1e-45, 7e-46, 1.4e-45, 1e-40, -1e-40,
             1.17549435e-38, flt_max, -flt_max, flt_max + half_ulp * 0.99,
             flt_max + half_ulp, flt_max * 2, -flt_max * 2, 1e300, -1e300,
             math.inf, -math.inf,
             1 + 2 ** -24, 1 + 3 * 2 ** -24, 1 + 2 ** -24 + 2 ** -50]  # ties to even + above
    rng = random.Random(0)
    cases += [rng.uniform(-1e6, 1e6) for _ in range(2000)] + [rng.random() * 1e-38 for _ in range(500)]
    for x in cases:
        with np.errstate(over="ignore"):
            want = float(np.float32(x))
        got = f32(x)
        assert got == want and math.copysign(1, got) == math.copysign(1, want), x
    assert math.isnan(f32(math.nan))


# --- Hit after the last played cycle (env stats) ---


def _episode(monkeypatch: pytest.MonkeyPatch, schedule: tuple[ShockEvent, ...], horizon: int,
             producing: str = "stock_aware", config: Configuration | None = None,
             live: list[dict] | None = None) -> tuple[dict, dict, list[float]]:
    """Run one env episode with `schedule` injected into the domain; return the runner's
    result dict, `env.episode_stats` and `env.capacity_trace` (and append the live
    end-of-episode capacities to `live` when given). The env wiring of schedules lands in
    task 3.1; here `build_system` is wrapped to inject it."""
    real = aec_env.build_system

    def with_schedule(*args: object, **kwargs: object) -> System:
        return real(*args, **kwargs, shock_schedule=schedule)

    monkeypatch.setattr(aec_env, "build_system", with_schedule)
    env = TokenomicsAECEnv(config=config or _phase1(), horizon=horizon)
    result = run_episode(env, PRODUCING_BASELINES[producing], ORDERING_BASELINES["proportional"],
                         seed=0)
    monkeypatch.setattr(aec_env, "build_system", real)
    if live is not None:
        live.append({pid: dict(p.capacities) for pid, p in env.system.producing_agents.items()})
    return dict(result), dict(env.episode_stats), list(env.capacity_trace)


@pytest.mark.parametrize("horizon", [1, 8])
def test_hit_after_the_last_played_cycle(monkeypatch: pytest.MonkeyPatch, horizon: int) -> None:
    """Scenario "Hit after the last played cycle": a hit scheduled for H + 1 leaves every
    episode statistic equal to the unshocked run: the runner's whole result (cap_final and
    cap_min included), every env stat (r_end included) and the capacity trace."""
    live: list[dict] = []
    calm = _episode(monkeypatch, (), horizon, live=live)
    late = _episode(monkeypatch, (_hit(horizon + 1, 0.6),), horizon, live=live)
    assert late == calm
    # the hit did land on the live state after the last cycle; only the snapshot shields it
    assert live[1]["p1"][CAP] < live[0]["p1"][CAP]
    result, stats, trace = calm
    assert stats["r_end"] is not None and stats["cycles"] == horizon == len(trace)
    assert result["cap_final"] == trace[-1]
    if horizon >= 2:
        # control: the same hit inside the played window does change the episode (the test
        # is sensitive); its direction depends on the policy's restoration response
        played = _episode(monkeypatch, (_hit(horizon, 0.6),), horizon)
        assert played[1]["r_end"] != stats["r_end"]
        assert played[0]["cap_final"] != result["cap_final"]


def test_hit_after_an_early_death(monkeypatch: pytest.MonkeyPatch) -> None:
    """Early termination: with no restoration the economy dies before the horizon; a hit
    scheduled for the cycle after death leaves every statistic unchanged."""
    base = _phase1()
    # p2 starts with twice the capacity at the same absolute decay, so it still holds units
    # when p1 (and the economy) dies
    cfg = replace(base, producer_configs=[
        replace(p, capacities={"capacity-2": 240}, degradation=2) if p.id == "p2" else p
        for p in base.producer_configs])
    live: list[dict] = []
    calm = _episode(monkeypatch, (), 40, producing="noop", config=cfg, live=live)
    result, stats, _ = calm
    assert stats["dead"] and stats["cycles"] < 40
    death = stats["cycles"]
    hit = _hit(death + 1, 0.6, producer="p2", types=("capacity-2",))
    late = _episode(monkeypatch, (hit,), 40, producing="noop", config=cfg, live=live)
    assert late == calm
    # the hit did land on the live state after the death cycle; only the snapshot shields it
    before, after = live[0]["p2"]["capacity-2"], live[1]["p2"]["capacity-2"]
    assert before > 0 and after == before - capacity_hit_removal(before, 0.6)
