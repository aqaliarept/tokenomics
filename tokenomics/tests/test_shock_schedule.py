"""Seeded shock schedules in the env (ADR-0055; add-shocks-i-drift design D3;
stochastic-environment "Seeded event schedule", "Events are independent of agent actions",
"Shocks are unannounced", "Calm regime is preserved", "Shocks keep the agent contract").
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tokenomics.domain.config import Configuration, load_config
from tokenomics.domain.shocks import ShockEvent, ShockGenerator, ShockKind, ShockModel, ShockTarget
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.shock_schedule import build_schedule, derived_rng, shock_digest
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.runner import run_episode

TESTS = Path(__file__).resolve().parent
PHASE1 = TESTS.parent / "tokenomics" / "config" / "phase1.json"
RANDOM_DEMAND = TESTS / "fixtures" / "random_demand.json"

HIT_GEN = ShockGenerator(ShockKind.CAPACITY_HIT, count=3, start_window=(2, 20), duration=(1, 1),
                         magnitude=(0.3, 0.6))
OUTAGE_GEN = ShockGenerator(ShockKind.PRODUCER_OUTAGE, count=2, start_window=(2, 20),
                            duration=(1, 3), magnitude=(1.0, 1.0))


def _shocked(base: Path = PHASE1, *generators: ShockGenerator,
             events: tuple[ShockEvent, ...] = ()) -> Configuration:
    return replace(load_config(base), shocks=ShockModel(events=events, generators=generators))


def _hits(schedule: tuple[ShockEvent, ...]) -> tuple[ShockEvent, ...]:
    return tuple(e for e in schedule if e.kind is ShockKind.CAPACITY_HIT)


def _reset(config: Configuration, seed: int | None, horizon: int = 12) -> TokenomicsAECEnv:
    env = TokenomicsAECEnv(config=config, horizon=horizon)
    env.reset(seed=seed)
    return env


# --- Seeded event schedule ---


def test_same_seed_same_events() -> None:
    """Scenario "Same seed, same events"."""
    cfg = _shocked(PHASE1, HIT_GEN, OUTAGE_GEN)
    a, b = _reset(cfg, 7, 20), _reset(cfg, 7, 20)
    assert a._shock_schedule == b._shock_schedule
    assert a.episode_stats["shock_digest"] == b.episode_stats["shock_digest"]
    assert a.episode_stats["shock_events"] == len(a._shock_schedule) == 5
    other = _reset(cfg, 8, 20)
    assert other.episode_stats["shock_digest"] != a.episode_stats["shock_digest"]


def test_schedule_is_well_formed() -> None:
    cfg = _shocked(PHASE1, HIT_GEN, OUTAGE_GEN)
    owned = {p.id: set(p.capacities) for p in cfg.producer_configs}
    for seed in range(20):
        schedule = build_schedule(cfg, seed, 100)
        assert schedule == tuple(sorted(schedule, key=lambda e: (e.start_cycle, e.kind.value,
                                                                  e.target.producer)))
        for e in schedule:
            assert 2 <= e.start_cycle <= 20
            if e.kind is ShockKind.CAPACITY_HIT:
                assert e.duration == 1 and 0.3 <= e.magnitude <= 0.6
                assert round(e.magnitude, 4) == e.magnitude
                assert set(e.target.capacity_types) <= owned[e.target.producer]
            else:
                assert 1 <= e.duration <= 3 and e.magnitude == 1.0
        replace(cfg, shocks=ShockModel(events=schedule)).validate()  # injectable as events


def test_adding_a_kind_leaves_earlier_schedules_unchanged() -> None:
    """Scenario "Adding a kind leaves earlier schedules unchanged": the capacity-hit schedule
    is byte-identical with outages on and off."""
    hits_only, both = _shocked(PHASE1, HIT_GEN), _shocked(PHASE1, HIT_GEN, OUTAGE_GEN)
    for seed in range(10):
        a, b = build_schedule(hits_only, seed, 100), build_schedule(both, seed, 100)
        assert _hits(b) == a
        assert shock_digest(_hits(b)) == shock_digest(a)
        assert len(b) > len(a)


def test_demand_draws_preserved() -> None:
    """Scenario "Demand draws preserved": seeded random demand (B9) is identical with shocks
    on and off for the same episode seed."""
    calm = load_config(RANDOM_DEMAND)
    shocked = _shocked(RANDOM_DEMAND, HIT_GEN, OUTAGE_GEN)
    digests = set()
    for seed in (0, 1, 5):
        a, b = _reset(calm, seed, 20), _reset(shocked, seed, 20)
        assert a.episode_stats["demand_digest"] == b.episode_stats["demand_digest"]
        assert a._demand_schedules == b._demand_schedules
        assert b.episode_stats["shock_events"] > 0
        digests.add(a.episode_stats["demand_digest"])
    assert len(digests) == 3  # the demand really is seeded


def test_longer_horizon_extends_the_schedule() -> None:
    """Scenario "Longer horizon extends the schedule": horizons 10 and 20 share cycles 1-10."""
    cfg = _shocked(PHASE1, HIT_GEN, OUTAGE_GEN)
    for seed in range(10):
        short, long = build_schedule(cfg, seed, 10), build_schedule(cfg, seed, 20)
        assert short == tuple(e for e in long if e.start_cycle <= 10)
        assert all(e.start_cycle <= 10 for e in short)
    assert any(len(build_schedule(cfg, s, 20)) > len(build_schedule(cfg, s, 10)) for s in range(10))


def test_streams_are_independent_and_stable() -> None:
    a, b = derived_rng(3, "shock:capacity_hit"), derived_rng(3, "shock:capacity_hit")
    assert a.integers(1 << 30, size=8).tolist() == b.integers(1 << 30, size=8).tolist()
    c = derived_rng(3, "shock:producer_outage").integers(1 << 30, size=8).tolist()
    legacy = np.random.default_rng(3).integers(1 << 30, size=8).tolist()
    assert c != derived_rng(3, "shock:capacity_hit").integers(1 << 30, size=8).tolist()
    assert c != legacy  # never the legacy demand stream default_rng(seed)


def test_generator_needs_a_seed() -> None:
    with pytest.raises(ValueError, match=r"needs an integer episode seed"):
        _reset(_shocked(PHASE1, HIT_GEN), None)
    explicit = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p2"), 3, 2, 1.0)
    env = _reset(_shocked(PHASE1, events=(explicit,)), None)  # explicit events need none
    assert env._shock_schedule == (explicit,)


def test_explicit_and_generated_events_merge_and_cut_at_horizon() -> None:
    late = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p2"), 30, 2, 1.0)
    early = ShockEvent(ShockKind.CAPACITY_HIT, ShockTarget(producer="p1",
                                                           capacity_types=("capacity-1",)), 2, 1, 0.5)
    cfg = _shocked(PHASE1, HIT_GEN, events=(late, early))
    schedule = build_schedule(cfg, 4, 25)
    assert early in schedule and late not in schedule
    assert late in build_schedule(cfg, 4, 30)
    assert len(schedule) == 4


# --- Events are independent of agent actions ---


def _run(env: TokenomicsAECEnv, prod: str, order: str, seed: int) -> dict:
    return run_episode(env, PRODUCING_BASELINES[prod], ORDERING_BASELINES[order], seed)


def test_two_policies_one_schedule() -> None:
    """Scenario "Two policies, one schedule": two baseline stacks face the same schedule,
    fixed at reset before the first action."""
    cfg = _shocked(PHASE1, HIT_GEN, OUTAGE_GEN)
    expected = build_schedule(cfg, 11, 12)
    stats = []
    for prod, order in (("stock_aware", "proportional"), ("always_restore", "even_split")):
        env = TokenomicsAECEnv(config=cfg, horizon=12)
        env.reset(seed=11)
        at_reset = env._shock_schedule
        assert at_reset == expected and env.system._shock_schedule == expected
        s = _run(env, prod, order, 11)
        assert env._shock_schedule == at_reset == env.system._shock_schedule
        stats.append(s)
    assert stats[0]["shock_digest"] == stats[1]["shock_digest"] == shock_digest(expected)
    assert stats[0]["consumer_completed_units"] != stats[1]["consumer_completed_units"] or \
        stats[0]["return"] != stats[1]["return"]  # the stacks really differ


# --- Shocks are unannounced ---


def _observations(env: TokenomicsAECEnv, prod: Policy, order: Policy, seed: int
                  ) -> list[tuple[int, str, bytes]]:
    """(cycle being played, agent, observation bytes) for every agent turn of an episode."""
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    log: list[tuple[int, str, bytes]] = []
    for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
        obs, _, term, trunc, _ = env.last()
        log.append((env.current_cycle + 1, agent, np.asarray(obs).tobytes()))
        if term or trunc:
            env.step(None)
            continue
        policy = prod if env._role[agent][0] == "prod" else order
        env.step(policy(env, agent))
    return log


CONTRACTS = {
    "legacy": (lambda cfg: TokenomicsAECEnv(config=cfg, horizon=12), "stock_aware"),
    "joint_stack": (lambda cfg: joint_stack_env(cfg, 12), "entity_planner"),
}


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
@pytest.mark.parametrize("kind", [ShockKind.CAPACITY_HIT, ShockKind.PRODUCER_OUTAGE])
def test_observation_before_a_scheduled_shock(contract: str, kind: ShockKind) -> None:
    """Scenario "Observation before a scheduled shock": every agent's observation at every
    cycle before t is byte-identical with and without that shock."""
    t = 6
    if kind is ShockKind.CAPACITY_HIT:
        shock = ShockEvent(kind, ShockTarget(producer="p4", capacity_types=("capacity-4",)), t, 1, 0.6)
    else:
        shock = ShockEvent(kind, ShockTarget(producer="p4"), t, 3, 1.0)
    other = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p2"), 9, 2, 1.0)
    make, prod = CONTRACTS[contract]
    policies = (PRODUCING_BASELINES[prod], ORDERING_BASELINES["proportional"])
    base = load_config(PHASE1)
    without = _observations(make(replace(base, shocks=ShockModel(events=(other,)))), *policies, 3)
    with_ = _observations(make(replace(base, shocks=ShockModel(events=(shock, other)))), *policies, 3)
    before = [e for e in without if e[0] < t]
    assert before and before == [e for e in with_ if e[0] < t]
    assert without != with_  # the shock is observed once it takes effect


# --- Shocks keep the agent contract / Calm regime is preserved ---


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_spaces_with_and_without_shocks(contract: str) -> None:
    """Scenario "Spaces with and without shocks"."""
    make, _ = CONTRACTS[contract]
    calm, shocked = make(load_config(PHASE1)), make(_shocked(PHASE1, HIT_GEN, OUTAGE_GEN))
    assert calm.possible_agents == shocked.possible_agents
    for a in calm.possible_agents:
        assert calm.observation_space(a) == shocked.observation_space(a)
        assert calm.action_space(a) == shocked.action_space(a)


def test_disabled_stochasticity() -> None:
    """Scenario "Disabled stochasticity" (the goldens pin the legacy numbers): no shocks and
    an empty shock model run identically, with an empty schedule."""
    base = load_config(PHASE1)
    a = _run(TokenomicsAECEnv(config=base, horizon=12), "stock_aware", "proportional", 0)
    b = _run(TokenomicsAECEnv(config=replace(base, shocks=ShockModel()), horizon=12),
             "stock_aware", "proportional", 0)
    assert a == b
    assert a["shock_events"] == 0 and a["shock_digest"] == shock_digest(())


@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_shocked_episode_completes(contract: str) -> None:
    """Smoke: a generated schedule with both kinds runs to the horizon under each contract."""
    make, prod = CONTRACTS[contract]
    cfg = _shocked(PHASE1, replace(HIT_GEN, start_window=(2, 8)),
                   replace(OUTAGE_GEN, start_window=(2, 8)))
    env = make(cfg)
    s = _run(env, prod, "proportional", 2)
    kinds = {e.kind for e in env._shock_schedule}
    assert kinds == {ShockKind.CAPACITY_HIT, ShockKind.PRODUCER_OUTAGE}
    assert s["shock_events"] == 5 and s["cycles"] == 12


# --- canonical digest (review concern 157) ---


def test_equal_schedules_digest_equal_across_numeric_forms_and_roundtrip() -> None:
    """Equal events digest identically whatever the numeric form (`magnitude=1` vs the JSON
    loader's `1.0`), including after a config JSON round-trip; the canonical payload and
    its hash are pinned independently."""
    import hashlib

    from tokenomics.domain.shocks import shock_model_from_dict, shock_model_to_dict

    as_int = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p2"), 3, 2, 1)
    as_float = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p2"), 3, 2, 1.0)
    model = ShockModel(events=(as_int,))
    tripped = shock_model_from_dict(shock_model_to_dict(model)).events
    assert as_int == as_float == tripped[0]
    payload = ('[{"duration":2,"kind":"producer_outage","magnitude":1.0,"startCycle":3,'
               '"target":{"producer":"p2"}}]')
    want = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert want == "6b1a5d3ba788584cd5ac680b6223d24b66f8a65dacde0ee6b72da1e3e96b9d69"
    assert shock_digest((as_int,)) == shock_digest((as_float,)) == shock_digest(tripped) == want
    assert shock_digest(()) == hashlib.sha256(b"[]").hexdigest()


# --- pinned generation (review concern 158) ---


def _pinned_config() -> Configuration:
    """phase1 with p1 owning two capacity types (so the sorted owned-type draw matters),
    hits restricted to p1/p4, outages over every producer."""
    base = load_config(PHASE1)
    p1 = base.producer_configs[0]
    # insertion order deliberately unsorted, so dropping the `sorted` in the owned-type draw
    # changes the pinned events
    producers = [replace(p1, capacities={"capacity-9": 50, "capacity-1": 120}),
                 *base.producer_configs[1:]]
    hits = ShockGenerator(ShockKind.CAPACITY_HIT, count=4, start_window=(2, 20), duration=(1, 1),
                          magnitude=(0.3, 0.6), targets=("p1", "p4"))
    outages = ShockGenerator(ShockKind.PRODUCER_OUTAGE, count=3, start_window=(2, 20),
                             duration=(1, 3), magnitude=(1.0, 1.0))
    return replace(base, producer_configs=producers,
                   shocks=ShockModel(generators=(hits, outages)))


# (kind, producer, capacity types, start, duration, magnitude) for episode seed 5, as drawn
# by D3: per-kind stream "shock:<kind>"; per event start, duration, magnitude (4 dp), target
# producer, then (hits) one owned type from the sorted owned types; sorted by start/kind/target.
PINNED_SEED5 = [
    ("capacity_hit", "p4", ("capacity-4",), 3, 1, 0.5163),
    ("capacity_hit", "p1", ("capacity-1",), 4, 1, 0.5156),
    ("producer_outage", "p1", (), 4, 2, 1.0),
    ("capacity_hit", "p4", ("capacity-4",), 7, 1, 0.4699),
    ("capacity_hit", "p1", ("capacity-9",), 10, 1, 0.4165),
    ("producer_outage", "p2", (), 15, 3, 1.0),
    ("producer_outage", "p2", (), 17, 3, 1.0),
]


def test_pinned_generation_per_kind_stream_and_draw_order() -> None:
    """Exact events for a fixed seed: guards the per-kind stream names, the per-event draw
    order, producer choice and the sorted owned-type choice against refactors."""
    cfg = _pinned_config()
    cfg.validate()
    got = [(e.kind.value, e.target.producer, e.target.capacity_types, e.start_cycle,
            e.duration, e.magnitude) for e in build_schedule(cfg, 5, 100)]
    assert got == PINNED_SEED5
    assert {e[2] for e in got if e[1] == "p1" and e[0] == "capacity_hit"} == {
        ("capacity-1",), ("capacity-9",)}
    assert shock_digest(build_schedule(cfg, 5, 100)) == (
        "71ef0770de747d50985f5671f1b9d54b987fa9e5f03313b2d35b946afda0280a")


def test_hit_stream_is_the_hit_generators_own() -> None:
    """The capacity-hit events come from stream "shock:capacity_hit" with this exact draw
    sequence: rebuilt here from derived_rng alone, independently of generate_events."""
    cfg = _pinned_config()
    rng = derived_rng(5, "shock:capacity_hit")
    expected = []
    for _ in range(4):
        start = int(rng.integers(2, 21))
        duration = int(rng.integers(1, 2))
        magnitude = round(float(rng.uniform(0.3, 0.6)), 4)
        producer = ("p1", "p4")[int(rng.integers(2))]
        types = sorted(next(p for p in cfg.producer_configs if p.id == producer).capacities)
        expected.append((producer, (types[int(rng.integers(len(types)))],), start, duration,
                         magnitude))
    got = [(e.target.producer, e.target.capacity_types, e.start_cycle, e.duration, e.magnitude)
           for e in build_schedule(cfg, 5, 100) if e.kind is ShockKind.CAPACITY_HIT]
    assert got == sorted(expected, key=lambda x: (x[2], x[0], x[1]))


@pytest.mark.parametrize("lo, hi", [(1e-5, 4e-5), (0.99996, 0.99999), (0.30004, 0.30004),
                                     (0.12344, 0.12346)])
def test_magnitude_rounding_is_clamped_to_the_range(lo: float, hi: float) -> None:
    """4-dp rounding can leave the configured range (to 0.0, to 1.0, or past a narrow
    endpoint); the generated magnitude is clamped back inside, and stays a valid hit."""
    from tokenomics.env.shock_schedule import _magnitude

    rng = np.random.default_rng(0)
    for _ in range(200):
        m = _magnitude(rng, lo, hi)
        assert lo <= m <= hi
    gen = ShockGenerator(ShockKind.CAPACITY_HIT, count=20, start_window=(2, 9), duration=(1, 1),
                         magnitude=(lo, hi))
    cfg = _shocked(PHASE1, gen)
    schedule = build_schedule(cfg, 1, 100)
    assert all(lo <= e.magnitude <= hi and 0.0 < e.magnitude < 1.0 for e in schedule)
    replace(cfg, shocks=ShockModel(events=schedule)).validate()
