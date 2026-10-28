"""Shock-reactive ordering baseline and draft roster v2 (add-shocks-i-drift task 5.4, design
D8/D9; gate-harness "Shock-reactive baseline in roster v2", "Shock-reactive baseline sees no
schedule").

`shock_reactive` is `blend_0.5` with every eligible producer whose recent rejected-or-failed
quantity exceeds half its recent routed quantity masked to -BOUND, while its share of the
product's recent routed flow is at least the fresh-evidence floor (re-admission). The smoke world is registry
v2's shocked 12-producer development world with the test-only `shocks-i-smoke` profile (D7), so
its 3 events (2 capacity hits + 1 outage) fire inside horizon 12."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from typing import Callable

import numpy as np
import pytest

from tokenomics.domain.config import (ConsumerConfig, Configuration, ProcessSheet,
                                      ProducingAgentConfig, Restoration, Upgrade)
from tokenomics.domain.shocks import ShockEvent, ShockKind, ShockModel, ShockTarget
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.decode import decode_ordering
from tokenomics.domain.telemetry import EconomyTelemetry
from tokenomics.eval.baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    SHOCK_REACTIVE_READMIT_CYCLES,
    Policy,
    shock_reactive_fresh_floor,
)
from tokenomics.eval.contract import ContractMonitor
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.roster import (
    DEFAULT_ROSTER,
    JOINT_STACK,
    LEGACY,
    ROSTER_V1_ORDERING,
    ROSTER_V2,
    SHOCK_REACTIVE,
    RosterMember,
    build_roster_v1,
    build_roster_v2,
    contract_for,
    load_roster,
    member_env,
)
from tokenomics.eval.runner import run_episode
from tokenomics.eval.shock_profiles import SHOCKS_I_SMOKE

HORIZON = 12
SHOCKED_WORLD = "dev-sparse_wide-12-shocked-00"
# The two stacks task 5.4 runs on the smoke world: roster v2's mandatory legacy pair and a
# joint-stack pair.
STACKS = (("obs_stock_aware", LEGACY), ("entity_planner", JOINT_STACK))


# --- Shock-reactive baseline in roster v2 ---


def test_shock_reactive_is_an_ordering_baseline() -> None:
    assert SHOCK_REACTIVE == "shock_reactive" and SHOCK_REACTIVE in ORDERING_BASELINES


def test_shock_reactive_baseline_in_roster_v2() -> None:
    # Scenario "Shock-reactive baseline in roster v2": the draft, before pruning, pairs every
    # producing baseline with the shock-reactive router.
    r = load_roster(ROSTER_V2)
    assert (r.name, r.frozen, r.pruning, r.replicates) == ("roster-v2", False, None, 5)
    paired = {m.producing for m in r.members if m.ordering == SHOCK_REACTIVE}
    assert paired == set(PRODUCING_BASELINES) and len(paired) == 8
    for p in PRODUCING_BASELINES:
        m = r.member(f"{p}/{SHOCK_REACTIVE}")
        assert m.contract == contract_for(p, SHOCK_REACTIVE)
        assert m.random == (p == "random")
    assert contract_for("obs_stock_aware", SHOCK_REACTIVE) == LEGACY
    assert r.member("obs_stock_aware/shock_reactive").contract == LEGACY
    assert r.member("entity_planner/shock_reactive").contract == JOINT_STACK
    assert r.member("random/shock_reactive").contract == JOINT_STACK
    assert r.replicates_for(r.member("random/shock_reactive")) == 5


def test_roster_v2_is_v1_draft_plus_shock_reactive_pairs() -> None:
    v1, v2 = build_roster_v1(), build_roster_v2()
    assert set(v2.member_ids) == set(v1.member_ids) | {
        f"{p}/{SHOCK_REACTIVE}" for p in PRODUCING_BASELINES}
    assert v1.members == tuple(m for m in v2.members if m.ordering != SHOCK_REACTIVE)
    assert set(ROSTER_V1_ORDERING) == set(ORDERING_BASELINES) - {SHOCK_REACTIVE}
    # literal pins: 8 x 6 members; random = 6 (random producing) + 7 (random ordering);
    # LEGACY = 6 (obs_stock_aware) + 8 (obs_price) - 1 (counted twice)
    assert len(v2.members) == 48
    assert sum(m.random for m in v2.members) == 13
    assert sum(m.contract == LEGACY for m in v2.members) == 13


def test_packaged_roster_v2_is_the_unfrozen_draft() -> None:
    """The packaged file is exactly `build_roster_v2()` (unscreened, unpruned), pinned by its
    content hash; the v2 screen (task 9.1) prunes and freezes it."""
    assert ROSTER_V2.read_text() == build_roster_v2().to_json()
    assert load_roster(ROSTER_V2).version == (
        "a050c99e4960eb422b9eda34fd4a0de56d259629c934ccf28e4ffcb33da816fe")


# --- the smoke world ---


def _smoke_world() -> tuple[Configuration, int]:
    entry = load_registry(REGISTRY_V2).entry(SHOCKED_WORLD)
    params = {**entry.generator["params"], "shock_profile": SHOCKS_I_SMOKE}
    entry = dataclasses.replace(entry, generator={**entry.generator, "params": params})
    return entry.config(), entry.episode_seeds[0]


@pytest.fixture(scope="module")
def smoke_world() -> tuple[Configuration, int]:
    return _smoke_world()


def _env(producing: str, config: Configuration) -> TokenomicsAECEnv:
    m = RosterMember(producing, SHOCK_REACTIVE, contract_for(producing, SHOCK_REACTIVE), False)
    return member_env(m, config, HORIZON)


@pytest.mark.parametrize("producing,contract", STACKS)
def test_smoke_episode_completes_without_contract_failures(
    smoke_world: tuple[Configuration, int], producing: str, contract: str
) -> None:
    config, seed = smoke_world
    env = _env(producing, config)
    assert contract_for(producing, SHOCK_REACTIVE) == contract
    assert env.product_stream_ordering == (contract == JOINT_STACK)
    mon = ContractMonitor(env)
    stats = run_episode(env, PRODUCING_BASELINES[producing], ORDERING_BASELINES[SHOCK_REACTIVE],
                        seed=seed, monitor=mon)
    mon.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=HORIZON)  # raises on any
    assert (stats["shock_events"], stats["cycles"], stats["dead"]) == (3, HORIZON, False)


# --- Shock-reactive baseline sees no schedule ---


class ScheduleRead(AssertionError):
    """A policy touched the hidden shock schedule."""


class _Sentinel:
    """Stands in for every piece of shock state while a policy runs: any attribute access,
    iteration, length, truth test, indexing, membership, comparison or hash raises."""

    def __getattribute__(self, name: str) -> object:
        raise ScheduleRead(f"attribute {name!r}")

    def __iter__(self) -> object:
        raise ScheduleRead("__iter__")

    def __len__(self) -> int:
        raise ScheduleRead("__len__")

    def __bool__(self) -> bool:
        raise ScheduleRead("__bool__")

    def __getitem__(self, key: object) -> object:
        raise ScheduleRead("__getitem__")

    def __contains__(self, key: object) -> bool:
        raise ScheduleRead("__contains__")

    def __eq__(self, other: object) -> bool:
        raise ScheduleRead("__eq__")

    def __hash__(self) -> int:
        raise ScheduleRead("__hash__")


SENTINEL = _Sentinel()
# Domain shock state: the injected schedule and the per-cycle indexes built from it.
SYSTEM_SHOCK_STATE = ("_shock_schedule", "_hits_by_cycle", "_outages_by_cycle")
STATS_SHOCK_KEYS = ("shock_digest", "shock_events")


def _hidden(policy: Policy) -> Policy:
    """Run `policy` with every piece of shock state replaced by the sentinel: the env's drawn
    schedule, the config's shock model, the domain's schedule and indexes, and the episode
    stats' shock digest and count. Restored after the call, so the domain still replays the
    shocks between policy calls."""

    def wrapped(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        saved_env = (env._shock_schedule, env.config, env.episode_stats)
        saved_sys = {n: getattr(env.system, n) for n in SYSTEM_SHOCK_STATE}
        config = copy.copy(env.config)
        object.__setattr__(config, "shocks", SENTINEL)
        stats = dict(env.episode_stats)
        stats.update({k: SENTINEL for k in STATS_SHOCK_KEYS})
        env._shock_schedule, env.config, env.episode_stats = SENTINEL, config, stats
        for n in SYSTEM_SHOCK_STATE:
            setattr(env.system, n, SENTINEL)
        try:
            return policy(env, agent)
        finally:
            env._shock_schedule, env.config, env.episode_stats = saved_env
            for n, v in saved_sys.items():
                setattr(env.system, n, v)

    return wrapped


Log = list[tuple[int, str, np.ndarray]]


def _recorded(policy: Policy, log: Log) -> Policy:
    def wrapped(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        action = policy(env, agent)
        log.append((env.system.cycle_counter, agent, np.array(action, copy=True)))
        return action

    return wrapped


def _actions(producing: str, config: Configuration, seed: int,
             wrap: Callable[[Policy], Policy]) -> tuple[Log, Log, dict]:
    prod_log: Log = []
    order_log: Log = []
    env = _env(producing, config)
    stats = run_episode(
        env,
        _recorded(wrap(PRODUCING_BASELINES[producing]), prod_log),
        _recorded(wrap(ORDERING_BASELINES[SHOCK_REACTIVE]), order_log),
        seed=seed,
    )
    return prod_log, order_log, stats


def _same(a: Log, b: Log) -> bool:
    return len(a) == len(b) and all(
        ca == cb and xa == xb and np.array_equal(va, vb) for (ca, xa, va), (cb, xb, vb) in zip(a, b))


# Ordering calls on the smoke episode, and how many of them differ from `blend_0.5`: non-zero,
# so the masking path is exercised by the schedule-hiding check, not skipped.
REACTIVE_CALLS = {"obs_stock_aware": (180, 38), "entity_planner": (360, 33)}


@pytest.mark.parametrize("producing,contract", STACKS)
def test_shock_reactive_baseline_sees_no_schedule(
    smoke_world: tuple[Configuration, int], producing: str, contract: str
) -> None:
    # Scenario "Shock-reactive baseline sees no schedule": the same episode, once with the shock
    # schedule readable and once with it hidden from every policy, gives identical actions.
    config, seed = smoke_world
    prod_open, order_open, stats_open = _actions(producing, config, seed, lambda p: p)
    prod_hid, order_hid, stats_hid = _actions(producing, config, seed, _hidden)
    assert _same(order_open, order_hid) and _same(prod_open, prod_hid)
    assert stats_open == stats_hid and stats_open["shock_events"] == 3

    env = _env(producing, config)
    calls = differ = 0

    def compare(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        nonlocal calls, differ
        action = ORDERING_BASELINES[SHOCK_REACTIVE](env, agent)
        calls += 1
        differ += not np.array_equal(action, ORDERING_BASELINES["blend_0.5"](env, agent))
        return action

    run_episode(env, PRODUCING_BASELINES[producing], compare, seed=seed)
    assert (calls, differ) == REACTIVE_CALLS[producing]


@pytest.mark.parametrize("peek", [
    lambda env: env._shock_schedule[0],
    lambda env: len(env.system._shock_schedule),
    lambda env: env.system._hits_by_cycle.get(2),
    lambda env: 3 in env.system._outages_by_cycle,
    lambda env: env.config.shocks.events,
    lambda env: env.episode_stats["shock_events"] == 3,
    lambda env: bool(env.episode_stats["shock_digest"]),
], ids=["env-schedule", "domain-schedule", "hits", "outages", "config", "stats-events",
        "stats-digest"])
def test_hidden_schedule_catches_a_peeking_policy(
    smoke_world: tuple[Configuration, int], peek: Callable[[TokenomicsAECEnv], object]
) -> None:
    """The sentinel bites: a policy that reads any piece of shock state raises under `_hidden`,
    so identical actions in the test above mean shock_reactive read none of it."""
    config, seed = smoke_world

    def peeking(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        peek(env)
        return ORDERING_BASELINES[SHOCK_REACTIVE](env, agent)

    readable = _env("entity_planner", config)
    readable.reset(seed=seed)
    peek(readable)  # the state is there to read without the sentinel
    with pytest.raises(ScheduleRead):
        run_episode(_env("entity_planner", config), PRODUCING_BASELINES["entity_planner"],
                    _hidden(peeking), seed=seed)


# --- stops routing to a producer whose sub-orders were rejected last cycle; re-admission ---

OUTAGE_START = 2


def _outage_config(outage_cycles: int = 3) -> Configuration:
    """Two identical producers of one primal product; consumer c1 orders 4 units each cycle, so
    `blend_0.5` splits them 2/2. Producer `pa` is out from cycle 2 for `outage_cycles` cycles
    and rejects every bid meanwhile."""

    def producer(pid: str) -> ProducingAgentConfig:
        return ProducingAgentConfig(pid, {"capacity-1": 100}, 1, Restoration(1, 50, "capacity-1"),
                                    Upgrade(1, 40, "capacity-1"), produces=frozenset({1}))

    outage = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="pa"), OUTAGE_START,
                        outage_cycles, 1.0)
    return Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 1})],
        producer_configs=[producer("pa"), producer("pb")],
        consumers=[ConsumerConfig("c1", 1, quantity_min=4, quantity_max=4)],
        shocks=ShockModel(events=(outage,)),
    )


def _routed_to(env: TokenomicsAECEnv, agent: str, policy: Policy) -> dict[str, int]:
    """Units each producer would get if `policy` acted now (decoded, not stepped)."""
    _, did = env._role[agent]
    view = env.system.ordering_agents[did].view(env.system.producer_infos)
    cmd = decode_ordering(env.spec_, policy(env, agent), view)
    out: dict[str, int] = {}
    for subs in cmd.orders.values():
        for s in subs:
            out[s.producer_id] = out.get(s.producer_id, 0) + s.quantity
    return out


def _outage_routing(contract: str, outage_cycles: int, horizon: int
                    ) -> dict[int, tuple[dict[str, int], dict[str, int]]]:
    """Per played cycle, the consumer order's units per producer under shock_reactive (which
    acts) and under blend_0.5 (decoded only), on the outage fixture under `contract`."""
    config = _outage_config(outage_cycles)
    env = (TokenomicsAECEnv(config=config, horizon=horizon) if contract == LEGACY
           else member_env(RosterMember("entity_planner", SHOCK_REACTIVE, JOINT_STACK, False),
                           config, horizon))
    assert env.product_stream_ordering == (contract == JOINT_STACK)
    reactive, blend = ORDERING_BASELINES[SHOCK_REACTIVE], ORDERING_BASELINES["blend_0.5"]
    routed: dict[int, tuple[dict[str, int], dict[str, int]]] = {}
    env.reset(seed=0)
    for agent in env.agent_iter(max_iter=900):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        role, did = env._role[agent]
        if role == "prod":
            env.step(PRODUCING_BASELINES["noop"](env, agent))
            continue
        if env.system.ordering_agents[did].incoming:  # the consumer's agent / stream
            routed[env.system.cycle_counter] = (_routed_to(env, agent, reactive),
                                                _routed_to(env, agent, blend))
        env.step(reactive(env, agent))
    assert sorted(routed) == list(range(1, horizon + 1))
    return routed


BOTH = {"pa": 2, "pb": 2}
# Cycle 2's sub-orders to pa were all rejected, so cycle 3 re-routes those 2 units too (4 + 2).
REROUTED = ({"pb": 6}, {"pa": 3, "pb": 3})
EXCLUDED = ({"pb": 4}, BOTH)
SAME = (BOTH, BOTH)


@pytest.mark.parametrize("contract", [LEGACY, JOINT_STACK])
def test_stops_routing_to_a_producer_whose_sub_orders_were_rejected(contract: str) -> None:
    # Outage of pa in cycles 2-4. Cycles 1-2: nothing rejected yet, same as blend_0.5. Cycles
    # 3-5: pa's cycle-2 sub-orders were rejected, so shock_reactive routes nothing to pa while
    # blend_0.5 keeps splitting evenly. Cycle 6: pa's routed share has decayed below the fresh-
    # evidence floor (4 cycles after its last routing, within D7's 6), so it is re-admitted; the
    # outage is over, its sub-orders complete, and it stays admitted.
    routed = _outage_routing(contract, outage_cycles=3, horizon=10)
    assert [routed[c] for c in range(1, 11)] == [
        SAME, SAME, REROUTED, EXCLUDED, EXCLUDED, SAME, SAME, SAME, SAME, SAME]


@pytest.mark.parametrize("contract", [LEGACY, JOINT_STACK])
def test_readmitted_producer_still_failing_is_excluded_again(contract: str) -> None:
    # Outage of pa in cycles 2-7 (D7's longest, 6 cycles). pa is re-admitted at cycle 6 as a
    # probe while still out; that routing is rejected, so it is excluded again in cycles 7-8,
    # re-admitted at cycle 9 (2 cycles after its last routing) and, the outage over, stays.
    routed = _outage_routing(contract, outage_cycles=6, horizon=12)
    assert [routed[c] for c in range(1, 13)] == [
        SAME, SAME, REROUTED, EXCLUDED, EXCLUDED, SAME, REROUTED, EXCLUDED,
        SAME, SAME, SAME, SAME]


def test_fresh_floor_is_derived_from_the_telemetry_alpha() -> None:
    """The floor re-admits within D7's longest outage: (1 - alpha)^6 with the domain's EMA rate.
    A producer at the whole flow decays below it after 6 idle cycles, not after 5."""
    alpha = EconomyTelemetry()._alpha
    floor = shock_reactive_fresh_floor(alpha)
    assert alpha == 0.3 and SHOCK_REACTIVE_READMIT_CYCLES == 6
    assert floor == pytest.approx(0.117649)
    assert (1 - alpha) ** 5 > floor and (1 - alpha) ** 6 <= floor


def test_every_eligible_producer_excluded_keeps_the_base_logits() -> None:
    """When telemetry excludes every eligible producer of an order, the order keeps blend_0.5's
    logits rather than an all -BOUND row."""
    config = dataclasses.replace(_outage_config(), shocks=None)
    env = TokenomicsAECEnv(config=config, horizon=8)
    env.reset(seed=0)
    while env._role[env.agent_selection][0] == "prod":
        env.step(PRODUCING_BASELINES["noop"](env, env.agent_selection))
    agent = env.agent_selection
    tel = env.system.telemetry
    for pid in ("pa", "pb"):
        tel.record_routed(pid, 1, 2, 0)
        tel.record_rejected(pid, 1, 2, 0, 0)
    tel.end_cycle()
    base = ORDERING_BASELINES["blend_0.5"](env, agent)
    assert np.array_equal(ORDERING_BASELINES[SHOCK_REACTIVE](env, agent), base)
    # pb routed 3 more units without rejection: its rejected share drops below half, pa's
    # stays above and pa keeps a fresh share (0.42 of 1.74 >= floor), so only pa is excluded
    tel.record_routed("pb", 1, 3, 1)
    tel.end_cycle()
    action = ORDERING_BASELINES[SHOCK_REACTIVE](env, agent).reshape(
        env.spec_.k_stream, env.spec_.num_producers)
    pa, pb = env.spec_.producer_ids.index("pa"), env.spec_.producer_ids.index("pb")
    assert action[0, pa] == -10.0 and action[0, pb] == base.reshape(action.shape)[0, pb]


def test_roster_v1_file_and_draft_unchanged() -> None:
    """Adding shock_reactive to ORDERING_BASELINES leaves roster v1 alone: the frozen file's
    bytes and the v1 draft's version are those of base commit 669cf75."""
    assert hashlib.sha256(DEFAULT_ROSTER.read_bytes()).hexdigest() == (
        "b1640571197d2680e97337699f04b42ae5acc122089a75d0bde0829fe57e70fc")
    assert build_roster_v1().version == (
        "8791e24b2760a24bc2b40ef588578ee96b7e8e93502b76c50ef0d2fac6b83239")
