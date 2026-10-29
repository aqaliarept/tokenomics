"""Teacher information set: observable history, shock posterior, posterior forks and demand
continuity (task 6.1; ADR-0056; add-m3-development-audit design D4; development-audit "Teacher
labels depend only on observable information", "Unresolved labels fall back", "Forked demand
keeps continuity")."""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from tokenomics.domain import producing_agent as pa
from tokenomics.domain.config import Configuration, DemandPattern
from tokenomics.domain.shocks import ShockEvent, ShockGenerator, ShockKind, ShockModel, ShockTarget
from tokenomics.domain.system import capacity_hit_removal
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env import shock_posterior as sp
from tokenomics.env.demand_digest import demand_digest
from tokenomics.env.demand_drift import drift_budget, draw_schedule_entry
from tokenomics.env.demand_patterns import seasonal_factor
from tokenomics.env.future_fork import continuation_seed, fork_future_demand
from tokenomics.env.shock_schedule import _magnitude, generate_events
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.bounded_rollout_orderer import (BoundedScenarioOrderer, StreamCandidate,
                                                     _one_cycle_bump)
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.orderer_oracle import _rollout_units
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.routing_diagnostics import DIAGNOSTIC_ROUTERS
from tokenomics.eval.shock_profiles import SHOCKS_I_SMOKE

Policy = Callable[[TokenomicsAECEnv, str], np.ndarray]
PROD = PRODUCING_BASELINES["entity_planner"]
ROUND_ROBIN = DIAGNOSTIC_ROUTERS["round_robin"]
SMOKE_WORLD = "dev-sparse_wide-12-shocked-00"
HORIZON = 12


def _entry_config(world: str, profile: str | None = None) -> tuple[Configuration, int]:
    entry = load_registry(REGISTRY_V2).entry(world)
    if profile is not None:
        params = {**entry.generator["params"], "shock_profile": profile}
        entry = dataclasses.replace(entry, generator={**entry.generator, "params": params})
    return entry.config(), entry.episode_seeds[0]


@pytest.fixture(scope="module")
def smoke() -> Configuration:
    """A 12-producer shocks-i world with onsets at cycles 2-8 and drift j = 0.34."""
    return _entry_config(SMOKE_WORLD, SHOCKS_I_SMOKE)[0]


def _inject(env: TokenomicsAECEnv, events: Sequence[ShockEvent]) -> None:
    """Replace the drawn schedule with `events` right after reset (a test-only truth)."""
    events = sp.sort_events(events)
    sp._set_schedule(env.system, events, 0)
    env._shock_schedule = events


def _play(config: Configuration, seed: int, stop: int, *, horizon: int = HORIZON,
          events: Sequence[ShockEvent] | None = None, keep_replay: bool = True,
          wrap: Callable[[Policy], Policy] = lambda p: p,
          ) -> tuple[TokenomicsAECEnv, sp.HistoryRecorder, str]:
    """entity_planner / round_robin to the first orderer turn of cycle `stop`."""
    env = joint_stack_env(config, horizon)
    rec = sp.HistoryRecorder(keep_replay=keep_replay)
    prod, order = wrap(rec.wrap(PROD)), wrap(rec.wrap(ROUND_ROBIN))
    env.reset(seed=seed)
    if events is not None:
        _inject(env, events)
    for agent in env.agent_iter():
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        if role == "order" and env.system.cycle_counter == stop:
            wrap(lambda e, a: rec.observe(e, a))(env, agent)
            return env, rec, agent
        env.step(prod(env, agent) if role == "prod" else order(env, agent))
    raise AssertionError(f"episode ended before cycle {stop}")


def _blocked(h: sp.ObservedHistory) -> list[tuple[int, str]]:
    return [(r.cycle, p) for r in h.records if r.status for p, s in r.status.items()
            if s == sp.BLOCKED]


def _hits(h: sp.ObservedHistory) -> list[tuple[int, tuple[str, str], int]]:
    return [(r.cycle, cell, v) for r in h.records for cell, v in sorted(r.removal.items())]


# --- the history extractor ---


def test_history_recovers_the_true_consequences(smoke: Configuration) -> None:
    env, rec, _ = _play(smoke, 0, 11)
    h = rec.history(env)
    true = env._shock_schedule
    hits = {(e.start_cycle, (e.target.producer, e.target.capacity_types[0]))
            for e in true if e.kind is ShockKind.CAPACITY_HIT}
    assert {(c, cell) for c, cell, _ in _hits(h)} == hits
    (out,) = [e for e in true if e.kind is ShockKind.PRODUCER_OUTAGE]
    active = set(range(out.start_cycle, out.start_cycle + out.duration))
    for c, p in _blocked(h):
        assert p == out.target.producer and c in active
    assert _blocked(h)  # the outage was seen
    assert sp.consistent(true, h) and sp.replay_reproduces(rec, true)


class ShockStateRead(AssertionError):
    pass


class _Sentinel:
    """Raises on any use: stands in for every piece of hidden shock state."""

    def __getattribute__(self, name: str) -> object:
        raise ShockStateRead(f"attribute {name!r}")

    def __iter__(self) -> object:
        raise ShockStateRead("__iter__")

    def __len__(self) -> int:
        raise ShockStateRead("__len__")

    def __bool__(self) -> bool:
        raise ShockStateRead("__bool__")

    def __getitem__(self, key: object) -> object:
        raise ShockStateRead("__getitem__")

    def __contains__(self, key: object) -> bool:
        raise ShockStateRead("__contains__")

    def __eq__(self, other: object) -> bool:
        raise ShockStateRead("__eq__")

    def __hash__(self) -> int:
        raise ShockStateRead("__hash__")

    def __deepcopy__(self, memo: dict) -> object:
        raise ShockStateRead("__deepcopy__")


SENTINEL = _Sentinel()


def _hidden(policy: Policy) -> Policy:
    """Run `policy` (here: the recorder and the policy it wraps) with the env's schedule, the
    config's shock model, the domain's schedule and indexes and the stats' shock digest and
    count replaced by the sentinel; restored afterwards, so the domain still replays them."""
    def wrapped(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        saved_env = (env._shock_schedule, env.config, env.episode_stats)
        saved_sys = {n: getattr(env.system, n) for n in sp.SYSTEM_SHOCK_FIELDS}
        config = copy.copy(env.config)
        object.__setattr__(config, "shocks", SENTINEL)
        stats = dict(env.episode_stats)
        stats.update({k: SENTINEL for k in sp.STATS_SHOCK_KEYS})
        env._shock_schedule, env.config, env.episode_stats = SENTINEL, config, stats
        for n in sp.SYSTEM_SHOCK_FIELDS:
            setattr(env.system, n, SENTINEL)
        try:
            return policy(env, agent)
        finally:
            env._shock_schedule, env.config, env.episode_stats = saved_env
            for n, v in saved_sys.items():
                setattr(env.system, n, v)
    return wrapped


def test_extractor_never_reads_hidden_shock_state(smoke: Configuration) -> None:
    """The recorder (observation, stripped copies, one-cycle replays, reset snapshot) runs
    under the sentinel and yields the same history as without it."""
    env_a, rec_a, _ = _play(smoke, 0, 11, wrap=_hidden)
    env_b, rec_b, _ = _play(smoke, 0, 11)
    assert rec_a.history(env_a) == rec_b.history(env_b)
    assert [(t.cycle, t.agent, t.obs, t.telemetry) for t in rec_a.turns] == \
        [(t.cycle, t.agent, t.obs, t.telemetry) for t in rec_b.turns]

    def peek(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        len(env.system._outages_by_cycle)
        return ROUND_ROBIN(env, agent)

    with pytest.raises(ShockStateRead):  # the sentinel bites
        _play(smoke, 0, 4, wrap=lambda p: _hidden(lambda e, a: (peek(e, a), p(e, a))[1]))


# --- posterior, labels, forks ---


def _labels(config: Configuration, env: TokenomicsAECEnv, rec: sp.HistoryRecorder, agent: str,
            *, task_id: str = "t") -> tuple[bytes, tuple, tuple, list[tuple]]:
    """The teacher label of the current orderer turn: EXP-0042's bounded rollout rule with
    posterior forks as its scenarios (depth 1), wrapped with the round_robin floor."""
    source = sp.PosteriorScenarioSource(rec, sp.ShockPosterior(config), task_id=task_id,
                                        episode_seed=0)
    teacher = BoundedScenarioOrderer(task_id=task_id, episode_seed=0, prod=PROD, depth=1,
                                     base=ROUND_ROBIN, ambiguity_prob_gap=1.01)

    def score(e: TokenomicsAECEnv, a: str, streams: Sequence[StreamCandidate],
              cid: int) -> Mapping[int, np.ndarray]:
        forked = source.fork(e, cid)
        base = _rollout_units(copy.deepcopy(forked), PROD, ROUND_ROBIN, 1)
        return {s.stream_index: np.asarray([
            float(_rollout_units(copy.deepcopy(forked), PROD, _one_cycle_bump(
                ROUND_ROBIN, a, s.stream_index, col, 1.0, e.current_cycle), 1) - base)
            for col in s.eligible_cols]) for s in streams}

    teacher.scenario_scorer = score
    action = sp.with_floor(teacher, ROUND_ROBIN, source)(env, agent)
    sample = source.sample(env)
    forks = tuple(source.fork(env, i)._shock_schedule for i in range(len(sample.particles)))
    return (np.asarray(action).tobytes(), tuple(p.events for p in sample.particles), forks,
            [dataclasses.astuple(r) for r in source.log] + [dataclasses.astuple(teacher.telemetry)[:-1]])


def test_two_envs_differ_only_in_hidden_state(smoke: Configuration) -> None:
    """Scenario "Two envs differ only in hidden state": same observable history to cycle 5,
    a different true remaining outage and a different future hit -> byte-identical sampled
    scenarios, forks and teacher labels for the same fork seed."""
    hit = lambda p, c, m: ShockEvent(ShockKind.CAPACITY_HIT, ShockTarget(p, ("capacity-0",)), c, 1, m)
    out = lambda d: ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget("p2_1"), 3, d, 1.0)
    a = (hit("p0_1", 2, 0.4), out(3), hit("p0_2", 7, 0.5))
    b = (hit("p0_1", 2, 0.4), out(6), hit("p0_3", 8, 0.35))
    env_a, rec_a, agent_a = _play(smoke, 0, 5, events=a)
    env_b, rec_b, agent_b = _play(smoke, 0, 5, events=b)
    assert agent_a == agent_b and rec_a.history(env_a) == rec_b.history(env_b)
    assert (4, "p2_1") in _blocked(rec_a.history(env_a))  # cycle 3: no bids, `free`
    la, lb = _labels(smoke, env_a, rec_a, agent_a), _labels(smoke, env_b, rec_b, agent_b)
    assert la == lb
    assert len(la[1]) == sp.MIN_ACCEPTED and la[3][0][4] is False  # resolved
    assert la[3][1][4] > 0  # the teacher searched (rollouts on posterior forks)


def test_hidden_rejection_reasons_scrambled(smoke: Configuration,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario "Hidden rejection reasons scrambled": the task 1.1 reasons replaced by a wrong
    or an empty value leave the history and the labels byte-identical."""
    env, rec, agent = _play(smoke, 0, 7)
    ref = (rec.history(env), _labels(smoke, env, rec, agent))
    produce = pa.ProducingAgent.produce
    for scramble in (lambda r: [pa.REJECT_OUTAGE if x != pa.REJECT_OUTAGE else pa.REJECT_NO_FIT
                                for x in r], lambda r: [""] * len(r)):
        def scrambled(self: pa.ProducingAgent, outage: bool = False,
                      scramble: Callable = scramble) -> pa.ProductionResult:
            res = produce(self, outage)
            return dataclasses.replace(res, rejection_reasons=scramble(res.rejection_reasons),
                                       failure_reasons=scramble(res.failure_reasons))
        monkeypatch.setattr(pa.ProducingAgent, "produce", scrambled)
        env2, rec2, agent2 = _play(smoke, 0, 7)
        assert (rec2.history(env2), _labels(smoke, env2, rec2, agent2)) == ref
        monkeypatch.setattr(pa.ProducingAgent, "produce", produce)


def test_hidden_fields_stripped(smoke: Configuration) -> None:
    """Scenario "Hidden fields stripped": a fork carries the particle's future and nothing of
    the true schedule (no pending event, no outage end cycle, no digest of it)."""
    env, rec, _ = _play(smoke, 2, 4)
    t = env.system.cycle_counter
    true_future = [e for e in env._shock_schedule
                   if e.start_cycle > t or (e.kind is ShockKind.PRODUCER_OUTAGE
                                            and e.start_cycle + e.duration - 1 >= t)]
    assert true_future
    bare = sp.stripped_env(env)
    assert (bare._shock_schedule, bare.system._shock_schedule) == ((), ())
    assert bare.system._hits_by_cycle == {} and bare.system._outages_by_cycle == {}
    assert bare.episode_stats["shock_digest"] != env.episode_stats["shock_digest"]
    mine = sp.Particle(sp.sort_events([
        ShockEvent(ShockKind.CAPACITY_HIT, ShockTarget("p1_0", ("capacity-1",)), 9, 1, 0.3),
        ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget("p0_0"), 7, 2, 1.0)]))
    fork = sp.posterior_fork(env, mine, task_id="t", decision_digest=1, continuation_id=0)
    assert fork.system._shock_schedule == mine.future(t) == fork._shock_schedule
    assert not set(true_future) & set(fork.system._shock_schedule)
    assert fork.system._outages_by_cycle == {7: {"p0_0"}, 8: {"p0_0"}}
    assert set(fork.system._hits_by_cycle) == {9}
    # no true future demand either (review concern 337): the reset-time demand record and its
    # digest describe the fork's own resampled schedules
    for cid, c in fork.system.consumers.items():
        if hasattr(c, "_schedule"):
            assert fork._demand_schedules[cid] == tuple(c._schedule)
    assert fork._demand_schedules != env._demand_schedules
    assert fork.episode_stats["demand_digest"] == demand_digest(fork._demand_schedules)
    assert fork.episode_stats["demand_digest"] != env.episode_stats["demand_digest"]
    # the source env is untouched
    assert env.system._shock_schedule == env._shock_schedule and true_future


def test_ambiguous_consequence(smoke: Configuration) -> None:
    """Scenario "Ambiguous consequence": a producer that rejected every bid of a cycle while
    full (`free`) is explained both by an outage and by its lack of capacity; both kinds of
    draw are kept, and every kept draw reproduces every other consequence (replay)."""
    gens = (ShockGenerator(ShockKind.CAPACITY_HIT, 1, (2, 4), (1, 1), (0.3, 0.6)),
            ShockGenerator(ShockKind.PRODUCER_OUTAGE, 2, (2, 10), (1, 2), (1.0, 1.0)))
    cfg = dataclasses.replace(smoke, shocks=ShockModel(generators=gens))
    env, rec, _ = _play(cfg, 0, 11)
    h = rec.history(env)
    bids: dict[int, set[str]] = {}
    free = [(r.cycle, p) for r in h.records if r.status for p, s in r.status.items()
            if s == sp.FREE]
    probe = sp.HistoryRecorder()
    original = probe._finalize

    def capture(live: object) -> None:
        bids[probe._pending.cycle_counter] = {p for p, a in probe._pending.producing_agents.items()
                                              if a.bids}
        original(live)

    probe._finalize = capture
    _play_with(cfg, 0, 11, probe)
    full = [(c, p) for c, p in free if p in bids.get(c, set())]
    assert full, "fixture: no full producer"
    post = sp.ShockPosterior(cfg, budget=4096, minimum=600)
    sample = post.sample(h, np.random.default_rng(0))
    assert not sample.unresolved
    c, p = full[0]
    covers = [any(e.kind is ShockKind.PRODUCER_OUTAGE and e.target.producer == p
                  and e.start_cycle <= c < e.start_cycle + e.duration for e in s.events)
              for s in sample.particles]
    assert any(covers) and not all(covers)
    for s in sample.particles[:40]:
        assert sp.replay_reproduces(rec, s.events)


def _play_with(config: Configuration, seed: int, stop: int, rec: sp.HistoryRecorder) -> None:
    env = joint_stack_env(config, HORIZON)
    prod, order = rec.wrap(PROD), rec.wrap(ROUND_ROBIN)
    env.reset(seed=seed)
    for agent in env.agent_iter():
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if env._role[agent][0] == "order" and env.system.cycle_counter == stop:
            return
        env.step(prod(env, agent) if env._role[agent][0] == "prod" else order(env, agent))


def _perturb(events: tuple[ShockEvent, ...], rng: np.random.Generator) -> tuple[ShockEvent, ...]:
    i = int(rng.integers(len(events)))
    e = events[i]
    shift = int(rng.choice([-1, 1]))
    if e.kind is ShockKind.CAPACITY_HIT and rng.random() < 0.5:
        e = dataclasses.replace(e, magnitude=round(min(0.6, max(0.3, e.magnitude + 0.05 * shift)), 4))
    elif e.kind is ShockKind.PRODUCER_OUTAGE and rng.random() < 0.5:
        e = dataclasses.replace(e, duration=max(1, e.duration + shift))
    else:
        e = dataclasses.replace(e, start_cycle=max(2, e.start_cycle + shift))
    return sp.sort_events(events[:i] + (e,) + events[i + 1:])


def test_predicates_agree_with_the_replay_reference(smoke: Configuration) -> None:
    """The consequence-level predicates equal the replay from reset on fixtures: the truth,
    posterior draws, prior draws and perturbed draws, accepted and rejected."""
    rng = np.random.default_rng(7)
    seen = {True: 0, False: 0}
    for seed, stop in ((0, 6), (1, 9), (2, 11), (3, 8)):
        env, rec, _ = _play(smoke, seed, stop)
        h = rec.history(env)
        post = sp.ShockPosterior(smoke)
        drawn = [p.events for p in post.sample(h, np.random.default_rng(seed)).particles[:4]]
        prior = [sp.sort_events(e for g in smoke.shocks.generators
                                for e in generate_events(g, smoke, 1000 + i)) for i in range(4)]
        cases = [tuple(env._shock_schedule)] + drawn + prior + [_perturb(d, rng) for d in drawn]
        for events in cases:
            fast = sp.consistent(events, h)
            assert fast == sp.replay_reproduces(rec, events), (seed, stop, events)
            seen[fast] += 1
    assert seen[True] >= 10 and seen[False] >= 10


def test_hand_built_wrong_draws_are_rejected(smoke: Configuration) -> None:
    """Each side of each predicate on its own (review concern 336): the truth plus one wrong
    change is rejected by the predicates and by the replay alike."""
    env, rec, _ = _play(smoke, 0, 11)
    h = rec.history(env)
    true = tuple(env._shock_schedule)
    (out,) = [e for e in true if e.kind is ShockKind.PRODUCER_OUTAGE]
    hit = next(e for e in true if e.kind is ShockKind.CAPACITY_HIT)
    cell = (hit.target.producer, hit.target.capacity_types[0])
    pre = h.records[hit.start_cycle - 1].pre_hit[cell]
    wrong_m = next(m for m in (0.3, 0.6) if capacity_hit_removal(pre, m) != h.records[
        hit.start_cycle - 1].removal[cell])
    normal = next((r.cycle, p) for r in h.records if r.status and r.cycle >= 2
                  for p, s in sorted(r.status.items()) if s == sp.NORMAL)
    wrong = {
        "unobserved past hit": true + (ShockEvent(ShockKind.CAPACITY_HIT,
                                                  ShockTarget("p0_0", ("capacity-0",)), 5, 1, 0.5),),
        "outage over a normal cycle": true + (ShockEvent(ShockKind.PRODUCER_OUTAGE,
                                                         ShockTarget(normal[1]), normal[0], 1, 1.0),),
        "wrong magnitude": tuple(dataclasses.replace(e, magnitude=wrong_m) if e is hit else e
                                 for e in true),
        "outage shortened": tuple(dataclasses.replace(e, duration=e.duration - 1) if e is out else e
                                  for e in true),
    }
    assert sp.consistent(true, h) and sp.replay_reproduces(rec, true)
    for name, events in wrong.items():
        events = sp.sort_events(events)
        assert not sp.consistent(events, h), name
        assert not sp.replay_reproduces(rec, events), name


def test_validation_guards_the_proposal(smoke: Configuration) -> None:
    """The proposal is consistent by construction, so validation is a guard: a corrupted plan
    (truncated outages that touch normal cycles) lowers the rate, and every kept draw is still
    consistent and reproduced by the replay."""
    gens = (ShockGenerator(ShockKind.CAPACITY_HIT, 2, (2, 8), (1, 1), (0.3, 0.6)),
            ShockGenerator(ShockKind.PRODUCER_OUTAGE, 4, (2, 8), (3, 6), (1.0, 1.0)))
    cfg = dataclasses.replace(smoke, shocks=ShockModel(generators=gens))
    env, rec, _ = _play(cfg, 0, 5)
    h = rec.history(env)
    post = sp.ShockPosterior(cfg)
    assert post.sample(h, np.random.default_rng(0)).rate == 1.0
    plan = post._plan(h)
    assert len(plan.runs) < 4  # a truncated outage slot is drawn
    plan.trunc_outages = [(p, s, d) for p in post.producers for s in range(2, 9) for d in range(3, 7)]
    post._plan = lambda _h: plan
    s = post.sample(h, np.random.default_rng(0))
    assert 0 < s.rate < 1 and not s.unresolved
    for p in s.particles:
        assert sp.consistent(p.events, h) and sp.replay_reproduces(rec, p.events)


@pytest.mark.parametrize("world", ["dev-sparse_wide-12-shocked-00", "dev-sparse_wide-42-shocked-00"])
def test_sampler_meets_the_floor_at_cycle_90(world: str) -> None:
    """A shocks-i world played by entity_planner / round_robin to cycle 90, with at least one
    observed hit and one observed outage: 16 accepted of at most 1,024 draws."""
    config, seed = _entry_config(world)
    env, rec, _ = _play(config, seed, 90, horizon=100, keep_replay=False)
    h = rec.history(env)
    assert _hits(h) and _blocked(h)
    source = sp.PosteriorScenarioSource(rec, sp.ShockPosterior(config), task_id=world,
                                        episode_seed=seed)
    s = source.sample(env)
    assert not s.unresolved and s.accepted >= sp.MIN_ACCEPTED and s.draws <= sp.DRAW_BUDGET
    assert source.sample(env) is s and len(source.log) == 1  # cached per decision cycle
    assert source.log[0].rate == s.rate > 0


def _prior_draw(gens: Sequence[ShockGenerator], cfg: Configuration,
                rng: np.random.Generator) -> tuple[ShockEvent, ...]:
    """One schedule from the generators' prior, `generate_events`' draw order on one stream."""
    owned = {p.id: sorted(p.capacities) for p in cfg.producer_configs}
    events = []
    for g in gens:
        targets = g.targets or tuple(owned)
        for _ in range(g.count):
            start = int(rng.integers(g.start_window[0], g.start_window[1] + 1))
            duration = int(rng.integers(g.duration[0], g.duration[1] + 1))
            magnitude = _magnitude(rng, *g.magnitude)
            pid = targets[int(rng.integers(len(targets)))]
            types = (owned[pid][int(rng.integers(len(owned[pid])))],) \
                if g.kind is ShockKind.CAPACITY_HIT else ()
            events.append(ShockEvent(g.kind, ShockTarget(pid, types), start, duration, magnitude))
    return sp.sort_events(events)


def test_constructive_draws_match_prior_rejection(smoke: Configuration) -> None:
    """On a tiny fixture where prior rejection is feasible (two targets, short windows), the
    constructive sampler's draws match prior-rejection draws in distribution: two-sample KS on
    magnitude, chi-squared on start and target (p > 0.01, fixed seeds)."""
    stats = pytest.importorskip("scipy.stats")
    targets = ("p0_0", "p0_1")
    gens = (ShockGenerator(ShockKind.CAPACITY_HIT, 1, (2, 4), (1, 1), (0.3, 0.6), targets),
            ShockGenerator(ShockKind.PRODUCER_OUTAGE, 1, (3, 8), (1, 3), (1.0, 1.0), targets))
    cfg = dataclasses.replace(smoke, shocks=ShockModel(generators=gens))
    env, rec, _ = _play(cfg, 4, 5, keep_replay=False)  # hit observed, outage still ahead
    h = rec.history(env)
    assert _hits(h) and not _blocked(h)
    n = 300
    constructive = sp.ShockPosterior(cfg, budget=10 * n, minimum=n).sample(
        h, np.random.default_rng(11)).particles
    assert len(constructive) == n
    prior: list[tuple[ShockEvent, ...]] = []
    rng = np.random.default_rng(12)
    for i in range(400_000):
        events = _prior_draw(gens, cfg, rng)
        if sp.consistent(events, h):
            prior.append(events)
            if len(prior) == n:
                break
    assert len(prior) == n
    a = [p.events for p in constructive]

    def column(rows: Sequence[tuple[ShockEvent, ...]], kind: ShockKind,
               f: Callable[[ShockEvent], object]) -> list[object]:
        return [f(e) for r in rows for e in r if e.kind is kind]

    for kind in (ShockKind.CAPACITY_HIT, ShockKind.PRODUCER_OUTAGE):
        for f in (lambda e: e.start_cycle, lambda e: e.target.producer):
            x, y = column(a, kind, f), column(prior, kind, f)
            cats = sorted(set(x) | set(y), key=str)
            table = np.array([[x.count(c) for c in cats], [y.count(c) for c in cats]])
            if len(cats) > 1:
                assert stats.chi2_contingency(table).pvalue > 0.01, (kind, cats, table)
    mx, my = (column(r, ShockKind.CAPACITY_HIT, lambda e: e.magnitude) for r in (a, prior))
    assert stats.ks_2samp(mx, my).pvalue > 0.01
    dx, dy = (column(r, ShockKind.PRODUCER_OUTAGE, lambda e: e.duration) for r in (a, prior))
    cats = sorted(set(dx) | set(dy))
    assert stats.chi2_contingency(np.array([[dx.count(c) for c in cats],
                                            [dy.count(c) for c in cats]])).pvalue > 0.01


@pytest.mark.parametrize("seed, stop, run_open", [(1, 7, True), (4, 10, False)])
def test_outage_run_matches_prior_rejection(smoke: Configuration, seed: int, stop: int,
                                            run_open: bool) -> None:
    """An observed run of blocked cycles (open at t: the outage may still be active, so its
    remaining duration is conditioned on the elapsed cycles; closed: it has ended) gets the
    posterior that prior rejection gives: chi-squared on (start, duration) and on the cycles
    still to come, p > 0.01, fixed seeds (review concern 334)."""
    stats = pytest.importorskip("scipy.stats")
    gens = (ShockGenerator(ShockKind.PRODUCER_OUTAGE, 1, (3, 8), (1, 4), (1.0, 1.0),
                           ("p0_0", "p0_1")),)
    cfg = dataclasses.replace(smoke, shocks=ShockModel(generators=gens))
    env, rec, _ = _play(cfg, seed, stop, keep_replay=False)
    h = rec.history(env)
    (last, _), *_ = sorted(_blocked(h), reverse=True)
    assert _blocked(h) and (last == stop - 1) == run_open
    n = 300
    a = [p.events for p in sp.ShockPosterior(cfg, budget=10 * n, minimum=n).sample(
        h, np.random.default_rng(21)).particles]
    rng, b = np.random.default_rng(22), []
    for _ in range(100_000):
        events = _prior_draw(gens, cfg, rng)
        if sp.consistent(events, h):
            b.append(events)
            if len(b) == n:
                break
    assert len(a) == len(b) == n
    for f in (lambda e: (e.start_cycle, e.duration),
              lambda e: max(0, e.start_cycle + e.duration - stop)):  # cycles still to come
        x, y = ([f(e) for r in rows for e in r] for rows in (a, b))
        cats = sorted(set(x) | set(y))
        if cats == [0]:  # a closed run: no particle keeps the outage past t
            assert not run_open
            continue
        table = np.array([[x.count(c) for c in cats], [y.count(c) for c in cats]])
        assert stats.chi2_contingency(table).pvalue > 0.01, (cats, table)


def test_posterior_too_narrow(smoke: Configuration) -> None:
    """Scenario "Posterior too narrow": fewer than 16 accepted within the budget -> the cycle
    is unresolved, the action is the floor's and the acceptance rate is recorded."""
    env, rec, agent = _play(smoke, 0, 9)
    no_hits = dataclasses.replace(smoke, shocks=ShockModel(generators=(
        ShockGenerator(ShockKind.PRODUCER_OUTAGE, 1, (2, 8), (3, 6), (1.0, 1.0)),)))
    sentinel = np.full(env.action_space(agent).shape, 7.0, dtype=np.float32)
    for posterior in (sp.ShockPosterior(no_hits),  # observed hits the generator cannot make
                      sp.ShockPosterior(smoke, budget=8)):  # 8 draws < 16 accepted
        source = sp.PosteriorScenarioSource(rec, posterior, task_id="t", episode_seed=0)
        action = sp.with_floor(lambda e, a: sentinel, ROUND_ROBIN, source)(env, agent)
        assert np.array_equal(action, ROUND_ROBIN(env, agent))
        (row,) = source.log
        assert row.unresolved and row.accepted < sp.MIN_ACCEPTED
        assert row.rate == (row.accepted / row.draws if row.draws else 0.0)
        assert (row.draws == 0) == (posterior.budget == sp.DRAW_BUDGET)  # infeasible: no draw
        with pytest.raises(ValueError, match="unresolved"):
            source.fork(env, 0)
    resolved = sp.PosteriorScenarioSource(rec, sp.ShockPosterior(smoke), task_id="t", episode_seed=0)
    assert np.array_equal(sp.with_floor(lambda e, a: sentinel, ROUND_ROBIN, resolved)(env, agent),
                          sentinel)


# --- demand continuity ---


def test_drifting_world_forked(smoke: Configuration) -> None:
    """Scenario "Drifting world forked": the first forked cycle differs from the current
    cycle's orders in at most floor(j*N) orders, and so does every later transition."""
    env, rec, _ = _play(smoke, 0, 6)
    ids = [c.id for c in smoke.consumers if c.is_randomized]
    k = drift_budget(smoke.demand_drift.j, len(ids))
    assert k >= 1
    now = env.system.consumers[ids[0]]._cycle
    moved = False
    s = sp.posterior_fork(env, sp.Particle(()), task_id="t", decision_digest=3, continuation_id=0)
    for fork in (s, fork_future_demand(env, task_id="t", decision_digest=3, continuation_id=1)):
        rows = [fork.system.consumers[c]._schedule for c in ids]
        truth = [env.system.consumers[c]._schedule for c in ids]
        assert all(r[:now] == t[:now] for r, t in zip(rows, truth))
        for i in range(now, len(rows[0])):
            assert sum(r[i] != r[i - 1] for r in rows) <= k
        moved |= any(r[now:] != t[now:] for r, t in zip(rows, truth))
    assert moved


def _seasonal(smoke: Configuration) -> Configuration:
    return dataclasses.replace(smoke, demand_drift=None, demand_pattern=DemandPattern(
        "seasonal", amplitude=0.6, period=12, phase=3.0))


def test_seasonal_world_forked(smoke: Configuration) -> None:
    """Scenario "Seasonal world forked": every forked cycle follows the world's envelope."""
    cfg = _seasonal(smoke)
    env, rec, _ = _play(cfg, 0, 5)
    p = cfg.demand_pattern
    for cid in (0, 1, 2):
        fork = sp.posterior_fork(env, sp.Particle(()), task_id="t", decision_digest=2,
                                 continuation_id=cid)
        for c in (c for c in cfg.consumers if c.is_randomized):
            sched, now = fork.system.consumers[c.id]._schedule, fork.system.consumers[c.id]._cycle
            assert sched[:now] == env.system.consumers[c.id]._schedule[:now]
            for i in range(now, len(sched)):
                s = seasonal_factor(p, i + 1)
                lo = max(c.quantity_min, math.floor(c.quantity_min * s))
                hi = max(c.quantity_min, math.ceil(c.quantity_max * s))
                assert lo <= sched[i][1] <= hi and sched[i][0] in c.product_pool


def test_burst_world_forked(smoke: Configuration) -> None:
    cfg = dataclasses.replace(smoke, demand_drift=None, demand_pattern=DemandPattern(
        "burst", burst_p=0.5))
    env, rec, _ = _play(cfg, 0, 5)
    fork = sp.posterior_fork(env, sp.Particle(()), task_id="t", decision_digest=2, continuation_id=0)
    p = cfg.demand_pattern
    for c in (c for c in cfg.consumers if c.is_randomized):
        sched, now = fork.system.consumers[c.id]._schedule, fork.system.consumers[c.id]._cycle
        for q in (e[1] for e in sched[now:]):
            assert c.quantity_min <= q <= c.quantity_max * p.burst_factor_max
            assert q <= c.quantity_max or any(q == b * f for b in range(c.quantity_min, c.quantity_max + 1)
                                              for f in range(p.burst_factor_min, p.burst_factor_max + 1))


@pytest.mark.parametrize("pattern", [DemandPattern("hit", hit_alpha=2.0, hit_period=10),
                                     DemandPattern("switch", switch_count=2, switch_window_min=5,
                                                   switch_window_max=18, switch_min_gap=4,
                                                   low_weight=0.0)])
def test_hit_and_switch_worlds_refused(smoke: Configuration, pattern: DemandPattern) -> None:
    cfg = dataclasses.replace(smoke, demand_drift=None, demand_pattern=pattern)
    env = joint_stack_env(cfg, HORIZON)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="hidden pattern state"):
        fork_future_demand(env, task_id="t", decision_digest=1, continuation_id=0)
    with pytest.raises(ValueError, match="hidden pattern state"):
        sp.posterior_fork(env, sp.Particle(()), task_id="t", decision_digest=1, continuation_id=0)


def test_b9_and_calm_fork_unchanged(smoke: Configuration) -> None:
    """B9 without drift keeps the EXP-0040 i.i.d. tail (sorted ids, `draw_schedule_entry`, the
    continuation stream); fixed schedules are never resampled."""
    cfg = dataclasses.replace(smoke, demand_drift=None, shocks=None)
    env = joint_stack_env(cfg, HORIZON)
    env.reset(seed=3)
    fork = fork_future_demand(env, task_id="t", decision_digest=5, continuation_id=2)
    rng = np.random.default_rng(continuation_seed("t", 5, 2))
    by_id = {c.id: c for c in cfg.consumers if c.is_randomized}
    assert by_id
    for cid in sorted(by_id):
        c, s = by_id[cid], env.system.consumers[cid]
        want = s._schedule[:s._cycle] + [draw_schedule_entry(rng, c.product_pool, c.quantity_min,
                                                             c.quantity_max)
                                         for _ in range(len(s._schedule) - s._cycle)]
        assert fork.system.consumers[cid]._schedule == want
    calm, _ = _entry_config("dev-sparse_wide-12-calm-00")
    env = joint_stack_env(calm, HORIZON)
    env.reset(seed=1)
    fork = fork_future_demand(env, task_id="t", decision_digest=5, continuation_id=2)
    assert {c: s._schedule for c, s in fork.system.consumers.items() if hasattr(s, "_schedule")} == \
        {c: s._schedule for c, s in env.system.consumers.items() if hasattr(s, "_schedule")}
