"""Teacher information set: observable history, shock posterior and posterior forks (ADR-0056;
add-m3-development-audit design D4; development-audit "Teacher labels depend only on observable
information", "Unresolved labels fall back").

A teacher that rolls out forks of the true env reads the hidden future: the remaining duration
of an active outage and every future hit (EXP-0038B's cause). This module replaces every hidden
part of a fork's future with a posterior draw given the public history and the world's known
shock generators. It lives in the env layer because it draws random numbers.

Public sources (`HistoryRecorder`). The recorder wraps the policies and reads, at each agent
turn, only public information: the agents' observations and actions, the producers' own
capacities (encoded in their observations), the orderer telemetry (routed / accepted /
completed / rejected / failed EMAs per producer and product) and the realised state the forks
keep anyway. It never reads the env's shock schedule, the config's shock model, the domain's
shock indexes, the episode stats' shock digest, or the diagnostic rejection / failure reasons
(task 1.1: `outage` there is computed from hidden state). It works on a *stripped* copy of the
system (`stripped_system`), whose shock fields are replaced while copying and never read. Per
played cycle it derives two consequences:
- capacity hits: the drop between a shock-free one-cycle replay of the stripped copy and the
  producers' next observation, per producer and capacity type, with the pre-hit capacity;
- outage status per producer: each producer's auction is run twice on copies, with and without
  an outage, and the result whose accepted / completed / rejected EMA fold matches the public
  telemetry is the explanation: `normal`, `blocked`, or `free` when both match (a producer that
  rejected every bid and assembled nothing: an outage or simply full, D4 "Ambiguous").
Each finalized cycle is checked: the one-cycle replay must reproduce every telemetry EMA.

Posterior (`ShockPosterior`). The `shocks-i` generators draw a fixed number of i.i.d. events per
kind, so the posterior factorises (D4). A particle is built constructively: each observed hit is
assigned one hit slot (start, target and type fixed; magnitude from the generator's rounded
uniform conditioned on floor(m*c) = observed drop); each run of blocked cycles is assigned one
outage slot ((start, duration) uniform over the pairs that cover the run without touching a
normal cycle; a run reaching the last played cycle is open, so the remaining duration is drawn
conditioned on the elapsed cycles); the remaining slots come from the generator truncated to
"not observable by t" (a hit after t or one that removes nothing; an outage whose played
cycles are all non-normal), and the free cycles are the generator-weighted mixture of D4 step 2.
Every particle is validated two-sidedly by exact consequence predicates (`consistent`): it must
reproduce every observed drop and outage status and imply none that was not observed. The
replay from reset (`replay_reproduces`) is the reference the tests compare them against. The
proposal is consistent by construction, so validation is a guard: an acceptance rate below 1
means a proposal bug, and "unresolved" comes from an infeasible plan (draws = 0, rate 0), e.g. an
observed hit outside the generator's window or targets or with no fitting magnitude, more
observed hits or blocked runs than the generator's count, a blocked run no single
(start, duration) covers, or no slot left for an unobserved event (ADR-0056 "Floor").

Budget (pre-registered, D4): at most 1,024 draws per decision cycle, stopping at 16 accepted
(the EXP-0042 maximum scenario count); fewer than 16 = "unresolved": the teacher acts as its
floor policy (`with_floor`). The acceptance rate is logged per cycle. Sampling runs once per
decision cycle and is cached for every orderer of that cycle (`PosteriorScenarioSource`).

Forks (`posterior_fork`). A fork is a stripped deep copy of the env: the true pending events and
outage end cycles never enter it (spec "Hidden fields stripped"); the particle's future events
are injected, and future demand is redrawn by `future_fork.resample_future_demand` (drift and
demand-pattern continuity). Explicit config events are outside ADR-0056 and refused.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..domain.common import CapacityType, ProducerId
from ..domain.config import Configuration
from ..domain.consumer import ScheduledConsumer
from ..domain.producing_agent import ProductionResult
from ..domain.shocks import ShockEvent, ShockGenerator, ShockKind, ShockTarget
from ..domain.system import System, capacity_hit_removal
from .aec_env import TokenomicsAECEnv
from .decode import decode_ordering, decode_producing
from .demand_digest import demand_digest
from .future_fork import continuation_seed, resample_future_demand
from .shock_schedule import MAGNITUDE_DECIMALS, shock_digest

Policy = Callable[[TokenomicsAECEnv, str], np.ndarray]

EXPERIMENT = "ADR-0056"
DRAW_BUDGET = 1024
MIN_ACCEPTED = 16
NORMAL, BLOCKED, FREE = "normal", "blocked", "free"

# Hidden shock state: replaced while copying, never read (the shock_reactive sentinel names).
SYSTEM_SHOCK_FIELDS = ("_shock_schedule", "_hits_by_cycle", "_outages_by_cycle")
STATS_SHOCK_KEYS = ("shock_digest", "shock_events")
# The telemetry series an auction writes (production results, never the reasons).
AUCTION_SERIES = ("_accepted", "_completed", "_rejected")
ALL_SERIES = ("_routed", "_accepted", "_completed", "_rejected", "_failed")

Cell = tuple[ProducerId, CapacityType]


class HistoryInconsistent(RuntimeError):
    """The one-cycle replay of a played cycle does not reproduce the public telemetry: a
    mechanic the extractor does not model (never expected; raised rather than guessed)."""


# --- stripping (never reads a hidden field: the copy's memo maps it away by identity) ---


def _strip_memo(system: System) -> dict[int, object]:
    """deepcopy memo entries that map each hidden field's object (by identity, never read) to
    `()`. Harmless when the object is the shared empty-tuple singleton; the indexes are
    rebuilt by `_set_schedule` after the copy. Never used for ints or strings, which are
    shared singletons a memo entry would replace everywhere."""
    return {id(getattr(system, n)): () for n in SYSTEM_SHOCK_FIELDS}


def _set_schedule(system: System, events: Sequence[ShockEvent], from_cycle: int) -> None:
    """Install `events` as the system's schedule, indexing only hits starting at
    `from_cycle` or later and outage cycles from `from_cycle` on (System.__init__'s indexes)."""
    system._shock_schedule = tuple(events)
    hits: dict[int, list[ShockEvent]] = {}
    outages: dict[int, set[ProducerId]] = {}
    for e in events:
        if e.kind is ShockKind.CAPACITY_HIT:
            if e.start_cycle >= from_cycle:
                hits.setdefault(e.start_cycle, []).append(e)
        elif e.kind is ShockKind.PRODUCER_OUTAGE:
            for c in range(max(e.start_cycle, from_cycle), e.start_cycle + e.duration):
                outages.setdefault(c, set()).add(e.target.producer)
    system._hits_by_cycle = hits
    system._outages_by_cycle = outages


def stripped_system(system: System) -> System:
    """A deep copy of the domain system with no shock schedule and no shock indexes."""
    clone = copy.deepcopy(system, _strip_memo(system))
    _set_schedule(clone, (), 0)
    return clone


def stripped_env(env: TokenomicsAECEnv) -> TokenomicsAECEnv:
    """A deep copy of the env whose drawn schedule, config shock model, domain shock state and
    stats shock digest are all replaced while copying (never read): the base of every fork."""
    stats = env.episode_stats
    memo = _strip_memo(env.system)
    memo[id(env._shock_schedule)] = ()
    memo[id(env.config)] = env.config  # frozen and known; shared, never copied or read
    memo[id(stats)] = {}
    clone = copy.deepcopy(env, memo)
    clone._shock_schedule = ()
    _set_schedule(clone.system, (), 0)
    clone.episode_stats = {k: copy.deepcopy(v) for k, v in stats.items()
                           if k not in STATS_SHOCK_KEYS}
    clone.episode_stats["shock_digest"] = shock_digest(())
    clone.episode_stats["shock_events"] = 0
    return clone


def sort_events(events: Iterable[ShockEvent]) -> tuple[ShockEvent, ...]:
    """The schedule order `build_schedule` uses (the order the domain applies hits in)."""
    return tuple(sorted(events, key=lambda e: (
        e.start_cycle, e.kind.value, e.target.producer or "", e.target.capacity_types,
        e.duration, e.magnitude)))


# --- the observable history ---


@dataclass
class CycleRecord:
    """One decision cycle's public consequences. `pre_hit` is every producer's capacity per
    type before this cycle's hits (end of the previous cycle; at cycle 1 the reset state),
    `removal` the observed drop (> 0 only); `status` the outage explanation of this cycle's
    production per producer, filled once the cycle is played."""

    cycle: int
    pre_hit: dict[Cell, int]
    removal: dict[Cell, int]
    status: dict[ProducerId, str] | None = None


def _capacities(system: System) -> dict[Cell, int]:
    return {(pid, t): c for pid, p in system.producing_agents.items()
            for t, c in p.capacities.items()}


def _auction_sums(pid: ProducerId, result: ProductionResult) -> dict[str, dict[object, float]]:
    """Per series, per (producer, product), the units an auction result records (the
    reason fields are never read)."""
    out: dict[str, dict[object, float]] = {s: {} for s in AUCTION_SERIES}
    for series, bids in (("_accepted", result.accepted), ("_completed", result.completed),
                         ("_rejected", result.rejected)):
        acc = out[series]
        for bid in bids:
            key = (pid, bid.product)
            acc[key] = acc.get(key, 0.0) + bid.quantity
    return out


def _fold_matches(before: System, after: System, pid: ProducerId,
                  sums: Mapping[str, Mapping[object, float]]) -> bool:
    """Would folding `sums` into `before`'s telemetry give `after`'s EMAs on `pid`'s keys?
    Same formula and operand order as `_EmaSeries.fold`, so equality is exact."""
    for series in AUCTION_SERIES:
        b, a = getattr(before.telemetry, series), getattr(after.telemetry, series)
        keys = {k for k in b._ema if k[0] == pid} | {k for k in a._ema if k[0] == pid} \
            | set(sums[series])
        for key in keys:
            predicted = b._alpha * sums[series].get(key, 0.0) + (1.0 - b._alpha) * b._ema.get(key, 0.0)
            if predicted != a._ema.get(key, 0.0):
                return False
    return True


def _digest_obs(obs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()


def _telemetry_digest(system: System) -> str:
    parts = []
    for series in ALL_SERIES:
        ema = getattr(system.telemetry, series)._ema
        parts.append(repr(sorted((repr(k), v) for k, v in ema.items())))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


@dataclass
class _Turn:
    cycle: int
    agent: str
    obs: str
    telemetry: str
    action: np.ndarray | None = None


class HistoryRecorder:
    """Records the public history of one episode by wrapping its policies (`wrap`). Every
    agent's policy must be wrapped from reset on, so the recorder sees every cycle and the last
    agent of each cycle (module docstring). `keep_replay` also keeps what the replay reference
    needs (a stripped reset snapshot, every action, observation and telemetry digest)."""

    def __init__(self, *, keep_replay: bool = False) -> None:
        self.keep_replay = keep_replay
        self.cycles: list[CycleRecord] = []
        self.turns: list[_Turn] = []
        self._system: System | None = None
        self._pending: System | None = None
        self._snapshot: TokenomicsAECEnv | None = None

    @property
    def cycle(self) -> int:
        return self.cycles[-1].cycle if self.cycles else 0

    def wrap(self, policy: Policy) -> Policy:
        def recorded(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
            self.observe(env, agent)
            action = policy(env, agent)
            self.record_action(env, agent, action)
            return action
        return recorded

    def observe(self, env: TokenomicsAECEnv, agent: str) -> None:
        """Call at an agent's turn before its policy runs (idempotent within a turn)."""
        system = env.system
        cycle = system.cycle_counter
        if self._system is not system:  # a new episode
            if cycle != 1 or env.current_cycle != 0:
                raise ValueError("the recorder must observe an episode from reset")
            self._system = system
            self.cycles, self.turns, self._pending = [], [], None
            self.cycles.append(CycleRecord(1, _capacities(system), {}))
            if self.keep_replay:
                self._snapshot = stripped_env(env)
        elif cycle != self.cycle:
            if cycle != self.cycle + 1 or self._pending is None:
                raise ValueError(f"the recorder missed a cycle ({self.cycle} -> {cycle})")
            self._finalize(system)
        if self.keep_replay and not (self.turns and self.turns[-1].agent == agent
                                     and self.turns[-1].cycle == cycle
                                     and self.turns[-1].action is None):
            self.turns.append(_Turn(cycle, agent, _digest_obs(env.observe(agent)),
                                    _telemetry_digest(system)))

    def record_action(self, env: TokenomicsAECEnv, agent: str, action: np.ndarray) -> None:
        if self.keep_replay:
            self.turns[-1].action = np.array(action, copy=True)
        if agent == env.possible_agents[-1]:
            self._pending = self._apply(stripped_system(env.system), env, agent, action)

    @staticmethod
    def _apply(system: System, env: TokenomicsAECEnv, agent: str, action: np.ndarray) -> System:
        """The last agent's action on the stripped copy, exactly as `TokenomicsAECEnv.step`."""
        role, did = env._role[agent]
        if role == "prod":
            view = system.producing_agents[did].view()
            system.producing_agent_action(did, decode_producing(env.spec_, np.asarray(action), view))
            system.start_ordering()
        else:
            view = system.ordering_agents[did].view(system.producer_infos)
            system.ordering_agent_action(did, decode_ordering(env.spec_, np.asarray(action), view))
        return system

    def _finalize(self, live: System) -> None:
        """Explain the played cycle from its stripped pre-production copy (module docstring)."""
        pre = self._pending
        assert pre is not None
        status: dict[ProducerId, str] = {}
        for pid, p in pre.producing_agents.items():
            normal = _fold_matches(pre, live, pid, _auction_sums(pid, copy.deepcopy(p).produce(False)))
            blocked = _fold_matches(pre, live, pid, _auction_sums(pid, copy.deepcopy(p).produce(True)))
            if not (normal or blocked):
                raise HistoryInconsistent(f"cycle {pre.cycle_counter}: producer {pid}'s auction "
                                          "matches neither explanation")
            status[pid] = FREE if normal and blocked else NORMAL if normal else BLOCKED
        played = pre.cycle_counter
        pre._outages_by_cycle = {played: {pid for pid, s in status.items() if s == BLOCKED}}
        pre.complete_cycle()  # shock-free hits: `pre` ends at the next cycle's pre-hit state
        if _telemetry_digest(pre) != _telemetry_digest(live):
            raise HistoryInconsistent(f"cycle {played}: the one-cycle replay does not reproduce "
                                      "the public telemetry")
        self.cycles[-1].status = status
        before, after = _capacities(pre), _capacities(live)
        removal = {cell: before[cell] - after[cell] for cell in before if before[cell] != after[cell]}
        if any(r < 0 for r in removal.values()) or set(before) != set(after):
            raise HistoryInconsistent(f"cycle {played + 1}: capacity rose without an explanation")
        self.cycles.append(CycleRecord(played + 1, before, removal))
        self._pending = None

    def history(self, env: TokenomicsAECEnv) -> ObservedHistory:
        """The history at the current decision cycle of `env` (which this recorder follows)."""
        if self._system is not env.system or self.cycle != env.system.cycle_counter:
            raise ValueError("the recorder is not synchronised with this env's decision cycle")
        return ObservedHistory(self.cycle, tuple(self.cycles))


@dataclass(frozen=True)
class ObservedHistory:
    """Cycles 1..t: pre-hit capacities and observed drops for each, outage statuses for the
    played cycles 1..t-1."""

    cycle: int
    records: tuple[CycleRecord, ...]

    def status(self, cycle: int, pid: ProducerId) -> str:
        s = self.records[cycle - 1].status
        assert s is not None, cycle
        return s[pid]


# --- magnitude prior: m = clip(round(U(lo, hi), 4), lo, hi), as shock_schedule._magnitude ---


@dataclass(frozen=True)
class _MagnitudeGrid:
    values: tuple[float, ...]
    masses: tuple[float, ...]

    @staticmethod
    def build(lo: float, hi: float) -> _MagnitudeGrid:
        if hi <= lo:
            return _MagnitudeGrid((lo,), (1.0,))
        scale, half = 10 ** MAGNITUDE_DECIMALS, 0.5 * 10 ** -MAGNITUDE_DECIMALS
        mass: dict[float, float] = {}
        for g in range(math.floor(lo * scale - 0.5), math.ceil(hi * scale + 0.5) + 1):
            v = g / scale
            width = min(v + half, hi) - max(v - half, lo)
            if width > 0:
                key = min(max(v, lo), hi)
                mass[key] = mass.get(key, 0.0) + width
        values = tuple(sorted(mass))
        return _MagnitudeGrid(values, tuple(mass[v] for v in values))

    def removing(self, capacity: int, removal: int) -> tuple[tuple[float, ...], np.ndarray]:
        """The grid values m with floor(m * capacity) == removal, and their masses. The removal
        is nondecreasing in m, so the set is a contiguous run found by bisection."""
        def first(at_least: int) -> int:
            lo, hi = 0, len(self.values)
            while lo < hi:
                mid = (lo + hi) // 2
                if capacity_hit_removal(capacity, self.values[mid]) >= at_least:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        a, b = first(removal), first(removal + 1)
        return self.values[a:b], np.asarray(self.masses[a:b], dtype=np.float64)


def _pick(rng: np.random.Generator, weights: np.ndarray) -> int:
    """One index with probability proportional to `weights` (a single uniform draw)."""
    cum = np.cumsum(weights)
    return min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), len(cum) - 1)


# --- the posterior ---


@dataclass(frozen=True)
class Particle:
    """One posterior draw of the whole generated schedule, in schedule order."""

    events: tuple[ShockEvent, ...]

    def future(self, cycle: int) -> tuple[ShockEvent, ...]:
        """Events that still act at decision cycle `cycle`: hits after it, outages active in
        or after it (hits at `cycle` are already applied in the realised state)."""
        return tuple(e for e in self.events
                     if (e.kind is ShockKind.CAPACITY_HIT and e.start_cycle > cycle)
                     or (e.kind is ShockKind.PRODUCER_OUTAGE
                         and e.start_cycle + e.duration - 1 >= cycle))


@dataclass(frozen=True)
class PosteriorSample:
    cycle: int
    particles: tuple[Particle, ...]
    draws: int
    accepted: int
    unresolved: bool

    @property
    def rate(self) -> float:
        """accepted / draws; 0.0 for an infeasible plan (no proposal exists, draws = 0)."""
        return self.accepted / self.draws if self.draws else 0.0


@dataclass
class _Plan:
    """Per-decision-cycle precomputation shared by every draw (D4 steps 1-3)."""

    feasible: bool
    observed_hits: list[tuple[int, ProducerId, CapacityType, tuple[float, ...], np.ndarray]] = \
        field(default_factory=list)
    trunc_hits: list[tuple[int, ProducerId, CapacityType, tuple[float, ...], np.ndarray]] = \
        field(default_factory=list)
    trunc_hit_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    runs: list[tuple[ProducerId, list[tuple[int, int]]]] = field(default_factory=list)
    trunc_outages: list[tuple[ProducerId, int, int]] = field(default_factory=list)


class ShockPosterior:
    """Posterior over a world's generated shocks given an `ObservedHistory` (module docstring).
    Built from the world config, i.e. the known generators; never from an env."""

    def __init__(self, config: Configuration, *, budget: int = DRAW_BUDGET,
                 minimum: int = MIN_ACCEPTED) -> None:
        model = config.shocks
        if model is not None and model.events:
            raise ValueError("explicit shock events are outside ADR-0056's posterior")
        self.generators: dict[ShockKind, ShockGenerator] = {
            g.kind: g for g in (model.generators if model is not None else ())}
        self.owned: dict[ProducerId, tuple[CapacityType, ...]] = {
            p.id: tuple(sorted(p.capacities)) for p in config.producer_configs}
        self.producers = tuple(p.id for p in config.producer_configs)
        self.budget, self.minimum = int(budget), int(minimum)
        hit = self.generators.get(ShockKind.CAPACITY_HIT)
        self._grid = _MagnitudeGrid.build(*hit.magnitude) if hit is not None else None
        self._cond: dict[tuple[int, int], tuple[tuple[float, ...], np.ndarray]] = {}

    def _targets(self, g: ShockGenerator) -> tuple[ProducerId, ...]:
        return g.targets or self.producers

    def _removing(self, capacity: int, removal: int) -> tuple[tuple[float, ...], np.ndarray]:
        key = (capacity, removal)
        if key not in self._cond:
            assert self._grid is not None
            self._cond[key] = self._grid.removing(capacity, removal)
        return self._cond[key]

    def _plan(self, h: ObservedHistory) -> _Plan:
        t = h.cycle
        plan = _Plan(feasible=True)
        hit = self.generators.get(ShockKind.CAPACITY_HIT)
        observed = [(r.cycle, pid, k, removal) for r in h.records
                    for (pid, k), removal in sorted(r.removal.items())]
        if observed and hit is None:
            return _Plan(feasible=False)
        if hit is not None:
            s_lo, s_hi = hit.start_window
            targets = self._targets(hit)
            for c, pid, k, removal in observed:
                vs, ws = self._removing(h.records[c - 1].pre_hit[(pid, k)], removal)
                if not (s_lo <= c <= s_hi) or pid not in targets or not vs:
                    return _Plan(feasible=False)
                plan.observed_hits.append((c, pid, k, vs, ws))
            if len(observed) > hit.count:
                return _Plan(feasible=False)
            seen = {(c, pid, k) for c, pid, k, _ in observed}
            weights: list[float] = []
            assert self._grid is not None
            prior = (self._grid.values, np.asarray(self._grid.masses, dtype=np.float64))
            total = float(prior[1].sum())
            for s in range(s_lo, s_hi + 1):
                for pid in targets:
                    types = self.owned[pid]
                    for k in types:
                        if s > t:
                            vs, ws = prior
                        elif (s, pid, k) in seen:
                            continue
                        else:  # a hit already applied that removed nothing
                            vs, ws = self._removing(h.records[s - 1].pre_hit[(pid, k)], 0)
                            if not vs:
                                continue
                        plan.trunc_hits.append((s, pid, k, vs, ws))
                        weights.append(float(ws.sum()) / total / len(types))
            plan.trunc_hit_weights = np.asarray(weights, dtype=np.float64)
            if hit.count > len(observed) and not weights:
                return _Plan(feasible=False)
        out = self.generators.get(ShockKind.PRODUCER_OUTAGE)
        blocked_any = any(r.status and BLOCKED in r.status.values() for r in h.records)
        if out is None:
            return plan if not blocked_any else _Plan(feasible=False)
        (s_lo, s_hi), (d_lo, d_hi) = out.start_window, out.duration
        targets = self._targets(out)

        def normal(c: int, pid: ProducerId) -> bool:
            return c <= t - 1 and h.status(c, pid) == NORMAL

        for pid in self.producers:
            c = 1
            while c <= t - 1:  # maximal runs of non-normal played cycles holding a blocked one
                if normal(c, pid):
                    c += 1
                    continue
                lo = c
                while c <= t - 1 and not normal(c, pid):
                    c += 1
                hi = c - 1  # the run is open when it reaches the last played cycle
                blocked = [x for x in range(lo, hi + 1) if h.status(x, pid) == BLOCKED]
                if not blocked:
                    continue
                b1, b2 = blocked[0], blocked[-1]
                pairs = [(s, d) for d in range(d_lo, d_hi + 1)
                         for s in range(max(s_lo, lo, b2 - d + 1), min(s_hi, b1) + 1)
                         if (hi == t - 1 or s + d - 1 <= hi)]
                if pid not in targets or not pairs:
                    return _Plan(feasible=False)
                plan.runs.append((pid, pairs))
        if len(plan.runs) > out.count:
            return _Plan(feasible=False)
        for pid in targets:
            for s in range(s_lo, s_hi + 1):
                for d in range(d_lo, d_hi + 1):
                    if not any(normal(c, pid) for c in range(s, min(s + d - 1, t - 1) + 1)):
                        plan.trunc_outages.append((pid, s, d))
        if out.count > len(plan.runs) and not plan.trunc_outages:
            return _Plan(feasible=False)
        return plan

    def _draw(self, plan: _Plan, rng: np.random.Generator) -> Particle:
        events: list[ShockEvent] = []
        hit = self.generators.get(ShockKind.CAPACITY_HIT)
        if hit is not None:
            def hit_event(s: int, pid: ProducerId, k: CapacityType, vs: tuple[float, ...],
                          ws: np.ndarray) -> ShockEvent:
                return ShockEvent(ShockKind.CAPACITY_HIT, ShockTarget(pid, (k,)), s, 1,
                                  vs[_pick(rng, ws)])
            for cell in plan.observed_hits:
                events.append(hit_event(*cell))
            for _ in range(hit.count - len(plan.observed_hits)):
                events.append(hit_event(*plan.trunc_hits[_pick(rng, plan.trunc_hit_weights)]))
        out = self.generators.get(ShockKind.PRODUCER_OUTAGE)
        if out is not None:
            for pid, pairs in plan.runs:
                s, d = pairs[int(rng.integers(len(pairs)))]
                events.append(ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(pid), s, d, 1.0))
            for _ in range(out.count - len(plan.runs)):
                pid, s, d = plan.trunc_outages[int(rng.integers(len(plan.trunc_outages)))]
                events.append(ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(pid), s, d, 1.0))
        return Particle(sort_events(events))

    def sample(self, h: ObservedHistory, rng: np.random.Generator) -> PosteriorSample:
        """Draw until `minimum` particles are accepted or the budget is spent."""
        plan = self._plan(h)
        if not plan.feasible:
            return PosteriorSample(h.cycle, (), 0, 0, True)  # nothing to draw from
        accepted: list[Particle] = []
        draws = 0
        while draws < self.budget and len(accepted) < self.minimum:
            draws += 1
            particle = self._draw(plan, rng)
            if consistent(particle.events, h):
                accepted.append(particle)
        return PosteriorSample(h.cycle, tuple(accepted), draws, len(accepted),
                               len(accepted) < self.minimum)


def consistent(events: Sequence[ShockEvent], h: ObservedHistory) -> bool:
    """Two-sided consequence predicates (D4 "Validation"): replayed with the recorded actions,
    the schedule reproduces every observed drop and outage status through cycle t and implies
    none that was not observed. Exact: the domain reaches the schedule only through each
    cycle's hit removals and each played auction's outage flag, and a `free` flag does not
    change the auction."""
    t = h.cycle
    hits_at: dict[int, list[ShockEvent]] = {}
    covered: set[tuple[int, ProducerId]] = set()
    for e in sort_events(events):
        if e.kind is ShockKind.CAPACITY_HIT and e.start_cycle <= t:
            hits_at.setdefault(e.start_cycle, []).append(e)
        elif e.kind is ShockKind.PRODUCER_OUTAGE:
            for c in range(e.start_cycle, min(e.start_cycle + e.duration - 1, t - 1) + 1):
                covered.add((c, e.target.producer))
    for r in h.records:
        cap = {}
        for e in hits_at.get(r.cycle, ()):
            for k in e.target.capacity_types:
                cell = (e.target.producer, k)
                now = cap.get(cell, r.pre_hit[cell])
                cap[cell] = now - capacity_hit_removal(now, e.magnitude)
        drops = {cell: r.pre_hit[cell] - c for cell, c in cap.items() if r.pre_hit[cell] != c}
        if drops != r.removal:
            return False
        if r.cycle <= t - 1:
            assert r.status is not None
            for pid, s in r.status.items():
                if s != FREE and (s == BLOCKED) != ((r.cycle, pid) in covered):
                    return False
    return True


def replay_reproduces(recorder: HistoryRecorder, events: Sequence[ShockEvent]) -> bool:
    """The reference (D4): replay from reset with `events` injected and the recorded actions,
    comparing every recorded observation and telemetry digest. Needs `keep_replay=True`."""
    if recorder._snapshot is None:
        raise ValueError("replay_reproduces needs a recorder with keep_replay=True")
    env = copy.deepcopy(recorder._snapshot)
    _set_schedule(env.system, sort_events(events), 0)
    for turn in recorder.turns:
        if env.agent_selection != turn.agent:
            return False
        if (_digest_obs(env.observe(turn.agent)) != turn.obs
                or _telemetry_digest(env.system) != turn.telemetry):
            return False
        if turn.action is None:
            break
        env.step(turn.action)
    return True


# --- posterior forks and the per-cycle source ---


def posterior_seed(task_id: str, decision_digest: int, *, experiment: str = EXPERIMENT) -> int:
    """The sampler's seed for one decision cycle (never hidden state)."""
    return continuation_seed(task_id, decision_digest, -1, experiment=f"{experiment}|posterior")


def posterior_fork(env: TokenomicsAECEnv, particle: Particle, *, task_id: str,
                   decision_digest: int, continuation_id: int,
                   experiment: str = EXPERIMENT) -> TokenomicsAECEnv:
    """A fork for one rollout: the stripped env, the particle's future events injected, and
    future demand redrawn on the continuation's stream (spec "Hidden fields stripped")."""
    fork = stripped_env(env)
    future = particle.future(env.system.cycle_counter)
    _set_schedule(fork.system, future, env.system.cycle_counter)
    fork._shock_schedule = future
    fork.episode_stats["shock_digest"] = shock_digest(future)
    fork.episode_stats["shock_events"] = len(future)
    rng = np.random.default_rng(continuation_seed(
        task_id, decision_digest, continuation_id, experiment=experiment))
    resample_future_demand(fork, rng)
    # The reset-time demand record holds the true future: replace it by the fork's own.
    fork._demand_schedules = {
        cid: tuple(c._schedule) if isinstance(c, ScheduledConsumer) else fork._demand_schedules[cid]
        for cid, c in fork.system.consumers.items()}
    fork.episode_stats["demand_digest"] = demand_digest(fork._demand_schedules)
    return fork


@dataclass(frozen=True)
class CycleLog:
    cycle: int
    draws: int
    accepted: int
    rate: float
    unresolved: bool


class PosteriorScenarioSource:
    """One posterior sample per decision cycle, cached for every orderer of that cycle, and
    the rollout forks built from it. `log` holds one row per sampled cycle (acceptance rate)."""

    def __init__(self, recorder: HistoryRecorder, posterior: ShockPosterior, *, task_id: str,
                 episode_seed: int, experiment: str = EXPERIMENT) -> None:
        self.recorder, self.posterior = recorder, posterior
        self.task_id, self.episode_seed, self.experiment = task_id, int(episode_seed), experiment
        self.log: list[CycleLog] = []
        self._cache: tuple[System, int, PosteriorSample] | None = None

    def decision_digest(self, cycle: int) -> int:
        blob = f"{self.experiment}|{self.task_id}|{self.episode_seed}|cycle:{cycle}".encode()
        return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")

    def sample(self, env: TokenomicsAECEnv) -> PosteriorSample:
        cycle = env.system.cycle_counter
        if self._cache is not None and self._cache[0] is env.system and self._cache[1] == cycle:
            return self._cache[2]
        h = self.recorder.history(env)
        rng = np.random.default_rng(posterior_seed(
            self.task_id, self.decision_digest(cycle), experiment=self.experiment))
        s = self.posterior.sample(h, rng)
        self._cache = (env.system, cycle, s)
        self.log.append(CycleLog(cycle, s.draws, s.accepted, s.rate, s.unresolved))
        return s

    def fork(self, env: TokenomicsAECEnv, scenario: int) -> TokenomicsAECEnv:
        s = self.sample(env)
        if s.unresolved:
            raise ValueError(f"cycle {s.cycle} is unresolved: the teacher acts as its floor")
        cycle = env.system.cycle_counter
        return posterior_fork(env, s.particles[scenario % len(s.particles)],
                              task_id=self.task_id, decision_digest=self.decision_digest(cycle),
                              continuation_id=scenario, experiment=self.experiment)


def with_floor(teacher: Policy, floor: Policy, source: PosteriorScenarioSource) -> Policy:
    """The teacher where the cycle resolves, its floor policy where it is unresolved (spec
    "Posterior too narrow")."""
    def policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        if source.sample(env).unresolved:
            return floor(env, agent)
        return teacher(env, agent)
    return policy
