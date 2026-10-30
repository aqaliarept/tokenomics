"""Agent-local history: each ordering agent's own past bid outcomes per producer (design D5 of
add-m4-orderer-redesign; ADR-0058).

Why: one shared portable orderer network serves every ordering agent (no agent ids), so agents
can differ only through what they themselves experienced. This module keeps, per (ordering
agent a, producer i), moving averages of a's own sub-orders' outcomes at i, beside the shared
producer-product telemetry (which is pooled over all agents and unchanged).

Information set (ADR-0058, ADR-0056):
- Input is the outcome *kind* of each `BidOutcome` of a's own sub-orders, its producer, its
  quantity and its routing cycle, nothing else. Kinds: `accepted` (the bid cleared the
  auction), `completed`, `failed` (the domain's dropped reservations and expired ones are one
  kind) and `rejected` (every kind of auction rejection is one kind). The domain's diagnostic
  cause fields are never read: two of their values encode hidden shock state.
- Ownership: an outcome belongs to the agent whose `incoming` held its order, read just before
  `complete_cycle` (every routed sub-order comes from the holder's turn; an order keeps its
  holder for its whole life).
- Folding happens only after `complete_cycle` returns, and the history is cleared at reset. What
  agent a sees in cycle t's ordering phase is therefore built from cycles < t only: nothing any
  ordering agent writes in cycle t, nor cycle t's production, reaches it.
- Routed units of a cycle = accepted + rejected units of that cycle: every bid placed in a cycle
  is either accepted or rejected in that cycle's auction (`ProducingAgent.produce`; task 1.3).

The 9 features of (a, i), evaluated as of cycle t = last folded cycle + 1 (`FEATURES` order):
0-4. EMAs of routed, accepted, completed, failed and rejected units per cycle. α is the domain
     telemetry's (`EconomyTelemetry`, read from the live system at reset). As in the telemetry's
     `_EmaSeries`, a pair starts folding in its first cycle with an outcome
     (EMA = α·units + (1 − α)·0) and then folds 0 in every cycle without one, so it decays.
5.   Completion latency: the quantity-weighted EMA of (completion cycle − routed cycle) over a's
     completed sub-orders at i only. Failed and rejected sub-orders never enter it. A cycle with
     no completion holds the value (no decay toward 0, which would read as faster delivery). The
     first completing cycle sets it to that cycle's weighted mean (no zero start). Before any
     completion it is `NO_LATENCY` = 0.0; the completed EMA and the rate tell that case apart.
6.   Completion rate with a 1-unit prior: (C + PRIOR_RATE · PRIOR_UNITS) / (R + PRIOR_UNITS) over
     the episode's raw totals since reset (C completed, R routed units), PRIOR_UNITS = 1,
     PRIOR_RATE = 0.5. Never routed → 0.5. Units routed in an earlier cycle that are still
     reserved count in R, not yet in C.
7.   Log recency: log(1 + t − r), r = the last cycle with routed units to i, r = 0 if never (as if
     routed at the episode start; cycles are numbered from 1), so it is always ≥ log 2 and grows
     while i is not routed.
8.   Never-routed flag: 1.0 if a has never routed to i since reset, else 0.0.

Opt-in from env only: `attach_agent_history(env)` installs callable hooks on `env.reset` and on
the system's `complete_cycle` (objects, not closures, so a deepcopy of the env, e.g. a teacher
fork, gets its own consistent history). Without it nothing changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..domain.common import OrderId, OrderingAgentId, ProducerId
from ..domain.system import CycleResult, System

FEATURES: tuple[str, ...] = (
    "routed_ema", "accepted_ema", "completed_ema", "failed_ema", "rejected_ema",
    "completion_latency", "completion_rate", "log_recency", "never_routed",
)
N_FEATURES = len(FEATURES)
KINDS: tuple[str, ...] = ("routed", "accepted", "completed", "failed", "rejected")
OUTCOME_KINDS = frozenset({"accepted", "completed", "failed", "rejected"})
PRIOR_UNITS = 1.0
PRIOR_RATE = 0.5
NO_LATENCY = 0.0

Pair = tuple[OrderingAgentId, ProducerId]


@dataclass
class _PairState:
    """One (agent, producer) pair: folded EMAs, raw totals since reset, latency, last routing."""

    ema: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in KINDS})
    total: dict[str, int] = field(default_factory=lambda: {k: 0 for k in KINDS})
    latency: float | None = None
    last_routed: int = 0


_EMPTY = _PairState()  # read-only stand-in for a pair without outcomes


class AgentHistory:
    """Per (ordering agent, producer) outcome history. `observe_owners` before and `fold` after
    each `complete_cycle`; `features` / `rows` read it. See the module docstring."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.cycle = 0  # the last folded cycle; features are as of cycle + 1
        self.unattributed = 0  # outcomes whose order had no recorded holder (expected 0)
        self._pairs: dict[Pair, _PairState] = {}
        self._owners: dict[OrderId, OrderingAgentId] = {}

    def reset(self, alpha: float | None = None) -> None:
        if alpha is not None:
            self.alpha = alpha
        self.cycle = 0
        self.unattributed = 0
        self._pairs = {}
        self._owners = {}

    # --- writing ---
    def observe_owners(self, system: System) -> None:
        """Record which ordering agent holds each open order. Call just before
        `complete_cycle`, which clears every agent's `incoming`."""
        for aid, agent in system.ordering_agents.items():
            for oid in agent.incoming:
                self._owners[oid] = aid

    def fold(self, cycle: int, result: CycleResult) -> None:
        """Fold played cycle `cycle`'s own-sub-order outcomes (kind, producer, quantity,
        routed cycle) into each agent's history."""
        if cycle <= self.cycle:
            raise ValueError(f"cycle {cycle} already folded (last {self.cycle})")
        units: dict[Pair, dict[str, int]] = {}
        lat: dict[Pair, list[float]] = {}  # [Σ latency · q, Σ q] over completions
        for o in result.bid_outcomes:
            kind = o.outcome
            if kind not in OUTCOME_KINDS:
                raise ValueError(f"unknown bid outcome kind {kind!r}")
            aid = self._owners.get(o.order_id)
            if aid is None:
                self.unattributed += 1
                continue
            key = (aid, o.producer_id)
            cell = units.setdefault(key, {k: 0 for k in KINDS})
            cell[kind] += o.quantity
            if kind in ("accepted", "rejected"):
                cell["routed"] += o.quantity
            if kind == "completed":
                acc = lat.setdefault(key, [0.0, 0.0])
                acc[0] += (cycle - o.routed_cycle) * o.quantity
                acc[1] += o.quantity
        a = self.alpha
        for key in sorted(set(self._pairs) | set(units)):
            st = self._pairs.setdefault(key, _PairState())
            cur = units.get(key)
            for k in KINDS:
                v = cur[k] if cur is not None else 0
                st.ema[k] = a * v + (1.0 - a) * st.ema[k]
                st.total[k] += v
            if cur is not None and cur["routed"] > 0:
                st.last_routed = cycle
            if key in lat and lat[key][1] > 0:
                value = lat[key][0] / lat[key][1]
                st.latency = value if st.latency is None else a * value + (1.0 - a) * st.latency
        for r in result.order_resolutions:
            self._owners.pop(r.order_id, None)
        self.cycle = cycle

    # --- reading ---
    def features(self, agent: OrderingAgentId, producer: ProducerId) -> tuple[float, ...]:
        """The 9 features of (agent, producer) as of cycle `self.cycle + 1` (FEATURES order)."""
        st = self._pairs.get((agent, producer)) or _EMPTY
        routed, done = st.total["routed"], st.total["completed"]
        rate = (done + PRIOR_RATE * PRIOR_UNITS) / (routed + PRIOR_UNITS)
        latency = NO_LATENCY if st.latency is None else st.latency
        recency = math.log1p(self.cycle + 1 - st.last_routed)  # last_routed 0 if never
        e = st.ema
        return (e["routed"], e["accepted"], e["completed"], e["failed"], e["rejected"],
                latency, rate, recency, 1.0 if routed == 0 else 0.0)

    def rows(self, agent: OrderingAgentId, producers: Sequence[ProducerId]) -> np.ndarray:
        """(len(producers), 9) float64 feature rows, in the given producer order."""
        out = np.zeros((len(producers), N_FEATURES), dtype=np.float64)
        for j, pid in enumerate(producers):
            out[j] = self.features(agent, pid)
        return out

    def totals(self, agent: OrderingAgentId, producer: ProducerId) -> dict[str, int]:
        """Raw units per kind since reset (routed = accepted + rejected)."""
        st = self._pairs.get((agent, producer))
        return dict(st.total) if st is not None else {k: 0 for k in KINDS}

    def pairs(self) -> list[Pair]:
        return sorted(self._pairs)

    def state(self) -> dict[str, object]:
        """Canonical plain-data state (pairs sorted), for byte-level comparison."""
        return {
            "alpha": self.alpha, "cycle": self.cycle, "unattributed": self.unattributed,
            "pairs": [[aid, pid, st.ema, st.total, st.latency, st.last_routed]
                      for (aid, pid), st in sorted(self._pairs.items())],
        }

    def snapshot(self) -> bytes:
        return json.dumps(self.state(), sort_keys=True, separators=(",", ":")).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.snapshot()).hexdigest()


def stream_producers(system: System, agent: OrderingAgentId,
                     producer_ids: Sequence[ProducerId]) -> list[ProducerId]:
    """The eligible producers of a product-stream agent's product (`produces`, ADR-0028), in
    `producer_ids` order (the spec's decode order)."""
    product = system.stream_products[agent]
    return [pid for pid in producer_ids if product in system.producing_agents[pid].produces]


# --- opt-in env hooks ---


class _FoldHook:
    """Replaces `system.complete_cycle`: records holders, plays the cycle, folds the result."""

    def __init__(self, system: System, history: AgentHistory,
                 inner: Callable[[], CycleResult]) -> None:
        self.system = system
        self.history = history
        self.inner = inner

    def __call__(self) -> CycleResult:
        cycle = self.system.cycle_counter  # the cycle complete_cycle is about to play
        self.history.observe_owners(self.system)
        result = self.inner()
        self.history.fold(cycle, result)
        return result


class _ResetHook:
    """Replaces `env.reset`: resets the env, then clears the history and hooks the new system."""

    def __init__(self, env: object, history: AgentHistory, inner: Callable[..., None]) -> None:
        self.env = env
        self.history = history
        self.inner = inner

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.inner(*args, **kwargs)
        system = self.env.system  # type: ignore[attr-defined]
        self.history.reset(alpha=system.telemetry._alpha)  # the domain telemetry's α
        system.complete_cycle = _FoldHook(system, self.history, system.complete_cycle)


def attach_agent_history(env: object) -> AgentHistory:
    """Give `env` an agent-local history (`env.agent_history`), reset with every `env.reset`.
    Attach before the first reset."""
    history = AgentHistory()
    env.agent_history = history  # type: ignore[attr-defined]
    env.reset = _ResetHook(env, history, env.reset)  # type: ignore[attr-defined]
    return history


def env_agent_id(env: object, agent: str) -> OrderingAgentId:
    """The domain ordering-agent id of an env ordering agent name (`order_<aid>`)."""
    role, aid = env._role[agent]  # type: ignore[attr-defined]
    if role != "order":
        raise ValueError(f"{agent} is not an ordering agent")
    return aid
