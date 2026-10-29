"""EXP-0040 counterfactual future-demand fork (evaluator-only intervention).

EXP-0040 audits whether the single-continuation marginal-value teacher label is a
stable function of the *deployable pre-action state* once future exogenous demand is
marginalized out. The domain is deterministic after `env.reset(seed=...)`, so re-running
the same clone reproduces the same labels — that is NOT a stability test. To probe
future sensitivity we must hold the exact pre-action state fixed while replacing only
the *unconsumed future demand suffix*.

`fork_future_demand` returns an independent deep clone of the env in which every
randomized `ScheduledConsumer` keeps every already-consumed schedule entry and has only
its unconsumed tail resampled from the same legal pool (`draw_schedule_entry`, the shared
legality authority). The current system state, open orders, current-cycle demand,
balances, telemetry, auctions, reservations, agent selector, and episode stats are all
preserved by the deep copy; fixed `ScriptedConsumer` demand and non-randomized consumers
are left untouched.

Task 6.1 (ADR-0056, add-m3-development-audit D4) extends the tail redraw to the teacher's
fork rule: a drifting world continues the drift process from the current cycle's orders, a
burst or seasonal world resamples its future from the pattern generator, and a moving-hit or
regime-switch world is refused. B9 without drift and fixed (cyclic) schedules are unchanged.

This is an evaluator-only diagnostic. The forked future is NEVER a deployable observation
feature (HANDOVER "Frozen Scope"): it exists solely to measure the target. The RNG seed is
derived from SHA-256 of `(experiment, task_id, decision_digest, continuation_id)` — never
Python's process-randomized `hash()` — so a fixed `(decision_digest, continuation_id)` is
bitwise deterministic and different continuation IDs give different legal suffixes.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np

from ..domain.config import (PATTERN_BURST, PATTERN_HIT, PATTERN_SWITCH, ConsumerConfig,
                             DemandPattern)
from ..domain.consumer import ScheduledConsumer
from .aec_env import TokenomicsAECEnv, draw_schedule_entry
from .demand_drift import drift_budget, redraw
from .demand_patterns import burst_entry, seasonal_entry

_TAG = "EXP-0040"
# Patterns whose future depends on hidden past draws (ADR-0056): never forked, never taught.
REFUSED_PATTERNS = frozenset({PATTERN_HIT, PATTERN_SWITCH})


def continuation_seed(
    task_id: str,
    decision_digest: int,
    continuation_id: int,
    *,
    experiment: str = _TAG,
) -> int:
    """Deterministic RNG seed for one continuation = SHA-256 of the stringified identity
    tuple `(experiment, task_id, decision_digest, continuation_id)`. The default remains
    `EXP-0040`, preserving that experiment's byte-for-byte seed contract. Stringified so the
    write/read sides agree regardless of int width; the full 256-bit digest is fed to
    `np.random.default_rng` (which accepts arbitrary-width ints) so distinct continuation
    IDs land on independent streams."""
    if not experiment:
        raise ValueError("experiment tag must be non-empty")
    blob = f"{experiment}|{task_id}|{decision_digest}|{continuation_id}".encode()
    return int.from_bytes(hashlib.sha256(blob).digest(), "big")


def fork_future_demand(
    env: TokenomicsAECEnv,
    *,
    task_id: str,
    decision_digest: int,
    continuation_id: int,
    experiment: str = _TAG,
) -> TokenomicsAECEnv:
    """Return an independent deep clone of `env` whose randomized consumers have only their
    *unconsumed* future-demand suffix resampled under the continuation-derived RNG.

    The source `env` is never mutated. Every already-consumed schedule entry (indices
    `< consumer._cycle`, which includes the current cycle's demand — already placed as open
    orders) is preserved verbatim; only `schedule[_cycle:]` is redrawn. Fixed
    `ScriptedConsumer`s and non-randomized `ScheduledConsumer`s are left unchanged.

    How the tail is redrawn depends on the world's demand generator (ADR-0056; design D4):
    - B9 without drift: i.i.d. from the consumer's legal `(pool, quantity_range)` via
      `draw_schedule_entry`, consumers in sorted-id order (the EXP-0040 stream, unchanged);
    - demand drift j: continued from the current cycle's orders by the drift process, at most
      floor(j·N) changes per transition (`_drift_tail`);
    - burst / seasonal pattern: each future order redrawn from the pattern's generator for its
      cycle (`_pattern_tail`); these draws never depend on an earlier draw;
    - moving hit / regime switch: refused, since the future depends on hidden pattern state
      (where the hit is, when the next switch comes) that ADR-0056 defines no posterior for.

    Because only the future suffix moves, every agent's observation, current open orders,
    and current demand are byte-identical to the source (validated in tests)."""
    clone = copy.deepcopy(env)
    resample_future_demand(clone, np.random.default_rng(continuation_seed(
        task_id, decision_digest, continuation_id, experiment=experiment)))
    return clone


def resample_future_demand(clone: TokenomicsAECEnv, rng: np.random.Generator) -> None:
    """Redraw `clone`'s unconsumed demand suffix in place on `rng`, by the world's demand
    generator (`fork_future_demand`). The teacher's posterior forks (env/shock_posterior.py)
    call it on their stripped copy."""
    pattern = clone.config.demand_pattern
    if pattern is not None and pattern.kind in REFUSED_PATTERNS:
        raise ValueError(f"fork_future_demand refuses a {pattern.kind!r} demand pattern: its "
                         "future depends on hidden pattern state (ADR-0056)")
    randomized = [c for c in clone.config.consumers
                  if c.is_randomized and isinstance(clone.system.consumers.get(c.id),
                                                    ScheduledConsumer)]
    if clone.config.demand_drift is not None:
        _drift_tail(clone, randomized, clone.config.demand_drift.j, rng)
    elif pattern is not None:
        _pattern_tail(clone, randomized, pattern, rng)
    else:
        cfg_by_id = {c.id: c for c in randomized}
        for cid in sorted(cfg_by_id):
            cfg = cfg_by_id[cid]
            consumer = clone.system.consumers[cid]
            start = consumer._cycle
            consumer._schedule[start:] = [
                draw_schedule_entry(rng, cfg.product_pool, cfg.quantity_min, cfg.quantity_max)
                for _ in range(len(consumer._schedule) - start)
            ]


def _drift_tail(clone: TokenomicsAECEnv, drifting: list[ConsumerConfig], j: float,
                rng: np.random.Generator) -> None:
    """Continue the drift process (env/demand_drift.py) from the current cycle's orders:
    cycle-major, each transition k = floor(j·N) distinct consumers chosen without
    replacement, each redrawn from its domain minus its current entry, in config order (the
    generator's own transition, on the fork's stream)."""
    if not drifting:
        return
    consumers = [clone.system.consumers[c.id] for c in drifting]
    start = consumers[0]._cycle
    length = len(consumers[0]._schedule)
    if any(s._cycle != start or len(s._schedule) != length for s in consumers):
        raise ValueError("drifting consumers are not cycle-aligned")
    if start < 1:
        raise ValueError("a drift fork continues from the current cycle's orders; "
                         "no order has been placed yet")
    k = drift_budget(j, len(drifting))
    current = [s._schedule[start - 1] for s in consumers]
    for i in range(start, length):
        if k:
            for n in sorted(int(x) for x in rng.choice(len(drifting), size=k, replace=False)):
                current[n] = redraw(rng, drifting[n], current[n])
        for s, e in zip(consumers, current):
            s._schedule[i] = e


def _pattern_tail(clone: TokenomicsAECEnv, patterned: list[ConsumerConfig],
                  pattern: DemandPattern, rng: np.random.Generator) -> None:
    """Redraw every future order of a burst / seasonal world from the pattern's generator,
    cycle-major and consumers in config order, as `pattern_schedules` does. Schedule index i
    is the order of cycle i + 1, so the seasonal envelope is evaluated at the forked cycle."""
    if not patterned:
        return
    consumers = [clone.system.consumers[c.id] for c in patterned]
    start = consumers[0]._cycle
    length = len(consumers[0]._schedule)
    if any(s._cycle != start or len(s._schedule) != length for s in consumers):
        raise ValueError("patterned consumers are not cycle-aligned")
    for i in range(start, length):
        for c, s in zip(patterned, consumers):
            s._schedule[i] = (burst_entry(rng, c, pattern) if pattern.kind == PATTERN_BURST
                              else seasonal_entry(rng, c, pattern, i + 1))
