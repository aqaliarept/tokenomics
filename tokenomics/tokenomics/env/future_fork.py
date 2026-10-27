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

from ..domain.consumer import ScheduledConsumer
from .aec_env import TokenomicsAECEnv, draw_schedule_entry

_TAG = "EXP-0040"


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
    orders) is preserved verbatim; only `schedule[_cycle:]` is redrawn from the consumer's
    legal `(pool, quantity_range)` via `draw_schedule_entry`. Consumers are visited in
    sorted-id order so the RNG stream is deterministic. Fixed `ScriptedConsumer`s and
    non-randomized `ScheduledConsumer`s are left unchanged.

    Because only the future suffix moves, every agent's observation, current open orders,
    and current demand are byte-identical to the source (validated in tests)."""
    if env.config.demand_drift is not None:
        # An i.i.d. redraw of each tail would break the drift bound (add-shocks-i-drift D4);
        # a drift-aware fork is part of the M3 teacher fork rule, not this diagnostic.
        raise ValueError("fork_future_demand does not support demand drift")
    clone = copy.deepcopy(env)
    rng = np.random.default_rng(continuation_seed(
        task_id, decision_digest, continuation_id, experiment=experiment))
    cfg_by_id = {c.id: c for c in clone.config.consumers}
    for cid in sorted(clone.system.consumers):
        consumer = clone.system.consumers[cid]
        cfg = cfg_by_id.get(cid)
        if not isinstance(consumer, ScheduledConsumer) or cfg is None or not cfg.is_randomized:
            continue
        pool = cfg.product_pool
        start = consumer._cycle
        consumer._schedule[start:] = [
            draw_schedule_entry(rng, pool, cfg.quantity_min, cfg.quantity_max)
            for _ in range(len(consumer._schedule) - start)
        ]
    return clone
