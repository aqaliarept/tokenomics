"""The five routers gate v3 promotes from the diagnostic routers into the gate's ordering
baselines (add-gate-v3 design D1): `round_robin`, `even_split_rotating`, `random_whole`,
`makeable_weighted_whole` and `makeable_split`, with the per-slot plumbing they share.

Moved here unchanged from `routing_diagnostics.py` (add-m3-development-audit D2 / D12, where
they were written): `routing_diagnostics` imports `baselines`, so `baselines` cannot import it
back. This module imports neither; `baselines.ORDERING_BASELINES` registers the five routers and
`routing_diagnostics` re-exports them, and the plumbing, under the same names in
`DIAGNOSTIC_ROUTERS`. Both entries are bound to the same function objects, so a router's actions
are identical as a baseline and as a diagnostic router.

They are white-box policies like `proportional` and emit the raw ordering action decode reads
(ADR-0025/0026); an exact integer split is emitted as `split_logits`. `random_whole` and
`makeable_weighted_whole` draw only from the env action-space RNG, which `run_episode` seeds with
the gate's replicate seed (`eval/seeding.py`), so they are `baselines.STOCHASTIC_BASELINES`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..domain.common import OrderId, ProducerId, Product
from ..domain.metrics import makeable_qty_now
from ..env.allocator import _allocate_int
from ..env.spaces import ACTION_LOGIT_BOUND

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv


# --- per-slot plumbing (shared with routing_diagnostics' routers) ---


def _slots(env: "TokenomicsAECEnv", agent: str) -> list[tuple[int, OrderId, int, list[int]]]:
    """(slot, order id, quantity, eligible producer indices in `spec.producer_ids` order) for
    every order the agent addresses through a slot; orders beyond `k_stream` have no slot and
    decode even-splits them, for every router alike."""
    sp = env.spec_
    _, aid = env._role[agent]
    view = env.system.ordering_agents[aid].view(env.system.producer_infos)
    out = []
    for k, oid in enumerate(sorted(view.incoming)):
        if k >= sp.k_stream:
            break
        eligible = [i for i, pid in enumerate(sp.producer_ids) if pid in view.eligible[oid]]
        out.append((k, oid, view.incoming[oid].quantity, eligible))
    return out


def _matrix(env: "TokenomicsAECEnv") -> np.ndarray:
    return np.zeros((env.spec_.k_stream, env.spec_.num_producers), dtype=np.float32)


def _flat(mat: np.ndarray) -> np.ndarray:
    return np.clip(mat, -ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND).astype(np.float32).flatten()


def split_logits(units: Sequence[int]) -> np.ndarray:
    """Logits whose decode gives exactly `units` (one entry per eligible producer, in decode
    order): `log(u)` where u > 0, `-BOUND` where u = 0. Softmax shares are then u / q up to
    float error (and e^-BOUND mass on the zeros), so every receiving producer's raw share sits
    within ~1e-4 of its integer and the largest-remainder step restores it (tested on every
    split of q <= 12 over up to 14 producers)."""
    return np.array([math.log(u) if u > 0 else -ACTION_LOGIT_BOUND for u in units],
                    dtype=np.float32)


def _whole(eligible: Sequence[int], chosen: int, quantity: int) -> np.ndarray:
    return split_logits([quantity if j == chosen else 0 for j in range(len(eligible))])


def _even_with_leftover(quantity: int, n: int, leftover_to: Sequence[int]) -> np.ndarray:
    """Even split of `quantity` over `n` producers; the `quantity % n` leftover units go to the
    positions `leftover_to` (distinct), one each."""
    base, rest = divmod(quantity, n)
    units = [base] * n
    for j in leftover_to[:rest]:
        units[j] += 1
    return split_logits(units)


def stable_hash(*parts: object) -> int:
    """Process-independent hash (no `hash()` salt): 64 bits of sha256."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


# --- deterministic rotation routers ---


def ordering_round_robin(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Whole order to eligible producer number (cycle + stable_hash(agent, slot)) mod E: a
    deterministic rotation over time (D2 (b)), never a split."""
    mat = _matrix(env)
    cycle = env.system.cycle_counter
    for k, _, q, elig in _slots(env, agent):
        if elig:
            chosen = (cycle + stable_hash(agent, k)) % len(elig)
            mat[k, elig] = _whole(elig, chosen, q)
    return _flat(mat)


def ordering_even_split_rotating(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """`even_split` with the leftover units placed by a cycle-indexed rotation (eligible
    positions cycle mod E onward) instead of on the lowest-index producers (D2 (c))."""
    mat = _matrix(env)
    cycle = env.system.cycle_counter
    for k, _, q, elig in _slots(env, agent):
        if elig:
            n = len(elig)
            mat[k, elig] = _even_with_leftover(q, n, [(cycle + i) % n for i in range(n)])
    return _flat(mat)


# --- stochastic routers (env action-space RNG only) ---


def _rng(env: "TokenomicsAECEnv", agent: str) -> np.random.Generator:
    return env.action_space(agent).np_random


def ordering_random_whole(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Whole order to one eligible producer drawn uniformly (D2 (b)/(f): random, never split)."""
    mat = _matrix(env)
    rng = _rng(env, agent)
    for k, _, q, elig in _slots(env, agent):
        if elig:
            mat[k, elig] = _whole(elig, int(rng.integers(len(elig))), q)
    return _flat(mat)


# --- makeable-steered routers (EXP-0050, atlas reach arms) ---


def makeable_units(env: "TokenomicsAECEnv", pid: ProducerId, product: Product) -> int:
    """Units of `product` producer `pid` can complete now: min(units its free capacity allows,
    units its inputs in stock allow), i.e. `domain.metrics.makeable_qty_now`, the
    `makeable_qty_now` field of the ordering agent's own product-stream view (encoded in the
    orderer obs v3/v4) and the portable scorer's distillation target (`train/orderer_scorer.py`
    `teacher_logits`). Unlike `spare_units` it counts input stock. It reads no shock state. It
    reads `env.system`, so it also runs under LEGACY (roster v3's `obs_stock_aware` members,
    add-gate-v3 D1); the audits run it joint_stack only."""
    return makeable_qty_now(env.system.producing_agents[pid], product)


def _makeables(env: "TokenomicsAECEnv", oid: OrderId, elig: Sequence[int]) -> list[int]:
    product = env.system.orders[oid].product
    pids = env.spec_.producer_ids
    return [makeable_units(env, pids[i], product) for i in elig]


def ordering_makeable_split(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """The scorer's teacher as a router: each order split over its eligible producers in
    proportion to their `makeable_units`, integer units by the decode's own largest remainder
    (`env.allocator._allocate_int`, ties to the lowest index); an even split when every eligible
    producer has 0. Deterministic."""
    mat = _matrix(env)
    for k, oid, q, elig in _slots(env, agent):
        if not elig:
            continue
        made = np.asarray(_makeables(env, oid, elig), dtype=np.float64)
        total = made.sum()
        props = made / total if total > 0 else np.full(len(elig), 1.0 / len(elig))
        mat[k, elig] = split_logits(_allocate_int(props, q))
    return _flat(mat)


def ordering_makeable_weighted_whole(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Whole order to one eligible producer drawn with probability proportional to its
    `makeable_units` (uniformly when every eligible producer has none): the same expected
    allocation as `makeable_split`, never split."""
    mat = _matrix(env)
    rng = _rng(env, agent)
    for k, oid, q, elig in _slots(env, agent):
        if not elig:
            continue
        made = np.asarray(_makeables(env, oid, elig), dtype=np.float64)
        total = made.sum()
        probs = made / total if total > 0 else np.full(len(elig), 1.0 / len(elig))
        mat[k, elig] = _whole(elig, int(rng.choice(len(elig), p=probs)), q)
    return _flat(mat)
