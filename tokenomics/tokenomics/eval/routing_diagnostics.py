"""Diagnostic ordering routers for the random-routing diagnosis (add-m3-development-audit design
D2, "Diagnostic routers"; development-audit "Diagnostic routers stay outside the roster").

Each router isolates one explanation of why `random` sets the hindsight baseline: rotation
without randomness (`round_robin`, `even_split_rotating`), load awareness
(`proportional_load_aware`), the logit scale (`random_scaled_s`), decorrelated but fixed
weights (`random_fixed_episode`), tempered noise (`blend_noise_s`), the order class
(`random_consumer_only` / `random_producer_only`) and whole versus split orders under
whole-order completion (`proportional_whole`, `random_whole`, `even_split_random_leftover`).

They are white-box `Policy`s like `proportional` / `blend_0.5` and emit the raw ordering action
decode reads (ADR-0025/0026). A router that wants an exact integer split (a whole order, a
chosen leftover placement) emits `log(units)` for each receiving producer and `-BOUND` for the
rest: the decode's softmax then reproduces the split up to float error far below one unit, and
its largest-remainder step lands every unit where it was meant to go (`split_logits`).
Stochastic routers draw only from the env action-space RNG, which `run_episode` seeds with the
gate's replicate seed (`eval/seeding.py`).

`DIAGNOSTIC_ROUTERS` is deliberately NOT part of `ORDERING_BASELINES`: the roster, the gate and
`signal_check` never see these routers (design D3: a fixed rule reaches the gate only under a
new name, through a roster v3 draft, after the user's decision).
"""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..domain.common import OrderId
from ..domain.system import order_kind
from ..env.spaces import ACTION_LOGIT_BOUND
from .baselines import Policy, ordering_blend, ordering_proportional, random_action
from .routing_trace import CONSUMER, PRODUCER, order_class

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

DETERMINISTIC = "deterministic"
STOCHASTIC = "stochastic"

_blend_half = ordering_blend(0.5)


# --- per-slot plumbing ---


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


# --- deterministic routers ---


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


def ordering_proportional_load_aware(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """`proportional` with each producer's throughput also net of the capacity of bids already
    placed on it this cycle (by the ordering agents that acted earlier), D2 (a)."""
    sp = env.spec_
    mat = _matrix(env)
    sheets = env.system.process_sheets
    _, aid = env._role[agent]
    view = env.system.ordering_agents[aid].view(env.system.producer_infos)
    for k, oid, _, elig in _slots(env, agent):
        require = sheets[view.incoming[oid].product].require
        for i in elig:
            p = env.system.producing_agents[sp.producer_ids[i]]
            reserved = p.reserved_capacity()
            placed: dict[str, int] = {}
            for bid in p.bids:
                for t, c in bid.capacities.items():
                    placed[t] = placed.get(t, 0) + c
            throughput = min((p.capacities[t] - reserved.get(t, 0) - placed.get(t, 0)) // c
                             for t, c in require.items())
            mat[k, i] = (-ACTION_LOGIT_BOUND if throughput <= 0
                         else min(math.log(throughput), ACTION_LOGIT_BOUND))
    return _flat(mat)


def ordering_proportional_whole(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Whole order to the producer with the largest `proportional` share (ties to the lowest
    index), no rotation: the deterministic whole-order control of D2 (b)/(f)."""
    sp = env.spec_
    prop = ordering_proportional(env, agent).reshape(sp.k_stream, sp.num_producers)
    mat = _matrix(env)
    for k, _, q, elig in _slots(env, agent):
        if elig:
            chosen = int(np.argmax(prop[k, elig]))  # first maximum = lowest index
            mat[k, elig] = _whole(elig, chosen, q)
    return _flat(mat)


# --- stochastic routers (env action-space RNG only) ---


def _rng(env: "TokenomicsAECEnv", agent: str) -> np.random.Generator:
    return env.action_space(agent).np_random


def ordering_random_scaled(scale: float) -> Policy:
    """Uniform logits in [-scale, scale]: `random`'s own draw rescaled, so scale = BOUND (10)
    is `random` action for action under the same seed (D2 (b) scale sweep)."""
    factor = float(scale) / ACTION_LOGIT_BOUND

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        return (random_action(env, agent) * np.float32(factor)).astype(np.float32)

    return policy


def ordering_random_fixed_episode() -> Policy:
    """Uniform [-BOUND, BOUND] logits per (agent, slot, producer), drawn at the agent's first
    decision after a reset and then held for the episode: decorrelated, but not varying over
    time (D2 (b)). The draw is keyed on the env's current `System`, which every reset replaces."""
    held: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        state = held.get(env)
        if state is None or state[0] is not env.system:
            state = (env.system, {})
            held[env] = state
        logits = state[1]
        if agent not in logits:
            logits[agent] = random_action(env, agent).astype(np.float32)
        return logits[agent].copy()

    return policy


def ordering_blend_noise(sigma: float) -> Policy:
    """`blend_0.5` logits plus N(0, sigma^2) noise (EXP-0006B's sigma range), clipped to the
    action bound (D2 (b))."""

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        base = _blend_half(env, agent)
        noise = _rng(env, agent).normal(0.0, sigma, size=base.shape)
        return _flat(base + noise)

    return policy


def ordering_random_one_class(randomised: str) -> Policy:
    """`random` on the slots holding orders of class `randomised` (consumer, or producer =
    replenishment + investment), `blend_0.5` on the others (D2 (d), the fund channel)."""
    if randomised not in (CONSUMER, PRODUCER):
        raise ValueError(f"order class must be {CONSUMER!r} or {PRODUCER!r}")

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        sp = env.spec_
        mat = _blend_half(env, agent).reshape(sp.k_stream, sp.num_producers).copy()
        draw = random_action(env, agent).reshape(sp.k_stream, sp.num_producers)
        for k, oid, _, _ in _slots(env, agent):
            if order_class(order_kind(env.system.orders[oid])) == randomised:
                mat[k] = draw[k]
        return _flat(mat)

    return policy


def ordering_random_whole(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Whole order to one eligible producer drawn uniformly (D2 (b)/(f): random, never split)."""
    mat = _matrix(env)
    rng = _rng(env, agent)
    for k, _, q, elig in _slots(env, agent):
        if elig:
            mat[k, elig] = _whole(elig, int(rng.integers(len(elig))), q)
    return _flat(mat)


def ordering_even_split_random_leftover(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Even split with the leftover units on distinct eligible producers drawn uniformly: the
    split counterpart of `random_whole` (D2 (f))."""
    mat = _matrix(env)
    rng = _rng(env, agent)
    for k, _, q, elig in _slots(env, agent):
        if elig:
            n = len(elig)
            rest = q % n
            picks = [int(j) for j in rng.choice(n, size=rest, replace=False)] if rest else []
            mat[k, elig] = _even_with_leftover(q, n, picks)
    return _flat(mat)


# Design D2 table. Not in ORDERING_BASELINES (module docstring).
DIAGNOSTIC_ROUTERS: dict[str, Policy] = {
    "round_robin": ordering_round_robin,
    "even_split_rotating": ordering_even_split_rotating,
    "proportional_load_aware": ordering_proportional_load_aware,
    "proportional_whole": ordering_proportional_whole,
    "random_scaled_0.5": ordering_random_scaled(0.5),
    "random_scaled_1": ordering_random_scaled(1.0),
    "random_scaled_2": ordering_random_scaled(2.0),
    "random_scaled_5": ordering_random_scaled(5.0),
    "random_fixed_episode": ordering_random_fixed_episode(),
    "blend_noise_0.15": ordering_blend_noise(0.15),
    "blend_noise_0.6": ordering_blend_noise(0.6),
    "random_consumer_only": ordering_random_one_class(CONSUMER),
    "random_producer_only": ordering_random_one_class(PRODUCER),
    "random_whole": ordering_random_whole,
    "even_split_random_leftover": ordering_even_split_random_leftover,
}

ROUTER_KIND: dict[str, str] = {
    name: (DETERMINISTIC if name in ("round_robin", "even_split_rotating",
                                     "proportional_load_aware", "proportional_whole")
           else STOCHASTIC)
    for name in DIAGNOSTIC_ROUTERS
}
