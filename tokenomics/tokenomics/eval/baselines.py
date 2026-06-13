"""Heuristic (non-RL) policies that act through the env action API.

These are white-box for measurement — they read `env.system` directly rather than
decoding the observation vector. A policy is `callable(env, agent) -> action` in
the env's action space (Discrete int for producing, logit vector for ordering).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

# A policy maps (env, agent name) -> an env-space action: a Discrete int for a
# producing agent, or a logit vector for an ordering agent.
Policy = Callable[["TokenomicsAECEnv", str], "int | np.ndarray"]


# --- producing policies: return int in {0:noop, 1:restore, 2:upgrade, 3:both} ---
def producing_noop(env: TokenomicsAECEnv, agent: str) -> int:
    return 0


def producing_always_restore(env: TokenomicsAECEnv, agent: str) -> int:
    return 1


def producing_always_both(env: TokenomicsAECEnv, agent: str) -> int:
    return 3


def producing_restore_when_low(threshold: float = 0.6) -> Policy:
    """Restore when current capacity drops below `threshold` of max; never upgrade."""

    def policy(env: TokenomicsAECEnv, agent: str) -> int:
        _, pid = env._role[agent]
        p = env.system.producing_agents[pid]
        return 1 if p.capacity < threshold * p.max_capacity else 0

    return policy


# --- ordering policies: return a flat logit vector of length k*num_types ---
def ordering_even_split(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    # zero logits -> uniform softmax over each order's required types
    return np.zeros(env.spec_.ordering_action_dim, dtype=np.float32)


def ordering_proportional(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    """Allocate each order's tokens proportional to the capacity it requires per
    type (logit = log(required capacity); softmax recovers the proportions)."""
    _, aid = env._role[agent]
    sp = env.spec_
    incoming = env.system.ordering_agents[aid].incoming
    mat = np.zeros((sp.k, sp.num_types), dtype=np.float32)
    ordered = [incoming[o] for o in sorted(incoming.keys())]
    for k, oi in enumerate(ordered):
        if k >= sp.k:
            break
        for t, cap in oi.required.items():
            mat[k, sp.type_index[t]] = float(np.log(max(cap, 1)))
    return mat.flatten()


def random_action(env: TokenomicsAECEnv, agent: str) -> int | np.ndarray:
    return env.action_space(agent).sample()


PRODUCING_BASELINES: dict[str, Policy] = {
    "noop": producing_noop,
    "always_restore": producing_always_restore,
    "always_both": producing_always_both,
    "restore_when_low": producing_restore_when_low(0.6),
    "random": random_action,
}

ORDERING_BASELINES: dict[str, Policy] = {
    "even_split": ordering_even_split,
    "proportional": ordering_proportional,
    "random": random_action,
}
