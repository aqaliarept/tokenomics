"""Heuristic (non-RL) policies that act through the env action API (ADR-0025).

White-box for measurement — they read `env.system` directly rather than decoding
the observation vector. A policy is `callable(env, agent) -> np.ndarray` in the
env's action space: both roles are now continuous `Box` actions (ADR-0026), so a
baseline emits the *raw* vector decode interprets.

Producing action = Box(G + 3): G replenishment dims then [restoration, upgrade,
inventory_expansion] toggles. A positive raw value fires a toggle; a full
(+BOUND) replenish dim asks for the per-cycle ceiling (decode caps it to
headroom). Ordering action = Box(K*P): per slot, a raw logit per producer; decode
restricts to the order's eligible producers and softmaxes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from ..env.spaces import ACTION_LOGIT_BOUND

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

# A policy maps (env, agent name) -> a raw env-space action vector.
Policy = Callable[["TokenomicsAECEnv", str], np.ndarray]


def _producing(
    env: "TokenomicsAECEnv",
    agent: str,
    *,
    restore: bool,
    upgrade: bool = False,
    stock_target: int = 0,
) -> np.ndarray:
    sp = env.spec_
    _, pid = env._role[agent]
    p = env.system.producing_agents[pid]
    act = np.zeros(sp.producing_action_dim, dtype=np.float32)
    if stock_target > 0:
        # top each relevant input up toward `stock_target` (current + inbound),
        # mapping the desired quantity back to its raw value (raw/BOUND*max -> qty).
        max_r = max(1, sp.max_replenish[pid])
        for g in sp.producer_inputs[pid]:
            have = p.inventory.get(g, 0) + p.outstanding_inbound.get(g, 0)
            want = max(0, stock_target - have)
            if want > 0:
                act[sp.input_index[g]] = min(1.0, want / max_r) * ACTION_LOGIT_BOUND
    if restore:
        act[sp.num_inputs + 0] = ACTION_LOGIT_BOUND
    if upgrade:
        act[sp.num_inputs + 1] = ACTION_LOGIT_BOUND
    return act


def _low_capacity(env: "TokenomicsAECEnv", pid: str, threshold: float) -> bool:
    p = env.system.producing_agents[pid]
    return any(p.capacities[t] < threshold * p.max_capacities[t] for t in p.capacities)


# --- producing policies ---
def producing_noop(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    return _producing(env, agent, restore=False)


def producing_always_restore(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    return _producing(env, agent, restore=True)


def producing_always_both(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    return _producing(env, agent, restore=True, upgrade=True)


def producing_restore_when_low(threshold: float = 0.6) -> Policy:
    """Restore when any owned capacity drops below `threshold` of its max; never
    replenishes — so it survives but cannot feed composite assembly."""

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        _, pid = env._role[agent]
        return _producing(env, agent, restore=_low_capacity(env, pid, threshold))

    return policy


def producing_stock_aware(threshold: float = 0.6, stock_target: int = 3) -> Policy:
    """Restore when low AND keep input stock topped up to `stock_target` — the only
    producing baseline that keeps composite assembly fed, so consumer demand can be
    met. Tops up toward the target (not the ceiling) to avoid flooding the fund with
    unfundable replenishment orders."""

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        _, pid = env._role[agent]
        return _producing(
            env, agent, restore=_low_capacity(env, pid, threshold), stock_target=stock_target
        )

    return policy


# --- ordering policies: raw logit vector of length k*P ---
def ordering_even_split(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    # zero logits -> uniform softmax over each order's eligible producers
    return np.zeros(env.spec_.ordering_action_dim, dtype=np.float32)


def ordering_proportional(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Weight each *eligible* producer by its throughput for the order's required
    bundle — units it could make this cycle, `min_t(capacity[t] // unit[t])` — so
    spare capacity in *irrelevant* types does not inflate its share. Logit =
    log(throughput), clamped to the action bound; softmax recovers the proportions."""
    sp = env.spec_
    _, aid = env._role[agent]
    view = env.system.ordering_agents[aid].view(env.system.producer_infos)
    mat = np.zeros((sp.k, sp.num_producers), dtype=np.float32)
    for k, oid in enumerate(sorted(view.incoming)):
        if k >= sp.k:
            break
        oi = view.incoming[oid]
        for pid in view.eligible[oid]:
            p = env.system.producing_agents[pid]  # white-box: capacity net of reservations
            reserved = p.reserved_capacity()
            avail = {t: p.capacities[t] - reserved.get(t, 0) for t in oi.unit_capacity}
            throughput = min(avail[t] // c for t, c in oi.unit_capacity.items())
            logit = float(np.log(max(throughput, 1)))
            mat[k, sp.producer_ids.index(pid)] = min(logit, ACTION_LOGIT_BOUND)
    return mat.flatten()


def random_action(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    return env.action_space(agent).sample()


PRODUCING_BASELINES: dict[str, Policy] = {
    "noop": producing_noop,
    "always_restore": producing_always_restore,
    "always_both": producing_always_both,
    "restore_when_low": producing_restore_when_low(0.6),
    "stock_aware": producing_stock_aware(0.6),
    "random": random_action,
}

ORDERING_BASELINES: dict[str, Policy] = {
    "even_split": ordering_even_split,
    "proportional": ordering_proportional,
    "random": random_action,
}
