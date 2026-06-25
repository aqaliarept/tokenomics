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

from ..domain.common import is_nan
from ..env.spaces import ACTION_LOGIT_BOUND, OBS_CLIP, EnvSpec

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

# A policy maps (env, agent name) -> a raw env-space action vector.
Policy = Callable[["TokenomicsAECEnv", str], np.ndarray]


# --- observation-compatible baselines (NEXT STEPS stage 1) ---
# Unlike the white-box baselines above, these decide from ONLY the observation
# vector the RL policy sees — never env.system. The decision is a pure function of
# (spec, obs): the spec supplies the *static* layout + normalization constants
# (cap/inv refs, replenish ceiling) that interpret the obs, exactly what a trained
# net learns implicitly. This makes them the fair "observation-compatible baseline"
# the release bar is measured against.
def decide_producing_obs(
    spec: EnvSpec,
    obs: np.ndarray,
    pid: str,
    *,
    threshold: float,
    stock_target: int,
) -> np.ndarray:
    """Obs-only analog of `producing_stock_aware`: restore when an observed capacity
    ratio drops below `threshold`. Reads current/max capacity straight from the
    producing obs layout (encode.encode_producing)."""
    t = spec.num_types
    caps = obs[0:t]
    maxs = obs[t : 2 * t]
    act = np.zeros(spec.producing_action_dim, dtype=np.float32)
    low = any(maxs[i] > 0 and caps[i] < threshold * maxs[i] for i in range(t))
    if low:
        act[spec.num_inputs + 0] = ACTION_LOGIT_BOUND  # restoration toggle

    # top each relevant input toward `stock_target`, reading observed stock from the
    # inventory section of the obs (denormalized by inv_ref) and mapping the wanted
    # quantity back to its raw value (raw/BOUND*max_r -> qty, mirrors decode).
    inv_start = 2 * t + 4  # after caps(2T), requested(1), 3 invest-running flags
    inbound_start = inv_start + spec.num_inputs  # in-flight replenishment (ADR-0041)
    max_r = max(1, spec.max_replenish[pid])
    for prod in spec.producer_inputs[pid]:
        gi = spec.input_index[prod]
        # count in-flight inbound so we do not re-order stock already on the way —
        # mirrors the white-box `producing_stock_aware`; without it a deep pipeline
        # never primes (Finding A).
        have = (obs[inv_start + gi] + obs[inbound_start + gi]) * spec.inv_ref
        want = stock_target - have
        if want > 0:
            act[gi] = min(1.0, want / max_r) * ACTION_LOGIT_BOUND
    return act


def _ordering_obs_price_action(spec: EnvSpec, prices: np.ndarray) -> np.ndarray:
    logits = np.where(prices > 0.0, -prices, -ACTION_LOGIT_BOUND).astype(np.float32)
    mat = np.tile(logits, (spec.k, 1))
    return np.clip(mat, -ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND).astype(np.float32).flatten()


def decide_ordering_obs(spec: EnvSpec, obs: np.ndarray) -> np.ndarray:
    """Obs-only ordering router: the ordering obs hides producer capacity (ADR-0029),
    so the only routing signal available is each producer's cut-off price. Route
    toward the *cheapest* (lowest cut-off price = most spare/least contested
    capacity) by emitting a per-producer logit of -price, replicated across slots;
    decode masks each slot to its eligible producers and softmaxes."""
    # version-aware: the cut-off price block moved under enriched v2 obs (strided in the
    # per-producer block), so read it via the spec's indices, not a hardcoded v1 offset —
    # keeps obs_price identical across v1/v2 (the search candidate held fixed).
    prices = np.asarray(obs[spec.ordering_price_indices()], dtype=np.float32)
    # prefer lower cut-off price (slack capacity), BUT a 0 price is "no production
    # yet" (UNDEFINED -> 0 in encode), not "cheapest" — often a producer that cannot
    # produce (capacity hidden, ADR-0029). Park unproven producers at the floor so a
    # proven cheap one always wins; at cold start all-0 collapses to an even split.
    return _ordering_obs_price_action(spec, prices)


def producing_obs_stock_aware(threshold: float = 0.6, stock_target: int = 3) -> Policy:
    """Observation-compatible counterpart of `producing_stock_aware`: same restore-
    when-low + top-up-inputs behavior, but decided from `env.observe(agent)` alone —
    never `env.system`. The fair producing bar for the RL policy."""

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        _, pid = env._role[agent]
        return decide_producing_obs(
            env.spec_, env.observe(agent), pid, threshold=threshold, stock_target=stock_target
        )

    return policy


def ordering_obs_price(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Observation-compatible ordering baseline: routes by observed cut-off price
    (no capacity/throughput, unlike the white-box `ordering_proportional`)."""
    return decide_ordering_obs(env.spec_, env.observe(agent))


def ordering_obs_price_from_state(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Fast search equivalent of `ordering_obs_price`.

    Ordering observations encode only the latest producer cut-off prices for this
    baseline. During ordering, `system.producer_infos` is the same snapshot used by
    `env.observe()`, so the action is identical without constructing the full
    product-stream/enriched observation. Keep `ORDERING_BASELINES["obs_price"]`
    observation-only; use this only in search internals.
    """
    prices = np.asarray([
        0.0 if is_nan(env.system.producer_infos[pid].cut_off_price)
        else np.clip(
            env.system.producer_infos[pid].cut_off_price / env.spec_.price_ref,
            0.0, OBS_CLIP,
        )
        for pid in env.spec_.producer_ids
    ], dtype=np.float32)
    return _ordering_obs_price_action(env.spec_, prices)


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
        # OrderInfo no longer carries capacity (ADR-0029); a white-box baseline
        # reads the recipe straight from the process sheet to gauge throughput.
        require = env.system.process_sheets[oi.product].require
        for pid in view.eligible[oid]:
            p = env.system.producing_agents[pid]  # white-box: capacity net of reservations
            reserved = p.reserved_capacity()
            avail = {t: p.capacities[t] - reserved.get(t, 0) for t in require}
            throughput = min(avail[t] // c for t, c in require.items())
            # A producer that can make 0 units must rank well below one that can make
            # 1 (log(max(.,1)) collapsed both to 0): floor it at -BOUND -> ~0 share.
            if throughput <= 0:
                logit = -ACTION_LOGIT_BOUND
            else:
                logit = min(float(np.log(throughput)), ACTION_LOGIT_BOUND)
            mat[k, sp.producer_ids.index(pid)] = logit
    return mat.flatten()


def random_action(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    return env.action_space(agent).sample()


PRODUCING_BASELINES: dict[str, Policy] = {
    "noop": producing_noop,
    "always_restore": producing_always_restore,
    "always_both": producing_always_both,
    "restore_when_low": producing_restore_when_low(0.6),
    "stock_aware": producing_stock_aware(0.6),
    # observation-compatible bar (NEXT STEPS stage 1): obs-only, no env.system.
    "obs_stock_aware": producing_obs_stock_aware(0.6),
    "random": random_action,
}

ORDERING_BASELINES: dict[str, Policy] = {
    "even_split": ordering_even_split,
    "proportional": ordering_proportional,
    "obs_price": ordering_obs_price,  # observation-compatible (cut-off price routing)
    "random": random_action,
}
