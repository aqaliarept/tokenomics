"""The fund turn and the accept turn of a producing agent (ADR-0062): spaces, observation encoders,
action decoders and the default policies that reproduce today's behaviour.

Both turns read only the producer's own state (ADR-0058): its own funding requests and stage-1
budget; its routed bids (with each bid's own-order remaining lifetime, ADR-0061), its carried
reservations and pending claims, its capacity, inventory and inbound. Actions are scores in
[-ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND]; the domain applies them in score order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from ..domain.order import Order
from ..domain.producing_agent import (K_BID, K_PEND, K_RES, Bid, ProducingAgent, TurnDecision,
                                      TurnLists, _Reservation)
from ..domain.requests import InvestmentType
from .spaces import ACTION_LOGIT_BOUND, EnvSpec

if TYPE_CHECKING:
    from .aec_env import TokenomicsAECEnv

B = ACTION_LOGIT_BOUND
FUND_KINDS = ("restoration", "replenishment", "upgrade", "inventory_expansion")
FUND_SLOT_DIM = len(FUND_KINDS) + 5  # present, kinds, target, quantity, stock, inbound
FUND_GLOBAL_DIM = 3  # budget, requests, inventory units
ORDER_KINDS = ("consumer", "replenishment", "investment")  # the parent order's kind (own-order)
BID_DIM = 8 + len(ORDER_KINDS)  # present, price, quantity, capacity, remaining ttl, composite,
#                                 input cover, fits alone, order kind
CLAIM_DIM = 6 + len(ORDER_KINDS)  # present, price, quantity, capacity, cycles waited, inputs, kind
ACC_GLOBAL_DIM = 5  # capacity, max capacity, new-work budget, inventory units, inbound units
ACCEPT_ACTION_DIM = 2 * K_BID + K_RES + K_PEND


def fund_slots(spec: EnvSpec) -> int:
    """3 investment kinds + the most input products any producer can replenish."""
    return 3 + max((len(v) for v in spec.producer_inputs.values()), default=0)


def fund_obs_space(spec: EnvSpec) -> spaces.Box:
    n = fund_slots(spec) * FUND_SLOT_DIM + FUND_GLOBAL_DIM
    return spaces.Box(-np.inf, np.inf, (n,), np.float32)


def fund_action_space(spec: EnvSpec) -> spaces.Box:
    return spaces.Box(-B, B, (fund_slots(spec),), np.float32)


def accept_obs_space(spec: EnvSpec) -> spaces.Box:
    n = K_BID * BID_DIM + (K_RES + K_PEND) * CLAIM_DIM + ACC_GLOBAL_DIM
    return spaces.Box(-np.inf, np.inf, (n,), np.float32)


def accept_action_space(spec: EnvSpec) -> spaces.Box:
    return spaces.Box(-B, B, (ACCEPT_ACTION_DIM,), np.float32)


# --- the fund turn ---


def fund_kind(o: Order) -> str:
    if o.replenishment_request is not None:
        return "replenishment"
    t = o.investment_request.type
    return {InvestmentType.RESTORATION: "restoration", InvestmentType.UPGRADE: "upgrade",
            InvestmentType.INVENTORY_EXPANSION: "inventory_expansion"}[t]


def encode_fund(spec: EnvSpec, p: ProducingAgent, reqs: list[Order], budget: int) -> np.ndarray:
    n = fund_slots(spec)
    slots = np.zeros((n, FUND_SLOT_DIM), np.float32)
    for i, o in enumerate(reqs[:n]):
        kind = fund_kind(o)
        target = o.funding_target()
        stock = p.inventory.get(o.product, 0) if kind == "replenishment" else 0
        inbound = p.outstanding_inbound.get(o.product, 0) if kind == "replenishment" else 0
        slots[i, 0] = 1.0
        slots[i, 1 + FUND_KINDS.index(kind)] = 1.0
        slots[i, 5] = (budget if target is None else target) / spec.token_ref
        slots[i, 6] = o.quantity / spec.inv_ref
        slots[i, 7] = stock / spec.inv_ref
        slots[i, 8] = inbound / spec.inv_ref
    glob = np.array([budget / spec.token_ref, len(reqs), p.inventory_units() / spec.inv_ref],
                    np.float32)
    return np.concatenate([slots.reshape(-1), glob])


def decode_fund(action: np.ndarray, n_requests: int) -> list[float]:
    """One score per request slot; requests beyond the slots keep today's order after them."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    k = a.shape[0]
    return [float(a[i]) if i < k else -B - 1.0 - (i - k) * 1e-3 for i in range(n_requests)]


def default_fund_scores(n: int) -> list[float]:
    """Today's stage-2 order (the requests are listed in it) as descending scores."""
    return [B - 2 * B * i / max(1, n) for i in range(n)]


def default_fund_policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    pid = env._role[agent][1]
    n = len(env.system.funding_requests(pid))
    out = np.full(fund_slots(env.spec_), -B, np.float32)
    s = default_fund_scores(n)[:out.shape[0]]
    out[:len(s)] = s
    return out


# --- the accept turn ---


def kind_onehot(bid: Bid) -> list[float]:
    return [float(bid.order_kind == k) for k in ORDER_KINDS]


def _cover(p: ProducingAgent, bid: Bid) -> float:
    """Min over inputs of (stock + inbound) / need, capped at 1; 1 for a primal bid."""
    if not bid.inputs:
        return 1.0
    return min(1.0, min((p.inventory.get(g, 0) + p.outstanding_inbound.get(g, 0)) / q
                        for g, q in bid.inputs.items()))


def encode_accept(spec: EnvSpec, p: ProducingAgent, lists: TurnLists) -> np.ndarray:
    budget = p._new_work_budget(lists)
    bids = np.zeros((K_BID, BID_DIM), np.float32)
    for i, b in enumerate(lists.bids[:K_BID]):
        life = -1.0 if b.remaining_ttl is None else float(b.remaining_ttl)
        bids[i] = [1.0, b.capacity_unit_price() / spec.price_ref, b.quantity / spec.inv_ref,
                   b.total_capacity() / spec.cap_ref, life, float(b.is_composite()), _cover(p, b),
                   float(p._fits(budget, b.capacities)), *kind_onehot(b)]

    def claims(rs: tuple[_Reservation, ...], k: int) -> np.ndarray:
        out = np.zeros((k, CLAIM_DIM), np.float32)
        for i, r in enumerate(rs[:k]):
            out[i] = [1.0, r.bid.capacity_unit_price() / spec.price_ref,
                      r.bid.quantity / spec.inv_ref, r.bid.total_capacity() / spec.cap_ref,
                      float(r.cycles_waited), float(p._has_inputs(r.bid)), *kind_onehot(r.bid)]
        return out

    glob = np.array([sum(p.capacities.values()) / spec.cap_ref,
                     sum(p.max_capacities.values()) / spec.cap_ref,
                     sum(budget.values()) / spec.cap_ref, p.inventory_units() / spec.inv_ref,
                     p.outstanding_inbound_units() / spec.inv_ref], np.float32)
    return np.concatenate([bids.reshape(-1), claims(lists.reservations, K_RES).reshape(-1),
                           claims(lists.pending, K_PEND).reshape(-1), glob])


def decode_accept(action: np.ndarray, lists: TurnLists) -> tuple[TurnDecision, int]:
    """(the decision, how many bids / claims fell beyond the slots). Bids beyond K_BID are
    declined (ADR-0061 D4); claims beyond the slots run after the listed ones in default order."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    acc, bp = a[:K_BID], a[K_BID:2 * K_BID]
    rp, pp = a[2 * K_BID:2 * K_BID + K_RES], a[2 * K_BID + K_RES:]
    nb = len(lists.bids)
    accept = tuple(bool(acc[i] > 0) if i < K_BID else False for i in range(nb))
    bid_pri = tuple(float(bp[i]) if i < K_BID else -B - 1.0 for i in range(nb))

    def pri(vals: np.ndarray, n: int, k: int) -> tuple[float, ...]:
        return tuple(float(vals[i]) if i < k else -B - 1.0 - (i - k) * 1e-3 for i in range(n))

    beyond = (max(0, nb - K_BID) + max(0, len(lists.reservations) - K_RES)
              + max(0, len(lists.pending) - K_PEND))
    return TurnDecision(accept, bid_pri, pri(rp, len(lists.reservations), K_RES),
                        pri(pp, len(lists.pending), K_PEND)), beyond


def encode_accept_action(d: TurnDecision) -> np.ndarray:
    """A decision as the accept turn's action (slots beyond K dropped; unused slots -B)."""
    out = np.full(ACCEPT_ACTION_DIM, -B, np.float32)
    for i, (ok, pr) in enumerate(zip(d.accept[:K_BID], d.bid_priority[:K_BID])):
        out[i] = 1.0 if ok else -1.0
        out[K_BID + i] = pr
    out[2 * K_BID:2 * K_BID + min(K_RES, len(d.reservation_priority))] = \
        d.reservation_priority[:K_RES]
    out[2 * K_BID + K_RES:2 * K_BID + K_RES + min(K_PEND, len(d.pending_priority))] = \
        d.pending_priority[:K_PEND]
    return out


def default_accept_policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
    """Today's auction and claim order as an accept-turn action (ADR-0062)."""
    p = env.system.producing_agents[env._role[agent][1]]
    return encode_accept_action(p.default_turn_decision())
