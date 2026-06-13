"""Raw policy actions -> domain commands.

Ordering action uses simplex normalization (ADR-0002): per active order, softmax
the logits of the capacity types that order requires, scale by the order's token
budget, and round to integers that sum exactly to the budget. Producing action
is a 2-bit discrete (restoration, upgrade).
"""

from __future__ import annotations

import numpy as np

from ..domain.order import OrderInfo
from ..domain.ordering_agent import OrderingAgentCommand
from ..domain.producing_agent import ProducingAgentCommand
from .spaces import EnvSpec


def decode_producing(action: int) -> ProducingAgentCommand:
    return ProducingAgentCommand(do_restoration=bool(action & 1), do_upgrade=bool(action & 2))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _allocate_int(proportions: np.ndarray, budget: int) -> list[int]:
    """Split `budget` into non-negative ints proportional to `proportions`,
    summing exactly to budget. Largest fractional remainders get the leftover."""
    if budget <= 0:
        return [0] * len(proportions)
    raw = proportions * budget
    floors = np.floor(raw).astype(int)
    remainder = budget - int(floors.sum())
    if remainder > 0:
        frac = raw - floors
        order = np.argsort(-frac)  # largest fractional part first
        for i in range(remainder):
            floors[order[i % len(floors)]] += 1
    return floors.tolist()


def decode_ordering(
    spec: EnvSpec, action: np.ndarray, incoming: dict[str, OrderInfo]
) -> OrderingAgentCommand:
    action = np.asarray(action, dtype=np.float64).reshape(spec.k, spec.num_types)
    ordered = [incoming[oid] for oid in sorted(incoming.keys())]
    orders: dict[str, dict[str, int]] = {}
    for k, oi in enumerate(ordered):
        req_types = list(oi.required.keys())
        if k < spec.k:
            idx = [spec.type_index[t] for t in req_types]
            props = _softmax(action[k, idx])
        else:
            # More orders than slots (K should cover the worst case): fall back to
            # an even split so every incoming order is still addressed — the
            # domain rejects a command that leaves any order unaddressed.
            props = np.full(len(req_types), 1.0 / len(req_types))
        tokens = _allocate_int(props, oi.tokens)
        alloc: dict[str, int] = {}
        for t, tok in zip(req_types, tokens):
            alloc[spec.type_to_producer[t]] = tok
        orders[oi.id] = alloc
    return OrderingAgentCommand(orders)
