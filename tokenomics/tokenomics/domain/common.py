"""Core type aliases for the tokenomics domain.

Faithful port of emulation/domain/common.go. These are plain aliases (not
NewType) to keep the port readable; the Go types are all primitives too.

Note: Capacity is a *signed* int in Go (`type Capacity int`). The auction in
ProducingAgent.produce relies on `remaining_capacity` going negative, so do not
make it unsigned here.
"""

from __future__ import annotations

import math
from typing import TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

# Tokens uint, Product uint, CapacityType string, Capacity int (signed)
Tokens = int
Product = int
CapacityType = str
Capacity = int

# Id types — all plain strings in Go (distinct named types, same representation).
OrderId = str
ProducerId = str
ConsumerId = str
OrderingAgentId = str

# CapacityUnitPrice is a float64 in Go with a NaN "undefined" sentinel.
CapacityUnitPrice = float
UNDEFINED_PRICE: float = math.nan


def is_nan(x: float) -> bool:
    return math.isnan(x)


def f32(x: float) -> float:
    """Round-trip through float32 to match Go's float32 arithmetic for prices
    and fund distribution (parity-sensitive). See plan §"Float determinism"."""
    import numpy as np

    return float(np.float32(x))


def from_producer_id(pid: ProducerId) -> OrderingAgentId:
    return pid


def from_consumer_id(cid: ConsumerId) -> OrderingAgentId:
    return cid


def stream_id(source_id: str, product: Product) -> OrderingAgentId:
    """Product-stream ordering-agent id (ADR-0052): one agent per (source, product)
    so a single agent never multiplexes products. `source_id` is the raw consumer
    or producer id (identity-mapped from `from_consumer_id`/`from_producer_id`)."""
    return f"{source_id}__{product}"


class NotFoundError(Exception):
    """Mirror of domain.ErrNotFound."""


class WrongStateError(Exception):
    """Mirror of domain.ErrWrongState."""


def must_get(m: dict[_K, _V], k: _K) -> _V:
    """Fetch or raise NotFoundError."""
    if k not in m:
        raise NotFoundError(f"{type(k).__name__}: {k!r}")
    return m[k]
