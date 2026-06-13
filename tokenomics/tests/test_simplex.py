"""Simplex normalization (decode_ordering) validity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenomics.domain.config import load_config
from tokenomics.domain.order import OrderInfo
from tokenomics.env.decode import _allocate_int, decode_ordering
from tokenomics.env.spaces import build_spec

CONFIG = Path(__file__).resolve().parents[1] / "tokenomics" / "config" / "phase1.json"


def spec():
    return build_spec(load_config(CONFIG))


def test_allocate_int_sums_to_budget():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = rng.integers(1, 5)
        props = rng.random(n)
        props = props / props.sum()
        budget = int(rng.integers(0, 2000))
        out = _allocate_int(props, budget)
        assert sum(out) == budget
        assert all(x >= 0 for x in out)


def test_decode_respects_required_types_and_budget():
    sp = spec()
    rng = np.random.default_rng(1)
    # product 1 requires capacity-1 (15) and capacity-3 (8); budget 137
    incoming = {
        "o1": OrderInfo("o1", 137, {"capacity-1": 15, "capacity-3": 8}),
    }
    for _ in range(200):
        action = rng.uniform(-10, 10, size=sp.ordering_action_dim).astype(np.float32)
        cmd = decode_ordering(sp, action, incoming)
        assert set(cmd.orders.keys()) == {"o1"}
        alloc = cmd.orders["o1"]
        # only producers for the two required types
        assert set(alloc.keys()) == {sp.type_to_producer["capacity-1"], sp.type_to_producer["capacity-3"]}
        assert sum(alloc.values()) == 137
        assert all(v >= 0 for v in alloc.values())


def test_decode_empty_incoming():
    sp = spec()
    action = np.zeros(sp.ordering_action_dim, dtype=np.float32)
    cmd = decode_ordering(sp, action, {})
    assert cmd.orders == {}


def test_decode_zero_budget():
    sp = spec()
    incoming = {"o9": OrderInfo("o9", 0, {"capacity-2": 180, "capacity-4": 20})}
    cmd = decode_ordering(sp, np.zeros(sp.ordering_action_dim, dtype=np.float32), incoming)
    assert sum(cmd.orders["o9"].values()) == 0
