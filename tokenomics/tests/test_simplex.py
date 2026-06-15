"""Env spaces + decode (ADR-0025/0026).

Covers `build_spec` (multi-capacity producers, eligible-producer ordering, the
global input-product vocabulary) and the two decoders:
  * decode_ordering — per-order quantity split over *eligible* producers, tokens
    following the realized integer quantities, zero-quantity sub-orders dropped.
  * decode_producing — joint replenishment over shared inventory headroom +
    investment toggles, with decode owning all gating.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenomics.domain.config import load_config
from tokenomics.domain.common import UNDEFINED_PRICE
from tokenomics.domain.order import OrderInfo
from tokenomics.domain.ordering_agent import OrderingAgentView
from tokenomics.domain.producing_agent import ProducerInfo
from tokenomics.env.decode import _allocate_int, decode_ordering
from tokenomics.env.spaces import build_spec

COMPOSITE = Path(__file__).resolve().parent / "fixtures" / "composite.json"


def spec():
    return build_spec(load_config(COMPOSITE))


def _pinfo(pid: str) -> ProducerInfo:
    # decode_ordering only reads the eligible producer *ids*; contents are unused.
    return ProducerInfo(pid, {}, {}, UNDEFINED_PRICE)


def _view(incoming: dict, eligible_ids: dict) -> OrderingAgentView:
    eligible = {oid: {pid: _pinfo(pid) for pid in pids} for oid, pids in eligible_ids.items()}
    return OrderingAgentView(incoming, eligible)


# --- spec: multi-capacity model ---
def test_spec_capacity_types_union_over_producers():
    sp = spec()
    # composite.json: p1..p4 own capacity-1..4 respectively
    assert sp.capacity_types == ["capacity-1", "capacity-2", "capacity-3", "capacity-4"]
    assert sp.num_types == 4
    assert sp.num_producers == 4


def test_spec_input_product_vocabulary():
    sp = spec()
    # composite recipes: product 20 inputs {10}; product 30 inputs {20, 11}
    assert sp.input_products == [10, 11, 20]
    assert sp.num_inputs == 3


def test_spec_action_dims():
    sp = spec()
    # ordering: K slots x P producers; producing: G replenish + 3 toggles
    assert sp.ordering_action_dim == sp.k * sp.num_producers
    assert sp.producing_action_dim == sp.num_inputs + 3


# --- decode_ordering: quantity split over eligible producers ---
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


def _oinfo(oid, tokens, product, quantity, unit_capacity):
    return OrderInfo(oid, tokens, product, quantity, unit_capacity, {})


def test_decode_conservation_quantity_and_tokens():
    sp = spec()
    rng = np.random.default_rng(1)
    # quantity 7, tokens 137, two eligible producers
    incoming = {"o1": _oinfo("o1", 137, 30, 7, {"capacity-4": 5})}
    eligible = {"o1": ["p3", "p4"]}
    for _ in range(200):
        action = rng.uniform(-10, 10, size=sp.ordering_action_dim).astype(np.float32)
        cmd = decode_ordering(sp, action, _view(incoming, eligible))
        subs = cmd.orders["o1"]
        assert {s.producer_id for s in subs} <= {"p3", "p4"}  # only eligible
        assert all(s.quantity > 0 for s in subs)  # no zero-quantity sub-order
        assert sum(s.quantity for s in subs) == 7
        assert sum(s.tokens for s in subs) == 137


def test_decode_quantity_one_picks_highest_logit_producer():
    sp = spec()
    incoming = {"o1": _oinfo("o1", 100, 30, 1, {"capacity-4": 5})}
    eligible = {"o1": ["p3", "p4"]}
    action = np.zeros(sp.ordering_action_dim, dtype=np.float32)
    # producer_ids sorted == [p1,p2,p3,p4]; slot 0 row, push p4 logit high
    p4_idx = sp.producer_ids.index("p4")
    action[p4_idx] = 10.0
    cmd = decode_ordering(sp, action, _view(incoming, eligible))
    subs = cmd.orders["o1"]
    assert len(subs) == 1  # the single unit goes to one producer
    assert subs[0].producer_id == "p4"
    assert subs[0].quantity == 1
    assert subs[0].tokens == 100  # tokens follow the realized quantity


def test_decode_tokens_follow_realized_quantities():
    sp = spec()
    # quantity 5, tokens 100, two eligible; skew so split is 4/1 -> tokens 80/20
    incoming = {"o1": _oinfo("o1", 100, 30, 5, {"capacity-4": 5})}
    eligible = {"o1": ["p3", "p4"]}
    action = np.zeros(sp.ordering_action_dim, dtype=np.float32)
    p3_idx = sp.producer_ids.index("p3")
    p4_idx = sp.producer_ids.index("p4")
    action[p3_idx] = np.log(4.0)
    action[p4_idx] = np.log(1.0)
    cmd = decode_ordering(sp, action, _view(incoming, eligible))
    by_pid = {s.producer_id: s for s in cmd.orders["o1"]}
    assert by_pid["p3"].quantity == 4 and by_pid["p4"].quantity == 1
    assert by_pid["p3"].tokens == 80 and by_pid["p4"].tokens == 20


def test_decode_empty_incoming():
    sp = spec()
    action = np.zeros(sp.ordering_action_dim, dtype=np.float32)
    cmd = decode_ordering(sp, action, _view({}, {}))
    assert cmd.orders == {}


# --- decode_producing: joint replenishment over headroom + investment toggles ---
from tokenomics.env.decode import decode_producing
from tokenomics.env.spaces import EnvSpec
from tokenomics.domain.producing_agent import ProducingAgentView


def _pspec(input_products, producer_inputs, max_replenish):
    idx = {p: i for i, p in enumerate(input_products)}
    pids = list(producer_inputs)
    return EnvSpec(
        producer_ids=pids, capacity_types=["c"], type_index={"c": 0},
        input_products=input_products, input_index=idx,
        producer_owns={p: {"c"} for p in pids}, producer_inputs=producer_inputs,
        max_replenish=max_replenish, cap_ref=100.0, token_ref=1000.0, price_ref=1.0, inv_ref=50.0,
    )


def _pview(pid, headroom, restoration_running=False, upgrade_running=False,
           inventory_expansion_running=False, has_inventory_expansion=True):
    return ProducingAgentView(
        id=pid, capacities={"c": 50}, max_capacities={"c": 100}, requested_capacity=0,
        degradation={"c": 1}, upgrade=0, restoration=0,
        upgrade_running=upgrade_running, restoration_running=restoration_running,
        inventory={}, inventory_limit=50, inventory_units=0, headroom=headroom,
        inventory_expansion_running=inventory_expansion_running,
        has_inventory_expansion=has_inventory_expansion,
    )


def test_decode_producing_zeros_is_noop():
    sp = _pspec([1, 2], {"p": [1, 2]}, {"p": 30})
    cmd = decode_producing(sp, np.zeros(sp.producing_action_dim), _pview("p", 50))
    assert cmd.replenishments == {}
    assert not (cmd.do_restoration or cmd.do_upgrade or cmd.do_inventory_expansion)


def test_decode_producing_replenishes_desired_within_headroom():
    sp = _pspec([1, 2], {"p": [1, 2]}, {"p": 30})
    act = np.zeros(sp.producing_action_dim)
    act[sp.input_index[1]] = 10.0  # full -> max_replenish = 30
    cmd = decode_producing(sp, act, _pview("p", headroom=50))
    assert cmd.replenishments == {1: 30}


def test_decode_producing_irrelevant_dims_are_noops():
    # producer with no relevant inputs ignores all replenish dims
    sp = _pspec([1, 2], {"p": []}, {"p": 30})
    act = np.full(sp.producing_action_dim, 10.0)
    cmd = decode_producing(sp, act, _pview("p", headroom=50))
    assert cmd.replenishments == {}


def test_decode_producing_scales_to_exactly_headroom():
    sp = _pspec([1, 2], {"p": [1, 2]}, {"p": 30})
    act = np.zeros(sp.producing_action_dim)
    act[sp.input_index[1]] = 10.0  # desired 30
    act[sp.input_index[2]] = 10.0  # desired 30 -> total 60 > H
    cmd = decode_producing(sp, act, _pview("p", headroom=10))
    assert sum(cmd.replenishments.values()) == 10


def test_decode_producing_toggle_gating():
    sp = _pspec([1], {"p": [1]}, {"p": 30})
    act = np.zeros(sp.producing_action_dim)
    act[sp.num_inputs + 0] = 5.0  # restoration
    act[sp.num_inputs + 1] = 5.0  # upgrade
    act[sp.num_inputs + 2] = 5.0  # inventory expansion
    # restoration already running, and no inventory expansion configured -> both gated
    v = _pview("p", 50, restoration_running=True, has_inventory_expansion=False)
    cmd = decode_producing(sp, act, v)
    assert not cmd.do_restoration
    assert cmd.do_upgrade
    assert not cmd.do_inventory_expansion
