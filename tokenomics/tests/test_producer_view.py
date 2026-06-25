"""EXP-0010A: ProducerEntityView — the domain's authoritative entity-factored
feature bundle for one producing agent's decision (ADR-0053). Mirrors
test_orderer_view.py: pin the domain facts directly (testable outside RL), plus the
ADR permutation gates on the env encoder.

Harness: drive ProducingAgent.produce() directly with synthetic bids (auction-level
pins), or build a System and read the view (faithfulness pins).
"""

from __future__ import annotations

from tokenomics.domain import build_system, load_config, metrics
from tokenomics.domain.config import (
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.domain.producing_agent import Bid, ProducingAgent
from tokenomics.env.aec_env import DEFAULT_CONFIG


def _system_at_decision():
    """build_system ends on start_cycle -> ORDERS_PLACEMENT, the producer decision
    point (where producing_agent_view/action live)."""
    config = load_config(DEFAULT_CONFIG)
    return build_system(config), config


def _make_producer(capacities: dict[str, int], degradation: int = 0) -> ProducingAgent:
    t = next(iter(capacities))
    cfg = ProducingAgentConfig(
        id="p",
        capacities=capacities,
        degradation=degradation,
        restoration=Restoration(require=1, restores=10, type=t),
        upgrade=Upgrade(require=1, increases=10, type=t),
    )
    return ProducingAgent(cfg)


def test_requested_capacity_by_type_sums_bid_capacities_per_type():
    """ReqPrev[p,t] (ADR-0053) = post-auction requested capacity per type = the sum
    of every bid's capacities[t] this cycle, accepted OR rejected, matching the way
    the scalar requested_capacity already aggregates."""
    p = _make_producer({"capacity-1": 10, "capacity-2": 10})
    fits = Bid({"capacity-1": 4, "capacity-2": 3}, tokens=70, order_id="fits")
    overflow = Bid({"capacity-1": 20, "capacity-2": 1}, tokens=210, order_id="over")
    p.place_bids([fits, overflow])
    p.produce()

    # both bids count toward requested pressure regardless of acceptance.
    assert p.requested_capacity_by_type == {"capacity-1": 24, "capacity-2": 4}
    # the scalar stays the total across types (back-compat).
    assert p.requested_capacity == sum(p.requested_capacity_by_type.values())


def test_capacity_demand_pressure_weights_open_qty_by_recipe_and_eligibility():
    """capacity_demand_pressure(p, t, open_qty) (ADR-0053) = sum_x eligible[p,x] *
    open_qty[x] * R[x,t] — the open-demand capacity load on type t, restricted to
    products this producer declares."""
    sheets = {
        10: ProcessSheet(product=10, require={"capacity-1": 2}),
        11: ProcessSheet(product=11, require={"capacity-1": 3, "capacity-2": 1}),
        12: ProcessSheet(product=12, require={"capacity-1": 5}),  # not declared
    }
    cfg = ProducingAgentConfig(
        id="p", capacities={"capacity-1": 100, "capacity-2": 100}, degradation=0,
        restoration=Restoration(1, 10, "capacity-1"), upgrade=Upgrade(1, 10, "capacity-1"),
        produces=frozenset({10, 11}),
    )
    p = ProducingAgent(cfg, sheets=sheets)
    open_qty = {10: 4, 11: 2, 12: 7}

    # cap-1: 4*2 (p10) + 2*3 (p11) = 14; product 12 excluded (not in produces).
    assert metrics.capacity_demand_pressure(p, "capacity-1", open_qty) == 14
    # cap-2: only p11 requires it: 2*1 = 2.
    assert metrics.capacity_demand_pressure(p, "capacity-2", open_qty) == 2


def test_restore_and_upgrade_effect_and_cost_are_per_targeted_type():
    """restore/upgrade effect+cost (ADR-0053 CapacityRow) are non-zero only on the
    type the investment targets; cost is 0 when unpriced (None)."""
    cfg = ProducingAgentConfig(
        id="p", capacities={"capacity-1": 50, "capacity-2": 50}, degradation=0,
        restoration=Restoration(require=1, restores=8, type="capacity-1", cost=30),
        upgrade=Upgrade(require=1, increases=5, type="capacity-2"),  # unpriced
        produces=frozenset(),
    )
    p = ProducingAgent(cfg)

    # restoration targets capacity-1 only.
    assert metrics.restore_effect(p, "capacity-1") == 8
    assert metrics.restore_cost(p, "capacity-1") == 30
    assert metrics.restore_effect(p, "capacity-2") == 0
    assert metrics.restore_cost(p, "capacity-2") == 0

    # upgrade targets capacity-2 only, and is unpriced (cost None -> 0).
    assert metrics.upgrade_effect(p, "capacity-2") == 5
    assert metrics.upgrade_cost(p, "capacity-2") == 0
    assert metrics.upgrade_effect(p, "capacity-1") == 0


def test_downstream_demand_pressure_weights_open_qty_by_input_recipe():
    """downstream_demand_pressure(p, g, open_qty) (ADR-0053 InputRow) = sum_x
    eligible[p,x] * open_qty[x] * I[x,g] — how many units of input g this producer's
    declared open demand pulls."""
    sheets = {
        30: ProcessSheet(product=30, require={"capacity-1": 1}, inputs={20: 1, 11: 2}),
        20: ProcessSheet(product=20, require={"capacity-1": 1}),
        11: ProcessSheet(product=11, require={"capacity-1": 1}),
    }
    cfg = ProducingAgentConfig(
        id="p", capacities={"capacity-1": 100}, degradation=0,
        restoration=Restoration(1, 10, "capacity-1"), upgrade=Upgrade(1, 10, "capacity-1"),
        produces=frozenset({30}),
    )
    p = ProducingAgent(cfg, sheets=sheets)
    open_qty = {30: 4}

    assert metrics.downstream_demand_pressure(p, 20, open_qty) == 4   # 4 * 1
    assert metrics.downstream_demand_pressure(p, 11, open_qty) == 8   # 4 * 2
    assert metrics.downstream_demand_pressure(p, 99, open_qty) == 0   # not an input


def test_reservation_qty_helpers_read_live_reservations():
    """active_reservation_qty(p, x) sums live-reservation quantity for product x;
    blocked_reservation_qty(p, g) sums the input-g units those waiting reservations
    still need (ADR-0053). A composite bid that clears with no input stock reserves."""
    sheets = {
        30: ProcessSheet(product=30, require={"capacity-1": 1}, inputs={20: 2}),
        20: ProcessSheet(product=20, require={"capacity-1": 1}),
    }
    cfg = ProducingAgentConfig(
        id="p", capacities={"capacity-1": 100}, degradation=0,
        restoration=Restoration(1, 10, "capacity-1"), upgrade=Upgrade(1, 10, "capacity-1"),
        produces=frozenset({30}), inventory_limit=100,
    )
    p = ProducingAgent(cfg, sheets=sheets)
    # composite bid for 3 units of 30; no stock of input 20 -> becomes a reservation.
    p.place_bids([Bid({"capacity-1": 3}, tokens=30, order_id="o", product=30,
                       inputs={20: 6}, quantity=3)])
    p.produce()
    assert len(p.reservations) == 1

    assert metrics.active_reservation_qty(p, 30) == 3
    assert metrics.active_reservation_qty(p, 99) == 0
    assert metrics.blocked_reservation_qty(p, 20) == 6   # 3 units * 2 input each
    assert metrics.blocked_reservation_qty(p, 99) == 0


def test_reserved_ratio_is_blocked_fraction_of_current_not_max_capacity():
    """reserved_ratio = Res[p,t] / C[p,t] divides by CURRENT usable capacity, not max
    (ADR-0053). A degraded producer (current < max) with a live reservation reports the
    true blocked fraction of what it can actually use, not an under-stated max fraction."""
    from tokenomics.domain.telemetry import EconomyTelemetry
    from tokenomics.domain.producer_view import build_producer_entity_view

    sheets = {30: ProcessSheet(product=30, require={"capacity-1": 25}, inputs={20: 1})}
    cfg = ProducingAgentConfig(
        id="p", capacities={"capacity-1": 50}, degradation=0,
        restoration=Restoration(1, 10, "capacity-1"), upgrade=Upgrade(1, 10, "capacity-1"),
        produces=frozenset(), inventory_limit=100,
    )
    p = ProducingAgent(cfg, sheets=sheets)
    p.max_capacities["capacity-1"] = 100  # degraded: current 50 < max 100
    # composite bid clears on capacity (25 <= 50) but holds as a reservation (no stock
    # of input 20), reserving 25 of current capacity.
    p.place_bids([Bid({"capacity-1": 25}, tokens=25, order_id="o", product=30,
                       inputs={20: 1}, quantity=1)])
    p.produce()

    row = build_producer_entity_view(p, EconomyTelemetry(), sheets, {}).capacity_rows[0]
    assert row.current == 50 and row.max == 100 and row.reserved == 25
    assert row.reserved_ratio == 25 / 50  # current, NOT 25/100


def _open_consumer_qty(orders) -> dict:
    out: dict = {}
    for o in orders.values():
        if o.consumer_request is not None:
            out[o.product] = out.get(o.product, 0) + o.unresolved_quantity()
    return out


def test_view_rows_cover_owned_types_products_and_inputs():
    """The view enumerates exactly the producer's owned capacity types, declared
    products (`produces`), and the input products those recipes consume — each as an
    ordered (sorted-id) row list (ADR-0053)."""
    system, _ = _system_at_decision()
    view = system.producer_entity_view("p4")  # p4 makes 30 = cap-4 + inputs {20,11}

    assert [r.capacity_type for r in view.capacity_rows] == ["capacity-4"]
    assert [r.product for r in view.product_rows] == [30]
    assert [r.input_product for r in view.input_rows] == [11, 20]  # sorted inputs of 30


def test_view_capacity_row_is_faithful_to_metrics_and_state():
    system, _ = _system_at_decision()
    p = system.producing_agents["p4"]
    open_qty = _open_consumer_qty(system.orders)
    view = system.producer_entity_view("p4")

    row = view.capacity_rows[0]
    assert row.capacity_type == "capacity-4"
    assert row.owned == 1.0
    assert row.current == p.capacities["capacity-4"] == 120
    assert row.max == p.max_capacities["capacity-4"] == 120
    assert row.degradation == p.degradation_for("capacity-4")  # ceil(5*120/100)=6
    assert row.reserved == 0
    assert row.free == 120
    assert row.previous_requested_capacity == 0  # no produce() yet this episode
    assert row.restore_effect == metrics.restore_effect(p, "capacity-4") == 60
    assert row.restore_cost == metrics.restore_cost(p, "capacity-4") == 40
    assert row.upgrade_effect == metrics.upgrade_effect(p, "capacity-4") == 40
    assert row.demand_pressure == metrics.capacity_demand_pressure(p, "capacity-4", open_qty)
    assert row.demand_pressure == 25  # 1 unit of 30 * 25 cap-4


def test_view_product_row_is_faithful_and_eligibility_gated():
    system, _ = _system_at_decision()
    p = system.producing_agents["p4"]
    open_qty = _open_consumer_qty(system.orders)
    q = open_qty.get(30, 0)
    view = system.producer_entity_view("p4")

    row = view.product_rows[0]
    assert row.product == 30
    assert row.eligible == 1.0
    assert row.open_demand_qty == 1
    assert row.product_depth == 2
    assert row.recipe_input_count == 2
    assert row.recipe_capacity_type_count == 1
    assert row.capacity_makeable_qty == metrics.capacity_makeable_qty(p, 30)  # 120//25=4
    assert row.complete_qty_now == metrics.complete_qty_now(p, 30)  # 0, no input stock
    assert row.capacity_bottleneck_ratio == metrics.capacity_bottleneck_ratio(p, 30, q)
    assert row.bottleneck_ratio == metrics.producer_product_bottleneck_ratio(p, 30, q)
    assert row.active_reservation_qty == 0


def test_view_input_row_is_faithful_to_state_and_pressure():
    system, _ = _system_at_decision()
    p = system.producing_agents["p4"]
    open_qty = _open_consumer_qty(system.orders)
    view = system.producer_entity_view("p4")

    by_g = {r.input_product: r for r in view.input_rows}
    assert by_g[20].inventory == p.inventory.get(20, 0) == 0
    assert by_g[20].outstanding_inbound == p.outstanding_inbound.get(20, 0) == 0
    assert by_g[20].downstream_demand_pressure == metrics.downstream_demand_pressure(p, 20, open_qty)
    assert by_g[20].downstream_demand_pressure == 1  # 1 unit of 30 pulls 1 of input 20
    assert by_g[11].downstream_demand_pressure == 1


def test_view_relations_indexed_by_row_position():
    """Recipe edges are keyed by (product_row_index, capacity/input_row_index) so they
    stay aligned when the encoder permutes rows (ADR-0053)."""
    system, _ = _system_at_decision()
    view = system.producer_entity_view("p4")

    # product 30 (row 0) requires 25 of capacity-4 (row 0).
    assert view.relations.product_capacity_required[(0, 0)] == 25
    # product 30 (row 0) uses 1 each of inputs 11 (row 0) and 20 (row 1).
    assert view.relations.product_input_required[(0, 0)] == 1
    assert view.relations.product_input_required[(0, 1)] == 1


# --- env layer: versioned, opt-in entity producer encoder (ADR-0053 §4-5) ---
def test_producer_entity_obs_dim_is_versioned_and_flat_unchanged():
    from tokenomics.env.spaces import build_spec

    cfg = load_config(DEFAULT_CONFIG)
    flat = build_spec(cfg)  # default producer_obs_version=1
    entity = build_spec(cfg, producer_obs_version=2)

    # the flat producer obs is untouched by the opt-in flag (default version=1).
    assert flat.producer_obs_version == 1
    assert entity.producer_obs_version == 2
    # the entity contract changes the producing obs dim (checkpoint-incompatible).
    assert entity.producing_obs_dim != flat.producing_obs_dim
    assert entity.producing_obs_dim == entity.producer_entity_obs_dim


def test_encode_producing_entity_matches_spec_dim():
    from tokenomics.env.encode import encode_producing_entity
    from tokenomics.env.spaces import OBS_CLIP, build_spec

    cfg = load_config(DEFAULT_CONFIG)
    spec = build_spec(cfg, producer_obs_version=2)
    system, _ = _system_at_decision()
    view = system.producer_entity_view("p4")

    obs = encode_producing_entity(spec, view, cycle_progress=0.3)
    assert obs.shape == (spec.producer_entity_obs_dim,)
    assert obs.dtype.name == "float32"
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= OBS_CLIP
    assert obs[-1] == 0.3  # cycle progress is the trailing scalar


def test_config_fingerprint_separates_producer_obs_versions():
    from tokenomics.env.spaces import config_fingerprint

    cfg = load_config(DEFAULT_CONFIG)
    assert config_fingerprint(cfg, producer_obs_version=1) != config_fingerprint(
        cfg, producer_obs_version=2
    )


# --- ADR-0053 permutation gates on the entity encoder ---
import copy as _copy

import numpy as _np


def _entity_blocks(spec, obs):
    """Slice an entity obs into (global, capacity-rows, product-rows, input-rows,
    cap-plane, input-plane) using the documented layout (mirror of the encoder)."""
    from tokenomics.env.spaces import (
        PRODUCER_CAP_ROW_FEATURES,
        PRODUCER_GLOBAL_FEATURES,
        PRODUCER_INPUT_ROW_FEATURES,
        PRODUCER_PRODUCT_ROW_FEATURES,
    )

    Q, T, G = spec.num_order_products, spec.num_types, spec.num_inputs
    i = 0
    glob = obs[i:i + PRODUCER_GLOBAL_FEATURES]; i += PRODUCER_GLOBAL_FEATURES
    cap = obs[i:i + T * PRODUCER_CAP_ROW_FEATURES].reshape(T, PRODUCER_CAP_ROW_FEATURES); i += T * PRODUCER_CAP_ROW_FEATURES
    prod = obs[i:i + Q * PRODUCER_PRODUCT_ROW_FEATURES].reshape(Q, PRODUCER_PRODUCT_ROW_FEATURES); i += Q * PRODUCER_PRODUCT_ROW_FEATURES
    inp = obs[i:i + G * PRODUCER_INPUT_ROW_FEATURES].reshape(G, PRODUCER_INPUT_ROW_FEATURES); i += G * PRODUCER_INPUT_ROW_FEATURES
    cap_plane = obs[i:i + Q * T].reshape(Q, T); i += Q * T
    in_plane = obs[i:i + Q * G].reshape(Q, G); i += Q * G
    return glob, cap, prod, inp, cap_plane, in_plane


def _rowset(a):
    return sorted(tuple(round(float(x), 6) for x in row) for row in a)


def test_encoder_is_equivariant_to_product_row_order():
    """Permuting the view's product rows permutes the encoded product-row slots and the
    relation planes identically, and leaves global/capacity blocks invariant — the
    encoder honors the view's row order, not EnvSpec's (ADR-0053)."""
    from tokenomics.env.encode import encode_producing_entity
    from tokenomics.env.spaces import build_spec

    cfg = load_config(DEFAULT_CONFIG)
    spec = build_spec(cfg, producer_obs_version=2)
    system, _ = _system_at_decision()
    # p4 makes 30 = cap-4 + inputs {11, 20}; permute its two input rows (the only
    # producer in this topology with >1 row of a kind to shuffle).
    view = system.producer_entity_view("p4")
    base = encode_producing_entity(spec, view, 0.5)

    # reverse the input-row order and rebuild input-relation columns accordingly.
    perm = list(reversed(range(len(view.input_rows))))  # new->old
    inv = {old: new for new, old in enumerate(perm)}
    pv = _copy.copy(view)
    pv.input_rows = [view.input_rows[perm[i]] for i in range(len(perm))]
    pv.relations = _copy.deepcopy(view.relations)
    pv.relations.product_input_required = {
        (xi, inv[gi]): n for (xi, gi), n in view.relations.product_input_required.items()
    }
    permuted = encode_producing_entity(spec, pv, 0.5)

    bg, bc, bp, bi, bcp, bip = _entity_blocks(spec, base)
    pg, pc, pp, pi, pcp, pip = _entity_blocks(spec, permuted)

    # global, capacity, product blocks and the cap-plane are untouched by an input perm.
    assert _np.array_equal(bg, pg)
    assert _np.array_equal(bc, pc)
    assert _np.array_equal(bp, pp)
    assert _np.array_equal(bcp, pcp)
    # input rows and the input-relation plane are permuted but carry the same row set.
    assert _rowset(bi) == _rowset(pi)
    assert _rowset(bip.T) == _rowset(pip.T)  # columns (per input) permuted


def _relabel(config, pm=None, cm=None):
    """Return a copy of `config` with product ids remapped by `pm` and capacity-type
    ids by `cm` (identity if absent). Consistent across every sheet/producer/consumer
    reference — a pure relabeling that preserves the economy's structure."""
    import dataclasses as _dc

    pm = pm or {}
    cm = cm or {}
    rp = lambda x: pm.get(x, x)
    rc = lambda t: cm.get(t, t)

    sheets = [
        _dc.replace(
            s, product=rp(s.product),
            require={rc(t): c for t, c in s.require.items()},
            inputs={rp(g): n for g, n in s.inputs.items()},
        )
        for s in config.process_sheets
    ]
    prods = []
    for p in config.producer_configs:
        p2 = _dc.replace(
            p,
            capacities={rc(t): c for t, c in p.capacities.items()},
            produces=frozenset(rp(x) for x in p.produces),
            restoration=_dc.replace(p.restoration, require=rp(p.restoration.require), type=rc(p.restoration.type)),
            upgrade=_dc.replace(p.upgrade, require=rp(p.upgrade.require), type=rc(p.upgrade.type)),
            inventory_expansion=(
                _dc.replace(p.inventory_expansion, require=rp(p.inventory_expansion.require))
                if p.inventory_expansion else None
            ),
        )
        prods.append(p2)
    consumers = [
        _dc.replace(c, product=rp(c.product), products=tuple(rp(x) for x in c.products))
        for c in config.consumers
    ]
    return _dc.replace(config, process_sheets=sheets, producer_configs=prods, consumers=consumers)


def _encoded_blocks_for(config, pid):
    from tokenomics.env.encode import encode_producing_entity
    from tokenomics.env.spaces import build_spec

    spec = build_spec(config, producer_obs_version=2)
    system = build_system(config)
    obs = encode_producing_entity(spec, system.producer_entity_view(pid), 0.5)
    return _entity_blocks(spec, obs)


def test_entity_obs_reflects_producer_action_mechanics():
    """Two configs differing ONLY in a producer's replenishment economics (ADR-0044
    action semantics) must NOT encode to the same entity producer obs. Otherwise an
    entity producer cannot tell apart decision problems whose replenish/expansion costs
    differ even though `decode_producing`/funding interpret the same action differently."""
    import dataclasses as _dc

    cfg = load_config(DEFAULT_CONFIG)
    prods = [
        _dc.replace(
            p,
            replenish_cost_per_unit=(p.replenish_cost_per_unit or 1) + 7,
            max_replenish=(p.max_replenish or p.inventory_limit) + 5,
        ) if p.id == "p4" else p
        for p in cfg.producer_configs
    ]
    bumped = _dc.replace(cfg, producer_configs=prods)

    bg, *_ = _encoded_blocks_for(cfg, "p4")
    rg, *_ = _encoded_blocks_for(bumped, "p4")
    assert not _np.array_equal(bg, rg)


def test_entity_global_block_carries_all_four_action_mechanics():
    """The global block's trailing 4 slots are the producer-global action economics
    (ADR-0044), normalized like the flat `_producer_mechanics` tail: max-replenish,
    replenish cost, inventory-expansion increases + cost."""
    from tokenomics.env.encode import encode_producing_entity
    from tokenomics.env.spaces import PRODUCER_GLOBAL_FEATURES, build_spec

    cfg = load_config(DEFAULT_CONFIG)
    spec = build_spec(cfg, producer_obs_version=2)
    system, _ = _system_at_decision()
    glob = encode_producing_entity(spec, system.producer_entity_view("p4"), 0.0)[
        :PRODUCER_GLOBAL_FEATURES
    ]
    m = spec.producer_mechanics["p4"]
    tail = glob[-4:]
    assert tail[0] == _np.float32(spec.max_replenish["p4"] / spec.inv_ref)
    assert tail[1] == _np.float32(m.replenish_cost_per_unit / spec.token_ref)
    assert tail[2] == _np.float32(m.inventory_expansion_increases / spec.inv_ref)
    assert tail[3] == _np.float32(m.inventory_expansion_cost / spec.token_ref)


def test_view_is_invariant_to_product_relabeling():
    """Relabeling product ids leaves the entity obs row features unchanged (same row
    set per block, same globals) — behavior depends on features, not ids (ADR-0053)."""
    cfg = load_config(DEFAULT_CONFIG)
    relabeled = _relabel(cfg, pm={10: 50, 11: 44, 20: 47, 30: 41})

    bg, bc, bp, bi, bcp, bip = _encoded_blocks_for(cfg, "p4")
    rg, rc, rp, ri, rcp, rip = _encoded_blocks_for(relabeled, "p4")

    assert _np.array_equal(bg, rg)
    assert _rowset(bc) == _rowset(rc)
    assert _rowset(bp) == _rowset(rp)
    assert _rowset(bi) == _rowset(ri)


def test_view_is_invariant_to_capacity_relabeling():
    """Same for capacity-type ids."""
    cfg = load_config(DEFAULT_CONFIG)
    relabeled = _relabel(
        cfg, cm={"capacity-1": "capacity-9", "capacity-2": "capacity-8",
                 "capacity-3": "capacity-7", "capacity-4": "capacity-6"},
    )

    bg, bc, bp, bi, bcp, bip = _encoded_blocks_for(cfg, "p4")
    rg, rc, rp, ri, rcp, rip = _encoded_blocks_for(relabeled, "p4")

    assert _np.array_equal(bg, rg)
    assert _rowset(bc) == _rowset(rc)
    assert _rowset(bp) == _rowset(rp)
    assert _rowset(bi) == _rowset(ri)


# --- env wiring: opt-in flag, parity, api (ADR-0053 §5) ---
def test_entity_producer_obs_flag_switches_observation_contract():
    from tokenomics.env import TokenomicsAECEnv

    flat = TokenomicsAECEnv(horizon=10)
    entity = TokenomicsAECEnv(horizon=10, entity_producer_obs=True)
    flat.reset(seed=0)
    entity.reset(seed=0)

    prod = flat._prod_agents[0]
    assert flat.observe(prod).shape == (flat.spec_.producing_obs_dim,)
    assert entity.observe(prod).shape == (entity.spec_.producer_entity_obs_dim,)
    assert entity.observe(prod).shape != flat.observe(prod).shape


def test_white_box_baseline_episode_is_byte_identical_under_entity_flag():
    """stock_aware reads env.system (not the obs), so flipping the entity-producer flag
    must leave the episode identical — the new obs contract is non-invasive (ADR-0053).
    v4 (a learned flat-v1 checkpoint reading env.observe) is out of scope -> EXP-0010B."""
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import run_episode
    from tokenomics.env import TokenomicsAECEnv

    prod = PRODUCING_BASELINES["stock_aware"]
    order = ORDERING_BASELINES["proportional"]
    flat = run_episode(TokenomicsAECEnv(horizon=40), prod, order, seed=0)
    entity = run_episode(
        TokenomicsAECEnv(horizon=40, entity_producer_obs=True), prod, order, seed=0
    )
    assert flat == entity


def test_pettingzoo_api_with_entity_producer_obs():
    from pettingzoo.test import api_test
    from tokenomics.env import TokenomicsAECEnv

    api_test(TokenomicsAECEnv(horizon=40, entity_producer_obs=True),
             num_cycles=20, verbose_progress=False)


def test_entity_view_recovers_producer_local_flat_facts():
    """Information-completeness (producer-local scope, ADR-0053 §6): every
    PRODUCER-LOCAL / action-relevant fact the flat `encode_producing` emits is
    recoverable from the entity view. The two flat GLOBALS are deliberately dropped
    (the anti-goal): all-order-product open demand (`_open_demand_features`) and the
    all-producer summary (`_producer_summary`, 2P) — the entity view keeps only THIS
    producer's products/demand, not flat ID-indexed cross-producer vectors."""
    system, _ = _system_at_decision()
    p = system.producing_agents["p4"]
    flat = p.view()  # the ProducingAgentView behind encode_producing
    view = system.producer_entity_view("p4")

    cap = {r.capacity_type: r for r in view.capacity_rows}
    # own per-type capacity / max / requested / degradation -> capacity rows.
    for t in p.capacities:
        assert cap[t].current == flat.capacities[t]
        assert cap[t].max == flat.max_capacities[t]
        assert cap[t].degradation == flat.degradation[t]
        assert cap[t].previous_requested_capacity == p.requested_capacity_by_type.get(t, 0)
    # investment-running flags -> globals.
    assert view.globals.active_investments == (
        1.0 if flat.restoration_running else 0.0,
        1.0 if flat.upgrade_running else 0.0,
        1.0 if flat.inventory_expansion_running else 0.0,
    )
    # inventory units / limit / headroom -> globals.
    assert view.globals.inventory_units == flat.inventory_units
    assert view.globals.inventory_limit == flat.inventory_limit
    assert view.globals.inventory_headroom == flat.headroom
    # own input inventory + outstanding inbound -> input rows.
    inp = {r.input_product: r for r in view.input_rows}
    for g in inp:
        assert inp[g].inventory == flat.inventory.get(g, 0)
        assert inp[g].outstanding_inbound == flat.outstanding_inbound.get(g, 0)
    # own-product open demand -> product rows (this producer's products only).
    prod = {r.product: r for r in view.product_rows}
    assert set(prod) == set(p.produces)


def test_eligible_guard_and_padded_rows_are_masked_zero():
    """Real product rows have eligible=1 and a mask of 1; padded rows are all-zero with
    mask 0 (mask, not slot, defines a real row — ADR-0053). The makeability eligible-guard
    still returns 0 for a non-`produces` product."""
    from tokenomics.env.encode import encode_producing_entity
    from tokenomics.env.spaces import build_spec

    cfg = load_config(DEFAULT_CONFIG)
    spec = build_spec(cfg, producer_obs_version=2)
    system, _ = _system_at_decision()
    view = system.producer_entity_view("p4")  # 1 declared product; Q=4 -> 3 padded rows
    obs = encode_producing_entity(spec, view, 0.0)
    _, _, prod, _, _, _ = _entity_blocks(spec, obs)

    assert prod.shape[0] == spec.num_order_products
    assert prod[0][0] == 1.0  # real row mask
    for padded in prod[len(view.product_rows):]:
        assert float(padded.sum()) == 0.0  # all-zero incl. mask 0

    # eligible-guard: metrics still return 0 for a product p4 does not declare.
    p = system.producing_agents["p4"]
    assert 10 not in p.produces
    assert metrics.capacity_makeable_qty(p, 10) == 0
    assert metrics.producer_product_bottleneck_ratio(p, 10, 5) == 0.0
