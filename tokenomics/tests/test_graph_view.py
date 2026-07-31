"""EXP-0039 graph_primitives_v1 domain view: pure relational decision-time facts.

The domain owns "what a routing decision looks like as a graph" (mirroring
OrdererEconomyView, EXP-0006A): typed node/relation primitives assembled from live
system state, no numpy/torch/serialization/labels. These tests drive a real blend_0.5
trajectory to a contested ordering decision and check the view exposes the schema the
ARCH.md graph_primitives_v1 contract requires.

Kept tiny (one depth-3 scale task, short horizon, seconds) per the compute rule — the
real s29/s77/s45 dump runs on a pod.
"""

from __future__ import annotations

from tokenomics.domain.graph_view import build_economy_graph_view
from tokenomics.eval.baselines import PRODUCING_BASELINES, ordering_blend
from tokenomics.eval.curriculum import TaskRecord, replay
from tokenomics.eval.joint_stack import joint_stack_env


def _scale_task(seed: int = 0, depth: int = 3) -> TaskRecord:
    return TaskRecord(
        task_id=f"scale-d{depth}-s{seed}", task_family="scale",
        generator_version="test", serializer_version="test",
        world_seed=seed, episode_seed=0,
        world_spec={"generator": "build_scale_config", "params": {"depth": depth}},
    )


def first_contested_decision(depth: int = 3, horizon: int = 12, seed: int = 0):
    """Drive blend_0.5 to the first ordering decision with >= 2 eligible producers and
    return `(env, orderer_id, order_id, eligible_ids)` at the acting agent's turn — the
    exact pre-action state EXP-0038 labels. Mirrors collect_error_records' traversal."""
    cfg = replay(_scale_task(seed, depth))
    prod = PRODUCING_BASELINES["stock_aware"]
    base = ordering_blend(0.5)
    env = joint_stack_env(cfg, horizon)
    env.reset(seed=seed)
    for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        role, aid = env._role[agent]
        if role == "prod":
            env.step(prod(env, agent))
            continue
        view = env.system.ordering_agents[aid].view(env.system.producer_infos)
        for k, oid in enumerate(sorted(view.incoming)):
            if k >= env.spec_.k_stream:
                break
            elig = view.eligible[oid]
            if len(elig) >= 2:
                return env, aid, oid, list(elig)
        env.step(base(env, agent))
    raise AssertionError("no contested ordering decision found in the smoke trajectory")


def test_view_emits_every_node_type_with_primitive_features():
    """Every configured producer, product, capacity type, and product-stream orderer is a
    node, and every live order in system.orders appears exactly once. Node features are the
    ARCH.md primitives — the current order carries its lifecycle accounting and a consumer
    kind one-hot; a producer carries funds + running flags + a cut-off-defined mask."""
    env, aid, oid, elig = first_contested_decision()
    sys = env.system
    view = build_economy_graph_view(sys, aid, oid, elig)

    assert {n.producer_id for n in view.producers} == set(sys.producing_agents)
    assert {n.product for n in view.products} == set(sys.process_sheets)
    owned_types = {t for p in sys.producing_agents.values() for t in p.capacities}
    assert {n.capacity_type for n in view.capacity_types} == owned_types
    assert {n.orderer_id for n in view.orderers} == set(sys.stream_products)

    order_nodes = {n.order_id: n for n in view.orders}
    assert set(order_nodes) == set(sys.orders)  # every live order, exactly once
    o = sys.orders[oid]
    n = order_nodes[oid]
    assert n.original_quantity == o.quantity
    assert n.unresolved_quantity == o.unresolved_quantity()
    assert n.available_tokens == o.available_tokens()
    assert n.ttl_remaining == max(0, 2 - o.cycle_counter)
    assert n.order_kind == (1.0, 0.0, 0.0)  # the contested order is a consumer order

    p0 = next(n for n in view.producers if n.producer_id == elig[0])
    prod0 = sys.producing_agents[elig[0]]
    assert p0.funds == prod0.funds
    assert p0.inventory_limit == prod0.inventory_limit
    assert p0.has_inventory_expansion == (prod0.inventory_expansion is not None)
    assert isinstance(p0.cutoff_is_defined, float) and p0.cutoff_is_defined in (0.0, 1.0)


def test_view_emits_static_recipe_eligibility_relations_and_anchors():
    """The static topology relations reproduce the config: recipe capacity/input
    requirements (with per-unit attributes), authoritative producer eligibility, and
    stream/source ownership. The decision anchors resolve to the acting order and orderer,
    and each order `requests` its product / is `owned` by its stream orderer."""
    env, aid, oid, elig = first_contested_decision()
    sys = env.system
    view = build_economy_graph_view(sys, aid, oid, elig)

    # recipe capacity requirement: product -> capacity_type, attr = units per output unit
    got_cap = {(e.product, e.capacity_type): e.units for e in view.requires_capacity}
    want_cap = {(x, t): c for x, ps in sys.process_sheets.items()
                for t, c in ps.require.items()}
    assert got_cap == want_cap
    got_inp = {(e.product, e.input_product): e.units for e in view.requires_input}
    want_inp = {(x, i): n for x, ps in sys.process_sheets.items()
                for i, n in ps.inputs.items()}
    assert got_inp == want_inp

    got_make = {(e.producer_id, e.product) for e in view.can_make}
    want_make = {(pid, x) for pid, p in sys.producing_agents.items() for x in p.produces}
    assert got_make == want_make

    assert view.current_order.order_id == oid
    assert view.current_orderer.orderer_id == aid
    o = sys.orders[oid]
    assert (o.id, o.product) in {(e.order_id, e.product) for e in view.requests}
    assert (aid, oid) in {(e.orderer_id, e.order_id) for e in view.owns}
    assert (aid, sys.stream_products[aid]) in {
        (e.orderer_id, e.product) for e in view.stream_for}


def test_producer_cutoff_mask_tracks_undefined_price():
    """UNDEFINED_PRICE is NaN, so `cutoff_is_defined` must be computed with isnan, never a
    `!= UNDEFINED_PRICE` comparison (NaN compares unequal to everything). An undefined cut-off
    reads as masked (0.0) with a zeroed price; a defined one reads 1.0 with the real price."""
    import math

    env, aid, oid, elig = first_contested_decision()
    sys = env.system
    view = build_economy_graph_view(sys, aid, oid, elig)
    for n in view.producers:
        undefined = math.isnan(sys.producing_agents[n.producer_id].cut_off_price)
        assert n.cutoff_is_defined == (0.0 if undefined else 1.0)
        if undefined:
            assert n.cut_off_price == 0.0
        assert not math.isnan(n.cut_off_price)


def test_building_the_view_does_not_mutate_the_system():
    """Causality (ARCH.md): extraction is a pure read. Building the graph must not perturb any
    live domain state, so the graph is a faithful pre-action snapshot and a future labeler
    change cannot leak backwards through a shared mutation."""
    import copy

    env, aid, oid, elig = first_contested_decision()
    sys = env.system
    before = copy.deepcopy(sys.orders), {
        pid: (dict(p.capacities), list(p.bids), list(p.reservations))
        for pid, p in sys.producing_agents.items()}
    build_economy_graph_view(sys, aid, oid, elig)
    after = sys.orders, {
        pid: (dict(p.capacities), list(p.bids), list(p.reservations))
        for pid, p in sys.producing_agents.items()}
    assert {k: (v.quantity, v.accepted_qty, v.completed_qty) for k, v in before[0].items()} == \
           {k: (v.quantity, v.accepted_qty, v.completed_qty) for k, v in after[0].items()}
    assert before[1] == after[1]


def test_view_emits_structural_zero_inventory_and_telemetry_relations():
    """Absence of a relation must mean "not applicable", not "currently zero" (ARCH.md). So
    every producer gets an `inventory_state` edge for each storable input product of what it
    can make, and a `telemetry_state` edge for every can_make pair — present even when all
    values are zero. Commitment edges reconstruct the live pending-bid / reservation state."""
    env, aid, oid, elig = first_contested_decision()
    sys = env.system
    view = build_economy_graph_view(sys, aid, oid, elig)

    inv = {(e.producer_id, e.input_product) for e in view.inventory_state}
    want_inv = {(pid, i) for pid, p in sys.producing_agents.items()
                for made in p.produces for i in sys.process_sheets[made].inputs}
    assert inv == want_inv  # structural: every storable input, zero or not

    tel = {(e.producer_id, e.product) for e in view.telemetry_state}
    want_tel = {(pid, x) for pid, p in sys.producing_agents.items() for x in p.produces}
    assert tel == want_tel  # structural: every can_make pair, zero or not

    # commitments reconstruct live domain state (empty or not, the mapping must be exact)
    got_bids = {(e.order_id, e.producer_id, e.quantity, e.tokens) for e in view.pending_bid}
    want_bids = {(b.order_id, pid, b.quantity, b.tokens)
                 for pid, p in sys.producing_agents.items() for b in p.bids}
    assert got_bids == want_bids
    got_res = {(e.order_id, e.producer_id, e.quantity) for e in view.reservation}
    want_res = {(r.bid.order_id, pid, r.bid.quantity)
                for pid, p in sys.producing_agents.items() for r in p.reservations}
    assert got_res == want_res


def test_view_reconstructs_injected_pending_bid_and_reservation():
    """A non-empty commitment case: inject one pending bid and one live reservation onto a
    producer and confirm the view surfaces them as `order -> producer` edges carrying the
    routed quantity/tokens and, for the reservation, cycles waited — the causal commitment
    state the teacher's rollout can depend on (ARCH.md)."""
    from tokenomics.domain.producing_agent import Bid, _Reservation

    env, aid, oid, elig = first_contested_decision()
    sys = env.system
    p = sys.producing_agents[elig[0]]
    prod = sys.orders[oid].product
    bid = Bid(capacities={t: c for t, c in sys.process_sheets[prod].require.items()},
              tokens=7, order_id=oid, product=prod, quantity=3, routed_cycle=sys.cycle_counter)
    p.bids.append(bid)
    p.reservations.append(_Reservation(bid=Bid(
        capacities={}, tokens=5, order_id=oid, product=prod, quantity=2), cycles_waited=1))

    view = build_economy_graph_view(sys, aid, oid, elig)
    pb = next(e for e in view.pending_bid if e.producer_id == elig[0] and e.order_id == oid)
    assert pb.quantity == 3 and pb.tokens == 7 and pb.routed_age >= 0
    rv = next(e for e in view.reservation if e.producer_id == elig[0] and e.order_id == oid)
    assert rv.quantity == 2 and rv.tokens == 5 and rv.cycles_waited == 1


def test_view_emits_decision_candidate_and_capacity_state():
    """Tracer bullet: at a contested decision the view has exactly one `decision` anchor,
    one `candidate` edge per eligible producer (aligned to the legacy eligible set), and a
    `capacity_state` edge for every capacity type each producer owns — the minimal
    relational skeleton the GNN scores."""
    env, aid, oid, elig = first_contested_decision()
    view = build_economy_graph_view(env.system, aid, oid, elig)

    assert view.decision is not None
    cand_producers = [e.producer_id for e in view.candidate]
    assert cand_producers == list(elig)  # one candidate edge per eligible producer, aligned

    # every producer owns >= 1 capacity type -> a capacity_state edge per owned type
    for pid, p in env.system.producing_agents.items():
        owned = set(p.capacities)
        got = {e.capacity_type for e in view.capacity_state if e.producer_id == pid}
        assert got == owned
