"""Diagnostic rejection / failure reasons and the routing tracer (add-m3-development-audit
task 1.1, design D2 "Telemetry"; development-audit "Rejection reasons are traceable").

A rejected bid is `outage`, `no_fit` (does not fit the budget available when the auction
starts) or `displaced` (fits it, but not what higher-priced bids left). A failed reservation is
`dropped` (lost re-validation) or `ttl` (expired waiting for inputs). The tracer only reads, so
an episode runs identically with and without it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tokenomics.domain.config import ProcessSheet, ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.producing_agent import (FAIL_DROPPED, FAIL_TTL, REJECT_DISPLACED,
                                               REJECT_NO_FIT, REJECT_OUTAGE, ProducingAgent)
from tokenomics.domain.producing_agent import SubOrder as RoutedSubOrder
from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.routing_trace import RoutingTracer, herfindahl
from tokenomics.eval.runner import run_episode
from tokenomics.eval.shock_profiles import SHOCKS_I_SMOKE

PRIMAL, COMPOSITE, CAP = 1, 2, "c"
SHEETS = {
    PRIMAL: ProcessSheet(PRIMAL, {CAP: 2}),
    COMPOSITE: ProcessSheet(COMPOSITE, {CAP: 3}, inputs={PRIMAL: 1}),
}
HORIZON = 12
SHOCKED_WORLD = "dev-sparse_wide-12-shocked-00"


def _producer(capacity: int = 10, degradation: int = 1) -> ProducingAgent:
    return ProducingAgent(ProducingAgentConfig(
        id="p", capacities={CAP: capacity}, degradation=degradation,
        restoration=Restoration(require=PRIMAL, restores=10, type=CAP),
        upgrade=Upgrade(require=PRIMAL, increases=10, type=CAP),
        produces=frozenset({PRIMAL, COMPOSITE}), inventory_limit=10,
    ), dict(SHEETS))


def _reasons(result: object, field: str) -> dict[str, str]:
    bids = result.rejected if field == "rejected" else result.failed  # type: ignore[attr-defined]
    reasons = result.rejection_reasons if field == "rejected" else result.failure_reasons  # type: ignore[attr-defined]
    assert len(bids) == len(reasons)
    return {b.order_id: r for b, r in zip(bids, reasons)}


def test_bid_displaced_by_a_higher_priced_bid() -> None:
    """Scenario "Bid displaced by a higher-priced bid": 10 units, two 6-unit bids (3 x primal);
    the dearer clears, the cheaper fit the auction-start budget but not the remainder."""
    p = _producer()
    p.place_sub_orders([RoutedSubOrder("rich", PRIMAL, 3, 90), RoutedSubOrder("poor", PRIMAL, 3, 30)])
    r = p.produce()
    assert [b.order_id for b in r.accepted] == ["rich"]
    assert _reasons(r, "rejected") == {"poor": REJECT_DISPLACED}


def test_bid_too_large_for_the_producer() -> None:
    """Scenario "Bid too large for the producer": a 12-unit bundle never fits 10 units."""
    p = _producer()
    p.place_sub_orders([RoutedSubOrder("big", PRIMAL, 6, 90), RoutedSubOrder("small", PRIMAL, 1, 10)])
    r = p.produce()
    assert [b.order_id for b in r.accepted] == ["small"]
    assert _reasons(r, "rejected") == {"big": REJECT_NO_FIT}


def test_no_fit_counts_capacity_held_by_reservations() -> None:
    """The auction-start budget excludes capacity reserved in earlier cycles: a bid that fits
    the raw capacity but not capacity minus reservations is `no_fit`, not `displaced`."""
    p = _producer(capacity=10, degradation=0)
    p.place_sub_orders([RoutedSubOrder("res", COMPOSITE, 2, 60)], routed_cycle=1)  # holds 6
    assert [b.order_id for b in p.produce().accepted] == ["res"]
    p.place_sub_orders([RoutedSubOrder("late", PRIMAL, 3, 90)])  # needs 6, only 4 free
    r = p.produce()
    assert _reasons(r, "rejected") == {"late": REJECT_NO_FIT}


def test_outage_rejection() -> None:
    p = _producer()
    p.place_sub_orders([RoutedSubOrder("o1", PRIMAL, 1, 50), RoutedSubOrder("o2", PRIMAL, 9, 10)])
    r = p.produce(outage=True)
    assert _reasons(r, "rejected") == {"o1": REJECT_OUTAGE, "o2": REJECT_OUTAGE}


def test_revalidation_drop() -> None:
    """A reservation whose capacity degraded away fails as `dropped`."""
    p = _producer(capacity=4, degradation=50)  # 4 -> 2 (50% of max) after reserving; needs 3
    p.place_sub_orders([RoutedSubOrder("o1", COMPOSITE, 1, 30)], routed_cycle=1)
    assert [b.order_id for b in p.produce().accepted] == ["o1"]
    r = p.produce()
    assert _reasons(r, "failed") == {"o1": FAIL_DROPPED}


def test_ttl_failure() -> None:
    """A reservation whose inputs never arrive fails at its reservation TTL as `ttl`."""
    p = _producer(capacity=10, degradation=0)
    p.place_sub_orders([RoutedSubOrder("o1", COMPOSITE, 1, 30)], routed_cycle=1)
    p.produce()
    assert not p.produce().failed  # waited one cycle
    r = p.produce()
    assert _reasons(r, "failed") == {"o1": FAIL_TTL}


def test_herfindahl() -> None:
    assert herfindahl([4]) == 1.0 and herfindahl([2, 2]) == 0.5 and herfindahl([]) == 0.0


def _shocked_env() -> TokenomicsAECEnv:
    entry = load_registry(REGISTRY_V2).entry(SHOCKED_WORLD)
    params = {**entry.generator["params"], "shock_profile": SHOCKS_I_SMOKE}
    entry = dataclasses.replace(entry, generator={**entry.generator, "params": params})
    return joint_stack_env(entry.config(), HORIZON)


def _run(tracer: RoutingTracer | None) -> tuple[dict, list[np.ndarray]]:
    env = _shocked_env()
    if tracer is not None:
        tracer.attach(env)
    actions: list[np.ndarray] = []
    prod, order = PRODUCING_BASELINES["entity_planner"], ORDERING_BASELINES["proportional"]

    def rec_prod(e: TokenomicsAECEnv, a: str) -> np.ndarray:
        actions.append(np.asarray(prod(e, a)).copy())
        return actions[-1]

    def rec_order(e: TokenomicsAECEnv, a: str) -> np.ndarray:
        actions.append(np.asarray(order(e, a)).copy())
        return actions[-1]

    seed = load_registry(REGISTRY_V2).entry(SHOCKED_WORLD).episode_seeds[0]
    return run_episode(env, rec_prod, rec_order, seed=seed), actions


def test_tracing_changes_no_statistic() -> None:
    """Scenario "Tracing changes no statistic": identical actions and episode statistics."""
    plain_stats, plain_actions = _run(None)
    tracer = RoutingTracer()
    traced_stats, traced_actions = _run(tracer)
    assert plain_stats["shock_events"] > 0  # the smoke profile puts shocks inside horizon 12
    assert traced_stats == plain_stats
    assert len(traced_actions) == len(plain_actions)
    assert all(np.array_equal(a, b) for a, b in zip(plain_actions, traced_actions))

    s = tracer.summary()
    assert tracer._c.cycles == plain_stats["cycles"]
    assert s["split_degree_consumer"] >= 1.0  # consumer orders were routed
    assert 0.0 <= s["herfindahl_mean"] <= 1.0
    assert 0.0 <= s["zero_throughput_route_share"] <= 1.0
    assert all(v >= 0.0 for v in s.values())
    series = tracer.series()
    assert len(series["capacity_ratio_trace"]) == plain_stats["cycles"]
    assert s["investment_rejection_rate"] == (plain_stats["investment_rejection_rate"] or 0.0)


def test_orphaned_reservation_is_a_ttl_failure_and_outcomes_align() -> None:
    """System level (phase1, c1 orders product 30 from p4, which never gets its inputs). An
    outage at cycle 2 rejects o2; o2 is re-routed and reserves at cycle 3; at cycle 4 its order's
    TTL fires while the reservation still waits, so the reservation is surfaced as failed with
    reason `ttl` by the system, not by `produce()`. Every cycle, per producer, the bid outcomes
    match `produce()`'s lists one for one, plus those orphans."""
    from pathlib import Path

    from tokenomics.domain import build_system
    from tokenomics.domain.config import load_config
    from tokenomics.domain.ordering_agent import OrderingAgentCommand, SubOrder
    from tokenomics.domain.producing_agent import ProducingAgentCommand, ProductionResult
    from tokenomics.domain.shocks import ShockEvent, ShockKind, ShockTarget

    phase1 = Path(__file__).resolve().parents[1] / "tokenomics" / "config" / "phase1.json"
    outage = ShockEvent(ShockKind.PRODUCER_OUTAGE, ShockTarget(producer="p4"), 2, 1, 1.0)
    system = build_system(load_config(phase1), shock_schedule=(outage,))
    produced: dict[str, ProductionResult] = {}
    for pid, p in system.producing_agents.items():
        def wrapped(outage: bool = False, _p=p, _pid=pid) -> ProductionResult:  # noqa: ANN001
            produced[_pid] = type(_p).produce(_p, outage)
            return produced[_pid]
        p.produce = wrapped  # type: ignore[method-assign]

    orphans: dict[int, list[tuple[str, str | None]]] = {}
    for cycle in range(1, 7):
        for pid in system.producing_agents:
            system.producing_agent_action(pid, ProducingAgentCommand(False, False))
        system.start_ordering()
        for aid, agent in system.ordering_agents.items():
            view = agent.view(system.producer_infos)
            orders = {oid: [SubOrder(next(iter(view.eligible[oid])), oi.quantity, oi.tokens)]
                      for oid, oi in view.incoming.items()}
            system.ordering_agent_action(aid, OrderingAgentCommand(orders))
        result = system.complete_cycle()
        orphans[cycle] = []
        for pid, r in produced.items():
            mine = [b for b in result.bid_outcomes if b.producer_id == pid]
            for outcome, bids in (("accepted", r.accepted), ("completed", r.completed),
                                  ("rejected", r.rejected)):
                assert [b.order_id for b in mine if b.outcome == outcome] == \
                    [b.order_id for b in bids], (cycle, pid, outcome)
            failed = [b for b in mine if b.outcome == "failed"]
            assert [(b.order_id, b.reason) for b in failed[:len(r.failed)]] == \
                list(zip((b.order_id for b in r.failed), r.failure_reasons)), (cycle, pid)
            orphans[cycle] += [(b.order_id, b.reason) for b in failed[len(r.failed):]]
        if cycle == 4:
            resolved = {o.order_id: o for o in result.order_resolutions}
            assert resolved["o2"].order_kind == "consumer" and not resolved["o2"].completed
    assert orphans[4] == [("o2", FAIL_TTL)]
    assert all(not orphans[c] for c in (1, 2, 3))


def test_tracer_metrics_exact_on_a_hand_fixture() -> None:
    """Exact values on a two-cycle hand fixture (stub system, hand-built cycle results). pa, pb
    and pz declare product 1 (2 capacity units per unit); only pa declares product 3. pb has
    exactly one unit of capacity free (not zero-throughput); pz has less than one (zero)."""
    from types import SimpleNamespace

    from tokenomics.domain.order import Order
    from tokenomics.domain.requests import ConsumerRequest, ReplenishmentRequest
    from tokenomics.domain.system import BidOutcome, CycleResult, OrderResolution

    sheets = {1: ProcessSheet(1, {CAP: 2}), 3: ProcessSheet(3, {CAP: 1})}

    def producer(pid: str, capacity: int, produces: set[int]) -> ProducingAgent:
        return ProducingAgent(ProducingAgentConfig(
            id=pid, capacities={CAP: capacity}, degradation=0,
            restoration=Restoration(require=1, restores=1, type=CAP),
            upgrade=Upgrade(require=1, increases=1, type=CAP),
            produces=frozenset(produces), inventory_limit=10,
        ), dict(sheets))

    agents = {"pa": producer("pa", 10, {1, 3}), "pb": producer("pb", 2, {1}),
              "pz": producer("pz", 1, {1})}

    def consumer_order(oid: str, product: int, qty: int) -> Order:
        return Order.new_consumer_order(oid, sheets[product], ConsumerRequest("c", product, 10, qty))

    orders = {oid: consumer_order(oid, x, q) for oid, x, q in (
        ("o_whole", 1, 2), ("o_split", 1, 2), ("o_q1", 1, 1), ("o_q1b", 1, 1),
        ("o_x", 3, 1), ("o_z", 1, 1))}
    rep = Order.new_replenishment_order("o_rep", sheets[1], ReplenishmentRequest("pb", 1, 1, 1.0))
    rep.fund(50)
    orders["o_rep"] = rep
    routes = {"pa": [("o_whole", 1, 2), ("o_split", 1, 1), ("o_q1", 1, 1), ("o_x", 3, 1)],
              "pb": [("o_split", 1, 1), ("o_q1b", 1, 1)], "pz": [("o_z", 1, 1)]}
    for pid, subs in routes.items():
        agents[pid].place_sub_orders([RoutedSubOrder(o, x, q, 10) for o, x, q in subs])

    c1 = CycleResult(score=0, bid_outcomes=(
        BidOutcome("pa", "o_whole", "consumer", 1, 2, 4, "accepted", None),
        BidOutcome("pb", "o_split", "consumer", 1, 1, 2, "rejected", REJECT_DISPLACED),
        BidOutcome("pz", "o_z", "consumer", 1, 1, 2, "rejected", REJECT_NO_FIT),
        BidOutcome("pa", "o_x", "consumer", 3, 1, 1, "failed", FAIL_TTL),  # pa in outage
    ), order_resolutions=(
        OrderResolution("o_whole", "consumer", 1, 2, True, 2),
        OrderResolution("o_split", "consumer", 1, 2, False, 1),  # wasted: 1 unit delivered
        OrderResolution("o_x", "consumer", 3, 1, False, 0),  # not wasted
        OrderResolution("o_inv", "investment", 5, 1, False, 0),  # never bid
        OrderResolution("o_inv2", "investment", 5, 1, True, 1),
    ))
    c2 = CycleResult(score=0, bid_outcomes=(
        BidOutcome("pz", "o_z", "consumer", 1, 1, 2, "rejected", REJECT_NO_FIT),  # a retry
    ), order_resolutions=(OrderResolution("o_z", "consumer", 1, 1, False, 0),))
    results = {1: c1, 2: c2}
    system = SimpleNamespace(
        cycle_counter=1, _outages_by_cycle={1: {"pa"}}, producing_agents=agents, orders=orders,
        ordering_agents={"a1": SimpleNamespace(incoming={"x": 0, "y": 0, "z": 0}),
                         "a2": SimpleNamespace(incoming={"w": 0})},
        investment_fund=200)
    system.complete_cycle = lambda: results[system.cycle_counter]

    tracer = RoutingTracer()
    tracer.attach_system(system, ["pa", "pb", "pz"], k_stream=2)  # type: ignore[arg-type]
    system.complete_cycle()
    # cycle 2: only o_z is re-bid; o_rep is no longer new (funded once, counted once)
    for p in agents.values():
        p.bids = []
    agents["pz"].place_sub_orders([RoutedSubOrder("o_z", 1, 1, 10)])
    system.orders = {"o_z": orders["o_z"], "o_rep": rep}
    rep.cycle_counter = 1
    system.ordering_agents = {"a1": SimpleNamespace(incoming={"x": 0})}
    system.cycle_counter = 2
    system.complete_cycle()

    s = tracer.summary()
    # split: cycle 1 six routed consumer orders over 7 producers, cycle 2 one over one
    assert s["split_share_consumer"] == 1 / 7 and s["split_degree_consumer"] == 8 / 7
    # Herfindahl per product x cycle; product 3 (one declarer) excluded
    assert s["herfindahl_mean"] == ((16 + 4 + 1) / 49 + 1.0) / 2
    # quantity-1 orders over 3 eligible: o_q1 -> pa (lowest), o_q1b -> pb, o_z -> pz twice
    assert s["q1_orders"] == 4 and s["q1_lowest_index_share"] == 0.25
    # 8 routes; pz twice below one unit, pb exactly at one unit (not zero-throughput)
    assert s["zero_throughput_route_share"] == 2 / 8
    assert s["overflow_orders"] == 1  # 3 incoming vs k_stream 2, once
    assert tracer.fund_shares() == {"pb": 0.25 / 2}
    # one cycle with grants (pb alone holds them: 1.0); cycle 2 grants nothing and is skipped
    assert s["fund_share_herfindahl"] == 1.0
    # bid-units count retries; true loss counts each resolved order once
    assert s["rejected_bid_units_consumer_no_fit"] == 2 and s["lost_units_consumer_no_fit"] == 1
    assert s["rejected_bid_units_consumer_displaced"] == 1 and s["lost_units_consumer_displaced"] == 1
    assert s["failed_bid_units_consumer_ttl_outage"] == 1 and s["failed_bid_units_consumer_ttl"] == 0
    assert s["lost_units_consumer_ttl_outage"] == 1
    assert s["lost_units_producer_never_bid"] == 1
    assert s["wasted_orders_consumer"] == 1 and s["wasted_units_consumer"] == 1
    assert s["investment_rejection_rate"] == 0.5


def _fund_tracer(cycles: list[dict[str, int]], fund: int = 200) -> RoutingTracer:
    """Trace a stub system whose producers are granted `cycles[t][pid]` tokens in cycle t + 1
    (one funded replenishment order each, new in that cycle)."""
    from types import SimpleNamespace

    from tokenomics.domain.order import Order
    from tokenomics.domain.requests import ReplenishmentRequest
    from tokenomics.domain.system import CycleResult

    sheets = {1: ProcessSheet(1, {CAP: 2})}

    def producer(pid: str) -> ProducingAgent:
        return ProducingAgent(ProducingAgentConfig(
            id=pid, capacities={CAP: 10}, degradation=0,
            restoration=Restoration(require=1, restores=1, type=CAP),
            upgrade=Upgrade(require=1, increases=1, type=CAP),
            produces=frozenset({1}), inventory_limit=10,
        ), dict(sheets))

    pids = sorted({p for grants in cycles for p in grants})
    system = SimpleNamespace(
        cycle_counter=1, _outages_by_cycle={}, producing_agents={p: producer(p) for p in pids},
        orders={}, ordering_agents={}, investment_fund=fund)
    system.complete_cycle = lambda: CycleResult(score=0)
    tracer = RoutingTracer()
    tracer.attach_system(system, pids, k_stream=2)  # type: ignore[arg-type]
    for t, grants in enumerate(cycles, start=1):
        orders = {}
        for pid, tokens in grants.items():
            o = Order.new_replenishment_order(f"o{t}{pid}", sheets[1],
                                              ReplenishmentRequest(pid, 1, 1, 1.0))
            o.fund(tokens)
            orders[f"o{t}{pid}"] = o
        system.orders = orders
        system.cycle_counter = t
        system.complete_cycle()
    return tracer


def test_fund_share_herfindahl_is_over_the_granted_fund() -> None:
    """Task 2.2: the per-cycle fund concentration is a Herfindahl over each producer's share of
    what was granted in a cycle, averaged over cycles with grants. pa is granted 3 and pb 1
    tokens of a 200-token fund in cycle 1 -> 0.75^2 + 0.25^2 = 0.625; cycle 2 grants nothing and
    is ignored. The old metric (squared mean shares of the whole fund) gives
    (3/400)^2 + (1/400)^2 = 6.25e-5."""
    tracer = _fund_tracer([{"pa": 3, "pb": 1}, {}])
    s = tracer.summary()
    assert s["fund_share_herfindahl"] == 0.625
    assert s["fund_share_herfindahl_episode"] == 0.625  # one grant cycle: both coincide
    shares = tracer.fund_shares()  # the raw per-producer series is unchanged: whole-fund shares
    assert shares == {"pa": 3 / 200 / 2, "pb": 1 / 200 / 2}
    old = sum(v * v for v in shares.values())
    assert old == pytest.approx(6.25e-5) and old != s["fund_share_herfindahl"]


def test_fund_share_herfindahl_per_cycle_versus_episode() -> None:
    """Two grant cycles to different producers: each cycle has one grantee (per-cycle mean
    1.0), but over the episode the grants are split evenly (episode value 0.5 = 1/N): the
    per-cycle value tracks grantees per cycle, the episode value starvation (D2 (d))."""
    s = _fund_tracer([{"pa": 3}, {"pb": 3}]).summary()
    assert s["fund_share_herfindahl"] == 1.0
    assert s["fund_share_herfindahl_episode"] == 0.5
    starved = _fund_tracer([{"pa": 3}, {"pa": 3}, {"pb": 1}]).summary()
    assert starved["fund_share_herfindahl_episode"] == pytest.approx((6 / 7) ** 2 + (1 / 7) ** 2)
