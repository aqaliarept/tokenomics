"""ADR-0061 (add-producer-acceptance-policy): order acceptance as a producer policy. The domain
executes accepted work when capacity allows (possibly later, before TTL); declines burn nothing;
the cut-off is the lowest accepted price; the default (no rule) and the auction-as-policy are
today's auction byte for byte; every sub-order's outcome is visible to its router next cycle."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tokenomics.domain.config import ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.producing_agent import (FAIL_TTL, REJECT_DECLINED, REJECT_NO_FIT,
                                               AcceptanceDecision, Bid, ProducingAgent)
from tokenomics.eval import acceptance as acc

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


def producer(cap: int = 10, ttl: int = 2) -> ProducingAgent:
    cfg = ProducingAgentConfig(id="p", capacities={"c": cap}, degradation=0,
                               restoration=Restoration(require=1, restores=10, type="c"),
                               upgrade=Upgrade(require=1, increases=10, type="c"))
    return ProducingAgent(cfg, reservation_ttl=ttl)


def bid(c: int, tokens: int, oid: str) -> Bid:
    return Bid(capacities={"c": c}, tokens=tokens, order_id=oid)


def accept_all(view):
    return [AcceptanceDecision(True) for _ in view.bids]


def test_an_accepted_bid_that_does_not_fit_executes_next_cycle_before_ttl() -> None:
    p = producer(10)
    p.acceptance = accept_all
    p.place_bids([bid(8, 80, "a"), bid(8, 40, "b")])  # 16 > 10: b cannot run this cycle
    r1 = p.produce()
    assert [b.order_id for b in r1.accepted] == ["a", "b"]  # both accepted: tokens burnt now
    assert [b.order_id for b in r1.completed] == ["a"] and r1.rejected == []
    assert [x.bid.order_id for x in p.pending] == ["b"]
    assert p.cut_off_price == bid(8, 40, "b").capacity_unit_price()  # lowest accepted price
    r2 = p.produce()  # capacity is a per-cycle budget: b runs now
    assert [b.order_id for b in r2.completed] == ["b"] and p.pending == []


def test_a_pending_claim_fails_at_ttl() -> None:
    p = producer(10, ttl=2)
    p.acceptance = accept_all
    p.place_bids([bid(12, 60, "big")])  # never fits
    p.produce()
    assert [x.bid.order_id for x in p.pending] == ["big"]
    r2 = p.produce()
    assert r2.failed == [] and len(p.pending) == 1
    r3 = p.produce()
    assert [b.order_id for b in r3.failed] == ["big"] and r3.failure_reasons == [FAIL_TTL]
    assert p.pending == [] and p.acceptance_stats["pending_failed_ttl"] == 1


def test_a_decline_burns_nothing_and_keeps_the_cut_off_when_nothing_is_accepted() -> None:
    p = producer(10)
    p.acceptance = lambda v: [AcceptanceDecision(False) for _ in v.bids]
    before = p.cut_off_price
    p.place_bids([bid(5, 50, "x")])
    r = p.produce()
    assert r.accepted == [] and [b.order_id for b in r.rejected] == ["x"]
    assert r.rejection_reasons == [REJECT_DECLINED]
    import math

    assert (math.isnan(before) and math.isnan(p.cut_off_price)) or p.cut_off_price == before
    assert p.funds == 0


def test_declines_below_the_cut_off_are_counted() -> None:
    p = producer(100)
    p.acceptance = lambda v: [AcceptanceDecision(b.order_id == "rich") for b in v.bids]
    p.place_bids([bid(5, 50, "rich"), bid(5, 10, "cheap")])
    p.produce()
    assert p.acceptance_stats["declined_below_cutoff"] == 1


@pytest.mark.parametrize("rule", ["auction"])
def test_the_auction_policy_equals_the_domain_auction_on_one_producer(rule: str) -> None:
    bids = [(8, 80, "a"), (5, 30, "b"), (3, 40, "c"), (4, 4, "d"), (13, 90, "huge")]
    plain, ruled = producer(12), producer(12)
    ruled.acceptance = acc.ACCEPTANCE_POLICIES[rule]
    for p in (plain, ruled):
        p.place_bids([bid(*x) for x in bids])
    a, b = plain.produce(), ruled.produce()
    for f in ("accepted", "completed", "failed", "rejected"):
        assert [x.order_id for x in getattr(a, f)] == [x.order_id for x in getattr(b, f)], f
    assert a.rejection_reasons == b.rejection_reasons and REJECT_NO_FIT in a.rejection_reasons
    assert plain.cut_off_price == ruled.cut_off_price and plain.funds == ruled.funds


def test_ttl_feasible_accepts_what_fits_next_cycle_and_declines_the_rest() -> None:
    p = producer(10)
    p.acceptance = acc.acceptance_ttl_feasible
    p.place_bids([bid(8, 80, "a"), bid(8, 40, "b"), bid(8, 20, "c")])  # now 10, next cycle 10
    r = p.produce()
    assert [x.order_id for x in r.accepted] == ["a", "b"]
    assert [x.order_id for x in r.rejected] == ["c"]


def _rows(acceptance: str | None, world: str, n: int = 2) -> list[dict]:
    from tokenomics.eval import supply_trace as st
    from tokenomics.eval.decoder_check import MWWK, decoder_arm, select_views, world_entry
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from tokenomics.eval.routing_audit import build_jobs
    from tokenomics.eval.teacher_audit import load_atlas

    view = select_views(load_atlas(ATLAS_V2), [world])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    prod = "entity_planner_buyahead_k5_rf5"
    arm = {**arm, "producing": prod, "subject": member_id(prod, arm["ordering"])}
    jobs = [j for j in build_jobs([world_entry(view)], [arm], 1, 40, None, {})][:n]
    out = []
    for j in jobs:
        j.update(arm=MWWK, order_ttl=2, **({"acceptance": acceptance} if acceptance else {}))
        out.append(st.run_supply_job(j))
    return out


@pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(ORDERER)), reason="records absent")
def test_auction_policy_rows_equal_the_default_rows_on_episodes() -> None:
    """D3: the auction as a policy reproduces the domain's own auction cell for cell on real
    episodes (k5_rf5 + MWWk, horizon 40), outside the policy's own counters."""
    for world in ("ws-atlas-v1-12-0111", "ws-atlas-v1-12-0391"):
        plain, ruled = _rows(None, world), _rows("auction", world)
        for a, b in zip(plain, ruled):
            diff = [k for k in a if k != "acceptance_stats" and a[k] != b[k]]
            assert not diff, (world, diff[:5])
            assert a["acceptance_stats"] == "" and b["acceptance_stats"]


def test_outcomes_are_visible_to_the_router_next_cycle() -> None:
    """D2: after a cycle, each stream agent's view carries last cycle's sub-order outcomes."""
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.baselines import producing_policy
    from tokenomics.eval.routing_audit import ordering_policy
    from tokenomics.domain.config import load_config

    env = joint_stack_env(load_config(REPO / "tokenomics/tokenomics/config/phase1.json"), 6)
    env.acceptance_rule = lambda v: [AcceptanceDecision(False) for _ in v.bids]
    env.reset(seed=1)
    prod, order = producing_policy("entity_planner"), ordering_policy("round_robin")
    seen = []
    while env.agents and env.system.cycle_counter <= 3:
        agent = env.agent_selection
        role, aid = env._role[agent]
        if role == "order" and env.system.cycle_counter >= 2:
            seen += list(env.system.product_stream_view(aid).last_outcomes)
        env.step(prod(env, agent) if role == "prod" else order(env, agent))
    assert seen and all(o.outcome == "declined" and o.reason == REJECT_DECLINED for o in seen)


def test_lost_declined_is_a_registered_b2_class() -> None:
    """Observer fix (EXP-0098 A2's consistency failures): a replenishment order declined by the
    supplier's acceptance policy has a b2 class, so the class totals add up to the trace."""
    from tokenomics.eval import supply_trace as st

    assert "b2_lost_declined" in st.B2 and "st_b2_lost_declined" in st.ST_COLUMNS
    assert st.CLASS_REASON["b2_lost_declined"] == "ttl"


def test_remaining_lifetime_travels_from_the_order_to_the_bid() -> None:
    """The order's remaining lifetime (ttl - cycles played) is on OrderInfo, the routed
    sub-order and the bid, and so in the producer's AcceptanceView."""
    from tokenomics.domain.order import Order, OrderInfo
    from tokenomics.domain.ordering_agent import OrderingAgent, OrderingAgentCommand, SubOrder
    from tokenomics.domain.producing_agent import AcceptanceView, ProducerInfo, SubOrder as Routed

    p = producer(10)
    p.sheets = {0: __import__("tokenomics.domain.config", fromlist=["ProcessSheet"]).ProcessSheet(
        0, {"c": 1})}
    p.produces = frozenset({0})
    seen = []
    p.acceptance = lambda v: (seen.append(v), [AcceptanceDecision(True) for _ in v.bids])[1]
    p.place_sub_orders([Routed("o1", 0, 2, 20, remaining_ttl=1)])
    p.produce()
    assert isinstance(seen[0], AcceptanceView) and seen[0].bids[0].remaining_ttl == 1
    info = OrderInfo("o2", 10, 0, 3, 2)
    agent = OrderingAgent("a")
    agent.place_order(info)
    routed = agent.handle_cmd(OrderingAgentCommand({"o2": [SubOrder("p", 3, 10)]}),
                              {"p": ProducerInfo("p", {"c": 5}, {"c": 5}, 1.0, frozenset({0}))})
    assert routed["p"][0].remaining_ttl == 2


def test_order_remaining_ttl_counts_down_to_its_last_cycle() -> None:
    from tokenomics.domain.order import Order

    o = Order("x", 10, 0, 1, None, None, True, ttl=2)
    assert [o.remaining_ttl()] == [2]
    o.cycle_counter = 2
    assert o.remaining_ttl() == 0 and o.info().remaining_ttl == 0


@pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(ORDERER)), reason="records absent")
def test_pending_claims_that_die_with_their_order_are_counted() -> None:
    """Observer fix: an accepted claim that never ran and fails when its parent order expires is
    traced (`pending_failed_parent`), not left to a difference; every accepted-later claim
    resolves to exactly one of ran later / failed at its own TTL / died with its order / open."""
    from tokenomics.eval.decoder_check import (MWWK, decoder_arm, decoder_policies, job_draw_key,
                                               select_views, world_entry)
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from tokenomics.eval.routing_audit import build_jobs, job_config
    from tokenomics.eval.routing_diagnostics import set_draw_key
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.teacher_audit import load_atlas

    view = select_views(load_atlas(ATLAS_V2), ["ws-atlas-v1-12-0111"])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    prod = "entity_planner_buyahead_k5_rf5"
    arm = {**arm, "producing": prod, "subject": member_id(prod, arm["ordering"])}
    job = {**build_jobs([world_entry(view)], [arm], 1, 40, None, {})[0], "order_ttl": 2}
    env = joint_stack_env(job_config(job), 40)
    set_draw_key(env, job_draw_key(job))
    env.acceptance_rule = acc.acceptance_ttl_feasible
    p, o, _ = decoder_policies(job)
    run_episode(env, p, o, seed=job["episode"], action_seed=job["action_seed"])
    stats: dict = {}
    for q in env.system.producing_agents.values():
        for k, v in q.acceptance_stats.items():
            stats[k] = stats.get(k, 0) + v
    open_now = sum(len(q.pending) for q in env.system.producing_agents.values())
    assert stats.get("accepted_pending", 0) > 0 and stats.get("pending_failed_parent", 0) > 0
    assert stats["accepted_pending"] == (stats.get("pending_ran_later", 0)
                                         + stats.get("pending_failed_ttl", 0)
                                         + stats["pending_failed_parent"] + open_now)
    assert stats["accepted"] == stats.get("ran_same_cycle", 0) + stats["accepted_pending"]
