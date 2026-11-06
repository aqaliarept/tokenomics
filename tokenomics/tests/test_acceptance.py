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
