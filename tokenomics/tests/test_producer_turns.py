"""ADR-0062 (add-m6-joint-mpg-gnn 4.2 / 4.6 / 4.7): the fund turn and the accept turn.

- **Defaults are today, byte for byte:** an env with the turns and their default policies plays the episode the
  default env plays (episode stats on a busy world; supply-trace rows against EXP-0088's k5_rf5 rows).
- **The env stays a PettingZoo AEC env** with the extra agents.
- **The domain applies the scores:**
  - stage 2 spends in score order;
  - one priority order runs every claim (stock and capacity);
  - re-validation drops the lowest score first;
  - declines carry their reason;
  - bids beyond the slots are declined and counted.
- **Information set (ADR-0058):** a producer's turn features do not move when another producer's bids or
  capacity do.
- **GNN:** turn heads leave the investment heads unchanged, and they survive a save / load.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tokenomics.domain.config import ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.producing_agent import (FAIL_DROPPED, K_BID, K_RES, REJECT_DECLINED,
                                               REJECT_DISPLACED, REJECT_NO_FIT, Bid,
                                               ProducingAgent, TurnDecision, _Reservation)
from tokenomics.env import producer_turns as pt

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
EXP0088 = (REPO / "experiments/EXP-0088-buffer-resweep-rf5/entity_planner_buyahead_k5_rf5/artifacts"
           / "episodes.csv")
MODELS = REPO / "experiments/EXP-0100-m6-imitation-v2/artifacts/fit/models.pt"
PROD = "entity_planner_buyahead_k5_rf5"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


needs = pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(ORDERER)), reason="records absent")


def producer(cap: int = 10, ttl: int = 2) -> ProducingAgent:
    cfg = ProducingAgentConfig(id="p", capacities={"c": cap}, degradation=0,
                               restoration=Restoration(require=1, restores=10, type="c"),
                               upgrade=Upgrade(require=1, increases=10, type="c"))
    return ProducingAgent(cfg, reservation_ttl=ttl)


def bid(c: int, tokens: int, oid: str, inputs: dict | None = None) -> Bid:
    return Bid(capacities={"c": c}, tokens=tokens, order_id=oid, product=5,
               inputs=dict(inputs or {}))


# --- the domain applies the scores ---


def test_the_default_decision_is_the_auction() -> None:
    bids = [(8, 80, "a"), (5, 30, "b"), (3, 40, "c"), (4, 4, "d"), (13, 90, "huge")]
    plain, turned = producer(12), producer(12)
    for p in (plain, turned):
        p.place_bids([bid(*x) for x in bids])
    turned.set_turn(turned.default_turn_decision())
    a, b = plain.produce(), turned.produce()
    for f in ("accepted", "completed", "failed", "rejected"):
        assert [x.order_id for x in getattr(a, f)] == [x.order_id for x in getattr(b, f)], f
    assert a.rejection_reasons == b.rejection_reasons
    assert plain.cut_off_price == turned.cut_off_price and plain.funds == turned.funds


def test_one_priority_order_runs_every_claim_for_stock() -> None:
    """A carried reservation waits for stock; with a higher score, this cycle's composite bid takes
    the one unit of stock first (today: carried first)."""
    def play(new_first: bool) -> list[str]:
        p = producer(20)
        p.reservations = [_Reservation(bid(5, 50, "old", {7: 1}))]
        p.inventory = {7: 1}
        p.place_bids([bid(5, 60, "new", {7: 1})])
        d = p.default_turn_decision()
        if new_first:
            d = TurnDecision(d.accept, (9.0,), (-9.0,), ())
        p.set_turn(d)
        return [x.order_id for x in p.produce().completed]

    assert play(False) == ["old"] and play(True) == ["new"]


def test_revalidation_drops_the_lowest_score_first() -> None:
    p = producer(10)
    p.reservations = [_Reservation(bid(6, 60, "rich", {7: 1})), _Reservation(bid(6, 12, "poor", {7: 1}))]
    d = p.default_turn_decision()  # 12 reserved > 10: today the cheaper one drops
    p.set_turn(TurnDecision((), (), (-5.0, 5.0), ()))  # slots: rich, poor (price order)
    r = p.produce()
    assert [x.order_id for x in r.failed] == ["rich"] and r.failure_reasons == [FAIL_DROPPED]
    assert d.reservation_priority[0] > d.reservation_priority[1]


def test_declines_carry_their_reason_and_burn_nothing() -> None:
    p = producer(10)
    p.place_bids([bid(6, 60, "a"), bid(6, 30, "b"), bid(12, 99, "huge"), bid(2, 2, "c")])
    # slots, price order: huge (8.25), a (10), b (5), c (1): a 10, huge 8.25, b 5, c 1
    order = [b.order_id for b in p.turn_lists().bids]
    want = {"a": True, "huge": False, "b": False, "c": False}
    p.set_turn(TurnDecision(tuple(want[o] for o in order), (0.0,) * 4, (), ()))
    r = p.produce()
    reasons = dict(zip([x.order_id for x in r.rejected], r.rejection_reasons))
    assert reasons == {"huge": REJECT_NO_FIT, "b": REJECT_DISPLACED, "c": REJECT_DECLINED}
    assert p.funds == 60


def test_bids_beyond_the_slots_are_declined_and_counted() -> None:
    p = producer(10_000)
    p.place_bids([bid(1, 10 + i, f"o{i}") for i in range(K_BID + 3)])
    lists = p.turn_lists()
    d, beyond = pt.decode_accept(np.ones(pt.ACCEPT_ACTION_DIM, np.float32), lists)
    assert beyond == 3 and sum(d.accept) == K_BID and not any(d.accept[K_BID:])


def _result(r) -> tuple:
    return tuple([x.order_id for x in getattr(r, f)] for f in ("accepted", "completed", "failed",
                                                                 "rejected")) + (
        tuple(r.failure_reasons), tuple(r.rejection_reasons))


def _many_reservations(cap: int) -> ProducingAgent:
    """K_RES + 1 carried composite reservations (4 capacity each, waited 1 cycle, so a reservation
    without stock fails at TTL this cycle), stock for K_RES of them, one new bid of 8."""
    p = producer(cap=cap, ttl=2)
    p.reservations = [_Reservation(bid(4, 400 - i, f"r{i}", {7: 1}), 1) for i in range(K_RES + 1)]
    p.inventory = {7: K_RES}
    p.place_bids([bid(8, 8, "new")])
    return p


@pytest.mark.parametrize("cap", [200, 190])
def test_reservations_beyond_the_slots_keep_carried_first(cap: int) -> None:
    """Review repro (e6cacf0): the 49th reservation (beyond K_RES, the cheapest) fails at TTL and
    frees the capacity the new bid needs, so the bid completes, not pends. cap 190: re-validation
    must drop the cheapest first, as today. The default decision through the action = the auction."""
    plain, turned = _many_reservations(cap), _many_reservations(cap)
    ref = plain.produce()
    d, beyond = pt.decode_accept(pt.encode_accept_action(turned.default_turn_decision()),
                                 turned.turn_lists())
    turned.set_turn(d)
    got = turned.produce()
    assert beyond == 1 and _result(got) == _result(ref) and not turned.pending
    assert [r.bid.order_id for r in turned.reservations] == [r.bid.order_id for r in plain.reservations]
    if cap == 200:
        assert "new" in _result(got)[1]


def test_outage_ignores_the_decision() -> None:
    """Under an outage re-validation and every claim run in today's order, whatever the scores."""
    def play(flip: bool) -> tuple:
        p = producer(10)
        p.reservations = [_Reservation(bid(6, 60, "rich", {7: 1})), _Reservation(bid(6, 12, "poor", {7: 1}))]
        p.place_bids([bid(2, 20, "b")])
        d = p.default_turn_decision()
        if flip:
            d = TurnDecision((True,), (9.0,), (-5.0, 5.0), ())
        p.set_turn(d)
        return _result(p.produce(outage=True)), [r.bid.order_id for r in p.reservations]

    assert play(True) == play(False)


def test_fund_requests_beyond_the_slots_are_counted() -> None:
    env = _phase1_env(True, 4)
    env.reset(seed=0)
    n = pt.fund_slots(env.spec_)
    pid = env.spec_.producer_ids[0]
    reqs = [object()] * (n + 2)
    env.system.funding_requests = lambda _pid: reqs  # type: ignore[method-assign]
    env.system.set_funding_priorities = lambda _pid, scores: None  # type: ignore[method-assign]
    env.agent_selection = f"fund_{pid}"
    env.step(np.zeros(n, np.float32))
    assert env.turn_beyond_k == 2
    assert len(pt.decode_fund(np.zeros(n), n + 2)) == n + 2


def test_fund_scores_order_stage_two() -> None:
    from tokenomics.domain.system import distribute_investment_fund

    class O:
        def __init__(self, oid: str, kind: int, target: int) -> None:
            self.id, self.kind, self.target, self.got = oid, kind, target, 0
            from tokenomics.domain.requests import InvestmentType
            self.investment_request = (None if kind == 1 else type("R", (), {"type": {
                0: InvestmentType.RESTORATION, 2: InvestmentType.UPGRADE}[kind]})())

        def requires_funding(self) -> bool:
            return self.got == 0

        def producer_id(self) -> str:
            return "p"

        def cut_off_price(self) -> float:
            return 1.0

        def funding_target(self) -> int:
            return self.target

        def fund(self, g: int) -> None:
            self.got = g

    def run(priority: dict | None) -> dict:
        orders = {o.id: o for o in (O("up", 2, 60), O("rest", 0, 60), O("buy", 1, 60))}
        distribute_investment_fund(orders, 100, priority)
        return {k: o.got for k, o in orders.items()}

    assert run(None) == {"rest": 60, "buy": 40, "up": 0}  # survival-first, boundary partial
    assert run({"up": 3.0, "rest": 1.0, "buy": 2.0}) == {"up": 60, "buy": 40, "rest": 0}
    assert pt.default_fund_scores(3) == sorted(pt.default_fund_scores(3), reverse=True)


# --- the env ---


def _phase1_env(turns: bool, horizon: int = 30):
    from tokenomics.env import TokenomicsAECEnv

    return TokenomicsAECEnv(horizon=horizon, producer_turns=turns)


def test_pettingzoo_api_with_the_turns() -> None:
    from pettingzoo.test import api_test

    env = _phase1_env(True, 40)
    assert any(a.startswith("fund_") for a in env.possible_agents)
    assert any(a.startswith("acc_") for a in env.possible_agents)
    api_test(env, num_cycles=20, verbose_progress=False)


def test_phase1_default_turns_equal_the_default_env() -> None:
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import run_episode

    for prod in ("restore_when_low", "always_both"):
        a = run_episode(_phase1_env(False), PRODUCING_BASELINES[prod],
                        ORDERING_BASELINES["proportional"], seed=3)
        b = run_episode(_phase1_env(True), PRODUCING_BASELINES[prod],
                        ORDERING_BASELINES["proportional"], seed=3)
        assert a == b, prod


def _supply_rows(world: str, turns: bool, n: int, horizon: int) -> list[dict]:
    from tokenomics.eval import supply_trace as st
    from tokenomics.eval.decoder_check import MWWK, decoder_arm, select_views, world_entry
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from tokenomics.eval.routing_audit import build_jobs
    from tokenomics.eval.teacher_audit import load_atlas

    view = select_views(load_atlas(ATLAS_V2), [world])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    arm = {**arm, "producing": PROD, "subject": member_id(PROD, arm["ordering"])}
    jobs = [j for j in build_jobs([world_entry(view)], [arm], 1, horizon, None, {})][:n]
    out = []
    for j in jobs:
        j.update(arm=MWWK, order_ttl=2, producer_turns=turns)
        out.append(st.run_supply_job(j))
    return out


@needs
@pytest.mark.skipif(not _lfs(EXP0088), reason="records absent")
@pytest.mark.parametrize("world", ["ws-atlas-v1-12-0391", "ws-atlas-v1-12-0060"])
def test_default_turns_replay_exp0088_rows(world: str) -> None:
    """The EXP-0088 k5_rf5 replay (ADR-0062 D4): default turns reproduce the recorded rows, cell
    for cell (columns added since must read zero); 12-0060 is shocked (capacity hits, outages)."""
    csv.field_size_limit(1 << 30)
    with EXP0088.open() as fh:
        ref = {(r["world"], r["episode"], r["subject"], r["replicate"]): r
               for r in csv.DictReader(fh) if r["world"] == world}
    rows = _supply_rows(world, True, 2, 100)
    for r in rows:
        k = (r["world"], r["episode"], r["subject"], r["replicate"])
        b = ref[k]
        diff = [c for c in r if c != "acceptance_stats" and (
            r[c] != b[c] if c in b else r[c] not in ("", "0", "0.0"))]
        assert not diff, (k, diff[:5])


# --- information set ---


def _drive_to(role: str, cycle: int):
    """The k5_rf5 + MWWk stack on 12-0391 with default turns, stopped at the first `role` turn of
    `cycle` (not yet acted)."""
    from tokenomics.eval.decoder_check import (MWWK, decoder_arm, decoder_policies, job_draw_key,
                                               select_views, world_entry)
    from tokenomics.eval.routing_diagnostics import set_draw_key
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster, member_id
    from tokenomics.eval.routing_audit import build_jobs, job_config
    from tokenomics.eval.teacher_audit import load_atlas
    from tokenomics.train import m6_gnn as g

    view = select_views(load_atlas(ATLAS_V2), ["ws-atlas-v1-12-0391"])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    arm = {**arm, "producing": PROD, "subject": member_id(PROD, arm["ordering"])}
    job = [j for j in build_jobs([world_entry(view)], [arm], 1, 8, None, {})][0]
    job.update(arm=MWWK, order_ttl=2)
    env = joint_stack_env(job_config(job), 8, producer_turns=True)
    set_draw_key(env, job_draw_key(job))
    prod, order, _ = decoder_policies(job)
    env.reset(seed=job["episode"])
    for _ in range((cycle + 1) * len(env.possible_agents)):
        a = env.agent_selection
        r = env._role[a][0]
        if r == role and env.system.cycle_counter >= cycle:
            return env
        act = (prod(env, a) if r == "prod" else order(env, a) if r == "order" else
               pt.default_fund_policy(env, a) if r == "fund" else pt.default_accept_policy(env, a))
        env.step(act)
    raise AssertionError(f"no {role} turn by cycle {cycle}")


@needs
def test_turn_features_read_only_the_producers_own_state() -> None:
    """At an accept turn, producer A's turn observation and GNN turn features do not change when
    another producer's bids and capacity change."""
    pytest.importorskip("torch")
    from tokenomics.train import m6_gnn as g

    env = _drive_to("acc", 3)
    me = env._role[env.agent_selection][1]
    other = next(p for p in env.system.producing_agents if p != me)
    p = env.system.producing_agents[me]

    def snapshot() -> tuple:
        lists = p.turn_lists()
        t = g.accept_features(p, lists, g._turn_view(env, me))
        return (env.observe(env.agent_selection).tobytes(), t["bid"].tobytes(), t["claim"].tobytes())

    before = snapshot()
    q = env.system.producing_agents[other]
    q.place_bids([Bid(capacities=dict.fromkeys(q.capacities, 1), tokens=999, order_id="x")])
    for t in q.capacities:
        q.capacities[t] = max(0, q.capacities[t] - 1)
    assert snapshot() == before


@needs
def test_fund_turn_reads_no_other_producers_scores() -> None:
    """ADR-0062 §6: at a fund turn, producer A's observation and GNN fund features (its requests and
    its stage-1 budget) do not change when another producer's fund scores change; the request sets
    are fixed (every investment action precedes the fund turns)."""
    pytest.importorskip("torch")
    from tokenomics.domain.system import funding_groups
    from tokenomics.train import m6_gnn as g

    env = _drive_to("fund", 3)
    s = env.system
    groups = funding_groups(s.orders)
    while env._role[env.agent_selection][1] not in groups:  # the first fund turn with requests
        assert env._role[env.agent_selection][0] == "fund"
        env.step(pt.default_fund_policy(env, env.agent_selection))
    me = env._role[env.agent_selection][1]
    others = [pid for pid in groups if pid != me]
    assert others and me in groups, "the probe needs requests from A and from another producer"

    def snapshot() -> tuple:
        p = s.producing_agents[me]
        t = g.fund_features(p, s.funding_requests(me), s.funding_budget(me), g._turn_view(env, me))
        return (env.observe(env.agent_selection).tobytes(), t["fund"].tobytes(),
                s.funding_budget(me), [o.id for o in s.funding_requests(me)])

    before = snapshot()
    rng = np.random.default_rng(0)
    for pid in others:
        n = len(s.funding_requests(pid))
        s.set_funding_priorities(pid, list(rng.uniform(-10, 10, n)))
        assert snapshot() == before
        s.set_funding_priorities(pid, list(-np.arange(n, dtype=float)))
        assert snapshot() == before


# --- GNN heads ---


@pytest.mark.skipif(not _lfs(MODELS), reason="records absent")
def test_turn_heads_leave_the_investment_heads_and_survive_save_load(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from tokenomics.train import m6_gnn as g
    from tokenomics.train import m6_imitation as im

    pm, om = im.load_models(MODELS)
    pm2, _ = im.load_models(MODELS)
    torch.manual_seed(0)
    pm2.add_turn_heads()
    assert set(pm.state_dict()) < set(pm2.state_dict())
    assert all(torch.equal(v, pm2.state_dict()[k]) for k, v in pm.state_dict().items())
    im.save_models(pm2, om, tmp_path / "m.pt")
    pm3, _ = im.load_models(tmp_path / "m.pt")
    assert pm3.turn is not None and all(torch.equal(v, pm3.state_dict()[k])
                                        for k, v in pm2.state_dict().items())
    f = {"global": np.zeros((1, 13), np.float32), "capacity": np.zeros((1, g.CAPACITY_DIM[pm.features]),
         np.float32), "product": np.zeros((0, 18), np.float32), "input": np.zeros((0, 6), np.float32),
         "prod_cap": [], "prod_in": [], "inputs": []}
    t = {"bid": np.ones((2, g.BID_DIM), np.float32), "bid_cap": [(0, 0), (1, 0)], "bid_in": [],
         "claim": np.ones((1, g.CLAIM_DIM), np.float32), "claim_cap": [], "claim_in": [], "n_res": 1}
    acc, bp, cp = pm3.accept_outputs(f, t)
    assert acc.shape == (2,) and bp.shape == (2,) and cp.shape == (1,)
    assert pm3.fund_scores(f, {"fund": np.ones((3, g.FUND_DIM), np.float32), "fund_cap": [],
                               "fund_in": []}).shape == (3,)
