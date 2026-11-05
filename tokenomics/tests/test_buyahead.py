"""`entity_planner_buyahead` (add-m4-orderer-redesign task 7.18): k = 0 is `entity_planner`; the
rule stocks k cycles of the producer's own recent input use ahead of demand; it runs through the
supply observer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tokenomics.domain.config import (ProcessSheet, ProducingAgentConfig, Restoration, Upgrade,
                                      load_config)
from tokenomics.domain.producer_planner import plan_producer_command
from tokenomics.domain.producer_view import build_producer_entity_view
from tokenomics.domain.producing_agent import ProducingAgent
from tokenomics.domain.telemetry import EconomyTelemetry
from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval import baselines as b
from tokenomics.eval.roster import build_roster_v1, build_roster_v2

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[1]
COMPOSITE = TESTS / "fixtures" / "composite.json"
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"

GOOD = ProcessSheet("good", {"a": 1}, {"raw": 2})  # 2 raw per unit of good
RAW = ProcessSheet("raw", {"u": 1})


def _view(completed_per_cycle: int, inventory: int):
    """An assembler P with 2 raw per good, `inventory` raw in stock, no open demand, and a
    telemetry history of `completed_per_cycle` goods completed every cycle (steady EMA)."""
    cfg = ProducingAgentConfig(id="P", capacities={"a": 50}, degradation=0,
                               restoration=Restoration("raw", 1, "a"),
                               upgrade=Upgrade("raw", 1, "a"), produces=frozenset({"good"}),
                               inventory_limit=100)
    p = ProducingAgent(cfg, {"good": GOOD})
    p.inventory["raw"] = inventory
    tel = EconomyTelemetry()
    for c in range(40):  # converge the EMA to the per-cycle rate
        if completed_per_cycle:
            tel.record_completed("P", "good", completed_per_cycle, c, c)
        tel.end_cycle()
    return build_producer_entity_view(p, tel, {"good": GOOD, "raw": RAW}, {})


def _replenish(view, k: float) -> dict:
    cmd = plan_producer_command(b.buyahead_view(view, k), max_replenish=100,
                                scores=b.score_producer_entity_view(view))
    return dict(cmd.replenishments)


def test_recent_input_use_is_completions_times_recipe() -> None:
    view = _view(completed_per_cycle=4, inventory=3)
    assert b.recent_input_use(view) == pytest.approx({"raw": 8.0})


def test_buyahead_stocks_k_cycles_of_use_before_any_order() -> None:
    """No open demand, no reservation: entity_planner only keeps its floor of 3 (have 3 ->
    orders nothing); buy-ahead targets 3 + k x 8 = 11 / 19 raw and orders the difference."""
    view = _view(completed_per_cycle=4, inventory=3)
    assert view.input_rows[0].downstream_demand_pressure == 0
    base = plan_producer_command(view, max_replenish=100)
    assert dict(base.replenishments) == {} == _replenish(view, 0)
    assert _replenish(view, 1) == {"raw": 5}   # max(3, 0 + 0 + 8) - 3
    assert _replenish(view, 2) == {"raw": 13}  # max(3, 16) - 3
    assert _replenish(view, 3) == {"raw": 21}  # max(3, 24) - 3 (task 7.20)
    assert _replenish(view, 4) == {"raw": 29}  # max(3, 32) - 3
    idle = _view(completed_per_cycle=0, inventory=3)
    assert _replenish(idle, 1) == {}  # no past use: nothing ahead


def test_buyahead_respects_inventory_headroom_and_ceiling() -> None:
    view = _view(completed_per_cycle=40, inventory=0)  # use 80/cycle: target far above limits
    assert _replenish(view, 1) == {"raw": 80}
    cmd = plan_producer_command(b.buyahead_view(view, 1), max_replenish=7,
                                scores=b.score_producer_entity_view(view))
    assert dict(cmd.replenishments) == {"raw": 7}  # the per-cycle ceiling holds
    assert view.globals.inventory_headroom == 100


def test_k0_is_entity_planner_on_an_episode() -> None:
    env_a = TokenomicsAECEnv(config=load_config(COMPOSITE), horizon=15)
    env_b = TokenomicsAECEnv(config=load_config(COMPOSITE), horizon=15)
    k0 = b.producing_entity_planner_buyahead(0)
    for env in (env_a, env_b):
        env.reset(seed=0)
    steps = 0
    for agent in env_a.agent_iter():
        assert agent == env_b.agent_selection
        _, _, term, trunc, _ = env_a.last()
        if term or trunc:
            act_a = act_b = None
        elif env_a._role[agent][0] == "producing":
            act_a = b.producing_entity_planner(env_a, agent)
            act_b = k0(env_b, agent)
            assert np.array_equal(act_a, act_b)
        else:
            act_a = b.ordering_even_split(env_a, agent)
            act_b = b.ordering_even_split(env_b, agent)
        env_a.step(act_a)
        env_b.step(act_b)
        steps += 1
    assert steps > 0


def test_not_in_producing_baselines_so_roster_drafts_are_unchanged() -> None:
    for name in b.EXTRA_PRODUCING:
        assert name not in b.PRODUCING_BASELINES
        assert b.producing_policy(name) is b.EXTRA_PRODUCING[name]
    assert b.producing_policy("entity_planner") is b.PRODUCING_BASELINES["entity_planner"]
    producers = {m.producing for r in (build_roster_v1(), build_roster_v2()) for m in r.members}
    assert not producers & set(b.EXTRA_PRODUCING)


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


@pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(ORDERER)), reason="records absent")
def test_supply_observer_runs_with_buyahead(tmp_path: Path) -> None:
    from tokenomics.eval import supply_trace as st

    out = tmp_path / "ba"
    assert st.main(["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run",
                    "--worlds", "ws-atlas-v1-12-0060", "--horizon", "12", "--workers", "1",
                    "--order-ttl", "2", "--producer", "entity_planner_buyahead",
                    "--out-dir", str(out)]) == 0
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["producer"] == "entity_planner_buyahead"
    assert set(spec["pins"]["subjects"].values()) == {"entity_planner_buyahead/mww_keyed",
                                                     "entity_planner_buyahead/round_robin"}
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    rows = list(csv.DictReader((out / "supply_trace.csv").open()))
    assert len(rows) == 6
    with pytest.raises(ValueError, match="belongs to the TTL sweep"):
        st.run_supply_trace(atlas_path=ATLAS_V2, out_dir=tmp_path / "x", orderer_ckpt=str(ORDERER),
                            producer="entity_planner_buyahead", dry_run=True)
    with pytest.raises(ValueError, match="takes none"):
        st.run_supply_trace(atlas_path=ATLAS_V2, out_dir=tmp_path / "y", orderer_ckpt=str(ORDERER),
                            order_ttl=2, producer="entity_planner_buyahead",
                            exp0057_episodes="e.csv", dry_run=True)


def test_registered_k_values() -> None:  # buy-ahead entries without the rf parts
    """Task 7.20: the registered buy-ahead producers and their k."""
    assert {n: p.ahead_cycles for n, p in b.EXTRA_PRODUCING.items()
            if not hasattr(p, "rf_parts")} == {
        "entity_planner_buyahead": 1, "entity_planner_buyahead_k2": 2,
        "entity_planner_buyahead_k3": 3, "entity_planner_buyahead_k4": 4}


# --- task 7.22: "reservations first" on top of k = 4 ---

GADGET = ProcessSheet("gadget", {"a": 1}, {"rare": 1})
RARE = ProcessSheet("rare", {"u": 1})


def _rf_view(limit: int, reserved_gadgets: int, good_per_cycle: int = 8):
    """P makes `good` (2 raw) and `gadget` (1 rare); it recently completed `good_per_cycle`
    goods a cycle (raw use 2x that) and holds a reservation for `reserved_gadgets` gadgets that
    still lacks its rare inputs (blocked shortfall = reserved_gadgets)."""
    from tokenomics.domain.producing_agent import Bid, _Reservation

    cfg = ProducingAgentConfig(id="P", capacities={"a": 50}, degradation=0,
                               restoration=Restoration("raw", 1, "a"),
                               upgrade=Upgrade("raw", 1, "a"),
                               produces=frozenset({"good", "gadget"}), inventory_limit=limit)
    p = ProducingAgent(cfg, {"good": GOOD, "gadget": GADGET})
    if reserved_gadgets:
        p.reservations = [_Reservation(Bid(capacities={"a": reserved_gadgets}, tokens=10,
                                           order_id=1, product="gadget",
                                           inputs={"rare": reserved_gadgets},
                                           quantity=reserved_gadgets))]
    tel = EconomyTelemetry()
    for c in range(40):
        tel.record_completed("P", "good", good_per_cycle, c, c)
        tel.end_cycle()
    sheets = {"good": GOOD, "gadget": GADGET, "raw": RAW, "rare": RARE}
    return build_producer_entity_view(p, tel, sheets, {})


def _rf(view, **kw) -> object:
    args = dict(ahead_cycles=4, max_replenish=100, replenish_cost_per_unit=None,
                funding_budget=None, kept_room=0, inventory_expansion_increases=0,
                inventory_expansion_cost=0)
    args.update(kw)
    return b.plan_reservations_first(view, **args)


def test_rf_blocked_shortfall_gets_headroom_first() -> None:
    view = _rf_view(limit=6, reserved_gadgets=4)
    assert b.blocked_shortfall(view) == {"raw": 0, "rare": 4}
    cmd = _rf(view)
    assert cmd.replenishments.get("rare") == 4  # the reservation's input first
    assert sum(cmd.replenishments.values()) <= 6  # within headroom
    off = _rf(view, blocked_first=False, keep_room=False, expand_early=False)
    k4 = plan_producer_command(b.buyahead_view(view, 4), max_replenish=100,
                               scores=b.score_producer_entity_view(view))
    assert off == k4  # all parts off: buy-ahead k = 4 exactly


def test_rf_keeps_room_from_the_k_term_only() -> None:
    view = _rf_view(limit=10, reserved_gadgets=0)  # base target 3 each; raw's k-term far above
    assert _rf(view, kept_room=0).replenishments == {"raw": 7, "rare": 3}  # 6 base + 4 k-term
    assert _rf(view, kept_room=3).replenishments == {"raw": 4, "rare": 3}  # k-term 10 - 6 - 3
    assert _rf(view, kept_room=9).replenishments == {"raw": 3, "rare": 3}  # base kept in full


def test_rf_expands_when_headroom_is_below_the_blocked_shortfall() -> None:
    view = _rf_view(limit=2, reserved_gadgets=4)  # headroom 2 < shortfall 4, headroom > 0
    assert view.globals.inventory_headroom == 2
    kw = dict(inventory_expansion_increases=10, inventory_expansion_cost=0)
    assert _rf(view, **kw).do_inventory_expansion
    assert not _rf(view, expand_early=False, **kw).do_inventory_expansion  # planner: only at 0
    assert not _rf(view, inventory_expansion_increases=0).do_inventory_expansion


def test_rf_all_parts_off_is_k4_on_an_episode() -> None:
    env_a = TokenomicsAECEnv(config=load_config(COMPOSITE), horizon=15)
    env_b = TokenomicsAECEnv(config=load_config(COMPOSITE), horizon=15)
    k4 = b.EXTRA_PRODUCING["entity_planner_buyahead_k4"]
    off = b.producing_entity_planner_buyahead_rf(4, blocked_first=False, keep_room=False,
                                                 expand_early=False)
    for env in (env_a, env_b):
        env.reset(seed=0)
    for agent in env_a.agent_iter():
        _, _, term, trunc, _ = env_a.last()
        if term or trunc:
            act_a = act_b = None
        elif env_a._role[agent][0] == "producing":
            act_a, act_b = k4(env_a, agent), off(env_b, agent)
            assert np.array_equal(act_a, act_b)
        else:
            act_a = act_b = b.ordering_even_split(env_a, agent)
        env_a.step(act_a)
        env_b.step(act_b)


def test_rf_registered_entries() -> None:
    parts = {n: (p.ahead_cycles, p.rf_parts) for n, p in b.EXTRA_PRODUCING.items()
             if hasattr(p, "rf_parts")}
    assert parts == {"entity_planner_buyahead_k4_rf": (4, (True, True, True)),
                     "entity_planner_buyahead_k4_rf1": (4, (True, False, False)),
                     "entity_planner_buyahead_k4_rf12": (4, (True, True, False)),
                     "entity_planner_buyahead_k4_rf3": (4, (False, False, True)),
                     "entity_planner_buyahead_k4_rf4": (4, (False, False, True))}
    assert hasattr(b.EXTRA_PRODUCING["entity_planner_buyahead_k4_rf4"], "upgrades_blocked")
    assert not any(hasattr(b.EXTRA_PRODUCING[n], "upgrades_blocked")
                   for n in ("entity_planner_buyahead_k4_rf3", "entity_planner_buyahead_k4"))


def _actions(policy, world: str, episode_index: int, horizon: int = 40) -> list[bytes]:
    """The policy's producing actions over one EXP-0057-path episode (MWWk r0) of `world`."""
    from tokenomics.eval.decoder_check import (MWWK, decoder_arm, decoder_policies,
                                               job_draw_key, select_views, world_entry)
    from tokenomics.eval.joint_stack import joint_stack_env
    from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
    from tokenomics.eval.routing_audit import build_jobs, job_config
    from tokenomics.eval.routing_diagnostics import set_draw_key
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.teacher_audit import load_atlas

    view = select_views(load_atlas(ATLAS_V2), [world])[0]
    arm = decoder_arm(MWWK, load_roster(DEFAULT_ROSTER))
    job = [j for j in build_jobs([world_entry(view)], [arm], 1, horizon, None, {})
           if j["replicate"] == 0][episode_index]
    env = joint_stack_env(job_config(job), horizon)
    set_draw_key(env, job_draw_key(job))
    seen: list[bytes] = []

    def recording(e, agent):
        a = policy(e, agent)
        seen.append(np.asarray(a).tobytes())
        return a

    run_episode(env, recording, decoder_policies(job)[1], seed=job["episode"],
                action_seed=job["action_seed"])
    return seen


@pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(ORDERER)), reason="records absent")
def test_rf_state_does_not_carry_across_episodes() -> None:
    """Review m4-orderer-7-22: two episodes run back to back through one rf policy object give the
    same actions as each run through a fresh one, and a finished episode leaves no state."""
    import gc

    shared = b.producing_entity_planner_buyahead_rf(4)
    first = _actions(shared, "ws-atlas-v1-12-0111", 0)
    second = _actions(shared, "ws-atlas-v1-12-0254", 0)
    third = _actions(shared, "ws-atlas-v1-12-0111", 0)  # the first world again, after two others
    assert first == _actions(b.producing_entity_planner_buyahead_rf(4), "ws-atlas-v1-12-0111", 0)
    assert second == _actions(b.producing_entity_planner_buyahead_rf(4), "ws-atlas-v1-12-0254", 0)
    assert third == first
    gc.collect()
    assert len(shared.rf_state) == 0  # weak keys: no system outlives its episode


def test_rf_state_restarts_when_the_cycle_counter_goes_back() -> None:
    from types import SimpleNamespace

    policy = b.producing_entity_planner_buyahead_rf(4)
    view = _rf_view(limit=2, reserved_gadgets=4)  # blocked shortfall 4 every cycle
    class FakeSystem:  # weak-referenceable, like System
        cycle_counter = 0
        investment_fund = 0

        def producer_entity_view(self, pid: str) -> object:
            return view

    system = FakeSystem()
    mech = SimpleNamespace(replenish_cost_per_unit=None, inventory_expansion_increases=0,
                           inventory_expansion_cost=0)
    env = SimpleNamespace(_role={"a": ("producing", "P")}, system=system,
                          spec_=SimpleNamespace(producer_mechanics={"P": mech},
                                                max_replenish={"P": 100}))
    import tokenomics.eval.baselines as bl
    orig = bl._action_from_producing_command
    bl._action_from_producing_command = lambda spec, pid, cmd: np.zeros(1)
    try:
        for c in range(1, 6):
            system.cycle_counter = c
            policy(env, "a")
        assert policy.rf_state[system]["P"][1] > 0
        system.cycle_counter = 1  # replayed from the start
        policy(env, "a")
        assert policy.rf_state[system]["P"] == (1, b.RF_ALPHA * 4)
    finally:
        bl._action_from_producing_command = orig


def test_rf3_buys_as_k4_and_only_expands_earlier() -> None:
    """M5 task 1.1: part 3 alone keeps k = 4's replenishment, restoration and upgrade exactly; only
    the inventory-expansion toggle can differ, and only when headroom is below the blocked
    shortfall."""
    kw = dict(inventory_expansion_increases=10, inventory_expansion_cost=0)
    for limit, reserved in ((2, 4), (6, 4), (10, 0), (100, 3)):
        view = _rf_view(limit=limit, reserved_gadgets=reserved)
        k4 = plan_producer_command(b.buyahead_view(view, 4), max_replenish=100,
                                   scores=b.score_producer_entity_view(view),
                                   inventory_expansion_increases=10, inventory_expansion_cost=0)
        rf3 = _rf(view, blocked_first=False, keep_room=False, **kw)
        assert rf3.replenishments == k4.replenishments, limit
        assert (rf3.do_restoration, rf3.do_upgrade) == (k4.do_restoration, k4.do_upgrade)
        short = sum(b.blocked_shortfall(view).values())
        early = view.globals.inventory_headroom < short
        assert rf3.do_inventory_expansion == (k4.do_inventory_expansion or early), limit
    assert b.EXTRA_PRODUCING["entity_planner_buyahead_k4_rf3"].rf_parts == (False, False, True)


# --- M5 task 4.2f: rf4 = rf3 + the sustainable-upgrade cap (DIAGNOSTICS 29) ---

def _cap_view(degradation: int, mx: int, upgrade: int, restore: int):
    from types import SimpleNamespace

    return SimpleNamespace(capacity_rows=[SimpleNamespace(
        degradation=degradation, max=mx, upgrade_effect=upgrade, restore_effect=restore)])


def test_the_cap_blocks_an_over_ceiling_upgrade_and_allows_an_under_ceiling_one() -> None:
    # 12-0016's p0_0 (6%, restore 16, upgrade 8): at max 263 the view shows ceil(15.78) = 16, and
    # 271 would degrade ceil(16.26) = 17 > 16 -> blocked; at max 200 (12 -> 13) allowed
    assert b.over_sustainable_max(_cap_view(16, 263, 8, 16))
    assert not b.over_sustainable_max(_cap_view(12, 200, 8, 16))
    assert not b.over_sustainable_max(_cap_view(16, 266, 0, 16))  # no upgrade on the type
    # a type the restoration does not target gets no cap (supervisor decision, notes/4.2f.md)
    assert not b.over_sustainable_max(_cap_view(40, 300, 8, 0))


@pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(ORDERER)), reason="records absent")
def test_rf4_equals_rf3_where_no_over_ceiling_upgrade_happens(monkeypatch) -> None:
    calls = []
    real = b.over_sustainable_max
    monkeypatch.setattr(b, "over_sustainable_max", lambda v: calls.append(real(v)) or calls[-1])
    rf3 = b.producing_entity_planner_buyahead_rf(4, blocked_first=False, keep_room=False)
    rf4 = b.producing_entity_planner_buyahead_rf(4, blocked_first=False, keep_room=False,
                                                 upgrade_cap=True)
    a4 = _actions(rf4, "ws-atlas-v1-12-0111", 0)
    assert not any(calls)  # no over-ceiling upgrade wanted on this episode
    assert a4 == _actions(rf3, "ws-atlas-v1-12-0111", 0)


@pytest.mark.skipif(not (_lfs(ATLAS_V2) and _lfs(ORDERER)), reason="records absent")
def test_rf4_first_departs_from_rf3_at_its_first_blocked_upgrade(monkeypatch) -> None:
    """On an episode with over-ceiling upgrades (atlas 12-0121, horizon 40) rf4's actions equal
    rf3's up to the first decision it blocks, and that decision differs."""
    real = b.over_sustainable_max
    rf4 = b.producing_entity_planner_buyahead_rf(4, blocked_first=False, keep_room=False,
                                                 upgrade_cap=True)
    seen = {"n": 0, "first": None}

    def counting(e, agent):
        seen["n"] += 1
        return rf4(e, agent)

    def over(v):
        hit = real(v)
        if hit and seen["first"] is None:
            seen["first"] = seen["n"] - 1
        return hit

    monkeypatch.setattr(b, "over_sustainable_max", over)
    a4 = _actions(counting, "ws-atlas-v1-12-0121", 0)
    a3 = _actions(b.producing_entity_planner_buyahead_rf(4, blocked_first=False, keep_room=False),
                  "ws-atlas-v1-12-0121", 0)
    first = seen["first"]
    assert first is not None
    assert a4[:first] == a3[:first] and a4[first] != a3[first]

