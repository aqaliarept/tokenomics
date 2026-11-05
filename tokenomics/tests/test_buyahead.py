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
