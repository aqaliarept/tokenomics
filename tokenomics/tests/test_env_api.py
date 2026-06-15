"""PettingZoo API compliance + random rollout (ADR-0025 model).

phase1.json is re-derived in B7; until then the env tests run on the composite
supply-chain fixture (eligibility holds: every product's required capacity types
are owned by a single producer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pettingzoo.test import api_test, seed_test

from tokenomics.domain import build_system
from tokenomics.domain.common import from_producer_id
from tokenomics.domain.config import load_config
from tokenomics.domain.producing_agent import ProducingAgentCommand
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.spaces import build_spec

COMPOSITE = Path(__file__).resolve().parent / "fixtures" / "composite.json"
PHASE1 = Path(__file__).resolve().parents[1] / "tokenomics" / "config" / "phase1.json"


def _cfg():
    return load_config(COMPOSITE)


def _env(horizon: int) -> TokenomicsAECEnv:
    return TokenomicsAECEnv(config=_cfg(), horizon=horizon)


def test_pettingzoo_api():
    api_test(_env(horizon=40), num_cycles=20, verbose_progress=False)


def test_seed_determinism():
    # env is deterministic (scripted consumers); same seed -> identical run
    seed_test(lambda: _env(horizon=30))


def test_random_rollout_to_termination():
    e = _env(horizon=50)
    e.reset(seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    for agent in e.agent_iter(max_iter=10000):
        obs, reward, term, trunc, info = e.last()
        if term or trunc:
            action = None
        else:
            space = e.action_space(agent)
            space.seed(int(rng.integers(0, 2**31)))
            action = space.sample()
        e.step(action)
        steps += 1
    assert steps > 0
    assert len(e.agents) == 0  # all agents reached done


def test_observation_in_space():
    e = _env(horizon=10)
    e.reset()
    for agent in e.possible_agents:
        obs = e.observe(agent)
        assert e.observation_space(agent).contains(obs), agent


def test_k_slots_cover_worst_case_producer_orders():
    # A producer's ordering agent can hold, in one cycle, 3 investment orders
    # (restoration/upgrade/expansion) plus one replenishment order per relevant
    # input. K must cover that so no order is dropped to an unobservable,
    # uncontrollable even-split fallback in decode (ADR-0025).
    cfg = load_config(PHASE1)  # shipped config: costs fund every order type
    sp = build_spec(cfg)
    sysm = build_system(cfg)
    full = ProducingAgentCommand(True, True, True, {20: 3, 11: 3})  # all 3 investments + 2 replenish
    for pid in sysm.producing_agents:
        sysm.producing_agent_action(pid, full if pid == "p4" else ProducingAgentCommand(False, False))
    sysm.start_ordering()
    counts = {aid: len(oa.incoming) for aid, oa in sysm.ordering_agents.items()}
    assert counts[from_producer_id("p4")] > 3  # the scenario really exceeds the old K=3
    for aid, n in counts.items():
        assert n <= sp.k, (aid, n, sp.k)


def test_global_state_shape():
    e = _env(horizon=10)
    e.reset()
    gs = e.global_state()
    assert gs.shape == (e.spec_.global_state_dim,)


def test_reward_shape():
    e = _env(horizon=5)
    e.reset()
    rng = np.random.default_rng(2)
    reward = 0.0
    for agent in e.agent_iter(max_iter=200):
        obs, reward, term, trunc, info = e.last()
        if term or trunc:
            e.step(None)
            continue
        sp = e.action_space(agent)
        sp.seed(int(rng.integers(0, 2**31)))
        e.step(sp.sample())
        if e.current_cycle >= 1:
            break
    assert isinstance(reward, float)


# --- B6 reward: inventory holding penalty ---
def test_holding_penalty_scales_with_units_held():
    from tokenomics.domain.system import CycleResult
    from tokenomics.env.aec_env import R_HOLDING_PER_UNIT, cycle_reward

    base = CycleResult(score=0)  # no completions/rejections
    r0 = cycle_reward(base, held=0)
    r5 = cycle_reward(base, held=5)
    r10 = cycle_reward(base, held=10)
    assert r0 == 0.0
    assert r5 == -5 * R_HOLDING_PER_UNIT
    # penalty is linear in units held
    assert (r0 - r5) == (r5 - r10)


def test_holding_penalty_independent_of_completions():
    from tokenomics.domain.system import CycleResult
    from tokenomics.env.aec_env import R_HOLDING_PER_UNIT, cycle_reward

    import pytest

    res = CycleResult(score=0, consumer_completed=2)
    delta = cycle_reward(res, held=3) - cycle_reward(res, held=0)
    assert delta == pytest.approx(-3 * R_HOLDING_PER_UNIT)
