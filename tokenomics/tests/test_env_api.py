"""PettingZoo API compliance + random rollout."""

from __future__ import annotations

import numpy as np
from pettingzoo.test import api_test, seed_test

from tokenomics.env import TokenomicsAECEnv, env


def test_pettingzoo_api():
    api_test(env(horizon=40), num_cycles=20, verbose_progress=False)


def test_seed_determinism():
    # env is deterministic (scripted consumers); same seed -> identical run
    seed_test(lambda: env(horizon=30))


def test_random_rollout_to_termination():
    e = TokenomicsAECEnv(horizon=50)
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
    e = TokenomicsAECEnv(horizon=10)
    e.reset()
    for agent in e.possible_agents:
        obs = e.observe(agent)
        assert e.observation_space(agent).contains(obs), agent


def test_reward_shape():
    e = TokenomicsAECEnv(horizon=5)
    e.reset()
    # drive one full cycle with no-op producing + even ordering via samples
    rng = np.random.default_rng(2)
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
