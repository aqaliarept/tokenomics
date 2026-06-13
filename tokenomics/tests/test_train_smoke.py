"""Smoke test for the PPO trainer — runs a couple of tiny iterations and checks
the loop completes and returns a per-iteration training curve (ADR-0003).
Skipped if torch is not installed (it's an optional `train` extra)."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")


def test_train_runs_two_iters():
    from tokenomics.train.smoke_ppo import IterStats, train

    history = train(
        iters=2, steps=128, horizon=40, seed=0, eval_seeds=2, log=lambda *a, **k: None
    )
    assert len(history) == 2
    assert all(isinstance(r, IterStats) for r in history)
    # optimization signals always present
    assert all(isinstance(r.mean_step_reward, float) for r in history)
    # greedy-eval task metrics populated (not the skipped-iter NaN)
    assert all(not math.isnan(r.eval_cycles) for r in history)
    assert all(r.eval_cycles > 0 for r in history)


def test_gae_truncation_bootstraps_termination_does_not():
    """ADR-0004: the final transition bootstraps on truncation (boot value used)
    but not on termination (term kills the value backup)."""
    from tokenomics.train.smoke_ppo import gae

    rew = [0.0, 0.0]
    val = [0.0, 0.0]
    # truncated last step with a positive bootstrap -> non-zero advantage flows back
    adv_trunc, _ = gae(rew, val, term=[0.0, 0.0], boot=[0.0, 5.0], gamma=0.99, lam=0.95)
    # terminated last step -> no bootstrap, all advantages zero
    adv_term, _ = gae(rew, val, term=[0.0, 1.0], boot=[0.0, 5.0], gamma=0.99, lam=0.95)

    assert adv_trunc[1] > 0.0
    assert adv_term[1] == 0.0


def test_vectorized_collect_fills_both_buffers():
    """ADR-0005: collect() drives a list of envs in round-robin and buckets
    decisions into producing + ordering buffers. `(env_idx, agent)` keying must
    not drop transitions when agent names repeat across envs."""
    import numpy as np

    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.smoke_ppo import (
        CategoricalActor,
        CentralCritic,
        GaussianActor,
        collect,
    )

    envs = [TokenomicsAECEnv(horizon=40) for _ in range(3)]
    e = envs[0]
    prod = CategoricalActor(e.observation_space(e._prod_agents[0]).shape[0], 4)
    order = GaussianActor(
        e.observation_space(e._order_agents[0]).shape[0], e.spec_.ordering_action_dim
    )
    critic = CentralCritic(e.spec_.global_state_dim)
    bound = float(e.action_space(e._order_agents[0]).high[0])

    pbuf, obuf = collect(envs, prod, order, critic, steps=200, bound=bound,
                         rng=np.random.default_rng(0))

    assert len(pbuf.obs) + len(obuf.obs) >= 200
    assert pbuf.obs and obuf.obs
    # global state width matches the centralized-critic input
    assert len(pbuf.glob[0]) == e.spec_.global_state_dim


def test_curriculum_ramps_and_scales_degradation():
    """ADR-0006: difficulty ramps d_start -> 1.0 and scales degradation; eval
    config stays real (untouched)."""
    from tokenomics.domain import load_config
    from tokenomics.env.aec_env import DEFAULT_CONFIG
    from tokenomics.train.smoke_ppo import Curriculum, scale_degradation

    cur = Curriculum(d_start=0.2, warmup_frac=0.6)
    assert cur.difficulty(0, 10) == pytest.approx(0.2)
    assert cur.difficulty(9, 10) == pytest.approx(1.0)
    # monotone non-decreasing
    ds = [cur.difficulty(i, 10) for i in range(10)]
    assert ds == sorted(ds)

    base = load_config(DEFAULT_CONFIG)
    eased = scale_degradation(base, 0.2)
    for b, e in zip(base.producer_configs, eased.producer_configs):
        assert e.degradation == max(0, round(b.degradation * 0.2))
        assert e.degradation <= b.degradation
    # base untouched (frozen + replace -> new object)
    assert scale_degradation(base, 1.0).producer_configs[0].degradation == \
        base.producer_configs[0].degradation


def test_curve_csv_written(tmp_path):
    from tokenomics.train.smoke_ppo import train

    out = tmp_path / "curve.csv"
    train(iters=2, steps=128, horizon=40, seed=0, eval_seeds=2, out=str(out),
          log=lambda *a, **k: None)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("iter,mean_step_reward")
    assert len(lines) == 3  # header + 2 iters
