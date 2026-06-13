"""Eval harness + the signal-check invariant: a sensible policy must clearly
beat do-nothing, otherwise the env poses no learning problem."""

from __future__ import annotations

from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval.baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    ordering_even_split,
    producing_noop,
    producing_restore_when_low,
)
from tokenomics.eval.runner import evaluate, run_episode


def test_run_episode_returns_stats():
    env = TokenomicsAECEnv(horizon=40)
    s = run_episode(env, producing_noop, ordering_even_split, seed=0)
    assert s["cycles"] > 0
    assert "consumer_completed" in s and "cap_min" in s


def test_signal_exists():
    seeds = range(5)
    noop = evaluate(PRODUCING_BASELINES["noop"], ORDERING_BASELINES["even_split"], seeds=seeds, horizon=100)
    good = evaluate(producing_restore_when_low(0.6), ordering_even_split, seeds=seeds, horizon=100)
    # do-nothing should die early; the sensible policy should survive much longer
    assert noop["dead_rate"][0] > 0.5
    assert good["cycles"][0] > 2 * noop["cycles"][0]
    # and complete materially more consumer demand
    assert good["consumer_completed"][0] > 1.5 * noop["consumer_completed"][0]


def test_all_baselines_run():
    for pname, pf in PRODUCING_BASELINES.items():
        for oname, of in ORDERING_BASELINES.items():
            env = TokenomicsAECEnv(horizon=20)
            run_episode(env, pf, of, seed=1)
