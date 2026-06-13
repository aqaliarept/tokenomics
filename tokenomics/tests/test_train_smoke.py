"""Smoke test for the PPO trainer — runs a couple of tiny iterations and checks
the loop completes and returns per-iteration reward history. Skipped if torch is
not installed (it's an optional `train` extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_train_runs_two_iters():
    from tokenomics.train.smoke_ppo import train

    history = train(iters=2, steps=128, horizon=40, seed=0, log=lambda *a, **k: None)
    assert len(history) == 2
    assert all(isinstance(r, float) for r in history)
