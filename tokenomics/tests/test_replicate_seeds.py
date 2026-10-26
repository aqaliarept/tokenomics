"""Random-baseline replicates (design D5, add-measurement-gate-harness; gate-harness
"Random-baseline replicates"): hash-derived action seeds, `run_episode(action_seed=...)`,
the replicate mean, and the `blend_0.5` roster router.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tokenomics.domain.config import load_config
from tokenomics.env import TokenomicsAECEnv
from tokenomics.eval import seeding
from tokenomics.eval.baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    Policy,
    ordering_blend,
)
from tokenomics.eval.runner import run_episode
from tokenomics.eval.seeding import replicate_mean, replicate_seed

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HORIZON = 8


def _random_demand_env() -> TokenomicsAECEnv:
    return TokenomicsAECEnv(config=load_config(FIXTURES / "random_demand.json"), horizon=HORIZON)


def _recording(policy: Policy, log: list[np.ndarray]) -> Policy:
    def wrapped(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        action = policy(env, agent)
        log.append(np.asarray(action).copy())
        return action
    return wrapped


def _random_run(action_seed: int | None, seed: int = 7) -> tuple[dict, list[np.ndarray]]:
    log: list[np.ndarray] = []
    rand = _recording(PRODUCING_BASELINES["random"], log)
    stats = run_episode(_random_demand_env(), rand, rand, seed=seed, action_seed=action_seed)
    return stats, log


def test_replicate_seed_is_the_d5_hash() -> None:
    expected = int(hashlib.sha256(b"W|7|0").hexdigest()[:16], 16)
    assert replicate_seed("W", 7, 0) == expected
    assert 0 <= expected < 2**64
    # every component matters
    seeds = {replicate_seed(w, e, r) for w in ("W", "V") for e in (7, 8) for r in (0, 1, 2)}
    assert len(seeds) == 12


def test_action_seed_is_independent_of_the_event_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scenario: replicate 0 of episode seed 7 on world W seeds the action RNG from the hash
    # of (W, 7, 0), not from 7 — while reset (the event/demand RNG) still gets 7.
    hashed = replicate_seed("W", 7, 0)
    assert hashed != 7
    env = _random_demand_env()
    reset_seeds: list[int | None] = []
    action_seeds: list[int] = []
    real_reset = env.reset

    def spy_reset(seed: int | None = None, options: dict | None = None) -> None:
        reset_seeds.append(seed)
        real_reset(seed=seed, options=options)

    monkeypatch.setattr(env, "reset", spy_reset)
    for a in env.possible_agents:
        space = env.action_space(a)
        real_seed = space.seed
        monkeypatch.setattr(space, "seed",
                            lambda s, _real=real_seed: (action_seeds.append(s), _real(s))[1])
    rand = PRODUCING_BASELINES["random"]
    run_episode(env, rand, rand, seed=7, action_seed=hashed)
    assert reset_seeds == [7]
    assert action_seeds and set(action_seeds) == {hashed}


def test_action_seed_varies_actions_not_demand() -> None:
    legacy_stats, legacy_actions = _random_run(None)
    rep_stats, rep_actions = _random_run(replicate_seed("W", 7, 0))
    # demand/event stream unchanged: same digest of the (seeded, B9) demand schedule
    assert rep_stats["demand_digest"] == legacy_stats["demand_digest"]
    # the random policy's actions do change with the action seed
    assert len(legacy_actions) > 0
    assert any(not np.array_equal(a, b) for a, b in zip(legacy_actions, rep_actions))
    # and the replicate is itself reproducible
    again_stats, again_actions = _random_run(replicate_seed("W", 7, 0))
    assert again_stats == rep_stats
    assert all(np.array_equal(a, b) for a, b in zip(rep_actions, again_actions))


def test_default_action_seed_is_todays_coupling() -> None:
    # None == seeding the action spaces with the event seed (legacy goldens pin the rest).
    none_stats, none_actions = _random_run(None)
    same_stats, same_actions = _random_run(7)
    assert none_stats == same_stats
    assert all(np.array_equal(a, b) for a, b in zip(none_actions, same_actions))


def test_replicate_seed_collision_with_event_seed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Collide:
        def hexdigest(self) -> str:
            return "0" * 64  # -> seed 0, equal to event seed 0

    monkeypatch.setattr(seeding.hashlib, "sha256", lambda _b: _Collide())
    with pytest.raises(ValueError, match="event seed"):
        replicate_seed("W", 0, 0)


def test_replicates_are_averaged_not_maximised() -> None:
    # Scenario: three replicates deliver 4, 10 and 7 units -> per-episode value 7.
    assert replicate_mean([4, 10, 7]) == 7
    with pytest.raises(ValueError):
        replicate_mean([])


def test_blend_half_is_a_registered_ordering_baseline() -> None:
    blend = ORDERING_BASELINES["blend_0.5"]
    env = TokenomicsAECEnv(horizon=HORIZON)
    env.reset(seed=0)
    halved = 0
    for agent in env.agent_iter(max_iter=len(env.possible_agents) * 4):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if env._role[agent][0] == "order":
            action = blend(env, agent)
            np.testing.assert_array_equal(action, ordering_blend(0.5)(env, agent))
            # the midpoint: exactly half the proportional logits (even_split is all zeros)
            prop = ORDERING_BASELINES["proportional"](env, agent)
            np.testing.assert_allclose(action, 0.5 * prop)
            halved += int(np.any(prop != 0))
            env.step(action)
        else:
            env.step(PRODUCING_BASELINES["stock_aware"](env, agent))
    assert halved > 0  # non-trivial: some step had non-zero proportional logits
