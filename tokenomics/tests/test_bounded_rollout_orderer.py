from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tokenomics.env.future_fork import continuation_seed
from tokenomics.eval import bounded_rollout_orderer as bro


def test_future_seed_default_preserves_exp0040_and_experiment_tag_isolated():
    default = continuation_seed("task", 17, 3)
    explicit = continuation_seed("task", 17, 3, experiment="EXP-0040")
    other = continuation_seed("task", 17, 3, experiment="EXP-0042")
    assert default == explicit
    assert other != default
    with pytest.raises(ValueError, match="non-empty"):
        continuation_seed("task", 17, 3, experiment="")


def test_resolve_stream_accepts_only_confident_unique_override():
    gains = np.array([[3.0, 0.0], [2.0, 0.0], [3.0, -1.0], [2.0, 0.0]])
    got = bro.resolve_stream(gains, (4, 7))
    assert got.resolved
    assert got.choice == 4
    assert got.reason == "confident_override"
    assert got.min_lcb > 0


def test_resolve_stream_falls_back_on_exact_zero_and_keeps_noise_unresolved():
    zero = bro.resolve_stream(np.zeros((4, 2)), (0, 1))
    assert zero.resolved and zero.choice is None and zero.reason == "deterministic_zero"

    noisy = bro.resolve_stream(
        np.array([[2.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]), (0, 1))
    assert not noisy.resolved
    assert noisy.choice is None


def test_softmax_gap_is_probability_not_logit_gap():
    assert bro._softmax_gap([0.0, 0.0]) == pytest.approx(0.0)
    assert bro._softmax_gap([2.0, 0.0]) > 0.25


class _FakeEnv:
    def __init__(self, base_logits=(0.0, 0.0)):
        self.current_cycle = 5
        self.spec_ = SimpleNamespace(
            k_stream=1, num_producers=2, producer_ids=["p0", "p1"])
        self._role = {"order": ("order", "a0")}
        view = SimpleNamespace(
            incoming={"o0": SimpleNamespace()}, eligible={"o0": ["p0", "p1"]})
        self.system = SimpleNamespace(
            ordering_agents={"a0": SimpleNamespace(
                view=lambda _producer_infos: view)}, producer_infos={})
        self.base_logits = np.asarray(base_logits, dtype=np.float32)


def _base(env, _agent):
    return env.base_logits.copy()


def test_policy_stops_at_first_checkpoint_and_applies_confident_bump():
    calls = []

    def scorer(_env, _agent, streams, cid):
        calls.append(cid)
        assert [s.stream_index for s in streams] == [0]
        return {0: np.array([2.0, 0.0])}

    pol = bro.BoundedScenarioOrderer(
        task_id="task", episode_seed=1, prod=lambda *_: None, depth=1,
        base=_base, scenario_scorer=scorer)
    action = pol(_FakeEnv(), "order")
    assert np.array_equal(action, np.array([1.0, 0.0], dtype=np.float32))
    assert calls == [0, 1, 2, 3]
    assert pol.telemetry.override_streams == 1
    assert pol.telemetry.stream_scenarios == 4


def test_policy_uses_full_budget_then_falls_back_when_unresolved():
    def scorer(_env, _agent, _streams, cid):
        return {0: np.array([2.0 if cid % 2 == 0 else -2.0, 0.0])}

    pol = bro.BoundedScenarioOrderer(
        task_id="task", episode_seed=1, prod=lambda *_: None, depth=2,
        base=_base, scenario_scorer=scorer)
    action = pol(_FakeEnv(), "order")
    assert np.array_equal(action, np.zeros(2, dtype=np.float32))
    assert pol.telemetry.unresolved_fallback_streams == 1
    assert pol.telemetry.stream_scenarios == 16


def test_policy_probability_gap_skips_rollout_entirely():
    def should_not_run(*_args):
        raise AssertionError("scenario scorer should be gated out")

    pol = bro.BoundedScenarioOrderer(
        task_id="task", episode_seed=1, prod=lambda *_: None, depth=1,
        base=_base, scenario_scorer=should_not_run)
    action = pol(_FakeEnv(base_logits=(2.0, 0.0)), "order")
    assert np.array_equal(action, np.array([2.0, 0.0], dtype=np.float32))
    assert pol.telemetry.ambiguity_gated_streams == 1
    assert pol.telemetry.search_actions == 0


def _rows(search_units_by_depth):
    rows = []
    for task in ("a", "b", "c"):
        for seed in (1, 2):
            rows.append({
                "task_id": task, "seed": seed, "arm": bro.FLOOR, "depth": 0,
                "consumer_units": 100.0, "dead": 0, "override_streams": 0,
                "rollout_calls": 0, "wall_seconds": 1.0})
            for depth, units in search_units_by_depth[task].items():
                rows.append({
                    "task_id": task, "seed": seed, "arm": f"search_d{depth}",
                    "depth": depth, "consumer_units": units, "dead": 0,
                    "override_streams": 2, "rollout_calls": 10, "wall_seconds": 2.0})
    return rows


def test_artifact_gate_promotes_shallowest_passing_depth():
    result = bro.analyze_episode_rows(_rows({
        "a": {1: 110, 2: 120}, "b": {1: 108, 2: 120}, "c": {1: 102, 2: 120}}))
    assert result["verdict"] == "PASS_BOUNDED_SEARCH"
    assert result["promoted_depth"] == 1


def test_artifact_gate_rejects_one_bad_episode_even_when_means_win():
    rows = _rows({"a": {1: 120}, "b": {1: 120}, "c": {1: 120}})
    bad = next(r for r in rows if r["task_id"] == "c" and r["seed"] == 1 and r["depth"] == 1)
    bad["consumer_units"] = 80.0
    result = bro.analyze_episode_rows(rows)
    assert result["verdict"] == "NO_BOUNDED_SEARCH_GAIN"
    assert result["arm_metrics"][0]["worst_episode_relative"] == pytest.approx(-0.2)
