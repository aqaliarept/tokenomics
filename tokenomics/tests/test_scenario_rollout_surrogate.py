from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from tokenomics.domain.consumer import ScheduledConsumer
from tokenomics.domain.graph_view import build_economy_graph_view
from tokenomics.eval.graph_value_router_dump import encode_graph
from tokenomics.eval.scenario_rollout_surrogate import (
    FUTURE_FEATURES,
    ScenarioBatch,
    ScenarioGroup,
    ScenarioRelationalGNN,
    ScenarioRowMLP,
    ScenarioSample,
    future_product_features,
    simulate_adapter,
    train_model,
    validate_registry,
)
from tokenomics.eval import scenario_rollout_surrogate as srs

from test_graph_view import first_contested_decision


def _graph():
    env, aid, oid, elig = first_contested_decision()
    return encode_graph(build_economy_graph_view(env.system, aid, oid, list(elig)))


def test_registered_episode_blocks_are_disjoint():
    validate_registry()


def test_future_product_features_align_products_and_time_without_identity_feature():
    c0 = ScheduledConsumer("c0", [(20, 2), (21, 3), (20, 4)])
    c1 = ScheduledConsumer("c1", [(21, 5), (21, 6), (20, 7)])
    c0._cycle = c1._cycle = 1
    env = SimpleNamespace(system=SimpleNamespace(consumers={"c1": c1, "c0": c0}))
    got = future_product_features(env, [20, 21])
    assert got.shape == (2, FUTURE_FEATURES)
    # next position: both consumers request 21, total quantity 9, count 2
    assert got[1, 0] == pytest.approx(np.log1p(9))
    assert got[1, 1] == 2
    # following position: both request 20, total quantity 11, count 2
    assert got[0, 2] == pytest.approx(np.log1p(11))
    assert got[0, 3] == 2
    assert got[0, 0] == got[1, 2] == 0


def test_row_and_relational_models_score_one_value_per_candidate():
    graph = _graph()
    future = np.zeros((graph.node_features["product"].shape[0], FUTURE_FEATURES), np.float32)
    batch = ScenarioBatch([graph], [future])
    n = len(graph.candidate_producers)
    assert ScenarioRowMLP()(batch).shape == (n,)
    assert ScenarioRelationalGNN(hidden=16, layers=2)(batch).shape == (n,)


def test_training_smoke_runs_scenario_weighted_loss(monkeypatch):
    graph = _graph()
    group = _group(graph, 0, np.array([1.0] + [0.0] * (len(graph.candidate_producers) - 1)),
                   "confident_override", 0, 1.0)
    items = [(group, sample) for sample in group.samples[:2]]
    monkeypatch.setattr(srs, "MAX_EPOCHS", 2)
    monkeypatch.setattr(srs, "MIN_EPOCHS", 1)
    monkeypatch.setattr(srs, "PATIENCE", 1)
    model, run = train_model(
        "row_mlp_h64", items, items, model_seed=0, device="cpu")
    assert run["epochs"] in (1, 2)
    assert np.isfinite(run["best_validation_loss"])
    assert model(ScenarioBatch(
        [group.graph], [items[0][1].future])).shape == (len(graph.candidate_producers),)


def _group(graph, index: int, gains: np.ndarray, reason: str, choice, mean_gain: float,
           task: str = "task"):
    samples = [ScenarioSample(cid, np.zeros(
        (graph.node_features["product"].shape[0], FUTURE_FEATURES), np.float32), gains.copy())
               for cid in range(16)]
    return ScenarioGroup(
        graph=graph, task_id=task, split="validation", seed=1, cycle=2,
        agent="order", stream_index=index, order_id=f"o{index}",
        eligible_cols=(0, 1), authority_reason=reason, authority_choice=choice,
        authority_mean_gain=mean_gain, train_keep=True, samples=samples)


def test_offline_adapter_counts_shared_base_and_passes_by_pruning_zero_streams():
    graph = _graph()
    groups = [_group(graph, 0, np.array([2.0, 0.0]),
                     "confident_override", 0, 2.0)]
    groups += [_group(graph, i, np.zeros(2), "deterministic_zero", None, 0.0)
               for i in range(1, 5)]
    means, spreads = [], []
    for group in groups:
        for sample in group.samples:
            means.append(sample.gains.copy())
            spreads.append(np.zeros(2))
    result = simulate_adapter(groups, means, spreads, zero_abs=0.1, spread=0.0)
    assert result["exact_override_recall"] == 1.0
    assert result["value_weighted_override_recall"] == 1.0
    assert result["authority_rollout_calls"] == 44  # 4 shared bases + 5*2*4 bumps
    assert result["hybrid_rollout_calls"] == 12     # 4 bases + one real stream's 8 bumps
    assert result["real_rollout_reduction"] > 0.70
    assert result["gate_pass"]


def test_offline_adapter_rejects_a_false_zero_override_certificate():
    graph = _graph()
    group = _group(graph, 0, np.array([2.0, 0.0]),
                   "confident_override", 0, 2.0)
    means = [np.zeros(2) for _ in group.samples]
    spreads = [np.zeros(2) for _ in group.samples]
    result = simulate_adapter([group], means, spreads, zero_abs=0.1, spread=0.0)
    assert result["value_weighted_override_recall"] == 0.0
    assert not result["gate_pass"]


def test_offline_gate_cannot_hide_one_failed_task_in_the_pooled_population():
    graph = _graph()
    groups = [_group(graph, i, np.array([2.0, 0.0]),
                     "confident_override", 0, 2.0, task="large")
              for i in range(100)]
    groups += [_group(graph, 100 + i, np.zeros(2),
                      "deterministic_zero", None, 0.0, task="large")
               for i in range(400)]
    groups += [_group(graph, 999, np.array([2.0, 0.0]),
                      "confident_override", 0, 2.0, task="small")]
    means, spreads = [], []
    for group in groups:
        for sample in group.samples:
            means.append(np.zeros(2) if group.task_id == "small" else sample.gains.copy())
            spreads.append(np.zeros(2))
    result = simulate_adapter(groups, means, spreads, zero_abs=0.15, spread=0.0)
    assert result["exact_override_recall"] > 0.95
    assert result["value_weighted_override_recall"] > 0.99
    assert result["real_rollout_reduction"] > 0.70
    assert not result["gate_pass"]
    small = next(row for row in result["task_metrics"] if row["task_id"] == "small")
    assert small["value_weighted_override_recall"] == 0.0
