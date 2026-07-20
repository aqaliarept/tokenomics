"""EXP-0037 value-routed orderer: the trainable residual scorer + deploy adapter.

torch-guarded (skips where torch is absent). Kept tiny — synthetic samples and a 1-task
seconds-long smoke — so it stays local per the compute rule; the real s29/s77/s45 gate
runs on a pod.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.curriculum import TaskRecord
from tokenomics.train.value_router import (
    PRODUCER_FEATURES,
    ResidualScoreModel,
    RouterSample,
    fit_residual_scorer,
    run_value_router,
    to_score_fn,
)


def _scale_task(seed: int, depth: int, episode_seed: int = 0) -> TaskRecord:
    return TaskRecord(
        task_id=f"scale-d{depth}-s{seed}", task_family="scale",
        generator_version="test", serializer_version="test",
        world_seed=seed, episode_seed=episode_seed,
        world_spec={"generator": "build_scale_config", "params": {"depth": depth}},
    )


def test_scorer_learns_the_marginal_value_objective():
    """The whole point of slice 4: the shared per-producer-row scorer can fit the marginal-value
    labels. A producer whose bump gains units must end up scored above one whose bump loses. We
    train on synthetic rows where feature 1 (throughput) is uninformative and feature 5 (input
    availability) carries the value, so a throughput-only reader could not fit it."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(64):
        rows = rng.standard_normal((3, PRODUCER_FEATURES)).astype(np.float32)
        mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        label = rows[:, 5] * 3.0  # value lives in the input-availability feature
        samples.append(RouterSample(rows, label.astype(np.float32), mask))

    model = ResidualScoreModel()
    losses = fit_residual_scorer(model, samples, epochs=200)
    assert losses[-1] < 0.25 * losses[0]  # masked MSE drops substantially -> it learned


def test_fit_handles_samples_with_different_producer_counts():
    """The gate spans tasks of different topology (s29 has 2 producers/layer, s77/s45 have 3), so
    per-sample producer-row matrices have different P. The fit must batch them anyway (the model
    is producer-count agnostic) — pad + mask, never assume a rectangular [N, P, F]."""
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(8):  # P = 2
        samples.append(RouterSample(rng.standard_normal((2, PRODUCER_FEATURES)).astype(np.float32),
                                    rng.standard_normal(2).astype(np.float32),
                                    np.ones(2, dtype=np.float32)))
    for _ in range(8):  # P = 3
        samples.append(RouterSample(rng.standard_normal((3, PRODUCER_FEATURES)).astype(np.float32),
                                    rng.standard_normal(3).astype(np.float32),
                                    np.ones(3, dtype=np.float32)))
    model = ResidualScoreModel()
    losses = fit_residual_scorer(model, samples, epochs=3)  # must not raise
    assert len(losses) == 3 and all(np.isfinite(losses))


def test_score_fn_is_zero_for_ineligible_and_absent_streams():
    """Deploy adapter contract: the residual it emits must be shaped like an ordering action and
    only touch eligible (stream, producer) cells — everything else stays 0 so the router falls
    back to blend_0.5 there."""
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(0, depth=3)
    env = TokenomicsAECEnv(config=cfg, horizon=100)
    env.reset(seed=0)
    prod = PRODUCING_BASELINES["stock_aware"]
    order_agent = None
    for agent in env.agent_iter(max_iter=10000):
        role = env._role[agent][0]
        if role == "order":
            order_agent = agent
            break
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        env.step(prod(env, agent))
    assert order_agent is not None

    score_fn = to_score_fn(ResidualScoreModel())
    residual = score_fn(env, order_agent)
    assert residual.shape == (env.spec_.ordering_action_dim,)

    # every nonzero cell must be an eligible (stream, producer) pair
    _, aid = env._role[order_agent]
    view = env.system.ordering_agents[aid].view(env.system.producer_infos)
    mat = residual.reshape(env.spec_.k_stream, env.spec_.num_producers)
    for k in range(env.spec_.k_stream):
        oids = sorted(view.incoming)
        elig = view.eligible[oids[k]] if k < len(oids) else {}
        for pi, pid in enumerate(env.spec_.producer_ids):
            if pid not in elig:
                assert mat[k, pi] == 0.0


def test_run_value_router_smoke_writes_gate_layout(tmp_path):
    """1-task seconds-long smoke: collect -> fit -> deploy -> gate end to end, and emit the EXP
    layout. Pipeline check only (not evidence about the model): finite deployed units per task."""
    tasks = [_scale_task(0, depth=2)]
    rows = run_value_router(
        tasks, prod=PRODUCING_BASELINES["stock_aware"], delta=1.0, lookahead=3,
        eval_seeds=1, horizon=25, epochs=5, clip=2.0, out_dir=str(tmp_path),
        log=lambda *a, **k: None,
    )
    assert len(rows) == 1
    r = rows[0]
    assert np.isfinite(r["learned_cu"]) and np.isfinite(r["blend_cu"])
    assert "beats_floor" in r
    assert (tmp_path / "spec.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "residual_scorer.pt").exists()


def test_run_value_router_is_reproducible_under_a_fixed_seed(tmp_path):
    """The gate verdict must be reproducible: with the same `seed`, re-running the fit gives the
    same deployed `learned_cu` (and therefore the same 2-of-3 stop count). Model init was the one
    unseeded source of variance in the original run — pin it here."""
    tasks = [_scale_task(0, depth=2)]
    kw = dict(prod=PRODUCING_BASELINES["stock_aware"], delta=1.0, lookahead=3,
              eval_seeds=1, horizon=25, epochs=5, clip=2.0, log=lambda *a, **k: None)
    a = run_value_router(tasks, seed=7, out_dir=str(tmp_path / "a"), **kw)
    b = run_value_router(tasks, seed=7, out_dir=str(tmp_path / "b"), **kw)
    assert [r["learned_cu"] for r in a] == [r["learned_cu"] for r in b]


def test_run_value_router_persists_audit_artifacts(tmp_path):
    """Root-cause evidence must be auditable from the pulled dir, not just the pod stdout: the run
    persists the full per-epoch loss history and per-(task, seed, policy) gate rows, and records
    the model seed in spec.json (P2 review corrections)."""
    import csv
    import json

    tasks = [_scale_task(0, depth=2)]
    run_value_router(
        tasks, prod=PRODUCING_BASELINES["stock_aware"], delta=1.0, lookahead=3,
        eval_seeds=2, horizon=25, epochs=5, clip=2.0, seed=3, out_dir=str(tmp_path),
        log=lambda *a, **k: None,
    )
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["seed"] == 3
    assert spec["loss_first"] is not None and spec["loss_last"] is not None

    loss_rows = list(csv.DictReader((tmp_path / "loss_history.csv").open()))
    assert [int(r["epoch"]) for r in loss_rows] == list(range(5))  # one row per epoch

    gate_rows = list(csv.DictReader((tmp_path / "gate_rows.csv").open()))
    # one row per (task, seed, policy): 1 task x 2 seeds x {learned, blend_0.5}
    assert len(gate_rows) == 4
    assert {r["policy"] for r in gate_rows} == {"learned", "blend_0.5"}
    assert {int(r["seed"]) for r in gate_rows} == set(range(2)) or len({r["seed"] for r in gate_rows}) == 2
