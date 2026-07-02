"""PRODUCT_AGNOSTIC O2: learned orderer candidate-scorer contract (local).

Contract-only TDD (no training): the scorer maps a ProductStreamView to one score
per producer row, with shared weights so it is order-equivariant over producer rows
and runs on any producer count P. The deterministic masked allocator (O1) turns those
scores into conserved SubOrders. Real distillation/training is cloud work.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from tokenomics.env.aec_env import TokenomicsAECEnv
from tokenomics.eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
from tokenomics.eval.runner import run_episode


def _active_stream_view():
    """Drive a few cycles, return a product-stream view with a non-empty backlog and
    the env (for producer ids)."""
    env = TokenomicsAECEnv(horizon=40, product_stream_ordering=True, corrected_stream_k=True)
    env.reset(seed=0)
    # step a couple cycles so orders are in flight
    for _ in range(3):
        for agent in list(env.agents):
            if env.terminations[agent] or env.truncations[agent]:
                continue
            role = env._role[agent][0]
            pol = PRODUCING_BASELINES["stock_aware"] if role == "prod" else ORDERING_BASELINES["proportional"]
            if env.agent_selection == agent:
                env.step(pol(env, agent))
    return env, env.system.product_stream_view("c1__30")


def test_scorer_emits_one_score_per_producer() -> None:
    # Tracer: forward_view returns a 1-D score tensor of length P (one per producer
    # row), no fixed K*P head.
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel()
    scores = model.forward_view(view)
    assert scores.shape == (len(view.producers),)


def test_score_follows_producer_stats_not_position() -> None:
    # Row-order equivariance: swapping two producers' stats (keeping ids/sort order
    # fixed) swaps their scores — the shared row encoder scores content, not slot.
    import torch
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel().eval()
    pids = sorted(view.producers)
    a, b = pids[0], pids[-1]
    with torch.no_grad():
        base = model.forward_view(view)
    swapped = replace(
        view,
        producers={**view.producers, a: view.producers[b], b: view.producers[a]},
        producer_products={**view.producer_products, a: view.producer_products[b],
                           b: view.producer_products[a]},
    )
    with torch.no_grad():
        new = model.forward_view(swapped)
    ia, ib = pids.index(a), pids.index(b)
    assert torch.allclose(new[ia], base[ib]) and torch.allclose(new[ib], base[ia])


def test_runs_on_variable_producer_count() -> None:
    # No fixed head: the same model scores a view with a dropped producer; output
    # length tracks P at runtime.
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel()
    drop = sorted(view.producers)[0]
    fewer = replace(
        view,
        producers={k: v for k, v in view.producers.items() if k != drop},
        producer_products={k: v for k, v in view.producer_products.items() if k != drop},
    )
    assert model.forward_view(fewer).shape == (len(view.producers) - 1,)


def test_scores_bridge_to_masked_allocator() -> None:
    # O1+O2 integration: model scores + an eligibility mask from the view decode into
    # conserved SubOrders routed only to eligible producers.
    import torch
    from tokenomics.env.allocator import allocate_suborders
    from tokenomics.train.orderer_scorer import OrdererScoreModel

    env, view = _active_stream_view()
    model = OrdererScoreModel().eval()
    pids = model.producer_ids(view)
    with torch.no_grad():
        scores = model.forward_view(view).numpy()
    mask = np.array([view.producer_products[p].eligible > 0 for p in pids])
    if not mask.any():
        pytest.skip("no eligible producer in this stream snapshot")
    subs = allocate_suborders(pids, scores, mask, quantity=9, tokens=90)
    assert sum(s.quantity for s in subs) == 9 and sum(s.tokens for s in subs) == 90
    assert all(mask[pids.index(s.producer_id)] for s in subs)


def test_schema_and_determinism() -> None:
    # Schema gate: row feature width ties to the domain dataclasses (incl. the
    # active_investments 3-tuple); a checkpoint version exists; eval is deterministic.
    import dataclasses as dc
    import torch
    from tokenomics.domain.orderer_view import (
        ProducerProductStats, ProducerStats, ProductStats)
    from tokenomics.train.orderer_scorer import (
        ORDERER_SCORER_CHECKPOINT_VERSION, _PRODUCER_ROW_DIM, _PRODUCT_DIM,
        OrdererScoreModel)

    assert _PRODUCT_DIM == len(dc.fields(ProductStats))
    # ProducerStats has one 3-wide tuple field (active_investments).
    assert _PRODUCER_ROW_DIM == len(dc.fields(ProducerStats)) - 1 + 3 + len(dc.fields(ProducerProductStats))
    assert ORDERER_SCORER_CHECKPOINT_VERSION >= 1

    env, view = _active_stream_view()
    model = OrdererScoreModel().eval()
    with torch.no_grad():
        assert torch.allclose(model.forward_view(view), model.forward_view(view))


def test_scorer_policy_emits_v4_action_and_drives_episode() -> None:
    # Deploy path: the scorer policy emits a k_stream*P action (per-producer scores
    # tiled across slots) that the v4 env decodes via the masked allocator; an episode
    # runs end to end with the learned orderer.
    from tokenomics.train.orderer_scorer import OrdererScoreModel, scorer_ordering_policy

    env = TokenomicsAECEnv(horizon=30, product_stream_ordering=True, corrected_stream_k=True)
    pol = scorer_ordering_policy(OrdererScoreModel())
    res = run_episode(env, PRODUCING_BASELINES["stock_aware"], pol, seed=0)
    assert res["cycles"] > 0
    # one orderer action has the corrected width.
    env.reset(seed=0)
    oa = next(a for a in env.agents if a.startswith("order_"))
    while env.agent_selection != oa:
        a = env.agent_selection
        role = env._role[a][0]
        p = PRODUCING_BASELINES["stock_aware"] if role == "prod" else ORDERING_BASELINES["proportional"]
        env.step(p(env, a))
    assert pol(env, oa).shape == (env.spec_.k_stream * env.spec_.num_producers,)


def test_teacher_logits_rank_and_mask() -> None:
    # Teacher target tracks makeable throughput and masks ineligible producers.
    from tokenomics.train.orderer_scorer import teacher_logits

    env, view = _active_stream_view()
    pids, target, mask = teacher_logits(view)
    assert mask.tolist() == [view.producer_products[p].eligible > 0 for p in pids]
    # a producer that can make more units gets a >= target than one that can make fewer.
    mk = {p: view.producer_products[p].makeable_qty_now for p in pids}
    hi = max(pids, key=lambda p: mk[p]); lo = min(pids, key=lambda p: mk[p])
    if mk[hi] > mk[lo]:
        assert target[pids.index(hi)] >= target[pids.index(lo)]


def test_smoke_distill_roundtrip() -> None:
    # End-to-end smoke (local-allowed): collect a few samples, fit a few iters, the
    # checkpoint round-trips, and eval returns a ratio row per held-out topology.
    import tempfile, os
    from tokenomics.train.orderer_scorer import (
        collect_scorer_samples, fit_scorer, save_scorer, load_scorer,
        eval_scorer, OrdererScoreModel)

    samples = collect_scorer_samples([0], depth=3, horizon=12)
    assert samples
    model = fit_scorer(OrdererScoreModel(hidden=16), samples, iters=5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scorer.pt")
        save_scorer(model, path)
        reloaded = load_scorer(path)
    rows = eval_scorer(reloaded, depth=3, seeds=[6], eval_seeds=1, horizon=12)
    assert rows and "ratio" in rows[0] and rows[0]["producers"] >= 1


def test_collect_scorer_samples_honors_explicit_configs() -> None:
    # A replay/curriculum caller passes explicit Configurations (e.g. archived tasks) so the
    # orderer trains on exactly those topologies, not a build_scale_config(seed, depth) stand-in.
    from tokenomics.train.orderer_scorer import collect_scorer_samples
    from tokenomics.eval.scale import build_scale_config

    cfgs = [build_scale_config(0, depth=2), build_scale_config(1, depth=2)]
    samples = collect_scorer_samples([0, 1], depth=5, horizon=12, configs=cfgs)  # depth ignored
    assert samples


def test_collect_scorer_samples_rejects_misaligned_configs() -> None:
    from tokenomics.train.orderer_scorer import collect_scorer_samples
    from tokenomics.eval.scale import build_scale_config

    with pytest.raises(ValueError):
        collect_scorer_samples([0, 1], depth=2, horizon=12, configs=[build_scale_config(0, depth=2)])
