"""EXP-0006A cloud-run plumbing: the M2 trainer/eval must be able to drive the
*enriched* (orderer_obs_version 2) observation end to end, else a cloud run would
silently retrain M2-v1. These tests pin the flag threading through the public
training/eval entry points (kept tiny — search/horizon shrunk to stay local).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.runner import evaluate
from tokenomics.eval.scale import build_scale_config
from tokenomics.env.spaces import build_spec
from tokenomics.train.orderer_improve import collect_orderer_samples


def _v1_v2_dims(cfg):
    return (
        build_spec(cfg, orderer_obs_version=1).ordering_obs_dim,
        build_spec(cfg, orderer_obs_version=2).ordering_obs_dim,
    )


def test_evaluate_enriched_feeds_orderer_the_v2_observation():
    cfg = build_scale_config(0, depth=3)
    v1, v2 = _v1_v2_dims(cfg)
    assert v2 != v1  # precondition: enriched obs is a distinct width

    seen: list[int] = []

    def spy_order(env, agent):
        seen.append(env.observe(agent).shape[0])
        return ORDERING_BASELINES["proportional"](env, agent)

    evaluate(
        PRODUCING_BASELINES["stock_aware"], spy_order,
        seeds=range(1), horizon=5, config=cfg, enriched_orderer_obs=True,
    )
    assert seen and all(d == v2 for d in seen)


def test_label_gen_enriched_records_v2_observation():
    """The DAgger label-gen rollout must build enriched envs, so the student is
    fit on (and sized to) the v2 obs — not v1."""
    cfg = build_scale_config(0, depth=3)
    _, v2 = _v1_v2_dims(cfg)
    samples = collect_orderer_samples(
        cfg,
        PRODUCING_BASELINES["stock_aware"],
        {"proportional": ORDERING_BASELINES["proportional"]},
        n_samples=1, sample_scale=1.0, lookahead=1, seeds=[0], horizon=4,
        enriched_orderer_obs=True,
    )
    assert samples and all(s.obs.shape[0] == v2 for s in samples)


def test_evaluate_defaults_to_legacy_v1_observation():
    cfg = build_scale_config(0, depth=3)
    v1, _ = _v1_v2_dims(cfg)
    seen: list[int] = []

    def spy_order(env, agent):
        seen.append(env.observe(agent).shape[0])
        return ORDERING_BASELINES["proportional"](env, agent)

    evaluate(
        PRODUCING_BASELINES["stock_aware"], spy_order,
        seeds=range(1), horizon=5, config=cfg,
    )
    assert seen and all(d == v1 for d in seen)
