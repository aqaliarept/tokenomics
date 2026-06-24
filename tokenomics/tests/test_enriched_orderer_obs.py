"""EXP-0006A: the enriched orderer observation (M2-v3-obs contract).

A versioned, opt-in obs (checkpoint-schema bump) so M2-v1 / existing v1 checkpoints
stay valid. The env only NORMALIZES the domain's OrdererEconomyView; the dims here
must stay in lockstep with the encoder (test D2 pins that).
"""

from __future__ import annotations

import numpy as np
from pettingzoo.test import api_test

from tokenomics.domain import build_system, load_config
from tokenomics.domain.common import from_consumer_id
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.aec_env import DEFAULT_CONFIG
from tokenomics.env.encode import encode_ordering_enriched
from tokenomics.env.spaces import (
    OBS_CLIP,
    ORDERER_PP_FEATURES,
    ORDERER_PRODUCER_FEATURES,
    ORDERER_PRODUCT_FEATURES,
    ORDERER_SLOT_SCALARS,
    build_spec,
    config_fingerprint,
)


def test_enriched_ordering_dim_is_versioned_and_v1_unchanged():
    cfg = load_config(DEFAULT_CONFIG)
    v1 = build_spec(cfg)  # default version
    v2 = build_spec(cfg, orderer_obs_version=2)

    P, K, Q = v2.num_producers, v2.k, v2.num_order_products
    # v1 is the legacy capacity-blind layout (ADR-0029) — must not move.
    assert v1.orderer_obs_version == 1
    assert v1.ordering_obs_dim == K * (3 + Q) + P + 1

    expected = (
        K * (ORDERER_SLOT_SCALARS + Q + ORDERER_PRODUCT_FEATURES + P * ORDERER_PP_FEATURES)
        + P * ORDERER_PRODUCER_FEATURES
        + 1
    )
    assert v2.orderer_obs_version == 2
    assert v2.ordering_obs_dim == expected
    assert v2.ordering_obs_space().shape == (expected,)
    assert v2.ordering_obs_space().dtype == np.float32
    # enriching the orderer must not touch the producer obs.
    assert v2.producing_obs_dim == v1.producing_obs_dim


def test_enriched_encoder_matches_dim_normalized_and_deterministic():
    cfg = load_config(DEFAULT_CONFIG)
    spec = build_spec(cfg, orderer_obs_version=2)
    system = build_system(cfg)
    system.start_ordering()
    view = system.orderer_economy_view(from_consumer_id("c1"))

    vec = encode_ordering_enriched(spec, view, cycle_progress=0.25)
    assert vec.shape == (spec.ordering_obs_dim,)
    assert vec.dtype == np.float32
    # normalized + clipped to [0, OBS_CLIP], all finite.
    assert np.isfinite(vec).all()
    assert (vec >= 0.0).all() and (vec <= OBS_CLIP).all()
    # unlike the capacity-blind v1, the enriched obs carries real producer state.
    assert vec.sum() > 0.0
    # first slot is active (a live consumer order is present).
    assert vec[0] == 1.0
    # deterministic for the same domain state.
    again = encode_ordering_enriched(spec, view, cycle_progress=0.25)
    assert np.array_equal(vec, again)
    # trailing cycle-progress scalar is last.
    assert vec[-1] == 0.25


def test_env_serves_enriched_obs_under_flag_without_touching_v1():
    cfg = load_config(DEFAULT_CONFIG)
    v1_dim = build_spec(cfg).ordering_obs_dim
    v2_dim = build_spec(cfg, orderer_obs_version=2).ordering_obs_dim

    # default env keeps the legacy v1 ordering obs (no regression for M2-v1).
    default_env = TokenomicsAECEnv()
    order_agent = next(a for a in default_env.possible_agents if a.startswith("order_"))
    assert default_env.observation_space(order_agent).shape == (v1_dim,)

    # opt-in enriched env serves the v2 obs to ordering agents...
    env = TokenomicsAECEnv(enriched_orderer_obs=True)
    assert env.observation_space(order_agent).shape == (v2_dim,)
    env.reset(seed=0)
    # drive one cycle so an ordering agent is actually selected and observed.
    for agent in env.agent_iter(max_iter=len(env.possible_agents) * 2):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        obs = env.observe(agent)
        expected = v2_dim if agent.startswith("order_") else env.spec_.producing_obs_dim
        assert obs.shape == (expected,)
        env.step(env.action_space(agent).sample())
        if agent.startswith("order_"):
            break

    # ...and the producer obs is unchanged by the flag.
    assert env.observation_space(env._prod_agents[0]).shape == (
        env.spec_.producing_obs_dim,
    )


def test_enriched_env_passes_pettingzoo_api():
    api_test(TokenomicsAECEnv(enriched_orderer_obs=True, horizon=20), num_cycles=5)


def test_config_fingerprint_distinguishes_orderer_obs_version():
    """A v2 checkpoint on the same economy must not collide with a v1 checkpoint —
    otherwise the compatibility check would treat them as interchangeable."""
    cfg = load_config(DEFAULT_CONFIG)
    v1 = config_fingerprint(cfg)  # default
    v2 = config_fingerprint(cfg, orderer_obs_version=2)
    assert v1 != v2
    assert config_fingerprint(cfg) == v1  # stable / deterministic
    assert config_fingerprint(cfg, orderer_obs_version=2) == v2
