"""Scale test (NEXT STEPS stage 5): a generator for a larger economy — 8-16 products,
10-20 producers, 3-5 consumers, 3-5 level supply chains, several eligible producers per
top product — with episode-seeded demand and bounded, seed-driven configuration
variation that KEEPS obs/action dims fixed (so one trained net spans the whole suite)."""

from __future__ import annotations

from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.spaces import build_spec, config_fingerprint


def _dims(cfg):
    sp = build_spec(cfg)
    return (sp.producing_obs_dim, sp.producing_action_dim,
            sp.ordering_obs_dim, sp.ordering_action_dim, sp.global_state_dim)


def _layer(pid: int) -> int:
    return pid // 10


def _chain_depth(cfg, product: int) -> int:
    """Longest recipe chain (levels) rooted at `product`, 1 for a primal product."""
    inputs = cfg.inputs_of(product)
    if not inputs:
        return 1
    return 1 + max(_chain_depth(cfg, i) for i in inputs)


def test_build_scale_config_meets_size_bounds_and_validates():
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=0)
    cfg.validate()  # a valid economy the env can run
    assert 8 <= len(cfg.process_sheets) <= 16
    assert 10 <= len(cfg.producer_configs) <= 20
    assert 3 <= len(cfg.consumers) <= 5


def test_scale_config_supply_chain_is_at_least_three_levels_deep():
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=0)
    deepest = max(_chain_depth(cfg, ps.product) for ps in cfg.process_sheets)
    assert deepest >= 3  # a consumed product reaches >=3 production levels down


def test_scale_config_dims_are_seed_invariant():
    from tokenomics.eval.scale import build_scale_config

    # the whole point: values/topology vary with the seed, dims do NOT (fixed envelope)
    d = {_dims(build_scale_config(seed=s)) for s in (1, 2, 3, 7)}
    assert len(d) == 1


def test_scale_config_varies_with_seed_but_is_deterministic():
    from tokenomics.eval.scale import build_scale_config

    a = config_fingerprint(build_scale_config(seed=1))
    b = config_fingerprint(build_scale_config(seed=2))
    a_again = config_fingerprint(build_scale_config(seed=1))
    assert a == a_again        # same seed -> identical economy (reproducible)
    assert a != b              # different seed -> different economy (bounded variation)


def test_scale_config_has_several_eligible_producers_per_top_product():
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=0)
    top = max(_layer(ps.product) for ps in cfg.process_sheets)
    top_products = [ps.product for ps in cfg.process_sheets if _layer(ps.product) == top]
    for prod in top_products:
        declarers = [p for p in cfg.producer_configs if prod in p.produces]
        assert len(declarers) >= 2  # routing has a real choice (competition)


def test_scale_config_consumers_demand_top_layer_with_randomized_episode_demand():
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=0)
    top = max(_layer(ps.product) for ps in cfg.process_sheets)
    for c in cfg.consumers:
        assert all(_layer(p) == top for p in c.product_pool)  # demand the finished goods
        assert c.is_randomized  # episode-seeded demand (B9), not a fixed scripted order


def test_scale_env_runs_an_episode_with_obs_baseline():
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.scale import build_scale_config

    env = TokenomicsAECEnv(config=build_scale_config(seed=0), horizon=20)
    s = run_episode(env, PRODUCING_BASELINES["obs_stock_aware"],
                    ORDERING_BASELINES["even_split"], seed=0)
    assert s["cycles"] > 0
