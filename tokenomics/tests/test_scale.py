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


def test_producing_obs_exposes_investment_economics():
    # ADR-0044: the mechanics block exposes the per-agent investment economics that
    # vary per scale seed — max-replenish (units/cycle), replenish cost/unit, and the
    # restoration / upgrade token costs. A None cost / no-replenish producer reads 0.
    import pytest
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=7)
    e = TokenomicsAECEnv(config=cfg, horizon=10)
    e.reset()
    sp = e.spec_
    # a non-primal producer carries every economic field (replenish + investment).
    pcfg = next(p for p in cfg.producer_configs if p.replenish_cost_per_unit is not None)
    pid = pcfg.id
    obs = e.observe(f"prod_{pid}")
    base = sp.producing_mechanics_start + sp.num_types + 2  # after deg + restore + upgrade
    assert obs[base + 0] == pytest.approx(sp.max_replenish[pid] / sp.inv_ref)
    assert obs[base + 1] == pytest.approx(pcfg.replenish_cost_per_unit / sp.token_ref)
    assert obs[base + 2] == pytest.approx(pcfg.restoration.cost / sp.token_ref)
    assert obs[base + 3] == pytest.approx(pcfg.upgrade.cost / sp.token_ref)


def test_producing_obs_exposes_inventory_expansion_economics():
    # ADR-0044: the mechanics block ends with the inventory-expansion economics
    # (units the limit grows + its token cost). A producer with no expansion (e.g. a
    # primal, inventory-free producer) reads 0 for both — the knob is absent.
    import pytest
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=7)
    e = TokenomicsAECEnv(config=cfg, horizon=10)
    e.reset()
    sp = e.spec_
    tail = sp.producing_mechanics_start + sp.num_types + 6  # last two block fields

    has_exp = next(p for p in cfg.producer_configs if p.inventory_expansion is not None)
    obs = e.observe(f"prod_{has_exp.id}")
    assert obs[tail + 0] == pytest.approx(has_exp.inventory_expansion.increases / sp.inv_ref)
    assert obs[tail + 1] == pytest.approx(has_exp.inventory_expansion.cost / sp.token_ref)

    no_exp = next(p for p in cfg.producer_configs if p.inventory_expansion is None)
    obs0 = e.observe(f"prod_{no_exp.id}")
    assert obs0[tail + 0] == 0.0
    assert obs0[tail + 1] == 0.0


def test_two_scale_seeds_expose_different_mechanics_in_obs():
    # Regression for the generalization root cause (plan #2): two scale seeds share
    # obs/action DIMS (one net spans the suite) but differ in producer mechanics —
    # degradation, restoration amount, max-replenish. Before ADR-0044 those were
    # hidden state, so the same obs vector meant a different decision problem and the
    # policy memorized one operating point. The obs mechanics block must now DIFFER.
    import numpy as np
    from tokenomics.eval.scale import build_scale_config

    cfg_a, cfg_b = build_scale_config(seed=1), build_scale_config(seed=2)
    assert _dims(cfg_a) == _dims(cfg_b)  # same shape: one net spans both

    ea, eb = TokenomicsAECEnv(config=cfg_a, horizon=10), TokenomicsAECEnv(config=cfg_b, horizon=10)
    ea.reset(); eb.reset()
    sp = ea.spec_
    pid = sp.producer_ids[0]
    block = slice(sp.producing_mechanics_start, sp.producing_mechanics_start + sp.producing_mechanics_len)

    ma = np.asarray(ea.observe(f"prod_{pid}"))[block]
    mb = np.asarray(eb.observe(f"prod_{pid}"))[block]
    assert ma.shape == mb.shape
    assert not np.allclose(ma, mb)  # mechanics are no longer hidden seed state


def test_two_scale_seeds_expose_different_mechanics_in_global_state():
    # Plan item 1 (critic mechanics state): the producing OBS got the mechanics block
    # (ADR-0044) but the centralized-critic GLOBAL STATE did not, so under topology DR
    # the critic sees the same global vector for two seeds with different return
    # dynamics -> noisy value targets -> weak advantages -> the producing policy does
    # not transfer. The critic must see the per-producer seed-varying mechanics too.
    import numpy as np
    from tokenomics.eval.scale import build_scale_config

    cfg_a, cfg_b = build_scale_config(seed=1), build_scale_config(seed=2)
    assert _dims(cfg_a) == _dims(cfg_b)  # same shape: one critic spans both

    ea, eb = TokenomicsAECEnv(config=cfg_a, horizon=10), TokenomicsAECEnv(config=cfg_b, horizon=10)
    ea.reset(); eb.reset()
    sp = ea.spec_
    block = slice(sp.global_mechanics_start, sp.global_mechanics_start + sp.global_mechanics_len)

    ga = np.asarray(ea.global_state())[block]
    gb = np.asarray(eb.global_state())[block]
    assert ga.shape == gb.shape
    assert not np.allclose(ga, gb)  # critic mechanics are no longer hidden seed state


def test_global_state_mechanics_block_matches_producer_economics():
    # Plan item 1: lock the per-producer critic mechanics layout. Each producer's
    # 9-field block (global_mechanics_len) is: summed degradation, restoration amount,
    # upgrade amount, max-replenish, replenish cost, restoration cost, upgrade cost,
    # inventory-expansion increases, its cost — normalized like the obs block.
    import pytest
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=7)
    e = TokenomicsAECEnv(config=cfg, horizon=10)
    e.reset()
    sp = e.spec_
    gs = e.global_state()
    pcfg = next(p for p in cfg.producer_configs if p.replenish_cost_per_unit is not None)
    pid = pcfg.id
    j = sp.producer_ids.index(pid)
    base = sp.global_mechanics_start + j * sp.global_mechanics_len
    view = e.system.producing_agents[pid].view()
    assert gs[base + 0] == pytest.approx(sum(view.degradation.values()) / sp.cap_ref)
    assert gs[base + 1] == pytest.approx(view.restoration / sp.cap_ref)
    assert gs[base + 2] == pytest.approx(view.upgrade / sp.cap_ref)
    assert gs[base + 3] == pytest.approx(sp.max_replenish[pid] / sp.inv_ref)
    assert gs[base + 4] == pytest.approx(pcfg.replenish_cost_per_unit / sp.token_ref)
    assert gs[base + 5] == pytest.approx(pcfg.restoration.cost / sp.token_ref)
    assert gs[base + 6] == pytest.approx(pcfg.upgrade.cost / sp.token_ref)
    assert gs[base + 7] == pytest.approx(pcfg.inventory_expansion.increases / sp.inv_ref)
    assert gs[base + 8] == pytest.approx(pcfg.inventory_expansion.cost / sp.token_ref)


def test_return_tracks_fulfilment_at_scale_so_fulfilling_beats_doing_nothing():
    # Finding B (ADR-0042) acceptance bar: at a DEEP scale (depth 4 — where the old
    # reward made do-nothing the return-maximizer) a fulfilling baseline must earn a
    # strictly higher return than noop. Return now == 10*units − per-producer holding,
    # so survival+fulfilment beats the quick-dying do-nothing policy.
    from tokenomics.eval.baselines import producing_obs_stock_aware, producing_noop, ordering_proportional
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.scale import build_scale_config

    for seed in (0, 1):
        cfg = build_scale_config(seed=7, depth=4)
        fulfilling = run_episode(TokenomicsAECEnv(config=cfg, horizon=60),
                                 producing_obs_stock_aware(0.6), ordering_proportional, seed)
        nothing = run_episode(TokenomicsAECEnv(config=cfg, horizon=60),
                              producing_noop, ordering_proportional, seed)
        assert fulfilling["consumer_completed_units"] > 0
        assert fulfilling["return"] > nothing["return"]
