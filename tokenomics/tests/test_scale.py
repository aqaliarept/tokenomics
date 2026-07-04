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


def test_capacity_tight_params_default_is_byte_identical():
    # Track A capacity axes must be OPT-IN: passing the defaults reproduces the exact
    # existing economy so the champion / prior experiments stay comparable.
    from tokenomics.eval.scale import build_scale_config

    for s in (0, 3, 7):
        base = config_fingerprint(build_scale_config(seed=s))
        opt_in = config_fingerprint(build_scale_config(
            seed=s, quantity_max=None, capacity_scale=1.0, degradation=None))
        assert base == opt_in


def test_primal_capacity_scale_default_is_byte_identical():
    # The layer-0-only tightening axis (Track A round 2) must be opt-in: the default
    # 1.0 draws the exact prior rng stream, so nothing else drifts.
    from tokenomics.eval.scale import build_scale_config

    for s in (0, 3, 7):
        base = config_fingerprint(build_scale_config(seed=s))
        opt_in = config_fingerprint(build_scale_config(seed=s, primal_capacity_scale=1.0))
        assert base == opt_in


def test_primal_capacity_scale_tightens_only_layer_zero():
    # scale down ONLY the primal (layer-0) producers; upper layers untouched. This is the
    # mechanism the s54 blind-tail regression exposed: layer-0 supply caps the whole chain,
    # so the cheapest lever is upgrading the tight primal source. round 2 mines that regime.
    from tokenomics.eval.scale import build_scale_config

    plain = build_scale_config(seed=0, producers_per_layer=3)
    primal = build_scale_config(seed=0, producers_per_layer=3, primal_capacity_scale=0.4)
    primal.validate()

    def cap_by_layer(cfg):
        by: dict[int, int] = {}
        for p in cfg.producer_configs:
            lvl = int(p.id[1:].split("_")[0])
            by[lvl] = by.get(lvl, 0) + sum(p.capacities.values())
        return by

    a, b = cap_by_layer(plain), cap_by_layer(primal)
    assert b[0] < a[0]                                   # layer-0 capacity shrank
    assert all(b[lvl] == a[lvl] for lvl in a if lvl != 0)  # every other layer identical


def test_primal_tight_cells_bind_at_the_primal_layer():
    # the round-2 grid: cells whose binding constraint is capacity-0 (the primal layer),
    # so mining surfaces primal-bottleneck tasks the augment replay must learn to solve.
    from tokenomics.eval.ceiling_bound import ceiling_bounds
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.eval.curriculum_capacity import primal_tight_cells

    cells = primal_tight_cells()
    assert len(cells) >= 1
    seeds = [20, 21, 22]
    for cell in cells:
        cfg = build_scale_config(seed=20, **cell)
        b = ceiling_bounds(cfg, horizon=100, episode_seeds=seeds, include_upgrade_slack=False)
        assert b["B2"].units < b["B0"].units - 0.5          # capacity binds below demand
        assert "capacity-0" in b["B2"].binding_constraint   # and it binds at the primal layer


def test_capacity_tight_params_set_demand_capacity_and_degradation():
    from tokenomics.eval.scale import build_scale_config

    cfg = build_scale_config(seed=0, quantity_max=16, capacity_scale=0.3, degradation=9)
    cfg.validate()                                   # still a runnable economy
    assert all(c.quantity_max == 16 for c in cfg.consumers)
    assert all(c.quantity_min <= c.quantity_max for c in cfg.consumers)
    assert all(p.degradation == 9 for p in cfg.producer_configs)
    # capacity scaled down vs the untouched economy (per-type total shrinks)
    plain = build_scale_config(seed=0)
    tight_cap = sum(sum(p.capacities.values()) for p in cfg.producer_configs)
    plain_cap = sum(sum(p.capacities.values()) for p in plain.producer_configs)
    assert tight_cap < plain_cap


def test_capacity_tight_params_flip_ceiling_to_capacity_bound():
    # The whole point of the axes: a tight cell must read capacity-bound under the
    # no-slack pre-flight tier (Slice 1), where the default cell is demand-bound.
    from tokenomics.eval.ceiling_bound import ceiling_bounds
    from tokenomics.eval.scale import build_scale_config

    seeds = [0, 1, 2]
    default = build_scale_config(seed=0, producers_per_layer=3)
    d = ceiling_bounds(default, horizon=60, episode_seeds=seeds, include_upgrade_slack=False)
    assert d["B2"].units >= d["B0"].units - 0.5  # demand-bound: capacity has slack

    tight = build_scale_config(seed=0, producers_per_layer=3, quantity_max=16, capacity_scale=0.25)
    t = ceiling_bounds(tight, horizon=60, episode_seeds=seeds, include_upgrade_slack=False)
    assert t["B2"].units < t["B0"].units - 0.5     # capacity binds below demand
    # the seed-mean binding may read "capacity:capacity-N" or, when different seeds bind on
    # different types, "mixed:capacity:..." — either way it is a capacity constraint.
    assert "capacity:" in t["B2"].binding_constraint


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


def _level_capacities(cfg):
    """Sorted multiset of total capacity per producer, grouped by level — the
    relabeling-invariant fingerprint of a scale economy's productive base."""
    by_level: dict[int, list[int]] = {}
    for p in cfg.producer_configs:
        lvl = int(p.id[1:].split("_")[0])
        by_level.setdefault(lvl, []).append(sum(p.capacities.values()))
    return {lvl: sorted(v) for lvl, v in by_level.items()}


def test_producer_permutation_relabels_slots_but_preserves_the_economy():
    # Phase 2 (ADR-0049) slot randomization: relabeling which producer config holds
    # which sorted id within a level permutes the per-producer OBS slots but leaves the
    # economy multiset identical — a true relabeling symmetry. base+base is only CLOSE,
    # not exactly equal: `proportional` routing + auction tie-breaking iterate producers
    # in sorted-id order, so slot position leaks into integer-split rounding (exactly the
    # memorization pressure this augmentation adds). Structure is the exact invariant.
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import evaluate
    from tokenomics.eval.scale import build_scale_config, permute_scale_config

    cfg = build_scale_config(0, depth=3)
    # identity on products, reverse the 4 producer slots within each level.
    perm = permute_scale_config(cfg, product_perm=[0, 1, 2, 3], producer_perm=[3, 2, 1, 0])
    assert _dims(perm) == _dims(cfg)                              # same shape: one net spans both
    assert config_fingerprint(perm) != config_fingerprint(cfg)   # a real relabel
    assert _level_capacities(perm) == _level_capacities(cfg)     # exact symmetry: same productive base

    base_p = PRODUCING_BASELINES["stock_aware"]
    base_o = ORDERING_BASELINES["proportional"]
    m0 = evaluate(base_p, base_o, seeds=range(3), horizon=100, config=cfg)["consumer_completed_units"][0]
    m1 = evaluate(base_p, base_o, seeds=range(3), horizon=100, config=perm)["consumer_completed_units"][0]
    assert abs(m1 - m0) <= 0.05 * m0  # isomorphic economy performs within 5%


def test_product_permutation_preserves_recipe_dag_and_eligibility():
    # The product-index permutation must be applied consistently across every level, so
    # each composite still consumes its same-(permuted-)index lower product — the recipe
    # DAG stays isomorphic and chain depths are preserved. `produces` stays a whole level
    # set, so eligibility (>=2 declarers per top product) survives.
    from tokenomics.eval.scale import build_scale_config, permute_scale_config

    cfg = build_scale_config(0, depth=3)
    perm = permute_scale_config(cfg, product_perm=[2, 0, 3, 1], producer_perm=[0, 1, 2, 3])
    assert _dims(perm) == _dims(cfg)
    perm.validate()
    # chain-depth spectrum is preserved (same multiset of recipe depths).
    spec0 = sorted(_chain_depth(cfg, ps.product) for ps in cfg.process_sheets)
    spec1 = sorted(_chain_depth(perm, ps.product) for ps in perm.process_sheets)
    assert spec0 == spec1
    top = max(_layer(ps.product) for ps in perm.process_sheets)
    for ps in perm.process_sheets:
        if _layer(ps.product) == top:
            assert len([p for p in perm.producer_configs if ps.product in p.produces]) >= 2


def test_scale_topology_set_expands_seeds_into_permuted_same_shape_configs():
    # Phase 2 step 5+1-4: the training set is each base seed plus `perms` slot-permuted
    # variants. All share the dims envelope (one net spans them), every variant is a
    # genuine relabel of its base (different fingerprint), and identity-only (perms=0)
    # returns exactly the base configs.
    from tokenomics.eval.scale import build_scale_config, config_to_dict, scale_topology_set

    base_only = scale_topology_set([0, 1], depth=3, perms=0)
    assert [config_to_dict(c) for c in base_only] == [
        config_to_dict(build_scale_config(s, depth=3)) for s in (0, 1)
    ]

    cfgs = scale_topology_set([0, 1], depth=3, perms=2, perm_seed=7)
    assert len(cfgs) == 2 * (1 + 2)                 # seed x (base + 2 perms)
    assert len({_dims(c) for c in cfgs}) == 1       # all same shape
    fps = [config_fingerprint(c) for c in cfgs]
    assert len(set(fps)) == len(fps)                # every variant distinct
    # the two permutations of seed 0 are relabelings of the same productive base.
    assert _level_capacities(cfgs[1]) == _level_capacities(cfgs[0])
    assert _level_capacities(cfgs[2]) == _level_capacities(cfgs[0])


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
