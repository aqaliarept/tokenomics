"""Controlled-generalization suite (NEXT STEPS stage 3): single-axis config
perturbations toward UNSEEN/harder settings, each changing exactly one dimension so
a failure is attributable to it. Pure config transforms (no torch) — every perturbed
config must keep the obs/action dims of the base so a policy stays loadable."""

from __future__ import annotations

from tokenomics.domain import load_config
from tokenomics.env.aec_env import DEFAULT_CONFIG
from tokenomics.env.spaces import build_spec


def _dims(cfg):
    sp = build_spec(cfg)
    return (
        sp.producing_obs_dim,
        sp.producing_action_dim,
        sp.ordering_obs_dim,
        sp.ordering_action_dim,
    )


def test_perturb_degradation_raises_decay_only_and_preserves_dims():
    from tokenomics.eval.generalization import perturb_degradation

    base = load_config(DEFAULT_CONFIG)
    hard = perturb_degradation(base, 2.0)

    for b, h in zip(base.producer_configs, hard.producer_configs):
        assert h.degradation == round(b.degradation * 2.0)
        assert h.degradation > b.degradation
        assert h.capacities == b.capacities  # only decay changed
    # base untouched (frozen + replace), and obs/action dims unchanged
    assert base.producer_configs[0].degradation == 5
    assert _dims(hard) == _dims(base)


def test_perturb_quantity_sets_unseen_demand_and_preserves_dims():
    from tokenomics.eval.generalization import perturb_quantity

    base = load_config(DEFAULT_CONFIG)  # phase1 consumers are fixed qty 1
    hard = perturb_quantity(base, quantity_max=4)

    for b, h in zip(base.consumers, hard.consumers):
        assert h.quantity_max == 4               # unseen upper quantity
        assert h.quantity_max > b.quantity_max
        assert h.quantity_min == b.quantity_min  # only the ceiling moved
        assert h.product_pool == b.product_pool  # mixture axis untouched
    assert base.consumers[0].quantity_max == 1   # base untouched
    assert _dims(hard) == _dims(base)


def test_perturb_producer_capacity_scales_values_keeps_types_and_dims():
    from tokenomics.eval.generalization import perturb_producer_capacity

    base = load_config(DEFAULT_CONFIG)
    varied = perturb_producer_capacity(base, 0.7)

    for b, v in zip(base.producer_configs, varied.producer_configs):
        assert set(v.capacities) == set(b.capacities)  # same capacity TYPES (dims)
        for t in b.capacities:
            assert v.capacities[t] == round(b.capacities[t] * 0.7)
            assert v.capacities[t] < b.capacities[t]   # numerically lower
    assert base.producer_configs[0].capacities["capacity-1"] == 120  # base untouched
    assert _dims(varied) == _dims(base)


def test_generalization_suite_each_config_changes_one_axis_and_keeps_dims():
    from tokenomics.eval.generalization import generalization_suite

    base = load_config(DEFAULT_CONFIG)
    suite = generalization_suite(base)

    # named single-axis perturbations, the dims-preserving stage-3 axes for phase1
    assert {"higher_degradation", "unseen_quantities", "leaner_producers"} <= set(suite)
    base_dims = _dims(base)
    for name, cfg in suite.items():
        assert _dims(cfg) == base_dims, f"{name} changed obs/action dims"
        assert cfg != base  # each is an actual perturbation


# --- NEXT STEPS stage 4: combined held-out suite (multiple axes at once) ---
def test_perturb_costs_scales_set_costs_keeps_none_and_preserves_dims():
    from tokenomics.eval.generalization import perturb_costs

    # competition.json producers carry restoration/upgrade/replenish/expansion costs
    from pathlib import Path
    base = load_config(Path(__file__).parent / "fixtures" / "competition.json")
    dear = perturb_costs(base, 1.5)

    for b, d in zip(base.producer_configs, dear.producer_configs):
        # set costs scale up; None stays None (no spurious target introduced)
        for attr in ("restoration", "upgrade"):
            bc, dc = getattr(b, attr).cost, getattr(d, attr).cost
            assert (dc == round(bc * 1.5)) if bc is not None else (dc is None)
        if b.replenish_cost_per_unit is not None:
            assert d.replenish_cost_per_unit == round(b.replenish_cost_per_unit * 1.5)
        else:
            assert d.replenish_cost_per_unit is None
        if b.inventory_expansion is not None and b.inventory_expansion.cost is not None:
            assert d.inventory_expansion.cost == round(b.inventory_expansion.cost * 1.5)
    assert _dims(dear) == _dims(base)
    assert dear != base


def test_combined_config_applies_every_axis_and_preserves_dims():
    from pathlib import Path
    from tokenomics.eval.generalization import combined_config

    base = load_config(Path(__file__).parent / "fixtures" / "competition.json")
    combo = combined_config(
        base, degradation_factor=2.0, quantity_max=3, capacity_factor=0.7, cost_factor=1.5
    )

    # every requested axis is reflected at once
    assert combo.producer_configs[0].degradation == round(base.producer_configs[0].degradation * 2.0)
    assert all(c.quantity_max == 3 for c in combo.consumers)
    p0 = base.producer_configs[0]
    for t, v in p0.capacities.items():
        assert combo.producer_configs[0].capacities[t] == round(v * 0.7)
    assert combo.producer_configs[0].restoration.cost == round(p0.restoration.cost * 1.5)
    # still a valid, dims-preserving config the trained net can load
    combo.validate()
    assert _dims(combo) == _dims(base)


def test_combined_config_identity_when_no_axis_changed():
    from pathlib import Path
    from tokenomics.eval.generalization import combined_config

    base = load_config(Path(__file__).parent / "fixtures" / "competition.json")
    # defaults = neutral on every axis -> an equivalent config (no accidental drift)
    same = combined_config(base)
    assert same == base


def test_random_domain_config_is_dims_preserving_and_reproducible():
    # Domain-randomized training (NEXT STEPS step-3 fix): each training iter draws a
    # config from a DISTRIBUTION of operating points (the consolidated fix the stage
    # 2-4 overfit findings all pointed to), so the policy can't memorize one point.
    # Each draw must keep obs/action dims (one net spans the suite) and be reproducible.
    from pathlib import Path
    import numpy as np
    from tokenomics.eval.generalization import random_domain_config

    base = load_config(Path(__file__).parent / "fixtures" / "competition.json")
    ranges = dict(degradation_range=(1.0, 1.5), quantity_choices=(1, 2, 3),
                  capacity_range=(0.7, 1.0), cost_range=(1.0, 1.5))

    draws = [random_domain_config(base, np.random.default_rng(7), **ranges) for _ in range(2)]
    assert draws[0] == draws[1]                       # same rng seed -> identical config
    draws[0].validate()
    assert _dims(draws[0]) == _dims(base)             # dims fixed across the distribution

    # over many seeds the configs actually vary (not collapsed to the base)
    seen = {id(None)}
    variants = {
        random_domain_config(base, np.random.default_rng(s), **ranges).producer_configs[0].degradation
        for s in range(20)
    }
    assert len(variants) >= 2
