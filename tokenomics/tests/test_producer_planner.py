"""EXP-0010B: deterministic producer planner over ProducerEntityView.

The planner is the first non-learned controller for the entity-factored producer
contract. Tests exercise the public domain interface (`plan_producer_command`) and
the env-policy adapter, not implementation internals.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from tokenomics.domain import build_system, load_config
from tokenomics.domain.config import Configuration
from tokenomics.env.aec_env import DEFAULT_CONFIG


def test_planner_replenishes_missing_inputs_to_a_local_stock_target():
    """A composite producer should prime missing input rows up to the transparent
    local stock target, without issuing speculative investments while capacity is
    healthy."""
    from tokenomics.domain.producer_planner import plan_producer_command

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    view = system.producer_entity_view("p4")  # product 30 needs one each of 11 and 20.

    cmd = plan_producer_command(
        view,
        max_replenish=10,
        replenish_cost_per_unit=40,
        funding_budget=500,
    )

    assert cmd.replenishments == {11: 3, 20: 3}
    assert cmd.do_restoration is False
    assert cmd.do_upgrade is False
    assert cmd.do_inventory_expansion is False


def test_entity_planner_baseline_emits_a_valid_env_action():
    """The planner is deployable through the normal producer-policy interface, using
    the entity view while still emitting the existing Box action contract."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.env import TokenomicsAECEnv

    env = TokenomicsAECEnv(horizon=10, entity_producer_obs=True)
    env.reset(seed=0)
    agent = "prod_p4"

    action = PRODUCING_BASELINES["entity_planner"](env, agent)

    assert env.action_space(agent).contains(action)


def test_planner_expands_inventory_when_headroom_blocks_needed_inputs():
    """If input demand exists but inventory headroom is exhausted, the planner should
    request the configured inventory expansion instead of issuing impossible
    replenishments."""
    from tokenomics.domain.producer_planner import plan_producer_command

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    p4 = system.producing_agents["p4"]
    p4.inventory[11] = p4.inventory_limit  # no headroom left for needed inputs.
    view = system.producer_entity_view("p4")

    cmd = plan_producer_command(
        view,
        max_replenish=10,
        replenish_cost_per_unit=40,
        funding_budget=500,
        inventory_expansion_increases=20,
        inventory_expansion_cost=30,
    )

    assert cmd.replenishments == {}
    assert cmd.do_inventory_expansion is True


def test_planner_respects_headroom_funding_and_running_investment_gates():
    """The command must be locally feasible: no replenishment beyond inventory
    headroom/funding and no restart of already-running investment streams."""
    from tokenomics.domain.producer_planner import plan_producer_command

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    p4 = system.producing_agents["p4"]
    p4.restoration_running = True
    p4.inventory_expansion_running = True
    view = system.producer_entity_view("p4")
    view = replace(
        view,
        globals=replace(
            view.globals,
            inventory_headroom=1,
            active_investments=(1.0, 0.0, 1.0),
        ),
    )

    cmd = plan_producer_command(
        view,
        max_replenish=10,
        replenish_cost_per_unit=40,
        funding_budget=40,
        inventory_expansion_increases=20,
        inventory_expansion_cost=30,
    )

    assert sum(cmd.replenishments.values()) <= 1
    assert cmd.do_restoration is False
    assert cmd.do_inventory_expansion is False


def test_planner_scores_are_transparent_and_local():
    """10B exposes simple local scores for the planned command: product urgency,
    input value, capacity restoration ROI, and inventory-expansion value."""
    from tokenomics.domain.producer_planner import score_producer_entity_view

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    p4 = system.producing_agents["p4"]
    p4.capacities["capacity-4"] = 40
    p4.inventory[11] = p4.inventory_limit
    view = system.producer_entity_view("p4")

    scores = score_producer_entity_view(view)

    assert scores.product_urgency[30] > 0.0
    assert scores.input_value[11] > 0.0
    assert scores.input_value[20] > 0.0
    assert scores.restoration_roi > 0.0
    assert scores.inventory_expansion_value > 0.0


def test_planner_splits_headroom_evenly_for_near_equal_input_scores():
    """Fork/join recipes need balanced paired inputs. A learned scorer emits near-equal
    but non-identical input values; with a relative tie tolerance the planner treats them
    as tied and splits scarce headroom evenly, instead of overfilling the marginally
    higher one and starving its mate (EXP-0017 producer-side fix)."""
    from tokenomics.domain.producer_planner import (
        ProducerLocalScores,
        plan_producer_command,
    )

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    view = system.producer_entity_view("p4")  # product 30 joins inputs 11 and 20.
    view = replace(view, globals=replace(view.globals, inventory_headroom=4))
    scores = ProducerLocalScores(input_value={11: 14.502, 20: 14.498})

    cmd = plan_producer_command(
        view,
        max_replenish=10,
        funding_budget=None,
        scores=scores,
        replenish_tie_rel_tol=0.02,
    )

    assert cmd.replenishments == {11: 2, 20: 2}


def test_tie_tolerance_does_not_merge_clearly_different_input_scores():
    """The tolerance only smooths near-equal noise. Inputs whose values differ well
    beyond the band stay strictly ordered: the higher-valued input takes its full need
    before any headroom spills to the lower one."""
    from tokenomics.domain.producer_planner import (
        ProducerLocalScores,
        plan_producer_command,
    )

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    view = system.producer_entity_view("p4")
    view = replace(view, globals=replace(view.globals, inventory_headroom=3))
    scores = ProducerLocalScores(input_value={11: 20.0, 20: 16.0})  # 20% apart >> 2%

    cmd = plan_producer_command(
        view,
        max_replenish=10,
        funding_budget=None,
        scores=scores,
        replenish_tie_rel_tol=0.02,
    )

    assert cmd.replenishments == {11: 3}


def test_planner_is_invariant_to_entity_row_order():
    """Reordering capacity/product/input rows should not change the producer-level
    command; relation planes are already row-position-gated by 10A."""
    from tokenomics.domain.producer_planner import plan_producer_command

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    view = system.producer_entity_view("p4")
    shuffled = copy.deepcopy(view)
    shuffled.capacity_rows = list(reversed(shuffled.capacity_rows))
    shuffled.product_rows = list(reversed(shuffled.product_rows))
    shuffled.input_rows = list(reversed(shuffled.input_rows))

    kwargs = dict(max_replenish=10, replenish_cost_per_unit=40, funding_budget=500)
    assert plan_producer_command(view, **kwargs) == plan_producer_command(shuffled, **kwargs)


def _relabel(config: Configuration, pm=None, cm=None) -> Configuration:
    import dataclasses as dc

    pm = pm or {}
    cm = cm or {}
    rp = lambda x: pm.get(x, x)
    rc = lambda t: cm.get(t, t)

    sheets = [
        dc.replace(
            s,
            product=rp(s.product),
            require={rc(t): c for t, c in s.require.items()},
            inputs={rp(g): n for g, n in s.inputs.items()},
        )
        for s in config.process_sheets
    ]
    producers = [
        dc.replace(
            p,
            capacities={rc(t): c for t, c in p.capacities.items()},
            produces=frozenset(rp(x) for x in p.produces),
            restoration=dc.replace(p.restoration, require=rp(p.restoration.require), type=rc(p.restoration.type)),
            upgrade=dc.replace(p.upgrade, require=rp(p.upgrade.require), type=rc(p.upgrade.type)),
            inventory_expansion=(
                dc.replace(p.inventory_expansion, require=rp(p.inventory_expansion.require))
                if p.inventory_expansion else None
            ),
        )
        for p in config.producer_configs
    ]
    consumers = [
        dc.replace(c, product=rp(c.product), products=tuple(rp(x) for x in c.products))
        for c in config.consumers
    ]
    out = dc.replace(config, process_sheets=sheets, producer_configs=producers, consumers=consumers)
    out.validate()
    return out


def _planned_for(config: Configuration, pid: str):
    from tokenomics.domain.producer_planner import plan_producer_command

    system = build_system(config)
    return plan_producer_command(
        system.producer_entity_view(pid),
        max_replenish=10,
        replenish_cost_per_unit=40,
        funding_budget=500,
    )


def test_planner_is_equivariant_to_product_relabeling():
    cfg = load_config(DEFAULT_CONFIG)
    pm = {10: 50, 11: 44, 20: 47, 30: 41}
    inv_pm = {v: k for k, v in pm.items()}

    base = _planned_for(cfg, "p4")
    relabeled = _planned_for(_relabel(cfg, pm=pm), "p4")
    unrelabeled_repl = {inv_pm[k]: v for k, v in relabeled.replenishments.items()}

    assert unrelabeled_repl == base.replenishments
    assert relabeled.do_restoration == base.do_restoration
    assert relabeled.do_upgrade == base.do_upgrade


def test_planner_is_invariant_to_capacity_relabeling():
    cfg = load_config(DEFAULT_CONFIG)
    cm = {
        "capacity-1": "capacity-9",
        "capacity-2": "capacity-8",
        "capacity-3": "capacity-7",
        "capacity-4": "capacity-6",
    }

    assert _planned_for(_relabel(cfg, cm=cm), "p4") == _planned_for(cfg, "p4")


def test_entity_planner_runs_on_larger_synthetic_topologies():
    """The deterministic planner should run on larger row counts without training or
    shape-specific code."""
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import run_episode
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.env import TokenomicsAECEnv

    cfg = build_scale_config(
        0,
        depth=4,
        products_per_layer=8,
        producers_per_layer=3,
        num_consumers=3,
    )
    stats = run_episode(
        TokenomicsAECEnv(config=cfg, horizon=8, entity_producer_obs=True),
        PRODUCING_BASELINES["entity_planner"],
        ORDERING_BASELINES["proportional"],
        seed=0,
    )

    assert stats["cycles"] > 0


def test_evaluate_can_enable_entity_producer_obs_for_planner_runs():
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.runner import evaluate

    result = evaluate(
        PRODUCING_BASELINES["entity_planner"],
        ORDERING_BASELINES["proportional"],
        seeds=[0],
        horizon=5,
        entity_producer_obs=True,
    )

    assert result["cycles"][0] == 5
