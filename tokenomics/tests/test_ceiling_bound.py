"""Offline achievable-units bounds (add-m3-development-audit task 4.2, design D12 "Bound and
ceiling"; development-audit "Upper bound stays above every played arm", "Demand cap reported as
such"). Restored from EXP-0033 (commit b96ecc0) and ported to the env-built demand schedule."""

from __future__ import annotations

import dataclasses

import pytest

from tokenomics.domain.config import (
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.demand_digest import demand_digest
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.ceiling_bound import (
    B_BASE,
    DEMAND,
    ceiling_bounds,
    env_demand,
    episode_demand,
    recipe_capacity_bundle,
    schedule_demand,
    upgrade_slack_ratio,
    world_ceiling,
)
from tokenomics.eval.families import build_family_config
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.roster import JOINT_STACK, contract_for
from tokenomics.eval.routing_audit import fanout_world
from tokenomics.eval.routing_diagnostics import DIAGNOSTIC_ROUTERS
from tokenomics.eval.runner import run_episode
from tokenomics.eval.shock_profiles import SHOCKS_I_SMOKE

CALM_12 = "dev-sparse_wide-12-calm-00"
SHOCKED_12 = "dev-sparse_wide-12-shocked-00"  # shocks-i: drift j = 0.34


def _producer(pid: str, cap_type: str, cap: int, produces: frozenset[int]) -> ProducingAgentConfig:
    return ProducingAgentConfig(
        id=pid,
        capacities={cap_type: cap},
        degradation=0,
        restoration=Restoration(require=1, restores=0, type=cap_type),
        upgrade=Upgrade(require=1, increases=0, type=cap_type),
        produces=produces,
    )


def _toy_config() -> Configuration:
    """A hand-checkable two-level economy.

    Top product 2 uses 5 units of c1 and two input units of product 1.  Product 1 uses
    2 units of c0, so B2 charges c1=5 and c0=4 per delivered top unit.
    """
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=1, require={"c0": 2}),
            ProcessSheet(product=2, require={"c1": 5}, inputs={1: 2}),
        ],
        producer_configs=[
            _producer("p0", "c0", 10, frozenset({1})),
            _producer("p1", "c1", 20, frozenset({2})),
        ],
        consumers=[ConsumerConfig(id="c", product=2, quantity_min=100, quantity_max=100)],
    )
    cfg.validate()
    return cfg


def test_episode_demand_is_the_env_built_schedule() -> None:
    """A fixed-quantity consumer replays (2, 100) every cycle; a cyclic schedule wraps."""
    assert episode_demand(_toy_config(), horizon=4, seed=123) == {2: 400}
    assert schedule_demand({"a": [(1, 2), (3, 1)], "b": [(1, 5)]}, horizon=3) == {1: 2 + 2 + 15, 3: 1}


def test_transitive_recipe_bundle_charges_input_products() -> None:
    cfg = _toy_config()
    assert recipe_capacity_bundle(cfg, 2, transitive=False) == {"c1": 5.0}
    assert recipe_capacity_bundle(cfg, 2, transitive=True) == {"c1": 5.0, "c0": 4.0}


def test_b0_b1_b2_hand_checked_toy_bounds() -> None:
    bounds = ceiling_bounds(_toy_config(), 4, episode_demand(_toy_config(), 4, 0))
    # Capacity cycles: c0 = 10*4 = 40; c1 = 20*4 = 80.  Direct top recipe binds at
    # 80/5 = 16 units; transitive input coupling binds at 40/4 = 10 units.
    assert bounds["B0"].units == 400.0
    assert bounds["B0"].binding_constraint == DEMAND
    assert bounds["B1"].units == 16.0
    assert bounds["B1"].binding_constraint == "capacity:c1"
    assert bounds["B2"].units == 10.0
    assert bounds["B2"].binding_constraint == "capacity:c0"


def _upgrade_config(cap: int, increases: int, demand: int) -> Configuration:
    """One-level economy whose producer can expand capacity via upgrade.

    Base capacity is `cap`; the (optimistic, free/instant) upgrade slack adds
    `increases * horizon * horizon`.  With demand between the two, the ceiling is
    demand-bound WITH slack but capacity-bound WITHOUT it.
    """
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"c0": 1})],
        producer_configs=[
            ProducingAgentConfig(
                id="p0",
                capacities={"c0": cap},
                degradation=0,
                restoration=Restoration(require=1, restores=0, type="c0"),
                upgrade=Upgrade(require=1, increases=increases, type="c0"),
                produces=frozenset({1}),
            )
        ],
        consumers=[ConsumerConfig(id="c", product=1, quantity_min=demand, quantity_max=demand)],
    )
    cfg.validate()
    return cfg


def test_no_upgrade_slack_tier_makes_capacity_bind() -> None:
    # base c0 = 10*4 = 40; upgrade slack adds 5*4*4 = 80 -> 120.  Demand = 25 * 4 = 100.
    cfg = _upgrade_config(cap=10, increases=5, demand=25)
    demand = episode_demand(cfg, 4, 0)
    with_slack = ceiling_bounds(cfg, 4, demand)
    assert with_slack["B2"].units == 100.0
    assert with_slack["B2"].binding_constraint == DEMAND
    no_slack = ceiling_bounds(cfg, 4, demand, include_upgrade_slack=False)
    assert no_slack["B2"].units == 40.0
    assert no_slack["B2"].binding_constraint == "capacity:c0"
    ceiling = world_ceiling(cfg, 4, demand)
    assert ceiling["B2"] == with_slack["B2"]
    assert (ceiling[B_BASE].tier, ceiling[B_BASE].units) == (B_BASE, 40.0)


def _registry_config(world: str) -> Configuration:
    return load_registry(REGISTRY_V2).entry(world).config()


def test_demand_bound_on_a_registry_world() -> None:
    """Registry v2 12-producer world, horizon 12: capacity never binds, so B2 (and B_base) equal
    the scheduled consumer units, which bound what an episode actually orders."""
    entry = load_registry(REGISTRY_V2).entry(CALM_12)
    cfg, seed = entry.config(), entry.episode_seeds[0]
    env = joint_stack_env(cfg, 12)
    env.reset(seed=seed)
    demand = env_demand(env)
    scheduled = float(sum(demand.values()))
    ceiling = world_ceiling(cfg, 12, demand)
    assert ceiling["B2"].units == ceiling[B_BASE].units == scheduled
    assert ceiling["B2"].binding_constraint == DEMAND
    stats = run_episode(env, PRODUCING_BASELINES["entity_planner"],
                        DIAGNOSTIC_ROUTERS["round_robin"], seed=seed)
    ordered = sum(stats["consumer_ordered_units"].values())
    assert 0 < ordered <= scheduled


def _tight_corner_config() -> Configuration:
    """The D10 high-tightness corner built by hand (the levers come in task 4.4): a 12-producer
    calm sparse_wide world at capacity_scale 0.25, with 8 copies of every consumer (k_c = 8)
    whose quantities are multiplied by 8 (m_q = 8)."""
    cfg = build_family_config(0, family="sparse_wide", producers=12, regime="calm", fanout=2,
                              capacity_scale=0.25)
    consumers = [dataclasses.replace(c, id=f"{c.id}_{k}", quantity_min=c.quantity_min * 8,
                                     quantity_max=c.quantity_max * 8,
                                     schedule=tuple((p, q * 8) for p, q in c.schedule))
                 for k in range(8) for c in cfg.consumers]
    tight = dataclasses.replace(cfg, consumers=consumers)
    tight.validate()
    return tight


def test_demand_cap_reported_as_such_and_base_capacity_binds_at_the_tight_corner() -> None:
    """Scenario "Demand cap reported as such": at horizon 100 the upgrade-slack B2 stays at the
    scheduled units (demand-bound) while B_base binds on capacity far below them."""
    cfg = _tight_corner_config()
    demand = episode_demand(cfg, 100, 0)
    scheduled = float(sum(demand.values()))
    ceiling = world_ceiling(cfg, 100, demand)
    assert ceiling["B2"].units == scheduled and ceiling["B2"].binding_constraint == DEMAND
    assert ceiling[B_BASE].units < 0.1 * scheduled
    assert ceiling[B_BASE].binding_constraint.startswith("capacity:")


def test_upgrade_slack_dwarfs_base_capacity_on_a_registry_world() -> None:
    """Why B2 is a demand cap on family worlds: the free-upgrade slack over horizon 100 is a
    large multiple of base capacity on every capacity type (module docstring: 33-74x)."""
    ratio = upgrade_slack_ratio(_registry_config(CALM_12), 100)
    assert len(ratio) == 3 and min(ratio.values()) > 10, f"upgrade slack / base capacity: {ratio}"


def _bound_worlds() -> list[tuple[str, Configuration, int]]:
    calm = load_registry(REGISTRY_V2).entry(CALM_12)
    f4 = fanout_world("sparse_wide", 12, 4, "shocked", 0, 1, SHOCKS_I_SMOKE)
    return [(calm.id, calm.config(), calm.episode_seeds[0]),
            (f4.id, f4.config(), f4.episode_seeds[0])]


@pytest.mark.parametrize("router", [*ORDERING_BASELINES, "round_robin"])
def test_bound_dominates_every_arm(router: str) -> None:
    """Scenario "Bound below an arm" never triggers: on a calm f2 registry world and a shocked
    f4 family world (shocks-i-smoke), horizon 12, B2 >= the consumer units of `entity_planner`
    with every roster ordering baseline and `round_robin`, each under its roster contract."""
    policy = ORDERING_BASELINES.get(router) or DIAGNOSTIC_ROUTERS[router]
    for world, cfg, seed in _bound_worlds():
        contract = contract_for("entity_planner", router) if router in ORDERING_BASELINES \
            else JOINT_STACK
        env = (joint_stack_env(cfg, 12) if contract == JOINT_STACK
               else TokenomicsAECEnv(config=cfg, horizon=12))
        stats = run_episode(env, PRODUCING_BASELINES["entity_planner"], policy, seed=seed,
                            action_seed=7)
        units = sum(stats["consumer_fulfilled_units"].values())
        bound = world_ceiling(cfg, 12, env_demand(env))["B2"].units
        assert bound + 0.01 >= units, (world, router, bound, units)


def test_drift_world_demand_is_the_digested_schedule() -> None:
    """On a drifting shocked registry world the bound reads exactly the schedules the env
    digests (`demand_digest`), and their demand sums equal `schedule_demand`'s."""
    entry = load_registry(REGISTRY_V2).entry(SHOCKED_12)
    cfg = entry.config()
    assert cfg.demand_drift is not None
    env = joint_stack_env(cfg, 12)
    env.reset(seed=entry.episode_seeds[0])
    assert demand_digest(env._demand_schedules) == env.episode_stats["demand_digest"]
    demand = env_demand(env)
    by_hand: dict[int, int] = {}
    for schedule in env._demand_schedules.values():
        for t in range(12):
            p, q = schedule[t % len(schedule)]
            by_hand[p] = by_hand.get(p, 0) + q
    assert demand == by_hand
    assert demand == episode_demand(cfg, 12, entry.episode_seeds[0])


def _one_type_config(products: dict[int, ProcessSheet], capacities: dict[str, int],
                     consumers: list[ConsumerConfig]) -> Configuration:
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=list(products.values()),
        producer_configs=[ProducingAgentConfig(
            id=f"p_{t}", capacities={t: cap}, degradation=0,
            restoration=Restoration(require=min(products), restores=0, type=t),
            upgrade=Upgrade(require=min(products), increases=0, type=t),
            produces=frozenset(x for x, s in products.items() if t in s.require))
            for t, cap in capacities.items()],
        consumers=consumers,
    )
    cfg.validate()
    return cfg


def test_binding_type_serves_the_cheapest_products_first() -> None:
    """One binding type over two demanded products of per-unit cost 1 and 4, demand 10 each,
    20 capacity-cycles (horizon 1): the relaxation serves the cheap product first, 10 + 10/4 =
    12.5 units. Serving the dear one first would give 20/4 = 5 and undercut a feasible plan
    (10 cheap units), so the order is what keeps the tier an upper bound."""
    cfg = _one_type_config(
        {1: ProcessSheet(1, {"c": 1}), 2: ProcessSheet(2, {"c": 4})}, {"c": 20},
        [ConsumerConfig(id="a", product=1, quantity_min=10, quantity_max=10),
         ConsumerConfig(id="b", product=2, quantity_min=10, quantity_max=10)])
    demand = episode_demand(cfg, 1, 0)
    assert demand == {1: 10, 2: 10}
    bounds = ceiling_bounds(cfg, 1, demand)
    for tier in ("B1", "B2"):
        assert bounds[tier].units == 12.5 and bounds[tier].binding_constraint == "capacity:c"
    assert bounds["B2"].units >= 10  # a real plan: all 10 cheap units


def test_cyclic_recipe_charges_each_requirement_once() -> None:
    """Configuration validation allows a cyclic recipe (1 needs 2, 2 needs 1). The transitive
    bundle ignores the repeated edge (optimistic), charging each direct requirement once, and
    the bound stays finite."""
    cfg = _one_type_config(
        {1: ProcessSheet(1, {"c": 2}, inputs={2: 1}), 2: ProcessSheet(2, {"d": 3}, inputs={1: 1})},
        {"c": 10, "d": 30},
        [ConsumerConfig(id="a", product=1, quantity_min=10, quantity_max=10)])
    assert recipe_capacity_bundle(cfg, 1, transitive=True) == {"c": 2.0, "d": 3.0}
    assert recipe_capacity_bundle(cfg, 2, transitive=True) == {"d": 3.0, "c": 2.0}
    bounds = ceiling_bounds(cfg, 2, episode_demand(cfg, 2, 0))  # demand 20; c 20/2, d 60/3
    assert bounds["B2"].units == 10.0 and bounds["B2"].binding_constraint == "capacity:c"
