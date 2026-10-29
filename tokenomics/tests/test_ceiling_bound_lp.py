"""B_lp, the input-output ceiling (add-m3-development-audit task 4.7, design D12 "Bound"):
hand-solved fixtures for every relaxation it keeps, and soundness against played arms."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from tokenomics.domain.config import (  # noqa: E402
    Configuration,
    ConsumerConfig,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
)
from tokenomics.env import TokenomicsAECEnv  # noqa: E402
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES  # noqa: E402
from tokenomics.eval.ceiling_bound import (  # noqa: E402
    B_LP,
    B_LP_BASE,
    DEMAND,
    LP_CAPACITY,
    ceiling_bounds,
    env_cycle_demand,
    episode_cycle_demand,
    episode_demand,
    lp_bound,
    lp_size,
    schedule_cycle_demand,
    world_ceiling,
)
from tokenomics.eval.joint_stack import joint_stack_env  # noqa: E402
from tokenomics.eval.roster import JOINT_STACK, contract_for  # noqa: E402
from tokenomics.eval.routing_diagnostics import DIAGNOSTIC_ROUTERS  # noqa: E402
from tokenomics.eval.runner import run_episode  # noqa: E402

from test_ceiling_bound import _bound_worlds, _tight_corner_config  # noqa: E402

BIG = 10 ** 6


def _producer(pid: str, cap: int, produces: set[int], *, degradation: int = 0,
              restore: tuple[int, int] = (1, 0), upgrade: tuple[int, int] = (1, 0),
              cap_type: str = "c0") -> ProducingAgentConfig:
    return ProducingAgentConfig(
        id=pid, capacities={cap_type: cap}, degradation=degradation,
        restoration=Restoration(require=restore[0], restores=restore[1], type=cap_type),
        upgrade=Upgrade(require=upgrade[0], increases=upgrade[1], type=cap_type),
        produces=frozenset(produces))


def _config(sheets: list[ProcessSheet], producers: list[ProducingAgentConfig]) -> Configuration:
    cfg = Configuration(cycle_emission=1000, process_sheets=sheets, producer_configs=producers,
                        consumers=[ConsumerConfig(id="c", product=1, quantity_min=1,
                                                  quantity_max=1)])
    cfg.validate()
    return cfg


def _flat(horizon: int, per_cycle: int, product: int = 1) -> dict[int, list[int]]:
    return {product: [per_cycle] * horizon}


def test_capacity_binds_like_the_base_tier_on_a_one_product_world() -> None:
    """One producer, capacity 10, no degradation or investment, 25 ordered per cycle for 4
    cycles: 40 units, the same as B_base, and below demand (100)."""
    cfg = _config([ProcessSheet(1, {"c0": 1})], [_producer("p", 10, {1})])
    b = lp_bound(cfg, 4, _flat(4, 25))
    assert (b.tier, b.units, b.binding_constraint) == (B_LP, pytest.approx(40.0), LP_CAPACITY)


def test_order_window_limits_service_to_ttl_cycles() -> None:
    """100 units ordered in cycle 0 only, capacity 10 per cycle, horizon 6: the order can be
    served by the auctions of cycles 0..2 (ORDER_TTL = 2), 30 units; aggregated over the
    horizon (window=None) the LP would allow 60."""
    cfg = _config([ProcessSheet(1, {"c0": 1})], [_producer("p", 10, {1})])
    demand = {1: [100, 0, 0, 0, 0, 0]}
    assert lp_bound(cfg, 6, demand).units == pytest.approx(30.0)
    assert lp_bound(cfg, 6, demand, window=None).units == pytest.approx(60.0)


def test_degradation_decays_capacity_geometrically() -> None:
    """Capacity 100 at 10% degradation, no restoration, horizon 3: the relaxation allows
    100 + 90 + 81 = 271, one unit above the domain's 100 + 90 + 80 (ceil(10% of max) = 10 a
    cycle), so it stays an upper bound."""
    cfg = _config([ProcessSheet(1, {"c0": 1})], [_producer("p", 100, {1}, degradation=10)])
    assert lp_bound(cfg, 3, _flat(3, BIG)).units == pytest.approx(271.0)


def test_restoration_is_charged_through_its_recipe() -> None:
    """Capacity 100 at 50% degradation, horizon 2. A restoration (+50) costs one unit of product
    2, which needs 10 capacity: a0 = 100, a1 = min(100, 50 + 50) = 100, minus 10 for the
    restoration product = 190. Without the restoration it would be 150."""
    sheets = [ProcessSheet(1, {"c0": 1}), ProcessSheet(2, {"c0": 10})]
    cfg = _config(sheets, [_producer("p", 100, {1, 2}, degradation=50, restore=(2, 50))])
    assert lp_bound(cfg, 2, _flat(2, BIG)).units == pytest.approx(190.0)
    no_rest = _config(sheets, [_producer("p", 100, {1, 2}, degradation=50, restore=(2, 0))])
    assert lp_bound(no_rest, 2, _flat(2, BIG)).units == pytest.approx(150.0)


def test_upgrades_are_a_costed_growth_bound_not_free_slack() -> None:
    """Capacity 10, upgrade +5 per unit of product 2 (1 capacity each), at most one per cycle,
    horizon 4: capacity 10, 15, 20, 25 = 70, minus the three upgrade units = 67. B_lp_base (no
    upgrades) is 40; B2's free slack (5 x 4^2) gives 120."""
    sheets = [ProcessSheet(1, {"c0": 1}), ProcessSheet(2, {"c0": 1})]
    cfg = _config(sheets, [_producer("p", 10, {1, 2}, upgrade=(2, 5))])
    demand = _flat(4, BIG)
    assert lp_bound(cfg, 4, demand).units == pytest.approx(67.0)
    base = lp_bound(cfg, 4, demand, upgrades=False)
    assert (base.tier, base.units) == (B_LP_BASE, pytest.approx(40.0))
    assert ceiling_bounds(cfg, 4, {1: 4 * BIG})["B2"].units == pytest.approx(120.0)


def test_capacity_of_a_non_declarer_is_not_pooled() -> None:
    """Two producers own c0 (10 each), only p0 declares product 1: 10 a cycle, while B2 and
    B_base pool both producers' capacity (20 a cycle)."""
    cfg = _config([ProcessSheet(1, {"c0": 1}), ProcessSheet(2, {"c0": 1})],
                  [_producer("p0", 10, {1}), _producer("p1", 10, {2})])
    assert lp_bound(cfg, 3, _flat(3, BIG)).units == pytest.approx(30.0)
    assert world_ceiling(cfg, 3, {1: 3 * BIG})["B_base"].units == pytest.approx(60.0)


def test_product_mix_is_joint_across_capacity_types() -> None:
    """Products 1 (needs a: 1, b: 2) and 2 (a: 2, b: 1), 10 ordered of each, capacities a = b = 9
    for one cycle. Per type each product alone fits 9 or 4.5 units, the per-type minimum serves
    9; one joint mix serves at most 6 (3 of each)."""
    sheets = [ProcessSheet(1, {"a": 1, "b": 2}), ProcessSheet(2, {"a": 2, "b": 1})]
    p = ProducingAgentConfig(id="p", capacities={"a": 9, "b": 9}, degradation=0,
                             restoration=Restoration(require=1, restores=0, type="a"),
                             upgrade=Upgrade(require=1, increases=0, type="a"),
                             produces=frozenset({1, 2}))
    cfg = _config(sheets, [p])
    demand = {1: [10], 2: [10]}
    assert lp_bound(cfg, 1, demand).units == pytest.approx(6.0)
    assert ceiling_bounds(cfg, 1, {1: 10, 2: 10})["B2"].units == pytest.approx(9.0)


def test_cyclic_recipe_is_balanced_exactly() -> None:
    """1 needs one unit of 2 and 2 needs one unit of 1: every unit of 1 consumes a unit of 2 that
    consumes a unit of 1, so no net unit of 1 is ever left for consumers. B_lp = 0, where B2's
    transitive expansion ignores the repeated edge and stays positive."""
    sheets = [ProcessSheet(1, {"c": 2}, inputs={2: 1}), ProcessSheet(2, {"d": 3}, inputs={1: 1})]
    cfg = _config(sheets, [_producer("pc", 10, {1}, cap_type="c"),
                           _producer("pd", 30, {2}, cap_type="d")])
    assert lp_bound(cfg, 2, _flat(2, 10)).units == pytest.approx(0.0, abs=1e-6)
    assert ceiling_bounds(cfg, 2, {1: 20})["B2"].units > 0


def test_cycle_demand_sums_to_the_aggregate_schedule() -> None:
    sched = {"a": [(1, 2), (3, 1)], "b": [(1, 5)]}
    assert schedule_cycle_demand(sched, 3) == {1: [7, 5, 7], 3: [0, 1, 0]}
    cfg = _tight_corner_config()
    by_cycle = episode_cycle_demand(cfg, 12, 0)
    assert {p: sum(q) for p, q in by_cycle.items()} == episode_demand(cfg, 12, 0)


def test_demand_cap_when_nothing_binds() -> None:
    cfg = _config([ProcessSheet(1, {"c0": 1})], [_producer("p", 10, {1})])
    b = lp_bound(cfg, 4, _flat(4, 3))
    assert (b.units, b.binding_constraint) == (pytest.approx(12.0), DEMAND)


def test_program_size_scales_with_horizon() -> None:
    cfg = _tight_corner_config()
    small, large = (lp_size(cfg, h, episode_cycle_demand(cfg, h, 0)) for h in (6, 12))
    assert large.variables > small.variables and large.constraints > small.constraints


def _played(cfg: Configuration, seed: int, router: str, producing: str, horizon: int
            ) -> tuple[float, TokenomicsAECEnv]:
    policy = ORDERING_BASELINES.get(router) or DIAGNOSTIC_ROUTERS[router]
    contract = contract_for(producing, router) if (
        router in ORDERING_BASELINES and producing == "entity_planner") else JOINT_STACK
    env = (joint_stack_env(cfg, horizon) if contract == JOINT_STACK
           else TokenomicsAECEnv(config=cfg, horizon=horizon))
    stats = run_episode(env, PRODUCING_BASELINES[producing], policy, seed=seed, action_seed=7)
    return float(sum(stats["consumer_fulfilled_units"].values())), env


@pytest.mark.parametrize("router", [*ORDERING_BASELINES, "round_robin"])
def test_lp_bound_dominates_every_arm(router: str) -> None:
    """Scenario "Bound below an arm" never triggers for B_lp: on a calm registry world, a
    shocked f4 family world and the tight corner (where B_lp sits below demand), horizon 12,
    B_lp >= the consumer units of `entity_planner` and of the upgrading `always_both` producer
    with every roster ordering baseline and `round_robin`."""
    tight = _tight_corner_config()
    worlds = [*_bound_worlds(), ("tight-corner", tight, 0)]
    for world, cfg, seed in worlds:
        for producing in ("entity_planner", "always_both"):
            units, env = _played(cfg, seed, router, producing, 12)
            bound = lp_bound(cfg, 12, env_cycle_demand(env)).units
            assert bound + 0.01 >= units, (world, router, producing, bound, units)
    assert lp_bound(tight, 12, episode_cycle_demand(tight, 12, 0)).units < 0.5 * sum(
        episode_demand(tight, 12, 0).values())
