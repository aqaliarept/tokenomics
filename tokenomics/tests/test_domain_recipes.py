"""B1: products & recipes — process sheets carry input products (composite) or
not (primal), and the recipe graph is queryable. See ADR-0025, plan B1."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.domain.config import (
    Configuration,
    ProcessSheet,
    ProducingAgentConfig,
    Restoration,
    Upgrade,
    load_config,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "composite.json"


def test_primal_sheet_has_no_inputs():
    ps = ProcessSheet(product=1, require={"capacity-1": 10})
    assert ps.is_primal
    assert not ps.is_composite
    assert ps.inputs == {}


def test_composite_inputs_load_from_json():
    cfg = load_config(FIXTURE)
    # product 20 is composite: 2 units of primal 10 + capacity
    sheet = cfg.process_sheet(20)
    assert sheet.is_composite
    assert cfg.inputs_of(20) == {10: 2}
    assert cfg.inputs_of(10) == {}  # primal


def test_validate_rejects_input_without_sheet():
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=1, require={"capacity-1": 5}, inputs={99: 1}),
        ],
        producer_configs=load_config(FIXTURE).producer_configs,
    )
    with pytest.raises(ValueError, match="input product 99"):
        cfg.validate()


def test_validate_rejects_producing_product_without_required_capacity():
    # Eligibility is authoritative (ADR-0028): p1 *declares* it produces product 1,
    # but product 1 needs both capacity-1 and capacity-2 and p1 owns only
    # capacity-1 -> it cannot satisfy the recipe it claims. Rejected at config time.
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5, "capacity-2": 5})],
        producer_configs=[
            # p1 declares product 1 but owns only capacity-1; p2 supplies capacity-2
            # so the "type unprovided" check passes and we reach the produces check.
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                                 produces=frozenset({1})),
            ProducingAgentConfig("p2", {"capacity-2": 100}, 5,
                                 Restoration(1, 50, "capacity-2"), Upgrade(1, 40, "capacity-2")),
        ],
    )
    with pytest.raises(ValueError, match="does not own required capacity"):
        cfg.validate()


def test_validate_rejects_product_produced_by_no_one():
    # product 1 has a process sheet but no producer lists it in `produces`, so it
    # can never be ordered -> rejected (ADR-0028), replacing the old derived check.
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                                 produces=frozenset()),
        ],
    )
    with pytest.raises(ValueError, match="no producer produces"):
        cfg.validate()


def test_validate_rejects_producing_product_with_no_sheet():
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                                 produces=frozenset({1, 99})),
        ],
    )
    with pytest.raises(ValueError, match="produces product 99 with no process sheet"):
        cfg.validate()


def test_validate_accepts_producer_that_declares_owned_recipe():
    # p1 owns both required types AND declares it produces product 1 -> ok
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5, "capacity-2": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100, "capacity-2": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                                 produces=frozenset({1})),
        ],
    )
    cfg.validate()  # no raise


def _single_producer_cfg(consumers):
    return Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                                 produces=frozenset({1})),
        ],
        consumers=consumers,
    )


def test_validate_rejects_duplicate_consumer_id():
    from tokenomics.domain.config import ConsumerConfig

    cfg = _single_producer_cfg([ConsumerConfig("c1", 1), ConsumerConfig("c1", 1)])
    with pytest.raises(ValueError, match="duplicate consumer id"):
        cfg.validate()


def test_validate_rejects_consumer_producer_id_collision():
    # consumer and producer share the ordering-agent / env-agent namespace; a
    # collision creates duplicate PettingZoo agents that double-step one agent.
    from tokenomics.domain.config import ConsumerConfig

    cfg = _single_producer_cfg([ConsumerConfig("p1", 1)])
    with pytest.raises(ValueError, match="collides with producer id"):
        cfg.validate()


def _cost_cfg(**producer_kwargs):
    from tokenomics.domain.config import ConsumerConfig

    return Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig(
                "p1", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1}), **producer_kwargs,
            ),
        ],
        consumers=[ConsumerConfig("c1", 1)],
    )


def test_validate_rejects_non_positive_restoration_cost():
    from tokenomics.domain.config import ConsumerConfig

    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig(
                "p1", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1", cost=0), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1}),
            ),
        ],
        consumers=[ConsumerConfig("c1", 1)],
    )
    with pytest.raises(ValueError, match="cost must be positive"):
        cfg.validate()


def test_validate_rejects_non_positive_replenish_cost_per_unit():
    cfg = _cost_cfg(replenish_cost_per_unit=-3)
    with pytest.raises(ValueError, match="replenish_cost_per_unit must be positive"):
        cfg.validate()


def test_validate_rejects_non_positive_max_replenish():
    cfg = _cost_cfg(max_replenish=0)
    with pytest.raises(ValueError, match="max_replenish must be positive"):
        cfg.validate()


def test_recipe_closure_is_transitive():
    cfg = load_config(FIXTURE)
    # 30 -> {20, 11}; 20 -> {10}; 10, 11 primal. Closure excludes 30 itself.
    assert cfg.recipe_inputs_closure(30) == {20, 11, 10}
    assert cfg.recipe_inputs_closure(10) == set()


def test_recipe_closure_terminates_on_cycle():
    # 1 -> 2 -> 1: a circular recipe must not loop forever.
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[
            ProcessSheet(product=1, require={"capacity-1": 5}, inputs={2: 1}),
            ProcessSheet(product=2, require={"capacity-1": 5}, inputs={1: 1}),
        ],
        producer_configs=[],
    )
    assert cfg.recipe_inputs_closure(1) == {1, 2}


# --- B9 step 2: consumer demand-randomization bounds validate ---
def _base_consumer_cfg(**consumer_kw):
    from tokenomics.domain.config import ConsumerConfig

    return Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1"),
                produces=frozenset({1})),
        ],
        consumers=[ConsumerConfig("c1", product=1, **consumer_kw)],
    )


def test_validate_rejects_demand_pool_product_without_sheet():
    cfg = _base_consumer_cfg(products=(1, 99))
    with pytest.raises(ValueError, match="product 99 with no process sheet"):
        cfg.validate()


def test_validate_rejects_inverted_quantity_range():
    cfg = _base_consumer_cfg(quantity_min=4, quantity_max=2)
    with pytest.raises(ValueError, match="quantity_max"):
        cfg.validate()


def test_validate_rejects_zero_quantity_min():
    cfg = _base_consumer_cfg(quantity_min=0, quantity_max=3)
    with pytest.raises(ValueError, match="quantity_min"):
        cfg.validate()


def test_loader_parses_demand_randomization_fields():
    cfg = load_config(Path(__file__).resolve().parent / "fixtures" / "random_demand.json")
    c = cfg.consumers[0]
    assert c.product_pool == (20, 30)
    assert (c.quantity_min, c.quantity_max) == (1, 3)
    assert c.is_randomized
