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


def test_validate_rejects_product_with_no_eligible_producer():
    # product 1 needs BOTH capacity-1 and capacity-2, but no single producer owns
    # the whole bundle -> no eligible producer (ADR-0025). Must be rejected here,
    # not crash later in env decode on an empty eligible set.
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5, "capacity-2": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1")),
            ProducingAgentConfig("p2", {"capacity-2": 100}, 5,
                                 Restoration(1, 50, "capacity-2"), Upgrade(1, 40, "capacity-2")),
        ],
    )
    with pytest.raises(ValueError, match="no eligible producer"):
        cfg.validate()


def test_validate_accepts_product_with_single_owner_of_bundle():
    # one producer owning both required types makes the product eligible -> ok
    cfg = Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5, "capacity-2": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100, "capacity-2": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1")),
        ],
    )
    cfg.validate()  # no raise


def _single_producer_cfg(consumers):
    return Configuration(
        cycle_emission=1000,
        process_sheets=[ProcessSheet(product=1, require={"capacity-1": 5})],
        producer_configs=[
            ProducingAgentConfig("p1", {"capacity-1": 100}, 5,
                                 Restoration(1, 50, "capacity-1"), Upgrade(1, 40, "capacity-1")),
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
