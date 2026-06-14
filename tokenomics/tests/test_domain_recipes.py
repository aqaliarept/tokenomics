"""B1: products & recipes — process sheets carry input products (composite) or
not (primal), and the recipe graph is queryable. See ADR-0025, plan B1."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenomics.domain.config import Configuration, ProcessSheet, load_config

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
