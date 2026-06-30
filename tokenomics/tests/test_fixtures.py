"""Out-of-family / scarcity / degenerate fixture gates (PRODUCT_AGNOSTIC steps 5 + 9).

These fixtures are the C3/P3 transfer targets: economies that leave the layered
single-input scale family (`build_scale_config`) along one axis — recipe DAG, capacity
sharing, producer strength, scarcity stress, or a degenerate edge. A fixture is only
useful if it (a) validates as a real economy and (b) genuinely departs the family along
its named axis — otherwise a "transfer" row proves nothing. These tests pin both before
the matrix consumes them.
"""

from __future__ import annotations

from collections import defaultdict

from tokenomics.eval.fixtures import (build_deep_join_config, build_forkjoin_config,
                                      build_shared_capacity_config,
                                      build_uneven_strength_config, degenerate_suite,
                                      scarcity_suite)
from tokenomics.eval.scale import build_scale_config


def _max_inputs(cfg) -> int:
    return max((len(ps.inputs) for ps in cfg.process_sheets), default=0)


def _cap_types(cfg) -> int:
    return len({ct for ps in cfg.process_sheets for ct in ps.require})


def test_forkjoin_fixture_is_valid_and_out_of_family() -> None:
    # The scale family is single-input (every composite product consumes exactly one
    # lower product). A fork/join fixture must carry at least one MULTI-input product —
    # that is what makes it out-of-family, not merely a re-seeded scale economy.
    assert _max_inputs(build_scale_config(0, depth=3)) == 1

    cfg = build_forkjoin_config(0, depth=3, width=4, inputs_per_product=2)
    cfg.validate()  # must be a real, feasible economy (raises otherwise)
    assert _max_inputs(cfg) == 2  # genuinely out-of-family: join of 2 inputs


def test_shared_capacity_fixture_couples_levels_through_fewer_types() -> None:
    # The scale family owns exactly one capacity type PER LEVEL (distinct types == depth) —
    # degradation/scarcity on a type touches a single level. A shared-capacity fixture is
    # out-of-family precisely because levels SHARE capacity types: fewer types than levels,
    # so one capacity pool couples several layers of the chain.
    assert _cap_types(build_scale_config(0, depth=3)) == 3

    cfg = build_shared_capacity_config(0, depth=3, width=4)
    cfg.validate()  # must be a real, feasible economy
    assert _cap_types(cfg) < 3  # levels share capacity types


def test_deep_join_fixture_is_deeper_and_wider_than_forkjoin() -> None:
    # forkjoin (the step-6 baseline out-of-family probe) is depth 3, join of 2. The
    # "deeper joins" fixture must push BOTH axes further — a longer chain AND wider joins —
    # so the matrix exercises recipe-graph depth, not just a single 2-input join.
    cfg = build_deep_join_config(0, depth=5, width=4, inputs_per_product=3)
    cfg.validate()  # must be a real, feasible economy
    assert _max_inputs(cfg) == 3  # wider join than forkjoin's 2
    # deeper chain: a top product reaches >= 4 transitive input products (depth-5 chain)
    top = max(ps.product for ps in cfg.process_sheets)
    assert len(cfg.recipe_inputs_closure(top)) >= 4


def test_uneven_strength_fixture_guarantees_extreme_within_level_spread() -> None:
    # The scale family's strong/weak split is RANDOM per producer (rng.random() < 0.6), so
    # a level can come out all-strong (no dispersion). The uneven-strength fixture is a
    # DEDICATED axis: every level is guaranteed one dominant producer and the rest weak, so
    # routing must concentrate. Pin a hard >= 8x within-level capacity gap on every level.
    cfg = build_uneven_strength_config(0, depth=3, width=4)
    cfg.validate()  # must be a real, feasible economy
    by_type: dict[str, list[int]] = defaultdict(list)
    for p in cfg.producer_configs:
        (cap_type, cap), = p.capacities.items()  # one type per producer in this family
        by_type[cap_type].append(cap)
    for caps in by_type.values():
        assert max(caps) / min(caps) >= 8


def _dims(cfg) -> tuple:
    return (len(cfg.producer_configs), len(cfg.process_sheets), len(cfg.consumers),
            _cap_types(cfg))


def test_scarcity_suite_stresses_one_axis_each_without_changing_dims() -> None:
    # The scarcity family stresses an existing economy along ONE axis at a time, staying
    # dims-preserving (same producers/products/consumers/cap-types) so a policy built on the
    # base loads unchanged and a failure is attributable to the stress, not a reshape.
    base = build_scale_config(0, depth=3)
    suite = scarcity_suite(base)
    for name in ("low_funding", "low_inventory", "high_backlog", "degraded",
                 "running_investment"):
        assert name in suite, name
    for cfg in suite.values():
        cfg.validate()
        assert _dims(cfg) == _dims(base)  # dims-preserving

    # each named axis moved in the harder direction
    assert suite["low_funding"].cycle_emission < base.cycle_emission
    inv = lambda c: sum(p.inventory_limit for p in c.producer_configs)
    assert inv(suite["low_inventory"]) < inv(base)
    assert max(c.quantity_max for c in suite["high_backlog"].consumers) \
        > max(c.quantity_max for c in base.consumers)
    deg = lambda c: max(p.degradation for p in c.producer_configs)
    assert deg(suite["degraded"]) > deg(base)
    cap = lambda c: sum(sum(p.capacities.values()) for p in c.producer_configs)
    assert cap(suite["running_investment"]) < cap(base)  # lean start forces early invest


def test_degenerate_suite_covers_edge_economies() -> None:
    # Degenerate fixtures exercise the boundaries of the producer/orderer contract: a single
    # eligible producer (no routing alternatives), an all-primal economy (no input
    # products), no inventory expansion, trivially-satisfiable demand (no backlog), and a
    # minimal economy that is mostly padding once encoded. Each must still be a valid
    # economy so the matrix can run it.
    suite = degenerate_suite(0)
    for name in ("single_producer", "primal_only", "no_inventory_expansion",
                 "no_backlog", "minimal"):
        assert name in suite, name
    for cfg in suite.values():
        cfg.validate()

    sp = suite["single_producer"]  # zero eligible alternatives: one declarer per product
    for ps in sp.process_sheets:
        declarers = [p for p in sp.producer_configs if ps.product in p.produces]
        assert len(declarers) == 1

    assert _max_inputs(suite["primal_only"]) == 0  # no input products
    assert all(p.inventory_expansion is None
               for p in suite["no_inventory_expansion"].producer_configs)
    assert all(c.quantity_max == 1 for c in suite["no_backlog"].consumers)  # no backlog
    assert len(suite["minimal"].process_sheets) <= 3  # one product per level: mostly padded
