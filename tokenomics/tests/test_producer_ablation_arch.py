"""Ablation #2 (step 14): flat-MLP / no-edge baselines vs the row+edge scorer.

These variants exist ONLY to answer "does the row/edge factoring buy portability".
Behavior under test is the architecture contract, not economic value:
  - `use_edges=False` (per-row, no recipe edges): loads on variable topologies (per-row,
    so it CAN transfer) but its per-row scores are INVARIANT to recipe relations, isolating
    the edge contribution.
  - flat-MLP: non-portable by construction — a fixed-size padded input head that cannot
    ingest a wider product/producer count than it was built for.
"""

from __future__ import annotations

from dataclasses import replace

from tokenomics.domain import build_system, load_config
from tokenomics.env.aec_env import DEFAULT_CONFIG


def _p4_view():
    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    return system.producer_entity_view("p4")


def _drop_edges(view):
    return replace(
        view,
        relations=replace(
            view.relations,
            product_capacity_required={},
            product_input_required={},
        ),
    )


def test_no_edge_model_scores_are_invariant_to_recipe_edges():
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import ProducerScoreModel, score_with_model

    torch.manual_seed(0)
    view = _p4_view()
    no_edge = ProducerScoreModel(hidden=16, use_edges=False)
    with_edges = ProducerScoreModel(hidden=16, use_edges=True)

    ne_full = score_with_model(no_edge, view)
    ne_drop = score_with_model(no_edge, _drop_edges(view))
    assert ne_full.product_urgency == pytest.approx(ne_drop.product_urgency)
    assert ne_full.capacity_bottleneck_value == pytest.approx(ne_drop.capacity_bottleneck_value)

    we_full = score_with_model(with_edges, view)
    we_drop = score_with_model(with_edges, _drop_edges(view))
    # the edge model actually uses recipe structure, so dropping edges moves scores
    assert we_full.product_urgency != pytest.approx(we_drop.product_urgency)


def test_no_edge_model_round_trips_and_stamps_its_arch(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.portable_schema import PortableSchema
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        load_producer_score_model,
        producer_portable_schema,
        save_producer_score_model,
        score_with_model,
    )

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16, use_edges=False)
    path = tmp_path / "no_edge.pt"
    save_producer_score_model(model, path)

    loaded = load_producer_score_model(path)
    assert loaded.use_edges is False
    view = _p4_view()
    assert score_with_model(loaded, view).product_urgency == pytest.approx(
        score_with_model(model, view).product_urgency
    )

    # the checkpoint's schema records the no-edge architecture, distinct from the row+edge
    # default, so the portable gate documents which variant it is
    ckpt = torch.load(path, weights_only=True)
    saved = PortableSchema.from_dict(ckpt["schema"])
    assert saved.architecture_version != producer_portable_schema().architecture_version
    assert saved.architecture_version == producer_portable_schema(use_edges=False).architecture_version


def test_flat_model_scores_within_pad_but_is_non_portable_beyond_it():
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import (
        ProducerFlatModel,
        flat_model_pads_for,
        score_with_model,
    )

    torch.manual_seed(0)
    view = _p4_view()
    pads = flat_model_pads_for(view)
    model = ProducerFlatModel(hidden=16, **pads)

    scores = score_with_model(model, view)
    # per-row scores align to the actual entity ids the planner keys on
    assert set(scores.product_urgency) <= {r.product for r in view.product_rows}
    assert set(scores.capacity_bottleneck_value) <= {r.capacity_type for r in view.capacity_rows}

    # NON-PORTABLE BY CONSTRUCTION: a topology wider than the fixed pad the flat head was
    # built for cannot be ingested — the row/edge scorer would just add rows.
    from dataclasses import replace as _replace

    oversize = _replace(view, capacity_rows=view.capacity_rows + view.capacity_rows)
    with pytest.raises(ValueError):
        score_with_model(model, oversize)


def test_transfer_matrix_row_labels_producer_architecture():
    """EXP-0022 caveat: a no-edge run's transfer rows must be distinguishable from the
    row+edge champion's — otherwise both look like generic portable_scorer rows."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.eval.transfer_matrix import (
        StackSpec,
        TargetTopology,
        evaluate_stack_on_target,
    )
    from tokenomics.train.producer_scorer import ProducerScoreModel

    target = TargetTopology(
        name="scale-d1-s6", config=build_scale_config(6, depth=1), depth=1
    )
    stack = StackSpec(
        "portable_producer_planner_gates+proportional",
        "portable_scorer_planner_gates",
        "proportional",
    )

    edge = evaluate_stack_on_target(
        stack, target, eval_seeds=1, producer_model=ProducerScoreModel(hidden=8, use_edges=True)
    )
    noedge = evaluate_stack_on_target(
        stack, target, eval_seeds=1, producer_model=ProducerScoreModel(hidden=8, use_edges=False)
    )
    assert edge["producer_arch"] == "producer-row-edge-scorer-v2"
    assert noedge["producer_arch"] == "producer-row-noedge-scorer-v2"
