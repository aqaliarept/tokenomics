"""EXP-0010C: learned local producer scorer inside the deterministic planner.

The learned component should supply local economic scores only. The public
contract stays:

    ProducerEntityView -> ProducerLocalScores -> deterministic planner -> command

so feasibility remains the planner's responsibility.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from tokenomics.domain import build_system, load_config
from tokenomics.env.aec_env import DEFAULT_CONFIG


def test_planner_accepts_external_scores_without_losing_headroom_constraints():
    """A learned scorer may reprioritize input values, but the deterministic planner
    still owns inventory headroom and emits the existing command contract."""
    from tokenomics.domain.producer_planner import (
        ProducerLocalScores,
        plan_producer_command,
    )

    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    view = system.producer_entity_view("p4")
    view = replace(view, globals=replace(view.globals, inventory_headroom=1))
    scores = ProducerLocalScores(input_value={11: 1.0, 20: 100.0})

    cmd = plan_producer_command(
        view,
        max_replenish=10,
        replenish_cost_per_unit=40,
        funding_budget=500,
        scores=scores,
    )

    assert cmd.replenishments == {20: 1}


def test_collect_producer_score_samples_records_rollout_views_and_labels():
    """Training data should come from real producer decision states, labeled with a
    local scorer that the learned model can imitate or improve."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.train.producer_scorer import (
        ProducerScoreSample,
        collect_producer_score_samples,
    )

    samples = collect_producer_score_samples(
        load_config(DEFAULT_CONFIG),
        PRODUCING_BASELINES["entity_planner"],
        ORDERING_BASELINES["proportional"],
        seeds=[0],
        horizon=4,
    )

    assert samples
    assert all(isinstance(s, ProducerScoreSample) for s in samples)
    assert {s.view.producer_id for s in samples}
    assert any(s.target.input_value for s in samples)


def _p4_view_with_input_stock(inv11: int, inv20: int):
    cfg = load_config(DEFAULT_CONFIG)
    system = build_system(cfg)
    p4 = system.producing_agents["p4"]
    p4.inventory[11] = inv11
    p4.inventory[20] = inv20
    return system.producer_entity_view("p4")


def test_learned_scorer_fits_input_value_labels_and_plans_through_constraints():
    """A shared local scorer can learn input-value scores from planner labels, and the
    deterministic planner can consume those learned scores without giving up feasibility."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.domain.producer_planner import (
        plan_producer_command,
        score_producer_entity_view,
    )
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        ProducerScoreSample,
        fit_producer_score_model,
        score_with_model,
    )

    torch.manual_seed(0)
    samples = [
        ProducerScoreSample(
            view=_p4_view_with_input_stock(inv11, inv20),
            target=score_producer_entity_view(_p4_view_with_input_stock(inv11, inv20)),
        )
        for inv11 in (0, 1, 3, 5)
        for inv20 in (0, 1, 3, 5)
    ]
    model = ProducerScoreModel(hidden=32)
    fit_producer_score_model(model, samples, epochs=60, lr=3e-2)

    held = _p4_view_with_input_stock(inv11=3, inv20=0)
    held = replace(held, globals=replace(held.globals, inventory_headroom=1))
    learned = score_with_model(model, held)
    cmd = plan_producer_command(
        held,
        max_replenish=10,
        replenish_cost_per_unit=40,
        funding_budget=500,
        scores=learned,
    )

    assert learned.input_value[20] > learned.input_value[11]
    assert cmd.replenishments == {20: 1}


def test_learned_scorer_policy_emits_a_valid_env_action():
    """Eval code should be able to deploy any local scorer closure through the normal
    producer policy interface for 10C comparisons."""
    from tokenomics.domain.producer_planner import ProducerLocalScores
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.eval.baselines import producing_entity_planner_from_scorer

    def scorer(view):
        return ProducerLocalScores(
            input_value={r.input_product: 1.0 for r in view.input_rows}
        )

    env = TokenomicsAECEnv(horizon=10, entity_producer_obs=True)
    env.reset(seed=0)
    action = producing_entity_planner_from_scorer(scorer)(env, "prod_p4")

    assert env.action_space("prod_p4").contains(action)


def _permute_entity_rows(view):
    from tokenomics.domain.producer_view import Relations

    out = copy.deepcopy(view)
    cap_order = list(reversed(range(len(view.capacity_rows))))
    prod_order = list(reversed(range(len(view.product_rows))))
    input_order = list(reversed(range(len(view.input_rows))))
    cap_pos = {old: new for new, old in enumerate(cap_order)}
    prod_pos = {old: new for new, old in enumerate(prod_order)}
    input_pos = {old: new for new, old in enumerate(input_order)}

    out.capacity_rows = [out.capacity_rows[i] for i in cap_order]
    out.product_rows = [out.product_rows[i] for i in prod_order]
    out.input_rows = [out.input_rows[i] for i in input_order]
    out.relations = Relations(
        product_capacity_required={
            (prod_pos[xi], cap_pos[ti]): amount
            for (xi, ti), amount in view.relations.product_capacity_required.items()
        },
        product_input_required={
            (prod_pos[xi], input_pos[gi]): amount
            for (xi, gi), amount in view.relations.product_input_required.items()
        },
    )
    return out


def test_learned_scorer_is_invariant_to_entity_row_order():
    """The model may emit row-aligned tensors internally, but public scores are keyed by
    product/input/capacity id and must not depend on row order."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import ProducerScoreModel, score_with_model

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16)
    view = _p4_view_with_input_stock(inv11=0, inv20=1)
    base = score_with_model(model, view)
    permuted = score_with_model(model, _permute_entity_rows(view))

    assert permuted.product_urgency == pytest.approx(base.product_urgency)
    assert permuted.input_value == pytest.approx(base.input_value)
    assert permuted.capacity_bottleneck_value == pytest.approx(base.capacity_bottleneck_value)
    assert permuted.restoration_roi == pytest.approx(base.restoration_roi)
    assert permuted.upgrade_roi == pytest.approx(base.upgrade_roi)
    assert permuted.inventory_expansion_value == pytest.approx(base.inventory_expansion_value)


def test_producer_scorer_checkpoint_round_trips(tmp_path):
    """A fitted scorer should be reusable as an experiment artifact."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        load_producer_score_model,
        save_producer_score_model,
        score_with_model,
    )

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16)
    view = _p4_view_with_input_stock(inv11=0, inv20=1)
    before = score_with_model(model, view)

    path = tmp_path / "producer_scorer.pt"
    save_producer_score_model(model, path)
    loaded = load_producer_score_model(path)
    after = score_with_model(loaded, view)

    assert after.input_value == pytest.approx(before.input_value)
    assert after.capacity_bottleneck_value == pytest.approx(before.capacity_bottleneck_value)
