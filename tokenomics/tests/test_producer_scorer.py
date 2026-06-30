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


def test_eval_producer_scorer_reports_value_vs_planner_across_held_out_seeds():
    """The producer analogue of the orderer `eval_scorer`: run the learned
    scorer+planner producing policy against the deterministic `entity_planner` teacher
    on held-out scale topologies and report a per-seed value ratio. Loading/running on a
    producer count the scorer was distilled at proves the id-free contract transfers."""
    import math

    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        collect_producer_score_samples,
        eval_producer_scorer,
        fit_producer_score_model,
    )

    torch.manual_seed(0)
    cfg = build_scale_config(0, depth=1)
    samples = collect_producer_score_samples(
        cfg,
        PRODUCING_BASELINES["entity_planner"],
        ORDERING_BASELINES["proportional"],
        seeds=[0],
        horizon=20,
    )
    model = ProducerScoreModel(hidden=16)
    fit_producer_score_model(model, samples, epochs=5, lr=3e-2)

    rows = eval_producer_scorer(model, depth=1, seeds=[6, 7], eval_seeds=1, horizon=20)

    assert [r["seed"] for r in rows] == [6, 7]
    for r in rows:
        assert r["depth"] == 1
        assert r["producers"] >= 1
        assert r["scorer_units"] >= 0.0
        assert r["planner_units"] >= 0.0
        assert math.isfinite(r["ratio"])
        # the EXP-0013 pass gate requires a no-death check, so the eval must surface
        # death rates for both policies (a row can hit an OK units ratio while dying).
        assert 0.0 <= r["dead"] <= 1.0
        assert 0.0 <= r["planner_dead"] <= 1.0


def test_distill_producer_scorer_reports_unseen_producer_counts(tmp_path):
    """Producer-count transfer must actually be exercised: distilling at depth 3 and
    asking for eval depths 1/2/3 should report rows at the smaller producer counts the
    scorer never trained on, not just the training P."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import distill_producer_scorer

    _, rows = distill_producer_scorer(
        depth=3,
        train_seeds=[0],
        eval_seeds_list=[6],
        eval_depths=[1, 2, 3],
        epochs=2,
        lr=3e-2,
        eval_seeds=1,
        horizon=12,
        hidden=16,
    )

    by_depth = {r["depth"]: r["producers"] for r in rows}
    assert set(by_depth) == {1, 2, 3}
    # producer count grows with depth on the scale family, so the smaller depths are
    # genuinely unseen producer counts relative to the depth-3 training topology.
    assert by_depth[1] < by_depth[3]


def test_distill_calibrates_and_persists_gate_thresholds(tmp_path):
    """The wrapped distill must calibrate per-gate probability thresholds on its training
    states and persist them on the model + checkpoint, so the cloud run deploys the tuned
    binary gates (EXP-0013), not the raw 0.5-default ones. The eval rows must also carry
    command-level fidelity (investment FP rates), which is what caught the bug."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import (
        _DEFAULT_GATE_THRESHOLDS,
        distill_producer_scorer,
        load_producer_score_model,
    )

    save = tmp_path / "producer_scorer.pt"
    model, rows = distill_producer_scorer(
        depth=2,
        train_seeds=[0],
        eval_seeds_list=[6],
        epochs=20,
        lr=3e-2,
        eval_seeds=1,
        horizon=20,
        hidden=16,
        save=str(save),
    )

    assert model.upgrade_threshold in _DEFAULT_GATE_THRESHOLDS
    assert model.inventory_expansion_threshold in _DEFAULT_GATE_THRESHOLDS
    loaded = load_producer_score_model(save)
    assert loaded.upgrade_threshold == model.upgrade_threshold
    assert loaded.inventory_expansion_threshold == model.inventory_expansion_threshold
    for r in rows:
        assert 0.0 <= r["upgrade_fp_rate"] <= 1.0
        assert 0.0 <= r["inventory_fp_rate"] <= 1.0
        assert 0.0 <= r["replen_mismatch_rate"] <= 1.0


def test_distill_producer_scorer_collects_fits_saves_and_evals(tmp_path):
    """One wrapped command for the cloud run: collect planner-labeled samples over train
    seeds, fit the scorer, save a reloadable checkpoint, and eval value vs the planner on
    held-out seeds — mirrors the orderer `distill_orderer_scorer`."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        distill_producer_scorer,
        load_producer_score_model,
    )

    save = tmp_path / "producer_scorer.pt"
    model, rows = distill_producer_scorer(
        depth=1,
        train_seeds=[0],
        eval_seeds_list=[6],
        epochs=3,
        lr=3e-2,
        eval_seeds=1,
        horizon=20,
        hidden=16,
        save=str(save),
    )

    assert isinstance(model, ProducerScoreModel)
    assert [r["seed"] for r in rows] == [6]
    assert save.exists()
    load_producer_score_model(save)  # reloads through the version gate


def test_scorer_policy_routes_planner_gate_decision_through_the_env_action():
    """The deployable producer policy must honor `planner_gates`: with the learned gates
    pinned off but the deterministic planner wanting the inventory expansion, the emitted
    env action sets the inventory-expansion toggle (it would stay off under learned gates).
    Same checkpoint, planner-owned investment gate — the P2/C3 deployment mode."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.env import TokenomicsAECEnv
    from tokenomics.train.producer_scorer import ProducerScoreModel, scorer_producing_policy

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16)
    model.upgrade_threshold = 1.0  # learned investment gates can never fire
    model.inventory_expansion_threshold = 1.0

    env = TokenomicsAECEnv(horizon=10, entity_producer_obs=True)
    env.reset(seed=0)
    agent = "prod_p4"
    pid = env._role[agent][1]
    # exhaust headroom so the planner's inventory-expansion gate would fire.
    p4 = env.system.producing_agents[pid]
    p4.inventory[11] = p4.inventory_limit

    learned = scorer_producing_policy(model)(env, agent)
    gated = scorer_producing_policy(model, planner_gates=True, replenish_tie_rel_tol=0.02)(env, agent)

    inv_slot = env.spec_.num_inputs + 2  # inventory-expansion toggle in the Box action
    assert env.action_space(agent).contains(gated)
    assert gated[inv_slot] > learned[inv_slot]


def test_planner_gates_override_learned_investment_gates_with_planner_decisions():
    """Out-of-family the learned investment gates generalize poorly (EXP-0017), so the
    deployment can hand the upgrade / inventory-expansion gate decisions back to the
    deterministic planner while keeping the learned continuous scores. With the learned
    gates pinned off, a view the planner would expand inventory on still gets a positive
    inventory-expansion gate, and the continuous scores are untouched."""
    import pytest

    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from tokenomics.domain.producer_planner import score_producer_entity_view
    from tokenomics.train.producer_scorer import ProducerScoreModel, score_with_model

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16)
    model.upgrade_threshold = 1.0  # learned gates can never fire
    model.inventory_expansion_threshold = 1.0
    # headroom exhausted + missing inputs -> the deterministic planner wants to expand.
    view = _p4_view_with_input_stock(inv11=0, inv20=0)
    view = replace(view, globals=replace(view.globals, inventory_headroom=0))
    planner = score_producer_entity_view(view)
    assert planner.inventory_expansion_value > 0.0  # precondition: planner would expand

    learned = score_with_model(model, view)  # learned gates -> off
    gated = score_with_model(model, view, planner_gates=True)

    assert learned.inventory_expansion_value == 0.0
    assert gated.inventory_expansion_value == planner.inventory_expansion_value
    assert gated.upgrade_roi == planner.upgrade_roi
    # continuous scores stay the learned heads, unaffected by gate ownership.
    assert gated.input_value == learned.input_value
    assert gated.product_urgency == learned.product_urgency
    assert gated.capacity_bottleneck_value == learned.capacity_bottleneck_value


def test_score_with_model_emits_binary_gate_scores_by_threshold():
    """Investment decisions are learned BINARY GATES, not thresholded ROI regressions: a
    head's logit -> probability, and the deterministic planner sees a clean `1.0`/`0.0`
    "do it" score depending on whether that probability clears the gate threshold. This
    stops regression noise reading as a spend signal (EXP-0013)."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        producer_gate_probs,
        score_with_model,
    )

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16)
    view = _p4_view_with_input_stock(inv11=0, inv20=1)
    p = producer_gate_probs(model, view)

    # threshold just under the gate prob -> fires a binary 1.0; just over -> 0.0.
    fire = score_with_model(
        model, view,
        upgrade_threshold=p["upgrade"] - 1e-6,
        inventory_expansion_threshold=p["inventory_expansion"] - 1e-6,
    )
    block = score_with_model(
        model, view,
        upgrade_threshold=min(1.0, p["upgrade"] + 1e-6),
        inventory_expansion_threshold=min(1.0, p["inventory_expansion"] + 1e-6),
    )

    assert fire.upgrade_roi == 1.0 and fire.inventory_expansion_value == 1.0
    assert block.upgrade_roi == 0.0 and block.inventory_expansion_value == 0.0
    # restoration is planner-owned: this field is never a learned spend signal.
    assert fire.restoration_roi == 0.0


def test_gate_heads_learn_planner_decisions_via_bce():
    """The gate heads are BCE-trained against the planner's gate booleans, so after fit a
    view the teacher upgrades on gets a higher upgrade probability than one it does not."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.domain.producer_planner import ProducerLocalScores
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        ProducerScoreSample,
        fit_producer_score_model,
        producer_gate_probs,
    )

    torch.manual_seed(0)
    upgrade_view = _p4_view_with_input_stock(inv11=0, inv20=0)
    no_upgrade_view = _p4_view_with_input_stock(inv11=5, inv20=5)
    samples = [
        ProducerScoreSample(view=upgrade_view, target=ProducerLocalScores(upgrade_roi=3.0)),
        ProducerScoreSample(view=no_upgrade_view, target=ProducerLocalScores(upgrade_roi=0.0)),
    ]
    model = ProducerScoreModel(hidden=16)
    fit_producer_score_model(model, samples, epochs=300, lr=3e-2)

    assert (
        producer_gate_probs(model, upgrade_view)["upgrade"]
        > producer_gate_probs(model, no_upgrade_view)["upgrade"]
    )


def test_calibrate_gate_thresholds_maximizes_command_agreement():
    """Per gate, pick the probability threshold with the best agreement to the planner's
    own gate booleans on held-out states — and never worse than the 0.5 default."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        calibrate_gate_thresholds,
        collect_producer_score_samples,
        fit_producer_score_model,
        producer_gate_probs,
    )

    torch.manual_seed(0)
    cfg = build_scale_config(0, depth=2)
    samples = collect_producer_score_samples(
        cfg,
        PRODUCING_BASELINES["entity_planner"],
        ORDERING_BASELINES["proportional"],
        seeds=[0],
        horizon=30,
    )
    model = ProducerScoreModel(hidden=16)
    fit_producer_score_model(model, samples, epochs=20, lr=3e-2)

    candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    chosen = calibrate_gate_thresholds(model, samples, candidates=candidates)

    def agree(key: str, label, t: float) -> int:
        return sum(
            int((producer_gate_probs(model, s.view)[key] >= t) == label(s)) for s in samples
        )

    up_label = lambda s: s.target.upgrade_roi > 0.0
    inv_label = lambda s: s.target.inventory_expansion_value > 0.0
    assert chosen["upgrade"] in candidates
    assert chosen["inventory_expansion"] in candidates
    assert agree("upgrade", up_label, chosen["upgrade"]) == max(
        agree("upgrade", up_label, c) for c in candidates
    )
    assert agree("upgrade", up_label, chosen["upgrade"]) >= agree("upgrade", up_label, 0.5)


def test_score_with_model_defaults_to_models_stored_thresholds():
    """Deployment (`scorer_producing_policy`) calls `score_with_model` with no explicit
    thresholds, so the model's calibrated per-gate thresholds must apply by default."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import ProducerScoreModel, score_with_model

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16)
    model.upgrade_threshold = 1.0  # unreachable prob -> gate never fires
    model.inventory_expansion_threshold = 1.0
    view = _p4_view_with_input_stock(inv11=0, inv20=1)

    scores = score_with_model(model, view)  # no explicit thresholds -> uses model's

    assert scores.upgrade_roi == 0.0
    assert scores.inventory_expansion_value == 0.0


def test_gate_thresholds_round_trip_in_checkpoint(tmp_path):
    """The calibrated per-gate thresholds travel with the checkpoint so the deployed
    scorer fires the gates it was tuned for."""
    import pytest

    pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        load_producer_score_model,
        save_producer_score_model,
    )

    model = ProducerScoreModel(hidden=16)
    model.upgrade_threshold = 0.7
    model.inventory_expansion_threshold = 0.3
    path = tmp_path / "producer_scorer.pt"
    save_producer_score_model(model, path)

    loaded = load_producer_score_model(path)
    assert loaded.upgrade_threshold == 0.7
    assert loaded.inventory_expansion_threshold == 0.3


def test_v1_checkpoint_rejected_by_version_gate(tmp_path):
    """v1 checkpoints regressed ROI with MSE; the v2 heads emit BCE gate logits, so a v1
    state dict must not silently deploy. The version gate rejects it."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.train.producer_scorer import ProducerScoreModel, load_producer_score_model

    model = ProducerScoreModel(hidden=16)
    path = tmp_path / "v1.pt"
    torch.save(
        {"version": 1, "hidden": model.hidden, "state_dict": model.state_dict()},
        path,
    )

    with pytest.raises(ValueError):
        load_producer_score_model(path)


def test_producer_command_agreement_reports_investment_false_positives():
    """EXP-0013 step 4: report command-level fidelity, not just units. A model whose gate
    thresholds are pinned so high the gates never fire must show zero investment false
    positives (it can only ever under-fire vs the teacher)."""
    import pytest

    torch = pytest.importorskip("torch")
    from tokenomics.eval.scale import build_scale_config
    from tokenomics.train.producer_scorer import (
        ProducerScoreModel,
        producer_command_agreement,
    )

    torch.manual_seed(0)
    model = ProducerScoreModel(hidden=16)
    model.upgrade_threshold = 1.0
    model.inventory_expansion_threshold = 1.0
    cfg = build_scale_config(0, depth=2)

    metrics = producer_command_agreement(model, cfg, seeds=[0], horizon=30)

    assert metrics["n"] > 0
    assert metrics["upgrade_fp_rate"] == 0.0
    assert metrics["inventory_expansion_fp_rate"] == 0.0
    for key in ("upgrade_fn_rate", "replenishment_mismatch_rate"):
        assert 0.0 <= metrics[key] <= 1.0


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
