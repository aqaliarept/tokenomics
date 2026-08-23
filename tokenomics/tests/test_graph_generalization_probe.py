"""EXP-0041 pooled probe: batching, arms, validation-only selection, and Offline Gate A.

The gate these lock is procedural as much as numerical: a final-test row must not be able to
reach a normalizer, an early-stopping decision, an arm choice, or an adapter choice, and the
offline verdict must be recomputable from prediction CSVs alone. Adversarial cases check that a
result which should FAIL cannot pass.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenomics.eval import graph_generalization_dump as gg  # noqa: E402
from tokenomics.train import graph_generalization_probe as gp  # noqa: E402
from tokenomics.train.graph_router_probe import RelationalGNN  # noqa: E402

from test_graph_view import _scale_task  # noqa: E402


# --- tiny end-to-end dump used by the integration tests ---------------------------------
def _fast_training() -> dict:
    return {"max_epochs": 6, "min_epochs": 1, "validation_every": 2,
            "early_stop_patience": 1, "batch_decisions_per_task": 4}


def _build_dump(tmp_path, *, seeds=(0, 1, 2, 3), horizon: int = 6, lookahead: int = 1) -> str:
    """Two tasks x (2 train, 1 validation, 1 test) episodes at smoke scale — enough shape for
    pooling, split isolation, and per-task arms without being a real run. Label collection is the
    expensive step, so this is built ONCE per module and shared."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    tasks = [_scale_task(0, depth=3), _scale_task(1, depth=3)]
    reg = {t.task_id: {"train": seeds[:2], "validation": (seeds[2],), "test": (seeds[3],)}
           for t in tasks}
    out = str(tmp_path / "dump")
    gg.run_generalization_dump(
        tasks, prod=PRODUCING_BASELINES["stock_aware"], target_mode="single_suffix",
        delta=1.0, lookahead=lookahead, horizon=horizon, out_dir=out, registry=reg,
        strict_splits=False, require_license=False, log=lambda *a, **k: None)
    return out


@pytest.fixture(scope="module")
def dump_dir(tmp_path_factory) -> str:
    return _build_dump(tmp_path_factory.mktemp("gendump"))


@pytest.fixture(scope="module")
def fit_dir(tmp_path_factory, dump_dir) -> str:
    """One frozen fit (2 arms x 2 model seeds, a handful of epochs) shared by the artifact,
    Gate A, and run-order tests."""
    out = str(tmp_path_factory.mktemp("fit") / "fit")
    gp.run_fit(dump_dir, out, arms=("row_mlp_h32", "graph_0hop_h32"), model_seeds=(0, 1),
               training=_fast_training(), train_size=2, curve_sizes=(2,),
               require_shape=False, log=lambda *a, **k: None)
    return out


@pytest.fixture(scope="module")
def curve_fit_dir(tmp_path_factory, dump_dir) -> str:
    """A fit that actually runs the mandatory 2/4/8 curve, at smoke scale: the curve arm only,
    with one size refitted and one reusing the primary fit (the two branches of the curve loop)."""
    out = str(tmp_path_factory.mktemp("curvefit") / "fit")
    gp.run_fit(dump_dir, out, arms=(gp.CURVE_ARM,), model_seeds=(0, 1),
               training=_fast_training(), train_size=2, curve_sizes=(1, 2),
               require_shape=False, log=lambda *a, **k: None)
    return out


def _decisions(n_tasks: int = 2, per_task: int = 5, k: int = 3) -> list[gp.Decision]:
    rng = np.random.default_rng(0)
    out = []
    for t in range(n_tasks):
        for j in range(per_task):
            out.append(gp.Decision(
                index=len(out), task_id=f"task{t}", split="train", seed=j, cycle=j,
                order_id=f"o{j}", flat=False, producer_ids=[f"p{i}" for i in range(k)],
                labels=rng.normal(size=k), rows=rng.normal(size=(k, 7)).astype(np.float32)))
    return out


# --- batching + loss --------------------------------------------------------------------
def test_batches_keep_decisions_intact_and_balance_tasks():
    """A minibatch never splits a decision (the loss centers within a decision, so half a
    decision is a corrupted target) and draws the same number of decisions from each task while
    both have rows. The shuffle is a pure function of (model_seed, epoch)."""
    decisions = _decisions(n_tasks=2, per_task=5)
    idx = list(range(len(decisions)))
    batches = gp.task_balanced_batches(decisions, idx, model_seed=0, epoch=0, per_task=2)
    seen = [i for b in batches for i in b]
    assert sorted(seen) == idx and len(seen) == len(set(seen))  # every decision exactly once
    for b in batches[:2]:  # while both tasks still have chunks, the batch is balanced
        counts = {}
        for i in b:
            counts[decisions[i].task_id] = counts.get(decisions[i].task_id, 0) + 1
        assert set(counts.values()) == {2}
    assert gp.task_balanced_batches(decisions, idx, model_seed=0, epoch=0, per_task=2) == batches
    assert gp.task_balanced_batches(decisions, idx, model_seed=1, epoch=0, per_task=2) != batches
    assert gp.task_balanced_batches(decisions, idx, model_seed=0, epoch=1, per_task=2) != batches


def test_task_balanced_loss_weights_tasks_equally_not_by_decision_count():
    """One task with many decisions must not define the pooled fit. With a perfect fit on the
    big task and a fixed error on the small one, the loss equals half that error — the
    equal-weight mean across tasks — not the decision-weighted average."""
    pred = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0])
    labels = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    group = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    task = torch.tensor([0, 0, 0, 1])  # 3 decisions in task 0, 1 in task 1
    loss = gp.task_balanced_loss(pred, labels, group, task, 4, 2)
    assert float(loss) == pytest.approx(0.5)  # (0 + 1) / 2, not 1/4


def test_epoch_task_weights_equalize_total_gradient_weight_over_the_epoch():
    """Per-batch balancing alone is not balance: a task with more decisions owns the tail batches
    of the epoch by itself and, weighted equally *within* each batch, ends up with more total
    gradient weight than a smaller task. The registered topologies differ by 1.43x in decisions,
    which under per-batch-only weighting is a 2.16x gradient-weight skew. `epoch_task_weights`
    weights each task by 1/(its batch count) so the epoch totals match."""
    decisions = _decisions(n_tasks=1, per_task=4)  # task0: 4 decisions
    decisions += [gp.Decision(index=99, task_id="task1", split="train", seed=j, cycle=j,
                              order_id=f"x{j}", flat=False, producer_ids=["p0", "p1"],
                              labels=np.zeros(2), rows=np.zeros((2, 7), dtype=np.float32))
                  for j in range(12)]  # task1: 12 decisions -> 3x the batches
    idx = list(range(len(decisions)))
    batches = gp.task_balanced_batches(decisions, idx, model_seed=0, epoch=0, per_task=4)
    weights = gp.epoch_task_weights(decisions, idx, per_task=4)

    per_batch_total, weighted_total = {"task0": 0.0, "task1": 0.0}, {"task0": 0.0, "task1": 0.0}
    for b in batches:
        present = {decisions[i].task_id for i in b}
        for t in present:
            per_batch_total[t] += 1.0 / len(present)   # old behaviour: equal share of this batch
            weighted_total[t] += weights[t]
    assert per_batch_total["task1"] > 2 * per_batch_total["task0"]   # the skew being fixed
    assert weighted_total["task0"] == pytest.approx(weighted_total["task1"])

    # equal-sized tasks must reduce EXACTLY to the old uniform 1/n_tasks weighting
    even = _decisions(n_tasks=2, per_task=4)
    assert set(gp.epoch_task_weights(even, list(range(len(even))), per_task=4).values()) == {0.5}


def test_task_balanced_loss_honours_supplied_task_weights():
    """The weights reach the loss: the same predictions score differently when one task is
    down-weighted, and uniform weights reproduce the unweighted mean."""
    pred = torch.tensor([1.0, -1.0, 0.0, 0.0])
    labels = torch.tensor([1.0, -1.0, 1.0, -1.0])
    group, task = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1])
    uniform = torch.tensor([0.5, 0.5])
    assert float(gp.task_balanced_loss(pred, labels, group, task, 2, 2, task_weight=uniform)) == \
        pytest.approx(float(gp.task_balanced_loss(pred, labels, group, task, 2, 2)))
    skewed = torch.tensor([0.9, 0.1])  # task1 (the one with the error) down-weighted
    assert float(gp.task_balanced_loss(pred, labels, group, task, 2, 2, task_weight=skewed)) == \
        pytest.approx(0.1)


# --- architecture -------------------------------------------------------------------------
def test_zero_hop_arm_runs_and_dropout_default_is_a_no_op():
    """`layers=0` is a legal forward pass (the 0-hop control that asks whether message passing
    buys anything), and the added `dropout` knob defaults to a literal no-op so the EXP-0039
    gate path is numerically unchanged."""
    from tokenomics.eval.graph_value_router_dump import encode_graph
    from tokenomics.train.graph_router_probe import Batch
    from test_graph_view import first_contested_decision
    from tokenomics.domain.graph_view import build_economy_graph_view

    env, aid, oid, elig = first_contested_decision()
    g = encode_graph(build_economy_graph_view(env.system, aid, oid, list(elig)))
    batch = Batch([g])
    torch.manual_seed(0)
    zero = RelationalGNN(hidden=8, layers=0)
    out = zero(batch)
    assert out.shape == (len(elig),) and torch.isfinite(out).all()

    torch.manual_seed(0)
    a = RelationalGNN(hidden=8, layers=2)
    torch.manual_seed(0)
    b = RelationalGNN(hidden=8, layers=2, dropout=0.0)
    assert torch.equal(a(batch), b(batch))  # same params, same math
    torch.manual_seed(0)
    drop = RelationalGNN(hidden=8, layers=2, dropout=0.5)
    drop.train()
    assert not torch.equal(drop(batch), drop(batch))  # dropout is actually active in train mode
    drop.eval()
    assert torch.equal(drop(batch), drop(batch))
    with pytest.raises(ValueError):
        RelationalGNN(hidden=8, layers=-1)


# --- dump loading + split isolation --------------------------------------------------------
def test_loader_validates_alignment_and_can_physically_exclude_test_rows(dump_dir):
    """The fit stage loads only train+validation, so a final-test row cannot enter a normalizer
    or a selection decision even by accident. Alignment (candidate order, producer decode, digest
    binding, one split per episode) is validated on load."""
    full = gp.load_generalization_dump(dump_dir)
    fit = gp.load_generalization_dump(dump_dir, splits=("train", "validation"))
    assert {d.split for d in full.decisions} == {"train", "validation", "test"}
    assert {d.split for d in fit.decisions} == {"train", "validation"}
    assert len(fit.decisions) < len(full.decisions) and len(fit.graphs) == len(fit.decisions)
    for d in full.decisions:
        assert len(d.labels) == len(d.producer_ids) == d.rows.shape[0] >= 2
    with pytest.raises(ValueError):  # the dump's target mode is asserted, never assumed
        gp.load_generalization_dump(dump_dir, expected_target_mode="continuation_mean")


def test_loader_rejects_a_corrupted_alignment_and_a_split_crossing_episode(tmp_path, dump_dir):
    """Adversarial: a swapped candidate order and an episode that appears in two splits must
    both raise — they would silently mislabel a candidate or leak an episode across the wall."""
    import shutil

    broken = str(tmp_path / "broken")
    shutil.copytree(dump_dir, broken)
    rows = list(csv.DictReader(open(f"{broken}/candidates.csv")))
    rows[0]["producer_id"] = "not-a-producer"
    _rewrite(f"{broken}/candidates.csv", rows)
    with pytest.raises(ValueError):
        gp.load_generalization_dump(broken)

    leaky = str(tmp_path / "leaky")
    shutil.copytree(dump_dir, leaky)
    decs = list(csv.DictReader(open(f"{leaky}/decisions.csv")))
    cands = list(csv.DictReader(open(f"{leaky}/candidates.csv")))
    victim = next(d for d in decs if d["split"] == "train")
    for d in decs:  # same episode, two split labels
        if d["task_id"] == victim["task_id"] and d["trajectory_seed"] == victim["trajectory_seed"]:
            d["split"] = "test" if d is victim else d["split"]
    for c in cands:
        if (c["decision_index"] == victim["decision_index"]):
            c["split"] = "test"
    _rewrite(f"{leaky}/decisions.csv", decs)
    _rewrite(f"{leaky}/candidates.csv", cands)
    with pytest.raises(ValueError):
        gp.load_generalization_dump(leaky)


def _rewrite(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_the_fit_stage_never_parses_a_final_test_label(tmp_path, dump_dir):
    """The strongest form of the split wall: poison every final-test label and actor feature with
    a value that is not a number. The fit load (`train`+`validation`) must still succeed — proving
    those cells are never parsed, not merely never returned — while a full load blows up on them.

    Returning-but-not-using would leave the test target one refactor away from a normalizer."""
    import shutil

    poisoned = str(tmp_path / "poisoned")
    shutil.copytree(dump_dir, poisoned)
    rows = list(csv.DictReader(open(f"{poisoned}/candidates.csv")))
    n = 0
    for r in rows:
        if r["split"] == "test":
            r["primary_label"] = r["original_suffix_label"] = "LEAK"
            r["feat_3"] = "LEAK"
            n += 1
    assert n, "fixture has no test rows to poison"
    _rewrite(f"{poisoned}/candidates.csv", rows)

    fit = gp.load_generalization_dump(poisoned, splits=("train", "validation"))
    assert {d.split for d in fit.decisions} == {"train", "validation"}
    assert all(np.isfinite(d.labels).all() for d in fit.decisions)
    with pytest.raises(ValueError):
        gp.load_generalization_dump(poisoned)


def test_control_view_swaps_in_the_single_suffix_target():
    """In mean mode the handover requires a single-suffix CONTROL fitted on the same rows. The
    control view must replace BOTH the label and the flatness flag — a decision can be flat under
    the averaged label and contested under the raw one, and training on a stale flat flag would
    quietly change which decisions the control sees."""
    d = gp.Decision(index=0, task_id="t", split="train", seed=0, cycle=0, order_id="o",
                    flat=True, producer_ids=["p0", "p1"], labels=np.array([1.0, 1.0]),
                    rows=np.zeros((2, 7), dtype=np.float32),
                    original_labels=np.array([0.0, 5.0]), original_flat=False)
    view = gp.control_view(gp.LoadedDump(graphs=[], decisions=[d], spec={}))
    assert np.array_equal(view.decisions[0].labels, np.array([0.0, 5.0]))
    assert view.decisions[0].flat is False
    assert np.array_equal(d.labels, np.array([1.0, 1.0]))  # the original dump is not mutated
    assert gp.ARMS[gp.CONTROL_ARM]["target"] == "original_suffix"
    assert gp.CONTROL_ARM not in gp.DEFAULT_ARMS  # only fitted when the mean target is licensed
    assert gp.CONTROL_ARM not in gp.SELECTABLE_ARMS


def test_mean_mode_fit_adds_the_single_suffix_control_arm(tmp_path):
    """A control that has to be remembered is a control that goes missing. On a
    `continuation_mean` dump the pooled single-suffix control is fitted automatically, on the same
    graph rows and model seeds, and its predictions are persisted like any other arm."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    task = _scale_task(0, depth=3)
    reg = {task.task_id: {"train": (0, 1), "validation": (2,), "test": (3,)}}
    dump = str(tmp_path / "mean_dump")
    gg.run_generalization_dump(
        [task], prod=PRODUCING_BASELINES["stock_aware"], target_mode="continuation_mean",
        continuations=2, delta=1.0, lookahead=1, horizon=6, out_dir=dump, registry=reg,
        splits=("train", "validation"), strict_splits=False, require_license=False,
        log=lambda *a, **k: None)
    out = str(tmp_path / "mean_fit")
    spec = gp.run_fit(dump, out, arms=("row_mlp_h32",), model_seeds=(0,),
                      training={"max_epochs": 2, "min_epochs": 1, "validation_every": 2,
                                "early_stop_patience": 1, "batch_decisions_per_task": 4},
                      train_size=2, curve_sizes=(), require_shape=False,
                      log=lambda *a, **k: None)
    assert spec["control_arm"] == gp.CONTROL_ARM
    preds = list(csv.DictReader(open(f"{out}/predictions.csv")))
    assert gp.CONTROL_ARM in {r["arm"] for r in preds}
    # the control's rows carry the ORIGINAL suffix label as their target, not the averaged one
    original = {(c["decision_index"], c["candidate_index"]): c["original_suffix_label"]
                for c in csv.DictReader(open(f"{dump}/candidates.csv"))}
    control = [r for r in preds if r["arm"] == gp.CONTROL_ARM]
    assert control and all(
        float(r["primary_label"]) == pytest.approx(float(original[(r["decision_index"],
                                                                   r["candidate_index"])]))
        for r in control)


def test_a_certified_fit_must_also_be_the_pre_registered_RUN(tmp_path, dump_dir):
    """`require_shape` certified only the DATASET, so a one-seed, one-arm, one-epoch fit could
    still stamp itself certified and go on to license an adapter and a rollout. The fit contract
    is now checked too — before any training compute is spent — and re-checked from the written
    spec by the rollout stages."""
    with pytest.raises(ValueError, match="model seeds"):
        gp.run_fit(dump_dir, str(tmp_path / "one_seed"), model_seeds=(0,),
                   source_commit="a" * 40, log=lambda *a, **k: None)
    with pytest.raises(ValueError, match="arms"):
        gp.run_fit(dump_dir, str(tmp_path / "one_arm"), arms=("row_mlp_h32",),
                   source_commit="a" * 40, log=lambda *a, **k: None)
    with pytest.raises(ValueError, match="training"):
        gp.run_fit(dump_dir, str(tmp_path / "short"), training={"max_epochs": 1},
                   source_commit="a" * 40, log=lambda *a, **k: None)
    with pytest.raises(ValueError, match="commit"):
        gp.run_fit(dump_dir, str(tmp_path / "uncommitted"), log=lambda *a, **k: None)

    good = {"require_shape": True, "arms": {a: {} for a in gp.DEFAULT_ARMS},
            "model_seeds": list(gp.MODEL_SEEDS), "train_size": 8,
            "curve_sizes": list(gp.NESTED_TRAIN_SIZES), "training": dict(gp.TRAINING),
            "source": {"commit": "a" * 40, "tree_sha256": "b" * 64}}
    gp.certify_fit_spec(good)
    for bad in ({"require_shape": False}, {"model_seeds": [0]}, {"train_size": 2},
                {"curve_sizes": [8]}, {"arms": {"row_mlp_h32": {}}},
                {"training": {**gp.TRAINING, "max_epochs": 1}}, {"source": {}}):
        with pytest.raises(ValueError):
            gp.certify_fit_spec({**good, **bad})
    # the control arm rides along in mean mode without displacing a registered arm
    gp.certify_fit_spec({**good, "arms": {a: {} for a in (*gp.DEFAULT_ARMS, gp.CONTROL_ARM)}})


def test_continuation_mean_dump_is_rejected_when_the_mean_does_not_match(tmp_path):
    """A `continuation_mean` dump whose primary label is not the mean of its persisted
    continuation labels is unusable: the target would not be the one the experiment declared."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    task = _scale_task(0, depth=3)
    reg = {task.task_id: {"train": (0,), "validation": (1,), "test": (2,)}}
    out = str(tmp_path / "cm")
    gg.run_generalization_dump(
        [task], prod=PRODUCING_BASELINES["stock_aware"], target_mode="continuation_mean",
        continuations=2, delta=1.0, lookahead=1, horizon=8, out_dir=out, registry=reg,
        splits=("train",), strict_splits=False, require_license=False,
        log=lambda *a, **k: None)
    gp.load_generalization_dump(out)  # honest dump loads
    rows = list(csv.DictReader(open(f"{out}/continuation_labels.csv")))
    rows[0]["label"] = repr(float(rows[0]["label"]) + 10.0)
    _rewrite(f"{out}/continuation_labels.csv", rows)
    with pytest.raises(ValueError):
        gp.load_generalization_dump(out)


# --- metrics ------------------------------------------------------------------------------
def test_decision_metrics_are_real_choice_tie_aware_and_match_the_random_floor():
    """Flat decisions are excluded (no preference to learn) but counted; top-1 is tie-aware; the
    random floor is best-minus-mean per decision; a perfect ranking has zero regret and r2 == 1."""
    labels = [np.array([1.0, 3.0, 2.0]), np.array([5.0, 5.0]), np.array([0.0, 4.0])]
    perfect = [np.array([1.0, 3.0, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 4.0])]
    m = gp.decision_metrics(labels, perfect)
    assert m["n_decisions"] == 2 and m["n_flat"] == 1
    assert m["mean_regret"] == 0.0 and m["top1_match"] == 1.0 and m["r2"] == pytest.approx(1.0)
    assert m["random_regret"] == pytest.approx(((3 - 2.0) + (4 - 2.0)) / 2)
    worst = [np.array([1.0, 3.0, 2.0])[::-1] * 0 + np.array([3.0, 1.0, 2.0]), np.array([0.0, 1.0]),
             np.array([4.0, 0.0])]
    mw = gp.decision_metrics(labels, worst)
    assert mw["mean_regret"] > 0 and mw["top1_match"] == 0.0
    assert 0.0 <= mw["winner_overlap"] <= 1.0


# --- fit stage ------------------------------------------------------------------------------
def test_fit_emits_recomputable_artifacts_and_never_touches_test(fit_dir):
    """End-to-end fit at smoke scale: predictions/metrics/curve/selection/checkpoints are written,
    every metric row is recomputable from predictions.csv, and NO test row appears anywhere in the
    fit artifacts."""
    out = fit_dir
    spec = json.load(open(f"{out}/spec.json"))
    preds = list(csv.DictReader(open(f"{out}/predictions.csv")))
    metrics = list(csv.DictReader(open(f"{out}/metrics_by_split_task_seed.csv")))
    assert preds and metrics
    assert {r["split"] for r in preds} == {"train", "validation"}
    assert {r["split"] for r in metrics} == {"train", "validation"}
    assert {r["arm"] for r in preds} == {"row_mlp_h32", "graph_0hop_h32"}
    assert {r["model_seed"] for r in preds} == {"0", "1", "ensemble"}
    assert all(np.isfinite(float(r["pred"])) for r in preds)
    assert json.load(open(f"{out}/architecture_selection.json")) == spec["selection"]
    assert json.load(open(f"{out}/normalization.json"))["graph_0hop_h32"]["node"]

    # recompute one metric row straight from predictions.csv (the artifact-only property)
    row = next(r for r in metrics if r["arm"] == "graph_0hop_h32" and r["model_seed"] == "0"
               and r["split"] == "validation" and r["trajectory_seed"] != "all")
    groups: dict[str, list[tuple[int, float, float]]] = {}
    for p in preds:
        if (p["arm"], p["model_seed"], p["split"], p["task_id"], p["trajectory_seed"]) == \
                (row["arm"], row["model_seed"], row["split"], row["task_id"],
                 row["trajectory_seed"]):
            groups.setdefault(p["decision_index"], []).append(
                (int(p["candidate_index"]), float(p["pred"]), float(p["primary_label"])))
    recomputed = gp.decision_metrics(
        [np.array([c[2] for c in sorted(v)]) for v in groups.values()],
        [np.array([c[1] for c in sorted(v)]) for v in groups.values()])
    assert recomputed["n_decisions"] == int(row["n_decisions"])
    assert recomputed["mean_regret"] == pytest.approx(float(row["mean_regret"]))
    assert recomputed["r2"] == pytest.approx(float(row["r2"]), nan_ok=True)
    # the fair row-baseline regret rides on every row, and the gaps are reported
    assert all(r["row_regret"] != "" for r in metrics)
    assert spec["generalization_gap"]


def test_arm_selection_uses_only_validation_and_prefers_the_smaller_arm_on_a_tie():
    """Selection rule: an arm without positive r2 on >=2 validation tasks is discarded; among the
    rest the lowest mean relative regret wins; a tie within 1% goes to the smaller arm. If nothing
    survives, no arm is selected — Gate A then fails WITHOUT any test read."""
    def rows(arm, r2s, regrets, split="validation"):
        return [{"arm": arm, "split": split, "model_seed": "ensemble", "trajectory_seed": "all",
                 "task_id": f"t{i}", "r2": r2, "mean_regret": reg, "random_regret": 10.0}
                for i, (r2, reg) in enumerate(zip(r2s, regrets))]

    good_small = rows("mpnn_2x32", [0.1, 0.2, -0.1], [5.0, 5.0, 5.0])
    good_big = rows("mpnn_4x64", [0.1, 0.2, 0.3], [4.99, 5.0, 5.0])
    bad = rows("graph_0hop_h32", [-1.0, -1.0, 0.1], [4.0, 4.0, 4.0])
    sel = gp.select_graph_arm(good_small + good_big + bad,
                              arms=("graph_0hop_h32", "mpnn_2x32", "mpnn_4x64"))
    assert sel["selected"] == "mpnn_2x32"  # 0-hop ineligible; <1% tie -> smaller arm
    assert sel["summary"]["graph_0hop_h32"]["eligible"] is False
    clear = gp.select_graph_arm(good_small + rows("mpnn_4x64", [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
                                arms=("mpnn_2x32", "mpnn_4x64"))
    assert clear["selected"] == "mpnn_4x64"  # a real margin beats the smaller-arm tiebreak
    none_ = gp.select_graph_arm(bad, arms=("graph_0hop_h32",))
    assert none_["selected"] is None
    # a test-split row can never influence the choice
    assert gp.select_graph_arm(good_small + good_big + rows(
        "mpnn_2x32", [9.0, 9.0, 9.0], [0.0, 0.0, 0.0], split="test"),
        arms=("mpnn_2x32", "mpnn_4x64"))["summary"]["mpnn_2x32"]["mean_relative_regret"] == \
        pytest.approx(0.5)


# --- Offline Gate A ---------------------------------------------------------------------------
def _gate_rows(*, arm_regret: float, row_regret: float, r2_sign: float, tasks=("a", "b", "c"),
               seeds=(0, 1, 2), arm="mpnn_4x64", traj_seeds=(1,),
               seeds_by_task: "dict | None" = None) -> list[dict]:
    """Synthetic test predictions with a controllable regret/r2 per arm. Each decision has two
    candidates with labels (0, 1); a prediction that ranks them correctly has zero regret."""
    rows = []
    for name, regret in ((arm, arm_regret), ("row_mlp_h32", row_regret)):
        for task in tasks:
            for ms in [*[str(s) for s in seeds], "ensemble"]:
                for ts in (seeds_by_task or {}).get(task, traj_seeds):
                    for d in range(10):
                        wrong = d < int(round(regret * 10))
                        pred = [1.0, 0.0] if wrong else [0.0, 1.0]
                        if name == arm and r2_sign < 0:
                            pred = [p * -3.0 for p in pred]
                        for ci in range(2):
                            rows.append({"arm": name, "model_seed": ms, "split": "test",
                                         "task_id": task, "trajectory_seed": ts,
                                         "decision_index": d, "candidate_index": ci,
                                         "producer_id": f"p{ci}", "pred": pred[ci],
                                         "primary_label": float(ci), "flat": 0})
    return rows


def test_gate_a_passes_only_a_genuinely_better_arm_and_fails_every_shortfall():
    """Gate A is mechanical and adversarial: a clearly better arm passes, and each individual
    shortfall — no r2, no margin over random, no win over the fair row baseline, a single bad
    model seed, a missing task — flips it to fail. `pass` must never be reachable by prose.
    (Synthetic tasks, so the registered-scope requirement is off here; its own test covers it.)"""
    def gate(rows, **kw):
        return gp.gate_a(rows, arm="mpnn_4x64", require_registered=False, **kw)

    good = gate(_gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0))
    assert good["pass"] and all(good["conditions"].values())

    # negative r2 (anti-correlated scores) fails conditions 1 and 5
    bad_r2 = gate(_gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=-1.0))
    assert not bad_r2["pass"] and not bad_r2["conditions"]["r2_pos_2of3"]

    # at the random floor: no 10% reduction, and no win over the row baseline
    at_floor = gate(_gate_rows(arm_regret=0.5, row_regret=0.5, r2_sign=1.0))
    assert not at_floor["pass"]
    assert not at_floor["conditions"]["beats_random_10pct_2of3"]
    assert not at_floor["conditions"]["beats_row_baseline_2of3"]

    # one unstable model seed sinks it even though the ensemble mean looks fine
    rows = _gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0)
    for r in rows:
        if r["arm"] == "mpnn_4x64" and r["model_seed"] == "2":
            r["pred"] = -float(r["pred"])
    unstable = gate(rows)
    assert not unstable["pass"] and not unstable["conditions"]["holds_for_every_model_seed"]

    # a missing task is a coverage failure, not a 2-of-2 pass
    partial = gate(_gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0, tasks=("a", "b")))
    assert not partial["pass"] and not partial["conditions"]["coverage_and_finite"]
    assert not gate([])["pass"]


def test_gate_a_defaults_to_the_registered_scope_instead_of_inferring_it_from_the_rows():
    """Omitting the expected scope used to mean "infer it from the rows on hand", so a prediction
    file holding ONE arbitrary episode per task certified itself as the whole final-test read. The
    default scope is now the registry: the real tasks with their real test episodes pass, a
    narrower set fails, and a caller that hands in its own narrow scope is told so rather than
    believed."""
    registered = {t: tuple(gg.SPLIT_REGISTRY[t]["test"]) for t in gg.TASK_IDS}
    full = _gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0, tasks=gg.TASK_IDS,
                      seeds_by_task=registered)
    assert gp.gate_a(full, arm="mpnn_4x64")["pass"]

    one_each = _gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0, tasks=gg.TASK_IDS,
                          seeds_by_task={t: v[:1] for t, v in registered.items()})
    thin = gp.gate_a(one_each, arm="mpnn_4x64")
    assert not thin["pass"] and not thin["conditions"]["coverage_and_finite"]

    # rows from three tasks that are not the registered ones no longer define their own scope
    synthetic = gp.gate_a(_gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0), arm="mpnn_4x64")
    assert not synthetic["pass"] and not synthetic["conditions"]["coverage_and_finite"]
    assert synthetic["coverage"]["expected_episode_seeds"] == {
        t: sorted(v) for t, v in registered.items()}

    narrow = gp.gate_a(full, arm="mpnn_4x64",
                       seeds_by_task={t: v[:1] for t, v in registered.items()})
    assert not narrow["pass"] and not narrow["conditions"]["registered_scope"]

    two_seeds = gp.gate_a(full, arm="mpnn_4x64", model_seeds=(0, 1))
    assert not two_seeds["pass"] and not two_seeds["conditions"]["registered_scope"]


def test_gate_a_coverage_must_be_the_whole_registered_test_set(tmp_path):
    """A partial final-test read is a DIFFERENT experiment, not a weaker pass. Dropping one
    registered episode, dropping decisions from one model seed, or duplicating a row must each
    fail coverage — the old check only asked for at least one trajectory seed per task, which all
    three of these satisfy."""
    seeds_by_task = {t: (229, 230, 231) for t in ("a", "b", "c")}
    full = _gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0, traj_seeds=(229, 230, 231))

    def gate(rows, **kw):
        kw.setdefault("seeds_by_task", seeds_by_task)
        return gp.gate_a(rows, arm="mpnn_4x64", require_registered=False, **kw)

    good = gate(full)
    assert good["pass"] and good["coverage"]["ok"]

    two_of_three = gate(
        _gate_rows(arm_regret=0.0, row_regret=0.4, r2_sign=1.0, traj_seeds=(229, 230)))
    assert not two_of_three["pass"] and not two_of_three["conditions"]["coverage_and_finite"]

    thin_seed = [r for r in full
                 if not (r["arm"] == "mpnn_4x64" and r["model_seed"] == "1"
                         and r["decision_index"] > 5)]
    partial = gate(thin_seed)
    assert not partial["pass"]
    assert not partial["coverage"]["identical_decision_coverage"]

    duped = gate([*full, dict(full[0])])
    assert not duped["pass"] and duped["coverage"]["duplicate_rows"] == 1

    missing_arm = gate([r for r in full if r["arm"] != "row_mlp_h32"])
    assert not missing_arm["pass"] and missing_arm["coverage"]["missing_groups"]


# --- manifest lock ------------------------------------------------------------------------------
def test_gate_a_stage_requires_a_verifying_manifest(tmp_path):
    """Run-order enforcement: the test read may only score the artifact set that validation froze.
    A manifest that no longer matches the files on disk aborts the gate stage."""
    d = tmp_path / "art"
    d.mkdir()
    (d / "m.pt").write_bytes(b"weights")
    manifest = str(tmp_path / "manifest.json")
    gp.lock_manifest([str(d / "m.pt")], manifest, base_dir=str(d))
    gp.verify_manifest(manifest, base_dir=str(d))  # unchanged -> ok
    (d / "m.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        gp.verify_manifest(manifest, base_dir=str(d))
    (d / "m.pt").unlink()
    with pytest.raises(ValueError):
        gp.verify_manifest(manifest, base_dir=str(d))


def test_the_lock_covers_the_adapter_and_refuses_a_root_it_cannot_check(tmp_path):
    """The adapter is the deployed decision, so it must be INSIDE the lock. Recording its hash in
    a field the verifier never reads left the selected (alpha, margin) editable after the freeze;
    and a locked root with no directory to check it against must abort rather than pass silently."""
    fit = tmp_path / "fit"
    roll = tmp_path / "rollout"
    fit.mkdir()
    roll.mkdir()
    (fit / "m.pt").write_bytes(b"weights")
    adapter = roll / "adapter.json"
    adapter.write_text(json.dumps({"selected": "gnn_alpha0.1_margin0", "selected_alpha": 0.1}))
    manifest = str(roll / "manifest.json")
    doc = gp.lock_manifest([str(fit / "m.pt")], manifest, base_dir=str(fit),
                           roots={"adapter": (str(roll), [str(adapter)])}, extra={"arm": "mpnn"})
    assert set(doc["roots"]) == {"fit", "adapter"}
    assert gp.verify_manifest(manifest, base_dir=str(fit),
                              roots={"adapter": str(roll)})["arm"] == "mpnn"
    with pytest.raises(ValueError):  # an unverifiable locked root is never a pass
        gp.verify_manifest(manifest, base_dir=str(fit))
    adapter.write_text(json.dumps({"selected": "gnn_alpha0.5_margin0", "selected_alpha": 0.5}))
    with pytest.raises(ValueError):  # swapping the frozen adapter is detected
        gp.verify_manifest(manifest, base_dir=str(fit), roots={"adapter": str(roll)})


def test_gate_a_takes_the_arm_from_the_locked_selection(tmp_path, dump_dir, fit_dir):
    """The scored arm is the one decision the whole test read is about, so it comes from the
    LOCKED `architecture_selection.json`, not from the mutable fit `spec.json`. An `--arm` that
    disagrees with the frozen selection is refused instead of quietly overriding it."""
    import glob
    import shutil

    fit_dir = str(shutil.copytree(fit_dir, tmp_path / "fit_copy"))
    manifest = str(tmp_path / "manifest.json")
    gp.lock_manifest([*glob.glob(f"{fit_dir}/checkpoints/*.pt"), f"{fit_dir}/normalization.json",
                      f"{fit_dir}/architecture_selection.json"], manifest, base_dir=fit_dir)
    selection = json.load(open(f"{fit_dir}/architecture_selection.json"))
    if selection["selected"] is None:  # nothing was eligible at smoke scale: no arm to bind to
        selection["selected"] = "graph_0hop_h32"
        with open(f"{fit_dir}/architecture_selection.json", "w") as f:
            json.dump(selection, f)
        with pytest.raises(ValueError):  # ... and editing it after the lock is caught
            gp.run_gate_a(dump_dir, fit_dir, str(tmp_path / "g"), manifest_path=manifest,
                          model_seeds=(0, 1), require_shape=False, log=lambda *a, **k: None)
        return
    assert gp.locked_arm(fit_dir, {"arm": selection["selected"]}) == selection["selected"]
    with pytest.raises(ValueError):
        gp.locked_arm(fit_dir, {"arm": "mpnn_4x64"})
    with pytest.raises(ValueError):
        gp.run_gate_a(dump_dir, fit_dir, str(tmp_path / "gate"), manifest_path=manifest,
                      arm="mpnn_2x32", model_seeds=(0, 1), require_shape=False,
                      log=lambda *a, **k: None)


def test_gate_a_stage_runs_end_to_end_from_the_frozen_fit(tmp_path, dump_dir, fit_dir):
    """The full immutable order at smoke scale: fit (train+validation) -> lock -> gate-a. The
    stage reloads the frozen checkpoints and persisted normalization (never a refit), scores the
    untouched test episodes once, and writes an artifact-recomputable verdict."""
    import glob
    import shutil

    # the last step tampers with an artifact, so work on a copy and leave the fixture clean
    fit_dir = str(shutil.copytree(fit_dir, tmp_path / "fit_copy"))

    manifest = str(tmp_path / "manifest.json")
    gp.lock_manifest([*glob.glob(f"{fit_dir}/checkpoints/*.pt"), f"{fit_dir}/normalization.json"],
                     manifest, base_dir=fit_dir)
    gate_dir = str(tmp_path / "gate")
    verdict = gp.run_gate_a(dump_dir, fit_dir, gate_dir, manifest_path=manifest,
                            arm="graph_0hop_h32", model_seeds=(0, 1), require_shape=False,
                            log=lambda *a, **k: None)
    rows = list(csv.DictReader(open(f"{gate_dir}/test_predictions.csv")))
    assert rows and {r["split"] for r in rows} == {"test"}
    assert {r["arm"] for r in rows} == {"graph_0hop_h32", "row_mlp_h32"}
    on_disk = json.load(open(f"{gate_dir}/gate_a.json"))["verdict"]
    # (per-model-seed keys become strings through JSON; compare the decision, not the encoding)
    assert (verdict["pass"], verdict["conditions"]) == (on_disk["pass"], on_disk["conditions"])
    # the verdict is exactly what the artifact rows recompute to
    assert verdict["pass"] == gp.gate_a(
        [{**r, "trajectory_seed": int(r["trajectory_seed"]),
          "decision_index": int(r["decision_index"]),
          "candidate_index": int(r["candidate_index"]), "pred": float(r["pred"]),
          "primary_label": float(r["primary_label"])} for r in rows],
        arm="graph_0hop_h32", tasks=sorted({r["task_id"] for r in rows}),
        model_seeds=(0, 1), require_registered=False)["pass"]
    # a tampered checkpoint after the lock cannot be scored
    open(f"{fit_dir}/normalization.json", "a").write(" ")
    with pytest.raises(ValueError):
        gp.run_gate_a(dump_dir, fit_dir, gate_dir, manifest_path=manifest, arm="graph_0hop_h32",
                      model_seeds=(0, 1), require_shape=False, log=lambda *a, **k: None)


# --- fit-stage compute budget --------------------------------------------------------------
def test_nonflat_fraction_counts_the_population_the_fit_actually_trains_on(dump_dir):
    """Training drops primary-target-flat decisions, so the fit budget must be projected from the
    NON-flat population. On the real EXP-0039 dump only ~31% of decisions survive, so budgeting on
    the collected count would overstate the fit 3x."""
    from tokenomics.eval import graph_generalization_budget as gb

    flat = gb.nonflat_fraction(dump_dir)
    assert 0 < flat["nonflat"] <= flat["total"]
    assert flat["nonflat_fraction"] == pytest.approx(flat["nonflat"] / flat["total"])
    # the per-task breakdown must reconstruct the totals, and carry a per-episode rate to scale by
    assert sum(v["total"] for v in flat["per_task"].values()) == flat["total"]
    assert sum(v["nonflat"] for v in flat["per_task"].values()) == flat["nonflat"]
    for v in flat["per_task"].values():
        assert v["episodes"] >= 1
        assert v["nonflat_per_episode"] == pytest.approx(v["nonflat"] / v["episodes"])

    # cross-check the re-derived flatness against the probe's own loader on the same dump
    loaded = gp.load_generalization_dump(dump_dir)
    assert sum(1 for d in loaded.decisions if not d.flat) == flat["nonflat"]


def test_fit_projection_scales_with_runs_epochs_and_data_fraction():
    """The projection is the fit-stage number the handover never had. It must count the per-task
    arm as one model PER TASK (27 optimizer runs, not 21), charge the learning curve at its
    nested data fractions, and add the control arm only in mean mode."""
    from tokenomics.eval import graph_generalization_budget as gb

    per_batch = {"mpnn_4x64": 1.0, "mpnn_2x32": 0.5, "graph_0hop_h32": 0.1, "row_mlp_h32": 0.01}
    p = gb.project_fit(per_batch, train_decisions=9600, batch_decisions=96, epochs=(50, 300))
    assert p["batches_per_epoch"] == 100
    # protocol v2 mandatory ladder: 4 arms x 3 seeds + 6 curve fits = 18; the control adds 3 in
    # mean mode; the opt-in per-task arm adds 9 (3 seeds x 3 task-models) only when asked for
    assert p["n_optimizer_runs"] == 18
    assert gb.project_fit(per_batch, train_decisions=9600, mean_mode=True)["n_optimizer_runs"] == 21
    assert gb.project_fit(per_batch, train_decisions=9600,
                          include_optional=True)["n_optimizer_runs"] == 27
    assert "mpnn_4x64_per_task" not in p["seconds_per_epoch_by_arm"]
    assert "mpnn_4x64_curve" in p["seconds_per_epoch_by_arm"]      # the curve is never optional
    # curve = 3 seeds at 1/4 data + 3 at 1/2 data = 2.25 pooled-equivalents
    assert p["seconds_per_epoch_by_arm"]["mpnn_4x64_curve"] == pytest.approx(2.25 * 100)
    assert p["core_hours"]["epochs_300"] == pytest.approx(6 * p["core_hours"]["epochs_50"])
    assert p["core_hours_per_epoch"] == pytest.approx(p["core_hours"]["epochs_50"] / 50)
    assert 0 < p["dominant_arm_share"] < 1
    # an arm with no measured cost is an error, not a silently-zero line item
    with pytest.raises(ValueError):
        gb.project_fit({"mpnn_4x64": 1.0}, train_decisions=9600)


def test_epoch_weights_preserve_loss_scale_on_full_batches_and_only_shrink_the_tail():
    """The frozen learning rate and clip norm are calibrated to the OLD loss magnitude, so the
    rebalance must not silently rescale the objective. It does not, but not exactly: with
    equal-sized tasks the weights are exactly the old uniform 1/T, and on the registered
    (unequal) profile a batch containing every task carries 1.023x the old loss — the weights sum
    to >=1 by AM-HM, with the excess set by the SPREAD of the per-task batch counts. The real
    change is confined to the tail batches, where the largest task used to be alone at full
    weight: it now carries 0.27, i.e. 3.7x less. That is the correction, not a side effect.

    Guarding the scale matters because a uniform inflation of the loss is indistinguishable from
    a raised learning rate, and would silently invalidate the pre-registered recipe."""
    per_task = 32
    equal = _decisions(n_tasks=3, per_task=4 * per_task)
    w_eq = gp.epoch_task_weights(equal, list(range(len(equal))), per_task=per_task)
    assert sum(w_eq.values()) == pytest.approx(1.0)
    assert sorted(w_eq.values()) == pytest.approx([1 / 3] * 3)

    # the registered non-flat train-8 profile: 3243 / 3509 / 4622 decisions -> 102 / 110 / 145 chunks
    skewed, index = [], 0
    for task, n in (("s29", 3243), ("s45", 3509), ("s77", 4622)):
        for j in range(n):
            skewed.append(gp.Decision(
                index=index, task_id=task, split="train", seed=0, cycle=j, order_id=f"o{j}",
                flat=False, producer_ids=["p0", "p1"], labels=np.zeros(2),
                rows=np.zeros((2, 7), dtype=np.float32)))
            index += 1
    w = gp.epoch_task_weights(skewed, list(range(len(skewed))), per_task=per_task)
    assert sum(w.values()) == pytest.approx(1.023, abs=5e-3)   # full-batch scale, within ~2%
    assert sum(w.values()) >= 1.0                              # AM-HM: never scales the loss DOWN
    assert w["s77"] == pytest.approx(0.274, abs=5e-3)          # was 1.0 in a tail batch
    assert 1 / w["s77"] == pytest.approx(3.66, abs=0.05)


# --- protocol v2: mandatory ladder + the curve as a diagnostic ------------------------------
def test_mandatory_ladder_keeps_the_curve_and_drops_the_per_task_arm():
    """Protocol v2. `mpnn_4x64_per_task` attributes a pooling effect and feeds no verdict, so it
    is an opt-in post-mortem; the 2/4/8 curve is REQUIRED, because it is what separates 'more
    independent trajectories help' from 'trajectory coverage is not the missing ingredient'."""
    assert gp.MANDATORY_ARMS == ("row_mlp_h32", "graph_0hop_h32", "mpnn_2x32", "mpnn_4x64")
    assert gp.DEFAULT_ARMS == gp.MANDATORY_ARMS
    assert "mpnn_4x64_per_task" in gp.OPTIONAL_DIAGNOSTIC_ARMS
    assert "mpnn_4x64_per_task" not in gp.FIT_CONTRACT["arms"]
    assert gp.FIT_CONTRACT["curve_required"] is True
    assert gp.FIT_CONTRACT["curve_arm"] == "mpnn_4x64" == gp.CURVE_ARM
    assert gp.FIT_CONTRACT["curve_is_gate"] is False
    assert tuple(gp.FIT_CONTRACT["curve_sizes"]) == (2, 4, 8)
    # the contract still refuses a run that swaps the ladder
    with pytest.raises(ValueError):
        gp.certify_fit_contract(arms=("mpnn_4x64",), model_seeds=(0, 1, 2), train_size=8,
                                curve_sizes=(2, 4, 8), training=None, source_commit="a" * 40)


def test_learning_curve_reports_regret_random_r2_and_dispersion_per_task_and_seed(curve_fit_dir):
    """The curve must be readable per (size, model seed, task) without recomputation, and carry
    `regret_over_random` explicitly — the tasks have different random floors (EXP-0039: 5.24 /
    5.29 / 6.23), so a curve read on raw regret can invert. Dispersion rides along so an
    improvement produced by seeds converging is distinguishable from one produced by every seed
    getting better."""
    with open(f"{curve_fit_dir}/learning_curve.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows, "curve is mandatory: the fit must emit learning_curve.csv"
    for col in ("arm", "n_train_episodes", "model_seed", "task_id", "mean_regret", "r2",
                "random_regret", "regret_over_random", "dispersion"):
        assert col in rows[0], f"learning_curve.csv is missing {col}"
    for r in rows:
        assert r["arm"] == gp.CURVE_ARM          # pooled mpnn_4x64 only
        if r["random_regret"] and r["regret_over_random"]:
            assert float(r["regret_over_random"]) == pytest.approx(
                float(r["mean_regret"]) / float(r["random_regret"]), rel=1e-6)
    # every (size, seed) pair covers the same task set
    by_key = {}
    for r in rows:
        by_key.setdefault((r["n_train_episodes"], r["model_seed"]), set()).add(r["task_id"])
    assert len(set(map(frozenset, by_key.values()))) == 1


def test_curve_is_diagnostic_only_and_cannot_move_arm_selection():
    """A flat curve must not be able to cancel a test that train-8 earned, and a rising curve must
    not rescue a train-8 that failed: selection reads the arm metric rows and nothing else."""
    import inspect

    metric_rows = [
        {"arm": a, "split": "validation", "task_id": t, "trajectory_seed": "all",
         "model_seed": "ensemble", "mean_regret": r, "r2": 0.1, "random_regret": 6.0,
         "row_regret": 5.0, "dispersion": 0.1}
        for a, r in (("mpnn_4x64", 4.0), ("mpnn_2x32", 4.5), ("graph_0hop_h32", 5.5))
        for t in ("t0", "t1")
    ]
    assert gp.select_graph_arm(metric_rows)["selected"] == "mpnn_4x64"
    assert set(inspect.signature(gp.select_graph_arm).parameters) == {"metric_rows", "arms"}
    assert gp.FIT_CONTRACT["curve_is_gate"] is False

    # A curve row is a structurally different record: it carries no split / trajectory_seed and
    # its model_seed is never "ensemble", so it cannot pass the selection filter even if the two
    # row streams were ever concatenated by mistake.
    curve = gp._curve_rows("graph_0hop_h32", 8, 0,
                           {"by_task": {"t0": {"mean_regret": 0.01, "r2": 0.99,
                                               "random_regret": 6.0}}})
    assert curve and "split" not in curve[0] and "trajectory_seed" not in curve[0]
    assert str(curve[0]["model_seed"]) != "ensemble"
