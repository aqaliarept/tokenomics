"""EXP-0041 generalization dump: split registry, target modes, and artifact alignment.

These lock the two things EXP-0041's verdict rests on before any compute is spent: that no
episode can leak between train/validation/test, and that the label written as `primary_label`
is exactly what the declared target mode says it is (identical to EXP-0039 under
`single_suffix`; the recomputable mean of the persisted continuation rows under
`continuation_mean`).
"""

from __future__ import annotations

import csv
import glob
import json

import numpy as np
import pytest

from tokenomics.eval import graph_generalization_dump as gg
from tokenomics.eval import graph_value_router_dump as gd

from test_graph_view import _scale_task


def _rows(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def decs_task(out: str) -> str:
    return _rows(f"{out}/decisions.csv")[0]["task_id"]


def _tiny_registry(task_id: str) -> dict:
    """A 3-episode stand-in for the real 8/3/3 table (smoke scale): one episode per split, so
    every split-crossing check still has something to check."""
    return {task_id: {"train": (0, 1), "validation": (2,), "test": (3,)}}


def _tiny_dump(tmp_path, *, target_mode: str, continuations: int = 0, splits=("train",),
               seeds=(0,), horizon: int = 8, lookahead: int = 1) -> tuple[str, dict]:
    """Label collection is the expensive part of a dump (a short rollout per candidate per
    decision), so the smoke knobs are pinned tiny and the results are shared between tests via
    module-scoped fixtures rather than rebuilt per test."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    task = _scale_task(0, depth=3)
    reg = {task.task_id: {"train": tuple(seeds), "validation": (98,), "test": (99,)}}
    out = str(tmp_path / f"dump_{target_mode}_{'_'.join(splits)}")
    spec = gg.run_generalization_dump(
        [task], prod=PRODUCING_BASELINES["stock_aware"], target_mode=target_mode,
        continuations=continuations, delta=1.0, lookahead=lookahead, horizon=horizon,
        out_dir=out, registry=reg, splits=splits, strict_splits=False,
        require_license=False, log=lambda *a, **k: None)
    return out, spec


@pytest.fixture(scope="module")
def single_dump(tmp_path_factory) -> tuple[str, dict]:
    return _tiny_dump(tmp_path_factory.mktemp("single"), target_mode="single_suffix")


@pytest.fixture(scope="module")
def mean_dump(tmp_path_factory) -> tuple[str, dict]:
    """`lookahead=2` is deliberate and load-bearing: at `lookahead=1` the teacher rolls a single
    cycle, so an order placed by the resampled future demand cannot complete inside the window and
    every continuation returns the identical label — the fork looks inert and the mean-mode test
    would be vacuous. The real run uses `lookahead=8`."""
    return _tiny_dump(tmp_path_factory.mktemp("mean"), target_mode="continuation_mean",
                      continuations=2, horizon=10, lookahead=2)


@pytest.fixture(scope="module")
def two_split_dump(tmp_path_factory) -> tuple[str, dict]:
    return _tiny_dump(tmp_path_factory.mktemp("splits"), target_mode="single_suffix",
                      splits=("train", "validation"), seeds=(0, 1))


# --- split registry -------------------------------------------------------------------
def test_registered_splits_are_disjoint_nested_and_never_reuse_a_spent_seed():
    """The pre-registered table passes every mechanical check: three tasks, 8/3/3 whole-episode
    splits, pairwise disjoint within a task, nested train prefixes for the 2->4->8 curve, and
    zero overlap with the EXP-0037..0040 trajectory seeds the earlier reads were tuned on."""
    gg.validate_split_registry()
    from tokenomics.train.graph_router_probe import GATE_TASK_IDS

    assert set(gg.TASK_IDS) == set(GATE_TASK_IDS)  # same frozen topologies as EXP-0039
    assert gd.schema_hash() == "9918b12f69760bcd"  # frozen observation contract
    for task, splits in gg.SPLIT_REGISTRY.items():
        allseeds = [s for name in gg.SPLITS for s in splits[name]]
        assert len(set(allseeds)) == len(allseeds) == 14
        assert not set(allseeds) & gg.SPENT_SEEDS[task]
        for size in gg.NESTED_TRAIN_SIZES:
            assert gg.nested_train_seeds(task, size) == tuple(splits["train"][:size])
            assert not set(gg.nested_train_seeds(task, size)) & set(splits["validation"])
            assert not set(gg.nested_train_seeds(task, size)) & set(splits["test"])


def test_validator_rejects_leakage_short_splits_and_spent_seeds():
    """Adversarial: each way a split table could contaminate the read must RAISE. Split leakage
    is undetectable downstream — a contaminated validation set just reports a better number."""
    import copy

    def mutated(**changes) -> dict:
        reg = copy.deepcopy({t: {k: tuple(v) for k, v in s.items()}
                             for t, s in gg.SPLIT_REGISTRY.items()})
        reg[gg.TASK_S29].update(changes)
        return reg

    with pytest.raises(ValueError):  # a test episode also used for training
        gg.validate_split_registry(mutated(test=(229, 230, 32)))
    with pytest.raises(ValueError):  # a validation episode duplicated into test
        gg.validate_split_registry(mutated(validation=(129, 130, 229)))
    with pytest.raises(ValueError):  # reuses an EXP-0039 anchor seed
        gg.validate_split_registry(mutated(train=(29, 33, 34, 35, 36, 37, 38, 39)))
    with pytest.raises(ValueError):  # too few train episodes for the 8-episode primary fit
        gg.validate_split_registry(mutated(train=(32, 33)))
    with pytest.raises(ValueError):  # duplicate seed inside one split
        gg.validate_split_registry(mutated(validation=(129, 129, 130)))
    with pytest.raises(ValueError):  # a missing task
        gg.validate_split_registry({gg.TASK_S29: gg.SPLIT_REGISTRY[gg.TASK_S29]})


def test_split_of_and_episode_plan_cover_exactly_the_registry():
    """`split_of` is total over the registry and raises for a stray episode (an unregistered
    seed must never be silently folded into a split); the plan visits every registered episode
    exactly once in deterministic order."""
    reg = _tiny_registry("t")
    assert gg.split_of("t", 0, registry=reg) == "train"
    assert gg.split_of("t", 2, registry=reg) == "validation"
    assert gg.split_of("t", 3, registry=reg) == "test"
    with pytest.raises(KeyError):
        gg.split_of("t", 7, registry=reg)
    plan = gg.episode_plan(registry=reg)
    assert plan == [("t", "train", 0), ("t", "train", 1), ("t", "validation", 2), ("t", "test", 3)]
    assert gg.episode_plan(registry=reg, splits=("test",)) == [("t", "test", 3)]


# --- target modes ---------------------------------------------------------------------
def test_target_mode_and_continuation_count_are_required_and_consistent(tmp_path):
    """No default target: EXP-0041 may not choose a mode before EXP-0040 licenses one, and a
    mis-set K (paying for forks that are unused, or averaging zero futures) must fail loudly."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    task = _scale_task(0, depth=3)
    reg = {task.task_id: {"train": (0,), "validation": (98,), "test": (99,)}}
    common = dict(prod=PRODUCING_BASELINES["stock_aware"], registry=reg, splits=("train",),
                  strict_splits=False, horizon=8, lookahead=1, require_license=False,
                  log=lambda *a, **k: None)
    for mode, k in (("guess", 0), ("single_suffix", 2), ("continuation_mean", 0)):
        with pytest.raises(ValueError):
            gg.run_generalization_dump([task], target_mode=mode, continuations=k,
                                       out_dir=str(tmp_path / f"x_{mode}_{k}"), **common)
    with pytest.raises(ValueError):
        gg.primary_label_vector("continuation_mean", np.zeros(3), np.zeros((0, 3)))


def test_single_suffix_labels_are_identical_to_the_exp0039_dump(tmp_path, single_dump):
    """Parity: on the same task and episode, `single_suffix` reproduces the EXP-0039 dump's
    decision population, order, producer alignment, graph payloads, and per-candidate teacher
    labels EXACTLY. Without this, an EXP-0041 number could differ from EXP-0039 because the
    collector changed rather than because the split/pooling changed."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    task = _scale_task(0, depth=3)
    old = str(tmp_path / "old")
    gd.run_graph_dump([task], prod=PRODUCING_BASELINES["stock_aware"], delta=1.0, lookahead=1,
                      eval_seeds=1, horizon=8, out_dir=old, log=lambda *a, **k: None)
    new, spec = single_dump

    old_c, new_c = _rows(f"{old}/candidates.csv"), _rows(f"{new}/candidates.csv")
    assert len(old_c) == len(new_c) > 0
    assert spec["n_candidates"] == len(new_c)
    for a, b in zip(old_c, new_c):
        assert (a["task_id"], a["seed"], a["cycle"], a["order_id"], a["candidate_index"],
                a["producer_id"]) == (b["task_id"], b["trajectory_seed"], b["cycle"],
                                      b["order_id"], b["candidate_index"], b["producer_id"])
        assert float(a["teacher_label"]) == float(b["primary_label"]) == \
            float(b["original_suffix_label"])
    old_g = [g for s in sorted(glob.glob(f"{old}/graph_shard_*.npz")) for g in gd.load_shard(s)]
    new_g = [g for s in sorted(glob.glob(f"{new}/graph_shard_*.npz")) for g in gd.load_shard(s)]
    assert len(old_g) == len(new_g) == spec["n_decisions"]
    for a, b in zip(old_g, new_g):
        assert a.decision_digest == b.decision_digest
        for t in gd.NODE_TYPES:
            assert np.array_equal(a.node_features[t], b.node_features[t])
        for r in gd.RELATION_TYPES:
            assert np.array_equal(a.edge_index[r], b.edge_index[r])
            assert np.array_equal(a.edge_attr[r], b.edge_attr[r])


def test_continuation_mean_is_recomputable_from_the_persisted_rows(mean_dump):
    """In mean mode the primary label must be exactly the mean of the K persisted continuation
    labels for that candidate (so the target is auditable without re-running the simulator), the
    original single-suffix label rides along as the diagnostic control, and different
    continuations really do produce different labels somewhere (otherwise the fork is inert and
    the mean is a disguised single-suffix label)."""
    out, spec = mean_dump
    cands = _rows(f"{out}/candidates.csv")
    conts = _rows(f"{out}/continuation_labels.csv")
    assert cands and len(conts) == 2 * len(cands) == spec["n_continuation_rows"]

    by_key: dict[tuple, list[float]] = {}
    for r in conts:
        by_key.setdefault((r["decision_index"], r["candidate_index"]), []).append(float(r["label"]))
    spread = 0
    for c in cands:
        labs = by_key[(c["decision_index"], c["candidate_index"])]
        assert len(labs) == 2
        assert float(c["primary_label"]) == pytest.approx(float(np.mean(labs)), abs=1e-6)
        assert c["target_mode"] == "continuation_mean" and c["continuation_count"] == "2"
        spread += int(max(labs) != min(labs))
    assert spread > 0, "no candidate's label moved across continuations — the fork did nothing"
    # the original realized-future label is persisted as a control and is NOT the primary target
    assert any(float(c["original_suffix_label"]) != float(c["primary_label"]) for c in cands)


def test_flatness_is_decided_by_the_primary_target(mean_dump):
    """A flat decision (no candidate preferred) is a property of the TARGET, never of a
    prediction, and in mean mode it must be read off the mean — not the original suffix. The
    dump's `primary_flat` flag equals flatness recomputed from candidates.csv."""
    assert gg.is_flat(np.array([1.0, 1.0, 5.0]), [0, 1])
    assert not gg.is_flat(np.array([1.0, 2.0]), [0, 1])
    assert gg.is_flat(np.array([3.0]), [0])  # a single candidate has nothing to prefer

    out, _ = mean_dump
    cands = _rows(f"{out}/candidates.csv")
    by_dec: dict[str, list[float]] = {}
    for c in cands:
        by_dec.setdefault(c["decision_index"], []).append(float(c["primary_label"]))
    for d in _rows(f"{out}/decisions.csv"):
        labs = by_dec[d["decision_index"]]
        assert int(d["primary_flat"]) == int(max(labs) - min(labs) <= gg.FLAT_TOL)


# --- EXP-0040 target license ---------------------------------------------------------
def test_only_exp0040_can_license_a_target_and_two_of_its_verdicts_forbid_the_run():
    """The target is not a CLI preference. Each EXP-0040 classification maps to exactly one
    (mode, K), `AMBIGUOUS` blocks the experiment, `SEARCH_ONLY` cancels it, and a mean target
    needs the CONVERGED K — the smallest of {4,8,16} EXP-0040 actually validated, never an
    arbitrary number that would silently change the label budget."""
    assert gg.target_license("COVERAGE_PATH_LICENSED").target_mode == "single_suffix"
    assert gg.target_license("COVERAGE_PATH_LICENSED").continuations == 0
    with pytest.raises(gg.Exp0040Blocked):
        gg.target_license("AMBIGUOUS")
    with pytest.raises(gg.Exp0040Cancelled):
        gg.target_license("SEARCH_ONLY")
    with pytest.raises(ValueError):
        gg.target_license("NOT_A_VERDICT")
    with pytest.raises(ValueError):  # coverage path may not average
        gg.target_license("COVERAGE_PATH_LICENSED", continuations=4)

    mean = gg.target_license("MEAN_TARGET_LICENSED",
                             convergence_by_k={"4": False, "8": True, "16": True})
    assert (mean.target_mode, mean.continuations) == ("continuation_mean", 8)
    with pytest.raises(ValueError):  # K must be one of the licensed values
        gg.target_license("MEAN_TARGET_LICENSED", continuations=2)
    with pytest.raises(ValueError):  # and the SMALLEST converged one, not a bigger licensed one
        gg.target_license("MEAN_TARGET_LICENSED", continuations=16,
                          convergence_by_k={"4": False, "8": True, "16": True})
    with pytest.raises(ValueError):  # no convergence evidence at all
        gg.target_license("MEAN_TARGET_LICENSED")


def test_license_is_read_from_the_exp0040_artifact_and_a_dump_cannot_run_without_one(tmp_path):
    """A real dump must carry the license that authorized it: no license at all is a hard stop,
    and a license that disagrees with the requested target cannot be talked past."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    path = tmp_path / "exp0040.json"
    path.write_text(json.dumps({"experiment": "EXP-0040",
                                "verdict": {"classification": "COVERAGE_PATH_LICENSED"}}))
    lic = gg.read_exp0040_license(str(path))
    assert (lic.target_mode, lic.continuations) == ("single_suffix", 0)
    assert len(lic.source_sha256 or "") == 64
    (tmp_path / "empty.json").write_text("{}")
    with pytest.raises(ValueError):
        gg.read_exp0040_license(str(tmp_path / "empty.json"))

    task = _scale_task(0, depth=3)
    reg = {task.task_id: {"train": (0,), "validation": (98,), "test": (99,)}}
    common = dict(prod=PRODUCING_BASELINES["stock_aware"], registry=reg, splits=("train",),
                  strict_splits=False, horizon=8, lookahead=1, log=lambda *a, **k: None)
    with pytest.raises(ValueError):  # no license: label compute is refused before it is spent
        gg.run_generalization_dump([task], target_mode="single_suffix",
                                   out_dir=str(tmp_path / "unlicensed"), **common)
    mean_license = gg.target_license("MEAN_TARGET_LICENSED", continuations=4)
    with pytest.raises(ValueError):  # licensed target != requested target
        gg.run_generalization_dump([task], target_mode="single_suffix", license=mean_license,
                                   out_dir=str(tmp_path / "mismatch"), **common)


def test_the_licensing_artifact_must_identify_itself_and_its_k_must_follow_from_evidence(tmp_path):
    """The license is EXP-0040's to give, and K comes with it. A JSON file that merely contains
    the winning string is not EXP-0040's result; there is deliberately no way to hand a K in on
    the side (that would let an artifact converged at 8 buy a 16-continuation label run); and a
    document whose declared K contradicts its own convergence map is a contradiction, not an
    override."""
    import inspect

    assert "continuations" not in inspect.signature(gg.read_exp0040_license).parameters

    unstamped = tmp_path / "anon.json"
    unstamped.write_text(json.dumps({"classification": "MEAN_TARGET_LICENSED",
                                     "convergence_by_k": {"4": True}}))
    with pytest.raises(ValueError):
        gg.read_exp0040_license(str(unstamped))

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"experiment": "EXP-0040",
                                "verdict": {"classification": "MEAN_TARGET_LICENSED",
                                            "converged_continuations": 16}}))
    with pytest.raises(ValueError):  # a claimed K with no convergence evidence behind it
        gg.read_exp0040_license(str(bare))

    contradiction = tmp_path / "contradiction.json"
    contradiction.write_text(json.dumps({
        "experiment": "EXP-0040",
        "verdict": {"classification": "MEAN_TARGET_LICENSED", "converged_continuations": 16,
                    "convergence_by_k": {"4": False, "8": True, "16": True}}}))
    with pytest.raises(ValueError):
        gg.read_exp0040_license(str(contradiction))

    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "experiment": "EXP-0040",
        "verdict": {"classification": "MEAN_TARGET_LICENSED",
                    "convergence_by_k": {"4": False, "8": True, "16": True}}}))
    lic = gg.read_exp0040_license(str(good))
    assert (lic.target_mode, lic.continuations) == ("continuation_mean", 8)
    assert lic.evidence == {"4": False, "8": True, "16": True}


def test_certify_rejects_a_dump_that_is_not_the_pre_registered_dataset(single_dump):
    """`certify_dump_contract` is what stands between a pulled artifact directory and a published
    verdict. Every field the verdict silently depends on must make it raise when wrong: the task
    set, the collected episode seeds, the frozen teacher config, the pinned inputs, the source
    commit, the license, and the declared row counts."""
    import copy

    out, _ = single_dump
    good = _registered_spec(json.load(open(f"{out}/spec.json")))
    gg.certify_dump_contract(good, n_decisions=good["n_decisions"],
                             n_candidates=good["n_candidates"])

    def broken(**changes) -> dict:
        spec = copy.deepcopy(good)
        spec.update(changes)
        return spec

    with pytest.raises(ValueError):
        gg.certify_dump_contract(broken(lookahead=1))          # not the frozen teacher window
    with pytest.raises(ValueError):
        gg.certify_dump_contract(broken(task_ids=list(gg.TASK_IDS[:2]) + ["other"]))
    with pytest.raises(ValueError):
        gg.certify_dump_contract(broken(splits=["train", "validation"]))
    with pytest.raises(ValueError):
        gg.certify_dump_contract(broken(inputs={"archive": {"sha256": "0" * 64}}))
    with pytest.raises(ValueError):
        gg.certify_dump_contract(broken(source={"tree_sha256": "a" * 64, "commit": "short"}))
    with pytest.raises(ValueError):
        gg.certify_dump_contract(broken(license=None))
    with pytest.raises(ValueError):  # license says mean, dump says single suffix
        gg.certify_dump_contract(broken(license={"classification": "MEAN_TARGET_LICENSED",
                                                 "continuations": 4, "source": "EXP-0040.json",
                                                 "source_sha256": "c" * 64,
                                                 "evidence": {"4": True}}))
    with pytest.raises(ValueError):  # an inline classification is a claim, not a license
        gg.certify_dump_contract(broken(license={**good["license"], "source": "inline",
                                                 "source_sha256": None}))
    with pytest.raises(ValueError):  # K must FOLLOW from the recorded convergence evidence
        gg.certify_dump_contract(broken(
            target_mode="continuation_mean", continuation_count=16,
            license={"classification": "MEAN_TARGET_LICENSED", "continuations": 16,
                     "source": "EXP-0040.json", "source_sha256": "c" * 64,
                     "evidence": {"4": False, "8": True, "16": True}}))
    with pytest.raises(ValueError):
        gg.certify_dump_contract(good, n_decisions=good["n_decisions"] + 1)
    # a single missing final-test episode is a different experiment, not a smaller one
    thin = copy.deepcopy(good)
    thin["split_manifest"]["collected"] = [c for c in thin["split_manifest"]["collected"]
                                           if c["trajectory_seed"] != gg.SPLIT_REGISTRY[
                                               gg.TASK_S29]["test"][0]]
    with pytest.raises(ValueError):
        gg.certify_dump_contract(thin)


def _registered_spec(spec: dict) -> dict:
    """Promote a smoke dump's spec to what the pre-registered run would emit, so the certifier can
    be exercised without paying for the real 42-episode dump."""
    from tokenomics.train.graph_router_probe import GATE_INPUT_SHA256

    collected = [{"task_id": t, "split": name, "trajectory_seed": s, "n_decisions": 1,
                  "n_flat_primary": 0, "n_candidates": 2}
                 for t in gg.SPLIT_REGISTRY for name in gg.SPLITS
                 for s in gg.SPLIT_REGISTRY[t][name]]
    return {**spec, "task_ids": list(gg.TASK_IDS), "splits": list(gg.SPLITS),
            "delta": 1.0, "lookahead": 8, "horizon": 100, "floor": "blend_0.5",
            "inputs": {"archive": {"sha256": GATE_INPUT_SHA256["archive"]},
                       "producer_ckpt": {"sha256": GATE_INPUT_SHA256["producer_ckpt"]}},
            "source": {"tree_sha256": "a" * 64, "commit": "b" * 40},
            "license": {"classification": "COVERAGE_PATH_LICENSED", "continuations": 0,
                        "source": "EXP-0040/RESULT.json", "source_sha256": "c" * 64,
                        "evidence": None},
            "split_manifest": {
                "registry": {t: {n: list(gg.SPLIT_REGISTRY[t][n]) for n in gg.SPLITS}
                             for t in gg.SPLIT_REGISTRY},
                "collected": collected},
            "n_decisions": len(collected), "n_candidates": 2 * len(collected)}


# --- artifacts ---------------------------------------------------------------------------
def test_candidate_rows_carry_deployable_row_features_and_align_to_the_graphs(single_dump):
    """The fair `row_mlp_h32` baseline must retrain on the SAME splits and target, so the 7
    deployable actor features ride with every candidate row. Alignment: each graph's candidate
    producers decode (via topologies.jsonl) to that decision's candidates.csv producer ids in
    order, and the graph payload is digest-bound to its decisions.csv row."""
    out, spec = single_dump
    cands = _rows(f"{out}/candidates.csv")
    decs = _rows(f"{out}/decisions.csv")
    assert all(f"feat_{i}" in cands[0] for i in range(gg.ROW_FEATURES))
    # feature 0 is the eligibility bit: every persisted candidate row is an eligible producer
    assert all(float(c["feat_0"]) == 1.0 for c in cands)
    assert all(np.isfinite(float(c[f"feat_{i}"])) for c in cands for i in range(gg.ROW_FEATURES))

    topo = {json.loads(l)["task_id"]: json.loads(l)["producer_ids"]
            for l in open(f"{out}/topologies.jsonl")}
    graphs = [g for s in sorted(glob.glob(f"{out}/graph_shard_*.npz")) for g in gd.load_shard(s)]
    by_dec: dict[int, list[dict]] = {}
    for c in cands:
        by_dec.setdefault(int(c["decision_index"]), []).append(c)
    assert len(graphs) == len(decs) == spec["n_decisions"]
    for di, d in enumerate(decs):
        group = sorted(by_dec[di], key=lambda c: int(c["candidate_index"]))
        assert [c["producer_id"] for c in group] == \
            [topo[d["task_id"]][int(i)] for i in graphs[di].candidate_producers]
        assert graphs[di].decision_digest == int(d["decision_digest"]) == gd.decision_digest(
            d["task_id"], d["trajectory_seed"], d["cycle"], d["order_id"])
        assert int(d["n_eligible"]) == len(group)


def test_no_split_seed_or_label_reaches_a_model_tensor(single_dump):
    """Identity/leakage isolation: the split name, trajectory seed, and every label live only in
    the CSVs. A graph carries no label attribute, its feature widths are exactly the frozen
    schema's, and its tensors are finite."""
    out, _ = single_dump
    graphs = [g for s in sorted(glob.glob(f"{out}/graph_shard_*.npz")) for g in gd.load_shard(s)]
    assert graphs
    for g in graphs:
        assert not hasattr(g, "labels") and not hasattr(g, "split")
        for t in gd.NODE_TYPES:
            assert g.node_features[t].shape[1] == len(gd.NODE_FEATURES[t])
            assert np.isfinite(g.node_features[t]).all()
        for r in gd.RELATION_TYPES:
            assert np.isfinite(g.edge_attr[r]).all()


def test_split_manifest_accounts_for_every_collected_episode(two_split_dump):
    """The manifest records the exact table used plus per (task, split, seed) decision /
    candidate / flat counts, so a reader can confirm from artifacts alone that each split was
    collected from whole, disjoint episodes."""
    out, spec = two_split_dump
    manifest = json.load(open(f"{out}/split_manifest.json"))
    assert manifest == spec["split_manifest"]
    # the WHOLE table is recorded (so a reader can validate disjointness) plus what was walked
    assert set(manifest["registry"][decs_task(out)]) == set(gg.SPLITS)
    assert manifest["collected_splits"] == ["train", "validation"]
    collected = manifest["collected"]
    assert {(c["split"], c["trajectory_seed"]) for c in collected} == \
        {("train", 0), ("train", 1), ("validation", 98)}
    assert sum(c["n_decisions"] for c in collected) == spec["n_decisions"] > 0
    assert sum(c["n_candidates"] for c in collected) == spec["n_candidates"]
    assert all(c["n_flat_primary"] <= c["n_decisions"] for c in collected)
    decs = _rows(f"{out}/decisions.csv")
    assert {(d["split"], int(d["trajectory_seed"])) for d in decs} == \
        {("train", 0), ("train", 1), ("validation", 98)}


def test_dump_spec_records_target_mode_schema_and_provenance(single_dump):
    """spec.json binds the dump to its target, observation contract, source tree, and runtime —
    the same provenance controls EXP-0039 hardened, so a probe can never score a dump built
    under a different label rule or a different revision."""
    out, spec = single_dump
    on_disk = json.load(open(f"{out}/spec.json"))
    assert on_disk == spec
    assert spec["experiment"] == "EXP-0041"
    assert spec["target_mode"] == "single_suffix" and spec["continuation_count"] == 0
    assert spec["schema_hash"] == gd.schema_hash()
    assert len(spec["source"]["tree_sha256"]) == 64
    assert spec["env"]["numpy"] and spec["delta"] == 1.0 and spec["lookahead"] == 1


# --- pre-spend compute budget -----------------------------------------------------------
def test_budget_projection_scales_with_k_and_the_registered_episode_count():
    """The handover requires a projected budget before any pod spend. The projection must scale
    the measured per-decision label cost by the REGISTERED episode count (42, not EXP-0039's 9)
    and extrapolate an unmeasured K linearly from the per-continuation cost — a K that was never
    timed must not be guessed."""
    from tokenomics.eval import graph_generalization_budget as gb

    assert gb.episodes_in_registry() == 42            # 3 tasks x (8 + 3 + 3)
    assert gb.episodes_in_registry(splits=("test",)) == 9
    single = {"seconds_per_decision": 1.0, "continuations": 0}
    mean = {"seconds_per_decision": 3.0, "continuations": 2}  # 1.0 fixed + 2 x 1.0 per fork
    p = gb.project(single, mean, ks=(4, 8))
    assert p["projected_decisions"] == pytest.approx(gb.DECISIONS_PER_EPISODE * 42)
    assert p["modes"]["continuation_mean_k4"]["seconds_per_decision"] == pytest.approx(5.0)
    assert p["modes"]["continuation_mean_k8"]["core_hours"] == pytest.approx(
        p["projected_decisions"] * 9.0 / 3600)
    assert (p["modes"]["single_suffix"]["core_hours"]
            < p["modes"]["continuation_mean_k4"]["core_hours"])
    assert p["artifacts_mb"]["graph_shards_mb"] > 0
    # with no mean measurement there is simply no mean projection (never an invented one)
    assert set(gb.project(single)["modes"]) == {"single_suffix"}
    assert p["rollout_episodes"]["validation_grid"] == (9 * 2 + 1) * 9


# --- protocol v2: staged label collection ---------------------------------------------------
def _tranche_spec(splits, **over):
    """A dump spec shaped like a real one, collected for `splits` only."""
    from tokenomics.train.graph_router_probe import GATE_INPUT_SHA256

    collected = [{"task_id": t, "split": n, "trajectory_seed": s, "n_decisions": 1,
                  "n_flat_primary": 0, "n_candidates": 2}
                 for t in gg.TASK_IDS for n in splits for s in gg.SPLIT_REGISTRY[t][n]]
    spec = {
        "experiment": "EXP-0041", **dict(gg.FROZEN_CONTRACT),
        "task_ids": list(gg.TASK_IDS), "splits": list(splits),
        "split_manifest": {
            "registry": {t: {n: list(gg.SPLIT_REGISTRY[t][n]) for n in gg.SPLITS}
                         for t in gg.SPLIT_REGISTRY},
            "collected": collected},
        "inputs": {"archive": {"sha256": GATE_INPUT_SHA256["archive"]},
                   "producer_ckpt": {"sha256": GATE_INPUT_SHA256["producer_ckpt"]}},
        "source": {"tree_sha256": "a" * 64, "commit": "b" * 40},
        "target_mode": "single_suffix", "continuation_count": 0,
        "license": {"classification": "COVERAGE_PATH_LICENSED", "continuations": 0,
                    "source": "EXP-0040/RESULT.json", "source_sha256": "c" * 64,
                    "evidence": None},
    }
    spec.update(over)
    return spec


def test_fit_tranche_certifies_without_test_labels_but_still_pins_the_test_seeds():
    """Staged collection: a train+validation tranche must certify on its own, so the fit can run
    before a single final-test label is paid for. What it may NOT do is leave the test episodes
    open — the full registry, test seeds included, is still required, which is what stops a later
    tranche from choosing its test seeds with the fit already in hand."""
    fit = _tranche_spec(gg.FIT_TRANCHE)
    gg.certify_dump_contract(fit, expected_splits=gg.FIT_TRANCHE)

    # ...and it is NOT a complete dump
    with pytest.raises(ValueError, match="tranche"):
        gg.certify_dump_contract(fit)

    # the test seeds must still be registered in the fit tranche
    stripped = _tranche_spec(gg.FIT_TRANCHE)
    for t in stripped["split_manifest"]["registry"]:
        stripped["split_manifest"]["registry"][t].pop("test")
    with pytest.raises(ValueError, match="registry"):
        gg.certify_dump_contract(stripped, expected_splits=gg.FIT_TRANCHE)

    # a fit tranche that quietly collected test labels anyway is refused
    leaky = _tranche_spec(gg.FIT_TRANCHE)
    leaky["split_manifest"]["collected"] += [
        {"task_id": t, "split": "test", "trajectory_seed": s}
        for t in gg.TASK_IDS for s in gg.SPLIT_REGISTRY[t]["test"]]
    with pytest.raises(ValueError, match="must not collect labels outside its own splits"):
        gg.certify_dump_contract(leaky, expected_splits=gg.FIT_TRANCHE)


def test_test_tranche_must_belong_to_the_same_experiment_as_the_fit_tranche():
    """Two tranches are collected at different times, so Gate A must prove they are one
    experiment before scoring a model from tranche A against labels from tranche B. Everything the
    verdict depends on has to match; the source COMMIT deliberately need not, because the fit
    happens in between and the tree may legitimately move."""
    fit = _tranche_spec(gg.FIT_TRANCHE)
    test = _tranche_spec(gg.TEST_TRANCHE)
    gg.assert_tranches_agree(fit, test)                      # the happy path

    # a commit that moved between tranches is fine
    moved = _tranche_spec(gg.TEST_TRANCHE, source={"tree_sha256": "d" * 64, "commit": "e" * 40})
    gg.assert_tranches_agree(fit, moved)

    for label, broken in (
        ("target", _tranche_spec(gg.TEST_TRANCHE, target_mode="continuation_mean")),
        ("K", _tranche_spec(gg.TEST_TRANCHE, continuation_count=8)),
        ("teacher window", _tranche_spec(gg.TEST_TRANCHE, lookahead=1)),
        ("inputs", _tranche_spec(gg.TEST_TRANCHE, inputs={"archive_sha256": "0" * 64})),
        ("licence", _tranche_spec(gg.TEST_TRANCHE,
                                  license={"classification": "MEAN", "continuations": 8,
                                           "source": "x", "source_sha256": "f" * 64})),
    ):
        with pytest.raises(ValueError):
            gg.assert_tranches_agree(fit, broken)

    # a registry that changed between tranches is the one that must never pass
    reshuffled = _tranche_spec(gg.TEST_TRANCHE)
    task = gg.TASK_IDS[0]
    reshuffled["split_manifest"]["registry"][task]["test"] = [9991, 9992, 9993]
    with pytest.raises(ValueError, match="registry"):
        gg.assert_tranches_agree(fit, reshuffled)

    # neither tranche may be passed in the other's role
    with pytest.raises(ValueError):
        gg.assert_tranches_agree(test, fit)


# --- protocol v2: rollout-only seeds ---------------------------------------------------------
def test_rollout_registry_is_pre_registered_and_cannot_touch_a_labeled_episode():
    """Rollout episodes need no teacher, so they are cheap enough to fix the thin-selection
    problem: 10 validation seeds per task to choose among the nine (alpha, margin) configurations
    and 20 test seeds per task for Gate B, instead of 3 and 3. Cheap is exactly why the discipline
    has to be explicit — a Gate B episode the model was fitted or checkpoint-selected on is not a
    held-out number, and nothing downstream can detect it."""
    gg.validate_rollout_registry()          # the registered table is legal

    assert set(gg.ROLLOUT_REGISTRY) == set(gg.TASK_IDS)
    for task, splits in gg.ROLLOUT_REGISTRY.items():
        assert len(splits["validation"]) == gg.ROLLOUT_VALIDATION_SEEDS_PER_TASK == 10
        assert len(splits["test"]) == gg.ROLLOUT_TEST_SEEDS_PER_TASK == 20
        labeled = {s for name in gg.SPLITS for s in gg.SPLIT_REGISTRY[task][name]}
        assert not (set(splits["validation"]) | set(splits["test"])) & labeled
        assert not (set(splits["validation"]) | set(splits["test"])) & gg.SPENT_SEEDS[task]

    task = gg.TASK_IDS[0]

    def broken(**changes):
        reg = {t: {k: tuple(v) for k, v in s.items()} for t, s in gg.ROLLOUT_REGISTRY.items()}
        reg[task].update(changes)
        return reg

    # reusing a labeled TRAIN episode is the dangerous one: it is not the same-named split, so a
    # naive check comparing validation-to-validation would wave it through
    with pytest.raises(ValueError, match="reuses labeled train"):   # sizes kept legal on purpose
        gg.validate_rollout_registry(broken(
            test=gg.SPLIT_REGISTRY[task]["train"][:2] + tuple(range(9000, 9018))))
    with pytest.raises(ValueError, match="reuses labeled test"):
        gg.validate_rollout_registry(broken(
            validation=gg.SPLIT_REGISTRY[task]["test"] + tuple(range(9100, 9107))))
    with pytest.raises(ValueError, match="already spent"):
        gg.validate_rollout_registry(broken(
            validation=tuple(gg.SPENT_SEEDS[task]) + tuple(range(9200, 9207))))
    with pytest.raises(ValueError, match="share seeds"):
        gg.validate_rollout_registry(broken(
            test=gg.ROLLOUT_REGISTRY[task]["validation"] + tuple(range(9300, 9310))))
    with pytest.raises(ValueError, match="duplicate"):
        gg.validate_rollout_registry(broken(validation=(1300,) * 10))
    with pytest.raises(ValueError, match="!= the registered"):
        gg.validate_rollout_registry(broken(validation=(1300, 1301)))
    with pytest.raises(ValueError, match="missing or empty"):
        gg.validate_rollout_registry(broken(test=()))
