"""EXP-0039 relational GNN probe: model invariants + the leave-one-seed-out gate plumbing.

torch-guarded. Locks the ARCH.md model contract — schema-shape independent of topology,
identity/permutation invariance, determinism — and that the gate reuses the EXP-0038B
decision-weighted metric machinery. Numbers here are smoke, not the experimental read.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

from tokenomics.domain.graph_view import build_economy_graph_view
from tokenomics.eval import graph_value_router_dump as gd
from tokenomics.train import graph_router_probe as gp

from test_graph_view import first_contested_decision


def _one_graph():
    env, aid, oid, elig = first_contested_decision()
    return gd.encode_graph(build_economy_graph_view(env.system, aid, oid, list(elig)))


def _permute_producers(g: gd.GraphArrays, perm: list[int]) -> gd.GraphArrays:
    """Relabel producer local indices by `perm` (new_index = pos[old]) and reorder the producer
    feature rows to match — a pure identity relabel that must not change predictions."""
    g = copy.deepcopy(g)
    pos = {old: i for i, old in enumerate(perm)}
    g.node_features["producer"] = g.node_features["producer"][perm]
    remap = np.vectorize(lambda x: pos[int(x)])
    for r, (src_t, dst_t, _a) in gd.RELATIONS.items():
        ei = g.edge_index[r]
        if ei.shape[1] == 0:
            continue
        if src_t == "producer":
            ei[0] = remap(ei[0])
        if dst_t == "producer":
            ei[1] = remap(ei[1])
    if g.candidate_producers.shape[0]:
        g.candidate_producers = remap(g.candidate_producers)
    return g


def test_forward_scores_one_value_per_candidate():
    """The shared head emits exactly one scalar per eligible candidate edge."""
    g = _one_graph()
    model = gp.RelationalGNN()
    out = model(gp.Batch([g]))
    assert out.shape == (g.candidate_producers.shape[0],)


def test_batch_of_different_sized_graphs_runs_through_one_model():
    """Topology independence (ARCH.md): a batch mixing graphs with different node/candidate
    counts runs through one model instance without any parameter reshape."""
    g0 = _one_graph()
    env, aid, oid, elig = first_contested_decision()
    g1 = gd.encode_graph(build_economy_graph_view(env.system, aid, oid, list(elig)[:-1]))
    model = gp.RelationalGNN()
    out = model(gp.Batch([g0, g1]))
    assert out.shape == (g0.candidate_producers.shape[0] + g1.candidate_producers.shape[0],)


def test_predictions_are_invariant_to_producer_relabeling():
    """Relabeling producer local indices (same producers, same candidate order) must leave every
    candidate prediction unchanged — no model parameter is indexed by identity (ARCH.md)."""
    g = _one_graph()
    n = g.node_features["producer"].shape[0]
    perm = list(range(n))[::-1]  # reverse the producer indexing
    gperm = _permute_producers(g, perm)
    model = gp.RelationalGNN()
    a = model(gp.Batch([g])).detach().numpy()
    b = model(gp.Batch([gperm])).detach().numpy()
    assert np.allclose(a, b, atol=1e-5)


def test_training_is_deterministic_for_a_fixed_seed():
    """Same training seed -> identical predictions (ARCH.md Phase-5 exit condition)."""
    g = _one_graph()
    c = g.candidate_producers.shape[0]
    labels = list(np.linspace(-1, 1, c))
    group = [0] * c
    m1, _ = gp.train_gnn([g], labels, group, seed=7, epochs=3)
    m2, _ = gp.train_gnn([g], labels, group, seed=7, epochs=3)
    assert np.array_equal(gp.predict(m1, [g]), gp.predict(m2, [g]))


def test_gate_verdict_applies_preregistered_pass_conditions():
    """The probe emits a computed pass/fail from spec.json's five conditions (not a human
    reinterpretation): beats actor on >=2/3, r2>0 on >=2/3, avg relative regret reduction
    >=10%, no task worse than actor by >5%, stable across >=3 seeds."""
    tasks = ["a", "b", "c"]
    per_seed = {ms: {t: {"mean_regret": 1.0, "r2": 0.2, "n_decisions": 10} for t in tasks}
                for ms in (0, 1, 2)}
    actor = {t: {"mean_regret": 2.0} for t in tasks}
    v = gp._verdict(per_seed, actor, tasks)
    assert v["pass"] is True
    assert v["conditions"]["beats_actor_2of3"] and v["conditions"]["no_task_worse_5pct"]
    # one task now far worse than its actor floor -> condition 4 fails -> overall fail
    bad = {ms: {**per_seed[ms], "a": {"mean_regret": 3.0, "r2": 0.2, "n_decisions": 10}}
           for ms in (0, 1, 2)}
    vb = gp._verdict(bad, actor, tasks)
    assert vb["pass"] is False and vb["conditions"]["no_task_worse_5pct"] is False


def test_gate_verdict_requires_the_three_task_shape():
    """The pre-registered gate is defined on exactly 3 tasks (s29/s77/s45); the '>=2 of 3'
    thresholds are meaningless with fewer. A 2-task result that clears every threshold must NOT
    pass — the shape itself is a gate condition."""
    tasks = ["a", "b"]  # only two tasks
    per_seed = {ms: {t: {"mean_regret": 1.0, "r2": 0.2, "n_decisions": 10} for t in tasks}
                for ms in (0, 1, 2)}
    actor = {t: {"mean_regret": 2.0} for t in tasks}
    v = gp._verdict(per_seed, actor, tasks)
    assert v["pass"] is False
    assert v["conditions"]["three_tasks"] is False


def test_gate_verdict_no_pass_when_a_seed_is_unstable():
    """Stability must apply to EVERY gate metric, not just regret-vs-floor. Two seeds have
    strongly negative r2 and one is wildly positive; the mean r2 is positive but the result is
    unstable across seeds -> HANDOVER.md:241 requires no-pass. (Regression: averaging r2 across
    seeds while checking only regret dispersion false-PASSED r2 = -100/-100/201.)"""
    tasks = ["a", "b", "c"]
    r2_by_seed = {0: -100.0, 1: -100.0, 2: 201.0}
    per_seed = {ms: {t: {"mean_regret": 1.0, "r2": r2_by_seed[ms], "n_decisions": 10}
                     for t in tasks} for ms in (0, 1, 2)}
    actor = {t: {"mean_regret": 2.0} for t in tasks}
    v = gp._verdict(per_seed, actor, tasks)
    assert v["pass"] is False
    assert v["conditions"]["stable_ge3_seeds"] is False


def test_fold_normalizer_standardizes_continuous_only_and_is_train_fitted():
    """ARCH.md preprocessing: fit mean/std on the training fold's continuous (log1p /
    signed_log1p) columns and apply to both train and held-out; binary/one-hot columns pass
    through untouched. After apply, a continuous producer column over the fitted set has ~0
    mean and ~unit std; a binary column is bit-identical."""
    g = _one_graph()
    graphs = [g, _one_graph()]
    normz = gp.FoldNormalizer.fit(graphs)
    out = normz.apply(graphs)

    cols = gd.NODE_FEATURES["producer"]
    cont = next(i for i, (_n, t) in enumerate(cols) if t == "log1p")
    binary = next(i for i, (_n, t) in enumerate(cols) if t == "binary")
    stacked = np.concatenate([o.node_features["producer"] for o in out], axis=0)
    assert abs(float(stacked[:, cont].mean())) < 1e-4
    # binary column is left exactly as-is (raw 0/1), not standardized
    raw_bin = np.concatenate([g.node_features["producer"] for g in graphs], axis=0)[:, binary]
    assert np.array_equal(stacked[:, binary], raw_bin)
    # held-out apply uses the SAME stored stats (no refit): idempotent stats object
    assert normz.apply([g])[0].node_features["producer"].shape == g.node_features["producer"].shape


def test_flat_decisions_are_excluded_from_training_and_counted():
    """HANDOVER.md: train PRIMARILY on non-flat decisions. A decision whose candidate teacher
    labels are all equal carries no within-decision preference — feeding it at equal
    decision-weight (flats are ~69% of the real dump) swamps the signal. The gate must drop
    flats from each training fold and report how many it excluded."""
    g = _one_graph()
    c = g.candidate_producers.shape[0]
    graphs, cand_rows = [], []
    di = 0
    for seed in (0, 1, 2):
        for flat in (False, True):  # one non-flat + one flat decision per seed
            graphs.append(g)
            for ci in range(c):
                lab = 0.5 if flat else float(ci)
                cand_rows.append({
                    "decision_index": di, "task_id": "t", "seed": seed, "candidate_index": ci,
                    "teacher_label": lab, "actor_score": float(c - ci)})
            di += 1
    res = gp.heldout_gate(graphs, cand_rows, model_seed=0, epochs=2)
    # 3 leave-one-seed-out folds, each trains on 2 seeds -> 2 flat + 2 non-flat per fold
    assert res["t"]["n_train_flat_excluded"] == 6
    assert res["t"]["n_train_decisions"] == 6


def test_load_dump_requires_schema_and_validates_per_decision_alignment(tmp_path):
    """_load_dump is the pre-scoring gate (ARCH.md:343): it REQUIRES graph_schema.json and
    validates one graph per decision with exact per-decision candidate coverage/index/producer
    alignment — not just the global row count. A dump missing the schema, or with candidate rows
    collapsed onto one decision, must be rejected before any compute."""
    import csv
    import os

    from tokenomics.eval.baselines import PRODUCING_BASELINES

    from test_graph_view import _scale_task

    out = str(tmp_path / "dump_graph")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, log=lambda *a, **k: None)
    graphs, cand_rows = gp._load_dump(out)  # a clean dump loads
    assert graphs and cand_rows

    schema_path = os.path.join(out, "graph_schema.json")
    os.rename(schema_path, schema_path + ".bak")
    with pytest.raises(ValueError):  # (a) schema is mandatory
        gp._load_dump(out)
    os.rename(schema_path + ".bak", schema_path)

    cpath = os.path.join(out, "candidates.csv")

    def _rewrite(mutate):
        with open(cpath) as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0].keys())
        rows = mutate([dict(r) for r in rows])
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # (b) collapsing every candidate onto decision 0 breaks per-decision coverage/counts
    def _collapse(rs):
        for r in rs:
            r["decision_index"] = 0
        return rs
    _rewrite(_collapse)
    with pytest.raises(ValueError):
        gp._load_dump(out)

    # (c) a candidate whose seed disagrees with decisions.csv must be rejected (metadata must be
    # validated against the decisions.csv authority, not just be internally consistent)
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, log=lambda *a, **k: None)

    def _bad_seed(rs):
        rs[0]["seed"] = "999999"
        return rs
    _rewrite(_bad_seed)
    with pytest.raises(ValueError):
        gp._load_dump(out)


def test_load_dump_rejects_a_swapped_graph_payload(tmp_path):
    """Finding-4.1: _load_dump must PROVE graphs[di] belongs to decisions.csv row di. Swapping
    two structurally-identical decisions (same candidate producer set, so every count/producer-
    alignment check still passes) but different live features previously went undetected. The
    per-graph decision_digest binding now catches it: after swapping two such graph payloads and
    re-serializing the shard, the load must raise."""
    import glob

    from tokenomics.eval.baselines import PRODUCING_BASELINES

    from test_graph_view import _scale_task

    out = str(tmp_path / "dump_graph")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, log=lambda *a, **k: None)
    gp._load_dump(out)  # clean dump loads

    shards = sorted(glob.glob(f"{out}/graph_shard_*.npz"))
    graphs = []
    for sh in shards:
        graphs.extend(gd.load_shard(sh))
    # find two decisions with identical candidate sets (all OTHER checks pass after a swap, so
    # only the decision_digest binding can catch it)
    pair = None
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            if (graphs[i].decision_digest != graphs[j].decision_digest
                    and np.array_equal(graphs[i].candidate_producers, graphs[j].candidate_producers)):
                pair = (i, j)
                break
        if pair:
            break
    assert pair, "smoke dump needs two same-candidate-set decisions to exercise the swap"
    i, j = pair
    graphs[i], graphs[j] = graphs[j], graphs[i]  # swap full payloads (digest rides along)
    import os
    for sh in shards:
        os.remove(sh)
    gd.write_shard(f"{out}/graph_shard_000.npz", graphs)  # rewrite as a single shard
    with pytest.raises(ValueError):
        gp._load_dump(out)


def test_load_dump_validates_preregistered_dump_config(tmp_path):
    """Finding-4.2: a dump can preserve the exact decision count/CSV while being generated on a
    DIFFERENT teacher or task set (wrong delta/lookahead/task_ids), which silently changes every
    label. _load_dump must read the dump's own spec.json and reject any deviation from the
    pre-registered gate config, and must reject a dump with no spec.json when a config is expected."""
    import os

    from tokenomics.eval.baselines import PRODUCING_BASELINES

    from test_graph_view import _scale_task

    out = str(tmp_path / "dump_graph")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, log=lambda *a, **k: None)
    task = next(iter(json_task_ids(out)))

    # matching config loads
    gp._load_dump(out, expected_config={"delta": 1.0, "lookahead": 2, "task_ids": [task]})
    # wrong teacher lookahead (the real gate is 8; this dump is 2) -> reject
    with pytest.raises(ValueError):
        gp._load_dump(out, expected_config={"lookahead": 8})
    # wrong task set -> reject (labels would be from a different trajectory)
    with pytest.raises(ValueError):
        gp._load_dump(out, expected_config={"task_ids": ["some-other-task"]})
    # a dump with no spec.json cannot be certified against a config
    os.remove(os.path.join(out, "spec.json"))
    with pytest.raises(ValueError):
        gp._load_dump(out, expected_config={"delta": 1.0})


def json_task_ids(out: str) -> list:
    import json
    return json.load(open(f"{out}/spec.json"))["task_ids"]


def test_load_dump_validates_task_seed_sets_and_input_fingerprints(tmp_path):
    """Finding-5.1/5.2: 'three seeds' and a name/shape config are not enough. A wrong producer or
    actor (same shape) silently changes labels/floor, and any three seed VALUES pass a count check.
    _load_dump must reject a dump whose recorded input SHA-256s or task->seed sets deviate from the
    pre-registered ones, and cannot certify inputs when the dump has no fingerprint manifest."""
    import json

    from tokenomics.eval.baselines import PRODUCING_BASELINES

    from test_graph_view import _scale_task

    out = str(tmp_path / "dump_graph")
    manifest = {"archive": {"path": "a", "sha256": "AAA"},
                "producer_ckpt": {"path": "p", "sha256": "PPP"},
                "actor_ckpt": {"path": "c", "sha256": "CCC"}}
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=2, horizon=12, out_dir=out,
        input_manifest=manifest, log=lambda *a, **k: None)
    good = {t: set(v) for t, v in json.load(open(f"{out}/spec.json"))["task_seeds"].items()}

    # correct seed sets + input fingerprints certify
    gp._load_dump(out, expected_task_seeds=good,
                  expected_inputs={"archive": "AAA", "producer_ckpt": "PPP", "actor_ckpt": "CCC"})
    # right COUNT, wrong seed VALUES -> reject (Finding-5.2)
    task = next(iter(good))
    bad_seeds = {t: (v if t != task else {900, 901}) for t, v in good.items()}
    with pytest.raises(ValueError):
        gp._load_dump(out, expected_task_seeds=bad_seeds)
    # wrong producer fingerprint -> reject (a wrong producer changes teacher labels, Finding-5.1)
    with pytest.raises(ValueError):
        gp._load_dump(out, expected_inputs={"producer_ckpt": "NOT-PPP"})
    # a dump with NO input manifest cannot satisfy an input-hash expectation
    out2 = str(tmp_path / "dump2")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=2, horizon=12, out_dir=out2, log=lambda *a, **k: None)
    with pytest.raises(ValueError):
        gp._load_dump(out2, expected_inputs={"producer_ckpt": "PPP"})


def test_heldout_gate_scores_each_task_leave_one_seed_out():
    """The gate trains on two seeds, holds out the third, pools held-out decisions, and returns
    decision-weighted metrics per task with the EXP-0038B metric keys. Smoke plumbing: synthetic
    graphs/labels over 3 seeds, tiny epochs."""
    g = _one_graph()
    c = g.candidate_producers.shape[0]
    graphs, cand_rows = [], []
    di = 0
    for seed in (0, 1, 2):
        for _ in range(2):  # two decisions per seed
            graphs.append(g)
            for ci in range(c):
                cand_rows.append({
                    "decision_index": di, "task_id": "t", "seed": seed, "candidate_index": ci,
                    "teacher_label": float(ci) + 0.1 * seed, "actor_score": float(c - ci)})
            di += 1
    res = gp.heldout_gate(graphs, cand_rows, model_seed=0, epochs=2)
    assert set(res) == {"t"}
    assert {"mean_regret", "top1_match", "r2", "n_decisions"} <= set(res["t"])
    assert res["t"]["mean_regret"] >= 0.0


def test_actor_floor_override_must_be_the_pinned_canonical_0038b_floor():
    """Finding-6.2: the `--actor-floor-json` fallback must not let arbitrary values weaken the
    licensing bar (the verdict needs GNN regret < floor regret, so a bigger floor is easier). The
    override is accepted only when it equals the pinned EXP-0038B deployed-actor floor."""
    gp.assert_override_is_canonical(dict(gp.GATE_ACTOR_FLOOR_0038B))  # exact canonical: ok
    k = next(iter(gp.GATE_ACTOR_FLOOR_0038B))
    for bump in (5.0, 5e-7):  # Finding-7.2: even a sub-epsilon inflation must be rejected (exact eq)
        inflated = dict(gp.GATE_ACTOR_FLOOR_0038B)
        inflated[k] += bump  # a larger floor silently makes a strict `<` gate easier to pass
        with pytest.raises(ValueError):
            gp.assert_override_is_canonical(inflated)
    with pytest.raises(ValueError):  # wrong task set (not the three gate tasks)
        gp.assert_override_is_canonical({"some-other-task": 1.0})


def test_source_identity_is_a_required_cross_stage_invariant(tmp_path):
    """Finding-7.1/8.1: recording the source hash is not enough — it must be ENFORCED, and the commit
    must be a FULL 40-char sha (an abbreviated/forged id is rejected). _load_dump rejects a dump whose
    source identity differs from the probe's (tampered/zeroed hash, or a dump from other code) BEFORE
    any training."""
    import json

    from tokenomics.eval.baselines import PRODUCING_BASELINES
    from tokenomics.eval.graph_value_router_dump import source_fingerprint

    from test_graph_view import _scale_task

    sha_a, sha_b = "a" * 40, "b" * 40
    gp._validate_commit(sha_a)  # a full 40-char sha is well-shaped
    # Finding-8.1: None/garbage AND abbreviated ids (the old 7-40 range accepted these) are rejected
    for bad in (None, "", "not-a-git-object", "a1b2c3d", "deadbee"):
        with pytest.raises(ValueError):
            gp._validate_commit(bad)

    live = source_fingerprint()["tree_sha256"]
    gp.assert_source_matches({"tree_sha256": live, "commit": sha_a}, expected_commit=sha_a)  # match: ok
    with pytest.raises(ValueError):  # zeroed/tampered hash != live tree
        gp.assert_source_matches({"tree_sha256": "0" * 64, "commit": sha_a}, expected_commit=sha_a)
    with pytest.raises(ValueError):  # dump built from a different revision than the probe
        gp.assert_source_matches({"tree_sha256": live, "commit": sha_a}, expected_commit=sha_b)

    # end-to-end: a dump stamped with the live source but a tampered spec is rejected by _load_dump
    out = str(tmp_path / "dump_graph")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=2, horizon=12, out_dir=out, source_commit=sha_a,
        log=lambda *a, **k: None)
    gp._load_dump(out, expected_source_commit=sha_a, require_source=True)  # intact: ok
    spec = json.load(open(f"{out}/spec.json"))
    spec["source"]["tree_sha256"] = "0" * 64  # forge the recorded identity
    json.dump(spec, open(f"{out}/spec.json", "w"))
    with pytest.raises(ValueError):
        gp._load_dump(out, expected_source_commit=sha_a, require_source=True)


def test_resolve_verified_commit_binds_a_clean_tree_to_a_real_commit(tmp_path):
    """Finding-8.1: the launcher-side binding resolves a FULL 40-char HEAD only when the object exists
    AND the producing tree is clean — a dirty/untracked tree (whose live fingerprint would not match
    the claimed revision) raises, and a non-repo raises."""
    import subprocess

    from tokenomics.eval.graph_value_router_dump import resolve_verified_commit

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                       capture_output=True, text=True)

    with pytest.raises(ValueError):  # not a git repo
        resolve_verified_commit(str(tmp_path))

    git("init", "-q")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    git("add", "-A"); git("commit", "-qm", "init")
    sha = resolve_verified_commit(str(tmp_path))  # clean tree -> full sha
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    (tmp_path / "b.py").write_text("y = 2\n")  # untracked file -> dirty producing tree
    with pytest.raises(ValueError):
        resolve_verified_commit(str(tmp_path))


def test_dump_and_probe_environment_manifest_is_captured_and_compared():
    """Finding-8.2/9.2: deps are open lower bounds installed fresh on the pod, so both stages record a
    runtime manifest (python + versions + machine ARCH + numpy BLAS + torch build) and the read compares
    them exactly — a probe cannot score a dump built under a different torch/numpy/arch/backend, even at
    identical package versions."""
    from tokenomics.eval.graph_value_router_dump import environment_manifest

    env = environment_manifest()
    assert env["python"] and env["torch"] and set(env) >= set(gp._ENV_GATED_KEYS)
    assert "machine" in gp._ENV_GATED_KEYS  # architecture is gated (Finding-9.2)
    gp.assert_env_matches(env)  # live env matches itself
    # each gated axis independently rejects a mismatch — version, ARCH, and numeric backend
    for key, bad in (("torch", "0.0.0-fake"), ("machine", "other-arch"),
                     ("numpy_blas", "different"), ("torch_build", "deadbeef")):
        with pytest.raises(ValueError):
            gp.assert_env_matches({**env, key: bad})
    with pytest.raises(ValueError):  # a dump with no env manifest cannot be certified
        gp.assert_env_matches(None)


def test_canonical_actor_floor_is_derived_from_the_pinned_artifact(tmp_path):
    """Finding-6.2: the safe way to build the fallback derives the floor mechanically from the
    canonical EXP-0038B artifact, gated by its SHA-256. The checked-in artifact yields exactly the
    pinned floor; any other file (wrong bytes) is rejected before its values are ever read."""
    import os

    if os.path.exists(gp.GATE_FLOOR_ARTIFACT):
        assert gp.canonical_actor_floor() == gp.GATE_ACTOR_FLOOR_0038B
    tampered = tmp_path / "floor.json"
    tampered.write_text('{"floors": {"actor": {}}}')
    with pytest.raises(ValueError):  # SHA-256 mismatch aborts before parsing
        gp.canonical_actor_floor(str(tampered))
