"""EXP-0039 graph_primitives_v1 encoder: EconomyGraphView -> id-free numeric arrays.

These lock the observation contract the GNN consumes: schema-ordered feature columns, the
declared log1p transform, candidate<->eligible alignment, correct edge endpoints/attributes,
and — the leakage guard — that no raw domain identity ever reaches a model tensor.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenomics.domain.graph_view import build_economy_graph_view
from tokenomics.eval import graph_value_router_dump as gd

from test_graph_view import _scale_task, first_contested_decision


def _view():
    env, aid, oid, elig = first_contested_decision()
    return env, build_economy_graph_view(env.system, aid, oid, elig), aid, oid, elig


def test_schema_hash_is_stable_and_feature_widths_match_arrays():
    """The schema hash is deterministic (so a probe is always scored against the exact
    observation it trained on) and every node feature matrix is exactly as wide as the
    schema's ordered feature list for that node type."""
    assert gd.schema_hash() == gd.schema_hash()
    _, view, *_ = _view()
    g = gd.encode_graph(view)
    for node_type, feats in gd.NODE_FEATURES.items():
        assert g.node_features[node_type].shape[1] == len(feats)
    for rel, (_s, _d, attrs) in gd.RELATIONS.items():
        assert g.edge_attr[rel].shape[1] == len(attrs)


def test_producer_features_are_log1p_and_binary_per_schema():
    """A log1p column stores log1p(raw); a binary column stores the raw flag. Checked on
    the producer node: funds is log1p, cutoff_is_defined is a pass-through 0/1."""
    env, view, aid, oid, elig = _view()
    g = gd.encode_graph(view)
    cols = [n for n, _ in gd.NODE_FEATURES["producer"]]
    pid = elig[0]
    li = g.id_maps["producer"].index(pid)
    p = env.system.producing_agents[pid]
    row = g.node_features["producer"][li]
    assert row[cols.index("funds")] == np.float32(np.log1p(p.funds))
    assert row[cols.index("cutoff_is_defined")] in (0.0, 1.0)


def test_candidate_indices_align_one_to_one_with_eligible_producers():
    """Candidate producer local indices map back, in order, to exactly the eligible set the
    legacy dump scored — the alignment the label join depends on."""
    _, view, aid, oid, elig = _view()
    g = gd.encode_graph(view)
    recovered = [g.id_maps["producer"][i] for i in g.candidate_producers]
    assert recovered == list(elig)
    # target_product resolves to the current order's product
    assert g.id_maps["product"][g.target_product] == \
        {e.order_id: e.product for e in view.requests}[oid]


def test_capacity_state_edges_resolve_to_the_right_pairs_and_log1p_attrs():
    """capacity_state edge endpoints decode to the intended (producer, capacity_type) domain
    pair, and its attributes are the log1p of current/max/reserved/free/headroom/... in
    schema order."""
    env, view, aid, oid, elig = _view()
    g = gd.encode_graph(view)
    src, dst = g.edge_index["capacity_state"]
    attrs = [n for n, _ in gd.RELATIONS["capacity_state"][2]]
    got = {}
    for k in range(src.shape[0] if src.ndim else 0):
        pid = g.id_maps["producer"][src[k]]
        t = g.id_maps["capacity_type"][dst[k]]
        got[(pid, t)] = g.edge_attr["capacity_state"][k]
    pid = elig[0]
    p = env.system.producing_agents[pid]
    t = next(iter(p.capacities))
    row = got[(pid, t)]
    assert row[attrs.index("current")] == np.float32(np.log1p(p.capacities[t]))
    assert row[attrs.index("max")] == np.float32(np.log1p(p.max_capacities[t]))


def test_shard_round_trip_is_exact_and_offsets_isolate_graphs(tmp_path):
    """NPZ shard round-trip preserves every numeric tensor exactly, and per-graph offsets
    keep neighboring graphs separate: two distinct decision graphs written to one shard come
    back individually identical to their originals, in order."""
    env, aid, oid, elig = first_contested_decision()
    # a second, structurally different graph: drop one eligible candidate
    g0 = gd.encode_graph(build_economy_graph_view(env.system, aid, oid, elig))
    g1 = gd.encode_graph(build_economy_graph_view(env.system, aid, oid, elig[:-1]))
    path = str(tmp_path / "graph_shard_000.npz")
    gd.write_shard(path, [g0, g1])
    back = gd.load_shard(path)
    assert len(back) == 2
    for orig, got in ((g0, back[0]), (g1, back[1])):
        for t in gd.NODE_TYPES:
            assert np.array_equal(orig.node_features[t], got.node_features[t])
        for r in gd.RELATION_TYPES:
            assert np.array_equal(orig.edge_index[r], got.edge_index[r])
            assert np.array_equal(orig.edge_attr[r], got.edge_attr[r])
        assert np.array_equal(orig.candidate_producers, got.candidate_producers)
        assert (orig.current_order, orig.target_product) == (got.current_order, got.target_product)
    # the two graphs really differ (candidate count), proving offsets did not bleed
    assert back[0].candidate_producers.shape[0] != back[1].candidate_producers.shape[0]


def test_graph_dump_smoke_writes_aligned_dump(tmp_path):
    """End-to-end (no actor, seconds): the dump emits one graph per contested decision in
    decisions.csv order, candidates.csv rows equal the total eligible candidates, and every
    graph's candidate set aligns with candidates.csv via the task topology. Labels/actor
    scores live only in candidates.csv, never in a graph tensor (which has no label field)."""
    import csv
    import json

    from tokenomics.eval.baselines import PRODUCING_BASELINES

    out = str(tmp_path / "dump_graph")
    spec = gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, log=lambda *a, **k: None)

    with open(f"{out}/decisions.csv") as f:
        drows = list(csv.DictReader(f))
    with open(f"{out}/candidates.csv") as f:
        crows = list(csv.DictReader(f))
    assert spec["n_decisions"] == len(drows) > 0
    assert spec["n_candidates"] == len(crows)
    assert len(crows) == sum(int(r["n_eligible"]) for r in drows)

    import glob
    graphs = []
    for shard in sorted(glob.glob(f"{out}/graph_shard_*.npz")):
        graphs.extend(gd.load_shard(shard))
    assert len(graphs) == spec["n_decisions"]

    topo = {json.loads(l)["task_id"]: json.loads(l)
            for l in open(f"{out}/topologies.jsonl")}
    # per decision: graph candidate count matches n_eligible, and candidate producers
    # decode (via topology) to exactly the candidates.csv producer_ids in order
    by_decision: dict[int, list] = {}
    for r in crows:
        by_decision.setdefault(int(r["decision_index"]), []).append(r)
    for di, drow in enumerate(drows):
        g = graphs[di]
        assert g.candidate_producers.shape[0] == int(drow["n_eligible"])
        pids = topo[drow["task_id"]]["producer_ids"]
        recovered = [pids[i] for i in g.candidate_producers]
        expected = [c["producer_id"] for c in sorted(by_decision[di],
                                                      key=lambda c: int(c["candidate_index"]))]
        assert recovered == expected
        assert np.isfinite(g.node_features["producer"]).all()

    schema = json.load(open(f"{out}/graph_schema.json"))
    assert schema["schema_hash"] == spec["schema_hash"] == gd.schema_hash()
    assert not hasattr(graphs[0], "labels")  # no teacher label ever rides in a graph tensor


def test_free_capacity_uses_signed_transform_and_stays_finite():
    """`free = current - reserved` is signable (a prior-cycle reservation can exceed
    post-degradation capacity), so it must use a sign-preserving transform — plain log1p
    yields NaN/-inf for free < 0 and would poison the GNN. signed_log1p keeps it finite and
    monotone, and the schema must declare `free` with it (regression: the -inf blocker)."""
    assert gd._transform(-1.0, "signed_log1p") == -float(np.log1p(1.0))
    for v in (-9.0, -1.0, 0.0, 3.0, 1e6):
        assert np.isfinite(gd._transform(v, "signed_log1p"))
    free_kind = dict(gd.RELATIONS["capacity_state"][2])["free"]
    assert free_kind == "signed_log1p", "free must not be plain log1p"


def test_dump_tensors_are_all_finite(tmp_path):
    """Exhaustive finite guard (not just the producer node): every node feature matrix AND
    every edge-attr matrix in a real dump is finite. run_graph_dump also asserts this before
    serializing, so a non-finite feature fails the dump loudly instead of reaching a pod."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    out = str(tmp_path / "dump_graph")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, log=lambda *a, **k: None)
    import glob
    graphs = []
    for shard in sorted(glob.glob(f"{out}/graph_shard_*.npz")):
        graphs.extend(gd.load_shard(shard))
    for g in graphs:
        for t in gd.NODE_TYPES:
            assert np.isfinite(g.node_features[t]).all()
        for r in gd.RELATION_TYPES:
            assert np.isfinite(g.edge_attr[r]).all()


def test_shard_binds_each_graph_to_its_decision_row(tmp_path):
    """Finding-4.1: a graph payload must be provably bound to decisions.csv row `di`, or a
    swap of two structurally-identical decisions (same candidate set, different live state)
    silently mislabels leakage-relevant features. The dump stamps each graph with a
    decision_digest of (task_id,seed,cycle,order_id) OUTSIDE the model tensors; write/load
    round-trips it, and every real-dump graph's digest equals the digest recomputed from its
    decisions.csv row."""
    import csv

    from tokenomics.eval.baselines import PRODUCING_BASELINES

    # round-trip: the digest survives shard serialization unchanged
    env, aid, oid, elig = first_contested_decision()
    g = gd.encode_graph(build_economy_graph_view(env.system, aid, oid, elig))
    g.decision_digest = gd.decision_digest("taskX", 2, 7, "order-9")
    path = str(tmp_path / "graph_shard_000.npz")
    gd.write_shard(path, [g])
    assert gd.load_shard(path)[0].decision_digest == g.decision_digest

    # real dump: each graph is bound to its decisions.csv row
    out = str(tmp_path / "dump_graph")
    gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, log=lambda *a, **k: None)
    import glob
    graphs = []
    for shard in sorted(glob.glob(f"{out}/graph_shard_*.npz")):
        graphs.extend(gd.load_shard(shard))
    with open(f"{out}/decisions.csv") as f:
        drows = list(csv.DictReader(f))
    assert graphs and len(graphs) == len(drows)
    for di, drow in enumerate(drows):
        expect = gd.decision_digest(drow["task_id"], drow["seed"], drow["cycle"], drow["order_id"])
        assert graphs[di].decision_digest == expect != 0


def test_dump_records_task_seed_sets_and_input_manifest(tmp_path):
    """Finding-5.1/5.2: the dump manifest must bind the ACTUAL inputs (a wrong producer changes
    labels, a wrong actor weakens the floor — neither fails a name/shape check) and the exact
    task->seed sets. `sha256_file` fingerprints a file; the dump spec.json records the passed input
    manifest verbatim and the exact anchored seed set per task (from the real trajectory)."""
    import csv
    import hashlib
    import json

    from tokenomics.eval.baselines import PRODUCING_BASELINES

    f = tmp_path / "x.bin"
    f.write_bytes(b"provenance")
    assert gd.sha256_file(str(f)) == hashlib.sha256(b"provenance").hexdigest()

    out = str(tmp_path / "dump_graph")
    manifest = {"archive": {"path": "gate_archive.jsonl", "sha256": "dead"},
                "producer_ckpt": {"path": "prod.pt", "sha256": "beef"}, "actor_ckpt": None}
    spec = gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=2, horizon=12, out_dir=out,
        input_manifest=manifest, log=lambda *a, **k: None)
    dspec = json.load(open(f"{out}/spec.json"))
    assert dspec["inputs"] == manifest
    # task_seeds records exactly the seeds that appear in candidates.csv for each task
    with open(f"{out}/candidates.csv") as fh:
        rows = list(csv.DictReader(fh))
    by_task: dict[str, set] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], set()).add(int(r["seed"]))
    assert {t: set(v) for t, v in dspec["task_seeds"].items()} == by_task
    task = spec["task_ids"][0]
    assert len(dspec["task_seeds"][task]) == 2  # eval_seeds anchored seeds


def test_input_manifest_validation_fails_a_wrong_input_before_dumping():
    """Finding-6.3: a wrong staged checkpoint must fail at LAUNCH, not after ~4 core-hours of
    teacher rollout. `assert_inputs_match` compares the dump's recorded {role: {sha256}} manifest
    against the pinned {role: sha256} and raises on any mismatch or missing role."""
    expected = {"archive": "aaa", "producer_ckpt": "bbb", "actor_ckpt": "ccc"}
    good = {"archive": {"path": "a", "sha256": "aaa"},
            "producer_ckpt": {"path": "p", "sha256": "bbb"},
            "actor_ckpt": {"path": "x", "sha256": "ccc"}}
    gd.assert_inputs_match(good, expected)  # exact match: no raise
    with pytest.raises(ValueError):  # a swapped producer (labels would change) is rejected
        gd.assert_inputs_match({**good, "producer_ckpt": {"path": "p", "sha256": "WRONG"}}, expected)
    with pytest.raises(ValueError):  # a missing/None role is rejected, not silently skipped
        gd.assert_inputs_match({**good, "actor_ckpt": None}, expected)


def test_dump_records_source_fingerprint(tmp_path):
    """Finding-6.1/7.1: every dump binds to the immutable identity of the code that produced it. The
    fingerprint hashes the bytes of EVERY `.py` in the synced package tree (not a hand-picked module
    list — so transitive producers like the teacher/env/curriculum are covered), is deterministic and
    git-independent (reproducible on the pod, where `.git/` is not synced), and records an optional
    commit verbatim. The dump spec.json carries it."""
    from tokenomics.eval.baselines import PRODUCING_BASELINES

    fp = gd.source_fingerprint()
    assert len(fp["tree_sha256"]) == 64 and fp["commit"] is None and fp["n_files"] > 4
    assert gd.source_fingerprint()["tree_sha256"] == fp["tree_sha256"]  # deterministic
    assert gd.source_fingerprint("cafe")["commit"] == "cafe"

    out = str(tmp_path / "dump_graph")
    spec = gd.run_graph_dump(
        [_scale_task(0, depth=3)], prod=PRODUCING_BASELINES["stock_aware"],
        delta=1.0, lookahead=2, eval_seeds=1, horizon=12, out_dir=out, source_commit="deadbeef",
        log=lambda *a, **k: None)
    import json
    dspec = json.load(open(f"{out}/spec.json"))
    assert dspec["source"] == spec["source"]
    assert dspec["source"]["commit"] == "deadbeef"
    assert dspec["source"]["tree_sha256"] == fp["tree_sha256"]


def test_no_raw_domain_id_reaches_a_model_tensor():
    """Identity isolation (ARCH.md): node feature matrices and edge index/attr arrays are
    numeric and hold only local indices — raw producer/product/order IDs live only in the
    separate audit id_maps. Edge indices must stay within their endpoint node counts."""
    _, view, *_ = _view()
    g = gd.encode_graph(view)
    for arr in g.node_features.values():
        assert arr.dtype == np.float32
    counts = {t: g.node_features[t].shape[0] for t in gd.NODE_TYPES}
    counts["decision"] = 1
    for rel, (src_t, dst_t, _a) in gd.RELATIONS.items():
        ei = g.edge_index[rel]
        assert ei.dtype == np.int64
        if ei.shape[1]:
            assert ei[0].max() < counts[src_t] and ei[1].max() < counts[dst_t]
