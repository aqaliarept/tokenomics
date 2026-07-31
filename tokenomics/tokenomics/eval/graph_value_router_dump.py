"""EXP-0039 graph_primitives_v1 encoder + dump: turn the domain EconomyGraphView into
id-free numeric graph arrays, drive the fixed blend_0.5 trajectory to materialize one
graph per contested routing decision, and serialize a pickle-free durable dump.

Layered per ARCH.md:
- SCHEMA: fixed node/relation registry, ordered feature names + transforms, reverse
  relations, commitment visibility, and a schema hash. Changing a primary feature's
  meaning/visibility requires a new schema version.
- encode_graph: EconomyGraphView -> GraphArrays. Assigns contiguous per-type LOCAL
  indices, applies the declared log1p/binary transforms (fold-local standardization is
  the probe's job, not the dump's), and keeps raw domain IDs in a SEPARATE audit map —
  never in a model tensor (ARCH.md identity isolation).

The GNN, fold preprocessing, and gate live in `train.graph_router_probe`; the teacher
label + actor floor reuse EXP-0037/0038 unchanged so the gate read matches EXP-0038B.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

from ..domain.graph_view import EconomyGraphView, build_economy_graph_view

if TYPE_CHECKING:
    from ..train.value_router import ResidualScoreModel
    from .baselines import Policy
    from .curriculum import TaskRecord


# --- SCHEMA: graph_primitives_v1 ------------------------------------------------------
SCHEMA_NAME = "graph_primitives_v1"

# node type -> ordered (feature_name, transform). "log1p" = nonnegative count/quantity/
# token/capacity/latency (log1p in the dump, standardized per fold in the probe);
# "binary" = 0/1 flag or one-hot component, left unstandardized. `decision` and
# `capacity_type` carry no domain features (schema-role embedding only).
NODE_FEATURES: dict[str, list[tuple[str, str]]] = {
    "decision": [],
    "order": [
        ("original_quantity", "log1p"), ("unresolved_quantity", "log1p"),
        ("accepted_quantity", "log1p"), ("completed_quantity", "log1p"),
        ("failed_quantity", "log1p"), ("available_tokens", "log1p"),
        ("burnt_tokens", "log1p"), ("order_age", "log1p"), ("ttl_remaining", "log1p"),
        ("funded", "binary"), ("kind_consumer", "binary"),
        ("kind_replenishment", "binary"), ("kind_investment", "binary"),
    ],
    "orderer": [("source_consumer", "binary"), ("source_producer", "binary")],
    "producer": [
        ("cut_off_price", "log1p"), ("cutoff_is_defined", "binary"), ("funds", "log1p"),
        ("inventory_limit", "log1p"), ("upgrade_running", "binary"),
        ("restoration_running", "binary"), ("inventory_expansion_running", "binary"),
        ("has_inventory_expansion", "binary"),
    ],
    "product": [("demand_ema", "log1p"), ("rejected_demand_ema", "log1p")],
    "capacity_type": [],
}
NODE_TYPES: list[str] = list(NODE_FEATURES)

# relation -> (src_type, dst_type, ordered (attr_name, transform)). Relations that share
# endpoint types but differ semantically (the three producer->product, the two
# order->producer) are DISTINCT relation names. Reverse edges are mirrored by the model.
RELATIONS: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "current_order": ("decision", "order", []),
    "current_orderer": ("decision", "orderer", []),
    "candidate": ("decision", "producer", []),
    "owns": ("orderer", "order", []),
    "stream_for": ("orderer", "product", []),
    "source_of": ("producer", "orderer", []),
    "requests": ("order", "product", []),
    "can_make": ("producer", "product", []),
    "requires_capacity": ("product", "capacity_type", [("units", "log1p")]),
    "requires_input": ("product", "product", [("units", "log1p")]),
    "capacity_state": ("producer", "capacity_type", [
        ("current", "log1p"), ("max", "log1p"), ("reserved", "log1p"),
        ("free", "signed_log1p"), ("headroom", "signed_log1p"),
        ("prior_requested_load", "log1p"), ("degradation", "log1p")]),
    "inventory_state": ("producer", "product", [
        ("inventory_units", "log1p"), ("outstanding_inbound_units", "log1p")]),
    "telemetry_state": ("producer", "product", [
        ("routed_ema", "log1p"), ("completed_ema", "log1p"), ("rejected_ema", "log1p"),
        ("failed_ema", "log1p"), ("completion_latency", "log1p"), ("failure_latency", "log1p")]),
    "pending_bid": ("order", "producer", [
        ("quantity", "log1p"), ("tokens", "log1p"), ("routed_age", "log1p")]),
    "reservation": ("order", "producer", [
        ("quantity", "log1p"), ("tokens", "log1p"), ("cycles_waited", "log1p")]),
}
RELATION_TYPES: list[str] = list(RELATIONS)
# reverse relation registry (ARCH.md): every forward relation has a distinct reverse type.
REVERSE_RELATIONS: dict[str, str] = {r: f"{r}__rev" for r in RELATION_TYPES}

# commitment visibility decision (Phase 0): pending bids + reservations of OTHER orderers
# are part of the deployable observation contract, exposed through the domain-owned
# EconomyGraphView (they exist before the current action -> not future leakage).
COMMITMENT_VISIBILITY = {
    "pending_bid": "visible",
    "reservation": "visible",
    "note": "exposed via domain EconomyGraphView; causal pre-action state, not future leakage",
}


def schema_dict() -> dict:
    """The full graph_primitives_v1 registry as a JSON-able dict (goes to
    graph_schema.json). Feature/relation ORDER is significant — it fixes array columns."""
    return {
        "schema_name": SCHEMA_NAME,
        "node_types": NODE_TYPES,
        "node_features": {t: [list(f) for f in NODE_FEATURES[t]] for t in NODE_TYPES},
        "relations": {r: {"src": s, "dst": d, "attrs": [list(a) for a in attrs]}
                      for r, (s, d, attrs) in RELATIONS.items()},
        "reverse_relations": REVERSE_RELATIONS,
        "commitment_visibility": COMMITMENT_VISIBILITY,
        "dtype": {"features": "float32", "edge_index": "int64"},
    }


def schema_hash() -> str:
    """Stable content hash of the schema registry — stamped into every dump/probe
    spec.json so a probe can never be silently scored against a different observation."""
    blob = json.dumps(schema_dict(), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def sha256_file(path: str) -> str:
    """Full-content SHA-256 of a file (streamed). Fingerprints the dump's INPUTS — archive,
    producer checkpoint, actor checkpoint — so a read can prove the dump was built from the
    pre-registered files, not arbitrary ones of compatible shape (Finding-5.1). A wrong producer
    silently changes teacher labels; a wrong actor silently weakens the licensing floor."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint(commit: "str | None" = None) -> dict:
    """Immutable identity of the code producing a run: SHA-256 over the bytes of EVERY `.py` file in
    the synced `tokenomics` package tree (sorted by relative path), plus an optional git commit
    recorded verbatim (Finding-6.1/7.1). Hashing the whole tree — not a hand-picked module list —
    covers all transitive producers of the trajectory and labels (`marginal_value_labels`,
    `joint_stack_env`, curriculum, actor scoring, domain deps); a four-module subset silently
    ignored changes to them. The hash is git-independent so it is reproducible ON the pod (where
    `.git/` is not synced); the commit is injected by the launcher (which does have git). Stamped
    into both specs AND cross-checked by `_load_dump` before training, so a dump built from source A
    can never earn a verdict under source B."""
    import hashlib as _hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]  # inner tokenomics package dir
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    h = _hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return {"tree_sha256": h.hexdigest(), "n_files": len(files), "commit": commit}


def resolve_verified_commit(tree_dir: "str | None" = None) -> str:
    """Launcher-side binding of the live source tree to a real, clean git revision (Finding-8.1).
    Needs git (NOT available on the pod — resolve LOCALLY and pass the result via `--source-commit`).
    Returns the full 40-char HEAD commit of the repo containing `tree_dir` after verifying: (1) HEAD
    resolves and the object EXISTS (`git cat-file -e`), and (2) the producing tree is CLEAN — no
    modified or untracked files under `tree_dir` (`git status --porcelain`). A clean tree guarantees
    the live `source_fingerprint` corresponds to that commit, so the artifact is recoverable from the
    claimed revision; a forged/short/nonexistent id or a dirty tree raises here rather than producing
    an unrecoverable run. `_validate_commit` (pod-side) only checks shape; this is the real binding."""
    import subprocess
    from pathlib import Path

    d = str(Path(tree_dir).resolve() if tree_dir else Path(__file__).resolve().parents[1])

    def git(*a: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)

    head = git("rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        raise ValueError(f"cannot resolve HEAD in {d}: {head.stderr.strip() or 'not a git repo'}")
    sha = head.stdout.strip()
    if len(sha) != 40 or git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        raise ValueError(f"HEAD {sha!r} is not a full existing commit object")
    dirty = git("status", "--porcelain", "--", d).stdout.strip()
    if dirty:
        raise ValueError(f"producing tree {d} is dirty — commit before launch so the artifact is "
                         f"recoverable from {sha}:\n{dirty}")
    return sha


def _blas_identity() -> dict:
    """Best-effort numeric-backend identity (Finding-9.2). Identical package versions on a different
    CPU arch or BLAS/build are NOT the same numerical environment, so also capture the numpy BLAS name
    and a hash of torch's build config (MKL/OpenBLAS, ISA, threading). Best-effort: None if unavailable."""
    import hashlib

    numpy_blas = torch_build = None
    try:
        import numpy as _np
        cfg = _np.show_config(mode="dicts")  # numpy>=1.26
        numpy_blas = ((cfg.get("Build Dependencies", {}) or {}).get("blas", {}) or {}).get("name")
    except Exception:
        pass
    try:
        import torch as _t
        torch_build = hashlib.sha1(_t.__config__.show().encode()).hexdigest()[:12]
    except Exception:
        pass
    return {"numpy_blas": numpy_blas, "torch_build": torch_build}


def environment_manifest() -> dict:
    """Runtime environment identity for a run (Finding-8.2/9.2; HANDOVER.md:413 requires it in RESULT).
    Deps are open lower bounds (`torch>=2.2`, …) and the pod installs current packages from the
    network, so labels/metrics/determinism depend on versions AND the machine arch + numeric backend,
    which must be captured and compared across the dump and probe stages. `image` records the pod image
    tag/digest when the launcher exports it (audit). Records python/platform/machine + the versions and
    BLAS/build config that actually shape the trajectory and the fit."""
    import os
    import platform
    import sys

    def _ver(mod: str) -> "str | None":
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "implementation": sys.implementation.name,
        "numpy": _ver("numpy"), "torch": _ver("torch"),
        "pettingzoo": _ver("pettingzoo"), "gymnasium": _ver("gymnasium"),
        "image": os.environ.get("EXP0039_POD_IMAGE"),
        **_blas_identity(),
    }


def assert_inputs_match(manifest: dict, expected: dict) -> None:
    """Abort a dump BEFORE it spends teacher-rollout compute if a staged input's SHA-256 does not
    match the pre-registered pin (Finding-6.3). `manifest` is {role: {path, sha256} | None} (as the
    dump records), `expected` is {role: sha256}. A missing role or a hash mismatch raises — so a
    wrong staged checkpoint fails at launch, not after the full ~4 core-hour dump."""
    for role, want in expected.items():
        got = manifest.get(role)
        got_sha = got.get("sha256") if isinstance(got, dict) else None
        if got_sha != want:
            raise ValueError(
                f"input '{role}' sha256 {got_sha} != pinned {want}; refusing to dump from an "
                "unpinned/mismatched input (a wrong producer changes labels, a wrong actor weakens "
                "the floor)")


def decision_digest(task_id, seed, cycle, order_id) -> int:
    """Stable signed-int64 digest of a decision's IDENTITY (task_id, seed, cycle, order_id).
    Stamped per-graph in the shard as an AUDIT array (never a model feature) so a graph payload
    is provably bound to its `decisions.csv` row: `_load_dump` recomputes this from the CSV row
    and rejects any graph whose stored digest disagrees. Components are stringified so the write
    side (ints) and the read side (CSV strings) agree. 0 is reserved as an 'unbound' sentinel."""
    blob = f"{task_id}|{seed}|{cycle}|{order_id}".encode()
    return int.from_bytes(hashlib.sha1(blob).digest()[:8], "big", signed=True)


# --- per-decision numeric arrays ------------------------------------------------------
@dataclass
class GraphArrays:
    """One decision's graph as id-free numeric arrays with contiguous per-type local
    indices. `id_maps` (local index -> raw domain id per node type) is audit-only and is
    NEVER serialized into a model feature tensor (ARCH.md identity isolation)."""

    node_features: dict[str, np.ndarray]          # node_type -> [n_nodes, n_feats] f32
    edge_index: dict[str, np.ndarray]             # relation -> [2, n_edges] int64 (local)
    edge_attr: dict[str, np.ndarray]              # relation -> [n_edges, n_attrs] f32
    current_order: int                            # order local index
    current_orderer: int                          # orderer local index
    target_product: int                           # current order's product local index
    candidate_producers: np.ndarray               # [n_candidates] producer local indices
    id_maps: dict[str, list] = field(default_factory=dict)  # audit only
    decision_digest: int = 0                       # audit only: binds this graph to its
    #                                                decisions.csv row (0 = unbound). NEVER a
    #                                                model feature — persisted as a separate
    #                                                shard array (ARCH.md identity isolation).


def _transform(value: float, kind: str) -> float:
    if kind == "log1p":
        return float(np.log1p(float(value)))
    if kind == "signed_log1p":
        # sign-preserving log compression for SIGNABLE counts (e.g. free capacity =
        # current - reserved can be negative when a prior-cycle reservation outlives
        # degraded capacity). Finite for all reals, unlike log1p(x<0).
        v = float(value)
        return float(np.sign(v) * np.log1p(abs(v)))
    return float(value)  # binary / one-hot pass through


def _order_row(n) -> dict:
    kc, kr, ki = n.order_kind
    return {
        "original_quantity": n.original_quantity, "unresolved_quantity": n.unresolved_quantity,
        "accepted_quantity": n.accepted_quantity, "completed_quantity": n.completed_quantity,
        "failed_quantity": n.failed_quantity, "available_tokens": n.available_tokens,
        "burnt_tokens": n.burnt_tokens, "order_age": n.order_age, "ttl_remaining": n.ttl_remaining,
        "funded": n.funded, "kind_consumer": kc, "kind_replenishment": kr, "kind_investment": ki,
    }


def _orderer_row(n) -> dict:
    sc, sp = n.source_role
    return {"source_consumer": sc, "source_producer": sp}


def _producer_row(n) -> dict:
    return {
        "cut_off_price": n.cut_off_price, "cutoff_is_defined": n.cutoff_is_defined,
        "funds": n.funds, "inventory_limit": n.inventory_limit,
        "upgrade_running": n.upgrade_running, "restoration_running": n.restoration_running,
        "inventory_expansion_running": n.inventory_expansion_running,
        "has_inventory_expansion": n.has_inventory_expansion,
    }


def _product_row(n) -> dict:
    return {"demand_ema": n.demand_ema, "rejected_demand_ema": n.rejected_demand_ema}


# per relation: (view attribute, src-id getter, dst-id getter, attr getter -> dict)
_REL_SPEC = {
    "current_order": (None, None, None, None),   # anchors, filled specially
    "current_orderer": (None, None, None, None),
    "candidate": (None, None, None, None),
    "owns": ("owns", lambda e: e.orderer_id, lambda e: e.order_id, lambda e: {}),
    "stream_for": ("stream_for", lambda e: e.orderer_id, lambda e: e.product, lambda e: {}),
    "source_of": ("source_of", lambda e: e.producer_id, lambda e: e.orderer_id, lambda e: {}),
    "requests": ("requests", lambda e: e.order_id, lambda e: e.product, lambda e: {}),
    "can_make": ("can_make", lambda e: e.producer_id, lambda e: e.product, lambda e: {}),
    "requires_capacity": ("requires_capacity", lambda e: e.product, lambda e: e.capacity_type,
                          lambda e: {"units": e.units}),
    "requires_input": ("requires_input", lambda e: e.product, lambda e: e.input_product,
                       lambda e: {"units": e.units}),
    "capacity_state": ("capacity_state", lambda e: e.producer_id, lambda e: e.capacity_type,
                       lambda e: {"current": e.current, "max": e.max, "reserved": e.reserved,
                                  "free": e.free, "headroom": e.headroom,
                                  "prior_requested_load": e.prior_requested_load,
                                  "degradation": e.degradation}),
    "inventory_state": ("inventory_state", lambda e: e.producer_id, lambda e: e.input_product,
                        lambda e: {"inventory_units": e.inventory_units,
                                   "outstanding_inbound_units": e.outstanding_inbound_units}),
    "telemetry_state": ("telemetry_state", lambda e: e.producer_id, lambda e: e.product,
                        lambda e: {"routed_ema": e.routed_ema, "completed_ema": e.completed_ema,
                                   "rejected_ema": e.rejected_ema, "failed_ema": e.failed_ema,
                                   "completion_latency": e.completion_latency,
                                   "failure_latency": e.failure_latency}),
    "pending_bid": ("pending_bid", lambda e: e.order_id, lambda e: e.producer_id,
                    lambda e: {"quantity": e.quantity, "tokens": e.tokens, "routed_age": e.routed_age}),
    "reservation": ("reservation", lambda e: e.order_id, lambda e: e.producer_id,
                    lambda e: {"quantity": e.quantity, "tokens": e.tokens,
                               "cycles_waited": e.cycles_waited}),
}


def _assert_finite(g: GraphArrays) -> None:
    """Fail loudly if any model tensor holds NaN/inf. Cheap insurance against a mis-declared
    transform (e.g. log1p on a signable field) silently shipping poison to a pod run."""
    for t, arr in g.node_features.items():
        if arr.size and not np.isfinite(arr).all():
            raise ValueError(f"non-finite node feature in '{t}'")
    for r, arr in g.edge_attr.items():
        if arr.size and not np.isfinite(arr).all():
            raise ValueError(f"non-finite edge attr in relation '{r}'")


def encode_graph(view: EconomyGraphView) -> GraphArrays:
    """Encode one decision view into id-free numeric arrays. Node local indices follow the
    view's node-list order (deterministic); candidate producer indices align one-to-one
    with the eligible set. log1p/binary transforms are applied here; standardization is
    fold-local in the probe."""
    # local index maps (id -> local), and audit id maps (local -> id)
    order_ids = [n.order_id for n in view.orders]
    orderer_ids = [n.orderer_id for n in view.orderers]
    producer_ids = [n.producer_id for n in view.producers]
    product_ids = [n.product for n in view.products]
    captype_ids = [n.capacity_type for n in view.capacity_types]
    idx = {
        "order": {v: i for i, v in enumerate(order_ids)},
        "orderer": {v: i for i, v in enumerate(orderer_ids)},
        "producer": {v: i for i, v in enumerate(producer_ids)},
        "product": {v: i for i, v in enumerate(product_ids)},
        "capacity_type": {v: i for i, v in enumerate(captype_ids)},
    }

    def _feat(rows: list, node_type: str) -> np.ndarray:
        spec = NODE_FEATURES[node_type]
        if not spec:
            return np.zeros((len(rows), 0), dtype=np.float32)
        return np.array(
            [[_transform(r[name], kind) for name, kind in spec] for r in rows],
            dtype=np.float32).reshape(len(rows), len(spec))

    node_features = {
        "decision": np.zeros((1, 0), dtype=np.float32),
        "order": _feat([_order_row(n) for n in view.orders], "order"),
        "orderer": _feat([_orderer_row(n) for n in view.orderers], "orderer"),
        "producer": _feat([_producer_row(n) for n in view.producers], "producer"),
        "product": _feat([_product_row(n) for n in view.products], "product"),
        "capacity_type": np.zeros((len(captype_ids), 0), dtype=np.float32),
    }

    edge_index: dict[str, np.ndarray] = {}
    edge_attr: dict[str, np.ndarray] = {}
    for rel, (src_t, dst_t, attr_spec) in RELATIONS.items():
        if rel in ("current_order", "current_orderer", "candidate"):
            continue  # anchor edges are derived from the anchor fields, below
        view_attr, get_src, get_dst, get_attrs = _REL_SPEC[rel]
        edges = getattr(view, view_attr)
        src = idx[src_t]
        dst = idx[dst_t]
        pairs = [(src[get_src(e)], dst[get_dst(e)]) for e in edges]
        edge_index[rel] = (np.array(pairs, dtype=np.int64).T.reshape(2, len(pairs)))
        if attr_spec:
            edge_attr[rel] = np.array(
                [[_transform(get_attrs(e)[name], kind) for name, kind in attr_spec] for e in edges],
                dtype=np.float32).reshape(len(edges), len(attr_spec))
        else:
            edge_attr[rel] = np.zeros((len(edges), 0), dtype=np.float32)

    # anchors + candidate edges (candidate = decision -> each eligible producer)
    current_order = idx["order"][view.current_order.order_id]
    current_orderer = idx["orderer"][view.current_orderer.orderer_id]
    cur_product = {e.order_id: e.product for e in view.requests}[view.current_order.order_id]
    target_product = idx["product"][cur_product]
    candidate_producers = np.array(
        [idx["producer"][e.producer_id] for e in view.candidate], dtype=np.int64)
    # materialize the anchor edges too, so the graph is self-contained for the model
    edge_index["current_order"] = np.array([[0], [current_order]], dtype=np.int64)
    edge_attr["current_order"] = np.zeros((1, 0), dtype=np.float32)
    edge_index["current_orderer"] = np.array([[0], [current_orderer]], dtype=np.int64)
    edge_attr["current_orderer"] = np.zeros((1, 0), dtype=np.float32)
    edge_index["candidate"] = np.array(
        [[0] * len(candidate_producers), candidate_producers.tolist()], dtype=np.int64
    ).reshape(2, len(candidate_producers))
    edge_attr["candidate"] = np.zeros((len(candidate_producers), 0), dtype=np.float32)

    return GraphArrays(
        node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
        current_order=current_order, current_orderer=current_orderer,
        target_product=target_product, candidate_producers=candidate_producers,
        id_maps={"order": order_ids, "orderer": orderer_ids, "producer": producer_ids,
                 "product": product_ids, "capacity_type": captype_ids},
    )


# --- pickle-free shard serialization --------------------------------------------------
# One shard packs many graphs by CONCATENATING per-type node features and per-relation
# edges, with CSR-style offset pointers so neighboring graphs never mix. Edge indices stay
# LOCAL to their graph (sliced back out by the pointers), so no global re-indexing is
# needed and a graph reconstructs identically regardless of shard load order.
def write_shard(path: str, graphs: list[GraphArrays]) -> None:
    """Serialize `graphs` to one `.npz` shard with explicit node/edge/candidate offsets.
    No Python pickle — every stored array is numeric."""
    out: dict[str, np.ndarray] = {"n_graphs": np.array([len(graphs)], dtype=np.int64)}
    for t in NODE_TYPES:
        feats = [g.node_features[t] for g in graphs]
        width = feats[0].shape[1] if feats else len(NODE_FEATURES[t])
        out[f"nodefeat__{t}"] = (np.concatenate(feats, axis=0) if feats
                                 else np.zeros((0, width), np.float32)).astype(np.float32)
        out[f"nodeptr__{t}"] = np.cumsum([0] + [f.shape[0] for f in feats]).astype(np.int64)
    for r, (_s, _d, attrs) in RELATIONS.items():
        eidx = [g.edge_index[r] for g in graphs]
        eatt = [g.edge_attr[r] for g in graphs]
        out[f"edgeidx__{r}"] = (np.concatenate(eidx, axis=1) if eidx
                                else np.zeros((2, 0), np.int64)).astype(np.int64)
        out[f"edgeattr__{r}"] = (np.concatenate(eatt, axis=0) if eatt
                                 else np.zeros((0, len(attrs)), np.float32)).astype(np.float32)
        out[f"edgeptr__{r}"] = np.cumsum([0] + [e.shape[1] for e in eidx]).astype(np.int64)
    out["current_order"] = np.array([g.current_order for g in graphs], dtype=np.int64)
    out["current_orderer"] = np.array([g.current_orderer for g in graphs], dtype=np.int64)
    out["target_product"] = np.array([g.target_product for g in graphs], dtype=np.int64)
    cand = [g.candidate_producers for g in graphs]
    out["candidate"] = (np.concatenate(cand) if cand else np.zeros((0,), np.int64)).astype(np.int64)
    out["candidate_ptr"] = np.cumsum([0] + [c.shape[0] for c in cand]).astype(np.int64)
    # per-graph decision digest (audit binding to decisions.csv row) — a numeric array, not a
    # model tensor. Travels WITH the graph payload, so swapping payloads mismatches its position.
    out["decision_digest"] = np.array([int(g.decision_digest) for g in graphs], dtype=np.int64)
    with open(path, "wb") as f:
        np.savez(f, **out)


def load_shard(path: str) -> list[GraphArrays]:
    """Reconstruct every graph in a shard from numeric arrays + offsets (allow_pickle
    False). Inverse of `write_shard`; graphs come back byte-for-byte identical."""
    z = np.load(path, allow_pickle=False)
    n = int(z["n_graphs"][0])
    graphs: list[GraphArrays] = []
    for i in range(n):
        node_features = {}
        for t in NODE_TYPES:
            p = z[f"nodeptr__{t}"]
            node_features[t] = z[f"nodefeat__{t}"][p[i]:p[i + 1]]
        edge_index, edge_attr = {}, {}
        for r in RELATION_TYPES:
            p = z[f"edgeptr__{r}"]
            edge_index[r] = z[f"edgeidx__{r}"][:, p[i]:p[i + 1]]
            edge_attr[r] = z[f"edgeattr__{r}"][p[i]:p[i + 1]]
        cp = z["candidate_ptr"]
        digest = int(z["decision_digest"][i]) if "decision_digest" in z.files else 0
        graphs.append(GraphArrays(
            node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
            current_order=int(z["current_order"][i]),
            current_orderer=int(z["current_orderer"][i]),
            target_product=int(z["target_product"][i]),
            candidate_producers=z["candidate"][cp[i]:cp[i + 1]],
            decision_digest=digest,
        ))
    return graphs


# --- trajectory dump ------------------------------------------------------------------
def run_graph_dump(
    records: "Sequence[TaskRecord]", *, prod: "Policy", base: "Policy | None" = None,
    model: "ResidualScoreModel | None" = None, delta: float = 1.0, lookahead: int = 8,
    eval_seeds: int = 3, horizon: int = 100, out_dir: str, shard_size: int = 2000,
    input_manifest: "dict | None" = None, source_commit: "str | None" = None,
    log: "Callable[..., None]" = print,
) -> dict:
    """Drive the fixed blend_0.5 trajectory and materialize ONE graph per contested routing
    decision (>= 2 eligible producers), in the exact EXP-0038 traversal order so
    `decisions.csv` is byte-identical (same `decision_row`) and the graph shard order matches
    it row-for-row. For each decision the graph is extracted BEFORE the teacher call and
    BEFORE `env.step` (causality), then labeled with the unchanged `marginal_value_labels`.

    Writes the ARCH.md durable dump: `decisions.csv`, `candidates.csv` (per-candidate label +
    evaluation-only actor score), `graph_schema.json`, `topologies.jsonl`, chunked
    `graph_shard_*.npz`, and `spec.json`. `model` (the EXP-0037 residual actor) fills the actor
    floor; without it actor scores are zero (smoke). Labels/actor scores live only in the CSVs,
    never in a graph tensor."""
    from ..domain.order import ORDER_TTL
    from ..train.value_router import producer_row_features
    from .baselines import ordering_blend
    from .curriculum import _episode_seeds, replay
    from .joint_stack import joint_stack_env
    from .value_router import marginal_value_labels
    from .value_router_errors import ErrorRecord, _actor_scores, _write_csv, decision_row

    base = base if base is not None else ordering_blend(0.5)
    recs: list[ErrorRecord] = []
    graphs: list[GraphArrays] = []
    cand_rows: list[dict] = []
    topologies: dict[str, dict] = {}

    for rec in records:
        cfg = replay(rec)
        for s in _episode_seeds(rec.episode_seed, eval_seeds):
            env = joint_stack_env(cfg, horizon)
            env.reset(seed=s)
            for agent in env.agent_iter(max_iter=env.horizon * len(env.possible_agents) * 3 + 100):
                if env.terminations[agent] or env.truncations[agent]:
                    env.step(None)
                    continue
                role, aid = env._role[agent]
                if role == "prod":
                    env.step(prod(env, agent))
                    continue
                view = env.system.ordering_agents[aid].view(env.system.producer_infos)
                incoming = sorted(view.incoming)
                for k, oid in enumerate(incoming):
                    if k >= env.spec_.k_stream:
                        break
                    elig = view.eligible[oid]
                    if len(elig) < 2:
                        continue
                    # extract graph FIRST (before teacher, before step) -> causal snapshot
                    g = encode_graph(build_economy_graph_view(env.system, aid, oid, list(elig)))
                    _assert_finite(g)  # no NaN/inf feature ever reaches a shard (blocker guard)
                    cols = [env.spec_.producer_ids.index(p) for p in elig]
                    labels = marginal_value_labels(env, prod, base, agent, k, cols, delta, lookahead)
                    rows = producer_row_features(env, oid, view)
                    mask = np.array([pid in elig for pid in env.spec_.producer_ids])
                    scores = (_actor_scores(model, rows, mask) if model is not None
                              else np.zeros(env.spec_.num_producers, dtype=np.float32))
                    oi = view.incoming[oid]
                    order = env.system.orders.get(oid)
                    age = order.cycle_counter if order is not None else -1
                    ttl = max(0, ORDER_TTL - age) if age >= 0 else -1
                    same_prod = sum(1 for o in incoming if view.incoming[o].product == oi.product)
                    rec_ = ErrorRecord(
                        task_id=rec.task_id, seed=s, cycle=env.current_cycle, agent=agent,
                        order_id=oid, stream_index=k, product=oi.product, quantity=oi.quantity,
                        order_age=age, ttl_remaining=ttl, backlog_size=len(incoming),
                        same_product_backlog=same_prod, eligible_cols=cols,
                        producer_ids=list(env.spec_.producer_ids), labels=labels, scores=scores,
                        rows=rows, full_rows=None)
                    di = len(recs)
                    # bind this graph payload to its decisions.csv row identity (audit-only)
                    g.decision_digest = decision_digest(rec.task_id, s, env.current_cycle, oid)
                    recs.append(rec_)
                    graphs.append(g)
                    for ci, col in enumerate(cols):
                        cand_rows.append({
                            "decision_index": di, "task_id": rec.task_id, "seed": s,
                            "cycle": env.current_cycle, "order_id": oid, "candidate_index": ci,
                            "producer_id": env.spec_.producer_ids[col],
                            "teacher_label": round(float(labels[col]), 6),
                            "actor_score": round(float(scores[col]), 6)})
                    topologies.setdefault(rec.task_id, {
                        "task_id": rec.task_id,
                        "producer_ids": g.id_maps["producer"],
                        "product_ids": g.id_maps["product"],
                        "capacity_type_ids": g.id_maps["capacity_type"],
                        "orderer_ids": g.id_maps["orderer"]})
                env.step(base(env, agent))
            log(f"[graph-dump] {rec.task_id} seed {s}: {len(recs)} decisions")

    os.makedirs(out_dir, exist_ok=True)
    d_rows = [decision_row(r) for r in recs]
    _write_csv(os.path.join(out_dir, "decisions.csv"), d_rows)
    _write_csv(os.path.join(out_dir, "candidates.csv"), cand_rows)
    with open(os.path.join(out_dir, "topologies.jsonl"), "w") as f:
        for t in topologies.values():
            f.write(json.dumps(t) + "\n")
    with open(os.path.join(out_dir, "graph_schema.json"), "w") as f:
        json.dump({**schema_dict(), "schema_hash": schema_hash()}, f, indent=2)
    n_shards = 0
    for start in range(0, len(graphs), shard_size):
        path = os.path.join(out_dir, f"graph_shard_{n_shards:03d}.npz")
        write_shard(path, graphs[start:start + shard_size])
        n_shards += 1
    # exact anchored task->seed sets actually traversed (Finding-5.2): a validator can enforce the
    # pre-registered sets (e.g. s29={29,30,31}), not merely "three seeds".
    task_seeds: dict[str, list[int]] = {}
    for r in recs:
        task_seeds.setdefault(r.task_id, set()).add(int(r.seed))  # type: ignore[attr-defined]
    task_seeds = {t: sorted(s) for t, s in task_seeds.items()}
    spec = {
        "schema_name": SCHEMA_NAME, "schema_hash": schema_hash(),
        "task_ids": [r.task_id for r in records], "delta": delta, "lookahead": lookahead,
        "eval_seeds": eval_seeds, "horizon": horizon, "floor": "blend_0.5",
        "has_actor": model is not None, "n_decisions": len(recs),
        "n_candidates": len(cand_rows), "n_shards": n_shards, "shard_size": shard_size,
        "task_seeds": task_seeds,
        # immutable input fingerprints (archive + producer/actor ckpt SHA-256); None in a smoke
        # that runs a baseline producer and no actor. Populated by main().
        "inputs": input_manifest,
        # immutable identity of the code that produced this dump (module bytes + optional commit),
        # so a later training/metric/verdict change is detectable even though it leaves the schema
        # hash unchanged and `.git/` is not on the pod (Finding-6.1).
        "source": source_fingerprint(source_commit),
        # runtime environment (python/platform + numpy/torch/pettingzoo/gymnasium versions) — deps are
        # open lower bounds installed fresh on the pod, so the read compares dump vs probe env (Finding-8.2).
        "env": environment_manifest(),
    }
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    log(f"{len(recs)} decisions, {len(cand_rows)} candidates, {n_shards} shards -> {out_dir}")
    return spec


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, help="task archive jsonl")
    ap.add_argument("--task-ids", required=True, help="comma-separated gate task ids (s29,s77,s45)")
    ap.add_argument("--prod", default="stock_aware", help="producer baseline held fixed")
    ap.add_argument("--prod-ckpt", default=None, help="producer scorer checkpoint (EXP-0035)")
    ap.add_argument("--actor-ckpt", default=None,
                    help="EXP-0037 residual actor for the actor floor; omit -> zero actor scores")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-commit", default=None,
                    help="full 40-char git commit of the source tree producing this dump, recorded in "
                         "spec.json (the pod has no .git; resolve LOCALLY via resolve_verified_commit "
                         "and pass it). Omit + --verify-source-commit to auto-resolve when git is present")
    ap.add_argument("--verify-source-commit", action="store_true",
                    help="resolve+verify the commit from the LOCAL git tree (existence + clean tree) "
                         "and use it — the launcher-side binding (Finding-8.1); needs git")
    ap.add_argument("--require-gate-inputs", action="store_true",
                    help="validate archive/producer/actor SHA-256 against the pinned EXP-0039 gate "
                         "inputs BEFORE dumping (fail a wrong staged input at launch, not after)")
    args = ap.parse_args()

    from ..train.producer_scorer import load_producer_score_model, scorer_producing_policy
    from .baselines import PRODUCING_BASELINES
    from .curriculum_replay import _load_archive

    by_id = {r.task_id: r for r in _load_archive(args.archive)}
    records = [by_id[t] for t in args.task_ids.split(",")]
    prod = (scorer_producing_policy(load_producer_score_model(args.prod_ckpt))
            if args.prod_ckpt else PRODUCING_BASELINES[args.prod])
    model = None
    if args.actor_ckpt:
        import torch

        from ..train.value_router import ResidualScoreModel
        ckpt = torch.load(args.actor_ckpt, map_location="cpu")
        model = ResidualScoreModel()
        model.load_state_dict(ckpt["state_dict"])

    # fingerprint the actual inputs so the read can bind to the pre-registered files (Finding-5.1)
    def _fp(path: "str | None") -> "dict | None":
        return {"path": path, "sha256": sha256_file(path)} if path else None

    input_manifest = {"archive": _fp(args.archive), "producer_ckpt": _fp(args.prod_ckpt),
                      "actor_ckpt": _fp(args.actor_ckpt)}

    # fail a wrong staged input at launch (before ~4 core-hours of teacher rollout), not when the
    # probe later loads the finished dump (Finding-6.3).
    if args.require_gate_inputs:
        from ..train.graph_router_probe import GATE_INPUT_SHA256
        assert_inputs_match(input_manifest, GATE_INPUT_SHA256)

    # bind to a real, clean local revision (Finding-8.1) when git is present; else validate the shape
    # of the passed commit BEFORE spending the dump budget (a missing/malformed commit fails at launch).
    source_commit = args.source_commit
    if args.verify_source_commit:
        source_commit = resolve_verified_commit()
    if args.require_gate_inputs:
        from ..train.graph_router_probe import _validate_commit
        _validate_commit(source_commit)

    run_graph_dump(records, prod=prod, model=model, delta=args.delta, lookahead=args.lookahead,
                   eval_seeds=args.eval_seeds, horizon=args.horizon, out_dir=args.out_dir,
                   shard_size=args.shard_size, input_manifest=input_manifest,
                   source_commit=source_commit)


if __name__ == "__main__":
    main()
