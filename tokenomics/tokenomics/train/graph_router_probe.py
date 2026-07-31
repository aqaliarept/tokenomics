"""EXP-0039 pre-registered relational GNN probe over graph_primitives_v1.

A minimal pure-Torch relational message-passing model (ARCH.md allows this over PyG): four
message-passing layers, hidden 64, learned schema-ROLE embeddings only (node-type and
relation-type — never per-producer/product/capacity identity embeddings), type-specific node
encoders, relation-specific edge encoders + message transforms, SUM aggregation, residual +
layer-norm updates, and one shared candidate head.

The probe answers the EXP-0039 question — is the EXP-0037 marginal-value teacher's within-
decision producer preference representable from relational deployable state? It is scored by
the corrected EXP-0038A/B gate (leave-one-seed-out, decision-weighted, tie-aware), reusing
`observation_ladder` metrics so the read matches EXP-0038B. NOT an orderer: a pass only
licenses a later deployed-GNN experiment.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..eval.graph_value_router_dump import (
    GraphArrays,
    NODE_FEATURES,
    NODE_TYPES,
    RELATIONS,
    RELATION_TYPES,
)

HIDDEN = 64
LAYERS = 4

# Pre-registered EXP-0039 gate configuration (spec.json §params / HANDOVER.md §Inputs). The real
# dump MUST be produced on exactly this trajectory + teacher: a wrong delta/lookahead/task set or
# a truncated horizon preserves decisions.csv counts while silently changing every teacher label,
# so the loader validates the dump's own spec.json against this before any compute (Finding-4.2).
GATE_TASK_IDS: tuple[str, ...] = (
    "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-"
    "producers_per_layer2-quantity_max14-s29",
    "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-"
    "producers_per_layer3-quantity_max14-s77",
    "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-"
    "producers_per_layer3-quantity_max14-s45",
)
GATE_CONFIG: dict = {
    "schema_name": "graph_primitives_v1", "delta": 1.0, "lookahead": 8, "horizon": 100,
    "eval_seeds": 3, "floor": "blend_0.5", "has_actor": True, "task_ids": list(GATE_TASK_IDS),
}
# EXACT anchored task->seed sets (Finding-5.2): the seeds are `range(episode_seed, +3)` and the
# task id encodes the base (…-s29 -> {29,30,31}); a count check alone accepts any three values.
GATE_TASK_SEEDS: dict = {
    GATE_TASK_IDS[0]: {29, 30, 31},
    GATE_TASK_IDS[1]: {77, 78, 79},
    GATE_TASK_IDS[2]: {45, 46, 47},
}
# Immutable SHA-256 fingerprints of the pre-registered inputs (Finding-5.1; spec.json §params). A
# wrong producer silently changes teacher labels, a wrong actor silently weakens the licensing floor
# — a name/shape config cannot distinguish them, so the dump's recorded input hashes are validated
# against these. Re-pin deliberately if a frozen input is ever regenerated.
GATE_INPUT_SHA256: dict = {
    "archive": "bc6ca4ebacff6835571453cf7c0bf10de5c366cc208674f44cf3a5860b778ceb",
    "producer_ckpt": "8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea",
    "actor_ckpt": "666533e4e6beaa845437e4c7bba09dcb8c3824ec3c88015d72c1aefb23ee96f1",
}

# Canonical EXP-0038B deployed-actor floor (Finding-6.2). The `--actor-floor-json` fallback (used
# only when the checked-in EXP-0037 residual_scorer.pt is pre-P2) must NOT accept arbitrary values:
# an inflated regret would silently make the licensing bar easier to clear. The authoritative floor
# is EXP-0038B/audit_sys/spec.json `floors.actor[task].mean_regret`, pinned by SHA-256; any override
# is validated against these exact values. `canonical_actor_floor` derives them mechanically.
GATE_FLOOR_ARTIFACT: str = str(
    Path(__file__).resolve().parents[3]
    / "experiments/EXP-0038B-producer-product-observation-ladder/audit_sys/spec.json")
GATE_FLOOR_SHA256: str = "44d6d28f525a4ef5b45c6e77616a5195d94cd5f38860a57a4bb8c86379114b14"
GATE_ACTOR_FLOOR_0038B: dict = {
    GATE_TASK_IDS[0]: 4.120065789473684,   # s29
    GATE_TASK_IDS[1]: 4.754183496826313,   # s77
    GATE_TASK_IDS[2]: 3.9346504559270516,  # s45
}


def canonical_actor_floor(artifact_path: str = GATE_FLOOR_ARTIFACT, *,
                          expected_sha256: str = GATE_FLOOR_SHA256) -> dict:
    """Derive the EXP-0038B deployed-actor floor {task_id: mean_regret} MECHANICALLY from the
    canonical artifact, validating its SHA-256 first (Finding-6.2). Because the file's bytes are
    pinned, the returned values cannot be tampered — this is the safe way to produce the
    `--actor-floor-json` fallback. Cross-checks the derived values against the pinned constant."""
    import json

    from ..eval.graph_value_router_dump import sha256_file

    sha = sha256_file(artifact_path)
    if expected_sha256 and sha != expected_sha256:
        raise ValueError(f"canonical floor artifact sha256 {sha} != pinned {expected_sha256}")
    actor = json.load(open(artifact_path))["floors"]["actor"]
    floor = {t: float(actor[t]["mean_regret"]) for t in GATE_TASK_IDS}
    assert_override_is_canonical(floor)
    return floor


def assert_override_is_canonical(override: dict) -> None:
    """Reject an actor-floor override that is not the pinned EXP-0038B floor (Finding-6.2). The
    verdict compares GNN regret < floor regret, so ANY floor larger than canonical weakens the gate;
    a tolerance would let a sub-epsilon inflation flip a strict `<` (Finding-7.2), so the check is
    EXACT equality (canonical floats round-trip exactly from repr/JSON). run_probe additionally
    substitutes the pinned constants for the verdict. Smoke passes require_shape=False and skips."""
    want = GATE_ACTOR_FLOOR_0038B
    if set(override) != set(want):
        raise ValueError(f"actor-floor override tasks {sorted(override)} != canonical "
                         f"{sorted(want)} (must be the EXP-0038B gate tasks)")
    for t, v in override.items():
        if float(v) != want[t]:
            raise ValueError(
                f"actor-floor override[{t}]={v!r} != canonical EXP-0038B floor {want[t]!r}; any "
                "mismatch (even sub-epsilon) re-scales the licensing bar (use canonical_actor_floor)")


# --- fold-local standardization (ARCH.md preprocessing) -------------------------------
class FoldNormalizer:
    """Fit mean/std on the training fold's CONTINUOUS columns (log1p / signed_log1p; binary
    and one-hot columns are excluded and pass through) and apply the SAME stats to training
    and held-out graphs. Fitting is per fold — never per graph or per task snapshot — so real
    demand/capacity scale is preserved across the batch (ARCH.md §Feature Normalization)."""

    def __init__(self, node_stats: dict, edge_stats: dict) -> None:
        self.node_stats = node_stats  # node_type -> (col_idx[], mean[], std[])
        self.edge_stats = edge_stats  # relation  -> (col_idx[], mean[], std[])

    @staticmethod
    def _cont_cols(spec: Sequence[tuple[str, str]]) -> list[int]:
        return [i for i, (_n, t) in enumerate(spec) if t != "binary"]

    @staticmethod
    def _fit_group(mats: list[np.ndarray], cols: list[int]) -> tuple:
        rows = [m[:, cols] for m in mats if m.shape[0]]
        if rows:
            stacked = np.concatenate(rows, axis=0)
            mean, std = stacked.mean(axis=0), stacked.std(axis=0)
        else:  # no rows in this fold — identity stats
            mean, std = np.zeros(len(cols)), np.ones(len(cols))
        return (np.asarray(cols, dtype=np.int64), mean.astype(np.float32),
                np.clip(std, 1e-6, None).astype(np.float32))

    @classmethod
    def fit(cls, graphs: Sequence[GraphArrays]) -> "FoldNormalizer":
        node_stats, edge_stats = {}, {}
        for t in NODE_TYPES:
            cols = cls._cont_cols(NODE_FEATURES[t])
            if cols:
                node_stats[t] = cls._fit_group([g.node_features[t] for g in graphs], cols)
        for r in RELATION_TYPES:
            cols = cls._cont_cols(RELATIONS[r][2])
            if cols:
                edge_stats[r] = cls._fit_group([g.edge_attr[r] for g in graphs], cols)
        return cls(node_stats, edge_stats)

    def apply(self, graphs: Sequence[GraphArrays]) -> list[GraphArrays]:
        out: list[GraphArrays] = []
        for g in graphs:
            nf = {t: arr.copy() for t, arr in g.node_features.items()}
            for t, (cols, mean, std) in self.node_stats.items():
                if nf[t].shape[0]:
                    nf[t][:, cols] = (nf[t][:, cols] - mean) / std
            ea = {r: arr.copy() for r, arr in g.edge_attr.items()}
            for r, (cols, mean, std) in self.edge_stats.items():
                if ea[r].shape[0]:
                    ea[r][:, cols] = (ea[r][:, cols] - mean) / std
            out.append(GraphArrays(
                node_features=nf, edge_index=g.edge_index, edge_attr=ea,
                current_order=g.current_order, current_orderer=g.current_orderer,
                target_product=g.target_product, candidate_producers=g.candidate_producers,
                id_maps=g.id_maps))
        return out

    def groups(self) -> dict:
        """Audit view (ARCH.md artifact contract §normalization group keys): the fitted
        mean/std per (node_type|relation, continuous column) so a probe read is reproducible."""
        def _dump(stats: dict) -> dict:
            return {k: {"cols": cols.tolist(), "mean": mean.tolist(), "std": std.tolist()}
                    for k, (cols, mean, std) in stats.items()}
        return {"node": _dump(self.node_stats), "edge": _dump(self.edge_stats)}


# --- batching -------------------------------------------------------------------------
class Batch:
    """A batch of graphs concatenated per node type with running offsets, edges remapped to
    the batched index space. `decision` has one node per graph (its index == graph id), so
    candidate rows carry the batched decision/order/product/producer indices they score
    from. Pure index bookkeeping — no model parameter depends on graph size."""

    def __init__(self, graphs: Sequence[GraphArrays]) -> None:
        self.n_graphs = len(graphs)
        # per-type node offsets
        off = {t: np.cumsum([0] + [g.node_features[t].shape[0] for g in graphs]) for t in NODE_TYPES}
        self.x = {t: torch.tensor(
            np.concatenate([g.node_features[t] for g in graphs], axis=0)
            if graphs else np.zeros((0, len(NODE_FEATURES[t])), np.float32), dtype=torch.float32)
            for t in NODE_TYPES}
        self.edges: dict[str, tuple] = {}
        for r in RELATION_TYPES:
            src_t, dst_t, _a = RELATIONS[r]
            srcs, dsts, attrs = [], [], []
            for i, g in enumerate(graphs):
                ei = g.edge_index[r]
                if ei.shape[1]:
                    srcs.append(ei[0] + off[src_t][i])
                    dsts.append(ei[1] + off[dst_t][i])
                    attrs.append(g.edge_attr[r])
            src = torch.tensor(np.concatenate(srcs) if srcs else np.zeros((0,), np.int64), dtype=torch.long)
            dst = torch.tensor(np.concatenate(dsts) if dsts else np.zeros((0,), np.int64), dtype=torch.long)
            attr = torch.tensor(
                np.concatenate(attrs, axis=0) if attrs else np.zeros((0, len(_a)), np.float32),
                dtype=torch.float32)
            self.edges[r] = (src, dst, attr)
        # candidate anchors (batched indices), one row per (graph, eligible producer)
        cd, co, cp, cprod, cg = [], [], [], [], []
        for i, g in enumerate(graphs):
            for local in g.candidate_producers:
                cd.append(i)                                    # decision node index == graph id
                co.append(off["order"][i] + g.current_order)
                cp.append(off["product"][i] + g.target_product)
                cprod.append(off["producer"][i] + int(local))
                cg.append(i)
        self.cand_decision = torch.tensor(cd, dtype=torch.long)
        self.cand_order = torch.tensor(co, dtype=torch.long)
        self.cand_product = torch.tensor(cp, dtype=torch.long)
        self.cand_producer = torch.tensor(cprod, dtype=torch.long)
        self.cand_graph = torch.tensor(cg, dtype=torch.long)
        self.node_counts = {t: int(off[t][-1]) for t in NODE_TYPES}

    def to(self, device: "torch.device | str") -> "Batch":
        """Move every batched tensor onto `device` (so a GPU pod accelerates the probe).
        In place; returns self for chaining."""
        self.x = {t: v.to(device) for t, v in self.x.items()}
        self.edges = {r: tuple(t.to(device) for t in trip) for r, trip in self.edges.items()}
        for a in ("cand_decision", "cand_order", "cand_product", "cand_producer", "cand_graph"):
            setattr(self, a, getattr(self, a).to(device))
        return self


# --- model ----------------------------------------------------------------------------
class RelationalGNN(nn.Module):
    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        # schema-role embeddings: one per node type (initial state) and per relation
        # (forward + reverse). NOT per-identity — no producer/product/capacity embedding.
        self.node_role = nn.ParameterDict(
            {t: nn.Parameter(torch.randn(hidden) * 0.1) for t in NODE_TYPES})
        self.node_enc = nn.ModuleDict(
            {t: nn.Linear(len(NODE_FEATURES[t]), hidden) for t in NODE_TYPES if NODE_FEATURES[t]})
        self.edge_enc = nn.ModuleDict(
            {r: nn.Linear(len(RELATIONS[r][2]), hidden) for r in RELATION_TYPES if RELATIONS[r][2]})
        self.rel_emb = nn.ParameterDict(
            {r: nn.Parameter(torch.randn(hidden) * 0.1) for r in RELATION_TYPES})
        self.rel_emb_rev = nn.ParameterDict(
            {r: nn.Parameter(torch.randn(hidden) * 0.1) for r in RELATION_TYPES})
        # relation-specific message transforms (forward + reverse), per layer
        self.msg = nn.ModuleList([nn.ModuleDict(
            {r: nn.Linear(hidden, hidden) for r in RELATION_TYPES}) for _ in range(layers)])
        self.msg_rev = nn.ModuleList([nn.ModuleDict(
            {r: nn.Linear(hidden, hidden) for r in RELATION_TYPES}) for _ in range(layers)])
        self.update = nn.ModuleList([nn.ModuleDict(
            {t: nn.Linear(hidden, hidden) for t in NODE_TYPES}) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.ModuleDict(
            {t: nn.LayerNorm(hidden) for t in NODE_TYPES}) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(5 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.cand_rel = nn.Parameter(torch.randn(hidden) * 0.1)

    def _edge_term(self, r: str, attr: torch.Tensor, rev: bool) -> torch.Tensor:
        emb = (self.rel_emb_rev[r] if rev else self.rel_emb[r])
        if r in self.edge_enc and attr.shape[0]:
            return self.edge_enc[r](attr) + emb
        return emb.expand(attr.shape[0], self.hidden)

    def forward(self, b: Batch) -> torch.Tensor:
        h = {}
        for t in NODE_TYPES:
            role = self.node_role[t].expand(b.node_counts[t], self.hidden)
            h[t] = role + self.node_enc[t](b.x[t]) if t in self.node_enc else role
        dev = next(self.parameters()).device
        for li in range(self.layers):
            agg = {t: torch.zeros(b.node_counts[t], self.hidden, device=dev) for t in NODE_TYPES}
            for r in RELATION_TYPES:
                src_t, dst_t, _a = RELATIONS[r]
                src, dst, attr = b.edges[r]
                if src.shape[0] == 0:
                    continue
                # forward: src -> dst
                fwd = torch.relu(self.msg[li][r](h[src_t][src] + self._edge_term(r, attr, False)))
                agg[dst_t].index_add_(0, dst, fwd)
                # reverse: dst -> src (mirrored, distinct relation role)
                rev = torch.relu(self.msg_rev[li][r](h[dst_t][dst] + self._edge_term(r, attr, True)))
                agg[src_t].index_add_(0, src, rev)
            for t in NODE_TYPES:
                h[t] = self.norm[li][t](h[t] + torch.relu(self.update[li][t](agg[t])))
        feat = torch.cat([
            h["decision"][b.cand_decision], h["order"][b.cand_order],
            h["product"][b.cand_product], h["producer"][b.cand_producer],
            self.cand_rel.expand(b.cand_producer.shape[0], self.hidden)], dim=1)
        return self.head(feat).squeeze(-1)


# --- training + prediction ------------------------------------------------------------
def _centered_loss(pred: torch.Tensor, labels: torch.Tensor, group: torch.Tensor,
                   n_groups: int) -> torch.Tensor:
    """Decision-weighted MSE on decision-centered candidate labels (ARCH.md primary loss):
    center both prediction and label within each decision (strip the between-decision level),
    take per-candidate squared error, average within each decision, then average over
    decisions so a many-candidate decision does not dominate."""
    dev = pred.device
    def center(v: torch.Tensor) -> torch.Tensor:
        sums = torch.zeros(n_groups, device=dev).index_add_(0, group, v)
        counts = torch.zeros(n_groups, device=dev).index_add_(0, group, torch.ones_like(v))
        return v - (sums / counts.clamp(min=1))[group]
    se = (center(pred) - center(labels)) ** 2
    per_dec = torch.zeros(n_groups, device=dev).index_add_(0, group, se)
    counts = torch.zeros(n_groups, device=dev).index_add_(0, group, torch.ones_like(se))
    valid = counts > 0
    return (per_dec[valid] / counts[valid]).mean()


def _pick_device(device: "str | None") -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def train_gnn(graphs: Sequence[GraphArrays], labels: Sequence[float],
              decision_of_candidate: Sequence[int], *, seed: int = 0, epochs: int = 200,
              lr: float = 1e-3, device: "str | None" = None,
              log=lambda *a, **k: None) -> tuple[RelationalGNN, list[float]]:
    """Fit the GNN on `graphs` with per-candidate teacher `labels` (candidate order matches
    each graph's candidate_producers, concatenated). `decision_of_candidate` groups candidates
    by decision for the decision-weighted loss. Deterministic for a fixed `seed`. Runs on
    `device` (default: cuda if available, else cpu) so a GPU pod accelerates the 27 fits."""
    torch.set_num_threads(1)  # single-thread BLAS: removes multithreaded FP reduction-order jitter
    torch.manual_seed(seed)   # so a fixed seed is bitwise-reproducible on CPU (HANDOVER.md:381)
    dev = _pick_device(device)
    model = RelationalGNN().to(dev)
    batch = Batch(graphs).to(dev)
    y = torch.tensor(np.asarray(labels, dtype=np.float32), device=dev)
    group = torch.tensor(np.asarray(decision_of_candidate, dtype=np.int64), device=dev)
    n_groups = int(group.max().item()) + 1 if group.shape[0] else 1
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(batch)
        loss = _centered_loss(pred, y, group, n_groups)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if ep % 50 == 0:
            log(f"  epoch {ep}: loss {loss.item():.5f}")
    return model, history


@torch.no_grad()
def predict(model: RelationalGNN, graphs: Sequence[GraphArrays]) -> np.ndarray:
    """Per-candidate scores for `graphs`, concatenated in candidate order (matches
    candidates.csv candidate_index within each decision)."""
    model.eval()
    if not graphs:
        return np.zeros((0,), dtype=np.float64)
    dev = next(model.parameters()).device
    return model(Batch(graphs).to(dev)).cpu().numpy().astype(np.float64)


# --- leave-one-seed-out gate ----------------------------------------------------------
def _decision_bundle(cand_rows: Sequence[dict], graphs: Sequence[GraphArrays],
                     decision_indices: Sequence[int]):
    """For a set of global decision indices, return (graph_list, flat_labels, group_ids) with
    candidates ordered by candidate_index within each decision so labels align to each graph's
    candidate_producers. Also returns the flat candidate-row list in the same order."""
    by_dec: dict[int, list[dict]] = defaultdict(list)
    for r in cand_rows:
        by_dec[int(r["decision_index"])].append(r)
    glist, labels, group, flat_rows = [], [], [], []
    for local_dec, di in enumerate(decision_indices):
        rows = sorted(by_dec[di], key=lambda r: int(r["candidate_index"]))
        glist.append(graphs[di])
        for r in rows:
            labels.append(float(r["teacher_label"]))
            group.append(local_dec)
            flat_rows.append(r)
    return glist, labels, group, flat_rows


def _flat_decisions(cand_rows: Sequence[dict], tol: float = 1e-9) -> set[int]:
    """Decision indices whose candidate teacher labels are all equal (no within-decision
    preference to learn). HANDOVER.md: these must not dominate training."""
    by_dec: dict[int, list[float]] = defaultdict(list)
    for r in cand_rows:
        by_dec[int(r["decision_index"])].append(float(r["teacher_label"]))
    return {di for di, labs in by_dec.items() if (max(labs) - min(labs)) <= tol}


def heldout_gate(graphs: Sequence[GraphArrays], cand_rows: Sequence[dict], *,
                 model_seed: int = 0, epochs: int = 200, device: "str | None" = None,
                 log=lambda *a, **k: None) -> dict:
    """The corrected EXP-0038A/B gate for the GNN: leave-one-seed-out per task. For each task,
    train on two anchored seeds, predict the held-out seed, POOL held-out (pred, decision) across
    folds so every decision counts once, then score with `observation_ladder._routing_metrics`
    (decision-weighted, tie-aware, real-choice only) — the same instrument that scored the
    EXP-0038B tabular probe and the actor floor. Returns {task_id: gnn_metrics}."""
    by_task_seed: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in cand_rows:
        di = int(r["decision_index"])
        by_task_seed[r["task_id"]][int(r["seed"])].append(di)
    for task in by_task_seed:
        for s in by_task_seed[task]:
            by_task_seed[task][s] = sorted(set(by_task_seed[task][s]))

    flat = _flat_decisions(cand_rows)  # excluded from TRAINING (still scored at test time)
    out: dict[str, dict] = {}
    for task, seed_map in by_task_seed.items():
        seeds = sorted(seed_map)
        pooled_rows, pooled_pred = [], []
        n_train, n_flat_excl = 0, 0
        folds: list[dict] = []  # per-fold audit artifacts (HANDOVER.md:379)
        for held in seeds:
            all_train = [di for s in seeds if s != held for di in seed_map[s]]
            train_dec = [di for di in all_train if di not in flat]
            if not train_dec:  # degenerate: whole fold is flat — nothing to learn from
                log(f"[gate] {task} hold seed {held}: all {len(all_train)} train decisions flat; "
                    "falling back to full fold to avoid an empty fit")
                train_dec = all_train
            n_train += len(train_dec)
            n_flat_excl += len(all_train) - len(train_dec)
            test_dec = seed_map[held]  # held-out is scored in full; real-choice filter is in the metric
            gtr, ytr, grp, _ = _decision_bundle(cand_rows, graphs, train_dec)
            normz = FoldNormalizer.fit(gtr)  # ARCH.md: fit stats on THIS fold's training graphs
            model, hist = train_gnn(normz.apply(gtr), ytr, grp, seed=model_seed, epochs=epochs,
                                    device=device, log=log)
            gte, _, _, test_rows = _decision_bundle(cand_rows, graphs, test_dec)
            preds = predict(model, normz.apply(gte))  # apply the SAME fold stats to held-out
            pooled_rows.extend(test_rows)
            pooled_pred.extend(preds.tolist())
            # per-fold artifacts: predictions (recompute the gate w/o a checkpoint), the ACTUAL
            # fold normalization used for training, the loss curve, and checkpoint provenance.
            folds.append({
                "task_id": task, "held_seed": held, "model_seed": model_seed,
                "n_train_decisions": len(train_dec),
                "loss_history": [float(x) for x in hist],
                "normalization": normz.groups(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "predictions": [
                    {"task_id": r["task_id"], "held_seed": held, "model_seed": model_seed,
                     "decision_index": int(r["decision_index"]),
                     "candidate_index": int(r["candidate_index"]),
                     "producer_id": r.get("producer_id", ""), "pred": float(p),
                     "teacher_label": float(r["teacher_label"]), "actor_score": float(r["actor_score"])}
                    for r, p in zip(test_rows, preds.tolist())],
            })
            log(f"[gate] {task} hold seed {held}: train {len(train_dec)} "
                f"(+{len(all_train) - len(train_dec)} flat dropped), test {len(test_rows)} cand")
        rows = [{"decision_key": (r["task_id"], int(r["seed"]), int(r["decision_index"])),
                 "teacher_label": float(r["teacher_label"]), "task_id": r["task_id"],
                 "actor_score": float(r["actor_score"])} for r in pooled_rows]
        from ..eval.observation_ladder import _routing_metrics
        m = _routing_metrics(rows, pooled_pred)
        m["folds"] = folds
        m["n_train_decisions"] = n_train
        m["n_train_flat_excluded"] = n_flat_excl
        out[task] = m
    return out


def _verdict(per_seed: dict, actor_floor: dict, tasks: Sequence[str],
             *, min_seeds: int = 3) -> dict:
    """Compute the spec.json pass/fail from the five pre-registered conditions so the read is
    a mechanical decision, not a post-hoc reinterpretation (CLAUDE.md: apply the gate without
    reinterpretation).

    Stability (HANDOVER.md:241) is a DISPERSION rule over EVERY gate metric, not just regret vs
    the floor: the four numeric conditions must hold for the seed-mean AND independently for each
    of the >=3 training seeds. If any single seed fails the gate, the result is unstable and the
    verdict is no-pass — averaging alone can never rescue an unstable metric (e.g. r2
    -100/-100/201 averages positive but two seeds are strongly negative)."""
    seeds = sorted(per_seed)

    def actor(t: str) -> float:
        return actor_floor.get(t, {}).get("mean_regret", float("nan"))

    def _conditions(regret: dict, r2: dict) -> dict:
        """The four numeric gate conditions for one point estimate (mean, or a single seed)."""
        beats = {t: regret[t] < actor(t) for t in tasks}
        rel_red = {t: ((actor(t) - regret[t]) / actor(t))
                   if actor(t) and not np.isnan(actor(t)) else float("nan") for t in tasks}
        not_worse = {t: regret[t] <= actor(t) * 1.05 for t in tasks}
        c1 = sum(beats.values()) >= 2
        c2 = sum(1 for t in tasks if r2[t] > 0) >= 2
        c3 = float(np.nanmean([rel_red[t] for t in tasks])) >= 0.10
        c4 = all(not_worse.values())
        return {"beats_actor_2of3": bool(c1), "r2_pos_2of3": bool(c2),
                "rel_reduction_10pct": bool(c3), "no_task_worse_5pct": bool(c4),
                "avg_rel_reduction": float(np.nanmean([rel_red[t] for t in tasks])),
                "beats": {t: bool(beats[t]) for t in tasks},
                "rel_red": rel_red, "pass": bool(c1 and c2 and c3 and c4)}

    mean_reg = {t: float(np.mean([per_seed[ms][t]["mean_regret"] for ms in seeds])) for t in tasks}
    mean_r2 = {t: float(np.mean([per_seed[ms][t]["r2"] for ms in seeds])) for t in tasks}
    mean_c = _conditions(mean_reg, mean_r2)
    per_seed_c = {ms: _conditions({t: per_seed[ms][t]["mean_regret"] for t in tasks},
                                  {t: per_seed[ms][t]["r2"] for t in tasks}) for ms in seeds}
    # dispersion for the report (std across seeds of every gate metric, per task)
    dispersion = {t: {"regret_std": float(np.std([per_seed[ms][t]["mean_regret"] for ms in seeds])),
                      "r2_std": float(np.std([per_seed[ms][t]["r2"] for ms in seeds])),
                      "r2_min": float(min(per_seed[ms][t]["r2"] for ms in seeds)),
                      "r2_max": float(max(per_seed[ms][t]["r2"] for ms in seeds))} for t in tasks}
    stable = len(seeds) >= min_seeds and all(per_seed_c[ms]["pass"] for ms in seeds)

    conditions = {"three_tasks": len(tasks) == 3,
                  "beats_actor_2of3": mean_c["beats_actor_2of3"], "r2_pos_2of3": mean_c["r2_pos_2of3"],
                  "rel_reduction_10pct": mean_c["rel_reduction_10pct"],
                  "no_task_worse_5pct": mean_c["no_task_worse_5pct"], "stable_ge3_seeds": bool(stable)}
    return {"pass": all(conditions.values()), "conditions": conditions,
            "avg_rel_reduction": mean_c["avg_rel_reduction"],
            "per_seed_pass": {ms: per_seed_c[ms]["pass"] for ms in seeds},
            "dispersion": dispersion,
            "per_task": {t: {"gnn_mean_regret": mean_reg[t], "gnn_r2": mean_r2[t],
                             "actor_mean_regret": actor(t), "rel_reduction": mean_c["rel_red"][t],
                             "beats_actor": mean_c["beats"][t]} for t in tasks}}


def _validate_dump_config(dump_dir: str, expected: dict) -> None:
    """Reject a dump whose OWN spec.json deviates from the pre-registered gate config (Finding-4.2).
    `task_ids` is compared as a set (traversal order is irrelevant); scalar teacher/trajectory keys
    must match exactly. A dump with no spec.json cannot be certified and is rejected."""
    import json
    import os

    spec_path = os.path.join(dump_dir, "spec.json")
    if not os.path.exists(spec_path):
        raise ValueError("dump has no spec.json; cannot certify it against the pre-registered config")
    spec = json.load(open(spec_path))
    for key, want in expected.items():
        if key == "task_ids":
            if set(spec.get("task_ids", [])) != set(want):
                raise ValueError(
                    f"dump task_ids {spec.get('task_ids')} != pre-registered {list(want)}")
        elif spec.get(key) != want:
            raise ValueError(f"dump {key}={spec.get(key)!r} != pre-registered {want!r}")


def _validate_dump_provenance(dump_dir: str, expected_task_seeds: "dict | None",
                              expected_inputs: "dict | None") -> None:
    """Bind the dump to the ACTUAL pre-registered inputs, not just their names/shapes (Finding-5.1/
    5.2). `expected_task_seeds` {task_id: set(seeds)} must match the dump's recorded `task_seeds`
    exactly (values, not just count). `expected_inputs` {role: sha256} must match the dump's recorded
    input fingerprints — a dump with no manifest, or a mismatched hash, is rejected."""
    import json
    import os

    spec_path = os.path.join(dump_dir, "spec.json")
    if not os.path.exists(spec_path):
        raise ValueError("dump has no spec.json; cannot certify its inputs/seeds")
    spec = json.load(open(spec_path))
    if expected_task_seeds is not None:
        got = {t: set(v) for t, v in (spec.get("task_seeds") or {}).items()}
        want = {t: set(v) for t, v in expected_task_seeds.items()}
        if got != want:
            raise ValueError(f"dump task_seeds {got} != pre-registered anchored seeds {want}")
    if expected_inputs:
        inputs = spec.get("inputs") or {}
        for role, want_sha in expected_inputs.items():
            got_sha = (inputs.get(role) or {}).get("sha256")
            if got_sha != want_sha:
                raise ValueError(
                    f"dump input '{role}' sha256 {got_sha} != pre-registered {want_sha} "
                    "(dump built from a wrong or unfingerprinted input file)")


def _validate_commit(commit: "str | None") -> None:
    """Pod-side SHAPE check of a source commit (Finding-7.1/8.1). The pod has no `.git`, so require a
    FULL 40-char hex sha — rejects `None`, empty, `not-a-git-object`, and abbreviated ids like
    `a1b2c3d`/`deadbee`. This does NOT prove the object exists or that the tree is clean; that binding
    is `resolve_verified_commit` (launcher-side, needs git) — resolve there and pass the result."""
    if not commit or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
        raise ValueError(f"source commit {commit!r} is not a full 40-char git sha; resolve it locally "
                         "with resolve_verified_commit (verifies existence + clean tree) and pass "
                         "--source-commit")


def assert_source_matches(dump_source: "dict | None", *, expected_commit: "str | None",
                          live_tree_sha: "str | None" = None) -> None:
    """Cross-stage source invariant (Finding-7.1): the dump's recorded source identity must equal the
    code now running the probe, and carry a valid commit that matches the probe's. Recording the hash
    was audit-only — here it is ENFORCED before any training, so a dump built from source A cannot be
    scored under source B (a tampered/zeroed hash mismatches the live tree; a null/garbage commit is
    rejected). `live_tree_sha` defaults to the running tree's fingerprint."""
    from ..eval.graph_value_router_dump import source_fingerprint

    if live_tree_sha is None:
        live_tree_sha = source_fingerprint()["tree_sha256"]
    got = (dump_source or {}).get("tree_sha256")
    if got != live_tree_sha:
        raise ValueError(f"dump source tree_sha256 {got} != probe source {live_tree_sha}: the dump "
                         "was produced by different code than the probe scoring it")
    dump_commit = (dump_source or {}).get("commit")
    _validate_commit(dump_commit)
    _validate_commit(expected_commit)
    if dump_commit != expected_commit:
        raise ValueError(f"dump source commit {dump_commit!r} != probe --source-commit "
                         f"{expected_commit!r}: dump and probe are not the same revision")


# Runtime the trajectory/labels/fit depend on — deps are open lower bounds installed fresh on the pod,
# so a dump and probe on mismatched versions/arch/backend could disagree silently (Finding-8.2/9.2).
# Compared exactly: versions + machine ARCH + numpy BLAS + torch build config (identical versions on a
# different CPU arch or math backend are NOT the same numerical env). `platform`/`image` are recorded
# for audit but not gated (`platform` varies trivially across hosts; `image` may be absent in-container).
_ENV_GATED_KEYS: tuple[str, ...] = (
    "python", "implementation", "numpy", "torch", "pettingzoo", "gymnasium",
    "machine", "numpy_blas", "torch_build",
)


def assert_env_matches(dump_env: "dict | None", live_env: "dict | None" = None) -> None:
    """The dump and probe must run on the same runtime (Finding-8.2). Compares the gated version keys
    exactly and raises on any mismatch or a missing dump manifest — so a probe cannot score a dump
    built under a different numpy/torch/pettingzoo/gymnasium/python than the code now reading it."""
    from ..eval.graph_value_router_dump import environment_manifest

    if live_env is None:
        live_env = environment_manifest()
    if not dump_env:
        raise ValueError("dump has no env manifest; cannot certify the runtime it was built under")
    diffs = {k: (dump_env.get(k), live_env.get(k)) for k in _ENV_GATED_KEYS
             if dump_env.get(k) != live_env.get(k)}
    if diffs:
        raise ValueError(f"dump vs probe environment mismatch {diffs}: the dump was built under a "
                         "different runtime than the probe scoring it (pin deps / reuse the image)")


def _load_dump(dump_dir: str, *, expected_counts: "dict | None" = None,
               expected_config: "dict | None" = None, expected_task_seeds: "dict | None" = None,
               expected_inputs: "dict | None" = None,
               expected_source_commit: "str | None" = None, require_source: bool = False,
               ) -> tuple[list, list[dict]]:
    """Load shard graphs (indexed by global decision) + candidates.csv, and FULLY VALIDATE the
    dump before it can be scored (ARCH.md:343 — the loader materializes one graph per decision
    and validates every candidate offset). Rejections (all raise ValueError):
    - graph_schema.json is MANDATORY and its hash must equal the probe's (never score against a
      different observation);
    - every node/edge tensor is finite;
    - one graph per decision, decisions.csv contiguous 0..N-1;
    - per decision: candidate_index is exactly 0..k-1, k equals the graph's candidate count AND
      decisions.csv n_eligible, and producer_ids (via topologies.jsonl) align in order;
    - each decision's candidate rows agree on task_id/seed, and task_id matches decisions.csv;
    - optional expected_counts {n_decisions, n_candidates} match (pod stop-check)."""
    import csv
    import glob
    import json
    import os

    from ..eval.graph_value_router_dump import decision_digest, load_shard, schema_hash

    schema_path = os.path.join(dump_dir, "graph_schema.json")
    if not os.path.exists(schema_path):
        raise ValueError("dump is missing graph_schema.json (required, ARCH.md artifact contract)")
    stamped = json.load(open(schema_path)).get("schema_hash")
    if stamped != schema_hash():
        raise ValueError(
            f"dump schema hash {stamped} != probe schema {schema_hash()}: "
            "the probe would score against a different observation")
    if expected_config:  # dump must be produced on the pre-registered trajectory + teacher
        _validate_dump_config(dump_dir, expected_config)
    if expected_task_seeds is not None or expected_inputs:  # bind to the exact seeds + input files
        _validate_dump_provenance(dump_dir, expected_task_seeds, expected_inputs)
    if require_source:  # cross-stage: dump code == probe code, both a committed revision (Finding-7.1)
        spec = json.load(open(os.path.join(dump_dir, "spec.json")))
        assert_source_matches(spec.get("source"), expected_commit=expected_source_commit)
        assert_env_matches(spec.get("env"))  # ...and the same runtime versions (Finding-8.2)

    graphs: list = []
    for shard in sorted(glob.glob(os.path.join(dump_dir, "graph_shard_*.npz"))):
        graphs.extend(load_shard(shard))
    for g in graphs:
        for arr in list(g.node_features.values()) + list(g.edge_attr.values()):
            if arr.size and not np.isfinite(arr).all():
                raise ValueError("non-finite feature in loaded dump")

    with open(os.path.join(dump_dir, "decisions.csv")) as f:
        drows = list(csv.DictReader(f))
    if len(graphs) != len(drows):
        raise ValueError(f"{len(graphs)} graphs but {len(drows)} decisions.csv rows")
    with open(os.path.join(dump_dir, "candidates.csv")) as f:
        cand_rows = list(csv.DictReader(f))
    topo = {}
    with open(os.path.join(dump_dir, "topologies.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            topo[rec["task_id"]] = rec["producer_ids"]

    by_dec: dict[int, list[dict]] = defaultdict(list)
    for r in cand_rows:
        by_dec[int(r["decision_index"])].append(r)
    if set(by_dec) != set(range(len(graphs))):
        raise ValueError("candidates.csv decision_index is not exactly one group per decision 0..N-1")
    for di, drow in enumerate(drows):
        group = sorted(by_dec[di], key=lambda r: int(r["candidate_index"]))
        k = len(group)
        if [int(r["candidate_index"]) for r in group] != list(range(k)):
            raise ValueError(f"decision {di}: candidate_index is not 0..{k - 1}")
        if k != graphs[di].candidate_producers.shape[0] or k != int(drow["n_eligible"]):
            raise ValueError(f"decision {di}: candidate count {k} != graph/n_eligible")
        pids = topo[drow["task_id"]]
        recovered = [pids[int(local)] for local in graphs[di].candidate_producers]
        if recovered != [r["producer_id"] for r in group]:
            raise ValueError(f"decision {di}: producer alignment mismatch vs candidates.csv")
        # every candidate row must carry the SAME decision identity as decisions.csv (task/seed/
        # cycle/order) — not merely agree with each other — so a label can never bind to the wrong
        # graph or leave-one-seed-out split.
        key = (drow["task_id"], drow["seed"], drow["cycle"], drow["order_id"])
        for r in group:
            if (r["task_id"], r["seed"], r["cycle"], r["order_id"]) != key:
                raise ValueError(f"decision {di}: candidate metadata disagrees with decisions.csv {key}")
        # the graph PAYLOAD itself must be bound to this decision row (not just the CSV rows agreeing
        # with each other). Recompute the identity digest from decisions.csv and compare to the digest
        # stamped into the shard — a swapped/mislabeled graph payload fails here even when its
        # candidate structure happens to match (Finding-4.1).
        expect = decision_digest(*key)
        if graphs[di].decision_digest != expect:
            raise ValueError(
                f"decision {di}: graph payload not bound to decisions.csv row "
                f"(digest {graphs[di].decision_digest} != {expect} for {key})")

    if expected_counts:  # accept both the dump spec keys and the n_-prefixed aliases
        exp_d = expected_counts.get("n_decisions", expected_counts.get("decisions"))
        exp_c = expected_counts.get("n_candidates", expected_counts.get("eligible_candidates"))
        if exp_d not in (None, len(graphs)):
            raise ValueError(f"n_decisions {len(graphs)} != expected {exp_d}")
        if exp_c not in (None, len(cand_rows)):
            raise ValueError(f"n_candidates {len(cand_rows)} != expected {exp_c}")
    return graphs, cand_rows


def _determinism_preflight(graphs, cand_rows, device: str, tol: float, epochs: int, log) -> dict:
    """HANDOVER.md:381 exit condition: two identical full runs with the same training seed must
    produce identical predictions within the declared tolerance. Certify this on a REAL leave-one-
    seed-out fold trained at the run's FULL `epochs` (not a truncated subset) so the certification
    matches the actual gate training path (Finding-4.4): bitwise-exact on CPU; CUDA index_add_ is
    order-nondeterministic → checked within tol. Raises if the device is not repeatable to `tol`."""
    by_task_seed: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in cand_rows:
        by_task_seed[r["task_id"]][int(r["seed"])].append(int(r["decision_index"]))
    flat = _flat_decisions(cand_rows)
    task = next((t for t in sorted(by_task_seed) if len(by_task_seed[t]) >= 2), None)
    if task is not None:  # a genuine leave-one-seed-out fold: train on the other seeds, hold one out
        seeds = sorted(by_task_seed[task])
        held = seeds[0]
        train_dec = [di for s in seeds if s != held
                     for di in sorted(set(by_task_seed[task][s])) if di not in flat]
        if not train_dec:  # whole fold flat — fall back to full fold rather than an empty fit
            train_dec = [di for s in seeds if s != held for di in sorted(set(by_task_seed[task][s]))]
        test_dec = sorted(set(by_task_seed[task][held]))
        fold = f"{task} hold seed {held}"
    else:  # single-seed data (smoke corner): no held-out fold — use one seed's decisions
        train_dec = sorted({int(r["decision_index"]) for r in cand_rows})
        test_dec = train_dec
        fold = "single-seed subset"
    gtr, ytr, grp, _ = _decision_bundle(cand_rows, graphs, train_dec)
    if not gtr:
        return {"device": device, "tolerance": tol, "max_abs_diff": 0.0, "exact": True,
                "epochs": epochs, "n_train_decisions": 0, "fold": fold}
    gte, _, _, _ = _decision_bundle(cand_rows, graphs, test_dec)
    n = FoldNormalizer.fit(gtr)
    m1, _ = train_gnn(n.apply(gtr), ytr, grp, seed=12345, epochs=epochs, device=device)
    m2, _ = train_gnn(n.apply(gtr), ytr, grp, seed=12345, epochs=epochs, device=device)
    diff = float(np.max(np.abs(predict(m1, n.apply(gte)) - predict(m2, n.apply(gte)))))
    if diff > tol:
        raise ValueError(f"determinism preflight failed on {device}: max abs pred diff {diff} > {tol}")
    log(f"[determinism] {device}: fold '{fold}' @ {epochs} epochs -> "
        f"max abs pred diff {diff:.2e} <= tol {tol:.2e}")
    return {"device": device, "tolerance": tol, "max_abs_diff": diff, "exact": diff == 0.0,
            "epochs": epochs, "n_train_decisions": len(train_dec), "fold": fold}


def run_probe(dump_dir: str, out_dir: str, *, model_seeds: Sequence[int] = (0, 1, 2),
              epochs: int = 200, require_shape: bool = True,
              expected_counts: "dict | None" = None, expected_config: "dict | None" = None,
              expected_inputs: "dict | None" = None, actor_floor_override: "dict | None" = None,
              actor_floor_override_source: "dict | None" = None, source_commit: "str | None" = None,
              device: "str | None" = None, log=print) -> dict:
    """Read a graph dump, run the leave-one-seed-out gate across `model_seeds`, and write the
    gate table alongside the random + deployed-actor floors (reused `floor_metrics`). Reports
    per-training-seed dispersion so instability is explicit (a gate-fail condition).

    `require_shape` (default True) enforces the pre-registered gate shape — exactly 3 tasks, >=3
    trajectory seeds per task, >=3 model seeds — so a partial dump can never earn a mechanical
    PASS (the smoke disables it). `expected_counts` (e.g. {"n_decisions":13940,...}, aliases
    "decisions"/"eligible_candidates" accepted) is validated in `_load_dump`. `actor_floor_override`
    ({task_id: mean_regret}) REPLACES the candidates.csv-derived floor for the verdict — the
    documented EXP-0038B fallback when the checked-in actor checkpoint is pre-P2 — and both floors
    are recorded. `device` pins CPU/CUDA (default: cuda if available)."""
    import csv
    import json
    import os

    from ..eval.observation_ladder import floor_metrics

    # the pre-registered run validates the dump's own config (teacher + task set + horizon), exact
    # anchored task->seed sets, AND the input SHA-256 fingerprints unless a caller overrides; a smoke
    # (require_shape=False) skips all of this.
    expected_task_seeds = None
    if require_shape:
        if expected_config is None:
            expected_config = GATE_CONFIG
        expected_task_seeds = GATE_TASK_SEEDS
        if expected_inputs is None:
            expected_inputs = GATE_INPUT_SHA256
        # a real run must be launched from a committed revision; the dump's source identity is then
        # cross-checked against the probe's in _load_dump before any training (Finding-7.1).
        _validate_commit(source_commit)
        # a fallback floor in the REAL gate must be the pinned EXP-0038B floor, not arbitrary values
        # that could weaken the licensing bar (Finding-6.2); the validated override is REPLACED with
        # the pinned constants so no sub-tolerance drift survives (Finding-7.2). Smoke is exempt.
        if actor_floor_override is not None:
            assert_override_is_canonical(actor_floor_override)
            actor_floor_override = {t: GATE_ACTOR_FLOOR_0038B[t] for t in actor_floor_override}
    graphs, cand_rows = _load_dump(dump_dir, expected_counts=expected_counts,
                                   expected_config=expected_config,
                                   expected_task_seeds=expected_task_seeds,
                                   expected_inputs=expected_inputs,
                                   expected_source_commit=source_commit, require_source=require_shape)
    tasks = sorted({r["task_id"] for r in cand_rows})
    if require_shape:
        seeds_per_task = {t: len({r["seed"] for r in cand_rows if r["task_id"] == t}) for t in tasks}
        want = GATE_CONFIG["eval_seeds"]
        # EXACTLY the anchored seeds: leave-one-seed-out is "train on two, hold out the third" — a
        # dump with extra trajectory seeds silently redefines the split (Finding-4.2).
        if len(tasks) != 3 or any(n != want for n in seeds_per_task.values()) or len(model_seeds) < 3:
            raise ValueError(
                f"pre-registered gate shape violated: tasks={len(tasks)} "
                f"seeds_per_task={seeds_per_task} (want exactly {want}) model_seeds={len(model_seeds)}; "
                "the gate is defined for exactly 3 tasks x 3 anchored trajectory seeds x >=3 model "
                "seeds (pass require_shape=False only for a smoke)")
        if set(tasks) != set(GATE_TASK_IDS):
            raise ValueError(f"dump tasks {tasks} != pre-registered gate tasks {list(GATE_TASK_IDS)}")
        # the SCORED seed VALUES (not just count) must be the anchored sets — validated directly on
        # candidates.csv, independent of the dump manifest (Finding-5.2).
        for t in tasks:
            got = {int(r["seed"]) for r in cand_rows if r["task_id"] == t}
            if got != GATE_TASK_SEEDS[t]:
                raise ValueError(f"task {t}: scored seeds {sorted(got)} != anchored "
                                 f"{sorted(GATE_TASK_SEEDS[t])}")

    dev = _pick_device(device)
    tol = 0.0 if dev == "cpu" else 1e-4  # CPU is bitwise-exact; CUDA index_add_ within tolerance
    determinism = _determinism_preflight(graphs, cand_rows, dev, tol, epochs, log)

    # attach a decision_key for the floor metrics (same convention as the gate)
    for r in cand_rows:
        r["_dk"] = (r["task_id"], int(r["seed"]), int(r["decision_index"]))
    floor_rows = [{"decision_key": r["_dk"], "teacher_label": float(r["teacher_label"]),
                   "task_id": r["task_id"], "actor_score": float(r["actor_score"])} for r in cand_rows]
    random_floor = floor_metrics(floor_rows, kind="random")
    actor_floor = floor_metrics(floor_rows, kind="actor")
    # the floor that DECIDES the verdict: the override when supplied (validated fallback), else derived
    verdict_floor = ({t: {"mean_regret": float(actor_floor_override[t])} for t in actor_floor_override}
                     if actor_floor_override else actor_floor)

    per_seed: dict[int, dict] = {}
    for ms in model_seeds:
        per_seed[ms] = heldout_gate(graphs, cand_rows, model_seed=ms, epochs=epochs, device=dev, log=log)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Pull per-fold audit artifacts out of the metrics dicts (state_dict tensors are not
    # JSON-able and predictions are large) and materialize them (HANDOVER.md:379/405/408).
    pred_rows: list[dict] = []
    folds_meta: list[dict] = []
    for ms in model_seeds:
        for task in tasks:
            for fold in per_seed[ms][task].pop("folds"):
                pred_rows.extend(fold["predictions"])
                ck_path = os.path.join(ckpt_dir, f"{task}_h{fold['held_seed']}_m{ms}.pt")
                torch.save(fold["state_dict"], ck_path)
                folds_meta.append({
                    "task_id": task, "held_seed": fold["held_seed"], "model_seed": ms,
                    "n_train_decisions": fold["n_train_decisions"],
                    "final_loss": fold["loss_history"][-1] if fold["loss_history"] else None,
                    "loss_history": fold["loss_history"],
                    "normalization": fold["normalization"],  # the ACTUAL fold stats used to train
                    "checkpoint": os.path.relpath(ck_path, out_dir)})
    # predictions.csv — every gate clause is recomputable from this WITHOUT a checkpoint or the
    # simulator (HANDOVER.md:408 exit condition).
    pred_fields = ["task_id", "held_seed", "model_seed", "decision_index", "candidate_index",
                   "producer_id", "pred", "teacher_label", "actor_score"]
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields)
        w.writeheader()
        w.writerows(pred_rows)

    # the floor that DECIDES the verdict, reported consistently in the table (Finding-4.3): under an
    # override it is the supplied per-task regret, else the candidates.csv-derived actor floor.
    floor_source = "override" if actor_floor_override else "derived"

    def _derived_full(t: str) -> float:
        return float(actor_floor.get(t, {}).get("mean_regret", float("nan")))

    def _vfloor_full(t: str) -> float:
        return float(verdict_floor.get(t, {}).get("mean_regret", float("nan")))

    def _derived(t: str) -> float:
        return round(_derived_full(t), 4)

    def _vfloor(t: str) -> float:
        return round(_vfloor_full(t), 4)

    rows_out = []
    for task in tasks:
        gnn_regrets = [per_seed[ms][task]["mean_regret"] for ms in model_seeds]
        gnn_r2 = [per_seed[ms][task]["r2"] for ms in model_seeds]
        rows_out.append({
            "task_id": task,
            "gnn_mean_regret": round(float(np.mean(gnn_regrets)), 4),
            "gnn_mean_regret_std": round(float(np.std(gnn_regrets)), 4),
            "gnn_r2": round(float(np.mean(gnn_r2)), 4),
            "gnn_r2_std": round(float(np.std(gnn_r2)), 4),
            # the floor the verdict compared against, and its provenance (never contradicts the read)
            "verdict_floor_regret": _vfloor(task),
            "floor_source": floor_source,
            "actor_mean_regret": _derived(task),  # derived actor floor, always reported as a baseline
            "random_mean_regret": round(random_floor.get(task, {}).get("mean_regret", float("nan")), 4),
            "n_decisions": per_seed[model_seeds[0]][task]["n_decisions"],
        })
    with open(os.path.join(out_dir, "metrics_by_task.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    # floor_by_task.csv — the verdict's reproduction artifact: records the derived floor AND the
    # effective verdict floor at FULL precision (no 4-dp rounding — 1.000049 rounded to 1.0 could flip
    # PASS->FAIL, Finding-5.3) plus the override's source path+sha256, so the (possibly overridden)
    # verdict is reproducible from this CSV alone (HANDOVER.md:408).
    override_src = ("" if not actor_floor_override_source
                    else f"{actor_floor_override_source.get('path', '')}"
                         f"@sha256:{actor_floor_override_source.get('sha256', '')}")
    with open(os.path.join(out_dir, "floor_by_task.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "derived_actor_regret", "random_regret",
                                          "verdict_floor_regret", "floor_source", "override_source"])
        w.writeheader()
        for task in tasks:
            w.writerow({"task_id": task, "derived_actor_regret": repr(_derived_full(task)),
                        "random_regret": repr(float(random_floor.get(task, {}).get(
                            "mean_regret", float("nan")))),
                        "verdict_floor_regret": repr(_vfloor_full(task)),
                        "floor_source": floor_source, "override_source": override_src})
    verdict = _verdict(per_seed, verdict_floor, tasks)
    from ..eval.graph_value_router_dump import environment_manifest, schema_hash, source_fingerprint
    result = {"source": source_fingerprint(source_commit), "env": environment_manifest(),
              "per_seed": {ms: per_seed[ms] for ms in model_seeds},
              "random_floor": random_floor, "actor_floor": actor_floor,
              "actor_floor_override": actor_floor_override,
              "actor_floor_override_source": actor_floor_override_source,
              "verdict_floor": "override" if actor_floor_override else "derived",
              "by_task": rows_out, "model_seeds": list(model_seeds), "epochs": epochs,
              "schema_hash": schema_hash(), "verdict": verdict,
              "device": dev, "torch_version": torch.__version__, "determinism": determinism,
              # per-fold normalization is the actual train-fold stats used (NOT an all-graph refit
              # over held-out+flats); loss history + checkpoint provenance travel with each fold.
              "folds": folds_meta,
              "predictions_csv": "predictions.csv", "n_predictions": len(pred_rows)}
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"[verdict] pass={verdict['pass']} conditions={verdict['conditions']}")
    return result


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True, help="graph dump dir (from graph_value_router_dump)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--model-seeds", default="0,1,2", help="comma-separated training seeds")
    ap.add_argument("--expect-decisions", type=int, default=None, help="assert dump n_decisions")
    ap.add_argument("--expect-candidates", type=int, default=None, help="assert dump n_candidates")
    ap.add_argument("--actor-floor-json", default=None,
                    help="path to {task_id: mean_regret} JSON that OVERRIDES the derived actor "
                         "floor for the verdict (EXP-0038B fallback when the checkpoint is pre-P2); "
                         "validated against the pinned canonical floor in the real gate; its "
                         "path+sha256 are recorded in floor_by_task.csv/spec.json")
    ap.add_argument("--actor-floor-exp0038b", nargs="?", const=GATE_FLOOR_ARTIFACT, default=None,
                    help="derive the fallback floor MECHANICALLY from the pinned EXP-0038B artifact "
                         "(SHA-256 validated); optional path arg (default: the checked-in artifact, "
                         "which lives OUTSIDE the synced pod tree — stage it or use "
                         "--actor-floor-canonical on a pod)")
    ap.add_argument("--actor-floor-canonical", action="store_true",
                    help="use the in-code pinned EXP-0038B floor (GATE_ACTOR_FLOOR_0038B) directly — "
                         "pod-safe (needs no artifact file); the values are identical to "
                         "--actor-floor-exp0038b (Finding-7.3)")
    ap.add_argument("--source-commit", default=None,
                    help="full 40-char git commit (the pod has no .git; resolve locally and pass the "
                         "SAME commit the dump used — _load_dump aborts unless they match)")
    ap.add_argument("--verify-source-commit", action="store_true",
                    help="resolve+verify the commit from the LOCAL git tree (existence + clean tree) "
                         "and use it (Finding-8.1); needs git")
    ap.add_argument("--device", default=None, help="cpu|cuda (default: cuda if available)")
    ap.add_argument("--no-require-shape", action="store_true",
                    help="disable the 3-task/3-seed shape + provenance guards (smoke only)")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.model_seeds.split(","))
    source_commit = args.source_commit
    if args.verify_source_commit:
        from ..eval.graph_value_router_dump import resolve_verified_commit
        source_commit = resolve_verified_commit()
    expected = {"n_decisions": args.expect_decisions, "n_candidates": args.expect_candidates}
    override = override_source = None
    if args.actor_floor_canonical:
        override = dict(GATE_ACTOR_FLOOR_0038B)
        override_source = {"path": "GATE_ACTOR_FLOOR_0038B (pinned constant)", "sha256": GATE_FLOOR_SHA256}
    elif args.actor_floor_exp0038b:
        override = canonical_actor_floor(args.actor_floor_exp0038b)
        override_source = {"path": args.actor_floor_exp0038b, "sha256": GATE_FLOOR_SHA256}
    elif args.actor_floor_json:
        import json as _json

        from ..eval.graph_value_router_dump import sha256_file
        override = {k: float(v) for k, v in _json.load(open(args.actor_floor_json)).items()}
        override_source = {"path": args.actor_floor_json, "sha256": sha256_file(args.actor_floor_json)}
    run_probe(args.dump_dir, args.out_dir, model_seeds=seeds, epochs=args.epochs,
              require_shape=not args.no_require_shape, expected_counts=expected,
              actor_floor_override=override, actor_floor_override_source=override_source,
              source_commit=source_commit, device=args.device)


if __name__ == "__main__":
    main()
