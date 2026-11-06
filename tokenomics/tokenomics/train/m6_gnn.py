"""M6 fully learned producer and orderer GNN policies (add-m6-joint-mpg-gnn D2 / D3).

Each agent's actor reads only its own view, which is ADR-0058's information set:
- a producer reads its `ProducerEntityView` (capacity, product and input rows with their recipe relations) plus
  the producer-local mechanics the env already gives every producing policy (max replenish, replenish cost,
  expansion size / cost) and the investment fund level;
- an ordering agent reads its `ProductStreamView` (product, producer rows, its own backlog with each order's
  remaining lifetime, and last cycle's sub-order outcomes) and its own routing view.

Both are typed message-passing networks: shared encoders per row type, 2 rounds over the view's relations, and
heads. They are topology-agnostic: no ids are features, and the parameter count does not depend on the producer
count. They emit the env's raw actions; the env decode owns legality (running / unconfigured investments,
headroom, conservation).
- **Producer head:** 3 toggle logits (restore / upgrade / expand) and a replenish fraction per input row.
- **Orderer head:** per order slot, logits over its eligible producers; one keyed draw (MWWk's key), the order
  routed whole (ADR-0059).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from ..domain.orderer_view import ProductStreamView
from ..domain.producer_view import ProducerEntityView
from ..env.spaces import ACTION_LOGIT_BOUND
from ..eval.baselines import blocked_shortfall, recent_input_use
from ..eval.rotation_routers import _flat, _matrix, _slots, _whole
from ..eval.routing_diagnostics import draw_key, inverse_cdf, keyed_uniform
from . import orderer_scorer as osc
from . import producer_scorer as psc

if TYPE_CHECKING:
    from ..env import TokenomicsAECEnv

Policy = Callable[["TokenomicsAECEnv", str], np.ndarray]
_q = psc._qty
OUTCOME_KINDS = ("accepted", "declined", "completed", "failed")
HIDDEN = 32
ROUNDS = 2


# --- features (from the agent's own view only) ---


@dataclass(frozen=True)
class ProducerContext:
    """Producer-local mechanics the env passes every producing policy (as k5_rf5 reads them)."""

    max_replenish: int
    replenish_cost_per_unit: float
    expansion_increases: int
    expansion_cost: float
    investment_fund: float


def producer_context(env: "TokenomicsAECEnv", pid: str) -> ProducerContext:
    m = env.spec_.producer_mechanics[pid]
    return ProducerContext(int(env.spec_.max_replenish[pid]), float(m.replenish_cost_per_unit or 0.0),
                           int(m.inventory_expansion_increases or 0),
                           float(m.inventory_expansion_cost or 0.0),
                           float(env.system.investment_fund))


# Producer feature versions: v2 = Stage I v1 / v2 (EXP-0099 / 0100; 16 capacity columns); v3 = Stage I v3
# (supervisor decision 2026-11-06, DIAGNOSTICS 32): + the signed ceiling margin per capacity row.
# v4 = Stage I v4 (DIAGNOSTICS 33): + the margin's sign, 1[margin < 0], a step input for the sharp boundary.
FEATURES_V2, FEATURES_V3, FEATURES_V4 = "v2", "v3", "v4"
CAPACITY_DIM = {FEATURES_V2: 16, FEATURES_V3: 17, FEATURES_V4: 18}
LATEST_FEATURES = FEATURES_V4
MARGIN_COL, SIGN_COL = 16, 17


def ceiling_margin(r: Any) -> float:
    """(restore_effect - deg_after) / max(restore_effect, deg_after, 1), deg_after =
    ceil(rate (max + upgrade_effect) / 100): after one more upgrade, how far one restoration per
    cycle stays ahead of degradation (< 0 = past the sustainable ceiling, DIAGNOSTICS 29 / 32). In
    [-1, 1] at any scale (a row without a restoration reads -1 once it degrades); the producer's own
    information (its view's rate, max, effects)."""
    after = math.ceil(float(getattr(r, "degradation_rate", 0)) * (r.max + r.upgrade_effect) / 100)
    return (r.restore_effect - after) / max(r.restore_effect, after, 1)


def v3_to_v4(f: dict[str, Any]) -> dict[str, Any]:
    """Recorded v3 producer features -> v4 (append the margin's sign); identical to recording v4."""
    cap = f["capacity"]
    sign = (cap[:, MARGIN_COL:MARGIN_COL + 1] < 0).astype(np.float32)
    return {**f, "capacity": np.concatenate([cap, sign], 1) if cap.shape[0] else
            np.zeros((0, CAPACITY_DIM[FEATURES_V4]), dtype=np.float32)}


def producer_features(view: ProducerEntityView, ctx: ProducerContext,
                      version: str = LATEST_FEATURES) -> dict[str, Any]:
    """Row features and relation edges of one producer decision (pure; numpy). `version` = the
    model's feature version (v3 adds `ceiling_margin` per capacity row)."""
    use = recent_input_use(view)
    short = blocked_shortfall(view)
    glob = psc._global_features(view) + [
        _q(ctx.investment_fund), _q(ctx.max_replenish), _q(ctx.replenish_cost_per_unit),
        _q(ctx.expansion_increases), _q(ctx.expansion_cost)]
    caps = []
    for r in view.capacity_rows:
        row = psc._capacity_features(r) + [float(getattr(r, "degradation_rate", 0)) / 100.0]
        if version in (FEATURES_V3, FEATURES_V4):
            m = ceiling_margin(r)
            row += [m] + ([float(m < 0)] if version == FEATURES_V4 else [])
        caps.append(row)
    prods = [psc._product_features(r) for r in view.product_rows]
    inputs = [psc._input_features(r) + [_q(use.get(r.input_product, 0.0)),
                                         _q(short.get(r.input_product, 0))]
              for r in view.input_rows]
    rel = view.relations
    return {"global": _arr([glob]), "capacity": _arr(caps, CAPACITY_DIM[version]),
            "product": _arr(prods, 18),
            "input": _arr(inputs, 6),
            "prod_cap": sorted(rel.product_capacity_required), "prod_in": sorted(
                rel.product_input_required),
            "inputs": [r.input_product for r in view.input_rows]}


def orderer_features(view: ProductStreamView, slots: Sequence[tuple[int, Any, int, list[int]]],
                     producer_ids: Sequence[str]) -> dict[str, Any]:
    """The stream agent's product, producer rows (with last cycle's outcomes per producer), and
    order slots (with each order's remaining lifetime); `elig` per slot in `producer_ids` order."""
    outcomes = {p: [0.0] * len(OUTCOME_KINDS) for p in view.producers}
    for o in getattr(view, "last_outcomes", ()):
        if o.producer_id in outcomes and o.outcome in OUTCOME_KINDS:
            outcomes[o.producer_id][OUTCOME_KINDS.index(o.outcome)] += float(o.quantity)
    pids = [p for p in producer_ids if p in view.producers]
    rows = [osc._producer_row_features(view.producers[p], view.producer_products[p])
            + [_q(x) for x in outcomes[p]] for p in pids]
    by_id = {s.order_id: s for s in view.backlog}
    orders, elig = [], []
    col = {p: i for i, p in enumerate(pids)}
    for _k, oid, _q_unused, el in slots:
        s = by_id.get(oid)
        orders.append([] if s is None else [
            _q(s.unresolved_quantity), _q(s.tokens_per_unit), _q(s.order_age),
            float(s.ttl_remaining), *[float(x) for x in s.order_type]])
        elig.append([col[producer_ids[i]] for i in el if producer_ids[i] in col])
    orders = [o if o else [0.0] * 7 for o in orders]
    return {"global": _arr([osc._product_features(view.product_stats)]),
            "producer": _arr(rows, osc._PRODUCER_ROW_DIM + len(OUTCOME_KINDS)),
            "order": _arr(orders, 7), "elig": elig, "pids": pids}


def _arr(rows: list[list[float]], width: int | None = None) -> np.ndarray:
    if not rows:
        return np.zeros((0, width or 0), dtype=np.float32)
    a = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=20.0, neginf=-20.0)


# --- normaliser (DIAGNOSTICS 31; M4's median / IQR style, per_order_scorer.FeatureNormaliser) ---

NORM_FORMAT = "m6-median-iqr-slog1p-v1"
NORM_CLIP = 5.0  # normalised values clipped to +-NORM_CLIP (as M4)
# Count / unit / token columns, signed-log1p'd before the median / IQR (raw orderer counts reached
# 5,966 in EXP-0099). The producer's quantities are log1p'd at feature build (psc._qty), and the
# orderer's order-slot and outcome columns too (`_q`), so those blocks transform nothing.
_ORD_PRODUCT_LOG1P = ("open_demand_quantity", "open_demand_tokens", "recent_average_cycle_demand",
                      "recent_rejected_quantity", "recipe_input_units", "eligible_producer_count")
_ORD_ROW_LOG1P = ("total_current_capacity", "total_max_capacity", "capacity_headroom",
                  "requested_load", "reserved_load", "total_load", "cut_off_price",
                  "outstanding_inbound_units", "accept_qty_now", "complete_qty_now",
                  "makeable_qty_now", "recent_routed_qty", "recent_completed_qty",
                  "recent_rejected_qty", "recent_throughput", "recent_completion_latency",
                  "recent_failure_latency")
ORDERER_LOG1P: dict[str, tuple[int, ...]] = {
    "global": tuple(osc._ORDERER_PRODUCT_FEATURES.index(n) for n in _ORD_PRODUCT_LOG1P),
    "producer": tuple(osc._ORDERER_PRODUCER_ROW_FEATURES.index(n) for n in _ORD_ROW_LOG1P)}
PRODUCER_LOG1P: dict[str, tuple[int, ...]] = {}


def slog1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class FeatureNorm(nn.Module):
    """Per block and column: z = clip((T(x) - median) / scale, +-NORM_CLIP), T = signed log1p on
    the registered count columns. Scale = IQR; IQR = 0 -> std; std = 0 -> 1 (M4's zero-spread
    rule). Buffers, so the fitted values live in the checkpoint; identity until `fit`."""

    def __init__(self, dims: dict[str, int], log1p: dict[str, tuple[int, ...]]) -> None:
        super().__init__()
        self.blocks = tuple(dims)
        self.clip: float | None = NORM_CLIP  # None: a pre-normaliser checkpoint (EXP-0099)
        self.log1p = {k: tuple(log1p.get(k, ())) for k in dims}
        for k, d in dims.items():
            self.register_buffer(f"c_{k}", torch.zeros(d))
            self.register_buffer(f"s_{k}", torch.ones(d))

    def transform(self, block: str, x: np.ndarray) -> np.ndarray:
        cols = list(self.log1p[block])
        if cols and x.shape[0]:
            x = x.copy()
            x[:, cols] = slog1p(x[:, cols])
        return x

    def forward(self, block: str, x: np.ndarray) -> torch.Tensor:
        z = (torch.from_numpy(self.transform(block, x)) - getattr(self, f"c_{block}")) / getattr(
            self, f"s_{block}")
        return z if self.clip is None else z.clamp(-self.clip, self.clip)

    def fit(self, feats: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Median / scale per column over the transformed rows of `feats` (the training
        samples, or a fixed stride of them); returns the record for the fit report."""
        rec = {}
        for k in self.blocks:
            rows = [f[k] for f in feats if f[k].shape[0]]
            if not rows:
                continue
            x = self.transform(k, np.concatenate(rows, 0)).astype(np.float64)
            q25, q50, q75 = np.percentile(x, [25.0, 50.0, 75.0], axis=0)
            iqr = q75 - q25
            sd = np.where(np.ptp(x, axis=0) == 0, 0.0, x.std(axis=0))
            scale = np.where(iqr > 0, iqr, np.where(sd > 0, sd, 1.0))
            getattr(self, f"c_{k}").copy_(torch.from_numpy(q50.astype(np.float32)))
            getattr(self, f"s_{k}").copy_(torch.from_numpy(scale.astype(np.float32)))
            rec[k] = {"rows": int(x.shape[0]), "median": q50.tolist(), "scale": scale.tolist()}
        return rec


# --- models ---


def _mlp(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))


def _mean_ctx(edges: Sequence[tuple[int, int]], n_dst: int, src: torch.Tensor, dst_first: bool
              ) -> torch.Tensor:
    """Mean of the neighbours' embeddings per destination row over (a, b) edges; `dst_first`
    means the destination is the edge's first index."""
    out = torch.zeros((n_dst, src.shape[1]))
    if not edges or n_dst == 0 or src.shape[0] == 0:
        return out
    d = torch.tensor([e[0] if dst_first else e[1] for e in edges])
    s = torch.tensor([e[1] if dst_first else e[0] for e in edges])
    out = out.index_add(0, d, src[s])
    cnt = torch.zeros(n_dst).index_add(0, d, torch.ones(len(edges))).clamp(min=1)
    return out / cnt.unsqueeze(1)


class ProducerGNN(nn.Module):
    """Typed message passing over a producer's own entity view (D3); heads per D2."""

    def __init__(self, hidden: int = HIDDEN, rounds: int = ROUNDS,
                 features: str = LATEST_FEATURES) -> None:
        super().__init__()
        self.hidden, self.rounds, self.features = hidden, rounds, features
        cap = CAPACITY_DIM[features]
        self.enc = nn.ModuleDict({"global": _mlp(13, hidden), "capacity": _mlp(cap, hidden),
                                  "product": _mlp(18, hidden), "input": _mlp(6, hidden)})
        self.norm = FeatureNorm({"global": 13, "capacity": cap, "product": 18, "input": 6},
                                PRODUCER_LOG1P)
        self.upd = nn.ModuleList([nn.ModuleDict({
            "product": _mlp(3 * hidden, hidden), "capacity": _mlp(2 * hidden, hidden),
            "input": _mlp(2 * hidden, hidden)}) for _ in range(rounds)])
        self.gates = nn.Linear(4 * hidden, 3)
        self.replenish = nn.Linear(5 * hidden, 1)

    def forward(self, f: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(3 toggle logits, a replenish fraction in (0, 1) per input row)."""
        gates, logit = self.heads(f)
        return gates, torch.sigmoid(logit)

    def embed(self, f: dict[str, Any]) -> torch.Tensor:
        """The pooled view embedding (4 x hidden): the CTDE critic's per-producer input."""
        return self._trunk(f)[1]

    def heads(self, f: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(3 toggle logits, the replenish pre-sigmoid logit per input row): the RL policy's
        distribution parameters (m6_mappo); `forward` = the same with the sigmoid applied."""
        h, pooled = self._trunk(f)
        gates = self.gates(pooled)
        n_in = h["input"].shape[0]
        logit = (self.replenish(torch.cat([h["input"], pooled.unsqueeze(0).expand(n_in, -1)], 1)
                                ).squeeze(-1) if n_in else torch.zeros(0))
        return gates, logit

    def _trunk(self, f: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        t = {k: self.norm(k, f[k]) for k in ("global", "capacity", "product", "input")}
        h = {k: (self.enc[k](v) if v.shape[0] else torch.zeros((0, self.hidden)))
             for k, v in t.items()}
        for layer in self.upd:
            n_p = h["product"].shape[0]
            pc = _mean_ctx(f["prod_cap"], n_p, h["capacity"], True)
            pi = _mean_ctx(f["prod_in"], n_p, h["input"], True)
            cp = _mean_ctx(f["prod_cap"], h["capacity"].shape[0], h["product"], False)
            ip = _mean_ctx(f["prod_in"], h["input"].shape[0], h["product"], False)
            h = {"global": h["global"],
                 "product": h["product"] + (layer["product"](torch.cat([h["product"], pc, pi], 1))
                                            if n_p else 0),
                 "capacity": h["capacity"] + (layer["capacity"](torch.cat([h["capacity"], cp], 1))
                                              if h["capacity"].shape[0] else 0),
                 "input": h["input"] + (layer["input"](torch.cat([h["input"], ip], 1))
                                        if h["input"].shape[0] else 0)}
        pooled = torch.cat([h["global"][0], *[_mean(h[k], self.hidden)
                                               for k in ("capacity", "product", "input")]])
        return h, pooled


class OrdererGNN(nn.Module):
    """Typed message passing over a stream agent's own view: producers <-> order slots over the
    eligibility edges, with the product as context; per (slot, eligible producer) a logit."""

    def __init__(self, hidden: int = HIDDEN, rounds: int = ROUNDS) -> None:
        super().__init__()
        self.hidden = hidden
        self.enc = nn.ModuleDict({"global": _mlp(osc._PRODUCT_DIM, hidden),
                                  "producer": _mlp(osc._PRODUCER_ROW_DIM + len(OUTCOME_KINDS),
                                                   hidden),
                                  "order": _mlp(7, hidden)})
        self.norm = FeatureNorm({"global": osc._PRODUCT_DIM,
                                 "producer": osc._PRODUCER_ROW_DIM + len(OUTCOME_KINDS),
                                 "order": 7}, ORDERER_LOG1P)
        self.upd = nn.ModuleList([nn.ModuleDict({"producer": _mlp(3 * hidden, hidden),
                                                 "order": _mlp(3 * hidden, hidden)})
                                  for _ in range(rounds)])
        self.score = _mlp(3 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, f: dict[str, Any]) -> list[torch.Tensor]:
        """Per slot, logits over its eligible producers (in `f["elig"][k]` order)."""
        g, hp, ho = self._trunk(f)
        out = []
        for k, el in enumerate(f["elig"]):
            if not el:
                out.append(torch.zeros(0))
                continue
            x = torch.cat([ho[k].unsqueeze(0).expand(len(el), -1), hp[el],
                           g.unsqueeze(0).expand(len(el), -1)], 1)
            out.append(self.out(torch.relu(self.score(x))).squeeze(-1))
        return out

    def embed(self, f: dict[str, Any]) -> torch.Tensor:
        """The pooled stream-view embedding (3 x hidden): the CTDE critic's per-stream input."""
        g, hp, ho = self._trunk(f)
        return torch.cat([g, _mean(hp, self.hidden), _mean(ho, self.hidden)])

    def _trunk(self, f: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.enc["global"](self.norm("global", f["global"]))[0]
        hp = self.enc["producer"](self.norm("producer", f["producer"])) if len(f["pids"]) else \
            torch.zeros((0, self.hidden))
        ho = self.enc["order"](self.norm("order", f["order"])) if len(f["elig"]) else \
            torch.zeros((0, self.hidden))
        edges = [(k, j) for k, el in enumerate(f["elig"]) for j in el]  # (order, producer)
        for layer in self.upd:
            oc = _mean_ctx(edges, ho.shape[0], hp, True)
            pc = _mean_ctx(edges, hp.shape[0], ho, False)
            gp = g.unsqueeze(0)
            ho = ho + (layer["order"](torch.cat([ho, oc, gp.expand(ho.shape[0], -1)], 1))
                       if ho.shape[0] else 0)
            hp = hp + (layer["producer"](torch.cat([hp, pc, gp.expand(hp.shape[0], -1)], 1))
                       if hp.shape[0] else 0)
        return g, hp, ho


def _mean(h: torch.Tensor, hidden: int) -> torch.Tensor:
    return h.mean(0) if h.shape[0] else torch.zeros(hidden)


# --- deployed policies (the GNN alone; the env decode owns legality) ---


def producer_action(model: ProducerGNN, env: "TokenomicsAECEnv", pid: str) -> np.ndarray:
    view = env.system.producer_entity_view(pid)
    f = producer_features(view, producer_context(env, pid), model.features)
    with torch.no_grad():
        gates, rep = model(f)
    sp = env.spec_
    act = np.zeros(sp.producing_action_dim, dtype=np.float32)
    for g, frac in zip(f["inputs"], rep.numpy()):
        if g in sp.input_index:
            act[sp.input_index[g]] = float(frac) * ACTION_LOGIT_BOUND
    act[sp.num_inputs:sp.num_inputs + 3] = np.where(gates.numpy() > 0, ACTION_LOGIT_BOUND,
                                                     -ACTION_LOGIT_BOUND)
    return act


def gnn_producing_policy(model: ProducerGNN) -> Policy:
    model.eval()

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        return producer_action(model, env, env._role[agent][1])

    return policy


def gnn_ordering_policy(model: OrdererGNN) -> Policy:
    """One keyed draw per order slot from softmax over its eligible producers' logits (MWWk's
    key), the order routed whole."""
    model.eval()

    def policy(env: "TokenomicsAECEnv", agent: str) -> np.ndarray:
        key, cycle = draw_key(env), env.system.cycle_counter
        _, aid = env._role[agent]
        slots = _slots(env, agent)
        f = orderer_features(env.system.product_stream_view(aid), slots, env.spec_.producer_ids)
        with torch.no_grad():
            logits = model(f)
        mat = _matrix(env)
        col = {p: i for i, p in enumerate(env.spec_.producer_ids)}
        for (k, _oid, q, el), lg, cols in zip(slots, logits, f["elig"]):
            if el and len(cols):
                w = np.exp(lg.numpy() - lg.numpy().max())
                pick = inverse_cdf(w, keyed_uniform(key, agent, cycle, k))
                chosen = el.index(col[f["pids"][cols[pick]]])
                mat[k, el] = _whole(el, chosen, q)
        return _flat(mat)

    return policy
