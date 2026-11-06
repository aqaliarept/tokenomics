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


def producer_features(view: ProducerEntityView, ctx: ProducerContext) -> dict[str, Any]:
    """Row features and relation edges of one producer decision (pure; numpy)."""
    use = recent_input_use(view)
    short = blocked_shortfall(view)
    glob = psc._global_features(view) + [
        _q(ctx.investment_fund), _q(ctx.max_replenish), _q(ctx.replenish_cost_per_unit),
        _q(ctx.expansion_increases), _q(ctx.expansion_cost)]
    caps = [psc._capacity_features(r) + [float(getattr(r, "degradation_rate", 0)) / 100.0]
            for r in view.capacity_rows]
    prods = [psc._product_features(r) for r in view.product_rows]
    inputs = [psc._input_features(r) + [_q(use.get(r.input_product, 0.0)),
                                         _q(short.get(r.input_product, 0))]
              for r in view.input_rows]
    rel = view.relations
    return {"global": _arr([glob]), "capacity": _arr(caps, 16), "product": _arr(prods, 18),
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

    def __init__(self, hidden: int = HIDDEN, rounds: int = ROUNDS) -> None:
        super().__init__()
        self.hidden, self.rounds = hidden, rounds
        self.enc = nn.ModuleDict({"global": _mlp(13, hidden), "capacity": _mlp(16, hidden),
                                  "product": _mlp(18, hidden), "input": _mlp(6, hidden)})
        self.upd = nn.ModuleList([nn.ModuleDict({
            "product": _mlp(3 * hidden, hidden), "capacity": _mlp(2 * hidden, hidden),
            "input": _mlp(2 * hidden, hidden)}) for _ in range(rounds)])
        self.gates = nn.Linear(4 * hidden, 3)
        self.replenish = nn.Linear(5 * hidden, 1)

    def forward(self, f: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """(3 toggle logits, a replenish fraction in (0, 1) per input row)."""
        t = {k: torch.from_numpy(f[k]) for k in ("global", "capacity", "product", "input")}
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
        gates = self.gates(pooled)
        n_in = h["input"].shape[0]
        rep = (torch.sigmoid(self.replenish(torch.cat(
            [h["input"], pooled.unsqueeze(0).expand(n_in, -1)], 1)).squeeze(-1))
            if n_in else torch.zeros(0))
        return gates, rep


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
        self.upd = nn.ModuleList([nn.ModuleDict({"producer": _mlp(3 * hidden, hidden),
                                                 "order": _mlp(3 * hidden, hidden)})
                                  for _ in range(rounds)])
        self.score = _mlp(3 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, f: dict[str, Any]) -> list[torch.Tensor]:
        """Per slot, logits over its eligible producers (in `f["elig"][k]` order)."""
        g = self.enc["global"](torch.from_numpy(f["global"]))[0]
        hp = self.enc["producer"](torch.from_numpy(f["producer"])) if len(f["pids"]) else \
            torch.zeros((0, self.hidden))
        ho = self.enc["order"](torch.from_numpy(f["order"])) if len(f["elig"]) else \
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
        out = []
        for k, el in enumerate(f["elig"]):
            if not el:
                out.append(torch.zeros(0))
                continue
            x = torch.cat([ho[k].unsqueeze(0).expand(len(el), -1), hp[el],
                           g.unsqueeze(0).expand(len(el), -1)], 1)
            out.append(self.out(torch.relu(self.score(x))).squeeze(-1))
        return out


def _mean(h: torch.Tensor, hidden: int) -> torch.Tensor:
    return h.mean(0) if h.shape[0] else torch.zeros(hidden)


# --- deployed policies (the GNN alone; the env decode owns legality) ---


def producer_action(model: ProducerGNN, env: "TokenomicsAECEnv", pid: str) -> np.ndarray:
    view = env.system.producer_entity_view(pid)
    f = producer_features(view, producer_context(env, pid))
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
