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
- **Producer turn heads (ADR-0062, optional `TurnHeads`):** in the fund turn a score per funding request; in the
  accept turn an accept logit and a priority per routed bid, and a priority per carried reservation and pending
  claim. Request, bid and claim nodes are linked to the capacity and input rows they use, and read the trunk's
  embeddings of the producer's own view.
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


# --- ADR-0062 turn features (the producer's own requests, bids and claims) ---

FUND_KINDS = ("restoration", "replenishment", "upgrade", "inventory_expansion")
FUND_DIM, BID_DIM, CLAIM_DIM = 9, 12, 11
# v15 (supervisor 2026-11-07, DIAGNOSTICS 38 / EXP-0121): linear, scale-free copies of each item's own-state
# quantities, appended after the log features: fund +1 (target / budget); bid and claim +3 (capacity / the
# producer's free new-work budget and / its current capacity, at the item's binding type; price / the turn's
# highest item price). No cumulative quantity: the heads still learn the running sum.
LINEAR_EXTRA = {"fund": 1, "bid": 3, "claim": 3}
FRAC_CLIP = 10.0


def turn_dims(linear: bool = False) -> dict[str, int]:
    base = {"fund": FUND_DIM, "bid": BID_DIM, "claim": CLAIM_DIM}
    return {k: v + (LINEAR_EXTRA[k] if linear else 0) for k, v in base.items()}


def _frac(num: float, den: float) -> float:
    return min(FRAC_CLIP, num / den) if den > 0 else (FRAC_CLIP if num > 0 else 0.0)


def _binding(caps: Mapping[Any, float], pool: Mapping[Any, float]) -> float:
    """max over the item's capacity types of need / pool (the binding type; one type in every current world)."""
    return max((_frac(float(c), float(pool.get(t, 0))) for t, c in caps.items()), default=0.0)


def _row_index(view: ProducerEntityView) -> tuple[dict[Any, int], dict[Any, int]]:
    return ({r.capacity_type: i for i, r in enumerate(view.capacity_rows)},
            {r.input_product: i for i, r in enumerate(view.input_rows)})


def _links(n: int, items: Sequence[tuple[Sequence[Any], Sequence[Any]]], cap: dict[Any, int],
           inp: dict[Any, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(node, capacity row) and (node, input row) edges for each node's (types, inputs)."""
    ec = [(k, cap[t]) for k, (types, _ins) in enumerate(items) for t in types if t in cap]
    ei = [(k, inp[g]) for k, (_types, ins) in enumerate(items) for g in ins if g in inp]
    return ec, ei


def fund_features(p: Any, reqs: Sequence[Any], budget: float, view: ProducerEntityView,
                  linear: bool = False) -> dict[str, Any]:
    """One row per funding request (the fund turn's slots, today's order): kind, target, quantity,
    the product's stock and inbound (replenishment), the producer's budget; linked to the capacity
    row it restores / upgrades or the input row it buys."""
    from ..env.producer_turns import fund_kind

    cap, inp = _row_index(view)
    rows, items = [], []
    for o in reqs:
        kind = fund_kind(o)
        target = o.funding_target()
        rep_ = kind == "replenishment"
        rows.append([float(kind == k) for k in FUND_KINDS] + [
            _q(budget if target is None else target), _q(o.quantity),
            _q(p.inventory.get(o.product, 0) if rep_ else 0),
            _q(p.outstanding_inbound.get(o.product, 0) if rep_ else 0), _q(budget)]
            + ([1.0 if target is None else _frac(float(target), float(budget))] if linear else []))
        types = ([p.restoration.type] if kind == "restoration" else
                 [p.upgrade.type] if kind == "upgrade" else [])
        items.append((types, [o.product] if rep_ else []))
    ec, ei = _links(len(rows), items, cap, inp)
    return {"fund": _arr(rows, turn_dims(linear)["fund"]), "fund_cap": ec, "fund_in": ei}


def accept_features(p: Any, lists: Any, view: ProducerEntityView, linear: bool = False
                    ) -> dict[str, Any]:
    """Bid rows (this cycle's routed bids, price-descending) and claim rows (carried reservations,
    then pending claims, in default order); each linked to the capacity and input rows it uses."""
    from ..env.producer_turns import _cover, kind_onehot

    cap, inp = _row_index(view)
    budget = p._new_work_budget(lists)
    items_all = [b for b in lists.bids] + [r.bid for r in (*lists.reservations, *lists.pending)]
    top = max((x.capacity_unit_price() for x in items_all), default=0.0)

    def lin(b: Any) -> list[float]:
        if not linear:
            return []
        return [_binding(b.capacities, budget), _binding(b.capacities, p.capacities),
                _frac(float(b.capacity_unit_price()), float(top))]

    bids, bitems = [], []
    for b in lists.bids:
        bids.append([_q(b.capacity_unit_price()), _q(b.quantity), _q(b.total_capacity()),
                     -1.0 if b.remaining_ttl is None else float(b.remaining_ttl),
                     float(b.is_composite()), _cover(p, b), float(p._fits(budget, b.capacities)),
                     _q(sum(budget.values())), _q(b.tokens), *kind_onehot(b), *lin(b)])
        bitems.append((list(b.capacities), list(b.inputs)))
    claims, citems = [], []
    for held, rs in ((1.0, lists.reservations), (0.0, lists.pending)):
        for r in rs:
            claims.append([held, 1.0 - held, _q(r.bid.capacity_unit_price()), _q(r.bid.quantity),
                           _q(r.bid.total_capacity()), float(r.cycles_waited),
                           float(p._has_inputs(r.bid)), float(p.reservation_ttl - r.cycles_waited),
                           *kind_onehot(r.bid), *lin(r.bid)])
            citems.append((list(r.bid.capacities), list(r.bid.inputs)))
    bc, bi = _links(len(bids), bitems, cap, inp)
    cc, ci = _links(len(claims), citems, cap, inp)
    d = turn_dims(linear)
    return {"bid": _arr(bids, d["bid"]), "bid_cap": bc, "bid_in": bi,
            "claim": _arr(claims, d["claim"]), "claim_cap": cc, "claim_in": ci,
            "n_res": len(lists.reservations)}


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
                 features: str = LATEST_FEATURES, capacity_extra: int = 0) -> None:
        """`capacity_extra` > 0: the last `capacity_extra` capacity columns enter through their own
        bias-free layer added to the first layer of a narrower encoder (`extend_to_v4`: EXP-0100's
        16-column layer kept as it is, the v4 columns at zero weight, so the start is bit-identical;
        a widened layer would re-order the float sums)."""
        super().__init__()
        self.hidden, self.rounds, self.features = hidden, rounds, features
        self.capacity_extra = capacity_extra
        cap = CAPACITY_DIM[features]
        self.cap_extra = nn.Linear(capacity_extra, hidden, bias=False) if capacity_extra else None
        self.enc = nn.ModuleDict({"global": _mlp(13, hidden),
                                  "capacity": _mlp(cap - capacity_extra, hidden),
                                  "product": _mlp(18, hidden), "input": _mlp(6, hidden)})
        self.norm = FeatureNorm({"global": 13, "capacity": cap, "product": 18, "input": 6},
                                PRODUCER_LOG1P)
        self.upd = nn.ModuleList([nn.ModuleDict({
            "product": _mlp(3 * hidden, hidden), "capacity": _mlp(2 * hidden, hidden),
            "input": _mlp(2 * hidden, hidden)}) for _ in range(rounds)])
        self.gates = nn.Linear(4 * hidden, 3)
        self.replenish = nn.Linear(5 * hidden, 1)
        self.turn: TurnHeads | None = None  # ADR-0062 (`add_turn_heads`)

    def add_turn_heads(self, set_layers: int = 0, bounded: bool = False, width: int | None = None,
                       linear: bool = False, ordered: bool = False) -> None:
        """Attach fresh fund / accept turn heads (the trunk and investment heads untouched)."""
        self.turn = TurnHeads(self.hidden, set_layers, bounded, width, linear, ordered)

    def fund_scores(self, f: dict[str, Any], t: dict[str, Any], trunk: Trunk | None = None
                    ) -> torch.Tensor:
        """One score per funding request row of `t = fund_features(...)`; `trunk` = a cached
        `_trunk(f)` (the fund and accept turns of one producer-cycle share it, `TurnCache`)."""
        assert self.turn is not None
        h, pooled = trunk if trunk is not None else self._trunk(f)
        return self.turn.turn_fund(t, h, pooled)

    def accept_outputs(self, f: dict[str, Any], t: dict[str, Any], trunk: Trunk | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(accept logit per bid, priority per bid, priority per claim) for `accept_features`."""
        assert self.turn is not None
        h, pooled = trunk if trunk is not None else self._trunk(f)
        return self.turn.turn_accept(t, h, pooled)

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

    def _encode(self, k: str, v: torch.Tensor) -> torch.Tensor:
        if not v.shape[0]:
            return torch.zeros((0, self.hidden))
        if k != "capacity" or self.cap_extra is None:
            return self.enc[k](v)
        n = v.shape[1] - self.capacity_extra
        first, act, second = self.enc[k]
        return second(act(first(v[:, :n]) + self.cap_extra(v[:, n:])))

    def _trunk(self, f: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        t = {k: self.norm(k, f[k]) for k in ("global", "capacity", "product", "input")}
        h = {k: self._encode(k, v) for k, v in t.items()}
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


Trunk = tuple[dict[str, torch.Tensor], torch.Tensor]


class TurnHeads(nn.Module):
    """ADR-0062 heads over request / bid / claim nodes: each node's encoding, the mean of its
    linked capacity and input rows' trunk embeddings, and the pooled view embedding.

    v14 (`set_layers` > 0, DIAGNOSTICS 38): self-attention layers over the turn's items (the fund
    requests; the bids and claims together) after that encoding, so an item's output can depend on
    the other items of its turn (the auction's cumulative rule is not hand-built; masking pads the
    batched sets). `bounded`: priorities and fund scores are B·tanh(·/B), so they never reach the
    action clip. `set_layers` = 0 and `bounded` = False is the original head, unchanged."""

    def __init__(self, hidden: int = HIDDEN, set_layers: int = 0, bounded: bool = False,
                 width: int | None = None, linear: bool = False, ordered: bool = False) -> None:
        """`hidden` = the trunk's embedding size (its rows and pooled view); `width` = the heads' own
        size (default `hidden`; v15's capacity probe)."""
        super().__init__()
        w = hidden if width is None else width
        self.hidden, self.width, self.set_layers, self.bounded = hidden, w, set_layers, bounded
        self.pre_bound: list[torch.Tensor] | None = None
        self.linear = linear
        self.dims = turn_dims(linear)
        self.enc = nn.ModuleDict({k: _mlp(v, w) for k, v in self.dims.items()})
        self.norm = FeatureNorm(dict(self.dims), {})
        self.mix = nn.ModuleDict({k: _mlp(w + 6 * hidden, w) for k in ("fund", "bid", "claim")})
        self.out = nn.ModuleDict({"fund": nn.Linear(w, 1), "bid": nn.Linear(w, 2),
                                  "claim": nn.Linear(w, 1)})
        # Order-aware accept (user decision 2026-11-07, DIAGNOSTICS 38): an additive recurrent pass over the
        # turn's bids and claims in the order of the head's own (bounded, detached) priority. The state is a
        # running sum, s <- s + U h(item, s), which can represent the auction's cumulative capacity exactly;
        # the accept logit of a bid is read from h. No gradient flows through the sort: the same student order
        # is used in training and at deploy (DIAGNOSTICS 37: the deployed path is the trained path).
        self.ordered = ordered
        if ordered:
            self.seq = _mlp(2 * w, w)
            self.inc = nn.Linear(w, w)
            self.seq_out = nn.Linear(w, 1)
        if set_layers:
            def enc() -> nn.TransformerEncoder:
                layer = nn.TransformerEncoderLayer(w, 4, 2 * w, dropout=0.0,
                                                   batch_first=True, norm_first=True)
                return nn.TransformerEncoder(layer, set_layers, enable_nested_tensor=False)
            self.att = nn.ModuleDict({"fund": enc(), "accept": enc()})

    def config(self) -> dict[str, Any]:
        return {"set_layers": self.set_layers, "bounded": self.bounded,
                **({"width": self.width} if self.width != self.hidden else {}),
                **({"linear": True} if self.linear else {}),
                **({"ordered": True} if self.ordered else {})}

    def node_embed(self, kind: str, x: np.ndarray, ec: Sequence[tuple[int, int]],
                   ei: Sequence[tuple[int, int]], h_cap: torch.Tensor, h_in: torch.Tensor,
                   pooled_per_node: torch.Tensor) -> torch.Tensor:
        """The per-item embedding before the set layers (rows of `x`; edges into h_cap / h_in)."""
        e = self.enc[kind](self.norm(kind, x))
        n = e.shape[0]
        cc = _mean_ctx(ec, n, h_cap, True)
        ci = _mean_ctx(ei, n, h_in, True)
        return torch.relu(self.mix[kind](torch.cat([e, cc, ci, pooled_per_node], 1)))

    def attend(self, group: str, emb: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        """Self-attention within each turn's items (`turn` = each row's turn id); identity without
        set layers. Rows are padded per turn and masked; no positional encoding (a set)."""
        if not self.set_layers or not emb.shape[0]:
            return emb
        ids, compact = torch.unique(turn, return_inverse=True)
        n_t = ids.shape[0]
        counts = torch.bincount(compact, minlength=n_t)
        order = torch.argsort(compact, stable=True)
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(emb.shape[0]) - start[compact[order]]
        pad = torch.zeros((n_t, int(counts.max()), emb.shape[1]))
        pad[compact[order], pos] = emb[order]
        mask = torch.arange(pad.shape[1]).unsqueeze(0) >= counts.unsqueeze(1)
        att = self.att[group](pad, src_key_padding_mask=mask)
        out = torch.empty_like(emb)
        out[order] = att[compact[order], pos]
        return out

    def _bound(self, x: torch.Tensor) -> torch.Tensor:
        """B·tanh(x / B). In float32 it rounds to exactly ±B once |x| > ~9B, so saturated scores tie;
        the fit keeps x small with a penalty on the pre-tanh values, collected in `pre_bound` when
        that is a list (m6_turn_imitation)."""
        if self.pre_bound is not None:
            self.pre_bound.append(x)
        return ACTION_LOGIT_BOUND * torch.tanh(x / ACTION_LOGIT_BOUND) if self.bounded else x

    def fund_out(self, emb: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        """One score per request (embeddings from `node_embed("fund", ...)`)."""
        return self._bound(self.out["fund"](self.attend("fund", emb, turn)).squeeze(-1))

    def accept_out(self, eb: torch.Tensor, tb: torch.Tensor, ec: torch.Tensor, tc: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(accept logit per bid, priority per bid, priority per claim); bids and claims of a turn
        attend to each other."""
        a = self.attend("accept", torch.cat([eb, ec]), torch.cat([tb, tc]))
        b, c = self.out["bid"](a[:eb.shape[0]]), self.out["claim"](a[eb.shape[0]:]).squeeze(-1)
        bp, cp = self._bound(b[:, 1]), self._bound(c)
        if not self.ordered:
            return b[:, 0], bp, cp
        h = self.ordered_pass(a, torch.cat([tb, tc]), torch.cat([bp, cp]).detach())
        return self.seq_out(h[:eb.shape[0]]).squeeze(-1), bp, cp

    def ordered_pass(self, emb: torch.Tensor, turn: torch.Tensor, priority: torch.Tensor) -> torch.Tensor:
        """Per item, h from (its embedding, the running state of the items ranked above it in its turn);
        items visited in descending `priority` within each turn (ties: input order). Padded steps leave the
        state unchanged."""
        if not emb.shape[0]:
            return torch.zeros((0, self.width))
        o1 = torch.argsort(-priority, stable=True)
        order = o1[torch.argsort(turn[o1], stable=True)]  # turn-major, priority-descending
        ids, compact = torch.unique(turn, return_inverse=True)
        counts = torch.bincount(compact, minlength=ids.shape[0])
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(emb.shape[0]) - start[compact[order]]
        n_t, length = ids.shape[0], int(counts.max())
        pad = torch.zeros((n_t, length, emb.shape[1]))
        pad[compact[order], pos] = emb[order]
        live = (torch.arange(length).unsqueeze(0) < counts.unsqueeze(1)).float().unsqueeze(-1)
        state = torch.zeros((n_t, self.width))
        hs = []
        for k in range(length):
            hk = torch.relu(self.seq(torch.cat([pad[:, k], state], 1)))
            state = state + live[:, k] * self.inc(hk)
            hs.append(hk)
        hpad = torch.stack(hs, 1)
        out = torch.empty((emb.shape[0], self.width))
        out[order] = hpad[compact[order], pos]
        return out

    def _embed_turn(self, kind: str, f: dict[str, Any], h: dict[str, torch.Tensor],
                    pooled: torch.Tensor) -> torch.Tensor:
        x = f[kind]
        if not x.shape[0]:
            return torch.zeros((0, self.width))
        return self.node_embed(kind, x, f[f"{kind}_cap"], f[f"{kind}_in"], h["capacity"],
                               h["input"], pooled.unsqueeze(0).expand(x.shape[0], -1))

    def turn_fund(self, f: dict[str, Any], h: dict[str, torch.Tensor], pooled: torch.Tensor
                  ) -> torch.Tensor:
        e = self._embed_turn("fund", f, h, pooled)
        return self.fund_out(e, torch.zeros(e.shape[0], dtype=torch.long))

    def turn_accept(self, f: dict[str, Any], h: dict[str, torch.Tensor], pooled: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eb, ec = self._embed_turn("bid", f, h, pooled), self._embed_turn("claim", f, h, pooled)
        z = lambda e: torch.zeros(e.shape[0], dtype=torch.long)  # noqa: E731
        return self.accept_out(eb, z(eb), ec, z(ec))

    def node_out(self, kind: str, f: dict[str, Any], h: dict[str, torch.Tensor],
                 pooled: torch.Tensor) -> torch.Tensor:
        """The original per-node output (no set layers, unbounded); kept for v12 / v13 readers."""
        x = f[kind]
        if not x.shape[0]:
            return torch.zeros((0, self.out[kind].out_features))
        return self.out[kind](self._embed_turn(kind, f, h, pooled))


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


def _turn_view(env: "TokenomicsAECEnv", pid: str) -> ProducerEntityView:
    """The producer's entity view at a turn (built directly: the system's accessor is for the
    investment phase only); the same own-state view, now with this cycle's commands placed."""
    from ..domain.producer_view import build_producer_entity_view

    s = env.system
    return build_producer_entity_view(s.producing_agents[pid], s.telemetry, s.process_sheets,
                                      s.orders)


class TurnCache:
    """One trunk pass per producer-cycle for the fund and accept turns. Between a producer's fund
    turn and its accept turn nothing its entity view reads changes: funding and routing touch orders
    and bids, which the view does not read (measured equal in 180 / 180 producer-cycles, 2026-11-07;
    `tests/test_producer_turns.py` pins it). The investment turn's view differs: the producer's own
    action raises its inbound, sets its active investments and lowers its headroom, so it gets its own
    pass. Computed lazily, at the first of the two turns that has something to score."""

    def __init__(self) -> None:
        self.cycle: tuple[int, int] | None = None  # (env, cycle) the entries belong to
        self.entries: dict[str, tuple[ProducerEntityView, dict[str, Any], Trunk]] = {}
        self.passes = 0

    def get(self, model: ProducerGNN, env: "TokenomicsAECEnv", pid: str
            ) -> tuple[ProducerEntityView, dict[str, Any], Trunk]:
        cyc = (id(env), int(env.system.cycle_counter))
        if cyc != self.cycle:
            self.cycle, self.entries = cyc, {}
        if pid not in self.entries:
            view = _turn_view(env, pid)
            f = producer_features(view, producer_context(env, pid), model.features)
            with torch.no_grad():
                self.entries[pid] = (view, f, model._trunk(f))
            self.passes += 1
        return self.entries[pid]


def fund_action(model: ProducerGNN, env: "TokenomicsAECEnv", pid: str,
                cache: TurnCache | None = None) -> np.ndarray:
    from ..env.producer_turns import fund_slots

    s = env.system
    reqs = s.funding_requests(pid)
    out = np.full(fund_slots(env.spec_), -ACTION_LOGIT_BOUND, np.float32)
    if not reqs:
        return out
    view, f, trunk = (cache or TurnCache()).get(model, env, pid)
    t = fund_features(s.producing_agents[pid], reqs, s.funding_budget(pid), view, model.turn.linear)
    with torch.no_grad():
        sc = model.fund_scores(f, t, trunk).numpy()
    n = min(len(sc), out.shape[0])
    out[:n] = np.clip(sc[:n], -ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND)
    return out


def accept_action(model: ProducerGNN, env: "TokenomicsAECEnv", pid: str,
                  cache: TurnCache | None = None) -> np.ndarray:
    """Greedy: accept iff the logit > 0; priorities as scored (clipped to the action bound)."""
    from ..domain.producing_agent import TurnDecision
    from ..env.producer_turns import encode_accept_action

    s = env.system
    p = s.producing_agents[pid]
    lists = s.producer_turn_lists(pid)
    if not (lists.bids or lists.reservations or lists.pending):
        return encode_accept_action(TurnDecision((), (), (), ()))
    view, f, trunk = (cache or TurnCache()).get(model, env, pid)
    t = accept_features(p, lists, view, model.turn.linear)
    with torch.no_grad():
        acc, bp, cp = (x.numpy() for x in model.accept_outputs(f, t, trunk))
    clip = lambda v: tuple(float(x) for x in np.clip(v, -ACTION_LOGIT_BOUND, ACTION_LOGIT_BOUND))
    nr = t["n_res"]
    return encode_accept_action(TurnDecision(tuple(bool(x > 0) for x in acc), clip(bp),
                                             clip(cp[:nr]), clip(cp[nr:])))


def gnn_turn_policies(model: ProducerGNN, cache: TurnCache | None = None
                      ) -> tuple[Policy, Policy]:
    """(fund policy, accept policy) of a producer GNN with turn heads (ADR-0062), sharing one trunk
    pass per producer-cycle (`TurnCache`)."""
    model.eval()
    c = cache if cache is not None else TurnCache()
    return (lambda env, agent: fund_action(model, env, env._role[agent][1], c),
            lambda env, agent: accept_action(model, env, env._role[agent][1], c))


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
