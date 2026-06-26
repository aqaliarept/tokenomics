"""Learned local producer scorers for EXP-0010C.

The learned component predicts `ProducerLocalScores` from `ProducerEntityView`.
Those scores are then consumed by the deterministic producer planner, which remains
responsible for inventory, funding, recipe, and running-investment feasibility.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from ..domain.common import CapacityType, Product
from ..domain.config import Configuration
from ..domain.producer_planner import ProducerLocalScores, score_producer_entity_view
from ..domain.producer_view import ProducerEntityView
from ..env import TokenomicsAECEnv
from ..eval.baselines import Policy

PRODUCER_SCORER_CHECKPOINT_VERSION = 1


@dataclass
class ProducerScoreSample:
    """One EXP-0010C supervised scorer example."""

    view: ProducerEntityView
    target: ProducerLocalScores


@dataclass
class ProducerScoreTensors:
    """Raw model outputs aligned to the input view's row order."""

    product_urgency: torch.Tensor
    input_value: torch.Tensor
    capacity_bottleneck_value: torch.Tensor
    restoration_roi: torch.Tensor
    upgrade_roi: torch.Tensor
    inventory_expansion_value: torch.Tensor


_GLOBAL_DIM = 8
_CAPACITY_DIM = 15
_PRODUCT_DIM = 18
_INPUT_DIM = 4


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class ProducerScoreModel(nn.Module):
    """Shared local row/edge scorer for producer entity views.

    The model does not use product ids or capacity-type ids as input features. It
    embeds global, product, input, and capacity rows with shared encoders, exchanges
    context only across sparse recipe edges, and emits local economic scores that the
    deterministic planner converts into a feasible command.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.global_encoder = _mlp(_GLOBAL_DIM, hidden)
        self.capacity_encoder = _mlp(_CAPACITY_DIM, hidden)
        self.product_encoder = _mlp(_PRODUCT_DIM, hidden)
        self.input_encoder = _mlp(_INPUT_DIM, hidden)
        four = 4 * hidden
        three = 3 * hidden
        self.product_head = nn.Linear(four, 1)
        self.input_head = nn.Linear(three, 1)
        self.capacity_head = nn.Linear(three, 1)
        self.investment_head = nn.Linear(three, 2)
        self.inventory_head = nn.Linear(three, 1)

    def forward_view(self, view: ProducerEntityView) -> ProducerScoreTensors:
        device = next(self.parameters()).device
        g = self.global_encoder(_tensor([_global_features(view)], device))[0]

        cap_h = self._encode_rows(_capacity_features, view.capacity_rows, self.hidden, device, self.capacity_encoder)
        prod_h = self._encode_rows(_product_features, view.product_rows, self.hidden, device, self.product_encoder)
        input_h = self._encode_rows(_input_features, view.input_rows, self.hidden, device, self.input_encoder)

        prod_cap_ctx = _forward_relation_context(
            view.relations.product_capacity_required, len(view.product_rows), cap_h, self.hidden
        )
        prod_input_ctx = _forward_relation_context(
            view.relations.product_input_required, len(view.product_rows), input_h, self.hidden
        )
        input_prod_ctx = _reverse_relation_context(
            view.relations.product_input_required, len(view.input_rows), prod_h, self.hidden
        )
        cap_prod_ctx = _reverse_relation_context(
            view.relations.product_capacity_required, len(view.capacity_rows), prod_h, self.hidden
        )

        if len(view.product_rows):
            product_scores = self.product_head(torch.cat([
                prod_h,
                prod_cap_ctx,
                prod_input_ctx,
                _expand(g, len(view.product_rows)),
            ], dim=1)).squeeze(-1)
        else:
            product_scores = torch.empty(0, device=device)

        if len(view.input_rows):
            input_scores = self.input_head(torch.cat([
                input_h,
                input_prod_ctx,
                _expand(g, len(view.input_rows)),
            ], dim=1)).squeeze(-1)
        else:
            input_scores = torch.empty(0, device=device)

        if len(view.capacity_rows):
            cap_context = torch.cat([
                cap_h,
                cap_prod_ctx,
                _expand(g, len(view.capacity_rows)),
            ], dim=1)
            cap_scores = self.capacity_head(cap_context).squeeze(-1)
            investment = self.investment_head(cap_context)
            restoration = investment[:, 0].max()
            upgrade = investment[:, 1].max()
        else:
            cap_scores = torch.empty(0, device=device)
            restoration = torch.zeros((), device=device)
            upgrade = torch.zeros((), device=device)

        inventory = self.inventory_head(torch.cat([
            g,
            _mean_or_zero(prod_h, self.hidden, device),
            _mean_or_zero(input_h, self.hidden, device),
        ]).unsqueeze(0)).squeeze()

        return ProducerScoreTensors(
            product_urgency=product_scores,
            input_value=input_scores,
            capacity_bottleneck_value=cap_scores,
            restoration_roi=restoration,
            upgrade_roi=upgrade,
            inventory_expansion_value=inventory,
        )

    @staticmethod
    def _encode_rows(feature_fn, rows, hidden: int, device, encoder: nn.Module) -> torch.Tensor:
        if not rows:
            return torch.empty((0, hidden), device=device)
        return encoder(_tensor([feature_fn(r) for r in rows], device))


def collect_producer_score_samples(
    config: Configuration,
    prod: Policy,
    order: Policy,
    *,
    seeds: Iterable[int],
    horizon: int,
    product_stream: bool = False,
    label_scorer=score_producer_entity_view,
) -> list[ProducerScoreSample]:
    """Collect producer decision states from real rollouts and label them with local scores.

    The behavior policy can be the deterministic entity planner, a search policy, or any
    other producer policy. The collector does not train or mutate the labels; it just
    records the public entity view at the decision point and the supplied local score.
    """
    out: list[ProducerScoreSample] = []
    for seed in seeds:
        env = TokenomicsAECEnv(
            config=config,
            horizon=horizon,
            entity_producer_obs=True,
            product_stream_ordering=product_stream,
        )
        env.reset(seed=seed)
        for agent in env.agent_iter(max_iter=horizon * len(env.possible_agents) * 3 + 100):
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            role, pid = env._role[agent]
            if role == "prod":
                view = copy.deepcopy(env.system.producer_entity_view(pid))
                out.append(ProducerScoreSample(view=view, target=label_scorer(view)))
                env.step(prod(env, agent))
            else:
                env.step(order(env, agent))
    return out


def fit_producer_score_model(
    model: ProducerScoreModel,
    samples: list[ProducerScoreSample],
    *,
    epochs: int,
    lr: float,
    log=None,
    log_every: int = 0,
) -> float:
    """Fit a local scorer to `ProducerLocalScores` labels with full-batch MSE."""
    if not samples:
        raise ValueError("fit_producer_score_model requires at least one sample")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.zeros(())
    for epoch in range(epochs):
        opt.zero_grad()
        loss = torch.stack([_sample_loss(model.forward_view(s.view), s.view, s.target) for s in samples]).mean()
        loss.backward()
        opt.step()
        if log is not None and log_every > 0 and ((epoch + 1) % log_every == 0 or epoch + 1 == epochs):
            log(f"    producer scorer fit epoch {epoch + 1}/{epochs}  mse {float(loss.detach()):.4f}")
    return float(loss.detach())


def score_with_model(model: ProducerScoreModel, view: ProducerEntityView) -> ProducerLocalScores:
    """Run a fitted scorer and map row-aligned tensors back to domain score ids."""
    model.eval()
    with torch.no_grad():
        out = model.forward_view(view)
    return ProducerLocalScores(
        product_urgency=_score_map(
            (r.product for r in view.product_rows), out.product_urgency
        ),
        input_value=_score_map(
            (r.input_product for r in view.input_rows), out.input_value
        ),
        capacity_bottleneck_value=_score_map(
            (r.capacity_type for r in view.capacity_rows), out.capacity_bottleneck_value
        ),
        restoration_roi=_nonnegative_scalar(out.restoration_roi),
        upgrade_roi=_nonnegative_scalar(out.upgrade_roi),
        inventory_expansion_value=_nonnegative_scalar(out.inventory_expansion_value),
    )


def save_producer_score_model(model: ProducerScoreModel, path) -> None:
    """Save a reusable EXP-0010C scorer checkpoint."""
    torch.save(
        {
            "version": PRODUCER_SCORER_CHECKPOINT_VERSION,
            "hidden": model.hidden,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_producer_score_model(path) -> ProducerScoreModel:
    """Load a versioned EXP-0010C scorer checkpoint."""
    ckpt = torch.load(path, weights_only=True)
    if ckpt.get("version") != PRODUCER_SCORER_CHECKPOINT_VERSION:
        raise ValueError(
            f"producer scorer checkpoint version {ckpt.get('version')!r} "
            f"!= {PRODUCER_SCORER_CHECKPOINT_VERSION}"
        )
    model = ProducerScoreModel(hidden=int(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _sample_loss(out: ProducerScoreTensors, view: ProducerEntityView, target: ProducerLocalScores) -> torch.Tensor:
    device = out.inventory_expansion_value.device
    parts = [
        _mse(out.product_urgency, [target.product_urgency.get(r.product, 0.0) for r in view.product_rows], device),
        _mse(out.input_value, [target.input_value.get(r.input_product, 0.0) for r in view.input_rows], device),
        _mse(
            out.capacity_bottleneck_value,
            [target.capacity_bottleneck_value.get(r.capacity_type, 0.0) for r in view.capacity_rows],
            device,
        ),
        (out.restoration_roi - torch.as_tensor(target.restoration_roi, dtype=torch.float32, device=device)) ** 2,
        (out.upgrade_roi - torch.as_tensor(target.upgrade_roi, dtype=torch.float32, device=device)) ** 2,
        (
            out.inventory_expansion_value
            - torch.as_tensor(target.inventory_expansion_value, dtype=torch.float32, device=device)
        ) ** 2,
    ]
    return torch.stack(parts).mean()


def _mse(pred: torch.Tensor, values: list[float], device) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device)
    target = torch.as_tensor(values, dtype=torch.float32, device=device)
    return ((pred - target) ** 2).mean()


def _score_map(keys: Iterable[Product | CapacityType], scores: torch.Tensor) -> dict:
    return {key: max(0.0, float(value)) for key, value in zip(keys, scores.detach().cpu())}


def _nonnegative_scalar(value: torch.Tensor) -> float:
    return max(0.0, float(value.detach().cpu()))


def _tensor(rows: list[list[float]], device) -> torch.Tensor:
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def _expand(row: torch.Tensor, n: int) -> torch.Tensor:
    return row.unsqueeze(0).expand(n, -1)


def _mean_or_zero(rows: torch.Tensor, hidden: int, device) -> torch.Tensor:
    if rows.numel() == 0:
        return torch.zeros(hidden, device=device)
    return rows.mean(dim=0)


def _forward_relation_context(
    edges: dict[tuple[int, int], int],
    n_src: int,
    dst_h: torch.Tensor,
    hidden: int,
) -> torch.Tensor:
    device = dst_h.device
    ctx = torch.zeros((n_src, hidden), device=device)
    weights = torch.zeros((n_src, 1), device=device)
    for (src, dst), amount in edges.items():
        if 0 <= src < n_src and 0 <= dst < len(dst_h):
            w = _edge_weight(amount)
            ctx[src] = ctx[src] + dst_h[dst] * w
            weights[src] = weights[src] + w
    return ctx / weights.clamp_min(1.0)


def _reverse_relation_context(
    edges: dict[tuple[int, int], int],
    n_dst: int,
    src_h: torch.Tensor,
    hidden: int,
) -> torch.Tensor:
    device = src_h.device
    ctx = torch.zeros((n_dst, hidden), device=device)
    weights = torch.zeros((n_dst, 1), device=device)
    for (src, dst), amount in edges.items():
        if 0 <= src < len(src_h) and 0 <= dst < n_dst:
            w = _edge_weight(amount)
            ctx[dst] = ctx[dst] + src_h[src] * w
            weights[dst] = weights[dst] + w
    return ctx / weights.clamp_min(1.0)


def _edge_weight(amount: int | float) -> float:
    return math.log1p(max(0.0, float(amount)))


def _qty(value: int | float) -> float:
    return math.log1p(max(0.0, float(value)))


def _global_features(view: ProducerEntityView) -> list[float]:
    g = view.globals
    return [
        _qty(0.0 if math.isnan(float(g.cut_off_price)) else g.cut_off_price),
        _qty(g.inventory_units),
        _qty(g.inventory_limit),
        _qty(g.inventory_headroom),
        _qty(g.outstanding_inbound_units),
        *[float(x) for x in g.active_investments],
    ]


def _capacity_features(r) -> list[float]:
    return [
        float(r.owned),
        _qty(r.current),
        _qty(r.max),
        _qty(r.headroom),
        float(r.ratio),
        _qty(r.degradation),
        _qty(r.reserved),
        _qty(r.previous_requested_capacity),
        _qty(r.free),
        float(r.reserved_ratio),
        _qty(r.restore_effect),
        _qty(r.restore_cost),
        _qty(r.upgrade_effect),
        _qty(r.upgrade_cost),
        _qty(r.demand_pressure),
    ]


def _product_features(r) -> list[float]:
    return [
        float(r.eligible),
        _qty(r.open_demand_qty),
        _qty(r.open_demand_tokens),
        _qty(r.product_depth),
        _qty(r.recipe_input_count),
        _qty(r.recipe_input_units),
        _qty(r.recipe_capacity_type_count),
        _qty(r.recent_routed_qty),
        _qty(r.recent_completed_qty),
        _qty(r.recent_rejected_qty),
        _qty(r.recent_completion_latency),
        _qty(r.active_reservation_qty),
        _qty(r.capacity_makeable_qty),
        _qty(r.input_makeable_qty if math.isfinite(float(r.input_makeable_qty)) else 0.0),
        _qty(r.complete_qty_now),
        float(r.capacity_bottleneck_ratio),
        float(r.input_bottleneck_ratio),
        float(r.bottleneck_ratio),
    ]


def _input_features(r) -> list[float]:
    return [
        _qty(r.inventory),
        _qty(r.outstanding_inbound),
        _qty(r.blocked_reservation_qty),
        _qty(r.downstream_demand_pressure),
    ]
