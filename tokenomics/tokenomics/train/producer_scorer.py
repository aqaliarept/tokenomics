"""Learned local producer scorers for EXP-0010C.

The learned component predicts `ProducerLocalScores` from `ProducerEntityView`.
Those scores are then consumed by the deterministic producer planner, which remains
responsible for inventory, funding, recipe, and running-investment feasibility.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from torch import nn

from ..domain.common import CapacityType, Product
from ..domain.config import Configuration
from ..domain.producer_planner import ProducerLocalScores, score_producer_entity_view
from ..domain.producer_view import ProducerEntityView
from ..env import TokenomicsAECEnv
from ..eval.baselines import Policy

# v2: investment heads emit BCE-trained gate logits (was v1 ROI-MSE regression) + the
# checkpoint carries per-gate probability thresholds (EXP-0013).
PRODUCER_SCORER_CHECKPOINT_VERSION = 2

# Named feature schema for the portable load gate (step 13). These names mirror the
# `_*_features` row builders below in order; the load gate compares them by name, so a
# feature rename/reorder rejects while a wider/relabeled topology (absent here) does not.
_PRODUCER_GLOBAL_FEATURES = (
    "cut_off_price", "inventory_units", "inventory_limit", "inventory_headroom",
    "outstanding_inbound_units", "active_restoration", "active_upgrade",
    "active_inventory_expansion",
)
_PRODUCER_CAPACITY_FEATURES = (
    "owned", "current", "max", "headroom", "ratio", "degradation", "reserved",
    "previous_requested_capacity", "free", "reserved_ratio", "restore_effect",
    "restore_cost", "upgrade_effect", "upgrade_cost", "demand_pressure",
)
_PRODUCER_PRODUCT_FEATURES = (
    "eligible", "open_demand_qty", "open_demand_tokens", "product_depth",
    "recipe_input_count", "recipe_input_units", "recipe_capacity_type_count",
    "recent_routed_qty", "recent_completed_qty", "recent_rejected_qty",
    "recent_completion_latency", "active_reservation_qty", "capacity_makeable_qty",
    "input_makeable_qty", "complete_qty_now", "capacity_bottleneck_ratio",
    "input_bottleneck_ratio", "bottleneck_ratio",
)
_PRODUCER_INPUT_FEATURES = (
    "inventory", "outstanding_inbound", "blocked_reservation_qty",
    "downstream_demand_pressure",
)


def producer_portable_schema(*, use_edges: bool = True) -> "PortableSchema":
    """The current portable-load schema for the producer scorer (step 13). Topology-free:
    named row features + entity/relation types + norm/decoder/architecture versions, no
    product/capacity ids or counts.

    `use_edges=False` = the step-14 no-edge ablation variant: same features/entities/decoder,
    but the recipe relations are unused, so the architecture (and hence load gate) differs."""
    from .portable_schema import PortableSchema

    return PortableSchema(
        schema_version=1,
        entity_feature_names={
            "global": _PRODUCER_GLOBAL_FEATURES,
            "capacity": _PRODUCER_CAPACITY_FEATURES,
            "product": _PRODUCER_PRODUCT_FEATURES,
            "input": _PRODUCER_INPUT_FEATURES,
        },
        entity_types=("global", "capacity", "product", "input"),
        relation_types=("product_capacity_required", "product_input_required"),
        normalization_version="log1p-nonneg-v1",
        decoder_version="plan_producer_command-v2-gates",
        architecture_version=(
            "producer-row-edge-scorer-v2" if use_edges else "producer-row-noedge-scorer-v2"
        ),
    )


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


@dataclass
class _ProducerScoreBatch:
    """Training-only tensor batch for views with the same sparse relation layout."""

    global_x: torch.Tensor
    capacity_x: torch.Tensor
    product_x: torch.Tensor
    input_x: torch.Tensor
    product_capacity_weight: torch.Tensor
    product_input_weight: torch.Tensor
    product_target: torch.Tensor
    input_target: torch.Tensor
    capacity_target: torch.Tensor
    restoration_target: torch.Tensor
    upgrade_target: torch.Tensor
    inventory_target: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.global_x.shape[0])


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

    def __init__(self, hidden: int = 64, *, use_edges: bool = True) -> None:
        super().__init__()
        self.hidden = hidden
        # Step-14 ablation #2: `use_edges=False` = per-row/no-edge baseline. The recipe
        # relation contexts are zeroed, so each row is scored from its own features + global
        # only. Head input dims are unchanged (edge slots become zeros), so the variant still
        # loads on variable topologies — it isolates whether the sparse edges carry transfer.
        self.use_edges = use_edges
        # Investment gates (upgrade / inventory expansion) are LEARNED BINARY DECISIONS,
        # not thresholded ROI regressions: the heads emit logits trained with BCE against
        # the planner's actual gate booleans, and deployment fires a gate when its
        # probability clears a per-gate threshold calibrated post-fit on command agreement
        # (EXP-0013). Restoration stays planner-owned (the planner ignores learned
        # restoration scores), so it gets no learned gate. Defaults = 0.5 (argmax).
        self.upgrade_threshold = 0.5
        self.inventory_expansion_threshold = 0.5
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

        if self.use_edges:
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
        else:
            prod_cap_ctx = torch.zeros((len(view.product_rows), self.hidden), device=device)
            prod_input_ctx = torch.zeros((len(view.product_rows), self.hidden), device=device)
            input_prod_ctx = torch.zeros((len(view.input_rows), self.hidden), device=device)
            cap_prod_ctx = torch.zeros((len(view.capacity_rows), self.hidden), device=device)

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


def flat_model_pads_for(view: ProducerEntityView) -> dict[str, int]:
    """Pad caps sized to one view's entity counts — the fixed input dims a `ProducerFlatModel`
    is locked to at construction."""
    return {
        "cap_max": len(view.capacity_rows),
        "prod_max": len(view.product_rows),
        "input_max": len(view.input_rows),
    }


def _flatten_padded(rows: list[list[float]], n_max: int, dim: int) -> list[float]:
    flat: list[float] = []
    for r in rows:
        flat += r
    flat += [0.0] * (dim * (n_max - len(rows)))
    return flat


class ProducerFlatModel(nn.Module):
    """Step-14 ablation #2 baseline: a flat MLP over one fixed-size padded feature vector.

    NON-PORTABLE BY CONSTRUCTION. The whole view (global + every capacity/product/input row)
    is concatenated into one fixed-length vector padded to `(cap_max, prod_max, input_max)`,
    fed through a single MLP, and read out as a fixed-size score vector sliced back to the
    actual row counts. A topology WIDER than the pad it was built for cannot be ingested
    (`forward_view` raises) — the exact transfer the row/edge scorer handles by just adding
    rows. Same `forward_view` -> `ProducerScoreTensors` interface as `ProducerScoreModel`, so
    it deploys through `score_with_model` + the deterministic planner unchanged."""

    def __init__(self, *, cap_max: int, prod_max: int, input_max: int, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.cap_max = cap_max
        self.prod_max = prod_max
        self.input_max = input_max
        self.upgrade_threshold = 0.5
        self.inventory_expansion_threshold = 0.5
        in_dim = _GLOBAL_DIM + cap_max * _CAPACITY_DIM + prod_max * _PRODUCT_DIM + input_max * _INPUT_DIM
        self.trunk = _mlp(in_dim, hidden)
        # prod_max + input_max + cap_max row scores, then restoration / upgrade / inventory
        self.head = nn.Linear(hidden, prod_max + input_max + cap_max + 3)

    def forward_view(self, view: ProducerEntityView) -> ProducerScoreTensors:
        device = next(self.parameters()).device
        n_cap, n_prod, n_input = len(view.capacity_rows), len(view.product_rows), len(view.input_rows)
        if n_cap > self.cap_max or n_prod > self.prod_max or n_input > self.input_max:
            raise ValueError(
                f"flat model pad ({self.cap_max},{self.prod_max},{self.input_max}) cannot ingest "
                f"topology ({n_cap},{n_prod},{n_input}) — non-portable by construction"
            )
        flat = (
            _global_features(view)
            + _flatten_padded([_capacity_features(r) for r in view.capacity_rows], self.cap_max, _CAPACITY_DIM)
            + _flatten_padded([_product_features(r) for r in view.product_rows], self.prod_max, _PRODUCT_DIM)
            + _flatten_padded([_input_features(r) for r in view.input_rows], self.input_max, _INPUT_DIM)
        )
        out = self.head(self.trunk(_tensor([flat], device)))[0]
        i = 0
        product_scores = out[i:i + self.prod_max][:n_prod]; i += self.prod_max
        input_scores = out[i:i + self.input_max][:n_input]; i += self.input_max
        cap_scores = out[i:i + self.cap_max][:n_cap]; i += self.cap_max
        return ProducerScoreTensors(
            product_urgency=product_scores,
            input_value=input_scores,
            capacity_bottleneck_value=cap_scores,
            restoration_roi=out[i],
            upgrade_roi=out[i + 1],
            inventory_expansion_value=out[i + 2],
        )


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
    device = next(model.parameters()).device
    batches = _prepare_score_batches(samples, device)
    total_samples = sum(batch.size for batch in batches)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.zeros((), device=device)
    for epoch in range(epochs):
        opt.zero_grad()
        loss_sum = torch.zeros((), device=device)
        for batch in batches:
            loss_sum = loss_sum + _batch_loss(model, batch)
        loss = loss_sum / total_samples
        loss.backward()
        opt.step()
        if log is not None and log_every > 0 and ((epoch + 1) % log_every == 0 or epoch + 1 == epochs):
            log(f"    producer scorer fit epoch {epoch + 1}/{epochs}  mse {float(loss.detach()):.4f}")
    return float(loss.detach())


def producer_gate_probs(model: ProducerScoreModel, view: ProducerEntityView) -> dict[str, float]:
    """Sigmoid gate probabilities for the two learned investment decisions.

    The upgrade / inventory-expansion heads emit logits (BCE-trained against the planner's
    gate booleans); this exposes their probabilities for calibration and command-level
    eval. Restoration is planner-owned and has no learned gate."""
    if model.training:
        model.eval()
    with torch.inference_mode():
        out = model.forward_view(view)
    return {
        "upgrade": _sigmoid(out.upgrade_roi),
        "inventory_expansion": _sigmoid(out.inventory_expansion_value),
    }


def score_with_model(
    model: ProducerScoreModel,
    view: ProducerEntityView,
    upgrade_threshold: float | None = None,
    inventory_expansion_threshold: float | None = None,
    *,
    planner_gates: bool = False,
) -> ProducerLocalScores:
    """Run a fitted scorer and map row-aligned tensors back to domain score ids.

    Replenishment scores (`product_urgency`, `input_value`, `capacity_bottleneck_value`)
    are learned regressions, unchanged. The investment decisions are LEARNED BINARY GATES:
    each head's logit becomes a probability, and the gate emits a `> 0` "do it" score iff
    that probability clears its threshold (per-gate, calibrated; defaults to the model's).
    This stops the deterministic planner reading regression noise as a spend signal
    (EXP-0013). Restoration is left 0.0 — the planner owns it and ignores this field.

    `planner_gates=True` hands the upgrade / inventory-expansion gate decisions back to
    the deterministic planner (`score_producer_entity_view`) while keeping the learned
    continuous scores. The learned investment gates generalize poorly out-of-family
    (EXP-0017); planner-owned gates recover the bar there until gate training covers those
    states. The continuous heads are untouched."""
    if model.training:
        model.eval()
    with torch.inference_mode():
        out = model.forward_view(view)
    up_thr = model.upgrade_threshold if upgrade_threshold is None else upgrade_threshold
    inv_thr = (
        model.inventory_expansion_threshold
        if inventory_expansion_threshold is None
        else inventory_expansion_threshold
    )
    if planner_gates:
        planner = score_producer_entity_view(view)
        upgrade_gate = planner.upgrade_roi
        inventory_gate = planner.inventory_expansion_value
    else:
        upgrade_gate = 1.0 if _sigmoid(out.upgrade_roi) >= up_thr else 0.0
        inventory_gate = 1.0 if _sigmoid(out.inventory_expansion_value) >= inv_thr else 0.0
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
        restoration_roi=0.0,
        upgrade_roi=upgrade_gate,
        inventory_expansion_value=inventory_gate,
    )


def eval_producer_scorer(
    model: ProducerScoreModel,
    depth: int,
    seeds,
    eval_seeds: int,
    horizon: int,
    log=lambda *a, **k: None,
) -> list[dict]:
    """Score the learned scorer+planner producing policy against the deterministic
    `entity_planner` teacher on held-out scale topologies (which vary producer count via
    depth) — the P1 fidelity + producer-count transfer check. Orderer fixed to
    `proportional`. Producer scores clone the planner's own labels, so the bar is the
    planner itself: ratio ~1.0 means the learned scorer reproduces the planner's value,
    and running at an unseen producer count proves the id-free contract transfers."""
    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from ..eval.runner import run_episode
    from ..eval.scale import build_scale_config

    learned = scorer_producing_policy(model)
    planner = PRODUCING_BASELINES["entity_planner"]
    order_pol = ORDERING_BASELINES["proportional"]
    rows: list[dict] = []
    for seed in seeds:
        cfg = build_scale_config(seed, depth=depth)

        def mean_units_dead(prod_pol) -> tuple[float, float]:
            tot = 0.0
            dead = 0
            for es in range(eval_seeds):
                env = TokenomicsAECEnv(config=cfg, horizon=horizon, entity_producer_obs=True)
                stats = run_episode(env, prod_pol, order_pol, seed=es)
                tot += stats["consumer_completed_units"]
                dead += int(bool(stats["dead"]))
            return tot / eval_seeds, dead / eval_seeds

        learned_u, learned_dead = mean_units_dead(learned)
        planner_u, planner_dead = mean_units_dead(planner)
        P = TokenomicsAECEnv(config=cfg, horizon=1).spec_.num_producers
        ratio = learned_u / planner_u if planner_u else float("nan")
        cmd = producer_command_agreement(model, cfg, [0], horizon)
        rows.append(
            {
                "depth": depth,
                "seed": seed,
                "producers": P,
                "scorer_units": round(learned_u, 2),
                "planner_units": round(planner_u, 2),
                "ratio": round(ratio, 4),
                "dead": round(learned_dead, 4),
                "planner_dead": round(planner_dead, 4),
                "upgrade_fp_rate": round(cmd["upgrade_fp_rate"], 4),
                "inventory_fp_rate": round(cmd["inventory_expansion_fp_rate"], 4),
                "replen_mismatch_rate": round(cmd["replenishment_mismatch_rate"], 4),
            }
        )
        log(
            f"depth {depth} seed {seed} (P={P}): scorer {learned_u:.1f} (dead {learned_dead:.2f}) "
            f"vs planner {planner_u:.1f} (dead {planner_dead:.2f}) -> {ratio:.3f}  "
            f"[up_fp {cmd['upgrade_fp_rate']:.3f} inv_fp {cmd['inventory_expansion_fp_rate']:.3f} "
            f"replen_mm {cmd['replenishment_mismatch_rate']:.3f}]"
        )
    return rows


def producer_command_agreement(
    model: ProducerScoreModel,
    cfg: Configuration,
    seeds,
    horizon: int,
) -> dict[str, float]:
    """Command-level fidelity of the learned scorer vs the planner (EXP-0013 step 4).

    Drives the teacher's own trajectory and, at every producer decision, plans both the
    teacher command (default planner scores) and the learned command (scorer scores) with
    the *same* env feasibility params, then tallies investment false-positive / -negative
    rates and replenishment mismatch. A units ratio can look fine while false investment
    spend is huge; this surfaces it directly. Returns rates in `[0, 1]` plus `n`."""
    from ..domain.producer_planner import plan_producer_command
    from ..eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES

    teacher = PRODUCING_BASELINES["entity_planner"]
    order_pol = ORDERING_BASELINES["proportional"]
    tally = {"n": 0, "up_fp": 0, "up_fn": 0, "inv_fp": 0, "inv_fn": 0, "replen_mismatch": 0}
    for seed in seeds:
        env = TokenomicsAECEnv(config=cfg, horizon=horizon, entity_producer_obs=True)
        env.reset(seed=seed)
        for agent in env.agent_iter(max_iter=horizon * len(env.possible_agents) * 3 + 100):
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            role, pid = env._role[agent]
            if role != "prod":
                env.step(order_pol(env, agent))
                continue
            view = env.system.producer_entity_view(pid)
            mechanics = env.spec_.producer_mechanics[pid]
            params = dict(
                max_replenish=env.spec_.max_replenish[pid],
                replenish_cost_per_unit=mechanics.replenish_cost_per_unit,
                funding_budget=env.system.investment_fund,
                inventory_expansion_increases=mechanics.inventory_expansion_increases,
                inventory_expansion_cost=mechanics.inventory_expansion_cost,
            )
            teacher_cmd = plan_producer_command(view, **params)
            learned_cmd = plan_producer_command(view, scores=score_with_model(model, view), **params)
            tally["n"] += 1
            tally["up_fp"] += int(learned_cmd.do_upgrade and not teacher_cmd.do_upgrade)
            tally["up_fn"] += int(teacher_cmd.do_upgrade and not learned_cmd.do_upgrade)
            tally["inv_fp"] += int(
                learned_cmd.do_inventory_expansion and not teacher_cmd.do_inventory_expansion
            )
            tally["inv_fn"] += int(
                teacher_cmd.do_inventory_expansion and not learned_cmd.do_inventory_expansion
            )
            tally["replen_mismatch"] += int(learned_cmd.replenishments != teacher_cmd.replenishments)
            env.step(teacher(env, agent))
    n = max(1, tally["n"])
    return {
        "n": tally["n"],
        "upgrade_fp_rate": tally["up_fp"] / n,
        "upgrade_fn_rate": tally["up_fn"] / n,
        "inventory_expansion_fp_rate": tally["inv_fp"] / n,
        "inventory_expansion_fn_rate": tally["inv_fn"] / n,
        "replenishment_mismatch_rate": tally["replen_mismatch"] / n,
    }


def broadened_producer_train_configs(
    seeds, *, depth: int = 3, width: int = 4
) -> list[tuple[str, Configuration]]:
    """Step-11 broadened producer distillation set: the scale base PLUS the five families
    step 11 names, so the learned scorer trains on out-of-family recipe DAGs and scarcity
    states the EXP-0013 scale-only run never saw — the suspected source of the fork/join
    producer-alone regression (a hypothesis the step-12 acceptance matrix tests, not a
    proven cause). One topology per (family, seed):

    - `scale`           — the in-family layered base (keeps EXP-0013 planner-parity),
    - `forkjoin`        — multi-input join recipe DAG,
    - `sharedcap`       — levels share capacity types,
    - `uneven`          — one dominant producer per level (concentration),
    - `scarcity-<axis>` — the five `scarcity_suite` single-axis stress variants of the
      scale base (lean fund, tight inventory, heavy backlog, harsh decay, and a
      lean-and-decaying `running_investment` start).

    Reuses the `eval.fixtures` builders (same generators the C3/P3 matrix consumes), so
    training coverage and the acceptance fixtures stay the same recipe family — only the
    seeds differ (train 0-5, acceptance 6-9). Pure: just config construction."""
    from ..eval.fixtures import (
        build_forkjoin_config,
        build_shared_capacity_config,
        build_uneven_strength_config,
        scarcity_suite,
    )
    from ..eval.scale import build_scale_config

    named: list[tuple[str, Configuration]] = []
    for seed in seeds:
        scale = build_scale_config(seed, depth=depth, products_per_layer=width)
        named.append((f"scale-d{depth}-s{seed}", scale))
        named.append((f"forkjoin-d{depth}-s{seed}",
                      build_forkjoin_config(seed, depth=depth, width=width)))
        named.append((f"sharedcap-d{depth}-s{seed}",
                      build_shared_capacity_config(seed, depth=depth, width=width)))
        named.append((f"uneven-d{depth}-s{seed}",
                      build_uneven_strength_config(seed, depth=depth, width=width)))
        for axis, cfg in scarcity_suite(scale).items():
            named.append((f"scarcity-{axis}-d{depth}-s{seed}", cfg))
    return named


def distill_producer_scorer(
    depth: int,
    train_seeds,
    eval_seeds_list,
    epochs: int,
    lr: float,
    eval_seeds: int,
    horizon: int,
    save: str | None = None,
    hidden: int = 64,
    eval_depths=None,
    gate_thresholds: dict[str, float] | None = None,
    train_configs: list[Configuration] | None = None,
    use_edges: bool = True,
    log=print,
) -> tuple[ProducerScoreModel, list[dict]]:
    """One wrapped distill: collect planner-labeled samples over the training topologies,
    fit the shared scorer, optionally save a versioned checkpoint, then eval value vs the
    planner on held-out `eval_seeds_list` across every depth in `eval_depths` (default:
    `[depth]`). Smaller eval depths have fewer producers than the training topology, so they
    are genuinely unseen producer counts — that is the P1 producer-count transfer check.

    By default the training topologies are `build_scale_config(seed, depth)` over
    `train_seeds` (the EXP-0013 scale-only set). Pass `train_configs` to distill over an
    explicit topology list instead — e.g. `broadened_producer_train_configs(...)` for the
    step-11 out-of-family + scarcity coverage. The held-out scale eval is unchanged either
    way (it is the in-family sanity readout; the fixture acceptance is a separate
    `transfer_matrix --fixtures` run). The cloud entrypoint for the producer P1 / step-11
    runs."""
    from ..eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
    from ..eval.scale import build_scale_config

    teacher = PRODUCING_BASELINES["entity_planner"]
    order_pol = ORDERING_BASELINES["proportional"]
    train_seeds = list(train_seeds)
    if train_configs is None:
        train_cfgs = [build_scale_config(seed, depth=depth) for seed in train_seeds]
    else:
        train_cfgs = list(train_configs)
    samples: list[ProducerScoreSample] = []
    for cfg in train_cfgs:
        samples += collect_producer_score_samples(
            cfg, teacher, order_pol, seeds=[0], horizon=horizon
        )
    log(f"collected {len(samples)} producer score samples over {len(train_cfgs)} topologies")
    model = ProducerScoreModel(hidden=hidden, use_edges=use_edges)
    if not use_edges:
        log("ablation #2: no-edge variant (recipe relations unused)")
    fit_producer_score_model(model, samples, epochs=epochs, lr=lr, log=log, log_every=max(1, epochs // 10))
    thresholds = calibrate_gate_thresholds(model, samples) if gate_thresholds is None else gate_thresholds
    model.upgrade_threshold = float(thresholds["upgrade"])
    model.inventory_expansion_threshold = float(thresholds["inventory_expansion"])
    log(
        f"{'calibrated' if gate_thresholds is None else 'fixed'} gate thresholds -> "
        f"upgrade {model.upgrade_threshold:.2f}  inventory {model.inventory_expansion_threshold:.2f}"
    )
    if save:
        save_producer_score_model(model, save)
        log(f"saved producer scorer -> {save}")
    eval_depths = [depth] if eval_depths is None else list(eval_depths)
    rows: list[dict] = []
    for d in eval_depths:
        rows += eval_producer_scorer(model, d, eval_seeds_list, eval_seeds, horizon, log=log)
    return model, rows


_DEFAULT_GATE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def calibrate_gate_thresholds(
    model: ProducerScoreModel,
    samples: list[ProducerScoreSample],
    candidates: Iterable[float] = _DEFAULT_GATE_THRESHOLDS,
) -> dict[str, float]:
    """Pick each investment gate's probability threshold by command agreement with the
    planner on labeled states (EXP-0013).

    For each gate independently, choose the threshold maximizing agreement between
    `prob >= threshold` and the planner's own gate boolean (`teacher roi > 0`). The other
    planner feasibility gates (budget, active-investment locks) are identical for the
    learned and teacher policies on the same view, so they cancel — score-gate agreement
    here equals command agreement on that investment. Ties resolve to the threshold
    closest to 0.5 (least biased). Returns `{"upgrade": ..., "inventory_expansion": ...}`.
    """
    candidates = sorted(set(float(c) for c in candidates))
    # entries: (gate_probs_dict, upgrade_bool, inventory_bool)
    probs = [
        (
            producer_gate_probs(model, s.view),
            s.target.upgrade_roi > 0.0,
            s.target.inventory_expansion_value > 0.0,
        )
        for s in samples
    ]
    return {
        "upgrade": _best_gate_threshold(probs, "upgrade", 1, candidates),
        "inventory_expansion": _best_gate_threshold(probs, "inventory_expansion", 2, candidates),
    }


def _best_gate_threshold(probs, key: str, label_idx: int, candidates) -> float:
    """Threshold with max planner-gate agreement; tie -> nearest 0.5 (EXP-0013)."""
    best_t, best_agree = 0.5, -1
    for t in candidates:
        agree = sum(int((entry[0][key] >= t) == entry[label_idx]) for entry in probs)
        if agree > best_agree or (agree == best_agree and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_agree = t, agree
    return best_t


def scorer_producing_policy(
    model: ProducerScoreModel,
    *,
    planner_gates: bool = False,
    replenish_tie_rel_tol: float = 0.0,
) -> Policy:
    """Wrap a fitted scorer as a producing policy through the deterministic planner.

    `planner_gates=True` deploys the P2/C3 mode: learned continuous/ranking scores with
    the upgrade / inventory-expansion gates owned by the deterministic planner (EXP-0017).
    `replenish_tie_rel_tol` lets the planner split scarce headroom evenly across inputs
    whose learned values are near-equal (fork/join paired inputs). Same checkpoint, same
    schema contract — only the producer policy adapter differs."""
    from ..eval.baselines import producing_entity_planner_from_scorer

    return producing_entity_planner_from_scorer(
        lambda view: score_with_model(model, view, planner_gates=planner_gates),
        replenish_tie_rel_tol=replenish_tie_rel_tol,
    )


def save_producer_score_model(model: ProducerScoreModel, path, *,
                              config_fingerprint: str | None = None) -> None:
    """Save a reusable EXP-0010C scorer checkpoint.

    Records the portable feature `schema` (step-13 load gate). `config_fingerprint`, if
    given, is stored as AUDIT metadata for transfer reports only — never a load gate (a
    portable checkpoint must load across topologies of different fingerprint)."""
    ckpt = {
        "version": PRODUCER_SCORER_CHECKPOINT_VERSION,
        "hidden": model.hidden,
        "use_edges": bool(model.use_edges),
        "upgrade_threshold": float(model.upgrade_threshold),
        "inventory_expansion_threshold": float(model.inventory_expansion_threshold),
        "state_dict": model.state_dict(),
        "schema": producer_portable_schema(use_edges=model.use_edges).to_dict(),
    }
    if config_fingerprint is not None:
        ckpt["config_fingerprint"] = config_fingerprint
    torch.save(ckpt, path)


def load_producer_score_model(path) -> ProducerScoreModel:
    """Load a versioned EXP-0010C scorer checkpoint.

    Gates on the portable feature `schema` when present (step 13): a feature/entity/relation/
    decoder/architecture change rejects with a named error. Legacy v2 checkpoints saved
    before the schema block (EXP-0013/0021) have no `schema` and load unchanged — the named
    gate is additive. The `config_fingerprint` (if any) is ignored on load."""
    from .portable_schema import PortableSchema, assert_compatible

    ckpt = torch.load(path, weights_only=True)
    if ckpt.get("version") != PRODUCER_SCORER_CHECKPOINT_VERSION:
        raise ValueError(
            f"producer scorer checkpoint version {ckpt.get('version')!r} "
            f"!= {PRODUCER_SCORER_CHECKPOINT_VERSION}"
        )
    use_edges = bool(ckpt.get("use_edges", True))
    if "schema" in ckpt:
        assert_compatible(
            PortableSchema.from_dict(ckpt["schema"]), producer_portable_schema(use_edges=use_edges)
        )
    model = ProducerScoreModel(hidden=int(ckpt["hidden"]), use_edges=use_edges)
    model.load_state_dict(ckpt["state_dict"])
    model.upgrade_threshold = float(ckpt.get("upgrade_threshold", 0.5))
    model.inventory_expansion_threshold = float(ckpt.get("inventory_expansion_threshold", 0.5))
    model.eval()
    return model


def _prepare_score_batches(samples: list[ProducerScoreSample], device) -> list[_ProducerScoreBatch]:
    groups: dict[tuple, list[ProducerScoreSample]] = {}
    for sample in samples:
        groups.setdefault(_batch_key(sample.view), []).append(sample)
    return [_make_score_batch(group, device) for group in groups.values()]


def _batch_key(view: ProducerEntityView) -> tuple:
    return (
        len(view.capacity_rows),
        len(view.product_rows),
        len(view.input_rows),
        tuple(sorted(view.relations.product_capacity_required.items())),
        tuple(sorted(view.relations.product_input_required.items())),
    )


def _make_score_batch(samples: list[ProducerScoreSample], device) -> _ProducerScoreBatch:
    first = samples[0].view
    n_capacity = len(first.capacity_rows)
    n_product = len(first.product_rows)
    n_input = len(first.input_rows)
    return _ProducerScoreBatch(
        global_x=_tensor([_global_features(s.view) for s in samples], device),
        capacity_x=_row_tensor(
            samples, n_capacity, _CAPACITY_DIM, _capacity_features, "capacity_rows", device
        ),
        product_x=_row_tensor(
            samples, n_product, _PRODUCT_DIM, _product_features, "product_rows", device
        ),
        input_x=_row_tensor(
            samples, n_input, _INPUT_DIM, _input_features, "input_rows", device
        ),
        product_capacity_weight=_relation_weight(
            first.relations.product_capacity_required, n_product, n_capacity, device
        ),
        product_input_weight=_relation_weight(
            first.relations.product_input_required, n_product, n_input, device
        ),
        product_target=_target_tensor(
            samples,
            n_product,
            "product_rows",
            lambda sample, row: sample.target.product_urgency.get(row.product, 0.0),
            device,
        ),
        input_target=_target_tensor(
            samples,
            n_input,
            "input_rows",
            lambda sample, row: sample.target.input_value.get(row.input_product, 0.0),
            device,
        ),
        capacity_target=_target_tensor(
            samples,
            n_capacity,
            "capacity_rows",
            lambda sample, row: sample.target.capacity_bottleneck_value.get(
                row.capacity_type, 0.0
            ),
            device,
        ),
        # Investment gates are binary: label = whether the planner's own gate fires
        # (its score-gate is `roi > 0`). The heads are BCE-trained against these booleans,
        # not the ROI magnitude (EXP-0013). Restoration is planner-owned -> no target.
        restoration_target=torch.zeros(len(samples), dtype=torch.float32, device=device),
        upgrade_target=torch.as_tensor(
            [float(s.target.upgrade_roi > 0.0) for s in samples], dtype=torch.float32, device=device
        ),
        inventory_target=torch.as_tensor(
            [float(s.target.inventory_expansion_value > 0.0) for s in samples],
            dtype=torch.float32, device=device,
        ),
    )


def _row_tensor(
    samples: list[ProducerScoreSample],
    n_rows: int,
    dim: int,
    feature_fn,
    attr: str,
    device,
) -> torch.Tensor:
    if n_rows == 0:
        return torch.empty((len(samples), 0, dim), dtype=torch.float32, device=device)
    return torch.as_tensor(
        [[feature_fn(row) for row in getattr(sample.view, attr)] for sample in samples],
        dtype=torch.float32,
        device=device,
    )


def _target_tensor(samples: list[ProducerScoreSample], n_rows: int, attr: str, value_fn, device) -> torch.Tensor:
    if n_rows == 0:
        return torch.empty((len(samples), 0), dtype=torch.float32, device=device)
    return torch.as_tensor(
        [[value_fn(sample, row) for row in getattr(sample.view, attr)] for sample in samples],
        dtype=torch.float32,
        device=device,
    )


def _relation_weight(edges: dict[tuple[int, int], int], n_src: int, n_dst: int, device) -> torch.Tensor:
    weights = torch.zeros((n_src, n_dst), dtype=torch.float32, device=device)
    for (src, dst), amount in edges.items():
        if 0 <= src < n_src and 0 <= dst < n_dst:
            weights[src, dst] = weights[src, dst] + _edge_weight(amount)
    return weights


def _batch_loss(model: ProducerScoreModel, batch: _ProducerScoreBatch) -> torch.Tensor:
    out = _forward_batch(model, batch)
    zeros = torch.zeros(batch.size, dtype=torch.float32, device=batch.global_x.device)
    bce = nn.functional.binary_cross_entropy_with_logits
    parts = [
        _row_mse(out.product_urgency, batch.product_target, zeros),
        _row_mse(out.input_value, batch.input_target, zeros),
        _row_mse(out.capacity_bottleneck_value, batch.capacity_target, zeros),
        # learned binary investment gates: BCE on the planner's gate booleans (EXP-0013).
        bce(out.upgrade_roi, batch.upgrade_target, reduction="none"),
        bce(out.inventory_expansion_value, batch.inventory_target, reduction="none"),
    ]
    return torch.stack(parts, dim=1).mean(dim=1).sum()


def _forward_batch(model: ProducerScoreModel, batch: _ProducerScoreBatch) -> ProducerScoreTensors:
    hidden = model.hidden
    device = batch.global_x.device
    n = batch.size
    g = model.global_encoder(batch.global_x)
    cap_h = _encode_batched_rows(batch.capacity_x, hidden, model.capacity_encoder)
    prod_h = _encode_batched_rows(batch.product_x, hidden, model.product_encoder)
    input_h = _encode_batched_rows(batch.input_x, hidden, model.input_encoder)

    n_product = prod_h.shape[1]
    n_input = input_h.shape[1]
    n_capacity = cap_h.shape[1]
    if model.use_edges:
        prod_cap_ctx = _forward_batch_relation_context(batch.product_capacity_weight, cap_h, n, hidden, device)
        prod_input_ctx = _forward_batch_relation_context(batch.product_input_weight, input_h, n, hidden, device)
        input_prod_ctx = _reverse_batch_relation_context(batch.product_input_weight, prod_h, n, hidden, device)
        cap_prod_ctx = _reverse_batch_relation_context(batch.product_capacity_weight, prod_h, n, hidden, device)
    else:
        prod_cap_ctx = torch.zeros((n, n_product, hidden), device=device)
        prod_input_ctx = torch.zeros((n, n_product, hidden), device=device)
        input_prod_ctx = torch.zeros((n, n_input, hidden), device=device)
        cap_prod_ctx = torch.zeros((n, n_capacity, hidden), device=device)

    if n_product:
        product_scores = model.product_head(torch.cat([
            prod_h,
            prod_cap_ctx,
            prod_input_ctx,
            _expand_batch(g, n_product),
        ], dim=2)).squeeze(-1)
    else:
        product_scores = torch.empty((n, 0), device=device)

    if n_input:
        input_scores = model.input_head(torch.cat([
            input_h,
            input_prod_ctx,
            _expand_batch(g, n_input),
        ], dim=2)).squeeze(-1)
    else:
        input_scores = torch.empty((n, 0), device=device)

    if n_capacity:
        cap_context = torch.cat([
            cap_h,
            cap_prod_ctx,
            _expand_batch(g, n_capacity),
        ], dim=2)
        cap_scores = model.capacity_head(cap_context).squeeze(-1)
        investment = model.investment_head(cap_context)
        restoration = investment[:, :, 0].max(dim=1).values
        upgrade = investment[:, :, 1].max(dim=1).values
    else:
        cap_scores = torch.empty((n, 0), device=device)
        restoration = torch.zeros(n, device=device)
        upgrade = torch.zeros(n, device=device)

    inventory = model.inventory_head(torch.cat([
        g,
        _mean_batch_or_zero(prod_h, hidden, device),
        _mean_batch_or_zero(input_h, hidden, device),
    ], dim=1)).squeeze(-1)

    return ProducerScoreTensors(
        product_urgency=product_scores,
        input_value=input_scores,
        capacity_bottleneck_value=cap_scores,
        restoration_roi=restoration,
        upgrade_roi=upgrade,
        inventory_expansion_value=inventory,
    )


def _encode_batched_rows(rows: torch.Tensor, hidden: int, encoder: nn.Module) -> torch.Tensor:
    n, n_rows, dim = rows.shape
    if n_rows == 0:
        return torch.empty((n, 0, hidden), dtype=rows.dtype, device=rows.device)
    return encoder(rows.reshape(n * n_rows, dim)).reshape(n, n_rows, hidden)


def _forward_batch_relation_context(
    weights: torch.Tensor,
    dst_h: torch.Tensor,
    n: int,
    hidden: int,
    device,
) -> torch.Tensor:
    n_src = weights.shape[0]
    if n_src == 0:
        return torch.empty((n, 0, hidden), device=device)
    if dst_h.shape[1] == 0:
        return torch.zeros((n, n_src, hidden), device=device)
    ctx = torch.einsum("sd,ndh->nsh", weights, dst_h)
    return ctx / weights.sum(dim=1).clamp_min(1.0).view(1, n_src, 1)


def _reverse_batch_relation_context(
    weights: torch.Tensor,
    src_h: torch.Tensor,
    n: int,
    hidden: int,
    device,
) -> torch.Tensor:
    n_dst = weights.shape[1]
    if n_dst == 0:
        return torch.empty((n, 0, hidden), device=device)
    if src_h.shape[1] == 0:
        return torch.zeros((n, n_dst, hidden), device=device)
    ctx = torch.einsum("sd,nsh->ndh", weights, src_h)
    return ctx / weights.sum(dim=0).clamp_min(1.0).view(1, n_dst, 1)


def _expand_batch(rows: torch.Tensor, n_rows: int) -> torch.Tensor:
    return rows.unsqueeze(1).expand(-1, n_rows, -1)


def _mean_batch_or_zero(rows: torch.Tensor, hidden: int, device) -> torch.Tensor:
    if rows.shape[1] == 0:
        return torch.zeros((rows.shape[0], hidden), device=device)
    return rows.mean(dim=1)


def _row_mse(pred: torch.Tensor, target: torch.Tensor, empty_value: torch.Tensor) -> torch.Tensor:
    if target.shape[1] == 0:
        return empty_value
    return ((pred - target) ** 2).mean(dim=1)


def _mse(pred: torch.Tensor, values: list[float], device) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device)
    target = torch.as_tensor(values, dtype=torch.float32, device=device)
    return ((pred - target) ** 2).mean()


def _score_map(keys: Iterable[Product | CapacityType], scores: torch.Tensor) -> dict:
    return {key: max(0.0, float(value)) for key, value in zip(keys, scores.detach().cpu())}


def _nonnegative_scalar(value: torch.Tensor) -> float:
    return max(0.0, float(value.detach().cpu()))


def _sigmoid(value: torch.Tensor) -> float:
    """Gate logit -> probability (EXP-0013 learned binary investment gates)."""
    return 1.0 / (1.0 + math.exp(-float(value.detach().cpu())))


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


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--train-seeds", type=str, default="0-5", help="scale seeds for distillation, e.g. 0-5")
    ap.add_argument("--broaden", action="store_true",
                    help="step 11: distill over the broadened topology set (scale + fork/join + "
                         "shared-capacity + uneven-strength + scarcity/running-investment) at "
                         "--train-seeds, not scale-only. TARGETS the out-of-family producer gap "
                         "(the step-12 acceptance matrix proves whether it closes it).")
    ap.add_argument("--width", type=int, default=4,
                    help="products/producers per layer for the broadened fixture families.")
    ap.add_argument("--eval-seeds-list", type=str, default="6-9", help="held-out scale seeds for value/transfer")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eval-seeds", type=int, default=5, help="episodes per held-out topology")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="CPU torch threads for tiny per-view MLPs; default 1 avoids threadpool overhead.")
    ap.add_argument("--eval-depths", type=str, default=None,
                    help="depths to eval the distilled scorer at (e.g. 1-3); smaller "
                         "depths = unseen producer counts. Default: just --depth.")
    ap.add_argument("--upgrade-threshold", type=float, default=None,
                    help="fix the upgrade gate probability threshold instead of calibrating "
                         "it on training command agreement (EXP-0013).")
    ap.add_argument("--inventory-threshold", type=float, default=None,
                    help="fix the inventory-expansion gate threshold (EXP-0013).")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no-edges", action="store_true",
                    help="step-14 ablation #2: distill the no-edge variant (recipe relations "
                         "unused) to measure the edge contribution to portable transfer.")
    args = ap.parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    def _range(s: str):
        a, b = s.split("-")
        return range(int(a), int(b) + 1)

    gate_thresholds = None
    if args.upgrade_threshold is not None and args.inventory_threshold is not None:
        gate_thresholds = {
            "upgrade": args.upgrade_threshold,
            "inventory_expansion": args.inventory_threshold,
        }

    eval_depths = list(_range(args.eval_depths)) if args.eval_depths else None
    train_configs = None
    if args.broaden:
        named = broadened_producer_train_configs(
            list(_range(args.train_seeds)), depth=args.depth, width=args.width
        )
        train_configs = [cfg for _, cfg in named]
        print(f"broadened training set: {len(named)} topologies "
              f"({sorted({n.split('-')[0] for n, _ in named})})")
    _, rows = distill_producer_scorer(
        depth=args.depth, train_seeds=_range(args.train_seeds),
        eval_seeds_list=_range(args.eval_seeds_list), epochs=args.epochs, lr=args.lr,
        eval_seeds=args.eval_seeds, horizon=args.horizon, save=args.save, hidden=args.hidden,
        eval_depths=eval_depths, gate_thresholds=gate_thresholds, train_configs=train_configs,
        use_edges=not args.no_edges,
    )
    cols = ("depth,seed,producers,scorer_units,planner_units,ratio,dead,planner_dead,"
            "upgrade_fp_rate,inventory_fp_rate,replen_mismatch_rate")
    print("\n" + cols)
    for r in rows:
        print(f"{r['depth']},{r['seed']},{r['producers']},{r['scorer_units']},"
              f"{r['planner_units']},{r['ratio']},{r['dead']},{r['planner_dead']},"
              f"{r['upgrade_fp_rate']},{r['inventory_fp_rate']},{r['replen_mismatch_rate']}")
    ratios = [r["ratio"] for r in rows]
    max_dead = max(r["dead"] for r in rows)
    max_up_fp = max(r["upgrade_fp_rate"] for r in rows)
    max_inv_fp = max(r["inventory_fp_rate"] for r in rows)
    print(f"\nmean ratio vs planner: {sum(ratios)/len(ratios):.4f}; min {min(ratios):.4f}; "
          f"max scorer dead-rate {max_dead:.3f}; max upgrade-FP {max_up_fp:.3f}; "
          f"max inventory-FP {max_inv_fp:.3f}; producer counts {sorted({r['producers'] for r in rows})}")


if __name__ == "__main__":
    main()
