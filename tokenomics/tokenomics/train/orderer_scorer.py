"""Learned local orderer candidate-scorer (PRODUCT_AGNOSTIC O2).

The learned component predicts one score per *producer row* from a
`ProductStreamView`. Those scores feed the deterministic masked allocator
(`env.allocator.allocate_suborders`), which owns eligibility masking and the
quantity/token conservation invariants. The model never consumes product or producer
ids — it embeds the product (global) features once and a shared encoder over the
per-producer rows, so it is order-equivariant over producers and runs on any producer
count P (no fixed `K*P` action head). Mirror of `producer_scorer.ProducerScoreModel`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..domain.common import ProducerId
from ..eval.runner import run_episode
from ..domain.orderer_view import (
    ProducerProductStats,
    ProducerStats,
    ProductStats,
    ProductStreamView,
)

ORDERER_SCORER_CHECKPOINT_VERSION = 1
_BOUND = 10.0  # action logit bound (mirror env.spaces.ACTION_LOGIT_BOUND)

# Feature schema (counts tie the encoders to the domain dataclasses; asserted in
# tests). Product/global features once; each producer row = ProducerStats fields then
# ProducerProductStats fields, in dataclass definition order.
_PRODUCT_DIM = 9
# ProducerStats (13 scalars + active_investments 3-tuple) + ProducerProductStats (15).
_PRODUCER_ROW_DIM = 13 + 3 + 15

# Named feature schema for the portable load gate (step 13). Names mirror
# `_product_features` / `_producer_row_features` in order; the gate compares them by name so
# a feature drift rejects while a wider/relabeled producer set (absent here) does not.
_ORDERER_PRODUCT_FEATURES = (
    "open_demand_quantity", "open_demand_tokens", "recent_average_cycle_demand",
    "recent_rejected_quantity", "product_depth", "recipe_input_count",
    "recipe_input_units", "recipe_capacity_type_count", "eligible_producer_count",
)
_ORDERER_PRODUCER_ROW_FEATURES = (
    "total_current_capacity", "total_max_capacity", "capacity_headroom",
    "min_capacity_ratio", "mean_capacity_ratio", "requested_load", "reserved_load",
    "total_load", "cut_off_price", "outstanding_inbound_units", "active_restoration",
    "active_upgrade", "active_inventory_expansion", "acceptance_rate", "completion_rate",
    "rejection_rate", "eligible", "accept_qty_now", "complete_qty_now", "makeable_qty_now",
    "capacity_bottleneck_ratio", "input_bottleneck_ratio", "bottleneck_ratio",
    "slot_makeability", "recent_routed_qty", "recent_completed_qty", "recent_rejected_qty",
    "recent_success_rate", "recent_throughput", "recent_completion_latency",
    "recent_failure_latency",
)


def orderer_portable_schema() -> "PortableSchema":
    """The current portable-load schema for the orderer scorer (step 13). Topology-free:
    named product/producer-row features + entity types + norm/decoder/architecture versions,
    no product/producer ids or counts. No sparse relations (the model reads a broadcast
    product context + independent producer rows), so `relation_types` is empty."""
    from .portable_schema import PortableSchema

    return PortableSchema(
        schema_version=1,
        entity_feature_names={
            "product": _ORDERER_PRODUCT_FEATURES,
            "producer_row": _ORDERER_PRODUCER_ROW_FEATURES,
        },
        entity_types=("product", "producer_row"),
        relation_types=(),
        normalization_version="nan-sanitized-bounded-v1",
        decoder_version="allocate_suborders-v1",
        architecture_version="orderer-producer-row-scorer-v1",
    )


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


def _product_features(s: ProductStats) -> list[float]:
    return [
        s.open_demand_quantity, s.open_demand_tokens, s.recent_average_cycle_demand,
        s.recent_rejected_quantity, s.product_depth, s.recipe_input_count,
        s.recipe_input_units, s.recipe_capacity_type_count, s.eligible_producer_count,
    ]


def _producer_row_features(p: ProducerStats, pp: ProducerProductStats) -> list[float]:
    return [
        p.total_current_capacity, p.total_max_capacity, p.capacity_headroom,
        p.min_capacity_ratio, p.mean_capacity_ratio, p.requested_load, p.reserved_load,
        p.total_load, p.cut_off_price, p.outstanding_inbound_units,
        *p.active_investments,  # (restoration, upgrade, inventory-expansion) running flags
        p.acceptance_rate, p.completion_rate, p.rejection_rate,
        pp.eligible, pp.accept_qty_now, pp.complete_qty_now, pp.makeable_qty_now,
        pp.capacity_bottleneck_ratio, pp.input_bottleneck_ratio, pp.bottleneck_ratio,
        pp.slot_makeability, pp.recent_routed_qty, pp.recent_completed_qty,
        pp.recent_rejected_qty, pp.recent_success_rate, pp.recent_throughput,
        pp.recent_completion_latency, pp.recent_failure_latency,
    ]


def _tensor(rows: list[list[float]], device) -> torch.Tensor:
    # Some domain facts are undefined in degenerate states (e.g. cut_off_price with no
    # accepted bids -> NaN); sanitize so the model never emits NaN scores into the
    # allocator. Bounded so an inf fact can't dominate.
    arr = np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=_BOUND, neginf=-_BOUND)
    return torch.from_numpy(arr).to(device)


class OrdererScoreModel(nn.Module):
    """Shared per-producer-row scorer for product-stream views (O2).

    No product/producer ids as input. The product (global) features are embedded once
    and broadcast to every producer row; a shared row encoder scores each producer
    independently, so permuting producer rows permutes the output scores and the model
    runs unchanged on any producer count."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.global_encoder = _mlp(_PRODUCT_DIM, hidden)
        self.row_encoder = _mlp(_PRODUCER_ROW_DIM, hidden)
        self.score_head = nn.Linear(2 * hidden, 1)

    def producer_ids(self, view: ProductStreamView) -> list[ProducerId]:
        # Row order is the sorted producer id list — the same stable order the env
        # encoder and `decode_ordering` / allocator use, so scores align to columns.
        return sorted(view.producers)

    def forward_view(self, view: ProductStreamView) -> torch.Tensor:
        device = next(self.parameters()).device
        pids = self.producer_ids(view)
        g = self.global_encoder(_tensor([_product_features(view.product_stats)], device))[0]
        rows = [
            _producer_row_features(view.producers[p], view.producer_products[p])
            for p in pids
        ]
        h = self.row_encoder(_tensor(rows, device))  # (P, hidden)
        ctx = g.unsqueeze(0).expand(len(pids), -1)
        return self.score_head(torch.cat([h, ctx], dim=1)).squeeze(-1)  # (P,)


# --- inference: scorer + masked allocator as an ordering Policy (O2 deploy) ---


def scorer_ordering_policy(model: "OrdererScoreModel"):
    """Wrap the scorer as a deployed product-stream ordering `Policy`.

    Each agent serves one (source, product) stream, so every order in its backlog
    shares the same eligible-producer set and the same per-producer scores. The policy
    emits those scores tiled across the k_stream backlog slots; the env's
    `decode_ordering` softmaxes per order over the eligible producers and hands the
    result to `allocate_suborders` (O1), which owns eligibility + conservation. Needs a
    `corrected_stream_k=True` (v4) env so the action width is k_stream * P."""
    model.eval()

    def policy(env, agent) -> np.ndarray:
        spec = env.spec_
        _, aid = env._role[agent]
        view = env.system.product_stream_view(aid)
        with torch.no_grad():
            scores = model.forward_view(view).cpu().numpy()  # aligned to sorted(view.producers)
        action = np.tile(scores.astype(np.float32), (spec.k_stream, 1))  # (k_stream, P)
        return np.clip(action, -_BOUND, _BOUND).flatten()

    return policy


# --- supervised teacher: proportional throughput logits per producer ---


def teacher_logits(view: ProductStreamView) -> tuple[list[ProducerId], np.ndarray, np.ndarray]:
    """Per-producer distillation target = log(makeable-units-now), the same throughput
    signal the `proportional` baseline routes by, clamped to the action bound; an
    ineligible or zero-throughput producer floors at -bound. Returns (producer ids,
    target logits, eligibility mask), aligned to sorted producer ids."""
    pids = sorted(view.producers)
    target = np.empty(len(pids), dtype=np.float32)
    mask = np.empty(len(pids), dtype=bool)
    for j, pid in enumerate(pids):
        pp = view.producer_products[pid]
        mask[j] = pp.eligible > 0
        mk = pp.makeable_qty_now
        target[j] = float(np.clip(np.log(mk), -_BOUND, _BOUND)) if mk > 0 else -_BOUND
    return pids, target, mask


def _center_clip_eligible(logits: np.ndarray, mask: np.ndarray, bound: float = _BOUND) -> np.ndarray:
    """Normalize a committed routing into a distill target (EXP-0032 oracle relabel).

    Sampled routing vectors have arbitrary scale, but the allocator softmaxes per order over
    the eligible producers, so only the *within-eligible* differences matter. Center the
    eligible logits to zero mean (softmax shift-invariance) and clip to the action `bound`;
    ineligible rows floor at -bound to mirror `teacher_logits` (they are masked out of the
    distill loss regardless)."""
    out = np.full(len(mask), -bound, dtype=np.float32)
    if mask.any():
        elig = np.asarray(logits, dtype=np.float32)[mask]
        out[mask] = np.clip(elig - elig.mean(), -bound, bound)
    return out


def _reduce_committed(action: np.ndarray, k_stream: int, num_producers: int,
                      n_occupied: int) -> np.ndarray:
    """Reduce a committed ordering action `(k_stream * num_producers)` to one per-producer
    routing (EXP-0032). The scorer emits a single vector tiled across the stream's backlog
    slots, so a committed routing that differs per slot is summarized by the mean over the
    occupied backlog slots (empty slots beyond the backlog carry no order and are dropped)."""
    rows = np.asarray(action, dtype=np.float32).reshape(k_stream, num_producers)
    n = max(1, min(n_occupied, k_stream))
    return rows[:n].mean(axis=0)


@dataclass
class ScorerSample:
    view: ProductStreamView
    target: np.ndarray  # per-producer teacher logits (sorted ids)
    mask: np.ndarray    # eligibility (sorted ids)


def collect_scorer_samples(seeds, depth: int, horizon: int, log=lambda *a, **k: None,
                           configs=None) -> list["ScorerSample"]:
    """Roll `stock_aware` + `proportional` across the given scale topologies and
    snapshot every product-stream view that holds a backlog, paired with the teacher
    target. On-teacher BC: the proportional orderer also drives the states.

    By default each seed rebuilds `build_scale_config(seed, depth)`. Pass `configs` (aligned
    1:1 with `seeds`, e.g. replayed archived `Configuration`s) to collect over an explicit
    topology list instead — so a mined/replay curriculum trains the orderer on exactly the
    tasks it was mined on, not a depth-`depth` scale stand-in."""
    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
    from ..eval.scale import build_scale_config

    prod_pol, order_pol = PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["proportional"]
    seeds = list(seeds)
    if configs is not None:
        configs = list(configs)
        if len(configs) != len(seeds):
            raise ValueError(f"configs ({len(configs)}) must align 1:1 with seeds ({len(seeds)})")
    samples: list[ScorerSample] = []
    for i, seed in enumerate(seeds):
        cfg = configs[i] if configs is not None else build_scale_config(seed, depth=depth)
        env = TokenomicsAECEnv(config=cfg, horizon=horizon,
                               product_stream_ordering=True, corrected_stream_k=True)
        env.reset(seed=seed)
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        for agent in env.agent_iter(max_iter=max_iter):
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            role = env._role[agent][0]
            if role == "order":
                _, aid = env._role[agent]
                view = env.system.product_stream_view(aid)
                if view.backlog:
                    pids, target, mask = teacher_logits(view)
                    if mask.any():
                        samples.append(ScorerSample(view, target, mask))
                env.step(order_pol(env, agent))
            else:
                env.step(prod_pol(env, agent))
        log(f"seed {seed}: {len(samples)} cumulative samples")
    return samples


def collect_oracle_orderer_samples(
    configs, seeds, *, prod, candidates: Sequence[str] = ("even_split", "proportional"),
    n_samples: int = 8, sample_scale: float = 1.0, lookahead: int = 6, horizon: int = 100,
    log=lambda *a, **k: None,
) -> list["ScorerSample"]:
    """Relabel the orderer from the mining bar's own per-cycle search (EXP-0032 v4).

    Drives each `(config, seed)` under the deployment-matched cross `prod` (the champion
    producer) + the bar's directed routing search (`_search_best_pol` over `candidates` +
    `n_samples` sampled routings, EXP-0005), and snapshots every product-stream view holding
    a backlog. The teacher target is the committed routing itself — reduced over the occupied
    backlog slots (`_reduce_committed`) and centered+clipped (`_center_clip_eligible`) — so the
    same masked-distill path (`fit_scorer`) that consumes `collect_scorer_samples` output can
    absorb the search-oracle headroom the proportional teacher cannot. Defaults reproduce the
    `entity_planner+mpc_search` bar's search (candidates, `n_samples=8`, `lookahead=6`), so the
    orderer teacher == the bar the mined gap was measured against. Reproducible: the sample RNG
    is seeded by `(seed, cycle)` inside `_search_best_pol`."""
    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.baselines import ORDERING_BASELINES
    from ..eval.orderer_oracle import _search_best_pol

    configs, seeds = list(configs), list(seeds)
    if len(configs) != len(seeds):
        raise ValueError(f"configs ({len(configs)}) must align 1:1 with seeds ({len(seeds)})")
    cand = {name: ORDERING_BASELINES[name] for name in candidates}
    samples: list[ScorerSample] = []
    for cfg, seed in zip(configs, seeds):
        # Deployment env (matches `joint_stack_env`): entity producer obs so a champion
        # score-model `prod` observes correctly + corrected v4 product streams.
        env = TokenomicsAECEnv(config=cfg, horizon=horizon, entity_producer_obs=True,
                               product_stream_ordering=True, corrected_stream_k=True)
        env.reset(seed=seed)
        spec = env.spec_
        adim = spec.ordering_action_dim
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        cur = -1
        order_pol = next(iter(cand.values()))
        for agent in env.agent_iter(max_iter=max_iter):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            if env.current_cycle != cur:  # re-search the committed routing once per cycle
                cur = env.current_cycle
                order_pol, _ = _search_best_pol(
                    env, prod, cand, n_samples, sample_scale, lookahead, seed, cur, adim)
            role, aid = env._role[agent]
            if role == "order":
                view = env.system.product_stream_view(aid)
                if view.backlog:
                    pids = sorted(view.producers)
                    mask = np.array([view.producer_products[p].eligible > 0 for p in pids])
                    if mask.any():
                        act = np.asarray(order_pol(env, agent), dtype=np.float32)
                        reduced = _reduce_committed(act, spec.k_stream, spec.num_producers,
                                                    len(view.backlog))
                        samples.append(ScorerSample(view, _center_clip_eligible(reduced, mask), mask))
                env.step(order_pol(env, agent))
            else:
                env.step(prod(env, agent))
        log(f"seed {seed}: {len(samples)} cumulative samples")
    return samples


# Stride between the per-search seeds of the K-averaged collector: each of the K searches at one
# state must draw an INDEPENDENT sampled-routing set (RNG seeded by (seed, cycle) in _candidate_list),
# so search k uses seed + k*stride; k=0 keeps the single-search seed for strict parity.
_K_SEARCH_SEED_STRIDE = 1_000_003


def collect_averaged_oracle_orderer_samples(
    configs, seeds, *, prod, k_searches: int = 4,
    candidates: Sequence[str] = ("even_split", "proportional"),
    n_samples: int = 8, sample_scale: float = 1.0, lookahead: int = 6, horizon: int = 100,
    log=lambda *a, **k: None,
) -> list["ScorerSample"]:
    """Denoised oracle relabel (Stage A0 / H1, EXP-0036). Same as `collect_oracle_orderer_samples`
    but each cycle runs `k_searches` INDEPENDENT searches (RNG stride `_K_SEARCH_SEED_STRIDE`) and
    commits their MEAN routing vector — both as the teacher target and to advance the trajectory.

    The mined failure was that a single per-cycle search commits one routing among near-tied
    candidates (2 named + `n_samples` per-cycle Gaussian vectors), so the per-stream target is
    tie-arbitrary noise. Averaging K independent searches cancels the tie-arbitrariness while
    keeping the systematic routing; comparing VE(averaged) to VE(single) is A0's discriminating
    check. Cross-trajectory averaging is invalid (committed routing diverges the trajectory after
    cycle 0), so the mean is committed to advance a SINGLE trajectory. With `k_searches=1` this is
    byte-identical to the single-search collector (nothing to average)."""
    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.baselines import ORDERING_BASELINES
    from ..eval.orderer_oracle import _search_best_pol

    configs, seeds = list(configs), list(seeds)
    if len(configs) != len(seeds):
        raise ValueError(f"configs ({len(configs)}) must align 1:1 with seeds ({len(seeds)})")
    if k_searches < 1:
        raise ValueError(f"k_searches must be >= 1, got {k_searches}")
    cand = {name: ORDERING_BASELINES[name] for name in candidates}
    samples: list[ScorerSample] = []
    for cfg, seed in zip(configs, seeds):
        env = TokenomicsAECEnv(config=cfg, horizon=horizon, entity_producer_obs=True,
                               product_stream_ordering=True, corrected_stream_k=True)
        env.reset(seed=seed)
        spec = env.spec_
        adim = spec.ordering_action_dim
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        cur = -1
        k_pols: list = []
        for agent in env.agent_iter(max_iter=max_iter):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            if env.current_cycle != cur:  # K independent searches, once per cycle
                cur = env.current_cycle
                k_pols = [
                    _search_best_pol(env, prod, cand, n_samples, sample_scale, lookahead,
                                     seed + k * _K_SEARCH_SEED_STRIDE, cur, adim)[0]
                    for k in range(k_searches)
                ]
            role, aid = env._role[agent]
            if role == "order":
                mean_act = np.mean([np.asarray(p(env, agent), dtype=np.float32) for p in k_pols],
                                   axis=0).astype(np.float32)
                view = env.system.product_stream_view(aid)
                if view.backlog:
                    pids = sorted(view.producers)
                    mask = np.array([view.producer_products[p].eligible > 0 for p in pids])
                    if mask.any():
                        reduced = _reduce_committed(mean_act, spec.k_stream, spec.num_producers,
                                                    len(view.backlog))
                        samples.append(ScorerSample(view, _center_clip_eligible(reduced, mask), mask))
                env.step(mean_act)
            else:
                env.step(prod(env, agent))
        log(f"seed {seed}: {len(samples)} cumulative samples (k={k_searches})")
    return samples


class OrdererChoiceModel(nn.Module):
    """Per-stream categorical head over the named+blend vocabulary (A0b / D1).

    Same product/row encoders as `OrdererScoreModel`, but producer rows are mean-pooled into one
    stream embedding (order-invariant, any producer count) and a linear head emits one logit per
    vocabulary entry. Tests whether the search's committed choice is predictable from the stream
    view alone — the learnability question `named_capture_frac` cannot answer."""

    def __init__(self, n_choices: int, hidden: int = 64) -> None:
        super().__init__()
        self.n_choices = n_choices
        self.global_encoder = _mlp(_PRODUCT_DIM, hidden)
        self.row_encoder = _mlp(_PRODUCER_ROW_DIM, hidden)
        self.choice_head = nn.Linear(2 * hidden, n_choices)

    def forward_view(self, view: ProductStreamView) -> torch.Tensor:
        device = next(self.parameters()).device
        g = self.global_encoder(_tensor([_product_features(view.product_stats)], device))[0]
        rows = [
            _producer_row_features(view.producers[p], view.producer_products[p])
            for p in sorted(view.producers)
        ]
        pooled = self.row_encoder(_tensor(rows, device)).mean(dim=0)
        return self.choice_head(torch.cat([pooled, g]))  # (n_choices,)


class GlobalChoiceModel(nn.Module):
    """Categorical head over the vocabulary from the FULL global state (A0c discriminator).

    Same fit/metrics protocol as `OrdererChoiceModel` (duck-typed `forward_view` on samples
    carrying `.view/.choice/.margin`), but the input is `env.global_state()` — everything the
    search could read at decision time. Inputs are standardized with train-set stats
    (`set_norm`); EXP-0009 showed raw global_state magnitudes wreck optimization."""

    def __init__(self, n_choices: int, state_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.n_choices = n_choices
        self.encoder = _mlp(state_dim, hidden)
        self.choice_head = nn.Linear(hidden, n_choices)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def set_norm(self, states: list[np.ndarray]) -> None:
        arr = np.nan_to_num(np.asarray(states, dtype=np.float32),
                            nan=0.0, posinf=_BOUND, neginf=-_BOUND)
        self.state_mean.copy_(torch.from_numpy(arr.mean(axis=0)))
        self.state_std.copy_(torch.from_numpy(arr.std(axis=0).clip(min=1e-6)))

    def forward_view(self, view: np.ndarray) -> torch.Tensor:
        device = next(self.parameters()).device
        x = _tensor([list(view)], device)[0]
        x = (x - self.state_mean) / self.state_std
        return self.choice_head(self.encoder(x))  # (n_choices,)


def fit_choice_model(model: nn.Module, samples: list, iters: int,
                     lr: float = 1e-3, log=lambda *a, **k: None) -> nn.Module:
    """Cross-entropy fit of the categorical head on committed-choice samples."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    model.train()
    device = next(model.parameters()).device
    for it in range(iters):
        opt.zero_grad()
        logits = torch.stack([model.forward_view(s.view) for s in samples])
        target = torch.tensor([s.choice for s in samples], dtype=torch.long, device=device)
        loss = ce(logits, target)
        loss.backward()
        opt.step()
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            log(f"iter {it}: choice-ce {loss.item():.4f}")
    model.eval()
    return model


def _top1_and_majority(model: nn.Module, samples: list
                       ) -> tuple[float | None, float | None]:
    if not samples:
        return None, None
    with torch.no_grad():
        hits = sum(int(int(model.forward_view(s.view).argmax()) == s.choice) for s in samples)
    counts = np.bincount([s.choice for s in samples], minlength=model.n_choices)
    return hits / len(samples), float(counts.max()) / len(samples)


def choice_metrics(model: nn.Module, samples: list) -> dict:
    """Top-1 accuracy of the categorical head vs the majority-class floor, overall and on
    DECISIVE samples only (margin > 0 — cycles where the search's choice carried signal;
    tie cycles are arbitrary, so accuracy on them is noise, not learnability evidence)."""
    model.eval()
    top1, majority = _top1_and_majority(model, samples)
    decisive = [s for s in samples if s.margin > 0.0]
    top1_d, majority_d = _top1_and_majority(model, decisive)
    return {
        "top1": top1, "majority": majority, "n": len(samples),
        "top1_decisive": top1_d, "majority_decisive": majority_d, "n_decisive": len(decisive),
    }


@dataclass
class ChoiceSample:
    view: ProductStreamView
    choice: int    # committed vocabulary index this cycle (global per cycle, tiled per stream)
    margin: float  # best minus second-best rollout units — 0 = tie, the search picked arbitrarily


@dataclass
class CycleTrace:
    """One per searched cycle (A0t trace analysis, EXP-0036): the FULL candidate score table
    the committed argmax throws away, the champion orderer scored at the SAME state by the
    same lookahead rollout (None when no champion handed in), and a JSON-serializable domain
    snapshot at decision time carrying the per-type detail the encoded views roll up. Together
    they turn aggregate top-1 accuracy into per-cycle REGRET (units[teacher] - units[student])
    and let a human read WHY the teacher chose differently at the cycles that cost units.
    `features` (A0f, opt-in via `with_features`) carries the named decision-time observation
    blocks the missing-observation probe fits on."""
    seed: int
    cycle: int
    cand_units: list[float]   # rollout units per vocabulary entry, insertion order
    choice: int               # argmax (first max wins ties — mirrors _search_best_pol)
    margin: float             # best minus second-best
    champion_units: float | None
    state: dict
    features: dict[str, np.ndarray] | None = None


def _cycle_state_snapshot(env) -> dict:
    """Readable decision-time snapshot for CycleTrace: per-type capacities/reserved/requested,
    inventory + inbound by product, investment flags, the live order book with ages, and open
    consumer demand. Field names follow CONTEXT.md so a trace reads without the encoder."""
    producers = {}
    for pid, p in env.system.producing_agents.items():
        reserved = p.reserved_capacity()
        producers[pid] = {
            "capacities": {t: int(c) for t, c in p.capacities.items()},
            "max_capacities": {t: int(c) for t, c in p.max_capacities.items()},
            "reserved": {t: int(c) for t, c in reserved.items() if c},
            "requested_by_type": {t: int(c) for t, c in p.requested_capacity_by_type.items() if c},
            "inventory": {g: int(n) for g, n in p.inventory.items() if n},
            "inbound": {g: int(n) for g, n in p.outstanding_inbound.items() if n},
            "investing": [name for name, on in (
                ("restoration", p.restoration_running), ("upgrade", p.upgrade_running),
                ("inventory_expansion", p.inventory_expansion_running)) if on],
        }
    orders = [
        {"product": o.product, "qty": int(o.quantity), "accepted": int(o.accepted_qty),
         "completed": int(o.completed_qty), "failed": int(o.failed_qty),
         "tokens": float(o.tokens), "age": int(o.cycle_counter),
         "type": ("consumer" if o.consumer_request is not None
                  else "investment" if o.investment_request is not None
                  else "replenishment")}
        for o in env.system.orders.values()
    ]
    open_demand = {
        prod: [int(q), float(tok)] for prod, (q, tok) in env._open_consumer_demand().items()}
    return {"cycle": int(env.current_cycle), "producers": producers, "orders": orders,
            "open_demand": open_demand}


def _sanitize32(vals: Sequence[float] | np.ndarray) -> np.ndarray:
    """Same NaN/inf policy as `_tensor`, as a flat float32 vector (feature blocks)."""
    return np.nan_to_num(np.asarray(vals, dtype=np.float32),
                         nan=0.0, posinf=_BOUND, neginf=-_BOUND)


def cycle_feature_blocks(env) -> dict[str, np.ndarray]:
    """Named decision-time observation blocks for the A0f missing-observation probe
    (EXP-0036 Task 1 / HANDOVER): identify which observation FAMILY the D1 wall is missing by
    fitting a value-table probe on cumulative block sets and reading held-out regret per set.

    Blocks (deterministic layout: sorted producers x sorted capacity types x sorted products):
    - `gstate`: the lossy ADR-0004 global snapshot — the A0c reference input.
    - `stream`: what the deployed orderer sees, pooled per cycle — mean+max over
      backlog-holding product-stream views of [product features || mean-pooled producer rows].
    - `per_type_capacity`: per producer x type [current, max, reserved, requested] — the
      per-type detail `encode_global` sums away (bottlenecks/death on primal-tight tasks are
      per-type).
    - `order_queue`: per product live order-book aggregates incl. age/TTL composition.
    - `inventory` / `inbound`: per producer x product unit counts (composition, not totals).
    - `relational`: per product x producer recipe-aware ratios (capacity current/free/max vs
      recipe need, input availability, makeable-now) — the product x eligible-producer x
      capacity-type family, collapsed over required types by min (the binding constraint).
    - `compact`: small per-cycle system summary — not a probe block itself; `history_features`
      stacks the previous k cycles of it into the `history` block."""
    from ..domain import metrics as dmetrics

    system = env.system
    producers = sorted(system.producing_agents)
    types = sorted({t for p in system.producing_agents.values() for t in p.max_capacities})
    products = sorted(system.process_sheets)
    orders = list(system.orders.values())

    stream_vecs: list[np.ndarray] = []
    for aid in sorted(system.stream_products):
        view = system.product_stream_view(aid)
        if not view.backlog:
            continue
        rows = _sanitize32([
            _producer_row_features(view.producers[p], view.producer_products[p])
            for p in sorted(view.producers)]).reshape(len(view.producers), -1)
        stream_vecs.append(np.concatenate([
            _sanitize32(_product_features(view.product_stats)), rows.mean(axis=0)]))
    if stream_vecs:
        arr = np.stack(stream_vecs)
        stream = np.concatenate([arr.mean(axis=0), arr.max(axis=0),
                                 np.asarray([len(stream_vecs)], dtype=np.float32)])
    else:
        stream = np.zeros(2 * (_PRODUCT_DIM + _PRODUCER_ROW_DIM) + 1, dtype=np.float32)

    cap: list[float] = []
    for pid in producers:
        p = system.producing_agents[pid]
        reserved = p.reserved_capacity()
        for t in types:
            cap += [p.capacities.get(t, 0), p.max_capacities.get(t, 0),
                    reserved.get(t, 0), p.requested_capacity_by_type.get(t, 0)]

    queue: list[float] = []
    for g in products:
        po = [o for o in orders if o.product == g]
        queue += [
            len(po),
            sum(o.unresolved_quantity() for o in po),
            sum(o.accepted_qty for o in po),
            sum(o.completed_qty for o in po),
            sum(o.failed_qty for o in po),
            sum(o.available_tokens() for o in po),
            max((o.cycle_counter for o in po), default=0),
            sum(o.unresolved_quantity() for o in po if o.cycle_counter >= 1),
            sum(1 for o in po if o.consumer_request is not None),
            sum(1 for o in po if o.replenishment_request is not None),
            sum(1 for o in po if o.investment_request is not None),
        ]
    queue += [len(orders), sum(o.unresolved_quantity() for o in orders)]

    inv = [system.producing_agents[pid].inventory.get(g, 0)
           for pid in producers for g in products]
    inb = [system.producing_agents[pid].outstanding_inbound.get(g, 0)
           for pid in producers for g in products]

    rel: list[float] = []
    for g in products:
        sheet = system.process_sheets[g]
        need = {t: c for t, c in sheet.require.items() if c > 0}
        for pid in producers:
            p = system.producing_agents[pid]
            reserved = p.reserved_capacity()
            cur = min((p.capacities.get(t, 0) / need[t] for t in need), default=_BOUND)
            free = min(((p.capacities.get(t, 0) - reserved.get(t, 0)) / need[t]
                        for t in need), default=_BOUND)
            mx = min((p.max_capacities.get(t, 0) / need[t] for t in need), default=_BOUND)
            inp = (min((p.inventory.get(i, 0) + p.outstanding_inbound.get(i, 0)) / n
                       for i, n in sheet.inputs.items())
                   if sheet.inputs else _BOUND)
            rel += [1.0 if g in p.produces else 0.0,
                    min(cur, _BOUND), max(min(free, _BOUND), -_BOUND),
                    min(mx, _BOUND), min(inp, _BOUND),
                    dmetrics.makeable_qty_now(p, g)]

    totals = {t: [0.0, 0.0, 0.0] for t in types}
    inventory_units = inbound_units = 0
    for pid in producers:
        p = system.producing_agents[pid]
        reserved = p.reserved_capacity()
        inventory_units += p.inventory_units()
        inbound_units += p.outstanding_inbound_units()
        for t in types:
            totals[t][0] += p.capacities.get(t, 0)
            totals[t][1] += reserved.get(t, 0)
            totals[t][2] += p.requested_capacity_by_type.get(t, 0)
    open_demand = env._open_consumer_demand()
    compact = [x for t in types for x in totals[t]] + [
        inventory_units, inbound_units, len(orders),
        sum(o.unresolved_quantity() for o in orders),
        sum(q for q, _ in open_demand.values()),
        sum(tok for _, tok in open_demand.values()),
    ]

    return {
        "gstate": _sanitize32(env.global_state()),
        "stream": _sanitize32(stream),
        "per_type_capacity": _sanitize32(cap),
        "order_queue": _sanitize32(queue),
        "inventory": _sanitize32(inv),
        "inbound": _sanitize32(inb),
        "relational": _sanitize32(rel),
        "compact": _sanitize32(compact),
    }


def history_features(traces: list["CycleTrace"], k: int = 4,
                     block: str = "compact") -> np.ndarray:
    """A0f history block: per trace, the previous `k` cycles' `block` vectors from the SAME
    episode seed, most recent first, zero-padded at episode start (history is the backward
    proxy for what the search's lookahead unrolls forward). Traces must be in collection
    order (per-seed ascending cycles — what `collect_choice_samples` returns)."""
    if not traces:
        return np.zeros((0, 0), dtype=np.float32)
    dim = traces[0].features[block].shape[0]
    out = np.zeros((len(traces), k * dim), dtype=np.float32)
    prev: dict[int, list[np.ndarray]] = {}
    for i, t in enumerate(traces):
        hist = prev.setdefault(t.seed, [])
        for j, vec in enumerate(reversed(hist[-k:])):
            out[i, j * dim:(j + 1) * dim] = vec
        hist.append(t.features[block])
    return out


_REL_PAIR_DIM = 6  # relational block features per (product, producer) pair


def set_features_from_blocks(blocks: dict[str, np.ndarray], *, n_producers: int,
                             n_types: int, n_products: int) -> dict[str, np.ndarray]:
    """Reshape the flat A0f blocks into the set view of the EXP-0036 HANDOVER "Proposed
    Model": a fixed-dim `context` (stream ‖ order_queue ‖ compact) plus one `producers` row
    per producer (its per-type capacities ‖ inventory ‖ inbound ‖ per-product relational
    column). Same information, different factorization — so saved A0f traces re-fit the set
    probe without a pod re-run, and the set-vs-flat regret delta reads as inductive bias."""
    P, T, G = n_producers, n_types, n_products
    rel = blocks["relational"].reshape(G, P, _REL_PAIR_DIM)  # block is product-major
    producers = np.concatenate([
        blocks["per_type_capacity"].reshape(P, 4 * T),
        blocks["inventory"].reshape(P, G),
        blocks["inbound"].reshape(P, G),
        rel.transpose(1, 0, 2).reshape(P, _REL_PAIR_DIM * G),
    ], axis=1)
    context = np.concatenate([blocks["stream"], blocks["order_queue"], blocks["compact"]])
    return {"context": context.astype(np.float32), "producers": producers.astype(np.float32)}


def cycle_set_features(env) -> dict[str, np.ndarray]:
    """Decision-time set view straight from a live env (deploy/collection path); delegates to
    `set_features_from_blocks` so live and trace-reconstructed features are byte-identical."""
    system = env.system
    return set_features_from_blocks(
        cycle_feature_blocks(env),
        n_producers=len(system.producing_agents),
        n_types=len({t for p in system.producing_agents.values() for t in p.max_capacities}),
        n_products=len(system.process_sheets))


class ValueTableModel(nn.Module):
    """A0f value-table probe (EXP-0036 Task 1 / HANDOVER Task 4 target form): regress the
    teacher's FULL per-cycle candidate value table from a flat feature vector, instead of
    classifying its argmax — so near-tied candidates stop injecting label noise and a miss is
    scored by REGRET against the true table, not by exact-argmax accuracy. Inputs are
    standardized with train-set stats (`set_norm`; EXP-0009: raw magnitudes wreck
    optimization); targets are per-row centered at fit time (only within-row differences
    drive the pick)."""

    def __init__(self, state_dim: int, n_choices: int, hidden: int = 64) -> None:
        super().__init__()
        self.n_choices = n_choices
        self.encoder = _mlp(state_dim, hidden)
        self.value_head = nn.Linear(hidden, n_choices)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    def set_norm(self, states: np.ndarray) -> None:
        arr = np.nan_to_num(np.asarray(states, dtype=np.float32),
                            nan=0.0, posinf=_BOUND, neginf=-_BOUND)
        self.state_mean.copy_(torch.from_numpy(arr.mean(axis=0)))
        self.state_std.copy_(torch.from_numpy(arr.std(axis=0).clip(min=1e-6)))

    def forward_states(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.state_mean) / self.state_std
        return self.value_head(self.encoder(x))  # (n, n_choices)


def fit_value_table(model: "ValueTableModel", states: np.ndarray, tables: np.ndarray,
                    iters: int, lr: float = 1e-3, log=lambda *a, **k: None
                    ) -> "ValueTableModel":
    """MSE fit of the value-table probe on per-row-centered, globally-scaled targets."""
    x = torch.from_numpy(np.asarray(states, dtype=np.float32))
    y = np.asarray(tables, dtype=np.float32)
    y = y - y.mean(axis=1, keepdims=True)
    y = torch.from_numpy(y / max(float(y.std()), 1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for it in range(iters):
        opt.zero_grad()
        loss = ((model.forward_states(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            log(f"iter {it}: value-mse {loss.item():.4f}")
    model.eval()
    return model


def value_table_metrics(model: "ValueTableModel", states: np.ndarray, tables: np.ndarray,
                        even_idx: int) -> dict:
    """Regret-first read of a value-table probe (HANDOVER: accuracy is the wrong primary
    metric). Per row: regret = units[true argmax] - units[probe argmax]; headroom =
    units[true argmax] - units[even_split]; `regret_frac` = sum(regret)/sum(headroom).
    `winner_hit` credits any member of the tied-winner set."""
    with torch.no_grad():
        pred = model.forward_states(
            torch.from_numpy(np.asarray(states, dtype=np.float32))).cpu().numpy()
    return _table_pick_metrics(pred, tables, even_idx)


def _table_pick_metrics(pred: np.ndarray, tables: np.ndarray, even_idx: int) -> dict:
    y = np.asarray(tables, dtype=np.float64)
    picks = pred.argmax(axis=1)
    best = y.max(axis=1)
    picked = y[np.arange(len(y)), picks]
    regret = best - picked
    headroom = best - y[:, even_idx]
    n = len(y)
    return {
        "n": n,
        "top1": round(float((picks == y.argmax(axis=1)).mean()), 4) if n else None,
        "winner_hit": round(float((picked >= best - 1e-6).mean()), 4) if n else None,
        "regret_total": round(float(regret.sum()), 2),
        "headroom_total": round(float(headroom.sum()), 2),
        "regret_frac": (round(float(regret.sum() / headroom.sum()), 4)
                        if headroom.sum() > 1e-9 else None),
    }


class ProducerSetValueModel(nn.Module):
    """EXP-0036 HANDOVER "Proposed Model": variable number of producers in, fixed candidate
    value table out. A shared MLP encodes each producer's feature row; masked mean+max
    pooling (permutation-invariant, count-agnostic) aggregates the set; the head reads
    [context ‖ pooled set] into one scalar per routing candidate. Invariants: reordering
    producer rows cannot change the table, producer count never changes model shape, a
    mask-0 (padded/ineligible) row is exactly absent, and the output size is fixed by the
    candidate vocabulary. Inputs standardized with train stats like `ValueTableModel`."""

    def __init__(self, context_dim: int, producer_dim: int, n_choices: int,
                 hidden: int = 64) -> None:
        super().__init__()
        self.n_choices = n_choices
        self.producer_encoder = _mlp(producer_dim, hidden)
        self.head = _mlp(context_dim + 2 * hidden, hidden)
        self.value_head = nn.Linear(hidden, n_choices)
        self.register_buffer("context_mean", torch.zeros(context_dim))
        self.register_buffer("context_std", torch.ones(context_dim))
        self.register_buffer("producer_mean", torch.zeros(producer_dim))
        self.register_buffer("producer_std", torch.ones(producer_dim))

    def set_norm(self, contexts: np.ndarray, producers: np.ndarray) -> None:
        ctx = _sanitize32(contexts).reshape(len(contexts), -1)
        rows = _sanitize32(producers).reshape(-1, self.producer_mean.shape[0])
        self.context_mean.copy_(torch.from_numpy(ctx.mean(axis=0)))
        self.context_std.copy_(torch.from_numpy(ctx.std(axis=0).clip(min=1e-6)))
        self.producer_mean.copy_(torch.from_numpy(rows.mean(axis=0)))
        self.producer_std.copy_(torch.from_numpy(rows.std(axis=0).clip(min=1e-6)))

    def forward_tables(self, contexts: torch.Tensor, producers: torch.Tensor,
                       mask: torch.Tensor | None = None) -> torch.Tensor:
        """contexts (n, dc), producers (n, P, dp), mask (n, P) with 1=real row; any P."""
        h = self.producer_encoder((producers - self.producer_mean) / self.producer_std)
        if mask is None:
            mask = torch.ones(producers.shape[:2], dtype=h.dtype, device=h.device)
        m = mask.unsqueeze(-1)
        pooled_mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        pooled_max = (h + (m - 1.0) * 1e9).max(dim=1).values
        ctx = (contexts - self.context_mean) / self.context_std
        return self.value_head(self.head(
            torch.cat([ctx, pooled_mean, pooled_max], dim=1)))  # (n, n_choices)


def fit_set_value_table(model: "ProducerSetValueModel", contexts: np.ndarray,
                        producers: np.ndarray, tables: np.ndarray, iters: int,
                        lr: float = 1e-3, mask: np.ndarray | None = None,
                        log=lambda *a, **k: None) -> "ProducerSetValueModel":
    """Set-view twin of `fit_value_table`: same per-row-centered, globally-scaled MSE target,
    so set-vs-flat regret deltas on the same traces read as inductive bias, not objective."""
    ctx = torch.from_numpy(np.asarray(contexts, dtype=np.float32))
    rows = torch.from_numpy(np.asarray(producers, dtype=np.float32))
    msk = None if mask is None else torch.from_numpy(np.asarray(mask, dtype=np.float32))
    y = np.asarray(tables, dtype=np.float32)
    y = y - y.mean(axis=1, keepdims=True)
    y = torch.from_numpy(y / max(float(y.std()), 1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for it in range(iters):
        opt.zero_grad()
        loss = ((model.forward_tables(ctx, rows, msk) - y) ** 2).mean()
        loss.backward()
        opt.step()
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            log(f"iter {it}: set-value-mse {loss.item():.4f}")
    model.eval()
    return model


def set_value_table_metrics(model: "ProducerSetValueModel", contexts: np.ndarray,
                            producers: np.ndarray, tables: np.ndarray, even_idx: int,
                            mask: np.ndarray | None = None) -> dict:
    """Regret-first read of the set probe — same keys/semantics as `value_table_metrics`."""
    with torch.no_grad():
        pred = model.forward_tables(
            torch.from_numpy(np.asarray(contexts, dtype=np.float32)),
            torch.from_numpy(np.asarray(producers, dtype=np.float32)),
            None if mask is None else torch.from_numpy(
                np.asarray(mask, dtype=np.float32))).cpu().numpy()
    return _table_pick_metrics(pred, tables, even_idx)


# --- Task 5: window/episode-level blend-policy value table (EXP-0036 HANDOVER) ---
# A0f/A0g predicted a PER-CYCLE candidate table whose winner ordering did not transfer
# across demand seeds. Task 5 changes the target time scale: the value of each CONSTANT
# blend policy over an episode/window. The set head (ProducerSetValueModel) is reused with
# one row per window instead of per searched cycle; deployment picks ONE constant blend via
# a conservative switch rule rather than chasing per-cycle winner flips.


def aggregate_set_window(feats: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Summarize a window of per-cycle set features (each `{context, producers}` from
    `cycle_set_features` / `set_features_from_blocks`) into one window row. Context is
    concat[mean-over-window, last-in-window] (the regime's average state plus where it
    ended up — the same summary the deployed policy computes from its rolling buffer);
    producer rows are mean-over-window (per-producer, so still permutation-set-shaped).
    Producer count P is fixed within a topology, so the stack is well defined."""
    if not feats:
        raise ValueError("aggregate_set_window needs at least one cycle of features")
    ctx = np.stack([f["context"] for f in feats]).astype(np.float32)
    prod = np.stack([f["producers"] for f in feats]).astype(np.float32)  # (W, P, dp)
    context = np.concatenate([ctx.mean(axis=0), ctx[-1]]).astype(np.float32)
    return {"context": context, "producers": prod.mean(axis=0).astype(np.float32)}


def blend_decision(table: np.ndarray, default_idx: int, threshold: float) -> int:
    """Conservative deploy rule (HANDOVER Task 5): stay on the default blend unless another
    candidate's predicted value beats the default by more than `threshold` — then take the
    best such candidate (first max wins ties). `threshold` is in the model's output units
    (the centered, globally std-scaled value table it was fit on), so it reads as "switch
    only when the advantage over the default is large", not a raw-unit gap."""
    t = np.asarray(table, dtype=np.float64).reshape(-1)
    adv = t - t[default_idx]
    better = np.where(adv > threshold)[0]
    if better.size == 0:
        return default_idx
    return int(better[np.argmax(t[better])])


def set_table_ordering_policy(model: "ProducerSetValueModel", vocab: dict[str, "Callable"],
                              default_idx: int, threshold: float, window: int
                              ) -> "Callable":
    """Deploy the window value-table head as a meta-orderer: each cycle, aggregate the last
    `window` cycles of set features (matching the training-time window summary), predict the
    blend-policy value table, pick ONE constant blend via `blend_decision`, and act with it
    for the whole cycle. The default blend is held until a switch clears the threshold — the
    "do not chase noisy cycle-by-cycle winner flips" rule. `model` is topology-specific (its
    context/producer dims match the task it was fit on), like the A0f/A0g probes."""
    names = list(vocab)
    model.eval()
    state: dict = {"cycle": None, "hist": [], "pol": vocab[names[default_idx]]}

    def policy(env, agent: str) -> np.ndarray:
        if env.current_cycle != state["cycle"]:
            state["cycle"] = env.current_cycle
            state["hist"].append(cycle_set_features(env))
            win = aggregate_set_window(state["hist"][-window:])
            with torch.no_grad():
                pred = model.forward_tables(
                    torch.from_numpy(win["context"][None].astype(np.float32)),
                    torch.from_numpy(win["producers"][None].astype(np.float32))).cpu().numpy()[0]
            state["pol"] = vocab[names[blend_decision(pred, default_idx, threshold)]]
        return state["pol"](env, agent)

    return policy


@dataclass
class GlobalChoiceSample:
    """One per cycle: the FULL global state (ADR-0004 centralized-critic input) paired with the
    committed vocabulary choice. Field named `view` so `fit_choice_model`/`choice_metrics`
    duck-type across both sample kinds. A0c discriminator: if a global-state classifier beats
    the majority floor where the stream-view one failed, the D1 wall is missing CONTEXT (enrich
    the obs); if it fails too, the choice is rollout NOISE (soft labels / deploy-time search)."""
    view: np.ndarray  # env.global_state() snapshot at the searched cycle, float32
    choice: int
    margin: float


def collect_choice_samples(
    configs, seeds, *, prod, alphas: Sequence[float] = (0.75, 0.5, 0.25),
    lookahead: int = 6, horizon: int = 100, champion=None, with_features: bool = False,
    log=lambda *a, **k: None,
) -> tuple[list["ChoiceSample"], list["GlobalChoiceSample"], list["CycleTrace"], list[str]]:
    """Categorical-teacher dataset (A0b / D1): drive each `(config, seed)` under the frozen
    producer + the per-cycle search restricted to the named+blend vocabulary (`blend_vocabulary`,
    n_samples=0), and snapshot every backlog-holding product-stream view paired with WHICH
    vocabulary entry the search committed plus the decisiveness margin.

    `named_capture_frac` (A0) only proved the headroom is reachable inside this vocabulary via a
    lookahead oracle; this dataset tests the separate learnability question — is the committed
    choice predictable from the stream observation alone? Ties (margin 0) are recorded, not
    dropped, so metrics can split decisive vs tie-arbitrary rows. The same pass also snapshots
    `env.global_state()` once per cycle (only cycles that yielded stream samples, so both probes
    share a cycle population) — the A0c missing-context vs choice-noise discriminator — and a
    `CycleTrace` per cycle (full candidate table + state snapshot; `champion` scored by the same
    lookahead rollout when provided) — the A0t regret dataset. Returns `(samples, global
    samples, traces, vocab names)`; the choice index follows vocabulary insertion order."""
    import copy

    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.orderer_oracle import _rollout_units, _score_candidates, blend_vocabulary

    configs, seeds = list(configs), list(seeds)
    if len(configs) != len(seeds):
        raise ValueError(f"configs ({len(configs)}) must align 1:1 with seeds ({len(seeds)})")
    vocab = blend_vocabulary(alphas)
    names = list(vocab)
    samples: list[ChoiceSample] = []
    global_samples: list[GlobalChoiceSample] = []
    traces: list[CycleTrace] = []
    for cfg, seed in zip(configs, seeds):
        env = TokenomicsAECEnv(config=cfg, horizon=horizon, entity_producer_obs=True,
                               product_stream_ordering=True, corrected_stream_k=True)
        env.reset(seed=seed)
        adim = env.spec_.ordering_action_dim
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        cur = -1
        choice, margin = 0, 0.0
        order_pol = vocab[names[0]]
        pending: tuple[GlobalChoiceSample, CycleTrace] | None = None
        n_at_cycle = len(samples)
        for agent in env.agent_iter(max_iter=max_iter):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            if env.current_cycle != cur:  # re-search the committed vocabulary entry once per cycle
                if pending is not None and len(samples) > n_at_cycle:
                    global_samples.append(pending[0])
                    traces.append(pending[1])
                cur = env.current_cycle
                snapshot = _cycle_state_snapshot(env)  # decision-time state, pre-search
                feats = cycle_feature_blocks(env) if with_features else None  # pre-search too
                scored = _score_candidates(env, prod, vocab, 0, 1.0, lookahead, seed, cur, adim)
                units = [u for _, _, u in scored]  # vocabulary insertion order preserved
                choice = int(np.argmax(units))     # first max wins ties (mirrors _search_best_pol)
                ranked = sorted(units, reverse=True)
                margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0
                champ_units = (float(_rollout_units(copy.deepcopy(env), prod, champion, lookahead))
                               if champion is not None else None)
                order_pol = scored[choice][1]
                pending = (
                    GlobalChoiceSample(
                        np.asarray(env.global_state(), dtype=np.float32), choice, margin),
                    CycleTrace(seed=seed, cycle=cur, cand_units=[float(u) for u in units],
                               choice=choice, margin=margin, champion_units=champ_units,
                               state=snapshot, features=feats))
                n_at_cycle = len(samples)
            role, aid = env._role[agent]
            if role == "order":
                view = env.system.product_stream_view(aid)
                if view.backlog:
                    pids = sorted(view.producers)
                    if any(view.producer_products[p].eligible > 0 for p in pids):
                        samples.append(ChoiceSample(view, choice, margin))
            env.step(prod(env, agent) if role == "prod" else order_pol(env, agent))
        if pending is not None and len(samples) > n_at_cycle:
            global_samples.append(pending[0])
            traces.append(pending[1])
        log(f"seed {seed}: {len(samples)} cumulative choice samples "
            f"({len(global_samples)} global)")
    return samples, global_samples, traces, names


def collect_deploy_traces(
    configs, seeds, *, prod, champion, alphas: Sequence[float] = (0.75, 0.5, 0.25),
    lookahead: int = 6, horizon: int = 100, log=lambda *a, **k: None,
) -> tuple[list[dict], list[str]]:
    """Champion-trajectory failure chains (A0u, EXP-0036): drive each `(config, seed)` with the
    CHAMPION orderer — the badly-played game itself — and at every cycle of its OWN trajectory
    score the blend vocabulary AND the champion by the same lookahead rollout. The per-cycle
    counterfactual (`choice` = what the teacher would pick HERE, `champion_units` = what the
    champion's continuation earns) localizes where the game goes wrong; the snapshot (plus a
    cumulative `progress` line) explains it. Complements `collect_choice_samples`, whose traces
    follow the TEACHER's trajectory and so never visit the champion's failure states.

    Returns `(episodes, vocab names)`; each episode carries seed, outcome stats, and its
    `CycleTrace` list (every searched cycle — no stream gating; a failure chain needs ALL)."""
    import copy

    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.orderer_oracle import _rollout_units, _score_candidates, blend_vocabulary

    configs, seeds = list(configs), list(seeds)
    if len(configs) != len(seeds):
        raise ValueError(f"configs ({len(configs)}) must align 1:1 with seeds ({len(seeds)})")
    vocab = blend_vocabulary(alphas)
    names = list(vocab)
    episodes: list[dict] = []
    for cfg, seed in zip(configs, seeds):
        env = TokenomicsAECEnv(config=cfg, horizon=horizon, entity_producer_obs=True,
                               product_stream_ordering=True, corrected_stream_k=True)
        env.reset(seed=seed)
        adim = env.spec_.ordering_action_dim
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        cur = -1
        traces: list[CycleTrace] = []
        for agent in env.agent_iter(max_iter=max_iter):
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            if env.current_cycle != cur:  # counterfactual search once per champion cycle
                cur = env.current_cycle
                snapshot = _cycle_state_snapshot(env)
                snapshot["progress"] = {
                    "units_delivered": int(env.episode_stats["consumer_completed_units"]),
                    "units_rejected": int(env.episode_stats["consumer_rejected_units"]),
                }
                scored = _score_candidates(env, prod, vocab, 0, 1.0, lookahead, seed, cur, adim)
                units = [float(u) for _, _, u in scored]
                choice = int(np.argmax(units))
                ranked = sorted(units, reverse=True)
                margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0
                champ_units = float(_rollout_units(copy.deepcopy(env), prod, champion, lookahead))
                traces.append(CycleTrace(seed=seed, cycle=cur, cand_units=units, choice=choice,
                                         margin=margin, champion_units=champ_units,
                                         state=snapshot))
            role, _aid = env._role[agent]
            env.step(prod(env, agent) if role == "prod" else champion(env, agent))
        episodes.append({
            "seed": seed, "traces": traces,
            "consumer_completed_units": int(env.episode_stats["consumer_completed_units"]),
            "consumer_rejected_units": int(env.episode_stats["consumer_rejected_units"]),
            "dead": bool(env.episode_stats["dead"]),
            "cycles": int(env.episode_stats["cycles"]),
        })
        log(f"seed {seed}: {len(traces)} deploy traces, "
            f"units {episodes[-1]['consumer_completed_units']}, dead {episodes[-1]['dead']}")
    return episodes, names


def fit_scorer(model: "OrdererScoreModel", samples: list["ScorerSample"], iters: int,
               lr: float = 1e-3, log=lambda *a, **k: None) -> "OrdererScoreModel":
    """Masked MSE distillation: scorer scores -> teacher logits over eligible rows."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    device = next(model.parameters()).device
    for it in range(iters):
        opt.zero_grad()
        loss = torch.zeros((), dtype=torch.float32, device=device)
        n = 0
        for s in samples:
            pred = model.forward_view(s.view)  # on model device
            m = torch.tensor(s.mask, dtype=torch.bool, device=device)
            if not m.any():
                continue
            tgt = torch.tensor(s.target, dtype=torch.float32, device=device)
            loss = loss + ((pred[m] - tgt[m]) ** 2).sum()
            n += int(m.sum())
        loss = loss / max(n, 1)
        loss.backward()
        opt.step()
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            log(f"iter {it}: masked-mse {loss.item():.4f}")
    model.eval()
    return model


def masked_variance_explained(target: np.ndarray, pred: np.ndarray) -> float | None:
    """Variance the fit explains over a route-uniform baseline, on already-masked (eligible-only)
    entries (Stage A0 / EXP-0036 diagnostic). `1 - mean((pred-target)^2) / mean(target^2)`.

    Targets are centered per stream by `_center_clip_eligible`, so `mean(target^2)` is exactly the
    residual a zero (route-uniform) predictor leaves — the denominator the diagnosis recomputed by
    hand. `None` when the target carries no variance (degenerate: nothing to explain)."""
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    zero_mse = float((target ** 2).mean()) if target.size else 0.0
    if zero_mse <= 0.0:
        return None
    model_mse = float(((pred - target) ** 2).mean())
    return 1.0 - model_mse / zero_mse


def scorer_sample_metrics(model: "OrdererScoreModel", samples: list["ScorerSample"]) -> dict:
    """Training-state diagnostics for an orderer scorer fit.

    These metrics evaluate the fitted scorer on the exact teacher states it trained on.
    If they are good but rollout units collapse, the failure is deploy-time dynamics
    rather than inability to regress the logged targets.
    """
    model.eval()
    sse = 0.0
    n = 0
    top1 = 0
    top1_n = 0
    masked_preds: list[np.ndarray] = []
    masked_targets: list[np.ndarray] = []
    with torch.no_grad():
        for s in samples:
            pred = model.forward_view(s.view).detach().cpu().numpy()
            mask = np.asarray(s.mask, dtype=bool)
            if not mask.any():
                continue
            target = np.asarray(s.target, dtype=np.float32)
            diff = pred[mask] - target[mask]
            sse += float((diff ** 2).sum())
            n += int(mask.sum())
            masked_preds.append(pred[mask])
            masked_targets.append(target[mask])

            elig = np.flatnonzero(mask)
            if len(elig) > 0:
                pred_top = int(elig[int(np.argmax(pred[elig]))])
                target_top = int(elig[int(np.argmax(target[elig]))])
                top1 += int(pred_top == target_top)
                top1_n += 1
    ve = (masked_variance_explained(np.concatenate(masked_targets), np.concatenate(masked_preds))
          if masked_targets else None)
    return {
        "sample_masked_mse": round(sse / max(n, 1), 6),
        "sample_masked_n": n,
        "sample_variance_explained": round(ve, 6) if ve is not None else None,
        "sample_top1_match": round(top1 / top1_n, 6) if top1_n else None,
        "sample_top1_n": top1_n,
    }


def save_scorer(model: "OrdererScoreModel", path: str, *,
                config_fingerprint: str | None = None) -> None:
    """Save an O2 orderer scorer checkpoint with its portable feature `schema` (step-13 load
    gate). `config_fingerprint`, if given, is audit metadata only — never a load gate."""
    ckpt = {
        "scorer_version": ORDERER_SCORER_CHECKPOINT_VERSION,
        "hidden": model.hidden,
        "product_dim": _PRODUCT_DIM,
        "producer_row_dim": _PRODUCER_ROW_DIM,
        "state": model.state_dict(),
        "schema": orderer_portable_schema().to_dict(),
    }
    if config_fingerprint is not None:
        ckpt["config_fingerprint"] = config_fingerprint
    torch.save(ckpt, path)


def load_scorer(path: str) -> "OrdererScoreModel":
    """Load an O2 orderer scorer checkpoint.

    Gates on the portable feature `schema` when present (step 13): a feature/entity/relation/
    decoder/architecture change rejects by name. Legacy checkpoints without the block (the
    live EXP-0012 orderer) still load through the version + feature-dim floor. The
    `config_fingerprint` (if any) is ignored on load."""
    from .portable_schema import PortableSchema, assert_compatible

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["scorer_version"] != ORDERER_SCORER_CHECKPOINT_VERSION:
        raise ValueError(f"scorer checkpoint version {ckpt['scorer_version']} != {ORDERER_SCORER_CHECKPOINT_VERSION}")
    if ckpt["producer_row_dim"] != _PRODUCER_ROW_DIM or ckpt["product_dim"] != _PRODUCT_DIM:
        raise ValueError("scorer feature schema mismatch")
    if "schema" in ckpt:
        assert_compatible(PortableSchema.from_dict(ckpt["schema"]), orderer_portable_schema())
    model = OrdererScoreModel(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def eval_scorer(model: "OrdererScoreModel", depth: int, seeds, eval_seeds: int, horizon: int,
                log=lambda *a, **k: None) -> list[dict]:
    """Score the scorer+allocator ordering policy against the `proportional` baseline on
    held-out scale topologies (which vary producer count) — the O2 value + producer-count
    transfer check. Producer fixed to `stock_aware`."""
    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
    from ..eval.scale import build_scale_config

    learned = scorer_ordering_policy(model)
    prod_pol = PRODUCING_BASELINES["stock_aware"]
    base_order = ORDERING_BASELINES["proportional"]
    rows: list[dict] = []
    for seed in seeds:
        cfg = build_scale_config(seed, depth=depth)
        def mean_units(order_pol):
            tot = 0.0
            for es in range(eval_seeds):
                env = TokenomicsAECEnv(config=cfg, horizon=horizon,
                                       product_stream_ordering=True, corrected_stream_k=True)
                tot += run_episode(env, prod_pol, order_pol, seed=es)["consumer_completed_units"]
            return tot / eval_seeds
        learned_u, base_u = mean_units(learned), mean_units(base_order)
        P = TokenomicsAECEnv(config=cfg, horizon=1).spec_.num_producers
        ratio = learned_u / base_u if base_u else float("nan")
        rows.append({"seed": seed, "producers": P, "scorer_units": round(learned_u, 2),
                     "proportional_units": round(base_u, 2), "ratio": round(ratio, 4)})
        log(f"seed {seed} (P={P}): scorer {learned_u:.1f} vs proportional {base_u:.1f} -> {ratio:.3f}")
    return rows


def distill_orderer_scorer(depth: int, train_seeds, eval_seeds_list, iters: int, eval_seeds: int,
                           horizon: int, save: str | None, hidden: int = 64,
                           log=print, train_configs=None) -> tuple["OrdererScoreModel", list[dict]]:
    train_seeds = list(train_seeds)
    samples = collect_scorer_samples(train_seeds, depth, horizon, log=log, configs=train_configs)
    log(f"collected {len(samples)} scorer samples over {len(train_seeds)} topologies")
    model = OrdererScoreModel(hidden=hidden)
    fit_scorer(model, samples, iters=iters, log=log)
    if save:
        save_scorer(model, save)
        log(f"saved scorer -> {save}")
    rows = eval_scorer(model, depth, eval_seeds_list, eval_seeds, horizon, log=log)
    return model, rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--train-seeds", type=str, default="0-5", help="scale seeds for distillation, e.g. 0-5")
    ap.add_argument("--eval-seeds-list", type=str, default="6-9", help="held-out scale seeds for value/transfer")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=5, help="episodes per held-out topology")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    def _range(s: str):
        a, b = s.split("-")
        return range(int(a), int(b) + 1)

    _, rows = distill_orderer_scorer(
        depth=args.depth, train_seeds=_range(args.train_seeds),
        eval_seeds_list=_range(args.eval_seeds_list), iters=args.iters,
        eval_seeds=args.eval_seeds, horizon=args.horizon, save=args.save, hidden=args.hidden,
    )
    print("\nseed,producers,scorer_units,proportional_units,ratio")
    for r in rows:
        print(f"{r['seed']},{r['producers']},{r['scorer_units']},{r['proportional_units']},{r['ratio']}")
    ratios = [r["ratio"] for r in rows]
    print(f"\nmean ratio vs proportional: {sum(ratios)/len(ratios):.4f}; "
          f"min {min(ratios):.4f}; producer counts {sorted({r['producers'] for r in rows})}")


if __name__ == "__main__":
    main()
