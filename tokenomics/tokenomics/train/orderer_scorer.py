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

from collections.abc import Sequence
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
