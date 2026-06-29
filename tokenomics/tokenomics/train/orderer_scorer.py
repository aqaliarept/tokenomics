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


@dataclass
class ScorerSample:
    view: ProductStreamView
    target: np.ndarray  # per-producer teacher logits (sorted ids)
    mask: np.ndarray    # eligibility (sorted ids)


def collect_scorer_samples(seeds, depth: int, horizon: int, log=lambda *a, **k: None) -> list["ScorerSample"]:
    """Roll `stock_aware` + `proportional` across the given scale topologies and
    snapshot every product-stream view that holds a backlog, paired with the teacher
    target. On-teacher BC: the proportional orderer also drives the states."""
    from ..env.aec_env import TokenomicsAECEnv
    from ..eval.baselines import PRODUCING_BASELINES, ORDERING_BASELINES
    from ..eval.scale import build_scale_config

    prod_pol, order_pol = PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["proportional"]
    samples: list[ScorerSample] = []
    for seed in seeds:
        cfg = build_scale_config(seed, depth=depth)
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


def save_scorer(model: "OrdererScoreModel", path: str) -> None:
    torch.save({
        "scorer_version": ORDERER_SCORER_CHECKPOINT_VERSION,
        "hidden": model.hidden,
        "product_dim": _PRODUCT_DIM,
        "producer_row_dim": _PRODUCER_ROW_DIM,
        "state": model.state_dict(),
    }, path)


def load_scorer(path: str) -> "OrdererScoreModel":
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["scorer_version"] != ORDERER_SCORER_CHECKPOINT_VERSION:
        raise ValueError(f"scorer checkpoint version {ckpt['scorer_version']} != {ORDERER_SCORER_CHECKPOINT_VERSION}")
    if ckpt["producer_row_dim"] != _PRODUCER_ROW_DIM or ckpt["product_dim"] != _PRODUCT_DIM:
        raise ValueError("scorer feature schema mismatch")
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
                           log=print) -> tuple["OrdererScoreModel", list[dict]]:
    samples = collect_scorer_samples(train_seeds, depth, horizon, log=log)
    log(f"collected {len(samples)} scorer samples over {len(list(train_seeds))} topologies")
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
