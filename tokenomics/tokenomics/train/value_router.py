"""EXP-0037 value-routed orderer: the trainable residual score model + deploy adapter.

The measurement half (residual router, marginal-value teacher, gate harness) is torch-free
in `eval.value_router`. This module adds the learned piece: a producer-count-agnostic
per-producer-row scorer trained on the marginal-value-of-logit-bump labels, deployed as the
residual over fixed `blend_0.5` (never the throughput teacher of EXP-0012, never the noisy
per-cycle search vector of EXP-0032 — HANDOVER non-goals).

Feature rows are white-box (read `env.system`, like the ordering baselines) and shared across
producers, so the model is order-equivariant and independent of producer count. The label is
the short-rollout marginal value of nudging one producer's blend_0.5 logit — an IMPROVEMENT
signal, not throughput imitation. Acceptance stays deployed rollout vs blend_0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from ..domain import metrics as dmetrics
from ..env import TokenomicsAECEnv
from ..eval.baselines import Policy, PRODUCING_BASELINES, ordering_blend
from ..eval.curriculum import TaskRecord, _episode_seeds, replay
from ..eval.joint_stack import joint_stack_env
from ..eval.value_router import (
    ScoreFn,
    marginal_value_labels,
    router_gate,
)

_BOUND = 10.0
PRODUCER_FEATURES = 7  # keep in sync with `producer_row_features`


# --- white-box per-(stream, producer) feature rows -----------------------------------
def producer_row_features(env: TokenomicsAECEnv, oid: str, view) -> np.ndarray:
    """One feature row per runtime producer (in `spec_.producer_ids` order) for order `oid`:
    the value-relevant facts a capacity-shaped blend cannot see (HANDOVER data list). Ineligible
    producers get a zero row; the eligibility bit is feature 0. Ratios are per required capacity
    type, collapsed by the binding (min) constraint, clipped to `_BOUND`."""
    sp = env.spec_
    oi = view.incoming[oid]
    sheet = env.system.process_sheets[oi.product]
    need = {t: c for t, c in sheet.require.items() if c > 0}
    eligible = view.eligible[oid]
    rows = np.zeros((sp.num_producers, PRODUCER_FEATURES), dtype=np.float32)
    for pid in eligible:
        p = env.system.producing_agents[pid]
        reserved = p.reserved_capacity()
        free = min(((p.capacities.get(t, 0) - reserved.get(t, 0)) / c for t, c in need.items()),
                   default=_BOUND)
        resv = min((reserved.get(t, 0) / c for t, c in need.items()), default=0.0)
        head = min(((p.max_capacities.get(t, 0) - p.capacities.get(t, 0)) / c
                    for t, c in need.items()), default=0.0)
        inp = (min((p.inventory.get(i, 0) + p.outstanding_inbound.get(i, 0)) / n
                   for i, n in sheet.inputs.items()) if sheet.inputs else _BOUND)
        rows[sp.producer_ids.index(pid)] = [
            1.0,
            float(np.clip(dmetrics.makeable_qty_now(p, oi.product), 0, _BOUND)),
            float(np.clip(free, -_BOUND, _BOUND)),
            float(np.clip(resv, 0, _BOUND)),
            float(np.clip(head, 0, _BOUND)),
            float(np.clip(inp, 0, _BOUND)),
            float(np.clip(p.inventory.get(oi.product, 0), 0, _BOUND)),
        ]
    return rows


# --- model ---------------------------------------------------------------------------
class ResidualScoreModel(nn.Module):
    """Shared per-producer-row MLP -> one residual logit per producer. Applied row-wise, so it
    is producer-count agnostic and order-equivariant (A0g inductive bias, fresh residual head)."""

    def __init__(self, feat_dim: int = PRODUCER_FEATURES, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        # rows: [..., P, F] -> [..., P]
        return self.net(rows).squeeze(-1)


@dataclass
class RouterSample:
    rows: np.ndarray   # [P, F] producer feature rows
    label: np.ndarray  # [P] marginal-value labels (0 for ineligible)
    mask: np.ndarray   # [P] 1.0 for eligible producers


def fit_residual_scorer(
    model: ResidualScoreModel, samples: Sequence[RouterSample], *,
    epochs: int, lr: float = 1e-2, log: Callable[..., None] = lambda *a, **k: None,
) -> list[float]:
    """Masked MSE regression of per-producer scores onto marginal-value labels (centered per
    sample over eligible producers, so the model learns *relative* preference, not scale).
    Returns per-epoch mean loss. Ineligible producers are masked out of the gradient.

    The gate spans tasks of different topology (different producer counts), so per-sample rows are
    `[P_i, F]` with varying `P_i`. We pad every sample to `Pmax` with zero rows and mask 0 — the
    padded rows are inert in the masked MSE and the mask-weighted centering, and the model is a
    shared per-row MLP that ignores padding — so the batch is rectangular without contaminating
    the loss."""
    pmax = max(s.rows.shape[0] for s in samples)

    def _pad2d(a: np.ndarray) -> np.ndarray:
        out = np.zeros((pmax, a.shape[1]), dtype=np.float32)
        out[: a.shape[0]] = a
        return out

    def _pad1d(a: np.ndarray) -> np.ndarray:
        out = np.zeros(pmax, dtype=np.float32)
        out[: a.shape[0]] = a
        return out

    rows = torch.as_tensor(np.stack([_pad2d(s.rows) for s in samples]), dtype=torch.float32)
    mask = torch.as_tensor(np.stack([_pad1d(s.mask) for s in samples]), dtype=torch.float32)
    lab = np.stack([_pad1d(s.label) for s in samples]).astype(np.float32)
    # center labels over eligible producers per sample (relative marginal preference)
    m = np.stack([_pad1d(s.mask) for s in samples]).astype(np.float32)
    denom = np.clip(m.sum(axis=1, keepdims=True), 1.0, None)
    lab = lab - (lab * m).sum(axis=1, keepdims=True) / denom * m
    target = torch.as_tensor(lab, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(rows)
        se = ((pred - target) ** 2) * mask
        loss = se.sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        log(f"epoch {ep} loss {loss.item():.4f}")
    return losses


# --- deploy adapter: model -> ScoreFn over the k_stream*P action layout ---------------
def to_score_fn(model: ResidualScoreModel) -> ScoreFn:
    """Wrap a trained model as a `ScoreFn`: per order-stream, score its producers from their
    feature rows and place the scores into the flat `k_stream*P` residual (0 for ineligible /
    absent streams), matching the ordering-action layout decode expects."""

    def score_fn(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        sp = env.spec_
        _, aid = env._role[agent]
        view = env.system.ordering_agents[aid].view(env.system.producer_infos)
        mat = np.zeros((sp.k_stream, sp.num_producers), dtype=np.float32)
        with torch.no_grad():
            for k, oid in enumerate(sorted(view.incoming)):
                if k >= sp.k_stream:
                    break
                rows = producer_row_features(env, oid, view)
                elig = np.array([pid in view.eligible[oid] for pid in sp.producer_ids])
                scores = model(torch.as_tensor(rows, dtype=torch.float32)).numpy()
                mat[k] = scores * elig
        return mat.flatten()

    return score_fn


# --- offline label collection (drive blend_0.5, label each decision) -----------------
def collect_router_samples(
    records: Sequence[TaskRecord], prod: Policy, *,
    delta: float, lookahead: int, eval_seeds: int, horizon: int, base: "Policy | None" = None,
    log: Callable[..., None] = lambda *a, **k: None,
) -> list[RouterSample]:
    """Drive the fixed `blend_0.5` trajectory on each task/seed and, at every ordering decision
    with >= 2 eligible producers, record (features, marginal-value labels, eligibility mask).
    The labels come from `marginal_value_labels` (short-rollout value of a logit bump), so the
    training distribution is blend_0.5's own states — the residual only has to correct there.
    Per (task, seed) progress is logged: this phase is the run's bottleneck (a short rollout per
    eligible producer per decision) and prints nothing else until it finishes."""
    base = base if base is not None else ordering_blend(0.5)
    samples: list[RouterSample] = []
    n_ts = len(records) * eval_seeds
    done = 0
    t0 = time.time()
    for ti, rec in enumerate(records):
        cfg = replay(rec)
        for s in _episode_seeds(rec.episode_seed, eval_seeds):
            before = len(samples)
            # SAME env contract as the gate/EXP-0036 floor (joint_stack_env), so labels are
            # collected on the states the deployed router actually visits.
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
                for k, oid in enumerate(sorted(view.incoming)):
                    if k >= env.spec_.k_stream:
                        break
                    elig = view.eligible[oid]
                    if len(elig) < 2:
                        continue
                    cols = [env.spec_.producer_ids.index(p) for p in elig]
                    label = marginal_value_labels(env, prod, base, agent, k, cols, delta, lookahead)
                    mask = np.zeros(env.spec_.num_producers, dtype=np.float32)
                    mask[cols] = 1.0
                    samples.append(RouterSample(producer_row_features(env, oid, view), label, mask))
                env.step(base(env, agent))
            done += 1
            log(f"[collect] task {ti + 1}/{len(records)} seed {s}: +{len(samples) - before} "
                f"({len(samples)} total) | {done}/{n_ts} task-seeds done, {time.time() - t0:.0f}s")
    return samples


# --- end-to-end driver ---------------------------------------------------------------
def run_value_router(
    records: Sequence[TaskRecord], *, prod: Policy, delta: float, lookahead: int,
    eval_seeds: int, horizon: int, epochs: int, clip: float, out_dir: str, seed: int = 0,
    log: Callable[..., None] = print,
) -> list[dict]:
    """Collect marginal-value labels on `blend_0.5` states, fit the residual scorer, deploy it
    as the residual router, and gate it against fixed `blend_0.5` by rollout on the SAME tasks.
    Writes the EXP layout (`spec.json`, `summary.csv`, per-epoch `loss_history.csv`, per-seed
    `gate_rows.csv`) and returns the gate rows. This is the probe's tiny local smoke; the real
    s29/s77/s45 gate runs on a pod.

    `seed` seeds torch model init so the fit — and therefore `learned_cu` and the 2-of-3 stop
    count — is reproducible (recorded in `spec.json`). The persisted loss history and per-seed
    rows make the root-cause evidence (flat loss, per-seed / death spread) auditable without
    re-running the pod."""
    samples = collect_router_samples(
        records, prod, delta=delta, lookahead=lookahead, eval_seeds=eval_seeds, horizon=horizon,
        log=log)
    log(f"collected {len(samples)} labeled decisions")
    torch.manual_seed(seed)
    model = ResidualScoreModel()
    losses = fit_residual_scorer(model, samples, epochs=epochs, log=log)
    rows, per_seed = router_gate(
        records, prod, to_score_fn(model), clip=clip, eval_seeds=eval_seeds, horizon=horizon)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump({"task_ids": [r.task_id for r in records], "delta": delta,
                   "lookahead": lookahead, "eval_seeds": eval_seeds, "horizon": horizon,
                   "epochs": epochs, "clip": clip, "floor": "blend_0.5", "seed": seed,
                   "n_samples": len(samples),
                   "loss_first": losses[0] if losses else None,
                   "loss_last": losses[-1] if losses else None,
                   "loss_min": min(losses) if losses else None}, f, indent=2)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "gate_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)
    with open(os.path.join(out_dir, "loss_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss"])
        w.writerows(enumerate(losses))
    torch.save({"state_dict": model.state_dict(), "feat_dim": PRODUCER_FEATURES},
               os.path.join(out_dir, "residual_scorer.pt"))
    wins = sum(1 for r in rows if r["beats_floor"])
    log(f"{wins}/{len(rows)} tasks beat fixed blend_0.5 -> {out_dir}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True, help="task archive jsonl")
    ap.add_argument("--task-ids", required=True, help="comma-separated gate task ids (s29,s77,s45)")
    ap.add_argument("--prod", default="stock_aware", help="producer baseline held fixed (isolates orderer)")
    ap.add_argument("--prod-ckpt", default=None,
                    help="producer checkpoint held fixed instead of --prod (e.g. EXP-0035 champion) "
                         "— the SAME producer the orderer deploys with, matching the EXP-0036 gate")
    ap.add_argument("--delta", type=float, default=1.0, help="logit-bump magnitude for labels")
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--clip", type=float, default=2.0, help="residual clip bound")
    ap.add_argument("--seed", type=int, default=0, help="torch seed for the residual-model init (reproducible fit)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from ..eval.curriculum_replay import _load_archive
    from .producer_scorer import load_producer_score_model, scorer_producing_policy

    by_id = {r.task_id: r for r in _load_archive(args.archive)}
    want = args.task_ids.split(",")
    records = [by_id[t] for t in want]
    # The EXP-0035 champion is a promoted producer SCORER checkpoint (the same one EXP-0036's
    # gate froze), not a legacy GaussianActor — load it with the scorer loader.
    prod = (scorer_producing_policy(load_producer_score_model(args.prod_ckpt))
            if args.prod_ckpt else PRODUCING_BASELINES[args.prod])
    run_value_router(
        records, prod=prod, delta=args.delta, lookahead=args.lookahead,
        eval_seeds=args.eval_seeds, horizon=args.horizon, epochs=args.epochs, clip=args.clip,
        seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
