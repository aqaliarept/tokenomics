"""Value model for the EXP-0009 value-guided planner.

The orderer search teacher (`eval.orderer_oracle`) scores every candidate routing by a
full `lookahead`-cycle deepcopy rollout. EXP-0009 cuts that cost by rolling only `short`
cycles under a candidate and estimating the remaining-window consumer units with a learned
value `V(state)`, where `state` is the existing centralized-critic system state
(`env.global_state()`, ADR-0004) — so no new state contract is added.

This module owns the regressor side: the `ValueModel` MLP, the (state, windowed-units)
sample collection over fixed (producer, orderer) rollouts, and the MSE fit. The planner
side (value-guided candidate scoring, episode driver, ranking diagnostic) lives in
`eval.orderer_oracle`, taking a torch-free `value_fn` closure from here. Real fits are
Runpod-only (CLAUDE.md); the collection/fit are small enough to test locally on depth-2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch
from torch import nn

from ..env import TokenomicsAECEnv
from ..eval.baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    Policy,
    ordering_obs_price_from_state,
)
from ..eval.orderer_oracle import (
    ValueFn,
    _agent_done,
    _load_prod_closure,
    planner_orderer_episode,
    ranking_agreement_episode,
)
from ..eval.scale import build_scale_config


@dataclass
class ValueSample:
    """One value-regression example: the system state (`env.global_state()`) at a cycle and
    the consumer units delivered over the next `window` cycles from that state (the target
    the planner's terminal value estimates)."""

    state: np.ndarray
    target: float


class ValueModel(nn.Module):
    """MLP estimating windowed consumer units from the centralized system state. Tiny by
    design (the state is already a compact system summary); deployed as a torch-free closure
    in the planner via `value_closure`."""

    def __init__(self, state_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Standardization buffers (EXP-0009 step-3 diagnostic): raw `global_state` dims span
        # 1e-2..1e3 with large offsets and ~30% dead dims, which saturate the tanh first layer
        # and dead-gradient the net to predicting the mean (pred_std=0). The net operates on
        # standardized inputs and de-normalizes its output, so the deployed `value_closure`
        # still returns unit-scale estimates. Identity by default; set by `fit_value_model`.
        self.register_buffer("in_mean", torch.zeros(state_dim))
        self.register_buffer("in_std", torch.ones(state_dim))
        self.register_buffer("out_mean", torch.zeros(()))
        self.register_buffer("out_std", torch.ones(()))

    def set_norm(self, states: torch.Tensor, targets: torch.Tensor) -> None:
        """Fit the standardization stats from the training data. Dead dims (std≈0) keep std=1
        so centering maps them to 0 (no divide-by-zero); the output is standardized too so the
        net learns a zero-mean unit-variance target."""
        self.in_mean = states.mean(0)
        std = states.std(0)
        std[std < 1e-8] = 1.0
        self.in_std = std
        self.out_mean = targets.mean()
        self.out_std = targets.std().clamp_min(1e-8)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = (state - self.in_mean) / self.in_std
        return self.net(z).squeeze(-1) * self.out_std + self.out_mean


def collect_value_samples(
    config,
    prod: Policy,
    order: Policy,
    *,
    window: int,
    seeds: "range | list[int]",
    horizon: int = 100,
    product_stream: bool = False,
    enriched_orderer_obs: bool = False,
    log=None,
) -> list[ValueSample]:
    """Roll episodes under the fixed `(prod, order)` continuation and record, at each cycle
    start, `(global_state, units delivered over the next `window` cycles)` — the value
    regression targets. Windowing is per-episode: the target at cycle `c` is the cumulative
    consumer units at the start of cycle `c + window` (or the episode's final total if the
    window runs past the end) minus the cumulative at cycle `c`. The state is `env.global_state()`
    (turn-independent system state), so one record per cycle regardless of agent count."""
    out: list[ValueSample] = []
    for si, s in enumerate(list(seeds)):
        env = TokenomicsAECEnv(config=config, horizon=horizon,
                               enriched_orderer_obs=enriched_orderer_obs,
                               product_stream_ordering=product_stream)
        env.reset(seed=s)
        for a in env.possible_agents:
            env.action_space(a).seed(s)
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        states: list[np.ndarray] = []
        cum: list[float] = []
        cur = -1
        for agent in env.agent_iter(max_iter=max_iter):
            if _agent_done(env, agent):
                env.step(None)
                continue
            if env.current_cycle != cur:
                cur = env.current_cycle
                states.append(np.asarray(env.global_state(), dtype=np.float32))
                cum.append(float(env.episode_stats["consumer_completed_units"]))
            role = env._role[agent][0]
            env.step(prod(env, agent) if role == "prod" else order(env, agent))
        final = float(env.episode_stats["consumer_completed_units"])
        for p in range(len(states)):
            nxt = cum[p + window] if p + window < len(cum) else final
            out.append(ValueSample(states[p], nxt - cum[p]))
        if log is not None:
            log(f"  value label-gen episode {si + 1} (seed {s}): {len(out)} rows so far")
    return out


def fit_value_model(
    model: ValueModel, samples: list[ValueSample], *, epochs: int, lr: float,
    log=None, log_every: int = 0,
) -> float:
    """Full-batch MSE fit of `model` to the (state, windowed-units) targets. Returns the
    final loss. `log`/`log_every` stream progress for long detached fits (silent by default)."""
    X = torch.as_tensor(np.stack([s.state for s in samples]), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray([s.target for s in samples]), dtype=torch.float32)
    model.set_norm(X, Y)  # standardize inputs+target so the tanh net does not mean-collapse
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for e in range(epochs):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        if log is not None and log_every > 0 and ((e + 1) % log_every == 0 or e + 1 == epochs):
            log(f"    value fit epoch {e + 1}/{epochs}  mse {float(loss.detach()):.4f}")
    return float(loss.detach())


def value_closure(model: ValueModel) -> ValueFn:
    """Wrap a fitted `ValueModel` as the torch-free `value_fn` the planner calls: a single
    global-state vector → scalar estimated units. NumPy-free inference is unnecessary (one
    call per scored candidate, not per agent), but `torch.no_grad` keeps it cheap."""
    def value_fn(state: np.ndarray) -> float:
        with torch.no_grad():
            return float(model(torch.as_tensor(state, dtype=torch.float32)))

    return value_fn


def run_value_planner_eval(
    *,
    depth: int,
    prod: Policy,
    order: Policy,
    candidates: list[str],
    n_samples: int,
    sample_scale: float,
    lookahead: int,
    short: int,
    train_seeds: "list[int]",
    held_out_seeds: "list[int]",
    label_seeds: int,
    horizon: int,
    epochs: int,
    lr: float,
    seed: int,
    hidden: int = 128,
    eval_seeds: int = 1,
    csv_out: "str | None" = None,
    log=print,
) -> list[dict]:
    """EXP-0009 accept harness: collect value targets on the train topologies under the fixed
    `(prod, order)` continuation (window = `lookahead - short`, the tail the planner's short
    rollout stops short of), fit the value model, then on each held-out topology score the
    value-guided planner (short window + value) against the rollout-only teacher (full
    `lookahead` rollout) — consumer units, simulated cycles, and per-cycle ranking agreement.
    The accept verdict (value-guided units match the teacher at fewer cycles) is read off these
    rows. Real fits are Runpod-only (CLAUDE.md); this stays small enough to test locally."""
    window = lookahead - short
    train_cfgs = [build_scale_config(s, depth=depth) for s in train_seeds]

    log(f"=== value-planner eval  depth={depth}  train={train_seeds}  held_out={held_out_seeds}  "
        f"lookahead={lookahead} short={short} (value window={window}) K={n_samples} ===")
    samples: list[ValueSample] = []
    for cfg in train_cfgs:
        samples += collect_value_samples(
            cfg, prod, order, window=window, seeds=range(label_seeds), horizon=horizon, log=log)
    log(f"value label-gen: {len(samples)} rows over {len(train_cfgs)} topos")

    torch.manual_seed(seed)
    state_dim = len(samples[0].state)
    model = ValueModel(state_dim, hidden=hidden)
    mse = fit_value_model(model, samples, epochs=epochs, lr=lr, log=log,
                          log_every=max(1, epochs // 8))
    log(f"value fit: mse {mse:.4f} ({len(samples)} rows)")
    value_fn = value_closure(model)

    cand = {name: (ordering_obs_price_from_state if name == "obs_price"
                   else ORDERING_BASELINES[name]) for name in candidates}
    rows: list[dict] = []
    log(f"  {'seed':>4} {'value_cu':>8} {'roll_cu':>8} {'v/r':>5} "
        f"{'value_cyc':>9} {'roll_cyc':>8} {'agree':>5}")
    for s in held_out_seeds:
        cfg = build_scale_config(s, depth=depth)
        v_units, v_cycles, r_units, r_cycles, agrees = [], [], [], [], []
        for es in range(eval_seeds):
            vs, vc = planner_orderer_episode(
                TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cand,
                n_samples, sample_scale, short, es, value_fn=value_fn)
            rs, rc = planner_orderer_episode(
                TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cand,
                n_samples, sample_scale, lookahead, es, value_fn=None)
            ag = ranking_agreement_episode(
                TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cand,
                n_samples, sample_scale, lookahead, short, value_fn, es)
            v_units.append(vs["consumer_completed_units"]); v_cycles.append(vc)
            r_units.append(rs["consumer_completed_units"]); r_cycles.append(rc)
            agrees.append(ag)
        v_cu = sum(v_units) / len(v_units)
        r_cu = sum(r_units) / len(r_units)
        v_cyc = sum(v_cycles) / len(v_cycles)
        r_cyc = sum(r_cycles) / len(r_cycles)
        ratio = v_cu / r_cu if r_cu else float("nan")
        agree = sum(agrees) / len(agrees)
        log(f"  {s:>4} {v_cu:>8.1f} {r_cu:>8.1f} {ratio:>5.2f} "
            f"{v_cyc:>9.0f} {r_cyc:>8.0f} {agree:>5.2f}")
        rows.append({
            "seed": s, "value_units": round(v_cu, 1), "rollout_units": round(r_cu, 1),
            "ratio": round(ratio, 3), "value_cycles": round(v_cyc, 1),
            "rollout_cycles": round(r_cyc, 1), "agreement": round(agree, 3),
        })

    if csv_out and rows:
        import csv

        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"{len(rows)} rows -> {csv_out}")
    wins = sum(1 for r in rows if r["ratio"] >= 1.0 and r["value_cycles"] < r["rollout_cycles"])
    log(f"\n{len(rows)} held-out: value-guided matches/beats rollout at fewer cycles on {wins}")
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--prod", default="stock_aware",
                    help="producer baseline for the fixed continuation (isolates the orderer)")
    ap.add_argument("--prod-ckpt", default=None,
                    help="producer checkpoint to hold fixed instead of --prod (e.g. EXP-0004 v4)")
    ap.add_argument("--order", default="proportional",
                    help="orderer baseline driving the value label-gen continuation")
    ap.add_argument("--candidates", default="even_split,proportional,obs_price",
                    help="named orderer baselines in the per-cycle search")
    ap.add_argument("--samples", type=int, default=12, help="sampled routing vectors per cycle")
    ap.add_argument("--sample-scale", type=float, default=1.0)
    ap.add_argument("--lookahead", type=int, default=12, help="rollout-only teacher horizon")
    ap.add_argument("--short", type=int, default=3,
                    help="value-guided rollout cycles before the learned terminal value")
    ap.add_argument("--train-seeds", default="0,1,2,3,4,5", help="topologies for value label-gen")
    ap.add_argument("--held-out-seeds", default="6,7,8,9", help="topologies for the planner compare")
    ap.add_argument("--label-seeds", type=int, default=2, help="episode seeds per train topology")
    ap.add_argument("--eval-seeds", type=int, default=5, help="episode seeds per held-out topology")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args()

    prod = _load_prod_closure(args.prod_ckpt) if args.prod_ckpt else PRODUCING_BASELINES[args.prod]
    order = (ordering_obs_price_from_state if args.order == "obs_price"
             else ORDERING_BASELINES[args.order])
    run_value_planner_eval(
        depth=args.depth, prod=prod, order=order,
        candidates=[s for s in args.candidates.split(",")],
        n_samples=args.samples, sample_scale=args.sample_scale,
        lookahead=args.lookahead, short=args.short,
        train_seeds=[int(x) for x in args.train_seeds.split(",")],
        held_out_seeds=[int(x) for x in args.held_out_seeds.split(",")],
        label_seeds=args.label_seeds, horizon=args.horizon, epochs=args.epochs,
        lr=args.lr, seed=args.seed, hidden=args.hidden, eval_seeds=args.eval_seeds,
        csv_out=args.csv_out,
    )


if __name__ == "__main__":
    main()
