"""Orderer expert-iteration trainer (EXP-0005 Stage 2, M2).

EXP-0005 Stage 1 isolated the improvement lever: holding the `stock_aware` producer
fixed, a *random* orderer beats `proportional` by 14–76% (1a), the gain is per-cycle
re-routing not a fixed shape (1c), and a **directed routing search** captures it
*deterministically* (+32–52% over `proportional`, 1d) — a clean, reproducible teacher
that beats the hardcoded baseline. This module turns that search oracle into a learned
orderer (AlphaGo-style expert iteration, NOT PPO; ADR-0049's decode-aware clone for the
orderer side):

  1. **Label generation** — drive episodes with the producer fixed at the EXP-0004 **v4**
     clone and, each cycle, run `orderer_oracle._search_best_pol` to pick the winning
     routing; record every orderer transition `(obs, applied per-agent action, decode
     mask)`. The label is the per-agent action of the *winning* policy (a named baseline
     for a baseline-win, the committed routing vector for a sampled win), so it is a
     proper per-agent target — not a global vector.
  2. **Fit** a fresh orderer `GaussianActor` by masked MSE over `ordering_action_mask`
     (only the eligible-producer dims of multi-eligible slots are behavioral; the rest
     are decode no-ops, ADR-0029).
  3. **DAgger** (optional rounds) — roll the *current* orderer clone forward (producer
     still v4), relabel the states it visits with the search oracle, refit from a fresh
     init on the grown dataset. Closes the train/visited-state gap the same way the
     producer warm-start does.

The saved checkpoint bundles the **v4 producer state** + the **new orderer state** so
`load_policies` / `signal_check` deploy the M2 pair directly. The open risk (RESULT.md):
the 1c gain is temporal diversification, so a *deterministic* clone captures it only if
the obs carries enough load/queue state — if fidelity drops, the fallback is a stochastic
orderer (the Gaussian `log_std`). Training is on Runpod per CLAUDE.md (real run); the
label search is local-feasible (domain is `deepcopy`-able ~3 ms).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from ..env import TokenomicsAECEnv
from ..eval.baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    Policy,
    ordering_obs_price_from_state,
)
from ..eval.orderer_oracle import ENVELOPE_COMBOS as _ENVELOPE_COMBOS
from ..eval.orderer_oracle import _agent_done
from ..eval.orderer_oracle import _search_elite_pols
from ..eval.runner import evaluate
from ..eval.scale import build_scale_config


@dataclass
class OrdererSample:
    """One orderer decision: the obs the learner sees, the search oracle's per-agent
    action label, and the decode mask (`ordering_action_mask`) marking the dims that
    actually allocate (eligible producers of multi-eligible slots)."""

    obs: np.ndarray
    label: np.ndarray
    mask: np.ndarray


def _greedy_closure(actor, bound: float) -> Policy:
    """Wrap a GaussianActor as the deployed greedy `Policy` (clip(mean), ADR-0026)."""
    import torch

    mean_fn = _numpy_mlp_mean_fn(actor)
    if mean_fn is not None:
        def pol(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
            return np.clip(mean_fn(env.observe(agent)), -bound, bound).astype(np.float32)

        return pol

    def pol(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        o = torch.as_tensor(env.observe(agent), dtype=torch.float32)
        with torch.no_grad():
            mean = actor.actor(o).numpy()
        return np.clip(mean, -bound, bound).astype(np.float32)

    return pol


def _numpy_mlp_mean_fn(actor):
    """Fast single-observation inference for this module's tiny `mlp` actors.

    Search rollouts call the fixed producer/orderer policies tens of thousands of
    times with batch size 1. NumPy avoids PyTorch's per-call module overhead there;
    training still uses the original torch modules. Unknown actor layers fall back
    to the torch path in `_greedy_closure`/`_stochastic_closure`.
    """
    import torch

    ops = []
    for layer in actor.actor:
        if isinstance(layer, torch.nn.Linear):
            ops.append((
                "linear",
                layer.weight.detach().cpu().numpy().astype(np.float32, copy=True),
                layer.bias.detach().cpu().numpy().astype(np.float32, copy=True),
            ))
        elif isinstance(layer, torch.nn.Tanh):
            ops.append(("tanh",))
        else:
            return None

    def mean(obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        for op in ops:
            if op[0] == "linear":
                x = op[1] @ x + op[2]
            else:
                x = np.tanh(x).astype(np.float32, copy=False)
        return x

    return mean


def _stochastic_closure(actor, bound: float) -> Policy:
    """Wrap a GaussianActor as a *sampling* deploy `Policy` (EXP-0006B): draw
    `a ~ Normal(mean, std)` and clip to ±bound, instead of the greedy `clip(mean)`. The
    learned per-dim `log_std` (from the stochastic NLL fit) carries the teacher's routing
    diversity, so sampling reproduces the per-cycle re-routing a deterministic mean
    collapses. The noise is drawn from the agent's **action-space RNG**, which
    `run_episode` seeds per episode (exactly like the `random` baseline) — so each
    (topology, episode) seed scores identically regardless of evaluation order, instead
    of a single closure-scoped RNG leaking state across seeds."""
    import torch

    mean_fn = _numpy_mlp_mean_fn(actor)
    std_np = actor.log_std.detach().cpu().numpy().astype(np.float32, copy=True)
    std_np = np.exp(std_np).astype(np.float32, copy=False)
    if mean_fn is not None:
        def pol(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
            mean = mean_fn(env.observe(agent))
            noise = env.action_space(agent).np_random.standard_normal(mean.shape)
            a = mean + noise.astype(np.float32) * std_np
            return np.clip(a, -bound, bound).astype(np.float32)

        return pol

    def pol(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        o = torch.as_tensor(env.observe(agent), dtype=torch.float32)
        with torch.no_grad():
            mean = actor.actor(o).numpy()
            std = actor.log_std.exp().numpy()
        noise = env.action_space(agent).np_random.standard_normal(mean.shape)
        a = mean + noise.astype(np.float32) * std
        return np.clip(a, -bound, bound).astype(np.float32)

    return pol


def collect_orderer_samples(
    config,
    prod: Policy,
    base_candidates: dict[str, Policy],
    *,
    n_samples: int,
    sample_scale: float,
    lookahead: int,
    seeds: "range | list[int]",
    horizon: int = 100,
    student_order: "Policy | None" = None,
    tag: str = "",
    enriched_orderer_obs: bool = False,
    product_stream: bool = False,
    elite_k: int = 1,
    elite_eps: float = 0.05,
    log=None,
) -> list[OrdererSample]:
    """Drive episodes (producer fixed at `prod`) and record orderer transitions labeled
    by the per-cycle directed routing search. Round 0 (`student_order is None`): the
    search policy both labels AND drives — on-oracle BC. DAgger rounds: the `student_order`
    drives (visits the clone's state distribution) while the search still provides the
    label (relabel). Reproducible — the search RNG is seeded by (seed, cycle). The search
    is silent and slow, so `log` (if given) streams per-episode progress — the label-gen
    phase otherwise produces no output for ~minutes (search is single-threaded).

    `elite_k>1` (EXP-0008 elite-set distillation): instead of one label per orderer state
    (the argmax routing), emit one sample per *elite* routing — every candidate within
    `elite_eps` of the best, capped at `elite_k`. The best elite still drives (a stable
    central state distribution); the extra near-optimal labels widen the masked Gaussian's
    fitted centroid+spread instead of collapsing a tie to one arbitrary winner. `elite_k=1`
    (default) is exactly the EXP-0006B argmax behavior (`_select_elite` top-1 = argmax)."""
    from .smoke_ppo import ordering_action_mask

    out: list[OrdererSample] = []
    seed_list = list(seeds)
    for si, s in enumerate(seed_list):
        env = TokenomicsAECEnv(config=config, horizon=horizon,
                               enriched_orderer_obs=enriched_orderer_obs,
                               product_stream_ordering=product_stream)
        env.reset(seed=s)
        for a in env.possible_agents:
            env.action_space(a).seed(s)
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
        adim = env.spec_.ordering_action_dim
        cur = -1
        first = next(iter(base_candidates.values()))
        elite_pols: list[Policy] = [first]
        for agent in env.agent_iter(max_iter=max_iter):
            if _agent_done(env, agent):
                env.step(None)
                continue
            if env.current_cycle != cur:
                cur = env.current_cycle
                elite_pols = [pol for pol, _ in _search_elite_pols(
                    env, prod, base_candidates, n_samples, sample_scale,
                    lookahead, s, cur, adim, eps=elite_eps, k=elite_k)]
            role, did = env._role[agent]
            if role == "prod":
                env.step(prod(env, agent))
                continue
            obs = np.asarray(env.observe(agent), dtype=np.float32)
            oview = env.system.ordering_agents[did].view(env.system.producer_infos)
            mask = ordering_action_mask(env.spec_, oview)
            for pol in elite_pols:  # one sample per elite routing (elite_k=1 ⇒ argmax only)
                label = np.asarray(pol(env, agent), dtype=np.float32)
                out.append(OrdererSample(obs, label, mask))
            drive = student_order if student_order is not None else elite_pols[0]
            env.step(drive(env, agent))
        if log is not None:
            log(f"  label-gen {tag} episode {si + 1}/{len(seed_list)} (seed {s}): "
                f"{len(out)} rows so far")
    return out


def _fit_orderer_clone(
    actor, samples: list[OrdererSample], bound: float, *, epochs: int, lr: float,
    stochastic: bool = False, log=None, log_every: int = 0,
) -> float:
    """Masked fit of the orderer to the search-oracle labels. Only the decode-active dims
    (`ordering_action_mask`: eligible producers of multi-eligible slots) carry gradient —
    the softmax over an order's eligible producers is the only thing that allocates
    (ADR-0029); the rest are no-ops and would just inject noise. Greedy deploys clip(mean),
    so targets are clipped to the action bound (ADR-0026).

    `stochastic=False` (EXP-0005 M2): masked MSE, mean only — the deployed std is later
    clamped near zero. `stochastic=True` (EXP-0006B): masked Gaussian **NLL**, fitting the
    mean *and* the per-dim `log_std`, so the std absorbs the per-cycle spread of the
    teacher's labels — distilling routing *diversity* rather than collapsing it into one
    mean (the EXP-0006A seed-9 residual). The std is then deployed by sampling, not
    clamped.

    `log`/`log_every`: when both are set, emit a `fit epoch e/epochs loss …` line every
    `log_every` epochs (and at the end) so a long detached fit is visibly alive instead of
    running 1500 epochs silently. Default (`log=None`) stays silent for existing callers."""
    import torch

    X = torch.as_tensor(np.stack([s.obs for s in samples]), dtype=torch.float32)
    Y = torch.clamp(
        torch.as_tensor(np.stack([s.label for s in samples]), dtype=torch.float32),
        -bound, bound)
    M = torch.as_tensor(np.stack([s.mask for s in samples]), dtype=torch.float32)
    params = list(actor.actor.parameters()) + ([actor.log_std] if stochastic else [])
    opt = torch.optim.Adam(params, lr=lr)
    loss = torch.tensor(0.0)
    for e in range(epochs):
        opt.zero_grad()
        out = actor.actor(X)
        if stochastic:
            # Gaussian NLL (drop the const 0.5*log(2π)): 0.5*((y-µ)/σ)² + log σ.
            # log_std is per-dim (act_dim,), broadcast over rows; masked dims (M=0)
            # get no gradient so their log_std stays at init.
            per = 0.5 * ((out - Y) / actor.log_std.exp()) ** 2 + actor.log_std
            loss = (per * M).sum() / M.sum().clamp_min(1.0)
        else:
            loss = (((out - Y) ** 2) * M).sum() / M.sum().clamp_min(1.0)
        loss.backward()
        opt.step()
        if log is not None and log_every > 0 and ((e + 1) % log_every == 0 or e + 1 == epochs):
            log(f"    fit epoch {e + 1}/{epochs}  loss {float(loss.detach()):.4f}")
    return float(loss.detach())


def dagger_orderer(
    configs,
    prod: Policy,
    bound: float,
    order_obs_dim: int,
    order_act_dim: int,
    *,
    candidates: dict[str, Policy],
    n_samples: int,
    sample_scale: float,
    lookahead: int,
    dagger_iters: int = 1,
    seeds: "range | list[int]" = range(2),
    horizon: int = 100,
    epochs: int = 1500,
    lr: float = 3e-3,
    seed: int = 0,
    log_std_init: float = -2.0,
    enriched_orderer_obs: bool = False,
    product_stream: bool = False,
    stochastic: bool = False,
    elite_k: int = 1,
    elite_eps: float = 0.05,
    log=print,
):
    """Expert-iterate a fresh orderer `GaussianActor` toward the search oracle, over one
    or more topologies (producer fixed at `prod`). Round 0 is on-oracle BC; each later
    round rolls the current clone forward, relabels its visited states with the search,
    and refits from a fixed init on the grown dataset (DAgger). Returns the fitted actor.

    `stochastic=True` (EXP-0006B): fit the per-dim `log_std` too (masked Gaussian NLL) so
    the deployed orderer samples — distilling the teacher's routing diversity instead of
    one mean. The DAgger relabel still rolls the *greedy* mean forward (a stable, central
    state distribution); only the saved/deployed policy samples.

    `elite_k>1` (EXP-0008): label each state with the whole within-`elite_eps` elite set
    (capped at `elite_k`), not just the argmax — the masked Gaussian then fits the elite
    centroid+spread. Pairs with `stochastic=True` (the elite spread is only realized when
    the std deploys by sampling)."""
    import torch

    from .smoke_ppo import GaussianActor

    cfg_list = list(configs) if isinstance(configs, (list, tuple)) else [configs]
    samples_by_cfg = [
        collect_orderer_samples(
            c, prod, candidates, n_samples=n_samples, sample_scale=sample_scale,
            lookahead=lookahead, seeds=seeds, horizon=horizon,
            enriched_orderer_obs=enriched_orderer_obs, product_stream=product_stream,
            elite_k=elite_k, elite_eps=elite_eps, tag=f"round0 topo{i}", log=log)
        for i, c in enumerate(cfg_list)
    ]

    order = None
    for it in range(dagger_iters):
        torch.manual_seed(seed)  # refit from a fixed init on the grown dataset
        order = GaussianActor(order_obs_dim, order_act_dim, log_std_init=log_std_init)
        flat = [s for smp in samples_by_cfg for s in smp]
        log(f"orderer-dagger round {it}: fitting {epochs} epochs on {len(flat)} rows "
            f"({len(cfg_list)} topos)...")
        loss = _fit_orderer_clone(order, flat, bound, epochs=epochs, lr=lr,
                                  stochastic=stochastic, log=log,
                                  log_every=max(1, epochs // 8))
        log(f"orderer-dagger round {it}: masked_mse {loss:.4f} ({len(flat)} rows, "
            f"{len(cfg_list)} topos)")
        if it < dagger_iters - 1:
            student = _greedy_closure(order, bound)
            for i, c in enumerate(cfg_list):
                samples_by_cfg[i] += collect_orderer_samples(
                    c, prod, candidates, n_samples=n_samples, sample_scale=sample_scale,
                    lookahead=lookahead, seeds=seeds, horizon=horizon,
                    student_order=student, enriched_orderer_obs=enriched_orderer_obs,
                    product_stream=product_stream, elite_k=elite_k, elite_eps=elite_eps,
                    tag=f"round{it+1} topo{i}", log=log)
    return order


def _load_v4_producer(path: str):
    """Load the EXP-0004 v4 producer actor + bound from its checkpoint, as the fixed
    producer for M2 (Stage 1: the producer is near-optimal, the orderer is the lever)."""
    from .clone_fidelity import load_producer_actor

    return load_producer_actor(path)


# --- evaluation against both bars (RESULT.md success criterion) ---
# best-of-baseline envelope (_ENVELOPE_COMBOS) = the strict M0 bar; canonical in
# eval.orderer_oracle (imported at top — one source of truth, no import cycle).


def _ci(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(n) if n > 1 else 0.0


def eval_m2(
    save: str,
    *,
    depth: int,
    seeds: "list[int]",
    eval_seeds: int = 5,
    horizon: int = 100,
    enriched_orderer_obs: bool = False,
    product_stream: bool = False,
    stochastic: bool = False,
    log=print,
) -> list[dict]:
    """Score the saved M2 pair vs M0 (`stock_aware+proportional`) and the best-of-baseline
    envelope on held-out topologies (mean ± 95% CI over `eval_seeds`). Primary metric is
    consumer units; rejects + investment churn are reported for the efficiency criterion.

    `stochastic=True` (EXP-0006B): deploy the orderer by *sampling* its learned `log_std`
    (`_stochastic_closure`) instead of greedy clip(mean) — the diversity is the lever, so
    it must be scored as deployed. Producer stays greedy."""
    from .smoke_ppo import load_order_actor, load_policies

    m2_prod, m2_order = load_policies(save)
    if stochastic:
        order_actor, bound = load_order_actor(save)
        m2_order = _stochastic_closure(order_actor, bound)
    rows: list[dict] = []
    log(f"\n=== M2 eval (v4 producer + learned orderer)  depth={depth}  seeds={seeds} ===")
    log(f"  {'seed':>4} {'M2_cu':>8} {'M0_cu':>8} {'env_cu':>8} {'M2/M0':>6} "
        f"{'M2/env':>6} {'M2_rej':>7} {'env_name':>22}")
    for s in seeds:
        cfg = build_scale_config(s, depth=depth)
        rng = range(eval_seeds)
        m2 = evaluate(m2_prod, m2_order, seeds=rng, horizon=horizon, config=cfg,
                      enriched_orderer_obs=enriched_orderer_obs,
                      product_stream=product_stream)
        m2_cu, m2_sd = m2["consumer_completed_units"]
        # Score the bars in the SAME agent layout as the candidate: a stream-mode M2
        # must be measured against stream-mode baselines. Deterministic baselines are
        # grouping-invariant (parity, ADR-0052), but `random` ordering consumes the
        # action-space RNG over a different set/order of agents when keyed by
        # (source, product), so a legacy-mode envelope would be a different bar.
        m0 = evaluate(PRODUCING_BASELINES["stock_aware"], ORDERING_BASELINES["proportional"],
                      seeds=rng, horizon=horizon, config=cfg, product_stream=product_stream)
        m0_cu = m0["consumer_completed_units"][0]
        env_best_cu, env_best_name = -1.0, ""
        for name, (pp, oo) in _ENVELOPE_COMBOS.items():
            m = evaluate(PRODUCING_BASELINES[pp], ORDERING_BASELINES[oo],
                         seeds=rng, horizon=horizon, config=cfg, product_stream=product_stream)
            if m["consumer_completed_units"][0] > env_best_cu:
                env_best_cu, env_best_name = m["consumer_completed_units"][0], name
        r_m0 = m2_cu / m0_cu if m0_cu else float("nan")
        r_env = m2_cu / env_best_cu if env_best_cu else float("nan")
        log(f"  {s:>4} {m2_cu:>7.0f}± {m2_cu and _ci(m2_sd, eval_seeds):<5.0f}"
            f"{m0_cu:>7.0f} {env_best_cu:>7.0f} {r_m0:>6.2f} {r_env:>6.2f} "
            f"{m2['consumer_rejected_units'][0]:>7.0f} {env_best_name:>22}")
        rows.append({
            "seed": s, "m2_cu": round(m2_cu, 1), "m2_ci": round(_ci(m2_sd, eval_seeds), 1),
            "m0_cu": round(m0_cu, 1), "env_cu": round(env_best_cu, 1),
            "env_name": env_best_name, "ratio_m0": round(r_m0, 3), "ratio_env": round(r_env, 3),
            "m2_dead": m2["dead_rate"][0], "m2_rej_units": round(m2["consumer_rejected_units"][0], 1),
            "m2_invest": round(m2["investment_completed"][0], 1),
        })
    won_m0 = sum(1 for r in rows if r["ratio_m0"] >= 1.10)
    won_env = sum(1 for r in rows if r["ratio_env"] >= 1.10)
    log(f"\n{len(rows)} seeds: M2 beats M0 by >=10% on {won_m0}; beats ENVELOPE by >=10% on {won_env}")
    return rows


def run_orderer_improve(
    *,
    depth: int,
    prod_ckpt: str,
    train_seeds: int,
    held_out_seeds: int,
    candidates: list[str],
    n_samples: int,
    sample_scale: float,
    lookahead: int,
    dagger_iters: int,
    label_seeds: int,
    horizon: int,
    epochs: int,
    lr: float,
    seed: int,
    log_std_save: float | None,
    eval_seeds: int,
    save: str,
    csv_out: "str | None" = None,
    train_seed_list: "list[int] | None" = None,
    held_out_seed_list: "list[int] | None" = None,
    enriched_orderer_obs: bool = False,
    product_stream: bool = False,
    stochastic: bool = False,
    elite_k: int = 1,
    elite_eps: float = 0.05,
    log=print,
) -> list[dict]:
    """EXP-0005 Stage 2 driver: expert-iterate the orderer toward the search oracle with
    the producer fixed at the v4 clone, save the M2 pair, score it vs M0 + envelope on
    held-out topologies. Returns the eval rows. Train/held-out topology seeds are the
    contiguous `range(train_seeds)` / `range(train_seeds, +held_out_seeds)` by default, or
    explicit `train_seed_list` / `held_out_seed_list` (M2-v2: widen coverage without
    overlapping the held-out suite — a non-contiguous split).

    `elite_k>1` (EXP-0008): distill over the per-state elite routing set (within `elite_eps`
    of best, capped at `elite_k`) instead of the single argmax — the elite-set candidate
    family. Saved-checkpoint schema is unchanged, so the champion-iteration gate is reused."""
    import torch

    prod_actor, bound = _load_v4_producer(prod_ckpt)
    prod = _greedy_closure(prod_actor, bound)

    train_ids = train_seed_list if train_seed_list is not None else list(range(train_seeds))
    held = (held_out_seed_list if held_out_seed_list is not None
            else list(range(train_seeds, train_seeds + held_out_seeds)))

    cfg0 = build_scale_config(train_ids[0], depth=depth)
    probe = TokenomicsAECEnv(config=cfg0, horizon=1,
                             enriched_orderer_obs=enriched_orderer_obs,
                             product_stream_ordering=product_stream)
    order_obs_dim = probe.observation_space(probe._order_agents[0]).shape[0]
    order_act_dim = probe.spec_.ordering_action_dim

    train_cfgs = [build_scale_config(s, depth=depth) for s in train_ids]
    cand = {
        name: (ordering_obs_price_from_state if name == "obs_price" else ORDERING_BASELINES[name])
        for name in candidates
    }
    log(f"=== orderer expert-iteration (M2)  depth={depth}  train_seeds={train_ids}  held_out={held}  "
        f"oracle: lookahead={lookahead} K={n_samples} cand={candidates}  dagger_iters={dagger_iters} ===")
    order = dagger_orderer(
        train_cfgs, prod, bound, order_obs_dim, order_act_dim, candidates=cand,
        n_samples=n_samples, sample_scale=sample_scale, lookahead=lookahead,
        dagger_iters=dagger_iters, seeds=range(label_seeds), horizon=horizon,
        epochs=epochs, lr=lr, seed=seed,
        enriched_orderer_obs=enriched_orderer_obs, product_stream=product_stream,
        stochastic=stochastic, elite_k=elite_k, elite_eps=elite_eps, log=log)

    from .smoke_ppo import save_policies

    # Producer (v4) deploys greedy → always tighten its log_std. The orderer log_std is
    # the *learned* exploration when stochastic (EXP-0006B) — keep it; only the
    # deterministic M2 clamps the orderer to the deploy floor.
    if log_std_save is not None:
        with torch.no_grad():
            prod_actor.log_std.fill_(log_std_save)
            if not stochastic:
                order.log_std.fill_(log_std_save)
    obs_version = 3 if product_stream else (2 if enriched_orderer_obs else 1)
    save_policies(prod_actor, order, bound, save, config=cfg0,
                  orderer_obs_version=obs_version)
    log(f"saved M2 (v4 producer + learned orderer) -> {save}")

    # eval_seeds<=0: the caller scores the saved checkpoint itself (the champion-iteration
    # driver re-evals candidate AND champion under identical conditions), so skip the
    # redundant internal eval rather than pay for a second candidate eval.
    if eval_seeds <= 0:
        log("eval_seeds<=0: skipping internal held-out eval (caller scores the checkpoint)")
        return []

    rows = eval_m2(save, depth=depth, seeds=held, eval_seeds=eval_seeds, horizon=horizon,
                   enriched_orderer_obs=enriched_orderer_obs, product_stream=product_stream,
                   stochastic=stochastic, log=log)
    if csv_out and rows:
        import csv

        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"{len(rows)} rows -> {csv_out}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--prod-ckpt", required=True, help="fixed producer checkpoint (EXP-0004 v4)")
    ap.add_argument("--train-seeds", type=int, default=6, help="topologies 0..N-1 for label-gen")
    ap.add_argument("--held-out-seeds", type=int, default=4, help="unseen topologies for eval (after train)")
    ap.add_argument("--train-seed-list", default=None,
                    help="explicit comma-separated train topology seeds (overrides --train-seeds; "
                         "M2-v2: widen coverage without overlapping held-out, e.g. 0,1,2,3,4,5,10,11,12,13)")
    ap.add_argument("--held-out-seed-list", default=None,
                    help="explicit comma-separated held-out topology seeds (overrides --held-out-seeds)")
    ap.add_argument("--candidates", default="even_split,proportional,obs_price",
                    help="named orderer baselines included in the per-cycle search")
    ap.add_argument("--samples", type=int, default=12, help="sampled routing vectors per cycle (Stage 1d K)")
    ap.add_argument("--sample-scale", type=float, default=1.0, help="stdev of sampled routing logits")
    ap.add_argument("--lookahead", type=int, default=12, help="search rollout horizon (cycles)")
    ap.add_argument("--dagger-iters", type=int, default=1, help="1 = on-oracle BC; >1 adds relabel rounds")
    ap.add_argument("--label-seeds", type=int, default=2, help="episode seeds per topology for label-gen")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-std-save", type=float, default=-2.5)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--save", required=True)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--enriched", action="store_true",
                    help="train/score the enriched orderer obs (orderer_obs_version 2, "
                         "EXP-0006A M2-v3-obs) instead of the legacy v1 obs")
    ap.add_argument("--stochastic", action="store_true",
                    help="EXP-0006B M2-v3-stochastic: distill a learned per-dim log_std "
                         "(masked Gaussian NLL) and deploy by sampling, preserving the "
                         "teacher's routing diversity instead of one deterministic mean")
    ap.add_argument("--elite-k", type=int, default=1,
                    help="EXP-0008 elite-set distillation: label each state with the whole "
                         "within-tolerance elite routing set (capped at K) instead of the "
                         "single argmax (1 = EXP-0006B behavior). Pair with --stochastic")
    ap.add_argument("--elite-eps", type=float, default=0.05,
                    help="elite tolerance: keep routings scoring >= (1-eps)*best (EXP-0008)")
    ap.add_argument("--product-stream", action="store_true",
                    help="EXP-0007A: train/score the product-stream orderer obs "
                         "(orderer_obs_version 3, ADR-0052) — one ordering agent per "
                         "(source, product), obs scales with producers not products")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training: load --save and score it vs M0 + envelope on the held-out "
                         "seeds (regenerate the eval CSV from an existing checkpoint)")
    args = ap.parse_args()
    train_seed_list = ([int(x) for x in args.train_seed_list.split(",")]
                       if args.train_seed_list else None)
    held_out_seed_list = ([int(x) for x in args.held_out_seed_list.split(",")]
                          if args.held_out_seed_list else None)
    if args.eval_only:
        held = (held_out_seed_list if held_out_seed_list is not None
                else list(range(args.train_seeds, args.train_seeds + args.held_out_seeds)))
        rows = eval_m2(args.save, depth=args.depth, seeds=held,
                       eval_seeds=args.eval_seeds, horizon=args.horizon,
                       enriched_orderer_obs=args.enriched,
                       product_stream=args.product_stream, stochastic=args.stochastic)
        if args.csv_out and rows:
            import csv

            with open(args.csv_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"{len(rows)} rows -> {args.csv_out}")
        return
    run_orderer_improve(
        depth=args.depth, prod_ckpt=args.prod_ckpt, train_seeds=args.train_seeds,
        held_out_seeds=args.held_out_seeds,
        candidates=[s for s in args.candidates.split(",")],
        n_samples=args.samples, sample_scale=args.sample_scale, lookahead=args.lookahead,
        dagger_iters=args.dagger_iters, label_seeds=args.label_seeds, horizon=args.horizon,
        epochs=args.epochs, lr=args.lr, seed=args.seed, log_std_save=args.log_std_save,
        eval_seeds=args.eval_seeds, save=args.save, csv_out=args.csv_out,
        train_seed_list=train_seed_list, held_out_seed_list=held_out_seed_list,
        enriched_orderer_obs=args.enriched, product_stream=args.product_stream,
        stochastic=args.stochastic, elite_k=args.elite_k, elite_eps=args.elite_eps,
    )


if __name__ == "__main__":
    main()
