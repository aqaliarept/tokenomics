"""Producer-only PPO improvement (EXP-0003).

The standalone-producer question: can the Phase-2 producer clone be pushed *beyond*
`stock_aware` by RL while the orderer stays fixed to the white-box `proportional`
baseline? `smoke_ppo --init-from` updates BOTH actors jointly, which confounds that
question — a producer "improvement" could really be the learned orderer co-adapting.
This driver isolates the producer:

  * the orderer NEVER trains and NEVER acts as a net — every ordering decision in the
    rollout AND in eval is `ORDERING_BASELINES["proportional"]`;
  * only the producer actor (+ the central critic) is updated;
  * a conservative BC anchor pulls the producer toward its clone (`--init-from`) so a
    fine-tune cannot drift off the cloned operating point (the EXP-0002 failure mode);
  * checkpoint selection ranks death first, then the consumer-units ratio vs
    `stock_aware + proportional` on the same staged rungs (`producer_rungs`).

Rollout trains across the Phase-2 config set (`scale_topology_set`: base seeds + slot
permutations) and the quantity rungs (`staged_config`), so the producer is improved on
a distribution, not one operating point. Reuses the `smoke_ppo` PPO machinery; the only
new mechanic is the baseline-orderer rollout (`collect_producer_only`)."""

from __future__ import annotations

import argparse
import copy
import csv
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..domain.config import Configuration
from ..env import TokenomicsAECEnv
from ..env.decode import decode_producing
from ..eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from ..eval.runner import evaluate
from ..eval.scale import build_scale_config, scale_topology_set
from .smoke_ppo import (
    CHECKPOINT_VERSION,
    REW_SCALE,
    Buf,
    CentralCritic,
    GaussianActor,
    RolloutStats,
    actor_update,
    capacity_health,
    critic_input,
    critic_input_dim,
    critic_update,
    drop_pending,
    explained_variance,
    gae,
    greedy_eval_policies,
    investment_request_cost,
    producing_action_mask,
    save_policies,
    staged_config,
)


def collect_producer_only(
    envs: list[TokenomicsAECEnv],
    prod: GaussianActor,
    critic: CentralCritic | None,
    order_policy: Policy,
    steps: int,
    bound: float,
    rng: np.random.Generator,
    death_penalty: float = 0.0,
    investment_cost: float = 0.0,
) -> tuple[Buf, RolloutStats]:
    """Roll `steps` producer decisions across `envs`; the orderer acts via the fixed
    `order_policy` baseline and is NOT buffered (it never trains). Mirrors
    `smoke_ppo.collect`'s producer branch exactly — same pending-reward bookkeeping
    (ADR-0017 successor links, truncation bootstrap), only ordering transitions are
    consumed by the baseline instead of a learned actor. Reward attribution stays
    correct: `env.last()` for a producer returns the cooperative reward since THAT
    producer last acted, which already spans the intervening ordering/production steps."""
    pbuf = Buf()
    for env in envs:
        if env.system is None:
            env.reset(seed=int(rng.integers(0, 2**31)))
    n = 0
    episodes = deaths = 0
    completed = rejected = completed_units = rejected_units = 0
    max_rounds = steps * 8 + len(envs) * 16
    rounds = 0
    while n < steps and rounds < max_rounds:
        rounds += 1
        for ei, env in enumerate(envs):
            if not env.agents:
                episodes += 1
                deaths += 1 if env.episode_stats.get("dead") else 0
                completed += env.episode_stats.get("consumer_completed", 0)
                rejected += env.episode_stats.get("consumer_rejected", 0)
                completed_units += env.episode_stats.get("consumer_completed_units", 0)
                rejected_units += env.episode_stats.get("consumer_rejected_units", 0)
                env.reset(seed=int(rng.integers(0, 2**31)))
            agent = env.agent_selection
            obs, reward, term, trunc, info = env.last()
            role = env._role[agent][0]
            if role != "prod":
                # ordering agent: act via the fixed baseline, never buffer (ADR-0049
                # standalone-producer condition). Honor terminal/truncation flags.
                if term or trunc:
                    env.step(None)
                    continue
                env.step(order_policy(env, agent))
                continue

            key = (ei, agent)
            prev = pbuf.pending.pop(key, None)
            if prev is not None:
                pbuf.rew[prev] += reward
                if death_penalty and term:
                    pbuf.rew[prev] -= death_penalty
                pbuf.term[prev] = 1.0 if term else 0.0
                if not term and critic is not None:
                    with torch.no_grad():
                        g = torch.as_tensor(critic_input(env, agent, obs), dtype=torch.float32)
                        pbuf.boot[prev] = float(critic.value(g))

            if term or trunc:
                env.step(None)
                continue

            o = torch.as_tensor(obs, dtype=torch.float32)
            glob = critic_input(env, agent, obs)
            pview = env.system.producing_agents[env._role[agent][1]].view()
            mask_np = producing_action_mask(env.spec_, pview)
            mask_t = torch.as_tensor(mask_np)
            with torch.no_grad():
                v = critic.value(torch.as_tensor(glob, dtype=torch.float32)) if critic is not None else 0.0
                d = prod.dist(o)
                a = d.sample()
                logp = (d.log_prob(a) * mask_t).sum()
                env_action = np.clip(a.numpy(), -bound, bound).astype(np.float32)
                stored = a.numpy()
            pbuf.obs.append(obs)
            pbuf.glob.append(glob)
            pbuf.act.append(stored)
            pbuf.mask.append(mask_np)
            pbuf.logp.append(float(logp))
            pbuf.val.append(float(v))
            action_cost = 0.0
            if investment_cost:
                did = env._role[agent][1]
                cmd = decode_producing(env.spec_, env_action, env.system.producing_agents[did].view())
                action_cost = investment_request_cost(cmd, investment_cost)
            pbuf.rew.append(-action_cost)
            pbuf.term.append(0.0)
            pbuf.boot.append(0.0)
            pbuf.next_idx.append(-1)
            cur = len(pbuf.obs) - 1
            if prev is not None:
                pbuf.next_idx[prev] = cur
            pbuf.pending[key] = cur
            env.step(env_action)
            n += 1
    drop_pending(pbuf)
    return pbuf, RolloutStats(episodes, deaths, completed, rejected, completed_units, rejected_units)


@dataclass
class RungEval:
    """Clone-vs-base rung table for selection/reporting (EXP-0003). `mean_dead` and
    `mean_ratio` drive the survival-first, ratio-second checkpoint rank; `min_ratio`
    surfaces the worst cell so a strong mean can't hide a collapsed rung."""

    mean_dead: float
    mean_ratio: float
    min_ratio: float
    cells: list[dict] = field(default_factory=list)


def _base_table(
    base_prod: Policy, order: Policy, cfgs: list[Configuration], rungs: list[float],
    eval_seeds: int, horizon: int,
) -> dict[tuple[int, float], tuple[float, float]]:
    """Precompute `stock_aware + proportional` (units, dead) per (cfg index, rung) once
    — the per-cell normalizer is fixed across the run, so it need not be re-evaluated."""
    table: dict[tuple[int, float], tuple[float, float]] = {}
    for ci, cfg in enumerate(cfgs):
        for p in rungs:
            m = evaluate(base_prod, order, seeds=range(eval_seeds), horizon=horizon,
                         config=staged_config(cfg, p))
            table[(ci, p)] = (m["consumer_completed_units"][0], m["dead_rate"][0])
    return table


def eval_rungs(
    prod_policy: Policy, order: Policy, cfgs: list[Configuration], rungs: list[float],
    base_table: dict[tuple[int, float], tuple[float, float]], eval_seeds: int, horizon: int,
) -> RungEval:
    """Score `train_prod + proportional` over the (cfg, rung) grid, normalized per cell
    by the precomputed `base_table`. Ratio = clone_units / base_units (the EXP-0002 gate
    normalization), so cells with different demand scales are comparable."""
    cells: list[dict] = []
    deads: list[float] = []
    ratios: list[float] = []
    for ci, cfg in enumerate(cfgs):
        for p in rungs:
            base_units, _bdead = base_table[(ci, p)]
            m = evaluate(prod_policy, order, seeds=range(eval_seeds), horizon=horizon,
                         config=staged_config(cfg, p))
            cu = m["consumer_completed_units"][0]
            ratio = cu / base_units if base_units > 0 else float("nan")
            dead = m["dead_rate"][0]
            cells.append({"cfg": ci, "rung": p, "clone_units": cu, "base_units": base_units,
                          "ratio": ratio, "clone_dead": dead})
            deads.append(dead)
            if not math.isnan(ratio):
                ratios.append(ratio)
    return RungEval(
        mean_dead=float(np.mean(deads)) if deads else float("nan"),
        mean_ratio=float(np.mean(ratios)) if ratios else float("nan"),
        min_ratio=float(np.min(ratios)) if ratios else float("nan"),
        cells=cells,
    )


@dataclass
class ImproveStats:
    """One per-iteration record of the producer-only run."""

    iter: int
    step_reward: float
    ploss: float
    closs: float
    prod_kl: float
    rollout_dead: float
    rollout_fill: float
    critic_ev: float
    sel_dead: float = float("nan")
    sel_ratio: float = float("nan")
    sel_min_ratio: float = float("nan")


def _build_actors(init_from: str) -> tuple[GaussianActor, GaussianActor, GaussianActor, dict]:
    """Load the producer to fine-tune, a FROZEN reference copy (BC anchor target), and
    the init orderer (saved unchanged alongside the trained producer so the checkpoint
    stays a valid two-actor `load_policies` artifact). Returns `(prod, ref, order, ckpt)`."""
    ckpt = torch.load(init_from, weights_only=True)
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"init_from checkpoint version {ckpt.get('version')!r} != {CHECKPOINT_VERSION}; "
            "behavior-clone with the current schema"
        )
    prod = GaussianActor(ckpt["prod_obs"], ckpt["prod_act"])
    prod.load_state_dict(ckpt["prod"])
    ref = copy.deepcopy(prod)
    for p in ref.parameters():
        p.requires_grad_(False)
    order = GaussianActor(ckpt["order_obs"], ckpt["order_act"])
    order.load_state_dict(ckpt["order"])
    return prod, ref, order, ckpt


def run_producer_improve(
    *,
    init_from: str,
    depth: int,
    rungs: list[float],
    train_seeds: int = 3,
    perms: int = 2,
    perm_seed: int = 1,
    iters: int = 200,
    steps: int = 4096,
    num_envs: int = 4,
    horizon: int = 100,
    actor_lr: float = 5e-5,
    target_kl: float = 0.003,
    bc_anchor: float = 0.05,
    ent_coef: float = 0.0,
    death_penalty: float = 0.0,
    investment_cost: float = 0.0,
    dwell: int = 10,
    explore_log_std: float | None = None,
    eval_every: int = 10,
    eval_seeds: int = 5,
    held_out_seeds: int = 3,
    held_out_perms: int = 3,
    seed: int = 0,
    save: str | None = None,
    out: str | None = None,
    log: Callable[[str], None] = print,
) -> list[ImproveStats]:
    """Producer-only PPO with a frozen `proportional` orderer + BC anchor to the clone.

    Rollout cycles (cfg, rung) every `dwell` iters off the Phase-2 train set; selection
    ranks the producer on the base train topologies' rungs (death, then -ratio); the
    final report is on disjoint held-out seeds. Eval/selection ALWAYS pair the producer
    with `proportional` — the standalone-producer condition."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order: Policy = ORDERING_BASELINES["proportional"]
    base_prod: Policy = PRODUCING_BASELINES["stock_aware"]

    prod, ref, order_net, ckpt = _build_actors(init_from)
    if explore_log_std is not None:
        # widen exploration off the (tight) clone log_std for fine-tuning; the clone is
        # saved near-deterministic, so without this PPO barely samples off the mean.
        with torch.no_grad():
            prod.log_std.fill_(float(explore_log_std))
    bound = float(ckpt["bound"])
    popt = torch.optim.Adam(prod.parameters(), lr=actor_lr)

    # config sets: rollout uses base seeds + slot permutations; selection uses the base
    # (un-permuted) train topologies; held-out uses disjoint seeds (+ their permutations).
    train_cfgs = scale_topology_set(range(train_seeds), depth=depth, perms=perms, perm_seed=perm_seed)
    select_cfgs = [build_scale_config(s, depth=depth) for s in range(train_seeds)]
    held_cfgs = scale_topology_set(
        range(train_seeds, train_seeds + held_out_seeds), depth=depth,
        perms=held_out_perms, perm_seed=perm_seed + 1000,
    )

    probe_env = TokenomicsAECEnv(config=select_cfgs[0], horizon=horizon)
    critic = CentralCritic(critic_input_dim(probe_env))
    copt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    log(f"producer_improve: init={init_from} depth={depth} rungs={rungs} "
        f"train_cfgs={len(train_cfgs)} select_cfgs={len(select_cfgs)} held_cfgs={len(held_cfgs)}")
    log("precomputing base (stock_aware+proportional) rung table ...")
    sel_base = _base_table(base_prod, order, select_cfgs, rungs, eval_seeds, horizon)

    history: list[ImproveStats] = []
    best_key: tuple[float, float] | None = None
    best_state: dict | None = None
    rollout_envs: list[TokenomicsAECEnv] | None = None
    rollout_cfg: Configuration | None = None
    dwell_left = 0

    for it in range(iters):
        if dwell_left <= 0:
            cfg_base = train_cfgs[int(rng.integers(0, len(train_cfgs)))]
            rung = float(rungs[int(rng.integers(0, len(rungs)))])
            cur_cfg = staged_config(cfg_base, rung)
            dwell_left = dwell
        dwell_left -= 1
        if rollout_envs is None or cur_cfg != rollout_cfg:
            rollout_envs = [TokenomicsAECEnv(config=cur_cfg, horizon=horizon) for _ in range(num_envs)]
            rollout_cfg = cur_cfg

        pbuf, roll = collect_producer_only(
            rollout_envs, prod, critic, order, steps, bound, rng,
            death_penalty=death_penalty, investment_cost=investment_cost,
        )
        mean_rew = float(np.mean(pbuf.rew)) if pbuf.rew else 0.0
        p_adv, p_ret = gae([r * REW_SCALE for r in pbuf.rew], pbuf.val, pbuf.term, pbuf.boot, pbuf.next_idx)

        # BC anchor target: frozen reference means on this rollout's producer obs.
        anchor_mean = None
        if bc_anchor and pbuf.obs:
            with torch.no_grad():
                anchor_mean = ref.actor(torch.as_tensor(np.array(pbuf.obs), dtype=torch.float32))
        pstats = actor_update(
            prod, pbuf, p_adv, popt, is_discrete=False, ent_coef=ent_coef,
            target_kl=target_kl, anchor_mean=anchor_mean, anchor_coef=bc_anchor,
        )
        cl = critic_update(critic, pbuf.glob, p_ret, copt)
        critic_ev = float("nan")
        if pbuf.glob:
            with torch.no_grad():
                preds = critic.value(torch.as_tensor(np.array(pbuf.glob), dtype=torch.float32)).cpu().numpy()
            critic_ev = explained_variance(p_ret, preds)

        rec = ImproveStats(
            iter=it, step_reward=mean_rew, ploss=pstats.loss, closs=cl,
            prod_kl=pstats.approx_kl, rollout_dead=roll.death_rate,
            rollout_fill=roll.fulfillment, critic_ev=critic_ev,
        )

        if eval_every and (it % eval_every == 0 or it == iters - 1):
            pp = greedy_eval_policies(prod, order_net, bound)[0]
            ev = eval_rungs(pp, order, select_cfgs, rungs, sel_base, eval_seeds, horizon)
            rec.sel_dead, rec.sel_ratio, rec.sel_min_ratio = ev.mean_dead, ev.mean_ratio, ev.min_ratio
            key = (ev.mean_dead, -ev.mean_ratio)
            if best_key is None or key < best_key:
                best_key = key
                best_state = copy.deepcopy(prod.state_dict())

        history.append(rec)
        log(
            f"iter {it:3d}  rung {rung:4.2f}  step_rew {mean_rew:8.3f}  ploss {pstats.loss:7.3f}  "
            f"closs {cl:6.3f}  pkl {pstats.approx_kl:6.4f}  cev {critic_ev:5.2f}  "
            f"roll_dead {roll.death_rate:4.2f}  rfill {roll.fulfillment:4.2f}"
            + (f"  | SEL dead {rec.sel_dead:4.2f}  ratio {rec.sel_ratio:5.3f}  min {rec.sel_min_ratio:5.3f}"
               if not math.isnan(rec.sel_ratio) else "")
        )

    if best_state is not None:
        prod.load_state_dict(best_state)
        log(f"best select: dead {best_key[0]:.3f}  ratio {-best_key[1]:.3f}")

    # final held-out report (disjoint seeds) — unbiased estimate of the kept checkpoint.
    pp = greedy_eval_policies(prod, order_net, bound)[0]
    held_base = _base_table(base_prod, order, held_cfgs, rungs, min(eval_seeds, 5), horizon)
    held = eval_rungs(pp, order, held_cfgs, rungs, held_base, min(eval_seeds, 5), horizon)
    log(f"held-out ({len(held_cfgs)} cfgs x {len(rungs)} rungs): "
        f"dead {held.mean_dead:.3f}  ratio {held.mean_ratio:.3f}  min {held.min_ratio:.3f}")

    if out:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "step_reward", "ploss", "closs", "prod_kl", "rollout_dead",
                        "rollout_fill", "critic_ev", "sel_dead", "sel_ratio", "sel_min_ratio"])
            for r in history:
                w.writerow([r.iter, round(r.step_reward, 4), round(r.ploss, 4), round(r.closs, 4),
                            round(r.prod_kl, 5), round(r.rollout_dead, 3), round(r.rollout_fill, 3),
                            round(r.critic_ev, 3), round(r.sel_dead, 3), round(r.sel_ratio, 4),
                            round(r.sel_min_ratio, 4)])
        log(f"curve -> {out}")

    if save:
        save_policies(prod, order_net, bound, save, config=select_cfgs[0])
        log(f"policy -> {save}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-from", required=True, help="producer clone checkpoint to fine-tune")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--rungs", default="0.00,0.10,0.24,0.25")
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--perms", type=int, default=2)
    ap.add_argument("--perm-seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--actor-lr", type=float, default=5e-5)
    ap.add_argument("--target-kl", type=float, default=0.003)
    ap.add_argument("--bc-anchor", type=float, default=0.05)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--death-penalty", type=float, default=0.0)
    ap.add_argument("--investment-cost", type=float, default=0.0)
    ap.add_argument("--dwell", type=int, default=10)
    ap.add_argument("--explore-log-std", type=float, default=None,
                    help="override producer log_std at init for exploration (default: keep clone's)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--held-out-seeds", type=int, default=3)
    ap.add_argument("--held-out-perms", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1, help="BLAS thread cap (parallel fan-out default)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    run_producer_improve(
        init_from=args.init_from, depth=args.depth,
        rungs=[float(x) for x in args.rungs.split(",")],
        train_seeds=args.train_seeds, perms=args.perms, perm_seed=args.perm_seed,
        iters=args.iters, steps=args.steps, num_envs=args.num_envs, horizon=args.horizon,
        actor_lr=args.actor_lr, target_kl=args.target_kl, bc_anchor=args.bc_anchor,
        ent_coef=args.ent_coef, death_penalty=args.death_penalty,
        investment_cost=args.investment_cost, dwell=args.dwell,
        explore_log_std=args.explore_log_std, eval_every=args.eval_every,
        eval_seeds=args.eval_seeds, held_out_seeds=args.held_out_seeds,
        held_out_perms=args.held_out_perms, seed=args.seed, save=args.save, out=args.out,
    )


if __name__ == "__main__":
    main()
