"""Minimal MAPPO (centralized-critic PPO) over the tokenomics AEC env.

A *smoke* trainer (plan slice 5, Phase 1.5 P2): proof the environment is
learnable and a readable reference for real MAPPO runs, not a tuned baseline.

CTDE (ADR-0004): two **decentralized actors** on local obs — both now diagonal
Gaussian over a Box (ADR-0026):
  - producing actor : Box(G+3) — G replenish quantities + 3 investment toggles
  - ordering actor  : Box(K*P) logit vector
and **one centralized critic** on the global state, shared by both. The reward
is cooperative and shared, so a single global value function is the correct
baseline for every agent.

Rewards are collected the AEC way: env.last() returns the reward earned since the
agent last acted, attached to that agent's previous transition. GAE bootstraps on
truncation (horizon) but not on termination (system death) — ADR-0004.

Run:  python -m tokenomics.train.smoke_ppo --iters 50
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..domain import load_config
from ..domain.config import Configuration
from ..domain.ordering_agent import OrderingAgentView
from ..domain.producing_agent import ProducingAgentCommand, ProducingAgentView
from ..env import TokenomicsAECEnv
from ..env.aec_env import DEFAULT_CONFIG
from ..env.decode import decode_producing
from ..eval.baselines import Policy
from ..eval.runner import evaluate


# ADR-0020 reinterpreted for the continuous producing actor (ADR-0026):
# exploration is the Gaussian log_std, not a softmax temperature. Track A's 0.25
# temperature has no mapping; this is a fresh hyperparameter, symmetric with the
# ordering actor by default.
DEFAULT_PROD_LOG_STD = -2.0
# P1a (review 2026-06-15): replenishment quantity dims need wide exploration and a
# positive mean init — at prod_log_std=-2 (σ=0.135) and a ~0 mean they round to
# zero ~99.99% of samples, so the consumer-reward path is never explored. σ=exp(0)=1
# and a +3 raw bias center replenishment where decode fires it (raw>~0.5).
DEFAULT_PROD_REPLEN_LOG_STD = 0.0
# 2.0 clears the B7 bar (seed 1: return 1330 > stock_aware 1267, 100/100 consumer,
# 0% death); 3.0 over-stocks (holding penalty drags return to 1248, just under).
DEFAULT_PROD_REPLEN_BIAS = 2.0
DEFAULT_ORDER_LOG_STD = -2.0
DEFAULT_TARGET_KL = 0.008
# Checkpoint schema version (ADR-0026). Bumped for the continuous-producing /
# producer-simplex-ordering redesign; the loader rejects older/missing versions.
# v3 (B9): ordering obs gained a per-slot product one-hot -> ordering_obs_dim grew,
# so a v2 ordering actor's input layer no longer matches.
CHECKPOINT_VERSION = 3


def mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


# --- decentralized actors (act on local obs) ---
# Both policies are diagonal-Gaussian over a Box (ADR-0026): the producing actor
# emits raw reals and decode owns all gating, so no action mask survives here.
class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        log_std_init: float | np.ndarray = 0.0,
        mean_bias: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        # P1a (review 2026-06-15): the replenishment dims round to zero unless the
        # raw action clears ~0.5, so a near-zero mean + std exp(-2)=0.135 explores
        # the multi-hop replenishment path ~0.01% of samples — no consumer-reward
        # gradient. Allow a per-dim log_std (wider exploration on quantity dims)
        # and a positive output-bias init (mean starts where replenishment fires),
        # so the path is sampled and greedy eval emits non-zero replenishment from
        # iter 0. Both are pure init — load_state_dict overwrites them, so trained
        # checkpoints are unaffected.
        if mean_bias is not None:
            with torch.no_grad():
                self.actor[-1].bias.add_(torch.as_tensor(mean_bias, dtype=torch.float32))
        init = torch.as_tensor(log_std_init, dtype=torch.float32)
        if init.ndim == 0:
            init = init.expand(act_dim).clone()
        self.log_std = nn.Parameter(init)

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.exp())


class ValueNorm(nn.Module):
    """EMA mean/variance for PopArt value targets (ADR-0019)."""

    def __init__(self, beta: float = 0.99) -> None:
        super().__init__()
        self.beta = beta
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))

    def std(self) -> torch.Tensor:
        return self.var.sqrt().clamp_min(1e-4)

    def update(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update moments and return old/new `(mean, std)` for PopArt rescaling.

        The variance includes the between-mean term. Averaging only batch
        variances underestimates dispersion whenever return means move.
        """
        old_mean = self.mean.clone()
        old_std = self.std().clone()
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        weight = 1.0 - self.beta
        new_mean = self.beta * old_mean + weight * batch_mean
        mean_delta = batch_mean - old_mean
        new_var = (
            self.beta * self.var
            + weight * batch_var
            + self.beta * weight * mean_delta.square()
        ).clamp_min(1e-8)
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        return old_mean, old_std, self.mean.clone(), self.std().clone()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std() + self.mean


# --- centralized critic (acts on global state, shared by all agents) ---
class CentralCritic(nn.Module):
    """Central value function with output-preserving PopArt normalization."""

    def __init__(self, global_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = mlp(global_dim, 1, hidden)
        self.vnorm = ValueNorm()

    def raw(self, glob: torch.Tensor) -> torch.Tensor:
        return self.net(glob).squeeze(-1)

    def value(self, glob: torch.Tensor) -> torch.Tensor:
        return self.vnorm.denormalize(self.raw(glob))

    def update_norm(self, targets: torch.Tensor) -> None:
        """Update target units without changing any denormalized prediction.

        If `V = sigma * (w h + b) + mu`, changing `(mu, sigma)` requires the
        inverse affine transform on the final layer. The old implementation
        changed stats alone, so every stored value drifted before the next
        optimizer step even at fixed network weights.
        """
        with torch.no_grad():
            old_mean, old_std, new_mean, new_std = self.vnorm.update(targets)
            output = self.net[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("critic MLP must end with nn.Linear for PopArt")
            output.weight.mul_(old_std / new_std)
            output.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)


@dataclass
class Buf:
    obs: list = field(default_factory=list)      # local obs (actor input)
    glob: list = field(default_factory=list)     # agent-conditioned central critic input
    act: list = field(default_factory=list)
    # P1b: per-dim active mask — log-prob/entropy/KL cover only dims decode exercises
    mask: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    val: list = field(default_factory=list)      # central-critic value at decision time
    rew: list = field(default_factory=list)
    term: list = field(default_factory=list)     # true terminal (death) -> no bootstrap
    boot: list = field(default_factory=list)      # bootstrap value for the last step (truncation)
    # ADR-0017: explicit successor link. next_idx[i] = buffer index of the next
    # transition for the SAME (env, agent), or -1 if the trajectory ends / leaves
    # the batch. GAE follows these links, never buffer adjacency — adjacent rows
    # interleave different envs/agents/episodes.
    next_idx: list = field(default_factory=list)
    # per-agent index of that agent's last open transition (reward pending)
    pending: dict = field(default_factory=dict)


def gae(
    rew: list[float],
    val: list[float],
    term: list[float],
    boot: list[float],
    next_idx: list[int],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE along the explicit trajectory links (ADR-0017), NOT buffer adjacency.
    The shared role buffer interleaves many (env, agent) trajectories and resets,
    so `t+1` is almost never the true successor of `t`; following `next_idx[t]`
    is. `next_idx[t] = -1` ends the trajectory: bootstrap with `boot[t]` (0 if
    terminated, V(s_final) if truncated/cut — ADR-0004). `term[t]=1` kills the
    value backup (death). Every link points forward (`next_idx[t] > t`), so a
    single reverse pass fills successors before predecessors."""
    adv = np.zeros(len(rew), dtype=np.float32)
    for t in reversed(range(len(rew))):
        nonterminal = 1.0 - term[t]
        nxt = next_idx[t]
        next_val = val[nxt] if nxt >= 0 else boot[t]
        next_adv = adv[nxt] if nxt >= 0 else 0.0
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        adv[t] = delta + gamma * lam * nonterminal * next_adv
    ret = adv + np.asarray(val, dtype=np.float32)
    return adv, ret


def capacity_health(env: TokenomicsAECEnv) -> float:
    """Bottleneck product makeability ∈ [0,1] — the potential for the capacity-floor
    shaping term (ADR-0010). 0 at system death (ADR-0030).

    P2 (review 2026-06-15): `is_dead()` is *per declared product* — a product dies
    when no declarer still holds every required capacity type. A mean over total
    capacity stays high near that boundary (one dead producer of four → 0.75), so
    `low_cap_thresh=0.3` never activates near the actual death cliff. Instead track
    the bottleneck: a product's health is its best declarer's weakest required-type
    ratio (cap/max); system health is the worst product. When the about-to-die
    product's sole declarer loses a required type, this → 0, mirroring `is_dead`."""
    system = env.system
    if not system.producing_agents:
        return 0.0
    worst = 1.0
    for product, ps in system.process_sheets.items():
        best = 0.0  # best declarer's makeability of this product
        for p in system.producing_agents.values():
            if product not in p.produces:
                continue
            ratios = [
                (p.capacities.get(t, 0) / p.max_capacities[t])
                if p.max_capacities.get(t, 0) else 0.0
                for t in ps.require
            ]
            best = max(best, min(ratios) if ratios else 0.0)
        worst = min(worst, best)
    return worst


def producing_action_mask(spec, view: ProducingAgentView) -> np.ndarray:
    """Active producing-action dims for one transition (P1b, review 2026-06-15).

    `decode_producing` no-ops every dim that cannot change this producer's command:
    inputs it does not assemble, and a toggle for an unavailable / already-running
    investment. Summing log-prob / entropy / KL over those dims injects gradient
    noise and spends the target-KL budget without altering behavior. The mask keeps
    the likelihood over only the dims the action actually exercises this step."""
    g = spec.num_inputs
    mask = np.zeros(g + 3, dtype=np.float32)
    # P2 (review 2026-06-15): replenishment dims are behavioral only when there is
    # shared inventory headroom and a positive replenish ceiling — decode emits no
    # replenishment otherwise (joint scaling to headroom 0 / max_replenish 0 yields
    # all-zero quantities). Masking them at zero headroom keeps the likelihood off
    # dims that cannot change the command this step.
    if view.headroom > 0 and spec.max_replenish[view.id] > 0:
        for prod in spec.producer_inputs[view.id]:
            mask[spec.input_index[prod]] = 1.0
    mask[g] = 0.0 if view.restoration_running else 1.0
    mask[g + 1] = 0.0 if view.upgrade_running else 1.0
    mask[g + 2] = (
        1.0 if (view.has_inventory_expansion and not view.inventory_expansion_running) else 0.0
    )
    return mask


def ordering_action_mask(spec, view: OrderingAgentView) -> np.ndarray:
    """Active ordering-action dims for one transition (P1, review 2026-06-15).

    The ordering Box is K*P raw logits (slot k × producer). `decode_ordering`
    softmaxes slot k over only the *eligible* producers of the k-th held order, so
    two dim classes are behavioral no-ops: (a) every dim of a slot whose order has
    ≤1 eligible producer — softmax over one is always 1, there is no allocation
    choice; (b) the non-eligible producer dims of a multi-eligible slot. In the
    canonical Phase 1 config every product has a single declarer (ADR-0028), so the
    whole ordering actor is inert — without the mask it still spends gradient and
    the target-KL budget with zero effect on the env. Slots past `spec.k` (decode
    falls back to an even split, no action effect) stay masked too."""
    mask = np.zeros(spec.k * spec.num_producers, dtype=np.float32)
    for k, oid in enumerate(sorted(view.incoming)):
        if k >= spec.k:
            break
        eligible = sorted(view.eligible[oid].keys())
        if len(eligible) <= 1:
            continue
        for pid in eligible:
            mask[k * spec.num_producers + spec.producer_ids.index(pid)] = 1.0
    return mask


def investment_request_cost(cmd: ProducingAgentCommand, coef: float) -> float:
    """Training-only cost per investment toggle *executed* (ADR-0022, generalized
    for the continuous action — ADR-0026).

    Charged off the gated command decode produced, not the raw action: a toggle
    whose raw real is > 0 but which decode ignores (the investment is already
    running, or unconfigured) opens no request, earns no +1, and so is not
    charged. Count the executed toggles (≤3: restoration, upgrade, inventory
    expansion). Replenishment is excluded — its counter-pressure is the inventory
    holding penalty (ADR-0025), not this cost. `coef=1` cancels the +1 reward for
    a completed investment, so investment must earn its return through later
    consumer fulfillment instead of request churn.
    """
    fired = int(cmd.do_restoration) + int(cmd.do_upgrade) + int(cmd.do_inventory_expansion)
    return coef * fired


def critic_input_dim(env: TokenomicsAECEnv) -> int:
    """Central state + padded local observation + acting-agent identity."""
    local_dim = max(env.observation_space(a).shape[0] for a in env.possible_agents)
    return env.spec_.global_state_dim + local_dim + len(env.possible_agents)


def critic_input(
    env: TokenomicsAECEnv, agent: str, obs: np.ndarray
) -> np.ndarray:
    """Agent-conditioned centralized value input (ADR-0023).

    The old critic saw producer state only. It could not distinguish acting
    agents or observe the current agent's orders, so one input corresponded to
    different return distributions. Preserve the global state and append the
    actor's padded local observation plus a one-hot agent id.
    """
    local_dim = max(env.observation_space(a).shape[0] for a in env.possible_agents)
    local = np.zeros(local_dim, dtype=np.float32)
    local[: len(obs)] = obs
    agent_id = np.zeros(len(env.possible_agents), dtype=np.float32)
    agent_id[env.possible_agents.index(agent)] = 1.0
    return np.concatenate(
        [env.global_state(), local, agent_id], dtype=np.float32
    )


def collect(
    envs: list[TokenomicsAECEnv],
    prod: GaussianActor,
    order: GaussianActor,
    critic: CentralCritic | None,
    steps: int,
    bound: float,
    rng: np.random.Generator,
    surv_coef: float = 0.0,
    death_penalty: float = 0.0,
    low_cap_coef: float = 0.0,
    low_cap_thresh: float = 0.3,
    investment_cost: float = 0.0,
) -> tuple[Buf, Buf, float]:
    """Roll `steps` total live decisions across `envs` in round-robin (ADR-0005),
    bucketed into a producing buffer and an ordering buffer. The centralized
    critic values the shared global state.

    Pending reward bookkeeping is keyed by `(env_index, agent)` so identical
    agent names across envs do not collide. A finished env is reset and reused
    until the decision budget is met.

    Also returns the **rollout death rate** at the current (training) difficulty:
    fraction of completed episodes that ended in system death. This is the
    eval-gated curriculum's advance signal (ADR-0009) — survival measured at the
    difficulty being trained, not at real difficulty (which is the eval/goal).

    `surv_coef` (ADR-0010) adds a capacity-floor shaping term to the *training*
    reward only: `surv_coef · (health − 1)` per cycle, where `health` ∈ [0,1] is
    mean capacity/max over producers. It penalizes letting capacity decay
    (max at death, where health→0) so survival pays densely — death itself
    carries no score penalty, only a lost future, which PPO under-credits. The
    env reward and eval metrics stay raw, so eval remains baseline-comparable."""
    pbuf, obuf = Buf(), Buf()
    for env in envs:
        # ADR-0018: preserve live episode state across PPO updates. Recreating or
        # resetting here limited every batch to the first ~steps/(envs*agents)
        # cycles, so the default 1024/4/12 setup never trained past cycle 21.
        if env.system is None:
            env.reset(seed=int(rng.integers(0, 2**31)))
    n = 0
    episodes = 0
    deaths = 0
    # cap outer rounds so a stuck env can never spin forever; dead-step flushing
    # and resets make real progress well within this bound.
    max_rounds = steps * 8 + len(envs) * 16
    rounds = 0
    while n < steps and rounds < max_rounds:
        rounds += 1
        for ei, env in enumerate(envs):
            if not env.agents:
                # episode finished last round — tally survival before reset wipes
                # episode_stats (ADR-0009 gate signal).
                episodes += 1
                deaths += 1 if env.episode_stats.get("dead") else 0
                env.reset(seed=int(rng.integers(0, 2**31)))
            agent = env.agent_selection
            obs, reward, term, trunc, info = env.last()
            role = env._role[agent][0]
            buf = pbuf if role == "prod" else obuf
            key = (ei, agent)
            # attach reward to this agent's previous (pending) transition, and
            # remember it so we can link this new transition as its successor.
            prev = buf.pending.pop(key, None)
            if prev is not None:
                buf.rew[prev] += (
                    reward + surv_coef * (capacity_health(env) - 1.0)
                    if surv_coef
                    else reward
                )
                # ADR-0016: one-time death penalty on the *training* reward only
                # (eval stays raw). System death carries no env score penalty, so
                # the survival basin has no restoring gradient at its edge — a
                # good policy drifts off and KL-stop can't pull it back (the
                # collapse-from-good-init mode). The penalty is the missing wall.
                if death_penalty and term:
                    buf.rew[prev] -= death_penalty
                # ADR-0016: graded low-capacity penalty — a *dense* restoring
                # gradient that ramps up as mean capacity health falls below
                # `low_cap_thresh`, quadratic to 0 at the threshold and to
                # `low_cap_coef` at health 0 (death). Unlike a terminal death
                # penalty it warns the policy *before* the cliff (better credit
                # assignment); unlike linear `surv_coef` it leaves healthy
                # operation (health ≥ thresh) unpenalized, so it doesn't distort
                # the optimum away from the danger zone.
                if low_cap_coef:
                    deficit = max(0.0, low_cap_thresh - capacity_health(env)) / low_cap_thresh
                    buf.rew[prev] -= low_cap_coef * deficit * deficit
                buf.term[prev] = 1.0 if term else 0.0
                # ADR-0017: record V(current state) as prev's bootstrap on every
                # non-terminal step. If prev gets a successor it's unused (GAE
                # follows the link); if the successor is dropped at the rollout
                # cut, or this is the truncation step (no successor appended),
                # prev bootstraps with this value — the correct r + γV(s_next).
                if not term and critic is not None:
                    with torch.no_grad():
                        g = torch.as_tensor(
                            critic_input(env, agent, obs), dtype=torch.float32
                        )
                        buf.boot[prev] = float(critic.value(g))

            if term or trunc:
                env.step(None)
                continue

            o = torch.as_tensor(obs, dtype=torch.float32)
            glob = critic_input(env, agent, obs)
            # Active-dim mask for this transition: each actor's likelihood/entropy/KL
            # covers only the dims decode actually exercises (P1b producing, P1
            # ordering). Producing gates on the producer view; ordering gates on the
            # held orders' eligible-producer counts.
            if role == "prod":
                pview = env.system.producing_agents[env._role[agent][1]].view()
                mask_np = producing_action_mask(env.spec_, pview)
            else:
                oview = env.system.ordering_agents[env._role[agent][1]].view(
                    env.system.producer_infos
                )
                mask_np = ordering_action_mask(env.spec_, oview)
            mask_t = torch.as_tensor(mask_np)
            with torch.no_grad():
                # no-critic mode (ADR-0014): v=0 → GAE collapses to a whitened
                # Monte-Carlo return-to-go (REINFORCE). actor_update normalizes
                # advantages, so the constant-0 baseline is the batch mean.
                v = critic.value(torch.as_tensor(glob, dtype=torch.float32)) if critic is not None else 0.0
                # Both actors are Gaussian over a Box now (ADR-0026): sample a
                # float vector, clip to the action bound for the env, store raw.
                # Likelihood covers only active dims (P1b).
                d = (prod if role == "prod" else order).dist(o)
                a = d.sample()
                logp = (d.log_prob(a) * mask_t).sum()
                env_action = np.clip(a.numpy(), -bound, bound).astype(np.float32)
                stored = a.numpy()
            buf.obs.append(obs)
            buf.glob.append(glob)
            buf.act.append(stored)
            buf.mask.append(mask_np)
            buf.logp.append(float(logp))
            buf.val.append(float(v))
            # Cost only the investment toggles decode will actually execute
            # (ADR-0022/0026): decode the gated command from the same clipped
            # action the env will step. Default coef 0 -> skip the extra decode.
            action_cost = 0.0
            if role == "prod" and investment_cost:
                did = env._role[agent][1]
                cmd = decode_producing(
                    env.spec_, env_action, env.system.producing_agents[did].view()
                )
                action_cost = investment_request_cost(cmd, investment_cost)
            buf.rew.append(-action_cost)
            buf.term.append(0.0)
            buf.boot.append(0.0)
            buf.next_idx.append(-1)
            cur = len(buf.obs) - 1
            # ADR-0017: link prev -> cur so GAE backs up along the real trajectory
            if prev is not None:
                buf.next_idx[prev] = cur
            buf.pending[key] = cur
            env.step(env_action)
            n += 1
    # ADR-0017: drop the still-open frontier — one transition per live (env, agent)
    # whose reward never arrived before the cut. They hold placeholder reward 0.0
    # and no successor; training on them injects fake zero-reward events. Their
    # predecessors already carry the right bootstrap (set above), so dropping them
    # leaves valid r + γV(s_next) targets.
    drop_pending(pbuf)
    drop_pending(obuf)
    # No completed episode is "no observation", not 100% death. Curriculum
    # gating must hold until a real terminal/truncation result arrives.
    death_rate = deaths / episodes if episodes else float("nan")
    return pbuf, obuf, death_rate


def drop_pending(buf: Buf) -> int:
    """Remove incomplete transitions (reward still pending at the rollout cut) and
    remap successor links (ADR-0017). Returns the count dropped. A predecessor
    pointing at a dropped row falls back to its recorded `boot` (V of the dropped
    row's state, set when the successor was appended), giving the standard
    finite-rollout target r + γV(s_next) without the unfinished action."""
    drop = set(buf.pending.values())
    buf.pending.clear()
    if not drop:
        return 0
    keep = [i for i in range(len(buf.obs)) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    old_next = buf.next_idx
    for name in ("obs", "glob", "act", "mask", "logp", "val", "rew", "term", "boot"):
        lst = getattr(buf, name)
        setattr(buf, name, [lst[i] for i in keep])
    # a link survives only if its target survived; else the predecessor ends here
    buf.next_idx = [remap[old_next[i]] if old_next[i] in remap else -1 for i in keep]
    return len(drop)


def greedy_eval_policies(
    prod: GaussianActor,
    order: GaussianActor,
    bound: float,
) -> tuple[Policy, Policy]:
    """Wrap the trained actors as deterministic eval `Policy` closures (ADR-0003).

    Greedy = the *deployed* policy: the Gaussian mean (clipped to the action
    bound) for both actors now (ADR-0026). They read the obs via `env.observe`
    so the eval harness measures them exactly like a baseline."""

    def greedy(actor: GaussianActor, env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        o = torch.as_tensor(env.observe(agent), dtype=torch.float32)
        with torch.no_grad():
            mean = actor.actor(o).numpy()
        return np.clip(mean, -bound, bound).astype(np.float32)

    def prod_policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        return greedy(prod, env, agent)

    def order_policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        return greedy(order, env, agent)

    return prod_policy, order_policy


def save_policies(
    prod: GaussianActor, order: GaussianActor, bound: float, path: str
) -> None:
    """Checkpoint the deployed actors + the dims needed to rebuild them, so the
    eval harness (signal_check) can score the trained policy against the baseline
    table without re-training. Stamped with `CHECKPOINT_VERSION` (ADR-0026)."""
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "prod_obs": prod.actor[0].in_features,
            "prod_act": prod.log_std.numel(),
            "order_obs": order.actor[0].in_features,
            "order_act": order.log_std.numel(),
            "bound": bound,
            "prod": prod.state_dict(),
            "order": order.state_dict(),
        },
        path,
    )


def load_policies(path: str) -> tuple[Policy, Policy]:
    """Rebuild the greedy eval `Policy` closures from a `save_policies`
    checkpoint — the trained policy as the eval harness sees a baseline.

    Rejects a checkpoint with a missing or older schema version loudly (ADR-0026):
    the producing action, ordering action meaning, and critic state all changed in
    the redesign, so a stale checkpoint holds semantically wrong tensors."""
    ckpt = torch.load(path, weights_only=True)
    version = ckpt.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint schema version {version!r} != {CHECKPOINT_VERSION} "
            "(ADR-0026 redesign); retrain — old checkpoints are not loadable"
        )
    prod = GaussianActor(ckpt["prod_obs"], ckpt["prod_act"])
    order = GaussianActor(ckpt["order_obs"], ckpt["order_act"])
    prod.load_state_dict(ckpt["prod"])
    order.load_state_dict(ckpt["order"])
    return greedy_eval_policies(prod, order, ckpt["bound"])


@dataclass
class IterStats:
    """One per-iteration training-curve record (ADR-0003). Optimization signals
    (rollout reward + actor/critic losses) plus greedy-eval task metrics from the
    eval harness. Eval fields are NaN on iters where eval was skipped."""

    iter: int
    mean_step_reward: float
    ploss: float
    oloss: float
    closs: float
    rollout_dead_rate: float = float("nan")
    prod_entropy: float = float("nan")
    order_std: float = float("nan")
    difficulty: float = 1.0
    eval_cycles: float = float("nan")
    eval_dead_rate: float = float("nan")
    eval_consumer_completed: float = float("nan")
    eval_consumer_rejected: float = float("nan")
    eval_cap_min: float = float("nan")
    eval_return: float = float("nan")


# Reward scale for optimization only (logging stays raw). Keeps critic targets
# and advantages at O(1) so value loss doesn't dominate — the cycle reward is a
# sum over many orders and is naturally large.
REW_SCALE = 0.05


def actor_update(
    actor: nn.Module,
    buf: Buf,
    adv: np.ndarray,
    opt: torch.optim.Optimizer,
    is_discrete: bool,
    epochs: int = 4,
    clip: float = 0.2,
    ent_coef: float = 0.01,
    target_kl: float | None = None,
) -> float:
    """Clipped PPO policy-gradient step on a decentralized actor (ADR-0004). No
    value loss here — the centralized critic is trained separately. `ent_coef`
    is decayed across the run by the caller (ADR-0008).

    ADR-0015: optional **target-KL early stop**. Clipping alone does not bound
    the aggregate step, so stop actor epochs when the rollout-policy KL exceeds
    `1.5·target_kl`. Earlier causal claims about collapse were confounded by
    ADR-0017 and ADR-0018."""
    if not buf.obs:
        return 0.0
    obs = torch.as_tensor(np.array(buf.obs), dtype=torch.float32)
    act = (
        torch.as_tensor(np.array(buf.act), dtype=torch.long)
        if is_discrete
        else torch.as_tensor(np.array(buf.act), dtype=torch.float32)
    )
    old_logp = torch.as_tensor(np.array(buf.logp), dtype=torch.float32)
    # P1b: active-dim mask — likelihood/entropy/KL skip dims decode no-ops.
    mask = torch.as_tensor(np.array(buf.mask), dtype=torch.float32)
    adv_t = torch.as_tensor(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # Review 2026-06-15 (P2, rounds 2–3): a dim masked in *every* row of the batch
    # gets zero gradient, but Adam still applies its stored momentum during
    # opt.step(), so an inactive dim drifts. Masking logp/entropy is not enough.
    #   (a) ALL dims frozen (e.g. the ordering actor when every order is singleton-
    #       eligible — canonical config): the masked loss is constant in the params,
    #       so the gradient is zero everywhere, yet opt.step() would still push the
    #       SHARED TRUNK on stored momentum and drift a behaviorless actor. Skip the
    #       step entirely — no parameter or optimizer-state change.
    #   (b) SOME dims frozen (e.g. zero-headroom replenishment while other dims act):
    #       the trunk legitimately updates from the active dims, but the per-dim
    #       output heads of the frozen dims must not move. Pin both their parameters
    #       (final-layer row+bias, log_std) AND their Adam moment slices across the
    #       step, so neither the param nor its optimizer state advances on zero grad.
    if not is_discrete and float(mask.sum()) == 0.0:
        return 0.0
    frozen = (mask.sum(0) <= 0) if not is_discrete else None
    pin = frozen is not None and bool(frozen.any())
    pin_params = (
        [actor.log_std, actor.actor[-1].weight, actor.actor[-1].bias] if pin else []
    )
    _MOMENTS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    last_loss = 0.0
    for _ in range(epochs):
        d = actor.dist(obs)
        if is_discrete:
            logp = d.log_prob(act)
            ent = d.entropy().mean()
        else:
            logp = (d.log_prob(act) * mask).sum(-1)
            # mean entropy over active dims (denom clamped for an all-inactive row),
            # so the entropy bonus scale matches the pre-mask `.mean()` when all
            # dims are active (e.g. ordering).
            ent = ((d.entropy() * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).mean()
        logratio = logp - old_logp
        ratio = logratio.exp()
        # Schulman approx-KL (non-negative, low variance). Stop before stepping
        # if this update would leave the trust region (ADR-0015).
        if target_kl is not None:
            approx_kl = ((ratio - 1.0) - logratio).mean().item()
            if approx_kl > 1.5 * target_kl:
                break
        s1 = ratio * adv_t
        s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
        pg = -torch.min(s1, s2).mean()
        loss = pg - ent_coef * ent
        opt.zero_grad()
        loss.backward()
        if pin:
            # snapshot frozen-dim param data + Adam moments (state may be absent on
            # the first-ever step; then there is no prior momentum to drift anyway).
            snaps = []
            for p in pin_params:
                st = opt.state.get(p, {})
                snaps.append((
                    p,
                    p.detach().clone(),
                    {k: st[k].detach().clone() for k in _MOMENTS if k in st},
                ))
        opt.step()
        if pin:
            with torch.no_grad():
                for p, data_snap, st_snap in snaps:
                    p.data[frozen] = data_snap[frozen]
                    st = opt.state.get(p, {})
                    for k, m_snap in st_snap.items():
                        st[k][frozen] = m_snap[frozen]
        last_loss = loss.item()
    return last_loss


def critic_update(
    critic: CentralCritic,
    glob: list,
    ret: np.ndarray,
    opt: torch.optim.Optimizer,
    epochs: int = 8,
) -> float:
    """Fit the centralized critic to returns pooled from both role buffers
    (ADR-0004) — one global value function for the shared cooperative return.

    ADR-0007: targets are normalized by a running mean/std (`ValueNorm`) and
    clipped to ±5σ so a stray large return can't spike the gradient. Reported
    loss is the *normalized* MSE — now an O(1) fit-quality signal, not the raw
    return magnitude that drowned out learning before."""
    if not glob:
        return 0.0
    g = torch.as_tensor(np.array(glob), dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)
    critic.update_norm(ret_t)
    target = critic.vnorm.normalize(ret_t).clamp(-5.0, 5.0)
    last_loss = 0.0
    for _ in range(epochs):
        v = critic.raw(g)
        loss = ((target - v) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()
    return last_loss


# --- difficulty curriculum (ADR-0006, gating ADR-0009) ---
@dataclass
class Curriculum:
    """Degradation-annealing schedule. `difficulty(it, iters)` ramps linearly
    from `d_start` to 1.0 over the first `warmup_frac` of the run, then holds at
    1.0 (full real difficulty) — the iter-based schedule.

    ADR-0009 adds a **rollout-gated** mode (`gate=True`): difficulty advances by
    `d_step` only after a completed episode at the current difficulty reaches
    `advance_below`. There is no iteration floor in gated mode; such a floor
    increases difficulty without evidence and defeats the gate."""

    d_start: float = 0.2
    warmup_frac: float = 0.6
    gate: bool = False
    advance_below: float = 0.2
    d_step: float = 0.1
    _gated_d: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._gated_d = self.d_start

    def difficulty(self, it: int, iters: int) -> float:
        if self.warmup_frac <= 0.0:
            return 1.0
        span = max(1.0, (iters - 1) * self.warmup_frac)
        return min(1.0, self.d_start + (1.0 - self.d_start) * (it / span))

    def level(self, it: int, iters: int) -> float:
        """Effective difficulty: iteration schedule or evidence-gated level."""
        if not self.gate:
            return self.difficulty(it, iters)
        return self._gated_d

    def observe(self, death_rate: float) -> None:
        """Advance the gated difficulty one `d_step` when the rollout survives
        the current difficulty (ADR-0009). No-op unless gating."""
        if self.gate and np.isfinite(death_rate) and death_rate <= self.advance_below:
            self._gated_d = min(1.0, self._gated_d + self.d_step)


def scale_degradation(cfg: Configuration, d: float) -> Configuration:
    """Copy `cfg` with every producer's degradation scaled by `d` (ADR-0006).
    `d=1.0` is the real config; smaller `d` slows decay so survival is reachable.
    No dimension changes, so the actor/critic shapes stay valid."""
    producers = [
        replace(p, degradation=max(0, round(p.degradation * d)))
        for p in cfg.producer_configs
    ]
    return replace(cfg, producer_configs=producers)


def write_curve(history: list[IterStats], path: str) -> None:
    """Dump the per-iteration training curve as CSV for offline plotting (ADR-0003)."""
    fields = list(asdict(history[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in history:
            w.writerow(asdict(s))


def train(
    iters: int = 50,
    steps: int = 1024,
    horizon: int = 100,
    seed: int = 0,
    num_envs: int = 4,
    curriculum: Curriculum | None = None,
    eval_every: int = 1,
    eval_seeds: int = 3,
    actor_lr: float = 3e-4,
    ent_coef: float = 0.01,
    surv_coef: float = 0.0,
    death_penalty: float = 0.0,
    low_cap_coef: float = 0.0,
    low_cap_thresh: float = 0.3,
    investment_cost: float = 0.0,
    prod_log_std: float = DEFAULT_PROD_LOG_STD,
    prod_replen_log_std: float = DEFAULT_PROD_REPLEN_LOG_STD,
    prod_replen_bias: float = DEFAULT_PROD_REPLEN_BIAS,
    log_std_init: float = DEFAULT_ORDER_LOG_STD,
    critic_warmup: int = 0,
    no_critic: bool = False,
    target_kl: float | None = DEFAULT_TARGET_KL,
    anneal: bool = True,
    out: str | None = None,
    save: str | None = None,
    log: Callable[[str], None] = print,
) -> list[IterStats]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Base (real) config; the curriculum (ADR-0006) eases degradation off it.
    base_cfg = load_config(DEFAULT_CONFIG)
    # Network shapes come from the real spec — degradation scaling never changes
    # obs/action dims (ADR-0006), so one build is valid for every difficulty.
    env = TokenomicsAECEnv(config=base_cfg, horizon=horizon)
    prod_obs = env.observation_space(env._prod_agents[0]).shape[0]
    order_obs = env.observation_space(env._order_agents[0]).shape[0]
    bound = float(env.action_space(env._order_agents[0]).high[0])

    # P1a: widen exploration + bias the mean on the replenishment dims (the first
    # G of the producing Box) so the multi-hop replenishment→assembly path is
    # sampled and greedy eval emits non-zero replenishment from iter 0; toggle dims
    # keep the tight `prod_log_std`. Pure init — overwritten on checkpoint load.
    g = env.spec_.num_inputs
    pdim = env.spec_.producing_action_dim
    prod_logstd = np.full(pdim, prod_log_std, dtype=np.float32)
    prod_logstd[:g] = prod_replen_log_std
    prod_bias = np.zeros(pdim, dtype=np.float32)
    prod_bias[:g] = prod_replen_bias
    prod = GaussianActor(
        prod_obs, pdim, log_std_init=prod_logstd, mean_bias=prod_bias
    )
    order = GaussianActor(order_obs, env.spec_.ordering_action_dim, log_std_init=log_std_init)
    # ADR-0014: --no-critic drops the learned baseline entirely (REINFORCE with
    # whitened MC returns), testing whether the critic↔actor coupling — not a
    # missing baseline — drove the collapse-from-good-init instability.
    critic = None if no_critic else CentralCritic(critic_input_dim(env))
    popt = torch.optim.Adam(prod.parameters(), lr=actor_lr)
    oopt = torch.optim.Adam(order.parameters(), lr=actor_lr)
    copt = None if critic is None else torch.optim.Adam(critic.parameters(), lr=1e-3)

    history: list[IterStats] = []
    # Best-on-eval checkpoint (ADR-0011): the deployed policy is the best one
    # *seen*, not the final one. Greedy survival peaks mid-run then drifts, so
    # the last iter is often worse than the best. Ranked survival-first
    # (lowest dead rate, then highest return).
    best_key: tuple[float, float] | None = None
    best_state: tuple[dict, dict] | None = None
    rollout_envs: list[TokenomicsAECEnv] | None = None
    rollout_cfg: Configuration | None = None
    for it in range(iters):
        # ADR-0008: linearly anneal actor LR to 10% and decay entropy coef to 0
        # over the run. The regression when curriculum pins at diff 1.0 was a
        # too-high LR late; the critic (ADR-0007) is steady, so only the actors
        # need to cool. `frac` = progress through the run.
        frac = it / max(1, iters - 1)
        cur_lr = actor_lr * (1.0 - 0.9 * frac) if anneal else actor_lr
        cur_ent = ent_coef * (1.0 - frac) if anneal else ent_coef
        for opt in (popt, oopt):
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
        # Preserve environments while the actual config is unchanged (ADR-0018).
        # PPO rollout cuts bootstrap values; they are not episode boundaries.
        # Curriculum config changes still require fresh envs because degradation
        # cannot be changed coherently in the middle of an episode.
        d = curriculum.level(it, iters) if curriculum else 1.0
        cfg = scale_degradation(base_cfg, d) if curriculum else base_cfg
        if rollout_envs is None or cfg != rollout_cfg:
            rollout_envs = [
                TokenomicsAECEnv(config=cfg, horizon=horizon) for _ in range(num_envs)
            ]
            rollout_cfg = cfg
        pbuf, obuf, roll_dead = collect(
            rollout_envs, prod, order, critic, steps, bound, rng,
            surv_coef, death_penalty, low_cap_coef, low_cap_thresh,
            investment_cost,
        )
        # Advance the gated curriculum only if the policy survived this difficulty
        # (ADR-0009): difficulty for the next iter never outruns the policy.
        if curriculum:
            curriculum.observe(roll_dead)
        mean_rew = float(np.mean(pbuf.rew)) if pbuf.rew else 0.0

        p_adv, p_ret = gae([r * REW_SCALE for r in pbuf.rew], pbuf.val, pbuf.term, pbuf.boot, pbuf.next_idx)
        o_adv, o_ret = gae([r * REW_SCALE for r in obuf.rew], obuf.val, obuf.term, obuf.boot, obuf.next_idx)
        # Optional historical ablation (ADR-0013): fit the critic while delaying
        # actor updates. Its original motivation predated the trajectory fix.
        warming = it < critic_warmup
        pl = 0.0 if warming else actor_update(prod, pbuf, p_adv, popt, is_discrete=False, ent_coef=cur_ent, target_kl=target_kl)
        ol = 0.0 if warming else actor_update(order, obuf, o_adv, oopt, is_discrete=False, ent_coef=cur_ent, target_kl=target_kl)
        # critic trained on the pooled global-state -> return targets from both roles
        cl = 0.0 if critic is None else critic_update(
            critic, pbuf.glob + obuf.glob, np.concatenate([p_ret, o_ret]), copt
        )
        with torch.no_grad():
            # P2 (review 2026-06-15): log entropy over the same per-transition active
            # dims `actor_update` uses. A flat `.mean()` lets the wide replenishment
            # dims inflate the diagnostic on transitions where they are inert.
            if pbuf.obs:
                pmask = torch.as_tensor(np.array(pbuf.mask), dtype=torch.float32)
                pent = prod.dist(
                    torch.as_tensor(np.array(pbuf.obs), dtype=torch.float32)
                ).entropy()
                prod_entropy = float(
                    ((pent * pmask).sum(-1) / pmask.sum(-1).clamp_min(1.0)).mean()
                )
            else:
                prod_entropy = float("nan")
        rec = IterStats(
            iter=it,
            mean_step_reward=mean_rew,
            ploss=pl,
            oloss=ol,
            closs=cl,
            rollout_dead_rate=roll_dead,
            prod_entropy=prod_entropy,
            order_std=float(order.log_std.exp().mean().detach()),
            difficulty=d,
        )

        # Greedy-eval the deployed policy through the eval harness (ADR-0003):
        # the real task metrics, comparable to the baseline table.
        if eval_every and (it % eval_every == 0 or it == iters - 1):
            pp, op = greedy_eval_policies(prod, order, bound)
            m = evaluate(pp, op, seeds=range(eval_seeds), horizon=horizon)
            rec.eval_cycles = m["cycles"][0]
            rec.eval_dead_rate = m["dead_rate"][0]
            rec.eval_consumer_completed = m["consumer_completed"][0]
            rec.eval_consumer_rejected = m["consumer_rejected"][0]
            rec.eval_cap_min = m["cap_min"][0]
            rec.eval_return = m["return"][0]
            # survival-first ranking: minimize dead rate, then maximize return
            key = (rec.eval_dead_rate, -rec.eval_return)
            if best_key is None or key < best_key:
                best_key = key
                best_state = (
                    copy.deepcopy(prod.state_dict()),
                    copy.deepcopy(order.state_dict()),
                )

        history.append(rec)
        log(
            f"iter {it:3d}  diff {d:4.2f}  step_rew {mean_rew:8.3f}  ploss {pl:7.3f}  oloss {ol:7.3f}  closs {cl:7.3f}"
            f"  roll_dead {roll_dead:4.2f}  pent {rec.prod_entropy:4.2f}  ostd {rec.order_std:4.2f}"
            f"  | eval cyc {rec.eval_cycles:6.1f}  dead {rec.eval_dead_rate:4.2f}"
            f"  cons✓ {rec.eval_consumer_completed:7.1f}  cons✗ {rec.eval_consumer_rejected:6.1f}"
            f"  capmin {rec.eval_cap_min:6.1f}  ret {rec.eval_return:8.1f}"
        )

    if out:
        write_curve(history, out)
        log(f"curve -> {out}")
    if save:
        # Restore the best-on-eval weights before checkpointing (ADR-0011).
        if best_state is not None:
            prod.load_state_dict(best_state[0])
            order.load_state_dict(best_state[1])
            log(f"best eval: dead {best_key[0]:.2f}  ret {-best_key[1]:.1f}")
        save_policies(prod, order, bound, save)
        log(f"policy -> {save}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=4, help="parallel rollout envs (ADR-0005)")
    ap.add_argument("--no-curriculum", action="store_true", help="train at full real difficulty (disable ADR-0006 curriculum)")
    ap.add_argument("--d-start", type=float, default=0.2, help="curriculum start difficulty (fraction of real degradation)")
    ap.add_argument("--warmup-frac", type=float, default=0.6, help="fraction of run to reach full difficulty (iter floor)")
    ap.add_argument("--gate", action="store_true", help="eval-gated curriculum: advance difficulty on rollout survival (ADR-0009)")
    ap.add_argument("--advance-below", type=float, default=0.2, help="rollout death-rate to advance the gated curriculum (ADR-0009)")
    ap.add_argument("--d-step", type=float, default=0.1, help="gated curriculum difficulty increment (ADR-0009)")
    ap.add_argument("--eval-every", type=int, default=1, help="eval the greedy policy every N iters (0=off)")
    ap.add_argument("--eval-seeds", type=int, default=3, help="seeds per greedy eval")
    ap.add_argument("--actor-lr", type=float, default=3e-4, help="initial actor LR (ADR-0008)")
    ap.add_argument("--ent-coef", type=float, default=0.01, help="initial entropy coef (ADR-0008)")
    ap.add_argument("--surv-coef", type=float, default=0.0, help="capacity-floor shaping coef on training reward (ADR-0010)")
    ap.add_argument("--death-penalty", type=float, default=0.0, help="one-time penalty on training reward at system death (ADR-0016)")
    ap.add_argument("--low-cap-coef", type=float, default=0.0, help="graded penalty coef as capacity health falls below --low-cap-thresh (ADR-0016)")
    ap.add_argument("--low-cap-thresh", type=float, default=0.3, help="capacity-health threshold below which the graded penalty ramps (ADR-0016)")
    ap.add_argument("--investment-cost", type=float, default=0.0, help="experimental training cost per restoration/upgrade request (ADR-0022)")
    ap.add_argument("--prod-log-std", type=float, default=DEFAULT_PROD_LOG_STD, help="initial producing-actor log std (toggle dims); lower = less exploration noise (ADR-0020/0026)")
    ap.add_argument("--prod-replen-log-std", type=float, default=DEFAULT_PROD_REPLEN_LOG_STD, help="initial log std on replenishment quantity dims; wide so the replenishment path is explored (P1a)")
    ap.add_argument("--prod-replen-bias", type=float, default=DEFAULT_PROD_REPLEN_BIAS, help="positive output-bias init on replenishment dims so greedy mean fires replenishment (P1a)")
    ap.add_argument("--log-std-init", type=float, default=DEFAULT_ORDER_LOG_STD, help="initial ordering-actor log std; lower = less exploration noise (ADR-0012/0024)")
    ap.add_argument("--critic-warmup", type=int, default=0, help="freeze actors for the first K iters so policy grads use a warm critic (ADR-0013)")
    ap.add_argument("--no-critic", action="store_true", help="drop the learned critic: REINFORCE with whitened MC returns (ADR-0014)")
    kl = ap.add_mutually_exclusive_group()
    kl.add_argument("--target-kl", type=float, dest="target_kl", help="actor target-KL (default 0.008; ADR-0015/0024)")
    kl.add_argument("--no-target-kl", action="store_const", const=None, dest="target_kl", help="disable actor target-KL early stopping")
    ap.set_defaults(target_kl=DEFAULT_TARGET_KL)
    ap.add_argument("--no-anneal", action="store_true", help="disable actor LR / entropy annealing (ADR-0008)")
    ap.add_argument("--out", type=str, default=None, help="write per-iter curve CSV to this path")
    ap.add_argument("--save", type=str, default=None, help="checkpoint the trained policy to this path (for signal_check)")
    args = ap.parse_args()
    curriculum = None if args.no_curriculum else Curriculum(
        d_start=args.d_start, warmup_frac=args.warmup_frac,
        gate=args.gate, advance_below=args.advance_below, d_step=args.d_step,
    )
    train(
        iters=args.iters, steps=args.steps, horizon=args.horizon, seed=args.seed,
        num_envs=args.num_envs, curriculum=curriculum, eval_every=args.eval_every,
        eval_seeds=args.eval_seeds, actor_lr=args.actor_lr, ent_coef=args.ent_coef,
        surv_coef=args.surv_coef, death_penalty=args.death_penalty,
        low_cap_coef=args.low_cap_coef, low_cap_thresh=args.low_cap_thresh,
        investment_cost=args.investment_cost,
        prod_log_std=args.prod_log_std,
        prod_replen_log_std=args.prod_replen_log_std,
        prod_replen_bias=args.prod_replen_bias,
        log_std_init=args.log_std_init,
        critic_warmup=args.critic_warmup, no_critic=args.no_critic,
        target_kl=args.target_kl,
        anneal=not args.no_anneal, out=args.out, save=args.save,
    )


if __name__ == "__main__":
    main()
