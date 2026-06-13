"""Minimal MAPPO (centralized-critic PPO) over the tokenomics AEC env.

A *smoke* trainer (plan slice 5, Phase 1.5 P2): proof the environment is
learnable and a readable reference for real MAPPO runs, not a tuned baseline.

CTDE (ADR-0004): two **decentralized actors** on local obs —
  - producing actor : Categorical over Discrete(4)
  - ordering actor  : diagonal Gaussian over the Box logit vector
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
import csv
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from ..env import TokenomicsAECEnv
from ..eval.baselines import Policy
from ..eval.runner import evaluate


def mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


# --- decentralized actors (act on local obs) ---
class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, n_actions)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.exp())


# --- centralized critic (acts on global state, shared by all agents) ---
class CentralCritic(nn.Module):
    def __init__(self, global_dim: int) -> None:
        super().__init__()
        self.net = mlp(global_dim, 1)

    def value(self, glob: torch.Tensor) -> torch.Tensor:
        return self.net(glob).squeeze(-1)


@dataclass
class Buf:
    obs: list = field(default_factory=list)      # local obs (actor input)
    glob: list = field(default_factory=list)     # global state (critic input)
    act: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    val: list = field(default_factory=list)      # central-critic value at decision time
    rew: list = field(default_factory=list)
    term: list = field(default_factory=list)     # true terminal (death) -> no bootstrap
    boot: list = field(default_factory=list)      # bootstrap value for the last step (truncation)
    # per-agent index of that agent's last open transition (reward pending)
    pending: dict = field(default_factory=dict)


def gae(
    rew: list[float],
    val: list[float],
    term: list[float],
    boot: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE with correct truncation handling (ADR-0004): `term[t]=1` kills the
    value backup (death); the trajectory's final step bootstraps with `boot[t]`
    (0 if terminated, V(s_final) if truncated)."""
    adv = np.zeros(len(rew), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rew))):
        nonterminal = 1.0 - term[t]
        next_val = val[t + 1] if t + 1 < len(val) else boot[t]
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + np.asarray(val, dtype=np.float32)
    return adv, ret


def collect(
    env: TokenomicsAECEnv,
    prod: CategoricalActor,
    order: GaussianActor,
    critic: CentralCritic,
    steps: int,
    bound: float,
    rng: np.random.Generator,
) -> tuple[Buf, Buf]:
    """Roll out ~`steps` env.step calls, bucketed into a producing buffer and an
    ordering buffer. The centralized critic values the shared global state."""
    pbuf, obuf = Buf(), Buf()
    env.reset(seed=int(rng.integers(0, 2**31)))
    n = 0
    for agent in env.agent_iter(max_iter=steps * 4):
        obs, reward, term, trunc, info = env.last()
        role = env._role[agent][0]
        buf = pbuf if role == "prod" else obuf
        # attach reward to this agent's previous (pending) transition
        if agent in buf.pending:
            i = buf.pending.pop(agent)
            buf.rew[i] = reward
            buf.term[i] = 1.0 if term else 0.0
            # truncation is not terminal: bootstrap the last step with V(s_final)
            if trunc and not term:
                with torch.no_grad():
                    g = torch.as_tensor(env.global_state(), dtype=torch.float32)
                    buf.boot[i] = float(critic.value(g))

        if term or trunc:
            env.step(None)
            continue

        o = torch.as_tensor(obs, dtype=torch.float32)
        glob = env.global_state()
        with torch.no_grad():
            v = critic.value(torch.as_tensor(glob, dtype=torch.float32))
            if role == "prod":
                d = prod.dist(o)
                a = d.sample()
                logp = d.log_prob(a)
                env_action = int(a.item())
                stored = env_action
            else:
                d = order.dist(o)
                a = d.sample()
                logp = d.log_prob(a).sum()
                env_action = np.clip(a.numpy(), -bound, bound).astype(np.float32)
                stored = a.numpy()
        buf.obs.append(obs)
        buf.glob.append(glob)
        buf.act.append(stored)
        buf.logp.append(float(logp))
        buf.val.append(float(v))
        buf.rew.append(0.0)
        buf.term.append(0.0)
        buf.boot.append(0.0)
        buf.pending[agent] = len(buf.obs) - 1
        env.step(env_action)
        n += 1
        if n >= steps and not env.agents:
            break
    return pbuf, obuf


def greedy_eval_policies(
    prod: CategoricalActor,
    order: GaussianActor,
    bound: float,
) -> tuple[Policy, Policy]:
    """Wrap the trained actors as deterministic eval `Policy` closures (ADR-0003).

    Greedy = the *deployed* policy: argmax over the producing Discrete(4),
    distribution mean for the ordering Box. They read the obs via `env.observe`
    so the eval harness measures them exactly like a baseline."""

    def prod_policy(env: TokenomicsAECEnv, agent: str) -> int:
        o = torch.as_tensor(env.observe(agent), dtype=torch.float32)
        with torch.no_grad():
            return int(torch.argmax(prod.actor(o)).item())

    def order_policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        o = torch.as_tensor(env.observe(agent), dtype=torch.float32)
        with torch.no_grad():
            mean = order.actor(o).numpy()
        return np.clip(mean, -bound, bound).astype(np.float32)

    return prod_policy, order_policy


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
) -> float:
    """Clipped PPO policy-gradient step on a decentralized actor (ADR-0004). No
    value loss here — the centralized critic is trained separately."""
    if not buf.obs:
        return 0.0
    obs = torch.as_tensor(np.array(buf.obs), dtype=torch.float32)
    act = (
        torch.as_tensor(np.array(buf.act), dtype=torch.long)
        if is_discrete
        else torch.as_tensor(np.array(buf.act), dtype=torch.float32)
    )
    old_logp = torch.as_tensor(np.array(buf.logp), dtype=torch.float32)
    adv_t = torch.as_tensor(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    last_loss = 0.0
    for _ in range(epochs):
        d = actor.dist(obs)
        logp = d.log_prob(act) if is_discrete else d.log_prob(act).sum(-1)
        ratio = (logp - old_logp).exp()
        s1 = ratio * adv_t
        s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
        pg = -torch.min(s1, s2).mean()
        ent = d.entropy().mean()
        loss = pg - 0.01 * ent
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()
    return last_loss


def critic_update(
    critic: CentralCritic,
    glob: list,
    ret: np.ndarray,
    opt: torch.optim.Optimizer,
    epochs: int = 4,
) -> float:
    """Fit the centralized critic to returns pooled from both role buffers
    (ADR-0004) — one global value function for the shared cooperative return."""
    if not glob:
        return 0.0
    g = torch.as_tensor(np.array(glob), dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)
    last_loss = 0.0
    for _ in range(epochs):
        v = critic.value(g)
        loss = ((ret_t - v) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()
    return last_loss


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
    eval_every: int = 1,
    eval_seeds: int = 3,
    out: str | None = None,
    log: Callable[[str], None] = print,
) -> list[IterStats]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = TokenomicsAECEnv(horizon=horizon)
    prod_obs = env.observation_space(env._prod_agents[0]).shape[0]
    order_obs = env.observation_space(env._order_agents[0]).shape[0]
    bound = float(env.action_space(env._order_agents[0]).high[0])

    prod = CategoricalActor(prod_obs, 4)
    order = GaussianActor(order_obs, env.spec_.ordering_action_dim)
    critic = CentralCritic(env.spec_.global_state_dim)
    popt = torch.optim.Adam(prod.parameters(), lr=3e-4)
    oopt = torch.optim.Adam(order.parameters(), lr=3e-4)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    history: list[IterStats] = []
    for it in range(iters):
        pbuf, obuf = collect(env, prod, order, critic, steps, bound, rng)
        mean_rew = float(np.mean(pbuf.rew)) if pbuf.rew else 0.0

        p_adv, p_ret = gae([r * REW_SCALE for r in pbuf.rew], pbuf.val, pbuf.term, pbuf.boot)
        o_adv, o_ret = gae([r * REW_SCALE for r in obuf.rew], obuf.val, obuf.term, obuf.boot)
        pl = actor_update(prod, pbuf, p_adv, popt, is_discrete=True)
        ol = actor_update(order, obuf, o_adv, oopt, is_discrete=False)
        # critic trained on the pooled global-state -> return targets from both roles
        cl = critic_update(
            critic, pbuf.glob + obuf.glob, np.concatenate([p_ret, o_ret]), copt
        )
        rec = IterStats(iter=it, mean_step_reward=mean_rew, ploss=pl, oloss=ol, closs=cl)

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

        history.append(rec)
        log(
            f"iter {it:3d}  step_rew {mean_rew:8.3f}  ploss {pl:7.3f}  oloss {ol:7.3f}  closs {cl:7.3f}"
            f"  | eval cyc {rec.eval_cycles:6.1f}  dead {rec.eval_dead_rate:4.2f}"
            f"  cons✓ {rec.eval_consumer_completed:7.1f}  cons✗ {rec.eval_consumer_rejected:6.1f}"
            f"  capmin {rec.eval_cap_min:6.1f}  ret {rec.eval_return:8.1f}"
        )

    if out:
        write_curve(history, out)
        log(f"curve -> {out}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1, help="eval the greedy policy every N iters (0=off)")
    ap.add_argument("--eval-seeds", type=int, default=3, help="seeds per greedy eval")
    ap.add_argument("--out", type=str, default=None, help="write per-iter curve CSV to this path")
    args = ap.parse_args()
    train(
        iters=args.iters, steps=args.steps, horizon=args.horizon, seed=args.seed,
        eval_every=args.eval_every, eval_seeds=args.eval_seeds, out=args.out,
    )


if __name__ == "__main__":
    main()
