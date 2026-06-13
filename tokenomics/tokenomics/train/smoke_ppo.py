"""Minimal parameter-shared PPO over the tokenomics AEC env.

This is a *smoke* trainer (ADR-0002 / plan slice 5): proof the environment is
learnable and a readable reference for the real MAPPO runs, not a tuned baseline.
Two shared policies (CTDE):
  - producing policy  : Categorical over Discrete(4)
  - ordering policy   : diagonal Gaussian over the Box logit vector

Cooperative shared reward, so both policies optimize the same return. Rewards are
collected the AEC way: env.last() returns the reward earned since the agent last
acted, which we attach to that agent's previous transition.

Run:  python -m tokenomics.train.smoke_ppo --iters 50
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from ..env import TokenomicsAECEnv


def mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, n_actions)
        self.critic = mlp(obs_dim, 1)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = mlp(obs_dim, 1)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


@dataclass
class Buf:
    obs: list = field(default_factory=list)
    act: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    val: list = field(default_factory=list)
    rew: list = field(default_factory=list)
    done: list = field(default_factory=list)
    # per-agent index of that agent's last open transition (reward pending)
    pending: dict = field(default_factory=dict)


def gae(
    rew: list[float],
    val: list[float],
    done: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros(len(rew), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rew))):
        nonterminal = 1.0 - done[t]
        next_val = val[t + 1] if t + 1 < len(val) else 0.0
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + np.asarray(val, dtype=np.float32)
    return adv, ret


def collect(
    env: TokenomicsAECEnv,
    prod: CategoricalPolicy,
    order: GaussianPolicy,
    steps: int,
    bound: float,
    rng: np.random.Generator,
) -> tuple[Buf, Buf]:
    """Roll out ~`steps` env.step calls, bucketed into a producing buffer and an
    ordering buffer."""
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
            buf.done[i] = 1.0 if (term or trunc) else 0.0

        if term or trunc:
            env.step(None)
            continue

        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            if role == "prod":
                d = prod.dist(o)
                a = d.sample()
                logp = d.log_prob(a)
                v = prod.value(o)
                env_action = int(a.item())
                stored = env_action
            else:
                d = order.dist(o)
                a = d.sample()
                logp = d.log_prob(a).sum()
                v = order.value(o)
                env_action = np.clip(a.numpy(), -bound, bound).astype(np.float32)
                stored = a.numpy()
        buf.obs.append(obs)
        buf.act.append(stored)
        buf.logp.append(float(logp))
        buf.val.append(float(v))
        buf.rew.append(0.0)
        buf.done.append(0.0)
        buf.pending[agent] = len(buf.obs) - 1
        env.step(env_action)
        n += 1
        if n >= steps and not env.agents:
            break
    return pbuf, obuf


# Reward scale for optimization only (logging stays raw). Keeps critic targets
# and advantages at O(1) so value loss doesn't dominate — the cycle reward is a
# sum over many orders and is naturally large.
REW_SCALE = 0.05


def ppo_update(
    policy: nn.Module,
    buf: Buf,
    opt: torch.optim.Optimizer,
    is_discrete: bool,
    epochs: int = 4,
    clip: float = 0.2,
) -> float:
    if not buf.obs:
        return 0.0
    obs = torch.as_tensor(np.array(buf.obs), dtype=torch.float32)
    if is_discrete:
        act = torch.as_tensor(np.array(buf.act), dtype=torch.long)
    else:
        act = torch.as_tensor(np.array(buf.act), dtype=torch.float32)
    old_logp = torch.as_tensor(np.array(buf.logp), dtype=torch.float32)
    scaled_rew = [r * REW_SCALE for r in buf.rew]
    adv, ret = gae(scaled_rew, buf.val, buf.done)
    adv_t = torch.as_tensor(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.as_tensor(ret)

    last_loss = 0.0
    for _ in range(epochs):
        d = policy.dist(obs)
        if is_discrete:
            logp = d.log_prob(act)
        else:
            logp = d.log_prob(act).sum(-1)
        ratio = (logp - old_logp).exp()
        s1 = ratio * adv_t
        s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
        pg = -torch.min(s1, s2).mean()
        v = policy.value(obs)
        vloss = ((ret_t - v) ** 2).mean()
        ent = d.entropy().mean()
        loss = pg + 0.5 * vloss - 0.01 * ent
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()
    return last_loss


def train(
    iters: int = 50,
    steps: int = 1024,
    horizon: int = 100,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = TokenomicsAECEnv(horizon=horizon)
    prod_obs = env.observation_space(env._prod_agents[0]).shape[0]
    order_obs = env.observation_space(env._order_agents[0]).shape[0]
    bound = float(env.action_space(env._order_agents[0]).high[0])

    prod = CategoricalPolicy(prod_obs, 4)
    order = GaussianPolicy(order_obs, env.spec_.ordering_action_dim)
    popt = torch.optim.Adam(prod.parameters(), lr=3e-4)
    oopt = torch.optim.Adam(order.parameters(), lr=3e-4)

    history: list[float] = []
    for it in range(iters):
        pbuf, obuf = collect(env, prod, order, steps, bound, rng)
        mean_rew = float(np.mean(pbuf.rew)) if pbuf.rew else 0.0
        pl = ppo_update(prod, pbuf, popt, is_discrete=True)
        ol = ppo_update(order, obuf, oopt, is_discrete=False)
        history.append(mean_rew)
        log(f"iter {it:3d}  mean_step_reward {mean_rew:8.3f}  ploss {pl:7.3f}  oloss {ol:7.3f}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(iters=args.iters, steps=args.steps, horizon=args.horizon, seed=args.seed)


if __name__ == "__main__":
    main()
