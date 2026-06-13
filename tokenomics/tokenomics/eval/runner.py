"""Run episodes under given policies and aggregate metrics over seeds."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable

from ..domain.config import Configuration
from ..env import TokenomicsAECEnv
from .baselines import Policy

# metric name -> (mean, std) over seeds
Aggregated = dict[str, tuple[float, float]]


def run_episode(env: TokenomicsAECEnv, prod_policy: Policy, order_policy: Policy, seed: int) -> dict:
    env.reset(seed=seed)
    # seed action spaces so `random` baselines are reproducible
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    for agent in env.agent_iter(max_iter=max_iter):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        action = prod_policy(env, agent) if role == "prod" else order_policy(env, agent)
        env.step(action)

    s = dict(env.episode_stats)
    trace = env.capacity_trace
    s["cap_final"] = trace[-1] if trace else 0.0
    s["cap_min"] = min(trace) if trace else 0.0
    return s


def evaluate(
    prod_policy: Policy,
    order_policy: Policy,
    seeds: Iterable[int] = range(10),
    horizon: int = 100,
    config: Configuration | None = None,
) -> Aggregated:
    runs = [run_episode(TokenomicsAECEnv(config=config, horizon=horizon), prod_policy, order_policy, s)
            for s in seeds]

    def agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in runs]
        return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

    return {
        "cycles": agg("cycles"),
        "dead_rate": (mean(1.0 if r["dead"] else 0.0 for r in runs), 0.0),
        "consumer_completed": agg("consumer_completed"),
        "consumer_rejected": agg("consumer_rejected"),
        "investment_completed": agg("investment_completed"),
        "return": agg("return"),
        "cap_final": agg("cap_final"),
        "cap_min": agg("cap_min"),
    }
