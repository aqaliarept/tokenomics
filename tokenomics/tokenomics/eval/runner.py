"""Run episodes under given policies and aggregate metrics over seeds."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Iterable, Mapping

from ..domain.config import Configuration
from ..env import TokenomicsAECEnv
from .baselines import Policy

# metric name -> (mean, std) over seeds
Aggregated = dict[str, tuple[float, float]]


@dataclass(frozen=True)
class SeedSummary:
    """Robustness of one metric across independent MODEL seeds (NEXT STEPS stage 2).

    The ladder demands *median and worst-case*, not one lucky checkpoint, so the
    headline is `worst` (the value a deployed model is at least as good as across
    seeds) and `median` (the typical seed), not the mean a single outlier can lift."""

    median: float
    worst: float
    best: float
    mean: float
    n: int


def summarize_seeds(
    per_seed: list[dict[str, float]],
    directions: Mapping[str, str],
) -> dict[str, SeedSummary]:
    """Summarize per-seed metrics into median + worst/best across seeds (stage 2).

    `directions[m]` is "higher" (good = larger, e.g. return, fulfilled units) or
    "lower" (good = smaller, e.g. dead_rate); the *worst* case is the bad-direction
    extreme — min for "higher", max for "lower". Pure function of the per-seed dicts
    so the robustness report is testable without training. Only metrics listed in
    `directions` are summarized."""
    if not per_seed:
        raise ValueError("summarize_seeds needs at least one seed result")
    out: dict[str, SeedSummary] = {}
    for m, direction in directions.items():
        vals = [r[m] for r in per_seed]
        lo, hi = min(vals), max(vals)
        worst, best = (lo, hi) if direction == "higher" else (hi, lo)
        out[m] = SeedSummary(
            median=median(vals), worst=worst, best=best, mean=mean(vals), n=len(vals)
        )
    return out


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
    enriched_orderer_obs: bool = False,
) -> Aggregated:
    runs = [run_episode(
                TokenomicsAECEnv(config=config, horizon=horizon,
                                 enriched_orderer_obs=enriched_orderer_obs),
                prod_policy, order_policy, s)
            for s in seeds]

    def agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in runs]
        return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

    return {
        "cycles": agg("cycles"),
        "dead_rate": (mean(1.0 if r["dead"] else 0.0 for r in runs), 0.0),
        "consumer_completed": agg("consumer_completed"),
        "consumer_completed_units": agg("consumer_completed_units"),
        "consumer_rejected": agg("consumer_rejected"),
        "consumer_rejected_units": agg("consumer_rejected_units"),
        "investment_completed": agg("investment_completed"),
        "return": agg("return"),
        "cap_final": agg("cap_final"),
        "cap_min": agg("cap_min"),
    }
