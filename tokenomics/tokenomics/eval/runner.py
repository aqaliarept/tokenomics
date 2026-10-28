"""Run episodes under given policies and aggregate metrics over seeds."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping, Protocol

from ..domain.config import Configuration
from ..env import TokenomicsAECEnv
from .baselines import Policy

# metric name -> (mean, std) over seeds
Aggregated = dict[str, tuple[float, float]]


class EpisodeMonitor(Protocol):
    """Observer of the step loop (`eval/contract.py`, add-shocks-i-drift D10): sees what
    `env.last()` returned for each agent turn, and each policy action before the env decodes
    it. It may raise to stop the episode; it never changes what the loop does."""

    def on_last(self, agent: str, observation: Any, terminated: bool, truncated: bool) -> None: ...

    def on_action(self, agent: str, action: Any) -> None: ...


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


def run_episode(
    env: TokenomicsAECEnv,
    prod_policy: Policy,
    order_policy: Policy,
    seed: int,
    action_seed: int | None = None,
    *,
    monitor: EpisodeMonitor | None = None,
    max_iter: int | None = None,
) -> dict:
    """Drive one episode. `seed` is the event seed (demand RNG at reset); `action_seed`
    seeds the action spaces `random` baselines sample from. None keeps the legacy coupling
    (action spaces seeded with `seed`, pinned by test_legacy_invariance); the gate passes
    `eval.seeding.replicate_seed(...)` so replicates vary actions over identical demand
    (design D5, add-measurement-gate-harness). `monitor` observes the loop (the gate's contract
    checks); `max_iter` overrides the step budget (tests only)."""
    env.reset(seed=seed)
    # seed action spaces so `random` baselines are reproducible
    for a in env.possible_agents:
        env.action_space(a).seed(seed if action_seed is None else action_seed)
    if max_iter is None:
        max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    for agent in env.agent_iter(max_iter=max_iter):
        obs, _, term, trunc, _ = env.last()
        if monitor is not None:
            monitor.on_last(agent, obs, term, trunc)
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        action = prod_policy(env, agent) if role == "prod" else order_policy(env, agent)
        if monitor is not None:
            monitor.on_action(agent, action)
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
    product_stream: bool = False,
    entity_producer_obs: bool = False,
) -> Aggregated:
    runs = [run_episode(
                TokenomicsAECEnv(config=config, horizon=horizon,
                                 enriched_orderer_obs=enriched_orderer_obs,
                                 product_stream_ordering=product_stream,
                                 entity_producer_obs=entity_producer_obs),
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
