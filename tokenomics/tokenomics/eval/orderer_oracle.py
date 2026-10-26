"""Orderer MPC / search oracle (EXP-0005 Stage 1, headroom ceiling probe).

EXP-0005 Stage 0 (`signal_check` headroom audit) proved the hardcoded baseline
(`stock_aware` producer + `proportional` orderer) leaves consumer orders rejected
(cons✗ 78–180 on s0/s3/s5, not a death wall) and that *random* beats it on s3 — so
there is **reachable headroom**, localized to the **orderer**. This oracle quantifies
how much of that orderer headroom is recoverable by *state-dependent* routing, the
thing a learned orderer (EXP-0005 M2) could capture but a single fixed baseline cannot.

It is a measurement, not a trainer (local per CLAUDE.md — evals are local). The domain
is deterministic and `deepcopy`-able (~3 ms), so we run **receding-horizon control**:
at each cycle, for each candidate orderer baseline, clone the env, roll forward
`lookahead` cycles under (`stock_aware`, candidate), score by consumer units gained,
and commit the argmax candidate for that cycle. The producer is held at `stock_aware`
throughout — this isolates the *orderer* contribution (the identified lever).

Output per (config_seed × episode_seed): the oracle's episode `cu✓` vs the fixed
`stock_aware+proportional` baseline, plus the per-cycle histogram of which candidate
won — evidence that no single fixed orderer is optimal (⇒ a learned orderer has room).
"""

from __future__ import annotations

import argparse
import copy
import csv
import statistics

import numpy as np

from .baselines import (
    ORDERING_BASELINES,
    PRODUCING_BASELINES,
    Policy,
    ordering_obs_price_from_state,
)
from .scale import build_scale_config
from ..env import TokenomicsAECEnv
from ..train.mlp_actor import staged_config

# Hindsight baseline (EXP-0005 Stage 0): the strict M0 bar per topology. The named
# baseline is stock_aware+proportional, but random beats it on some seeds, so a credible
# "learning is superior" claim must clear the best of the simple baselines chosen in
# hindsight (incl. random+random), not just the fixed combo. Canonical here (eval);
# `train.orderer_improve` imports it for the M2 eval (one source of truth, no cycle).
HINDSIGHT_BASELINE_COMBOS = {
    "stock_aware+proportional": ("stock_aware", "proportional"),
    "stock_aware+even_split": ("stock_aware", "even_split"),
    "stock_aware+random": ("stock_aware", "random"),
    "random+random": ("random", "random"),
}


def _agent_done(env: TokenomicsAECEnv, agent: str) -> bool:
    """Termination check for hot rollout loops.

    `AECEnv.last()` constructs the current observation. The search rollouts only need
    done flags before choosing a policy action, so using the flags directly avoids
    building and then discarding thousands of product-stream observations.
    """
    return env.terminations[agent] or env.truncations[agent]


def _hindsight_baseline_units(cfg, eval_seeds: int, horizon: int) -> tuple[float, str]:
    """Hindsight-baseline consumer units on one config (max over HINDSIGHT_BASELINE_COMBOS),
    over `eval_seeds` episode seeds — the strict bar a learned policy must beat."""
    from .runner import evaluate

    best_cu, best_name = -1.0, ""
    for name, (pp, oo) in HINDSIGHT_BASELINE_COMBOS.items():
        m = evaluate(PRODUCING_BASELINES[pp], ORDERING_BASELINES[oo],
                     seeds=range(eval_seeds), horizon=horizon, config=cfg)
        cu = m["consumer_completed_units"][0]
        if cu > best_cu:
            best_cu, best_name = cu, name
    return best_cu, best_name


def _load_prod_closure(prod_ckpt: str) -> Policy:
    """Greedy producer closure from a checkpoint (e.g. EXP-0004 v4), so the orderer
    oracle/audit runs on the SAME fixed producer the M2 student deploys with."""
    import torch

    from ..train.clone_fidelity import load_producer_actor

    actor, bound = load_producer_actor(prod_ckpt)

    def pol(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        o = torch.as_tensor(env.observe(agent), dtype=torch.float32)
        with torch.no_grad():
            mean = actor.actor(o).numpy()
        return np.clip(mean, -bound, bound).astype(np.float32)

    return pol


def _const_policy(vec: np.ndarray) -> Policy:
    """A candidate routing as a policy returning a fixed logit vector (the env
    softmaxes it per order over eligible producers). Used by the per-cycle search:
    a sampled vector is committed for one cycle, re-searched the next — directed
    temporal diversification (EXP-0005 Stage 1c: only per-cycle re-routing wins)."""
    def pol(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
        return vec
    return pol


def _rollout_units(
    env: TokenomicsAECEnv, prod: Policy, order: Policy, lookahead: int
) -> int:
    """Play a cloned environment forward for at most `lookahead` cycles and return
    the consumer units delivered over that window."""
    start_cycle = env.current_cycle
    start_units = env.episode_stats["consumer_completed_units"]
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    for agent in env.agent_iter(max_iter=max_iter):
        if _agent_done(env, agent):
            env.step(None)
            continue
        if env.current_cycle - start_cycle >= lookahead:
            break
        role = env._role[agent][0]
        env.step(prod(env, agent) if role == "prod" else order(env, agent))
    return int(env.episode_stats["consumer_completed_units"] - start_units)


def mpc_orderer_episode(
    env: TokenomicsAECEnv,
    prod: Policy,
    candidates: dict[str, Policy],
    lookahead: int,
    seed: int,
) -> tuple[dict, dict[str, int]]:
    """Drive one episode where the orderer is re-selected per cycle by `lookahead`-cycle
    MPC over `candidates`. Producer fixed (isolates the orderer). Returns the final
    `episode_stats` and the per-cycle histogram of which candidate was chosen."""
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100

    choice_counts: dict[str, int] = {name: 0 for name in candidates}
    cur = -1
    order_pol: Policy = next(iter(candidates.values()))
    for agent in env.agent_iter(max_iter=max_iter):
        if _agent_done(env, agent):
            env.step(None)
            continue
        if env.current_cycle != cur:
            cur = env.current_cycle
            best_name, best_units = None, -1
            for name, cand in candidates.items():
                units = _rollout_units(copy.deepcopy(env), prod, cand, lookahead)
                if units > best_units:
                    best_units, best_name = units, name
            order_pol = candidates[best_name]
            choice_counts[best_name] += 1
        role = env._role[agent][0]
        env.step(prod(env, agent) if role == "prod" else order_pol(env, agent))
    return env.episode_stats, choice_counts


def _search_best_pol(
    env: TokenomicsAECEnv,
    prod: Policy,
    base_candidates: dict[str, Policy],
    n_samples: int,
    sample_scale: float,
    lookahead: int,
    seed: int,
    cur: int,
    adim: int,
) -> tuple[Policy, bool]:
    """The per-cycle directed routing search primitive (EXP-0005 Stage 1d): score the
    named baselines plus `n_samples` freshly-sampled fixed routing vectors by a
    `lookahead`-cycle rollout from `env`, return (winning policy, whether a *sampled*
    routing won). Reproducible — sample RNG seeded by (seed, cycle). Shared by the
    measurement episode and the Stage 2 label generator (`train.orderer_improve`)."""
    scored = _score_candidates(
        env, prod, base_candidates, n_samples, sample_scale, lookahead, seed, cur, adim)
    best_name, best_pol, _ = max(scored, key=lambda c: c[2])  # first max wins ties
    return best_pol, best_name.startswith("s")


def _candidate_list(
    base_candidates: dict[str, Policy],
    n_samples: int,
    sample_scale: float,
    seed: int,
    cur: int,
    adim: int,
) -> list[tuple[str, Policy]]:
    """The per-cycle candidate set: the named baselines + `n_samples` freshly-sampled fixed
    routing vectors (named `s{j}`). Sample RNG seeded by (seed, cycle) so the same set is
    reproduced by `_score_candidates`."""
    rng = np.random.default_rng((seed + 1) * 100003 + cur)
    cands: list[tuple[str, Policy]] = list(base_candidates.items())
    for j in range(n_samples):
        vec = (rng.standard_normal(adim) * sample_scale).astype(np.float32)
        cands.append((f"s{j}", _const_policy(vec)))
    return cands


def _score_candidates(
    env: TokenomicsAECEnv,
    prod: Policy,
    base_candidates: dict[str, Policy],
    n_samples: int,
    sample_scale: float,
    lookahead: int,
    seed: int,
    cur: int,
    adim: int,
) -> list[tuple[str, Policy, float]]:
    """Score the named baselines + `n_samples` freshly-sampled fixed routings by a
    `lookahead`-cycle rollout from `env` (EXP-0005 Stage 1d). Returns `(name, pol, units)`
    per candidate. Sample RNG is seeded by (seed, cycle) for reproducibility."""
    return [
        (name, pol, float(_rollout_units(copy.deepcopy(env), prod, pol, lookahead)))
        for name, pol in _candidate_list(base_candidates, n_samples, sample_scale, seed, cur, adim)
    ]


def search_orderer_episode(
    env: TokenomicsAECEnv,
    prod: Policy,
    base_candidates: dict[str, Policy],
    n_samples: int,
    sample_scale: float,
    lookahead: int,
    seed: int,
) -> tuple[dict, int]:
    """Drive one episode where, each cycle, the orderer routing is chosen by DIRECTED
    search (`_search_best_pol`): score the named baselines plus `n_samples`
    freshly-sampled fixed routing vectors by `lookahead`-cycle rollout, commit the
    argmax for that cycle. Producer fixed (isolates the orderer). Reproducible: sample
    RNG seeded by (seed, cycle). Returns final `episode_stats` and the count of cycles
    a *sampled* routing won (vs a named baseline) — the headroom a fixed baseline
    cannot reach."""
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    adim = env.spec_.ordering_action_dim

    cur = -1
    order_pol: Policy = next(iter(base_candidates.values()))
    sample_wins = 0
    for agent in env.agent_iter(max_iter=max_iter):
        if _agent_done(env, agent):
            env.step(None)
            continue
        if env.current_cycle != cur:
            cur = env.current_cycle
            order_pol, won = _search_best_pol(
                env, prod, base_candidates, n_samples, sample_scale, lookahead, seed, cur, adim)
            sample_wins += int(won)
        role = env._role[agent][0]
        env.step(prod(env, agent) if role == "prod" else order_pol(env, agent))
    return env.episode_stats, sample_wins


def _baseline_units(env: TokenomicsAECEnv, prod: Policy, order: Policy, seed: int) -> dict:
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100
    for agent in env.agent_iter(max_iter=max_iter):
        if _agent_done(env, agent):
            env.step(None)
            continue
        role = env._role[agent][0]
        env.step(prod(env, agent) if role == "prod" else order(env, agent))
    return dict(env.episode_stats)


def orderer_oracle(
    *,
    depth: int,
    rung: float | None,
    config_seeds: list[int],
    eval_seeds: int,
    lookahead: int,
    candidates: list[str],
    n_samples: int,
    sample_scale: float,
    prod_name: str,
    base_order: str,
    horizon: int,
    out: str,
    prod_ckpt: "str | None" = None,
    report_hindsight_baseline: bool = False,
) -> None:
    prod = _load_prod_closure(prod_ckpt) if prod_ckpt else PRODUCING_BASELINES[prod_name]
    prod_label = prod_ckpt if prod_ckpt else prod_name
    cand = {
        name: (ordering_obs_price_from_state if name == "obs_price" else ORDERING_BASELINES[name])
        for name in candidates
    }
    base_ord = ORDERING_BASELINES[base_order]
    mode = f"search(samples={n_samples},scale={sample_scale})" if n_samples > 0 else "switch"

    print(f"\n=== orderer oracle [{mode}]  prod={prod_label}  base_candidates={candidates}  "
          f"lookahead={lookahead}  depth={depth}  rung={rung}  seeds={config_seeds}"
          f"{'  +hindsight-baseline' if report_hindsight_baseline else ''} ===")
    hdr = f"  {'cseed':>5} {'oracle_cu':>9} {'base_cu':>8} {'o/base':>6} "
    if report_hindsight_baseline:
        hdr += f"{'hb_cu':>8} {'o/hb':>6} {'hb_name':>22} "
    print(hdr + f"{'or_dead':>7}  {'detail':>0}")
    rows = []
    for cs in config_seeds:
        cfg = build_scale_config(cs, depth=depth)
        if rung is not None:
            cfg = staged_config(cfg, rung)
        or_units, bs_units, ords, deads_o, deads_b, swins = [], [], {}, [], [], []
        for es in range(eval_seeds):
            if n_samples > 0:
                stats_o, sw = search_orderer_episode(
                    TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cand,
                    n_samples, sample_scale, lookahead, es)
                swins.append(sw)
            else:
                stats_o, choice = mpc_orderer_episode(
                    TokenomicsAECEnv(config=cfg, horizon=horizon), prod, cand, lookahead, es)
                for k, v in choice.items():
                    ords[k] = ords.get(k, 0) + v
            stats_b = _baseline_units(
                TokenomicsAECEnv(config=cfg, horizon=horizon), prod, base_ord, es)
            or_units.append(stats_o["consumer_completed_units"])
            bs_units.append(stats_b["consumer_completed_units"])
            deads_o.append(1.0 if stats_o["dead"] else 0.0)
            deads_b.append(1.0 if stats_b["dead"] else 0.0)
        o_mean = statistics.mean(or_units)
        b_mean = statistics.mean(bs_units)
        ratio = o_mean / b_mean if b_mean else float("nan")
        detail = (f"sample_wins/ep={statistics.mean(swins):.0f}" if n_samples > 0
                  else ",".join(f"{k}:{v}" for k, v in sorted(ords.items(), key=lambda kv: -kv[1])))
        row = {
            "config_seed": cs, "rung": rung if rung is not None else "native",
            "oracle_cu": round(o_mean, 2), "base_cu": round(b_mean, 2),
            "ratio": round(ratio, 4), "oracle_dead": round(statistics.mean(deads_o), 3),
            "base_dead": round(statistics.mean(deads_b), 3), "detail": detail,
        }
        line = (f"  {cs:>5} {o_mean:>9.1f} {b_mean:>8.1f} {ratio:>6.2f} ")
        if report_hindsight_baseline:
            hb_cu, hb_name = _hindsight_baseline_units(cfg, eval_seeds, horizon)
            r_hb = o_mean / hb_cu if hb_cu else float("nan")
            row.update({"hb_cu": round(hb_cu, 2), "hb_name": hb_name,
                        "ratio_hb": round(r_hb, 4)})
            line += f"{hb_cu:>8.1f} {r_hb:>6.2f} {hb_name:>22} "
        print(line + f"{statistics.mean(deads_o):>7.2f}  {detail}")
        rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    wins = sum(1 for r in rows if r["ratio"] > 1.02)
    msg = f"\n{len(rows)} rows -> {out}  ({wins} cells where oracle beats base_order by >2%)"
    if report_hindsight_baseline:
        hb_wins = sum(1 for r in rows if r.get("ratio_hb", 0) >= 1.0)
        print(msg + f"; oracle >= hindsight baseline on {hb_wins}/{len(rows)}")
    else:
        print(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--rung", type=float, default=None, help="staged_config rung; omitted = native")
    ap.add_argument("--config-seeds", default="0,3,4,5")
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--lookahead", type=int, default=8, help="receding-horizon rollout cycles")
    ap.add_argument("--candidates", default="even_split,proportional,obs_price",
                    help="named orderer baselines included in the per-cycle search")
    ap.add_argument("--samples", type=int, default=0,
                    help="if >0: DIRECTED ROUTING SEARCH — sample this many fixed routing vectors "
                         "per cycle (+ baselines), commit the best by rollout. 0 = switch among baselines")
    ap.add_argument("--sample-scale", type=float, default=1.0, help="stdev of sampled routing logits")
    ap.add_argument("--prod", default="stock_aware", help="producer baseline held fixed (isolates orderer)")
    ap.add_argument("--prod-ckpt", default=None,
                    help="producer checkpoint to hold fixed instead of --prod (e.g. EXP-0004 v4); "
                         "runs the audit on the SAME producer M2 deploys with")
    ap.add_argument("--base-order", default="proportional", help="baseline orderer (denominator)")
    ap.add_argument("--hindsight-baseline", action="store_true",
                    help="also report the hindsight baseline per seed (oracle vs the strict M0 bar)")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    orderer_oracle(
        depth=args.depth, rung=args.rung,
        config_seeds=[int(x) for x in args.config_seeds.split(",")],
        eval_seeds=args.eval_seeds, lookahead=args.lookahead,
        candidates=[s for s in args.candidates.split(",")],
        n_samples=args.samples, sample_scale=args.sample_scale,
        prod_name=args.prod, base_order=args.base_order,
        horizon=args.horizon, out=args.out,
        prod_ckpt=args.prod_ckpt, report_hindsight_baseline=args.hindsight_baseline,
    )


if __name__ == "__main__":
    main()
