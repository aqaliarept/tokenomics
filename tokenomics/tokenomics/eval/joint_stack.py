"""C1 portable-stack eval (PRODUCT_AGNOSTIC combined ladder).

Drives the full portable stack — producer scorer/planner + orderer scorer/allocator —
in one env and scores it against the deterministic `entity_planner + proportional`
bar. The producer view (`entity_producer_obs`) and the corrected v4 product-stream
ordering are orthogonal env flags; this is the first place both run together.

The two learned policies read their own white-box views off `env.system`, so the
deterministic planner (producer feasibility) and the masked allocator (orderer
conservation/eligibility) keep the joint command feasible for any scores.
"""

from __future__ import annotations

from dataclasses import replace

from ..domain.config import Configuration
from ..env import TokenomicsAECEnv
from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from .runner import run_episode
from .scale import build_scale_config


def relabel_capacity_types(config: Configuration) -> Configuration:
    """Rename every capacity type (`capacity-N` -> `relabel-capacity-N`) — a pure
    schema relabel the id-free contract must not key on. Numerically identical economy."""
    types = sorted({t for s in config.process_sheets for t in s.require})
    cap_map = {t: f"relabel-{t}" for t in types}
    sheets = [
        replace(s, require={cap_map.get(t, t): a for t, a in s.require.items()})
        for s in config.process_sheets
    ]
    producers = [
        replace(
            p,
            capacities={cap_map.get(t, t): a for t, a in p.capacities.items()},
            restoration=replace(p.restoration, type=cap_map.get(p.restoration.type, p.restoration.type)),
            upgrade=replace(p.upgrade, type=cap_map.get(p.upgrade.type, p.upgrade.type)),
        )
        for p in config.producer_configs
    ]
    out = replace(config, process_sheets=sheets, producer_configs=producers)
    out.validate()
    return out


def joint_stack_env(config: Configuration, horizon: int) -> TokenomicsAECEnv:
    """The env both portable modules need: entity producer obs + corrected v4 streams."""
    return TokenomicsAECEnv(
        config=config,
        horizon=horizon,
        entity_producer_obs=True,
        product_stream_ordering=True,
        corrected_stream_k=True,
    )


def run_joint_stack(producer_model, orderer_model, config: Configuration, seed: int, horizon: int) -> dict:
    """Run one episode of the full portable stack on `config`."""
    from ..train.orderer_scorer import scorer_ordering_policy
    from ..train.producer_scorer import scorer_producing_policy

    env = joint_stack_env(config, horizon)
    prod_pol = scorer_producing_policy(producer_model)
    order_pol = scorer_ordering_policy(orderer_model)
    return run_episode(env, prod_pol, order_pol, seed=seed)


def eval_joint_stack(producer_model, orderer_model, depth, seeds, eval_seeds, horizon,
                     log=lambda *a, **k: None) -> list[dict]:
    """Per held-out seed, the portable stack's mean consumer units vs the deterministic
    `entity_planner + proportional` bar on the same joint env. Reports value ratio and
    both death rates (the C1 value gate; feasibility/relabel are the contract tests)."""
    from ..train.orderer_scorer import scorer_ordering_policy
    from ..train.producer_scorer import scorer_producing_policy

    prod_pol = scorer_producing_policy(producer_model)
    order_pol = scorer_ordering_policy(orderer_model)
    bar_prod = PRODUCING_BASELINES["entity_planner"]
    bar_order = ORDERING_BASELINES["proportional"]

    rows: list[dict] = []
    for seed in seeds:
        cfg = build_scale_config(seed, depth=depth)
        joint_u = joint_d = bar_u = bar_d = 0.0
        for es in range(eval_seeds):
            js = run_episode(joint_stack_env(cfg, horizon), prod_pol, order_pol, seed=es)
            bs = run_episode(joint_stack_env(cfg, horizon), bar_prod, bar_order, seed=es)
            joint_u += js["consumer_completed_units"]; joint_d += float(js["dead"])
            bar_u += bs["consumer_completed_units"]; bar_d += float(bs["dead"])
        joint_u /= eval_seeds; bar_u /= eval_seeds
        joint_d /= eval_seeds; bar_d /= eval_seeds
        P = joint_stack_env(cfg, 1).spec_.num_producers
        joint_units, bar_units = round(joint_u, 2), round(bar_u, 2)
        ratio = joint_units / bar_units if bar_units else 0.0
        rows.append({"seed": seed, "producers": P, "joint_units": joint_units,
                     "bar_units": bar_units, "ratio": ratio,
                     "dead": joint_d, "bar_dead": bar_d})
        log(f"seed {seed} (P={P}): joint {joint_u:.1f} vs bar {bar_u:.1f} -> {ratio:.3f}")
    return rows


def relabel_gate(producer_model, orderer_model, depth, seeds, horizon) -> list[dict]:
    """Per seed, joint episode units on the base vs capacity-relabeled config. The
    id-free contract requires a byte-identical episode (no relabel mismatch)."""
    rows: list[dict] = []
    for seed in seeds:
        cfg = build_scale_config(seed, depth=depth)
        base = run_joint_stack(producer_model, orderer_model, cfg, seed=0, horizon=horizon)
        rel = run_joint_stack(producer_model, orderer_model, relabel_capacity_types(cfg),
                              seed=0, horizon=horizon)
        match = (base["consumer_completed_units"] == rel["consumer_completed_units"]
                 and base["cycles"] == rel["cycles"] and base["dead"] == rel["dead"])
        rows.append({"seed": seed, "base_units": base["consumer_completed_units"],
                     "relabel_units": rel["consumer_completed_units"], "match": match})
    return rows


def main() -> None:
    """Local C1 eval (EXP-0014): load the distilled EXP-0012 orderer + EXP-0013 producer
    checkpoints, run the joint portable stack on train-size held-out seeds, print value
    ratios vs `entity_planner + proportional` and the relabel-invariance gate."""
    import argparse

    from ..train.orderer_scorer import load_scorer
    from ..train.producer_scorer import load_producer_score_model

    ap = argparse.ArgumentParser(description="C1 portable-stack smoke (joint producer+orderer scorers)")
    ap.add_argument("--producer-ckpt", required=True)
    ap.add_argument("--orderer-ckpt", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seeds", default="0-5", help="held-out scale seeds, e.g. 0-5 or 6,7")
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=100)
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    prod = load_producer_score_model(args.producer_ckpt)
    order = load_scorer(args.orderer_ckpt)

    rows = eval_joint_stack(prod, order, args.depth, seeds, args.eval_seeds, args.horizon, log=print)
    print("\nseed,producers,joint_units,bar_units,ratio,dead,bar_dead")
    ratios = []
    for r in rows:
        ratios.append(r["ratio"])
        print(f"{r['seed']},{r['producers']},{r['joint_units']},{r['bar_units']},"
              f"{r['ratio']:.4f},{r['dead']},{r['bar_dead']}")
    print(f"\nmean ratio vs entity_planner+proportional: {sum(ratios)/len(ratios):.4f}; "
          f"min {min(ratios):.4f}; max joint dead-rate {max(r['dead'] for r in rows):.3f}")

    gate = relabel_gate(prod, order, args.depth, seeds, args.horizon)
    bad = [g for g in gate if not g["match"]]
    print(f"\nrelabel gate: {len(gate) - len(bad)}/{len(gate)} seeds byte-identical under "
          f"capacity relabel" + (f" — MISMATCH {[g['seed'] for g in bad]}" if bad else ""))


def _parse_seeds(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


if __name__ == "__main__":
    main()
