"""Policy-swap boundary probe (VERIFICATION.md).

For each trained checkpoint, cross trained {prod,order} actors against the strongest
white-box baseline (`stock_aware` producing + `proportional` ordering) on the cell's
fixed eval topology. The 2x2 grid isolates which side is the blocker: if
`base_prod+train_ord` >> `train_prod+base_ord`, the producer is what RL failed to
learn. Reports consumer_completed_units (the goal) + dead rate per combo."""

from __future__ import annotations

import argparse

from .baselines import PRODUCING_BASELINES, ORDERING_BASELINES
from .runner import evaluate
from .scale import build_scale_config
from ..train.smoke_ppo import load_policies


def swap(pt: str, depth: int, seeds: int, horizon: int) -> None:
    cfg = build_scale_config(0, depth=depth)
    base_prod = PRODUCING_BASELINES["stock_aware"]
    base_ord = ORDERING_BASELINES["proportional"]
    train_prod, train_ord = load_policies(pt)
    combos = {
        "base  + base ": (base_prod, base_ord),
        "train + train": (train_prod, train_ord),
        "train + base ": (train_prod, base_ord),
        "base  + train": (base_prod, train_ord),
    }
    print(f"\n=== {pt}  depth={depth}  seeds={seeds}  horizon={horizon} ===")
    print(f"{'combo':16}  {'units✓':>8}  {'dead':>5}  {'cons✓':>7}  {'ret':>8}")
    for name, (pp, op) in combos.items():
        m = evaluate(pp, op, seeds=range(seeds), horizon=horizon, config=cfg)
        print(
            f"{name:16}  {m['consumer_completed_units'][0]:8.1f}  "
            f"{m['dead_rate'][0]:5.2f}  {m['consumer_completed'][0]:7.1f}  "
            f"{m['return'][0]:8.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pt", required=True, help="trained checkpoint path")
    ap.add_argument("--depth", type=int, required=True, help="cell topology depth")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=100)
    args = ap.parse_args()
    swap(args.pt, args.depth, args.seeds, args.horizon)


if __name__ == "__main__":
    main()
