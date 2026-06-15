"""Signal check: do different policies produce materially different outcomes?

If a do-nothing policy scores about the same as a sensible one, the env poses no
learning problem and the dynamics need tuning. Run:

    python -m tokenomics.eval.signal_check
    python -m tokenomics.eval.signal_check --horizon 100 --seeds 20
"""

from __future__ import annotations

import argparse

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .runner import evaluate

# (producing baseline, ordering baseline) pairs worth comparing
COMBOS = [
    ("noop", "even_split"),
    ("always_restore", "even_split"),
    ("restore_when_low", "even_split"),
    ("stock_aware", "even_split"),
    ("stock_aware", "proportional"),
    ("always_both", "even_split"),
    ("random", "random"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--policy",
        type=str,
        default=None,
        help="checkpoint from train.smoke_ppo --save; adds a trained-policy row (needs torch)",
    )
    args = ap.parse_args()
    seeds = range(args.seeds)

    def m(x: tuple[float, float]) -> float:  # mean only
        return x[0]

    header = f"{'producing':17}{'ordering':14}{'cycles':>7}{'dead%':>7}{'cons✓':>8}{'cons✗':>8}{'return':>9}{'capmin':>8}"
    print(header)
    print("-" * len(header))

    def row(pname: str, oname: str, prod: Policy, order: Policy) -> None:
        res = evaluate(prod, order, seeds=seeds, horizon=args.horizon)
        print(
            f"{pname:17}{oname:14}"
            f"{m(res['cycles']):7.0f}"
            f"{m(res['dead_rate'])*100:7.0f}"
            f"{m(res['consumer_completed']):8.1f}"
            f"{m(res['consumer_rejected']):8.1f}"
            f"{m(res['return']):9.0f}"
            f"{m(res['cap_min']):8.1f}"
        )

    for pname, oname in COMBOS:
        row(pname, oname, PRODUCING_BASELINES[pname], ORDERING_BASELINES[oname])

    # Trained policy (MAPPO; ADR-0004/0007/0008) scored on the same harness as a
    # row against the baseline bar. Lazy torch import keeps the eval layer
    # torch-optional (CLAUDE.md).
    if args.policy:
        from ..train.smoke_ppo import load_policies

        prod_p, order_p = load_policies(args.policy)
        print("-" * len(header))
        row("trained(mappo)", "trained", prod_p, order_p)


if __name__ == "__main__":
    main()
