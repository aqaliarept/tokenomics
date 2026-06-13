"""Signal check: do different policies produce materially different outcomes?

If a do-nothing policy scores about the same as a sensible one, the env poses no
learning problem and the dynamics need tuning. Run:

    python -m tokenomics.eval.signal_check
    python -m tokenomics.eval.signal_check --horizon 100 --seeds 20
"""

from __future__ import annotations

import argparse

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from .runner import evaluate

# (producing baseline, ordering baseline) pairs worth comparing
COMBOS = [
    ("noop", "even_split"),
    ("always_restore", "even_split"),
    ("restore_when_low", "even_split"),
    ("always_both", "even_split"),
    ("restore_when_low", "proportional"),
    ("random", "random"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = range(args.seeds)

    def m(x: tuple[float, float]) -> float:  # mean only
        return x[0]

    header = f"{'producing':17}{'ordering':14}{'cycles':>7}{'dead%':>7}{'cons✓':>8}{'cons✗':>8}{'return':>9}{'capmin':>8}"
    print(header)
    print("-" * len(header))
    for pname, oname in COMBOS:
        res = evaluate(
            PRODUCING_BASELINES[pname], ORDERING_BASELINES[oname], seeds=seeds, horizon=args.horizon
        )
        print(
            f"{pname:17}{oname:14}"
            f"{m(res['cycles']):7.0f}"
            f"{m(res['dead_rate'])*100:7.0f}"
            f"{m(res['consumer_completed']):8.1f}"
            f"{m(res['consumer_rejected']):8.1f}"
            f"{m(res['return']):9.0f}"
            f"{m(res['cap_min']):8.1f}"
        )


if __name__ == "__main__":
    main()
