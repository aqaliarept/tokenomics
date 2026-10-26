"""Signal check: do different policies produce materially different outcomes?

If a do-nothing policy scores about the same as a sensible one, the env poses no
learning problem and the dynamics need tuning. Run:

    python -m tokenomics.eval.signal_check
    python -m tokenomics.eval.signal_check --horizon 100 --seeds 20
"""

from __future__ import annotations

import argparse

from ..domain import load_config
from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .runner import evaluate

# (producing baseline, ordering baseline) pairs worth comparing
COMBOS = [
    ("noop", "even_split"),
    ("always_restore", "even_split"),
    ("restore_when_low", "even_split"),
    ("stock_aware", "even_split"),
    ("stock_aware", "proportional"),
    # observation-compatible bar (NEXT STEPS stage 1): obs-only, the fair bar for RL.
    ("obs_stock_aware", "even_split"),
    ("obs_stock_aware", "obs_price"),
    ("always_both", "even_split"),
    ("random", "random"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="env config JSON to score baselines on (default: phase1.json)",
    )
    ap.add_argument(
        "--policy",
        type=str,
        default=None,
        help="checkpoint written by train.mlp_actor.save_policies; adds a trained-policy row (needs torch)",
    )
    args = ap.parse_args()
    seeds = range(args.seeds)
    cfg = load_config(args.config) if args.config else None

    def m(x: tuple[float, float]) -> float:  # mean only
        return x[0]

    header = f"{'producing':17}{'ordering':14}{'cycles':>7}{'dead%':>7}{'cons✓':>8}{'cu✓':>8}{'cons✗':>8}{'return':>9}{'capmin':>8}"
    print(header)
    print("-" * len(header))

    def row(pname: str, oname: str, prod: Policy, order: Policy) -> None:
        res = evaluate(prod, order, seeds=seeds, horizon=args.horizon, config=cfg)
        print(
            f"{pname:17}{oname:14}"
            f"{m(res['cycles']):7.0f}"
            f"{m(res['dead_rate'])*100:7.0f}"
            f"{m(res['consumer_completed']):8.1f}"
            f"{m(res['consumer_completed_units']):8.1f}"
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
        from ..env.spaces import build_spec, config_fingerprint
        from ..train.mlp_actor import (
            checkpoint_fingerprint,
            checkpoint_obs_dims,
            load_policies,
        )

        # #7: warn if the policy is being scored on a config it was not trained on.
        # The fingerprint covers shape + economy identity; a mismatch means the row
        # below is at best a transfer test, at worst a silently wrong comparison.
        from ..env.aec_env import DEFAULT_CONFIG

        eval_cfg = cfg if cfg is not None else load_config(DEFAULT_CONFIG)
        trained_fp = checkpoint_fingerprint(args.policy)
        eval_fp = config_fingerprint(eval_cfg)
        if trained_fp is not None and trained_fp != eval_fp:
            print(f"WARNING: policy trained on config {trained_fp} but evaluating on {eval_fp} "
                  "(#7 fingerprint mismatch) — this is a transfer/off-config eval")

        # [P2 review]: a fingerprint mismatch may also be a *shape* mismatch — the
        # actor's input/output layers were sized to a different spec. Scoring anyway
        # crashes deep in the actor on a matrix-shape RuntimeError. Compare the stored
        # actor dims to the eval spec and skip the trained row when incompatible; only
        # same-shape transfer tests can run.
        sp = build_spec(eval_cfg)
        ck = checkpoint_obs_dims(args.policy)
        expected = {
            "prod_obs": sp.producing_obs_dim,
            "prod_act": sp.producing_action_dim,
            "order_obs": sp.ordering_obs_dim,
            "order_act": sp.ordering_action_dim,
        }
        mismatched = {k: (ck[k], expected[k]) for k in expected if ck[k] != expected[k]}
        if mismatched:
            print("-" * len(header))
            print("SKIP trained(mappo): checkpoint actor dims incompatible with eval "
                  f"config spec {mismatched} (checkpoint vs spec) — cannot score, "
                  "retrain on this config or eval on the trained one")
            return

        prod_p, order_p = load_policies(args.policy)
        print("-" * len(header))
        row("trained(mappo)", "trained", prod_p, order_p)


if __name__ == "__main__":
    main()
