"""Quantity-rung producer evaluator (EXP-0003 step 2).

The swap gate (`swap_eval`) scores the clone only at the `p=0` (easiest) operating
point of a single topology. EXP-0003 asks the harder question — does the standalone
producer hold as the *quantity* axis climbs? This driver crosses the trained producer
against the strong white-box `stock_aware` producer, BOTH paired with the fixed
`proportional` orderer (the standalone-producer condition: orderer never varies), over
the curriculum quantity rungs `staged_config(build_scale_config(seed), p)` for several
config seeds. The verdict per row is the same normalization the warm-start gate uses:
`ratio = clone_units / base_units >= 0.80` with clone death `<= 0.05` (ADR-0049).

Emits a CSV (one row per checkpoint × rung × config_seed) so a regression on any rung
or topology is visible before any improvement training is attempted."""

from __future__ import annotations

import argparse
import csv

from ..eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from ..eval.runner import evaluate
from ..eval.scale import build_scale_config
from ..train.smoke_ppo import load_policies, staged_config

# A row clears when the clone matches the base within the gate margin AND stays alive.
RATIO_FLOOR = 0.80
MAX_DEAD = 0.05


def _verdict(ratio: float, clone_dead: float) -> str:
    return "ok" if (ratio >= RATIO_FLOOR and clone_dead <= MAX_DEAD) else "FAIL"


def producer_rungs(
    pt: str | None,
    *,
    depth: int,
    rungs: list[float],
    config_seeds: list[int],
    eval_seeds: int,
    horizon: int,
    out: str,
    base_prod_name: str = "stock_aware",
    order_name: str = "proportional",
    train_prod_name: str | None = None,
) -> list[dict]:
    """Evaluate the train-side producer + `proportional` vs `stock_aware` + `proportional`
    on each (config_seed, rung) staged topology. Returns the rows and writes them to `out`.

    The train side is a trained checkpoint (`pt`) by default; pass `train_prod_name` to use
    a white-box baseline producer instead (e.g. `obs_stock_aware` vs `stock_aware`). The
    latter is the EXP-0004 crux diagnostic: it isolates *expert* weakness on a rung from
    *cloning* infidelity, since no checkpoint is involved.

    Rung application matches the trainer's curriculum: `staged_config(base, p)` ramps
    the quantity axis off `build_scale_config(seed, depth)`, so a rung evaluator and the
    `producer_improve` trainer score the producer on identical operating points."""
    if train_prod_name is not None:
        train_prod = PRODUCING_BASELINES[train_prod_name]
        label = f"baseline:{train_prod_name}"
    else:
        if pt is None:
            raise ValueError("producer_rungs needs either `pt` or `train_prod_name`")
        train_prod, _train_ord = load_policies(pt)
        label = pt
    base_prod: Policy = PRODUCING_BASELINES[base_prod_name]
    order: Policy = ORDERING_BASELINES[order_name]

    rows: list[dict] = []
    print(f"\n=== {label}  vs {base_prod_name}  depth={depth}  "
          f"config_seeds={config_seeds}  eval_seeds={eval_seeds} ===")
    print(f"{'rung':>6}  {'seed':>4}  {'clone✓':>8}  {'base✓':>8}  {'ratio':>6}  "
          f"{'cl_dead':>7}  {'bs_dead':>7}  verdict")
    for seed in config_seeds:
        base_cfg = build_scale_config(seed, depth=depth)
        for p in rungs:
            cfg = staged_config(base_cfg, p)
            cm = evaluate(train_prod, order, seeds=range(eval_seeds), horizon=horizon, config=cfg)
            bm = evaluate(base_prod, order, seeds=range(eval_seeds), horizon=horizon, config=cfg)
            clone_units = cm["consumer_completed_units"][0]
            base_units = bm["consumer_completed_units"][0]
            ratio = clone_units / base_units if base_units > 0 else float("nan")
            clone_dead = cm["dead_rate"][0]
            base_dead = bm["dead_rate"][0]
            verdict = _verdict(ratio, clone_dead)
            rows.append({
                "checkpoint": label,
                "rung": f"{p:.2f}",
                "config_seed": seed,
                "clone_units": round(clone_units, 1),
                "base_units": round(base_units, 1),
                "ratio": round(ratio, 3),
                "clone_dead": round(clone_dead, 3),
                "base_dead": round(base_dead, 3),
                "verdict": verdict,
            })
            print(f"{p:6.2f}  {seed:4d}  {clone_units:8.1f}  {base_units:8.1f}  "
                  f"{ratio:6.2f}  {clone_dead:7.2f}  {base_dead:7.2f}  {verdict}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "checkpoint", "rung", "config_seed", "clone_units", "base_units",
            "ratio", "clone_dead", "base_dead", "verdict",
        ])
        w.writeheader()
        w.writerows(rows)
    n_fail = sum(1 for r in rows if r["verdict"] != "ok")
    print(f"\n{len(rows)} rows -> {out}  ({n_fail} FAIL)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", help="trained checkpoint (the train-side producer)")
    ap.add_argument("--train-baseline",
                    help="use this white-box baseline producer as the train side instead of "
                         "--pt (e.g. obs_stock_aware) — the expert-vs-expert crux diagnostic")
    ap.add_argument("--base-prod", default="stock_aware", help="comparison (denominator) producer")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--rungs", default="0.00,0.10,0.24,0.25",
                    help="comma-separated quantity rungs p in [0,1] for staged_config")
    ap.add_argument("--config-seeds", default="0,3,4,5",
                    help="comma-separated build_scale_config seeds (topologies)")
    ap.add_argument("--eval-seeds", type=int, default=5, help="episode seeds per cell")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--order", default="proportional", choices=tuple(ORDERING_BASELINES),
                    help="ordering baseline held fixed for both arms (orderer-coupling probe)")
    ap.add_argument("--out", required=True, help="CSV output path")
    args = ap.parse_args()
    if not args.pt and not args.train_baseline:
        ap.error("pass either --pt or --train-baseline")
    producer_rungs(
        args.pt,
        depth=args.depth,
        rungs=[float(x) for x in args.rungs.split(",")],
        config_seeds=[int(x) for x in args.config_seeds.split(",")],
        eval_seeds=args.eval_seeds,
        horizon=args.horizon,
        out=args.out,
        base_prod_name=args.base_prod,
        order_name=args.order,
        train_prod_name=args.train_baseline,
    )


if __name__ == "__main__":
    main()
