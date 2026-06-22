"""Producer-first warm-start gate (ADR-0049 / Phase 1).

The depth-3 producer wall (EXP-0001): from-scratch PPO collapses on `d3_p00` while
the baseline producer holds ~362 cu✓. The fix this driver scaffolds is staged —
clone the `obs_stock_aware` producer (it acts through the same observation contract
as the learned net), then gate on the producer-isolated swap column before any joint
PPO. `swap_gate` is that gate: it runs the 2×2 swap and renders PASS/FAIL on
`clone+base` (cloned producer, baseline orderer), the column that isolates the
producer as the blocker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from ..domain.config import Configuration
from ..eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from ..eval.runner import evaluate
from ..eval.scale import build_scale_config, scale_topology_set
from ..train.clone_fidelity import dagger_producer_first
from ..train.smoke_ppo import behavior_clone, load_policies


@dataclass
class GateRow:
    """One swap combo's headline metrics. `name` is `<prod>+<order>` (e.g.
    `clone+base`); `units` is mean consumer_completed_units (the objective)."""

    name: str
    units: float
    dead_rate: float


@dataclass
class GateResult:
    """The 2×2 swap table plus the producer-isolated verdict (ADR-0049). The gate
    reads only the `clone+base` row against `base+base`, since that column is what
    isolates the producer as the depth-3 blocker."""

    rows: list[GateRow]
    pass_frac: float
    max_dead: float

    def row(self, name: str) -> GateRow:
        return next(r for r in self.rows if r.name == name)

    @property
    def base_base(self) -> GateRow:
        return self.row("base+base")

    @property
    def clone_base(self) -> GateRow:
        return self.row("clone+base")

    @property
    def passed(self) -> bool:
        return (
            self.clone_base.units >= self.pass_frac * self.base_base.units
            and self.clone_base.dead_rate <= self.max_dead
        )


def swap_gate(
    clone_prod: Policy,
    clone_order: Policy,
    base_prod: Policy,
    base_order: Policy,
    *,
    config: Configuration,
    swap_seeds: int = 5,
    horizon: int = 100,
    pass_frac: float = 0.8,
    max_dead: float = 0.05,
) -> GateResult:
    """Run the 2×2 producer/order swap and score the producer-isolated gate. Combos
    are named `<prod>+<order>`; the verdict (ADR-0049) is `clone+base units >=
    pass_frac * base+base units` with `clone+base` death `<= max_dead`."""
    combos = {
        "base+base": (base_prod, base_order),
        "clone+clone": (clone_prod, clone_order),
        "clone+base": (clone_prod, base_order),
        "base+clone": (base_prod, clone_order),
    }
    rows: list[GateRow] = []
    for name, (pp, op) in combos.items():
        m = evaluate(pp, op, seeds=range(swap_seeds), horizon=horizon, config=config)
        rows.append(GateRow(name, m["consumer_completed_units"][0], m["dead_rate"][0]))
    return GateResult(rows=rows, pass_frac=pass_frac, max_dead=max_dead)


@dataclass
class HeldOutRow:
    """One held-out topology's `clone+base` score against that topology's own
    `base+base` (Phase 2 step 6). Absolute units are not comparable across topologies
    (demand bounds / capacities vary by seed), so `ratio = clone+base / base+base` is
    the fair judgment — same normalization the d3_p00 gate uses."""

    seed: int
    units: float        # clone+base
    dead: float         # clone+base death rate
    base_units: float   # base+base on the SAME topology (the per-seed reference)
    ratio: float        # clone+base / base+base


def held_out_report(
    clone_prod: Policy,
    base_prod: Policy,
    base_order: Policy,
    *,
    depth: int,
    seeds: "list[int]",
    eval_seeds: int = 3,
    horizon: int = 100,
    log: Callable[[str], None] = print,
) -> list[HeldOutRow]:
    """Score the cloned producer + base orderer on UNSEEN same-depth topologies, each
    normalized by that topology's own `base+base` (ADR-0049 Phase 2). Surfaces slot /
    operating-point memorization: if the ratio collapses relative to the trained-topology
    gate, the clone overfit the training topologies rather than learning the local rule."""
    rows: list[HeldOutRow] = []
    log("held-out topologies (clone+base vs base+base):")
    for s in seeds:
        cfg = build_scale_config(s, depth=depth)
        cm = evaluate(clone_prod, base_order, seeds=range(eval_seeds), horizon=horizon, config=cfg)
        bm = evaluate(base_prod, base_order, seeds=range(eval_seeds), horizon=horizon, config=cfg)
        cu = cm["consumer_completed_units"][0]
        bu = bm["consumer_completed_units"][0]
        ratio = cu / bu if bu > 0 else float("nan")
        row = HeldOutRow(s, cu, cm["dead_rate"][0], bu, ratio)
        rows.append(row)
        log(f"  seed {s}: clone+base={cu:8.1f}  base+base={bu:8.1f}  "
            f"ratio={ratio:5.2f}  dead={row.dead:5.2f}")
    return rows


def print_gate(res: GateResult, log: Callable[[str], None] = print) -> None:
    """Render the swap table + the producer-isolated verdict (ADR-0049)."""
    log(f"{'combo':12}  {'units✓':>8}  {'dead':>5}")
    for r in res.rows:
        log(f"{r.name:12}  {r.units:8.1f}  {r.dead_rate:5.2f}")
    cb, bb = res.clone_base, res.base_base
    verdict = "PASS" if res.passed else "FAIL"
    log(f"gate clone+base {cb.units:.1f} vs {res.pass_frac:.2f}*base+base "
        f"{res.pass_frac * bb.units:.1f}  dead {cb.dead_rate:.2f}<= {res.max_dead:.2f}"
        f"  -> {verdict}")


def run_warm_start(
    *,
    depth: int,
    mode: str = "dagger",
    expert: str = "obs_stock_aware",
    order_expert: str = "proportional",
    dagger_iters: int = 3,
    seeds: int = 2,
    horizon: int = 100,
    epochs: int = 1500,
    lr: float = 3e-3,
    seed: int = 0,
    log_std_save: float | None = -2.5,
    save: str | None = None,
    train_seeds: int = 1,
    perms: int = 0,
    perm_seed: int = 0,
    held_out_seeds: int = 0,
    swap_seeds: int = 5,
    pass_frac: float = 0.8,
    max_dead: float = 0.05,
    trace: bool = False,
    trace_seed: int = 0,
    log: Callable[[str], None] = print,
) -> GateResult:
    """Phase-1 producer-first driver (ADR-0049): clone the `obs_stock_aware` producer
    (DAgger by default — same observation contract as the learned net), save a
    PPO-loadable checkpoint, then score the producer-isolated swap gate. `clone+base`
    is the column that isolates the producer wall; the base columns hold the strong
    white-box `stock_aware`+`proportional` reference (EXP-0001 `base+base`)."""
    cfg = build_scale_config(0, depth=depth)  # the d3_p00 gate topology (seed 0)
    expert_prod = PRODUCING_BASELINES[expert]
    expert_ord = ORDERING_BASELINES[order_expert]

    base_ord = ORDERING_BASELINES["proportional"]
    if mode == "dagger":
        # decode-aware producer-first DAgger (ADR-0049 Phase 1 fix): plain full-vector
        # MSE clones a producer that decodes the wrong command (restoration false
        # positives at decode's >0 boundary). Relabel under the BASE orderer to match
        # the gate's `clone+base` deployment condition. Train across `train_seeds`
        # same-shape topologies (Phase 2 multi-topology sampling) to weaken slot /
        # operating-point memorization; seed 0 (the gate topology) is always included.
        # `perms` adds slot-permuted relabelings of each seed (Phase 2 steps 1-4) so the
        # net can't lean on fixed product/producer slot positions.
        train_configs = scale_topology_set(
            range(train_seeds), depth=depth, perms=perms, perm_seed=perm_seed,
        )
        dagger_producer_first(
            train_configs, expert_prod, expert_ord, base_ord, dagger_iters=dagger_iters,
            seeds=range(seeds), horizon=horizon, epochs=epochs, lr=lr, seed=seed,
            log_std_save=log_std_save, save=save, log=log,
        )
    elif mode == "bc":
        # legacy plain-MSE clone (the documented low-fidelity baseline / diagnostic).
        behavior_clone(
            cfg, expert_prod, expert_ord, seeds=range(seeds), horizon=horizon,
            epochs=epochs, lr=lr, seed=seed, log_std_save=log_std_save,
            save=save, log=log,
        )
    else:
        raise ValueError(f"unknown mode {mode!r} (want 'bc' or 'dagger')")

    clone_prod, clone_ord = load_policies(save)
    base_prod = PRODUCING_BASELINES["stock_aware"]
    res = swap_gate(
        clone_prod, clone_ord, base_prod, base_ord, config=cfg,
        swap_seeds=swap_seeds, horizon=horizon, pass_frac=pass_frac, max_dead=max_dead,
    )
    print_gate(res, log)

    if held_out_seeds > 0:
        # score unseen same-depth topologies (seeds past the training set) so slot /
        # operating-point memorization is visible next to the trained-topology gate.
        held = list(range(train_seeds, train_seeds + held_out_seeds))
        held_out_report(clone_prod, base_prod, base_ord, depth=depth, seeds=held,
                        eval_seeds=min(swap_seeds, 3), horizon=horizon, log=log)

    if trace:
        # producer trace for the gate column + the stock_aware ceiling on one seed
        # (ADR-0049 acceptance: L1 capacity must not reach zero).
        from ..eval.producer_trace import print_trace, trace_episode
        from ..env import TokenomicsAECEnv

        for label, prod in (("clone+base", clone_prod), ("stock_aware+base", base_prod)):
            env = TokenomicsAECEnv(config=cfg, horizon=horizon)
            snaps = trace_episode(env, prod, base_ord, trace_seed)
            print_trace(f"{label}  seed={trace_seed}", snaps, depth)

    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--mode", choices=("bc", "dagger"), default="dagger")
    ap.add_argument("--expert", default="obs_stock_aware", help="PRODUCING_BASELINES key")
    ap.add_argument("--order-expert", default="proportional", help="ORDERING_BASELINES key")
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2, help="clone rollout seeds")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-std-save", type=float, default=-2.5)
    ap.add_argument("--save", required=True)
    ap.add_argument("--train-seeds", type=int, default=1,
                    help="number of same-depth topologies to clone across (Phase 2 "
                         "multi-topology; seed 0 = the gate topology is always included)")
    ap.add_argument("--perms", type=int, default=0,
                    help="slot-permuted relabelings added per training seed (Phase 2 "
                         "slot randomization; 0 = none)")
    ap.add_argument("--perm-seed", type=int, default=0, help="RNG seed for slot permutations")
    ap.add_argument("--held-out-seeds", type=int, default=0,
                    help="unseen topologies to score clone+base on after the gate")
    ap.add_argument("--swap-seeds", type=int, default=5)
    ap.add_argument("--pass-frac", type=float, default=0.8)
    ap.add_argument("--max-dead", type=float, default=0.05)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-seed", type=int, default=0)
    args = ap.parse_args()

    run_warm_start(
        depth=args.depth, mode=args.mode, expert=args.expert,
        order_expert=args.order_expert, dagger_iters=args.dagger_iters,
        seeds=args.seeds, horizon=args.horizon, epochs=args.epochs, lr=args.lr,
        seed=args.seed, log_std_save=args.log_std_save, save=args.save,
        train_seeds=args.train_seeds, perms=args.perms, perm_seed=args.perm_seed,
        held_out_seeds=args.held_out_seeds,
        swap_seeds=args.swap_seeds, pass_frac=args.pass_frac, max_dead=args.max_dead,
        trace=args.trace, trace_seed=args.trace_seed,
    )


if __name__ == "__main__":
    main()
