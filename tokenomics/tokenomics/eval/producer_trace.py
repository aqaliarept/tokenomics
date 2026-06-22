"""Producer trace probe (EXP-0001 RECOMMENDATIONS step 1).

The boundary grid showed the depth-3 wall is producer-side: `train+base` collapses
to ~39 cu✓ (80% dead) while `base+base` (stock_aware producer) holds ~362. The swap
probe says *which side* fails; this trace says *why* — it drives an episode on the
SAME eval config the swap uses (`build_scale_config(0, depth)`) and snapshots, per
supply-chain layer per cycle, the producer-side state that separates the failure
modes:

  - inv        : input stock held (assembly fuel) summed over the layer
  - inbound    : replenishment ordered but not yet delivered (in-flight)
  - blocked    : composite reservations waiting on missing inputs (count)
  - resvCap    : capacity held hostage by those waiting reservations
  - cap        : current total capacity (restoration / death signal)
  - repl±      : replenishment completed (+) / rejected (−) this cycle (system-wide)
  - cu         : consumer units delivered this cycle (the objective)

Reading the columns (RECOMMENDATIONS diagnosis menu):
  inv & inbound flat at 0            -> under-ordering (never replenishes)
  inbound>0 but inv stays 0 + repl−  -> orders rejected (unfunded) — fund starvation
  resvCap high while needed inv=0    -> mistiming: capacity reserved before inputs land,
                                        reservations TTL-fail (blocked stays >0 then drops)
  cap decaying toward 0              -> dying: restoration starved, system death

Run:
    python -m tokenomics.eval.producer_trace \\
        --pt ../experiments/EXP-0001-boundary-grid/artifacts/d3_p00.pt --depth 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .baselines import ORDERING_BASELINES, PRODUCING_BASELINES, Policy
from .scale import build_scale_config
from ..domain.config import Configuration
from ..env import TokenomicsAECEnv
from ..train.smoke_ppo import load_policies


@dataclass
class LayerSnap:
    """End-of-cycle producer state summed over the producers in one supply-chain
    layer (level). `cap` is summed current capacity; primal layer 0 holds no
    inventory so inv/inbound/blocked/resvCap stay 0 there."""

    inv: int
    inbound: int
    blocked: int
    resv_cap: int
    cap: int


@dataclass
class CycleSnap:
    cycle: int
    layers: dict[int, LayerSnap]  # level -> snapshot
    cu: int  # consumer units delivered this cycle (delta)
    repl_done: int  # replenishment orders completed this cycle (delta)
    repl_rej: int  # replenishment orders rejected this cycle (delta)


def _level_of(pid: str) -> int:
    """Scale topology ids are `p{level}_{k}` (eval.scale)."""
    return int(pid.split("_")[0][1:])


def _snapshot_layers(env: TokenomicsAECEnv) -> dict[int, LayerSnap]:
    acc: dict[int, LayerSnap] = {}
    for pid, p in env.system.producing_agents.items():
        lvl = _level_of(pid)
        reserved = sum(p.reserved_capacity().values())
        snap = acc.get(lvl) or LayerSnap(0, 0, 0, 0, 0)
        snap.inv += p.inventory_units()
        snap.inbound += p.outstanding_inbound_units()
        snap.blocked += len(p.reservations)
        snap.resv_cap += reserved
        snap.cap += sum(p.capacities.values())
        acc[lvl] = snap
    return acc


def trace_episode(
    env: TokenomicsAECEnv, prod_policy: Policy, order_policy: Policy, seed: int
) -> list[CycleSnap]:
    """Drive one episode, snapshotting per-layer producer state at each cycle end.

    Mirrors `eval.runner.run_episode` but records the trace; cycle boundaries are
    detected by `env.current_cycle` advancing (the env completes a cycle on the last
    agent's step). Per-cycle system deltas (cu, repl±) come from diffing the running
    `episode_stats`, so no env instrumentation is needed."""
    env.reset(seed=seed)
    for a in env.possible_agents:
        env.action_space(a).seed(seed)
    max_iter = env.horizon * len(env.possible_agents) * 3 + 100

    snaps: list[CycleSnap] = []
    prev_cycle = 0
    prev_cu = prev_done = prev_rej = 0
    for agent in env.agent_iter(max_iter=max_iter):
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        role = env._role[agent][0]
        action = prod_policy(env, agent) if role == "prod" else order_policy(env, agent)
        env.step(action)
        if env.current_cycle > prev_cycle:
            s = env.episode_stats
            snaps.append(CycleSnap(
                cycle=env.current_cycle,
                layers=_snapshot_layers(env),
                cu=s["consumer_completed_units"] - prev_cu,
                repl_done=s["replenishment_completed"] - prev_done,
                repl_rej=s["replenishment_rejected"] - prev_rej,
            ))
            prev_cycle = env.current_cycle
            prev_cu = s["consumer_completed_units"]
            prev_done = s["replenishment_completed"]
            prev_rej = s["replenishment_rejected"]
    return snaps


def _sample_cycles(n: int) -> list[int]:
    """First 15 cycles (priming dynamics) then every 10th — enough to read the
    pipeline filling or starving without dumping 100 rows."""
    idx = list(range(min(n, 15)))
    idx += [i for i in range(15, n) if (i + 1) % 10 == 0]
    return idx


def print_trace(label: str, snaps: list[CycleSnap], depth: int) -> None:
    cu_total = sum(s.cu for s in snaps)
    done_total = sum(s.repl_done for s in snaps)
    rej_total = sum(s.repl_rej for s in snaps)
    dead = len(snaps) < 100  # truncation horizon is 100; shorter => system death
    print(f"\n=== {label} ===")
    print(f"  episode: cycles={len(snaps)} dead={dead}  "
          f"cu_total={cu_total}  repl_done={done_total}  repl_rej={rej_total}")
    levels = list(range(depth))
    hdr = f"  {'cyc':>3} " + " ".join(
        f"|L{l}{'':>1}{'inv':>4}{'inb':>4}{'blk':>3}{'rsv':>5}{'cap':>5}" for l in levels
    ) + f" | {'repl+':>5} {'repl-':>5} {'cu':>3}"
    print(hdr)
    for i in _sample_cycles(len(snaps)):
        s = snaps[i]
        cells = []
        for l in levels:
            ls = s.layers.get(l, LayerSnap(0, 0, 0, 0, 0))
            cells.append(f"|  {ls.inv:>4}{ls.inbound:>4}{ls.blocked:>3}{ls.resv_cap:>5}{ls.cap:>5}")
        print(f"  {s.cycle:>3} " + " ".join(cells) +
              f" | {s.repl_done:>5} {s.repl_rej:>5} {s.cu:>3}")


def run(pt: str, depth: int, seed: int, horizon: int, config: Configuration) -> None:
    train_prod, _train_ord = load_policies(pt)
    base_prod = PRODUCING_BASELINES["stock_aware"]
    base_ord = ORDERING_BASELINES["proportional"]
    # Both runs hold the orderer at the strong baseline (isolating the producer, as the
    # swap's `train+base` vs `base+base` columns do).
    for label, prod in (("train+base", train_prod), ("stock_aware+base", base_prod)):
        env = TokenomicsAECEnv(config=config, horizon=horizon)
        snaps = trace_episode(env, prod, base_ord, seed)
        print_trace(f"{label}  seed={seed}", snaps, depth)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True, help="trained checkpoint (the train-side producer)")
    ap.add_argument("--depth", type=int, required=True, help="cell topology depth")
    ap.add_argument("--seed", type=int, default=0, help="episode seed to trace in detail")
    ap.add_argument("--horizon", type=int, default=100)
    args = ap.parse_args()
    cfg = build_scale_config(0, depth=args.depth)
    run(args.pt, args.depth, args.seed, args.horizon, cfg)


if __name__ == "__main__":
    main()
