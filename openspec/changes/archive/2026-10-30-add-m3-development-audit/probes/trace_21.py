"""Task 2.1 one-shot trace (add-m3-development-audit D2 check (c)): one episode of one router on one
registry v2 development world, horizon 100, entity_planner producer, the router's routing_audit
contract and action seed (replicate 0 for stochastic routers).

Runs the episode three times: (1) logging run = tracer + per-order decision log + per-cycle bid
outcomes (also a warm-up), (2) plain runs = the driver's job without tracer (ContractMonitor on),
(3) traced runs = the driver's job with tracer. Checks (1)-(3) give identical episode stats. (2) and (3) run --repeats times; the min is the
per-episode time.
Writes <out>/<world>__<router>.json.

usage: python trace_21.py --world W --router R --out DIR [--horizon 100]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from tokenomics.domain.system import order_kind
from tokenomics.env.decode import decode_ordering
from tokenomics.eval.baselines import PRODUCING_BASELINES
from tokenomics.eval.contract import ContractMonitor
from tokenomics.eval.joint_stack import joint_stack_env
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.roster import JOINT_STACK, ROSTER_V2, load_roster
from tokenomics.eval.routing_audit import ordering_policy, router_table
from tokenomics.eval.routing_trace import RoutingTracer, order_class
from tokenomics.eval.runner import run_episode
from tokenomics.eval.seeding import replicate_seed

EXCERPT_CYCLES = 3
EXCERPT_ORDERS_PER_CYCLE = 3


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def tie_info(shares: np.ndarray, q: int) -> tuple[bool, bool, float]:
    """(remainder handed out among tied fractional parts, lowest-index tied producer won it,
    units moved by rounding = sum |q_i - share_i q| / 2), replaying decode's largest remainder."""
    raw = shares * q
    floors = np.floor(raw).astype(int)
    rem = q - int(floors.sum())
    frac = raw - floors
    order = np.argsort(-frac)
    final = floors.copy()
    for i in range(rem):
        final[order[i % len(final)]] += 1
    moved = float(np.abs(final - raw).sum() / 2)
    if rem <= 0 or rem >= len(frac):
        return False, False, moved
    cut = frac[order[rem - 1]]
    tied = np.flatnonzero(np.abs(frac - cut) < 1e-9)
    losers = [j for j in tied if j not in set(order[:rem])]
    if not losers:
        return False, False, moved
    winners = [j for j in tied if j in set(order[:rem])]
    return True, bool(min(winners) < min(losers)), moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2, help="timed plain/traced runs (min taken)")
    args = ap.parse_args()

    entry = load_registry(REGISTRY_V2).entry(args.world)
    ep = entry.episode_seeds[0]
    info = next(r for r in router_table(load_roster(ROSTER_V2)) if r["router"] == args.router)
    assert info["contract"] == JOINT_STACK
    action_seed = replicate_seed(entry.id, ep, 0) if info["random"] else None
    cfg = entry.config()
    prod = PRODUCING_BASELINES["entity_planner"]
    router = ordering_policy(args.router)

    # (1) logging run
    env = joint_stack_env(cfg, args.horizon)
    tracer = RoutingTracer()
    tracer.attach(env)
    decisions: list[dict] = []
    outcomes: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    resolutions: dict[int, dict[str, tuple]] = defaultdict(dict)
    wrapped: set[int] = set()

    def logged(e, agent):  # noqa: ANN001, ANN202
        s = e.system
        if id(s) not in wrapped:
            wrapped.add(id(s))
            inner = s.complete_cycle

            def complete_cycle():  # noqa: ANN202
                c = s.cycle_counter
                res = inner()
                for b in res.bid_outcomes:
                    outcomes[c][b.order_id].append((b.producer_id, b.quantity, b.outcome, b.reason))
                for r in res.order_resolutions:
                    resolutions[c][r.order_id] = (r.completed, r.completed_qty)
                return res

            s.complete_cycle = complete_cycle
        act = router(e, agent)
        sp = e.spec_
        _, aid = e._role[agent]
        view = s.ordering_agents[aid].view(s.producer_infos)
        cmd = decode_ordering(sp, np.asarray(act), view)
        mat = np.asarray(act, dtype=np.float64).reshape(sp.k_stream, sp.num_producers)
        for k, oid in enumerate(sorted(view.incoming)):
            oi = view.incoming[oid]
            elig = [i for i, pid in enumerate(sp.producer_ids) if pid in view.eligible[oid]]
            row = mat[k] if k < sp.k_stream else np.zeros(sp.num_producers)
            shares = softmax(row[elig]) if elig else np.zeros(0)
            tied, lowest_won, moved = tie_info(shares, oi.quantity) if elig else (False, False, 0.0)
            decisions.append({
                "cycle": s.cycle_counter, "agent": agent, "order": oid, "q": oi.quantity,
                "class": order_class(order_kind(s.orders[oid])), "slot": k,
                "overflow": k >= sp.k_stream,
                "eligible": [sp.producer_ids[i] for i in elig],
                "shares": [round(float(x), 3) for x in shares],
                "split": {sub.producer_id: sub.quantity for sub in cmd.orders[oid]},
                "tie": tied, "lowest_won_tie": lowest_won, "rounding_moved": moved})
        return act

    t0 = time.perf_counter()
    logged_stats = run_episode(env, prod, logged, seed=ep, action_seed=action_seed)
    t_logged = time.perf_counter() - t0
    summary, series = tracer.summary(), tracer.series()

    # (2) plain and (3) traced, both as the driver's job (ContractMonitor on)
    def job(trace: bool) -> tuple[float, dict]:
        env = joint_stack_env(cfg, args.horizon)
        if trace:
            RoutingTracer().attach(env)
        monitor = ContractMonitor(env)
        t0 = time.perf_counter()
        stats = run_episode(env, prod, router, seed=ep, action_seed=action_seed, monitor=monitor)
        dt = time.perf_counter() - t0
        monitor.verify(dead=bool(stats["dead"]), cycles=stats["cycles"], horizon=args.horizon)
        return dt, stats

    plain = [job(False) for _ in range(args.repeats)]
    traced = [job(True) for _ in range(args.repeats)]
    t_plain, plain_stats = min(t for t, _ in plain), plain[0][1]
    t_traced, traced_stats = min(t for t, _ in traced), traced[0][1]
    assert all(s == plain_stats for _, s in plain + traced)
    assert plain_stats == traced_stats == logged_stats, "tracer/logging changed the episode"

    # excerpt: the first EXCERPT_CYCLES cycles with consumer decisions
    cons = [d for d in decisions if d["class"] == "consumer"]
    cycles = sorted({d["cycle"] for d in cons})[:EXCERPT_CYCLES]
    excerpt = []
    for c in cycles:
        for d in [d for d in cons if d["cycle"] == c][:EXCERPT_ORDERS_PER_CYCLE]:
            excerpt.append({**d, "outcomes": outcomes[c].get(d["order"], []),
                            "resolution": resolutions[c].get(d["order"])})

    def agg(cls: str) -> dict:
        ds = [d for d in decisions if d["class"] == cls]
        multi = [d for d in ds if len(d["eligible"]) > 1]
        small = [d for d in multi if d["q"] <= 3]
        ties = [d for d in multi if d["tie"]]
        qhist: dict[int, int] = defaultdict(int)
        for d in ds:
            qhist[d["q"]] += 1
        return {
            "decisions": len(ds), "multi_eligible": len(multi),
            "q_hist": dict(sorted(qhist.items())),
            "split_decisions": sum(len(d["split"]) > 1 for d in multi),
            "tie_decisions": len(ties),
            "tie_lowest_index_won": sum(d["lowest_won_tie"] for d in ties),
            "small_q_decisions": len(small),
            "small_q_rounded": sum(d["rounding_moved"] >= 0.25 for d in small),
            "rounding_units_moved_mean": (float(np.mean([d["rounding_moved"] for d in multi]))
                                          if multi else 0.0),
            "overflow_decisions": sum(d["overflow"] for d in ds),
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "world": args.world, "episode": ep, "router": args.router, "kind": info["kind"],
        "action_seed": action_seed, "horizon": args.horizon,
        "units": plain_stats["consumer_completed_units"], "dead": plain_stats["dead"],
        "cycles": plain_stats["cycles"],
        "seconds": {"logged": t_logged, "plain": t_plain, "traced": t_traced,
                    "plain_all": [t for t, _ in plain], "traced_all": [t for t, _ in traced]},
        "tracer_overhead": t_traced / t_plain,
        "summary": summary,
        "fund_share_by_producer": series["fund_share_by_producer"],
        "decision_aggregates": {"consumer": agg("consumer"), "producer": agg("producer")},
        "excerpt": excerpt,
    }
    (out / f"{args.world}__{args.router}.json").write_text(json.dumps(doc, indent=1, default=str))
    with (out / f"{args.world}__{args.router}.decisions.jsonl").open("w") as fh:
        for d in decisions:
            fh.write(json.dumps(d) + "\n")
    print(f"{args.world} {args.router}: units {doc['units']} plain {t_plain:.2f}s traced "
          f"{t_traced:.2f}s (x{doc['tracer_overhead']:.3f}) logged {t_logged:.2f}s")


if __name__ == "__main__":
    main()
