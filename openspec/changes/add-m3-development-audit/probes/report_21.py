"""Render the task 2.1 trace JSONs (trace_21.py output) as markdown tables for routing_trace.md
and sizing.md. usage: python report_21.py DIR > tables.md"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROUTERS = ["even_split", "proportional", "blend_0.5", "random", "round_robin",
           "even_split_rotating", "proportional_load_aware", "proportional_whole", "random_whole"]
WORLDS = ["dev-sparse_wide-42-shocked-00", "dev-sparse_wide-12-calm-03"]


def fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def main() -> None:
    d = Path(sys.argv[1])
    docs = {(w, r): json.loads((d / f"{w}__{r}.json").read_text()) for w in WORLDS for r in ROUTERS}
    for w in WORLDS:
        print(f"\n### {w} (episode {docs[(w, ROUTERS[0])]['episode']})\n")
        print("Tracer aggregates (`RoutingTracer.summary()`, task 1.1), one column per router:\n")
        print("| metric | " + " | ".join(ROUTERS) + " |")
        print("|---|" + "---|" * len(ROUTERS))
        print("| units | " + " | ".join(fmt(docs[(w, r)]["units"]) for r in ROUTERS) + " |")
        print("| dead | " + " | ".join(str(docs[(w, r)]["dead"]) for r in ROUTERS) + " |")
        for k in docs[(w, ROUTERS[0])]["summary"]:
            print(f"| {k} | " + " | ".join(fmt(docs[(w, r)]["summary"][k]) for r in ROUTERS) + " |")
        print("\nDecision-level checks (every routed order of the episode, consumer class; "
              "replayed from the router's action):\n")
        cols = ["decisions", "multi_eligible", "split_decisions", "tie_decisions",
                "tie_lowest_index_won", "small_q_decisions", "small_q_rounded",
                "rounding_units_moved_mean", "overflow_decisions"]
        print("| router | q histogram | " + " | ".join(cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for r in ROUTERS:
            a = docs[(w, r)]["decision_aggregates"]["consumer"]
            print(f"| {r} | {a['q_hist']} | " + " | ".join(fmt(a[c]) for c in cols) + " |")
        print("\nSame, producer orders (replenishment + investment):\n")
        print("| router | q histogram | " + " | ".join(cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for r in ROUTERS:
            a = docs[(w, r)]["decision_aggregates"]["producer"]
            print(f"| {r} | {a['q_hist']} | " + " | ".join(fmt(a[c]) for c in cols) + " |")
        print("\nPer-order excerpt: first 3 consumer orders of each of the first 3 cycles with "
              "consumer demand. `split` = integer split decode applied; outcomes = this cycle's "
              "bids (producer, units, outcome, reason); `resolved` = (completed, completed units) "
              "if the order resolved this cycle.\n")
        for r in ROUTERS:
            print(f"<details><summary>{r}</summary>\n\n```")
            for x in docs[(w, r)]["excerpt"]:
                outs = "; ".join(f"{p} {q}u {o}" + (f"({why})" if why else "")
                                 for p, q, o, why in x["outcomes"])
                print(f"c{x['cycle']} {x['agent']} {x['order']} q={x['q']} elig={x['eligible']} "
                      f"shares={x['shares']} split={x['split']} -> [{outs}] "
                      f"resolved={x['resolution']}")
            print("```\n\n</details>\n")
    print("\n### Timing (seconds per horizon-100 episode, min of 2 runs, ContractMonitor on)\n")
    print("| world | router | plain | traced | overhead | logged (warm-up) |")
    print("|---|---|---|---|---|---|")
    for w in WORLDS:
        for r in ROUTERS:
            s = docs[(w, r)]["seconds"]
            print(f"| {w} | {r} | {s['plain']:.2f} | {s['traced']:.2f} | "
                  f"{docs[(w, r)]['tracer_overhead']:.3f} | {s['logged']:.2f} |")


if __name__ == "__main__":
    main()
