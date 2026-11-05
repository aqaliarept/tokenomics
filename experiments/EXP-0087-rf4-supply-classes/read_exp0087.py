"""EXP-0087 offline read (producer finalization step 1): rf4's consumer-loss classes on EXP-0057's
worlds (TTL 2, MWWk R = 2 and RR), from EXP-0084's supply-trace rows (the identical
`supply_trace --producer entity_planner_buyahead_k4_rf4 --order-ttl 2` run; no pod), against
EXP-0074 (entity_planner, MWWk), EXP-0077 (k4), EXP-0079 (rf3). Usage: python read_exp0087.py"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
G = {"a_capacity": ["a_held", "a_degraded", "a_max", "a_dropped"], "b1_not_ordered": ["b1_not_ordered"],
     "b2_not_delivered": ["b2_late", "b2_short", "b2_unfunded", "b2_open", "b2_lost_outage",
                          "b2_lost_no_fit", "b2_lost_displaced", "b2_lost_dropped", "b2_lost_ttl",
                          "b2_lost_ttl_outage"],
     "b3_upstream_unable": ["b3_upstream_unable"], "b4_contention": ["b4_contention"],
     "c_outbid": ["c_displaced"], "d_outage": ["d_outage", "d_ttl_outage"], "never_bid": ["never_bid"]}
SUB = G["b2_not_delivered"] + G["a_capacity"]
RUNS = {"entity_planner (EXP-0074)": "EXP-0074-supply-trace/supply_trace.csv",
        "k4 (EXP-0077)": "EXP-0077-buffer-sweep/entity_planner_buyahead_k4/supply_trace.csv",
        "rf3 (EXP-0079)": "EXP-0079-expand-early/entity_planner_buyahead_k4_rf3/supply_trace.csv",
        "rf4 (EXP-0084)": "EXP-0084-sustainable-upgrade/entity_planner_buyahead_k4_rf4/supply_trace.csv"}


def main() -> dict:
    out = {"sources": {}, "cells": []}
    for name, rel in RUNS.items():
        p = EXP / rel
        out["sources"][name] = {"file": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        agg = defaultdict(lambda: defaultdict(float))
        n = defaultdict(int)
        for r in csv.DictReader(p.open()):
            k = (int(r["size"]), r.get("arm", "MWWk"))
            n[k] += 1
            for c in {"units", "lost", *SUB, *(c for cs in G.values() for c in cs)}:
                agg[k][c] += float(r.get(c) or 0)
        for k in sorted(agg):
            a = agg[k]
            groups = {g: sum(a[c] for c in cs) for g, cs in G.items()}
            out["cells"].append({
                "run": name, "size": k[0], "arm": k[1], "episodes": n[k],
                "units_per_episode": a["units"] / n[k], "lost_per_episode": a["lost"] / n[k],
                "group_share": {g: v / a["lost"] for g, v in groups.items()},
                "group_per_episode": {g: v / n[k] for g, v in groups.items()},
                "sub_per_episode": {c: a[c] / n[k] for c in SUB}})
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
