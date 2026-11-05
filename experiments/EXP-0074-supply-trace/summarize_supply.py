"""EXP-0074 offline summary (task 7.15): per size, the registered reading's group shares (from
reading.json), the class and b2 sub-cause breakdown, and per-producer-cycle aggregates
(capacity ratio, restoration / upgrade outcomes, fund, replenishment outcomes).
Usage: python summarize_supply.py <EXP-0074 dir>  (reads reading.json, artifacts/*.csv)."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(1 << 26)  # the st_producer_cycles JSON cell


def main(exp: Path) -> dict:
    reading = json.loads((exp / "reading.json").read_text())
    art = exp / "artifacts" if (exp / "artifacts" / "episodes.csv").exists() else exp
    size_of = {}
    for r in csv.DictReader((art / "episodes.csv").open()):
        size_of[r["world"]] = int(r["size"])
    agg: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader((art / "producer_cycles.csv").open()):
        a = agg[size_of[r["world"]]]
        a["producer_cycles"] += 1
        a["capacity_ratio_sum"] += float(r["capacity_ratio"])
        for k in ("fund_granted", "input_stock_units", "inbound_units", "restoration_placed",
                  "restoration_completed", "restoration_rejected", "upgrade_placed",
                  "upgrade_completed", "upgrade_rejected", "replenishment_units_placed",
                  "replenishment_units_delivered", "replenishment_units_rejected"):
            a[k] += float(r[k])
    producers = {s: {**{k: v for k, v in a.items() if k != "capacity_ratio_sum"},
                     "capacity_ratio_mean": a["capacity_ratio_sum"] / a["producer_cycles"]}
                 for s, a in agg.items()}
    out = {"reading": reading.get("reading"), "tie_with": reading.get("tie_with"),
           "status": reading.get("status"), "neutral_rows": reading.get("neutral_rows"),
           "compared_rows": reading.get("compared_rows"), "per_size": {}}
    for s, p in reading["per_size"].items():
        lost = p["lost_units"]
        cls = {k: v for k, v in p["classes_units"].items() if v}
        out["per_size"][s] = {
            "lost_units": lost, "filled_units": p["filled_units"],
            "group_shares": {g: round(v, 4) for g, v in p["shares"].items() if v},
            "class_shares": {k: round(v / lost, 4) for k, v in sorted(cls.items(),
                                                                      key=lambda kv: -kv[1])},
            "largest": p["largest"], "within_5_points": p["within_5_points"],
            "producers": producers.get(int(s))}
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
