"""Task 3.5 verification and summary of an EXP-0050 dir, independent of `routing_audit`'s
analysis code.

Checks: contract_failures.csv header-only; every planned (world, episode, subject, replicate)
row present once; one demand + shock digest pair per world-episode; every pinned world's
measured declarers per product equal its fanout; staged checkpoint sha256 = spec pins;
fanout_summary.csv's units / gains / counterpart columns recomputed from the episodes.
Prints per-cell tables and paired contrasts with world-clustered standard errors.

usage (from /srv/research/tokenomics):
  python3 openspec/changes/add-m3-development-audit/probes/exp0050_check.py \
      experiments/EXP-0050-routing-vs-fanout
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

csv.field_size_limit(1 << 30)
COUNTERPART = {"two_choice_spare": "random_whole", "spare_weighted_whole": "random_whole",
               "spare_rotating_whole": "round_robin"}
METRICS = ["split_share_consumer", "receiver_change_share_consumer", "herding_hhi",
           "wasted_units_consumer", "lost_units_consumer_ttl", "lost_units_consumer_no_fit",
           "lost_units_consumer_displaced", "lost_units_consumer_outage"]
CONTRASTS = [("round_robin", "random"), ("round_robin", "BEST_DET"), ("random", "BEST_DET"),
             ("two_choice_spare", "random_whole"), ("spare_weighted_whole", "random_whole"),
             ("spare_rotating_whole", "round_robin"), ("random_whole", "random"),
             ("trained_orderer", "round_robin"), ("trained_orderer", "random"),
             ("trained_orderer", "BEST_DET"), ("portable", "trained_orderer")]


def episodes(d: Path) -> list[dict[str, str]]:
    p = d / "artifacts" / "episodes.csv"
    return list(csv.DictReader((p if p.exists() else d / "episodes.csv").open()))


def wm(by_world: dict) -> float:
    return fmean(fmean(v) for v in by_world.values())


def main() -> None:
    d = Path(sys.argv[1])
    spec = json.loads((d / "spec.json").read_text())
    fails = (d / "contract_failures.csv").read_text().splitlines()
    print("contract_failures rows:", len(fails) - 1, "| status:", spec["status"],
          "| commit:", spec["pins"]["source"]["commit"], "| plan:", spec["pins"]["plan"])
    rows = episodes(d)
    routers = spec["pins"]["routers"]
    R = spec["pins"]["replicates"]
    grid = spec["pins"]["grid"]
    planned = {(w, str(ep), r["subject"], str(k)) for w, g in grid.items() for ep in g["episodes"]
               for r in routers for k in range(R if r["random"] else 1)}
    keys = [(r["world"], r["episode"], r["subject"], r["replicate"]) for r in rows]
    print("rows", len(rows), "planned", len(planned), "unique", len(set(keys)),
          "missing", len(planned - set(keys)), "extra", len(set(keys) - planned))
    dig: dict = defaultdict(set)
    for r in rows:
        dig[(r["world"], r["episode"])].add((r["demand_digest"], r["shock_digest"]))
    print("world-episodes", len(dig), "with >1 digest pair:", sum(len(v) > 1 for v in dig.values()))
    worlds = {w["id"]: w for w in spec["pins"]["worlds"]}
    bad = [w for w in worlds.values() if w["declarers_per_product"] != [w["fanout"]]]
    print("worlds", len(worlds), "with measured declarers != fanout:", len(bad))
    pins = spec["pins"]["checkpoints"]
    for role, rel in (("producer", "tokenomics/_pod_ckpts/producer_candidate.pt"),
                      ("orderer", "tokenomics/_pod_ckpts/orderer_scorer.pt")):
        p = Path(rel)
        print(f"ckpt {role} pin {pins[role][:12]}", "| staged sha = pin:",
              hashlib.sha256(p.read_bytes()).hexdigest() == pins[role] if p.exists() else "n/a")

    name = {r["subject"]: r["router"] for r in routers}
    order = [r["router"] for r in routers]
    det = [r["router"] for r in routers if r["roster"] and not r["random"]]
    val: dict = defaultdict(lambda: defaultdict(list))
    met: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fill: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key, rn = (r["world"], r["episode"]), name[r["subject"]]
        val[key][rn].append(float(r["units"]))
        if float(r["ordered_units"]) > 0:
            fill[key][rn].append(float(r["fulfilled_units"]) / float(r["ordered_units"]))
        for m in METRICS:
            met[key][rn][m].append(float(r["trace_" + m]))

    def cell(w: str) -> tuple:
        x = worlds[w]
        return (x["family"], x["size"], x["fanout"], x["regime"])

    units: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fills: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    diffs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (w, ep), per in val.items():
        c = cell(w)
        mean = {rn: fmean(v) for rn, v in per.items()}
        mean["BEST_DET"] = max(mean[m] for m in det)
        for rn in per:
            units[c][rn][w].append(mean[rn])
            fills[c][rn][w].append(fmean(fill[(w, ep)][rn]))
            for m in METRICS:
                mets[c][rn][m][w].append(fmean(met[(w, ep)][rn][m]))
        for a, b in CONTRASTS:
            diffs[c][(a, b)][w].append(mean[a] - mean[b])
        units[c]["BEST_DET"][w].append(mean["BEST_DET"])

    # fanout_summary.csv recompute
    summ = list(csv.DictReader((d / "fanout_summary.csv").open()))
    nbad = 0
    for r in summ:
        c = (r["family"], int(r["size"]), int(r["fanout"]), r["regime"])
        rn = r["router"]
        mine = {"units": wm(units[c][rn])}
        best = {w: [x - y for x, y in zip(units[c][rn][w], units[c]["BEST_DET"][w])]
                for w in units[c][rn]}
        mine["gain_vs_best_det"] = wm(best)
        for ref, col in (("random", "gain_vs_random"), ("round_robin", "gain_vs_round_robin"),
                         ("random_whole", "gain_vs_random_whole"),
                         (COUNTERPART.get(rn), "gain_vs_counterpart")):
            if ref is None:
                continue
            g = {w: [x - y for x, y in zip(units[c][rn][w], units[c][ref][w])] for w in units[c][rn]}
            mine[col] = wm(g)
        for col, v in mine.items():
            if r[col] == "" or abs(float(r[col]) - v) > 1e-9:
                nbad += 1
        if rn in COUNTERPART and (r["beats_counterpart"] == "true") != (mine["gain_vs_counterpart"] > 0):
            nbad += 1
    print("fanout_summary rows", len(summ), "recompute mismatches", nbad)

    cells = sorted(units, key=lambda c: (c[0] != "sparse_wide", c[1], c[2], c[3]))
    print("\nper cell: router | units | fill | gain vs best det | " + " | ".join(METRICS))
    for c in cells:
        print(f"-- {c[0]} {c[1]} f{c[2]} {c[3]} (worlds {len(units[c]['random'])})")
        print(f"  {'BEST_DET':22s} {wm(units[c]['BEST_DET']):8.1f}")
        for rn in order:
            g = {w: [x - y for x, y in zip(units[c][rn][w], units[c]["BEST_DET"][w])]
                 for w in units[c][rn]}
            print(f"  {rn:22s} {wm(units[c][rn]):8.1f} {wm(fills[c][rn]):6.3f} {wm(g):+8.1f} "
                  + " ".join(f"{wm(mets[c][rn][m]):7.3f}" for m in METRICS))
    print("\npaired contrasts a - b: mean [world-clustered SE, t] (worlds a>b / n)")
    for a, b in CONTRASTS:
        print(f"  {a} - {b}:")
        for c in cells:
            by = diffs[c][(a, b)]
            wmeans = [fmean(v) for v in by.values()]
            m = fmean(wmeans)
            se = stdev(wmeans) / math.sqrt(len(wmeans)) if len(wmeans) > 1 else float("nan")
            t = m / se if se and se == se and se > 0 else float("nan")
            print(f"    {c[0]:11s} {c[1]:2d} f{c[2]:<2d} {c[3]:7s} {m:+8.1f} [SE {se:6.1f}, t {t:6.2f}] "
                  f"({sum(x > 0 for x in wmeans)}/{len(wmeans)})")


if __name__ == "__main__":
    main()
