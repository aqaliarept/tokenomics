"""Task 3.1 verification of an EXP-0048 dir, independent of `routing_audit`'s analysis code.

Checks: contract_failures.csv header-only; every planned (world, episode, subject, replicate)
row present once; demand and shock digests equal per world-episode; the six roster v2 members'
rows equal EXP-0047's units (random's 5 replicates included); edge.csv's cell / rule rows
recomputed from artifacts/episodes.csv (or a top-level episodes.csv) with the D2 rule (0.8 of the edge, equal-weight world means, 2 of
the 3 random-dominated cells). Prints a per-cell summary for RESULT.md.

usage: python exp0048_check.py EXP0048_DIR EXP0047_DIR
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

DOMINANT = [(12, "shocked"), (42, "calm"), (42, "shocked")]


def main() -> None:
    d48, d47 = Path(sys.argv[1]), Path(sys.argv[2])
    spec = json.loads((d48 / "spec.json").read_text())
    fails = (d48 / "contract_failures.csv").read_text().splitlines()
    print("contract_failures rows:", len(fails) - 1, "| status:", spec["status"],
          "| commit:", spec["pins"]["source"]["commit"])
    ep48 = d48 / "artifacts" / "episodes.csv"  # moved under artifacts/ (git-lfs) after the pull
    rows = list(csv.DictReader((ep48 if ep48.exists() else d48 / "episodes.csv").open()))
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

    exp47 = {(r["world"], r["episode"], r["subject"], r["replicate"]): r["units"]
             for r in csv.DictReader((d47 / "episodes.csv").open())}
    roster = [r for r in routers if r["roster"]]
    n = mism = 0
    for r in rows:
        if any(r["subject"] == m["subject"] for m in roster):
            n += 1
            k = (r["world"], r["episode"], r["subject"], r["replicate"])
            if float(exp47.get(k, "nan")) != float(r["units"]):
                mism += 1
    print(f"roster rows {n}, EXP-0047 units mismatches {mism}")

    # edge recompute
    name = {r["subject"]: r["router"] for r in routers}
    det = [r["router"] for r in roster if not r["random"]]
    val: dict = defaultdict(lambda: defaultdict(list))
    cell_of = {}
    for r in rows:
        val[(r["world"], r["episode"])][name[r["subject"]]].append(float(r["units"]))
        cell_of[r["world"]] = (int(r["size"]), r["regime"])
    edges: dict = defaultdict(lambda: defaultdict(list))
    gains: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (w, ep), per in val.items():
        best = max(fmean(per[m]) for m in det)
        edges[cell_of[w]][w].append(fmean(per["random"]) - best)
        for rname, v in per.items():
            gains[cell_of[w]][rname][w].append(fmean(v) - best)

    def wm(by_world: dict) -> float:
        return fmean(fmean(v) for v in by_world.values())

    rec = {}
    for cell in sorted(edges):
        e = wm(edges[cell])
        for rname in gains[cell]:
            g = wm(gains[cell][rname])
            rec[(cell, rname)] = g / e if e > 0 else None
    edge_csv = list(csv.DictReader((d48 / "edge.csv").open()))
    bad = 0
    for r in edge_csv:
        if r["level"] != "cell":
            continue
        mine = rec[((int(r["size"]), r["regime"]), r["router"])]
        theirs = None if r["recovered_fraction"] == "" else float(r["recovered_fraction"])
        if (mine is None) != (theirs is None) or (mine is not None and abs(mine - theirs) > 1e-9):
            bad += 1
    print("edge.csv cell rows", sum(r["level"] == "cell" for r in edge_csv), "recompute mismatches", bad)
    rule = {r["router"]: r for r in edge_csv if r["level"] == "rule"}
    rbad = 0
    for rname in rule:
        got = sum(1 for c in DOMINANT if rec.get((c, rname)) is not None
                  and rec[(c, rname)] >= 0.8 - 1e-9)
        if got != int(rule[rname]["cells_recovered"]) or (got >= 2) != (rule[rname]["recovers_most"] == "true"):
            rbad += 1
    print("rule rows", len(rule), "recompute mismatches", rbad)

    print("\ncell edges (equal-weight world mean, random minus best deterministic member):")
    for cell in sorted(edges):
        print(f"  {cell}: edge {wm(edges[cell]):.2f} over {len(edges[cell])} worlds")
    print("\nrecovered fraction per router (12-calm, 12-shocked, 42-calm, 42-shocked) | dominant cells >= 0.8:")
    for rname in [r["router"] for r in routers]:
        fr = [rec.get((c, rname)) for c in sorted(edges)]
        got = sum(1 for c in DOMINANT if rec.get((c, rname)) is not None and rec[(c, rname)] >= 0.8 - 1e-9)
        print(f"  {rname:28s} " + "  ".join("   n/a" if f is None else f"{f:6.2f}" for f in fr) + f"  | {got}")


if __name__ == "__main__":
    main()
