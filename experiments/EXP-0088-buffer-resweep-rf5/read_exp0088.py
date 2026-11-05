"""EXP-0088 offline read (producer finalization PF-5): buffer re-sweep on rf5, TTL 2.

k* rule (M5 tasks.md PF-5, pre-registered 2026-11-05 before data; as EXP-0077's, with rf5 = k 4 as
the reference):
- candidates k in {3, 4, 5, 6}: k = 4 is rf5 (EXP-0089's rows), k = 3 / 5 / 6 from this run;
- per orderer o in {MWWk, RR} at 12 producers (42 reported):
  eligible(k) = deaths(k) <= deaths(rf5) and run-down check failures (r_end < 0.7516 r_start) of k
  minus those of rf5 <= 5 (absolute);
  best = the eligible k with the largest mean units per episode;
  within(k) = mean_w(units_best - units_k) <= mean_w(floor_w) (paired per world; floor_w = the
  atlas v2 noise_floor, EXP-0073 ttl2 spec);
  k*_o = the smallest eligible k with within(k);
- k* = max over the two orderers of k*_o; none eligible -> no k*.
Validity: every k = 3 / 5 / 6 row's demand and shock digests equal rf5's row of the same (world,
episode, router, replicate), and the row sets are equal.
Usage: python read_exp0088.py <EXP-0088 dir>"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

csv.field_size_limit(1 << 30)
HERE = Path(__file__).resolve().parent
X_RUNDOWN = 0.7516
ORDERERS = {"MWWk": "mww_keyed", "RR": "round_robin"}
CANDIDATES = (3, 4, 5, 6)


def load(exp: Path, k: int) -> dict:
    name = f"entity_planner_buyahead_k{k}_rf5"
    d = (HERE.parent / "EXP-0089-exact-upgrade-cap" if k == 4 else exp) / name
    p = d / "artifacts" / "episodes.csv"
    with (p if p.exists() else d / "episodes.csv").open() as f:
        return {(r["world"], r["episode"], r["subject"].split("/")[1], r["replicate"]): r
                for r in csv.DictReader(f)}


def fnum(r: dict, c: str) -> float:
    v = r.get(c, "")
    return float(v) if v not in ("", "None") else 0.0


def main(exp: Path) -> dict:
    floors = {w["world"]: float(w["noise_floor"]) for w in json.loads(
        (HERE.parent / "EXP-0073-ttl-sweep" / "ttl2" / "spec.json").read_text())["pins"]["sample"]}
    data = {k: load(exp, k) for k in CANDIDATES}
    digests = {}
    for k in (3, 5, 6):
        assert set(data[k]) == set(data[4]), f"k = {k}: row set differs from rf5"
        bad = [x for x, r in data[k].items() if (r["demand_digest"], r["shock_digest"])
               != (data[4][x]["demand_digest"], data[4][x]["shock_digest"])]
        digests[k] = {"rows": len(data[k]), "mismatched": len(bad)}
    out: dict = {"digests_vs_rf5": digests, "cells": [], "k_star": {}}
    for size in ("12", "42"):
        per_o = {}
        for o, router in ORDERERS.items():
            keys = [x for x, r in data[4].items() if r["size"] == size and x[2] == router]
            worlds = sorted({x[0] for x in keys})
            stats = {}
            for k in CANDIDATES:
                rs = [data[k][x] for x in keys]
                stats[k] = {
                    "episodes": len(rs), "units_mean": mean(float(r["units"]) for r in rs),
                    "deaths": sum(r["dead"] == "true" for r in rs),
                    "rundown_fails": sum(float(r["r_end"]) < X_RUNDOWN * float(r["r_start"])
                                         for r in rs),
                    "holding_units_per_cycle_mean": mean(fnum(r, "holding_units_per_cycle")
                                                         for r in rs),
                    "investment_rejection_rate_mean": mean(fnum(r, "investment_rejection_rate")
                                                           for r in rs),
                    "upgrades_blocked": sum(fnum(r, "upgrades_blocked") for r in rs),
                    "input_late": sum(fnum(r, "st_b2_late") for r in rs),
                    "late_headroom_blocked": sum(fnum(r, "st_b2_late_headroom_blocked") for r in rs),
                    "input_died_upstream": sum(fnum(r, "st_b2_lost_ttl")
                                               + fnum(r, "st_b2_lost_ttl_outage") for r in rs),
                    "contention": sum(fnum(r, "st_b4_contention") for r in rs),
                    "lost": sum(fnum(r, "st_lost_consumer_total") for r in rs)}
            ref = stats[4]
            eligible = [k for k in CANDIDATES if stats[k]["deaths"] <= ref["deaths"]
                        and stats[k]["rundown_fails"] - ref["rundown_fails"] <= 5]
            mf = mean(floors[w] for w in worlds)

            def gap(a: int, b: int) -> float:  # mean_w(units_a - units_b), paired per world
                return mean(mean(float(data[a][x]["units"]) - float(data[b][x]["units"])
                                 for x in keys if x[0] == w) for w in worlds)

            best = max(eligible, key=lambda k: stats[k]["units_mean"]) if eligible else None
            within = {k: gap(best, k) <= mf for k in eligible} if best else {}
            kstar = next((k for k in sorted(eligible) if within[k]), None)
            per_o[o] = kstar
            out["cells"].append({"size": int(size), "orderer": o, "worlds": len(worlds),
                                 "mean_floor": mf, "stats": {str(k): v for k, v in stats.items()},
                                 "vs_rf5": {k: gap(k, 4) for k in CANDIDATES if k != 4},
                                 "eligible": eligible, "best": best,
                                 "best_minus_k": {k: gap(best, k) for k in eligible} if best else {},
                                 "within_floor_of_best": within, "k_star": kstar})
        ks = list(per_o.values())
        out["k_star"][size] = {"per_orderer": per_o,
                               "k_star": max(ks) if all(k is not None for k in ks) else None}
    out["deciding_k_star"] = out["k_star"]["12"]["k_star"]
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1, default=str))
