"""EXP-0077 offline read (task 7.21): buffer sweep of `entity_planner_buyahead`, TTL 2.

k* rule (tasks.md 7.21, registered; operational choices fixed before the pull, 2026-11-05):
- candidates k in {2, 3, 4}: k = 2 and `entity_planner` (EP) from EXP-0076, k = 3 / 4 from this
  run; k = 1 (EXP-0076) is reported for context only;
- per orderer o in {MWWk, RR} at 12 producers (42 reported):
  eligible(k) = deaths(k) <= deaths(EP) and run-down check failures (r_end < 0.7516 r_start) of k
  minus those of EP <= 5 (absolute, per orderer: 86 MWWk / 43 RR episodes);
  best = the eligible k with the largest mean units per episode;
  within(k) = mean_w(units_best - units_k) <= mean_w(floor_w) (paired per world; floor_w = the
  atlas v2 noise_floor, EXP-0073 ttl2 spec);
  k*_o = the smallest eligible k with within(k);
- k* = max over the two orderers of k*_o (it then holds for both); none eligible -> no k*.
Validity: every k = 3 / 4 row's demand and shock digests equal the EXP-0076 EP row's of the same
(world, episode, router, replicate), and the row sets are equal.
Usage: python read_exp0077.py <EXP-0077 dir>"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

csv.field_size_limit(1 << 26)
HERE = Path(__file__).resolve().parent
X_RUNDOWN = 0.7516
EP = "entity_planner"
K = {1: "entity_planner_buyahead", 2: "entity_planner_buyahead_k2",
     3: "entity_planner_buyahead_k3", 4: "entity_planner_buyahead_k4"}
CANDIDATES = (2, 3, 4)
ORDERERS = {"MWWk": "mww_keyed", "RR": "round_robin"}


def load(producer: str, exp: Path) -> dict[tuple, dict[str, str]]:
    base = exp if producer in (K[3], K[4]) else HERE.parent / "EXP-0076-buyahead"
    d = base / producer
    p = d / "artifacts" / "episodes.csv"
    with (p if p.exists() else d / "episodes.csv").open() as f:
        return {(r["world"], r["episode"], r["subject"].split("/")[1], r["replicate"]): r
                for r in csv.DictReader(f)}


def fnum(r: dict[str, str], c: str) -> float:
    v = r.get(c, "")
    return float(v) if v not in ("", "None") else 0.0


def main(exp: Path) -> dict:
    floors = {w["world"]: float(w["noise_floor"]) for w in json.loads(
        (HERE.parent / "EXP-0073-ttl-sweep" / "ttl2" / "spec.json").read_text())["pins"]["sample"]}
    data = {EP: load(EP, exp), **{k: load(p, exp) for k, p in K.items()}}
    digests = {}
    for k in (3, 4):
        assert set(data[k]) == set(data[EP]), f"k = {k}: row set differs from EXP-0076's"
        bad = [key for key, r in data[k].items()
               if (r["demand_digest"], r["shock_digest"])
               != (data[EP][key]["demand_digest"], data[EP][key]["shock_digest"])]
        digests[k] = {"rows": len(data[k]), "mismatched": len(bad)}
    out: dict = {"digests_vs_exp0076": digests, "cells": [], "k_star": {}}
    for size in ("12", "42"):
        per_o = {}
        for o, router in ORDERERS.items():
            keys = [k for k, r in data[EP].items() if r["size"] == size and k[2] == router]
            worlds = sorted({k[0] for k in keys})
            stats = {}
            for k in (EP, 1, 2, 3, 4):
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
                    "input_late": sum(fnum(r, "st_b2_late") for r in rs),
                    "input_died_upstream": sum(fnum(r, "st_b2_lost_ttl")
                                               + fnum(r, "st_b2_lost_ttl_outage") for r in rs),
                    "contention": sum(fnum(r, "st_b4_contention") for r in rs)}
            ep = stats[EP]
            eligible = [k for k in CANDIDATES if stats[k]["deaths"] <= ep["deaths"]
                        and stats[k]["rundown_fails"] - ep["rundown_fails"] <= 5]
            mf = mean(floors[w] for w in worlds)

            def gap(a: int, b: int) -> float:  # mean_w(units_a - units_b), paired per world
                return mean(mean(float(data[a][x]["units"]) - float(data[b][x]["units"])
                                 for x in keys if x[0] == w) for w in worlds)

            best = max(eligible, key=lambda k: stats[k]["units_mean"]) if eligible else None
            within = {k: gap(best, k) <= mf for k in eligible} if best else {}
            kstar = next((k for k in sorted(eligible) if within[k]), None)
            vs_ep = {k: gap(k, EP) for k in (1, 2, 3, 4)}
            per_o[o] = kstar
            out["cells"].append({"size": int(size), "orderer": o, "worlds": len(worlds),
                                 "mean_floor": mf, "stats": {str(k): v for k, v in stats.items()},
                                 "vs_entity_planner": vs_ep, "eligible": eligible, "best": best,
                                 "best_minus_k": {k: gap(best, k) for k in eligible} if best else {},
                                 "within_floor_of_best": within, "k_star": kstar})
        ks = list(per_o.values())
        out["k_star"][size] = {"per_orderer": per_o,
                               "k_star": max(ks) if all(k is not None for k in ks) else None}
    out["deciding_k_star"] = out["k_star"]["12"]["k_star"]
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1, default=str))
