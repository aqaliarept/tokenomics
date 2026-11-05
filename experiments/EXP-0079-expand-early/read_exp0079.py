"""EXP-0079 offline read (M5 task 1.2): "expand inventory earlier" alone (rf3) on top of buy-ahead k = 4, TTL 2.

Registered reading (M5 tasks.md 1.2, pre-registered 2026-11-05 before data; 12 producers decide,
42 reported), as EXP-0076's against k = 4:
- per orderer o in {MWWk, RR} and producer P = rf3:
  diff_w = mean over world w's paired (episode, replicate) rows of units(P) - units(k4);
  floor_w = the atlas v2 noise_floor (EXP-0073 ttl2 spec pins);
  exceeds = mean_w(diff_w) > mean_w(floor_w); below = mean_w(diff_w) < -mean_w(floor_w);
  guard = deaths(P) <= deaths(k4) and run-down check failures (r_end < 0.7516 r_start) of P minus
  those of k4 <= 5 (absolute, per orderer);
- rf_helps if exceeds and guard for BOTH orderers; harmful if below or extra deaths for EITHER;
  no_gain otherwise.
Validity: every row's demand and shock digests equal EXP-0077's k = 4 row of the same (world,
episode, router, replicate), and the row sets are equal.
Usage: python read_exp0079.py <EXP-0079 dir>"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

csv.field_size_limit(1 << 30)
HERE = Path(__file__).resolve().parent
X_RUNDOWN = 0.7516
K4 = "entity_planner_buyahead_k4"
P = {"rf3": "entity_planner_buyahead_k4_rf3"}
ORDERERS = {"MWWk": "mww_keyed", "RR": "round_robin"}


def load(exp: Path, producer: str) -> dict[tuple, dict[str, str]]:
    base = HERE.parent / "EXP-0077-buffer-sweep" if producer == K4 else exp
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
    data = {"k4": load(exp, K4), **{n: load(exp, p) for n, p in P.items()}}
    digests = {}
    for n in P:
        assert set(data[n]) == set(data["k4"]), f"{n}: row set differs from EXP-0077 k = 4"
        bad = [k for k, r in data[n].items() if (r["demand_digest"], r["shock_digest"])
               != (data["k4"][k]["demand_digest"], data["k4"][k]["shock_digest"])]
        digests[n] = {"rows": len(data[n]), "mismatched": len(bad)}
    out: dict = {"digests_vs_exp0077_k4": digests, "cells": [], "readings": {}}
    for size in ("12", "42"):
        for o, router in ORDERERS.items():
            keys = [k for k, r in data["k4"].items() if r["size"] == size and k[2] == router]
            worlds = sorted({k[0] for k in keys})
            mf = mean(floors[w] for w in worlds)
            stats = {}
            for n in ("k4", *P):
                rs = [data[n][k] for k in keys]
                stats[n] = {
                    "episodes": len(rs), "units_mean": mean(float(r["units"]) for r in rs),
                    "deaths": sum(r["dead"] == "true" for r in rs),
                    "rundown_fails": sum(float(r["r_end"]) < X_RUNDOWN * float(r["r_start"])
                                         for r in rs),
                    "holding_units_per_cycle_mean": mean(fnum(r, "holding_units_per_cycle")
                                                         for r in rs),
                    "investment_rejection_rate_mean": mean(fnum(r, "investment_rejection_rate")
                                                           for r in rs),
                    "input_late": sum(fnum(r, "st_b2_late") for r in rs),
                    "late_headroom_blocked": (sum(fnum(r, "st_b2_late_headroom_blocked")
                                                  for r in rs) if n != "k4" else None),
                    "input_died_upstream": sum(fnum(r, "st_b2_lost_ttl")
                                               + fnum(r, "st_b2_lost_ttl_outage") for r in rs),
                    "contention": sum(fnum(r, "st_b4_contention") for r in rs),
                    "capacity": sum(fnum(r, f"st_{c}") for r in rs
                                    for c in ("a_held", "a_degraded", "a_max", "a_dropped")),
                    "lost": sum(fnum(r, "st_lost_consumer_total") for r in rs)}
            vs = {}
            for n in P:
                diff_w = [mean(float(data[n][k]["units"]) - float(data["k4"][k]["units"])
                               for k in keys if k[0] == w) for w in worlds]
                md = mean(diff_w)
                extra = stats[n]["deaths"] > stats["k4"]["deaths"]
                vs[n] = {"mean_diff": md, "mean_floor": mf,
                         "worlds_beyond": sum(d > floors[w] for d, w in zip(diff_w, worlds)),
                         "worlds_below": sum(d < -floors[w] for d, w in zip(diff_w, worlds)),
                         "exceeds": md > mf, "below": md < -mf, "extra_deaths": extra,
                         "rundown_delta": stats[n]["rundown_fails"] - stats["k4"]["rundown_fails"],
                         "guard": not extra and stats[n]["rundown_fails"]
                         - stats["k4"]["rundown_fails"] <= 5}
            out["cells"].append({"size": int(size), "orderer": o, "worlds": len(worlds),
                                 "stats": stats, "vs_k4": vs})
        for n in P:
            v = [c["vs_k4"][n] for c in out["cells"] if c["size"] == int(size)]
            out["readings"][f"{size}/{n}"] = (
                "rf_helps" if all(x["exceeds"] and x["guard"] for x in v)
                else "harmful" if any(x["below"] or x["extra_deaths"] for x in v)
                else "no_gain")
    out["deciding"] = out["readings"]["12/rf3"]
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1, default=str))
