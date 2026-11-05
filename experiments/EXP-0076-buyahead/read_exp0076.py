"""EXP-0076 offline read (task 7.19): buy-ahead vs entity_planner at TTL 2.

Registered reading (tasks.md 7.19; 12 producers decide, 42 reported), operational rules fixed
before the pull (2026-11-05):
- per orderer o in {MWWk, RR} and producer B (k = 1 deciding; k = 2 reported):
  diff_w = mean over world w's paired (episode, replicate) rows of units(B) - units(entity_planner);
  floor_w = the atlas v2 noise_floor (EXP-0073 ttl2 spec pins);
  exceeds = mean_w(diff_w) > mean_w(floor_w); within = |mean_w(diff_w)| <= mean_w(floor_w);
  below = mean_w(diff_w) < -mean_w(floor_w);
  guard = deaths(B) <= deaths(EP) and run-down check failures (r_end < 0.7516 r_start) of B
  minus those of EP <= 5 (absolute, per orderer: 86 MWWk / 43 RR episodes).
- buyahead_helps if exceeds and guard for BOTH orderers; harmful if below or extra deaths for
  EITHER orderer; no_gain otherwise. Per-orderer cells are reported.
Validity: the entity_planner control rows must equal EXP-0057 / EXP-0073 ttl2 (driver
neutrality, 213 / 213 each) and EXP-0075 ttl2 (checked here on every shared column).
Usage: python read_exp0076.py <EXP-0076 dir>"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

csv.field_size_limit(1 << 26)
HERE = Path(__file__).resolve().parent
X_RUNDOWN = 0.7516
EP, K1, K2 = "entity_planner", "entity_planner_buyahead", "entity_planner_buyahead_k2"
ORDERERS = {"MWWk": "mww_keyed", "RR": "round_robin"}
TABLE = {
    "input_late": ("b2_late",), "input_died_upstream": ("b2_lost_ttl", "b2_lost_ttl_outage"),
    "input_outbid_or_no_fit_upstream": ("b2_lost_displaced", "b2_lost_no_fit"),
    "input_not_ordered": ("b1_not_ordered",), "contention": ("b4_contention",),
    "capacity": ("a_held", "a_degraded", "a_max", "a_dropped"), "outbid": ("c_displaced",),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def episodes(exp: Path, producer: str) -> list[dict[str, str]]:
    d = exp / producer
    p = d / "artifacts" / "episodes.csv"
    return rows(p if p.exists() else d / "episodes.csv")


def fnum(r: dict[str, str], c: str) -> float:
    v = r.get(c, "")
    return float(v) if v not in ("", "None") else 0.0


def key(r: dict[str, str]) -> tuple:
    return (r["world"], r["episode"], r["subject"].split("/")[1], r["replicate"])


def control_vs_exp0075(exp: Path) -> dict:
    ref = {key(r): r for r in rows(HERE.parent / "EXP-0075-ttl-sweep-fixed" / "ttl2" / "artifacts"
                                     / "episodes.csv")}
    diffs: dict[str, int] = defaultdict(int)
    n = missing = 0
    for r in episodes(exp, EP):
        o = ref.get(key(r))
        if o is None:
            missing += 1
            continue
        n += 1
        for c in set(r) & set(o):
            if r[c] != o[c]:
                diffs[c] += 1
    return {"rows": n, "missing": missing, "differing_columns": dict(diffs),
            "equal": missing == 0 and not diffs}


def main(exp: Path) -> dict:
    floors = {w["world"]: float(w["noise_floor"]) for w in json.loads(
        (HERE.parent / "EXP-0073-ttl-sweep" / "ttl2" / "spec.json").read_text())["pins"]["sample"]}
    data = {p: {key(r): r for r in episodes(exp, p)} for p in (EP, K1, K2)}
    assert set(data[K1]) == set(data[EP]) == set(data[K2]), "row sets differ"
    for p in (K1, K2):
        for k, r in data[p].items():
            assert r["demand_digest"] == data[EP][k]["demand_digest"], (p, k)
            assert r["shock_digest"] == data[EP][k]["shock_digest"], (p, k)
    out: dict = {"control_vs_exp0075": control_vs_exp0075(exp),
                 "driver": {p: {k: json.loads((exp / p / "reading.json").read_text()).get(k)
                                for k in ("status", "neutral_rows", "compared_rows",
                                          "neutral_rows_exp0073", "compared_rows_exp0073")}
                            for p in (EP, K1, K2)},
                 "cells": [], "readings": {}}
    for size in ("12", "42"):
        for o, router in ORDERERS.items():
            keys = [k for k, r in data[EP].items() if r["size"] == size and k[2] == router]
            worlds = sorted({k[0] for k in keys})
            per: dict = {}
            for p in (EP, K1, K2):
                rs = [data[p][k] for k in keys]
                cls = {g: sum(fnum(r, f"st_{c}") for r in rs for c in cs) for g, cs in TABLE.items()}
                per[p] = {
                    "episodes": len(rs), "units_mean": mean(float(r["units"]) for r in rs),
                    "deaths": sum(r["dead"] == "true" for r in rs),
                    "rundown_fails": sum(float(r["r_end"]) < X_RUNDOWN * float(r["r_start"])
                                         for r in rs),
                    "r_ratio_mean": mean(float(r["r_end"]) / float(r["r_start"]) for r in rs),
                    "holding_units_per_cycle_mean": mean(fnum(r, "holding_units_per_cycle")
                                                         for r in rs),
                    "investment_rejection_rate_mean": mean(fnum(r, "investment_rejection_rate")
                                                           for r in rs),
                    "lost": sum(fnum(r, "st_lost_consumer_total") for r in rs), **cls}
            cell = {"size": int(size), "orderer": o, "worlds": len(worlds), "per": per, "vs": {}}
            for p in (K1, K2):
                diff_w = [mean(float(data[p][k]["units"]) - float(data[EP][k]["units"])
                               for k in keys if k[0] == w) for w in worlds]
                md, mf = mean(diff_w), mean(floors[w] for w in worlds)
                extra_deaths = per[p]["deaths"] > per[EP]["deaths"]
                guard = not extra_deaths and per[p]["rundown_fails"] - per[EP]["rundown_fails"] <= 5
                cell["vs"][p] = {
                    "mean_diff": md, "mean_floor": mf,
                    "worlds_beyond": sum(d > floors[w] for d, w in zip(diff_w, worlds)),
                    "worlds_below": sum(d < -floors[w] for d, w in zip(diff_w, worlds)),
                    "exceeds": md > mf, "within": abs(md) <= mf, "below": md < -mf,
                    "extra_deaths": extra_deaths, "guard": guard,
                    "rundown_fails_delta": per[p]["rundown_fails"] - per[EP]["rundown_fails"]}
            out["cells"].append(cell)
    for size in (12, 42):
        for p in (K1, K2):
            vs = [c["vs"][p] for c in out["cells"] if c["size"] == size]
            if all(v["exceeds"] and v["guard"] for v in vs):
                rd = "buyahead_helps"
            elif any(v["below"] or v["extra_deaths"] for v in vs):
                rd = "harmful"
            else:
                rd = "no_gain"
            out["readings"][f"{size}/{p}"] = rd
    out["deciding"] = out["readings"]["12/" + K1]
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1, default=str))
