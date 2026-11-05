"""EXP-0073 offline read (add-m4-orderer-redesign task 7.13): TTL 2 neutrality against EXP-0057,
then the pre-registered timing / scarcity / mixed reading per stack (12 producers decide, 42
reported), plus expired units, run-down and deaths per TTL.

Operational rules, fixed before the pull (2026-11-05):
- diff_w = mean over a world's paired (episode, replicate) rows of units(T) - units(2);
- floor_w = the atlas v2 world's noise_floor (pinned in each run's spec.json);
- "rise beyond the floor": mean_w(diff_w) > mean_w(floor_w) (EXP-0067's mean clause);
- "within the floor": |mean_w(diff_w)| <= mean_w(floor_w);
- "no worse run-down / deaths": deaths(T) <= deaths(2) and the count of episodes failing the
  gate run-down check r_end >= X * r_start (X = 0.7516, side_effect_params_v2) not higher;
- timing = rise and no worse; scarcity = within; mixed = anything else.
Expired units = consumer units lost to TTL (trace_lost_units_consumer_ttl + _ttl_outage).
Completion latency: not in the driver's rows (agent-history EMA only); not measured.
Usage: python read_ttl_sweep.py <EXP-0073 dir> <EXP-0057 episodes.csv>
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

X_RUNDOWN = 0.7516
TTLS = (2, 3, 4)
STACKS = ("MWWk", "RR")
KEY = ("world", "episode", "replicate", "arm")
# columns that legitimately differ between drivers / runs (none expected beyond these)
SKIP = {"subject"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def episodes_csv(run: Path) -> Path:
    for p in (run / "episodes.csv", run / "artifacts" / "episodes.csv"):
        if p.exists():
            return p
    raise FileNotFoundError(run)


def neutrality(ttl2: list[dict[str, str]], ref: list[dict[str, str]]) -> dict:
    index = {tuple(r[k] for k in KEY): r for r in ref if r.get("arm") in STACKS}
    diffs: dict[str, int] = defaultdict(int)
    missing = 0
    shared = None
    for r in ttl2:
        k = tuple(r[c] for c in KEY)
        o = index.get(k)
        if o is None:
            missing += 1
            continue
        cols = sorted((set(r) & set(o)) - SKIP)
        shared = len(cols)
        for c in cols:
            if r[c] != o[c]:
                diffs[c] += 1
    return {"rows": len(ttl2), "missing_in_exp0057": missing, "shared_columns": shared,
            "differing_columns": dict(diffs), "neutral": missing == 0 and not diffs}


def fnum(r: dict[str, str], c: str) -> float:
    v = r.get(c, "")
    return float(v) if v not in ("", "None") else 0.0


def rundown_fail(r: dict[str, str]) -> bool:
    return float(r["r_end"]) < X_RUNDOWN * float(r["r_start"])


def expired(r: dict[str, str]) -> float:
    return fnum(r, "trace_lost_units_consumer_ttl") + fnum(r, "trace_lost_units_consumer_ttl_outage")


def main(exp: Path, exp0057: Path) -> dict:
    runs = {t: rows(episodes_csv(exp / f"ttl{t}")) for t in TTLS}
    floors = {}
    for t in TTLS:
        spec = json.loads((exp / f"ttl{t}" / "spec.json").read_text())
        assert spec["pins"]["order_ttl"] == t and spec["pins"]["profile"] == "ttl_sweep", t
        for w in spec["pins"]["sample"]:
            floors.setdefault(w["world"], float(w["noise_floor"]))
    out: dict = {"neutrality_ttl2_vs_exp0057": neutrality(runs[2], rows(exp0057))}
    by = {t: {tuple(r[c] for c in KEY): r for r in runs[t]} for t in TTLS}
    assert all(set(by[t]) == set(by[2]) for t in TTLS), "row sets differ across TTLs"
    for t in TTLS:  # demand and shock schedules must be identical across TTLs
        for k, r in by[t].items():
            assert r["demand_digest"] == by[2][k]["demand_digest"], (t, k)
            assert r["shock_digest"] == by[2][k]["shock_digest"], (t, k)
    sizes = sorted({r["size"] for r in runs[2]}, key=int)
    table: list[dict] = []
    readings: dict = {}
    for size in sizes:
        for arm in STACKS:
            keys = [k for k, r in by[2].items() if r["size"] == size and k[3] == arm]
            worlds = sorted({k[0] for k in keys})
            per_ttl = {}
            for t in TTLS:
                rs = [by[t][k] for k in keys]
                per_ttl[t] = {
                    "units_mean_per_episode": mean(float(r["units"]) for r in rs),
                    "expired_units": sum(expired(r) for r in rs),
                    "expired_units_inputs_short": sum(fnum(r, "trace_ttl_lost_units_consumer_inputs_short") for r in rs),
                    "deaths": sum(r["dead"] == "true" for r in rs),
                    "rundown_fails": sum(rundown_fail(r) for r in rs),
                    "r_ratio_mean": mean(float(r["r_end"]) / float(r["r_start"]) for r in rs),
                    "pooled_fill": sum(fnum(r, "fulfilled_units") for r in rs)
                    / sum(fnum(r, "ordered_units") for r in rs),
                    "overflow_orders": sum(fnum(r, "trace_overflow_orders") for r in rs),
                    "episodes": len(rs)}
            cell = {"size": int(size), "arm": arm, "worlds": len(worlds), "per_ttl": per_ttl, "vs_ttl2": {}}
            for t in (3, 4):
                diff_w, beyond, below = [], 0, 0
                for w in worlds:
                    ks = [k for k in keys if k[0] == w]
                    d = mean(float(by[t][k]["units"]) - float(by[2][k]["units"]) for k in ks)
                    diff_w.append(d)
                    beyond += d > floors[w]
                    below += d < -floors[w]
                md, mf = mean(diff_w), mean(floors[w] for w in worlds)
                rise, within = md > mf, abs(md) <= mf
                no_worse = (per_ttl[t]["deaths"] <= per_ttl[2]["deaths"]
                            and per_ttl[t]["rundown_fails"] <= per_ttl[2]["rundown_fails"])
                reading = "timing" if rise and no_worse else "scarcity" if within else "mixed"
                cell["vs_ttl2"][t] = {"mean_diff": md, "mean_floor": mf, "worlds_beyond": beyond,
                                      "worlds_below": below, "rise": rise, "within": within,
                                      "no_worse": no_worse, "reading": reading}
            table.append(cell)
            readings[f"{size}/{arm}"] = {t: cell["vs_ttl2"][t]["reading"] for t in (3, 4)}
    out["cells"] = table
    out["readings"] = readings
    out["deciding"] = {k: v for k, v in readings.items() if k.startswith("12/")}
    return out


if __name__ == "__main__":
    res = main(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(res, indent=1, default=str))
