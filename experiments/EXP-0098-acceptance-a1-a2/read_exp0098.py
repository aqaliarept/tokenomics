"""EXP-0098 offline read (add-producer-acceptance-policy task 3.1): A1 (`ttl_feasible`) and A2
(`ttl_inputs`) acceptance vs the k5_rf5 default (EXP-0088's k5_rf5 rows), TTL 2.

Registered reading (tasks.md 3.1, pre-registered 2026-11-06 before data; 12 producers decide, 42
reported), per arm A, as EXP-0084's: per orderer diff_w = mean paired (episode, replicate)
units(A) - units(default) per world; floor_w = the atlas v2 noise_floor (EXP-0073 ttl2 spec);
exceeds = mean_w(diff_w) > mean_w(floor_w); below = mean_w(diff_w) < -mean_w(floor_w); guard =
deaths(A) <= deaths(default) and run-down failures of A minus the default's <= 5; A_helps if exceeds
and guard for BOTH orderers; harmful if below or extra deaths for EITHER; no_gain otherwise.
Validity: every row's digests equal the default row's, and the row sets are equal.
Reported only: fund gaming (declines below the cut-off; fund share moved vs the default), pending
claims completed late vs failed at TTL, units lost to e_declined vs b2_late, deaths, stock, run-down.
Usage: python read_exp0098.py <EXP-0098 dir>"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

csv.field_size_limit(1 << 30)
HERE = Path(__file__).resolve().parent
X_RUNDOWN = 0.7516
PROD = "entity_planner_buyahead_k5_rf5"
ARMS = {"A1": "ttl_feasible", "A2": "ttl_inputs"}
ORDERERS = {"MWWk": "mww_keyed", "RR": "round_robin"}
PC_FUND = 5  # supply_trace PC_FIELDS index of fund_granted


def load(path: Path) -> dict:
    with path.open() as f:
        return {(r["world"], r["episode"], r["subject"].split("/")[1], r["replicate"]): r
                for r in csv.DictReader(f)}


def fnum(r: dict, c: str) -> float:
    v = r.get(c, "")
    return float(v) if v not in ("", "None") else 0.0


_FS: dict[int, dict[str, float]] = {}


def fund_shares(r: dict) -> dict[str, float]:
    if id(r) not in _FS:
        _FS[id(r)] = _fund_shares(r)
    return _FS[id(r)]


def _fund_shares(r: dict) -> dict[str, float]:
    tot: dict[str, float] = defaultdict(float)
    for q in json.loads(r.get("st_producer_cycles") or "[]"):
        tot[q[1]] += q[PC_FUND] or 0.0
    s = sum(tot.values()) or 1.0
    return {p: v / s for p, v in tot.items()}


def main(exp: Path) -> dict:
    floors = {w["world"]: float(w["noise_floor"]) for w in json.loads(
        (HERE.parent / "EXP-0073-ttl-sweep" / "ttl2" / "spec.json").read_text())["pins"]["sample"]}
    base = load(HERE.parent / "EXP-0088-buffer-resweep-rf5" / PROD / "artifacts" / "episodes.csv")
    data = {}
    for a, name in list(ARMS.items()):
        d = exp / name
        if sum(1 for _ in (d / "contract_failures.csv").open()) > 1:  # an invalid run: no reading
            print(f"[read] {a} ({name}) is invalid (contract failures); skipped", file=sys.stderr)
            del ARMS[a]
            continue
        p = d / "artifacts" / "episodes.csv"
        data[a] = load(p if p.exists() else d / "episodes.csv")
        assert set(data[a]) == set(base), f"{a}: row set differs from the default"
        assert all((data[a][k]["demand_digest"], data[a][k]["shock_digest"])
                   == (base[k]["demand_digest"], base[k]["shock_digest"]) for k in base), a
    out: dict = {"cells": [], "readings": {}}
    for size in ("12", "42"):
        for o, router in ORDERERS.items():
            keys = [k for k, r in base.items() if r["size"] == size and k[2] == router]
            worlds = sorted({k[0] for k in keys})
            mf = mean(floors[w] for w in worlds)

            def stats(rows: dict, arm: bool) -> dict:
                rs = [rows[k] for k in keys]
                st = {"units_mean": mean(float(r["units"]) for r in rs),
                      "deaths": sum(r["dead"] == "true" for r in rs),
                      "rundown_fails": sum(float(r["r_end"]) < X_RUNDOWN * float(r["r_start"])
                                           for r in rs),
                      "holding_mean": mean(fnum(r, "holding_units_per_cycle") for r in rs),
                      "lost_b2_late": sum(fnum(r, "st_b2_late") for r in rs),
                      "lost_e_declined": sum(fnum(r, "st_e_declined") for r in rs),
                      "lost_total": sum(fnum(r, "st_lost_consumer_total") for r in rs)}
                if arm:
                    acc: dict[str, int] = defaultdict(int)
                    for r in rs:
                        for kk, v in json.loads(r.get("acceptance_stats") or "{}").items():
                            acc[kk] += v
                    st["acceptance"] = dict(acc)
                return st

            cell = {"size": int(size), "orderer": o, "worlds": len(worlds), "mean_floor": mf,
                    "default": stats(base, False)}
            for a in ARMS:
                rows = data[a]
                diff_w = [mean(float(rows[k]["units"]) - float(base[k]["units"])
                               for k in keys if k[0] == w) for w in worlds]
                st = stats(rows, True)
                moved = mean(0.5 * sum(abs(fund_shares(rows[k]).get(p, 0) - fund_shares(base[k]).get(p, 0))
                                       for p in set(fund_shares(rows[k])) | set(fund_shares(base[k])))
                             for k in keys)
                extra = st["deaths"] > cell["default"]["deaths"]
                cell[a] = {**st, "mean_diff": mean(diff_w), "exceeds": mean(diff_w) > mf,
                           "below": mean(diff_w) < -mf, "extra_deaths": extra,
                           "rundown_delta": st["rundown_fails"] - cell["default"]["rundown_fails"],
                           "worlds_beyond": sum(d > floors[w] for d, w in zip(diff_w, worlds)),
                           "worlds_below": sum(d < -floors[w] for d, w in zip(diff_w, worlds)),
                           "fund_share_moved": moved}
                cell[a]["guard"] = not extra and cell[a]["rundown_delta"] <= 5
            out["cells"].append(cell)
        for a in ARMS:
            v = [c[a] for c in out["cells"] if c["size"] == int(size)]
            out["readings"][f"{size}/{a}"] = (
                f"{a}_helps" if all(x["exceeds"] and x["guard"] for x in v)
                else "harmful" if any(x["below"] or x["extra_deaths"] for x in v) else "no_gain")
    out["deciding"] = {a: out["readings"][f"12/{a}"] for a in ARMS}
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1, default=str))
