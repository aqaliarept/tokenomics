"""EXP-0115 offline read (add-m6-joint-mpg-gnn task 4.1): why A1 acceptance is harmful.

The rule is pre-registered in openspec/changes/add-m6-joint-mpg-gnn/notes/exp0115-prereg.md (pinned by
SHA-256 below, frozen before data). Arms: auction (neutrality vs EXP-0088), ttl_feasible (A1,
reproduction vs EXP-0098), ttl_feasible_life (A1-life), ttl_feasible_new_first (A1-new-first); all
diagnostics, never candidates. Usage: python read_exp0115.py <EXP-0115 dir>
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

csv.field_size_limit(1 << 30)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTE = REPO / "openspec/changes/add-m6-joint-mpg-gnn/notes/exp0115-prereg.md"
NOTE_SHA = "7a8ecbb4be42181bbc30a3b419e8f5671b865723d7ca5f47b75f010c53c8629c"
X_RUNDOWN = 0.7516
PROD = "entity_planner_buyahead_k5_rf5"
ARMS = {"auction": "auction", "A1": "ttl_feasible", "A1-life": "ttl_feasible_life",
        "A1-new-first": "ttl_feasible_new_first"}
ORDERERS = {"MWWk": "mww_keyed", "RR": "round_robin"}
PC = {"inbound": 4, "fund": 5, "delivered": 13}  # supply_trace PC_FIELDS indices
IGNORED = {"acceptance_stats"}  # the policy's own counters: outside neutrality / reproduction


def load(path: Path) -> dict:
    with path.open() as f:
        return {(r["world"], r["episode"], r["subject"].split("/")[1], r["replicate"]): r
                for r in csv.DictReader(f)}


def episodes(d: Path) -> Path:
    p = d / "artifacts" / "episodes.csv"
    return p if p.exists() else d / "episodes.csv"


def fnum(r: dict, c: str) -> float:
    v = r.get(c, "")
    return float(v) if v not in ("", "None") else 0.0


def pc_rows(r: dict) -> list:
    return json.loads(r.get("st_producer_cycles") or "[]")


def fund_shares(r: dict) -> dict[str, float]:
    tot: dict[str, float] = defaultdict(float)
    for q in pc_rows(r):
        tot[q[1]] += q[PC["fund"]] or 0.0
    s = sum(tot.values()) or 1.0
    return {p: v / s for p, v in tot.items()}


def same(a: dict, b: dict) -> list[str]:
    """Cells of `a` (this run) that differ from reference row `b`; a column the older reference
    lacks (added since, e.g. `st_b2_lost_declined`) must read zero."""
    return [k for k in set(a) | set(b) if k not in IGNORED and (
        a.get(k) != b[k] if k in b else a.get(k) not in ("", "0", "0.0"))]


def main(exp: Path) -> dict:
    from tokenomics.eval.supply_trace import GROUPS

    if hashlib.sha256(NOTE.read_bytes()).hexdigest() != NOTE_SHA:
        raise SystemExit("the pre-registration note does not match its pinned SHA-256")
    floors = {w["world"]: float(w["noise_floor"]) for w in json.loads(
        (REPO / "experiments/EXP-0073-ttl-sweep/ttl2/spec.json").read_text())["pins"]["sample"]}
    base = load(REPO / "experiments/EXP-0088-buffer-resweep-rf5" / PROD / "artifacts/episodes.csv")
    a1_ref = load(episodes(REPO / "experiments/EXP-0098-acceptance-a1-a2/ttl_feasible"))
    data, problems = {}, []
    for a, name in ARMS.items():
        d = exp / name
        if sum(1 for _ in (d / "contract_failures.csv").open()) > 1:
            problems.append(f"{a}: contract failures")
        data[a] = load(episodes(d))
        if set(data[a]) != set(base):
            problems.append(f"{a}: row set differs from EXP-0088's")
            continue
        if not all((data[a][k]["demand_digest"], data[a][k]["shock_digest"])
                   == (base[k]["demand_digest"], base[k]["shock_digest"]) for k in base):
            problems.append(f"{a}: digests differ from EXP-0088's")
    if not problems:
        bad = [k for k in base if same(data["auction"][k], base[k])]
        if bad:
            problems.append(f"auction neutrality: {len(bad)} rows differ from EXP-0088 "
                            f"(first {bad[0]}: {same(data['auction'][bad[0]], base[bad[0]])[:5]})")
        bad = [k for k in base if same(data["A1"][k], a1_ref[k])]
        if bad:
            problems.append(f"A1 reproduction: {len(bad)} rows differ from EXP-0098 "
                            f"(first {bad[0]}: {same(data['A1'][bad[0]], a1_ref[bad[0]])[:5]})")
    out: dict = {"note_sha256": NOTE_SHA, "validity": problems or "valid", "cells": []}
    if problems:
        out["reading"] = "invalid"
        return out
    classes = sorted({c for g in GROUPS.values() for c in g})
    for size in ("12", "42"):
        for o, router in ORDERERS.items():
            keys = [k for k, r in base.items() if r["size"] == size and k[2] == router]
            worlds = sorted({k[0] for k in keys})
            mf = mean(floors[w] for w in worlds)

            def per_world(arm: str, ref: str) -> list[float]:
                return [mean(float(data[arm][k]["units"]) - float(data[ref][k]["units"])
                             for k in keys if k[0] == w) for w in worlds]

            def arm_stats(arm: str) -> dict:
                rs = [data[arm][k] for k in keys]
                acc: dict[str, float] = defaultdict(float)
                for r in rs:
                    for kk, v in json.loads(r.get("acceptance_stats") or "{}").items():
                        acc[kk] += v
                pcs = [q for r in rs for q in pc_rows(r)]
                deliv = sum(q[PC["delivered"]] or 0 for q in pcs)
                disp = acc.get("displaced_waiting", 0) + acc.get("displaced_declined", 0)
                return {
                    "units_mean": mean(float(r["units"]) for r in rs),
                    "deaths": sum(r["dead"] == "true" for r in rs),
                    "rundown_fails": sum(float(r["r_end"]) < X_RUNDOWN * float(r["r_start"])
                                         for r in rs),
                    "lost_total": sum(fnum(r, "st_lost_consumer_total") for r in rs),
                    "lost_by_class": {c: sum(fnum(r, f"st_{c}") for r in rs) for c in classes},
                    "mean_cutoff": acc["cutoff_sum"] / acc["cutoff_cycles"]
                    if acc.get("cutoff_cycles") else None,
                    "declined_below_cutoff": acc.get("declined_below_cutoff", 0),
                    "displaced": disp,
                    "displaced_share_of_waiting_or_declined": disp / max(
                        1.0, acc.get("accepted_pending", 0) + acc.get("declined", 0)),
                    "displaced_pricier_share": acc.get("displaced_pricier", 0) / max(1.0, disp),
                    "displaced_mean_price": acc.get("displaced_price_sum", 0) / max(1.0, disp),
                    "pending_ran_mean_price": acc.get("pending_ran_price_sum", 0) / max(
                        1.0, acc.get("pending_ran_later", 0)),
                    "replenish_lag_cycles": sum(q[PC["inbound"]] or 0 for q in pcs) / deliv
                    if deliv else None,
                    "acceptance": dict(acc)}

            cell: dict = {"size": int(size), "orderer": o, "worlds": len(worlds), "mean_floor": mf,
                          "default": arm_stats("auction")}
            loss = -mean(per_world("A1", "auction"))
            cell["L"] = loss
            for a in ("A1", "A1-life", "A1-new-first"):
                st = arm_stats(a)
                dw = per_world(a, "auction")
                st["mean_diff_vs_default"] = mean(dw)
                st["worlds_below"] = sum(d < -floors[w] for d, w in zip(dw, worlds))
                st["fund_share_moved"] = mean(0.5 * sum(
                    abs(fund_shares(data[a][k]).get(p, 0) - fund_shares(base[k]).get(p, 0))
                    for p in set(fund_shares(data[a][k])) | set(fund_shares(base[k])))
                    for k in keys)
                if a != "A1":
                    st["R"] = mean(per_world(a, "A1")) / loss if loss else None
                cell[a] = st
            extra = {c: cell["A1"]["lost_by_class"][c] - cell["default"]["lost_by_class"][c]
                     for c in classes}
            tot = sum(extra.values())
            cell["extra_loss"] = {"total": tot, "by_class": extra,
                                  "by_group": {g: sum(extra[c] for c in cs)
                                               for g, cs in GROUPS.items()}}
            out["cells"].append(cell)
    c = next(x for x in out["cells"] if x["size"] == 12 and x["orderer"] == "MWWk")
    r_life, r_new = c["A1-life"]["R"], c["A1-new-first"]["R"]
    if c["L"] <= c["mean_floor"]:
        reading = "not_reproduced"
    elif r_life >= 0.5 and r_new < 0.5:
        reading = "lifetime"
    elif r_new >= 0.5 and r_life < 0.5:
        reading = "displacement"
    elif r_life >= 0.5 and r_new >= 0.5:
        reading = "both"
    else:
        g = max(c["extra_loss"]["by_group"], key=lambda k: c["extra_loss"]["by_group"][k])
        cl = max(GROUPS[g], key=lambda k: c["extra_loss"]["by_class"][k])
        reading = f"neither:{g}/{cl}"
    out["reading"] = reading
    out["deciding"] = {"L": c["L"], "floor": c["mean_floor"], "R_life": r_life, "R_new_first": r_new}
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1, default=str))
