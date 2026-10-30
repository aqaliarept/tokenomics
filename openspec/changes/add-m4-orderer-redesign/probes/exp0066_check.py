"""Independent check of an EXP-0066 run dir (task 4.6). Stdlib only; imports nothing from the
package. Usage:

    python3 -I exp0066_check.py <EXP-0066 dir> <EXP-0057 episodes.csv>

Checks: contract_failures.csv header-only; the 284 rows = EXP-0057's TS / MWWk keys once each;
every common scalar cell (and the fulfilled trace) equals EXP-0057's row; in both streams every
count's classes sum to the traced no_fit loss and the observer total equals it; self-check
counters 0; the per-size shares and the reading recomputed from the rows = reading.json; the
per-world shares recomputed = split_need_worlds.csv. Exit 0 on ALL PASS, else 1."""
from __future__ import annotations

import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

ARMS = {"TS": "entity_planner/trained_sampled_whole", "MWWk": "entity_planner/mww_keyed"}
CLASSES = ("choice", "split", "unfillable")
COUNTS = ("joint", "L", "P", "units")
STREAMS = ("consumer", "producer")
THRESHOLD = Fraction(5, 100)
FAILS = ("dropped", "ttl", "ttl_outage")

fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def key(d: dict) -> tuple:
    return (d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))


def F(x: str) -> Fraction:
    return Fraction(x) if x not in ("", None) else Fraction(0)


def find(root: Path, name: str) -> Path:
    for p in (root / name, root / "artifacts" / name):
        if p.exists():
            return p
    raise SystemExit(f"missing {name} under {root}")


def completion(d: dict) -> Fraction:
    done = F(d["units"]) + F(d["trace_wasted_units_consumer"])
    failed = sum((F(d[f"trace_failed_bid_units_consumer_{k}"]) for k in FAILS), Fraction(0))
    return done / (done + failed) if done + failed else Fraction(0)


def main(root: Path, ref_path: Path) -> int:
    cf = list(csv.reader(open(find(root, "contract_failures.csv"))))
    check(len(cf) == 1, f"contract_failures.csv header-only ({len(cf) - 1} rows)")

    rows = list(csv.DictReader(open(find(root, "episodes.csv"))))
    ref = list(csv.DictReader(open(ref_path)))
    want = {key(d): d for d in ref if d["arm"] in ARMS}
    got = {key(d): d for d in rows}
    check(len(rows) == 284 and len(got) == 284, f"284 rows, distinct ({len(rows)}, {len(got)})")
    check(set(got) == set(want), f"keys = EXP-0057 TS/MWWk keys (missing {len(set(want) - set(got))},"
                                 f" extra {len(set(got) - set(want))})")
    check(all(d["arm"] in ARMS and ARMS[d["arm"]] == d["subject"] for d in rows), "arm ↔ subject")
    sizes = {d["size"] for d in rows}
    n_we = {s: len({(d["world"], d["episode"]) for d in rows if d["size"] == s}) for s in sizes}
    n_w = {s: len({d["world"] for d in rows if d["size"] == s}) for s in sizes}
    check(n_w == {"12": 23, "42": 14} and n_we == {"12": 43, "42": 28},
          f"worlds {n_w}, world-episodes {n_we}")

    # neutrality: every scalar column the two files share, plus fulfilled_trace (JSON series)
    common = [c for c in ref[0] if c in rows[0] and c not in ("arm",)]
    diffs, compared = {}, set()
    for k, d in got.items():
        b = want.get(k)
        if b is None:
            continue
        for c in common:
            compared.add(c)
            if d[c] != b[c]:
                diffs.setdefault(c, 0)
                diffs[c] += 1
    check(not diffs, f"neutrality: {len(compared)} common cells (incl. series) equal EXP-0057 on all"
                     f" rows; differing {diffs}")

    # consistency
    bad = []
    for d in rows:
        for s in STREAMS:
            t = F(d[f"trace_lost_units_{s}_no_fit"])
            if F(d[f"sn_lost_total_{s}"]) != t:
                bad.append((key(d), s, "total"))
            for c in COUNTS:
                if sum(F(d[f"sn_{c}_{s}_{k}_units"]) for k in CLASSES) != t:
                    bad.append((key(d), s, c))
        for k in ("replay_mismatch", "dup_bids", "qty_mismatch"):
            if F(d[f"sn_{k}"]):
                bad.append((key(d), k))
    check(not bad, f"classes sum to traced no_fit (4 counts × 2 streams) + counters 0; bad {bad[:5]}")
    tot = {s: sum(F(d[f"trace_lost_units_{s}_no_fit"]) for d in rows) for s in STREAMS}
    print(f"     traced no_fit units: consumer {float(tot['consumer']):.0f}, producer"
          f" {float(tot['producer']):.0f}")

    # reading
    rd = json.load(open(find(root, "reading.json")))
    rd = rd.get("reading", rd) if isinstance(rd.get("reading"), dict) else rd
    fired = []
    for s in ("12", "42"):
        for arm in ARMS:
            rr = [d for d in rows if d["size"] == s and d["arm"] == arm]
            filled = sum(F(d["units"]) for d in rr)
            corr = sum(F(d["sn_joint_consumer_split_units"]) * completion(d) for d in rr)
            share = corr / filled
            jl = {c: sum(F(d[f"sn_{c}_consumer_split_units"]) for d in rr) / filled for c in COUNTS}
            lost = sum(F(d["trace_lost_units_consumer_no_fit"]) for d in rr)
            js = rd["per_size"][s][arm]
            check(abs(js["share_corrected"] - float(share)) < 1e-12
                  and abs(js["filled_units"] - float(filled)) < 1e-9
                  and abs(js["corrected_split_units"] - float(corr)) < 1e-6,
                  f"[{s}] {arm}: corrected share {float(share):.4%} ({float(corr):.1f} / "
                  f"{float(filled):.0f}) = reading.json; rows {len(rr)}")
            print(f"     [{s}] {arm}: no_fit lost {float(lost):.0f} ({float(lost / filled):.1%} of filled);"
                  f" uncorrected split shares " + ", ".join(f"{c} {float(v):.2%}" for c, v in jl.items()))
            if arm == "MWWk" and share >= THRESHOLD:
                fired.append(s)
    mine = "split_needed" if fired else "whole_suffices"
    check(rd["reading"] == mine, f"reading from rows = {mine} (fired at {fired}); reading.json"
                                 f" {rd['reading']}")

    # per-world shares
    wpath = find(root, "split_need_worlds.csv")
    wrows = list(csv.DictReader(open(wpath)))
    worst = 0.0
    for w in wrows:
        rr = [d for d in rows if d["world"] == w["world"] and d["arm"] == w["arm"]]
        filled = sum(F(d["units"]) for d in rr)
        corr = sum(F(d["sn_joint_consumer_split_units"]) * completion(d) for d in rr)
        if filled and w.get("share_corrected") not in ("", None):
            worst = max(worst, abs(float(w["share_corrected"]) - float(corr / filled)))
    check(len(wrows) == 74 and worst < 1e-9, f"split_need_worlds.csv {len(wrows)} rows; per-world"
                                             f" corrected share max |Δ| {worst:.2e}")
    print("ALL PASS" if not fails else f"{len(fails)} FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
