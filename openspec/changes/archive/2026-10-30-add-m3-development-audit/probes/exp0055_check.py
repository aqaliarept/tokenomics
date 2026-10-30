"""Independent check of EXP-0055 (task 11.1): rebuild cells.csv and slopes.csv from
timings.csv with stdlib arithmetic written apart from `eval/deployed_cost.py` (no import of
the driver), and check the grid, the pins and the host record.

    python3 probes/exp0055_check.py <EXP-0055 dir> [--ckpt producer.pt --ckpt orderer.pt]

Prints PASS / FAIL lines; exit 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import fmean, median

AXES = {  # D7: (producers, fanout, consumer multiplier) per axis, increasing
    "producers": [(12, 2, 1), (24, 2, 1), (42, 2, 1)],
    "fanout": [(42, 2, 1), (42, 4, 1), (42, 7, 1), (42, 14, 1)],
    "consumers_per_producer": [(42, 4, 1), (42, 4, 2), (42, 4, 4)],
}
SEEDS = (1000, 1001, 1002)
REPEATS = 3
HORIZON = 20
STAGES = ("obs", "forward", "decode", "domain", "harness")
POLICY = ("obs", "forward", "decode")
FLAG_AXES = ("fanout", "consumers_per_producer")
PINS = {"producer": "8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea",
        "orderer": "d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c"}

fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def cid(p: int, e: int, k: int) -> str:
    return f"P{p}-E{e}-k{k}"


def per_measure(r: dict[str, str]) -> dict[tuple[str, str], float]:
    """(measure, stage) -> seconds over the measure's denominator, for one episode."""
    out: dict[tuple[str, str], float] = {}
    cyc, dec = float(r["cycles"]), float(r["decisions"])
    for s in STAGES:
        out[("decision", s)] = float(r[f"{s}_s"]) / dec
        out[("cycle", s)] = float(r[f"{s}_s"]) / cyc
    out[("decision", "total")] = float(r["wall_s"]) / dec
    out[("cycle", "total")] = float(r["wall_s"]) / cyc
    for role, m in (("prod", "prod_decision"), ("order", "order_decision")):
        n = float(r[f"{role}_decisions"])
        vals = [float(r[f"{role}_{s}_s"]) for s in POLICY]
        for s, v in zip(POLICY, vals):
            out[(m, s)] = v / n
        out[(m, "total")] = sum(vals) / n
    return out


def slope(xs: list[float], ys: list[float]) -> float:
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return (sum((a - mx) * (b - my) for a, b in zip(lx, ly))
            / sum((a - mx) ** 2 for a in lx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path)
    ap.add_argument("--ckpt", action="append", default=[])
    a = ap.parse_args()
    d = a.exp_dir
    rows = list(csv.DictReader(open(d / "timings.csv")))
    spec = json.loads((d / "spec.json").read_text())

    # grid completeness
    cells = sorted({c for ax in AXES.values() for c in ax})
    want = {(cid(*c), s, r) for c in cells for s in SEEDS for r in range(REPEATS)}
    got = [(r["cell"], int(r["world_seed"]), int(r["repeat"])) for r in rows]
    reps = sorted({int(r["repeat"]) for r in rows})
    if reps and reps[0] == 1:  # repeats numbered from 1
        want = {(c, s, r + 1) for c, s, r in want}
    check(len(got) == len(set(got)) == len(want) == 72 and set(got) == want,
          f"timings.csv holds every cell x seed x repeat once ({len(got)} rows, want 72)")
    check(all(int(r["horizon"]) == HORIZON and int(r["cycles"]) == HORIZON
              and int(r["dead"]) == 0 for r in rows),
          "every episode ran horizon 20 to the end, no death")
    # determinism across repeats (calm worlds): same decisions / units per (cell, seed)
    by_cs: dict[tuple[str, int], set] = {}
    for r in rows:
        by_cs.setdefault((r["cell"], int(r["world_seed"])), set()).add(
            (r["decisions"], r["prod_decisions"], r["order_decisions"], r["units"]))
    check(all(len(v) == 1 for v in by_cs.values()),
          "repeats of one world play identical decisions and units")
    # exclusive accounting
    worst = max(abs(sum(float(r[f"{s}_s"]) for s in STAGES) - float(r["wall_s"]))
                / float(r["wall_s"]) for r in rows)
    check(worst <= 0.05, f"stages sum to wall within 5% (worst {worst:.4%})")
    # timings.csv rounds seconds to 6 dp: three rounded terms differ by <= 1.5e-6 s
    worst_role = max(abs(float(r[f"prod_{s}_s"]) + float(r[f"order_{s}_s"]) - float(r[f"{s}_s"]))
                     for r in rows for s in POLICY)
    check(worst_role <= 1.5e-6, f"prod + order = pooled policy stage (worst {worst_role:.1e} s)")

    # cells: per world median over repeats, mean over worlds
    table: dict[str, dict[tuple[str, str], float]] = {}
    for c in cells:
        worlds = []
        for s in SEEDS:
            eps = [per_measure(r) for r in rows if r["cell"] == cid(*c)
                   and int(r["world_seed"]) == s]
            worlds.append({k: median(e[k] for e in eps) for k in eps[0]})
        table[cid(*c)] = {k: fmean(w[k] for w in worlds) for k in worlds[0]}
    cells_csv = {r["cell"]: r for r in csv.DictReader(open(d / "cells.csv"))}
    # cells.csv rounds to 6 dp in seconds, i.e. +-5e-7 s on per-decision values of ~1e-5 s;
    # compare at that absolute tolerance and report the precision the file keeps
    worst_c, min_sig = 0.0, 99.0
    for c, vals in table.items():
        for (m, s), v in vals.items():
            col = (f"{'prod_' if m == 'prod_decision' else 'order_' if m == 'order_decision' else ''}"
                   f"{s}_per_{'cycle' if m == 'cycle' else 'decision'}_s")
            worst_c = max(worst_c, abs(float(cells_csv[c][col]) - v))
            min_sig = min(min_sig, math.log10(v / 5e-7))
    check(worst_c <= 5.000001e-7,
          f"cells.csv recomputes from timings.csv to its 6 dp (worst abs {worst_c:.1e} s)")
    print(f"note: cells.csv keeps as few as {min_sig:.1f} significant digits on its smallest "
          "per-decision column (6 dp in seconds); read per-decision values from timings.csv")

    # slopes
    slopes_csv = {(r["axis"], r["measure"], r["stage"]): r
                  for r in csv.DictReader(open(d / "slopes.csv"))}
    mine: dict[tuple[str, str, str], tuple[float, bool]] = {}
    for axis, members in AXES.items():
        xs = [{"producers": p, "fanout": e,
               "consumers_per_producer": None}[axis] for p, e, k in members]
        if axis == "consumers_per_producer":
            xs = [float(next(r["consumers"] for r in rows if r["cell"] == cid(*c))) / c[0]
                  for c in members]
        for key in table[cid(*members[0])]:
            ys = [table[cid(*c)][key] for c in members]
            if min(ys) <= 0:
                continue
            b = slope([float(x) for x in xs], ys)
            flagged = axis in FLAG_AXES and key[0] != "cycle" and b > 1.1
            mine[(axis, *key)] = (b, flagged)
    check(set(mine) == set(slopes_csv),
          f"slopes.csv has exactly the recomputed rows ({len(slopes_csv)} vs {len(mine)})")
    bad = [k for k, (b, f) in mine.items() if k in slopes_csv and (
        abs(float(slopes_csv[k]["slope"]) - b) > 6e-5
        or bool(slopes_csv[k]["flagged"]) != f)]
    check(not bad, f"slopes.csv slopes (4 dp) and flags recompute ({len(bad)} mismatches)")
    flagged = sorted(k for k, (_, f) in mine.items() if f)
    print("flagged rows:", flagged or "none")

    # pins + host
    blob = json.dumps(spec)
    check(all(p in blob for p in PINS.values()), "spec.json carries both registered pins")
    for path in a.ckpt:
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        check(h in PINS.values(), f"staged {Path(path).name} sha256 {h[:12]} is a registered pin")
    host = spec.get("host", {})
    print("host:", {k: host.get(k) for k in ("cpu_model", "vcpu", "os_cpu_count", "affinity",
                                             "cgroup_cpu_quota", "loadavg_start",
                                             "loadavg_end")})
    print("source commit:", spec.get("source", {}).get("commit"), "status:", spec.get("status"))
    print("FAIL" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
