"""Task 3.3 verification and summary of an EXP-0049 dir, independent of `routing_audit`'s
analysis code.

Checks: contract_failures.csv header-only; every planned (world, episode, subject, replicate)
row present once; one demand + shock digest pair per world-episode; every reference row (roster
members and the four references) equals EXP-0048's units for the same key; every `portable` row
on the shocked world-episodes equals EXP-0046's `candidate` units (and digests); staged
checkpoint sha256 = spec pins = EXP-0046 pins; edge.csv's cell / rule rows recomputed (D2 rule).
Prints per-cell tables (units, gain over the best deterministic member, recovered fraction, the
routing metrics) and the pre-registered matched contrasts for RESULT.md.

usage (from /srv/research/tokenomics):
  python3 openspec/changes/add-m3-development-audit/probes/exp0049_check.py \
      experiments/EXP-0049-informed-routing-and-trained-trace \
      experiments/EXP-0048-random-routing-diagnosis experiments/EXP-0046-m2-exit-shocked-champion
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

csv.field_size_limit(1 << 30)
DOMINANT = [(12, "shocked"), (42, "calm"), (42, "shocked")]
CELLS = [(12, "calm"), (12, "shocked"), (42, "calm"), (42, "shocked")]
METRICS = ["split_degree_consumer", "split_share_consumer", "receiver_change_share_consumer",
           "receiver_change_share_producer", "herding_hhi", "herfindahl_mean",
           "wasted_units_consumer", "lost_units_consumer_ttl", "fund_share_herfindahl_episode"]
CONTRASTS = [("two_choice_spare", "random_whole"), ("spare_weighted_whole", "random_whole"),
             ("spare_rotating_whole", "round_robin"), ("trained_orderer", "proportional"),
             ("trained_orderer", "random"), ("trained_orderer", "round_robin"),
             ("portable", "trained_orderer")]


def episodes(d: Path) -> list[dict[str, str]]:
    p = d / "artifacts" / "episodes.csv"
    return list(csv.DictReader((p if p.exists() else d / "episodes.csv").open()))


def wm(by_world: dict) -> float:
    return fmean(fmean(v) for v in by_world.values())


def main() -> None:
    d49, d48, d46 = (Path(a) for a in sys.argv[1:4])
    spec = json.loads((d49 / "spec.json").read_text())
    fails = (d49 / "contract_failures.csv").read_text().splitlines()
    print("contract_failures rows:", len(fails) - 1, "| status:", spec["status"],
          "| commit:", spec["pins"]["source"]["commit"], "| plan:", spec["pins"]["plan"])
    rows = episodes(d49)
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

    # reproduction of EXP-0048 (reference rows)
    e48 = {(r["world"], r["episode"], r["subject"], r["replicate"]): r for r in episodes(d48)}
    n = mism = 0
    for r in rows:
        k = (r["world"], r["episode"], r["subject"], r["replicate"])
        if k in e48:
            n += 1
            o = e48[k]
            if (float(o["units"]) != float(r["units"]) or o["demand_digest"] != r["demand_digest"]
                    or o["shock_digest"] != r["shock_digest"]):
                mism += 1
    refs = sorted({r["subject"] for r in rows
                   if (r["world"], r["episode"], r["subject"], r["replicate"]) in e48})
    print(f"reference rows in EXP-0048: {n} ({len(refs)} subjects), mismatches {mism}")

    # portable vs EXP-0046 candidate
    e46 = {(r["world"], r["episode"]): r for r in csv.DictReader((d46 / "episodes.csv").open())
           if r["subject"] == "candidate" and r["world"].startswith("dev-")}
    n = mism = 0
    for r in rows:
        if r["subject"] == "portable" and (r["world"], r["episode"]) in e46:
            n += 1
            o = e46[(r["world"], r["episode"])]
            if (float(o["units"]) != float(r["units"]) or o["demand_digest"] != r["demand_digest"]
                    or o["shock_digest"] != r["shock_digest"]):
                mism += 1
    print(f"portable rows matched to EXP-0046 candidate: {n} of {len(e46)}, mismatches {mism}")
    s46 = json.loads((d46 / "spec.json").read_text())["pins"]["checkpoints"]
    pins = spec["pins"]["checkpoints"]
    print("ckpt pins: producer", pins["producer"][:12], "= EXP-0046",
          pins["producer"] == s46["portable_producer"], "| orderer", pins["orderer"][:12],
          "= EXP-0046", pins["orderer"] == s46["portable_orderer"])
    for role, rel in (("producer", "tokenomics/_pod_ckpts/producer_candidate.pt"),
                      ("orderer", "tokenomics/_pod_ckpts/orderer_scorer.pt")):
        p = Path(rel)
        if p.exists():
            print(f"staged {role} sha256 = pin:", hashlib.sha256(p.read_bytes()).hexdigest() == pins[role])

    # per world-episode replicate means
    name = {r["subject"]: r["router"] for r in routers}
    order = [r["router"] for r in routers]
    det = [r["router"] for r in routers if r["roster"] and not r["random"]]
    val: dict = defaultdict(lambda: defaultdict(list))
    met: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dead: dict = defaultdict(lambda: defaultdict(list))
    cell_of = {}
    for r in rows:
        key, rn = (r["world"], r["episode"]), name[r["subject"]]
        val[key][rn].append(float(r["units"]))
        dead[key][rn].append(1.0 if r["dead"] == "true" else 0.0)
        for m in METRICS:
            met[key][rn][m].append(float(r["trace_" + m]))
        cell_of[r["world"]] = (int(r["size"]), r["regime"])
    edges: dict = defaultdict(lambda: defaultdict(list))
    gains: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    units: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    deaths: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (w, ep), per in val.items():
        c = cell_of[w]
        best = max(fmean(per[m]) for m in det)
        edges[c][w].append(fmean(per["random"]) - best)
        for rn, v in per.items():
            gains[c][rn][w].append(fmean(v) - best)
            units[c][rn][w].append(fmean(v))
            deaths[c][rn][w].append(fmean(dead[(w, ep)][rn]))
            for m in METRICS:
                mets[c][rn][m][w].append(fmean(met[(w, ep)][rn][m]))
    rec = {}
    for c in sorted(edges):
        e = wm(edges[c])
        for rn in gains[c]:
            rec[(c, rn)] = wm(gains[c][rn]) / e if e > 0 else None

    edge_csv = list(csv.DictReader((d49 / "edge.csv").open()))
    bad = 0
    for r in edge_csv:
        if r["level"] == "cell":
            mine = rec[((int(r["size"]), r["regime"]), r["router"])]
            theirs = None if r["recovered_fraction"] == "" else float(r["recovered_fraction"])
            if (mine is None) != (theirs is None) or (mine is not None and abs(mine - theirs) > 1e-9):
                bad += 1
    rule = {r["router"]: r for r in edge_csv if r["level"] == "rule"}
    rbad = 0
    for rn in rule:
        got = sum(1 for c in DOMINANT if rec.get((c, rn)) is not None and rec[(c, rn)] >= 0.8 - 1e-9)
        if got != int(rule[rn]["cells_recovered"]) or (got >= 2) != (rule[rn]["recovers_most"] == "true"):
            rbad += 1
    print("edge.csv cell rows", sum(r["level"] == "cell" for r in edge_csv), "mismatches", bad,
          "| rule rows", len(rule), "mismatches", rbad)

    print("\nedge per cell (random minus best deterministic member, equal-weight world mean):")
    for c in CELLS:
        print(f"  {c}: {wm(edges[c]):.1f} over {len(edges[c])} worlds")
    print("\nper cell: router | units | gain | recovered | dead | " + " | ".join(METRICS))
    for c in CELLS:
        print(f"-- {c[0]}-{c[1]}")
        for rn in order:
            fr = rec.get((c, rn))
            print(f"  {rn:22s} {wm(units[c][rn]):8.1f} {wm(gains[c][rn]):+8.1f} "
                  + ("   n/a" if fr is None else f"{fr:6.2f}") + f" {wm(deaths[c][rn]):5.2f} "
                  + " ".join(f"{wm(mets[c][rn][m]):7.3f}" for m in METRICS))
    print("\nrule: dominant cells recovered (>= 0.8)")
    for rn in order:
        got = sum(1 for c in DOMINANT if rec.get((c, rn)) is not None and rec[(c, rn)] >= 0.8 - 1e-9)
        print(f"  {rn:22s} {got} of 3")

    print("\nmatched contrasts: a - b, equal-weight world mean units per cell "
          "[worlds with a > b / worlds] (episodes a > b / = / <)")
    for a, b in CONTRASTS:
        cells = []
        beats = 0
        for c in CELLS:
            diff = {w: [x - y for x, y in zip(units[c][a][w], units[c][b][w])] for w in units[c][a]}
            d = wm(diff)
            wpos = sum(fmean(v) > 0 for v in diff.values())
            eps = [x for v in diff.values() for x in v]
            cells.append(f"{c[0]}-{c[1]} {d:+7.1f} [{wpos}/{len(diff)}] "
                         f"({sum(x > 0 for x in eps)}/{sum(x == 0 for x in eps)}/{sum(x < 0 for x in eps)})")
            beats += int(c in DOMINANT and d > 0)
        print(f"  {a} - {b}: " + "; ".join(cells) + f" | dominant cells positive: {beats} of 3")


if __name__ == "__main__":
    main()
