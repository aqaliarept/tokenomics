"""Task 4.7: re-bound EXP-0051 with the input-output LP (`ceiling_bound.lp_bound`), offline.

`compute` rebuilds every EXP-0051 world-episode's configuration (world records in spec.json; the
8 anchors from `atlas.anchor_worlds`) and per-cycle demand (env reset with the recorded episode
seed), solves B_lp (guaranteed: TTL window, costed upgrades) and B_lp_base (the same without
upgrades: capacity-aware, not guaranteed), and appends one JSON line per world-episode (resumable;
chunk with --start/--count so each command stays under 2 min). `analyse` checks soundness
(B_lp >= every recorded arm's units + 0.01) and reclassifies every world with the ceiling read
against B_lp or B_lp_base in place of B_base, using the stdlib re-implementation in
exp0051_check.py, then reruns the selection and split. EXP-0051's files are only read.

usage (from /srv/research/tokenomics/tokenomics, venv with the lp extra):
  ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/rebound_47.py compute \
      --start 0 --count 80 --out ../openspec/changes/add-m3-development-audit/probes/rebound_47.jsonl
  ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/rebound_47.py analyse \
      --out ../openspec/changes/add-m3-development-audit/probes/rebound_47.jsonl [--json SUMMARY]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

csv.field_size_limit(1 << 30)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP = ROOT / "experiments" / "EXP-0051-headroom-atlas-screen"
HORIZON = 100
TOL = 0.01


def world_episodes() -> list[tuple[str, int]]:
    seen: dict[str, int] = {}
    with (EXP / "artifacts" / "episodes.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            seen.setdefault(r["world"], int(r["episode"]))
    return sorted(seen.items())


def compute(start: int, count: int, out: Path) -> None:
    from tokenomics.eval import atlas
    from tokenomics.eval.ceiling_bound import episode_cycle_demand, lp_bound
    from tokenomics.eval.routing_audit import job_config
    from tokenomics.eval.world_space import point_from_record

    spec = json.loads((EXP / "spec.json").read_text())
    recs = {r["world_id"]: r for r in spec["pins"]["worlds"] if "world_id" in r}
    anchors = {e.id: e for e in atlas.anchor_worlds()}
    done = set()
    if out.exists():
        done = {json.loads(line)["world"] for line in out.read_text().splitlines() if line}
    for w, ep in world_episodes()[start:start + count]:
        if w in done:
            continue
        e = anchors[w] if w in anchors else atlas.space_entry(point_from_record(recs[w]), [ep])
        cfg = job_config({"world": e.id, "world_seed": e.world_seed, "generator": e.generator})
        t0 = time.perf_counter()
        demand = episode_cycle_demand(cfg, HORIZON, ep)
        b = lp_bound(cfg, HORIZON, demand)
        base = lp_bound(cfg, HORIZON, demand, upgrades=False)
        line = {"world": w, "episode": ep, "b_lp": b.units, "b_lp_binding": b.binding_constraint,
                "b_lp_base": base.units, "b_lp_base_binding": base.binding_constraint,
                "demand": sum(sum(q) for q in demand.values()),
                "seconds": round(time.perf_counter() - t0, 3)}
        with out.open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
        print(f"{w} B_lp {b.units:.1f} B_lp_base {base.units:.1f} {line['seconds']} s", flush=True)


def analyse(out: Path, summary_path: Path | None) -> int:
    sys.path.insert(0, str(HERE))
    import exp0051_check as c

    spec = json.loads((EXP / "spec.json").read_text())
    recs = {r["world_id"]: r for r in spec["pins"]["worlds"] if "world_id" in r}
    lp = {d["world"]: d for d in map(json.loads, out.read_text().splitlines()) if d}
    rows = c.read(EXP / "artifacts" / "episodes.csv")
    vals = c.values(rows)
    missing = sorted(set(vals) - set(lp))
    table = c.arms()
    sp = spec["result"]["provisional_spreads"]
    infos = {}
    for w in vals:
        if w in recs:
            lv = recs[w]["levers"]
            infos[w] = {"size": recs[w]["size"], "choice": lv["choice"], "pattern": lv["pattern"],
                        "shocks": bool(lv["shocks"]),
                        "differing": not (lv["pattern"] == "cyclic" and not lv["shocks"])}
        else:
            (size, choice), _ = c.exp0050_stratum(w)
            infos[w] = {"size": size, "choice": choice, "pattern": "anchor", "shocks": True,
                        "differing": True}
    problems = [f"{len(missing)} world-episodes without a B_lp line, e.g. {missing[:3]}"] if (
        missing) else []
    viol = []
    for r in rows:
        d = lp.get(r["world"])
        if d and float(r["units"]) > d["b_lp"] + TOL:
            viol.append((r["world"], r["subject"], r["replicate"], r["units"], d["b_lp"]))
    if viol:
        problems.append(f"{len(viol)} rows above B_lp, e.g. {viol[:3]}")

    def classes(field: str | None) -> dict[str, dict]:
        out_m = {}
        for w in vals:
            if w not in lp:
                continue
            eps = vals[w]
            if field is not None:
                for ep in eps.values():
                    ep["base"] = lp[w][field]
            m = c.measure(eps, table)
            se, floor, sign, cls = c.classify(m, infos[w], sp.get(f"{infos[w]['size']}/"
                                                                 f"{infos[w]['choice']}"))
            m.update(cls=cls, floor=floor)
            out_m[w] = m
        return out_m

    base_bb = {w: {e: ep["base"] for e, ep in eps.items()} for w, eps in vals.items()}
    variants = {"B_base (EXP-0051)": classes(None)}
    for field, name in (("b_lp", "B_lp (guaranteed)"), ("b_lp_base", "B_lp_base")):
        for w, eps in vals.items():
            for e, ep in eps.items():
                ep["base"] = base_bb[w][e]
        variants[name] = classes(field)
    ref = variants["B_base (EXP-0051)"]
    summary: dict = {"violations": len(viol), "missing": len(missing), "variants": {}}
    sel_csv = c.read(EXP / "selection.csv")
    old_sel = {(int(s["size"]), s["world"]) for s in sel_csv}
    for name, ms in variants.items():
        v: dict = {}
        for size in (12, 42):
            ws = [w for w in ms if w in recs and infos[w]["size"] == size]
            if not ws:
                continue
            v[f"{size}"] = dict(Counter(ms[w]["cls"] for w in ws))
            v[f"{size} changed"] = dict(Counter(f"{ref[w]['cls']}->{ms[w]['cls']}" for w in ws
                                                if ms[w]["cls"] != ref[w]["cls"]))
            cands = [{"world": w, "cls": ms[w]["cls"], "gain": ms[w]["gain"], "bar": ms[w]["bar"],
                      "room": ms[w]["room"], "rec": recs[w]} for w in ws]
            sel = c.select(cands, c.K[size], c.bins(size, [recs[w] for w in ws]))
            new = {(size, x["world"]) for x, _ in sel}
            v[f"{size} selection"] = {"n": len(new), "kept": len(new & old_sel),
                                      "dropped": sorted(w for s, w in old_sel - new if s == size),
                                      "added": sorted(w for s, w in new - old_sel)}
        anc = {w: ms[w]["cls"] for w in ms if w not in recs}
        v["anchors"] = anc
        summary["variants"][name] = v
        print(f"\n== ceiling against {name}")
        for size in (12, 42):
            if f"{size}" not in v:
                continue
            print(f"  {size}: {v[f'{size}']}; changes vs EXP-0051 {v[f'{size} changed']}")
            s = v[f"{size} selection"]
            print(f"  {size} selection: {s['kept']} of {s['n']} kept; dropped {len(s['dropped'])},"
                  f" added {len(s['added'])}")
        print(f"  anchors: {anc}")
    rat = [(lp[w]["b_lp"], lp[w]["b_lp_base"], vals[w][e]["b2"], base_bb[w][e]) for w in lp
           if w in vals for e in vals[w]]
    eq_b2 = sum(abs(a - b2) <= 1e-6 * max(1.0, b2) for a, _, b2, _ in rat)
    below_base = sum(lb < bb - 1e-6 for _, lb, _, bb in rat)
    above_base = sum(lb > bb + TOL for _, lb, _, bb in rat)  # design D12: B_lp_base <= B_base
    above_b2 = sum(a > b2 + TOL for a, _, b2, _ in rat)
    if above_base:
        problems.append(f"B_lp_base above B_base on {above_base} world-episodes")
    secs = [lp[w]["seconds"] for w in lp]
    summary.update(n=len(rat), b_lp_equals_b2=eq_b2, b_lp_base_below_b_base=below_base,
                   b_lp_base_above_b_base=above_base, b_lp_above_b2=above_b2,
                   b_lp_below_demand=sum(lp[w]["b_lp"] < lp[w]["demand"] - TOL for w in lp),
                   b_lp_base_over_b_base_median=sorted(lb / bb for _, lb, _, bb in rat)[len(rat) // 2],
                   seconds_total=round(sum(secs), 1), seconds_max=max(secs))
    print(f"\nworld-episodes {len(rat)}: B_lp = B2 on {eq_b2}; B_lp_base < B_base on {below_base} "
          f"(median ratio {summary['b_lp_base_over_b_base_median']:.4f}); "
          f"solve time total {summary['seconds_total']} s, max {summary['seconds_max']} s")
    print(f"rows above B_lp: {len(viol)}; B_lp_base above B_base: {above_base}; B_lp above B2: "
          f"{above_b2}; B_lp below demand: {summary['b_lp_below_demand']}")
    if summary_path:
        summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("compute", "analyse"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10 ** 6)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.mode == "compute":
        compute(a.start, a.count, a.out)
        return 0
    return analyse(a.out, a.json)


if __name__ == "__main__":
    sys.exit(main())
