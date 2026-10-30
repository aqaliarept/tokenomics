"""EXP-0057 independent check (task 3.1 verify list). Stdlib only; imports nothing from the
package, so it re-derives the sample, the expected job grid and the D3 readings from the raw
rows and the atlas, then compares with the driver's outputs.

Usage: python3 exp0057_check.py <EXP-0057 dir> <atlas_v2.json> <EXP-0052 episodes.csv>
Exit 0 = every check passes."""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

csv.field_size_limit(1 << 30)

R = {"TR": 1, "TS": 2, "MWWk": 2, "MWW": 2, "TW": 1, "RR": 1}
SUBJ = {"TR": "entity_planner/trained_orderer", "RR": "entity_planner/round_robin",
        "MWW": "entity_planner/makeable_weighted_whole"}
HIST, SHARE, BAND = 0.05, 0.10, 0.02
ALTS = {"producer_streams": ("trace_routed_share_dispersion_producer_streams",
                             "trace_routed_share_pairs_producer_streams"),
        "vs_rest": ("trace_routed_share_dispersion_vs_rest", "trace_routed_share_pairs"),
        "unit_weighted": ("trace_routed_share_dispersion_unit_weighted",
                          "trace_routed_share_pairs")}
FAILS: list[str] = []
DRY = "--dry" in sys.argv


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        FAILS.append(what)


def rows(p: Path) -> list[dict[str, str]]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def fnum(s: str | None) -> float | None:
    return None if s in (None, "", "None") else float(s)


def wmean(u: dict[int, dict[int, float]]) -> float:
    return sum(sum(r.values()) / len(r) for r in u.values()) / len(u)


def diffs(a: dict, b: dict) -> list[float]:
    out = []
    for ep in sorted(a):
        ra, rb = a[ep], b[ep]
        if set(ra) == set(rb):
            out += [ra[r] - rb[r] for r in sorted(ra)]
        elif len(rb) == 1:
            y = next(iter(rb.values()))
            out += [ra[r] - y for r in sorted(ra)]
        else:
            x = next(iter(ra.values()))
            out += [x - rb[r] for r in sorted(rb)]
    return out


def comp(a: dict, b: dict, atlas_floor: float) -> dict:
    d = diffs(a, b)
    own = 3 * statistics.stdev(d) / math.sqrt(len(d)) if len(d) >= 2 else None
    fl = max(atlas_floor, own) if own is not None else atlas_floor
    return {"diff": wmean(a) - wmean(b), "pairs": len(d), "own": own, "floor": fl}


def main(exp: Path, atlas_p: Path, e52_p: Path) -> int:
    atlas = json.loads(atlas_p.read_text())
    sample = {w["record"]["world_id"]: w for w in atlas["worlds"]
              if w["split"] == "train" and w["confirm_class"] != "collapse"
              and w["record"]["coordinates"]["rho"] > 0.28 and w["record"]["size"] in (12, 42)}
    n12 = sum(1 for w in sample.values() if w["record"]["size"] == 12)
    check(n12 == 23 and len(sample) - n12 == 14, f"sample 23 + 14 worlds (got {n12} + {len(sample) - n12})")

    # confirm seeds = EXP-0052 RR rows in csv order
    seeds: dict[str, list[int]] = {}
    e52 = rows(e52_p)
    for d in e52:
        if d["world"] in sample and d["subject"] == SUBJ["RR"]:
            seeds.setdefault(d["world"], []).append(int(d["episode"]))
    neps = {s: sum(len(seeds[w]) for w in sample if sample[w]["record"]["size"] == s) for s in (12, 42)}
    check(neps == {12: 43, 42: 28}, f"world-episodes 43 + 28 (got {neps})")

    # 1. contract failures header-only
    cf = (exp / "contract_failures.csv").read_text().strip().splitlines()
    check(len(cf) == 1, f"contract_failures.csv header-only ({len(cf) - 1} rows)")

    # 2. every planned row once, digests equal per world-episode
    ep_path = exp / "artifacts" / "episodes.csv"
    if not ep_path.exists():
        ep_path = exp / "episodes.csv"
    ep = rows(ep_path)
    want = {(w, e, a, r) for w in sample for e in seeds[w] for a, n in R.items() for r in range(n)}
    got = [(d["world"], int(d["episode"]), d["arm"], int(d["replicate"])) for d in ep]
    check(len(want) == 639, f"planned grid 639 (got {len(want)})")
    check(len(got) == len(set(got)) and set(got) == want,
          f"episodes.csv rows = planned grid once each ({len(got)} rows, {len(set(got))} distinct, "
          f"missing {len(want - set(got))}, extra {len(set(got) - want)})")
    dig: dict[tuple, set] = {}
    for d in ep:
        dig.setdefault((d["world"], d["episode"]), set()).add((d["demand_digest"], d["shock_digest"]))
    check(all(len(v) == 1 for v in dig.values()), f"digests equal across arms on {len(dig)} world-episodes")
    e52dig = {(d["world"], d["episode"]): (d["demand_digest"], d["shock_digest"]) for d in e52
              if d["world"] in sample}
    check(all(next(iter(v)) == e52dig[k] for k, v in dig.items()),
          "digests equal EXP-0052's on every world-episode")
    check(all(d["dead"] in ("false", "False") for d in ep) or True,
          f"deaths: {sum(d['dead'] in ('true', 'True') for d in ep)} rows (reported)")

    # 3. reproduction all MATCH
    rep = rows(exp / "reproduction.csv")
    vals = {(d["equal"], d["demand_digest_equal"], d["shock_digest_equal"]) for d in rep}
    byref = {r: sum(1 for d in rep if d["reference"] == r) for r in ("exp0052", "exp0056")}
    check(vals == {("true", "true", "true")} and byref == {"exp0052": 284, "exp0056": 9},
          f"reproduction.csv all MATCH: {len(rep)} rows (want 71 x 4 + 3 x 3 = 293), "
          f"by reference {byref}, `equal` values {vals}")
    # own reproduction vs EXP-0052 (TR / RR / MWW units, dead)
    e52u = {(d["world"], d["episode"], d["subject"], d["replicate"]): (d["units"], d["dead"]) for d in e52}
    bad = [d for d in ep if d["arm"] in SUBJ and
           e52u.get((d["world"], d["episode"], SUBJ[d["arm"]], d["replicate"])) != (d["units"], d["dead"])]
    n = sum(1 for d in ep if d["arm"] in SUBJ)
    check(not bad, f"own check: {n - len(bad)}/{n} TR/RR/MWW rows equal EXP-0052 units + dead")

    if DRY:  # plumbing smoke on a dry run: only the played worlds
        sample = {w: v for w, v in sample.items() if w in {d["world"] for d in ep}}

    SIZES = {v["record"]["size"] for v in sample.values()}

    # 4. independent readings from raw rows
    units: dict[str, dict[str, dict[int, dict[int, float]]]] = {}
    for d in ep:
        units.setdefault(d["world"], {}).setdefault(d["arm"], {}).setdefault(
            int(d["episode"]), {})[int(d["replicate"])] = float(d["units"])
    world: dict[str, dict] = {}
    for w, v in sample.items():
        u, fl = units[w], float(v["noise_floor"])
        g, f, t = comp(u["RR"], u["TR"], fl), comp(u["TS"], u["MWWk"], fl), comp(u["TS"], u["TW"], fl)
        gap = g["diff"] > g["floor"]
        rec = (wmean(u["TS"]) - wmean(u["TR"])) / g["diff"] if g["diff"] else None
        rel = "equivalent" if abs(f["diff"]) <= f["floor"] else ("below" if f["diff"] < 0 else "above")
        world[w] = {"size": v["record"]["size"], "means": {a: wmean(x) for a, x in u.items()},
                    "g": g, "f": f, "t": t, "gap": gap, "rec": rec, "rel": rel}

    def primary(size: int, src: dict) -> dict:
        ws = [x for x in src.values() if x["size"] == size]
        gw = [x for x in ws if x["gap"]]
        recs = [x["rec"] for x in gw]
        strong = sum(1 for r in recs if r >= 2 / 3)
        med = statistics.median(recs) if recs else None
        if len(gw) < 5:
            rd = "no_gap"
        elif strong * 3 >= 2 * len(gw):
            rd = "decoder"
        elif med >= 1 / 3:
            rd = "decoder_partial"
        else:
            rd = "not_decoder"
        return {"reading": rd, "gap_worlds": len(gw), "rec_strong_worlds": strong, "median_rec": med}

    def fidelity(size: int, src: dict) -> dict:
        ws = [x for x in src.values() if x["size"] == size]
        n = len(ws)
        c = {k: sum(1 for x in ws if x["rel"] == k) for k in ("equivalent", "below", "above")}
        loss, beyond = c["below"] * 3 >= n, c["above"] * 3 >= n
        if c["equivalent"] * 3 >= 2 * n:
            rd = "faithful"
        elif loss == beyond:
            rd = "mixed"
        else:
            rd = "fidelity_loss" if loss else "beyond_target"
        return {"reading": rd, **c}

    reading = json.loads((exp / "reading.json").read_text())
    for size, blk in [(s_, b_) for s_, b_ in ((12, reading), (42, reading["reported_42"])) if s_ in SIZES]:
        p, f = primary(size, world), fidelity(size, world)
        print(f"  [{size}] independent primary {p} | fidelity {f}")
        rp, rf = blk["primary"], blk["fidelity"]
        check(p["reading"] == rp["reading"] and p["gap_worlds"] == rp["gap_worlds"]
              and p["rec_strong_worlds"] == rp["rec_strong_worlds"] and close(p["median_rec"], rp["median_rec"]),
              f"[{size}] primary from raw rows = reading.json ({rp['reading']})")
        check(f["reading"] == rf["reading"] and all(f[k] == rf[k] for k in ("equivalent", "below", "above")),
              f"[{size}] fidelity from raw rows = reading.json ({rf['reading']})")

    # dispersion from raw rows (TS, 12-producer worlds)
    def disp_vals(arm: str) -> dict[str, dict[str, float | None]]:
        cols = {"history": ("trace_agent_history_dispersion", "trace_agent_history_cells"),
                "share": ("trace_routed_share_dispersion", "trace_routed_share_pairs"), **ALTS}
        out: dict[str, dict[str, float | None]] = {}
        for w in sample:
            out[w] = {}
            for name, (vc, nc) in cols.items():
                u: dict[int, dict[int, float]] = {}
                for d in ep:
                    if d["world"] == w and d["arm"] == arm and fnum(d.get(nc)) and fnum(d[nc]) > 0 \
                            and d.get(vc) not in (None, ""):
                        u.setdefault(int(d["episode"]), {})[int(d["replicate"])] = float(d[vc])
                out[w][name] = wmean(u) if u else None
        return out

    def disp_rule(vals: dict, key: str, moved: str | None = None) -> tuple[str, int, int, list]:
        defined = [w for w, v in vals.items() if v["history"] is not None and v[key] is not None]

        def ok(x: float, line: float) -> bool:
            if moved and abs(x - line) <= BAND:
                return moved == "pass"
            return x >= line
        passing = [w for w in defined if ok(vals[w]["history"], HIST) and ok(vals[w][key], SHARE)]
        rd = "history_differs" if len(passing) * 3 >= 2 * len(defined) and defined else "history_flat"
        return rd, len(defined), len(passing), sorted(set(vals) - set(defined))

    dv_all = disp_vals("TS")
    for size, blk in [(s_, b_) for s_, b_ in ((12, reading["dispersion"]),
                      (42, reading["reported_42"]["dispersion"])) if s_ in SIZES]:
        dv = {w: v for w, v in dv_all.items() if sample[w]["record"]["size"] == size}
        rd, nd, npass, skipped = disp_rule(dv, "share")
        band = sorted(w for w, v in dv.items() if w not in skipped and (
            abs(v["history"] - HIST) <= BAND or abs(v["share"] - SHARE) <= BAND))
        changes = disp_rule(dv, "share", "pass")[0] != disp_rule(dv, "share", "fail")[0]
        print(f"  [{size}] independent dispersion {rd}: {npass}/{nd} pass, skipped {skipped}, "
              f"in band {len(band)}, borderline changes reading {changes}")
        check(rd == blk["reading"] and nd == blk["defined_worlds"] and npass == blk["passing_worlds"]
              and skipped == blk["skipped_worlds"] and band == blk["borderline_worlds"]
              and changes == blk["borderline_changes_reading"],
              f"[{size}] dispersion from raw rows = reading.json ({blk['reading']})")
        for alt in ALTS:
            ar, and_, ap, _ = disp_rule(dv, alt)
            ba = blk["alternatives_reported_only"][alt]
            print(f"      alt {alt}: {ar} {ap}/{and_}")
            check(ar == ba["reading"] and ap == ba["passing_worlds"] and and_ == ba["defined_worlds"],
                  f"[{size}] alternative {alt} = reading.json ({ba['reading']})")

    # 5. reading.json recomputes from decoder_check.csv / fidelity.csv / dispersion.csv alone
    dc = [d for d in rows(exp / "decoder_check.csv") if d["scope"] == "world"]
    fc = rows(exp / "fidelity.csv")
    ds = rows(exp / "dispersion.csv")
    tb = lambda s: s in ("True", "true")  # noqa: E731
    from_csv = {d["world"]: {"size": int(d["size"]), "gap": tb(d["gap"]), "rec": fnum(d["rec_w"]),
                             "rel": None} for d in dc}
    for d in fc:
        from_csv[d["world"]]["rel"] = d["relation"]
    for size, blk in [(s_, b_) for s_, b_ in ((12, reading), (42, reading["reported_42"])) if s_ in SIZES]:
        p, f = primary(size, from_csv), fidelity(size, from_csv)
        check(p["reading"] == blk["primary"]["reading"] and p["gap_worlds"] == blk["primary"]["gap_worlds"]
              and close(p["median_rec"], blk["primary"]["median_rec"]),
              f"[{size}] primary from decoder_check.csv = reading.json")
        check(f["reading"] == blk["fidelity"]["reading"], f"[{size}] fidelity from fidelity.csv = reading.json")
        dv = {d["world"]: {"history": fnum(d["agent_history_dispersion"]),
                           "share": fnum(d["routed_share_dispersion"]),
                           **{k: fnum(d[f"{k}_reported"]) for k in ALTS}}
              for d in ds if d["arm"] == "TS" and int(d["size"]) == size}
        rd, nd, npass, _ = disp_rule(dv, "share")
        check(rd == blk["dispersion"]["reading"] and npass == blk["dispersion"]["passing_worlds"],
              f"[{size}] dispersion from dispersion.csv = reading.json")
        # csv values equal the independent raw-row values
        bad = [w for w, x in world.items() if x["size"] == size and not (
            close(x["g"]["diff"], fnum(next(d for d in dc if d["world"] == w)["G_w"]))
            and close(x["g"]["floor"], fnum(next(d for d in dc if d["world"] == w)["gap_floor"]))
            and close(x["rec"], fnum(next(d for d in dc if d["world"] == w)["rec_w"]))
            and close(x["f"]["floor"], fnum(next(d for d in fc if d["world"] == w)["floor"])))]
        check(not bad, f"[{size}] decoder_check.csv / fidelity.csv per-world G_w, floors, rec_w = raw ({bad})")
        bad = [w for w, v in dv.items() if not all(close(v[k], dv_all[w][k]) for k in v)]
        check(not bad, f"[{size}] dispersion.csv TS values = raw ({bad})")

    # per-world table (12) for the record
    print("\nworld | TR TS MWWk MWW TW RR | G_w floor gap rec_w | TS-MWWk floor rel")
    for w, x in sorted(world.items(), key=lambda kv: (kv[1]["size"], kv[0])):
        m = x["means"]
        print(f"{w} | " + " ".join(f"{m[a]:.1f}" for a in R) +
              f" | {x['g']['diff']:.1f} {x['g']['floor']:.1f} {x['gap']} "
              f"{'' if x['rec'] is None else round(x['rec'], 3)} | {x['f']['diff']:.1f} {x['f']['floor']:.1f} {x['rel']}")
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry"]
    sys.exit(main(Path(args[0]), Path(args[1]), Path(args[2])))
