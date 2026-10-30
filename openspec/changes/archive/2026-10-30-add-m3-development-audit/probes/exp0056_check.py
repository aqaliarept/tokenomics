"""Task 11.3 verification and summary of an EXP-0056 (DIAGNOSTICS 19 split check) dir,
independent of `tokenomics.eval.teacher_audit` analysis code (stdlib only; D13's arithmetic and
reading are re-implemented from design D13).

Checks: contract_failures.csv header-only; the sample = the 3 shocked world-episodes of EXP-0054
(same world + episode seed); every planned row (3 world-episodes x TR / TW / RS / RR replicate 0,
RW replicates 0 and 1) present exactly once, no other row; one demand + shock digest pair per
world-episode, equal to EXP-0053's and EXP-0054's; split_check.csv recomputed from episodes.csv
(per world: gap RR - TR, gains TW - TR / RS - TR and their shares, RR - RS, RR - TW, interaction,
recovery = gap > floor and gain > floor and gain >= 2/3 gap, joint; >= 2 of 3 worlds; floor =
atlas floor_se x sigma_ep of the world's size/choice cell, from atlas_v2.json); manipulation.csv
recomputed (per world x arm, mean over replicates, minus TR's); reproduction of TR / RR / RW
against EXP-0053 and TR against EXP-0054 recomputed and compared with spec.json `reproduction`;
spec.json pins (run commit, orderer SHA-256, atlas content hash, horizon 100, not a dry run);
staged files' SHA-256 when given.

usage (from /srv/research/tokenomics):
  python3 openspec/changes/add-m3-development-audit/probes/exp0056_check.py \
      experiments/EXP-0056-split-vs-rotation \
      --atlas experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json \
      --exp0053 experiments/EXP-0053-teacher-on-mined-worlds \
      --exp0054 experiments/EXP-0054-role-attribution-mined \
      [--ckpt orderer=PATH --ckpt atlas=PATH] [--json OUT]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

RUN_COMMIT = "40dc1873ab9fd7a03ebc028eca1b5f2142331088"
ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
PINS = {"orderer": "d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c",
        "atlas": "7517314f297ac39e1c00da3fb734bb40fcfecc9492ee2e67936fc910f8cf9568"}
ARMS = {"TR": "entity_planner/trained_orderer", "TW": "entity_planner/trained_whole",
        "RS": "entity_planner/trained_split_rotating", "RR": "entity_planner/round_robin",
        "RW": "entity_planner/random_whole"}
REPLICATES = {"TR": (0,), "TW": (0,), "RS": (0,), "RR": (0,), "RW": (0, 1)}
X, MIN_WORLDS = 2 / 3, 2
MEASURES = ("trace_split_share_consumer", "trace_split_degree_consumer",
            "lead_concurrent_top_share_consumer", "lead_receiver_change_share_consumer")


def num(x: object) -> float | None:
    return None if x in (None, "", "None") else float(x)  # type: ignore[arg-type]


def flag(x: object) -> bool:
    return str(x) in ("True", "true", "1")


def read(p: Path) -> list[dict[str, str]]:
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def close(a: object, b: object) -> bool:
    fa = a if isinstance(a, float) else num(a)
    fb = b if isinstance(b, float) else num(b)
    if fa is None or fb is None:
        return fa is None and fb is None
    return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))


def episodes(d: Path) -> list[dict[str, str]]:
    p = d / "artifacts" / "episodes.csv"
    return read(p if p.exists() else d / "episodes.csv")


def canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "content_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def floors(atlas: dict, worlds: set[str]) -> dict[str, float]:
    out = {}
    for w in atlas["worlds"]:
        rec = w["record"]
        if rec["world_id"] in worlds:
            cell = atlas["spreads"][f"{rec['size']}/{rec['levers']['choice']}"]
            out[rec["world_id"]] = float(atlas["rules"]["floor_se"]) * float(cell["sigma_ep"])
    return out


def per_world(u: dict[str, float], floor: float) -> dict:
    tr, tw, rs, rr = u["TR"], u["TW"], u["RS"], u["RR"]
    gap = rr - tr
    ok = gap > floor

    def rec(g: float) -> bool:
        return ok and g > floor and g >= X * gap

    return {"gap": gap, "gap_clears": ok, "whole_gain": tw - tr, "rotation_gain": rs - tr,
            "whole_recovery": (tw - tr) / gap if gap > 0 else None,
            "rotation_recovery": (rs - tr) / gap if gap > 0 else None,
            "split_cost_at_rotation": rr - rs, "receiver_gain_whole": rr - tw,
            "split_cost_at_rotation_share": (rr - rs) / gap if gap > 0 else None,
            "receiver_gain_whole_share": (rr - tw) / gap if gap > 0 else None,
            "interaction": tr - tw - rs + rr, "whole_recovers": rec(tw - tr),
            "rotation_recovers": rec(rs - tr),
            "joint": ok and not rec(tw - tr) and not rec(rs - tr)}


def reading(rows: list[dict]) -> str:
    n = {k: sum(bool(r[k]) for r in rows)
         for k in ("gap_clears", "whole_recovers", "rotation_recovers", "joint")}
    if n["gap_clears"] < MIN_WORLDS:
        return "no_gap"
    f, b = n["whole_recovers"] >= MIN_WORLDS, n["rotation_recovers"] >= MIN_WORLDS
    if f:
        return "f+b" if b else "f"
    if b:
        return "b"
    return "joint" if n["joint"] >= MIN_WORLDS else "neither"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--exp0053", type=Path, required=True)
    ap.add_argument("--exp0054", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            fails.append(msg)

    cf = (a.exp / "contract_failures.csv").read_text().strip().splitlines()
    check(len(cf) == 1, f"contract_failures.csv header-only ({len(cf) - 1} rows)")

    rows = episodes(a.exp)
    e53, e54 = episodes(a.exp0053), episodes(a.exp0054)
    sample = sorted({(d["world"], int(d["episode"])) for d in e54})
    check(len(sample) == 3, f"EXP-0054 sample has 3 world-episodes: {sample}")
    by_subject = {s: k for k, s in ARMS.items()}
    planned = {(w, e, ARMS[k], r) for w, e in sample for k in ARMS for r in REPLICATES[k]}
    keys = [(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])) for d in rows]
    check(len(keys) == 18 and len(set(keys)) == 18, f"18 distinct rows ({len(keys)} rows)")
    check(set(keys) == planned, "rows = planned grid (3 world-episodes x TR/TW/RS/RR + RW x 2)")
    check(all(d.get("arm") == by_subject.get(d["subject"]) for d in rows),
          "arm column matches subject")

    def digests(ds: list[dict[str, str]]) -> dict[tuple[str, int], set[tuple[str, str]]]:
        out: dict[tuple[str, int], set[tuple[str, str]]] = {}
        for d in ds:
            out.setdefault((d["world"], int(d["episode"])), set()).add(
                (d["demand_digest"], d.get("shock_digest", "")))
        return out

    mine, d53, d54 = digests(rows), digests(e53), digests(e54)
    for where in sample:
        check(len(mine.get(where, set())) == 1, f"{where[0]}: one digest pair within the run")
        check(mine.get(where) == d53.get(where) == d54.get(where),
              f"{where[0]}: digests equal EXP-0053 and EXP-0054")
    check(all(d["dead"] in ("False", "0", "false") for d in rows),
          f"no deaths ({sum(d['dead'] not in ('False', '0', 'false') for d in rows)})")

    # split_check.csv
    atlas = json.loads(a.atlas.read_text())
    check(canonical_hash(atlas) == ATLAS_HASH == atlas["content_hash"], "atlas content hash")
    fl = floors(atlas, {w for w, _ in sample})
    units: dict[str, dict[str, list[float]]] = {}
    for d in rows:
        units.setdefault(d["world"], {}).setdefault(by_subject[d["subject"]], []).append(
            float(d["units"]))
    mean = {w: {k: sum(v) / len(v) for k, v in arms.items()} for w, arms in units.items()}
    mine_rows = {w: per_world(mean[w], fl[w]) for w in sorted(mean)}
    expect = reading(list(mine_rows.values()))
    sc = read(a.exp / "split_check.csv")
    wrows = {r["world"]: r for r in sc if r["scope"] == "world"}
    allrow = next(r for r in sc if r["scope"] == "all")
    check(set(wrows) == set(mine_rows), "split_check.csv: one row per world")
    for w, mr in mine_rows.items():
        r = wrows.get(w, {})
        check(close(r.get("floor"), fl[w]), f"{w}: floor {fl[w]:.2f}")
        for k in ("TR", "TW", "RS", "RR", "RW"):
            check(close(r.get(k), mean[w][k]), f"{w}: {k} units {mean[w][k]}")
        for k, v in mr.items():
            ok = (flag(r.get(k)) == v) if isinstance(v, bool) else close(r.get(k), v)
            check(ok, f"{w}: {k} = {v}")
    for k in ("gap_clears", "whole_recovers", "rotation_recovers", "joint"):
        n = sum(bool(m[k]) for m in mine_rows.values())
        check(int(allrow[k]) == n, f"all: {k} count {n}")
    for k in ("TR", "TW", "RS", "RR", "gap", "whole_gain", "rotation_gain",
              "split_cost_at_rotation", "receiver_gain_whole", "interaction"):
        tot = sum(mean[w][k] if k in mean[w] else mine_rows[w][k] for w in mine_rows)
        check(close(allrow[k], tot), f"all: sum {k} = {tot}")
    check(allrow["reading"] == expect, f"reading recomputed = {expect} (file: {allrow['reading']})")

    # manipulation.csv
    man = read(a.exp / "manipulation.csv")
    acc: dict[tuple[str, str], list[dict[str, str]]] = {}
    for d in rows:
        acc.setdefault((d["world"], by_subject[d["subject"]]), []).append(d)
    mm = {k: {m: sum(float(d[m]) for d in ds) / len(ds) for m in MEASURES}
          for k, ds in acc.items()}
    check(len(man) == len(mm), f"manipulation.csv: {len(mm)} rows")
    for r in man:
        key = (r["world"], r["arm"])
        ok = key in mm and int(r["replicates"]) == len(acc[key]) and all(
            close(r[m], mm[key][m]) and close(r[f"d_{m}"], mm[key][m] - mm[(key[0], "TR")][m])
            for m in MEASURES)
        check(ok, f"manipulation {key} recomputes")

    # reproduction
    spec = json.loads((a.exp / "spec.json").read_text())
    ref_units = {"exp0053": {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])):
                             float(d["units"]) for d in e53},
                 "exp0054": {(d["world"], int(d["episode"]), d["subject"], int(d["replicate"])):
                             float(d["units"]) for d in e54}}
    repro = []
    for d in rows:
        k = (d["world"], int(d["episode"]), d["subject"], int(d["replicate"]))
        for name, ref in ref_units.items():
            if k in ref:
                repro.append({"reference": name, "world": k[0], "subject": k[2],
                              "replicate": k[3], "units": float(d["units"]),
                              "reference_units": ref[k], "equal": float(d["units"]) == ref[k]})
    for r in repro:
        print(f"  repro {r['reference']} {r['world'][-7:]} {r['subject']} r{r['replicate']}: "
              f"{r['units']:.0f} vs {r['reference_units']:.0f} {'=' if r['equal'] else '!='}")
    got = {(name := r["reference"], r["world"], r["subject"], int(r["replicate"])):
           (float(r["units"]), float(r["reference_units"])) for r in spec.get("reproduction", [])}
    mine_r = {(r["reference"], r["world"], r["subject"], r["replicate"]):
              (r["units"], r["reference_units"]) for r in repro}
    check(got == mine_r, f"spec.json reproduction recomputes ({len(mine_r)} pairs)")
    subj53 = {(r["subject"], r["reference"]) for r in repro}
    check({(ARMS[k], "exp0053") for k in ("TR", "RR", "RW")} <= subj53
          and (ARMS["TR"], "exp0054") in subj53,
          "TR / RR / RW against EXP-0053 and TR against EXP-0054 are compared")

    # pins
    blob = json.dumps(spec)
    check(RUN_COMMIT in blob, "spec cites the run commit")
    check(PINS["orderer"] in blob, "spec pins orderer SHA-256")
    check(ATLAS_HASH in blob, "spec pins atlas content hash")
    check(spec.get("horizon", spec.get("pins", {}).get("horizon")) in (100, None)
          and '"dry_run": true' not in blob, "horizon 100, not a dry run")
    for c in a.ckpt:
        name, path = c.split("=", 1)
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        check(sha == PINS[name], f"staged {name} SHA-256 {sha[:12]}")

    out = {"reading": expect, "floors": fl, "units": mean, "worlds": mine_rows,
           "reproduction": repro, "failures": fails}
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
    for w, mr in mine_rows.items():
        print(f"{w}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                   for k, v in mr.items()))
    print(f"reading: {expect}; {len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
