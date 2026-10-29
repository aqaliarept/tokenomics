"""Task 9.1 verification and summary of an EXP-0054 (PoC-C role attribution) dir, independent of
`tokenomics.eval.teacher_audit` analysis code (stdlib only; D6's arithmetic and the sample rule
are re-implemented from design D6 + D5).

Checks: contract_failures.csv header-only; the sample equals the shocked worlds of D5's step-2
sample recomputed from atlas_v2.json, each at EXP-0053's episode seed for that world; every
planned row (3 world-episodes x subjects S / A / B / T, replicate 0) present exactly once, no
other row; one demand + shock digest pair per world-episode, equal to EXP-0053's for the same
world-episode; B (`entity_planner/trained_orderer`) reproduces EXP-0053's units for the same
subject (same arm, same world-episode); T is compared with EXP-0053's `teacher/d2` and the
spec's `t_reproduction` recomputed; attribution.csv (size row + one row per world) recomputed
from episodes.csv: G = sum(T - S), c_O = sum(A - S) / G, c_P = sum(B - S) / G, interaction
T - A - B + S, tie band 0.15, G <= 0 undefined; spec.json pins (run commit, checkpoint SHA-256,
atlas content hash + file SHA-256, horizon 100, not a dry run, no filters, T not reused);
staged files' SHA-256 when given; a decisions file per A / T job.

usage (from /srv/research/tokenomics):
  python3 openspec/changes/add-m3-development-audit/probes/exp0054_check.py \
      experiments/EXP-0054-role-attribution-mined \
      --atlas experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json \
      --exp0053 experiments/EXP-0053-teacher-on-mined-worlds \
      [--ckpt producer=PATH --ckpt orderer=PATH --ckpt atlas=PATH] [--json OUT]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

RUN_COMMIT = "4e96255becbef380c0ba9ea515a2afd525314c04"
ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
ATLAS_SHA = "7517314f297ac39e1c00da3fb734bb40fcfecc9492ee2e67936fc910f8cf9568"
PINS = {"producer": "8b51b77cb5ac763114120d3bcbcbc9cd2c566362fe6e599d67e8c8ac883e36ea",
        "orderer": "d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c",
        "atlas": ATLAS_SHA}
PATTERNS = {"cyclic", "iid", "drift", "burst", "seasonal"}
ROLES = {"S": "portable", "A": "teacher/d2_portable", "B": "entity_planner/trained_orderer",
         "T": "teacher/d2"}
TEACHER_ROLES = ("A", "T")
BAND = 0.15


def num(x: object) -> float | None:
    return None if x in (None, "", "None") else float(x)  # type: ignore[arg-type]


def read(p: Path) -> list[dict[str, str]]:
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def close(a: object, b: object) -> bool:
    fa, fb = (a if isinstance(a, float) else num(a)), (b if isinstance(b, float) else num(b))
    if fa is None or fb is None:
        return fa is None and fb is None
    return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def d5_step2_shocked(atlas: dict) -> list[str]:
    """Shocked worlds of D5's reduction-step-2 sample (4 hard by gain + 4 open by room, each
    class best n // 2 shocked + best n - n // 2 unshocked, shortfalls filled by the best
    remaining; train split, 12 producers, allowed patterns)."""
    pool = [w for w in atlas["worlds"] if w["split"] == "train" and w["record"]["size"] == 12
            and w["record"]["levers"]["pattern"] in PATTERNS]
    out = []
    for cls, key in (("hard", "gain"), ("open", "room")):
        ws = sorted([w for w in pool if w["confirm_class"] == cls],
                    key=lambda w: (-w[key], w["record"]["world_id"]))
        n = min(4, len(ws))
        shocked = [w for w in ws if w["record"]["levers"]["shocks"]]
        calm = [w for w in ws if not w["record"]["levers"]["shocks"]]
        pick = shocked[: n // 2] + calm[: n - n // 2]
        pick += [w for w in ws if w not in pick][: n - len(pick)]
        out += [w["record"]["world_id"] for w in sorted(pick, key=lambda w: (-w[key], w["record"]["world_id"]))
                if w["record"]["levers"]["shocks"]]
    return out


def attribution(units: dict[str, dict[str, float]]) -> dict:
    """D6 arithmetic over world -> role -> mean units (worlds with all four roles only)."""
    full = sorted(w for w, u in units.items() if all(r in u for r in ROLES))
    tot = {r: sum(units[w][r] for w in full) for r in ROLES}
    s, a, b, t = tot["S"], tot["A"], tot["B"], tot["T"]
    g = t - s
    res = {"worlds": len(full), **tot, "G": g, "interaction": t - a - b + s,
           "c_O": None, "c_P": None, "closure_difference": None, "reading": "undefined"}
    if full and g > 0:
        c_o, c_p = (a - s) / g, (b - s) / g
        res.update(c_O=c_o, c_P=c_p, closure_difference=c_o - c_p,
                   reading="tie" if abs(c_o - c_p) < BAND
                   else "orderer" if c_o > c_p else "producer")
    elif not full:
        res["reading"] = "incomplete"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--exp0053", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            fails.append(msg)

    exp = a.exp
    ep_path = next(p for p in (exp / "artifacts" / "episodes.csv", exp / "episodes.csv")
                   if p.exists())
    rows = read(ep_path)
    check(read(exp / "contract_failures.csv") == [], "contract_failures.csv has rows")
    spec = json.loads((exp / "spec.json").read_text())
    pins = spec["pins"]
    atlas = json.loads(a.atlas.read_text())

    # sample: shocked worlds of D5 step 2, at EXP-0053's seed
    sample = d5_step2_shocked(atlas)
    check(len(sample) == 3, f"expected 3 shocked sample worlds, got {sample}")
    ref53 = read(a.exp0053 / "artifacts" / "episodes.csv")
    seed53 = {}
    for r in ref53:
        seed53.setdefault(r["world"], set()).add(int(r["episode"]))
    planned = {(w, ep) for w in sample for ep in seed53.get(w, ())}
    check(all(len(seed53.get(w, ())) == 1 for w in sample), "EXP-0053 not 1 episode per world")
    grid = {(w, ep) for w, g in pins["grid"].items() for ep in g["episodes"]}
    check(grid == planned, f"grid {sorted(grid)} != planned {sorted(planned)}")
    check([s["world"] for s in pins["sample"]] == sample, "spec sample order/worlds differ")
    check(all(s["shocks"] for s in pins["sample"]), "an unshocked world in the sample")

    # rows: each planned (world, episode, subject, replicate 0) once, nothing else
    keys = [(r["world"], int(r["episode"]), r["subject"], int(r["replicate"])) for r in rows]
    want = {(w, ep, s, 0) for (w, ep) in planned for s in ROLES.values()}
    check(len(keys) == len(set(keys)), "duplicate episode rows")
    check(set(keys) == want, f"rows differ: missing {sorted(want - set(keys))}, "
                             f"extra {sorted(set(keys) - want)}")

    # digests: one pair per world-episode, equal to EXP-0053's
    pairs: dict[tuple[str, int], set] = {}
    for r in rows:
        pairs.setdefault((r["world"], int(r["episode"])), set()).add(
            (r["demand_digest"], r["shock_digest"]))
    pairs53: dict[tuple[str, int], set] = {}
    for r in ref53:
        pairs53.setdefault((r["world"], int(r["episode"])), set()).add(
            (r["demand_digest"], r["shock_digest"]))
    for k, v in pairs.items():
        check(len(v) == 1, f"{k}: {len(v)} digest pairs")
        check(v == pairs53.get(k), f"{k}: digests differ from EXP-0053")
        check(all(s for _, s in v), f"{k}: empty shock digest on a shocked world")

    units = {(r["world"], int(r["episode"]), r["subject"]): float(r["units"]) for r in rows}
    u53 = {(r["world"], int(r["episode"]), r["subject"]): float(r["units"])
           for r in ref53 if r["replicate"] in ("0", "")}
    deaths = [k for k, r in zip(keys, rows) if r["dead"].lower() == "true"]

    # B = EXP-0053's entity_planner/trained_orderer (same arm, same world-episode)
    b_repro = {}
    for (w, ep) in sorted(planned):
        k = (w, ep, ROLES["B"])
        b_repro[w] = (units.get(k), u53.get(k))
        check(units.get(k) == u53.get(k), f"{w}: B {units.get(k)} != EXP-0053 {u53.get(k)}")

    # T against EXP-0053 teacher/d2; spec t_reproduction recomputed
    t_rep = []
    for (w, ep) in sorted(planned):
        rerun, src = units.get((w, ep, ROLES["T"])), u53.get((w, ep, "teacher/d2"))
        t_rep.append({"world": w, "episode": ep, "rerun": rerun, "source": src,
                      "equal": rerun == src})
    spec_rep = sorted(spec.get("t_reproduction", []), key=lambda d: d["world"])
    check(len(spec_rep) == len(t_rep) and all(
        d["world"] == e["world"] and d["episode"] == e["episode"] and close(d["rerun"], e["rerun"])
        and close(d["source"], e["source"]) and d["equal"] == e["equal"]
        for d, e in zip(spec_rep, t_rep)), "spec t_reproduction does not recompute")

    # attribution.csv recomputed
    wu: dict[str, dict[str, float]] = {}
    by_sub = {s: r for r, s in ROLES.items()}
    for (w, ep, s), u in units.items():
        wu.setdefault(w, {})[by_sub[s]] = u  # one episode, one replicate per world
    att = read(exp / "attribution.csv")
    mine = [("size", "", attribution(wu))] + [("world", w, attribution({w: wu[w]}))
                                              for w in sorted(wu)]
    check(len(att) == len(mine), f"attribution.csv has {len(att)} rows, expected {len(mine)}")
    for row, (scope, w, m) in zip(att, mine):
        check(row["scope"] == scope and row["world"] == w and row["size"] == "12",
              f"attribution row order: {row['scope']}/{row['world']}")
        for f in ("worlds", "S", "A", "B", "T", "G", "c_O", "c_P", "interaction",
                  "closure_difference"):
            check(close(row[f], m[f] if m[f] is None else float(m[f])),
                  f"attribution {scope}/{w} {f}: {row[f]} != {m[f]}")
        check(row["reading"] == m["reading"], f"attribution {scope}/{w} reading")
        check(close(row["band"], BAND), "band")

    # pins
    src = spec.get("source", {})
    check(pins["source"]["commit"] == RUN_COMMIT and src.get("commit") == RUN_COMMIT,
          "run commit pin")
    check(pins["checkpoints"] == {"producer": PINS["producer"], "orderer": PINS["orderer"]},
          "checkpoint pins")
    check(pins["atlas"]["content_hash"] == ATLAS_HASH and pins["atlas"]["sha256"] == ATLAS_SHA,
          "atlas pins")
    check(pins["horizon"] == 100 and pins["dry_run"] is False and pins["poc"] == "c",
          "horizon / dry_run / poc")
    check(spec.get("kind") == "audit", "spec kind")
    check(pins["shock_profile"] is None and pins["max_search_actions"] is None,
          "filters in a real run")
    check(pins["role_arms"] == ["S", "A", "B", "T"], "role arms")
    check(pins["t_reuse"]["reused"] is False, "T reused (expected rerun)")
    check(spec["t_reuse"]["registered"] != spec["t_reuse"]["fingerprint"]
          and spec["t_reuse"]["problems"] != [], "T rerun without a recorded reason")
    for kv in a.ckpt:
        name, path = kv.split("=", 1)
        check(sha(Path(path)) == PINS[name], f"staged {name} SHA-256 != pin")

    # decisions file per teacher job
    for (w, ep) in planned:
        for r in TEACHER_ROLES:
            f = exp / "artifacts" / "decisions" / f"{w}__{ep}__{ROLES[r].replace('/', '_')}.csv"
            check(f.exists() and f.stat().st_size > 0, f"missing decisions {f.name}")

    summary = {"rows": len(rows), "planned": len(want), "deaths": deaths,
               "units": {w: wu[w] for w in sorted(wu)}, "attribution": mine[0][2],
               "per_world": {w: m for _, w, m in mine[1:]}, "b_vs_exp0053": b_repro,
               "t_reproduction": t_rep, "fails": fails}
    print(json.dumps(summary, indent=1, default=str))
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1, default=str))
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
