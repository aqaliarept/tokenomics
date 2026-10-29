"""Task 7.1 verification and summary of an EXP-0053 (PoC-A teacher audit) dir, independent of
`tokenomics.eval.teacher_audit` / `tokenomics.eval.atlas` analysis code (stdlib only; the rules
are re-implemented from design D5 + D12, the arm table from the roster v2 file).

Checks: contract_failures.csv header-only; the sample equals D5's step-2 rule recomputed from
atlas_v2.json (4 hard + 4 open, train split, 12 producers, patterns cyclic / iid / drift / burst /
seasonal, open = best 2 shocked + best 2 unshocked by room, hard by gain) at the confirm's first
episode seed; every planned (world-episode, subject, replicate) row present once (14 blind /
reach rows, `teacher/floor`, `teacher/d2`, and `bound/ub` exactly on the worlds with shocks or a
stochastic pattern); one demand + shock digest pair per world-episode; `bound/ub` rows flagged
hidden information and never `teacher`; the blind / reach rows reproduce EXP-0052's units for
the same world, episode, subject and replicate; `teacher/floor` units = `round_robin` units;
0 B2 violations by any played arm; every ratios.csv field recomputed from episodes.csv (bar =
best blind arm mean, reach = best reach arm mean, C = max(min(B2, B_lp, B_lp_base or B_base),
best played baseline row), D5 readings, 1.05 / 1.10 reference ratios); the per-episode
unresolved fraction and acceptance rate recomputed from teacher_decisions.csv (binary
acceptance, ADR-0056); spec.json pins (atlas hash, orderer SHA-256, run commit, reduction);
staged files' SHA-256 when given.

usage (from /srv/research/tokenomics):
  python3 openspec/changes/add-m3-development-audit/probes/exp0053_check.py \
      experiments/EXP-0053-teacher-on-mined-worlds \
      --atlas experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json \
      --confirm experiments/EXP-0052-headroom-atlas-confirm \
      [--ckpt orderer=PATH] [--json OUT]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[4]
ROSTER = ROOT / "tokenomics/tokenomics/eval/registry/roster_v2.json"
EP = "entity_planner"
RUN_COMMIT = "6abb218d3c2c04b9ab6331c07bfdf73239abd06a"
ATLAS_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
ORDERER_SHA = "d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c"
PATTERNS = {"cyclic", "iid", "drift", "burst", "seasonal"}
DETERMINISTIC_PATTERNS = {"cyclic"}
FLOOR, D2, UB = "teacher/floor", "teacher/d2", "bound/ub"
MARGIN = {False: 1.05, True: 1.10}
TOL = 0.01


def num(x: object) -> float | None:
    return None if x in (None, "", "None") else float(x)  # type: ignore[arg-type]


def flag(x: object) -> bool | None:
    return None if x in (None, "", "None") else str(x).strip().lower() in ("true", "1")


def read(p: Path) -> list[dict[str, str]]:
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def close(a: object, b: object) -> bool:
    fa, fb = (a if isinstance(a, float) else num(a)), (b if isinstance(b, float) else num(b))
    if fa is None or fb is None:
        return fa is None and fb is None
    return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def first_max(means: dict[str, float], order: list[str]) -> str | None:
    best = None
    for s in order:
        if s in means and (best is None or means[s] > means[best]):
            best = s
    return best


def arm_table() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """(subject, replicates) for the D12 blind arms (roster v2 in file order, `random` R = 2;
    `round_robin`; `random_whole` R = 2) and reach arms (`makeable_split`,
    `makeable_weighted_whole` R = 2, `trained_orderer`), as the confirm stage played them."""
    members = json.loads(ROSTER.read_text())["members"]
    blind = []
    for m in members:
        s = m if isinstance(m, str) else f"{m['producing']}/{m['ordering']}"
        blind.append((s, 2 if s.endswith("/random") else 1))
    blind += [(f"{EP}/round_robin", 1), (f"{EP}/random_whole", 2)]
    reach = [(f"{EP}/makeable_split", 1), (f"{EP}/makeable_weighted_whole", 2),
             (f"{EP}/trained_orderer", 1)]
    return blind, reach


def d5_step2(atlas: dict) -> list[dict]:
    """D5 sample, reduction step 2: 4 hard (by gain) + 4 open (by room), each class half
    shocked (best n // 2 shocked, best n - n // 2 unshocked, shortfalls filled by the best
    remaining), train split, 12 producers, allowed patterns."""
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
        out += sorted(pick, key=lambda w: (-w[key], w["record"]["world_id"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--confirm", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            fails.append(msg)

    exp = a.exp
    ep_path = exp / "artifacts/episodes.csv"
    if not ep_path.exists():
        ep_path = exp / "episodes.csv"
    rows = read(ep_path)
    cf = read(exp / "contract_failures.csv")
    check(not cf, f"contract_failures.csv has {len(cf)} rows")
    spec = json.loads((exp / "spec.json").read_text())
    spec_s = json.dumps(spec)
    check(RUN_COMMIT in spec_s, "run commit not in spec.json")
    check(ATLAS_HASH in spec_s, "atlas content hash not in spec.json")
    check(ORDERER_SHA in spec_s, "orderer SHA-256 pin not in spec.json")
    check("step2" in spec_s, "reduction step2 not recorded in spec.json")
    atlas = json.loads(a.atlas.read_text())
    check(atlas["content_hash"] == ATLAS_HASH, "atlas file content_hash differs")
    for c in a.ckpt:
        name, path = c.split("=", 1)
        check(name != "orderer" or sha(Path(path)) == ORDERER_SHA, f"{name} sha256 differs")

    # sample
    sample = d5_step2(atlas)
    entries = {w["record"]["world_id"]: w for w in sample}
    worlds_run = sorted({r["world"] for r in rows})
    check(sorted(entries) == worlds_run, f"sample {sorted(entries)} != run worlds {worlds_run}")
    for w in sample:
        check(w["split"] == "train" and w["record"]["size"] == 12, f"{w} not train / 12")
    conf = read(a.confirm / "artifacts/episodes.csv")
    conf_eps: dict[str, list[int]] = defaultdict(list)
    for r in conf:
        e = int(r["episode"])
        if e not in conf_eps[r["world"]]:
            conf_eps[r["world"]].append(e)
    first_seed = {w: conf_eps[w][0] for w in entries}

    # planned rows
    blind, reach = arm_table()
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["world"], int(r["episode"]), r["subject"])].append(r)
    for (w, e, s), rs in by.items():
        reps = Counter(r["replicate"] for r in rs)
        check(all(v == 1 for v in reps.values()), f"duplicate replicate {w} {e} {s}")
    ub_worlds = []
    for wid, w in entries.items():
        lv = w["record"]["levers"]
        eps = sorted({int(r["episode"]) for r in rows if r["world"] == wid})
        check(eps == [first_seed[wid]], f"{wid} episodes {eps} != first confirm seed")
        stochastic = lv["shocks"] or lv["pattern"] not in DETERMINISTIC_PATTERNS
        if stochastic:
            ub_worlds.append(wid)
        plan = blind + reach + [(FLOOR, 1), (D2, 1)] + ([(UB, 1)] if stochastic else [])
        subj = {r["subject"] for r in rows if r["world"] == wid}
        check(subj == {s for s, _ in plan}, f"{wid} subjects {sorted(subj ^ {s for s, _ in plan})}")
        for s, n in plan:
            check(len(by[(wid, eps[0], s)]) == n, f"{wid} {s}: {len(by[(wid, eps[0], s)])} rows")
        dig = {(r["demand_digest"], r["shock_digest"]) for r in rows if r["world"] == wid}
        check(len(dig) == 1, f"{wid}: {len(dig)} digest pairs")
    n_planned = len(entries) * (sum(n for _, n in blind + reach) + 2) + len(ub_worlds)
    check(len(rows) == n_planned, f"{len(rows)} rows, planned {n_planned}")
    for r in rows:
        hid = flag(r.get("hidden_information"))
        if r["subject"] == UB:
            check(hid is True and r.get("side") != "teacher", f"ub row not flagged {r['world']}")
        elif r["subject"] in (FLOOR, D2):
            check(hid is not True, f"teacher row flagged hidden {r['world']}")

    # reproduction against EXP-0052
    conf_units = {(r["world"], int(r["episode"]), r["subject"], r["replicate"]): r["units"]
                  for r in conf}
    repro_n, repro_bad = 0, []
    for r in rows:
        if r["subject"] in (FLOOR, D2, UB):
            continue
        k = (r["world"], int(r["episode"]), r["subject"], r["replicate"])
        repro_n += 1
        if conf_units.get(k) != r["units"]:
            repro_bad.append((k, r["units"], conf_units.get(k)))
    check(not repro_bad, f"{len(repro_bad)} blind/reach rows differ from EXP-0052: {repro_bad[:3]}")
    for wid in entries:
        e = first_seed[wid]
        fl = by[(wid, e, FLOOR)][0]["units"]
        rr = by[(wid, e, f"{EP}/round_robin")][0]["units"]
        check(fl == rr, f"{wid} floor {fl} != round_robin {rr}")

    # bounds
    for r in rows:
        b2 = num(r.get("bound_b2"))
        if b2 is not None and r["subject"] != UB:
            check(float(r["units"]) <= b2 + TOL, f"B2 violation {r['world']} {r['subject']}")

    # ratios
    dec_path = exp / "artifacts/teacher_decisions.csv"
    dec = read(dec_path if dec_path.exists() else exp / "teacher_decisions.csv")
    ratios = {(r["world"], int(r["episode"])): r for r in read(exp / "ratios.csv")}
    check(len(ratios) == len(entries), f"ratios.csv {len(ratios)} rows")
    order_b = [s for s, _ in blind]
    order_r = [s for s, _ in reach]
    summary = []
    for wid, w in entries.items():
        e = first_seed[wid]
        shocked = bool(w["record"]["levers"]["shocks"])
        nf = w["noise_floor"]

        def mean(s: str) -> float | None:
            rs = by.get((wid, e, s), [])
            return fmean(float(r["units"]) for r in rs) if rs else None

        bm = {s: mean(s) for s in order_b}
        rm = {s: mean(s) for s in order_r}
        bar_arm = first_max({k: v for k, v in bm.items() if v is not None}, order_b)
        reach_arm = first_max({k: v for k, v in rm.items() if v is not None}, order_r)
        bar, rch = bm[bar_arm], rm[reach_arm]
        base_rows = [r for s in order_b + order_r for r in by[(wid, e, s)]]
        best = max(float(r["units"]) for r in base_rows)
        r0 = base_rows[0]
        cap = num(r0.get("bound_lp_base"))
        cap = cap if cap is not None else num(r0.get("bound_base"))
        bounds = [b for b in (num(r0.get("bound_b2")), num(r0.get("bound_lp")), cap)
                  if b is not None]
        ceiling = max(min(bounds), best) if bounds else best
        fl, d2, ub = mean(FLOOR), mean(D2), mean(UB)
        d2_beats = d2 - bar > nf
        ub_beats = ub is not None and ub - bar > nf
        if w["confirm_class"] == "open":
            reading = ("search_headroom" if d2_beats else "fork_limited" if ub_beats
                       else "ceiling_limited")
        else:
            reading = "beyond_trained" if d2 >= rch else "candidate_or_depth_limited"
        drs = [d for d in dec if d["world"] == wid and int(d["episode"]) == e
               and d["arm"] == "d2"]
        posterior = [d for d in drs if num(d.get("acceptance_rate")) is not None]
        unres = [d for d in drs if flag(d["unresolved"])]
        uf = len(unres) / len(drs) if drs else None
        acc_vals = sorted({float(d["acceptance_rate"]) for d in posterior})
        mar = fmean(float(d["acceptance_rate"]) for d in posterior) if posterior else None
        d2row = by[(wid, e, D2)][0]
        dead = {s: flag(by[(wid, e, s)][0]["dead"]) if by.get((wid, e, s)) else None
                for s in (FLOOR, D2, UB)}
        mine = {"bar_arm": bar_arm, "bar": bar, "reach_arm": reach_arm, "reach": rch,
                "ceiling": ceiling, "floor_units": fl, "d2_units": d2,
                "d2_minus_bar": d2 - bar, "d2_over_bar": d2 / bar,
                "d2_minus_reach": d2 - rch, "d2_over_ceiling": d2 / ceiling,
                "d2_room_closed": (d2 - bar) / (ceiling - bar) if ceiling > bar else None,
                "d2_beats_bar": d2_beats, "d2_ge_reach": d2 >= rch, "d2_below_bar": d2 < bar,
                "gate_margin_ref": MARGIN[shocked],
                "d2_clears_margin_ref": d2 >= MARGIN[shocked] * bar,
                "d2_unresolved_fraction": uf, "reading": reading,
                "noise_floor": nf, "clairvoyant_ub_units": ub,
                "clairvoyant_ub_beats_bar": ub_beats if ub is not None else None}
        rr = ratios.get((wid, e), {})
        for k, v in mine.items():
            got = rr.get(k)
            if isinstance(v, bool) or v is None:
                ok = (flag(got) == v) if v is not None else got in (None, "", "None")
            elif isinstance(v, str):
                ok = got == v
            else:
                ok = close(v, got)
            if k == "bar_arm" and not ok and close(bm.get(got, math.nan) or math.nan, bar):
                ok = True  # tie between blind arms
            check(ok, f"ratios {wid} {k}: mine {v} vs {got}")
        check(close(num(d2row.get("unresolved_fraction")), uf),
              f"{wid} unresolved_fraction {d2row.get('unresolved_fraction')} vs {uf}")
        check(all(v in (0.0, 1.0) for v in acc_vals) or not posterior,
              f"{wid} acceptance not binary: {acc_vals[:5]}")
        summary.append({"world": wid, "class": w["confirm_class"], "shocks": shocked,
                        "pattern": w["record"]["levers"]["pattern"],
                        "choice": w["record"]["levers"]["choice"], "episode": e,
                        "noise_floor": round(nf, 1), "bar_arm": bar_arm, "bar": bar,
                        "reach_arm": reach_arm, "reach": rch, "ceiling": ceiling,
                        "b2": num(r0.get("bound_b2")), "floor": fl, "d2": d2, "ub": ub,
                        "d2_minus_bar": d2 - bar, "d2_over_bar": round(d2 / bar, 4),
                        "ub_minus_bar": ub - bar if ub is not None else None,
                        "room_closed": mine["d2_room_closed"], "reading": reading,
                        "decisions": len(drs), "posterior_decisions": len(posterior),
                        "unresolved": len(unres), "mean_acceptance": mar,
                        "dead": dead, "clears_ref": mine["d2_clears_margin_ref"]})

    opens = [s for s in summary if s["class"] == "open"]
    hards = [s for s in summary if s["class"] == "hard"]
    beats = sum(s["reading"] == "search_headroom" for s in opens)
    out = {"rows": len(rows), "planned": n_planned, "reproduced_rows": repro_n,
           "ub_worlds": ub_worlds, "worlds": summary,
           "open": {"n": len(opens), "d2_beats_bar": beats,
                    "search_headroom": beats * 2 >= len(opens),
                    "readings": dict(Counter(s["reading"] for s in opens))},
           "hard": {"n": len(hards), "readings": dict(Counter(s["reading"] for s in hards))},
           "decisions": sum(s["decisions"] for s in summary),
           "unresolved": sum(s["unresolved"] for s in summary),
           "failures": fails}
    for s in summary:
        print(f"{s['world'][-4:]} {s['class']:4} shk={int(s['shocks'])} {s['pattern']:8} "
              f"nf={s['noise_floor']:6} bar={s['bar']:7.1f} reach={s['reach']:7.1f} "
              f"C={s['ceiling']:7.1f} floor={s['floor']:6.0f} d2={s['d2']:6.0f} "
              f"ub={s['ub'] if s['ub'] is None else round(s['ub'])} "
              f"d2-bar={s['d2_minus_bar']:+7.1f} d2/bar={s['d2_over_bar']:.3f} "
              f"unres={s['unresolved']}/{s['decisions']} {s['reading']}")
    print(json.dumps({k: v for k, v in out.items() if k != "worlds"}, default=str))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
