"""Task 5.1 verification and summary of an EXP-0051 (atlas screen) dir, independent of
`tokenomics.eval.atlas`'s analysis code (stdlib only; the rules are re-implemented from design
D12, the arm table from the roster v2 file).

Checks: contract_failures.csv header-only; every planned (world-episode, arm, replicate) row
present once, one demand + shock digest pair per world-episode; 0 B2 violations; the anchor rows
reproduce EXP-0050's units; the provisional spreads (EXP-0050) recomputed and equal to spec.json's;
every worlds.csv measure and class, arms_above_base, and selection.csv (rank, pass, split)
recomputed from episodes.csv; checkpoint sha256 pins. Prints class counts per size and per region
(choice, pattern, shocks, coordinate quartiles), the selection and split, and arms above B_base.

usage (from /srv/research/tokenomics):
  python3 openspec/changes/add-m3-development-audit/probes/exp0051_check.py \
      experiments/EXP-0051-headroom-atlas-screen \
      --exp0050 experiments/EXP-0050-routing-vs-fanout/artifacts/episodes.csv \
      [--ckpt producer=PATH --ckpt orderer=PATH] [--json OUT]
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
from statistics import fmean, median

csv.field_size_limit(1 << 30)
ROOT = Path(__file__).resolve().parents[4]
ROSTER = ROOT / "tokenomics/tokenomics/eval/registry/roster_v2.json"
EP = "entity_planner"
CHOICES = {12: ["sparse_f2", "sparse_f4", "dense"],
           42: ["sparse_f2", "sparse_f4", "sparse_f7", "sparse_f14", "dense"]}
PATTERNS = ["cyclic", "iid", "drift", "burst", "hit", "seasonal", "switch"]
COORDS = ["rho", "sigma", "cv_capacity", "cv_degradation", "cv_cost",
          "shock_events_per_producer", "demand_concentration", "demand_volatility"]
K = {12: 64, 42: 32}
TOL = 0.01


def arms() -> list[dict]:
    """D12 screen arms in table order: roster v2 (random R=1), round_robin, random_whole R=2
    (blind); makeable_split, makeable_weighted_whole R=1, trained_orderer (reach)."""
    roster = json.loads(ROSTER.read_text())["members"]
    out = [{"s": f"{m['producing']}/{m['ordering']}", "side": "blind", "rnd": m["random"],
            "R": 1} for m in roster]
    out += [{"s": f"{EP}/round_robin", "side": "blind", "rnd": False, "R": 1},
            {"s": f"{EP}/random_whole", "side": "blind", "rnd": True, "R": 2},
            {"s": f"{EP}/makeable_split", "side": "reach", "rnd": False, "R": 1},
            {"s": f"{EP}/makeable_weighted_whole", "side": "reach", "rnd": True, "R": 1},
            {"s": f"{EP}/trained_orderer", "side": "reach", "rnd": False, "R": 1}]
    return out


def num(x: object) -> float | None:
    return None if x in (None, "", "None") else float(x)  # type: ignore[arg-type]


def read(p: Path) -> list[dict[str, str]]:
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def values(rows: list[dict[str, str]]) -> dict:
    """world -> episode -> {"b2", "base", "arms": subject -> [row in replicate order]}."""
    out: dict = defaultdict(dict)
    for r in sorted(rows, key=lambda r: (r["world"], int(r["episode"]), r["subject"],
                                         int(r["replicate"]))):
        e = out[r["world"]].setdefault(int(r["episode"]), {"b2": None, "base": None,
                                                          "arms": defaultdict(list)})
        if e["b2"] is None:
            e["b2"] = num(r.get("bound_b2"))
        if e["base"] is None:
            e["base"] = num(r.get("bound_base"))
        e["arms"][r["subject"]].append(r)
    return out


def u(rs: list[dict]) -> list[float]:
    return [float(r["units"]) for r in rs]


def collapsed(r: dict) -> bool:
    tr = r.get("trace_capacity_ratio_trace")
    return r["dead"].lower() == "true" and bool(tr) and len(json.loads(tr)) < 50


def first_max(means: dict[str, float], order: list[str]) -> str | None:
    best = None
    for s in order:
        if s in means and (best is None or means[s] > means[best]):
            best = s
    return best


def measure(eps: dict, table: list[dict]) -> dict:
    keys = sorted(eps)
    present = [a for a in table if all(a["s"] in eps[e]["arms"] for e in keys)]
    means = {a["s"]: fmean(fmean(u(eps[e]["arms"][a["s"]])) for e in keys) for a in present}
    order = [a["s"] for a in present]
    side = {a["s"]: a["side"] for a in present}
    rnd = {a["s"]: a["rnd"] for a in present}
    bar_arm = first_max({s: v for s, v in means.items() if side[s] == "blind"}, order)
    reach_arm = first_max({s: v for s, v in means.items() if side[s] == "reach"}, order)
    d, d_reps, stoch, rss, rdf = [], [], {}, 0.0, 0
    if bar_arm and reach_arm:
        for e in keys:
            ru, bu = u(eps[e]["arms"][reach_arm]), u(eps[e]["arms"][bar_arm])
            d.append(fmean(ru) - fmean(bu))
            d_reps.append([ru[i % len(ru)] - bu[i % len(bu)] for i in range(max(len(ru),
                                                                                len(bu)))])
        for s in (bar_arm, reach_arm):
            if rnd[s]:
                stoch[s] = min(len(eps[e]["arms"][s]) for e in keys)
                for e in keys:
                    x = u(eps[e]["arms"][s])
                    if len(x) >= 2:
                        m = fmean(x)
                        rss += sum((v - m) ** 2 for v in x)
                        rdf += len(x) - 1
    ceil, fills, above = [], [], Counter()
    for e in keys:
        ep = eps[e]
        best = max(v for rs in ep["arms"].values() for v in u(rs))
        bs = [b for b in (ep["b2"], ep["base"]) if b is not None]
        ceil.append(max(min(bs), best) if bs else best)
        if ep["base"] is not None:
            for s, rs in ep["arms"].items():
                if max(u(rs)) > ep["base"] + TOL:
                    above[s] += 1
        if bar_arm:
            for r in ep["arms"][bar_arm]:
                o, f = num(r.get("ordered_units")), num(r.get("fulfilled_units"))
                if o and f is not None:
                    fills.append(f / o)
    c = fmean(ceil)
    bar = means.get(bar_arm) if bar_arm else None
    return {"n": len(keys), "bar_arm": bar_arm, "bar": bar, "reach_arm": reach_arm,
            "reach": means.get(reach_arm) if reach_arm else None,
            "gain": fmean(d) if d else None, "d": d, "d_reps": d_reps, "stoch": stoch,
            "rss": rss, "rdf": rdf, "fill": fmean(fills) if fills else None,
            "ceiling": c, "room": (c - bar) / c if bar is not None and c > 0 else 0.0,
            "collapse": all(collapsed(r) for e in keys for rs in eps[e]["arms"].values()
                            for r in rs),
            "above": dict(above), "means": means}


def exp0050_stratum(world: str) -> tuple[tuple[int, str], bool]:
    """fan-<family>-<size>-f<fanout>-<regime>-<nn> -> ((size, choice), shocked)."""
    _, fam, size, f, regime, _ = world.split("-")
    return (int(size), "dense" if fam == "dense" else f"sparse_{f}"), regime == "shocked"


def spreads_exp0050(rows: list[dict[str, str]], table: list[dict]) -> dict:
    """sigma_ep / sigma_rep per size x choice from EXP-0050 (bar over the blind arms, reach =
    trained orderer, the arms both runs have); a stratum with no df takes its size's pool."""
    vals = values([r for r in rows if r["world"].startswith("fan-")])
    ep: dict = defaultdict(lambda: [0.0, 0])
    rep: dict = defaultdict(lambda: [0.0, 0])
    for w, eps in vals.items():
        s, shocked = exp0050_stratum(w)
        m = measure(eps, table)
        ep[s], rep[s]  # noqa: B018 (every stratum gets an entry)
        if shocked and len(m["d"]) >= 2:
            mu = fmean(m["d"])
            ep[s][0] += sum((x - mu) ** 2 for x in m["d"])
            ep[s][1] += len(m["d"]) - 1
        rep[s][0] += m["rss"]
        rep[s][1] += m["rdf"]
    out = {}
    for s in sorted(set(ep) | set(rep)):
        (es, ed), (rs, rd) = ep[s], rep[s]
        me, mr = ed < 1, rd < 1
        if me:
            es, ed = (sum(v[0] for k, v in ep.items() if k[0] == s[0]),
                      sum(v[1] for k, v in ep.items() if k[0] == s[0]))
        if mr:
            rs, rd = (sum(v[0] for k, v in rep.items() if k[0] == s[0]),
                      sum(v[1] for k, v in rep.items() if k[0] == s[0]))
        out[f"{s[0]}/{s[1]}"] = {"sigma_ep": math.sqrt(es / ed) if ed else None, "df_ep": ed,
                                 "sigma_rep": math.sqrt(rs / rd) if rd else None, "df_rep": rd,
                                 "merged_ep": me, "merged_rep": mr}
    return out


def classify(m: dict, info: dict, sp: dict | None) -> tuple[float, float, bool, str]:
    var = 0.0
    se = None
    if info["differing"]:
        if not sp or sp["sigma_ep"] is None:
            se = math.inf
        else:
            var += sp["sigma_ep"] ** 2
    for r in m["stoch"].values():
        if not sp or sp["sigma_rep"] is None:
            se = math.inf
        else:
            var += sp["sigma_rep"] ** 2 / r
    if se is None:
        se = math.sqrt(var) / math.sqrt(m["n"])
    floor = max(2.0 * se, 0.01 * (m["bar"] or 0.0))
    sign = bool(m["d_reps"]) and all(x > 0 for reps in m["d_reps"] for x in reps)
    if m["collapse"]:
        c = "collapse"
    elif (m["fill"] is not None and m["fill"] >= 0.95) or (
            m["bar"] is not None and m["bar"] >= 0.97 * m["ceiling"]):
        c = "saturated"
    elif m["gain"] is not None and m["gain"] > floor and sign:
        c = "hard"
    elif m["room"] >= 0.10:
        c = "open"
    else:
        c = "flat"
    return se, floor, sign, c


def holds(b: tuple, rec: dict) -> bool:
    axis, label, lo, hi = b
    if axis in ("choice", "shocks", "pattern"):
        return str(rec["levers"][axis]) == label
    v = rec["coordinates"][axis]
    return (lo is None or v > lo) and (hi is None or v <= hi)


def bins(size: int, recs: list[dict]) -> list[tuple]:
    out = [("choice", c, None, None) for c in CHOICES[size]]
    out += [("shocks", str(b), None, None) for b in (False, True)]
    out += [("pattern", p, None, None) for p in PATTERNS]
    for c in COORDS:
        v = sorted(r["coordinates"][c] for r in recs)
        cuts = [v[max(0, math.ceil(q * len(v)) - 1)] for q in (0.25, 0.5, 0.75)]
        edges = [None, *cuts, None]
        out += [(c, f"Q{i + 1}", edges[i], edges[i + 1]) for i in range(4)]
    return out


def select(cands: list[dict], k: int, bs: list[tuple]) -> list[tuple[dict, str]]:
    def key(c: dict) -> tuple:
        if c["cls"] == "hard":
            return (0, -((c["gain"] or 0.0) / c["bar"] if c["bar"] else math.inf), c["world"])
        return (1, -c["room"], c["world"])

    pool = sorted((c for c in cands if c["cls"] in ("hard", "open")), key=key)
    chosen: list[tuple[dict, str]] = []
    ids: set[str] = set()
    for b in bs:
        if len(chosen) >= k:
            break
        if any(holds(b, c["rec"]) for c, _ in chosen):
            continue
        for c in pool:
            if c["world"] not in ids and holds(b, c["rec"]):
                chosen.append((c, f"coverage:{b[0]}={b[1]}"))
                ids.add(c["world"])
                break
    need = min(int(k * 0.25), sum(c["cls"] == "open" for c in pool))
    for c in pool:
        if len(chosen) >= k:
            break
        if c["world"] in ids:
            continue
        if c["cls"] == "hard" and k - len(chosen) <= need - sum(
                x["cls"] == "open" for x, _ in chosen):
            continue
        chosen.append((c, "score"))
        ids.add(c["world"])
    return chosen


def split(sel: list[dict]) -> dict[str, str]:
    out, strata = {}, defaultdict(list)
    for w in sel:
        if w["size"] == 42 and w["pattern"] == "switch":
            out[w["world"]] = "held-out-region"
        else:
            strata[(w["size"], w["choice"], w["pattern"])].append(w["world"])
    for ids in strata.values():
        ids.sort(key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        n = len(ids)
        nh = math.floor(0.2 * n + 0.5)
        nv = math.floor(0.4 * n + 0.5) - nh
        for i, w in enumerate(ids):
            out[w] = "held-out" if i < nh else "validation" if i < nh + nv else "train"
    return out


def screen_rep_spread(vals: dict, infos: dict, recs: dict) -> dict:
    """The screen's own replicate spread (task 5.1 review): `random_whole` runs R = 2 on every
    world, so its within-episode SD, pooled per size x choice, measures sigma_rep on atlas
    worlds directly (absolute, and relative = deviation / episode mean)."""
    acc: dict = defaultdict(lambda: [0.0, 0.0, 0])
    for w, eps in vals.items():
        if w not in recs:
            continue
        s = f"{infos[w]['size']}/{infos[w]['choice']}"
        for ep in eps.values():
            x = u(ep["arms"].get(f"{EP}/random_whole", []))
            if len(x) >= 2:
                m = fmean(x)
                acc[s][0] += sum((v - m) ** 2 for v in x)
                acc[s][1] += sum(((v - m) / m) ** 2 for v in x) if m else 0.0
                acc[s][2] += len(x) - 1
    return {s: {"sigma_rep": math.sqrt(a / df), "rel_sigma_rep": math.sqrt(r / df), "df": df}
            for s, (a, r, df) in sorted(acc.items()) if df}


def sensitivity(mine: dict, infos: dict, recs: dict, sp: dict, own: dict) -> dict:
    """Hard counts per size with the screen's own sigma_rep in place of EXP-0050's (sigma_ep
    kept): absolute pooled, and relative x bar; at the screen's 2 x SE and confirm's 3 x SE."""
    out: dict = {}
    for mode in ("absolute", "relative"):
        for mult in (2.0, 3.0):
            cnt: Counter = Counter()
            for w, m in mine.items():
                if w not in recs:
                    continue
                key = f"{infos[w]['size']}/{infos[w]['choice']}"
                base = sp.get(key) or {}
                o = own.get(key)
                srep = (o["sigma_rep"] if mode == "absolute" else
                        o["rel_sigma_rep"] * (m["bar"] or 0.0)) if o else None
                var = (base.get("sigma_ep") or 0.0) ** 2 if infos[w]["differing"] else 0.0
                var += sum((srep or 0.0) ** 2 / r for r in m["stoch"].values())
                floor = max(mult * math.sqrt(var / m["n"]), 0.01 * (m["bar"] or 0.0))
                if m["cls"] == "hard" and m["gain"] > floor:
                    cnt[infos[w]["size"]] += 1
            out[f"{mode} x{mult:g}"] = dict(cnt)
    return out


def close(a: object, b: object) -> bool:
    fa, fb = num(a), num(b)
    if fa is None or fb is None:
        return fa is None and fb is None
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path)
    ap.add_argument("--exp0050", type=Path, required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="producer|orderer=PATH, sha256 vs spec pins")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    d = a.exp_dir
    problems: list[str] = []
    spec = json.loads((d / "spec.json").read_text())
    ep_path = d / "artifacts" / "episodes.csv"
    rows = read(ep_path if ep_path.exists() else d / "episodes.csv")
    table = arms()
    planned = [(t["s"], r) for t in table for r in range(t["R"])]

    # contract failures
    cf = read(d / "contract_failures.csv")
    if cf:
        problems.append(f"contract_failures.csv has {len(cf)} rows")

    # worlds: pinned records (+ anchors = EXP-0050 fan-* worlds)
    recs = {r["world_id"]: r for r in spec["pins"]["worlds"] if "world_id" in r}
    vals = values(rows)
    anchors = sorted(w for w in vals if w.startswith("fan-"))
    missing_w = sorted(set(recs) - set(vals))
    extra_w = sorted(set(vals) - set(recs) - set(anchors))
    if missing_w or extra_w:
        problems.append(f"worlds missing {missing_w[:5]} extra {extra_w[:5]}")
    if len(anchors) != 8:
        problems.append(f"{len(anchors)} anchors, expected 8")
    sizes = Counter(r["size"] for r in recs.values())
    if dict(sizes) != {12: 512, 42: 128}:
        problems.append(f"world counts per size {dict(sizes)}")

    # completeness and digests
    keys = Counter((r["world"], r["episode"], r["subject"], int(r["replicate"])) for r in rows)
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        problems.append(f"{len(dup)} duplicate rows, e.g. {dup[0]}")
    n_missing = 0
    digest_bad = 0
    for w, eps in vals.items():
        if w in recs and len(eps) != 1:
            problems.append(f"{w}: {len(eps)} episodes, expected 1")
        for e, ep in eps.items():
            for s, r in planned:
                if (w, str(e), s, r) not in keys:
                    n_missing += 1
            pairs = {(x["demand_digest"], x["shock_digest"]) for rs in ep["arms"].values()
                     for x in rs}
            digest_bad += len(pairs) != 1
            extra = set(ep["arms"]) - {t["s"] for t in table}
            if extra:
                problems.append(f"{w}/{e}: unplanned subjects {sorted(extra)}")
    if n_missing:
        problems.append(f"{n_missing} planned rows missing")
    if digest_bad:
        problems.append(f"{digest_bad} world-episodes with != 1 digest pair")
    n_rows_planned = len(vals) * len(planned)

    # B2
    b2_viol = [r for r in rows if num(r["bound_b2"]) is not None
               and float(r["units"]) > float(r["bound_b2"]) + TOL]
    if b2_viol:
        problems.append(f"{len(b2_viol)} B2 violations")

    # anchors vs EXP-0050
    ref_rows = read(a.exp0050)
    ref = {(r["world"], r["episode"], r["subject"], r["replicate"]): float(r["units"])
           for r in ref_rows}
    anc = [(r, ref.get((r["world"], r["episode"], r["subject"], r["replicate"])))
           for r in rows if r["world"] in anchors]
    anc_cmp = [(r, x) for r, x in anc if x is not None]
    anc_bad = [(r["world"], r["subject"], r["replicate"], r["units"], x) for r, x in anc_cmp
               if abs(float(r["units"]) - x) > 1e-9]
    if anc_bad or len(anc_cmp) == 0:
        problems.append(f"anchor mismatches {anc_bad[:5]} of {len(anc_cmp)}")
    anchors_csv = read(d / "anchors.csv") if (d / "anchors.csv").exists() else []
    if anchors_csv and (len(anchors_csv) != len(anc_cmp)
                        or any(x["match"].lower() != "true" for x in anchors_csv)):
        problems.append("anchors.csv disagrees (count or a non-match)")

    # spreads
    sp = spreads_exp0050(ref_rows, table)
    sp_spec = spec.get("result", {}).get("provisional_spreads") or {}
    for k, v in sp.items():
        sv = sp_spec.get(k)
        if sv is None or not all(close(v[f], sv[f]) if isinstance(v[f], float) or v[f] is None
                                 else v[f] == sv[f] for f in v):
            problems.append(f"spread {k}: probe {v} vs spec {sv}")

    # measures and classes
    wcsv = {r["world"]: r for r in read(d / "worlds.csv")}
    infos, mine = {}, {}
    for w in sorted(vals):
        if w in recs:
            lv = recs[w]["levers"]
            info = {"size": recs[w]["size"], "choice": lv["choice"], "pattern": lv["pattern"],
                    "shocks": bool(lv["shocks"]),
                    "differing": not (lv["pattern"] == "cyclic" and not lv["shocks"])}
        else:
            (size, choice), _ = exp0050_stratum(w)
            info = {"size": size, "choice": choice, "pattern": "anchor", "shocks": True,
                    "differing": True}
        infos[w] = info
        m = measure(vals[w], table)
        se, floor, sign, cls = classify(m, info, sp.get(f"{info['size']}/{info['choice']}"))
        m.update(se=se, floor=floor, sign=sign, cls=cls)
        mine[w] = m
        r = wcsv.get(w)
        if r is None:
            problems.append(f"{w} not in worlds.csv")
            continue
        for col, val in (("bar_arm", m["bar_arm"]), ("reach_arm", m["reach_arm"]),
                         ("class", cls), ("sign_consistent", str(sign).lower())):
            if (r[col].lower() if col == "sign_consistent" else r[col] or None) != val:
                problems.append(f"{w} {col}: csv {r[col]} probe {val}")
        for col, val in (("bar", m["bar"]), ("reach", m["reach"]), ("gain", m["gain"]),
                         ("bar_fill", m["fill"]), ("ceiling", m["ceiling"]),
                         ("room", m["room"]), ("se", se), ("noise_floor", floor),
                         ("arms_above_base", sum(m["above"].values()))):
            if not close(r[col], val):
                problems.append(f"{w} {col}: csv {r[col]} probe {val}")

    # selection and split
    sel_csv = read(d / "selection.csv")
    sel_rows = []
    for size in (12, 42):
        ws = [w for w in mine if w in recs and infos[w]["size"] == size]
        if not ws:
            continue
        cands = [{"world": w, "cls": mine[w]["cls"], "gain": mine[w]["gain"],
                  "bar": mine[w]["bar"], "room": mine[w]["room"], "rec": recs[w]} for w in ws]
        for rank, (c, how) in enumerate(select(cands, K[size], bins(size, [recs[w]
                                                                          for w in ws]))):
            sel_rows.append({"size": size, "rank": rank, "world": c["world"], "class": c["cls"],
                             "pass": how, **{k: infos[c["world"]][k]
                                             for k in ("choice", "pattern")}})
    sp_of = split([{"world": s["world"], "size": s["size"], "choice": s["choice"],
                    "pattern": s["pattern"]} for s in sel_rows])
    for s in sel_rows:
        s["split"] = sp_of[s["world"]]
    got = [(int(s["size"]), int(s["rank"]), s["world"], s["class"], s["pass"], s["split"])
           for s in sel_csv]
    want = [(s["size"], s["rank"], s["world"], s["class"], s["pass"], s["split"])
            for s in sel_rows]
    if got != want:
        diff = [(g, w) for g, w in zip(got, want) if g != w][:3]
        problems.append(f"selection.csv differs ({len(got)} vs {len(want)} rows): {diff}")
    for w, r in wcsv.items():
        if (r.get("split") or None) != sp_of.get(w):
            problems.append(f"{w} split in worlds.csv {r.get('split')} vs {sp_of.get(w)}")
            break

    # checkpoints
    pins = spec["pins"].get("checkpoints", {})
    for c in a.ckpt:
        name, path = c.split("=", 1)
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        pin = pins.get(name)
        print(f"ckpt {name}: {h[:12]} pin {str(pin)[:12]} {'OK' if h == pin else 'MISMATCH'}")
        if h != pin:
            problems.append(f"ckpt {name} sha mismatch")

    # summary
    print(f"rows {len(rows)} planned {n_rows_planned}; worlds {len(recs)} + anchors "
          f"{len(anchors)}; contract failures {len(cf)}; B2 violations {len(b2_viol)}; "
          f"anchor rows compared {len(anc_cmp)} mismatches {len(anc_bad)}")
    space = [w for w in mine if w in recs]
    summary: dict = {"classes": {}, "regions": {}, "above_base": {}, "anchors": {},
                     "selection": {}, "coords": {}}
    for size in (12, 42):
        ws = [w for w in space if infos[w]["size"] == size]
        if not ws:
            continue
        summary["classes"][size] = dict(Counter(mine[w]["cls"] for w in ws))
        print(f"\n== size {size}: {len(ws)} worlds, classes {summary['classes'][size]}")
        for axis, levels in (("choice", CHOICES[size]), ("pattern", PATTERNS),
                             ("shocks", [False, True])):
            for lvl in levels:
                g = [w for w in ws if infos[w][axis] == lvl]
                cnt = Counter(mine[w]["cls"] for w in g)
                fill = [mine[w]["fill"] for w in g if mine[w]["fill"] is not None]
                gains = [mine[w]["gain"] / mine[w]["bar"] for w in g if mine[w]["bar"]]
                summary["regions"][f"{size}/{axis}={lvl}"] = {"n": len(g), **cnt}
                print(f"  {axis}={lvl}: n={len(g)} {dict(cnt)} median bar fill "
                      f"{median(fill):.3f} median gain/bar {median(gains):+.3f}"
                      if g else f"  {axis}={lvl}: n=0")
        recs_s = [recs[w] for w in ws]
        for c in COORDS:
            v = sorted(r["coordinates"][c] for r in recs_s)
            cuts = [v[max(0, math.ceil(q * len(v)) - 1)] for q in (0.25, 0.5, 0.75)]
            edges = [None, *cuts, None]
            line = []
            for i in range(4):
                g = [w for w in ws if holds((c, "", edges[i], edges[i + 1]), recs[w])]
                cnt = Counter(mine[w]["cls"] for w in g)
                line.append(f"Q{i + 1}(n={len(g)} sat {cnt['saturated']} hard {cnt['hard']} "
                            f"open {cnt['open']} flat {cnt['flat']} col {cnt['collapse']})")
                summary["regions"][f"{size}/{c}=Q{i + 1}"] = {"n": len(g), "lo": edges[i],
                                                             "hi": edges[i + 1], **cnt}
            summary["coords"][f"{size}/{c}"] = {"min": v[0], "cuts": cuts, "max": v[-1]}
            print(f"  {c} [{v[0]:.3g} | {cuts[0]:.3g} {cuts[1]:.3g} {cuts[2]:.3g} | "
                  f"{v[-1]:.3g}]: " + " ".join(line))
        above = Counter()
        for w in ws:
            for s, n in mine[w]["above"].items():
                above[s] += n
        summary["above_base"][size] = dict(above)
        print(f"  arms above B_base (world-episodes): {dict(above)}")
        print(f"  reach argmax among hard: "
              f"{dict(Counter(mine[w]['reach_arm'] for w in ws if mine[w]['cls'] == 'hard'))}")
        print(f"  bar argmax: {dict(Counter(mine[w]['bar_arm'] for w in ws))}")
        sel = [s for s in sel_rows if s["size"] == size]
        summary["selection"][size] = {"n": len(sel), "classes": dict(Counter(s["class"]
                                                                            for s in sel)),
                                      "split": dict(Counter(s["split"] for s in sel)),
                                      "coverage_pass": sum(s["pass"] != "score" for s in sel)}
        print(f"  selection {summary['selection'][size]}")
    print("\n== anchors (screen class; bar arm / reach arm units)")
    for w in anchors:
        m = mine[w]
        summary["anchors"][w] = {"cls": m["cls"], "bar": m["bar"], "reach": m["reach"],
                                 "fill": m["fill"], "room": m["room"]}
        print(f"  {w}: {m['cls']} bar {m['bar']:.0f} ({m['bar_arm']}) reach {m['reach']:.0f} "
              f"({m['reach_arm']}) fill {m['fill']:.3f} room {m['room']:.3f}")
    own = screen_rep_spread(vals, infos, recs)
    print("\n== screen's own sigma_rep (random_whole R = 2) vs EXP-0050's provisional")
    for k, o in own.items():
        print(f"  {k}: {o['sigma_rep']:.1f} (relative {o['rel_sigma_rep']:.3f}, df {o['df']}) "
              f"vs {sp.get(k, {}).get('sigma_rep') or float('nan'):.1f}")
    summary["screen_sigma_rep"] = own
    summary["hard_with_screen_sigma_rep"] = sensitivity(mine, infos, recs, sp, own)
    print(f"  hard (of the screen's hard) with the screen's own sigma_rep: "
          f"{summary['hard_with_screen_sigma_rep']}")
    print(f"  hard worlds with a stochastic argmax arm: "
          f"{dict(Counter(infos[w]['size'] for w in space if mine[w]['cls'] == 'hard' and mine[w]['stoch']))}")
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print("\nPROBLEMS:" if problems else "\nALL CHECKS PASS")
    for p in problems[:40]:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
