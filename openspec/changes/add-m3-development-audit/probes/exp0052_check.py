"""Task 5.2 verification and summary of an EXP-0052 (atlas confirm) dir, independent of
`tokenomics.eval.atlas`'s analysis code (stdlib only, plus `world_space.episode_seeds` for the
seed pin; the rules are re-implemented from design D12 + task 4.7, the arm table from the
roster v2 file). Adapted from exp0051_check.py (shared helpers imported from it).

Checks: contract_failures.csv header-only; every selected world (EXP-0051 selection.csv) present
with every planned (episode, arm, replicate) row once, one demand + shock digest pair per
world-episode; confirm episode seeds = episode_seeds(wid, n, "confirm") (n = 2, 1 for cyclic
without shocks) and disjoint from the screen's; 0 B2 / B_lp violations and exactly one LP row per
world-episode; the fixed argmax arms equal the screen's selection; the pooled spreads (confirm's
own, per size x choice, merged within size below 30 df), every worlds.csv measure, floor (3 x SE)
and class, deciding_check.csv and atlas_v1.json (content hash, classes, split) recomputed from
episodes.csv; the split equals EXP-0051's; checkpoint sha256 pins; both run commits.

Task 5.3 (`--reclassified DIR`, the derived dir of `atlas --mode reclassify`): recompute the
revised classification independently (spreads = normal-consistent median absolute within-group
residual, same strata and 30-df merge; dead-arm rule: a fixed bar or reach arm that dies in any
episode or replicate makes the world collapse) and compare every class, spread and floor with
DIR/worlds.csv, `class_v1` with the recorded worlds.csv, atlas_v2.json (content hash, classes,
split, reclassified_from hashes); print the counts under both rules, the flips with their cause,
each hard world's own episode spread against sigma_ep, and PoC-A's sample (design D5) under both
rules.

usage (from /srv/research/tokenomics):
  python3 openspec/changes/add-m3-development-audit/probes/exp0052_check.py \
      experiments/EXP-0052-headroom-atlas-confirm \
      --screen experiments/EXP-0051-headroom-atlas-screen \
      [--ckpt producer=PATH --ckpt orderer=PATH] [--json OUT] \
      [--reclassified experiments/EXP-0052-headroom-atlas-confirm/reclassified]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp0051_check import EP, ROSTER, close, first_max, num, read  # noqa: E402

TOL = 0.01
MIN_DF = 30
SCREEN_COMMIT = "92f317725ca4fe02bd703b950008ee2a3cb64335"
CONFIRM_COMMIT = "9d442202d2184837d3f4c5408d194a6102b37042"
DENSE = "dense"
ME = f"{EP}/makeable_split"
MWW = f"{EP}/makeable_weighted_whole"
TO = f"{EP}/trained_orderer"


def arms() -> list[dict]:
    """D12 confirm arms in table order: roster v2 (random R=2), round_robin, random_whole R=2
    (blind); makeable_split, makeable_weighted_whole R=2, trained_orderer (reach); portable
    (reported, never an argmax)."""
    roster = json.loads(ROSTER.read_text())["members"]
    out = [{"s": f"{m['producing']}/{m['ordering']}", "side": "blind", "rnd": m["random"],
            "R": 2 if m["random"] else 1} for m in roster]
    out += [{"s": f"{EP}/round_robin", "side": "blind", "rnd": False, "R": 1},
            {"s": f"{EP}/random_whole", "side": "blind", "rnd": True, "R": 2},
            {"s": ME, "side": "reach", "rnd": False, "R": 1},
            {"s": MWW, "side": "reach", "rnd": True, "R": 2},
            {"s": TO, "side": "reach", "rnd": False, "R": 1},
            {"s": "portable", "side": "reported", "rnd": False, "R": 1}]
    return out


def values(rows: list[dict[str, str]]) -> dict:
    """world -> episode -> {"b2", "base", "lp", "lp_base", "n_lp", "arms": subject -> rows}."""
    out: dict = defaultdict(dict)
    for r in sorted(rows, key=lambda r: (r["world"], int(r["episode"]), r["subject"],
                                         int(r["replicate"]))):
        e = out[r["world"]].setdefault(int(r["episode"]), {
            "b2": None, "base": None, "lp": None, "lp_base": None, "n_lp": 0,
            "arms": defaultdict(list)})
        for k, col in (("b2", "bound_b2"), ("base", "bound_base"), ("lp", "bound_lp"),
                       ("lp_base", "bound_lp_base")):
            if e[k] is None:
                e[k] = num(r.get(col))
        e["n_lp"] += num(r.get("bound_lp")) is not None
        e["arms"][r["subject"]].append(r)
    return out


def u(rs: list[dict]) -> list[float]:
    return [float(r["units"]) for r in rs]


def collapsed(r: dict) -> bool:
    tr = r.get("trace_capacity_ratio_trace")
    return r["dead"].lower() == "true" and bool(tr) and len(json.loads(tr)) < 50


def measure(eps: dict, table: list[dict], fixed: tuple[str, str]) -> dict:
    keys = sorted(eps)
    present = [a for a in table if all(a["s"] in eps[e]["arms"] for e in keys)]
    means = {a["s"]: fmean(fmean(u(eps[e]["arms"][a["s"]])) for e in keys) for a in present}
    rnd = {a["s"]: a["rnd"] for a in present}
    bar_arm = fixed[0] if fixed[0] in means else None
    reach_arm = fixed[1] if fixed[1] in means else None
    d, d_reps, stoch, rss, rdf = [], [], {}, 0.0, 0
    if bar_arm and reach_arm:
        for e in keys:
            ru, bu = u(eps[e]["arms"][reach_arm]), u(eps[e]["arms"][bar_arm])
            d.append(fmean(ru) - fmean(bu))
            d_reps.append([ru[i % len(ru)] - bu[i % len(bu)]
                           for i in range(max(len(ru), len(bu)))])
        for s in (bar_arm, reach_arm):
            if rnd[s]:
                stoch[s] = min(len(eps[e]["arms"][s]) for e in keys)
                for e in keys:
                    x = u(eps[e]["arms"][s])
                    if len(x) >= 2:
                        m = fmean(x)
                        rss += sum((v - m) ** 2 for v in x)
                        rdf += len(x) - 1
    ceil, fills, above, lp_room = [], [], Counter(), []
    for e in keys:
        ep = eps[e]
        best = max(v for rs in ep["arms"].values() for v in u(rs))
        cap = ep["lp_base"] if ep["lp_base"] is not None else ep["base"]
        bs = [b for b in (ep["b2"], ep["lp"], cap) if b is not None]
        ceil.append(max(min(bs), best) if bs else best)
        if cap is not None:
            for s, rs in ep["arms"].items():
                if max(u(rs)) > cap + TOL:
                    above[s] += 1
        if bar_arm:
            for r in ep["arms"][bar_arm]:
                o, f = num(r.get("ordered_units")), num(r.get("fulfilled_units"))
                if o and f is not None:
                    fills.append(f / o)
    c = fmean(ceil)
    bar = means.get(bar_arm) if bar_arm else None
    mean_of = lambda k: (fmean(x) if (x := [eps[e][k] for e in keys  # noqa: E731
                                            if eps[e][k] is not None]) else None)
    return {"n": len(keys), "bar_arm": bar_arm, "bar": bar, "reach_arm": reach_arm,
            "reach": means.get(reach_arm) if reach_arm else None,
            "gain": fmean(d) if d else None, "d": d, "d_reps": d_reps, "stoch": stoch,
            "rss": rss, "rdf": rdf, "fill": fmean(fills) if fills else None,
            "ceiling": c, "room": (c - bar) / c if bar is not None and c > 0 else 0.0,
            "b2": mean_of("b2"), "base": mean_of("base"), "lp": mean_of("lp"),
            "lp_base": mean_of("lp_base"),
            "collapse": all(collapsed(r) for e in keys for rs in eps[e]["arms"].values()
                            for r in rs),
            "above": dict(above), "means": means}


def pooled(mine: dict, infos: dict) -> dict:
    ep: dict = defaultdict(lambda: [0.0, 0])
    rep: dict = defaultdict(lambda: [0.0, 0])
    for w, m in mine.items():
        s = (infos[w]["size"], infos[w]["choice"])
        ep[s], rep[s]  # noqa: B018
        if infos[w]["differing"] and len(m["d"]) >= 2:
            mu = fmean(m["d"])
            ep[s][0] += sum((x - mu) ** 2 for x in m["d"])
            ep[s][1] += len(m["d"]) - 1
        rep[s][0] += m["rss"]
        rep[s][1] += m["rdf"]
    out = {}
    for s in sorted(set(ep) | set(rep)):
        (es, ed), (rs, rd) = ep[s], rep[s]
        me, mr = ed < MIN_DF, rd < MIN_DF
        if me:
            es = sum(v[0] for k, v in ep.items() if k[0] == s[0])
            ed = sum(v[1] for k, v in ep.items() if k[0] == s[0])
        if mr:
            rs = sum(v[0] for k, v in rep.items() if k[0] == s[0])
            rd = sum(v[1] for k, v in rep.items() if k[0] == s[0])
        out[f"{s[0]}/{s[1]}"] = {"sigma_ep": math.sqrt(es / ed) if ed else None, "df_ep": ed,
                                 "sigma_rep": math.sqrt(rs / rd) if rd else None, "df_rep": rd,
                                 "merged_ep": me, "merged_rep": mr}
    return out


def se_of(n: int, differing: bool, stoch: dict, sp: dict | None) -> float:
    var = 0.0
    if differing:
        if not sp or sp["sigma_ep"] is None:
            return math.inf
        var += sp["sigma_ep"] ** 2
    for r in stoch.values():
        if not sp or sp["sigma_rep"] is None:
            return math.inf
        var += sp["sigma_rep"] ** 2 / r
    return math.sqrt(var) / math.sqrt(n) if n else math.inf


def classify(m: dict, floor: float) -> tuple[bool, str]:
    sign = bool(m["d_reps"]) and all(x > 0 for reps in m["d_reps"] for x in reps)
    if m["collapse"]:
        return sign, "collapse"
    if (m["fill"] is not None and m["fill"] >= 0.95) or (
            m["bar"] is not None and m["bar"] >= 0.97 * m["ceiling"]):
        return sign, "saturated"
    if m["gain"] is not None and m["gain"] > floor and sign:
        return sign, "hard"
    if m["room"] >= 0.10:
        return sign, "open"
    return sign, "flat"


def eligible(size: int, choice: str) -> int:
    return size // 3 if choice == DENSE else int(choice.removeprefix("sparse_f"))


def deciding(vals: dict, infos: dict, mine: dict, sp: dict, table: list[dict]) -> list[dict]:
    rnd = {a["s"]: a["rnd"] for a in table}
    rows, flags = [], defaultdict(list)
    for w in sorted(vals):
        i, eps = infos[w], vals[w]
        if mine[w]["cls"] != "hard" or eligible(i["size"], i["choice"]) < 4:
            continue
        if not all(s in eps[e]["arms"] for e in eps for s in (ME, MWW, TO)):
            continue
        row = {"kind": "world", "world": w}
        for name, (a, b) in (("split_vs_trained", (ME, TO)), ("whole_vs_split", (MWW, ME))):
            dd = fmean(fmean(u(eps[e]["arms"][a])) - fmean(u(eps[e]["arms"][b])) for e in eps)
            reps = {s: min(len(eps[e]["arms"][s]) for e in eps) for s in (a, b) if rnd[s]}
            se = se_of(len(eps), i["differing"], reps, sp.get(f"{i['size']}/{i['choice']}"))
            row[f"{name}_gain"] = dd
            row[f"{name}_floor"] = max(3.0 * se, 0.01 * (mine[w]["bar"] or 0.0))
        within = abs(row["split_vs_trained_gain"]) <= row["split_vs_trained_floor"]
        whole = row["whole_vs_split_gain"] > row["whole_vs_split_floor"]
        below = row["split_vs_trained_gain"] < -row["split_vs_trained_floor"]
        row.update(split_within_trained=within, whole_above_split=whole,
                   split_below_trained=below)
        for k, v in (("input_aware_weighting", within), ("splitting_costs", whole),
                     ("scorer_beyond_target", below)):
            flags[k].append(v)
        rows.append(row)
    for k in ("input_aware_weighting", "splitting_costs", "scorer_beyond_target"):
        n, hits = len(flags[k]), sum(flags[k])
        rows.append({"kind": "branch", "branch": k, "worlds": n, "count": hits,
                     "share": hits / n if n else None, "holds": bool(n) and hits / n >= 2 / 3})
    for scope, ws in (("all", sorted(vals)),
                      ("hard_e4", [r["world"] for r in rows if r["kind"] == "world"])):
        for a in table:
            sh = ok = 0.0
            for w in ws:
                for ep in vals[w].values():
                    for r in ep["arms"].get(a["s"], []):
                        for k, v in r.items():
                            if k.startswith("trace_ttl_lost_units_") and v not in ("", None):
                                if k.endswith("_inputs_short"):
                                    sh += float(v)
                                elif k.endswith("_inputs_ok"):
                                    ok += float(v)
            rows.append({"kind": "ttl", "scope": scope, "arm": a["s"],
                         "ttl_lost_inputs_short": sh, "ttl_lost_inputs_ok": ok,
                         "share_inputs_short": sh / (sh + ok) if sh + ok else None})
    return rows


def died(eps: dict, arms_: tuple) -> bool:
    return any(r["dead"].lower() == "true" for e in eps.values() for s in arms_ if s
               for r in e["arms"].get(s, []))


def sensitivity(vals: dict, mine: dict, infos: dict, table: list[dict]) -> dict:
    """Not pre-registered (task 5.2 review aid): what drives the confirm floor, and the hard
    count per size under other spread estimates, at 3 x SE and 2 x SE.
    - share of the pooled sigma_ep sum of squares from the 5 largest worlds per size;
    - "no-death pool": sigma_ep / sigma_rep pooled per size over worlds whose fixed bar and reach
      arms never died (deaths make the paired gain swing by whole episodes);
    - "relative": sigma_ep of d / bar and sigma_rep of replicate / episode mean, pooled per size,
      times the world's bar (spreads scale with the world's units: bar 500-12,000);
    - "median-based": sigma from the median absolute pair difference (episode pairs, replicate
      pairs), robust to the few worlds that dominate the sums of squares;
    - "free bar": the bar re-argmaxed on the confirm's own blind means (the fixed screen arm can
      die on fresh seeds while another blind arm survives)."""
    out: dict = {}
    print("\n== sensitivity (not pre-registered)")
    for size in (12, 42):
        ws = [w for w in mine if infos[w]["size"] == size]
        if not ws:
            continue
        ss = sorted(((sum((x - fmean(mine[w]["d"])) ** 2 for x in mine[w]["d"]), w)
                     for w in ws if infos[w]["differing"] and len(mine[w]["d"]) >= 2),
                    reverse=True)
        tot = sum(s for s, _ in ss)
        top5 = sum(s for s, _ in ss[:5]) / tot if tot else 0.0
        clean = [w for w in ws if not died(vals[w], (mine[w]["bar_arm"], mine[w]["reach_arm"]))]
        pools: dict = {}
        for name, use in (("all", ws), ("no-death", clean)):
            e = [(sum((x - fmean(mine[w]["d"])) ** 2 for x in mine[w]["d"]), len(mine[w]["d"]) - 1)
                 for w in use if infos[w]["differing"] and len(mine[w]["d"]) >= 2]
            r = [(mine[w]["rss"], mine[w]["rdf"]) for w in use]
            sde = math.sqrt(sum(a for a, _ in e) / sum(b for _, b in e)) if e else None
            sdr = (math.sqrt(sum(a for a, _ in r) / sum(b for _, b in r))
                   if sum(b for _, b in r) else None)
            pools[name] = {"sigma_ep": sde, "sigma_rep": sdr,
                           "df_ep": sum(b for _, b in e), "df_rep": sum(b for _, b in r)}
        rel_e, rel_r = [0.0, 0], [0.0, 0]
        for w in ws:
            m = mine[w]
            if infos[w]["differing"] and len(m["d"]) >= 2 and m["bar"]:
                x = [v / m["bar"] for v in m["d"]]
                rel_e[0] += sum((v - fmean(x)) ** 2 for v in x)
                rel_e[1] += len(x) - 1
            for s in m["stoch"]:
                for e in vals[w].values():
                    xs = u(e["arms"][s])
                    if len(xs) >= 2 and fmean(xs):
                        mu = fmean(xs)
                        rel_r[0] += sum(((v - mu) / mu) ** 2 for v in xs)
                        rel_r[1] += len(xs) - 1
        rel = {"sigma_ep": math.sqrt(rel_e[0] / rel_e[1]) if rel_e[1] else None,
               "sigma_rep": math.sqrt(rel_r[0] / rel_r[1]) if rel_r[1] else None}
        # median-based: sigma = median |pair difference| / (sqrt 2 x 0.6745) (normal-consistent)
        k = math.sqrt(2) * 0.6745
        pe = [abs(mine[w]["d"][0] - mine[w]["d"][1]) for w in ws
              if infos[w]["differing"] and len(mine[w]["d"]) == 2]
        pr = [abs(x[0] - x[1]) for w in ws for s in mine[w]["stoch"]
              for e in vals[w].values() if len(x := u(e["arms"][s])) == 2]
        rob = {"sigma_ep": median(pe) / k if pe else None,
               "sigma_rep": median(pr) / k if pr else None}
        # outlier exclusion: the 5 largest sigma_ep / sigma_rep contributors per size leave the
        # pools (pre-registered pooling otherwise: per size x choice, merged below 30 df); every
        # world is still classified
        top_ep = [w for _, w in ss[:5]]
        rs_ = sorted(((mine[w]["rss"], w) for w in ws if mine[w]["rdf"]), reverse=True)
        rtot = sum(x for x, _ in rs_)
        top5_rep = sum(x for x, _ in rs_[:5]) / rtot if rtot else 0.0
        top_rep = [w for _, w in rs_[:5]]
        excl = {}
        for name, drop in (("minus top-5 sigma_ep", set(top_ep)),
                           ("minus top-5 sigma_rep", set(top_rep)),
                           ("minus both", set(top_ep) | set(top_rep))):
            excl[name] = pooled({w: mine[w] for w in ws if w not in drop}, infos)
        res = {"top5_share_ss_ep": top5, "top5_share_ss_rep": top5_rep, "top_ep": top_ep,
               "top_rep": top_rep, "excluded_pools": excl,
               "pools": pools, "relative": rel, "median_based": rob,
               "n_clean": len(clean), "hard": {}}
        blind = [t["s"] for t in table if t["side"] == "blind"]
        for mult in (3.0, 2.0):
            for name in ("pre-registered", "no-death pool", "relative", "median-based",
                         "free bar", "minus top-5 sigma_ep", "minus top-5 sigma_rep",
                         "minus both"):
                n = 0
                for w in ws:
                    m, i = mine[w], infos[w]
                    if name == "free bar":
                        fb = first_max({s: v for s, v in m["means"].items() if s in blind}, blind)
                        m = measure(vals[w], table, (fb, m["reach_arm"]))
                        sp_ = {"sigma_ep": None, "sigma_rep": None}
                        key = f"{i['size']}/{i['choice']}"
                        sp_ = SP_CACHE.get(key, sp_)
                        se = se_of(m["n"], i["differing"], m["stoch"], sp_)
                    elif name == "pre-registered":
                        se = m["se"]
                    elif name.startswith("minus"):
                        se = se_of(m["n"], i["differing"], m["stoch"],
                                   excl[name].get(f"{i['size']}/{i['choice']}"))
                    elif name == "median-based":
                        se = se_of(m["n"], i["differing"], m["stoch"], rob)
                    elif name == "no-death pool":
                        se = se_of(m["n"], i["differing"], m["stoch"], pools["no-death"])
                    else:
                        sp_ = {k: (v * (m["bar"] or 0.0) if v is not None else None)
                               for k, v in rel.items()}
                        se = se_of(m["n"], i["differing"], m["stoch"], sp_)
                    floor = max(mult * se, 0.01 * (m["bar"] or 0.0))
                    n += classify(m, floor)[1] == "hard"
                res["hard"][f"{name} x{mult:g}"] = n
        sc = sum(1 for w in ws if mine[w]["sign"] and mine[w]["cls"] not in ("saturated",
                                                                           "collapse"))
        res["sign_consistent_positive"] = sc
        res["fragile"] = [w for w in ws if died(vals[w], (mine[w]["bar_arm"],
                                                           mine[w]["reach_arm"]))]
        out[size] = res
        f = lambda x: "None" if x is None else f"{x:.1f}"  # noqa: E731
        print(f"  {size}: top-5 worlds give {top5:.0%} of the sigma_ep sum of squares; pools "
              f"all ep {f(pools['all']['sigma_ep'])} rep {f(pools['all']['sigma_rep'])} | no-death "
              f"({len(clean)} worlds) ep {f(pools['no-death']['sigma_ep'])} rep "
              f"{f(pools['no-death']['sigma_rep'])} | relative ep {rel['sigma_ep']:.3f} rep "
              f"{rel['sigma_rep']:.3f} | median-based ep {f(rob['sigma_ep'])} rep "
              f"{f(rob['sigma_rep'])}")
        print(f"      top-5 sigma_rep share {top5_rep:.0%} ({top_rep}); top-5 sigma_ep {top_ep}")
        for name, e in excl.items():
            print(f"      {name}: " + "; ".join(f"{k} ep {f(v['sigma_ep'])} rep {f(v['sigma_rep'])}"
                                              for k, v in e.items()))
        print(f"      hard counts {res['hard']}; gain positive in every episode+replicate: {sc}; "
              f"worlds where the fixed bar or reach arm died: {len(res['fragile'])}")
    return out


SP_CACHE: dict = {}
MAD_K = 0.6744897501960817  # Phi^-1(0.75)
POC_PATTERNS = {"cyclic", "iid", "drift", "burst", "seasonal"}  # design D5


def resid(xs: list[float]) -> list[float]:
    n = len(xs)
    if n < 2:
        return []
    mu = fmean(xs)
    return [(x - mu) * math.sqrt(n / (n - 1)) for x in xs]


def pooled_mad(vals: dict, mine: dict, infos: dict) -> dict:
    """Task 5.3 spreads: per size x choice (merged within size below 30 df), sigma = median
    |residual| / Phi^-1(0.75) over the episode gains of differing worlds (sigma_ep) and the
    replicates of the stochastic fixed arms per episode (sigma_rep)."""
    ep: dict = defaultdict(lambda: [[], 0])
    rp: dict = defaultdict(lambda: [[], 0])
    for w, m in mine.items():
        s = (infos[w]["size"], infos[w]["choice"])
        ep[s], rp[s]  # noqa: B018
        if infos[w]["differing"] and len(m["d"]) >= 2:
            ep[s][0] += resid(m["d"])
            ep[s][1] += len(m["d"]) - 1
        for arm in m["stoch"]:
            for e in vals[w].values():
                x = u(e["arms"][arm])
                if len(x) >= 2:
                    rp[s][0] += resid(x)
                    rp[s][1] += len(x) - 1
    out = {}
    for s in sorted(set(ep) | set(rp)):
        cells = {}
        for name, pool in (("ep", ep), ("rep", rp)):
            r, df = pool[s]
            merged = df < MIN_DF
            if merged:
                r = [x for k, v in pool.items() if k[0] == s[0] for x in v[0]]
                df = sum(v[1] for k, v in pool.items() if k[0] == s[0])
            cells[f"sigma_{name}"] = median(abs(x) for x in r) / MAD_K if df else None
            cells[f"df_{name}"], cells[f"merged_{name}"] = df, merged
        out[f"{s[0]}/{s[1]}"] = cells
    return out


def poc_a_sample(worlds: dict, cls_key: str) -> list[tuple[str, str]]:
    """Design D5: train split, 12 producers, patterns cyclic/iid/drift/burst/seasonal; 6 hard by
    gain and 6 open by room, the other class filling a shortfall (12 in all); per class the best
    n // 2 shocked and n - n // 2 unshocked, a shortfall on either side filled by the class's
    best remaining worlds."""
    pool = {w: x for w, x in worlds.items() if x["split"] == "train" and x["size"] == 12
            and x["pattern"] in POC_PATTERNS}
    ranked = {c: sorted((w for w, x in pool.items() if x[cls_key] == c),
                        key=lambda w: -(pool[w]["gain"] if c == "hard" else pool[w]["room"]))
              for c in ("hard", "open")}
    n_hard = min(6, len(ranked["hard"]))
    n = {"hard": n_hard, "open": min(12 - n_hard, len(ranked["open"]))}
    n["hard"] = min(12 - n["open"], len(ranked["hard"]))
    out = []
    for c in ("hard", "open"):
        want = {True: n[c] // 2, False: n[c] - n[c] // 2}
        pick = []
        for shocked in (True, False):
            pick += [w for w in ranked[c] if pool[w]["shocks"] == shocked][:want[shocked]]
        pick += [w for w in ranked[c] if w not in pick][:n[c] - len(pick)]
        out += [(c, w) for w in sorted(pick, key=ranked[c].index)]
    return out


def reclassified_check(rdir: Path, d: Path, vals: dict, mine: dict, infos: dict,
                       wcsv: dict) -> tuple[list[str], dict]:
    problems: list[str] = []
    sp = pooled_mad(vals, mine, infos)
    got = {r["world"]: r for r in read(rdir / "worlds.csv")}
    if set(got) != set(mine):
        problems.append("reclassified worlds != confirm worlds")
    new: dict = {}
    for w, m in mine.items():
        i = infos[w]
        key = f"{i['size']}/{i['choice']}"
        se = se_of(m["n"], i["differing"], m["stoch"], sp.get(key))
        floor = max(3.0 * se, 0.01 * (m["bar"] or 0.0))
        dead = died(vals[w], (m["bar_arm"], m["reach_arm"]))
        sign, cls = classify(m, floor)
        if dead:
            cls = "collapse"
        new[w] = {"cls": cls, "floor": floor, "se": se, "dead": dead, "v1": m["cls"],
                  "gain": m["gain"], "room": m["room"], "split": m["split"],
                  "size": i["size"], "pattern": i["pattern"], "shocks": i["shocks"],
                  "choice": i["choice"], "bar": m["bar"], "d": m["d"],
                  "sigma_ep": sp[key]["sigma_ep"]}
        r = got.get(w)
        if r is None:
            continue
        for col, val in (("class", cls), ("class_v1", m["cls"]),
                         ("class_v1", wcsv[w]["class"]), ("split", m["split"])):
            if r[col] != val:
                problems.append(f"reclassified {w} {col}: csv {r[col]} probe {val}")
        if r["argmax_died"].lower() != str(dead).lower():
            problems.append(f"reclassified {w} argmax_died: csv {r['argmax_died']} probe {dead}")
        for col, val in (("se", se), ("noise_floor", floor), ("gain", m["gain"]),
                         ("room", m["room"]), ("sigma_ep", sp[key]["sigma_ep"]),
                         ("sigma_rep", sp[key]["sigma_rep"])):
            if not close(r[col], val):
                problems.append(f"reclassified {w} {col}: csv {r[col]} probe {val}")
    atlas = json.loads((rdir / "atlas_v2.json").read_text())
    if canonical_hash(atlas) != atlas.get("content_hash"):
        problems.append("atlas_v2.json content hash does not match its body")
    if atlas.get("format") != "atlas-v2":
        problems.append(f"atlas_v2.json format {atlas.get('format')}")
    src = atlas.get("reclassified_from", {})
    ep_path = d / "artifacts" / "episodes.csv"
    for k, path in (("episodes_sha256", ep_path if ep_path.exists() else d / "episodes.csv"),
                    ("worlds_sha256", d / "worlds.csv")):
        if src.get(k) != hashlib.sha256(path.read_bytes()).hexdigest():
            problems.append(f"atlas_v2.json reclassified_from.{k} differs")
    v1 = json.loads((d / "atlas_v1.json").read_text())
    if src.get("atlas_v1_content_hash") != v1.get("content_hash"):
        problems.append("atlas_v2.json reclassified_from.atlas_v1_content_hash differs")
    for x in atlas["worlds"]:
        w = x["record"]["world_id"]
        if w in new and (x["confirm_class"] != new[w]["cls"] or x["class_v1"] != new[w]["v1"]
                         or x["split"] != new[w]["split"]):
            problems.append(f"atlas_v2.json {w} differs")
    # summary
    print("\n== task 5.3 reclassification (robust spread + dead-arm rule)")
    f = lambda x: "None" if x is None else f"{x:.1f}"  # noqa: E731
    for k, v in sp.items():
        print(f"  {k}: sigma_ep {f(v['sigma_ep'])} (df {v['df_ep']}"
              f"{' merged' if v['merged_ep'] else ''}) sigma_rep {f(v['sigma_rep'])} "
              f"(df {v['df_rep']}{' merged' if v['merged_rep'] else ''})")
    summary: dict = {"spreads": sp, "counts": {}, "flips": [], "own_spread": [], "poc_a": {}}
    for size in (12, 42):
        ws = [w for w in new if new[w]["size"] == size]
        c_old, c_new = Counter(new[w]["v1"] for w in ws), Counter(new[w]["cls"] for w in ws)
        fl = sorted(new[w]["floor"] for w in ws if math.isfinite(new[w]["floor"]))
        fo = sorted(mine[w]["floor"] for w in ws if math.isfinite(mine[w]["floor"]))
        summary["counts"][size] = {"pre-registered": dict(c_old), "new": dict(c_new),
                                   "median_floor_v1": median(fo), "median_floor": median(fl)}
        print(f"  size {size}: pre-registered {dict(c_old)} -> new {dict(c_new)}; median floor "
              f"{median(fo):.1f} -> {median(fl):.1f}")
    for w, x in sorted(new.items()):
        if x["cls"] != x["v1"]:
            why = "fixed arm died" if x["dead"] else (
                f"gain {x['gain']:.1f} > floor {x['floor']:.1f} (was {mine[w]['floor']:.1f})")
            summary["flips"].append({"world": w, "from": x["v1"], "to": x["cls"], "why": why,
                                     "split": x["split"]})
            print(f"  flip {w}: {x['v1']} -> {x['cls']} ({why}; {x['split']})")
        if x["cls"] == "hard" and len(x["d"]) == 2:
            own = abs(x["d"][0] - x["d"][1]) / math.sqrt(2)
            summary["own_spread"].append({"world": w, "own": own, "sigma_ep": x["sigma_ep"],
                                          "ratio": own / x["sigma_ep"]})
    noisy = [y for y in summary["own_spread"] if y["ratio"] > 2]
    print(f"  hard worlds with 2 episodes: {len(summary['own_spread'])}; own episode spread > "
          f"2 x sigma_ep: {[(y['world'], round(y['ratio'], 2)) for y in noisy]}")
    v1w = {w: dict(x, cls=x["v1"]) for w, x in new.items()}
    for name, src_ in (("pre-registered", v1w), ("new", new)):
        sample = poc_a_sample(src_, "cls")
        summary["poc_a"][name] = [{"class": c, "world": w, "shocks": src_[w]["shocks"],
                                   "pattern": src_[w]["pattern"], "choice": src_[w]["choice"],
                                   "gain": src_[w]["gain"], "room": src_[w]["room"]}
                                  for c, w in sample]
        print(f"  PoC-A sample ({name}): " + ", ".join(
            f"{c}:{w.removeprefix('ws-atlas-v1-')}{'*' if src_[w]['shocks'] else ''}"
            for c, w in sample))
    return problems, summary


def canonical_hash(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "content_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path)
    ap.add_argument("--screen", type=Path, required=True, help="EXP-0051 dir")
    ap.add_argument("--ckpt", action="append", default=[], help="producer|orderer=PATH")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--reclassified", type=Path, help="task 5.3 derived dir")
    a = ap.parse_args()
    d, sd = a.exp_dir, a.screen
    problems: list[str] = []
    spec = json.loads((d / "spec.json").read_text())
    sspec = json.loads((sd / "spec.json").read_text())
    ep_path = d / "artifacts" / "episodes.csv"
    rows = read(ep_path if ep_path.exists() else d / "episodes.csv")
    table = arms()
    planned = [(t["s"], r) for t in table for r in range(t["R"])]

    cf = read(d / "contract_failures.csv")
    if cf:
        problems.append(f"contract_failures.csv has {len(cf)} rows")
    if spec.get("status") != "complete" or spec.get("kind") != "confirm":
        problems.append(f"spec status/kind {spec.get('status')}/{spec.get('kind')}")

    # commits
    c_commit = spec["pins"]["source"].get("commit")
    s_commit = spec["pins"]["screen"].get("source_commit")
    if c_commit != CONFIRM_COMMIT:
        problems.append(f"confirm commit {c_commit}")
    if s_commit != SCREEN_COMMIT or sspec["pins"]["source"].get("commit") != SCREEN_COMMIT:
        problems.append(f"screen commit {s_commit}")

    # selection (EXP-0051) -> worlds, fixed arms, split, screen class
    sel = {r["world"]: r for r in read(sd / "selection.csv")}
    if spec["pins"]["screen"].get("selection_sha256") != hashlib.sha256(
            (sd / "selection.csv").read_bytes()).hexdigest():
        problems.append("selection sha256 differs from the pinned one")
    recs = {r["world_id"]: r for r in sspec["pins"]["worlds"] if "world_id" in r}
    infos = {}
    for w in sel:
        lv = recs[w]["levers"]
        infos[w] = {"size": recs[w]["size"], "choice": lv["choice"], "pattern": lv["pattern"],
                    "shocks": bool(lv["shocks"]),
                    "differing": not (lv["pattern"] == "cyclic" and not lv["shocks"])}
    vals = values(rows)
    if set(vals) != set(sel):
        problems.append(f"worlds missing {sorted(set(sel) - set(vals))[:5]} extra "
                        f"{sorted(set(vals) - set(sel))[:5]}")

    # seeds
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tokenomics"))
    from tokenomics.eval.world_space import episode_seeds  # noqa: E402
    screen_eps: dict = defaultdict(set)
    s_ep = sd / "artifacts" / "episodes.csv"
    for r in read(s_ep if s_ep.exists() else sd / "episodes.csv"):
        if r["world"] in sel:
            screen_eps[r["world"]].add(int(r["episode"]))
    seed_bad = overlap = 0
    for w in sel:
        n = 2 if infos[w]["differing"] else 1
        if set(vals.get(w, {})) != set(episode_seeds(w, n, "confirm")):
            seed_bad += 1
        overlap += bool(set(vals.get(w, {})) & screen_eps[w])
    if seed_bad:
        problems.append(f"{seed_bad} worlds with episodes != episode_seeds(.., 'confirm')")
    if overlap:
        problems.append(f"{overlap} worlds share an episode seed with the screen")

    # completeness, digests, LP rows
    keys = Counter((r["world"], r["episode"], r["subject"], int(r["replicate"])) for r in rows)
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        problems.append(f"{len(dup)} duplicate rows, e.g. {dup[0]}")
    n_missing = digest_bad = lp_bad = n_we = 0
    for w, eps in vals.items():
        for e, ep in eps.items():
            n_we += 1
            n_missing += sum((w, str(e), s, r) not in keys for s, r in planned)
            pairs = {(x["demand_digest"], x["shock_digest"]) for rs in ep["arms"].values()
                     for x in rs}
            digest_bad += len(pairs) != 1
            lp_bad += ep["n_lp"] != 1
            extra = set(ep["arms"]) - {t["s"] for t in table}
            if extra:
                problems.append(f"{w}/{e}: unplanned subjects {sorted(extra)}")
    for name, n in (("planned rows missing", n_missing),
                    ("world-episodes with != 1 digest pair", digest_bad),
                    ("world-episodes with != 1 LP row", lp_bad)):
        if n:
            problems.append(f"{n} {name}")

    # bounds
    b2_viol = sum(num(r["bound_b2"]) is not None and float(r["units"]) > float(r["bound_b2"])
                  + TOL for r in rows)
    lp_viol = sum(float(r["units"]) > ep["lp"] + TOL for w, eps in vals.items()
                  for ep in eps.values() if ep["lp"] is not None
                  for rs in ep["arms"].values() for r in rs)
    if b2_viol or lp_viol:
        problems.append(f"bound violations: B2 {b2_viol}, B_lp {lp_viol}")

    # measures, spreads, classes
    mine = {w: measure(vals[w], table, (sel[w]["bar_arm"], sel[w]["reach_arm"]))
            for w in sorted(vals) if w in sel}
    sp = pooled(mine, infos)
    SP_CACHE.update(sp)
    sp_spec = spec.get("result", {}).get("spreads") or {}
    for k, v in sp.items():
        sv = sp_spec.get(k)
        if sv is None or not all(close(v[f], sv[f]) if not isinstance(v[f], bool)
                                 else v[f] == sv[f] for f in v):
            problems.append(f"spread {k}: probe {v} vs spec {sv}")
    wcsv = {r["world"]: r for r in read(d / "worlds.csv")}
    for w, m in mine.items():
        i = infos[w]
        se = se_of(m["n"], i["differing"], m["stoch"], sp.get(f"{i['size']}/{i['choice']}"))
        floor = max(3.0 * se, 0.01 * (m["bar"] or 0.0))
        sign, cls = classify(m, floor)
        m.update(se=se, floor=floor, sign=sign, cls=cls, screen_cls=sel[w]["class"],
                 split=sel[w]["split"])
        r = wcsv.get(w)
        if r is None:
            problems.append(f"{w} not in worlds.csv")
            continue
        for col, val in (("bar_arm", m["bar_arm"]), ("reach_arm", m["reach_arm"]),
                         ("class", cls), ("screen_class", sel[w]["class"]),
                         ("split", sel[w]["split"])):
            if (r[col] or None) != val:
                problems.append(f"{w} {col}: csv {r[col]} probe {val}")
        if r["sign_consistent"].lower() != str(sign).lower():
            problems.append(f"{w} sign_consistent: csv {r['sign_consistent']} probe {sign}")
        for col, val in (("bar", m["bar"]), ("reach", m["reach"]), ("gain", m["gain"]),
                         ("bar_fill", m["fill"]), ("ceiling", m["ceiling"]),
                         ("room", m["room"]), ("se", se), ("noise_floor", floor),
                         ("b2", m["b2"]), ("b_lp", m["lp"]), ("b_lp_base", m["lp_base"]),
                         ("arms_above_base", sum(m["above"].values()))):
            if not close(r[col], val):
                problems.append(f"{w} {col}: csv {r[col]} probe {val}")

    # deciding check
    want = deciding(vals, infos, mine, sp, table)
    got = read(d / "deciding_check.csv")
    if len(got) != len(want):
        problems.append(f"deciding_check.csv {len(got)} rows vs {len(want)}")
    for g, x in zip(got, want):
        for k, v in x.items():
            gv = g.get(k)
            ok = (str(v).lower() == (gv or "").lower() if isinstance(v, bool) or isinstance(v, str)
                  else close(gv, v))
            if not ok:
                problems.append(f"deciding {x.get('world') or x.get('branch') or x.get('arm')} "
                                f"{k}: csv {gv} probe {v}")
                break

    # atlas_v1.json
    atlas = json.loads((d / "atlas_v1.json").read_text())
    if canonical_hash(atlas) != atlas.get("content_hash"):
        problems.append("atlas_v1.json content hash does not match its body")
    aw = {x["record"]["world_id"]: x for x in atlas["worlds"]}
    if set(aw) != set(sel):
        problems.append("atlas_v1.json worlds != selection")
    for w, x in aw.items():
        m = mine.get(w)
        if m and (x["confirm_class"] != m["cls"] or x["screen_class"] != m["screen_cls"]
                  or x["split"] != sel[w]["split"] or not close(x["gain"], m["gain"])
                  or not close(x["room"], m["room"])):
            problems.append(f"atlas_v1.json {w} differs")
    if atlas["screen"].get("source_commit") != SCREEN_COMMIT or atlas["confirm"].get(
            "source_commit") != CONFIRM_COMMIT:
        problems.append("atlas_v1.json commits")

    # checkpoints
    pins = spec["pins"].get("checkpoints", {})
    for c in a.ckpt:
        name, path = c.split("=", 1)
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        print(f"ckpt {name}: {h[:12]} pin {str(pins.get(name))[:12]} "
              f"{'OK' if h == pins.get(name) else 'MISMATCH'}")
        if h != pins.get(name):
            problems.append(f"ckpt {name} sha mismatch")

    # ---- summary ----
    print(f"rows {len(rows)} planned {n_we * len(planned)}; worlds {len(mine)}; world-episodes "
          f"{n_we}; contract failures {len(cf)}; B2 viol {b2_viol}; B_lp viol {lp_viol}; "
          f"commits screen {str(s_commit)[:7]} confirm {str(c_commit)[:7]}")
    s_sp = sspec.get("result", {}).get("provisional_spreads", {})
    swc = {r["world"]: r for r in read(sd / "worlds.csv")}
    summary: dict = {"classes": {}, "transitions": {}, "hard_by_axis": {}, "spreads": sp,
                     "screen_spreads": s_sp, "deciding": [x for x in want if x["kind"] != "ttl"],
                     "worlds": {}}
    print("\n== pooled spreads: confirm (own) vs screen provisional (EXP-0050)")
    for k, v in sp.items():
        sv = s_sp.get(k, {})
        f = lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)  # noqa: E731
        print(f"  {k}: sigma_ep {f(v['sigma_ep'])} (df {v['df_ep']}{' merged' if v['merged_ep'] else ''}) "
              f"vs {f(sv.get('sigma_ep'))}; sigma_rep {f(v['sigma_rep'])} (df {v['df_rep']}"
              f"{' merged' if v['merged_rep'] else ''}) vs {f(sv.get('sigma_rep'))}")
    for size in (12, 42):
        ws = [w for w in mine if infos[w]["size"] == size]
        if not ws:
            continue
        summary["classes"][size] = dict(Counter(mine[w]["cls"] for w in ws))
        tr = Counter(f"{mine[w]['screen_cls']}->{mine[w]['cls']}" for w in ws)
        summary["transitions"][size] = dict(tr)
        fl_c = [mine[w]["floor"] for w in ws if math.isfinite(mine[w]["floor"])]
        fl_s = [num(swc[w]["noise_floor"]) for w in ws if num(swc[w]["noise_floor"]) is not None
                and math.isfinite(num(swc[w]["noise_floor"]))]
        print(f"\n== size {size}: {len(ws)} worlds, confirm classes {summary['classes'][size]}")
        print(f"  screen -> confirm: {dict(tr)}")
        print(f"  median noise floor: confirm {median(fl_c):.1f} vs screen {median(fl_s):.1f}"
              if fl_c and fl_s else "  noise floor n/a")
        for axis in ("choice", "pattern", "shocks"):
            g = defaultdict(Counter)
            for w in ws:
                g[infos[w][axis]][mine[w]["cls"]] += 1
            summary["hard_by_axis"][f"{size}/{axis}"] = {str(k): dict(v) for k, v in g.items()}
            print(f"  {axis}: " + "; ".join(f"{k}: {dict(v)}" for k, v in sorted(
                g.items(), key=lambda kv: str(kv[0]))))
        hard = [w for w in ws if mine[w]["cls"] == "hard"]
        if hard:
            print(f"  hard: median gain/bar {median(mine[w]["gain"] / mine[w]["bar"] if mine[w]["bar"] else math.inf for w in hard):+.3f}, "
                  f"reach arms {dict(Counter(mine[w]['reach_arm'] for w in hard))}")
        po = [(mine[w]["means"].get("portable", math.nan) - mine[w]["bar"]) / mine[w]["bar"]
              for w in ws if mine[w]["bar"]]
        print(f"  portable vs bar: median {median(po):+.3f}, above bar {sum(x > 0 for x in po)}/{len(po)}")
        lpr = [mine[w]["lp"] / mine[w]["lp_base"] for w in ws if mine[w]["lp"] and mine[w]["lp_base"]]
        print(f"  median room {median(mine[w]['room'] for w in ws):.3f}; median B_lp/B_lp_base "
              f"{median(lpr):.3f}; median bar/ceiling "
              f"{median(mine[w]['bar'] / mine[w]['ceiling'] for w in ws):.3f}")
    print("\n== deciding check (hard, E >= 4)")
    for x in want:
        if x["kind"] == "world":
            print(f"  {x['world']}: split-trained {x['split_vs_trained_gain']:+.1f} (floor "
                  f"{x['split_vs_trained_floor']:.1f}) whole-split {x['whole_vs_split_gain']:+.1f} "
                  f"(floor {x['whole_vs_split_floor']:.1f})")
        elif x["kind"] == "branch":
            print(f"  branch {x['branch']}: {x['count']}/{x['worlds']} holds={x['holds']}")
        elif x["scope"] == "all" and x["arm"] in (ME, MWW, TO) or (
                x["scope"] == "all" and x["share_inputs_short"] is not None
                and x["arm"].endswith(("round_robin", "random_whole", "/proportional"))):
            print(f"  ttl {x['scope']} {x['arm']}: inputs-short share "
                  f"{x['share_inputs_short']}")
    summary["sensitivity"] = sensitivity(vals, mine, infos, table)
    if a.reclassified:
        p5, summary["reclassified"] = reclassified_check(a.reclassified, d, vals, mine, infos,
                                                         wcsv)
        problems += p5
    for w, m in mine.items():
        summary["worlds"][w] = {k: m[k] for k in ("bar_arm", "bar", "reach_arm", "reach", "gain",
                                                  "se", "floor", "ceiling", "room", "fill",
                                                  "lp", "lp_base", "cls", "screen_cls", "split")}
        summary["worlds"][w].update(infos[w], portable=m["means"].get("portable"))
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print("\nPROBLEMS:" if problems else "\nALL CHECKS PASS")
    for p in problems[:40]:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
