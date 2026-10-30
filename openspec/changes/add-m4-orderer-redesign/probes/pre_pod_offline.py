"""Task 1.2 (add-m4-orderer-redesign, design D11 checks c / d / e): offline, no episodes.

Reads the EXP-0052 confirm rows and the reclassified Atlas v2, writes §c, §d and §e of
notes/pre_pod_checks.md between fixed markers (other sections, e.g. task 1.3's §a / §b, are
kept) and prints the same block. Deterministic: no clock, no randomness, sorted iteration.

    ../../../../.venv/bin/python probes/pre_pod_offline.py [--out notes/pre_pod_checks.md]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path("/srv/research/tokenomics")
EXP = REPO / "experiments/EXP-0052-headroom-atlas-confirm"
EPISODES = EXP / "artifacts/episodes.csv"
ATLAS = EXP / "reclassified/atlas_v2.json"
ATLAS_WORLDS = EXP / "reclassified/worlds.csv"
PARAMS = REPO / "tokenomics/tokenomics/eval/registry/side_effect_params_v2.json"
CHANGE = REPO / "openspec/changes/add-m4-orderer-redesign"
DEFAULT_OUT = CHANGE / "notes/pre_pod_checks.md"

ATLAS_CONTENT_HASH = "220886a7cf12aaac0cc0a48dc03c57f2dc2484bb2af6185a6160ec509c8a2a23"
RHO_LOADED = 0.28
X_ROUNDED, Y_ROUNDED = 0.7516, 0.7302  # task text; the frozen exact values are used

RR = "entity_planner/round_robin"
MWW = "entity_planner/makeable_weighted_whole"
TR = "entity_planner/trained_orderer"  # D3's TR: trained orderer as deployed, entity_planner
PORTABLE = "portable"  # portable producer + trained orderer; §c context only
ARMS_C = (RR, MWW, TR)
SHORT = {RR: "RR", MWW: "MWW", TR: "TR", PORTABLE: "portable"}
EXPECTED_COUNTS = {"12/train": 23, "42/train": 14, "12/validation": 9}

BEGIN = "<!-- BEGIN pre_pod_offline.py (generated; do not edit by hand) -->"
END = "<!-- END pre_pod_offline.py -->"
COLS = ("world", "size", "episode", "subject", "replicate", "units", "dead", "r_start", "r_end",
        "consumer_fill_rate")


@dataclass(frozen=True)
class Row:
    world: str
    episode: str
    subject: str
    replicate: int
    units: float
    dead: bool
    r_start: float
    r_end: float
    fill: dict[str, float | None]


@dataclass
class World:
    wid: str
    size: int
    split: str
    cls: str
    rho: float
    choice: str
    pattern: str
    shocks: bool
    noise_floor: float
    bar_arm: str
    bar_csv: float
    rows: dict[str, list[Row]] = field(default_factory=dict)  # subject -> rows

    def episodes(self) -> list[str]:
        return sorted({r.episode for rs in self.rows.values() for r in rs})

    def ep_mean(self, subject: str) -> dict[str, float]:
        """Per-episode mean over replicates."""
        acc: dict[str, list[float]] = {}
        for r in self.rows[subject]:
            acc.setdefault(r.episode, []).append(r.units)
        return {e: statistics.fmean(v) for e, v in sorted(acc.items())}

    def mean(self, subject: str) -> float:
        """Equal-weight mean over the world's episodes of the replicate means (atlas D12)."""
        return statistics.fmean(self.ep_mean(subject).values())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atlas_content_hash(doc: dict) -> str:
    """Same canonical JSON as eval/atlas.py content_hash."""
    body = {k: v for k, v in doc.items() if k != "content_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def die(msg: str) -> None:
    print(f"STOP: {msg}", file=sys.stderr)
    sys.exit(2)


def load() -> tuple[dict, dict[str, World], list[str], dict]:
    atlas = json.loads(ATLAS.read_text())
    got = atlas_content_hash(atlas)
    if got != ATLAS_CONTENT_HASH or atlas.get("content_hash") != ATLAS_CONTENT_HASH:
        die(f"atlas content hash {got} != registered {ATLAS_CONTENT_HASH}")
    ep_sha = sha256_file(EPISODES)
    if ep_sha != atlas["reclassified_from"]["episodes_sha256"]:
        die(f"episodes.csv sha256 {ep_sha} != atlas reclassified_from.episodes_sha256")
    wcsv = {d["world"]: d for d in csv.DictReader(ATLAS_WORLDS.open())}
    worlds: dict[str, World] = {}
    for w in atlas["worlds"]:
        rec = w["record"]
        wid = rec["world_id"]
        d = wcsv[wid]
        if d["class"] != w["confirm_class"]:
            die(f"{wid}: worlds.csv class {d['class']} != atlas confirm_class {w['confirm_class']}")
        worlds[wid] = World(wid, int(rec["size"]), w["split"], w["confirm_class"],
                            float(rec["coordinates"]["rho"]), rec["levers"]["choice"],
                            rec["levers"]["pattern"], bool(rec["levers"]["shocks"]),
                            float(w["noise_floor"]), w["bar_arm"], float(d["bar"]))
    csv.field_size_limit(1 << 30)
    subjects: dict[str, int] = {}
    with EPISODES.open(newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        idx = [head.index(c) for c in COLS]
        for raw in rd:
            v = dict(zip(COLS, (raw[i] for i in idx)))
            subjects[v["subject"]] = subjects.get(v["subject"], 0) + 1
            if v["world"] not in worlds:
                die(f"episodes.csv world {v['world']} not in the atlas")
            fill = json.loads(v["consumer_fill_rate"]) if v["consumer_fill_rate"] else {}
            row = Row(v["world"], v["episode"], v["subject"], int(v["replicate"]),
                      float(v["units"]), v["dead"] == "true", float(v["r_start"]),
                      float(v["r_end"]), fill)
            worlds[v["world"]].rows.setdefault(v["subject"], []).append(row)
    for s in (RR, MWW, TR, PORTABLE):
        if s not in subjects:
            die(f"expected subject {s!r} missing from episodes.csv; subjects: {sorted(subjects)}")
    params = json.loads(PARAMS.read_text())
    return atlas, worlds, sorted(subjects.items()), params


def loaded(worlds: dict[str, World], size: int, split: str | None) -> list[World]:
    """Loaded (rho > 0.28), v2 class not collapse; split None = every split."""
    return sorted((w for w in worlds.values() if w.size == size and w.rho > RHO_LOADED
                   and w.cls != "collapse" and (split is None or w.split == split)),
                  key=lambda w: w.wid)


def f1(x: float) -> str:
    return f"{x:.1f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "0/0"


def chi2_cdf(x: float, k: int) -> float:
    """Regularized lower incomplete gamma P(k/2, x/2) by its power series (stdlib only)."""
    if x <= 0:
        return 0.0
    a, z = k / 2.0, x / 2.0
    term = total = 1.0 / a
    n = 1
    while term > total * 1e-15:
        term *= z / (a + n)
        total += term
        n += 1
    return total * math.exp(-z + a * math.log(z) - math.lgamma(a))


def chi2_ppf(q: float, k: int) -> float:
    """Quantile by bisection (deterministic)."""
    lo, hi = 0.0, 10.0 * k + 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if chi2_cdf(mid, k) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --- §c -------------------------------------------------------------------------------------

def flags(r: Row, x: float, y: float) -> dict[str, bool]:
    starve = any(v is not None and v < y for v in r.fill.values())
    rundown = r.r_end < x * r.r_start
    return {"dead": r.dead, "rundown": rundown, "starve": starve,
            "any": r.dead or rundown or starve}


def section_c(worlds: dict[str, World], x: float, y: float) -> list[str]:
    out = ["## §c. X / Y and death violation rates (RR / MWW / TR), loaded world-episodes", ""]
    out += [f"Filter: atlas v2 worlds with rho > {RHO_LOADED} and v2 class (`confirm_class`) not "
            "`collapse`, **every split** (train, validation, held-out, held-out-region), all "
            "confirm episodes; every replicate row counts (MWW has R = 2, so 2 rows per "
            "world-episode).",
            f"Flags per row: `dead` = system death; `rundown` = r_end < X·r_start with "
            f"X = {x!r} (frozen side_effect_params_v2; task text {X_ROUNDED}); `starve` = some "
            f"consumer's `consumer_fill_rate` < Y = {y!r} (task text {Y_ROUNDED}). A consumer "
            "whose fill rate is null (ordered 0) is skipped, never a violation; `any` = union.",
            ""]
    for size in (12, 42):
        ws = loaded(worlds, size, None)
        neps = sum(len(w.episodes()) for w in ws)
        by_split: dict[str, int] = {}
        for w in ws:
            by_split[w.split] = by_split.get(w.split, 0) + 1
        out += [f"### {size} producers: {len(ws)} worlds, {neps} world-episodes "
                f"({', '.join(f'{k} {v}' for k, v in sorted(by_split.items()))})", ""]
        out += ["| arm | subject | rows | dead | rundown | starve | any |",
                "|---|---|---|---|---|---|---|"]
        nulls = 0
        for s in ARMS_C + (PORTABLE,):
            rows = [r for w in ws for r in w.rows[s]]
            fl = [flags(r, x, y) for r in rows]
            nulls += sum(1 for r in rows for v in r.fill.values() if v is None)
            tag = SHORT[s] + (" (context)" if s == PORTABLE else "")
            out.append(f"| {tag} | `{s}` | {len(rows)} | "
                       + " | ".join(pct(sum(f[k] for f in fl), len(rows))
                                    for k in ("dead", "rundown", "starve", "any")) + " |")
        out += ["", f"Null per-consumer fill rates skipped across these rows: {nulls}.", ""]
        # Discordance: pair rows on (world, episode); MWW's replicates each pair with the
        # deterministic arm's single row of that episode; MWW vs itself never paired.
        out += ["Discordance (pairs on the same world-episode; each MWW replicate row is paired "
                "with the other arm's row of that episode): share of pairs where only A / only B "
                "is flagged.", "",
                "| A vs B | pairs | flag | A only | B only | both |", "|---|---|---|---|---|---|"]
        for a, b in ((RR, MWW), (RR, TR), (MWW, TR)):
            pairs = []
            for w in ws:
                ra = {(r.episode, r.replicate): r for r in w.rows[a]}
                rb = {(r.episode, r.replicate): r for r in w.rows[b]}
                for (e, ia), x_a in sorted(ra.items()):
                    for (e2, ib), x_b in sorted(rb.items()):
                        if e2 == e:
                            pairs.append((flags(x_a, x, y), flags(x_b, x, y)))
            for k in ("dead", "rundown", "starve", "any"):
                ao = sum(1 for p, q in pairs if p[k] and not q[k])
                bo = sum(1 for p, q in pairs if q[k] and not p[k])
                bt = sum(1 for p, q in pairs if p[k] and q[k])
                out.append(f"| {SHORT[a]} vs {SHORT[b]} | {len(pairs)} | {k} | "
                           f"{pct(ao, len(pairs))} | {pct(bo, len(pairs))} | "
                           f"{pct(bt, len(pairs))} |")
        out.append("")
    # Sensitivity of the rounded thresholds.
    changed = 0
    for size in (12, 42):
        for w in loaded(worlds, size, None):
            for s in ARMS_C:
                for r in w.rows[s]:
                    if flags(r, x, y) != flags(r, X_ROUNDED, Y_ROUNDED):
                        changed += 1
    out += [f"Rows of RR / MWW / TR whose flags change with the rounded X / Y "
            f"({X_ROUNDED} / {Y_ROUNDED}): {changed}.", ""]
    mww = [flags(r, x, y)["any"] for size in (12, 42) for w in loaded(worlds, size, None)
           for r in w.rows[MWW]]
    most = sum(mww) > len(mww) / 2
    out += [f"D11 trigger: the A0-like arm (MWW) is flagged (`any`) on {pct(sum(mww), len(mww))} "
            "of its loaded rows at 12 and 42 → "
            + ("most: D6's relative rule (flags no worse than RR's, not absolute X / Y) is "
               "confirmed as necessary." if most else "not most: D11's confirmation does not "
               "fire."), ""]
    return out


# --- §d -------------------------------------------------------------------------------------

@dataclass
class Gap:
    world: str
    n_pairs: int
    rr: float
    tr: float
    sub: float
    g: float
    atlas_floor: float
    own3se: float | None
    floor: float
    alt_floor: float
    is_gap: bool
    is_gap_alt: bool
    rec: float | None


def own_3se(diffs: list[float]) -> float | None:
    """3 x SE of the paired per-world differences; None below 2 paired observations (n = 1)."""
    if len(diffs) < 2:
        return None
    return 3.0 * statistics.stdev(diffs) / math.sqrt(len(diffs))


def alt_atlas_floor(w: World, sigma_ep: float | None, n_eps: int, bar: float) -> float:
    """Sensitivity only: atlas floor formula re-evaluated for a deterministic pair (no sigma_rep
    term): max(3 sigma_ep / sqrt(n) [differing], 0.01 bar)."""
    differing = not (w.pattern == "cyclic" and not w.shocks)
    se = (sigma_ep / math.sqrt(n_eps)) if (differing and sigma_ep is not None) else 0.0
    return max(3.0 * se, 0.01 * bar)


def gaps(ws: list[World], atlas: dict, sub: str) -> list[Gap]:
    out = []
    for w in ws:
        rr, tr, sb = w.ep_mean(RR), w.ep_mean(TR), w.ep_mean(sub)
        eps = sorted(rr)
        if sorted(tr) != eps or sorted(sb) != eps:
            die(f"{w.wid}: episodes differ across RR / TR / {sub}")
        diffs = [rr[e] - tr[e] for e in eps]  # R = 1 each: pairs = episodes
        g = statistics.fmean(diffs)
        o = own_3se(diffs)
        floor = max(w.noise_floor, o) if o is not None else w.noise_floor
        sp = atlas["spreads"].get(f"{w.size}/{w.choice}", {})
        alt = alt_atlas_floor(w, sp.get("sigma_ep"), len(eps), w.mean(w.bar_arm))
        alt_floor = max(alt, o) if o is not None else alt
        is_gap = g > floor
        rec = (statistics.fmean(sb.values()) - statistics.fmean(tr.values())) / g if g else None
        out.append(Gap(w.wid, len(diffs), statistics.fmean(rr.values()),
                       statistics.fmean(tr.values()), statistics.fmean(sb.values()), g,
                       w.noise_floor, o, floor, alt_floor, is_gap, g > alt_floor, rec))
    return out


def reading(gs: list[Gap], use_alt: bool = False) -> tuple[str, int, float | None, float | None]:
    """D3 primary reading, first match wins; returns (reading, n_gap, share rec>=2/3, median)."""
    gw = [g for g in gs if (g.is_gap_alt if use_alt else g.is_gap)]
    if len(gw) < 5:
        return "no_gap", len(gw), None, None
    recs = [g.rec for g in gw]
    share = sum(1 for r in recs if r >= 2.0 / 3.0) / len(recs)
    med = statistics.median(recs)
    if share >= 2.0 / 3.0:
        return "decoder", len(gw), share, med
    if med >= 1.0 / 3.0:
        return "decoder_partial", len(gw), share, med
    return "not_decoder", len(gw), share, med


def gap_table(gs: list[Gap], ws: dict[str, World]) -> list[str]:
    out = ["| world | choice | pattern/shocks | pairs | RR_w | TR_w | MWW_w | G_w | atlas floor "
           "| own 3·SE | floor_w | gap | rec_w |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in gs:
        w = ws[g.world]
        out.append(f"| {g.world.removeprefix('ws-atlas-v1-')} | {w.choice} | {w.pattern}/"
                   f"{'shock' if w.shocks else 'calm'} | {g.n_pairs} | {f1(g.rr)} | {f1(g.tr)} | "
                   f"{f1(g.sub)} | {f1(g.g)} | {f1(g.atlas_floor)} | "
                   f"{f1(g.own3se) if g.own3se is not None else '—'} | {f1(g.floor)} | "
                   f"{'yes' if g.is_gap else 'no'} | "
                   f"{f3(g.rec) if g.rec is not None else '—'} |")
    return out


def section_d(worlds: dict[str, World], atlas: dict) -> list[str]:
    ws12 = loaded(worlds, 12, "train")
    ws42 = loaded(worlds, 42, "train")
    g12 = gaps(ws12, atlas, MWW)
    rd, n_gap, share, med = reading(g12)
    rd_alt, n_gap_alt, share_alt, med_alt = reading(g12, use_alt=True)
    gw = [g for g in g12 if g.is_gap]
    out = ["## §d. Step-1 primary reading with MWW in place of TS (prediction for EXP-0057)", ""]
    out += [f"**Predicted primary reading under `faithful`: `{rd}`** — {n_gap} gap worlds of "
            f"{len(g12)}; rec_w ≥ 2/3 on "
            + (f"{share:.3f} of gap worlds; median rec_w = {med:.3f}." if share is not None
               else "— (fewer than 5 gap worlds)."), ""]
    out += ["This is the written prediction for EXP-0057's primary reading under the fidelity "
            "reading `faithful` (TS ≈ MWWk within floor on ≥ 2/3 of worlds): if TS carries "
            "MWW's target, EXP-0057 should read as above. MWW stands in for MWWk: same weights, "
            "a different random stream (`action_space.np_random`, not the keyed uniform).", ""]
    out += [f"Worlds: atlas v2 split `train`, size 12, rho > {RHO_LOADED}, v2 class not "
            f"`collapse`: {len(ws12)} worlds, {sum(len(w.episodes()) for w in ws12)} "
            "world-episodes (the confirm's seeds; 1 episode on cyclic shock-free worlds).", ""]
    out += ["Readings used (D3 with TS → MWW):",
            "- RR_w, TR_w, MWW_w = equal-weight mean over the world's confirm episodes of the "
            "replicate means (RR, TR: R = 1; MWW: R = 2, `action_space.np_random`, the atlas arm "
            "itself, so this is MWW, not MWWk).",
            "- G_w = RR_w − TR_w; gap world: G_w > floor_w; rec_w = (MWW_w − TR_w) / G_w, no floor "
            "on rec_w (D3 gives none).",
            "- floor_w (pair RR vs TR) = max(atlas `noise_floor`, 3·SE of the per-episode RR − TR "
            "differences). The atlas term is the world's recorded `noise_floor` as is: the run's "
            "counts are the confirm's counts, so no rescaling. The own term needs ≥ 2 paired "
            "observations; RR and TR are both R = 1, so paired observations = episodes "
            "(\"where the two arms differ\" read as the count of paired episode × replicate "
            "observations, not as the count of non-zero differences; see the zero-difference "
            "check below). 1-episode worlds use the atlas floor alone (n = 1 rule).",
            "- Precedence, first match wins: `no_gap` (< 5 gap worlds) → `decoder` (rec_w ≥ 2/3 on "
            "≥ 2/3 of gap worlds) → `decoder_partial` (median rec_w ≥ 1/3) → `not_decoder`.",
            ""]
    zero = sum(1 for w in ws12 for e in w.episodes() if w.ep_mean(RR)[e] == w.ep_mean(TR)[e])
    out += [f"Zero RR − TR per-episode differences among these world-episodes: {zero} (so the "
            "two readings of \"paired observations\" coincide here" + (")." if zero == 0 else
                                                                         " — they do NOT)."),
            ""]
    out += gap_table(g12, worlds) + [""]
    recs = sorted(g.rec for g in gw)
    out += [f"Gap worlds ({n_gap}): "
            + ", ".join(g.world.removeprefix("ws-atlas-v1-") for g in gw) + ".",
            "Sorted rec_w on gap worlds: " + ", ".join(f3(r) for r in recs) + ".", ""]
    flip = "same branch" if rd_alt == rd else f"branch changes to `{rd_alt}`"
    out += [f"Sensitivity (not the reading): replacing the recorded atlas `noise_floor` by the "
            "atlas formula re-evaluated for the deterministic pair (max(3·sigma_ep/√n "
            "[differing worlds], 0.01·bar), no replicate term, plus the own 3·SE as above) gives "
            f"`{rd_alt}` with {n_gap_alt} gap worlds"
            + (f", share rec_w ≥ 2/3 = {share_alt:.3f}, median {med_alt:.3f}"
               if share_alt is not None else "") + f" ({flip}).", ""]
    # EXP-0056 worlds and the TR-vs-MWW sign, for the record.
    e56 = [g for g in g12 if g.world.endswith(("12-0060", "12-0111", "12-0344"))]
    out += ["EXP-0056 worlds in the set: "
            + ", ".join(f"{g.world.removeprefix('ws-atlas-v1-')} (G_w {f1(g.g)}, gap "
                        f"{'yes' if g.is_gap else 'no'}, rec_w "
                        f"{f3(g.rec) if g.rec is not None else '—'})" for g in e56) + ".", ""]
    g42 = gaps(ws42, atlas, MWW)
    rd42, n42, s42, m42 = reading(g42)
    out += [f"### 42 producers (reported, never in the reading): {len(ws42)} worlds", "",
            f"Same computation: {n42} gap worlds; the rule applied to them would give `{rd42}`"
            + (f" (share {s42:.3f}, median {m42:.3f})" if s42 is not None else "") + ".", ""]
    out += gap_table(g42, worlds) + [""]
    return out


# --- §e -------------------------------------------------------------------------------------

def section_e(worlds: dict[str, World]) -> list[str]:
    ws = loaded(worlds, 12, "validation")
    out = ["## §e. Paired per-episode MWW − RR on the loaded 12-producer validation worlds", ""]
    out += [f"Worlds: atlas v2 split `validation`, size 12, rho > {RHO_LOADED}, v2 class not "
            f"`collapse`: {len(ws)} worlds.",
            "d_we = mean over MWW's 2 replicates − RR (R = 1) on episode e of world w. "
            "D_w = mean_e d_we.", ""]
    out += ["| world | episodes | bar arm | bar | d_we | D_w |", "|---|---|---|---|---|---|"]
    ss, df = 0.0, 0
    ss_w: dict[str, float] = {}  # world -> its share of the pooled sum of squares
    rep_ss, rep_df = 0.0, 0
    resid: list[float] = []
    bars, dws = [], []
    for w in ws:
        m, r = w.ep_mean(MWW), w.ep_mean(RR)
        d = [m[e] - r[e] for e in sorted(r)]
        dw = statistics.fmean(d)
        bar = w.mean(w.bar_arm)
        if abs(bar - w.bar_csv) > 1e-9:
            die(f"{w.wid}: bar from episodes {bar} != worlds.csv bar {w.bar_csv}")
        bars.append(bar)
        dws.append(dw)
        if len(d) >= 2:
            ss_w[w.wid] = sum((x - dw) ** 2 for x in d)
            ss += ss_w[w.wid]
            df += len(d) - 1
            k = math.sqrt(len(d) / (len(d) - 1))
            resid += [(x - dw) * k for x in d]
        by_ep: dict[str, list[float]] = {}
        for row in w.rows[MWW]:
            by_ep.setdefault(row.episode, []).append(row.units)
        for v in by_ep.values():
            if len(v) >= 2:
                mu = statistics.fmean(v)
                rep_ss += sum((x - mu) ** 2 for x in v)
                rep_df += len(v) - 1
        out.append(f"| {w.wid.removeprefix('ws-atlas-v1-')} | {len(d)} | "
                   f"{SHORT.get(w.bar_arm, w.bar_arm)} | {f1(bar)} | "
                   f"{', '.join(f1(x) for x in d)} | {f1(dw)} |")
    n_w = len(ws)
    mean_bar = statistics.fmean(bars)
    s_d = math.sqrt(ss / df)  # per-episode sd of d at MWW R = 2
    s_mad = statistics.median(abs(x) for x in resid) / 0.6744897501960817
    s_rep2 = rep_ss / rep_df  # MWW replicate variance
    s_ep2 = s_d ** 2 - s_rep2 / 2  # episode component without replicate noise

    def mde(sd: float, n_e: int) -> float:
        return 3.0 * sd / math.sqrt(n_e) / math.sqrt(n_w)

    out += ["", f"**Mean validation bar used: {f1(mean_bar)}** (equal-weight mean over the "
            f"{n_w} worlds of the atlas bar = the recorded bar arm's mean units, recomputed from "
            "episodes.csv and checked equal to reclassified/worlds.csv `bar`).",
            f"Mean paired difference (equal-weight world mean of D_w): {f1(statistics.fmean(dws))}.",
            "",
            "Estimator: SE of the equal-weight mean over the 9 worlds of paired per-episode "
            "differences, from the pooled within-world SD of d_we "
            f"(df = Σ(n_w − 1) = {df}, worlds with 2 episodes): s_d = {f1(s_d)}; "
            "SE(n episodes per world) = s_d / √n / √9; detectable effect = 3·SE. Between-world "
            "spread of D_w is a fixed property of the world set, not sampling noise, so it is "
            "not in the SE.", "",
            "| episodes per world | replicates | 3·SE (units) | share of mean bar |",
            "|---|---|---|---|"]
    rows_e = []
    for n_e in (4, 8):
        v = mde(s_d, n_e)
        rows_e.append((n_e, v / mean_bar))
        out.append(f"| {n_e} | 2 (as confirm) | {f1(v)} | {100 * v / mean_bar:.2f}% |")
    for n_e in (4, 8):
        if s_ep2 > 0:
            v = mde(math.sqrt(s_ep2 + s_rep2), n_e)
            out.append(f"| {n_e} | 1 (rescaled) | {f1(v)} | {100 * v / mean_bar:.2f}% |")
    def rule(sd: float) -> str:
        s4, s8 = mde(sd, 4) / mean_bar, mde(sd, 8) / mean_bar
        return ("≤ 2% at 4 episodes → 4-episode reads" if s4 <= 0.02 else
                ("> 3% at 8 episodes → to the user before the pilot" if s8 > 0.03 else
                 "> 2% at 4 episodes → 8-episode reads"))

    def pct_bar(sd: float, n_e: int = 4) -> str:
        return f"{100 * mde(sd, n_e) / mean_bar:.2f}%"

    out += ["", f"Replicate decomposition: MWW replicate SD = {f1(math.sqrt(s_rep2))} "
            f"(df {rep_df}); episode component without replicate noise = "
            + (f"{f1(math.sqrt(s_ep2))}" if s_ep2 > 0 else "≤ 0 (not separable)") + ".",
            f"Robust check: pooled MAD-sigma of the same residuals = {f1(s_mad)} → 3·SE at 4 "
            f"episodes {f1(mde(s_mad, 4))} ({pct_bar(s_mad)}).", ""]
    # Fragility (review round 1; DIAGNOSTICS 21: a pooled SD carried by few worlds).
    top = max(ss_w, key=lambda k: (ss_w[k], k))
    loo = {k: math.sqrt((ss - v) / (df - 1)) for k, v in sorted(ss_w.items())}
    loo_rest = [v for k, v in loo.items() if k != top]
    lo_q, hi_q = chi2_ppf(0.025, df), chi2_ppf(0.975, df)
    sd_lo, sd_hi = s_d * math.sqrt(df / hi_q), s_d * math.sqrt(df / lo_q)
    sd_at_2 = 0.02 * mean_bar * math.sqrt(4) * math.sqrt(n_w) / 3.0
    r1 = math.sqrt(s_ep2 + s_rep2) if s_ep2 > 0 else None
    out += ["### §e fragility (the 4-vs-8 episode choice is not settled here)", "",
            f"- Share of the pooled sum of squares (df {df}) per world: "
            + ", ".join(f"{k.removeprefix('ws-atlas-v1-')} {100 * v / ss:.1f}%"
                        for k, v in sorted(ss_w.items(), key=lambda kv: (-kv[1], kv[0])))
            + f". The top world ({top.removeprefix('ws-atlas-v1-')}) carries "
            f"{100 * ss_w[top] / ss:.1f}%.",
            "- Leave-one-out (drop one world from the SD pool only; the 9-world mean and the mean "
            f"bar are kept), detectable effect at 4 episodes: dropping {top.removeprefix('ws-atlas-v1-')} "
            f"→ {pct_bar(loo[top])}; dropping any other world → {pct_bar(min(loo_rest))}–"
            f"{pct_bar(max(loo_rest))} ("
            + ", ".join(f"{k.removeprefix('ws-atlas-v1-')} {pct_bar(v)}" for k, v in loo.items()
                        if k != top) + ").",
            f"- 95% χ² interval on s_d at df {df}: [{f1(sd_lo)}, {f1(sd_hi)}] → detectable effect "
            f"at 4 episodes [{pct_bar(sd_lo)}, {pct_bar(sd_hi)}], at 8 episodes "
            f"[{pct_bar(sd_lo, 8)}, {pct_bar(sd_hi, 8)}]. The interval spans "
            + ("4-episode and 8-episode reads" if mde(sd_lo, 4) / mean_bar <= 0.02
               < mde(sd_hi, 4) / mean_bar else "one side of the 2% line")
            + ("; its 8-episode upper end crosses the > 3%-at-8 user trigger."
               if mde(sd_hi, 8) / mean_bar > 0.03 else
               "; its 8-episode upper end stays below the > 3%-at-8 user trigger (the trigger "
               "is defined at 8 episodes, so the 4-episode upper end above 3% does not fire it)."),
            f"- The 2% line at 4 episodes sits at s_d = {f1(sd_at_2)}, "
            f"{100 * (sd_at_2 / s_d - 1):.1f}% above the point estimate {f1(s_d)}.",
            f"- At R = 1 (rescaled, s = {f1(r1)}): {pct_bar(r1)} at 4 episodes → rule gives "
            f"\"{rule(r1)}\"." if r1 is not None else "- R = 1: not separable.",
            "",
            f"D7 power rule, **R = 2 point estimate only (first look; not robust, see above; task "
            f"5.4 decides episodes and R)**: {rule(s_d)}. At R = 1 the same rule gives "
            + (f"{rule(r1)}." if r1 is not None else "—."), ""]
    return out


def render(atlas: dict, worlds: dict[str, World], subjects: list[tuple[str, int]],
           params: dict) -> list[str]:
    x, y = float(params["x"]), float(params["y"])
    counts = {"12/train": len(loaded(worlds, 12, "train")),
              "42/train": len(loaded(worlds, 42, "train")),
              "12/validation": len(loaded(worlds, 12, "validation"))}
    ok = counts == EXPECTED_COUNTS
    out = [BEGIN, "", "Generated by `/srv/research/tokenomics/openspec/changes/"
           "add-m4-orderer-redesign/probes/pre_pod_offline.py` (task 1.2, design D11 c / d / e; "
           "offline, no episodes).", "",
           f"- Episodes: `{EPISODES}` (sha256 {sha256_file(EPISODES)}, equal to the atlas's "
           "`reclassified_from.episodes_sha256`).",
           f"- Atlas: `{ATLAS}` (content_hash {atlas['content_hash']}, recomputed and checked).",
           f"- X / Y: `{PARAMS}` (x = {x!r}, y = {y!r}).",
           "- Subjects in episodes.csv (rows): "
           + ", ".join(f"`{s}` ({n})" for s, n in subjects) + ".",
           f"- Matched: RR = `{RR}`; MWW = `{MWW}`; TR (\"the trained orderer\", D3's TR with "
           f"`entity_planner`) = `{TR}`; `{PORTABLE}` (portable producer + trained orderer) is "
           "shown in §c as context only.",
           f"- World counts (rho > {RHO_LOADED}, v2 class not collapse): 12/train "
           f"{counts['12/train']}, 42/train {counts['42/train']}, 12/validation "
           f"{counts['12/validation']} — " + ("matches 23 / 14 / 9." if ok else
                                                f"EXPECTED {EXPECTED_COUNTS}, MISMATCH."), ""]
    out += section_c(worlds, x, y)
    out += section_d(worlds, atlas)
    out += section_e(worlds)
    out.append(END)
    return out


def write(out_path: Path, block: str) -> None:
    if out_path.exists():
        text = out_path.read_text()
        if BEGIN in text and END in text:
            pre, rest = text.split(BEGIN, 1)
            post = rest.split(END, 1)[1]
            out_path.write_text(pre + block + post)
            return
        out_path.write_text(text.rstrip("\n") + "\n\n" + block + "\n")
        return
    out_path.write_text("# Pre-pod local checks (add-m4-orderer-redesign, design D11)\n\n"
                        "§a and §b come from task 1.3 (`probes/trace_agents.py`); §c, §d and §e "
                        "from task 1.2 (below).\n\n" + block + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stdout-only", action="store_true", help="print, do not write --out")
    a = ap.parse_args()
    atlas, worlds, subjects, params = load()
    block = "\n".join(render(atlas, worlds, subjects, params))
    print(block)
    if not a.stdout_only:
        write(a.out, block)


if __name__ == "__main__":
    main()
