"""Task 4.6 extrapolation for EXP-0051 / EXP-0052 (sizing.md §EXP-0051, §EXP-0052).

Reads timing_46.jsonl (timing_46.py's lines) and calibrate_46.jsonl (calibrate_46.py --loads),
both in this directory, and prints the per-cell table, the load factor L and the screen / confirm
arithmetic. Pure arithmetic, no episodes.

Per size x choice level, per world: a splitting router (the five deterministic roster rules,
`random`, `makeable_split`) costs what `proportional` costs, a whole-order router (`round_robin`,
`random_whole`, `makeable_weighted_whole`) what `round_robin` costs (proportional is up to 1.8x
round_robin on loaded 42-producer worlds; makeable_split measured between the two). The cell's
sampled-mix cost is the mean over its three load-quantile worlds (q50, q75, q95), about central
for the level's sample (by load alone it overstates, but c, shocks and pattern move cost as much;
5 random 42-producer worlds averaged 0.95 x their level's mix, sizing.md §EXP-0051);
L = mix / neutral. A linear fit of cost on k_c x m_q evaluated at the level's mean
load is printed beside it as a check.

Screen world-episode: 11 router jobs (7 splitting: 5 deterministic roster members, random x 1,
makeable_split; 4 whole: round_robin, random_whole x 2, makeable_weighted_whole x 1) + the
trained orderer; plus the 8
anchors at EXP-0050's per-cell times (sizing.md §EXP-0050). Confirm world-episode: 13 router jobs
(8 splitting with random x 2; 5 whole with random_whole and makeable_weighted_whole x 2) + trained orderer + portable, 2 episodes per
world, K = 64 at 12 and 32 at 42 at the screen's (equal) choice shares. Pod vCPU-h = local s /
3600 x 0.646; cpu16 wall = pod vCPU-h / 16.

Confirm recheck (`--selection <screen EXP dir>`, before EXP-0052 launches): reads the screen's
`selection.csv` and the selected worlds' records in its `spec.json` (`pins.worlds`: levers
k_c, m_q, pattern, shocks), prices each selected world at its size x choice level's confirm cost
interpolated piecewise-linearly in load = k_c x m_q through the level's 4 timed worlds (beyond the
highest timed load, extended at the level's least-squares slope, never below 0), times its
episodes (1 for cyclic demand without shocks, else 2), and compares the total with the 6 vCPU-h
stop.

Run: python3 extrapolate_46.py [--screen-12 N] [--screen-42 N] [--selection DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
POD = 0.646
JOBS = {"screen": (7, 4), "confirm": (8, 5)}  # (splitting, whole) router jobs per world-episode
CONFIRM_EPISODES = 2
K = {12: 64, 42: 32}
QS = ("q50", "q75", "q95")
# EXP-0050 per-cell times (sizing.md §EXP-0050): router, trained orderer, for the anchors
ANCHORS = {(12, "sparse_f2"): (0.825, 1.72), (12, "sparse_f4"): (1.173, 2.59),
           (12, "dense"): (1.281, 2.90), (42, "sparse_f2"): (4.544, 8.70),
           (42, "sparse_f4"): (6.688, 13.12), (42, "sparse_f7"): (9.755, 19.55),
           (42, "sparse_f14"): (18.502, 36.12), (42, "dense"): (7.095, 14.51)}


def fit_at(points: list[tuple[float, float]], x: float) -> float:
    mx = statistics.fmean(p for p, _ in points)
    my = statistics.fmean(c for _, c in points)
    sxx = sum((p - mx) ** 2 for p, _ in points)
    b = sum((p - mx) * (c - my) for p, c in points) / sxx if sxx else 0.0
    return my + b * (x - mx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-12", type=int, default=1024)
    ap.add_argument("--screen-42", type=int, default=128)
    ap.add_argument("--selection", help="a finished screen's EXP dir: price its selection")
    args = ap.parse_args()
    n_screen = {12: args.screen_12, 42: args.screen_42}
    t: dict = defaultdict(dict)
    load: dict = {}
    for line in (HERE / "timing_46.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("skipped"):
            continue
        scale = 100 / d["horizon"]  # a horizon-50 time is doubled
        t[(d["size"], d["choice"], d["world"])][d["arm"]] = d["seconds"] * scale
        load[(d["size"], d["choice"], d["world"])] = d["load"]
    load_mean = {}
    size = None
    for line in (HERE / "calibrate_46.jsonl").read_text().splitlines():
        d = json.loads(line)
        if "size" in d:  # a size's summary line precedes its per-choice load lines
            size = d["size"]
        elif "load_mean" in d:
            load_mean[(size, d["choice"])] = d["load_mean"]
    cells = sorted({(s, c) for s, c, _ in t})
    totals = {"screen": defaultdict(float), "confirm": defaultdict(float)}
    fitted = {"screen": defaultdict(float), "confirm": defaultdict(float)}
    print("size choice | world: screen/confirm s per world-episode | mix screen confirm | L")
    for s, c in cells:
        def cost(w: str, kind: str) -> float:
            a = t[(s, c, w)]
            split, whole = JOBS[kind]
            routers = split * a["proportional"] + whole * a["round_robin"]
            if kind == "screen":
                return routers + a["trained_orderer"]
            return routers + a["trained_orderer"] + a["portable"]
        row = []
        for w in ("neutral", *QS):
            a = t[(s, c, w)]
            row.append(f"{w} {cost(w, 'screen'):.1f}/{cost(w, 'confirm'):.1f}")
        mix = {k: statistics.fmean(cost(w, k) for w in QS) for k in ("screen", "confirm")}
        neutral = {k: cost("neutral", k) for k in ("screen", "confirm")}
        fits = {k: fit_at([(load[(s, c, w)], cost(w, k)) for w in ("neutral", *QS)],
                          load_mean[(s, c)]) for k in ("screen", "confirm")}
        fit = fits["screen"]
        share = 1 / len([x for x in cells if x[0] == s])
        fitted["screen"][s] += n_screen[s] * share * fits["screen"]
        fitted["confirm"][s] += K[s] * share * CONFIRM_EPISODES * fits["confirm"]
        totals["screen"][s] += n_screen[s] * share * mix["screen"]
        totals["confirm"][s] += K[s] * share * CONFIRM_EPISODES * mix["confirm"]
        print(f"{s} {c:10s} | {'; '.join(row)} | {mix['screen']:.1f} {mix['confirm']:.1f} | "
              f"L {mix['screen'] / neutral['screen']:.2f} fit-L "
              f"{fit / neutral['screen']:.2f} (mean load {load_mean[(s, c)]})")
    if args.selection:
        recheck(Path(args.selection), t, load)
        return
    anchors = sum(sum(JOBS["screen"]) * r + to for (s, c), (r, to) in ANCHORS.items())
    for kind in ("screen", "confirm"):
        local = sum(totals[kind].values()) + (anchors if kind == "screen" else 0.0)
        pod = local / 3600 * POD
        parts = ", ".join(f"{s}: {v:,.0f} s" for s, v in sorted(totals[kind].items()))
        extra = f", anchors {anchors:,.0f} s" if kind == "screen" else ""
        print(f"{kind}: {parts}{extra}; local {local:,.0f} s = {local / 3600:.2f} vCPU-h; "
              f"pod {pod:.2f} vCPU-h; cpu16 wall {pod / 16 * 60:.0f} min")
        flocal = sum(fitted[kind].values()) + (anchors if kind == "screen" else 0.0)
        fparts = ", ".join(f"{s}: {v:,.0f} s" for s, v in sorted(fitted[kind].items()))
        print(f"  (linear-fit check: {fparts}; local {flocal:,.0f} s; pod "
              f"{flocal / 3600 * POD:.2f} vCPU-h)")


def confirm_cost(a: dict) -> float:
    split, whole = JOBS["confirm"]
    return (split * a["proportional"] + whole * a["round_robin"] + a["trained_orderer"]
            + a["portable"])


def price(points: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear in load through `points`; beyond the last one, the least-squares slope
    (>= 0); below the first one, the first point's cost."""
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else max(y0, y1)
    mx = statistics.fmean(p for p, _ in pts)
    my = statistics.fmean(c for _, c in pts)
    sxx = sum((p - mx) ** 2 for p, _ in pts)
    slope = max(0.0, sum((p - mx) * (c - my) for p, c in pts) / sxx) if sxx else 0.0
    return pts[-1][1] + slope * (x - pts[-1][0])


def recheck(screen: Path, t: dict, load: dict) -> None:
    """The EXP-0052 recheck against a finished screen's selection (module docstring)."""
    import csv

    spec = json.loads((screen / "spec.json").read_text())
    recs = {r["world_id"]: r for r in spec["pins"]["worlds"] if "world_id" in r}
    with (screen / "selection.csv").open(newline="") as fh:
        selection = list(csv.DictReader(fh))
    total = 0.0
    by_size: dict = defaultdict(float)
    for row in selection:
        lv = recs[row["world"]]["levers"]
        size, choice = int(row["size"]), lv["choice"]
        pts = [(load[(size, choice, w)], confirm_cost(t[(size, choice, w)]))
               for w in ("neutral", *QS)]
        episodes = 1 if lv["pattern"] == "cyclic" and not lv["shocks"] else CONFIRM_EPISODES
        cost = episodes * price(pts, lv["k_c"] * lv["m_q"])
        by_size[size] += cost
        total += cost
    pod = total / 3600 * POD
    parts = ", ".join(f"{s}: {v:,.0f} s" for s, v in sorted(by_size.items()))
    print(f"confirm recheck on {len(selection)} selected worlds: {parts}; local {total:,.0f} s; "
          f"pod {pod:.2f} vCPU-h; cpu16 wall {pod / 16 * 60:.0f} min; stop 6: "
          f"{'UNDER - may launch' if pod <= 6 else 'ABOVE - do not launch, report to the supervisor'}")


if __name__ == "__main__":
    main()
