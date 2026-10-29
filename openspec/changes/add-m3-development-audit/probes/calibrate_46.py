"""Task 4.6 (a) calibration probe for EXP-0051 (sizing.md §EXP-0051): no episodes.

Builds every config of one size's screen sample (`world_space.sample_worlds(atlas.ATLAS, size,
--worlds)`, default `atlas.SCREEN_WORLDS`: 512 at 12, 128 at 42) with its D10 world record (measured coordinates), and times the restored bounds
(B2 + B_base, env reset for the episode's demand included) on the screen's first episode seed.
Prints one JSON line per size: the range and quartiles of every measured coordinate, the count of
every categorical level, the coverage verdict (rho and sigma each span at least [0.05, 1.5],
every level present) and the bound's seconds per world-episode. With `--loads` it also prints,
per choice level, the worlds at the load quantiles (k_c x m_q) used by timing_46.py.

`calibrate_46.jsonl` was recorded with `--worlds 1024` at 12 (the sample before task 4.6's
reduction step 1) and the default 128 at 42.

Run from /srv/research/tokenomics/tokenomics:
  ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/calibrate_46.py --size 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter

from tokenomics.eval.atlas import ATLAS, COORDINATES, SCREEN_EPISODES, SCREEN_WORLDS
from tokenomics.eval.ceiling_bound import episode_demand, world_ceiling
from tokenomics.eval.families import SHOCKS_I_WINDOW
from tokenomics.eval.world_space import (CHOICES, PATTERNS, episode_seeds, sample_worlds,
                                         world_record)

SPAN = (0.05, 1.5)
LOAD_QUANTILES = (0.5, 0.75, 0.95)


def quantile(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, max(0, round(q * (len(v) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--loads", action="store_true")
    ap.add_argument("--worlds", type=int, help="sample size (default atlas.SCREEN_WORLDS[size])")
    args = ap.parse_args()
    start = time.perf_counter()
    points = sample_worlds(ATLAS, args.size, args.worlds or SCREEN_WORLDS[args.size],
                           SHOCKS_I_WINDOW)
    records, bound_s = [], []
    for p in points:
        records.append(world_record(p))
        cfg = p.config()
        t = time.perf_counter()
        world_ceiling(cfg, 100, episode_demand(cfg, 100, episode_seeds(p.world_id,
                                                                      SCREEN_EPISODES)[0]))
        bound_s.append(time.perf_counter() - t)
    coords = {c: [r["coordinates"][c] for r in records] for c in COORDINATES}
    levels = {"choice": Counter(r["levers"]["choice"] for r in records),
              "shocks": Counter(str(r["levers"]["shocks"]) for r in records),
              "pattern": Counter(r["levers"]["pattern"] for r in records)}
    missing = ([c for c in CHOICES[args.size] if not levels["choice"][c]]
               + [p for p in PATTERNS if not levels["pattern"][p]]
               + [s for s in ("False", "True") if not levels["shocks"][s]])
    spans = {c: (min(coords[c]) <= SPAN[0] and max(coords[c]) >= SPAN[1])
             for c in ("rho", "sigma")}
    print(json.dumps({
        "atlas": ATLAS, "size": args.size, "worlds": len(records),
        "coordinates": {c: {"min": round(min(v), 4), "q25": round(quantile(v, 0.25), 4),
                            "median": round(statistics.median(v), 4),
                            "q75": round(quantile(v, 0.75), 4), "max": round(max(v), 4)}
                        for c, v in coords.items()},
        "levels": {k: dict(sorted(v.items())) for k, v in levels.items()},
        "spans_ok": spans, "missing_levels": missing,
        "bound_seconds": {"mean": round(statistics.fmean(bound_s), 4),
                          "max": round(max(bound_s), 4)},
        "seconds": round(time.perf_counter() - start, 1)}), flush=True)
    if args.loads:
        for choice in CHOICES[args.size]:
            mine = sorted((r["levers"]["k_c"] * r["levers"]["m_q"], r["index"])
                          for r in records if r["levers"]["choice"] == choice)
            loads = [x for x, _ in mine]
            picks = {q: mine[min(len(mine) - 1, round(q * (len(mine) - 1)))]
                     for q in LOAD_QUANTILES}
            print(json.dumps({"choice": choice, "worlds": len(mine),
                              "load_mean": round(statistics.fmean(loads), 3),
                              "picks": {str(q): {"index": i, "load": round(x, 3)}
                                        for q, (x, i) in picks.items()}}), flush=True)


if __name__ == "__main__":
    main()
