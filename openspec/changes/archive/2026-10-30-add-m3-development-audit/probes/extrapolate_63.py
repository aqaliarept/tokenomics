"""Task 6.3: extrapolate EXP-0053 (sizing.md §EXP-0053) from `timing_63.jsonl`.

Per teacher arm and world, the horizon-100 cells give search seconds per cycle (timed streams x
their seconds, scaled to every stream the cycle would search; here all streams were timed). An
episode = 100 cycles x that rate + the sampler (posterior arms, measured per call) + the
floor-speed episode itself (`rr` on 12-0060) + the recorder's replay (posterior arms; the
`sampler` mode's played time). Every sampled world and arm is priced by its own cells: `central`
= the mean of its timed cycles, `high` = its most expensive timed cycle; an unmeasured world or
arm is an error (no pooled rate: concern #346, 12-0408 cost 3x the pooled worst). The
blind / reach reruns cost §EXP-0052's confirm world-episode at 12 producers (18 router-equivalents
with `portable`, which the audit does not run: conservative). Local s x 0.646 = pod vCPU-s.

Wall per pod shape: the driver's own job order (per world in sample order: the blind / reach
arms, `floor`, `d2`, `ub`), each job taken by the first free worker (`ProcessPoolExecutor`
FIFO); billed vCPU-h = vCPU x makespan. Also the longest-first order a driver change could use.
Run: python3 extrapolate_63.py [timing_63.jsonl] [--drop-ub] [--sample-8]  (stdlib only; the
sample is pre-registered; the flags apply D5's reduction steps 1 and 2)
"""

from __future__ import annotations

import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

POD = 0.646
SAFETY = 2.0
HORIZON = 100
STOP = 8.0  # pod vCPU-h (D5)
SHAPES = (32, 16, 8, 4)
# D5 sample (teacher_audit.pre_registered_sample on EXP-0052 reclassified/atlas_v2.json):
# (world, class, scenario source of d2)
SAMPLE = [
    ("ws-atlas-v1-12-0060", "hard", "posterior"),
    ("ws-atlas-v1-12-0176", "hard", "deterministic"),
    ("ws-atlas-v1-12-0005", "hard", "deterministic"),
    ("ws-atlas-v1-12-0348", "hard", "deterministic"),
    ("ws-atlas-v1-12-0111", "open", "posterior"),
    ("ws-atlas-v1-12-0344", "open", "posterior"),
    ("ws-atlas-v1-12-0230", "open", "posterior"),
    ("ws-atlas-v1-12-0308", "open", "posterior"),
    ("ws-atlas-v1-12-0392", "open", "posterior"),
    ("ws-atlas-v1-12-0254", "open", "posterior"),
    ("ws-atlas-v1-12-0155", "open", "deterministic"),
    ("ws-atlas-v1-12-0408", "open", "posterior"),
]
# D5 reduction step 2: 4 hard + 4 open by the same rule (hard: 0060 the only shocked, then the
# best unshocked; open: the best 2 shocked and 2 unshocked)
SAMPLE_8 = {"ws-atlas-v1-12-0060", "ws-atlas-v1-12-0176", "ws-atlas-v1-12-0005",
            "ws-atlas-v1-12-0348", "ws-atlas-v1-12-0111", "ws-atlas-v1-12-0344",
            "ws-atlas-v1-12-0308", "ws-atlas-v1-12-0392"}
BASELINE_JOBS = 14  # blind + reach arms of the confirm without `portable`
BASELINE_S = 46.1  # §EXP-0052 confirm s per world-episode, 12 dense (the largest 12 level)


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def rates(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(world, arm) -> per-cycle search seconds (h100 cells), streams, sampler seconds."""
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("mode") == "cell" and r["horizon"] == HORIZON and r["timed_streams"]:
            by[(r["world"], r["arm"])].append(r)
    out = {}
    for k, cs in by.items():
        per = [c["timed_seconds"] * c["streams"] / c["timed_streams"] for c in cs]
        out[k] = {"source": cs[0]["source"], "cycles": [c["cycle"] for c in cs],
                  "s_per_cycle": fmean(per), "per_cycle": per,
                  "streams": fmean(c["streams"] for c in cs),
                  "s_per_stream": fmean(c["s_per_stream"] for c in cs),
                  "sampler": fmean(c["sampler_s"] for c in cs if c["sampler_s"] is not None)
                  if any(c["sampler_s"] is not None for c in cs) else 0.0}
    return out


def schedule(jobs: list[float], workers: int) -> float:
    free = [0.0] * workers
    heapq.heapify(free)
    end = 0.0
    for d in jobs:
        t = heapq.heappop(free) + d
        end = max(end, t)
        heapq.heappush(free, t)
    return end


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    drop_ub, eight = "--drop-ub" in sys.argv, "--sample-8" in sys.argv
    path = Path(args[0]) if args else Path(__file__).with_name("timing_63.jsonl")
    rows = load(path)
    r = rates(rows)
    rr = fmean(x["seconds"] for x in rows if x.get("mode") == "rr"
               and x["world"].startswith("ws-"))
    replay = fmean(x["played_s_with_recorder"] for x in rows
                   if x.get("mode") == "sampler-summary")
    print("measured horizon-100 cells (local s):")
    for (w, a), v in sorted(r.items()):
        print(f"  {w} {a:3} {v['source']:13} cycles {v['cycles']}: streams/cycle "
              f"{v['streams']:.1f}, s/stream {v['s_per_stream']:.3f}, s/cycle "
              f"{v['s_per_cycle']:.2f} ({', '.join(f'{p:.1f}' for p in v['per_cycle'])}), "
              f"sampler {v['sampler'] * 1000:.0f} ms")
    samplers = [v["sampler"] for v in r.values() if v["source"] == "posterior"]
    print(f"floor-speed episode {rr:.2f} s, recorder replay {replay:.2f} s, sampler "
          f"{fmean(samplers) * 1000:.0f} ms/call")
    results = {}
    for case in ("central", "high"):
        jobs: list[tuple[str, float]] = []
        named: list[tuple[str, str, float]] = []
        totals = defaultdict(float)
        for w, _, src in SAMPLE:
            if eight and w not in SAMPLE_8:
                continue
            for _ in range(BASELINE_JOBS):
                jobs.append(("baseline", BASELINE_S / BASELINE_JOBS))
            jobs.append(("floor", rr))
            arms = [("d2", src)] + ([("ub", "clairvoyant")] if src == "posterior"
                                    and not drop_ub else [])
            for arm, s in arms:
                own = r.get((w, arm))
                if own is None:
                    raise SystemExit(f"{w} {arm} has no horizon-100 cell: time it first")
                rate = own["s_per_cycle"] if case == "central" else max(own["per_cycle"])
                t = HORIZON * rate + rr
                if s == "posterior":
                    t += HORIZON * fmean(samplers) + replay
                jobs.append((arm, t))
                named.append((w, arm, t))
        for name, t in jobs:
            totals[name] += t
        local = sum(totals.values())
        pod = local * POD / 3600
        print(f"\n== {case}{' drop-ub' if drop_ub else ''}{' sample-8' if eight else ''}: local s by arm " + ", ".join(f"{k} {v:,.0f}" for k, v in
                                                       totals.items())
              + f"; total {local:,.0f} s = {local / 3600:.2f} local vCPU-h -> {pod:.2f} pod "
              f"vCPU-h (x {POD}); x {SAFETY}: {pod * SAFETY:.2f}")
        longest = max(t for _, t in jobs)
        top = sorted(((t, w, n) for w, n, t in named), reverse=True)[:3]
        print("   longest jobs: " + ", ".join(f"{w[-4:]} {n} {t * POD / 60:.1f} pod min"
                                           for t, w, n in top))
        print(f"   longest job {longest:,.0f} local s = {longest * POD / 60:.1f} pod min "
              f"(x {SAFETY}: {longest * POD * SAFETY / 60:.1f})")
        heavy = sum(1 for n, t in jobs if n in ("d2", "ub"))
        print(f"   jobs: {len(jobs)} ({heavy} d2/ub, {sum(1 for n, _ in jobs if n == 'd2')} d2)")
        for order in ("driver", "longest-first"):
            seq = [t for _, t in jobs] if order == "driver" else sorted(
                (t for _, t in jobs), reverse=True)
            for wkr in SHAPES:
                mk = schedule(seq, wkr) * POD
                billed = wkr * mk / 3600
                print(f"   {order:13} {wkr:2} vCPU: wall {mk / 60:5.1f} min, billed "
                      f"{billed:5.2f} vCPU-h; x {SAFETY}: wall {mk * SAFETY / 60:5.1f} min, "
                      f"billed {billed * SAFETY:5.2f} (stop {STOP} = {STOP * 60 / wkr:.0f} min)")
        results[case] = pod
    return 0


if __name__ == "__main__":
    sys.exit(main())
