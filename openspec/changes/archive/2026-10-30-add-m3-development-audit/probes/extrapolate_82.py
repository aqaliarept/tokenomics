"""Task 8.2: extrapolate EXP-0054 (sizing.md §EXP-0054) from `timing_82.jsonl`.

The method is task 6.3's (`extrapolate_63.py`). Arm A is priced from its own horizon-100 cells,
world by world: search seconds per cycle = timed seconds x streams / timed streams. `central` is
the mean of the world's timed cycles and `high` its most expensive one. An A episode =
100 x that rate + 100 sampler calls + the played episode at the portable producer's speed with
the recorder on (the `sampler` mode's played time to cycle 90, scaled to 100 cycles). Local s x
0.646 = pod s. An unmeasured world is an error. T is rerun (notes/8.1.md: the teacher-path
fingerprint changed). It is priced from EXP-0053's recorded pod `search_seconds` (the whole `d2`
episode on the pod), not x 0.646. S and B are search-free: each is priced at the slowest measured
`plain` episode x 0.646. The `calibrated` case scales each world's A by that world's EXP-0053
ratio of pod-measured T to 6.3's composed-cost T prediction. It is a cross-check, never the
sizing basis. The `cycle-pick` case divides each world's central A by T's cycle-pick factor
(review concern 1): the mean search seconds of the probe's cycles over the whole-episode mean,
from T's own pod trajectory in EXP-0053 `teacher_decisions.csv`.

Wall per pod shape: the driver's job order (per world in sample order S, A, B, T; FIFO
`ProcessPoolExecutor`), billed vCPU-h = vCPU x (makespan + the pod overhead). The overhead is
EXP-0053's pod life minus its driver wall (22.9 - 21.8 min), rounded up to 2 min and never x 2.
Run: python3 extrapolate_82.py [timing_82.jsonl]   (stdlib only)
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path
from statistics import fmean

POD = 0.646
SAFETY = 2.0
HORIZON = 100
STOP = 4.0  # billed vCPU-h (D6)
SHAPES = (16, 8, 4, 2)
OVERHEAD_S = 120.0  # pod setup + pull + stop beyond the driver (EXP-0053: 1.1 min)
# D6 sample: the shocked worlds of EXP-0053's sample as run (teacher_audit.role_swap_sample)
SAMPLE = ("ws-atlas-v1-12-0060", "ws-atlas-v1-12-0111", "ws-atlas-v1-12-0344")
# EXP-0053 artifacts/episodes.csv, `d2` rows: pod search_seconds (the whole episode)
T_POD_S = {"ws-atlas-v1-12-0060": 937.4098723810166, "ws-atlas-v1-12-0111": 487.3199076850433,
           "ws-atlas-v1-12-0344": 1117.377086260938}
# sizing.md §EXP-0053: 6.3's composed-cost `d2` prediction (pod min per episode, central)
T_PRED_POD_MIN = {"ws-atlas-v1-12-0060": 25.9, "ws-atlas-v1-12-0111": 9.0,
                  "ws-atlas-v1-12-0344": 28.6}
# Review concern 1 (round 1): T's own pod trajectory in EXP-0053 `teacher_decisions.csv`
# (`d2`): mean search s of the probe's cycles over the episode's mean per-cycle search s. The
# probe's cycles miss 0060's / 0344's expensive first 30 cycles and hit 0111's dearer first half.
CYCLE_PICK = {"ws-atlas-v1-12-0060": 0.84, "ws-atlas-v1-12-0111": 1.33,
              "ws-atlas-v1-12-0344": 0.88}
ROLES = ("S", "A", "B", "T")


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


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
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
        "timing_82.jsonl")
    rows = load(path)
    cells: dict[str, list[dict]] = {}
    for r in rows:
        if (r.get("mode") == "cell" and r["role"] == "A" and r["horizon"] == HORIZON
                and r["timed_streams"]):
            cells.setdefault(r["world"], []).append(r)
    sampler = fmean(r["seconds"] for r in rows if r.get("mode") == "sampler")
    played90 = fmean(r["played_s_with_recorder"] for r in rows
                     if r.get("mode") == "sampler-summary")
    plain = max(r["seconds"] for r in rows if r.get("mode") == "plain")
    rest = HORIZON * sampler + played90 * HORIZON / 90
    print("arm A horizon-100 cells (local s):")
    for w in SAMPLE:
        cs = cells.get(w)
        if not cs:
            raise SystemExit(f"{w} A has no horizon-100 cell: time it first")
        per = [c["timed_seconds"] * c["streams"] / c["timed_streams"] for c in cs]
        print(f"  {w[-4:]} cycles {[c['cycle'] for c in cs]}: turns {cs[0]['orderer_turns']}, "
              f"streams {[c['streams'] for c in cs]}, cand {cs[0]['candidates_per_stream']:.0f}, "
              f"s/stream {', '.join(f'{c['s_per_stream']:.2f}' for c in cs)}, s/cycle "
              f"{', '.join(f'{p:.1f}' for p in per)} (mean {fmean(per):.1f})")
    print(f"sampler {sampler * 1000:.0f} ms/call; played episode with recorder (portable "
          f"producer) {played90:.2f} s to cycle 90; A rest {rest:.1f} s; S/B episode {plain:.2f}"
          f" s local")
    for case in ("central", "high", "calibrated", "cycle-pick"):
        jobs: list[tuple[str, str, float]] = []  # (world, role, pod s)
        for w in SAMPLE:
            per = [c["timed_seconds"] * c["streams"] / c["timed_streams"] for c in cells[w]]
            rate = max(per) if case == "high" else fmean(per)
            a = (HORIZON * rate + rest) * POD
            if case == "cycle-pick":
                a /= CYCLE_PICK[w]
            if case == "calibrated":
                a *= T_POD_S[w] / (T_PRED_POD_MIN[w] * 60)
            for role in ROLES:
                t = {"S": plain * POD, "B": plain * POD, "A": a, "T": T_POD_S[w]}[role]
                jobs.append((w, role, t))
        tot = {r: sum(t for _, x, t in jobs if x == r) for r in ROLES}
        pod = sum(tot.values()) / 3600
        print(f"\n== {case}: pod s by role " + ", ".join(f"{r} {v:,.0f}" for r, v in tot.items())
              + f"; total {pod:.2f} pod vCPU-h; x {SAFETY}: {pod * SAFETY:.2f}")
        print("   jobs (pod min): " + ", ".join(f"{w[-4:]} {r} {t / 60:.1f}" for w, r, t in jobs
                                              if r in ("A", "T")))
        longest = max(jobs, key=lambda j: j[2])
        print(f"   longest job {longest[0][-4:]} {longest[1]} {longest[2] / 60:.1f} pod min "
              f"(x {SAFETY}: {longest[2] * SAFETY / 60:.1f})")
        for order in ("driver", "longest-first"):
            seq = [t for _, _, t in jobs]
            if order == "longest-first":
                seq = sorted(seq, reverse=True)
            for wkr in SHAPES:
                mk = schedule(seq, wkr)
                b1 = wkr * (mk + OVERHEAD_S) / 3600
                b2 = wkr * (mk * SAFETY + OVERHEAD_S) / 3600
                print(f"   {order:13} {wkr:2} vCPU: wall {mk / 60:5.1f} min, billed {b1:5.2f}; "
                      f"x {SAFETY}: wall {mk * SAFETY / 60:5.1f} min, billed {b2:5.2f} "
                      f"(stop {STOP} = {STOP * 60 / wkr:.0f} min incl. overhead)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
