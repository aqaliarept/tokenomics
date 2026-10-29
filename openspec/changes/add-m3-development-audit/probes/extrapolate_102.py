"""Task 10.2: extrapolate the PoC-B probe (`timing_102.jsonl`) to the full D7 grid.

Stdlib only. Run from anywhere: `python3 extrapolate_102.py`. Prints the per-cell table, the
compute totals and a FIFO schedule of the 72 jobs, in the driver's repeat-major order, on each
candidate pod shape (workers = vCPU - 2, D7). sizing.md §EXP-0055 quotes its output.
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL_TO_POD = 0.646  # EXP-0047's measured local -> pod factor (D8)
SAFETY = 2.0
HORIZON, PROBE_HORIZON = 20, 12
SEEDS, REPEATS = 3, 3
OVERHEAD_MIN = 3.0  # pod life outside the driver: EXP-0053 1.1 min, EXP-0054 2.9 min, rounded up
STOP_VCPU_H = 1.0   # D7's hard stop
SHAPES = (4, 8, 16)
CELLS = ["P12-E2-k1", "P24-E2-k1", "P42-E2-k1", "P42-E4-k1", "P42-E7-k1", "P42-E14-k1",
         "P42-E4-k2", "P42-E4-k4"]


def load() -> list[dict[str, object]]:
    with (HERE / "timing_102.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def schedule(durations: list[float], workers: int, init_s: float) -> float:
    """Wall of FIFO jobs on `workers` processes that each pay `init_s` before their first job."""
    free = [init_s] * workers
    heapq.heapify(free)
    end = 0.0
    for d in durations:
        t = heapq.heappop(free) + d
        end = max(end, t)
        heapq.heappush(free, t)
    return end


def main() -> None:
    recs = load()
    probe = {r["cell"]: r for r in recs if r["mode"] == "cells"}
    h20 = [r for r in recs if r["mode"] == "h20"]
    seeds = [r for r in recs if r["mode"] == "seeds"]
    # horizon scaling: the larger of 20 / 12 and every measured h20 / h12 ratio of the same cell
    ratios = {r["cell"]: r["wall_s"] / probe[r["cell"]]["wall_s"] for r in h20}
    hscale = max([HORIZON / PROBE_HORIZON, *ratios.values()])
    print(f"h20 / h12 ratios: {', '.join(f'{c} {v:.3f}' for c, v in ratios.items())}; "
          f"20 / 12 = {HORIZON / PROBE_HORIZON:.3f}; used {hscale:.3f}")
    s0 = [r["wall_s"] for r in seeds]
    print(f"seed check P42-E14-k1 h12 (1000 / 1001 / 1002): "
          f"{' / '.join(f'{w:.2f}' for w in s0)} s; max / seed-1000 {max(s0) / s0[0]:.3f}")
    init_local = max(r["process_s"] - r["wall_s"] * r["episodes_in_process"] for r in recs)
    print(f"process overhead (start + imports + checkpoint load + outputs), local max: "
          f"{init_local:.2f} s")

    print("\n| cell | decisions h12 | wall h12 local s | episode h20 local s | episode h20 pod s "
          "| cell (9 episodes) pod s | × 2 |")
    print("|---|---|---|---|---|---|---|")
    per_episode_pod: dict[str, float] = {}
    for c in CELLS:
        r = probe[c]
        local20 = r["wall_s"] * hscale + r["setup_s"]
        pod = local20 * LOCAL_TO_POD
        per_episode_pod[c] = pod
        print(f"| {c} | {r['decisions']:,} | {r['wall_s']:.2f} | {local20:.2f} | {pod:.2f} | "
              f"{pod * SEEDS * REPEATS:.1f} | {pod * SEEDS * REPEATS * SAFETY:.1f} |")
    sweep_local = sum(probe[c]["wall_s"] for c in CELLS)
    compute = sum(per_episode_pod.values()) * SEEDS * REPEATS
    print(f"\nprobe episodes (8 cells, h12, local): {sweep_local:.2f} s")
    print(f"full grid compute: {compute:.0f} pod s = {compute / 3600:.4f} vCPU-h "
          f"(× 2: {compute * SAFETY:.0f} s = {compute * SAFETY / 3600:.4f})")

    jobs = [per_episode_pod[c] for _ in range(REPEATS) for c in CELLS for _ in range(SEEDS)]
    init_pod = init_local * LOCAL_TO_POD
    print(f"\n{len(jobs)} jobs; longest job {max(jobs):.1f} pod s (× 2 {max(jobs) * SAFETY:.1f}); "
          f"worker init {init_pod:.1f} pod s (× 2 {init_pod * SAFETY:.1f})")
    print("\n| shape | workers | jobs per worker | driver wall central / × 2 | pod life × 2 "
          "(+ overhead) | billed central / × 2 vCPU-h | stop 1 vCPU-h as pod life | margin |")
    print("|---|---|---|---|---|---|---|---|")
    for v in SHAPES:
        w = max(1, v - 2)
        # parent: imports + checkpoint hashing ≈ one worker init, before the pool starts
        wall = init_pod + schedule(jobs, w, init_pod)
        wall2 = SAFETY * (init_pod + schedule(jobs, w, init_pod))
        life, life2 = OVERHEAD_MIN + wall / 60, OVERHEAD_MIN + wall2 / 60
        stop_min = STOP_VCPU_H * 60 / v
        print(f"| {v} vCPU | {w} | {len(jobs) / w:.1f} | {wall / 60:.2f} / {wall2 / 60:.2f} min | "
              f"{life2:.2f} min | {v * life / 60:.3f} / {v * life2 / 60:.3f} | {stop_min:.2f} min "
              f"| {stop_min / life2:.2f}× |")


if __name__ == "__main__":
    main()
