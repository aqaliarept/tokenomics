"""Task 2.3: price EXP-0057 from `timing_23.jsonl` (sizing.md §EXP-0057).

Builds the real job list exactly as `decoder_check.run_decoder_check` does (sample, confirm seeds,
arm table, `build_jobs`, `longest_first`: 639 jobs), prices each job, and simulates the driver's
pool: jobs submitted in `longest_first` order to N workers, each idle worker taking the next job
(greedy list scheduling, as a `ProcessPoolExecutor` drains its queue). Two job-price models:

- `cell`: every job at its size x arm cell measured on the loaded world (12-0352 / 42-0076,
  horizon 100) - the supervisor's model, conservative (12-0352 has the slowest RR of the 23
  12-producer worlds; 42-0076 is a loaded f14 world, the costliest choice level);
- `world`: the job's world's measured RR seconds (horizon 100, first confirm episode) x the arm's
  largest measured arm / RR ratio at its size - the central check.

Pod seconds = 0.646 x local (m3 sizing.md §EXP-0051). Billed vCPU-h = vCPU x (setup + wall),
setup = 10 min (design D10), with a sensitivity at the 3 min observed on EXP-0052..0056.

    python probes/extrapolate_23.py --atlas A [--jsonl probes/timing_23.jsonl]
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict
from pathlib import Path

from tokenomics.eval import decoder_check as d
from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
from tokenomics.eval.routing_audit import build_jobs

POD = 0.646
SETUP_MIN = 10.0
SETUP_OBSERVED_MIN = 3.0
WORKER_START_S = 3.5   # local: spawn + imports + checkpoint load per worker (probe cmd overhead)
PARENT_S = 60.0        # pod: parent atlas / pin checks before, readings + reproduction after
LOADED = {12: "ws-atlas-v1-12-0352", 42: "ws-atlas-v1-42-0076"}
SHAPES = (8, 16, 32)


def load_lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def real_jobs(atlas_path: Path, mww_r: int = 2) -> list[dict]:
    atlas = d.load_atlas(atlas_path)
    views = d.select_views(atlas)
    roster = load_roster(DEFAULT_ROSTER)
    entries = [d.world_entry(v) for v in views]
    table = [d.decoder_arm(a, roster) for a in d.ARMS]
    pin = {"orderer": {"path": "x", "sha256": "x"}}
    jobs = build_jobs(entries, table, 1, d.HORIZON, None, pin)
    by_subject = {a["subject"]: a["arm"] for a in table}
    for j in jobs:
        j["arm"] = by_subject[j["subject"]]
    if mww_r == 1:
        jobs = [j for j in jobs if not (j["arm"] == d.MWW and j["replicate"] > 0)]
    return d.longest_first(jobs)


def models(lines: list[dict]) -> tuple[dict, dict]:
    h100 = [x for x in lines if x["horizon"] == 100]
    cell = {(x["size"], x["arm"]): x["seconds"] for x in h100 if x["world"] == LOADED[x["size"]]}
    rr = {x["world"]: x["seconds"] for x in h100 if x["arm"] == d.RR}
    ratio: dict[tuple[int, str], float] = defaultdict(float)
    for x in h100:
        if x["world"] in rr:
            k = (x["size"], x["arm"])
            ratio[k] = max(ratio[k], x["seconds"] / rr[x["world"]])
    world = {"rr": rr, "ratio": dict(ratio)}
    return cell, world


def price(job: dict, model: str, cell: dict, world: dict) -> float:
    if model == "cell":
        return cell[(job["size"], job["arm"])]
    return world["rr"][job["world"]] * world["ratio"][(job["size"], job["arm"])]


def simulate(costs: list[float], workers: int) -> tuple[float, float]:
    """(makespan, last job start) in seconds for list scheduling in submission order."""
    free = [WORKER_START_S * POD] * workers
    heapq.heapify(free)
    last_start = 0.0
    end = 0.0
    for c in costs:
        t = heapq.heappop(free)
        last_start = max(last_start, t)
        end = max(end, t + c)
        heapq.heappush(free, t + c)
    return end, last_start


def report(jobs: list[dict], model: str, cell: dict, world: dict, label: str) -> dict:
    local = [price(j, model, cell, world) for j in jobs]
    pod = [POD * c for c in local]
    out = {"label": label, "model": model, "jobs": len(jobs), "local_s": round(sum(local)),
           "local_vcpu_h": round(sum(local) / 3600, 2), "pod_vcpu_h": round(sum(pod) / 3600, 2),
           "by_size_local_s": {s: round(sum(c for j, c in zip(jobs, local) if j["size"] == s))
                               for s in (12, 42)},
           "longest_job_pod_s": round(max(pod), 1), "shapes": {}}
    for n in SHAPES:
        span, last = simulate(pod, n)
        wall_min = (span + PARENT_S) / 60
        out["shapes"][n] = {
            "wall_min": round(wall_min, 1), "last_start_min": round(last / 60, 1),
            "busy": round(sum(pod) / (n * span), 2),
            "billed": round(n * (SETUP_MIN + wall_min) / 60, 2),
            "billed_setup3": round(n * (SETUP_OBSERVED_MIN + wall_min) / 60, 2)}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True)
    p.add_argument("--jsonl", default=str(Path(__file__).with_name("timing_23.jsonl")))
    a = p.parse_args()
    lines = load_lines(Path(a.jsonl))
    cell, world = models(lines)
    print(json.dumps({"cell_s": {f"{k[0]}/{k[1]}": v for k, v in sorted(cell.items())},
                      "ratio_vs_rr": {f"{k[0]}/{k[1]}": round(v, 2)
                                      for k, v in sorted(world["ratio"].items())}}))
    for mww_r in (2, 1):
        jobs = real_jobs(Path(a.atlas), mww_r)
        for model in ("cell", "world"):
            print(json.dumps(report(jobs, model, cell, world, f"MWW R={mww_r}")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
