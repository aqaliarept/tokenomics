"""Task 4.5: price EXP-0066 from `timing_45.jsonl` (sizing.md §EXP-0066).

Builds the real job list exactly as `split_need.run_split_need` does (sample, confirm seeds, arms
TS / MWWk at R = 2, `build_jobs`, `longest_first`: 284 jobs), prices each job and simulates the
pool with task 2.3's `extrapolate_23.simulate` (list scheduling in submission order). Two models:

- `cell` (the estimate): every job at its size x arm cell, the slower replicate measured on the
  loaded world (12-0352 / 42-0076, horizon 100, observer on);
- `world` (central check): task 2.3's world model (the world's measured RR seconds x the arm's
  largest arm / RR ratio, `timing_23.jsonl`) x the observer overhead measured here (this run's
  cell / EXP-0057's cell on the same world and arm, the larger one per size x arm).

Rates, setup and parent time as in task 2.3 (pod = 0.646 x local; billed = vCPU x (10 min setup +
wall); sensitivity at the 3 min observed setup).

    python probes/extrapolate_45.py --atlas A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extrapolate_23 import (  # noqa: E402  (task 2.3's pool model, reviewed)
    PARENT_S, POD, SETUP_MIN, SETUP_OBSERVED_MIN, load_lines, models, simulate)

from tokenomics.eval import split_need as sn  # noqa: E402
from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster  # noqa: E402
from tokenomics.eval.routing_audit import build_jobs  # noqa: E402

HERE = Path(__file__).resolve().parent
LOADED = {12: "ws-atlas-v1-12-0352", 42: "ws-atlas-v1-42-0076"}
SHAPES = (8, 16)
STOP_CAP = 8.0


def real_jobs(atlas_path: Path) -> list[dict]:
    views = sn.select_views(sn.load_atlas(atlas_path))
    roster = load_roster(DEFAULT_ROSTER)
    table = [sn.decoder_arm(a, roster) for a in sn.ARMS]
    jobs = build_jobs([sn.world_entry(v) for v in views], table, 1, sn.HORIZON, None,
                      {"orderer": {"path": "x", "sha256": "x"}})
    by_subject = {a["subject"]: a["arm"] for a in table}
    for j in jobs:
        j["arm"] = by_subject[j["subject"]]
    return sn.longest_first(jobs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True)
    a = p.parse_args()
    new = load_lines(HERE / "timing_45.jsonl")
    if any(x["problems"] for x in new):
        raise SystemExit("a timed row was not neutral / consistent")
    cell = {}
    for x in new:
        if x["world"] == LOADED[x["size"]]:
            k = (x["size"], x["arm"])
            cell[k] = max(cell.get(k, 0.0), x["seconds"])
    old_cell, world = models(load_lines(HERE / "timing_23.jsonl"))
    overhead = {k: v / old_cell[k] for k, v in cell.items()}
    jobs = real_jobs(Path(a.atlas))
    print(json.dumps({"jobs": len(jobs),
                      "by_size_arm": {f"{s}/{arm}": sum(1 for j in jobs if (j["size"], j["arm"])
                                                        == (s, arm))
                                      for s in (12, 42) for arm in sn.ARMS},
                      "cell_s": {f"{k[0]}/{k[1]}": v for k, v in sorted(cell.items())},
                      "observer_overhead": {f"{k[0]}/{k[1]}": round(v, 3)
                                            for k, v in sorted(overhead.items())}}))
    for model in ("cell", "world"):
        def price(j: dict) -> float:
            k = (j["size"], j["arm"])
            if model == "cell":
                return cell[k]
            return world["rr"][j["world"]] * world["ratio"][k] * max(1.0, overhead[k])

        local = [price(j) for j in jobs]
        pod = [POD * c for c in local]
        out = {"model": model, "local_s": round(sum(local)),
               "by_size_local_s": {s: round(sum(c for j, c in zip(jobs, local) if j["size"] == s))
                                   for s in (12, 42)},
               "local_vcpu_h": round(sum(local) / 3600, 2),
               "pod_vcpu_h": round(sum(pod) / 3600, 2), "longest_job_pod_s": round(max(pod), 1),
               "shapes": {}}
        for n in SHAPES:
            span, last = simulate(pod, n)
            wall = (span + PARENT_S) / 60
            billed = n * (SETUP_MIN + wall) / 60
            out["shapes"][n] = {
                "wall_min": round(wall, 1), "last_start_min": round(last / 60, 1),
                "busy": round(sum(pod) / (n * span), 2), "billed": round(billed, 2),
                "stop": round(min(2 * billed, STOP_CAP), 2),
                "stop_pod_life_min": round(min(2 * billed, STOP_CAP) * 60 / n, 1),
                "billed_setup3": round(n * (SETUP_OBSERVED_MIN + wall) / 60, 2)}
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
