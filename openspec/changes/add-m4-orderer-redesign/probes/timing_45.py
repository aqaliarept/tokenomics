"""Task 4.5 timing probe for EXP-0066 (`tokenomics.eval.split_need`; sizing.md §EXP-0066).

Times single split-need jobs through the driver's own worker, `split_need.run_split_job`
(EXP-0057's job path + the read-only observer), on jobs built exactly as `run_split_need` builds
them (`world_entry`, `decoder_arm`, `build_jobs`). The orderer checkpoint is loaded (and
hash-checked) before the clock starts, as a pool worker loads it once per process. Each timed row
is also checked against its EXP-0057 row (`neutrality`) and the observer self-checks
(`consistency`), so a timing never comes from a non-neutral run.

    python probes/timing_45.py --atlas A --orderer-ckpt O --exp0057-episodes E \
        --world W --arms TS,MWWk --replicates 0,1 [--episode-index 0] [--out probes/timing_45.jsonl]

Run with `OMP_NUM_THREADS=1`; torch is pinned to one thread here. One command per (world, arm
set, replicate set) keeps every command under 2 minutes.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from tokenomics.eval import split_need as sn
from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
from tokenomics.eval.routing_audit import build_jobs, trained_policy


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True)
    p.add_argument("--orderer-ckpt", required=True)
    p.add_argument("--exp0057-episodes", required=True)
    p.add_argument("--world", required=True)
    p.add_argument("--arms", default="TS,MWWk")
    p.add_argument("--replicates", default="0,1")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()
    t_start = time.perf_counter()
    import torch

    torch.set_num_threads(1)
    (view,) = sn.select_views(sn.load_atlas(Path(a.atlas)), [a.world])
    entry = sn.world_entry(view)
    episode = entry.episode_seeds[a.episode_index]
    roster = load_roster(DEFAULT_ROSTER)
    pin = {"path": str(a.orderer_ckpt), "sha256": sha256_file(a.orderer_ckpt)}
    ref = sn.read_reference(Path(a.exp0057_episodes))
    reps = [int(x) for x in a.replicates.split(",") if x]
    lines = []
    for arm in [x for x in a.arms.split(",") if x]:
        table = [sn.decoder_arm(arm, roster)]
        for rep in reps:
            (job,) = [j for j in build_jobs([entry], table, 1, sn.HORIZON, None, {"orderer": pin})
                      if j["episode"] == episode and j["replicate"] == rep]
            job["arm"] = arm
            if job["checkpoints"]:
                trained_policy(job, "orderer")
            t0 = time.perf_counter()
            row = sn.run_split_job(job)
            sec = time.perf_counter() - t0
            if "units" not in row:
                raise SystemExit(f"job failed: {row}")
            neutral, problems = sn.neutrality([row], ref)
            problems += sn.consistency([row])
            lv = view["record"]["levers"]
            line = {"world": a.world, "size": view["size"], "choice": lv["choice"], "arm": arm,
                    "horizon": sn.HORIZON, "episode": episode, "replicate": rep,
                    "seconds": round(sec, 3), "units": float(row["units"]),
                    "lost_no_fit_consumer": float(row["trace_lost_units_consumer_no_fit"]),
                    "joint_split_consumer": float(row["sn_joint_consumer_split_units"]),
                    "neutral_cells": neutral[0]["compared"] if neutral else 0,
                    "problems": [m for _, m in problems],
                    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                         / 1024, 1)}
            lines.append(line)
            print(json.dumps(line), flush=True)
    print(json.dumps({"process_s": round(time.perf_counter() - t_start, 2)}), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
