"""Local-to-pod factor for sizing.md §EXP-0048: time EXP-0047's job mix (draft roster v2, 48
members, random members R = 5 -> 100 runs) on one world-episode, locally, through the gate's own
job runner (`gate_cli.run_job`, ContractMonitor on). `--part i/n` runs the i-th of n slices of
the job list so each command stays under 2 min. Appends JSON lines to --out.

usage: python exp47_sample.py --world W --part 1/4 --out FILE
"""

from __future__ import annotations

import argparse
import json
import time

from tokenomics.eval import gate_cli
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.roster import build_roster_v2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    i, n = (int(x) for x in args.part.split("/"))
    entry = load_registry(REGISTRY_V2).entry(args.world)
    jobs = gate_cli.build_jobs([entry], build_roster_v2(), None, episodes_per_world=1)
    jobs = jobs[i - 1::n]
    gate_cli._worker_init(100, None, None)
    with open(args.out, "a") as fh:
        for job in jobs:
            t0 = time.perf_counter()
            row = gate_cli.run_job(job)
            dt = time.perf_counter() - t0
            fh.write(json.dumps({"world": args.world, "size": entry.size, "subject": job["subject"],
                                 "replicate": job["replicate"], "seconds": dt,
                                 "dead": row.get("dead"), "units": row.get("units"),
                                 "failure": "scope" in row}) + "\n")
    print(f"{args.world} part {args.part}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
