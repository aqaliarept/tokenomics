"""Task 3.4 timing probe for EXP-0050 (sizing.md §EXP-0050).

Times one horizon-100 episode per arm on world 0 of one fanout cell (first episode seed),
through the driver's own job (`routing_audit.run_job`: tracer and contract monitor on),
`--workers 1`, replicate 0. The checkpoint load is timed apart.

Run from /srv/research/tokenomics/tokenomics:
  ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/timing_34.py \
      --cell sparse_wide,42,14,shocked --routers round_robin,two_choice_spare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.roster import ROSTER_V2, load_roster
from tokenomics.eval.routing_audit import (EXP0050, FANOUT_EPISODES, ROLES, build_jobs,
                                           fanout_world, is_failure, router_table, run_job,
                                           trained_policy)

ROOT = Path(__file__).resolve().parents[4]
CKPTS = {
    "producer": ROOT / "experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt",
    "orderer": ROOT / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="family,producers,fanout,regime")
    ap.add_argument("--routers", required=True)
    args = ap.parse_args()
    family, producers, fanout, regime = args.cell.split(",")
    world = fanout_world(family, int(producers), int(fanout), regime, 0, FANOUT_EPISODES[regime])
    table = router_table(load_roster(ROSTER_V2), args.routers.split(","), plan=EXP0050)
    ckpts = {k: {"path": str(v), "sha256": sha256_file(str(v))} for k, v in CKPTS.items()}
    jobs = [j for j in build_jobs([world], table, 1, 100, 1, ckpts) if j["replicate"] == 0]
    if any(j["checkpoints"] for j in jobs):
        start = time.perf_counter()
        for role in ROLES:
            trained_policy({"checkpoints": ckpts}, role)
        print(json.dumps({"load_seconds": round(time.perf_counter() - start, 3)}), flush=True)
    for job in jobs:
        start = time.perf_counter()
        row = run_job(job)
        seconds = time.perf_counter() - start
        if is_failure(row):
            raise SystemExit(f"contract failure: {row}")
        print(json.dumps({"world": job["world"], "subject": job["subject"],
                          "seconds": round(seconds, 3), "dead": row["dead"],
                          "units": float(row["units"])}), flush=True)


if __name__ == "__main__":
    main()
