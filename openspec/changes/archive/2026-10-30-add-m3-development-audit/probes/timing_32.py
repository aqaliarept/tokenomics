"""Task 3.2 timing probe for EXP-0049 (sizing.md §EXP-0049).

Times one horizon-100 episode per router of the EXP-0049 plan on one development world, through
the driver's own job (`routing_audit.run_job`: tracer on, contract monitor on), `--workers 1`.
Replicate 0 for stochastic routers. A reference router (`round_robin`) is timed too, to tie the
numbers to sizing.md §EXP-0048's measurements on the same worlds.

Run from /srv/research/tokenomics/tokenomics:
  ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/timing_32.py \
      --world dev-sparse_wide-12-shocked-00 --routers two_choice_spare,trained_orderer
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.registry import REGISTRY_V2, load_registry
from tokenomics.eval.roster import ROSTER_V2, load_roster
from tokenomics.eval.routing_audit import (EXP0049, ROLES, build_jobs, is_failure,
                                           router_table, run_job, select_worlds, trained_policy)

ROOT = Path(__file__).resolve().parents[4]
CKPTS = {
    "producer": ROOT / "experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt",
    "orderer": ROOT / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--routers", required=True)
    ap.add_argument("--horizon", type=int, default=100)
    args = ap.parse_args()
    registry = load_registry(REGISTRY_V2)
    world = [w for w in select_worlds(registry) if w.id == args.world]
    names = args.routers.split(",")
    table = router_table(load_roster(ROSTER_V2), names, plan=EXP0049)
    ckpts = {k: {"path": str(v), "sha256": sha256_file(str(v))} for k, v in CKPTS.items()}
    jobs = [j for j in build_jobs(world, table, 1, args.horizon, 1, ckpts) if j["replicate"] == 0]
    if any(j["checkpoints"] for j in jobs):  # one-off per worker process: load, timed apart
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
        print(json.dumps({"world": job["world"], "episode": job["episode"],
                          "subject": job["subject"], "seconds": round(seconds, 3),
                          "dead": row["dead"],
                          "units": float(row["units"])}), flush=True)


if __name__ == "__main__":
    main()
