"""Task 2.3 timing probe for EXP-0057 (add-m4-orderer-redesign D3 / D10).

Times single decoder-check jobs through the driver's own worker, `decoder_check.run_decoder_job`
(joint_stack_env, set_draw_key, routing tracer, ContractMonitor, ScoreFidelity on trained arms),
on jobs built exactly as `run_decoder_check` builds them (`world_entry`, `decoder_arm`,
`build_jobs`). One job per (world, arm) = the confirm's first episode seed, replicate 0. The
orderer checkpoint is loaded (and hash-checked) before the clock starts, as a pool worker loads
it once per process; the load is timed apart.

    python probes/timing_23.py --atlas A --orderer-ckpt O --worlds W1,W2 --arms TR,TS \
        --horizon 100 [--out probes/timing_23.jsonl]
    python probes/timing_23.py --atlas A --list        # the sample with its levers

Run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; torch is pinned to one
thread here. One command per (world set, arm set) keeps every command under 2 minutes.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from tokenomics.eval import decoder_check as d
from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
from tokenomics.eval.routing_audit import build_jobs, trained_policy


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def views_by_id(atlas: dict) -> dict[str, dict]:
    return {v["world"]: v for v in d.sample_views(atlas)}


def list_sample(atlas: dict) -> None:
    for v in d.sample_views(atlas):
        lv = v["record"]["levers"]
        print(json.dumps({"world": v["world"], "size": v["size"], "choice": lv["choice"],
                          "rho": round(v["rho"], 3), "load": round(lv["k_c"] * lv["m_q"], 2),
                          "pattern": lv["pattern"], "shocks": bool(lv["shocks"]),
                          "episodes": len(d.confirm_seeds(v)), "class": v["class"]}))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True)
    p.add_argument("--orderer-ckpt")
    p.add_argument("--worlds", default="")
    p.add_argument("--arms", default=",".join(d.ARMS))
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--out")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    t_import = time.perf_counter()
    atlas = d.load_atlas(Path(a.atlas))
    if a.list:
        list_sample(atlas)
        return 0
    import torch

    torch.set_num_threads(1)
    views = views_by_id(atlas)
    roster = load_roster(DEFAULT_ROSTER)
    pin = {"path": str(a.orderer_ckpt), "sha256": sha256_file(a.orderer_ckpt)}
    arms = [x for x in a.arms.split(",") if x]
    lines = []
    for w in [x for x in a.worlds.split(",") if x]:
        view = views[w]
        entry = d.world_entry(view)
        first = entry.episode_seeds[0]
        for arm in arms:
            table = [d.decoder_arm(arm, roster)]
            jobs = [j for j in build_jobs([entry], table, 1, a.horizon, None, {"orderer": pin})
                    if j["episode"] == first and j["replicate"] == 0]
            assert len(jobs) == 1, jobs
            job = jobs[0]
            job["arm"] = arm
            load_s = 0.0
            if job["checkpoints"]:
                t0 = time.perf_counter()
                trained_policy(job, "orderer")
                load_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            row = d.run_decoder_job(job)
            sec = time.perf_counter() - t0
            if "units" not in row:
                raise SystemExit(f"job failed: {row}")
            lv = view["record"]["levers"]
            line = {"world": w, "size": view["size"], "choice": lv["choice"],
                    "load": round(lv["k_c"] * lv["m_q"], 2), "arm": arm,
                    "horizon": a.horizon, "episode": first, "replicate": 0,
                    "seconds": round(sec, 3), "ckpt_load_s": round(load_s, 3),
                    "dead": row["dead"], "units": float(row["units"]),
                    "peak_rss_mb": round(peak_rss_mb(), 1)}
            lines.append(line)
            print(json.dumps(line), flush=True)
    print(json.dumps({"process_s": round(time.perf_counter() - t_import, 2),
                      "peak_rss_mb": round(peak_rss_mb(), 1)}), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
