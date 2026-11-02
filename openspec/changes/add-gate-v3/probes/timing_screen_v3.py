"""Screen and pilot timing probe (add-gate-v3 task 5.2, design D3 step 0 / waves 1-2).

Times the screen's own worker, `gate_cli.run_job`, on jobs built by `gate_cli.build_jobs` from
the Drafts registry v3 (`drafts/worlds_v3_draft.json`): one roster-member job per call, the
world's first registered episode, replicate 0 (a stochastic member's action seed is
`replicate_seed(world, episode, 0)`, as in the screen). Every line of the JSONL output records
the job, its wall seconds, the row's units / death, the load average and the free memory before
and after, and the process's peak RSS, so a reading taken while the shared box was busy can be
told apart.

Run from `/srv/research/tokenomics/tokenomics`, one command at a time:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 ../.venv/bin/python \\
        ../openspec/changes/add-gate-v3/probes/timing_screen_v3.py \\
        --world dev-ws_wide-42-calm-01 --members entity_planner/round_robin --horizon 100

`--guard-load` / `--guard-mem-mb` refuse to start a job while the 1-min load average is above
the bound or MemAvailable below it (the box is shared; design note in notes/5.2.md).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parent / "drafts" / "worlds_v3_draft.json"
OUT = HERE / "timing_screen_v3.jsonl"
DEFAULT_MEMBERS = ("entity_planner/round_robin", "entity_planner/proportional",
                   "entity_planner/makeable_weighted_whole", "entity_planner/makeable_split",
                   "noop/even_split", "random/random")


def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", required=True)
    ap.add_argument("--members", default=",".join(DEFAULT_MEMBERS),
                    help="comma-separated producing/ordering pairs, run in order")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--registry", default=str(DRAFT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="", help="free label recorded on every line (e.g. rerun)")
    ap.add_argument("--guard-load", type=float, default=6.0)
    ap.add_argument("--guard-mem-mb", type=int, default=400)
    args = ap.parse_args(argv)

    from tokenomics.eval.gate_cli import _worker_init, build_jobs, run_job
    from tokenomics.eval.registry import load_registry
    from tokenomics.eval.roster import Roster, RosterMember, contract_for, is_stochastic

    reg = load_registry(args.registry)
    e = reg.entry(args.world)
    episode = e.episode_seeds[args.episode_index]
    rec = (e.region or {}).get("record", {})
    lv = e.generator["params"].get("levers", {})
    _worker_init(args.horizon, None, None)
    for pair in args.members.split(","):
        prod, order = pair.split("/")
        roster = Roster(name="probe", replicates=5, members=(
            RosterMember(prod, order, contract_for(prod, order), is_stochastic(prod, order)),))
        job = next(j for j in build_jobs([e], roster, None)
                   if j["episode"] == episode and j["replicate"] == 0)
        l0, m0 = load1(), mem_available_mb()
        if l0 > args.guard_load or m0 < args.guard_mem_mb:
            print(f"[probe] refused {args.world} {pair}: load {l0} > {args.guard_load} or "
                  f"MemAvailable {m0} MB < {args.guard_mem_mb}", file=sys.stderr)
            return 3
        t0 = time.perf_counter()
        row = run_job(job)
        dt = time.perf_counter() - t0
        failed = "units" not in row
        line = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": args.tag,
            "world": e.id, "family": e.family, "size": e.size, "regime": e.regime,
            "choice": lv.get("choice", f"fanout_{e.generator['params'].get('fanout')}"),
            "rho": rec.get("coordinates", {}).get("rho"),
            "load_kc_mq": (lv["k_c"] * lv["m_q"]) if lv else None,
            "member": pair, "stochastic": is_stochastic(prod, order),
            "contract": job["contract"], "horizon": args.horizon, "episode": episode,
            "replicate": 0, "seconds": round(dt, 4),
            "units": None if failed else float(row["units"]),
            "dead": None if failed else row["dead"] == "true",
            "failure": row if failed else None,
            "load1_before": l0, "load1_after": load1(),
            "mem_available_mb_before": m0, "mem_available_mb_after": mem_available_mb(),
            "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        }
        with Path(args.out).open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
        print(f"[probe] {e.id} {pair} h{args.horizon}: {dt:.2f} s units {line['units']} "
              f"dead {line['dead']} load {l0}->{line['load1_after']} mem {m0}->"
              f"{line['mem_available_mb_after']} MB rss {line['maxrss_mb']} MB", flush=True)
        if failed:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
