"""Task 4.6 (b) timing probe for EXP-0051 / EXP-0052 (sizing.md §EXP-0051, §EXP-0052).

Times one episode per world of one arm, through the atlas's own job (`atlas.run_atlas_job`:
`routing_audit.run_job` with tracer and contract monitor on, plus the bound columns),
`--workers 1`, replicate 0, the screen's first episode seed. Worlds of one size x choice level:
`neutral` (every other lever neutral: the family world, calm, cyclic) and the screen sample's
worlds at the load (k_c x m_q) quantiles `q50`, `q75`, `q95` of that level (calibrate_46.py
--loads prints their indices), in a sample of `--worlds-in-sample` worlds (default
`atlas.SCREEN_WORLDS`). `timing_46.jsonl` was recorded with `--worlds-in-sample 1024` at 12 (the
sample before task 4.6's reduction step 1; its ids name worlds of that sample) and the default
128 at 42. The checkpoint load is timed apart. A world is not started when
the command would pass `--budget` seconds (it is printed as skipped), so every command stays
under 2 min; a world whose one episode exceeds 2 min is timed at `--horizon 50` and doubled in
sizing.md.

Run from /srv/research/tokenomics/tokenomics:
  ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/timing_46.py \
      --size 42 --choice sparse_f14 --arm round_robin --worlds neutral,q50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokenomics.eval.atlas import (ATLAS, CONFIRM, SCREEN_EPISODES, SCREEN_WORLDS, arm_table,
                                   run_atlas_job, space_entry)
from tokenomics.eval.contract import is_failure
from tokenomics.eval.provenance import sha256_file
from tokenomics.eval.roster import DEFAULT_ROSTER, load_roster
from tokenomics.eval.routing_audit import ROLES, build_jobs, trained_policy
from tokenomics.eval.world_space import (Levers, WorldPoint, episode_seeds, sample_worlds,
                                         world_seed)

ROOT = Path(__file__).resolve().parents[4]
CKPTS = {
    "producer": ROOT / "experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt",
    "orderer": ROOT / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt",
}
QUANTILES = {"q50": 0.5, "q75": 0.75, "q95": 0.95}


def points(size: int, choice: str, which: list[str], n: int) -> dict[str, WorldPoint]:
    """The named worlds: `neutral` and the `n`-world sample's load-quantile worlds of `choice`."""
    sample = [p for p in sample_worlds(ATLAS, size, n)
              if p.levers.choice == choice]
    ranked = sorted(sample, key=lambda p: (p.levers.k_c * p.levers.m_q, p.index))
    out = {}
    for name in which:
        if name == "neutral":
            out[name] = WorldPoint("neutral", size, 0, world_seed("neutral", size, 0),
                                   Levers(choice=choice))
        else:
            q = QUANTILES[name]
            out[name] = ranked[min(len(ranked) - 1, round(q * (len(ranked) - 1)))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--choice", required=True)
    ap.add_argument("--arm", required=True,
                    help="round_robin, proportional, trained_orderer or portable")
    ap.add_argument("--worlds", default="neutral,q50,q75,q95")
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--worlds-in-sample", type=int,
                    help="sample size (default atlas.SCREEN_WORLDS[size]; 1024 reproduces the "
                         "recorded 12-producer lines)")
    args = ap.parse_args()
    start = time.perf_counter()
    table = arm_table(load_roster(DEFAULT_ROSTER), CONFIRM, [args.arm])
    ckpts = {k: {"path": str(v), "sha256": sha256_file(str(v))} for k, v in CKPTS.items()}
    if table[0]["trained"]:
        t = time.perf_counter()
        for role in ROLES:
            trained_policy({"checkpoints": ckpts}, role)
        print(json.dumps({"load_seconds": round(time.perf_counter() - t, 3)}), flush=True)
    last = 0.0
    n = args.worlds_in_sample or SCREEN_WORLDS[args.size]
    for name, p in points(args.size, args.choice, args.worlds.split(","), n).items():
        if time.perf_counter() - start + 1.5 * last > args.budget:
            print(json.dumps({"world": name, "skipped": True}), flush=True)
            continue
        entry = space_entry(p, episode_seeds(p.world_id, SCREEN_EPISODES))
        job = build_jobs([entry], table, 1, args.horizon, None, ckpts)[0]
        t = time.perf_counter()
        row = run_atlas_job(job)
        last = time.perf_counter() - t
        if is_failure(row):
            raise SystemExit(f"contract failure: {row}")
        lv = p.levers
        print(json.dumps({
            "size": args.size, "choice": args.choice, "arm": args.arm, "world": name,
            "id": p.world_id, "horizon": args.horizon, "seconds": round(last, 3),
            "load": round(lv.k_c * lv.m_q, 3), "k_c": round(lv.k_c, 3), "m_q": round(lv.m_q, 3),
            "c": round(lv.capacity_scale, 3), "shocks": lv.shocks, "pattern": lv.pattern,
            "units": float(row["units"]), "dead": row["dead"]}), flush=True)


if __name__ == "__main__":
    main()
