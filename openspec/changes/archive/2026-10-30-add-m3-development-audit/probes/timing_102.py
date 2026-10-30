"""Task 10.2: PoC-B timing probe (sizing.md §EXP-0055).

Runs `python -m tokenomics.eval.deployed_cost --dry-run` once per grid cell (1 world seed,
1 repeat, `--workers 1`), each cell in its own process, and appends one jsonl row per timed
episode with the driver's `timings.csv` fields plus the process wall (interpreter start, torch
import and checkpoint load included). Run from `/srv/research/tokenomics/tokenomics`:

    ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/timing_102.py cells
    ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/timing_102.py h20
    ../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/timing_102.py seeds

`cells`: all 8 cells at horizon 12 (the probe). `h20`: P12-E2-k1 and P42-E14-k1 at horizon 20
(checks the 20 / 12 cycle scaling). `seeds`: P42-E14-k1 at horizon 12 on all 3 world seeds
(checks that seed 1000 prices the other two). Every command stays under 2 min locally.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PKG = REPO / "tokenomics"
PY = REPO / ".venv" / "bin" / "python"
OUT = HERE / "timing_102.jsonl"
PROD = REPO / "experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt"
ORDER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
CELLS = ["P12-E2-k1", "P24-E2-k1", "P42-E2-k1", "P42-E4-k1", "P42-E7-k1", "P42-E14-k1",
         "P42-E4-k2", "P42-E4-k4"]
NUMERIC = ("producers", "fanout", "consumer_multiplier", "consumers", "world_seed", "repeat",
           "horizon", "cycles", "dead", "units", "decisions", "prod_decisions",
           "order_decisions")


def run_cell(mode: str, cell: str, horizon: int, seeds: int) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "o"
        cmd = [str(PY), "-m", "tokenomics.eval.deployed_cost", "--dry-run", "--cells", cell,
               "--seeds", str(seeds), "--repeats", "1", "--horizon", str(horizon),
               "--workers", "1", "--prod-ckpt", str(PROD), "--orderer-ckpt", str(ORDER),
               "--out-dir", str(out)]
        load0 = os.getloadavg()
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=PKG, check=True, stdout=subprocess.DEVNULL)
        process_s = time.perf_counter() - t0
        load1 = os.getloadavg()
        host = json.loads((out / "spec.json").read_text())["host"]
        with (out / "timings.csv").open() as f:
            rows = list(csv.DictReader(f))
    recs = []
    for r in rows:
        rec: dict[str, object] = {"mode": mode, "cell": r["cell"]}
        for k, v in r.items():
            if k == "cell":
                continue
            rec[k] = int(v) if k in NUMERIC else float(v)
        rec.update(process_s=round(process_s, 3), episodes_in_process=len(rows),
                   loadavg_start=load0[0], loadavg_end=load1[0], cpu_model=host["cpu_model"],
                   vcpu=host["vcpu"])
        recs.append(rec)
    return recs


def main(mode: str) -> None:
    if mode == "cells":
        plan = [(c, 12, 1) for c in CELLS]
    elif mode == "h20":
        plan = [("P12-E2-k1", 20, 1), ("P42-E14-k1", 20, 1)]
    elif mode == "seeds":
        plan = [("P42-E14-k1", 12, 3)]
    else:
        raise SystemExit(f"unknown mode {mode!r}: cells | h20 | seeds")
    with OUT.open("a") as f:
        for cell, horizon, seeds in plan:
            for rec in run_cell(mode, cell, horizon, seeds):
                f.write(json.dumps(rec) + "\n")
                print(f"{mode} {rec['cell']} h{rec['horizon']} seed {rec['world_seed']}: "
                      f"wall {rec['wall_s']:.2f} s, setup {rec['setup_s']:.3f} s, "
                      f"{rec['decisions']} decisions, process {rec['process_s']:.2f} s",
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cells")
