"""EXP-0121: capacity / fit-budget probe for the v14 turn heads (supervisor 2026-11-07, after EXP-0120's underfit;
pre-registration note `notes/exp0121-prereg.md`). No rollouts: EXP-0120's persisted held-out records (F40 trunk
embeddings, default labels) are split in two contiguous halves (the records are in job order, so at most one
world-episode straddles); each grid setting fits on the first half and is read on the second.

Grid: head width {32, 64} × set layers {2, 3} × epochs {30, 90}, cosine learning-rate schedule from 3e-3, batch
512, ≤ 400 pairs per turn, seed 0 (everything else v14). Winner: the first setting in (width, layers, epochs)
order whose probe-held-out accept agreement and accept-turn rank both reach 0.99 (none → reported, no v15).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from . import m6_imitation as im
from . import m6_turn_imitation as ti

GRID = [{"width": w, "set_layers": layer, "epochs": e} for w in (32, 64) for layer in (2, 3) for e in (30, 90)]
LR, BATCH, CAP, SEED, GATE = 3e-3, 512, 400, 0, 0.99


def _split(records: str, limit: int | None) -> tuple[list[dict], list[dict]]:
    recs = ti.load_records(Path(records), limit)
    h = len(recs) // 2
    return recs[:h], recs[h:]


def probe_one(job: dict) -> dict[str, Any]:
    torch.set_num_threads(1)
    train, test = _split(job["records"], job.get("limit"))
    pm, _ = im.load_models(job["models"])
    t0 = time.perf_counter()
    rep = ti.fit(pm, train, job["epochs"], LR, BATCH, CAP, SEED, log=lambda *a: None,
                 heads_cfg={"set_layers": job["set_layers"], "bounded": True, "width": job["width"]},
                 schedule="cosine")
    fit_s = time.perf_counter() - t0
    heads = pm.turn
    assert heads is not None
    return {**{k: job[k] for k in ("width", "set_layers", "epochs")},
            "params": sum(p.numel() for p in heads.parameters()),
            "seconds_per_epoch": fit_s / job["epochs"], "train_turns": len(train), "test_turns": len(test),
            "seconds_per_1k_turn_epoch": 1000 * fit_s / job["epochs"] / max(1, len(train)),
            "loss": [h["loss"] for h in rep["history"]],
            "heldout": ti.agreement(pm, test), "train": ti.agreement(pm, train, whole=False)}


def winner(results: Sequence[dict]) -> dict | None:
    for r in sorted(results, key=lambda r: (r["width"], r["set_layers"], r["epochs"])):
        h = r["heldout"]
        if (h["accept"] or 0) >= GATE and (h["rank_accept"] or 0) >= GATE:
            return r
    return None


def main(argv: Sequence[str] | None = None) -> int:
    from ..eval.provenance import sha256_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="EXP-0120 heldout_records.npz")
    ap.add_argument("--models", required=True, help="the F40 candidate (heads only need its hidden size)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true", help="400 turns, 2 settings at 2 epochs")
    a = ap.parse_args(argv)
    grid = ([{"width": 32, "set_layers": 2, "epochs": 2}, {"width": 64, "set_layers": 3, "epochs": 2}]
            if a.dry_run else GRID)
    jobs = [{**g, "records": a.records, "models": a.models, "limit": 400 if a.dry_run else None}
            for g in sorted(grid, key=lambda g: -g["width"] * g["set_layers"] * g["epochs"])]
    t0 = time.perf_counter()
    if a.workers <= 1:
        results = [probe_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=get_context("spawn")) as ex:
            results = list(ex.map(probe_one, jobs))
    w = winner(results)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"winner": None if w is None else {k: w[k] for k in ("width", "set_layers", "epochs")},
           "winner_seconds_per_epoch": None if w is None else w["seconds_per_epoch"],
           "results": results, "records_sha256": sha256_file(a.records), "models_sha256": sha256_file(a.models),
           "source_commit": a.source_commit, "dry_run": a.dry_run, "seconds": round(time.perf_counter() - t0, 1),
           "lr": LR, "schedule": "cosine", "batch": BATCH, "cap": CAP, "seed": SEED, "gate": GATE}
    (out / "probe.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    for r in sorted(results, key=lambda r: (r["width"], r["set_layers"], r["epochs"])):
        h = r["heldout"]
        print(f"[turn_probe] w{r['width']} L{r['set_layers']} e{r['epochs']}: accept {h['accept']:.4f} "
              f"rank_accept {h['rank_accept']:.4f} rank_fund {h['rank_fund']} displaced "
              f"{h['accept_errors']['declined_displaced']['rate']} ({r['seconds_per_epoch']:.1f} s/epoch)",
              flush=True)
    print(f"[turn_probe] winner {doc['winner']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
