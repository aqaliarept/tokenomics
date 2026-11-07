"""EXP-0121 / EXP-0122: capacity / fit-budget probe for the v14 turn heads (supervisor 2026-11-07, after EXP-0120's underfit;
pre-registration note `notes/exp0121-prereg.md`). No rollouts: EXP-0120's persisted held-out records (F40 trunk
embeddings, default labels) are split in two contiguous halves (the records are in job order, so at most one
world-episode straddles); each grid setting fits on the first half and is read on the second.

EXP-0122 (`--recollect`, v15 option (a)): the same 32 held-out jobs are replayed (deterministic keyed episodes)
with the linear own-state fractions appended; every existing column must equal EXP-0120's records, so the halves
are the same turns. The same grid and winner rule, with the boundary-slack error split (DIAGNOSTICS 38).

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import m6_imitation as im
from . import m6_turn_imitation as ti

GRID = [{"width": w, "set_layers": layer, "epochs": e} for w in (32, 64) for layer in (2, 3) for e in (30, 90)]
# EXP-0125: the two best EXP-0124 sizes, each with and without the arrival-position feature
GRID_0125 = [{"width": w, "set_layers": layer, "epochs": 90, "arrival": arr}
             for w, layer in ((32, 3), (64, 2)) for arr in (False, True)]
LR, BATCH, CAP, SEED, GATE = 3e-3, 512, 400, 0, 0.99


def _split(records: str, limit: int | None, strip_arrival: bool = False
           ) -> tuple[list[dict], list[dict]]:
    recs = ti.load_records(Path(records), limit)
    if strip_arrival:  # the records carry the arrival column last; the arms without the feature drop it
        for r in recs:
            if r["kind"] == "accept":
                r["bid"], r["claim"] = r["bid"][:, :-1], r["claim"][:, :-1]
    h = len(recs) // 2
    return recs[:h], recs[h:]


SLACK_BINS = ("<-2", "[-2,-0.5)", "[-0.5,0)", "[0,0.5)", "[0.5,2)", ">=2")


def slack_split(pm: Any, recs: list[dict]) -> dict[str, dict[str, Any]]:
    """Accept-bit errors of bids that fit alone, by slack = capacity left after the higher-priced bids
    minus own capacity, in units of own capacity (the greedy auction replayed on the log1p-inverted bid
    rows: price col 0, capacity col 2, the new-work budget col 7; one capacity type per producer)."""
    bins = {b: [0, 0] for b in SLACK_BINS}
    acc = [r for r in recs if r["kind"] == "accept" and len(r["bid"])]
    with torch.no_grad():
        for k in range(0, len(acc), 2000):
            part = acc[k:k + 2000]
            tg = ti.turn_targets(part, None, None)
            _, out = ti._losses(pm.turn, tg)
            pred = (out["acc"] > 0).numpy()
            off = 0
            for r in part:
                x, n = r["bid"], len(r["bid"])
                price, cap, budget = np.expm1(x[:, 0]), np.rint(np.expm1(x[:, 2])), np.rint(np.expm1(x[:, 7]))
                rem, sl = budget[0], {}
                for i in np.argsort(-price, kind="stable"):
                    sl[i] = rem - cap[i]
                    if cap[i] <= rem + 1e-6:
                        rem -= cap[i]
                for i in range(n):
                    if x[i, ti.FITS_COL] > 0:
                        v = sl[i] / max(cap[i], 1)
                        b = ("<-2" if v < -2 else "[-2,-0.5)" if v < -0.5 else "[-0.5,0)" if v < 0 else
                             "[0,0.5)" if v < 0.5 else "[0.5,2)" if v < 2 else ">=2")
                        bins[b][0] += int(pred[off + i] != bool(r["acc"][i]))
                        bins[b][1] += 1
                off += n
    return {b: {"wrong": w, "n": n, "rate": w / n if n else None} for b, (w, n) in bins.items()}


def tie_dependent(r: dict) -> np.ndarray:
    """Bids whose default accept bit flips when equal-priced bids are taken in reverse order: the greedy
    auction replayed from the log1p-inverted rows (price col 0, capacity col 2, the new-work budget col 7; one
    capacity type per producer). These labels depend on the arrival order, which no v14 / v15 feature carries."""
    x = r["bid"]
    n = len(x)
    if n == 0:
        return np.zeros(0, bool)
    price = np.round(np.expm1(x[:, 0]), 6)
    cap, budget = np.rint(np.expm1(x[:, 2])), float(np.rint(np.expm1(x[0, 7])))

    def auction(tiebreak: np.ndarray) -> np.ndarray:
        rem, acc = budget, np.zeros(n, bool)
        for i in np.lexsort((tiebreak, -price)):
            if cap[i] <= rem + 1e-6:
                acc[i], rem = True, rem - cap[i]
        return acc

    return auction(np.arange(n)) != auction(-np.arange(n))


def tie_split(pm: Any, recs: list[dict]) -> dict[str, dict[str, Any]]:
    """Accept agreement on tie-dependent bits, determinable bits and all bits (EXP-0124 / 0125)."""
    ok = {"tie_dependent": [0, 0], "determinable": [0, 0]}
    acc = [r for r in recs if r["kind"] == "accept"]
    with torch.no_grad():
        for k in range(0, len(acc), 2000):
            part = acc[k:k + 2000]
            tg = ti.turn_targets(part, None, None)
            _, out = ti._losses(pm.turn, tg)
            pred, lab = (out["acc"] > 0).numpy(), tg["bits"].numpy() > 0
            amb = np.concatenate([tie_dependent(r) for r in part]) if part else np.zeros(0, bool)
            for name, m in (("tie_dependent", amb), ("determinable", ~amb)):
                ok[name][0] += int((pred[m] == lab[m]).sum())
                ok[name][1] += int(m.sum())
    out = {k: {"n": n, "agreement": c / n if n else None} for k, (c, n) in ok.items()}
    n = ok["tie_dependent"][1] + ok["determinable"][1]
    out["overall"] = {"n": n, "agreement": (ok["tie_dependent"][0] + ok["determinable"][0]) / n if n else None}
    return out


def recollect(training_dir: str, models: str, replenish: str, reference: str, out: Path, workers: int,
              dry_run: bool, arrival: bool = False) -> tuple[Path, dict[str, Any]]:
    """Replay EXP-0120's held-out jobs with the linear fractions; check every existing column equals the
    reference records (same turns, same order); write the new records."""
    ts, _train, mon = im._training(training_dir)
    jobs = [{**j, "replenish": replenish, "linear": True, "arrival": arrival}
            for j in ti._jobs(ts, mon[:1] if dry_run else mon, "turn_monitor", models, 6 if dry_run else 100)]
    played = im._pool(ti.record_job, jobs, workers)
    recs = [r for x in played for r in x["records"]]
    # a dry run (horizon 6) can only match the reference's first job prefix: compare that job's turns
    n_cmp = len(played[0]["records"]) if dry_run else None
    ref = ti.load_records(Path(reference), n_cmp)
    base = {"fund": ti.g.FUND_DIM, "bid": ti.g.BID_DIM, "claim": ti.g.CLAIM_DIM}
    problems = [] if dry_run or len(ref) == len(recs) else [f"{len(recs)} turns vs {len(ref)} recorded"]
    emb_diff = 0.0
    for k, (a, b) in enumerate(zip(recs[:n_cmp] if dry_run else recs, ref)):
        if a["kind"] != b["kind"]:
            problems.append(f"turn {k}: kind differs")
            break
        for f, v in a.items():
            if f in base:
                same = np.array_equal(v[:, :base[f]], b[f])
            elif f in ("h_cap", "h_in", "pooled"):
                emb_diff = max(emb_diff, float(np.abs(v - b[f]).max()) if v.size else 0.0)
                same = True
            elif isinstance(v, np.ndarray):
                same = np.array_equal(v, b[f])
            else:
                same = v == b[f]
            if not same:
                problems.append(f"turn {k}: {f} differs")
        if len(problems) > 5:
            break
    if emb_diff > 1e-5:
        problems.append(f"trunk embeddings differ by {emb_diff:.2e}")
    path = out / ("heldout_records_linear_arrival.npz" if arrival else "heldout_records_linear.npz")
    ti.save_records(path, recs)
    return path, {"turns": len(recs), "reference_turns": len(ref), "max_embedding_diff": emb_diff,
                  "problems": problems, "match": not problems}


def probe_one(job: dict) -> dict[str, Any]:
    torch.set_num_threads(1)
    train, test = _split(job["records"], job.get("limit"), job.get("strip_arrival", False))
    pm, _ = im.load_models(job["models"])
    t0 = time.perf_counter()
    rep = ti.fit(pm, train, job["epochs"], LR, BATCH, CAP, SEED, log=lambda *a: None,
                 heads_cfg={"set_layers": job["set_layers"], "bounded": True, "width": job["width"],
                            "linear": job.get("linear", False), "ordered": job.get("ordered", False),
                            "arrival": job.get("arrival", False)},
                 schedule="cosine")
    fit_s = time.perf_counter() - t0
    heads = pm.turn
    assert heads is not None
    return {**{k: job[k] for k in ("width", "set_layers", "epochs")}, "linear": job.get("linear", False),
            "ordered": job.get("ordered", False), "arrival": job.get("arrival", False),
            "tie_split": tie_split(pm, test),
            "slack": slack_split(pm, test),
            "params": sum(p.numel() for p in heads.parameters()),
            "seconds_per_epoch": fit_s / job["epochs"], "train_turns": len(train), "test_turns": len(test),
            "seconds_per_1k_turn_epoch": 1000 * fit_s / job["epochs"] / max(1, len(train)),
            "loss": [h["loss"] for h in rep["history"]],
            "heldout": ti.agreement(pm, test), "train": ti.agreement(pm, train, whole=False)}


def winner(results: Sequence[dict], need_arrival: bool = False) -> dict | None:
    for r in sorted(results, key=lambda r: (r["width"], r["set_layers"], r["epochs"])):
        if need_arrival and not r.get("arrival"):
            continue
        h = r["heldout"]
        if (h["accept"] or 0) >= GATE and (h["rank_accept"] or 0) >= GATE:
            return r
    return None


def main(argv: Sequence[str] | None = None) -> int:
    from ..eval.provenance import sha256_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="EXP-0120 heldout_records.npz")
    ap.add_argument("--recollect", action="store_true", help="EXP-0122: replay with the linear fractions")
    ap.add_argument("--training-dir")
    ap.add_argument("--replenish", help="F40's replenish (EXP-0110 start_models.pt), for --recollect")
    ap.add_argument("--ordered", action="store_true", help="EXP-0124: the order-aware accept pass")
    ap.add_argument("--linear", action="store_true", help="the records carry v15's linear fractions")
    ap.add_argument("--grid", choices=("exp0121", "exp0125"), default="exp0121")
    ap.add_argument("--arrival", action="store_true", help="--recollect with the arrival-position feature")
    ap.add_argument("--models", required=True, help="the F40 candidate (heads only need its hidden size)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true", help="400 turns, 2 settings at 2 epochs")
    a = ap.parse_args(argv)
    full = GRID_0125 if a.grid == "exp0125" else GRID
    grid = ([{**full[0], "epochs": 2}, {**full[-1], "epochs": 2}] if a.dry_run else full)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, check = a.records, None
    if a.recollect:
        rp, check = recollect(a.training_dir, a.models, a.replenish, a.records, out, a.workers, a.dry_run,
                              a.arrival)
        print(f"[turn_probe] recollect {check}", flush=True)
        if not check["match"]:
            (out / "probe.json").write_text(json.dumps({"recollect": check}, indent=1) + "\n")
            return 2
        records = str(rp)
    jobs = [{**g, "records": records, "models": a.models, "limit": 400 if a.dry_run else None,
             "linear": a.recollect or a.linear, "ordered": a.ordered,
             "arrival": bool(g.get("arrival", False)) and a.arrival,
             "strip_arrival": a.arrival and not g.get("arrival", False)}
            for g in sorted(grid, key=lambda g: -g["width"] * g["set_layers"] * g["epochs"])]
    t0 = time.perf_counter()
    # each setting's result is appended as soon as it finishes, so a hard stop keeps the finished ones
    partial = out / "probe_partial.jsonl"
    results = []
    if a.workers <= 1:
        for j in jobs:
            results.append(probe_one(j))
            with partial.open("a") as fh:
                fh.write(json.dumps(results[-1], default=float) + "\n")
    else:
        with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=get_context("spawn")) as ex:
            for fut in as_completed([ex.submit(probe_one, j) for j in jobs]):
                results.append(fut.result())
                with partial.open("a") as fh:
                    fh.write(json.dumps(results[-1], default=float) + "\n")
    w = winner(results, need_arrival=a.grid == "exp0125")
    doc = {"recollect": check, "grid": a.grid,
           "winner": None if w is None else {k: w.get(k) for k in ("width", "set_layers", "epochs", "arrival")},
           "winner_seconds_per_epoch": None if w is None else w["seconds_per_epoch"],
           "results": results, "records_sha256": sha256_file(records), "models_sha256": sha256_file(a.models),
           "source_commit": a.source_commit, "dry_run": a.dry_run, "seconds": round(time.perf_counter() - t0, 1),
           "lr": LR, "schedule": "cosine", "batch": BATCH, "cap": CAP, "seed": SEED, "gate": GATE}
    (out / "probe.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    for r in sorted(results, key=lambda r: (r["width"], r["set_layers"], r["epochs"])):
        h = r["heldout"]
        print(f"[turn_probe] w{r['width']} L{r['set_layers']} e{r['epochs']} arrival {r.get('arrival')}: "
              f"ties {r['tie_split']['tie_dependent']['agreement']} determinable "
              f"{r['tie_split']['determinable']['agreement']} accept {h['accept']:.4f} "
              f"rank_accept {h['rank_accept']:.4f} rank_fund {h['rank_fund']} displaced "
              f"{h['accept_errors']['declined_displaced']['rate']} ({r['seconds_per_epoch']:.1f} s/epoch)",
              flush=True)
    print(f"[turn_probe] winner {doc['winner']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
