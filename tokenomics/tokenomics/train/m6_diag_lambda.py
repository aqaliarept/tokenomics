"""EXP-0128: why 4.4 read `absent` (supervisor 2026-11-07; the root-cause rule before 4.5). A 2 × 2 on v16's machinery
and seeds (`m6_stage0c`), pre-registration v17:
- λ: full (0.736, EXP-0110's calibration) vs λ / 4 (0.184, 4.5's re-weighting);
- trainable heads: all (v16: gates, replenish, fund order, accept order and bits) vs gates only (EXP-0112's parameter
  set: the turn heads and replenish played greedily and frozen).

Cells: A = (full, all), B = (full, gates), C = (λ/4, gates) run here; D = (λ/4, all) is EXP-0127 (same design, seeds
and code path), re-used. A cell reads `signal` under v9's rule (its primary reading, or for an all-heads cell its
replenish-greedy counterfactual, the variant 4.5 would play when replenish swamps).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import m6_imitation as im
from . import m6_stage0c as sc

PREREG_V17_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v17.json"
PREREG_V17_SHA256: str | None = "419518c02cd153f2aae65ad195052d12b51dada8f4e4d17b52688ec056e0a2a1"  # frozen before EXP-0128 data
CELLS = {"A": (1.0, "all"), "B": (1.0, "gates"), "C": (0.25, "gates")}

READINGS = {  # (A, B, C) signal -> reading; D = EXP-0127 = absent
    (True, True, False): "lambda_cut",
    (False, True, True): "parameter_spread",
    (False, True, False): "both",
    (True, True, True): "either_suffices",
    (False, False, False): "neither",
}


def registered_prereg_v17() -> dict:
    return {
        "format": "m6-prereg-v17", "change": "add-m6-joint-mpg-gnn", "task": "4.4 diagnostic",
        "stage": "EXP-0128: why the on-policy power check read absent (λ re-weighting vs parameter spread)",
        "design": "2 × 2 on m6_stage0c (v16's machinery, seeds, calibration, ε rule): λ ∈ {full 0.736, λ/4 0.184} × "
                  "heads ∈ {all, gates only (EXP-0112's set: turn heads and replenish greedy and frozen)}; cells A = "
                  "(full, all), B = (full, gates), C = (λ/4, gates) run in one pod; D = (λ/4, all) = EXP-0127, "
                  "re-used (identical design and seeds)",
        "cell_signal": "v9's rule at 12 producers on the cell's primary reading; an all-heads cell also counts its "
                       "replenish-greedy counterfactual (what 4.5 plays when replenish swamps)",
        "reading": {"A, B signal, C absent": "lambda_cut", "B, C signal, A absent": "parameter_spread",
                    "B only": "both (full λ and gates only are both needed)",
                    "A, B, C signal": "either_suffices (each change alone restores it)",
                    "none": "neither (something else; reported, not guessed)",
                    "any other pattern": "other (reported as observed)"},
        "reported": "per cell: the reading, n80 per estimator, |∇J|² with CI, per-head score shares, flip / swap rates",
    }


def freeze() -> str:
    from ..eval.provenance import sha256_file

    PREREG_V17_PATH.write_text(json.dumps(registered_prereg_v17(), indent=1, sort_keys=True) + "\n")
    return sha256_file(str(PREREG_V17_PATH))


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V17_SHA256 is None:
        return ["no pinned m6_prereg_v17.json (`--mode freeze`)"]
    if not PREREG_V17_PATH.exists() or sha256_file(str(PREREG_V17_PATH)) != PREREG_V17_SHA256:
        return ["m6_prereg_v17.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V17_PATH.read_text()) != registered_prereg_v17():
        return ["the v17 pre-registration differs from the code"]
    return []


def cell_signal(doc: Mapping[str, Any]) -> bool:
    if doc.get("reading") == "signal":
        return True
    cf = doc.get("without_replenish_score")
    return bool(doc.get("cell", {}).get("heads", "all") == "all" and cf and cf.get("reading") == "signal")


def combine(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    sig = {k: cell_signal(v) for k, v in cells.items()}
    key = (sig["A"], sig["B"], sig["C"])
    reading = READINGS.get(key, "other") if not sig.get("D") else "other"
    return {"reading": reading, "signal": sig}


def main(argv: Sequence[str] | None = None) -> int:
    from ..eval.provenance import sha256_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("freeze", "run"), required=True)
    for a_ in ("--training-dir", "--atlas", "--turn-models", "--replenish-models", "--exp0110", "--monitor-pkl",
               "--turn-records", "--out-dir", "--exp0127-reading"):
        ap.add_argument(a_)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cells", default="A,B,C")
    a = ap.parse_args(argv)
    if a.mode == "freeze":
        print(freeze())
        return 0
    im.SOURCE_COMMIT = a.source_commit
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs: dict[str, Any] = {}
    rc = 0
    for name in a.cells.split(","):
        lam, heads = CELLS[name]
        print(f"[diag_lambda] cell {name}: λ × {lam}, heads {heads}", flush=True)
        rc |= sc.run(a.training_dir, a.atlas, a.turn_models, a.replenish_models, a.exp0110, a.monitor_pkl,
                     a.turn_records, str(out / f"cell_{name}"), a.workers, a.dry_run, lambda_scale=lam, heads=heads,
                     prereg_check=prereg_problems)
        docs[name] = json.loads((out / f"cell_{name}" / "reading.json").read_text())
    d = json.loads(Path(a.exp0127_reading).read_text())
    docs["D"] = {**d, "cell": {"lambda_scale": 0.25, "heads": "all", "reused": "EXP-0127"}}
    comb = combine(docs) if all(k in docs for k in "ABCD") else {"reading": None}
    summary = {k: {"reading": v.get("reading"), "counterfactual": (v.get("without_replenish_score") or {}).get("reading"),
                   "n80": (v.get("seeds_needed") or {}).get("12/producer"),
                   "gradient": (v.get("gradient") or {}).get("12/producer"),
                   "shares": v.get("family_score_share"), "rates": v.get("rates"), "cell": v.get("cell")}
               for k, v in docs.items()}
    doc = {"combined": comb, "cells": summary, "prereg": PREREG_V17_SHA256, "dry_run": a.dry_run,
           "exp0127_reading_sha256": sha256_file(a.exp0127_reading), "source_commit": a.source_commit}
    (out / "reading.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(f"[diag_lambda] reading {comb['reading']} signal {comb.get('signal')}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
