"""EXP-0129: is F40's start flatter, or does the new on-policy code path differ? (supervisor 2026-11-07, after EXP-0128's
partial result: cell B, full λ and gates only from F40, read absent). Pre-registration v18.

EXP-0128's cell B again (full λ, the investment gates only, the turn heads and replenish greedy and frozen; v16's
machinery and seeds), but from **EXP-0100's start** (EXP-0110 `start_models.pt`, EXP-0112's start). That start has no
turn heads, so the turns play the default auction (ADR-0062: byte-identical to EXP-0112's env). It replicates EXP-0112
(producer |∇J|² 95% CI [5.7e4, 8.7e5], E3 n80 = 48 seeds per half) on the new code.

Reading: `signal_present` if the producer |∇J|² 95% CI's lower bound is > 0 **and** E3's n80 ≤ 96 seeds per half
(EXP-0112's order); then F40's start is flatter (its gates sit near a local optimum). Otherwise `absent`: the new code
path differs from EXP-0112's, so diff the two paths (estimator, perturbation, ε, reward) before anything else.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import m6_imitation as im
from . import m6_stage0c as sc

PREREG_V19_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v19.json"
PREREG_V19_SHA256: str | None = "7e223cd285ddc49380b540fadf5237cfabf6c22086181aae15cca07b40167146"  # EXP-0130 (cell B from F40), frozen before data
PREREG_V18_PATH = Path(__file__).resolve().parent / "registry" / "m6_prereg_v18.json"
PREREG_V18_SHA256: str | None = "4c4b1b1a7cc6b448b4ec48d846e063a5d8cb790de005daf7bc68579f6d08109a"  # frozen before EXP-0129 data
EXP0112 = {"grad_norm_sq_ci95": [56983.6, 865678.1], "n80_E3": 48, "params": 27524}


def registered_prereg_v18() -> dict:
    return {
        "format": "m6-prereg-v18", "change": "add-m6-joint-mpg-gnn", "task": "4.4 diagnostic",
        "stage": "EXP-0129: EXP-0128's cell B from EXP-0100's start (EXP-0112 replicated on the new code)",
        "design": "m6_stage0c with heads = gates (the investment gates only; replenish and the turns greedy and "
                  "frozen), λ full (EXP-0110's calibration), v16's 24 seeds × 4 siblings and ±ε along 4 directions at "
                  "KL 0.01; start = EXP-0110 start_models.pt for both the trunk / gates and the replenish head; no "
                  "turn heads, so the turns play the default auction",
        "reference": f"EXP-0112 (same parameter set, full λ): {EXP0112}",
        "reading": "signal_present if the producer |∇J|² 95% CI lower bound > 0 and E3 n80 ≤ 96 seeds per half "
                   "(F40's start is flatter: a DIAGNOSTICS entry); else absent (the new code path differs from "
                   "EXP-0112's: diff estimator, perturbation, ε and reward before anything else)",
        "reported": "v9's rule reading, every estimator's n80 and cosine, |∇J|² with CI, flip rates, deaths",
        "operations": "one seed timed first; stop = 1.5 × that timing; each job's row and npz written as it "
                      "finishes; a watchdog pulls the run directory before any stop (CPU pods have no volume)",
    }


def registered_prereg_v19() -> dict:
    """EXP-0130: EXP-0128's cell B re-run from F40 (its CI / n80 were lost), under v18's exact criterion."""
    v18 = registered_prereg_v18()
    return {
        "format": "m6-prereg-v19", "change": "add-m6-joint-mpg-gnn", "task": "4.4 diagnostic",
        "stage": "EXP-0130: EXP-0128's cell B from F40 (the same configuration as EXP-0129, F40's start)",
        "design": v18["design"].replace(
            "start = EXP-0110 start_models.pt for both the trunk / gates and the replenish head; no turn heads, so "
            "the turns play the default auction",
            "start = the F40 + v15 composite (EXP-0126 models.pt: F40's trunk, gates, v15 turn heads; the EXP-0110 "
            "start's replenish head), the turn heads and replenish greedy and frozen"),
        "reference": "EXP-0129 (same configuration, EXP-0100's start): |∇J|² CI [4.9e4, 5.8e5], E3 n80 96; "
                     f"EXP-0112: {EXP0112}",
        "reading": "v18's exact criterion: signal_present if the producer |∇J|² 95% CI lower bound > 0 and E3 n80 ≤ "
                   "96 seeds per half; else absent. With EXP-0129 signal_present: absent here = F40's start is "
                   "flatter (a DIAGNOSTICS entry); signal_present here = not flatter on this criterion, so the "
                   "turn heads or λ / 4 carry 4.4's absent (EXP-0127)",
        "reported": v18["reported"], "operations": v18["operations"],
    }


def freeze(variant: str = "exp0100") -> str:
    from ..eval.provenance import sha256_file

    path, doc = ((PREREG_V19_PATH, registered_prereg_v19()) if variant == "f40"
                 else (PREREG_V18_PATH, registered_prereg_v18()))
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return sha256_file(str(path))


def prereg_problems_v19() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V19_SHA256 is None:
        return ["no pinned m6_prereg_v19.json (`--mode freeze --variant f40`)"]
    if not PREREG_V19_PATH.exists() or sha256_file(str(PREREG_V19_PATH)) != PREREG_V19_SHA256:
        return ["m6_prereg_v19.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V19_PATH.read_text()) != registered_prereg_v19():
        return ["the v19 pre-registration differs from the code"]
    return []


def prereg_problems() -> list[str]:
    from ..eval.provenance import sha256_file

    if PREREG_V18_SHA256 is None:
        return ["no pinned m6_prereg_v18.json (`--mode freeze`)"]
    if not PREREG_V18_PATH.exists() or sha256_file(str(PREREG_V18_PATH)) != PREREG_V18_SHA256:
        return ["m6_prereg_v18.json does not match its pinned SHA-256"]
    if json.loads(PREREG_V18_PATH.read_text()) != registered_prereg_v18():
        return ["the v18 pre-registration differs from the code"]
    return []


def reading(doc: dict) -> dict:
    gr = (doc.get("gradient") or {}).get("12/producer") or {}
    lo = (gr.get("grad_norm_sq_ci95") or [None, None])[0]
    n80 = ((doc.get("seeds_needed") or {}).get("12/producer") or {}).get("E3", {}).get("n80")
    present = lo is not None and lo > 0 and n80 is not None and n80 <= 96
    return {"reading": "signal_present" if present else "absent", "grad_norm_sq_ci95": gr.get("grad_norm_sq_ci95"),
            "grad_norm_sq": gr.get("grad_norm_sq"), "n80_E3": n80, "v9_reading": doc.get("reading"),
            "reference_exp0112": EXP0112}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("freeze", "run", "time"), required=True)
    for a_ in ("--training-dir", "--atlas", "--start-models", "--exp0110", "--monitor-pkl", "--turn-records",
               "--out-dir"):
        ap.add_argument(a_)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--source-commit", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variant", choices=("exp0100", "f40"), default="exp0100")
    ap.add_argument("--turn-models", help="f40: EXP-0126 models.pt (the composite's trunk / gates / turn heads)")
    a = ap.parse_args(argv)
    if a.mode == "freeze":
        print(freeze(a.variant))
        return 0
    im.SOURCE_COMMIT = a.source_commit
    f40 = a.variant == "f40"
    rc = sc.run(a.training_dir, a.atlas, a.turn_models if f40 else a.start_models, a.start_models, a.exp0110,
                a.monitor_pkl, a.turn_records, a.out_dir, a.workers, a.dry_run, time_only=a.mode == "time",
                lambda_scale=1.0, heads="gates", prereg_check=prereg_problems_v19 if f40 else prereg_problems)
    if a.mode == "run":
        doc = json.loads((Path(a.out_dir) / "reading.json").read_text())
        rd = reading(doc)
        name = "reading_exp0130.json" if f40 else "reading_exp0129.json"
        (Path(a.out_dir) / name).write_text(json.dumps({**rd, "prereg": PREREG_V19_SHA256 if f40 else PREREG_V18_SHA256},
                                                       indent=1, default=float) + "\n")
        print(f"[diag_start] {rd}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
