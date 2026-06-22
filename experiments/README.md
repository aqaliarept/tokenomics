# Experiments

Empirical training/eval runs and their results. **ADRs** (`docs/adr/`) record *design
decisions*; **experiments** record *what happened when we ran it*. `VERIFICATION.md` holds open
questions and methodology and links out to the experiment that answered each.

## Index

| ID | Date | Question | Verdict |
|---|---|---|---|
| [EXP-0001](EXP-0001-boundary-grid/RESULT.md) | 2026-06-21 | Depth-3 wall: chain depth or variable quantity? Producer or orderer? | Wall = **depth** (fails at depth-3, p=0.00; depth-2 fine). Blocker = **producer**. Variable quantity = separate survival failure. |
| [EXP-0002](EXP-0002-producer-warm-start/RESULT.md) | 2026-06-22 | Can a clone of `obs_stock_aware` clear the d3_p00 producer gate, why does plain cloning fail, and does it survive slot permutation? | **Phase 1 PASS** (decode-aware clone+base 373.6, ratio 1.03; plain MSE FAILS 0.60 — gated decode + sign error, restore FP 485). **Phase 2 PASS**: multi-topology + slot permutation hold (held-out seeds 0.92–1.19; held-out *permutations* of trained economy 0.85–1.02, above bar — learned local rule, not slots; d1/d2 regress 1.00). |

## Layout & conventions

Each experiment is a self-contained dir `EXP-NNNN-kebab-title/` (numbering parallels ADRs —
sortable, stable, citable):

```
EXP-NNNN-title/
  spec.json        # declarative reproducer: driver, params, seeds, configs, compute, date
  RESULT.md        # the finding — question, method, result table, conclusions, follow-up
  summary.csv      # machine-readable per-cell/per-run metrics
  configs/         # materialized INPUT configs (the experiment definition)
  artifacts/       # checkpoints (.pt), raw logs, per-iter curves  → git-lfs (see .gitattributes)
```

- **`spec.json` / `RESULT.md` / `summary.csv` / `configs/`** are normal git (small, diff-able —
  they *are* the durable record).
- **`artifacts/`** is tracked via **git-lfs** (`.gitattributes`: `experiments/**/artifacts/** filter=lfs ...`).
  Checkpoints and raw logs are large/binary; LFS keeps the working tree light while preserving
  full provenance. Run `git lfs install` once after cloning.

## Producing a new experiment

The boundary-grid driver writes the whole layout directly — point `--out-dir` at a fresh EXP dir:

```
cd tokenomics
python -m tokenomics.train.boundary_grid --iters 400 --steps 1024 \
    --out-dir ../experiments/EXP-0002-<title>
```

It materializes `configs/`, drops heavy files in `artifacts/`, and emits `spec.json` +
`summary.csv` + an auto `RESULT.md` skeleton (fill in the Conclusions, then add a row above).
For off-box runs, wrap with `run_pod_training.py` and `--pull` the EXP dir back. Other
experiment types should follow the same dir contract (see `_TEMPLATE/`).
