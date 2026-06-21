# Experiments

Empirical training/eval runs and their results. **ADRs** (`docs/adr/`) record *design
decisions*; **experiments** record *what happened when we ran it*. `VERIFICATION.md` holds open
questions and methodology and links out to the experiment that answered each.

## Index

| ID | Date | Question | Verdict |
|---|---|---|---|
| [EXP-0001](EXP-0001-boundary-grid/RESULT.md) | 2026-06-21 | Depth-3 wall: chain depth or variable quantity? Producer or orderer? | Wall = **depth** (fails at depth-3, p=0.00; depth-2 fine). Blocker = **producer**. Variable quantity = separate survival failure. |

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
