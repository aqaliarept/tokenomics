# Experiments

Empirical training/eval runs and their results. **ADRs** (`docs/adr/`) record *design
decisions*; **experiments** record *what happened when we ran it*. `VERIFICATION.md` holds open
questions and methodology and links out to the experiment that answered each.

## Index

| ID | Date | Question | Verdict |
|---|---|---|---|
| [EXP-0001](EXP-0001-boundary-grid/RESULT.md) | 2026-06-21 | Depth-3 wall: chain depth or variable quantity? Producer or orderer? | Wall = **depth** (fails at depth-3, p=0.00; depth-2 fine). Blocker = **producer**. Variable quantity = separate survival failure. |
| [EXP-0002](EXP-0002-producer-warm-start/RESULT.md) | 2026-06-22 | Can a clone of `obs_stock_aware` clear the d3_p00 producer gate, why does plain cloning fail, and does it survive slot permutation? | **Phase 1 PASS** (decode-aware clone+base 373.6, ratio 1.03; plain MSE FAILS 0.60 — gated decode + sign error, restore FP 485). **Phase 2 PASS**: multi-topology + slot permutation hold (held-out seeds 0.92–1.19; held-out *permutations* of trained economy 0.85–1.02, above bar — learned local rule, not slots; d1/d2 regress 1.00). |
| [EXP-0003](EXP-0003-producer-standalone-improvement/RESULT.md) | 2026-06-23 | Can the Phase-2 producer clone be improved **beyond** `stock_aware` by producer-only RL while the orderer stays fixed to `proportional`? | **REJECT (v1 plateaus).** Clone survives all rungs (0 death) but has a systematic **rung-0.24** weakness (0.58–0.69) + a weak held-out tail. Producer-only PPO (conservative, BC-anchored) only **matches** the clone (best-select ratio 1.030) — critic EV ≈ 0 so PPO has no signal; it *redistributes* the weakness and regresses a held-out cell (0.95→0.58). Real failure surface remains → **EXP-0004** warranted. New tools: `eval.producer_rungs`, `train.producer_improve`. |
| [EXP-0004](EXP-0004-rung-quantity-weakness/RESULT.md) | 2026-06-23 | Is the EXP-0003 rung-0.24 weakness an **expert** weakness (fix the labeling rule / adversarial gen) or a **cloning-fidelity** gap, and does rung-staged DAgger close it? | **Stage A: fidelity gap** — `obs_stock_aware` (teacher) == `stock_aware` (bar) on all rungs (1.00); `d3_perm` underperforms *its own teacher* at rung 0.24. **B→C→D: best clone, still REJECT (strict).** Coverage is the lever — v1 (`--rungs {0..0.25}`) closes rung-0.24 but kills native; v2 (full ladder) restores native but re-weakens rung-0.24 (Pareto); **v3 (full ladder + `--rung-weights 0.00:3,0.10:3,0.24:3`, Runpod CPU) broke the trade** (5/16→3/16); **v4 (`--train-seeds 6`, Runpod CPU) closed seed-5 + generalizes to 3/4 unseen seeds** but split the residual: rung-0.24 *relocated* to held-out seed-9 (0.41, diminishing returns), and seed-3 p=0 was coverage-proof. **The orderer probe then resolved seed-3 p=0** — it fails *only* under `proportional` (0.04), recovers to **1.00** under `even_split`/`obs_price` → an **orderer-coupled** min-quantity cliff, **excluded from the producer bar**. **Verdict: ACCEPT v4 as the new producer-clone default** (dominates v3 + `d3_perm`); lone producer residual = held-out seed-9 p=0.24, diminishing-returns coverage; never RL. Tools: `producer_rungs --train-baseline/--order`, `warm_start --rungs/--rung-weights`, `producer_trace --config-seed/--rung`. |

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
