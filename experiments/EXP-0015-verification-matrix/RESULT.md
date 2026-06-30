# EXP-0015 — Verification / transfer matrix (C-1, the first deliverable)

Status: **DONE (PASS)** — `tokenomics.eval.transfer_matrix` emits the PRODUCT_AGNOSTIC
"Evaluation Output" compatibility-and-behavior matrix from the already-distilled
checkpoints, with **contract flags separated from value** per cell. 30 rows (6 stacks ×
5 targets), **zero contract failures**, floor reads **0 units**, the capacity-relabel
target is **value-identical** (units/rejections/death/ratio) to its base seed for every
stack, and the portable cells reproduce EXP-0012 / 0013 / 0014. Runner gated by
`tests/test_transfer_matrix.py` (5 tests). Local eval only.

(The matrix records aggregate value columns, so its relabel check is *value*-identical.
The stronger **byte-identical** relabel gate — units **and cycles and death** equal — is
EXP-0014's `joint_stack.relabel_gate` (4/4), not re-derived here.)

## Question
Can one reusable runner produce the doc's stated **first deliverable** — a matrix that
tells whether a missing number is a *contract* failure (checkpoint can't run here) or a
*policy* failure (runs but poor) — populated from artifacts we already have? See
`spec.json`.

## Setup
- **Runner** — `transfer_matrix(stacks, targets)` → one row per (stack × target);
  `write_matrix_csv` emits the canonical columns (model/kinds, target dims, schema-load
  + contract-compatible flags, `runs_episode`, `failure_class`, value, death,
  `elapsed_sec`, baseline family/id/units/death, `ratio_vs_practical_baseline`,
  `ratio_vs_best_available_bar`, `baseline_contract_compatible`). Every
  stack runs on the **same** `joint_stack_env` (entity producer obs + corrected v4
  product-stream ordering) so deterministic bars and portable modules are
  apples-to-apples.
- **Stacks** — floor (`noop+proportional`), practical bar (`entity_planner+proportional`,
  the ratio denominator), `stock_aware+proportional`, **portable producer + det. orderer**
  (C0 producer side, EXP-0013), **det. producer + portable orderer** (C0 orderer side,
  EXP-0012), **full portable stack** (C1, EXP-0014). No re-train.
- **Targets** — train-size held-out scale seeds 6–9, depth 3 (P=12), 5 episodes each,
  + one capacity-relabel target (`scale-d3-s6-relabel`). Horizon 100.

## Result (ratio vs `entity_planner + proportional` on the same target)

| stack | s6 | s7 | s8 | s9 | mean | min | dead |
|-------|----|----|----|----|------|-----|------|
| noop+proportional (floor) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 |
| entity_planner+proportional (bar) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| stock_aware+proportional | 0.911 | 0.910 | 0.788 | 0.840 | 0.862 | 0.788 | 0 |
| **portable_producer + proportional** (C0 prod) | 1.000 | 1.000 | 1.022 | 1.012 | **1.009** | 1.000 | 0 |
| **stock_aware + portable_orderer** (C0 ord) | 1.075 | 0.990 | 1.157 | 1.158 | **1.095** | 0.990 | 0 |
| **portable + portable** (C1 full stack) | 1.073 | 1.042 | 1.173 | 1.149 | **1.109** | 1.042 | 0 |

All 30 cells: `producer_schema_loads = orderer_schema_loads = True`,
`failure_class = none`. The `scale-d3-s6-relabel` rows equal the `scale-d3-s6` rows
exactly (units + ratio) for every stack. Full rows in `summary.csv`.

## Conclusion
**PASS.** The matrix is the deliverable and it cleanly separates the four failure modes:

1. **Contract vs policy is now a column, not a guess.** Every cell records
   `*_schema_loads` / `*_contract_compatible` / `runs_episode` / `failure_class`
   independently of value. Here all portable + deterministic stacks are
   contract-compatible (zero contract failures); a missing value would therefore read as
   policy, not shape. (Legacy fixed-contract `GaussianActor` checkpoints are the cells
   that *will* fail contract once pointed at an off-size target — a later rung; none is
   included here.)
2. **The floor proves the targets are nontrivial.** `noop+proportional` = 0 units on
   every seed, so the positive ratios are real economic work, not reward inflation.
3. **C0 (independent portable modules) is now an apples-to-apples row** — the missing
   table called out in the C0 write-up. Portable producer + deterministic orderer
   **ties/beats** the planner bar (mean 1.009, min 1.000); deterministic producer +
   portable orderer adds the routing lift (mean 1.095). Each portable module wins solo
   before being composed.
4. **The full portable stack reproduces C1** (mean 1.109, min 1.042, zero death) —
   exact cross-check of EXP-0014, confirming the matrix path and the joint_stack path
   agree.
5. **Same-shape relabel equivariance** holds end-to-end: the relabel target is
   value-identical (units/rejections/death/ratio) to its base seed for all six stacks.
   The full byte-identical proof (incl. cycles) is EXP-0014's `relabel_gate`.

**Reading the numbers.** The lift is orderer-driven — the portable producer matches the
planner, the portable orderer beats `proportional` — so `portable+portable` ≈
`stock_aware+portable_orderer` and both sit above the bar. `stock_aware+proportional`
(0.862) shows the practical bar is the *strong* deterministic stack, not a weak one.

**Not covered:** size transfer (C2 — larger product/producer counts) and out-of-family
recipes (P2 / O3 remainder / C3). Those are the next matrix targets; the runner already
takes arbitrary `TargetTopology`s, so they are added by widening `targets`, not new code.

## Rerun command
```
python -m tokenomics.eval.transfer_matrix \
  --producer-ckpt experiments/EXP-0013-portable-producer-scorer/artifacts/producer_scorer.pt \
  --orderer-ckpt experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt \
  --depth 3 --seeds 6-9 --eval-seeds 5 --horizon 100 \
  --out experiments/EXP-0015-verification-matrix/summary.csv
```
Local eval (evaluation of existing checkpoints — no training).

## Artifacts
- `summary.csv` — the 30-row verification matrix (canonical "Evaluation Output" columns).
- (checkpoints reused in place from `EXP-0012-.../artifacts/` and
  `EXP-0013-.../artifacts/`; this experiment adds no new ckpt.)
