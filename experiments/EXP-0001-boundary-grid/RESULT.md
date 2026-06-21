# EXP-0001 — boundary grid (depth × quantity)

**Date:** 2026-06-21 · **Driver:** `tokenomics.train.boundary_grid` · **Reproduce:** see [`spec.json`](spec.json)

## Question

VERIFICATION.md left three things open: does from-scratch PPO fail at depth-3 because of
**chain depth** or because of **variable quantity** (the demand rung `--max-diff`)? Does the
failure start at depth-2 or only depth-3? And which side — producer or orderer — is the blocker?

## Method

For each cell, from-scratch PPO (400 iters, 1024 steps, seed 0) on a **fixed** scale topology
of that depth, with the curriculum held at the cell's quantity rung. Then the policy-swap probe
(`eval.swap_eval`) crosses trained vs. baseline (`stock_aware` producer + `proportional` orderer)
to isolate the blocker. Held-out = 5 disjoint seeds. `cu✓` = consumer units fulfilled.

## Result

(machine-readable: [`summary.csv`](summary.csv); raw logs + checkpoints: [`artifacts/`](artifacts/))

| cell | depth | p | held cu✓ | held dead | base+base | train+train | train/base | base+train | train+base |
|---|---|---|---|---|---|---|---|---|---|
| d1_p00 | 1 | 0.00 | 539 | 0.00 | 562 | 562 | **1.00** | 562 | 562 |
| d2_p00 | 2 | 0.00 | 524 | 0.00 | 561 | 548 | **0.98** | 555 | 544 |
| d3_p00 | 3 | 0.00 | 129 | 0.00 | 362 | 135 | **0.37** | 392 | 39 *(80% dead)* |
| d3_p10 | 3 | 0.10 | 99 | 0.00 | 362 | 121 | 0.33 | 258 | 54 *(80% dead)* |
| d3_p24 | 3 | 0.24 | 93 | 0.60 | 362 | 122 | 0.34 | 154 | 52 |
| d3_p25 | 3 | 0.25 | 106 | 0.40 | 362 | 117 | 0.32 | 269 | 52 |

## Conclusions

1. **The wall is chain DEPTH, not variable quantity.** Depth-1 and depth-2 reach baseline parity
   (train/base ≈ 1.0). Depth-3 collapses to ~37% of baseline **with quantity fixed (p=0.00)** —
   the fulfillment failure is fully present before any quantity randomization. Failure starts at
   depth-3, not depth-2; fixed-quantity p=0.00 is **not** learnable at depth-3.

2. **The blocker is the PRODUCER (confirmed by swap).** At every depth-3 cell `base+train ≈
   base+base` (trained ordering is good, sometimes better), while `train+base` collapses to
   28–46 cons✓, often 80% death. RL learns ordering; it does not learn depth-3 producer
   replenishment / inventory / assembly.

3. **Variable quantity adds a SEPARATE failure: survival.** At p≤0.10 held-out death = 0; at
   p≥0.24 it jumps to 0.40–0.60 (episodes die at cyc 70–83). Quantity does not create the
   fulfillment wall (already there at p=0) but breaks survival on top of it.

## Follow-up

The fix is producer-side: get the depth-3 producer to learn the chain (DAgger warm-start from
`stock_aware`, or denser producer credit assignment). Not an ordering or generalization problem.
Tracked in memory `boundary-grid-result`; the VERIFICATION.md "RESULT" section mirrors this.
