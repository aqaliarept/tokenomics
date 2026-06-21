The current evidence says learning starts failing much earlier than the full complex environment.

  The first documented failure point is:

  fixed depth-3 topology, no topology randomization, no degradation, no weak-producer competition, single product, variable
  quantity near full range
  That is roughly --max-diff 0.24.

  At that point:

  - topology is fixed;
  - product mixture is still off;
  - producer competition is still off;
  - degradation is still off;
  - survival is fine;
  - critic can look healthy in some probes;
  - but the trained policy fulfills almost nothing.

  The policy does not fail because the world is fully randomized. It fails because it does not discover the depth-3
  production chain.

  In simple words:

  Consumer wants final product.
  Final product requires intermediate product.
  Intermediate product requires lower product.
  Producers must replenish inputs, hold inventory, reserve capacity, assemble, and route supply in time.

  The learned producer does not learn that chain. It mostly learns “do not spend capacity.” So it survives, but does not
  fulfill.

  The strongest evidence is the policy-swap test:

  baseProd + baseOrd      157 units
  trainProd + trainOrd     22 units
  trainProd + baseOrd       1 unit
  baseProd + trainOrd     339 units

  Meaning:

  - trained ordering is actually good;
  - trained producing is the blocker;
  - if we keep baseline producers and use trained ordering, performance is excellent;
  - if we use trained producers, fulfillment dies.

  So the real failure is not “RL cannot order.” It is:

  RL cannot learn producer-side replenishment / inventory / assembly behavior in the depth-3 chain from sparse final
  consumer reward.

  DAgger was added because PPO from scratch could not discover this behavior. DAgger means Dataset Aggregation: run the
  learner, see the states it reaches, ask the handwritten policy what it would do there, add those examples, retrain. It is

  Your concern is valid: DAgger is not yet “trained agent discovers better-than-human policy.” It is closer to “trained
  agent learns to imitate a working policy across states.” It is useful because it proved the network can execute much of
  the behavior, but it does not satisfy the larger research goal by itself.

  - Simple/shallow environment: PPO can learn.
  - Big fixed depth-3, easiest-ish demand rung: PPO fails on producing.
      - held-out: 214 < 242/253

  - Still not beating handwritten held-out policy.

  What we do not yet know precisely:

  - whether failure starts at depth-2 or only depth-3;
  - whether fixed quantity p=0.00 is learnable at depth-3;
  - whether failure starts exactly when quantity randomization increases, or already with depth-3 fixed quantity.

  So the next clean experiment should map the boundary:

  depth 1, p=0.00
  depth 2, p=0.00
  depth 3, p=0.00
  depth 3, p=0.10
  depth 3, p=0.24
  depth 3, p=0.25

  For each, run from-scratch PPO and policy-swap. That will tell us whether the first hard wall is chain depth, variable
  quantity, or product mixture.

  My current read: the first real wall is probably depth-3 producer credit assignment, made worse by variable quantity.
  Product mixture and topology randomization are later generalization problems, not the original learning failure.

---

## RESULT — boundary grid (2026-06-21)

Ran the exact grid above: from-scratch PPO (400 iters, 1024 steps, seed 0) on a FIXED scale
topology per depth, held at the cell's quantity rung (`--max-diff`), then the policy-swap
probe. Driver: `tokenomics.train.boundary_grid` (materializes the depth configs, fans out via
`tokenomics.train.parallel`, runs `eval.swap_eval` per checkpoint). Held-out = 5 disjoint seeds;
cu✓ = consumer units fulfilled.

| cell    | depth | p    | train cu✓ | base+base | train/base | base+train | train+base       | verdict            |
|---------|-------|------|-----------|-----------|------------|------------|------------------|--------------------|
| d1_p00  | 1     | 0.00 | 539       | 562       | 1.00       | 562        | 562              | solved (parity)    |
| d2_p00  | 2     | 0.00 | 524       | 561       | 0.98       | 555        | 544              | solved (parity)    |
| d3_p00  | 3     | 0.00 | 129       | 362       | 0.37       | 392        | 39  (80% dead)   | WALL — producer    |
| d3_p10  | 3     | 0.10 | 99        | 362       | 0.34       | 258        | 54  (80% dead)   | producer           |
| d3_p24  | 3     | 0.24 | 93        | 362       | 0.34       | 154        | 52               | producer + death   |
| d3_p25  | 3     | 0.25 | 106       | 362       | 0.40       | 269        | 52               | producer + death   |

Conclusions (the open questions above, resolved):

1. **The wall is chain DEPTH, not variable quantity.** Depth 1 and depth 2 reach baseline parity
   (ratio ~1.0). Depth 3 collapses to ~37% of baseline *with quantity fixed* (p=0.00) — the
   fulfillment failure is fully present before any quantity randomization. Failure starts at
   depth 3, not depth 2; fixed quantity p=0.00 is NOT learnable at depth 3.

2. **The blocker is the PRODUCER (confirmed by swap).** At every depth-3 cell `base+train ≈
   base+base` (trained ordering is good, sometimes better), while `train+base` collapses to
   28–46 cons✓, often 80% death. RL learns ordering; it does not learn depth-3 producer
   replenishment/inventory/assembly.

3. **Variable quantity adds a SEPARATE failure: survival.** At p≤0.10 held-out death = 0; at
   p≥0.24 it jumps to 0.40–0.60 (episodes die at cyc 70–83). Quantity does not create the
   fulfillment wall (already there at p=0) but breaks survival on top of it.

Artifacts: `tokenomics/boundary_runs/grid-cloud/grid/` (per-cell `.log`/`.pt`/`.csv`/`.swap.log`
+ materialized `configs/scale_d{1,2,3}.json`). Run on a Runpod 25-vCPU pod, ~15 min wall.

Next: the fix is producer-side — get the depth-3 producer to learn the chain (DAgger warm-start
from `stock_aware`, or denser producer credit assignment). Not an ordering or generalization
problem.