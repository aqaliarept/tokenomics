# Producer-first staged training before graph-policy migration

Status: accepted

## Context

EXP-0001 mapped the first learning wall: from-scratch PPO reaches baseline parity at
depth 1 and depth 2, but fails at depth 3 even with fixed topology and fixed consumer
quantity (`d3_p00`). Policy-swap evaluation isolates the blocker to the producing
policy: trained ordering with baseline producers is healthy, while trained producers
with the baseline orderer collapse.

The producer trace for `d3_p00` sharpened the diagnosis. The failure is not global
under-ordering. The learned shared MLP producer over-invests in outer layers, starves
the middle layer's restoration until middle capacity reaches zero, and accumulates
blocked composite reservations before required inputs arrive. The handwritten
`stock_aware` and `obs_stock_aware` baselines avoid this with a local rule: restore
when own capacity is low and keep each required input stocked using inventory plus
inbound orders.

This creates two distinct concerns:

1. **Credit assignment and retention.** End-to-end PPO from sparse final consumer reward
   does not reliably discover or retain the depth-3 producer discipline.
2. **Representation for random depth.** A fixed-vector MLP can behave differently by
   product/layer position even when weights are shared. Shared weights do not by
   themselves create depth-agnostic behavior.

The long-term random-depth target may require graph or set-based policies. However, the
current failed benchmark is narrower: fix the depth-3 producer wall first and then test
whether representation, not learning, is the remaining blocker.

## Decision

Switch the training paradigm from "end-to-end PPO discovers the whole producer chain
from scratch" to a staged producer-first process:

1. Train producer behavior in isolation before full joint PPO.
2. Use `obs_stock_aware` as the primary supervised/DAgger expert because it acts through
   the same observation contract as the learned policy.
3. Use producer-isolated evaluation (`train+base`) as the main gate for producer work.
4. Fine-tune with conservative PPO only after the cloned/DAgger producer clears the
   depth-3 producer gate.
5. Add graph-policy work only after the MLP producer is proven to solve `d3_p00` but
   fails depth/topology generalization.

The immediate pass bar is:

```text
d3_p00 train+base >= 0.8 * base+base, with near-zero death
```

Given EXP-0001's `base+base ~= 362` consumer units, the first pass bar is about 290
consumer units on the producer-isolated swap column.

## Implementation Direction

- Add a new `tokenomics.train.warm_start` driver that can build a BC/DAgger checkpoint
  from `obs_stock_aware`, save a PPO-loadable checkpoint, run the swap gate, and
  optionally print the producer trace.
- Keep the baseline orderer fixed to `proportional` for producer-isolated evaluation
  and producer-only training.
- Select checkpoints on `train+base` consumer units and death rate, not on
  `train+train` return alone.
- Preserve raw evaluation reward. If denser producer credit is needed, add it only as
  training-only shaping around the traced failure modes: middle-layer makeability,
  layer capacity balance, reserved capacity waiting on missing inputs, and downstream
  input starvation.
- Treat graph neural networks as a conditional phase. The GNN path should reuse the
  same expert data, DAgger loop, swap gate, and trace checks instead of replacing the
  benchmark.

## Considered Options

- **Continue end-to-end PPO from scratch.** Rejected for the next phase. EXP-0001 shows
  this fails at the simplest depth-3 fixed-quantity cell.
- **Move directly to a GNN producer.** Deferred. A graph policy is likely the right
  long-term representation for random-depth economies, but it would mix a
  representation change with an unresolved producer-credit problem.
- **Clone `stock_aware` as the primary expert.** Rejected as the main target.
  `stock_aware` is useful as a ceiling and diagnostic, but `obs_stock_aware` is the
  fairer scaffold because it uses the same observation contract as the learned policy.
- **Restore producer-order rewards.** Rejected. ADR-0042 showed instrumental churn
  rewards can swamp consumer fulfillment and reward the wrong behavior.

## Consequences

- The research claim changes. The next milestone is not "PPO discovers a better
  producer from scratch"; it is "the learned actor can execute and retain the required
  producer behavior, then PPO can improve or preserve it."
- Producer work has its own acceptance gate before full joint training resumes.
- GNN implementation is not blocked, but it is gated behind evidence that MLP plus
  staged training solves the fixed-depth wall and then fails variable-depth or
  topology generalization.
- Existing depth-1/depth-2 parity remains useful as a regression check, but `d3_p00`
  becomes the primary producer-wall benchmark.
