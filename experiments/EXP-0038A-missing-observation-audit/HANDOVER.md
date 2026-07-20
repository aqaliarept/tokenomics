# EXP-0038A Handover — Missing Observation Audit

## Plain Goal

EXP-0038A continues EXP-0038. The question is no longer "does the current actor learn badly?" — it
does. The question is:

```text
What information does the teacher use that the actor cannot currently see?
```

The teacher label is produced by rolling the full environment forward after a small routing-logit
bump. That gives the teacher access to consequences of the whole system: other orderers, other
orders, downstream bottlenecks, future fulfilment timing, and producer interactions. In the real
application we cannot deploy "teacher emulation" as the policy. We can only add observations that
are available at decision time.

So this experiment must separate two cases:

- **Deployable missing observation.** The label becomes much more stable when we add fields the
  real actor can observe, such as order quantity, order age, global backlog, per-type producer
  state, or recent routing history. Then observation enrichment is licensed.
- **Teacher-only information.** The label only becomes stable when using rollout/future/outcome
  facts. Then this marginal-value teacher is not deployable as an actor target, even if the teacher
  itself is strong.

No new actor should be trained until this audit shows that the label is approximately a function of
some deployable observation.

## Starting Evidence

EXP-0038 dumped 13,940 decisions over s29/s77/s45. It showed:

- Current 7-feature producer rows alias the label badly: repeated rows seen at least 5 times have
  median teacher-label range 10 units and worst range 90.
- The current residual actor explains almost none of the centered label variance: about 1-2% from
  the persisted dump.
- The overclaim was "no feature add rescues it." That is not proven. A post-run audit found that
  adding logged decision context, especially product and quantity, reduces fixed-feature label
  conflicts substantially. EXP-0038A must finish that analysis properly.

## Required Observation Blocks

Build a cumulative observation ladder. Each row is one eligible producer for one order decision.

1. **current_row** — the EXP-0037 7-feature producer row:
   eligibility, makeable, free capacity, reserved capacity, headroom, input availability, output
   inventory.
2. **+order** — product, quantity, stream index, order age, TTL remaining.
3. **+local_queue** — ordering agent backlog size, same-product backlog count, queued quantity
   summaries, oldest/newest age, stream position.
4. **+full_producer_state** — do not collapse to min ratios only. Include per-required-capacity
   capacity, reserved, max capacity, headroom; per-input inventory and inbound; current output
   inventory.
5. **+system_demand** — other ordering agents' visible queues by product/layer, total queued
   quantity by product, consumer demand pressure, same-product demand elsewhere in the system.
6. **+recipe_bottleneck** — product depth, required inputs, downstream product demand, aggregate
   spare capacity by required type, transitive input scarcity.
7. **+history** — last K cycles per producer/orderer: routed quantity, completed quantity,
   rejected/expired orders, received inbound, capacity changes.
8. **oracle_only** — rollout/future/outcome facts, only as an upper bound. Examples: base rollout
   units, bumped rollout units, future demand, downstream fulfilment realized after the decision.
   These must be clearly marked non-deployable.

## Metrics

For every block prefix, compute the same target-learnability checks:

- `n_rows`, `n_decisions`, and rows covered by repeated feature groups.
- Exact-alias label spread for groups seen at least 5 times:
  median range, mean range, max range, fraction with range > 1.
- Near-neighbor label spread, because richer continuous features may not repeat exactly.
- Simple probe fit, not a deployed policy:
  train on two anchored seeds per task, evaluate on the held-out anchored seed; report real-choice
  top-1, regret, centered MSE, and variance explained versus zero predictor.
- Compare every block to simple floors:
  zero predictor, majority/top producer by current row, fixed `blend_0.5`.
- Missingness attribution for top mistakes:
  which newly added fields separate teacher-best from actor-best?

The important metric is not train loss. The important metric is whether held-out label variance and
held-out regret improve when a deployable observation block is added.

## Decision Gate

Observation enrichment is licensed only if a deployable block prefix passes both:

- Label conflict drops materially: median exact/near-neighbor label range near 0 and fraction
  `range > 1` much lower than current_row, with enough repeated/neighbor coverage to be meaningful.
- Held-out probe improves on real choices on at least 2 of 3 gate tasks, not just in-sample.

If only `oracle_only` fixes the conflict, stop this marginal-value actor target. That means the
teacher's best value is only achievable by simulating the teacher, which is not a deployable real
application policy.

If no deployable block helps, stop observation enrichment and change the target.

If a deployable block helps, then and only then open a follow-up actor-training experiment using
that exact observation contract.

## Implementation Notes

- Start by reusing the existing EXP-0038 `dump/` for the blocks it already contains:
  current row, product, quantity, age, TTL, backlog, same-product backlog, stream index, agent.
- If those fields are not enough, extend `tokenomics.eval.value_router_errors` or add a sibling
  diagnostic driver to capture the missing full-system blocks from live env state.
- Keep this diagnostic separate from actor training. It should write an auditable table first:
  `block_summary.csv`, `alias_by_block.csv`, `probe_by_block.csv`, `top_mistakes_by_block.csv`,
  and `spec.json`.
- Do not use teacher rollout outputs as deployable features. They belong only in `oracle_only`.

## Non-Goals

- Do not train or deploy a new orderer.
- Do not run the 81-row archive.
- Do not declare the teacher target unlearnable until the deployable observation ladder fails.
- Do not declare observation enrichment useful from in-sample loss alone.
