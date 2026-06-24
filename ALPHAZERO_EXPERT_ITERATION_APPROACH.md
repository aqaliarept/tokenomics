# AlphaZero-Style Expert Iteration Approach

Status: proposed next paradigm

## Goal

Train agents that outperform the best human-defined policy envelope, not only the
currently deployed hardcoded baseline.

The target claim is:

```text
learned policy stack > best human-defined / simple-search baseline envelope
```

The current proven claim from EXP-0005 is narrower:

```text
v4 producer + M2-v1 learned orderer > stock_aware + proportional
```

That is already enough to show training is useful. The next paradigm is for the
larger research goal: make learned agents beat any reasonably complex hand-written
algorithm we can define for the benchmark.

## Current Starting Point

Accepted artifacts:

- Producer: `experiments/EXP-0004-rung-quantity-weakness/artifacts/d3_rung_dagger_v4.pt`
- Full stack: `experiments/EXP-0005-learning-vs-baseline/artifacts/m2_v1.pt`

Known facts:

- The producer problem is mostly solved for the current depth-3 suite. v4 matches the
  `stock_aware` producer well enough to use as the fixed producer default.
- The main exploitable headroom is orderer-side.
- M2-v1 proves learned improvement over `stock_aware + proportional`.
- M2-v1 does not reliably beat the strict random-inclusive envelope.
- EXP-0005 Stage 2b showed the search teacher can beat that envelope.
- EXP-0005 Stage 2c showed "more topology data" does not close the clone gap.

Therefore the next blocker is not PPO, not more producer work, and not more naive
coverage. The next blocker is policy representation and distillation quality for
orderer behavior.

## Why This Is AlphaZero-Style, Not Literal Self-Play

The environment is not naturally a two-player game. There is no opponent that can
generate strategic pressure the way Go self-play does.

The useful analogy is expert iteration:

```text
current policy -> planner/search improves actions -> train policy/value models on
the improved decisions -> evaluate -> promote if better -> repeat
```

AlphaZero uses MCTS to improve actions before training the policy. In this project,
the planner is a deterministic rollout/search oracle over producer/orderer actions.
For the orderer, that currently means sampled per-cycle routing search. Later it can
be upgraded to CEM, beam search, value-guided rollout, or joint producer-orderer
planning.

## Baseline Envelope

The benchmark must compare against an envelope, not one convenient baseline.

Minimum envelope:

- `stock_aware + proportional`
- `stock_aware + even_split`
- `stock_aware + obs_price`
- `stock_aware + random`
- `random + random`
- `v4 + M2-v1`

Optional stronger envelope:

- search teacher from EXP-0005 Stage 2b
- any new hand-coded orderer heuristic
- any static routing heuristic discovered during analysis

The promoted learned model must beat the envelope on held-out topologies, not only
beat `stock_aware + proportional`.

## Training Loop

One expert-iteration cycle has five steps.

1. Roll out the current deployed candidate.

   Start from `v4 + M2-v1`. Collect states from the policy's own visited
   distribution, not only from the oracle's distribution.

2. Improve the action with search.

   At each relevant decision state, run a planner from that exact state and select
   the best action or action distribution. For orderer states, the initial planner
   is sampled routing search with lookahead. Later planners can use CEM or a learned
   value function.

3. Store policy and value targets.

   For each visited state, record:

   - observation
   - role (`producer` or `orderer`)
   - decode-active action mask
   - planner action target
   - planner action distribution, if available
   - planner score
   - final episode score
   - topology seed, episode seed, cycle, rung/domain parameters

4. Train the student.

   Train the actor to match the improved planner behavior. Train a value head to
   predict final consumer units, survival, and optionally reject/waste metrics.

5. Evaluate and promote.

   Promote a model only if it beats the current champion and the envelope on held-out
   topologies with confidence intervals and zero or acceptable death rate.

## Immediate EXP-0006 Scope

EXP-0006 should not start with joint producer/orderer search. The evidence says the
producer is not the current bottleneck. EXP-0006 should focus on the orderer-side
clone gap found in EXP-0005.

Question:

```text
Can the learned orderer distill the search teacher well enough to beat the strict
random-inclusive envelope?
```

Fixed:

- Producer fixed to EXP-0004 v4.
- Search teacher from EXP-0005 Stage 2b.
- Held-out suite remains seeds 6,7,8,9 for continuity.

EXP-0006 has two phases:

1. EXP-0006A tests whether the clone gap is caused by missing orderer observation
   signal.
2. EXP-0006B tests model capacity/stochastic distillation only if observation
   enrichment does not close the gap.

Models:

- M2-v1: current accepted learned orderer.
- M2-v3-obs: deterministic orderer with enriched observation.
- M2-v3-wide: wider/deeper deterministic orderer MLP, only if M2-v3-obs fails.
- M2-v3-topo: topology-conditioned orderer, if a stable topology feature vector is
  available and M2-v3-obs is insufficient.
- M2-v3-stochastic: stochastic routing distillation, preserving the teacher's
  temporal diversification instead of collapsing it into one deterministic mean.

Primary gate:

```text
M2-v3 beats the strict envelope on held-out seeds 6,7,8,9, with zero death.
```

Secondary gates:

- beats M2-v1 on mean consumer completed units
- does not regress seed 9, the known hard held-out case
- improves or preserves consumer rejected units
- passes a small train-vs-held-out fidelity report against the teacher

## Planner Design

Start with the existing orderer search teacher because it already cleared the
envelope in EXP-0005 Stage 2b.

Then upgrade only if needed:

1. Existing sampled routing search.

   Keep as the baseline teacher. It is already reproducible and can produce labels.

2. CEM routing search.

   Replace raw random samples with a cross-entropy method loop:

   ```text
   sample routing logits -> evaluate rollouts -> keep elite samples -> refit sample
   distribution -> repeat -> commit best or distribution
   ```

   This is better matched to continuous routing logits than MCTS.

3. Value-guided search.

   Train a value model from rollout outcomes, then use:

   ```text
   short rollout score + value(state_after_lookahead)
   ```

   This reduces brute-force rollout cost and allows deeper effective planning.

4. Joint producer-orderer search.

   Only open this after the orderer clears its current envelope gap. Joint search is
   more expensive and mixes two problems.

## Distillation Design

The current masked-MSE clone is probably too weak for the envelope goal. It can
learn the deployed-baseline improvement, but EXP-0005 showed it loses teacher
behavior on the hardest held-out case.

Before treating this as pure network capacity, test whether the orderer observation
is missing the state needed to reproduce the search teacher. The current orderer
observation is intentionally sparse: per-order slot data, product identity, producer
cut-off prices, and cycle progress. It does not expose enough producer load/capacity
state to explain why the search teacher re-routes away from bottlenecks.

Use decode-aware losses:

- For deterministic routing targets, train only decode-active orderer dimensions.
- Normalize per order slot so irrelevant logit offsets do not dominate.
- Prefer KL/cross-entropy over eligible producers when the teacher can provide a
  distribution.
- Track teacher-action match and outcome match separately.

For stochastic distillation:

- Preserve a distribution over good routings, not only the single best routing.
- Evaluate the stochastic policy by repeated episode seeds and report CIs.
- Promote stochastic only if mean improvement clears the envelope without high tail
  risk.

## Value Model

Add a value head after EXP-0006A and the first orderer-capacity experiment, not
before.

Value targets:

- final consumer completed units
- death
- consumer rejected units
- optional auxiliary: cap minimum, replenishment rejects, investment churn

Use the value model for planner scoring only after it is calibrated against rollout
returns. A bad value model can make search worse, so it must pass an offline
prediction gate first.

Offline value gate:

```text
held-out value ranking agrees with rollout ranking on planner candidates
```

It does not need perfect absolute prediction. It needs to rank candidate actions
well enough to guide search.

## Experiment Ladder

### EXP-0006A: Orderer Observation Enrichment

Purpose:

```text
Test whether the EXP-0005 clone gap is caused by missing orderer state rather than
network capacity.
```

Rationale:

EXP-0005 Stage 2b showed the search teacher beats the strict envelope, while the
student loses seed 9 and only ties parts of the random-inclusive envelope. Stage 2c
showed more topology data makes this worse. Before moving to wider networks or
stochastic distillation, expose the public state the teacher appears to use:
producer load, makeability, order urgency, and product/economy context.

Do not expose future information. The observation may include current state and
history-derived statistics, but not final episode outcome, future demand, or planner
scores.

Implementation boundary:

These attributes should be added to the **domain model first**, not only computed
inside `env.encode_ordering`. The domain should own the authoritative current and
historical economy facts: routed quantities, completed quantities, rejected
quantities, producer/product throughput, latency, in-flight workload, and product
demand history. The env layer should only adapt that domain state into normalized
fixed-width observations. This keeps the statistics testable outside RL, usable by
white-box diagnostics/search, and prevents the observation encoder from becoming a
second hidden simulation model. See
`docs/adr/0051-orderer-observation-economy-metrics.md` for the canonical metric
definitions and formulas.

Candidate observation additions:

Order slot features:

- order age / TTL remaining per slot
- order type: consumer, replenishment, investment
- unresolved quantity and tokens per unit
- eligible producer count per slot
- explicit eligible-producer mask per slot

Producer features:

- producer total current capacity
- producer total max capacity
- producer capacity headroom
- producer min/mean capacity ratio across owned capacity types
- producer requested/reserved load
- producer cut-off price
- in-flight producer workload
- recent producer acceptance/completion rate
- recent producer rejection/failure rate

Product features:

- open demand quantity by product
- open demand tokens by product
- recent average cycle demand by product
- recent rejected quantity by product per cycle
- product depth/layer in the economy graph
- recipe complexity / number of inputs
- product eligible-producer count

Producer-product features:

- producer/product eligibility
- producer/product makeable quantity now
- producer/product bottleneck ratio
- per-slot/per-producer makeability for the current order quantity
- recent routed quantity by producer/product per cycle
- recent completed quantity by producer/product per cycle
- recent rejected quantity by producer/product per cycle
- recent producer/product success rate
- recent producer/product production throughput
- recent producer/product production duration or completion latency

Use recent EMA/windowed statistics rather than lifetime averages. Routing quality is
state-dependent, so long-term averages can lag the current bottleneck. Also keep
true demand product-level: `average demand per producer/product` is policy-dependent
if it means "what this producer was routed before", so record that as routed-load
history, not as demand.

Feature calculation guide:

Use these definitions as the canonical formulas. The **domain** should maintain the
raw values and recent-history EMAs; the env should normalize them for the observation
vector.

Notation:

- `p` = producer
- `x` = product
- `k` = order slot
- `t` = capacity type
- `g` = input product
- `C[p,t]` = producer `p` current capacity for type `t`
- `M[p,t]` = producer `p` max capacity for type `t`
- `R[x,t]` = per-unit capacity required by product `x`
- `I[x,g]` = per-unit input product `g` required by product `x`
- `Inv[p,g]` = producer `p` current inventory of input product `g`
- `q[k]` = unresolved quantity in order slot `k`
- `tok[k]` = available tokens in order slot `k`
- `eligible(p,x) = 1 if x in producer[p].produces else 0`
- `EMA_next = alpha * current_cycle_value + (1 - alpha) * EMA_prev`
- `ORDER_TTL` = maximum order age before rejection, currently `2` in the domain

Order slot formulas:

- `order_age = order.cycle_counter`
- `ttl_remaining = max(0, ORDER_TTL - order.cycle_counter)`
- `order_type = one_hot(consumer, replenishment, investment)` from the order request kind
- `unresolved_quantity = order.unresolved_quantity()`
- `tokens_per_unit = order.available_tokens() / max(1, unresolved_quantity)`
- `eligible_producer_count = sum_p eligible(p, order.product)`
- `eligible_mask[k,p] = eligible(p, order[k].product)`

Producer formulas:

- `producer_total_current_capacity[p] = sum_t C[p,t]`
- `producer_total_max_capacity[p] = sum_t M[p,t]`
- `producer_capacity_headroom[p] = sum_t max(0, M[p,t] - C[p,t])`
- `capacity_ratio[p,t] = C[p,t] / max(1, M[p,t])`
- `producer_min_capacity_ratio[p] = min_t capacity_ratio[p,t]` over owned types
- `producer_mean_capacity_ratio[p] = mean_t capacity_ratio[p,t]` over owned types
- `producer_requested_load[p] = producer[p].requested_capacity`
- `producer_reserved_load[p] = sum(reservation.bid.total_capacity for reservation in producer[p].reservations)`
- `producer_total_load[p] = producer_requested_load[p] + producer_reserved_load[p]`
- `producer_cut_off_price[p] = producer[p].cut_off_price`
- `producer_outstanding_inbound_units[p] = sum_g outstanding_inbound[p,g]`
- `producer_active_investments[p] = one_hot(restoration_running, upgrade_running, inventory_expansion_running)`
- `producer_in_flight_workload[p]` is the normalized tuple
  `(producer_reserved_load[p] / cap_ref,
  producer_outstanding_inbound_units[p] / inv_ref,
  producer_active_investments[p])`; do not add raw capacity units and inventory
  units together in the domain model
- `producer_acceptance_rate[p] = EMA(accepted_qty[p]) / max(eps, EMA(routed_qty[p]))`
- `producer_completion_rate[p] = EMA(completed_qty[p]) / max(eps, EMA(routed_qty[p]))`
- `producer_rejection_rate[p] = EMA(rejected_qty[p] + failed_qty[p]) / max(eps, EMA(routed_qty[p]))`

Product formulas:

- `open_demand_quantity[x] = sum unresolved_quantity(order)` over open consumer orders for `x`
- `open_demand_tokens[x] = sum available_tokens(order)` over open consumer orders for `x`
- `recent_average_cycle_demand[x] = EMA(new_consumer_demand_quantity_this_cycle[x])`
- `recent_rejected_quantity[x] = EMA(rejected_consumer_quantity_this_cycle[x])`
- `product_depth[x] = 0` for primal products; otherwise `1 + max(product_depth[input])`
- `recipe_input_count[x] = len(I[x,*])`
- `recipe_input_units[x] = sum_g I[x,g]`
- `recipe_capacity_type_count[x] = len(R[x,*])`
- `eligible_producer_count[x] = sum_p eligible(p,x)`

Producer-product formulas:

- `producer_product_eligible[p,x] = eligible(p,x)`
- `capacity_makeable_qty[p,x] = 0` if not eligible; otherwise `min_t floor(C[p,t] / R[x,t])`
  over capacity types required by `x`
- `input_makeable_qty[p,x] = infinity` if `x` has no inputs; otherwise
  `min_g floor(Inv[p,g] / I[x,g])` over input products required by `x`
- `producer_product_accept_qty_now[p,x] = capacity_makeable_qty[p,x]`; this is how
  much can clear the capacity auction now, even if composite inputs are not yet in
  stock and the bid would become a reservation
- `producer_product_complete_qty_now[p,x] = min(capacity_makeable_qty[p,x], input_makeable_qty[p,x])`;
  this is how much can complete immediately with current capacity and current input
  inventory
- `producer_product_makeable_qty_now[p,x]` should mean `complete_qty_now` unless an
  experiment explicitly wants `accept_qty_now`; record both if possible
- `capacity_bottleneck_ratio[p,x,q] = min_t C[p,t] / max(1, q * R[x,t])`
- `input_bottleneck_ratio[p,x,q] = 1.0` if `x` has no inputs; otherwise
  `min_g Inv[p,g] / max(1, q * I[x,g])`
- `producer_product_bottleneck_ratio[p,x,q] =
  min(capacity_bottleneck_ratio[p,x,q], input_bottleneck_ratio[p,x,q])`
- `slot_makeability[k,p] = min(q[k], producer_product_makeable_qty_now[p, order[k].product]) / max(1, q[k])`
- `recent_routed_qty[p,x] = EMA(quantity routed to producer p for product x this cycle)`
- `recent_completed_qty[p,x] = EMA(quantity completed by producer p for product x this cycle)`
- `recent_rejected_qty[p,x] = EMA(quantity rejected or failed by producer p for product x this cycle)`
- `recent_success_rate[p,x] = EMA(completed_qty[p,x]) /
  max(eps, EMA(completed_qty[p,x] + rejected_qty[p,x] + failed_qty[p,x]))`
- `recent_production_throughput[p,x] = EMA(completed_qty[p,x])`
- `recent_completion_latency[p,x] = quantity_weighted_EMA(completion_cycle - routed_cycle)`
- `recent_failure_latency[p,x] = quantity_weighted_EMA(failure_cycle - routed_cycle)`

Normalization for observations:

- capacity-like values use `/ cap_ref`
- quantity-like values use `/ inv_ref`
- token-like values use `/ token_ref`
- price-like values use `/ price_ref`
- ratios and rates stay in `[0, 1]` when naturally bounded; clip otherwise
- cycle/latency values use `/ ORDER_TTL` or `/ horizon`, depending on whether the
  measure is an order lifecycle value or an episode-progress value

Minimal first slice:

```text
order TTL + explicit eligibility mask + producer capacity/headroom/load summary +
producer/product makeability and bottleneck score + recent product demand EMA +
recent producer/product throughput and latency EMA
```

Work:

- add an enriched orderer observation version behind a checkpoint-schema bump
- keep producer observation unchanged
- keep producer fixed to EXP-0004 v4
- regenerate the Stage 2b teacher labels with the enriched observation
- train deterministic M2-v3-obs using the same DAgger/search setup as EXP-0005
- evaluate M2-v1, M2-v3-obs, the search teacher, and the full envelope on held-out
  seeds 6,7,8,9

Accept:

```text
M2-v3-obs beats the strict envelope on held-out seeds 6,7,8,9, with zero death.
```

Reject:

```text
the enriched-observation student still fails while the teacher clears the envelope.
```

Interpretation:

- If accepted, the blocker was observability. Continue expert iteration with the
  enriched observation as the new orderer contract.
- If rejected with low teacher-action fidelity, move to EXP-0006B capacity or
  stochastic distillation.
- If rejected despite high teacher-action fidelity, the teacher action labels may
  be outcome-sensitive or multimodal; prefer stochastic distillation before PPO.

### EXP-0006B: Orderer Distillation Capacity

Purpose:

```text
Beat the strict random-inclusive envelope after EXP-0006A if observation enrichment
alone is not enough.
```

Work:

- reuse the enriched observation if EXP-0006A improved fidelity
- train a wider/deeper deterministic orderer
- add topology-conditioned features if a stable topology vector exists
- train a stochastic orderer if deterministic still collapses routing diversity
- evaluate against M2-v1 and the full envelope

Accept:

```text
held-out envelope ratio > 1.00 on all seeds and mean >= 1.10, with zero death
```

Reject:

```text
teacher beats envelope but all student variants fail
```

If rejected, the bottleneck is representation/interface, likely requiring graph or
set-based orderer architecture.

### EXP-0007: Value-Guided Planner

Purpose:

```text
Reduce search cost and increase effective planning horizon.
```

Work:

- train value head from EXP-0006A/0006B rollout/search data
- validate candidate ranking quality
- use value-guided scoring in the planner

Accept:

```text
value-guided planner matches or beats rollout-only planner at lower compute
```

### EXP-0008: Iterative Policy Improvement

Purpose:

```text
Run multiple promote-and-relabel cycles from the current champion.
```

Work:

- collect states from champion rollouts
- relabel with planner
- train next candidate
- promote only on held-out improvement

Accept:

```text
at least two consecutive promoted generations improve the envelope-adjusted score
without increasing death
```

### EXP-0009: Joint Producer-Orderer Improvement

Purpose:

```text
Test whether producer-side headroom appears after the orderer is strong.
```

Work:

- keep current orderer champion fixed
- run producer search/teacher audit
- only train producer if search shows producer-side headroom

Accept:

```text
joint stack beats orderer-only champion by a meaningful margin
```

## Promotion Rules

A model becomes the new champion only if all are true:

- held-out evaluation uses topologies not used for label generation
- mean score beats current champion
- score beats the baseline envelope by the configured margin
- death rate stays zero or below an explicitly accepted threshold
- no known hard cell regresses below its prior accepted level
- result is reproducible from a saved checkpoint and recorded in an experiment dir

Recommended metrics:

- consumer completed units
- consumer completed orders
- consumer rejected units
- death rate
- cap minimum
- replenishment rejected units
- investment completed / churn

Use at least 20 episode seeds for final promotion. Cheap probes can use fewer.

## Compute Rules

Follow the repository training rule:

- cheap diagnostics and evals can run locally
- real training runs on Runpod through `run_pod_training.py`
- every accepted or rejected result gets an experiment directory under `experiments/`

Large runs must save:

- checkpoint
- summary CSV
- result document
- exact command/spec
- train/eval split
- teacher configuration

## What Not To Do Yet

Do not restart PPO from scratch. It already failed to discover the depth-3 producer
discipline and is not the current bottleneck.

Do not spend more on producer-only work until orderer-side envelope performance is
resolved.

Do not solve the strict envelope by adding more topology data alone. EXP-0005 Stage
2c already tested that and saw regression.

Do not claim "beats all algorithms" from a single fixed baseline. The benchmark must
use an envelope that includes the strongest known hand-written and simple randomized
policies.

## Plain-English Summary

The next paradigm is:

```text
Use search as the coach.
Train the neural agent to copy the coach.
Let the improved neural agent create new states.
Run the coach again on those states.
Repeat, and only promote models that beat the full baseline envelope.
```

The first concrete move is EXP-0006A: keep the v4 producer, enrich the orderer
observation with public producer/product/order state, and test whether the
already-strong search teacher can be distilled well enough to beat the strict
random-inclusive envelope. Only move to wider/stochastic orderer models if that
observability probe fails.
