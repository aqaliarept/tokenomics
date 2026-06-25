# Entity-factored producer planner/scorer contract

Status: proposed

## Context

The current producer actor is one agent per producer, which is the right control
boundary: a producer owns shared capacity, inventory, reservations, and investment
state. Those resources are coupled. For example, one CNC-like capacity type can be
shared by several products, so independent per-product producer agents would compete
for the same scarce machine and would need another coordinator to resolve conflicts.
That coordinator is the producer itself.

The current producer observation/action contract is still mostly flat:

```text
global capacity-type slots
global input-product slots
global product-demand slots
producer summary slots
investment toggles
```

This works for the current fixed-size experiments, but it makes training sensitive
to arbitrary product IDs, capacity-type IDs, slot positions, and topology-specific
vocabularies. The main producer-side goal is therefore not to split producer
control. It is to make the single producer controller see a structured production
problem in a form that is less dependent on product/capacity permutations.

The stronger long-term goal is stricter than slot invariance: train on small
topologies and run on much larger product/capacity sets. A larger flat neural
producer is not a credible way to get that behavior. The learned part should not be
responsible for discovering capacity conservation, inventory budgets, or investment
locks from examples. Those constraints should live in a deterministic planner. The
learned model, when introduced, should provide local economic scores that the
planner can use.

## Decision

Keep one producer agent per producer:

```text
prod_p4 = the orchestrator for producer p4
```

Replace or supplement the flat producer tensor with an entity-factored producer
state:

```text
producer global state
capacity-type rows
product rows
input-product rows
reservation/backlog rows
relation masks/edges between those rows
```

The domain model should own the raw entity view. The env layer should only normalize,
pad/mask, and encode it for RL.

Use that entity view as the input contract for a constrained producer planner:

```text
entity view -> local scores -> deterministic constrained planner -> ProducingAgentCommand
```

The planner owns global feasibility and coordination:

```text
capacity limits
inventory limits
funding limits
running investment locks
recipe feasibility
shared-resource contention between products
```

The learned component should be a local scorer, not a monolithic allocator. It may
learn scores such as product urgency, input value, capacity bottleneck value, and
investment ROI. The planner then turns those scores into valid replenishment and
investment commands.

This should be versioned and opt-in first:

```text
producer_obs_model = "flat_v1" | "entity_factored_v1"
producer_control_model = "v4_clone" | "entity_planner_v1" | "entity_planner_scorer_v1"
```

The goal is not to remove products or capacity types from state. The goal is to
remove dependence on arbitrary product/capacity IDs and reduce the number of
permutations the policy must memorize during training. For size extrapolation, the
goal is to make the algorithmic part correct for any number of rows, bounded by
compute and configured batching limits.

## Non-Goals

Do not introduce independent producer stream agents for each product or input.

Do not change production auction semantics in the first slice.

Do not train a new producer champion in the same experiment that introduces the
entity-factored contract.

Do not train a larger neural producer and expect it to learn global feasibility
rules from small-topology examples.

Do not use product IDs or capacity-type IDs as the primary signal. Stable IDs may
exist for diagnostics, but the learned policy should rely on row features and
relations.

## Producer Entity View

### Producer Global State

One row or vector for producer-level facts:

- producer id for diagnostics only,
- cycle progress,
- cut-off price,
- inventory units,
- inventory limit,
- free inventory headroom,
- total outstanding inbound units,
- restoration/upgrade/inventory-expansion running flags,
- available funding or token context if exposed later.

### Capacity-Type Rows

One row per capacity type relevant to this producer.

Notation:

- `p` = producer
- `t` = capacity type
- `x` = product
- `C[p,t]` = current capacity
- `M[p,t]` = maximum capacity
- `D[p,t]` = degradation this cycle
- `Res[p,t]` = capacity reserved by live reservations
- `ReqPrev[p,t]` = most recent post-auction requested capacity load for producer
  `p` and capacity type `t`
- `R[x,t]` = per-unit capacity requirement for product `x`
- `open_qty[x]` = unresolved consumer demand quantity for product `x`
- `eligible[p,x] = 1 if x in producer[p].produces else 0`
- `eps` = small denominator guard

Recommended raw fields:

```text
owned[p,t] = 1 if producer p owns capacity type t else 0
current_capacity[p,t] = C[p,t]
max_capacity[p,t] = M[p,t]
capacity_headroom[p,t] = max(0, M[p,t] - C[p,t])
capacity_ratio[p,t] = C[p,t] / max(eps, M[p,t])
degradation[p,t] = D[p,t]
reserved_capacity[p,t] = Res[p,t]
previous_requested_capacity[p,t] = ReqPrev[p,t]
free_capacity[p,t] = max(0, C[p,t] - Res[p,t])
reserved_ratio[p,t] = Res[p,t] / max(eps, C[p,t])
restore_effect[p,t] = capacity restored by the configured restoration if it targets t
restore_cost[p,t] = token cost of that restoration
upgrade_effect[p,t] = max-capacity increase if the configured upgrade targets t
upgrade_cost[p,t] = token cost of that upgrade
demand_pressure[p,t] = sum_x eligible[p,x] * open_qty[x] * R[x,t]
```

`ReqPrev[p,t]` should be added to the domain view as a per-type value. A scalar
`requested_capacity` is not enough once a producer owns multiple capacity types.
At the producer decision point, current-cycle orders have not been routed to
producers yet. This field must therefore reflect previous or most-recent observed
auction pressure, not future current-cycle bids.

### Product Rows

One row per product the producer can make or may be evaluated against.

Notation:

- `Inv[p,g]` = inventory of input product `g`
- `I[x,g]` = per-unit input requirement of product `x`

Recommended raw fields:

```text
eligible[p,x]
open_demand_quantity[x]
open_demand_tokens[x]
recipe_capacity_type_count[x]
recipe_input_count[x]
recipe_input_units[x]
product_depth[x]
active_reservation_qty[p,x]
recent_routed_qty[p,x]
recent_completed_qty[p,x]
recent_rejected_or_failed_qty[p,x]
recent_completion_latency[p,x]
```

Derived makeability fields:

```text
capacity_makeable_qty[p,x] =
  0 if not eligible[p,x]
  else min_t floor(free_capacity[p,t] / R[x,t]) over required capacity types

input_makeable_qty[p,x] =
  0 if not eligible[p,x]
  else infinity if x has no input requirements
  else min_g floor(Inv[p,g] / I[x,g]) over required input products

complete_now_qty[p,x] =
  0 if not eligible[p,x]
  else min(capacity_makeable_qty[p,x], input_makeable_qty[p,x])

capacity_bottleneck_ratio[p,x,q] =
  0 if not eligible[p,x]
  else min_t free_capacity[p,t] / max(1, q * R[x,t])

input_bottleneck_ratio[p,x,q] =
  0 if not eligible[p,x]
  else 1.0 if x has no input requirements
  else min_g Inv[p,g] / max(1, q * I[x,g])
```

### Input-Product Rows

One row per input product relevant to the producer.

Recommended raw fields:

```text
inventory[p,g] = Inv[p,g]
outstanding_inbound[p,g]
free_inventory_headroom[p] = inventory_limit[p] - inventory_units[p] - outstanding_inbound_units[p]
recent_replenishment_requested[p,g]
recent_replenishment_completed[p,g]
recent_replenishment_rejected[p,g]
recent_replenishment_latency[p,g]
blocked_reservation_qty[p,g] = quantity of live reservations waiting for g
downstream_demand_pressure[p,g] = sum_x eligible[p,x] * open_qty[x] * I[x,g]
```

### Reservation/Backlog Rows

Keep a bounded set of local reservation/backlog rows. These rows represent work the
producer already accepted or is considering:

```text
product
quantity
age
ttl_remaining
tokens_per_capacity
capacity requirement summary
missing input summary
reserved capacity by type relation
```

If the first implementation needs a smaller slice, reservation rows may be deferred
as long as equivalent aggregate fields remain in the product/input/capacity rows.

## Relations

Use relation masks or edge tensors instead of forcing every relation into one flat
feature vector:

```text
product_capacity_required[x,t] = R[x,t]
product_input_required[x,g] = I[x,g]
producer_owns_capacity[p,t]
producer_can_make_product[p,x]
product_uses_input[x,g]
```

These relations let the model learn from structure:

```text
products compete through shared capacity rows
products depend on input rows
capacity health affects all products that require that capacity type
```

## Planner And Scorer Contract

The first scalable controller should be a deterministic planner over the entity
view. It should be able to run without a learned scorer by using transparent
heuristic scores. That gives a debuggable baseline before introducing learning.

The planner may use algorithms such as:

```text
priority queues for backlog/input urgency
greedy scheduling with feasibility checks
bounded knapsack for inventory/funding allocation
min-cost or max-benefit flow where the domain shape allows it
```

When a learned scorer is introduced, it should use shared encoders over rows and
edges:

```text
capacity row encoder
product row encoder
input row encoder
reservation row encoder
local product-capacity edge scorer
local product-input edge scorer
masked pooling for aggregate context
```

The important property is permutation behavior:

- permuting capacity rows should permute capacity-related action outputs the same way,
- permuting product rows should not change producer-level decisions except through
  the corresponding permuted outputs,
- pooled producer decisions should be invariant to row order.

Padding and masks are acceptable for batching. The mask, not the numeric slot, should
define whether a row is real.

Dense all-product attention is not the default for size extrapolation. Prefer sparse
relations: a product should communicate primarily with the capacities and inputs it
actually requires. A policy trained on four products cannot be expected to learn
stable 400-product global coordination from dense interactions alone.

## Action Contract

Keep actions coordinated by the producer orchestrator.

Initial scorer outputs:

```text
product urgency score
input replenishment value score
capacity bottleneck value score
restoration ROI score over eligible capacity/investment targets
upgrade ROI score over eligible capacity/investment targets
inventory expansion value score
```

The planner consumes those scores and emits the existing domain command:

```python
ProducingAgentCommand(
    do_restoration=...,
    do_upgrade=...,
    do_inventory_expansion=...,
    replenishments={...},
)
```

The planner/decoder remains responsible for shared constraints:

- one inventory headroom budget,
- no invalid replenishment products,
- no already-running investment restart,
- no unowned capacity investment,
- no overcommitment caused by independent product decisions.

Future planner outputs may add product-priority, pricing, or acceptance/allocation
controls, but those should be separate experiments because they change producer
semantics.

## Training And Permutation Reduction

The entity-factored contract should reduce required training permutations by making
the architecture and planner respect entity structure. The target is not only
relabeling robustness; it is controlled size extrapolation.

Required gates:

- product relabeling gate: rename product IDs and verify equivalent behavior after
  unrenaming actions/metrics,
- capacity relabeling gate: rename capacity-type IDs and verify equivalent behavior,
- topology hold-out gate: do not promote on train-topology gains alone,
- row-order gate: shuffle entity rows and verify invariant/equivariant outputs,
- size-holdout gate: train on smaller topologies and evaluate on larger product and
  capacity counts, including explicit stress suites such as `8`, `16`, `64`, and
  `400` products where runtime permits.

Recommended training setup:

- clone `stock_aware` first through the new contract,
- implement a deterministic planner before training a learned producer scorer,
- train local scores only after the planner has behavior-parity and feasibility
  tests,
- use product/capacity relabeling augmentation during training,
- keep shared row encoders across all products and capacity types,
- avoid one learned actor per product or capacity type.

## Experiment Plan

### EXP-0010A: Entity-Factored Producer Entity View

Question:

```text
Can one producer orchestrator expose product/capacity/entity state while preserving
existing producer behavior and reducing product/capacity permutation dependence?
```

Method:

- add the domain-owned producer entity view,
- add opt-in env encoding and schema/fingerprint versioning,
- keep the domain producer command and production auction unchanged,
- replay/clone existing `stock_aware` and v4 behavior through the new contract,
- run permutation and behavior-parity tests.

Accept:

```text
baseline producer behavior preserved; row-order/product/capacity relabeling gates
pass; observation/action contract is versioned and checkpoint-incompatible with flat_v1
```

### EXP-0010B: Deterministic Producer Planner

Build the scalable controller without learning first.

Accept:

```text
planner preserves or improves v4 behavior, never violates producer constraints,
passes row-order/product/capacity relabeling gates, and runs on larger synthetic
product/capacity counts
```

### EXP-0010C: Learned Local Producer Scorer

Train local scores inside the deterministic planner instead of training a monolithic
producer allocator.

Accept:

```text
learned scorer improves the deterministic planner or v4 under the champion orderer
without strict-envelope regressions
```

### EXP-0010D: Producer Size-Holdout Audit

Only after EXP-0010B or EXP-0010C passes, test whether the controller actually
generalizes by size.

Accept:

```text
trained on small topologies; passes held-out evaluations on larger product/capacity
counts without relabeling, feasibility, or death regressions
```

Only after EXP-0010D should joint producer-orderer work be considered.

## Consequences

- The producer remains the coordination boundary.
- The tensor becomes more structured and likely larger.
- Producer improvement shifts from "train a better MLP" to "build a feasible
  planner and optionally learn local economic scores."
- Training can use permutation-aware local models instead of memorizing
  product/capacity slots.
- Domain metrics must become more per-type and per-product explicit.
- Checkpoints trained on the old flat producer contract must not load silently under
  the entity-factored contract.
