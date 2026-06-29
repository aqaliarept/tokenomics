# Product-Agnostic Architecture Notes

## Principle

Learned policies must not consume product-specific names, ids, or vocabulary
positions as input features. Product ids may exist inside a local dataset only as
internal keys used to construct relations, joins, labels, and audit fixtures. Once a
training or inference tensor is built, the policy should see only local relational
facts such as eligibility, backlog, capacity, latency, bottleneck ratios, prices, and
same-stream inventory state.

This keeps the intended transfer contract clear:

- product relabeling must not change observations or decisions except for equivalent
  row/edge permutations,
- larger product sets should not require a wider learned input unless they also add
  more simultaneously visible local relations,
- product names and product ids are dataset metadata, not model semantics.

## Current Coupling Findings

The current codebase has already removed several direct product-id scalar features,
especially in the newer product-stream orderer and producer entity view. The learned
PPO/DAgger actors are still not product/topology agnostic, though, because their
contracts are fixed-width tensors derived from the training config.

### Producer Coupling

The legacy flat producer contract is directly tied to the global topology:

- `EnvSpec.producing_obs_dim` depends on global capacity types, input products,
  producer count, and order-product vocabulary:
  `2*T + 2*G + 2*P + 2*Q + ...`.
- `encode_producing` fills those features by iterating sorted global vocabularies:
  capacity-type slots, input-product slots, producer-summary slots, and open-demand
  product slots.
- `producing_action_dim = G + 3`; `decode_producing` interprets the first `G`
  outputs as replenishment quantities by `spec.input_index`.

This means a learned producer trained with one input-product set cannot issue
replenishment actions for a different input-product set without a different output
head.

The entity producer contract is semantically better but still fixed-width at the env
tensor boundary:

- `producer_entity_obs_dim` pads to config-wide `num_types`, `num_order_products`,
  and `num_inputs`.
- relation planes are flattened as `Q*T` and `Q*G`.
- row order defaults to sorted product/capacity ids in the domain view. The encoder
  preserves row permutations correctly, but a plain MLP over the flattened tensor can
  still learn slot position unless the model architecture or training enforces
  permutation behavior.

The scalable producer path is not the flat actor. It is the existing
`ProducerEntityView -> ProducerScoreModel -> deterministic planner` path, where the
learned component is a shared row/edge scorer and the planner owns feasibility,
inventory, funding, recipe constraints, and investment locks.

### Orderer Coupling

Legacy orderer observations are product-vocabulary dependent:

- v1 uses a per-slot product one-hot of width `Q`.
- v2 enriched observations include the same product one-hot plus a per-slot
  producer-product grid.

Product-stream orderer v3 removes the product one-hot and is the right direction:
each agent serves a fixed `(source, product)` stream and sees product features once.
However, the current implementation still inherits a topology-width dependency
through global `K`:

```text
K = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)
```

where `max_rel` is the largest number of replenishable input products for any
producer. In product-stream mode that is too broad: every stream agent already
handles one product stream, so `K` should be bounded by one stream's own possible
concurrent orders (`ORDER_LIFETIME`), not by the producer's whole product/input
width. See "Orderer Contract Fix" for the corrected bound.

The orderer action remains `K * P` logits. This is acceptable only within a fixed
producer-count and fixed stream-backlog contract. Transfer across different producer
counts needs a row-wise producer scorer plus a deterministic masked allocator, not a
single fixed output layer.

### Training And Checkpoint Coupling

Current PPO and expert-iteration trainers build `GaussianActor(obs_dim, act_dim)`
from the first training config. Scale/domain randomization is explicitly
dimension-preserving. Validation rejects mismatched actor/critic dimensions.

Checkpoints store exact actor dimensions and a config fingerprint that includes
shape-determining structure and full economy content. That is useful for safety, but
it also means the current learned checkpoints are config-contract artifacts, not
portable product/topology-independent policies.

### Size Probe

With fixed depth and producer count, increasing only `products_per_layer` changes the
learned tensor contracts:

```text
products/layer | flat producer obs/act | entity producer obs | stream orderer obs/act
1              | 59 / 5                | 143                 | 436 / 72
2              | 69 / 7                | 237                 | 463 / 108
4              | 89 / 11               | 461                 | 517 / 180
8              | 129 / 19              | 1053                | 625 / 324
16             | 209 / 35              | 2813                | 841 / 612
```

A checkpoint trained at product width 4 cannot load into the actor heads needed for
product width 16. That is contract incompatibility, not merely policy failure.

Note the stream-orderer column still grows with product width even though
`_ordering_obs_dim_stream` has no direct `Q` term. The growth is entirely
indirect, through `k`: in this probe `k` runs `6, 9, 15, 27, 51` as width runs
`1, 2, 4, 8, 16`, because `k` depends on `max_rel` (see the `k` formula above and
the "Orderer Contract Fix"). So the ADR-0052 goal "stream obs scales with producers,
not products" does not hold in practice until `K_stream` is corrected. The stream
action `K * P` inherits the same growth.

## Plain-English Version

The current learned actors are like spreadsheets with fixed columns.

Example:

```text
column 1 = product 10 inventory
column 2 = product 11 inventory
column 3 = product 20 inventory
```

If a new topology has different products, or twice as many products, the spreadsheet
columns no longer mean the same thing. A neural net trained on the old columns cannot
reliably understand the new spreadsheet. If the column count changes, the checkpoint
cannot even load.

The product-agnostic approach is to give the model a list of entity rows instead:

```text
input row:
  inventory = 2
  outstanding inbound = 1
  downstream demand = 5
  blocked reservations = 3
```

The model should not care whether that row is product 10, product 44, or product
9001. It should learn what the row means from the row's facts.

### Producer Example

Current flat producer actor:

```text
model output dim 0 -> replenish product 10
model output dim 1 -> replenish product 11
model output dim 2 -> replenish product 20
```

This does not transfer. A producer trained with products `[10, 11, 20]` has no
output for a new input product `77`.

Better producer pattern:

```text
for each input product row:
  learned scorer says "how valuable is replenishing this input?"

deterministic planner:
  sorts/scales those scores
  respects inventory headroom
  respects funding
  respects max replenish
  emits ProducingAgentCommand(replenishments={...})
```

Then the same learned scorer can run on 2 input rows, 20 input rows, or 200 input
rows. The planner owns the hard constraints; the model only supplies reusable local
judgment.

### Orderer Example

Current orderer actor:

```text
action = K backlog slots * P producers
```

If `K` or `P` changes, the output layer changes. If product-stream `K` grows because
a producer has more input products, a stream orderer trained on a small product set
cannot load for the larger one.

Better orderer pattern:

```text
for each eligible producer row for this product stream:
  learned scorer says "how good is routing this order to this producer?"

masked allocator:
  ignores ineligible producers
  softmaxes or ranks eligible producer scores
  converts scores into integer quantities and tokens
  emits SubOrder rows
```

Then one orderer scorer can handle 3 producers or 30 producers because it is reused
per producer row. The action is no longer "output slot 17 means producer p4"; it is
"score this runtime producer candidate."

### Checkpoint Example

A fixed-contract checkpoint says:

```text
producer input width = 89
producer output width = 11
orderer input width = 517
orderer output width = 180
```

That checkpoint is tied to the training topology.

A portable checkpoint should instead say:

```text
product row fields = demand, recipe shape, makeability, history
input row fields = inventory, inbound, blocked, downstream pressure
producer row fields = capacity, price, load, recent success
decoder = masked order allocation v1
planner = producer planner v1
```

That checkpoint can be loaded on a different topology as long as the same row fields
and decoder semantics exist.

## Common Solution Pattern

The common way to handle dynamic product/topology dimensions is to stop making the
neural model own environment cardinality. The learned part should operate on
runtime entity rows or graph nodes with shared weights, then a deterministic decoder
or planner should turn scores into valid domain commands.

### Model Pattern Ideas

#### Deep Sets / Masked Pooling

Idea:

```text
apply the same small neural network to every row
pool real rows into a summary
ignore padded rows with a mask
```

Example:

```text
input rows:
  row A: inventory=0, inbound=0, downstream_demand=6
  row B: inventory=5, inbound=2, downstream_demand=1
  row C: inventory=1, inbound=0, downstream_demand=4

shared row encoder:
  score row A
  score row B
  score row C

pool:
  max demand pressure
  mean stock shortage
  sum blocked reservations
```

Why it helps:

The same row encoder works whether there are 3 rows or 300 rows. Product ids and
row positions stop being the main signal.

How to use it here:

- producer input rows: score each input product's replenishment value,
- producer product rows: score each output product's urgency,
- orderer producer rows: score each eligible producer for a product stream.

This is the simplest first portable architecture. It should be the default before
using heavier attention or graph models.

#### Entity Attention / Set Transformer

Idea:

```text
rows are still rows
but each row can look at other rows before scoring
```

Example:

```text
product row 30 attends to:
  its input product rows 11 and 20
  the capacity rows needed by its recipe
  current inventory and active reservations
```

Why it helps:

Some decisions depend on comparisons. For example, a producer may have two input
products with shortages, but only one inventory headroom budget. Attention can help
the model compare rows without hard-coding a product slot order.

How to use it here:

- producer scorer: let product rows compare bottlenecks before the planner chooses
  replenishments,
- orderer scorer: let producer rows compare price, capacity, recent failures, and
  backlog pressure before routing,
- keep masks explicit so padded rows cannot affect attention.

Attention is useful after the simple row-scorer path is working. It should not be
used as a substitute for deterministic feasibility rules.

#### Graph Or Relational Networks

Idea:

Represent the economy as nodes and edges:

```text
nodes:
  producer p4
  product 30
  input product 11
  capacity type capacity-4
  order o123

edges:
  p4 produces 30
  30 consumes 11
  30 requires capacity-4
  o123 requests 30
  p2 eligible_for 11
```

The model passes messages along real edges instead of flattening every possible
product-capacity pair.

Why it helps:

The topology can grow without creating a huge dense tensor. A product only needs to
communicate with the inputs and capacities it actually uses.

How to use it here:

- producer: use existing product-capacity and product-input relations from
  `ProducerEntityView`,
- orderer: build relation rows from `ProductStreamView`: stream product, eligible
  producers, producer-product makeability, active backlog,
- use sparse edges rather than `Q*T` and `Q*G` dense planes for portable training.

This is the best long-term shape for arbitrary product graphs, especially when
recipes stop being simple layered chains.

#### Local Scorer Plus Deterministic Planner

Idea:

```text
learned model = judgment
planner/decoder = rules and feasibility
```

Example:

```text
model scores:
  replenish input 11: 0.9
  replenish input 20: 0.4
  restore capacity-4: 0.8

planner:
  sees only 3 units of inventory headroom
  sees restoration already running or not
  sees available funding
  emits a valid ProducingAgentCommand
```

Why it helps:

The model does not need to rediscover inventory conservation, funding limits,
integer rounding, or "do not restart an already-running restoration" from examples.
Those constraints are stable software logic, not learned behavior.

How to use it here:

- producer: this is already the direction of `ProducerScoreModel` and
  `plan_producer_command`,
- orderer: add an orderer scorer, then keep integer quantity/token allocation in a
  deterministic masked allocator,
- checkpoints should save scorer schema and planner version separately.

This is the most important pattern for this project.

### Action Pattern Ideas

#### Candidate Scoring

Idea:

```text
build runtime candidates
score candidates with shared weights
select or allocate among valid candidates
```

Producer example:

```text
candidates:
  replenish input 11
  replenish input 20
  restore capacity-4
  upgrade capacity-4

model scores candidates
planner emits valid command
```

Orderer example:

```text
candidates for order o123:
  route to producer p1
  route to producer p2
  route to producer p7

model scores each candidate
allocator splits quantity and tokens
```

How to use it here:

- replace producer output dimension `G + 3` with candidate rows plus planner logic,
- replace orderer output dimension `K * P` with producer candidate scores per active
  stream/order,
- keep candidate masks in training loss and inference decode.

#### Pointer-Style Selection

Idea:

Instead of outputting "dimension 17", the model points at one current row.

Example:

```text
current producer rows:
  row 0 = p1
  row 1 = p2
  row 2 = p7

model chooses row 1
decoder maps row 1 -> producer p2
```

Why it helps:

The output meaning is tied to the runtime row, not a checkpoint-time slot.

How to use it here:

- orderer: choose the best producer row for a stream when a single-winner routing
  mode is acceptable,
- producer: choose the most urgent input row to replenish when headroom is scarce,
- use this mainly for ranking/selection; allocation still needs a decoder.

#### Parameterized Actions

Idea:

Choose an action kind first, then predict parameters for that action.

Example:

```text
action kind = replenish
target row = input 11
parameter = quantity 3
```

Another example:

```text
action kind = route
target row = producer p2
parameter = share 0.35
```

How to use it here:

- producer: action kind can be `replenish`, `restore`, `upgrade`, or
  `inventory_expand`,
- orderer: action kind can be `route_to_producer` with a quantity/share parameter,
- decoder must still clamp parameters to inventory, funding, backlog, and eligibility
  constraints.

This can be useful later, but candidate scoring plus deterministic allocation is a
simpler first slice.

#### Autoregressive Allocation

Idea:

Allocate one valid piece at a time:

```text
step 1: send 2 units of order o123 to p2
step 2: send 1 unit of order o123 to p7
step 3: stop
```

Why it helps:

It can represent complex variable-size allocations without a fixed output vector.

Tradeoff:

It is slower, harder to train, and introduces more sequence decisions. It should be
reserved for cases where one-shot candidate scoring cannot express the needed
allocation.

How to use it here:

- not needed for the first orderer refactor,
- possible future route if orders need multi-step combinatorial planning across many
  products, producers, and funding constraints.

#### Invalid-Action Masking

Idea:

The model should not learn about invalid choices as if they were normal output
dimensions. Invalid choices should be masked before loss, sampling, ranking, or
allocation.

Example:

```text
order product = 30
eligible producers = p1, p4
ineligible producers = p2, p3, p5

masked allocator only scores/normalizes p1 and p4
```

How to use it here:

- producer: mask input rows that this producer cannot replenish,
- producer: mask investments that are already running or unconfigured,
- orderer: mask producers that are not eligible for the stream product,
- training loss should ignore masked candidates so gradients match real behavior.

The existing action-mask work in PPO is useful, but portable models should mask
runtime candidates rather than fixed no-op dimensions.

### Project Application Plan

Short-term:

- keep flat `GaussianActor(obs_dim, act_dim)` checkpoints as legacy fixed-contract
  artifacts,
- fix product-stream `K` so stream orderer shape no longer grows with producer
  input-product width,
- train/evaluate producer scorer through `ProducerEntityView` and
  `plan_producer_command`, not through flattened producer `Box` observations.

Medium-term:

- add `OrdererScoreModel` over `ProductStreamView`,
- produce per-candidate scores for eligible producers,
- decode those scores into existing `SubOrder` allocations,
- evaluate on same-size relabeling, then producer-count and product-count holdouts.

Long-term:

- move portable training batches from padded env `Box` tensors to ragged entity or
  graph batches,
- make full config fingerprint audit metadata for transfer reports,
- load portable checkpoints by feature schema and decoder version, not by topology
  size.

Portable checkpoint schema should save:

- feature schema and feature names by entity type,
- node/row types and edge/relation types,
- normalization version and normalization stats by feature name, not slot index,
- decoder/planner version and action semantics,
- model hidden sizes and architecture version.

For portable checkpoints, the full economy/config fingerprint should become audit
metadata for transfer reporting, not the primary load gate. Legacy flat checkpoints
should keep exact dimension and fingerprint guards.

Reference patterns:

- Deep Sets: https://arxiv.org/abs/1703.06114
- Set Transformer: https://arxiv.org/abs/1810.00825
- Graph Networks: https://arxiv.org/abs/1806.01261
- Relational Deep RL: https://arxiv.org/abs/1806.01830
- Pointer Networks: https://arxiv.org/abs/1506.03134
- Invalid Action Masking: https://arxiv.org/abs/2006.14171
- Action Branching: https://arxiv.org/abs/1711.08946
- Parameterized Action Space RL: https://arxiv.org/abs/1511.04143

## Versatility Verification Design

Before replacing the fixed-width actors, verify that the new approach is versatile
enough to justify implementation. The goal is to separate four failure modes:

```text
contract failure      = model/checkpoint cannot run on the target topology
equivariance failure  = relabeling or row shuffling changes the decision incorrectly
policy failure        = model runs but makes poor economic decisions
scaling failure       = model runs and behaves correctly but is too slow or memory-heavy
```

The verification infra should make those failures visible separately.

### Existing Verification Assets To Reuse

The repo already has useful verification examples with different topologies,
product sets, relabelings, and held-out seeds. They should be reused instead of
starting from scratch.

Existing assets:

- `tokenomics.eval.scale.build_scale_config`: creates varied scale topologies by
  seed. Capacities, degradation, costs, demand bounds, and strong/weak producers
  vary across seeds.
- `tokenomics.eval.scale.scale_topology_set`: adds product and producer slot
  permutations within the same shape. This is useful for detecting slot
  memorization.
- `tokenomics.eval.generalization`: applies controlled perturbations such as higher
  degradation, unseen demand quantities, leaner producers, and higher costs.
- `tokenomics.eval.held_out`: stores reproducible held-out manifests and release
  gates against baselines.
- `tests/fixtures/*.json`: includes non-identical authored configs such as
  `composite`, `complex`, `competition`, and `random_demand`.
- `tests/test_product_stream_ordering.py`: already checks product relabeling and
  producer-row permutation behavior for product-stream orderers.
- `tests/test_producer_view.py`, `tests/test_producer_planner.py`, and
  `tests/test_producer_scorer.py`: already check entity row behavior, relabeling,
  planner invariants, and scorer row-order invariance.
- `tokenomics.eval.producer_size_holdout`: already audits the deterministic entity
  planner over larger product counts and separates quality from runtime/memory
  scaling.
- `tokenomics.eval.producer_champion_audit.audit_producer_size_transfer`: already
  distinguishes fixed-checkpoint compatibility from entity-planner runability by
  size.

What those assets already prove:

- the simulator can generate different economies and held-out topology seeds,
- several baseline and planner paths run on different product sets,
- relabeling and row-order gates exist for parts of the newer entity contracts,
- size-holdout audits already exist for deterministic producer planning.

What they mostly do **not** prove yet:

- a learned checkpoint can load across different product counts or producer counts,
- a learned orderer can handle variable `P` without a fixed `K * P` output head,
- a learned producer can handle variable input-product count without a fixed `G + 3`
  output head,
- portable checkpoint loading by feature schema rather than full config fingerprint,
- out-of-family recipe graph transfer for learned scorer/allocator models.

So the new verification infra should extend the existing examples, not replace
them. The main addition is to turn "different topology, same dimensions" checks into
"different topology and different dimensions, same portable schema" checks.

### Evaluation-First Dynamic Approach

Do not try to make old learned producer/orderer checkpoints portable. They are
fixed-contract artifacts by construction:

```text
flat producer checkpoint  -> fixed obs_dim and fixed G + 3 action head
flat/orderer checkpoint   -> fixed obs_dim and fixed K * P action head
```

That is enough to analytically know they are not compatible with arbitrary product
sets, producer counts, capacity sets, or topology sizes. Keep them as legacy
same-shape references only.

The evaluation-first plan should instead start from the theoretically dynamic
approach:

```text
producer = ProducerEntityView -> row/edge scorer -> deterministic planner
orderer  = ProductStreamView  -> candidate scorer -> masked allocator
```

Then compare this dynamic approach against baselines on increasingly different
topologies. Continue improving it only if it wins on both axes:

```text
versatility = same checkpoint/schema runs across different product and producer counts
value       = policy quality beats or matches practical baselines
```

Old learned actors remain useful for two things:

- proving the fixed-dimension limitation on paper or by a simple spec-dim check,
- acting as optional same-shape champion bars where their tensor contract is
  compatible.

They should not be a blocker or target for the dynamic implementation.

#### Models To Compare First

Evaluate dynamic candidates and baselines as separate rows:

- deterministic `entity_planner`,
- `ProducerScoreModel + plan_producer_command` when a scorer exists,
- future portable producer scorer checkpoints,
- future portable `OrdererScoreModel + masked allocator`,
- deterministic orderer baselines: `proportional`, `even_split`, `obs_price`,
- practical stacks: `stock_aware + proportional`,
  `entity_planner + proportional`, and later
  `portable_producer + portable_orderer`,
- current learned champions only on shape-compatible cells.

This gives the useful reference classes:

```text
fixed learned actor      = legacy same-shape reference only
deterministic planner    = dynamic baseline
portable scorer/planner  = target learned architecture
```

#### Target Topology Matrix

Use a target matrix that increases distance from the training environment one axis at
a time:

```text
A. same shape, relabeled products/capacities/producers
B. same shape, different numeric economics
C. same shape, different authored product set / recipe graph
D. larger product count, same producer count
E. larger producer count, same product count
F. larger product count and larger producer count
G. deeper supply chain
H. out-of-family recipes: fork/join, multi-input products, shared capacities
I. scarcity stress: low headroom, low funding, high degradation, high backlog
```

For each target, record whether the dynamic producer contract, dynamic orderer
contract, and baseline stacks are compatible.

Example transfer table:

```text
model stack                    train topology   target topology   loads   runs   CU   dead
entity_planner + proportional  none             width 16 depth 3  yes     yes    ...  ...
portable producer + prop       width 4 depth 3  width 16 depth 3  yes     yes    ...  ...
portable producer + orderer    width 4 depth 3  width 16 depth 3  yes     yes    ...  ...
legacy v4 + legacy m2          width 4 depth 3  width 16 depth 3  no      no     n/a  n/a
```

#### Producer-Only Transfer Audit

Hold the orderer fixed and vary only the producer side. The producer under test
should be the deterministic planner or the new portable scorer/planner path, not the
legacy flat actor except as an optional same-shape reference.

Suggested pairs:

```text
producer = entity_planner               orderer = proportional or champion
producer = portable scorer + planner    orderer = proportional or champion
producer = legacy fixed actor           orderer = proportional or champion, same-shape only
```

Targets:

- same-shape relabeling,
- same-shape different economics,
- larger product/input count,
- larger capacity-type count,
- out-of-family recipes,
- scarcity stress.

Report:

- schema/load compatibility for portable scorer checkpoints,
- row/candidate compatibility with runtime input rows,
- invalid command rate,
- consumer completed units,
- death rate,
- replenishment and investment behavior,
- transfer ratio versus deterministic `entity_planner` and `stock_aware`.

Expected result:

- deterministic `entity_planner` should run across sizes,
- portable scorer/planner should load across sizes when feature schema matches,
- legacy flat learned producers should be marked `legacy incompatible` on shape
  changes and excluded from dynamic promotion decisions.

#### Orderer-Only Transfer Audit

Hold the producer fixed and vary only the orderer side. The orderer under test should
be the new candidate scorer/allocator path once available. Existing learned orderers
are same-shape references only.

Suggested pairs:

```text
producer = stock_aware or entity_planner   orderer = proportional
producer = stock_aware or entity_planner   orderer = portable candidate scorer/allocator
producer = stock_aware or entity_planner   orderer = legacy product-stream champion, same-shape only
```

Targets:

- same-shape product relabeling,
- producer row permutation,
- larger product count with same producer count,
- larger producer count,
- corrected `K_stream` vs legacy `K`,
- out-of-family recipes with different eligibility patterns.

Report:

- schema/load compatibility for portable orderer checkpoints,
- candidate compatibility with runtime producer rows,
- eligibility-mask violations,
- sub-order conservation,
- consumer completed units,
- death rate,
- ratio versus deterministic `proportional` and current champion where compatible.

Expected result:

- corrected product-stream `K_stream` removes product-count leakage for fixed `P`,
- portable candidate scorer/allocator should load across variable `P`,
- legacy product-stream checkpoints remain same-shape references only.

#### Joint Transfer Audit

After producer-only and orderer-only audits, evaluate the dynamic stack.

Targets:

- same-shape relabeling,
- same-shape held-out scale seeds,
- larger product count,
- larger producer count,
- larger product and producer count together.

Report every missing row explicitly:

```text
not run: portable schema incompatible
not run: decoder/planner version incompatible
run: compatible, policy quality measured
legacy reference omitted: fixed actor shape incompatible
```

This avoids mixing legacy fixed-contract failures with dynamic-model quality.

#### Evaluation Output

The first deliverable should be a compatibility and behavior matrix, not a new
training run.

Minimum CSV columns:

```text
model_id
producer_kind
orderer_kind
train_topology
target_topology
target_products
target_producers
target_capacity_types
target_depth
producer_schema_loads
orderer_schema_loads
producer_contract_compatible
orderer_contract_compatible
runs_episode
failure_class
consumer_completed_units
dead_rate
consumer_rejected_units
elapsed_sec
peak_memory_kib
baseline_ratio
notes
```

This matrix tells which problem to improve next:

```text
schema failures -> fix portable schema/actions
few contract failures but poor quality -> prioritize model/scorer training
good quality but slow -> prioritize batching/sparse graph efficiency
relabel failures -> prioritize masks, row encoders, and augmentation
```

#### Baseline Requirements

Every evaluation target should include baseline rows. A learned model result without
a baseline on the same target topology is hard to interpret.

Baseline classes:

- **No-op / weak floor:** proves the target is nontrivial and catches accidental
  reward inflation.
- **Simple deterministic producer baselines:** `stock_aware`, `restore_when_low`,
  and `entity_planner`.
- **Simple deterministic orderer baselines:** `even_split`, `proportional`, and
  `obs_price`.
- **Current champion stack:** existing best learned producer/orderer where the
  contract is shape-compatible.
- **Oracle/search teacher where affordable:** useful as an upper bar on small
  evaluation cells.

Run baselines under the same target matrix as learned models:

```text
same relabeling
same product count
same producer count
same demand seed
same horizon
same product-stream/legacy mode
same corrected-K or legacy-K contract
```

Report at least three bars for each target:

```text
floor baseline       = no-op or weak deterministic policy
practical baseline   = stock_aware + proportional / entity_planner + proportional
best available bar   = current champion or search teacher when compatible/affordable
```

For incompatible learned checkpoints, still run compatible baselines. That tells us
whether the target topology itself is viable.

Example:

```text
target = width 16, depth 3
learned v4 producer + learned stream orderer = contract failure
entity_planner + proportional                = runs, 740 CU, 0 death
stock_aware + proportional                   = runs, 690 CU, 0 death
```

This means the target is feasible; the learned stack failed because of its contract,
not because the economy is impossible.

Baseline columns to add to reports:

```text
baseline_family
baseline_model_id
baseline_consumer_completed_units
baseline_dead_rate
ratio_vs_practical_baseline
ratio_vs_best_available_bar
baseline_contract_compatible
```

### Reusable Test Fixtures

Build a small family of topology generators instead of testing only one scale
family:

- **same-shape relabel fixtures:** same economy, renamed products/capacities and
  permuted producers. These catch slot memorization.
- **size holdout fixtures:** same recipe pattern, larger product count, larger input
  count, and optionally larger producer count. These catch fixed tensor/action
  contracts.
- **out-of-family fixtures:** different recipe graph shape, for example fork/join
  recipes, products with multiple inputs, products sharing capacity types, and
  competing producers with uneven strengths. These catch overfitting to layered
  scale topology.
- **scarcity fixtures:** low inventory headroom, low funding, degraded capacity,
  high backlog, and already-running investments. These catch planner/decoder
  feasibility mistakes.
- **degenerate fixtures:** zero eligible alternatives, one eligible producer, no
  current backlog, no input products, no inventory expansion, and all rows padded
  except one. These catch mask and empty-row bugs.

Each fixture should expose both the raw domain view and the encoded/training view so
tests can check whether a failure is in domain facts, encoding, model scoring, or
decoding.

### Contract And Schema Gates

Portable checkpoints should be load-tested against many topologies before policy
quality is measured.

Gate examples:

```text
train on width 4 -> load on width 1, 2, 8, 16
train on 12 producers -> load on 3, 8, 20 producers
train on depth 3 -> load on depth 1, 2, 4, 5
```

Expected result:

- legacy flat checkpoints fail clearly with dimension mismatch,
- portable checkpoints load when feature schema and decoder version match,
- portable checkpoints reject only real schema/semantics changes, not product count
  changes.

Checkpoint metadata to assert:

- feature names by row/entity type,
- row/entity type set,
- relation/edge type set,
- normalization version,
- decoder/planner version,
- model architecture version.

The config/economy fingerprint should be recorded in transfer reports, but it
should not block portable checkpoint loading by itself.

### Invariance And Equivariance Gates

These are local, cheap tests that should run before any long training job.

Producer gates:

- product relabeling: after unmapping product ids in the command, replenishment
  decisions are equivalent,
- capacity relabeling: restoration/upgrade decisions are equivalent,
- input row permutation: scores move with rows, pooled producer-level decisions stay
  unchanged,
- padded row invariance: adding masked padded rows does not change scores or
  commands,
- tie behavior: equal-score candidates are handled without hidden product-id
  preference unless a deterministic documented tie-breaker is required.

Orderer gates:

- product relabeling: product-stream observations/scores are identical or
  equivalent after unmapping,
- producer row permutation: candidate scores and routed sub-orders move with the
  producer rows,
- eligibility masking: ineligible producers receive no probability, no gradient, and
  no routed quantity,
- single-eligible case: allocator emits the only valid route without requiring a
  learned distinction,
- empty-backlog case: scorer/allocator emits no-op commands without relying on a
  fixed padded slot.

Combined-stack gates:

- producer relabel plus orderer relabel in the same episode preserves episode
  metrics within deterministic tolerance,
- row shuffling during inference does not change aggregate metrics after unmapping,
- stochastic policies remain seed-reproducible under row permutations when the same
  logical candidates are present.

### Feasibility And Conservation Gates

The learned model should never be trusted to enforce domain constraints alone.
Every candidate decoder and planner should have invariant tests:

- no replenishment for products the producer cannot order,
- no over-allocation beyond inventory headroom,
- no spending beyond funding budget when a budget is supplied,
- no restart of already-running restoration, upgrade, or inventory expansion,
- no routed order to an ineligible producer,
- sub-order quantities sum exactly to unresolved quantity,
- sub-order tokens sum exactly to available tokens,
- zero-quantity sub-orders are dropped,
- all live orders are addressed or explicitly deferred by documented logic.

These tests should run with random score tensors too. Random scores are good at
finding decoder bugs because they do not hide behind a reasonable model.

### Quality Metrics

Report quality and contract compatibility separately.

Contract metrics:

```text
loaded = true/false
schema_match = true/false
rows_seen = count by entity type
masked_candidates = count
```

Policy metrics:

```text
consumer_completed_units
dead_rate
consumer_rejected_units
investment_completed
replenishment_completed
inventory_holding_units
```

Robustness metrics:

```text
mean / worst over topology seeds
min transfer ratio vs baseline
performance drop from train-size to holdout-size
row-permutation mismatch rate
product-relabel mismatch rate
```

Scaling metrics:

```text
elapsed seconds per episode
peak memory
rows per decision
candidate scores per decision
allocator calls per episode
```

Every report should show whether a missing result is a contract incompatibility or a
policy-quality failure.

### Producer Experiment Ladder

P-1. Dynamic producer baseline comparison:

- train: none for deterministic baselines; optional supervised scorer if already
  available,
- evaluate `entity_planner` and the portable scorer/planner path on same-shape
  relabels, same-shape numeric perturbations, larger product/input counts, larger
  capacity-type counts, and out-of-family recipes,
- hold orderer fixed to `proportional` or another deterministic orderer baseline,
- include legacy learned producers only on shape-compatible cells as reference bars,
- pass: the dynamic producer path loads/runs across target sizes and reports value
  against deterministic baselines; legacy incompatibility is recorded but does not
  block promotion.

P0. Deterministic planner parity:

- controller: `entity_planner`,
- train: none,
- eval: same-shape relabel, row permutation, size holdout,
- pass: feasible commands, relabel gates pass, no major regression vs current
  deterministic baseline.

P1. Supervised local scorer clone: **DONE** — contract (local TDD) + value + producer-count
transfer (EXP-0013). v2 gate-head distill (seeds 0–5, depth 3) reproduces the
`entity_planner` on held-out 6–9 at depths 1/2/3 (P=4/8/12): mean ratio **1.0029**, min
**1.0000**, zero death, investment false-positive rates ≈0; the depth-3 checkpoint runs
unchanged at the unseen P=4/8. Replenishment is a learned regression; investment decisions
(upgrade, inventory-expansion) are learned **binary gates** — logits BCE-trained against
the planner's gate booleans, fired by per-gate thresholds calibrated on command agreement
(saved in checkpoint v2). Restoration stays planner-owned. The v1 MSE-ROI heads
PARTIAL-failed at depth-3 (0.773) by reading regression noise as hard spend toggles; the
gate heads + `producer_command_agreement` FP eval fixed and proved it. ADR-0053.

- train scorer on small topologies using deterministic planner labels,
- evaluate scorer+planner on same size, relabeled same size, and larger sizes,
- pass: checkpoint loads across sizes; decoded commands are close to planner labels;
  no feasibility failures.

P2. Out-of-family producer scorer:

- train on layered scale topology,
- eval on fork/join recipes, shared capacities, multi-input products, uneven producer
  strengths,
- pass: degradation is graceful and attributable to policy quality, not schema or
  decoder incompatibility.

P3. Stress and scarcity:

- eval low headroom, low funding, high backlog, running investments,
- pass: no invalid commands; quality remains above simple stock-aware fallback on
  most cells.

### Orderer Experiment Ladder

O-1. Dynamic orderer baseline comparison:

- train: none for deterministic baselines; optional candidate scorer if already
  available,
- evaluate deterministic orderers and the portable candidate scorer/allocator path
  on same-shape relabels, producer permutations, larger product counts, larger
  producer counts, and out-of-family eligibility patterns,
- hold producer fixed to `stock_aware` or `entity_planner`,
- include legacy product-stream orderers only on shape-compatible cells as reference
  bars,
- pass: the dynamic orderer path loads/runs across target sizes and reports value
  against deterministic baselines; legacy incompatibility is recorded but does not
  block promotion.

O0. Corrected product-stream contract: **DONE** (ADR-0054).

- implement corrected `K_stream`, — `EnvSpec.k_stream = ORDER_LIFETIME +
  INVESTMENT_STREAMS` at `orderer_obs_version = 4` (covers a (source, product) stream's
  replenishment + shared-product investment orders; a constant, so still product-width
  independent — review fix, see ADR-0054 Correction), selected by
  `TokenomicsAECEnv(product_stream_ordering=True, corrected_stream_k=True)`,
- keep legacy v3 loadable, — default stays v3; `k_stream == k` for v1/v2/v3 so they
  are byte-identical,
- pass: orderer obs/action dimensions do not change when only product vocabulary
  width changes. — verified in `tests/test_product_stream_ordering.py`
  (`test_v4_*`); all Acceptance Gates green. Fresh v4 orderer artifact still owed
  (O2/O3).

O1. Deterministic candidate allocator: **DONE** (ADR-0054).

- input: hand-authored producer candidate scores, — `env.allocator.allocate_suborders`
  takes one runtime score per producer row + an eligibility mask (variable `P`, no
  fixed `K*P` head),
- decoder: masked allocator to `SubOrder`s, — owns eligibility mask, softmax over
  eligible rows, largest-remainder integer split, token apportionment; `decode_ordering`
  refactored onto it (byte-identical),
- pass: conservation, eligibility, row permutation, and single-eligible gates. —
  verified in `tests/test_orderer_allocator.py` (incl. conservation under 200 random
  score tensors, zero-quantity dropped).

O2. Supervised orderer scorer: **DONE** — contract (local TDD) + value (EXP-0012:
distilled scorer+allocator beats `proportional` on all held-out 6–9, mean 1.28×, min
1.09×, 0/20 death; ADR-0054).

- train `OrdererScoreModel` from the current search teacher or strong baseline, —
  model exists (`train/orderer_scorer.py`): `ProductStreamView` → one score per
  producer row, shared row encoder + once-embedded product/global context, no
  product/producer ids. Distillation/training is the cloud step (not yet run).
- eval same-size held-out, product relabeling, producer permutation, producer-count
  holdout, — **contract gates green locally** (`tests/test_orderer_scorer.py`):
  one-score-per-producer, score-follows-stats-not-position (row equivariance),
  runs-on-variable-producer-count, scores→`allocate_suborders` conserved+masked,
  schema/version + eval determinism.
- pass: checkpoint loads across producer counts; score rows move with producer rows;
  quality is competitive with current product-stream champion on compatible sizes. —
  **all met** (EXP-0012): the depth-3 checkpoint loads + runs at P=4/8/12 unchanged,
  and on held-out 6–9 / depth 3 (producer fixed `stock_aware`) it **ties the v3
  champion** `m2_stream_stochastic.pt` (mean 1.023×, min 0.983×, 0 death; ADR-0054).

O3. Size and topology transfer: **DONE (both size axes, zero-shot)** — same EXP-0012
checkpoint, no re-distill:
- *producer-count axis* — held-out 6–9 at depth 1 (P=4) / depth 2 (P=8): no contract
  failure, never regresses `proportional` (mean 1.000 / 1.005), 0/20 death; wins 1.28×
  at depth 3 where routing binds.
- *product-count axis* — fixed P=12, `products_per_layer` 4→8→16 (streams 52→96→184):
  v4 obs/action/k_stream dims constant, lever **strengthens** (mean 1.279→1.329→1.509,
  min always >1.08, 0/60 death). Zero-shot — the id-free scorer is reused per stream.
Transfer loss reported vs `proportional` and the v3 champion. **Still owed:** the
*out-of-family recipe* axis (different recipe DAG / capacity-type set, not just wider).

- train on small producer counts and product counts,
- eval on larger producer counts, larger product counts, and out-of-family recipes,
- pass: no contract failure; transfer loss is reported against deterministic
  baselines and current champion where shape-compatible.

### Combined Experiment Ladder

C-1. Dynamic stack first comparison:

- train: none for deterministic baselines; optional supervised scorer checkpoints if
  already available,
- evaluate `entity_planner + proportional`, portable producer with deterministic
  orderer, deterministic producer with portable orderer, and finally the full
  portable stack,
- targets: same-shape relabels, held-out scale seeds, larger product counts, larger
  producer counts, and larger product plus producer counts,
- include existing joint learned checkpoints only on shape-compatible cells as
  reference bars,
- pass: the matrix separates versatility failures from value failures and shows
  whether the dynamic stack is worth improving.

C0. Independent portable modules:

- producer = portable scorer/planner,
- orderer = deterministic proportional or a shape-compatible current champion,
- then orderer = portable scorer/allocator with deterministic producer,
- pass each side independently before joint evaluation.

C1. Portable stack smoke: **DONE** — both portable modules run jointly (EXP-0014). The
producer scorer/planner (EXP-0013) and orderer scorer/allocator (EXP-0012) coexist in
one env (`entity_producer_obs` + corrected v4 product-stream ordering, orthogonal flags),
reusing both distilled checkpoints unchanged. On held-out depth-3 seeds 6–9 the joint
stack **beats** the deterministic `entity_planner + proportional` bar (mean ratio
**1.109**, min **1.042**, zero death), and every seed is **byte-identical under a
capacity relabel** (4/4). Lift is orderer-driven (orderer beats `proportional` ~1.28×;
producer ties the planner). Local eval. New: `tokenomics.eval.joint_stack`.

- producer scorer/planner + orderer scorer/allocator,
- eval on train-size and relabeled train-size,
- pass: no feasibility failures, no relabel mismatches, episode metrics comparable to
  deterministic baselines.

C2. Size transfer:

- train both modules on small topology set,
- eval on larger product count and larger producer count,
- pass: both checkpoints load; failures are policy/scaling failures, not contract
  failures.

C3. Out-of-family transfer:

- eval on recipe graphs not seen during training,
- pass: report worst cells and failure reasons; promote only if policy degradation is
  bounded and explainable.

### Ablations

Use ablations to prove which part provides versatility:

- flat MLP vs shared row scorer,
- dense relation planes vs sparse edges,
- no row permutation augmentation vs row permutation augmentation,
- no product/capacity relabel augmentation vs relabel augmentation,
- learned direct allocation vs learned scorer plus deterministic allocator,
- config-fingerprint load gate vs schema-version load gate for portable checkpoints.

The expected result is not that every portable model wins everywhere. The expected
result is that portable contracts load across topology sizes, preserve relabel/row
semantics, and make policy failures measurable rather than hidden behind tensor
shape incompatibility.

## Orderer Contract Fix

Product-stream ordering agents are scoped to a single `(source, product)` stream. A
stream orderer should therefore not scale with the total product set or with the
number of products a producer can consume.

The current implementation still has a legacy dependency through global backlog
width `K`: `EnvSpec.k` is sized with a source-keyed formula equivalent to:

```text
K = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)
```

where `max_rel` grows with the number of producer input products. That formula was
valid for the old source-keyed orderer, where one orderer could manage multiple
product streams. It is not valid for product-stream orderers, where every agent
already manages one product stream.

Required fix:

- keep the existing product-stream v3 contract loadable for old checkpoints,
- add a corrected versioned orderer contract for product-stream mode,
- compute product-stream `K` from per-stream concurrency only, not from producer
  product-set width,
- ensure orderer observation and action shapes may depend on producer count and the
  fixed stream backlog bound, but not on product ids, product names, product
  vocabulary size, or `max_rel`,
- retrain or redistill the orderer for the corrected contract instead of silently
  loading old v3 weights into a different tensor shape.

The corrected stream bound is:

```text
K_stream = ORDER_LIFETIME
```

Rationale: a product-stream agent is scoped to a single `(source, product)` stream.
A consumer or replenishment stream holds at most `ORDER_LIFETIME` concurrent orders
(TTL), and never carries investment orders — those belong to their own per-type
streams, each gated to one in flight (well under `ORDER_LIFETIME`). So no single
stream needs `INVESTMENT_STREAMS` added on top.

Note the old `K = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)`
is `ORDER_LIFETIME` only when `max_rel == 0`; for `max_rel >= 1` it is already
`>= INVESTMENT_STREAMS + ORDER_LIFETIME`. A flat `K_stream = ORDER_LIFETIME + INVESTMENT_STREAMS`
would therefore be a *regression* on narrow topologies (`max_rel == 0`: `3 -> 6`)
and only a partial win on wide ones — avoid it. The important constraint is that
`K_stream` is bounded by one stream's possible simultaneous orders and does not
multiply by the number of products in the local topology.

## Checkpoint Impact

This fix does not invalidate all trained models. It affects trained models that
depend on the orderer observation/action shape:

- existing product-stream orderer checkpoints,
- joint checkpoints that include that orderer module,
- saved configs that assume the old product-stream `EnvSpec.k`.

It should not affect deterministic producer planning, producer scorer/controller
models, consumer models with separate observation contracts, or environment
dynamics.

## Acceptance Gates

The corrected product-stream orderer contract is accepted only if:

- product relabeling leaves orderer observations bit-identical,
- increasing product-set width with fixed producer count does not change orderer
  observation/action dimensions,
- producer row permutation only permutes producer-local rows/actions,
- the old v3 contract remains explicitly versioned and checkpoint-compatible,
- the corrected contract requires a fresh orderer training/distillation artifact,
- size-transfer audits distinguish true policy failure from contract
  incompatibility.
