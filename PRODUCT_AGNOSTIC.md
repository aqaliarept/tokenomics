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

P0. Deterministic planner parity: **DONE** (no training; covered by existing gates).

- controller: `entity_planner`,
- train: none,
- eval: same-shape relabel, row permutation, size holdout,
- pass: feasible commands, relabel gates pass, no major regression vs current
  deterministic baseline. — **all met.** Feasibility + relabel + row-permutation
  invariants are unit-gated in `tests/test_producer_view.py` and
  `tests/test_producer_planner.py` (product/capacity relabel equivalence, input-row
  permutation, padded-row invariance). Size holdout = `eval/producer_size_holdout.py`
  + `eval/producer_champion_audit.audit_producer_size_transfer` (EXP-0010D): the
  deterministic `entity_planner` runs across product/producer counts with no contract
  failure and no major regression. The portable scorer/planner reproduces it at P1
  (EXP-0013, mean 1.0029). No separate run needed — P0 is the deterministic floor P1
  already cleared.

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
  (`test_v4_*`); all Acceptance Gates green. Fresh v4 orderer artifact satisfied by
  EXP-0012; the remaining O3 gap is out-of-family recipe transfer.

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

C-1. Dynamic stack first comparison: **DONE (train-size + relabel axes)** — EXP-0015.
The reusable matrix runner `tokenomics.eval.transfer_matrix` emits the "Evaluation
Output" CSV with contract flags separated from value per cell. 30 rows (6 stacks ×
held-out depth-3 seeds 6–9 + a capacity-relabel target), all on the same
`joint_stack_env`: floor `noop+proportional` = 0 units (targets nontrivial), zero
contract failures, relabel target byte-identical to its base seed for every stack.
Stacks cover the doc's required comparison — `entity_planner + proportional` (bar),
portable producer + deterministic orderer (mean 1.009), deterministic producer +
portable orderer (mean 1.095), full portable stack (mean 1.109, min 1.042, 0 death =
EXP-0014 cross-check). **Still owed:** larger product/producer-count targets (C2) and
out-of-family recipes (C3) — added by widening `targets`, not new code.

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

C0. Independent portable modules: **DONE** (each side passed solo before C1).

- producer = portable scorer/planner, — EXP-0013: portable producer scorer/planner
  with the **deterministic `proportional`** orderer beats/ties `entity_planner +
  proportional` on held-out 6–9 at depths 1/2/3 (mean 1.0029, zero death).
- orderer = deterministic proportional or a shape-compatible current champion,
- then orderer = portable scorer/allocator with deterministic producer, — EXP-0012:
  portable orderer scorer/allocator with the **deterministic `stock_aware`** producer
  beats `proportional` on held-out 6–9 (mean 1.28×, min 1.09×, 0/20 death) and ties
  the v3 champion.
- pass each side independently before joint evaluation. — **met:** each portable
  module was validated against a deterministic counterpart (P1 = portable producer +
  det. orderer; O2 = portable orderer + det. producer) before they were composed in
  C1 (EXP-0014). Missing apples-to-apples row: a single side-by-side table pairing
  both solo results against the joint result — folded into the C-1 verification matrix
  (next).

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
K_stream = ORDER_LIFETIME + INVESTMENT_STREAMS
```

Rationale: a product-stream agent is scoped to a single `(source, product)` stream.
A consumer or replenishment stream holds at most `ORDER_LIFETIME` concurrent orders
(TTL), and producer-funded restoration/upgrade/inventory orders are still routed by
`(producer, product)` when their requirement uses that product. A valid stream can
therefore contain replenishment orders plus the fixed set of gated investment
orders. The bound stays independent of product vocabulary width and `max_rel`.

Note the old `K = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)`
is `ORDER_LIFETIME` only when `max_rel == 0`; for `max_rel >= 1` it is already
`>= INVESTMENT_STREAMS + ORDER_LIFETIME`. The important constraint is that
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

## Ordered Action Plan

This is the current close-out order for the product-agnostic ladder. The next priority is
to lock the verified baseline evidence, then fix the producer-side EXP-0017 policy gap
before expanding to harder C3/P3 fixtures. The joint stack can hide this producer issue,
so producer-alone rows are now a required gate before any final portability claim.

1. **Reconcile status documentation.** — **DONE.**
   - P0 written as PASS using existing deterministic planner relabel, row-permutation,
     and EXP-0010D size-holdout evidence.
   - C0 written as effectively PASS from P1 with deterministic orderer and O2 with
     deterministic producer; missing apples-to-apples rows folded into the matrix.
   - O0/O2/O3 wording reconciled: EXP-0012 satisfies the fresh v4 orderer artifact;
     O3 still owes only remaining out-of-family recipe axes.

2. **Build the verification matrix CSV (C-1 first deliverable).** — **DONE (EXP-0015).**
   - Add a reusable runner that emits the columns listed in "Evaluation Output":
     stack identity, train and target topology, schema-load flags, contract flags,
     episode run flag, failure class, value, death, baseline family, and ratios. —
     `tokenomics.eval.transfer_matrix` (`StackSpec`/`TargetTopology`/`transfer_matrix`/
     `write_matrix_csv`/CLI); all canonical columns emitted; gated by
     `tests/test_transfer_matrix.py`.
   - Populate initial rows from already available artifacts: deterministic
     `entity_planner + proportional`, portable producer + deterministic orderer,
     deterministic producer + portable orderer, and EXP-0014 full portable stack. —
     done + a `noop` floor and `stock_aware+proportional`; 30 rows in
     `experiments/EXP-0015-verification-matrix/summary.csv`.
   - Record legacy fixed checkpoints only on shape-compatible cells and mark
     incompatible cells as contract failures, not policy failures. — runner supports it
     (per-side `*_schema_loads`/`*_contract_compatible`/`failure_class`); no legacy
     `GaussianActor` ckpt wired in yet, so this matrix has zero contract failures (all
     portable/deterministic). Legacy incompatibility shows once a flat ckpt is pointed
     at an off-size target (a later rung).

3. **Run C2 size transfer with the EXP-0014 harness.** — **DONE (EXP-0016).**
   - Reuse the EXP-0012 and EXP-0013 checkpoints with no retrain. — done; same ckpts,
     `transfer_matrix` + new `size_transfer_targets`, no policy code changed.
   - Evaluate larger product count, larger producer count, and larger product plus
     producer count targets. — done: `prodW8` (24 products), `agentW8` (24 producers),
     `prodW8-agentW8` (24/24), off base seed 6, depth fixed 3.
   - Report load/run/death/value separately from scaling metrics so contract,
     policy, and runtime failures stay distinct. — done: every portable cell
     `runs_episode=True`/`failure_class=none` (zero contract failures), floor `noop`=0
     (targets nontrivial), zero death, full stack ties/beats the planner bar on all
     three directions (prodW8 1.286 / agentW8 1.063 / both 1.147). C3 (out-of-family)
     still open.

4. **Record the first out-of-family recipe probe.** — **PARTIAL C3 DONE (EXP-0017).**
   - Fork/join + multi-input fixture added via
     `eval/fixtures.py::build_forkjoin_config`; same checkpoints run unretrained.
   - Contract PASS: all portable cells run with `failure_class=none`, zero death.
   - Joint value PASS: full portable stack beats the planner bar (mean 2.86x,
     min 2.64x), but producer-alone under `proportional` regresses on s8/s9.
   - Treat EXP-0017 as baseline evidence, not final C3 closure: shared-capacity,
     deeper joins, uneven strengths, scarcity, and degenerate fixtures still remain.

5. **Land the current verified slice before broadening fixtures.**
   - Commit the current verified work as a stable checkpoint: orderer O0-O3, producer
     P0/P1, C0/C1, EXP-0015 matrix, EXP-0016 size transfer, EXP-0017 fork/join,
     EXP-0018 planner-gated producer acceptance, docs, tests, and review fixes.
   - Keep the producer-alone caveat explicit: EXP-0018 materially reduces the
     fork/join producer gap, but strict producer-alone no-regression remains open on
     s9 (`0.955`). Later training work should compare against this evidence rather
     than rewriting the baseline.

6. **Fix the producer-side out-of-family policy gap surfaced by EXP-0017.** — **CODE
   DONE (local TDD); acceptance rerun = step 7.**
   - Diagnosis on fork/join s8/s9: the portable producer contract holds, but
     producer-alone value under `proportional` regresses because learned investment
     gates generalize poorly out-of-family. On s8, replacing only the learned
     upgrade/inventory gates with deterministic planner gates recovers the bar
     (`35.2 -> 78.6` units; `0.450x -> 1.005x`), while keeping learned gates stays
     bad even with planner replenishment scores.
   - Immediate robust path: keep upgrade and inventory-expansion gates planner-owned
     for P2/C3 deployment, and let the learned producer scorer own continuous/ranking
     scores only until gate training covers fork/join, shared-capacity, scarcity, and
     running-investment states. — **landed:** `score_with_model(model, view,
     planner_gates=True)` sources the upgrade / inventory-expansion gates from the
     deterministic `score_producer_entity_view` while keeping the learned continuous
     heads; `scorer_producing_policy(model, planner_gates=True, replenish_tie_rel_tol=…)`
     deploys it. Same checkpoint, same schema — only the policy adapter differs. Exposed
     in `transfer_matrix` as the distinct producer kind `portable_scorer_planner_gates`
     (no silent flip of `portable_scorer`) with a `producer_policy_mode` =
     `learned_gates | planner_gates` CSV column; `DEFAULT_STACKS` now carries
     `portable_producer_planner_gates+proportional` and
     `…+portable_orderer` alongside the original learned-gate rows (kept as regression
     bars). Gated by `tests/test_producer_scorer.py` + `tests/test_transfer_matrix.py`.
   - Secondary fix: make replenishment allocation tie-tolerant for near-equal input
     scores. Fork/join recipes need balanced paired inputs; learned `input_value`
     noise can break the planner's exact-tie split and overfill one input while
     starving its mate. — **landed but ships OFF (`tol=0.0`).** `plan_producer_command(…,
     replenish_tie_rel_tol=…)` widens the planner's exact-tie group to a relative band
     (`0.0` = unchanged deterministic behavior). A local ablation over fork/join s8/s9
     (planner-gated producer, `proportional` orderer) showed the tolerance is **not
     monotonic**: it helps s9 (0.955→0.992 at tol≥0.01) but hurts s8 (1.005→0.946 at
     0.02), the s8 loss outweighing the s9 gain by mean ratio. So it stays a **tunable
     ablation knob** — `transfer_matrix --tie-rel-tol`, each row labeled by
     `producer_tie_rel_tol` — not a shipped default (`PLANNER_GATE_TIE_REL_TOL = 0.0`).
     Gated by `tests/test_producer_planner.py`
     (`test_planner_splits_headroom_evenly_for_near_equal_input_scores`,
     `test_tie_tolerance_does_not_merge_clearly_different_input_scores`) and
     `tests/test_transfer_matrix.py`
     (`test_planner_gated_tie_tolerance_defaults_to_zero_and_is_tunable_per_row`).
   - **Acceptance smoke (local, fork/join s8/s9, producer-alone under `proportional`, at
     the SHIPPED default `tol=0.0`):** planner-gated producer recovers s8 0.450→**1.005**
     (clears the bar) and s9 0.826→**0.955** (still below), vs the learned-gate regression
     (0.450 / 0.826); joint stack unchanged-strong (2.83 / 2.97), zero contract failures,
     zero death. (At the `tol=0.02` ablation point the split shifts to s8 0.946 / s9
     0.992 — the non-monotonicity that keeps tie tolerance off by default; see the
     secondary-fix note above.) **Status: producer policy gap MATERIALLY REDUCED, not
     fully closed** — s9 sits at 0.955 producer-alone, so the "no producer-alone seed
     below 1.0" gate is not strictly cleared at any single tolerance. Final P2/C3
     producer-side closure needs training coverage (fork/join, shared-capacity, scarcity,
     running-investment states) or a better replenishment/investment treatment that
     clears **both** s8 and s9 without a seed-specific tradeoff. The full EXP-0018 writeup
     + multi-seed (6–9) rerun is recorded in step 7.

7. **Rerun EXP-0017 as the producer-fix acceptance gate.** — **DONE (EXP-0018,
   local eval).**
   - Reuse the same fork/join seeds 6-9, same checkpoints, same `transfer_matrix`
     stacks, and same `entity_planner + proportional` bar.
   - Gate: zero contract failures, zero death, joint stack still passes, and then make
     the producer-alone bar explicit: either **materially reduced** or strict **no
     producer-alone seed below 1.0**.
   - **Result** (`experiments/EXP-0018-producer-planner-gate-acceptance/`, 8-stack
     rerun, shipped `tol=0.0`): all 28 non-floor cells `runs_episode=True` /
     `failure_class=none` / `dead=0` (contract + death unchanged from EXP-0017). Joint
     stack PASS — planner-gated joint **2.91×** mean (min 2.67×), edges out the
     learned-gate joint 2.86×. Planner-gated **producer-alone** under `proportional`
     recovers the regression: s8 **0.450→1.005**, s9 **0.826→0.955**, s6 1.106, s7
     1.010 → **min 0.450→0.955, mean 0.874→1.019, 3/4 seeds ≥1.0**.
   - **Acceptance bar DECISION**: *"materially reduced"* — **MET**; strict *"no
     producer-alone seed below 1.0"* — **NOT MET** (s9=0.955). So step 7 lands as the
     materially-reduced gate; the strict gate stays open on s9. Confirms the step-6
     single-seed prediction across all four held-out seeds at `tol=0.0`.
   - **Residual**: closing s9 too (without a seed-specific tie-tolerance tradeoff — the
     knob is non-monotonic) needs producer-side **training coverage** of fork/join,
     shared-capacity, scarcity, and running-investment states, not another deterministic
     patch. That is a producer training/fidelity item (steps 8-12 / future EXP), distinct
     from this gate. EXP-0018 recorded separately (kept EXP-0017 as baseline evidence).

8. **Run P-1 and O-1 dynamic baseline audits through the matrix.** — **DONE (EXP-0019,
   local eval).**
   - Producer side: compare `entity_planner`, learned-gate portable scorer,
     planner-gated portable scorer, and shape-compatible legacy producers under
     `proportional` or another deterministic orderer. — done via the new
     `transfer_matrix --audit producer` mode + `PRODUCER_AUDIT_STACKS` (orderer fixed
     `proportional`). Legacy flat producers omitted (no checkpoint wired;
     contract-incompatible off-size — recorded, not run).
   - Orderer side: compare `proportional`, portable scorer/allocator, and
     shape-compatible legacy orderers under `stock_aware` or `entity_planner`. — done via
     `--audit orderer` + `ORDERER_AUDIT_STACKS` (producer fixed `stock_aware`; orderers
     `even_split`/`proportional`/`obs_price`/portable). Legacy product-stream orderers
     omitted (same-shape references only, unwired).
   - Include same-shape relabels, numeric perturbations, larger product counts, and
     larger producer counts; keep contract failures distinct from policy failures. — done:
     two reusable same-shape target generators added (`numeric_perturbation_targets`
     reusing `eval.generalization`; `producer_permutation_targets` reusing
     `permute_scale_config`), TDD'd in `tests/test_transfer_matrix.py`.
   - **Result** (`experiments/EXP-0019-dynamic-baseline-audit/`): **both audits clean —
     zero contract failures**, floor `noop`=0 (targets nontrivial), contract vs policy
     stays separable on every cell. **Producer** (orderer `proportional`): learned-gate
     portable **ties/beats the planner bar on every in-family target** (mean **1.012**,
     min **1.000**) → the EXP-0017/0018 producer-alone regression is **out-of-family
     (fork/join) specific, not a general weakness**; planner-gated mirror (mean 1.001,
     min 0.975 on doubly-widened `prodW8-agentW8`) trades a little in-family upside for the
     out-of-family robustness it was built for; `stock_aware` 0.93. **Orderer** (producer
     `stock_aware`): the portable scorer/allocator **beats every deterministic orderer on
     all 13 targets** (mean **1.088**, min 0.990, zero death; lever strengthens on `prodW8`
     to 1.256), `proportional` is the best deterministic (0.93), `obs_price` weak (0.58,
     partial death on 6 targets). **Equivariance**: capacity relabel byte-identical for all
     stacks; under the **producer-row permutation** the portable orderer is **exactly
     invariant** (501.2→501.2) while every deterministic orderer shifts (the domain auction
     ties-break by producer id) — the O-1 producer-permutation gate met by the portable
     orderer, missed by the baselines.
   - **Takeaway for next steps**: no schema/contract work owed on these axes. The only
     producer-side gap is the out-of-family one (already materially-reduced via EXP-0018
     planner gates; strict s9 closure = producer training coverage, steps 9–12); the
     orderer side has no open gap on the step-8 axes.

9. **Add the remaining reusable fixture families before chasing s9.** — **DONE (builders
   landed in `eval/fixtures.py`, TDD'd in `tests/test_fixtures.py`; no run yet — that is
   step 10).**
   - Out-of-family: shared capacity types, deeper joins, and a dedicated uneven-strength
     producer axis. Fork/join + multi-input is already covered by EXP-0017/0018. — done:
     `build_shared_capacity_config` (levels cycle through `max(1,depth-1)` capacity types,
     so fewer types than levels), `build_deep_join_config` (fork/join pushed to depth 5 +
     3-input joins), `build_uneven_strength_config` (one dominant producer per level, the
     rest weak → guaranteed ≥12× within-level capacity gap). Each isolates ONE axis off the
     scale family; the two single-input builders share one `_build_single_input_layered`
     scaffold parameterized by a cap-type fn and a capacity-amount fn.
   - Scarcity: low inventory headroom, low funding, degraded capacity, high backlog,
     and already-running investments. — done: `scarcity_suite(base)` returns five
     dims-preserving single-axis stress variants (`low_funding` = lean investment fund via
     new `scale_funding`; `low_inventory` = tight buffers via new
     `scale_inventory_headroom`; `high_backlog`/`degraded` reuse `generalization.perturb_*`;
     `running_investment` = lean-and-decaying start, the static stand-in for in-flight
     orders, which are runtime state not config).
   - Degenerate: zero eligible alternatives, one eligible producer, no backlog, no
     input products, no inventory expansion, and mostly padded rows. — done:
     `degenerate_suite(seed)` = `single_producer` (one declarer per product → zero
     routing alternatives), `primal_only` (depth-1, no input products), `no_inventory_
     expansion` (expansion stripped), `no_backlog` (fixed product, quantity 1), `minimal`
     (one product + one producer per level → mostly padding once encoded).
   - Purpose: expose the full producer/orderer failure surface before retraining or
     tuning for strict producer-alone closure, so the s9 fix is not seed-specific. — the
     families are now reusable target generators for step 10; nothing run/classified yet.

10. **Run the broad C3/P3 fixture matrix and classify failures.** — **DONE
    (`experiments/EXP-0020-broad-fixture-matrix/`).**
    - Run producer-only, orderer-only, and joint-stack rows on shared-capacity,
      deeper-join, uneven-strength, scarcity, running-investment, and degenerate
      fixtures. — done: new `transfer_matrix --fixtures` target source (6 reusable
      generators wrapping the step-9 fixtures + `fixture_matrix_targets` umbrella; composes
      with `--audit`; TDD'd in `tests/test_transfer_matrix.py`). 22 targets × 3 modes,
      EXP-0012/0013 checkpoints, seeds 6–9, 5 eval seeds, horizon 100.
    - Classify every miss as schema/load, decoder/allocator contract, producer policy
      quality, orderer policy quality, scarcity/stress behavior, or scaling. — done:
      **schema/load 0, contract 0** (id-free contract loads/runs on every new family);
      producer dips small + concentrated on **uneven-strength** + a few scarcity axes;
      orderer **0 misses** (ties-or-beats every deterministic orderer everywhere, joint
      mean 1.231 / max 2.002 on uneven-strength); the only scaling result is
      `deepjoin-d5` = depth×TTL feasibility wall (0 units for ALL policies incl.
      `entity_planner`), not a contract/policy failure. The two degenerate sub-1.0 orderer
      cells are no-routing-choice economies where all orderers tie (fixed-producer floor).
    - Keep the EXP-0018 planner-gated rows as the current producer deployment baseline;
      learned-gate rows remain regression bars. — done: both portable producer modes carried
      (planner-gate ≈ learned on every fixture; both `producer_policy_mode`-labeled).
    - **Feed-forward to step 11:** the broadened surface localizes the only remaining
      producer gap to **uneven-strength** (concentration routing/investment) + a few
      **scarcity** axes — the exact coverage to add. `deepjoin-d5` is a feasibility floor
      (depth>TTL), so a *feasible* depth-3 wider-join probe is the open item to isolate
      join-WIDTH policy quality from the depth wall.

11. **Do one producer training/fidelity pass for strict closure.** — **DONE (EXP-0021);
    primary target CLOSED, two residual sub-1.0 cells named. Acceptance = step 12.**
    - **Result** (`experiments/EXP-0021-producer-broaden-training/`, Runpod CPU distill,
      depth 3, seeds 0–5, 300 epochs, 54 topologies): the fork/join producer-alone **s9**
      regression steps 6–8 could not close deterministically is **closed 0.955 → 1.000**.
      In-family scale 6–9 (d1/2/3) mean **1.0011** / min **0.9921**, 0 death, invest-FP ≈ 0 —
      a small in-family cost (one cell, s9d3, 1.0000 → 0.9921) bought the out-of-family
      robustness. Confirms the hypothesis: the s9 gap lived in the learned *continuous*
      replenishment heads (planner-gated investment deploy untouched, EXP-0018). **Not a
      strict clean sweep** — see step 12 for the one remaining held-out cell.
    - Use the failures from steps 8-10 to broaden producer training coverage in one
      pass: fork/join, shared-capacity, uneven strengths, scarcity, and
      running-investment states. — **landed:** `producer_scorer.broadened_producer_train_
      configs(seeds, depth, width)` materializes the scale base + the exact five families
      step 10 localized (reusing the `eval.fixtures` builders the C3/P3 matrix consumes;
      train seeds 0–5 vs acceptance seeds 6–9), and `distill_producer_scorer` gained a
      `train_configs` param + a `--broaden` CLI flag so it distills over that 54-economy set
      instead of scale-only. Same `ProducerScoreModel` / schema / checkpoint v2 — only
      training coverage changes, so the new checkpoint stays load-compatible with every
      fixture/size target. Gated by `tests/test_producer_scorer.py`
      (`test_broadened_producer_train_configs_cover_the_step11_families`,
      `test_distill_producer_scorer_accepts_explicit_train_configs`).
    - Target the actual remaining issue: strict producer-alone no-regression under a
      deterministic orderer. Avoid seed-specific tie-tolerance tuning unless it also
      improves the broadened matrix. — rationale: EXP-0017 step-6 showed the obvious
      deterministic knob (replenish tie tolerance) is **non-monotonic** (helps s9, hurts
      s8), so broadening training coverage is the cleaner next bet than more knob-chasing —
      not a proof that no deterministic patch could work. Broadening targets the LEARNED
      CONTINUOUS heads (replenishment / product urgency / capacity bottleneck) — the
      suspected driver of the fork/join s9 gap; deploy stays planner-gated for investment
      (EXP-0018), tie tolerance stays off. EXP-0021 + step 12 now support that hypothesis
      for fork/join, shared-cap, scarcity, and degenerate fixtures, with uneven-strength s6
      and scale s9d3 recorded as the residuals.
    - Acceptance candidate: producer-alone planner-gated scorer clears every covered
      fixture seed (`ratio_vs_practical_baseline >= 1.0`), with zero death and zero
      contract failures. — measured in step 12 via `transfer_matrix --fixtures` on the
      pulled checkpoint. Run command + the required step-11/12 readouts live in
      `experiments/EXP-0021-producer-broaden-training/`; strict closure is not claimed
      because uneven-strength s6 and scale s9d3 remain sub-1.0.
    - **Primary fallback for close-but-under residuals = DAgger**, not architecture change:
      the likely issue is on-policy distribution shift — the scorer's own replenishment
      choices reach states a teacher-only rollout never labeled. Collect on the learned
      scorer's rollouts (deferred at EXP-0013) before touching the model. Then: wider BCE
      gate coverage / threshold recal; last, narrow the claim (step 15) — never launder a
      producer-alone miss through joint-stack success.

12. **Rerun the strict producer and joint acceptance gates.** — **DONE (EXP-0021 step 12,
    local eval with the pulled broad checkpoint); MATERIALLY CLOSED, one residual cell.**
    - Re-ran EXP-0018 fork/join seeds 6–9 (8-stack, orderer ckpt EXP-0012) + the broad
      C3/P3 producer audit (`transfer_matrix --audit producer --fixtures`). Producer-alone
      = `portable_producer_planner_gates+proportional`.
    - **Result**: zero contract failures across BOTH matrices, and zero death for all
      non-floor, non-deepjoin policy rows. Deepjoin-d5 is a depth×TTL feasibility wall,
      where the `entity_planner` bar also completes 0 units and sometimes dies — a scaling
      result, excluded. Producer-alone planner-gated **fork/join
      1.000/1.034/1.033/1.000 (min 1.000, was 0.955)**, shared-cap min 1.000, scarcity (all
      5 s6 axes) ≥ 1.0, degenerate all 1.0. Joint stack fork/join **2.71–3.02×**, all pass.
    - **Strict gate NOT fully met**: exactly **one** producer-alone cell stays < 1.0 —
      held-out **uneven-strength s6 = 0.975** (its s7/s8/s9 pass) — plus the in-family scale
      s9d3 = 0.9921. Both named; joint success is NOT treated as sufficient.
    - **Final claim (per step 15)**: portable stack is product-agnostic + strict
      producer-alone on the fork/join + shared-cap + scarcity + degenerate family; the one
      open producer-alone cell is uneven-strength s6. Next move if strict-everywhere is
      wanted = DAgger on the learned scorer's own rollouts (concentration-routing states on
      uneven-strength), not a deterministic patch or architecture change.

13. **Tighten portable checkpoint schema gates.** — **DONE (local TDD).**
    - Move portable load acceptance toward feature names, row/entity types,
      relation/edge types, normalization version, decoder/planner version, and
      architecture version. — **landed:** shared `train/portable_schema.py`
      (`PortableSchema` dataclass + `assert_compatible`, the exact unit step 14's
      "fingerprint vs schema-version gate" ablation compares). Both scorers expose their
      current schema (`producer_portable_schema()` = global/capacity/product/input named
      features + `product_capacity_required`/`product_input_required` relations + norm/
      `plan_producer_command`/architecture versions; `orderer_portable_schema()` =
      product/producer-row named features, no relations, `allocate_suborders` decoder). Save
      writes the `schema` block; load asserts it by NAME (a feature rename / entity-type /
      decoder / architecture change rejects with a field-named `ValueError`).
    - Keep the full config fingerprint as audit metadata rather than the primary
      portable load gate. — **landed:** `save_*` take an optional `config_fingerprint`
      stored as audit-only metadata; load never compares it (test: two checkpoints with
      different fingerprints, same schema, both load).
    - Add negative tests proving real schema/semantics changes reject cleanly while
      product count, producer count, and relabel changes do not. — **landed:**
      `tests/test_portable_schema.py` (renamed feature → reject; tampered entity_types →
      reject; different fingerprint → loads). Topology is absent from the schema by
      construction, so widening/relabeling can't trigger a reject (cross-width RUN already
      proven in EXP-0016/O3). **Backward-compatible**: the named gate is additive — legacy
      v2/v1 checkpoints saved before the block (live EXP-0012/0013/0021 artifacts) load
      unchanged through the existing version + feature-dim floor (regression-tested).

14. **Run targeted ablations after the matrix and fixtures exist. — DONE (sufficient
    ablations complete).** #1 + #2 answer the portability claim directly; #3 deferred as
    low marginal value (see below); dense-planes / aug-on-off left unless a reviewer
    challenges sparse edges or augmentation. Evidence chain now closed:
    fingerprint-vs-schema (schema gate necessary) → flat MLP (fixed-shape not portable) →
    row no-relation (row factoring gives portability) → row + sparse relation (small value
    win under contention) → planner-gated producer + deterministic allocator (champion,
    contract-stable).
    - Prioritize the ablations that answer portability questions directly: flat MLP
      vs row scorer, dense planes vs sparse edges, augmentation on/off,
      learned-direct allocation vs scorer plus deterministic allocator, and
      fingerprint gate vs schema-version gate.
    - Run only on representative matrix cells so ablation cost stays bounded.
    - **#1 fingerprint gate vs schema-version gate — DONE (local, `tokenomics.eval.gate_ablation`,
      test `test_gate_ablation.py`).** Both candidate load gates run on the SAME portable
      producer checkpoint (stamped with the base economy's `config_fingerprint`) across the
      topologies a portable policy must transfer to. Result:

      | target | fingerprint changed | schema gate loads | fingerprint gate loads |
      |---|---|---|---|
      | base | no | ✓ | ✓ |
      | capacity_relabel | yes | ✓ | ✗ |
      | wider_product (8/layer) | yes | ✓ | ✗ |
      | wider_producer (8/layer) | yes | ✓ | ✗ |

      The `config_fingerprint` gate rejects every non-identical topology (it folds in
      producer ids / capacity types / vocab / dims); the topology-free `PortableSchema`
      gate loads all four. Empirical case for schema-version loading — a fingerprint gate
      would block the exact transfers the portable contract claims. Weights irrelevant
      (untrained scorer): only LOAD-GATE behavior under test.
    - **#2 flat MLP vs row scorer — DONE (EXP-0022). Edges = value lever, not portability
      lever; flat MLP non-portable by construction.** Two baselines vs the row+edge champion
      (`EXP-0021 .../producer_scorer_broad.pt`):
      - **A. fixed-pad flat MLP (`ProducerFlatModel`) — non-portable BY CONSTRUCTION**
        (`test_producer_ablation_arch.py::test_flat_model_scores_within_pad_but_is_non_portable_beyond_it`).
        `forward_view` raises `ValueError` the moment an entity count exceeds its build pad —
        cannot ingest a wider economy, not a viable portable policy. No pod run: the
        construction failure IS the result.
      - **B. per-row, no-edge scorer (`ProducerScoreModel(use_edges=False)`) — distilled on
        Runpod, fully portable.** Held-out scale 6–9 ratio-vs-planner 1.0000; producer-alone
        fixtures audit = **0 contract failures / 0 death**, equals the edge champion on 19/22,
        **never better**, loses only on **3 stressed cells**: scarcity low_inventory −2.5%,
        sharedcap s8 −1.8%, scarcity high_backlog −0.8%. So the PER-ROW structure gives id-free
        portability; the EDGES add a small strictly-non-negative value only under scarcity /
        shared-cap. `transfer_matrix` now stamps `producer_arch` so no-edge rows are
        self-labeled (`producer-row-noedge-scorer-v2` vs champion `-edge-`).

      No new champion — row+edge scorer (EXP-0021) stays; EXP-0022 quantifies the cost of
      dropping edges. Unaffected EXP-0021 residuals named (uneven-strength s6=0.975, scale
      s9d3=0.9921 — identical no-edge vs edge).
    - **#3 learned-direct vs scorer + deterministic allocator — DEFERRED (low marginal
      value).** The main portability claim is already carried by #1/#2 + the retained
      champion. This ablation is expensive and easy to MISREAD: a failure could be
      action-head design, masking, quantity conservation, training coverage, or topology
      mismatch — many confounded causes, unlike the clean single-variable ablations already
      run. Spend it ONLY if a reviewer specifically challenges the deterministic
      planner/allocator split; otherwise it stays deferred.

15. **Promote or narrow the final claim.** — **NARROWED CLAIM PROMOTED (steps 13 + 14
    complete).**

    **Promoted claim:** the portable id-free stack (ADR-0053 producer + ADR-0054 orderer)
    is **product-agnostic across the covered schema family** — schema-compatible product /
    producer counts, capacity relabels, fork/join, shared-capacity, scarcity, and
    degenerate topologies — with **zero contract/load failures** anywhere covered
    (EXP-0016/0017/0020, and the schema-version load gate of step 13 that makes this
    load-portable, not fingerprint-locked). The **row + sparse-relation producer champion**
    (EXP-0021) is retained: step-14 ablations show the row factoring is what buys
    portability, the relation context adds a small contention-only value win (EXP-0022),
    a flat/fixed-shape architecture is not portable, and the schema gate (not a config
    fingerprint) is the correct load gate. Joint stack beats the best-available bar
    everywhere covered.

    **Open residuals (named, not laundered through the joint):** producer-alone
    uneven-strength s6 = 0.975 and in-family scale s9d3 = 0.9921. **`deepjoin-d5` is a
    depth×TTL feasibility wall (chains ≥4 vs TTL=2), not a policy failure** — `entity_planner`
    also completes 0 units there. Closing the two sub-1.0 cells to a strict-everywhere
    claim is a **bounded, targeted DAgger** item (learned-scorer rollouts on those state
    families — NOT an architecture change, NOT deterministic tie-tolerance), deferred: run
    only if "strict producer-alone every covered cell ≥ 1.0" becomes a hard release gate.
    Learned-direct allocation (ablation #3) stays deferred as low marginal value.

    ---
    _Prior provisional standing (retained for history):_
    - If C2, C3, and P3 pass with independent producer/orderer rows, promote the
      portable stack as product-agnostic within the covered schema family and list
      remaining stress/ablation caveats.
    - If producer-alone out-of-family or scarcity fails, narrow the claim to the
      passing topology family and record the next training-strength or architecture fix
      before claiming general recipe transfer.
    - **Current standing (EXP-0021)**: the portable id-free stack is product-agnostic
      (zero contract failures across scale + fork/join + shared-cap + uneven + scarcity +
      degenerate + size axes) AND strict producer-alone (≥ 1.0, planner-gated, vs
      `entity_planner + proportional`) on the **fork/join + shared-cap + scarcity +
      degenerate** family; the joint stack beats the bar everywhere covered. **Two open
      sub-1.0 cells, both named, neither hidden by the joint**: held-out uneven-strength
      s6 = 0.975 (producer-alone) and in-family scale s9d3 = 0.9921. Closing them to a
      strict-everywhere claim is a **bounded, targeted DAgger** item (learned-scorer
      rollouts on the uneven-strength / that-cell state families — NOT an architecture
      change, NOT deterministic tie-tolerance), deferred: run it only if "strict
      producer-alone every covered cell ≥ 1.0" becomes a hard release gate. Steps 13
      (schema-version load gates) and 14 (ablations) proceed under this narrowed claim.
