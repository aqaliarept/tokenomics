# EXP-0039 Graph Architecture

## Purpose

EXP-0039 asks whether the EXP-0037 marginal-value teacher is representable from
deployable decision-time economy state when that state is presented relationally.
This document fixes the primary `graph_primitives_v1` observation and model boundary
before graph dumping or probe training starts.

The graph is deliberately not independent of economic topology. Recipe dependencies,
producer eligibility, capacity ownership, order ownership, and inventory flows are the
signal the model is meant to use. The model is instead independent of:

- concrete producer, product, capacity, orderer, and order identities;
- the order in which nodes are serialized;
- fixed producer, product, capacity, order, or orderer counts;
- a particular task generator or task ID; and
- padded positions such as producer column or stream index.

The central rule is:

```text
fixed schema and shared parameters + variable topology and live state
```

## Architecture Layers

The implementation has three data layers and one model layer.

### 1. Graph schema

The schema fixes node types, relation types, feature names, feature transforms, and
their semantic meaning. It is versioned as `graph_primitives_v1`. Changing the
meaning, visibility, or presence of a primary feature requires a new schema version.

### 2. Topology instance

The topology compiler reads a `Configuration` and produces the static part of an
economy graph:

- producers, products, capacity types, and product-stream orderers;
- recipe capacity and input relations;
- producer eligibility and capacity ownership;
- orderer stream-product relations; and
- producer-to-orderer source relations for producer-funded streams.

The compiler may use domain IDs to join objects while constructing the graph. Those
IDs are metadata and local-index lookup keys, not model features.

### 3. Decision snapshot

The snapshotter overlays live decision-time state:

- live orders and their owners;
- current producer capacity and inventory;
- outstanding inbound inventory;
- folded telemetry from completed past cycles;
- current-cycle pending bids from earlier ordering agents;
- live composite reservations;
- the current order and current orderer anchors; and
- eligible candidate producer edges.

One snapshot is materialized as one graph per routing decision. When one ordering
agent has several incoming orders, those graphs share the same pre-action economy
state but have different `current_order` and `candidate` edges.

### 4. Shared relational model

The GNN consumes only typed features, local adjacency indexes, graph batch indexes,
and candidate indexes. No model parameter is indexed by a domain identity or topology
position.

## Tested Topology Sizes

The current three tasks compile to:

| Task | Products | Producers | Capacity types | Product-stream orderers |
|---|---:|---:|---:|---:|
| s29 | 12 | 6 | 3 | 32 |
| s77 | 12 | 9 | 3 | 42 |
| s45 | 12 | 9 | 3 | 42 |

Live order and commitment counts vary by decision. These counts are descriptive,
not tensor-shape constants.

The corrected leave-one-seed-out gate holds out trajectories within each task. It
does not hold out an unseen topology. Passing EXP-0039 therefore establishes
within-topology, held-out-trajectory representability, not topology transfer.

## Snapshot Boundary

The snapshot must be taken:

1. while the environment is at the labeled ordering agent's turn;
2. after producer actions and investment funding for the cycle;
3. after any earlier ordering agents in the AEC sequence have acted;
4. before the current ordering agent action is applied;
5. before calling the marginal-value teacher; and
6. before reading or attaching teacher labels or actor scores.

`marginal_value_labels` currently deep-copies the environment and does not mutate the
source environment. Extraction still happens first so causality is structural and a
future labeler change cannot silently contaminate graph construction.

The current order is one order within the ordering agent's joint action. The other
incoming orders remain visible as ordinary live orders. No hypothetical result of
routing the current order is visible.

## Node Types And Features

All continuous feature names and transforms must be written to
`graph_schema.json`. Binary features remain unstandardized.

### `decision`

Exactly one node. It has no domain scalar features. Its learned initial state is the
schema-role embedding for `decision`.

Current context is expressed structurally through `current_order`,
`current_orderer`, and `candidate` relations.

### `order`

Include every order in `system.orders`, not just the current order or current
orderer's queue.

Primitive features:

- original quantity;
- unresolved quantity;
- accepted quantity;
- completed quantity;
- failed quantity;
- available tokens;
- burnt tokens;
- order age;
- TTL remaining;
- funded flag; and
- one-hot order kind: consumer, replenishment, or investment.

Do not add `is_current` or `same_orderer` scalar flags. The `current_order` and
`owns` graph relations express those facts without duplicating them.

### `orderer`

Include configured product-stream orderers. Features are limited to a source-role
one-hot distinguishing consumer-funded and producer-funded streams. The fixed stream
product and producer source, when present, are relations.

Do not expose the orderer ID or its serialized position.

### `producer`

Include all configured producers, including currently ineligible or zero-capacity
producers.

Producer-wide primitive features:

- cut-off price plus a separate `cutoff_is_defined` mask;
- funds;
- joint inventory limit;
- upgrade-running flag;
- restoration-running flag;
- inventory-expansion-running flag; and
- has-inventory-expansion flag.

Per-capacity, per-product, and historical state belongs on relations below.

### `product`

Include every configured product. No scalar product ID and no learned per-product
embedding are allowed.

Allowed primitive historical features:

- folded demand quantity EMA; and
- folded rejected-demand quantity EMA.

Current open demand is represented by live order nodes and must not be pre-summed
onto the product.

### `capacity_type`

Include every configured capacity type. It has no identity or domain scalar
features. Its learned initial state is only the `capacity_type` schema-role
embedding.

## Relation Types And Features

Every forward relation has an explicit reverse relation with its own relation type.
Relations with the same endpoint types but different semantics remain distinct.

| Forward relation | Meaning | Edge attributes |
|---|---|---|
| `decision -> order: current_order` | Order being scored | none |
| `decision -> orderer: current_orderer` | Acting product stream | none |
| `decision -> producer: candidate` | Eligible output candidate | none |
| `orderer -> order: owns` | Queue/source ownership | none |
| `orderer -> product: stream_for` | Fixed stream product | none |
| `producer -> orderer: source_of` | Producer-funded stream source | none |
| `order -> product: requests` | Requested product | none |
| `producer -> product: can_make` | Authoritative eligibility | none |
| `product -> capacity_type: requires_capacity` | Recipe capacity requirement | units per output unit |
| `product -> product: requires_input` | Direct recipe input | input units per output unit |
| `producer -> capacity_type: capacity_state` | Owned per-type capacity | current, max, reserved, free, headroom, prior requested load, degradation |
| `producer -> product: inventory_state` | Structurally relevant input stock | inventory units, outstanding inbound units |
| `producer -> product: telemetry_state` | Past-cycle performance | routed, completed, rejected, failed EMAs; completion and failure latency |
| `order -> producer: pending_bid` | Already routed in the current auction | quantity, tokens, routed age |
| `order -> producer: reservation` | Accepted bundle waiting for inputs | quantity, tokens, cycles waited |

The candidate edge has no stream-index attribute. `stream_index` is retained only in
artifact metadata so candidate labels can be audited against the legacy flat action.

### Structural zero relations

Absence of a relation must mean "not applicable," not "currently zero."

- Create `inventory_state` edges for every structurally storable input product of a
  producer, even when inventory and inbound values are zero.
- Create `telemetry_state` edges for every `can_make` pair, even when all telemetry
  values are zero.
- Create `capacity_state` edges for every capacity type the producer owns, including
  a currently depleted type.

This prevents a zero observation from being confused with a missing relation.

### Simple local arithmetic

`free = current - reserved` and `headroom = max - current` are allowed even though
they are algebraically derived. They are local per-relation state, not topology-wide
feature engineering. Global scarcity, opportunity cost, transitive recipe counts,
and throughput summaries remain prohibited.

## Commitment Visibility

Pending bids and reservations are important because the teacher's rollout can depend
on both:

- Earlier orderers may already have placed bids into a producer's current auction.
- Composite reservations hold capacity and have product-specific input and expiry
  consequences that total reserved capacity does not preserve.

The primary recommendation is to expose these as centralized, causal orderer state.
They exist before the current action and are therefore not future leakage.

Before implementation, this must be accepted as part of the deployable observation
contract. If deployed orderers are intentionally forbidden from observing other
orderers' pending bids or reservation details, remove those relations from
`graph_primitives_v1`, record the restriction in `graph_schema.json`, and scope any
negative result accordingly. Do not read private simulator internals only for the
offline probe and later call the graph deployable.

The clean implementation is a domain-owned graph economy view, analogous to
`OrdererEconomyView`, rather than an evaluator reaching arbitrarily into the
environment.

## Identity And Topology Isolation

Domain IDs are permitted only in:

- topology compiler lookup maps;
- `decisions.csv` and `candidates.csv` audit metadata;
- split keys; and
- reconstruction checks.

They are prohibited from:

- node or edge feature matrices;
- embedding-table indexes;
- normalization group keys;
- sort-position features;
- producer columns or stream indexes presented to the model; and
- topology-specific learned parameters.

Serialized nodes receive contiguous local indexes per node type. Reordering those
indexes while applying the same permutation to edge indexes must permute candidate
outputs and leave their values unchanged.

Adding or removing nodes changes graph arrays and offsets, never parameter shapes.

## Feature Transforms

Use the following pipeline:

1. Replace undefined semantic values with zero only when accompanied by a defined
   mask, such as cut-off price.
2. Apply `log1p` to declared nonnegative count, quantity, token, capacity, and
   latency fields.
3. Leave binary and one-hot fields unchanged.
4. Fit mean and standard deviation using training graphs from the current fold only.
5. Apply those statistics to training and held-out graphs.
6. Clip only non-finite or numerically pathological values; do not cap real economy
   scale at a convenience bound.

Do not normalize separately per graph or per task snapshot. That would erase demand
and capacity scale that may be causally relevant.

## Materialized Artifact Contract

The durable dump must not depend on pickled Python objects.

Suggested layout:

```text
dump_graph/
  decisions.csv
  candidates.csv
  graph_schema.json
  topologies.jsonl
  graph_shard_000.npz
  graph_shard_001.npz
  ...
  spec.json
```

`graph_schema.json` records:

- schema name and version;
- node and relation type registry;
- ordered feature names and transforms for each type;
- reverse-relation registry;
- visibility decision for commitments;
- dtype conventions; and
- schema hash.

`topologies.jsonl` records static local-index mappings and static edges. Raw IDs may
appear in the audit mapping but the loader must return them separately from model
tensors.

Each NPZ shard stores numeric arrays plus offsets:

- dynamic node features by node type;
- dynamic and static edge indexes by relation type;
- edge attributes by relation type;
- graph pointers;
- candidate producer local indexes;
- decision, current-order, and target-product local indexes; and
- decision/candidate row offsets.

`candidates.csv` stores labels and evaluation-only actor scores. Neither is placed in
NPZ feature arrays.

The loader materializes one graph per decision and validates every candidate offset
against `candidates.csv`.

## Primary GNN

Pre-register one primary model to avoid choosing architecture after seeing held-out
results:

- four relation-aware message-passing layers;
- hidden size 64;
- learned schema-role embeddings only;
- type-specific node feature encoders;
- relation-specific edge-attribute MLPs;
- relation-specific message transforms;
- sum aggregation;
- residual updates and layer normalization; and
- one shared scalar candidate head.

Four layers are chosen because the tested recipe depth is three. They can propagate
context through paths such as:

```text
current order -> target product -> input product -> source producer -> capacity
```

Sum aggregation preserves economically meaningful totals such as queued quantity and
competing capacity. Mean-only aggregation would make different numbers of equal
orders indistinguishable.

For candidate `p`, the shared head consumes:

```text
[decision embedding,
 current-order embedding,
 target-product embedding,
 candidate-producer embedding,
 candidate relation embedding]
```

The head is applied independently to every candidate edge after joint message
passing. Therefore candidate ordering is permutation-equivariant and candidate-set
context can still flow through the decision node.

The primary target is the decision-centered teacher label. The primary loss is the
mean of per-decision candidate MSEs so decisions with many candidates do not receive
more weight.

## Required Tests

### Schema and extraction

- Minimal environment emits every required node and relation type.
- All feature arrays match the ordered names in `graph_schema.json`.
- Current order, current orderer, target product, and candidate indexes resolve to
  the intended domain objects.
- All live orders appear exactly once.
- Structural zero relations are present.
- Pending bids and reservations, if licensed, reconstruct the live domain state.

### Causality and leakage

- Extraction is invoked before teacher labeling and before the current orderer step.
- Extracting a graph does not mutate the environment.
- Teacher labels, actor scores, task IDs, seeds, raw IDs, and result fields cannot be
  found in model tensors.
- Current-cycle post-action auction results are absent.
- Telemetry equals the folded past-cycle state and excludes current-cycle
  accumulators.

### Alignment

- Candidate rows are in one-to-one correspondence with eligible producer edges.
- Candidate local indexes map to the same producers as legacy eligible columns.
- Labels and actor-floor decisions reproduce the existing dump.
- `decisions.csv` is byte-identical to the EXP-0038 dump.

### Topology independence

- Permuting each node type preserves predictions up to candidate permutation.
- Relabeling producer, product, capacity, order, and orderer domain IDs preserves
  predictions.
- Adding or removing producers changes graph size without changing model shape.
- Adding or removing products or capacity types changes graph size without changing
  model shape.
- A batch containing s29 and s77 graphs runs through one model instance.

### Serialization

- NPZ round-trip preserves every numeric tensor exactly.
- Offsets isolate neighboring graphs in a shard.
- Loading graphs in a different shard order does not change predictions.
- No Python pickle is required to reconstruct model inputs.

## Experimental Interpretation

A pass means relational decision-time state contains held-out signal for this
teacher under this schema and probe. It licenses a later deployed GNN orderer
experiment.

A failure means `graph_primitives_v1` plus the pre-registered relational probe did
not license this target. It does not prove that no possible observation can predict
the teacher.

A pass does not establish unseen-topology transfer, deployed rollout improvement, or
policy safety. Those require separate experiments.
