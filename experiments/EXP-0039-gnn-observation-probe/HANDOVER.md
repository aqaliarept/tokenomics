# EXP-0039 Handover - GNN Observation Probe

## Current Status - 2026-07-29

EXP-0038B tested the richer deployable tabular observation path and failed the corrected routing
gate. Phase 1 added per-producer/product metrics (`pp_*` + `h_*`). Phase 2 added system-wide
demand/capacity/recipe/cutoff/other-actor context (`sys_*`). On the same s29/s77/s45 trajectory,
every tested block lost to the deployed actor floor on all 3 tasks for both ridge+interactions and
kNN.

This does not prove that no deployable observation can help. It proves that the hand-declared metric
ladder did not make the marginal-value teacher's within-decision producer preference learnable.

EXP-0039 is the next, narrower question:

```text
Can a graph model over raw deployable economy state recover the teacher preference that the
hand-metric ladder missed?
```

Do not train or deploy a new orderer yet. Start with a supervised GNN representability probe. If the
probe fails, stop this branch and treat the marginal-value target as rollout/emulation-shaped. If the
probe passes the gate, open a later experiment for an actual GNN orderer.

## Plain Goal

A GNN will not magically discover facts from nothing. It still needs a declared graph ontology:
producers, products, capacity types, live orders, orderers, recipe edges, capacity edges, inventory
edges, and telemetry edges.

The point is different from EXP-0038B: do not hand-compress those relations into metrics like HHI,
scarcity ratios, transitive counts, or max product throughput. Give the model primitive,
decision-time facts and let message passing learn useful relational summaries.

## Question

Can a deployable graph observation, scored by a GNN probe, predict the EXP-0037 marginal-value
teacher's within-decision candidate ranking better than the incumbent actor floor?

Subquestions:

- Was EXP-0038B a feature-engineering failure, where the useful signal exists in relational state but
  our hand metrics compressed it badly?
- Or is the teacher preference mainly produced by the 8-cycle rollout itself, so no practical
  decision-time observation model should be trained on this target?

## Non-Goals

- No full GNN orderer training in this experiment.
- No deployed rollout gate for a GNN policy yet.
- No teacher-rollout outputs, future outcomes, post-decision auction results, or labels as input
  features.
- No use of `task_id`, `seed`, or raw producer/order/product/capacity IDs as model features.
- No learned identity embeddings for individual producers, products, capacity types, orderers, or
  orders. Learned embeddings are allowed only for schema roles such as node type and edge type.
- No actor-score feature in the primary graph. Actor scores remain an evaluation floor. A separate
  `graph_plus_actor_score` ablation is allowed only after the primary probe is read.

## Inputs

Use the same gate setup as EXP-0038 / EXP-0038A / EXP-0038B:

- Archive: `tokenomics/_pod_inputs/gate_archive.jsonl`
- Producer checkpoint: `experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt`
- Base orderer/floor trajectory: `blend_0.5`
- Teacher: `marginal_value_labels(delta=1.0, lookahead=8)`
- Eval seeds: 3 anchored seeds per task
- Horizon: 100
- Tasks:
  - `mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29`
  - `mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77`
  - `mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45`

The graph dump must reproduce the same decision trajectory as EXP-0038:

- `13,940` decisions total
- `37,804` eligible producer candidate rows
- `decisions.csv` byte-identical to `experiments/EXP-0038-actor-teacher-error-microscope/dump/decisions.csv`

## Graph Observation Contract

Build one graph per routing decision. The model outputs one score for each eligible producer
candidate for that decision.

Primary schema name:

```text
graph_primitives_v1
```

The normative schema, topology boundary, commitment-visibility decision, serialization contract,
and primary GNN are specified in [`ARCH.md`](ARCH.md). If this handover and `ARCH.md` differ,
`ARCH.md` controls graph construction; the gate and experiment scope in this handover still control
the experimental conclusion.

### Node Types

- `decision`: one anchor node for the current routing decision.
- `order`: all live order nodes, including the current order. Attributes include quantity and token
  accounting, age, remaining TTL, funding state, and order kind. Current-order and same-orderer
  context are represented structurally, not as identity-like scalar flags.
- `orderer`: ordering agents. Attributes should be minimal; include ownership / current-orderer
  edges rather than IDs.
- `producer`: all live producers, not only eligible producers. Candidate eligibility is an edge or
  mask.
- `product`: product nodes. Do not encode product ID as a scalar feature and do not attach a learned
  per-product embedding. Products must be distinguishable only through graph structure and primitive
  attributes.
- `capacity_type`: capacity type nodes. Do not encode capacity ID as a scalar feature and do not
  attach a learned per-capacity embedding. Capacity types must be distinguishable only through graph
  structure and primitive attributes.

Optional only if it stays primitive and causal:

- `cycle_clock`: normalized time-to-horizon / current cycle. Leave out in the first run unless a test
  explicitly proves it is not acting as seed memorization.

### Edge Types And Primitive Attributes

Use directed edges, and either add reverse edges explicitly or let the GNN implementation mirror them.

- `decision -> order`: current order link.
- `decision -> orderer`: current ordering-agent link.
- `decision -> producer`: eligible candidate link. Stream index may be retained as audit metadata
  but must not be an edge feature. Do not include teacher label or actor score.
- `order -> product`: requested product.
- `orderer -> order`: ownership / queue membership.
- `orderer -> product`: fixed product served by a product-stream orderer.
- `producer -> orderer`: source relation for producer-funded ordering streams.
- `producer -> product`: can-make relation.
- `product -> capacity_type`: recipe capacity requirement. Attribute: units of capacity type required
  per unit of product.
- `product -> product`: recipe input requirement. Attribute: units of input product required per
  output unit.
- `producer -> capacity_type`: current producer capacity state. Attributes: current capacity,
  max capacity, reserved capacity, free capacity, headroom. These are primitive per-type facts, not
  min/mean/max summaries.
- `producer -> product`: inventory / inbound relation. Attributes: inventory units,
  outstanding inbound units, output inventory limit if producer/product-specific.
- `producer -> product`: telemetry relation. Attributes: folded past-cycle routed/completed/rejected
  EMAs, completion rate, rejection/failure rate, latency if available.
- `order -> producer`: current-cycle pending-bid relation. Attributes: routed quantity and tokens.
- `order -> producer`: live-reservation relation. Attributes: quantity, tokens, and cycles waited.

The three `producer -> product` relations and the two `order -> producer` relations are distinct
typed relations despite sharing endpoint node types. Include explicit reverse relations.

Pending bids and reservation details are causal but require a centralized observation contract.
Before implementing the dump, confirm they are deployable state and expose them through a
domain-owned graph view. If they are intentionally private, omit them from `graph_primitives_v1`,
record that restriction in `graph_schema.json`, and scope a negative result to the reduced
visibility contract. Do not expose evaluator-only simulator internals and call them deployable.

Allowed normalization:

- `log1p` for nonnegative counts and quantities.
- Train-fold-only standardization of continuous attributes.
- Clipping only to guard non-finite values, not to flatten real demand/capacity scale.

Avoid precomputing these in the primary graph:

- HHI
- global scarcity ratios
- transitive input counts
- downstream demand unlocked
- max target throughput
- opportunity-cost summaries

Those were the EXP-0038B hand metrics. The GNN is meant to learn such compositions from primitive
relations if they are learnable.

## Model Probe

Start simple and deterministic:

- Primary model: 4 relation-aware message-passing layers and hidden size 64. This is pre-registered
  because the tested recipe topology is three layers deep.
- Type embeddings for schema node types and edge types only. These are role embeddings
  (`producer`, `product`, `capacity_type`, `producer -> product`, etc.), not identity embeddings for
  individual products, producers, or capacity types.
- Edge attributes encoded by small MLPs.
- Sum aggregation so repeated orders and capacity contributions remain countable.
- Residual updates and layer normalization.
- Candidate head scores each eligible `decision -> producer` candidate from:
  - current decision embedding
  - candidate producer embedding
  - current order / target product embedding if convenient
  - candidate edge embedding

Acceptable implementations:

- PyTorch Geometric hetero graph if the pod environment can install it cleanly.
- A minimal pure-Torch relational message-passing implementation if PyG install cost is high.

Do not let implementation complexity hide the experimental question. The probe only needs enough
capacity to test relational representability, not a production architecture.

## Training Target

Use the same label semantics as EXP-0038B:

- One label per eligible candidate producer: teacher marginal value.
- Fit decision-centered labels by default: label minus the mean label within that decision.
- Train and score real-choice decisions primarily: decisions with at least two eligible producers
  and non-flat teacher labels.
- Flat decisions may be included as zero-centered training examples only if decision weighting
  prevents them from dominating. Report whether they were included.

Primary loss:

```text
decision-weighted MSE on decision-centered candidate labels
```

Secondary allowed if MSE underfits but ranking is plausible:

- pairwise margin loss within each decision
- listwise softmax / cross-entropy over teacher-centered labels

Do not change the gate because a secondary loss looks better. The final read is still held-out
routing regret against the actor floor.

## Split And Gate

Use the corrected EXP-0038A/B gate:

- Leave-one-seed-out per task.
- For each task, train on two anchored seeds and hold out the third; pool held-out decisions across
  folds so every decision counts once.
- Score only real-choice decisions.
- Decision-weighted metrics.
- Tie-aware top1.
- Compare to the same deployed actor floor from the dump.

Pass condition for licensing a later GNN orderer experiment:

- GNN mean regret beats the deployed actor floor on at least 2 of 3 tasks.
- Centered within-decision r2 is positive on at least 2 of 3 tasks.
- Average relative regret reduction vs actor floor is at least 10 percent across the three tasks.
- No remaining task is worse than actor floor by more than 5 percent.
- Result is stable across at least 3 training seeds, or instability is explicitly reported and treated
  as no pass.

If the GNN only beats random but not the actor floor, the experiment fails. The incumbent actor floor
is the licensing bar.

## Required Baselines

Report these in the same table as the GNN:

- random floor
- deployed actor floor
- current-row ridge/MLP if cheap
- EXP-0038B full tabular `producer_product_system_v1` probe numbers, copied from `audit_sys/`
- optional `graph_plus_actor_score` ablation after the primary graph read

The key comparison is GNN vs deployed actor floor, not GNN vs random.

## Implementation Plan

### Phase 0 - Freeze Visibility And Schema

1. Review `ARCH.md` against the deployed observation boundary.
   - Decide whether other orderers' current-cycle pending bids are centrally observable.
   - Decide whether live reservation product, producer, quantity, and age are centrally observable.
   - Record both decisions in `graph_schema.json`; do not silently read evaluator-only state.
2. Freeze the ordered node-feature and edge-feature registries.
   - Assign every semantic relation a distinct relation name.
   - Declare `log1p`, binary, defined-mask, and standardization treatment per feature.
   - Compute a schema hash and store it in every dump/probe `spec.json`.
3. Add a domain-owned graph economy view.
   - Follow the existing `OrdererEconomyView` ownership pattern.
   - Keep domain object assembly free of numpy, torch, serialization, and labels.
   - Keep topology compilation independent of task generator, task ID, and fixed entity counts.

Exit condition: a reviewed schema fixture can be constructed from a minimal `Configuration`, and
every field has a documented causal source and visibility decision.

### Phase 1 - Implement Static Topology Compilation

1. Add a focused module such as `tokenomics.eval.graph_value_router_dump`; do not grow
   `value_router_errors.py` into a graph/model module.
2. Compile all configured producers, products, capacity types, and product-stream orderers into
   per-type local indexes.
3. Compile static recipe, eligibility, capacity ownership, stream-product, and producer-source
   relations plus explicit reverses.
4. Keep raw domain IDs in a separate audit mapping only. Model-facing arrays contain local indexes,
   relation types, and numeric primitive attributes.
5. Validate the known task sizes:
   - s29: 12 products, 6 producers, 3 capacity types, 32 orderers.
   - s77/s45: 12 products, 9 producers, 3 capacity types, 42 orderers.

Exit condition: s29, s77, and s45 topologies compile with one schema and can be batched without
changing a model parameter shape.

### Phase 2 - Implement Causal Decision Snapshots

1. Extract one graph for each incoming order with at least two eligible producers.
   - Extract while the environment is at the ordering agent's turn.
   - Extract before the teacher call and before `env.step(base(...))`.
   - For several incoming orders, reuse the same pre-action state and change only the current anchors
     and candidate set.
2. Add every live `system.orders` order with quantity, tokens, lifecycle accounting, type, funding,
   age, and TTL.
3. Overlay producer capacity, inventory, inbound, and folded past-cycle telemetry.
4. Overlay pending-bid and reservation edges if Phase 0 licenses their visibility.
5. Add current-order, current-orderer, ownership, requested-product, and candidate edges.
6. Generate labels only after graph extraction.
   - Reuse `marginal_value_labels(delta=1.0, lookahead=8)`.
   - Preserve the exact `blend_0.5` trajectory and actor floor.
   - Store teacher labels and actor scores only in evaluation tables.

Exit condition: snapshot extraction is side-effect free, precedes labeling/action, and graph tensors
contain no task, seed, raw ID, actor, teacher, or result values.

### Phase 3 - Build Durable Artifacts

1. Write audit tables:
   - `decisions.csv`, byte-identical to the EXP-0038 dump.
   - `candidates.csv`, one row per eligible candidate with label and evaluation-only actor score.
2. Write `graph_schema.json` with ordered feature registries, transforms, reverse relations,
   visibility decisions, dtypes, and schema hash.
3. Write static topology mappings separately from dynamic decision arrays.
4. Write chunked NPZ graph shards with explicit graph, node, edge, and candidate offsets.
5. Implement a loader that reconstructs one model graph without Python pickle.
6. Round-trip every array and verify candidate indexes against `candidates.csv`.

Exit condition:

```text
13,940 decisions
37,804 eligible candidates
byte-identical decisions.csv
zero graph/candidate alignment errors
```

### Phase 4 - Add Tests Before Any Real Dump

1. Schema tests:
   - Minimal environment produces every required node and edge type.
   - Ordered feature names exactly match array columns.
   - Structural zero inventory/telemetry/capacity edges are present.
2. Causality tests:
   - Extraction occurs before the current order is routed.
   - Extraction does not mutate the environment.
   - Current-cycle post-action results and telemetry accumulators are absent.
   - Teacher, actor, task, seed, and result fields cannot enter model tensors.
3. Alignment tests:
   - Candidate producer indexes align one-to-one with candidate labels.
   - Current order, owner, target product, and eligible producers map to legacy rows.
4. Equivariance and topology tests:
   - Permuting any node type only permutes corresponding outputs.
   - Relabeling all domain IDs preserves predictions.
   - Adding/removing producers, products, or capacity types changes graph size, not model shape.
   - Mixed s29/s77 graph batches execute in one model.
5. Serialization tests:
   - NPZ round-trip is exact.
   - Shard boundaries and offsets do not mix graphs.
   - Shard load order does not change predictions.

Exit condition: focused graph dump/model tests pass locally before label generation is run at scale.

### Phase 5 - Implement The Pre-Registered Probe

1. Implement the primary four-layer, hidden-64 relational GNN from `ARCH.md`.
   - Type-specific feature encoders and schema-role embeddings.
   - Relation-specific edge encoders and message transforms.
   - Sum aggregation, residual updates, and layer normalization.
   - Shared candidate head using decision, current order, target product, and producer embeddings.
2. Implement fold-local preprocessing.
   - Apply only declared `log1p` transforms.
   - Fit continuous mean/std on training graphs for that fold.
   - Reuse those statistics unchanged on the held-out seed.
3. Train decision-centered labels with decision-weighted MSE.
   - Use the predeclared deterministic optimizer/training schedule.
   - Run at least three model seeds.
   - Select checkpoints using training data only or a deterministic validation split inside the two
     training seeds; never select on the held-out seed.
4. Emit loss history, normalization statistics, per-fold predictions, and checkpoint provenance.

Exit condition: two identical runs with the same training seed produce identical predictions and
metrics within the declared deterministic tolerance.

### Phase 6 - Local Smoke

1. Generate a small legal archive and run horizon 10 to 20.
2. Exercise topology compilation, snapshot dumping, NPZ reload, one training epoch, and metrics.
3. Verify all expected files are written and contain finite values.
4. Do not interpret smoke metrics or alter the gate based on them.

Exit condition: a clean checkout can run the documented smoke command end to end.

### Phase 7 - Real Dump And Probe

1. Generate the s29/s77/s45 graph dump on the fixed EXP-0038 trajectory.
   - CPU is acceptable for graph generation.
   - Stop immediately if count, CSV identity, alignment, causality, or schema checks fail.
2. Run leave-one-seed-out training separately per task.
   - Train on two anchored seeds and hold out the third.
   - Repeat for all folds and at least three training seeds.
3. Score only corrected real-choice decisions with decision weighting and tie-aware top1.
4. Report random, deployed actor, cheap current-row probe, copied EXP-0038B full-tabular numbers,
   and the primary graph probe in the same table.
5. Read the primary result before optionally running `graph_plus_actor_score`.
6. Pull the dump, metrics, predictions, logs, schema, normalization state, and checkpoint provenance
   into this experiment folder.

Exit condition: every gate clause can be recomputed from saved candidate predictions without loading
a checkpoint or rerunning the simulator.

### Phase 8 - Result And Decision

1. Write `RESULT.md` with exact commands, commit, environment, schema hash, counts, and artifacts.
2. Report per-fold, pooled per-task, and training-seed dispersion.
3. State the topology limitation: the gate holds out trajectories, not unseen topology.
4. Apply the predeclared gate without reinterpretation.
   - Fail: graph-structured deployable state under `graph_primitives_v1` did not license this target.
   - Pass: open a follow-up experiment to train and deploy a GNN orderer.
   - Mixed or unstable: no pass unless a concrete implementation defect invalidates the run.
5. Do not claim deployed improvement from a supervised probe.

## Expected Artifacts

Suggested layout:

```text
experiments/EXP-0039-gnn-observation-probe/
  HANDOVER.md
  spec.json
  RESULT.md
  dump_graph/
    decisions.csv
    candidates.csv
    graph_schema.json
    graphs.npz or graphs.jsonl
    spec.json
  probe/
    metrics_by_task.csv
    metrics_by_fold.csv
    loss_history.csv
    spec.json
    model_seed_*.pt
```

If model checkpoints are large, put them under `artifacts/` for git-lfs rather than normal git.

## Planned Read

### Pass

The hand metrics were too compressed. A deployable graph observation contains relational signal about
the teacher's within-decision producer preference. Next experiment should train a GNN orderer and
score deployed rollout against `blend_0.5`.

### Fail

The signal still does not appear as a stable function of deployable decision-time state, even when
the state is represented relationally. Retire the marginal-value actor target for learned orderers.
Future work should use either:

- rollout/search/emulation as a compute lever, or
- a different objective whose target is directly observable and held-out learnable.

### Mixed

If the GNN wins one task or one training seed but fails the full gate, do not train an orderer. Record
the failure mode and only continue if there is a concrete bug or leakage-free model-capacity issue to
fix.

## Review Notes To Carry Forward

- EXP-0038B `audit_sys/spec.json` currently records a stale `source_dump` path
  (`dump_sys/dump` instead of `dump_sys`). Do not copy that provenance bug into EXP-0039.
- Do not repeat the overclaim "not a function of any observation." The stronger, defensible claim is
  "not learnable under the tested deployable observation schema and probe gate."
- If this probe passes, it licenses another experiment. It does not by itself prove deployed rollout
  improvement.
