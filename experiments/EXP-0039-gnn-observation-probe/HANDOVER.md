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
- No use of `task_id`, `seed`, or raw producer/order IDs as model features.
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

### Node Types

- `decision`: one anchor node for the current routing decision.
- `order`: live order nodes, including the current order. Attributes: unresolved quantity, age,
  remaining TTL, is_current_order, same-orderer flag if available.
- `orderer`: ordering agents. Attributes should be minimal; include ownership / current-orderer
  edges rather than IDs.
- `producer`: all live producers, not only eligible producers. Candidate eligibility is an edge or
  mask.
- `product`: product nodes. Do not encode product ID as a scalar feature. Use structure and learned
  type embeddings.
- `capacity_type`: capacity type nodes. Do not encode capacity ID as a scalar feature.

Optional only if it stays primitive and causal:

- `cycle_clock`: normalized time-to-horizon / current cycle. Leave out in the first run unless a test
  explicitly proves it is not acting as seed memorization.

### Edge Types And Primitive Attributes

Use directed edges, and either add reverse edges explicitly or let the GNN implementation mirror them.

- `decision -> order`: current order link.
- `decision -> producer`: eligible candidate link. Attributes: candidate mask / stream index if
  needed for reconstruction. Do not include teacher label or actor score.
- `order -> product`: requested product. Attributes: unresolved quantity, original quantity, age,
  TTL remaining, is_current.
- `orderer -> order`: ownership / queue membership.
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

- Relation-aware message passing, 2 to 4 layers.
- Hidden size 64 or 128.
- Type embeddings for node and edge types.
- Edge attributes encoded by small MLPs.
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

1. Add graph dump code.
   - Prefer a new module, e.g. `tokenomics.eval.graph_value_router_probe` or
     `tokenomics.eval.graph_value_router_dump`, rather than growing `value_router_errors.py` too much.
   - Reuse EXP-0038 label collection path so the teacher, trajectory, and actor floor stay identical.
   - Write `decisions.csv` and candidate labels in a familiar shape for auditability.
   - Store graph arrays in chunked JSONL or NPZ with offsets. Do not rely on pickled Python objects
     as the only durable artifact.

2. Add graph schema tests before the pod run.
   - Minimal env produces expected node/edge types.
   - Candidate producer indexes align with candidate labels.
   - Graph extraction happens before the current order is routed.
   - No teacher/actor/result columns appear in graph features.
   - Producer order permutation does not change predictions except for the same permutation of
     candidate outputs.
   - Product/capacity relabeling does not leak through scalar IDs.

3. Add the supervised GNN probe.
   - Deterministic train seeds.
   - Train-fold-only normalization.
   - Leave-one-seed-out splits.
   - Writes per-fold metrics and pooled per-task metrics.

4. Run a local smoke.
   - Small generated archive, horizon 10 to 20.
   - Verify files are written and metrics can be computed.
   - Do not interpret smoke metrics.

5. Run the real s29/s77/s45 graph dump and probe on a pod.
   - CPU is acceptable for dump generation.
   - Use GPU only for GNN training if local/pod CPU is too slow.
   - Pull all artifacts into this EXP folder.

6. Write `RESULT.md`.
   - Keep the conclusion scoped to the tested graph schema and probe model.
   - If the gate fails, say: graph-structured deployable state did not license this target.
   - If the gate passes, say: open a follow-up experiment to train/deploy a GNN orderer; do not claim
     deployment success from the probe alone.

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
