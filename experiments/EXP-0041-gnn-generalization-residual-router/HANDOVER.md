# EXP-0041 Handover - GNN Generalization And Residual Routing

## Plain Goal

EXP-0039 answered the capacity question but failed the generalization question:

```text
The relational MPNN can fit the teacher on training trajectories, but its candidate
scores fall to random on a held-out trajectory from the same topology.
```

EXP-0041 asks whether that failure can be repaired with the smallest defensible set
of changes:

1. train on many independent trajectories instead of two;
2. pool the three task topologies into one identity-free MPNN fit;
3. use an untouched validation/test split and early stopping;
4. compare against the old row actor on the exact same held-out protocol; and
5. deploy a passing GNN only as a bounded, confidence-gated residual over
   `blend_0.5`, never as hard single-producer routing.

EXP-0041 is conditional on EXP-0040. It must not generate labels or spend pod compute
until EXP-0040 mechanically determines whether the target is the original single
future suffix or a continuation average.

## What EXP-0039 Did And Did Not Show

The current model is already a relational heterogeneous MPNN:

- type-specific node encoders;
- relation-specific forward and reverse messages;
- sum aggregation;
- residual node updates and layer normalization;
- one scalar score per eligible candidate producer.

It reached approximately `0.92-0.999` inferred in-sample r2 across its training folds,
so width and optimization capacity were not the immediate limit. Its effective sample
size was small, however: every fold trained on only two independent trajectories.
Thousands of adjacent decision graphs from two episodes are not thousands of
independent samples.

The EXP-0039 canonical actor floor also does not prove that the row actor generalizes
better. The residual actor was trained on the same registered trajectories on which
its teacher regret was later reported, while the GNN was evaluated leave-one-seed-out.
EXP-0041 therefore retrains the row actor with the exact same train/validation/test
episodes as the GNN.

Finally, EXP-0039 used `argmax` only to score candidate ranking offline. It did not
deploy a hard GNN router. A deployed EXP-0041 candidate must retain the split-routing
behavior of `blend_0.5`.

## Experiment Question

```text
Given a target licensed by EXP-0040, can one pooled, regularized relational MPNN learn
producer preferences that transfer to independent trajectory seeds within the same
three topologies, and can its bounded soft residual improve closed-loop delivery over
blend_0.5 without increasing failure risk?
```

This experiment still does not test a completely unseen topology. Topology transfer
is a later gate and is forbidden until within-topology trajectory transfer passes.

## Hard Dependency On EXP-0040

Read `EXP-0040/RESULT.md` and its artifact-derived classification before selecting a
target mode:

| EXP-0040 classification | EXP-0041 action |
|---|---|
| `MEAN_TARGET_LICENSED` | Use continuation-mean labels. Set `K` to the smallest value in `{4,8,16}` that met EXP-0040 convergence against the full ensemble. Persist the original-suffix label as a diagnostic control. |
| `COVERAGE_PATH_LICENSED` | Use the original single-suffix label. Do not average labels. |
| `AMBIGUOUS` | Block EXP-0041. Pre-register a narrower target audit first. |
| `SEARCH_ONLY` | Cancel EXP-0041. Move to bounded rollout/search. |

The implementing agent may build and test target-mode plumbing before the EXP-0040
read, but must not choose a mode, generate the real dataset, or launch a pod early.

### EXP-0040 fork precondition

Before using `fork_future_demand`, correct the Step-1 regression-test gap identified
in review: graph equality must recursively compare every `node_features`,
`edge_index`, and `edge_attr` array plus scalar anchors and candidate arrays. The
current test checks only top-level NumPy fields and skips the graph dictionaries.

## Frozen Inputs

Reuse the EXP-0039 environment contract:

- task IDs: registered s29, s77, and s45 topology instances;
- graph schema: `graph_primitives_v1`;
- schema hash: `9918b12f69760bcd`;
- producer: EXP-0035 `producer_candidate.pt`;
- behavior and teacher continuation policy: `blend_0.5`;
- marginal bump: `delta=1.0`;
- lookahead: `8` cycles;
- episode horizon: `100`;
- only contested decisions with at least two eligible producers;
- model inputs contain no task ID, trajectory seed, cycle ID, order ID, producer ID,
  serialized position, or teacher/result field.

The graph schema is frozen. Do not add attention, new observations, actor scores,
future schedules, or identity embeddings in this experiment.

## Episode Splits

Use new seeds that were not part of EXP-0037 through EXP-0040. The final test set is
not the repeatedly inspected EXP-0039 anchor set.

| Task | Nested train-2 | Nested train-4 | Primary train-8 | Validation | Final test |
|---|---|---|---|---|---|
| s29 | 32, 33 | 32-35 | 32-39 | 129-131 | 229-231 |
| s77 | 80, 81 | 80-83 | 80-87 | 177-179 | 277-279 |
| s45 | 48, 49 | 48-51 | 48-55 | 145-147 | 245-247 |

Rules:

- Split by whole episode, never by decision row.
- No normalization statistic, early-stopping decision, architecture choice,
  residual scale, confidence threshold, or label-budget choice may read final-test
  rows.
- Train sets are nested exactly as shown so the `2 -> 4 -> 8` learning curve changes
  only independent episode count.
- The source trajectory for every split is always driven by `blend_0.5` in the
  primary experiment.
- Report every metric by task and episode seed. Aggregate decision rows only after
  the per-seed table exists.

## Dataset And Labels

Materialize one graph per contested routing decision using the EXP-0039 causal
snapshot boundary. Each candidate row records:

```text
task, split, trajectory_seed, cycle, order_id, decision_digest,
candidate_index, producer audit ID, original_suffix_label,
primary_label, target_mode, continuation_count
```

Raw IDs remain audit metadata and split keys, never model tensors.

If the mode is `single_suffix`, `primary_label = original_suffix_label`.

If the mode is `continuation_mean`:

1. fork the exact pre-action state with EXP-0040's helper;
2. use identical future demand for the base and all candidate bumps in one
   continuation;
3. average raw candidate labels over the licensed `K` continuation IDs; and
4. retain every raw continuation label so the mean is independently recomputable.

Flat decisions are determined from the primary target. Exclude primary-target-flat
decisions from model training and primary offline metrics, but count and report them
by task/split/seed. Never decide flatness from predictions.

## Primary Learning Change

EXP-0039 fitted one model per task. EXP-0041's primary fit pools all three tasks into
one MPNN. The model still receives no task identity. Pooling forces it to learn
relations that work across different producer/order counts and gives the fit 24
independent training episodes at train-8 rather than two episodes per fold.

Use one global fold normalizer fitted only on pooled train graphs. Standardize
declared continuous node/edge columns; leave binary columns unchanged. Do not fit
per-task normalizers.

Minibatches must keep complete decisions intact and balance tasks:

- default: 32 decisions per task per batch, 96 total when all tasks have rows;
- deterministic task-local shuffle from `(model_seed, epoch)`;
- single-thread CPU math and `num_workers=0` for the canonical read;
- task-balanced loss: mean candidate MSE within decision, mean decisions within
  task, then equal mean across tasks.

## Model Arms

Train every arm with model seeds `{0,1,2}` on the primary train-8 split.

| Arm | Definition | Role |
|---|---|---|
| `row_mlp_h32` | Existing seven-feature `ResidualScoreModel`, hidden 32, retrained on the exact pooled splits and primary target | Fair held-out actor baseline; not eligible as the GNN winner |
| `graph_0hop_h32` | Graph node encoders and candidate head, zero message-passing layers, hidden 32 | Tests whether graph propagation adds held-out value |
| `mpnn_2x32` | Relational MPNN, 2 layers, hidden 32 | Lower-capacity relational candidate |
| `mpnn_4x64` | Current EXP-0039 relational MPNN, 4 layers, hidden 64 | Current-architecture candidate |
| `mpnn_4x64_per_task` | Current architecture fitted separately per task | Diagnostic only: isolates whether pooled training helps |

All graph arms share the same schema, batches, primary labels, loss, optimizer, and
validation protocol. Do not introduce an attention model, graph transformer, or
ranking-loss arm after seeing validation or test results.

For the episode-count learning curve, additionally fit pooled `mpnn_4x64` on the
nested train-2 and train-4 sets. The train-8 fit above completes the curve.

If EXP-0040 selects `continuation_mean`, train one pooled `mpnn_4x64` single-suffix
control on the same train-8 graph rows. It is diagnostic and cannot replace the
pre-registered primary target after test results are visible.

## Training Recipe

Keep the EXP-0039 decision-centered MSE objective so the experiment does not change
data, target, architecture, and loss simultaneously.

```text
max_epochs = 300
min_epochs = 50
validation_every = 5 epochs
early_stop_patience = 25 validation checks
learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.10
gradient_clip_norm = 1.0
model_seeds = 0,1,2
```

Center labels and predictions within each decision. Model outputs remain relative
candidate scores, not calibrated total consumer units.

Select each checkpoint using validation decision-weighted mean regret. Validation r2,
top1, and loss are reported but do not replace the selection metric. Test data is read
only after all three model-seed checkpoints, the graph-arm selection, and the residual
adapter selection are frozen and persisted.

### Graph-arm selection

Using ensemble-mean validation predictions:

1. discard an arm unless centered r2 is positive on at least 2 of 3 validation tasks;
2. among remaining arms, choose the lowest mean relative regret versus the validation
   random floor;
3. if two arms are within 1% relative regret, choose the smaller arm;
4. if no graph arm clears step 1, offline Gate A fails without reading test labels.

The selected graph arm is the only arm eligible to license residual deployment. All
other test metrics remain diagnostics.

## Required Offline Metrics

For train, validation, and final test, emit by model seed, task, and trajectory seed:

- real-choice decision count and flat-decision count;
- decision-centered MSE and r2;
- tie-aware top1 and winner-set overlap;
- decision-weighted teacher regret;
- uniform-random regret;
- fair `row_mlp_h32` regret;
- prediction dispersion across model seeds;
- train-to-validation and train-to-test generalization gaps.

Persist candidate-level predictions for every arm and split. The complete offline
verdict must be recomputable from CSV without checkpoints or the simulator.

## Offline Gate A

The selected pooled graph arm licenses a rollout adapter only if all conditions hold
on final-test episodes:

1. centered r2 is positive on at least 2 of 3 tasks;
2. mean regret is at least 10% below the uniform-random regret on at least 2 of 3
   tasks;
3. mean regret beats the fairly held-out `row_mlp_h32` on at least 2 of 3 tasks;
4. no task is worse than the row baseline by more than 5%;
5. conditions 1 and 2 hold for the ensemble mean and independently for every model
   seed; and
6. every expected task/seed is present with finite predictions and exact candidate
   alignment.

If Gate A fails, stop. Do not soften the gate, tune a ranking loss on test results, or
run a deployed GNN policy.

The `2 -> 4 -> 8` curve is diagnostic, not a loophole. A rising curve does not license
deployment if train-8 misses Gate A.

## Residual Routing Adapter

After graph-arm selection, build and select the conservative routing adapter on
validation episodes, then freeze it before reading any final-test label or rollout.
Gate A still controls whether that already-frozen adapter may be executed on final-test
episodes. For each decision:

```text
r_m       = center(score from model seed m over eligible producers)
r_mean    = mean(r_0, r_1, r_2)
unanimous = all three models have overlapping top winner sets
confident = unanimous and top_two_margin(r_mean) >= margin_threshold
residual  = alpha * clip(r_mean, -1.0, 1.0) if confident else 0
final     = blend_0.5_logits + residual
action    = existing masked soft routing decoder(final)
```

This is explicitly not `argmax`. It preserves routing mass on multiple producers.
When models disagree or the margin is small, the policy is byte-identical to
`blend_0.5` for that decision.

Select `(alpha, margin_threshold)` using validation rollouts only:

```text
alpha in {0.10, 0.25, 0.50}
margin_threshold in {0.0, 0.5, 1.0}
clip = 1.0 (fixed)
```

Selection rule:

1. reject a pair if any validation task loses more than 2% consumer units versus
   `blend_0.5` or increases deaths;
2. among survivors, choose the highest mean validation consumer units;
3. ties within 0.5% choose smaller `alpha`, then larger margin threshold;
4. if no pair beats blend on at least 2 of 3 validation tasks, stop before final-test
   rollout.

Also deploy the fair row baseline through the same adapter/grid for comparison. It
cannot substitute for a failed graph candidate.

### Immutable run order

The real run must execute this state machine mechanically:

1. train models and select checkpoints from training/validation only;
2. select the graph arm from validation predictions only;
3. select and persist the GNN and row residual adapters from validation rollouts only;
4. lock the model/normalizer/adapter manifest;
5. read final-test labels and compute Offline Gate A;
6. only if Gate A passes, execute the already-locked adapters on final-test episodes
   and compute Gate B.

No final-test result may cause a return to steps 1-4.

## Closed-Loop Gate B

Evaluate the frozen GNN ensemble and frozen adapter once on the final-test seeds. Run
the exact same episodes under `blend_0.5` and the fair row-residual policy.

The GNN residual passes only if:

1. mean consumer completed units beat `blend_0.5` on at least 2 of 3 tasks;
2. average relative improvement over blend across tasks is at least 5%;
3. no task is worse than blend by more than 5%;
4. no individual test episode is worse than its paired blend episode by more than
   10%;
5. system death count is no higher than blend on any task; and
6. every route conserves quantity, masks ineligible producers, and sends nonzero mass
   to multiple eligible producers whenever blend does and the residual remains finite.

Gate B, not train loss and not offline teacher regret, decides whether a later broader
GNN actor experiment is licensed.

## Mechanical Verdict

Emit exactly one result:

| Result | Meaning |
|---|---|
| `PASS_RESIDUAL_WITHIN_TOPOLOGY` | Gates A and B pass. The selected pooled MPNN generalizes across new trajectories in the registered topologies and improves conservative soft routing. Open a separate unseen-topology experiment. |
| `OFFLINE_ONLY` | Gate A passes but validation adapter selection or Gate B fails. Scores transfer offline but do not improve play; do not deploy. |
| `NO_GENERALIZATION` | Gate A fails. More independent trajectories plus pooled/regularized MPNNs did not recover the target. Close this supervised target under the tested contract. |
| `CANCELLED_BY_EXP0040` | EXP-0040 returns `SEARCH_ONLY`. No EXP-0041 data or training run is allowed. |

No other prose interpretation may override this field.

## Provenance And Artifact Contract

Reuse the hardened EXP-0039 controls:

- clean source commit injected by the launcher;
- whole-package source-tree SHA-256;
- archive and producer-checkpoint SHA-256 verified before label spend;
- schema hash and full graph schema in every dump;
- exact environment manifest, architecture, BLAS, Torch build, and pod image;
- graph-to-decision digest binding;
- exact split manifest and expected per-task seed sets;
- finite checks during dump, load, train, prediction, and rollout;
- raw loss histories, normalization statistics, predictions, checkpoints, rollout
  rows, commands, and logs retained;
- artifact-only gate recomputation tests.

Required layout:

```text
experiments/EXP-0041-gnn-generalization-residual-router/
  HANDOVER.md
  spec.json
  RESULT.md
  artifacts/
    data/
      decisions.csv
      candidates.csv
      continuation_labels.csv        # mean-target mode only
      graph_schema.json
      graph_shard_*.npz
      split_manifest.json
      spec.json
    probe/
      predictions.csv
      metrics_by_split_task_seed.csv
      learning_curve.csv
      architecture_selection.json
      normalization.json
      loss_history.csv
      checkpoints/*.pt
      spec.json
    rollout/
      validation_grid.csv
      test_rows.csv
      summary.csv
      adapter.json
      spec.json
    logs/
      dump.log
      probe.log
      rollout.log
```

## Test-First Implementation Order

Suggested ownership boundaries:

- `tokenomics.eval.graph_generalization_dump`: explicit split/target-mode collection
  and durable graph/label artifacts, reusing EXP-0039 encoding and provenance helpers;
- `tokenomics.train.graph_generalization_probe`: pooled minibatches, model arms,
  validation selection, predictions, Offline Gate A, and checkpoints;
- `tokenomics.eval.graph_residual_router`: ensemble adapter, validation grid, paired
  closed-loop rows, and Gate B;
- focused tests in `test_graph_generalization_dump.py`,
  `test_graph_generalization_probe.py`, and `test_graph_residual_router.py`.

Do not silently change EXP-0039 defaults or its historical artifact loader. Extract
shared helpers only when both old and new contracts remain regression-tested.

1. Strengthen the EXP-0040 graph-identity fork test noted above.
2. Add an exact split registry and tests for disjointness, nesting, task coverage,
   and whole-episode grouping.
3. Generalize the graph collector to explicit seed sets and conditional target modes;
   test single-label parity and continuation-mean recomputation.
4. Add pooled, task-balanced decision minibatches; test that no decision is split and
   each task contributes equal loss weight.
5. Parameterize `RelationalGNN` layers/hidden/dropout, including zero-layer forward;
   retain permutation, relabeling, variable-size, and mixed-topology tests.
6. Add validation-only early stopping and prove test rows cannot enter normalizers,
   checkpoint selection, architecture selection, or adapter selection.
7. Retrain the row MLP through the identical split/target/metric path.
8. Implement artifact-only Offline Gate A and adversarial false-pass tests.
9. Implement the ensemble residual adapter; test centering, clipping, unanimity,
   margin fallback, eligibility, conservation, multi-producer routing, and exact
   `alpha=0`/not-confident parity with blend.
10. Implement validation-grid selection without test access.
11. Implement artifact-only Gate B and adversarial per-task/per-seed safety tests.
12. Run a tiny two-task, two-seed smoke locally and independently recompute all
    emitted metrics.
13. Review code and smoke artifacts before any pod spend.
14. After EXP-0040 licenses the target and the user approves spend, commit the clean
    source and run dump -> probe -> conditional rollout on the registered CPU image.
15. Pull all artifacts and logs, recompute both gates from CSV, then write the result.

## Stop Conditions

- EXP-0040 is incomplete, `AMBIGUOUS`, or `SEARCH_ONLY`.
- The future fork changes any current graph/state field.
- Any overlap exists between train, validation, and test trajectory seeds.
- Any task/seed/candidate/digest/input/source/environment alignment check fails.
- A model tensor contains a domain identity, task ID, seed, cycle, or result field.
- Any non-finite graph feature, target, loss, prediction, residual, or action appears.
- No graph arm satisfies validation selection without reading test rows.
- Offline Gate A fails.
- No residual adapter clears the validation safety rule.
- Closed-loop Gate B fails.

Do not respond to a stop by adding features, widening the MPNN, weakening a threshold,
or reading the final test again. A new hypothesis requires a new experiment.

## Compute Posture

Real label generation is expected to dominate cost. The implementing agent must first
time a two-decision smoke for both target modes and write a projected CPU-core-hour and
artifact-size budget. No pod launch is authorized by this handover. Code and local
smoke implementation may proceed; cloud spend requires a separate explicit go-ahead.
