# EXP-0041 - GNN Generalization And Residual Routing

**Date:** 2026-08-09 | **Status:** CANCELLED BY EXP-0040 (`SEARCH_ONLY`), NOT RUN
| **Plan:** [HANDOVER.md](HANDOVER.md) | **Spec:** [spec.json](spec.json)

## Question

Can a pooled, regularized relational MPNN transfer teacher preference to independent
trajectory seeds within the registered topologies, then improve closed-loop delivery
as a bounded confidence-gated residual over `blend_0.5`?

## Method

The full pipeline is implemented and unit/integration-tested locally at smoke scale. **No target
mode was selected, no dataset was generated, and no pod spend was made.** EXP-0040 completed on
2026-08-24 with classification `SEARCH_ONLY`; this pipeline's prerequisite contract therefore
cancels the run before label collection.

| Stage | Module | What it owns |
|---|---|---|
| 1. data | `tokenomics.eval.graph_generalization_dump` | pre-registered whole-episode split registry (8 train / 3 validation / 3 test per task, nested 2→4→8 train prefixes), explicit seed-set collection, `single_suffix` vs `continuation_mean` target, durable graph/label artifacts + provenance |
| 2. fit | `tokenomics.train.graph_generalization_probe --stage fit` | pooled task-balanced decision minibatches, the five model arms, validation-only early stopping + checkpoint + arm selection, predictions/metrics/learning curve |
| 3-4. adapter | `tokenomics.eval.graph_residual_router --stage select` | `(alpha, margin)` grid on validation rollouts, safety selection rule, content-hash manifest lock |
| 5. Gate A | `...graph_generalization_probe --stage gate-a` | the first and only read of final-test labels; artifact-recomputable offline verdict |
| 6. Gate B | `...graph_residual_router --stage test` | paired final-test rollouts vs `blend_0.5` and the row residual, route-integrity audit, closed-loop verdict |
| budget | `tokenomics.eval.graph_generalization_budget` | the handover-required two-decision timing smoke and core-hour / artifact-size projection |

Design points that the tests pin, in plain English:

- **The target is not ours to choose.** The dump refuses to spend a single label unless it is
  handed an EXP-0040 licence read from that experiment's result: the classification picks the
  mode, `AMBIGUOUS` blocks the run, `SEARCH_ONLY` cancels it, and a mean target must use the
  *smallest* continuation count of {4,8,16} that EXP-0040 actually showed converged. The licensing
  document has to identify itself (`experiment: EXP-0040`) and carry its own convergence evidence:
  K is derived from that map — there is deliberately no flag to hand one in — and certification
  re-derives it rather than believing the number the dump copied down. Every downstream stage then
  re-certifies that the dump on disk is the pre-registered one — right topologies, right episode
  seeds, frozen teacher settings, pinned archive and producer checkpoint, committed source — before
  it reads anything.
- **A certified run, not just a certified dataset.** The fit contract is checked before any
  training compute is spent: the registered arms, the three model seeds, train-8, the 2/4/8 curve,
  an unmodified training recipe, and a real 40-char commit. Both rollout stages re-check it from
  the written fit spec, so a one-seed one-epoch fit cannot call itself certified and go on to
  license an adapter and a closed-loop number. Fit, Gate A and both rollouts also compare their
  live source tree and runtime against the dump's, exactly as EXP-0039 does.
- **Splits cannot leak.** The registry is validated for disjointness, whole-episode grouping, and
  nesting, and it *rejects* any seed already spent by EXP-0037–0040 — the anchor set those reads
  were tuned against. On top of that, the fit stage reduces every out-of-split decision to its
  identity columns *before* any number is parsed, so a final-test label, feature, or continuation
  row is never read at all — not merely never returned. The test for this poisons every test-split
  label with a non-numeric token: the fit load still succeeds, a full load raises.
- **The target is what it says it is.** In `single_suffix` mode the collected decisions, graphs,
  and labels are byte-identical to the EXP-0039 dump on the same episode (parity test), so an
  EXP-0041 number can only differ from EXP-0039 because the splits/pooling changed. In
  `continuation_mean` mode every raw continuation label is persisted and the loader *re-derives*
  the mean, rejecting a dump whose primary label is not the mean of its own rows.
- **The row baseline is fair.** EXP-0039's actor floor was fitted on the trajectories it was later
  scored on. Here the 7 deployable actor features ride along in `candidates.csv` and
  `row_mlp_h32` is retrained through the identical splits, target, loss, and metric path.
- **Deployment stays soft.** The adapter never uses `argmax`: it centers each model seed's scores,
  requires the seeds to agree on a winner and the top-two margin to clear a threshold, and adds
  `alpha * clip(mean, -1, 1)` only then. Otherwise the action is byte-identical to `blend_0.5`.
  A route audit compares the decoded routing against the base routing at the same state
  (conservation, eligibility masking, preserved multi-producer split) and is a Gate B condition —
  it is demonstrably able to fail (an absurd `alpha` collapses routing onto one producer and the
  audit counts it).
- **Both gates are mechanical.** `gate_a()` reads prediction rows, `gate_b()` reads rollout rows;
  neither needs a checkpoint or the simulator, and adversarial tests confirm each individual
  shortfall (no r², no margin over random, no win over the row baseline, one unstable model seed,
  a missing task, a single 10%-worse episode, an extra death, a route violation) flips the verdict
  to fail.
- **A partial result is a different experiment, not a weaker pass.** Both gates demand exact
  coverage: the registered tasks and the registered episode seeds, every arm and model seed
  scored over the identical decision/candidate set, every learned test episode paired with its
  blend episode, and no duplicated row. Two of three episodes fails; one arm scored on fewer
  decisions fails. Crucially the expected scope *is the registry by default* — it is never
  inferred from the rows the gate happens to be holding, which is how a one-episode-per-task file
  used to certify itself as the whole final-test read.
- **The freeze is a real freeze.** The content-hash lock covers the adapter itself and the fit
  `spec.json` (the file the rollout contract is read out of), each in its own manifest root, and
  *every* recorded hash is recomputed on verification — a locked root the verifier has no
  directory for is an error, not a skip. Gate A takes the scored arm from the locked
  `architecture_selection.json`. Gate B does not trust `gate_a.json`: it re-derives the offline
  verdict from the prediction CSV whose SHA-256 that file recorded, and checks the manifest hash,
  the arm, and the dump's source fingerprint agree. `gate_b.json` is bound to `test_rows.csv` the
  same way, and the standalone `--stage verdict` recomputes both gates instead of reading their
  `pass` fields. A hand-written `pass: true` is refused before an episode is played.

### Running the local smoke

Everything — the label contract, both gates, the confidence rule, the route audit, and the
end-to-end pipeline smoke (materialize a real label dump → fit → lock → Gate A → validation grid
→ Gate B) — runs in the default suite, in ~60 s:

```
../.venv/bin/python -m pytest tests/test_graph_generalization_dump.py \
    tests/test_graph_generalization_probe.py tests/test_graph_residual_router.py \
    tests/test_future_fork.py tests/test_graph_dump.py -q      # from tokenomics/
```

(81 tests: 16 dump, 24 probe, 18 router, 9 future-fork, 14 EXP-0039 regression.)

### Deviations from the handover's suggested layout

1. Gate A writes `test_predictions.csv` / `test_metrics_by_split_task_seed.csv` / `gate_a.json`
   into its own stage directory rather than appending to `probe/predictions.csv`, so the locked
   fit directory is never mutated by the test read.
2. `manifest.json` (the content-hash lock over checkpoints, normalization, arm selection, and the
   adapter) lives in the rollout/adapter directory.
3. The EXP-0040 future-demand fork lives in `env/` rather than `domain/`: it needs RNG plus config
   legality, and `domain/` stays RNG-free per CLAUDE.md. Legality is shared with the episode
   schedule builder through `draw_schedule_entry`, so a forked suffix cannot drift.

### EXP-0040 precondition — done

The handover required strengthening the fork's graph-identity test before use. The old check
looped over top-level `numpy` attributes and therefore skipped `node_features`, `edge_index`, and
`edge_attr` — every dict that actually holds the state. It now compares those recursively plus the
scalar anchors, candidate array, and id maps, and a guard-on-the-guard test proves the check fails
when a single buried cell moves.

### Implementation review — six findings, all fixed

An independent code review of the first implementation found six defects, four of them capable of
invalidating a result. All are fixed and each now has a test that fails without the fix:

| # | Defect | Fix |
|---|---|---|
| 1 | The run-order lock was forgeable: `adapter.json`'s hash sat in a field the verifier never read, Gate A took the scored arm from the unlocked fit `spec.json`, and Gate B accepted any JSON containing `pass: true`. | The lock is keyed by root and every recorded hash is verified (an unverifiable root raises); Gate A reads the arm from the locked `architecture_selection.json`; Gate B recomputes the offline verdict from the SHA-256-bound prediction CSV and cross-checks manifest, arm, and dump source. |
| 2 | Both gates could pass on a partial test set — one episode per task, or arms scored on different decision subsets. | Exact coverage in both: registered tasks *and* episode seeds, identical decision/candidate coverage per (arm, model seed), one-for-one blend pairing, no duplicate rows. |
| 3 | Nothing mechanically licensed a real run: any target mode, any K ≥ 1, optional provenance checks, and rollouts that accepted any archive/producer/seeds/horizon. | `TargetLicense` read from the EXP-0040 result gates the dump; K must be the smallest converged value in {4,8,16}; `certify_dump_contract` and `assert_rollout_shape` pin every stage; the CLIs verify pinned inputs and a real commit by default. |
| 4 | The fit stage parsed and validated every final-test label before filtering splits. | Out-of-split decisions are reduced to identity columns before any number is parsed; proved by the poisoned-label test. |
| 5 | Missing required controls/artifacts: the mean-mode single-suffix control, per-arm final-test predictions, train flat counts, train→test gaps, rollout `spec.json`, and the emitted verdict. | All present: the control arm is fitted automatically in mean mode, Gate A scores every fitted arm and writes the gaps, train rows cover the full split, and both rollout stages emit `spec.json` plus `verdict.json` (there is also a standalone `--stage verdict`). |
| 6 | `RESULT.md` documented `pytest -m exp0041`, a marker that was never registered — the command collected nothing. | The command above runs the real files; everything is in the default suite. |

### Second review — four residual findings, all fixed

A re-review of those fixes found four ways the guarantees were still bypassable. All four are
fixed, again with a test per fix:

| # | Defect | Fix |
|---|---|---|
| 1 | The gates still *inferred* their expected scope from the rows when it was omitted, so the original one-arbitrary-episode reproductions passed; `_verified_gate_a` fell back to the same leniency when the gate document recorded no scope. | `gate_a()`/`gate_b()` default to `require_registered=True`: the scope **is** the registry (tasks, final-test episodes, model seeds 0-2), a narrower caller-supplied scope is refused as `registered_scope`, and a Gate A document that never recorded the scope it certified cannot authorize a rollout. `require_registered=False` is the explicit smoke escape. |
| 2 | `require_shape` certified the *dataset*, not the *run*: arbitrary arms, model seeds, train size, curve sizes and `--max-epochs` were accepted, and no stage compared its live source tree/runtime against the dump's. | `FIT_CONTRACT` + `certify_fit_contract()` refuse an off-contract fit before compute is spent; `certify_fit_spec()` re-checks it from the written spec in both rollout stages; fit, Gate A and the rollouts call EXP-0039's `assert_source_matches` / `assert_env_matches` against the dump. |
| 3 | `fit/spec.json` sat outside the lock although Gate B reads the registry seeds, input hashes, model seeds, source and `require_shape` out of it; `--stage verdict` trusted stored booleans. | The fit spec is a locked manifest entry; `gate_b.json` is SHA-256-bound to `test_rows.csv` like `gate_a.json`; `--stage verdict` recomputes both gates (`_verified_gate_a` / `_verified_gate_b`). |
| 4 | The licence was not authoritative: the document did not have to identify itself as EXP-0040, `--continuations` could override its converged K, and certification ignored `source_sha256` and the convergence evidence. | The artifact must stamp `experiment: EXP-0040`; the `--continuations` flag is gone — K comes from the artifact's own `convergence_by_k` and a contradicting `converged_continuations` raises; `certify_dump_contract` requires a real source file plus 64-char SHA-256 and re-derives K from the recorded evidence. |

### Third review — pooled task balance was per-batch only (fixed)

An independent feasibility review of the built pipeline found the pooled fit was not task-balanced
over an epoch. `task_balanced_batches` gives each task an equal share of every batch *while it still
has decisions left*; the tasks with more decisions then own the tail batches alone, and
`task_balanced_loss` weighted the tasks *present* equally, so those tail batches handed the largest
topology full weight. On the registered topologies s77 holds **1.43×** the decisions of s29 but drew
**2.16×** the epoch gradient weight — the pooled fit would have been defined by the largest
topology, which is what pooling exists to prevent, and nothing in the loss curve would have shown it.

Fixed by weighting, not by truncating or resampling: `epoch_task_weights` weights each task
`1 / (its batch count)`, so its total per-epoch weight is invariant to how many batches it appears
in. Sampling stays without replacement (no decision seen twice, no task truncated).

**Loss-scale check (raised in review, verified separately).** A uniform rescale of the loss is
indistinguishable from a changed learning rate, so the rebalance must not quietly inflate the
objective the frozen LR and clip norm were calibrated against. Measured, not asserted: with
equal-sized tasks the weights are **exactly** the old uniform `1/n_tasks`; on the registered
profile (102/110/145 batches per task) a batch holding every task carries **1.023×** the old loss —
the weights sum to ≥ 1 by AM-HM, with the excess set by the *spread* of the per-task batch counts,
and full batches are 70% of the epoch. The substantive change is confined to the tail batches,
where s77 used to sit alone at weight 1.0 and now carries **0.274 (3.66× less)** — that is the
correction itself. Pinned by
`test_epoch_weights_preserve_loss_scale_on_full_batches_and_only_shrink_the_tail`. The mode is
recorded in the persisted recipe as
`TRAINING["task_weighting"] = "epoch_balanced"` (`per_batch` reproduces the old behaviour), so a
result states which semantics produced it. Registered as `docs/DIAGNOSTICS.md` entry 15. Two tests
cover it: the epoch-total equality (which uniform weights fail) and the weights reaching the loss.

This is a pre-run correction — no dataset, fit, or gate read has been produced under the old
weighting, so nothing published needs revisiting.

### Compute budget (measured 2026-08-09, lower bound)

A two-decision timing smoke on the real s29 topology at `horizon=100`, `lookahead=8` projects the
42-episode run at **65,053 decisions / 176,419 candidates**:

| Target | s / decision | Core-hours (lower bound) | Scaled by the EXP-0039 producer ratio |
|---|---:|---:|---:|
| `single_suffix` | 0.061 | 1.1 | ~19 |
| `continuation_mean` K=4 | 0.187 (3.07×) | 3.4 | ~57 |
| `continuation_mean` K=8 | 0.312 (5.14×) | 5.6 | ~96 |
| `continuation_mean` K=16 | 0.564 (9.29×) | 10.2 | ~173 |

**Labels are not the whole bill — the FIT stage was never budgeted.** The table above (and, before
this, all of `graph_generalization_budget`) projects label generation only. The fit is 27 optimizer
runs (30 in mean mode) of up to 300 epochs, and it is the same order as the labels or larger.
Measured 2026-08-16 with the new `--fit-dump-dir` / `--fit-only` mode of the budget module:
single-threaded forward+backward on 96 **real** EXP-0039 graphs, `Batch` construction included,
exactly the step `train_arm` runs. Projected over the train-8 population of **11,373 non-flat
decisions** = 118 batches/epoch — 31% of decisions survive the flat filter, measured from the dump
by `nonflat_fraction()`, not assumed; budgeting on the collected count would overstate it 3×.

Measured on **RunPod CPU pod `xtyr889fxi7pml`** (16 vCPU `cpu5c`, idle, torch 2.13.0+cpu — the same
runtime the EXP-0039 dump was produced on), single-threaded as the recipe pins:

| Arm | s / 96-decision batch | s / epoch (per run) | Optimizer runs |
|---|---:|---:|---:|
| `mpnn_4x64` | 0.106 | 12.6 | 3 pooled + 9 per-task + 6 curve (at 1/4, 1/2 data) |
| `mpnn_2x32` | 0.029 | 3.4 | 3 |
| `graph_0hop_h32` | 0.004 | 0.5 | 3 |
| `row_mlp_h32` | <0.001 | 0.04 | 3 |

**0.032 core-hours per epoch across all 27 runs** → **1.6 core-h** at the `min_epochs=50` floor,
**4.8** at 150, **9.6** at the `max_epochs=300` cap. The single longest run — one `mpnn_4x64` fit at
train-8 — is **10 min at the floor and 1.0 h at the cap**, which is the wall-clock floor for the
whole stage however it is scheduled.

**So the fit is NOT the largest line item: labels (19–173 core-h) dominate it by 2–100×.** This
reverses an earlier reading in this file, and the reversal is worth recording because the failure
mode is reusable.

> **Correction (2026-08-16).** An earlier revision of this section reported 88–778 core-h for the
> fit and a 7–41 h critical path, and concluded the fit was plausibly the largest cost in the
> experiment. That was wrong. Those numbers were measured in the dev container **while the full
> pytest suite was saturating all 8 of its cores** — the same contention that stretched a ~3 min
> test suite to 33 min. Re-measured on the idle box, single-thread `mpnn_4x64` is **0.217 s/batch**,
> not 4.17; the pod is 0.106. The inflation was ~19×, and it propagated into every downstream
> figure. A FLOP estimate would have caught it immediately: 96 graphs are 8414 nodes / 23158 edges
> ≈ 5.4 GFLOP with backward, so 4.17 s implies **1.3 GFLOP/s** — an order of magnitude below what a
> single core does. Cross-check timings against a work estimate before drawing a feasibility
> conclusion from them, and measure on the target shape (CLAUDE.md's rule, which exists for this).

### Epoch probe (pod `xtyr889fxi7pml`, 2026-08-16) — and an unwelcome early signal

One pooled `mpnn_4x64`, model seed 0, frozen recipe, on the existing EXP-0039 labels (no new labels
spent). **Early stop at epoch 179/180, best epoch 54** — so `max_epochs=300` does not bind and needs
no change. At 180 epochs the full ladder is **5.8 core-h**, the trimmed ladder ~2.4, and the critical
path ~**38 min**. Note that the stopping rule spends 70% of the run confirming a plateau: patience 25
at `validation_every` 5 means 125 epochs must pass after the best checkpoint.

The validation curve, however, reproduces EXP-0039's negative result **under pooling**:

| epoch | val_regret | train_loss |
|---:|---:|---:|
| 4 | 6.176 | 29.71 |
| 54 (best) | **5.993** | 18.34 |
| 99 | 6.241 | 10.14 |
| 179 | 6.456 | 4.71 |

Train loss falls **84%**; validation regret improves **3.0%** at best and then degrades past where it
started, ending around EXP-0039's per-task random floors (5.24 / 5.29 / 6.23). That is memorization
without transfer — the signature EXP-0039 closed on.

**This is not a verdict on EXP-0041.** The experiment makes two interventions on EXP-0039: pool the
tasks, and train on 8 independent trajectories instead of 2. This probe has the first and not the
second — the EXP-0039 dump holds only 3 episodes per task, so one held out for validation leaves a
train-2 fit. It is therefore evidence that *pooling alone does not close the gap*, and silent on
whether more trajectories do. One seed, `single_suffix` target, one validation episode per task.

The consequence for the ladder is direct, and the curve was **reinstated as mandatory** on the
strength of it: the 2/4/8 curve is the instrument that separates the two live explanations —
*rising* means additional independent trajectories help (though train-8 may still be insufficient),
*flat* means increasing trajectory coverage over the tested range does not solve the problem.

Its specification (protocol v2): pooled `mpnn_4x64` only, model seeds {0,1,2}, train-2 and train-4
as additional fits with train-8 **reused** from the primary fit, an identical validation split
across sizes, a **separate train-fold normalizer per size**, and per (task, model seed) reporting of
regret, regret/random, r² and dispersion.

**It is a diagnostic, not a gate.** It does not select the architecture and does not reach the
adapter, Gate A or Gate B. If train-8 fails the pre-registered validation arm-selection the run
stops before test labels; if train-8 passes, a flat curve may **not** cancel the test, and a rising
curve may **not** rescue a train-8 that failed selection.

**What the curve does not do is save the label budget.** train-2 and train-4 are nested *prefixes*
of train-8, so the mandatory run still needs all 24 train + 9 validation episodes — about **79%** of
the label spend. Only the 9 final-test episodes (**~21%**) are contingent, and it is staged
collection, not the curve, that defers them.

**Threading is not a latency lever here, and the single-thread pin costs nothing.** Measured on the
same batch: 1 thread 0.217 s, 8 threads **0.401 s** — these are many small per-relation matmuls, so
threading overhead exceeds the parallel gain. The recipe's `cpu-single-thread` pin buys exact
reproducibility for free.

Two things still make the fit figure a **lower bound**. (1) It is a `single_suffix` population. Flatness is an exact
tie between candidate labels, and a mean over K continuations breaks ties that a single suffix left
flat — so `continuation_mean` plausibly trains on *more* decisions than 11,376, scaling the fit cost
with it. Count the non-flat population in mean mode before trusting a number here. (2) It excludes
the validation pass (`arm_metrics` every 5 epochs) and the 30th run (the mean-mode control arm).

**The label caveat is the headline.** The pinned EXP-0035 producer checkpoint is not present locally, so
the smoke ran with the `stock_aware` baseline producer. EXP-0039's real dump cost roughly 4
core-hours for 13,940 decisions (~1.03 s/decision) versus ~0.06 s here — the scorer producer
dominates the rollout by roughly 17×, which is the right-hand column. **Re-measure with the pinned
checkpoint staged before committing to a K.** Artifacts: ~579 MB of graph shards, ~27 MB
`candidates.csv`, ~15 MB of continuation labels per K. Rollouts add 171 validation-grid episodes
and 27 final-test episodes.

## Result

**Cancelled before data collection.** EXP-0040 failed its target-stability gate after the full
K=32 continuation budget, even though its expected selector passed the usefulness gate on all
three tasks. Its mechanical classification is `SEARCH_ONLY`. `read_exp0040_license()` maps that
classification to `Exp0040Cancelled`, so EXP-0041 has no legal target mode and cannot start its
dump, fit, Gate A, or Gate B stages.

## Conclusions

The cancellation is the intended operation of the precondition, not a missing experiment result.
Spending the EXP-0041 label and fit budget would repeat direct imitation on a target that did not
stabilize. The implementation remains as a tested reference, but this protocol must not be run
unless a new experiment establishes a different licensed target.
