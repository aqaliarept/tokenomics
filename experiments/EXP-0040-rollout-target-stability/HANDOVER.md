# EXP-0040 Handover - Rollout Target Stability

## Plain Goal

EXP-0039 established that the relational GNN can fit its training trajectories but
does not transfer to held-out trajectories. Its held-out producer choices are at the
random floor and its centered r2 is negative on all three gate tasks.

EXP-0040 asks the next question before any new actor is trained:

```text
Is the single-continuation marginal-value label a stable and useful function of the
deployable pre-action state after future exogenous demand is marginalized out?
```

This is a target audit, not another GNN architecture experiment. It distinguishes:

1. **Single-future label noise.** The original teacher scores each candidate against
   one realized future demand suffix. A stable expected preference exists, but one
   suffix is a poor sample of it. Averaged labels may rescue supervised learning.
2. **Ordinary coverage/generalization failure.** The original and expected preferences
   agree, so future demand is not the main cause. More independent trajectories and
   stronger regularization are the next lever.
3. **Unavoidable future sensitivity.** Even an affordable continuation ensemble does
   not identify a stable action, or a stable expected action offers little improvement
   over the incumbent. Direct teacher imitation should remain retired; use rollout or
   search at decision time.

The experiment must not add future demand to `graph_primitives_v1`. Counterfactual
future schedules are evaluator-only interventions used to measure the target. They
are never deployable observation features.

## Starting Evidence

EXP-0039 used `delta=1.0`, `lookahead=8`, the fixed `blend_0.5` continuation policy,
and the EXP-0035 producer over three fixed topologies. It trained one graph probe per
task with leave-one-trajectory-seed-out folds.

| Task | GNN regret | Random regret | Canonical actor regret | Held-out r2 |
|---|---:|---:|---:|---:|
| s29 | 5.290 | 5.380 | 4.120 | -0.759 |
| s77 | 6.231 | 6.241 | 4.754 | -0.811 |
| s45 | 5.250 | 5.239 | 3.935 | -0.720 |

The low training loss rules out simple under-optimization. It does not distinguish
memorization from a target that depends on future facts absent from the observation.

The simulator is deterministic after `env.reset(seed=...)`. Repeating
`marginal_value_labels` on an unchanged clone therefore produces the same labels and
is not a stability test. EXP-0040 must intervene on the *future demand suffix* while
holding the exact pre-action state fixed.

## Frozen Scope

Reuse the exact EXP-0039 environment and label contract:

- task IDs: the registered s29, s77, and s45 topology instances;
- trajectory seeds: s29 `{29,30,31}`, s77 `{77,78,79}`, s45 `{45,46,47}`;
- horizon: `100`;
- producer: EXP-0035 `producer_candidate.pt`;
- trajectory and continuation orderer: `blend_0.5`;
- marginal bump: `delta=1.0`;
- rollout window: `lookahead=8`;
- graph observation: `graph_primitives_v1`, schema hash `9918b12f69760bcd`;
- decision boundary: the same pre-action orderer state used by EXP-0039.

The existing EXP-0039 decisions and candidates are the sampling authority. The
canonical EXP-0038B `dump_sys` rows are the actor-choice authority. Join them only by
`(task_id, seed, cycle, order_id, producer_id)` and require exact coverage.

Do not substitute the pre-P2 checked-in residual scorer for the canonical actor rows.

## Counterfactual Continuation

Implement one domain-owned helper with semantics equivalent to:

```text
fork_future_demand(env, continuation_id) -> independent cloned env
```

For a fork made at the current ordering agent's turn:

1. Deep-copy the full environment.
2. Preserve the complete system state, current open orders, current-cycle demand,
   balances, telemetry, auctions, reservations, agent selector, and episode stats.
3. Leave fixed `ScriptedConsumer` instances unchanged.
4. For each randomized `ScheduledConsumer`, preserve every schedule entry already
   consumed and replace only entries beginning at its next unconsumed cycle.
5. Sample replacement products and quantities from that consumer's configured legal
   pool and inclusive quantity range.
6. Derive the RNG seed from SHA-256 of
   `(EXP-0040, task_id, decision_digest, continuation_id)`. Do not use Python's
   process-randomized `hash()`.
7. Use the same forked future for the unperturbed baseline and every candidate bump
   within a decision. This common-random-number control isolates the routing change.

The source trajectory continues under its original future after labels are collected.
A diagnostic fork must never alter which later source-trajectory decisions are
visited.

## Required Validity Tests

Write these before the collector:

- A fork leaves the encoded pre-action graph byte-identical to the source graph.
- Current open orders and current demand are identical before and after forking.
- Only unconsumed randomized schedule entries differ.
- Fixed-demand consumers are unchanged.
- A fixed `(decision_digest, continuation_id)` is bitwise deterministic.
- Different continuation IDs produce different legal suffixes when the task has
  randomized demand.
- The baseline and all candidate bumps for one replicate consume the same suffix.
- Candidate evaluation does not mutate either the source environment or sibling
  forks.
- With the original suffix, labels exactly reproduce the EXP-0039 candidate rows.
- Decision and candidate joins against EXP-0039 and canonical EXP-0038B have exact
  per-decision coverage and producer alignment.
- All emitted values are finite, and all counts and provenance checks run before
  analysis.

If a future suffix cannot be replaced without changing the current graph or current
open demand, stop. Do not approximate this audit by resetting a new episode and
calling its later state "the same state."

## Decision Sample

Use only the existing EXP-0039 decision population. Select deterministically by the
lowest SHA-256 rank of the decision digest within each stratum.

Strata are `(task, trajectory_seed, cycle_quartile, original_label_class)`, where
cycle quartiles are `0-24`, `25-49`, `50-74`, and `75-99`.

- Signal sample: up to 10 contested, non-flat EXP-0039 decisions per
  `(task, seed, quartile)`, at most 360 decisions.
- Flat control: up to 3 contested, flat decisions per `(task, seed, quartile)`, at
  most 108 decisions.
- If a stratum is smaller, take all of it and record the shortfall. Do not backfill
  from another stratum.

The primary verdict uses the signal sample because that is the population scored by
the EXP-0039 gate. Flat controls answer whether averaging reveals stable preferences
that a single continuation hid. Report task and trajectory-seed results separately;
do not treat adjacent decisions as independent episodes.

## Label Collection

For every sampled decision:

1. Reproduce the original pre-action state on the unchanged base trajectory.
2. Record the original-suffix labels once as a parity control.
3. Generate continuation IDs `0..15`.
4. For each continuation, score the unperturbed base and every eligible producer
   bump with `marginal_value_labels` semantics.
5. Persist raw, unrounded candidate labels before calculating summaries.

Sixteen continuations are the registered primary budget. If and only if the Monte
Carlo convergence gate below fails, extend the same decisions to IDs `16..31` once.
There is no second escalation and no threshold tuning after viewing the result.

## Metrics

All candidate metrics are decision-centered. Winner metrics are tie-aware and operate
on winner sets, not arbitrary first-index tie breaking.

For each decision, report:

- original-suffix candidate labels and winner set;
- continuation mean, standard deviation, standard error, and 95% bootstrap interval;
- ensemble-mean winner set and top-two expected margin;
- probability each candidate belongs to a continuation-specific winner set;
- original-to-ensemble winner-set overlap and centered rank correlation;
- signal-to-noise ratio:
  `variance_across_candidate_means / mean_candidate_continuation_variance`;
- scenario regret of the ensemble-mean selector, canonical actor selector, and a
  uniform random selector against each continuation-specific oracle;
- value of perfect future information: continuation-oracle value minus the value of
  the ensemble-mean selector.

For convergence, use 100 deterministic balanced splits of the continuation IDs. At
16 continuations compare two groups of 8; after the one allowed escalation compare
two groups of 16. Report task-level distributions of:

- split-half centered-label Spearman correlation;
- split-half winner-set overlap;
- absolute change in top-two expected margin; and
- absolute change in actor and ensemble-selector regret.

Aggregate decision-weighted within each task. Also emit every metric by trajectory
seed so one seed cannot hide instability in another.

## Pre-Registered Read

### Gate 1 - Monte Carlo convergence

The expected target is affordable and numerically stable only if, on the signal
sample, every task has:

- median split-half centered-label Spearman `>= 0.70`; and
- mean split-half winner-set overlap `>= 0.80`.

Apply the conditions to the task aggregate and independently to each of its three
trajectory seeds. If 16 continuations fail, extend to 32 exactly once. Failure at 32
is a hard stop for averaged-label actor training at this rollout budget.

### Gate 2 - Expected-action usefulness

Using raw continuation outcomes, compare the ensemble-mean selector with the
canonical actor selector. Averaged-label training is useful enough to attempt only
if the perfect ensemble-mean selector:

- reduces mean scenario regret versus the canonical actor by at least 10% on at
  least 2 of 3 tasks;
- is not worse than the canonical actor by more than 5% on the remaining task; and
- satisfies both conditions independently on every trajectory seed used by a
  passing task.

This is an oracle ceiling, not a model result. If the ceiling cannot clear the actor,
no supervised approximation to that expected target is licensed.

### Diagnostic classification

Classify the result mechanically after Gates 1 and 2:

| Classification | Conditions | Consequence |
|---|---|---|
| `MEAN_TARGET_LICENSED` | Gates 1 and 2 pass; original-to-mean winner overlap `<0.70` on at least 2 tasks | Single-suffix labels are unstable samples. Open an averaged-label GNN experiment. |
| `COVERAGE_PATH_LICENSED` | Gates 1 and 2 pass; original-to-mean winner overlap `>=0.80` and centered rank correlation `>=0.70` on at least 2 tasks | Future suffix is not the main failure. Open a many-trajectory control before changing the target. |
| `AMBIGUOUS` | Gates 1 and 2 pass but neither diagnostic pattern holds | Do not choose a rescue from aggregate means. Inspect per-seed and margin-conditioned artifacts, then pre-register a narrower follow-up. |
| `SEARCH_ONLY` | Gate 1 or Gate 2 fails after the allowed continuation budget | Keep direct marginal-value imitation retired. Move to bounded rollout/search or a directly observable return objective. |

The experiment cannot produce `GNN_ACTOR_LICENSED`. It contains no trained actor and
no deployed rollout.

## Conditional Follow-Up

The implementation handover for this conditional follow-up is
[EXP-0041](../EXP-0041-gnn-generalization-residual-router/HANDOVER.md). Its newly
registered validation and final-test seeds supersede using the repeatedly inspected
EXP-0039 anchor seeds as the primary final test, and it retrains the old row actor on
the identical held-out protocol instead of treating its in-sample regret as evidence
of generalization.

If `MEAN_TARGET_LICENSED`, open a separate experiment with these fixed boundaries:

- retain `graph_primitives_v1` and the EXP-0039 GNN architecture;
- generate many independent base trajectories, keeping the original gate seeds
  completely held out;
- choose the smallest continuation count in `{4,8,16}` that met the EXP-0040
  convergence thresholds against the full ensemble;
- compare a single-suffix MSE control with an ensemble-mean MSE treatment on the
  same graph rows and model seeds;
- introduce a listwise or pairwise loss only as a later, separately identified arm;
- apply the unchanged EXP-0039 regret/r2/actor-floor gate to the untouched original
  trajectory seeds.

If `COVERAGE_PATH_LICENSED`, first run the same architecture and original MSE target
with at least eight independent training trajectories per task. Do not change labels,
features, architecture, and coverage in the same primary comparison.

If `SEARCH_ONLY`, benchmark short rollout depths `{1,2,3}` as the next compute lever.
The GNN may later be tested as a candidate-pruning or rollout-value component, but it
must not be treated as a standalone actor based on EXP-0039 training fit.

Only a later supervised probe that clears the held-out EXP-0039 gate may license a
closed-loop GNN actor run. Only a later fixed-topology pass may license an unseen-
topology split.

## Artifacts

The run must write:

```text
experiments/EXP-0040-rollout-target-stability/
  spec.json
  RESULT.md
  sample.csv
  artifacts/
    continuation_labels.csv
    decision_stability.csv
    metrics_by_task.csv
    metrics_by_seed.csv
    selection_manifest.json
    run.log
    spec.json
```

`continuation_labels.csv` is the audit authority. Every summary and verdict must be
recomputable from it without the simulator.

Both the selection manifest and run spec must record source revision, source-tree
fingerprint, runtime environment, pod image, exact input hashes, task IDs, trajectory
seeds, continuation derivation, and the original EXP-0039 decision/candidate hashes.

## Implementation Order

1. Add the future-demand fork API and its invariant tests.
2. Add deterministic sample selection and authority-alignment tests.
3. Add the raw collector and original-label parity test.
4. Add the artifact-only analyzer and mechanical classification tests.
5. Run a two-decision smoke test locally.
6. Review the emitted smoke artifacts and recompute its verdict independently.
7. Commit the producing source and launch the registered CPU pod run.
8. Pull raw artifacts and logs before terminating the pod.
9. Recompute all summaries from `continuation_labels.csv` and write `RESULT.md`.

## Stop Conditions

- Any mismatch with the EXP-0039 original label, candidate set, or graph digest.
- Any mutation of current observable state under a continuation fork.
- Any dependence of a label on candidate evaluation order.
- Any missing or non-finite raw label.
- Any source, input, environment, or decision-authority fingerprint mismatch.
- Gate 1 failure after 32 continuations.
- Gate 2 failure even with the oracle ensemble-mean selector.

Do not train a GNN, tune a ranking loss, expand the observation, or run an unseen-
topology evaluation after a stop condition fires.
