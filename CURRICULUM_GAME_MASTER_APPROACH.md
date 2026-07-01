# Curriculum Game-Master Approach

Status: active roadmap replacing the previous opponent-centric self-improvement plan.

This document defines the self-improvement path for the portable producer/orderer
stack. The next step is not a learned consumer actor and not a full learned
adversarial world generator. The immediate target is a legal, replayable curriculum
pipeline:

```text
sample or script legal tasks ->
run current stack and reference bars/search ->
classify failures ->
archive useful high-regret tasks ->
replay/relabel/train portable scorers ->
evaluate transfer/stress matrices ->
only then consider mutation or learned generation
```

The core design lesson from the research report is that the first production-capable
win usually comes from task selection, curation, and replay. A learned game master is
a later-stage tool, justified only after the archive/replay/mutation path plateaus.

## Goal

Build a self-improvement loop that improves held-out and stress-suite performance
without breaking the portable scorer contract.

The loop must find tasks where:

```text
a legal task is replayable;
a reference method handles it better than the current learner;
the gap is a policy weakness, not a feasibility wall or implementation failure;
training on the task improves transfer or tail-risk metrics.
```

The game master is not part of the promoted system stack. It is a curriculum and
evaluation tool. Promotion remains human-approved and must use experiment artifacts,
not ad hoc runs.

## Current System Boundary

The active learner remains the portable scorer stack.

Producer:

- View: `ProducerEntityView`.
- Learned component: `ProducerScoreModel`.
- Decoder/planner: deterministic `plan_producer_command`.
- Deployment mode: planner-gated investments for out-of-family safety.
- Current checkpoint: `experiments/EXP-0021-producer-broaden-training/artifacts/producer_scorer_broad.pt`.

Orderer:

- View: `ProductStreamView`.
- Learned component: `OrdererScoreModel`.
- Decoder/allocator: deterministic masked `allocate_suborders`.
- Current checkpoint: `experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt`.

Non-negotiable boundary:

- Learned models produce reusable local scores.
- Domain code owns feasibility: funding, inventory, capacity, eligibility, recipes,
  quantity conservation, and legal action shape.
- Product ids, producer ids, and topology-specific fingerprints remain out of learned
  features except as replay/diagnostic labels.

## Design Principles

1. Do not reward raw failure.

   A task is useful only if a stronger reference method can do better. Otherwise the
   generator will drift toward impossible worlds, degenerate shocks, or implementation
   bugs.

2. Separate gates from ranking.

   First reject illegal, non-replayable, infeasible, or bug-triggering tasks. Only then
   rank surviving tasks by training value.

3. Prefer replayable tasks over adaptive opponents.

   A replayable world/shock trace lets current champion, candidate, and reference bars
   run on the same condition. Closed-loop adversaries can come later.

4. Preserve independent evaluation.

   The stress suite used for promotion must not be the same tasks the replay trainer
   repeatedly optimizes on.

5. Increase generator complexity only when the prior stage proves useful and plateaus.

   Broad randomization and hard-case mining come before mutation. Mutation comes before
   learned generation. Open-ended co-evolution is a separate research branch.

## Task Object

Every generated or scripted curriculum item should serialize to a task record.

Minimum fields:

```json
{
  "task_id": "stable-id",
  "task_family": "scale|fixture|stress|random|mutated|learned",
  "generator_version": "v1",
  "serializer_version": "v1",
  "world_seed": 0,
  "episode_seed": 0,
  "world_spec": {},
  "initial_state_spec": {},
  "event_trace": [],
  "demand_trace": [],
  "shock_trace": [],
  "legal": true,
  "replayable": true,
  "reference_status": "solved|partial|failed|not_run",
  "failure_class": "none|policy_weakness|feasibility_wall|contract_failure|implementation_failure|distribution_mismatch",
  "notes": ""
}
```

The exact schema can evolve, but every top-ranked task must be reproducible from the
record without relying on hidden process state.

## Scoring Model

Use hard gates first:

```text
G(task) =
  legal
  and replayable
  and no implementation failure
  and reference-solvable-or-reference-better
```

Then rank surviving tasks:

```text
gap(task) = reference_score(task) - learner_score(task)

score(task) =
  bandpass(gap(task))
  * (
      alpha * normalized_gap(task)
    + beta  * novelty(task, archive)
    + gamma * uncertainty(task)
    + delta * forgetting_value(task)
  )
```

`bandpass` is critical. It should down-rank:

- tasks where all policies succeed easily;
- tasks where all policies fail;
- tasks where the reference advantage is caused by a contract/load failure;
- tasks outside the current claim unless explicitly labeled as exploratory.

Default normalized gap:

```text
normalized_gap =
  (reference_completed_units - learner_completed_units)
  / max(1, feasible_or_requested_units)
```

If a reliable feasible upper bound is unavailable, use requested units for ranking but
keep the feasibility label conservative.

## Failure Classification

Every matrix row and archived task must distinguish:

- `policy_weakness`: reference/search/bar beats the learner on the same replayed task.
- `feasibility_wall`: every tested policy fails and the task is within a known
  feasibility limit, or a domain proof/diagnostic supports impossibility.
- `contract_failure`: checkpoint cannot load, schema mismatches, or tensor/action shape
  fails.
- `implementation_failure`: crash, timeout, NaN, illegal state transition, desync, or
  serializer/replay mismatch.
- `distribution_mismatch`: task is legal but outside the promoted claim.
- `regression`: candidate improves a mined task but loses a previously promoted row.

Do not collapse these into one scalar.

## Stage 0: Experiment Index And Baseline Hygiene

Purpose:

Use the existing experiment index and make the current portable stack and reference
bars explicit in the next experiment record.

Tasks:

- Use the existing `experiments/README.md` index and `EXP-NNNN-*` directory contract.
- Record producer checkpoint, orderer checkpoint, schema versions, decoder/planner
  versions, and deployment adapter in `spec.json` and `RESULT.md`.
- If automation needs machine-readable active-stack metadata, update the existing
  champion/stack metadata file rather than creating a second registry.
- Record current transfer/stress matrices and known residuals.
- Define reference bars available per task family:
  - deterministic producer planner;
  - deterministic/proportional orderer bars;
  - local routing search where available;
  - bounded short-rollout search where available.
- Define curriculum-specific artifact filenames inside the existing experiment layout.

Acceptance criteria:

- Portable producer and orderer checkpoints load through schema gates.
- The experiment record identifies the portable stack under test and any legacy stack
  used only as a historical baseline.
- One-seed smoke of the existing transfer matrix runs.
- Known residuals are listed and not hidden by joint-stack success.
- No driver can overwrite champion checkpoints or claim docs without explicit approval.

Reject if:

- A new parallel experiment index or directory convention is introduced.
- An experiment record implies a single fixed-shape actor is the active portable stack.
- A result can be promoted without writing an experiment record.
- Contract failures are mixed with policy scores.

## Stage 1: Independent Stress Suite

Purpose:

Create an evaluation-only suite that catches regressions and overfitting to generated
curriculum tasks.

Tasks:

- Define hand-scripted stress scenarios and edge fixtures.
- Include held-out topology families, size transfer, shared capacity, scarcity,
  degenerate fixtures, known residuals, and known feasibility walls.
- Add scenario metadata: family, seed, expected claim boundary, shock/demand pattern,
  and reference-bar availability.
- Keep this suite separate from training replay.

Acceptance criteria:

- Every stress task is legal and replayable.
- Every stress task has a stable id and serialized task record.
- Every row reports learner score, reference score when available, death, rejected
  units, contract status, and failure class.
- The suite can be run against current champion and deterministic bars.
- Known feasibility walls remain labeled separately from policy failures.

Reject if:

- The stress suite is used as the main training replay buffer.
- A candidate can pass by improving generated tasks while regressing stress-suite
  tail risk.
- Stress rows omit failure class or reference identity.

## Stage 2: Broad Legal Randomization

Purpose:

Build the task serialization, legality checks, replay path, and broad sampling baseline
before adding intelligence to the generator.

Tasks:

- Define a random task generator over legal world parameters, initial state, demand
  regimes, and bounded shock/event traces.
- Implement task serialization and replay.
- Implement legality checks before rollout.
- Implement replay determinism checks after rollout.
- Run learner and reference bars on sampled tasks.
- Write every sampled task and result to an archive.

Acceptance criteria:

- At least one broad random pool can be generated and replayed locally.
- Top-ranked and random-sampled task records reproduce deterministically, or within a
  documented stochastic tolerance.
- Illegal-task rate and implementation-failure rate are reported.
- Reference bars can run on a smoke subset.
- Archive rows include all fields needed for later replay and matrix comparison.

Reject if:

- Task records cannot reproduce the original run.
- The generator can create illegal states that pass the legality gate.
- Implementation failures are scored as useful hard cases.
- Randomization changes learned feature contracts.

## Stage 3: Random Hard-Case Mining

Purpose:

Find useful policy weaknesses without training a learned generator.

Tasks:

- Sample a large pool of legal random tasks.
- Run the current portable stack on each task.
- Run the strongest affordable reference method on the same task.
- Compute normalized regret/gap.
- Classify each failure.
- Archive high-gap, reference-better, legal, replayable tasks.
- Track novelty descriptors so the archive does not collapse to one failure type.

Ranking default:

```text
score_mine =
  hard_gates
  * bandpass(reference_score - learner_score)
  * (alpha * normalized_gap + beta * novelty + gamma * uncertainty)
```

Acceptance criteria:

- Mined tasks include replayable traces and reference identities.
- High-ranked tasks are mostly `policy_weakness`, not infeasible or bug-triggering.
- Archive diversity descriptors show more than one hard-case family, or the collapse is
  explicitly reported.
- A blind subset of mined tasks reproduces the original ranking.
- Mined tasks do not replace the original transfer/stress matrix.

Reject if:

- Ranking is based on raw learner failure only.
- Most top-ranked tasks are impossible, out-of-claim, or reference-failed.
- The same exploit dominates the archive.
- No reference/bar comparison exists for high-ranked tasks.

## Stage 4: Prioritized Replay And Relabeling

Purpose:

Use mined tasks as curriculum for portable scorers while preserving the scorer/decoder
split.

Tasks:

- Split archive into train replay, validation replay, and blind holdout replay.
- Select replay tasks by priority:

```text
priority_i proportional to (epsilon + score_i) ^ eta
```

- Collect producer and orderer states from replayed tasks.
- Relabel producer states with planner/search teachers.
- Relabel orderer states with local routing or short-rollout search.
- Train producer and/or orderer scorers without schema changes.
- Evaluate candidate on:
  - original transfer matrix;
  - independent stress suite;
  - mined validation replay;
  - blind replay holdout;
  - producer-alone rows for producer-side misses;
  - joint-stack rows for system value.

Acceptance criteria:

- Candidate improves or ties champion on mined validation tasks.
- Candidate improves tail risk or worst-case metric on blind replay holdout.
- Candidate does not regress promoted transfer/stress rows.
- Zero new contract failures.
- Death does not increase on non-feasibility-wall rows.
- Producer-alone and joint-stack metrics are both reported where relevant.
- Any close win passes paired seed/trace comparison, not just aggregate mean.

Reject if:

- Candidate only improves the exact replay training tasks.
- Candidate regresses existing transfer/stress gates.
- Gains come from changing deterministic feasibility code without a separate audit.
- Producer residuals are hidden by joint-stack orderer lift.

## Stage 5: ACCEL-Style Local Mutation

Purpose:

Generate new frontier tasks by mutating archived high-value tasks instead of generating
from scratch.

Tasks:

- Define 2 to 5 local mutation operators, such as:
  - bounded parameter shift;
  - initial-state perturbation;
  - demand/event trace splice;
  - bounded shock timing shift;
  - single structural edit within schema compatibility.
- Validate that each mutation preserves legality and replayability.
- Select parent tasks from high-score archive cells.
- Score child tasks using the same hard gates plus regret/novelty bandpass.
- Compare mutation against pure random mining and pure replay.

Scoring default:

```text
score_mutated =
  hard_gates
  * bandpass(reference_score - learner_score)
  * (alpha * normalized_gap + beta * novelty)
```

Acceptance criteria:

- Mutations produce new useful tasks at a higher rate than broad random sampling, or the
  result is rejected as not worth the complexity.
- Child tasks are legal, replayable, and mostly reference-better.
- Archive diversity increases by meaningful descriptors.
- Training with mutated tasks improves blind replay or stress-suite metrics beyond
  prioritized replay alone.
- Mutation operators and parent/child lineage are serialized.

Reject if:

- Mutations mostly create impossible tasks or trivial duplicates.
- Small edits do not preserve task semantics.
- Mutation improves archive score but not held-out or stress-suite performance.
- Mutation makes failure classification ambiguous.

## Stage 6: Learned Game Master

Purpose:

Train a generator only if curated replay and mutation have proven useful and plateaued.

Entry criteria:

- Stage 4 replay improved held-out or stress-suite metrics.
- Stage 5 mutation improved task discovery or was shown to plateau.
- Legality, replayability, reference, and bug filters are reliable.
- There is enough compute budget for unstable bilevel/self-play-like training.

Tasks:

- Define the learned generator action space over already validated generation/edit
  primitives.
- Start with a learned editor over archived tasks, not unconstrained world generation.
- Use gated regret-to-reference reward:

```text
reward_gm =
  hard_gates
  * (
      alpha * max(0, reference_score - learner_score)
    + beta  * novelty
    - lambda_easy * easy_task_penalty
    - lambda_bug  * exploit_or_bug_penalty
  )
```

- Freeze generator versions into an adversary/generator league.
- Evaluate against random mining, prioritized replay, and mutation baselines.

Acceptance criteria:

- Learned generator improves downstream held-out/stress metrics beyond mutation at the
  same or documented compute budget.
- Generated tasks pass legality/replayability/bug gates at an acceptable rate.
- Top-ranked tasks are diverse and reference-better.
- Removing exploit penalties or feasibility gates demonstrably worsens task quality in
  an ablation.
- Old generator versions remain available for regression evaluation.

Reject if:

- The learned generator mostly finds bugs, timeouts, or illegal states.
- It increases training regret without improving blind held-out metrics.
- It overfits to the current champion and fails against candidate/champion swaps.
- Its compute cost is not justified by downstream gains over mutation/replay.

## Stage 7: Open-Ended Or Population-Based Branch

Purpose:

Explore capability discovery, not production curriculum.

Entry criteria:

- The project explicitly wants open-ended discovery rather than bounded robustness.
- There is a separate budget and evaluation plan.
- Success is defined by novelty, transfer value, and new skill discovery, not only the
  existing promoted claim.

Tasks:

- Define task descriptors for a quality-diversity archive.
- Define minimal criteria: not too easy, not too hard, legal, replayable.
- Maintain populations of tasks and candidate solutions.
- Attempt cross-task transfer and record transfer gains.
- Keep this branch separate from promotion gates unless a task graduates into the
  independent stress suite.

Acceptance criteria:

- The archive covers meaningful behavior/task descriptors.
- Discovered tasks are replayable and classifiable.
- At least some discovered tasks expose transferable skill gaps beyond the existing
  stress suite.
- Compute cost and evaluation ambiguity are explicitly reported.

Reject if:

- Open-ended metrics replace transfer/stress promotion gates.
- Novelty is rewarded despite infeasibility or implementation failure.
- The branch cannot explain how discovered tasks improve the bounded product claim.

## Promotion Gates For Any System Candidate

A candidate scorer stack is eligible for promotion only if all are true:

- Checkpoints load by portable schema, not config fingerprint.
- Contract failures are zero on the covered claim.
- Death is zero on non-floor, non-feasibility-wall acceptance rows.
- Original transfer matrix passes.
- Independent stress suite passes.
- Candidate improves or ties current champion on frozen curriculum/replay validation.
- Candidate does not regress named residuals unless explicitly marked as a tradeoff.
- Producer-alone and joint-stack metrics are both reported where relevant.
- Adversarial/curriculum failures are classified, not collapsed into one scalar.
- Evaluation uses paired task/seed/trace comparison for close wins.
- Result is written to an experiment directory.
- Promotion is human-approved.

For close wins:

```text
lower 95% CI of paired delta >= 0
```

If that is too expensive, mark the result as promising but not promoted.

## Experiment Record

Every curriculum run must use the existing experiment convention documented in
`experiments/README.md`:

```text
experiments/EXP-XXXX-title/
  spec.json
  RESULT.md
  summary.csv
  configs/
  artifacts/
```

Curriculum-specific files should be added inside that structure, usually under
`artifacts/`:

```text
artifacts/
  task_archive.jsonl
  curriculum_matrix.csv
  replay_validation.csv
  logs/
  producer_candidate.pt
  orderer_candidate.pt
```

The existing `spec.json` / `RESULT.md` / `summary.csv` record must state:

- stage number and generator version;
- stack/champion checkpoints used;
- reference bars/search used;
- task families sampled;
- train/validation/blind split;
- legality/replayability/implementation-failure rates;
- infeasible/reference-failed rate;
- archive diversity descriptors;
- whether producer, orderer, or both were trained;
- transfer/stress matrix result;
- whether `PROMOTE_RECOMMENDED=true|false`.

Drivers should write the experiment directory. A reviewed human update should add the
row to `experiments/README.md` using the existing convention. Claim documents and
status/memory files should change only when promotion or claim updates are explicitly
approved.

## What Not To Do

Do not start with a learned consumer actor.

Do not reward the generator for raw learner failure.

Do not train a full learned world generator before legality, replayability, reference,
and bug filters are reliable.

Do not let generated curriculum replace the independent stress suite.

Do not promote on generated-task reward without rerunning the original transfer matrix.

Do not count reference-failed tasks as policy weaknesses.

Do not hide producer-alone misses behind joint-stack value.

Do not change learned feature contracts as part of curriculum generation.

Do not mutate topology or state mid-episode outside legal event/shock primitives.

Do not treat open-ended discovery as the same objective as bounded product robustness.

## Immediate Next Experiment

The next concrete experiment should be Stage 0 plus Stage 1 scaffolding.

Title:

```text
EXP-0023-curriculum-task-archive-and-stress-suite
```

Purpose:

```text
Replace the opponent framing with a replayable task archive and independent stress
suite, then prove the current portable stack and reference bars can run on serialized
tasks with failure classification.
```

Tasks:

- Create the experiment under the existing `experiments/EXP-NNNN-*` layout.
- Record the portable stack under test in `spec.json` and `RESULT.md`.
- Define `TaskRecord` serialization.
- Convert existing fixture/stress rows into task records where possible.
- Add a one-seed replay smoke for current champion and reference bars.
- Emit `artifacts/task_archive.jsonl` and `artifacts/curriculum_matrix.csv` when those
  files add detail beyond `summary.csv`.
- Report legality, replayability, reference availability, and failure class.

Accept:

- Current portable stack loads from the existing checkpoint paths recorded in the
  experiment spec.
- At least one task from each existing accepted fixture family serializes and replays.
- Champion and reference bars can run on the replay smoke.
- Failure classes are present in every row.
- No promotion recommendation is made from this scaffolding experiment.

Reject:

- Task replay depends on hidden process state.
- Existing fixture behavior changes without explanation.
- Contract failures are mixed into policy weakness.
- The new archive cannot represent known residuals and feasibility walls.
