# Curriculum Game-Master Approach

Status: active roadmap replacing the previous opponent-centric self-improvement plan.
Stages 0-4 executed and closed on the depth-3 scale family (EXP-0023 .. EXP-0033) — see
"Executed History And Current Position". EXP-0030 is now the human-approved incumbent
producer stack after EXP-0033 demoted its advisory gate miss to saturated-family noise.
The active stage is Stage 4b: apply the same machinery to distributions with real
headroom and exit only with a promoted next champion. Stage 5 waits on that exit.

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

## Executed History And Current Position (EXP-0023 .. EXP-0033)

Stages 0-4 ran to completion on the depth-3 scale family. Findings that now bind this
roadmap:

- Stage 0/1 (EXP-0023): scaffold worked; the fixed practical bar was too weak
  (0 policy_weakness rows). Reference strength, not the learner, was the first blocker.
- Stage 0 bars (EXP-0024): the `entity_planner+mpc_search` short-rollout search bar
  produced the first in-claim policy_weakness rows.
- Stage 3 (EXP-0025, EXP-0028): single-cell mining collapses novelty; the topology grid
  (width x producer density) fixed it. Weakness concentrates at LOW producer density —
  the capacity-tight axis.
- Stage 4 (EXP-0026, EXP-0027, EXP-0029, EXP-0030): mined-only distill REPLACED broad
  coverage and regressed out-of-family cells; augment (broad + mined) closed every
  producer-side regression; freezing the orderer removed orderer replay noise. The best
  candidate ties champion on mined validation with zero transfer regressions.
- Attribution (EXP-0031): the producer channel closes 5.27% of the archived gap
  (train split 0.0%). Producer replay is not the lever.
- Orderer headroom (EXP-0032, EXP-0032b): champion orderer >= the mpc_search teacher on
  26/26 archived tasks under the frozen producer (8 negative). Nothing to distill.
- Ceiling audit (EXP-0033): analytic upper bounds (`eval/ceiling_bound.py`) are
  demand-bound on 26/26 archived tasks; champion fulfils mean 98.3% (min 97.6%) of
  exact demand, the bar 99.0%.

Position: the Stage-4 loop reached a real fixed point on this task family, and the
family itself is saturated — capacity never binds, so a near-perfect score is
structurally easy. Improvement requires new task distributions, not a stronger teacher
on the same archive. The EXP-0030 candidate is promoted by human override: zero
transfer regressions, wins in the scarcity/degenerate/uneven regimes Track A cares
about, and its one mined blind miss sits on the EXP-0033 saturated family.

## Current System Boundary

The active learner remains the portable scorer stack.

Producer:

- View: `ProducerEntityView`.
- Learned component: `ProducerScoreModel`.
- Decoder/planner: deterministic `plan_producer_command`.
- Deployment mode: planner-gated investments for out-of-family safety.
- Current checkpoint: `experiments/EXP-0030-stage4-freeze-orderer/artifacts/producer_candidate.pt`.

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

Amendments (binding, from EXP-0025/0028/0031/0032b/0033):

- Ceiling pre-flight: before mining a cell, run the offline upper bound
  (`eval/ceiling_bound.py`). Do not mine cells where demand is the binding ceiling
  (B1/B2 >= B0): gaps there are the last percent of a saturated objective.
- Recoverability gate: a policy_weakness row enters the archive only if a cheap
  cross-stack probe (gap attribution / orderer headroom scan) attributes the gap to a
  trainable component. Joint-bar-mismatch gaps are unlearnable by component distill.
- Mining seeds must be disjoint from held-out eval seeds, or the Stage-4 leakage guard
  later strips exactly the high-priority tasks.

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

Amendments (binding, from EXP-0026/0027/0029/0030):

- Augment, never replace: the candidate trains on the champion's broad set PLUS the
  mined replay. Mined-only distill trades robustness for nothing.
- Replay priority uses the raw gap, not the mining score: the novelty term belongs to
  mining coverage and makes the score bimodal (first-in-cell bonus), which concentrates
  sampling onto one task. Cap duplicates (<=2) or stratify per topology cell.
- A validation tie at measured-zero recoverable headroom is a ceiling, not a miss:
  treat improve-or-tie with a noise-aware paired comparison, and gate promotion on the
  transfer/stress side.

## Stage 4b: New-Distribution Curriculum To A New Champion

Purpose:

Stage 4 validated the replay machinery and produced the EXP-0030 human-approved
incumbent, but it also hit a real fixed point on a demand-saturated family
(EXP-0031/0032b/0033, see "Executed History"). Stage 4b applies the same Stage 3/4
machinery — with the amendments above — to distributions where headroom exists by
construction. The stage exits only when a candidate passes the promotion gates and a
next champion is promoted; only then does Stage 5 mutation start.

Three tracks, in order of cost-to-signal. Track A needs no domain change, Track B uses
reserved task-record machinery, Track C is a domain design change requiring an ADR.
Track A and Tracks B/C can proceed in parallel (A is a spec + run; B/C are code arcs).

### Track A: Capacity-Binding Mining (first)

Purpose: rerun Stage 3 where capacity, not demand, is the ceiling — the only regime
where headroom exists by construction. EXP-0028 already showed the gradient: weakness
triples at the tightest tested density cell (producers=12 vs 18).

Tasks:

- Grid axes: producer density below the current envelope (`producers_per_layer=3`),
  higher demand quantity, harsher degradation; include the `scarcity_variant`
  reproducers (already registered in `eval/curriculum.py`) as mineable families, not
  just stress labels.
- Ceiling pre-flight per cell (Stage 3 amendment): keep only cells where B1/B2 < B0.
- Recoverability probe per archived task (Stage 3 amendment).
- Then the existing Stage-4 replay machinery with its amendments (augment, gap
  priority, duplicate cap).

Accept:

- Mined archive concentrates in capacity-bound cells with component-attributable gaps.
- Stage-4 replay on it moves mined validation, not only transfer cells.

Reject:

- Cells pass pre-flight but the archive still saturates (report as a claim-boundary
  finding, not a failure).
- Gaps are recoverable by neither component (joint-bar mismatch again).

Amendment (binding, from EXP-0036): learnability-gated orderer harvest.

EXP-0036 mined a real capacity-binding archive (headroom exists), but the first
orderer harvest attempt failed at the SIGNAL, not the model. Distilling the search's
per-cycle committed routing gave a target that is mostly noise: on the 3 instrumented
probe tasks, variance-explained over a route-uniformly baseline was 25.4% / 9.8% /
64.1% (s29/s77/s45, recomputed from saved samples), 29-50% of rows carried no routing
lesson (search committed even_split), and the same recipe with a different torch init
deployed 0 vs 800 units. Root cause (DIAGNOSTICS entry 11): the search commits ONE
global routing per cycle among near-tied candidates (2 named + 8 per-cycle Gaussian
vectors), sliced into per-stream local targets — the informative part is global +
tie-arbitrary, so per-stream MSE fits noise. NOT a representability wall, NOT plain
covariate shift.

Binding constraints on any orderer harvest from here:

- No learned critic / value model. EXP-0009 already showed units-to-go from
  `global_state` has NEGATIVE held-out R^2 across topologies ("do not spend cloud
  rescuing the value model; expect the same wall elsewhere in the arc"). A critic-
  picks-best actor or critic-gated runtime search rests on a generalization the arc
  disproved. Deploy-time search, if used, is gated by a CHEAP HEURISTIC trigger
  (capacity reserve low / rejection rate high), never a learned death-risk head.
- No categorical soft-labels over named routes A/B/C. Candidates are per-cycle random
  Gaussian vectors (`eval/orderer_oracle.py:_candidate_list`), re-drawn each cycle;
  the deploy action is a continuous per-stream simplex. There is no stable route to
  label. Denoising is value-weighted regression, not classification.

The harvest is gated on the discriminating diagnostic first — it can FAIL and end the
orderer track:

Stage A0 (one instrumented pod run; build/`--dry-run` local on 1 task). On the 3
probe tasks emit one decision table:

```text
1. Headroom (H3, evaluate FIRST): cumulative consumer units under
   even_split, proportional, search-oracle, champion.
   oracle ~= even_split  -> STOP: no orderer headroom in this regime;
                            accept champion or re-mine tighter. Falsifiable gate.
   oracle >> even_split   -> proceed.
2. Representation (H4): is the search's headroom a per-cycle CHOICE among named
   policies, or genuinely-continuous producer weightings? Run the SAME per-cycle
   search restricted to named candidates only (n_samples=0, still switching
   even_split/proportional per cycle, no Gaussian samples) -> search_named_cu.
   named_capture_frac = (search_named - even_split) / (search_full - even_split).
   ~1 -> the lever is per-cycle named-policy choice (a LEARNABLE categorical/blend
         teacher captures it; the raw continuous vector was the wrong output space) -> A1b.
   ~0 -> the win is continuous routings a named vocabulary would throw away -> keep the
         continuous target, go to Denoise (H1).
   NOTE: do NOT read this off the fixed-policy columns. even_split_cu / proportional_cu
   are single policies held all episode; a per-cycle CHOICE among them is strictly
   stronger and sits between max(fixed named) and search_full. Only search_named_cu
   (per-cycle named switching) answers H4.
3. Denoise (H1): per visited state run K=4 independent searches (different sample
   RNG), commit the mean vector to advance (cross-trajectory averaging is invalid;
   committed routing diverges the trajectory after cycle 0). Recompute
   VE = 1 - model_mse/zero_mse for single vs averaged targets.
4. Global-context upper bound (H2): refit the student with a small system-summary
   feature block (capacity reserve by layer, downstream backlog) and recompute VE.
   NOTE: per-stream deciders are independent; global features raise an upper bound
   but do not reconstruct a jointly-committed routing — H2-positive means the obs
   contract, not the loss, is the lever.
```

Branches (read in order): named_capture ~1 -> H4, representation is the fix -> Stage A1b
(categorical/blend head). Else averaged VE jumps -> H1, denoise suffices -> Stage A1.
Only averaged+global jumps -> H2, obs change needed -> Stage A1 with global obs. Neither
moves but headroom exists -> Stage A2 (deploy-time search is the only lever). Neither moves
and no headroom -> orderer track done, report claim boundary.

Stage A1 (only if A0 green): distill the denoised target with the existing per-stream
orderer, loss = advantage-weighted regression, advantage = `search_score -
even_split_score` computed OFFLINE from search rollouts (not a learned critic). Flat /
near-tie rows down-weight for free. Gate: deployed consumer units >= champion on the
held-out mined suite (standard candidate gate + Stage 4b promotion gates).

Stage A1b (only if H4 positive — named-policy teacher representation): the single-choice
noise is an OUTPUT-SPACE problem, not just a label-noise problem. Instead of regressing a
raw per-producer routing vector (whose tie-arbitrary winner among near-tied Gaussian
candidates is unlearnable RNG), change the teacher output to a small, state-reproducible
vocabulary and have the student predict THAT:
  - Categorical head: per stream, classify {even_split, proportional, don't-care} (the
    committed named policy the per-cycle search chose). Near-ties become well-defined soft
    labels (e.g. even_split 0.5 / proportional 0.5) — the very thing rejected earlier ONLY
    because candidates were Gaussian; a named vocabulary makes soft labels legitimate.
    `don't-care` is the no-lesson (true-tie) class named explicitly, not down-weighted.
  - Blend head (middle ground if named_capture is close-but-<1): output one interpolation
    coefficient alpha per stream, route alpha*even_split + (1-alpha)*proportional. 1 scalar/
    stream — far more learnable than the P-vector, more expressive than hard classification.
  Decode by EXECUTING the chosen/blended named policy (reproducible, no simplex regression).
  Gate: deployed consumer units >= champion on the held-out mined suite (same as A1). Why it
  beats A1 when H4 holds: kills the RNG noise at the source (reproducible targets) AND shrinks
  the output space (categorical/1-scalar), so the student learns the actual per-cycle-switching
  policy the search runs. If named_capture ~0, this stage is skipped — the vocabulary does not
  contain the headroom.

Stage A2 (fallback / guaranteed harvest): deploy-time light search (2-4 candidates)
fired only on the heuristic danger trigger above; measure units gained vs search-calls
per cycle (cost knob). The `~1000u` deploy-time headroom is the known floor.

Accept (orderer harvest): A0 shows headroom AND a downstream stage (A1b categorical/blend,
A1 continuous-denoise, or A2 deploy-time) clears the champion bar on held-out mined tasks.
Reject: A0 shows oracle ~= even_split (no headroom), OR named_capture ~0 AND VE columns stay
at the noise floor with no deploy-time win — report as a claim-boundary finding, do not spend
on width/epochs/DAgger (the wall is upstream).

### Track B: Scripted Demand And Shock Traces

Purpose: test and train closed-loop adaptivity — the historically weakest documented
axis (operating-point memorization). This is the intended use of the reserved
`demand_trace` / `shock_trace` / `event_trace` task-record fields, and the legal
primitive route Stage 5 mutation expects ("demand/event trace splice", "bounded shock
timing shift").

Tasks:

- Implement trace injection in the replay path: scripted per-cycle demand deltas and
  bounded mid-episode shocks (capacity dip, demand spike), serialized in the task
  record, deterministic on replay.
- Extend the generator with bounded trace sampling; legality gates unchanged.
- Mine and replay with the same Stage 3/4 machinery and amendments.

Accept:

- Traces replay deterministically from the task record alone.
- Mined trace tasks are policy_weakness with recoverable attribution.
- No learned feature-contract change.

Reject:

- Traces mutate topology/state outside legal event/shock primitives.
- Shock tasks are mostly feasibility walls.

### Track C: Per-Order TTL (urgency signal + depth axis)

Purpose: turn `ORDER_TTL` from a global domain constant into a per-order field. TTL
becomes a demand-urgency signal policies can learn to triage, a curriculum dial the
game master can anneal, and the feasibility gate for depth>=4 families (deep_join d5
is a known TTL=2 wall).

Design decisions (pin these in an ADR before code):

1. `Order.ttl` field, default `ORDER_TTL=2`. Gate: full suite green AND byte-identical
   episodes at defaults before any other slice lands.
2. Sub-orders inherit the parent's REMAINING TTL — urgency propagates downstream or
   the signal dies at the first split.
3. Producer investment/input orders keep the hardcoded constant — isolates the change
   to the demand side; auction/restoration dynamics untouched.
4. Schedule semantics: a scripted per-cycle TTL applies to NEWLY placed consumer
   orders only. In-flight orders never mutate (a mid-flight cut would be a
   `shock_trace` event — Track B, separate).
5. The consumer TTL schedule lives in the config `consumers` block; serializable and
   generator-ref reproducible.
6. Observation contract: `ttl_remaining` normalization assumes max=2 today; variable
   TTL needs a versioned opt-in encoder change (champion stays byte-compatible at the
   old version).
7. Score unchanged: -1 per in-progress cycle and -5 per rejection stay; longer TTL
   naturally accrues more holding pressure.

Slices (each independently green):

```text
ADR + CONTEXT.md ->
Order.ttl + sub-order inheritance + byte-identical gate ->
consumer TTL schedule in config + generator param ->
obs version bump (opt-in) ->
feasibility-frontier probe
```

The probe: depth in {4,5} x consumer TTL in {2,4,6,8} x {cold, primed initial
inventory}, with a ceiling pre-flight on every opened cell. TTL ~ depth+1 .. 2*depth is
the expected feasible band; a large flat TTL (10-20) removes timing pressure entirely
and re-creates a saturated regime — explicitly not the target.

Deep-family curriculum endgame: start deep chains at TTL ~ 2*depth, anneal down as
competence rises. TTL annealing is difficulty scheduling inside the game-master design
— no topology mutation, no contract change.

### Stage 4b acceptance criteria (stage exit)

- At least one track yields a mined archive concentrated in policy_weakness rows with
  component-recoverable gaps.
- Stage-4 replay (with amendments) on that archive produces a candidate that passes
  the full promotion gates — including mined-validation improvement, since real
  headroom now exists and a tie is no longer the ceiling.
- A new champion is promoted (human-approved) and recorded in an experiment directory.

Reject if:

- All pursued tracks saturate at pre-flight or mine only joint-mismatch gaps (report
  as a claim-boundary finding and reassess the roadmap before touching Stage 5).
- A candidate is promoted on new-distribution gains while regressing the existing
  transfer/stress matrix.

## Stage 5: ACCEL-Style Local Mutation

Purpose:

Generate new frontier tasks by mutating archived high-value tasks instead of generating
from scratch.

Entry criteria:

- Stage 4b exited: a new champion was promoted from at least one Stage 4b track, so
  there is a live archive of useful, component-recoverable tasks worth mutating.

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
