# Experiment history — compacted outline

One document for the whole empirical record: 49 experiments, June–August 2026.
Replaces the per-experiment verdict table and the running progress log.

Each experiment still has its own directory (`spec.json` + `RESULT.md` + `summary.csv`)
for the full detail. This file is the map: what was asked, what came back, and what
each result licensed or killed.

**Reading the verdicts.** `ACCEPT` = the thing was adopted into the stack or the toolchain.
`REJECT` = it worked as an experiment but did not earn promotion. `CLOSED` / `FAIL` = the
branch is dead and should not be reopened without new evidence. `NOT RUN` = built and
tested, never executed against real compute.

---

## 0. Where the project stands

**The champion stack** (`tokenomics/champion.json`):

| Role | Checkpoint | Origin |
|---|---|---|
| Orderer | `m2_stream_stochastic.pt` | EXP-0007A |
| Producer (v4 clone) | `d3_rung_dagger_v4.pt` | EXP-0004 |
| Portable orderer scorer | `orderer_scorer.pt` | EXP-0012 |
| Portable producer scorer | `producer_scorer_broad.pt` | EXP-0021 |
| Promoted producer candidate | `producer_candidate.pt` | EXP-0035 |

Primary metric is `consumer_completed_units` against a strict best-of-baseline
envelope on held-out topology seeds 6–9, with a hard 0% death-rate requirement.

**What is settled.** The environment poses a real learning problem; imitation of a
strong heuristic works; a learned orderer beats the best hand-written baseline; the
policy contract is product-agnostic and transfers across topology size and recipe shape.

**What is dead.** Three separate attempts to find more headroom — a learned value
function, a curriculum game-master, and a marginal-value routing teacher — all closed
negative. Their causes are diagnosed and recorded below; do not restart any of them
without addressing the named cause.

**What is open.** One branch: a graph neural network over raw economy state
(EXP-0039 → 0041). Its first probe failed; the follow-up is implemented but has never
been run, and is blocked on a target-stability audit that was planned and never executed.

---

## 1. Finding the wall (EXP-0001)

**Question.** Depth-3 economies fail. Is the wall chain *depth* or variable order
*quantity*, and is the failing agent the producer or the orderer?

**Answer.** Depth, and the producer. A grid over (chain depth × demand quantity) with
from-scratch PPO showed depth-2 fine and depth-3 failing even at zero quantity variance,
so quantity is a separate, milder survival problem. A policy-swap probe — swap one side
of a trained pair for a baseline and see which swap recovers the score — put the blame on
the producing policy.

**Follow-up diagnosis.** Restoration misallocation, not under-ordering: the middle layer
starves (layer-1 capacity 485 → 0) while capacity sits over-reserved elsewhere.

*Verdict: depth is the wall, producer is the blocker.*

---

## 2. Teaching the producer by imitation (EXP-0002 → 0004)

The idea: stop training the producer from scratch and clone a good heuristic instead.

- **EXP-0002 — warm start. PASS.** Cloning `obs_stock_aware` cleared the depth-3 gate
  (82 → 373.6 against a 290 bar). Plain mean-squared-error cloning *failed*, and the
  reason mattered: the producer's action has a gated toggle (restore: yes/no) plus
  continuous quantities, and averaging across the gate produced 267 false restoration
  firings. Fixed with a masked-MSE + binary-cross-entropy toggle loss. Survives slot
  permutation.
- **EXP-0003 — producer-only RL on top of the clone. REJECT.** Improvement plateaued;
  the best selected policy (1.030) was indistinguishable from the clone. The wall was a
  critic whose expected value sat at ~0 — no learning signal to push on.
- **EXP-0004 — is the weak spot the teacher or the student? Fidelity gap.** At demand
  rung 0.24 the clone underperformed *its own teacher*, and the teacher matched the bar
  everywhere. So this was a cloning-fidelity problem, not an expert weakness. Cause:
  DAgger had only ever seen rung p=0. Fix was rung-staged DAgger, not RL. `v4` became
  the accepted producer clone.

*Lesson carried forward: when a student underperforms, check it against its own teacher
before blaming the teacher.*

---

## 3. A learned orderer that beats the baseline (EXP-0005 → 0008)

Imitation was proven. The next question was whether a learned agent could *beat* the
hand-written baseline rather than match it.

- **EXP-0005 — headroom audit first. PASS.** Before trying to beat the baseline, check
  there is anything to win. Demand was not saturated, and `random` routing beat
  `stock_aware + proportional` on seed 3 — proof of reachable headroom. The lever was the
  orderer. The bar became a strict *best-of-baseline envelope* (per seed, the best any
  baseline achieved), which is much harder than beating one named baseline.
- **EXP-0006A — is the gap missing observation? REJECT (strict).** Enriching the orderer's
  view from capacity-blind to load/makeability-aware was a partial lever: it missed the
  envelope on seeds 6 and 9, both dominated by `stock_aware + random`. A "cliff" at seed 9
  turned out to be an artifact of a buggy price observation, since fixed.
- **EXP-0006B — is the gap missing *diversity*? ACCEPT (strict).** The two seeds 6A missed
  were exactly the ones where a *random* orderer won. So the missing ingredient was
  stochasticity, not state. Learning a per-dimension `log_std` and deploying by sampling
  cleared the strict envelope on all four held-out seeds (1.026 / 1.086 / 1.057 / 1.022)
  with zero death — the first policy in the arc to do so.
- **EXP-0007A — make the contract scale. ACCEPT.** Ordering agents were refactored to one
  per `(source, product)` stream (ADR-0052), with an observation whose width scales with
  the number of *producers* rather than the product vocabulary. Re-distilling the champion
  on this contract preserved every strict win. This checkpoint is still the champion.
- **EXP-0007B — automate the loop. ACCEPT automation, NO PROMOTION.** A champion-iteration
  driver that runs a candidate end-to-end and writes a promotion-gated record.
- **EXP-0008 — elite-set distillation. REJECT.** Relabelling each state with the whole
  within-ε elite routing set (K=6, ε=0.1) instead of the single best did not beat the
  champion.

---

## 4. The compute lever, and the first dead end (EXP-0009)

**Question.** The orderer's teacher is an expensive 12-cycle rollout search. Can a learned
value function `V(global_state) → units-to-go` replace the tail of that rollout with a
3-cycle rollout plus a terminal value estimate?

**Answer.** ACCEPT the compute saving (3.82× fewer simulated cycles, held-out mean +0.4%)
— **but the win was not the value function.** It was short-rollout sufficiency: three
cycles of lookahead were simply enough, and the learned tail added nothing.

**Why it died.** A diagnostic found two separate things. First, a genuine optimization bug:
raw `global_state` saturated `tanh` and collapsed the model to predicting the mean — fixed
by standardizing inputs and targets, with a regression test. Second, and fatal: **no
cross-topology generalization.** Held-out R² was negative for every model tried, including
plain ridge regression. An earlier R² of 0.70 had been in-sample memorization. Six training
economies are not enough to learn units-to-go as a function of global state.

*Verdict: value lever STOPPED. This is the first of three headroom hunts to close negative.*

---

## 5. Restructuring the producer's view (EXP-0010A → 0011A)

- **EXP-0010A — entity-factored producer observation. ACCEPT.** Replaced the flat producer
  observation with a domain-owned view of capacity-type / product / input rows plus
  recipe-edge relations (ADR-0053), behind a versioned opt-in encoder. Order-equivariant,
  passes row-order / product-relabel / capacity-relabel gates, and `stock_aware` stayed
  byte-identical.
- **EXP-0010B — deterministic planner on that view. ACCEPT** the controller slice.
- **EXP-0010C — a learned scorer inside the planner. ACCEPT infrastructure, REJECT the
  controller.**
- **EXP-0010D — size transfer. ACCEPT** the local deterministic gates; no learned-scorer
  promotion.
- **EXP-0011A — replace the champion's v4 producer with the deterministic planner?
  REJECT.** Did not survive the promotion gates.

---

## 6. Making the policies product-agnostic (EXP-0012 → 0022)

The largest constructive arc. Goal: policies that carry no product or producer identity,
so the same checkpoint runs unretrained on a different economy.

**The two portable modules**
- **EXP-0012 — portable orderer** (`OrdererScoreModel` → `allocate_suborders`, ADR-0054).
  ACCEPT: matches/beats `proportional` on held-out topologies.
- **EXP-0013 — portable producer** (`ProducerScoreModel` → `plan_producer_command`,
  ADR-0053). ACCEPT: reproduces the deterministic planner's value, runs at unseen
  producer counts.
- **EXP-0014 — do they coexist? ACCEPT.** The joint portable stack holds value with no
  feasibility failures.

**Proving the transfer claims**
- **EXP-0015 — verification matrix. ACCEPT.** One reusable runner emitting the
  compatibility-and-behaviour matrix, and — importantly — *separating contract failure
  from policy quality* per cell. Contract and value are independent axes; that separation
  is what makes every later result readable.
- **EXP-0016 — size transfer. ACCEPT.** Same checkpoints, unretrained, on 24 products /
  24 producers / both. Zero contract failures.
- **EXP-0017 — out-of-family recipe shape (fork/join). CONTRACT PASS, value PASS.** The
  id-free contract transfers to a different recipe *shape*, not just a different count.
  Joint stack 2.86× the bar. **Caveat:** the producer *alone* regressed on seeds 8 and 9
  (0.45 / 0.83) — a policy gap out of family, masked by a strong orderer.
- **EXP-0018 — fix that with deterministic planner gates. MATERIALLY REDUCED, not closed.**
  Producer-alone minimum went 0.450 → 0.955; 3 of 4 seeds ≥ 1.0. Strict "no seed below
  1.0" still unmet.
- **EXP-0019 — dynamic baseline audits. BOTH CLEAN.** Zero contract failures. The portable
  orderer beat every deterministic orderer on all 13 targets and was the *only* orderer
  exactly invariant to producer-row order.
- **EXP-0020 — broad fixture matrix, 22 fixtures × all stacks. ZERO schema and contract
  failures.** Joint portable stack mean 1.231× the bar. Surfaced a real scaling limit:
  `deepjoin-d5` completes zero units *for every policy including the deterministic
  planner* — chains of depth ≥ 4 are infeasible against TTL = 2. That is a feasibility
  wall, not a policy failure.
- **EXP-0021 — broaden producer training. PRIMARY TARGET CLOSED.** Distilling over 54
  topologies across five families closed the fork/join seed-9 regression (0.955 → 1.000).
  One residual cell remains: uneven-strength seed 6 at 0.975.
- **EXP-0022 — do the recipe edges earn their keep? EDGES = VALUE, NOT PORTABILITY.**
  A fixed-pad flat MLP raises an error the moment any entity count exceeds its build pad —
  so flat models are non-portable *by construction*, and the per-row structure is what
  buys id-free portability. Zeroing the edges stays fully portable and ties the champion on
  19 of 22 fixtures, losing only on 3 contention cells (scarcity, shared capacity) by
  0.8–2.5%. Edges are a small, strictly non-negative value add under scarcity.

---

## 7. The curriculum game-master, and the second dead end (EXP-0023 → 0036)

The idea: if hand-picked topologies are exhausted, *mine* hard cases automatically —
generate worlds, find the ones where the champion underperforms a reference bar, archive
them, and replay them as training data.

**Building the machine (all worked)**
- **EXP-0023 — scaffolding.** Replayable task archives with per-row failure classification.
  First blocker identified: the reference bar was too weak to expose weakness.
- **EXP-0024 — a stronger bar.** Per-cycle directed short-rollout orderer search. First
  in-claim `policy_weakness` in the arc.
- **EXP-0025 — mining at scale works.** Also established that the paired-seed noise floor
  is load-bearing: without it, coin-flips read as real gaps.
- **EXP-0028 — mine a topology *grid*, not one seed.** Fixed the single-cell archive
  blocker; produced a diverse 26-task archive.

**Trying to harvest it (all failed)**
- **EXP-0026 — prioritized replay. REJECT.** A narrow single-cell archive closes nothing
  and regresses out-of-family.
- **EXP-0027 — rerun after fixing two real data bugs** (replay was inert; the orderer had
  trained on stand-in configs rather than the archived ones). **REJECT stands** — and is
  now scientifically valid rather than a training-data artifact.
- **EXP-0029 — augment instead of replace. AUGMENT WORKS.** Training on the champions'
  original broad sets *plus* mined replay closed the producer regression.
- **EXP-0030 — freeze the orderer, augment the producer.** Removed the orderer-noise
  regressions. Later PROMOTED by human override.
- **EXP-0031 — gap-attribution diagnostic. Producer replay closure ~5%.** A cheap
  cross-stack read (no training run) on the saturated family: replaying mined tasks into the
  producer recovered almost none of the gap. The first hard signal that this family was the
  problem, not the method — confirmed by EXP-0033.
- **EXP-0032 — can the orderer represent the search bar's routing? INVALID PROBE.** The
  two chosen "high-gap" tasks had ~0 orderer headroom under the frozen producer — the read
  was confounded.
- **EXP-0032b — rerun over all 26 tasks. DECISIVE: orderer lever spent.** Zero of 26 tasks
  had ≥2% orderer headroom against the bar. STOP.
- **EXP-0033 — is the champion near the true ceiling? NEAR-BOUND.** An offline
  achievable-units upper bound (exact seeded demand, direct capacity cap, transitive
  recipe-capacity cap) showed the champion delivering **98.3% of exact demand**. The grid
  was over-capacitated; the agent was effectively ideal on that distribution. *This is why
  the arc found nothing: there was nothing there.*

**One real recovery**
- **EXP-0034 — mine in a *capacity-binding* regime instead.** Pre-flight keeps only worlds
  whose no-slack capacity ceiling sits below demand. Premise validated: 146 of 400 mined
  tasks showed weakness. The candidate mean-improved (val +14.8u, blind +8.3u) but traded
  the tail (worst blind task −107u), tripping the conservative gate. **Only ~15% of mined
  headroom was captured** — the open question.
- **EXP-0035 — root-caused and fixed. FIDELITY WALL BROKEN.** The low harvest was
  multi-task optimization interference: one scorer across 121 topologies at 400 epochs
  averages away per-task investment behaviour. The fix was a single knob — 2000 epochs.
  Closure went 0.159 → **1.096** train, 0.19 → 0.860 validation, 0.12 → **1.000** blind.
  Tasks worse than champion: 26/57 → 3/57. The −107u tail task healed to +61.7u.
  **PROMOTED.**
- **EXP-0036 — round 2, primal-tight worlds. Headroom is large and ORDERER-owned.**
  81 archived tasks, median gap 53u (round 1: mean 3.67u). Search-orderer cross-recovery
  reclaimed 89% of the gap; the producer only 14%. But the orderer probe ladder then
  failed: the headroom lives in a *named-policy vocabulary* (capture 0.90 with a blend
  vocabulary), yet **choice learnability failed 3 of 3** — held-out top-1 accuracy below
  the majority-class floor while training cross-entropy fell to 0.10–0.17. Fits train,
  anti-transfers.

*Verdict: the curriculum arc is CLOSED. It built working machinery and proved the
distribution had no headroom to give. Future mining needs capacity-binding regimes by
construction.*

---

## 8. The marginal-value routing teacher, and the third dead end (EXP-0037 → 0038B)

If the orderer cannot *select* among policies, can it learn direct value-based routing
weights instead?

- **EXP-0037 — value residual over a fixed blend. BRANCH CLOSED (1/3).** Policy = fixed
  `blend_0.5` base plus a clipped learned residual per eligible producer; label = the
  marginal value of a logit bump. Judged by deployed rollout: seed 29 −3.9%, seed 77 +0.5%
  (noise), seed 45 −56% *with double the deaths*. The residual added harmful noise, not
  stable improvement. `blend_0.5` remains the zero-cost orderer floor.
- **EXP-0038 — why did the actor learn the teacher so badly? SCOPED DIAGNOSIS.** Over
  13,940 labelled decisions: the current 7-feature producer row **aliases** the label
  badly. 69.4% of teacher decisions are flat; on the 30.6% that are real choices, top-1
  accuracy is 0.451. Identical observed rows recur with a median label range of 10 and a
  worst case of 90.
- **EXP-0038A — which missing information explains it? Enrichment NOT licensed.** Adding
  order and queue context collapses the alias conflict (median range 10 → 0) — but this is
  a between-decision *level* artifact. Those fields are constant across a decision's
  eligible producers, so they add no *within-decision* contrast. No probe beat the deployed
  actor floor on a single task. And 96.7% of the actor's top mistakes are on rows that
  *are* distinguishable — the actor simply ranks them wrong.
- **EXP-0038B — richer per-producer and system-wide observations. GATE FAILED; TARGET
  RETIRED.** Two fresh dumps (60 per-producer/product columns; then +32 system columns).
  Zero of three tasks beat the actor floor, for both ridge-with-interactions and kNN.
  Centered within-decision R² ≤ 0.

**The diagnosis that closed it.** The teacher's within-decision producer preference is
**not a function of the deployable observation at all** — it lives in the 8-cycle rollout.
No amount of observation enrichment can recover it. Labels spanning a 90-unit range at
fixed features are not a missing-feature problem.

*Rule adopted from this: before training on any teacher target, verify that the label is
approximately a function of the observation. Recorded as DIAGNOSTICS #12.*

---

## 9. The graph branch — closed by target instability (EXP-0039 → 0041)

- **EXP-0039 — can a relational GNN over raw economy state recover the target?
  FAIL (negative result).** Same 13,940 decisions, leave-one-seed-out, 3 model seeds,
  bitwise-deterministic. The GNN's held-out regret **equalled random** on every task
  (5.29 / 6.23 / 5.25 vs random 5.38 / 6.24 / 5.24); centered R² negative on all three;
  never beat the actor floor. Training loss fell 49 → 1.5 — it fits the training seeds and
  does not generalize. Same wall as EXP-0038B. Scoped claim: not learnable under
  `graph_primitives_v1` with this gate. No follow-up licensed.
- **EXP-0040 — is the label even stable? SEARCH_ONLY.** Replayed 424 exact EXP-0039
  states (325 signal + 99 flat controls) and collected 36,828 raw label rows: 1,116 candidates
  times an original-suffix control plus K=32 continuations. The expected selector was genuinely useful: oracle regret
  fell 38.9–44.6% versus the canonical actor on all three tasks, passing every trajectory-seed
  usefulness check. But its action ranking did not converge. At the final estimator K=16,
  task winner overlap was only 0.599 / 0.602 / 0.743 against the registered 0.80 minimum;
  all three task aggregates and all nine task/seed cells failed stability. K-prefix convergence
  was false at 4, 8, and 16. Mechanical classification: **SEARCH_ONLY**.
- **EXP-0041 — pooled MPNN as a bounded confidence-gated residual. CANCELLED, NOT RUN.**
  Fully built and locally tested, but its own prerequisite contract maps EXP-0040
  `SEARCH_ONLY` to cancellation. It carries a heavily hardened protocol:
  whole-episode train/validation/test splits rejecting every prior anchor seed; a fair
  held-out baseline retrained on identical splits; 0-hop and 2-layer controls; a nested
  2/4/8-episode learning curve as a *diagnostic that cannot act as a gate*; validation-only
  model selection behind a content-hash manifest lock; staged label collection where a
  later tranche cannot pick its test episodes with the fit already in hand; and a
  rollout-only seed registry disjoint from every labelled split. Hard argmax deployment is
  prohibited. Gate pins four artifacts by SHA-256 (all still present in-tree).

**Status.** The graph-training branch is closed without EXP-0041 label or fit spend. The
continuation-averaged target has oracle value but remains too sensitive to future demand to
serve as a stable supervised label. That is a positive case for bounded online rollout/search,
not for another direct-imitation GNN.

---

## 10. What the record says, in one page

**Three levers were tried for extra headroom, and all three closed negative:**

| Lever | Arc | Cause of death |
|---|---|---|
| Learned value function | EXP-0009 | No cross-topology generalization; 6 economies too few |
| Curriculum mining | EXP-0023–0036 | Distribution was already at 98.3% of its ceiling |
| Marginal-value routing teacher | EXP-0037–0038B | Target is not a function of the observation |

**One lever worked and is in the stack:** stochastic ordering (EXP-0006B) — the winning
insight being that the missing ingredient was routing *diversity*, not more state.

**The contract work is solid and independent of the value work** (EXP-0015): policies are
product-agnostic and transfer across size and recipe shape with zero contract failures.

**Known hard limits, not bugs:**
- Chains of depth ≥ 4 are infeasible against TTL = 2 (EXP-0020).
- A flat fixed-pad model cannot be portable by construction (EXP-0022).
- Over-capacitated topology families have no headroom to mine (EXP-0033).

**If this signal is picked up again**, EXP-0040 fixes the boundary: do not train EXP-0041
or another direct imitator on it. Test it as bounded online rollout/search, where future-demand
uncertainty is sampled at decision time; the registered follow-up is a depth {1, 2, 3} comparison.
