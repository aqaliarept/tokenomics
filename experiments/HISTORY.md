# Experiment history — compacted outline

One document for the whole empirical record: EXP-0001 → EXP-0056 (64 directories, lettered
variants included), June–October 2026.
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
hindsight baseline on held-out topology seeds 6–9, with a hard 0% death-rate requirement.

**The measurement-gate subject** (design D1, M1) is the portable stack: the EXP-0035 promoted
producer `producer_candidate.pt` with the EXP-0012 orderer scorer `orderer_scorer.pt`. Its first
development reading on the frozen M1 gate (sparse-wide, 12/42 producers) is EXP-0045: **REJECT**.
Mean ratio over the hindsight baseline is 1.011–1.041 against margins of 1.05 calm / 1.10
shocked; there are losing calm worlds, starvation on calm 42-producer worlds, and zero deaths
(§12).

**Where M3 left it (2026-10-29; §13 "M3 exit summary").** The development audit is done
(EXP-0048–0056, ≈ 27.9 vCPU-h billed of 40). `random` set the gate's bar because it routes
orders whole and varies the receiver, not because it is random; the deterministic
`round_robin` does both and beats it on every episode of the cells `random` dominated, so a
simple rule outside roster v2 would raise the gate's bar (roster v3 is the user's call). Over a
wider space of worlds (the headroom atlas, atlas v2), a cheap informed policy beats the best
simple rule by more than chance on 13 of 32 selected worlds at 42 producers (11 of them with 14
possible makers per product) and on 11 of 64 at 12 (dense and 4-maker worlds), with larger
gains at 42; most worlds are open, with room under the analytic ceiling that no cheap policy reaches.
The depth-2 search teacher is safe but weak: it beats the bar beyond noise on 2 of 4 open
12-producer worlds, a thin call (both margins under 40 units over the one-episode floor), and
closes only 0.1–3.3% of their room. The ordering role is the one to train first: swapping it for the teacher closes the
whole gap to the teacher stack, swapping the producer about a tenth (c_O = 1.023 against
c_P = 0.118, on 3 worlds); on those worlds the trained orderer's loss to `round_robin`
needs whole routing and varied receivers together (EXP-0056, `joint`; which receiver property
matters is open). No stage of a deployed decision is superlinear up to 42 producers
(EXP-0055). Waiting on the user: roster v3, atlas worlds in a gate registry, a 42-producer
teacher read, and how M4 trains the orderer.

**What is settled.** The environment poses a real learning problem; imitation of a
strong heuristic works; a learned orderer beats the best hand-written baseline; the
policy contract is product-agnostic and transfers across topology size and recipe shape.

**What is dead.** Four separate attempts to find more headroom — a learned value
function, a curriculum game-master, a marginal-value routing teacher, and a graph neural
network imitating that teacher (EXP-0039 → 0041, §9: the label's action ranking is not
stable under future demand, so no model can imitate it) — all closed negative. Their
causes are diagnosed and recorded below; do not restart any of them without addressing
the named cause.

**What is open.** Online search works but is not deployable: EXP-0042 confidence-bounded
rollout passes its fixed-topology gate (§10) at ~16k–33k rollouts per episode. EXP-0043
(scenario rollout surrogate, teacher-cost reduction) was cancelled before running: the
baseline plan rejects it because it still rolls out at decision time. The forward plan — measurement gate, shocks, development audit, bounded
evolution-strategies training on whole-episode outcomes instead of per-decision imitation —
is the OpenSpec change `openspec/changes/establish-project-baseline/` (`design.md`,
milestones M1–M7), against the acceptance specs in `openspec/specs/`. M1 (gate), M2 (shocks)
and M3 (development audit) are done; M4 (bounded ES, the ordering role first) waits on the user's
decisions listed in §13.

**Where the code for closed arcs went.** The drivers of every closed or rejected arc (the
PPO trainer, the curriculum game-master, the marginal-value router, the graph branch, and
EXP-0043) were removed from `new-model` on 2026-10-25. They are preserved on branch
`archive-1` (commit `7aeef1f`); reproduce those experiments from there.

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
  orderer. The bar became a strict *hindsight baseline* (per seed, the best any
  baseline achieved), which is much harder than beating one named baseline.
- **EXP-0006A — is the gap missing observation? REJECT (strict).** Enriching the orderer's
  view from capacity-blind to load/makeability-aware was a partial lever: it missed the
  hindsight baseline on seeds 6 and 9, both dominated by `stock_aware + random`. A "cliff" at seed 9
  turned out to be an artifact of a buggy price observation, since fixed.
- **EXP-0006B — is the gap missing *diversity*? ACCEPT (strict).** The two seeds 6A missed
  were exactly the ones where a *random* orderer won. So the missing ingredient was
  stochasticity, not state. Learning a per-dimension `log_std` and deploying by sampling
  cleared the strict hindsight baseline on all four held-out seeds (1.026 / 1.086 / 1.057 / 1.022)
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

## 10. The online-search branch works (EXP-0042)

EXP-0040's `SEARCH_ONLY` classification did not mean the signal was useless. It meant
the future-sensitive target belonged inside a planner, where uncertainty can be sampled
at decision time, rather than inside a standalone actor.

- **EXP-0042 — confidence-bounded online rollout. PASS; depth 2 promoted.** Starting
  from fixed `blend_0.5`, search was limited to ambiguous routes, sampled common future
  scenarios at K=4/8/16, accepted only confidence-qualified +1-logit producer bumps, and
  fell back on every unresolved stream. Over 30 fresh paired episodes, depth 2 beat the
  floor on all three tasks: s45 +18.8%, s77 +13.6%, s29 +13.6%; task-average relative
  gain **+15.3%**. Worst paired episode was -7.3%, inside the registered -10% bound,
  and deaths did not increase. Depth 1 improved all three task means (+6.5% average) but
  one s77 episode fell -14.4%, so it failed safety. Depth 3 passed more strongly (+27.2%,
  no paired regression) but was not selected because the protocol promotes the shallowest
  passing depth.

**What changed.** This is the first successful use of the marginal-value routing signal
after EXP-0037–0041: online scenario sampling handles the hidden future that made direct
GNN imitation fail. It is not yet a production-latency policy. Depth 2 spent roughly
16k–33k simulator rollout calls per episode and changed only 22–45 streams; most searched
states were exact-zero at the first checkpoint. The next model opportunity is therefore
to reduce planning cost—candidate pruning or a scenario-conditioned rollout surrogate—
while depth 2 supplies the value target and `blend_0.5` remains the fallback.

---

## 11. What the record says, in one page

**Three standalone learning/mining levers were tried for extra headroom, and all three
closed negative:**

| Lever | Arc | Cause of death |
|---|---|---|
| Learned value function | EXP-0009 | No cross-topology generalization; 6 economies too few |
| Curriculum mining | EXP-0023–0036 | Distribution was already at 98.3% of its ceiling |
| Marginal-value routing teacher | EXP-0037–0038B | Target is not a function of the observation |

**Two routing levers worked:** stochastic ordering (EXP-0006B) is in the stack, and
confidence-bounded online search (EXP-0042) clears the fixed-topology value/safety gate.
The first adds routing *diversity*; the second handles hidden future sensitivity by
sampling it at decision time rather than pretending it is observable to an actor.

**The contract work is solid and independent of the value work** (EXP-0015): policies are
product-agnostic and transfer across size and recipe shape with zero contract failures.

**Known hard limits, not bugs:**
- Chains of depth ≥ 4 are infeasible against TTL = 2 (EXP-0020).
- A flat fixed-pad model cannot be portable by construction (EXP-0022).
- Over-capacitated topology families have no headroom to mine (EXP-0033).

**Current routing direction:** keep EXP-0041 and direct imitation cancelled. EXP-0042
validated confidence-bounded online rollout and promotes depth 2 on the three fixed gate
topologies, as a teacher only: deployed search breaks the bounded per-decision cost rule.
The cost-reduction follow-up (EXP-0043) was cancelled by the baseline plan; the next steps
are that plan's milestones (§0).

---

## 12. The measurement gate (EXP-0044 →)

The baseline plan's first milestone (M1, OpenSpec change `add-measurement-gate-harness`) builds
a gate that judges an agent against the per-episode best of 40 hand-written baselines. It uses
dense and sparse-wide world families at 12 and 42 producers (the large size moved from 120 to
42 on 2026-10-27). Before any agent is judged, a roster-only screen fixes the development
worlds, checks that the baselines leave headroom, and calibrates the side-effect bounds X and Y.

- **EXP-0044 — calibration screen. HEADROOM FAILED on dense-12.** The screen ran all 22,528
  roster episodes. The median best-baseline fill was 0.954 on dense-12, above the 0.95
  ceiling. 39 of its 64 episodes were filled to ≥ 95%, mostly by `entity_planner/random`;
  this is one screen, and the spread across repeat screens is not measured.
  Dense-42 (0.925), sparse-wide-12 (0.934) and sparse-wide-42 (0.886) pass. No world needed
  a reserve. Dense-12 is the `build_scale_config` economy, which EXP-0033 already found
  over-capacitated (98.3% of exact demand). Why dense-42 keeps headroom is not isolated. The
  user-approved response (D10, tasks 10.2–10.5) keeps the 0.95 ceiling: retune the dense
  family at both sizes with one family-level scale, probe candidate scales on baseline
  evidence only, then screen again. The champion report (12.1) waits for an admissible world
  set.
- **EXP-0044 retune probe (10.3). Capacity scale does not work.** Dense capacity cut to
  0.95–0.8× left dense-12's probe median at 0.955–0.962; dense-42 fell only from 0.926 to
  0.912. On calm worlds (the only ones where the h = 20 vs h = 100 subtraction is matched;
  post-hoc review) the best baseline's shortfall is mostly warm-up loss while the depth-3 chain
  fills: 54% of it in cycles 1–20 at 12 producers and 85% at 42, with 97.5% / 99.5% served
  after cycle 20. Capacity stays about 12× above demand even at 0.8×. Shocked worlds were not
  diagnosed. The retune lever has to change: what the headroom check measures, demand pressure,
  or a much deeper cut. That is a user decision.
- **EXP-0044 attempt 2 (10.5). Sparse-wide screen passed.** M1 now gates on sparse-wide only
  (user decision after the probe). The decision rests on the headroom statistic itself, which
  sparse-wide passes and dense-12 fails. The warm-up split is only a side note: on calm worlds
  sparse-42 keeps clear post-warm-up headroom (0.89 served after cycle 20), but at sparse-12 the
  most frequent winner serves 0.99 after warm-up, so headroom there is weak. Headroom medians: 0.934 at
  12 producers, 0.886 at 42. No world needed a reserve. Pruning dropped 35 of 40 baselines:
  only the five `entity_planner` producers ever set the per-episode best. X = 0.7297, Y = 0.7681
  (q 0.05). The frozen world registry, roster and side-effect parameters are ready for 11.1 and
  the champion report (12.1).
- **EXP-0045 — champion development report (12.1). REJECT, the M1 starting point.** The
  portable stack (EXP-0035 producer + EXP-0012 orderer scorer) on the frozen gate: 24
  sparse-wide development worlds, the 5-member `entity_planner` roster as the hindsight
  baseline, X 0.7297 / Y 0.7681.
  - Mean ratio: 1.011 (12 calm), 1.033 (12 shocked), 1.038 (42 calm), 1.041 (42 shocked),
    against 1.05 / 1.10.
  - 5 losing calm worlds (worst 0.941, 12-calm-03).
  - Run-down on one world (12-shocked-00, 8 of 10 episodes).
  - Starvation on 6 calm worlds (5 of them at 42) and one shocked episode.
  - Zero deaths, no contract failures; `--mode recompute` MATCH.
  The stack is a few percent better than the strongest hand-written producer but far from the
  margins. The cause of the starvation (orderer split vs producer investment) is not yet
  diagnosed. This opens the measurement arc's baseline for M2+.
- **EXP-0046 — M2 exit, champion shocked and unretrained (add-shocks-i-drift 8.1). CONTRACT
  HOLDS.** Both stacks, never trained on shocks, ran the `shocks-i` profile (capacity hits and
  producer outages) at horizon 100, each shocked episode paired with its shock-free twin (same
  seed and demand, shocks removed). Portable stack (EXP-0035 producer + EXP-0012 orderer) on the
  8 registry v2 shocked development worlds at 12 and 42 producers; flat champion
  (`champion.json`) on topologies 6–9. 120 episodes + 120 twins at commit d04ee94.
  - 0 contract failures, 0 deaths, no missing rows: the M2 exit criterion is met.
  - Units against the twin: 0.994 (portable 12 and 42), 0.999 (flat); worst episode 0.955.
  - 678 of 680 shocks recovered, median 0 cycles; outages dip 3–6 units each, capacity hits
    about 1. The 2 unrecovered rows are outages starting at cycle 90, censored by the horizon.
  `shocks-i` barely dents these stacks. Whether shocked worlds still separate policies is the
  calibration question of EXP-0047. ≈ 0.41 vCPU-h billed.
- **EXP-0047 — registry v2 calibration screen (add-shocks-i-drift 9.1). HEADROOM PASSED.** The
  v1 screen repeated on the v2 drafts: shocked worlds now carry `shocks-i`, and roster v2 adds
  the `shock_reactive` ordering baseline (48 members). 12,800 roster episodes on the 24
  sparse-wide development worlds at commit ce6b367.
  - No reserve was needed.
  - Headroom median best fill: 0.9037 at 12 producers, 0.8749 at 42 (v1 0.9342 / 0.8856). The
    16 calm worlds are identical to v1; `shocks-i` lowers the best fill on the 8 shocked worlds
    by 0.03–0.07, so v2 leaves more headroom.
  - Pruning keeps 6 of 48: v1's five `entity_planner` members plus
    `entity_planner/shock_reactive`, which sets the best on 4 shocked 12-producer episodes.
  - X 0.7516 (v1 0.7297), Y 0.7302 (v1 0.7681); the change comes from the shocked worlds.
  The frozen v2 registry, roster and parameters share `screened_on` and wait for task 10.1.
  ≈ 4.0 vCPU-h billed.

## 13. The development audit (M3)

M3 (OpenSpec change `add-m3-development-audit`) audits the development bar before any teacher
is read against it. Its first question is why the `random` ordering baseline sets the hindsight
baseline on most registry v2 development episodes (95 of 128 in EXP-0047, all 40 on
42-producer shocked worlds).

- **EXP-0048 — random-routing diagnosis. Diagnosed: rotation plus whole orders, not
  randomness.**
  21 ordering routers, each with the `entity_planner` producer: the six roster v2 members and
  15 diagnostic routers that change one thing at a time. They ran on all 128 development
  world-episodes with R = 5 and the routing tracer on. A deterministic rotation,
  `round_robin` (whole order to a producer that rotates with the cycle), recovers random's edge
  in all three random-dominated cells (12-shocked, 42-calm, 42-shocked: 4.72 / 2.51 / 1.62 of
  it, +52 to +125 units). It beats `random` on all 104 of their episodes. Fixed random weights
  held for the episode (`random_fixed_episode`) are among the worst routers and lose 89–241
  units to it; the pre-registered whole-order control `proportional_whole` recovers in none but
  also herds every orderer onto one producer, so it mixes rotation with herding. Random
  variation over time helps too; rotation helps somewhat more. Not separated: alternation over
  time against the balance of a fixed assignment (no balanced fixed assignment was run).
  Routing whole is the second factor: at the same kind of variation, whole instead of split is
  worth 0.78–4.04 of the edge, and split orders waste completed units. The two combine:
  rotating whole orders is best. Inside the rotation factor is a rule defect: `even_split`
  always gives the odd unit to the lowest-index producer, and rotating that unit
  (`even_split_rotating`) recovers the edge in 2 of 3 cells; a random leftover gains most of
  the same units but stays under the bar, because the order is still split.
  Packing (load-aware netting), the investment-fund channel (one-class random routers) and
  replicate averaging (R = 1 gives the same edge) are ruled out by their checks. The tracer reproduces EXP-0047's
  roster units exactly (1,280 rows). Design D3 now applies: a rotating, whole-order rule
  would enter a roster v3 under a new name, through a re-screen that raises the bar. That is the user's
  decision; M3 stops for it. The mechanism behind rotation's gain (fewer reservations expiring
  while they wait for inputs: 228.7 against 397.2 lost units per 42-shocked episode) is not yet
  traced. ≈ 4.4 vCPU-h billed.
- **EXP-0049 — informed routing and the trained orderer's routing. Diagnosed: spare capacity is
  not the lever; the trained orderer learned the time variation, not the whole orders.**
  Two follow-up questions from the user after EXP-0048. First, can a rule that reads each
  producer's spare capacity spread whole orders better than the blind `round_robin`? Not the
  three rules tested on registry v2, where every product has two eligible producers; informed
  spreading at fanout ≥ 3 (e.g. the power of two choices) is untested. Spare capacity almost
  never binds at routing time (orders of 1.6–1.9 units; the smaller of the two eligible
  producers' spare is 13–22 units on average; units lost to bids turned away for lack of room
  ≤ 0.15 per episode; a skip-if-full rotation skips on 0–2.7% of decisions and equals
  `round_robin`). Following the signal makes an agent keep choosing the same producer over time:
  the argmax rule (`two_choice_spare`, which every product's fanout of 2 reduced to an argmax)
  changes producer on 17–30% of decisions against blind `random_whole`'s 50% and loses 15–152
  units per episode, almost all as expiring reservations; spare-weighted draws vary a little
  less and lose up to 39. Receiver change tracks units in every cell; per-cycle herding does
  not (`round_robin` herds at 0.89–0.90 on 12-producer worlds and wins).
  Second, did the trained portable orderer learn random's trick? Partly: it changes the
  receiving producer as often as `round_robin` (0.64–0.67), with no sampling, but still splits
  29–40% of consumer orders (`proportional`: 45–59%; the scorer was distilled from an
  input-aware target, log of the units a maker can finish now, not from `proportional`). It
  beats `random` on 126 of 128 episodes and every roster v2 member in every cell, but loses to
  `round_robin` on the shocked worlds (−28 and −14 units). So the roster v3 candidates stay
  `round_robin` and `even_split_rotating`, and M4 needs whole-order routing, not spare capacity
  or sampling. The power-of-two-choices idea stays untested (needs fanout ≥ 3). The reference
  rows reproduce EXP-0048 (2,304 rows) and `portable` reproduces EXP-0046 (80 of 80). ≈ 2.5
  vCPU-h billed.
- **EXP-0050 — routing versus producers per product. Routing matters far more with more
  possible makers; the trained orderer leads at wider fanout; why is undiagnosed.**
  Registry v2 gives every product exactly two possible makers. The same arms ran on family
  worlds with 2, 4, 7 or 14 makers per product (sparse_wide at 12 and 42 producers, and the dense
  family), calm and shocked, 4 worlds per cell. At 42 producers the gap between `round_robin`
  and the best hand-written roster rule grows from +116 / +150 units per episode at fanout 2 to
  +335 / +421 at fanout 7 (cells are not paired across fanouts, so this is a pattern). Routers
  that split lose more as makers multiply: `random` splits more at higher fanout (1.07 → 1.42
  makers per order at 42 producers) and routing its choice whole (`random_whole`) recovers +254 /
  +261 at fanout 14. (`even_split` collapses at fanout 7, but mostly through its lowest-index
  tie-break, so it is not counted as evidence.) Lack of room is never the cause (no-fit losses ≈ 0). The spare-capacity
  routers still do not beat blind choice: `two_choice_spare`, now really picking 2 of k, loses to
  `random_whole` in 14 of 16 cells and gains at most 0.4% at fanout 14. The trained portable
  orderer beats every hand-written rule and `random` in all 16 cells by mean (clearly, t ≥ 3.2,
  in 12 and 13), clearly beats `round_robin` at 42 producers with fanout 4 and 14 (+153 / +151
  at 42-f14), ties at fanout 7 and loses at fanout 2 (clearly in the 2 shocked cells); its leads
  on 12-producer fanout-4 and dense-12 worlds sit in near-saturated cells. It splits as much as
  the failing rules (1.6–1.8 producers per order at fanout ≥ 4) but its split orders complete
  (10–15 wasted units against 126–271) and it loses the fewest units to expiring reservations,
  which qualifies EXP-0049's "M4 needs whole-order routing": where the parts go matters at least
  as much as whether the order is split.
  Why its parts complete is not diagnosed. Its distillation target is input-aware (log of
  `makeable_qty_now`, capacity and inputs in stock; `proportional` weighs capacity only), which
  points at input availability; the deciding check is the teacher itself as a router, its
  whole-order version, and a trace of expired reservations by the producer's input stock. Registry v2's fanout 2 is where routing matters least; a registry v3 could add
  wider-fanout 42-producer worlds (12-producer fanout-4 and dense-12 worlds are near saturation).
  ≈ 3.9 vCPU-h billed.

### The headroom atlas

EXP-0048–0050 left the development bar in an easy corner: registry v2 gives every product two
possible makers, where routing matters least, and on EXP-0050's 12-producer fanout-4 and dense
family cells the best simple rule fills 97–99% of consumer orders, so a learner has little to
show there; the differences between routers grew only as the number of possible makers grew. The atlas (design D10–D12, ADR-0057) asks where in a wider space
of worlds a better policy is reachable at all, before any teacher or training is spent. It is not
a reopening of the §7 mining arc (EXP-0023–0036), which mined topology and capacity tightness
against a search bar and found none (EXP-0033: 98.3% of the bound). The atlas differs in the
three places that arc failed: it varies seven axes (size, choice, tightness, order size,
heterogeneity, shocks, demand pattern) instead of topology alone; it measures room per world
against an analytic ceiling (the restored B2 and base-capacity bounds) before any training, so a
saturated world is filtered rather than mined; and it calls a world hard only when a cheap reach
arm already beats the bar by more than a pre-registered paired noise floor (EXP-0025's lesson),
with a split of the selected worlds recorded before confirm reads them.

- **EXP-0051 — atlas screen. Provisional: saturation is rare outside the easy corner; measured
  headroom is narrower than the screen's labels.** 512 worlds at 12 producers and 128 at 42 (one episode each) plus 8 EXP-0050
  anchors, played by the eight blind and hand-written bar arms and three reach arms, all with
  `entity_planner`. Only 21 of 512 worlds at 12 producers and none of 128 at 42 are saturated: the
  low-tightness, small-order corner (10 by fill, ρ ≤ 0.23) and 11 over-demanded worlds whose bar
  already reaches 97% of the ceiling. 213 / 68 worlds are labelled hard (median gain ≈ 5–6% of the bar,
  12% at 42 producers with 14 makers per product, where 22 of 26 worlds are hard against 2 of 26
  with 2 makers) and 274 / 58 open (high tightness, bar fill ≈ 0.4). The reach argmax is
  `makeable_weighted_whole` (whole orders to a maker drawn ∝ what it can finish now) on 226 of 281
  hard worlds. The anchors reproduce EXP-0050 exactly (80 rows) and sit where EXP-0050 put them
  (12-f4 and three 42-producer cells saturated; dense-12 just under the fill cut). 96 worlds are
  selected (72 hard, 24 open) and split (train 60, validation 16, held-out 16, out-of-region 4).
  The hard labels are an upper bound: the pre-registered floor borrowed EXP-0050's replicate
  spread (2.6 units at 12, 10–21 at 42, from easy-corner worlds), while the screen's own
  `random_whole` replicates differ by 46–77 and 156–343 units, 18–35 times more; with that spread
  only 63–71 worlds at 12 and 34 at 42 stay hard (DIAGNOSTICS 21). The screen also rests on one
  episode per world and a stochastic arm played once; EXP-0052 confirms on fresh seeds with its
  own pooled spreads. ≈ 7.3 vCPU-h billed.

- **EXP-0052 — atlas confirm. Measured headroom is thin at 12 producers and real at 42 with wide
  choice; atlas v1 frozen.** The screen's 96 selected worlds replayed on fresh seeds (2 episodes,
  stochastic arms R = 2, `portable` reported; bar and reach arms fixed from the screen; ceiling
  with task 4.7's input-output bound), with spreads pooled from the confirm's own runs and a 3 × SE
  floor. Of the 72 selected hard worlds, 4 of 48 stay hard at 12 producers and 10 of 24 at 42; all
  other selected worlds are open (none saturated, flat or collapsed). The confirm's own spreads are
  large (σ_ep 164 / 156 units, σ_rep 90–131 / 315; median floor 33% / 15% of the bar). The count
  is D12's pre-registered one, but both spreads are concentrated in a few worlds (5 worlds give
  σ_ep 89% / 72% and σ_rep 69% / 87% of the sum of squares; at 42 the floor is mostly σ_rep, led
  by a `random` bar replicate that dies), so it depends on the estimator: across eight spread
  estimates (outlier worlds out of the pools, relative, median-based, bar re-chosen) the hard count
  is 2–12 at 12 and 5–17 at 42. Confirmed headroom is a 42-producer wide-choice phenomenon (9 of 14
  f14 worlds plus 1 dense; none at f2/f4/f7; median +16% of the bar), won by input-aware routing,
  mostly whole (`makeable_weighted_whole` 6, `makeable_split` 3, trained 1); at 12 producers only
  two exact cyclic worlds and one marginal world hold (12-0435's hard label is an artifact of its
  fixed bar arm dying at cycle 4). Screen gains shrank on fresh seeds at 12 (median 9.4% → 5.7% of
  the bar), not at 42. EXP-0050's deciding check found no pre-registered branch at 2/3 (input-aware
  weighting 4 / 14, splitting costs 8 / 14, scorer beyond target 2 / 14); input-aware routers cut
  the share of expired consumer reservations whose receiver lacked inputs from ≈ 0.89 to 0.71–0.73
  (more on investment orders). 12-0435's hard label is a rule gap (the fixed bar arm dies → bar 0),
  so PoC-A's pre-registered 12-producer train sample has only 2 hard worlds, one of them this
  artifact. **Task 5.3** fixed both causes and reclassified the recorded rows offline (atlas v2,
  `reclassified/`): a median-based pooled spread (σ_ep 55 / 141, median floor 145 / 349) and a
  dead-arm collapse rule give hard 11 / open 49 / collapse 4 at 12 producers and 13 / 17 / 2 at
  42; headroom at 12 now appears at dense and f4 choice (5–22% of the bar), and PoC-A's train
  sample has 4 hard worlds (0060, 0176, 0005, 0348) plus 8 open; the pre-registered counts stay
  the run's recorded result. Open
  worlds (bar fill ≈ 0.46, room ≈ 0.5 under a ceiling ≈ demand) remain the teacher's sample, their
  reachable room unproven. `portable` sits below the bar (−10% / −31%). Atlas v1
  (`atlas_v1.json`, `f241464c4bc4…`) keeps EXP-0051's split; gate files untouched. ≈ 3.8 vCPU-h
  billed on an 8 vCPU secure pod (16 vCPU unavailable; a failed 16 vCPU pod added ≈ 2.8 if its
  initialization was billed); M3 ≈ 21.9 of 40 vCPU-h (≈ 24.7 upper).

### The teacher on mined worlds (PoC-A)

- **EXP-0053 — PoC-A. A safe but weak teacher: search headroom on 2 of 4 open worlds; the
  clairvoyant arm is no upper bound.** The EXP-0042 depth-2 rollout search, re-pointed (D5) to a `round_robin`
  floor (whole orders; EXP-0048–0050's diagnosed cause: splitting floors start 100–400 units
  below the bar) with whole-order candidates plus the trained orderer's action and ADR-0056
  posterior forks, on atlas v2's pre-registered 12-producer train sample reduced to 8
  world-episodes (4 hard, 4 open; `ub`, the same search on the true future, kept on the 5 worlds
  with random futures). The teacher is never below the bar (+7 to +203 units, 1.002–1.127 × the
  bar) or its floor. On the open worlds it beats the bar by more than the noise floor on 2 of 4
  (12-0344, 12-0308; by 17–39 units beyond the one-episode floor 3 × σ_ep = 164.3, the atlas's
  159.5 being a 2-episode-mean floor), which meets D5's "at least half": search finds
  headroom the cheap arms miss there, and those two are M4's teacher-label candidates; the other
  two neither it nor `ub` beats the bar (the rule's "ceiling-limited"). It closes only 0.1–3.3% of the open worlds' room, whose ceiling is
  essentially demand (fill 24–35%). On the hard worlds it ties or trails the best cheap informed
  router (beyond trained only on 12-0005; 12-0348 below the trained orderer by more than its
  floor although that action is a candidate), overriding `round_robin` on 1.6–6.3% of searched
  streams; whether the candidate set, the depth or the two-cycle objective limits it is not
  separated. `ub`, the same search on the true future, lands 1–99 units *below* `d2` on all 5
  worlds: the search scores candidates by units gained in a two-cycle window, and on the true
  future more decisions are exactly neutral there, so `ub` keeps the floor where posterior forks
  expose small one-sided gains that `d2` takes and that pay off later (DIAGNOSTICS 23). `ub` is
  therefore no upper bound for this search, and D5's fork-limited reading is not determinable
  here. The posterior resolved every decision (16 / 16, 0 unresolved). For M4: a teacher exists
  in D5's sense on 12-producer mined worlds; what limits it (candidates, depth, the myopic
  window, forks) is open and needs a discriminating check first. ≈ 3.1 vCPU-h billed; M3 ≈ 25.0 of 40
  (≈ 27.8 upper).

### Which role to train first (PoC-C)

- **EXP-0054 — PoC-C. The orderer: swapping it for the teacher closes the whole gap, swapping
  the producer 12%.** On the 3 shocked world-episodes of PoC-A's sample as run (D6: 12-0060
  hard, 12-0111 and 12-0344 open, burst demand + `shocks-i`, one episode each), the portable
  stack S delivers 2,440 units (26%) fewer than the teacher stack T (`entity_planner` + `d2`).
  Swapping only the orderer for `d2` (A, the search planning with the portable producer) closes
  c_O = 1.02 of that gap; swapping only the producer for `entity_planner` (B) closes c_P = 0.12;
  the difference 0.90 is far outside D6's 0.15 tie band, every world reads the same way and
  A − B clears the one-episode noise floor (3σ_ep = 164.3 units) on all three, so by D6's rule
  **M4 trains the ordering role first** (not a tie; the choice does not go to the user). The
  secondary readings mostly sit inside that floor: A ≈ T (c_O ≈ 1), and the producer swap and
  the interaction (−344 summed) are distinguishable from zero only on 12-0111. Against
  EXP-0053's same-digest `round_robin` rows, 72% of the gap is the trained orderer trailing plain
  whole-order rotation under `entity_planner` (by 274–891 units per world, above the floor on
  all three), 16% is the search's gain over that floor and 12% the producer. Why the trained
  orderer trails rotation is not diagnosed: it is DIAGNOSTICS 19's symptom, and EXP-0053's traces
  fit both splitting (f) and less receiver rotation (b); the discriminating check (the trained
  orderer forced whole against itself and `round_robin` on these worlds, a new router first)
  decides how M4 trains the orderer, including whether `d2` labels are the source. T had to be rerun (task
  8.1 changed the search worker, so the teacher-path fingerprint differs) and reproduced EXP-0053
  exactly, units and search counts. Limits: the producer reference is hand-written, not a
  teacher; 3 worlds, one episode each, 12 producers only. ≈ 2.6 vCPU-h billed; M3 ≈ 27.6 of 40
  (≈ 30.4 upper).

### Why the trained orderer trails rotation (DIAGNOSTICS 19 split check)

- **EXP-0056 — split check. Joint: neither whole routing nor `round_robin`-like receivers
  alone recovers the trained orderer's loss to `round_robin`; both together do.** On PoC-C's 3
  shocked burst world-episodes (D13), with `entity_planner`, the trained orderer (TR) trails
  `round_robin` (RR) by 274 / 891 / 589 units, reproduced exactly from EXP-0053. A 2 × 2 over the
  same scorer separates its split rule from its receiver rule: TW sends each order whole to the
  scorer's top producer, RS keeps the scorer's split sizes but hands them out in
  `round_robin`'s rotation. Removing splitting alone (TW − TR) recovers −0.48 / 0.61 / 0.34 of
  the gap, changing receivers alone (RS − TR) −0.02 / 0.23 / 0.09; neither reaches D13's 2/3 on
  any world, so the pre-registered reading is **`joint`** on 3 of 3. Each change pays mostly
  once the other is made (RR − RS = 0.86 and RR − TW = 0.65 of the summed gap; interaction
  +411 / +145 / +335, above the 164.3 floor on 2 worlds, but 12-0060's also carries TW's failure
  on producers' own purchases there, investment rejection 0.56 and mean capacity 194.7 against
  227.9–242.3, so the clean above-floor complement evidence is 12-0344 alone). The traces describe it (not a causal
  test): the scorer sends 98–99% of a product's concurrent orders to one producer, which
  brings displaced bids and expiries with inputs in stock; its splits move part of that load
  elsewhere, so going whole under the same picks adds displacement, while under rotation the
  splits mostly waste capacity on orders that do not complete. Limits (D13): an outside cause
  is undetectable, `joint` is the remainder (the interaction's sign carries the complement
  claim), the receiver factor covers spread, rotation and which producers alike, and the split
  factor is the whole-routing rule, not the mechanism; both factors also change how producers'
  own purchases are routed. For M4 this resolves DIAGNOSTICS 19's EXP-0054 note: on these 3
  shocked burst-demand 12-producer worlds (one episode each) the cheapest rule changes that
  close the gap are whole orders plus `round_robin`-like receivers together, so M4's orderer
  needs both abilities available (route whole, vary receivers); which receiver property
  matters (spread, rotation, which producers) is open. ≈ 0.1 vCPU-h
  billed; M3 ≈ 27.9 of 40 (≈ 30.7 upper).

### Deployed-path cost (PoC-B)

- **EXP-0055 — PoC-B. No stage of the deployed path is superlinear up to 42 producers.** The
  portable stack (EXP-0035 producer + EXP-0012 orderer) was timed per stage (`obs`, `forward`,
  `decode`, `domain`, `harness`) on calm `sparse_wide` worlds along three separate axes
  (producers 12 / 24 / 42; fanout 2 / 4 / 7 / 14 at 42; consumers × 1 / 2 / 4 at 42, fanout 4),
  3 world seeds × 3 repeats, horizon 20. Against D7's rule (a per-decision slope > 1.1 on the
  fanout or the consumer axis is a "Superlinear stage"), nothing is flagged in any measure:
  per orderer decision the cost is flat in fanout (slope 0.03, ≈ 280 µs), per producer decision
  it grows 0.48 (399 → 1,016 µs; its `forward` 0.54 is the run's largest flagged-axis slope,
  0.52–0.56 under per-repeat / min-max refits; why it grows is not profiled), and every
  consumer-axis per-decision slope is ≤ 0.25. Observation build and network forward are ≈ 80%
  of the time; the simulation is 1.5–3.5%. On producers at fixed fanout the orderer decision
  grows with slope 0.34 (`obs` 0.52), not the ≈ 1 that every stream rebuilding every producer's
  rows would give, so the DIAGNOSTICS 17 blow-up has not come back (per cycle 1.24; DIAGNOSTICS
  17's pre-fix ≈ 2.2 is context only, not like-for-like: different subject, range and host). The
  bounded per-decision cost rule holds at the 12 → 42 target, so the D2
  refactor (eligible-only rows) is not needed for cost there; it would matter only if economies
  above 42 return (capped by the user, 2026-10-28). A horizon-100 portable episode at 42 producers
  costs ≈ 5–19 pod s, so M4/M5's cost will be the teacher's, not the deployed path's. Fresh
  4 vCPU pod (no 8 vCPU secure capacity), ≈ 0.23 vCPU-h; M3 ≈ 27.8 of 40 (≈ 30.6 upper).

### M3 exit summary (2026-10-29)

**What M3 asked, and what it found, in plain English.** M3 set out with five questions. Is the
development bar a fair one, given that the `random` routing baseline sets it on most episodes?
Where, in a much wider space of worlds, can a better policy than the best simple rule be reached
at all? Can an expensive lookahead teacher find it? Which of the two trained roles should M4
improve first? Does the deployed agent stay cheap as the economy grows to 42 producers? Two
follow-up diagnostics joined on the way: the deciding check (answer 3) and the split check
(answer 6). Terms used below: *fanout* fN = N possible makers (producers allowed to make it) per
product, so f14 = 14 makers and *dense* = every producer of a layer makes every product of that
layer; `entity_planner` = the best hand-written producer policy; `d2` = the PoC-A teacher, a
depth-2 lookahead search over orderer actions. The answers, in order:

1. **Why `random` sets the bar (EXP-0048–0050).** Not because it is random. It sends each order
   whole to one producer, and it varies which producer from one cycle to the next. A
   deterministic rule that does both, `round_robin`, recovers at least 1.6× random's edge in
   every cell random dominated and beats it on all 104 of those episodes. The hand-written roster
   rules lose because they split orders (a split order completes only if every part does) and
   keep sending to the same producer. Reading the producers' spare capacity does not help: it
   almost never limits an order, and following it removes the variation over time. Routing
   matters more as each product gets more possible makers: at 42 producers `round_robin`'s lead
   over the best roster rule grows from +116 / +150 units per episode (calm / shocked) at f2 to
   +335 / +421 at f7. The trained portable orderer is the strongest router with many makers
   (+153 / +151 over `round_robin` at f14) but trails `round_robin` at f2 (−28.3 and −14.2 units
   on registry v2's shocked worlds), the gate's setting.
2. **Where the headroom is (EXP-0051, EXP-0052, tasks 4.7 and 5.3).** The *atlas bar* is the
   best blind or hand-written rule on a world; the *ceiling* is an analytic upper bound on the
   consumer units any policy could deliver there; the *noise floor* is how large a gain could be
   luck; a *reach arm* is a cheap informed policy (input-aware routers and the trained orderer).
   A world is *hard* when a reach arm beats the bar by more than the noise floor, *open* when no
   cheap arm does but the ceiling leaves room above the bar, *saturated* when the bar nearly
   fills demand or the ceiling. Saturation is rare: 21 of 640 screened worlds, 10 in the
   low-tightness, small-order corner where registry v2's worlds live and 11 over-demanded worlds (tightness ρ = demand against starting throughput; ρ up to 14) where the bar already delivers ≥ 97% of what the ceiling allows. The screen's
   first count of hard worlds was too high, because its noise floor was borrowed from
   easy-corner worlds whose replicate spread is 18–35 times smaller than the atlas worlds' own
   (DIAGNOSTICS 21, the borrowed-floor lesson: a stage with its own spread decides). On fresh
   seeds with its own spreads the confirm kept 4 of 48 selected hard worlds at 12 producers and
   10 of 24 at 42 (atlas v1; the selected worlds are split train 60 / validation 16 / held-out 16
   / held-out-region 4: PoC-A samples from train only, no M3 choice reads held-out, and
   held-out-region is every selected 42-producer regime-switch world). That pooled floor was in turn set by
   a few wildly swinging worlds, and one world was hard only because its bar arm died. On the
   user's "fix floor, then sample", task 5.3 switched to a median-based spread and a dead-arm
   collapse rule and reclassified the recorded rows offline (atlas v2; the median floor falls
   from 397 to 145 units at 12 producers and from 577 to 349 at 42): hard 11 / open 49 /
   collapse 4 at 12 producers and hard 13 / open 17 / collapse 2 at 42. By rate, headroom is
   denser and larger at 42 producers (13 of 32 selected worlds; 11 f14, 1 f7, 1 dense; median
   +16% of the bar under atlas v1) than at 12 (11 of 64; 6 dense and 5 f4, none at f2; gains
   5–22% of the bar, one at 83%, 12-0273); the addendum notes the earlier "isolated at 12"
   reading came from the floor. The visible cheap lever is input-aware routing, mostly whole
   (`makeable_weighted_whole`). Most worlds are *open*: the bar fills about 46% of orders and the
   ceiling leaves about half the room. Task 4.7's input-output ceiling (one linear program of the
   whole economy) did not shrink that room: it equals scheduled demand on 601 of 648
   world-episodes and is never below 0.913 of it, while no played row exceeds it (0 of 7,776).
   So the open room is either a limit the program leaves out (input lead time, the investment
   fund, inventory limits) or a matter of how the economy is run; the bound cannot tell which.
3. **The deciding check (follow-up to EXP-0050's open question: why the trained orderer's split
   orders complete).** Read on the hard worlds with 4 or more makers, none of its pre-registered
   branches reaches 2/3: input-aware weighting, splitting costs and scorer beyond its target hold
   on 4 / 8 / 2 of 14 worlds under atlas v1 and 5 / 24, 15 / 24, 7 / 24 under atlas v2.
   Input-aware routers do cut the share of expired consumer reservations whose producer lacked
   inputs, from ≈ 0.89 to 0.71–0.73. The cause stays undiagnosed.
4. **The teacher on mined worlds (PoC-A, EXP-0053).** `d2` starts from `round_robin` and, where
   shocks are hidden, plays its lookahead on forks sampled from the posterior (the shock states
   consistent with what the agents have observed). It never falls below the bar (+7 to +203
   units) on its mined sample, 8 world-episodes at 12 producers. On open worlds it beats the bar
   by more than the noise floor on 2 of 4 (12-0344, 12-0308), M4's teacher-label candidates,
   but the call is thin: the margins over the matching one-episode floor are +38.7 and +16.7
   units, 2 of 4 is the minimum for D5's "at least half", each world rests on one episode, and
   the search's own sampling variation is in no floor, so one world falling inside noise on a
   second seed would flip the reading. It closes only 0.1–3.3% of the open worlds' room, and
   whether that room is reachable at all is unproven. On hard worlds it ties or trails the best
   cheap informed router. Its clairvoyant variant `ub` (the same search on the true future)
   scores 1–99 units *below* it on all 5 worlds where it ran: the search scores candidates over
   a two-cycle window, more decisions tie on the true future, and ties keep the floor. So `ub` is
   not an upper bound for this search (DIAGNOSTICS 23). A safe but weak teacher; what limits it
   (candidates, depth, the two-cycle window, forks) is open.
5. **Which role first (PoC-C, EXP-0054).** On the 3 shocked mined worlds the deployed portable
   stack S trails the teacher stack T (`entity_planner` + `d2`) by 2,440 units. Swapping only the
   orderer for the teacher (stack A) closes c_O = 1.023 of that gap; swapping only the producer for
   `entity_planner` closes c_P = 0.118. M4 trains the ordering role first. Caveats: c_O above 1
   is not an overshoot (A − T is inside the floor on every world, so c_O ≈ 1); the producer swap
   and the interaction clear the floor only on 12-0111; c_P measures only the distance to the
   hand-written `entity_planner`, not to a producer teacher; the sample is three shocked
   12-producer worlds, one episode each. Against EXP-0053's same-digest `round_robin` rows
   (cross-run arithmetic, under `entity_planner`), 72% of the gap is the trained orderer
   trailing plain `round_robin` (by 274–891 units per world), 16% the search's gain over that
   floor and 12% the producer.
6. **Why the trained orderer trails `round_robin` (follow-up, EXP-0056).** A 2 × 2 on the same
   scorer: sending its orders whole (TW) recovers −0.48 / 0.61 / 0.34 of the per-world gap,
   handing its split parts out in `round_robin`'s rotation (RS) −0.02 / 0.23 / 0.09; neither
   reaches the pre-registered 2/3 on any world, and `round_robin` (both changes) closes it.
   Reading `joint` on 3 of 3 (D13's name for "no single change, both together"). Single changes do help on some worlds (on 12-0111 TW +544 and RS
   +202 units, both above the 164.3 floor), just not enough. Limits: `joint` is the remainder
   branch by construction, and only the interaction speaks to the two changes being complements
   (positive on 3 worlds, above the floor on 2, clean of TW's investment failure only on 12-0344,
   +335); the receiver factor covers spread across concurrent orders, rotation over time and
   which producers alike, so which property matters is open; the traces (the scorer sends nearly
   all of a product's concurrent orders to one producer, concurrent top share 0.98–0.99) are a
   description, not a causal test; three shocked burst-demand 12-producer worlds, one episode
   each. For M4, on these worlds: the orderer needs both abilities available, route an order
   whole and vary its receivers (which property is open).
7. **Deployed-path cost (PoC-B, EXP-0055).** Up to 42 producers no stage of a deployed decision
   is superlinear: the orderer decision is flat in the number of makers (slope 0.03, 268 → 285
   µs), the producer decision grows with slope 0.48, and every per-decision consumer-axis slope
   is ≤ 0.25, against the 1.1 flag. Per cycle, cost grows with slope 1.24 on producers (reported,
   not flagged: more producers mean more decisions per cycle). A 100-cycle episode at 42
   producers costs 5–19 pod-seconds, so M4/M5's cost will be the teacher's, not the deployed
   path's.

**Verdict lines** (each quoted from its `RESULT.md`):

| EXP | Verdict line |
|---|---|
| 0048 | "Verdict: diagnosed, two factors." |
| 0049 | "Not the three rules tested, on these worlds, and the reason is diagnosed." / "Partly, and without randomness." |
| 0050 | "Yes, routing matters much more with more makers, mostly because the hand-written rules break down." |
| 0051 | "Measured headroom is narrower than the screen's labels" |
| 0052 | "Confirmed headroom is a 42-producer, wide-choice (f14) phenomenon"; task 5.3 addendum: hard 11 / open 49 / collapse 4 (12), hard 13 / open 17 / collapse 2 (42) |
| 0053 | "Not an acceptance read"; "open — search headroom on 2 of 4" |
| 0054 | "D6 reading: **orderer**" |
| 0055 | "Not an acceptance read; no stage flagged." |
| 0056 | "Reading: `joint` (3 of 3 worlds)." |

**Pod spend.** Billed vCPU-h per record: EXP-0048 4.4, EXP-0049 2.5, EXP-0050 3.9, EXP-0051 7.3,
EXP-0052 3.8, EXP-0053 3.1, EXP-0054 2.6, EXP-0055 0.23, EXP-0056 0.1: **≈ 27.9 vCPU-h** of the
40 vCPU-h M3 budget; ≈ 30.7 if the failed 16 vCPU pod before EXP-0052 (≈ 2.8 vCPU-h-equivalent)
was billed, which is not confirmed. On 2026-10-29, after ≈ 27.8 vCPU-h (≈ 28, before EXP-0056),
the user reset the M3 budget counter to a fresh 40; against it M3 used ≈ 0.1 (recorded in
EXP-0056's RESULT.md and `sizing.md` §EXP-0056).

**The user's decisions during M3.**
- 2026-10-28: after EXP-0048, two follow-up questions (can an informed rule spread whole orders
  better; did the trained orderer learn random's trick), which became EXP-0049.
- 2026-10-28: after EXP-0049, "we have only 2 factories to choose from — increase the number"
  (EXP-0050).
- 2026-10-28: replace "PoC-A on the gate worlds" with the headroom atlas (option a, design
  D10–D12).
- 2026-10-28: PoC-B capped at 42 producers; no larger economy anywhere in M3.
- 2026-10-29: experiments with a hard stop ≤ 40 vCPU-h launch without asking (per run; the
  milestone budget still binds).
- 2026-10-29: secure pods only, never community cloud; the largest secure CPU shape first.
- 2026-10-29: the input-output ceiling (task 4.7).
- 2026-10-29: "fix floor, then sample" (task 5.3: robust floor, dead-arm rule, atlas v2 before
  PoC-A).
- 2026-10-29: add the DIAGNOSTICS 19 split check to M3 before this summary (design D13, tasks
  11.2 / 11.3, EXP-0056).
- 2026-10-29: M3 budget counter reset (above).

**Decisions still with the user.**
- **Roster v3** (design D3): either add `round_robin` (and `even_split_rotating`) as roster
  members under new names, through a roster v3 draft and a baseline-only re-screen (≈ 4 vCPU-h)
  that raises the gate's hindsight baseline, or keep roster v2 and its bar unchanged.
- **Atlas worlds or families in a gate registry** (design D12), and which regions: the evidence
  points at the 42-producer wide-choice hard worlds (f14); the 12-producer dense / f4 hard
  worlds are a smaller second region; or none.
- **A 42-producer teacher read** (PoC-A / PoC-C at 42 producers; design D5, D8): its spend.
- **How M4 trains the orderer, given EXP-0056** (scoped to its three worlds): on those worlds the orderer needed both abilities, route whole
  and vary receivers (which property is open). Options for the
  label source: diagnose what limits `d2` first (root-cause rule), `d2` labels on its two thin
  open candidate worlds, or `round_robin`-like targets.

**Open, undiagnosed** (the root-cause rule asks for these before M4 designs training on them):
what limits `d2` (candidates, depth, the two-cycle window, forks; EXP-0053); which receiver
property matters (spread, rotation, which producers; EXP-0056 limit (ii)); why the trained
orderer's split orders complete (the deciding check, answer 3).

Housekeeping left to this change: amend `establish-project-baseline` D4's "up to 1200
producers" in a separate change; sync the specs and archive `add-m3-development-audit`. Branch
`new-model` is ahead of `origin`; pushing is a human gate.
