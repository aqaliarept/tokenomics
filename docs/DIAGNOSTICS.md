# Diagnostics registry — known problems → verification checks

Companion to the CLAUDE.md **"Root cause before next strategy"** rule. When an
experiment fails or underperforms, scan this registry FIRST: if the symptom matches a
known problem, run the listed check instead of inventing a new diagnostic or jumping to
the next strategy. Each entry states the problem in plain English, the discriminating
check (tool + command shape), and what each outcome means for the next move.

Maintenance:
- New failure cause diagnosed with evidence → add an entry (problem statement in plain
  English, check, outcome branches, EXP reference).
- Humans may add a plain-English problem statement to **"Unverified problems"** at the
  bottom without knowing the check; the agent designs and attaches the check later.
- An entry is evidence-backed: it must cite at least one experiment or task where the check
  discriminated between competing causes.

Run profile: checks marked **pod** follow the CLAUDE.md compute rule (real runs on
Runpod); **local** checks are quick one-shot diagnostics.

History: entries 1–16 were removed from this branch by the `cleanup` commit 18c717d and are
preserved in git (`git show 18c717d^:docs/DIAGNOSTICS.md`). Numbering continues from them.
Entries 6, 9, 11 and 12 are restored verbatim (2026-10-30, referenced by
add-m4-orderer-redesign); 1–5, 7, 8, 10 and 13–16 stay in git only.

---

## 6. Model looks good — is it memorizing?

_Restored 2026-10-30 from 18c717d^ (referenced by add-m4-orderer-redesign)._

**Problem.** A learned model (value model, scorer) shows strong fit where it was
trained; is it a transferable signal or in-sample memorization?

**Check** (local if data exists): held-out topology evaluation — R²/correlation on
economies never seen in fitting. In-sample metrics are inadmissible as evidence.

**Outcomes.** held-out R² ≤ 0 or corr ≪ in-sample → **memorization**: stop the lever or
add train diversity; high held-out → real signal.

**Seen in.** EXP-0009 (value model in-sample R² 0.70, held-out negative for EVERY model
class incl. ridge → value lever stopped). Also the open-loop policy collapse under
single-axis shift (stages 3–4, ADR-0038/0039 → domain-randomized training).

---

## 9. Pod run stalls / crawls at high load

_Restored 2026-10-30 from 18c717d^ (referenced by add-m4-orderer-redesign)._

**Problem.** A multi-worker pod run makes no progress; box load explodes.

**Check** (ops): SSH in, `uptime` + progress lines. Load ≫ vCPU with 0 tasks done →
thread oversubscription.

**Fix.** Every ProcessPoolExecutor worker must pin threads in its initializer:
`OMP/MKL/OPENBLAS_NUM_THREADS=1` + `torch.set_num_threads(1)`.

**Seen in.** EXP-0034 gap_attribution (16 unpinned workers → load 157, 0/119 in 62 min;
fixed in `_init_worker`).

---

## 11. Distill-from-search fails — is the TEACHER SIGNAL even learnable?

_Restored 2026-10-30 from 18c717d^ (referenced by add-m4-orderer-redesign)._

**Problem.** A policy distilled on per-cycle search's committed routings underperforms
its own teacher on its own training task, with weak in-sample agreement and huge
run-to-run deploy variance (identical recipe, different torch init → 0 vs 800 units).
Before blaming the network or the obs contract, check whether the regression TARGET
carries learnable signal at all: the search commits ONE global routing per cycle, picked
among near-tied candidates (incl. sampled Gaussian vectors), then that global choice is
sliced into per-stream local targets — the mapping local-view → committed-routing can be
mostly noise.

**Check** (local, seconds — needs saved `samples.pkl.gz` from the instrumented probe):
1. Fraction of flat targets: eligible target vector ≈ 0 after centering (search committed
   even_split — no routing signal in that row).
2. Zero-predictor baseline: masked-mse of predicting all-zeros (= route uniformly);
   variance explained = 1 − model_mse / zero_mse.
3. Top-1 agreement judged against chance = mean(1/n_eligible), not against 1.0.

**Outcome branches.**
- Variance explained low (≲ ~25%) on train states → **target/recipe failure**: the
  informative part of the teacher's decision is global + tie-arbitrary; a local
  per-stream MSE regressor is fitting noise. Before choosing a SIGNAL fix, run the
  **representation discriminator (H4)**: does the headroom live inside a NAMED-policy
  vocabulary or in continuous routings? Re-run the same per-cycle search with the sampled
  Gaussian candidates OFF (`n_samples=0`, named candidates only, still switching per
  cycle) → `search_named_cu`; `named_capture_frac = (search_named − even_split) /
  (search_full − even_split)`.
    - named_capture ≈ 1 → the single-choice noise is an **output-space** problem: the raw
      per-producer routing was the wrong target. Change the teacher output to the named
      policy the search chose (categorical {even_split, proportional, don't-care}, near-ties
      → soft labels, or a blend coefficient α per stream) — state-reproducible, so the RNG
      tie-noise vanishes, and a small categorical/scalar output is learnable. Beats denoising.
    - named_capture ≈ 0 → the win is genuinely continuous → denoise the continuous target
      (average committed routings over K independent searches, committing the mean to advance;
      margin-filter/advantage-weight commits that don't beat even_split), add global context,
      or skip distillation (deploy-time search). Do NOT spend on width/epochs/DAgger.
  Do NOT read H4 off fixed-policy baselines — a per-cycle CHOICE among named policies is
  strictly stronger than any single fixed one; only `search_named_cu` (per-cycle named
  switching) answers it.
- Variance explained high but deploy still collapses → covariate shift (entry 2's other
  branch): student drifts off the teacher's state distribution; DAgger-style collection.

**Seen in.** EXP-0036 orderer oracle-relabel probes (probe 1: 5/6 tasks deploy to exactly
0 units incl. alive worlds; instrumented retry: 42–50% flat targets, variance explained
25%/10%/64% on s29/s77/s45, top-1 0.50–0.60 vs chance 0.33–0.50; same recipe swung
0→800u across runs). **A0 gate (orderer_a0/):** H4 named_capture_frac = 1.00 / 0.69 / 0.66
(s29/s77/s45, mean 0.78) → the single-choice noise is largely an OUTPUT-SPACE problem: a
per-cycle NAMED-only search (even_split/proportional switching, no Gaussian samples) recovers
66–100% of the headroom. Denoise (H1) also lifts VE (0.27→0.52, 0.26→0.58, 0.64→0.75) but
even averaged VE tops out ~0.5–0.75. Champion orderer delivers **0 units on s77 and s45**
(< even_split's 514 / 200) — the mined tight tasks collapse the memorized champion. Fix chosen:
named/blend teacher head (Stage A1b), not continuous-vector denoising.
**A0b gate (orderer_a0b/):** blend value CONFIRMED (mean capture 0.78 named → 0.90 blend;
s77 0.69→0.95) but choice learnability FAILED 3/3 — held-out top-1 BELOW the majority floor
on every task (0.27/0.53, 0.38/0.56, 0.34/0.47) while train CE fits to 0.10–0.17: the
committed choice fits in-episode and anti-transfers across seeds. Pure per-stream
obs→choice A1b NOT green-lit. Open discriminator: fit the same classifier on the FULL
GLOBAL STATE — beats majority → missing-context (enrich obs); still fails → choice noise
(soft labels / deploy-time search), enrichment useless.
**A0c gate (orderer_a0c/), verdict CORRECTED on review 2026-07-05:** the ADR-0004 global
snapshot fails too (held-out gchoice_top1 vs floor: 0.31/0.53, 0.29/0.54, 0.40/0.42; 2/3
clear fails, 1/3 tie) — but this only kills enrichment with the EXISTING global features,
NOT observation enrichment in general. The original "no observation can predict it / obs
enrichment DEAD" read overclaimed: `encode_global` is a lossy roll-up (per-producer capacity
SUMMED ACROSS TYPES — bottlenecks/death on tight tasks are per-type; no order-queue/TTL
state; inventory as unit count not composition; no history), and the probe saw only ~200
cycles/task. Since domain+search are deterministic, the choice IS an exact function of the
FULL env state. Three causes standing: (1) lossy snapshot, (2) missing temporal/aggregated
features (history recovers hidden state given a lossy snapshot), (3) genuine
incompressibility at feasible sample size. Discriminating check = **A0d complete-state
probe**: same protocol, lossless env-state serialization ± k-cycle history stack, ~10× mined
cycles. Fails → (3) diagnosed with evidence, go A2 deploy-time light search / A1
soft-advantage labels (critic ranking gate first). Passes → enrichment alive; find the
compact features (temporal aggregates first). Trace-level teacher-vs-probe comparison also
still unread (decisive-cycle accuracy WORSE than overall says errors are not near-tie noise).
**A0t gate (orderer_a0t/):** the trace-regret read — misses are EXPENSIVE, not tie noise
(probe_regret_frac 0.98/0.86/0.56); no system-level aggregate separates even-wins from
concentration-wins cycles; champion beats the whole 5-entry vocabulary at ~40% of teacher
states (its 0-unit episodes are trajectory-level compounding).
**A0f gate (orderer_a0f/) — closes the TESTED missing-observation ladder (HANDOVER Task 1);
for these families/models/scale the discriminator lands on cause (3).** Cumulative
feature-block VALUE-TABLE probe (predicts the full candidate table, so argmax tie noise is
out of the label) over stream / per-type capacity / order-book+TTL / inventory / inbound /
4-cycle history / relational recipe-ratio blocks, 3 tasks x 3 anchored seeds: (a) NO block
set gives a stable held-out regret reduction — per task 0.90–1.14 (s29), 0.76–1.26 (s77),
0.365–0.89 (s45, where the two SMALLEST sets are best and every added block hurts: the
small-sample overfitting signature); train_regret_frac 0.0 for every set with dim >= 109
(only the 81-dim stream block underfits, 0.53/0.49/0.81) — memorizes train, transfers
nothing; (b) CONSTANT mid-blend picks match or beat the fitted probes almost everywhere
(always-blend_0.5 → 0.90/0.97/0.29; sole exception s77 +order_queue 0.7555 vs 0.7821 — a
0.027, one-task/one-seed edge); (c) 1-seed→2-seed training RAISES held-out regret on all 3
tasks (anti-transfer, not a starvation slope); (d) winner-set BCE and margin-weighted CE
don't rescue it. Diagnosis (scoped): the per-cycle lookahead table on a new demand seed
behaves as seed-idiosyncratic noise around a stable "mid-blend > even_split" mean preference
— none of the TESTED observation families, target forms, or (at this scale) more data makes
the per-cycle switch learnable; untested richer models (e.g. a graph encoder) have no
positive signal here to build on. Learner-trajectory DAgger is moot for this wall
(train/held-out are both teacher trajectories; only the demand seed differs). Branch: A2
deploy-time light search (vocabulary + champion candidates), with FIXED mid-blend policies
in the acceptance comparison as the cheap floor they proved to be. Re-fit tooling:
`orderer_a0f/analysis_followup.py` over `artifacts/feature_traces.json.gz` (local, seconds).
**A0g gate (orderer_a0g/) — the HANDOVER "Proposed Model" (producer-count-agnostic set
value table) does NOT change the A0f verdict on the same traces.** `ProducerSetValueModel`
(shared per-producer MLP → masked mean+max pool → fixed 5-candidate head; permutation/count
invariants unit-tested) fit on the identical A0f data reshaped losslessly into
[context ‖ producer-row set] (`set_features_from_blocks`): train regret ~0.0 everywhere,
held-out at/near the constant floors — s45 decisive (every set variant 0.51–0.75 vs
constant blend_0.5 0.2926); the two sub-floor readings are ~0.03 edges in DIFFERENT
variants on different tasks (s29 set+history 0.8652 vs 0.8965; s77 1-seed 0.7476 vs
0.7821); 1-seed beats 2-seed held-out on s77/s45 (anti-transfer again); cross-task pooling
(600 rows, possible only in the set view) doesn't rescue (s77 1.08 > even_split). Scope:
mean+max pool + centered-table MSE + reconstructible features; attention pooling, ranking
losses, and per-producer routed/completed/rejected history not tested. The deployable
model/feature path stays in `train/orderer_scorer.py` (`cycle_set_features`,
`fit_set_value_table`) as the portable distillation head if a later arc (e.g. A2 traces)
finds signal worth learning.

**A0w gate (orderer_a0w/) — changing the target TIME SCALE (episode/window blend-policy
returns, HANDOVER Task 5) does NOT rescue the trained orderer; the deployed learned
meta-orderer loses to a fixed blend_0.5 on all 3 tasks.** This is the discriminating
check the others could not give: instead of table error, DEPLOY the model and measure
units. `--a0w` fits the A0g set head on episode/window value tables (value of each CONSTANT
blend over a window, not the per-cycle search argmax), then rolls out champion producer +
the learned meta-orderer (conservative default-sticky switch rule) on the held-out seed.
Verdict by deployed units (the real metric — table regret is NOT trusted): learned vs
default(blend_0.5) vs best_const = s29 810/960/768, s77 666/749/893, s45 588/621/621 —
learned < default 3/3, < best_const 2/3 → STOP. The fixed blend_0.5 is strong (near champion
on s29; far above the *dead* champion on tight s77/s45) and switching only costs units (s29
960→810). Critically, table regret improved with longer windows (s77 episode regret 0.0) yet
deploy still lost, and `closure_vs_search` 0.65/0.60 was a red herring (recovery over a dead
champion). Branch outcomes: this closes the last untested lever of the A0f/A0g arc — across
A0b/A0c/A0f/A0g/A0w the deployable learned orderer never beats a fixed mid-blend on held-out
demand, so the per-cycle/window search winner is not a stable supervised target at any tested
time scale. Ship/compare against a fixed mid-blend; remaining headroom is a deploy-time light
search (Stage A2, a compute lever), not a learned policy. Scope: single threshold 0.25 (no
sweep — unwarranted since A0w didn't look promising), narrow s29/s77/s45 trio (not the 81-row
archive), attention pooling / ranking losses / per-producer routing history still untested.
Driver + deploy path: `--a0w` in `eval/orderer_representability.py`, `set_table_ordering_policy`
in `train/orderer_scorer.py`.

---

## 12. Actor fits the teacher badly — missing observation, or an unlearnable target?

_Restored 2026-10-30 from 18c717d^ (referenced by add-m4-orderer-redesign)._

**Problem.** A learned policy underperforms and its regression/fit loss is flat-high (never
drops). Tempting to blame the network, but the first question is whether the TARGET is a function
of the observation the actor actually gets. A per-decision label that is a *difference of two full
multi-agent rollouts* (marginal value of a logit bump, per-cycle advantage, etc.) can depend on
other orders, other actors, bottlenecks, and future fulfilment. Identical local observations can
therefore carry very different labels. The fix might be deployable observation enrichment, or it
might require teacher emulation/future rollout information that cannot be used in the real
application. Related to entries 11 (search-committed routing target) and 6 (memorizing).

**Check** (local, seconds — needs the per-decision dump: teacher label + actor score + exact
feature row per eligible producer, from `eval.value_router_errors`):
1. **Flat-target fraction.** Decisions where every eligible producer's label is equal (after
   centering) → the teacher has no preference there; nothing to learn. If most decisions are flat,
   the signal is sparse regardless of the actor.
2. **Mistake locality.** Among real mistakes (actor-best ≠ teacher-best, regret > 0), do the two
   producers differ on a *visible* feature? If they differ ≫ they're identical, the actor already
   has the info and missing-observation is NOT the story (a small identical-feature fraction rules
   out per-producer aliasing).
3. **Target-is-a-function test.** Hash each exact feature row; for rows seen many times, measure the
   spread of the teacher label across occurrences. Near-zero spread → label is a function of the
   features (fit failure is the actor/training). Large spread → the current observation aliases the
   target; add candidate deployable observation blocks and repeat before declaring the target dead.

**Between-decision LEVEL vs within-decision CONTRAST (the alias-collapse trap).** When a candidate
block shrinks the exact-alias label spread, that is necessary but NOT sufficient to license it. A
*routing* decision is a within-decision argmax over the eligible producers; a feature that is
**constant across a decision's eligible producers** (order quantity/TTL/product, backlog, most
system/recipe aggregates) can only shift the *level* of that decision's labels, which is exactly
what collapses the *between-decision* alias spread — while carrying **zero within-decision contrast**
to say which producer is best. So an alias-spread collapse can be a pure level artifact. Second,
discriminating check: fit a held-out probe (leave-one-seed-out; ridge WITH current×context
interactions AND a nonparametric kNN, so a decision-constant field is at least *given the chance* to
re-rank) and compare its **real-choice regret** to the incumbent floor (the *deployed* actor's own
`actor_score` argmax, which already sees the base features). License enrichment only if the enriched
probe **beats that incumbent floor on ≥2/3 tasks for BOTH probe families**. If centered within-decision
`r2` ≈ 0 and the incumbent wins, the block resolved level, not routing → do not enrich.

**Score the check correctly or it lies (EXP-0038A hazard).** This probe/floor comparison is only
trustworthy if probe and floor use an *identical* metric: (1) restrict to **real-choice decisions**
(≥2 eligible AND non-flat teacher) — flat decisions dilute every method equally and mask the signal;
(2) **decision-center** the labels before fitting (the between-decision level is the confound you are
trying to strip); score `r2` on that within-decision contrast; (3) **decision-weight** by pooling
held-out folds (each decision scored once), never average fold-means — fold-mean averaging weights
small and large folds equally and is not comparable to a decision-weighted floor; (4) make top-1
**tie-aware** on both sides. A first cut of EXP-0038A violated 1–4 and spuriously reported "+order kNN
beats actor 1/3"; corrected, it is 0/3 for both probes on every task.

**Outcome branches.**
- Large label spread at the **current** features + mistakes mostly on distinguishable producers →
  the current observation aliases the target. Continue with a deployable observation ladder
  (order/system/history/global context) — but score each block by held-out within-decision regret
  vs the incumbent floor, not alias spread alone (see the level-vs-contrast trap above). STOP this
  teacher only if the label remains unstable after the candidate deployable observation blocks; if
  only oracle/future/rollout facts stabilize it, the teacher is not deployable as an actor target.
- **Alias spread collapses when a block is added, but held-out routing does NOT beat the incumbent
  floor (and r2 stays negative)** → the added block resolved the between-decision *level*, not the
  within-decision *choice*; it is (usually) a decision-constant field → not licensed. Only
  per-producer blocks (full per-type producer state, per-producer history) can re-rank within a
  decision — test those before declaring the ladder exhausted.
- Small label spread but low top-1 / flat loss → actor/training failure (capacity, optimization,
  covariate shift) → fix the model/fit, not the target.
- High identical-feature fraction on mistakes (pure aliasing) → genuine **missing per-producer
  observation** → the dump names which field separates the tied producers; add exactly that.
- Mostly flat targets → the teacher rarely prefers anything → the lever is weak; re-examine whether
  the floor policy is already near-optimal (it usually is when flat-fraction is high).

**Seen in.** EXP-0038 actor/teacher error microscope (marginal-value residual orderer, gate tasks
s29/s77/s45, 13,940 decisions). Flat-teacher 69.4%; on the 30.6% real choices top-1 0.451, mean
regret 4.32 (top 5% of all decisions = 65% of regret). Mistakes: only 3.9% between identical
current-feature producers, 96.1% differ on a current visible feature. Exact current 7-feature rows
(seen ≥5×) receive teacher labels with median range 10 / mean 14.3 units, 66.9% > 1 unit, worst 90
units. Loss flat at ~12.9 is explained for the current row. This is NOT final proof that all
deployable observations fail: adding logged product/quantity/context reduces fixed-feature
conflict. Continuation: EXP-0038A missing-observation audit. Driver:
`tokenomics.eval.value_router_errors` (`decisions.csv`, `producers.csv`, `top_mistakes.csv`,
`alias_groups.csv`, `summary.csv`).

EXP-0038A (`tokenomics.eval.observation_ladder`, cumulative deployable-observation ladder over the
same dump) then hit the alias-collapse trap head-on: adding +order context collapsed the exact-alias
median label range 10→0 (frac>1 0.669→0.224) BUT the held-out routing probes (ridge+interactions and
kNN, leave-one-seed-out, scored real-choice-only / decision-centered / decision-weighted / tie-aware)
never beat the deployed current_row-only actor floor — **0/3 for BOTH probes on every task** (+order
kNN closest at 4.78/6.15/4.68 vs actor 3.93/4.75/4.12); probes cleared the random floor but centered
within-decision r2 ≈ 0. Order/queue fields are decision-constant → level only, no within-decision
contrast. 96.7% of top actor mistakes differed on the current row (distinguishable producers ranked
wrong, not a missing per-producer feature). Verdict: the dump's deployable blocks do not license
enrichment; only richer per-producer/product observations remain live. Continuation:
EXP-0038B opens a producer/product ladder: dedicated target-product throughput, multi-capacity
dispersion, capacity opportunity cost, dependency cover/fulfilment speed, base/cutoff margins, and
last-5-cycle producer history first; system-wide demand/capacity/recipe context second and only as
interaction context.

EXP-0038B Phase 1 (fresh CPU-pod re-dump, 60 deployable `pp_*` columns over the same 13,940
decisions, scored with the identical corrected gate) **fails the per-producer/product ladder but does
not retire the target.** Exact-alias conflict falls on repeated rows: median label range at fixed
features 10.0→0.0, frac>1 0.669→0.034 (+pp_throughput)→0.012 (+pp_base). This is secondary evidence
only because repeated-row coverage collapses as continuous `pp_*`/`h_*` columns are added (75.1% at
current_row, 26.2% at +pp_throughput, 8.7% at +pp_base, 4.6% at +history). The licensing metric still
fails on **every** tested per-producer/product block: 0/3 beats-actor for BOTH probes, centered
within-decision r² ≈ 0 (ridge best 0.0027, mostly negative; kNN always negative), and adding pp_
blocks raises regret (ridge 5.31→5.66, kNN 5.20→5.52). Deployed actor floor regret s45/s77/s29 =
3.93/4.75/4.12; no probe on any Phase-1 block approaches it. **Verdict: do not train an enriched
actor from Phase 1; keep `blend_0.5` as the orderer floor while continuing to Phase 2.** Phase 2 must
test system-wide demand/capacity/recipe/other-actor context as `sys_*` interaction features because
decision-constant context cannot re-rank producers additively. Retiring the marginal-value teacher is
licensed only if those system-wide interaction blocks also fail the same held-out regret gate.
Driver: `tokenomics.eval.observation_ladder` over the `--feature-set producer_product_v1` dump.

EXP-0038B Phase 2 (fresh CPU-pod re-dump `5gi98ku7gfnf2e`, 32 deployable `sys_*` columns added over
the same 13,940 decisions, scored with the identical gate) **also fails — so the target is now
retired.** The `sys_*` blocks (S1 demand / S2 capacity / S3 recipe / S4 cutoff-urgency / S5
other-actor) are mostly decision-constant, so `observation_ladder._split_cols` scores them as ridge
interaction *context* for the producer-varying columns (kNN reads them as distance dims — the additive
cross-check). Every `sys_*` block is **0/3 beats-actor for BOTH probes**, centered within-decision r²
≤ 0 and *worsening* as blocks accumulate (ridge −0.02→−0.18→−0.22→−0.21→−0.25; kNN ≈ −0.05); ridge
regret 5.6–5.9, kNN 5.5–5.7 — at/above the random floor (5.24/6.24/5.38), never near the actor floor
(3.93/4.75/4.12). Alias collapse is not a pass (repeated-row coverage falls to 543 rows / 93 groups at
+sys_demand). **Verdict: neither deployable class (per-producer/product NOR system-wide interaction)
makes the marginal-value teacher label a within-decision function of the observation — it lives in the
8-cycle rollout. Retiring the marginal-value teacher as an actor training target is LICENSED
(EXP-0038B planned read). A learned orderer must emulate the rollout (a compute lever) or optimise a
different, obs-measurable objective; `blend_0.5` stays the orderer floor.** Driver:
`tokenomics.eval.observation_ladder` over the `--feature-set producer_product_system_v1` dump.

---

## 17. 120-producer joint-stack episodes cost two orders of magnitude more than 12-producer ones

**Problem (plain English):** under the joint-stack contract (entity producer observation +
product-stream orderers, `eval/joint_stack.joint_stack_env`), the cost of one episode
cycle grew about 150× (sparse-wide) to 175× (dense) from 12 to 120 producers. Ten times
the producers should not cost that much. The legacy contract on the same worlds grew only
about 30×. Projected to the M1 gate runs, this made EXP-0044/0045 cost about 1,600 + 900
vCPU-hours (task 8.3, `openspec/changes/add-measurement-gate-harness/sizing.md`). The
competing explanations call for different responses:
- the simulation itself (auction, production) scales badly. That is a domain-model problem
  and would cost the legacy contract too;
- a per-agent observation cost. Fix the observation builder;
- policy cost or harness overhead (PettingZoo bookkeeping, baselines). Leave the domain
  alone.

**Discriminating check (local, about a minute):** cProfile one joint-stack episode at a
short horizon (h ≤ 4) on a dense-120 and a sparse-120 family world. BLAS pinned to 1
thread, idle box (entry 16). Sort by cumulative time. The registry holds no 120-producer
worlds since task 8.7 (gate sizes 12 and 42), so the check builds the world directly.
```
python -c "import cProfile, pstats; \
from tokenomics.eval.families import build_family_config; \
from tokenomics.eval.joint_stack import joint_stack_env; \
from tokenomics.eval.runner import run_episode; \
from tokenomics.eval.baselines import PRODUCING_BASELINES as P, ORDERING_BASELINES as O; \
env = joint_stack_env(build_family_config(1000, family='dense', producers=120, regime='calm'), 3); \
cProfile.run('run_episode(env, P[\"entity_planner\"], O[\"proportional\"], 0)', '/tmp/p'); \
pstats.Stats('/tmp/p').sort_stats('cumulative').print_stats(25)"
```

**Outcome branches:**
- **(a) `env/aec_env.observe` → `domain/orderer_view.build_product_stream_view` dominates.**
  This is the cause found here. Every product-stream orderer (≈300 sparse, ≈500 dense at 120)
  rebuilt every producer's rows (`_producer_stats`, `_producer_product_stats`). Each
  producer total scanned every telemetry EMA key (`telemetry.sum_by_producer`), so the cost
  was streams × producers × keys per cycle. Sparse-120 at h = 3 before the fix: observe
  96%, stream view 86%, 1,004,640 `sum_by_producer` calls.
  Fixed in task 8.4, with byte-identical observations, actions and episode statistics:
  - per-producer EMA totals indexed once per fold;
  - producer rows memoized while no producer row can change (`StreamViewCache`);
  - one shared row for undeclared, no-history (producer, product) pairs;
  - env producer infos built only for the encoders that read them.
  The guard is `tests/test_orderer_view_perf_equivalence.py`.
  If this branch shows up again, look for a new per-stream × per-producer computation
  added to the view.
- **(b) `System.complete_cycle` (auction / production) dominates.** The domain model scales
  badly. Not this case: env step was 2% of the episode before the fix.
- **(c) Policy calls or PettingZoo bookkeeping (`_accumulate_rewards`, `_clear_rewards`)
  dominate.** Harness overhead. After the 8.4 fix these are the remaining tail (about 20%
  of a dense-120 episode), next to the observation encode (about 30%).

**Seen in.** Task 8.3 sizing (`sizing.md` §5) and task 8.4, before → after, single process
on an idle box. Episode time at h = 12, `entity_planner/proportional`:

| world | before | after |
|---|---|---|
| sparse-120 | 28.8 s | 3.4 s |
| dense-120 | 53.3 s | 5.2–5.4 s |

The portable subject at h = 8 went from 40.0 s to 4.6 s (sparse) and from 47.4 s to 7.0 s
(dense).

**Seen in EXP-0055 (PoC-B, after the fix).** The portable stack, timed per stage on calm
`sparse_wide` worlds up to 42 producers (the target; nothing larger is timed). The reading that
bears on branch (a): at fixed fanout E = 2 the cost of one orderer decision grows with slope 0.34
on producers (`obs` 0.52, ≈ √P), not the ≈ 1 of every stream rebuilding every producer's rows, so
the blow-up has not come back on the deployed path. Per cycle the slope is 1.24 (10.3 → 49.3 pod
ms from 12 to 42); the ≈ 2.2 above (12 → 120, `entity_planner/proportional`, local) is context only,
not like-for-like (different subject, range and host). No per-decision stage is superlinear on
fanout or consumers (max 0.54, the producer decision's `forward` on fanout; the orderer decision is
flat in fanout, 0.03). `obs` + `forward` are ≈ 80% of the wall, `complete_cycle` 1.5–3.5% (not
(b)). The remaining growth sits in branch (a)'s place, the orderer's observation. Read from stage
timers, not a profile. If economies above
42 return, profile the stream view first (eligible-only rows, the D2 refactor).

---

## 18. A family fails (or passes) the headroom check, and capacity cuts do not move it

**Problem (plain English):** the gate's headroom check (design D10) compares a world family's
median best-baseline fill rate with the 0.95 ceiling. When a family is too close to the ceiling,
the obvious retune is to give it less capacity. In EXP-0044 that did nothing: dense-12 stayed at
about 0.955 from 1.0× down to 0.8× capacity. The competing explanations call for different levers:
- capacity binds, but the cut was too small: probe deeper cuts;
- the shortfall is not capacity at all but **warm-up / edge loss**: orders placed while the
  supply chain fills, or still in flight at the horizon, expire (TTL 2). Change what the check
  measures, or the demand pressure, not the capacity.

**Discriminating check (pod, cheap):** replay the family's strongest baseline on every
development gate world and split the undelivered units into cycles 1–20 and 21–100. It spans
many worlds at the full horizon and gates a decision, so it runs on a pod (CLAUDE.md compute
rule), not locally. Do it as **one h = 100 run per world with a snapshot of ordered and fulfilled
units at cycle 20**. Do NOT subtract an h = 20 run from an h = 100 run: on shocked worlds each
randomized consumer's horizon-length schedule comes from one shared random stream, so later
consumers get different demand at the two horizons (post-hoc review of EXP-0044). The EXP-0044
scripts used that subtraction and are valid on calm worlds only. They are kept as a record,
not as the recipe: `experiments/EXP-0044-gate-calibration-12-42/retune/warmup_share.py <size>`
and `warmup_share_family.py <family> <size> <member>`, run from `tokenomics/` as
`../.venv/bin/python ../experiments/EXP-0044-gate-calibration-12-42/retune/<script>`.

**Outcome branches:**
- **(a) Most of the shortfall is in cycles 1–20, and the fill rate after cycle 20 is near 1.**
  Warm-up dominates. This is the case found on EXP-0044's calm dense worlds: dense-12 had 54%
  of its shortfall in cycles 1–20 and a 0.975 median fill after; dense-42 had 85% and 0.995.
  Capacity is not binding (demand ≈ 6% of top-layer throughput), so a mild capacity cut cannot
  help. Shocked worlds were not diagnosed (unmatched subtraction).
- **(b) The shortfall is spread over the whole episode and the late fill rate drops as capacity
  is cut.** Capacity binds; a capacity scale is the right lever. Find the size of the cut with
  `gate --mode probe`.

**Seen in.** EXP-0044 retune probe (task 10.3), `experiments/EXP-0044-gate-calibration-12-42/retune/`.

---

## 19. A random baseline beats every hand-written rule

**Problem (plain English):** a baseline that routes orders at random sets the hindsight baseline
on most episodes. It beats rules that look at capacity or prices. That usually means something
is wrong with the rules or with the measurement, not that randomness is good. The competing
explanations call for different fixes:
- (a) **packing**: the rules pile sub-orders on the same producers, and atomic auctions reject
  them. Fix the rule's load awareness;
- (b) **temporal variation**: what helps is changing the receiving producer over time, which a
  deterministic rotation can do. Fix the rule, and no stochastic policy is needed;
- (c) **rule defects**: lowest-index tie-break, rounding of small orders, routes to producers
  that cannot make a unit, slot overflow. Fix that defect;
- (d) **investment-fund channel**: random routing of producers' own purchases changes who gets
  investment;
- (e) **scoring artifact**: the random baseline is a mean of replicates and the rules are single
  runs;
- (f) **split penalty**: the rules split small orders, and a split order completes only if every
  part does.

**Discriminating checks (pod for the full grid; local for the trace):**
1. Full grid with diagnostic routers that change one thing each (EXP-0048 driver):
   `python -m tokenomics.eval.routing_audit --workers 16 --source-commit {SOURCE_COMMIT}
   --out-dir EXP-NNNN-title` on a cpu16 pod (≈ 4 vCPU-h on registry v2). Read `edge.csv`. A
   router *recovers the edge* in a size × regime cell when its equal-weight world-mean gain over
   the best deterministic roster member is at least 0.8 of random's. It recovers it in *most
   cells* at 2 of the 3 random-dominated cells.
2. (a) `proportional_load_aware` against `proportional`; rejection reasons
   (`rejected_bid_units_*`, `lost_units_*`) in `routing_metrics.csv`.
3. (b) `round_robin` and `random_whole` against `random_fixed_episode` (decorrelated and
   mostly whole, but fixed over time: the cleaner time-variation contrast); `round_robin`
   against `proportional_whole` (both whole, but `proportional_whole` also herds every orderer
   onto the argmax producer: check `herfindahl_mean`); the `random_scaled_s` sweep read with its
   split degree. A balanced fixed assignment is not in the router set, so alternation over time
   and the long-run balance of a fixed assignment are not separated.
4. (c) local single-seed trace,
   `openspec/changes/add-m3-development-audit/probes/trace_21.py --world <w> --router <r>`
   (tie decisions, rounding, `zero_throughput_route_share`, `overflow_orders`), plus
   `even_split_rotating` against `even_split`.
5. (d) `random_consumer_only` / `random_producer_only` against `blend_0.5`;
   `fund_share_herfindahl_episode` against 1/N.
6. (e) the `edge_r1` and `random_worst_beats_best_det` columns of `edge.csv` (R = 1 against
   R = 5). (f) the `split_2x2` rows of `edge.csv` (whole/split × deterministic/random at matched
   split degree) and `wasted_units_*`.

**Outcome branches:**
- **(b) confirmed** (`round_robin` recovers; fixed assignments, `random_fixed_episode` and
  `proportional_whole`, do not): variation of the receiver over time is the lever. A deterministic rotating rule is a roster v3 candidate (design D3, a user decision),
  and stochastic deployment is not needed for the edge.
- **(c) confirmed** (a fixed variant recovers): the rule is defective. Fix it under a new name
  through roster v3 (D3); never edit the frozen rule in place.
- **(a)**, **(d)** confirmed: change the rule's load awareness, or read the producer-purchase
  routing together with role attribution (PoC-C).
- **(e)** confirmed (R = 1 removes the edge): no rule change; document the averaging in the gate.
- **(f)** confirmed (at the same kind of variation, routing whole instead of split recovers
  most of the edge; `split_2x2` rows and the matched contrasts in the EXP's RESULT): a
  whole-order rule is a roster v3 candidate, and a learned residual must be able to route an
  order whole. (f) can combine with (b): read the contrasts with one factor held fixed, not only
  the single-factor pattern "whole recovers, split does not".

**Seen in EXP-0048** (`experiments/EXP-0048-random-routing-diagnosis/`): (b), (c) and (f)
confirmed, and they combine. `round_robin` (rotating, whole) recovered 4.72 / 2.51 / 1.62 of the
edge (12-shocked, 42-calm, 42-shocked; +52 to +125 units), and beat `random_fixed_episode` by
89–241 units; `proportional_whole` (whole, not rotating, herding) recovered 0.46 / −3.34 /
−0.73. `even_split_rotating` recovered 0.69 / 0.81 / 0.84, against `even_split`'s −3.74 /
−4.65 / −1.07; a random leftover gains most of the same units. Random variation helps; rotation
helps somewhat more. Routing whole instead of split at the same variation is worth 0.78–4.04 of the
edge. (a), (d) and (e) were ruled out.

**Seen in EXP-0049** (`experiments/EXP-0049-informed-routing-and-trained-trace/`, follow-up to
(b)): on registry v2 (every product has two eligible producers), three rules that read the
producers' observed spare capacity do not improve on blind rotation; fanout ≥ 3 is untested.
Check:
`routing_audit --plan exp0049` (adds `two_choice_spare`, `spare_weighted_whole`,
`spare_rotating_whole`, and the trained subjects, sha256-pinned checkpoints); compare each with
its blind counterpart (`random_whole`, `round_robin`) on units, `receiver_change_share_*` and
`herding_hhi`, and count how often spare binds
(`openspec/changes/add-m3-development-audit/probes/spare_33.py`). Outcome: spare rarely binds
(skip 0–2.7% of decisions; `lost_units_consumer_no_fit` ≈ 0 for every router), and following a
shared capacity signal keeps each orderer on the same producer over time (receiver change
0.17–0.30 for the argmax rule against 0.50 for blind random), losing units as TTL expiries.
Read `receiver_change_share_*`, not `herding_hhi`: receiver change tracks units in every cell,
per-cycle herding does not (`round_robin` herds at 0.89–0.90 on 12-producer worlds and wins).
The trained portable orderer varies receivers like `round_robin` (0.64–0.67) without sampling,
but splits 29–40% of consumer orders; it beats `random` and loses to `round_robin` on shocked
worlds. Caveat: at
fanout 2, "pick two, take the roomier" is an argmax; testing it needs worlds with fanout ≥ 3.

**Seen in EXP-0050** (`experiments/EXP-0050-routing-vs-fanout/`, the same routers on worlds with
2–14 possible makers per product; registry v2 has 2): the diagnosis at fanout 2 understates how
much routing matters. Check: `routing_audit --plan exp0050` and its `fanout_summary.csv` /
`routing_metrics.csv` (split degree, `wasted_units_consumer`, `lost_units_consumer_ttl`,
`lost_units_consumer_no_fit` per cell). Outcome: at 42 producers `round_robin` beats the best
hand-written rule by +116 / +150 units at fanout 2 and +335 / +421 at fanout 7; routers that
split lose more with more makers (losses are TTL expiries, no-fit ≈ 0): `random` splits more at
fanout 14 and falls 268–296 units below `round_robin`, and `random_whole` recovers it
(`even_split`'s fanout-7 collapse comes mostly from its lowest-index tie-break). The spare-capacity routers still lose to blind choice. The trained portable orderer splits
as much as the failing rules but its split orders complete, and it clearly leads at 42
producers with fanout 4 and 14.
Open: why its parts complete. Its distillation target is input-aware (log of
`makeable_qty_now`, capacity and inputs in stock; `proportional` weighs capacity only). Check:
the teacher as a router (split by `makeable_qty_now`), its whole-order version, and a trace of
expired reservations by producer input stock.

**Seen in EXP-0052** (`experiments/EXP-0052-headroom-atlas-confirm/`, the deciding check above,
pre-registered in EXP-0050 and design D12, read on the atlas confirm's 14 hard worlds with at least
4 makers per product: 4 at 12 producers, 10 at 42; `deciding_check.csv`). Outcome: **no branch
reaches 2/3.** `makeable_split` within the floor of the trained orderer (input-aware weighting is
the lever): 4 of 14. `makeable_weighted_whole` above `makeable_split` beyond the floor (splitting
still costs): 8 of 14, the nearest, and 7 of 10 at 42 producers (by up to 8,700 units; the split
by size was not pre-registered). `makeable_split` below the trained orderer (the scorer learned
more than its target): 2 of 14; at 42 producers it is the other way round on 7 of 10 (the teacher
target as a router beats the trained orderer by 409–3,286 units). Floors are 330–580 units, so the
"within" calls are low-powered (without 12-0435, a dead-bar artifact, entry 20: 3 / 8 / 2 of
13). The tracer split answers the open question in part. On consumer orders, the share of units
lost to expired reservations whose receiver's inputs could not cover the order is 0.88–0.91 for
`round_robin` / `random_whole` / `random`, 0.87–0.89 for the other roster rules, and 0.705
(`makeable_split`), 0.73 (trained orderer, `portable`), 0.77 (`makeable_weighted_whole`);
on investment orders 0.37 / 0.50–0.52 / 0.82 against 0.87–0.91 for the blind bar rules
(`deciding_check.csv` sums both sides). Input-aware routing, learned or hand-written, removes a
clear but partial share of the input-shortage cause of consumer expiries (≈ 16–18 points),
consistent with the trained orderer's split parts completing more often (not a causal test). What
it does not settle is whether splitting itself costs; the 42-producer evidence leans that way.

**Seen in EXP-0054** (`experiments/EXP-0054-role-attribution-mined/`, PoC-C; resolved by
EXP-0056 below: `joint`). On the 3 shocked, burst-demand 12-producer world-episodes of PoC-A's sample
(12-0060, 12-0111, 12-0344), the trained orderer with `entity_planner` trails whole-order
`round_robin` by 274 / 891 / 589 units (EXP-0053's rows; each above the one-episode floor
3σ_ep = 164.3), 72% of the portable stack's gap to the teacher stack. EXP-0053's trace columns
(trained orderer / `round_robin`) fit two branches and do not separate them: (f) split share of
consumer orders 0.36 / 0.23 / 0.26 against 0, `wasted_units_consumer` 190 / 428 / 345 (sum 963
of the 1,754); (b) `receiver_change_share_consumer` 0.83 / 0.46 / 0.55 against 0.94 / 0.71 /
0.68, `rejected_bid_units_consumer_displaced` 493 / 3,489 / 2,886 against 68 / 1,346 / 1,749.
Check (driver built in task 11.2, run as EXP-0056; design D13 pre-registers it):
`python -m tokenomics.eval.teacher_audit --poc split --reference <EXP-0053> --reference
<EXP-0054>`, a 2 × 2 over the same scorer on these 3 world-episodes (`entity_planner`, routing
tracer on): TR the trained orderer as is, TW its scores with each order whole to the argmax, RS
its split profile on `round_robin`'s receivers (largest part to `round_robin`'s pick), RR
`round_robin`; `random_whole` as reference. The forced-whole arm alone does not decide: it also
pins each stream to its argmax (lead receiver change 0 in the 11.2 dry run), and the tracer's
`receiver_change_share` compares receiver sets, which a split order keeps. The receiver rule has
two sides, read from each order's lead receiver (largest part): rotation over time,
`lead_receiver_change_share_*`, and spread across concurrent orders of a product,
`lead_concurrent_top_share_*` (the scorer tiles one score row over its slots, so its argmax
sends them all to one producer: 1.0 for TR and TW against 0.55 for `round_robin` in the dry
run). RS changes both sides; the columns say which one differs. Branches (`split_check.csv`; a
change recovers the gap RR − TR on a world when its gain from TR clears 164.3 and is ≥ 2/3 of
the gap; ≥ 2 of 3 worlds each): TW recovers it → (f), the whole-routing rule recovers the gap
(splitting under the scorer's rule is the cost at the rule level; TW also changes realized
rotation, so not shown to be the mechanism), and M4's orderer must be able to route whole (with
RS too → f+b); else RS does → (b), the receiver rule is the
lever (concurrent spread and / or rotation over time), M4's orderer must spread / rotate
receivers; else neither single change recovers it and only both together do → `joint`, M4's
orderer needs whole routing and spread receivers together (the interaction tells complements
from additive halves); else the worlds disagree → `neither`, diagnose world by world; the gap
does not clear the floor on 2 worlds → `no_gap`, EXP-0053's finding did not reproduce. The
factors are rules held by construction (TW keeps the scorer's receiver rule, RS its split rule);
realized rotation and splitting follow each arm's trajectory and are read from
`manipulation.csv` as how a change worked, never as a gate. Limits: a cause outside the two rule
changes is not detectable here; `joint` is the remainder (not proof of complements); the
receiver factor also covers which producers are picked, so (b) / `joint` do not pin spread or
rotation (design D13).

**Seen in EXP-0056** (`experiments/EXP-0056-split-vs-rotation/`, the check above; resolves the
EXP-0054 note). Outcome: **`joint` on 3 of 3 worlds** — (f) and (b) combine, neither alone is
the cause. Gap RR − TR 274 / 891 / 589 (reproduced exactly from EXP-0053). Whole routing at the
scorer's receivers, TW − TR, recovers −0.48 / 0.61 / 0.34 of it; the receiver rule at the
scorer's split, RS − TR, −0.02 / 0.23 / 0.09; neither ≥ 2/3 on any world. At the other level
each change is large: RR − RS 280 / 689 / 537, RR − TW 405 / 347 / 387 (all above the floor);
interaction +411 / +145 / +335 (above the floor on 0060, 0344), the complement pattern; but on
0060 TW also fails on producers' own purchases (investment rejection 0.56, mean capacity 194.7
against 227.9–242.3, the (d) channel), which RR − TW and the interaction absorb, so the clean
above-floor consumer-side evidence is 0344 alone.
`manipulation.csv`: RS keeps TR's split share (except 0060, 0.57 vs 0.36) and takes RR's
concurrent top share (0.54–0.75 vs TR 0.98–0.99) and lead change; TW keeps TR's concurrency
(0.99). Traces (descriptive): the scorer's single pick per product brings displaced bids and
expiries with inputs in stock, which its splits partly relieve; under rotation the splits
waste 260 / 310 / 508 units. Consequence (these 3 shocked burst 12-producer worlds, one
episode each): the cheapest rule changes that close the gap are whole orders plus
`round_robin`-like receivers together, so a learned orderer needs both abilities, route whole
and vary receivers. Which receiver property matters (spread across concurrent orders, rotation
over time, or *which* producers are picked, D13 limit (ii)) is open; both factors also change
how producers' own purchases are routed. The pre-registered `joint` branch text above ("whole routing and
spread receivers together") is narrowed by this result to "whole routing and varied receivers
(property open)". No new cause
registered: this is (f) + (b) combined, the case the (f) branch above already names.

**Seen across the atlas (EXP-0051 addendum, 2026-10-30)**: offline over 639 screen worlds, the trained orderer beats `round_robin` on only 13% (median −21%); its relative result falls with load (t −7.4) and, against the prediction, with makers per product, more so under load (log E × log ρ t −4.2). EXP-0050's wins sit in the very-light-load corner (ρ ≈ 0.06), where it ties. Consistent with the concentration mechanism of EXP-0056; association, not proof.

**Seen in EXP-0057** (`experiments/EXP-0057-decoder-check/`, M4 step 1, design D3 of
`add-m4-orderer-redesign`; follows the `joint` reading above). Question: is the trained orderer's
loss to whole-order `round_robin` in its fixed decode (softmax split, one score row tiled over
every order) rather than in its scores? Check: `python -m tokenomics.eval.decoder_check` (CLAUDE.md
"Decoder check") on the 23 loaded atlas v2 train worlds at 12 producers (14 at 42 reported):
TS = the same EXP-0012 scores, one producer drawn per order from softmax(score), the order
whole, against TR (as deployed), TW (argmax whole), MWWk (the scorer's target
`makeable_qty_now` as a keyed whole sampler with TS's draws), MWW and RR; read
`decoder_check.csv` (G_w = RR − TR, its floor, rec_w = (TS − TR) / G_w) and `fidelity.csv`.
Outcome: **`decoder_partial`**, the decode is most of the cause. 17 of 23 worlds have a gap
above the floor; TS recovers ≥ 2/3 of it on 11 (12 needed for `decoder`), median 0.70, pooled
0.70. The rest: ≈ 0.19 of the gap is the scores short of their own target (MWWk recovers 0.89;
TS within the floor of MWWk on 17 / 23, below on 6, above on none) and ≈ 0.11 is `round_robin`
over the target rule. Going whole by argmax (TW) recovers only 0.29 and is the only arm that
dies (4 episodes), so spreading concurrent orders across producers by sampling, not whole
routing alone, carries the recovery (concurrent top share TR 0.98, TW 0.99, TS 0.84, MWWk 0.73,
RR 0.63; descriptive). This narrows the `joint` reading for M4: "vary receivers" is delivered by
per-order sampling from the scores, and a choice decode recovers most of the gap without
retraining. At 42 producers the scores themselves fall short (`fidelity_loss`, 8 of 14 worlds),
reported only. No new cause registered: `decoder_partial` is a licensed reading, not a failure.

## 20. A world set has no headroom

**Problem (plain English):** on some worlds no better policy can be reached: every cheap policy
already delivers about as much as the world allows, or nothing beats the best simple rule by
more than the run-to-run noise. Training or mining on such worlds cannot show improvement,
however good the learner. Registry v2 looks like this (EXP-0047, EXP-0050: fills of 0.97–0.99 on
the 12-producer sparse f4 and dense cells). The question is where in the world space headroom
exists, and of which kind.

**Discriminating check:** the headroom atlas (`python -m tokenomics.eval.atlas`, design D12,
ADR-0057). `--mode screen` plays a world-space sample with the blind and hand-written arms (the
*atlas bar*: the best of them) and the reach arms (`makeable_split`, `makeable_weighted_whole`,
the trained orderer); per world-episode it computes the restored bounds B2 (a demand cap) and
B_base (the base-capacity ceiling). `--mode confirm` replays the selected worlds on fresh seeds.
Per world: gain = best reach − bar from paired episodes; ceiling C = max(min(B2, B_base), best
played arm); room = (C − bar) / C; noise floor = max(3 × SE from spreads pooled per size ×
choice, 0.01 × bar) (2 × SE in the screen, from EXP-0050's spreads). Read `worlds.csv` classes
per size × choice × pattern (and `coverage.csv` for where the sample is thin):
```
python -m tokenomics.eval.atlas --mode screen --exp0050-episodes <EXP-0050 episodes.csv> \
  --prod-ckpt P.pt --orderer-ckpt O.pt --workers 16 --source-commit {SOURCE_COMMIT} --out-dir EXP-NNNN-title
python -m tokenomics.eval.atlas --mode confirm --screen-dir <screen EXP dir> \
  --prod-ckpt P.pt --orderer-ckpt O.pt --workers 16 --source-commit {SOURCE_COMMIT} --out-dir EXP-NNNN-title
```
A row above B2 is a bug (run invalid, naming world, episode and arm), never headroom.

**Outcome branches:**
- **saturated** (bar fills ≥ 0.95 of consumer orders, or reaches 0.97 of C): nothing to gain;
  filter these worlds out of mining and of any acceptance set that should measure improvement;
- **flat** (no reach arm beats the bar beyond the floor, and room < 0.10): no lever was found
  cheaply; leave the region unless a new reach arm or a cause says otherwise;
- **open** (room ≥ 0.10, not hard): the ceiling leaves room no cheap arm reached; send these to
  the teacher sample (PoC-A). B_base is not guaranteed, so check the count of arms above it;
- **hard** (gain above the floor and positive in every episode and replicate): a better policy
  is reachable and measured; mine and train here;
- **collapse** has two meanings (task 5.3), told apart by `worlds.csv` `argmax_died`:
  (i) every arm dies before cycle 50 in every episode (`argmax_died` may be false): a
  world-design problem, not headroom; (ii) confirm only: the fixed bar or reach arm dies in some
  episode or replicate while other arms may deliver normally (`argmax_died` true): the
  headroom reading against the cheap reference arms is unreliable on that world, which is
  excluded from hard / open (the teacher sample) on purpose, not because the economy cannot
  run. Read (ii) worlds as fragile, not empty.

**Seen in:** EXP-0051 (screen, provisional classes, run commit 92f3177): the branch depends on the region. Saturated only in the low-tightness, small-order corner where registry v2 and the EXP-0050 family worlds sit (10 of 512 worlds at 12 producers by fill, all ρ ≤ 0.23; the 12-f4 and three 42-producer anchors) plus 11 over-demanded worlds by the ceiling; elsewhere labelled hard (213 at 12, 68 at 42; mostly by `makeable_weighted_whole`, at lower ρ and, at 42, wide choice) or open (274 / 58; high ρ, bar fill ≈ 0.4). The hard labels are an upper bound: the screen's floor borrowed EXP-0050's spreads, 18–35× too small on these worlds (entry 21); with the screen's own replicate spread 63–71 / 34 stay hard. EXP-0052 (confirm, task 5.2, run commit 9d44220): on fresh seeds with its own pooled spreads (3 × SE) the selected worlds read **hard** 4 of 64 at 12 producers and 10 of 32 at 42, every other one **open**; none saturated, flat or collapsed. Hard holds at 42 producers with wide choice (9 of 14 f14 worlds, 1 dense; none at f2/f4/f7; median +16% of the bar, won by input-aware routing); at 12 only two exact cyclic worlds and one marginal world (12-0435 is an artifact: its fixed bar arm dies at cycle 4 while other blind arms survive). The count is D12's pre-registered one but depends on the estimator: across eight spread estimates (entry 21) it is 2–12 at 12 and 5–17 at 42. Open worlds: bar fill ≈ 0.46, room ≈ 0.5 under B_lp ≈ B_lp_base ≈ demand, a ceiling that ignores input lead time, the fund and inventory (EXP-0051 addendum), so their reachable room is unproven. Frozen as atlas v1 (`atlas_v1.json`). **Rule gap seen in EXP-0052** (12-0435): confirm fixes the bar arm from the screen; when that arm dies on fresh seeds (here `random_whole` at cycle 4 in all 4 runs, bar 0) while other blind arms survive (470–790 units), the world reads hard (gain +874, room 1.00) although against the best blind arm on the same runs (+218) it is not. Collapse needs every arm dead and nothing checks a dead fixed bar arm. Check: compare the fixed bar arm's units and deaths with the best blind arm per world (`exp0052_check.py` sensitivity "free bar"); branch: if the fixed arm died, report the world as unresolved rather than hard (a code task, not done here). The world stays hard in `atlas_v1.json`; PoC-A's 12-producer train sample has only it and 12-0348 as hard worlds. **Resolved by task 5.3:** confirm's dead-arm rule classes a world collapse when its fixed bar or reach arm dies in any episode or replicate (`worlds.csv` `argmax_died`); applied offline to EXP-0052 with the robust floor of entry 21 (atlas v2, `EXP-0052/reclassified/`), it makes 6 worlds collapse (12-0435, 0140, 0312, 0423; 42-0045, 0051) and the counts read hard 11 / open 49 / collapse 4 at 12 producers and hard 13 / open 17 / collapse 2 at 42; PoC-A's 12-producer train sample gets 4 hard worlds (0060, 0176, 0005, 0348) plus 8 open.

## 21. A noise floor borrowed from another world population

**Problem (plain English):** a run that cannot measure its own chance variation (one episode per
world, a stochastic policy played once) borrows a spread from an earlier run. If the earlier run
played a different kind of world, the borrowed spread can be far too small (or large), and the
"beats the bar by more than noise" call then counts noise as a gain. Seen when the atlas screen
took EXP-0050's spreads, measured on easy-corner worlds that fill 97%+ of orders, for worlds
that fill 40–60%.

**Discriminating check:** find any replicated arm the run itself played and compare its pooled
within-episode SD with the borrowed spread, per stratum. In the atlas screen `random_whole` runs
R = 2 on every world: `python3 openspec/changes/add-m3-development-audit/probes/exp0051_check.py
<EXP dir> --exp0050 <EXP-0050 episodes.csv>` prints "screen's own sigma_rep" against the
provisional σ_rep, and the hard count recomputed with it (absolute and relative × bar, at 2 and
3 × SE).

**Outcome branches:**
- **own spread ≈ borrowed** (within about 2×): the borrowed floor holds; read the classes;
- **own spread ≫ borrowed**: the borrowed-floor classes are an upper bound (hard) / lower bound
  (open, flat); report the recount with the run's own spread, keep the run (the selection may
  still be useful), and let a stage with its own replicated spread decide (atlas confirm pools
  σ_ep and σ_rep from its own episodes);
- **no replicated arm in the run**: the floor is unverifiable; say so and do not quote hard
  counts as measured.

**Seen in:** EXP-0051 (screen): σ_rep 46–77 units at 12 producers and 156–343 at 42 against the
borrowed 2.6 and 10–21 (18–35×); hard 213 → 63–71 at 12, 68 → 34 at 42 at 2 × SE. σ_ep (episode
to episode) is not measurable in the screen.
EXP-0052 (confirm, the "own spread decides" branch): pooled from its own replicated episodes, σ_ep
= 164 units at 12 producers and 156 at 42 (EXP-0050: 6–14 and 6–37), σ_rep = 90–131 and 315 (EXP-0050:
2.6 and 10–21); median floor 397 / 577 against the screen's 14 / 76; 4 of 48 and 10 of 24 selected
hard worlds stay hard (D12's pooled estimator). A second form of the same problem showed inside the
confirm: both of its own pooled spreads are concentrated in a few worlds. 5 worlds give 89% / 72%
of the σ_ep sum of squares (gains that swing by hundreds of units between episodes; median
episode-to-episode change 52 / 134 units) and 69% / 87% of the σ_rep sum of squares; at 42
producers the floor is mostly σ_rep (σ_rep²/2 ≈ 49,600 vs σ_ep² ≈ 24,200), and its largest term
is a death: one `random` bar replicate on 42-0045 dies at cycle 17–23 while the other delivers
≈ 1,900 units. Check: the "sensitivity" section of
`openspec/changes/add-m3-development-audit/probes/exp0052_check.py` (pools without the 5 largest
σ_ep / σ_rep worlds or both, without deaths, relative to the bar, median-based, bar re-chosen);
hard counts 2–12 at 12 and 5–17 at 42 at 3 × SE (12 / 17 with both outlier sets out of the
pools). A spread pooled in absolute units over worlds whose bars range from a few hundred to
16,400 units, with deaths in random replicates, is a lower-quality floor than per-class, relative
or robust pooling; later reads should pre-register one of those.
**Resolution (task 5.3):** the confirm now pools the normal-consistent median absolute
within-group residual (`atlas.mad_sigma`, design D12 "spread estimator"; 50% breakdown point, so a
minority of heavy-tailed worlds cannot set a stratum's floor), with the pooled SD reported beside
it (`sd_ep`, `sd_rep`); chosen on that property after these counts were known (stated in D12 and
EXP-0052's addendum). On EXP-0052: σ_ep 55 / 141, σ_rep 41–73 / 122, median floor 145 / 349 (was
397 / 577). Check for the residual risk (a world noisier than its stratum's median gets too low a
floor): `exp0052_check.py ... --reclassified <dir>` prints each hard world's own episode spread
against σ_ep; at 12 producers all ≤ 1.1×, at 42 three hard worlds exceed 2× (0066, 0076, 0098).

---

## 22. A rollout on a fork advances the live episode (traced env)

**Problem (plain English):** a lookahead search plays copies ("forks") of the running game
forward to score its choices. The routing tracer records each cycle by replacing the game's
end-of-cycle step with its own function, and that function remembers the *original* game. Python's
deep copy does not copy functions, so every fork keeps calling the original game's end-of-cycle
step: playing a fork forward finishes cycles of the live episode. Sometimes this crashes at once
(`WrongStateError` inside a rollout, seen in task 6.2's first teacher dry run); at other stop
points it silently skips an agent's action in the live episode and moves it to the next cycle
(the 6.2 reviewer's repro: a fork taken at the last agent's turn of cycle 2 advanced the live env
to cycle 3 before raising). The same applies to anything that deep-copies the system and runs it,
e.g. the ADR-0056 history recorder's stripped copies.

**Discriminating check (local):** on the env the search runs on,
`"complete_cycle" in vars(env.system)` (an instance hook is present), or after
`fork = copy.deepcopy(env)`, `fork.system.complete_cycle is env.system.complete_cycle`.
`bounded_rollout_orderer.assert_forkable` runs the first check at every search call and refuses
the env (test `test_teacher_audit::test_traced_env_refused`).

**Outcome branches:**
- **hook present on a searched / forked env**: the run's search results are invalid; rerun the
  search untraced (the teacher audit's `floor` / `d2` / `ub` episodes run without the tracer, so
  their `trace_*` cells are empty);
- **no hook**: forks are independent of the live episode; look elsewhere.

**Seen in:** task 6.2 (teacher audit dry run, `ws-atlas-v1-12-0060`). Not in any recorded result:
no deepcopy user (`env/future_fork`, `env/shock_posterior`, `eval/bounded_rollout_orderer`,
`eval/orderer_oracle`, `train/orderer_improve`; `eval/registry` copies configs,
`train/producer_scorer` a view) is on the routing_audit / atlas / gate job paths, which trace
but never fork; traced and untraced episodes give identical `episode_stats`
(dev-sparse_wide-12-shocked-00 / -calm-00, horizon 12), and a horizon-100 rerun of
ws-atlas-v1-12-0060 seed 1295251592 reproduces EXP-0052 (`round_robin` 2,429, trained 2,155,
`makeable_weighted_whole` 2,609 / 2,648; reviewer check). EXP-0048–0052 are unaffected.

---

## 23. A clairvoyant search scores below the same search on posterior forks

**Problem (plain English):** the teacher audit's `ub` arm runs the teacher's lookahead search
with the true future (the real pending shocks and demand) as its only scenario. It is meant as an
upper reference: with perfect information the same search should do at least as well as with
guessed futures (`d2`, ADR-0056 posterior forks). In EXP-0053 it did worse on all 5 worlds where
it ran (1–99 consumer units below `d2`). Read naively, that says foresight is worth nothing and
better forks cannot help. The competing explanations call for different responses:
- a fork defect: the clairvoyant fork does not actually carry the true future (wrong demand or
  shock stream), so `ub` is searching a wrong world. Fix the fork;
- a structural effect: the search scores each candidate by the consumer units it adds inside a
  short window (depth 2 cycles, then the floor), and a single exact scenario often shows exactly
  zero gain there for every candidate, so the search keeps the floor (`deterministic_zero`),
  while posterior forks show small one-sided gains for weakly better actions in a few futures and
  the confidence test accepts them; those actions pay off beyond the window. Then `ub` is not an
  upper bound for this search, the fork-limited reading is not determinable, and the fix (if
  one is wanted) is a different objective or reference, not the forks.

**Discriminating check (local, minutes):** from the run's `teacher_decisions.csv`, (1) per
world, the `deterministic_zero` share of searched streams for `ub` and `d2`; (2) the first
decision where the two arms differ (same cycle and agent), and its resolution in each; (3) a
one-decision repro at that point: a `ClairvoyantSource` fork played to the episode end must have
the recorded episode's demand and shock digests, and the `d2` per-scenario gains at that decision
show how the confidence test passed.

**Outcome branches:**
- **clairvoyant fork digests differ from the episode's**: fork defect; `ub` results are invalid,
  fix the scenario source and rerun `ub`;
- **digests match, `ub`'s `deterministic_zero` share is higher on every world and the first
  divergence is `ub` keeping the floor by `deterministic_zero` where `d2` overrides**: structural
  window / tie effect; report `ub` as not an upper bound, D5's fork-limited reading as
  not determinable, and make no claim about forks from it. A meaningful clairvoyant reference
  needs a longer window or a non-myopic score (a design change, not a fix);
- **digests match but the shares are equal and the divergence is elsewhere**: neither; look at
  the confidence rule's parameters (z, minimum expected gain) and candidate dedup.

**Seen in:** EXP-0053 (task 7.1, review concern #358). Digests match at 0060 cycle 2
(`41b0057e66b5` / `c961073d1524`); `deterministic_zero` share `ub` vs `d2`: 0060 70.0 / 66.9%,
0111 82.5 / 81.3, 0308 89.0 / 86.0, 0344 74.2 / 68.9, 0392 79.7 / 78.8; first override
divergences 0060 cycle 2, 0308 cycle 8, 0392 cycle 8, each `ub` `deterministic_zero` against a
`d2` override (0060: nonzero `d2` gains in 3 of 16 posterior futures, mean 0.56 units, none
negative). Outcome: structural; EXP-0053 records `ub` as no upper bound.

---

## Unverified problems (plain-English inbox)

Humans: describe a suspected problem here in plain English — no tooling knowledge
needed. The agent turns it into a check and promotes it to a numbered entry once it has
discriminated a real case.

_(empty)_
