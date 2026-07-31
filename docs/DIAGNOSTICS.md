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
- An entry is evidence-backed: it must cite at least one experiment where the check
  discriminated between competing causes.

Run profile: checks marked **pod** follow the CLAUDE.md compute rule (real runs on
Runpod); **local** checks are quick one-shot diagnostics.

---

## 1. Mined headroom not harvested (low closure)

**Problem.** Mining found a gap between the champion and a search bar, replay trained a
candidate on those tasks, but the candidate captures only a small fraction of the gap
("closure" = (candidate − champion) / (bar − champion)). Two causes demand opposite
fixes: a training/fidelity wall (fix the distill; more mining is wasted) vs a
generalization gap (candidate learned its train tasks but not the held-out ones; mine
denser).

**Check** (pod): `tokenomics.eval.train_closure` — evaluate the candidate on the tasks
it was TRAINED on, compare closure across train / val / blind splits.
```
python -m tokenomics.eval.train_closure --archive <recoverable_archive.jsonl> \
  --attribution <gap_attribution.csv> --producer-ckpt <cand.pt> --orderer-ckpt <ord.pt> \
  --out-dir <out> --split train --eval-seeds 3 --horizon 100 --workers 16
```

**Outcomes.**
- train ≈ val ≈ blind, all low → **fidelity wall**: training never absorbed its own
  data. Go to entry 2. More mining rounds ON HOLD.
- train high, val/blind low → **generalization gap**: mine denser / more tasks per
  regime; distill recipe is fine.

**Seen in.** EXP-0034 (train 0.159 ≈ val 0.19 ≈ blind 0.12 → fidelity wall).

**Standing gate.** Run train_closure as a standard output of EVERY replay/distill
experiment — it is eval-only and cheap; a low train closure is the earliest detector
for entries 2–3.

## 2. Fidelity wall — architecture can't express it, or training can't reach it?

**Problem.** Entry 1 showed the clone underperforms its own teacher on its own training
data. Is the scorer's observation/architecture fundamentally unable to represent the
teacher's behavior (dead end — change the contract), or able but the multi-task
training process fails (fix the training knobs)?

**Check** (pod): `tokenomics.eval.overfit_probe` — distill on ONE mined task at a time
with a generous epoch budget, exactly the replay recipe, deploy with the frozen
champion orderer on the task's paired seeds. Pick tasks where full training failed
worst (low/negative per-task closure).
```
python -m tokenomics.eval.overfit_probe --archive <archive.jsonl> \
  --attribution <gap_attribution.csv> --orderer-ckpt <ord.pt> --out-dir <out> \
  --task-ids <id1,id2,id3> --hiddens 64,256 --epochs 2000
```

**Outcomes.**
- Solo clones match the teacher (fidelity ≥ 0.98) → **multi-task
  optimization/interference**: fix trials = at-scale epochs ≫ 400, at-scale width,
  sample balance, staged/rung distillation. Judge each by entry 1's check.
- Solo clones don't match even here → **representability or recipe** (obs contract
  can't express the behavior, or collect-vs-deploy covariate shift): change obs/arch
  or the collect recipe before any scale knob.

**When to re-run.** NOT every experiment — the answer is a property of (architecture,
obs contract, distill recipe, task family) and doesn't rot. Re-run only when one of
those four changes, or when entry 1 fires with no other explanation.

**Seen in.** EXP-0034 (4/6 arms matched → multi-task interference; representability
cleared).

## 3. Policy weak on a regime — weak teacher, or weak cloning?

**Problem.** The distilled policy underperforms on a specific regime/rung. Is the
expert (teacher) itself weak there (need a better expert), or did cloning miss behavior
the teacher has (need staged/denser distillation)?

**Check** (local per-rung, pod for sweeps): compare three levels on the regime —
search bar vs teacher vs clone (`eval.producer_rungs`, or the teacher stack column in
`gap_attribution.csv`).

**Outcomes.**
- teacher ≈ bar, clone < teacher → **cloning-fidelity gap**: staged/rung DAgger,
  include the regime in collect. Entry 2 refines the cause.
- teacher < bar → **expert weakness**: improve the teacher/search, cloning is not the
  bottleneck.

**Seen in.** EXP-0004 (rung-0.24 weakness = cloning gap, teacher matched bar; fix =
rung-staged DAgger). EXP-0034 (teacher stack 1093u vs clone 1013u on train split).

## 4. Candidate regresses on a specific task — skill trade or noise?

**Problem.** A trained candidate loses badly to the champion on one task (e.g. a blind
outlier tripping a tail gate). Is it a reproducible behavioral change (real trade-off
the promotion call must accept) or eval noise?

**Check** (local): single-task paired-seed trace — run both policies on ALL paired
seeds (`range(episode_seed, episode_seed + eval_seeds)`), trace capacity/upgrade/
rejection series via `env.system.producing_agents` (`eval.producer_trace` or ad-hoc
trace script in the repo, never /tmp).

**Outcomes.**
- Same sign + same mechanism on all paired seeds (e.g. never upgrades the bottleneck
  primal, replenishment rejections climb) → **behavioral skill trade**: name the
  mechanism in RESULT.md; promotion accepts it or next round adds covering cells.
- Mixed signs / one-seed artifact → **noise**: widen eval seeds before concluding.

**Seen in.** EXP-0034 s54 (−107u = upgrade misallocation at layer-0 primal, all 3
seeds; real trade).

## 5. Which channel owns a gap — producer or orderer?

**Problem.** A task shows a champion-vs-bar gap. Which policy channel (producing agent
vs ordering agent) must improve to close it? Training the wrong channel wastes a round.

**Check** (pod for archives, local for one task): stack-swap attribution —
`eval.gap_attribution` (swap producer-only, orderer-only, both against the bar) or the
boundary-grid policy-swap probe.

**Outcomes.** producer-swap closes the gap → producer-recoverable (replay the producer);
orderer-swap closes → orderer lever; neither alone → interaction/noise, drop from the
training archive (`filter-recoverable`).

**Seen in.** EXP-0001 (producer is the depth-3 blocker), EXP-0034 (producer closure
1.026 vs orderer 0.386 → 113/119 recoverable).

## 6. Model looks good — is it memorizing?

**Problem.** A learned model (value model, scorer) shows strong fit where it was
trained; is it a transferable signal or in-sample memorization?

**Check** (local if data exists): held-out topology evaluation — R²/correlation on
economies never seen in fitting. In-sample metrics are inadmissible as evidence.

**Outcomes.** held-out R² ≤ 0 or corr ≪ in-sample → **memorization**: stop the lever or
add train diversity; high held-out → real signal.

**Seen in.** EXP-0009 (value model in-sample R² 0.70, held-out negative for EVERY model
class incl. ridge → value lever stopped). Also the open-loop policy collapse under
single-axis shift (stages 3–4, ADR-0038/0039 → domain-randomized training).

## 7. Objective mismatch — return says one thing, the goal says another

**Problem.** A policy "wins" on return but seems useless (or vice versa). At scale,
return is dominated by investment-order churn — do-nothing can win on return.

**Check** (local): re-score the same episodes on **consumer fulfilled units**
(`consumer_completed_units`); compare rankings under both metrics.

**Outcomes.** Rankings flip → return miscalibrated for this regime; use fulfilled units
as the objective (established for scale work). Rankings agree → objective fine, the
result stands.

**Seen in.** Stage 5 / ADR-0040 (return invalid at scale; fulfilled-units adopted).

## 8. Does the env even pose a learning problem here?

**Problem.** Before training on a new config/family: can a sensible policy beat
do-nothing at all? If not, training results are noise.

**Check** (local): `python -m tokenomics.eval.signal_check --config <cfg>` — baseline
matrix. Guardrail test: `test_eval::test_signal_exists`.

**Outcomes.** Strongest baseline ≫ noop → signal exists; that baseline is the bar RL
must clear. No separation → fix the config/family before any training.

**Seen in.** Phase-1 signal check (original harness rationale); EXP-0005 Stage-0
headroom audit (demand NOT saturated → lever confirmed).

## 9. Pod run stalls / crawls at high load

**Problem.** A multi-worker pod run makes no progress; box load explodes.

**Check** (ops): SSH in, `uptime` + progress lines. Load ≫ vCPU with 0 tasks done →
thread oversubscription.

**Fix.** Every ProcessPoolExecutor worker must pin threads in its initializer:
`OMP/MKL/OPENBLAS_NUM_THREADS=1` + `torch.set_num_threads(1)`.

**Seen in.** EXP-0034 gap_attribution (16 unpinned workers → load 157, 0/119 in 62 min;
fixed in `_init_worker`).

---

## 10. Probe numbers contradict the archive — same task, different result

**Problem.** A diagnostic/probe harness reports champion (or bar) units on an archived
task that disagree with what mining/`gap_attribution` recorded — e.g. champion alive and
scoring high where the archive says it died. The domain is deterministic, so the same
policies on the same config can only differ if the **episode seeds** differ.

**Check.** Compare the harness's eval seeds against the archive convention:
every consumer of a `TaskRecord` must evaluate at
`_episode_seeds(record.episode_seed, eval_seeds)` (the contiguous block anchored at the
record's seed), never `range(eval_seeds)` or the world seed. Grep the harness for
`range(eval_seeds)`; cross-check one task against its `gap_attribution` row.
Collect/deploy must use the SAME seeds (a distill collected at the archived seed but
deployed at 0-2 craters for a second reason).

**Outcome branches.**
- Seeds mismatch → the probe measured a different demand trace, i.e. a different task;
  all its verdicts on archived gaps are invalid. Fix the anchoring, rerun.
- Seeds match → real disagreement; suspect config reconstruction (`replay()`/generator
  drift) or policy loading.

**Seen in.** EXP-0036 orderer representability probe (champ_cu 957 alive at seeds 0-2 vs
archived full-death gap 973 at seeds 77-79; collector anchored, evals not — pod killed
mid-run, fixed + regression-tested in `test_headroom_row_anchors_at_record_episode_seed`).
**Caveat propagated:** EXP-0032b's headroom scan (`scan_headroom`) shared the bug — its
"0/26 orderer headroom" was measured at seeds 0-2, not the archived seeds. Its
conclusion stands only via EXP-0033 (family demand-bound by exact-seed analytic ceiling),
not on its own measurement.

## 11. Distill-from-search fails — is the TEACHER SIGNAL even learnable?

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

## 13. A supervised probe FAILS — is the input even valid, or is the fit real?

**Problem.** Before you trust a representability probe's negative result (e.g. "the target is not a
function of this observation → retire it"), the probe's *inputs* must be sound. Two silent input
defects both bias a probe toward a FALSE FAIL and neither shows up in the loss curve:
1. **Non-finite features from a mis-declared transform.** `log1p` applied to a field that can go
   negative (`free = current − reserved` when a prior-cycle reservation outlives degraded capacity;
   any `a − b` "headroom/slack" count) yields `NaN`/`−inf` (`log1p(x≤−1)`), which poisons every
   downstream activation and the loss. A field is only `log1p`-safe if it is provably ≥ 0.
2. **Flat-decision domination (within-decision probes).** For a *within-decision* argmax target
   (routing preference), a decision whose candidate labels are all equal carries **no** signal. If
   flats are the majority (EXP-0038 dump: 69% of 13,940) and training weights every decision equally,
   the model is rewarded for predicting ties and the real (non-flat) contrast is swamped — a FAIL that
   is a training-composition artifact, not a representability fact.

**Check** (seconds, before any pod spend):
- **Finite assert.** After encoding, assert every node/edge feature tensor `np.isfinite(...).all()`;
  make the dump raise on the first non-finite value. Any signable count uses a sign-preserving
  transform (`signed_log1p(x)=sign(x)·log1p(|x|)`), not `log1p`.
- **Flat fraction + training filter.** Report the flat-decision fraction; train PRIMARILY on non-flat
  decisions (exclude flats or downweight so they cannot dominate) and record how many were excluded.
- **Fold-local standardization present.** Continuous columns fit mean/std on the *training fold only*,
  applied to held-out; binary/one-hot columns untouched. A probe over unstandardized mixed-scale
  features (log1p tokens vs 0/1 flags) underfits for a reason unrelated to representability.

**Outcome.** If any input defect is present, the FAIL is **invalid** — fix inputs and re-run before
concluding anything about the target. Only a probe with finite inputs, a non-flat-dominated training
set, and fold-local standardization can license retiring (or promoting) an observation/target.
(Seen: EXP-0039 GNN probe pre-flight — `free` declared `log1p` produced `−inf`; the gate trained on all
13,940 decisions including 9,675 flats; no standardization. All three fixed before the pod run.)

**And don't trust a PASS either — audit the verdict, not just the inputs.** A "mechanical" gate can
false-PASS or become unauditable in three ways, all worth checking before a read licenses the next
experiment: (a) **dispersion must cover every metric.** A seed-mean that averages an unstable metric
hides it — require each pre-registered condition to hold for the mean AND independently for each of
the ≥3 training seeds (r² = −100/−100/201 averages positive but two seeds are negative → must be
no-pass). (b) **The read must be recomputable from saved predictions**, without a checkpoint or
re-running the simulator: persist per-candidate `predictions.csv` (decision, candidate, producer,
pred, teacher, actor) + the per-fold normalization *actually used to train* (not an all-data refit
that leaks held-out/flats) + loss history + checkpoint provenance. (c) **Validate the dump before
scoring it**: require the schema file, match its hash, and check per-decision candidate
coverage/index/count/producer-alignment — a global row-count check passes even when every candidate
is collapsed onto one decision. (d) **Validate metadata against the authority, and enforce the
pre-registered shape.** Each candidate row's task/seed/cycle/order must equal the decisions.csv row
(intra-group consistency is not enough — a flipped seed still passes); and the gate's own shape
(exactly N tasks × ≥K seeds) is a *pass condition* — a "≥2 of 3" threshold trivially passes on 2
tasks, so a partial dump must be rejected, not scored. (e) **Certify determinism on the TARGET
device.** A licensing read must be reproducible: pin BLAS to one thread for bitwise-exact CPU, and run
a same-seed twice-trained preflight that aborts if predictions diverge beyond a *declared* tolerance
(CUDA `index_add_` is order-nondeterministic → don't run the deterministic read on GPU without a
recorded tolerance). (Seen: EXP-0039 second+third reviews — verdict averaged r² while checking only
regret dispersion; `_load_dump` accepted a missing schema, mis-grouped candidates, and a flipped
candidate seed; `_verdict` passed on 2 tasks; persisted normalization was an all-graph refit; CPU
predictions jittered ~5e-4 across runs until BLAS was pinned to 1 thread. All fixed before the pod
run.)

**And the labels must be bound to the RIGHT graph, generated by the RIGHT config.** Two more ways a
dump that passes every count/alignment check can still carry mislabeled or wrong-trajectory data:
(f) **bind each graph payload to its decision row.** Per-decision *metadata* agreeing with itself does
not prove `graphs[di]` is the graph built for decision `di` — swap two structurally-identical decisions
(same candidate set, different live features) and coverage/count/producer-alignment all still pass while
every dynamic feature is now mislabeled. Stamp each graph with a `decision_digest` of its identity
`(task_id, seed, cycle, order_id)` as a **separate audit array** (never a model tensor, preserving
identity isolation), and have the loader recompute it from the authority CSV and reject a mismatch.
(g) **validate the dump was produced on the pre-registered config.** The decision *count* and
`decisions.csv` can be byte-identical while the teacher (`delta`/`lookahead`), task set, or horizon
differ — which changes every *label* silently. Read the dump's own `spec.json` and reject any deviation
from the pinned gate config (exact task-ids set, teacher params, horizon, seed count, floor, has-actor),
and enforce **exactly** the anchored trajectory-seed count (not "≥") so leave-one-seed-out stays
"train-two-hold-one". Also: whatever floor the verdict actually used (a derived actor floor or an
override) must appear in the metrics table + a floor artifact with its source, or an overridden read
contradicts its own table. (Seen: EXP-0039 fourth review — `_load_dump` accepted a shard with graph
payloads 0↔35 swapped and accepted a dump with changed task-ids/delta/lookahead/horizon; an override
floor of 999 sat next to a table reporting the derived 0.33. All fixed before the pod run; schema hash
unchanged since the fixes are audit-binding/validation, not observation changes.)

**And bind the actual INPUT files, exact seeds, and full-precision floor — names/shapes aren't
provenance.** A dump can pass config, task-id, seed-count, digest and alignment checks and still be built
from the wrong models on the wrong seeds: (h) **fingerprint the input files, don't just name them.** A
producer or actor checkpoint of the right *shape* but wrong *content* silently changes the teacher labels
(producer) or the licensing floor (actor); "has_actor: true" cannot tell them apart. Record a SHA-256 of
every input (archive, producer ckpt, actor ckpt) in the dump manifest and validate it against the frozen
pre-registered hashes at read time. (i) **enforce the exact seed VALUES, not the count.** "Exactly three
anchored seeds" still accepts any three integers; pin the real sets (here `range(episode_seed,+3)` →
s29={29,30,31}, …) and check them against both the dump manifest and the scored rows. (j) **the
reproduction artifact must keep full precision.** A floor CSV rounded to 4 dp can turn a `1.000049` PASS
into a `1.0` FAIL — write the verdict-deciding floor at round-trippable precision (`repr(float)`) and
record any override's source path+hash, so the CSV reproduces the verdict independently. And (k) **don't
overclaim the determinism scope**: a single full-epoch fold repeated is a *representative* certification
(sound because CPU bitwise determinism is by construction), not a re-run of the whole multi-fold,
multi-seed gate — say which one you did. (Seen: EXP-0039 fifth review — the dump bound only `has_actor`,
accepted any three seed values, and `floor_by_task.csv` rounded a floor that could flip the verdict. All
fixed before the pod run.)

Three more sources of a rigged-but-green run, all about what the artifacts are *licensed against* rather
than their numbers: (l) **bind the run to an immutable code revision, not just the schema.** The
observation-schema hash is blind to changes in training/metric/verdict code, and a pod rsync that excludes
`.git/` means the artifact carries no revision at all — record a git-independent SHA-256 over the bytes of
the producing modules (reproducible on the pod) plus the injected commit, in every dump/probe spec, and
commit the source before launch. (m) **a fallback bar must be authoritative, not caller-supplied.** An
override floor read from an arbitrary `{task: regret}` JSON lets an inflated value silently make the bar
easier (the verdict wants model-regret < floor-regret); pin the canonical floor artifact by hash, derive
the override mechanically from it, and abort if a supplied override disagrees. (n) **validate expensive
inputs before you spend on them, not after.** Recording input hashes only helps at read time — the wrong
staged checkpoint has already burned the full rollout; re-check the hashes at the *start* of the dump CLI
so a mismatch fails at launch. (Seen: EXP-0039 sixth review — core files were uncommitted with no source
hash in artifacts, `--actor-floor-json` accepted any values, and input hashes were checked only after the
~4-core-hour dump. All fixed before the pod run.)

Provenance that is *recorded* but not *enforced* is not provenance — three ways a green run is still
rigged: (o) **make the source identity a required cross-stage invariant, not audit metadata.** Stamping a
source hash into the spec does nothing if the reader never checks it — a dump with a zeroed hash and a
bogus commit sails through, so a dump from source A earns a verdict under source B. Compare the dump's
recorded hash to the *live* fingerprint of the code now reading it, **before** training, and require a
real commit (reject `None` and `not-a-git-object` by shape). And fingerprint the **whole synced tree**,
not a hand-picked module list — a subset silently ignores changes to the teacher, env, curriculum, and
domain deps that produce the very labels being scored. (p) **a canonical-value check needs exact equality,
not a tolerance.** `abs(v - pinned) <= 1e-6` still lets a `+5e-7` inflation through, and a strict `<`
verdict gate can flip inside that interval; compare exactly (canonical floats round-trip exactly from
repr/JSON) and, better, *substitute* the pinned constant into the verdict so no supplied drift survives.
(q) **a fallback must be reachable in the environment that needs it.** A default artifact path under
`experiments/…` does not exist on a pod that syncs only the package tree — ship the fallback as an in-code
pinned constant (a flag that needs no file), or the documented recovery path fails exactly when used.
(Seen: EXP-0039 seventh review — `_load_dump` never validated the recorded source, the floor check used a
`1e-6` tolerance, and the canonical-floor artifact was outside the synced pod tree. All fixed before the
pod run.)

Even an enforced identity is only as good as what it's bound to — two last gaps: (r) **a recorded commit
must be a real, clean revision, not a well-shaped string.** Accepting any 7–40 hex id lets a dirty tree be
stamped with a bogus commit whose tree hash still matches — the artifact is not recoverable from the
claimed revision. On the machine that HAS git (the launcher, not the pod), resolve the full 40-char HEAD,
verify the object exists (`git cat-file -e`) AND the producing tree is clean (`git status --porcelain`),
and inject that; reject abbreviations pod-side. And validate the commit's shape at the *start* of the
expensive stage, not only in the later reader. (s) **capture and compare the runtime, not just the code.**
Open dependency bounds installed fresh per run mean two stages can execute different numpy/torch/pettingzoo
versions that silently disagree on labels or the fit; record a version manifest (python + the libraries
that shape the trajectory and training) in both stages' specs and compare them exactly at read time, or
pin a lockfile / reuse one immutable image. (Seen: EXP-0039 eighth review — `_validate_commit` accepted
`a1b2c3d`/`deadbee`/forty zeros, the dump CLI never checked the commit before spending, and only
`torch_version` was recorded. All fixed before the pod run.)

A control that lives in prose is not a control — the last two: (t) **enforce the binding in the launcher,
not in a manual pre-command.** A helper that resolves+verifies a clean commit does nothing if the launch
script never calls it — the operator can still pass a hand-typed (or bypassed) value, and the remote side,
lacking `.git`, only checks shape. Do the resolution+verification in the launcher itself, before the code
is shipped, and inject the result into the remote command (placeholder substitution); a dirty/non-repo
tree must abort the launch. (u) **an environment gate must cover the machine, not just package versions.**
Identical `numpy`/`torch` versions on a different CPU architecture or math backend (MKL vs OpenBLAS) are
not the same numerical environment; gate at least machine arch and a BLAS/build identity (e.g. a hash of
`torch.__config__`), and record the image digest — otherwise a cross-arch or cross-backend mismatch passes.
(Seen: EXP-0039 ninth review — `resolve_verified_commit` was never wired into `run_pod_training.py`, and
the env gate ignored arch/BLAS so a `machine="other-arch"` manifest was accepted. Both fixed; a recorded
test-count error was also corrected against actual collection.)

Last mile — the verified set must include the verifier, and advertised fields must be filled: (v) **the
launcher is producing code too.** Verifying the cleanliness of only the synced payload misses the launch
script itself, which resolves the commit, rewrites the command, and performs the launch — a launcher
modified after HEAD yields a "clean" commit that does not describe what ran. Include the launcher file in
the cleanliness check (scoped to payload + launcher, not the whole repo, so intentionally-dirty pulled
artifacts don't block a later stage). (w) **a manifest field nobody populates is worse than none** — it
reads as provenance while carrying `null`. If the manifest advertises an image/digest, the launcher must
actually obtain it (e.g. REST `imageName`) and export it into the run's environment; otherwise drop the
field. (Seen: EXP-0039 tenth review — `resolve_clean_commit` excluded `run_pod_training.py`, and
`EXP0039_POD_IMAGE` was read by the manifest but never set by the launcher. Both fixed before the pod run.)

---

## 14. A load/decode step OOMs a big-RAM host on a SMALL dataset

**Problem (plain English):** a run dies with `Killed` / exit 137 (SIGKILL = the kernel OOM killer)
while *loading* data — before any training/epoch output — even though the dataset on disk is tiny
relative to the host RAM (e.g. a 124 MB dump killing a 32 GB pod). The "just use a bigger pod" reflex is
wrong here: a ~250× blowup means the loader's memory is not O(dataset) but O(dataset × items), so a bigger
host only delays the same death. Root-cause the blowup first (CLAUDE.md: cause before next strategy).

**Discriminating check:** load ONE unit (one shard/file) locally under a peak-RSS meter
(`resource.getrusage(...).ru_maxrss`), timing it, in two ways: (a) as written; (b) with every
container member read **once** into a dict up front, then sliced. If (b) collapses both peak RSS and
wall-time by orders of magnitude (seen: OOM/hang → 66 MB / 0.09 s for one 2000-graph shard), the loader
is **re-reading a lazy container per item**. The classic culprit is NumPy's `np.load(...)` `NpzFile`:
`__getitem__` re-reads and **re-decompresses the whole member array from the zip on every access**, so
indexing it inside a per-item loop is O(n_items × member_bytes).

**Outcome branches:**
- **(a) Per-item re-read of a lazy container (NpzFile, h5py, memmap-per-access):** materialize each
  member once (`arr = {k: z[k] for k in z.files}`), then slice views out of the in-memory arrays. Peak
  memory drops to ~O(dataset); the per-item tensors become cheap views. Guard it with a test that counts
  member reads (`test_load_shard_reads_each_npz_member_at_most_once`). (Seen: EXP-0039 probe — the 32 GB
  OOM was entirely `load_shard`, not training; fixed, 13940 graphs load in 0.45 s / 246 MB.)
- **(b) Loader is O(dataset) but the COMPUTE is quadratic** (a dense N×N built from a sparse graph, an
  accidental cartesian product): the check shows load is cheap and the kill lands *after* the first
  forward/batch — fix the algorithm (sparse `index_add_`/scatter, minibatch), don't resize.
- **(c) Genuinely O(dataset) and the host is just too small:** only *then* is a bigger/memory-optimized
  host the answer — and record the measured peak so the shape is sized from evidence, not guessed.

---

## Unverified problems (plain-English inbox)

Humans: describe a suspected problem here in plain English — no tooling knowledge
needed. The agent turns it into a check and promotes it to a numbered entry once it has
discriminated a real case.

_(empty)_
