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
**A0c gate (orderer_a0c/):** discriminator ANSWERED — global state fails too (held-out
gchoice_top1 vs floor: 0.31/0.53, 0.29/0.54, 0.40/0.42; 2/3 clear fails, 1/3 tie). The wall
is NOT missing context: the deterministic search's choice hinges on lookahead consequences
that don't compress into a state-readable pattern. Obs enrichment is DEAD as a fix for this
symptom. Fix directions: A2 deploy-time light search (realizes headroom by construction) or
A1 soft/advantage labels (validate critic held-out ranking first — decisive-cycle fails say
ranking from state is hard too).

---

## Unverified problems (plain-English inbox)

Humans: describe a suspected problem here in plain English — no tooling knowledge
needed. The agent turns it into a check and promotes it to a numbered entry once it has
discriminated a real case.

_(empty)_
