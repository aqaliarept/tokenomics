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

## Unverified problems (plain-English inbox)

Humans: describe a suspected problem here in plain English — no tooling knowledge
needed. The agent turns it into a check and promotes it to a numbered entry once it has
discriminated a real case.

_(empty)_
