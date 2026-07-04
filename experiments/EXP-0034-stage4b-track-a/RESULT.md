# EXP-0034 — Stage 4b Track A: Capacity-Binding Mining

Stage 4b Track A of `CURRICULUM_GAME_MASTER_APPROACH.md`. Rerun Stage-3 hard-case
mining in a regime where **capacity**, not demand, is the ceiling — the only regime
with headroom by construction. Stages 0–4 hit a fixed point on a demand-saturated
family (EXP-0031/0032b/0033: champion already 98.3% of exact demand, producer replay
dead at 5% closure). Track A asks: does a capacity-tight family surface trainable
producer weakness, and does producer augment-replay actually move the needle there?

## Question

Given a ceiling **pre-flight** that keeps only grid cells whose no-slack capacity
ceiling sits below demand at every mined world seed, does mining that family surface
reproduced in-claim `policy_weakness`, is the gap **producer-recoverable**, and does
the Stage-4 augment-replay producer beat the incumbent on held-out capacity-binding
tasks without regressing the broad demand-bound family?

## Pipeline (each stage gated on the prior)

1. **Pre-flight** (`eval.curriculum_capacity preflight`, local, torch-free): 4/4
   candidate cells capacity-bound across world seeds 20–119 (`b2_over_b0_worst < 1`,
   `binds_frac == 1.0` each). No-slack B2 is a **mining selector**, not a ceiling
   certificate — real policies can upgrade past it. → `artifacts/vetted_grid.json`.
2. **Mining** (`eval.curriculum_mining --grid`, pod, 16 workers): 4 cells × seeds
   20–119 = 400 tasks, bar `entity_planner+mpc_search`, gates `min_gap 2.0`,
   `min_win_frac 0.6`. **146/400 policy_weakness → 119 archived** (all 4 config cells).
3. **Gap attribution** (`eval.gap_attribution`, pod, 16 workers, orderer-cross on):
   119 tasks, 4 stacks. **producer_closure_ratio 1.026**, orderer 0.386. Channels:
   producer 83 / orderer 16 / mixed 14 / interaction_or_noise 6.
4. **Recoverability filter** (`curriculum_capacity filter-recoverable`, local):
   **113/119 kept** (dropped the 6 `interaction_or_noise`).
5. **Augment replay** (`eval.curriculum_replay --augment --freeze-orderer`, pod):
   producer distill on the 113 recoverable tasks (57 train / 28 val / 28 blind),
   champion = EXP-0030 producer_candidate, orderer frozen at EXP-0012.

## Result — premise VALIDATED; candidate advisory-fail (human promotion call)

- **Headroom exists where the saturated family had none.** 36.5% of the capacity-tight
  pool is `policy_weakness` vs ~0 on the demand-saturated family.
- **Gaps are producer-recoverable.** Producer-only swap recovers >100% of total positive
  gap units (1.026); orderer swap only 0.386. 70% of tasks are producer-dominant.
- **Producer augment-replay finally moves the needle** — the lever that was *dead on
  arrival* on the saturated family (EXP-0031: 5% closure, 21/26 not_producer):

  | split | candidate − champion (mean) | better / worse | notes |
  |---|---|---|---|
  | validation | **+14.80u** | 16 / 12 | net + |
  | blind | **+8.33u** | 18 / 10 | net + |
  | broad transfer (seeds 6–9) | **0.0** (byte-identical) | — | no forgetting |

- **Not Pareto → advisory-fail.** ~⅓ of tasks regress; the worst blind task (−107u,
  `capacity_scale0.35 ppl2 qmax14 s54`) trips the conservative **tail** gate
  (`blind_tail 721.67 < champion 772.33`). `objective_gate_pass=false`,
  `promote_recommended=false`. 0 contract failures, 0 transfer regressions, 0 deaths.

**Verdict: Track A works as a mining regime — capacity binding revives producer weakness
and the producer lever, closing the "family demand-saturated" fixed point of Stages 0–4.
The Track-A producer is a mean-improving, tail-trading candidate; promotion is a human
call (same posture as the EXP-0030 incumbent's own advisory-fail override).**

## Diagnosis of the −107u blind task (post-hoc, 2026-07-04)

Local single-task trace of `...capacity_scale0.35-ppl2-qmax14-s54` (candidate 771.33 vs
champion 878.0, paired seeds 54–56; champion was already 99.2% of bar here, mined gap
6.67u). The regression is **behavioral, not instability**: candidate loses on all 3
seeds (−115/−165/−40), 0 deaths, cap_min identical, cycles full.

Mechanism = **upgrade misallocation at the layer-0 primal bottleneck**. s54's layer-0 is
dominated by `p0_0` (initial max 48 vs sibling's 6; one upgrade = +31 for 35 tokens).
The champion buys that single upgrade early (max 48→79 by cycle 10) and holds `p0_0`
near max all game; the candidate **never upgrades `p0_0` in 100 cycles**, running it at
28–46 of max 48, and spends its upgrades mid/late-chain instead (`p1_1` max 193→234;
fewer `p2_*` upgrades: champion ends 360+417 max vs candidate 252+323). Layer-0 supply
caps the whole chain: replenishment rejections climb (135→171 on seed 55), cumulative
units track even through cycle 30, then diverge steadily — a compounding input-starvation
deficit, no cliff. Same signature on all 3 seeds (cap_final lower, repl_rej higher).

Reading: the Track-A replay taught the candidate to shift upgrade priority toward the
layers that bind on most mined tasks, and it **traded away the champion's "upgrade the
tight primal source first" behavior** — the augment's broad mix protected the demand-bound
family (byte-identical) but not this in-family skill. The −107u is a real skill trade,
not noise: promotion accepts losing small-primal-bottleneck economies, or the next replay
round needs primal-tight cells in the mix.

## Train-closure diagnostic (2026-07-04) — harvest bottleneck = DISTILL FIDELITY

Why did the candidate capture so little of the mined headroom (+1.4% of 7.3%)? Two
candidate causes demand opposite responses: a fidelity/representability wall (fix
training; more mining useless) vs a generalization gap (mine denser; more rounds pay).
Discriminating diagnostic: `eval.train_closure` — run the candidate on the 57 tasks it
was TRAINED on and measure gap closure there (champion + bar joined from
`gap_attribution.csv`, same paired-seed convention; candidate-only rollouts, pod).

| split | closure of (bar − champion) |
|---|---|
| **train** | **0.159 aggregate, 0.098 median/task** |
| val   | 0.19 |
| blind | 0.12 |

Train ≈ val ≈ blind → **no generalization gap**. Sharper: the exact distill teacher
stack (`bar_producer_champ_orderer` = entity_planner producer + champion orderer)
scores **1093.3u**/task on the train split; the candidate distilled from it scores
**1013.0** — 13% of its own teacher's gap captured on its own training data, clone at
**92.7% of teacher**. 26/57 train tasks regress vs the starting champion; per-task
closure quartiles −0.51 / 0.10 / 0.57. EXP-0004's pattern again: **cloning-fidelity
gap**, not expert weakness, not coverage.

Consequences: (a) more Track-A mining rounds (incl. the EXP-0035 primal-tight grid)
are ON HOLD — new ore, same leaky refinery; (b) the promotion question for this
candidate is unchanged (its +8-15u means are real), but the growth path is a distill
fix, not more cells; (c) any Track B mining harvests through the same recipe, so the
fidelity wall is on Track B's critical path too. Next discriminator (not run): overfit
ONE mined task (single-task distill, generous epochs) — teacher matched → optimization/
capacity at scale (bigger scorer, staged DAgger à la EXP-0004); not matched →
representability (scorer obs can't express the planner's lookahead behavior).

Artifacts: `train_closure/train_closure.csv` (+ summary JSON); tool
`tokenomics.eval.train_closure` (reusable closure-by-split diagnostic).

## Artifacts

- `artifacts/preflight.csv`, `artifacts/vetted_grid.json` — pre-flight table + vetted grid
- `artifacts/task_archive.jsonl` (119), `artifacts/recoverable_archive.jsonl` (113)
- `summary.csv` — mining pool (400 rows)
- `gap_attribution/gap_attribution.csv` + `summary.json` — channel split
- `replay/` — candidate producer + val/blind/transfer CSVs + advisory gate (`replay/RESULT.md`)

## Reproduce

Pre-flight local; the three pod stages via `run_pod_training.py --create-pod --cpu
--python python3.12` (16 vCPU), staging the two champion ckpts + each archive under
`tokenomics/_pod_ckpts/` (driver syncs the inner package dir). Mining grid =
`artifacts/vetted_grid.json`; replay = `curriculum_replay --augment --freeze-orderer
--depth 3` (champion-matched defaults). See `PROGRESS.md` (2026-07-03 Track A entry)
for the exact invocations, including the `gap_attribution` thread-pin fix
(`_init_worker` now pins `OMP/MKL/OPENBLAS=1` + `torch.set_num_threads(1)` per worker;
without it 16 workers oversubscribed the box, load 157, 0/119 done in 62 min).
