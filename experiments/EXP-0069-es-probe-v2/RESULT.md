# EXP-0069: Re-probe under pre-registration v2 (minibatch 24, noise margin, larger-step arm)

**Status: complete, valid. Not an acceptance read.** Task 6.1e of `add-m4-orderer-redesign`.
- **D7 reading (A1): `weak`**, with cosine 0.222.
- **Step reading (A1 at σ vs A1-S2 at 2σ): `heterogeneity`**. Both arms clear 0.1 (0.222 and
  0.440).
- **A2:** cosine −0.407, so A2 uses R = 2 (`a2_arm` A2-R2).
- **Driver and commit:** `order_es --mode probe --arms A1,A2,A1-S2 --prereg …/es_prereg_m4_v2.json`
  at `e35c4a6b7a39…`, verified clean, from a clean worktree.
- **Inputs:**
  - pre-registration v2 `511e939a4121…`;
  - EXP-0058's bar (normaliser `2d574ec8a3df…`, `bar.csv` `3e078c43c41b…`) with its A0 rows as
    the margin's references (`episodes.csv` = EXP-0058 `artifacts/bar_episodes.csv`,
    `0580eabcf33c…`);
  - EXP-0012 `orderer_scorer.pt` (`d25388bfa33e…`).
- **Run:** 3,456 jobs (3 arms × 24 members × 2 minibatches × 24 world-episodes), horizon 100,
  `entity_planner`, on one fresh secure 16 vCPU pod. ≈ 3.0 vCPU-h billed.
- **Validity:** 0 contract failures. The cosines and the reading recompute offline from
  `probe_fitness.csv`, and the member scores recompute from `probe_episodes.csv` (144 / 144).

> **Erratum (review round 1, 2026-11-03): `probe.json` mislabels A1-S2's noise stream.**
> - It records `"noise": {"A1-S2": {"purpose": "probe|A1-S2"}}`. The run used A1's stream,
>   `probe|A1`, as registered: `probe_noise` resolves the base arm via `base_arm`, so A1-S2 has
>   the same ε_i as A1.
> - The label is wrong only because the record writes `f"probe|{arm}"` from the arm label
>   (`order_es.py` probe.json writer). The run itself is unaffected.
> - Recomputed with the true stream (`probe|A1`), A1-S2's cosine is **0.4396**, as recorded. The
>   reading (`heterogeneity`) and every number in this file stand.
> - Recomputing from the label literally (`probe|A1-S2`) would give ≈ 0.049 (reviewer), which
>   is wrong.
> - The raw `probe.json` is left untouched. The code fix (`f"probe|{base_arm(a)}"`) is the
>   supervisor's call, before the next probe.

## In plain English

**The question.** Before training, the ES gradient-signal probe asks whether one ES step points
in a consistent direction. It scores the same 12 perturbation pairs on two disjoint sets of
training world-episodes and compares the two gradient estimates by their cosine: 1 means the
same direction, 0 means unrelated.

EXP-0058 asked this with sets of 6 world-episodes and found no direction (cosine −0.07). The
diagnosis (EXP-0068) left two explanations open:
- **heterogeneous:** small batches disagree because each world pulls its own way;
- **flat at σ:** the perturbations are too small to beat random noise.

This run changes three things, all registered before the data (pre-registration v2):
1. **Bigger sets:** 24 world-episodes each.
2. **A noise margin on the disqualification rule.** A perturbed policy is penalised only when
   it is worse than the unchanged policy (A0) by more than A0's own measured random variation.
3. **A second copy of A1 at twice the perturbation size (A1-S2).** It is diagnostic only: if
   only the bigger step finds a direction, the step was too small ("flat at σ").

**What happened.**
- **A1 (the arm the probe reads): cosine 0.22.** That is `weak`, between the registered 0.1 and
  0.3, so there is a direction, though a noisy one. Under v2 both `weak` and `signal` train at
  minibatch 24.
- **A1-S2 (twice the step): cosine 0.44.** The two estimates agree more strongly.
- **The registered step reading is `heterogeneity`.** Both the normal and the doubled step find
  a direction (each ≥ 0.1). By the pre-registered table, that means the earlier failure was
  about batch size, minibatch 24 suffices, and training proceeds at the normal step σ.
- **A2 (the larger model): cosine −0.41.** Its two estimates point in opposite directions. The
  registered rule ("A2 below 0.1 while A1 is at least 0.1") switches A2 to its rank-2 form
  (A2-R2) for training.

**Three things the reading does not say** (reported, never reading-changing):
1. **A1's direction leans on the disqualification ranking.**
   - Even with the margin, 28 of A1's 48 member scores are disqualified (12 pairs × 2 signs × 2
     minibatches). Under the v1 rule all 48 would have been, which would have zeroed the
     estimate.
   - Ranking A1 on raw fitness with the rule off gives a cosine of only 0.058.
   - A1-S2 does not depend on it: 0.49 with the rule off.
   - So A1's `weak` is partly carried by which members are flagged. The flags sit on a few
     worlds: 12-0055, atlas 12-0361, 12-0014, 12-0045, mostly starvation.
2. **The doubled step reads about twice as strongly** (0.44 against 0.22, `signal`-strength by
   D7's threshold). That is also what a landscape that is partly flat at σ would show. The
   registered table reads `heterogeneity` because A1 also cleared 0.1. It does not grade the
   gap, and the step stays σ unless the user decides otherwise.
3. **A2 is not just weak but anti-aligned** (−0.41). Its estimate norms are the largest
   (15.6 / 18.3). The registered remedy is R = 2.

**What changes because of it.** The probe licenses the pilot under v2: `weak` at minibatch 24,
step σ, A2 at R = 2 for its main run. Tranche A's extension to 125 becomes available (it was
conditional on A1 reading `signal` / `weak`). The supervisor decides the pilot launch (EXP-0059,
task 6.2). DIAGNOSTICS 25 gets this evidence.

## Details

### Readings (`probe.json`)

| arm | σ | cosine | estimate norms (mb 1 / 2) | variance ratio (paired / unpaired) | disqualified (of 48) | cosine, rule off (reported) |
|---|---|---|---|---|---|---|
| **A1 (reads D7)** | 0.05 | **0.2224** | 8.89 / 8.96 | 0.021 | 28 | 0.058 |
| A2 | 0.05 | −0.4072 | 15.60 / 18.31 | 0.014 | 28 | −0.052 |
| **A1-S2 (step arm)** | 0.10 | **0.4396** | 3.92 / 3.38 | 0.036 | 29 | 0.490 |

- **D7 on A1:** 0.1 ≤ 0.222 < 0.3 → **`weak`**, training minibatch 24 (v2).
- **A2:** −0.407 < 0.1 while A1 ≥ 0.1 → A2 uses R = 2 (`a2_arm` A2-R2, `a2_rank` 2).
- **Step table (v2 `probe.sigma_arms`):** A1 ≥ 0.1 and A1-S2 ≥ 0.1 → **`heterogeneity`**:
  "heterogeneity was the issue; minibatch 24 suffices; pilot at sigma".
- **CRN pairing works:** the paired / unpaired variance ratios are 0.014–0.036.
- The A1-S2 estimate norms are smaller because the ES estimate is divided by σ.

### Disqualifications under the v2 margin (`probe_fitness.csv` reasons)

| arm | minibatch 1 / 2 | starvation / run-down / death flags | world-episodes with most flags |
|---|---|---|---|
| A1 | 16 / 12 | 23 / 9 / 1 | 12-0055 (9), atlas 12-0361 (7), 12-0014 (5), 12-0045 (4) |
| A2 | 13 / 15 | 23 / 13 / 1 | 12-0055 (9), 12-0014 (8), atlas 12-0361 (8), 12-0045 (5) |
| A1-S2 | 14 / 15 | 20 / 13 / 1 | 12-0055 (9), 12-0045 (7), atlas 12-0361 (7), 12-0014 (6) |

Re-scored offline under the v1 rule (no margin), every one of the 48 member scores in every arm
is disqualified. This is reported only; v2 is the registered rule.

### Method
- **Minibatches:** two disjoint sets of 24 training world-episodes on 48 distinct worlds, from
  the recorded probe stream (`order_es_worlds.probe_minibatches(…, 24)`). They are not the bar's
  size-6 entry.
- **Members:** each arm's registered initial θ ± σ·ε_i, i = 0..11. A1-S2 uses A1's ε_i and 2σ.
  Every member plays every world-episode of both minibatches at the training draw key
  (replicate 0). Members share keys (CRN).
- **Scoring:** fitness = units / training bar − h · holding / demand (D6), and v2's relative
  disqualification with the noise margin (run-down 0.0961, starvation 0.1169). The A0
  references are the bar's A0 rows, checked against `bar.csv` on load.
- **Estimate:** centered ranks per minibatch. Read on A1 with D7's thresholds (0.3 / 0.1); the
  step table compares A1 with A1-S2.

### Verification
- **Pre-launch (local, no pod yet):**
  - SHA-256s of the staged orderer, the A0 references and the pre-registration;
  - the driver's real-run checks (`prereg_problems`, `prereg_file_problems`,
    `real_run_problems`) on the staged inputs at the commit: `[]`;
  - 2 × 24 world-episodes on 48 worlds, 128 A0 references.
- **After the pull:**
  - `probe_from_rows(probe_fitness.csv)` gives the same cosines, variance ratios,
    disqualification counts and reading as `probe.json`. One A1-S2 estimate norm differs in the
    last float digit (3.38007578183061 vs …6097, pod vs local arithmetic).
  - Re-scoring all 144 arm × minibatch × member cells from `probe_episodes.csv` (v2 margin)
    reproduces `probe_fitness.csv` exactly: fitness repr, disqualified, reasons.
  - `spec.json`: status complete, pins prereg `511e939a…`, commit `e35c4a6…`.
  - `contract_failures.csv` header only.

### Cost

| pod | shape | life (UTC) | billed |
|---|---|---|---|
| `nhpvt01m74ktf7` | secure 16 vCPU | 10:40:53 launch → terminated by 10:52:03 (≤ 11.2 min) | **≈ 3.0 vCPU-h** (upper bound from client times) |

- Estimate 5.6 (10-min setup; 3.2 at a 1-min setup), stop 11.2. The 40-minute watchdog never
  fired.
- Launch-fit 12.2 + 11.2 = 23.4 ≤ 100. Tranche A billed so far ≈ 15.2.
- Default detached flow: pull, stop, terminate. `runpodctl pod list` = [] afterwards.
- No local step ran while the pod billed.

### Files

| file | what |
|---|---|
| `spec.json` | pins (bar hashes, normaliser, orderer, pre-registration v2, the minibatches, the noise margin, the A0-reference hash, source tree), status, result |
| `probe.json` | per-arm cosine, norms, variance ratio, counts; the reading with `step_reading`; the registered probe section. **Its `noise.A1-S2.purpose` should read `probe|A1` (erratum above)** |
| `probe_fitness.csv` | 144 rows: arm × minibatch × member fitness, rank, disqualified, reasons |
| `artifacts/probe_episodes.csv` | 3,456 episode rows (gate + trace + ES columns; git-lfs), SHA-256 `ec0572f535e7…` |
| `contract_failures.csv` | header only |

## Conclusions

- **D7 (A1): `weak`** (0.222). **Step table: `heterogeneity`** (A1 0.222, A1-S2 0.440, both ≥
  0.1). **A2 → R = 2** (−0.407). By the registered rules the pilot is licensed at minibatch 24,
  step σ. The launch is the supervisor's decision (task 6.2).
- **Reported caveats:**
  - A1's `weak` depends on the disqualification ranking (0.058 with the rule off), which still
    flags 28 / 48 member scores on a few worlds;
  - the doubled step reads twice as strongly (0.44), which is consistent with partial flatness
    at σ;
  - A2's two estimates are anti-aligned.
  - None of these changes the registered reading.
