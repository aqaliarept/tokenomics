# EXP-0111: power calibration of the Stage 0 gradient check: absent or underpowered? (DIAGNOSTICS 35)

Supervisor decision 2026-11-06. Reading rule pre-registered in
`openspec/changes/add-m6-joint-mpg-gnn/notes/exp0111-prereg.md` (SHA `43b02d23…`), before any computation. Code
dd322dc + 9cee716 (`train/m6_power.py`). EXP-0110's rows (SHA `eb4bfa39…`).
- **Attempt 1:** pod h5ouxz8scbb148 refused at start (the note was outside the synced package dir), ≈ 0.6
  vCPU-h.
- **Attempt 2:** pod 9gce2b3c9penyp (secure 8 vCPU / 16 GB + volume, EUR-IS-1), 20:33–20:44 UTC ≈ 1.5 vCPU-h.
- **Total ≈ 2.1 vCPU-h** (stop 3).

## Result: **`absent`**
At 12 producers, neither role's true gradient is distinguishable from zero (the 95% interval of |∇J|² includes
0), and no estimator reaches 80% power at ≤ 384 seeds per half.

**True gradient size.** Paired ±ε perturbations along random unit directions; ε per direction gives a mean
per-decision KL of 0.01 (PPO-sized). Common random numbers. 12: 24 seeds × 4 directions; 42: 4 seeds × 3
directions.

| size / role | ε (mean) | directional derivatives D (± SE) | \|∇J\|² (95% CI) |
|---|---|---|---|
| 12 / producer | 0.060 | 11.2 ± 10.3, 37.1 ± 15.7, 26.2 ± 25.3, 15.0 ± 14.0 | 8.4e6 (−2.5e6 … 5.7e7) |
| 12 / orderer | 5.64 | 0.30 ± 0.25, −0.09 ± 0.19, 0.04 ± 0.22, 0.32 ± 0.23 | ≈ 0 (−905 … 5,430) |
| 42 / producer | 0.063 | −21 ± 68, −2.4 ± 40, −17 ± 41 | ≈ 0 (−8.1e7 … 2.0e8) |
| 42 / orderer | 5.84 | −0.50 ± 0.37, 0.13 ± 0.71, 0.38 ± 0.45 | ≈ 0 |

**Detection power at the measured size** (12 / producer, the only positive point estimate; |∇J| ≈ 2,900).

| estimator | per-seed noise RMS | power at 12 / 48 / 96 / 192 / 384 seeds per half |
|---|---|---|
| E3 | 3.5e4 (signal / noise per seed ≈ 0.08) | 0.07 / 0.09 / 0.15 / 0.37 / **0.68** |
| E2 | 6.6e4 | 0.18 at 384 |
| E1 | 3.9e5 | 0.05 at 384 |

**Planted sweep** (12, smallest detectable signal as a multiple of the per-seed noise RMS, at 12 / 24 / 48 seeds
per half): E3 producer 0.63 / 0.40 / 0.32. The measured 0.08 would need roughly 700 seeds per half.

**Mechanism** (sibling-0 episodes, start policy; producer score split by term):

| size | producer gate term | producer replenish term | gate flip rate | orderer score (vs producer) |
|---|---|---|---|---|
| 12 | 0.6% | **99.4%** | 0.8% | 2.4e4 vs 7.2e8 |
| 42 | 0.3% | **99.7%** | 1.1% | 1.0e5 vs 2.4e9 |

## Diagnosis
- **Orderer: no measurable return sensitivity.** At a PPO-sized step (KL 0.01), random directions move the return
  by D ≈ 0 ± 0.25 per unit step. This matches the role swaps (EXP-0100: the GNN orderer vs MWWk, +5.7).
- **Producer: the signal may exist but is buried.**
  - The point estimate is positive at 12 (all four directional derivatives > 0), but its interval includes 0.
  - Its score noise is 99.4–99.7% the replenish term, the Gaussian exploration on replenish amounts
    (σ = 0.22 in logit space; the score scales as 1 / σ).
  - The gates, which carry the investment decisions that matter (DIAGNOSTICS 32–34), explore almost nothing:
    flip rate 0.8–1.1%, 0.3–0.6% of the score.
  - Per-seed signal / noise ≈ 0.08 for the best estimator → about 700 seeds per half per gradient estimate.
- **So `absent` names a mechanism, not "no gradient anywhere":** the start policy's exploration is spent on
  replenish noise, which dominates the score without (detectably) moving the return. The investment gates that do
  move it are near-deterministic.
- **Not fixable by more seeds at an affordable cost:**
  - one estimate at about 700 seeds per half ≈ 1,400 seeds × 4 siblings × 13 s ≈ 20 vCPU-h at 12 alone, every
    iteration;
  - or about 600 vCPU-h at 42 (D0.5: 110 s per episode).

## Costed options (strategy changes go to the user)
1. **Re-shape exploration, then re-test** (a new pre-registration v9: Stage 0 + this power check, ≈ 6 vCPU-h).
   - Make replenish deterministic in the RL action (it stays the imitated value), so the score holds only the
     gates and the orderer.
   - Put a temperature on the producer gates, so they flip about 5% per head.
   - D0.3 / power then show whether the gate signal is detectable at ≤ 96 seeds per half.
2. **Drop 42 from Stage 1 training** (keep it in the reads). It cuts the per-iteration cost about 8×. It does not
   fix the 12-producer signal.
3. **Stop RL; the imitated EXP-0100 stack is the M6 result.** Validated `better` at 12 (EXP-0102, EXP-0103
   held-out). The 42 drain (DIAGNOSTICS 32) stays a known limit.
