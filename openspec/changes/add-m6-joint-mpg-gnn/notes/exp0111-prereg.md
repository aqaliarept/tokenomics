# EXP-0111 pre-registration: power calibration of the Stage 0 gradient check (DIAGNOSTICS 35)

Frozen 2026-11-06 before any EXP-0111 computation. Pinned by SHA in `train/m6_power.py`.

**Inputs:**
- EXP-0110's rows (`stage0_rows.jsonl`, SHA `eb4bfa39…`), calibration, and start models;
- reading v8's γ per size.

**1. Noise model.**
- Per size, role and estimator (E1, E2, E3 as in v8), the per-seed estimate ĝ_seed = (1/G) Σ_i A_i · Σ_t a_t s_i,t
  (projected, k = 256).
- The noise is the seeds' centred estimates, resampled with replacement (residual bootstrap). This keeps their
  low-dimensional structure.

**2. Planted-signal sweep.**
- Per size, role and estimator, simulate D0.3 at n = 12, 24 and 48 seeds per half (2,000 draws).
- The null is simulated the same way with μ = 0; its 95th percentile is the bar.
- The signal is μ = m · u along random unit directions u.
- Report the smallest m (as a multiple of the per-seed noise RMS, √tr Σ̂) with detection ≥ 80%.

**3. True gradient size.**
- Per role and size, measure the directional derivative D_u = (R(θ + εu) − R(θ − εu)) / (2ε) of the normalised
  return J along K random unit directions u in that role's parameter space.
- Common random numbers: the same seeds, sibling 0's policy draws, and EXP-0110's reward weights and unit
  scales.
- 12: 24 seeds, K = 4. 42: 4 seeds, K = 3.
- ε is set so that the mean per-decision KL(π_θ ‖ π_θ+εu) on the monitor samples is 0.01, a PPO-sized step.
- Estimate: |∇J|² ≈ P · (mean_k D̄_k² − mean_k SE_k²), with a 95% interval from a bootstrap over seeds and
  directions.
- The improvement per update along the true gradient is ε · |∇J|.

**4. Seeds needed.**
- n80 is the smallest n ∈ {12, 24, 48, 96, 192, 384} seeds per half at which the noise model detects μ = |∇J| (a
  random direction) with ≥ 80% probability, for the best estimator.

**5. Mechanism.**
- In 4 seeds at 12 and 1 at 42 (sibling 0, start policy), split each cycle's producer score into the gate term
  (Bernoulli) and the replenish term (Gaussian).
- Report each term's share of the producer score variance, the orderer's score variance, and the gate flip
  rates.

**Reading (12 producers decide; 42 reported):**
- **`underpowered`:** for both roles at 12, the lower 95% bound of |∇J|² is > 0 and n80 ≤ 384. Report n80 and the
  Stage 1 cost it implies (D0.5).
- **`absent`:** otherwise. Name the mechanism from part 5 (e.g. near-deterministic producer gates carrying no
  usable score variance).
