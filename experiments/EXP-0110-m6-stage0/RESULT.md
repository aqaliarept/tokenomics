# EXP-0110: M6 critic-free RL, Stage 0 diagnostic (pre-registration v8)

Supervisor decision 2026-11-06 (stop tuning imitation; Stage 0 from the EXP-0100 stack). Pre-registration
`m6_prereg_v8.json` (SHA `d55e757c…`), code 4b3ecb0 (`train/m6_stage0.py`). Pod wacrn1uzipxeww (secure 16 vCPU / 32 GB +
20 GB volume, EUR-IS-1), 18:40–18:51 UTC ≈ **2.9 vCPU-h** (stop 8). The start policy is EXP-0100's, plus zero-weight
v4 inputs; it is bit-identical to EXP-0100 (tested). The per-cycle rows (158 MB, `artifacts/local/`, uncommitted)
are pinned by `artifacts/stage0_rows_sha256.txt`.

## Calibration (k5_rf5 + MWWk, r1..r4, training split only)
| size | λ | μ | ν | world-episodes / episodes |
|---|---|---|---|---|
| 12 (h100) | 0.736 | 0.095 | 100 | 112 / 448 |
| 42 (h200) | 0.056 | 0.037 | 200 | 38 / 152 |

## Result: **`no_signal`**
No estimator's split-half gradient cosine beats its permutation null's 95th percentile, at either size or for
either role.

| size | D0.1 scenario share | D0.2 t* (γ) | D0.5 seconds / episode |
|---|---|---|---|
| 12 (24 seeds × 4 siblings, h100) | **0.979** | none within 100 cycles (0.99); the median sibling correlation of Φ_t falls 0.98 → 0.72 | 13.0 |
| 42 (8 seeds × 4, h200) | **0.970** | 5 (0.90); noisy, 0.37 at t = 5 then 0.6–0.96 | **109.6** (max 178) |

D0.3 (cosine / null q95; none passes):
| estimator | 12 producer | 12 orderer | 42 producer | 42 orderer |
|---|---|---|---|---|
| E1 (unpaired, episode) | 0.393 / 0.510 | 0.337 / 0.586 | −0.164 / 0.404 | 0.013 / 0.677 |
| E2 (same-seed LOO, episode) | −0.395 / 0.564 | −0.598 / 0.705 | 0.407 / 0.622 | 0.184 / 0.684 |
| E3 (same-seed LOO, per cycle) | −0.089 / 0.532 | −0.577 / 0.598 | 0.580 / 0.584 | 0.237 / 0.690 |

## What the evidence shows (descriptive; the cause of `no_signal` is not separated)
- **Pairing works:** same-seed siblings share 97–98% of the return variance. The leave-one-out baseline removes
  almost all of it.
- **Only a little within-seed variation is left, and mostly from the orderer.** Offline, on the monitor samples:
  - The start producer's gates almost never flip: expected flips 0.047 per producer decision; P(sample ≠ argmax)
    0.009 / 0.021 / 0.016 per head.
  - The orderer is far more random: entropy 0.91 nats, top probability 0.53.
  - At 12, the siblings' per-cycle rewards stay correlated (≥ 0.72) for all 100 cycles.
- **The permutation nulls are high** (95th percentiles 0.40–0.70).
  - The scores are zero-mean: |mean| / mean |x| ≈ 0.10–0.15 ≈ 1 / √n, and the median cosine to the mean is
    0.05–0.25.
  - So the high nulls come from a **low effective dimension** of the gradient noise. Roughly 1.645² / q95², that
    is 5–15 directions.
  - In one local 12-producer episode, the producer score's norm is 30% replenish head and 22% the global encoder.
- **With 12 seeds per half (4 at 42), a true gradient must agree at cosine above 0.5–0.7 to pass.** Whether the
  signal is absent or the test is underpowered is undiagnosed.
  - **Discriminating check:** a power calibration on these rows. Plant a known gradient of growing size into the
    recorded scores and rewards; find where D0.3 detects it at 12 / 24 / 48 seeds per half. Compare with the
    observed E1 cosines (0.39 / 0.34 at 12).
- **D0.5:** a 42-producer GNN episode with scoring costs about 110 s of worker time at h200. Stage 1's budget
  (30 seeds × 4 siblings × 150 iterations) would be about 30 vCPU-h at 42 per 10 seeds per iteration, before
  12-producer episodes and updates.

## Conclusions
- **Pre-registered reading `no_signal`:** stop. Stage 1 does not run. B (seeds) or R (siblings) change only by a
  further pre-registration.
- **The next move depends on the discriminating check:**
  - underpowered → more seeds per estimate;
  - absent → the start policy explores too little on the producer side (near-deterministic gates).
