# EXP-0112: critic-free Stage 0 with reshaped exploration, plus the power check (pre-registration v9)

User decision 2026-11-06 (after EXP-0111 `absent`). `m6_prereg_v9.json` (SHA `660a7597…`), code 096be8c
(`train/m6_stage0b.py`). Pod svyc2sej5ixz0c (secure 16 vCPU / 32 GB + 20 GB volume, EUR-IS-1), 20:59–21:28 UTC ≈ **7.7
vCPU-h** (stop 10).
- **Policy (training rollouts only):** replenish frozen at the start's deterministic output; producer gates
  tempered (T = 3.452 / 2.563 / 2.400 for 5% offline greedy flips); orderer as v8.
- **Data:** the per-decision npz files (1.8 GB) stay local (`artifacts/local/`, uncommitted), pinned by
  `artifacts/npz_sha256.txt`.

## Result: **`absent` under the pre-registered rule**
No estimator's producer cosine at 12 beats its permutation null with the 12 seeds per half that Stage 0 has.
**The power check now finds a real producer gradient at 12**, detectable at **48 seeds per half with E3**.

**Exploration as realised (closed loop):**

| size | restore flips | upgrade flips | expand flips | producer share of the projected score² |
|---|---|---|---|---|
| 12 | 5.9% | 3.2% | 3.5% | **96.6%** |
| 42 | 5.9% | 2.2% | 3.7% | 54.9% |

EXP-0111 had about 1% flips, with 99.4% of the producer score in replenish noise.

**D0.1 / D0.2 / D0.5:**

| size | D0.1 | D0.2 t* (γ) | D0.5 s / episode (per-decision scoring) | deaths |
|---|---|---|---|---|
| 12 (96 episodes) | 0.980 | none (0.99) | 62.6 | 0 |
| 42 (32 episodes) | **0.329** | 3 (0.90) | **431.1** | **1** (42-0043, sibling 1: return −204) |

The 42 scenario share falls from 0.97 (v8) because one exploration-induced death dominates the within-seed
variance.

**D0.3 (cosine / null q95; none passes):**

| estimator | 12 producer | 12 orderer | 42 producer | 42 orderer |
|---|---|---|---|---|
| E1 | −0.392 / 0.421 | 0.107 / 0.579 | −0.272 / 0.336 | −0.671 / 0.754 |
| E2 | −0.060 / 0.380 | −0.798 / 0.619 | −0.293 / 0.387 | 0.124 / 0.603 |
| E3 | 0.038 / 0.377 | −0.532 / 0.613 | 0.263 / 0.393 | −0.311 / 0.504 |
| E4 | −0.122 / 0.558 | −0.244 / 0.697 | −0.305 / 0.536 | −0.318 / 0.579 |

**E4's held-out explained variance** (V fitted on one half, two halves):
- 12: producer 0.06 / 0.34, orderer 0.25 / 0.54;
- 42: producer 0.01 / −1.52, orderer 0.12 / −2.04 (the death row).

**True gradient** (paired ±ε, KL 0.01 per direction; 12: 24 seeds × 4 directions, 42: 4 × 3):

| size / role | ε | D per direction (± SE) | \|∇J\|² (95% CI) |
|---|---|---|---|
| 12 / producer | 1.23 | 2.02 ± 1.08, 2.62 ± 1.83, −4.97 ± 2.04, −4.46 ± 1.37 | **3.1e5 (5.7e4 … 8.7e5): above 0** |
| 12 / orderer | 5.64 | 0.03 ± 0.26, 0.43 ± 0.27, 0.07 ± 0.16, −0.19 ± 0.16 | 233 (−1.0e3 … 6.8e3) |
| 42 / producer | 1.26 | −0.07 ± 3.41, 2.46 ± 5.26, 2.39 ± 1.83 | ≈ 0 (−4.9e5 … 1.0e6) |
| 42 / orderer | 5.84 | −0.33 ± 0.65, −0.50 ± 0.26, 0.54 ± 0.41 | ≈ 0 |

**n80 at the measured |∇J|** (12 / producer; |∇J| ≈ 557):
- **E3 48 seeds per half** (per-seed noise RMS 1.9e3);
- E2 192;
- E4 > 384 (0.63 at 384);
- E1 > 384.

The orderer reaches no n80 (no gradient). At 42 nothing is detectable.

## Diagnosis
- **The reshaping worked on its target.**
  - The investment gates now explore at about 3–6% and carry 97% of the producer score at 12. Replenish noise is
    gone.
  - With that, the producer's return gradient at 12 is **measurably non-zero**: its 95% interval excludes 0, and
    a KL-0.01 random step moves the return by about 2–5 normalised units (± 1–2), about 2–5% of a base episode.
- **Why `absent` all the same:** Stage 0 has 12 seeds per half. E3 needs 48 for 80% power, so a miss at 12 per
  half was expected (E3's cosine 0.04 vs a null of 0.38). The pre-registered clause "beats its null at 12 per
  half" is stricter than the power analysis's "≤ 96 seeds".
- **E4 does not help:**
  - its value baseline explains little held-out variance (0.06–0.34 for the producer at 12, negative at 42);
  - its n80 > 384.
- **The orderer has no return gradient,** again.
- **At 42:**
  - nothing is detectable;
  - exploration caused one death in 32 episodes;
  - per-decision scoring costs 431 s per episode (Stage 1 needs per-cycle scores only; v8 measured 110 s).

## Conclusions
- **The reading is `absent` under v9's conjunctive rule.** The supervisor and the user decide whether the power
  evidence (E3, 48 seeds per half at 12) licenses a Stage 1 proposal.
- **The mechanism is diagnosed:** without gate exploration there was no usable signal (EXP-0111). With it, a
  producer signal exists at 12 and is weak (signal / noise per seed ≈ 0.3 for E3).
- **Costed outlook:** a Stage 1 gradient estimate at 48 seeds per half (96 seeds × 4 siblings at 12, per-cycle
  scoring ≈ 13 s / episode) costs about 1.4 vCPU-h per iteration at 12 producers alone. About 50 iterations fit
  the remaining tranche (≈ 68 vCPU-h). 42 stays reported, not trained.
