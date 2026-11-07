# EXP-0127: the 4.4 power check for on-policy RL of every producer head (F40 + v15)

**Setup.**
- `m6_prereg_v16` (SHA `6ad4e49f…`, frozen before data). Code cb2fe36 (`train/m6_stage0c.py`, `train/m6_onpolicy.py`).
- **Start:** the composite: F40's trunk, gates and v15 turn heads + the start's replenish head on its frozen trunk.
- **Reward:** the 4.5 reward (λ / 4).
- 24 seeds at 12 producers × 4 scored siblings; ±ε along 4 directions of the 112,489 trainable parameters, twice:
  with replenish noise, and for the "replenish greedy" counterfactual (replenish slice zeroed, σ_r = 0).
- **Runs:**
  - timing pod lpjx5nv04kvnxp (8 vCPU, 21:07–21:12 UTC ≈ 0.6 vCPU-h). The 4.5 timing failed there with "Too many
    open files"; fixed in cb2fe36.
  - main pod kkvree7wl1fo6a (secure 16 vCPU + 20 GB volume, EUR-IS-1, 21:36–21:58 UTC ≈ **5.9 vCPU-h**, stop 8.8,
    M6-A).
- **Data:** the per-decision score npz (1.6 GB) is kept untracked in `artifacts/local/`; its SHA-256 list is
  `npz_sha256.txt`.

**Calibration** (start decisions, fixed draws): σ_r 0.423, τ_f 2.74, τ_p 0.73, T_a 3.63. The targets were hit
exactly.

In the rollouts:
- gate flips: restore 5.6%, upgrade 2.7%, expand 2.5%;
- fund swaps 5.0%, order swaps 3.8%, bit flips 5.3%;
- 0 deaths in 96 scored episodes.

## Reading: **`absent`**, and `absent` in the counterfactual too

| | primary (replenish noise σ_r 0.42) | counterfactual (replenish greedy) |
|---|---|---|
| \|∇J\|² (95% CI) | 3.4e5 [−1.4e5, 2.1e6] | 2.6e4 [−1.2e4, 1.4e5] |
| n80 (seeds per half): E1 / E2 / E3 / E4 | – / – / **192** / – | – / – / – / – (none ≤ 384) |
| D0.3 producer cosine vs null q95 | E1 0.61 vs 0.68; E2 −0.38; E3 −0.33; E4 −0.11 | – |
| reading | `absent` | `absent` |

**The head families' share of the squared producer score:**

| family | share |
|---|---|
| replenish | **81.4%** |
| accept order | 9.7% |
| accept bits | 6.4% |
| gates | 2.5% |
| fund order | 0.02% |

Replenish swamps the signal (> 0.5, the pre-registered line).

## Conclusions
- **4.5 is not licensed under v9's rule.**
  - With replenish noise, the replenish score swamps (81%), and E3 would need 192 seeds per half (2× the line).
  - With replenish greedy (exactly frozen), the remaining gradient is not resolvable at ≤ 384 seeds per half.
  - Its |∇J|² point estimate (2.6e4) is below EXP-0112's investment-only gradient (CI 5.7e4 … 8.7e5, n80 48).
- **Not diagnosed here (candidates):**
  - The λ / 4 re-weighting removed most of the gradient's run-down component. EXP-0116 attributed 72% of the
    training-return gain to run-down relief.
  - The extra turn-head parameters spread the score: noise grows, signal does not.
- **The 4.5 sizing (firm, from 32 real seeds × 4 siblings on 16 workers):**
  - rollouts 110.8 s for 32 seeds, so 332 s for 96;
  - PPO 1.70 s per step × 100 steps = 170 s;
  - **≈ 8.4 min and ≈ 2.2 vCPU-h per iteration** on 16 vCPU.
