# EXP-0129: EXP-0128's cell B from EXP-0100's start (EXP-0112 replicated on the new code)

**Setup.**
- `m6_prereg_v18` (SHA `4c4b1b1a…`, frozen before data). Code 15465ff (`train/m6_diag_start.py` on `m6_stage0c`).
- **Start:** EXP-0100's (EXP-0110 `start_models.pt`, EXP-0112's start).
- **Configuration:** full λ, the investment gates only (27,363 trainable parameters; EXP-0112: 27,524 incl. the
  frozen 161-weight replenish head). Replenish and the turns played greedily; with no turn heads, the turns play the
  default auction (ADR-0062: byte-identical).
- v16's 24 seeds × 4 siblings and ±ε along 4 directions at KL 0.01.
- **Runs:**
  - timing seed on pod 4rh3vwugm0kz67 (4 vCPU, 23:20–23:22 UTC ≈ 0.2 vCPU-h): scored sibling 11.5 s, perturbed
    episode 2.9 s, npz load 0.008 s per episode, ε search 16 s;
  - main run on pod w8pj2hdapthw29 (secure 8 vCPU, 23:23–23:33 UTC ≈ **1.3 vCPU-h**), stop 15 min = 1.5 × the timed
    estimate, a pulling watchdog;
  - total ≈ 1.5 vCPU-h, M6-A.
- **Data:** the score npz (1.1 GB) is kept untracked in `artifacts/local/`, with `npz_sha256.txt`.

## Reading: **`signal_present`**

| | EXP-0129 (new code, EXP-0100 start) | EXP-0112 (old code, same start) |
|---|---|---|
| producer \|∇J\|² (95% CI) | 2.1e5 **[4.9e4, 5.8e5]** | 3.1e5 [5.7e4, 8.7e5] |
| E3 n80 (seeds per half) | **96** | 48 |
| other estimators' n80 | E1 / E2 / E4: none ≤ 384 | E2 192 |
| v9 cosine reading (24 seeds) | absent (E1 0.24 vs q95 0.37) | absent |

Gate flips: restore 5.9%, upgrade 3.2%, expand 3.5%. No deaths in 96 episodes.

## Conclusions
- **The new on-policy code reproduces EXP-0112's gradient from EXP-0100's start.** The CIs overlap, and n80 is of
  the same order (96 vs 48). The code path (estimator, perturbation, ε, reward) is not the cause of 4.4's `absent`,
  so no bug hunt is needed.
- **Pre-registered consequence: F40's start is flatter.** But one input to that inference was lost.
  - EXP-0128's cell B (the same configuration from F40) survives only as its v9 cosine reading (`absent`). That is
    the same as this run's v9 reading.
  - B's gradient CI and n80, the criterion that separates the two, were lost with EXP-0128's pod.
  - So "flatter" is not yet shown on the same criterion. Re-running cell B under v18's criterion (≈ 2 vCPU-h)
    would settle it.
- **Where 4.4 stands, for comparison:** EXP-0127 (F40, all heads, λ / 4) had \|∇J\|² CI [−1.4e5, 2.1e6] (all heads)
  and [−1.2e4, 1.4e5] (replenish greedy).
