# EXP-0130: EXP-0128's cell B from F40 under v18's criterion

**Reading: `absent`** (attempt 2; attempt 1 was hard-stopped with no data).

**Setup.**
- `m6_prereg_v19` (SHA `7e223cd2…`, frozen before data). Code cd70e85 (`train/m6_diag_start.py --variant f40`).
- **Start:** the F40 + v15 composite (EXP-0126 `models.pt`), the EXP-0110 start's replenish head.
- **Configuration:** the same as EXP-0129: full λ, the investment gates only, replenish and the turns greedy and frozen;
  v16's 24 seeds × 4 siblings and ±ε along 4 directions at KL 0.01.

**Runs (2026-11-08 UTC).**
- Failed launch, 2026-11-07 23:35: no pod was created.
  - The call was correct, and the same call worked on retry.
  - `launch_exp.sh` exited in about a second, leaving no worktree and no launch log. `worktree add` is the only step
    that could leave that state, but its error went to `/dev/null`, so the cause is not known.
  - The watchdog then started with an empty pod id, read its arguments shifted by one, and exited harmlessly.
  - Both scripts now report these errors.
- Timing seed on pod r0t21w9qi666sb (4 vCPU, 12:49–12:52 ≈ 0.2 vCPU-h): scored sibling 10.2 s, perturbed episode
  4.0 s, npz load 0.007 s per episode, ε search 14.9 s.
- Main run on pod 9ygj1v9u4l1nc2 (secure 8 vCPU, 12:52:34–13:08:54 ≈ 2.2 vCPU-h): stop 16 min = 1.5 × about
  10.5 min, which is EXP-0129's real wall clock scaled by the timed cost.
- Attempt 2 on pod sz35zqceadm53v (secure 8 vCPU, 13:12:00–13:23:44 ≈ 1.6 vCPU-h): setup 54 s, job 13:13:12–13:22:51
  (9.7 min). Under the supervisor's rule the abort was 5 min of setup and the stop 16 min after job start; neither fired.
- Total ≈ 4.0 vCPU-h, M6-A (limit raised by the supervisor to 5.5).

**Why attempt 1 produced nothing.**
- The run directory on the pod was created at 13:06:30, about 14 min after the pod was created. The timing pod needed
  under 1.5 min for the same setup.
- The job then had 2 min 15 s before the 13:08:45 hard stop, and had not finished calibration. The watchdog pulled an
  empty run directory.
- The launch log has no timestamps inside setup, so the 14 min cannot be split between its steps (sync, venv, install).
- The stop was measured from pod creation, and its margin assumed EXP-0129's normal setup time.

## Reading: **`absent`**

| | EXP-0130 (F40 start) | EXP-0129 (EXP-0100 start, same configuration) |
|---|---|---|
| producer \|∇J\|² | 4.0e4 | 2.1e5 |
| 95% CI | **[2.2e1, 2.0e5]** | [4.9e4, 5.8e5] |
| E3 n80 (seeds per half) | **384** | 96 |
| other estimators' n80 | E1 / E2 / E4: none ≤ 384 | E1 / E2 / E4: none ≤ 384 |
| v9 cosine reading (24 seeds) | absent | absent |

Gate flips: restore 5.8%, upgrade 2.7%, expand 2.6%. No deaths. 288 / 288 jobs. The ε values equal EXP-0128 cell B's.

## Conclusions
- **F40's start is flatter on the pre-registered criterion.** With the code path ruled out by EXP-0129, `absent` here
  reads as F40's start having the weaker gradient for the investment gates.
- **The criterion fails on n80, not on the CI.** The CI's lower bound is above 0, barely (22). The two runs' CIs overlap,
  so the 5× lower point estimate is suggestive; the separating number is E3's n80, 384 against 96.
- n80 is the smallest seed count in the grid (12 to 384, doubling) with power ≥ 0.8 at the measured \|∇J\|. E3's power
  here is 0.36 at 96, 0.66 at 192 and 0.88 at 384, the top of the grid. Only E3 reaches 0.8 at all.
- Consequence: the turn heads and λ / 4 are not needed to explain 4.4's `absent` (EXP-0127), since gates only at full λ
  is already `absent` from F40. This run does not test whether they also contribute.

**Data.** `timing.json`, `calibration.json`, `episode_rows.jsonl`, `eps_rows.jsonl`, `reading.json`,
`reading_exp0130.json`, `launch_attempt1.log`, `launch.log`, `launch_timing.log`. The score npz (1.1 GB, and the timing
seed's) are untracked in `artifacts/local/`, hashes in `npz_sha256.txt` and `npz_sha256_timing_seed.txt`.
