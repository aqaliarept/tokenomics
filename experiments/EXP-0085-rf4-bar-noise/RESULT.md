# EXP-0085: M5 L1 on rf4, bar and noise floor (task 4.2i)

**Question.** The same L1 as EXP-0080, on the new base `entity_planner_buyahead_k4_rf4` (rf4) with MWWk, on M4's
pinned training set (128 world-episodes at 12 producers, horizon 100). Not an acceptance read.

**Setup.**
- Commit 875c594, `producer_es --mode bar`, then `--mode noise-floor`.
- Cross-check against EXP-0084's rf4 rows.
- Pod: secure 16 vCPU in EUR-IS-1 (`t70u0084cnq136`), 20:21–20:23 UTC. Its creation was retried for about
  14 minutes: there was no capacity in EUR-IS-1 at 16 or at 8 vCPU.

## Results

| check | rf4 (EXP-0085) | rf3 (EXP-0080), for reference |
|---|---|---|
| r0 reproduces the bar | **128 / 128** | 128 / 128 |
| cross-check against the base's reference rows | **2 / 2** equal vs EXP-0084 (137 columns) | 2 / 2 vs EXP-0079 |
| bar mean consumer units | **3,418.2** | 3,196.5 |
| training-set deaths (bar r0; noise floor r0..r4) | **0 and 0** | 1 (12-0022 r0) and 1 |
| 12-0016 episodes, units | 3,568 / 3,484, alive | 1,669 / 1,886, alive |
| 12-0004 episodes, units | 4,490 / 4,745, alive | 3,282 / 3,595, alive |
| run-down check fails / starved (bar) | 106 / 102 of 128 | 110 / 102 |
| margin (c = 2) run-down / starvation | 0.0952 / 0.1150 | 0.0966 / 0.1180 |
| **self-flags under the margin (per episode)** | **28 / 512 = 5.47% > 5%: not met** | 24 / 512 = 4.69% |
| flag-count calibration (`flag_calibration.json`) | T = 5/3; rf4 self-disqualified 14 / 328 = 4.27% ≤ 5% | T = 5/3; 9 / 328 = 2.74% |
| contract failures | 0 | 0 |

- All 28 self-flags are starvation flags, on 13 world-episodes.
- The most frequent are knife-edge 12-0007 (5), 12-0013 (4), 12-0044 (4), 12-0302 (3) and 12-0050 (3), the
  worlds DIAGNOSTICS 25 already names.

## Reading: **L1 stop branch**
- Pre-registered L1 (v1–v4, and v5's draft text) requires "rf4's self-flag rate under the margin ≤ 0.05; else
  stop, the user". It is 5.47%.
- Every other L1 check passes. rf4 also passes v5's own flag-count target: 4.27% of re-keyed minibatches,
  against ≤ 5%.
- Since v3, the per-episode rate no longer decides any disqualification; the minibatch flag-count rule does.
  The criterion stays registered, though, so **v5 is not frozen** and EXP-0086 does not launch. The criterion
  goes to the supervisor and the user.
- **Spend:** ≈ 0.6 vCPU-h billed (stop 3). M5-P ≈ 4.2 / 80.
