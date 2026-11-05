# EXP-0090: M5 L1 on the final producer k5_rf5 (task R-2)

**Question.** L1 (bar plus noise floor) on `entity_planner_buyahead_k5_rf5` with MWWk, on M4's pinned training
set (128 world-episodes at 12 producers, horizon 100), as EXP-0085. It supplies v6's bar, normaliser, margins
and flag threshold. Not an acceptance read.

**Setup.**
- Commit 674d10d. `producer_es --mode bar`, then `--mode noise-floor`.
- Cross-check against EXP-0088's k5_rf5 rows.
- Pod: secure 16 vCPU in **EUR-IS-1** (`4krsd6lgs828tu`), 21:49–21:52 UTC.

## Results

| check | k5_rf5 (EXP-0090) | rf4 (EXP-0085) | rf3 (EXP-0080) |
|---|---|---|---|
| r0 reproduces the bar | **128 / 128** | 128 / 128 | 128 / 128 |
| cross-check against the base's reference rows | **2 / 2** vs EXP-0088 (137 columns) | 2 / 2 | 2 / 2 |
| bar mean consumer units | **3,524.7** | 3,418.2 | 3,196.5 |
| training-set deaths (bar r0; noise r0..r4) | **0 / 0** | 0 / 0 | 1 / 1 |
| run-down fails / starved (bar, of 128) | 104 / 98 | 106 / 102 | 110 / 102 |
| margin run-down / starvation | 0.0939 / 0.1200 | 0.0952 / 0.1150 | 0.0966 / 0.1180 |
| per-episode self-flags (reported only) | 21 / 512 = 4.10% | 5.47% | 4.69% |
| **flag-count calibration (L1's criterion, ≤ 5%)** | **T = 5/3; 11 / 328 = 3.35%: pass** | 4.27% | 2.74% |
| contract failures | 0 | 0 | 0 |

## Conclusions
- **L1 passes.**
- **Pre-registration v6 is frozen from these files:** SHA `5cd0e302…`, T = 5/3. The sources are
  `noise_episodes.csv` `a7d59888…`, `margins.json` `79c2ddae…` and `bar.csv` `de67a546…`.
- The probe, EXP-0086, follows under v6.
- **Spend:** ≈ 0.9 vCPU-h billed (stop 3). M5-P ≈ 7.0 / 80.
