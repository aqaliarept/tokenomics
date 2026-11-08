# EXP-0131: the final held-out read of M6's result, F40 + v15 turn heads

**Reading: `better` at 12 producers (deciding); 42 (reported) `better`.** F40 + default turns also reads `better` at both.

User decision 2026-11-08: the RL track is closed, and M6's result is F40 + v15 heads. `m6_prereg_v20` (SHA `8fe01a0c…`,
frozen before data). Ledger entry `held-out/m6-f40-v15`, pending before launch, `done` after.

**Setup.**
- **Subject:** F40's trunk, gates and v15 turn heads (EXP-0126 `models.pt`, SHA `c62a3ce8…`), the EXP-0110 start's
  replenish head (SHA `697d94c0…`), the frozen orderer. It plays only through `m6_onpolicy.deployed_policies`
  (DIAGNOSTICS 37).
- **Rule:** EXP-0102's `parity` / `better` / `short` with the side-effect clauses, against k5_rf5 + MWWk, on the loaded
  held-out worlds (8 at 12, 2 at 42; all 16 played), 8 episodes × replicates 0 / 1, horizon 100.

**Held-out re-use (stated in the pre-registration).**
- The 16 worlds were played before, by EXP-0103 and by EXP-0104's diagnosis of 42-0125, at episodes of stage
  `m5-heldout`. This is a second comparison on this set, not a read of unseen worlds.
- This read uses 8 episodes of a new stage, `m6-final-read`. None of its 128 seeds appears in the text records under
  `experiments/`, and only EXP-0103 and EXP-0104 name a held-out world among EXP-0100 to EXP-0130.
- A local dry run played 8 cycles of the first episode of an earlier stage name (`m6-final-heldout`, world 12-0016)
  before the freeze. That stage is abandoned, and dry runs now play train worlds only.
- 42-0125's diagnosis informed later producer versions, so the 42-producer read is not independent of this set.

**Runs (2026-11-08 UTC).**
- Timing pod y2siqxxkmf928o (4 vCPU, 14:53–14:56 ≈ 0.15 vCPU-h), on train worlds 12-0352 and 42-0081.
- Attempt 1, pod 3ftfzrf31bu6ad (16 vCPU, 14:57–15:07 ≈ 2.9 vCPU-h, commit f1f8507): hard-stopped 577 s after job
  start with 384 / 768 jobs done. 42-producer jobs took 2.6× the timing (train 42-0081 was not representative of the
  held-out 42 worlds); 12-producer jobs matched. Rows were pulled, and no units were read.
- **Resume** (supervisor decision), pod 3jrb6qp18yjr0n (8 vCPU, no 16 vCPU capacity; 15:11–15:17 ≈ 0.7 vCPU-h, setup 72 s,
  stop 390 s): the 384 missing jobs on the same seeds, with `--resume-rows` / `--resume-log`.
  - Driver change: commit 1236377. It adds resume arguments and a merge of the rows, and does not touch job
    construction, seeds or policies. A pinned fingerprint of `_build_jobs` + `final_job` + the stage must match
    attempt 1's.
  - Before playing, the resume verified attempt 1's commit (launch log), the pre-registration SHA, unique job keys all
    belonging to the read, the horizon and the digests, reading identity fields only.
  - Final set: 768 rows, 768 distinct job keys, paired demand and shock digests across the arms. Rows SHA recorded in
    `reading.json`.
- Total ≈ 3.8 vCPU-h, M6-A.

## Reading

| size | arm | mean units (gating worlds) | − base | floor | deaths | run-down share | starvation share | holding / cycle |
|---|---|---|---|---|---|---|---|---|
| 12 | **F40 + v15 heads (deciding)** | 3,937.6 | **+608.4 (+18.3%)** | 83.7 | 0 | 0.453 | 0.750 | 375.1 |
| 12 | F40 + default turns | 3,887.8 | +558.5 | 83.0 | 0 | 0.469 | 0.750 | 377.5 |
| 12 | k5_rf5 + MWWk (base) | 3,329.3 | — | — | 1 | 0.859 | 0.758 | 276.8 |
| 42 | F40 + v15 heads | 17,184.3 | +2,101.4 (+13.9%) | 269.8 | 0 | 0.000 | 1.000 | 2,295.9 |
| 42 | F40 + default turns | 16,526.8 | +1,443.8 | 203.8 | 0 | 0.000 | 1.000 | 2,481.4 |
| 42 | base | 15,082.9 | — | — | 0 | 0.500 | 1.000 | 2,375.4 |

**Per world, v15 heads − base (gating worlds).**
- 12: 0033 +2,111; 0278 +1,341; 0273 +744; 0368 +275; 0239 +218; 0172 +182; 0016 −2; 0413 −4. Ahead on 6 of 8.
- 42: 0125 +3,422; 0005 +781.

**The turn heads' effect** (v15 heads minus default turns, a point estimate with no floor of its own): +49.9 units at 12,
+657.6 at 42. At 42 it is uneven: 0125 +1,469, 0005 −153.

**Reported measures.** Investment rejection 0.137 vs base 0.215 at 12, 0.206 vs 0.182 at 42. Min capacity ratio 0.693 vs
0.539 at 12, 0.772 vs 0.697 at 42. Bids past the accept turn's slots (played in default order) occur only at 42: 141 in
total with the heads, 4,271 with the default turns.

## Conclusions
- **M6's result clears the user's goal on held-out worlds at 12 producers:** it completes 18.3% more consumer units than
  the best hand-written stack, with fewer deaths and a lower run-down and starvation share. Every clause holds.
- **The gain is concentrated.** Three of eight worlds carry most of it; without 0033 the mean over the other seven is
  about +393, still well above the floor. Two worlds are level.
- **The turn heads add little at 12** (+49.9 against +558.5 for F40 alone). Most of the gain is F40's trained gates.
- **42 reads `better` and the run-down share drops to 0, against EXP-0103's 1.000 for the imitated stack.** It remains
  reported only, and not independent of this set (above).
- **Holding is 1.36× the base at 12**, under EXP-0102's rule a reported measure, not a clause.
- The base died once at 12 (0033), so that world's mean includes one short base episode.
- Roster, registry and champion files were not changed; they are the user's gates.

**Data.** `reading.json`, `episode_rows.jsonl` (768 rows, merged), `episode_rows_resume.jsonl`, `run/episode_rows.jsonl`
(attempt 1's 384), `timing.json`, the launch and watchdog logs.
