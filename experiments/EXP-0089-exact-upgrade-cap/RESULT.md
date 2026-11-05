# EXP-0089: rf5 (rf3 + the exact sustainable-upgrade cap) vs rf3 (producer finalization PF-4)

**Question.** Does rf5 help on EXP-0057's worlds? rf5 is rf4's cap with the producer's own degradation rate,
not the view estimate that erred on 12-0391 (DIAGNOSTICS 29).
- Worlds: 23 at 12 producers, 14 at 42. TTL 2, MWWk R = 2 and RR, observer on.
- rf3's rows come from EXP-0079 and rf4's from EXP-0084. Digests match for 213 / 213 rows each.

**Setup.**
- Commit 0bbc939. `supply_trace --producer entity_planner_buyahead_k4_rf5`, plus a reported-only horizon-200
  dry run (`h200/`).
- Pods: secure 8 vCPU in EUR-IS-1 (16 was unavailable).
  - Attempt 1 (`ue32ajv4vbi8fn`, 21:21–21:23): a transient pip network read error during the venv install, so
    nothing ran; the pod was removed.
  - Attempt 2 (`wzd7ompcmzwnnv`, 21:23–21:27): complete.
- Read: `read_exp0089.py` → `reading_exp0089.json`.

## Pre-registered reading (12 decides): **`no_gain`** (42 reported: `no_gain`)

| size | orderer | rf3 → rf5 units | mean diff vs floor | worlds beyond / below | deaths | run-down fails | upgrades blocked | rf4 (EXP-0084), for reference |
|---|---|---|---|---|---|---|---|---|
| 12 | MWWk | 3,448.5 → 3,592.8 | **+135.3** vs 146.5 | 4 / 0 | 0 / 0 | 71 → 66 | 2,725 | +116.8, 3 / 1 |
| 12 | RR | 3,565.7 → 3,731.7 | **+155.1** vs 146.5 | 3 / 0 | 0 / 0 | 32 → 29 | 1,227 | +113.7, 2 / 1 |
| 42 | MWWk | 8,843.9 → 9,184.8 | +340.9 vs 362.2 | 3 / 0 | 0 / 0 | 53 → 52 | 10,315 | +325.9 |
| 42 | RR | 8,620.5 → 9,014.4 | +393.9 vs 362.2 | 2 / 0 | 0 / 0 | 26 → 26 | 5,734 | +367.4 |

There were 0 contract failures.

- **rf5 clears the floor with RR (+155.1) but not with MWWk (+135.3, 11 short).** The rule needs both, so the
  reading is `no_gain`.
- **rf5 is never below rf3 on any world** (0 worlds below at either size). 12-0391's loss under rf4 is gone.
- **rf5 vs rf4 (reported):** +18.5 (MWWk) / +44.4 (RR) at 12, with fewer upgrades blocked (2,725 vs 2,863;
  1,227 vs 1,566), all of them correct ones.
- **Horizon 200** (reported, MWWk, 12, 86 episodes): rf3 7,145 / rf4 7,742 / **rf5 7,795** (+649, +9.1% over
  rf3). No deaths for any.
- **Multi-type producers:** none of the 37 EXP-0057 worlds has a producer with more than one capacity type.
  The cap's "any row over" rule (inherited from rf4; supervisor's note) is therefore untested here.

## Conclusions
- **`no_gain` by the registered rule, though it nearly clears.** rf5 is about +4% at 12 producers with both
  orderers, with no deaths, fewer run-down failures and no world below the floor. It holds, and it beats rf4.
- **PF-5 (EXP-0088, the buffer re-sweep) runs on rf5,** per the supervisor's rule ("rf5 if it helps or holds").
- **Spend:** ≈ 0.7 vCPU-h billed (both attempts; stop 6). M5-P ≈ 4.9 / 80.
