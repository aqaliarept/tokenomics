# EXP-0084: rf4 (rf3 + never upgrade past the sustainable max) vs rf3 (M5 task 4.2g)

**Question.** Does the cap from DIAGNOSTICS 29 (the upgrade trap) help on M4's evaluation worlds? It blocks an
upgrade whose post-upgrade degradation would exceed one restoration. The test is EXP-0057's worlds (23 at 12
producers, 14 at 42), TTL 2, MWWk R = 2 and RR, observer on. rf3's rows come from EXP-0079; digests match for
213 / 213 rows.

**Setup.**
- Commit 6916722. `tokenomics.eval.supply_trace --producer entity_planner_buyahead_k4_rf4`, plus the
  reported-only horizon-200 dry runs (`h200/`).
- Pod: secure 16 vCPU in EUR-IS-1 (`91m4a0f6czzczj`), 19:44–19:48 UTC.
- Read: `read_exp0084.py` → `reading_exp0084.json`.

## Pre-registered reading (12 decides): **`no_gain`** (42 reported: `no_gain`)

| size | orderer | rf3 → rf4 units | mean diff vs floor | worlds beyond / below | deaths | run-down fails | upgrades blocked | upgrades completed | held units / cycle |
|---|---|---|---|---|---|---|---|---|---|
| 12 | MWWk | 3,448.5 → 3,573.0 | **+116.8** vs 146.5 | 3 / 1 | 0 / 0 | 71 → 66 | 2,863 | 8,968 → 7,420 | 257.6 → 248.7 |
| 12 | RR | 3,565.7 → 3,687.3 | **+113.7** vs 146.5 | 2 / 1 | 0 / 0 | 32 → 29 | 1,566 | 4,393 → 3,692 | 275.6 → 267.1 |
| 42 | MWWk | 8,843.9 → 9,169.8 | +325.9 vs 362.2 | 3 / 0 | 0 / 0 | 53 → 53 | 11,501 | 25,249 → 19,127 | 1,256 → 1,244 |
| 42 | RR | 8,620.5 → 8,987.9 | +367.4 vs 362.2 | 3 / 0 | 0 / 0 | 26 → 26 | 7,060 | 13,780 → 10,180 | 1,366 → 1,361 |

There were 0 contract failures.

- **Where the cap never fires, rf4 equals rf3** (the diff is exactly 0). The cap fires on 11 / 23 worlds at 12
  under MWWk and on 9 / 23 under RR.
- **The gains are concentrated:**
  - atlas 12-0121: +2,056 / +2,016;
  - 12-0352: +788 / +980;
  - 12-0321: +248 (MWWk only).
- **One world loses:** 12-0391, −402 / −482. There the blocked upgrades would have paid within 100 cycles; this
  is not diagnosed.
- rf3 dies on none of these worlds at horizon 100, and neither does rf4. The trap costs units here, not
  survival.

## Reported only: horizon 200, MWWk, the 23 12-producer worlds (86 episodes)

| | units | deaths |
|---|---|---|
| rf3 | 7,145.4 | 0 |
| rf4 | 7,741.6 (**+596**, +8.3%) | 0 |

The cap's gain grows with the horizon: +3.6% (MWWk) at 100 cycles, +8.3% at 200. This fits a drain that bites late.

## Conclusions
- **`no_gain` by the registered rule.** rf4 is about +3.4–3.6% at 12 producers with both orderers, with no deaths
  and fewer run-down failures. That is below the noise floor's mean (146.5) because the gain sits on 2–3 worlds
  and 12-0391 loses.
- **Whether rf4 becomes the base**, and the redo of L1 / the probe, is the supervisor's / user's decision.
  - For: no deaths, never worse in units where the cap does not fire, larger gains at longer horizons, and the
    training-set deaths of EXP-0081 (DIAGNOSTICS 29) removed.
  - Against: the registered evaluation reads `no_gain`.
- **Spend:** ≈ 0.9 vCPU-h billed (stop 6). M5-P ≈ 3.6 / 80.
