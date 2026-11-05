# EXP-0087: rf4's consumer-loss classes (producer finalization, step 1)

**Question.** After rf4, where are consumer units still lost on the producer side?
- Worlds: EXP-0057's, 23 at 12 producers and 14 at 42.
- TTL 2, MWWk R = 2 and RR.
- Compared with EXP-0074 (`entity_planner`, MWWk), EXP-0077 (k4) and EXP-0079 (rf3).

**Setup.** No pod. EXP-0084 already ran the identical command, `supply_trace --producer
entity_planner_buyahead_k4_rf4 --order-ttl 2`, with the observer on. `read_exp0087.py` reads its
`supply_trace.csv` and the three references, each pinned by SHA-256 in `reading_exp0087.json`. Not an
acceptance read.

## 12 producers, consumer units lost per episode (MWWk / RR)

| class | entity_planner | k4 | rf3 | rf4 | rf4 share (MWWk) |
|---|---|---|---|---|---|
| delivered units | 2,096 | 3,101 / 3,200 | 3,449 / 3,566 | **3,573 / 3,687** | |
| lost units | 6,212 | 5,228 / 5,132 | 4,882 / 4,771 | **4,767 / 4,655** | 100% |
| b2 input arrived late (after the order's TTL) | 2,022 | 2,071 / 2,103 | 1,785 / 1,850 | 1,765 / 1,812 | **37%** |
| b2 input order died upstream (supplier's reservation TTL) | 1,808 | 893 / 915 | 805 / 836 | 842 / 840 | 18% |
| b4 contention (inputs present, taken by another reservation) | 635 | 786 / 816 | 808 / 823 | 840 / 805 | 18% |
| c outbid at the auction | 546 | 476 / 333 | 499 / 321 | 471 / 268 | 10% |
| b2 input displaced upstream | 403 | 341 / 339 | 414 / 390 | 437 / 424 | 9% |
| b2 input no-fit upstream | 195 | 148 / 192 | 165 / 183 | 147 / 231 | 3% |
| a capacity (held / degraded / max / dropped) | 324 | 286 / 213 | 259 / 219 | **118 / 130** | 2.5% |
| b2 short / unfunded; b1 not ordered; d outage | 42 / 48; 33; 34 | ≈ 50 / 45; 35; 40 | ≈ 15 / 45; 6; 30 | ≈ 14 / 36; 6; 35 / 16 | ≈ 2% |

At 42 producers (reported only) the order is the same:
- rf4 MWWk loses 9,628 per episode;
- late inputs 3,253 (34%), died upstream 2,138 (22%), contention 989 (10%), displaced 1,019 (11%);
- capacity 569 (6%, from 802 under rf3).

## Conclusions
- **rf4 fixed what it targeted.** The capacity class roughly halves: 259 → 118 at 12, 802 → 569 at 42. Every
  other class is about unchanged against rf3.
- **The largest remaining class is still timing:** an input that arrives after the consumer order's TTL (37% of
  losses at 12). Next come input orders that die upstream because the supplier's own reservation expired
  (18%), and contention between reservations for inputs already in stock (18%).
- Together the input-timing classes (late + died upstream + displaced) are about 64% of the remaining loss.
- 0 vCPU-h.
