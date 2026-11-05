# EXP-0088: buffer re-sweep on rf5 (producer finalization PF-5)

**Question.** With the exact upgrade cap (rf5), which input buffer k (cycles of the producer's own recent input
use kept in stock) is best?
- k ∈ {3, 5, 6} were played here; k = 4 is rf5, from EXP-0089's rows.
- TTL 2, MWWk R = 2 and RR, EXP-0057's worlds, observer on.
- The k* rule was pre-registered in tasks.md PF-5, as EXP-0077's, with rf5 as the reference.

**Setup.**
- Commit b05267a. `supply_trace --producer entity_planner_buyahead_k{3,5,6}_rf5`.
- Pod: secure 8 vCPU in **EUR-IS-1** (`zddpeuez441r8d`), 21:31–21:40 UTC. EUR-IS-1 refused 16 and 8 vCPU at
  21:29 and gave 8 vCPU on the first retry; no other data center was needed.
- Digests equal rf5's for 213 / 213 rows per k. 0 contract failures.
- Read: `read_exp0088.py` → `reading_exp0088.json`.

## Reading: **k* = 5** at 12 producers (both orderers); 42 reported: k* = 4

| 12 producers | k = 3 | k = 4 (rf5) | k = 5 | k = 6 |
|---|---|---|---|---|
| MWWk units / episode (vs rf5) | 3,443.9 (−144.6) | 3,592.8 | 3,703.2 (**+106.4**) | 3,761.7 (+161.8) |
| RR units / episode (vs rf5) | 3,576.0 (−153.7) | 3,731.7 | 3,841.9 (**+106.1**) | 3,896.3 (+158.9) |
| best − k, MWWk / RR (floor 146.5) | 306 / — (k 3 not eligible, RR run-down +6) | 162 / 159 | **55 / 53** (within) | 0 |
| deaths | 0 | 0 | 0 | 0 |
| run-down fails, MWWk / RR | 61 / 35 | 66 / 29 | 67 / 32 | 66 / 32 |
| held units / cycle, MWWk | 219 | 250 | 274 | 295 |
| late inputs (units), MWWk | 156,957 | 155,294 | 153,712 | 152,948 |
| input died upstream (units), MWWk | 78,320 | 70,569 | 63,517 | 60,928 |
| contention (units), MWWk | 69,346 | 70,591 | 71,719 | 70,728 |

- **The rule's steps:**
  - The best eligible k is 6, the edge of the range.
  - k = 5 is within the floor of k = 6 (55 / 53 < 146.5) and k = 4 is not (162 / 159), so k*_o = 5 for both
    orderers.
- **The gains diminish:** +145 (3 → 4), +106 (4 → 5), +55 (5 → 6) with MWWk. Going past 6 would add less than
  the floor per step.
- **At 42 producers (reported):** k = 4 is within the floor of k = 6 (+280 / +248 vs a floor of 362), so
  k* = 4.

## What the buffer changes
- **The gain comes almost entirely from fewer input orders dying upstream:** 78k → 61k from k = 3 to 6 (MWWk).
  Suppliers hold more stock, so their own reservations wait less.
- **Late inputs at the assembler barely move:** 157k → 153k, −2.6%. Contention does not move either.
- The buffer costs held stock (+18% from k = 4 to k = 6) and slightly more investment rejections (0.098 →
  0.105).

## Conclusions
- **Final buffer for the producer: k* = 5, i.e. `entity_planner_buyahead_k5_rf5`.** It is +106 per episode over
  rf5 at 12 (both orderers), with no deaths and run-down within +3.
- **The largest remaining class, late inputs (37% of losses), does not respond to the buffer size.**
- **Spend:** ≈ 1.2 vCPU-h billed (stop 4). M5-P ≈ 6.1 / 80.
