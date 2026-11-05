# EXP-0079: "Expand inventory earlier" alone on buy-ahead k = 4

**Status: complete, valid.** M5 task 1.2 (`add-m5-potential-game-gnn`, design D8 pre-step). Not an
acceptance read.

## In plain English

**The question.** EXP-0078 found that inventory room is the binding resource, and that the only
helpful part of "reservations first" was asking for more inventory space earlier. That part was
never tested alone. `entity_planner_buyahead_k4_rf3` is buy-ahead k = 4 with exactly one change:
it requests an inventory expansion whenever its free room is below what its waiting orders still
need, not only when the room is completely full. Its buying is k = 4's exactly.

**Validity.** All 213 rows have the same demand and shock schedules as EXP-0077's k = 4 rows.
0 contract failures. The reading was pre-registered (M5 tasks.md 1.2) before the run.

**What happened (12 producers decide; consumer units per episode).**

| orderer | k = 4 | rf3 | difference | floor | worlds ahead beyond floor / below |
|---|---|---|---|---|---|
| MWWk | 3,101 | 3,449 | +326 (+11%) | 146.5 | 8 / 0 |
| round-robin | 3,200 | 3,566 | +345 (+11%) | 146.5 | 9 / 0 |

- **The registered reading is `rf_helps`.** It beats k = 4 beyond the noise floor with both
  orderers, is ahead beyond the floor on 8 / 9 of 23 worlds, and is below the floor on none.
  There are no deaths, and run-down failures change by 0 / −1.
- **At 42 producers (reported):** `rf_helps`, +868 / +940 (+11% / +12%) against a floor of 362, with
  no deaths and run-down +2 / 0.
- **Where it helps (12 MWWk):** late inputs fall 178k → 153k (−14%). Input orders dying upstream
  also fall, 77k → 69k, and capacity losses fall slightly.
- **The cost:** held stock rises 233 → 258 units per cycle. The investment rejection rate rises
  0.078 → 0.099, as more expansion orders compete for the fund: 22,246 requested, 12,882
  completed, 9,364 rejected, across all rows.

**What it means.**
- **Adding inventory room earlier is a clean producer improvement.** About +11% on top of buy-ahead
  k = 4, with no side effects beyond the margin.
- **Re-allocating the room did not help** (EXP-0078); getting more of it does.
- **Plan consequence:** `entity_planner_buyahead_k4_rf3` is the strongest known producer. It
  becomes the base of the M5 learning ladder (design D8), so the learner must now beat rf3 + MWW.

## Details
- **Driver:** `supply_trace --order-ttl 2 --producer entity_planner_buyahead_k4_rf3`, at commit
  `0352821` from a clean worktree. k = 4 rows reused from EXP-0077.
- **Inputs:** atlas v2 `7517314f…`, EXP-0012 `d25388bf…` (pinned only).
- **Reading** (`read_exp0079.py`, EXP-0078's rules on rf3):

  | cell | diff | mean floor | beyond / below | Δ run-down | deaths |
  |---|---|---|---|---|---|
  | 12 MWWk | +326.2 | 146.5 | 8 / 0 | 0 | 0 |
  | 12 RR | +344.9 | 146.5 | 9 / 0 | −1 | 0 |
  | 42 MWWk | +868.4 | 362.2 | 4 / 0 | +2 | 0 |
  | 42 RR | +940.3 | 362.2 | 4 / 0 | 0 | 0 |

- **Loss classes, k4 → rf3:**
  - 12 MWWk: late 178,082 → 153,496 (headroom-blocked 9,660); died upstream 76,945 → 69,435;
    contention 67,581 → 69,467; capacity 24,594 → 22,263; held stock 233.1 → 257.6; investment
    rejection 0.078 → 0.099;
  - 12 RR: late 90,450 → 79,565; died upstream 39,482 → 36,106; held stock 250.3 → 275.6;
  - 42 MWWk: late 225,882 → 183,529; held stock 1,167 → 1,256; investment rejection 0.171 → 0.202.
- **Cost:**
  - One fresh secure 16 vCPU pod in EUR-IS-1, `2o5lqkdzg94d56`, ≈ 14:18:24–14:21:45 UTC; the run
    started 14:20:23.
  - **≈ 0.9 vCPU-h billed** (estimate 0.7, stop 3, watchdog 11 min, not reached), charged to M5-P.
  - **M5-P ≈ 0.9 of 80.**
- **Files:** `entity_planner_buyahead_k4_rf3/` holds `spec.json`, `reading.json`,
  `supply_trace.csv`, `contract_failures.csv`, and `artifacts/episodes.csv` (git-lfs, SHA-256
  `ae4744db…`) / `producer_cycles.csv`. Also `read_exp0079.py` → `reading_exp0079.json`, and
  `launch.log`.

## Conclusions
- **`rf_helps`:** 12 producers +326 / +345 (+11%) vs a floor of 146.5, with no deaths and no extra
  run-down; 42 producers +868 / +940.
- **Expanding inventory earlier is the producer's next lever after buying ahead.** rf3 replaces k = 4
  as the M5 base and the strongest known producer. Roster inclusion is a human decision.
