# EXP-0101: side effects of the imitated M6 stack (deciding check before MAPPO)

add-m6-joint-mpg-gnn task 2.7, pre-registered in tasks.md before any data. Code 511f8d1; EXP-0100 models (SHA
`5acf8b82…`); pod g6wve1cq1ekowi (secure 16 vCPU, EUR-IS-1), 11:47–11:49 UTC ≈ **0.6 vCPU-h** (stop 8).

## Question
EXP-0100's GNN stack beat k5_rf5 + MWWk by +196 units, and its producer fires restore and expand more often
than the teacher. Is that gain bought with a side effect that the units hide? The test uses L5's clauses as
in M5: deaths, the run-down share (r_end < X·r_start) and the starvation share (any consumer fill < Y), each
against the base.

## Result: **`clean`**
No clause is worse than the base, in any arm or at either horizon.

| horizon | arm | units | deaths | run-down share | starvation share | r_end / r_start | min cap. ratio | holding / cycle | inv. rejection |
|---|---|---|---|---|---|---|---|---|---|
| 100 | GNN + GNN (deciding) | 4,067.4 | 0 | **0.563** | **0.656** | 0.737 | 0.658 | 346.3 | 0.124 |
| 100 | GNN producer + MWWk | 4,078.4 | 0 | 0.500 | 0.688 | 0.744 | 0.660 | 345.5 | 0.122 |
| 100 | k5_rf5 + MWWk (base) | 3,871.4 | 0 | 0.781 | 0.750 | 0.696 | 0.636 | 282.3 | 0.099 |
| 200 | GNN + GNN | 9,007.0 | 0 | 0.563 | 0.563 | 0.719 | 0.642 | 416.0 | 0.099 |
| 200 | base | 8,632.9 | 0 | 0.781 | 0.625 | 0.693 | 0.625 | 331.3 | 0.080 |

- **The GNN stack has less run-down than the base, not more.** Its shares are 0.56 / 0.66 against the
  base's 0.78 / 0.75, and its capacity holds higher (r_end / r_start 0.74 vs 0.70).
- **No draining past the horizon:** at 200 cycles it keeps its lead (+374 units), its run-down share stays at
  0.56, and there are 0 deaths.
- **Reported (never deciding):**
  - Holding: 1.23× the base at 100 cycles and 1.26× at 200; under M5's stock-heavy flag (1.5×).
  - Investment rejection: 0.124 vs 0.099.
  - Fire rates on its own states: restore 0.41 vs 0.37, upgrade 0.13 vs 0.12, expand 0.058 vs 0.027.
  - Per-producer fund shares: near the base's (`episodes.csv` `fund_shares`).

## World 12-0121 (+1,285 units for the stack, +1,394 for the GNN producer)
- **The world:** capacity-bound. The base rejects 42% of consumer units, almost all of them by TTL expiry
  (7,665 units lost to `ttl`).
- **What the GNN producer does differently:** it restores (0.53 vs 0.46), upgrades (0.14 vs 0.12) and
  expands inventory (0.085 vs 0.050) more.
- **Its capacity rises above the base's from about cycle 30:** mean total capacity per producer 404 vs 372 at
  cycle 30, and 488 vs 407 at cycle 90.
- **The result:** TTL losses fall to 6,488 (−1,177), consumer rejection to 35%, and the minimum consumer fill
  rises from 0.33 to 0.39.
- **The cost:** holding 640 vs 478 per cycle, investment rejection 0.03 vs 0.02; fund shares almost
  unchanged (concentrated on p2_* in both).
- **So the gain is more productive capacity on a supply-limited world, not drained capacity.** r_end /
  r_start is 0.81 vs 0.77.

## Conclusions
- The imitated stack's gain over k5_rf5 + MWWk has no side effect on L5's clauses. It improves both
  run-down and starvation, and holds at 200 cycles.
- The GNN producer departs from its teacher toward more investment, and that departure is beneficial here. It
  is the producer the RL stage starts from.
- MAPPO (tasks 2.2b / 2.3) is unblocked on this evidence; the decision stays with the supervisor.
