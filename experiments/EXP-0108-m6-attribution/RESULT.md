# EXP-0108: attribution of EXP-0107's shortfall; the composite candidate (v4 producer + EXP-0100 orderer)

add-m6-joint-mpg-gnn, supervisor decision 2026-11-06. Pre-registration `m6_prereg_v7.json` (SHA `970fa4aa…`). Code
0166964. Pod ddraj6ise5j20c (secure 16 vCPU / 32 GB, EUR-IS-1), 16:52–17:37 UTC; **hard-stopped by the watchdog at
45 min (stop 12 vCPU-h ≈ 12.0 billed)**.

## What ran
- **Phase 1 (deciding) completed.** Its reading is in the launch log (`artifacts/launch.log`).
- **Phase 2 (the reported attribution arms) was cut off by the hard stop.**
  - The pod had no persistent volume, so `reading.json` and the phase 2 rows were lost with the stopped
    container. The pod was then deleted.
  - My estimate (≈ 40 min) was too tight for 5 arms at 42 producers and horizon 200.

## Result: candidate C **`short`** on (a), (b) units, (c); the ceiling part of (b) passes
| clause | C (v4 producer + EXP-0100 orderer) | EXP-0107 stack (v4 producer + v4 orderer) | needed |
|---|---|---|---|
| (a) 12 vs k5_rf5 + MWWk | **−179.1**, floor 155.3; deaths 0 / 0 | −136.6, floor 82.7 | ≥ −floor |
| (b) 42 past the ceiling | 0 vs 0 ✓ | 0 vs 0 ✓ | ≤ the base's |
| (b) 42 units | **−1,706.0**, floor 579.6; deaths 0 / 0 | −1,774.1, floor 568.4 | ≥ −floor |
| (c) vs the EXP-0100 stack | **−375.2**, floor 177.5 | −332.7, floor 131.6 | ≥ −floor |

## Attribution
- **The orderer is not the cause.** Swapping the diverged v4 orderer (KL 1.67, top-1 0.55) for EXP-0100's faithful
  orderer leaves the clauses unchanged, within their floors: (a) −179 vs −137; (b) −1,706 vs −1,774; (c) −375 vs
  −333.
- EXP-0100's orderer was itself neutral against MWWk (+5.7, EXP-0100). So **the shortfall belongs to the v4
  producer**, at both sizes, although it now holds the ceiling (0 past it).
- The reported arms that would confirm this directly (v4 producer + MWWk; k5_rf5 + each orderer) were lost.
- **What differs in the v4 producer, from the fit's fidelity metrics (not a closed-loop diagnosis):**

  | head | label | v4 | v2 (EXP-0100) |
  |---|---|---|---|
  | restore, monitor | 0.373 | **0.319** | 0.548 |
  | restore, 42 gate set | 0.486 | 0.434 | — |
  | expand, monitor | 0.027 | 0.032 | 0.111 |

  - v4 under-restores by about 5 points (−14%).
  - v2 over-restored and over-expanded. EXP-0101 traced its gain over k5_rf5 to more restore / expand → more
    capacity.
  - Hypothesis, not tested: v4's units shortfall comes from under-restoration (closer to k5_rf5 on upgrades and
    expansions, below it on restores).

## Conclusions
- **No candidate yet:** C is `short`. The EXP-0100 stack stays the imitated result (EXP-0102 `better`,
  EXP-0103 `better` at 12).
- **The shortfall is the v4 producer's,** not the orderer's (C ≈ the EXP-0107 stack).
- **Its mechanism is undiagnosed.** The discriminating check: the producers' restore decisions in closed loop
  (fire rate where the teacher restores, and capacity over cycles), v4 against k5_rf5 and v2 on the same keys.
