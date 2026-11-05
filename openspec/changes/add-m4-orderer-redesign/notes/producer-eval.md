# Producer evaluation (user decision 2026-11-05)

Goal: measure how much room the producer side has on the loaded worlds, against what is achievable rather than against another hand-written rule (EXP-0035/0036/0054 never did).

## Step 1 — bound check (supervisor, offline, 2026-11-05)
Source: EXP-0052 `worlds.csv` (atlas confirm), worlds with ρ > 0.28, not collapse; arm `entity_planner/makeable_weighted_whole` (MWW stack).

| bound | 12 producers (n = 44), MWW / bound median (range) | 42 producers (n = 24) |
|---|---|---|
| demand cap B2 | 0.28 (0.07–0.86) | 0.42 (0.07–0.79) |
| base capacity B_base | 0.43 (0.19–0.97) | 0.49 (0.07–0.79) |
| input-output LP B_lp | 0.28 (0.07–0.86) | 0.42 (0.07–0.79) |
| tightest bound | **0.44** (0.21–0.98) | **0.49** (0.10–0.79) |

Tightest bound at 12: B_lp_base 31, B2 8, B_lp 3, B_base 2; at 42: B2 11, B_lp_base 10, B_lp 2, B_base 1.
Caveats: bounds assume no degradation (free, perfect restoration), no TTL / auction / whole-order frictions → upper limit, not demonstrated headroom. Rough degradation correction (bound × MWW's mean r_end ≈ 0.66): MWW ≈ 0.67 of it (circular: a better producer keeps capacity higher). All simple orderers sit in the same place, so the gap is not routing.
Reading: real room likely on the supply side → step 2.

## Step 2 — traced supply replay (tasks 7.14 / 7.15, EXP-0074)
Done 2026-11-05 (EXP-0074, `eval/supply_trace.py`, `entity_planner` + MWWk on EXP-0057's 71 world-episodes, TTL 2; neutral 142 / 142, consistent).
Reading: **`b2_not_delivered`** — inputs ordered but not delivered in time are 74.6% of lost consumer units at 12 (79.5% at 42); runner-up contention 10.2% (no tie).
| class | 12 | 42 |
|---|---|---|
| b2 late (input arrived after the reservation expired) | 32.6% | 31.7% |
| b2 lost_ttl (input order lost upstream: supplier's reservation expired) | 29.1% | 30.6% |
| b2 lost displaced / no_fit upstream | 9.6% | 11.7% |
| b2 unfunded / short / open | 2.5% | 4.5% |
| b4 contention | 10.2% | 7.0% |
| c outbid | 8.8% | 3.6% |
| a capacity (degraded 2.6%) | 5.2% | 6.9% |
| b1 never ordered | 0.5% | 2.0% |
| d outage | 0.5% | 1.0% |
| b3 upstream unable | 0.1% | 0.04% |
Restorations complete (37,571 / 37,741 at 12); replenishment units rejected 47% / 61%.
Decision to target in step 3: **input buying lead time / timing** (order earlier or stock ahead), not capacity or restoration. Open: does a longer reservation lifetime let inputs arrive (7.17, EXP-0075)?
## Step 3 — producer teacher (later, decided by step 2)
