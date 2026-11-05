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
## Step 3 — producer teacher (later, decided by step 2)
