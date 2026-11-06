# EXP-0102: validation of the imitated M6 stack on unseen worlds (before MAPPO)

add-m6-joint-mpg-gnn task 2.8 (supervisor decision 2026-11-06). Pre-registration `m6_prereg_v3.json` (SHA
`409643f2…`, frozen before data; v1 / v2 untouched). Code 3fb3218; EXP-0100 models (SHA `5acf8b82…`). Pod
o78goq826u3u1d (secure 16 vCPU, EUR-IS-1), 11:54–12:01 UTC ≈ **1.9 vCPU-h** (stop 8). The held-out set was
not touched.

## Question
Does the self-trained (imitated) GNN stack already meet the user's goal on worlds it never trained on? The
goal is completion at parity with, or better than, the best hand-written stack, k5_rf5 + MWWk. The test uses
M5's L5 set and rule: the 9 loaded atlas v2 validation worlds at 12 producers (deciding) and 3 at 42
(reported), 8 episodes × replicates 0 / 1, horizon 100, L5's floors.

## Result: **`better`** at 12 producers, and at 42 (reported)
| size | arm | mean units (world-equal) | − base | floor | deaths | run-down share | starvation share | holding / cycle |
|---|---|---|---|---|---|---|---|---|
| 12 | GNN + GNN (deciding) | 3,411.2 | **+95.9** | 35.0 | 0 | 0.625 | 0.778 | 333.0 |
| 12 | GNN producer + MWWk | 3,405.9 | +90.7 | 39.0 | 0 | 0.611 | 0.778 | 332.4 |
| 12 | k5_rf5 + MWWk (base) | 3,315.3 | — | — | 0 | 0.681 | 0.778 | 282.9 |
| 42 | GNN + GNN | 10,182.5 | **+196.1** | 160.0 | 0 | 0.562 | 1.000 | 1,634.9 |
| 42 | GNN producer + MWWk | 10,111.7 | +125.4 | 118.3 | 0 | 0.583 | 1.000 | 1,631.8 |
| 42 | base | 9,986.4 | — | — | 0 | 0.958 | 1.000 | 1,561.0 |

- **All clauses hold:**
  - no deaths;
  - run-down share at or below the base's (0.625 vs 0.681 at 12; 0.562 vs 0.958 at 42);
  - starvation share equal to the base's.
- **The gain is +2.9% at 12 and +2.0% at 42, and it is uneven across worlds.**
  - At 12, the stack is ahead on 5 of 9 worlds. Two carry most of it (+570, +317 units); four are slightly
    behind (−47 … −2).
  - At 42 it is ahead on 2 of 3 (+444, +182, −38).
  - The equal-weight mean clears its floor at both sizes (95.9 > 35.0; 196.1 > 160.0).
- **The GNN producer again carries most of it** (the swap arm +90.7 / +125.4). The GNN orderer adds about
  +5 units at 12 and +71 at 42.
- Holding is 1.18× the base at 12 and 1.05× at 42; never stock-heavy.

## Conclusions
- **The user's goal is met on the validation worlds.** Self-trained agents (behaviour-cloned GNNs, no
  hand-written planner at deployment) reach better-than-parity order completion against the best
  hand-written stack, with no side effect on L5's clauses, at 12 and 42 producers.
- **This is a validation result, not a held-out confirmation.** L6 (held-out, one shot, ledger) has not run;
  the human gate decides it.
- **For MAPPO,** this is a validated start and a fallback: the RL stage must not end below this stack.
