# EXP-0126: turn-head imitation v15 on F40 (task 4.3)

**Setup.**
- `m6_prereg_v15` (SHA `15b7c798…`, frozen before data). Code 8a3af7e.
- **Start:** F40 (pinned: EXP-0119's reading 246a94a7…, candidate_it40 ace0fabf…, the start's replenish 697d94c0…).
- **Heads:** 3 attention layers + the gated ordered accept pass; linear fractions + arrival position; width 32.
- **Fit:** 30 epochs cosine over 427,150 training turns, 1,655 s; final loss 0.041. The final model is read.
- **Held out:** 62,926 turns from 16 monitor worlds × 1 fresh episode × r0 / r1.
- Pod 30fw6eam8ac7ls (secure 4 vCPU + 20 GB volume, EUR-IS-1), 18:05–18:52 UTC ≈ **3.1 vCPU-h** (stop 166 min, M6-A).

## Offline gate: **pass**

| metric (fresh held-out, clipped / bounded scores) | value | gate |
|---|---|---|
| accept agreement | **0.9983** (127,651 bits) | 0.99 |
| accept-turn rank | **0.9995** (328,861 pairs) | 0.99 |
| fund rank | **1.000** (47,173 pairs) | 0.99 |
| whole schedule | 0.9947 of turns | reported |

- **Accept errors** (DIAGNOSTICS 38 split): no_fit 0 / 5,422; displaced 107 / 25,395 (0.42%); accepted 116 / 96,834 (0.12%).
- **Other reported quantities:**
  - ties 0;
  - saturation: priority 0, fund 0;
  - identical pairs 0 (the arrival position separates them).

## Closed loop (pre-registered gate): **`fail`**, on the run-down clause only
F40 + the fitted heads against F40 + default turns, on the same 16 fresh world-episodes × r0 / r1 (32 rows):

| | heads | default turns |
|---|---|---|
| units / episode | 4,262.8 | 4,226.4 |
| g / F | +36.4 / 54.8 | |
| deaths | 0 | 0 |
| run-down share | **0.375 (12 / 32)** | 0.281 (9 / 32) |
| starvation share | 0.625 | 0.625 |

The pass rule needs g ≥ −F *and* run-down / starvation shares no worse. The run-down share is worse by 3 rows of
32.

**42 producers (reported):** `pass`, g +190.9 vs F 189.9 (12,083.6 vs 11,892.8). Run-down 0.25 vs 0.25, starvation
0.75 vs 0.875, no deaths.

## Closed-loop diagnostic (EXP-0123's design, a second fresh set): **`no_cost`**
16 monitor worlds × 2 more fresh episodes (stage `m6-turn-imit-v15-closed-loop`) × r0 / r1 = 64 rows per arm:

| | heads | default turns |
|---|---|---|
| units / episode | 4,283.4 | 4,255.3 |
| g / F | +28.1 / 47.4 | |
| deaths | 0 | 0 |
| run-down share | 0.344 | 0.375 |
| starvation share | 0.594 | 0.609 |

- **Closed-loop accept agreement: 0.9978** (257,047 of 257,618; EXP-0123's heads: 0.956).
- **Differing bits:** 267 extra accepts, 304 extra declines.
- **Pending claims created:** 334 (EXP-0123: 13,360). 189 ran later, 1 failed at TTL.

## Conclusions
- **The imitation itself is solved.** Offline 0.998 / 0.9995 / 1.000 on never-played episodes. In closed loop the
  heads agree with the auction on 99.8% of bits, and the late-pending harm is gone (334 pending vs 13,360).
- **The pre-registered closed-loop gate reads `fail`** on one clause: the run-down share, 12 vs 9 of 32 rows, with
  units ahead inside the floor.
- On the second fresh set (64 rows) the run-down share goes the other way (0.344 vs 0.375), with all clauses no
  worse. The share clause has no noise floor, so 3 rows of 32 is within what two fresh sets disagree on.
- **The reading stands as pre-registered.** Whether 4.4 / 4.5 start from these heads is the user's decision.

## User decision (2026-11-07): accepted as the 4.4 / 4.5 start, on combined evidence
- The pre-registered closed-loop gate stays **`fail`**, on the run-down clause alone.
- Pooled over the 96 pairs (gate set 32 + diagnostic set 64), the run-down share was 34 vs 33 rows.
- Units were within noise in both sets, closed-loop agreement 99.8%, and the late-pending work is gone (334 vs
  13,360 claims).
- Task 4.3 is done.

