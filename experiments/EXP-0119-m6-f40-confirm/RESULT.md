# EXP-0119: F40 confirmed on fresh seeds

**Setup.**
- User decision 2026-11-07: "confirm F40, then use it".
- Pre-registration note `notes/exp0119-prereg.md` (SHA `195e8d91…`, committed f678f06 before data). Driver
  `train/m6_f40_confirm.py`.
- **F40:** EXP-0114's `candidate_it40.pt` (SHA `ace0fabf…`), greedy investment gates, with the EXP-0110 start's
  replenish (`start_models.pt`, `697d94c0…`).
- **Reference:** the EXP-0100 stack. The orderer is the same frozen one in both.
- Fresh seeds (stage `m6-f40-confirm`), horizon 100: 496 episodes.
- Pod m5c0ixjx8x8oz8 (secure 16 vCPU + 20 GB volume, EUR-IS-1), 12:27–12:34 UTC ≈ **1.7 vCPU-h** (stop 6, M6-A).

## Reading: **`gain`**
v11's metric and floors: M5's floors rule over world-episodes. Units are per episode.

| set | world-episodes | F40 units | EXP-0100 units | g | F | deaths | run-down share | starvation share | reading |
|---|---|---|---|---|---|---|---|---|---|
| **deciding** (monitor fresh ∪ validation 12) | 112 | 3,705.4 | 3,503.4 | **+202.0** | **52.7** | 0 vs 1 | 0.39 vs 0.61 | 0.62 vs 0.67 | **gain** |
| monitor worlds, fresh episodes | 64 | 4,249.0 | 4,004.4 | +244.7 | 62.3 | 0 vs 0 | 0.39 vs 0.58 | 0.61 vs 0.67 | gain |
| atlas validation, 12 producers | 48 | 2,980.5 | 2,835.5 | +145.0 | 90.7 | 0 vs 1 | 0.40 vs 0.66 | 0.63 vs 0.66 | gain |
| atlas validation, 42 producers (reported) | 12 | 10,531.1 | 10,055.2 | +475.9 | 236.2 | 0 vs 0 | 0.00 vs 0.67 | 1.00 vs 1.00 | gain |

## Conclusions
- **F40's gain holds on seeds never played**, and on validation worlds it was never trained on:
  - +5.8% units at 12 producers (pooled), with no deaths and lower run-down and starvation;
  - +4.7% at 42 producers, with no 42-producer training at all.
- **Per the user's decision:** F40 is the M6 start for 4.3 (v14) and 4.5. Its turn heads are re-fit on F40's trunk.
- **The deployed policy is a composite:** F40 = the candidate's gates + the start's replenish. 4.5 trains every
  head on-policy (user decision 2), so the composite becomes one model there.
