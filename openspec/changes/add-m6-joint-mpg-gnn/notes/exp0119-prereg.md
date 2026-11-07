# EXP-0119 pre-registration: confirm F40 on fresh seeds

Frozen before any data (user decision 2026-11-07, item 1). Driver `tokenomics/train/m6_f40_confirm.py`.

## Question
EXP-0116 found that F40 is +230.1 units per episode ahead of the EXP-0100 stack on the monitor (F 136.4). F40 is
EXP-0114's iteration-40 investment gates (`candidate_it40.pt`) with the frozen start's replenish (EXP-0110
`start_models.pt`, SHA `697d94c0…`). That monitor was the set the runs were already gated on. Does F40's gain hold
on seeds never played before?

## Arms
- `ref`: the EXP-0100 stack (`models.pt`, its own replenish).
- `F40`: greedy investment gates of `candidate_it40.pt` with the start's replenish (`m6_gap_probe` mode
  `start_rep`).
- The orderer is the same frozen EXP-0100 orderer in both.

## Sets
Fresh episode seeds, stage `m6-f40-confirm`, horizon 100.
- **`monitor_fresh`:** the 16 monitor worlds × 4 new episodes × r0 / r1.
- **`validation_12`:** the atlas v2 validation worlds at 12 producers, not `collapse` (12 worlds) × 4 episodes × r0 /
  r1.
- **`validation_42`** (reported only): the 3 validation worlds at 42 producers, not `collapse` × 4 episodes × r0 / r1.

**Sizing:** 448 episodes at 12 and 48 at 42 ≈ 2–3 vCPU-h. Secure 16 vCPU with a persistent volume, EUR-IS-1, **stop
6**, M6-A.

## Reading
v11's metric and floors: M5's floors rule (g, F) over world-episodes.

**Deciding:** one pooled comparison over `monitor_fresh` ∪ `validation_12`:
- **`gain`:** g > F, and deaths, run-down and starvation shares no worse.
- **`hold`:** |g| ≤ F.
- **`worse`:** otherwise.

Each set is also reported alone, and so is `validation_42`.

## Consequence
- **`gain`:** 4.3 and 4.5 start from F40. The turn heads are re-fit on F40's trunk, v13 unchanged except for the
  start.
- **Otherwise:** the M6 start stays EXP-0100's stack. Report to the supervisor and the user.
