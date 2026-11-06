# EXP-0097: M5 L5 validation under v8 (run-down term in the fitness), on k5_rf5

**Question.** The pre-registered L5 (v8 `9ce8e3bd…`, its rules unchanged): EXP-0096's generation-40 candidates
against k5_rf5.
- Worlds: the loaded atlas v2 validation worlds, 9 at 12 producers (deciding) and 3 at 42. 8 episodes × 2
  replicates.
- **Readings, by precedence:** kill → transfer_failure → success (both seeds > F_val at 12, no extra deaths,
  run-down / starvation shares ≤ the base's) → plateau.

**Setup.**
- Code 69b438d. `producer_read --mode validate`.
- Pod: secure 16 vCPU in **EUR-IS-1** (`agriqq956h8oe9`), 07:21–07:28 UTC. 0 contract failures.

## Registered reading: **`plateau`** (`success` missed on the run-down clause again)

| 12 producers (144 rows / arm) | − k5_rf5 (floor) | deaths | run-down share (rows) | starvation | held / cycle | paired r_end − base |
|---|---|---|---|---|---|---|
| k5_rf5 | — | 0 | 0.681 (98) | 0.778 | 283 | — |
| **cand-s0, v8** | **+130.3** (37.5) | 0 | **0.715** (103) | 0.757 | 341 | −0.0091 (SE 0.0036) |
| **cand-s1, v8** | **+90.2** (36.2) | 0 | **0.708** (102) | 0.778 | 347 | −0.0094 (SE 0.0039) |
| cand-s0, v7 (EXP-0095) | +139.9 | 0 | 0.743 (107) | 0.764 | 346 | −0.0114 |
| cand-s1, v7 | +105.9 | 0 | 0.736 (106) | 0.771 | 356 | −0.0110 |

- **42 producers (reported):** cand-s0 +349.0 (floor 129.6), cand-s1 +260.1 (157.5). 0 deaths. Run-down
  0.917 / 0.938 vs the base's 0.958, i.e. better than the base there.
- **`stock_heavy`:** not flagged (1.15×).

## What the run-down term did
- **Most of the shift remains.** It cut 4 of the 9 extra run-down rows at 12 (v7 +9 / +8 → v8 +5 / +4). The
  paired end-capacity shift moved only from −0.011 to −0.009, still about 2.5 SE below the base.
- **The units gain kept about 90%** (+130 / +90 vs +140 / +106).
- **Likely why it is weak** (not diagnosed):
  - ρ is one end-of-episode snapshot, with a re-key SD of 0.034 per episode.
  - The shift to remove is 0.01, about 0.3 SD. At λ = 1.4 it is worth about 0.014 in fitness, against a units
    term whose member spread is a few times that.
  - The ES ranks therefore barely see it.

## Conclusions
- **L5 `plateau`, a stop.** v8 produces a validated unit gain of +2–4% at 12 and +3–4% at 42, beyond the floor,
  with 0 deaths and modest stock, but its run-down share is still 4–5 rows above the base's at 12.
- **The design question for the supervisor / user:** a stronger or less noisy run-down signal (a larger λ by a
  different rule, ρ averaged over the episode instead of its end, or a constraint), or accepting `plateau`.
- **Spend:** ≈ 1.8 vCPU-h billed (stop 3). **M5-P ≈ 65.2 / 80**, past the 65 report line and under the 72 cap.
