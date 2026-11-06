# EXP-0095: M5 L5 validation under v7 (θ_e fixed at 0), on k5_rf5

**Question.** The pre-registered L5 (v7 `986a8191…`, its rules unchanged from v6): EXP-0094's generation-40
candidates against k5_rf5.
- Worlds: the loaded atlas v2 validation worlds, 9 at 12 producers (deciding) and 3 at 42.
- 8 episodes × 2 replicates; MWWk.
- **Readings, by precedence:** kill → transfer_failure → success (both seeds > F_val at 12, no extra deaths,
  run-down and starvation shares ≤ the base's) → plateau.
- **Caveat recorded in v7:** these validation worlds informed v7's design (EXP-0093), so L6 held-out would be
  the clean read.

**Setup.**
- Code 487dfee. `producer_read --mode validate`.
- Pod: secure 16 vCPU in **EUR-IS-1** (`59kjrf7qr7t3uf`), 00:18–00:24 UTC. 0 contract failures.

## Registered reading: **`plateau`** (`success` missed on the run-down clause)

| 12 producers (9 worlds, 144 rows / arm) | − k5_rf5 (floor) | deaths | run-down share | starvation share | held units / cycle |
|---|---|---|---|---|---|
| k5_rf5 (base) | — | 0 | 0.681 (98 rows) | 0.778 | 283 |
| **cand-s0** | **+139.9** (37.0): clears | 0 | **0.743** (107) | 0.764 | 346 |
| **cand-s1** | **+105.9** (42.7): clears | 0 | **0.736** (106) | 0.771 | 356 |
| k4 | −521.7 | 1 | 0.715 | 0.778 | 242 |
| `entity_planner` | −1,429.7 | 0 | 0.708 | 1.000 | 127 |

- **42 producers (reported):** cand-s0 +420.9 (floor 117.8), cand-s1 +244.5 (121.4). 0 deaths for every arm.
  Run-down 0.958 / 0.979 vs the base's 0.958.
- **`stock_heavy`:** not flagged. The selected cand-s0 holds 699 vs the base's 602 over both sizes, a ratio of
  1.16 < 1.5.
- **v7 vs v6 (EXP-0093), at 12:** gains about a third (+106 to +140 vs +383 to +413), 0 extra deaths (v6: 1),
  stock 1.16× (v6: 1.58×).

## Why the run-down clause fails (offline, from the read's rows)
- **A small, systematic shift:** the paired r_end difference (candidate − base) is **−0.0114** (SE 0.0037) for
  seed 0 and **−0.0110** (SE 0.0041) for seed 1. That is small against k5_rf5's own re-keyed spread of about
  0.030 per episode.
- **It lands on near-threshold episodes:** 26 of the base's 144 episodes sit within ±0.03 of the run-down line
  (r_end = 0.7516 r_start). The shift moves 10 (s0) / 11 (s1) episodes into failure and 1 / 3 out.
- **Associated changes, per episode:**
  - restoration rejections +0.9 to +1.7 (5.8 → 6.7–7.5);
  - upgrade orders placed +16 to +20 (139 → 155–159), with about the same completed (78 → 79–80);
  - replenishment units placed +9% (10.8k → 11.8k), rejected +17%.
  The candidates buy more inputs, and their producers ask for more upgrades (most capped). Their capacity ends
  about 1.6% lower. That the extra buying shares the fund with restorations is inferred, not traced per order.
- **The clause has no noise margin.** It compares raw shares, unlike the training guard's margin, so a small
  systematic shift on a crowded threshold decides it.

## Conclusions
- **L5 `plateau`, a stop.**
  - Without θ_e, the learned producer validates a real unit gain: +3–4% at 12, +3–5% at 42, beyond floor, with
    no deaths and modest stock.
  - It runs capacity down slightly more, and that fails the registered "run-down share ≤ base" clause.
- **L6 is gated on `success`**, so it does not run.
- **The next step is the supervisor's / user's:**
  - accept `plateau` as M5's producer result;
  - or a new pre-registration (a run-down-aware fitness or guard term, or a margin on L5's run-down clause),
    which needs its own justification.
- **Spend:** ≈ 1.5 vCPU-h billed (stop 7). M5-P ≈ 47.9 / 80.
