# EXP-0109: is the v4 producer's units shortfall under-restoration? (closed loop, horizon 200)

add-m6-joint-mpg-gnn task 2.15 (supervisor decision 2026-11-06; reading rule pre-registered in tasks.md before
data). Code b75a772 (`train/m6_restore_probe.py`). Pod vpy88jxjmd6v86 (secure 16 vCPU / 32 GB + 20 GB volume,
EUR-IS-1), 18:16–18:27 UTC ≈ **2.8 vCPU-h** (stop 16).

## Question
EXP-0108 placed the shortfall on the v4 producer, which under-restores offline (0.319 vs k5_rf5's 0.373 on
k5_rf5's recorded states). Is under-restoration the cause?
- **Arms:** v4c (v4 producer + EXP-0100 orderer), v2 (the EXP-0100 stack), base (k5_rf5 + MWWk), and v4c_bias
  (v4c with the restore bias shifted by +7.23, so it fires at k5_rf5's rate on the monitor samples; agreement
  0.941 → 0.966).
- **Sets:** the 12 monitor world-episodes × r0 / r1, and the four 42 worlds with EXP-0107's largest shortfalls
  (0000, 0043, 0076, 0039), 4 episodes × r0 / r1, all at horizon 200.

## Result: **`other`.** Restoring at k5_rf5's rate recovers nothing
Recovered share r = (v4c_bias − v4c) / (base − v4c): **−0.020 at 12, −0.035 at 42.**

| set | arm | units | units / cycle, cycles 0–49 / 150–199 | capacity, cycles 0–49 / 150–199 | restore ask rate, 0–49 / 150–199 | upgrade ask rate, 0–49 | upgrades done | expansions done |
|---|---|---|---|---|---|---|---|---|
| 12 | base | 8,633 | 31.5 / 47.8 | 129.3 / 159.8 | 0.345 / 0.421 | 0.198 | 104.8 | 28.0 |
| 12 | v2 | **9,007** | 33.1 / 49.5 | 145.2 / 171.2 | 0.391 / 0.438 | 0.208 | 107.5 | 49.3 |
| 12 | v4c | 8,390 | **29.0** / 47.3 | **120.6** / 147.8 | 0.339 / 0.409 | **0.176** | 96.1 | 25.6 |
| 12 | v4c_bias | 8,385 | 28.8 / 47.0 | 124.2 / 153.4 | 0.369 / 0.420 | 0.178 | 96.0 | 25.4 |
| 42 | base | 24,425 | 93.2 / 135.0 | 164.4 / 199.0 | 0.387 / 0.473 | 0.287 | 390.9 | 150.7 |
| 42 | v2 | **25,333** | 100.2 / 136.3 | 188.1 / 232.0 | 0.410 / 0.495 | 0.278 | 401.3 | 216.4 |
| 42 | v4c | 21,816 (1 death) | **80.4** / 125.8 | **143.6** / 176.8 | 0.384 / 0.465 | **0.239** | 354.8 | 118.2 |
| 42 | v4c_bias | 21,725 (1 death) | 80.3 / 125.2 | 146.9 / 184.0 | 0.411 / 0.491 | 0.238 | 357.2 | 117.6 |

Losses by class (mean per episode, `SupplyObserver`), v4c against base:
- **12:** input supply lost to TTL (`b2_lost_ttl`) **997 vs 871**; late inputs (`b2_late`) 1,725 vs 1,602.
- **42:** `b2_lost_ttl` **7,619 vs 5,949**; `b2_late` 8,192 vs 7,721.

## Diagnosis
- **Not under-restoration.**
  - In closed loop, v4c asks to restore at about the base's rate (12: 0.339 → 0.409 vs 0.345 → 0.421; 42:
    0.384 → 0.465 vs 0.387 → 0.473). The offline 0.319 was measured on k5_rf5's states, not its own.
  - The calibrated bias raises restorations (+5% at 12, +6% at 42) and the capacity ratio, but not the units.
- **What differs is growth early in the episode.**
  - v4c asks for fewer upgrades in cycles 0–49 (12: 0.176 vs 0.198; 42: 0.239 vs 0.287). It completes fewer
    upgrades (−8% / −9%) and expansions (−9% / −22%).
  - So it starts with less total capacity: −7% at 12 and −13% at 42 in cycles 0–49.
  - The units gap is concentrated there (12: −2.5 units per cycle in cycles 0–49, −0.5 in cycles 150–199;
    42: −12.8 vs −9.2).
  - It also loses more inputs to TTL (`b2_lost_ttl` +14% / +28%), a replenishment-side loss.
- **v2,** the over-investing imitation, has the most capacity and the most units at both sizes. Here it also
  out-produces the base over 200 cycles at 42 despite 82 producers past the ceiling.
- **Not discriminated:** whether the early under-upgrading, the fewer expansions, or the input-supply losses
  carry the shortfall. All three point away from the restore head.
- The step input and boundary weight that fixed the ceiling (EXP-0107) are plausibly making the upgrade gate
  more conservative below it as well. That is a hypothesis: offline, on k5_rf5's states, v4 does not under-fire
  above zero (0.142 vs label 0.133).
- DIAGNOSTICS 34.
