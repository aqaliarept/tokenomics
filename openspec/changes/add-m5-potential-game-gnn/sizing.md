# M5 sizing (task 3.1, 2026-11-05)

## Measured (local box, one process, BLAS 1 thread; commit 912e36f)
Wall seconds per episode, MWWk at DrawKey r0:

| episode | horizon 12 | horizon 30 | per cycle | horizon 100 (fit) |
|---|---|---|---|---|
| 12 producers, rf3 | 0.248 | 0.502 | | |
| 12 producers, k4 | 0.219 | 0.517 | | |
| 12 producers, residual θ ≠ 0 (normalised) | 0.230 | 0.522 | 0.016 | **1.65** |
| 12 producers, read job (observer on), rf3 / residual | 0.240 / 0.237 | | | |
| 42 producers, read job, rf3, observer on / off | 3.69 / 3.68 | 8.80 / 8.80 | 0.284 | **28.7** |
| 42 producers, read job, residual | 3.78 | | | |

- **The residual costs the same as rf3 and k4**, and the observer adds nothing measurable.
- **Pods are faster than this box.** EXP-0079's 213 supply-trace jobs (129 at 12 producers, 84 at
  42) need ≈ 2,620 CPU-s at these rates, i.e. ≈ 164 s ideal on 16 workers; they ran in ≈ 77 s. So
  a pod is ≈ 2× this box. The estimates below use the local rates (≈ 2× conservative), plus a
  pod overhead of ≈ 3 min (setup, install, pull) at 16 vCPU ≈ 0.8 vCPU-h.

## Counts
- **Training:** 128 world-episodes at 12 producers.
  - A generation is 2 × 12 members × 24 world-episodes = 576 episodes.
  - A monitor is 16 world-episodes × 2 replicates (the centre), plus rf3's 32 once.
- **Probe:** 2 σ arms × 2 minibatches × 24 members × 24 = 2,304 episodes.
- **L5:** the loaded validation worlds, 9 at 12 and 3 at 42. 8 episodes × 2 replicates × 5 arms:
  720 + 240 episodes.
- **L6:** the held-out worlds, 12 at 12 (8 loaded) and 4 at 42, all played. 8 × 2 × 4 arms: 768 +
  256 episodes.

## Estimates and stops (16 vCPU, pinned EUR-IS-1; stop = 2 × estimate)

| run | compute (local rates) | wall, ≈ | estimate billed | stop |
|---|---|---|---|---|
| L1 (task 4.1, EXP-0080): bar 128 + noise floor 640 + cross-check 2 | 0.35 | 5 min | **1.5** | 3 |
| L2 (4.2, EXP-0081): probe, 2,304 episodes | 1.06 | 7 min | **2.0** | 4 |
| L3 (5.1, EXP-0082): pilot, 20 generations (11,520 + monitors) | 5.3 | 25 min | **7.0** | 14 |
| L4 (5.2): seed 0, generations 21–40 | 5.3 | 25 min | **7.0** | 14 |
| L4 (5.2): seed 1, 40 generations | 10.6 | 48 min | **13.5** | 27 |
| L5 (5.3, EXP-0083): validation, 720 + 240 episodes | 2.2 | 12 min | **3.5** | 7 |
| L6 (6.1, EXP-0084): held-out, 768 + 256 episodes | 2.4 | 13 min | **3.5** | 7 |
| **success path** | | | **≈ 38** | |

The 42-producer read jobs (≈ 29 s) dominate L5 / L6 wall time. Longest-first submission keeps the
tail within one job.

## Launch-fit against M5-P (80 vCPU-h, billed spend)
Spent so far: EXP-0079 ≈ 0.9 (notes/tranche-ledger.md).

| run | spent before | + stop | ≤ 80 |
|---|---|---|---|
| L1 | 0.9 | 3 | 3.9 ✓ |
| L2 | ≈ 2.4 | 4 | 6.4 ✓ |
| L3 | ≈ 4.4 | 14 | 18.4 ✓ |
| L4 seed 0 | ≈ 11.4 | 14 | 25.4 ✓ |
| L4 seed 1 | ≈ 18.4 | 27 | 45.4 ✓ |
| L5 | ≈ 31.9 | 7 | 38.9 ✓ |
| L6 | ≈ 35.4 | 7 | 42.4 ✓ |

- **Design D9's 60 re-sized to ≈ 38** on the success path.
- **About 42 stays free for failed pods** (each no-SSH failure costs ≈ 2.7 at 16 vCPU) and re-runs.
