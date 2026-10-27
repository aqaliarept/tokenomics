# Sizing for tasks 10.1 and 12.1 (tasks 8.3 and 8.5)

Two measurement rounds, both on the local box (AMD EPYC-Rome, 2 vCPU, 1 thread per core), single
process, BLAS pinned to 1 thread, one run at a time, with the 1-minute load average recorded per
run in the raw logs. Horizon of the real runs: 100 (the `python -m tokenomics.eval.gate` default;
neither 10.1 nor 12.1 overrides it).

- **After** (Part A, task 8.5): code at commit c31e1a7, which includes the byte-identical
  joint-stack observation speed-up of task 8.4 (docs/DIAGNOSTICS.md entry 17). Measured
  2026-10-27. The recorded 1-minute load average ranged 0.30–1.52 over 134 runs (99 above 1.0) on
  this 2-core box. Two runs exceeded 1.5, both at 1.52: the sparse-120 h = 12 member-spread runs of
  `random/obs_price` and `random/even_split`. The sparse-120 member-spread runs as a whole were
  taken at 1.04–1.52. Other agent sessions share this box, so background load cannot be ruled out.
  Each run's load is in its JSON line.
- **Before** (Part B, task 8.3): code at commit 0da8ce8. Measured 2026-10-26 and re-measured
  2026-10-27; maximum load 1.23. Kept unchanged below for comparison.

## Plain-English summary

> **Interim (2026-10-27).** These numbers size the large economy at 120 producers. The user has
> since moved the large size to 42 producers (task 8.7). Task 8.8 re-sizes 10.1 and 12.1 at
> 12/42, and its Part A replaces the figures below; this round then becomes history.

The task 8.4 speed-up cut the dominant cost, a 120-producer episode under the joint-stack
contract, **8.6× (sparse-wide) to 9.9× (dense)**. That is measured old vs new code at the same
horizon (h = 12); at h = 16 the sparse-wide ratio is 8.4×. At horizon 100, now measured directly,
one such episode takes **27 s (sparse-wide) to 37 s (dense)** of one CPU. The 8.3 figures of 4 to
7.5 minutes were extrapolations of the old code from h ≤ 16. The old code was never run at
h = 100 at 120 producers, so whether that extrapolation was too high or too low is unknown. The
old/new ratio of the h = 100 figures (9.6× and 12.3×) therefore mixes the measured speed-up with
an unverified extrapolation, and should not be read as the speed-up. Joint-stack episodes now cost
2–3× a legacy-contract episode at 120 producers, down from 18–38×. Twelve-producer episodes cost
about 1 s, against 1.6–2.5 s before (2.0–2.3×, measured at h = 100 on both codes).

Totals are **conditional scenarios, not bounds** (§A4). The **reference scenario**, where every
roster job costs as much as the measured `entity_planner/proportional` episode, puts:
- **10.1 (screen) at ≈90 CPU-hours**, about 5.6 h wall on a cpu16 pod or 2.8 h on cpu32. This
  uses on-demand reserves, the user's choice on 2026-10-27, implemented in task 8.6: reserve
  worlds run only when a gate world has a zero hindsight baseline. None is expected (§A4); each
  one that occurs adds 0.05–7.7 CPU-hours.
- **12.1 (report) at ≈90 CPU-hours**, about 6 h on cpu16 or 3 h on cpu32.

Running every reserve up front (`--reserves eager`) would put 10.1 at ≈162. Before the speed-up
the eager screen was ≈1,600 and the report ≈910. Those old totals rested on the h = 100
extrapolation above, so their ≈10× drop is not a measured speed-up either. Within what the data
supports, the scenarios span **≈75–106 CPU-hours for 10.1** (on-demand, no replacement) and
**≈77–108 for 12.1**. Pod CPU speed and contention between concurrent workers are not measured,
and they come on top of that range. 120-producer jobs are 97% of the cost.

The next dominant cost, inside a 120-producer joint episode, is still the orderer observation.
In one profiled h = 12 episode:
- encoding the ≈3,700-number vector per stream is ≈32–35% of the time;
- building the stream view is another ≈17–30%;
- the producer observation is ≈12–27%.

The following estimates hold only if those profiled h = 12 shares carry over to unprofiled h = 100
roster jobs (not shown; profiling inflates Python-call-heavy code). On that condition, removing
the encode cost entirely would save about 47 of 10.1's 162 reference CPU-hours. Removing encode and
view together would save about 80, which is roughly the deferred eligible-only-rows refactor, and
that refactor changes the observation (§A5); these figures are on the eager basis (162). Of
the two plan-level levers in §A6, the user chose on-demand reserves (now the default). R stays at
5. For the record, a smaller R would give, for an eager 10.1:

| R | 10.1 CPU-hours |
|---|---|
| 5 | 162 |
| 3 | 116 |
| 2 | 93 |
| 1 | 70 |

The user kept R = 5 (2026-10-27).

# Part A. After the task 8.4 optimization (commit c31e1a7)

## A1. Per-episode timing at h = 12, before → after (one episode per family × size × contract)

Same worlds, members and method as §B1: `dev-<family>-<P>-calm-00`, its first episode seed.
"joint" = `entity_planner/proportional` under `joint_stack_env`; "legacy" =
`obs_stock_aware/obs_price` under the flat env; "portable" = the 12.1 subject (EXP-0035 producer +
EXP-0012 orderer). Delivered units are identical before and after at every horizon measured, as
the byte-identity requirement demands.

| family | P | joint before → after | legacy before → after | portable before → after |
|---|---|---|---|---|
| dense | 12 | 0.322 → 0.141 s (2.3×) | 0.055 → 0.055 s | 0.875 → 0.463 s (1.9×) |
| sparse_wide | 12 | 0.189 → 0.091 s (2.1×) | 0.049 → 0.047 s | 0.508 → 0.308 s (1.6×) |
| dense | 120 | 53.26 → 5.36 s (9.9×) | 1.457 → 1.430 s | ≈71–76 (not run) → 10.26 s |
| sparse_wide | 120 | 28.80 → 3.35 s (8.6×) | 1.642 → 1.691 s | ≈60–63 (not run) → 6.67 s |

Legacy is unchanged within noise: the speed-up touches only the joint-stack orderer view.

## A2. Horizon scaling, after (fit of t = c·h^b, least squares in log-log)

Every series now reaches h = 100, so **no h = 100 number below is an extrapolation**. At 120
producers each horizon is a separate run under 2 min.

| contract | family | P | measured (h: s) | b ± se | episode at h = 100 (measured) | 8.3 value at h = 100 |
|---|---|---|---|---|---|---|
| joint | dense | 12 | 12: 0.141, 25: 0.301, 50: 0.587, 100: 1.106 | 0.971 ± 0.019 | 1.11 s | 2.53 s (measured) |
| joint | sparse_wide | 12 | 12: 0.091, 25: 0.184, 50: 0.392, 100: 0.764 | 1.012 ± 0.018 | 0.76 s | 1.56 s (measured) |
| joint | dense | 120 | 12: 5.356, 25: 10.145, 50: 19.229, 100: 37.378 | 0.917 ± 0.014 | 37.38 s | 461 s (extrapolated) |
| joint | sparse_wide | 120 | 12: 3.346, 25: 6.842, 50: 13.741, 100: 27.084 | 0.988 ± 0.004 | 27.08 s | 260 s (extrapolated) |
| legacy | dense | 12 | 12: 0.055, 25: 0.110, 50: 0.231, 100: 0.426 | 0.976 ± 0.024 | 0.43 s | 0.47 s |
| legacy | sparse_wide | 12 | 12: 0.047, 25: 0.100, 50: 0.188, 100: 0.378 | 0.976 ± 0.016 | 0.38 s | 0.38 s |
| legacy | dense | 120 | 12: 1.430, 25: 2.994, 50: 5.916, 100: 11.879 | 0.997 ± 0.003 | 11.88 s | 11.5 s (extrapolated) |
| legacy | sparse_wide | 120 | 12: 1.691, 25: 3.693, 50: 6.880, 100: 13.717 | 0.979 ± 0.022 | 13.72 s | 13.4 s (extrapolated) |
| portable | dense | 12 | 12: 0.463, 25: 0.924, 50: 1.870, 100: 3.684 | 0.982 ± 0.010 | 3.68 s | 7.01 s |
| portable | sparse_wide | 12 | 12: 0.308, 25: 0.598, 50: 1.460, 100: 2.472 | 1.012 ± 0.067 | 2.47 s | 4.27 s |
| portable | dense | 120 | 12: 10.26, 25: 20.28, 50: 40.79, 100: 79.29 | 0.969 ± 0.011 | 79.29 s | 879 s (extrapolated) |
| portable | sparse_wide | 120 | 12: 6.667, 25: 14.20, 50: 27.75, 100: 55.72 | 0.998 ± 0.008 | 55.72 s | 695 s (extrapolated) |

Reading: on the new code, cost is linear in horizon (b = 0.92–1.01) over the full 12–100 range
at both sizes. The "± se" is the standard error of the fitted slope on these points (fit
quality), not a prediction interval. The horizon caveat of §B2 no longer applies to the new
numbers, because the planning horizon is measured.

The last column is not a speed-up. At 120 producers the 8.3 values are extrapolations of the old
code from h ≤ 16 (12 for dense), and the old code was never run at h = 100 there. So the
old-extrapolated / new-measured ratios (dense 461 / 37.4 = 12.3×, sparse 260 / 27.1 = 9.6×)
combine the speed-up with the unknown error of that extrapolation. The measured speed-up comes from
runs at the same horizon on both codes:

| run | old code | new code | speed-up |
|---|---|---|---|
| dense-120, h = 12 | 53.26 s | 5.36 s | 9.9× |
| sparse-120, h = 12 | 28.80 s | 3.35 s | 8.6× |
| sparse-120, h = 16 | 38.51 s | 4.56 s | 8.4× |
| dense-12, h = 100 | 2.53 s | 1.11 s | 2.3× |
| sparse-12, h = 100 | 1.56 s | 0.76 s | 2.0× |

The new-code sparse-120 h = 16 run is in `/tmp/gate_perf/after/t120j.jsonl` (task 8.4). What 8.4
did to the old code's growth beyond h = 16 cannot be separated without an old-code h = 100 run.

Regime, measured at 120 producers this time (`after85/t120_shocked.jsonl` vs `t120j.jsonl`, h =
50): shocked vs calm is 13.720 vs 13.741 s on sparse-120 (−0.2%) and 20.474 vs 19.229 s on
dense-120 (+6.5%).

Member spread at 120 producers. All 40 roster v1 members were run at h = 12 on both calm worlds
(`after85/members120{s,d}.jsonl`); none died within 12 cycles. Per joint member, relative to
`entity_planner/proportional`:

| world | mean (replicate-weighted) | range |
|---|---|---|
| sparse-120 | 0.992× | 0.86–1.08× |
| dense-120 | 0.927× | 0.75–1.18× |

Weak members do die early at 120 producers. On sparse-120, `noop/even_split` and
`always_restore/proportional` die at cycles 13 and 14. On dense-120, `noop/proportional` and
`always_both/blend_0.5` die at cycles 27 and 35 (`after85/members120{s,d}_weak_h100.jsonl`). That
covers two members per family; the other weak members (the 12-producer list in §B2) are assumed to
behave alike.

## A3. Job counts (unchanged; see §B3)

10.1: 80 worlds (development gate + reserves), 464 world-episodes, 40,832 jobs. 12.1: 48 worlds,
256 world-episodes, 22,784 jobs (roster v1, R = 5: 68 joint + 20 legacy jobs per world-episode,
plus 1 subject job for 12.1).

## A4. Scenarios: CPU-hours and wall time (h = 100, one worker per vCPU, BLAS 1 thread)

Each row is a conditional scenario: the CPU-hours that follow if its assumption holds, from the
measured h = 100 costs of §A2. Multipliers apply only to the 120-producer joint-stack jobs: 87% of
10.1's reference total and 85% of 12.1's. All 120-producer jobs, legacy included, are 97%. Computed
by `/tmp/gate_timing/after85/size85.py`; output in `after85/size85.out`.

The low row's death factor per 120-producer joint job is (55 + 13·c/100) / 68. Of the 68 joint
jobs per world-episode, 13 belong to members that died early at 12 producers (§B2: always_both/blend_0.5,
always_both/proportional, always_restore/{blend_0.5, even_split, proportional},
noop/{blend_0.5, even_split, proportional}, and noop/random × 5 replicates). Those 13 are assumed to
die at cycle c and the other 55 to run all 100 cycles. c = 14 (sparse) and c = 31 (dense) are
representative values from two members per family measured at 120 producers (sparse 13 and 14,
dense 27 and 35). The other weak members are assumed to die at the same cycle; that assumption is
not measured. The 12-producer joint factor of 0.885 is inherited from §B2, measured on the old code
at h = 50, and it moves the totals by under 1%.

| scenario | assumption | 10.1 CPU-h | 12.1 CPU-h |
|---|---|---|---|
| low | member mix (measured at 120 producers, h = 12) × early deaths (formula above, from two members per family) (sparse 0.992 × 0.836, dense 0.927 × 0.868; 12-producer joint 0.885 from §B2) | 135.7 | 77.4 |
| **reference** | every job runs the full horizon at the `entity_planner/proportional` cost | **162.4** | **92.1** |
| member mix | every 120-producer joint job costs 1.18× (the most expensive measured member) | 187.8 | 106.1 |
| regime | shocked share (69% / 62.5%) × measured 120-producer shocked/calm ratio | 165.9 | 93.9 |
| high | member mix × regime | 192.0 | 108.2 |

| scenario | 10.1 cpu16 / cpu32 wall | 12.1 cpu16 / cpu32 wall |
|---|---|---|
| low | 8.5 h / 4.2 h | 4.8 h / 2.4 h |
| reference | 10.1 h / 5.1 h | 5.8 h / 2.9 h |
| high | 12.0 h / 6.0 h | 6.8 h / 3.4 h |

The 10.1 rows above are the **eager** screen, where every reserve runs. **10.1 as planned uses
on-demand reserves** (`--reserves on-demand`, the default since task 8.6). Wave 1 screens the 64
development gate worlds only. Further waves run just the reserves needed to replace zero-baseline
worlds, in fixed order. The frozen files are identical to the eager screen's (tested). Same
multipliers, gate worlds only (shocked share 62.5%; `after85/size86.out`):

| scenario (10.1 on-demand, no replacement) | CPU-h | cpu16 / cpu32 wall |
|---|---|---|
| low | 74.9 | 4.7 h / 2.3 h |
| **reference** | **89.6** | **5.6 h / 2.8 h** |
| member mix | 103.6 | 6.5 h / 3.2 h |
| regime | 91.4 | 5.7 h / 2.9 h |
| high | 105.7 | 6.6 h / 3.3 h |

**Expected replacements: none.** A gate world needs replacing only if all 40 roster members
deliver 0 units on every episode. One member alone, `entity_planner/proportional`, delivered
327–4,283 units at h = 100 (shocked at h = 50 for 120 producers) on each of the 8 development
worlds timed: calm-00 and shocked-00 for each family × size, in `after85/` and `evidence/`. The
zero baselines seen in the 8.3 dry runs came from a 3-member subset of only `always_both/*`
members at h ≤ 12 (§B6), which the full roster at h = 100 does not reproduce. Those 8 worlds are
8 of the 64 gate worlds, so a zero world elsewhere cannot be excluded. Each one adds the cost of
the reserves it consults (§A6(a) table), and waves run one after another:

| sensitivity | CPU-h | cpu16 / cpu32 wall |
|---|---|---|
| reference + 1 dense-120 shocked replacement (7.72 CPU-h) | 97.3 | 6.1 h / 3.0 h, plus one sequential wave |
| reference + 4 such replacements | 120.5 | 7.5 h / 3.8 h, plus up to four sequential waves |

A wave cannot overlap the one before it. One dense-120 shocked reserve is 10 episodes × 88 jobs,
about 7.7 CPU-hours, so ≈0.5 h on cpu16 if its jobs keep all workers busy.

Reference split for 10.1 by family × size: dense-120 89.6, sparse-120 68.2, dense-12 2.7,
sparse-12 1.9 CPU-hours. Within the 120-producer part, the joint-stack jobs are 141 of the 158
CPU-hours and legacy is 17. The subject in 12.1 now costs ≈2.5 CPU-hours (≈28 in §B2): 128
120-producer jobs at 56–79 s each, plus 128 small ones. The 8.3 reference totals were 1,604 and 913
CPU-hours (§B4).

The horizon-growth scenario of §B4 is gone: h = 100 is now measured. Not covered by any row:
- **Pod CPU speed**: every row scales by the pod's single-thread speed relative to this box.
- **Worker contention**: 16–32 concurrent workers sharing caches and memory bandwidth, plus the
  driver's per-row append and fsync. All timings here are single process; contention can only
  add time.
- **World and seed spread at 120 producers**: one world per family × size was timed.

Memory is as in §B2: about 49 MB per baseline worker, about 627 MB with torch (portable subject),
and about 617 MB for the driver. 16 or 32 workers fit a cpu16 (32 GB) or cpu32 pod.

## A5. Where the time goes now, and what a further optimization could save

cProfile of one 120-producer joint episode (`entity_planner/proportional`, h = 12), bucketed by
cumulative time. Script: `/tmp/gate_timing/after85/prof_buckets.py`; output:
`after85/prof_buckets_{sparse_wide,dense}120_h12.json`; full stats in `after85/prof_*.pstats`.
The buckets can overlap slightly: the `entity_planner` policy also builds producer entity views.
Profiling inflates Python-call-heavy code, so the shares are approximate.

| bucket | sparse-120 | dense-120 |
|---|---|---|
| orderer observation: `encode_ordering_stream` | 35% | 32% |
| orderer observation: `build_product_stream_view` | 17% | 30% |
| producer observation: `producer_entity_view` + encode | 27% | 12% |
| env step (decode, domain actions, `complete_cycle`) | 13% | 15% |
| — of which `complete_cycle` (auction / production) | 2% | 3% |
| policies (`ordering_proportional`, `producing_entity_planner`) | 9% | 14% |
| PettingZoo bookkeeping (`_accumulate_rewards`, `_clear_rewards`) | 5% | 6% |

**The next dominant cost is still the orderer observation:** 52% of the episode on sparse-120
and 62% on dense-120. Each of the 300 (sparse) or 504 (dense) orderer streams gets a vector with a
row for every one of the 120 producers (≈3,700 numbers), every cycle. The 8.4 change stopped
recomputing those rows, but each stream still assembles and converts its own copy. The
simulation itself (`complete_cycle`) is 2–3%.

What further work on it could save in 10.1. These are **conditional estimates, not ceilings**.
Each assumes the bucket shares of the profiled h = 12 episode carry over to unprofiled h = 100
roster jobs. That is not verified: the profile inflates Python-call-heavy code (profiled 6.4 /
11.4 s vs 3.3 / 5.4 s unprofiled), and other members and horizons may split their time
differently. Applied to the reference scenario, where the 120-producer joint jobs are 141 of the
162 CPU-hours:
- **Encode only.** A faster, still byte-identical vector assembly that removed the encode bucket
  entirely would save about 0.33 × 141 ≈ 47 CPU-hours (10.1 → ≈115). A partial speed-up saves
  proportionally less. An env-side memo of encoded rows was tried in 8.4 and gave nothing (the
  cost is the per-stream assembly, not the row normalization).
- **Encode + view.** Removing both would save about 80 CPU-hours (sparse 52% of 59 + dense 62% of
  82; 10.1 → ≈80). That is the territory of the deferred eligible-only-rows refactor: give each
  stream only the producers that declare its product. It changes the observation itself (not
  byte-identical) and the scorer contract, so it is a design decision, not a performance fix.
- **Legacy-equivalent case** (hypothetical, not a proven minimum). If a joint episode cost as much
  as a legacy one (11.9–13.7 s at h = 100), 10.1 would be ≈77 CPU-hours.
12.1 scales the same way (reference 92 CPU-hours).

Nothing here is implemented. Per the root-cause rule the supervisor and the user decide whether
any of it is worth doing.

## A6. Plan-level levers (arithmetic only, reference scenario)

**(a) On-demand reserves in 10.1. Chosen by the user on 2026-10-27 and implemented in task 8.6**
(`--mode screen --reserves on-demand`, the default; the 10.1 plan in §A4). The eager screen runs
every reserve world up front: 208 of its 464 world-episodes, 72.8 of its 162.4 CPU-hours (44.8%).
Screening gate worlds only costs 89.6 CPU-hours. Each reserve actually consulted adds its own
cost:

| reserve world | CPU-hours |
|---|---|
| dense-120 shocked | 7.72 |
| sparse-120 shocked | 5.88 |
| dense-120 calm | 2.32 |
| sparse-120 calm | 1.76 |
| any 12-producer reserve | 0.05–0.23 |

So 10.1 costs 89.6 + Σ(consulted reserves) CPU-hours. Each wave is decided only after the
previous one completes, from the rows alone, so the waves are deterministic (task 8.6).

**(b) R, the replicates per random member.** Jobs per world-episode are 18 + 10·R joint and
10 + 2·R legacy.

| R | jobs per world-episode (joint + legacy) | 10.1 CPU-h | 12.1 CPU-h |
|---|---|---|---|
| 5 (current, kept by the user) | 68 + 20 | 162.4 | 92.1 |
| 3 | 48 + 16 | 116.2 | 66.6 |
| 2 | 38 + 14 | 93.1 | 53.9 |
| 1 | 28 + 12 | 70.1 | 41.2 |

R is a pre-registered design default (D5): changing it changes how the hindsight baseline averages
random members. The user kept R = 5 (2026-10-27).

## A7. Raw data for Part A

All of the following are throwaway and not committed.
- `/tmp/gate_timing/after85/`: per-run JSON lines with load averages (`t12`, `t120j`, `t120l`,
  `t120p`, `t120_shocked`, `members120s`, `members120d`, `members120*_weak_h100`).
- Scripts: `size85.py` and its output `size85.out`; `prof_buckets.py`, its JSON output and
  `.pstats` files.
- The timing scripts are the 8.3 ones, `/tmp/gate_timing/time_ep.py` and `members.py`, run on
  commit c31e1a7.
- The 8.4 before/after profiles are `/tmp/gate_perf/prof_{before,after}_*.txt`.

# Part B. Before the optimization: task 8.3 measurements (commit 0da8ce8)

## B0. Summary at task 8.3 (before the optimization)

Almost all of the cost is 120-producer episodes under the joint-stack contract, the env that 28 of
the 40 roster members and the candidate run in. Measured up to 12–16 cycles and extrapolated to
horizon 100, one such episode takes about **4 minutes (sparse-wide) to 7.5 minutes (dense)** of one
CPU. Twelve-producer episodes take seconds. At 120 producers, legacy-contract episodes are 18–38
times cheaper than joint-stack ones. Within the measured range, cost grows linearly with horizon
(fitted exponent 0.97–1.04). The h = 100 figures at 120 producers are an extrapolation, not a
measurement (§B2). The per-cycle cost of the joint-stack orderer observation grows with the
number of producers.

These totals are **scenarios, not bounds**: each one fixes a cost per job and multiplies it by the
job counts. The **reference scenario** gives every job the measured cost of
`entity_planner/proportional` over the full horizon. It puts **10.1 (screen) at ≈1,600 CPU-hours**
(about 100 h wall on a cpu16 pod, 50 h on cpu32) and **12.1 (report) at ≈910 CPU-hours** (57 h /
28 h). Varying the member mix, the regime and the horizon extrapolation within what the data
supports gives **≈1,420–2,690 CPU-hours for 10.1** and **≈810–1,500 for 12.1** (§B4).
Pod CPU speed and contention between workers are not measured and come on top of that range. Both
runs are feasible but expensive. 120-producer joint-stack jobs are 95–98% of the cost.

## B1. Per-episode timing at h = 12 (one episode per family × size × contract)

World `dev-<family>-<P>-calm-00` from the draft registry, its first episode seed. "joint" =
`entity_planner/proportional` under `joint_stack_env`; "legacy" = `obs_stock_aware/obs_price`
under the flat env; "portable" = the real 12.1 subject (EXP-0035 producer + EXP-0012 orderer,
pulled with git-lfs: sha256 8b51b77c… and d25388bf…, matching the LFS oids).

| family | P | joint (s) | legacy (s) | portable (s) |
|---|---|---|---|---|
| dense | 12 | 0.32 | 0.055 | 0.88 |
| sparse_wide | 12 | 0.19 | 0.049 | 0.51 |
| dense | 120 | 53.3 | 1.46 | ≈ 71–76 (not run at h = 12) |
| sparse_wide | 120 | 28.8 | 1.64 | ≈ 60–63 (not run at h = 12) |

The 120-producer portable runs were stopped at h = 8 to keep each command under 2 min. Their h = 12
values are computed two ways. Linear from h = 8: 47.41 / 8 × 12 = 71.1 s (dense) and 40.00 / 8 × 12
= 60.0 s (sparse). The two-point power fit of §B2: 47.41 × 1.5^1.156 = 75.8 s and
40.00 × 1.5^1.131 = 63.3 s.

## B2. Horizon scaling (fit of t = c·h^b, least squares in log-log)

| contract | family | P | measured (h: seconds) | b ± se | episode at h = 100: fit / linear from longest run |
|---|---|---|---|---|---|
| joint | dense | 12 | 12: 0.322, 25: 0.659, 50: 1.271, 100: 2.532 | 0.970 ± 0.006 | 2.5 / 2.5 s |
| joint | sparse_wide | 12 | 12: 0.189, 25: 0.391, 50: 0.755, 100: 1.557 | 0.990 ± 0.013 | 1.5 / 1.6 s |
| legacy | dense | 12 | 12: 0.055, 25: 0.114, 50: 0.217, 100: 0.470 | 1.003 ± 0.026 | 0.5 / 0.5 s |
| legacy | sparse_wide | 12 | 12: 0.049, 25: 0.097, 50: 0.193, 100: 0.375 | 0.963 ± 0.008 | 0.4 / 0.4 s |
| joint | dense | 120 | 4: 17.29, 8: 34.52, 12: 53.26 | 1.021 ± 0.019 | 461 / 444 s |
| joint | sparse_wide | 120 | 4: 9.15, 8: 19.00, 12: 28.80, 16: 38.51 | 1.038 ± 0.007 | 260 / 241 s |
| legacy | dense | 120 | 12: 1.46, 25: 3.10, 50: 5.80 | 0.969 ± 0.037 | 11.5 / 11.6 s |
| legacy | sparse_wide | 120 | 12: 1.64, 25: 3.45, 50: 6.73 | 0.988 ± 0.014 | 13.4 / 13.5 s |
| portable | dense | 12 | 12: 0.875, 25: 1.799, 50: 3.511, 100: 7.007 | 0.980 ± 0.004 | 7.0 / 7.0 s |
| portable | sparse_wide | 12 | 12: 0.508, 25: 0.995, 50: 2.089, 100: 4.272 | 1.010 ± 0.023 | 4.2 / 4.3 s |
| portable | dense | 120 | 4: 21.28, 8: 47.41 | 1.16 (2 points) | 879 / 593 s |
| portable | sparse_wide | 120 | 4: 18.27, 8: 40.00 | 1.13 (2 points) | 695 / 500 s |

Reading, measured range: every series with three or more points has b between 0.96 and 1.04. The
"± se" is the standard error of the slope on these few points. It measures how well a power law
fits the measured range; it is not a prediction interval for h = 100. At 12 producers the series
reach h = 100, so their h = 100 numbers are measurements. At 120 producers the longest runs are
h = 12 (dense) and h = 16 (sparse), so their h = 100 numbers extrapolate 6–8× beyond the data.

Reading, beyond the measured range (an inference, not verified): the marginal cost per cycle,
taken from the same deterministic episode prefix run to successive horizons, is

| world | cycles 1–4 | 5–8 | 9–12 | 13–16 |
|---|---|---|---|---|
| sparse_wide-120 joint | 2.287 s | 2.463 s | 2.451 s | 2.427 s |
| dense-120 joint | 4.323 s | 4.307 s | 4.684 s | — |

Sparse is flat after the first block. Dense rises 9% in its last block, and three blocks cannot
tell whether that is a trend or noise. The code bounds the hot loop: §B5 shows it scans the
telemetry's EMA keys, one per (producer, product) pair that has seen an event, not the cycle
history. That finite key universe guarantees per-cycle cost eventually stops growing. It does not
show *when*: key counts were not measured, so it cannot be claimed that the plateau is reached
within the measured cycles. This **corrects my task-6.1 note**, which said cost grew with the cycle
history; it grows with the key set, which is bounded. To expose the uncertainty, §B4
includes a **horizon-growth scenario**. In it, the marginal per-cycle cost keeps rising after the
last measured block at the slope from the first block to the last (dense +0.045 s per cycle per
cycle, sparse +0.012). That gives 648 s (dense) and 286 s (sparse) per h = 100 episode, ×1.41 and
×1.10 on the fitted values.

The 120-producer portable exponent rests on two short-horizon points, so the planning numbers use
the larger of fit and linear (879 s and 695 s). The subject is 256 / 22,784 = 1.1% of 12.1's jobs
(the 120-producer ones are 128 = 0.56%). Its estimated cost is ≈28 CPU-hours, 3.1% of 12.1's
reference total. An error of ±50% on the subject moves that total by about ±1.5%.

Regime (re-measured 2026-10-27, `evidence/regime12.jsonl`): at h = 100, shocked worlds cost +3%
over calm on sparse-12 (1.542 vs 1.495 s) and +20% on dense-12 (3.034 vs 2.533 s). The 120-producer
regime effect was not measured. Section 4 applies the 12-producer ratios to the shocked share of
world-episodes (69% for 10.1, 62.5% for 12.1) as a scenario.

Member spread: all 40 roster v1 members were timed on `dev-sparse_wide-12-calm-00` at h = 50. Joint
members that survive take 0.65–0.87 s, against 0.76 s for `entity_planner/proportional`. Nine weak
joint members (noop/*, always_restore/*, two always_both/*) die by cycles 14–19 and cost about 0.2
s. Weighted by replicates, the mean joint job is 0.885× the `entity_planner/proportional` cost. At
120 producers, six members were sampled on one calm world at h = 4. Relative to
`entity_planner/proportional` (9.146 s), the joint ones ran at 0.59× (noop/even_split), 0.73×
(always_both/even_split), 1.07× (random/proportional), 1.16× (stock_aware/blend_0.5) and 1.18×
(entity_planner/even_split); legacy ran at 0.14 s per cycle. So the reference member is not the most
expensive one: some members cost up to 18% more. Whether weak members also die early at 120
producers by h = 100 was not measured (each run would exceed 2 min). One calm world per family ×
size was timed, with no seed or world spread at 120 producers. Section 4 turns these into
scenarios rather than bounds.

Memory (re-measured 2026-10-27 with `/usr/bin/time`, `evidence/rss_*.log`): a baseline worker
peaks at 49 MB RSS (sparse-120, h = 2). A worker with torch (portable subject, dense-120, h = 2)
peaks at 627 MB. The driver process (`--mode report --dry-run`, 12 producers) peaks at 617 MB,
because `environment_manifest` imports torch to record its version. 16 workers × 627 MB ≈ 10 GB, and
32 × 627 MB ≈ 20 GB. Both fit a cpu16 (32 GB) or cpu32 pod; roster-only workers need far less.

## B3. Job counts (from `eval/registry/worlds_v1.json` 9ec52c4d… and `eval/registry/roster_v1.json` 8791e24b…)

Roster v1: 40 members, R = 5. Each world-episode has **68 joint-stack jobs** (18 deterministic +
10 random × 5) and **20 legacy jobs** (10 deterministic + 2 random × 5). For 12.1 these job counts
are upper bounds, since pruning in 10.1 can only remove members.

10.1 runs the screen on development gate worlds plus reserves, roster only:

| family | P | worlds | world-episodes | joint jobs | legacy jobs | CPU-h (reference scenario) |
|---|---|---|---|---|---|---|
| dense | 12 | 20 | 116 | 7,888 | 2,320 | 5.9 |
| dense | 120 | 20 | 116 | 7,888 | 2,320 | 1,016.7 |
| sparse_wide | 12 | 20 | 116 | 7,888 | 2,320 | 3.7 |
| sparse_wide | 120 | 20 | 116 | 7,888 | 2,320 | 577.4 |
| **total** | | 80 | 464 | 31,552 | 9,280 | **1,603.6** |

12.1 runs the report on development gate worlds: the subject plus the frozen roster.

| family | P | worlds | world-episodes | joint jobs | legacy jobs | subject jobs | CPU-h (reference scenario) |
|---|---|---|---|---|---|---|---|
| dense | 12 | 12 | 64 | 4,352 | 1,280 | 64 | 3.4 |
| dense | 120 | 12 | 64 | 4,352 | 1,280 | 64 | 576.6 |
| sparse_wide | 12 | 12 | 64 | 4,352 | 1,280 | 64 | 2.1 |
| sparse_wide | 120 | 12 | 64 | 4,352 | 1,280 | 64 | 330.9 |
| **total** | | 48 | 256 | 17,408 | 5,120 | 256 | **912.9** |

The 256 world-episodes for 12.1 match the tasks.md estimate. The tasks.md figure for 10.1
(≈22.5k = 256 × (28 + 12·R)) counts development gate world-episodes only. 10.1 also runs the
reserves (208 more world-episodes), so the files give 464 × 88 = 40,832 jobs.

## B4. Extrapolated wall time and vCPU-hours (h = 100, one worker per vCPU, BLAS 1 thread)

Each row is a conditional scenario: the CPU-hours that follow if its assumption holds. None of
them is a bound. Multipliers apply to the 120-producer joint-stack jobs, which are 95–98% of the
total; other jobs keep their reference cost (the low row also applies 0.885 at 12 producers).
Computed by `/tmp/gate_timing/sens2.py`; output in `evidence/scenarios.txt`.

| scenario | assumption | 10.1 CPU-h | 12.1 CPU-h |
|---|---|---|---|
| low | joint jobs cost 0.885× reference (replicate-weighted 12-producer member mix, early deaths included) | 1,421 | 812 |
| **reference** | every job runs the full horizon at the `entity_planner/proportional` cost | **1,604** | **913** |
| member mix | every 120-producer joint job costs 1.18× (the most expensive sampled member) | 1,888 | 1,070 |
| regime | shocked share × 12-producer shocked/calm ratio (dense +20%, sparse +3%) | 1,755 | 988 |
| horizon growth | marginal per-cycle cost keeps rising (§B2; dense ×1.41, sparse ×1.10) | 2,072 | 1,171 |
| high | member mix × regime × horizon growth | 2,687 | 1,498 |

| scenario | 10.1 cpu16 / cpu32 wall | 12.1 cpu16 / cpu32 wall |
|---|---|---|
| low | 89 h / 44 h | 51 h / 25 h |
| reference | 100 h / 50 h | 57 h / 29 h |
| high | 168 h / 84 h | 94 h / 47 h |

Dominant term in the reference scenario: dense-120 joint is 63% of 10.1 and 61% of 12.1; all
120-producer joint jobs are 98% and 95%.

vCPU-hours billed equal CPU-hours under full packing. The longest single job (dense-120 portable,
≈15 min reference) is small next to the wall time, so packing is near-perfect. Not covered by any
row:

- **Pod CPU speed.** A pod vCPU is assumed to be as fast as one core of this EPYC-Rome box. A
  different CPU scales every row by its single-thread speed ratio.
- **Worker contention.** 16 or 32 concurrent workers share caches and memory bandwidth, and the
  driver appends rows and fsyncs as they arrive. Neither was measured here (all timings are single
  process), and both can only add time.
- **World and seed spread at 120 producers.** One world per family × size was timed.

## B5. Feasibility verdict and cause

**Both runs are feasible but expensive.** 10.1 is ≈1,600 vCPU-hours in the reference scenario
(≈1,420–2,690 across §B4's scenarios): about 3.7–7 days on cpu16 or 1.9–3.5 days on cpu32. 12.1 is
≈910 (≈810–1,500). Pod CPU speed and contention are not included. The 12-producer half of either run
costs under 10 CPU-hours.

The cause is the per-cycle cost of the joint-stack orderer observation at 120 producers, not the
horizon and not the number of jobs at 12. Profile of `dev-sparse_wide-120-calm-00`,
`entity_planner/proportional`, h = 3, cProfile, 14.56 s episode. My original profile output was
printed to the session and not saved. The reviewer reproduced it independently on 2026-10-27 (bus
event 151 on META 72: same world, member and horizon, load 0.22, 14.394 s). That run recorded
`observe` 13.896 s (96.5%), `build_product_stream_view` 12.381 s (86%), 1,196 view calls, 143,520
producer-stats calls, 1,004,640 `sum_by_producer` calls and 4.353 s genexpr self time. The call
counts are identical and the times agree within 2%:

| function | cumulative s | share |
|---|---|---|
| `env/aec_env.py observe` | 14.04 | 96% |
| `domain/orderer_view.build_product_stream_view` (1,196 calls, ≈400 orderer streams per cycle) | 12.52 | 86% |
| `domain/orderer_view._producer_stats` (143,520 calls = 1,196 × 120 producers) | 8.93 | 61% |
| `domain/telemetry.sum_by_producer` (1,004,640 calls; genexpr over the EMA keys: 4.36 s self) | 5.67 | 39% |
| `domain/orderer_view._producer_product_stats` | 3.36 | 23% |
| env `step` (the simulation itself) | 0.28 | 2% |

Every orderer stream recomputes the per-producer stats for all 120 producers each cycle, and each
stat scans the whole EMA key set. Per-cycle cost therefore scales roughly as streams × producers ×
keys. Measured: from 12 to 120 producers, per-cycle cost grows about 150× (sparse) and 175× (dense)
for 10× the producers. The legacy contract does not build this view and grows only about 30×.

Per the root-cause rule, no fix is proposed or implemented here. The supervisor and the user choose
the next step: run as sized, change the run plan, or open a separate performance task in `domain/`.

## B6. Dry runs (`--dry-run`, baseline subject, single worker, temp dirs under `/tmp/gate_dry/`)

| run | command shape | run time (retained) | output |
|---|---|---|---|
| screen, 12 producers, h = 12 | `--mode screen --dry-run --no-reserves --sizes 12 --max-worlds 1 --episodes-per-world 1 --max-members 3 --replicates 1` | 2.06 s | EXP layout + `frozen/` (headroom passed: dense-12 median fill 0.907) |
| report, 12 producers, h = 12 | `--mode report --dry-run --subject baseline:entity_planner/proportional --params <tmp> --sizes 12 --max-worlds 1 --episodes-per-world 1 --max-members 3 --replicates 1` | 2.14 s | EXP layout; recompute MATCH |
| screen, 120 producers, h = 4 | as above, `--sizes 120 --horizon 4` | 29.66 s | EXP layout + `frozen/` |
| report, 120 producers, h = 4 | as above, `--sizes 120 --horizon 4` | 34.15 s | EXP layout; recompute MATCH |

The run times are measured from each `spec.json`'s `started` timestamp to the final write of the
same file (its mtime), both retained in `/tmp/gate_dry/<run>/`. They cover the whole run after
startup but exclude interpreter start and imports, about 1 s per command (the re-measured 12-producer
report dry run took 3.30 s under `/usr/bin/time`, `evidence/rss_driver_report12.log`). Total from
the retained artifacts: 68.0 s, plus about 4 s of startup. My session's `/usr/bin/time` readings
(3.1, 3.2, 30.8, 35.3 s = 72 s) were not saved. Either way the total is under the 2-min budget.
Deviations:

1. **120-producer dry runs use h = 4, not 12.** At h = 12, three joint-stack members plus the
   candidate on `dev-dense-120-calm-00` take more than 2 min per command (about 53 s each, §B1).
2. **The screen dry runs use a 3-member roster** (`entity_planner/proportional`,
   `stock_aware/blend_0.5`, `obs_stock_aware/obs_price`), still with `--max-members 3`. With the
   literal roster v1 and `--max-members 3`, the first three members in id order are
   `always_both/blend_0.5`, `always_both/even_split` and `always_both/obs_price`. They deliver 0
   units at h = 12, so the world's hindsight baseline is 0, and with `--no-reserves` the screen
   correctly stops: "zero hindsight baseline and no usable reserve left" (exit 1; that run's
   directory was replaced by the rerun, so no timing is retained). This is
   the zero-baseline rule working on a degenerate roster subset, not a defect. Note for 10.1:
   dry-runs with `--max-members` should pick delivering members.
3. The report dry run at 120 producers also has a zero baseline (the `always_both/*` members
   deliver nothing in 4 cycles). That world is reported as non-gating, as designed.

## B7. Raw data for Part B

All of the following are throwaway and not committed.

- `/tmp/gate_timing/*.jsonl`: per-run JSON lines, each with its load average. They are written by
  `time_ep.py` and `members.py`, which build the world from the draft registry, pin BLAS, and time
  `run_episode` on the world's first episode seed with `time.perf_counter`.
- `/tmp/gate_timing/size.py`: the fits and the reference totals.
- `sens.py`: marginal costs and shocked shares.
- `sens2.py`: the scenario table.
- `/tmp/gate_timing/evidence/`, the 2026-10-27 re-measures: `regime12.jsonl`, `rss_baseline120.log`,
  `rss_portable120.log`, `rss_driver_report12.log`, `scenarios.txt` and load averages.
- `/tmp/gate_dry/*/spec.json`: the dry-run timestamps.
