# Sizing for tasks 10.1 and 12.1 (task 8.3)

Measured 2026-10-26 (re-measures for regime and memory on 2026-10-27) on the local box (AMD
EPYC-Rome, 2 vCPU, 1 thread per core), single process, BLAS pinned to 1 thread, one run at a time.
The 1-minute load average was at most 1.23 at any timing; each run's value is recorded in the raw
logs listed at the end. Code: commit 0da8ce8. Horizon of the real runs: 100 (the
`python -m tokenomics.eval.gate` default; neither 10.1 nor 12.1 overrides it).

## Plain-English summary

Almost all of the cost is 120-producer episodes under the joint-stack contract, the env that 28 of
the 40 roster members and the candidate run in. Measured up to 12–16 cycles and extrapolated to
horizon 100, one such episode takes about **4 minutes (sparse-wide) to 7.5 minutes (dense)** of one
CPU. Twelve-producer episodes take seconds. At 120 producers, legacy-contract episodes are 18–38
times cheaper than joint-stack ones. Within the measured range, cost grows linearly with horizon
(fitted exponent 0.97–1.04). The h = 100 figures at 120 producers are an extrapolation, not a
measurement (section 2). The per-cycle cost of the joint-stack orderer observation grows with the
number of producers.

These totals are **scenarios, not bounds**: each one fixes a cost per job and multiplies it by the
job counts. The **reference scenario** gives every job the measured cost of
`entity_planner/proportional` over the full horizon. It puts **10.1 (screen) at ≈1,600 CPU-hours**
(about 100 h wall on a cpu16 pod, 50 h on cpu32) and **12.1 (report) at ≈910 CPU-hours** (57 h /
28 h). Varying the member mix, the regime and the horizon extrapolation within what the data
supports gives **≈1,420–2,690 CPU-hours for 10.1** and **≈810–1,500 for 12.1** (section 4).
Pod CPU speed and contention between workers are not measured and come on top of that range. Both
runs are feasible but expensive. 120-producer joint-stack jobs are 95–98% of the cost.

## 1. Per-episode timing at h = 12 (one episode per family × size × contract)

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
= 60.0 s (sparse). The two-point power fit of section 2: 47.41 × 1.5^1.156 = 75.8 s and
40.00 × 1.5^1.131 = 63.3 s.

## 2. Horizon scaling (fit of t = c·h^b, least squares in log-log)

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
tell whether that is a trend or noise. The code bounds the hot loop: section 5 shows it scans the
telemetry's EMA keys, one per (producer, product) pair that has seen an event, not the cycle
history. That finite key universe guarantees per-cycle cost eventually stops growing. It does not
show *when*: key counts were not measured, so it cannot be claimed that the plateau is reached
within the measured cycles. This **corrects my task-6.1 note**, which said cost grew with the cycle
history; it grows with the key set, which is bounded. To expose the uncertainty, section 4
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

## 3. Job counts (from `eval/registry/worlds_v1.json` 9ec52c4d… and `eval/registry/roster_v1.json` 8791e24b…)

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

## 4. Extrapolated wall time and vCPU-hours (h = 100, one worker per vCPU, BLAS 1 thread)

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
| horizon growth | marginal per-cycle cost keeps rising (section 2; dense ×1.41, sparse ×1.10) | 2,072 | 1,171 |
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

## 5. Feasibility verdict and cause

**Both runs are feasible but expensive.** 10.1 is ≈1,600 vCPU-hours in the reference scenario
(≈1,420–2,690 across section 4's scenarios): about 3.7–7 days on cpu16 or 1.9–3.5 days on cpu32. 12.1 is
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

## 6. Dry runs (`--dry-run`, baseline subject, single worker, temp dirs under `/tmp/gate_dry/`)

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
   candidate on `dev-dense-120-calm-00` take more than 2 min per command (about 53 s each, section 1).
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

## Raw data

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
