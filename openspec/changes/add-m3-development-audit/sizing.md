# Sizing (add-m3-development-audit)

Timing probes that size this change's pod runs (design D8). Each section is written by the task
that measures it.

## §EXP-0048 (task 2.1)

**Question.** How long will EXP-0048 take on a `cpu16` pod: 128 registry v2 development
world-episodes × 69 runs (9 deterministic + 12 stochastic routers × R = 5) = 8,832 horizon-100
episodes, routing tracer on?

**Local box.** 2 vCPU AMD EPYC-Rome, 3.8 GB, no swap; one process at a time, `--workers 1`.
Load average 0.47 before and 1.15 after the 18 traced commands; 0.18 before and 1.18 after the
EXP-0047 sample below (the measuring process itself is one core of load).

**Per-episode times.** From `/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/trace_21.py` (routing_trace.md; raw JSON under
`/tmp/m3a12/trace/` is ephemeral, rerun to regenerate): the driver's job
(`entity_planner` + router, `joint_stack_env`, ContractMonitor on), horizon 100, timed twice
without and twice with the tracer, minimum taken. Each command (logging run + 4 timed runs)
took 4–5 s on the 12-producer world and 22–25 s on the 42-producer world, all under 2 min.

| world | plain s/episode (mean over 9 routers) | traced s/episode (mean; min–max) | tracer overhead (traced / plain) |
|---|---|---|---|
| `dev-sparse_wide-12-calm-03` | 0.774 | 0.788 (0.711–0.861) | 1.018 (per router 0.97–1.08) |
| `dev-sparse_wide-42-shocked-00` | 4.479 | 4.516 (4.275–4.872) | 1.008 (per router 0.995–1.02) |

The tracer overhead is about 1–2%, inside the run-to-run noise at 0.7 s (D2 assumed ≤ 1.5).

**Local-to-pod factor.** EXP-0047's 3.7 vCPU-h (12,800 episodes, ≈ 1.04 vCPU-s each) cannot be
reused per episode: its mix had 53% early deaths (6,740 of 12,800 rows), which are cheap, while
every one of the 18 traced EXP-0048-style episodes survived to horizon 100. Instead, EXP-0047's
own job mix (draft roster v2, 48 members, R = 5: 100 runs per world-episode) was timed locally
through `gate_cli.run_job` on the same two world-episodes (`/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/exp47_sample.py`, run from
`/srv/research/tokenomics/tokenomics` as `../.venv/bin/python /srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/exp47_sample.py --world
dev-sparse_wide-12-calm-03 --part 1/1 --out /tmp/m3a12/exp47_sample.jsonl` and, for
`dev-sparse_wide-42-shocked-00`, `--part 1/3`, `2/3`, `3/3`: 4 commands of 59–92 s; 200 runs,
0 contract failures, units equal to EXP-0047's rows 200 of 200; the JSONL output is ephemeral):

| size, outcome | local runs | local s/episode | EXP-0047 rows |
|---|---|---|---|
| 12, survived | 55 | 0.685 | 3,489 |
| 12, died | 45 | 0.456 | 2,911 |
| 42, survived | 39 | 3.861 | 2,571 |
| 42, died | 61 | 1.823 | 3,829 |

EXP-0047 at local speed, weighting each (size, outcome) time by its EXP-0047 row count:
20,625 s = 5.73 vCPU-h. The pod executed it in 3.7 vCPU-h, so **pod vCPU-s = 0.646 × local s**
(this includes the pool's idle tail, since 3.7 vCPU-h is 16 vCPU × the 14.0 min execution).

**Extrapolation.** 4,416 episodes per size (64 world-episodes × 69 runs), at the traced local
time of the measured world of that size, × 0.646:

| estimate | 12-producer local s | 42-producer local s | local vCPU-h | pod vCPU-h | `cpu16` wall |
|---|---|---|---|---|---|
| mean router time | 4,416 × 0.788 = 3,480 | 4,416 × 4.516 = 19,943 | 6.51 | **4.20** | **15.8 min** |
| slowest router time | 4,416 × 0.861 = 3,802 | 4,416 × 4.872 = 21,515 | 7.03 | 4.54 | 17.0 min |

Billed, with pod start-up and pull (EXP-0047: 4.0 billed for 3.7 executed): about 4.5–4.9
vCPU-h, 17–19 min of pod life. For comparison, scaling EXP-0047's per-episode rate (1.04
vCPU-s × 8,832 × 1.02 overhead) gives 2.6 vCPU-h; that understates, because of EXP-0047's early
deaths.

**Against the plan.** Above tasks.md 3.1's expected 2.5–4 vCPU-h by about 0.2–0.5 vCPU-h, well
under the **8 vCPU-h hard stop**, so task 3.1 can launch as planned (`cpu16`, `--workers 16`).

**Uncertainties.** One world per size was timed: 42-calm and 12-shocked worlds may cost
somewhat differently; an EXP-0048 episode that dies early is cheaper (none of the 18 did). 9 of
the 21 routers were timed; the other 12 are `random` variants or `blend_0.5` plus noise and do
the same work per decision. `entity_planner/obs_price` (legacy contract, 1 of 69 runs) was not
timed. The factor comes from two world-episodes of EXP-0047's mix, not all 128.

<details><summary>Per-router timing (seconds per horizon-100 episode, min of 2 runs)</summary>

| world | router | plain | traced | overhead | logged (warm-up) |
|---|---|---|---|---|---|
| dev-sparse_wide-42-shocked-00 | even_split | 4.26 | 4.28 | 1.004 | 4.69 |
| dev-sparse_wide-42-shocked-00 | proportional | 4.40 | 4.49 | 1.018 | 4.77 |
| dev-sparse_wide-42-shocked-00 | blend_0.5 | 4.49 | 4.49 | 0.999 | 4.96 |
| dev-sparse_wide-42-shocked-00 | random | 4.78 | 4.87 | 1.018 | 5.07 |
| dev-sparse_wide-42-shocked-00 | round_robin | 4.48 | 4.53 | 1.011 | 4.79 |
| dev-sparse_wide-42-shocked-00 | even_split_rotating | 4.55 | 4.53 | 0.995 | 4.86 |
| dev-sparse_wide-42-shocked-00 | proportional_load_aware | 4.48 | 4.55 | 1.017 | 4.91 |
| dev-sparse_wide-42-shocked-00 | proportional_whole | 4.44 | 4.45 | 1.003 | 4.95 |
| dev-sparse_wide-42-shocked-00 | random_whole | 4.42 | 4.47 | 1.010 | 4.82 |
| dev-sparse_wide-12-calm-03 | even_split | 0.73 | 0.71 | 0.974 | 0.85 |
| dev-sparse_wide-12-calm-03 | proportional | 0.77 | 0.75 | 0.982 | 0.94 |
| dev-sparse_wide-12-calm-03 | blend_0.5 | 0.80 | 0.83 | 1.042 | 0.91 |
| dev-sparse_wide-12-calm-03 | random | 0.88 | 0.86 | 0.977 | 0.97 |
| dev-sparse_wide-12-calm-03 | round_robin | 0.73 | 0.77 | 1.056 | 0.89 |
| dev-sparse_wide-12-calm-03 | even_split_rotating | 0.76 | 0.81 | 1.076 | 0.93 |
| dev-sparse_wide-12-calm-03 | proportional_load_aware | 0.81 | 0.78 | 0.967 | 0.92 |
| dev-sparse_wide-12-calm-03 | proportional_whole | 0.74 | 0.80 | 1.073 | 0.86 |
| dev-sparse_wide-12-calm-03 | random_whole | 0.75 | 0.77 | 1.024 | 0.86 |

</details>

## §EXP-0049 (task 3.2)

**Question.** How long will EXP-0049 take on a `cpu16` pod: `routing_audit --plan exp0049`,
128 registry v2 development world-episodes × 31 runs = 3,968 horizon-100 episodes, tracer on?
Per world-episode: the six roster v2 members (5 deterministic + `random` × 5 = 10 runs), the
references `round_robin`, `even_split_rotating`, `proportional_whole` (1 each) and
`random_whole` × 5, the informed routers `two_choice_spare` × 5, `spare_weighted_whole` × 5 and
`spare_rotating_whole` × 1 (29 router runs), and the two trained subjects,
`entity_planner/trained_orderer` and `portable`, once each (deterministic).

**Local box.** As for §EXP-0048; one process, `--workers 1`. Load average 0.39 before the first
command and 0.99 after the last.

**Per-episode times.** `/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/timing_32.py`, run from
`/srv/research/tokenomics/tokenomics` (e.g. `../.venv/bin/python
../openspec/changes/add-m3-development-audit/probes/timing_32.py --world dev-sparse_wide-42-shocked-00 --routers
trained_orderer,portable`). It times the driver's own job (`routing_audit.run_job`: tracer and
contract monitor on) for one horizon-100 episode per arm, replicate 0, first episode seed of
the world; the checkpoint load (once per worker process) is timed apart. Three commands: 9.6 s
(12, all six arms), 19.4 s (42, routers), 22.3 s (42, trained); all episodes survived to
horizon 100, 0 contract failures.

| arm | `dev-sparse_wide-12-shocked-00` s | `dev-sparse_wide-42-shocked-00` s |
|---|---|---|
| `round_robin` (reference) | 0.822 | 4.795 |
| `two_choice_spare` | 0.807 | 4.572 |
| `spare_weighted_whole` | 0.807 | 4.938 |
| `spare_rotating_whole` | 0.837 | 4.766 |
| `entity_planner/trained_orderer` | 1.733 | 8.820 |
| `portable` | 2.556 | 11.389 |
| checkpoint load (both, per process) | 1.23 | 1.29 |

The informed routers cost what `round_robin` costs: reading spare capacity is a few integer
operations per eligible producer. `round_robin` here is within 6% of its §EXP-0048 times
(0.77 at 12-calm, 4.53 at 42-shocked). The trained orderer is about 2× a router episode (torch
forward per stream decision), the full portable stack 2.4–3.1×.

**Extrapolation.** Per size, 64 world-episodes: 1,856 router episodes at the mean of the four
measured router times (all other routers are `round_robin`-like or roster members, whose times
§EXP-0048 measured within the same range), plus 64 episodes of each trained subject; × 0.646
(§EXP-0048's local-to-pod factor):

| estimate | 12-producer local s | 42-producer local s | local vCPU-h | pod vCPU-h | `cpu16` wall |
|---|---|---|---|---|---|
| mean router time | 1,856 × 0.818 + 64 × (1.733 + 2.556) = 1,793 | 1,856 × 4.768 + 64 × (8.820 + 11.389) = 10,142 | 3.32 | **2.15** | **8.0 min** |
| slowest router time | 1,856 × 0.837 + 275 = 1,828 | 1,856 × 4.938 + 1,293 = 10,458 | 3.42 | 2.21 | 8.3 min |

The checkpoint loads add 16 × 1.3 s. Billed, with pod start-up, install and pull (EXP-0048:
4.4 billed for 4.1 executed): about 2.5–3 vCPU-h, 10–12 min of pod life. Inside the expected
1–3 vCPU-h and well under the **6 vCPU-h hard stop**: task 3.3 can launch as planned (`cpu16`,
`--workers 16`).

**Joining EXP-0048's reference rows instead.** Skipping the 18 reference runs per world-episode
(roster members and the four references) would save 64 × 18 × (0.818 + 4.768) × 0.646 / 3,600 ≈
1.15 vCPU-h. Not taken (design D2, "Informed spreading routers and trained-agent trace"): those
rows lack the receiver-variation and herding columns both questions read, and rerunning them
makes every reference row a reproduction check against EXP-0048.

**Uncertainties.** One shocked world per size was timed; calm worlds and early deaths change the
mix slightly (§EXP-0048: 12-calm 0.79 s against 12-shocked 0.82 s here). The trained subjects'
time depends on how many stream decisions an episode has; one episode per size was timed. The
0.646 factor comes from EXP-0047's job mix (§EXP-0048).

## §EXP-0050 (task 3.4)

**Question.** How long will EXP-0050 take on a `cpu16` pod: `routing_audit --plan exp0050`,
8 cell types (sparse_wide 12 × fanout 2 / 4, sparse_wide 42 × fanout 2 / 4 / 7 / 14, dense 12
and 42) × calm / shocked, 31 runs per world-episode (29 router runs: the six roster members
with `random` × 5, `round_robin`, `random_whole` × 5, `proportional_whole`,
`even_split_rotating`, `two_choice_spare` × 5, `spare_weighted_whole` × 5,
`spare_rotating_whole`; plus `entity_planner/trained_orderer` and `portable` once each)?

**Local box.** As for §EXP-0048; one process, `--workers 1`; load average 0.20 before the first
command, 1.13 after the last.

**Per-episode times.** `/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/timing_34.py`, run from
`/srv/research/tokenomics/tokenomics` (e.g. `../.venv/bin/python
../openspec/changes/add-m3-development-audit/probes/timing_34.py --cell sparse_wide,42,14,shocked --routers
round_robin,two_choice_spare,trained_orderer,portable`): the driver's own job (tracer and
contract monitor on), one horizon-100 episode per arm on world 0 of each cell, shocked
(`shocks-i`), first episode seed, replicate 0; checkpoint load (1.3–1.4 s per process) timed
apart. 8 commands of 8–119 s (the slowest, sparse_wide 42 f14, at 118.9 s), all under 2 min;
every episode survived to horizon 100, 0 contract failures. Seconds per episode:

| cell | `round_robin` | `two_choice_spare` | `trained_orderer` | `portable` | per world-episode (local s) | per cell, 12 world-episodes |
|---|---|---|---|---|---|---|
| sparse_wide 12 f2 | 0.83 | 0.82 | 1.72 | 2.61 | 29 × 0.825 + 1.72 + 2.61 = 28.2 | 339 |
| sparse_wide 12 f4 | 1.16 | 1.18 | 2.59 | 3.63 | 29 × 1.173 + 2.59 + 3.63 = 40.2 | 483 |
| dense 12 (f4) | 1.27 | 1.30 | 2.90 | 3.87 | 29 × 1.281 + 2.90 + 3.87 = 43.9 | 527 |
| sparse_wide 42 f2 | 4.54 | 4.54 | 8.70 | 11.44 | 29 × 4.544 + 8.70 + 11.44 = 151.9 | 1,823 |
| sparse_wide 42 f4 | 6.76 | 6.61 | 13.12 | 17.21 | 29 × 6.688 + 13.12 + 17.21 = 224.3 | 2,691 |
| sparse_wide 42 f7 | 9.67 | 9.84 | 19.55 | 23.89 | 29 × 9.755 + 19.55 + 23.89 = 326.3 | 3,916 |
| sparse_wide 42 f14 | 18.11 | 18.89 | 36.12 | 43.73 | 29 × 18.502 + 36.12 + 43.73 = 616.4 | 7,397 |
| dense 42 (f14) | 7.00 | 7.19 | 14.51 | 18.45 | 29 × 7.095 + 14.51 + 18.45 = 238.7 | 2,864 |

The router-run time is the mean of `round_robin` and `two_choice_spare` (the other routers and
roster members do the same work per decision; §EXP-0048's traced times for nine of them span
0.71–0.86 s at 12 and 4.28–4.87 s at 42 producers).
Episode cost grows with fanout at 42 producers (sparse_wide f2 → f14: 4.5 → 18.5 s; not
profiled, presumably because each routing decision sees more eligible producers); dense 42,
with 4 products per layer instead of 14, costs 7.1 s.

**Grid choice and extrapolation.** The first sizing, 2 worlds × (3 calm + 5 shocked episodes)
per cell (16 world-episodes), extrapolates to 26,721 local s = 7.42 vCPU-h × 0.646 = **4.79 pod
vCPU-h**: under the 6 vCPU-h stop but above the 2–4 target. Calm episodes of one world are
redundant for deterministic arms (calm demand is fixed per world: EXP-0049, 176 of 240 calm
subject-world pairs give identical units and digests across their 3 episodes, and every
differing pair is a stochastic arm, which has its own 5 replicates). So the grid keeps the
budget for more worlds instead: **4 worlds per cell × regime, 1 calm + 2 shocked episodes per
world** = 12 world-episodes per cell, 96 in all, 2,976 episodes.

| estimate | local s | local vCPU-h | pod vCPU-h (× 0.646) | `cpu16` wall |
|---|---|---|---|---|
| 4 worlds × (1 calm + 2 shocked) per cell | 20,040 | 5.57 | **3.60** | **13.5 min** |

Billed, with pod start-up, install and pull (EXP-0049: 2.50 billed for 2.25 executed): about
4.0 vCPU-h. Inside the 2–4 vCPU-h target and under the **6 vCPU-h hard stop**: task 3.5 can
launch (`cpu16`, `--workers 16`). The longest single job (sparse_wide 42 f14 `portable`, ≈ 28 s
on the pod) is far below the run's wall, so the pool's tail stays short.

**Power of this grid** (design D2 "Routing versus producers per product", *Power and
limitations*). With 4 worlds per cell × regime the minimum detectable paired difference (80%
power, two-sided 5%, 4.16 × the world-clustered SE estimated from EXP-0049's episodes
subsampled to 4 worlds and 1 calm / 2 shocked episodes) is, for `round_robin` − `random`, 45 /
18 / 38 / 30 units (12-calm / 12-shocked / 42-calm / 42-shocked), and for `spare_rotating_whole`
− `round_robin` 14 / 3 / 37 / 25. Calm cells are underpowered for EXP-0049-sized effects (+26 /
+31 on calm worlds); doubling the calm worlds (8 per cell, 1 episode each) would cost about
+1.2 pod vCPU-h (to ≈ 4.8) and cut the calm SE by about √2. Not done: it is above the 2–4 vCPU-h
target, and the shocked cells carry the comparison; the supervisor may choose it.

**Uncertainties.** Only shocked worlds were timed; calm worlds have no shocks and fixed demand
and may cost a little differently. One world per cell, one episode per arm. The 0.646 factor
comes from EXP-0047's job mix at fanout 2; wider worlds may scale differently on the pod's CPUs.

## §EXP-0051 (task 4.6)

**Question.** How long will the atlas screen take on a `cpu16` pod (`python -m
tokenomics.eval.atlas --mode screen`, design D12): a world-space sample of 1,024 worlds at 12
producers and 128 at 42, 1 episode per world, 11 router jobs per world-episode (the six roster v2
members with `random` × 1, `round_robin`, `random_whole` × 2, `makeable_split`,
`makeable_weighted_whole` × 1) plus `entity_planner/trained_orderer` (13 router-equivalents),
plus the 8 EXP-0050 anchors? And does the sample cover the lever space it should?

**Local box.** 2 CPUs, `--workers 1`, one arm per command; load average 0.3 before the first
command, about 1.0 after the last. Every command under 2 min (the longest 94 s); no
horizon-50 fallback was needed.

### (a) Calibration (no episodes)

`/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/calibrate_46.py
--size 12|42 --loads` builds every screen config with its D10 world record (measured
coordinates; data in `probes/calibrate_46.jsonl`). Requirement: per size, ρ and σ each span at
least [0.05, 1.5], and every categorical level is present.

With D10's initial capacity scale c ∈ [0.25, 1], σ (mean order size against the median
declarer capacity) did not reach the low end: its minimum was 0.058 at 12 and 0.071 at 42, so no
world had orders small against capacity. In-process trials: c ∈ [0.25, 1.5] gave a σ minimum of
0.041 / 0.049 (42 barely short), c ∈ [0.25, 2] 0.032 / 0.038. **c's range is now [0.25, 2]**
(log-uniform; 2 is `families.check_capacity_scale`'s own limit; neutral 1 unchanged) in
`/srv/research/tokenomics/tokenomics/tokenomics/eval/world_space.py` `LEVERS`, recorded in
ADR-0057's lever table and calibration note. `tests/test_world_space.py` (34 tests, including
the lever corners with c = 2 and the all-high corner, contract monitor on, `round_robin` and the
trained orderer) passes; the full suite is green.

Coverage on the final ranges (min / median / max; level counts):

| size | worlds | ρ | σ | cv capacity | cv degradation | cv cost | shock events / producer | demand concentration | demand volatility |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 1,024 | 0.024 / 0.52 / 14.6 | 0.032 / 0.34 / 10.1 | 0.004 / 0.44 / 1.13 | 0 / 0.23 / 0.61 | 0 / 0.21 / 0.52 | 0 / 0.08 / 0.75 | 0.25 / 0.27 / 1.0 | 0 / 0.17 / 1.04 |
| 42 | 128 | 0.028 / 0.45 / 13.0 | 0.038 / 0.35 / 7.1 | 0.008 / 0.50 / 0.89 | 0 / 0.28 / 0.59 | 0.007 / 0.24 / 0.51 | 0 / 0.07 / 0.74 | 0.074 / 0.089 / 0.32 | 0.029 / 0.092 / 0.46 |

| size | choice | shocks off / on | pattern (cyclic, iid, drift, burst, hit, seasonal, switch) |
|---|---|---|---|
| 12 | sparse_f2 342, sparse_f4 341, dense 341 | 512 / 512 | 147, 146, 146, 147, 146, 146, 146 |
| 42 | sparse_f2 26, f4 26, f7 25, f14 26, dense 25 | 64 / 64 | 19, 18, 18, 19, 18, 18, 18 |

Both sizes pass: ρ and σ span [0.05, 1.5] with room on both sides, and every level is present.
The bound (B2 + B_base, the env reset for the episode's demand included) costs 9 ms per
world-episode at 12 (max 81 ms) and 33 ms at 42 (max 198 ms) at horizon 100: about 10 s for the
whole 12-producer sample, negligible.

### (b) Timing

`probes/timing_46.py --size S --choice C --arm A --worlds ...` times one horizon-100 episode per
world through the atlas's own job (`atlas.run_atlas_job`: tracer, contract monitor and bound
columns on), replicate 0, the screen's first episode seed; checkpoint load (1.2–1.9 s per
process) timed apart. Per size × choice level: the `neutral` world (every other lever neutral:
the family world, calm, cyclic) and the 3 sample worlds of that level at the 50 / 75 / 95%
quantiles of load = k_c × m_q (more consumers, larger orders). 34 commands, 134 episodes (raw
lines in `probes/timing_46.jsonl`), every episode survived to horizon 100, no contract failure.
Seconds per episode:

| size, choice | world | id | load (k_c × m_q) | c | shocks, pattern | `round_robin` | `proportional` | `trained_orderer` | `portable` |
|---|---|---|---|---|---|---|---|---|---|
| 12 sparse_f2 | neutral | ws-neutral-12-0000 | 1.00 | 1.00 | off, cyclic | 0.78 | 0.72 | 1.68 | 2.58 |
| 12 sparse_f2 | q50 | 12-0648 | 5.49 (0.87 × 6.28) | 0.79 | off, iid | 0.85 | 0.84 | 1.99 | 2.73 |
| 12 sparse_f2 | q75 | 12-0396 | 11.90 (2.02 × 5.89) | 0.41 | off, cyclic | 1.14 | 1.15 | 2.33 | 3.13 |
| 12 sparse_f2 | q95 | 12-0846 | 35.18 (6.63 × 5.31) | 0.96 | off, hit | 2.47 | 2.50 | 5.84 | 6.80 |
| 12 sparse_f4 | neutral | ws-neutral-12-0000 | 1.00 | 1.00 | off, cyclic | 1.13 | 1.08 | 2.55 | 3.55 |
| 12 sparse_f4 | q50 | 12-0912 | 5.88 (1.63 × 3.62) | 0.38 | off, drift | 1.46 | 1.82 | 3.31 | 4.29 |
| 12 sparse_f4 | q75 | 12-0681 | 11.92 (1.68 × 7.11) | 0.40 | off, seasonal | 1.58 | 1.89 | 3.29 | 4.25 |
| 12 sparse_f4 | q95 | 12-0885 | 32.72 (6.25 × 5.24) | 1.85 | on, switch | 2.96 | 3.49 | 6.72 | 8.00 |
| 12 dense | neutral | ws-neutral-12-0000 | 1.00 | 1.00 | off, cyclic | 1.13 | 1.10 | 2.67 | 3.82 |
| 12 dense | q50 | 12-0629 | 5.62 (3.38 × 1.66) | 0.67 | off, hit | 1.91 | 2.15 | 4.93 | 5.84 |
| 12 dense | q75 | 12-0475 | 11.03 (2.77 × 3.98) | 0.59 | off, burst | 1.91 | 2.39 | 4.61 | 5.54 |
| 12 dense | q95 | 12-0649 | 30.18 (7.65 × 3.94) | 0.72 | on, cyclic | 3.13 | 3.83 | 7.25 | 8.43 |
| 42 sparse_f2 | neutral | ws-neutral-42-0000 | 1.00 | 1.00 | off, cyclic | 4.49 | 4.35 | 8.45 | 11.63 |
| 42 sparse_f2 | q50 | 42-0040 | 6.79 (2.02 × 3.36) | 0.93 | on, burst | 6.04 | 6.08 | 10.88 | 13.85 |
| 42 sparse_f2 | q75 | 42-0074 | 12.67 (4.31 × 2.94) | 0.31 | off, hit | 11.42 | 11.62 | 23.19 | 24.57 |
| 42 sparse_f2 | q95 | 42-0057 | 30.55 (6.89 × 4.43) | 0.48 | off, seasonal | 11.23 | 11.60 | 22.02 | 25.38 |
| 42 sparse_f4 | neutral | ws-neutral-42-0000 | 1.00 | 1.00 | off, cyclic | 6.52 | 6.36 | 13.09 | 16.43 |
| 42 sparse_f4 | q50 | 42-0117 | 3.84 (0.61 × 6.30) | 1.38 | off, switch | 6.40 | 7.04 | 13.13 | 16.93 |
| 42 sparse_f4 | q75 | 42-0086 | 13.29 (3.17 × 4.19) | 0.57 | on, iid | 10.01 | 11.61 | 19.50 | 22.78 |
| 42 sparse_f4 | q95 | 42-0039 | 37.70 (7.26 × 5.19) | 1.98 | off, iid | 14.68 | 18.32 | 28.00 | 31.41 |
| 42 sparse_f7 | neutral | ws-neutral-42-0000 | 1.00 | 1.00 | off, cyclic | 9.41 | 10.09 | 19.53 | 24.43 |
| 42 sparse_f7 | q50 | 42-0095 | 6.27 (1.27 × 4.92) | 0.42 | off, drift | 11.89 | 14.43 | 23.56 | 28.30 |
| 42 sparse_f7 | q75 | 42-0110 | 9.97 (3.42 × 2.91) | 0.52 | on, cyclic | 15.60 | 19.38 | 29.52 | 33.95 |
| 42 sparse_f7 | q95 | 42-0107 | 32.95 (6.50 × 5.07) | 0.44 | off, burst | 19.91 | 29.86 | 42.14 | 47.05 |
| 42 sparse_f14 | neutral | ws-neutral-42-0000 | 1.00 | 1.00 | off, cyclic | 17.87 | 17.89 | 35.00 | 42.88 |
| 42 sparse_f14 | q50 | 42-0054 | 4.68 (3.77 × 1.24) | 0.80 | on, cyclic | 23.66 | 27.82 | 43.33 | 50.66 |
| 42 sparse_f14 | q75 | 42-0084 | 11.10 (2.08 × 5.34) | 1.73 | off, cyclic | 22.56 | 34.66 | 44.14 | 52.78 |
| 42 sparse_f14 | q95 | 42-0064 | 29.41 (5.28 × 5.57) | 1.00 | on, iid | 29.01 | 52.97 | 61.34 | 71.56 |
| 42 dense | neutral | ws-neutral-42-0000 | 1.00 | 1.00 | off, cyclic | 6.57 | 6.58 | 14.14 | 17.22 |
| 42 dense | q50 | 42-0097 | 6.06 (3.23 × 1.88) | 0.38 | on, drift | 12.64 | 17.70 | 24.21 | 27.27 |
| 42 dense | q75 | 42-0102 | 13.75 (5.13 × 2.68) | 1.71 | on, burst | 16.02 | 25.67 | 32.56 | 35.91 |
| 42 dense | q95 | 42-0062 | 26.62 (7.77 × 3.42) | 1.47 | off, hit | 21.43 | 33.94 | 44.93 | 47.87 |

The neutral times match EXP-0050's (e.g. 42 sparse_f14 `round_robin` 17.9 s here, 18.1 s
there). Load makes episodes slower, and makes splitting routers slower than whole-order ones:
on loaded 42-producer worlds `proportional` costs up to 1.8 × `round_robin` (42 f14 q95: 53.0 against
29.0 s). Two extra checks of the deciding-check routers: `makeable_split` 17.7 / 42.6 s on 42 f14
neutral / q95 (between the two), `makeable_weighted_whole` 1.21–3.25 s on 12 dense (like
`round_robin`).

**Cost model.** Per world: a splitting router (the five deterministic roster rules, `random`,
`makeable_split`) costs what `proportional` costs, a whole-order router (`round_robin`,
`random_whole`, `makeable_weighted_whole`) what `round_robin` costs. Screen world-episode = 7 ×
`proportional` + 4 × `round_robin` + `trained_orderer`; the bound adds 0.01–0.03 s (ignored).

**Load factor L** = the sampled mix's cost ÷ the neutral world's cost, per size × choice level.
The mix cost is the mean of the three load-quantile worlds (q50, q75, q95). By load alone that
would overstate the mix (the cheaper lower half of each level's load distribution is not timed),
but the other levers (c, shocks, pattern) move cost about as much as load does, and a random-world
check puts the mix **about central**: the reviewer timed 5 random draws from the 42-producer
sample (`random.Random(46)`, horizon 100, `atlas.run_atlas_job`, the same 7 × `proportional` +
4 × `round_robin` proxy) against their level's mix: 42-0019 f14 0.92, 42-0102 dense 1.00,
42-0010 f7 1.02, 42-0058 f4 0.68, 42-0038 f14 1.10; mean **0.95** (0.68–1.10). As a second
check, a linear fit of cost on load through the 4 timed worlds, evaluated at the level's mean
load (8.8–10.1), gives a lower L (about 0.78 × the mix); it would underestimate the f14 worlds,
which dominate the 42-producer cost. Arithmetic in `probes/extrapolate_46.py`
(`python3 extrapolate_46.py [--screen-12 N]`):

| size, choice | screen s per world-episode: neutral / q50 / q75 / q95 | mix | L | L (fit check) | worlds | local s |
|---|---|---|---|---|---|---|
| 12 sparse_f2 | 9.9 / 11.3 / 14.9 / 33.2 | 19.8 | 2.01 | 1.48 | 341.3 | 6,758 |
| 12 sparse_f4 | 14.6 / 21.9 / 22.8 / 43.0 | 29.2 | 2.00 | 1.55 | 341.3 | 9,980 |
| 12 dense | 14.9 / 27.6 / 29.0 / 46.6 | 34.4 | 2.31 | 1.77 | 341.3 | 11,742 |
| 42 sparse_f2 | 56.8 / 77.6 / 150.2 / 148.1 | 125.3 | 2.21 | 1.72 | 25.6 | 3,208 |
| 42 sparse_f4 | 83.7 / 88.0 / 140.8 / 215.0 | 147.9 | 1.77 | 1.39 | 25.6 | 3,786 |
| 42 sparse_f7 | 127.8 / 172.1 / 227.6 / 330.8 | 243.5 | 1.91 | 1.56 | 25.6 | 6,234 |
| 42 sparse_f14 | 231.7 / 332.7 / 377.0 / 548.2 | 419.3 | 1.81 | 1.52 | 25.6 | 10,734 |
| 42 dense | 86.5 / 198.7 / 276.3 / 368.2 | 281.1 | 3.25 | 2.40 | 25.6 | 7,195 |

L lies at 1.8–3.3 (fit check 1.4–2.4), above D12's provisional [1, 2] in places: the loaded dense
42 worlds cost most relative to neutral (dense neutral has 4 products per layer and is cheap). The
8 anchors add 648 local s (EXP-0050's per-cell times, same job mix).

| screen grid | local s | local vCPU-h | pod vCPU-h (× 0.646) | `cpu16` wall | fit check (pod vCPU-h) |
|---|---|---|---|---|---|
| 1,024 at 12 + 128 at 42 + 8 anchors | 28,480 + 31,157 + 648 = 60,285 | 16.75 | **10.82** | 41 min | 8.49 |
| **512 at 12** + 128 at 42 + 8 anchors | 14,240 + 31,157 + 648 = 46,045 | 12.79 | **8.26** | **31 min** | 6.54 |

**Reduction (D12 order).** The full grid, 10.82 pod vCPU-h (≈ 11.1 billed with pod start-up,
install and pull, as EXP-0049's 2.50 billed for 2.25 executed), is above the **10 vCPU-h hard
stop**. Step 1 of D12's reduction order, 512 worlds at 12, brings it to **8.26** (≈ 8.6 billed;
the fit check gives 6.54): under the stop, and the run can launch. Step 2 (64 worlds at 42) is
not needed. The 512-world sample (a Latin hypercube drawn with n = 512, not the first 512 of
1,024) still covers: ρ 0.027–14.2, σ 0.021–7.3, choice 171 / 171 / 170, shocks 256 / 256,
patterns 73–74 each; its mean load per choice level (8.5–9.0) is no higher than the 1,024
sample's, so the per-world cost above holds. Applied in
`/srv/research/tokenomics/tokenomics/tokenomics/eval/atlas.py` (`SCREEN_WORLDS[12] = 512`; a real
run refuses `--worlds-per-size`). The calibration and timing above were taken on the 1,024-world
sample at 12: both probes now default to `atlas.SCREEN_WORLDS` (512), and reproduce the recorded
lines with `calibrate_46.py --size 12 --worlds 1024` and `timing_46.py --worlds-in-sample 1024`
(the 42-producer lines use the unchanged default, 128).

**Estimator used for the decision.** The run is planned against the mix estimator (mean of the
three load-quantile worlds per level; about central, above): 10.82 for the full grid, 8.26 after
step 1. The linear-fit estimator (8.49 / 6.54) is reported as a check only; it would have let the
full grid pass, but it rests on 4 points per level and a load-only model, and underestimates the
random worlds above. **Margin:** 8.26 is 1.74 vCPU-h (17%) under the stop; the random-world
check's worst ratio, + 10%, gives about 9.1, still under 10. The router pricing is on the safe
side: on 42-0102 (dense q75) the reviewer measured the other splitting routers at `blend_0.5`
1.02, `shock_reactive` 1.01, `even_split` 0.87, `random` 0.76, `obs_price` 0.23 × `proportional`
(mean of the six splitting arms 0.81 ×), so pricing every splitting router at `proportional`
overstates them. (An earlier message to the
supervisor quoted 10.53 / 8.01 and 8.32 / 6.39: those used the plain mean of `round_robin` and
`proportional` for every router; the figures here weight the splitting routers at
`proportional`'s cost, which is higher on loaded worlds.) The 42-producer sample stays at 128
worlds, and EXP-0051's hard stop stays **10 vCPU-h**.

**Final screen grid:** 512 worlds at 12 + 128 at 42 (1 episode each) + 8 anchors, 11 router
jobs + the trained orderer per world-episode, `cpu16`, `--workers 16`: **8.26 pod vCPU-h,
≈ 31 min wall** (≈ 8.6 billed). The longest single job (42 f14 q95 `trained_orderer`, 61 s
local, ≈ 40 s on the pod) is far below the wall, so the pool's tail stays short.

**Provisional spreads for the screen** (D12 "Screen": σ_ep and σ_rep per size × fanout, dense
its own, from EXP-0050's `artifacts/episodes.csv`, computed by `atlas.provisional_spreads`, which
the screen recomputes from `--exp0050-episodes` and records in its `spec.json`). σ_ep: pooled
SD of the paired per-episode gain (trained orderer − best blind arm) over the 4 shocked worlds
per cell, df 4 each. σ_rep: pooled SD between replicates of the stochastic argmax arms; a cell
whose argmax arms were all deterministic has no df and takes its size's pool (the atlas's
recorded driver choice):

| cell | σ_ep (df) | σ_rep (df) |
|---|---|---|
| 12 sparse_f2 | 14.20 (4) | 2.61 (4, size pool) |
| 12 sparse_f4 | 6.28 (4) | 2.61 (4, size pool) |
| 12 dense | 5.89 (4) | 2.61 (4) |
| 42 sparse_f2 | 30.78 (4) | 13.88 (40, size pool) |
| 42 sparse_f4 | 5.76 (4) | 9.67 (4) |
| 42 sparse_f7 | 16.22 (4) | 10.09 (8) |
| 42 sparse_f14 | 36.58 (4) | 21.28 (8) |
| 42 dense | 20.54 (4) | 12.02 (20) |

With the screen's 2 × SE and 1 episode, the floor on a stochastic world is about 2 × σ_ep:
12–28 units at 12, 12–73 at 42.

**Uncertainties.** One episode per timed world and 4 worlds per level; the levers other than
load (c, shocks, pattern) vary between the timed worlds and explain some of the spread (42 f2
q75 costs as much as q95). The 0.646 local-to-pod factor comes from EXP-0047's fanout-2 job
mix. The splitting / whole assignment of routers is approximate (`random` and the roster rules
were not timed here; §EXP-0048 found nine routers within ±10% at neutral load).

## §EXP-0052 (task 4.6)

**Question.** How long will the atlas confirm take (`--mode confirm`, D12): K = 64 selected
worlds at 12 and 32 at 42, 2 episodes per world (1 for cyclic demand without shocks), 13 router
jobs per world-episode (`random`, `random_whole`, `makeable_weighted_whole` at R = 2), the trained
orderer and `portable` (18 router-equivalents)?

**Cost model.** From the same timed worlds (§EXP-0051 table): confirm world-episode = 8 ×
`proportional` (5 roster rules, `random` × 2, `makeable_split`) + 5 × `round_robin`
(`round_robin`, `random_whole` × 2, `makeable_weighted_whole` × 2) + `trained_orderer` +
`portable`. The selected worlds are drawn at the screen's lever distribution (equal choice
shares; the load mix of §EXP-0051), and every world is counted at 2 episodes (the 1-episode
cyclic, shock-free worlds are about 1/14 of a uniform draw; ignoring them is conservative).

| size, choice | confirm s per world-episode: neutral / q50 / q75 / q95 | mix | worlds | local s (× 2 episodes) |
|---|---|---|---|---|
| 12 sparse_f2 | 13.9 / 15.7 / 20.3 / 45.0 | 27.0 | 21.3 | 1,152 |
| 12 sparse_f4 | 20.4 / 29.5 / 30.5 / 57.4 | 39.1 | 21.3 | 1,670 |
| 12 dense | 21.0 / 37.5 / 38.8 / 62.0 | 46.1 | 21.3 | 1,967 |
| 42 sparse_f2 | 77.3 / 103.5 / 197.8 / 196.4 | 165.9 | 6.4 | 2,123 |
| 42 sparse_f4 | 113.0 / 118.4 / 185.2 / 279.4 | 194.3 | 6.4 | 2,487 |
| 42 sparse_f7 | 171.7 / 226.8 / 296.5 / 427.7 | 317.0 | 6.4 | 4,057 |
| 42 sparse_f14 | 310.3 / 434.8 / 487.0 / 701.7 | 541.2 | 6.4 | 6,927 |
| 42 dense | 116.8 / 256.3 / 353.9 / 471.4 | 360.5 | 6.4 | 4,615 |

| confirm grid | local s | local vCPU-h | pod vCPU-h (× 0.646) | `cpu16` wall | fit check (pod vCPU-h) |
|---|---|---|---|---|---|
| 64 at 12 + 32 at 42, 2 episodes | 4,790 + 20,210 = 25,000 | 6.94 | **4.49** | **17 min** | 3.59 |

**Final confirm grid:** as designed, no reduction: **4.49 pod vCPU-h, ≈ 17 min wall**
(≈ 4.8 billed), under the **6 vCPU-h hard stop**. Longest single job: 42 f14 q95 `portable`,
72 s local (≈ 46 s on the pod).

**Uncertainties.** The confirm worlds are the screen's hard and open worlds, not a uniform draw;
if the selection favours loaded worlds the cost rises. Worst case, every selected world at its
level's q95 cost: 64 × 2 × 54.8 + 32 × 2 × 415.3 = 33,594 local s = **6.03** pod vCPU-h, just
above the stop. The selection's actual load mix is known when the screen ends, so EXP-0052 is
**rechecked before launch** against it:
```
python3 /srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/extrapolate_46.py \
  --selection /srv/research/tokenomics/experiments/EXP-0051-<title>
```
It reads the screen's `selection.csv` and the selected worlds' records in its `spec.json`
(k_c, m_q, pattern, shocks), prices each world at its size × choice level's confirm cost,
interpolated piecewise-linearly in load = k_c × m_q through the level's 4 timed worlds (beyond the
highest timed load, extended at the level's least-squares slope), counts 1 episode for cyclic
demand without shocks and 2 otherwise, and prints the pod vCPU-h against the 6 vCPU-h stop.
**If the recheck is above 6, EXP-0052 does not launch and the implementer reports to the
supervisor** (D8): no reduction is pre-registered for the confirm, so any cut (fewer 42-producer
worlds, 1 episode per world, a smaller K) is the supervisor's or the user's decision.

**Budget.** Planned: screen 8.26 + confirm 4.49 = 12.75 pod vCPU-h (≈ 13.4 billed), against
D8's central 11.5 for the two and their stops' sum of 16. With ≈ 10.8 billed so far, M3 would
stand at ≈ 24.2 of 40 after the atlas.

**B_lp in confirm (task 4.7).** Confirm also solves B_lp and B_lp_base once per world-episode
(on the first job of each). Measured in the EXP-0051 re-bound (`probes/rebound_47.jsonl`, both
programs, local box): 0.59 s mean per world-episode at 12 producers and 7.7 s at 42 (max 23 s).
On the 96 selected worlds (122 world-episodes at 12, 63 at 42; cyclic worlds without shocks play
one episode) that is 73 + 597 = 670 local s, **0.12 pod vCPU-h** (× 0.646), so the confirm
recheck becomes 3.91 + 0.12 = **4.03 pod vCPU-h**, still under the 6 vCPU-h stop. The solve sits
on one job per world-episode; the longest such job (42 f14 with an 18.8 s LP) stays far below the
≈ 15 min wall. The pod installs the `lp` extra (scipy) for it.

## §EXP-0053 (task 6.3)

**Question.** How much will the PoC-A teacher audit cost (`eval/teacher_audit.py`, design D5):
12 world-episodes at 12 producers (the pre-registered sample: hard 0060\*, 0176, 0005, 0348; open
0111\*, 0344\*, 0230\*, 0308, 0392, 0254\*, 0155, 0408; \* shocked), horizon 100, arms `floor`, `d2`
and `ub` (`ub` on the 8 worlds with shocks or stochastic demand), plus the 14 blind / reach arms per
world-episode? And which pod shape runs it under the **8 vCPU-h hard stop**?

**Probe.** `probes/timing_63.py` (run from `/srv/research/tokenomics/tokenomics` with
`../.venv/bin/python`; every command < 2 min, `--workers 1`; output in `probes/timing_63.jsonl`)
builds each teacher episode exactly as `teacher_audit.run_teacher_episode` does, but searches only
on chosen cycles (the floor elsewhere) and stops searching after a time budget (the remaining
streams of that cycle are counted, not timed). Seconds per searched stream = the decision's
seconds over its streams (forks and the shared floor rollouts included), so the scenario mix
(streams resolved at 4, 8 or 16 scenarios) is in the measured time, not assumed.

**Timed cells.**

Horizon 12 (`shocks-i-smoke`, the task's cell): 12-0060

| arm | cycles | streams (timed) | s per searched stream | decisions by scenarios used | overrides |
|---|---|---|---|---|---|
| `d2` (posterior) | 0–11 (search capped at 90 s from cycle 4) | 405 (168) | 0.36, 0.40, 0.55, 0.72, 0.90 (cycles 0–4); mean 0.54 | 4: 117, 8: 1, 16: 18 | 2 |
| `ub` (clairvoyant) | 0–11 | 406 (406) | 0.06–0.10; mean 0.09 | 1: 342 | 12 |

Every stream had 4 candidates (E = 4 whole-order receivers + the trained row − the floor's row).
Posterior: 12 of 12 cycles resolved, acceptance rate 1.0, unresolved fraction 0. The per-stream
time grows over cycles 0–4 because more streams need 16 scenarios once shocks are observed, so the
horizon-12 cell understates a horizon-100 episode; the extrapolation uses the horizon-100 cells.

Horizon 100 (registered shock window), every sampled world and arm, search on three cycles of a
played episode (the floor elsewhere; one command per cell, 0408 `d2` one command per cycle):

| world | class | arm (scenarios) | orderer turns per cycle | cycles | streams per cycle | decisions by scenarios used | search s per cycle | mean | search pod min per episode (100 × mean × 0.646) |
|---|---|---|---|---|---|---|---|---|---|
| 0060 | hard | `d2` posterior | 54 | 30, 60, 90 | 21, 27, 25 | 4: 43, 8: 1, 16: 29 | 23.6, 22.9, 25.6 | 24.0 | 25.9 |
| 0060 | | `ub` | 54 | 30, 60, 90 | 21, 27, 22 | 1: 70 | 2.2, 2.9, 2.6 | 2.6 | 2.8 |
| 0176 | hard | `d2` deterministic | 46 | 30, 60, 90 | 18, 16, 15 | 1: 47 | 1.4, 1.3, 1.1 | 1.3 | 1.4 |
| 0005 | hard | `d2` deterministic | 54 | 30, 60, 90 | 23, 17, 18 | 1: 55 | 1.9, 1.4, 1.5 | 1.6 | 1.7 |
| 0348 | hard | `d2` deterministic | 53 | 30, 60, 90 | 11, 10, 10 | 1: 31 | 0.8, 0.7, 0.7 | 0.7 | 0.8 |
| 0111 | open | `d2` posterior | 40 | 10, 50, 80 | 26, 22, 20 | 4: 53, 8: 1, 16: 9 | 8.7, 8.5, 7.8 | 8.3 | 9.0 |
| 0111 | | `ub` | 40 | 30, 60, 90 | 24, 28, 21 | 1: 69 | 1.5, 1.8, 1.4 | 1.6 | 1.7 |
| 0344 | open | `d2` posterior | 62 | 30, 60, 90 | 38, 41, 36 | 4: 74, 8: 3, 16: 23 | 29.4, 25.2, 25.1 | 26.6 | 28.6 |
| 0344 | | `ub` | 62 | 30, 60, 90 | 38, 38, 36 | 1: 100 | 3.5, 3.7, 3.9 | 3.7 | 4.0 |
| 0230 | open | `d2` posterior | 48 | 30, 60, 90 | 22, 16, 21 | 4: 49, 16: 5 | 7.8, 6.0, 10.6 | 8.1 | 8.8 |
| 0230 | | `ub` | 48 | 30, 60, 90 | 22, 16, 21 | 1: 54 | 1.8, 1.2, 1.8 | 1.6 | 1.7 |
| 0308 | open | `d2` posterior | 38 | 30, 60, 90 | 33, 26, 26 | 4: 64, 8: 1, 16: 8 | 10.0, 10.7, 7.3 | 9.3 | 10.0 |
| 0308 | | `ub` | 38 | 30, 60, 90 | 33, 26, 26 | 1: 73 | 1.8, 1.6, 1.6 | 1.7 | 1.8 |
| 0392 | open | `d2` posterior | 58 | 30, 60, 90 | 36, 28, 38 | 4: 69, 16: 17 | 25.4, 31.1, 28.4 | 28.3 | 30.5 |
| 0392 | | `ub` | 58 | 30, 60, 90 | 36, 28, 37 | 1: 86 | 4.4, 3.3, 5.1 | 4.3 | 4.6 |
| 0254 | open | `d2` posterior | 60 | 30, 60, 90 | 37, 47, 40 | 4: 107, 16: 5 | 12.9, 17.4, 14.8 | 15.0 | 16.2 |
| 0254 | | `ub` | 60 | 30, 60, 90 | 37, 47, 40 | 1: 112 | 2.7, 3.6, 3.3 | 3.2 | 3.5 |
| 0155 | open | `d2` deterministic | 28 | 30, 60, 90 | 17, 16, 10 | 1: 38 | 0.6, 0.5, 0.3 | 0.5 | 0.5 |
| 0408 | open | `d2` posterior | 108 | 30, 60, 90 | 45, 46, 51 | 4: 78, 8: 5, 16: 45 | 64.0, 83.7, 79.9 | **75.8** | **81.7** |
| 0408 | | `ub` | 108 | 30, 60, 90 | 45, 41, 47 | 1: 122 | 7.7, 7.5, 9.6 | 8.2 | 8.9 |

Candidates per stream (all cells): 4 on the dense and sparse_f4 worlds (0060, 0176, 0005, 0348,
0230, 0392, 0408: E = 4 whole-order receivers + the trained row − the floor's row) and 2 on the
sparse_f2 worlds (0111, 0344, 0308, 0254, 0155: E = 2 + trained − floor's row). Per stream ≈ 0.3–0.4 s when it stops at 4 scenarios and ≈ 1.4–1.8 s
at 16; per-cycle cost is flat over the history (e.g. 0060: 23.6 / 22.9 / 25.6 s at cycles 30 / 60
/ 90). Posterior acceptance rate 1.0 and no unresolved decision in any cell.

**Why the first version of this section was wrong** (review concern #346). It timed 3 posterior
worlds (0060, 0392, 0111) chosen without a cost driver and priced the other 5 posterior worlds at
their mean (central) or maximum (high). The cost of a world's `d2` episode scales with its orderer
turns per cycle (each floor rollout plays them all), its streams per cycle, the candidates per
stream and the share of streams that need 16 scenarios. 12-0408 (dense, seasonal, k_c 5.53) has
108 orderer turns per cycle against 28–62 elsewhere, 45–51 streams with 4 candidates each (2 on
the sparse_f2 worlds) and 35% of its decisions at 16 scenarios: 75.8 s per cycle, 2.7× the pooled maximum. It is now measured; no world is priced by
a pool.

Posterior sampler at a horizon-100 history (`timing_63.py sampler`, 12-0060, `round_robin` played
with the recorder on): 0.019 / 0.025 / 0.016 s at cycles 30 / 60 / 90 as recorded in `timing_63.jsonl` (16 of 16 accepted; an earlier, unrecorded run gave 0.019 / 0.022 / 0.016); in
the search cells 2–33 ms per cycle. ≤ 3.3 s per episode: negligible. The recorder's replay over
90 cycles: 2.7 s per episode.

0.646 factor check (`timing_63.py rr`): one traced `round_robin` horizon-100 episode on
dev-sparse_wide-12-calm-03 takes 0.82 / 0.83 s (§EXP-0048: 0.73–0.77 s, §EXP-0049 reference
0.82 s): the box runs at the speed the factor was derived at, so pod vCPU-s = 0.646 × local s
stands. On 12-0060: 1.60–1.66 s, 2,429 units (EXP-0052's `round_robin` row reproduced).

**Arithmetic** (`probes/extrapolate_63.py`). Per teacher episode = 100 cycles × the world's own
search s per cycle (central: mean of its three cycles; high: its most expensive cycle) + the
floor-speed episode (1.6 s) + for posterior arms the sampler (100 × 13 ms) and the recorder
(2.7 s). An unmeasured world or arm is an error. Blind / reach reruns at 46.1 s per world-episode
(§EXP-0052's 12-dense confirm figure, 18 router-equivalents with `portable`, which the audit does
not run: conservative). × 0.646 = pod vCPU-h.

| sample | case | `d2` local s | `ub` local s | rest | pod vCPU-h | × 2 | longest job (pod min) |
|---|---|---|---|---|---|---|---|
| full 12, `floor` / `d2` / `ub` | central | 20,019 | 2,698 | 572 | **4.18** | **8.36** | 0408 `d2` 81.7 |
| | high | 22,252 | 3,068 | 572 | 4.65 | 9.29 | 0408 `d2` 90.1 |
| D5 step 1: drop `ub` | central | 20,019 | – | 572 | 3.69 | 7.39 | 0408 `d2` 81.7 |
| D5 steps 1 + 2: 8 world-episodes, no `ub` | central | 10,049 | – | 382 | **1.87** | **3.74** | 0392 `d2` 30.5 |
| | high | 11,009 | – | 382 | 2.04 | 4.09 | 0392 `d2` 33.6 |
| step 2 only (8 world-episodes with `ub`) | central | 10,049 | 1,385 | 382 | 2.12 | 4.24 | 0392 `d2` 30.5 |

Step 2's sample by D5's rule: hard 0060, 0176, 0005, 0348 (all four; 0060 the only shocked one);
open the best 2 shocked by room (0111, 0344) and the best 2 unshocked (0308, 0392). 0408 (open,
room 0.546, the last of the open worlds) is not in it.

**Job granularity and pod shape.** The run is 200 jobs (per world-episode 14 blind / reach arms,
`floor`, `d2`, and `ub` on 8 worlds); all the cost is in the 12 `d2` and 8 `ub` jobs. One episode
is one sequential job and the driver cannot split it, so the wall is at least the longest job and
billed vCPU-h = vCPU × wall. A 32-worker pod is never filled (at most 20 long jobs, and after the
first minutes only a handful still run). Driver order (per world the blind / reach arms, then
`floor`, `d2`, `ub`; the first free worker takes the next job), secure CPU pods, workers = vCPU:

| sample | shape | stop 8 vCPU-h as wall | central wall / billed | × 2 wall / billed | high wall / billed |
|---|---|---|---|---|---|
| full 12 | 32 vCPU | 15 min | 82.0 min / 43.7 | 164 min / 87.5 | 90.4 min / 48.2 |
| full 12 | 16 vCPU | 30 min | 83.9 min / 22.4 | 168 min / 44.7 | 92.5 min / 24.7 |
| full 12 | 8 vCPU | 60 min | 92.7 min / 12.4 | 185 min / 24.7 | 102.6 min / 13.7 |
| full 12 | 4 vCPU | 120 min | 114.9 min / 7.7 | 230 min / 15.3 | 126.9 min / 8.5 |
| step 1 (no `ub`) | 8 vCPU | 60 min | 85.4 min / 11.4 | 171 min / 22.8 | 94.2 min / 12.6 |
| steps 1 + 2 | 16 vCPU | 30 min | 30.9 min / 8.2 | 62 min / 16.5 | 33.9 min / 9.1 |
| **steps 1 + 2** | **8 vCPU** | **60 min** | **31.8 min / 4.2** | **63.6 min / 8.5** | **34.9 min / 4.7** |
| steps 1 + 2 | 4 vCPU | 120 min | 42.1 min / 2.8 | 84.3 min / 5.6 | 45.7 min / 3.0 |
| step 2 only (with `ub`) | 8 vCPU | 60 min | 33.3 min / 4.4 | 66.7 min / 8.9 | 36.7 min / 4.9 |

(Submitting the teacher jobs first, longest first, a driver change not made, would cut the
step-1 + 2 8-vCPU run to 30.5 min / 4.1, × 2 8.15.)

**D5 reduction order.** By D5's compute rule (× 2 of the summed job time), the full sample
(8.36) is above the stop and step 1 (drop `ub`, 7.39) brings it under. On a real pod that is not
enough: 0408's `d2` episode alone takes 81.7 pod min (164 at × 2), so with or without `ub` the
full sample bills 11.4–12.4 vCPU-h on 8 vCPU at central (≥ 22.8 at × 2) and 7.3–7.7 on 4 vCPU
over ~2 h. Step 2 (8 world-episodes) removes 0408; with steps 1 + 2 the longest job is 0392's
`d2` (30.5 pod min) and an 8 vCPU pod runs it in ≈ 32 min for ≈ 4.2 billed vCPU-h (× 2: 64 min,
8.5).

**Result.** After the D5 reduction order (steps 1 and 2: 8 world-episodes, arms `floor` / `d2`,
no `ub`): **1.87 pod vCPU-h compute (× 2: 3.74); on an 8 vCPU secure CPU pod ≈ 32 min wall,
≈ 4.2 billed vCPU-h (× 2: ≈ 64 min, 8.5)**. Against the plan's 8 vCPU-h stop the × 2 billed
figure on the recommended 8 vCPU shape is still just above it (a 4 vCPU pod gives 5.6 at × 2 over
≈ 84 min), so this goes to the supervisor; nothing launches until it decides. Against the 10
vCPU-h stop of the superseded decision below (75 min wall on 8 vCPU) the reduced run fits at × 2.

**Superseded supervisor decision (2026-10-29, taken on the first, wrong version of this section;
kept for the record).** EXP-0053 to run the full pre-registered sample, no reduction, on an 8 vCPU
secure CPU pod, `--workers 8`, hard stop raised from 8 to 10 billed vCPU-h = a 75 min wall on 8
vCPU (the × 2 case was then 72 min, 9.7 vCPU-h); fallback 16 vCPU secure (37.5 min wall), never
community; no driver change. With the measured 0408 the full sample cannot finish inside that
wall even at × 1 (92.7 min central on 8 vCPU). Replaced by the decision below.

**Supervisor decision (2026-10-29, on the measured figures; replaces the superseded one).** EXP-0053
runs **D5 step 2 only: the 8 world-episodes** (hard 0060, 0176, 0005, 0348; open 0111, 0344, 0308,
0392), **with `ub` kept**, arms `floor` / `d2` / `ub` plus the blind and reach reruns, on an **8 vCPU
secure CPU pod, `--workers 8`**, with a **hard stop of 10 billed vCPU-h = a 75 min wall** on that
shape (central 33.3 min / 4.4 billed vCPU-h; × 2 66.7 min / 8.9, inside the stop; high 36.7 min /
4.9). **Deviation from the written reduction order, decided on cost alone before any result was
seen:** D5 drops `ub` first (step 1) and only then reduces the sample (step 2). With step 2
applied the run fits the 10 vCPU-h stop with `ub` kept, and `ub` costs ≈ 0.2–0.5 vCPU-h while it is
the only measure of how much the hidden information is worth (D5's "fork-limited" reading needs
it), so step 1 is skipped. **12-0408 leaves the sample by the step-2 rule** (the last open world by
room, 0.546); it is also the costliest world measured: 108 orderer turns per cycle, 75.8 s of
search per cycle, an 81.7 pod-min `d2` episode. **Why 8 vCPU and not the
preferred larger pods (32 → 16 → 8), a deviation made on cost:** the wall is bounded by the longest
job (0392's `d2`, ≈ 31 pod min) and billed vCPU-h = vCPU × wall, so on this sample 32 vCPU runs
30.7 min for 16.4 billed vCPU-h (× 2 61 min / 32.7) and 16 vCPU 31.0 min for 8.3 (× 2 62 min /
16.5, above the 10 stop), against 8 vCPU's 33.3 min / 4.4 (× 2 8.9): the larger pods buy
≈ 2 min of wall for 2–4× the bill; 16 vCPU stays the fallback. Budget: M3 ≈ 22–25 vCPU-h used +
10 worst case = 35 of 40. Fallback if no 8 vCPU secure pod is available: 16 vCPU secure (the same 10 vCPU-h stop =
a 37.5 min wall; the longest job, 0392's `d2`, takes ≈ 31 pod min at × 1; if the run would exceed
the wall, it reports rather than extends), never community. No driver change.

**Final sample:** 8 world-episodes (4 hard, 4 open; 3 with shocks: 0060, 0111, 0344), the confirm's
first episode seed each, horizon 100; `floor` and `d2` on all 8, `ub` on the 5 with shocks or
stochastic demand (0060, 0111, 0344, 0308, 0392); **2.12 pod vCPU-h compute (× 2: 4.24); 8 vCPU ≈
33 min wall, ≈ 4.4 billed vCPU-h (× 2: ≈ 67 min, 8.9) against the 10 vCPU-h / 75 min stop.**
