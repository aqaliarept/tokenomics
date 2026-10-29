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
