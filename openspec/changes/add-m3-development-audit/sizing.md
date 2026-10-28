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
