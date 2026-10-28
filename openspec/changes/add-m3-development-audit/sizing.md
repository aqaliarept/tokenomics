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
