# Sizing for tasks 8.1 (EXP-0046) and 9.1 (EXP-0047) — task 6.2

Timing probe at base commit d7e07c7 (clean working tree when timed), local box: AMD
EPYC-Rome, 2 cores, 3.8 GB, BLAS pinned to 1 thread, one process (`--workers 1` equivalent).
Load average (1 min) during the timings: 0.55–0.86 (each timing waited for load < 1).

## Plain-English summary

The question: how much pod time will the two remaining cloud runs of this change cost, and does
either cross the 20 vCPU-hour stop line? To answer it, one shocked episode and its shock-free
twin were timed for each stack and size the exit run (8.1) plays, plus two roster baselines per
size on a shocked world for the calibration screen (9.1). Every timed cell used the test-only
`shocks-i-smoke` profile, so the shocks actually fire inside the short horizon; the registered
"shocks-i" profile is untouched.

Each cell was played at horizons 4, 8 and 12. Over those horizons time grew in a straight line
with the number of cycles played, and the fixed cost per episode (building the env, rebuilding
the schedule, writing the row) was under 0.07 s. The estimates **assume** a 100-cycle episode
costs the per-cycle cost times 100. That is an assumption, not a measurement: no timing here
goes past cycle 12, and later cycles can cost more or less (inventory, open orders and telemetry
grow; investment and deaths change the trajectory). Direct 100-cycle timings from the earlier
gate sizing, on other worlds, seeds and code (§3), land within 0.4–11% of the extrapolation for
the portable stack and `entity_planner/proportional`; that supports the assumption but does not
bound its error. The flat stack and `shock_reactive` have no 100-cycle anchor at all.

This box is not the pod. Two past pod runs give the conversion: EXP-0045 and EXP-0044
attempt 2 each used about 0.65 pod vCPU-hours per local CPU-hour of the same work (§4). The
estimates below use 0.65 and, as a cautious case, 1.0 (pod no faster than this box). Every
figure below is conditional on these assumptions.

- **8.1 (EXP-0046, cpu8):** 240 jobs (120 shocked episodes + 120 twins). About 0.23 vCPU-hours
  of work (0.35 cautious), i.e. 2–3 minutes on 8 vCPU. With pod start, sync and install it is
  about **0.6–0.8 vCPU-hours billed**; 1.8 if the pod waits 11 minutes for SSH as in EXP-0044.
- **9.1 (EXP-0047, cpu16):** 12,800 roster jobs (128 development gate world-episodes × 100
  runs), plus reserve worlds only if the on-demand screen has to replace a gate world whose
  baseline delivers nothing. Without replacements: about 3.7–4.0 vCPU-hours of work, i.e.
  14–15 minutes on 16 vCPU, about **4.5–5 vCPU-hours billed** (expected). The cautious
  scenarios without replacements reach ≈ 13 billed.
- **Reserves.** The registry holds 104 more development world-episodes in reserve (per size: 4
  calm reserves × 3 episodes + 4 shocked × 10 = 52). Each consulted reserve costs 0.03–0.85
  vCPU-hours at the calibrated speed (a shocked 42-producer reserve is the dearest, §5). If
  every reserve ran, the expected case becomes ≈ 8 billed. Only the most pessimistic
  combination, all 104 reserve world-episodes **and** every job at the costliest measured member
  **and** a pod no faster than this box **and** an 11-minute SSH wait, crosses the line:
  **≈ 21.0 vCPU-hours billed**.

**Decision for the supervisor.** The expected 8.1 and 9.1 costs are far below 20 vCPU-hours, but
one 9.1 stress scenario exceeds it, so this task stops here and reports that; nothing launches
on this report alone (§6). EXP-0044 attempt 2 consulted no reserve on these same calm worlds
(`screening.json` `replacements: []`); a new replacement can only come from a shocked world whose
baseline now delivers nothing under "shocks-i".

Design D13 and the 8.1 / 9.1 Resources lines in `tasks.md` were corrected with this task: they
had sized 9.1 from "EXP-0044 attempt 2 ≈ 11 vCPU-h billed", but attempt 2 (sparse-wide only)
billed 4.1 vCPU-hours and executed in 3.3; the ≈ 10.4–16.6 figures belong to attempt 1 (dense +
sparse-wide). The earlier 12.5 vCPU-h for 9.1 was therefore about 2.5× too high for the
no-replacement case.

## 1. Timed cells (h = 4 / 8 / 12, best of 3 warm runs, seconds of one CPU)

Exit-driver cells run through `eval/shock_contract.run_job` (the same code path as 8.1: config
build, `joint_stack_env` or the champion's product-stream env, `ContractMonitor`, schedule
rebuild, row). Worlds: `dev-sparse_wide-12-shocked-00`, `dev-sparse_wide-42-shocked-00` (first
episode seed each), `flat-scale-d3-s6` (first hash-derived seed), all with `shocks-i-smoke`.
Roster cells run through `eval/gate_cli.run_job` on the same two portable worlds.

| cell | subject / member | shock_events (h = 12) | t(4) | t(8) | t(12) | setup | per cycle | **t(100)** |
|---|---|---|---|---|---|---|---|---|
| portable 12 | candidate | 3 (2 hits, 1 outage) | 0.110 | 0.211 | 0.312 | 0.009 | 0.0252 | **2.53** |
| portable 12 | twin | 0 | 0.100 | 0.214 | 0.314 | −0.005 | 0.0268 | **2.67** |
| portable 42 | candidate | 11 (7 hits, 4 outages) | 0.502 | 0.968 | 1.387 | 0.067 | 0.1107 | **11.13** |
| portable 42 | twin | 0 | 0.502 | 0.951 | 1.436 | 0.028 | 0.1168 | **11.71** |
| flat 12 (s6) | candidate | 3 (2 hits, 1 outage) | 0.072 | 0.135 | 0.201 | 0.006 | 0.0162 | **1.63** |
| flat 12 (s6) | twin | 0 | 0.067 | 0.139 | 0.199 | 0.003 | 0.0165 | **1.65** |
| roster 12 | `entity_planner/shock_reactive` | 3 | 0.037 | 0.073 | 0.108 | 0.002 | 0.0088 | **0.89** |
| roster 12 | `entity_planner/proportional` | 3 | 0.037 | 0.070 | 0.099 | 0.006 | 0.0078 | **0.78** |
| roster 42 | `entity_planner/shock_reactive` | 11 | 0.224 | 0.415 | 0.589 | 0.044 | 0.0457 | **4.61** |
| roster 42 | `entity_planner/proportional` | 11 | 0.208 | 0.386 | 0.574 | 0.023 | 0.0458 | **4.60** |

- `setup` and `per cycle` are the least-squares fit t = setup + per_cycle · h over the three
  horizons; **t(100) = setup + 100 · per_cycle** is the scaling assumption (cost linear in cycles
  played, setup paid once), unverified past cycle 12 (§3). No episode died, so every cell played
  all its cycles.
- At h = 12 the shock counts are D7's resolved counts (hits = round(P/6), outages =
  round(P/12)). At h = 4 and 8 the schedule keeps only onsets inside the horizon (smoke onsets
  fall in cycles 2–8): portable 12 had 0 / 3 events, portable 42 4 / 11, flat 2 / 3. Since the
  twins (0 events) cost the same as the shocked episodes, the fewer events at short horizons do
  not bias the fit.
- `shock_events` is from the candidate row; twins have an empty schedule (0) by construction.
  Over cycles 1–12, shocked and twin episodes cost the same within ≈ 5% (the twin is not
  cheaper): the shock mechanics themselves add no measurable per-cycle cost there. Whether a
  shocked trajectory costs more or less than its twin over 100 cycles under the real "shocks-i"
  profile (onsets in cycles 21–90; recovery, investment, deaths) is not measured.
- `shock_reactive` costs 0–14% more than `proportional` with the same producer.
- Cold costs (not in t(100)): the first job of a stack in a process pays the torch import and the
  checkpoint load, 1.37 s for the first portable job and 0.08 s for the flat stack after it; a
  pod worker pays this once per stack, ≈ 1.5 s per worker, negligible at both run sizes.

## 2. Job counts

- **8.1** (`eval/shock_contract.py`, D11; checked by `test_recorded_grid_is_80_plus_40_...`):
  portable 4 worlds × 10 episode seeds at 12 and at 42 = 80 episodes + 80 twins; flat
  topologies 6–9 × 10 seeds = 40 episodes + 40 twins. **240 jobs**: 80 at portable-12 cost
  (40 + 40), 80 at portable-42, 80 at flat-12.
- **9.1** (registry v2 draft `worlds_v2.json` 3032147f1f17, roster v2 draft `roster_v2.json`):
  development gate world-episodes per size = 8 calm × 3 + 4 shocked × 10 = 64; × 2 sizes =
  **128 world-episodes**. Roster v2: 48 members (35 deterministic, 13 random at R = 5) =
  **100 runs per world-episode** (35 joint-stack-contract and 13 legacy members). **12,800
  roster jobs** (6,400 per size), plus reserves only if the on-demand screen replaces a
  zero-baseline world (none were needed on these calm worlds in EXP-0044 attempt 2).
- **9.1 reserves** (same registry; `--reserves on-demand` runs a reserve only when it is
  consulted): per size 4 calm reserves × 3 episodes + 4 shocked × 10 = **52 world-episodes**,
  104 over both sizes, i.e. up to **10,400 more roster jobs**. A consulted calm reserve adds
  3 × 100 jobs, a shocked one 10 × 100.

## 3. Supporting evidence for the horizon-100 extrapolation (not a validation)

No timing in this probe goes past cycle 12, so t(100) is an extrapolation. The only direct
100-cycle timings on this box come from the archived gate sizing
(`openspec/changes/archive/2026-10-27-add-measurement-gate-harness/sizing.md` §A1, script
`/tmp/gate_timing/time_ep.py`), and they differ from this probe in almost every condition:

| condition | archive §A1 | this probe |
|---|---|---|
| commit | af192e8 (before shocks, drift, telemetry-driven `shock_reactive`) | d7e07c7 |
| registry | `worlds_v1.json` draft | `worlds_v2.json` draft |
| world | `dev-sparse_wide-{12,42}-calm-00` | `dev-sparse_wide-{12,42}-shocked-00` (and flat s6) |
| episode seed | the world's first registry seed (`e.episode_seeds[0]`; the archive heading says "seed 0") | the world's first registry seed |
| shocks | none (calm world, old mechanics) | `shocks-i-smoke` (onsets cycles 2–8) |
| horizon | 25 / 50 / 100, measured | 4 / 8 / 12, extrapolated to 100 |

| case | direct h = 100 (archive) | fitted t(100) here | difference |
|---|---|---|---|
| portable 12 | 2.52 | 2.53 | +0.4% |
| portable 42 | 12.28 | 11.13 (shocked) / 11.71 (twin) | −9% / −5% |
| `entity_planner/proportional` 12 | 0.70 | 0.78 | +11% |
| `entity_planner/proportional` 42 | 4.28 | 4.60 | +7% |

The archive also found cost roughly linear in h between 25 and 100 (×1.76–2.09 per doubling)
for its subjects. The agreement above is consistent with the linear assumption for the portable
stack and `entity_planner/proportional`, but it is not a validation and gives no error bound
for the current code: worlds, profile and code differ, and per-cycle cost after cycle 12 can
change with the state (inventory, open orders, telemetry) and with investment and death
trajectories. A second, indirect check: the two pod calibrations in §4 agree (0.66 using this
probe's extrapolated t(100), 0.65 using the archive's direct timings), which would be unlikely
if the extrapolation for these subjects were far off; it too is supporting evidence only.

Unverified sensitivities, carried as such in every estimate:
- the flat stack and `entity_planner/shock_reactive` have no 100-cycle anchor at all;
- horizon growth of per-cycle cost after cycle 12, for every subject;
- the real "shocks-i" profile at horizon 100 (shock vs twin cost difference over a whole
  episode).

## 4. Local box vs pod

| reference run | work | local-CPU estimate of the same work | pod execution, vCPU-h | pod / local |
|---|---|---|---|---|
| EXP-0045 (cpu5c 8 vCPU, 5.5 min) | 64 portable + 576 roster (5 `entity_planner` members, R = 5) episodes per size, h = 100 | 64 × (2.53 + 11.13) + 576 × (0.78 + 4.60) s = 1.10 h (this probe's t(100); `entity_planner/proportional` stands for every member) | 5.5 × 8 / 60 = 0.73 | **0.66** |
| EXP-0044 attempt 2 (cpu5c 16 vCPU, 12.3 min) | 11,264 roster jobs (88 per world-episode, roster v1) | 64 × (47.1 + 236.2) s = 5.04 h (archive §A2: measured roster seconds per world-episode, sparse-wide 12 / 42) | 12.3 × 16 / 60 = 3.28 | **0.65** |

Both pod figures are wall × vCPU, so they include the pool's packing tail and any early deaths
of weak roster members; both runs agree on ≈ 0.65. The estimates use 0.65 (calibrated) and 1.0
(cautious: the pod no faster than this box).

Pod overhead beyond execution, from the same runs: EXP-0045 pod life 6.2 min vs 5.5 min
execution (≈ 1 min); EXP-0044 attempt 2 15.4 vs 12.3 (≈ 3 min); EXP-0044 attempt 1 waited
≈ 11 min for SSH. Billed figures below add 1, 3 and 11 minutes.

## 5. Estimates

**8.1 (EXP-0046), cpu8 fresh pod.** Local work 40 × (2.53 + 2.67) + 40 × (11.13 + 11.71) +
40 × (1.63 + 1.65) = 1,253 s = 0.348 local CPU-h.

| factor | execution vCPU-h | execution wall on 8 vCPU | billed vCPU-h at +1 / +3 / +11 min overhead |
|---|---|---|---|
| 0.65 (calibrated) | 0.23 | 1.7 min | 0.36 / 0.63 / 1.70 |
| 1.0 (cautious) | 0.35 | 2.6 min | 0.48 / 0.75 / 1.81 |

Packing is not a concern: the longest job (a portable-42 episode) is ≈ 12 s against a 2–3 min
run. Memory: each worker holds both stacks (two small torch models each); 16 GB on cpu8 is ample.

**9.1 (EXP-0047), cpu16 fresh pod.** Three scenarios:
- **A — scale the measured EXP-0044 attempt 2:** 3.28 vCPU-h × 100 / 88 = **3.73**. Same worlds
  (calm entries are copied unchanged into v2; shocked entries add `shocks-i`, whose per-cycle cost
  over cycles 1–12 was not measurable in §1, while its 100-cycle effect is unmeasured), same
  members plus the 12 `shock_reactive` runs.
- **B — archive per-world-episode roster cost + 12 extra runs at the timed `shock_reactive`
  cost:** 64 × ((47.1 + 12 × 0.89) + (236.2 + 12 × 4.61)) s = 6.21 local CPU-h → **4.04** at 0.65,
  6.21 at 1.0.
- **C — every one of the 100 runs at the costliest measured member:** 0.89 s at 12 (this probe,
  `entity_planner/shock_reactive`; the archive's costliest sparse-12 member was 0.83 s) and
  4.73 s at 42 (archive §A2, `entity_planner/blend_0.5` on calm-00, above this probe's 4.61 s):
  64 × 100 × (0.89 + 4.73) s = 9.98 local CPU-h → 6.49 at 0.65, **9.98** at 1.0. Pessimistic:
  most members are cheaper and weak ones die early. It is not a proven maximum: only a few
  members, worlds and seeds were timed, and pod hardware varies.

Every scenario is conditional on its assumptions; none is a bound.

| scenario (gate worlds only, no replacement) | execution vCPU-h | execution wall on 16 vCPU | billed vCPU-h at +1 / +3 / +11 min overhead |
|---|---|---|---|
| A | 3.73 | 14.0 min | 3.99 / 4.53 / 6.66 |
| B at 0.65 | 4.04 | 15.1 min | 4.30 / 4.84 / 6.97 |
| B at 1.0 | 6.21 | 23.3 min | 6.48 / 7.01 / 9.14 |
| C at 0.65 | 6.49 | 24.3 min | 6.76 / 7.29 / 9.42 |
| C at 1.0 | 9.98 | 37.4 min | 10.25 / 10.78 / 12.92 |

**Reserves.** Each consulted reserve adds its world-episodes × 100 runs at the same per-run cost
(per run: B 0.577 s at 12 and 2.915 s at 42, C 0.886 s and 4.730 s, local):

| consulted reserve | jobs | B at 0.65 / 1.0, vCPU-h | C at 0.65 / 1.0, vCPU-h |
|---|---|---|---|
| calm, 12 producers | 300 | 0.03 / 0.05 | 0.05 / 0.07 |
| shocked, 12 producers | 1,000 | 0.10 / 0.16 | 0.16 / 0.25 |
| calm, 42 producers | 300 | 0.16 / 0.24 | 0.26 / 0.39 |
| shocked, 42 producers | 1,000 | 0.53 / 0.81 | 0.85 / 1.31 |

With **all** 104 reserve world-episodes (232 world-episodes, 23,200 jobs; a sensitivity case —
on-demand runs a reserve only when consulted):

| scenario (gate + all reserves) | execution vCPU-h | execution wall on 16 vCPU | billed vCPU-h at +1 / +3 / +11 min overhead |
|---|---|---|---|
| A | 6.76 | 25.3 min | 7.02 / 7.56 / 9.69 |
| B at 0.65 | 7.31 | 27.4 min | 7.58 / 8.11 / 10.25 |
| B at 1.0 | 11.25 | 42.2 min | 11.52 / 12.05 / 14.19 |
| C at 0.65 | 11.76 | 44.1 min | 12.03 / 12.56 / 14.70 |
| C at 1.0 | 18.10 | 67.9 min | 18.36 / 18.90 / **21.03** |

Expected: ≈ 4.5–5 vCPU-h billed (A/B at 0.65, +3 min, no replacement). One combination crosses
20: all reserves, C at 1.0, +11 min (21.0).

## 6. Decision

- **8.1:** below 20 vCPU-h in every scenario computed here (≤ 1.8 billed; 8.1 has no reserves). Profile `cpu8` as planned.
- **9.1:** the expected case (≈ 4.5–5 billed) and every no-replacement scenario (≤ 12.9) are
  below 20, but the all-reserves stress scenario (C at 1.0, +11 min) reaches ≈ 21.0. Per the
  task, this is reported to the supervisor and nothing launches on this report alone; whether to
  launch 9.1 as planned (`cpu16`, `--reserves on-demand`) is the supervisor's decision.
  Evidence relevant to it: EXP-0044 attempt 2 replaced no world on the identical calm worlds,
  and at the calibrated speed even consulting all eight shocked reserves (both sizes) adds
  ≈ 2.5–4.1 vCPU-h (B–C at 0.65), i.e. ≈ 7.4–11.4 billed with +3 min overhead.

## 7. Method and raw data

- Probe: an out-of-tree script (`/tmp/probe62/probe.py`, ephemeral; raw timings
  `/tmp/probe62/cells.jsonl`, fit `/tmp/probe62/analyze.py` → `fit.json`). It builds the 8.1
  jobs with `shock_contract.portable_worlds(..., profile="shocks-i-smoke", worlds_per_size=1)`,
  `shock_contract.flat_grid(...)` and `shock_contract.build_jobs`, and times
  `shock_contract.run_job` per job; roster jobs come from `gate_cli.build_jobs` on the same worlds
  with a one-member roster and are timed with `gate_cli.run_job` after `gate_cli._worker_init(h,
  None, None)`. Cold runs clear the worker caches first; warm values are the best of 3 repeats
  with caches filled (the roster config cache is per horizon, so its first repeat is excluded).
- No contract failure occurred in any of the 96 timed runs; no episode died.
- Not covered: the spread over worlds and episode seeds (one world per size, one seed); pod
  contention among concurrent workers (absorbed in the 0.65 calibration, which comes from
  concurrent pod runs).
