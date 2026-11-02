# Sizing (add-gate-v3)

## §pilot and §screen (task 5.2)

**Question.** What will the v3 pre-screen pilot (design D3 step 0, task 6.1) and the v3
development screen (D3 waves 1–2, task 7.1) cost on a pod, on which shape, and does either need
one of D3's pre-registered reductions to keep its hard stop (2 × billed) ≤ 40 vCPU-h?

**Method.**
- **Timing.** `/srv/research/tokenomics/openspec/changes/add-gate-v3/probes/timing_screen_v3.py`
  times the screen's own worker, `gate_cli.run_job`. It runs jobs built by `gate_cli.build_jobs`
  from Drafts' `worlds_v3_draft.json`: one roster-member job per call, horizon 100, the world's
  first episode, replicate 0. A stochastic member uses action seed
  `replicate_seed(world, episode, 0)`, as in the screen.
- **Members.** `entity_planner` × {`round_robin`, `proportional`, `makeable_weighted_whole`,
  `makeable_split`}, `noop` × `even_split` and `random` × `random`, plus `stock_aware` ×
  `proportional` (notes/5.2.md, departure 1).
- **Worlds.** For each (family, regime) cell I timed the **most loaded** development gate world
  (largest k_c × m_q) of **each choice level** the cell holds, and every member is priced at the
  max over those worlds. Each price is a load-based bound on its cell (by k_c × m_q), not a
  per-world guarantee (see Uncertainties):
  - `ws_loaded` calm: dense `dev-ws_loaded-12-calm-00` (k_c·m_q 7.1) and f4
    `dev-ws_loaded-12-calm-01` (13.8, the cell's maximum);
  - `ws_loaded` shocked: dense `dev-ws_loaded-12-shocked-00` (12.0, the cell's maximum) and f4
    `dev-ws_loaded-12-shocked-01` (5.1);
  - `ws_wide` calm: `dev-ws_wide-42-calm-01` (37.2, ρ 2.01), the cell's maximum;
  - `ws_wide` shocked: `dev-ws_wide-42-shocked-01` (16.4), the cell's maximum;
  - one v2 world, `dev-sparse_wide-12-calm-03`, for the 0.646 check. It is the world on which
    add-m3-development-audit derived the factor.
  - Round 0 timed only one world per `ws_loaded` cell (dense calm, f4 shocked) and called it the
    cell's most loaded, which was wrong for `ws_loaded`: the f4 calm and dense shocked worlds are
    more loaded. Review round 1 caught this. The two missing worlds are now timed, and the
    numbers below include them.
- **Raw lines** are in `probes/timing_screen_v3.jsonl`: 54 lines, 52 of them at horizon 100. Each
  line carries the load average, MemAvailable and peak RSS.
- **Extrapolation.**
  `/srv/research/tokenomics/openspec/changes/add-gate-v3/probes/extrapolate_screen_v3.py` builds
  the real job lists with `gate_cli.build_jobs` over the Drafts registry's development gate worlds
  and rosters. It prices each job and simulates the spawn pool:
  - jobs are submitted in `build_jobs` order (world, episode, member), as `gate_cli._execute`
    submits them, and each idle worker takes the next job;
  - each pool adds 3.5 s × 0.646 of worker start, and each run adds 60 s of parent work.
  - Outputs: `probes/extrapolate_screen_v3.json` (estimate) and
    `probes/extrapolate_screen_v3.central.json` (central).

**Local box.** 2 CPUs, shared, one command at a time, `--workers 1`, `nice -n 19`,
`OMP/MKL/OPENBLAS_NUM_THREADS=1`.
- Every horizon-100 command ran with load ≤ 2.15 and MemAvailable ≥ 721 MB (the round-1
  re-measure at load < 1). Peak RSS was about
  50 MB per process, since roster-only jobs import no torch.
- Two earlier attempts stalled when other sessions' test runs pushed the load to about 6 and
  MemAvailable to about 150 MB. They wrote no line. After that the probe refuses to start a job
  while the load is above 4 (`--guard-load 4`), and those cells were re-measured when the box was
  quiet.
- No 42-producer horizon-100 episode swapped. The longest command was 67.4 s, so nothing had to
  be extrapolated from a shorter horizon. The horizon-12 scout lines are tagged `h12-scout` and
  are not used.

### Measured seconds (local, horizon 100, first episode; † = system death)

| world | choice | ρ | k_c·m_q | EP/RR | EP/prop | EP/MWW | EP/MS | noop/ES | random/random | stock_aware/prop |
|---|---|---|---|---|---|---|---|---|---|---|
| dev-sparse_wide-12-calm-03 (v2) | f2 | – | – | 0.82 | 0.80 | 0.89 | 0.87 | 0.09 † | 0.73 † | 0.72 |
| dev-ws_loaded-12-calm-00 | dense | 0.49 | 7.1 | 1.97 | 2.45 | 2.65 | 2.40 | 0.73 † | 2.48 | 3.16 |
| dev-ws_loaded-12-calm-01 | f4 | 0.38 | 13.8 | 2.36 | 2.91 | 2.91 | 2.76 | 0.40 † | 2.36 † | 2.46 |
| dev-ws_loaded-12-shocked-00 | dense | 0.35 | 12.0 | 2.24 | 2.55 | 2.22 | 2.33 | 0.49 † | 2.40 | 2.06 |
| dev-ws_loaded-12-shocked-01 | f4 | 0.98 | 5.1 | 2.09 | 1.86 | 1.82 | 1.92 | 0.22 † | 2.65 | 2.87 |
| dev-ws_wide-42-calm-01 | f14 | 2.01 | 37.2 | 31.18 | **67.37** | 31.90 | 45.68 | 6.88 † | 42.53 | 41.46 |
| dev-ws_wide-42-shocked-01 | f14 | 0.61 | 16.4 | 27.63 | 54.70 | 30.70 | 36.08 | 9.65 † | 42.73 | 37.53 |

- **`ws_loaded`:** on the more loaded world of each cell the EP class prices rise against the
  round-0 world by 1.10 / 1.19 (calm, whole / splitting) and 1.07 / 1.33 (shocked). The calm-01
  timings sit with the reviewer's own (EP/RR 2.59 s against 2.36 s here).
  `ws_loaded` is a small share of either run, so totals move little.
- **Splitting routers cost most at 42 f14.** `proportional` is 2.0–2.2 × `round_robin`, and
  `makeable_split` 1.3–1.5 ×. Whole routers (RR, MWW) cost the same as each other.
- These sit with M4's measurements: on less loaded f14 worlds M4 timed `round_robin` at
  19.4–23.1 s (m4 sizing.md §EXP-0057), and the earlier m3 figures were 22.6–29.0 s for RR and
  27.8–53.0 s for proportional.
- **The 0.646 check.**
  - On the v2 world the surviving members took 0.72–0.89 s (mean 0.82 s). The m3 measurement on
    the same world was 0.685 s per surviving run, over EXP-0047's 48-member mix.
  - That gives a factor of **1.197**. It is either a slower box today or `entity_planner`'s
    members being slower than the mix average; the two cannot be separated here.
  - The estimate assumes the worst case: v2 runs are scaled up by 1.197, and today's region
    timings are used undivided. The **central** variant reads the factor as box slowdown, which
    lowers every number by about 15–17%.
  - **Reproduction.** On the v2 world the four probed members that EXP-0047 also ran
    (EP/proportional, noop/even_split, random/random, stock_aware/proportional) give EXP-0047's
    units exactly.
- **Determinism** (verify item). I reran the shocked `ws_wide` world, `dev-ws_wide-42-shocked-01`,
  for EP/RR and for the two stochastic members EP/MWW and random/random. Units equal the first run
  for all three: 11743, 12926 and 1830. The rerun lines are tagged `rerun`.

### Price models

- **v2 worlds:** EXP-0047's job mix as timed locally by m3 (0.581 s per run at 12 producers,
  2.642 s at 42, row-weighted over survived and died runs) × the 1.197 factor × 0.646. On a pod,
  EXP-0047 realized exactly this per-run cost without the factor. The promoted routers on the v2
  world cost the same as the v2 routers (0.82–0.89 s).
- **Region worlds, class (the estimate).** Each member is priced at its measured class, taking
  the max over the measured worlds of its family and regime:
  - EP × whole routers (RR, RW, MWW): max(RR, MWW);
  - EP × splitting routers (the other 8): max(proportional, MS);
  - noop: noop/ES;
  - random producer: random/random;
  - stock_aware and obs_stock_aware: stock_aware/prop;
  - the producers never timed (always_restore, always_both, restore_when_low): the cell's
    slowest measured member.
- **Region worlds, upper:** every member at the cell's slowest measured member.
- **Wave-2 roster (the model):** `entity_planner` × 11 plus 2 non-EP members, one of them
  stochastic. That is 13 members, 4 stochastic and 29 runs per world-episode, D3's "≈ 13 members,
  4 stochastic". The real W2 is decided by wave 1.
- **Reserves:** none are run in these totals. See the contingency below.
- **Billing:** billed vCPU-h = vCPU × (setup + driver wall). The setup is either **observed**,
  about 1 min on recent pods, or the **10-min planning basis**. Stop = 2 × billed on the planning
  basis.

### §pilot (task 6.1)

The pilot is a dry-run screen with the pilot roster (`entity_planner` × 11, 23 runs per
world-episode) on all 36 development gate worlds (24 v2 + 6 `ws_loaded` + 6 `ws_wide`), with no
reserves: 192 world-episodes, 4,416 jobs.

| model | pod compute vCPU-h (sparse_wide / ws_loaded / ws_wide) |
|---|---|
| **class (estimate)** | **7.43** (1.02 / 0.34 / 6.06) |
| upper | 9.26 |
| central (class) | 6.20 |

| shape | driver wall | longest job (pod) | workers busy | billed, observed | **billed, planning** | **stop = 2 × planning** |
|---|---|---|---|---|---|---|
| **cpu32** | **15.3 min** | 43.5 s | 0.97 | 8.7 | **13.5** | **27.0** (51 min of cpu32 pod life) |
| cpu16 | 29.2 min | 43.5 s | 0.99 | 8.1 | 10.5 | 20.9 (78 min of cpu16 pod life) |

- **Recommended shape: cpu32**, by the prefer-larger-pods rule (largest secure CPU pod first,
  workers = vCPU). It costs 13.5 billed on the planning basis and 8.7 on the observed one, with
  about 15 min of driver wall, and a stop of 27.0.
- **Fallback: cpu16**, at 10.5 billed and a stop of 20.9. On the upper model it is 12.2, with a
  stop of 24.5.
- **No reduction is needed.** Every shape and model stops ≤ 40; the largest is cpu32 upper at
  30.4.
- **Against task 6.1 and D10.** Task 6.1 registers "cpu16, ≈ 7 billed vCPU-h (≈ 5.1 compute),
  hard stop 14". That is too low: the measured compute is 7.4, not 5.1, so cpu16 bills 10.5 with a
  stop of 20.9. The difference is `ws_wide`. D3 assumed about 19 pod-s per f14 run; the class
  model averages about 30 pod-s on the most loaded worlds, because the splitting routers dominate
  (EP/prop costs 43.5 pod-s). 6.1's resources line needs the supervisor's re-size; it is not
  edited here.

### §screen (task 7.1)

- **Wave 1:** all 88 draft members (216 runs per world-episode) on the 24 v2 and 6 `ws_loaded`
  development gate worlds.
- **Wave 2:** the modelled W2 (29 runs per world-episode) on the 6 `ws_wide` worlds.
- Both waves run in one pod run, one pool each.

| variant | jobs | pod compute vCPU-h (sw / repro / ws_loaded / ws_wide), class | upper | central |
|---|---|---|---|---|
| no reduction | 35,488 | 20.18 (9.57 / – / 3.20 / 7.41) | 23.17 | 16.86 |
| reduction 1 (shocked region worlds at 5 episodes) | 33,038 | 17.02 (9.57 / – / 2.23 / 5.22) | 19.21 | 14.22 |
| reductions 1 + 2 (also reuse EXP-0047's v2 rows for the 48 v2-draft members; reproduction check: 1 world-episode per (size, regime) × 48 members) | 20,638 | 12.73 (5.14 / 0.14 / 2.23 / 5.22) | 14.92 | 10.63 |

Class model, driver wall = the two pool walls plus parent; longest job 43.5 pod-s in every
variant:

| variant | shape | driver wall (wave 1 + wave 2) | billed, observed | billed, planning | stop = 2 × planning | ≤ 40? |
|---|---|---|---|---|---|---|
| no reduction | cpu32 | 39.2 min (24.0 + 14.2) | 21.5 | 26.3 | 52.5 | no |
| no reduction | cpu16 | 77.0 min (48.0 + 28.0) | 20.8 | 23.2 | 46.4 | no |
| reduction 1 | cpu32 | 33.3 min (22.2 + 10.1) | 18.3 | 23.1 | 46.2 | no |
| reduction 1 | cpu16 | 65.1 min (44.3 + 19.8) | 17.6 | 20.0 | 40.06 | no (upper model: 44.4) |
| **reductions 1 + 2** | **cpu32** | **25.2 min (14.1 + 10.1)** | **14.0** | **18.80** | **37.6** (70 min of cpu32 pod life) | **yes** (upper model: 41.7) |
| reductions 1 + 2 | cpu16 | 49.0 min (28.2 + 19.8) | 13.3 | 15.7 | 31.5 | yes (upper model: 35.8) |

- **D3's rule** applies the reductions in order, only while 2 × billed exceeds 40.
- **Recommended: cpu32 with reductions 1 + 2.** That is 18.8 billed on the planning basis
  (14.0 observed), about 25 min of driver wall, and a stop of 37.6. On cpu32, reduction 1 alone
  leaves the stop at 46.2, so reduction 2 is needed.
  - Reduction 2 needs its reproduction check before the reuse: one world-episode per
    (size, regime) × all 48 v2-draft members, which must match EXP-0047 exactly. It costs 0.14
    compute and is included.
  - D3 records the reductions in `spec.json` and the registry (task 3.1's flags
    `--region-shocked-episodes 5` and `--reuse-rows <EXP-0047 dir>`).
- **cpu16 with reduction 1 only** now stops at 40.06 on the estimate (44.4 on the upper model),
  just over the cap, so it does not qualify. At round 0, with the under-priced `ws_loaded`, it sat
  at exactly 40.0.
- **On the observed setup basis**, reduction 1 alone fits cpu32: 2 × 18.3 = 36.6. If the
  supervisor adopts the observed basis for stops, reduction 2 and its reproduction dependency
  could be dropped. The planning basis is the convention used so far (m4 sizing.md), so the
  recommendation follows it.
- **Reserve contingency (not in the totals).**
  - Zero-baseline replacement never fired in EXP-0047. Region worlds, however, also get
    bar-death replacement (D3 step 3).
  - Suppose each of the 4 region gate groups consults one reserve: one calm reserve at 3 episodes
    and one shocked reserve at 5 (after reduction 1), per family.
  - That adds about 0.8 compute in wave 1 (`ws_loaded`: 8 world-episodes × 216 runs at about
    1.6 pod-s each) and about 1.9 in wave 2 (`ws_wide`: 8 × 29 runs at about 29 pod-s each).
  - Each extra reserve sub-wave is a separate pool, about 3–5 min of cpu32 wall.
  - About 2.7 compute plus about 3 vCPU-h of tail would raise the billed to about 24.5. That is
    within the 37.6 stop.
- **Against D3 / D10.** D3 estimated about 16.4 compute and D10 about 18 billed on cpu32 with a
  stop of 36, all without reductions. The measured compute without reductions is 20.2 (central
  16.9), so the unreduced stop of 52.5 exceeds 40. With D3's pre-registered reductions 1 + 2 the
  cpu32 run lands at 18.8 billed with a stop of 37.6, close to D10's line.

### Tranche G1 launch fit (65 vCPU-h, D10)

Each check is the **estimated billed spend of the earlier runs (class model, planning basis; not
the "central" variant)** plus this run's stop, on the recommended shapes. D10 calls this sum
"central billed spend", but the figures here are the class-model estimates. The teacher-read
and sealed-screen stops are D10's.
- pilot: 0 + 27.0 = 27.0 ✓;
- screen: 13.5 + 37.6 = 51.1 ✓;
- 42-producer teacher read: 13.5 + 18.8 + 40 (the largest stop its probe may set) =
  **72.3 > 65 ✗**;
- sealed screen: 13.5 + 18.8 + 12 + 5 = 49.3 ✓.

The teacher-read check misses by 7.3. Moving the pilot to cpu16 lowers it to 69.3, still a
miss. With the central variant's earlier spend (12.2 + 16.7) it would be 68.9, also a miss. This goes to the supervisor, to resize G1 or reorder, per D10. Nothing here edits D10 or
tasks.md.

### Uncertainties

- **The 1.197 factor.** It is box slowdown or member mix; the estimate assumes the worse
  reading. The central variant is about 15–17% lower.
- **Only the most loaded world per choice level** of each region cell was timed, at its first
  episode. Other worlds of the cell are priced at the max over those worlds, which bounds them if
  load drives cost: `ws_wide` gate worlds span k_c·m_q 2.4–37.2. Cost is not strictly
  monotone in k_c·m_q (e.g. `random` dies on calm-01), so this is a load-based bound, not a
  guarantee per world.
- **Untimed producers.** always_restore, always_both and restore_when_low were never timed. They
  are priced at the world's slowest member. On v2 worlds they ride on EXP-0047's realized
  per-run cost.
- **W2 is modelled.** D3's ≈ 13 members are a model; if wave 1 carries more winners, wave 2
  grows by about 0.25 compute per extra deterministic non-EP member.
- **The 0.646 factor** comes from EXP-0047's job mix. On EXP-0052 the realized driver vCPU-h was
  0.90 × the estimate made with it (m4 sizing.md §EXP-0057).

**Supervisor decision (2026-11-02, G1).** Task 5.2's measured costs make the teacher read's
launch-fit 13.48 + 18.80 + 40 = 72.3 > 65. Under the user's grant (gate-v3 budget from the M4
tranche under the supervisor's authority; design D10), **tranche G1 is resized to 75 vCPU-h**,
separate from M4's tranche A; run order unchanged. Pilot on cpu32 (stop 27.0), screen on cpu32
with D3 reductions 1 + 2 (stop 37.59). tasks.md 6.1 / 7.1 Resources lines are re-sized to these
rows in the next gate-v3 plan pass (carry-forward). Recorded in
add-m4-orderer-redesign/notes/user-decisions.md as D10 requires.
