# Sizing (add-m4-orderer-redesign)

## §EXP-0057 (task 2.3)

**Question.** How much will the decoder check (`python -m tokenomics.eval.decoder_check`,
design D3) cost on a pod, and on which shape? The real run is 639 jobs: 71 world-episodes
(43 at 12 producers over 23 worlds, 28 at 42 over 14 worlds) × 9 arm-replicates (TR, TS × 2,
MWWk × 2, MWW × 2, TW, RR), horizon 100, `entity_planner` producer, tracer on.

**Method.** `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/probes/timing_23.py`
builds single jobs exactly as `run_decoder_check` builds them (`world_entry`, `decoder_arm`,
`build_jobs`) and times `decoder_check.run_decoder_job`, the driver's own worker
(`joint_stack_env`, `set_draw_key`, `RoutingTracer`, `ContractMonitor`, `ScoreFidelity` on the
trained arms). One job per command per world × arm: the confirm's first episode seed,
replicate 0. The orderer checkpoint is loaded before the clock starts, as a pool worker loads it
once per process. Raw lines are in `probes/timing_23.jsonl`.

**Local box.** 2 CPUs, shared, one command at a time, `OMP/MKL/OPENBLAS_NUM_THREADS=1`,
`torch.set_num_threads(1)`. The load average was 0.3–1.15 throughout. Every command took
under 2 min; the longest was 63.2 s (42-0076 TR, horizon 100).

**Worlds.** The loaded worlds are:
- 12 producers: `ws-atlas-v1-12-0352` (dense, ρ 4.49, load k_c × m_q 30.8, shocks, switch). It
  has the slowest RR of the 23 sample worlds.
- 42 producers: `ws-atlas-v1-42-0076` (sparse_f14, ρ 1.24, load 5.7, shocks, burst). f14 is the
  costliest choice level (m3 sizing.md §EXP-0051), and 0076 is the most loaded f14 world in the
  sample.

### Measured seconds (local, one episode)

| world | arm | horizon 12 | horizon 100 | ratio 100 / 12 (linear 8.33) | arm / RR at 100 |
|---|---|---|---|---|---|
| 12-0352 | TR | 1.24 | 9.15 | 7.4 | 2.34 |
| 12-0352 | TS | 1.12 | 9.06 | 8.1 | 2.32 |
| 12-0352 | TW | – | 8.23 | – | 2.10 |
| 12-0352 | MWWk | 0.53 | 3.74 | 7.0 | 0.96 |
| 12-0352 | MWW | – | 3.73 | – | 0.95 |
| 12-0352 | RR | 0.48 | 3.91 | 8.1 | 1 |
| 42-0076 | TR | 6.23 | 60.93 | 9.8 | 2.68 |
| 42-0076 | TS | 5.94 | 54.75 | 9.2 | 2.41 |
| 42-0076 | TW | – | 38.56 | – | 1.70 |
| 42-0076 | MWWk | 2.80 | 22.64 | 8.1 | 1.00 |
| 42-0076 | MWW | – | 22.95 | – | 1.01 |
| 42-0076 | RR | 2.63 | 22.71 | 8.6 | 1 |

There are two ratio checks at horizon 100 on other worlds:
- 12-0060 (sparse_f4, RR 1.60 s): TR 3.98 (2.49 × RR), TS 3.97 (2.49), TW 3.80 (2.38), MWWk 1.73
  (1.08), MWW 1.84 (1.15);
- 42-0039 (sparse_f4, the most loaded 42 world, RR 15.58 s): TR 32.02 (2.06), TS 32.83 (2.11).

**RR at horizon 100 on every sample world** (first confirm episode) gives the per-world scale:
- 12 producers: 0.83–3.91 s, mean 1.81 s. 12-0352 is the slowest.
- 42 producers:
  - f14: 19.4–23.1 s;
  - dense 42-0119: 17.1 s;
  - f4 / f7: 6.9–15.6 s;
  - f2 42-0040: 6.2 s.

**Reproduction.** Every one of the probe's 44 TR / RR / MWW r0 rows at horizon 100 equals its
EXP-0052 confirm row in units. The driver CLI dry run below also reproduces the probe's units.

**Horizon 12 → 100.** The ratio is 7.0–9.8 against a linear 8.33. The 12-producer arms are
sub-linear, because the fixed per-episode cost weighs more at horizon 12. The 42-producer scorer
arms are super-linear (TR 9.8, TS 9.2), since state grows over the episode. Extrapolating
horizon 12 linearly would underprice the dominant 42-producer scorer jobs by up to 15%, so the
sizing below uses only the horizon-100 numbers.

**Driver cross-check.** The real CLI was run with `--dry-run --worlds ws-atlas-v1-12-0352 --arms
TR,MWWk,RR --horizon 100 --workers 1` (8 jobs, both confirm episodes). It took 42.3 s wall,
with 611 MB peak RSS. The cell model below prices those 8 jobs at 41.1 s, before about 4 s of
process start-up. The probe's per-job seconds are the driver's, and are slightly conservative
(the second episode is no slower).

**Memory.** In-process peak RSS was 495–512 MB. Each probe command's maxrss was 580–625 MB
(torch imported). A pod with N workers plus the parent needs about (N + 1) × 0.63 GB: 5.7 GB on
cpu8, 10.7 GB on cpu16 and 20.8 GB on cpu32. That fits `--mem 16`, `32` and `64` respectively.

### Extrapolation

`probes/extrapolate_23.py --atlas <EXP-0052>/reclassified/atlas_v2.json` builds the **real job
list** with the driver's functions (sample, confirm seeds, arm table, `build_jobs`,
`longest_first`): 639 jobs. It prices each job and simulates the pool. Jobs go to N workers in
`longest_first` order, and each idle worker takes the next job, as the spawn
`ProcessPoolExecutor` drains its queue. That order is all 42-producer scorer jobs, then 42-producer
rules, 12-producer scorers and 12-producer rules. The simulation adds 3.5 s of worker start
(× 0.646) and 60 s of parent work: pins and atlas before the jobs, readings and the reproduction
reads after.

Rates (m3 sizing.md §EXP-0051, design D10):
- pod s = 0.646 × local s;
- billed vCPU-h = vCPU × (10 min setup + wall).

There are two price models:
- **cell (the estimate):** every job is priced at its size × arm cell on the loaded world
  (table above). This is conservative: 12-0352 is the slowest 12-producer world, and 42-0076 is
  a loaded f14 world, the costliest level.
- **world (central check):** the job's world's measured RR seconds × the largest measured
  arm / RR ratio at its size. The ratios are:

  | size | TR | TS | TW | MWWk | MWW |
  |---|---|---|---|---|---|
  | 12 | 2.49 | 2.49 | 2.38 | 1.08 | 1.15 |
  | 42 | 2.68 | 2.41 | 1.70 | 1.00 | 1.01 |

| model | local s (12 / 42) | local vCPU-h | pod compute vCPU-h | longest job (pod) |
|---|---|---|---|---|
| **cell** | 2,336 / 9,040 = 11,376 | 3.16 | **2.04** | 39 s (42 TR) |
| world | 1,215 / 5,926 = 7,141 | 1.98 | 1.28 | 40 s (42-0031 TR) |

The 42-producer worlds are 79–83% of the compute, consistent with the 2.2 review's > 80%.

### Shapes (MWW R = 2, as registered)

| shape | wall (cell) | last job starts | workers busy | **billed (cell)** | hard stop = 2 × est. | billed (world) | billed, setup 3 min (cell) |
|---|---|---|---|---|---|---|---|
| **cpu8** | **16.4 min** | 15.3 min | 1.00 | **3.52** | **7.0 (52 min of pod life)** | 2.75 | 2.58 |
| cpu16 | 8.7 min | 7.7 min | 0.99 | 4.99 | 10.0 → capped at 8 (30 min) | 4.23 | 3.12 |
| cpu32 | 4.9 min | 3.8 min | 0.98 | 7.94 | 15.9 → capped at 8 (15 min) | 7.17 | 4.21 |

- Workers stay busy on every shape: 639 jobs, and the longest single job (≈ 40 s on the pod) is
  under 5% of even the cpu32 wall. The tail is short because the longest jobs are submitted
  first.
- Setup dominates the bill. At 10 min it costs 1.33 vCPU-h on cpu8, 2.67 on cpu16 and 5.33 on
  cpu32, against about 2.0 of compute. cpu16 costs +1.5 vCPU-h (+42%) to save 7.7 min of wall,
  and cpu32 costs +4.4 to save 11.5 min.
- **Sensitivity.** The non-driver pod life observed on EXP-0052–0056 was 1–3 min, not 10. At
  3 min, cpu8 is still the cheapest (2.58 against 3.12 / 4.21). The world model gives the same
  order.

**Recommended shape: cpu8**, a fresh secure pod (`--create-pod --vcpu 8 --mem 16`,
`--workers 8`):
- **estimate 3.5 billed vCPU-h** (2.04 compute + 1.33 setup, plus 0.15 for parent and tail);
- **≈ 16 min driver wall**, ≈ 26 min of pod life;
- **hard stop 7.0 vCPU-h**, which is 52 min of cpu8 pod life;
- launch-fit 0 + 7.0 ≤ 95.

It is the cheapest shape under both models and both setup assumptions, and its workers stay
fully busy. cpu16 would cost 5.0 billed. Its 2 × estimate (10.0) is above the task's ≤ 8 stop,
so it would run with a stop at the 8 cap, only 1.6 × its estimate. cpu8 was also the shape that
started first time on EXP-0052 / EXP-0053, after secure 16 vCPU creation failed.

**Reduction (pre-registered: MWW at R = 1 if the estimate is > 4 vCPU-h): not triggered.** The
estimate on the recommended cpu8 is 3.52 ≤ 4, so MWW keeps R = 2, and EXP-0052's MWW r0 / r1
rows are both reproduced. For the record, the reduction would remove 71 jobs (568 left): cell
compute falls 2.04 → 1.90, and billed falls on cpu8 3.52 → 3.37, cpu16 4.99 → 4.85 and cpu32
7.94 → 7.80. It cannot bring cpu16 under 4, because on that shape setup, not compute, is the
excess.

**Against design D10.** D10's EXP-0057 row (cpu16, 4.0 = 1.3 + 2.7, stop 8, grounding ≈ 7,300
local s) is re-sized here to cpu8, 3.5, stop 7.0. The compute is higher than D10 assumed: 11,376
local s on the cell model, mostly the 42-producer scorer arms at 55–61 s each, against D10's
7,300. The cheaper shape more than offsets it. In D10's launch-fit chain, EXP-0057's own check
becomes 0 + 7.0, and every later row's "spend before" falls by 0.5 (EXP-0058: 3.5 + 4 = 7.5;
the success path's largest spend + stop 91.9 → 91.4). The supervisor records this; D10 is not
edited here. D10's "per-order
policy ≈ 1.3 × the trained orderer (probes 2.3 / 5.4 confirm)" cannot be confirmed here,
because no per-order policy exists yet. TS (one whole-order draw from the same scorer per order)
costs 0.90–1.03 × TR. Task 5.4 must time the per-order policy itself.

**Uncertainties.**
- Only the first confirm episode of each world was timed. The driver dry run timed the second
  episode of 12-0352 as no slower.
- The 0.646 local-to-pod factor comes from EXP-0047's job mix. On EXP-0052 the realized driver
  vCPU-h was 0.90 × the recheck estimate made with it.
- Scorer jobs at 42 producers vary between 2.06 and 2.68 × RR across the two timed worlds. The
  cell model prices all 42 TR jobs at 60.9 s, the highest measured.
- 42-0076 is not the slowest 42-producer world: under the world model 42-0031's TR job (40 s on
  the pod) is about 1.6% slower than 42-0076's (39 s), so the cell model's per-job prices are not
  a strict upper bound at 42 (2.3 review). Its total stays above the world model's.

## §EXP-0066 (task 4.5)

**Question.** How much will the split-need re-test (`python -m tokenomics.eval.split_need`, task
4.5) cost on a pod, and on which shape? The real run is 284 jobs: EXP-0057's 71 world-episodes
(43 at 12 producers over 23 worlds, 28 at 42 over 14 worlds) × 4 arm-replicates (TS × 2, MWWk ×
2), horizon 100, `entity_planner` producer, tracer and split-need observer on.

**Method.** `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/probes/timing_45.py`
builds single jobs exactly as `run_split_need` builds them and times `split_need.run_split_job`,
the driver's own worker (EXP-0057's job path + the observer). The orderer is loaded before the
clock starts. Each timed row is also checked against its EXP-0057 row (`neutrality`: 79 / 79
cells equal on all 8) and the observer self-checks (`consistency`: 0 problems). Raw lines are in
`probes/timing_45.jsonl`. `probes/extrapolate_45.py` builds the real 284-job list with the
driver's functions (`select_views`, `world_entry`, `decoder_arm`, `build_jobs`, `longest_first`)
and simulates the pool with task 2.3's `extrapolate_23.simulate`, rates, 60 s parent time and
10 min setup.

**Local box.** 2 CPUs, shared with task 5.1's implementer, one command at a time,
`OMP_NUM_THREADS=1`, torch 1 thread. Load average 0.6–1.3. Longest command 55.7 s (42-0076 TS r0).

### Measured seconds (local, horizon 100, first confirm episode)

The two world-episodes are 12-0352 (the slowest 12-producer world in §EXP-0057) and 42-0076 (the
loaded f14 world, the costliest choice level).

| world | arm | r0 | r1 | EXP-0057 cell (no observer, r0) | observer overhead (slower rep / EXP-0057) |
|---|---|---|---|---|---|
| 12-0352 | TS | 8.99 | 9.25 | 9.06 | 1.02 |
| 12-0352 | MWWk | 3.78 | 3.72 | 3.74 | 1.01 |
| 42-0076 | TS | 54.36 | 49.82 | 54.75 | 0.99 |
| 42-0076 | MWWk | 24.31 | 24.61 | 22.64 | 1.09 |

The observer costs ≤ 9% (within run-to-run noise on the TS arms). Peak RSS 454–518 MB in
process; ≈ 0.63 GB per worker with torch, so cpu8 needs ≈ 5.7 GB (`--mem 16`).

### Extrapolation (284 jobs)

- **cell (the estimate):** every job at its size × arm cell, the slower measured replicate.
- **world (central check):** §EXP-0057's world model (world RR seconds × the largest arm / RR
  ratio) × the observer overhead above.

| model | local s (12 / 42) | local vCPU-h | pod compute vCPU-h | longest job (pod) |
|---|---|---|---|---|
| **cell** | 1,121 / 4,423 = 5,544 | 1.54 | **0.99** | 35 s (42 TS) |
| world | 577 / 2,913 = 3,491 | 0.97 | 0.63 | 36 s |

| shape | wall (cell) | workers busy | **billed (cell)** | stop = 2 × est. (≤ 8) | billed (world) | billed, setup 3 min (cell) |
|---|---|---|---|---|---|---|
| **cpu8** | **8.5 min** | 0.99 | **2.47** | **4.9 (37 min of pod life)** | 2.10 | 1.54 |
| cpu16 | 4.8 min | 0.98 | 3.94 | 7.9 (30 min) | 3.57 | 2.08 |

- Setup dominates again: 10 min costs 1.33 vCPU-h on cpu8, 2.67 on cpu16, against ≈ 1.0 of
  compute. cpu16 costs +1.5 vCPU-h to save ≈ 4 min of wall.
- **Calibration.** EXP-0057 ran 639 jobs in ≈ 8.7 min of driver wall on cpu8 (sized 16.4 min,
  0.53 ×) with setup under a minute: billed 1.4 against 3.5. This run's cell compute is 0.49 ×
  EXP-0057's (5,544 vs 11,376 local s), so the realized bill on cpu8 is likely ≈ 0.7–1.0
  vCPU-h. The estimate stays on the conservative cell model.

**Recommended shape: cpu8**, a fresh secure pod (`--create-pod --vcpu 8 --mem 16`, `--workers
8`):
- **estimate 2.5 billed vCPU-h** (0.99 compute + 1.33 setup + parent and tail);
- **≈ 8.5 min driver wall**;
- **hard stop 4.9 vCPU-h = 37 min of cpu8 pod life** (2 × estimate, ≤ 8). `run_pod_training.py`
  has no stop flag: the experiment implementer enforces it by wall clock (carry-forward).

### Pod command (task 4.6)

Stage the three inputs into the git-ignored `tokenomics/_pod_ckpts/` (the driver syncs the inner
package dir only; `_pod_ckpts/` is in `.gitignore`, so `--verify-source-commit` still sees a
clean tree). Check the SHA-256s after copying:

| staged file | source | SHA-256 |
|---|---|---|
| `tokenomics/_pod_ckpts/exp0066/atlas_v2.json` | `experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json` | `7517314f297ac39e1c00da3fb734bb40fcfecc9492ee2e67936fc910f8cf9568` (content hash `220886a7cf12…`, checked by the driver) |
| `tokenomics/_pod_ckpts/exp0066/orderer_scorer.pt` | `experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt` | `d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c` |
| `tokenomics/_pod_ckpts/exp0066/exp0057_episodes.csv` | `experiments/EXP-0057-decoder-check/artifacts/episodes.csv` (git-lfs: pull it first) | `69656f9e2034bc2f41049591d492a70e692cd9d770b10e60d14291021d9d593c` |

```
cd /srv/research/tokenomics
mkdir -p tokenomics/_pod_ckpts/exp0066
cp experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json tokenomics/_pod_ckpts/exp0066/
cp experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt tokenomics/_pod_ckpts/exp0066/
cp experiments/EXP-0057-decoder-check/artifacts/episodes.csv tokenomics/_pod_ckpts/exp0066/exp0057_episodes.csv
sha256sum tokenomics/_pod_ckpts/exp0066/*
python3 run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu \
  --python python3.12 --vcpu 8 --mem 16 --disk 30 --pod-name exp0066-split-need \
  --verify-source-commit \
  --train "-u -m tokenomics.eval.split_need \
    --atlas /workspace/tokenomics/_pod_ckpts/exp0066/atlas_v2.json \
    --orderer-ckpt /workspace/tokenomics/_pod_ckpts/exp0066/orderer_scorer.pt \
    --exp0057-episodes /workspace/tokenomics/_pod_ckpts/exp0066/exp0057_episodes.csv \
    --workers 8 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0066" \
  --pull "/workspace/tokenomics/runs/exp0066:./experiments/EXP-0066-split-need"
rm -rf tokenomics/_pod_ckpts/exp0066
```

The driver refuses a real run unless the atlas content hash, the orderer SHA-256 and the EXP-0057
`episodes.csv` SHA-256 are the registered ones, the horizon is 100, the commit is 40-hex and no
filter is set; it exits 2 if any row is not neutral against EXP-0057 or the observer's classes do
not add up. `--verify-source-commit` needs a clean tree: commit 4.5 first.

**Uncertainties.**
- Only the first confirm episode of the two loaded worlds was timed. 42-0031's TS job took
  52–55 s in task 4.4 (observer on), about the cell price (54.4 s; up to 1% above it).
- The spread between TS's two replicates at 42 (49.8 vs 54.4 s) is larger than the observer's
  measured cost, so the overhead column is noise-level on the TS arms.

## §EXP-0067 (task 5.5, uTWF; re-sized after the 2026-11-02 rework)

**Question.** How much will the uTWF arm-compare run (`python -m tokenomics.eval.decoder_check
--mode arm-compare --arm utwf`, task 5.5) cost on a pod, and on which shape?
- The real run is 213 jobs: EXP-0057's 71 world-episodes (43 at 12 producers over 23 worlds,
  28 at 42 over 14 worlds) × 3. Each world-episode runs uTWF × 2 replicates plus one RR
  neutrality replay.
- Horizon 100, `entity_planner` producer, tracer on.
- RR, MWWk and TS are read from EXP-0057's rows and never rerun. No arm loads torch.
- The rework replaces the WF arm (`mww_waterfill`) with uTWF. The job count is unchanged; the
  per-job cost changed, so the section is re-measured.

**Method.**
- A throwaway probe (`/tmp/impl_m4_5_5/timing_55.py`; its core loop is quoted in
  `notes/5.5.md`) builds single jobs exactly as `run_arm_compare` builds them (`world_entry`,
  `compare_arm("utwf")`, `decoder_arm(RR)`, `build_jobs`). It times
  `decoder_check.run_decoder_job`, the driver's own worker.
- Checks on every timed row: its digests against EXP-0057's rows of the same world-episode, and
  the RR row against EXP-0057's RR row on every compared cell (`split_need.neutrality`). No
  problem on any row.
- The pool is simulated with task 2.3's `extrapolate_23.simulate`: list scheduling in
  `longest_first` order, pod = 0.646 × local, 3.5 s worker start, 60 s parent, billed = vCPU ×
  (10 min setup + wall).

**Local box.** 2 CPUs, heavily shared during this measurement: other agents' test suites kept
the load average at 2.5–4.8. `OMP_NUM_THREADS=1`, one command at a time, the confirm's first
episode, horizon 100.

### Measured seconds

| world | arm | r0 | r1 | quiet-box reference (round 1) |
|---|---|---|---|---|
| 12-0352 | uTWF | 4.59 | 4.49 | MWWk 3.74 (§EXP-0057) |
| 12-0352 | RR | 3.47 | – | 3.34 |
| 42-0076 | uTWF | 32.90 | 32.35 | MWWk 22.64 (§EXP-0057) |
| 42-0076 | RR | 51.83, repeat 24.15 | – | 22.96 |

- The first 42-0076 RR timing (51.8 s) is a contention spike. The repeat (24.2 s) and round 1's
  quiet-box value (23.0 s) agree.
- uTWF costs ≈ 1.3–1.4 × RR / MWWk: it runs Algorithm 1 twice per agent and product turn on
  exact `Fraction`s. Some of the gap is the load on the box.
- Peak RSS per job process is 51–53 MB (no torch). The parent's `spec.json` environment
  manifest imports torch once (CLI dry run: 632 MB peak), so cpu8 needs ≈ 0.6 + 8 × 0.1 GB, far
  inside `--mem 16`.

### Extrapolation (213 jobs)

| model | local s (12 / 42) | pod compute vCPU-h | longest job (pod) |
|---|---|---|---|
| **cell, the 51.8 s RR spike kept (the estimate)** | 544 / 3,294 = 3,838 | **0.69** | 33.5 s (42 RR) |
| cell, the spike replaced by its repeat | 544 / 2,519 = 3,063 | 0.55 | 21.3 s (42 uTWF) |
| world (§EXP-0057's per-world RR × this section's uTWF / RR, spike excluded) | 289 / 1,553 = 1,842 | 0.33 | 20.3 s |

| shape | wall (estimate) | workers busy | **billed (estimate)** | stop = 2 × est. | billed, setup 3 min / 1 min |
|---|---|---|---|---|---|
| **cpu8** | **6.2 min** | 0.99 | **2.16** | **4.3 (32 min of pod life)** | 1.23 / 0.96 |
| cpu16 | 3.6 min | 0.98 | 3.64 | 7.3 (27 min) | 1.77 / 1.24 |

- Setup is ≈ 60% of the cpu8 bill (1.33 of 2.16 at 10 min). EXP-0057 and EXP-0066 set up in
  under a minute; at 1 min cpu8 bills ≈ 1.0.
- cpu16 costs +1.5 vCPU-h at 10 min setup to save ≈ 2.6 min of wall.
- Against round 1's WF sizing (cpu8 1.94 billed, stop 3.9), this is +0.2. The stop moves
  3.9 → 4.3.

**Recommended shape: cpu8** (task 5.6's `[remote:cpu8]`): a fresh secure pod (`--create-pod
--vcpu 8 --mem 16`, `--workers 8`).
- **Estimate 2.2 billed vCPU-h**: 0.69 compute + 1.33 setup, plus parent and tail.
- **≈ 6 min driver wall.**
- **Hard stop 4.3 vCPU-h = 32 min of cpu8 pod life** (2 × estimate). `run_pod_training.py` has
  no stop flag; the experiment implementer enforces the stop by wall clock (carry-forward).

Under the "prefer larger pods" rule, cpu16 estimates 3.6 with a stop of 7.3. Both fit under the
40 vCPU-h auto-approval.

### Pod command (task 5.6)

Stage the three inputs into the git-ignored `tokenomics/_pod_ckpts/` (`.gitignore:24`). The
driver syncs the inner package dir only, and `--verify-source-commit` still sees a clean tree.
Check the SHA-256s after copying:

| staged file | source | SHA-256 |
|---|---|---|
| `tokenomics/_pod_ckpts/exp0067/atlas_v2.json` | `experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json` | `7517314f297ac39e1c00da3fb734bb40fcfecc9492ee2e67936fc910f8cf9568` (content hash `220886a7cf12…`, checked by the driver) |
| `tokenomics/_pod_ckpts/exp0067/orderer_scorer.pt` | `experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt` | `d25388bfa33ea3fc82aebbac48c75c886ed5c091f6016f1ce36eeb4293744a8c` (pinned by SHA-256 only: EXP-0057's TS rows came from it; no arm plays it) |
| `tokenomics/_pod_ckpts/exp0067/exp0057_episodes.csv` | `experiments/EXP-0057-decoder-check/artifacts/episodes.csv` (git-lfs: pull it first) | `69656f9e2034bc2f41049591d492a70e692cd9d770b10e60d14291021d9d593c` |

```
cd /srv/research/tokenomics
mkdir -p tokenomics/_pod_ckpts/exp0067
cp experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json tokenomics/_pod_ckpts/exp0067/
cp experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt tokenomics/_pod_ckpts/exp0067/
cp experiments/EXP-0057-decoder-check/artifacts/episodes.csv tokenomics/_pod_ckpts/exp0067/exp0057_episodes.csv
sha256sum tokenomics/_pod_ckpts/exp0067/*
python3 run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu \
  --python python3.12 --vcpu 8 --mem 16 --disk 30 --pod-name exp0067-utwf \
  --verify-source-commit \
  --train "-u -m tokenomics.eval.decoder_check --mode arm-compare --arm utwf \
    --atlas /workspace/tokenomics/_pod_ckpts/exp0067/atlas_v2.json \
    --orderer-ckpt /workspace/tokenomics/_pod_ckpts/exp0067/orderer_scorer.pt \
    --exp0057-episodes /workspace/tokenomics/_pod_ckpts/exp0067/exp0057_episodes.csv \
    --workers 8 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0067" \
  --pull "/workspace/tokenomics/runs/exp0067:./experiments/EXP-0067-waterfill-router"
rm -rf tokenomics/_pod_ckpts/exp0067
```

**What the driver refuses and exits 2 on.**
- It refuses a real run unless:
  - the atlas content hash, the orderer SHA-256 and the EXP-0057 `episodes.csv` SHA-256 are the
    registered ones;
  - the roster is frozen, the horizon is 100 and the commit is 40-hex;
  - no filter is set.

  It also refuses before any job if EXP-0057's file lacks any compared RR / MWWk / TS row.
- It exits 2 on contract failures, missing rows, a digest that differs across arms or from
  EXP-0057's, or an RR replay that is not neutral against EXP-0057.
- `--verify-source-commit` needs a clean tree: commit 5.5 first.

**Uncertainties.**
- The timings were taken on a contended box. The estimate keeps the slowest measured value per
  cell, including the 51.8 s RR spike, so it is conservative. Without the spike, cpu8 bills 2.0
  with a stop of 4.0.
- Only the first confirm episode of the two loaded worlds was timed.
- uTWF's per-turn cost grows with the eligible producers per product (Algorithm 1 is
  O(min(N, a)) per pour, on `Fraction`s). 42-0076 (f14) is the costliest choice level in the
  sample, so the 42 cell should bound the 42 jobs.

## §EXP-0058–0060 (task 5.4)

**Result first.** The paired MDE gives **8-episode reads**: TR − RR is 2.50% at 4 episodes and
1.77% at 8. **The 8-episode tranche-A launch-fit exceeds 95 under every planning model that keeps
D10's 10-minute pod setup**, so the task ends by reporting to the supervisor (tasks.md 5.4).
Nothing below is launched or authorised by this section.

**Question.** How much will the step-2 runs cost on pods? Which episode count does D7's power
rule pick? Do the runs fit tranche A?
- EXP-0058 is the bar precompute and the gradient probe, on one cpu8 pod (supervisor decision 1,
  notes/carry-forward.md).
- EXP-0059 is the pilot.
- EXP-0060a–e are the main runs and the validation reads.

**Pre-registration.** `tokenomics/tokenomics/train/registry/es_prereg_m4_v1.json`, SHA-256
**`b9d449c15db6860950aab8dbc49bc0e3d67635d4d50e6f1c817d1e04fedcbabf`** (= `PREREG_SHA256`,
re-checked with `sha256sum`).

### Method
- **Horizon 12, the drivers' own timings.** Each job's `episode_seconds` is recorded by the
  driver.
  - `order_es_worlds --mode bar --dry-run --atlas-worlds 2 --target 2 --pool 1
    --episodes-per-world 1 --horizon 12 --shock-profile shocks-i-smoke`: the first two loaded
    atlas training worlds, 12-0005 and 12-0060, one seed each, all 10 bar arms.
  - `order_es --mode train --dry-run --arm A2 --generations 1 --pairs 2 --monitor-every 1` on
    that bar: A2 members and the monitor (centre and A0 at R = 2).
  - `order_es_validate --mode validate --dry-run --episodes 1` on 12-0050 and on 42-0066:
    every validation arm, including the shared-only ablation.
- **Horizon 100, a direct cross-check.** A probe (`/tmp/impl_m4_5_4/h_probe.py`) runs the
  drivers' workers on the bar's jobs: `run_bar_job` for TR / RR, and `run_es_job` for A0 (the
  monitor job) and an A2 member. It ran on 12-0060 and 12-0005, one episode per command, plus the
  validation dry run at horizon 100 on 12-0050. Each probe row reproduces its bar row's units
  (TR 68, A0 132 at horizon 12).
- **Box.** 2 CPUs, shared. Commands ran one at a time with BLAS / torch at 1 thread, no worker
  pool (`--workers 1`), at `nice -n 19`. The 1-minute load average (before / after) is
  recorded per command below.

### Measured seconds (local)

| what | world | arm | seconds | load (1 min) |
|---|---|---|---|---|
| bar, h12 | 12-0005 / 12-0060 | TR | 0.504 / 0.488 | 1.00 → 1.00 |
| | | A0 | 0.442 / 0.456 | |
| | | RR | 0.232 / 0.220 | |
| | | MWW r0, r1 | 0.219, 0.218 / 0.245, 0.247 | |
| | | roster (blend, even, obs_price, prop., shock_reactive) | 0.11–0.23 / 0.12–0.27 | |
| ES dry train, h12 | 12-0060 / 12-0005 | A2 member (centre and ± pairs, 5 each) | 0.445–0.462 / 0.448–0.484 | 0.93 → 0.95 |
| | | monitor A2 centre (R = 2, generations 0 and 1) | 0.448–0.552 / 0.442–0.497 | |
| | | monitor A0 (R = 2) | 0.394, 0.457 / 0.376, 0.371 | |
| probe, h12 | 12-0060 | TR / A0 / A2 | 0.59 / 0.48 / 0.58 | 0.57–0.67 |
| **probe, h100** | **12-0060** | **RR / TR / A0 / A2** | **1.78 / 3.65 / 3.11 / 3.61** | 0.74–1.10 |
| **probe, h100** | **12-0005** | **RR / TR / A0 / A2** | **1.57 / 3.44 / 2.85 / 3.31** | 1.17–1.25 |
| validation dry run, h12 | 12-0050 | candidate (A2) r0, r1 / ablation / A0 / TR / RR / MWW | 0.455, 0.473 / 0.728, 0.586 / 0.398, 0.403 / 0.478 / 0.206 / 0.210, 0.212 | 1.64 → 1.69 |
| **validation dry run, h100** | **12-0050** | **candidate / ablation / A0 / TR / RR / MWW** | **3.58, 3.81 / 4.48, 4.35 / 3.26, 3.26 / 3.46 / 1.60 / 1.72, 1.72** | 0.62 → 0.92 |
| validation dry run, h12 | 42-0066 | candidate / ablation / A0 / TR / RR / MWW | 6.18, 6.33 / 9.07, 9.24 / 5.30, 5.27 / 6.84 / 3.46 / 4.06, 3.70 | 1.50 → 1.42 |

Peak RSS was 470–490 MB per command; torch is loaded once per process. The horizon-12 A2 times
are at the registered initial θ (the centre). Perturbed members cost the same: 0.445–0.484 s
against the centre's 0.462–0.476 s.

### Per-order / TR ratio (design D10 assumed ≈ 1.3)

| size, horizon | A2 (per-order, residual) / TR | A0 / TR | ablation / TR | ablation / candidate |
|---|---|---|---|---|
| 12, h100 (12-0060 / 12-0005 / 12-0050) | 0.99 / 0.96 / 1.10 | 0.85 / 0.83 / 0.94 | – / – / 1.29 | 1.18 |
| 12, h12 (12-0060 probe; 12-0050 validation) | 0.98; 0.99 | 0.82; 0.84 | 1.52 | 1.54 |
| 42, h12 (42-0066 validation) | 0.93 | 0.77 | 1.35 | 1.46 |

- The per-order policy costs **0.96–1.10 × TR at 12 producers and 0.93 × at 42**, not 1.3 ×.
- The shared-only ablation is the most expensive arm: 1.29 × TR at 12 (horizon 100) and
  1.35 × at 42 (horizon 12). This answers carry-forward 5.4, "time the shared-only ablation at
  42", at horizon 12. No 42-producer horizon-100 episode was run locally: notes/5.2 records a
  24-minute memory thrash for 42-producer episodes on this box.

### Extrapolation to horizon 100 (`/tmp/impl_m4_5_4/size_54.py`)

- **Per-job price.** A job's local horizon-100 seconds = its world's RR seconds × the largest
  measured arm / RR ratio at that size. At 42 producers it is TR × the arm / TR ratio.
  - The ratios at 12, horizon 100:

    | arm | member (A1 / A2) | A0 | TR | MWW | roster rules | ablation | `portable` |
    |---|---|---|---|---|---|---|---|
    | ratio to RR | 2.38 | 2.04 | 2.49 | 1.15 | 1.22 | 2.80 | 3.7 (assumed 1.5 × TR, for EXP-0061 only) |

    TR 2.49 and MWW 1.15 are §EXP-0057's on 12-0060; the roster rules use the horizon-12 bar
    rows.
  - At 42 producers: the arm / TR ratios above (member 0.93, A0 0.78, ablation 1.35), with TR
    and RR priced from §EXP-0057.
- **Two price levels.**
  - **cell**: every 12-producer job at 12-0352, the slowest of the 23 atlas training worlds (RR
    3.91 s). Every 42-producer job at 42-0076 (TR 60.93 s, RR 22.71 s).
  - **world**: each atlas world's own RR seconds (§EXP-0057, `probes/timing_23.jsonl`; mean 1.81
    s at 12, 14.89 s at 42). The 41 fresh draws, which have the same lever distribution, are
    priced at the mean.
  - At 12 the two differ by 2.2×, because 12-0352 is an outlier. The ES runs are almost all
    12-producer jobs drawn over 64 worlds, so the world level is the central estimate and the
    cell level the conservative bound.
- **Job counts.**

  | run | jobs |
  |---|---|
  | bar | 64 worlds × 2 episodes × 10 arms = 1,280, plus a replacement wave of 5 dropped fresh draws (12% of 41, the carry-forward's upper drop rate) and one tail wave: 1,400 |
  | probe | 2 arms × 24 members × 12 world-episodes = 576 |
  | pilot | 20 × 300 + 3 × 32 centre + 32 A0 = 6,128 |
  | EXP-0060a (resumed; this dir replays A0's 32 monitor jobs once) | 20 × 300 + 2 × 32 + 32 = 6,096 |
  | EXP-0060b / d | 40 × 300 + 5 × 32 + 32 = 12,192 |
  | seed-0 validation | 16 worlds (13 at 12, 3 at 42) × episodes × 14 (2 candidates × (R = 2 + ablation R = 2), A0 × 2, RR, MWW × 2, TR) |
  | final validation | × 4 (the seed-1 candidate and its ablation) |

- **Pool model** (task 2.3's `extrapolate_23.simulate`): list scheduling, longest first, pod =
  0.646 × local, 60 s parent, 4 s per pool opened.
  - ES generations are barriers, simulated one by one. 300 jobs on 32 workers leave workers
    idle at each generation's end, so busy = **0.77–0.84** on cpu32. D10 assumed full use.
  - Billed = vCPU × (setup + wall).

### Minimum detectable effect (design D7; `order_es_validate --mode power`, frozen rule)

Inputs: EXP-0052 confirm rows on the 9 gating validation worlds (12-0050 0089 0092 0109 0330
0350 0384 0464 0494); `episodes.csv` `0e22ddd0…`, atlas `220886a7…`, reclassified
`worlds.csv` `9993bb66…`. Mean validation bar **1,831.4**. The pooled within-world SD of the
paired per-episode differences has df = 7: 12-0330 and 12-0464 have one confirm episode.

| pair (stochastic arms R = 2) | s | MDE at 4 episodes | at 8 episodes |
|---|---|---|---|
| MWW − RR | 68.95 | 34.5 units = 1.882% | 24.4 = 1.331% |
| **TR − RR (deciding: the max)** | 91.65 | **45.8 = 2.502%** | **32.4 = 1.769%** |

- **D7 rule: 2.502% > 2% at 4 episodes → 8 episodes; 1.769% ≤ 3% at 8 → not to the user.** The
  reviewer's values are confirmed exactly.
- **Reported only, never used:**
  - Single-arm `se` column (reclassified `worlds.csv`), world mean 3·√Σ se_w² / 9: 48.1 units
    (2.63%) at the confirm's counts. Scaled to n episodes as se_w·√(e_w / n): 33.2 (1.81%) at 4
    and 23.5 (1.28%) at 8.
  - Atlas formula (`atlas.standard_error` at the read's counts, 5.3's `atlas_se`, world-mean
    term 3·√Σ SE_w² / 9): MWW − RR 30.8 / 21.8 units (1.68% / 1.19%), TR − RR 24.2 / 17.1
    (1.32% / 0.93%) at 4 / 8.
  - D7's text quotes "44 / 31" for the single-arm column. My scaling reproduces its "48" at the
    confirm's counts but not 44 / 31, so the convention behind those figures is unknown. Neither
    value enters the rule.

### Estimates per run (billed vCPU-h at a 10-min setup; stop = 2 × estimate, capped at 40)

| run | shape | compute (pod vCPU-h), world / cell | wall min, world / cell | **estimate, world** | estimate, cell | stop, world (cell) |
|---|---|---|---|---|---|---|
| EXP-0058 bar + probe (one pod, 2 min restart) | cpu8 | 0.63 + 0.44 / 1.37 + 0.96 | 6.1 + 4.4 / 11.6 + 8.3 | **3.0** | 4.3 | 6.0 (8.5); **probe step alone 0.85 → stop 1.7 (cell 2.8) ≤ 4** |
| EXP-0059 pilot A1 s0 g1–20 | cpu32 | 4.74 / 10.23 | 12.6 / 23.8 | **12.1** | 18.0 | 24.1 (36.1) |
| EXP-0060a A1 s0 resumed g21–40 | cpu32 | 4.72 / 10.17 | 12.5 / 23.6 | **12.0** | 17.9 | 24.0 (35.8) |
| EXP-0060b A2 s0 g1–40 | cpu32 | 9.44 / 20.36 | 24.1 / 46.4 | **18.2** | 30.1 | 36.4 (**40 cap; only 1.33 × the cell estimate**) |
| EXP-0060c seed-0 validation, 8 episodes | cpu16 | 3.23 / 5.59 | 13.2 / 22.1 | **6.2** | 8.6 | 12.4 (17.1) |
| (EXP-0060c at 4 episodes, for comparison) | cpu16 | 1.61 / 2.79 | 7.2 / 11.6 | 4.6 | 5.8 | 9.2 (11.5) |
| EXP-0060d seed 1, selected arm (conditional) | cpu32 | as 0060b | | **18.2** | 30.1 | 36.4 (40) |
| EXP-0060e final validation, 8 episodes (conditional) | cpu16 | 1.13 / 1.95 | 5.4 / 8.4 | **4.1** | 4.9 | 8.2 (9.8) |

- cpu16 for the validation reads: cpu32 would cost +3 vCPU-h of setup to save ≈ 6–10 min.
- Every stop is ≤ 40, and the probe step is ≤ 4.
- At a **1-minute setup**, which EXP-0057 / 0066 actually saw, the world-level estimates fall
  to:

  | run | 0058 | 0059 | 0060a | 0060b / d | 0060c | 0060e |
  |---|---|---|---|---|---|---|
  | estimate | 1.8 | 7.3 | 7.2 | 13.4 | 3.8 | 1.7 |

### Launch-fit, 8-episode reads (tranche A, authorised 100; check line 95)

"Spend before" uses billed actuals for runs already done (EXP-0057 1.41, EXP-0066 ≈ 1.7) and
planned estimates after them. EXP-0067 (task 5.6, not yet run) is counted at its §EXP-0067
estimate 2.16 / stop 4.33. EXP-0061 is re-priced with this section's per-job prices: 12 held-out
worlds at 12 + 8 at 42, 8 episodes, candidate + ablation + A0 at R = 2, RR, MWW × 2, TR and
`portable`. Task 8.2 owns its final sizing. EXP-0062 keeps D10's 1.6 / 3.

| run | estimate, world | stop | spend before + stop |
|---|---|---|---|
| EXP-0067 uTWF (planned) | 2.16 | 4.33 | 3.1 + 4.3 = 7.4 |
| EXP-0058 bar + probe | 3.0 | 6.0 | 5.3 + 6.0 = 11.3 |
| EXP-0059 pilot | 12.1 | 24.1 | 8.3 + 24.1 = 32.4 |
| EXP-0060a | 12.0 | 24.0 | 20.3 + 24.0 = 44.3 |
| EXP-0060b | 18.2 | 36.4 | 32.3 + 36.4 = 68.7 |
| EXP-0060c (8 ep) | 6.2 | 12.4 | 50.5 + 12.4 = 62.9 |
| EXP-0060d (conditional) | 18.2 | 36.4 | 56.7 + 36.4 = **93.1** |
| EXP-0060e (8 ep, conditional) | 4.1 | 8.2 | 74.9 + 8.2 = 83.1 |
| EXP-0061 held-out (8 ep, re-priced) | 8.2 | 16.3 | 79.0 + 16.3 = **95.3 > 95** |
| EXP-0062 | 1.6 | 3.0 | 87.2 + 3.0 = 90.2 |
| success-path total | **88.8** | | max **95.3** |

The maximum under each model and setup assumption:

| model, setup | max, EXP-0061 re-priced | max, D10's EXP-0061 (8.1 / 17) | §EXP-0058–0060 alone | success-path total |
|---|---|---|---|---|
| world, 10 min (the table above) | **95.3** | **96.0** | 93.1 | 88.8 |
| cell, 10 min | **141.9** | 136.1 | 124.1 (EXP-0060d) | 132.1 |
| world, 1 min | 65.5 | 70.8 | — | 61.1 |
| cell, 1 min | 111.9 | 110.9 | — | 104.5 |

**Verdict: the 8-episode launch-fit exceeds 95.** It exceeds even on the central (world) model
with D10's 10-minute setup, at EXP-0061, and far exceeds on the conservative cell model. Only the
observed sub-minute setup together with world-level prices fits (≤ 71). Per tasks.md 5.4 this is
reported to the supervisor, who decides between three options:
- accept the world model as the planning estimate;
- re-base setup on the observed sub-minute setups;
- take a budget or plan change to the user (e.g. fewer 42-producer validation jobs).

None of these is decided here.

**Why it moved against D10.**
- (+) The minibatch barrier leaves cpu32 workers 16–23% idle.
- (+) 8-episode reads, and the shared-only ablation at 1.3–1.35 × TR on 2 candidates.
- (+) EXP-0061 re-priced: 42-producer jobs at 42-0076's TR level.
- (−) The per-order policy costs ≈ 1.0 × TR, not 1.3 ×.
- (±) D10 priced member episodes at 5.9 s local. This section prices them at 4.3 s (world) or
  9.3 s (cell).

**Uncertainties.**
- Horizon 12 → 100 measured 6.2–6.4 × on these worlds, sub-linear because of the fixed
  per-episode cost. The horizon-100 numbers above are measured directly at 12 producers, and
  derived from horizon 12 with §EXP-0057's 42-producer horizon-100 TR only at 42.
- The ablation's premium over the candidate falls from 1.5 × at horizon 12 to 1.18 × at
  horizon 100 (12-0050). Using the horizon-12 ratio at 42 is conservative.
- The bar's drop and replacement rate is assumed at the carry-forward's upper 12%. The A0 death
  rate is unknown until EXP-0058.
- The fresh draws' per-world cost is assumed equal to the atlas training mean.
- The plan text still has the older placement:
  - tasks.md 6.1 / 6.2 put the bar inside the pilot's cpu32 pod (6.2), with 6.1 = probe only
    (2.0 / 4);
  - this section sizes supervisor decision 1 (bar then probe on EXP-0058's cpu8 pod, 6.2 reusing
    the bar), as notes/5.1 / 5.2 and their commands do;
  - the plan-text pass should align tasks.md 6.1 / 6.2.

**Supervisor decision (2026-11-02, under the user's M4 budget authority, notes/user-decisions.md).**
Task 5.4's stop fired: with 8-episode reads the world model's max spend + stop is 95.3 (EXP-0061),
over the 95 reporting line but under the approved tranche A of 100; the cell model gives 141.9.
- **Planning basis: the world model at D10's 10-minute setup.** It prices each atlas world at its
  own measured time; the cell model prices every world as the slowest (12-0352, 2.2 × the mean)
  and is a worst case, kept as a bound, not a plan. The observed setup (< 1 min on EXP-0057 /
  0066 / 0067) is extra margin, not planned on.
- **The binding guard is the per-run launch-fit rule on billed spend** (design D10): before each
  launch, billed spend so far in tranche A + that run's hard stop ≤ 100, recorded in the task.
  Billed so far in tranche A: EXP-0057 ≈ 1.41, EXP-0066 ≈ 1.7, EXP-0067 ≈ 2.0 (16 vCPU, ≈ 7–8 min
  of pod life) ≈ 5.1, against D10's planned 3.5 before EXP-0058.
- **On a launch-fit miss later:** the supervisor decides (budget authority). The validation
  sample is frozen in the pre-registration and is not trimmed to fit; the response is a recorded
  tranche extension, never a changed read.
- tasks.md 6.x / 7.x "Resources" lines and the 6.1 / 6.2 placement are aligned in the plan-text
  pass at the 5.4 commit.
