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
