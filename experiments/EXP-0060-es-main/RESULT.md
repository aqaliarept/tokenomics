# EXP-0060: ES main runs and validation reads (in progress)

**Status: in progress.** Task group 7 of `add-m4-orderer-redesign`. Done so far:
- **EXP-0060a** (task 7.1): A1 seed 0, resumed for generations 21–40;
- **EXP-0060b** (task 7.2): A2 at rank 2 (A2-R2), seed 0, generations 1–40.

Still to run: the seed-0 validation read (7.3), and the seed-1 run and final read if licensed
(7.4–7.5).

## EXP-0060a: A1 seed 0, generations 21–40 (resumed from EXP-0059)

**Generation 40 clears the training floor.** On the training-monitor set,
g_tr = **+73.2** units against F_tr = **54.75**. Not an acceptance read: D7's seed-0 reading
comes from validation (7.3), after A2 (7.2).

### In plain English

**The question.** The pilot (EXP-0059) left the router +29.8 units ahead of its starting point
A0 on the fixed 16-world-episode monitor set after 20 generations. That was below the floor
(54.75) but still rising. The supervisor's offline diagnosis (DIAGNOSTICS 27, checks 1–2) found
losses shrinking and gains growing, which reads as under-trained, so this run continued the same
training for 20 more generations.

**What happened.**
- **The gain kept rising and now clears the floor.** On the monitor set the center went from
  1,979.7 consumer units at generation 20 to 2,006.7 at generation 30 and 2,023.1 at generation
  40, against A0's 1,949.9. So g_tr is **+73.2 (≈ 3.8%) > F_tr 54.75**.
- **It also became broader:**
  - 10 of 16 world-episodes are now above A0 (7 at generation 20);
  - the summed losses shrank from −319 to −118 while the summed gains grew from +795 to
    +1,289;
  - the median world-episode moved from about −10 to about +13.
- **Still uneven.** One world (atlas 12-0121, +515) is the largest single gain, and individual
  worlds swing by ±100–180 units between checkpoints.
- **Still improving:** +16.4 over the last 10 generations, after +27.0 over generations 21–30.

**What changes because of it.** The training gain is now resolvable. Whether it carries to
unseen validation worlds is the seed-0 read's question (7.3). A2's main run (7.2) is the
supervisor's call.

### Details

**Monitor set** (read offline with `order_es_validate.load_candidate` / `run_stats`; generations
10 and 20 from EXP-0059):

| generation | center | A0 | g_tr | F_tr (atlas / own 3·SE) | improvement over 10 gens | up / down |
|---|---|---|---|---|---|---|
| 10 | 1962.63 | 1949.91 | +12.72 | 54.75 (54.75 / 45.00) | +12.72 | 8 / 8 |
| 20 | 1979.69 | 1949.91 | +29.78 | 54.75 (54.75 / 54.59) | +17.06 | 7 / 9 |
| 30 | 2006.69 | 1949.91 | +56.78 | 55.42 (54.75 / 55.42) | +27.00 | 12 / 4 |
| **40** | **2023.09** | 1949.91 | **+73.19** | **54.75** (54.75 / 52.64) | **+16.41** | **10 / 6** |

The per-world-episode table for generations 10–40 is in DIAGNOSTICS 27.

**Verification (task 7.1's verify line):**
- `contract_failures.csv` is header only.
- Generations 21–40 are present: `generations.csv` covers 1–40, 21–40 from this run.
- The resume pins equal EXP-0059's: `spec.json` `pins.training` is identical, and
  `resume_from` = generation 20, checkpoint content hash `0e22b44e…`.
- Final θ `169f402915ec…` = `candidate.json` = `checkpoints/gen-0040.json`. Commit `7b16e55…`,
  pre-registration v2.

**Cost:**

| pod | shape | life (UTC) | billed |
|---|---|---|---|
| `b6qr79wkykgzhb` | secure 32 vCPU | 20:32:51 launch → terminated by 20:53:27 (≤ 20.6 min) | **≈ 11.0 vCPU-h** (upper bound) |

- **Estimate and stop.** Estimate ≈ 14 at a 10-min setup; stop 28 (2 × the measured-rate
  estimate). Launch-fit 23.0 + 28 = 51.0 ≤ 125. Tranche A billed so far ≈ 34.0.
- **Capacity:** cpu32 was available this time.
- **Flow:** default detached (pull, stop, terminate). `runpodctl pod list` = [] afterwards. No
  local step ran while the pod billed.

**Files:** `artifacts/a1-s0/`, the run dir as pulled: `spec.json`, `candidate.json`,
`checkpoints/`, `generations.csv`, `member_fitness.csv`, `monitor.csv`, `monitor_summary.csv`,
`members.csv` (git-lfs) and a header-only `contract_failures.csv`.

## EXP-0060b: A2-R2 seed 0, generations 1–40

**Generation 40 also clears the training floor:** g_tr **+83.7** against F_tr **54.75** on the
training-monitor set. Not an acceptance read: selection between A1 and A2 and D7's seed-0
reading are 7.3's.

### In plain English

**The question.** A2 is the larger per-order model: A1's residual plus a rank-2 bilinear term
over order and producer features, 156 parameters against A1's 44. It trains at rank 2 because
EXP-0069's probe found its rank-4 estimates anti-aligned, and the registered rule then sets
R = 2. This run trains it from scratch for 40 generations under the same pre-registration v2
settings as A1, so 7.3 can compare the two on validation worlds.

**What happened.**
- **A2-R2 started like A1, stalled, then caught up.** Its gain over A0 on the monitor set was
  +13.1 at generation 10, +11.1 at 20 (a small dip), +55.7 at 30 and **+83.7 at 40**. A1 was
  at +12.7 / +29.8 / +56.8 / +73.2.
- **It clears the floor and is broader than A1 at generation 40:**
  - 12 of 16 world-episodes are above A0 (A1: 10);
  - summed losses are −94.5 (A1: −117.5);
  - the median world-episode is ≈ +54 (A1: ≈ +13).
- **Still uneven and still rising.** atlas 12-0121 (+418) is again the largest gain, and the
  run gained +28.0 over the last 10 generations.

**What changes because of it.** Both seed-0 arms are trained to generation 40. The seed-0
validation read (7.3) compares A1 and A2-R2 against A0, RR, MWW and TR on the 12-producer
validation worlds, selects one, and gives D7's reading.

### Details

**Monitor set** (offline, `order_es_validate.load_candidate` / `run_stats`):

| generation | center | g_tr | F_tr (atlas / own 3·SE) | improvement over 10 gens | up / down | summed gains / losses |
|---|---|---|---|---|---|---|
| 10 | 1963.00 | +13.09 | 54.75 (54.75 / 35.01) | +13.09 | 9 / 7 | +515.5 / −306.0 |
| 20 | 1961.00 | +11.09 | 61.60 (54.75 / 61.60) | −2.00 | 10 / 6 | +486.0 / −308.5 |
| 30 | 2005.59 | +55.69 | 54.75 (54.75 / 48.71) | +44.59 | 11 / 5 | +1116.5 / −225.5 |
| **40** | **2033.56** | **+83.66** | **54.75** (54.75 / 44.80) | **+27.97** | **12 / 4** | **+1433.0 / −94.5** |

- **Per world-episode at generation 40** (center − A0): −51.0, −21.5, −21.5, −0.5, +30.0,
  +39.5, +40.0, +47.0, +60.5, +61.0, +71.0, +91.0, +151.0, +197.5, +226.5, +418.0.
- **Verification (as 7.1, for generations 1–40):**
  - `contract_failures.csv` is header only, and `generations.csv` covers 1–40;
  - the pins are arm A2-R2 (rank 2, 156 parameters), minibatch 24, prereg v2 `511e939a…`, the
    EXP-0058 bar hashes, commit `7b16e55…`, with no resume;
  - final θ `bfafd85f2f04…` = `candidate.json` = `checkpoints/gen-0040.json`.

**Cost:**

| pod | shape | life (UTC) | billed |
|---|---|---|---|
| `g4jvzlk7xt8hnm` | secure 32 vCPU | 20:55:39 launch → terminated by 21:27:47 (≤ 32.1 min) | **≈ 17.1 vCPU-h** (upper bound) |

- Stop 44 (2 × the EXP-0060a-rate estimate, within the recorded exception). Launch-fit
  34.0 + 44 = 78.0 ≤ 125. Tranche A billed so far ≈ 51.1.
- Default detached flow. `runpodctl pod list` = [] afterwards. No local step ran while the pod
  billed.

**Files:** `artifacts/a2-s0/`, the run dir as pulled. `members.csv` is git-lfs.

## Conclusions

- **EXP-0060a:** A1 seed 0 at generation 40 has g_tr +73.2 > F_tr 54.75 on the training-monitor
  set. The gain broadened (10 / 16 up, losses −118) and was still rising (+16.4 over the last
  10). DIAGNOSTICS 27 → diagnosed: under-trained.
- **EXP-0060b:** A2-R2 seed 0 at generation 40 has g_tr +83.7 > F_tr 54.75 (12 / 16 up,
  losses −94.5, still rising +28.0).
- **Next:** the seed-0 validation read (7.3: A1 vs A2-R2, selection and D7's reading). The
  supervisor decides.
