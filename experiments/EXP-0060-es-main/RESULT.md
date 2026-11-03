# EXP-0060: ES main runs and validation reads (in progress)

**Status: in progress.** Task group 7 of `add-m4-orderer-redesign`. Done so far:
**EXP-0060a** (task 7.1), A1 seed 0 resumed for generations 21–40. Still to run: A2 seed 0
(7.2), the seed-0 validation read (7.3), and the seed-1 run and final read if licensed (7.4–7.5).

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

## Conclusions

- **EXP-0060a:** A1 seed 0 at generation 40 has g_tr +73.2 > F_tr 54.75 on the training-monitor
  set. The gain broadened (10 / 16 up, losses −118) and was still rising (+16.4 over the last
  10). DIAGNOSTICS 27 → diagnosed: under-trained.
- **Next:** 7.2 (A2 seed 0, 40 generations) and the seed-0 validation read (7.3). The supervisor
  decides.
