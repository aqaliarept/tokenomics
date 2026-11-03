# EXP-0060: ES main runs and validation reads, step 2 of the M4 ladder

**Status: complete, valid. Final reading (D7, both seeds): `plateau`.**
- **Selected:** A1, seed 0.
- **Labels:** `herding` and `history_inert`, on both seeds.
- **Next (registered):** record `history_inert` and `herding`, and ask the user to release
  tranche B1, the relational check (step-3 gate).
- **Runs:**
  - EXP-0060a (7.1): A1 seed 0, resumed for generations 21–40;
  - EXP-0060b (7.2): A2-R2 seed 0, generations 1–40;
  - val-seed0 (7.3): the seed-0 read;
  - EXP-0060d (7.4): A1 seed 1, generations 1–40;
  - final (7.5): the seed-combined read.

## In plain English: what step 2 found

**The question.** Can a small learned correction to the per-order router, trained by
evolution strategies on whole-episode outcomes, beat the simple routing rules on worlds it never
trained on?

**The answer: it learns, and the learning repeats and generalises, but only up to the level of
the simple rules.**
- **It learns.** Both optimizer seeds of A1 improve on the router's starting point A0 on the
  training-monitor worlds by about +70 consumer units (+3.5–3.8%).
- **The gain holds on unseen validation worlds:** +68.7 (seed 0) and +45.3 (seed 1) over A0.
- **It does not beat the simple rules.** Against round robin it is level (+26.0 and +2.5, both
  inside the noise floor). Against MWW, the strongest simple rule, it is clearly worse (−59.0 and
  −82.5, beyond the floor). It beats the old trained orderer by about +600.
- **Other findings.** The larger model (A2-R2) is no better. The router's agent-local history
  adds nothing measurable (`history_inert`). Its choices herd (`herding`).
- **Why the reading is `plateau`** (pre-registered):
  - the training gain levelled off below the floor's size over the last 10 generations;
  - the larger model is not better;
  - the candidate is within or below the floor of MWW or of RR.

**What it licenses.** D7 says a `plateau` goes to the step-3 gate: the relational check (A2+R,
tranche B1). Releasing B1 is the user's decision. `history_inert` must be cited before history
becomes GNN edge attributes. `herding` is carried into the relational check's record.

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

## val-seed0: the seed-0 validation read (task 7.3)

**Reading: `plateau`. Selected: A1-s0. Labels: `history_inert`, `herding`.**
- **Next (D7):** seed 1 for A1 (7.4), then the step-3 gate.
- **Read:** `order_es_validate --mode validate --prereg …v2.json --episodes 8` at `7b16e55…`, on
  the 16 validation worlds × 8 episodes.
- **Gating:** D7 gates on the 9 loaded 12-producer worlds.

### In plain English

**The question.** Both trained routers (A1 and the larger A2-R2) beat their starting point A0
on the training monitor. Do they also beat the simple and the old orderers on validation worlds
they never trained on? The pre-registered references are:
- A0, the starting point;
- RR, round robin;
- MWW, makeable-weighted whole routing, the strongest simple rule;
- TR, the old trained orderer.

`success` would need the router to beat every one of them by more than each comparison's floor,
with no deaths and no worse run-down or starvation than RR.

**What happened** (mean consumer units per world-episode, 9 worlds × 8 episodes):

| comparison | A1-s0 (selected) | floor | A2-R2-s0 | floor |
|---|---|---|---|---|
| − A0 (g_val) | **+68.7** | 37.7 | +54.6 | 29.3 |
| − RR | +26.0 | 39.1 | +11.9 | 36.6 |
| − MWW | **−59.0** | 35.8 | −73.1 | 29.9 |
| − TR | +614.4 | 36.9 | +600.2 | 37.0 |
| − its shared-only ablation | +2.8 | 35.8 | +19.5 | 33.2 |

- **The training gain carries over.** On unseen worlds, A1 beats A0 by +68.7. That is close to
  its training-monitor gain (+73.2), so this is not a cross-world failure (`kill` would need
  g_val < g_tr / 3).
- **It does not beat the simple rules.** It is level with round robin (+26, inside the floor)
  and clearly below MWW (−59, beyond the floor). It is far above the old trained orderer
  (+614). So **no `success`**:
  - not every reference is beaten;
  - A1 also had 1 death;
  - its run-down share is worse than RR's (0.74 vs 0.65).
- **`plateau` holds.** All three registered conditions are true:
  - (i) both arms were still improving on the monitor set, but by less than F_tr over the last
    10 generations (A1 +16.4, A2-R2 +28.0, against 54.75);
  - (ii) A2-R2 is not better than A1: −14.1, inside the floor of 39.8. A1 is selected;
  - (iii) A1 sits within or below the floor of MWW or of RR.
- **Labels.**
  - `history_inert`: the agent-local history adds nothing measurable (A1 − its shared-only
    ablation = +2.8, floor 35.8).
  - `herding` is flagged for both arms. Under `plateau` it is recorded and carried into the
    relational check's record (D7).
  - `not_via_deconcentration` does not apply (no gain to explain).

**What it licenses.** Per D7's seed rule: seed 1 for the selected arm (A1, task 7.4), and the
final read (7.5) combines both seeds. A final `plateau` would record `history_inert` and
`herding` and ask the user to release tranche B1, the relational check. Nothing else is
licensed.

### Details
- **Verify (task 7.3's line):**
  - `contract_failures.csv` is header only;
  - 1,792 episode rows on 128 world-episodes, each with one demand / shock digest pair;
  - every played world is in the atlas validation split (no held-out world);
  - `order_es_validate --mode recompute` from `episodes.csv` and the two runs' logs gives
    **MATCH** (`plateau`).
- **Per-candidate (`reading.json`):**
  - A1-s0: mean 1,828.3; 1 death; worlds lost to RR beyond floor_w: 1 (≤ 1/6 of 9 holds);
    run-down share 0.736 against RR 0.653; starved share 1.0 against RR 1.0.
  - A2-R2-s0: mean 1,814.2; 0 deaths; 2 worlds lost to RR; run-down 0.750.
- **Gating worlds:** atlas 12-0050, 0089, 0092, 0109, 0330, 0350, 0384, 0464, 0494.
- **Reported tables:** the 42-producer, light-world and ρ-quartile tables are in
  `validation.csv`; concentration in `concentration.csv`; herding in `herding.csv`.
- **Cost:** pod `km36vk8roekqa3`, secure 16 vCPU, 21:29:26 launch → terminated by 21:42:24
  (≤ 13.0 min), **≈ 3.5 vCPU-h** billed.
  - Estimate 6.2, stop 12.4. Launch-fit 51.1 + 12.4 = 63.5 ≤ 125. Tranche A ≈ 54.6.
  - Staging under `_pod_ckpts/` only, from a worktree at 7b16e55, with the real-run checks
    passed locally first.

## EXP-0060d: A1 seed 1, generations 1–40 (task 7.4)

**Not a read.** This is the seed-1 run of the selected arm, licensed by val-seed0's `plateau`.
The final reading is 7.5's.

### In plain English

The same A1 training as seed 0, with a different optimizer seed, so different perturbation
draws and minibatches. It asks whether the learning repeats.
- **It does, at about the same level.** On the training-monitor set the gain over A0 was +5.7 at
  generation 10, +13.4 at 20, +65.7 at 30 and **+68.0 at 40**. Seed 0 was at +12.7 / +29.8 /
  +56.8 / +73.2.
- **Unlike seed 0, it has flattened.** It gained only +2.3 over generations 31–40.
- At generation 40, 11 of 16 world-episodes are above A0, with summed losses of −247.

That flattening matters for the final read. D7's seed-1 `plateau` clause needs the seed-1 run's
own last-10-generation improvement below F_tr (54.75), and `budget_limited` needs it at or above.
The read itself is 7.5's.

### Details

| generation | center | g_tr | F_tr (atlas / own 3·SE) | improvement over 10 gens | up / down | summed gains / losses |
|---|---|---|---|---|---|---|
| 10 | 1955.59 | +5.69 | 54.75 (54.75 / 54.71) | +5.69 | 8 / 8 | +503.0 / −412.0 |
| 20 | 1963.34 | +13.44 | 54.75 (54.75 / 41.05) | +7.75 | 10 / 6 | +641.0 / −426.0 |
| 30 | 2015.59 | +65.69 | 54.75 (54.75 / 48.83) | +52.25 | 12 / 4 | +1187.5 / −136.5 |
| **40** | **2017.94** | **+68.03** | **54.75** (54.75 / 47.53) | **+2.34** | **11 / 5** | **+1335.5 / −247.0** |

- **Verify (as 7.1):**
  - `contract_failures.csv` is header only, and generations 1–40 are present;
  - the pins are arm A1, opt seed 1, minibatch 24, prereg v2, the EXP-0058 bar, commit
    `7b16e55…`, with no resume;
  - final θ `726a2192a51b…` = `candidate.json` = `checkpoints/gen-0040.json`.
- **Cost:** pod `ggnnxno5w69shj`, secure 16 vCPU (cpu32 refused for capacity, no pod created),
  21:44:19 launch → terminated by 22:51:28 (≤ 67.2 min), **≈ 17.9 vCPU-h** billed.
  - Stop 37. Launch-fit 54.6 + 37 = 91.6 ≤ 125. Tranche A ≈ 72.5.
- **Files:** `artifacts/a1-s1/`, the run dir as pulled.

## final: the seed-combined read (task 7.5)

**Final reading: `plateau`** (seed 0 `plateau` + seed 1 `plateau`). Selected A1-s0. Labels
`herding` and `history_inert` on both seeds. `seed_disagreement` does not arise.

### Seed 1 on validation (A1-s1, 9 gating worlds × 8 episodes)

| comparison | A1-s1 | floor | (A1-s0, from val-seed0) |
|---|---|---|---|
| − A0 (g_val) | +45.3 | 30.0 | +68.7 |
| − RR | +2.5 | 32.8 | +26.0 |
| − MWW | −82.5 | 26.4 | −59.0 |
| − TR | +590.9 | 31.6 | +614.4 |
| − its shared-only ablation | +4.1 | 36.7 | +2.8 |

- **Mean:** 1,804.9. 1 death; 2 worlds lost to RR beyond floor_w; run-down share 0.73 against
  RR's 0.65.
- **Not `success`:** the references are not all beaten, there is a death, and the flags are
  worse than RR's.
- **Seed-1 clauses** (D7 seed combination): g_tr +68.0 > F_tr 54.75, so not `unresolved`.
  g_val 45.3 ≥ g_tr / 3, so not `kill`.
  - The run's own last-10-generation improvement is +2.3 < F_tr, so not `budget_limited`, and
    plateau (i) holds.
  - (ii) comes from seed 0; (iii) holds.
- **Label table:**

  | label | value | consequence |
  |---|---|---|
  | `history_inert` | yes, both seeds | recorded; must be cited before history becomes GNN edge attributes (D8, task 12.1) |
  | `herding` | yes, both seeds | under `plateau`: recorded and carried into the relational check's record |
  | `not_via_deconcentration` | n/a (no gain to explain) | — |

### Verification (task 7.5's verify line)
- `contract_failures.csv` is header only.
- 2,304 rows on 128 world-episodes, each with one digest pair, equal to val-seed0's digests.
- The 1,792 reused val-seed0 rows are identical in the combined `episodes.csv`; 512 rows are new
  (A1-s1 and its ablation).
- Every played world is in the validation split.
- `--mode recompute` gives **MATCH** (`plateau`). It was run on a copy whose `pins.reuse.path`
  points to the local val-seed0, because the recorded path is the pod's.

### Cost
- Pod `hluwwmpny5rpyh`, secure 16 vCPU, 22:53:00 launch → terminated by 22:58:45 (≤ 5.8 min),
  **≈ 1.5 vCPU-h** billed.
- Stop 8.2. Launch-fit 72.5 + 8.2 = 80.7 ≤ 125.
- **Tranche A billed in total ≈ 74.0 of 125.**
- **Pull location:** the read was pulled into `final/` rather than the EXP root, so the rsync
  could not overwrite this filled RESULT.md.

## Conclusions

- **Final reading: `plateau`.** Selected A1-s0. `herding` and `history_inert` are recorded on
  both seeds.
- **The learned per-order correction is real and generalises:** over A0, +68.7 / +45.3 on
  validation and about +70 in training, on both seeds.
- **It ends level with round robin and below MWW:** −59.0 / −82.5, beyond the floor. Step 2 does
  not produce a router that beats the simple rules.
- **Training:** A1 seed 0 g_tr +73.2 (EXP-0060a), A2-R2 +83.7 (EXP-0060b), A1 seed 1 +68.0
  (EXP-0060d), all > F_tr 54.75. A2-R2 is not better than A1 on validation (−14.1, inside the
  floor).
- **Next (registered):** ask the user to release tranche B1, the relational check (step-3 gate).
- **No DIAGNOSTICS entry is due:** that is required only on `unresolved` / `kill` / `other`.
- **Tranche A ≈ 74.0 of 125 billed.**
