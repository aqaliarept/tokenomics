# EXP-0067 — uTWF router: does the paper's water filling beat `round_robin`?

**Status: complete, valid. Not an acceptance read. Reading `utwf_not_better`: uTWF ties
`round_robin` at 12 producers. Recorded; step 2 unchanged.** `python -m tokenomics.eval.decoder_check
--mode arm-compare --arm utwf`, run commit `a21788333eb2…` (task 5.6 of `add-m4-orderer-redesign`;
arm and reading pre-registered in task 5.5). Inputs: atlas v2 = EXP-0052 `reclassified/atlas_v2.json`
(content hash `220886a7cf12…`), EXP-0012 `orderer_scorer.pt` (`d25388bfa33e…`, pinned only; no arm
plays it), reference EXP-0057 `episodes.csv` (`69656f9e2034…`). 213 episode rows = 71
world-episodes (43 at 12 producers over 23 worlds, 28 at 42 over 14) × 3 (uTWF × 2 replicates +
one RR replay), horizon 100, `entity_planner` producer, routing tracer on. 0 contract failures;
71 / 71 RR replays equal EXP-0057's rows. Fresh 16 vCPU secure pod (`lioey2txcstc77`, created and
terminated), exit 0 ([Details](#details)).

## In plain English

**The problem this addresses.** Every ordering agent that routes a product decides at the same
moment, from the same picture of the producers as the cycle starts, and none of them sees what
the others choose (ADR-0058). If every agent sends its order to the producer that looks best,
they all send it to the same one, which then cannot take them all. This is *herding*. The
trained orderer herds badly: its concurrent top share is 0.98, the share of a product's
concurrent orders whose lead receiver is the single most-used producer (EXP-0057). Drawing the
producer at random in proportion to its spare capacity (MWWk, `mww_keyed`) spreads the orders
(0.72–0.74), but each agent still aims at the whole spare capacity, as if it were alone.

**What uTWF does.** Distributed load balancing has a known answer for many dispatchers acting on
the same stale snapshot: *Tidal Water Filling* (Goren, Vargaftik and Moses, "Distributed
Dispatching in the Parallel Server Model", DISC 2020).
- Each dispatcher randomises so that the dispatchers' expected work, taken together, fills the
  servers' spare room evenly, like water poured into a container shaped by the queues.
- The *unsplittable* variant, uTWF (Definition 2, Eq. (29)), sends all of a dispatcher's jobs of
  a round to one server. It needs only local information:
  - the dispatcher's own job count a;
  - the number of dispatchers M (it assumes the others received as many jobs);
  - the queue lengths.
- Here, the arm `utwf`:
  - each ordering agent uses its own order quantity as a, and the product's configured number of
    ordering agents (streams) as M;
  - the producers' spare capacity (`makeable_qty_now`) plays the queues: more spare means a
    shorter queue;
  - it draws one producer with the paper's probabilities and sends its orders whole there.
- It reads no other agent's orders and no total demand. It uses EXP-0057's random draws, so it
  is paired with MWWk. The deviations from the paper are listed in
  `openspec/changes/add-m4-orderer-redesign/notes/5.5.md`.

**The question.** On EXP-0057's 23 loaded 12-producer worlds, does uTWF beat both `round_robin`
(RR, the strongest simple rule there) and MWWk by more than the noise? RR, MWWk and TS
(`trained_sampled_whole`) are EXP-0057's recorded rows. RR is replayed once per world-episode,
only to check that this run reproduces EXP-0057.

**Terms.**
- **ρ (tightness)**: how heavily a world's capacity is loaded, i.e. consumer demand against the
  producers' starting throughput. A **loaded** world has ρ > 0.28; the decoder check's (and so
  this run's) worlds are all loaded.
- **Floor** of a comparison on a world: the larger of the atlas noise floor and 3 × the
  standard error of this run's own paired per-episode differences. The own SE counts only when
  there are ≥ 2 paired observations. A difference counts only above its floor.
- **Mean rule**: the mean difference over the 23 worlds must exceed the mean of the per-world
  floors.

**The pre-registered reading (task 5.5, fixed before the run).** `utwf_strongest` needs all
three of:
- uTWF − RR passes the mean rule;
- uTWF − MWWk passes the mean rule;
- uTWF loses to RR by more than the floor on at most 3 of the 23 worlds.

Otherwise the reading is `utwf_not_better`. Only the 12-producer worlds decide; the 42-producer
worlds are reported.

**What happened.**

1. **12 producers (deciding): a tie with `round_robin`.**
   - Mean consumer units: uTWF 2,094.9, RR 2,095.2, MWWk 2,036.4, TS 1,900.3.
   - **uTWF − RR = −0.3 units against a mean floor of 166.5.** It is beyond the floor above RR
     on 2 worlds (12-0060, 12-0176) and below it on 3 (12-0111, 12-0254, 12-0352).
   - **uTWF − MWWk = +58.5 against a mean floor of 202.0.** It is above MWWk beyond the floor on
     5 worlds and below it on none.
   - It loses to RR beyond the floor on exactly 3 of 23 worlds, the cap, so that clause alone
     passes.
   - Both mean clauses fail: **`utwf_not_better`**.
   - Against TS (reported only), uTWF is +194.6, above a mean floor of 180.9.
2. **Where the levelling helps and where it hurts.**
   - uTWF's 5 clear wins over MWWk:
     - three dense worlds at ρ 0.6–4.5: 12-0408 +538 (ρ 1.1), 12-0352 +222 (ρ 4.5), 12-0230 +146
       (ρ 0.6);
     - two four-maker (`sparse_f4`) worlds: 12-0392 +670 (ρ 1.0, the median of the 23) and
       12-0176 +113 (ρ 0.3, the least loaded of the 23).

     They are not one kind of world, and this run tests no explanation of why the levelling
     helps on these.
   - Its 3 losses to RR beyond the floor are two two-maker (`sparse_f2`) worlds, 12-0111 −248
     and 12-0254 −410, and the most loaded world, 12-0352 (ρ 4.5) −207. On these three worlds
     RR's fixed rotation does better than every random draw measured (uTWF, MWWk, TS).
   - On most of the other worlds uTWF and MWWk are within a few tens of units of each other.
3. **Herding columns** (12 producers, calm / shocked):
   - concurrent top share: uTWF 0.70 / 0.69, MWWk 0.72 / 0.74, RR 0.62 / 0.63;
   - cycle-to-cycle share swing: uTWF 0.60 / 0.51, MWWk 0.59 / 0.52, RR 0.72 / 0.64.

   So uTWF de-concentrates concurrent orders only slightly more than MWWk, and less than RR's
   rotation. Its routing pattern is close to MWWk's, which matches its units.
4. **42 producers (reported only, 14 worlds): the largest gain over RR on these worlds, still
   below the floor.**
   - Means: uTWF 5,948.9, RR 5,606.2, MWWk 5,924.8, TS 4,597.5.
   - **uTWF − RR = +342.7 against a mean floor of 483.1.** It is beyond the floor on 5 worlds,
     mostly f14 (14 makers per product): 42-0126 +1,428, 42-0031 +1,139, 42-0119 +891, 42-0076
     +644, 42-0093 +423. It is below RR beyond the floor on no world; the largest shortfall,
     42-0081 −916, sits just inside that world's floor (1,089).
   - This is the largest gain over RR of any arm measured on these 14 worlds: EXP-0057's MWWk is
     +318.6 and MWW +316.2. But it is only 24 units above MWWk (uTWF − MWWk +24.1, mean floor
     475.8), and the 42-producer table never decides.
   - Against TS: +1,351.4.
5. **Validity.**
   - 0 contract failures, 0 missing rows.
   - Every RR replay equals its EXP-0057 row on every compared cell.
   - Every row's demand and shock digests equal EXP-0057's.
   - No deaths in any arm.

**What changes because of it.** As pre-registered, **`utwf_not_better` is recorded and step 2
is unchanged.** Step 2's frozen base stays the EXP-0012 scores, and uTWF does not go into roster
v3. This closes the water-filling branch: the paper-faithful, local-information uTWF ties RR at
12 producers and does not clear the floor at 42.

The earlier, non-local variant that read the product's total demand (WF) was replaced by the
user's "paper-faithful only" decision before any data. It was never run.

The run tests no explanation for the remaining gap between uTWF and a clear win. One candidate,
recorded as a hypothesis only (not diagnosed): the configured stream count M counts streams that
are idle in a cycle, so the assumed total load M·a overstates the real load. This would push the
draw toward even spreading.

## Details

### Method

- **Sample.** EXP-0057's: atlas v2 train split, ρ > 0.28, v2 class not collapse.
  - 23 worlds at 12 producers and 14 at 42.
  - The confirm's episode seeds: 2 per world, 1 on 12-0005 / 0155 / 0176.
  - 71 world-episodes.
- **Arms played.** Both use `entity_planner` and the routing tracer.
  - uTWF (`routing_diagnostics.utwf`), R = 2, at EXP-0057's draw keys DrawKey(world, episode,
    replicate). The keyed uniform is MWWk's, at the agent's first slot of the product.
  - RR replay (`round_robin`).
  - 213 jobs, longest first, 16 workers.
- **Arms read from EXP-0057.** RR, MWWk (R = 2), TS (R = 2).
- **Floors.**
  - Per world and comparison: floor_w = max(atlas `noise_floor`, own 3·SE). The own SE counts
    only with ≥ 2 paired observations: uTWF − RR pairs each uTWF replicate against RR's row, and
    uTWF − MWWk / TS pair replicate by replicate.
  - The mean rule compares mean_w(diff_w) with mean_w(floor_w), as fixed by the supervisor
    before any data.
  - The pooled-SE alternative is reported only.
- **Validity checks** (exit 2 on any): contract failures; missing rows; digests across arms and
  against EXP-0057; RR neutrality against EXP-0057 (`neutrality.csv`, check
  `exp0057_neutrality`).

### Means and floors (`compare.csv` mean rows; `reading.json`)

| size | worlds | uTWF | RR | MWWk | TS | uTWF − RR (mean floor) | worlds beyond / below | uTWF − MWWk (mean floor) | beyond / below | losses to RR |
|---|---|---|---|---|---|---|---|---|---|---|
| **12 (decides)** | 23 | 2,094.9 | 2,095.2 | 2,036.4 | 1,900.3 | **−0.3 (166.5)** | 2 / 3 | **+58.5 (202.0)** | 5 / 0 | 3 (cap 3) |
| 42 (reported) | 14 | 5,948.9 | 5,606.2 | 5,924.8 | 4,597.5 | +342.7 (483.1) | 5 / 0 | +24.1 (475.8) | 0 / 0 | 0 (cap 2) |

- Pooled-SE alternative floor (reported only): uTWF − RR 146.5 at 12 and 455.6 at 42; uTWF −
  MWWk 146.5 / 362.2. It fails under the alternative too.
- uTWF − TS (reported): +194.6 (floor 180.9) at 12 and +1,351.4 (584.0) at 42.

### Per world, 12 producers (`compare.csv`)

| world | cell | ρ | uTWF | RR | MWWk | TS | uTWF − RR (floor) | uTWF − MWWk (floor) |
|---|---|---|---|---|---|---|---|---|
| 12-0005 | dense | 0.5 | 910.0 | 834.0 | 929.0 | 935.0 | +76.0 (93.4) | −19.0 (105.0) |
| 12-0060 | f4 | 0.3 | 2609.0 | 2434.0 | 2600.5 | 2443.0 | **+175.0 (145.0)** | +8.5 (145.0) |
| 12-0094 | f4 | 0.3 | 762.2 | 735.5 | 724.2 | 739.8 | +26.8 (116.2) | +38.0 (116.2) |
| 12-0111 | f2 | 0.5 | 2968.8 | 3217.0 | 2989.0 | 2877.5 | **−248.2 (159.5)** | −20.2 (229.2) |
| 12-0121 | f4 | 2.2 | 4286.5 | 4320.5 | 4407.0 | 3564.2 | −34.0 (380.6) | −120.5 (731.1) |
| 12-0139 | f4 | 0.5 | 700.5 | 680.5 | 696.8 | 697.0 | +20.0 (116.2) | +3.8 (116.2) |
| 12-0155 | f2 | 0.8 | 670.0 | 710.0 | 717.5 | 652.5 | −40.0 (218.6) | −47.5 (262.5) |
| 12-0176 | f4 | 0.3 | 2013.0 | 1814.0 | 1900.0 | 1899.5 | **+199.0 (86.7)** | **+113.0 (86.7)** |
| 12-0230 | dense | 0.6 | 1009.2 | 977.5 | 863.5 | 828.2 | +31.8 (133.7) | **+145.8 (133.7)** |
| 12-0254 | f2 | 2.3 | 3285.2 | 3695.5 | 3528.5 | 3115.0 | **−410.2 (191.5)** | −243.2 (323.5) |
| 12-0302 | dense | 1.1 | 3465.8 | 3347.5 | 3482.0 | 3116.5 | +118.2 (212.8) | −16.2 (149.1) |
| 12-0308 | f2 | 1.1 | 2693.8 | 2746.5 | 2741.8 | 2637.8 | −52.8 (159.5) | −48.0 (186.8) |
| 12-0321 | f2 | 1.5 | 627.8 | 716.5 | 598.2 | 656.0 | −88.8 (193.4) | +29.5 (193.4) |
| 12-0340 | dense | 0.3 | 1796.8 | 1695.5 | 1782.0 | 1740.0 | +101.2 (133.7) | +14.8 (133.7) |
| 12-0344 | f2 | 0.6 | 3213.0 | 3240.0 | 3153.2 | 2977.2 | −27.0 (273.8) | +59.8 (249.9) |
| 12-0352 | dense | 4.5 | 1403.8 | 1610.5 | 1181.5 | 1100.2 | **−206.8 (149.1)** | **+222.2 (149.1)** |
| 12-0361 | dense | 0.5 | 1040.5 | 951.5 | 1039.2 | 1053.5 | +89.0 (149.1) | +1.2 (149.1) |
| 12-0365 | dense | 1.2 | 1346.2 | 1226.5 | 1353.0 | 1289.0 | +119.8 (133.7) | −6.8 (133.7) |
| 12-0391 | f2 | 2.4 | 2458.2 | 2495.5 | 2457.2 | 2294.2 | −37.2 (159.5) | +1.0 (159.5) |
| 12-0392 | f4 | 1.0 | 4151.8 | 4138.5 | 3482.2 | 3320.2 | +13.2 (131.4) | **+669.5 (199.0)** |
| 12-0408 | dense | 1.1 | 4226.5 | 4229.5 | 3688.2 | 3341.8 | −3.0 (149.1) | **+538.2 (351.8)** |
| 12-0436 | f2 | 1.7 | 1219.0 | 1126.5 | 1260.0 | 1175.5 | +92.5 (193.4) | −41.0 (193.4) |
| 12-0483 | dense | 1.8 | 1325.0 | 1247.5 | 1262.2 | 1253.2 | +77.5 (149.1) | +62.8 (149.1) |

Bold = beyond the floor. The 42-producer per-world rows are in `compare.csv`.

### Concentration and herding (`routing_metrics.csv`; MWWk from EXP-0057's)

| size, regime | concurrent top share, uTWF / MWWk / RR | share swing (consumer), uTWF / MWWk / RR | routed-share dispersion, uTWF / MWWk / RR |
|---|---|---|---|
| 12 calm | 0.696 / 0.720 / 0.621 | 0.603 / 0.589 / 0.719 | 0.171 / 0.147 / 0.166 |
| 12 shocked | 0.687 / 0.744 / 0.631 | 0.514 / 0.523 / 0.639 | 0.160 / 0.140 / 0.147 |
| 42 calm | 0.576 / 0.565 / 0.433 | 0.680 / 0.712 / 0.721 | 0.282 / 0.291 / 0.325 |
| 42 shocked | 0.689 / 0.637 / 0.533 | 0.739 / 0.704 / 0.803 | 0.307 / 0.334 / 0.332 |

### Files

| file | what |
|---|---|
| `spec.json` | pins, the arm table, the sample, provenance (run commit `a21788333eb2…`, 16 workers) |
| `reading.json` | the 12-producer reading, the reported 42 table, the rules, neutrality counts |
| `compare.csv` | per world and the mean rows: uTWF, RR, RR replayed, MWWk, TS, each comparison's diff, atlas floor, own 3·SE, pairs, floor |
| `neutrality.csv` | the 71 RR replays against EXP-0057 (all equal) |
| `routing_metrics.csv` | tracer scalars per size × regime for uTWF and RR |
| `contract_failures.csv` | header only (0 failures) |
| `artifacts/episodes.csv` (git-lfs) | the 213 rows (gate columns + `trace_*` + `arm`), SHA-256 `8aa7d38a15df…`; moved there from the out-dir root, as EXP-0057 / 0066 |

- **Cost.** One fresh 16 vCPU secure pod run by the supervisor. Billed ≈ 2.0 vCPU-h
  (supervisor's budget log: 16 vCPU × ≈ 7–8 min of pod life; the 213 jobs ran 12:43:18–12:45:32
  UTC, the rest is pod setup, install, pull and stop). Tranche A billed so far ≈ 5.1
  (EXP-0057 ≈ 1.41, EXP-0066 ≈ 1.7, EXP-0067 ≈ 2.0). The task 5.5 estimate (cpu8)
  was 2.2, stop 4.3.
- **Reproduce.** The pod command is in `openspec/changes/add-m4-orderer-redesign/sizing.md`
  §EXP-0067 (run here on cpu16 with `--workers 16`). The router is defined in notes/5.5.md.

## Conclusions

- **Pre-registered reading: `utwf_not_better`. The water-filling branch is closed.**
  - At 12 producers the paper-faithful uTWF ties `round_robin`: −0.3 units against a mean floor
    of 166.5.
  - It sits above MWWk without clearing the floor: +58.5 against 202.0.
  - Its clear wins over MWWk are on five worlds of mixed kinds (three dense, two `sparse_f4`), and
    its losses to RR beyond the floor are on two two-maker worlds and the most loaded one.
- **42 producers (reported only).** uTWF's +342.7 over RR is the largest gain over RR of any arm
  measured on these 14 worlds. It is only 24 units above MWWk and below the mean floor of 483.1,
  so it grants nothing.
- **Next: none.** As pre-registered, step 2 is unchanged: the EXP-0012 scores stay the frozen
  base, and uTWF does not enter roster v3. No follow-up strategy is proposed. The idle-stream
  inflation of M is recorded as an untested hypothesis, not a diagnosed cause.
