# EXP-0068 — Noise floor for EXP-0058's `no_signal` (DIAGNOSTICS 25 checks 2 and 3)

**Status: complete, valid. Not an acceptance read.**
- **Check 3** (registered, one-sided): A1's minibatch median ratio is 1.76 > 1. With the probe's
  cross-minibatch disagreement, this reads `heterogeneous` **by the registered statistic**.
- **That reading is not robust to the reference estimate.** The registered reference is the
  spread of A0's minibatch-mean fitness over 5 replicates, which has 4 degrees of freedom.
  Estimated instead from the per-world-episode spreads (24 degrees of freedom), it is about twice
  as large. Against that reference the same one-sided rule reads inconclusive: A1 0.85, A2 0.94.
- **Check 2:** a re-keyed A0 alone is self-disqualified on 16 of 48 episodes (33%). Task 6.1b of `add-m4-orderer-redesign`.
- **Driver and commit:** `order_es --mode noise-floor` (task 6.1c) at commit `6f0eda62b9c5…`,
  verified clean, from a clean worktree.
- **Inputs:** EXP-0058's bar (normaliser `2d574ec8a3df…`) and its recorded probe files
  (`probe.json` `facfacfa8e75…`, `probe_fitness.csv` `379d5895f185…`, `probe_episodes.csv`
  `3c3e4df7895e…`), EXP-0012 `orderer_scorer.pt` (`d25388bfa33e…`).
- **Run:** 60 jobs (A0 at r0..r4 on the probe's 12 world-episodes), horizon 100,
  `entity_planner`. One fresh secure 8 vCPU pod, ≈ 0.23 vCPU-h billed.
- **Validity:** 0 contract failures; A0(r0) reproduces the bar's A0 row on 12 / 12; `--recompute`
  offline MATCHes.

## In plain English

**What was open.** EXP-0058's gradient probe found no usable direction: the ES gradient
estimates from two disjoint batches of 6 training world-episodes pointed in unrelated directions
(cosine −0.07). Task 6.1a ruled out the first explanation: switching off the disqualification
rule did not bring a direction back. Two explanations were left.
- **Flat at σ:** the small perturbations change outcomes no more than random draw noise does.
- **Heterogeneous:** the perturbations do change outcomes, but in a direction that depends on
  which worlds are in the batch, so two batches disagree.

**What this run measured.** It replays A0, the uncorrected policy, on the probe's 12
world-episodes with 4 fresh random-draw keys (r1–r4), besides the original key r0. That shows
how much a world-episode's outcome moves from draw noise alone. Two readings follow from it.
- **Check 3** compares each perturbation pair's fitness difference, f(+i) − f(−i), with that
  noise. The noise reference is √2 × A0's spread across keys, i.e. the noise of a difference
  between two independently keyed episodes.
  - The reading is one-sided and was registered before the data. A ratio above 1 means the pair
    difference is above draw noise; a ratio at or below 1 is inconclusive. (The two members of a
    pair share one key, so their difference is usually less noisy than this reference.)
  - **The deciding number** is the median ratio over A1's 24 minibatch-level pairs (12 pairs × 2
    minibatches), the level the probe's estimate uses. Everything else is reported only.
- **Check 2** applies the probe's relative disqualification rule to A0 against itself: A0 at a
  new key against A0 at the original key. It asks how often the rule fires from draw noise
  alone.

**What happened.**
1. **Check 3: above draw noise.**
   - **A1's minibatch median ratio is 1.76** (A2: 1.92). 75% of the 24 A1 pairs are above 1, and
     half are above 2.
   - The perturbations move a minibatch's mean fitness by a median of 0.019 of the training bar.
     Re-keying A0 moved the minibatch mean with an SD of 0.0084 / 0.0065 (minibatch 1 / 2), each
     from only 5 replicates.
   - Together with 6.1a's finding that the pairs' signs agree across the two minibatches no
     better than chance (5 / 12 for A1, correlation −0.13), the registered rule reads
     **heterogeneous**.
   - **But the reading rests on a noisy reference, and a better estimate of the same quantity
     reads inconclusive** (review round 1; "Sensitivity of check 3" below).
     - Each world-episode has its own draw key, so the minibatch mean's re-keying SD can also
       be computed from the six per-world-episode SDs: √(Σ SD_w²) / 6, with 24 degrees of
       freedom.
     - That gives 0.0180 / 0.0129, about twice the 4-df values.
     - Against it, A1's median ratio is **0.85** and A2's 0.94, with 46% of pairs above 1:
       inconclusive under the same one-sided rule.
     - So this run does not show that the perturbations have effects above draw noise, and it
       does not distinguish "heterogeneous" from "flat at σ".
   - Reported only: per world-episode the median ratio is 0.68 (A1) / 0.81 (A2), also below 1.
     The gap to the minibatch level's 1.76 is not because averaging reduces noise. Averaging six
     independent worlds shrinks the pair differences' noise by the same factor as the reference,
     and the six per-world differences add up barely more coherently than noise (below). The gap
     comes mostly from the low 4-df minibatch reference.
2. **Check 2: the rule fires on noise alone.**
   - A re-keyed A0 is "disqualified" against itself on **16 of 48 episodes (33%)**: 12 run-down
     flags and 5 starvation flags (one episode has both).
   - They fall on the 5 world-episodes where the probe's members drew run-down or starvation
     flags. The sixth world-episode with member flags, 12-0155, where the members died, shows
     none:

     | world-episode | re-keyed A0 flagged (of 4) | flags |
     |---|---|---|
     | 12-0005 | 4 | run-down |
     | 12-0139 | 3 | run-down |
     | 12-0000 | 3 | run-down |
     | 12-0014 | 3 | starvation (c8) |
     | 12-0044 | 3 | run-down / starvation (c9) |

     None of the other 7 world-episodes flagged A0. That includes 12-0155 (the members' flags
     there were deaths) and 12-0002.
   - So on those worlds the relative rule disqualifies by draw noise, not by anything the
     perturbation did. This confirms 6.1a's hypothesis for those worlds. 6.1a also showed the
     rule is not why the probe found no signal.

**What changes because of it.** DIAGNOSTICS 25 moves from "partially diagnosed" to
**heterogeneous by the registered statistic, not robust to the reference estimate**. The
registered reading stands as registered, because rules come before data. The better-estimated
reference reads inconclusive, so flat-at-σ is not excluded. Also on record: the relative rule fires on draw noise on 5 of the
probe's 12 world-episodes. As tasks.md 6.1b says, the supervisor takes the diagnosed branch to
the user before anything else runs. It is a strategy fork, for example the minibatch size or
sampling. The rule's noise firing is a second, separate input. 6.2 stays gated. No new run is
proposed here.

## Details

### Method
- **World-episodes:** the probe's two recorded minibatches from EXP-0058's `probe.json`.
  - Minibatch 1: 12-0014, 12-0004, 12-0005, 12-0003, 12-0051, atlas 12-0139.
  - Minibatch 2: 12-0000, atlas 12-0155, 12-0044, 12-0002, atlas 12-0308, 12-0033.
- **Jobs:** A0 through the ES monitor-job path, at replicate keys r = 0..4 (DrawKey(world,
  episode, r)), 60 jobs, 8 workers. Every row's demand and shock digests equal the bar's.
- **r0 reproduction:** A0(r0) equals the bar's A0 row (units, death, holding) on all 12
  (`reproduction.csv`).
- **Check 2:** `disqualifying(A0(r_k), bar row)` for k = 1..4, D6's relative rule unchanged.
- **Check 3:** ratio = |f(+i) − f(−i)| / (√2 × SD over r0..r4 of A0's fitness).
  - At the minibatch level, it uses the minibatch-mean fitness from EXP-0058's
    `probe_fitness.csv`.
  - At the world-episode level, it uses per-world-episode fitness from `probe_episodes.csv`.

### Check 3 summary (`check3.json`)

| level / arm | pairs | median ratio | share > 1 | share > 2 | median \|d\| |
|---|---|---|---|---|---|
| **minibatch / A1 (decides)** | 24 | **1.759** | 0.75 | 0.50 | 0.0187 |
| minibatch / A2 | 24 | 1.924 | 0.75 | 0.46 | 0.0195 |
| world-episode / A1 | 144 | 0.677 | 0.35 | 0.10 | 0.0306 |
| world-episode / A2 | 144 | 0.814 | 0.41 | 0.10 | 0.0261 |

- A0's re-keying SD of the minibatch-mean fitness is 0.0084 (minibatch 1) and 0.0065
  (minibatch 2).
- Per world-episode it is 0.008 (12-0000) to 0.086 (12-0004).

### Sensitivity of check 3 (review round 1; reported only, the registered reading stands)

- **Reference.** The minibatch mean is the mean of 6 world-episodes, each with its own draw
  key, so the episodes are independent by construction. Its re-keying SD is therefore
  √(Σ_w SD_w²) / 6, built from the per-world-episode SDs in `check3.json` with 6 × 4 = 24 df.
  - Minibatch 1: **0.0180** against the driver's 0.00845, which has 4 df.
  - Minibatch 2: **0.0129** against 0.00652.
  - A 4-df SD at about half its expected value is a lowish draw (χ²₄ ≈ 0.9, p ≈ 0.07), not an
    impossible one.
- **Ratios against that reference:**

  | arm | median ratio | share > 1 |
  |---|---|---|
  | A1 (deciding) | **0.846** | 0.46 |
  | A2 | 0.938 | 0.46 |

  Both are inconclusive under the registered one-sided rule.
- **Coherence within a minibatch.** Per pair, the measure is |mean of the 6 per-world d| ÷
  (rms(d) / √6).
  - Medians: A1 0.93, A2 0.86.
  - For pure noise (6 Gaussian values, simulated) the median is ≈ 0.76 and the mean ≈ 0.83.
  - So the per-world pair differences add up barely more coherently than noise.
- **Reproduction.** The reviewer's script is `/tmp/rev_m4_6_1b/scripts/recompute.py`; the
  implementer's recomputation from `check3.csv` / `check3.json` and EXP-0058's `probe.json`
  gives the same values.

### Check 2 (`check2.json`)

| | value |
|---|---|
| re-keyed A0 episodes | 48 (12 world-episodes × r1..r4) |
| self-disqualified | **16 (33.3%)** |
| flags | run-down 12, starvation 5 (one episode both) |
| world-episodes with any self-disqualification | 5: 12-0005 (4 / 4), 12-0139 (3 / 4), 12-0000 (3 / 4), 12-0014 (3 / 4), 12-0044 (3 / 4) |

### Cost

| pod | shape | life (UTC) | billed |
|---|---|---|---|
| `6ovyw6ph5b7ugc` | secure 8 vCPU | 17:58:17 → 18:00:00 (≈ 1.7 min) | **≈ 0.23 vCPU-h** |

- Setup was ≈ 1 min and the driver ran 17:59:35 → ≈ 17:59:50. The pull, stop and terminate were
  automatic (default detached flow); `runpodctl pod list` = [] afterwards.
- Estimate 1.5 (at the planning basis's 10-min setup), stop 3.1, launch-fit 12.0 + 3.1 = 15.1.
  Tranche A billed so far ≈ 12.2 of 100.
- No local step ran while the pod billed. The inputs were checked before the pod existed
  (SHA-256s; `noise_floor_problems` = []), and everything else happened after the pull.

### Files

| file | what |
|---|---|
| `spec.json` | pins (bar hashes, normaliser, orderer, the three probe-file SHA-256s, the world-episodes, K = 4, source tree), status |
| `noise_floor.csv` | the 60 A0 rows (gate columns + trace + ES columns) |
| `reproduction.csv` | A0(r0) against the bar's A0 row, 12 / 12 equal |
| `check2.json` | self-disqualification: rate, by flag, by world-episode, the flagged list |
| `check3.csv`, `check3.json` | per pair ratios at both levels; summaries; A0's SDs |
| `contract_failures.csv` | header only |

**Recompute:** `python -m tokenomics.train.order_es --mode noise-floor --recompute --bar-dir
<EXP-0058>/bar --probe-dir <EXP-0058> --probe-episodes <EXP-0058>/artifacts/probe_episodes.csv
--orderer-ckpt <EXP-0012>/artifacts/orderer_scorer.pt --out-dir <EXP-0068>` gives MATCH.

## Conclusions

- **Check 3 (registered, one-sided): `heterogeneous` by the registered statistic, not robust.**
  - A1's minibatch median ratio is 1.76 > 1. With 6.1a's cross-minibatch sign disagreement
    (5 / 12, r = −0.13), EXP-0058's `no_signal` reads heterogeneous under the registered rule.
  - Against the better-estimated 24-df reference, the same rule reads inconclusive (A1 0.85,
    A2 0.94). The per-world ratios (0.68 / 0.81) and the within-minibatch coherence (≈ noise)
    point the same way.
  - So whether the perturbations have effects above draw noise at all is open, and flat at σ is
    not excluded.
- **Check 2: the relative rule fires on draw noise** on 5 of the 12 world-episodes (33% of
  re-keyed A0 episodes). It is not the cause of `no_signal` (6.1a), but it would tie members
  there in any ES generation.
- **Next:** the supervisor takes both findings to the user (a strategy fork). Nothing is
  launched or proposed here; 6.2 stays gated.
  - The user's fork ("Bigger batches + fix rule", task 6.1d) was taken on the registered
    reading.
  - The sensitivity above, recorded in review round 1, bears on that fork. A minibatch-24
    re-probe (EXP-0069) tests the estimate's agreement directly, whichever branch holds.
