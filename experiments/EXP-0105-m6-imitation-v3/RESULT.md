# EXP-0105: M6 Stage I v3, imitation with the ceiling-margin feature, collected at 12 and 42

add-m6-joint-mpg-gnn task 2.11 (supervisor decision 2026-11-06, DIAGNOSTICS 32). Pre-registration
`m6_prereg_v5.json` (SHA `4fdf45ef…`; the pre-data `63d85a94…` superseded by the bounded denominator). Collect:
commit 260f2dc (attempt 1). Fit and check: commit 134c267 (attempt 3), reusing attempt 1's collect at its pinned
SHA (`artifacts/collect_sha256.txt`). Secure 16 vCPU / 32 GB, EUR-IS-1. Models SHA in `artifacts/fit/fit.json`.

## Attempts (≈ 16.2 vCPU-h in total)
1. **l5cqpvr1zpowxt, 1.3 vCPU-h:** the collect succeeded (150 world-episodes; 294,000 producer decisions, 159,600
   of them at 42; 451,673 orderer decisions). The fit was then killed out of memory: 16 workers each held the
   samples (Hogwild pickled shards; sync workers inherited the full lists through fork). Fixed in 24dc65c with
   per-worker shard files.
2. **pmwe2rrpy4mmsj, 2.1 vCPU-h:** `Too many open files` (torch's default `file_descriptor` sharing strategy) hung
   the mode probe. Stopped after 5 min without progress. Fixed in 134c267 (`file_system` strategy).
3. **53yvp4v7yj77ub, 12.8 vCPU-h:** probe → sync (1.26×), 20 epochs, 2,017 s; then the check.

## Question
Does the scale-free ceiling-margin feature, with training data at both 12 and 42 producers, make the imitated
producer respect the sustainable-upgrade ceiling at 42 without losing the 12-producer result?

## Result: **`short` on all three clauses**
| clause | measured | needed |
|---|---|---|
| (a) 12, v3 vs k5_rf5 + MWWk on the monitor set | g = **−274.2** (3,597.2 vs 3,871.4), floor 113.3; deaths 0 / 0 | g ≥ −floor |
| (b) 42, producers past the ceiling (h200, 8 worlds × 4 × r0/r1) | **381** vs k5_rf5's 0 | ≤ the base's |
| (b) 42, units vs k5_rf5 + MWWk | **−2,972.8**, floor 321.6; deaths 0 / 0 | ≥ −floor |
| (c) v3 vs the EXP-0100 stack on the monitor set | g = **−470.3** (3,597.2 vs 4,067.4), floor 131.3 | ≥ −floor |

Per 42 world (the v3 GNN stack, 8 episodes each):
| world | past the ceiling | units, GNN | units, k5_rf5 |
|---|---|---|---|
| 0000 | 14 | 10,575 | 11,626 |
| 0013 | 71 | 47,235 | 50,175 |
| 0039 | 0 | 61,523 | 63,506 |
| 0043 | 32 | 10,420 | 10,809 |
| 0066 | 5 | 12,509 | 14,305 |
| 0076 | 108 | 7,593 | 11,759 |
| 0081 | 102 | 22,084 | 30,665 |
| 0126 | 49 | 8,401 | 11,278 |

In EXP-0104, the v2 GNN had 8 producers past the ceiling in one 42-0076 episode. Here v3 averages 13.5 per
episode there.

## Fidelity (train / monitor)
- **Orderer:** KL ≈ 0, top-1 0.936 / 0.945. Unchanged from v2.
- **Producer:** exact command 0.467 / 0.432 (v2 monitor 0.436); gate accuracy 0.933 / 0.930 (v2 0.889);
  replenish MAE 0.90 / **1.58 units** (v2 0.99).
- **The upgrade head over-fires more than v2's:** on the monitor it predicts 0.184 against a label rate of 0.122
  (v2: 0.153). On the training samples, both sizes pooled: 0.172 against 0.110.
- Restore predicted 0.422 against 0.373; expand 0.080 against 0.027.
- Fit curve: producer loss 0.279 → 0.059 (epoch 13) → 0.066 (epoch 20). The orderer is at its label-entropy
  floor, about 1.32, from epoch 1.

## Conclusions
- **The v3 imitation is worse than v2 on every clause.** The fix did not work: the imitated producer passes the
  ceiling more often at 42, and loses units at 12 against both the base and the EXP-0100 stack.
- **Cause: not diagnosed.** The fidelity metrics locate it in the producer's upgrade, expand and replenish heads.
  The orderer is fine.
- **Candidate explanations, none tested** (root-cause rule: the discriminating check comes first):
  - **(i)** the gate head reads a mean-pool over capacity rows, so the upgrade row's margin is diluted by the
    other rows, including the −1 of rows without a restoration;
  - **(ii)** the 42 data changes the label mix the shared heads fit: the upgrade label rate is 0.110 pooled,
    against 0.122 at 12;
  - **(iii)** the extra feature changes the normaliser and the fit's optimum.
- **Discriminating check (offline, on the collected samples):** on decisions whose upgrade-type row has margin
  < 0 (k5_rf5's label there is always 0), does the v3 upgrade logit respond to the margin's sign?
  - Split by size, and compare against the same decisions under the v2 model.
  - The samples are 1.7 GB pickled: too big for the local box, so a small CPU pod or a chunked loader.
- **Stage 0 waits;** the EXP-0100 stack stays the imitated result.
- **Overlap noted:** the 42 check worlds include collect worlds (different seeds) and validation 42-0066.
