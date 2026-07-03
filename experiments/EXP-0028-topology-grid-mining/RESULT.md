# EXP-0028 — Stage 3 Random Hard-Case Mining over a TOPOLOGY GRID

Fixes the EXP-0025 blocker: that run mined a **single cell** (all 24 archived tasks were
`build_scale_config depth=3`, only the seed differed), so the archive had no topology
diversity and Stage 4 (EXP-0026/0027) trained a producer *from scratch on one narrow cell* —
which replaced the champion's broad coverage and regressed uneven-strength transfer. Here the
pool is a **grid of 9 in-claim topology cells**; the mined archive now spans **8 distinct
descriptor cells**.

## Setup

- **Grid (9 cells):** `depth=3` fixed (the claimed depth / wall axis — held constant so a gap
  is honestly an in-claim `policy_weakness`, never `distribution_mismatch`), varying only
  **width** `products_per_layer ∈ {3,4,5}` × **producer density** `producers_per_layer ∈
  {4,5,6}` → products 9/12/15, producers 12/15/18, all inside the stage-5 envelope
  (8–16 products, 10–20 producers, 3 consumers). 9 distinct `(depth,products,producers,caps)`
  descriptor cells.
- **Pool:** 30 seeds/cell (0–29) = **270 tasks**. Learner = portable champion stack
  (`producer_scorer_broad.pt` + `orderer_scorer.pt`); reference = `entity_planner+mpc_search`;
  eval-seeds 3 (paired); horizon 100; gates `min_gap=2.0`, `min_win_frac=0.6`; rank
  `α=1.0, β=0.5` (regret + novelty).
- **Infra:** fresh 16-vCPU CPU pod via `--create-pod` (auto-terminated after pull); 9 parallel
  shards (one per cell). Also the first live run on the fixed detached-launch path (EXP-0027
  ops bug) — client reached `monitor_remote` and streamed with no hang.

## Result — diverse archive (blocker fixed)

```
270 sampled over 9 grid cells; failure classes: {none: 221, policy_weakness: 49}
archived 26 policy_weakness tasks; pool topology cells=9; archived topology cells=8
```

- **26 archived** (was 24 in EXP-0025), now across **8 of 9 topology cells** (was **1**).
  Gaps **2.0–7.33u**, mean **3.67** (EXP-0025: 2.0–7.67, mean 3.24 — comparable magnitude,
  now spread across topologies). **12/26 are robust 3-of-3 paired wins** (EXP-0025: 8/24).
- **Even spread** across cells (archived count by `(width,producers)`): w3p4=4, w3p5=4, w3p6=1,
  w4p4=5, w4p5=3, w4p6=2, w5p4=6, w5p5=1 (only the most-slack cell w5p6 contributed none —
  out-competed in greedy novelty selection).
- **0 contract failures, 0 learner deaths** across all 270 rows — the grid is entirely
  in-claim and healthy; every gap is a real in-claim policy weakness.

## Finding — weakness concentrates in leaner producer density

Pre-gate `policy_weakness` yield per cell (of 30), by `(products, producers)`:

| products → | 9 | 12 | 15 |
|---|---|---|---|
| **producers 12** | 8 | 8 | 9 |
| **producers 15** | 5 | 7 | 4 |
| **producers 18** | 2 | 5 | 1 |

Yield is highest at **low producer density** (producers=12 → 8/8/9) and falls as producers
grow (producers=18 → 2/5/1), roughly independent of width. Interpretation: more producers =
more capacity slack = easier routing/restoration, so the champion matches the search oracle;
the mineable weakness lives where **capacity is tight relative to demand**. This is a topology
*axis* the single-cell EXP-0025 could not see, and it is exactly the kind of coverage Stage 4
needs to avoid the uneven-strength (capacity-asymmetry) regression.

## Conclusions

**Blocker fixed; ready for Stage 4 v2.** The archive is now diverse (8 cells, 26 tasks,
capacity-pressure axis represented) instead of one depth-3 cell. No promotion — mining only
curates tasks; the learner is unchanged.

Next (item 3): **Stage 4 v2 = augment, not replace.** Distill the candidate producer/orderer
from the champion's *original broad* train set **plus** this prioritized mined replay (weighted
by regret), rather than mined-only — the EXP-0027 conclusion. The lean-producer-density cells
here should give the producer the capacity-tightness coverage whose absence caused the
uneven-strength regression.

**Kept honest:** depth stayed at the claimed 3, so nothing here is mislabeled as a policy
weakness when it is really out-of-claim. Deeper chains (depth≥4) remain out of this grid (wall /
out-of-claim) and are not mined as weaknesses.
