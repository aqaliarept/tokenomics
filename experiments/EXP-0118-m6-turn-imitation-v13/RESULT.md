# EXP-0118: turn-head imitation, v13 (task 4.3; method check on EXP-0100's trunk)

**Setup.**
- `m6_prereg_v13` (SHA `97474c22…`, frozen before data). Code 8d140eb (`train/m6_turn_imitation.py`).
- Pod 80tp1npy7zrvam (secure 16 vCPU + 20 GB volume, EUR-IS-1), 11:40–11:49 UTC ≈ **2.3 vCPU-h** (stop 13, tranche
  M6-A).
- Collection: 419,535 training turns (188,294 fund, 231,241 accept) and 61,731 held-out turns, in 95 s.
- Fit: 20 epochs, 297 s; final loss 0.343.
- **Pull path incident (my error):** the pull went into EXP-0116's `artifacts/`, because the launcher script was
  copied with a stale destination. The files were moved here, and EXP-0116's `reading.json` was restored from
  git (927e432).
- EXP-0117 (v12, stopped during its fit, ≈ 3.6 vCPU-h, nothing pulled) used the same launcher. It would have hit
  the same path.

## Reading: **`offline_fail`** (no closed loop)
Held-out turns, clipped scores. Excluded pairs: fund replenishment vs replenishment 43,374 (47%); accept-turn
identical 17,420 (4.8%).

| metric | value | gate |
|---|---|---|
| accept agreement | **0.887** (131,286 bits) | 0.99 |
| rank, accept turn | **0.9655** (348,725 pairs) | 0.99 |
| rank, fund turn | **1.000** (48,663 pairs) | 0.99 |
| whole schedule | 0.801 of turns | reported |
| saturation | fund 0.0, priority 0.070, accept logit 0.330 | reported |
| identical bid pairs, bits differ | 607 of 12,929 | reported |

## Diagnosis (local, from `heldout_records.npz` + `models.pt`; 5,000 accept turns, 18,572 bids)
**Accept bits, wrong by the label's cause.** "Fits alone" is the bid feature: the bid fits the producer's free
capacity by itself.

| label | bids | wrong |
|---|---|---|
| declined, does not fit alone (`no_fit`) | 870 | **0** |
| declined, fits alone but higher-priced bids took the capacity (`displaced`) | 3,893 | **1,323 (34%)** |
| accepted | 13,809 | 762 (5.5%) |

- Every accept error is a bid that fits alone. Whether it is accepted depends on the other bids of the same turn.
- The turn heads score each bid from its own row, its linked capacity / input rows and the producer's pooled view.
  Nothing carries the other bids, so the auction's cumulative rule cannot be represented, at any fit quality.

**Priority order, by pair kind:**

| pairs | n | right (unclipped) | right (clipped) | ties at ±10 |
|---|---|---|---|---|
| bid–bid | 22,192 | 97.3% | 95.6% | 398 |
| claim–claim | 9,025 | 98.1% | 92.9% | 480 |
| claim–bid | 19,231 | 99.9% | 99.9% | 0 |

- Saturation (7% of priorities at the bound) costs 1.7 points on bid–bid and 5.3 on claim–claim; `score_l2` = 1e-3
  is too weak.
- A further 2–3 points are wrong even unclipped. The defaults are price-descending within kind, and the price is a
  feature, so this part is learnable (fit / capacity, not structure).

## Conclusions
- **Structural (DIAGNOSTICS 38):** a per-item head with no set context cannot imitate a cumulative allocation
  (greedy fill by price). The fund order is learned perfectly; its rule is per-item.
- **Fix candidates for v14, a design decision:**
  - **Set context for bids:** either own-state features such as "capacity demanded by the higher-ranked bids of
    this turn, per type", or a set encoder over the turn's bids and claims.
  - **A bounded score:** a stronger penalty, or B·tanh(z / B) with a tie-aware metric.
- **4.3 is not passed.** The user's F40 decision re-fits the heads on F40's trunk anyway, so v14 would carry both
  changes.
