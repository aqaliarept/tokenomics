# EXP-0122 pre-registration: the EXP-0121 probe with v15's linear own-state fractions

Frozen before any data (supervisor 2026-11-07: v15 = option (a)). Driver `tokenomics/train/m6_turn_probe.py
--recollect`. Never an acceptance read.

## Question
EXP-0121 found the turn heads miss the auction's boundary (slack within half a bid's capacity: 15–23% wrong). The
information is complete, but the quantities enter log-scaled. Do linear, scale-free own-state fractions let a
setting reach the gate?

## Features (v15 option (a); appended after the log features, which stay)
- **Fund rows, +1:** target / the producer's budget (1.0 when the target is unbounded).
- **Bid and claim rows, +3:**
  - capacity / the producer's free capacity for new work;
  - capacity / its current capacity;
  - price / the turn's highest item price (bids and claims).

  The capacity fractions are taken at the item's binding type (max over its types; every current world has one
  type per producer). Clipped at 10.
- **No cumulative or "after pricier bids" quantity.** The heads still learn the running sum.

## Data
- EXP-0120's 32 held-out jobs (monitor world-episodes × r0 / r1, F40 with the start's replenish, default turns),
  replayed deterministically with the fractions appended.
- **Validity:** every existing column and the trunk embeddings must equal EXP-0120's records turn for turn (the
  same 62,854 turns, so the same halves as EXP-0121). Otherwise the run stops, invalid.

## Grid, winner, reported
Identical to EXP-0121:
- **Grid:** width {32, 64} × layers {2, 3} × epochs {30, 90}; cosine LR from 3e-3; batch 512; ≤ 400 pairs; seed 0.
- **Fit / read:** fit on the first half, read on the second.
- **Winner:** the first setting in (width, layers, epochs) order reaching 0.99 on both accept agreement and
  accept-turn rank.
- **Also reported per setting:**
  - the accept errors split by slack bin (DIAGNOSTICS 38: < −2, [−2, −0.5), [−0.5, 0), [0, 0.5), [0.5, 2), ≥ 2
    own capacities);
  - the cost per 1k turn-epochs.

## Then
- **A winner:** v15 = v14 + the fractions + the winning setting + the cosine schedule + the fresh held-out set,
  run on F40.
- **No winner:** report; no option (b).

**Sizing:** pod 8 vCPU with a persistent volume, EUR-IS-1, ≈ 2.5 vCPU-h, **stop 3**, M6-A.
