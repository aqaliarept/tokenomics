# EXP-0124 pre-registration: the EXP-0121 probe with the order-aware accept pass

Frozen before any data (user decision 2026-11-07, order-aware network; review fixes in the same commit). Driver
`tokenomics/train/m6_turn_probe.py --ordered`. Never an acceptance read.

## Question
Does the gated recurrent pass over each turn's items, in the heads' own learned-priority order
(`notes/4.3-ordered.md`), reach the 4.3 gate where the set encoder alone stalled at 0.94–0.95 (EXP-0121 / 0122)?

## Data
EXP-0122's `heldout_records_linear.npz`: EXP-0120's 62,854 held-out turns (F40 trunk embeddings, default labels)
with v15's linear own-state fractions; verified equal to EXP-0120's records turn for turn. The same contiguous
halves as EXP-0121 / 0122: fit on the first, read on the second.

## Heads and fit
- **Heads:** `TurnHeads(set_layers ∈ {2, 3}, width ∈ {32, 64}, bounded, linear, ordered)`.
- **Fit:** EXP-0121's settings (cosine LR from 3e-3, batch 512, ≤ 400 pairs per turn, seed 0, the pre-tanh
  penalty), plus:
  - the ranking loss alone for the first 3 epochs;
  - declined bids ranked in the loss (default price order).
- **Order:** the accept pass always reads the student's own order.

## Grid and winner
- **Grid:** width {32, 64} × layers {2, 3} × epochs {30, 90}.
- **Winner:** the first setting in (width, layers, epochs) order whose probe-held-out accept agreement and
  accept-turn rank (claims + accepted bids, as registered) both reach 0.99.
- **Reported:** the slack-bin accept-error split, the cost per 1k turn-epochs, the train-half agreement.

## Then
- **A winner:** v15 = v14 + the linear fractions + the ordered pass + the winning setting + the fresh held-out set,
  run on F40 (4.3), then the closed-loop check.
- **No winner:** report back. Accept bits are not relabelled by student-order replay (that is a gate change, the
  user's).

**Sizing:** pod 8 vCPU with a persistent volume, EUR-IS-1, ≈ 2.5 vCPU-h, **stop 3**, M6-A.
