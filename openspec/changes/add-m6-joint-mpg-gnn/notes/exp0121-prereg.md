# EXP-0121 pre-registration: capacity / fit-budget probe for the v14 turn heads

Frozen before any data (supervisor 2026-11-07, after EXP-0120's underfit). Driver `tokenomics/train/m6_turn_probe.py`.
No rollouts, never an acceptance read.

## Question
EXP-0120's v14 heads underfit: train ≈ held-out (accept 0.965 / 0.962, accept-turn rank 0.9885 / 0.986), with the
loss still falling at epoch 20. Which head size and fit budget reach the gate (0.99 accept bits and 0.99 accept-turn
rank)?

## Data
- EXP-0120's `heldout_records.npz`: 62,854 turns, F40 trunk embeddings, the default labels.
- Split into two contiguous halves (≈ 31.4k turns each). The records are in job order, so at most one
  world-episode straddles the halves.
- Each setting fits on the first half and is read on the second, using v13 / v14's `agreement` on bounded scores
  (excluded pairs, ties and accept-error split as v14).

## Grid
- Head width {32, 64} (the trunk stays 32) × set layers {2, 3} × epochs {30, 90}: 8 settings.
- Every setting: a cosine learning-rate schedule from 3e-3 to 0 over its epochs, batch 512, ≤ 400 pairs per turn,
  seed 0, bounded scores with the pre-tanh penalty. Everything else as v14.

## Winner
The first setting in (width, layers, epochs) ascending order whose probe-held-out accept agreement and accept-turn
rank both reach 0.99. Fund-turn rank is reported (1.000 in EXP-0118 / 0120).
- If none reaches 0.99: report before v15; the gate is not relaxed.

**Reported per setting:** seconds per epoch, seconds per 1k turn-epochs (the heads' per-turn fit cost for the 4.5
sizing), parameters, the loss curve, the train-half agreement.

## Then
v15 = v14 with the winning width, layers, epochs and the cosine schedule. Its offline gate and closed loop use a
**fresh held-out set** (new episodes of the monitor worlds, never played), because this probe tunes on EXP-0120's
held-out episodes.

**Sizing:** pod 8 vCPU with a persistent volume, EUR-IS-1, 8 workers (one per setting), ≈ 3 vCPU-h, **stop 5**,
M6-A.
