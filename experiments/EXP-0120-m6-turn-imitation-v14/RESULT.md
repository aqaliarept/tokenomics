# EXP-0120: turn-head imitation, v14 (learned set context; F40 start)

**Setup.**
- `m6_prereg_v14` (SHA `21b60013…`, re-frozen after the review, before data). Code c13bd0a.
- **Start:** chosen by EXP-0119's reading (`gain`) → F40:
  - candidate_it40 `ace0fabf…` (trunk and investment gates);
  - the EXP-0110 start's replenish `697d94c0…`.

  All pinned.
- **Heads:** 2 self-attention layers over each turn's items; bounded 10·tanh scores with a pre-tanh penalty.
- Pod wnes87bgzymub3 (secure 16 vCPU + 20 GB volume, EUR-IS-1), 14:05–14:22 UTC ≈ **4.7 vCPU-h** (stop 12, M6-A).
- Collection: 427,150 training turns and 62,854 held-out turns, in 108 s.
- Fit: 20 epochs, 686 s. Loss 0.317 → 0.126, still falling (−0.0009 over the last epoch).

## Reading: **`offline_fail`** (no closed loop)

| metric (clipped / bounded scores) | held-out | train | EXP-0118 held-out | gate |
|---|---|---|---|---|
| accept agreement | **0.962** | 0.965 | 0.887 | 0.99 |
| rank, accept turn | **0.986** | 0.9885 | 0.9655 | 0.99 |
| rank, fund turn | **1.000** | 1.000 | 1.000 | 0.99 |
| whole schedule | 0.905 | – | 0.801 | reported |

**Accept errors, split as in DIAGNOSTICS 38 (held-out):**

| label | bids | wrong |
|---|---|---|
| declined, no fit | 5,483 | 0 |
| declined, displaced | 25,977 | 2,424 (9.3%; EXP-0118 34%) |
| accepted | 97,278 | 2,508 (2.6%; EXP-0118 5.5%) |

**Other reported quantities:**
- Saturation: priority 0, fund 0.
- Tied learnable pairs: 0 / 0.
- Identical-bid pairs with different default bits: 518 of 11,954. Those cap accept agreement at ≈ 0.996.

## Conclusions
- **The set context works:** displaced-bid errors fell from 34% to 9.3%, and accept agreement rose from 0.887 to
  0.962. The ties are gone.
- **The remaining gap is underfitting:**
  - train ≈ held-out (0.965 vs 0.962; displaced 8.6% vs 9.3%);
  - the loss was still falling at epoch 20;
  - 200 turns overfit to 100% displaced / 99.4% accepted (local capacity check).

  At width 32, 2 layers and 20 epochs over 427k turns, the fit does not reach the auction's exact cumulative
  arithmetic. This is a fit-budget / capacity problem, not a missing input.
- **Next, a design decision:** a capacity / budget probe on the persisted held-out records (no rollouts): width
  64, 3 layers, more epochs, a learning-rate schedule, trained on one half and read on the other. Then v15 with
  the chosen setting.
