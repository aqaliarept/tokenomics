# EXP-0100: M6 Stage I imitation, v2 (normalised features, sync fit, role swap)

add-m6-joint-mpg-gnn task 2.6 rerun after EXP-0099 `short` (DIAGNOSTICS 31). Code c9f45da, pre-registration
`train/registry/m6_prereg_v2.json` (SHA `2460cd07…`; v1 untouched). Pod ap4hmm4hf1tm8e (secure 16 vCPU,
EUR-IS-1), ≈ 11:15–11:42 UTC ≈ **7.2 vCPU-h** (stop 12, extended by the supervisor to 12:25 ≈ 19; not needed).
Samples: EXP-0099's collect, reused at their pinned SHA-256s. Models SHA `5acf8b82…`.

## Question
After the fix (in-model median / IQR normaliser over signed-log1p counts, gradient clipping 1.0, a
deterministic sync fit), does the behaviour-cloned GNN stack reach k5_rf5 + MWWk? And which role carries
any gap or gain (reported-only swap arms on the same keys)?

## Result: **`faithful`**
g_imit = **+196.1** units against F_imit = 90.5; deaths 0 in every arm.

| arm (producer + orderer) | mean units (16 world-episodes × r0/r1) | − base | floor | deaths |
|---|---|---|---|---|
| GNN + GNN (the imitated stack) | **4,067.4** | **+196.1** | 90.5 | 0 |
| k5_rf5 + MWWk (base) | 3,871.4 | — | — | 0 |
| GNN producer + MWWk (swap) | 4,078.4 | **+207.0** | 92.1 | 0 |
| k5_rf5 + GNN orderer (swap) | 3,877.1 | +5.7 | 54.8 | 0 |

**Role attribution:** the gain is the GNN producer's. With MWWk it is +207 (13 of 16 world-episodes ahead,
median +122; the largest +1,394 on ws-atlas-v1-12-0121). The GNN orderer is neutral, +5.7 within its
floor. The stack is +196 (13 / 16 ahead, median +128.5).

## Fit
- **Mode probe:** one epoch each, Hogwild 60.2 s, sync 57.6 s (0.96×) → **sync** (rule: sync if ≤ 1.5×).
- **Curve:** both losses finite every epoch (`artifacts/fit/fit.json`).
  - Producer loss: 0.272, 0.127, 0.089, … 0.051 at epoch 20, noisy from epoch 10.
  - Orderer loss: 1.084 after epoch 1, then flat at 1.072 for epochs 2–20. This is the soft-label
    entropy floor, not a stall. On the monitor samples, CE 0.9096 equals the labels' entropy 0.9096; a
    uniform guess costs 1.154. The orderer learned within one epoch (`artifacts/diag_orderer_floor.py`).
  - Orderer weights max |w| ≤ 0.91 (EXP-0099: ≈ 500).
- Fit 1,229 s. Normaliser and per-epoch times are in `fit.json`; the mode and probe are in `fit/spec.json`.

## Fidelity (train / monitor)
| metric | train | monitor | EXP-0099 monitor |
|---|---|---|---|
| orderer KL | 9.1e-7 | 8.2e-7 | 0.325 |
| orderer top-1 | 0.944 | 0.944 | 0.445 |
| producer exact command | 0.478 | 0.436 | 0.511 |
| producer gate accuracy | 0.900 | 0.889 | 0.963 |
| replenish MAE (units) | 0.86 | 0.99 | 0.92 |

Per producer head, monitor (label rate / predicted rate / accuracy):
| head | v2 | EXP-0099 |
|---|---|---|
| restore | 0.373 / **0.548** / 0.825 | 0.373 / 0.358 / 0.943 |
| upgrade | 0.122 / 0.153 / 0.925 | 0.122 / 0.101 / 0.958 |
| expand | 0.027 / **0.111** / 0.917 | 0.027 / 0.022 / 0.989 |

Replenish rows, monitor (exact share, MAE units):
- label 0: 0.855, 0.41;
- label interior: 0.128, 2.48;
- label at bound: 0.732, 0.64.

Commands, monitor:
- exact 0.436;
- toggles only wrong 0.098;
- replenish only wrong 0.268;
- both wrong 0.198.

## Conclusions
- **DIAGNOSTICS 31 is fixed, by evidence.** On the same samples and architecture, v2's orderer copies MWW
  almost exactly: KL ≈ 1e-6, top-1 0.944.
  - Not separated: the share of the feature scale vs Hogwild. Both changed; the probe chose sync.
- **The imitated stack clears the pre-registered check, and with a margin.** It beats the base by +196.
  The GNN producer carries the gain.
- **The GNN producer is not a faithful copy at the decision level.** It fires restore about 1.5× and expand
  about 4× as often as k5_rf5. That over-investment is the likely source of the units gain. It is not
  diagnosed: the check records no run-down (r_end), fund or holding stats, so a horizon-100 run-down cannot
  be excluded.
- Next per the plan is RL (L1…). It waits for the supervisor ("don't build MAPPO until I say"). A
  discriminating check of the producer's extra investment (run-down / fund share against k5_rf5) is the
  open question before it.
