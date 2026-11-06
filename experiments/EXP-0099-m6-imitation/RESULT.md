# EXP-0099: M6 Stage I imitation (collect → fit → check)

add-m6-joint-mpg-gnn task 2.6. Code a66f132, pre-registration `train/registry/m6_prereg_v1.json`
(SHA `38db5c0c…`), pod eyy1zg07jwa0uz (secure 16 vCPU, EUR-IS-1), 09:49:36–10:08:24 ≈ **5.0 vCPU-h**.
Models SHA `6c5ec2f7…`.

## Question
Can the fully learned GNN actors (producer + orderer), trained by behaviour cloning on k5_rf5 + MWW
labels, reproduce the k5_rf5 + MWWk stack's consumer units before any RL? (Pre-registered: `faithful` if
g_imit ≥ −F_imit and no extra deaths, else `short` → stop, the user hears it before RL.)

## Result: **`short`**
| | GNN stack | k5_rf5 + MWWk |
|---|---|---|
| mean consumer units (16 monitor world-episodes × replicates 0/1, horizon 100) | **3,360.75** | **3,871.38** |
| deaths | 0 | 0 |

g_imit = −510.6 against F_imit = 158.0 (−13.2% of the base's units).

Collect: train 112 world-episodes (134,400 producer / 322,785 orderer decisions), monitor 16
(19,200 / 51,285). Fit: 20 epochs, Hogwild × 16, lr 3e-3, batch 64.

Fidelity (train / monitor):
| metric | train | monitor |
|---|---|---|
| producer exact command | 0.548 | 0.511 |
| producer gate accuracy | 0.959 | 0.963 |
| replenish MAE (units) | 0.83 | 0.92 |
| orderer KL (clipped at 1e-9) | 0.327 | 0.325 |
| orderer top-1 | 0.490 | 0.445 |

Fit curve: producer loss 0.42 → 0.16, steady. **Orderer loss diverged**: 1.8e6 after epoch 1, 1.6e18 after 2,
≈ 1.2e24 from epoch 7 on.

## Diagnosis of the divergence (local, offline, `artifacts/diag_orderer_scale.py`, monitor samples)
- **Orderer input features are unscaled.** Per-column maxima reach 5,966 (global col 1), 2,688 / 2,550 /
  1,008 (producer-row cols 7 / 5 / 0–1); the producer GNN's features are log-scaled (all ≤ ~16 in weight).
- **The orderer's weights blew up**: max |w| ≈ 500–526 across encoder, update and head layers (producer
  GNN: ≤ 16). Logits: median |max| 9e31, max 6e33.
- The loss is the soft cross-entropy −Σ y·log_softmax; with logits ~1e31 any slot whose target mass sits
  on a non-argmax producer costs ~1e24–1e26. Median slot loss stays 1.39; the mean is carried by those
  slots. The reported KL (clipped at 1e-9) hides it (≤ 20.7 per slot). DIAGNOSTICS 31.
- Not separated: how much Hogwild (16 unsynchronised Adam updates on shared weights) adds on top of the
  feature scale. The scale alone gives the weights a path to grow; the producer, with scaled features and
  the same Hogwild fit, converged.

**Not diagnosed: which actor causes the units shortfall.** The check plays both GNN actors together. The
orderer is degenerate (top-1 0.445), but the producer is imperfect too (exact command 0.51). The discriminating
check is the role swap: GNN producer + MWWk, and k5_rf5 + GNN orderer, on the same 16 monitor world-episodes.

## Conclusions
- Stage I is `short`; per the pre-registration, stop before RL (no MAPPO, no critic).
- Candidate fix (needs a new pre-registration, v2; v1 stays as frozen): log1p / standardised orderer
  features (as in the producer GNN) and gradient clipping; the role-swap check decides whether the producer
  imitation also needs work.
