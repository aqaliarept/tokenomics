# EXP-0107: M6 Stage I v4, step input and boundary weight on the upgrade gate (offline gate, then check)

add-m6-joint-mpg-gnn (supervisor decision 2026-11-06 after EXP-0106, DIAGNOSTICS 33). Pre-registration
`m6_prereg_v6.json` (SHA `1bd23f29…`). Code 4a8bc0a. EXP-0105's collect reused at its pinned SHAs. Pod o0ga8nfebym9nc
(secure 16 vCPU / 32 GB, EUR-IS-1), 15:34–16:44 UTC ≈ **18.7 vCPU-h** (stop 24).

## Question
With the margin's sign as a step input and a boundary weight on the upgrade gate (w = 1.38 at 42, 1.99 at 12, by
the pre-registered rule), does the imitated producer hold the sustainable ceiling (offline gate), and does the
stack then pass EXP-0105's three-clause check?

## Result: offline gate **`gate_pass`**; three-clause check **`short` on (a), (b) units, (c)**; the ceiling part of (b) passes
**Offline gate** (recorded k5_rf5 decisions; 42 held-out worlds 0000 / 0040 / 0076 / 0126):

| measure | v4 | v2 (EXP-0100) |
|---|---|---|
| upgrade firing at margin < 0, 12 monitor | **0.000** | 0.024 |
| upgrade firing at margin < 0, 42 gate | **0.000** | 0.233 |
| upgrade accuracy at margin ≥ 0, 12 | 0.958 | 0.921 |
| restore accuracy (12 / 42) | 0.941 / 0.943 | 0.825 / 0.876 |
| expand accuracy (12 / 42) | 0.991 / 0.993 | 0.917 / 0.979 |

**Closed-loop check** (with the v4 producer **and the v4 orderer**):

| clause | measured | needed |
|---|---|---|
| (a) 12 vs k5_rf5 + MWWk | g = **−136.6** (3,734.8 vs 3,871.4), floor 82.7; deaths 0 / 0 | ≥ −floor |
| (b) 42 past the ceiling | **0** vs 0 | ≤ the base's ✓ |
| (b) 42 units | **−1,774.1**, floor 568.4; deaths 0 / 0 | ≥ −floor |
| (c) vs the EXP-0100 stack | g = **−332.7** (3,734.8 vs 4,067.4), floor 131.6 | ≥ −floor |

42 units by world (GNN vs base):

| world | GNN | base |
|---|---|---|
| 0000 | 8,476 | 11,626 |
| 0013 | 48,044 | 50,175 |
| 0039 | 61,374 | 63,506 |
| 0043 | 7,752 | 10,809 |
| 0066 | 13,172 | 14,305 |
| 0076 | 9,190 | 11,759 |
| 0081 | 31,656 | 30,665 |
| 0126 | 10,266 | 11,278 |

## The orderer diverged late in the fit
- **Fit curve:** the orderer sat at its label-entropy floor (1.3196–1.3197) through epoch 16. At epoch 17 it rose to
  **5.3e9**, and to 1.5e11 at epoch 20. The producer loss jumped at the same epoch, 0.049 → 0.38.
- **Final orderer:**
  - KL 2.23 (train) / 1.67 (monitor), against ≈ 0 in EXP-0100 and EXP-0105;
  - top-1 0.52 / 0.55, against 0.94;
  - logit spread on the monitor slots: median 5.5e11, max 8.8e12 (all finite);
  - max |w| 9.7, against 1.1 in v3 (EXP-0105, same orderer samples and settings).
- **The producer metrics are at the same final weights and still good.** The gate passed with exact command
  0.475 (v3 0.432), gate accuracy 0.965, and replenish MAE 0.93 units on the monitor (v3 1.58).
- **So the check played a broken orderer.** The units shortfall cannot be attributed to the v4 producer: the
  check has no role-swap arm (v4 producer + MWWk). Not diagnosed.
- Why the orderer diverged at epoch 17 here and not in EXP-0100 / EXP-0105 is **undiagnosed**: DIAGNOSTICS 31's
  "features scaled but still diverging" branch.
  - **Observed:** one Adam and one global clip are shared by both actors, the learning rate is constant
    (3e-3), and the orderer had been at its loss floor for 15 epochs.
  - **Hypotheses only:** Adam's step on near-zero gradients at the floor; coupling through the shared global
    clip; the producer's changed gradients (sign column, weights) acting through the shared step.

## Conclusions
- **The DIAGNOSTICS 33 fix works offline.** The v4 producer never upgrades past the ceiling on recorded
  decisions at either size (0.000, against v2's 0.024 / 0.233). In closed loop at 42 it puts 0 producers past
  the ceiling, against 381 for v3. Its gate fidelity is better than v2's on every head.
- **The reading is still `short`,** with a confound: the orderer diverged during the same fit. Stop here, per the
  decision.
- **Discriminating checks before any fix:**
  - **(1)** the v4 producer with MWWk (and the v4 orderer with k5_rf5) on the check's sets: does the shortfall
    belong to the orderer?
  - **(2)** the divergence: the orderer's loss per step around epoch 17, and the same fit with separate
    optimisers and clip per actor, or the orderer frozen at its EXP-0105 weights.
