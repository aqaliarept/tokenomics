# EXP-0094: M5 L3 + L4 on k5_rf5 under pre-registration v7 (θ_e fixed at 0)

**Question.** The v7 ladder (`986a8191…`; v6 with the expansion trigger θ_e fixed at 0, after EXP-0093's `kill`,
DIAGNOSTICS 30):
- **L3:** the pilot. Seed 0 to 20 generations; `feasible` if g_tr > F_tr on the 16 monitor world-episodes.
- **L4:** seed 0 to 40, then seed 1 from θ = 0 to 40.
- Training-set numbers, not an acceptance read.

**Setup.**
- Code 487dfee (HEAD, v7 pinned). `producer_es --mode train`.
- The L3 read ran on the pod (`producer_read.run_pilot`). Seed 0 continued to 40 in the same process only on
  `feasible`; seed 1's pod launched only after that.
- Pods: secure 16 vCPU, **EUR-IS-1**:
  - seed 0 on `fyglmetpdcu512`, 23:25–23:55 UTC;
  - seed 1 on `hhofuqj2skyppj`, 23:41–00:17 UTC.
- 0 contract failures. 40 checkpoints each, `frozen_dims` 18–22 pinned; θ_e = 0 exactly in every checkpoint.

## L3: **`feasible`** (g_tr +192.2 vs F_tr 155.9 at generation 20, seed 0). L4: complete

| monitor (16 world-episodes × 2 replicates; k5_rf5 3,871.4) | gen 10 | 20 | 30 | 40 |
|---|---|---|---|---|
| seed 0 g_tr | +121.8 | **+192.2** (F_tr 155.9) | +186.3 | +175.0 (F_tr 96.0) |
| seed 1 g_tr | +178.3 | +117.7 | +168.9 | +180.6 (F_tr 215.1) |
| seed 0 / 1 centre held units / cycle (base 282.3) | 305 / 315 | 323 / 317 | 325 / 330 | 335 / 340 |
| seed 0 / 1 centre investment rejection (base 0.099) | 0.117 / 0.113 | 0.122 / 0.126 | 0.126 / 0.126 | 0.128 / 0.130 |
| monitor deaths | 0 / 160 episodes for seed 0, 0 / 160 for seed 1 | | | |

- **Per-generation minibatch diagnostics (reported only):** held stock 290–365, investment rejection 0.09–0.15,
  and 0–1 disqualified members per generation, for both seeds.
- **θ at generation 40:** ‖θ_k‖ 0.75 / 0.79, ‖θ_b‖ 0.71 / 0.36, θ_e = 0.

**One note, not a registered reading:** the pilot rule recomputed at generation 40 (as verification) gives
seed 0 `feasible` (+175.0 vs 96.0) and seed 1 below its floor (+180.6 vs 215.1, a wide own 3·SE). L4 has no
reading of its own, and the registered L5 decides on both seeds, so L5 was licensed and launched.

## Conclusions
- **Without θ_e the training gain is about a third of v6's** (+175 to +192 vs +467 to +574), but the side effects
  are mild: held stock about 1.2×, investment rejection 0.13 vs 0.10 (v6: 1.8× and about 3×).
- **L5 (EXP-0095) is next.**
- **Spend:** ≈ 8.1 (seed 0: pilot + L4) + 9.5 (seed 1) = 17.6 vCPU-h billed. M5-P ≈ 46.4 / 80.
