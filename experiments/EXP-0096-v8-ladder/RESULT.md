# EXP-0096: M5 L3 + L4 on k5_rf5 under pre-registration v8 (run-down term in the fitness)

**Question.** The v8 ladder (`9ce8e3bd…`; v7 plus λ (ρ_member − ρ_base) in the per-episode fitness):
- ρ = r_end / r_start, with the base on the same world-episode;
- λ = σ_u / σ_ρ = 1.398, from EXP-0090's training re-keys only;
- θ_e fixed at 0.

The steps: L3 (seed 0 to 20, the pilot rule), then L4 (seed 0 to 40, seed 1 0 → 40). Training-set numbers, not an
acceptance read.

**Setup.**
- Code 69b438d (HEAD, v8 pinned). As EXP-0094: the L3 read on the pod, seed 0 continuing only on `feasible`, and
  seed 1's pod launched after.
- Pods: secure 16 vCPU, **EUR-IS-1**:
  - seed 0 on `2mnkag1dvyv70b`, 06:36–07:04 UTC;
  - seed 1 on `pk83zzeyacr8c6`, 06:51–07:20 UTC.
- 0 contract failures. 40 checkpoints each; `rundown_lambda` and `frozen_dims` pinned.

## L3: **`feasible`** (g_tr +135.8 vs F_tr 117.2 at generation 20). L4: complete

| monitor (k5_rf5 3,871.4) | gen 10 | 20 | 30 | 40 |
|---|---|---|---|---|
| seed 0 g_tr | +167.6 | **+135.8** | +201.8 | +190.6 |
| seed 1 g_tr | +148.9 | +119.1 | +115.9 | +114.3 |
| seed 0 / 1 held units / cycle (base 282.3) | 309 / 304 | 319 / 311 | 334 / 323 | 331 / 326 |
| seed 0 / 1 investment rejection (base 0.099) | 0.114 / 0.115 | 0.126 / 0.120 | 0.119 / 0.119 | 0.132 / 0.129 |
| seed 0 / 1 mean ρ − base (monitor) | −0.005 / −0.004 | −0.002 / −0.001 | −0.001 / +0.001 | +0.000 / −0.006 |
| seed 0 / 1 run-down episodes of 32 (base 25) | 22 / 24 | 23 / 22 | 21 / 25 | 20 / 23 |

- Monitor deaths: 0 for both seeds at every monitored generation.
- **For comparison, v7's ρ gaps on the same monitor** (EXP-0094) were within ±0.011 too. Sixteen world-episodes do
  not resolve a gap of this size; L5 decides.

## Conclusions
- **L3 `feasible`, L4 complete.** The units gain on the monitor is similar to v7's for seed 0 (+191 vs +175) and
  smaller for seed 1 (+114 vs +181). Side effects are mild: stock about 1.16×, investment rejection 0.13.
- **L5 (EXP-0097) follows.**
- **Spend:** ≈ 7.6 (seed 0: pilot + L4) + 7.9 (seed 1) = 15.5 vCPU-h billed (stops 10 / 11). M5-P ≈ 63.4 / 80.
