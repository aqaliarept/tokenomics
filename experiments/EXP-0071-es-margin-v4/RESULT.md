# EXP-0071: Noise margin re-measured on the MWW base (pre-registration v4)

**Status: complete, valid; step (a) of task 7.9 only.** Steps (c) / (d), the v4 retrain and its
read, were **not run** (user decision "Close step 2 now", 2026-11-04). Not an acceptance read.

## In plain English

**The question.** In EXP-0070 the learned correction on top of MWW stalled. About half of the
perturbed policies were "disqualified" in every ES generation, meaning they were penalised for
running producers down or starving consumers more than the unchanged policy A0 (= MWWk). The
safety margin in that rule had been measured on the old starting policy (TS, EXP-0068).

The suspicion was that the margin was simply mis-sized for MWW, so the rule fired on random
noise. This run re-measured it on MWWk itself.

**What was measured.** A0 = MWWk was replayed on all 128 training world-episodes of the EXP-0070
bar, at its original random-draw key and four fresh ones (640 episodes). That shows how much
run-down and consumer fill move from draw noise alone. The registered formula then gives the
margin (2 × √2 × that spread), and the rule is checked on A0 against itself.

**What happened.**
- **The margin on MWW is about the same size as before:** run-down 0.110 (old 0.096),
  starvation 0.102 (old 0.117).
- **It does its job on noise.** Re-keyed A0 is flagged against itself on 24 of 512 episodes
  (4.7%), within the registered ≤ 5% target; without a margin it would be 89. Pre-registration
  v4 was frozen with this margin.
- **It does not change the policies.** Re-scoring EXP-0070's 1,000 recorded member scores:

  | rule | member scores disqualified |
  |---|---|
  | no margin | 970 |
  | old margin (v3) | 497 |
  | **new margin (v4)** | **542** |

  The new margin disqualifies slightly more, not fewer. A0's own noise trips it about 5% of the
  time, while the perturbed policies trip it about 50% of the time.

**What it means.** The disqualifications were not noise. Policies perturbed away from MWW really
do run producers down or starve consumers more than MWW does. So retraining under v4 would
have met the same wall. The user closed M4 step 2 instead (HISTORY §14).

## Details
- **Driver:** `order_es --mode noise-floor --bar-dir <EXP-0070 bar> --base mww` (task 7.8), at
  commit `353acc4e0403…` from a clean worktree.
- **Inputs:** the EXP-0070 bar (`bar.csv` `83449d36…`, A0 rows `9378d7be…`, normaliser
  `0691beff…`) and EXP-0012 `orderer_scorer.pt` (`d25388bf…`).
- **Rows:** 640 (128 world-episodes × r0..r4). r0 reproduces the bar's A0 rows 128 / 128. 0
  contract failures. `--recompute` of `margins.json` gives MATCH.
- **Margin (`margins.json`):**
  - run-down σ 0.03880 (df 512), margin **0.10973**;
  - starvation σ 0.03592 (df 5,400), margin **0.10158**;
  - self-flags 89 → **24 / 512** (target ≤ 25, met);
  - leave-one-world-out (reported only): 24 flagged, refit margins run-down 0.1069–0.1102,
    starvation 0.1007–0.1022.
- **Pre-registration v4** `es_prereg_m4_v4.json` (`3f067e9a…`): v3 with this margin. The rows
  are staged as `train/registry/es_noise_floor_m4_v4.csv` (`45172d2c…`), and the SHA is pinned
  in `order_es_validate.PREREG_V4_SHA256` (commit 80c4420). It was never used for training.
- **Re-score:** offline `score_member` over EXP-0070 `artifacts/a1-s0/members.csv` (40
  generations × 25 members = 1,000 scores), with the EXP-0070 bar's A0 rows as references.
  - v1 rule: 970 disqualified (run-down 1,165 / starvation 2,165 flags);
  - v3 margin: 497 (412 / 295);
  - v4 margin: 542 (307 / 495).
- **Cost:** pod `36ck9abcfx33d8`, secure 8 vCPU, 10:20:48 → terminated by 10:24:14 (≤ 3.4 min),
  **≈ 0.45 vCPU-h** billed.
  - Stop 3.6. Launch-fit 88.3 + 3.6 = 91.9 ≤ 125.
  - Tranche A final ≈ **88.8 of 125**.
- **Files:** `noise_floor/` holds `spec.json`, `noise_floor.csv`, `reproduction.csv`,
  `margins.json`, a header-only `contract_failures.csv` and the driver's `RESULT.md`.

## Conclusions
- **The MWW-base margin is about the old size** (0.110 / 0.102 against 0.096 / 0.117). It keeps
  A0's noise flags at 4.7% (target ≤ 5%), and v4 is frozen and pinned.
- **The "mis-calibrated margin" explanation of EXP-0070 is not supported** (DIAGNOSTICS 28).
  Under v4, 542 / 1000 member scores are still disqualified, about 10 × A0's noise rate.
  Perturbing MWW's choices genuinely worsens run-down / starvation.
- **Steps (c) / (d) were not run, and M4 step 2 is closed** (user decision 2026-11-04).
