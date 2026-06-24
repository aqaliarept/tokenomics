# EXP-0006B — Stochastic orderer distillation

**Date:** 2026-06-24 · **Driver:** `tokenomics.train.orderer_improve --enriched --stochastic` · **Reproduce:** see [`spec.json`](spec.json)

## Question

EXP-0006A (clean re-run) established that enriching the orderer observation (v1 → v2) is only a
**partial** lever: the deterministic clone **M2-v3-obs** beats `stock_aware+proportional` (M0) on all
four held-out seeds but still **misses the strict random-inclusive envelope on s6 (0.99) and s9
(0.96)** — and *both* of those seeds' envelope-best is `stock_aware+`**`random`**. The residual was
diagnosed as **routing diversity, not state**: a single deterministic mean cannot reproduce the
per-cycle variety that a randomized orderer wins with. EXP-0005 Stage 2c had already refuted "more
data," and Stage 2b proved the search *teacher* clears the envelope — so the gap is the
**distillation form**, not the teacher.

EXP-0006B tests the form directly. Holding the entire EXP-0006A recipe fixed (v4 producer, enriched
v2 obs, L12/K12 search teacher, DAgger ×2, train 0–5, held-out 6,7,8,9), change **only** how the
orderer is distilled and deployed: learn a per-dim `log_std` by masked Gaussian NLL and **deploy by
sampling** instead of collapsing to `clip(mean)`.

```text
Accept (primary): M2-v3-stochastic envelope ratio > 1.00 on ALL held-out seeds 6,7,8,9, zero death.
Hard-cell:        s6 and s9 (the sa+random misses) must improve over M2-v3-obs (0.99, 0.96).
Secondary:        mean envelope ratio >= 1.10 (if feasible).
```

## Method

A clean A/B over the **distillation form only**, vs EXP-0006A M2-v3-obs:

- **Loss:** masked **Gaussian NLL** fits mean + per-dim `log_std` over the decode-active orderer
  dims (ADR-0029); masked dims keep init. (Deterministic path was masked-MSE over mean only.)
- **Deploy:** sample `a ~ Normal(mean, exp(log_std))`, clip to ±bound. Per-episode RNG is drawn
  from `env.action_space(agent).np_random` (seeded per episode by `run_episode`, like the `random`
  baseline) → **order-independent per seed** (the P2 fix; verified pre-cloud that seed 9 reads
  identically in `[9]` and `[6,7,8,9]`).
- **No orderer log_std clamp on save** (the producer stays greedy-clamped; only the orderer head
  is stochastic).
- **Eval:** `--eval-seeds 10` per held-out topology (sampling adds variance → more seeds for tighter
  CIs). Same bars as EXP-0005/0006A: M0 = `stock_aware+proportional`; envelope = best of
  `stock_aware`×{prop, even, random} + `random+random`. Metric: consumer completed units, mean ± 95% CI.
- **Compute:** Runpod CPU pod `3joly6h2z0hboa` (8 vCPU) via `run_pod_training.py --cpu --python
  python3.12`, run under the new **detached client-server** mode (survives disconnects). Pre-cloud
  smoke green (v2 fingerprint, learned log_std, P2 order-independence) before launch.

## Result

(machine-readable: [`summary.csv`](summary.csv) / [`artifacts/m2_v3_obs_stochastic_heldout.csv`](artifacts/);
checkpoint [`artifacts/m2_v3_obs_stochastic.pt`](artifacts/) md5 `7be79ccbd90e`, version 6, v2
fingerprint `548120bd…`, **64 learned `log_std` dims** σ∈[0.14, 0.60]. DAgger fit MSE round0 1.288 →
round1 1.680 on 54k rows.)

| seed | M2-stoch cu✓ ±95%CI | M0 cu✓ | M2/M0 | envelope cu✓ (best) | **M2/env** | det-obs/env | M2-v1/env | dead |
|---|---|---|---|---|---|---|---|---|
| 6 | 491.0 ± 9.7 | 421.6 | 1.17 | 478.7 (`sa+random`) | **1.026** ✓ | 0.99 ✗ | 0.99 | 0 |
| 7 | 475.2 ± 6.8 | 437.7 | 1.09 | 437.7 (`sa+proportional`) | **1.086** ✓ | 1.10 ✓ | 1.06 | 0 |
| 8 | 566.5 ± 9.9 | 396.9 | 1.43 | 535.9 (`sa+random`) | **1.057** ✓ | 1.05 ✓ | 1.02 | 0 |
| 9 | 513.9 ± 13.4 | 399.3 | 1.29 | 502.8 (`sa+random`) | **1.022** ✓ | 0.96 ✗ | 0.91 | 0 |

Mean M2/env: **1.048** (det-obs 1.025, M2-v1 0.995). Zero death all 4. M2 beats M0 by 9–43%.

### Confidence audit — 40 eval-seeds (`artifacts/m2_v3_obs_stochastic_audit40.csv`)

The recorded run used `--eval-seeds 10`; s9's win there was **mean-only** (lower 95% CI ≈500.5 just
under the envelope point 502.8). Re-scoring the *same* checkpoint with **40 eval-seeds** (CI ∝ 1/√N →
halved) to settle it — pure eval, `--eval-only`, no retrain:

| seed | M2 cu✓ ±95%CI | envelope | M2/env | **lower CI bound** | clears at lower bound | dead |
|---|---|---|---|---|---|---|
| 6 | 488.5 ± 4.1 | 473.5 | 1.032 | 484.4 | ✓ (>473.5) | 0 |
| 7 | 473.7 ± 3.4 | 436.4 | 1.085 | 470.3 | ✓ | 0 |
| 8 | 561.6 ± 5.2 | 533.2 | 1.053 | 556.4 | ✓ | 0 |
| 9 | 507.9 ± 6.6 | 498.3 | 1.019 | **501.3** | ✓ (>498.3) | 0 |

**The s9 caveat is resolved: s9 now clears the envelope even at its lower 95% CI bound (501.3 >
498.3).** All four held-out seeds beat the strict envelope at the lower CI bound, zero death. Mean
ratio **1.047** (unchanged → secondary 1.10 still not met, as expected — this audit tightens the
*primary* gate, it does not move the mean). The ACCEPT verdict strengthens from
mean-with-marginal-CI to **clears-at-lower-CI on all four**.

## Conclusions

**Verdict: ACCEPT (strict primary gate).** The stochastic orderer clears the strict random-inclusive
envelope on **all four** held-out seeds (1.022–1.086) with **zero death** — the first policy in the
EXP-0001→0006 arc to beat the strict envelope everywhere. The hard-cell condition is met: the two
deterministic misses **lift over the line** — **s6 0.99 → 1.026** and **s9 0.96 → 1.022** — and s9
now beats *both* prior champions (M2-v1 0.91, M2-v3-obs 0.96), exactly as pre-registered.

1. **Routing diversity was the missing form, as predicted.** The gains land precisely on the
   `sa+random`-dominated seeds (s6, s9) the deterministic clone could not match, while s8 (also
   `sa+random`) widens (1.05 → 1.057). On the one seed whose envelope-best is *not* random
   (s7, `sa+proportional`) the stochastic head trades down slightly (1.10 → 1.086) but still clears
   — sampling is a small cost where determinism was already optimal, a large win where it wasn't.
   This confirms EXP-0006A's diagnosis (diversity, not state) on a clean A/B.
2. **It is a learned stochastic policy, not noise.** Only the 64 decode-active dims learned a
   non-trivial `log_std` (σ up to 0.60); the 116 masked dims stayed at init. The fit MSE rising
   round0→round1 (1.29 → 1.68) is the NLL trading point-fit for calibrated spread — expected when
   the objective is distributional, not pointwise.
3. **Secondary not met, and that's fine.** Mean envelope ratio is **1.048**, below the
   "if feasible" 1.10. The primary strict gate (beat the envelope *everywhere*, zero death) is the
   acceptance criterion and it passes; the 1.10 mean was a stretch target. The 10-seed run left one
   honesty caveat — s9's win was mean-only (lower 95% CI ≈500.5 just under the envelope point 502.8).
   **The 40-eval-seed confidence audit (above) resolves it: s9 now clears at its lower CI bound
   (501.3 > 498.3), as do all four seeds.** The mean stays 1.047 — tightening the CI does not move
   the mean, so the 1.10 stretch remains the only unmet (optional) target.

**This closes the orderer-improvement question opened in EXP-0005: a learned agent beats the strict
random-inclusive envelope.** Promote **M2-v3-stochastic** as the new orderer champion (v4 producer +
enriched v2 obs + sampled orderer).

## Follow-up

- **Promote** M2-v3-stochastic as the default orderer for downstream expert iteration; keep the
  enriched v2 obs contract and the stochastic head.
- **Tighten s9** if the mean-only margin matters: more eval seeds (shrink CI) or a light
  elite-set / mixture distillation (the doc's next rung) to push the mean ratio toward 1.10.
- Producer side is unchanged (v4); the lone known producer residual (held-out seed-9 p=0.24,
  diminishing-returns coverage — EXP-0004) is orthogonal to this orderer result.

Cross-link: `docs/adr/0051-orderer-observation-economy-metrics.md`, memory `exp0006a-observation-enrichment`,
`exp0005-headroom`, `generalization-overfit`.
