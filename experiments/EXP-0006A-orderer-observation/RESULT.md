# EXP-0006A — Orderer observation enrichment

**Date:** 2026-06-24 · **Driver:** `tokenomics.train.orderer_improve --enriched` · **Reproduce:** see [`spec.json`](spec.json)

> **Correction (P1 re-run, 2026-06-24).** The first run of this experiment used a **buggy
> `obs_price`** search candidate: `decide_ordering_obs` read the producer cut-off price at
> the *v1* observation offset, which under enriched v2 obs points into the wrong block — so
> the v2 teacher's candidate set was not held fixed vs the v1 (EXP-0005) teacher, breaking
> the clean A/B. Fixed via `EnvSpec.ordering_price_indices()` (version-aware); `obs_price`
> now emits identical actions in v1 and v2. **This document reports the clean re-run**
> (`m2_v3_obs_fixed.*`); the original teacher-degraded numbers are kept in the
> `buggy_ratio_env` column of [`summary.csv`](summary.csv) for provenance. The verdict is
> unchanged (REJECT strict) but the *mechanism* changed materially — see Conclusions.

## Question

EXP-0005 closed with a documented limit: the deterministic orderer **clone** ties/loses the
strict random-inclusive **envelope** on held-out seeds (esp. seed 9, 0.91), while the Stage-2b
audit proved the *same* search **teacher** beats the envelope on all four held-out seeds — so the
residual is a **clone gap**, not a teacher ceiling. EXP-0005 Stage 2c refuted "more data" as the
fix (doubling topologies regressed the held-out envelope). Two candidate causes remained:
**(a) observability** — the legacy v1 orderer obs is capacity-blind (ADR-0029), so the clone
cannot see the producer load/makeability the teacher routes by; **(b) network capacity**.

EXP-0006A tests **(a)** in isolation. Holding the entire EXP-0005 M2-v1 recipe fixed and changing
**only** the observation — legacy v1 → enriched v2 (the ADR-0051 feature catalog: producer
load/headroom, per-(producer,product) makeability/bottleneck, recent throughput/latency/demand
EMAs) — does the distilled **M2-v3-obs** beat the strict envelope on held-out seeds 6,7,8,9 with
zero death?

```text
Accept: M2-v3-obs beats the strict envelope on held-out seeds 6,7,8,9, with zero death.
Reject: the enriched-observation student still fails while the teacher clears the envelope.
```

## Method

A clean A/B over the **observation only**. Everything below matches EXP-0005 M2-v1 exactly except
`orderer_obs_version` (1 → 2):

- **Producer:** fixed at the EXP-0004 v4 clone (`m2_inputs/d3_rung_dagger_v4.pt`).
- **Search teacher:** directed per-cycle routing search, lookahead 12, K=12, candidates
  {`even_split`, `proportional`, `obs_price`} — the same labeler that cleared the envelope in
  EXP-0005 Stage 2b.
- **Distillation:** DAgger ×2 (round 0 on-oracle BC; round 1 relabel of the clone's visited
  states), masked-MSE over the decode-active orderer dims (ADR-0029).
- **Observation:** enriched v2 (ADR-0051). The domain `OrdererEconomyView` owns the authoritative
  facts; `env.encode_ordering_enriched` only normalizes + clips. Checkpoint carries the v2
  `config_fingerprint` (cannot be mis-scored against a v1 economy).
- **Train topologies:** scale seeds 0–5 × 3 episode seeds. **Held-out:** seeds 6,7,8,9 × 5
  episode seeds (unseen by label-gen — same split as EXP-0005 for continuity).
- **Bars:** M0 = `stock_aware`+`proportional`; envelope = best of `stock_aware`×{prop, even,
  random} + `random+random`. Primary metric: consumer completed units (mean ± 95% CI).
- **Compute:** Runpod CPU pod via `run_pod_training.py --cpu --python python3.12` (HARD RULE —
  real training off-box). Trainer/eval enriched-obs threading added + TDD'd locally first
  (`tests/test_orderer_improve_enriched.py`), local smoke green before launch.

## Result

(machine-readable: [`summary.csv`](summary.csv) / [`artifacts/m2_v3_obs_heldout_fixed.csv`](artifacts/);
checkpoint: [`artifacts/m2_v3_obs_fixed.pt`](artifacts/) (md5 `41f68672…`). Clean re-run on
Runpod CPU pod `9ytl7r42avgpmm` (8 vCPU), ~50 min. DAgger fit MSE: round0 **0.8886** (was 0.99
with the buggy teacher → the corrected candidate set is more self-consistent) → round1 1.2546 on
54k rows.)

| seed | M2-v3-obs cu✓ ±95%CI | M0 cu✓ | M2/M0 | envelope cu✓ (best) | **M2/env** | (buggy was) | M2-v1/env | dead | rej✗ |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 478.2 ± 12.0 | 423.8 | 1.13 | 483.4 (`sa+random`) | **0.99** ✗ | 1.01 | 0.99 | 0 | 30 |
| 7 | 484.0 ± 11.5 | 438.6 | 1.10 | 438.6 (`sa+proportional`) | **1.10** ✓ | 1.04 | 1.06 | 0 | 23 |
| 8 | 557.6 ± 16.2 | 403.4 | 1.38 | 531.4 (`sa+random`) | **1.05** ✓ | 1.08 | 1.02 | 0 | 54 |
| 9 | 482.2 ± 18.1 | 405.4 | 1.19 | 504.0 (`sa+random`) | **0.96** ✗ | 0.87 | 0.91 | 0 | 76 |

Anchors: `buggy` = the original teacher-degraded run (P1); M2-v1 = EXP-0005 legacy v1 obs (same
held-out / producer / recipe). Mean M2/env: **1.025** (buggy 0.999, M2-v1 0.995). Zero death all 4.

## Conclusions

**Verdict: REJECT (strict).** The pre-registered accept rule is *M2-v3-obs beats the strict
envelope on **all** held-out seeds 6,7,8,9 with zero death*. Death is zero on all four, but seeds
**6 (0.99)** and **9 (0.96)** fall just below the envelope. So observability enrichment alone, even
with a correct teacher, does not clear the bar.

1. **Observability is a real but partial lever.** v2 obs (capacity-blind v1 → load/makeability-aware)
   beats M0 on all four seeds (1.10–1.38) and lifts the mean envelope ratio to **1.025** (v1 M2-v1
   was 0.995). It clears the envelope on s7 (1.10) and s8 (1.05) but misses on s6 and s9.
2. **The "seed-9 cliff" was largely a teacher artifact.** With the buggy `obs_price` (P1) s9 read
   0.87 with a wide ±42 CI and 121 rejects — it looked like a pathological residual. The corrected
   teacher recovers most of it: **s9 → 0.96**, rejects 121→76, CI ±42→±18. So the original "s9 is
   special / unmoved by state" reading was overstated; the clean residual is a smaller, *more
   uniform* shortfall.
3. **The residual is routing diversity, not state.** Both failing seeds' envelope-best is
   `stock_aware+**random**` — the deterministic clone loses precisely where a *randomized* orderer is
   the envelope winner, and beats the envelope on the two seeds where it isn't (s7 `sa+proportional`,
   s8 it overcomes). One deterministic mean cannot reproduce that diversity regardless of inputs.
   Consistent with EXP-0005 Stage 2c (capacity, not data), this is the doc's "reject → prefer
   stochastic distillation before PPO" branch — now on a **clean A/B**, not a degraded teacher.

## Follow-up

- **EXP-0006B** is warranted on clean evidence (this re-run "improves but still fails" the strict
  rule), and the evidence steers it toward **stochastic distillation (M2-v3-stochastic)** over a
  merely wider deterministic net — **both** missing seeds (6, 9) have envelope-best `sa+random`, a
  direct signal that routing *diversity* is the missing ingredient. A topology-conditioned net is a
  secondary option. The stochastic path is already built + TDD'd (`orderer_improve --stochastic`,
  learned per-dim `log_std` + sampled deploy).
- Keep the **enriched v2 obs as the orderer contract** going forward: it strictly helped 3/4 seeds,
  zero death, and is the right input substrate for the stochastic head.
- The trainer/eval now drive v2 end-to-end (`orderer_improve --enriched`,
  `evaluate(enriched_orderer_obs=True)`), TDD'd in `tests/test_orderer_improve_enriched.py`.

Cross-link: `docs/adr/0051-orderer-observation-economy-metrics.md`, memory `exp0005-headroom`,
`generalization-overfit`, `exp0006a-observation-enrichment`.

## Follow-up

- If **accept**: the blocker was observability; the enriched obs becomes the new orderer contract
  for expert iteration (continue EXP-0006/0007/0008 on v2).
- If **reject with low teacher-action fidelity**: move to **EXP-0006B** (wider/deeper or stochastic
  distillation).
- If **reject despite high fidelity**: teacher labels may be multimodal → prefer stochastic
  distillation before PPO.

Cross-link: `docs/adr/0051-orderer-observation-economy-metrics.md`, memory `exp0005-headroom`,
`generalization-overfit`.
