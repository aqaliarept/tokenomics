# EXP-0011A - Deterministic Producer Champion Audit

**Question.** Can the deterministic entity-factored producer planner replace the v4
producer in the current product-stream stochastic champion stack, without
per-topology regression, death, or strict-envelope regression?

**Verdict: REJECT producer replacement promotion.** The deterministic entity planner
is viable and often better, but it does not clear the no-regression gate against the
current champion producer. At 40 eval seeds it regresses held-out topology seed `6`
by `1.125` consumer units on the mean. Keep the v4 producer in the champion stack.

## Method

Checkpoint: `tokenomics/m2_inputs/m2_stream_stochastic.pt`

- Champion stack: v4 producer from checkpoint + stochastic product-stream orderer.
- Candidate stack: `PRODUCING_BASELINES["entity_planner"]` + the same stochastic
  product-stream orderer.
- Held-out topology seeds: `6,7,8,9`.
- Horizon: `100`.
- No training and no checkpoint overwrite.

The audit helper is `tokenomics.eval.producer_champion_audit`.

## Ten-Seed Envelope Smoke

This local smoke includes the strict random-inclusive envelope. It looked positive:
candidate clears champion mean and envelope mean on every topology, zero death.

| Seed | Champion CU | Entity CU | Delta | Envelope CU | Entity/env | Death |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 494.9 | 495.5 | +0.6 | 473.8 | 1.0458 | 0.0 |
| 7 | 475.4 | 492.5 | +17.1 | 437.7 | 1.1252 | 0.0 |
| 8 | 567.6 | 571.3 | +3.7 | 537.3 | 1.0633 | 0.0 |
| 9 | 523.6 | 529.4 | +5.8 | 504.4 | 1.0496 | 0.0 |

## Forty-Seed Producer Replacement Audit

This is the decisive producer-vs-champion gate. It does not rescore the envelope,
only whether the candidate producer can replace v4 under the same champion orderer.

| Seed | Champion CU | Champion CI95 | Entity CU | Entity CI95 | Delta | Ratio | Death |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 490.875 | 4.362 | 489.750 | 4.414 | -1.125 | 0.9977 | 0.0 |
| 7 | 471.675 | 4.796 | 490.025 | 4.021 | +18.350 | 1.0389 | 0.0 |
| 8 | 565.375 | 5.119 | 567.325 | 6.567 | +1.950 | 1.0034 | 0.0 |
| 9 | 524.075 | 6.540 | 528.300 | 6.271 | +4.225 | 1.0081 | 0.0 |

Summary: zero death and mean delta `+5.85`, but seed `6` violates the per-topology
no-regression gate. Promotion is therefore not recommended.

## Finding

The deterministic planner is a strong producer baseline and remains useful for
planner/search work, but it is not a strict producer replacement for v4. The current
champion stack stays:

```text
v4 producer + M2 product-stream stochastic orderer
```

The next useful work is not to overwrite the producer champion. If we continue joint
producer-orderer work, treat the deterministic planner as a feasible teacher/search
component, not as the promoted producer policy.

## Note On Compute

The 40-seed local audit took long enough that future 40-seed or stricter audits should
run in cloud execution, even when they are eval-only rather than training.
