# EXP-0037 — Value-Routed Orderer

**Date:** 2026-07-15 · **Driver:** TBD · **Reproduce:** see [`spec.json`](spec.json)

## Question

EXP-0036 showed that a learned orderer cannot beat a fixed mid-blend when the action is
"choose one predefined blend policy." Did we test the wrong action space?

This experiment asks whether a trained orderer can learn the **routing weights directly**:

```text
(order, eligible producer, global context) -> marginal value score
softmax over eligible producers -> routing weights
```

The floor is fixed `blend_0.5`, not the old champion.

## Starting Hypothesis

The EXP-0036 blend vocabulary is too narrow. `even_split`, `blend_0.75`, `blend_0.5`,
`blend_0.25`, and `proportional` are all capacity-shaped routers. They can only change how
strongly the orderer follows current throughput; they cannot learn that a lower-capacity
producer is strategically better because of inventory, inbound stock, TTL, queue pressure,
recipe position, or downstream scarcity.

## Proposed Policy Shape

Start conservative:

```text
base_logits = blend_0.5(order, producers)
residual_logits = model(order, producer, context)
final_logits = base_logits + clip(residual_logits, -r, r)
route = softmax(final_logits over eligible producers)
```

This keeps `blend_0.5` as the zero-cost floor but lets the model override capacity when
value evidence is strong.

## First Diagnostic

Before training a deployable policy, build an offline representability probe:

1. Collect states from the fixed `blend_0.5` trajectory on the EXP-0036 s29/s77/s45 tasks.
2. For each order, score one-step or short-rollout producer-weight perturbations around
   `blend_0.5`, not just the five predefined blend policies.
3. Train a pairwise/value model that predicts which eligible producer should receive more
   marginal weight.
4. Evaluate by rollout with the residual router.

## Metrics

```text
learned_cu
blend_0.5_cu
best_constant_blend_cu
search_cu
death rate
delta_vs_blend_0.5
```

## Stop Rule

Stop if the residual router does not beat fixed `blend_0.5` on at least 2 of the 3 gate
tasks by deployed rollout units. Do not trust table error alone.

## Status

Scaffold only. No code or run yet.

## Conclusions

Pending.

