# EXP-0037 Handover — Value-Routed Orderer

## Why This Exists

EXP-0036 closed the trained-orderer path **only for blend-policy selection**. It did not
test a model that directly chooses producer routing weights.

The failed action space was:

```text
state -> choose one of even_split / blend_0.75 / blend_0.5 / blend_0.25 / proportional
```

Those policies are all capacity-shaped. They mostly ask how sharply to follow current
producer throughput.

The new action space is:

```text
(order, producer, context) -> value score
softmax scores over eligible producers -> routing weights
```

## Required Floor

Every read must compare against fixed `blend_0.5`. Champion closure or
`closure_vs_search` is secondary; EXP-0036 showed those can be misleading when the champion
dies.

## Recommended First Task

Build a small residual-router probe:

```text
base = blend_0.5 logits
model = per-order/per-producer value scorer
final = base + clipped residual
```

Use the EXP-0036 gate tasks first: s29, s77, s45.

## Data To Include

Minimum per candidate producer row:

- order product and quantity;
- TTL / order age / queue pressure;
- producer available capacity by required type;
- reserved capacity;
- inventory and inbound stock for required inputs;
- producer recipe relation to the order product;
- recent routed/completed/rejected history if available;
- global scarcity summaries.

## Labels / Objective

Avoid "which predefined blend won" labels.

Prefer one of:

- marginal value of increasing a producer's routing logit around `blend_0.5`;
- pairwise preference between two eligible producers for the same order;
- short-rollout value of residual perturbations;
- policy-gradient style improvement with `blend_0.5` as the behavior/fallback policy.

## Definition Of Done

- Deployed rollout beats fixed `blend_0.5` on at least 2 of s29/s77/s45.
- The policy is evaluated as actual routing weights, not table error.
- Failure cases show whether the residuals are too small, too noisy, or directionally wrong.
- If it loses to `blend_0.5` on 2 of 3, stop this branch.

## Non-Goals

- Do not add another selector over `even_split` / blends / `proportional`.
- Do not use `closure_vs_search` as the primary success metric.
- Do not run the full 81-row archive before the 3-task gate passes.

