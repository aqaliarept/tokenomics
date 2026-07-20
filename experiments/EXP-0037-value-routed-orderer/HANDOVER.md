# EXP-0037 Handover — Value-Routed Orderer

## Why This Exists

EXP-0036 closed the trained-orderer path **for blend-policy selection**. It did not close
all possible learned orderers.

Important historical clarification: the project **has** already tried direct producer
scoring. The deployed EXP-0012 `OrdererScoreModel` is:

```text
product-stream view + producer row features -> one score per producer
softmax/decode over eligible producers -> routing weights
```

It was trained from a proportional-throughput teacher:

```text
target = log(makeable_qty_now)
```

That made a real producer-count-agnostic orderer and beat `proportional` on the earlier
held-out scale tasks. But it learned "who can make more right now", not "which routing
weight improves future value on primal-tight tasks".

The next direct-weight attempt was EXP-0032/early EXP-0036 oracle relabeling:

```text
per-cycle search commits one continuous routing vector
slice that vector into per-stream producer-score labels
train the same scorer to imitate those labels
```

That failed because the target was noisy. Search often chose one routing among near-tied
sampled vectors; after centering/slicing, many labels were flat or arbitrary. Diagnostics
entry 11 found roughly 42-50% flat targets on the key tasks, weak variance explained, and
large deploy variance from different torch initializations.

So policies were introduced for a specific reason: **output-space denoising**.

The failed action space was:

```text
state -> choose one of even_split / blend_0.75 / blend_0.5 / blend_0.25 / proportional
```

This converted the teacher output from a noisy producer-count-dependent routing vector into
a small, producer-count-agnostic vocabulary. The intended question was:

```text
Does the search headroom mostly live in a stable named/blend routing family?
```

Then the ladder fixed successive target problems:

```text
raw continuous routing vector
  -> too noisy / arbitrary

named policy hard choice
  -> producer-count agnostic, less arbitrary
  -> but single argmax label still noisy

policy value table
  -> fixes hard-label / tie noise
  -> but still only scores capacity-shaped policies

episode/window policy value table
  -> fixes target time-scale concern
  -> deployed learned switcher still loses to fixed blend_0.5
```

Those policies are all capacity-shaped. They mostly ask how sharply to follow current
producer throughput. They cannot express:

```text
Producer A has more current capacity,
but Producer B is strategically better because of inventory, inbound stock,
queue pressure, TTL, recipe position, or downstream scarcity.
Send more to B.
```

The new action space is:

```text
(order, producer, context) -> value score
softmax scores over eligible producers -> routing weights
```

The genuinely new question is therefore **not** "can we build a direct producer scorer?"
That already exists. The new question is:

```text
Can a constrained value-residual producer scorer improve over blend_0.5
without relearning noisy one-off search vectors?
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

Do not repeat EXP-0012 or EXP-0032. The first implementation must differ in all three ways:

1. Keep `blend_0.5` as the base/fallback, not free-form scores from scratch.
2. Train on value improvement or marginal producer preference, not throughput imitation.
3. Judge by deployed rollout against `blend_0.5`, not by imitation/table loss.

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

Avoid both old failure modes:

- "which predefined blend won" labels;
- "copy this one search-committed continuous routing vector" labels.

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

## Final Outcome

Status: **CLOSED**.

The 3-task gate ran on 2026-07-16 and failed the stop rule:

```text
task  learned_cu  blend_0.5_cu  delta
s29      856.3        891.0      -34.7
s77      696.0        692.3       +3.7
s45      175.0        400.0     -225.0
```

Only 1 of 3 tasks beat fixed `blend_0.5`; s45 also had higher death under the residual
router. Treat EXP-0037 as closed for this residual-router implementation.

Audit caveat: the pulled `gate/` artifacts are from the original pre-P2 pod run, so they
contain `summary.csv`, `spec.json`, and `residual_scorer.pt`, but not the later
`loss_history.csv` / `gate_rows.csv` audit files. The code now emits those on rerun.

Next work should not be another blind feature addition. Build EXP-0038's error microscope:
compare teacher marginal labels to actor residual scores on the same decisions, inspect the
largest mistakes, and identify whether order/backlog/TTL context is actually missing.

## Non-Goals

- Do not add another selector over `even_split` / blends / `proportional`.
- Do not rebuild the EXP-0012 throughput teacher under a new name.
- Do not imitate one noisy per-cycle search vector as a regression target.
- Do not use `closure_vs_search` as the primary success metric.
- Do not run the full 81-row archive before the 3-task gate passes.
