# Product-Agnostic Architecture Notes

## Principle

Learned policies must not consume product-specific names, ids, or vocabulary
positions as input features. Product ids may exist inside a local dataset only as
internal keys used to construct relations, joins, labels, and audit fixtures. Once a
training or inference tensor is built, the policy should see only local relational
facts such as eligibility, backlog, capacity, latency, bottleneck ratios, prices, and
same-stream inventory state.

This keeps the intended transfer contract clear:

- product relabeling must not change observations or decisions except for equivalent
  row/edge permutations,
- larger product sets should not require a wider learned input unless they also add
  more simultaneously visible local relations,
- product names and product ids are dataset metadata, not model semantics.

## Orderer Contract Fix

Product-stream ordering agents are scoped to a single `(source, product)` stream. A
stream orderer should therefore not scale with the total product set or with the
number of products a producer can consume.

The current implementation still has a legacy dependency through global backlog
width `K`: `EnvSpec.k` is sized with a source-keyed formula equivalent to:

```text
K = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)
```

where `max_rel` grows with the number of producer input products. That formula was
valid for the old source-keyed orderer, where one orderer could manage multiple
product streams. It is not valid for product-stream orderers, where every agent
already manages one product stream.

Required fix:

- keep the existing product-stream v3 contract loadable for old checkpoints,
- add a corrected versioned orderer contract for product-stream mode,
- compute product-stream `K` from per-stream concurrency only, not from producer
  product-set width,
- ensure orderer observation and action shapes may depend on producer count and the
  fixed stream backlog bound, but not on product ids, product names, product
  vocabulary size, or `max_rel`,
- retrain or redistill the orderer for the corrected contract instead of silently
  loading old v3 weights into a different tensor shape.

A conservative corrected stream bound is:

```text
K_stream = ORDER_LIFETIME + INVESTMENT_STREAMS
```

If investment streams are separated further by type, this may be reducible to
`ORDER_LIFETIME`; the important constraint is that `K_stream` is bounded by one
stream's possible simultaneous orders and does not multiply by the number of
products in the local topology.

## Checkpoint Impact

This fix does not invalidate all trained models. It affects trained models that
depend on the orderer observation/action shape:

- existing product-stream orderer checkpoints,
- joint checkpoints that include that orderer module,
- saved configs that assume the old product-stream `EnvSpec.k`.

It should not affect deterministic producer planning, producer scorer/controller
models, consumer models with separate observation contracts, or environment
dynamics.

## Acceptance Gates

The corrected product-stream orderer contract is accepted only if:

- product relabeling leaves orderer observations bit-identical,
- increasing product-set width with fixed producer count does not change orderer
  observation/action dimensions,
- producer row permutation only permutes producer-local rows/actions,
- the old v3 contract remains explicitly versioned and checkpoint-compatible,
- the corrected contract requires a fresh orderer training/distillation artifact,
- size-transfer audits distinguish true policy failure from contract
  incompatibility.
