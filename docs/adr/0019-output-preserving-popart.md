# ADR-0019: Output-preserving PopArt value normalization

## Status
Accepted.

## Context

ADR-0007 normalized critic targets using EMA mean and variance. The statistics
were updated without compensating the critic's final layer:

```text
V(s) = sigma * normalized_value(s) + mu
```

Changing `mu` or `sigma` therefore changed every denormalized prediction even
when network weights were fixed. Stored values and rollout bootstraps were in
the old units, while the critic immediately moved to new units before any
gradient step.

The variance update also averaged within-batch variances but omitted variance
caused by movement of the batch mean.

## Decision

Use output-preserving PopArt:

1. Update EMA mean and variance, including the between-mean variance term.
2. Rescale the final linear layer so denormalized predictions are invariant:

```text
w' = (sigma / sigma') * w
b' = (sigma * b + mu - mu') / sigma'
```

3. Fit the normalized target after the invariant rescaling.

## Consequences

- Updating normalization statistics alone cannot change `V(s)`.
- GAE values, finite-rollout bootstraps, and critic predictions remain in one
  consistent return unit.
- The normalized critic loss remains an O(1) optimization diagnostic.

## Test

`test_popart_stats_update_preserves_critic_values` records predictions, updates
normalization with a substantially different target distribution, and requires
the denormalized predictions to remain equal within floating-point tolerance.
