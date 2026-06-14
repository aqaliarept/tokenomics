# ADR-0020: Lower producing-policy sampling temperature

## Status
Accepted for Track A.

## Context

After ADR-0017 through ADR-0019, deterministic greedy evaluation and stochastic
training survival still disagreed sharply. A representative final checkpoint:

| producing policy | ordering policy | death rate |
|------------------|-----------------|------------|
| greedy | greedy | 0% |
| stochastic | greedy | 65% |
| greedy | stochastic | 0% |
| stochastic | stochastic | 45% |

The producing Categorical entropy was `1.334`, close to the four-action maximum
`ln(4) = 1.386`. The actor had learned useful action ordering, so argmax worked,
but sampling remained nearly random. PPO optimized this stochastic policy while
deployment used argmax, making rollout feedback and the deployed policy
materially different.

The generic entropy bonus was not the sole cause: setting it to zero left
entropy at `1.326` after 40 iterations. The shared reward can rank producing
actions without rapidly separating their probabilities.

## Decision

Add an explicit temperature to `CategoricalActor`:

```text
Categorical(logits = network(obs) / temperature)
```

Use `--prod-temperature 0.25` by default for Track A.

- Positive temperature scaling does not change the greedy argmax action.
- It makes stochastic rollouts closer to the policy that will be deployed.
- Temperature is stored in policy checkpoints; older checkpoints load with the
  previous default `1.0`.

## Evidence

Holding a trained network fixed and changing only sampling temperature:

| temperature | stochastic producing death | return |
|-------------|-----------------------------|--------|
| 1.0 | 70% | 1801 |
| 0.5 | 0% | 2297 |
| 0.25 | 0% | 3410 |
| 0.10 | 0% | 4141 |

`0.25` removes system death while retaining more exploration than the colder
settings.

## Consequences

- PPO likelihood ratios and entropy use the same tempered distribution as
  collection, so the update remains on-policy.
- Actor gradients are sharper by the inverse temperature; ADR-0024 therefore
  makes target-KL `0.008` the matched default.
- Producing and ordering exploration are now controlled independently:
  `--prod-temperature` and `--log-std-init`.

## Test

`test_producing_temperature_reduces_sampling_entropy_not_greedy_action`
requires lower temperature to reduce entropy while preserving argmax.
