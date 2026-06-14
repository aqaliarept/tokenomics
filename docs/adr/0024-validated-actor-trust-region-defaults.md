# ADR-0024: Make the validated actor trust region the default

## Status
Accepted for Track A.

## Context

ADR-0017 through ADR-0023 changed the optimization problem materially:

- GAE now follows real `(environment, agent)` trajectories.
- Rollout batches continue through complete episodes.
- PopArt no longer moves value predictions when its statistics change.
- The producing actor masks commands the environment cannot execute.
- The critic sees the acting agent and its local decision context.
- Producing rollouts use temperature `0.25` instead of `1.0`.

The last item preserves the greedy action but multiplies the derivative of the
Categorical logits by `1 / temperature`. At temperature `0.25`, the producing
actor's logit gradient is four times sharper. The valid-action mask also
concentrates probability on a smaller support. These are desirable changes, but
they make unconstrained four-epoch PPO updates materially more aggressive.

The old default left `target_kl=None`, while earlier experiments recommended
`0.015`. After the agent-conditioned critic was added, the `0.015`
configuration regressed seed 1 even though seeds 2 and 5 finished well:

| target-KL | seed 1 final | seed 2 final | seed 5 final |
|-----------|--------------|--------------|--------------|
| `0.015` | dead 1, ret 2422 | dead 0, ret 3545 | dead 0, ret 4002 |
| `0.008` | dead 0, ret 3568 | dead 0, ret 3438 | dead 0, ret 4050 |

All rows use 100 iterations, 1024 decisions per batch, four rollout
environments, full difficulty, producing temperature `0.25`, ordering
`log_std_init=-2.0`, and greedy evaluation every ten iterations.

The earlier claim that `0.008` did not help good initializations was measured
before the trajectory, rollout, action-alias, and critic-input fixes. It cannot
be carried forward as evidence about the corrected trainer.

## Decision

Use the matched cross-seed configuration as the training default:

```text
producing temperature = 0.25
ordering log-std init = -2.0  (std approximately 0.135)
actor target-KL       = 0.008
```

Before each actor epoch, compute the sampled-action approximate KL against the
rollout policy. If it exceeds `1.5 * target_kl`, stop before applying that
epoch's optimizer step.

The CLI keeps the ablation paths:

- `--target-kl VALUE` selects a different bound.
- `--no-target-kl` restores unconstrained actor epochs.
- `--log-std-init` and `--prod-temperature` remain independently configurable.

## Evidence

Final greedy evaluation against the baseline target (`dead=0`, return about
3230):

| seed | best observed return | final return | final death |
|------|----------------------|--------------|-------------|
| 1 | 3725 | 3568 | 0 |
| 2 | 3514 | 3438 | 0 |
| 5 | 4395 | 4050 | 0 |

This is the first matched configuration in the corrected trainer that finishes
above the baseline on all three canonical regression seeds.

## Limits

This is a stability control, not another credit-assignment fix. Seed 5 still
temporarily falls to `dead=1` at evaluations from iterations 30 through 60,
then recovers by iteration 70. Stochastic rollout death also remains noisier
than greedy evaluation. The validated default fixes the final regression
criterion but does not prove monotonic learning or eliminate the
stochastic-versus-greedy objective mismatch.

Lowering the actor learning rate to `1e-4` also kept seed 1 alive, but finished
lower (`ret 3334`) than target-KL `0.008` (`ret 3568`). Target-KL is preferable
because it limits only updates that leave the measured trust region instead of
slowing every update.

## Tests

- `test_validated_stability_controls_are_training_defaults` pins the three
  matched defaults.
- `test_target_kl_stops_before_an_out_of_region_actor_step` verifies that an
  already out-of-region epoch does not mutate actor parameters.
