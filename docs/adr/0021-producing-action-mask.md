# ADR-0021: Mask producing actions whose investment is already running

## Status
Accepted.

## Context

The producing action is a two-bit command:

- bit 0: restoration;
- bit 1: upgrade.

When an investment is already running, `TokenomicsAECEnv.step()` silently clears
the corresponding requested bit before executing the command. For example,
while restoration is running:

```text
sampled action 1 (restore) -> executed action 0 (noop)
sampled action 3 (both)    -> executed action 2 (upgrade)
```

PPO stored the sampled action and its log probability, then assigned the reward
from the transformed command to it. The policy gradient therefore credited
actions that were not executed. It also counted aliases as distinct exploratory
actions, inflating producing entropy.

The observation already exposes `upgrade_running` and `restoration_running`, so
valid actions are known before sampling.

## Decision

Mask invalid Categorical logits to negative infinity:

- if restoration is running, mask actions 1 and 3;
- if upgrade is running, mask actions 2 and 3;
- if both are running, only action 0 is valid.

Use the same masked distribution for:

- stochastic rollout collection;
- PPO log-probability recomputation;
- entropy;
- deterministic greedy evaluation.

The environment keeps its defensive bit clearing for API robustness.

## Consequences

- Collected actions and executed commands now match.
- PPO likelihood ratios remain on-policy.
- Producing entropy measures meaningful command choice rather than action
  aliases.
- Greedy evaluation selects the best valid command instead of relying on
  post-selection mutation by the environment.

## Test

`test_producing_actor_masks_in_flight_investments` verifies zero probability for
invalid actions and a deterministic noop when both investments are running.
