# ADR-0023: Condition the centralized critic on local observation and agent id

## Status
Accepted.

## Context

The critic input contained only producer state:

```text
capacity, max capacity, requested capacity, degradation,
investment-running flags, cut-off prices, cycle progress
```

It omitted:

- which producing or ordering agent was acting;
- the acting ordering agent's incoming orders and token budgets;
- the acting producing agent's local view as an explicit value input.

The environment is AEC and the two shared policies act at different positions
within a cycle. Two transitions could therefore have the same critic input but
different remaining decisions and different return distributions. For ordering
agents, the omitted incoming orders directly determine the bids and next score.

This violates the centralized critic's basic requirement: its input must be
sufficient to predict the value of the current agent decision.

## Decision

Use:

```text
critic input =
    producer global state
    + padded current-agent local observation
    + one-hot acting-agent identity
```

The local observation is padded to the larger of the producing and ordering
observation widths. The one-hot identity also encodes role and AEC position.

The actors remain decentralized and unchanged.

## Consequences

- The critic can distinguish all twelve agent trajectories.
- Ordering values can depend on current incoming orders.
- The input is larger but still small for Phase 1.
- Existing critic checkpoints are architecture-incompatible; actor-only policy
  checkpoints are unaffected because the critic is not deployed.

## Test

`test_critic_input_distinguishes_agents_and_includes_local_obs` verifies width,
dtype, and distinct encodings for different acting agents.
