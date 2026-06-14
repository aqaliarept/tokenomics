# ADR-0018: Preserve episodes across PPO rollout cuts

## Status
Accepted.

## Context

`train()` created new environments every iteration and `collect()` reset every
environment at entry. With the default `steps=1024`, `num_envs=4`, and 12 AEC
agents, one batch advances each environment only about 21 cycles:

```text
1024 / 4 / 12 ~= 21
```

The horizon is 100 cycles. Training therefore saw cycles 0-21 repeatedly and
discarded every later state, even though greedy failures occurred much later.
The rollout cut was incorrectly treated as an episode boundary.

This also invalidated the curriculum gate. Surviving episodes did not finish
inside a batch, so `collect()` had zero completed episodes and reported the
fallback `death_rate=1.0`.

## Decision

- Keep rollout environments alive across PPO iterations while their
  `Configuration` is unchanged.
- `collect()` initializes only environments that have never been reset.
- Terminal and truncated environments are still flushed, tallied, reset, and
  reused normally.
- If curriculum changes the actual configuration, create fresh environments;
  degradation cannot be changed coherently mid-episode.
- Return `NaN` when no episode completed in a batch. `Curriculum.observe()`
  treats that as no observation and holds its gated level.
- In gated mode, remove the iteration-based difficulty floor so one
  configuration remains active long enough to produce a real episode result.

ADR-0017 frontier handling remains unchanged: unfinished actions at a batch cut
are dropped from training, while completed predecessors bootstrap from the
frontier state. The environment itself continues from that state.

## Consequences

- Training reaches late-cycle capacity states and system-death precursors.
- A batch boundary no longer changes the environment's initial-state
  distribution.
- Up to one unfinished action per live `(environment, agent)` is still omitted
  at each cut, as documented by ADR-0017.
- Ungated schedules still restart episodes when their rounded degradation
  configuration changes. Gated schedules now hold configuration until a real
  episode result is observed.

## Test

`test_collect_preserves_live_episode_state_across_batches` calls `collect()`
twice on the same live environments and requires cycle progress to increase
rather than return to the first-batch value.
