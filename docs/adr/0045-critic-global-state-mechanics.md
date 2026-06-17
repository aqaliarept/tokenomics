# Producer-mechanics block in the centralized-critic global state (scale generalization)

Status: accepted

## Context

ADR-0044 added a per-agent **producer-mechanics block** to the producing *observation* so
the decentralized producing actor could adapt its restore/replenish cadence to the
seed-varying action semantics (degradation, restoration/upgrade amounts and costs,
replenish and inventory-expansion economics). That closed part of the held-out gap
(67 → 103 units, 33% → 17% death) but did not close it.

The plan ("Current Diagnosis", item 1) names the next limiter: the **centralized critic**.
The actor now sees the seed-varying mechanics, but `encode_global` (the critic's global
state) still carried only per-producer rolled-up state — total/max capacity, requested,
inventory units, cut-off price — plus open consumer demand and cycle progress. None of the
mechanics. So under topology domain-randomization two scale seeds with the **same global
vector shape** but **different return dynamics** map to the same critic input. A critic that
cannot see the mechanics driving returns gives weak, noisy value targets, so advantages are
noisy and the producing policy does not transfer (ADR-0044's residual gap).

## Decision

Append a per-producer **mechanics block** to the global state, sited after the existing
5-field per-producer state loop (so the front offsets stay valid) and before the open-demand
block. Per producer, 9 fields (`EnvSpec.global_mechanics_len`):

1. total degradation this cycle, summed over types (`/cap_ref`) — the global state already
   sums capacity per producer, so degradation is summed too,
2. restoration amount (`/cap_ref`),
3. upgrade amount (`/cap_ref`),
4. max-replenish (`/inv_ref`),
5. replenish cost/unit (`/token_ref`),
6. restoration cost (`/token_ref`),
7. upgrade cost (`/token_ref`),
8. inventory-expansion increases (`/inv_ref`),
9. inventory-expansion cost (`/token_ref`).

`global_state_dim` grows by `9 · num_producers`. A new `EnvSpec.global_mechanics_start`
(`= 5·num_producers`) and `global_mechanics_len` locate the block self-documentingly. The
block is the critic counterpart of ADR-0044's obs block: the obs gives the *acting* producer
its own mechanics; the critic gets *every* producer's, since the global state is the shared
cooperative system state.

Checkpoint schema bumped `5 → 6`: the critic input dim changed, so a v5 critic layer no
longer matches and older checkpoints are rejected loudly (ADR-0026).

## Considered options

- **Mirror the full per-type obs block (num_types degradation + 8) per producer.** Rejected:
  the global state already sums capacity per producer, so a per-type degradation vector is
  inconsistent with the rest of the block and inflates the dim with no extra signal at scale
  (one capacity type per producer per level). Summed degradation matches the layering.
- **Expose only degradation + max-replenish.** Rejected for the same reason as ADR-0044: the
  costs and investment amounts also vary per seed and set what each return is worth; a critic
  blind to restoration/upgrade economics still cannot value investments across topologies.
- **Leave the critic on rolled-up state and rely on a deeper net.** Rejected: the missing
  variables are *inputs*, not capacity. No depth recovers state the network never receives.

## Consequences

- Two same-shape scale seeds now expose **different** mechanics in the global state
  (regression `test_two_scale_seeds_expose_different_mechanics_in_global_state`), and the
  per-producer layout is locked by
  `test_global_state_mechanics_block_matches_producer_economics`. The seed-varying return
  dynamics are no longer hidden state to the critic — the precondition for clean value
  targets under topology DR. Whether it closes the held-out bar is a training question,
  measured separately (A/B, plan item 5).
- Full suite green (194 tests). The pettingzoo `api_test` and checkpoint-version guard pass.
- v5 checkpoints (critic trained on the smaller global state) no longer load; they were
  trained on a different critic input and are not comparable on the new economy anyway.
