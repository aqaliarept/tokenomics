# Outstanding-inbound in the producing observation (deep-chain viability)

Status: accepted

## Context

ADR-0040 (stage 5) reported **Finding A: "depth ≥4 is infeasible — the multi-hop
pipeline cannot prime fast enough against order TTL=2."** Before doing the reward work
(Finding B) we re-investigated Finding A empirically, because "raise TTL or extend
priming" is a domain change and we wanted the real binding constraint first.

The TTL hypothesis is **wrong**. Measured on `build_scale_config(seed=7)`, stronger
heuristic baselines + `ordering_proportional` (the competent router, not `even_split`):

- Raising `RESERVATION_TTL` 2→3→4→6 does **not** help — it slightly *hurts* (depth 4:
  145→120 units; depth 5: 23→8). Longer-lived reservations hold capacity while still
  starved of inputs, blocking new bids. TTL is not the limiter.
- A larger investment fund does not lift depth 4 (10× fund: 145→150); it helps depth 5
  only partially. Fund starvation is not the depth-4 limiter either.
- The **white-box** `producing_stock_aware` fulfils depth 4 (≈145 units, 0% death)
  while the **observation-compatible** `obs_stock_aware` — the actual release bar —
  collapses to **0–6 units** at depth 4 and **0** at depth 5.

The only difference between the two is one term: the white-box policy tops up inputs to
`have = inventory + outstanding_inbound`, while the obs policy could read only
`inventory` from the obs vector. Isolating that term confirms it is *the* cause:

| depth | stock_aware WITH inbound | same policy WITHOUT inbound |
|------:|-------------------------:|----------------------------:|
| 3     | 429                      | 428                         |
| 4     | 145                      | **6**                       |
| 5     | 23                       | **0**                       |

**Root cause: the producing observation did not expose outstanding inbound** (units
ordered via replenishment but not yet delivered). A policy blind to in-flight stock
re-orders the same inputs every cycle; at depth 3 the pipeline latency is ≤1 cycle so
this is harmless, but at depth ≥4 the latency grows and the producer perpetually
double-orders and never primes the chain. An RL policy, which also sees only the obs
vector, faces the identical handicap — so this *is* "asking RL to learn in a degenerate
environment," but the degeneracy is in the **observation**, not the TTL.

## Decision

Add a per-input **outstanding-inbound block** to the producing observation, mirroring
the existing per-input inventory-stock block:

- `ProducingAgentView` carries `outstanding_inbound: dict[Product, int]` (defaulted, so
  existing constructors stay valid). The domain already tracked this state on
  `ProducingAgent`; the view just surfaces it. `domain/` stays pure.
- `encode_producing` appends `num_inputs` features (inbound per input, normalized by
  `inv_ref`) **right after** the inventory-stock block and before
  `inventory_units/limit/headroom`.
- `producing_obs_dim` grows by `num_inputs` (`+ self.num_inputs` → `2*self.num_inputs`).
  Tail-relative obs assertions (open-demand block indexed from the end) are unaffected.
- The obs-compatible baseline `decide_producing_obs` now counts inbound
  (`have = inventory + inbound`), exactly mirroring the white-box `producing_stock_aware`.

## Considered options

- **Raise `RESERVATION_TTL` / `ORDER_LIFETIME`.** Rejected: measured to not help (and to
  hurt deep chains); `ORDER_LIFETIME` also drives the ordering `k` and would move dims.
- **Leave the obs as-is and document depth 3 as the ceiling.** Rejected: depth 4–5 are
  in the stage-5 spec ("3–5 levels deep") and are genuinely fulfillable once the policy
  can see what it already ordered.

## Consequences

Deep chains are now fulfillable **from the observation alone**. With the inbound signal,
`obs_stock_aware` + `ordering_proportional` over 5 config seeds:

| depth | obs_stock_aware (before) | obs_stock_aware (after) | white-box | noop |
|------:|-------------------------:|------------------------:|----------:|-----:|
| 3     | ~352                     | 352                     | 359       | 0    |
| 4     | **~3**                   | **142**                 | 145       | 0    |
| 5     | **0**                    | **115**                 | 106       | 0    |

The obs-compatible bar now tracks the white-box baseline at every depth. This corrects
ADR-0040 Finding A: depth 4–5 are viable; the generator's depth-3 default stands for
convenience, not feasibility. Finding B (return rewards inaction at scale) is separate
and addressed in ADR-0042.
