# Producing observation and critic state carry open consumer demand

Status: accepted

## Context

ADR-0031 made each ordering slot's product observable, fixing hidden state on the
routing (ordering) side. The producing side had the symmetric gap on the demand
side: a producing agent decides each cycle what to assemble and whether to
replenish/invest, yet its observation carried **no demand signal at all** — only
its own capacities/inventory, a per-producer summary (capacity + cut-off price),
and cycle progress. The same omission existed in the centralized-critic global
state (ADR-0004), which rolled up producer state and prices but not demand.

Consumer orders are placed in `System.start_cycle()` (via `_place_consumer_orders`)
*before* any producing agent acts, so open demand already exists in `system.orders`
at observation time — it is real, available signal, not future information. With
seeded/randomized demand (B9), the product and quantity a consumer wants varies per
episode and per cycle. A producer could only infer demand indirectly and lagged,
through cut-off prices after the auction. That makes "what to produce ahead of
demand" a partial-observability guess rather than a policy decision, and starves
the critic of the variable that most drives episode value under randomized demand
(see [[b9-randomized-suite-behavior]]: the critic blurs value across demand
regimes).

## Decision

**Expose aggregate open consumer demand per orderable product to both the producing
observation and the centralized critic state.**

- A shared helper `_open_demand_features(spec, open_demand)` emits `2P` features
  (`P = len(order_products)`): the first `P` are unresolved quantity normalized by
  `inv_ref`, the next `P` are available backing tokens normalized by `token_ref`,
  indexed by `order_product_index`.
- `aec_env._open_consumer_demand()` aggregates open **consumer** orders only
  (`order.consumer_request is not None`) by product into `(unresolved_quantity,
  available_tokens)`. Investment/replenishment orders are excluded — they are
  producer-internal, not end demand.
- `encode_producing` and `encode_global` both append the `2P` block immediately
  before the cycle-progress scalar.
- `producing_obs_dim` grows by `2P`; `global_state_dim` grows from `5P_prod + 1` to
  `5P_prod + 2P + 1`.

The producing **action** layout, `decode_producing`, the ordering observation, and
all auction/routing logic are unchanged — this is purely observation/critic
enrichment. The demand aggregation lives in the env layer; `domain/` stays pure.

## Considered options

- **Aggregate (summed) demand vs. raw per-order slots.** Chose aggregate-by-product:
  a producing agent assembles whole products and reasons about how much of each
  product is wanted, not about individual consumer orders (which the ordering side
  owns). Per-order slots would also couple producing-obs width to order count.
- **Quantity only, drop tokens.** Rejected: tokens encode demand *priority* (the
  auction's allocation signal); a producer choosing what to make ahead of demand
  benefits from both how much and how strongly it is wanted.
- **Critic only, leave producing obs alone.** Rejected: the producer is the agent
  that acts on demand; CTDE wants both the decentralized actor and the centralized
  critic informed, and the prior no-critic-only fixes did not lift survival.

## Consequences

- `CHECKPOINT_VERSION` bumps `3 → 4`: producing-actor and critic input layers
  changed shape, so a v3 checkpoint no longer loads (the loader rejects mismatched
  versions loudly, ADR-0026).
- Baselines are unaffected (they read `env.system` directly, not observations), so
  the `signal_check` bar is unchanged.
- Demand is sourced at observation time from `system.orders`, so it reflects the
  current cycle's placed consumer orders (unresolved quantity shrinks as sub-orders
  are accepted within the cycle).
- This is the demand-side counterpart to ADR-0031; together they close the
  observability gaps on both the ordering (product identity) and producing (demand)
  sides under randomized demand.
