# Orderer observation enrichment with domain-owned economy metrics

Status: accepted — domain + env layer implemented (EXP-0006A); cloud
training/eval of M2-v3-obs vs the envelope still pending. See "Implementation".

## Context

EXP-0005 closed the first improvement claim: `v4 producer + M2-v1 learned orderer`
beats the deployed hardcoded baseline `stock_aware + proportional`. The stricter
random-inclusive envelope remains open. The Stage 2b search teacher clears that
envelope, while the current student does not reliably clone it, especially on the
known held-out hard case.

The current ordering observation is intentionally sparse:

- per-order slot activity, tokens, quantity, and product one-hot,
- per-producer cut-off price,
- cycle progress.

That contract was useful when the ordering side was kept capacity-free, but the
EXP-0005 residual suggests the orderer cannot see enough public economy state to
reproduce the planner's routing decisions. The missing state is not future
information. It is current and historical economy state: producer load, capacity
health, product demand, producer/product throughput, and producer/product latency.

## Decision

Add a domain-owned economy metrics model for orderer observability. The domain should
own the authoritative raw values and recent-history statistics. The env layer should
only normalize and encode those values into the fixed-width orderer observation.

This is a domain contract, not just an `encode_ordering` convenience. The statistics
must be testable outside RL and reusable by diagnostics, search teachers, and future
graph/entity observations.

The first use is EXP-0006A:

```text
Test whether the EXP-0005 orderer clone gap is caused by missing orderer state rather
than model capacity.
```

Do not expose future information. Allowed signals are current state, open orders that
already exist, and history-derived statistics. Disallowed signals are future demand,
final episode outcome, and planner/search scores.

## Metrics And Formulas

Use these definitions as canonical formulas. The domain keeps raw counts and EMAs;
the env normalizes them for observations.

Notation:

- `p` = producer
- `x` = product
- `k` = order slot
- `t` = capacity type
- `g` = input product
- `C[p,t]` = producer `p` current capacity for type `t`
- `M[p,t]` = producer `p` max capacity for type `t`
- `R[x,t]` = per-unit capacity required by product `x`
- `I[x,g]` = per-unit input product `g` required by product `x`
- `Inv[p,g]` = producer `p` current inventory of input product `g`
- `q[k]` = unresolved quantity in order slot `k`
- `tok[k]` = available tokens in order slot `k`
- `eligible(p,x) = 1 if x in producer[p].produces else 0`
- `EMA_next = alpha * current_cycle_value + (1 - alpha) * EMA_prev`
- `ORDER_TTL` = maximum order age before rejection, currently `2`
- `eps` = small positive denominator guard

### Order Slot Metrics

- `order_age = order.cycle_counter`
- `ttl_remaining = max(0, ORDER_TTL - order.cycle_counter)`
- `order_type = one_hot(consumer, replenishment, investment)` from request kind
- `unresolved_quantity = order.unresolved_quantity()`
- `tokens_per_unit = order.available_tokens() / max(1, unresolved_quantity)`
- `eligible_producer_count = sum_p eligible(p, order.product)`
- `eligible_mask[k,p] = eligible(p, order[k].product)`

### Producer Metrics

- `producer_total_current_capacity[p] = sum_t C[p,t]`
- `producer_total_max_capacity[p] = sum_t M[p,t]`
- `producer_capacity_headroom[p] = sum_t max(0, M[p,t] - C[p,t])`
- `capacity_ratio[p,t] = C[p,t] / max(1, M[p,t])`
- `producer_min_capacity_ratio[p] = min_t capacity_ratio[p,t]` over owned types
- `producer_mean_capacity_ratio[p] = mean_t capacity_ratio[p,t]` over owned types
- `producer_requested_load[p] = producer[p].requested_capacity`
- `producer_reserved_load[p] = sum(reservation.bid.total_capacity for reservation in producer[p].reservations)`
- `producer_total_load[p] = producer_requested_load[p] + producer_reserved_load[p]`
- `producer_cut_off_price[p] = producer[p].cut_off_price`
- `producer_outstanding_inbound_units[p] = sum_g outstanding_inbound[p,g]`
- `producer_active_investments[p] = one_hot(restoration_running, upgrade_running, inventory_expansion_running)`
- `producer_acceptance_rate[p] = EMA(accepted_qty[p]) / max(eps, EMA(routed_qty[p]))`
- `producer_completion_rate[p] = EMA(completed_qty[p]) / max(eps, EMA(routed_qty[p]))`
- `producer_rejection_rate[p] = EMA(rejected_qty[p] + failed_qty[p]) / max(eps, EMA(routed_qty[p]))`

`producer_in_flight_workload[p]` is an encoded tuple, not a raw summed scalar:

```text
(
  producer_reserved_load[p] / cap_ref,
  producer_outstanding_inbound_units[p] / inv_ref,
  producer_active_investments[p]
)
```

Do not add raw capacity units and inventory units together in the domain model.

### Product Metrics

- `open_demand_quantity[x] = sum unresolved_quantity(order)` over open consumer orders for `x`
- `open_demand_tokens[x] = sum available_tokens(order)` over open consumer orders for `x`
- `recent_average_cycle_demand[x] = EMA(new_consumer_demand_quantity_this_cycle[x])`
- `recent_rejected_quantity[x] = EMA(rejected_consumer_quantity_this_cycle[x])`
- `product_depth[x] = 0` for primal products; otherwise `1 + max(product_depth[input])`
- `recipe_input_count[x] = len(I[x,*])`
- `recipe_input_units[x] = sum_g I[x,g]`
- `recipe_capacity_type_count[x] = len(R[x,*])`
- `eligible_producer_count[x] = sum_p eligible(p,x)`

Keep true demand product-level. A metric named "average demand per producer/product"
is policy-dependent if it means "what this producer was routed before"; record that
as routed-load history, not as demand.

### Producer-Product Metrics

- `producer_product_eligible[p,x] = eligible(p,x)`
- `capacity_makeable_qty[p,x] = 0` if not eligible; otherwise `min_t floor(C[p,t] / R[x,t])`
  over capacity types required by `x`
- `input_makeable_qty[p,x] = infinity` if `x` has no inputs; otherwise
  `min_g floor(Inv[p,g] / I[x,g])` over input products required by `x`
- `producer_product_accept_qty_now[p,x] = capacity_makeable_qty[p,x]`
- `producer_product_complete_qty_now[p,x] = min(capacity_makeable_qty[p,x], input_makeable_qty[p,x])`
- `producer_product_makeable_qty_now[p,x]` means `complete_qty_now` unless an
  experiment explicitly uses `accept_qty_now`; record both if possible
- `capacity_bottleneck_ratio[p,x,q] = min_t C[p,t] / max(1, q * R[x,t])`
- `input_bottleneck_ratio[p,x,q] = 1.0` if `x` has no inputs; otherwise
  `min_g Inv[p,g] / max(1, q * I[x,g])`
- `producer_product_bottleneck_ratio[p,x,q] =
  min(capacity_bottleneck_ratio[p,x,q], input_bottleneck_ratio[p,x,q])`
- `slot_makeability[k,p] =
  min(q[k], producer_product_makeable_qty_now[p, order[k].product]) / max(1, q[k])`
- `recent_routed_qty[p,x] = EMA(quantity routed to producer p for product x this cycle)`
- `recent_completed_qty[p,x] = EMA(quantity completed by producer p for product x this cycle)`
- `recent_rejected_qty[p,x] = EMA(quantity rejected or failed by producer p for product x this cycle)`
- `recent_success_rate[p,x] =
  EMA(completed_qty[p,x]) / max(eps, EMA(completed_qty[p,x] + rejected_qty[p,x] + failed_qty[p,x]))`
- `recent_production_throughput[p,x] = EMA(completed_qty[p,x])`
- `recent_completion_latency[p,x] = quantity_weighted_EMA(completion_cycle - routed_cycle)`
- `recent_failure_latency[p,x] = quantity_weighted_EMA(failure_cycle - routed_cycle)`

Use recent EMA/windowed statistics rather than lifetime averages. Routing quality is
state-dependent, and lifetime averages can hide current bottlenecks.

## Observation Normalization

The domain should not store normalized RL features. It should store raw domain units.
The env observation encoder normalizes:

- capacity-like values by `cap_ref`
- quantity-like values by `inv_ref`
- token-like values by `token_ref`
- price-like values by `price_ref`
- lifecycle age/latency by `ORDER_TTL`
- episode progress values by `horizon`
- ratios and rates remain in `[0, 1]` when naturally bounded and are clipped otherwise

## Minimal First Slice

EXP-0006A should start with the smallest feature set likely to answer the question:

```text
order TTL
explicit eligibility mask
producer capacity/headroom/load summary
producer/product accept-now and complete-now makeability
producer/product bottleneck ratio
recent product demand EMA
recent producer/product throughput EMA
recent producer/product completion latency EMA
```

This is enough to test whether the student lacked observable load and makeability
signals before moving to larger network capacity or stochastic distillation.

## Implementation

The domain + env layer is built and test-driven (full suite 264 passed, +28 new
tests). The **full feature catalog** above was implemented, not the minimal first
slice — the minimal slice remains the fallback story if the result is ambiguous.
Cloud training/eval (relabel Stage 2b teacher, train M2-v3-obs, envelope eval on
held-out seeds 6,7,8,9) is the pending follow-up under the repo compute rule.

Module layout (domain owns facts, env only normalizes):

- `domain/metrics.py` — pure, stateless snapshot formulas (no persistent state):
  the producer, producer-product, and static recipe-graph metrics. Reusable by
  white-box diagnostics/search.
- `domain/telemetry.py` — `EconomyTelemetry`: the persistent EMA history (routed /
  accepted / completed / rejected / failed by `(p,x)`, derived rates, latency,
  product demand). Pure Python bookkeeping (no IO/RL/numpy), allowed under the
  `domain/` purity rule.
- `domain/orderer_view.py` — `OrdererEconomyView` and its `OrderSlotStats` /
  `ProducerStats` / `ProductStats` / `ProducerProductStats` dataclasses: the
  authoritative per-ordering-agent feature bundle.
- `domain/system.py` — owns one `EconomyTelemetry`, records routed at
  `ordering_agent_action`, production events at `complete_cycle`, consumer demand at
  order placement, and folds once per cycle (`telemetry.end_cycle()`). Exposes
  `orderer_economy_view(aid)`. A `_routed_cycle[(producer, order_id)]` map dates each
  production event so completion/failure latency can be aged.
- `domain/order.py` — `ORDER_TTL = 2` is now a named constant (was a literal),
  owned by the domain so the obs can expose order TTL/age.
- `env/spaces.py` — `EnvSpec.orderer_obs_version` (1 = legacy, 2 = enriched) and the
  `ORDERER_SLOT_SCALARS / ORDERER_PRODUCT_FEATURES / ORDERER_PP_FEATURES /
  ORDERER_PRODUCER_FEATURES` block constants that drive the versioned
  `ordering_obs_dim`. `build_spec(config, orderer_obs_version=1)`.
- `env/encode.py` — `encode_ordering_enriched(spec, view, cycle_progress)`,
  normalizing the view into a fixed-width vector whose width is computed from the
  same block constants (a test pins encoder length == `ordering_obs_dim`).
- `env/aec_env.py` — `TokenomicsAECEnv(enriched_orderer_obs=False)`; when true,
  `observe` serves the v2 obs to ordering agents. Producer obs and the v1 contract
  are untouched.

### Concrete decisions made during implementation

- **Versioned, opt-in, not a replacement.** The enriched obs is `orderer_obs_version
  2`; default stays v1 so M2-v1 and every existing checkpoint keep their exact
  contract. `config_fingerprint(config, orderer_obs_version=1)` takes the version
  explicitly, builds the spec at it, and hashes it in — so a v2 model cannot be
  silently scored on a v1 economy. (`save_policies` forwards the version; an earlier
  draft wrongly assumed the version would flow in implicitly via `ordering_obs_dim`,
  but the fingerprint had always built the default v1 spec.)
- **Latency dates per bid, cycle-safe under re-routing.** The routed cycle travels on
  each `Bid` (`place_sub_orders(subs, routed_cycle)`), not in a `(producer, order)`
  map. A partially accepted order whose remainder is re-routed to the same producer
  gets a fresh bid, so it cannot overwrite an older live reservation's route cycle;
  completion/failure latency is therefore aged from the real acceptance cycle.
- **`recent_rejected_qty` is rejected OR failed.** Per the catalog, the exported
  producer-product rejection signal sums `rejected_ema + failed_ema`
  (`telemetry.rejected_or_failed_ema`), so a composite reservation that TTL/degrades
  after acceptance still shows the orderer rejection pressure.
- **`product_depth` is cycle-safe.** The recipe graph may legally contain cycles
  (inventory breaks the dependency), so depth cuts a back edge to a product already on
  the current path (contributes 0) instead of recursing forever.
- **Makeability/bottleneck use available, not raw, capacity.** `accept_qty_now`,
  `capacity_makeable_qty`, `capacity_bottleneck_ratio` (and hence `slot_makeability`)
  subtract capacity held by live reservations (`available = current - reserved`),
  mirroring the auction, which reserves before accepting new bids. Producer-level
  health metrics (total/headroom/ratios) keep raw current capacity. Orphaned
  reservations that the orderer cannot see would otherwise make a producer look
  acceptant when its relevant capacity is already committed.
- **Orphaned reservations surface as failed.** When a composite reservation outlives
  its parent order (the order's TTL fires while the reservation still waits, reachable
  under partial/delayed acceptance before the reservation's own TTL), `System` records
  it via `_record_production("failed", ...)` before the telemetry fold, then drops it.
  So it contributes to `failed_ema` / `recent_rejected_qty` / failure latency rather
  than being silently discarded.
- **Latency EMA holds on idle cycles.** `quantity_weighted_EMA(latency)` is folded
  only on cycles with completion/failure events; an idle cycle leaves the latency EMA
  unchanged rather than decaying it toward zero (decay would falsely imply faster
  delivery). All other EMAs fold every cycle, so idle keys decay toward zero — the
  intended recency.
- **Producer-level rates by summation.** `producer_acceptance/completion/rejection_rate`
  sum the per-`(p,x)` EMAs over the producer (the EMA recurrence is linear), avoiding
  a parallel per-producer series.
- **Eligibility is carried once.** The per-`(slot, producer)` `eligible` field in the
  producer-product grid is the encoder's eligibility signal; no separate
  `eligible_mask` block is emitted, to avoid duplicating it.
- **Static graph facts left in raw units.** `product_depth` and the recipe counts are
  small integers within `[0, OBS_CLIP]`, encoded raw (then clipped) since the
  normalization guide gives them no reference; only capacity/quantity/token/price/
  latency values use their refs.
- The producer-product grid is built over the **full `slot x producer`** product so
  the obs is a complete grid matching the `K x P` action layout; ineligible cells are
  zero (the metrics return 0 for ineligible producers).

### Test coverage

- `test_domain_metrics.py` — every pure formula (11).
- `test_domain_telemetry.py` — EMA fold/decay, derived rates, weighted latency,
  demand (6).
- `test_telemetry_integration.py` — `System` feeds telemetry from a real episode (3).
- `test_orderer_view.py` — `OrdererEconomyView` order-slot / producer / product /
  producer-product assembly (4).
- `test_enriched_orderer_obs.py` — versioned dim, encoder dim/normalization/
  determinism, env flag wiring, pettingzoo api_test (4).

## Considered Options

- **Compute all metrics inside `encode_ordering`.** Rejected. That hides domain logic
  inside the RL adapter, makes the statistics harder to test, and prevents search
  diagnostics from reusing the same state.
- **Expose only producer total capacity.** Rejected as insufficient. A producer can
  own several capacity types; the total can look healthy while the product-specific
  bottleneck type is exhausted.
- **Expose full private producer internals without structure.** Rejected for v1. The
  orderer should get public economy metrics that explain routing quality, not a raw
  dump of every internal field.
- **Use lifetime averages.** Rejected. The routing problem is nonstationary, so recent
  EMA/windowed metrics are better aligned with the current bottleneck.
- **Treat routed quantity as product demand.** Rejected. Routed quantity is
  policy-dependent history; true demand remains product-level.

## Consequences

- Orderer observation width changes under v2, so the checkpoint schema bumps — but
  the bump is **versioned and opt-in** (`orderer_obs_version`), not a breaking
  replacement. v1 stays the default.
- Old orderer checkpoints (M2-v1) remain fully usable: they run on v1, whose layout
  is unchanged. A v2 checkpoint and a v1 economy get distinct `config_fingerprint`s,
  so a cross-version eval is caught rather than silently mis-scored.
- Domain unit tests should cover the raw metrics and EMA updates independently of RL.
- Env tests should cover normalization and observation layout.
- EXP-0006A becomes the first gate before wider/stochastic orderer work: if enriched
  observation closes the EXP-0005 clone gap, observability was the blocker; if not,
  proceed to orderer capacity or stochastic distillation.
