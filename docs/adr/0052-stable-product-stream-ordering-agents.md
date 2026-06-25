# Stable product-stream ordering agents

Status: accepted (EXP-0007A parity + training gate passing — product-stream contract promoted)

## Implementation status (EXP-0007A)

Behaviour-preservation gate passing. Product-stream ordering is an opt-in flag
(`TokenomicsAECEnv(product_stream_ordering=True)` /
`build_system(..., product_stream_ordering=True)`); the legacy source-keyed path is
the default so existing checkpoints (EXP-0006B champion) and their regression tests
are untouched.

Restricted coverage is materialized by `domain.product_stream_agents(config)` — the
single source of truth for both System agent creation and env `possible_agents`.
phase1 yields 8 streams: `c1__30`, `p1__11`, `p2__10`, `p3__10`, `p3__11`,
`p4__10`, `p4__11`, `p4__20`.

Verified in `tests/test_product_stream_ordering.py`:

- `stock_aware + proportional` and `stock_aware + even_split` give bit-identical
  episode metrics legacy vs stream (per-order routing is independent of grouping),
- stream agent set is restricted and deterministic,
- each stream agent only ever holds its own product's orders,
- PettingZoo `api_test` passes in stream mode.

### Product-stream observation contract (orderer_obs_version 3)

Done. Product-stream mode now serves a dedicated simplified obs instead of reusing
the v1/v2 encoders. `TokenomicsAECEnv(product_stream_ordering=True)` selects
`orderer_obs_version = 3`. The action contract is unchanged (`k x P` logits with an
eligibility mask in `decode_ordering`), so baseline parity above still holds.

Layout (`env.spaces.STREAM_*`, mirrored by `encode.encode_ordering_stream`, lockstep
guarded): product features once (9, no product id/one-hot) + `K` backlog slots
(`STREAM_SLOT_SCALARS = 9`, no product one-hot) + one producer row per producer
(`STREAM_PRODUCER_FEATURES = 31` = 16 producer-global + 15 producer-product for the
fixed product) + cycle progress. Width scales with producers and backlog, **never**
with the product vocabulary `Q`.

Domain owns the facts: `domain.orderer_view.ProductStreamView` /
`build_product_stream_view` (one producer row per producer, quantity-dependent facts
computed against the stream's total open backlog), exposed by
`System.product_stream_view(aid)`. The env layer only normalizes + clips.

Proven before any cloud training (`tests/test_product_stream_ordering.py`):

- legacy v1/v2 contracts and the EXP-0006B champion regression unchanged,
- deterministic baselines still preserve behaviour under the new obs,
- product relabeling leaves every observation bit-identical (no product id in obs),
- producer-row permutation relocates rows without changing content, and
  `decode_ordering` routes a peaked logit to the intended producer under any
  producer ordering (action-column equivariance),
- obs width is independent of product count (scales with producers),
- `config_fingerprint` separates v1 / v2 / v3 contracts.

### Training gate (EXP-0007A) — PASSED

Re-distilling the EXP-0006B stochastic orderer on the v3 product-stream contract
(`orderer_improve --product-stream --stochastic`, all other recipe held fixed) clears
the strict random-inclusive envelope on all held-out seeds 6,7,8,9, scored in its own
(stream) layout (env-ratio 1.045/1.086/1.056/1.038 @10 eval seeds), zero death — and is
behaviour-neutral-or-better vs the v2 champion's own-layout ratios (1.026/1.086/1.057/1.022:
≥ on s6 and the hard s9 1.022→1.038, tied on s7/s8). The bars are scored in the
candidate's agent layout (`eval_m2` passes `product_stream=` to the baseline evals): the
`random` envelope draws action-space RNG over the stream-keyed agent set, so a legacy-mode
envelope would be a different bar — deterministic baselines stay bit-identical (parity,
0/10 episode mismatch at depth-3 scale). The simpler, relabel-invariant,
product-count-independent representation clears the strict envelope everywhere → the
product-stream contract is the orderer representation; `m2_stream_stochastic.pt` is the v3
champion. See `experiments/EXP-0007A-product-stream-ordering/`.

Next (deferred): point EXP-0007B automation at this contract.

## Context

The current ordering-agent model groups orders by source:

```text
consumer -> one ordering agent
producer -> one ordering agent
```

That means one ordering agent can hold multiple active order slots whose products
vary. The learned orderer must then solve several structural problems at once:

- which slot is active,
- which product is in the slot,
- which producers are eligible for that product,
- how product-specific makeability/throughput should affect routing,
- how to remain invariant to product-slot and product-id permutations.

EXP-0006A added enriched product/producers metrics to make that state observable, and
EXP-0006B proved that a stochastic orderer can beat the strict random-inclusive
envelope. The next representation simplification is to remove product multiplexing
from a single ordering agent.

The proposed refactor is stable, not dynamic: create ordering agents for product
streams up front so PettingZoo `possible_agents` and spaces stay stable.

## Decision

Introduce stable product-stream ordering agents:

```text
ordering agent = source x product
```

Examples:

```text
order_c1_product30
order_c1_product20
order_p4_product11
```

Each product-stream ordering agent receives only orders for its fixed `(source,
product)` stream. The policy remains shared across all ordering agents; this is not
one neural network per product.

The routing decision becomes:

```text
given this product stream's active order(s), choose a split over eligible producers
```

The observation can then focus on:

- same-product backlog/TTL/tokens,
- fixed product metrics,
- producer metrics,
- producer-product makeability/throughput/latency for that product.

The action remains producer logits with an eligibility mask.

## Scope

This is a domain/interface refactor. It should be behavior-preserving for existing
baselines before being used for new training.

Keep:

- shared orderer policy weights,
- stable `possible_agents`,
- deterministic agent ordering,
- v2 enriched economy metrics as the source of producer/product facts,
- stochastic champion as the reference policy for later adaptation.

Do not:

- create one learned model per product,
- dynamically create/destroy PettingZoo agents per order,
- drop product features entirely if they are still needed for cross-product
  generalization,
- mix this refactor with a new distillation objective in the same first experiment.

## Design

### Agent Identity

For every source that can place orders, create stable product-stream ordering agents.

Consumer sources:

```text
for consumer c
for product x in consumer.product_pool or configured orderable products
create order_{c}_{x}
```

Producer sources:

```text
for producer p
for product x that p can order for replenishment/investment needs
create order_{p}_{x}
```

For maximum simplicity, producer streams may initially be created for every orderable
product. For scale, restrict them to products the producer can actually need:

- replenishment input products for products it declares in `produces`,
- restoration/upgrade/inventory-expansion requirement products.

### Order Placement

When a new order is funded and ready to route, assign it to:

```text
ordering_agent_id = stream_id(order.source, order.product)
```

The stream agent stores only orders for that fixed product.

If multiple orders for the same source/product overlap because of TTL, keep a small
same-product `K_stream` backlog. This is simpler than the current mixed-product `K`
because all slots share product semantics.

### Observation

The product stream observation should not need a product one-hot for slot
disambiguation. It may still include product features so a shared policy can
generalize across streams.

Recommended observation blocks:

- stream header: source type, product features, cycle progress,
- backlog slots: active, quantity, tokens, tokens/unit, age, TTL,
- producer block: capacity/load/cut-off/recent producer rates,
- producer-product block for this fixed product: eligibility, accept-now,
  complete-now, bottleneck ratio, throughput, success, latency.

This should reduce or remove:

- mixed product slot one-hot handling,
- product permutation pressure inside a single agent,
- `K x product` ambiguity.

### Action

Keep a fixed action width over all producers:

```text
action = logits[P]
```

Decode masks to producers eligible for the stream product.

This keeps spaces stable and allows the existing stochastic Gaussian orderer path to
be adapted with minimal infrastructure change. A later graph/entity router can
replace this with an edge scorer, but that is not required for the first refactor.

## Acceptance Criteria

First acceptance is behavior preservation, not new learning.

The refactor passes only if:

- existing white-box baselines still run,
- `stock_aware + proportional` remains statistically unchanged,
- `stock_aware + random` remains statistically unchanged within stochastic variance,
- order conservation still holds,
- funded/unfunded lifecycle behavior is unchanged,
- PettingZoo API still passes,
- v1/v2 observation contracts are versioned and checkpoint-incompatible where needed.

Only after behavior preservation should a retrained stochastic orderer be compared
against the EXP-0006B champion.

## Experiment Plan

### EXP-0007A: Product-Stream Ordering Refactor

Question:

```text
Can stable product-stream ordering agents preserve baseline behavior while reducing
orderer observation/action complexity?
```

Method:

- implement product-stream ordering agents,
- keep policies/baselines behavior-compatible,
- run baseline matrix against pre-refactor metrics,
- run current champion adaptation/retraining only after baseline parity.

Accept:

```text
baseline behavior preserved; no lifecycle regressions; observation/action contract is simpler and stable
```

### Later: Product-Stream Champion Retrain

Only after EXP-0007A passes:

```text
train stochastic orderer on product-stream obs
compare against M2-v3-stochastic champion
promote only if strict envelope and champion gates pass
```

## Considered Options

- **Dynamic orderer per order.** Rejected for the first refactor. It matches the
  conceptual decision but complicates PettingZoo `possible_agents`, spaces, and
  checkpoint compatibility.
- **One neural actor per product.** Rejected. That reintroduces product/topology
  coupling and harms generalization under product relabeling.
- **Keep current mixed-product ordering agents.** Accepted so far, but it forces the
  observation tensor to carry product/slot disambiguation and makes product
  permutation harder.
- **Graph/entity orderer immediately.** Deferred. Product-stream agents are a smaller
  bridge toward an eventual edge-scoring router.

## Consequences

- Number of ordering agents increases roughly to `sources x products`.
- Most product-stream agents may be idle in many cycles.
- Memory overhead should be small; stepping overhead must be measured.
- Observation layout and checkpoint schema will change.
- The orderer policy should remain shared across all product streams.
- This refactor can simplify later graph/entity work by aligning domain agents with
  per-order/per-product routing decisions.
