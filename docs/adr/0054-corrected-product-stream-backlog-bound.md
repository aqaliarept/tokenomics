# Corrected product-stream orderer backlog bound (k_stream)

Status: accepted (PRODUCT_AGNOSTIC O0 — contract slice landed, gates green; no training yet)

## Context

ADR-0052 made ordering agents product-stream-scoped: one agent per `(source,
product)` stream (`orderer_obs_version = 3`). The goal was an orderer contract whose
width **scales with producers, not products**. The v3 obs achieved that *directly*
(no product one-hot, no `Q` term) but still leaked product width **indirectly**
through the global backlog scalar `EnvSpec.k`:

```text
k = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)
```

`max_rel` = the largest replenishable-input count over all producers. That formula
was correct for the **old source-keyed** orderer, where one agent multiplexed every
stream a source owned (a producer's investment streams + one replenishment stream per
input). It is wrong for a **product-stream** agent, which is scoped to a single
`(source, product)` stream. So v3 obs (`k * STREAM_SLOT_SCALARS`) and action
(`k * P`) still grew with a producer's input-product width — the ADR-0052 goal did
not hold in practice (see the size-probe table in `PRODUCT_AGNOSTIC.md`: `k` runs
`6,9,15,27,51` as product width runs `1,2,4,8,16`).

## Decision

Add a **corrected, versioned** product-stream contract, `orderer_obs_version = 4`,
that bounds the per-stream backlog by per-stream concurrency only:

```text
k_stream = ORDER_LIFETIME + INVESTMENT_STREAMS
```

Rationale: a product-stream agent is keyed by `(source, product)` — **not** by order
type. So one `(producer, X)` stream can hold both the producer's replenishment orders
for input `X` (up to `ORDER_LIFETIME` in the TTL window) **and** its gated investment
orders whose means-of-production product is also `X`. There are `INVESTMENT_STREAMS`
independently-gated investment types (restoration/upgrade/inventory, each one in
flight), and all three can target the same product, so the worst-case per-stream
concurrency is `ORDER_LIFETIME + INVESTMENT_STREAMS`. Each order needs a controllable
decode slot; an order past the bound falls back to **uniform routing** (`decode_ordering`
zero-fills overflow slots), a hidden, uncontrolled allocation — a correctness gap, not
just a quality one.

The binding constraint is only that `k_stream` not multiply by the **local product /
input count**: the dropped term is the global legacy `k`'s `ORDER_LIFETIME * max_rel`,
which leaks producer input-product width into the contract. `INVESTMENT_STREAMS` is a
**constant** (3), so adding it keeps the ADR-0052 "scales with producers, not products"
goal intact while covering every order a stream can actually hold.

> **Correction (review fix).** The original O0 bound was `k_stream = ORDER_LIFETIME`, on
> the false premise that streams "never carry investment orders." They do: `stream_id =
> (source, product)` collapses replenishment and investment for a shared product. On
> phase1 + the scale grid the old bound was never exceeded (max observed concurrency = 2
> < 3), so it was **latent**, and widening `3 -> 6` left EXP-0012 byte-identical
> (re-verified: mean 1.279, min 1.088). But the bound is now provably sufficient rather
> than empirically lucky. The earlier compactness argument against
> `+ INVESTMENT_STREAMS` is withdrawn — correctness over a 2× narrower head.

### Implementation

- `EnvSpec.k_stream` property: `ORDER_LIFETIME + INVESTMENT_STREAMS if orderer_obs_version >= 4 else k`.
  All product-stream paths read `k_stream`, never `k` — so v1/v2/v3 are byte-identical
  (`k_stream == k` there) and only v4 narrows.
  - `spaces._ordering_obs_dim_stream`, `ordering_action_dim`,
    `ordering_price_indices`,
  - `encode.encode_ordering_stream` (backlog loop),
  - `decode.decode_ordering` (reshape + slot guard),
  - ordering baselines `_ordering_obs_price_action` / `ordering_proportional` (action
    matrix rows). `even_split` already sized off `ordering_action_dim`.
- `TokenomicsAECEnv(product_stream_ordering=True, corrected_stream_k=True)` selects
  version 4. Default stays v3 so EXP-0007A champion checkpoints keep loading.
- `config_fingerprint` already folds `orderer_obs_version` + the derived dims, so v3
  and v4 fingerprints separate — a v3 checkpoint cannot be mis-loaded into the v4
  tensor shape.

## Consequences

- The ADR-0052 "scales with producers not products" goal now actually holds: with
  producer count fixed, widening the product set (incl. growing `max_rel`) leaves v4
  obs **and** action dims unchanged.
- v4 is a **fresh artifact**: the EXP-0007A `m2_stream_stochastic.pt` v3 champion is
  not loadable as v4 (different shape) and must be re-distilled/retrained against the
  corrected contract before promotion. v3 remains a same-shape reference bar.
- Domain dynamics, producer planning, and consumer contracts are untouched.

## Verification (`tests/test_product_stream_ordering.py`)

All local, no training:

- `k_stream` = `ORDER_LIFETIME + INVESTMENT_STREAMS` at v4, `= k` at v3, and covers a
  stream holding `ORDER_LIFETIME + INVESTMENT_STREAMS` orders (replenishment + shared
  investment) so decode controls every one (no uniform-routing overflow),
- growing product width (so `max_rel` grows): v3 obs+action dims grow, v4 invariant,
- encoder length == `v4.ordering_obs_dim` for an active and an idle stream,
- v4 action is `k_stream * P`; `decode_ordering` reshapes at that bound and routes a
  peaked logit to the intended producer,
- product relabeling leaves v4 observations bit-identical across an episode,
- producer-row permutation relocates rows without changing content under the v4 head,
- v3 widths unchanged and `config_fingerprint(cfg,3) != config_fingerprint(cfg,4)`,
- `corrected_stream_k=True` selects v4, narrows the widths, preserves the agent set,
  and runs an episode end to end.

## O1 — portable masked allocator (done)

The corrected v4 contract is still a fixed `k_stream * P` action head; variable
producer count `P` needs a decode that consumes runtime per-producer scores instead.
`env.allocator.allocate_suborders(producer_ids, scores, eligible_mask, quantity,
tokens)` is that portable decoder: one score per producer **row** (variable P, no
fixed K*P), an eligibility mask, and it owns the softmax-over-eligible, largest-
remainder integer split, token apportionment, and conservation (Σq == quantity,
Σtokens == tokens, zero-quantity dropped). `decode_ordering` now builds a per-slot
score row + eligibility mask and delegates — byte-identical to the old inline split
(same sorted-producer order, same largest-remainder rule; full suite + baseline
parity green). Verified in `tests/test_orderer_allocator.py`: two-eligible
conserve/rank, ineligible-receive-nothing, single-eligible-takes-all, zero-quantity
dropped, row-permutation equivariance, conservation under 200 random score tensors.

## O2 — learned candidate scorer contract (built + TDD'd locally)

`train.orderer_scorer.OrdererScoreModel` is the learned half that feeds the O1
allocator: `forward_view(ProductStreamView) -> Tensor[P]`, one score per producer
row. It embeds the product (global) features once and a shared encoder over the
per-producer rows (ProducerStats + ProducerProductStats), with no product/producer
ids — so it is order-equivariant over producers and runs on any producer count P (no
fixed `K*P` head). Mirror of `ProducerScoreModel`.

Contract gates green locally, **no training** (`tests/test_orderer_scorer.py`):
one-score-per-producer-row, score-follows-stats-not-position (swap two producers'
stats → swap their scores), runs-on-variable-producer-count (drop a row → shorter
output), scores → `allocate_suborders` give conserved + eligibility-masked SubOrders,
schema width ties to the domain dataclasses (incl. the `active_investments` 3-tuple) +
`ORDERER_SCORER_CHECKPOINT_VERSION` + eval determinism.

**Value proven (EXP-0012):** distilled from the proportional throughput teacher
(masked MSE, mse 406→0.21), the scorer+allocator beats `proportional` on every held-out
scale economy (seeds 6–9, depth 3): ratios 1.183/1.086/1.470/1.379, mean **1.28×**, min
1.09×, **0/20 death**. It beats the teacher's own baseline because the learned scorer
reads richer per-producer/per-product facts than `min throughput`. Checkpoint
`experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt` (schema-
versioned, reloads via `load_scorer`).

**Producer-count transfer (EXP-0012):** the *same* depth-3 (P=12) checkpoint, run
unmodified on held-out seeds 6–9 at depth 1 (P=4) and depth 2 (P=8), loads + runs with no
shape change, never regresses `proportional` (mean 1.000 / 1.005), and 0/20 death at each —
empirical proof of the id-free order-equivariant contract over producer count. It only
*ties* at shallow depth because routing is slack there; it wins (1.28×) at depth 3 where
routing binds.

**Vs the v3 learned champion (EXP-0012):** head-to-head on held-out seeds 6–9 / depth 3,
producer held at `stock_aware` for both, the portable scorer matches `m2_stream_stochastic.pt`
(mean 1.023×, min 0.983×, 0 death). The `proportional` bar is byte-identical across the v3
and v4 groupings (parity, ADR-0052), so the comparison is on the same economies. The
cheap proportional-teacher distillation thus ties the full EXP-0004→0007A search+DAgger+
stochastic arc, and portability over producer count is free. The orderer portability arc
(O0–O3) is closed.

**Product-count transfer (zero-shot, EXP-0012):** holding P=12 and widening
`products_per_layer` 4→8→16 (streams 52→96→184), the v4 obs/action/`k_stream` dims stay
constant (436/72/6) and the same checkpoint — distilled only at width 4 — beats
`proportional` by a *growing* margin (mean 1.279→1.329→1.509, min always >1.08, 0/60
death) with no re-distill. The id-free per-producer scorer is reused identically on every
stream, so product-count portability is free at inference; the optional cloud re-distill
is unnecessary. Out-of-family recipe transfer (different DAG / capacity-type set) remains
the one open axis.

## Open / deferred

- Distill/train a v4 orderer artifact and an `OrdererScoreModel` checkpoint (O2/O3) —
  cloud, per the training HARD RULE.
- Producer-count transfer audit for the scorer+allocator stack (O3 / orderer-only
  transfer matrix).
