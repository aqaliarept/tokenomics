# Ordering observation carries per-slot product identity

Status: accepted

## Context

ADR-0029 reduced an ordering slot's observation to `(active, tokens, quantity)`
and the producer summary to cut-off price only. That was sound while demand was
fixed: in the canonical `phase1.json` each consumer always orders the same product
and each producer's ordering agent always sees the same product stream, so product
identity was a constant the policy did not need encoded.

B9 step 2 (seeded consumer demand) breaks that assumption. A consumer can now
demand a product sampled per episode from a bounded pool, so the product behind a
given ordering slot varies across episodes (and, with arrival randomization, across
cycles). Product identity determines which producers are eligible and what the
order is worth, yet the slot observation did not encode it — so a *trained* policy
faced hidden state: it could not tell which product it was routing. (White-box
baselines are unaffected; they read `env.system` directly.)

The plan's observability contract requires every randomized parameter needed for a
decision to be present in local observations, else the task tests memory under
partial observability rather than policy robustness.

## Decision

**The ordering observation identifies each slot's product with a one-hot over the
order-product vocabulary.**

- `EnvSpec` gains `order_products` (sorted list of every process-sheet product —
  the universe of orderable products) and `order_product_index`.
- `encode_ordering` appends, per slot, a length-`Q` one-hot with `1.0` at the
  slot order's product (all-zero for an empty slot), where `Q = len(order_products)`.
- `ordering_obs_dim` grows from `k·3 + P + 1` to `k·(3 + Q) + P + 1`.

The ordering **action** layout, `decode_ordering`, the per-slot eligibility mask
(`ordering_action_mask`), the producing observation, and the centralized critic
state are all unchanged — this is purely an actor-observation enrichment.

## Considered options

- **Per-slot eligibility mask over producers (P dims) instead of product one-hot.**
  Rejected as the primary signal: eligibility is a deterministic function of the
  product, so the one-hot subsumes it; the mask also costs `P` dims/slot and still
  omits product identity needed for value/quantity reasoning. The trainer already
  masks ineligible action dims, so the policy is not misled about eligibility.
- **Normalized scalar product index.** Rejected: imposes a false ordinal metric on
  categorical product ids.
- **Leave the obs as-is and treat product as hidden state.** Rejected: violates the
  observability contract; turns B9 into a memory/POMDP experiment rather than a
  routing-robustness one.

## Consequences

- `CHECKPOINT_VERSION` bumps `2 → 3`: the ordering actor's input layer changed
  shape, so a v2 checkpoint no longer loads (the loader already rejects older
  versions loudly, ADR-0026).
- Behavior-preserving for the baseline bar: baselines do not consume the
  observation, so `signal_check` is unchanged (`stock_aware` + `even_split` = 100
  cycles, 0% death, 100/100 consumer, return 1267). The prior B8 *trained* result
  (return ≈ 1325–1330) was measured on the v2 obs and must be re-confirmed by a
  retrain on v3 — folded into the B9 training step, not this slice.
- Producing obs and critic are untouched; the env-layer change is confined to
  `spaces.EnvSpec` and `encode_ordering`.
