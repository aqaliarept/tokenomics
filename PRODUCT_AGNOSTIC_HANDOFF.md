# Prompt: Product-Agnostic Architecture — Design Discussion

You are joining an in-progress design discussion about making the learned policies
in a cooperative multi-agent RL environment **product- and topology-agnostic**. Read
this whole file first, then read `PRODUCT_AGNOSTIC.md` (the design doc this discussion
revolves around) and `CONTEXT.md` (the authoritative domain glossary — use its exact
terms). Your job is to help refine the design, not to start implementing.

## What the system is

A cooperative multi-agent **RL environment** for a tokenomics economy game (Python
only, all code under `tokenomics/`). Agents learn to fulfil consumer demand by
distributing tokens and managing production capacity. It is a PettingZoo AEC env.

Two shared policies under CTDE (centralized training, decentralized execution):
- **Producing agents** — own capacity, assemble products, decide what to produce,
  what inputs to buy, and what to invest in (restoration / upgrade / inventory
  expansion).
- **Ordering agents** — split each order's quantity + backing tokens across eligible
  producing agents. One ordering agent per `(source, product)` stream (ADR-0052).

Read `CONTEXT.md` for the exact meanings of: consumer, producing agent, ordering
agent, token, emission, investment fund, capacity, process sheet, primal vs composite
product, bid, cut-off price, auction, order, TTL, cycle, score, episode, system death.
Do not substitute synonyms (no "factory", "money", "recipe", "market", etc.).

## Architecture (layering is a hard rule)

```
tokenomics/tokenomics/
  domain/   pure sim logic — NO RL, NO numpy, NO IO, NO logging
  env/      RL adapter only — gym spaces, encode/decode, normalization, reward
  eval/     heuristic baselines + metrics harness (signal_check)
  train/    PPO + DAgger/expert-iteration + boundary_grid (torch optional)
  config/   phase1.json
```

- `domain/` stays pure. `env/` owns all tensor/space/reward concerns. Design
  decisions live in `docs/adr/`; empirical results live one-dir-each under
  `experiments/`.
- The domain is **deterministic** (scripted consumers, no RNG). Same policy + same
  config → same episode. Seeds only change stochastic policies.

## The problem under discussion

Learned policies must not consume product-specific names, ids, or vocabulary
positions as input features. Today the learned PPO/DAgger actors are **fixed-width
tensor contracts derived from the training config**, so they are not
product/topology agnostic:

- **Producer (flat, `producer_obs_version=1`)**: `producing_obs_dim` scales with
  global capacity types `T`, input products `G`, producer count `P`, and order-product
  vocabulary `Q`. Action `producing_action_dim = G + 3` (G replenishment quantities by
  `input_index`, + 3 investment toggles). A producer trained on one input-product set
  has no output head for a different one.
- **Producer (entity, `producer_obs_version=2`, ADR-0053)**: semantically better
  (entity rows: globals + capacity rows + product rows + input rows + recipe-edge
  relation planes), but still a **fixed-width padded Box** — padded to config-wide
  `num_types`, `num_order_products`, `num_inputs`, with relation planes flattened as
  `Q*T` and `Q*G`. Row-order is preserved by the encoder, but a plain MLP can still
  learn slot position.
- **Orderer (product-stream, v3, ADR-0052)**: drops the product one-hot and the
  `K*P` producer-product grid (collapses to `P`), so it scales with producers not
  products *directly*. BUT it still inherits product-width dependence **indirectly
  through the global backlog width `k`**:
  `k = max(ORDER_LIFETIME, INVESTMENT_STREAMS + ORDER_LIFETIME * max_rel)`, where
  `max_rel` grows with a producer's replenishable input count. Action is `k * P`
  logits. So obs and action still grow with product width.

The intended portable shape (already partly built on the producer side):
```
producer = ProducerEntityView -> ProducerScoreModel -> plan_producer_command (deterministic planner)
orderer  = ProductStreamView  -> (future) candidate scorer -> masked allocator
```
The learned part is a shared per-row/per-edge scorer; a deterministic planner/decoder
owns feasibility, conservation, funding, eligibility, integer rounding. Patterns
catalogued in the doc: Deep Sets / masked pooling, set-transformer attention, graph
nets, local-scorer-plus-planner; action patterns: candidate scoring, pointer
selection, parameterized actions, autoregressive allocation, invalid-action masking.

## State of the design doc

`PRODUCT_AGNOSTIC.md` was just reviewed against the code. Verified accurate: the
dimension formulas, the size-probe table, the `k` formula, the decode logic, and all
referenced components/baselines (`ProducerScoreModel`, `plan_producer_command`,
`ProductStreamView`, `ProducerEntityView`, baselines `stock_aware`, `entity_planner`,
`obs_price`, `proportional`, `even_split`). `OrdererScoreModel` does **not** exist yet
(future work). Fixes already applied in this session:

1. **Orderer contract fix** — the corrected stream backlog bound is now
   `K_stream = ORDER_LIFETIME` (not the earlier `ORDER_LIFETIME + INVESTMENT_STREAMS`,
   which regresses on narrow topologies where `max_rel == 0`). A product-stream agent
   is scoped to one `(source, product)` stream; investment orders live in their own
   per-type gated streams and don't add to a consumer/replenishment stream's backlog.
2. Cross-linked the size-probe table to its cause (`k`/`max_rel`, not `Q`).
3. Fixed the Graph Networks arXiv id to 1806.01261.

## What to discuss next (your task)

Help pressure-test and sharpen the design. Open threads:

- Is `K_stream = ORDER_LIFETIME` truly sufficient per stream, or are there stream
  types (investment vs replenishment vs consumer) that need a per-stream-type bound
  rather than one global `k`? Should `k` stop being a single global scalar?
- The portable **orderer** path (candidate scorer + masked allocator) is unbuilt —
  what is the minimal first slice, and what are its feasibility/conservation
  invariants (sub-order quantities sum to unresolved quantity, tokens sum to available
  tokens, no routing to ineligible producers, zero-qty dropped)?
- Portable **checkpoint schema**: load by feature-schema + decoder/planner version
  rather than full config fingerprint. What exactly must be asserted at load time?
- The verification plan separates four failure modes — contract, equivariance,
  policy, scaling. Is the proposed target matrix (A–I) and the producer-only /
  orderer-only / joint transfer audits the right first deliverable (a compatibility +
  behavior matrix, not a new training run)?

## Working rules

- All real training runs on cloud infra (Runpod), never locally — only smoke runs
  (`smoke_ppo --iters` ≲ 50, single-cell sanity, `--dry-run`) and evals run locally.
- New design decision → ADR in `docs/adr/`. New domain term → `CONTEXT.md`. Empirical
  result → a dir under `experiments/`. Tests must stay green (`pytest -q`).
- Stay in the domain language. Be precise about what exists in code vs. what is
  proposed.

Start by reading `PRODUCT_AGNOSTIC.md` end to end, then raise the weakest points in
the current design.
