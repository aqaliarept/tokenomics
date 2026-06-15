# Authoritative `produces` list: eligibility is declared, not derived

Status: accepted

## Context

ADR-0025 derived a producer's eligibility for a product from capacity ownership:
a producer was eligible iff it owned *every* capacity type the product's recipe
requires. Eligibility was therefore an emergent property — computed in
`OrderingAgent._eligible` (and mirrored in `env/spaces.build_spec` to decide which
inputs a producer may replenish) by intersecting the recipe's required types with
the producer's owned types.

This couples two distinct facts: *what a producer is physically capable of*
(capacity types it owns) and *what it is authored to make* (its role in the supply
chain). With multi-type producers the derived rule also over-grants eligibility —
a producer that happens to own every type a recipe needs becomes eligible for that
product even when the economy never intended it to make it. There is no way to
author a producer that owns a broad capacity base but is responsible for only a
subset of products.

## Decision

**A producer declares the products it makes via an explicit `produces` set, and
that set is the authoritative eligibility signal.** Eligibility is no longer
derived from capacity ownership:

- `ProducingAgentConfig.produces: frozenset[Product]` (JSON: `"produces": [..]`,
  an array of int product ids; absent → makes nothing).
- `ProducerInfo.produces` carries it to the ordering side. `OrderingAgent`
  eligibility is now `product in producer.produces` — capacity ownership is not
  consulted at ordering time.
- `env/spaces.build_spec` derives a producer's replenishable inputs from the
  composite products it *declares* (`produces`), not from owned types.

**Capacity ownership becomes a config-time invariant, not a runtime gate.**
`Configuration.validate()` enforces:

1. a producer may only list a product whose recipe it can satisfy — it must own
   every capacity type that product's process sheet requires (else the declared
   product could never actually be assembled);
2. every process-sheet product is produced by at least one producer (else it can
   never be ordered, and env decode would later crash on an empty eligible set —
   the guarantee the old derived check provided).

This keeps the auction and bundle mechanics unchanged: a bid still carries the
full capacity bundle and clears atomically against owned capacity. Only *who is
allowed to receive a sub-order for a product* moves from derived to declared.

## Considered options

- **Keep derived eligibility (ADR-0025).** Rejected: conflates capability with
  authored role and over-grants eligibility for multi-type producers; no way to
  scope a producer to a product subset.
- **`produces` filters derived eligibility (intersection).** Rejected: redundant
  — once `validate()` guarantees a listed product's types are owned, the
  intersection equals the list. The list alone is simpler and the single source of
  truth.
- **Drop capacity types entirely (generic capacity + product list).** Rejected:
  far larger blast radius across ADR-0025/0026/0027 and all of `env/`; typed
  capacity is load-bearing for the per-type auction, degradation, and obs layout.

## Consequences

- `phase1.json` and `composite.json` (and all inline test configs) gain a
  `produces` list per producer; the legacy single-type `"type"/"capacity"`
  shorthand is dropped from the shipped configs in favor of the `capacities` map.
- Behavior is preserved for configs where each product had exactly one
  capacity-eligible producer (the Phase 1 economy): `signal_check` is unchanged
  (`stock_aware` + `even_split` = 100 cycles, 0% death, 100/100 consumer, return
  1267).
- A producer can now own a capacity type without being eligible for every product
  that uses it — eligibility is authored, enabling future multi-product producers
  scoped to a subset of what their capacity could physically make.
- `validate()` rejects a producer that declares a product it cannot satisfy, and
  a process sheet no producer makes — catching dead config at load time.
