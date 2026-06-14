# Product-assembly supply chain: whole-product orders, multi-capacity producers, inventory pipelining

Status: accepted

## Context

Phase 1 modelled production flat: each producing agent owned **one** capacity
type, every product was capacity-only (no product-to-product inputs), and the
ordering agent split an order across capacity types — effectively reasoning about
recipes on the ordering side. This cannot express a real supply chain (composite
products assembled from other products) and pushes recipe knowledge into the
wrong agent.

We redesign the domain around whole-product orders and a producer-side supply
chain.

## Decision

**Products are primal or composite.** A primal product needs only capacity (e.g.
a natural resource). A composite product needs input products plus capacity to
assemble them. A process sheet lists, per unit, the capacity each type consumes
and the input products (and quantities) consumed.

**Producers own multiple capacity types and do the decomposition.** A producing
agent owns one or more capacity types. It is *eligible* for a product when it
owns every capacity type the recipe requires. Accepting a composite order, it
assembles from input products it holds, applying its own capacity. There is no
designated producer for a product — many producers may be eligible for the same
one. Ordering agents never decompose a product into recipe components; that is
the producer's job.

**Ordering agents split by quantity, not by recipe.** For each order it holds, an
ordering agent splits the order's **quantity** across one or more eligible
producers (a per-order simplex over producers), handing each a sub-order it must
fulfil as a whole. Tokens follow quantity proportionally — one allocation, not
two. Within-order price is therefore uniform; meaningful price competition is
*between* ordering agents at each producer, which is what forms the cut-off.

**Auction is over whole-bundle bids, atomic.** A bid carries the full capacity
bundle its sub-order requires (units of every type in the recipe) and its backing
tokens. Price = `tokens / (total capacity units summed across all required
types)`. Bids sort by price descending, accepted greedily. Acceptance is atomic:
a bid clears only if the producer has enough remaining capacity of *every*
required type. Cut-off price stays a single scalar per producer (fund
distribution and the obs encoding already assume this).

**Inventory pipelines production (Theory of Constraints).** A producer holds
inventory of inputs bought for future assembly and finished products awaiting
delivery. Inventory breaks circular recipe dependencies and lets a producer
assemble from stock while replenishing. Total inventory is bounded by a unit-count
limit, expandable by investment. Carrying inventory is a liability: a flat
per-unit holding penalty is applied **in the env layer as a training reward**, not
via tokens (tokens carry no value — keeping the penalty out of the domain
preserves domain purity).

**Composite completion requires inputs in stock.** Acceptance is automatic on
price, but capacity is applied only *at assembly time*. An accepted composite
order whose inputs are not yet in stock waits in-progress, carries across cycles,
and is rejected on TTL if its inputs never arrive. No capacity is reserved while
waiting, so a stalled order does not block other bids. Pre-stocking inputs ahead
of demand is the producing policy's replenishment job; the holding penalty is the
counter-pressure against over-stocking. This is the core TOC tension the policy
must learn: too little stock → timeouts (−5); too much → holding penalty.

**Tokens stay a pure demand signal; the central fund stays the sole producer
funding.** Consumer tokens are burnt when their order is accepted — they signal
demand, then leave circulation; they do not cascade upstream. Producers never
spend tokens of their own. All producer purchasing (input products and
investments) is funded by the central investment fund — half of each cycle's
emission — distributed to producers proportionally to their cut-off price.

**Investment has three types.** Restoration (recover degraded capacity), upgrade
(expand max capacity of an owned type), and inventory expansion (raise the
inventory-unit limit). Creating a brand-new capacity type a producer did not own
is out of scope for Phase 1.

**Decision split per agent.**
- Producing policy learns: replenishment (what inputs + how much) and investment
  (restoration / upgrade / inventory expansion). Acceptance and production run
  automatically.
- Ordering policy learns: the per-order quantity split over eligible producers
  (consumer orders and producer input orders alike — one policy, both).

**Reward and termination.** Reward keeps `+10` completed / `−1` in-progress /
`−5` rejected, minus the env-layer inventory holding penalty; the goal metric
stays consumer orders fulfilled. System death is redefined: an episode terminates
immediately when some capacity type reaches zero across *all* producers — a
product becomes permanently unmakeable — rather than the old all-producers-at-zero
cascade.

## Considered options

- **Ordering agent decomposes recipes (status quo).** Rejected: puts recipe
  knowledge in the wrong agent and cannot express composite products at all.
- **Single-select ordering (one producer per order).** Considered and initially
  chosen for a discrete, low-dimension action; reverted to quantity-splitting so
  the ordering policy can spread load across producers. Action returns to a
  continuous per-order simplex over producers.
- **Decoupled quantity/token allocation.** Rejected: doubles the action dimension
  to buy within-order price differentiation that cross-agent competition already
  provides.
- **Per-capacity-type auctions.** Rejected: a bid could clear one type and starve
  on another, forcing partial-booking bookkeeping across types and a non-scalar
  cut-off.
- **Inventory holding cost as a token burn.** Rejected: tokens carry no value;
  modelling cost through them creates a purpose duality. The penalty lives in the
  env as a training reward instead.
- **Producers hold token balances / removing the central fund.** Rejected:
  producers only ever distribute centrally-propagated tokens; the central fund
  stays the sole source of producer-to-producer flow.

## Consequences

- Env obs/action spaces must be resized: ordering action becomes a per-order
  simplex over producers; producing obs/action gain multi-capacity-type and
  inventory state. (Implementation detail, tracked separately.)
- The auction and producing agent must track multiple capacity types and atomic
  bundle acceptance; `produce()`'s single-type partial-booking logic generalizes
  to per-type remaining capacity.
- A new inventory subsystem (stock, unit-limit, expansion investment, assembly
  consumption) is added to the domain; the holding penalty is added to the env
  reward.
- Curriculum and baselines from Track A must be re-derived against the new model;
  prior target numbers (return ≈ 3230) do not transfer.
