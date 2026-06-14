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

**Auction is over whole-bundle bids, atomic, single-cycle.** A bid carries the
full capacity bundle its sub-order requires (units of every type in the recipe)
and its backing tokens. Price = `tokens / (total capacity units summed across all
required types)`. Bids sort by price descending, accepted greedily. Acceptance is
atomic: a bid clears only if the producer has enough remaining capacity of *every*
required type **this cycle**. There is no multi-cycle capacity carry-over — a bid
that does not fit is rejected, not booked across cycles (the old single-type
`in_progress` carry-over is removed). Demand too large for one producer is spread
by the ordering agent's quantity split and TTL re-orders, not by one producer
grinding a giant bid. Cut-off price stays a single scalar per producer (fund
distribution and the obs encoding already assume this).

**Inventory pipelines production (Theory of Constraints).** A producer holds
inventory of inputs bought for future assembly and finished products awaiting
delivery. Inventory breaks circular recipe dependencies and lets a producer
assemble from stock while replenishing. Total inventory is bounded by a unit-count
limit, expandable by investment. Carrying inventory is a liability: a flat
per-unit holding penalty is applied **in the env layer as a training reward**, not
via tokens (tokens carry no value — keeping the penalty out of the domain
preserves domain purity).

**Composite completion requires inputs in stock; capacity is reserved at
acceptance.** Acceptance is automatic on price. When a composite bid clears the
atomic auction, its capacity bundle is **reserved at acceptance**. If its inputs
are in stock that cycle, it assembles immediately — inputs and the reserved
capacity are consumed → completed. If inputs are not yet in stock, it **holds the
reserved capacity** and waits in-progress up to TTL; inputs arriving in time →
assemble, TTL expiring first → the bid **fails (rejected, −5)**. Because capacity
is reserved at acceptance, there is no assembly-time revalidation and no
overcommit: a producer cannot accept more bundles than its capacity backs.

Reservations are honored out of *post-degradation* capacity, with no change to the
degradation calculation. A reservation is **componentwise** over capacity types —
`reservation[t]` — since producers own several. Capacity available to new bids is
per type: `available[t] = capacity[t] − Σ_live reservation[t]`. Each cycle, after
degradation, the live reservations are **re-validated by re-running the atomic
greedy auction over them**: sorted by price descending, each is re-accepted into
the post-degradation per-type capacity atomically; one that no longer fits (a type
it needs went deficient) is dropped and **fails (−5)**. Because acceptance is
per-type and price-greedy, reservations not touching the deficient type are never
dropped, and among those competing for a deficient type the **cheapest fails
first**. Over-reserving while capacity erodes is thus a real, learnable risk —
reinforcing the keep-capacity-high pressure.

A stalled composite (reserved capacity sitting idle, inputs missing) therefore
**blocks other bids** — and that is the intended pressure. It teaches the producer
to keep capacity headroom high and not let acceptance outrun the inputs it can
stock. The block is removed by replenishment: pre-stocking inputs ahead of demand
means an accepted composite assembles the same cycle and consumes capacity
productively instead of holding it idle. The holding penalty is the opposite cost
(over-stocking). This is the core TOC tension the policy must learn: too little
stock → blocked capacity and timeouts (−5); too much → holding penalty. The
producer has no direct reject lever (acceptance is automatic on price); it governs
over-acceptance *indirectly*, through replenishment and capacity investment.

**Tokens stay a pure demand signal; the central fund stays the sole producer
funding.** Consumer tokens are burnt when their order is accepted — they signal
demand, then leave circulation; they do not cascade upstream. Producers never
spend tokens of their own. All producer purchasing (input products and
investments) is funded by the central investment fund — half of each cycle's
emission — distributed in **two stages**: first per *producer*, proportional to
its (scalar) cut-off price; then, within each producer's allotment, across its own
orders by a fixed survival-first priority — **restoration → replenishment →
upgrade → inventory expansion** — greedily until the budget is exhausted, unfunded
requests expiring. Per-producer-first allocation is deliberate: splitting the fund
per *order* would let a producer grab a larger share just by issuing more orders.

**Replenishment respects shared inventory headroom jointly.** The inventory limit
is a single unit count across all products, so per-input order quantities are
allocated *together*, not clipped independently. Headroom is
`H = inventory_limit − current_inventory_units − outstanding_inbound_units`
(orders placed but not yet delivered already claim headroom). Per-input desired
quantities (each capped by a `max_replenish` config constant) are used as-is when
they sum to ≤ `H`; otherwise they are scaled to sum to exactly `H` by the
largest-remainder rule, preserving proportions. Total ordered never exceeds `H`.

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
  simplex over producers (quantity split apportioned by the largest-remainder rule
  — quantity first, tokens following the realized integer quantities, zero-quantity
  bids dropped, with conservation tests); producing action becomes a `Box`
  (`I` replenishment quantities + 3 investment toggles) and gains multi-capacity
  and inventory obs. The continuous producing action drives a trainer interface
  rewrite — see ADR-0026.
- The auction and producing agent must track multiple capacity types and atomic
  bundle acceptance. Acceptance is **single-cycle**: `produce()`'s old single-type
  multi-cycle `in_progress` carry-over is *removed*, not generalized — a bundle
  that does not fit this cycle is rejected. The only surviving in-progress state is
  a composite that cleared the auction (capacity reserved) but is waiting for its
  inputs.
- A new inventory subsystem (stock, unit-limit, expansion investment, assembly
  consumption) is added to the domain; the holding penalty is added to the env
  reward.
- Curriculum and baselines from Track A must be re-derived against the new model;
  prior target numbers (return ≈ 3230) do not transfer.
