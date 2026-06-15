# Capacity-free ordering: the producer owns recipe expansion

Status: accepted

## Context

ADR-0025 states ordering agents never decompose a product into recipe components —
that is the producer's job. But the implementation leaked capacity onto the
ordering side anyway:

- `OrderInfo` carried `unit_capacity` and `unit_inputs` (the per-unit recipe),
- `OrderingAgent.handle_cmd` read them and built a fully-expanded
  `Bid(capacities=…, inputs=…)` scaled by quantity,
- the RL ordering *observation* encoded a per-type capacity bundle per order slot
  and each producer's total capacity.

So the ordering agent both knew recipes and reasoned over producer capacity —
exactly the coupling ADR-0025 meant to avoid. Capacity is a producer-internal
concern; the ordering side should reason about products, quantities, tokens, and
price only.

## Decision

**The ordering side is capacity-free; the producer owns its recipes and expands
sub-orders into capacity bundles itself.**

- `OrderInfo` reduces to `(id, tokens, product, quantity)` — no capacity, no
  inputs. `Order` no longer stores `unit_capacity` / `unit_inputs`.
- `OrderingAgent.handle_cmd` validates the policy's split (eligibility +
  quantity/token conservation) and routes each slice as a capacity-free
  `SubOrder(order_id, product, quantity, tokens)` (the producer-facing routed
  unit, in `producing_agent.py`). It never constructs a `Bid`.
- `ProducingAgent` is constructed with the process sheets for the products it
  `produces` and gains `place_sub_orders`, which expands each routed sub-order into
  an internal `Bid` (capacity bundle + input requirements scaled by quantity) from
  its own sheet, then runs the existing atomic auction unchanged.
- The RL **ordering observation** drops the per-slot capacity bundle and the
  capacity half of the producer summary: a slot is `(active, tokens, quantity)` and
  the producer summary is cut-off price only. The ordering policy competes on price
  and quantity, blind to capacity. Producing obs and the centralized critic state
  are unchanged (a producer rightly sees capacity; the critic sees everything).

The low-level auction API is untouched: `Bid` and `place_bids(list[Bid])` remain
the producer-internal auction unit, so the auction/assembly unit tests that
construct hand-crafted bundles still exercise the mechanism directly.

## Considered options

- **Keep ordering-side expansion (status quo).** Rejected: conflates ordering with
  recipe knowledge and producer capacity, contradicting ADR-0025.
- **System expands product×quantity when routing bids** (producer stays
  recipe-free). Rejected: the System becomes the recipe authority and the producer
  still cannot reason about its own production; "producer manages capacity
  internally" is better served by giving the producer its sheets.
- **Strip capacity from the domain but keep it in the ordering observation.**
  Rejected: the ordering *policy* would still be trained on capacity it has no
  domain channel to act on — the obs must match the interface.

## Consequences

- `ProducingAgent(config, sheets=…)` gains a sheets argument (defaults to empty so
  auction unit tests that drive `place_bids` directly need no sheets). `System`
  injects each producer's produced-product sheets.
- `ordering_obs_dim` shrinks from `k·(3+T) + 2P + 1` to `k·3 + P + 1`. Old ordering
  checkpoints are incompatible — folded into the deferred B8 obs rewrite (old
  checkpoints are already being discarded under ADR-0026).
- White-box baselines may still read capacity from `env.system` (they are explicit
  white-box measurement policies, not the agent interface): `ordering_proportional`
  now reads the recipe from the process sheet rather than `OrderInfo`.
- Behavior-preserving for the Phase 1 economy: `signal_check` unchanged
  (`stock_aware` + `even_split` = 100 cycles, 0% death, 100/100 consumer, return
  1267).
