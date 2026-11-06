# ADR-0061: Order acceptance is a producer policy; the domain keeps feasibility and execution

Status: accepted (user decision 2026-11-06; change `add-producer-acceptance-policy`; cut-off stays "lowest accepted this cycle", fund gaming recorded as a risk)

## Context
ADR-0025 fixed acceptance as an atomic, single-cycle, price-greedy auction: "a bid that does not fit is rejected,
not booked across cycles". Capacity is reserved at acceptance (no overcommit), and a rejected sub-order is
re-routed next cycle. ADR-0028 (declared `produces`) and ADR-0029 (the producer owns recipe expansion) put the
producer in charge of what it makes and how.

Acceptance is still a domain rule, though. The user (2026-11-06) wants it to be the producer's decision:
- an order should not be rejected just because capacity is busy now, when it could finish before TTL;
- every order needs an explicit outcome the orderer sees next cycle;
- no same-cycle rejection loop.

Lost-unit traces show where today's rule costs units:
- inputs arriving after the TTL, 37% (EXP-0087);
- the acceptance-to-order delay (EXP-0078);
- outbid / no-fit, 13%.

## Decision (proposed)
1. **The domain checks feasibility and executes:** reservations against post-degradation capacity, assembly from
   inputs, TTL, degradation and token accounting.
2. **Acceptance is a producer policy,** at one decision point per cycle after routing: accept or decline per
   routed sub-order.
3. **Accepted work** may execute in a later cycle before TTL, and fails at TTL otherwise. Tokens are burnt at
   acceptance; the cut-off price is the lowest accepted price.
4. **Every sub-order gets an explicit outcome,** observable by its ordering agent next cycle.
5. **Today's auction is the default acceptance policy** and reproduces all existing behaviour byte for byte.
6. **No same-cycle re-routing:** the orderer reacts next cycle.

This supersedes ADR-0025's single-cycle acceptance rule, and its "reserved at acceptance, no overcommit"
invariant becomes policy-dependent. Everything else in ADR-0025 (whole-product orders, quantity splits, inventory,
holding penalty) stands.

## Considered options
- **Keep the auction in the domain and add a carry-over queue.** Rejected: acceptance stays a fixed rule, which a
  learned producer cannot shape.
- **Same-cycle decline and re-route.** Rejected: it makes the cycle recursive (the user's rule).
- **Move the input check only to the orderer.** Kept as the orderer's job. A producer policy may duplicate it
  (experiment A2).

## Consequences
- **A new producer acceptance phase in the env** (K slots), plus additive outcome fields in the orderer view.
- **Existing records stay valid,** behind a goldens gate on the default policy.
- **Learned acceptance becomes possible** in a later change.
