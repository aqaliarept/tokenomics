# Proposal: add-producer-acceptance-policy

## Why
Today a producer accepts routed sub-orders by a hard-coded auction (ADR-0025 §"Auction is over whole-bundle bids,
atomic, single-cycle"): bids are price-greedy, and a bid that does not fit this cycle's free capacity is rejected
in the same cycle.

The user's direction (2026-11-06):
- missing free capacity should not reject an order automatically, because capacity may free up later, before TTL;
- the auction is a policy, not physics, and as a fixed rule it can undermine trained producer behaviour;
- every order needs an explicit accept / reject outcome the orderer can observe next cycle;
- no same-cycle rejection-and-retry, which would make the cycle recursive;
- the "missing inputs" check belongs to the orderer, and may also be duplicated in a producer's acceptance policy.

There is evidence it matters:
- **Late inputs:** at the final assembler, inputs arriving after the TTL are 37% of rf4's lost consumer units
  (EXP-0087) and do not respond to the buffer (EXP-0088). EXP-0078 showed the acceptance-to-order delay against a
  2-cycle TTL is structural for today's acceptance.
- **Outbid and no-fit:** about 10% and 3% of lost units, decided purely by the fixed auction.

## What changes
- **The domain keeps feasibility and execution only:** reservations, assembly, TTL, degradation and token
  accounting.
- **Which routed sub-orders to accept becomes a producer policy decision,** at a new decision point after routing
  (D1).
- **Each sub-order gets an explicit outcome** (accepted / declined / completed / failed, with a reason),
  observable by its ordering agent next cycle (D2).
- **Today's auction becomes the default acceptance policy** and reproduces every existing record byte for byte
  (D3).
- **The env gains a producer acceptance phase** (D4).
- **Three first experiments** follow (D5).

## Impact
- **Code:**
  - `domain/` (producing_agent, system, order);
  - `env/` (an AEC phase, spaces, encode / decode);
  - `eval/baselines.py` (acceptance policies);
  - orderer and producer views (additive fields).
- **Records:** none invalidated (D3's goldens gate).
- **New ADR:** 0061, a draft; it supersedes ADR-0025's single-cycle rule.
- **Not in scope:** learning an acceptance policy (a later change), and the orderer's own use of outcomes.
