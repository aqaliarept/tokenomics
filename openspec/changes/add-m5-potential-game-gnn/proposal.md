# Proposal: producer-side learning on top of buy-ahead (M5), joint learning later

## Why

The producer evaluation (M4 tasks 7.12–7.22b, user decisions 2026-11-05) located the remaining gap
on the supply side, and found a strong hand-written producer.
- **EXP-0074:** about three quarters of lost consumer units are inputs ordered but not delivered
  in time. Producers do order their inputs; upstream can make them. Capacity and restoration
  are minor.
- **EXP-0075:** a longer order life recovers about a quarter of that, mostly the upstream cascade
  (a supplier's own reservation expiring). Directly late inputs barely move.
- **EXP-0076 / EXP-0077:** buying inputs ahead (`entity_planner_buyahead`, a stock of k cycles of
  own recent use) adds +46% consumer units at 12 producers at k* = 4. The gain is still rising at
  the sweep's edge, and MWWk's run-down guard sits at its +5 limit there.
- **EXP-0078:** re-allocating inventory room toward waiting orders only moves the loss. Inventory
  room is binding, and early expansion is the only part of "reservations first" that adds value.
- **M4 step 2** (DIAGNOSTICS 25 / 27 / 28): ES on the orderer needed larger minibatches and more
  generations to see any signal. On the MWW base it hit a real guard wall.

The hand-written rule has fixed knobs: one buffer k for every input, expansion only by rule. A
learned correction can set them per input and per situation from the producer's own view. The
orderer (MWW) has no measured headroom left, so the producer is learned first and alone.

## What Changes

- **A producer residual policy** on top of `entity_planner_buyahead_k4`. Shared parameters adjust
  three decisions from agent-local, past-only features (ADR-0058):
  - per-input buffer size (the k of each input row);
  - input buying (replenishment units beyond the planner's target);
  - inventory-expansion timing.
  At θ = 0 it equals the base exactly. MWW stays the orderer, frozen.
- **Learning:** ES with paired episodes (the M4 `order_es` scaffolding: bar precompute, keyed
  draws, the noise margin, floors, checkpoints), with the guard margin measured on this base
  first.
- **A gated ladder with pre-registered readings:** noise floor, gradient probe, pilot, main,
  validation at 12 and 42, and a held-out read. The success bar is to beat buy-ahead k = 4 + MWW
  on held-out worlds.
- **Topology-agnostic by construction:** one parameter vector shared by every producer and input
  row, trained at 12 and read unchanged at 42.
- **Joint learning is a later, gated phase.** The potential-game / GNN content of this change's
  earlier version applies only after a `confirmed` producer read, and only if a diagnostic then
  shows orderer-side headroom.

## Capabilities

### New Capabilities
- `producer-learning`: a learned producer correction on top of the strongest hand-written
  producer. It covers:
  - the residual policy and its information set;
  - θ = 0 equivalence;
  - the ES learner and its pre-registration;
  - the ladder and its readings;
  - 12 → 42 transfer;
  - the held-out confirmation;
  - the gate for a later joint phase.

### Modified Capabilities
None. The gate harness, roster, side-effect guards and the M4 drivers are read-only to this
change. A learned producer enters a roster only by a human decision.

## Impact

- **New code:**
  - `train/producer_residual.py` (policy class, features);
  - `train/producer_es.py` (the learner, reusing `order_es`);
  - its pre-registration under `train/registry/`;
  - a validation / held-out read reusing `order_es_validate` and the M4 ledger design.
  No `domain/` change.
- **Compute:** Runpod only. A new tranche **M5-P ≈ 100 vCPU-h** for the user to approve (design D9);
  tranche A has ≈ 17 left.
- **Records:** EXP-0079+, HISTORY §15, DIAGNOSTICS entries on stop branches.
