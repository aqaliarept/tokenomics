# Proposal: add-m6-joint-mpg-gnn

## Why
The project goal (user, 2026-11-06): **self-trained agents reach the same or better consumer order completion as
the best hand-written stack.** Hand-written policies are now baselines and teachers only.

The bar is `entity_planner_buyahead_k5_rf5` (k5_rf5) with MWWk / RR on EXP-0057's worlds: 3,703 / 3,842 consumer
units per episode at 12 producers (EXP-0088), with 42 producers reported.

M5's producer-only residual reached `plateau` (EXP-0095, EXP-0097): about +2–4% units, failing the run-down
clause. M4's orderer-only residual on MWW found no gain (DIAGNOSTICS 28). Each role alone damages what only the
other can repair. So M6 trains **both roles jointly**:
- as a Markov potential game (MPG): one shared objective, so independent policy updates are coordinated;
- with **one topology-agnostic GNN** that runs unchanged at 12 and 42 producers.

## What changes
- **A potential:** consumer units minus side-effect terms, in `env/` (D1).
- **A graph view and encoder:** producers, products, orders and supply links (D3). It follows ADR-0058's
  information set and includes the new remaining-lifetime and outcome fields.
- **Two shared policy heads:**
  - the producer head: investments and buying, plus acceptance through the ADR-0061 hook, gated;
  - the orderer head: whole-order routing (ADR-0059).
  Both start exactly at k5_rf5 + MWWk (D2, D4).
- **A CTDE learner** (MAPPO; D5) and an M5-style pre-registered ladder (D6).
- **A new tranche** for the user to approve (sizing.md).

## Impact
- **New code:**
  - `domain/` graph view, additive and pure;
  - `env/` potential, graph encoder and the acceptance agent turn;
  - `train/` the GNN, MAPPO and reads;
  - `eval/` the decoding check.
- **Reused:** M4 / M5 machinery (bar precompute, keyed draws, versioned pre-registrations, noise floor,
  flag-count guard, validation / held-out reads and the ledger).
- **Gate:** read-only. Success makes a gate candidate; a gate run is a human gate.
