# Design: add-m6-joint-mpg-gnn

## Context: earlier failures, their diagnosed causes, and how M6 avoids them (root-cause rule)

| earlier attempt | diagnosed cause (source) | how M6 avoids it |
|---|---|---|
| Joint PPO from scratch (EXP-0001) | producer restoration misallocation starved the middle layer, with both roles learning from scratch on a moving partner (HISTORY §1) | start exactly at the strong pair (D4); run-down and starvation are in the potential (D1) |
| Relational GNN by imitation (EXP-0039 → 0041) | the supervised label was unstable (EXP-0040 `SEARCH_ONLY`, winner overlap 0.60–0.74); its graph view carried `pending_bid`, now forbidden (ADR-0058) (HISTORY §9) | no labels: outcome RL; an ADR-0058 graph view; the cross-topology decoding check (D3) |
| Trained orderer (EXP-0012), losing to RR | it needed whole routing **and** varied receivers together (EXP-0056 `joint`, DIAGNOSTICS 19); which receiver property matters is **undiagnosed** | the orderer head is a whole-order choice decode (ADR-0059) starting at MWWk; the receiver property is left to learning |
| M4 orderer ES residual on MWW (EXP-0070 / 0071) | perturbing routing caused real run-down / starvation that only producers can repair (DIAGNOSTICS 28); whether MWW is near the routing ceiling is **undiagnosed** | the producer head learns jointly on the same potential |
| Trained producer (EXP-0035) | fidelity wall diagnosed and fixed (multi-task interference; 2000 epochs). It is a distilled imitation, capped near its teacher. Its gate starvation (EXP-0045) is **undiagnosed** | RL beyond the teacher; starvation is in the potential |
| M5 v6, θ_e expansion trigger (EXP-0093 `kill`) | fund dilution: extra investment orders put more producers in the fund split, and a sole maker's restoration went unfunded (DIAGNOSTICS 30) | fund state is a feature, deaths and run-down are charged in the potential, and expansion is a gated head (D2); the pilot reports investment rejection |
| M5 v7 / v8 (EXP-0095, EXP-0097 `plateau`) | the run-down share stayed above the base. The term's weakness is a **hypothesis, not a diagnosis**: end-of-episode ρ is noisy (SD 0.034) against a 0.01 shift | the potential uses a per-cycle capacity term, not the end snapshot (D1). If L5 plateaus on run-down again, first a diagnostic that compares per-cycle and end-ρ signal-to-noise |
| Learned acceptance, hand-written A1 (EXP-0098 `harmful`) | **undiagnosed** (the first explanation was a counter artifact, corrected) | learned acceptance is gated behind a diagnostic (D2, risks) |
| Imitation by the scorers (EXP-0034 / 0035 fidelity wall) | multi-task optimisation interference: one scorer across many topologies, too few epochs, averaged away per-task behaviour (EXP-0035, fixed at 2000 epochs) | Stage I imitates **actions per agent** (each producer's command, each order's routing distribution) on M4's pinned training worlds; it trains to a fidelity criterion, not a fixed epoch count; the check is behavioural (D4) |
| GNN imitation of a search label (EXP-0039 / 0040) | the label was unstable: continuation-averaged, so future-sensitive (`SEARCH_ONLY`) | the label is the **deterministic** hand-written action given the agent's view (k5_rf5's command; MWW's weights before the keyed draw), verified by a determinism test (same view, same label); features are ADR-0058-only, verified by leak tests (a policy reading anything else raises) |

**Supporting work:**
- the earlier MPG + GNN draft (add-m5 at 980c307: networked potential, GNN, PG ladder);
- docs/research/cloud-social-manufacturing.md: losses are upstream-supply and maintenance-coupled, not routing
  alone; the ranked mechanisms all need producer + orderer state together.

## Decisions (each with a default; every one is a decision for the user)

**D1. Potential / team reward.** Per cycle:
- Φ_t = consumer units completed − λ · Σ_p max(0, X − ρ_p,t) − μ · starving consumers_t − ν · death_t;
- ρ_p,t is producer p's capacity ratio each cycle, a dense signal, not the end snapshot (the M5 lesson);
- X and Y come from the side-effect params v2;
- λ and μ are set by M5's noise-scaled rule (σ_units / σ_term on k5_rf5's re-keys, training rows only); ν is
  large (a death ≈ the episode's base units).
- **Rewards:** identical interest (every agent gets Φ, an exact MPG). Option: a κ = 1 networked share (Zhou et
  al.), which is the probe's second arm.
- **Readings:** units and L5 / L6's side-effect clauses, unchanged.

**D2. Action heads, fully learned (user decision 2026-11-06).** The GNN outputs the actions itself; no
hand-written planner runs at deployment.
- **Producer, per producer:**
  - three investment logits (restore / upgrade / expand);
  - per input row, a replenish fraction in [0, 1] of the producer's `max_replenish`, which the env's decode
    turns into units.
  - The upgrade cap is not a filter: the learner must learn it, and Stage I imitates k5_rf5's capped
    decisions.
- **Orderer, per order slot:** logits over the producers; masked to the eligible ones; one keyed draw (MWWk's
  key); the order routed whole (ADR-0059).
- **Acceptance:** default **off** in the first ladder; the default auction stays. A gated later stage adds an
  accept head per routed sub-order through the ADR-0061 hook, made an AEC turn (the slot action deferred from
  ADR-0061). Gate: the A1 diagnostic (risks).

**D3. GNN graph.**
- **Nodes:**
  - producer: capacity per type, ratio, degradation rate, inventory, inbound, reservations / pending claims,
    fund received, cut-off;
  - product: recipe depth, demand EMAs;
  - order: quantity, tokens, **remaining lifetime**, type;
  - ordering agent.
- **Edges:**
  - produces (producer → product);
  - recipe (product → input product);
  - supply links (a producer's buys from another, last cycle);
  - order → product, and order → eligible producer;
  - **last cycle's sub-order outcomes** on the (ordering agent, producer) edges, as accepted / declined /
    failed counts by reason, with the outage reason withheld.
- **Rules:**
  - **Both new fields are included:** the remaining lifetime and the per-sub-order outcomes are exactly the
    own-order and next-cycle-outcome information ADR-0061 added.
  - No IDs; ADR-0058 holds (no same-cycle routing or bids).
- **Trunk:** typed message passing, 2 rounds, hidden 32. The parameter count is independent of N.
- **As built (task 2.1):** each actor's graph is its own ADR-0058 view: the producer's entity view (capacity,
  product and input rows with recipe relations), and the stream agent's view (producer rows with last cycle's
  outcomes, order slots with remaining lifetime, the product). The global producers + products + orders +
  supply-links graph is for the CTDE critic only (task 2.3).

**D4. Initialisation by imitation (Stage I; user decision 2026-11-06).**
- **Behaviour cloning** of k5_rf5 + MWWk on M4's pinned training worlds (EXP-0058 `bar/`), excluding the 16
  monitor world-episodes.
- **Labels, per agent decision:**
  - producer: k5_rf5's command (BCE on the 3 gates; Huber on the replenish fractions);
  - orderer: MWW's normalised weights over the eligible producers, as a soft label (KL);
  - all deterministic given the view (tested).
- **Fidelity metrics, reported:**
  - producer: exact-command agreement, gate accuracy, replenish MAE;
  - orderer: mean KL and top-1 agreement.
- **The imitation check (pre-registered, deciding).** The imitated policy (the GNN alone) is played on the 16
  monitor world-episodes at replicates 0 / 1, against k5_rf5 + MWWk's bar rows at the same keys.
  - g_imit = the equal-weight mean of (imitated − bar) units; F_imit = M5's floors rule.
  - **`faithful`** if g_imit ≥ −F_imit, with no extra deaths. Otherwise **`short`** (stop: the user hears it
    before any RL), with the agreement metrics locating the shortfall.
- **Then MAPPO** from the imitated weights (D5).

**D5. Learner.**
- **Default: MAPPO under CTDE.** Shared actor per role; a centralised GNN critic on the global graph (training
  only); per-decision credit through log-probabilities.
- **Why not ES:** ES needed 40 generations for 18–23 parameters (M5) and scales poorly to thousands.
- **Compute:** CPU pods, batch-1 Python rollouts on one process per core, torch 1 thread per worker; updates on
  the pod CPU. The GNN forward cost is measured in L0.

**D6. Ladder** (pre-registered per step in a frozen `prereg_m6_v<n>.json`; M5 rules and floors):

| step | reading |
|---|---|
| **L0 (code):** zero-residual identity; cross-topology decoding check at 12 / 42 / one unseen family; ADR-0058 information tests; timing probe | pass / fail |
| **L1:** bar + noise floor for k5_rf5 + MWWk / RR; reuses EXP-0090 where pins match | as M5 |
| **L2:** gradient probe, PPO advantage-weighted gradient cosine on two disjoint minibatches (identical vs networked arms) | `signal` / `weak` / `no_signal` |
| **L3:** pilot, seed 0 | `feasible` (g_tr > F_tr) / `unresolved` |
| **L4:** seeds 0 and 1 to full length | — |
| **L0b:** Stage I imitation + **the imitation check** (D4) | `faithful` / `short` → the user |
| **L5:** validation at 12 (deciding) and 42 | **pre-registered parity rule** (user decision): `success` if for both seeds the candidate − k5_rf5 + MWWk ≥ −F_val at 12, with no extra deaths and run-down / starvation shares ≤ the base's; `kill` if below −F_val or extra deaths; `transfer_failure` if success at 12 and below −F_val at 42 |
| **L6:** held-out, one shot (ledger) | confirmed / not_confirmed |
| **(gated) LA:** learned acceptance, after its diagnostic | — |

Success is L6 `confirmed`: the parity rule above, on held-out worlds, against k5_rf5 + MWWk (and RR reported).
**Parity counts** (user decision 2026-11-06).

**D7. Gate.** M6 does not change the gate. `confirmed` makes a gate candidate. The roster v3 draft carries k5_rf5
(carry-forward note); a gate run is a human gate.

## Risks
- **Learned acceptance.** A1's harm is undiagnosed. **The diagnostic first:** a per-cycle displacement trace on
  EXP-0098's worlds. Do pending claims taking capacity before new bids (`produce` step 3b) displace
  higher-priced new work, and does that explain the units lost? Offline replays of A1 and of the default,
  local. Its outcome decides whether an accept head is safe.
- **Credit at 42 producers** with identical interest. Mitigation: the networked arm in L2.
- **GNN forward cost.** L0's timing probe re-sizes the tranche before any pod.
- **Fund dilution recurring** through learned investments (DIAGNOSTICS 30). Investment rejection and fund
  share are reported per generation; deaths are charged.
