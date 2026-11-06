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

**D2. Action heads.**
- **Producer, per producer node:**
  - investment logits (restore / upgrade / expand);
  - per-input buffer and extra-buy offsets on k5_rf5's planner. The upgrade cap stays as a domain-legality
    filter.
- **Orderer, per order:** logits over its eligible producers, added to MWW's log-weights. One keyed draw, the
  order routed whole.
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

**D4. Initialisation.** Zero-initialised residual heads, so θ = 0 is exactly k5_rf5 + MWWk, tested byte for
byte as in M5. Training starts at the bar. A BC pre-train of the trunk is an option, not the default
(EXP-0035's fidelity wall).

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
| **L5:** validation at 12 (deciding) and 42 | success / plateau / kill / transfer_failure, M5 rules, against both orderers' bars |
| **L6:** held-out, one shot (ledger) | confirmed / not_confirmed |
| **(gated) LA:** learned acceptance, after its diagnostic | — |

Success is L6 `confirmed` against **both** k5_rf5 + MWWk and k5_rf5 + RR. The default is M5's rule: beyond the
floor, with no extra deaths and no worse side-effect shares. **User decision:** whether parity counts, i.e.
within the floor and not below it (the goal says "the same or better").

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
