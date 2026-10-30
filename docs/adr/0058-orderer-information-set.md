# Orderer information set: nothing written during the current ordering phase

Status: accepted (add-m4-orderer-redesign, milestone M4; user decision 2026-10-30). Wording
corrected in plan revision 3 (review round 2): "previous cycles" was too narrow, because cycle t's
consumer orders and producing phase are written before ordering and are inputs.

## Context

ADR-0056 fixed what a lookahead **teacher** may read: realised observable history and state,
never hidden shock state. M4 redesigns the deployed **ordering agent** (per-order choice,
agent-local history; `add-m4-orderer-redesign`). The independent review of that plan proposed
a live "units already routed to this producer this cycle by all agents" feature, read from
`p.bids` during the ordering phase, so that orderers could spread concurrent orders of a product
across producers.

The env is AEC: ordering agents act one after another inside a cycle. A same-cycle cross-agent
signal would make each agent's choice depend on the choices of the agents before it in the AEC
order. The deployed policy would then be a sequential coordination protocol whose result depends
on agent order. Every agent would need to re-plan when an earlier one changes, which is a
recursive re-planning problem. A real distributed economy has no such shared instantaneous view.
Time quanta bound how stale information is, and collisions under eventual consistency are
acceptable.

What the code already does is consistent with excluding that signal:
- `ProducerStats.requested_load` (`domain/metrics.py` `producer_requested_load`) reads
  `requested_capacity`, which `producing_agent.py` writes after the auction of the previous
  cycle. It is last cycle's value.
- `StreamViewCache` (`domain/orderer_view.py`) freezes every producer row for the whole ordering
  phase. Routing (`place_sub_orders`) only appends to `p.bids`, which no row reads, and telemetry
  accumulators fold only in `complete_cycle`.

## Decision

An ordering agent's decision in cycle t SHALL depend only on:
1. **its own orders of cycle t**: product, quantity, tokens, TTL, kind;
2. **everything written before cycle t's `start_ordering`, and nothing written by any ordering
   agent in cycle t.** This state includes previous cycles' outcomes. It also includes what
   cycle t has written before ordering begins: emission, the consumers' orders, the capacity
   hits' effect on capacity, and the producing phase (producer actions, investment funding,
   `outstanding_inbound`) (`domain/system.py` 458–468, 518–525). Concretely:
   - the producer rows (capacities, reservations, inventory, inbound, investment flags,
     cut-off price, last cycle's requested load);
   - the shared producer-product telemetry EMAs folded at the end of cycle t − 1;
   - product statistics;
3. **its own history**: outcomes of its own past sub-orders, folded at the end of earlier cycles
   (agent-local history);
4. **its own earlier decisions in the same turn** (it is the agent's own information);
5. **its own sampling stream**, keyed by (world, episode, replicate, agent, cycle, slot). The
   agent id is a seed key only, never a model input.

It SHALL NOT depend on:
- sub-orders or bids placed by other ordering agents in cycle t (`p.bids` during ordering, a
  graph's current-cycle `pending_bid` edges);
- the AEC position of the agent;
- hidden shock state, or any diagnostic `reason` field of a rejection or failure
  (`domain/producing_agent.py` marks them diagnostic-only; the `outage` rejection and the TTL
  failures during an outage encode shock state, ADR-0056). Agent-local history is therefore
  built from bid-outcome kinds only: accepted, completed, failed (dropped and TTL merged),
  rejected.

**Diversity across agents comes from history, not coordination.** One shared, portable network
(no agent ids) serves every ordering agent. Agents differ only through their agent-local
history and their sampling streams. Each orderer thus ends up with slightly different producer
priorities, learned from its own data.

**Collisions are accepted.** Two agents may send concurrent orders of a product to the same
producer in the same cycle. The atomic auction resolves them, and the next cycle's history and
telemetry reflect the outcome.

## Consequences

- No feature, wrapper or graph edge may read same-cycle routing by other agents. A test permutes
  and alters the actions of the *ordering* agents before agent a in the AEC order. Agent a's
  policy inputs (shared rows, base scores, order features, own-turn features, agent-local
  history) and its sub-orders must stay byte-identical. Producing agents' actions are not
  permuted: the producing phase is written before `start_ordering` and is a legitimate input.
- `StreamViewCache`'s invariant (rows frozen during ordering) is now intended behaviour, not an
  optimisation accident. A change that makes a row read `p.bids` during ordering violates this
  ADR.
- Herding is a known risk. If every agent reacts to the same last-cycle shared signal, they can
  all switch together and oscillate. The M4 plan measures it with a pre-registered
  cycle-to-cycle share-swing metric, contrasting agent-local history with shared-only inputs.
- The step-3 graph input (`graph_primitives_v1`, archive-1) is restored without its
  `pending_bid` relations. `reservation` relations are kept, because a reservation is formed by an
  earlier cycle's auction and is state as of the start of the ordering phase.
- Teachers keep ADR-0056's information set. A teacher label that is a function of other agents'
  same-cycle routing cannot be imitated under this ADR and is not a valid student target.

## Alternatives rejected

- **Live cross-agent routed units (review concern 1's fix).** It makes the policy order-dependent
  and recursive, and it assumes an instantaneous shared view that a real system lacks.
- **Learned agent ids or per-agent offsets.** These are not portable across topologies (the product-agnostic contract,
  `experiments/HISTORY.md` §6, EXP-0012–0022) and break "train small, play large".
- **Deterministic coordination** (e.g. a fixed per-agent rotation offset as `round_robin` uses).
  It is fine for a hand-written baseline, but it is not a learned policy input.
