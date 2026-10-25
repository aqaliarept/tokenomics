# Proposal

## Why

The project's mission — train agents that play the cooperative tokenomics game excellently,
without unexpected side effects ("the result machine must not turn the world into
paperclips") — has no written acceptance bar. 49 experiments have optimised the last
10-20% of headroom on a fixed Phase 1 ruleset that strong heuristics already play near its
ceiling, while the rules and side-effect pillars of the mission stalled. The design record
that once held these intentions (`docs/adr/`, `docs/DIAGNOSTICS.md`) was removed in commit
`18c717d`. This change writes the mission, scope, and every agreed acceptance constraint
down as the project's first baseline specs, so Phase 1 has an exit test and the rules can
grow while training continues.

## What Changes

- Record the mission and treat the training/evolution pipeline as an intermediate step,
  not the product.
- Fix the rules-evolution scope: stochastic demand, supply shocks, longer TTL / multi-stage
  orders are in; new capacity types are out, so system death keeps its permanent meaning
  (ADR-0030 semantics).
- Fix the order of rule work: stochastic demand + supply shocks first, then the order
  lifecycle (TTL / multi-stage).
- Define agent acceptance as a regime-split margin over the best-of-baseline envelope
  (>= +5% calm, >= +10% shocked, >= 1.0 on every world, 0% death) on held-out worlds.
- Define environment-agnostic behaviour: train small (~12 producers), gate at ~10x
  (~120 producers), record 100x as a future target, bound per-decision cost to at most
  linear in eligible producers; search policies are teachers, never deployed.
- Define side effects as harm done outside the optimised metric; make run-down and
  consumer starvation hard constraints; monitor token-priority inversion, edge-running
  margin, and investment crowd-out; keep the existing holding penalty for hoarding.
- Define the stochastic-environment contract: seeded, action-independent, unannounced
  event schedules generated outside the domain (the existing B9 pattern), so the domain
  stays RNG-free and paired comparison stays exact; all five shock kinds are specified,
  capacity hit and producer outage are built first.

Threshold values not yet fixed (X, Y, demand drift limit j, shock magnitudes) are named as
parameters in the specs and are set by a later design.

## Capabilities

### New Capabilities

- `project-mission`: the mission, the rules/agents co-evolution principle, and the
  rules-evolution scope and ordering.
- `agent-acceptance`: the gate a candidate policy stack must pass — margins by regime,
  death, held-out evaluation, scale transfer, and per-decision cost.
- `side-effect-guards`: what counts as a side effect, which side effects are hard
  constraints, and which are monitored.
- `stochastic-environment`: the contract for random demand and supply shocks — seeding,
  action independence, observability, determinism of the domain, and the shock taxonomy.

### Modified Capabilities

None — no specs exist yet.

## Impact

- No code changes in this change; it is specification only.
- Future work bound by these specs: `env/` (event-schedule generation, new stats),
  `domain/` (applying shock events, per-consumer accounting), `eval/` (acceptance gate,
  side-effect metrics, regime-split reporting, scale battery), and `champion.json`
  promotion criteria.
- `CLAUDE.md`, `README.md`, and `ARCHITECTURE.md` still reference the deleted `docs/`;
  reconciling those references is out of scope here and belongs in a follow-up.
