# Teacher information set and posterior forks

Status: accepted (add-m3-development-audit, milestone M3 development audit; design D4; task 6.1)

## Context

A lookahead teacher labels a decision by rolling out forks of the env. A fork that is a plain
deep copy of the true env carries the hidden future: the remaining cycles of an active outage
and every future capacity hit. Its labels then depend on information no observation holds, so
no student can learn them. This is the EXP-0038B cause ("target not a function of the
observation"). `establish-project-baseline` D4 fixed the rule in outline: keep the realised
observable history and state, resample every forecast-relevant hidden shock state from the
posterior given that history and the known generator, keep only draws that reproduce the
observed consequences, log the acceptance rate, fall back below a floor, strip hidden domain
fields before rollout and keep drift continuous. ADR-0055 made shocks seeded and unannounced,
and ADR-0057 added demand patterns to the world space. This ADR fixes the rule as implemented in
`env/shock_posterior.py` and `env/future_fork.py`.

## Decision

### Public sources

The history extractor (`HistoryRecorder`) wraps the policies and reads only:
- every agent's encoded observation and action, including each producer's own capacities;
- the orderer telemetry: routed, accepted, completed, rejected and failed EMAs per producer and
  product;
- the realised state that forks keep anyway (orders, bids, reservations, inventory).

It never reads the env's shock schedule, the config's shock model, the domain's schedule or
shock indexes, the episode stats' shock digest, or the diagnostic rejection and failure reasons
(task 1.1 computes `outage` there from hidden state). It works on a *stripped* copy of the
system: each hidden field is mapped away while copying, by identity, and is never read. A test
replaces all of them with a sentinel that raises on any use. Another test scrambles or removes
the reasons; in both cases the history and the labels stay byte-identical.

Per played cycle the extractor derives two kinds of consequence:
- **Capacity hits.** A one-cycle replay of the stripped pre-production copy gives each
  producer's capacity before the next cycle's hits. The drop to the producer's next observation
  is the hit's removal, recorded with the pre-hit capacity.
- **Outage status.** Each producer's auction is run on copies with and without an outage. The
  result whose accepted / completed / rejected EMA fold equals the public telemetry is the
  explanation: `normal` or `blocked`. If both results fold to the telemetry, the status is
  `free`: for example, a producer that rejected every bid while full could have been in an
  outage or simply full.

The one-cycle replay must reproduce every telemetry EMA; otherwise the extractor raises
rather than guessing.

### Posterior and two-sided acceptance

`shocks-i` draws a fixed number of i.i.d. events per kind, so the posterior factorises. A
particle is built constructively, without prior rejection, which D4 shows is infeasible:
1. Each observed hit gets one hit slot. Start, target and capacity type are fixed. The
   magnitude comes from the generator's rounded uniform, conditioned on floor(m·c) equal to the
   observed drop.
2. Each run of `blocked` cycles gets one outage slot. (start, duration) is uniform over the
   pairs that cover the run without touching a `normal` cycle. A run that reaches the last
   played cycle is open, so the remaining duration is drawn conditioned on the elapsed cycles.
3. The other slots come from the generator truncated to "not observable by t": a hit after t,
   or one that removed nothing; an outage whose played cycles are all non-normal. The `free`
   cycles thus form D4's generator-weighted mixture of "outage" and "full".

Every particle is then validated two-sidedly by exact consequence predicates: it must reproduce
every observed drop and outage status and imply none that was not observed. The replay from
reset with the recorded actions, comparing every observation and the telemetry, is the
reference. Tests check that the predicates agree with it on accepted and rejected draws.

### Floor

Pre-registered numbers:
- at most 1,024 draws per decision cycle;
- sampling stops once 16 particles are accepted (the EXP-0042 maximum scenario count);
- the acceptance rate is logged for every decision cycle.

With fewer than 16 accepted particles, the decision is **unresolved** and the teacher acts as
its floor policy (`round_robin` in PoC-A; `blend_0.5` in EXP-0042). Sampling runs once per
decision cycle and is cached for every orderer of that cycle. The sampler is seeded from the
task, the episode seed and the cycle, never from hidden state.

What the logged rate means. The constructive proposal is consistent by construction, so the
two-sided validation is a guard, not a filter:
- **Feasible plan:** every draw is accepted (rate 1.0), and sampling stops after 16 draws. A
  rate below 1 means a proposal bug, and a test checks that the guard catches one.
- **Infeasible plan:** no particle can explain the history. The cycle is unresolved, with
  draws = 0, accepted = 0 and rate 0.0.

The causes of an infeasible plan:
- an observed hit outside the generator's window or targets, or one whose drop no magnitude in
  the range produces;
- more observed hits, or more blocked runs, than the generator's count;
- a blocked run that no single (start, duration) pair covers, e.g. two overlapping outages on
  one producer spanning more than the maximum duration;
- an event slot left over with nowhere unobservable to go, e.g. at t ≥ the window's end with
  fewer hits observed than the count.

### Stripping

A fork is a stripped deep copy of the env. Its true pending events, outage end cycles and shock
digest are replaced while copying. Only the particle's future events are injected: hits after
the decision cycle and outages active from it on.

### Demand continuity

`future_fork.resample_future_demand` redraws only the unconsumed demand suffix, on the fork's
own stream:
- **B9 without drift:** keeps the EXP-0040 i.i.d. tail.
- **Drift:** continues the drift process from the current cycle's orders, with at most
  floor(j·N) changes per transition.
- **Burst and seasonal patterns:** resample the future from the pattern's generator. These
  draws never depend on an earlier draw, and the seasonal envelope is evaluated at the forked
  cycle.
- **Moving hit and regime switch:** refused. Their futures depend on hidden pattern state that
  this rule defines no posterior for, so those worlds are not taught (D5). Covering them is an
  amendment to this ADR.
- **Fixed (cyclic) schedules:** never resampled.

### Deterministic futures

On a world with cyclic demand and no shocks nothing is hidden, so every scenario is identical
and the teacher uses one scenario. With zero variance, the confidence rule accepts a strictly
positive gain. Pre-registered (D4).

## Considered Options

- **Prior sampling that ignores the history.** Rejected: its draws contradict observed
  consequences, so it produces labels no observation supports (the EXP-0038B cause).
- **Prior rejection against the history.** Rejected: correct but infeasible under `shocks-i`
  (≈ 3e-4 acceptance per observed hit at 12 producers), so the unresolved fallback would take
  over the shocked episodes.
- **The replay from reset as the validator.** Rejected for production: it costs one episode per
  particle per decision cycle. It stays the test reference.

## Consequences

- Labels are a function of the public history: two envs with the same history but different
  hidden futures give byte-identical labels for the same fork seed.
- The acceptance rate does not measure residual ambiguity; it is binary by construction
  ("Floor"). Ambiguity is in the spread of the accepted particles, and "unresolved" means an
  infeasible plan. On the `shocks-i` worlds at 12 and 42 producers played to cycle 90, 16
  particles were accepted out of 16 draws.
- Known approximations; each needs two shocks on one producer and is negligible under
  `shocks-i`:
  - **Outages.** A run of blocked cycles is covered by a single slot, so a cover by two partial
    outages on the same producer is outside the proposal. A truncated slot that overlaps an
    assigned run is counted under both assignments. Two outages on one producer cannot happen
    at 12 producers under `shocks-i` and are rare at 42.
  - **Hits.** Each observed drop is one hit. Two hits on the same cell in the same cycle whose
    removals sum to the drop are outside the proposal, and so is a second hit on an observed
    cell that removes nothing.
- The posterior refuses explicit config events. They are not part of a generator posterior,
  and atlas worlds use generators only.
- Related: `establish-project-baseline` D4, ADR-0055 (seeded shock schedules), ADR-0057 (world
  space and demand patterns).
