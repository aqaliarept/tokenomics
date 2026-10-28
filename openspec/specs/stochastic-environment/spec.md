# stochastic-environment Specification

## Purpose

Defines how randomness enters the economy — stochastic demand and supply shocks — while
keeping the simulation reproducible and every policy comparison exactly paired on the same
sequence of events.

## Requirements

### Requirement: Seeded event schedule
All stochastic demand and shock events for an episode SHALL be generated, before the
episode starts, as a schedule determined only by the world configuration and the episode
seed. Each shock kind and demand drift SHALL draw from its own stream derived from the
episode seed, so that enabling one kind changes no other kind's schedule and no existing
demand draw.

#### Scenario: Same seed, same events
- **WHEN** two episodes run on the same world with the same episode seed
- **THEN** they receive identical demand and shock schedules

#### Scenario: Adding a kind leaves earlier schedules unchanged
- **WHEN** a world that generates capacity hits also enables producer outages, with the
  same episode seed
- **THEN** its capacity-hit schedule is byte-identical to the one generated without outages

#### Scenario: Demand draws preserved
- **WHEN** shocks are enabled on a world with seeded random demand and no drift
- **THEN** every consumer's demand schedule is identical to the run without shocks for the
  same episode seed

#### Scenario: Longer horizon extends the schedule
- **WHEN** the same world and episode seed run with horizons 10 and 20
- **THEN** the shock events and drifting demand of cycles 1 to 10 are identical in both

### Requirement: Events are independent of agent actions
No scheduled event's timing, target, or magnitude SHALL depend on any agent action or on
any state that agent actions influence.

#### Scenario: Two policies, one schedule
- **WHEN** two different policy stacks run the same world and episode seed
- **THEN** both face exactly the same demand and shock events, so any outcome difference
  is caused by their decisions

### Requirement: Domain stays RNG-free
The domain layer SHALL contain no random number generation; it SHALL only replay the
injected event schedule. Schedule generation SHALL live in the env layer.

#### Scenario: Domain run from a schedule
- **WHEN** the domain is given a world configuration and an event schedule
- **THEN** its run is fully deterministic with no seed of its own

### Requirement: Shocks are unannounced
Agents SHALL NOT observe a shock before it takes effect; they SHALL observe only its
consequences in the existing state (capacity, availability, orders, inventory).

#### Scenario: Observation before a scheduled shock
- **WHEN** a capacity hit is scheduled for cycle t
- **THEN** no agent observation at cycles before t differs from the same episode with that
  shock removed

### Requirement: Calm regime is preserved
With stochastic demand and shocks disabled, episodes SHALL be identical to the
pre-existing deterministic behaviour for the same configuration.

#### Scenario: Disabled stochasticity
- **WHEN** a configuration enables no stochastic demand and no shocks
- **THEN** episode statistics match the existing deterministic run exactly

### Requirement: Smooth demand drift
Stochastic demand SHALL support drifting schedules: between consecutive cycles, at most
floor(j * N) of the N consumer orders scheduled for the cycle change, where j in [0, 1] is
a configuration parameter. An order changes if its product or its integer quantity differs
from the same consumer's order in the previous cycle. Per-cycle independent sampling within
bounds SHALL remain available.

#### Scenario: Drift bound respected
- **WHEN** a drifting schedule has N = 10 consumer orders per cycle and j = 0.2
- **THEN** between any two consecutive cycles at most 2 of those orders change product or
  quantity

#### Scenario: Product switch is a whole-order change
- **WHEN** a consumer's order changes from product P quantity 1 to product Q quantity 1
- **THEN** that counts as one changed order, and is permitted only if the floor(j * N)
  budget for that cycle is not already spent

#### Scenario: Budget rounds down to zero
- **WHEN** N = 3 and j = 0.2, so floor(j * N) = 0
- **THEN** no order changes between consecutive cycles and demand stays constant

### Requirement: Shock kinds
The shock model SHALL define exactly five shock kinds, each with a target, start cycle,
duration, and magnitude: capacity hit, producer outage, supply-link cut, demand surge, and
fund shock.

#### Scenario: Schedule lists a shock
- **WHEN** a shock schedule is inspected
- **THEN** every entry names one of the five kinds with its target, start cycle, duration,
  and magnitude

#### Scenario: Target fits the kind
- **WHEN** a configuration schedules a capacity hit on a capacity type its target producer
  does not own, or names a producer or consumer that does not exist
- **THEN** configuration validation fails with an error naming the entry and the bad target

#### Scenario: Shocks start after the first cycle
- **WHEN** a configuration schedules a shock with start cycle 1
- **THEN** configuration validation fails, so the state observed right after reset is never
  shocked

### Requirement: Capacity hit
A capacity hit SHALL reduce a target producer's current capacity of one or more capacity
types by a fraction of current capacity at the start cycle; max capacity is unchanged and
normal restoration applies afterward. The removed amount SHALL be floor(m * c) units for
magnitude m in (0, 1) and current capacity c, so a hit never empties a positive capacity.

#### Scenario: 40% hit
- **WHEN** a capacity hit of magnitude 0.4 targets producer P's capacity type T at cycle t
- **THEN** P's current capacity of T at cycle t is 60% of its pre-hit value and P can
  restore it by investment

#### Scenario: Removal rounds down
- **WHEN** a capacity hit of magnitude 0.4 targets a capacity of 7 units
- **THEN** 2 units are removed and 5 remain

#### Scenario: Hit after the last played cycle
- **WHEN** an episode ends at its horizon H and a capacity hit is scheduled for cycle H + 1
- **THEN** no episode statistic, including the end-of-episode capacity ratio, reflects that
  hit

### Requirement: Producer outage
A producer outage SHALL make a target producer unable to accept bids or complete
production for its duration, while its capacity keeps degrading; it resumes afterward.
Completing production includes assembling reserved production accepted before the outage.
The producer SHALL keep receiving deliveries of what it bought.

#### Scenario: Orders during an outage
- **WHEN** producer P is in an outage for cycles t to t+N-1
- **THEN** sub-orders routed to P in those cycles are not accepted, and P accepts bids
  again from cycle t+N

#### Scenario: In-flight production during an outage
- **WHEN** P accepted a composite bid at cycle t-1 that waits for inputs, the inputs
  arrive, and P is in an outage at cycle t
- **THEN** the bid is not completed at cycle t, and it keeps aging toward its reservation
  TTL as if its inputs were still missing

#### Scenario: Capacity degrades during an outage
- **WHEN** P is in an outage for N cycles
- **THEN** P's capacity after each of those cycles equals what normal degradation gives
  without production, and restoration P bought can still complete

### Requirement: Supply-link cut
A supply-link cut SHALL make a target producer unable to sell a target product to other
producing agents for its duration; consumer orders for that product from that producer are
unaffected.

#### Scenario: Composite input unavailable
- **WHEN** a link cut blocks producer Q from selling product P to producers for N cycles
- **THEN** producers needing P as an input must source it from another eligible producer
  during those cycles

### Requirement: Demand surge
A demand surge SHALL multiply a target consumer's ordered quantity by a magnitude factor
for its duration, without changing that consumer's token share of the emission.

#### Scenario: Surge with fixed tokens
- **WHEN** a surge of factor 3 targets consumer C for N cycles
- **THEN** C orders three times its scheduled quantity, backed by the same tokens as
  without the surge

### Requirement: Fund shock
A fund shock SHALL reduce the investment-fund share of the cycle emission by a magnitude
fraction for its duration; the consumer share is unchanged.

#### Scenario: Fund cut
- **WHEN** a fund shock of magnitude 0.5 is active
- **THEN** the investment fund receives half its normal share of that cycle's emission

### Requirement: Shock delivery order
Capacity hits and producer outages SHALL be implemented first; supply-link cuts, demand
surges, and fund shocks SHALL follow. Until a kind is implemented, a configuration that
schedules it SHALL be rejected at validation.

#### Scenario: Unimplemented kind scheduled
- **WHEN** a configuration schedules a fund shock before fund shocks are implemented
- **THEN** configuration validation fails with an error naming the unsupported shock kind

#### Scenario: Unimplemented kind in a generator
- **WHEN** a configuration enables seeded generation of supply-link cuts or demand surges
  before those kinds are implemented
- **THEN** configuration validation fails with an error naming the unsupported shock kind

### Requirement: Resilience is reported
Every evaluation in a shocked regime SHALL report, per shock, the dip in fulfilled units
relative to the matched calm episode and the cycles until fulfilled units recover to
within a fixed tolerance of that calm baseline. The matched episode SHALL be the same world
and episode seed with the identical demand schedule and no shocks.

#### Scenario: Recovery measured
- **WHEN** a shocked gate episode completes
- **THEN** the report lists each shock's unit dip and recovery time against the same world
  and episode seed run without shocks

#### Scenario: Recovery never reached
- **WHEN** fulfilled units have not returned within tolerance of the matched episode by the
  end of the episode
- **THEN** that shock is reported as unrecovered, with its dip summed to the last played
  cycle

#### Scenario: Overlapping shocks flagged
- **WHEN** another shock is active between a shock's start and its recovery
- **THEN** that shock's report row is flagged as overlapped

### Requirement: Deterministic cyclic demand
A consumer SHALL be configurable with an explicit finite schedule of (product, quantity)
entries that is replayed cyclically, one entry per cycle, identically in every episode
regardless of the episode seed. A consumer without this setting SHALL behave exactly as
before.

#### Scenario: Schedule repeats
- **WHEN** a consumer is configured with the schedule [(P, 1), (Q, 2)] for a 5-cycle episode
- **THEN** it orders P×1, Q×2, P×1, Q×2, P×1

#### Scenario: Episode seed has no effect
- **WHEN** a world whose consumers all use explicit schedules runs with episode seeds 0 and 9
- **THEN** both episodes receive the identical demand

#### Scenario: Legacy fixed demand untouched
- **WHEN** a configuration sets no explicit schedule on any consumer
- **THEN** every episode statistic matches the run before the setting existed

### Requirement: Shocks keep the agent contract
Enabling shocks or demand drift SHALL NOT change any observation or action space, so that
checkpoints trained without shocks load and act on shocked worlds unchanged.

#### Scenario: Spaces with and without shocks
- **WHEN** the same world is built with and without its shock schedule and drift
- **THEN** every agent's observation and action space is identical in both
