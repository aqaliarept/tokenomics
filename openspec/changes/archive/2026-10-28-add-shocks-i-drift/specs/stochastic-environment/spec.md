# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Shocks keep the agent contract
Enabling shocks or demand drift SHALL NOT change any observation or action space, so that
checkpoints trained without shocks load and act on shocked worlds unchanged.

#### Scenario: Spaces with and without shocks
- **WHEN** the same world is built with and without its shock schedule and drift
- **THEN** every agent's observation and action space is identical in both
