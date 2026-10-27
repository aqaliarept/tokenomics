# Spec Delta

## MODIFIED Requirements

### Requirement: Frozen baseline roster
The hindsight baseline SHALL be computed from a named, versioned roster of baseline stacks. Roster v1
SHALL contain every hand-written producing baseline paired with every hand-written ordering
baseline, plus `blend_0.5`. Roster v2 SHALL also pair every producing baseline with a
shock-reactive ordering baseline that reacts only to observed consequences of shocks. Every
gate report SHALL name the roster version it used.

#### Scenario: blend_0.5 is a roster member
- **WHEN** roster v1 is loaded
- **THEN** it contains a stack whose ordering policy is `blend_0.5`

#### Scenario: Roster edited without a new version
- **WHEN** a roster file's member list no longer matches the content hash recorded for its
  version
- **THEN** the harness refuses to compute a hindsight baseline from it

#### Scenario: Shock-reactive baseline in roster v2
- **WHEN** the draft roster v2 is loaded, before pruning
- **THEN** it contains a stack whose ordering policy is the shock-reactive baseline for every
  producing baseline

#### Scenario: Shock-reactive baseline sees no schedule
- **WHEN** the shock-reactive baseline runs the same episode twice, once with the shock
  schedule readable and once with it hidden from every policy
- **THEN** its actions are identical in both runs

### Requirement: Identical demand across observation contracts
For a given world and episode seed, the candidate and every roster member SHALL face the
identical demand schedule and the identical shock schedule, even when they run under
different observation or ordering contracts. The harness SHALL verify this by comparing a
demand-schedule digest and a shock-schedule digest per episode.

#### Scenario: Digest mismatch
- **WHEN** a roster member's episode demand-schedule digest differs from the candidate's
  for the same world and episode seed
- **THEN** the gate run fails with an error naming the world, episode and member

#### Scenario: Shock digest mismatch
- **WHEN** a roster member's episode shock-schedule digest differs from the candidate's for
  the same world and episode seed
- **THEN** the gate run fails with an error naming the world, episode and member

#### Scenario: Twin shares the demand
- **WHEN** a shocked episode's shock-free twin is run
- **THEN** its demand-schedule digest equals the shocked episode's and its shock schedule
  is empty

### Requirement: Regimes
Every registry world SHALL declare one regime. A calm world SHALL have deterministic demand
and no shocks. A shocked world SHALL have seeded stochastic demand, supply shocks, or both.
At each economy size, the registry SHALL hold at least twice as many calm worlds as shocked
worlds. From registry v2 on, every shocked gate world SHALL carry supply shocks and demand
drift from a named, versioned shock profile.

#### Scenario: Stochastic demand is shocked
- **WHEN** a world uses seeded random demand and no supply shocks
- **THEN** it is registered and scored in the shocked regime

#### Scenario: Too few calm worlds
- **WHEN** a registry holds 10 shocked and 15 calm worlds at 42 producers
- **THEN** registry validation fails

#### Scenario: Calm world with shocks
- **WHEN** a registry entry declared calm configures shocks or demand drift
- **THEN** registry validation fails naming the entry

#### Scenario: Shocked v2 world without shocks
- **WHEN** a registry v2 shocked gate entry has no shock profile
- **THEN** registry validation fails naming the entry

### Requirement: Gate report contents
A gate report SHALL list, per world: candidate and hindsight baseline mean units, the world ratio,
deaths, and each hard side-effect check. Per regime it SHALL list the mean ratio and the
margin and no-losing-world verdicts. It SHALL also list the monitored side-effect metrics
and an overall verdict for each acceptance clause. On worlds with shocks it SHALL list the
candidate's per-shock resilience rows, and it SHALL list the contract-failure count.

#### Scenario: One clause fails
- **WHEN** every margin passes but one episode violates the starvation constraint
- **THEN** the report's overall verdict is reject and it names the failing clause, world and
  episode

#### Scenario: Resilience rows in a report
- **WHEN** a report covers a shocked world whose episodes carry shocks
- **THEN** it lists, for every candidate episode and every shock, the shock's kind, target,
  start, unit dip, recovery cycles and overlap flag, computed against the shock-free twin

#### Scenario: Resilience is not an acceptance clause
- **WHEN** a candidate recovers slowly from every shock but passes every acceptance clause
- **THEN** the overall verdict is unchanged by the resilience rows

## ADDED Requirements

### Requirement: Versioned registry for shocked worlds
Real shocks SHALL enter the gate only through new registry, roster and side-effect
parameter versions. Frozen v1 files SHALL NOT change. Registry v2 SHALL keep every v1 world,
world seed and episode seed. A report on registry v2 SHALL use a roster and parameter
version frozen by a baseline-only screen of registry v2.

#### Scenario: v1 untouched
- **WHEN** the shock model and registry v2 land
- **THEN** the frozen v1 registry, roster and parameter files have their recorded content
  hashes and every v1 world replays exactly as before

#### Scenario: Calm worlds carried over
- **WHEN** registry v2 is generated
- **THEN** each of its calm entries has the same world seed, episode seeds and replayed
  configuration as the v1 entry with the same id

#### Scenario: Mixed versions refused
- **WHEN** a real report is requested on registry v2 with roster v1 or parameter version 1
- **THEN** the harness refuses to run and names the mismatched version

### Requirement: Contract failures are counted
A gate or contract run SHALL count as contract failures: a checkpoint that does not load, an
episode that raises, an observation or action that does not fit its space, an episode that
ends neither by termination nor by truncation, and a digest mismatch between paired runs.
Death and low value SHALL NOT count as contract failures.

#### Scenario: Episode raises
- **WHEN** one candidate episode raises an exception during a run
- **THEN** the run records one contract failure naming the world, episode and error, and
  finishes the remaining jobs

#### Scenario: Action outside its space
- **WHEN** a policy returns an action of the right shape that holds NaN or exceeds the action
  bounds
- **THEN** the run records a contract failure for that job, even though the env would clip
  or decode the action without raising

#### Scenario: Episode without a terminal outcome
- **WHEN** an episode's step loop stops before the episode is terminated or truncated
- **THEN** the run records a contract failure for that job

#### Scenario: Failed run stays invalid
- **WHEN** a run with a recorded contract failure is resumed and the failed job then
  succeeds, or its verdict is recomputed from its rows
- **THEN** the failure record is kept and the run's status is invalid, never an acceptance
  verdict built from partial rows

#### Scenario: Death is not a contract failure
- **WHEN** a candidate episode ends in system death
- **THEN** the death is reported under the zero-death clause and the contract-failure count
  is unchanged
