# Spec Delta

## Purpose

Defines how the M3 development audit is run and read: it diagnoses a baseline anomaly before it
reads a teacher against the hindsight baseline, limits a teacher label to information the agents
could observe, measures headroom, role attribution and deployed cost, and ends every failed probe
in a recorded diagnosis.

## ADDED Requirements

### Requirement: Baseline anomaly diagnosed before the teacher audit
The audit SHALL diagnose why the `random` ordering baseline sets the hindsight baseline on most
v2 development world-episodes before the teacher audit runs. Each competing explanation SHALL
get its own discriminating check, and the outcome SHALL be recorded in the experiment's result
and in the diagnostics registry.

#### Scenario: Teacher audit requested before the diagnosis
- **WHEN** the teacher audit is about to launch and the random-routing diagnosis has no
  recorded result
- **THEN** the teacher audit does not launch

#### Scenario: Every explanation has a check
- **WHEN** the random-routing diagnosis is recorded
- **THEN** it reports a result for packing under atomic acceptance, temporal variation versus
  randomness, rule defects in the hand-written routers, the investment-fund channel, the
  scoring of random replicates and the penalty of splitting an order under whole-order
  completion, each with the check that discriminated it

### Requirement: Rejection reasons are traceable
A diagnostic trace SHALL classify every rejected bid as rejected by an outage, as not fitting
the producer's capacity available at the start of its auction, or as displaced by
higher-priced bids. It SHALL classify every failed reservation as dropped on re-validation or
expired waiting for inputs.

#### Scenario: Bid displaced by a higher-priced bid
- **WHEN** a bid fits the capacity a producer has at the start of its auction but not the
  capacity left after higher-priced bids were accepted
- **THEN** the trace records it as displaced, not as not fitting

#### Scenario: Bid too large for the producer
- **WHEN** a bid needs more of some capacity type than the producer has at the start of its
  auction
- **THEN** the trace records it as not fitting

#### Scenario: Tracing changes no statistic
- **WHEN** an episode is run with and without the trace
- **THEN** every episode statistic and every action is identical in both runs

### Requirement: Diagnostic routers stay outside the roster
Routers built only to diagnose the baseline anomaly SHALL NOT be roster members, SHALL NOT be
added to the baseline dictionaries the gate and the signal check read, and SHALL NOT change any
frozen registry, roster or parameter file.

#### Scenario: Diagnostic router added
- **WHEN** the diagnostic routers are available
- **THEN** the frozen roster files keep their content hashes and the hindsight baseline of
  every gate report is unchanged

### Requirement: Teacher labels depend only on observable information
A teacher's forks SHALL keep the realised observable history and state, and SHALL replace
every hidden part of the future (active shocks' remaining durations, future shock onsets,
targets and magnitudes, future demand) with posterior draws given that history and the known
generators. A draw SHALL be kept only if it reproduces every observed consequence and implies
none that was not observed.

#### Scenario: Two envs differ only in hidden state
- **WHEN** two environments have the same observable history and state but different true
  remaining outage durations and future shock schedules
- **THEN** the teacher gives byte-identical labels in both for the same fork seed

#### Scenario: Ambiguous consequence
- **WHEN** a producer rejected every bid in a cycle and both an outage and a lack of capacity
  reproduce that observation
- **THEN** posterior draws of both kinds are kept, and no draw that contradicts any other
  observed consequence is kept

#### Scenario: Hidden rejection reasons scrambled
- **WHEN** the rejection and failure reasons recorded for the diagnostic trace are scrambled or
  removed in an environment
- **THEN** the teacher's labels are byte-identical to those of the unchanged environment

#### Scenario: Hidden fields stripped
- **WHEN** a fork is prepared for a rollout
- **THEN** it carries no outage end cycle and no pending shock event from the true schedule

### Requirement: Unresolved labels fall back
The teacher SHALL log the posterior acceptance rate of every decision cycle. When it cannot
draw the pre-registered number of posterior samples within the pre-registered draw budget, it
SHALL label the decision "unresolved" and act as `blend_0.5`.

#### Scenario: Posterior too narrow
- **WHEN** fewer accepted posterior samples than the pre-registered minimum are found within
  the draw budget
- **THEN** the decision is labelled unresolved, the action equals `blend_0.5`'s, and the
  acceptance rate is recorded

### Requirement: Forked demand keeps drift continuity
On a world with demand drift, the demand of a fork SHALL continue from the current cycle's
consumer orders under the same drift bound, so that at most floor(j·N) orders change between
consecutive cycles of the fork.

#### Scenario: Drifting world forked
- **WHEN** a fork resamples future demand on a world with drift parameter j and N drifting
  consumers
- **THEN** its first forked cycle differs from the current cycle's orders in at most
  floor(j·N) orders, and so does every later transition

### Requirement: Clairvoyant bound is a reference only
A clairvoyant teacher that rolls out the true future SHALL be reported only as an upper
reference for headroom. It SHALL NOT be a label source, a training target or a promotion
candidate.

#### Scenario: Bound in a report
- **WHEN** the teacher audit reports the clairvoyant arm
- **THEN** it is listed separately from the teacher arms and marked as using hidden
  information

### Requirement: Teacher audit reading
The teacher audit SHALL compute each arm's per-world ratio over the hindsight baseline, using
the frozen roster on identical world-episodes with matching demand and shock digests. It SHALL
report the regime means, the worlds below 1.0 and deaths. It SHALL NOT produce an acceptance
verdict.

#### Scenario: Shocked headroom read
- **WHEN** the depth-3 teacher's mean shocked ratio is at least 1.10 and no shocked world is
  below 1.0
- **THEN** the audit records that a teacher reaches the shocked margin on development worlds,
  and no candidate is accepted by it

#### Scenario: Digest mismatch in the audit
- **WHEN** a teacher episode's demand or shock digest differs from the roster's for the same
  world and episode seed
- **THEN** the audit fails, naming the world, episode and arm

### Requirement: Role attribution decides the first trained role
The role attribution SHALL measure, on shocked development worlds, how much of the gap between
the portable stack and the teacher-orderer reference each single-role swap closes. The role
whose swap closes more SHALL be the role M4 trains first. A tie within the pre-registered band
SHALL go to the user.

#### Scenario: Orderer swap closes more
- **WHEN** swapping in the teacher orderer closes 0.6 of the gap and swapping in the
  reference producer closes 0.2
- **THEN** the audit records the orderer as the role M4 trains first

#### Scenario: Closures within the band
- **WHEN** the two closures differ by less than the pre-registered band
- **THEN** the audit records no automatic choice and the decision goes to the user

### Requirement: Deployed-path cost on independent axes
The cost audit SHALL time each stage of the deployed portable path per decision. It SHALL
vary total producers at fixed fanout and fanout at fixed total producers, both within the
12 → 42 producer target, and SHALL report a log-log slope per stage and axis.

#### Scenario: Superlinear stage
- **WHEN** a stage's per-decision time grows with a slope above 1.1 on the eligible-producer
  axis
- **THEN** the audit names that stage as violating the bounded per-decision cost rule

#### Scenario: Larger economies not run
- **WHEN** the cost audit runs in this phase without the user's approval of a larger point
- **THEN** it times no economy above 42 producers

#### Scenario: User-approved larger point
- **WHEN** the user approves one timing point above 42 producers
- **THEN** that point is reported as non-gating and does not change the 12 → 42 scale target

### Requirement: A failed probe ends in a diagnosis
When the teacher audit, the role attribution or the cost audit falls short of its reading, the
audit SHALL record the diagnosed cause, with evidence, before any next strategy is proposed.
No margin, roster or registry SHALL be changed in reaction to the result of one of these three
probes.

#### Scenario: Teacher below the shocked margin
- **WHEN** the depth-3 teacher's mean shocked ratio is below 1.10
- **THEN** the result records a diagnosed cause (world family, roster or fork) and the
  acceptance margins are unchanged
