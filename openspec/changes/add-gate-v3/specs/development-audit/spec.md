## MODIFIED Requirements

### Requirement: Diagnostic routers stay outside the roster
Routers built only to diagnose or to measure headroom SHALL NOT be roster members, SHALL NOT be
added to the baseline dictionaries the gate and the signal check read, and SHALL NOT change any
frozen registry, roster or parameter file. A diagnostic router MAY leave this rule only when the
user decides to promote it: it then enters the baseline dictionaries under its own name with
unchanged actions, joins only a roster version drafted after the promotion, and stays available to
the audits as a diagnostic router.

#### Scenario: Diagnostic router added
- **WHEN** the diagnostic routers are available
- **THEN** the frozen roster files keep their content hashes and the hindsight baseline of
  every gate report is unchanged

#### Scenario: Router promoted for roster v3
- **WHEN** `round_robin` is promoted on the user's decision of 2026-10-30
- **THEN** it appears in the gate's ordering baselines and in the roster v3 draft, the frozen v1
  and v2 rosters keep their content hashes, and the routing audits still run it as a diagnostic
  router with identical actions

#### Scenario: Unpromoted router stays out
- **WHEN** roster v3 is drafted
- **THEN** no diagnostic router other than the five promoted ones appears in it

### Requirement: Atlas evidence does not change the gate
Atlas worlds SHALL NOT enter a gate registry, roster or report, and the atlas SHALL NOT edit any
frozen gate file. Adding atlas worlds or families to a gate registry SHALL be a user decision.
When the user decides that a gate registry covers an atlas region, the registry SHALL draw fresh
worlds from that region's lever distribution, never reuse an atlas world of any split, and SHALL
record the atlas evidence that defined the region.

#### Scenario: Hard region found
- **WHEN** the atlas finds a region of hard worlds that no gate family covers
- **THEN** the result records the evidence and the gate registry, roster and parameter files are
  unchanged

#### Scenario: Region added by user decision
- **WHEN** registry v3 adds the 42-producer fourteen-maker region on the user's decision
- **THEN** its worlds are fresh draws, none of them is an atlas world, and the registry records the
  atlas version and worlds whose evidence defined the region

## ADDED Requirements

### Requirement: The 42-producer teacher read is a reference only
The 42-producer teacher read SHALL run the search teacher and the role swap on a pre-registered
sample of 42-producer atlas train-split worlds with fourteen makers per product, excluding collapse,
moving-hit and regime-switch worlds. Its rows SHALL NOT be a label source, a training target, a
gate input or a promotion candidate.

#### Scenario: Teacher read on a validation world
- **WHEN** the 42-producer read is offered a world of the atlas validation or held-out split
- **THEN** it refuses the world

### Requirement: The 42-producer teacher is read as a lower bound
The read SHALL report the teacher against the atlas bar, the best reach arm, `round_robin`,
`makeable_weighted_whole` and the trained orderer, with the atlas's analytic ceilings beside it. It
SHALL NOT run the clairvoyant arm, and SHALL name the teacher's units a lower bound on what
lookahead reaches, not a ceiling.

#### Scenario: Ceiling wording
- **WHEN** the read reports the teacher's units on a world
- **THEN** it lists the atlas's analytic ceilings for the same world-episode as the upper
  references and marks the teacher's units as a lower bound

### Requirement: The role swap at 42 does not change M4's role
A producer closure at least as large as the orderer closure at 42 producers, or a tie within the
band, SHALL be recorded and go to the user without changing the role M4 trains.

#### Scenario: Producer swap closes more at 42
- **WHEN** at 42 producers the producer closure is 0.5 and the orderer closure 0.3
- **THEN** the record states it, the decision goes to the user, and M4's training role is unchanged

### Requirement: The 42-producer teacher read follows a pre-registered reduction ladder
The read SHALL be sized by a local timing probe before any pod run. If the probe's estimate
exceeds the run's stop, it SHALL be reduced only by the pre-registered steps, in order: the hard
worlds only; at most four candidates per stream; at most eight searched streams per cycle; no role
swap; two worlds. If the estimate still exceeds the stop, the read SHALL NOT launch and the result
SHALL go to the user. The record SHALL name every applied step, and every reading SHALL carry it.

#### Scenario: Probe above the stop after the ladder
- **WHEN** the probe's estimate after the last step still exceeds the stop
- **THEN** no pod is started and the sizing record goes to the supervisor and the user

#### Scenario: Candidate cap applied
- **WHEN** the ladder reaches the candidate cap
- **THEN** each searched stream's candidates are the trained orderer's row and the whole order to
  each of the three eligible producers with the most makeable units, and every reading names the
  cap
