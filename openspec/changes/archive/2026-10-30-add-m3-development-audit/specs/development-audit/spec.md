# Spec Delta

## Purpose

Defines how the M3 development audit is run and read: it diagnoses a baseline anomaly before it
reads a teacher, measures per-world headroom and mines a headroom atlas over a multi-dimension
world space, limits a teacher label to information the agents could observe, measures headroom
with a teacher on a mined sample, role attribution and deployed cost, and ends every failed probe
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
expired waiting for inputs. For every reservation that expired waiting for inputs, the trace
SHALL record whether the receiving producer's input stock at routing time could cover the order.

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

#### Scenario: Expired reservation with inputs short
- **WHEN** a reservation expires waiting for inputs and, when its order was routed, the receiving
  producer held fewer input units than the order needed
- **THEN** the trace counts its units as lost with inputs short, not with inputs available

### Requirement: Diagnostic routers stay outside the roster
Routers built only to diagnose or to measure headroom SHALL NOT be roster members, SHALL NOT be
added to the baseline dictionaries the gate and the signal check read, and SHALL NOT change any
frozen registry, roster or parameter file.

#### Scenario: Diagnostic router added
- **WHEN** the diagnostic routers are available
- **THEN** the frozen roster files keep their content hashes and the hindsight baseline of
  every gate report is unchanged

### Requirement: Per-world headroom
The audit SHALL measure on every screened world, on identical world-episodes with matching
digests, the atlas bar (best blind or hand-written arm), the best reach arm, an offline upper
bound and a noise floor. A world SHALL be called hard only when the best reach arm beats the bar
by more than the noise floor and in every episode; failure of every arm SHALL NOT make it hard.

#### Scenario: Better policy reachable
- **WHEN** on a world the best reach arm beats the atlas bar by more than the world's noise floor
- **THEN** the world is classed hard

#### Scenario: Everyone fails
- **WHEN** every arm delivers few units on a world and no reach arm beats the atlas bar by more
  than the noise floor
- **THEN** the world is not classed hard

#### Scenario: Gain within the noise
- **WHEN** the best reach arm beats the atlas bar by less than the noise floor
- **THEN** the world is not classed hard

#### Scenario: Gain not consistent across episodes
- **WHEN** the best reach arm beats the atlas bar on average by more than the noise floor but
  falls below it in one of the world's episodes
- **THEN** the world is not classed hard


### Requirement: A dying reference arm is collapse
In the confirm stage, a world whose fixed bar arm or fixed reach arm reaches system death in any
episode or replicate SHALL be classed collapse, never hard or open, with the dying arm recorded.

#### Scenario: Fixed bar arm dies
- **WHEN** in the confirm stage the bar arm fixed from the screen reaches system death at cycle 4
  in every run while the reach arm and other blind arms survive, so the gain exceeds the noise
  floor in every episode
- **THEN** the world is classed collapse, not hard, and its record names the dying arm


### Requirement: Noise floor is pooled
The noise floor SHALL be pre-registered and SHALL use a spread pooled over the episodes of many
worlds of the same stratum, not a standard error from the world's own one or two episodes. In the
confirm stage the pooled spread SHALL be a robust scale estimate, the normal-consistent median
absolute within-group residual, so that a minority of heavy-tailed worlds cannot set the floor of
their stratum; the pooled standard deviation of the same residuals SHALL be reported beside it.

#### Scenario: Two episodes that agree by chance
- **WHEN** a world's two episodes give nearly equal gains, so their own standard error is near
  zero
- **THEN** the noise floor still uses the pooled spread of its stratum

#### Scenario: One heavy-tailed world in the pool
- **WHEN** a stratum pools 40 worlds whose paired gains move by about 10 units between episodes
  and one world whose gain moves by 2,000 units
- **THEN** the pooled spread stays near that of the 40 worlds, the pooled standard deviation is
  reported beside it, and a world of the stratum with a consistent gain of 60 units is classed
  hard

#### Scenario: Recorded confirm reclassified
- **WHEN** a recorded confirm is classified again under revised rules
- **THEN** nothing is played, the confirm's recorded files and frozen atlas are not edited, and the
  derived classification records each world's previous class beside the new one and the hashes of
  the rows it read

### Requirement: Upper bound stays above every played arm
The offline achievable-units upper bounds SHALL be computed from the demand schedule the
environment built for the episode, SHALL be reported only as ceilings and never as a policy, and
SHALL each be at least the units of every arm played on that world-episode.

#### Scenario: Demand cap reported as such
- **WHEN** the upper bound's capacity term does not bind on a world
- **THEN** the bound is recorded as demand-bound and is not used alone to judge that world's room

#### Scenario: Bound below an arm
- **WHEN** an arm delivers more units than the upper bound on the same world-episode
- **THEN** the run is invalid and the result names the world, episode and arm

#### Scenario: Bound on a patterned world
- **WHEN** a world has a demand pattern or demand drift
- **THEN** its upper bound uses that episode's built demand schedule, including every pattern draw

### Requirement: Input-output bound
The atlas SHALL also compute an input-output upper bound per world-episode: one linear program over
every producer, capacity type and product that honours each producer's declared products, recipe
inputs, degradation with costed restorations, costed upgrades and the order time-to-live window. It
SHALL be at least the units of every arm played on that world-episode.

#### Scenario: Input-output bound below an arm
- **WHEN** an arm delivers more units than the input-output bound on the same world-episode
- **THEN** the run is invalid and the result names the world, episode and arm

#### Scenario: Upgrades are paid for
- **WHEN** a producer can upgrade its capacity
- **THEN** the input-output bound lets it complete at most one upgrade per cycle, each consuming one
  unit of the upgrade product made with capacity, instead of adding capacity for free

#### Scenario: Capacity of a producer that does not declare the product
- **WHEN** a producer owns a capacity type a product needs but does not declare that product
- **THEN** the input-output bound does not count that producer's capacity toward the product

### Requirement: Room is read against base capacity
Each world's room SHALL be read against a ceiling that is the smallest of the upper bounds and a
base-capacity ceiling (base capacity at its initial maximum, no upgrades), and never below the best
played arm's units. Where the input-output program was solved, the base-capacity ceiling SHALL be
that program without upgrades. Every world-episode on which an arm exceeds the base-capacity ceiling SHALL be
counted and reported.

#### Scenario: Demand beyond base capacity
- **WHEN** a world's scheduled demand far exceeds what its base capacity can make and the atlas bar
  reaches 0.97 of the base-capacity ceiling
- **THEN** the world is classed saturated, not open

#### Scenario: Base-capacity ceiling from the input-output program
- **WHEN** the input-output program was solved for a world-episode
- **THEN** its base-capacity ceiling is the program's value without upgrades, which is never above
  the base-capacity bound that pools capacity per type

#### Scenario: Arm above the base-capacity ceiling
- **WHEN** an arm delivers more units than the base-capacity ceiling on a world-episode
- **THEN** the ceiling for that world is at least that arm's units and the event is counted

### Requirement: Saturated worlds are filtered
A world whose atlas bar fills at least the registry's saturation ceiling of consumer orders, or
reaches at least 0.97 of the world's ceiling, SHALL be classed saturated and SHALL NOT be selected
into the atlas.

#### Scenario: Near-full world
- **WHEN** the atlas bar fills 0.98 of consumer orders on a world
- **THEN** the world is classed saturated and is not selected

### Requirement: Atlas sampling covers the world space
The atlas screen SHALL sample worlds space-fillingly over every world-space axis at each size,
with categorical axis levels balanced, and SHALL record each world's measured coordinates and
class.

#### Scenario: Balanced categorical levels
- **WHEN** the atlas screen samples 1,024 worlds at one size over a categorical axis with 7 levels
- **THEN** each level holds 146 or 147 worlds

#### Scenario: Screen record
- **WHEN** the atlas screen finishes
- **THEN** every screened world has its levers, seed, measured coordinates, arm units, bound,
  noise floor and class recorded

### Requirement: Atlas selection and split
The atlas SHALL select worlds from the hard and open classes by a pre-registered deterministic rule
that favours headroom and coverage of the axis bins, SHALL confirm each selected world on episode
and replicate seeds not used for its selection, and SHALL record a train, validation and held-out
split by world before the confirm stage. No choice made in the audit SHALL read the held-out worlds'
results.

#### Scenario: Uncovered bin
- **WHEN** an axis bin holds hard or open candidates but no selected world
- **THEN** the selection adds the best-scored candidate of that bin before filling by score

#### Scenario: Confirm on fresh seeds
- **WHEN** a selected world is confirmed
- **THEN** its confirm episodes and replicates use seeds different from those of its screen run,
  and a change of class is reported

#### Scenario: Held-out sealed
- **WHEN** the teacher sample is drawn from the atlas
- **THEN** it contains train-split worlds only

### Requirement: Atlas evidence does not change the gate
Atlas worlds SHALL NOT enter a gate registry, roster or report, and the atlas SHALL NOT edit any
frozen gate file. Adding atlas worlds or families to a gate registry SHALL be a user decision.

#### Scenario: Hard region found
- **WHEN** the atlas finds a region of hard worlds that no gate family covers
- **THEN** the result records the evidence and the gate registry, roster and parameter files are
  unchanged

### Requirement: Teacher labels depend only on observable information
A teacher's forks SHALL keep the realised observable history and state, and SHALL replace
every hidden part of the future (active shocks' remaining durations, future shock onsets,
targets and magnitudes, future demand including demand-pattern draws) with posterior draws given
that history and the known generators. A draw SHALL be kept only if it reproduces every observed
consequence and implies none that was not observed.

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
SHALL label the decision "unresolved" and act as its floor policy.

#### Scenario: Posterior too narrow
- **WHEN** fewer accepted posterior samples than the pre-registered minimum are found within
  the draw budget
- **THEN** the decision is labelled unresolved, the action equals the floor policy's, and the
  acceptance rate is recorded

### Requirement: Forked demand keeps continuity
On a world with demand drift, the demand of a fork SHALL continue from the current cycle's
consumer orders under the same drift bound, so that at most floor(j·N) orders change between
consecutive cycles of the fork. On a world with a demand pattern whose future draws do not depend
on hidden past draws, a fork SHALL resample the future from the pattern's generator. A world
whose pattern carries hidden state the fork rule does not cover SHALL NOT be taught.

#### Scenario: Drifting world forked
- **WHEN** a fork resamples future demand on a world with drift parameter j and N drifting
  consumers
- **THEN** its first forked cycle differs from the current cycle's orders in at most
  floor(j·N) orders, and so does every later transition

#### Scenario: Seasonal world forked
- **WHEN** a fork resamples future demand on a world with a seasonal pattern
- **THEN** every forked cycle's demand follows the world's seasonal envelope for that cycle

#### Scenario: Moving-hit world offered to the teacher
- **WHEN** the teacher sample is drawn and a candidate world has the moving-hit or regime-switch
  pattern
- **THEN** that world is not in the sample

### Requirement: Clairvoyant bound is a reference only
A clairvoyant teacher that rolls out the true future SHALL be reported only as an upper
reference for headroom. It SHALL NOT be a label source, a training target or a promotion
candidate.

#### Scenario: Bound in a report
- **WHEN** the teacher audit reports the clairvoyant arm
- **THEN** it is listed separately from the teacher arms and marked as using hidden
  information

### Requirement: Expensive teacher only on a mined sample
The search teacher SHALL run only on a pre-registered sample of atlas worlds, sized by a timing
probe, with a hard compute stop. It SHALL NOT run on a world above 42 producers, and a run above
its stop SHALL NOT launch without the user's approval.

#### Scenario: Probe above the stop
- **WHEN** the teacher probe extrapolates the sample above its hard stop after the pre-registered
  reduction order
- **THEN** the teacher run does not launch and the implementer reports to the supervisor

### Requirement: Teacher audit reading
The teacher audit SHALL compute, per sampled world, the teacher's units against the atlas bar,
the best reach arm and the upper bound, on identical world-episodes with matching demand and shock
digests. It SHALL report the per-class reading, worlds below the bar, deaths and the unresolved
fraction. It SHALL NOT produce an acceptance verdict.

#### Scenario: Teacher finds headroom on open worlds
- **WHEN** the teacher beats the atlas bar by more than the noise floor on at least half of the
  sampled open worlds
- **THEN** the audit records that search reaches headroom the cheap arms missed there, and no
  candidate is accepted by it

#### Scenario: Digest mismatch in the audit
- **WHEN** a teacher episode's demand or shock digest differs from a bar arm's for the same
  world and episode seed
- **THEN** the audit fails, naming the world, episode and arm

### Requirement: Role attribution decides the first trained role
The role attribution SHALL measure, on the shocked worlds of the teacher's mined sample, how much
of the gap between the portable stack and the teacher-orderer reference each single-role swap
closes. The role whose swap closes more SHALL be the role M4 trains first. A tie within the
pre-registered band SHALL go to the user.

#### Scenario: Orderer swap closes more
- **WHEN** swapping in the teacher orderer closes 0.6 of the gap and swapping in the
  reference producer closes 0.2
- **THEN** the audit records the orderer as the role M4 trains first

#### Scenario: Closures within the band
- **WHEN** the two closures differ by less than the pre-registered band
- **THEN** the audit records no automatic choice and the decision goes to the user

### Requirement: Splitting separated from the receiver rule
When the trained orderer trails a whole-order rotation on the role-attribution worlds, the audit
SHALL decide between splitting and the receiver rule with a matched contrast that changes one
factor of the same trained scores while holding the other at its trained level. A factor SHALL
count only when its gain clears the one-episode noise floor and recovers most of the gap on
most worlds.

#### Scenario: Whole routing recovers the gap
- **WHEN** routing the trained orderer's orders whole to its argmax receiver recovers at least
  two thirds of its gap to `round_robin`, beyond the noise floor, on two of the three worlds
- **THEN** the audit records the whole-routing rule as recovering the gap (splitting under the
  scorer's rule is the cost, not shown to be the mechanism), and M4's orderer must be able to
  route an order whole

#### Scenario: Forced-whole variant loses its rotation
- **WHEN** the forced-whole variant does not recover the gap but the trained split profile on
  `round_robin`'s receivers does
- **THEN** the audit records the receiver rule (spread across concurrent orders and / or
  rotation over time) as the cause, not that splitting is harmless

#### Scenario: Only both changes together recover the gap
- **WHEN** on two of the worlds the gap clears the floor and neither single change recovers two
  thirds of it
- **THEN** the audit records that whole routing and the rotation's receivers are needed
  together, and reports the interaction

#### Scenario: Worlds disagree
- **WHEN** no reading holds on two of the three worlds
- **THEN** the audit records no cause and the worlds are diagnosed further before M4 designs
  orderer training

#### Scenario: Rotation contrast holds the split
- **WHEN** the rotation contrast routes an order
- **THEN** its parts have the sizes the trained orderer would route, the largest goes to the
  producer `round_robin` picks, and a whole order is routed exactly as `round_robin` routes it

### Requirement: Deployed-path cost on independent axes
The cost audit SHALL time each stage of the deployed portable path per decision. It SHALL
vary total producers at fixed fanout, fanout at fixed total producers, and consumers per producer
at fixed total producers and fanout, all within the 12 → 42 producer target, and SHALL report a
log-log slope per stage and axis. It SHALL NOT time any economy above 42 producers.

#### Scenario: Superlinear stage
- **WHEN** a stage's per-decision time grows with a slope above 1.1 on the eligible-producer
  axis or the consumer axis
- **THEN** the audit names that stage as violating the bounded per-decision cost rule

#### Scenario: Larger economy requested
- **WHEN** the cost audit is asked to time an economy above 42 producers
- **THEN** it refuses, naming the size

### Requirement: A failed probe ends in a diagnosis
When the atlas, the teacher audit, the role attribution or the cost audit falls short of its
reading, the audit SHALL record the diagnosed cause, with evidence, before any next strategy is
proposed. No margin, roster or registry SHALL be changed in reaction to the result of one of
these probes.

#### Scenario: Teacher does not beat the bar on open worlds
- **WHEN** the teacher beats the atlas bar on fewer than half of the sampled open worlds
- **THEN** the result records a diagnosed cause (bound-limited, fork-limited, or floor or
  candidate-limited, from the clairvoyant arm and the unresolved rate) and the acceptance
  margins are unchanged

#### Scenario: Atlas finds no hard worlds
- **WHEN** the atlas confirm stage classes no selected world hard
- **THEN** the result records why per region (saturated, flat or open, with the bound and the
  noise floor) and the gate files are unchanged
