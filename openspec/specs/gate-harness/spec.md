# gate-harness Specification

## Purpose

Defines how a candidate policy stack is measured against the acceptance contract: which
baselines form the hindsight baseline, how the hindsight baseline and ratios are computed, which worlds may be
used for what, and what a gate report must contain so its verdict is honest and
reproducible.

## Requirements

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

### Requirement: Roster pruning uses development evidence only
A stack SHALL be dropped from a roster only before that roster version is frozen, and only
by a recorded rule applied to baseline-only runs on development worlds. Candidate results
SHALL NOT influence roster membership.

#### Scenario: Never-winning stack dropped
- **WHEN** a stack is the per-episode best member, and so sets the hindsight baseline, on no
  development episode at either economy size
- **THEN** it may be dropped before freezing, and the roster file records the rule and the
  development runs it was applied to

### Requirement: Random-baseline replicates
A roster member with a random policy SHALL run R replicates per episode, with R frozen in
the roster version. Each replicate's action seed SHALL be derived by a cryptographic hash of
(world id, episode seed, replicate index) and SHALL NOT equal the episode's event seed. The
member's per-episode value SHALL be the mean over its replicates.

#### Scenario: Action seed is independent of the event seed
- **WHEN** a random stack runs replicate 0 of episode seed 7 on world W
- **THEN** its action RNG is seeded from the hash of (W, 7, 0), not from 7

#### Scenario: Replicates are averaged, not maximised
- **WHEN** a random stack's three replicates deliver 4, 10 and 7 units on one episode
- **THEN** its per-episode value for the hindsight baseline is 7

### Requirement: Hindsight-baseline arithmetic
For each world, the per-episode hindsight baseline SHALL be the maximum over roster members of their
per-episode units. The world hindsight baseline SHALL be the mean of per-episode hindsight baselines over the
world's episodes. The world ratio SHALL be candidate mean units over world hindsight baseline on the
same episodes. A regime ratio SHALL be the equal-weight mean of its worlds' ratios.

#### Scenario: Per-episode max differs from max of means
- **WHEN** on two episodes stack A delivers 10 and 4 units, stack B delivers 6 and 8, and
  the candidate delivers 9 and 9
- **THEN** the world hindsight baseline is 9 (mean of 10 and 8), not 7, and the world ratio is 1.0

#### Scenario: Worlds weigh equally
- **WHEN** a regime has two worlds with ratios 1.2 and 1.0 and different episode counts
- **THEN** the regime ratio is 1.1

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

### Requirement: World registry
Gate and development worlds SHALL be listed in a versioned registry. Each entry SHALL give
a world id, family, economy size, regime, a replayable generator reference, a world seed,
its episode seeds, and a split of `development` or `sealed`. Development and sealed worlds
SHALL use disjoint world seeds.

#### Scenario: Overlapping seeds rejected
- **WHEN** a registry lists the same family and world seed under both splits
- **THEN** loading the registry fails validation

### Requirement: Sealed worlds need a pre-registration
The harness SHALL refuse to evaluate sealed worlds unless given a frozen pre-registration
manifest binding the candidate checkpoint hashes and the roster, registry and side-effect
parameter versions. Every sealed evaluation SHALL be appended to an access log.

#### Scenario: Sealed run without a manifest
- **WHEN** a sealed world is requested without a pre-registration manifest
- **THEN** the harness exits with an error before running any episode

#### Scenario: Checkpoint differs from the manifest
- **WHEN** the candidate checkpoint's hash differs from the one in the manifest
- **THEN** the harness refuses the run and records nothing in the gate report

### Requirement: Zero-baseline screening
Before any candidate is evaluated on a registry version, the roster SHALL be run alone on
its worlds. A world whose hindsight baseline mean is zero SHALL be replaced by the next unused world
from its family's reserve list, in a fixed order. The outcome SHALL be recorded in the
registry version.

#### Scenario: Replacement is candidate-independent
- **WHEN** a proposed world has a zero hindsight baseline mean
- **THEN** it is replaced by the family's next reserve world before any candidate run, and
  the replacement is the same whichever candidate is later evaluated

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

### Requirement: Scale families
The harness SHALL provide sparse-wide and dense layered world families, each buildable at
about 12 and about 42 producers. Each family SHALL set its fanout — the number of
producers that declare each product — explicitly. In M1 the gate SHALL admit the sparse-wide
family only: the dense family SHALL stay buildable but SHALL NOT appear in the M1 world
registry (design D10 note).

#### Scenario: Dense world in the M1 registry
- **WHEN** a registry that holds dense worlds is used for a real screen or report
- **THEN** it is refused because dense is not a gate family in M1

#### Scenario: Why dense does not gate in M1
- **WHEN** EXP-0044's dense-12 worlds fail the headroom check (median best fill 0.954 > 0.95),
  and capacity cuts to 0.8× leave them at 0.955–0.962
- **THEN** dense is held out of M1 and sparse-wide, whose headroom medians are below the
  ceiling, gates alone; on calm worlds dense's shortfall is mostly warm-up (54% of it in cycles
  1–20 at 12 producers, 85% at 42), so capacity is not the lever; dense is revisited later with
  a demand-side retune

#### Scenario: Sparse-wide fanout
- **WHEN** a sparse-wide world is built with fanout 2 at 42 producers
- **THEN** every product is declared by exactly 2 producers

#### Scenario: Dense fanout
- **WHEN** a dense world is built at 42 producers
- **THEN** every product is declared by every producer in its layer

### Requirement: Constant economy ratios across sizes
For each family, demand units per unit of initial top-layer throughput SHALL be equal between
its 12-producer and 42-producer worlds within a tolerance recorded in the family definition,
computed over the family's worlds under a sampling protocol recorded there. Cycle emission
per producer SHALL be equal between the two sizes in every world.

#### Scenario: More producers, same pressure
- **WHEN** a family's 42-producer worlds are built
- **THEN** their consumer count and cycle emission are scaled so the family's demand ratio
  over its recorded sample matches the 12-producer sample within the recorded tolerance,
  and every world's emission per producer equals the 12-producer value

#### Scenario: A single world deviates
- **WHEN** one 12-producer world's own demand ratio differs from the 42-producer sample's
  by more than the tolerance, because of its seeded strong or marginal capacity draws
- **THEN** the requirement still holds if the family's ratios over the recorded samples
  (summed demand over summed throughput) agree within the tolerance

### Requirement: Development headroom check
A family SHALL enter the registry only if baseline-only runs on its development worlds show,
at both sizes, a hindsight baseline fill rate (fulfilled over ordered consumer units) below a
saturation ceiling recorded in the registry.

#### Scenario: Saturated family rejected
- **WHEN** the hindsight baseline already fulfils more ordered units than the saturation ceiling on a
  family's development worlds at 42 producers
- **THEN** that family is not admitted to the registry

### Requirement: Side-effect parameter calibration
The run-down factor X and the starvation floor Y SHALL be computed by a recorded rule from
baseline-only runs on development worlds, then frozen in a versioned parameter file before
any gate run reads them. Every gate report SHALL name the parameter version it used.

#### Scenario: Parameters changed after a gate run
- **WHEN** a candidate has been gated under parameter version 1
- **THEN** any change to X or Y creates version 2, and earlier reports keep naming version 1

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

### Requirement: Verdict is reproducible from rows
The gate verdict SHALL be a deterministic function of the report's per-episode rows and
its frozen parameter, roster and registry versions, so that anyone can recompute it from
the report alone.

#### Scenario: Recompute from the CSV
- **WHEN** the verdict is recomputed from a report's per-episode rows and named versions
- **THEN** it matches the verdict recorded in the report exactly

### Requirement: Provenance binding
Every gate report SHALL record the source tree fingerprint and commit, the SHA-256 of every
policy checkpoint, the roster, registry and side-effect parameter versions, and the runtime
environment.

#### Scenario: Report inspected later
- **WHEN** a reader opens a gate report
- **THEN** they can identify the exact code, checkpoints and frozen versions that produced
  it

### Requirement: Legacy measurement unchanged
Adding the harness SHALL NOT change the values of existing episode statistics keys, the
output of the existing scale generator, or the default seeding of existing episode runs.

#### Scenario: Existing fixture rerun
- **WHEN** an existing fixture world and policy pair is rerun after the harness lands
- **THEN** every pre-existing episode statistic is identical to the value before the change

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
