# Spec Delta

## Purpose

Defines how a candidate policy stack is measured against the acceptance contract: which
baselines form the hindsight baseline, how the hindsight baseline and ratios are computed, which worlds may be
used for what, and what a gate report must contain so its verdict is honest and
reproducible.

## ADDED Requirements

### Requirement: Frozen baseline roster
The hindsight baseline SHALL be computed from a named, versioned roster of baseline stacks. Roster v1
SHALL contain every hand-written producing baseline paired with every hand-written ordering
baseline, plus `blend_0.5`. Every gate report SHALL name the roster version it used.

#### Scenario: blend_0.5 is a roster member
- **WHEN** roster v1 is loaded
- **THEN** it contains a stack whose ordering policy is `blend_0.5`

#### Scenario: Roster edited without a new version
- **WHEN** a roster file's member list no longer matches the content hash recorded for its
  version
- **THEN** the harness refuses to compute a hindsight baseline from it

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
identical demand schedule, even when they run under different observation or ordering
contracts. The harness SHALL verify this by comparing a schedule digest per episode.

#### Scenario: Digest mismatch
- **WHEN** a roster member's episode demand-schedule digest differs from the candidate's
  for the same world and episode seed
- **THEN** the gate run fails with an error naming the world, episode and member

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
worlds.

#### Scenario: Stochastic demand is shocked
- **WHEN** a world uses seeded random demand and no supply shocks
- **THEN** it is registered and scored in the shocked regime

#### Scenario: Too few calm worlds
- **WHEN** a registry holds 10 shocked and 15 calm worlds at 42 producers
- **THEN** registry validation fails

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
- **THEN** dense is held out of M1, because its shortfall is mostly warm-up (62% of it in
  cycles 1–20 at 12 producers, 86% at 42) while sparse-wide keeps headroom after warm-up; dense
  is revisited later with a demand-side retune

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
and an overall verdict for each acceptance clause.

#### Scenario: One clause fails
- **WHEN** every margin passes but one episode violates the starvation constraint
- **THEN** the report's overall verdict is reject and it names the failing clause, world and
  episode

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
