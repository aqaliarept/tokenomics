## MODIFIED Requirements

### Requirement: Frozen baseline roster
The hindsight baseline SHALL be computed from a named, versioned roster of baseline stacks. Roster v1
SHALL contain every hand-written producing baseline paired with every hand-written ordering
baseline, plus `blend_0.5`. Roster v2 SHALL also pair every producing baseline with a
shock-reactive ordering baseline that reacts only to observed consequences of shocks. The roster v3
draft SHALL also pair every producing baseline with each of `round_robin`, `even_split_rotating`,
`random_whole`, `makeable_weighted_whole` and `makeable_split`, which enter the gate's ordering
baselines under those names with the same actions they have as diagnostic routers. A roster
version's pairings SHALL range over the hand-written baselines that exist when that version is
drafted; a baseline added later SHALL enter only a later roster version. Trained policies and
search teachers SHALL NOT be roster members. Every gate report SHALL name the roster version it
used.

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

#### Scenario: Baseline added after a freeze
- **WHEN** a new hand-written ordering baseline is added after roster v2 was frozen
- **THEN** roster v2 stays valid without it, and the new baseline appears only in a draft
  roster version drafted after it was added

#### Scenario: Roster v3 draft
- **WHEN** the draft roster v3 is loaded, before pruning
- **THEN** it holds every producing baseline paired with every v2 ordering baseline and with each
  of the five promoted routers, and no stack whose orderer or producer is a trained checkpoint

#### Scenario: Promoted router acts as before
- **WHEN** a promoted router plays an episode as a gate ordering baseline and as a diagnostic
  router with the same seeds
- **THEN** its actions are identical in both runs, and the frozen v1 and v2 roster files keep
  their content hashes

### Requirement: Random-baseline replicates
A roster member with a stochastic policy (`random`, `random_whole` or `makeable_weighted_whole`)
SHALL run R replicates per episode, with R frozen in the roster version. Each replicate's action
seed SHALL be derived by a cryptographic hash of (world id, episode seed, replicate index) and
SHALL NOT equal the episode's event seed. The member's per-episode value SHALL be the mean over
its replicates.

#### Scenario: Action seed is independent of the event seed
- **WHEN** a random stack runs replicate 0 of episode seed 7 on world W
- **THEN** its action RNG is seeded from the hash of (W, 7, 0), not from 7

#### Scenario: Replicates are averaged, not maximised
- **WHEN** a random stack's three replicates deliver 4, 10 and 7 units on one episode
- **THEN** its per-episode value for the hindsight baseline is 7

#### Scenario: Stochastic promoted router
- **WHEN** the draft roster v3 pairs `entity_planner` with `makeable_weighted_whole`
- **THEN** that member runs R replicates with the hashed action seeds and is valued at their mean

### Requirement: Roster pruning uses development evidence only
A stack SHALL be dropped from a roster only before that roster version is frozen, and only
by a recorded rule applied to baseline-only runs on development worlds. Candidate results
SHALL NOT influence roster membership. A screen MAY run its development worlds in pre-registered
member waves: a later wave SHALL run every member that set the per-episode best on at least one
episode of an earlier wave, plus every member the plan names, and a member not run in a later
wave SHALL be dropped and recorded as not carried forward.

#### Scenario: Never-winning stack dropped
- **WHEN** a stack is the per-episode best member, and so sets the hindsight baseline, on no
  development episode at either economy size
- **THEN** it may be dropped before freezing, and the roster file records the rule and the
  development runs it was applied to

#### Scenario: Wave-1 winner carried to wave 2
- **WHEN** a stack whose producer is not `entity_planner` sets the per-episode best on one wave-1
  episode of the v3 screen
- **THEN** it runs on every wave-2 world, and it is kept if it is the per-episode best on any
  episode of either wave

#### Scenario: Waves recorded
- **WHEN** a two-wave screen freezes a roster
- **THEN** the roster file records each wave's members, episodes and row digest, and the members
  not carried to the later wave

### Requirement: Scale families
The harness SHALL provide sparse-wide and dense layered world families, each buildable at
about 12 and about 42 producers. Each family SHALL set its fanout — the number of
producers that declare each product — explicitly. In M1 the gate SHALL admit the sparse-wide
family only: the dense family SHALL stay buildable but SHALL NOT appear in the M1 world
registry (design D10 note). From registry v3 on, the gate SHALL also admit the world-space region
families `ws_loaded` (12 producers; dense or four makers per product) and `ws_wide` (42 producers;
fourteen makers per product); the legacy dense family SHALL stay outside every gate registry.

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

#### Scenario: Region family in a v2 registry
- **WHEN** a registry named v2 or v1 holds a `ws_wide` or `ws_loaded` entry
- **THEN** it is refused for a real screen or report

#### Scenario: Region family size
- **WHEN** a `ws_wide` entry declares 12 producers
- **THEN** registry validation fails naming the entry

### Requirement: Constant economy ratios across sizes
For each family built at both sizes, demand units per unit of initial top-layer throughput SHALL
be equal between its 12-producer and 42-producer worlds within a tolerance recorded in the family
definition, computed over the family's worlds under a sampling protocol recorded there. Cycle
emission per producer SHALL be equal between the two sizes in every world. A region family built
at one size only SHALL be exempt from the ratio comparison.

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

#### Scenario: Single-size region family
- **WHEN** registry v3 holds `ws_wide` worlds at 42 producers and none at 12
- **THEN** no ratio comparison is made for `ws_wide`, and the sparse-wide comparison is unchanged

### Requirement: Development headroom check
A family SHALL enter the registry only if baseline-only runs on its development worlds show,
at every size where the family has worlds, a hindsight baseline fill rate (fulfilled over ordered
consumer units) below a saturation ceiling recorded in the registry.

#### Scenario: Saturated family rejected
- **WHEN** the hindsight baseline already fulfils more ordered units than the saturation ceiling on a
  family's development worlds at 42 producers
- **THEN** that family is not admitted to the registry

#### Scenario: Region family checked at its own size
- **WHEN** the v3 screen checks headroom
- **THEN** it checks `sparse_wide` at 12 and 42, `ws_loaded` at 12 and `ws_wide` at 42, and any
  failure stops the screen without frozen files

### Requirement: Side-effect parameter calibration
The run-down factor X and the starvation floor Y SHALL be computed by a recorded rule from
baseline-only runs on development worlds, then frozen in a versioned parameter file before
any gate run reads them. From parameter version 3 on, X and Y SHALL be computed and recorded per
gate family, each from that family's development episodes only, and every run-down and starvation
check SHALL use the values of its world's family. Every gate report SHALL name the parameter
version it used.

#### Scenario: Parameters changed after a gate run
- **WHEN** a candidate has been gated under parameter version 1
- **THEN** any change to X or Y creates version 2, and earlier reports keep naming version 1

#### Scenario: Family values used
- **WHEN** a v3 report checks starvation on a `ws_wide` world
- **THEN** it uses the `ws_wide` value of Y recorded in parameter version 3, not the sparse-wide
  value

#### Scenario: Scalar versions read unchanged
- **WHEN** a report or recompute reads parameter version 1 or 2
- **THEN** it applies their single X and Y to every world, as before

## ADDED Requirements

### Requirement: Registry v3 keeps v2 entries
Registry v3 SHALL keep every registry v2 entry with the same id, world seed, episode seeds,
generator reference, status and reserve position.

#### Scenario: v2 worlds carried over
- **WHEN** registry v3 is drafted
- **THEN** each of its v2 entries replays the same configuration and episode seeds as the frozen
  v2 entry with the same id

### Requirement: Region families in registry v3
Registry v3 SHALL add, per split, the region families `ws_loaded` at 12 producers and `ws_wide` at
42. The development split SHALL hold, per family, 4 calm worlds of 3 episodes and 2 shocked worlds
of 10 episodes; the sealed split 8 calm and 4 shocked with the same episode counts. Each group SHALL
have 4 reserves per regime.

#### Scenario: Development group sizes
- **WHEN** the registry v3 draft is loaded
- **THEN** its development `ws_wide` gate worlds are 4 calm with 3 episode seeds each and 2 shocked
  with 10 each, and each regime has 4 reserves

### Requirement: Region regimes
A calm region world SHALL have cyclic deterministic demand and no shocks. A shocked region world
SHALL carry the `shocks-i` profile and its demand drift exactly.

#### Scenario: Shocked region world
- **WHEN** a shocked `ws_wide` world is built
- **THEN** its shock kinds equal the registered `shocks-i` profile's and it has `shocks-i`'s demand
  drift

#### Scenario: Calm region world
- **WHEN** a calm `ws_loaded` world is built
- **THEN** its demand is the cyclic deterministic schedule and it has no shocks

### Requirement: Region entries carry their world record
Every region entry SHALL record its world record: levers, seed, generator version, measured
coordinates and configuration hash. Loading SHALL refuse an entry whose replayed configuration hash
differs from the recorded one.

#### Scenario: Record mismatch
- **WHEN** a region entry's replayed configuration hash differs from its recorded one
- **THEN** loading the registry fails naming the entry

### Requirement: Region worlds are fresh draws disjoint from atlas and M4 worlds
Region worlds SHALL be drawn from the world-space lever distribution on seed streams of the gate's
own, accepted in a fixed draw order when their measured tightness ρ lies in (0.28, 3.1]. No region
world seed SHALL equal a registry v2 world seed, a world seed of the atlas-v1 sample or its
anchors, or a world seed of the M4 training-draw stream, and the development and sealed region
seeds SHALL be disjoint. No atlas world SHALL be a registry world.

#### Scenario: Atlas world offered to the registry
- **WHEN** a draft registry entry has the world seed and levers of an atlas v2 world of any split
- **THEN** registry validation fails naming the entry and the atlas world

#### Scenario: Draw outside the band
- **WHEN** a region draw's measured ρ is 0.2 or 4.0
- **THEN** it is skipped, recorded with its ρ, and the next draw in order is tried

#### Scenario: Pool exhausted
- **WHEN** the draw pool holds too few accepted draws for a group's gate worlds and reserves
- **THEN** the draft refuses to build instead of shrinking the group

### Requirement: Region worlds whose best baseline dies are replaced
During a screen, a region world on which the per-episode best roster member reaches system death
in any episode SHALL be replaced by the next unused reserve of its group, in the fixed reserve
order, before headroom, pruning and calibration are computed. The replacement SHALL be recorded in
the registry version. Sparse-wide worlds SHALL keep their v2 screening rules.

#### Scenario: Best baseline dies
- **WHEN** on a `ws_loaded` development world the member with the most units dies at cycle 70 in
  one episode
- **THEN** the world is replaced by its group's next reserve and the screening record names the
  world, episode and member

### Requirement: Sealed split screened by size
The sealed split SHALL be screened by a roster-only run of the frozen roster with zero-baseline and
bar-death replacement only, for named economy sizes, before any candidate is evaluated on sealed
worlds of those sizes. The screen SHALL NOT prune the roster or recalibrate parameters, SHALL keep
`screened_on`, and SHALL record which (split, size) groups are screened. A sealed report on a size
whose sealed groups are unscreened SHALL be refused.

#### Scenario: Sealed read before its screen
- **WHEN** a sealed report at 12 producers is requested on a registry whose 12-producer sealed
  groups are unscreened
- **THEN** the harness refuses before running any episode

#### Scenario: Screen leaves the roster alone
- **WHEN** the sealed 12-producer screen finishes
- **THEN** the roster and parameter files keep their content hashes and the registry keeps
  `screened_on`

### Requirement: Per-order candidate subject
A gate report SHALL accept a per-order orderer checkpoint under the `choice-v1` decode as the
candidate's orderer, paired with a hand-written producing baseline as the deciding producer. Only
the deciding rows SHALL enter the verdict.

#### Scenario: Deciding producer is white-box
- **WHEN** a per-order candidate is reported with `entity_planner` as the deciding producer
- **THEN** the verdict is computed from rows whose producer is `entity_planner` and whose orderer
  is the checkpoint under `choice-v1`

#### Scenario: Composite under the legacy decode
- **WHEN** a per-order composite checkpoint is requested under the `logits-v1` decode
- **THEN** the harness refuses before running any episode

### Requirement: Candidate replicates for a sampled orderer
A per-order candidate SHALL run R replicates per episode, R equal to the frozen roster's R. Each
replicate SHALL use the hashed action seed and a draw key of (world id, episode seed, replicate).
Its per-episode units SHALL be the replicate mean. Deaths, run-down and starvation SHALL be checked
on every replicate.

#### Scenario: Wrong replicate count
- **WHEN** a real report requests 3 candidate replicates against a roster frozen with R = 5
- **THEN** the harness refuses before running any episode

#### Scenario: One replicate dies
- **WHEN** one of five candidate replicates of an episode reaches system death and the replicate
  mean of units clears the margin
- **THEN** the zero-death clause fails, naming the world, episode and replicate

### Requirement: Reported producer column
A per-order report MAY add a reported column: the same orderer, seeds and draw keys with a
checkpoint producer. Reported rows SHALL be kept apart from the verdict and SHALL be listed per
world and regime with their ratio against the same hindsight baseline, deaths and hard side-effect
checks, marked as reported and not gating. Recomputing the report SHALL rebuild the reported table
and compare it with the recorded one.

#### Scenario: Reported column fails a clause
- **WHEN** the reported column misses a margin while the deciding column passes every clause
- **THEN** the overall verdict is pass and the reported table shows the miss

#### Scenario: Recompute covers the reported table
- **WHEN** a per-order report is recomputed from its rows
- **THEN** the result is MATCH only if the verdict, the resilience table and the reported table
  all equal the recorded ones

### Requirement: Decode and component pins
A per-order report SHALL pin, in its run specification and resume check, the ordering decode, the
decoder and architecture versions, the checkpoint format, the SHA-256 of the composite checkpoint,
its base and its normaliser, the parameter hash, the pre-registration file hash, the deciding
producer's name, the reported producer checkpoint's SHA-256 and R. A sealed pre-registration
manifest for a per-order candidate SHALL bind every one of these pins.

#### Scenario: Resume with a different base
- **WHEN** a per-order report is resumed on the same output directory with a composite whose base
  SHA-256 differs from the pinned one
- **THEN** the harness refuses to resume

#### Scenario: Sealed manifest misses a pin
- **WHEN** a sealed per-order read is requested with a manifest that does not bind the
  normaliser hash
- **THEN** the harness refuses before running any episode
