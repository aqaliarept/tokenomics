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
consumer units) below a saturation ceiling recorded in the registry. From registry v3 on, headroom
SHALL be decided per family and size: a family that fails at a size SHALL be excluded at that
size in both splits, with its rows reported only, while other families proceed. A screen SHALL
stop without frozen files only when a size is left with no admitted family. A family whose worlds
run in an earlier member wave SHALL be checked before a later wave starts.

#### Scenario: Saturated family rejected
- **WHEN** the hindsight baseline already fulfils more ordered units than the saturation ceiling on a
  family's development worlds at 42 producers
- **THEN** that family is not admitted to the registry

#### Scenario: Region family checked at its own size
- **WHEN** the v3 screen checks headroom
- **THEN** it checks `sparse_wide` at 12 and 42, `ws_loaded` at 12 and `ws_wide` at 42

#### Scenario: One family fails under v3
- **WHEN** after wave 1 of the v3 screen `sparse_wide` at 12 producers has median best fill 0.96
  and `ws_loaded` 0.49
- **THEN** the `sparse_wide` 12-producer entries are marked excluded, their rows are reported, and
  the screen goes on to wave 2

#### Scenario: Exclusion covers both splits
- **WHEN** `sparse_wide` at 12 producers is excluded by the development screen
- **THEN** its 12-producer sealed gate worlds and reserves are excluded too, and the sealed
  screen and every sealed read skip them

#### Scenario: Recorded exclusion passes admission
- **WHEN** a frozen v3 registry records `sparse_wide` at 12 producers as failed and excluded
- **THEN** a real development report, the sealed 12-producer screen and a 12-producer scoped
  sealed read all pass admission, and a failed headroom record without an exclusion is still
  refused

#### Scenario: Excluded rows do not shape the frozen files
- **WHEN** a member sets the per-episode best only on episodes of an excluded family × size
- **THEN** it is not carried to wave 2 for that reason, it counts as never-winning for pruning,
  and the excluded episodes enter no family's X / Y

#### Scenario: A size without an admitted family
- **WHEN** after wave 1 both families at 12 producers fail headroom
- **THEN** the screen stops before wave 2 and writes no frozen files

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

### Requirement: World registry
Gate and development worlds SHALL be listed in a versioned registry. Each entry SHALL give
a world id, family, economy size, regime, a replayable generator reference, a world seed,
its episode seeds, and a split of `development` or `sealed`. Development and sealed worlds
SHALL use disjoint world seeds. Legacy families SHALL draw world seeds from the registry's per-split
seed ranges; region families SHALL draw them from split-named hashed streams, and their
disjointness SHALL be checked as explicit sets on load and proven by a test.

#### Scenario: Overlapping seeds rejected
- **WHEN** a registry lists the same family and world seed under both splits
- **THEN** loading the registry fails validation

#### Scenario: Hashed region seeds
- **WHEN** registry v3 holds a `ws_wide` development world whose seed lies outside 1000–1999
- **THEN** it loads, because region seeds are checked by set disjointness against the sealed
  region seeds and every legacy seed, not by range

#### Scenario: Region seed shared across splits
- **WHEN** a region world seed appears under both the development and the sealed split
- **THEN** loading the registry fails naming both entries

## ADDED Requirements

### Requirement: Attainable margins are reported
A screen SHALL report, per (size, regime) and per family, the perfect-fill ratio, the mean over
worlds of the world's ordered units over its hindsight baseline, against the regime's margin,
and SHALL flag a cell below its margin as unattainable. The flag SHALL NOT change admission,
pruning or calibration.

#### Scenario: Shocked margin out of reach
- **WHEN** the v3 screen's sparse_wide 12-producer shocked worlds give a perfect-fill ratio of 1.09
- **THEN** that cell is flagged unattainable against 1.10 in the screen record, and the frozen
  files are unchanged by the flag

### Requirement: Routing reading per world
A screen SHALL report per world the relative gain of the input-aware routers over the best blind
router, both paired with the best producing baseline, and count a world when that gain is at
least 0.05 and above the world's own floor (pre-registered in the design per world type, 0 when both
sides are deterministic on a calm world). A family SHALL read `routing_matters` when at
least a third of its admitted worlds count, else `routing_flat`. The reading SHALL NOT gate.

#### Scenario: Flat region family
- **WHEN** on 1 of a region family's 6 admitted worlds the input-aware routers beat the best blind
  router by at least 0.05 and by more than the floor
- **THEN** the screen record reads `routing_flat` for that family and admission is unchanged

#### Scenario: Deterministic sides on a calm world
- **WHEN** on a calm world `makeable_split` beats `round_robin`, the best blind router, by 6% and
  both are deterministic
- **THEN** the floor is 0 and the world counts

#### Scenario: Poor routers do not make routing matter
- **WHEN** on every world of a family `even_split` delivers half of `round_robin`'s units, and the
  input-aware routers are within 0.02 of `round_robin`
- **THEN** the family reads `routing_flat`

### Requirement: Size-scoped real report
A real report MAY declare a size scope. Its verdict SHALL be computed over the scoped sizes only,
the scope SHALL be recorded in the run specification and bound in any pre-registration manifest,
and recomputing SHALL use the recorded scope. From registry v3 on, admission SHALL check each
family at the sizes it is declared for.

#### Scenario: 12-producer sealed read
- **WHEN** a sealed report is scoped to 12 producers on registry v3
- **THEN** the verdict covers only 12-producer worlds, no margin failure is recorded for 42, and
  the region family `ws_wide` does not block admission

#### Scenario: Scope differs from the manifest
- **WHEN** a sealed read is scoped to 12 producers but its manifest binds both sizes
- **THEN** the harness refuses before running any episode

### Requirement: Packaging a gate version does not move driver defaults
Adding a new registry, roster and parameter version to the packaged files SHALL NOT change the
version any driver reads by default. The default SHALL move only after the user approves and every
driver that reads it pins its version explicitly.

#### Scenario: v3 packaged during M4
- **WHEN** the v3 files are packaged
- **THEN** the default registry, roster and parameter files are still v2, and the M4 training-world
  driver loads the same parameters and refuses the same rosters as before

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
candidate's orderer, paired with a hand-written producing baseline as the deciding producer. Its
episodes SHALL run in the per-order policy's own environment, agent-local history included. Only
the deciding rows SHALL enter the verdict. A non-finite or invalid choice SHALL be a contract
failure.

#### Scenario: Deciding producer is white-box
- **WHEN** a per-order candidate is reported with `entity_planner` as the deciding producer
- **THEN** the verdict is computed from rows whose producer is `entity_planner` and whose orderer
  is the checkpoint under `choice-v1`

#### Scenario: Composite under the legacy decode
- **WHEN** a per-order composite checkpoint is requested under the `logits-v1` decode
- **THEN** the harness refuses before running any episode

#### Scenario: Non-finite choice
- **WHEN** the candidate's policy raises on a non-finite score in one job
- **THEN** that job is a contract failure, the other jobs finish, and the run is invalid

#### Scenario: Resume across decodes
- **WHEN** a per-order report is resumed onto an output directory written by a `logits-v1` report,
  or the reverse
- **THEN** the harness refuses to resume

### Requirement: Candidate replicates for a sampled orderer
A per-order candidate SHALL run R replicates per episode, R equal to the frozen roster's R, each
with the hashed action seed and a draw key of (world id, episode seed, replicate), and valued at
the replicate mean of units.

#### Scenario: Wrong replicate count
- **WHEN** a real report requests 3 candidate replicates against a roster frozen with R = 5
- **THEN** the harness refuses before running any episode

### Requirement: Side effects of a replicated candidate
From parameter version 3 on, a replicated candidate's run-down and starvation SHALL be judged on
the replicate means the calibration uses, with per-replicate violations reported as monitored, and
a death in any replicate SHALL fail the zero-death clause. Reports on earlier parameter versions
SHALL keep their per-replicate checks.

#### Scenario: One replicate dies
- **WHEN** one of five candidate replicates of an episode reaches system death and the replicate
  mean of units clears the margin
- **THEN** the zero-death clause fails, naming the world, episode and replicate

#### Scenario: One replicate starves a consumer
- **WHEN** one of five replicates leaves a consumer below Y while that consumer's mean fill rate
  over the five is above Y
- **THEN** the starvation clause passes and the replicate's violation is listed as monitored

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
A per-order report SHALL pin, in its run specification and resume check, the per-order policy's
own pin set, including any key that set later adds, plus the composite checkpoint's SHA-256 and
format, the pre-registration file hash, the deciding producer's name, the reported producer
checkpoint's SHA-256, R and the size scope. A sealed pre-registration manifest for a per-order
candidate SHALL bind the same pin set.

#### Scenario: Resume with a different base
- **WHEN** a per-order report is resumed on the same output directory with a composite whose base
  SHA-256 differs from the pinned one
- **THEN** the harness refuses to resume

#### Scenario: Sealed manifest misses a pin
- **WHEN** a sealed per-order read is requested with a manifest that does not bind the
  normaliser hash
- **THEN** the harness refuses before running any episode

#### Scenario: Policy pin set grows
- **WHEN** the per-order policy's pin set gains a key, such as an input-transform hash
- **THEN** the report pins that key without a change to the gate's pin list

