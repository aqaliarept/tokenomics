## MODIFIED Requirements

### Requirement: Frozen baseline roster
The hindsight baseline SHALL be computed from a named, versioned roster of baseline stacks. Roster v1
SHALL contain every hand-written producing baseline paired with every hand-written ordering
baseline, plus `blend_0.5`. Roster v2 SHALL also pair every producing baseline with a
shock-reactive ordering baseline that reacts only to observed consequences of shocks. A roster
version's pairings SHALL range over the hand-written baselines that exist when that version is
drafted; a baseline added later SHALL enter only a later roster version. Every gate report SHALL
name the roster version it used.

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

## ADDED Requirements

### Requirement: Baseline rule corrections create a new roster version
A defect found in a hand-written baseline SHALL be fixed under a new baseline name. The
original baseline SHALL keep its behaviour, so that frozen rosters naming it replay unchanged.
The fixed baseline SHALL reach the gate only as a member of a new roster version, after a
baseline-only screen that freezes new registry, roster and parameter versions.

#### Scenario: Defective rule fixed
- **WHEN** a diagnosis shows that a hand-written ordering rule misroutes orders
- **THEN** the fixed rule is added under a new name, the original rule's actions are
  unchanged, and the frozen v2 registry, roster and parameter files keep their content hashes

#### Scenario: Fixed rule in the hindsight baseline
- **WHEN** a fixed rule is to set the hindsight baseline
- **THEN** a draft roster version holding it is screened on the development worlds, and new
  frozen registry, roster and parameter versions sharing one `screened_on` are written before
  any report uses it

### Requirement: Raising the bar is a user decision
A new roster version that changes the hindsight baseline SHALL become the default for gate
reports only after the user approves it. Earlier reports SHALL keep naming the versions they
used.

#### Scenario: Re-screen finished
- **WHEN** a re-screen freezes roster v3 and its registry and parameter versions
- **THEN** the gate default stays v2 until the user approves v3, and EXP-0045 and EXP-0047
  keep naming their versions
