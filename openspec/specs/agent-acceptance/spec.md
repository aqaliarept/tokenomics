# agent-acceptance Specification

## Purpose

Defines the gate a candidate policy stack (one producing policy plus one ordering policy)
must pass to count as playing the game excellently, including how it must transfer across
economy size and what it may cost per decision.

## Requirements

### Requirement: Primary metric
Acceptance SHALL be judged on fulfilled consumer units (`consumer_completed_units`) per
episode, compared pairwise against the hindsight baseline: per evaluated world and
episode seed, the best result any hand-written baseline achieves on that same episode.

#### Scenario: Hindsight-baseline ratio computed per world
- **WHEN** a candidate and all baselines are evaluated on the same world and episode seeds
- **THEN** the candidate's ratio for that world is its mean units divided by the hindsight baseline's
  mean units on those identical episodes

### Requirement: Held-out evaluation
Acceptance SHALL be measured only on held-out worlds whose world seeds were not used for
training, distillation, mining, or checkpoint selection, with evaluation seeds anchored at
each world's recorded episode seed.

#### Scenario: Training world in the gate set
- **WHEN** a gate world's seed appears in the candidate's training or selection data
- **THEN** the gate run is invalid and is not used for acceptance

### Requirement: Gate worlds have a non-zero hindsight baseline
Every gate world SHALL have a strictly positive hindsight baseline mean, checked from baseline runs
alone before any candidate is evaluated. A world whose hindsight baseline mean is zero SHALL be
replaced by another held-out world before the gate runs, so every hindsight-baseline ratio is finite
and the gate set does not depend on candidate results.

#### Scenario: All baselines deliver zero
- **WHEN** every baseline delivers 0 units on a proposed gate world
- **THEN** that world is removed from the gate set and replaced before any candidate run

#### Scenario: Candidate positive on a zero-baseline world
- **WHEN** a candidate would deliver positive units on a world whose hindsight baseline mean is 0
- **THEN** the world is still excluded, no infinite ratio enters any regime mean, and the
  result is reported separately as a non-gating observation

### Requirement: Calm-regime margin
On held-out worlds with neither supply shocks nor stochastic demand, a candidate SHALL reach
a mean hindsight-baseline ratio of at least 1.05 across the gate worlds.

#### Scenario: Calm margin met
- **WHEN** the candidate's mean hindsight-baseline ratio over calm gate worlds is 1.06
- **THEN** the calm-regime margin passes

#### Scenario: Calm margin missed
- **WHEN** the candidate's mean hindsight-baseline ratio over calm gate worlds is 1.03
- **THEN** the candidate is rejected

#### Scenario: Stochastic demand without shocks
- **WHEN** a held-out world has seeded stochastic demand but no supply shocks
- **THEN** it is judged by the shocked-regime margin, not the calm-regime margin

### Requirement: Shocked-regime margin
On held-out worlds with supply shocks or stochastic demand enabled, a candidate SHALL
reach a mean hindsight-baseline ratio of at least 1.10 across the gate worlds.

#### Scenario: Shocked margin missed
- **WHEN** the candidate passes the calm margin but its mean shocked hindsight-baseline ratio is 1.07
- **THEN** the candidate is rejected

### Requirement: No losing world
A candidate SHALL reach a hindsight-baseline ratio of at least 1.0 on every individual gate world,
in every regime.

#### Scenario: One world below the hindsight baseline
- **WHEN** the candidate's mean ratio passes but one gate world scores 0.97
- **THEN** the candidate is rejected

### Requirement: Zero death
A candidate SHALL produce zero system deaths across all gate episodes in every regime.

#### Scenario: Single death
- **WHEN** one gate episode ends in system death
- **THEN** the candidate is rejected regardless of its margins

### Requirement: Side-effect guards are part of acceptance
A candidate SHALL pass every hard constraint in `side-effect-guards` on all gate episodes.

#### Scenario: Margin passes, run-down fails
- **WHEN** a candidate passes all margins but violates the run-down constraint
- **THEN** the candidate is rejected

### Requirement: Train small, play large
A candidate SHALL be trained only on economies of roughly 12 producers or fewer, and SHALL
pass this gate unretrained on economies of roughly 42 producers (about 3.5x), at every chain
depth the current order lifecycle makes feasible.

#### Scenario: Scale gate unretrained
- **WHEN** a candidate trained on ~12-producer economies is evaluated on ~42-producer held-out
  economies
- **THEN** the same checkpoints run without retraining and meet every margin, death, and
  side-effect requirement there

#### Scenario: Retraining at the large size
- **WHEN** the candidate only passes at ~42 producers after training on that size
- **THEN** the scale requirement fails

### Requirement: Future scale target
Scales of roughly 10x (about 120 producers) and roughly 100x (1000+ producers) SHALL be
recorded as future targets and SHALL NOT gate acceptance in this phase.

#### Scenario: 120-producer result reported
- **WHEN** a candidate is evaluated at about 120 producers
- **THEN** the result is reported but does not change the accept/reject decision

#### Scenario: 100x result reported
- **WHEN** a candidate is evaluated at 100x
- **THEN** the result is reported but does not change the accept/reject decision

### Requirement: Bounded per-decision cost
A deployed policy's compute per decision SHALL grow at most linearly with the number of
eligible producers it chooses among, and SHALL NOT run full-episode or multi-cycle
simulator rollouts at decision time.

#### Scenario: Search policy as candidate
- **WHEN** a candidate performs per-cycle rollout search at decision time
- **THEN** it is ineligible for promotion and is used only as a teacher for distillation

### Requirement: Human promotion
Passing this gate SHALL make a candidate eligible for champion promotion; replacing the
champion SHALL remain a human decision.

#### Scenario: Automated pass
- **WHEN** an automated pipeline reports that a candidate passed the gate
- **THEN** the champion is unchanged until a human promotes the candidate
