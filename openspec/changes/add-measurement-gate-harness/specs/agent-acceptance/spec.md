# Spec Delta

## MODIFIED Requirements

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
