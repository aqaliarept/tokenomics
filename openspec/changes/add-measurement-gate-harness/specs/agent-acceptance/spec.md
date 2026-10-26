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
