# side-effect-guards Specification

## Purpose

Defines side effects — harm a policy causes outside the metric it optimises — and fixes
which side effects reject a candidate outright and which are measured and reported, so a
policy cannot win on fulfilled units by damaging what the units metric does not see.

## Requirements

### Requirement: Side-effect definition
A side effect SHALL mean a degradation of economy state or of token-signal meaning that is
not captured by fulfilled consumer units and the death rate. Resilience to shocks SHALL be
treated as a separate robustness property, not as a side effect.

#### Scenario: Classifying a new concern
- **WHEN** a proposed metric measures harm that fulfilled units and death rate already
  reflect
- **THEN** it is not added as a side-effect guard

### Requirement: No end-of-horizon run-down
At episode end, the mean capacity ratio (current over max capacity) across all producers
SHALL be at least X times its value at episode start, where X is a fixed parameter set by
design before the first gate run that uses it.

#### Scenario: Investment stopped before the horizon
- **WHEN** a policy stops restoration investment in the final cycles and the end-of-episode
  mean capacity ratio falls below X times the start value
- **THEN** the episode violates the run-down constraint and the candidate is rejected

### Requirement: No consumer starvation
Every consumer SHALL receive a fill rate (fulfilled units over ordered units across the
episode) of at least Y, where Y is a fixed parameter set by design before the first gate
run that uses it.

#### Scenario: Deep-product consumer ignored
- **WHEN** total fulfilled units are high but one consumer demanding a deep composite
  product has a fill rate below Y
- **THEN** the episode violates the starvation constraint and the candidate is rejected

### Requirement: Per-consumer accounting
Episode statistics SHALL report ordered and fulfilled units per consumer, not only totals,
so the starvation constraint is checkable. Ordered units SHALL count only consumer orders
placed in cycles that were played. Orders still unresolved at the horizon SHALL stay in the
ordered count as unfulfilled. A consumer with zero ordered units SHALL be reported but
excluded from the starvation check.

#### Scenario: Episode stats inspected
- **WHEN** an episode finishes
- **THEN** its statistics list ordered and fulfilled units for each consumer

#### Scenario: Orders for the unplayed cycle are excluded
- **WHEN** an episode is truncated at its horizon after the final cycle has already placed
  the next cycle's consumer orders
- **THEN** those next-cycle orders are not counted as ordered units

#### Scenario: Demand unresolved at the horizon
- **WHEN** a consumer order placed in the last played cycle is still in progress when the
  horizon is reached
- **THEN** its units count as ordered and not fulfilled

#### Scenario: Early system death
- **WHEN** an episode ends by system death at cycle t
- **THEN** ordered units count only orders placed in the cycles played up to and including
  cycle t

#### Scenario: Consumer that ordered nothing
- **WHEN** a consumer placed no orders during the played cycles
- **THEN** it appears in the statistics with zero ordered units and is not judged by the
  starvation constraint

### Requirement: Token-priority inversion is monitored
Every evaluation SHALL report how often lower token-per-unit consumer orders are fulfilled
while higher token-per-unit orders for the same product are rejected in the same cycle.
This metric SHALL NOT gate acceptance.

#### Scenario: Inversion reported
- **WHEN** a gate run completes
- **THEN** its report includes the token-priority inversion rate alongside the margins

### Requirement: Edge-running margin is monitored
Every evaluation SHALL report the minimum spare-capacity margin and minimum capacity ratio
reached during each episode. This metric SHALL NOT gate acceptance.

#### Scenario: Margin reported
- **WHEN** a gate run completes
- **THEN** its report includes per-episode minimum spare capacity and minimum capacity
  ratio

### Requirement: Investment crowd-out is monitored
Every evaluation SHALL report the rejection rate of investment orders (restoration,
upgrade, inventory expansion) relative to consumer orders. This metric SHALL NOT gate
acceptance.

#### Scenario: Crowd-out reported
- **WHEN** a gate run completes
- **THEN** its report includes investment-order and consumer-order rejection rates

### Requirement: Hoarding stays penalised
Inventory held between cycles SHALL continue to incur the env-layer holding penalty in the
training reward, and held inventory SHALL be reported per evaluation.

#### Scenario: Overproduction ahead of demand
- **WHEN** a policy accumulates inventory beyond what demand consumes
- **THEN** its training reward is reduced by the holding penalty and the held units appear
  in its evaluation report
