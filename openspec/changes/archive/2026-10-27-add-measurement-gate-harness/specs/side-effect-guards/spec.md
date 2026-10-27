# Spec Delta

## MODIFIED Requirements

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
