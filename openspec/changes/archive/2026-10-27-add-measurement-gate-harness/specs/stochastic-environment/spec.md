# Spec Delta

## ADDED Requirements

### Requirement: Deterministic cyclic demand
A consumer SHALL be configurable with an explicit finite schedule of (product, quantity)
entries that is replayed cyclically, one entry per cycle, identically in every episode
regardless of the episode seed. A consumer without this setting SHALL behave exactly as
before.

#### Scenario: Schedule repeats
- **WHEN** a consumer is configured with the schedule [(P, 1), (Q, 2)] for a 5-cycle episode
- **THEN** it orders P×1, Q×2, P×1, Q×2, P×1

#### Scenario: Episode seed has no effect
- **WHEN** a world whose consumers all use explicit schedules runs with episode seeds 0 and 9
- **THEN** both episodes receive the identical demand

#### Scenario: Legacy fixed demand untouched
- **WHEN** a configuration sets no explicit schedule on any consumer
- **THEN** every episode statistic matches the run before the setting existed
