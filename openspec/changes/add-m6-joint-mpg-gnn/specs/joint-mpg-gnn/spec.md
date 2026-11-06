## ADDED Requirements

### Requirement: Imitation before reinforcement learning
The fully learned policy SHALL first imitate k5_rf5 + MWWk per agent decision, and a pre-registered imitation check SHALL decide whether reinforcement learning starts.

#### Scenario: Imitation check
- **WHEN** the imitated policy plays the monitor world-episodes against the bar at the same keys
- **THEN** it reads `faithful` (not below the bar beyond the floor, no extra deaths) or `short`, and `short` stops before any RL

### Requirement: Topology-agnostic graph policy
The policy SHALL use one parameter set at 12 and 42 producers and on unseen topology families, with ADR-0058's information set.

#### Scenario: Cross-topology decoding
- **WHEN** the policy runs with random nonzero parameters at 12, 42 and an unseen family
- **THEN** every command is legal and there are zero contract failures

### Requirement: Success against the best hand-written stack
Success SHALL be a held-out confirmation that consumer units are not below the k5_rf5 + MWWk bar beyond its floor (parity counts), with no extra deaths and side-effect shares no worse.

#### Scenario: Held-out read
- **WHEN** the L6 read runs once on held-out worlds at 12 and 42
- **THEN** its pre-registered reading is `confirmed` or `not_confirmed`
