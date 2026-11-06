## ADDED Requirements

### Requirement: Joint learning starts at the hand-written bar
The joint producer and orderer policy SHALL equal `entity_planner_buyahead_k5_rf5` with MWWk action for action at zero parameters.

#### Scenario: Zero residual equals the bar
- **WHEN** both heads' parameters are zero
- **THEN** every producing and ordering action equals k5_rf5 + MWWk's on the same episode and keys

### Requirement: Topology-agnostic graph policy
The policy SHALL use one parameter set at 12 and 42 producers and on unseen topology families, with ADR-0058's information set.

#### Scenario: Cross-topology decoding
- **WHEN** the policy runs with random nonzero parameters at 12, 42 and an unseen family
- **THEN** every command is legal and there are zero contract failures

### Requirement: Success against the best hand-written stack
Success SHALL be a held-out confirmation that consumer units exceed the k5_rf5 + MWWk and k5_rf5 + RR bars beyond their floors (default; parity within the floor is a user decision), with no extra deaths and side-effect shares no worse.

#### Scenario: Held-out read
- **WHEN** the L6 read runs once on held-out worlds at 12 and 42
- **THEN** its pre-registered reading is `confirmed` or `not_confirmed`
