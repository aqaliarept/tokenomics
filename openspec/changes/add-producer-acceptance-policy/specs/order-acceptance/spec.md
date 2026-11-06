## ADDED Requirements

### Requirement: Producer acceptance decision
Each cycle, after ordering agents route sub-orders, each producer's acceptance policy SHALL return accept or decline for every sub-order routed to it. Declined sub-orders SHALL NOT be re-routed within the same cycle.

#### Scenario: Busy capacity does not veto acceptance
- **WHEN** a producer accepts a sub-order whose bundle does not fit its free capacity this cycle
- **THEN** the sub-order is reserved and executes in a later cycle when capacity and inputs allow, or fails at its order's TTL

#### Scenario: Decline is final for the cycle
- **WHEN** a producer declines a sub-order
- **THEN** no tokens are burnt and the order's holder can re-route that quantity only in the next cycle

### Requirement: Explicit sub-order outcomes
The domain SHALL record an outcome (accepted, declined, completed, failed with reason) for every sub-order, and the ordering agent that routed it SHALL observe it in its next cycle's view.

#### Scenario: Outcome observable next cycle
- **WHEN** a sub-order is declined or fails in cycle t
- **THEN** its routing agent's view in cycle t + 1 includes that outcome and reason

### Requirement: Default acceptance reproduces today
The default acceptance policy SHALL be today's atomic price-greedy auction (accept iff the bundle fits now), and with it every existing run SHALL be byte-identical.

#### Scenario: Goldens unchanged
- **WHEN** the legacy invariance goldens run with the default acceptance policy
- **THEN** all pass unchanged
