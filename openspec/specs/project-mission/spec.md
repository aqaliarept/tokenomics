# project-mission Specification

## Purpose

States what the project is for, how the rules and the agents are allowed to evolve, and
which rule mechanics are in or out of scope, so every later change can be checked against
the mission rather than against the current training metric.

## Requirements

### Requirement: Mission
The project SHALL aim to produce software agents that play the cooperative tokenomics game
excellently without unexpected side effects. The training and evolution pipeline SHALL be
treated as an intermediate means to that mission, never as the deliverable.

#### Scenario: Pipeline work is justified by the mission
- **WHEN** a change proposes new training or evolution-pipeline work
- **THEN** its proposal names which acceptance criterion in `agent-acceptance` or
  `side-effect-guards` that work moves toward

### Requirement: Rules and agents co-evolve
Rule mechanics SHALL be extendable while agent training continues; an unmet agent
acceptance bar on the current ruleset SHALL NOT block specifying or implementing the next
in-scope rule mechanic.

#### Scenario: New mechanic while the champion fails a margin
- **WHEN** the champion has not yet passed the margin on the current ruleset
- **AND** an in-scope rule mechanic is ready to be added
- **THEN** the mechanic is added without waiting for the champion to pass the margin

### Requirement: Agent contract survives rule changes
Adding an in-scope rule mechanic SHALL NOT require changing the agent observation/action
contract in a way that invalidates existing checkpoints; existing champion checkpoints
SHALL load and run on the extended ruleset with zero contract failures.

#### Scenario: Champion runs on a shocked ruleset unretrained
- **WHEN** supply shocks are enabled
- **THEN** the current champion checkpoints load and complete episodes with zero contract
  failures, without retraining

### Requirement: In-scope rule mechanics
The rules SHALL be extended with exactly these mechanics in this phase: stochastic demand,
supply shocks, and a longer order lifecycle (TTL beyond 2 cycles and multi-stage orders).

#### Scenario: Proposed mechanic outside the list
- **WHEN** a change proposes a rule mechanic not in this list
- **THEN** the change first amends this requirement to add it to scope

### Requirement: New capacity types are out of scope
Producing agents SHALL NOT acquire capacity types they did not own at episode start, and
no investment type SHALL create a new capacity type. System death SHALL keep its
permanent meaning: some product is unmakeable by every producer that declares it.

#### Scenario: Death stays permanent
- **WHEN** no producer that declares product P holds every capacity type P's recipe needs
- **THEN** the episode terminates as system death, because no mechanic can restore the
  missing type

### Requirement: Order of rule work
Stochastic demand and supply shocks SHALL be delivered before the longer order lifecycle.

#### Scenario: Lifecycle work requested first
- **WHEN** order-lifecycle work is proposed while stochastic demand and supply shocks are
  not yet delivered
- **THEN** the proposal records why the order is being changed and amends this
  requirement
