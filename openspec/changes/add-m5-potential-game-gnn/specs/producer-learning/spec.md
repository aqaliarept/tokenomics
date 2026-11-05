# Spec Delta

## Purpose
A learned producer correction on top of the strongest hand-written producer (`entity_planner_buyahead_k4`, or its "expand earlier" variant if that reads `helps`). It is trained by evolution strategies against a frozen orderer (MWW), read on pre-registered gates, and confirmed on held-out worlds. Joint learning of both roles is a later, gated phase.

## ADDED Requirements

### Requirement: Residual on the strongest hand-written producer
The learned producer SHALL be a residual on the registered base producer. It SHALL change only the per-input buffer size, the input units bought beyond the planner's target, and the inventory-expansion trigger, and keep the base's restoration, upgrade and input priority. At zero parameters it SHALL equal the base producer action for action.

#### Scenario: Zero parameters equal the base
- **WHEN** the residual with all parameters zero plays an episode at 12 or at 42 producers
- **THEN** every producing action equals the base producer's on the same episode

### Requirement: Agent-local, past-only information
The residual's features SHALL come only from the producer's own entity view and its own per-episode history of past cycles. They SHALL hold no other agent's same-cycle routing or bids (ADR-0058). Episode state SHALL be keyed by the episode's system object, never by its address, so no state carries from one episode into the next.

#### Scenario: Back-to-back episodes
- **WHEN** two episodes run back to back through one policy object in one process
- **THEN** each episode's actions equal those of a fresh policy object on the same episode

### Requirement: Size- and topology-independent parameters
One parameter vector SHALL serve every producer and every input row. The parameter count SHALL NOT depend on the number of producers, inputs or the topology. A policy trained at 12 producers SHALL run unchanged at 42.

#### Scenario: Same parameters at 42
- **WHEN** parameters trained at 12 producers are loaded on a 42-producer world
- **THEN** the policy runs without change and with zero contract failures

### Requirement: Pre-registered ladder
Training and reads SHALL follow the registered ladder:
1. the base's noise floor and guard margin;
2. a gradient probe;
3. a pilot;
4. the main runs;
5. validation at 12 (deciding) and at 42 (reported);
6. a held-out read.
Each step's readings and thresholds SHALL be frozen, by SHA-256, before its data exist, and a run whose pins differ SHALL be refused. Any reading other than the one gating the next step SHALL stop the ladder for a diagnosis.

#### Scenario: Changed pre-registration refused
- **WHEN** a real run is started with a pre-registration whose SHA-256 differs from the frozen one
- **THEN** the run is refused before any job

### Requirement: Side-effect guard relative to the base
The training guard SHALL compare each candidate's run-down and starvation with the base producer's (not with `entity_planner`), with a margin measured on the base's own re-keyed replays so that the base flags itself on at most 5% of episodes.

#### Scenario: Margin measured before training
- **WHEN** the noise-floor run of the base completes
- **THEN** the margin and the base's self-flag rate are recorded, and the rate is at most 5%

### Requirement: Held-out confirmation and success bar
A candidate SHALL count as confirmed only if, on held-out worlds with fresh seeds, it beats the base producer with MWW beyond the noise floor at 12 producers, is not below the floor at 42, and adds no deaths and no side effects beyond the margin. The held-out read SHALL be recorded in a ledger with a pending entry committed before launch, and SHALL never be re-run automatically.

#### Scenario: One held-out read
- **WHEN** a held-out read has been closed in the ledger as done or invalid
- **THEN** a second real held-out read for the same key is refused

### Requirement: Joint phase gated
Learning the orderer, or both roles on a shared potential, SHALL start only after a confirmed held-out producer read and a diagnostic showing orderer-side headroom given the learned producer. Its plan SHALL be a separate update of this change.

#### Scenario: Joint work before confirmation
- **WHEN** no confirmed held-out producer read exists
- **THEN** no joint-learning task is started
