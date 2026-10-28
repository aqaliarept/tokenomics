## ADDED Requirements

### Requirement: Demand patterns
A world SHALL be configurable with at most one demand pattern on its randomized consumers, of one
of four kinds: burst (random orders multiplied by an integer factor), moving hit (rank-skewed
popularity whose top product moves periodically), seasonal (quantities scaled by a fixed periodic
envelope) and regime switch (a favoured half of the products swapped with its complement at
switch points).

#### Scenario: Burst factor
- **WHEN** a world has a burst pattern with factor range 5 to 10 and an order is drawn as a burst
- **THEN** its quantity is its unpatterned quantity times an integer factor between 5 and 10, and
  its tokens are unchanged

#### Scenario: Hit moves
- **WHEN** a world has a moving-hit pattern with period 10
- **THEN** the most popular product of cycles 1 to 10 differs from that of cycles 11 to 20

#### Scenario: Seasonal envelope
- **WHEN** two episodes with different seeds run on the same world with a seasonal pattern
- **THEN** both follow the same envelope, and their quantities differ only by the seeded rounding

#### Scenario: Regime switch
- **WHEN** a world has a regime-switch pattern with one switch at cycle 40
- **THEN** the favoured products of cycles before 40 are the unfavoured ones from cycle 40 on

### Requirement: Patterns replace drift on randomized consumers only
A demand pattern SHALL replace demand drift on the consumers it applies to, and SHALL NOT apply to
consumers with an explicit cyclic schedule.

#### Scenario: Cyclic consumer untouched
- **WHEN** a world with a burst pattern has a consumer with an explicit cyclic schedule
- **THEN** that consumer's orders equal its schedule in every episode

### Requirement: Patterned orders keep their tokens
The tokens backing a consumer's order SHALL stay that consumer's emission share whatever a demand
pattern does to the order's quantity.

#### Scenario: Seasonal peak
- **WHEN** a seasonal pattern doubles a consumer's quantity in a cycle
- **THEN** the order is backed by the same tokens as without the pattern

### Requirement: Demand patterns draw on their own streams
Each demand-pattern kind SHALL draw from its own stream derived from the episode seed. A pattern
SHALL NOT change any shock schedule, the drift stream, or the draws of consumers it does not apply
to, and a shorter horizon's patterned demand SHALL be a prefix of a longer one's.

#### Scenario: Shocks unchanged by a pattern
- **WHEN** a shocked world is run with and without a demand pattern for the same episode seed
- **THEN** its shock schedule is byte-identical in both

#### Scenario: Prefix
- **WHEN** a patterned world runs the same episode seed with horizons 10 and 20
- **THEN** the demand of cycles 1 to 10 is identical in both

### Requirement: Demand patterns are unannounced
No observation feature SHALL expose a demand pattern's kind, parameters, phase, hit position or
switch points; agents SHALL observe a pattern only through the orders placed. Enabling a pattern
SHALL NOT change any observation or action space.

#### Scenario: Spaces with and without a pattern
- **WHEN** the same world is built with and without a demand pattern
- **THEN** every agent's observation and action space is identical in both

### Requirement: Demand patterns are paired and digested
The demand digest SHALL cover every demand-pattern draw, and the shock-free twin of a patterned
shocked world SHALL keep the pattern, so that its demand digest equals the shocked episode's.
A demand pattern SHALL be part of the configuration's fingerprint when present.

#### Scenario: Twin keeps the pattern
- **WHEN** a shocked world with a burst pattern and its shock-free twin run the same episode seed
- **THEN** both receive the same bursts and their demand digests are equal

#### Scenario: Burst is demand, not a shock
- **WHEN** resilience is reported for a shocked world with a burst pattern
- **THEN** no burst appears as a shock in the resilience rows

### Requirement: No pattern, no change
A configuration without a demand pattern SHALL produce exactly the episodes, statistics and
fingerprint it produced before demand patterns existed.

#### Scenario: Legacy world
- **WHEN** any existing configuration is run after demand patterns are added
- **THEN** every episode statistic and its fingerprint match the run before the addition
