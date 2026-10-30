# world-space Specification

## Purpose

Defines how a world is described as a point on named axes of a world space, how the generator
realizes and records such a world, and that the neutral point of every axis reproduces the
existing worlds exactly.

## Requirements

### Requirement: Worlds are points on named axes
The world-space generator SHALL build a world from a size, a world seed and one value per lever of
these axes: choice, capacity tightness, order size against capacity, producer heterogeneity,
shocks (capacity hits and producer outages) and demand pattern.

#### Scenario: One lever per axis
- **WHEN** a world is requested without a value for one of the axes' levers
- **THEN** the generator uses that lever's neutral value and records it

### Requirement: Every world is recorded
For every world it builds, the generator SHALL record the lever values, the seed, the generator
version, the measured coordinates (tightness, order size, heterogeneity spreads, shock intensity,
demand concentration and volatility), the declarers per product and the configuration's hash.

#### Scenario: World record
- **WHEN** the generator builds a world
- **THEN** its record holds every lever value, the seed, the generator version, every measured
  coordinate, the declarers per product and the configuration hash

#### Scenario: Same record, same world
- **WHEN** a world is rebuilt from its record
- **THEN** the configuration hash equals the recorded one

### Requirement: Neutral levers reproduce existing worlds
With every lever at its neutral value, the generator SHALL produce a configuration identical to
the existing family generator's for the same family, size, fanout, regime and seed, with the same
fingerprint and the same episode statistics.

#### Scenario: Neutral point
- **WHEN** a world is built with every lever neutral
- **THEN** its configuration, fingerprint and episode statistics equal those of the existing family
  world with the same parameters

### Requirement: Levers do not move the topology
Changing a capacity, heterogeneity, quantity or shock lever SHALL NOT change the world's recipes,
declared products or any value drawn before the lever is applied.

#### Scenario: Heterogeneity changed
- **WHEN** the same world seed is built with capacity spread at its neutral value and at zero
- **THEN** both worlds have identical recipes and declared products, and in the zero-spread world
  every producer of a layer has the same capacity

### Requirement: Sizes stay within the scale target
The world-space generator SHALL build only 12- and 42-producer worlds.

#### Scenario: Larger size requested
- **WHEN** a world with 120 producers is requested from the world-space generator
- **THEN** the generator refuses, naming the size

### Requirement: Every generated world keeps the agent contract
Every world the generator can build SHALL keep every observation and action inside its space for
the existing agents, so that existing checkpoints load and act on it.

#### Scenario: Lever corner
- **WHEN** a world is built with every lever at an end of its range and played by an existing
  stack
- **THEN** the contract monitor records no failure
