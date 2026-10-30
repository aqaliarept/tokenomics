## ADDED Requirements

### Requirement: Region samples
The world-space generator SHALL build region samples: worlds whose continuous levers come from the
atlas lever distribution's Latin hypercube for a named stream and size, and whose categorical
levers (choice, shocks, demand pattern) and shock levers are fixed by the caller. Every region world
SHALL be recorded like any generated world, and the sample SHALL be a function of (stream, size,
pool size, fixed levers) only.

#### Scenario: Fixed choice overrides the hypercube
- **WHEN** a region sample of 42-producer worlds is built with choice fixed at fourteen makers per
  product
- **THEN** every world declares each product by fourteen producers, whatever the hypercube's choice
  column holds

#### Scenario: Same stream, same sample
- **WHEN** a region sample is built twice with the same stream, size, pool size and fixed levers
- **THEN** both samples hold the same worlds with the same configuration hashes

#### Scenario: Different streams do not share seeds
- **WHEN** region samples are built on two different stream names at the same size
- **THEN** no world seed appears in both
