# Tokenomics Simulation

A cooperative multi-agent economic simulation where software agents learn to efficiently fulfill consumer demand by distributing tokens and managing production capacity. The system serves as an RL training environment.

## Language

### Actors

**Consumer:**
An external entity that receives tokens via emission and places orders for products. Not an agent — its behavior is part of the environment.
_Avoid_: Customer, user, player

**Producing Agent:**
An agent that owns capacity of a specific type. Decides whether to invest in restoration or upgrade each cycle. Each producing agent provides exactly one capacity type.
_Avoid_: Factory, producer, manufacturer

**Ordering Agent:**
An agent that distributes tokens across producing agents to fulfill orders. Each consumer and each producing agent has a dedicated ordering agent.
_Avoid_: Distributor, allocator, broker

### Economy

**Token:**
The unit of exchange in the system. Tokens carry information about relative value of orders and regulate resource distribution among agents. Not currency — a signal mechanism.
_Avoid_: Money, currency, coin, credit

**Emission:**
The system creates a fixed amount of tokens each cycle. Half goes equally to consumers, half goes to the investment fund for producing agent maintenance and upgrades.
_Avoid_: Minting, printing, generation

**Investment Fund:**
The half of each cycle's emission reserved for producing agent restoration and upgrade orders. Distributed proportionally to producers' cut-off prices.
_Avoid_: Reserve, pool, treasury

### Production

**Capacity:**
Abstract units of a specific type that a producing agent can apply toward manufacturing. Capacity degrades each cycle and can be restored or expanded through investment.
_Avoid_: Power, resources, output

**Capacity Type:**
A named category of capacity (e.g., "capacity-1"). Different types are not interchangeable — each produces different components needed by process sheets.
_Avoid_: Resource type, skill

**Process Sheet:**
A recipe that defines how much capacity of each type is required to produce a given product. Fixed and exogenous — does not change during simulation.
_Avoid_: Recipe, blueprint, technology card (технологическая карта)

**Product:**
The output of a completed order. Defined by its process sheet. Products are consumed by consumers (fulfilling demand) or by producing agents (for restoration/upgrade).
_Avoid_: Good, item, output

**Degradation:**
The automatic reduction of a producing agent's capacity each cycle, calculated as a percentage of max capacity. Creates ongoing pressure to invest in restoration.
_Avoid_: Decay, wear, depreciation

**Restoration:**
An investment that recovers a producing agent's capacity toward its max capacity. Requires a specific product (which itself requires capacity from other producers).
_Avoid_: Repair, maintenance, recovery

**Upgrade:**
An investment that increases a producing agent's max capacity (and current capacity). Requires a specific product. Unlike restoration, permanently expands the ceiling.
_Avoid_: Expansion, improvement, enhancement

### Exchange

**Bid:**
A token allocation from an ordering agent to a specific producing agent for a specific order. Contains the capacity requested and tokens offered.
_Avoid_: Offer, request, order line

**Cut-off Price:**
The lowest tokens-per-capacity-unit ratio among all bids accepted by a producing agent in a cycle. Used to prioritize investment fund distribution and signals market tightness.
_Avoid_: Market price, clearing price, minimum price

**Auction:**
The mechanism by which a producing agent allocates its capacity to bids. Bids are sorted by tokens-per-capacity-unit descending; capacity is allocated greedily from highest bidder down.
_Avoid_: Exchange, marketplace, market

### Orders

**Order:**
A demand for a product, backed by tokens. Has parts corresponding to each required capacity type. Can be a consumer order (from a consumer) or an investment order (from a producing agent).
_Avoid_: Request, demand, task

**TTL (Time to Live):**
The maximum number of cycles an order can remain in progress before being rejected. Currently fixed at 2 cycles.
_Avoid_: Timeout, deadline, expiry

### Simulation

**Cycle:**
One discrete time step of the simulation. Within a cycle: tokens are emitted, orders are placed, producing agents decide on investments, ordering agents distribute tokens, production runs, orders complete or reject.
_Avoid_: Turn, tick, round, step

**Score:**
The per-cycle reward signal. Composed of: +10 per completed order, -1 per order still in progress, -5 per rejected order. Higher is better.
_Avoid_: Points, penalty, fitness

**Episode:**
A complete simulation run from initialization to termination. Ends after a fixed number of cycles (default 100) or when all producers reach zero capacity.
_Avoid_: Game, session, run

**System Death:**
The irrecoverable state where all producing agents have zero capacity. Triggers early episode termination. Can cascade due to circular dependencies between producers.
_Avoid_: Collapse, failure, game over
