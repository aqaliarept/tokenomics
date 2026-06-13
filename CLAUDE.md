# CLAUDE.md

Caveman style here. Terse. Keep technical terms exact.

## What this is
Cooperative multi-agent **RL environment** for a tokenomics economy game. Agents learn to fulfil consumer demand by distributing tokens + managing production capacity. Python only. All work lives in `tokenomics/`.

## Map
```
tokenomics/tokenomics/
  domain/   pure sim logic (no RL, no IO, no logging)
  env/      PettingZoo AEC adapter: spaces, encode, decode, aec_env
  eval/     heuristic baselines + metrics harness (signal check)
  train/    smoke PPO (optional: needs torch)
  config/   phase1.json
docs/        ADRs (adr/), deep-research-report (6-phase curriculum), requirements
CONTEXT.md   domain glossary — READ FIRST, use these exact terms
phase1-implementation-plan.md   roadmap + status + RL target numbers
```

## Commands (run from `tokenomics/`, venv at repo `.venv`)
```
../.venv/bin/pip install -e .          # install (torch extra: .[train])
../.venv/bin/python -m pytest -q       # tests (must stay green)
../.venv/bin/python -m tokenomics.eval.signal_check    # baseline table
../.venv/bin/python -m tokenomics.train.smoke_ppo --iters 50
```

## Architecture rules
- **`domain/` stays pure.** No RL, no numpy, no IO, no logging. Just the simulation.
- **`env/` = RL adapter only.** All gym spaces, normalization, reward live here — never in domain.
- **Two shared policies (CTDE)**: one for all producing agents, one for all ordering agents.
- Details (obs/action shapes, reward design) live in ADRs — read `docs/adr/`, don't reinvent.

## Config
- Env built from `config/phase1.json`: `cycleEmission`, `processSheets`, `producerConfigs`, `consumers`.
- `TokenomicsAECEnv(config=None, horizon=100)` — `config=None` loads `phase1.json`. `horizon` = max cycles before truncation.
- Episode ends on **system death** (all producers at 0 capacity, → termination) OR **horizon** reached (→ truncation).
- Domain is **deterministic** (scripted consumers, no RNG). Same policy + same config → same episode. Seed only changes **stochastic** policies (random baseline, RL policy).

## Evaluation (`eval/`)
- **Baselines** = non-RL policies, white-box (read `env.system` directly). Producing: `noop`, `always_restore`, `always_both`, `restore_when_low`, `random`. Ordering: `even_split`, `proportional`, `random`. In `eval/baselines.py`.
- `run_episode(env, prod_policy, order_policy, seed)` → drives one episode, returns metrics from `env.episode_stats` + `capacity_trace`.
- `evaluate(prod, order, seeds, horizon)` → runs many seeds, averages (mean, std).
- `signal_check` → runs a matrix of (producing × ordering) combos → prints comparison table.
- **Metrics**: `cycles` (survival len), `dead%`, `cons✓` (consumer orders fulfilled = the goal), `return`, `capmin` (lowest capacity = collapse warning).
- **Why it exists**: prove env poses a learning problem — a sensible policy must clearly beat do-nothing. Strongest baseline = the bar RL must clear (see plan for target numbers).

## Conventions
- New design decision → write an ADR in `docs/adr/`.
- New baseline → add to the dicts in `eval/baselines.py` so `signal_check` picks it up.
- New domain term → add to `CONTEXT.md`.
- Status/findings changed → update `phase1-implementation-plan.md`.

## Code style
- `from __future__ import annotations` top of every module.
- **Type annotations required.** Every function/method: annotate all params AND return (`-> None` for procedures). Callables get a type alias (e.g. `Policy`). No untyped `def`.
- Value types = `@dataclass`. Type aliases in `domain/common.py`.
- snake_case. Match surrounding code. No new deps without reason.
- Docstrings explain **why** + cite the ADR, not what. Keep short.

## Tests
- `pytest`, one file per layer (`test_domain_*`, `test_simplex`, `test_env_api`, `test_eval`, `test_train_smoke`).
- torch tests guard with `pytest.importorskip("torch")`.
- `test_eval::test_signal_exists` = guardrail: a good policy must beat do-nothing. Keep it passing.
- Touch env? `pettingzoo.test.api_test` runs in `test_env_api`.


## Domain Language

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
