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
  train/    smoke PPO + parallel fan-out + boundary_grid driver (optional: needs torch)
  config/   phase1.json
docs/        ADRs (adr/), deep-research-report (6-phase curriculum), requirements
experiments/ empirical runs + results, one dir per experiment — see experiments/README.md
PROGRESS.md  active work-in-progress status, handoff notes, blockers, next actions
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

## Where training + experiments run (HARD RULE)
**All real training AND every experiment run on cloud infra (Runpod), never locally.** Wrap every real run with `run_pod_training.py` (see "Remote GPU training" / "Experiments" below — `--cpu` for CPU pods). "Real" = anything beyond a smoke check: `boundary_grid`, `warm_start`/`clone_fidelity` DAgger, `producer_improve`, full `smoke_ppo`/`parallel` sweeps, any multi-topology or multi-iter run.

**Experiments go to the cloud.** Any run that materializes an `EXP-NNNN` result — the full multi-seed / multi-task / multi-topology run that produces `summary.csv` + `artifacts/` (training, boundary/transfer/curriculum matrices, replay suites, acceptance sweeps) — runs on Runpod, wrapped with `run_pod_training.py --pull` to bring the EXP dir back. The **only** things that stay local: **smoke checks** (`smoke_ppo --iters` ≲ 50, a single-cell sanity check, a fast `--dry-run`) and **quick one-shot diagnostics/evals** (`signal_check`, `producer_rungs`, `swap_eval`, a single-seed trace). Build/TDD the experiment driver locally against a tiny subset (`--dry-run`, 1–2 cells, ≤1 seed) to prove it runs, then launch the **full** run on the cloud. If unsure, it goes to the cloud.

## Parallel training (use the cores)
One PPO run is single-threaded: rollout `collect` is a sequential Python round-robin over `num_envs`, and inference is batch-1 — extra cores do nothing for one run. Pin BLAS to 1 thread (`--threads 1`, the default) and get throughput by fanning out **independent** runs, one process per core, via `tokenomics.train.parallel`:
```
# grid (fixed depth, climbing demand rung)
../.venv/bin/python -m tokenomics.train.parallel --workers 8 \
  --common "--iters 400 --steps 1024 --topology-depth 3" \
  --job d3_p00 "--max-diff 0.00" --job d3_p25 "--max-diff 0.25"
# seed sweep: one process per seed
../.venv/bin/python -m tokenomics.train.parallel --workers 8 --seeds 0-7 --common "..."
```
Each child auto-gets `--threads 1`; logs stream to `<log-dir>/<name>.log`; `--dry-run` prints commands first. Prefer this over `--num-seeds` (in-process, sequential) for multi-run work.

## Experiments (boundary grid)
A full experiment = a question + materialized configs + many runs + a result, kept as **one dir** under `experiments/`. **Read `experiments/README.md` first** — it owns the layout/convention (`spec.json`, `RESULT.md`, `summary.csv`, `configs/` tracked; `artifacts/` = checkpoints/logs via **git-lfs**). ADRs = design decisions; experiments = empirical results; `VERIFICATION.md` = open questions linking out.

`tokenomics.train.boundary_grid` is the driver (don't hand-roll raw `parallel`/`smoke_ppo` for this). It materializes the fixed depth-1/2/3 scale topologies, fans out from-scratch PPO across the depth × quantity grid (via `tokenomics.train.parallel`), runs the policy-swap probe per checkpoint, and **emits the whole experiments-dir layout directly** — point `--out-dir` at a fresh `EXP-NNNN` dir:
```
# local
../.venv/bin/python -m tokenomics.train.boundary_grid --iters 400 --steps 1024 \
  --workers 6 --held-out-seeds 5 --swap-seeds 5 --out-dir ../experiments/EXP-0002-<title>
# subset / fast check: --cells d3_p00,d3_p25 --dry-run
```
Then fill in `RESULT.md`'s Conclusions and add an index row to `experiments/README.md`. Worked example: `experiments/EXP-0001-boundary-grid/` (depth is the wall, producer is the blocker).

**Off-box**: wrap the same command with the Runpod driver below — `--train` runs boundary_grid on the pod, `--pull` brings the EXP dir back:
```
python run_pod_training.py --pod-id <id> \
  --train "-u -m tokenomics.train.boundary_grid --iters 400 --steps 1024 --workers 6 \
           --held-out-seeds 5 --swap-seeds 5 --out-dir /workspace/tokenomics/boundary_runs/grid" \
  --pull "/workspace/tokenomics/boundary_runs/grid:./experiments/EXP-NNNN-<title>"
```
`-u` keeps the driver's stdout unbuffered (per-cell iter logs already flush live to their files). Pod (25 vCPU) runs all 6 cells concurrently; full grid ≈ 15 min.

## Remote GPU training (Runpod)
`run_pod_training.py` (repo root) drives a full off-box run on a Runpod GPU pod: **start pod → rsync code → venv+install → train → stop pod**. Pod stop runs in a `finally`, so the pod is stopped even on error / Ctrl-C (the whole point: don't pay for an idle GPU). Needs `runpodctl` authed (`runpodctl doctor`), plus `ssh`+`rsync` locally.
```
# default = smoke PPO; stops pod when done
python run_pod_training.py --pod-id <id>
# real run: save a checkpoint, pull it back, then stop
python run_pod_training.py --pod-id <id> \
  --train "-m tokenomics.train.smoke_ppo --iters 400 --steps 1024 --save /workspace/model.pt" \
  --pull /workspace/model.pt:./model.pt
```
- `--train "<args>"` = args handed to the pod's `.venv/bin/python` (default: smoke PPO). `--pull REMOTE:LOCAL` (repeatable) rsyncs files back before stopping.
- Pod addressed by **id**; ip/port resolved fresh each run via `runpodctl ssh info` (they change across restarts). Code goes to `/workspace/tokenomics`; pod venv is `--system-site-packages` to reuse the image's CUDA torch.
- Skips: `--no-stop` (leave pod up for inspection), `--no-sync`, `--no-install`. `--ssh-key` / `--ready-timeout` override defaults.
- **Caveat**: training runs in the **foreground** over SSH — the script must stay connected for the whole run. Fine for short/smoke runs; for long runs that must survive a laptop sleep/disconnect, switch to a detached `nohup`+poll model first (not yet implemented).

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
- Ongoing progress / handoff / blockers / next actions → update `PROGRESS.md`.
- Empirical findings → write/update the relevant `experiments/EXP-NNNN-*` record and index.
- Roadmap or target changes → update the relevant design/plan document.

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
An agent that owns capacity of one or more types and assembles products to fulfil orders. It declares the products it makes via an authoritative **`produces`** set (ADR-0028) — it must own every capacity type each listed product's recipe requires, but owning the capacity does not by itself make it eligible. To assemble a composite product it buys the required input products from other producing agents and applies its own capacity to the assembly. Decides each cycle what to produce, what inputs to buy, and whether to invest in new or expanded capacity. There is no designated producer for a product — the same product may be produced by multiple producing agents.
_Avoid_: Factory, producer, manufacturer

**Ordering Agent:**
An agent that, for each order it holds, splits the order's quantity (and the tokens backing it) across one or more eligible producing agents, handing each a sub-order it must fulfil as a whole. It never splits a product into its recipe components — that decomposition is the producing agent's job. A producing agent is eligible for a product when the product is in its **`produces`** set — an authored, authoritative list (ADR-0028), not derived from capacity ownership. Config validation guarantees a producer only lists products whose required capacity types it owns. Each consumer and each producing agent has a dedicated ordering agent.
_Avoid_: Distributor, allocator, broker

### Economy

**Token:**
A demand-estimation signal. Tokens carry no intrinsic value; they encode the relative priority of orders and drive auction allocation and cut-off prices. Not currency, not money — never accumulated as profit, only propagated and distributed. Tokens backing an order are burnt when that order is accepted: they signal demand, then leave circulation. Fresh tokens are emitted each cycle.
_Avoid_: Money, currency, coin, credit, value

**Emission:**
The system creates a fixed amount of tokens each cycle. Half goes equally to consumers (to back their orders as demand signal), half goes to the central investment fund.
_Avoid_: Minting, printing, generation

**Investment Fund:**
The half of each cycle's emission that funds all producer purchasing — buying input products for assembly and means of production for investment. Distributed to producers proportionally to their cut-off prices. The sole source of producer-to-producer token flow; producers never spend tokens of their own.
_Avoid_: Reserve, pool, treasury

### Production

**Capacity:**
Abstract units of a specific type that a producing agent can apply toward manufacturing. Capacity degrades each cycle and can be restored or expanded through investment.
_Avoid_: Power, resources, output

**Capacity Type:**
A named category of capacity (e.g., "capacity-1"). Different types are not interchangeable — each produces different components needed by process sheets.
_Avoid_: Resource type, skill

**Inventory:**
Stock of products a producing agent holds between cycles — inputs bought for future assembly and finished products awaiting delivery. Inventory enables pipelined production (assemble from stock while replenishing) and breaks circular recipe dependencies. Following the Theory of Constraints, carrying inventory is a liability: overproducing ahead of demand is discouraged — penalised as a negative training reward in the env layer (not via tokens, which carry no value). Total inventory is bounded by a limit on the number of product units held; that limit can be expanded by investment.
_Avoid_: Stock, warehouse

**Process Sheet:**
A recipe that defines, per unit of a product, how much capacity of each type is consumed to produce it and which input products (and how many) it consumes. Primal products list only capacity; composite products also list input products. Fixed and exogenous — does not change during simulation.
_Avoid_: Recipe, blueprint, technology card (технологическая карта)

**Product:**
The output of a completed order, delivered assembled as a whole. Defined by its process sheet. A **primal** product requires no input products (e.g. a natural resource) — only capacity to produce. A **composite** product requires other products as inputs, plus capacity to assemble them. Products are consumed by consumers (fulfilling demand) or by producing agents (as inputs, or for restoration/upgrade).
_Avoid_: Good, item, output

**Degradation:**
The automatic reduction of a producing agent's capacity each cycle, calculated as a percentage of max capacity. Creates ongoing pressure to invest in restoration.
_Avoid_: Decay, wear, depreciation

**Investment:**
A producing agent's purchase of a means-of-production product to improve its productive base, funded by the central investment fund. Three types: restoration (recover degraded capacity), upgrade (expand max capacity of an owned type), and inventory expansion (raise the inventory-unit limit). Creating a brand-new capacity type a producer did not own is out of scope for Phase 1.
_Avoid_: Capital expenditure, capex

**Restoration:**
An investment that recovers a producing agent's capacity toward its max capacity. Requires a specific product (which itself requires capacity from other producers).
_Avoid_: Repair, maintenance, recovery

**Upgrade:**
An investment that increases a producing agent's max capacity (and current capacity). Requires a specific product. Unlike restoration, permanently expands the ceiling.
_Avoid_: Expansion, improvement, enhancement

### Exchange

**Bid:**
A producing agent's **internal** auction unit for a sub-order: the whole capacity bundle the sub-order requires (units of every capacity type in the recipe) and the tokens backing it. Its price is `tokens / (total capacity units summed across all required types)`. The ordering agent does **not** build bids — it routes a capacity-free sub-order (product + quantity + tokens, ADR-0029) and the producing agent expands it into a bid from its own process sheet.
_Avoid_: Offer, request, order line

**Cut-off Price:**
The lowest bid price (tokens per total-capacity-unit) among all bids accepted by a producing agent in a cycle. Used to prioritize investment fund distribution and signals market tightness.
_Avoid_: Market price, clearing price, minimum price

**Auction:**
The mechanism by which a producing agent allocates its capacity to bids. Bids are sorted by price descending and accepted greedily from the highest down. Acceptance is atomic: a bid clears only if the producer has enough remaining capacity of every type the bundle requires.
_Avoid_: Exchange, marketplace, market

### Orders

**Order:**
A demand for a quantity of a product as a whole, backed by tokens. Capacity and input-product requirements scale with the quantity. Can be a consumer order (from a consumer) or an investment order (from a producing agent).
_Avoid_: Request, demand, task

**TTL (Time to Live):**
The maximum number of cycles an order can remain in progress before being rejected. Currently fixed at 2 cycles.
_Avoid_: Timeout, deadline, expiry

### Simulation

**Cycle:**
One discrete time step of the simulation. Within a cycle: tokens are emitted, orders are placed, producing agents decide on investments, ordering agents distribute tokens, production runs, orders complete or reject.
_Avoid_: Turn, tick, round, step

**Score:**
The per-cycle reward signal. Composed of: +10 per completed order, -1 per order still in progress, -5 per rejected order, minus a flat holding penalty per inventory unit held. The inventory penalty lives in the env layer (a training reward), not in the domain. Higher is better.
_Avoid_: Points, penalty, fitness

**Episode:**
A complete simulation run from initialization to termination. Ends after a fixed number of cycles (default 100) or on system death (some capacity type reaches zero across all producers).
_Avoid_: Game, session, run

**System Death:**
The irrecoverable state where some product becomes permanently unmakeable — no producer that **declares** it (`produces`) still holds every capacity type its recipe requires (ADR-0030). Because `produces` is authoritative, capacity is not interchangeable across owners: a type held by a producer that does not declare the product cannot supply it. Triggers immediate episode termination. Can cascade due to circular dependencies between producers.
_Avoid_: Collapse, failure, game over
