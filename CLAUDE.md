# CLAUDE.md

Caveman style here. Terse. Keep technical terms exact.

## Code navigation (HARD RULE)
**Always use the LSP tool** for code intelligence: go-to-definition, find-references, hover (types/docs), symbol search. Never use grep/find as a substitute for these. Load the tool first with `ToolSearch` query `select:LSP` if not yet in scope.

## What this is
Cooperative multi-agent **RL environment** for a tokenomics economy game. Agents learn to fulfil consumer demand by distributing tokens + managing production capacity. Python only. All work lives in `tokenomics/`.

## Map
```
tokenomics/tokenomics/
  domain/   pure sim logic (no RL, no IO, no logging)
  env/      PettingZoo AEC adapter: spaces, encode, decode, aec_env
  eval/     heuristic baselines + metrics harness (signal check)
  train/    scorer distillation, flat MLP actors + BC/DAgger, champion iteration (optional: needs torch)
  config/   phase1.json
docs/        ADRs (adr/), deep-research-report (6-phase curriculum), requirements
experiments/ empirical runs + results, one dir per experiment — see experiments/README.md
experiments/HISTORY.md  compacted record of all 49 experiments, grouped by arc
ARCHITECTURE.md  how training/eval/improvement fit together
Domain glossary  the "Domain Language" section at the bottom of THIS file — use these exact terms
```

## Commands (run from `tokenomics/`, venv at repo `.venv`)
```
../.venv/bin/pip install -e .          # install (torch extra: .[train])
../.venv/bin/python -m pytest -q       # tests (must stay green)
../.venv/bin/python -m tokenomics.eval.signal_check    # baseline table
```

Code for closed or rejected arcs — the PPO trainer, the curriculum game-master, the
marginal-value router, the graph (GNN) branch, the EXP-0043 surrogate — is removed from
this branch and preserved on branch `archive-1` (commit `7aeef1f`). Restore from there only
with a diagnosed reason to reopen that arc (see "Root cause before next strategy").

## Root cause before next strategy (HARD RULE)
An experiment failed or underperformed → do **NOT** propose or start the next strategy/axis/arc until the failure cause is **diagnosed with evidence** (trace, closure-by-split, ablation — not a hypothesis) and written down (the EXP `RESULT.md`, and an arc paragraph in `experiments/HISTORY.md` if it closes or opens a branch). When competing explanations demand different responses (e.g. distill-fidelity wall → fix training vs generalization gap → mine denser), run the **discriminating diagnostic first**; it decides the move. Every new-strategy proposal must name the diagnosed cause it addresses. **Check `docs/DIAGNOSTICS.md` first** — the registry of known problems → verification checks (plain-English problem statements, discriminating check, outcome branches); if the symptom matches an entry, run that check before inventing a new diagnostic. New cause diagnosed → add a registry entry. (Failure seen: EXP-0034 candidate captured ~15% of mined headroom; jumping to "mine more cells" without knowing WHY the harvest is low would have spent pods on the wrong fix.)

## Explain experiments simply (RULE)
When reporting an experiment/diagnostic result to the user, explain it in **plain English first**: what question was asked, what was measured, what the numbers mean, and what changes because of them. Full sentences; define any invented term (e.g. "harvest rate", "closure") the first time it is used; assume the reader did not watch the pipeline run. Compressed jargon summaries and metric tables belong in `RESULT.md`/`HISTORY.md` — the chat explanation must stand alone without them.

## File links in chat (RULE)
When referencing a file to the user, always give the **absolute path** (e.g. `/srv/research/tokenomics/openspec/changes/<name>/specs/<cap>/spec.md`, optionally `:line`). Never a path relative to the repo root, the package dir, or a change dir — the working directory shifts between `tokenomics/`, `tokenomics/tokenomics/` and the repo root, so relative links resolve wrong. Applies to tables and lists too. (Failure seen: spec files listed as `specs/project-mission/spec.md` — unclickable.)

## Where training + experiments run (HARD RULE)
**All real training AND every experiment run on cloud infra (Runpod), never locally.** Wrap every real run with `run_pod_training.py` (see "Remote GPU training" / "Experiments" below — `--cpu` for CPU pods). "Real" = anything beyond a smoke check: `clone_fidelity` DAgger, scorer distillation, EXP-0042 bounded-rollout search, any multi-topology or multi-iter run.

**Experiments go to the cloud.** Any run that materializes an `EXP-NNNN` result — the full multi-seed / multi-task / multi-topology run that produces `summary.csv` + `artifacts/` (training, transfer matrices, replay suites, acceptance sweeps) — runs on Runpod, wrapped with `run_pod_training.py --pull` to bring the EXP dir back. The **only** things that stay local: stage-appropriate smoke checks (a replay of 1-2 archived tasks, a single-cell sanity check, a fast `--dry-run`) and **quick one-shot diagnostics/evals** (`signal_check`, `producer_trace`, a single-seed trace). Build/TDD the experiment driver locally against a tiny subset (`--dry-run`, 1–2 cells, ≤1 seed) to prove it runs, then launch the **full** run on the cloud. If unsure, it goes to the cloud.

**Classify by COMPUTE PROFILE + PURPOSE, not task count** (this is where the rule gets broken). "1-2 archived tasks" is necessary but NOT sufficient for local — a run stays local only if it is *also* fast AND throwaway. It goes to the **cloud** if ANY of these is true:
- it materializes a result you will base a **go/no-go decision** on (representability check, gate, acceptance read) — that's a verification run, not a diagnostic;
- it **trains/distills to convergence** (real `iters`/`epochs`, not a handful) or runs **full-horizon per-cycle search / rollouts** (`search_orderer_episode`, `mpc_search`, planner rollouts, deepcopy-per-candidate);
- it sweeps `eval_seeds > 1`, or spans **multiple tasks / topologies / seeds**;
- it would take more than ~1-2 min wall-clock.

A genuine local smoke has ALL knobs pinned tiny: `horizon ≤ ~12`, `iters ≤ ~5`, `eval_seeds ≤ 1`, 1-2 tasks, finishes in **seconds**. If any knob sits at a "real" value, it is the real run → cloud.

**Naming trap:** calling something a "smoke" — in your plan, a function name, a CLI flag — does NOT make it one. Check the profile against the list above, never the label you gave it. (Failure seen: a `run_representability_smoke` at `--iters 300 --eval-seeds 3 --horizon 100` over 2 tasks with per-cycle search — ~14 heavy search episodes, 18+ min — was run locally because it was *called* a smoke and touched only 2 tasks. Both wrong: it was a decision-gating verification run and belonged on the pod.)

## Parallel work (use the cores)
Inference here is batch-1 and rollouts are sequential Python, so extra cores do nothing for a
single run. Pin BLAS to 1 thread per worker and get throughput by fanning out **independent**
runs, one process per core. `eval.bounded_rollout_orderer --workers N` does this internally
(one `ProcessPoolExecutor` worker per paired episode, `torch.set_num_threads(1)` per worker). Watch the load average against `nproc`: a worker pool
whose children each spawn threads oversubscribes the box and can stall it outright
(DIAGNOSTICS 9).

## Experiments
A full experiment = a question + materialized configs + many runs + a result, kept as **one dir** under `experiments/`. **Read `experiments/README.md` first** — it owns the layout/convention (`spec.json`, `RESULT.md`, `summary.csv`, `configs/` tracked; `artifacts/` = checkpoints/logs via **git-lfs**), including which pinned artifacts must never be deleted. **Read `experiments/HISTORY.md`** for what has already been tried and why each closed branch closed — check it before proposing an arc that was already run. ADRs = design decisions; experiments = empirical results.

Point the driver's `--out-dir` at a fresh `EXP-NNNN-<title>/` so it emits the whole
experiments-dir layout directly (`configs/`, `artifacts/`, `spec.json`, `summary.csv`,
a `RESULT.md` skeleton). Then fill in `RESULT.md`'s Conclusions, add an index row to
`experiments/README.md`, and — if the result closes or opens an arc — a paragraph in
`experiments/HISTORY.md`.

**Off-box**: wrap the same command with the Runpod driver below — `--train` runs it on the
pod, `--pull` brings the EXP dir back:
```
python run_pod_training.py --pod-id <id> --cpu --python python3.12 \
  --train "-u -m <driver module> ... --out-dir /workspace/tokenomics/runs/exp" \
  --pull "/workspace/tokenomics/runs/exp:./experiments/EXP-NNNN-<title>"
```
`-u` keeps the driver's stdout unbuffered (per-cell logs already flush live to their files).

## Remote GPU training (Runpod)
`run_pod_training.py` (repo root) drives a full off-box run: **start pod → rsync code → venv+install → train (detached) → pull results → stop pod**. Pod stop runs in a `finally`, so the pod is stopped even on error / Ctrl-C (the whole point: don't pay for an idle pod). Needs `runpodctl` authed (`runpodctl doctor`), plus `ssh`+`rsync` locally.

### Concrete steps (do these in order)
1. **Get pod id**: `runpodctl pod list`. Note the id; a stopped pod restarts automatically (its `/workspace` volume persists across stop/start).
2. **Stage input checkpoints INTO the synced tree** (the gotcha). The driver rsyncs the **inner package dir** `tokenomics/` (the `pyproject.toml` root, so `pip install -e .` works) — **NOT** the repo root. So repo-root `experiments/EXP-*/artifacts/*.pt` do **not** reach the pod. Any ckpt a run READS must be copied under `tokenomics/` first, e.g. `mkdir -p tokenomics/_pod_ckpts && cp experiments/.../foo.pt tokenomics/_pod_ckpts/`, then reference it on the pod as `/workspace/tokenomics/_pod_ckpts/foo.pt`. Delete `_pod_ckpts/` after. (Runs that only WRITE a ckpt on the pod — smoke/train — don't need this.)
3. **Pick the profile**: GPU (default, `runpod/pytorch:*` image, reuses system CUDA torch) or **CPU** (`--cpu`, `runpod/base:*` image, installs the `[train]` CPU-torch extra). CPU-bound work (search bars, distillation, batch-1 rollout) → `--cpu`; GPU buys nothing there. **On a CPU/base pod add `--python python3.12`** — the base image's python3.8 has no `ensurepip` and venv creation fails.
4. **Choose pod paths**: write outputs under `/workspace/...` (e.g. `--out-dir /workspace/tokenomics/runs/exp0042`) and `--pull REMOTE:LOCAL` them back into a fresh `experiments/EXP-NNNN-*` dir.
5. **Launch** (detached by default — survives SSH drop / laptop sleep; the client auto-reconnects and finishes when the remote exit marker appears; Ctrl-C just detaches):
```
# CPU experiment reading staged inputs, writing an EXP dir, pulled back
python run_pod_training.py --pod-id <id> --cpu --python python3.12 --verify-source-commit \
  --train "-u -m tokenomics.eval.bounded_rollout_orderer \
    --archive   /workspace/tokenomics/_pod_inputs/gate_archive.jsonl \
    --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt \
    --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0042" \
  --pull "/workspace/tokenomics/runs/exp0042:./experiments/EXP-NNNN-<title>"
# GPU smoke: default profile, prints the baseline table
python run_pod_training.py --pod-id <id> --train "-m tokenomics.eval.signal_check"
```
6. **Watch / reattach**: live progress mirrors to `train.live.log` (`tail -f`). If the client detaches, reconnect: `python run_pod_training.py --pod-id <id> --attach --remote-log /workspace/train.log --cpu`.
7. **After**: pod auto-stops (unless `--no-stop`); delete any staged `tokenomics/_pod_ckpts/`.

### Existing pod host stuck? Create a fresh pod (`--create-pod`)
When an existing pod's host keeps failing `pod start`, skip `--pod-id` and enrol a **fresh** pod, run, then **stop AND terminate** it (only a pod created by the run is ever removed — a `--pod-id` is only stopped):
```
python run_pod_training.py --create-pod --image <img-with-py3.11+torch+ssh> \
  --vcpu 16 --mem 32 --disk 30 \
  --train "-u -m tokenomics.eval.bounded_rollout_orderer ..." \
  --pull /workspace/tokenomics/runs/exp:./experiments/EXP-NNNN-title
```
- Spec flags default to a **16 vCPU / 32 GB / 30 GB CPU pod** (flavor `cpu5c`, the CPU-bound shape). Override `--cpu-flavor`, `--compute-type GPU --gpu-type ... --gpu-count N` for GPU; `--volume-size`/`--data-center`/`--network-volume-id`/`--community` as needed. `--clone-from <pod-id>` seeds the spec from an existing pod instead of the flags.
- **Create/clone go through the RunPod REST API, not `runpodctl`** — `runpodctl pod create` has no vCPU flag, so a CPU pod would come up at 2 vCPU. `create_pod` POSTs `rest.runpod.io/v1/pods` with `vcpuCount`+`cpuFlavorIds` (verified 16 vCPU/cpu5c); `--clone-from` reads the template via REST `GET /v1/pods/<id>` (only REST returns `cpuFlavorId`). API key from `RUNPOD_API_KEY` or `~/.runpod/config.toml` (never logged).
- **Pick the `--image` to skip installing Python/torch.** An image with python3.11+ **and** torch, run WITHOUT `--cpu` (default GPU profile → `--system-site-packages` venv reuses the image's Python+torch) means no `--python python3.12`/ensurepip workaround and no torch download. Image must ship SSH.
- `--keep-pod` = stop but don't terminate (inspect). Ctrl-C on a detached run leaves the created pod up (run continues) and prints the `runpodctl remove pod <id>` to clean up after `--attach`.

### Reference
- `--train "<args>"` = args handed to the pod's `.venv/bin/python` (default: `-m tokenomics.eval.signal_check`). `-u` = unbuffered stdout (live per-iter logs). `--pull REMOTE:LOCAL` (repeatable) rsyncs back before stopping.
- Pod addressed by **id** (existing) or created via `--create-pod`/`--clone-from`; ip/port resolved fresh each run via `runpodctl ssh info` (they change across restarts). Code lands at `/workspace/tokenomics`.
- Skips: `--no-stop` (leave pod up), `--no-sync` (code already on pod), `--no-install`, `--foreground` (legacy blocking stream). `--ssh-key` / `--ready-timeout` override defaults.

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
- **Gate harness** (`eval/gate.py` pure arithmetic + verdict, `eval/gate_cli.py` driver; ARCHITECTURE.md §7.2). Inputs: world registry `eval/registry/worlds_v1.json` (`eval/registry.py`), roster `eval/registry/roster_v1.json` (`eval/roster.py`), side-effect params file (X, Y, q; content-hash version). Report (cloud for real runs; `--split sealed` needs `--preregistration`):
  ```
  python -m tokenomics.eval.gate --mode report --subject portable --prod-ckpt P.pt --orderer-ckpt O.pt \
    --params side_effect_params_v1.json --split development --workers 16 \
    --source-commit {SOURCE_COMMIT} --out-dir EXP-NNNN-title          # on the pod; locally: --verify-source-commit
  python -m tokenomics.eval.gate --mode recompute --out-dir EXP-NNNN-title   # rebuild verdict from episodes.csv, MATCH/MISMATCH vs verdict.json
  python -m tokenomics.eval.gate --mode screen [--prune] --q 0.05 --workers 16 --source-commit {SOURCE_COMMIT} --out-dir EXP-NNNN-title
  ```
  Screen = roster only on development gate worlds; reserves run on demand (`--reserves on-demand`, default: later waves run only the reserves needed to replace zero-baseline worlds, fixed order; `eager` = all up front, same frozen files; `none` = dry runs only). Order: zero-baseline replacement → headroom per family×size (median argmax fill < registry `saturation_ceiling`; failure → exit 3, no `frozen/`) → optional pruning → X/Y = nearest-rank q-quantiles of the argmax member's run-down / fill rates. Writes candidate `frozen/{worlds,roster,side_effect_params}_v1.json`, `screening.json`, `headroom.csv`.
  Probe (task 10.2, D10 retuning evidence; never an acceptance read): `--mode probe --scales 1.0,0.9 --families dense --members a/b,c/d --replicates 1 --episodes-per-world 3` → per-scale `scale-<s>/` runs (generator refs carry `capacity_scale`, omitted when 1.0) + `probe_headroom.csv` + `probe.json`. Families carry a recorded `capacity_scale` (neutral 1.0) that multiplies capacity, `restores`, `increases`.
  A real report needs a frozen roster, a screened registry, a 40-hex source commit and no filters. Anything else needs `--dry-run` (smoke; marked "not an acceptance read" in `spec.json`/`RESULT.md`): filters `--sizes --families --max-worlds --max-members --episodes-per-world --replicates`; `--subject baseline:<producing>/<ordering>` runs without checkpoints. Rerun on the same `--out-dir` resumes only if every pin (inputs, source tree+commit, job grid) matches. Writes `spec.json` (pins, provenance), `episodes.csv`, `worlds.csv`, `regimes.csv`, `verdict.json`, `RESULT.md`.
- **Gate stats** (`episode_stats` keys): `consumer_ordered_units` / `consumer_fulfilled_units` (per consumer, orders placed in played cycles); `consumer_fill_rate` (fulfilled/ordered, None if ordered 0); `r_start` / `r_end` (mean current/max capacity ratio after reset / last played cycle; run-down check `r_end ≥ X·r_start`); `inversion_inverted_pairs` / `inversion_differing_pairs` / `inversion_rate` (completed order cheaper per unit than a rejected one, same cycle+product); `min_capacity_ratio` (min r over played cycles); `min_spare_fraction` (min 1 − accepted/capacity over producer-cycles); `investment_rejection_rate` / `consumer_rejection_rate`; `holding_units_per_cycle`; `demand_digest` (hash of the episode's demand schedule; must match across subjects).

## Conventions
- New design decision → write an ADR in `docs/adr/`.
- New baseline → add to the dicts in `eval/baselines.py` so `signal_check` picks it up.
- New domain term → add to the *Domain Language* section at the bottom of this file.
- New diagnosed failure cause / diagnostic tool → add an entry to `docs/DIAGNOSTICS.md` (problem in plain English + check + outcome branches).
- Empirical findings → write/update the relevant `experiments/EXP-NNNN-*` record and index.
- Roadmap or target changes → update the relevant design/plan document.
- **After every task done → update the record it belongs to.** An empirical finding updates
  the `experiments/EXP-NNNN-*/RESULT.md` and the `experiments/README.md` index row; a result
  that closes or opens an arc also gets a paragraph in `experiments/HISTORY.md`; a design
  decision gets an ADR. Non-negotiable: a completed task isn't done until the durable record
  reflects it. There is no running progress log — the experiment records are the log.

## Code style
- `from __future__ import annotations` top of every module.
- **Type annotations required.** Every function/method: annotate all params AND return (`-> None` for procedures). Callables get a type alias (e.g. `Policy`). No untyped `def`.
- Value types = `@dataclass`. Type aliases in `domain/common.py`.
- snake_case. Match surrounding code. No new deps without reason.
- Docstrings explain **why** + cite the ADR, not what. Keep short.

## Tests
- `pytest`, one file per layer (`test_domain_*`, `test_simplex`, `test_env_api`, `test_eval`, `test_mlp_actor`).
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
