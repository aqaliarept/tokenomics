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
