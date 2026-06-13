# Phase 1 Implementation Plan — Python PettingZoo AEC Environment

Port the Go domain (`emulation/domain/`) to pure Python, expose as a PettingZoo AEC environment for MARL training. Per ADR-0001 and ADR-0002.

> **Status (2026-06-13):** Env + port + smoke PPO done (slices 1–5), 17 tests pass. 4 latent Go bugs fixed (`tokenomics/PORT-NOTES.md`). ADR-0002 corrected K=2→K=3. **Now in Phase 1.5 — make the env a real learning problem (see below) before any serious training.**

## Phase 1.5 — make it a learning problem

The smoke PPO runs but a **random policy survives the full horizon**, so reward is throughput-only and barely differentiates policies. Before spending compute we make the task pose a genuine RL problem. Priority ladder:

| P | Work | Why |
|---|------|-----|
| **0** | ✅ **DONE** — baselines + metrics harness + signal check; consumer-weighted reward | Gates everything |
| 1 | Logging/eval (completion rate, rejection rate, survival length, capacity trajectories, deaths) — foundation built in `eval/`; add per-iter training curves next | Can't improve what you can't see |
| 2 | Centralized-critic MAPPO (CTDE) — actor on local obs, critic on global state | Fixes shared-reward credit assignment; the "critic distributes reward properly" goal |
| 3 | Vectorize (SuperSuit, many parallel envs) | Throughput, only after the loop is proven |
| 4 | Curriculum per deep-research 6 phases | Scale toward the real game |

**Reward objective fix (P0):** investment-order completion was rewarded equal to consumer-order completion (+10 each), but investment is *instrumental* (keep capacity alive) while consumer fulfillment is *terminal* (the actual goal). Reward is reweighted so consumer completion dominates; investment gets a small positive; the centralized critic learns investment's indirect value.

**Known training-side bug (fix at P2):** smoke PPO's GAE treats truncation and termination identically. Once early death makes episode length vary, truncation must bootstrap the value (it is not a true terminal). Currently fine because episodes always hit the horizon.

### P0 result (2026-06-13): signal is strong, no dynamics tuning needed

`python -m tokenomics.eval.signal_check` over 12 seeds, horizon 100:

| producing | ordering | cycles | dead% | cons✓ | return | capmin |
|---|---|---|---|---|---|---|
| noop | even_split | 30 | 100 | 58 | 216 | 0 |
| always_restore | even_split | 100 | 0 | 342 | 3147 | 167 |
| restore_when_low | even_split | 100 | 0 | **359** | **3227** | 105 |
| always_both | even_split | 99 | 100 | 282 | 1699 | 0 |
| restore_when_low | proportional | 100 | 0 | 359 | 3238 | 99 |
| random | random | 74 | 100 | 231 | 1739 | 0 |

Findings: (1) survival is *earnable*, not free — noop/random/always_both all die. (2) ~15× return gap between worst and best → strong learning signal. (3) **Non-trivial optimum**: `restore_when_low` > `always_restore` > `always_both` — over-investing (`always_both`) starves the economy and kills it, so the agent must learn *when* to invest, not just to invest. (4) Ordering policy barely matters yet (1:1 producers, ≤2-type orders) — it will matter once a capacity type has multiple producers (post-Phase-1).

**Target for RL: beat `restore_when_low` + `even_split` — return ≈ 3230, ≈ 359 consumer completions, 0 % death over horizon 100.**

## Goal

A `tokenomics.env` that:
- Reproduces the Go cycle logic faithfully (emit → place orders → producer investment → ordering token distribution → produce → score).
- Exposes the AEC API (`reset`, `step`, `observe`, `agent_iter`) with the observation/action spaces from ADR-0002.
- Runs fast in-process, vectorizable, no I/O.

## Package layout

```
tokenomics/
  pyproject.toml
  tokenomics/
    __init__.py
    domain/
      __init__.py
      common.py          # Tokens, Product, CapacityType, Capacity aliases; small helpers
      config.py          # Configuration, ProcessSheet, ProducingAgentConfig; load_config(json)
      order.py           # Order, OrderInfo, part status, CompleteCycle event logic
      producing_agent.py # ProducingAgent: HandleCmd, PlaceBids, Produce (auction), invest callbacks
      ordering_agent.py  # OrderingAgent: PlaceOrder, View, HandleCmd
      system.py          # System: startCycle, StartOrdering, CompleteCycle, fund distribution
      consumer.py        # Consumer protocol + ScriptedConsumer (Phase 1 fixed demand)
    env/
      __init__.py
      spaces.py          # build observation/action Box/Discrete; normalization constants
      encode.py          # domain view -> np.float32 obs vector
      decode.py          # raw action -> domain command (simplex normalize)
      aec_env.py         # TokenomicsAECEnv(AECEnv)
    config/
      phase1.json        # copy of emulation/config.json
  tests/
    test_domain_parity.py
    test_env_api.py
    test_simplex.py
```

Keep `domain/` a faithful 1:1 port — no RL concepts leak in. `env/` is the RL adapter layer on top. This split lets us validate the port against Go independently of the gym wrapper.

## Step 1 — Domain port (no RL yet)

Port each Go file to Python, preserving names and method semantics. Order matters (dependency order):

1. **`common.py`** — type aliases (`Tokens=int`, `Capacity=int`, `CapacityType=str`, `Product=int`). `Capacity` is signed int in Go (`type Capacity int`) — keep signed; the auction relies on negative `remainingCapacity`.
2. **`config.py`** — dataclasses for `ProcessSheet`, `ProducingAgentConfig` (with nested `Restoration`/`Upgrade`), `Configuration`. `load_config(path)` parses the existing JSON. Port `Validate()`.
3. **`order.py`** — `Order` with `parts: dict[CapacityType, Part]`, status enum (`unknown/processing/rejected/completed`). Port `Processing`, `Completed`, `Rejected`, `Fund`, `Info`, `AgentId`, and `CompleteCycle() -> (Score, OrderEvent)`. Event types as small dataclasses (`ConsumerRequestCompleted`, `InvestmentRequestRejected`, `OrderStillProcessing`, ...).
4. **`producing_agent.py`** — port `producerState`/`consumerState`, `HandleCmd`, `PlaceBids`, `Produce` (the greedy auction + `inProgress` carry-over), `InvesetmentCompleted/Rejected`, `View`, `Info`, `capacityDegradation`, `cutOffPrice`. **Watch the auction carefully** — sort bids by `tokens/capacity` desc, the `inProgress` booking subtracts from `remainingCapacity` first, `cutOffPrice` updates per accepted bid.
5. **`ordering_agent.py`** — `PlaceOrder`, `View`, `HandleCmd`, `CompleteCycle`. Port validation (bids sum == order tokens, all orders addressed).
6. **`system.py`** — `startCycle` (emit + place consumer orders), `StartOrdering` (`distibuteInvestmentFund` + distribute orders to ordering agents, flip state), `CompleteCycle` (produce, apply bids to orders, score, dispatch order events, restart cycle). Port `distibuteInvestmentFund` with its NaN-cut-off-price branch exactly.
7. **`consumer.py`** — `Consumer` protocol (`id`, `order() -> list[ConsumerRequest]`, `emit(tokens)`). `ScriptedConsumer` for Phase 1: returns a fixed list of `ConsumerRequest` each cycle, accumulates emitted tokens. This is the one piece Go left empty (`emulator.go` inits empty consumer map) — we define it now.

**Drop in the port:** all `logEvent` / `slog` calls (replace with nothing, or a thin optional `log()` no-op). The HTTP `application/` layer is not ported. The `SystemState` machine *can* stay in the domain for parity testing but the AEC env drives turn order itself (ADR-0001).

**Float determinism note:** Go computes cut-off price and fund distribution in `float32`. Python defaults to `float64`. To match Go parity tests, cast through `numpy.float32` at the same spots (`Bid.CapacityUnitPrice`, fund proportion). Decide tolerance in parity tests rather than chasing bit-exactness.

## Step 2 — Consumer demand for Phase 1

Per deep-research Phase 1: "static network, single product per firm type, fixed prices, hard-coded base-stock rules." So:
- Fixed set of consumers, each placing a **fixed product order every cycle**.
- Consumer tokens come from emission: `cycleEmission/2/numConsumers` (Go `emit()`). Order token budget = accumulated emitted tokens. Rejected orders refund remaining tokens back to the consumer (`Emit(remaining)` in Go `CompleteCycle`).
- Start simple: N consumers = N products, one order each, budget = its emitted tokens. Configurable via `phase1.json` extension (add a `consumers` block) so demand is data, not code.

## Step 3 — Spaces (ADR-0002)

`spaces.py` builds, from a loaded `Configuration`:
- **Producing agent obs:** `Box(16,)` — own state (capacity, max_capacity, requested_capacity, degradation, upgrade_running, restoration_running) + global producer summary (capacity + cut-off price per producer) + cycle_progress. All normalized to ~[0,1] / [-1,1].
- **Producing agent action:** `Discrete(4)` = (doRestoration, doUpgrade) bit pairs.
- **Ordering agent obs:** `Box(21,)` — K=2 order slots (active flag, normed token budget, capacity-per-type) + producer summary + cycle_progress.
- **Ordering agent action:** `Box(K * num_capacity_types,)` raw logits, simplex-normalized per order at decode.

Normalization constants (max capacity, max tokens ≈ cycleEmission, etc.) derived from config + an episode-level scale factor. Put them in one place so encode/decode share them.

## Step 4 — Encode / decode

- **`encode.py`** — turn a `ProducingAgentView` / `OrderingAgentView` + global `producerInfos` + cycle progress into the fixed float vector. Pad/mask empty order slots with zeros and active=0.
- **`decode.py`** — the simplex normalization (ADR-0002 §"How simplex normalization works"):
  1. Reshape raw action to `(K, num_capacity_types)`.
  2. Mask inactive order slots and capacity types this order doesn't require.
  3. Softmax over active entries per order → proportions.
  4. Scale by order token budget → integer token amounts (round, fix residual so sum == budget exactly; Go validation rejects mismatched sums).
  5. Map capacity-type slot → the producer providing it → build `OrderingAgentCommand.Orders[orderId][producerId] = tokens`.

Integer rounding must hit the exact budget — last slot absorbs the rounding residual.

## Step 5 — AEC environment

`aec_env.py` — `TokenomicsAECEnv(pettingzoo.AECEnv)`:
- `agents` = producing agent ids + ordering agent ids. Flat per-cycle agent sequence (ADR-0002): all producing agents act, then all ordering agents act, then env runs `Produce`/score and advances cycle.
- `reset()` — new `System` from config + scripted consumers, build agent order, return first obs.
- `observe(agent)` — encode that agent's view.
- `step(action)` — buffer the action into the right domain command; when the last producing agent acts, call `StartOrdering`; when the last ordering agent acts, call `CompleteCycle`, compute shared reward from the cycle score, assign to all agents, advance to next cycle's producing phase.
- **Reward** (ADR-0002): shared, `+10` per completed order, `-1` per in-progress, `-5` per rejected/timeout, summed over the cycle, given to every agent. Note Go's `Score` is penalty-only and unverified — compute reward in the env layer from the order events, not from Go's `Score`.
- **Termination:** fixed horizon (default 100 cycles) → truncation; all-producers-zero-capacity → termination (system death). Set `terminations`/`truncations` accordingly.
- Standard PettingZoo plumbing: `rewards`, `_cumulative_rewards`, `terminations`, `truncations`, `infos`, `agent_selection`, `_was_dead_step`.

## Step 6 — Tests

1. **`test_domain_parity.py`** — drive the Python `System` through scripted cycles, assert order outcomes / cut-off prices / capacities match hand-computed expectations (and, if practical, a captured Go run). This is the safety net for the port.
2. **`test_simplex.py`** — decode: random logits → valid command (non-negative ints, sum == budget, only required capacity types, only producers that supply them).
3. **`test_env_api.py`** — run `pettingzoo.test.api_test` and `seed_test`; run a full random-rollout episode to termination without crashing.

## Step 7 — Smoke train (optional, ends Phase 1)

Wrap with SuperSuit (pad + parameter-sharing → vector env), run a short MAPPO / PPO run (CleanRL or RLlib) for a few hundred cycles on the M1 to confirm reward moves and the loop is learnable. Not full training — just proof the env trains. Hand to Sergey for real runs.

## Dependencies

`pyproject.toml`: `pettingzoo`, `gymnasium`, `numpy`. Test: `pytest`. Smoke train (separate optional extra): `supersuit`, `torch` (CPU), one of `rllib` / cleanrl-style PPO. `.venv` already exists (py3.12) — install into it.

## Build order (PR slices)

1. `domain/` port + parity tests. ← biggest risk, do first, verify against Go.
2. `consumer.py` + `phase1.json` demand.
3. `spaces` + `encode`/`decode` + simplex test.
4. `aec_env.py` + API test + random rollout.
5. (optional) smoke train script.

Each slice is independently testable. Slice 1 is the one to get exactly right — everything downstream trusts the port.
