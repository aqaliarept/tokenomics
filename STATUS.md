# STATUS — EXP-0006A implementation (enriched orderer observation)

Implementing `ALPHAZERO_EXPERT_ITERATION_APPROACH.md`, first concrete move
(EXP-0006A): **enrich the orderer observation with public producer/product/order
state, in the domain model first, then normalized by the env layer.**

Scope of this work (agreed):
- Build the **domain statistics layer** + the **enriched orderer observation**
  contract (M2-v3-obs input), TDD'd locally, **full feature catalog**.
- Stop before cloud training/relabel/eval — those run on Runpod (compute rule) and
  are a follow-up, not TDD-able units.

Method: TDD, vertical slices (one test → one impl). Producer fixed to EXP-0004 v4
(unchanged). Producer observation unchanged. Enriched orderer obs is versioned /
opt-in so M2-v1 and existing checkpoints/tests stay valid (checkpoint-schema bump).

## ✅ DOMAIN + ENV WORK COMPLETE
All four phases done, 28 new tests, full suite **264 passed** (was 236), zero
regressions. New files: `domain/metrics.py`, `domain/telemetry.py`,
`domain/orderer_view.py`; new tests: `test_domain_metrics.py`,
`test_domain_telemetry.py`, `test_telemetry_integration.py`, `test_orderer_view.py`,
`test_enriched_orderer_obs.py`. Touched: `domain/system.py` (telemetry wiring +
`orderer_economy_view`), `domain/order.py` (`ORDER_TTL` constant), `env/spaces.py`
(versioned dim + block constants), `env/encode.py` (`encode_ordering_enriched`),
`env/aec_env.py` (`enriched_orderer_obs` flag).

Use it: `TokenomicsAECEnv(enriched_orderer_obs=True)` → ordering agents get the v2
obs; `build_spec(cfg, orderer_obs_version=2)`.

## Architecture
- `domain/metrics.py` — pure snapshot-derived formulas (no persistent state).
- `domain/telemetry.py` — `EconomyTelemetry`, persistent per-cycle EMA stats, wired
  into `System`.
- `domain` `OrdererEconomyView` — authoritative per-ordering-agent feature bundle.
- `env/encode.py` + `env/spaces.py` — versioned enriched `encode_ordering`.

## Slice ladder (status: TODO / WIP / DONE)

### Phase A — domain/metrics.py (state-derived, pure)  ✅ DONE (11 tests)
- [x] A1 producer capacity summary: total current / total max / headroom
- [x] A2 capacity ratios: per-type, producer min / mean over owned types
- [x] A3 producer load: requested / reserved / total
- [x] A4 producer in-flight: outstanding-inbound units + active-investment one-hot
- [x] A5 capacity_makeable_qty[p,x]
- [x] A6 input_makeable_qty[p,x]
- [x] A7 accept_qty_now / complete_qty_now / makeable_qty_now
- [x] A8 bottleneck ratios (capacity / input / combined) for qty q
- [x] A9 slot_makeability[k,p]
- [x] A10 product_depth (recipe graph)
- [x] A11 recipe complexity: input_count / input_units / capacity_type_count /
      eligible_producer_count

### Phase B — domain/telemetry.py (persistent EMA)  ✅ DONE (6 tests)
- [x] B1 EMA fold mechanics + routed qty per (p,x) EMA (merged tracer bullet)
- [x] B3 completed / rejected / failed qty per (p,x) EMA
- [x] B4 success rate / throughput / producer accept-complete-reject rates (+accepted)
- [x] B5 completion latency / failure latency (quantity-weighted EMA, no idle decay)
- [x] B6 product demand EMA + rejected-demand EMA

### Phase C — System integration + OrdererEconomyView  ✅ DONE (7 tests)
- [x] C1 System wires telemetry across a cycle (routed/produced recorded)
- [x] C2 consumer demand + rejection recorded
- [x] C3 OrdererEconomyView order-slot features (+ ORDER_TTL domain constant)
- [x] C4 OrdererEconomyView producer features
- [x] C5 OrdererEconomyView product + producer-product features

### Phase D — env enriched ordering obs  ✅ DONE (4 tests)
- [x] D1 EnvSpec enriched ordering dim + `orderer_obs_version` flag (build_spec param)
- [x] D2 encode_ordering_enriched: dim lockstep w/ spec constants, normalized + clipped
- [x] D3 env `enriched_orderer_obs=True` serves v2 obs; pettingzoo api_test green; v1 +
      producer obs untouched

## Follow-up (NOT in this session — cloud)
- regenerate Stage-2b teacher labels with enriched obs
- train deterministic M2-v3-obs (Runpod), eval vs envelope on held-out 6,7,8,9
- write `experiments/EXP-0006A-orderer-observation/` with RESULT.md

## Notes / decisions
- Telemetry is domain bookkeeping (pure Python, no IO/RL/numpy) — allowed under the
  `domain/` purity rule; it is simulation state, not RL.
- `config_fingerprint(config, orderer_obs_version=1)` hashes the version in
  explicitly (builds the spec at it), so a v2 checkpoint cannot be mis-scored against
  a v1 economy. `save_policies` forwards the version.

## Review fixes (post-implementation, 5 regression tests, suite 269 passed)
- [x] `recent_rejected_qty` = rejected **or failed** (`telemetry.rejected_or_failed_ema`)
- [x] route cycle travels per `Bid` (cycle-safe latency under partial re-routing);
      removed the `(producer, order)` route-cycle map from `System`
- [x] `metrics.product_depth` is cycle-safe (breaks recipe-graph back edges)
- [x] `config_fingerprint` takes/incorporates `orderer_obs_version` (was always v1)

## Review fixes round 2 (2 regression tests, suite 271 passed)
- [x] orphaned reservations (outlive parent order) recorded as failed before the
      telemetry fold, then dropped — feed `failed_ema`/`recent_rejected_qty`/latency
- [x] `accept_qty_now`/`capacity_makeable_qty`/`capacity_bottleneck_ratio`/
      `slot_makeability` use available capacity (current − live reservations), not raw
