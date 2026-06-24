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

## EXP-0006A cloud run — M2-v3-obs training (IN PROGRESS)
Crossing into the cloud step of EXP-0006A. The obs contract existed but no trainer
could *drive* it — `orderer_improve` + `eval.runner.evaluate` built v1 envs only, so
a cloud run would have silently retrained M2-v1. Wired the enriched flag end-to-end
(TDD, +3 tests, suite **274 passed**):
- [x] `eval.runner.evaluate(enriched_orderer_obs=False)` → builds v2 envs when set
- [x] `orderer_improve`: `collect_orderer_samples` / `dagger_orderer` /
      `run_orderer_improve` / `eval_m2` thread the flag; probe sized to v2; CLI
      `--enriched`; `save_policies(orderer_obs_version=2)` (ckpt fp == v2, verified)
- [x] local smoke (tiny dagger+eval) GREEN end-to-end before launch
- [x] **Runpod CPU run DONE** (pod bh6app36yc1o67, ~70 min, exit 0): `orderer_improve
      --enriched` = EXP-0005 M2-v1 recipe + obs v2 only (clean A/B), held-out 6,7,8,9.
      Artifacts pulled + ckpt fp verified == v2. Experiment written:
      `experiments/EXP-0006A-orderer-observation/` (spec/RESULT/summary.csv), README row.

### EXP-0006A verdict: REJECT (strict) — observability is a PARTIAL lever
M2-v3-obs vs strict envelope (M2-v1 in parens): s6 **1.01**(0.99) s7 **1.04**(1.06)
s8 **1.08**(1.02) s9 **0.87**(0.91). Zero death all 4; beats M0 on all 4 (1.04–1.42).
Enriched obs lifts 3/4 seeds over the envelope but **s9 regressed** (wide CI ±42,
rej 121). s9 envelope-best = `sa+random` → routing *diversity* wins; a deterministic
mean can't reproduce it regardless of inputs → residual = **distillation form, not
state**. Next = **EXP-0006B stochastic distillation**, keep v2 obs as the contract.

## EXP-0006B — stochastic orderer distillation (LOCAL IMPL DONE, awaiting cloud)
Acting on the EXP-0006A finding (s9 residual = distillation *form*: a single deterministic
mean can't reproduce the `sa+random` routing diversity that wins s9). Distill a **stochastic**
orderer: learn the per-dim `log_std` and **deploy by sampling** instead of collapsing the
teacher's per-cycle variety into clip(mean). Form chosen (user): **learned log_std (MLE)** —
reuses the existing single-label pipeline, smallest new surface. TDD, +3 tests, suite **277**.
- [x] **Slice 1** `_fit_orderer_clone(stochastic=True)`: masked Gaussian **NLL** fits mean +
      per-dim log_std; std recovers the teacher label spread on decode-active dims, masked
      dims keep init (deterministic masked-MSE path unchanged).
- [x] **Slice 2** `_stochastic_closure(actor, bound, seed)`: deploy = sample
      `a~Normal(mean,std)`, clip to ±bound; varies per call, reproducible per seed.
- [x] **Slice 3** `--stochastic` end-to-end: `dagger_orderer`/`run_orderer_improve`/`eval_m2`
      thread the flag; save **skips the orderer log_std clamp** (producer stays greedy-clamped);
      eval deploys via `load_order_actor` + `_stochastic_closure`. Tests:
      `tests/test_orderer_stochastic.py`.
- [x] **Pre-cloud smoke (local, allowed)** — step-5 gates all PASS before spending cloud:
      (a) tiny `orderer_improve --enriched --stochastic` runs end-to-end train→save→eval;
      (b) saved ckpt `config_fingerprint` == **seed-0 v2** exactly (548120bd…), `version` 6;
      (c) orderer `log_std` is **learned** on decode-active dims (~−1.9) while masked dims stay
      at the −2.0 init — NLL fit the behavioral dims only, not the producer's greedy clamp;
      (d) **P2** order-independence holds: seed 9 = `103±9` identical in `[9]` and `[6,7,8,9]`.
- ✅ **Cloud run DONE — ACCEPT (strict)** (pod 3joly6h2z0hboa, 8 vCPU, ~75 min, DETACHED mode).
      M2-v3-stochastic vs strict envelope, held-out 6,7,8,9 (eval-seeds 10): s6 **1.026**, s7
      **1.086**, s8 **1.057**, s9 **1.022** — **beats the envelope on ALL 4, zero death**. First
      policy in the EXP-0001→0006 arc to clear the strict envelope everywhere. Hard-cell met: the
      two `sa+random` misses lift over the line (s6 0.99→1.026, s9 0.96→1.022; s9 > M2-v1 0.91).
      Mean env 1.048 (< secondary 1.10 "if feasible"; s9 mean-only, lower CI marginal). Diversity
      WAS the missing form (gains land on the random-dominated seeds; s7=sa+prop trades 1.10→1.086
      but clears). 64 learned log_std dims (σ≤0.60), v2 fp, md5 7be79ccbd90e. Written:
      `experiments/EXP-0006B-orderer-stochastic/` (spec/RESULT/summary.csv), README row.
      **Promote M2-v3-stochastic as orderer champion.**
- ✅ **Confidence audit DONE — s9 caveat RESOLVED** (local `--eval-only`, same ckpt, **40**
      eval-seeds, CI halved). All 4 held-out clear the strict envelope at the **lower 95% CI
      bound**: s6 484>473, s7 470>436, s8 556>533, **s9 501.3>498.3** (was mean-only @10 seeds).
      Mean ratio 1.047 unchanged (CI tightening doesn't move the mean → secondary 1.10 still the
      only unmet, optional, target). `artifacts/m2_v3_obs_stochastic_audit40.csv`; RESULT/spec/
      README/memory amended. Per user call: **bank the win, skip elite-set margin-chasing**.

## Champion regression gates (TDD, +4 tests, suite 287 + 1 opt-in)
Pins the shipped EXP-0006B champion so a retrain/refactor can't silently swap a weaker or
deterministic orderer. `tests/test_champion_regression.py`; `champion` marker registered,
deselected by default (`addopts = -m 'not champion'`) so `pytest -q` stays fast, opt in with
`pytest -m champion`. All skip if the LFS artifact is absent.
- [x] **Provenance** (fast): version 6 + enriched-v2 fingerprint 548120bd + order_act 180.
- [x] **Stochastic head** (fast): 64 dims grew above the masked init (σ≤0.605), ≥100 dims pinned
      at init — a learned diversity head, not a collapsed deterministic mean.
- [x] **P2 order-independence** (fast, ~13s): the champion scores a held-out topology
      identically whether evaluated alone or after another in-process (per-episode env RNG).
- [x] **Envelope bar** (slow `-m champion`, ~13 min): 40-seed eval, every held-out seed beats the
      strict envelope at its lower 95% CI bound, zero death. Verified green (1 passed, exit 0).

## run_pod_training.py — detached client-server rewrite (TDD, +6 tests, suite 284)
The CLAUDE.md "foreground over SSH dies on disconnect" caveat is now resolved. The pod
runs the job **detached** (`setsid`, own session) writing log + `.pid`/`.exit` markers;
the client **polls and auto-reconnects** (re-resolves ip/port) on any link drop, finishing
only when the remote exit marker appears. Ctrl-C just detaches (run keeps going); reconnect
with `--attach`. Pure helpers TDD'd one slice at a time in `tests/test_run_pod_detached.py`:
- [x] `build_launch_cmd` — detached `setsid` launch, records pid (liveness) + `$?` (exit)
- [x] `parse_poll_output` — split streamed log bytes from the `__RUNPOD_POLL__ size/exit/alive` sentinel
- [x] `build_poll_cmd` — one round-trip: `wc -c` size + `tail -c +offset` new bytes + sentinel
- [x] `marker_paths` — derive `.pid`/`.exit` siblings from `--remote-log`
- [x] `run_monitor_loop` — DI'd loop: reconnect on `SshDropped`, offset never regresses, no lost/dup content
Integration glue (`ssh_capture` raises `SshDropped` on rc 255/timeout; `monitor_remote`).
New CLI: `--foreground` (legacy), `--attach` (resume an already-launched run), `--poll-interval`.
Default flow = detached+resilient; finally-stop is skipped on a detached Ctrl-C so the run survives.

### Review fixes (2 correctness bugs, +1 test, suite 278)
Pre-cloud review of the enriched/stochastic paths caught two result-invalidating bugs:
- [x] **P1 — obs_price corrupted under v2.** `decide_ordering_obs` read the producer cut-off
      price at the hardcoded **v1** offset `k*(3+q)`; under enriched v2 obs that block is
      strided inside the per-producer block (cut-off = idx 8 of 16), so the `obs_price`
      search candidate parsed garbage → the v2 teacher's candidate set was NOT held fixed
      vs v1. Fix: `EnvSpec.ordering_price_indices()` (version-aware, single source of truth);
      obs_price now emits identical actions in v1 and v2 (test guards it).
- [x] **P2 — stochastic eval order-dependent.** `eval_m2` built one `_stochastic_closure`
      with a single closure RNG reused across all held-out seeds, so seed-9's score depended
      on what was sampled before it (`seeds=[9]` ≠ `seeds=[6,7,8,9]`). Fix: draw noise from
      `env.action_space(agent).np_random` (seeded per episode by `run_episode`, like the
      `random` baseline) → reproducible per seed, order-independent.
- ✅ **EXP-0006A clean re-run DONE** (pod 9ytl7r42avgpmm, 8 vCPU, ~50 min; artifacts
      `m2_v3_obs_fixed.*`). Verdict **unchanged: REJECT (strict)** — but the mechanism is now
      clean. Fixed-teacher M2/env: s6 **0.99**(buggy 1.01), s7 **1.10**(1.04), s8 **1.05**(1.08),
      s9 **0.96**(0.87); beats M0 on all 4 (1.10–1.38), mean env 1.025 (buggy 0.999), 0 death.
      The "seed-9 cliff" was **largely a teacher artifact** (0.87→0.96, rejects 121→76, CI
      ±42→±18). Now misses on **s6 + s9**, both `sa+random`-dominated → residual = **routing
      diversity, not state**. Per the decision rule ("improves but still fails → 6B justified"),
      **EXP-0006B is GO on clean evidence**.
- Infra fixes this round: (a) **P1** obs_price v2 offset (above); (b) **P2** stochastic eval
      per-seed RNG (above); (c) RunPod pods need `PUBLIC_KEY` env set or sshd never starts
      (connection refused) — the working pod had it; stopped pods can't restart on full hosts
      (deploy fresh). `run_pod_training.py` SSH resolution left as the proven original (a
      mid-debug `runtime.ports` change was reverted — this runpodctl's `ssh info` already
      returns the direct endpoint).
