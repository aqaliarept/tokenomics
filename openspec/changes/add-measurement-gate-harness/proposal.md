# Proposal

## Why

The acceptance contract (`agent-acceptance`, `side-effect-guards`) is written, but nothing
in the tree can measure it. Today's hindsight baseline is a max over four baseline means, not a
per-episode max over a frozen roster. `blend_0.5` is not in the baseline registry. There is
no per-consumer accounting, no run-down or starvation metric, and no split between
development and sealed worlds. The scale generator cannot build sparse economies or keep
demand in step with capacity when the economy grows from 12 to 120 producers. So no
training spend can yet be judged honestly. This is milestone M1 of the agreed plan
(`establish-project-baseline` design D1/D2), and every later milestone (shocks, the
development audit, evolution strategies, the sealed 120-producer gate) reads its numbers
from this harness.

## What Changes

- **Per-consumer, played-cycle accounting**: episode statistics gain ordered and fulfilled
  units per consumer, counted only over cycles that were actually played. Demand still
  unresolved at the horizon stays in the denominator, and the unplayed next-cycle orders
  placed by the final `start_cycle` are excluded. Existing `episode_stats` keys keep their
  current values.
- **Side-effect metrics**: hard constraints are run-down (end/start mean capacity ratio)
  and starvation (per-consumer fill rate). Monitored only: token-priority inversion,
  edge-running margin, investment crowd-out and held inventory. X and Y are frozen from
  baseline-only calibration on development worlds before any gate run.
- **Frozen, versioned baseline roster**: every hand-written producer × orderer stack plus
  `blend_0.5`. Random policies get R replicates whose seeds are derived by hash, never the
  event seed.
- **Hindsight-baseline arithmetic as a pure function**: per-episode max over the roster, then the
  mean over episodes, then the candidate/hindsight-baseline ratio per world, then an equal-weight
  world mean per regime.
- **World registry**: a versioned file listing development and sealed worlds on disjoint
  seeds. Sealed evaluation requires a frozen pre-registration manifest and is logged.
  Worlds with a zero hindsight baseline are replaced from a reserve list, decided by baseline runs
  alone.
- **Regimes**: *calm* worlds use explicitly configured deterministic cyclic demand
  schedules, a new opt-in consumer field; legacy fixed demand is never silently replaced.
  *Shocked* worlds use seeded stochastic demand (B9) until supply shocks arrive in M2. The
  calm set has at least twice as many worlds as the shocked set.
- **Scale families**: new sparse-wide and dense layered generators with an explicit
  `fanout` (eligible producers per product). Demand-to-capacity and fund-to-producer ratios
  are held constant from 12 to 120 producers by co-scaling consumers and emission.
  Development worlds must show headroom at both sizes. `build_scale_config` stays
  byte-identical.
- **Gate driver + report**: a CLI that runs the subject stack and the roster on registry
  worlds (sharded across worker processes) and writes an experiment-dir layout with
  per-episode rows, per-world ratios, regime verdicts, side-effect and monitored metrics,
  and provenance for rules, policies, roster and registry.
- **Exit deliverable**: an `EXP-NNNN` gate report for the portable champion stack
  (EXP-0012 orderer scorer + EXP-0035 producer) on development families at 12 and 120
  producers, run on a CPU pod.
- **Not in this change**: the deployed-path cost refactor (eligible-only rows, incremental
  telemetry, O(E) remainder; D2's last bullet) becomes its own change, because the M1
  report does not depend on it and it carries its own byte-identical invariant. Supply
  shocks, drift and the shock-reactive roster member are M2 and arrive as roster v2.

## Capabilities

### New Capabilities
- `gate-harness`: how a candidate stack is measured against the acceptance contract. Covers
  the baseline roster and its versioning, hindsight-baseline arithmetic, the world registry and
  sealed-world discipline, regimes, scale families, side-effect calibration, the gate report
  and its provenance binding.

### Modified Capabilities
- `side-effect-guards`: "Per-consumer accounting" gains the played-cycle cohort rule, how
  demand unresolved at the horizon is treated, and how zero-ordered consumers are handled.
- `agent-acceptance`: "Calm-regime margin" now names calm worlds as having neither supply
  shocks nor stochastic demand. That matches the shocked-regime definition, which already
  counts stochastic demand as shocked.
- `stochastic-environment`: adds "Deterministic cyclic demand", the opt-in calm demand
  source, and keeps "Calm regime is preserved" true for legacy configs.

## Impact

- **Domain** (`domain/system.py`, `domain/consumer.py`, `domain/config.py`): a per-order
  consumer outcome record on `CycleResult` and a placement-cycle tag. `ConsumerConfig`
  gains an optional cyclic `schedule`. Still pure: no RNG, IO or numpy.
- **Env** (`env/aec_env.py`): per-consumer stats and side-effect accumulators. Optional
  action-space seed decoupled from the event seed. Existing keys and default behaviour
  unchanged.
- **Eval**: new `eval/gate.py` (driver, hindsight baseline, verdict), `eval/roster.py`,
  `eval/registry.py` + a registry JSON, and `eval/families.py` (sparse-wide/dense).
  `eval/baselines.py` gains a `blend_0.5` entry; `eval/runner.py` gains an `action_seed`
  argument. `eval/task_archive.py` `REPRODUCERS` gains the new families.
- **Experiments**: calibration, registry screening and the champion report run on a CPU
  pod through `run_pod_training.py`. A new `EXP-NNNN` record, an index row, and a HISTORY
  paragraph that opens the measurement arc.
- **Checkpoints**: the EXP-0012 and EXP-0035 portable checkpoints must be pulled with
  git-lfs and staged under `tokenomics/_pod_ckpts/` for the pod run.
- No new dependencies.
