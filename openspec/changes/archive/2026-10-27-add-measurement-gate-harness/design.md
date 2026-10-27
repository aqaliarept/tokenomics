# Design

## Context

See `proposal.md` (Why). Requirements live in `specs/gate-harness`, plus the deltas to
`side-effect-guards`, `agent-acceptance` and `stochastic-environment`. Current state that
shapes the approach:

- **Hindsight baseline today**: `eval/orderer_oracle.py:46-76` `_hindsight_baseline_units` takes the max over
  four stacks of their *mean* units, on the legacy env contract. Nothing computes a
  per-episode max, and `blend_0.5` (`eval/baselines.py:322`) is missing from
  `ORDERING_BASELINES` (`eval/baselines.py:346`).
- **Random baselines reuse the event seed**: `eval/runner.py:56-61` seeds every action space
  with the episode seed, which also seeds the demand RNG (`env/aec_env.py:174`).
- **Only totals exist**: `CycleResult` (`domain/system.py:80-93`) and `episode_stats`
  (`env/aec_env.py:210-224`) hold totals only. The final `complete_cycle` calls
  `start_cycle` (`domain/system.py:460`), which places consumer orders for a cycle that
  is never played. Degradation happens inside `produce()`, so `start_cycle` does not move
  capacity.
- **Demand**: `ConsumerConfig` (`domain/config.py:78-110`) can express a fixed product or
  B9 random bounds only. `ScheduledConsumer` (`domain/consumer.py:50-81`) already replays a
  finite schedule cyclically, but nothing in the config can supply one.
- **Scale generator**: `build_scale_config` (`eval/scale.py:87-189`) is dense by
  construction: every layer producer declares every layer product. It has 3 consumers and
  emission `2000 + 200·depth·producers_per_layer`, so at 42 (or 120) producers demand would
  not grow with capacity.
- **Champion stack**: the flat `champion.json` checkpoints have fixed input and output
  widths (180-dim orderer action, `tests/test_champion_regression.py:56`). A flat model
  cannot be portable by construction (EXP-0022). The portable stack (`eval/joint_stack.py`,
  EXP-0012 orderer scorer + EXP-0035 producer) runs at any size (EXP-0016). Both portable
  checkpoints are git-lfs pointers locally.
- **Reusable pieces**: `eval/task_archive.py` (generator-ref replay), `eval/provenance.py`
  (input pins, tree fingerprint, environment), and the `bounded_rollout_orderer`
  worker-pool pattern.
- **Local box**: 2 cores. Every calibration, screening and report run is a pod run.

## Goals / Non-Goals

**Goals:**
- Every number the acceptance specs mention is computed by one tested code path. Raw
  per-episode rows are separated from pure verdict arithmetic.
- The roster, side-effect parameters and registry are frozen by baseline-only runs before
  any candidate touches them.
- One champion report at 12 and 42 producers on development worlds, recording where the
  champion stands with no pass expected.

**Economy sizes (user decision 2026-10-27).** The gate sizes are 12 and **42** producers (14
per layer). The goal is to beat the hand-written baselines at 42 first, then scale. About 120
producers is a future, non-gating target (agent-acceptance "Future scale target"). Earlier
drafts and the task 8.3–8.5 sizing used 120; evidence quoted here from those rounds says so.

**Non-Goals:**
- No sealed-world evaluation. The sealed path is built and tested on fixtures only; M5
  uses it.
- No supply shocks, drift, resilience report or shock-reactive baseline (M2, roster v2).
- No deployed-path cost refactor (separate change), except its observation-cost part. The
  user pulled that into M1 on 2026-10-27 (task 8.4) after task 8.3 sized the runs at
  ≈1,600 + 900 vCPU-hours. Producer rows are memoized per phase and per-producer
  telemetry totals are indexed, with byte-identical observations (docs/DIAGNOSTICS.md
  entry 17). Eligible-only rows, incremental telemetry and the O(E) remainder stay
  deferred. No training or candidate selection.
- No change to legacy metrics, the legacy scale generator, or default episode seeding.

## Decisions

### D1. Gate subject is the portable stack
The report scores the portable stack: EXP-0012 orderer scorer and EXP-0035 producer, under
the `joint_stack_env` contract. The flat `champion.json` stack is excluded: its fixed
widths cannot run on the new families or at 42 producers, and the 42-producer clause
needs a policy that runs unretrained there. *Alternative:* report the flat stack at 12
producers only. Rejected because it fits only the legacy depth-3 scale economy, which is not
a registry family. HISTORY §0 is updated to say which stack the gate measures.

### D2. Accounting from domain outcome records, attributed in the env
The domain tags each consumer order with its placement cycle. `CycleResult` gains a tuple
of per-order consumer outcomes: consumer, product, quantity, tokens at placement,
placement cycle, completed or rejected. These are pure data, with no RNG or IO. The env
keeps per-consumer `ordered_units` and `fulfilled_units`. An order counts as ordered once
its placement cycle has been played, so the final `start_cycle`'s orders never count.
Orders still open at the horizon stay ordered and unfulfilled. Existing counters are left
untouched.

*Alternative:* derive ordered units from completed + rejected + open-at-end, with no
placement tags. Rejected: it cannot tell played from unplayed placements without the tag
anyway. Fixtures cover horizon 1, early death, an open order at the horizon, and a
zero-order consumer.

### D3. Side-effect metric definitions
These are computed in the env from domain data. Hard checks:
- **Run-down**: `r_end ≥ X · r_start`. `r` is the mean over producers of their mean over
  owned types of current/max capacity. `r_start` is read after `reset`; `r_end` after the
  last played cycle.
- **Starvation**: fulfilled/ordered ≥ Y for every consumer with ordered > 0.

Monitored only:
- **Inversion**: within each (cycle, product), count consumer-order pairs where the
  completed order has lower tokens per unit than the rejected one. The rate is inverted
  pairs over pairs with differing tokens per unit.
- **Edge margin**: the minimum over cycles of the capacity ratio, and the minimum over
  producer-cycles of the spare fraction. Spare fraction is 1 − accepted bundle units /
  current capacity.
- **Crowd-out**: investment rejection rate and consumer rejection rate.
- **Hoarding**: `holding_units` (already present), total and per-cycle mean.

### D4. X and Y calibration rule
For every development episode, take the roster member that attains the per-episode
hindsight baseline. Over those episodes:
- X is the q-quantile of their run-down ratio.
- Y is the q-quantile of their per-consumer fill rates.

The default is q = 0.05, written into `side_effect_params_v1.json` before the calibration
run. The bar is one that the best baselines themselves clear about 95% of the time, so
"good" is not defined by a candidate. *Alternative:* the minimum over baseline episodes.
Rejected: a single pathological episode would set the bar.

### D5. Roster v1 and env contracts
Roster v1 is `PRODUCING_BASELINES` × (`ORDERING_BASELINES` + `blend_0.5`): 40 stacks, 12
of them with a random policy, each run with R replicates. Each member declares its env
contract:
- white-box baselines run under `joint_stack_env`, the candidate's ordering mechanics;
- `obs_*` baselines run under the legacy flat-observation contract they read.

Pairing across contracts holds because demand depends only on config + episode seed
(`env/aec_env.py:169-195`). The harness asserts this with a per-episode schedule digest.

Pruning is optional and happens only before freezing. The rule: drop a member that is never
the per-episode argmax on any development episode at either size. The roster file stores
its member list, R, the pruning record and a content hash; that hash is the version.

`run_episode` gains an `action_seed` argument (default: today's behaviour). Replicate seeds
are `int(sha256(f"{world}|{episode}|{replicate}")[:16], 16)`. A random member's
per-episode value is the replicate mean, because a max over replicates would reward luck
and grow with R.

### D6. Rows first, arithmetic pure
The driver writes one row per (world, episode, member or candidate, replicate) with units,
death, side-effect fields and the schedule digest. Hindsight baseline, ratios, regime means and the
verdict are pure functions over those rows plus the frozen files (`eval/gate.py`). The
report's `--recompute` mode rebuilds the verdict from the CSV alone. *Alternative:*
aggregate inside workers. Rejected: the verdict could not be audited or recomputed.

### D7. World registry
`eval/registry/worlds_v1.json` is shipped as package data. Each entry is a task-archive
generator reference plus split, regime, size, family, episode seeds and reserve position.
The version is the content hash; validation is on load.

Defaults, pre-registered in the file:

| Split | Seed range | Calm per family × size | Shocked per family × size |
|---|---|---|---|
| Development | 1000–1999 | 8 worlds × 3 episode seeds | 4 worlds × 10 episode seeds |
| Sealed | 5000–5999 | 16 worlds × 3 episode seeds | 8 worlds × 10 episode seeds |

Sizes are the gate sizes, 12 and 42 producers ("Economy sizes"; task 8.7 regenerated the draft
from 120 to 42 without changing any seed rule). **M1 registers the sparse-wide family only**
(task 10.4, D10 note): the generator still allocates seeds over both families in a fixed order
(`SEED_ORDER`) but emits only `GATE_FAMILIES`, so every sparse-wide world, with its seeds and
episode seeds, is the one EXP-0044 attempt 1 screened. A real screen or report refuses a
registry that holds a non-gate family (`coverage_problems`). Each family × size also has a reserve list for
zero-baseline replacement. Screening runs by
split and records the splits it covered: M1 screens development only (no sealed evaluation in
M1); the sealed split gets its own roster-only screen in M5, before any candidate touches a
sealed world. The headroom ceiling (D10) is pre-registered in the registry file.

Calm worlds get several episode seeds even though their demand is fixed, because future
stochastic candidates may sample actions. Sealed evaluation requires `--preregistration
<manifest>`: candidate checkpoint SHA-256 plus roster, registry and parameter versions,
checked through `eval/provenance.py`. Each sealed run appends to
`experiments/SEALED_ACCESS.log`.

### D8. Calm worlds through explicit cyclic schedules
`ConsumerConfig.schedule: tuple[tuple[Product, int], ...] = ()`. When it is set, the env
builds a `ScheduledConsumer` from it and ignores the B9 bounds. A calm world is its
family's world with each consumer's schedule drawn once, at world-generation time, from
the world seed (default length 7, within the consumer's pool and quantity range). The
schedule is part of the config, so every episode sees the same demand. JSON load and
`config_to_dict` round-trip the field. *Alternative:* reuse B9 with a fixed episode seed.
Rejected: that hides stochastic machinery inside a "calm" label and breaks the
seed-independence the spec requires.

### D9. Families
`eval/families.py` builds a layered economy: depth 3, with P/3 producers per layer. It
keeps `build_scale_config`'s strong/marginal capacity draws and maintenance products, with
two refinements so the economy pressure stays constant from 12 to the large size (42; the
evidence below was measured at 120, where the effect is largest):
- **Forced-strong every m-th producer**, m = min(fanout, 4), instead of only the first
  producer of a layer. At dense-12 this is the same rule. With first-only forcing, the dense
  family's demand ratio is 11.8% apart between 12 and 120 (forced fraction 1/4 → 1/40); with
  every 4th it is 0.3% apart. At 42 the dense forced fraction is 4/14 (14 per layer is not a
  multiple of 4), close to 1/4. For sparse, m = fanout gives every product a strong declarer.
- **Maintenance round-robin over layer-0 products** instead of all on product 0. With
  product 0 only, a sparse-120 world loads all 120 producers' maintenance on its 2
  declarers, 10× the per-maker load at 12 (3.5× at 42); round-robin gives every sparse
  layer-0 product exactly 3 users at any size (dense: 3 at 12, 10–11 at 42).
- **Dense**: W = 4 products per layer, and every layer producer declares all of them
  (fanout = P/3).
- **Sparse-wide**: W = P/3 products per layer, each declared by `fanout` producers
  (default 2, round-robin).

Both families stay buildable; in M1 only sparse-wide gates (D10 note). Consumers scale as
3·P/12 **rounded half up** (3 at 12, 11 at 42, 30 at 120). At 42 the exact
value is 10.5; Python's banker's `round` gives 10, which put the 42-producer demand ratio 7–9%
under the 12-producer one, close to the tolerance, while 11 keeps it within 3% (task 8.7). Emission
per consumer is therefore exact at 12 and 120 and 1400 instead of 1467 (−4.5%) at 42. Pools are the whole top layer for dense worlds and a
contiguous window of top products for sparse worlds, so demand covers every chain.
Emission scales linearly with P from the 12-producer value. A test asserts that demand per
unit of initial top-layer throughput matches between 12 and 42 within the recorded
tolerance (default 10%) as a family-level ratio of sums over the sampling protocol recorded
in the family definition (a single 12-producer world has 4 top producers and swings with
its strong/marginal draw), and that emission per producer is equal in every world. The protocol
is 100 worlds at 12, 40 at 42 and 20 at 120 (400–800 top-layer producers each, seeds from
1000). Measured gaps 42 vs 12 (task 8.7): dense calm +1.6%, dense shocked +2.8%, sparse-wide
calm +2.3%, sparse-wide shocked +0.7%; at 120 (non-gating, still tested): −0.6%, +0.3%,
−2.7%, −2.9%. Both families are registered in
`task_archive.REPRODUCERS`. `build_scale_config` is untouched.

### D10. Headroom check
A family × size (sizes 12 and 42) is admitted only if the median hindsight baseline fill rate over
its development worlds is below the saturation ceiling, default 0.95. EXP-0033 found over-capacitated
worlds with no headroom at 98.3%. A failing family is retuned with `capacity_scale` and
rescreened on baseline evidence only. Each attempt is recorded in the calibration EXP.

**M1 outcome (tasks 10.1–10.4, user decision 2026-10-27).**
- EXP-0044 attempt 1 failed headroom on dense-12 (0.954).
- The capacity-scale probe (10.3) did not move it: 0.955–0.962 at 0.8×–1.0×.
- The diagnostic (DIAGNOSTICS 18), valid on calm worlds only (its h = 20 vs h = 100 subtraction
  is unmatched on shocked ones), suggests why. On calm worlds dense's best-baseline shortfall is
  mostly warm-up: 54% of it in cycles 1–20 at 12 producers and 85% at 42, with ≥ 97.5% served
  afterwards. Capacity is ≈ 12× above demand, so it is not the lever.
- The decision rests on the headroom statistic itself: sparse-wide passes (0.934 at 12, 0.886
  at 42) and dense-12 fails. Post-warm-up headroom on sparse-wide is clear at 42 (0.89 served
  after cycle 20 on calm worlds) but weak at 12 (0.99 for the most frequent 12-producer winner).
M1 therefore gates on sparse-wide only; dense stays buildable and non-gating, to be revisited
with a demand-side retune. The 0.95 ceiling is unchanged.

### D11. Driver and parallelism
`python -m tokenomics.eval.gate` has three modes:
- `--mode screen`: roster only. Zero-baseline replacement, headroom, pruning, X/Y
  calibration.
- `--mode report`: subject plus frozen roster on a split.
- `--mode recompute`: verdict rebuilt from the CSV.

Jobs fan out over a `ProcessPoolExecutor`, one per core, with BLAS pinned to 1 thread
(the `bounded_rollout_orderer` pattern). Each completed row is appended immediately, so a
failed pod run can resume. Output is the EXP-dir layout: `spec.json` with provenance,
`episodes.csv`, `worlds.csv`, `regimes.csv`, `verdict.json`, and a `RESULT.md` skeleton.

### D12. Records
There are two experiments, in order:
1. **EXP-0044, calibration** (baseline-only): screening, headroom, roster pruning and the
   X/Y freeze. It freezes roster v1, registry v1 and parameters v1.
2. **EXP-0045, champion development report**: the portable stack at 12 and 42 producers
   on development worlds.

Each gets a `RESULT.md`, an index row and a HISTORY paragraph; EXP-0045 opens the
measurement arc. Before sizing the pods, single-episode timings at horizon ≤ 12 for each
family × size are measured locally (a quick one-shot diagnostic).

## Risks / Trade-offs

- **The portable contract fails on sparse-wide worlds** (E = 2, 14 products per layer at 42
  producers). →
  The report keeps contract failures separate from value, as EXP-0015 did. A contract
  failure is a reported finding, not a reason to drop the family.
- **The roster is too costly at the large size.** There are 28 + 12R member runs per
  world-episode (R = 5: 88). The 120-producer sizing made this real (task 8.3); the
  observation speed-up (8.4), on-demand reserves (8.6) and the move to 42 producers (8.7)
  answer it, re-sized in task 8.8 (`sizing.md`). → Measure timings first, then apply the
  development-only pruning rule. Scale pods; never the arithmetic.
- **Headroom retuning looks like tuning toward a result.** → Retuning uses baseline-only
  development evidence, happens before any candidate run, and every attempt is logged in
  EXP-0044.
- **The q = 0.05 calibration is too strict or too lax.** → It is frozen before any candidate
  run. A change makes a new parameter version, and earlier reports keep the version they
  used.
- **The champion is far below the bar at 42 producers.** → That is the expected result
  and it is the information M3 needs. The M1 exit is a complete, honest report, not a pass.
- **The domain outcome record perturbs existing behaviour.** → Fields are added only. The
  byte-identical fixture test reruns every existing fixture and compares every legacy
  episode statistic.

## Migration Plan

Changes are additive: new fields have empty or default values, new modules are new
entry points, and `run_episode` keeps its current seeding by default. Rollback means
reverting the change; no stored artifact depends on it until EXP-0044 freezes its files.

## Open Questions

Numeric defaults are written into the frozen files before first use. None of them changes
the approach or the tasks:
- R (random replicates)
- world and episode counts per family × size
- the sparse-wide fanout
- the calm schedule length
- the ratio tolerance
- the saturation ceiling
- the calibration quantile q
