# Champion Expert-Iteration Approach

Status: current roadmap after EXP-0006B

## Goal

Build a repeatable training loop that produces agents better than the best
human-defined and simple randomized policy envelope, then keeps improving them under
strict held-out gates.

The project has now crossed the first strict bar:

```text
v4 producer + M2-v3-stochastic orderer + v2 enriched obs
  > strict random-inclusive baseline envelope
```

The next practical goal is not another one-off experiment. It is an automated
champion-iteration loop:

```text
champion -> collect champion states -> search/planner improves them ->
distill candidate -> evaluate -> promote only if gates pass
```

Promotion remains human-approved. Automation should make the loop reproducible and
cheap to rerun; it should not silently change the scientific hypothesis or overwrite
the champion.

## Current Champion

Champion stack:

- Producer: EXP-0004 v4 producer clone.
- Orderer: EXP-0006B M2-v3-stochastic.
- Observation contract: enriched orderer observation v2.
- Checkpoint:
  `experiments/EXP-0006B-orderer-stochastic/artifacts/m2_v3_obs_stochastic.pt`

Champion result:

- Beats the strict envelope on held-out seeds `6,7,8,9`.
- Zero death.
- 40-eval-seed confidence audit clears the envelope at the lower 95% CI bound on all
  four held-out seeds.
- Mean envelope ratio is about `1.047`; the optional `1.10` stretch was not met.

This is the first accepted policy in the EXP-0001..0006 sequence that clears the
strict random-inclusive envelope everywhere.

## Evidence So Far

### EXP-0001..0004: Producer Discipline

End-to-end PPO failed at depth 3 because the producer learned topology-specific and
credit-starved behavior. Producer-first supervised/DAgger work fixed the core
producer discipline.

Accepted outcome:

```text
v4 producer clone ~= stock_aware producer
```

The producer is not the current bottleneck for the strict envelope work.

### EXP-0005: Learned Improvement Over Deployed Baseline

Holding the producer fixed, orderer search showed large reachable headroom over
`proportional`. M2-v1 learned enough to beat the deployed baseline:

```text
v4 + M2-v1 > stock_aware + proportional
```

But deterministic M2-v1 did not clear the strict random-inclusive envelope.

### EXP-0006A: Enriched Orderer Observation

EXP-0006A tested whether the deterministic clone gap was missing state. The enriched
v2 observation exposed producer/product/order state such as load, makeability,
product demand, and producer-product throughput.

Clean rerun result:

- Beats M0 on all four held-out seeds.
- Mean envelope ratio around `1.025`.
- Still misses strict envelope on `s6` and `s9`.

Conclusion:

```text
observability is a partial lever, but not sufficient
```

The canonical metric definitions for v2 observation live in:

```text
docs/adr/0051-orderer-observation-economy-metrics.md
```

### EXP-0006B: Stochastic Orderer Distillation

EXP-0006B kept the EXP-0006A recipe fixed and changed only the distillation/deploy
form:

```text
deterministic mean -> learned log_std + sampled orderer
```

Result:

- Clears strict envelope on all held-out seeds.
- Fixes the random-dominated hard cells `s6` and `s9`.
- Confirms the residual was routing diversity, not state.

Accepted outcome:

```text
M2-v3-stochastic is the current orderer champion
```

## Baseline Envelope

Every promoted champion must beat an envelope, not one convenient baseline.

Minimum envelope:

- `stock_aware + proportional`
- `stock_aware + even_split`
- `stock_aware + obs_price`
- `stock_aware + random`
- `random + random`
- current learned champion

The strict gate is evaluated per held-out topology, not only on a blended mean.

## Why This Is AlphaZero-Style

The environment is not a two-player game, so literal self-play is not the useful
analogy. The useful loop is expert iteration:

```text
current policy -> planner/search improves actions -> train policy/value models on
the improved decisions -> evaluate -> promote if better -> repeat
```

In AlphaZero terms:

- policy/student = learned producer/orderer actors
- planner = rollout/search oracle over candidate actions
- critic/value model = learned value estimator used later to make planning cheaper
- champion gate = held-out envelope evaluation

Current planner:

- sampled per-cycle orderer routing search, lookahead 12, K 12 by default

Future planner upgrades:

- elite-set / mixture teacher labels
- CEM routing search
- value-guided search
- joint producer-orderer search only after a producer-side headroom audit

## Automation Target

Add a champion-iteration driver. It should orchestrate existing pieces rather than
replace them.

Proposed command:

```bash
python -m tokenomics.train.champion_iterate \
  --registry champion.json \
  --out-dir experiments/EXP-0007-champion-iteration-r1 \
  --teacher samples=12,lookahead=12 \
  --distill stochastic_mle \
  --eval-seeds 40
```

The driver should perform:

```text
load champion registry
validate checkpoint/schema/fingerprint
collect states from champion rollouts
relabel states with search/planner
train candidate
evaluate candidate against envelope and champion
write experiment record
emit PROMOTE_RECOMMENDED=true/false
```

It should not silently overwrite the champion.

## Champion Registry

Keep a small checked-in or generated JSON document that names the active champion and
the gate configuration.

Example:

```json
{
  "champion_id": "m2_v3_stochastic",
  "checkpoint": "experiments/EXP-0006B-orderer-stochastic/artifacts/m2_v3_obs_stochastic.pt",
  "producer": "v4",
  "orderer_obs_version": 2,
  "checkpoint_version": 6,
  "held_out_topology_seeds": [6, 7, 8, 9],
  "eval_episode_seeds": 40,
  "strict_envelope": true,
  "primary_metric": "consumer_completed_units",
  "max_death_rate": 0.0
}
```

The registry gives the automation one source of truth for:

- active champion
- observation version
- held-out suite
- eval seed count
- promotion thresholds
- current envelope definition

## Automated Pipeline

### 1. Preflight

Fail fast before launching cloud work.

Checks:

- checkpoint exists
- checkpoint schema version is expected
- checkpoint fingerprint matches v2 enriched observation
- orderer observation dim matches the v2 spec
- stochastic head is present for stochastic champions
- default fast regression tests pass
- optional `pytest -m champion` passes or is explicitly skipped with a reason

### 2. State Collection

Roll out the current champion, not an older deterministic student.

Record:

- topology seed
- episode seed
- cycle
- role
- enriched v2 observation
- current champion action
- decode-active mask
- final episode outcome
- route/product/producer identifiers needed for diagnostics

For EXP-0007 orderer iteration, the producer remains fixed to v4 and only orderer
states need planner relabeling.

### 3. Planner Relabeling

For each selected orderer state:

- restore or replay the rollout prefix to the state
- run the search teacher from that state
- record the best action and score
- optionally record an elite set of good actions and their scores

Tracer-bullet mode can keep the existing L12/K12 teacher. Optimization work should
cache enriched views or rollout prefixes if label generation dominates runtime.

### 4. Candidate Training

Initial candidate family:

```text
stochastic_mle
```

That means:

- train mean + `log_std`
- use masked Gaussian NLL over decode-active orderer dims
- deploy by sampling from the learned Gaussian
- keep masked dims at init

Next candidate family if margin becomes the goal:

```text
elite_set_distillation
```

That means preserving multiple good routings for the same state instead of fitting a
single Gaussian around one chosen teacher action.

### 5. Evaluation

Evaluate candidate against:

- strict envelope
- current champion
- M0 `stock_aware + proportional`
- hard cells `s6` and `s9`

Report:

- mean consumer completed units
- 95% CI
- envelope ratio
- champion ratio
- death rate
- consumer rejected units
- stochastic order-independence check

Use at least 40 episode seeds for final promotion of stochastic candidates.

### 6. Experiment Record

Every run, accepted or rejected, writes:

```text
experiments/EXP-XXXX-title/
  spec.json
  RESULT.md
  summary.csv
  artifacts/
    checkpoint.pt
    heldout.csv
    logs/
```

The driver should also update nothing outside the experiment directory unless
promotion is explicitly approved.

### 7. Promotion Recommendation

The driver emits:

```text
PROMOTE_RECOMMENDED=true|false
```

Promotion requires human approval.

If approved, update:

- champion registry
- `experiments/README.md`
- `STATUS.md`
- memory/status pointer if used

## Promotion Gates

A candidate becomes eligible for promotion only if all are true:

- held-out topologies were not used for label generation
- checkpoint fingerprint matches the intended observation contract
- death rate is zero
- strict envelope ratio is greater than `1.00` on every held-out topology
- candidate mean beats current champion mean on the held-out suite
- no known hard cell regresses below the current champion
- stochastic eval is reproducible per episode seed and independent of topology list order
- result is recorded in an experiment directory

For close wins, use the stricter audit:

```text
lower 95% CI bound > envelope point
```

EXP-0006B now satisfies this stricter audit on all four held-out seeds.

## Regression Gates

Default tests should remain fast.

Use two tiers:

```text
pytest -q
  structural and contract tests

pytest -m champion
  slow champion envelope audit
```

Fast in-suite gates:

- checkpoint loads
- version/fingerprint/observation dims match
- v2 `obs_price` indexing matches v1 behavior
- stochastic RNG is scoped per episode seed
- stochastic action path produces finite clipped actions
- masked `log_std` behavior is as expected

Slow champion gate:

- evaluate champion on held-out seeds `6,7,8,9`
- 40 episode seeds
- strict envelope comparison
- zero death
- fail on order-dependent stochastic results

The slow gate can take minutes and should be opt-in.

## Runpod Automation

Use the detached client-server Runpod path for all real training runs.

Required behavior:

- start training under `setsid` on the pod
- write pid/log/exit markers
- allow client disconnect
- allow `--attach` resume
- pull artifacts automatically
- stop pod cleanly after run

The EXP-0006B run validated this path end to end. Future automation should call this
runner rather than hand-managing SSH sessions.

## Next Experiments

### EXP-0007: Champion Iteration Driver

Purpose:

```text
Automate one full champion -> candidate -> gate -> record loop.
```

Work:

- implement champion registry
- implement `tokenomics.train.champion_iterate`
- wrap existing orderer trainer/evaluator
- write experiment records automatically
- emit promotion recommendation

Accept:

```text
driver can reproduce an EXP-0006B-style candidate run and record a valid experiment
without manual stitching
```

### EXP-0008: Elite-Set Or Mixture Distillation

Purpose:

```text
Improve margin beyond M2-v3-stochastic, especially on random-dominated cells.
```

Only pursue if we want mean ratio closer to `1.10` or if champion iteration stalls.

Work:

- collect multiple high-scoring teacher actions per state
- train mixture/elite-set objective
- compare against M2-v3-stochastic

Accept:

```text
beats current champion on held-out mean without losing any strict-envelope cell
```

### EXP-0009: Value-Guided Planner

Purpose:

```text
Reduce search cost and increase effective planning horizon.
```

Work:

- train value model from rollout/search data
- validate candidate ranking quality offline
- use value for short-rollout + terminal-value scoring

Accept:

```text
value-guided planner matches or beats rollout-only teacher at lower compute
```

### EXP-0010: Producer-Side Headroom Audit

Purpose:

```text
Check whether producer-side or joint coordination headroom remains after the
stochastic orderer champion.
```

Work:

- keep M2-v3-stochastic orderer fixed
- run producer search/teacher probes
- compare v4 producer to producer-improved search

Accept:

```text
producer-side search improves over v4 under the champion orderer
```

Only if accepted should we open joint producer-orderer training.

## Joint Producer-Orderer Work

Do not restart naive joint PPO.

If joint work becomes justified, use planner-supervised expert iteration with strict
topology gates:

- role-specific actor losses
- topology-randomized training
- held-out and permutation gates
- no promotion on train-topology gains alone
- likely graph/entity architecture before broad random-depth claims

Joint search means the planner chooses producer and orderer actions as a coordinated
pair from the same state. It is more expensive and mixes two problems, so it is a
later-stage tool, not the next default move.

## Compute Priorities

Current bottlenecks:

- enriched observation/view construction during DAgger rollouts
- Python simulation/search rollouts
- stochastic eval seed count

Practical improvements before scaling:

- cache enriched views inside a cycle when multiple candidates read the same state
- cache/replay rollout prefixes for planner relabeling
- separate CPU-heavy label generation from GPU-friendly neural fitting only if
  profiling shows training dominates
- keep full 40-seed champion audit opt-in

## What Not To Do

Do not restart from-scratch PPO.

Do not pursue joint producer-orderer training before a producer-side headroom audit.

Do not promote a model because it beats M0 only; the strict envelope is now the
standard.

Do not overwrite the champion automatically.

Do not treat routed quantity as true demand. True demand remains product-level;
producer/product routed history is policy-dependent load history.

## Plain-English Summary

The project now has a strict-envelope champion. The next engineering task is to make
improvement repeatable:

```text
Keep a champion registry.
Roll out the champion.
Use search as the coach.
Train a candidate.
Evaluate against the strict envelope and the champion.
Write the experiment record.
Recommend promotion only when all gates pass.
```

That automated loop is the practical path toward stronger agents. It preserves the
lesson from the PPO failures: improvement must be planner-supervised, topology-gated,
and promoted only on held-out evidence.
