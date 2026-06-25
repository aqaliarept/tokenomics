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

The next practical goal is not another one-off experiment. It is a two-step
infrastructure move:

```text
EXP-0007A: simplify the ordering-agent contract with stable product-stream agents
EXP-0007B: automate champion iteration on that stable contract
```

The automated loop should then be:

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
- producer-side constrained planner with learned local scores
- joint producer-orderer search only after producer size-holdout gates pass

## Automation Target

Add a champion-iteration driver after EXP-0007A proves the product-stream ordering
contract preserves behavior. The driver should orchestrate existing pieces rather
than replace them.

Do not anchor the first automation implementation to the old mixed-product ordering
contract unless it is only a temporary compatibility adapter.

Proposed command:

```bash
python -m tokenomics.train.champion_iterate \
  --registry champion.json \
  --out-dir experiments/EXP-0007B-champion-iteration-r1 \
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

For EXP-0007B orderer iteration, the producer remains fixed to v4 and only orderer
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

### EXP-0007A: Product-Stream Ordering Refactor

Purpose:

```text
Reduce mixed-product slot complexity by moving to stable source-product ordering
agents while preserving baseline behavior.
```

This is a domain/interface refactor, not a new training objective. See:

```text
docs/adr/0052-stable-product-stream-ordering-agents.md
```

Accept:

```text
baseline behavior preserved; PettingZoo/API/lifecycle tests pass; observation/action
contract is simpler and versioned
```

Only retrain the champion on the new product-stream contract after this parity gate
passes.

### EXP-0007B: Champion Iteration Driver

Purpose:

```text
Automate one full champion -> candidate -> gate -> record loop after the
product-stream ordering contract is stable.
```

Work:

- implement champion registry
- implement `tokenomics.train.champion_iterate`
- wrap existing orderer trainer/evaluator
- write experiment records automatically
- emit promotion recommendation

Accept:

```text
driver can reproduce an EXP-0006B-style candidate run on the stable ordering-agent
contract and record a valid experiment without manual stitching
```

### EXP-0008: Elite-Set Or Mixture Distillation — DONE, REJECT (elite-set, K=6 ε=0.1)

Purpose:

```text
Improve margin beyond M2-v3-stochastic, especially on random-dominated cells.
```

Work done:

- elite teacher: `orderer_oracle._search_elite_pols` / `_select_elite` keep the per-state
  within-`ε` elite routing set (capped at `K`), not just the argmax
- `orderer_improve` relabels each state with the elite set (one sample per elite action,
  `--elite-k/--elite-eps`); the existing masked Gaussian NLL then fits the elite
  centroid + spread. Checkpoint schema unchanged → the champion gate is reused as-is
- driven through `champion_iterate --distill elite_set` (CPU Runpod, 40 eval seeds)

Result (`experiments/EXP-0008-elite-set-distillation/`):

```text
PROMOTE_RECOMMENDED=false. Candidate valid (zero death, clears strict envelope on all 4
held-out: 1.041/1.097/1.048/1.035) but held-out mean 511.7 < champion 513.4, and hard
cell s9 regressed (516.0 < 524.5).
```

The finding is the per-seed split: elite-set **helps** the `proportional`-dominated cells
(s6 +2.0, s7 +6.4) and **loses** the `stock_aware+random` cells (s8 −6.7, s9 −8.5) — the
*opposite* of the hypothesis that it would lift the random-dominated cells. EXP-0006B's
learned `log_std` around the argmax already supplies the useful routing diversity; adding an
elite **centroid** target shifts the mean off-argmax and over-spreads, washing out the
high-value random-cell routings. **Elite-set is not the margin lever at these settings; the
champion stays M2-v3-stochastic.**

Accept (was):

```text
beats current champion on held-out mean without losing any strict-envelope cell  -> NOT MET
```

Not ruled out (cheap follow-up, not pursued): a **tighter** elite (small `ε`≈0.02 or
`elite_k=2`, true near-ties only) might keep the s6/s7 gains without the s8/s9 regression —
audit at `--eval-seeds 10` first, promote-audit at 40 only on a candidate that wins.

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

Machinery landed (TDD, 2026-06-25): the planner scored every candidate routing by a full
`lookahead`-cycle deepcopy rollout (`orderer_oracle._score_candidates`). EXP-0009 replaces the
rollout tail with a learned value: roll only `short` cycles under a candidate, then add
`V(global_state)` for the remaining window. The value model reads the existing versioned system
state (`env.global_state()`, the centralized-critic input) — **no new state contract**.

New pieces (all unit-tested, `tests/test_value_planner.py`):

- `train.value_model`: `ValueModel` (MLP over `global_state_dim`), `collect_value_samples`
  (per-cycle `(state, units over next W cycles)` targets), `fit_value_model` (MSE),
  `value_closure`, and `run_value_planner_eval` (the accept harness) + CLI `main`.
- `eval.orderer_oracle`: `_rollout_window(...)→(score, cycles)`, value-aware
  `_score_candidates_valued`, the single `planner_orderer_episode(..., value_fn=None)` driver
  (`value_fn=None, window=lookahead` reproduces the rollout-only teacher; set + short window =
  value-guided), and the `ranking_agreement_episode` offline diagnostic. `_score_candidates`
  was refactored onto a shared `_candidate_list` so both scorers see identical candidates.

Local tracer (depth-2, K=3, lookahead=4 short=1): value-guided **matched** rollout-only units
(58.0 = 58.0) at **50 vs 170 simulated cycles** (~3.4× fewer), 0.90 ranking agreement — the lever
works in miniature. The real depth-3 accept run is Runpod-only (CLAUDE.md):

```bash
python -m tokenomics.train.value_model --depth 3 --prod-ckpt <v4.pt> \
  --train-seeds 0,1,2,3,4,5 --held-out-seeds 6,7,8,9 \
  --lookahead 12 --short 3 --samples 12 --csv-out heldout.csv
```

Result (Runpod CPU, depth-3, held-out 6–9) — see `experiments/EXP-0009-value-guided-planner/`:
**ACCEPT the compute lever; win is short-rollout sufficiency, NOT the learned value.**
Value-guided units beat the rollout-only teacher on 2/4 held-out (s6 +3.8, s7 +6.0) and tie
within 0.15% on the other 2 (s8 −0.4, s9 −0.8) — ratios 1.008 / 1.012 / 0.999 / 0.999, held-out
mean **+2.15 units (+0.4%)** at **3.82× fewer simulated cycles** (4455 vs ~17010). But `V` collapsed to ≈mean
(fit MSE plateaued at 146.6 instantly; ranking agreement only 0.13–0.24) → the savings come
from a 3-cycle rollout already being outcome-sufficient at depth-3, not from the learned tail.
Usable as a cheaper teacher now. Review fix landed: terminal-state value tail zeroed (no phantom
score at death/horizon).

Step-3 diagnostic (local, no cloud) settled the value lever. The collapse had two layers:
(1) **optimization** — raw `global_state` (dims 1e-2..1e3, ~36/193 dead) saturates the tanh net
→ mean-collapse (`pred_std=0`, MSE=variance); fixed by standardizing inputs+targets in
`ValueModel` (`set_norm`), regression-tested. (2) **the wall** — no cross-topology generalization:
trained on topologies 0–5, **held-out R² is negative for every model incl. linear ridge**
(−0.10 → −6.6), held-out corr ≈ 0.15 ≈ the planner's 0.13–0.24 ranking agreement; the earlier
in-sample R²=0.70 was memorization (train R²=1.0). Units-to-go from `global_state` doesn't transfer
at 6 train economies → **stop the value lever (don't rescue blindly), proceed to EXP-0010A.** The
compute lever (short-rollout teacher) stands.

### EXP-0010A: Producer Entity View

Purpose:

```text
Give the single producer orchestrator a product/capacity/entity view that is less
dependent on arbitrary product IDs, capacity IDs, and row permutations.
```

The producer should remain one coordinated agent because products compete for shared
capacity, inventory, and investment state. Do not split producers into independent
product/input stream agents. See:

```text
docs/adr/0053-entity-factored-producer-orchestrator.md
```

Work:

- add a domain-owned producer entity view
- encode capacity-type rows, product rows, input-product rows, reservation/backlog
  rows, and relation masks
- keep the existing producer command and production auction unchanged
- add schema/fingerprint versioning for the new producer contract
- add product/capacity relabeling and row-order permutation gates
- replay or clone existing `stock_aware` / v4 behavior through the new contract
- correct requested-load metrics as previous/most-recent auction pressure, not
  future current-cycle bids
- gate all makeability and bottleneck fields by producer-product eligibility

Accept:

```text
baseline producer behavior preserved; product/capacity relabeling gates pass;
row-order invariance/equivariance gates pass; checkpoint compatibility is versioned
```

### EXP-0010B: Deterministic Producer Planner

Purpose:

```text
Build a scalable producer controller that uses the entity view but does not depend
on a learned neural allocator for feasibility.
```

Work:

- implement transparent local scores for product urgency, input value, capacity
  bottleneck value, and investment ROI
- implement a deterministic planner/decoder that enforces inventory, funding,
  capacity, recipe, and running-investment constraints
- emit the existing `ProducingAgentCommand`
- compare against v4 under the current stochastic orderer champion
- run row-order, product-relabeling, capacity-relabeling, and synthetic size tests

Accept:

```text
planner preserves or improves v4 behavior, never violates producer constraints, and
runs on larger synthetic product/capacity counts
```

### EXP-0010C: Learned Local Producer Scorer

Purpose:

```text
Learn local economic scores inside the deterministic planner instead of training a
monolithic producer allocator.
```

Work:

- keep the deterministic planner responsible for feasibility
- train shared local row/edge scorers from planner/search/rollout data
- score product urgency, input replenishment value, capacity bottleneck value, and
  investment ROI
- use sparse product-capacity and product-input relations rather than dense
  all-product interactions by default
- compare against deterministic planner and v4 under the current stochastic orderer

Accept:

```text
learned scorer improves held-out mean without strict-envelope, feasibility, death,
or permutation regressions
```

### EXP-0010D: Producer Size-Holdout Audit

Purpose:

```text
Test the actual train-small/run-large claim for producers.
```

Work:

- train on small generated topologies
- evaluate on larger product/capacity counts such as `8`, `16`, `64`, and `400`
  where runtime permits
- require row-order, product-relabeling, and capacity-relabeling gates at each size
- report runtime and memory scaling separately from reward quality

Accept:

```text
small-topology training transfers to larger producer problems without feasibility,
death, relabeling, or strict-envelope regressions
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
- producer planner/scorer size-holdout evidence before broad random-depth claims

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

Do not pursue joint producer-orderer training before the producer entity view,
deterministic planner, learned-scorer audit, and size-holdout evidence.

Do not split a producer into independent product/input producer agents. Products
compete through shared producer capacity, inventory, and investment state; the
producer remains the orchestrator.

Do not treat a larger producer tensor plus a bigger neural network as the path to
train-on-4/run-on-400 generalization. Global feasibility belongs in the planner;
learning should supply local scores.

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
