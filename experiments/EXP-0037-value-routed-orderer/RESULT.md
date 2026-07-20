# EXP-0037 — Value-Routed Orderer

**Date:** 2026-07-15 · **Driver:** `tokenomics.train.value_router` · **Reproduce:** see [`spec.json`](spec.json)

## Question

EXP-0036 showed that a learned orderer cannot beat a fixed mid-blend when the action is
"choose one predefined blend policy." Did we test the wrong action space?

This experiment asks whether a trained orderer can learn the **routing weights directly**:

```text
(order, eligible producer, global context) -> marginal value score
softmax over eligible producers -> routing weights
```

The floor is fixed `blend_0.5`, not the old champion.

## Starting Hypothesis

The EXP-0036 blend vocabulary is too narrow. `even_split`, `blend_0.75`, `blend_0.5`,
`blend_0.25`, and `proportional` are all capacity-shaped routers. They can only change how
strongly the orderer follows current throughput; they cannot learn that a lower-capacity
producer is strategically better because of inventory, inbound stock, TTL, queue pressure,
recipe position, or downstream scarcity.

## Proposed Policy Shape

Start conservative:

```text
base_logits = blend_0.5(order, producers)
residual_logits = model(order, producer, context)
final_logits = base_logits + clip(residual_logits, -r, r)
route = softmax(final_logits over eligible producers)
```

This keeps `blend_0.5` as the zero-cost floor but lets the model override capacity when
value evidence is strong.

## First Diagnostic

Before training a deployable policy, build an offline representability probe:

1. Collect states from the fixed `blend_0.5` trajectory on the EXP-0036 s29/s77/s45 tasks.
2. For each order, score one-step or short-rollout producer-weight perturbations around
   `blend_0.5`, not just the five predefined blend policies.
3. Train a pairwise/value model that predicts which eligible producer should receive more
   marginal weight.
4. Evaluate by rollout with the residual router.

## Metrics

```text
learned_cu
blend_0.5_cu
best_constant_blend_cu
search_cu
death rate
delta_vs_blend_0.5
```

## Stop Rule

Stop if the residual router does not beat fixed `blend_0.5` on at least 2 of the 3 gate
tasks by deployed rollout units. Do not trust table error alone.

## Status

**CLOSED.** The full 3-task gate ran on a Runpod CPU pod on 2026-07-16 and failed the
predeclared deployed-rollout stop rule: only 1 of 3 tasks beat fixed `blend_0.5`.

### What was built

- `eval/value_router.py` (torch-free measurement):
  - `residual_router(base, score_fn, clip)` — the policy shape. `final = base + clip(residual)`;
    a zero score is byte-identical to `blend_0.5` (the floor), `clip` bounds any override.
  - `marginal_value_labels(...)` — the teacher. At a cycle, for each eligible producer, nudge
    its stream logit by `+delta`, follow `base`, roll `lookahead` cycles under a fixed producer;
    label = consumer units gained vs the unperturbed `blend_0.5` rollout. (Marginal value of a
    logit bump — an improvement signal, NOT throughput imitation.)
  - `evaluate_router_gate(...)` — the DoD instrument. Deploys the router by rollout at the
    archive-anchored seeds (`_episode_seeds`), returns per task `learned_cu / blend_cu /
    delta_vs_blend / *_dead / beats_floor`. Judged on deployed units only.
- `train/value_router.py` (torch): white-box per-(stream,producer) feature rows
  (`producer_row_features`: throughput, free/reserved/headroom capacity ratios, input
  availability, own inventory — the value facts a blend cannot see), a producer-count-agnostic
  shared-row `ResidualScoreModel`, masked-MSE `fit_residual_scorer` on centered marginal labels,
  the `to_score_fn` deploy adapter, `collect_router_samples`, and the `run_value_router` driver +
  CLI (emits `spec.json`/`summary.csv`/`residual_scorer.pt`).

### Review corrections (2026-07-16, three P1s fixed before any pod run)

1. **Label isolation.** The marginal-bump teacher gated only on cycle, so in the joint-stack env
   (many product-stream ordering agents per cycle) the `+delta` nudge leaked into every other
   agent sharing that stream/producer index — a confounded label. `_one_cycle_bump` /
   `marginal_value_labels` now take the labeled `agent` and perturb only its decision
   (`test_bump_perturbs_only_the_labeled_agent`).
2. **Env contract.** The gate + collector built a plain `TokenomicsAECEnv`, but EXP-0036 measured
   its `blend_0.5` floor in `joint_stack_env` (entity producer obs + corrected v4 product-stream
   ordering). Both now use `joint_stack_env` + `run_episode`, so `blend_cu`/`delta_vs_blend` are
   comparable to the EXP-0036 floor.
3. **Producer loader.** The EXP-0035 champion is a promoted producer *scorer* checkpoint (as
   EXP-0036 froze it), not a legacy GaussianActor; the CLI `--prod-ckpt` now loads it via
   `scorer_producing_policy(load_producer_score_model(...))` instead of `_load_prod_closure`
   (which would raise a version mismatch). Verified end to end against
   `EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt`.

Differs from EXP-0012/0032 in all three required ways: base is `blend_0.5` (not free-form
scores), label is marginal value (not throughput), acceptance is deployed rollout (not table
error).

### Tests

`tests/test_value_router.py` + `tests/test_value_router_train.py`. The value-router targeted
suite is **12 passed** after the P2 auditability fixes. The earlier full-suite run reported
green during implementation; the latest independent recheck was targeted, not full-suite.

## Gate result (2026-07-16, pod `ymys3x82siomyk`, ~40 min CPU)

Full params: `--delta 1.0 --lookahead 8 --eval-seeds 3 --horizon 100 --epochs 300 --clip 2.0`,
EXP-0035 producer scorer frozen on `joint_stack_env`, 13,940 labeled decisions collected.

| task | learned_cu | blend_cu | Δ | learned_dead | blend_dead | beats_floor |
|------|-----------:|---------:|------:|-------------:|-----------:|:-----------:|
| s29  | 856.3      | 891.0    | −34.7 (−3.9%) | 0.00 | 0.00 | ✗ |
| s77  | 696.0      | 692.3    | +3.7 (+0.5%)  | 0.00 | 0.00 | ✓ |
| s45  | 175.0      | 400.0    | −225 (−56%)   | 0.67 | 0.33 | ✗ |

**1/3 beats fixed `blend_0.5` → the predeclared stop rule (≥2/3) fires. Branch CLOSED.**

## Conclusions

The value-routed residual orderer does **not** clear the floor. **The verdict rests on deployed
rollout** (1/3 beats `blend_0.5`, stop rule fired) — that is the primary, auditable evidence and
it does not depend on the loss trajectory. Root cause, with the strength of each claim scoped:

1. **The residual adds harmful noise, not a stable improvement (primary, deployed evidence).** The
   lone "win" (s77 +0.5%) is within seed noise and in-sample. On s45 the residual is actively
   destructive: −56% units **and** it drives the economy to system death more often than the floor
   (death rate 0.67 vs 0.33). Keeping `blend_0.5` as the base with a ±2.0 logit clip bounded the
   damage but did **not** prevent it — an ill-fit score still steers routing into collapsing
   states. This alone answers the handover's stop rule: **no**.
2. **The fit was flat over the tried feature set (supporting, scoped evidence).** Training loss
   barely moved — epoch 0 = 13.16, epoch 299 = 12.97 (centered MSE, units²), rising over the last
   ~10 epochs (from the pod-run stdout of the original run; the driver now persists
   `loss_history.csv` so a re-run is auditable from the pulled dir). The shared-row scorer could
   not fit the marginal-value-of-logit-bump labels **from `producer_row_features`** — 7 white-box
   producer facts (throughput, capacity ratios, input availability, own inventory). This is
   consistent with the **same failure family EXP-0036 diagnosed** (flat/near-tied per-cycle routing
   targets), now from a value-residual action space rather than a blend selector.

   **Scope caveat (P2 review):** the rows carry *producer* state and `oi.product`, but **not**
   order-level context (order age / TTL / backlog / same-product order multiplicity). Two decisions
   that share a product and producer state but differ in order age can map to *identical rows with
   different labels* — so a flat fit here is **consistent with** an unlearnable marginal-value
   target but does **not prove** it; feature aliasing on this limited row is not excluded. Cheaply
   distinguishing the two would mean adding order/backlog fields to the row and re-fitting. We did
   **not** do that: the deployed-rollout verdict (claim 1) already fires the stop rule, and the
   handover forbids chasing a richer scorer past a failed 3-task gate.

So the genuinely-new question the handover posed — "can a constrained value-residual producer
scorer improve over `blend_0.5` without relearning noisy one-off search vectors?" — is answered
**no** for this EXP-0037 implementation on the decisive primal-tight gate: the residual harms
more than it helps on deployed units, and the fit showed no learnable signal over the
producer-only feature set tried.

Per the handover Non-Goals, we did **not** run the 81-row archive (the 3-task gate did not pass).
This closes the value-routed-orderer branch. `blend_0.5` remains the zero-cost orderer floor
(EXP-0036), and no learned orderer — blend-selector (EXP-0036) or value-residual (EXP-0037) —
has beaten it on primal-tight worlds.

### Auditability corrections (2026-07-18, P2 review — code fixed, result unchanged)

A second review flagged that the pulled artifacts under-supported the *supporting* evidence. The
headline gate table is internally consistent with `gate/summary.csv` and the verdict is unchanged;
these are reproducibility fixes so a re-run is fully auditable from the pulled dir:

- **Model seed.** `run_value_router`/CLI now take `--seed` (default 0), seed torch model init, and
  record it in `spec.json` — the fit (hence `learned_cu` and the 2-of-3 count) is reproducible.
  The original pod run did not pin this; a re-run at a fixed seed would.
- **Loss history.** The driver now writes per-epoch `loss_history.csv` (was: only `final_loss`).
  The epoch-0/299 numbers cited above are from the **original pod run's stdout**, not a persisted
  artifact — a seeded re-run regenerates them auditable.
- **Per-seed gate rows.** The gate now emits `gate_rows.csv` with one row per (task, seed, policy),
  so the s77 "seed noise" read and the s45 death counts (0.67 vs 0.33) are checkable from raw rows
  rather than only means. `router_gate` returns `(aggregate, per_seed)` in one rollout pass.
- **Feature-set scope.** Conclusion 2 narrowed: the flat fit is over the producer-only feature row,
  which lacks order-age/TTL/backlog context — consistent with an unlearnable target, not proof
  (feature aliasing not excluded). See the Scope caveat above.

The `gate/` artifacts below are from the **original** pod run (pre-fix), so they do not yet contain
`loss_history.csv` / `gate_rows.csv`; those are produced by the reproduce command in `spec.json`.

Artifacts: [`gate/summary.csv`](gate/summary.csv), [`gate/spec.json`](gate/spec.json),
[`gate/residual_scorer.pt`](gate/residual_scorer.pt).

## Follow-up

Do **not** start by adding another learned orderer. The next experiment should first build an
error microscope: compare the marginal-value teacher's producer ranking against the actor's
predicted residual ranking on the same decisions, then inspect the largest mistakes and feature
aliases. If the mistakes point to missing order/backlog/TTL context, enrich the observation in a
separate follow-up. If the labels remain inconsistent for identical-looking states, the problem is
teacher/target noise rather than model capacity.
