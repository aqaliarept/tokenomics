# EXP-0038 Handover — Actor/Teacher Error Microscope

## Plain Goal

**EXP-0038 is not trying to improve the orderer yet.** It does not train a new orderer, tune the
residual, or ship a policy. Its single job is to explain **why the EXP-0037 actor learns the
teacher badly** — by comparing the teacher's marginal values against the actor's scores on the
*exact same producer choices*, then identifying whether the actor's wrong choices correlate with
**missing order/backlog/TTL/context features** or with **flat / noisy teacher labels**.

For each real routing decision, compare:

```text
teacher says: send more weight to producer X
actor says: send more weight to producer Y
state says: what is different between X and Y?
```

The comparison is over the identical eligible producer set per decision, so a disagreement is a
genuine actor mistake, not a mismatch of what each side was scoring. The two candidate causes it
must discriminate:

- **Missing observation.** Wrong choices cluster where the teacher-best and actor-best producers
  differ on facts the actor cannot currently see (order age / TTL / backlog / same-product backlog
  / scarcity / recipe position) — and, more sharply, decisions with *identical actor features* map
  to *conflicting* teacher picks (feature aliasing). → adding those specific features is justified.
- **Flat / noisy teacher target.** The teacher labels are near-constant across eligible producers
  (nothing to learn) or inconsistent for equivalent states (the target itself is the blocker). →
  the fix is the teacher/target, or stopping learned-orderer work — not more features.

This is a diagnosis, not a fix: it names which of the two is dominant, with evidence from the
mistake dump, before any new orderer strategy is proposed (root-cause-before-next-strategy).

## Inputs

- EXP-0036/0037 gate tasks: `s29`, `s77`, `s45`.
- Fixed producer: EXP-0035 producer scorer.
- Base routing floor: `blend_0.5`.
- Teacher: EXP-0037 `marginal_value_labels` around `blend_0.5`.
- Actor: EXP-0037 `ResidualScoreModel` under a fixed seed, or the pulled
  `gate/residual_scorer.pt` when reproducing the original gate.

Prefer a seeded rerun path for auditability because the current EXP-0037 `gate/` artifact is
pre-P2 and lacks `loss_history.csv` / `gate_rows.csv`.

## Handoff State

**The EXP-0038 dump generator has run on the real s29/s77/s45 gate tasks.** The pulled dump is in
[`dump/`](dump/). This experiment is now a scoped current-row diagnosis, not final closure of all
observation enrichment.

Built (see `PROGRESS.md` 2026-07-19 entries):

- `tokenomics.eval.value_router_errors` — the microscope. Reuses the EXP-0037 teacher
  (`marginal_value_labels`) and actor (`ResidualScoreModel`), so both signals match the gate.
  Public surface: `decision_metrics` (teacher-best vs actor-best, `decision_regret`, `top1_match`,
  tie-aware `rank_correlation`), `collect_error_records`, `decision_row` / `producer_rows`,
  `summarize`, `top_mistakes`, `alias_groups`, `fit_reproducible_actor` (seeded rerun of the gate
  actor), and `run_value_router_errors` + `main()` emitting `decisions.csv`, `producers.csv`,
  `top_mistakes.csv`, `alias_groups.csv`, `summary.csv`, `spec.json`.
- Tests: `tests/test_value_router_errors.py` (torch-free) + `tests/test_value_router_errors_run.py`
  (torch-guarded), 15 passed; full suite 670 passed. Local smoke over 2 depth-3 scale tasks writes
  all six files — schema + mistake/alias extraction confirmed. **Those toy-task numbers are schema
  proof only, not decision evidence.**
- Review corrections applied (do not re-introduce): CLI loads the producer scorer from
  `tokenomics.train.producer_scorer`; `rank_correlation` reports 0.0 for flat labels (constancy on
  raw values, tie-aware ranks); alias `feature_hash` is over exact float32 rows (no rounding, so
  near-but-not-identical features never collapse into a false alias).

Post-run correction: EXP-0038 proved that the current 7-feature actor row aliases the teacher
target, but it did **not** prove that every deployable observation would fail. The continuation is
[`EXP-0038A`](../EXP-0038A-missing-observation-audit/HANDOVER.md): audit which order/system/history
fields, if any, make the teacher label a stable function of deployable state. Do not train a new
actor before that audit passes.

Current EXP-0037 state to inherit:

- `tokenomics.eval.value_router` contains the corrected residual router, isolated
  marginal-value teacher, and `joint_stack_env` gate.
- `tokenomics.train.value_router` contains the corrected seeded residual fit path and now emits
  `loss_history.csv` plus per-seed `gate_rows.csv` on rerun.
- The checked-in `EXP-0037/gate/` artifacts are from the original pre-P2 pod run. They are useful
  for the headline gate result, but not enough for the new microscope. Prefer regenerating rows
  from code under a fixed seed.
- Latest targeted value-router tests were started before this handoff prep and interrupted by the
  user; the previous completed targeted run was 12 passed. The next agent should rerun the targeted
  tests before changing behavior.

## First Implementation Pass

Add a diagnostic driver, preferably separate from the training driver, such as:

```text
tokenomics.eval.value_router_errors
```

The first pass should be local-smoke sized:

```text
1 task, horizon 12, eval_seeds 1, lookahead 2, epochs 3
```

It should prove the file schema and mistake extraction before any pod spend. After the smoke is
green, run the real s29/s77/s45 gate-sized dump on a pod only if needed.

Recommended implementation shape:

1. Reuse `collect_router_samples` / `marginal_value_labels` so teacher labels match EXP-0037.
2. For each labeled decision, compute teacher-best producer and actor-best producer over the same
   eligible producer set.
3. Record the teacher value at the actor pick and at the teacher pick; their difference is
   `decision_regret`.
4. Emit the decision-level and producer-level CSVs first; derive `top_mistakes.csv`,
   `alias_groups.csv`, and `summary.csv` from those rows.
5. Keep conclusions descriptive. Do not claim "missing observations" until the mistake rows show
   which missing fields separate the actor's wrong choice from the teacher's choice.

## What To Log

Decision-level row:

```text
task_id
episode_seed
cycle
agent
order_id
stream_index
product
quantity
order_age / TTL / due information if available
backlog size and same-product backlog count
teacher_best_producer
actor_best_producer
teacher_margin
actor_margin
teacher_value_at_actor_pick
teacher_value_at_teacher_pick
decision_regret
rank_correlation
```

Producer-level row for each eligible producer:

```text
teacher_label
actor_score
base_logit
final_logit
eligible
free capacity by required type
reserved capacity by required type
input inventory + inbound
recent routed/completed/rejected stats
recipe relation / product depth
global scarcity summaries
```

Mistake row:

```text
teacher-best producer facts
actor-best producer facts
delta facts = teacher_best - actor_best
```

Alias row:

```text
feature_hash
n_decisions
label_variance
teacher_best_distribution
example decisions
```

## Required Analyses

1. Sort the top 50 mistakes by `decision_regret`.
2. Bucket mistakes by teacher margin: high-margin misses matter most.
3. Compare teacher-best vs actor-best feature deltas.
4. Check whether missing order/backlog/TTL fields separate contradictory labels.
5. Report whether adding specific observations is justified.

## Definition Of Done

- `summary.csv` says how often actor top-1 matches teacher top-1 and how much teacher regret
  actor choices incur.
- `top_mistakes.csv` makes the largest failures inspectable without rerunning.
- `alias_groups.csv` shows whether identical current actor features map to conflicting labels.
- The conclusion names a concrete next action:
  - add specific order/backlog/TTL features;
  - change target/teacher;
  - or stop learned orderer work.

## Non-Goals

- Do not run the 81-row archive.
- Do not train a new policy as the main deliverable.
- Do not claim "missing observations" unless the mistake dump identifies which fields are
  missing and why they explain teacher/actor disagreement.
