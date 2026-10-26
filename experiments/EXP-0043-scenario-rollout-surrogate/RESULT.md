# EXP-0043 — Scenario Rollout Surrogate

**Date:** 2026-08-27 (registered) · 2026-10-25 (cancelled) | **Status:** CANCELLED, NOT RUN
| **Driver:** `tokenomics.eval.scenario_rollout_surrogate` (on branch `archive-1`) | **Spec:** [spec.json](spec.json)

## Question

Can a scenario-conditioned relational GNN certify zero-gain depth-2 bump evaluations
inside the EXP-0042 planner and remove at least 70% of its real rollout calls, without
losing depth-2 decisions or closed-loop value?

## Method

The driver was implemented and tested locally at smoke scale (commit `0d41d8b`). No
labels were collected, no gate seeds were inspected, and no pod time was spent.

## Result

None. The run was cancelled before any data collection.

## Conclusions

1. **Cancelled by the plan, not by evidence.** The agreed baseline plan
   (`openspec/changes/establish-project-baseline/design.md`, D5) rejects this hybrid.
   The surrogate only skips streams it predicts are exact zero; every other searched
   stream still runs real depth-2 rollouts at decision time. That cannot meet the
   `agent-acceptance` requirement of bounded per-decision cost, which forbids
   multi-cycle rollouts at decision time.
2. **A cheaper teacher is not on the plan's path either.** PoC-A uses the EXP-0042
   teacher directly, and the M4 learning approach (evolution strategies on a residual)
   does not need per-decision teacher labels.
3. The code was removed from `new-model` and is preserved on branch `archive-1`
   (commit `7aeef1f`).

## Follow-up

Reopen this only if teacher cost becomes the measured bottleneck, for example in the
deferred M4 teacher warm-start, and only after the fresh target-stability audit that
D5 requires.
