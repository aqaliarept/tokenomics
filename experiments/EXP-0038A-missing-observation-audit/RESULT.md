# EXP-0038A — Missing Observation Audit

**Date:** 2026-07-20 · **Driver:** TBD · **Reproduce:** see [`spec.json`](spec.json)

**Status: OPEN.** This continuation exists because EXP-0038 proved the current 7-feature actor row
aliases the marginal-value teacher, but did not prove that every deployable observation would fail.

## Question

Which information is missing from the actor observation, and is that information deployable at
decision time?

The teacher can evaluate a candidate by rolling the whole environment forward. The actor cannot do
that in the real application. EXP-0038A must determine whether the useful teacher signal can be
explained by observable state, or only by rollout/emulation-only facts.

## Planned Read

- If adding deployable order/system/history features makes labels stable and improves held-out
  probe regret, observation enrichment is licensed.
- If only oracle/future/rollout facts make labels stable, the marginal-value teacher is not a
  deployable actor target.
- If no deployable block helps, stop this target and change the teacher/objective.

No result yet.
