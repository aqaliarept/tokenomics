# Reward recalibration: fulfilment-only objective at scale

Status: accepted

## Context

ADR-0040 Finding B: at producer scale the per-cycle reward was dominated by
*instrumental producer-order churn*, so **do-nothing had the least-negative return** —
a return-maximizing policy would degenerate to inaction. The old reward was

```
+10 * consumer_completed_units            (objective)
+1  * (investment + replenishment completed)   (instrumental)
-1  * in_progress                              (transient drag)
-5  * consumer_rejected_units                  (unmet demand, per unit)
-5  * (investment + replenishment rejected)    (producer-order churn)
-0.1 * held                                    (B6 overstock)
```

Measured attribution (`build_scale_config(7, depth=4)`, obs_stock_aware): consumer
reward **+1450**, producer-order rejection **−9770**, in-progress **−1622**, consumer
rejection **−2220**. The objective was an order of magnitude below the churn.

Two structural reasons, both tied to **scale**:

1. **Producer-order terms scale with producer count × cycles.** Investment/replenishment
   rejected (−5) and in-progress (−1) accumulate across every producer every cycle, so
   at 16–20 producers they dwarf the objective, which scales only with the 3 consumers.
2. **Per-unit consumer rejection scales with demand, not policy quality.** At depth 4–5
   demand structurally outstrips a deep chain's throughput (e.g. depth-5 worst seed:
   14 units fulfilled, ~580 rejected). The −5/unit term therefore makes a struggling-
   but-alive system score *below* a do-nothing system that dies early and so generates
   no further demand to reject. No per-unit rejection weight > 0 survives the depth-5
   worst case.

## Decision

The training reward is the **consumer-fulfilment objective minus a per-producer-
normalized holding penalty** — nothing else (`env/aec_env.py::cycle_reward`):

```
reward = 10 * consumer_completed_units  −  0.1 * held / num_producers
```

- **Dropped** the instrumental terms: investment/replenishment completed (+1),
  in-progress (−1), producer-order rejection (−5), and per-unit consumer rejection
  (−5). Rejecting demand is still penalised — *implicitly*, as the forgone +10/unit.
  Survival is still rewarded — *implicitly*, since only a live system keeps completing.
- **Normalized** the B6 holding penalty by producer count. Total held inventory grows
  with the number of producers (deep pipelines must buffer inputs at every level), so a
  flat per-unit penalty over-punished scale; dividing by `num_producers` keeps its
  per-producer weight comparable at phase1 and at scale. `cycle_reward` gains a
  `num_producers` parameter (default 1 = no normalization for unit tests).

## Considered options

- **Keep rejection at a smaller weight (−1, −2).** Rejected: even −1/unit fails the
  depth-5 worst case (≈580 rejections × 1 ≫ 14 units × 10).
- **Add an explicit per-cycle survival bonus.** Rejected: it creates a "just survive,
  never fulfil" reward plateau (e.g. always_restore) that can trap RL before it learns
  the pipeline. Implicit survival (only the living fulfil) avoids the plateau.
- **Keep `investment_completed` (+1) for a denser bootstrap signal.** Rejected: at
  phase1 it contributed +331 of the ~1266 "bar" — pure churn reward, exactly the
  distortion to remove. Bootstrapping is a `train/` concern (curriculum), not the
  objective's.

## Consequences

Return now tracks fulfilled units across scales. Acceptance check — *worst case over
seeds*, obs_stock_aware + proportional vs noop (noop's return is exactly 0: it fulfils
and holds nothing):

| scale  | producers | obs_stock_aware return (worst → mean) | noop | beats noop |
|--------|----------:|--------------------------------------:|-----:|:----------:|
| phase1 |  4        | 985 → 985                             | 0    | ✓          |
| depth3 | 12        | 4139 → 4334                           | 0    | ✓          |
| depth4 | 16        | 633 → 1007                            | 0    | ✓          |
| depth5 | 20        | 96 → 373                              | 0    | ✓          |

The release bar (the strongest observation-compatible baseline's return) is now
re-defined under this reward; prior absolute numbers (e.g. the B7 ~1267 phase1 bar,
which included +331 of churn reward) are **not comparable** and must be re-measured —
stages 2–5 are re-run on the recalibrated reward. Guardrails: `test_env_api.py`
unit-tests the new `cycle_reward`; `test_scale.py` asserts a fulfilling baseline beats
do-nothing on return at depth 4. Pairs with ADR-0041 (deep-chain obs fix).
