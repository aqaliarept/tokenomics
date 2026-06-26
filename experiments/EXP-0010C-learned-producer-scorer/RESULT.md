# EXP-0010C - Learned Local Producer Scorer

**Question.** Can a learned local scorer provide product urgency, input value,
capacity bottleneck, and investment ROI scores inside the deterministic EXP-0010B
producer planner, while the planner keeps all feasibility constraints?

**Verdict: ACCEPT infrastructure, REJECT controller promotion.** The learned scorer
contract is implemented and tested, but the cheap learned-only smoke does not improve
10B or v4. It is zero-death and feasible, but it regresses held-out consumer units
badly, so it is not a candidate controller.

## What landed

- `plan_producer_command(..., scores=...)` lets an external scorer replace the
  heuristic score step while preserving the existing `ProducingAgentCommand` output.
- `ProducerLocalScores` now includes `capacity_bottleneck_value` in addition to
  product urgency, input value, restoration ROI, upgrade ROI, and inventory expansion
  value.
- `tokenomics.train.producer_scorer` adds:
  - `ProducerScoreModel`: shared row encoders for globals/capacity/product/input rows.
  - Sparse relation pooling over `product_capacity_required` and
    `product_input_required`; no dense all-product interaction is used.
  - `collect_producer_score_samples`, `fit_producer_score_model`, `score_with_model`.
  - `save_producer_score_model` / `load_producer_score_model`.
- `producing_entity_planner_from_scorer` adapts any local scorer closure to the normal
  producer policy interface.

## Local Smoke

Held-out topology seeds `6,7,8,9`, horizon `100`, eval seed `0`, current stochastic
product-stream champion orderer from `tokenomics/m2_inputs/m2_stream_stochastic.pt`.

The local learned scorer was trained only on a tiny default-topology input-stock label
grid (`p4`, 16 samples, 60 epochs, `lr=0.03`). This was a plumbing smoke, not a
promotion-quality training run.

| Seed | v4 units | 10B planner units | learned scorer units | learned dead |
|---:|---:|---:|---:|---|
| 6 | 519 | 512 | 401 | False |
| 7 | 498 | 512 | 268 | False |
| 8 | 598 | 590 | 373 | False |
| 9 | 555 | 556 | 389 | False |
| Mean | 542.5 | 542.5 | 357.75 | 0.0 |

Finding: a learned scorer trained on a narrow local label slice does not generalize
to held-out scale topologies. The deterministic planner's constraints prevent death
or invalid actions, but bad scores still starve the pipeline.

## Tests

- `tests/test_producer_scorer.py`: scorer seam, rollout-label collection, supervised
  fit, deployment adapter, row-order invariance, checkpoint round-trip.
- Adjacent suites: `tests/test_producer_planner.py`, `tests/test_eval.py`,
  `tests/test_producer_view.py`, `tests/test_env_api.py`.

Next useful work is not promotion. Either train scorer labels from scale-rollout or
search data across train topologies, or keep 10B as the producer controller and move
to the EXP-0010D size-holdout audit.
