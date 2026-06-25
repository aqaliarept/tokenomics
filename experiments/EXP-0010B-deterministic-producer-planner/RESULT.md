# EXP-0010B - Deterministic Producer Planner

**Question.** Can a transparent deterministic controller consume the EXP-0010A
`ProducerEntityView`, enforce local producer constraints, and emit the existing
`ProducingAgentCommand` without changing the producer command or auction?

**Verdict: ACCEPT as the deterministic-controller slice.** The planner lands as a
local, non-learned controller and passes the 10B unit gates. It also ties the v4 producer
on a small local smoke comparison under the current stochastic product-stream champion
orderer. This is not a promotion audit and not a learned-scorer result.

## What landed

- `tokenomics.domain.producer_planner`
  - `ProducerLocalScores`
  - `score_producer_entity_view(view)`
  - `plan_producer_command(view, ...) -> ProducingAgentCommand`
- Transparent local scores:
  - product urgency
  - input value
  - restoration ROI
  - upgrade ROI
  - inventory-expansion value
- Feasibility gates:
  - inventory headroom
  - per-cycle max replenish
  - optional local funding budget
  - already-running restoration/upgrade/inventory-expansion streams
- `PRODUCING_BASELINES["entity_planner"]` adapter:
  - reads `System.producer_entity_view(pid)`
  - plans a domain command
  - maps it back to the existing raw producer action Box
- `evaluate(..., entity_producer_obs=True)` for standard local audits.

## Important fix

The first planner used only immediate downstream demand. It was feasible but failed the
held-out smoke comparison badly because upstream composite producers cannot see future
downstream replenishment orders during the same producer phase. The fix is a small,
transparent input-pipeline priming rule: composite producers top relevant inputs toward
`DEFAULT_INPUT_STOCK_TARGET = 3`. That matches the existing stock-aware discipline while
keeping the planner row-based and constraint-gated.

## Gates

`tokenomics/tests/test_producer_planner.py` - 10 tests:

| Gate | Result |
|---|---|
| Replenishes missing inputs to a local stock target | pass |
| Env baseline adapter emits a valid action | pass |
| Requests inventory expansion when headroom blocks needed inputs | pass |
| Respects headroom, funding, and running-investment gates | pass |
| Exposes transparent local scores | pass |
| Row-order invariant command | pass |
| Product-relabel equivariant replenishment ids | pass |
| Capacity-relabel invariant command | pass |
| Runs on larger synthetic topology | pass |
| `evaluate(entity_producer_obs=True)` works | pass |

Local verification:

```text
tests/test_producer_planner.py tests/test_eval.py: 27 passed
tests/test_producer_view.py tests/test_env_api.py: 48 passed
```

## Smoke Comparison

Held-out topology seeds `6,7,8,9`, episode seed `0`, horizon `100`, product-stream
stochastic champion orderer from `tokenomics/m2_inputs/m2_stream_stochastic.pt`:

| Seed | v4 producer CU | entity_planner CU | v4 dead | planner dead |
|---:|---:|---:|---:|---:|
| 6 | 519 | 512 | false | false |
| 7 | 498 | 512 | false | false |
| 8 | 598 | 590 | false | false |
| 9 | 555 | 556 | false | false |
| Mean | 542.5 | 542.5 | 0.0 | 0.0 |

## Limits

This is not the 40-seed champion audit. It proves the deterministic planner is wired,
constraint-gated, permutation-gated, and locally non-regressing in a cheap smoke check.
EXP-0010C can now add learned local scorers inside this planner; EXP-0010D remains the
real size-holdout audit.
