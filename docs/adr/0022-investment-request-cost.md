# ADR-0022: Charge the training policy for opening investment requests

## Status
Experimental; rejected as the default.

## Context

After action masking, regressed producing policies repeatedly selected
`both` whenever neither investment was running. A representative episode
created roughly:

- 500 completed investment orders;
- 150 rejected investment orders;
- fewer than 300 completed consumer orders.

The environment gives `+1` for every completed investment order. Although small
relative to a completed consumer order (`+10`), this creates a direct reward for
investment churn. Restoration and upgrade are instrumental actions; opening
orders should be worthwhile only if they improve future consumer fulfillment.

## Decision

Add a training-only cost for each newly requested investment bit:

```text
request_cost = coefficient * (restore_requested + upgrade_requested)
```

The flag remains available, but the default is `0.0`.

At coefficient 1 this exactly cancels the direct `+1` reward if an investment order completes.
Its future capacity benefit and avoided consumer rejection remain in the return,
so useful restoration/upgrade can still have positive advantage. Evaluation and
the baseline-comparable environment return are unchanged.

The cost is attached to the producing transition at action time and the delayed
AEC score is added later. It is not charged when a running investment masks the
request, because ADR-0021 prevents that invalid action from being sampled.

## Consequences

- Removes the direct incentive to manufacture investment completions.
- Both actions cost twice a single restoration or upgrade.
- This is reward shaping and must be judged by cross-seed evaluation, not by
  training return.

The 0.5/1.0/2.0 sweep did not prevent seed-5 collapse, so it is not part of the
recommended configuration.

## Test

`test_investment_request_cost_counts_action_bits` verifies the two-bit cost.
