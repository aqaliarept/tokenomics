# EXP-0031 - Stage 4 Gap Attribution Diagnostic

Stage 4 follow-up after EXP-0030. This is a **producer-ceiling** diagnostic, not a
training run: reuse EXP-0028's archived champion and `entity_planner+mpc_search` bar units,
then evaluate the cheap cross stack:

```text
entity_planner producer + champion orderer
```

This answers the relevant question for v3b producer replay: if the producer teacher
(`entity_planner`) cannot close the archived gap while the orderer is fixed at champion,
then more producer replay sampling cannot close the mined objective gap.

## Setup

- Archive: EXP-0028 topology-grid mining archive, all 26 archived rows.
- Baselines: champion/full-bar units reused from EXP-0028 `curriculum_matrix.csv`.
- Cross stack: `entity_planner + champion_orderer`.
- Eval: 3 episode seeds per task, horizon 100.
- Expensive symmetric cross skipped: `champion_producer + mpc_search_orderer`.

## Result

Producer-side ceiling is too small to justify v3b as the next promotion path.

```json
{
  "rows": 26,
  "positive_gap_rows": 26,
  "sum_positive_gap_units": 95.34,
  "sum_positive_producer_only_gain": 5.02,
  "producer_closure_ratio": 0.0527,
  "dominant_channel_counts": {
    "not_producer": 21,
    "producer_partial": 4,
    "producer": 1
  }
}
```

By split:

| split | rows | gap | positive producer gain | closure |
| --- | ---: | ---: | ---: | ---: |
| train | 9 | 32.67 | 0.00 | 0.0% |
| val | 6 | 24.67 | 3.01 | 12.2% |
| blind | 6 | 21.67 | 0.34 | 1.6% |
| train leakage dropped | 5 | 16.33 | 1.67 | 10.2% |

The EXP-0030 blind-tail failure row is especially decisive:

```text
mine-build_scale_config-depth3-producers_per_layer4-products_per_layer3-s8
gap=+4.00, producer-only gain=-4.33, remaining gap with orderer fixed=8.33
```

## Conclusion

Stop polishing producer replay as the promotion path. The mined gap is not materially
recoverable through the Stage 4 producer teacher channel. v3b sampling fixes may still be
useful hygiene, but they should not be the next expensive run.

Next promotion path: v4 orderer oracle relabel. Teach the orderer from the mining bar's
MPC/search allocations and broaden the orderer anchor set before running another Stage 4
promotion candidate.
