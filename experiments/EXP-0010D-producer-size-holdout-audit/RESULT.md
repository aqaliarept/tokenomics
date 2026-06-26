# EXP-0010D - Producer Size-Holdout Audit

**Question.** Does the accepted entity-factored producer controller transfer to
larger product/capacity row counts without feasibility, death, row-order,
product-relabel, or capacity-relabel regressions, while reporting runtime and memory
separately from reward quality?

**Verdict: ACCEPT local deterministic size-transfer gates; no learned-scorer
promotion.** The deterministic 10B planner runs at larger row counts without death in
the local smoke. Row-order, product-relabel, and capacity-relabel gates now pass at
every tested size after fixing the scarce-headroom tie path to avoid product-id
selection.

10C did not promote a learned scorer, so this is not a learned small-train/large-run
promotion audit. It is the size audit for the currently accepted producer controller.

## What Landed

- `tokenomics.eval.producer_size_holdout`
  - `audit_entity_planner_size_holdout(...)`
  - `producer_size_gates(...)`
  - `write_size_audit_csv(...)`
- `ProducerSizeAuditRow` separates reward/death columns from runtime/memory columns.
- `build_scale_config` now supports `products_per_layer > 10` by preserving the old
  `stride=10` product IDs for existing widths and using wider per-level strides for
  16/64/400-style stress cases.
- `plan_producer_command` no longer uses product id as the final replenishment
  tie-breaker. For exact equal-value/equal-need input groups, it allocates equal
  integer quantities to every tied input and leaves unallocatable remainder unspent.

## Local Audit

Controller: `PRODUCING_BASELINES["entity_planner"]`  
Orderer: `ORDERING_BASELINES["proportional"]`  
Topology seed: `0`, depth `3`, producers/layer `4`, consumers `3`.

| Products/layer | Horizon | Units | Dead | Runtime sec | Peak KiB | Row order | Product relabel | Capacity relabel |
|---:|---:|---:|---|---:|---:|---|---|---|
| 8 | 8 | 13.0 | False | 0.329 | 211.9 | True | True | True |
| 16 | 8 | 14.0 | False | 0.661 | 325.8 | True | True | True |
| 64 | 8 | 1.0 | False | 1.924 | 1650.4 | True | True | True |
| 400 | 1 | 0.0 | False | 3.863 | 49041.6 | True | True | True |

The 400-width case is a stress/gate smoke only (`horizon=1`), included to verify the
representation can be constructed and stepped locally. Reward quality is not
interpretable at that horizon.

## Finding

The first audit run failed product-relabel at widths 16/64/400. At those widths, the
selected composite producer has more missing input rows than inventory headroom, and
all inputs have equal local value at the audited decision point. The old planner
sorted exact ties by product id, so reversing product labels changed which equally
scored subset received replenishment.

The fix treats exact ties as symmetric: when headroom is too scarce to allocate the
same integer quantity to every member of the tied group, the planner leaves the
remainder unspent instead of choosing a labeled subset. The retest passes the
product-relabel gate at 16/64/400.

## Tests

- `tests/test_producer_size_holdout.py`
- Compatibility checks over product-stream/eval and topology dimension tests.

Next: if joint producer-orderer work is opened later, keep this size gate in the
promotion path and add a slower multi-seed/strict-envelope audit rather than relying
on this local smoke alone.
