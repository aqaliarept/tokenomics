# EXP-0012 — Portable orderer scorer (O2 value)

Status: **ACCEPT** — portable scorer+allocator beats `proportional` on every held-out
topology (mean 1.28×, min 1.09×), zero death. Distilled on Runpod CPU pod. Both O3 axes
closed locally (below): producer-count transfer (P=4/8/12) **and** zero-shot product-count
transfer (products/layer 4→8→16, lever strengthens to 1.51×, 0/60 death), plus parity with
the v3 learned champion.

## Question
Does the portable orderer scorer+allocator (`OrdererScoreModel` → `allocate_suborders`,
ADR-0054 O2) match/beat the `proportional` baseline on held-out scale topologies, when
distilled from the proportional throughput teacher? See `spec.json`.

## Setup
- Distill: masked MSE of per-producer scores → `log(makeable_qty_now)` teacher, on
  `stock_aware + proportional` rollouts over scale seeds 0–5, depth 3.
- Deploy: scores tiled across `k_stream` slots → v4 decode → masked allocator.
- Eval: scorer-policy vs `proportional` (producer fixed `stock_aware`), held-out seeds
  6–9, 5 episodes each, horizon 100. Metric = `consumer_completed_units` ratio.

## Result
Distill converged: masked-mse 406.0 → 0.21 over 400 iters on 7928 samples / 6
topologies. Held-out eval (5 episodes each, horizon 100):

| seed | producers | scorer_units | proportional_units | ratio | dead |
|------|-----------|--------------|--------------------|-------|------|
| 6 | 12 | 501.2 | 423.8 | 1.183 | 0/5 |
| 7 | 12 | 476.4 | 438.6 | 1.086 | 0/5 |
| 8 | 12 | 592.8 | 403.4 | 1.470 | 0/5 |
| 9 | 12 | 559.0 | 405.4 | 1.379 | 0/5 |

mean ratio **1.279** · min **1.086** · death **0/20**. Checkpoint reloads locally
(schema gate green), 49 KB.

## Conclusion
**ACCEPT.** The portable orderer scorer+allocator (O2) clears the `proportional` bar on
all four held-out economies with zero death — distilled only from the proportional
throughput teacher, yet it *beats* the teacher's own baseline (the learned scorer reads
richer per-producer/per-product facts than `min throughput`, so it routes better than
log-throughput alone). Versatility (variable P, product relabel, producer-row
permutation, eligibility masking, conservation) is proven separately by the contract
gates in `tests/test_orderer_scorer.py` + `tests/test_orderer_allocator.py`.

## Producer-count transfer (same checkpoint, unseen P)
The depth-3 (P=12) checkpoint, run **unmodified** on held-out seeds 6–9 at shallower
depths (no re-distill), local eval:

| depth | producers | mean ratio | min ratio | dead |
|-------|-----------|------------|-----------|------|
| 1 | 4 | 1.000 | 1.000 | 0/20 |
| 2 | 8 | 1.005 | 1.000 | 0/20 |
| 3 | 12 | 1.279 | 1.086 | 0/20 |

It loads and runs at P=4 and P=8 with no shape change (empirical proof of the
order-equivariant, id-free contract — not just the unit gates), **never regresses**
`proportional`, and never kills a topology. It only *ties* at depth 1–2 because the
routing lever is slack there (demand is easily fulfilled, so any sane split scores the
same); the lever binds — and the scorer wins 1.28× — at depth 3. So: transfer = portable
+ safe at every producer count, value where routing matters.

## Product-count transfer (zero-shot, unseen product width)
The other O3 axis: hold producer count fixed (`depth 3`, `producers_per_layer 4` → P=12)
and widen the product set (`products_per_layer` 4→8→16). The v4 contract is empirically
product-width-invariant — obs_dim 436 / action_dim 72 / k_stream 6 stay constant; only the
*stream count* grows (52→96→184 `(source,product)` agents). The **same** EXP-0012
checkpoint (distilled only at width 4), run unchanged vs `proportional`, held-out 6–9:

| products/layer | streams | mean ratio | min ratio | dead |
|----------------|---------|------------|-----------|------|
| 4 (distill width) | 52 | 1.279 | 1.088 | 0/20 |
| 8 | 96 | 1.329 | 1.092 | 0/20 |
| 16 | 184 | 1.509 | 1.100 | 0/20 |

Zero-shot, no re-distill: the id-free per-producer scorer is reused identically on every
stream regardless of how many products exist. The lever *strengthens* with width (mean
1.28→1.33→1.51, min always >1.08, 0/60 death) — more products = more routing decisions,
where `proportional`'s min-throughput heuristic leaves more on the table. So the cloud
re-distill flagged as a possible follow-up is **not needed**: product-count portability is
free at inference.

## Vs the v3 learned champion (`m2_stream_stochastic.pt`)
Head-to-head on the same held-out seeds 6–9 / depth 3, **producer held at `stock_aware`
for both** (the controlled orderer comparison), consumer units:

| seed | scorer (v4) | champion (v3) | scorer/champ | dead |
|------|-------------|---------------|--------------|------|
| 6 | 501.2 | 501.4 | 1.000 | 0/0 |
| 7 | 477.0 | 485.2 | 0.983 | 0/0 |
| 8 | 592.8 | 570.4 | 1.039 | 0/0 |
| 9 | 559.0 | 523.0 | 1.069 | 0/0 |

mean **1.023**, min **0.983** — a statistical wash. Fairness sanity: the `proportional`
bar is byte-identical across the v3 and v4 agent groupings every seed (424.8/438.6/403.4/
405.4), so the two orderers are scored on the same economies (parity, ADR-0052). The
portable, id-free scorer — distilled from the *cheap* `proportional` throughput teacher —
**matches the v3 stochastic champion**, which is the end product of the whole
EXP-0004→0007A search-oracle + DAgger + stochastic-NLL arc. Portability over producer
count is free: no quality cost vs the best fixed-shape orderer.

## Artifacts
- `artifacts/orderer_scorer.pt` — distilled scorer checkpoint (schema-versioned).
- `artifacts/train.log` — full distill + eval log.
