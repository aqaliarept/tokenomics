# EXP-0016 — C2 size transfer (ordered plan step 4)

Status: **DONE (PASS)** — the portable producer (EXP-0013) and orderer (EXP-0012)
checkpoints, **unretrained**, load and run on topologies **larger than they were
distilled on** (24 products, 24 producers, both) with **zero contract failures**, zero
death, and no regression vs the deterministic `entity_planner + proportional` bar. The
id-free contract scales with product and producer count. Reuses the EXP-0015
`transfer_matrix` runner by widening `targets` only — no new policy code, no retrain.
Local eval only. Gated by
`tests/test_transfer_matrix.py::test_size_transfer_targets_widen_product_and_producer_axes`.

## Question
The portable scorers were distilled at depth 3 / width 4 (P=12 products, P=12
producers). Do the **same checkpoints** stay contract-compatible and keep their value
lift when the topology is **wider** than training — more products, more producers, or
both — with contract / run / death / value reported separately from the scaling metric
(so a miss is classifiable)? See `spec.json`.

## Setup
- **Runner** — `transfer_matrix` + new `size_transfer_targets(seed, depth,
  product_width, producer_width)`: off base seed 6 it builds three C2 directions —
  `prodW8` (products_per_layer 4→8 ⇒ 24 products, 12 producers), `agentW8`
  (producers_per_layer 4→8 ⇒ 12 products, 24 producers), and `prodW8-agentW8` (24/24).
  Depth (= capacity-type count) is **held fixed at 3** so a contract/scaling failure on
  a wider target stays separable from the train-size rows.
- **Stacks** — floor (`noop+proportional`), practical bar
  (`entity_planner+proportional`), `stock_aware+proportional`, portable producer + det.
  orderer (C0 prod), det. producer + portable orderer (C0 ord), full portable stack
  (C1). No retrain — EXP-0012 + EXP-0013 ckpts reused in place.
- Base `scale-d3-s6` and `scale-d3-s6-relabel` rows are carried as the in-family
  reference. 5 episodes/cell, horizon 100.

## Result (ratio vs `entity_planner + proportional` on the SAME target)

| target | products | producers | portable_producer +prop | stock_aware +portable_ord | **portable + portable** | dead |
|--------|----------|-----------|-------------------------|---------------------------|-------------------------|------|
| scale-d3-s6 (base) | 12 | 12 | 1.000 | 1.075 | **1.073** | 0 |
| scale-d3-s6-prodW8  | **24** | 12 | 1.085 | 1.256 | **1.286** | 0 |
| scale-d3-s6-agentW8 | 12 | **24** | 1.013 | 1.016 | **1.063** | 0 |
| scale-d3-s6-prodW8-agentW8 | **24** | **24** | 1.008 | 1.123 | **1.147** | 0 |

All non-floor cells: `producer_schema_loads = orderer_schema_loads =
producer_contract_compatible = orderer_contract_compatible = True`, `runs_episode =
True`, `failure_class = none`. Floor `noop+proportional` = 0 units / dead 1.0 on every
target. Full rows in `summary.csv`.

## Conclusion
**PASS.** Size transfer holds end-to-end with no contract code touched:

1. **No contract failure from widening.** Every portable cell on every size direction
   `runs_episode=True`, `failure_class=none`. The id-free scorers (per-row producer
   entity view, per-(source,product)-stream orderer) consume the wider topology with no
   shape change — confirming product/producer count is genuinely outside the tensor
   contract, not just untested.
2. **Targets are nontrivial.** `noop` = 0 units / full death on all three wider targets,
   so the positive ratios are real economic work.
3. **No regression vs the planner bar.** The full portable stack ties-or-beats
   `entity_planner + proportional` on every direction (min 1.063), zero death.
4. **Wider products → bigger orderer lift; wider producers → smaller.** Product width
   8 lifts the full stack to **1.286** (vs 1.073 base) — more products = more routing
   slack the portable orderer exploits (consistent with the O3 zero-shot product-count
   finding). Producer width 8 alone shrinks the lift to **1.063** — more producers give
   the deterministic bar its own routing slack, so the orderer's edge narrows. The
   both-wide target lands between (**1.147**).
5. **The lift stays orderer-driven at scale.** Portable producer + proportional tracks
   the planner (1.008–1.085); the gain rides on the portable orderer in every column —
   same decomposition as C1/EXP-0014, now confirmed under size transfer.

**Not covered:** out-of-family recipes (different DAG shape / capacity-type set — P2 /
O3 remainder / C3) and scarcity/degenerate fixtures. Those are separate axes (steps
5–6); this rung widens counts within the layered scale family only.

## Rerun command
```
python -m tokenomics.eval.transfer_matrix \
  --producer-ckpt experiments/EXP-0013-portable-producer-scorer/artifacts/producer_scorer.pt \
  --orderer-ckpt experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt \
  --depth 3 --seeds 6 --size-transfer --product-width 8 --producer-width 8 \
  --eval-seeds 5 --horizon 100 --out experiments/EXP-0016-size-transfer/summary.csv
```
Local eval (evaluation of existing checkpoints — no training).

## Artifacts
- `summary.csv` — the 30-row size-transfer matrix (5 targets × 6 stacks, canonical
  "Evaluation Output" columns).
- (checkpoints reused in place from `EXP-0012-.../artifacts/` and
  `EXP-0013-.../artifacts/`; this experiment adds no new ckpt.)
