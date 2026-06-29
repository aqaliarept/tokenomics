# EXP-0013 — Portable producer scorer (P1 value)

Status: **DONE (PASS)** — v2 gate-head Runpod CPU distill complete (depth-3, seeds 0–5,
300 epochs). Mean ratio **1.0029** vs the planner, **min 1.0000**, zero death, investment
false-positive rates ≈0 — clean across all three depths incl. the unseen producer counts.
Calibrated gate thresholds: upgrade **0.60**, inventory **0.50**. Driver + gate contract
unit-tested (`tests/test_producer_scorer.py`, 17 passed; full suite 429 passed).

## Question
Does the portable producer scorer+planner (P1: `ProducerScoreModel → plan_producer_command`)
reproduce the deterministic `entity_planner`'s value on held-out scale topologies when
distilled from the planner's own labels, and load/run unchanged at unseen producer counts?
See `spec.json`.

## Setup
- Distill: replenishment scores by MSE on `score_producer_entity_view` labels; investment
  decisions (upgrade, inventory-expansion) as learned **binary gates** — logits BCE-trained
  against the planner's own gate booleans — on `entity_planner + proportional` rollouts over
  scale seeds 0–5, depth 3. Per-gate probability thresholds calibrated post-fit on training
  command agreement, saved in the checkpoint. Restoration stays planner-owned (no learned
  gate). Checkpoint v2.
- Deploy: `scorer_producing_policy` → `producing_entity_planner_from_scorer` →
  `plan_producer_command` (planner keeps all feasibility).
- Eval: scorer+planner vs `entity_planner` (orderer fixed `proportional`), held-out seeds
  6–9, 5 episodes each, horizon 100, at **eval depths 1/2/3** (P=4/8/12 — smaller depths =
  unseen producer counts). Reports **value** (units ratio, death) from those five episodes
  and **command fidelity** (upgrade/inventory false-positive rates, replenishment mismatch)
  on the seed-0 teacher trajectory for each topology.

## Result (v2, gate heads)

Per held-out seed (scorer+planner vs `entity_planner`, orderer fixed `proportional`,
5 episodes, horizon 100). The depth-3-distilled (P=12) checkpoint evaluated unchanged at
depths 1/2 = unseen producer counts P=4/8. Full rows in `summary.csv`. Command-fidelity
rates are the seed-0 teacher-trajectory diagnostic; a stricter post-check over all five
eval seeds still had low FP rates (max upgrade FP 0.0137, inventory FP 0.000) and did not
change the value/death verdict.

| depth | producers | seed | ratio | dead | planner_dead | upgrade_fp | inventory_fp | replen_mm |
|-------|-----------|------|-------|------|--------------|------------|--------------|-----------|
| 1 | 4  | 6 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 1 | 4  | 7 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 1 | 4  | 8 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 1 | 4  | 9 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 2 | 8  | 6 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 2 | 8  | 7 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 2 | 8  | 8 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 2 | 8  | 9 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 3 | 12 | 6 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 3 | 12 | 7 | 1.000 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 3 | 12 | 8 | 1.022 | 0 | 0 | 0.000 | 0.000 | 0.001 |
| 3 | 12 | 9 | 1.012 | 0 | 0 | 0.003 | 0.000 | 0.000 |

Per-depth mean ratio: **depth 1 = 1.000 · depth 2 = 1.000 · depth 3 = 1.0086.**
Overall mean ratio **1.0029** · min **1.0000** · max scorer dead-rate **0.000** ·
max upgrade-FP **0.003** · max inventory-FP **0.000**. Calibrated thresholds:
upgrade **0.60**, inventory **0.50**.

## Prior result (v1, MSE ROI-regression heads) — the failure this run fixes

| depth | P | mean ratio (seeds 6–9) | death |
|-------|----|------------------------|-------|
| 1 | 4  | 1.000 | 0 |
| 2 | 8  | 0.998 | 0 |
| 3 | 12 | 0.773 | 0 |

Overall mean 0.9233 · min 0.7092 · zero death. Transfer to unseen P=4/8 was clean, but
depth-3/P=12 lost ~23%. **Diagnosis** (learned vs planner commands on held-out trajectories):
replenishment was fine (5/24000 mismatch) — the loss was structural in the investment gates.
`score_with_model` clamped tiny-positive ROI noise and `plan_producer_command` reads
`upgrade_roi > 0` / `inventory_expansion_value > 0` as hard "spend" toggles, so near-zero
regression noise became action: **upgrade false positives 11442/24000, inventory-expansion
false positives 6875/24000**. Units were low because false investment spend was huge while
the units ratio alone still looked plausible — which is exactly why v2 reports command-level
FP rates as a gate.

## Conclusion
**PASS.** All gate conditions hold:

1. **Value** — mean ratio 1.0029 vs the planner, min 1.0000; depth-3/P=12 (the v1 failure
   point) recovered from 0.773 to 1.0086, with two held-out seeds *exceeding* the planner
   (1.022, 1.012) by spending less on needless investment. Zero death everywhere.
2. **Command fidelity** — investment false-positive rates collapsed: upgrade FP from v1's
   11442/24000 (≈0.48) to ≤0.003, inventory-expansion FP from 6875/24000 (≈0.29) to 0.000.
   Replenishment was never the problem and stays ≈0 mismatch.
3. **Producer-count transfer** — the one depth-3 (P=12) checkpoint runs unchanged at the
   unseen P=4/8 (all ratio 1.000), no shape change. The id-free contract carries.
4. **Contract** — v2 checkpoint reloads through the version gate carrying its calibrated
   per-gate thresholds (upgrade 0.60, inventory 0.50).

**Why the gate heads fix it:** treating upgrade / inventory-expansion as learned *binary
gates* (logits BCE-trained against the planner's own gate booleans, fired by a calibrated
probability threshold) removes the v1 failure mode where MSE ROI-regression noise was read
by `plan_producer_command` as a hard "spend" toggle. The bug was gate calibration, not
scorer capacity — no depth-staged distillation or extra samples were needed. The P1
portability claim (id-free producer scorer faithful to the planner across producer count)
is **proven**.

## Rerun command
```
python run_pod_training.py --pod-id <id> --cpu --python python3.12 \
  --train "-u -m tokenomics.train.producer_scorer --depth 3 --train-seeds 0-5 \
           --eval-seeds-list 6-9 --epochs 300 --lr 0.01 --eval-seeds 5 --horizon 100 \
           --eval-depths 1-3 --save /workspace/producer_scorer.pt" \
  --pull /workspace/producer_scorer.pt:./experiments/EXP-0013-portable-producer-scorer/artifacts/producer_scorer.pt
```
Gate thresholds auto-calibrate in-flow; no manual threshold flags needed.

## Artifacts
- `artifacts/producer_scorer.pt` — distilled v2 scorer checkpoint (gate heads + calibrated
  per-gate thresholds, schema-versioned; reload-verified).
- `summary.csv` — per-(depth,seed) value + command-fidelity rows.
