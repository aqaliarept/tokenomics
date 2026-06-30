# EXP-0017 — C3 out-of-family recipe (fork/join, multi-input)

Status: **DONE — contract PASS, value PASS (joint stack); producer-alone caveat.** The
portable producer (EXP-0013) and orderer (EXP-0012) checkpoints, **unretrained**, load
and run on a genuinely **out-of-family** recipe DAG — a fork/join economy where every
composite product joins two distinct lower products — with **zero contract failures**
across held-out seeds 6–9 and **zero death**. The id-free contract transfers to
multi-input recipes (the dominant open transfer claim). The **full portable stack beats**
the deterministic `entity_planner + proportional` bar by a wide margin (mean **2.86×**,
min 2.64×). Honest caveat: the portable **producer alone** (under a proportional orderer)
**regresses** below the bar on 2/4 seeds — a *policy* gap out-of-family, not a contract
failure, masked when paired with the portable orderer. Reuses the transfer_matrix runner
via new `eval/fixtures.py` + `out_of_family_targets`. Local eval only.

## Question
The portable scorers were distilled on the layered, single-input scale family (each
composite product consumes exactly one lower product at the same within-level index). Do
the **same checkpoints** even *run* on a different recipe DAG — a fork/join, multi-input
economy — and if so, do they hold value, with contract / run / death / value reported
separately so a miss is classifiable? See `spec.json`.

## Setup
- **Fixture** — `eval/fixtures.py::build_forkjoin_config(seed, depth=3, width=4,
  inputs_per_product=2)`: each level-`l` composite product (l≥1) **joins** the two
  lower products at indices `j, j+1 mod width`, so every lower product **forks** to
  several upper products. Max inputs/product 1→2 — out-of-family by construction;
  validates as a feasible economy (layer 0 primal, first producer per level strong).
  Capacity stays one type per level (recipe DAG is the axis here).
- **Runner / stacks** — `transfer_matrix` + `out_of_family_targets`, one fork/join
  target per held-out seed 6–9; the same six stacks (floor, planner bar,
  `stock_aware+proportional`, portable producer + det. orderer, det. producer + portable
  orderer, full portable stack) on the same `joint_stack_env`. 5 episodes/cell, horizon
  100. No retrain.

## Result (ratio vs `entity_planner + proportional` on the SAME fork/join target)

| stack | s6 | s7 | s8 | s9 | mean | min | dead |
|-------|----|----|----|----|------|-----|------|
| noop+proportional (floor) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 |
| entity_planner+proportional (bar) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| stock_aware+proportional | 1.064 | 1.190 | 0.205 | 0.243 | 0.675 | 0.205 | 0 |
| portable_producer + proportional | 1.175 | 1.046 | 0.450 | 0.826 | 0.874 | 0.450 | 0 |
| stock_aware + portable_orderer | 2.200 | 1.798 | 1.934 | 2.024 | 1.989 | 1.798 | 0 |
| **portable + portable** (full stack) | 2.894 | 2.637 | 2.824 | 3.072 | **2.857** | 2.637 | 0 |

All 24 cells: `producer_schema_loads = orderer_schema_loads =
producer_contract_compatible = orderer_contract_compatible = True`, `runs_episode =
True`, `failure_class = none`. Full rows in `summary.csv`.

## Conclusion
**Contract: PASS — the id-free contract transfers to a different recipe DAG.** Every
portable cell runs on every fork/join seed with no shape change (zero contract failures).
Product/producer **count** transfer was EXP-0016; this is the harder claim — a different
recipe *shape* (multi-input join, fork fan-out) — and the per-row producer entity view
and per-(source,product)-stream orderer consume it with no contract break. This is the
dominant open transfer claim and it clears the contract bar.

**Value (joint stack): PASS, large.** The full portable stack beats the deterministic
bar on every seed (mean 2.86×, min 2.64×), zero death. Fork/join is a *harder* economy
(the bar makes only ~70–95 units vs ~466 on scale — multi-input replenishment under
TTL=2 squeezes the deterministic stack); the learned routing handles the join pressure
far better.

**Value (producer-alone): a policy caveat, not a contract failure.** The portable
**producer alone** (proportional orderer) regresses below the bar on s8 (0.45) and s9
(0.83), mean 0.874. The deterministic `stock_aware+proportional` sibling collapses worse
(0.205 / 0.243 on s8/s9). So the *producer side* is fragile out-of-family under a weak
orderer — a genuine policy-quality gap on the recipe axis, surfaced precisely because the
matrix reports producer-alone separately. It is fully masked in the joint stack by the
strong portable orderer (the orderer alone, on a deterministic producer, already lifts
~2×). **Reading:** out-of-family lift is orderer-driven and robust; producer-side
out-of-family policy strength is the open item (a P2 training/fidelity question, not a
contract one).

## Producer-side investigation notes

A follow-up trace on the failed fork/join seeds compared `entity_planner+proportional`
against `portable_producer+proportional` at the producer-command level. The failure is
not loading, action-shape, feasibility, or death: the learned policy emits legal commands
and runs to horizon. The damage is behavioral, concentrated in the first composite layer.

On fork/join s8:
- planner bar: **78.2** mean units; portable producer alone: **35.2** mean units
  (`0.450x`).
- learned-trajectory command diffs: **593 / 6000** producer decisions.
- main diffs: **260** upgrade false positives, **184** inventory-expansion false
  negatives, and **152** replenishment mismatches; most are level-1 producers.
- gate probabilities are confident, not threshold jitter: upgrade false positives had
  mean probability **0.922** against threshold **0.6**; inventory-expansion false
  negatives had probability essentially **0.0** against threshold **0.5**.

Clean ablations isolate the cause:

| variant on s8 | units | ratio vs planner |
|---------------|-------|------------------|
| learned scores + learned gates | 35.2 | 0.450 |
| learned scores + planner gates | 78.6 | 1.005 |
| planner input scores + learned gates | 20.0 | 0.256 |
| planner input scores + planner gates | 78.2 | 1.000 |

So the primary miss is the learned **upgrade / inventory-expansion gates** generalizing
poorly out-of-family. Replacing only those gates with deterministic planner gates
recovers the s8 bar. The s9 miss is milder but follows the same direction: learned
scores + planner gates improves from **61.8** units (`0.826x`) to **71.4** units
(`0.955x`).

There is a secondary replenishment-ranking issue. Fork/join recipes need balanced paired
inputs. The deterministic planner often sees exact equal input values and splits scarce
headroom evenly; the learned `input_value` head emits near-equal but non-identical scores
and can overfill one input while starving its mate. Example from s8 cycle 1:

```
planner input scores: 0=20.0, 1=20.0, 2=16.0, 3=16.0
learned input scores: 0=14.498, 1=14.502, 2=12.921, 3=12.918
planner replenishment: {0: 7, 1: 7}
learned replenishment: {1: 13, 0: 2}
```

Fix path: keep upgrade and inventory-expansion gates planner-owned for P2/C3 deployment
until gate training includes fork/join, shared-capacity, scarcity, and running-investment
states; then add tie-tolerant replenishment allocation or score quantization for
near-equal input values. The verification gate should require producer-alone matrix rows
to stop regressing under `proportional`, so joint-stack success cannot hide the producer
gap.

**Not covered:** shared capacity types across levels, deeper joins
(`inputs_per_product` > 2), uneven-strength competing producers as a dedicated axis, and
the scarcity / degenerate fixtures (step 5 remainder). Added by widening
`eval/fixtures.py`, no new runner code.

## Rerun command
```
python -m tokenomics.eval.transfer_matrix \
  --producer-ckpt experiments/EXP-0013-portable-producer-scorer/artifacts/producer_scorer.pt \
  --orderer-ckpt experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt \
  --depth 3 --seeds 6-9 --out-of-family --inputs-per-product 2 \
  --eval-seeds 5 --horizon 100 --out experiments/EXP-0017-out-of-family-recipe/summary.csv
```
Local eval (evaluation of existing checkpoints — no training).

## Artifacts
- `summary.csv` — the 24-row out-of-family matrix (4 fork/join seeds × 6 stacks).
- (checkpoints reused in place from `EXP-0012-.../artifacts/` and
  `EXP-0013-.../artifacts/`; this experiment adds no new ckpt.)
