# EXP-0021 — Producer broaden training (ordered-plan steps 11 + 12)

Status: **DONE — primary target CLOSED, one residual cell.** Broadened Runpod CPU distill
complete (depth-3, seeds 0–5, 300 epochs, 54 topologies). The fork/join **s9** producer-alone
regression that steps 6–8 could not close deterministically is **closed: 0.955 → 1.000**
(EXP-0018 → this run). Step-12 acceptance run locally with the pulled checkpoint. **Not a
strict clean sweep:** one held-out producer-alone cell (uneven-strength s6 = 0.975) and a
tiny in-family scale dip (s9d3 = 0.9921) remain sub-1.0. Zero contract failures, zero death
(outside the deepjoin-d5 feasibility wall). "Broadening should generalize" is now supported
by numbers — with the two residuals named, not hidden.

## Question

Does broadening the portable producer scorer's distillation set past the scale family —
to the five families step 11 names (fork/join, shared-capacity, uneven-strength, scarcity,
running-investment) at train seeds 0–5 — close the out-of-family producer-alone regression
(EXP-0017/0018 fork/join **s9 = 0.955** under a deterministic `proportional` orderer)
WITHOUT regressing the in-family scale value EXP-0013 proved, at zero contract failure /
zero death? See `spec.json`.

## Setup

- **Trainer**: `tokenomics.train.producer_scorer --broaden`. New
  `broadened_producer_train_configs(seeds, depth, width)` materializes, per seed, the scale
  base + `forkjoin` + `sharedcap` + `uneven` + the five `scarcity_suite` axes = 9
  topologies/seed × 6 seeds (0–5) = **54 economies**. `distill_producer_scorer` gained a
  `train_configs` param so it distills over that explicit list instead of scale-only;
  scale-only stays the default. Same `ProducerScoreModel` / schema / checkpoint v2 as
  EXP-0013 — only training coverage changes, so the checkpoint stays load-compatible with
  every fixture/size target.
- **Teacher / labels**: unchanged from EXP-0013 — `score_producer_entity_view` (MSE) for
  the continuous heads, the planner's gate booleans (BCE) for upgrade / inventory-expansion;
  restoration planner-owned. `entity_planner + proportional` rollouts.
- **In-family sanity (this run's eval)**: held-out scale seeds 6–9 at eval-depths 1/2/3
  (P=4/8/12, unseen producer counts) — the no-regression readout vs EXP-0013.
- **Out-of-family acceptance (step 12, separate)**: `transfer_matrix --fixtures` over
  fork/join seeds 6–9 + shared-cap + uneven + scarcity, producer-only under `proportional`
  + joint stack, both producer modes (planner-gated = deployment, learned-gate = bar).

## Reproduce

```
python run_pod_training.py --pod-id <id> --cpu --python python3.12 \
  --train "-u -m tokenomics.train.producer_scorer --broaden --depth 3 --train-seeds 0-5 \
           --width 4 --eval-seeds-list 6-9 --epochs 300 --lr 0.01 --eval-seeds 5 \
           --horizon 100 --eval-depths 1-3 --save /workspace/producer_scorer_broad.pt" \
  --pull /workspace/producer_scorer_broad.pt:./experiments/EXP-0021-producer-broaden-training/artifacts/producer_scorer_broad.pt
```

## Step 11 result (did NOT accept "it trained" as success)

- [x] **pulled checkpoint** `artifacts/producer_scorer_broad.pt` — reloads through the v2
      version gate (hidden 64, calibrated gates upgrade 0.50 / inventory 0.50).
- [x] **in-family scale seeds 6–9, depths 1/2/3** (P=4/8/12) — **mean 1.0011, min 0.9921**.
- [x] **zero death** across all in-family rows; **invest-FP ≈ 0** (max upgrade-FP 0.0033,
      inventory-FP 0, replenishment-mismatch ≤ 0.0008).
- [~] **no regression vs EXP-0013** (bar mean 1.0029 / min 1.0000) — **small regression**:
      mean −0.0018, and one cell (scale s9d3) dips 1.0000 → **0.9921**. Broadening bought
      out-of-family robustness at a ~0.8% in-family cost on that single cell. Reported, not
      hidden.

| stage | metric | value | bar |
|-------|--------|-------|-----|
| in-family scale 6–9 (d1/2/3) | mean / min ratio vs planner | **1.0011 / 0.9921** | ≥ EXP-0013 (1.0029 / 1.0000) — *slightly under* |
| in-family | max death / max invest-FP | **0 / 0.0033** | 0 death; FP ≈ 0 — met |

## Step 12 result — out-of-family acceptance (the real fix)

`transfer_matrix` with the pulled checkpoint, producer-alone under `proportional`
(`portable_producer_planner_gates+proportional` = the EXP-0018 deployment mode). Full CSVs
in `artifacts/step12_forkjoin.csv` (8-stack fork/join, orderer ckpt EXP-0012) and
`artifacts/step12_fixtures.csv` (producer audit over the broad C3/P3 families).

| fixture (producer-alone, planner-gated) | s6 | s7 | s8 | s9 | verdict |
|---|---|---|---|---|---|
| **fork/join** | 1.000 | 1.034 | 1.033 | **1.000** | **PASS** (EXP-0018 s9 was 0.955) |
| **shared-cap** | 1.000 | 1.000 | 1.019 | 1.000 | PASS |
| **uneven-strength** | **0.975** | 1.000 | 1.000 | 1.005 | **s6 residual < 1.0** |
| scarcity (s6: fund/inv/backlog/degraded/running) | 1.000 / 1.026 / 1.000 / 1.000 / 1.000 | | | | PASS |
| degenerate (s6, 5 edge economies) | all 1.000 | | | | PASS (floors) |
| deepjoin-d5 | bar = 0 units (entity_planner also 0 / dies) | | | | feasibility wall, excluded |

- **Contract**: 0 schema/load + 0 contract failures across **both** matrices, every stack.
- **Death**: 0 for all non-floor, non-deepjoin policy rows. Deepjoin-d5 is a depth 5 >
  TTL 2 wall where the `entity_planner` bar also completes 0 units and sometimes dies — a
  scaling/feasibility result, not a producer policy failure. The `noop` floor rows die by
  construction and are not policy-acceptance rows.
- **Joint stack** (planner-gated producer + portable orderer, fork/join 6–9): **2.71–3.02×**,
  all pass, 0 death — unchanged-strong.
- **Learned-gate** producer-alone still regresses out-of-family (fork/join s8 0.875, s9
  0.920) → planner-gated deploy stays the correct P2/C3 mode; broadening lifted the
  *continuous* heads, which is exactly where the s9 gap lived.

## Conclusions

1. **Hypothesis supported.** Broadening the distillation set to out-of-family + scarcity
   states closed the fork/join producer-alone s9 gap (0.955 → 1.000) that no deterministic
   knob could (EXP-0017 tie tolerance is non-monotonic). The lift is in the learned
   *continuous* replenishment heads — planner-gated deploy is untouched.
2. **Strict "no producer-alone seed below 1.0 on the covered set" is NOT met.** One
   held-out cell remains: **uneven-strength s6 = 0.975** (its s7/s8/s9 pass). Plus a tiny
   in-family scale s9d3 = 0.9921. Both named, neither laundered through the (strong) joint.
3. **Next (fallback ladder, plan step 11):** the two residuals are *close-but-under* → the
   primary next move is **DAgger** (collect on the learned scorer's own rollouts — on-policy
   distribution shift the teacher-only labels miss), before any architecture change. Or, per
   step 15, **narrow the final claim** to the fork/join + shared-cap + scarcity family and
   record uneven-strength s6 as the open cell.

## Reproduce (step 12)

```
# fork/join 8-stack (needs the EXP-0012 orderer ckpt for the joint/portable-orderer rows)
python -m tokenomics.eval.transfer_matrix --out-of-family --seeds 6-9 --depth 3 \
  --producer-ckpt experiments/EXP-0021-producer-broaden-training/artifacts/producer_scorer_broad.pt \
  --orderer-ckpt experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt \
  --eval-seeds 5 --horizon 100 --out step12_forkjoin.csv

# broad C3/P3 producer audit (orderer fixed proportional)
python -m tokenomics.eval.transfer_matrix --audit producer --fixtures --seeds 6-9 --depth 3 \
  --producer-ckpt experiments/EXP-0021-producer-broaden-training/artifacts/producer_scorer_broad.pt \
  --eval-seeds 5 --horizon 100 --out step12_fixtures.csv
```
