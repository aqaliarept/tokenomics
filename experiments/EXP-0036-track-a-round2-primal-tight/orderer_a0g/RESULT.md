# EXP-0036 Stage A0g — Producer-Count-Agnostic Value Table (HANDOVER "Proposed Model")

## Question

The HANDOVER's "Proposed Model" section specifies a deployable orderer head: variable
number of producers in, fixed candidate value table out (shared producer encoder → set
pooling → one scalar per routing candidate). A0f showed no *flat* feature family transfers.
Does the **set inductive bias** — same information, factored as [context ‖ per-producer
row set] with a permutation-invariant encoder — change the held-out regret picture on the
same three tasks?

## Method

- Model implemented as `ProducerSetValueModel` in `tokenomics/train/orderer_scorer.py`:
  shared `_mlp` per producer row → masked mean+max pool → head on [context ‖ pooled] →
  `n_choices` values. Invariants unit-tested: producer-permutation invariance, producer
  count never changes model shape, mask-0 rows exactly absent, fixed candidate output.
- Features: `set_features_from_blocks` reshapes the saved A0f blocks losslessly — producer
  row = per-type capacity (12) ‖ inventory (12) ‖ inbound (12) ‖ relational column (72) =
  108 dims; context = stream ‖ order_queue ‖ compact = 230 dims (+60 with k=4 history).
  `cycle_set_features(env)` provides the identical live path (unit-tested equal).
- Data: the A0f pod traces (`orderer_a0f/artifacts/feature_traces.json.gz`) — per task 200
  train cycles (2 anchored seeds) + 100 held-out cycles (next anchored seed). Same target
  (per-row-centered value-table MSE, `fit_set_value_table`), same iters (1000), same
  init seed, same regret_frac metric. Local run is a saved-artifact recompute
  (HANDOVER-allowed local); script = `probe_set_value.py`, log = `probe_set_value.log`.
- Bonus read the flat probes could not give: because T=3/G=12 match across tasks, the set
  view has identical model shape on all three (only P differs, 6 vs 9) — so one **pooled**
  model was fit on all 600 train cycles (P padded to 9 with mask) and read per held-out.

## Results (held-out regret_frac; constant floors in parentheses)

| variant | s29 | s77 | s45 |
|---|---|---|---|
| best constant | 0.8965 (blend_0.5) | 0.7821 (blend_0.75) | 0.2926 (blend_0.5) |
| best flat A0f | 0.8965 (stream) | 0.7555 (+order_queue) | 0.365 (stream) |
| set | 0.9961 | 0.8495 | 0.7534 |
| set+history | **0.8652** | 0.9922 | 0.6429 |
| set+history 1-seed | 0.9688 | **0.7476** | 0.6008 |
| set+history pooled (600 rows, 3 tasks) | 0.9141 | 1.0815 | **0.5147** |

Train regret_frac ≈ 0.0 for every variant (the set encoder fits the train cycles fully,
like the flat probes at dim ≥ 109).

## Conclusions

**The set inductive bias does not change the A0f verdict on these traces.** The
producer-count-agnostic value table fits train perfectly and lands at or near the constant
mid-blend floors held-out — no variant produces a stable held-out regret reduction across
the three tasks:

- s45 is decisive: every set variant (0.51–0.75) sits far above the constant blend_0.5
  floor (0.2926) and above the flat stream probe (0.365). More structure hurt where less
  already hurt (the A0f smallest-sets-best signature again).
- The two sub-floor readings are marginal and inconsistent: s29 set+history 0.8652 vs
  floor 0.8965 (−0.031), s77 set+history *1-seed* 0.7476 vs floor 0.7821 (−0.035) — the
  same ~0.03 noise-level edge as A0f's s77 +order_queue exception, and each appears in a
  different variant on a different task.
- Learning curves stay wrong-signed/inconsistent: on s77 and s45 the 1-seed fit beats the
  2-seed fit held-out (0.7476 vs 0.9922; 0.6008 vs 0.6429) — more teacher data does not
  help, matching the A0f anti-transfer read.
- Cross-task pooling (600 rows) — the coverage rung only the set model could test — does
  not rescue: s77 goes *above* even_split (1.0815); s45 improves over per-task set fits
  (0.5147) but stays ~1.8× the constant floor.

Scope: this closes the HANDOVER "Proposed Model" as tested — mean+max pooling, centered
value-table MSE, the reconstructible feature set, 200–600 train cycles on these three
tasks. Untested: attention pooling, ranking/winner-set losses on the set encoder, and
per-producer features that do not exist in the saved traces (recent routed/completed/
rejected quantities). Given every observation/representation rung so far lands on the same
constant-floor wall, the evidence keeps pointing at the A0f diagnosis — the per-cycle
lookahead-6 winner on a new demand seed behaves as seed-idiosyncratic noise around a
stable mid-blend preference — not at a still-missing input.

**What the stage leaves behind:** the deployable model + set feature path now exist in the
codebase (`ProducerSetValueModel`, `cycle_set_features`, `fit_set_value_table`,
`set_value_table_metrics`; 6 unit tests). If a future arc finds a signal worth learning
(e.g. A2 deploy-time search traces), this head is the portable way to distill it.

## Artifacts

- `summary.csv` — one row per (task, variant) + pooled rows; constant floors embedded.
- `probe_set_value.py` / `probe_set_value.log` — the recompute script and full output.
- Input traces: `../orderer_a0f/artifacts/feature_traces.json.gz` (not duplicated).
