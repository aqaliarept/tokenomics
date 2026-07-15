# EXP-0036 Handover: Orderer Training Stuck

Date: 2026-07-14

## Plain-English State

The orderer is not stuck because it needs more epochs. It fits the training seeds almost
perfectly, then fails on held-out demand traces. The failure is the training target and
state coverage:

- The search teacher makes one global per-cycle routing choice after short rollouts.
- The learned orderer sees local product-stream features.
- Many search targets are ties or near-ties, but the current target records one hard argmax.
- Mistakes are not harmless: held-out regret shows the probe loses most of the teacher's
  local edge.

Update after A0f/A0g: the trained-agent observation path has been tested and did not
produce transferable held-out behavior. Missing-observation probes and the proposed
producer-count-agnostic set value-table head both fit train and fail held-out. The next
trained-agent work should focus on changing the teacher/target distribution or collecting
a different data distribution, not adding more of the same observation blocks.

## Evidence To Trust

Use raw artifacts, not prose-only conclusions:

- `mine/summary.csv`: 400 mined tasks, 96 raw `policy_weakness` rows.
- `gap_attribution/gap_attribution.csv`: 81 archived round-2 rows used in the attribution
  claims.
- `orderer_a0c/summary.csv`: blend-vocabulary value and held-out choice/general-state
  probe.
- `orderer_a0t/summary.csv` plus `orderer_a0t/artifacts/traces.json.gz`: regret-weighted
  held-out decision failures.
- `orderer_a0f/summary.csv` plus `orderer_a0f/artifacts/feature_traces.json.gz`:
  missing-observation feature-block value-table probe.
- `orderer_a0g/summary.csv`: producer-count-agnostic set value-table probe built from
  the saved A0f traces.
- `orderer_probe_retry/orderer_probe_retry/debug/*/samples.pkl.gz`: saved oracle-relabel
  samples for recomputing flat-target fraction and variance explained.

Recomputed key numbers:

```text
Round-2 archive gaps: n=81, min=2.67, median=53.0, mean=99.86, max=973.33, sum=8088.31
Attribution channels: 45 orderer, 15 mixed, 11 producer, 10 interaction/noise
Orderer-only gain sum over all round-2 rows: 7171.64
Producer-only gain sum over all round-2 rows: 1122.94
Champion death rows: 6 partial/full death, 5 full death
```

```text
A0c means:
named_capture_frac = 0.7839
blend_capture_frac = 0.9008
choice_top1 = 0.3400 vs choice_majority = 0.5192
gchoice_top1 = 0.3333 vs gchoice_majority = 0.4967
```

```text
A0t held-out regret:
s29 probe_regret_frac = 0.9824
s77 probe_regret_frac = 0.8621
s45 probe_regret_frac = 0.5607
```

Saved-sample recompute from `samples.pkl.gz` + `orderer_oracle.pt`:

```text
s29 flat targets 42.4%, variance explained 0.254, top1 0.598 vs chance 0.500
s77 flat targets 50.2%, variance explained 0.098, top1 0.502 vs chance 0.333
s45 flat targets 29.1%, variance explained 0.641, top1 0.554 vs chance 0.333
```

The target-majority floor is higher than the trained top-1 on these train-state samples,
so do not read top-1 as "good enough".

A0f result: tested feature enrichment failed to find a transferable missing observation.
Best held-out `regret_frac` by task:

```text
s29 best A0f = 0.8965 (stream)
s77 best A0f = 0.7555 (+order_queue)
s45 best A0f = 0.3650 (stream)
```

A0g result: the producer-count-agnostic set value-table model was implemented and tested.
It did not beat the constant mid-blend floors in a stable way:

```text
s29 best constant blend_0.5 = 0.8965; best A0g set+history = 0.8652
s77 best constant blend_0.75 = 0.7821; best A0g 1-seed = 0.7476
s45 best constant blend_0.5 = 0.2926; best A0g pooled = 0.5147
```

Read the tiny sub-floor A0g edges on s29/s77 as weak and inconsistent: they are about
0.03 regret_frac, appear in different variants on different tasks, and do not survive
the decisive s45 constant-floor check.

## Known Pitfalls

1. Seed anchoring is non-negotiable. Archived tasks must evaluate at
   `_episode_seeds(record.episode_seed, eval_seeds)`, not `range(eval_seeds)`.
2. `env.global_state()` is not the full state. It loses per-type capacity detail,
   queue/TTL state, inventory composition, and history.
3. Accuracy is the wrong primary metric. Use regret/units lost, because ties and wrong
   winner-set handling make top-1 misleading.
4. Do not mix the 96 raw mined weaknesses with the 81 archived round-2 attribution rows.
   The headline 8088u attribution sum is the 81-row archive population.
5. More epochs/width on the same observation and same hard argmax or centered-vector target is low priority.
   The logs already show train CE near zero while held-out regret remains large.
6. A0g pooled rows are useful but have a normalization caveat: padded producer rows are
   masked in forward/training, but the current `set_norm` call includes padded zero rows
   in producer normalization statistics. This mainly affects the pooled cross-task read,
   not the per-task A0g rows.

## Task Queue

### Task 1: Missing-Observation Probe

Status: **DONE by A0f**. No tested feature block produced a stable held-out regret
reduction.

Goal: identify whether the orderer lacks state, lacks trajectory coverage, or only has
bad labels.

Build probes that predict the teacher value table or winner set, not a single argmax.
Add feature blocks one at a time and measure held-out regret:

```text
local product-stream obs
+ per-type producer capacity/free/reserved/requested
+ order queue, TTL, age, backlog composition
+ inventory by product
+ inbound by product
+ 3-5 cycle history/aggregates
+ relational product x eligible-producer x capacity-type features
+ champion/learner trajectory states
```

Definition of done:

- Produce a table with `regret_frac` per feature block.
- If one block sharply reduces regret, that is the missing data family.
- If no flat block reduces regret, move to relational features before changing the label
  objective. The trained-agent goal stays in scope.

### Task 2: Complete Relational Observation Probe

Status: **DONE for the flat-relational slice by A0f and for the set-representation slice
by A0g**. Relational features and the producer-set value-table representation did not
change the verdict on the saved traces.

Goal: test whether the missing signal is in the full state but absent from the current
local stream observation or lossy `global_state()`.

Use a lossless or near-lossless observation builder, not just a larger flat global vector.
Required input should include:

```text
per-producer per-capacity-type current/max/reserved/requested
per-product inventory and inbound
all active orders with product, qty, accepted/completed/failed, TTL/age, type
open consumer demand
short history of capacity, rejections, routed quantity, completed units
```

Preferred representation:

```text
product-stream node
eligible producer nodes
capacity-type features per producer
queue/order tokens with TTL/age
inventory/inbound by product
3-5 cycle history features
```

Definition of done:

- Compare local obs, lossy global obs, full flat state, and relational state.
- Evaluate held-out `regret_frac`, not top-1.
- If relational state wins, compress it into the smallest stable feature set.
- If relational state still fails, the next hypothesis is label/objective noise, not
  "just add more local features".

Open caveats if this path is ever reopened:

- attention pooling was not tested;
- ranking/winner-set losses on the set encoder were not tested as a full experiment;
- per-producer routing-history features not present in the saved A0f traces were not
  tested.

### Task 3: Learner-Trajectory / Champion-Trajectory Collection

Status: **OPEN**, but lower priority than changing the target/data distribution. A0f/A0g
already fail on teacher trajectories, so learner-trajectory coverage alone cannot explain
the current wall.

Goal: detect covariate shift. A policy can look acceptable on teacher states and still
die when its own choices move it into different states.

Use or extend the A0u path in `tokenomics.eval.orderer_representability`:

```text
--a0u --task-ids <three A0t tasks> --eval-seeds 3 --horizon 100
```

Collect the champion/learner trajectory, then ask the same rollout teacher what it would
do at those visited states.

Definition of done:

- Each bad episode has an onset cycle, peak-regret cycle, and state snapshot.
- If failures only appear on champion/learner states, use DAgger-style relabeling with
  the improved observation from Tasks 1-2.
- If failures also appear on teacher states, fix observation/target first.

### Task 4: Replace Hard Labels With Value/Winner-Set Targets

Status: **PARTIALLY TESTED**. A0f/A0g value-table MSE did not rescue held-out regret.
Follow-up local checks also tried winner-set BCE and margin-weighted CE on saved A0f
features without a decisive rescue. Remaining target work must be more substantial than
just replacing hard argmax with a simple soft label.

Goal: stop punishing harmless alternatives and stop learning arbitrary argmax tie noise.
This comes after the missing-observation work, because soft labels cannot recover facts
the agent never sees.

Current bad target:

```text
choose blend_0.75
```

Better target:

```text
candidate values: even_split=45, blend_0.75=97, blend_0.5=97,
blend_0.25=97, proportional=97
winner set: {blend_0.75, blend_0.5, blend_0.25, proportional}
avoid: even_split, regret 52
```

Train/evaluate one of:

- softmax over candidate values;
- multi-label winner-set target;
- advantage/regret-weighted classification;
- pairwise ranking: candidate A should beat candidate B by observed value gap.

Definition of done:

- Held-out `regret_frac` improves, not just top-1.
- The model does not collapse to the majority candidate.
- Ties are handled as sets/distributions, not as arbitrary first argmax.

### Task 5: Window/Episode-Level Blend-Policy Distillation

Status: **DONE; TRAINED-ORDERER PATH STOPPED**.

Goal: train a useful orderer by changing the time scale of the target. A0f/A0g fixed the
hard-label problem by using per-cycle value tables, but they still tried to predict a
per-cycle table whose winner/value ordering does not transfer across demand seeds. The
next target should be stable over an episode or multi-cycle window.

Current implementation:

- `tokenomics/tokenomics/eval/orderer_representability.py` has `--a0w`, including fixed
  blend episode rollouts, episode/window value-table construction, constant-baseline
  regret, conservative decision-regret, EXP emission, and deployed rollout scoring.
- `tokenomics/tokenomics/train/orderer_scorer.py` has the deploy path:
  `aggregate_set_window`, `blend_decision`, and `set_table_ordering_policy`.
- The A0g `ProducerSetValueModel` is reused unchanged.
- Tests were added for A0w target math and deploy helpers; the reported full suite is
  647 passed.
- A local smoke run wrote `/tmp/a0w_smoke` and verified the end-to-end EXP layout. Treat
  those numbers as pipeline smoke only, not evidence about the model.

Pod result:

```text
Artifacts:
  experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0w/

Run config:
  --iters 1000
  --horizon 100
  --eval-seeds 3
  --windows 5,10,20
  --default-blend blend_0.5
  --threshold 0.25

Deployed rollout verdict:
  task  learned  default blend_0.5  best constant
  s29      810        960              768  (blend_0.25)
  s77      666        749              893  (blend_0.25)
  s45      588        621              621  (blend_0.5)
```

Read: learned loses to the default fixed blend on all 3 tasks and loses to the best
constant blend on 2 of 3. This meets the predeclared stop rule. Do not spend the full
81-row archive or threshold sweep on the learned A0w meta-orderer unless the owner
explicitly reopens this line.

Failed target shape:

```text
cycle 17: predict values for even_split / blends / proportional
cycle 18: predict values for even_split / blends / proportional
cycle 19: predict values for even_split / blends / proportional
```

New target shape:

```text
episode or 10-cycle window:
  even_split      -> value
  blend_0.75      -> value
  blend_0.5       -> value
  blend_0.25      -> value
  proportional    -> value
```

Plain-English lesson the model should learn:

```text
This regime usually wants a mid-blend routing style.
Do not chase noisy cycle-by-cycle winner flips.
Switch away from the default only when the advantage is large and stable.
```

Candidate policies to score:

```text
even_split
blend_0.75
blend_0.5
blend_0.25
proportional
champion_orderer  # optional baseline column, not necessarily a train output
```

Experiment design:

1. For each archived task and seed, evaluate each fixed blend policy over the whole episode
   and over rolling windows, e.g. 5, 10, and 20 cycles.
2. Build targets from the episode/window returns, not from per-cycle search argmaxes.
3. Train the producer-count-agnostic set value-table head, but with one row per
   episode/window instead of one row per searched cycle.
4. Add a conservative decision rule:

```text
default = best global robust blend, likely blend_0.5
switch only if predicted advantage over default > threshold
```

5. Evaluate the deployed learned orderer by rollout:

```text
champion producer + learned orderer
consumer_completed_units
death rate
closure_vs_search
held-out seeds
full 81-row round-2 archive
```

Metrics:

```text
episode/window heldout_regret_frac
deployed consumer units
death rate
closure_vs_search = (learned_cu - champion_cu) / (search_cu - champion_cu)
default-blend baseline comparison
```

Definition of done:

- Fixed blend baselines are reported first; if a constant blend already closes most of the
  gap, the learned model must beat that constant, not just beat champion/even_split.
- Learned model is evaluated on deployed rollouts, not only teacher-state table error.
- The report separates episode-level targets from window-level targets.
- The report states whether the learned model beats the best constant blend by more than
  seed noise.
- Primary acceptance is the deployed row: `learned_cu` must beat both `default_blend_cu`
  and `best_const_cu` on the held-out rollout. Good table regret without deployed-unit
  improvement is not enough.
- If learned deploy loses to default/best-constant on 2 of the 3 tasks, stop this path;
  do not spend the full 81-row run.
- If it wins strongly on all 3, or wins strongly on 2 of 3 including s45, then run a
  broader archive check and a threshold sweep.

Observed outcome:

- Stop condition fired. Learned deploy loses to default `blend_0.5` on 3/3 and to
  `best_const` on 2/3.
- Longer-window and episode table regret were not reliable deployment signals. Example:
  s77 episode/table regret looked perfect, but deployed learned rollout still lost to both
  `blend_0.5` and the train-selected best constant.
- Positive `closure_vs_search` on s77/s45 is not enough, because the champion was dead
  there; the honest fixed-blend baselines beat the learned model.

Why this is different from A0f/A0g:

```text
A0f/A0g:
  per-cycle feature -> per-cycle candidate value table

Task 5:
  episode/window feature summary -> episode/window blend-policy value table
```

This task kept the trained-agent goal while avoiding the seed-idiosyncratic per-cycle
search winner. It still failed in deployed rollouts, so the trained orderer arc is closed
for the tested A0b/A0c/A0f/A0g/A0w ladder.

## Tested Model: Producer-Count Agnostic Value Table

A0g implemented and evaluated the proposed deployable learned-agent head. It combines two
properties that previously existed only separately in the codebase:

```text
variable number of producers in
fixed candidate value table out
```

Current deployed `OrdererScoreModel` is producer-count agnostic, but it predicts
per-producer routing scores:

```text
producer A -> score
producer B -> score
producer C -> score
```

That answers "which producer should this stream prefer?" It does not directly answer
"which routing strategy is best here?"

A0f's `ValueTableModel` predicts the right kind of output:

```text
value(even_split)
value(blend_0.75)
value(blend_0.5)
value(blend_0.25)
value(proportional)
```

But A0f is a diagnostic probe with flat topology-specific features. Its input dimension
changes with producer/product count, for example `relational` dim 923 on s29 vs 1247 on
s77/s45. It is not a portable orderer.

The implemented deployable learned agent reads producers as a set:

```text
stream context:
  product demand, recipe facts, queue/TTL, open demand, recent history

producer rows:
  for each eligible producer:
    per-type current/max/free/reserved/requested capacity
    inventory and inbound for relevant products
    makeable-now and input availability
    recent routed/completed/rejected quantities

shared producer encoder:
  h_i = ProducerMLP(producer_i_features)

set aggregation:
  H = masked mean+max pool over {h_i}

candidate value head:
  output one scalar per routing candidate
```

Minimal candidate table:

```text
even_split
blend_0.75
blend_0.5
blend_0.25
proportional
```

Training target:

```text
teacher_values = rollout_units_per_candidate
target = teacher_values - mean(teacher_values)
```

Loss candidates:

```text
MSE on centered value table
pairwise ranking loss on value differences
softmax / KL over value-derived candidate probabilities
winner-set BCE for tied best candidates
regret-weighted CE as an auxiliary, not the primary target
```

Primary metric:

```text
heldout_regret_frac
```

Do not select by top-1 accuracy alone.

Implemented invariants:

- Reordering producer IDs must not change candidate values.
- Adding/removing producers changes only the producer set, not the model shape.
- Ineligible producers must be masked or represented with an explicit eligibility bit.
- Candidate output size is fixed by the candidate vocabulary, not by producer count.
- It was evaluated on unseen episode seeds from the saved A0f traces.

This design differs from the current implementations as follows:

```text
current deployed scorer:
  variable producer count -> per-producer routing scores

A0f diagnostic probe:
  fixed topology flat features -> candidate value table

proposed trained orderer:
  variable producer count -> candidate value table
```

Verdict from A0g:

```text
The set inductive bias did not change A0f's verdict.
The model fits train almost perfectly and lands near constant mid-blend floors held-out.
```

Use this code path as reusable infrastructure if a future experiment finds a stable signal
worth distilling, but do not assume the current EXP36 per-cycle blend winner is a stable
supervised target.

## Suggested Execution Order

1. Treat Task 1 and Task 2 as closed for the tested feature/model families.
2. Fix the A0g pooled-normalization caveat if pooled cross-task reads are reused.
3. Treat A0w as the trained-orderer stop gate: it lost to fixed blends in deployed units.
4. Use fixed `blend_0.5` as the zero-cost orderer floor, or per-task best-constant when
   an oracle/task-selection context permits it.
5. If EXP-0036 headroom remains worth chasing, open Stage A2 deploy-time light search over
   the blend vocabulary with the fixed mid-blend as the acceptance floor. That is a compute
   lever, not a learned-policy lever.

## Commands And Paths

Current useful local checks:

```bash
rtk .venv/bin/python -m pytest tokenomics/tests/test_orderer_scorer.py -q
rtk .venv/bin/python -m tokenomics.eval.orderer_representability --help
```

A0t command shape from the completed run:

```bash
rtk .venv/bin/python -m tokenomics.eval.orderer_representability --a0t \
  --archive <task_archive.jsonl> \
  --producer-ckpt <producer_champion_exp0035.pt> \
  --orderer-ckpt <orderer_scorer.pt> \
  --task-ids <comma-separated-task-ids> \
  --iters 1000 --horizon 100 --eval-seeds 3 \
  --alphas 0.75,0.5,0.25 \
  --out-dir <out-dir>
```

A0w command shape used for the completed pod run:

```bash
rtk .venv/bin/python -m tokenomics.eval.orderer_representability --a0w \
  --archive <task_archive.jsonl> \
  --producer-ckpt <producer_champion_exp0035.pt> \
  --orderer-ckpt <orderer_scorer.pt> \
  --task-ids <s29-task-id>,<s77-task-id>,<s45-task-id> \
  --iters 1000 --horizon 100 --eval-seeds 3 \
  --alphas 0.75,0.5,0.25 \
  --windows 5,10,20 \
  --default-blend blend_0.5 \
  --threshold 0.25 \
  --deploy-window 10 \
  --out-dir experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0w
```

Run heavy/full-horizon experiment work on a pod via `run_pod_training.py`; use local only
for code inspection, saved-artifact recomputes, and small tests. A0w has already consumed
its narrow 3-task gate; do not expand it without an explicit new reason.

## Worktree Note

As of this handover, the worktree already has uncommitted changes in EXP36 result files,
diagnostics, tests, and orderer code. Do not revert them unless the owner explicitly asks.
This handover intentionally adds only this new file.
