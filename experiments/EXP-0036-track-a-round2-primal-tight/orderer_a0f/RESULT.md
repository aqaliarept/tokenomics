# EXP-0036 Stage A0f missing-observation probe (which feature block carries the choice)

HANDOVER Task 1: the orderer fits its training seeds and fails held-out — is the wall
missing STATE, missing trajectory coverage, or bad labels? Per cumulative block set a
value-table probe (full candidate table target, no hard-argmax tie noise) is fit on the
train-seed cycles; every row reports held-out regret_frac — the share of the teacher's
per-cycle edge over even_split the probe loses (cost-weighted, ties free).

**Read order:**

1. Per task, scan `regret_frac` down the ladder. A block that sharply reduces it names
   the missing observation family — that block is the enrichment candidate.
2. No flat block reduces it but `+relational` does => the signal is relational, compress
   that block into the smallest stable feature set (Task 2).
3. Nothing reduces it: check `train_regret_frac`. High => the target is not even fittable
   on train states with full features (label/objective noise — Task 4 soft targets).
   Low => fits train, fails held-out => trajectory/coverage wall (Task 3 DAgger states).
4. `gstate` row = the lossy A0c input under the VALUE target — compares target forms
   at fixed features against the A0t hard-argmax probe (regret_frac 0.98/0.86/0.56).

## Regret ladder per task

### mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29

```
block_set                dim  train_rf heldout_rf   top1 winner_hit
gstate                   109    0.0000     0.9863 0.1600     0.5500
stream                    81    0.5290     0.8965 0.1100     0.5800
+per_type_capacity       153    0.0000     0.9258 0.2800     0.5900
+order_queue             287    0.0000     1.1406 0.1000     0.4500
+inventory               359    0.0000     1.0664 0.2000     0.4700
+inbound                 431    0.0000     0.9590 0.2700     0.5400
+history                 491    0.0000     1.0508 0.2400     0.5300
+relational              923    0.0000     0.9902 0.2500     0.5200
```

### mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77

```
block_set                dim  train_rf heldout_rf   top1 winner_hit
gstate                   151    0.0000     1.1254 0.1900     0.4200
stream                    81    0.4874     1.2586 0.1100     0.3900
+per_type_capacity       189    0.0000     1.1003 0.1900     0.4200
+order_queue             323    0.0000     0.7555 0.2600     0.5000
+inventory               431    0.0000     0.7633 0.2400     0.5000
+inbound                 539    0.0000     0.7774 0.2500     0.5000
+history                 599    0.0000     0.7618 0.2300     0.5200
+relational             1247    0.0000     1.1301 0.2100     0.4200
```

### mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45

```
block_set                dim  train_rf heldout_rf   top1 winner_hit
gstate                   151    0.0000     0.4070 0.4400     0.6600
stream                    81    0.8096     0.3650 0.3400     0.6200
+per_type_capacity       189    0.0000     0.8933 0.3300     0.4500
+order_queue             323    0.0000     0.6477 0.2000     0.4800
+inventory               431    0.0000     0.6458 0.2300     0.4600
+inbound                 539    0.0000     0.8033 0.2700     0.4400
+history                 599    0.0000     0.7857 0.2400     0.4500
+relational             1247    0.0000     0.8483 0.1900     0.4100
```

## Per-row detail

```json
[
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "gstate",
    "blocks": [
      "gstate"
    ],
    "dim": 109,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.46,
    "train_winner_hit": 1.0,
    "regret_frac": 0.9863,
    "top1": 0.16,
    "winner_hit": 0.55,
    "regret_total": 505.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "stream",
    "blocks": [
      "stream"
    ],
    "dim": 81,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.529,
    "train_top1": 0.09,
    "train_winner_hit": 0.55,
    "regret_frac": 0.8965,
    "top1": 0.11,
    "winner_hit": 0.58,
    "regret_total": 459.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "+per_type_capacity",
    "blocks": [
      "stream",
      "per_type_capacity"
    ],
    "dim": 153,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.475,
    "train_winner_hit": 1.0,
    "regret_frac": 0.9258,
    "top1": 0.28,
    "winner_hit": 0.59,
    "regret_total": 474.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "+order_queue",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue"
    ],
    "dim": 287,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.675,
    "train_winner_hit": 1.0,
    "regret_frac": 1.1406,
    "top1": 0.1,
    "winner_hit": 0.45,
    "regret_total": 584.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "+inventory",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory"
    ],
    "dim": 359,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.73,
    "train_winner_hit": 1.0,
    "regret_frac": 1.0664,
    "top1": 0.2,
    "winner_hit": 0.47,
    "regret_total": 546.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "+inbound",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound"
    ],
    "dim": 431,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.44,
    "train_winner_hit": 1.0,
    "regret_frac": 0.959,
    "top1": 0.27,
    "winner_hit": 0.54,
    "regret_total": 491.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "+history",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound",
      "history"
    ],
    "dim": 491,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.385,
    "train_winner_hit": 1.0,
    "regret_frac": 1.0508,
    "top1": 0.24,
    "winner_hit": 0.53,
    "regret_total": 538.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "block_set": "+relational",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound",
      "history",
      "relational"
    ],
    "dim": 923,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.54,
    "train_winner_hit": 1.0,
    "regret_frac": 0.9902,
    "top1": 0.25,
    "winner_hit": 0.52,
    "regret_total": 507.0,
    "headroom_total": 512.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "gstate",
    "blocks": [
      "gstate"
    ],
    "dim": 151,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.8,
    "train_winner_hit": 1.0,
    "regret_frac": 1.1254,
    "top1": 0.19,
    "winner_hit": 0.42,
    "regret_total": 718.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "stream",
    "blocks": [
      "stream"
    ],
    "dim": 81,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.4874,
    "train_top1": 0.135,
    "train_winner_hit": 0.48,
    "regret_frac": 1.2586,
    "top1": 0.11,
    "winner_hit": 0.39,
    "regret_total": 803.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "+per_type_capacity",
    "blocks": [
      "stream",
      "per_type_capacity"
    ],
    "dim": 189,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.635,
    "train_winner_hit": 1.0,
    "regret_frac": 1.1003,
    "top1": 0.19,
    "winner_hit": 0.42,
    "regret_total": 702.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "+order_queue",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue"
    ],
    "dim": 323,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.585,
    "train_winner_hit": 1.0,
    "regret_frac": 0.7555,
    "top1": 0.26,
    "winner_hit": 0.5,
    "regret_total": 482.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "+inventory",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory"
    ],
    "dim": 431,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.89,
    "train_winner_hit": 1.0,
    "regret_frac": 0.7633,
    "top1": 0.24,
    "winner_hit": 0.5,
    "regret_total": 487.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "+inbound",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound"
    ],
    "dim": 539,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.835,
    "train_winner_hit": 1.0,
    "regret_frac": 0.7774,
    "top1": 0.25,
    "winner_hit": 0.5,
    "regret_total": 496.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "+history",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound",
      "history"
    ],
    "dim": 599,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.63,
    "train_winner_hit": 1.0,
    "regret_frac": 0.7618,
    "top1": 0.23,
    "winner_hit": 0.52,
    "regret_total": 486.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "block_set": "+relational",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound",
      "history",
      "relational"
    ],
    "dim": 1247,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.83,
    "train_winner_hit": 1.0,
    "regret_frac": 1.1301,
    "top1": 0.21,
    "winner_hit": 0.42,
    "regret_total": 721.0,
    "headroom_total": 638.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "gstate",
    "blocks": [
      "gstate"
    ],
    "dim": 151,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.83,
    "train_winner_hit": 1.0,
    "regret_frac": 0.407,
    "top1": 0.44,
    "winner_hit": 0.66,
    "regret_total": 416.0,
    "headroom_total": 1022.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "stream",
    "blocks": [
      "stream"
    ],
    "dim": 81,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.8096,
    "train_top1": 0.335,
    "train_winner_hit": 0.49,
    "regret_frac": 0.365,
    "top1": 0.34,
    "winner_hit": 0.62,
    "regret_total": 373.0,
    "headroom_total": 1022.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "+per_type_capacity",
    "blocks": [
      "stream",
      "per_type_capacity"
    ],
    "dim": 189,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.67,
    "train_winner_hit": 1.0,
    "regret_frac": 0.8933,
    "top1": 0.33,
    "winner_hit": 0.45,
    "regret_total": 913.0,
    "headroom_total": 1022.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "+order_queue",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue"
    ],
    "dim": 323,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.66,
    "train_winner_hit": 1.0,
    "regret_frac": 0.6477,
    "top1": 0.2,
    "winner_hit": 0.48,
    "regret_total": 662.0,
    "headroom_total": 1022.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "+inventory",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory"
    ],
    "dim": 431,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.875,
    "train_winner_hit": 1.0,
    "regret_frac": 0.6458,
    "top1": 0.23,
    "winner_hit": 0.46,
    "regret_total": 660.0,
    "headroom_total": 1022.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "+inbound",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound"
    ],
    "dim": 539,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.65,
    "train_winner_hit": 1.0,
    "regret_frac": 0.8033,
    "top1": 0.27,
    "winner_hit": 0.44,
    "regret_total": 821.0,
    "headroom_total": 1022.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "+history",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound",
      "history"
    ],
    "dim": 599,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.595,
    "train_winner_hit": 1.0,
    "regret_frac": 0.7857,
    "top1": 0.24,
    "winner_hit": 0.45,
    "regret_total": 803.0,
    "headroom_total": 1022.0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "block_set": "+relational",
    "blocks": [
      "stream",
      "per_type_capacity",
      "order_queue",
      "inventory",
      "inbound",
      "history",
      "relational"
    ],
    "dim": 1247,
    "n_train": 200,
    "n_heldout": 100,
    "train_regret_frac": 0.0,
    "train_top1": 0.7,
    "train_winner_hit": 1.0,
    "regret_frac": 0.8483,
    "top1": 0.19,
    "winner_hit": 0.41,
    "regret_total": 867.0,
    "headroom_total": 1022.0
  }
]
```

## Conclusions

**Verdict: NONE of the tested feature families produced a stable held-out regret reduction
on these three tasks — the tested missing-observation ladder is closed (HANDOVER Task 1
done). This does not rule out all possible observations/models; it rules out these families,
under these probe models, at this data scale.** (Run: pod 3f6606gset773e, iters 1000,
horizon 100, eval_seeds 3, alphas 0.75/0.5/0.25; follow-up recomputes in
`analysis_followup.py` / `.log` over `artifacts/feature_traces.json.gz`.)

Evidence (three independent reads, all pointing the same way):

1. **No block produces a stable held-out reduction.** Per task, held-out `regret_frac`
   ranges: s29 0.90–1.14 (best = stream-only 0.8965, exactly the constant-blend_0.5 floor);
   s77 0.76–1.26; s45 0.365–0.89 (best = the two SMALLEST sets, stream 0.365 / gstate 0.407
   — every added block makes s45 WORSE, the small-sample overfitting signature). Several
   sets sit above 1.0, i.e. worse than always picking even_split. `+relational` (recipe-
   aware product x producer x capacity ratios) never helps — Task 2's flat-relational read
   is answered in the same run. Train side: `train_regret_frac = 0.0` for every set with
   dim >= 109 (perfect memorization, no transfer); only the 81-dim stream block underfits
   (train 0.53/0.49/0.81).
2. **Constant picks match or beat the fitted probes almost everywhere**
   (`analysis_followup.log`): always-`blend_0.5` gives held-out regret_frac 0.90/0.97/0.29
   (s29/s77/s45), always-`blend_0.75` 0.96/0.78/0.37. On s29 the best fit ties the constant
   floor; on s45 the best fit (0.365) loses to constant blend_0.5 (0.293). The single
   exception: s77 `+order_queue` 0.7555 edges constant blend_0.75 (0.7821) by 0.027 — a
   marginal, one-task/one-held-out-seed edge that vanishes again at `+relational` (1.13),
   not a stable reduction.
3. **More data makes it worse.** 1-seed (n=100) -> 2-seed (n=200) training RAISES held-out
   regret on all three tasks (0.88->0.97, 1.06->1.13, 0.62->0.80). The second seed's cycles
   actively mislead on the held-out seed — cross-seed target shift, not a sample-starvation
   slope worth riding.
4. **Task 4 target softening does not rescue it**: winner-set multi-label BCE and
   margin-weighted CE (fit on the same features) also land at or above the constant floors
   (best 0.65 winner-set/gstate on s45 vs 0.29 constant).

What IS stable: the teacher's edge over even_split is substantially carried by a constant
"prefer a mid blend" policy (dramatically so on s45), plus a per-cycle switching residual
that none of the tested observations predicts across demand seeds. The lookahead-6 candidate
table on a new seed behaves like noise around that mean — consistent with A0t's margin==0
cycles carrying 52u regret and the 42–50% flat-target fractions in the saved-sample
recompute.

Branch decision (HANDOVER queue): Task 1 done (tested ladder closed, no family reduces
regret), Task 2's flat-relational variant done (fails; a graph encoder over the same facts
remains untested but has no positive signal to build on), Task 4's soft targets tested
(fail). Task 3 (learner-trajectory DAgger) is moot for THIS wall: train and held-out are
BOTH teacher trajectories differing only by demand seed, and deployment always faces new
seeds — relabeling cannot remove cross-seed idiosyncrasy. The live branch is **A2
deploy-time light search** (candidate set = blend vocabulary + champion, per the A0t read)
— with the new cheap datum that a FIXED mid-blend policy already captures a large share on
some tasks and belongs in the acceptance comparison.
