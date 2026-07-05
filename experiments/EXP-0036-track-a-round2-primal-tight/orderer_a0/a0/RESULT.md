# EXP-0036 Stage A0 orderer-harvest gate

Learnability-gated orderer harvest (EXP-0036 amendment). Per probe task: H3 headroom
baselines + a representation diagnostic + H1 variance-explained of a single-search vs
K-averaged distill, champion producer frozen.

**Read order:**

1. **Headroom (H3, first):** `search_cu` vs `even_split_cu` per task. If oracle ~= even_split
   there is no orderer lever in this regime => STOP (accept champion or re-mine tighter).
2. **Representation:** `named_capture_frac` = share of the search headroom a per-cycle
   NAMED-only search (even_split/proportional switching, no Gaussian samples) recovers.
   ~1 => the lever is per-cycle named-policy choice => a learnable categorical/blend teacher
   head captures it (switch representation). ~0 => the win is continuous routings => keep the
   continuous target and denoise it (H1).
3. **Denoise (H1):** does `ve_averaged` jump over `ve_single`? A jump => the single-search
   target was tie-arbitrary noise the K-average cancels => proceed to Stage A1. Neither moves
   but headroom exists => Stage A2 (deploy-time search). Neither moves, no headroom => track done.

**Means:** teacher_headroom=699.0, named_capture_frac=0.7839, ve_single=0.3892003333333333, ve_averaged=0.6180156666666666.

## Per-task

```json
[
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "champ_cu": 952.67,
    "search_cu": 1070.67,
    "even_split_cu": 809,
    "proportional_cu": 837.67,
    "search_named_cu": 1070.67,
    "named_capture_frac": 1.0,
    "teacher_headroom": 118.0,
    "headroom_frac": 0.1239,
    "k_searches": 4,
    "ve_single": 0.269441,
    "ve_averaged": 0.521589,
    "n_samples_single": 3709,
    "n_samples_averaged": 3794
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "champ_cu": 0,
    "search_cu": 997.33,
    "even_split_cu": 513.67,
    "proportional_cu": 681.33,
    "search_named_cu": 847.67,
    "named_capture_frac": 0.6906,
    "teacher_headroom": 997.33,
    "headroom_frac": null,
    "k_searches": 4,
    "ve_single": 0.255795,
    "ve_averaged": 0.578624,
    "n_samples_single": 5127,
    "n_samples_averaged": 5156
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "champ_cu": 0,
    "search_cu": 981.67,
    "even_split_cu": 199.67,
    "proportional_cu": 367,
    "search_named_cu": 716.67,
    "named_capture_frac": 0.6611,
    "teacher_headroom": 981.67,
    "headroom_frac": null,
    "k_searches": 4,
    "ve_single": 0.642365,
    "ve_averaged": 0.753834,
    "n_samples_single": 4608,
    "n_samples_averaged": 4625
  }
]
```

## Conclusions

_TODO (human): read headroom first, then the VE jump; pick the branch._
