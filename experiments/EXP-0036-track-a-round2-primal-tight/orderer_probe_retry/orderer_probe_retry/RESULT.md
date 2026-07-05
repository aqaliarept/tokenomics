# EXP-0032 v4 orderer oracle-relabel representability

Decision gate for v4 (EXP-0032): can the portable orderer scorer head represent the
mining bar's per-cycle search routing? Distill a fresh orderer on each task's
oracle-relabel targets (champion producer drives collection), then compare on-task
champion / search-teacher / oracle / bar consumer units. Producer frozen at champion.

**Gate:** mean `recovered_vs_teacher` = -0.159 (>= ~0.5 => head represents the search).

**Verdict:** architecture limit likely real — escalate (richer stream obs / deploy-time search).

## Per-task

```json
[
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "champ_cu": 952.67,
    "search_cu": 1070.67,
    "oracle_cu": 800,
    "bar_cu": 1086,
    "teacher_headroom": 118.0,
    "bar_headroom": 133.33,
    "recovered": -152.67,
    "recovered_vs_teacher": -1.294,
    "recovered_vs_bar": -1.145,
    "n_samples": 3709,
    "sample_masked_mse": 0.219857,
    "sample_masked_n": 7418,
    "sample_top1_match": 0.598274,
    "sample_top1_n": 3709,
    "trace_cycles": 300,
    "trace_dead": false,
    "trace_first_dead_cycle": null,
    "trace_first_zero_capacity_cycle": null,
    "trace_min_capacity_type_current": 13
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "champ_cu": 0,
    "search_cu": 997.33,
    "oracle_cu": 439,
    "bar_cu": 973.33,
    "teacher_headroom": 997.33,
    "bar_headroom": 973.33,
    "recovered": 439,
    "recovered_vs_teacher": 0.44,
    "recovered_vs_bar": 0.451,
    "n_samples": 5127,
    "sample_masked_mse": 0.23256,
    "sample_masked_n": 15381,
    "sample_top1_match": 0.502438,
    "sample_top1_n": 5127,
    "trace_cycles": 208,
    "trace_dead": false,
    "trace_first_dead_cycle": 8,
    "trace_first_zero_capacity_cycle": 8,
    "trace_min_capacity_type_current": 0
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "champ_cu": 0,
    "search_cu": 981.67,
    "oracle_cu": 370,
    "bar_cu": 870.67,
    "teacher_headroom": 981.67,
    "bar_headroom": 870.67,
    "recovered": 370,
    "recovered_vs_teacher": 0.377,
    "recovered_vs_bar": 0.425,
    "n_samples": 4608,
    "sample_masked_mse": 0.38626,
    "sample_masked_n": 13824,
    "sample_top1_match": 0.553819,
    "sample_top1_n": 4608,
    "trace_cycles": 213,
    "trace_dead": false,
    "trace_first_dead_cycle": 13,
    "trace_first_zero_capacity_cycle": 13,
    "trace_min_capacity_type_current": 0
  }
]
```

## Conclusions

_TODO (human): confirm go/no-go on the full v4 `run_stage4_replay` run._
