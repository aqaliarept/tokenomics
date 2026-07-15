# EXP-0036 Stage A0t trace-regret analysis (which decisions go wrong, and what they cost)

A0b/A0c only measured aggregate top-1 vs the majority floor — that cannot separate
'misses ties that don't matter' from 'misses the calls that carry the units'. Here every
held-out searched cycle keeps its full candidate table, so a student's miss is weighted
by REGRET (units[teacher pick] - units[student pick], lookahead-window units — a local
proxy, not episode units). The champion orderer is scored at the SAME states by the same
rollout (champ_regret; negative = champion beats the whole vocabulary there).

**Read order:**

1. **probe_regret_frac** (probe_regret_total / local_headroom_total): the share of the
   teacher's per-cycle edge over even_split the global-state probe loses. ~0 despite low
   top-1 => misses are ties, the D1 'wall' costs nothing, distillation is fine as-is.
   Large => the misses are expensive; go to 2.
2. **Top-regret cycles** (below, with state snapshots): read WHAT the teacher saw coming
   that the pick missed — arriving inbound stock, per-type bottleneck, TTL cliff. Infer
   the missing feature from evidence, not from a menu.
3. **champ_regret_frac / n_champ_better:** where the deployed champion stands at the
   same states — large positive => champion's misses are costly too (consistent with its
   0-unit collapses); many negatives => the vocabulary itself is the ceiling, widen it.

**Means:** probe_top1=0.35000000000000003, probe_regret_frac=0.8017333333333333, champ_regret_frac=0.2985333333333333, local_headroom_total=724.0.

## Per-task

```json
[
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ],
    "n_traces_train": 200,
    "n_traces_heldout": 100,
    "probe_top1": 0.28,
    "probe_regret_total": 503.0,
    "probe_regret_mean": 5.03,
    "probe_regret_max": 22.0,
    "n_probe_miss": 50,
    "champ_regret_total": 265.0,
    "champ_regret_mean": 2.65,
    "champ_regret_max": 33.0,
    "n_champ_worse": 52,
    "n_champ_better": 40,
    "local_headroom_total": 512.0,
    "probe_regret_frac": 0.9824,
    "champ_regret_frac": 0.5176
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ],
    "n_traces_train": 200,
    "n_traces_heldout": 100,
    "probe_top1": 0.38,
    "probe_regret_total": 550.0,
    "probe_regret_mean": 5.5,
    "probe_regret_max": 39.0,
    "n_probe_miss": 50,
    "champ_regret_total": 130.0,
    "champ_regret_mean": 1.3,
    "champ_regret_max": 39.0,
    "n_champ_worse": 41,
    "n_champ_better": 47,
    "local_headroom_total": 638.0,
    "probe_regret_frac": 0.8621,
    "champ_regret_frac": 0.2038
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ],
    "n_traces_train": 200,
    "n_traces_heldout": 100,
    "probe_top1": 0.39,
    "probe_regret_total": 573.0,
    "probe_regret_mean": 5.73,
    "probe_regret_max": 52.0,
    "n_probe_miss": 43,
    "champ_regret_total": 178.0,
    "champ_regret_mean": 1.78,
    "champ_regret_max": 47.0,
    "n_champ_worse": 46,
    "n_champ_better": 41,
    "local_headroom_total": 1022.0,
    "probe_regret_frac": 0.5607,
    "champ_regret_frac": 0.1742
  }
]
```

## Top-regret held-out cycles — mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29 (seed 31)

### cycle 73 — probe_regret 22.0, margin 6.0
- candidates: even_split=41.0, blend_0.75=52.0, blend_0.5=68.0, blend_0.25=55.0, proportional=74.0
- teacher pick: `proportional`; probe pick: `blend_0.75`; champion: 55.0 (regret 19.0)

```json
{
 "cycle": 73,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 95
   },
   "max_capacities": {
    "capacity-0": 115
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 34
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 64
   },
   "max_capacities": {
    "capacity-0": 100
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 26
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 119
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 72
   },
   "requested_by_type": {
    "capacity-1": 210
   },
   "inventory": {
    "1": 3,
    "2": 3,
    "0": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 124
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 72
   },
   "requested_by_type": {},
   "inventory": {
    "1": 2,
    "2": 15,
    "0": 7,
    "3": 5
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 164
   },
   "max_capacities": {
    "capacity-2": 173
   },
   "reserved": {
    "capacity-2": 77
   },
   "requested_by_type": {
    "capacity-2": 41
   },
   "inventory": {
    "11": 3,
    "12": 5,
    "13": 8,
    "10": 5
   },
   "inbound": {
    "13": 9,
    "10": 9,
    "12": 9
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 161
   },
   "max_capacities": {
    "capacity-2": 168
   },
   "reserved": {
    "capacity-2": 102
   },
   "requested_by_type": {
    "capacity-2": 39
   },
   "inventory": {
    "12": 3,
    "11": 2,
    "10": 5
   },
   "inbound": {
    "13": 12,
    "10": 9,
    "12": 11
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 20,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 9,
   "accepted": 9,
   "completed": 2,
   "failed": 0,
   "tokens": 45.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 9,
   "accepted": 9,
   "completed": 2,
   "failed": 0,
   "tokens": 45.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 7,
   "tokens": 45.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 55.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 6,
   "accepted": 6,
   "completed": 3,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 12,
   "accepted": 12,
   "completed": 6,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 13,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 0,
   "tokens": 45.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 12,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 60.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 12,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 13,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 7,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "20": [
   20,
   1066.0
  ],
  "22": [
   0,
   0.0
  ],
  "21": [
   12,
   533.0
  ],
  "23": [
   0,
   0.0
  ]
 }
}
```

### cycle 74 — probe_regret 22.0, margin 3.0
- candidates: even_split=49.0, blend_0.75=64.0, blend_0.5=71.0, blend_0.25=58.0, proportional=68.0
- teacher pick: `blend_0.5`; probe pick: `even_split`; champion: 60.0 (regret 11.0)

```json
{
 "cycle": 74,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 91
   },
   "max_capacities": {
    "capacity-0": 115
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 34
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 58
   },
   "max_capacities": {
    "capacity-0": 100
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 20
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 112
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 84
   },
   "requested_by_type": {
    "capacity-1": 114
   },
   "inventory": {
    "0": 3,
    "3": 9
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 115
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 48
   },
   "requested_by_type": {
    "capacity-1": 124
   },
   "inventory": {
    "1": 2,
    "3": 5
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 173
   },
   "max_capacities": {
    "capacity-2": 173
   },
   "reserved": {
    "capacity-2": 67
   },
   "requested_by_type": {
    "capacity-2": 83
   },
   "inventory": {
    "11": 3,
    "12": 14,
    "13": 1,
    "10": 10
   },
   "inbound": {
    "13": 9,
    "10": 11,
    "11": 5
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 168
   },
   "max_capacities": {
    "capacity-2": 168
   },
   "reserved": {
    "capacity-2": 84
   },
   "requested_by_type": {
    "capacity-2": 57
   },
   "inventory": {
    "12": 14,
    "11": 4,
    "10": 2
   },
   "inbound": {
    "13": 12,
    "10": 12
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 21,
   "qty": 6,
   "accepted": 6,
   "completed": 3,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 12,
   "accepted": 12,
   "completed": 6,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 13,
   "accepted": 13,
   "completed": 7,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 13,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 0,
   "tokens": 45.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 12,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 60.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 55.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 5,
   "accepted": 5,
   "completed": 2,
   "failed": 0,
   "tokens": 25.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 12,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 60.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 7,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 2,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 7,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "21": [
   0,
   0.0
  ],
  "23": [
   2,
   533.0
  ],
  "20": [
   14,
   1066.0
  ]
 }
}
```

### cycle 91 — probe_regret 22.0, margin 14.0
- candidates: even_split=56.0, blend_0.75=55.0, blend_0.5=78.0, blend_0.25=64.0, proportional=64.0
- teacher pick: `blend_0.5`; probe pick: `even_split`; champion: 59.0 (regret 19.0)

```json
{
 "cycle": 91,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 75
   },
   "max_capacities": {
    "capacity-0": 115
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 38
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 100
   },
   "max_capacities": {
    "capacity-0": 100
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 26
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 101
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 74
   },
   "requested_by_type": {
    "capacity-1": 120
   },
   "inventory": {
    "1": 2,
    "0": 9,
    "3": 5
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 106
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 24
   },
   "requested_by_type": {
    "capacity-1": 92
   },
   "inventory": {
    "1": 1,
    "3": 4
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 173
   },
   "max_capacities": {
    "capacity-2": 173
   },
   "reserved": {
    "capacity-2": 75
   },
   "requested_by_type": {
    "capacity-2": 40
   },
   "inventory": {
    "11": 3,
    "12": 11,
    "13": 2,
    "10": 8
   },
   "inbound": {
    "10": 14,
    "11": 10,
    "13": 4
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 161
   },
   "max_capacities": {
    "capacity-2": 168
   },
   "reserved": {
    "capacity-2": 66
   },
   "requested_by_type": {
    "capacity-2": 50
   },
   "inventory": {
    "12": 6,
    "11": 4,
    "10": 6,
    "13": 5
   },
   "inbound": {
    "10": 16,
    "13": 5
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 21,
   "qty": 13,
   "accepted": 13,
   "completed": 6,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 9,
   "accepted": 7,
   "completed": 0,
   "failed": 0,
   "tokens": 45.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 10,
   "accepted": 10,
   "completed": 5,
   "failed": 0,
   "tokens": 50.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 8,
   "accepted": 4,
   "completed": 0,
   "failed": 0,
   "tokens": 40.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 22,
   "qty": 4,
   "accepted": 4,
   "completed": 2,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 11,
   "accepted": 11,
   "completed": 5,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 5,
   "accepted": 2,
   "completed": 0,
   "failed": 0,
   "tokens": 25.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 4,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 20.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 8,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 40.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 5,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 25.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 5,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 7,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 5,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "21": [
   0,
   0.0
  ],
  "20": [
   5,
   533.0
  ],
  "22": [
   5,
   533.0
  ],
  "23": [
   7,
   533.0
  ]
 }
}
```

### cycle 55 — probe_regret 21.0, margin 15.0
- candidates: even_split=57.0, blend_0.75=63.0, blend_0.5=63.0, blend_0.25=63.0, proportional=78.0
- teacher pick: `proportional`; probe pick: `even_split`; champion: 65.0 (regret 13.0)

```json
{
 "cycle": 55,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 115
   },
   "max_capacities": {
    "capacity-0": 115
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 36
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 76
   },
   "max_capacities": {
    "capacity-0": 100
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 16
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 137
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 96
   },
   "requested_by_type": {
    "capacity-1": 48
   },
   "inventory": {
    "1": 3,
    "2": 3,
    "0": 3,
    "3": 8
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 142
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 60
   },
   "requested_by_type": {
    "capacity-1": 40
   },
   "inventory": {
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 165
   },
   "max_capacities": {
    "capacity-2": 173
   },
   "reserved": {
    "capacity-2": 20
   },
   "requested_by_type": {
    "capacity-2": 85
   },
   "inventory": {
    "11": 5,
    "12": 3,
    "13": 2,
    "10": 3
   },
   "inbound": {
    "13": 10,
    "10": 7,
    "12": 11
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 160
   },
   "max_capacities": {
    "capacity-2": 168
   },
   "reserved": {
    "capacity-2": 60
   },
   "requested_by_type": {
    "capacity-2": 80
   },
   "inventory": {
    "11": 5,
    "10": 5,
    "13": 1
   },
   "inbound": {
    "13": 6,
    "12": 12
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 13,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 50.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 6,
   "accepted": 6,
   "completed": 0,
   "failed": 0,
   "tokens": 30.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 22,
   "qty": 11,
   "accepted": 11,
   "completed": 6,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 14,
   "accepted": 7,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 10,
   "accepted": 10,
   "completed": 5,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 7,
   "accepted": 7,
   "completed": 3,
   "failed": 0,
   "tokens": 35.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 11,
   "accepted": 11,
   "completed": 5,
   "failed": 0,
   "tokens": 55.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 60.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 7,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 9,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 6,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "22": [
   7,
   267.0
  ],
  "20": [
   7,
   533.0
  ],
  "23": [
   15,
   1066.0
  ]
 }
}
```

### cycle 65 — probe_regret 18.0, margin 18.0
- candidates: even_split=50.0, blend_0.75=51.0, blend_0.5=51.0, blend_0.25=57.0, proportional=75.0
- teacher pick: `proportional`; probe pick: `blend_0.25`; champion: 54.0 (regret 21.0)

```json
{
 "cycle": 65,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 75
   },
   "max_capacities": {
    "capacity-0": 115
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 35
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 64
   },
   "max_capacities": {
    "capacity-0": 100
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 13
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 121
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 60
   },
   "requested_by_type": {
    "capacity-1": 76
   },
   "inventory": {
    "1": 6,
    "2": 4,
    "0": 2,
    "3": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 100
   },
   "max_capacities": {
    "capacity-1": 175
   },
   "reserved": {
    "capacity-1": 48
   },
   "requested_by_type": {
    "capacity-1": 60
   },
   "inventory": {
    "1": 6,
    "2": 3,
    "0": 1,
    "3": 3
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 173
   },
   "max_capacities": {
    "capacity-2": 173
   },
   "reserved": {
    "capacity-2": 95
   },
   "requested_by_type": {
    "capacity-2": 66
   },
   "inventory": {
    "11": 14,
    "12": 2,
    "13": 7,
    "10": 3
   },
   "inbound": {
    "10": 11,
    "12": 11
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 168
   },
   "max_capacities": {
    "capacity-2": 168
   },
   "reserved": {
    "capacity-2": 96
   },
   "requested_by_type": {
    "capacity-2": 66
   },
   "inventory": {
    "11": 2,
    "13": 8
   },
   "inbound": {
    "10": 11,
    "12": 15
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 21,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 11,
   "accepted": 11,
   "completed": 5,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 11,
   "accepted": 11,
   "completed": 6,
   "failed": 0,
   "tokens": 533.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 8,
   "accepted": 8,
   "completed": 4,
   "failed": 0,
   "tokens": 40.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 11,
   "accepted": 5,
   "completed": 0,
   "failed": 0,
   "tokens": 55.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 12,
   "accepted": 12,
   "completed": 6,
   "failed": 0,
   "tokens": 60.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 22,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 3,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 15.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 55.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 3,
   "accepted": 3,
   "completed": 2,
   "failed": 0,
   "tokens": 15.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 10,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 14,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 3,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 533.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "21": [
   10,
   533.0
  ],
  "20": [
   3,
   533.0
  ],
  "22": [
   0,
   0.0
  ],
  "23": [
   14,
   533.0
  ]
 }
}
```

## Top-regret held-out cycles — mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77 (seed 79)

### cycle 67 — probe_regret 39.0, margin 3.0
- candidates: even_split=94.0, blend_0.75=97.0, blend_0.5=94.0, blend_0.25=83.0, proportional=58.0
- teacher pick: `blend_0.75`; probe pick: `proportional`; champion: 95.0 (regret 2.0)

```json
{
 "cycle": 67,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 33
   },
   "max_capacities": {
    "capacity-0": 53
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 30
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 70
   },
   "max_capacities": {
    "capacity-0": 110
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 41
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 27
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 21
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 88
   },
   "max_capacities": {
    "capacity-1": 141
   },
   "reserved": {
    "capacity-1": 61
   },
   "requested_by_type": {
    "capacity-1": 22
   },
   "inventory": {
    "3": 5,
    "2": 3,
    "0": 9,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 79
   },
   "max_capacities": {
    "capacity-1": 118
   },
   "reserved": {
    "capacity-1": 40
   },
   "requested_by_type": {
    "capacity-1": 16
   },
   "inventory": {
    "3": 3,
    "0": 8,
    "1": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 79
   },
   "max_capacities": {
    "capacity-1": 139
   },
   "reserved": {
    "capacity-1": 58
   },
   "requested_by_type": {
    "capacity-1": 22
   },
   "inventory": {
    "2": 2,
    "3": 3,
    "0": 10,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 195
   },
   "max_capacities": {
    "capacity-2": 227
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-2": 36
   },
   "inventory": {
    "12": 8,
    "13": 6,
    "10": 2,
    "11": 4
   },
   "inbound": {
    "10": 9,
    "11": 8
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 177
   },
   "max_capacities": {
    "capacity-2": 229
   },
   "reserved": {
    "capacity-2": 24
   },
   "requested_by_type": {
    "capacity-2": 54
   },
   "inventory": {
    "10": 1,
    "11": 1,
    "12": 11,
    "13": 3
   },
   "inbound": {
    "10": 13,
    "11": 11
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 204
   },
   "max_capacities": {
    "capacity-2": 222
   },
   "reserved": {
    "capacity-2": 96
   },
   "requested_by_type": {
    "capacity-2": 54
   },
   "inventory": {
    "10": 1,
    "12": 3,
    "11": 8,
    "13": 3
   },
   "inbound": {
    "10": 14,
    "11": 11
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 20,
   "qty": 6,
   "accepted": 6,
   "completed": 4,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 13,
   "accepted": 13,
   "completed": 4,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 9,
   "accepted": 9,
   "completed": 3,
   "failed": 0,
   "tokens": 54.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 26.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 28.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 5,
   "accepted": 5,
   "completed": 3,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 8,
   "accepted": 8,
   "completed": 5,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 11,
   "accepted": 11,
   "completed": 7,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 48.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 22.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 22.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 3,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 2,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 11,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "20": [
   3,
   633.0
  ],
  "21": [
   13,
   1266.0
  ]
 }
}
```

### cycle 23 — probe_regret 35.0, margin 0.0
- candidates: even_split=41.0, blend_0.75=76.0, blend_0.5=76.0, blend_0.25=76.0, proportional=76.0
- teacher pick: `blend_0.75`; probe pick: `even_split`; champion: 67.0 (regret 9.0)

```json
{
 "cycle": 23,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 41
   },
   "max_capacities": {
    "capacity-0": 53
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 15
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 90
   },
   "max_capacities": {
    "capacity-0": 110
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 27
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 23
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 10
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 98
   },
   "max_capacities": {
    "capacity-1": 141
   },
   "reserved": {
    "capacity-1": 28
   },
   "requested_by_type": {
    "capacity-1": 33
   },
   "inventory": {
    "3": 4,
    "2": 2,
    "0": 2,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 97
   },
   "max_capacities": {
    "capacity-1": 118
   },
   "reserved": {
    "capacity-1": 24
   },
   "requested_by_type": {
    "capacity-1": 26
   },
   "inventory": {
    "3": 1,
    "0": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 89
   },
   "max_capacities": {
    "capacity-1": 139
   },
   "reserved": {
    "capacity-1": 33
   },
   "requested_by_type": {
    "capacity-1": 31
   },
   "inventory": {
    "2": 2,
    "3": 3,
    "0": 2,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 178
   },
   "max_capacities": {
    "capacity-2": 178
   },
   "reserved": {
    "capacity-2": 66
   },
   "requested_by_type": {
    "capacity-2": 66
   },
   "inventory": {
    "12": 5,
    "13": 10,
    "10": 4,
    "11": 3
   },
   "inbound": {
    "11": 12
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 167
   },
   "max_capacities": {
    "capacity-2": 167
   },
   "reserved": {
    "capacity-2": 30
   },
   "requested_by_type": {
    "capacity-2": 60
   },
   "inventory": {
    "10": 11,
    "11": 4,
    "12": 4,
    "13": 11
   },
   "inbound": {
    "11": 13
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 191
   },
   "max_capacities": {
    "capacity-2": 191
   },
   "reserved": {
    "capacity-2": 33
   },
   "requested_by_type": {
    "capacity-2": 54
   },
   "inventory": {
    "10": 6,
    "12": 3,
    "11": 3,
    "13": 2
   },
   "inbound": {
    "11": 15,
    "13": 3
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 23,
   "qty": 10,
   "accepted": 10,
   "completed": 3,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 3,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 18.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 1,
   "accepted": 1,
   "completed": 0,
   "failed": 0,
   "tokens": 2.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 3,
   "accepted": 3,
   "completed": 2,
   "failed": 0,
   "tokens": 6.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 10,
   "accepted": 10,
   "completed": 6,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 0,
   "tokens": 54.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 26.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 28.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 13,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 5,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 13,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "23": [
   0,
   0.0
  ],
  "21": [
   13,
   633.0
  ],
  "20": [
   13,
   633.0
  ],
  "22": [
   5,
   633.0
  ]
 }
}
```

### cycle 40 — probe_regret 34.0, margin 6.0
- candidates: even_split=47.0, blend_0.75=39.0, blend_0.5=67.0, blend_0.25=73.0, proportional=59.0
- teacher pick: `blend_0.25`; probe pick: `blend_0.75`; champion: 81.0 (regret -8.0)

```json
{
 "cycle": 40,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 29
   },
   "max_capacities": {
    "capacity-0": 53
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 10
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 105
   },
   "max_capacities": {
    "capacity-0": 110
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 20
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 25
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {},
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 133
   },
   "max_capacities": {
    "capacity-1": 141
   },
   "reserved": {
    "capacity-1": 55
   },
   "requested_by_type": {
    "capacity-1": 55
   },
   "inventory": {
    "3": 2,
    "0": 1,
    "1": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 70
   },
   "max_capacities": {
    "capacity-1": 118
   },
   "reserved": {
    "capacity-1": 52
   },
   "requested_by_type": {
    "capacity-1": 50
   },
   "inventory": {
    "2": 3,
    "3": 1,
    "0": 3,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 91
   },
   "max_capacities": {
    "capacity-1": 139
   },
   "reserved": {
    "capacity-1": 75
   },
   "requested_by_type": {
    "capacity-1": 75
   },
   "inventory": {
    "2": 3,
    "3": 3,
    "0": 3,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 143
   },
   "max_capacities": {
    "capacity-2": 178
   },
   "reserved": {
    "capacity-2": 9
   },
   "requested_by_type": {
    "capacity-2": 24
   },
   "inventory": {
    "12": 6,
    "13": 1,
    "10": 6,
    "11": 6
   },
   "inbound": {
    "13": 9
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 130
   },
   "max_capacities": {
    "capacity-2": 167
   },
   "reserved": {
    "capacity-2": 21
   },
   "requested_by_type": {
    "capacity-2": 21
   },
   "inventory": {
    "10": 12,
    "11": 6,
    "12": 8,
    "13": 1
   },
   "inbound": {
    "13": 13
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 127
   },
   "max_capacities": {
    "capacity-2": 191
   },
   "reserved": {
    "capacity-2": 12
   },
   "requested_by_type": {
    "capacity-2": 15
   },
   "inventory": {
    "10": 7,
    "11": 7
   },
   "inbound": {
    "12": 9,
    "13": 14
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 22,
   "qty": 6,
   "accepted": 6,
   "completed": 2,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 12,
   "qty": 9,
   "accepted": 9,
   "completed": 6,
   "failed": 0,
   "tokens": 18.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 23,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 7,
   "accepted": 7,
   "completed": 5,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 13,
   "qty": 9,
   "accepted": 9,
   "completed": 2,
   "failed": 0,
   "tokens": 54.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 26.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 28.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 12,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 6,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 8,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "22": [
   6,
   633.0
  ],
  "23": [
   8,
   633.0
  ],
  "20": [
   12,
   633.0
  ]
 }
}
```

### cycle 22 — probe_regret 29.0, margin 22.0
- candidates: even_split=49.0, blend_0.75=78.0, blend_0.5=56.0, blend_0.25=56.0, proportional=56.0
- teacher pick: `blend_0.75`; probe pick: `even_split`; champion: 81.0 (regret -3.0)

```json
{
 "cycle": 22,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 45
   },
   "max_capacities": {
    "capacity-0": 53
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 14
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 95
   },
   "max_capacities": {
    "capacity-0": 110
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 16
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 25
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 8
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 106
   },
   "max_capacities": {
    "capacity-1": 141
   },
   "reserved": {
    "capacity-1": 52
   },
   "requested_by_type": {
    "capacity-1": 17
   },
   "inventory": {
    "3": 7,
    "2": 1,
    "0": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 64
   },
   "max_capacities": {
    "capacity-1": 118
   },
   "reserved": {
    "capacity-1": 37
   },
   "requested_by_type": {
    "capacity-1": 12
   },
   "inventory": {
    "2": 1,
    "3": 5,
    "0": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 99
   },
   "max_capacities": {
    "capacity-1": 139
   },
   "reserved": {
    "capacity-1": 44
   },
   "requested_by_type": {
    "capacity-1": 19
   },
   "inventory": {
    "2": 3,
    "3": 5,
    "0": 3
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 156
   },
   "max_capacities": {
    "capacity-2": 178
   },
   "reserved": {
    "capacity-2": 12
   },
   "requested_by_type": {
    "capacity-2": 42
   },
   "inventory": {
    "12": 5,
    "13": 1,
    "10": 4,
    "11": 3
   },
   "inbound": {
    "13": 9,
    "11": 3
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 145
   },
   "max_capacities": {
    "capacity-2": 167
   },
   "reserved": {
    "capacity-2": 30
   },
   "requested_by_type": {
    "capacity-2": 27
   },
   "inventory": {
    "10": 13,
    "11": 10,
    "12": 4
   },
   "inbound": {
    "13": 11
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 163
   },
   "max_capacities": {
    "capacity-2": 191
   },
   "reserved": {
    "capacity-2": 45
   },
   "requested_by_type": {
    "capacity-2": 27
   },
   "inventory": {
    "10": 8,
    "11": 9,
    "13": 2
   },
   "inbound": {
    "11": 1,
    "13": 12
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 23,
   "qty": 11,
   "accepted": 11,
   "completed": 3,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 5,
   "accepted": 5,
   "completed": 3,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 13,
   "qty": 7,
   "accepted": 7,
   "completed": 5,
   "failed": 0,
   "tokens": 42.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 16.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 0,
   "tokens": 18.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 23,
   "qty": 10,
   "accepted": 10,
   "completed": 3,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 10,
   "accepted": 10,
   "completed": 4,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 3,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 18.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 2,
   "accepted": 2,
   "completed": 1,
   "failed": 0,
   "tokens": 12.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 3,
   "accepted": 3,
   "completed": 1,
   "failed": 0,
   "tokens": 6.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 1,
   "accepted": 1,
   "completed": 0,
   "failed": 0,
   "tokens": 2.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 3,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 6.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 10,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 14,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 6,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "23": [
   0,
   0.0
  ],
  "22": [
   0,
   0.0
  ],
  "21": [
   24,
   1266.0
  ],
  "20": [
   6,
   633.0
  ]
 }
}
```

### cycle 68 — probe_regret 24.0, margin 2.0
- candidates: even_split=77.0, blend_0.75=75.0, blend_0.5=75.0, blend_0.25=63.0, proportional=53.0
- teacher pick: `even_split`; probe pick: `proportional`; champion: 79.0 (regret -2.0)

```json
{
 "cycle": 68,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 29
   },
   "max_capacities": {
    "capacity-0": 53
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 26
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 65
   },
   "max_capacities": {
    "capacity-0": 110
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 35
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 25
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 17
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 80
   },
   "max_capacities": {
    "capacity-1": 141
   },
   "reserved": {
    "capacity-1": 16
   },
   "requested_by_type": {},
   "inventory": {
    "3": 5,
    "2": 3,
    "0": 5,
    "1": 2
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 70
   },
   "max_capacities": {
    "capacity-1": 118
   },
   "reserved": {
    "capacity-1": 20
   },
   "requested_by_type": {
    "capacity-1": 6
   },
   "inventory": {
    "2": 3,
    "3": 3,
    "1": 7
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 112
   },
   "max_capacities": {
    "capacity-1": 139
   },
   "reserved": {
    "capacity-1": 16
   },
   "requested_by_type": {},
   "inventory": {
    "2": 2,
    "3": 3,
    "0": 5,
    "1": 8
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 181
   },
   "max_capacities": {
    "capacity-2": 227
   },
   "reserved": {
    "capacity-2": 24
   },
   "requested_by_type": {
    "capacity-2": 36
   },
   "inventory": {
    "12": 8,
    "13": 6,
    "10": 10,
    "11": 4
   },
   "inbound": {
    "11": 8
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 163
   },
   "max_capacities": {
    "capacity-2": 229
   },
   "reserved": {
    "capacity-2": 24
   },
   "requested_by_type": {
    "capacity-2": 36
   },
   "inventory": {
    "10": 13,
    "12": 11,
    "13": 3
   },
   "inbound": {
    "11": 12
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 188
   },
   "max_capacities": {
    "capacity-2": 222
   },
   "reserved": {
    "capacity-2": 30
   },
   "requested_by_type": {
    "capacity-2": 24
   },
   "inventory": {
    "12": 3,
    "11": 1,
    "13": 3
   },
   "inbound": {
    "11": 12
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 20,
   "qty": 5,
   "accepted": 5,
   "completed": 3,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 8,
   "accepted": 8,
   "completed": 6,
   "failed": 0,
   "tokens": 48.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 22.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 22.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 11,
   "accepted": 11,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 1,
   "accepted": 1,
   "completed": 0,
   "failed": 0,
   "tokens": 2.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 1,
   "accepted": 1,
   "completed": 0,
   "failed": 0,
   "tokens": 2.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 8,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 14,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 10,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "20": [
   8,
   633.0
  ],
  "21": [
   14,
   633.0
  ],
  "22": [
   10,
   633.0
  ]
 }
}
```

## Top-regret held-out cycles — mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45 (seed 47)

### cycle 37 — probe_regret 52.0, margin 0.0
- candidates: even_split=45.0, blend_0.75=97.0, blend_0.5=97.0, blend_0.25=97.0, proportional=97.0
- teacher pick: `blend_0.75`; probe pick: `even_split`; champion: 79.0 (regret 18.0)

```json
{
 "cycle": 37,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 81
   },
   "max_capacities": {
    "capacity-0": 113
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 30
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 20
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 21
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 89
   },
   "max_capacities": {
    "capacity-0": 89
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 24
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 104
   },
   "max_capacities": {
    "capacity-1": 144
   },
   "reserved": {
    "capacity-1": 40
   },
   "requested_by_type": {
    "capacity-1": 36
   },
   "inventory": {
    "2": 1,
    "0": 10,
    "1": 4,
    "3": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 107
   },
   "max_capacities": {
    "capacity-1": 119
   },
   "reserved": {
    "capacity-1": 42
   },
   "requested_by_type": {
    "capacity-1": 34
   },
   "inventory": {
    "0": 6,
    "2": 2
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 80
   },
   "max_capacities": {
    "capacity-1": 104
   },
   "reserved": {
    "capacity-1": 34
   },
   "requested_by_type": {
    "capacity-1": 30
   },
   "inventory": {
    "0": 8
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 150
   },
   "max_capacities": {
    "capacity-2": 176
   },
   "reserved": {
    "capacity-2": 32
   },
   "requested_by_type": {
    "capacity-2": 24
   },
   "inventory": {
    "10": 9,
    "12": 8,
    "13": 1,
    "11": 10
   },
   "inbound": {
    "10": 9
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 116
   },
   "max_capacities": {
    "capacity-2": 149
   },
   "reserved": {
    "capacity-2": 22
   },
   "requested_by_type": {
    "capacity-2": 20
   },
   "inventory": {
    "11": 3
   },
   "inbound": {
    "10": 24,
    "12": 9
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 126
   },
   "max_capacities": {
    "capacity-2": 150
   },
   "reserved": {
    "capacity-2": 22
   },
   "requested_by_type": {
    "capacity-2": 20
   },
   "inventory": {
    "11": 10
   },
   "inbound": {
    "10": 19
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 20,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 10,
   "accepted": 10,
   "completed": 6,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 24.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 48.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 0,
   "tokens": 45.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 24.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 9,
   "accepted": 9,
   "completed": 6,
   "failed": 0,
   "tokens": 18.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 7,
   "accepted": 7,
   "completed": 0,
   "failed": 0,
   "tokens": 28.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 7,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 9,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 1,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "20": [
   0,
   0.0
  ],
  "21": [
   7,
   633.0
  ],
  "22": [
   9,
   633.0
  ],
  "23": [
   1,
   633.0
  ]
 }
}
```

### cycle 32 — probe_regret 44.0, margin 23.0
- candidates: even_split=84.0, blend_0.75=61.0, blend_0.5=61.0, blend_0.25=40.0, proportional=40.0
- teacher pick: `even_split`; probe pick: `proportional`; champion: 65.0 (regret 19.0)

```json
{
 "cycle": 32,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 101
   },
   "max_capacities": {
    "capacity-0": 113
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 21
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 30
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 12
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 77
   },
   "max_capacities": {
    "capacity-0": 89
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 15
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 129
   },
   "max_capacities": {
    "capacity-1": 144
   },
   "reserved": {
    "capacity-1": 66
   },
   "requested_by_type": {
    "capacity-1": 42
   },
   "inventory": {
    "2": 10,
    "0": 6,
    "1": 2,
    "3": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 87
   },
   "max_capacities": {
    "capacity-1": 119
   },
   "reserved": {
    "capacity-1": 42
   },
   "requested_by_type": {
    "capacity-1": 28
   },
   "inventory": {
    "0": 6,
    "2": 4,
    "1": 3,
    "3": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 62
   },
   "max_capacities": {
    "capacity-1": 104
   },
   "reserved": {},
   "requested_by_type": {},
   "inventory": {
    "0": 6,
    "1": 3,
    "2": 3,
    "3": 2
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 133
   },
   "max_capacities": {
    "capacity-2": 176
   },
   "reserved": {
    "capacity-2": 28
   },
   "requested_by_type": {
    "capacity-2": 16
   },
   "inventory": {
    "10": 5,
    "12": 17,
    "11": 10
   },
   "inbound": {
    "13": 18
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 67
   },
   "max_capacities": {
    "capacity-2": 107
   },
   "reserved": {
    "capacity-2": 24
   },
   "requested_by_type": {
    "capacity-2": 10
   },
   "inventory": {
    "10": 3,
    "11": 10
   },
   "inbound": {
    "12": 4,
    "13": 17
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 85
   },
   "max_capacities": {
    "capacity-2": 115
   },
   "reserved": {
    "capacity-2": 24
   },
   "requested_by_type": {
    "capacity-2": 14
   },
   "inventory": {
    "10": 3,
    "11": 6,
    "12": 3
   },
   "inbound": {
    "13": 19
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 22,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 4,
   "accepted": 4,
   "completed": 2,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 13,
   "qty": 9,
   "accepted": 9,
   "completed": 3,
   "failed": 0,
   "tokens": 45.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 4,
   "accepted": 4,
   "completed": 1,
   "failed": 0,
   "tokens": 8.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 5,
   "accepted": 5,
   "completed": 1,
   "failed": 0,
   "tokens": 10.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 7,
   "accepted": 7,
   "completed": 2,
   "failed": 0,
   "tokens": 28.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 22,
   "qty": 5,
   "accepted": 5,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 2,
   "accepted": 2,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 13,
   "qty": 9,
   "accepted": 9,
   "completed": 0,
   "failed": 0,
   "tokens": 45.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 24.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 48.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 21,
   "qty": 6,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 2,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 10,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "22": [
   2,
   633.0
  ],
  "23": [
   0,
   0.0
  ],
  "21": [
   6,
   633.0
  ],
  "20": [
   10,
   633.0
  ]
 }
}
```

### cycle 45 — probe_regret 37.0, margin 13.0
- candidates: even_split=37.0, blend_0.75=77.0, blend_0.5=64.0, blend_0.25=64.0, proportional=40.0
- teacher pick: `blend_0.75`; probe pick: `proportional`; champion: 59.0 (regret 18.0)

```json
{
 "cycle": 45,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 101
   },
   "max_capacities": {
    "capacity-0": 113
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 30
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 21
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 3
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 83
   },
   "max_capacities": {
    "capacity-0": 89
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 15
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 129
   },
   "max_capacities": {
    "capacity-1": 144
   },
   "reserved": {
    "capacity-1": 114
   },
   "requested_by_type": {
    "capacity-1": 92
   },
   "inventory": {
    "0": 1,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 109
   },
   "max_capacities": {
    "capacity-1": 119
   },
   "reserved": {
    "capacity-1": 76
   },
   "requested_by_type": {
    "capacity-1": 84
   },
   "inventory": {
    "0": 1,
    "2": 2,
    "3": 4
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 80
   },
   "max_capacities": {
    "capacity-1": 104
   },
   "reserved": {
    "capacity-1": 42
   },
   "requested_by_type": {
    "capacity-1": 46
   },
   "inventory": {
    "1": 1,
    "3": 5
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 149
   },
   "max_capacities": {
    "capacity-2": 176
   },
   "reserved": {
    "capacity-2": 6
   },
   "requested_by_type": {
    "capacity-2": 30
   },
   "inventory": {
    "10": 6,
    "12": 5,
    "13": 1
   },
   "inbound": {
    "13": 4,
    "11": 9,
    "12": 4
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 112
   },
   "max_capacities": {
    "capacity-2": 149
   },
   "reserved": {
    "capacity-2": 14
   },
   "requested_by_type": {
    "capacity-2": 30
   },
   "inventory": {
    "12": 1
   },
   "inbound": {
    "13": 5,
    "10": 8,
    "12": 12,
    "11": 10
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 112
   },
   "max_capacities": {
    "capacity-2": 150
   },
   "reserved": {
    "capacity-2": 52
   },
   "requested_by_type": {
    "capacity-2": 44
   },
   "inventory": {
    "10": 8,
    "11": 4
   },
   "inbound": {
    "13": 6,
    "12": 13,
    "11": 10
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 20,
   "qty": 9,
   "accepted": 9,
   "completed": 6,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 8,
   "accepted": 8,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 13,
   "qty": 4,
   "accepted": 4,
   "completed": 0,
   "failed": 0,
   "tokens": 20.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 3,
   "accepted": 3,
   "completed": 2,
   "failed": 0,
   "tokens": 6.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 2,
   "accepted": 2,
   "completed": 1,
   "failed": 0,
   "tokens": 4.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 5,
   "accepted": 5,
   "completed": 0,
   "failed": 0,
   "tokens": 10.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 3,
   "accepted": 3,
   "completed": 2,
   "failed": 0,
   "tokens": 12.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 6,
   "accepted": 6,
   "completed": 0,
   "failed": 0,
   "tokens": 24.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 4,
   "accepted": 4,
   "completed": 3,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 13,
   "accepted": 13,
   "completed": 4,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 14,
   "accepted": 14,
   "completed": 8,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 9,
   "accepted": 9,
   "completed": 5,
   "failed": 0,
   "tokens": 45.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 4,
   "accepted": 4,
   "completed": 2,
   "failed": 0,
   "tokens": 20.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 5,
   "accepted": 5,
   "completed": 1,
   "failed": 0,
   "tokens": 10.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 20.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 20.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 40.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 40.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 23,
   "qty": 1,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 13,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 3,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "20": [
   13,
   633.0
  ],
  "23": [
   1,
   633.0
  ],
  "22": [
   0,
   0.0
  ],
  "21": [
   3,
   633.0
  ]
 }
}
```

### cycle 20 — probe_regret 28.0, margin 0.0
- candidates: even_split=20.0, blend_0.75=48.0, blend_0.5=48.0, blend_0.25=48.0, proportional=48.0
- teacher pick: `blend_0.75`; probe pick: `even_split`; champion: 61.0 (regret -13.0)

```json
{
 "cycle": 20,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 99
   },
   "max_capacities": {
    "capacity-0": 113
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 30
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 0
   },
   "max_capacities": {
    "capacity-0": 1
   },
   "reserved": {},
   "requested_by_type": {},
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 65
   },
   "max_capacities": {
    "capacity-0": 89
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 18
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 126
   },
   "max_capacities": {
    "capacity-1": 144
   },
   "reserved": {
    "capacity-1": 60
   },
   "requested_by_type": {
    "capacity-1": 50
   },
   "inventory": {
    "0": 5
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 109
   },
   "max_capacities": {
    "capacity-1": 119
   },
   "reserved": {
    "capacity-1": 46
   },
   "requested_by_type": {
    "capacity-1": 34
   },
   "inventory": {
    "0": 6
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 84
   },
   "max_capacities": {
    "capacity-1": 104
   },
   "reserved": {
    "capacity-1": 48
   },
   "requested_by_type": {
    "capacity-1": 46
   },
   "inventory": {
    "0": 3,
    "1": 1
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 158
   },
   "max_capacities": {
    "capacity-2": 176
   },
   "reserved": {
    "capacity-2": 42
   },
   "requested_by_type": {
    "capacity-2": 32
   },
   "inventory": {
    "10": 12,
    "12": 3,
    "13": 3
   },
   "inbound": {
    "11": 7,
    "10": 6
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 107
   },
   "max_capacities": {
    "capacity-2": 107
   },
   "reserved": {
    "capacity-2": 27
   },
   "requested_by_type": {
    "capacity-2": 25
   },
   "inventory": {
    "10": 1,
    "13": 3
   },
   "inbound": {
    "10": 12,
    "11": 12
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 115
   },
   "max_capacities": {
    "capacity-2": 115
   },
   "reserved": {
    "capacity-2": 35
   },
   "requested_by_type": {
    "capacity-2": 25
   },
   "inventory": {
    "11": 2,
    "13": 3
   },
   "inbound": {
    "11": 6,
    "10": 17
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 20,
   "qty": 13,
   "accepted": 13,
   "completed": 4,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 3,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 2,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 10,
   "accepted": 10,
   "completed": 4,
   "failed": 0,
   "tokens": 20.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 12,
   "accepted": 12,
   "completed": 4,
   "failed": 0,
   "tokens": 48.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 1,
   "accepted": 1,
   "completed": 0,
   "failed": 0,
   "tokens": 4.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 22,
   "qty": 3,
   "accepted": 3,
   "completed": 1,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 20,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 10,
   "qty": 6,
   "accepted": 6,
   "completed": 0,
   "failed": 0,
   "tokens": 30.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 7,
   "accepted": 7,
   "completed": 0,
   "failed": 0,
   "tokens": 35.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 2,
   "accepted": 2,
   "completed": 0,
   "failed": 0,
   "tokens": 4.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 12,
   "accepted": 12,
   "completed": 0,
   "failed": 0,
   "tokens": 24.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 10,
   "qty": 5,
   "accepted": 5,
   "completed": 0,
   "failed": 0,
   "tokens": 20.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 5,
   "accepted": 5,
   "completed": 0,
   "failed": 0,
   "tokens": 20.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 20,
   "qty": 9,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 13,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 12,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "20": [
   9,
   633.0
  ],
  "23": [
   12,
   633.0
  ],
  "22": [
   13,
   633.0
  ],
  "21": [
   0,
   0.0
  ]
 }
}
```

### cycle 51 — probe_regret 26.0, margin 12.0
- candidates: even_split=80.0, blend_0.75=68.0, blend_0.5=68.0, blend_0.25=60.0, proportional=54.0
- teacher pick: `even_split`; probe pick: `proportional`; champion: 60.0 (regret 20.0)

```json
{
 "cycle": 51,
 "producers": {
  "p0_0": {
   "capacities": {
    "capacity-0": 77
   },
   "max_capacities": {
    "capacity-0": 113
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 27
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_1": {
   "capacities": {
    "capacity-0": 27
   },
   "max_capacities": {
    "capacity-0": 35
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 18
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p0_2": {
   "capacities": {
    "capacity-0": 89
   },
   "max_capacities": {
    "capacity-0": 89
   },
   "reserved": {},
   "requested_by_type": {
    "capacity-0": 15
   },
   "inventory": {},
   "inbound": {},
   "investing": []
  },
  "p1_0": {
   "capacities": {
    "capacity-1": 99
   },
   "max_capacities": {
    "capacity-1": 144
   },
   "reserved": {
    "capacity-1": 68
   },
   "requested_by_type": {
    "capacity-1": 86
   },
   "inventory": {
    "2": 3,
    "0": 5,
    "1": 3
   },
   "inbound": {},
   "investing": []
  },
  "p1_1": {
   "capacities": {
    "capacity-1": 73
   },
   "max_capacities": {
    "capacity-1": 119
   },
   "reserved": {
    "capacity-1": 30
   },
   "requested_by_type": {
    "capacity-1": 78
   },
   "inventory": {
    "1": 3,
    "3": 1
   },
   "inbound": {},
   "investing": []
  },
  "p1_2": {
   "capacities": {
    "capacity-1": 92
   },
   "max_capacities": {
    "capacity-1": 104
   },
   "reserved": {
    "capacity-1": 84
   },
   "requested_by_type": {
    "capacity-1": 66
   },
   "inventory": {
    "1": 3,
    "2": 3,
    "3": 3
   },
   "inbound": {},
   "investing": []
  },
  "p2_0": {
   "capacities": {
    "capacity-2": 113
   },
   "max_capacities": {
    "capacity-2": 176
   },
   "reserved": {
    "capacity-2": 45
   },
   "requested_by_type": {
    "capacity-2": 45
   },
   "inventory": {
    "10": 11,
    "12": 3,
    "11": 4
   },
   "inbound": {
    "11": 9,
    "13": 12,
    "12": 9
   },
   "investing": []
  },
  "p2_1": {
   "capacities": {
    "capacity-2": 130
   },
   "max_capacities": {
    "capacity-2": 149
   },
   "reserved": {
    "capacity-2": 18
   },
   "requested_by_type": {
    "capacity-2": 38
   },
   "inventory": {
    "10": 2,
    "11": 2,
    "12": 3
   },
   "inbound": {
    "11": 7,
    "13": 13,
    "12": 11
   },
   "investing": []
  },
  "p2_2": {
   "capacities": {
    "capacity-2": 114
   },
   "max_capacities": {
    "capacity-2": 150
   },
   "reserved": {
    "capacity-2": 16
   },
   "requested_by_type": {
    "capacity-2": 36
   },
   "inventory": {
    "10": 13,
    "11": 2,
    "12": 3
   },
   "inbound": {
    "11": 4,
    "13": 13,
    "12": 11
   },
   "investing": []
  }
 },
 "orders": [
  {
   "product": 11,
   "qty": 3,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 15.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 3,
   "accepted": 3,
   "completed": 2,
   "failed": 0,
   "tokens": 15.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 3,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 6.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 3,
   "accepted": 3,
   "completed": 2,
   "failed": 0,
   "tokens": 6.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 2,
   "accepted": 2,
   "completed": 0,
   "failed": 0,
   "tokens": 8.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 3,
   "accepted": 3,
   "completed": 2,
   "failed": 0,
   "tokens": 12.0,
   "age": 2,
   "type": "replenishment"
  },
  {
   "product": 22,
   "qty": 14,
   "accepted": 14,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 23,
   "qty": 13,
   "accepted": 13,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 13,
   "accepted": 13,
   "completed": 8,
   "failed": 0,
   "tokens": 633.0,
   "age": 1,
   "type": "consumer"
  },
  {
   "product": 11,
   "qty": 6,
   "accepted": 6,
   "completed": 0,
   "failed": 0,
   "tokens": 30.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 9,
   "accepted": 9,
   "completed": 3,
   "failed": 0,
   "tokens": 45.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 9,
   "accepted": 9,
   "completed": 6,
   "failed": 0,
   "tokens": 45.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 4,
   "accepted": 4,
   "completed": 0,
   "failed": 0,
   "tokens": 8.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 11,
   "accepted": 3,
   "completed": 0,
   "failed": 0,
   "tokens": 22.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 10,
   "accepted": 10,
   "completed": 0,
   "failed": 0,
   "tokens": 20.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 11,
   "qty": 2,
   "accepted": 2,
   "completed": 0,
   "failed": 0,
   "tokens": 8.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 12,
   "qty": 11,
   "accepted": 7,
   "completed": 0,
   "failed": 0,
   "tokens": 44.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 13,
   "qty": 10,
   "accepted": 10,
   "completed": 7,
   "failed": 0,
   "tokens": 40.0,
   "age": 1,
   "type": "replenishment"
  },
  {
   "product": 22,
   "qty": 7,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 21,
   "qty": 12,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  },
  {
   "product": 22,
   "qty": 11,
   "accepted": 0,
   "completed": 0,
   "failed": 0,
   "tokens": 633.0,
   "age": 0,
   "type": "consumer"
  }
 ],
 "open_demand": {
  "22": [
   18,
   1266.0
  ],
  "23": [
   0,
   0.0
  ],
  "21": [
   12,
   633.0
  ]
 }
}
```

## Conclusions

**1. The probe's misses are EXPENSIVE — the D1 wall is real in units, not tie noise.**
Held-out probe_regret_frac = 0.98 (s29) / 0.86 (s77) / 0.56 (s45): the state→choice
classifier throws away most of the teacher's edge over even_split. Dominant failure mode
(winner-set analysis on the trace tables): picking `even_split` on cycles where it is NOT
in the tied-winner set — 9–14 cycles/task costing 154–166 units each task. Note margin==0
cycles can carry huge regret (s45 cycle 37: 4-way tie at 97 vs even_split 45, regret 52)
— "decisive = margin>0" was the wrong noise split; the real target at tie cycles is a
winner SET, not an argmax.

**2. The champion is far better per-decision at teacher states than the probe — and beats
the ENTIRE vocabulary at ~40% of states.** champ_regret_frac = 0.52 / 0.20 / 0.17;
n_champ_better = 40 / 47 / 41 of 100. Two implications: (a) the champion's 0-unit episode
collapses on s77/s45 are NOT uniformly-bad routing — per-decision it is decent at
teacher-visited states, so the collapse is trajectory-level compounding (drift into states
where its choices degrade); (b) the 5-entry vocabulary is NOT a cover of good behavior —
any deploy-time search should include the champion as a 6th candidate (the union dominates
both by construction).

**3. Feature inference from the lossless snapshots came up EMPTY at the system level.**
Hand-built aggregates computed from the complete state — min per-type capacity ratio,
capacity imbalance across producers, inbound in flight, inventory, order count, order age,
open demand, episode phase — are FLAT between even-wins and concentration-wins cycles
(e.g. s29 min_type_ratio 0.64 vs 0.63, imbalance 0.54 vs 0.55). The evidence-first search
for "the missing metric" found none at this granularity; anything predictive would have to
be relational (backlog product × eligible producer × per-type free capacity), which the
snapshot supports but no simple aggregate expresses.

**4. Temporal structure exists but is shallow.** Committed-choice persistence 59–63%
(majority floor 42–54%); previous pick still a winner 66–72% of cycles; a sticky
repeat-last-choice strategy still loses 24–50% of the edge; the even-vs-proportional edge
signal has lag-1 autocorrelation only 0.35–0.44 with 13–23 sign flips (≥10 units both
sides) per 100 cycles inside 5-cycle windows. History features would capture only the slow
component — they cannot close the gap alone.

**Branch implication:** the trace evidence weakens BOTH enrichment sub-hypotheses from the
A0c correction (lossy snapshot: natural aggregates of the lossless state don't separate;
temporal: sticky/persistence numbers above) and strengthens A2 deploy-time light search,
now with a concrete design improvement: candidate set = {5-entry blend vocabulary +
champion policy}, per-cycle argmax by short rollout. A0d (complete-state probe) remains
possible but low-prior — it would need relational features, not more inputs.
