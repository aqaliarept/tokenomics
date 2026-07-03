# EXP-0032b orderer-headroom scan (probe re-selection)

EXP-0032 was an invalid probe (near-zero headroom). This scan measures `teacher_headroom
= champ_prod+search − champ_prod+champ_ord` over the whole archive under the FROZEN
EXP-0030 producer, and keeps tasks with `headroom_frac >= 2%`.

**Result:** NO task clears the 2% floor — the augmented producer already closed the mined orderer gap; the orderer lever is spent (STOP, don't escalate).

## Qualifying probe task_ids

_none_

## Per-task (headroom-sorted)

```json
[
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer3-s8",
    "world_seed": 8,
    "champ_cu": 488.67,
    "search_cu": 498,
    "teacher_headroom": 9.33,
    "headroom_frac": 0.0191,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer4-s9",
    "world_seed": 9,
    "champ_cu": 490.33,
    "search_cu": 497.67,
    "teacher_headroom": 7.33,
    "headroom_frac": 0.015,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer5-s18",
    "world_seed": 18,
    "champ_cu": 496,
    "search_cu": 500,
    "teacher_headroom": 4,
    "headroom_frac": 0.0081,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer5-s15",
    "world_seed": 15,
    "champ_cu": 546.33,
    "search_cu": 550.33,
    "teacher_headroom": 4.0,
    "headroom_frac": 0.0073,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer4-s4",
    "world_seed": 4,
    "champ_cu": 543,
    "search_cu": 547,
    "teacher_headroom": 4,
    "headroom_frac": 0.0074,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer4-s28",
    "world_seed": 28,
    "champ_cu": 492.33,
    "search_cu": 496,
    "teacher_headroom": 3.67,
    "headroom_frac": 0.0074,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer4-s6",
    "world_seed": 6,
    "champ_cu": 491.33,
    "search_cu": 495,
    "teacher_headroom": 3.67,
    "headroom_frac": 0.0075,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer5-s14",
    "world_seed": 14,
    "champ_cu": 493.67,
    "search_cu": 497,
    "teacher_headroom": 3.33,
    "headroom_frac": 0.0068,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer3-s6",
    "world_seed": 6,
    "champ_cu": 545,
    "search_cu": 547.67,
    "teacher_headroom": 2.67,
    "headroom_frac": 0.0049,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer5-s12",
    "world_seed": 12,
    "champ_cu": 446,
    "search_cu": 448.67,
    "teacher_headroom": 2.67,
    "headroom_frac": 0.006,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer4-s7",
    "world_seed": 7,
    "champ_cu": 496.67,
    "search_cu": 499,
    "teacher_headroom": 2.33,
    "headroom_frac": 0.0047,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer5-s26",
    "world_seed": 26,
    "champ_cu": 445.33,
    "search_cu": 447.33,
    "teacher_headroom": 2.0,
    "headroom_frac": 0.0045,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer3-s21",
    "world_seed": 21,
    "champ_cu": 492.33,
    "search_cu": 494,
    "teacher_headroom": 1.67,
    "headroom_frac": 0.0034,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer5-s22",
    "world_seed": 22,
    "champ_cu": 447.67,
    "search_cu": 448.67,
    "teacher_headroom": 1.0,
    "headroom_frac": 0.0022,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer3-s20",
    "world_seed": 20,
    "champ_cu": 542,
    "search_cu": 543,
    "teacher_headroom": 1,
    "headroom_frac": 0.0018,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer3-s7",
    "world_seed": 7,
    "champ_cu": 495.67,
    "search_cu": 496,
    "teacher_headroom": 0.33,
    "headroom_frac": 0.0007,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer3-s19",
    "world_seed": 19,
    "champ_cu": 498.67,
    "search_cu": 499,
    "teacher_headroom": 0.33,
    "headroom_frac": 0.0007,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer4-s27",
    "world_seed": 27,
    "champ_cu": 444.67,
    "search_cu": 444.67,
    "teacher_headroom": 0.0,
    "headroom_frac": 0.0,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer6-products_per_layer3-s7",
    "world_seed": 7,
    "champ_cu": 491.67,
    "search_cu": 490,
    "teacher_headroom": -1.67,
    "headroom_frac": -0.0034,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer4-s21",
    "world_seed": 21,
    "champ_cu": 497,
    "search_cu": 494.33,
    "teacher_headroom": -2.67,
    "headroom_frac": -0.0054,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer3-s12",
    "world_seed": 12,
    "champ_cu": 548.67,
    "search_cu": 545,
    "teacher_headroom": -3.67,
    "headroom_frac": -0.0067,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer3-s21",
    "world_seed": 21,
    "champ_cu": 499.33,
    "search_cu": 495,
    "teacher_headroom": -4.33,
    "headroom_frac": -0.0087,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer5-s15",
    "world_seed": 15,
    "champ_cu": 542.67,
    "search_cu": 535,
    "teacher_headroom": -7.67,
    "headroom_frac": -0.0141,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer6-products_per_layer4-s11",
    "world_seed": 11,
    "champ_cu": 493.33,
    "search_cu": 484.33,
    "teacher_headroom": -9.0,
    "headroom_frac": -0.0182,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer6-products_per_layer4-s20",
    "world_seed": 20,
    "champ_cu": 541.33,
    "search_cu": 522.33,
    "teacher_headroom": -19.0,
    "headroom_frac": -0.0351,
    "qualifies": false
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer5-products_per_layer4-s20",
    "world_seed": 20,
    "champ_cu": 545,
    "search_cu": 518.67,
    "teacher_headroom": -26.33,
    "headroom_frac": -0.0483,
    "qualifies": false
  }
]
```

## Conclusions

**STOP the orderer lever (v4 oracle relabel is spent).** Across all 26 archived tasks, under the
frozen EXP-0030 augmented producer, the champion orderer has **no recoverable headroom** vs the
per-cycle search:

- **0/26** tasks clear the 2% floor; the single best is **+9.33u (+1.91%)** — under the floor.
- The whole distribution sits in a tight band **[−4.83%, +1.91%]** around zero.
- **8/26** tasks have **negative** headroom — the champion orderer *beats* the search (down to
  −26.33u / −4.83%). The `mpc_search` teacher v4 wanted to distill is not even a ceiling for the
  orderer anymore; it's myopic (lookahead 6, fixed candidate set) and the distribution-trained
  champion orderer already exceeds it on a third of the archive.

**Why EXP-0032 read as a failure:** it wasn't an architecture limit — the 2 probe tasks simply had
~0 headroom (0.7% / 0.07%) under this producer. The re-scan confirms that's true archive-wide: the
augmented producer (EXP-0029/0030) already absorbed the orderer-side share of the mined gap.

**Closes the Stage-4 improvement thread.** EXP-0029 solved the producer side; EXP-0031 showed more
producer replay can't close the residual; EXP-0032/0032b show the orderer side has no residual
headroom either. The remaining mined gap vs the `entity_planner+mpc_search` bar is **not
recoverable** through either score-model channel with the current teachers — it's a producer/bar
mismatch (the bar's `entity_planner` producer), not a champion-orderer deficiency.

Next (out of this lever): if further improvement is wanted, it must come from a *different* form
than deterministic search-relabel — e.g. a stronger/less-myopic reference than `mpc_search`, or the
stochastic/elite-set orderer forms (`[[exp0006b-stochastic]]`) — but there is no evidence the
current bar leaves recoverable orderer headroom to justify it.
