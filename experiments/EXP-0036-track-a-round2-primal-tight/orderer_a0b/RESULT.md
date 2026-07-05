# EXP-0036 Stage A0b pre-A1b checks (blend value + choice learnability)

A0's `named_capture_frac` proved the orderer headroom is REACHABLE inside a named-policy
vocabulary via the lookahead oracle. It did NOT prove (a) that the alpha-blend family adds
anything over hard selection, nor (b) that the committed choice is predictable from the
stream observation — the two assumptions Stage A1b's head rests on. This run tests both.

**Read order:**

1. **Blend value (D2):** `blend_capture_frac` vs `named_capture_frac`. A clear gain => the
   continuous residual is blend-recoverable => keep the alpha head. ~equal => plain
   categorical selection suffices => drop alpha.
2. **Choice learnability (D1):** held-out `choice_top1` vs `choice_majority` (and the
   decisive-margin split). Near-majority => the categorical target inherits the same
   unpredictability wall (A1b would fail; fall back to A1 denoise / A2 deploy-time search).
   Well above majority => A1b is green-lit.

**Means:** named_capture_frac=0.7839, blend_capture_frac=0.9008, choice_top1=0.3269003319702157, choice_majority=0.5191927114636595, choice_top1_decisive=0.2741807046414352, choice_majority_decisive=0.4139432909595889.

## Per-task

```json
[
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "search_cu": 1070.67,
    "even_split_cu": 809,
    "search_named_cu": 1070.67,
    "search_blend_cu": 1062.67,
    "named_capture_frac": 1.0,
    "blend_capture_frac": 0.9694,
    "choice_top1": 0.26823529411764707,
    "choice_majority": 0.5294117647058824,
    "choice_top1_decisive": 0.17375231053604437,
    "choice_majority_decisive": 0.4269870609981516,
    "n_choice_decisive": 541,
    "n_choice_train": 2414,
    "n_choice_heldout": 1275,
    "choice_heldout": true,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ]
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "search_cu": 997.33,
    "even_split_cu": 513.67,
    "search_named_cu": 847.67,
    "search_blend_cu": 972.33,
    "named_capture_frac": 0.6906,
    "blend_capture_frac": 0.9483,
    "choice_top1": 0.3765468473777254,
    "choice_majority": 0.5615792575132587,
    "choice_top1_decisive": 0.3973105134474328,
    "choice_majority_decisive": 0.5293398533007335,
    "n_choice_decisive": 818,
    "n_choice_train": 3510,
    "n_choice_heldout": 1697,
    "choice_heldout": true,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ]
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "search_cu": 981.67,
    "even_split_cu": 199.67,
    "search_named_cu": 716.67,
    "search_blend_cu": 813.33,
    "named_capture_frac": 0.6611,
    "blend_capture_frac": 0.7847,
    "choice_top1": 0.33591885441527447,
    "choice_majority": 0.4665871121718377,
    "choice_top1_decisive": 0.2514792899408284,
    "choice_majority_decisive": 0.28550295857988167,
    "n_choice_decisive": 676,
    "n_choice_train": 3110,
    "n_choice_heldout": 1676,
    "choice_heldout": true,
    "vocab": [
      "even_split",
      "blend_0.75",
      "blend_0.5",
      "blend_0.25",
      "proportional"
    ]
  }
]
```

## Conclusions

**D2 — blend value: KEEP the blend vocabulary.** Mean capture rises 0.784 (named) →
0.901 (blend). On the two tasks where hard even/proportional switching left a residual,
the alpha-blends recover most of it: s77 0.69 → 0.95 (334 → 459 of 484 headroom units),
s45 0.66 → 0.78. On s29 (named already 1.00) blend 0.97 is trajectory-divergence noise,
not a loss. Caveat: s45 still leaves ~21% of its headroom outside the blend family — the
vocabulary is a large improvement, not a complete cover.

**D1 — choice learnability: FAIL, 3/3 tasks.** Held-out top-1 is BELOW the
majority-class floor everywhere: 0.27 vs 0.53 (s29), 0.38 vs 0.56 (s77), 0.34 vs 0.47
(s45); decisive-margin split is no better (0.17/0.43, 0.40/0.53, 0.25/0.29). Final train
CE per fit = 0.157 / 0.171 / 0.097 → the model fits its TRAINING episodes almost
perfectly and anti-transfers to the held-out seed. The wall is transfer, not fit.

**Verdict: Stage A1b as a pure per-stream obs → choice head is NOT green-lit.** The
representation is right (headroom lives in the blend vocabulary) but the committed
choice is not predictable from the local stream view across episode seeds. Two competing
causes remain, demanding different fixes (root-cause rule):

- (a) **missing global context** (H2): the choice is driven by system-level state the
  stream view doesn't carry → fix = compact global features feeding the choice head;
- (b) **choice noise**: lookahead-6 rollout margins are within noise, argmax is
  arbitrary even on "decisive" cycles → fix = value-aware soft labels / margin filtering
  (A1 denoise) or deploy-time light search (A2); obs enrichment would buy nothing.

**Discriminating check (next, cheap):** fit the SAME classifier on the full global state
vector on the same samples. Beats majority held-out → (a), enrichment justified. Still
fails → (b), no observation can predict the target; go soft-label / deploy-search.
