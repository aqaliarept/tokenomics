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
3. **Global-state probe (A0c, read only if D1 failed):** held-out `gchoice_top1` vs
   `gchoice_majority` — the same protocol from the FULL global state, one sample per
   searched cycle. Beats its majority floor => the D1 wall is missing CONTEXT (enrich the
   orderer obs with global features). Fails too => the committed choice is rollout NOISE;
   no observation can predict it — go soft labels / deploy-time search, not enrichment.

**Means:** named_capture_frac=0.7839, blend_capture_frac=0.9008, choice_top1=0.3400446576781035, choice_majority=0.5191927114636595, choice_top1_decisive=0.27742211830071123, choice_majority_decisive=0.4139432909595889, gchoice_top1=0.3333333333333333, gchoice_majority=0.4966666666666667, gchoice_top1_decisive=0.23737867239919225, gchoice_majority_decisive=0.41870236466679694.

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
    "choice_top1": 0.27058823529411763,
    "choice_majority": 0.5294117647058824,
    "choice_top1_decisive": 0.13123844731977818,
    "choice_majority_decisive": 0.4269870609981516,
    "n_choice_decisive": 541,
    "n_choice_train": 2414,
    "n_choice_heldout": 1275,
    "gchoice_top1": 0.31,
    "gchoice_majority": 0.53,
    "gchoice_top1_decisive": 0.19047619047619047,
    "gchoice_majority_decisive": 0.40476190476190477,
    "n_gchoice_train": 200,
    "n_gchoice_heldout": 100,
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
    "choice_top1": 0.41720683559222155,
    "choice_majority": 0.5615792575132587,
    "choice_top1_decisive": 0.43031784841075793,
    "choice_majority_decisive": 0.5293398533007335,
    "n_choice_decisive": 818,
    "n_choice_train": 3510,
    "n_choice_heldout": 1697,
    "gchoice_top1": 0.29,
    "gchoice_majority": 0.54,
    "gchoice_top1_decisive": 0.19607843137254902,
    "gchoice_majority_decisive": 0.5490196078431373,
    "n_gchoice_train": 200,
    "n_gchoice_heldout": 100,
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
    "choice_top1": 0.33233890214797135,
    "choice_majority": 0.4665871121718377,
    "choice_top1_decisive": 0.27071005917159763,
    "choice_majority_decisive": 0.28550295857988167,
    "n_choice_decisive": 676,
    "n_choice_train": 3110,
    "n_choice_heldout": 1676,
    "gchoice_top1": 0.4,
    "gchoice_majority": 0.42,
    "gchoice_top1_decisive": 0.32558139534883723,
    "gchoice_majority_decisive": 0.3023255813953488,
    "n_gchoice_train": 200,
    "n_gchoice_heldout": 100,
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

**A0c verdict: the D1 wall is NOT missing context — the committed choice is unpredictable
even from the FULL global state.** Held-out gchoice_top1 vs gchoice_majority:

- s29: 0.31 vs 0.53 (decisive 0.19 vs 0.40) — clear FAIL
- s77: 0.29 vs 0.54 (decisive 0.20 vs 0.55) — clear FAIL
- s45: 0.40 vs 0.42 (decisive 0.33 vs 0.30) — at the floor (within n=100 binomial noise)

2/3 clear fails, 1/3 statistical tie; nowhere does the global state beat its majority floor.
D2 and D1 reproduce A0b exactly (deterministic domain — same search_cu / captures; stream
choice_top1 varies only with torch init).

**Branch picked: (b) — stop chasing observation features.** Since even the centralized-critic
input (ADR-0004 global_state) cannot predict the search's committed choice, no obs enrichment
of the orderer can either. Interpretation: with n_samples=0 the search is deterministic, so
the choice IS a function of state — but it hinges on fine-grained 6-cycle-lookahead
consequences that do not compress into a pattern learnable from ~200 cycles/task. The lever
is the TARGET or the DEPLOYMENT, not the observation:

- **A2 deploy-time light search (primary):** the per-cycle vocabulary search realizes the
  headroom by construction (blend capture 0.90 mean); the vocabulary is only 5 candidates ×
  lookahead-6 rollouts per cycle, and EXP-0009 already showed short rollouts suffice.
- **A1 soft/advantage labels (secondary):** train on the per-candidate score table instead of
  the argmax. Caveat: A0c's decisive-cycle fails suggest even ranking candidates from state is
  hard — validate a critic's held-out ranking correlation before investing in a student.

Open caveat: per-sample margins were not persisted; "decisive" = margin > 0, which can still
sit within rollout-noise scale. A margin-distribution read would sharpen the noise story but
does not change the branch (both sub-branches assume the argmax is not a clean target).
