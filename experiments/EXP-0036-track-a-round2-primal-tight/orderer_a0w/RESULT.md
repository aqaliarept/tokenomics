# EXP-0036 Stage A0w window/episode blend-policy distillation (HANDOVER Task 5)

A0f/A0g predicted a PER-CYCLE candidate table whose winner ordering did not transfer
across demand seeds. A0w changes the target time scale: the value of each CONSTANT
blend policy over an episode / rolling window, one row per window. The set head is
deployed with a conservative switch rule (default blend unless the predicted advantage
clears a threshold).

**Read order (definition of done):**

1. Compare `regret_frac` (learned model) against `const_regret_frac` (best single blend
   chosen on train). If a constant blend already closes most of the gap, the learned
   model must beat that constant — not just champion/even_split.
2. `decision_regret_frac` = the conservative deploy rule's held-out regret (what ships).
3. Episode vs window rows are separated; `deploy` rows are the rollout, not table error.
4. `closure_vs_search = (learned_cu - champ_cu)/(search_cu - champ_cu)` on the held-out
   seed; compare `learned_cu` to `default_blend_cu` and `best_const_cu`.

## Held-out regret per granularity

```
task                         grain      train_rf heldout_rf  const_rf decision_rf   best_const  switch
er_layer2-quantity_max14-s29 episode      0.0000     1.2800    1.2800      1.2800   blend_0.25  1.0000
er_layer2-quantity_max14-s29 window5      0.0000     1.0603    1.1353      1.0206   blend_0.25  0.7188
er_layer2-quantity_max14-s29 window10     0.0000     1.0102    1.1836      0.9665   blend_0.25  0.8681
er_layer2-quantity_max14-s29 window20     0.0000     0.9916    1.2530      0.9505   blend_0.25  0.8765
er_layer3-quantity_max14-s77 episode      0.0000     0.0000    0.0000      0.0000   blend_0.25  1.0000
er_layer3-quantity_max14-s77 window5      0.0000     0.5420    0.3354      0.5287   blend_0.25  0.6250
er_layer3-quantity_max14-s77 window10     0.0000     0.4579    0.1959      0.4684   blend_0.25  0.6593
er_layer3-quantity_max14-s77 window20     0.0000     0.2996    0.0953      0.2969   blend_0.25  0.6667
er_layer3-quantity_max14-s45 episode      0.0000     0.8889    0.8889      0.8889   blend_0.75  1.0000
er_layer3-quantity_max14-s45 window5      0.0000     1.0208    0.9492      0.9694    blend_0.5  0.6146
er_layer3-quantity_max14-s45 window10     0.0000     0.8754    0.9459      0.8790    blend_0.5  0.8681
er_layer3-quantity_max14-s45 window20     0.0000     0.8674    1.0081      0.9332    blend_0.5  0.3951
```

## Deployed rollout (held-out seed)

```
task                          learned  default best_const   champ  search  closure  dead
er_layer2-quantity_max14-s29      810      960        768    1004     986      n/a False
er_layer3-quantity_max14-s77      666      749        893       0    1019   0.6536 False
er_layer3-quantity_max14-s45      588      621        621       0     984   0.5976 False
```

## Per-row detail

```json
[
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "granularity": "episode",
    "window": "",
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 2,
    "n_heldout": 1,
    "train_regret_frac": 0.0,
    "regret_frac": 1.28,
    "top1": 0.0,
    "winner_hit": 0.0,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 1.28,
    "decision_regret_frac": 1.28,
    "decision_switch_rate": 1.0,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "granularity": "window5",
    "window": 5,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 192,
    "n_heldout": 96,
    "train_regret_frac": 0.0,
    "regret_frac": 1.0603,
    "top1": 0.1875,
    "winner_hit": 0.1875,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 1.1353,
    "decision_regret_frac": 1.0206,
    "decision_switch_rate": 0.7188,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "granularity": "window10",
    "window": 10,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 182,
    "n_heldout": 91,
    "train_regret_frac": 0.0,
    "regret_frac": 1.0102,
    "top1": 0.1868,
    "winner_hit": 0.1978,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 1.1836,
    "decision_regret_frac": 0.9665,
    "decision_switch_rate": 0.8681,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "granularity": "window20",
    "window": 20,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 162,
    "n_heldout": 81,
    "train_regret_frac": 0.0,
    "regret_frac": 0.9916,
    "top1": 0.1111,
    "winner_hit": 0.1111,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 1.253,
    "decision_regret_frac": 0.9505,
    "decision_switch_rate": 0.8765,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.3-producers_per_layer2-quantity_max14-s29",
    "world_seed": 29,
    "granularity": "deploy",
    "window": 10,
    "heldout_seed": 31,
    "learned_cu": 810,
    "default_blend_cu": 960,
    "best_const_cu": 768,
    "champ_cu": 1004,
    "search_cu": 986,
    "learned_dead": false,
    "default_blend": "blend_0.5",
    "best_const_blend": "blend_0.25",
    "closure_vs_search": null
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "granularity": "episode",
    "window": "",
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 2,
    "n_heldout": 1,
    "train_regret_frac": 0.0,
    "regret_frac": 0.0,
    "top1": 1.0,
    "winner_hit": 1.0,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 0.0,
    "decision_regret_frac": 0.0,
    "decision_switch_rate": 1.0,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "granularity": "window5",
    "window": 5,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 192,
    "n_heldout": 96,
    "train_regret_frac": 0.0,
    "regret_frac": 0.542,
    "top1": 0.2604,
    "winner_hit": 0.2812,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 0.3354,
    "decision_regret_frac": 0.5287,
    "decision_switch_rate": 0.625,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "granularity": "window10",
    "window": 10,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 182,
    "n_heldout": 91,
    "train_regret_frac": 0.0,
    "regret_frac": 0.4579,
    "top1": 0.2967,
    "winner_hit": 0.3297,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 0.1959,
    "decision_regret_frac": 0.4684,
    "decision_switch_rate": 0.6593,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "granularity": "window20",
    "window": 20,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 162,
    "n_heldout": 81,
    "train_regret_frac": 0.0,
    "regret_frac": 0.2996,
    "top1": 0.3457,
    "winner_hit": 0.3457,
    "best_const_blend": "blend_0.25",
    "const_regret_frac": 0.0953,
    "decision_regret_frac": 0.2969,
    "decision_switch_rate": 0.6667,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s77",
    "world_seed": 77,
    "granularity": "deploy",
    "window": 10,
    "heldout_seed": 79,
    "learned_cu": 666,
    "default_blend_cu": 749,
    "best_const_cu": 893,
    "champ_cu": 0,
    "search_cu": 1019,
    "learned_dead": false,
    "default_blend": "blend_0.5",
    "best_const_blend": "blend_0.25",
    "closure_vs_search": 0.6536
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "granularity": "episode",
    "window": "",
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 2,
    "n_heldout": 1,
    "train_regret_frac": 0.0,
    "regret_frac": 0.8889,
    "top1": 0.0,
    "winner_hit": 0.0,
    "best_const_blend": "blend_0.75",
    "const_regret_frac": 0.8889,
    "decision_regret_frac": 0.8889,
    "decision_switch_rate": 1.0,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "granularity": "window5",
    "window": 5,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 105,
    "n_heldout": 96,
    "train_regret_frac": 0.0,
    "regret_frac": 1.0208,
    "top1": 0.125,
    "winner_hit": 0.1458,
    "best_const_blend": "blend_0.5",
    "const_regret_frac": 0.9492,
    "decision_regret_frac": 0.9694,
    "decision_switch_rate": 0.6146,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "granularity": "window10",
    "window": 10,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 95,
    "n_heldout": 91,
    "train_regret_frac": 0.0,
    "regret_frac": 0.8754,
    "top1": 0.0769,
    "winner_hit": 0.0769,
    "best_const_blend": "blend_0.5",
    "const_regret_frac": 0.9459,
    "decision_regret_frac": 0.879,
    "decision_switch_rate": 0.8681,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "granularity": "window20",
    "window": 20,
    "context_dim": 460,
    "producer_dim": 108,
    "n_train": 81,
    "n_heldout": 81,
    "train_regret_frac": 0.0,
    "regret_frac": 0.8674,
    "top1": 0.1235,
    "winner_hit": 0.1235,
    "best_const_blend": "blend_0.5",
    "const_regret_frac": 1.0081,
    "decision_regret_frac": 0.9332,
    "decision_switch_rate": 0.3951,
    "default_blend": "blend_0.5",
    "threshold": 0.25
  },
  {
    "task_id": "mine-build_scale_config-capacity_scale0.6-depth3-primal_capacity_scale0.22-producers_per_layer3-quantity_max14-s45",
    "world_seed": 45,
    "granularity": "deploy",
    "window": 10,
    "heldout_seed": 47,
    "learned_cu": 588,
    "default_blend_cu": 621,
    "best_const_cu": 621,
    "champ_cu": 0,
    "search_cu": 984,
    "learned_dead": false,
    "default_blend": "blend_0.5",
    "best_const_blend": "blend_0.5",
    "closure_vs_search": 0.5976
  }
]
```

## Conclusions

**Verdict: STOP the trained-orderer path. The learned window/episode meta-orderer loses to
simply holding the default constant blend on all 3 tasks — switching destroys value.** The
deployed rollout is the verdict (not table regret), per the HANDOVER reading rule:

```
task   learned  default(blend_0.5)  best_const        beats both?
s29      810          960            768 (blend_0.25)   NO (loses to default)
s77      666          749            893 (blend_0.25)   NO (loses to both)
s45      588          621            621 (blend_0.5)    NO (loses to both)
```

Learned < default on 3/3 and < best_const on 2/3 (on s29 it edges best_const 810>768 but
still trails default 960). The rule says stop if it loses to default/best-constant on 2 of
3 — here it loses to the default on all three.

- **The default constant blend_0.5 is the thing to run, and it is strong.** It delivers
  960/749/621 units — near the champion on s29 (960 vs 1004) and far above the *dead*
  memorized champion on the tight s77/s45 (champion = 0 units, system death; the A0 finding
  that mined-tight tasks collapse the champion). A single fixed mid-blend is the deployable
  policy; a learned per-cycle switcher is not needed and actively hurts.
- **The switch rule destroys value.** The learned meta-orderer switches away from blend_0.5
  often (switch rate 0.40–0.88 on the deployed window-10 models) and every switch, on
  balance, costs units: s29 drops 960 → 810 purely from switching. `--threshold 0.25`
  makes it too eager; but the direction is wrong, not just the magnitude — even the
  best-constant blend beats the learned model on s77 (893 vs 666).
- **Table regret does not predict deployed value — exactly the trap the rule warns about.**
  Longer windows lower held-out `regret_frac` (s77: 0.54 → 0.30 from window5 → window20;
  s77 *episode* regret = 0.0), and yet the deployed s77 model still loses to both baselines.
  Where the table looked best (s77), the best *constant* blend's table regret was lower
  still (const_rf 0.34/0.20/0.10 < learned 0.54/0.46/0.30) — the learned model never clears
  the constant even on its own metric there.
- **`closure_vs_search` is a red herring here.** s77/s45 show closure 0.65/0.60, but that is
  recovery over a *dead* champion (champ = 0); the honest baselines (default/best-constant)
  beat the learned model outright, so the positive closure is an artifact of the collapsed
  champion, not evidence the learned orderer helps.

**What this closes (scoped).** Changing the target *time scale* — the last untested lever
from the A0f/A0g arc — does not rescue the trained orderer on these three primal-tight
tasks. Across the whole ladder (A0b per-cycle choice, A0c global-state, A0f flat feature
blocks, A0g producer-set head, A0w episode/window returns) the deployable learned orderer
never beats a fixed mid-blend on held-out demand. Consistent with the standing diagnosis:
the per-cycle/window search winner is seed-idiosyncratic noise around a stable mid-blend
preference; there is no stable supervised target for a learned orderer here at any tested
time scale.

**Scope / untested.** `--threshold 0.25` is a single normalized-output threshold — a
threshold sweep was NOT run (a promising A0w would have earned one first; it didn't). Also
untested: attention pooling, ranking/winner-set losses on the set head, per-producer
routing-history features (absent from the collected features), and the full 81-row archive
(this is the narrow s29/s77/s45 trio only). None of these change the deploy verdict on the
three decisive tasks.

**Recommended move.** Ship / compare against a fixed mid-blend (blend_0.5, or per-task
best-constant) as the orderer; drop the learned per-cycle/window orderer for this regime.
The remaining EXP-0036 headroom, if pursued, is a **deploy-time light search** over the
blend vocabulary (spec Stage A2) — a compute lever, not a learned policy — with the fixed
mid-blend as the zero-cost acceptance floor these baselines just established.
