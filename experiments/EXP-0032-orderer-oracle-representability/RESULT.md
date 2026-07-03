# EXP-0032 v4 orderer oracle-relabel representability

Decision gate for v4 (EXP-0032): can the portable orderer scorer head represent the
mining bar's per-cycle search routing? Distill a fresh orderer on each task's
oracle-relabel targets (champion producer drives collection), then compare on-task
champion / search-teacher / oracle / bar consumer units. Producer frozen at champion.

**Gate:** mean `recovered_vs_teacher` = -16.200 (>= ~0.5 => head represents the search).

**Verdict:** INCONCLUSIVE — the auto-gate ("architecture limit real") is confounded by a
near-zero-headroom task selection. See Conclusions. Do NOT green-light the full v4 run on this.

## Per-task

```json
[
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer5-s14",
    "world_seed": 14,
    "champ_cu": 493.67,
    "search_cu": 497,
    "oracle_cu": 489,
    "bar_cu": 497,
    "teacher_headroom": 3.33,
    "bar_headroom": 3.33,
    "recovered": -4.67,
    "recovered_vs_teacher": -1.4,
    "recovered_vs_bar": -1.4,
    "n_samples": 1267
  },
  {
    "task_id": "mine-build_scale_config-depth3-producers_per_layer4-products_per_layer3-s19",
    "world_seed": 19,
    "champ_cu": 498.67,
    "search_cu": 499,
    "oracle_cu": 488.33,
    "bar_cu": 499,
    "teacher_headroom": 0.33,
    "bar_headroom": 0.33,
    "recovered": -10.33,
    "recovered_vs_teacher": -31.0,
    "recovered_vs_bar": -31.0,
    "n_samples": 1150
  }
]
```

## Conclusions

**The probe is invalid: these tasks have no on-task orderer headroom under the frozen
EXP-0030 producer.** `teacher_headroom = search_cu − champ_cu` = **3.33u (0.7%)** and **0.33u
(0.07%)**. The champion orderer already ≈ the per-cycle search AND the bar. The archived
"high-gap" (5.67u / ~7u) was measured against a WEAKER champion (EXP-0028 used
`producer_scorer_broad`); the augmented EXP-0030 producer we froze here has already eaten that
gap. With ~0 headroom to recover, `recovered_vs_teacher` (denominator 0.33u) is noise: a −10u
miss reads as −31. **This run does not answer the representability question.**

**Two real signals survive the confound:**
1. **Oracle relabel underperforms the champion orderer** by −4.67u / −10.33u even where the
   masked-MSE converged (0.79 on s14; but s19 plateaus at 2.05 — the head fits the search
   targets loosely there). Distilling the committed per-cycle argmax routing into a single
   state→score map is *worse* than the proportional teacher on these tasks.
2. This matches the arc's own prior finding (`[[exp0006b-stochastic]]`): the search's benefit is
   **temporal re-routing** (per-cycle switching), and collapsing it to a deterministic scorer
   loses exactly the diversity EXP-0006B/0008 showed was the lever. A deterministic oracle
   relabel may be the wrong *form*.

**Do NOT green-light step 2.** Before any full v4 run, two things must change:
- **Re-select probe tasks by measured headroom**, not archived score: first measure
  `champ_prod+search` vs `champ_prod+champ_ord` under the EXP-0030 producer across the archive,
  keep only tasks with `teacher_headroom` above a floor (e.g. ≥ ~2% of champ_cu). If FEW tasks
  clear the floor, the deeper conclusion is that **the augmented producer already closed the
  mined gap** → the orderer lever is largely spent → STOP, not escalate.
- **Explain the oracle<champion degradation**: is the mean-over-slots + center/clip target lossy,
  or does the deterministic distill genuinely lose the search's per-cycle switching?

Gate metric fix: require `teacher_headroom ≥ floor` for a task to count, and report absolute
`recovered` units alongside the ratio (the ratio is unstable as headroom→0).
