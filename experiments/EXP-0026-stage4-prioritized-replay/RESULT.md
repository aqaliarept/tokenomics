# EXP-0026 — Stage 4: Prioritized Replay & Relabeling

Stage 4 of CURRICULUM_GAME_MASTER_APPROACH.md: replay the Stage-3 mined archive,
relabel + train candidate portable score models, evaluate vs champion. No promotion.

## Gate (advisory — promotion is human-approved)

```json
{
  "val_candidate_units": 511.392,
  "val_champion_units": 511.333,
  "val_improves_or_ties": true,
  "blind_candidate_tail": 444.33,
  "blind_champion_tail": 444.33,
  "blind_tail_not_worse": true,
  "new_contract_failures": 0,
  "transfer_regressions": 4,
  "no_transfer_regression": false,
  "promote_recommended": false
}
```

## Caveat

Archive is **single-cell** (all mined tasks `build_scale_config depth=3`), so the best
outcome is *promising, not promoted*. The orderer score model distills only from the
proportional teacher and cannot absorb the search-oracle headroom the mined gap used —
the orderer half is expected to reproduce the champion (a scorer-architecture limit, not
a training-data gap). `PROMOTE_RECOMMENDED=false`.

## Transfer/stress regressions vs champion

```json
[
  {
    "target": "uneven-d3-w4-s6",
    "champion_units": 584.0,
    "candidate_units": 492.33,
    "kind": "value"
  },
  {
    "target": "uneven-d3-w4-s7",
    "champion_units": 584.0,
    "candidate_units": 495.33,
    "kind": "value"
  },
  {
    "target": "uneven-d3-w4-s8",
    "champion_units": 487.67,
    "candidate_units": 465.67,
    "kind": "value"
  },
  {
    "target": "uneven-d3-w4-s9",
    "champion_units": 537.33,
    "candidate_units": 465.33,
    "kind": "value"
  }
]
```

## Conclusions

**REJECT (no promotion). The narrow single-cell replay does not close the mined family and
regresses out-of-family transfer.** Two distinct findings:

1. **No gain on the mined family.** Candidate ≈ champion on mined validation
   (511.39 vs 511.33, +0.06u tie), identical blind-holdout tail (444.33), and a wash on the
   in-family held-out joint (candidate 532.66 vs champion 530.50 over scale s6–9). As predicted
   from the teachers: the portable **orderer** score model distills only from the `proportional`
   teacher, so it structurally cannot absorb the search-oracle headroom the mined gap
   (`entity_planner+mpc_search`, 2–7u) was defined by — retraining reproduces the champion. This
   is a **scorer-architecture limit, not a training-data gap**; the search lever lives in the
   GaussianActor stochastic orderer, which the shared transfer/stress harness cannot score.

2. **Out-of-family regression, producer-side.** All 4 transfer regressions are the
   **uneven-strength** fixture (one dominant producer per level): candidate 465–495 vs champion
   487–584 (−22 to −92u). Everywhere else the candidate ties or slightly beats champion (in-family
   scale, shared-cap, every scarcity + degenerate cell; deep-join d5 is 0/0, the known TTL wall,
   correctly not counted). Attribution: `candidate_orderer_alone` barely moves, and uneven-strength
   is a producer capacity-allocation topology — so the loss is the **producer** scorer. Training it
   *from scratch* on only 12 narrow depth-3 scale seeds **replaced** the champion's broad coverage
   (EXP-0021 `broadened_producer_train_configs`: out-of-family + scarcity), shedding the
   uneven-strength transfer.

**Lesson for the loop.** A single-cell mined weakness of 2–7u is too small and too narrow to be a
useful training signal, and prioritized replay that *replaces* broad training trades out-of-family
robustness for nothing. Two concrete fixes before the next Stage-4 attempt:
- **Broaden the mined archive across topologies first** (the EXP-0025 blocker / PROGRESS point 1) so
  replay has diversity, not one cell.
- **Augment, don't replace**: mix mined replay into the champion's original broad training set
  (prioritized *addition*), rather than distilling from scratch on mined-only configs.

Failure class: this is a **distribution/coverage regression** introduced by training, not a closed
`policy_weakness`. Zero contract failures; deaths unchanged. `PROMOTE_RECOMMENDED=false`.

Pipeline (driver + gate + observability) is validated end-to-end and reusable; the negative result
is about the archive + the replace-vs-augment training choice, not the machinery.
