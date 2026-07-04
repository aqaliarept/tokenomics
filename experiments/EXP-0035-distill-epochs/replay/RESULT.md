# Stage 4 v3a (frozen orderer): Prioritized Replay & Relabeling

Stage 4 of CURRICULUM_GAME_MASTER_APPROACH.md: replay the Stage-3 mined archive,
relabel + train candidate portable score models, evaluate vs champion. No promotion.

## Gate (advisory — promotion is human-approved)

```json
{
  "val_candidate_units": 1106.739,
  "val_champion_units": 1038.405,
  "val_improves_or_ties": true,
  "blind_candidate_tail": 813.33,
  "blind_champion_tail": 772.33,
  "blind_tail_not_worse": true,
  "new_contract_failures": 0,
  "transfer_regressions": 0,
  "no_transfer_regression": true,
  "objective_gate_pass": true,
  "promote_recommended": false
}
```

## Caveat

Producer trained with the selected Stage-4 replay mode, but the orderer is
**frozen at the champion checkpoint**. This isolates producer augment from orderer
replay noise; promotion remains human-approved. `PROMOTE_RECOMMENDED=false`.

## Transfer/stress regressions vs champion

none

## Conclusions

_TODO (human): fill from the artifacts above._
