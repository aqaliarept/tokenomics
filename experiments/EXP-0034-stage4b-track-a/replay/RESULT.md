# Stage 4 v3a (frozen orderer): Prioritized Replay & Relabeling

Stage 4 of CURRICULUM_GAME_MASTER_APPROACH.md: replay the Stage-3 mined archive,
relabel + train candidate portable score models, evaluate vs champion. No promotion.

## Gate (advisory — promotion is human-approved)

```json
{
  "val_candidate_units": 1053.202,
  "val_champion_units": 1038.405,
  "val_improves_or_ties": true,
  "blind_candidate_tail": 721.67,
  "blind_champion_tail": 772.33,
  "blind_tail_not_worse": false,
  "new_contract_failures": 0,
  "transfer_regressions": 0,
  "no_transfer_regression": true,
  "objective_gate_pass": false,
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

Producer augment-replay on the 113 recoverable **capacity-binding** tasks (57 train /
28 val / 28 blind, champion = EXP-0030 producer_candidate, orderer frozen at EXP-0012):

- **Validation +14.80u mean** (16/28 better, 12 worse) — candidate beats champion.
- **Blind +8.33u mean** (18/28 better, 10 worse) — net improvement, but the **tail**
  gate fails: worst blind task −107u (`capacity_scale0.35 ppl2 qmax14 s54`) drags
  `blind_tail` to 721.67 < champion 772.33.
- **Broad transfer (seeds 6–9) byte-identical** to champion → 0 forgetting, 0 transfer
  regressions, 0 contract failures, 0 deaths.

`objective_gate_pass=false` is driven **only** by the tail metric; the mean improves on
both held-out capacity-binding splits. This is the FIRST Stage-4 producer replay to move
val at all — contrast EXP-0031's 5.27% closure / 21-of-26 not_producer on the
demand-saturated family. The candidate is a mean-improving, tail-trading producer;
promotion is a human call (same posture as the EXP-0030 incumbent override). See the
parent `../RESULT.md` for the full Track A pipeline.
