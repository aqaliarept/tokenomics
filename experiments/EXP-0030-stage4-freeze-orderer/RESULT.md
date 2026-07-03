# EXP-0030 - Stage 4 v3a: Producer Augment + Frozen Orderer

Stage 4 of CURRICULUM_GAME_MASTER_APPROACH.md: replay the Stage-3 mined archive,
relabel + train candidate portable score models, evaluate vs champion. No promotion.

Controlled follow-up to EXP-0029: keep the successful augmented producer path, but freeze the
orderer at the champion checkpoint instead of retraining it on mined replay. This tests whether
EXP-0029's two transfer regressions were orderer replay noise.

## Setup

- Archive: EXP-0028 topology-grid mining archive.
- Producer: augmented train set, broad 54 champion topologies plus 8 mined replay configs after
  held-out leakage drop.
- Orderer: frozen champion checkpoint (`orderer_train_mode=frozen_champion`).
- Run: fresh 16-vCPU CPU Runpod pod, created and terminated by `run_pod_training.py`.
- Hyperparams: producer epochs 300, lr 0.01, eta 1.0, epsilon 0.01, eval-seeds 3, horizon 100,
  fixtures on.

## Gate (advisory — promotion is human-approved)

```json
{
  "val_candidate_units": 493.89,
  "val_champion_units": 494.055,
  "val_improves_or_ties": false,
  "blind_candidate_tail": 472.67,
  "blind_champion_tail": 484.0,
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

**REJECT for promotion, but v3a confirms the orderer-noise diagnosis.**

1. **Transfer gate is clean.** Freezing the orderer removes EXP-0029's `scale-d3-s9` and
   `sharedcap-d3-w4-s9` regressions. Transfer/stress has 0 contract failures and 0 value
   regressions. Held-out scale and shared-cap cells tie champion exactly; scarcity high-backlog
   improves 992.67 -> 1010.33; uneven has no gate regression.

2. **Objective gate still fails on mined replay.** Mined validation is a small miss:
   493.890 vs 494.055 (-0.165u). Blind holdout fails the tail check: 472.67 vs 484.0, caused
   by one task (`products_per_layer=3`, `producers_per_layer=4`, seed 8) dropping -11.33u.

3. **Producer/orderer attribution is now clean.** The orderer is champion-exact by construction;
   the held-out scale decomposition ties champion for every stack and seed. The remaining miss is
   not orderer replay drift. It is the augmented producer's replay fit moving one mined blind cell
   while preserving transfer.

**Next lever:** do not promote. Run a stratified/duplicate-capped producer replay variant, or reduce
producer mined-replay concentration, because the sampled replay is still seed-14 heavy
(`[14, 14, 15, 14, 26, 14, 14, 14]`) and did not improve mined validation/blind robustly.
