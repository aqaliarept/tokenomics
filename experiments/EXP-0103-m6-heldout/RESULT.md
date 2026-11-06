# EXP-0103: held-out read of the imitated M6 stack (one shot)

add-m6-joint-mpg-gnn task 2.9 (user decision 2026-11-06: held-out now, then MAPPO). Pre-registration
`m6_prereg_v4.json` (SHA `f06fdf0c…`, frozen before data). Ledger entry `held-out/m6-imitated` pending before
the launch, closed `done` after. Code 012acc4; EXP-0100 models (SHA `5acf8b82…`). Pod 0ryv5idl4w17ex (secure
16 vCPU, EUR-IS-1), 12:11–12:18 UTC ≈ **2.0 vCPU-h** (stop 8). A first launch (pod lya3nhr20bnmtl, b2069af)
was removed after about 1 min, before syncing: the ledger lives outside the synced package dir, so the read
would have refused. Fixed in 012acc4 with `--ledger` and a staged copy; ≈ 0.2 vCPU-h. Total ≈ 2.2.

## Question
Does EXP-0102's `better` hold on the held-out worlds, read once? M5's L6 set: every atlas v2 held-out world
played (12 at 12, 4 at 42), the loaded ones gating (8 / 2), 8 episodes (stage m5-heldout) × replicates 0 / 1,
horizon 100, the parity / better / short rule with L5's side-effect clauses; 12 producers deciding, 42
reported.

## Result: **`better` at 12 producers (deciding)**; 42 (reported) **`short` on the run-down clause**
| size | arm | mean units (gating worlds) | − base | floor | deaths | run-down share | starvation share | holding / cycle |
|---|---|---|---|---|---|---|---|---|
| 12 | GNN + GNN (deciding) | 3,490.8 | **+184.4 (+5.6%)** | 82.2 | 1 | 0.758 | 0.766 | 335.4 |
| 12 | GNN producer + MWWk | 3,485.1 | +178.7 | 83.8 | 1 | 0.789 | 0.773 | 334.2 |
| 12 | k5_rf5 + MWWk (base) | 3,306.4 | — | — | 2 | 0.867 | 0.781 | 275.6 |
| 42 | GNN + GNN | 15,825.8 | +840.9 (+5.6%) | 223.1 | 0 | **1.000** | 1.000 | 2,347.4 |
| 42 | GNN producer + MWWk | 15,806.9 | +822.1 | 168.3 | 0 | 0.969 | 1.000 | 2,357.7 |
| 42 | base | 14,984.9 | — | — | 0 | **0.500** | 1.000 | 2,353.1 |

**At 12 producers:**
- Ahead on 7 of the 8 gating worlds (−16 … +624).
- Fewer deaths (1 vs 2), lower run-down share (0.758 vs 0.867), lower starvation share (0.766 vs 0.781).
- Over all 12 played worlds, a mean of +127.0.

**At 42 producers:** the units clear the floor on both gating worlds (+302 on 42-0005, +1,380 on 42-0125),
but the run-down clause fails, on world 42-0125:
- The base never runs down there: 0 / 16, r_end / r_start 0.772.
- The GNN stack runs down in all 16 episodes, at 0.646.
- **This is a drain, not an artifact of upgrades raising the max.** Mean total capacity per producer rises
  with the base (520 → 632 → 671 → 688 at cycles 25 / 50 / 75 / 99). With the GNN it peaks near cycle 50
  (616) and falls to 550 by cycle 99.
- The GNN restores slightly less there (0.457 vs 0.474), upgrades more (0.205 vs 0.168), and has a higher
  investment rejection (0.159 vs 0.111).
- Cause undiagnosed. A horizon-200 run at 42 would show whether the units fall once the drain bites.
- On 42-0005 both arms run down in nearly every episode (16 / 16 each).

## Conclusions
- **The pre-registered, deciding reading is `better`:** on held-out worlds at 12 producers, the self-trained
  stack completes 5.6% more consumer units than k5_rf5 + MWWk, with no side effect worse. This is the user's
  goal, confirmed once at 12.
- **At 42 producers (reported, not deciding)**, the gain comes with a capacity drain on one world. This must
  not be read as a 42-producer confirmation. It is a known side effect to carry into MAPPO: the potential's
  per-cycle run-down term, and a 42 clause in the RL pre-registration.
- MAPPO's later held-out read on this set is a second comparison (pre-registered in v4).

## Addendum (2026-11-06, EXP-0104)
The 42-producer run-down on 42-0125 is diagnosed: the imitated producer upgrades past the sustainable ceiling at 42-producer scale (DIAGNOSTICS 32, the upgrade trap of entry 29, reached by imitation). Vetoing those upgrades removes the drain. The 42 read stays reported only; the 12-producer reading is unaffected.
