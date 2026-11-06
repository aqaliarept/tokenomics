# User decisions: add-m6-joint-mpg-gnn

- 2026-11-06 (user, via the supervisor):
  1. **Policy form: fully learned, starting from imitation.** The GNN outputs producer and orderer actions
     itself; no hand-written code runs at deployment. Residual heads are dropped from the default.
     - **Stage I:** imitate k5_rf5 + MWWk.
     - **Imitation check, pre-registered:** after imitation and before any RL, does the imitated policy reach
       the bar (fidelity metric and stop rule named)? If it falls short, the user hears it before RL.
     - **Then MAPPO** (CTDE) from the imitated weights.
  2. **Success at validation: parity counts.** Not worse than k5_rf5 + MWWk beyond the floor, with no extra
     deaths and side effects no worse; a pre-registered rule.
  3. **Budget: tranche M6 = 150 vCPU-h**, the supervisor deciding within it; per-run limit 40.
     - The timing probe comes first, before any pod.
     - Learned acceptance stays off in the first ladder, gated behind the A1 diagnostic.
