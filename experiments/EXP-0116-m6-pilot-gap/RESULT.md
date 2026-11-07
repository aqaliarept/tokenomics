# EXP-0116: why EXP-0114's training return rose while the greedy gate's units did not

**Setup.**
- Pre-registration `notes/exp0116-prereg.md` (SHA `5ef8221f…`, committed 364b070 before data). Driver
  `train/m6_gap_probe.py`.
- Pod obj0bkagr1msh0 (secure 16 vCPU + 20 GB volume, EUR-IS-1), 11:16–11:27 UTC ≈ **3.1 vCPU-h** (stop 8,
  tranche M6-A).
- 1,760 episodes at horizon 100, 12 producers. The frozen orderer is the same in every arm.

**Validity: valid.**
- `G40` reproduces EXP-0114's iteration-40 gate (4,145.125 units): MATCH.
- `ref` reproduces EXP-0114's reference rows row for row: MATCH.
- `start` plays exactly as `ref`, unit for unit: the start is EXP-0100's producer, extended to v4 at zero weight.

## Reading: **`replenish_drift`**
The gate scored a policy RL never trained.
- **In training:** input buying (replenish) is the frozen start's output.
- **In the gate:** the candidate uses its own replenish head. That head got no gradient, but it reads the shared
  trunk, which RL moved.

**The trained policy itself (`F`: greedy gates + the start's replenish) gains steadily.** Monitor, 16
world-episodes × r0 / r1, against the EXP-0100 stack. Units and the g / F gap are per episode:

| iteration | F (as trained) units | g / F | G (as gated) units | g / F |
|---|---|---|---|---|
| 10 | 4,183.7 | +116.3 / 140.2 | 4,174.5 | +107.1 / 154.1 |
| 20 | 4,240.8 | +173.3 / 122.7 | 4,191.2 | +123.7 / 100.1 |
| 30 | 4,274.4 | +206.9 / 118.8 | 4,069.7 | +2.3 / 115.1 |
| 40 | **4,297.6** | **+230.1 / 136.4** | 4,145.1 | +77.7 / 116.6 |

Reference: 4,067.4.

**F40's side effects (monitor):**
- deaths 0 vs 0;
- run-down share **0.28** vs 0.56;
- starvation share **0.56** vs 0.66.

On the pre-registered v11 rule F40 would read `gain`. It is a diagnostic arm, so EXP-0114's reading stays `hold`.

**The drift costs:**
- **Monitor:** Δ(F40 − G40) = **+152.4 ± 46.1** units per episode, against a flag line of ½ F = 58.
- **Training set** (112 world-episodes × r0 / r1): **+117.9 ± 13.6**.

**The other candidates are ruled out:**
- **Greedy vs tempered:** the tempered training policy T40 gains +242.4 on the monitor, and greedy F40 +230.1.
  Greedy play keeps the gain (`greedy_loses_it` false).
- **Monitor power:** on the training set, F40 − start = +238.5 (F 36.3), consistent with the monitor. The
  monitor resolves it once the right policy is scored (`monitor_power` false).
- **Potential vs units:** on the training set, T40 − T_start gains Φ +38.0 ± 2.6 normalised. Its parts:

  | term | gain | share |
  |---|---|---|
  | units | +7.3 | 19% |
  | run-down penalty | +27.3 | 72% |
  | starvation | +2.9 | |
  | death | +0.4 | |

  The units gain is still real (+260 units, F 46), so `not_units` is false. Most of the return is run-down
  relief, though: λ dominates Φ's gradient.

## Conclusions
- **Cause (DIAGNOSTICS 37):** the train / deploy mismatch on input buying.
  - RL trained the gates with the start's replenish.
  - The deployed candidate's replenish head silently follows the moving trunk: |Δ fraction| 15% at
    iteration 10 → 33–36% at 30–40, measured locally before data.
  - That cost about 150 units at the gate.
- **What RL learned:** the trained policy reaches +230 units per episode (+5.7%) at iteration 40, rising at every
  look, with no deaths and lower run-down and starvation.
- **For 4.5 (and any trunk-sharing RL):** every head that reads a trainable trunk must be played in training as it
  will be deployed. Options, a design decision before 4.5:
  - **(a)** deploy and gate with the frozen start's replenish (and turn heads), i.e. save the candidate with a
    frozen trunk copy for those heads;
  - **(b)** freeze the trunk and train only heads;
  - **(c)** train all heads on-policy.

  4.3 (EXP-0117) is not affected: its trunk is frozen.
- **Φ weighting:** 72% of the training-return gain is the run-down term. If 4.5 is meant to buy units, λ / μ
  should be re-weighted (a design decision, reported).
