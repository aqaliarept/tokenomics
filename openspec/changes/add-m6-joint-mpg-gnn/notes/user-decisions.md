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
- 2026-11-06 (user, via the supervisor), after EXP-0102 `better`: **held-out now, then MAPPO.**
  - EXP-0103 reads the imitated stack (EXP-0100 models) once on the held-out worlds (M5's L6 set and rule),
    pre-registered in `m6_prereg_v4`; MAPPO's later held-out read is a second comparison on the same set.
  - Then the RL pre-registration (task 2.5b), frozen before any MAPPO data. MAPPO succeeds only if it beats the
    imitated stack beyond the floor with no side effect worse; otherwise the imitated stack stays the result.
- 2026-11-06 (user, via the supervisor): **the RL stage is critic-free** (source: docs/research/critic-free-team-learning.md,
  8ef3425, "Recommendation"). Each world-seed is played G = 4 times by the current team. Each copy's per-cycle
  return-to-go is scored against the mean of its siblings at that cycle (leave-one-out, no std division). No learned
  critic in training. PPO-clip with a KL anchor to the imitation policy. A frozen champion team (the EXP-0100
  models) is used only for the promotion gate: a paired test on the same seeds. The potential penalties stay in the
  per-cycle reward, normalised per world. A Stage 0 diagnostic (pinned m6_prereg_v5) decides whether and how
  Stage 1 runs; the critic code is kept only for Stage 0's offline audit.
- 2026-11-06 (user, via the supervisor), after EXP-0111 `absent`: **reshape exploration and re-test** (EXP-0112, pinned
  m6_prereg_v9; v8 stays).
  - Input buying is frozen at its imitated deterministic value (no Gaussian noise, outside the RL gradient).
  - The producer's investment gates get a temperature, set offline so each of restore / upgrade / expand flips
    about 5% (training rollouts only; the deployed greedy policy is unchanged).
  - The orderer is as is.
  - A new estimator E4 (per-agent local value on the agent's own view, cross-fitted on split halves; the user's
    IPPO / DPO report).
  - The same power check as EXP-0111.
  - `signal` needs a producer estimator to beat its null at 12, with ≤ 96 seeds per half for 80% power; 42 is
    reported.
- 2026-11-06 (user, via the supervisor), after EXP-0112: **a capped Stage 1 training pilot** (EXP-0113, pinned m6_prereg_v10).
  - **It overrides v9's `absent`, deliberately:**
    - The power check found a real producer gradient at 12 (|∇J|² 95% CI above 0), with n80 = 48 seeds per half
      for E3, inside the rule's 96.
    - D0.3 at Stage 0's 12 seeds per half was underpowered by construction, so its miss is not evidence of no
      signal.
  - **Pilot design:**
    - the producer learns at 12 producers; the orderer is frozen at EXP-0100's weights; input buying frozen; gates
      tempered (training only);
    - E3 at 48 seeds per half (96 per iteration); PPO-clip + KL anchor; the per-agent Stepper with step logs;
    - 10 iterations, a greedy gate at 5 and 10 against the EXP-0100 stack on the monitor set;
    - early stop on a loss, death or worse run-down at 5; reading `gain` / `hold` / `worse` at 10.
  - **Continuation** only on `gain` (or `hold` with a rising curve, the user's call), as a later decision.
- 2026-11-07 (user, via the supervisor), after EXP-0113 `hold` with a rising curve: **continue the pilot 30 more iterations,
  to iteration 40** (EXP-0114, pinned m6_prereg_v11 `f55aea40…`; v10 stays).
  - **Setup:** resume from EXP-0113's iteration-10 checkpoint (SHA-pinned), every v10 setting unchanged; the resume
    pins are checked.
  - **Gates at 15, 20, 30 and 40:** the same monitor gate against EXP-0100. 15 / 20 / 30 stop the run on g < −F,
    more deaths or a higher run-down share. The deciding reading is at 40 only (gain / hold / worse as v10), so the
    best of four looks is never picked. The monitor set is the same at every gate.
  - **Reported:** the curve, KL (plateau or not), clip fraction, flip rates, training deaths.
  - **Budget:** stop ≤ 40 vCPU-h (≈ 36 expected), M6 → ≈ 129 of 150, leaving room for validation and a held-out read.
  - **On `gain`:** validation on the unseen L5 worlds next (set up by the supervisor after).
