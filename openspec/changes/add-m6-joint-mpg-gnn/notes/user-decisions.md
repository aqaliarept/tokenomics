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
- 2026-11-07 (user, via the supervisor): **a new tranche M6-A = 70 vCPU-h for learned acceptance** (tasks §4, plan
  committed 89d9381); the supervisor decides within it. Its own ledger section; EXP-0114 stays on M6's main tranche.
  Hand-written acceptance variants (A1, A1-life, A1-new-first) are diagnostics only, never candidates.
- 2026-11-07 (user, via the supervisor): **the fund spend order (stage 2 inside a producer) and reservation scheduling
  also become learned producer decisions** (tasks §4b, 4.6 / 4.7, committed e8c2f71; audit notes/hardcoded-audit.md,
  2703876). Both fold into the 4.2 producer turn and its single review round: an ADR, defaults byte-identical, priority
  heads per funding request and per reservation / claim. The 4.3–4.5 ladder covers them.
- 2026-11-07 (user, via the supervisor; after EXP-0116 `replenish_drift`):
  1. **Confirm F40, then use it.** F40 = EXP-0114's iteration-40 investment gates + the frozen start's replenish.
     - A pre-registered confirmation read: F40 against the EXP-0100 stack on fresh seeds (the monitor worlds' new
       episodes and the atlas validation worlds), with v11's metric and floors, ≈ 3–5 vCPU-h on M6-A.
     - On `gain`, 4.3 and 4.5 start from F40, and the turn heads are re-fit on F40's trunk (v13 unchanged except
       for the start, ≈ 5 vCPU-h).
     - EXP-0118 continues as the method check on EXP-0100's trunk.
  2. **4.5 trains every head on-policy:** investment, replenish, fund and accept, each played in training exactly
     as deployed. The joint start is F40 as one model. Option (c): the replenish head trains too, starting from the
     start's replenish (settled in the 4.5 design).
  3. **4.5's reward is re-weighted toward consumer units,** with a smaller run-down term kept as a guard. The
     weights are pre-registered before 4.5.
- 2026-11-07 (user, via the supervisor; 4.3 blocked after EXP-0121 / EXP-0122, no probe setting reaches 0.99):
  **closed-loop diagnostic first.**
  - EXP-0120's heads on F40 against the default turns, on fresh monitor episodes (never played), paired, with v11's
    metric and floors and side effects (deaths included). Also reported: the closed-loop accept-bit agreement, the
    boundary split of the differing decisions, and how often a disagreement becomes a late pending claim
    (DIAGNOSTICS 36).
  - Pre-registered reading: `no_cost` (|g| within the floor, side effects no worse) / `cost` (g < −floor or worse
    side effects) / `gain` (g > floor).
  - The 4.3 offline gate is unchanged and still reads `offline_fail`. ≤ 3 vCPU-h on M6-A, its own EXP;
    `test_m6_pilot.py` runs on the same pod.
  - Whether 4.5 starts from these heads goes back to the user.
- 2026-11-07 (user, via the supervisor; after EXP-0123 `cost`): **an order-aware network for the accept turn.**
  - Each item's accept decision follows the items its own learned priority ranks above it: a causal mask by
    learned score, or a small recurrent pass in learned order (the simpler that can represent an exact running
    sum).
  - No hand-written rule: the order comes from the head's own priority output, and there are no cumulative hand
    features.
  - The deployed path must equal the trained path where it matters (DIAGNOSTICS 37).
  - Order of work: code with tests, one review round, a probe on EXP-0121's halves (≤ 3 vCPU-h, pre-registered);
    on 0.99, v15 with the fresh held-out set and 4.3 on F40, then the closed-loop check; otherwise report.
- 2026-11-07 (supervisor, after EXP-0125's winner): **v15's fit budget is 30 epochs on the full 427k training
  turns** (≈ 4.5× the probe's steps, for margin over 0.9924), cosine over the full run. The gate reads the final
  model only (no early stopping, no best-epoch pick). A checkpoint is saved each epoch for crash safety only.
  - v15 = ordered pass + linear fractions + arrival position, w32 / 3 layers, on F40, the fresh held-out set,
    pre-registered before data.
  - If the offline gate passes, the closed-loop check (EXP-0123's design, fresh episodes) runs in the same or a
    follow-up pod.
- 2026-11-07 (user, via the supervisor): **v15's heads (EXP-0126) are accepted as the 4.4 / 4.5 start**, on combined
  evidence.
  - The pre-registered closed-loop gate stays `fail`, on the run-down clause alone.
  - Pooled over the 96 pairs (gate + diagnostic sets), run-down was 34 vs 33 rows.
  - Units were within noise, closed-loop agreement 99.8%, and the late-pending work is gone.
  - 4.3 is done.
- 2026-11-08 (user, via the supervisor; after EXP-0127–0130): **close the RL track.**
  - 4.4 reads `absent`: F40's start is flatter on the pre-registered criterion (EXP-0127–0130).
  - 4.5 is not run, by user decision.
  - M6's result is F40 + v15 turn heads.
  - One final read, EXP-0131, on M6-A (≤ 8 vCPU-h): F40 + v15 heads against k5_rf5 + MWWk, with EXP-0102's rule on
    the held-out set, 12 producers deciding and 42 reported, through `deployed_policies` only, pre-registered before
    data. The roster, registry and champion files stay the user's gates.
