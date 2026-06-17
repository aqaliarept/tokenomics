# Critic fit gate for actor updates

Status: accepted

## Context

ADR-0013 lets the trainer freeze the actors for a fixed `--critic-warmup K` iters so
policy gradients use a warm critic. At depth-3 scale that fixed count is the wrong knob:
VERIFICATION §13.6 shows the centralized critic has **negative** explained variance on
cloned-policy rollouts even after 25 warmup iters, so the warmup expires while value
learning is still untrustworthy. Once actors unfreeze, PPO advantages — built on an unfit
critic — drive a good (behavior-cloned) policy back toward capacity hoarding / under-
production: validation units fall ~78→17, step reward ~17→3. The blocker is value
learning, not exploration, so the freeze condition should track the critic's fit, not a
guessed iter count.

## Decision

Add a **critic fit gate**: keep actors frozen until the critic demonstrates a positive
fit, then let PPO drive the policy.

- `actors_should_freeze(it, critic_warmup, recent_evs, critic_gate_ev)` is a pure function
  (beside `explained_variance`). Frozen when (a) inside the fixed `critic_warmup` window,
  or (b) `critic_gate_ev` is set and the critic has not earned the right to move — either
  no finite explained-variance reading exists yet (`recent_evs` empty) or the rolling-mean
  EV is below the threshold. `critic_gate_ev=None` preserves ADR-0013 warmup-only
  behaviour, so the canonical phase1 path is unchanged.
- `train()` keeps a rolling deque of the last `--critic-gate-window` **finite** EV
  readings and gates the existing `warming` flag on it. The reading lags one iter (EV is
  measured after the update), which is harmless. Each `IterStats` carries `actors_frozen`
  so the curve shows when PPO actually drove the policy vs. when only the critic learned.
- CLI: `--critic-gate-ev` (off by default) and `--critic-gate-window` (default 3).

## Considered options

- **Tune `--critic-warmup` higher.** Rejected: §13.6 shows EV stays negative past 25
  warmup iters, so no fixed count is reliable across topologies; the gate conditions on the
  actual fit instead.
- **Gate on the current iter's EV (no lag).** Rejected: EV is read after the actor update;
  gating on it would require a second critic pass before the update. The one-iter lag over
  a rolling window is simpler and stable.
- **Single-iter EV instead of a rolling mean.** Rejected: per-iter EV is noisy and would
  flap the actors frozen/unfrozen; the window smooths the decision.

## Consequences

- Makes the §13.6 recommendation runnable: warm-start from the BC checkpoint and train with
  `--critic-gate-ev 0.0` to pin actors near the clone until the critic stops being worse
  than predicting the mean.
- Pure-helper truth table and an end-to-end weight-invariance test (an unreachable EV
  threshold keeps actor weights at init across all iters; the no-gate default never
  freezes). Full suite green (213 tests).
- Whether the gate lets PPO improve on the clone at depth-3 is the open A/B question,
  measured separately once a run clears a positive EV threshold.
