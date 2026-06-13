# ADR-0006: RL difficulty curriculum (degradation annealing)

## Status
Accepted (Phase 1.5 P4).

## Context
After P2/P3 the trained policy still evaluates at `dead 1.00` / `capmin 0`: it
never learns to keep capacity alive. The real `phase1.json` is a hard survival
task — degradation (3–8 % of max capacity per cycle) outruns what a randomly
initialized policy can fund through restoration before the network dies, so the
policy sees almost no episodes where investment pays off. Sparse-survival
bootstrapping is the textbook case for a **difficulty curriculum**: start on an
easier task where survival is reachable, anneal toward the real one as the policy
improves.

This is an RL-training concern, not a domain change — it lives in `train/`, never
in `domain/` (which stays pure) and never in the shipped config.

## Decision
Anneal a single difficulty axis: **producer degradation**.

- `Curriculum(d_start, warmup_frac)` maps training progress to a difficulty
  `d ∈ [d_start, 1.0]`. Linear ramp from `d_start` at iter 0 to `1.0` once
  `warmup_frac` of the run has elapsed; flat at `1.0` after.
- `scale_degradation(cfg, d)` returns a copy of the base `Configuration` with
  every producer's `degradation` replaced by `max(0, round(base * d))`.
  `dataclasses.replace` on the frozen configs — base config untouched.
- Each iteration rebuilds the rollout envs from the current-difficulty config.
  Degradation scaling changes **no** observation/action dimension (producer ids,
  capacity types, `cap_ref`, emission all fixed), so the actor/critic networks
  built once at `d=1.0` spec stay valid for every difficulty.
- **Eval always runs on the real config** (`evaluate()` builds a default-config
  env). The training curve therefore charts progress on the *true* task even
  while rollouts are eased — the number we care about is real-task survival.
- Default ON (`d_start=0.2`, `warmup_frac=0.6`): the curriculum exists to fix the
  observed `dead 1.00`. `--no-curriculum` reproduces the flat real-difficulty run.

## Why degradation (not emission)
Degradation is the direct driver of system death (ADR-0002 termination) and a
clean scalar per producer. Raising emission would also ease funding but distorts
the token/price normalization (`token_ref = cycle_emission`) and the auction
dynamics. Degradation scaling keeps every normalization constant fixed.

## Consequences
- Survival becomes learnable early; difficulty rises as the policy adapts.
- Curve stays comparable to the baseline table (eval = real task).
- One scalar axis only. Multi-axis curricula (emission, demand load, TTL) and
  automatic/teacher-driven pacing are future work behind the same `Curriculum`
  seam.
- A risk remains: if `warmup_frac` ramps faster than the policy learns, late
  iters can regress toward `dead 1.00` — visible directly in the curve, tune
  `warmup_frac`/`d_start` from there.
