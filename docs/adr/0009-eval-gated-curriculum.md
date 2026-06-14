# Eval-gated curriculum — advance on survival, not iteration count

Track A step 3 (Phase 1 plan). The iter-based curriculum (ADR-0006) ramps
difficulty on a fixed schedule regardless of whether the policy can survive the
current difficulty. ADR-0007 (critic) + ADR-0008 (schedule) cracked the bar on
seed 0 (`dead 0.00`, `ret ~3500`), but seed 1 **collapsed**: it survived at
diff 0.20 (iter 0), then as the schedule forced difficulty up it died at every
difficulty from ~0.5 on and never recovered (`dead 1.00`, `ret ~1900` at the
end). The schedule outran the policy — the exact failure this step targets.

## Decision — advance difficulty only when the rollout survives

`Curriculum.gate=True` adds a gated mode:

- The rollout (`collect`) reports its **death rate at the current training
  difficulty**: fraction of completed episodes that ended in system death.
- After each rollout, `Curriculum.observe(death_rate)` steps difficulty up by
  `d_step` (0.1) only when `death_rate <= advance_below` (0.25). Otherwise it
  **holds** — the policy gets more iterations at the difficulty it is failing
  until it masters it.
- `Curriculum.level(it, iters)` returns the gated level directly. There is no
  iteration-based floor: advancing without a completed survival result defeats
  the purpose of gating.

## Why gate on rollout survival, not the eval metric

The plan says "gate on eval." But eval (ADR-0003) runs at **real** difficulty —
it is the goal, not the training difficulty. Gating the *ramp* on real-difficulty
survival is circular: you would only raise training difficulty once the policy
already survives real difficulty, i.e. once it is already done. The signal that
actually prevents outrunning is survival **at the difficulty being trained**,
which is the rollout death rate. Eval stays the reported goal metric; the gate
uses the rollout. Same intent ("hold each difficulty until dead drops"), correct
signal.

## Trade-offs

- The rollout death rate is coarse. Persistent rollout environments continue
  across batches until a real episode result exists; batches without one report
  `NaN` and hold difficulty.
- `Curriculum` is now stateful (`_gated_d`) — one instance per run; do not share
  across runs. `__post_init__` seeds it to `d_start`.
- `collect` now returns a 3-tuple `(pbuf, obuf, death_rate)`; callers/tests
  updated.
- Ungated mode (`gate=False`) is unchanged: `observe()` is a no-op and `level`
  is the old iter schedule, so ADR-0006 behavior and its test stand.
