# Survival reward shaping — death penalty and graded low-capacity penalty

Track A, explored then **shelved**: did not address the good-init collapse.

## Context

System death carries no env score penalty — it just ends the episode (CONTEXT.md:
Score). So survival pays only through far-off foregone reward, and the survival
basin has no restoring gradient at its edge. Hypothesis: a good policy on the
boundary drifts off because nothing pushes back. Two shaping terms on the
**training** reward only (eval stays raw, baseline-comparable):

## Decision

- `--death-penalty P`: one-time `−P` on the transition where system death occurs.
- `--low-cap-coef C` / `--low-cap-thresh τ` (default 0.3): graded, *dense*
  penalty `C · (max(0, τ − health)/τ)²` where `health` = mean capacity/max over
  producers. Zero above `τ`, quadratic to `C` at `health = 0`. Unlike a terminal
  penalty it warns *before* the cliff; unlike linear `surv_coef` (ADR-0010) it
  leaves healthy operation unpenalized, so it does not distort the optimum.

## Result

No effect on the collapse-from-good-init. With critic + KL, the low-cap penalty
(`C=50, τ=0.3`) left the seed-1 eval trajectory **byte-identical** to the
no-penalty run. Two reasons: (1) magnitude — after `REW_SCALE 0.05` the penalty
is ~2.5, negligible vs returns in the hundreds; (2) mis-targeted — seed 1's crash
is an **abrupt** drop from full health (`iter 15→30: cyc 100→33`), not a slow
capacity drift, so a penalty gated on low capacity never sees it coming. The
diagnostic that followed (ADR-0015) showed the collapse is init-quality-driven,
which no reward wall addresses.

Flags retained for experiments; all default 0 (off).
