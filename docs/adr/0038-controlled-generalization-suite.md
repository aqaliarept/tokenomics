# Controlled-generalization suite (single-axis, dims-preserving perturbations)

Status: accepted

## Context

NEXT STEPS stage 3 ("controlled generalization") asks to evaluate a policy on
unseen demand schedules, unseen quantities / product mixtures, numeric producer
variations, multiple viable producers, and higher degradation — **changing one
dimension at a time to identify failures**, while (stage 5) keeping action and
observation dimensions fixed across the suite.

`train.smoke_ppo` already has config-scaling functions (`scale_degradation`,
`scale_demand`, `scale_competition`) but they are *curriculum easers* — they ramp a
difficulty scalar toward `p=0` (easier than the real task) and live in a module that
imports torch. Generalization needs the opposite: perturbations toward *unseen /
harder* settings, usable from the torch-optional eval layer.

## Decision

Add **`eval.generalization`** — pure (no torch, no RL) single-axis config transforms,
each `dataclasses.replace` over the immutable config:

- `perturb_degradation(cfg, factor)` — scale per-cycle decay up (unseen harsher decay).
- `perturb_quantity(cfg, quantity_max)` — raise consumer demand ceiling to an unseen value.
- `perturb_producer_capacity(cfg, factor)` — scale owned capacity (numeric producer variation).
- `generalization_suite(base_cfg)` — one named config per dims-preserving axis
  (`higher_degradation`, `unseen_quantities`, `leaner_producers`).

**Every perturbation preserves obs/action dims** (it changes only numeric values —
never the producer / product / capacity-type *set*), enforced by a test that builds
the spec for each suite config and asserts the four dims equal the base. So a policy
trained on the base is directly scoreable on every axis; failures are attributable to
the single changed dimension.

The two axes that need a different *topology* than canonical phase1 — **product
mixture** and **multiple viable producers** (phase1 has one declarer per product,
ADR-0028, so there is no routing choice and no second product in the consumer pool) —
are deliberately **excluded** from `generalization_suite(phase1)`; they require a
multi-declarer base (competition.json) and are recorded as such rather than faked.

## Considered options

- **Reuse `scale_*` from `smoke_ppo` with factor>1.** Rejected: importing them pulls
  torch into the eval layer (CLAUDE.md: eval is torch-optional), and they encode the
  easing direction + curriculum semantics, not unseen-hard perturbation. Kept as
  separate pure functions; the small overlap is justified by the layer boundary.
- **Change topology to exercise all five axes on phase1.** Rejected: that breaks the
  fixed-dims requirement and conflates axes. Mixture / multi-producer belong on a
  multi-declarer base.

## Consequences

- Stage 3 is now a cheap, attributable sweep: build `generalization_suite(cfg)`,
  `evaluate` any policy/baseline per axis, read which single dimension breaks it.
  Reused by stage 4 (combined held-out suite) and stage 6 (stress) with more axes.
- The suite is the same harness for the trained policy and the observation-compatible
  baseline (ADR-0036), so each axis is reported head-to-head.

### Result (phase1.json, trained seed 0 vs obs baseline, 10 seeds/axis, horizon 100)

| axis               | who      | dead% | cons✓ | units✓ | return |
|--------------------|----------|------:|------:|-------:|-------:|
| base (phase1)      | trained  |     0 | 100.0 |  100.0 | 1252.7 |
| base (phase1)      | obs_base |     0 | 100.0 |  100.0 | 1255.1 |
| higher_degradation | trained  | **100** | 54.0 |  54.0 |  732.9 |
| higher_degradation | obs_base |     0 | 100.0 |  100.0 | 1291.1 |
| unseen_quantities  | trained  |     0 |  60.0 |  119.1 |  542.7 |
| unseen_quantities  | obs_base |     0 |  56.6 |  101.9 |  255.1 |
| leaner_producers   | trained  |     0 | 100.0 |  100.0 | 1259.3 |
| leaner_producers   | obs_base |     0 | 100.0 |  100.0 | 1265.2 |

Changing one dimension at a time localizes the behavior cleanly:

- **`higher_degradation` (decay 5→10) is a catastrophic generalization failure for the
  trained policy: 100% death, 54% fulfilled, return 732** — while the
  observation-compatible baseline shrugs it off (0% death, 100% fulfilled, 1291). The
  baseline is a *closed-loop* rule (restore-when-low), so it tracks any decay rate; the
  trained greedy policy restores/replenishes **open-loop** (memory: greedy mean is
  open-loop) at a cadence tuned to the training degradation, so doubling decay collapses
  it. This **fails the stage-6 bar** ("degrade gradually rather than collapse") — the
  trained policy overfits the training decay rate.
- **`unseen_quantities` (fixed 1 → max 4): the trained policy generalizes well** — it
  survives (0% death) and delivers MORE units and return than the baseline (119 vs 102
  units, 543 vs 255). Order-count fulfillment drops to ~60% for both because qty-4
  demand outstrips capacity; that is the env getting harder, not a policy regression.
- **`leaner_producers` (capacity ×0.7) is non-discriminating** — phase1 has large
  capacity headroom (120 owned vs recipe need ~2–25), so 30% less changes nothing for
  either. A sharper factor (or a tighter base) would be needed to probe this axis.

**Verdict: stage 3 PASSES on the quantity axis, is INCONCLUSIVE on capacity slack, and
FAILS hard on degradation** — the trained policy does not generalize to an unseen decay
rate. Fix is reward/curriculum (train across a degradation *range*, not a fixed rate),
recorded for a follow-up; the suite now makes the failure attributable to one axis.
