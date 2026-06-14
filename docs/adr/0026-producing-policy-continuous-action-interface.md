# Producing policy: continuous action interface

Status: accepted

## Context

ADR-0025 gives the producing agent two new jobs the old flat model never had:
**replenishment** (which input products to buy, and how much) and a **third
investment type** (inventory expansion). The old producing action was
`Discrete(4)` — two independent bits (restoration, upgrade) — decoded by
`CategoricalActor.dist()`, which hardcodes that 2-bit layout and the producing
obs indices 4/5 (upgrade-/restoration-running) for action masking. The
training-only cost `investment_request_cost` reads the same two bits.

Replenishment is a per-input continuous quantity, not a bit. A discrete space can
not express it without combinatorial blow-up, so the producing action becomes a
`Box` (ADR-0025 / B5: `dim = I + 3`, I = replenishable input products, +3
investment toggles). That makes several Track A trainer ADRs — written against the
Categorical interface — interface-specific. This ADR records how each is disposed.

## Decision

**Producing actor: Categorical → Gaussian.** The producing policy becomes a second
`GaussianActor`, the same class the ordering policy already uses. Both policies are
now continuous; the trainer's discrete rollout branch (storing producing actions
as ints) is removed — both agents store float-vector actions.

**ADR-0020 (producing temperature) → Gaussian `log_std`.** Exploration is now the
Gaussian standard deviation, the mechanism already tuned for ordering, not a
softmax temperature. The intent (keep stochastic rollouts close to the deployed
mean action) carries over.

**ADR-0021 (producing action mask) → retired.** The masked failure mode was
Categorical action *aliasing* (distinct action ints decoding to the same effect,
corrupting credit). A Gaussian Box does not alias. The "investment already
running" gating it also did moves to decode: a toggle for a running investment is
ignored. No mask survives in the actor.

**ADR-0022 (investment-request cost) → generalized.** `investment_request_cost`
now counts investment toggles fired (≤3: restoration, upgrade, inventory
expansion) instead of the 2-bit value. Replenishment is excluded — its
counter-pressure is the inventory holding penalty (ADR-0025), not this cost.
Default stays 0 (off), as in Track A.

**ADR-0017 / 0018 / 0019 / 0023 / 0024 carry forward unchanged.** GAE successor
links, persistent rollout episodes, output-preserving PopArt, the
agent-conditioned critic, and target-KL `0.008` are model-agnostic. The critic
global state vector resizes with the new producer state (B5), but the conditioning
mechanism is untouched.

**All model and optimizer state starts fresh; no migration.** Not just the
producing actor: the ordering action changes meaning (simplex over *producers*, not
capacity types) and its obs layout changes, so old ordering weights are
semantically stale even where shapes still match — the dangerous case. The critic
global-state dim resizes. Optimizer, PopArt, and obs-normalizer states are keyed to
the old shapes and statistics. So **both actors, the critic, and all
optimizer/PopArt/normalizer state are reinitialized**. A **checkpoint
schema-version tag** is added and bumped for this redesign; the loader **rejects**
any checkpoint with an older or missing version loudly rather than load
shape-mismatched or stale tensors. Migration is out of scope — Track A numbers do
not transfer regardless.

## Considered options

- **Hybrid action space (Discrete investment + Box replenishment).** Rejected:
  a per-agent `Dict`/tuple action multiplies trainer plumbing (two distributions,
  two log-probs, two entropy terms per producing step) to preserve a discrete
  investment head that thresholded Gaussian dims model just as well.
- **Keep Categorical, auto-script replenishment to a fixed target.** Rejected:
  ADR-0025 makes replenishment a *learned* decision; scripting it removes the core
  TOC trade-off the policy is meant to learn.
- **Retune the Categorical mask for 3 investment bits and bolt replenishment on
  later.** Rejected: defers the unavoidable continuous switch and leaves B7/B8
  unable to replenish stock.

## Consequences

- One actor class (`GaussianActor`) for both policies; the Categorical actor and
  its mask logic are deleted, simplifying the trainer.
- Decode owns all producing-action gating (already-running investments, integer
  rounding of replenishment quantities) — the actor emits raw reals only.
- Training is re-validated from scratch — every actor, the critic, and all
  optimizer/PopArt/normalizer state reinitialized behind a rejected-on-mismatch
  checkpoint version tag. Track A's *numbers* do not transfer, but its correctness
  fixes (0017–0019, 0023–0024) do.
- ADR-0021 is superseded; ADR-0020 and ADR-0022 are reinterpreted for the
  continuous interface, not discarded.
