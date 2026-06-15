# Config fingerprint on checkpoints and held-out seed separation

Status: accepted

## Context

Two evaluation-hygiene gaps surfaced in B9:

1. **Wrong-config evaluation.** A checkpoint encodes actor input/output layers sized
   to one config's spec. Scoring it on a different config (different producer count,
   product vocabulary, …) either crashes deep in the actor on a dim mismatch (the
   "1x27 vs 29x64" gotcha, see [[b9-randomized-suite-behavior]]) or — worse if dims
   happen to line up — silently produces a meaningless comparison. Nothing on the
   checkpoint recorded which economy it was trained on.
2. **Selection/reporting on the same seeds.** `train()` picked the best checkpoint by
   greedy-eval on `range(eval_seeds)` and that same number was the reported result.
   Selecting and reporting on the same seeds biases the figure upward (it is the max
   over checkpoints of an estimate on those exact seeds).

## Decision

**Stamp a config fingerprint on every checkpoint, and separate validation seeds
(selection) from held-out seeds (reporting).**

- `env.spaces.config_fingerprint(config)` — a short SHA-256 over **both** (a) the
  shape-determining structure (producer ids, capacity types, orderable and input
  product vocabularies, derived space dims + `k`) **and (b) the full economic content**
  (recipes, capacities, degradation, investment/replenish costs, emission, consumer
  demand distribution), via a canonical `dataclasses.asdict` + `json.dumps(sort_keys)`
  serialization. It lives in the torch-free `spaces` module so both the trainer (stamp)
  and `signal_check` (compare) can use it without importing torch.
  **Revised by review P1 (2026-06-15):** the original shape-only fingerprint collided
  for configs with identical IDs/dims but different *dynamics* — e.g. `competition.json`
  vs `complex.json` (fixed vs randomized demand), `random_demand.json` vs
  `composite.json` (different recipes) — so `signal_check` reported a *valid match* for
  an off-config comparison, defeating the fingerprint's purpose. The economy hash fixes
  the collision.
- `save_policies(..., config=base_cfg)` records `config_fingerprint(base_cfg)` in the
  checkpoint; `checkpoint_fingerprint(path)` reads it back (None for pre-fingerprint
  checkpoints). `signal_check --policy` warns when the trained fingerprint differs
  from the eval config's — the row is then an explicit transfer/off-config eval, not
  a silent mistake.
- **Shape-incompatibility abort (review P2, 2026-06-15).** A fingerprint mismatch can
  also be a *shape* mismatch (actor layers sized to a different spec); scoring anyway
  crashed deep in the actor on a matrix-shape `RuntimeError`. `save_policies` already
  stores the actor dims; `checkpoint_obs_dims(path)` reads them, and `signal_check`
  compares them to the eval spec and **skips the trained row with a message** when
  incompatible — only same-shape transfer tests proceed past the warning.
- `train(held_out_seeds=N)` evaluates the *selected* checkpoint on
  `range(HELD_OUT_SEED_OFFSET, HELD_OUT_SEED_OFFSET + N)` (offset `10_000`), disjoint
  from the validation seeds `range(eval_seeds)` and from `signal_check`'s default
  `0..N`. The held-out line is an unbiased estimate of the chosen checkpoint, logged
  separately from the validation figure it was selected on.

## Considered options

- **Hard-reject a mismatched fingerprint at load (like the version check).** Rejected:
  off-config evaluation is a legitimate transfer test; a warning preserves intent
  while a hard error would block it. The `CHECKPOINT_VERSION` check still hard-rejects
  genuinely unloadable (wrong-schema) checkpoints.
- ~~**Fingerprint the entire config (all fields, costs, emissions).** Rejected: the
  goal is shape/economy *compatibility* for a trained actor, not byte-identity;
  hashing the spec-derived structure captures exactly what makes a checkpoint
  loadable+comparable, and tolerates irrelevant field reordering.~~ **ADOPTED after
  review P1.** The original rejection conflated "shape-compatible" with "same economy".
  A trained-policy *comparison* needs both: shape to load, AND economy to be a
  meaningful benchmark. The canonical `asdict` serialization still tolerates field
  reordering (`sort_keys`) — it hashes economic *content*, not byte layout.
- **Reuse the validation seeds for the final report.** Rejected: that is the bias
  this ADR removes.

## Consequences

- Checkpoints gain a `config_fingerprint` field (None when saved without a config);
  no `CHECKPOINT_VERSION` bump is needed — the field is additive and read defensively.
- The retrain report (#9) should quote the **held-out** number, not the validation
  number, as the headline result.
- `signal_check --policy --config X` now prints a warning instead of crashing or
  silently misreporting when the policy was trained on a different economy.
