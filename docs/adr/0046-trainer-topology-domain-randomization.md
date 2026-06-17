# Topology domain-randomization as a supported trainer path

Status: accepted

## Context

Held-out scale generalization (the active goal) requires training on a *distribution* of
scale topologies, not one fixed economy — the open-loop-overfit finding (ADR-0038/0039,
memory `generalization-overfit`). That training existed only as a scratch closure in
`_scale_lab.py`: it built a fresh `build_scale_config(seed)` topology each iteration, held it
for `dwell` iters, and re-applied the curriculum easing by hand. The plan (item 2) requires
this to be a first-class, tested trainer path so checkpoint selection and A/B runs are
reproducible.

The scratch closure handled two subtleties:
1. **Easing re-application.** `train()` eases `base_cfg` via `staged_config(base, d)`. A
   full-difficulty random topology discards that easing, so survival never bootstraps. The
   closure re-eased the random topology itself, reading the curriculum's private `_gated_d`.
2. **Dwell.** A new topology every iter changes the rollout config every iter, forcing
   env rebuilds; in-progress episodes never reach horizon, so the gate starves (no episodes
   → no evidence → difficulty pinned). Holding one topology for `dwell` iters keeps envs
   stable so episodes complete and the gate gets evidence.

## Decision

Add a `topology_random` provider hook to `train()`, distinct from the existing
`domain_random` (axis-perturbation DR). It is applied **before** staging:

```
active_base = topology_random(base_cfg, rng) if topology_random else base_cfg
cfg = staged_config(active_base, d) if curriculum else active_base
```

Siting the provider before `staged_config` makes subtlety 1 structural: the trainer's own
curriculum difficulty `d` eases whatever topology the provider returns — no private
`_gated_d` read, no hand re-easing, and it is correct for both gated and iter-scheduled
curricula. A `topology_dr(depth, dwell)` factory owns subtlety 2 (dwell), returning a fresh
`build_scale_config` topology and holding it for `dwell` iters.

CLI: `--topology-random`, `--topology-depth`, `--topology-dwell`. When `--topology-random` is
set with no explicit `--config`, the base defaults to `build_scale_config(0, depth)` so the
net/critic — built once on `base_cfg` — match the DR stream's dims (a phase1-sized net would
dim-mismatch the first rotated topology's global state).

`_scale_lab.py` now delegates to `topology_dr` instead of its own closure, so the scratch and
supported paths cannot drift.

## Considered options

- **Extend `domain_random` to also do topology resampling.** Rejected: topology resampling
  (a new economy with the same dims) and axis perturbation (scaling an existing economy's
  degradation/capacity/cost) are different concepts; overloading one hook forces the topology
  path to re-ease by hand. A separate before-staging hook keeps easing in the trainer.
- **Pass the difficulty `d` into the `domain_random` callback.** Rejected: changes the
  callback contract for the existing `--domain-random` path; the before-staging placement
  achieves the same correctness without touching that contract.

## Consequences

- Topology DR is a tested trainer path: `test_topology_dr_holds_each_topology_for_dwell_iters`
  (dwell), `test_topology_dr_preserves_obs_action_dims` (dims-preserving), and
  `test_train_runs_with_topology_dr_and_curriculum` (end-to-end staging). CLI runs verified.
- Easing re-application is correct by construction for any curriculum mode, removing the
  scratch closure's gated-only `_gated_d` coupling.
- Full suite green. The provider is dims-preserving, so one net/critic spans the DR stream.
