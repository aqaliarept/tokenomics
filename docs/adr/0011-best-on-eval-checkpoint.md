# Deploy the best-on-eval checkpoint, not the final policy

Track A. After fixing the critic (ADR-0007) and schedule (ADR-0008), every seed
exhibits the same shape: greedy-eval survival **peaks** mid-run (`dead 0.00`,
`ret > 4000`) and then **drifts back** toward collapse (`dead 1.00`) by the
final iter. The policy *finds* a surviving region on every seed we have tried;
it does not reliably *stay* there. Saving the final-iter weights therefore
throws away the best policy the run produced.

## Decision

`train()` tracks the best policy seen on the eval metric and checkpoints **that**,
not the final weights:

- Ranking is **survival-first**: minimize `eval_dead_rate`, then maximize
  `eval_return` — key `(dead_rate, -return)`, smaller is better. A 0%-death
  policy always outranks any dying one regardless of return.
- On each eval iter the current actor `state_dict`s are deep-copied if they beat
  the incumbent. `--save` restores the best snapshot before writing the
  checkpoint and logs `best eval: dead .. ret ..`.

This is standard practice — keep the best model on a validation metric. The eval
harness (ADR-0003) is exactly that validation signal, scored identically to the
baseline table, so "best on eval" is the honest deployed policy.

## Why this is the right deliverable, not a patch over instability

The mid-run collapse is a real stability question (high-variance PPO on a sharp
survival region; the ordering Gaussian explores with `std=1`). Best-checkpoint
does not *fix* the collapse — it decouples the **deliverable** from it: the
trained policy that ships is a genuine survivor scored on held-out eval seeds.
Hardening the collapse itself (lower exploration, smaller actor steps) is a
separate, optional lever tracked for later; it changes training dynamics, not
what "the trained policy" means.

## Trade-offs

- Holds two extra `state_dict` copies in memory for the run — negligible at smoke
  scale.
- Best is chosen on the eval seeds (`eval_seeds`), so it can mildly overfit that
  small set. Mitigated by scoring the saved policy on a *different/larger* seed
  set in `signal_check` before trusting it.
- Requires `eval_every > 0`; with eval off there is no signal to rank and the
  final policy is saved as before.
