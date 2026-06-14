# Target-KL early stop on the actor epochs

## Status
Accepted. The pre-correctness-fix result interpretation below is superseded by
ADR-0024, which makes `0.008` the validated default.

Track A. The mid-run collapse (ADR-0011) had two distinguishable causes. This
ADR fixes one of them and, in failing to fix the other, pins down what that
other cause actually is.

## Decision

`actor_update` takes an optional `target_kl`. Each PPO epoch, before stepping,
compute the Schulman approx-KL from the rollout policy
(`((ratio − 1) − logratio).mean()`, non-negative, low variance) and **break the
epoch loop** if it exceeds `1.5 · target_kl`.

Clipping alone does not bound the step — the ratio escapes the clip range on
states the batch under-covers — so PPO can take an over-large update that tips a
policy out of the survival basin, and plasticity loss then blocks recovery
(see [No Representation, No Trust](https://arxiv.org/abs/2405.00662)). The KL
break restores the trust region the clip only approximates.

## Historical result before ADR-0017 through ADR-0023

**Mechanism 1 — learn-and-hold collapse (FIXED).** Seeds that start *below*
survival and learn up used to peak then drift back. With target-KL they reach
`dead 0.00` and **hold to the final iter**. Robust across `target_kl` 0.008 and
0.015. Seeds 0 and 9 now finish at `ret 3545 / 3893` — genuine trained survivors
(not the init) that beat the bar.

**Mechanism 2 — collapse-from-good-init (NOT fixed, now diagnosed).** Seeds whose
*random init* already survives collapse within ~15–30 iters regardless of
target-KL. Halving the trust region (0.015→0.008) did not touch the iter-30
crash, so it is **not an over-large step**. A 10-seed sweep (critic + KL 0.015):

| init @ iter 0 | seeds | survive @ final |
|---------------|-------|-----------------|
| good (`dead 0`) | 1, 2, 5 | **0/3** |
| bad (`dead 1`)  | 0, 3, 4, 6, 7, 8, 9 | 2/7 (0, 9) |

Every surviving init is destroyed by training; every learned survivor comes from
a bad init. The collapse is **init-quality-driven, not a hyperparameter**: a good
init is a knife-edge the optimizer walks off. Mechanisms ruled out for it, each
with a run: trust-region size (this ADR), reward shaping (ADR-0016), exploration
noise (`log_std_init` −1.0 → −1.6 → −2.0, all collapse).

Those runs used corrupted cross-agent GAE targets, reset episodes at every
rollout cut, allowed producing action aliases, and omitted acting-agent context
from the critic. They remain useful investigation history but do not establish
the correct threshold for the repaired trainer.

## Corrected-trainer result

ADR-0024 reran one matched configuration after the correctness fixes. Target-KL
`0.008` finishes alive and above the baseline for seeds 1, 2, and 5. The default
is now `0.008`; `--no-target-kl` retains the historical unconstrained ablation.

## Trade-offs

- Target-KL bounds repeated PPO epochs; it does not repair bad advantages.
- The threshold remains a task-specific control, not a universal PPO constant.
- `--no-target-kl` is required for comparisons with the old update path.
