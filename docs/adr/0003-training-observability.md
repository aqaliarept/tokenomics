# Training observability: per-iteration evaluation curves

Phase 1.5 P1. The smoke trainer logged only `mean_step_reward` and the two PPO
losses per iteration. Step reward is a poor progress signal: it mixes the two
shared policies, is dominated by the reward scale, and says nothing about the
metrics we actually care about (consumer fulfilment, survival, death). We could
not tell whether a run was learning the *task* or just lowering loss.

## Decision

Each training iteration (every `eval_every` iters) we snapshot the current
policies as **greedy** deterministic eval policies and run them through the
existing `eval/` harness (`evaluate`) over a small fixed seed set. The harness
already produces the Phase 1.5 P1 metrics — completion rate, rejection rate,
survival length, capacity trajectory (`cap_min`), death rate, return — so
training reuses the *same* measurement code as `signal_check`. One source of
truth for "how good is this policy", whether the policy is a baseline or RL.

`train()` returns a list of per-iteration records (`IterStats`) instead of a
list of floats, and optionally writes them as a CSV curve for offline plotting.

## Why greedy snapshots, not rollout reward

The collection rollout samples stochastically (exploration) and reports reward
*during* learning. The thing we want to chart is the *deployed* policy: argmax
over the producing `Discrete(4)`, distribution mean for the ordering `Box`. This
matches how the trained policy is deployed for real runs and makes the
RL curve directly comparable to the baseline table (target: beat
`restore_when_low` + `even_split`).

## Why reuse the eval harness instead of instrumenting the rollout

The rollout buffers are bucketed by role and reward-shifted for GAE; extracting
clean episode metrics from them is fiddly and would drift from the baseline
definitions. The eval harness is white-box, deterministic, and already tested.
Cost is a few short episodes per logged iter — negligible at smoke scale, and
`eval_every` throttles it for longer runs.

## Trade-offs

- Extra envs per logged iteration. Bounded by `eval_every` and `eval_seeds`.
- `train()` now depends on `eval/` (one-way; `eval/` never imports `train/`).
- Greedy eval hides exploration collapse — entropy/loss still logged alongside.
