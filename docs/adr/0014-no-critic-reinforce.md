# Optional no-critic mode — REINFORCE with whitened MC returns

Track A, a probe kept as a diagnostic, not the default.

## Decision

`--no-critic` drops the learned centralized critic entirely. With `val=0` and
`boot=0`, GAE collapses to a discounted Monte-Carlo return-to-go, and the
existing advantage whitening in `actor_update` (subtract mean / divide std)
supplies the batch-mean baseline. No critic net, no `ValueNorm`, no value
bootstrap. Built to test whether the critic↔actor coupling — not a missing
baseline — drove the collapse-from-good-init.

## Result

It does not isolate a single cause; it **flips** which seeds work. No-critic gave
seed 1 its strongest survivor (`iter 10: dead 0, cons✗ 0, ret 4399`, beating the
critic's 3367) but broke seed 0 (best `dead 1, ret 1460`). Critic + warmup is the
mirror image. So the critic is not the culprit; advantage variance only shifts
which init lands in the basin. The decisive result came later from the 10-seed
init-quality sweep (ADR-0015): the collapse tracks init quality, not the baseline.

Kept behind the flag as a cheap REINFORCE reference. Default off (critic on).
