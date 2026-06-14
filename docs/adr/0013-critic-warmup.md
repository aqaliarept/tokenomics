# Critic warm-up — freeze actors for the first K iters

Track A, explored then **shelved** in favour of ADR-0015.

## Decision

`--critic-warmup K`: skip both actor updates for the first `K` iters while the
critic still fits every iter, so the first policy-gradient steps fire only once
the value baseline is warm. Motivation: good-init seeds collapse in the first
~30 updates; suspect was destructive early actor steps on not-yet-warm
advantages.

## Result

For seed 1 (good init), warm-up produced a genuine survivor at the first
post-warmup update (`iter 15: dead 0.00, ret 3367`) which `best-on-eval` banks —
but the policy collapsed again immediately after. So warm-up does **not** stop
the collapse; it only times the first good step where the checkpoint can catch
it. Worse, it **regressed** seed 0, whose learn-up was delayed/destabilised by
the frozen start.

Superseded as a stability lever by target-KL (ADR-0015), which fixes the learn-up
collapse directly. The flag remains for experiments; default 0 (off).
