# Value-target normalization and a wider critic

Track A step 1 (Phase 1 plan). The centralized critic (ADR-0004) would not fit:
`closs ~1000`, flat across iterations, while the actors barely moved and the
greedy policy stalled at `dead 1.00`. PPO advantages are only as good as the
critic baseline — a critic that never converges gives noisy value estimates, so
the actors chase noise and never learn to survive.

## Why the critic underfit

The fit target is the GAE return: a discounted sum over the (up-to-horizon)
trajectory. Even with the optimization-time `REW_SCALE = 0.05`, those returns
reach O(100) with large variance. A plain MSE on raw returns makes the value
loss O(1000), so the critic gradient is dominated by absolute return magnitude
rather than relative fit, and a 64-wide net can't track it.

## Decision

**Normalize the value target (`ValueNorm`, PopArt-lite).** Track a running
mean/std of returns via EMA (`beta = 0.99`). The critic head learns the
*normalized* return; `critic.value()` denormalizes back to return space so GAE
and advantages stay in raw units. EMA (not a frozen fit) so the statistics
follow the return distribution as the policy improves.

- Targets clamped to ±5σ before the loss so a stray large return can't spike the
  gradient (the "return clipping" option in the plan).
- Reported `closs` is now the **normalized** MSE — an O(1) fit-quality signal,
  not the raw return magnitude.

**Wider critic + more epochs.** Hidden 64 → 128; critic epochs 4 → 8. The actors
are unchanged (decentralized, deployed as-is) — only the centralized training
critic grows.

## Result

`closs` drops from ~1000 (flat) to ~2 and decreasing within 5 iters; the value
loss is now a usable convergence signal. Survival (`dead`) is unaddressed here —
that is Track A steps 2–4 (longer runs + LR schedule, eval-gated curriculum,
reward shaping). This step only makes the critic — and therefore the advantages
every downstream step depends on — trustworthy.

## Trade-offs

- `ValueNorm` adds critic state (`mean`, `var` buffers) that must persist across
  iterations. Carried on the critic module; serializes with it.
- EMA `beta` is a new knob. Fixed at 0.99; not exposed as a CLI flag until a run
  shows it matters.
- Still a smoke trainer — no minibatching, single critic MLP (ADR-0004 caveats
  stand).
