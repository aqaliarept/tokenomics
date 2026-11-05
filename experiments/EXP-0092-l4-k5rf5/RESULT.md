# EXP-0092: M5 L4 on k5_rf5, seeds 0 and 1 to 40 generations (task 5.2)

**Question.** The pre-registered L4 (v6 `5cd0e302…`): seed 0 resumed from EXP-0091's generation-20 checkpoint
to 40, and seed 1 from θ = 0 to 40. Verify that g_tr recomputes and that checkpoints and pins are recorded. L4
has no reading of its own; it licenses L5. Training-set numbers, not an acceptance read.

**Setup.**
- Both runs at code commit **2e6ec61**, the source pinned in seed 0's spec. Its package tree equals b3af893's;
  resuming needed the same pins. `producer_es --mode train --generations 40`.
- Two pods in parallel, secure, **EUR-IS-1**:
  - seed 0 on `t3rz3girkclaih` (8 vCPU; 16 was unavailable), 22:29–23:07 UTC, resumed;
  - seed 1 on `ktmtqah2dz6pz4` (16 vCPU), 22:29–23:04 UTC.
- 0 contract failures. 40 checkpoints each; spec status complete; prereg v6 pinned.

## L4 verify: passes. Both seeds reach 40 generations; g_tr recomputes from the monitor rows (`read_s*_g*`)

| monitor (16 world-episodes × 2 replicates) | gen 0 | 10 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| seed 0 g_tr (k5_rf5 3,871.4) | 0 | +405.1 | +467.5 | +545.8 | **+569.2** (F_tr 222.6) |
| seed 1 g_tr | 0 | +438.3 | +443.3 | +526.7 | **+573.7** (F_tr 224.1) |
| seed 0 / 1 centre held units / cycle (base 282.3) | 282 | 446 / 446 | 499 / 488 | 503 / 501 | 517 / 509 |
| seed 0 / 1 centre investment rejection (base 0.099) | 0.10 | 0.32 / 0.30 | 0.33 / 0.30 | 0.32 / 0.30 | 0.33 / 0.31 |
| deaths on the monitor | 0 / 32 at every monitored generation, both seeds | | | | |

**Per-generation minibatch diagnostics (reported only, `generations.csv`):**
- Held stock is about 350 at generation 5, then 425–555 from generation 10 on, for both seeds.
- Investment rejection is about 0.26–0.31 from generation 10 on.
- 0–1 disqualified members per generation.

## Conclusions
- **L4 is complete and verified.** The two seeds converge to about the same training gain: +14.7% on the
  monitor, about 2.5× the floor.
- **The stock / rejection side effects are stable,** not growing: held stock about 1.8× the base's,
  investment rejection about 3×.
- **L5 (validation at 12 and 42 on unseen atlas worlds) is licensed and launched next.** The `stock_heavy`
  flag goes to the supervisor with L5's read.
- **Spend:** ≈ 4.9 (seed 0) + 9.1 (seed 1) = 14.0 vCPU-h billed (stops 14 / 27). M5-P ≈ 27.3 / 80.
