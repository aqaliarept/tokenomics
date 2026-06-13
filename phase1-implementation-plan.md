# Phase 1 Implementation Plan — Python PettingZoo AEC Environment

Port the Go domain (`emulation/domain/`) to pure Python, expose as a PettingZoo AEC environment for MARL training. Per ADR-0001 and ADR-0002.

The env, port, and a MAPPO smoke loop are built (ADR-0001–0006): observable per-iter eval curves, centralized critic, vectorized rollout, difficulty curriculum. The loop learns but does not yet survive — it stalls at `dead 1.00`. **Now on Track A: tune the loop until the trained policy clears the baseline bar.**

**Target: beat `restore_when_low` + `even_split` — return ≈ 3230, ≈ 359 consumer completions, 0 % death over horizon 100.**

## Track A — crack Phase 1 (beat the baseline bar)

Stay in this env; make the policy survive. Machinery is done — this is a tuning/scale problem. Ordered by expected value:

| A | Work | Why |
|---|------|-----|
| **1** | **Fix the critic.** `closs ~1000` = critic underfit, the current bottleneck. Value-target normalization (running return mean/std), wider critic (128–256), more critic epochs, optional GAE return clipping. | Advantages are only as good as the critic; noisy value → wandering actors |
| **2** | **Longer runs + LR schedule.** 200–500 iters, anneal actor LR, decay entropy coef. The regression once curriculum pins at diff 1.0 smells like too-high LR late. | Smoke scale (20 iters) can't converge |
| **3** | **Gate the curriculum on eval, not iters.** Hold each difficulty until `dead` drops, then ramp (`warmup_frac` 0.8–0.9 as fallback). | Stops difficulty outrunning the policy |
| **4** | **Reward-shaping check.** Confirm investment's `+1` is enough signal that restoration pays before death; consider a survival / capacity-floor shaping term. | If survival never pays in time, no schedule fixes it |

Start at A1 — every downstream advantage depends on a trustworthy critic. Each step: write/extend an ADR for the design change, keep tests green, eyeball the eval curve against the target bar.

**Post-Phase-1 (separate arc):** the deep-research 6-phase roadmap — RL-per-decision-block, graph/history critic, investment & durability, planner/regulator. Not Track A.
