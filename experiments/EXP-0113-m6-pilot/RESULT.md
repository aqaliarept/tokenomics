# EXP-0113: the capped critic-free Stage 1 training pilot (pre-registration v10)

User decision 2026-11-06 (a capped pilot that overrides v9's `absent`, on EXP-0112's power evidence).
`m6_prereg_v10.json` (SHA `0123274c…`), code d1d8f54 (`train/m6_pilot.py`). Pod cctsg5g6uuw1j6 (secure 16 vCPU / 32 GB + 20 GB
volume, EUR-IS-1), 23:34–00:16 UTC ≈ **11.2 vCPU-h** (stop 22).
- **What learns:** the producer only, at 12 producers. The orderer is frozen at EXP-0100's weights; replenish is
  frozen; the gates are tempered (training only).
- **Update:** E3 at 96 seeds × 4 siblings per iteration; PPO-clip + KL anchor; the per-agent Stepper.
- **Iterations:** about 4 min each (rollouts 115–156 s). No stall on the pod; the local stall was memory pressure.

## Result: **`hold`, with a rising curve**
**Greedy-policy gate** vs the EXP-0100 stack (12 monitor world-episodes × r0 / r1, paired keys, M5's floors):

| gate | candidate units | EXP-0100 stack | g | floor | deaths | run-down share | starvation share |
|---|---|---|---|---|---|---|---|
| iteration 5 | 4,162.6 | 4,067.4 | +95.1 | 149.3 | 0 vs 0 | **0.469** vs 0.563 | 0.625 vs 0.656 |
| iteration 10 | **4,174.5** | 4,067.4 | **+107.1** | 154.1 | 0 vs 0 | **0.438** vs 0.563 | 0.625 vs 0.656 |

- Iteration 5: no early stop.
- Iteration 10: `hold` (|g| ≤ F); every side effect is better than the reference.

**Training curve** (stochastic rollouts with gate exploration):

| iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| return (normalised potential) | −38.8 | −42.7 | −38.2 | −29.3 | −27.7 | −31.3 | −18.6 | −23.5 | −16.7 | −20.1 |
| units / episode | 3,609 | 3,673 | 3,627 | 3,811 | 3,721 | 3,815 | 3,710 | 3,851 | 3,691 | 3,759 |
| deaths (of 384) | 2 | 2 | 2 | 0 | 1 | 3 | 0 | 0 | 0 | 3 |
| KL to the start | 0.002 | 0.005 | 0.007 | 0.012 | 0.017 | 0.021 | 0.026 | 0.029 | 0.027 | 0.027 |

- Clip fraction: 0.014–0.019.
- Flip rates: steady at restore 5.6–5.8%, upgrade 3.0–3.3%, expand 3.3–3.8%.

## Conclusions
- **`hold` under the pre-registered rule.** The greedy candidate is ahead of the EXP-0100 stack at both gates
  (+95, then +107 units, +2.6%), but inside the floor. Its run-down and starvation shares are lower, and it has no
  deaths.
- **The curve is rising, not converged.**
  - The training return improved by about 19–22 normalised units over 10 iterations (about 20% of a base episode,
    in the potential's units, including its penalties).
  - KL to the start grew steadily to about 0.027 and has flattened in the last three iterations.
  - The deaths in the training rollouts (0–3 of 384) come from exploration. The greedy gates show none.
- **Continuation is the user's call** (`hold` with a rising curve, per the decision record): for example, 20–30
  more iterations at about 1.1 vCPU-h each, gated at 15 / 20 / 30. The M6 tranche has ≈ 57 vCPU-h left.
