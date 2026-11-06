# Sizing (re-sized by the L0 timing probe before any pod)

**Rates:**
- k5_rf5 + MWWk, measured: about 0.8 s per 12-producer episode and 15 s at 42 on a pod.
- GNN policies, measured (notes/timing.md): 4.4× the base per 12-producer episode and 3.4× at 42, locally; about 1 ms per decision.
- Stops are 2× the estimate, with a 40 vCPU-h cap per run.

| stage | content | estimate vCPU-h | stop |
|---|---|---|---|
| L0 | code checks and the timing probe, local (seconds) | 0 | — |
| L0b | Stage I: collect labels on 112 training world-episodes, BC fit (Hogwild × 16, 20 epochs), imitation check on 32 + 32 monitor episodes | 8 | 16 |
| L1 | bar + noise floor (MWWk from EXP-0090 if pins match; RR new) | 1.5 | 3 |
| L2 | probe, 2 arms × 2 minibatches | 6 | 12 |
| L3 | pilot, seed 0, 25 iterations × 256 episodes | 15 | 30 |
| L4 | seed 0 to 50 iterations (+15), seed 1 (30) | 45 | 30 + 40 |
| L5 | validation, 2 candidates + references, 12 / 42 | 5 | 10 |
| L6 | held-out, 12 / 42 | 5 | 10 |
| **success path** | | **≈ 86** | |
| LA (gated) | the acceptance diagnostic (local) + L3–L5 with the accept head | ≈ 45 | |

**Tranche M6 = 150 vCPU-h** (user decision 2026-11-06): the success path, the gated acceptance stage and about
20 of margin. The supervisor decides within it; launch-fit on billed spend.
