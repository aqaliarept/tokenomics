# Sizing (draft; re-sized by L0's timing probe)

**Rates (measured):** the k5_rf5 + MWWk episode is about 0.8 s per 12-producer episode on a pod and 15 s at 42
(EXP-0088 / 0090). The GNN forward is assumed at 3× the hand-written decision cost, re-measured in L0. Stops are
2× the estimate.

| stage | content | estimate vCPU-h | stop |
|---|---|---|---|
| L0 | local code and a timing probe (seconds) | 0 | — |
| L1 | bar + noise floor (k5_rf5 + RR new; MWWk from EXP-0090 if pins match) | 1.5 | 3 |
| L2 | probe, 2 arms × 2 minibatches | 4 | 8 |
| L3 | pilot, seed 0, 25 iterations × 256 episodes | 10 | 20 |
| L4 | seed 0 to 50 iterations (+10), seed 1 (20) | 30 | 40 + 40 |
| L5 | validation, 2 candidates + 3 references, 12 / 42 | 4 | 8 |
| L6 | held-out, 12 / 42 | 4 | 8 |
| **success path** | | **≈ 54** | |
| LA (gated) | the acceptance diagnostic (local, 0) + L3–L5 with the accept head | ≈ 40 | |

**Request: tranche M6 = 120 vCPU-h** (the success path, the gated acceptance stage and one re-run), with
launch-fit on billed spend and the 40-per-run cap.
