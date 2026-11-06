# L0 timing probe (2026-11-06, local box, one process, torch 1 thread; untrained GNNs)

`python -m tokenomics.train.m6_imitation --mode timing --horizon 12` (k5_rf5 + MWWk vs the GNN stack):

| world | stack | horizon-12 episode | producer decision | orderer decision | decisions (producer / orderer) |
|---|---|---|---|---|---|
| atlas 12-0111 (12 producers) | k5_rf5 + MWWk | 0.14 s | 0.09 ms | 0.04 ms | 144 / 480 |
| | GNN | 0.62 s (4.4×) | 0.95 ms | 0.74 ms | |
| atlas 42-0000 (42 producers) | k5_rf5 + MWWk | 1.45 s | 0.17 ms | 0.04 ms | 504 / 3,168 |
| | GNN | 4.96 s (3.4×) | 1.07 ms | 0.89 ms | |

**At horizon 100 (extrapolated per cycle):** the GNN stack takes about 5.2 s per 12-producer episode and about
41 s at 42, locally. Pods ran about 2× faster than this box in M5.

**Stage I (one training world-episode, horizon 100):**
- **Collection:** 2.1 s, 1,200 producer decisions and 1,750 kept orderer decisions. Orderer decisions are
  dropped when no slot has more than one eligible producer: from 5,400 to 1,750.
- **BC epoch:** 7.9 s single-process, 3.4 s Hogwild × 2 on this 2-core box.
- **For 112 world-episodes on 16 vCPU (Hogwild × 16):** about 1 min per epoch, about 20 min for 20 epochs.

**Re-sizing:** L0b (collect + fit + check) ≈ 8 vCPU-h (stop 16). RL stages: PPO collection at about 3× the
base's per-episode cost at 12 producers on the pod, plus a PPO update of similar cost, so the L2–L4 estimates
stand.
