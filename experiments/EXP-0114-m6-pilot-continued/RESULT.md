# EXP-0114: the Stage 1 pilot continued to iteration 40 (pre-registration v11)

`m6_prereg_v11` (SHA `f55aea40…`), code 1918721 (`train/m6_pilot.py --prereg v11`).
- **Start:** EXP-0113's iteration-10 checkpoint (SHA `e346b769…`); every v10 setting unchanged.
- **Setup:** 96 seeds × 4 siblings per iteration, PPO-clip + KL anchor, producer only at 12 producers.
- **Frozen:** the orderer (EXP-0100) and replenish.

**Runs.** One pod, 89blmrhmt5hfmx (secure 16 vCPU / 32 GB, EUR-IS-1). Iterations took ≈ 5.7 min, against
EXP-0113's ≈ 4, so 11–40 did not fit one 150-minute run. The supervisor split it, each run ≤ 40 vCPU-h:
- **Run 1, iterations 11–35:** 06:48–09:11 UTC ≈ **38.1 vCPU-h**. Stopped cleanly after iteration 35's
  checkpoint and pulled.
- **Run 2, iterations 36–40:** 09:11–10:03 UTC ≈ **13.8 vCPU-h**.
  - The pod had no persistent `/workspace`, so the restart wiped the out dir and the first launch began again
    at iteration 11.
  - It was killed after one iteration (≈ 25 min of billing lost) and relaunched with run 1's iteration-35 state
    uploaded from the local copy (checkpoint SHA `7f1ee39a…` equal).
  - It resumed at iteration 36.
  - `curve.jsonl` lines 11–35 equal run 1's, and so does `candidate_it30.pt`.
- **Total ≈ 51.9 vCPU-h** (tranche M6).
- **Validity:** the resume pins hold, and EXP-0100's reference rows are reproduced (`ref_reproduction` MATCH).

## Result: **`hold`**
**Greedy-policy gate:** the candidate against the EXP-0100 stack, on the 16 monitor world-episodes × r0 / r1, with
paired keys and M5's floors (g = mean unit difference, F = floor). Only iteration 40 decides; 15 / 20 / 30 are
stop-only.

| gate | candidate units | EXP-0100 stack | g | F | deaths | run-down share | starvation share | reading |
|---|---|---|---|---|---|---|---|---|
| 10 (EXP-0113) | 4,174.5 | 4,067.4 | +107.1 | 154.1 | 0 vs 0 | 0.438 vs 0.563 | 0.625 vs 0.656 | hold |
| 15 | | 4,067.4 | +135.1 | 139.0 | | | | continue |
| 20 | 4,191.2 | 4,067.4 | +123.7 | 100.1 | 0 vs 0 | 0.406 vs 0.563 | 0.563 vs 0.656 | continue |
| 30 | | 4,067.4 | +2.3 | 115.1 | | | | continue |
| **40** | **4,145.1** | 4,067.4 | **+77.7** | **116.6** | 0 vs 0 | **0.406** vs 0.563 | **0.594** vs 0.656 | **hold** |

At iteration 20, g was above F, but stop-only gates are not read.

**Training curve** (stochastic rollouts, 384 episodes per iteration):
- **Return:** −16.6 at iteration 11, about −13 over 12–17, about −6 to −9 over 18–36, then −2.0 / −4.9 / −6.6 / −5.6
  over 37–40.
- **Units per episode:** 3,631–4,009, with no trend after iteration 23.
- **Training deaths:** 14 of 11,520 episodes over iterations 11–40.
- **KL to the start:** 0.034 → 0.156, still rising (no plateau).
- **Clip fraction:** 0.013–0.021. Flip rates steady at restore 5.6–5.9%, upgrade 2.5–3.0%, expand 2.8–3.5%.

## Conclusions
- **`hold` under the pre-registered rule.** At iteration 40 the greedy candidate is +77.7 units per episode (+1.9%)
  ahead of the EXP-0100 stack, inside the floor of 116.6. It has no deaths, and lower run-down and starvation
  shares.
- **30 more iterations bought no measurable unit gain over iteration 10:**
  - g moved +107 → +78, and was +2 at iteration 30, all inside the floors;
  - the training return kept improving (−20 → −6), and so did the side effects.
- **The gap between the training signal and greedy units is not diagnosed here.** Candidates:
  - the return measures the potential, not units;
  - greedy play differs from tempered training play;
  - a 32-row gate cannot see a +2% change (F ≈ 115).

  Before more Stage 1 compute, the discriminating check is to read the per-row gate deltas and the potential's
  terms (DIAGNOSTICS 34 / 35).
- **For the ladder (tasks §4):** 4.0's start is EXP-0114's candidate only on `gain`, so 4.3 starts from
  EXP-0100's producer.
- **M6 tranche:** 93.0 + 51.9 = 144.9 of 150 vCPU-h. About 5 are left.

**Addendum (EXP-0116, 2026-11-07):** the gap is diagnosed. The gate scored the candidate with its own replenish head,
which followed the RL-moved trunk. Training used the start's replenish. The trained policy (greedy gates + the start's
replenish) is +230.1 units per episode at iteration 40 (F 136.4), and the drift cost about 150 (DIAGNOSTICS 37). The
pre-registered reading stays `hold`.
