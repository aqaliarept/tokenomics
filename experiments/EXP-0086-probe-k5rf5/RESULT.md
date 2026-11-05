# EXP-0086: M5 L2 probe on the final producer k5_rf5, pre-registration v6 (task R-3)

**Question.** At θ = 0 on k5_rf5, do the ES gradient estimates on two disjoint minibatches of 24 training
world-episodes point the same way? Arms S (σ 0.05) and S2 (2σ). The pre-registered reading is v6 `5cd0e302…`:
cosine ≥ 0.3 is `signal`, ≥ 0.1 is `weak`; arm S licenses the pilot, and the step reading sets its σ. Fresh
rows, not EXP-0081's. Not an acceptance read.

**Setup.**
- Commit b3af893. `producer_es --mode probe`.
- Inputs: EXP-0090's bar, normaliser, margins and re-keyed rows. Flag-count rule T = 5/3; penalised death
  P = B_we.
- 2,304 episodes, horizon 100, MWWk.
- Pod: secure 16 vCPU in **EUR-IS-1** (`50fps99ilp6z46`), 21:59–22:04 UTC.

## Registered reading: **`signal`**; step reading **`heterogeneity`** (pilot at σ)

| arm | cosine | disqualified A / B | fitness mean A / B |
|---|---|---|---|
| S (σ 0.05) | **0.967** | 2 / 1 | 1.0206 / 1.0218 |
| S2 (σ 0.10) | **0.947** | 1 / 0 | 1.0236 / 1.0151 |

- **No member dies** (0 of 2,304 episodes), against 24 of 48 on 12-0016 in EXP-0081 under rf3. The upgrade
  trap is gone.
- The guard disqualifies at most 2 of 24 members per minibatch. There were 0 contract failures and all 2,304
  rows are present.
- **Mean member fitness is about +2% over the bar** on both minibatches: random small steps already help
  slightly on average.

## Conclusions
- **The pilot (L3, task 5.1) is licensed:** 20 generations, opt seed 0, σ 0.05, under v6.
- DIAGNOSTICS 25's chain is resolved. The earlier `no_signal` readings (EXP-0081..0083) came from rf3's upgrade
  trap (DIAGNOSTICS 29), as fixed by the final producer, and from the guard design (the v3 flag-count rule).
  They did not come from the landscape.
- **Spend:** ≈ 1.3 vCPU-h billed (stop 4). M5-P ≈ 8.3 / 80.
