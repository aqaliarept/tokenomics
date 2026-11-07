# EXP-0116 pre-registration: why EXP-0114's training return rises while greedy units do not

Frozen before any data (supervisor 2026-11-07, item 2; gates task 4.5, not 4.3 / 4.4). Driver
`tokenomics/train/m6_gap_probe.py`. Never an acceptance read.

## Question
EXP-0114's greedy candidate at iteration 40 is +77.7 units per episode ahead of the EXP-0100 stack, inside the floor
of 116.6 (`hold`). Over iterations 10 → 40 the training return rose from about −20 to −6. Why does the training
signal not show in the gate?

## Evidence before data (local read, 2026-11-07)
- **Recorded artifacts:** only the gate aggregates and the reference rows. There are no candidate rows and no
  per-episode Φ terms, so replays are needed.
- **Replenish drift:**
  - **In training:** input buying is the frozen start's deterministic output (`pm0`).
  - **In the gate:** the candidate uses its own replenish head (`gnn_producing_policy(pm)`). That head had no
    gradient, but it reads the shared trunk, which RL moved.
  - **Measured** on two monitor world-episodes at horizon 20 (480 producer decisions), mean |Δ replenish
    fraction| against the start:

    | iteration | absolute | relative |
    |---|---|---|
    | 10 | 0.026 | 15% |
    | 20 | 0.054 | 25% |
    | 30 | 0.092 | 36% |
    | 40 | 0.090 | 33% |

  - Over the same span, the greedy gates disagree with the start on 1.4–4.1% of decisions.

## Arms
**Common setup:**
- Every arm plays the same frozen orderer, which is identical in the start and in EXP-0100.
- Monitor set: 16 world-episodes × replicates 0 / 1, the gate's keys.
- Horizon 100, 12 producers.

**The arms:**
- `ref`: the EXP-0100 stack (the gate's reference).
- `start`: the RL start (EXP-0110 `start_models.pt`), greedy.
- `G{10,20,30,40}`: the candidate as gated: greedy gates, its own replenish.
- `F{10,20,30,40}`: greedy gates with the start's replenish. This is the trained policy, made deterministic.
- `T_start`, `T40`: tempered gates (v9 TEMPS) with the start's replenish, the training policy. One seeded draw
  per row.
- On the 112 training world-episodes × replicates 0 / 1: `ref`, `start`, `G40`, `F40`, `T_start`, `T40`.

**Per row:** the gate row, units, and the raw Φ terms, scaled as in training. Unit scale = H / bar, and λ, μ, ν, X, Y
come from EXP-0110's calibration.

**Sizing:**
- 1,760 episodes at ≈ 9 s ≈ 4.4 vCPU-h.
- Pod: secure 16 vCPU with a persistent volume, EUR-IS-1. **Stop 8**, M6-A.

## Validity
- `G40`'s monitor mean units must equal EXP-0114's iteration-40 gate (4,145.125).
- `ref` must equal EXP-0114's `ref_rows.json` row for row.
- Otherwise the run is invalid.

## Reading
g / F use M5's floors rule (the gate's) for an arm against a base on one set. Δ is a paired mean over shared keys,
with SE over world-episode means.

**Flags (all reported):**
1. **`replenish_drift`:** Δ(F40 − G40) on the monitor > ½ F(G40 vs ref), and g(F40 vs ref) > F. The deployed
   candidate's replenish loses what the trained policy gained.
2. **`greedy_loses_it`:** g(T40 vs T_start) > F on the monitor, and g(F40 vs start) ≤ F. The tempered gain does
   not survive greedy play.
3. **`not_units`:** on the training set, the tempered gain in Φ is > 2 SE, its unit term is < ½ of it, and
   g(T40 vs T_start) ≤ F. The return gain is penalty terms (run-down, starvation, death), not units.
4. **`monitor_power`:** on the training set, Δ(F40 − start) > 2 SE, and g(F40 vs start) ≤ F on the monitor. A
   real gain the 32-row gate cannot resolve.

**Primary:** the first that holds, in this order; else `unexplained`.

**Consequences:**
- **`replenish_drift`:** deploy and gate with the start's replenish, or train replenish (a design decision
  before 4.5). Add a DIAGNOSTICS entry.
- **`greedy_loses_it`:** gate the stochastic policy, or anneal the temperature in training.
- **`not_units`:** re-weight Φ (λ / μ) before 4.5.
- **`monitor_power`:** a larger gate set for 4.5's reading.
