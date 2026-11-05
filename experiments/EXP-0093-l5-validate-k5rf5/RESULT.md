# EXP-0093: M5 L5 validation of the two L4 candidates on k5_rf5 (task 5.3)

**Question.** The pre-registered L5 (v6 `5cd0e302…`): both seeds' generation-40 candidates against k5_rf5, on
unseen atlas v2 validation worlds.
- Worlds: the loaded validation worlds, 9 at 12 producers (deciding) and 3 at 42 (reported).
- 8 episodes × 2 replicates; references k4 and `entity_planner`; all with MWWk.
- **Readings, by precedence:** kill (below −F_val or extra deaths) → transfer_failure → success (both seeds >
  F_val at 12, no extra deaths, run-down / starvation shares ≤ the base's) → plateau.

**Setup.**
- Code 9cfac98. `producer_read --mode validate --runs s0 s1` (EXP-0092).
- Pod: secure 16 vCPU in **EUR-IS-1** (`y2z5jxvyntpl8f`), 23:08–23:14 UTC. 0 contract failures.

## Registered reading: **`kill`**, from one extra death

| 12 producers (9 worlds, 144 rows / arm) | − k5_rf5 (floor) | deaths | run-down share | starvation share | held units / cycle |
|---|---|---|---|---|---|
| k5_rf5 (base) | — | 0 | 0.681 | 0.778 | 283 |
| **cand-s0** | **+412.8** (60.6): clears | **1** | 0.681 | 0.750 | 502 |
| **cand-s1** | **+383.4** (53.3): clears | 0 | **0.688** (99 vs 98 rows) | 0.736 | 489 |
| k4 | −521.7 | 1 | 0.715 | 0.778 | 242 |
| `entity_planner` | −1,429.7 | 0 | 0.708 | 1.000 | 127 |

- **42 producers (reported):** cand-s0 +579.4 (floor 131.8), cand-s1 +556.7 (149.0). 0 deaths for every arm.
- **The extra death:** cand-s0 dies on atlas 12-0109, episode 1213081781, replicate 1, at cycle 46. k5_rf5 lives
  there; k4 dies there too.
- **Without the death:** cand-s1 still misses `success` by one run-down row (99 vs 98 of 144).
- **`stock_heavy` (reported, never deciding; to the supervisor):** flagged. The selected cand-s0 holds 950.5 vs
  the base's 602.4 per cycle over both sizes, a ratio of 1.58 > 1.5.

## Diagnosis of the death (`probes/l5_death_0109.py`, local, one episode)
- **What dies:** product 1 has two makers.
  - p0_1 has max capacity 1 (useless).
  - p0_0 has max 144, far below its sustainable ceiling (7%, restores 44 → 628), so this is **not the upgrade
    trap**.
- **Why p0_0 drains:** it receives **0 from the investment fund from cycle 40 on**. Its restoration orders expire
  unfunded (rejected at cycles 41 and 44) and it drains 11 per cycle: 103 at cycle 36, 0 at cycle 46.
- **Why the fund dries up:** the fund (2,200 per cycle) is split across the producers with funding orders, in
  proportion to their cut-off prices. Under the candidate, the network places 2–5 inventory expansions per cycle
  (θ_e; 0 under the base) on top of its upgrades. More producers compete for the split, and the low-capacity
  p0_0's share falls to nothing.
- **Under k5_rf5:** p0_0 is also unfunded in cycles 40–46, but entered the window at 98 after an earlier
  restoration completed, and outlasts it.
- **The same cost shows in aggregate:** investment rejection 0.22 vs 0.10 on this episode, and ≈ 3× on the
  monitor (EXP-0092).

## Conclusions
- **L5 `kill` (stop branch).** The candidates deliver large validated gains (+383 to +413 at 12, +557 to +579
  at 42), but seed 0 adds a death.
- **The mechanism is a side effect of the learned expansion trigger (θ_e):** extra investment orders dilute the
  fund, and a sole maker's restoration goes unfunded. New entry: DIAGNOSTICS 30.
- **Responses** (the supervisor's / user's call; a new pre-registration version each):
  - drop θ_e (train the buffer and extra buying only);
  - constrain the expansion trigger to cycles where the producer's own restoration is funded;
  - accept `kill` and close the arc.
- **Spend:** ≈ 1.5 vCPU-h billed (stop 7). M5-P ≈ 28.8 / 80.

## Discriminating check (DIAGNOSTICS 30 check 2, local): θ_e = 0 removes the death
- cand-s0 was replayed with its expansion correction zeroed (θ_k and θ_b kept) on all 16 episodes of 12-0109:
  **0 deaths**.
- On the killing episode: 3,187 units, alive (cand-s0: 1,216, dead; k5_rf5: 2,871). Investment rejection 0.08.
- On 12-0109 the world mean is k5_rf5 2,672, cand-s0 3,033 (+361), θ_e = 0 2,929 (+257): about 70% of the gain
  is kept on this world.
- The fund-dilution branch is confirmed on this world. Whether θ_e = 0 keeps the validated gain elsewhere is a
  new run's question.
