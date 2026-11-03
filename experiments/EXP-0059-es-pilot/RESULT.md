# EXP-0059: ES pilot, A1 seed 0, generations 1–20 (pre-registration v2)

**Status: complete, valid. Pilot reading (D7): `unresolved`.** On the training-monitor set,
g_tr = **+29.8** units, which does not clear F_tr = **54.75**. Per D7 the run stops for
diagnosis: no main runs, and DIAGNOSTICS 27 is opened. Task 6.2 of `add-m4-orderer-redesign`.
- **Driver and commit:** `order_es --mode train --arm A1 --opt-seed 0 --generations 20 --prereg
  …/es_prereg_m4_v2.json` at `7b16e55aa1d1…`, verified clean, from a clean worktree.
- **Inputs:**
  - pre-registration v2 `511e939a4121…`, so minibatch 24 and the D6 noise margin;
  - EXP-0058's bar (all bar hashes and normaliser `2d574ec8a3df…` pinned) with its A0 rows;
  - EXP-0069's `probe.json` (`a556cb78…`, `weak`);
  - EXP-0012 `orderer_scorer.pt` (`d25388bf…`).
- **Run:** 20 generations × (24 members + the center) × 24 world-episodes, plus the monitor
  set (16 world-episodes, R = 2) at generations 0, 10 and 20. Horizon 100, `entity_planner`.
  One fresh **secure 8 vCPU** pod; cpu32 and then cpu16 were refused for lack of capacity.
  ≈ 7.8 vCPU-h billed.

## In plain English

**The question.** The pilot asks whether 20 ES generations make the per-order router measurably
better than where it started. Its starting point is A0, the unchanged trained orderer's whole
decode. "Measurably" is pre-registered: the gain on a fixed set of 16 training world-episodes,
each played twice, must exceed that comparison's noise floor F_tr. If it does (`feasible`), the
main runs train to generation 40. If not (`unresolved`), the cause is diagnosed first.

**What happened.**
- **The router improved on that set, but not by enough to count.** Consumer units on the
  monitor set rose from A0's 1,949.9 to 1,962.6 at generation 10 and 1,979.7 at generation 20.
  That is a gain of **+29.8 units (≈ 1.5%)** against a floor of **54.75** (≈ 2.8%).
  - The floor's two parts agree: the atlas-spread floor is 54.75 and 3 × the standard error of
    the run's own paired differences is 54.59.
  - So the reading is **`unresolved`**.
- **The gain is not broad.** At generation 20, 7 of the 16 monitor world-episodes are above A0
  and 9 below. Two of them (+241 and +284 units) carry the mean, and the median world is
  slightly below A0. At generation 10 the split was 8 up, 8 down.
- **It was still rising.** The center gained +17.1 units over the last 10 generations, after
  +12.7 over the first 10. More generations might clear the floor, but the pre-registration
  sends an unresolved pilot to diagnosis, not to more training.
- **Training itself ran cleanly.** No contract failures or non-finite policies; every
  generation's members shared their worlds' demand and shock draws. The noise margin worked as
  intended: 0–15 of 24 members disqualified per generation (median 8), where EXP-0058's v1 rule
  had tied most member scores (A1 37 / 48).

**What changes because of it.** The main runs (EXP-0060, group 7) do not launch. DIAGNOSTICS 27
records the competing explanations and the checks that would tell them apart, because more
training, a more powerful monitor set, and a world-dependence problem each call for a different
next step. The choice goes to the supervisor and the user.

## Details

### Monitor set (`monitor_summary.csv`; read offline with `order_es_validate.run_stats`)

| generation | center units | A0 units | g_tr | F_tr | improvement over 10 gens | worlds up / down |
|---|---|---|---|---|---|---|
| 0 | 1949.91 | 1949.91 | 0.0 | — | — | — |
| 10 | 1962.63 | 1949.91 | +12.72 | 54.75 (own 3·SE 45.00) | +12.72 | 8 / 8 |
| **20** | **1979.69** | 1949.91 | **+29.78** | **54.75** (own 3·SE 54.59) | **+17.06** | **7 / 9** |

- **Per-world-episode center − A0 at generation 20** (16 values): −66.5, −66.5, −45.5, −39.5,
  −36.5, −23.5, −20.0, −16.5, −4.0, +9.0, +41.5, +51.0, +80.5, +88.0, +241.0, +284.0.
- **D7 pilot rule:** `feasible` iff g_tr > F_tr, so 29.78 ≤ 54.75 reads **`unresolved`**.

### Training (`generations.csv`)
- Center fitness (share of the training bar, on each generation's own minibatch): 0.62 at
  generation 1, 0.66 at generation 20, ranging 0.56–0.72. Each generation draws a different
  minibatch, so the series is not a learning curve.
- Disqualified members per generation (of 24): 8, 10, 5, 15, 2, 15, 5, 12, 7, 8, 9, 1, 8, 0, 9,
  6, 4, 9, 15, 5. ES estimate norms 5.3–11.5; final θ `ba35f8d9a6e9…`, τ 1.21.

### Verification (task 6.2's verify line)
- `contract_failures.csv` is header only.
- The run pins EXP-0058's bar: every bar hash and the normaliser equal `load_inputs(EXP-0058
  bar)`. The pre-registration is v2 (`disqualification.prereg_version` 2, minibatch 24), commit
  `7b16e55…`.
- **Members share digests:** 12,000 member rows in 480 generation × world-episode groups, every
  group with one (demand, shock) digest pair.
- **g_tr recomputes:** `order_es_validate.load_candidate` / `run_stats` from `monitor.csv`
  gives 29.78125 = the run's `monitor_summary.csv`.
- **Resume pins:** the generation-20 checkpoint `checkpoints/gen-0020.json` holds θ
  `ba35f8d9a6e9…` = `candidate.json`, and `spec.json` holds the training pins.

### Cost

| pod | shape | life (UTC) | billed |
|---|---|---|---|
| `vrhxob67krcem6` | secure 8 vCPU | 19:29:25 launch → terminated by 20:28:07 (≤ 58.7 min) | **≈ 7.8 vCPU-h** (upper bound) |

- **Capacity.** cpu32, retried once, and cpu16 were refused: "no instances available", HTTP 500,
  no pod created, nothing billed. cpu8 follows the larger-pods-first rule.
- **Measured rate:** ≈ 168 s per generation on 8 workers (600 member jobs), ≈ 56 min of driver
  time. The cpu32 estimate was 16.7 billed at a 10-min setup. Stop 33.3, never approached.
- Launch-fit 15.2 + 33.3 = 48.5 ≤ 125. Tranche A billed so far ≈ 23.0.
- Default detached flow: pull, stop, terminate. `runpodctl pod list` = [] afterwards. No local
  step ran while the pod billed.

### Files

| file | what |
|---|---|
| `spec.json` | training pins (bar, normaliser, orderer, prereg v2, probe reading, arm, seed, minibatch 24), result |
| `candidate.json` | the generation-20 composite (θ, normaliser) |
| `checkpoints/gen-00NN.json` | per-generation checkpoints, for resume (7.1, if ever licensed) |
| `generations.csv` | per-generation fitness, disqualified count, estimate / step norms, θ hash |
| `member_fitness.csv` | per-member fitness and rank per generation |
| `monitor.csv`, `monitor_summary.csv` | the monitor-set rows (center, A0, R = 2) and per-checkpoint summary |
| `artifacts/members.csv` | 12,000 member episode rows (git-lfs) |
| `contract_failures.csv` | header only |

## Conclusions

- **`unresolved`:** g_tr +29.8 ≤ F_tr 54.75 after 20 generations. Per D7: diagnose, stop. The
  main runs are not licensed.
- **Recorded, not deciding:**
  - the monitor gain was still rising (+17.1 over the last 10 generations);
  - it is concentrated on 2 of 16 world-episodes, with 9 of 16 below A0;
  - the floor is ≈ 2.8% of the bar, against a +1.5% gain.
- **DIAGNOSTICS 27** holds the competing explanations and discriminating checks. The next move
  is the supervisor's and the user's.
