# EXP-0082: M5 L2 probe re-scored under pre-registration v3 (task 4.2b)

**Question.** Does EXP-0081's probe give a direction once the guard is v3's flag-count rule, which the user chose
after EXP-0081 ("Compare flag rates")? The reading is σ arm S, plus the step reading.

**Setup.**
- **Supervisor decision (2026-11-05): an offline re-score, no pod.** EXP-0081's member rows are the same inputs:
  θ = 0 members, keys, minibatches and noise.
- EXP-0081's `probe_episodes.csv` (SHA `spec.json` → `rescore`) was pre-seeded. `producer_es.run_probe` then ran
  with a job function that refuses to play, so every row was reused. The rows file is unchanged, byte for byte.
- Real-run pins: commit d825401 (clean worktree), prereg v3 `97c3332c…`, T = 5/3, EXP-0080 sources.
- v3 was frozen before this re-score.

## Registered reading: **`no_signal`**, step `no_signal`. Stop branch.

| arm | disqualified A / B | cosine |
|---|---|---|
| S (σ 0.05) | 0 / 12 | **−0.670** |
| S2 (σ 0.10) | 1 / 13 | **−0.748** |

There were 0 contract failures.

## Diagnosis: the absolute death clause (`checks/guard_split.py`, offline)

| guard | S cosine | S2 cosine | disqualified on B (S / S2) |
|---|---|---|---|
| v3 (deaths + flags) | −0.670 | −0.748 | 12 / 13 |
| deaths only | −0.670 | −0.799 | 12 / 13 |
| flags only | **0.840** | **0.836** | 0 / 0 |
| off | 0.840 | 0.902 | 0 / 0 |

- **The flag-count part does its job.** It removes EXP-0081's noise disqualifications (0 on B) and keeps the signal.
- **Every remaining disqualification on B is a death.**
  - 11 of S's 12 die on train 12-0016, and 1 on 12-0004.
  - On 12-0016, rf3 lives at all 5 keys, while 24 of 48 members die.
- **The deaths cost few units.** The dying members' mean on 12-0016 is 1,590.9 against rf3's 1,669 (−4.7%), so
  the deaths are presumably late in the episode (inferred; these rows have no trace). Ranked worst, however,
  12 deaths dominate B's ranking. Their direction is opposite to the units direction both minibatches share
  (with deaths off, A and B agree at 0.84).
- **So small steps along the units gradient tip 12-0016 into system death.** rf3 sits close to death on that
  world. That is a real side effect, not draw noise.

## Conclusions
- **The pilot is not licensed under v3.** The ladder stops.
- **The cause is diagnosed.** The death clause on one knife-edge training world (12-0016) opposes the units
  direction. The choice of response is the user's:
  - keep deaths absolute and accept that the residual cannot move along that direction;
  - exclude or down-weight 12-0016-type worlds;
  - treat a death by its units cost.
  Each is a new pre-registration version.
- **Spend:** 0 vCPU-h (offline). M5-P ≈ 2.7 / 80.
