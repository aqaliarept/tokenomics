# EXP-0083: M5 L2 probe re-scored under pre-registration v4 (task 4.2c)

**Question.** Does EXP-0081's probe give a direction when a death scores units / B_we − 1 instead of the worst
rank? That is v4's "penalised death" with P = B_we, the user's decision after EXP-0082. The flag-count rule
is unchanged.

**Setup.**
- An offline re-score, no pod (supervisor decision). EXP-0081's `probe_episodes.csv` was pre-seeded (SHA in
  `probe/spec.json` → `rescore`), and no job ran. The rows are byte-identical.
- Real-run pins: commit 9ce304d (clean worktree), prereg v4 `4a99c824…`.
- v4 was frozen before this re-score.

## Registered reading: **`no_signal`**, step `no_signal`. Stop branch.

| arm | disqualified A / B | fitness mean A / B | cosine |
|---|---|---|---|
| S (σ 0.05) | 0 / 0 | 1.0345 / 0.9832 | **−0.239** |
| S2 (σ 0.10) | 1 / 0 | 1.0401 / 0.9753 | **−0.161** |

## Diagnosis: the penalty is large against the members' spread (`checks/penalty_scale.py`, offline)

- **The scale.** On minibatch B, member fitness (units only) has an SD of 0.020 (S) and 0.033 (S2). One
  penalised death moves a member's minibatch mean by 1/24 = 0.042, which is 2 SD at σ. With 12 of 24 S members
  dying on 12-0016, B's ranking is still sorted mostly by "died on 12-0016 or not".
- **The cosine against the penalty P = p · B_we** (reported only; post hoc on this probe's rows):

  | p | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 1 (v4) |
  |---|---|---|---|---|---|---|
  | S | 0.840 | 0.809 | 0.803 | 0.788 | 0.517 | −0.239 |
  | S2 | 0.836 | 0.820 | 0.810 | 0.713 | 0.472 | −0.161 |

- **This table cannot pick P.** Choosing P from it would tune the guard on the read it decides. A new P needs
  a reason independent of these rows. One candidate is p = 0: a death costs exactly the units it loses (about
  −5% here), already counted in the fitness. L5 / L6 still refuse extra deaths.

## Conclusions
- **The pilot is not licensed under v4.** The ladder stops.
- **The cause is diagnosed.** P = B_we is about 2 SD of the members' fitness spread, so the deaths on
  knife-edge world 12-0016 still set minibatch B's direction against A's.
- **The death penalty's size is the user's decision.** It is a new pre-registration version, with a P
  justified without these rows.
- **Spend:** 0 vCPU-h. M5-P ≈ 2.7 / 80.
