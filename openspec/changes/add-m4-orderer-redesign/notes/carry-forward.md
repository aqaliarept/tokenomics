# Carry-forward items (supervisor)

Items raised in reviews that a later task must handle. Each names the task that owns it.

- **Task 5.4** (from the 1.2 review): the D7 power estimate is fragile. One world (12-0050) carries 82% of the pooled variance; the detectable effect is 1.88% at 4 episodes × R = 2 but 2.31% at R = 1; the 95% interval on the spread is 1.24–3.83% at 4 episodes. 5.4 must pin R and the episode count, and state up front which paired difference sets the effect, since TR − RR gives 2.50% and MWW − TR 2.14% on the same rows.
- **Task 2.2 / EXP-0057**: the 1.2 prediction is `decoder` (17/23 gap worlds, 16/17 with rec_w ≥ 2/3, median 0.908). It flips only if TS reaches less than about 0.78 of MWW's gain over TR.
- **Minor hardening (optional, any later probe task)**: `pre_pod_offline.py` prints MISMATCH on a world-count mismatch instead of exiting, and does not hash the side-effect params file or `worlds.csv`.
