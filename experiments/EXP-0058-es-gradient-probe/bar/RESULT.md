# bar

Bar precompute for step-2 ES (add-m4-orderer-redesign D6, task 5.1), driver `tokenomics.train.order_es_worlds`,
horizon 100. Status: complete: 64 training worlds, 128 world-episodes, 0 replaced; normaliser 2d574ec8a3df. Rows: 1380.

Outputs: `bar.csv` (B_we per training world-episode), `training_worlds.json`, `schedule.json`,
`normaliser.json` / `normaliser_report.csv`, `candidates.csv`, `timings.csv`, `bar_summary.json`.

## Conclusions

Recorded in the experiment's `../RESULT.md` (EXP-0058: the bar, then the gradient probe). Bulk files moved
to `../artifacts/` (git-lfs): `bar_episodes.csv` (this dir's `episodes.csv`) and `bar_fit_rows/` (this dir's
`fit_rows/`); move them back here before `order_es_worlds --mode fit-normaliser --bar-dir bar/`.
