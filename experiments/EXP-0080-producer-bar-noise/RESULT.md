# EXP-0080: M5 L1, the bar and the noise floor on rf3 (task 4.1)

**Question.** What does the base `entity_planner_buyahead_k4_rf3` (rf3, with MWWk) score on M4's pinned training
set? How noisy is it across draw keys, and which disqualification margin keeps rf3 from flagging itself? This is
design D5 / D6's L1, and it is not an acceptance read.

**Setup.**
- Commit 912e36f, run with `--verify-source-commit`.
- Driver: `tokenomics.train.producer_es`, `--mode bar` then `--mode noise-floor`, in one pod.
- Training set: EXP-0058 `bar/`, 64 worlds at 12 producers, 128 world-episodes. Training-worlds SHA `02abf7f7…`,
  schedule SHA `122ebe64…`.
- Horizon 100.
- Pod: secure 8 vCPU in EUR-IS-1 (`85rao6zslpjdtd`), 18:08–18:13 UTC. 16 vCPU was unavailable in EUR-IS-1, so
  the run used the pinned fallback.

## Results

| check | result |
|---|---|
| bar (rf3, r0), 128 world-episodes | mean 3,196.5 consumer units. 1 death. Run-down check (r_end < 0.7516 r_start) failed in 110 / 128. Starved consumers in 102 / 128. |
| normaliser | fitted on the bar's rf3 rows (`bar/normaliser.json`; fit rows in `artifacts/bar_fit_rows/`) |
| r0 replayed by the noise floor | **128 / 128 rows equal** (`noise_floor/reproduction.csv`) |
| EXP-0079 cross-check (rf3 on 2 EXP-0057 world-episodes) | **2 / 2 equal**, 136 columns compared (`bar/crosscheck.csv`) |
| noise floor, re-keys r1–r4 (512 episodes) | σ run-down 0.0341, σ starvation 0.0417 |
| margin = c · √2 · σ, with c = 2 | run-down 0.0966, starvation 0.1180 |
| self-flags (rf3 against itself) | **24 / 512 = 4.69% ≤ 5%** (96 without the margin) |
| contract failures | **0** (bar and noise floor) |

## Conclusions
- **L1 passes all four verify items in task 4.1.** The bar, `normaliser.json` and `margins.json` are the
  inputs to L2 (EXP-0081).
- **The self-flag rate of 4.69% sits close to the 5% target.** The pre-registered rule (c = 2) holds, so
  nothing is re-tuned.
- **rf3 fails the absolute run-down check in most training episodes** (110 / 128). That is why D5's guard
  is relative to rf3 and not absolute: a candidate is flagged only for being worse than rf3 by more than the
  margin.
- **Spend:** ≈ 0.7 vCPU-h billed (stop 3). M5-P stands at ≈ 1.6 / 80.
