# Orderer oracle-relabel representability probe — ABORTED at 6/10 tasks (2026-07-04)

Run: `tokenomics.eval.orderer_representability` on pod pjw7prk43xbq47 (seed-anchoring
fixed per DIAGNOSTICS entry 10), `--iters 1000 --eval-seeds 3 --horizon 100`, champion
producer (EXP-0035) frozen, fresh orderer distilled per task on the search's own
committed-routing targets.

Aborted deliberately after 6 rows: the "high recovery on most tasks" green-light branch
was already mathematically dead (5/6 rows ≤ 0), and the remaining 4 rows could not
change the verdict. `summary.csv` here is hand-assembled from the run's `[repr]` log
lines (the module only writes its own summary after all tasks; full log preserved as
`train.live.log` copy below). No distilled checkpoints or sample sets were saved by the
probe — a rerun must add artifact saving before deeper analysis.

## Read (per-task, mean over the 3 anchored episode seeds)

- 5/6 tasks: distilled orderer delivers **exactly 0 consumer units** — including the two
  alive tasks where the champion stack delivers 253 (s68) and 953 (s29). Its own teacher
  (per-cycle search, same seeds) delivers 582 and 1071 there.
- 1/6 (p0.22-s45, champion-death task): recovered 560 of 982 teacher units
  (recovered_vs_teacher 0.571) — proves the scorer head + decode CAN express
  life-saving search routing at least sometimes.
- Final masked-mse anti-correlates with outcome (best fit 0.224 on s29 → worst deploy
  −953u; the one success had the worst fit 0.358): in-sample fit is not the failure axis.

## RESOLVED by the instrumented retry (`../orderer_probe_retry/`, 2026-07-05)

The retry (3 tasks, saved samples/ckpts/traces) settled the branch: **target/recipe
failure, not covariate shift and not a representability wall** — see DIAGNOSTICS
entry 11. 42–50% of oracle targets are flat (search committed even_split); the trained
model explains only 25%/10%/64% of target variance vs a route-uniformly zero-predictor
(s29/s77/s45, on TRAIN states); top-1 agreement 0.50–0.60 ≈ chance (0.33–0.50 with 2–3
eligible producers); identical recipe swung 0→800 deployed units across runs (torch init
only). The search's informative decisions are global + tie-arbitrary; per-stream local
MSE regression fits noise. Original hypothesis section below kept for the record.

## Diagnosis status (DIAGNOSTICS entry 2 branch)

Solo distill with generous budget failed to match its teacher on-task → entry 2's
"representability or recipe" branch: obs contract can't express the behavior, OR
collect-vs-deploy covariate shift (model trained only on search-trajectory states; at
deploy its own errors move it off-distribution and, in these capacity-tight worlds,
mis-routing kills the economy → 0 units). The exact-zero pattern on alive worlds + the
mse anti-correlation point at covariate shift, but this is NOT yet diagnosed with
evidence. Discriminating check (not run): save samples + distilled ckpt, measure
in-sample routing agreement vs teacher at teacher states, and trace the deployed
episode (death cycle, capacity trace) on s29/s77/s45.
