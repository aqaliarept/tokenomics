# EXP-0128: λ × heads diagnosis of 4.4's `absent` (hard-stopped; results lost)

**Setup.**
- `m6_prereg_v17` (SHA `419518c0…`, frozen before data). Code 3b2052e (`train/m6_diag_lambda.py` on `m6_stage0c`).
- Cells A (full λ, all heads), B (full λ, gates only), C (λ/4, gates only), run in sequence. D = EXP-0127, re-used.
- Pod 3gkwpfexljrk5s (secure 16 vCPU, EUR-IS-1), 22:25–23:13 UTC, plus a restart 23:14–23:16 to try to recover the
  results: ≈ **13.2 vCPU-h** (stop 47 min ≈ 12.5, M6-A).

## Outcome: **no reading**
- **The hard stop fired during cell C.** The sizing (31 min, estimated) undercounted the per-cell serial work, which
  was not measured.
  - The analysis loads every score npz (≈ 1.6 GB per all-heads cell) twice in one process.
  - That breaks the supervisor's rule (stops from real-data timings + 50%). My error.
- **The results were lost.**
  - The launcher pulls only at the end. When the watchdog stopped the pod, the run directory was gone on restart.
  - These created CPU pods have **no persistent volume**: the pod spec reads `volumeInGb: 0` despite
    `--volume-size 20`, as with EXP-0114.
  - My launch notes since EXP-0115 said "persistent volume". That was wrong.
- **What the log preserves:**
  - **cell A:** primary reading `absent`; its replenish-greedy counterfactual (which counts for the cell) was not
    printed. Score shares and ε are identical to EXP-0127's: neither depends on λ.
  - **cell B:** `absent`; there is no counterfactual for gates only.
  - **cell C:** not finished.
- Under v17's table, B absent rules out `lambda_cut`, `parameter_spread`, `both` and `either_suffices`. The reading is
  `neither` if A's counterfactual and C are absent, else `other`. Neither is known.

## Fixes before any re-run
- Each cell runs as its own pod run, pulling its own results.
- The watchdog pulls the run directory before it stops a pod.
- Stops are sized from a timed cell on real data, including the analysis.
- The run notes no longer claim a persistent volume.
