# Tranche M6 ledger (150 vCPU-h, billed)

| run | date | pod | billed vCPU-h | cumulative |
|---|---|---|---|---|
| EXP-0099 Stage I (collect → fit → check) | 2026-11-06 | eyy1zg07jwa0uz, 16 vCPU, EUR-IS-1, 09:49:36–10:08:24 | 5.0 | 5.0 |
| EXP-0100 Stage I v2 (fit-mode probe → 20-epoch fit → 4-arm check) | 2026-11-06 | ap4hmm4hf1tm8e, 16 vCPU, EUR-IS-1, ≈ 11:15–11:42 | 7.2 (stop 12, extended by the supervisor at 11:22 to 12:25 ≈ 19; not reached) | 12.2 |
| EXP-0101 side effects (h100 3 arms + h200 2 arms) | 2026-11-06 | g6wve1cq1ekowi, 16 vCPU, EUR-IS-1, 11:47–11:49 | 0.6 (stop 8) | 12.8 |
| EXP-0102 validation (L5 worlds, 3 arms) | 2026-11-06 | o78goq826u3u1d, 16 vCPU, EUR-IS-1, 11:54–12:01 | 1.9 (stop 8) | 14.7 |
| EXP-0103 held-out (aborted launch lya3nhr20bnmtl ≈ 0.2 + run 0ryv5idl4w17ex) | 2026-11-06 | 16 vCPU, EUR-IS-1, 12:11–12:18 | 2.2 (stop 8) | 16.9 |
| EXP-0104 drain diagnostic (h200, 9 worlds at 42) | 2026-11-06 | o0vjweodztdyvt, 16 vCPU, EUR-IS-1, 12:23–12:30 | 1.9 (stop 6) | 18.8 |
| EXP-0105 attempt 1 (collect done; fit killed, out of memory: 16 workers each held the samples) | 2026-11-06 | l5cqpvr1zpowxt, 16 vCPU / 32 GB, EUR-IS-1, 13:44–13:49 | 1.3 (stop 24, raised from 16 by the supervisor) | 20.1 |
| EXP-0105 attempt 2 (fit hung: 'Too many open files', file_descriptor sharing; stopped after 5 min without progress) | 2026-11-06 | pmwe2rrpy4mmsj, 16 vCPU / 32 GB, EUR-IS-1, 14:19–14:27 | 2.1 | 22.2 |
| EXP-0105 attempt 3 (fit + check, commit 134c267) | 2026-11-06 | 53yvp4v7yj77ub, 16 vCPU / 32 GB, EUR-IS-1, 14:27–15:15 | 12.8 (stop 24) | 35.0 |
| EXP-0106 upgrade-gate probe (offline, recorded decisions) | 2026-11-06 | ts6iux5ztcn8ek, 8 vCPU / 16 GB, EUR-IS-1, 15:20–15:24 | 0.6 (stop 3) | 35.6 |
| EXP-0107 Stage I v4 (fit → gate → check, commit 4a8bc0a) | 2026-11-06 | o0ga8nfebym9nc, 16 vCPU / 32 GB, EUR-IS-1, 15:34–16:44 | 18.7 (stop 24) | 54.3 |
| EXP-0108 attribution (commit 0166964; hard-stopped at 45 min, phase 2 lost) | 2026-11-06 | ddraj6ise5j20c, 16 vCPU / 32 GB, EUR-IS-1, 16:52–17:37 | 12.0 (stop 12) | 66.3 |
| EXP-0109 restore probe (commit b75a772) | 2026-11-06 | vpy88jxjmd6v86, 16 vCPU / 32 GB + 20 GB volume, EUR-IS-1, 18:16–18:27 | 2.8 (stop 16) | 69.1 |
| EXP-0110 Stage 0 (commit 4b3ecb0) | 2026-11-06 | wacrn1uzipxeww, 16 vCPU / 32 GB + 20 GB volume, EUR-IS-1, 18:40–18:51 | 2.9 (stop 8) | 72.0 |
| EXP-0111 power calibration (attempt 1 refused at start ≈ 0.6; attempt 2 9gce2b3c9penyp 20:33–20:44 ≈ 1.5) | 2026-11-06 | 8 vCPU / 16 GB + 20 GB volume, EUR-IS-1 | 2.1 (stop 3) | 74.1 |
| EXP-0112 Stage 0b (commit 096be8c) | 2026-11-06 | svyc2sej5ixz0c, 16 vCPU / 32 GB + 20 GB volume, EUR-IS-1, 20:59–21:28 | 7.7 (stop 10) | 81.8 |
| EXP-0113 Stage 1 pilot (commit d1d8f54) | 2026-11-06/07 | cctsg5g6uuw1j6, 16 vCPU / 32 GB + 20 GB volume, EUR-IS-1, 23:34–00:16 | 11.2 (stop 22) | 93.0 |
| EXP-0114 run 1 (iterations 11–35; commit 1918721): ~5.7 min per iteration, so 11–40 did not fit 150 min; stopped cleanly after iteration 35's checkpoint (supervisor 2026-11-07: split, each run ≤ 40) | 2026-11-07 | 89blmrhmt5hfmx, 16 vCPU / 32 GB, EUR-IS-1, 06:48–09:11 | ≈ 38.1 (stop 40) | ≈ 131.1 |
| EXP-0114 run 2 (iterations 36–40; reading `hold`, g +77.7 vs F 116.6). The pod had no persistent /workspace; the first launch restarted at iteration 11 and was killed (≈ 25 min lost); relaunched from run 1's iteration-35 state | 2026-11-07 | same pod, 09:11–10:03 | ≈ 13.8 (stop ≤ 19) | ≈ 144.9 |

# Tranche M6-A ledger (learned acceptance, tasks §4; 70 vCPU-h, billed; user 2026-11-07, the supervisor decides within it)

| run | date | pod | billed vCPU-h | cumulative |
|---|---|---|---|---|
| EXP-0115 attempt 1 (commit 5d6187b): all 4 arms ran, but the pull failed (local disk full) and the pod had no volume, so stopping it lost the rows | 2026-11-07 | u97t6mh3oxrmbv, 8 vCPU, EUR-IS-1, 07:16–07:28 + 08:17–08:19 restart | ≈ 1.9 (stop 6) | ≈ 1.9 |
| EXP-0115 attempt 2 (same commit, 20 GB volume, pulled into the main repo) | 2026-11-07 | hk7lwtdbnyr7qg, 8 vCPU, EUR-IS-1, 08:19–08:29 | ≈ 1.5 (stop ≤ 4.1) | ≈ 3.4 |
| EXP-0116 EXP-0114 gap diagnosis (commit 364b070; reading `replenish_drift`) | 2026-11-07 | obj0bkagr1msh0, 16 vCPU + 20 GB volume, EUR-IS-1, 11:16–11:27 | ≈ 3.1 (stop 8) | ≈ 6.5 |
| EXP-0117 turn-head imitation v12 (commit d7567dc): stopped by the supervisor during the fit (v12's offline gate could not pass: unlearnable label pairs); nothing pulled, pod removed | 2026-11-07 | wcr7gxfxmcqxyc, 16 vCPU + 20 GB volume, EUR-IS-1, 11:23–11:36 | ≈ 3.6 (stop 12) | ≈ 10.1 |
| EXP-0118 turn-head imitation v13 (commit 8d140eb; reading `offline_fail`, DIAGNOSTICS 38) | 2026-11-07 | 80tp1npy7zrvam, 16 vCPU + 20 GB volume, EUR-IS-1, 11:40–11:49 | ≈ 2.3 (stop 13) | ≈ 12.4 |
| EXP-0119 F40 confirmation (commit f678f06; reading `gain`) | 2026-11-07 | m5c0ixjx8x8oz8, 16 vCPU + 20 GB volume, EUR-IS-1, 12:27–12:34 | ≈ 1.7 (stop 6) | ≈ 14.1 |
| EXP-0120 turn-head imitation v14 (commit c13bd0a; F40 start; reading `offline_fail`, underfit) | 2026-11-07 | wnes87bgzymub3, 16 vCPU + 20 GB volume, EUR-IS-1, 14:05–14:22 | ≈ 4.7 (stop 12) | ≈ 18.8 |
| EXP-0121 turn-head capacity probe (commit 83de1fd; no winner) | 2026-11-07 | s26tp6kup137bk, 8 vCPU + 20 GB volume, EUR-IS-1, 14:38–14:55 | ≈ 2.3 (stop 5) | ≈ 21.1 |
| EXP-0122 turn-head probe with linear fractions (commit 8189555; no winner) | 2026-11-07 | wn3a4ttnj5oey1, 8 vCPU + 20 GB volume, EUR-IS-1, 15:19–15:38 | ≈ 2.6 (stop 3) | ≈ 23.7 |
| EXP-0123 closed-loop diagnostic of EXP-0120's heads (commit 96f393d; reading `cost`; + test_m6_pilot.py) | 2026-11-07 | 65bih6r0l19dzj, 8 vCPU + 20 GB volume, EUR-IS-1, 15:56–16:03 | ≈ 1.0 (stop 3) | ≈ 24.7 |
| EXP-0124 ordered-pass probe (commit 264c857): hard-stopped before the slowest settings finished; results written only at the end, nothing pulled (sizing error) | 2026-11-07 | 8tcohug99rbfsy, 8 vCPU, EUR-IS-1, 16:35–16:58 | ≈ 3.1 (stop 3) | ≈ 27.8 |
| EXP-0124 attempt 2 (commit df880e1, per-setting saves, stop raised to 5 by the supervisor; no winner) + a mis-pathed launch removed within ≈ 1 min (≈ 0.1) | 2026-11-07 | nv3l2cockbiz53 (+ xpeoeosi2jh7fk), 8 vCPU, EUR-IS-1, 17:00–17:23 | ≈ 3.2 (stop 5) | ≈ 31.0 |
| EXP-0125 arrival-position probe (commit a9cd94e; winner w32 L3 e90 + arrival) | 2026-11-07 | j8n79qomsxkbrt, 4 vCPU + 20 GB volume, EUR-IS-1, 17:35–17:59 | ≈ 1.6 (stop 2.7) | ≈ 32.6 |
| EXP-0126 turn-head imitation v15 + closed-loop diagnostic (commit 8a3af7e; offline pass, closed-loop gate fail on run-down, diagnostic no_cost) | 2026-11-07 | 30fw6eam8ac7ls, 4 vCPU + 20 GB volume, EUR-IS-1, 18:05–18:52 | ≈ 3.1 (stop 11) | ≈ 35.7 |
| EXP-0127 timing pod (commit fb9ed80; 4.4 seed timed, 4.5 timing failed: file descriptors) | 2026-11-07 | lpjx5nv04kvnxp, 8 vCPU, EUR-IS-1, 21:07–21:12 | ≈ 0.6 (stop 5.3) | ≈ 36.3 |
| EXP-0127 4.4 power check + 4.5 iteration timing (commit cb2fe36; reading `absent`) | 2026-11-07 | kkvree7wl1fo6a, 16 vCPU + 20 GB volume, EUR-IS-1, 21:36–21:58 | ≈ 5.9 (stop 8.8) | ≈ 42.2 |
