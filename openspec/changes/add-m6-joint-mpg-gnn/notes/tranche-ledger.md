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
| EXP-0114 run 1 (iterations 11–35; commit 1918721), in progress: ~5.7 min per iteration, so 11–40 does not fit 150 min; stopped cleanly after iteration 35's checkpoint (supervisor 2026-11-07: split, each run ≤ 40) | 2026-11-07 | 89blmrhmt5hfmx, 16 vCPU / 32 GB + 20 GB volume, EUR-IS-1, 06:48– | stop 40 | |

# Tranche M6-A ledger (learned acceptance, tasks §4; 70 vCPU-h, billed; user 2026-11-07, the supervisor decides within it)

| run | date | pod | billed vCPU-h | cumulative |
|---|---|---|---|---|
| EXP-0115 attempt 1 (commit 5d6187b): all 4 arms ran, but the pull failed (local disk full) and the pod had no volume, so stopping it lost the rows | 2026-11-07 | u97t6mh3oxrmbv, 8 vCPU, EUR-IS-1, 07:16–07:28 + 08:17–08:19 restart | ≈ 1.9 (stop 6) | ≈ 1.9 |
| EXP-0115 attempt 2 (same commit, 20 GB volume, pulled into the main repo) | 2026-11-07 | hk7lwtdbnyr7qg, 8 vCPU, EUR-IS-1, 08:19–08:29 | ≈ 1.5 (stop ≤ 4.1) | ≈ 3.4 |
