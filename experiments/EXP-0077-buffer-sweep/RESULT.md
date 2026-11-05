# EXP-0077: Buffer sweep of `entity_planner_buyahead` (k = 3, 4), task 7.21

**Status: complete, valid** (attempt 3). Not an acceptance read.

## In plain English

**The question.** EXP-0076 showed that buying inputs ahead helps, and that the gain had not levelled
off at k = 2 (k = the number of cycles of its own recent input use a producer keeps in stock). How
large should the buffer be? This run adds k = 3 and k = 4:
- the same stacks (with MWWk and with round-robin), TTL 2, EXP-0057's 71 world-episodes;
- k = 2 and `entity_planner` (no buffer) reused from EXP-0076;
- the k* rule registered before the data: the smallest k whose units are within the noise floor of
  the best k, with run-down failures at most +5 and no extra deaths vs `entity_planner`.

**Validity.**
- The k = 3 / 4 rows have the same demand and shock schedules as EXP-0076's (213 / 213 each).
- The k = 3 rows are identical, cell for cell, to the crashed attempt 2's.
- 0 contract failures.

**What happened (12 producers decide; consumer units per episode).**

| k | MWWk | RR | held stock / cycle (MWWk) | run-down fails (MWWk, of 86) | investment rejection (MWWk) |
|---|---|---|---|---|---|
| `entity_planner` | 2,096 | 2,163 | 132 | 66 | 0.058 |
| 1 (EXP-0076) | 2,323 | 2,398 | 156 | 70 | 0.064 |
| 2 (EXP-0076) | 2,669 | 2,761 | 180 | 70 | 0.067 |
| 3 | 2,935 | 3,065 | 208 | 64 | 0.074 |
| 4 | 3,101 | 3,200 | 233 | 71 | 0.078 |

- **Units keep rising with k:** +46% over `entity_planner` at k = 4 with either orderer (+969 / +997).
  Each step adds less: k 2 → 3 adds +266 / +304, and k 3 → 4 adds +166 / +135.
- **No deaths at any k.** Run-down failures stay within the limit. MWWk at k = 4 sits exactly at the
  +5 edge (71 vs 66); RR is at 33–35 of 43 throughout.
- **The k* rule gives k* = 4.** With MWWk, k = 3 is 163 units below k = 4, beyond the floor (146.5).
  With round-robin, k = 3 is within the floor (133). The registered k* is the larger of the two, so
  that it holds for both.
- **At 42 producers (reported):** k* = 3 for both orderers. k = 4 is +2,051 / +2,074 (+35% / +37%)
  over `entity_planner`.
- **Where the gain comes from (12, MWWk):** input orders dying upstream keep falling (156k → 105k →
  86k → 77k for EP / k = 2 / 3 / 4). Late inputs at the final assembler stay flat (174k → 178k).
  Contention rises (55k → 68k).
- **The cost:** held stock up 77% at k = 4, and investment rejections 0.058 → 0.078.

**What it means.**
- **k* = 4 is the edge of the tested range.** The best k is the largest one tried, and the gains are
  shrinking but still beyond the floor with MWWk. So a larger buffer might still be better, and k =
  4 is not shown to be the optimum.
- **The step to k = 4 uses up the run-down allowance** for MWWk (+5, the limit), so further k would
  need to watch that guard.
- **Step 4 (task 7.22) builds on k* = 4** as registered. Its target is the remaining "input arrived
  late" loss at the final assembler, which buffering does not move.

## Details
- **Driver:** `supply_trace --order-ttl 2 --producer P` for P = `entity_planner_buyahead_k3` /
  `_k4`, one `-c` loop, at commit `08bb01a` (with the CSV-limit fix), from a clean worktree. No
  references (buy-ahead rows), so validity is checked against EXP-0076 offline.
- **Inputs:** atlas v2 `7517314f…`, EXP-0012 `d25388bf…` (pinned only).
- **Rows:** 213 per k. Digests equal EXP-0076's `entity_planner` rows; the k = 3 rows equal
  `attempt2/k3_episodes.csv` on every cell; contract failures are header-only.
- **k\* rule** (`read_exp0077.py`, fixed before the data):
  - candidates k ∈ {2, 3, 4};
  - eligible = no extra deaths, and run-down failures (r_end < 0.7516 r_start) up ≤ 5 (absolute,
    per orderer);
  - best = the most units; within = mean_w(best − k) ≤ mean_w(floor_w), paired per world, atlas
    floors;
  - k*_o = the smallest eligible k within; k* = max over orderers.

  | cell | eligible | best | best − k (k = 2 / 3) | mean floor | k*_o |
  |---|---|---|---|---|---|
  | 12 MWWk | 2, 3, 4 | 4 | 422.3 / 163.3 | 146.5 | 4 |
  | 12 RR | 2, 3, 4 | 4 | 423.7 / 133.2 | 146.5 | 3 |
  | 42 MWWk | 2, 3, 4 | 4 | 867.0 / 318.9 | 362.2 | 3 |
  | 42 RR | 2, 3, 4 | 4 | 898.0 / 333.0 | 362.2 | 3 |

  **k\* = 4 at 12 producers** (deciding); 3 at 42 (reported).
- **42 producers, units EP / k1 / k2 / k3 / k4:** MWWk 5,925 / 6,387 / 7,108 / 7,657 / 7,976; RR
  5,606 / 6,024 / 6,782 / 7,347 / 7,680. Held stock (MWWk) 814 / 899 / 982 / 1,068 / 1,167. Run-down
  fails 50 / 51 / 50 / 52 / 51. Deaths 0.
- **12 RR detail:** held stock 146 / 172 / 197 / 227 / 250; investment rejection 0.065 / 0.069 / 0.073
  / 0.075 / 0.080; died upstream 86k / 73k / 54k / 42k / 39k; late 88k / 88k / 94k / 94k / 90k.
- **Files:**
  - `entity_planner_buyahead_k3/`, `entity_planner_buyahead_k4/` hold `spec.json`, `reading.json`,
    `supply_trace.csv`, `contract_failures.csv`, and `artifacts/episodes.csv` /
    `producer_cycles.csv` (git-lfs; episodes SHA-256 `80658831…` / `11a1513e…`);
  - `read_exp0077.py` → `reading_exp0077.json`; `attempt2/` (crashed run: k = 3 rows, spec, log);
    `launch_attempt3.log`.

## Attempts
### Attempt 1 (unpinned 32 vCPU)
- **The run:** from a clean worktree at `971eb6e`. The read script `read_exp0077.py` (k* rule
  fixed) was written before the launch.
- **The 32 vCPU attempts:**
  - A 32 vCPU secure request pinned to EUR-IS-1 was refused (no capacity; no pod).
  - The unpinned 32 vCPU request created `lifokam7dcg0xl` at ≈ 12:51:30 UTC. Runpod never
    published its SSH endpoint ("ssh endpoint not reported yet") within the launcher's 600 s
    ready-timeout. The launcher stopped and terminated it by 13:01:45. No code was synced and no
    job ran.
- **Billed ≈ 5.5 vCPU-h wasted** (≈ 10.3 min × 32 vCPU) against the run's stop of 6. The
  remaining ≈ 0.5 does not cover a run (estimate ≈ 1.0), so no second launch was made.
  **Tranche A ≈ 104.5 of 125.**
- **Pattern today:**
  - Unpinned secure CPU pods got no SSH endpoint in 3 of 3 launches (EXP-0075 × 2, EXP-0077).
  - Pods pinned to EUR-IS-1 got SSH in 2 of 2 (EXP-0075 relaunch, EXP-0076).
  - EXP-0074's unpinned pod at 09:55 was fine.

### Attempt 2 (relaunch pinned to EUR-IS-1, new stop 4)
- **The pod:** `a1ahhc26d73iah`, secure 16 vCPU, EUR-IS-1, ≈ 13:02:39–13:05:10 UTC. SSH came up
  normally.
- **What ran:** the k = 3 run's 213 jobs completed. Then `supply_trace` crashed re-reading its own
  `episodes.csv` (`gate_cli.read_rows`): `_csv.Error: field larger than field limit (131072)`. The
  `-c` loop exited 1 before k = 4 started. No `reading.json`, consistency or neutrality output.
- **Root cause:** the `st_producer_cycles` JSON cell of a 42-producer row is ≈ 300k characters (84
  of 213 rows exceed 131,072, in EXP-0076 too). EXP-0074–0076 passed only because each process
  first loaded a reference through `decoder_check.read_reference`, which raises
  `csv.field_size_limit` process-wide as a side effect. EXP-0077's buy-ahead runs take no
  reference.
- **Fix (committed as `08bb01a`):** `supply_trace.allow_large_cells()`, called first in
  `run_supply_trace`. The test reproduces the default-limit failure in a fresh process; a mutation
  check (the call removed) makes it fail.
- **Kept:** the crashed run's k = 3 rows, `attempt2/k3_episodes.csv` (213 rows, unprocessed), for a
  cross-check of the relaunch, plus `attempt2/k3_spec.json` and `attempt2/launch.log`.
- **Billed ≈ 0.67 vCPU-h** (2.5 min × 16 vCPU).

### Attempt 3 (pinned EUR-IS-1, 16 vCPU, after the fix `08bb01a`)
- Pod `su2sv2w7aewy2m`, ≈ 13:07:43–13:11:43 UTC. SSH came up normally; both runs completed
  (k = 3 13:09:16, k = 4 13:10:19).
- **≈ 1.07 vCPU-h.**

## Cost
- Attempt 1 (no SSH): **≈ 5.5 vCPU-h lost**. Attempt 2 (crash after k = 3): **≈ 0.67**.
  Attempt 3: **≈ 1.07**.
- **EXP-0077 total ≈ 7.2 vCPU-h. Tranche A ≈ 106.2 of 125.**
- Stops: 6 for attempt 1; a new stop of 4 for attempts 2 + 3 (1.74 used).
