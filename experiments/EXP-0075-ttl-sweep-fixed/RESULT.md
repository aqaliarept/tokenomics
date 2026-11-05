# EXP-0075: TTL sweep with the reservation lifetime fixed, supply observer on

**Status: complete, valid.** Task 7.17 of `add-m4-orderer-redesign` (re-run of EXP-0073 after task
7.16; user decision "fix and re-run"). Not an acceptance read.

## In plain English

**The question.** EXP-0073 asked whether letting orders live longer (TTL 3 or 4 cycles instead of
2) delivers more consumer units. But a producer waiting for inputs still had to give up after 2
cycles (`RESERVATION_TTL`), so the main loss, inputs arriving too late, was never tested. Task
7.16 made the reservation lifetime follow the TTL. This run repeats EXP-0073 exactly:
- the same stacks (`entity_planner` with MWWk or round-robin);
- the same 71 world-episodes and random keys;
- the same pre-registered reading.
The 7.14 observer is on, so every lost unit's cause is recorded per TTL.

**Validity.** The TTL-2 rows equal EXP-0057 and EXP-0073 exactly: 213 of 213, on all compared
columns. So the fix changes nothing at the default. There are 0 contract failures in all three
runs, and in every row the observer's classes add up to the traced loss.

**What happened (12 producers decide).**

| stack | units / episode, TTL 2 → 3 → 4 | TTL 3 vs 2 | TTL 4 vs 2 | mean floor |
|---|---|---|---|---|
| round-robin | 2,163 → 2,499 → 2,690 | +323: **`timing`** | +512: **`timing`** | 146.5 |
| MWWk | 2,096 → 2,391 → 2,545 | +285: **`mixed`** | +437: **`mixed`** | 146.5 |

- **Units now rise a lot with TTL:** +14–15% at TTL 3 and +21–24% at TTL 4, about three times
  EXP-0073's gain. No world loses beyond its floor. No episode dies.
- **Round-robin reads `timing`:** the gain is beyond the floor, with no more run-down failures
  (31 vs 34 of 43 episodes).
- **MWWk reads `mixed`:** the gain is beyond the floor on 14 (TTL 3) / 19 (TTL 4) of 23 worlds,
  but more episodes fail the gate's run-down check: 71 / 74 vs 66 of 86. The mean
  end-to-start capacity ratio falls 0.664 → 0.653 → 0.642. The registered rule counts any
  increase as worse.
- **At 42 producers (reported):** round-robin `timing` / `timing` (+806 / +1,198 vs a floor of
  362). MWWk `timing` at TTL 3 (+707), `mixed` at TTL 4 (+1,091; run-down failures 51 vs 50).

**Where the gain comes from** (lost consumer units, 12 producers, MWWk; RR in brackets):

| cause | TTL 2 | TTL 3 | TTL 4 |
|---|---|---|---|
| input arrived after the waiting reservation expired | 173,893 (88,283) | 165,973 (86,675) | 155,959 (82,358) |
| input order died upstream (supplier's reservation expired) | 155,974 (86,254) | 110,931 (59,163) | 84,705 (42,478) |
| input order outbid / no-fit upstream | 51,424 (29,331) | 52,577 (26,573) | 51,088 (25,898) |
| contention (stock used by another assembly) | 54,598 (25,417) | 62,615 (28,912) | 63,685 (30,467) |
| capacity-short | 27,894 (9,935) | 34,380 (13,969) | 39,789 (17,534) |
| outbid | 46,922 (13,569) | 53,328 (20,088) | 63,182 (24,211) |
| other | 23,513 (11,365) | 23,187 (11,855) | 26,462 (13,307) |
| **total lost** | 534,218 (264,154) | 502,991 (247,235) | 484,870 (236,253) |

- **More time mostly rescues the chained failures.** Input orders that died upstream fall by
  46% (MWWk) / 51% (RR) at TTL 4: a supplier now waits long enough for its own inputs.
- **Directly late inputs barely fall** (−10% / −7%). These inputs come later still, so a longer
  wait cannot catch them.
- **Of the late + died-upstream loss, more time recovers 16% at TTL 3 and 27% at TTL 4** (RR 16% /
  29%; at 42 producers 14% / 23% and 17% / 25%).
- **Part of the gain is spent elsewhere.** As more orders stay alive, contention, capacity-short
  and outbid losses rise. MWWk holds more capacity in waiting reservations, and its producers
  run down slightly more.

**What it means.**
- **The losses do have a large timing part.** A longer wait turns about a quarter of the
  input-timing loss into deliveries, mostly where one supplier waits on another.
- **The rest of the late-input loss is lead time no feasible wait covers.** The producer
  decision named by EXP-0074, input buying lead time, is the remaining target: inputs must be
  bought earlier.
- **A TTL change is a world-design decision** (it moves every stack's units, gate and atlas
  worlds), so it goes to the user. It is not a policy result.

## Details
- **Driver:** `supply_trace --order-ttl T` (tasks 7.14 / 7.17a; reservation lifetime = TTL, task
  7.16). One `python -u -c` loop over T = 2, 3, 4, at commit `0a6c928e53b4…`, from a clean
  worktree.
- **Inputs:** atlas v2 `7517314f…` (content `220886a7cf12…`), EXP-0012 `d25388bf…` (pinned only),
  EXP-0057 `episodes.csv` `69656f9e…`, EXP-0073 `ttl2/artifacts/episodes.csv` `19888e26…`.
- **Rows:** 213 per TTL. Neutrality at TTL 2: 213 / 213 vs EXP-0057 and EXP-0073 ttl2
  (`ttl2/neutrality*.csv`; the offline check also 213 / 213 on 99 shared columns). Contract
  failures header-only in all three runs.
- **Reading:** EXP-0073's registered rules, unchanged (`../EXP-0073-ttl-sweep/read_ttl_sweep.py`,
  through `read_exp0075.py`, which takes the atlas floors from EXP-0073's ttl2 spec) →
  `reading_exp0075.json`.

  | cell | TTL 3: diff / beyond / below / reading | TTL 4: diff / beyond / below / reading |
  |---|---|---|
  | 12 MWWk | +285.0 / 14 / 0 / mixed | +436.6 / 19 / 0 / mixed |
  | 12 RR | +323.1 / 16 / 0 / timing | +511.8 / 17 / 0 / timing |
  | 42 MWWk | +706.9 / 7 / 0 / timing | +1,090.9 / 9 / 0 / mixed |
  | 42 RR | +805.6 / 8 / 0 / timing | +1,198.4 / 10 / 0 / timing |

  - Run-down check failures, TTL 2 / 3 / 4: 12 MWWk 66 / 71 / 74 of 86, 12 RR 34 / 31 / 31 of 43,
    42 MWWk 50 / 49 / 51 of 56, 42 RR 26 / 25 / 26 of 28.
  - Deaths 0 everywhere. Pooled fill at 12 producers: MWWk 0.248 / 0.283 / 0.301, RR 0.256 /
    0.296 / 0.319.
- **Loss classes:** `reading_exp0075.json` `loss_classes` (all four cells, every TTL).
  - "input arrived late" = `b2_late`; "died upstream" = `b2_lost_ttl` + `b2_lost_ttl_outage`;
    "outbid / no-fit upstream" = `b2_lost_displaced` + `b2_lost_no_fit`;
  - "capacity" = `a_*`; "outbid" = `c_displaced`.
  - Completion latency is not measured (no per-episode stat).
- **Cost:**
  - **Two unreachable pods** (provider gave no SSH endpoint within 600 s; no job ran; the
    launcher terminated both): `v6n1hw2x2z5yjz` (≈ 10:28:55–10:39:10 UTC) and `2bu0n15kviqvne`
    (10:39:34–≈ 10:49:45). 16 vCPU each, unpinned data center, **≈ 5.4 vCPU-h wasted**.
  - **The run:** relaunch under the supervisor's new stop of 8 (launch-fit 96.2 + 8 = 104.2 ≤
    125), data center pinned. The 16 vCPU attempts were refused in EU-RO-1, EU-CZ-1, US-KS-2,
    CA-MTL-1, EU-SE-1 and US-IL-1 (no capacity, no pod), then created in **EUR-IS-1**:
    `61txojz0zwm3uq`, ≈ 10:51:29–10:56:50, jobs 10:52:47–10:56:29, **≈ 1.4 vCPU-h**.
  - My create-loop script then misread the launcher log and tried US-GA-2, which was refused
    (500, no pod). No second pod existed (pod list empty).
  - **EXP-0075 total ≈ 6.9 vCPU-h. Tranche A ≈ 97.7 of 125.**
- **Files:** `ttl{2,3,4}/` hold `spec.json`, `reading.json` (7.15 shares per arm),
  `supply_trace.csv`, `neutrality.csv`, `neutrality_exp0073.csv`, `contract_failures.csv`, and
  `artifacts/episodes.csv` / `artifacts/producer_cycles.csv` (git-lfs; episodes SHA-256
  `ad567a4f…` / `d83f0a8a…` / `3074b883…`). Also `read_exp0075.py` → `reading_exp0075.json`, and
  `ttl_launch_train.live.log`.

## Conclusions
- **12 producers:** round-robin `timing` at TTL 3 and 4; MWWk `mixed` at both (units beyond the
  floor, run-down failures 66 → 71 / 74). **42 (reported):** RR `timing` / `timing`, MWWk `timing` /
  `mixed`.
- **With the reservation lifetime fixed, a longer TTL adds 14–24% consumer units.** It recovers
  16% (TTL 3) / 27% (TTL 4) of the input-timing loss, almost all from chained upstream expiries.
  Directly late inputs fall only 7–10%.
- **EXP-0073's "mostly scarcity" conclusion is superseded.** The loss has a substantial timing
  component. The residual late-input loss points to input lead time (buy earlier) as the
  producer decision to target. Any TTL change is the user's world-design call.
