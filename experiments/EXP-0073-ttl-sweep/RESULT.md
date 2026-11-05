# EXP-0073: Order TTL sweep (TTL 2 / 3 / 4) on the hand-written stacks

> **Correction (2026-11-05, supervisor; found in task 7.14).** This run changed only the order lifetime. A producer that accepted an order but was waiting for inputs held it as a reservation that still failed after a fixed 2 cycles (`RESERVATION_TTL = 2` in `domain/producing_agent.py`, not tied to `order_ttl`). So TTL 3 / 4 never gave producers more time to wait for inputs, and the conclusion that input-wait losses are "scarcity" is **untested**. The gain from orders that lost an auction getting more tries stands. The sweep is re-run with the reservation lifetime tied to the world TTL (tasks 7.16 / 7.17).


**Status: complete, valid.** Task 7.13 of `add-m4-orderer-redesign`. Not an acceptance read.

## In plain English

**The question.** An order lives for a fixed number of cycles (its TTL, 2 so far) and is
rejected if it has not been delivered by then. Most of the consumer units that MWW loses are
orders that expire while the producer waits for inputs. Are those losses a **timing** problem,
meaning orders would be delivered if they could wait longer? Or a **scarcity** problem, meaning
the economy cannot make that much and a longer wait only turns rejections into late deliveries
or other rejections?

**What was run.**
- The two strongest hand-written stacks: the `entity_planner` producer with MWWk (2 replicates)
  and with round-robin (RR).
- EXP-0057's full sample: 23 worlds at 12 producers and 14 at 42, 71 world-episodes, horizon
  100. Every world was played at TTL 2, 3 and 4.
- Demand and shocks are identical across TTLs (equal digests). Only the order lifetime changes.

**Validity.** The TTL-2 run reproduces EXP-0057 exactly: 213 of 213 rows equal on all 105
shared columns. So the TTL setting (task 7.12) changes nothing at its default.

**What happened (12 producers, which decide).**

| stack | units / episode, TTL 2 → 3 → 4 | TTL 3 vs 2 | TTL 4 vs 2 | mean floor |
|---|---|---|---|---|
| RR | 2,163 → 2,298 → 2,363 | +131: **`scarcity`** | +195: **`timing`** | 146.5 |
| MWWk | 2,096 → 2,208 → 2,277 | +108: **`scarcity`** | +175: **`mixed`** | 146.5 |

- **Units rise steadily with TTL, by about 5–6% at TTL 3 and 8–9% at TTL 4.** No world falls
  below its noise floor at any TTL. The rise passes the mean floor only at TTL 4.
- **At TTL 4, RR reads `timing`:** the rise is beyond the floor, with no deaths and no more
  run-down failures (33 vs 34 of 43 episodes).
- **At TTL 4, MWWk reads `mixed`:** the rise is beyond the floor (14 of 23 worlds individually),
  but 70 of 86 episodes fail the gate's run-down check against 66 at TTL 2. The rule fixed
  before the pull counts any increase as worse; it has no noise margin. The mean run-down ratio
  moves only 0.664 → 0.656. That is small next to the MWWk run-down noise margin EXP-0071
  measured (0.110), but the reading stands as registered.
- **At 42 producers (reported only):**
  - RR reads `timing` at both TTLs (+381 and +544 against a mean floor of 362).
  - MWWk reads `scarcity` at TTL 3 (+275) and `mixed` at TTL 4 (+478; run-down failures 52 vs 50).
    TTL 3 also has one death, the only one in the run.

**Where the extra units come from.** This answers the question behind the sweep.
- **Expired units barely fall.** At 12 producers, MWWk loses 456.5k units to TTL at TTL 2 and
  456.8k at TTL 4; RR goes from 240.0k to 233.9k.
- **Expiries while waiting for inputs fall only 2–5%** (MWWk 340.2k → 332.4k).
- **The gain comes from orders that lost an auction**, by not fitting (`no_fit` 26.3k → 18.2k)
  or by being outbid (`displaced` 46.9k → 36.7k). A longer life gives them more auctions to win.
- **Three quarters of all ordered units are still undelivered at TTL 4** (pooled fill MWWk 0.25
  → 0.27, RR 0.26 → 0.28).

**What it means.**
- **The losses are mainly scarcity.** Orders waiting for inputs keep expiring however long they
  may wait. Demand is about four times what the economy delivers.
- **There is a smaller timing component.** It is real and monotone, about +8% at TTL 4, and it
  works through extra auction retries, not through rescued input waits.
- **Raising the TTL is a world-design change, not a policy improvement.** It moves every
  stack's units, so the gate and atlas worlds would change. That is a human gate.
- **What stays open.** The task's premise ("TTL 2 penalises deep chains") is supported only
  weakly. Where the input shortage itself comes from is not answered here.

## Details
- **Driver:** `decoder_check --order-ttl T` (task 7.12's TTL-sweep profile: EXP-0057's sample
  and draw keys, arms MWWk and RR, no references, D3 readings withheld).
  - Run as one `python -u -c` loop over T = 2, 3, 4 on one pod, at commit
    `2757aa7b513e…` from a clean worktree.
  - Inputs: atlas v2 `7517314f…` (content `220886a7cf12…`); EXP-0012 `orderer_scorer.pt`
    `d25388bf…` (pinned only; no arm plays it).
- **Rows:** 213 per TTL (71 world-episodes × MWWk r0 / r1 + RR). Contract failures: header-only
  in all three runs; no missing rows. Demand and shock digests are equal across TTLs, row by row.
- **Neutrality:** the TTL-2 rows equal EXP-0057 `episodes.csv` (`69656f9e…`) on 105 shared
  columns, 213 / 213 (`reading_ttl_sweep.json` `neutrality_ttl2_vs_exp0057`).
- **Reading rules** (fixed in `read_ttl_sweep.py` before the pull):
  - diff_w = the mean over world w's paired (episode, replicate) rows of units(T) − units(2);
  - floor_w = the atlas v2 `noise_floor` (pinned in `spec.json` `pins.sample`);
  - "rise" = mean_w(diff_w) > mean_w(floor_w); "within" = |mean_w(diff_w)| ≤ mean_w(floor_w);
  - "no worse" = deaths(T) ≤ deaths(2), and the number of episodes failing r_end ≥ 0.7516 · r_start
    (side-effect params v2) at T ≤ at TTL 2;
  - timing = rise and no worse; scarcity = within; mixed = otherwise.
- **Per TTL (12 producers; deaths 0 everywhere):**

  | stack | TTL | units/ep | expired units | of which inputs short | no_fit | displaced | run-down fails | mean r_end/r_start | pooled fill |
  |---|---|---|---|---|---|---|---|---|---|
  | MWWk | 2 | 2,096.0 | 457,725 | 340,166 | 26,295 | 46,922 | 66 / 86 | 0.664 | 0.248 |
  | MWWk | 3 | 2,208.4 | 457,509 | 337,625 | 22,078 | 39,825 | 74 / 86 | 0.655 | 0.262 |
  | MWWk | 4 | 2,276.7 | 458,100 | 332,374 | 18,181 | 36,715 | 70 / 86 | 0.656 | 0.270 |
  | RR | 2 | 2,163.3 | 240,496 | 205,796 | 8,826 | 13,569 | 34 / 43 | 0.685 | 0.256 |
  | RR | 3 | 2,298.3 | 236,670 | 198,828 | 7,022 | 12,652 | 31 / 43 | 0.684 | 0.272 |
  | RR | 4 | 2,363.1 | 234,550 | 195,922 | 6,044 | 12,446 | 33 / 43 | 0.681 | 0.280 |

  Expired = `trace_lost_units_consumer_ttl` + `_ttl_outage`. Worlds beyond / below the floor:
  - MWWk: 5 / 0 at TTL 3, 14 / 0 at TTL 4;
  - RR: 10 / 0 at TTL 3, 10 / 0 at TTL 4.
- **42 producers:**
  - MWWk: units/ep 5,924.8 → 6,200.0 → 6,402.9; deaths 0 / 1 / 0; run-down fails 50 / 51 / 52 of
    56; expired 636.1k → 625.5k → 629.1k.
  - RR: 5,606.2 → 5,986.8 → 6,150.3; deaths 0; run-down fails 26 / 26 / 26 of 28; expired
    332.1k → 327.1k → 324.8k.
  - Mean floor 362.2. Worlds beyond: MWWk 4 / 7, RR 5 / 8; 0 below anywhere.
- **Slot overflow** (`trace_overflow_orders`): 0 at every TTL, so the TTL-sized slots (7.12)
  hold every live order.
- **Completion latency: not measured.** The driver's rows carry no per-episode latency; it exists
  only as an agent-history EMA. A proxy is the units still in progress at the episode's end
  (ordered − fulfilled − lost), which grow with TTL: MWWk at 12 has 11.8k → 13.8k → 14.8k. A
  per-episode latency stat would need a small code task.
- **Cost:**
  - One fresh secure 16 vCPU pod, `5frcadembgunty` (cpu5c). It was created at 08:40:57 UTC and
    terminated by 08:45:59. The 639 jobs ran 08:42:15–08:45:45.
  - **≈ 1.4 vCPU-h billed** (estimate 4, stop 10, enforced by a local watchdog). A 32 vCPU
    secure request failed first (no capacity; no pod created).
  - Launch-fit 88.8 + 10 = 98.8 ≤ 125. **Tranche A ≈ 90.2 of 125.**
- **Files:**
  - `ttl{2,3,4}/`: `spec.json`, `reading.json` (D3 readings withheld), `routing_metrics.csv`,
    `reproduction.csv` (header-only: no references), `contract_failures.csv` (header-only),
    `artifacts/episodes.csv` (git-lfs; SHA-256 `19888e26…` / `87d12383…` / `436f9ffe…`);
  - `read_ttl_sweep.py` and its output `reading_ttl_sweep.json`.
  - Reproduce: `python3 read_ttl_sweep.py . ../EXP-0057-decoder-check/artifacts/episodes.csv`.

## Conclusions
- **Readings at 12 producers:**
  - RR: TTL 3 `scarcity`, TTL 4 `timing`;
  - MWWk: TTL 3 `scarcity`, TTL 4 `mixed` (run-down failures 70 vs 66, no margin in the rule).
- **Readings at 42 producers (reported):** RR `timing` / `timing`; MWWk `scarcity` / `mixed`.
- **Units rise monotonically with TTL** (+8–9% at TTL 4), and no world loses beyond its floor.
- **Expiries hardly fall**, input-wait expiries by 2–5%. The gain is extra auction retries for
  orders that did not fit or were outbid. The dominant loss is scarcity: fill stays ≈ 0.27.
- **TTL is not the lever for the M4 / M5 orderer.** A TTL change is a world-design decision for
  the user (it changes every stack's numbers).
