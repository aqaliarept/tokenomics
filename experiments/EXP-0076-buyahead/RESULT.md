# EXP-0076: Buying inputs ahead (`entity_planner_buyahead`) at TTL 2

**Status: complete, valid.** Task 7.19 of `add-m4-orderer-redesign` (user decision "Test buy-ahead at
TTL 2"; producer evaluation step 2c). Not an acceptance read.

## In plain English

**The question.** Most lost consumer units are inputs that arrive too late (EXP-0074), and a
longer order lifetime recovers only part of that (EXP-0075). Can producers fix it themselves, at
today's TTL of 2, by buying inputs ahead of demand?
- The new producer `entity_planner_buyahead` is the strongest hand-written producer
  (`entity_planner`) with one change: it keeps an input stock of k extra cycles of its own
  recent input use.
- k = 1 is registered and decides; k = 2 is reported.
- Each version was paired with two orderers (MWWk and round-robin) on EXP-0057's 71
  world-episodes, with the supply observer on.

**Validity.** The `entity_planner` control rows equal EXP-0057, EXP-0073 and EXP-0075 at TTL 2
exactly: 213 of 213. There are 0 contract failures in all three runs. Demand and shocks are
identical across producers (equal digests).

**What happened (12 producers decide).**

| orderer | entity_planner | buy-ahead k = 1 | buy-ahead k = 2 | mean floor |
|---|---|---|---|---|
| MWWk | 2,096 | 2,323 (+217) | 2,669 (+547) | 146.5 |
| round-robin | 2,163 | 2,398 (+224) | 2,761 (+574) | 146.5 |

- **The registered reading is `buyahead_helps`.** With either orderer, k = 1 beats
  `entity_planner` by more than the noise floor. It is ahead beyond its floor on 12 / 11 of 23
  worlds, and below on none.
- **No side effects.** No episode dies, and run-down check failures change by +4 (MWWk, 70 vs 66
  of 86) and −1 (RR), inside the registered +5 limit.
- **k = 2 gains more than twice as much:** +27–28% consumer units, ahead beyond the floor on 18 of 23
  worlds. It too has no deaths and no extra run-down beyond the limit.
- **At TTL 2, k = 2 already beats EXP-0075's TTL 4** (MWWk 2,669 vs 2,545; RR 2,761 vs 2,690).
- **At 42 producers (reported):** `buyahead_helps` for both k. k = 1 gives +462 / +418 against a
  floor of 362; k = 2 gives +1,184 / +1,176 (+20%).

**Where the gain comes from** (lost consumer units, 12 producers, MWWk; RR in brackets):

| cause | entity_planner | k = 1 | k = 2 |
|---|---|---|---|
| input order died upstream (supplier's reservation expired) | 155,974 (86,254) | 134,226 (73,037) | 104,938 (54,290) |
| input arrived late | 173,893 (88,283) | 173,476 (87,741) | 177,243 (93,767) |
| input outbid / no-fit upstream | 51,424 (29,331) | 51,288 (30,395) | 48,862 (25,396) |
| contention (stock used by another assembly) | 54,598 (25,417) | 58,357 (27,658) | 63,544 (31,478) |
| capacity-short | 27,894 (9,935) | 27,035 (9,965) | 25,756 (9,532) |
| outbid | 46,922 (13,569) | 46,968 (13,482) | 44,125 (14,151) |

- **The gain again comes from input orders that used to die upstream** (−14% at k = 1, −33% at
  k = 2). Suppliers that hold their own inputs in stock no longer let a reservation expire while
  they wait.
- **"Input arrived late" at the final assembler does not fall** (flat at k = 1, +2% at k = 2). The
  direct late loss is not fixed by stocking at the assembler.
- **Contention rises.** More stock is shared by more live assemblies.
- **The cost is stock and investment.**
  - Held stock rises: 132 → 156 → 180 units per cycle at 12 producers (MWWk, +18% / +37%); at 42,
    814 → 899 → 982.
  - The investment rejection rate rises 0.058 → 0.064 → 0.067.

**What it means.**
- **Buying inputs ahead is a real producer-side improvement at today's TTL.** The simple
  k = 1 rule passes the registered test with both orderers, and k = 2 adds about a quarter more
  consumer units, more than a longer TTL gave.
- **The gain has not saturated at k = 2.** The benefit doubles from k = 1 to k = 2, and side
  effects stay small.
- **Where it works:** it works through the supply chain's intermediate producers (died-upstream
  losses), not at the final assembler (late losses).
- **It is the strongest known producer now.** Under the "roster holds the strongest policy" rule,
  it belongs in the next roster draft. That is a human gate.

## Details
- **Driver:** `supply_trace --order-ttl 2 --producer P` (tasks 7.14 / 7.17a / 7.18), three runs in
  one `python -u -c` loop at commit `0e6ecdd6a9f1…`, from a clean worktree.
- **Inputs:** atlas v2 `7517314f…`, EXP-0012 `d25388bf…` (pinned only), EXP-0057 `episodes.csv`
  `69656f9e…` and EXP-0073 ttl2 `19888e26…` (the control's references).
- **Rows:** 213 per producer. Control neutrality 213 / 213 vs EXP-0057 and EXP-0073 (driver), and vs
  EXP-0075 ttl2 on every shared column (`read_exp0076.py`). Contract failures are header-only.
- **Reading rules** (fixed in `read_exp0076.py` before the pull):
  - diff_w = the mean of paired (episode, replicate) units differences per world;
  - floor_w = the atlas v2 `noise_floor`;
  - helps = mean diff > mean floor, with no extra deaths and run-down failures up ≤ 5
    (absolute, per orderer), for both orderers;
  - harmful = below −floor or extra deaths for either orderer; else no_gain.

  | cell | k = 1: diff / floor / beyond / below / Δ run-down | k = 2 |
  |---|---|---|
  | 12 MWWk | +217.2 / 146.5 / 12 / 0 / +4 | +546.8 / 146.5 / 18 / 0 / +4 |
  | 12 RR | +224.0 / 146.5 / 11 / 0 / −1 | +573.6 / 146.5 / 18 / 0 / 0 |
  | 42 MWWk | +462.1 / 362.2 / 5 / 0 / +1 | +1,183.7 / 362.2 / 7 / 0 / 0 |
  | 42 RR | +418.0 / 362.2 / 4 / 0 / 0 | +1,176.0 / 362.2 / 6 / 0 / 0 |

  Readings: `buyahead_helps` for 12 / k = 1 (deciding), 12 / k = 2, 42 / k = 1 and 42 / k = 2.
- **Holding units per cycle** (mean), entity_planner / k = 1 / k = 2:
  - 12 MWWk 131.7 / 155.8 / 179.9; 12 RR 145.5 / 172.4 / 197.3;
  - 42 MWWk 813.5 / 898.6 / 982.4; 42 RR 934.9 / 1,014.3 / 1,087.2.
- **Investment rejection rate** (mean): 12 MWWk 0.058 / 0.064 / 0.067; 12 RR 0.065 / 0.069 /
  0.073; 42 MWWk 0.162 / 0.169 / 0.166; 42 RR 0.166 / 0.170 / 0.166.
- **Mean r_end / r_start:** 12 MWWk 0.664 / 0.664 / 0.660; 12 RR 0.685 / 0.685 / 0.682.
- **Cost:**
  - One fresh secure 16 vCPU pod in EUR-IS-1, `b1fo5g30rmstsu`. It was created at 12:36:22 UTC
    and terminated by ≈ 12:41:05; the runs started 12:37:32 / 12:38:35 / 12:39:39.
  - **≈ 1.25 vCPU-h billed** (estimate 1.6, stop 6, watchdog at 22 min). 32 vCPU secure requests
    (unpinned, EUR-IS-1, EU-RO-1) failed first, with no capacity and no pod.
  - Launch-fit 97.7 + 6 = 103.7 ≤ 125. **Tranche A ≈ 99.0 of 125.**
- **Files:**
  - `entity_planner/`, `entity_planner_buyahead/`, `entity_planner_buyahead_k2/` hold
    `spec.json`, `reading.json`, `supply_trace.csv`, `neutrality*.csv`,
    `contract_failures.csv`, and `artifacts/episodes.csv` / `producer_cycles.csv` (git-lfs;
    episodes SHA-256 `e8c3b520…` / `fdf4e3f5…` / `c6d0e13d…`);
  - `read_exp0076.py` → `reading_exp0076.json`; `launch.log`.

## Conclusions
- **`buyahead_helps`** (12 producers, k = 1, both orderers: +217 / +224 vs a floor of 146.5; no
  deaths; run-down +4 / −1). k = 2: +27–28%. 42 producers: helps for both k.
- **The recovered loss is mostly input orders dying upstream** (−14% / −33%). "Input arrived late"
  at the final assembler is unchanged. Costs: +18% / +37% held stock and slightly more investment
  rejections.
- **Open (user / supervisor):**
  - adding `entity_planner_buyahead` (which k) to the next roster draft;
  - whether a larger k or a per-level rule is worth testing, since the gain had not saturated at
    k = 2.
