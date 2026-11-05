# EXP-0078: "Reservations first" on top of buy-ahead k = 4

**Status: complete, valid.** Task 7.22b of `add-m4-orderer-redesign` (producer evaluation step 4).
Not an acceptance read.

## In plain English

**The question.** With a 4-cycle input buffer (k = 4), producers still lose about 178k consumer
units at 12 producers (MWWk) to inputs that arrive just too late at the final assembler. The
"reservations first" rule (rf) changes only how the producer spends its inventory room:
1. the inputs that accepted orders are waiting for come first;
2. the buffer may not use the last few units of room;
3. it asks for more inventory space as soon as room falls below what waiting orders need.

Does it beat k = 4? Two ablations show what each part does: rf1 = part 1 only, rf12 = parts 1 + 2.

**Validity.** Every row has the same demand and shock schedule as EXP-0077's k = 4 rows (213 / 213
per producer). 0 contract failures. The rules were pre-registered in tasks.md 7.22b before the
run.

**What happened (12 producers decide; consumer units per episode).**

| producer | MWWk | vs k = 4 | RR | vs k = 4 | reading |
|---|---|---|---|---|---|
| k = 4 (EXP-0077) | 3,101 | — | 3,200 | — | — |
| rf1: part 1 | 2,887 | −199 | 2,951 | −230 | `harmful` |
| rf12: parts 1 + 2 | 2,550 | −519 | 2,620 | −546 | `harmful` |
| **rf: parts 1 + 2 + 3** | **3,165** | **+59** | **3,259** | **+55** | **`no_gain`** |

(Mean noise floor 146.5. No deaths anywhere. Run-down failures change by −1 to +1.)

- **The registered rule reads `no_gain`.** It is slightly ahead of k = 4 with both orderers, but
  inside the floor: ahead beyond its floor on 5 / 3 of 23 worlds, below on 1 / 3.
- **It does cut the target loss.** Late inputs fall 178k → 159k (−11%) with MWWk, and 90k → 83k
  (−9%) with RR. But input orders dying upstream rise 77k → 85k, so most of the gain is spent.
- **The parts pull in opposite directions:**
  - **Part 1 alone hurts** (−199 / −230). Giving the waiting orders' inputs the room first takes
    room from the buffer of other inputs, and late losses rise to 197k. 31k of those late units
    are blocked by headroom.
  - **Part 2 hurts more** (rf12 −519 / −546). Holding room back from the buffer shrinks it: held
    stock 233 → 185 units per cycle, and orders dying upstream rise to 112k.
  - **Part 3, expanding inventory earlier, recovers it all.** rf is +577 / +601 over rf12. Room is
    the binding resource, and more room helps where re-allocating it does not.
- **The cost of part 3:** investment rejections rise (0.078 → 0.091 MWWk), since more expansion
  orders compete for the fund.
- **At 42 producers (reported):** rf `no_gain` (+47 / +122 vs a floor of 362). rf1 and rf12
  `harmful` (−530 / −550 and −1,418 / −1,453).

**What it means.**
- **Re-allocating the inventory room does not help.** The room buy-ahead uses is as valuable as the
  room waiting orders need, so giving one to the other only moves the loss: late losses rise with
  part 1, losses upstream with part 2.
- **The only part that adds value is getting more room** (part 3). The data don't test whether it
  helps alone on k = 4, because no "part 3 only" arm was run.
- **The late-input loss at the final assembler is mostly not a room-allocation problem.** The order
  for the input still comes one cycle after the acceptance and arrives about one cycle too late.
  No rule over today's replenishment command can order before the acceptance.

## Details
- **Driver:** `supply_trace --order-ttl 2 --producer P` for P in rf / rf1 / rf12, one `-c` loop at
  commit `a79c7cf` (rule + observer additions + the state-leak fix), from a clean worktree. k = 4
  rows are reused from EXP-0077.
- **Inputs:** atlas v2 `7517314f…`, EXP-0012 `d25388bf…` (pinned only).
- **Reading** (`read_exp0078.py`; rules from tasks.md 7.22b):

  | cell | rf: diff / beyond / below / Δ run-down | rf1 | rf12 |
  |---|---|---|---|
  | 12 MWWk | +58.8 / 5 / 1 / −1 | −199.2 / 0 / 8 / +1 | −518.7 / 0 / 14 / −1 |
  | 12 RR | +55.3 / 3 / 3 / +1 | −230.0 / 0 / 9 / +1 | −545.7 / 0 / 12 / −1 |
  | 42 MWWk | +47.4 / 2 / 1 / −1 | −529.9 / 0 / 3 / 0 | −1,417.7 / 0 / 6 / −1 |
  | 42 RR | +122.2 / 4 / 2 / 0 | −550.1 / 1 / 5 / 0 | −1,452.7 / 0 / 6 / 0 |

  Mean floors: 146.5 (12), 362.2 (42). Readings: rf `no_gain` at 12 and 42; rf1 / rf12 `harmful` at
  12 and 42.
- **Loss classes, 12 MWWk** (k4 / rf1 / rf12 / rf):
  - input arrived late 178,082 / 197,238 / 192,533 / 159,263;
  - of which headroom-blocked — / 30,736 / 25,950 / 9,067 (not measured for k4: EXP-0077 predates
    the column);
  - died upstream 76,945 / 82,122 / 111,973 / 84,670;
  - contention 67,581 / 67,364 / 60,943 / 68,367;
  - capacity 24,594 / 24,380 / 26,628 / 23,656;
  - held stock 233.1 / 214.9 / 185.1 / 225.2; investment rejection 0.078 / 0.074 / 0.072 / 0.091;
  - run-down fails 71 / 72 / 70 / 70.
- **12 RR** (k4 / rf1 / rf12 / rf): late 90,450 / 99,269 / 98,578 / 82,613; headroom-blocked — /
  14,804 / 14,591 / 4,788; died upstream 39,482 / 44,694 / 59,670 / 44,580; held stock 250.3 /
  231.2 / 200.4 / 243.2.
- **Cost:**
  - One fresh secure 16 vCPU pod in EUR-IS-1, `gss18kgeyoh9ik`, ≈ 13:44:43–13:51:52 UTC. The runs
    started 13:47:15 / 13:48:40 / 13:50:05.
  - **≈ 1.9 vCPU-h billed** (estimate 1.6, stop 4, watchdog 14 min, not reached).
  - Launch-fit 106.2 + 4 = 110.2 ≤ 125. **Tranche A ≈ 108.1 of 125.**
- **Files:**
  - `entity_planner_buyahead_k4_rf{,1,12}/` hold `spec.json`, `reading.json`, `supply_trace.csv`,
    `contract_failures.csv`, and `artifacts/episodes.csv` / `producer_cycles.csv` (git-lfs;
    episodes SHA-256 `c7ce86a8…` / `f897dcb6…` / `69ae116f…`);
  - `read_exp0078.py` → `reading_exp0078.json`; `launch.log`.

## Conclusions
- **`no_gain`** for "reservations first" (12: +59 / +55 vs a floor of 146.5; 42 also `no_gain`).
  Buy-ahead k = 4 stays the strongest producer.
- **Diagnosed from the ablations:**
  - re-allocating inventory room toward waiting orders (part 1) or away from the buffer (part 2)
    loses units, moving the loss to late or upstream classes;
  - only expanding the room (part 3) adds value;
  - the remaining late loss is the one-cycle ordering delay after acceptance, which the
    replenishment command cannot remove.
- **Open, untested:** "expand earlier" alone on k = 4.
