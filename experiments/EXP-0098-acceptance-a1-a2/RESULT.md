# EXP-0098: producer acceptance policies A1 / A2 vs the k5_rf5 default (add-producer-acceptance-policy 3.1)

**Question.** Does letting a producer accept work it cannot run this cycle help? EXP-0057's worlds, TTL 2, MWWk
R = 2 and RR, observer on.
- **A1 (`ttl_feasible`):** accept also bids that fit the next cycle after the queued claims.
- **A2 (`ttl_inputs`):** A1, and decline composites whose inputs are neither in stock nor in flight.

The default is k5_rf5 under today's auction (EXP-0088's rows; the build proved it byte-identical). The reading was
pre-registered in tasks.md 3.1 before the data.

**Setup.**
- Code bedc717. `supply_trace --order-ttl 2 --producer entity_planner_buyahead_k5_rf5 --acceptance {ttl_feasible,
  ttl_inputs}`.
- Pod: secure 8 vCPU in **EUR-IS-1** (`nl2uq9fzwhuzrh`), 08:48–08:54 UTC. EUR-IS-1 refused at 08:46 and gave 8 vCPU
  on the first retry. Digests equal the default's.
- Read: `read_exp0098.py` → `reading_exp0098.json`.

## Readings

**A1: `harmful` at 12** (deciding) and at 42:

| | default units / episode | A1 − default (floor) | worlds beyond / below | deaths (default) | run-down fails (default) |
|---|---|---|---|---|---|
| 12 MWWk | 3,703.2 | **−208.0** (146.5) | 0 / 7 | 0 (0) | 61 (67) |
| 12 RR | 3,841.9 | **−146.5** (146.5) | 1 / 5 | **1** (0) | 31 (32) |
| 42 MWWk | 9,325.4 | −643.3 (362.2) | 0 / 6 | 0 | 52 (50) |
| 42 RR | 9,115.2 | −517.2 (362.2) | 1 / 5 | 0 | 26 (26) |

**A2: invalid, no reading.** All 213 rows are present with matching digests, but every job fails the supply
observer's consistency check (`supply_trace_consistency`: the classes sum below the traced TTL loss). The cause:
- A2 declines replenishment orders.
- The consumer units lost because of them are classed `b2_lost_declined` (`supply_trace._input_class`).
- That class is not among the registered `B2_SUBS`, so the class totals fall short of the trace.
- An observer bookkeeping gap, not a domain error. A fix (add `lost_declined` to `B2_SUBS`) and a rerun are needed
  before A2 can be read.

## Why A1 loses (reported only, from A1's own counters; 12 producers, MWWk)
- **Pending claims rarely complete.** 46,235 bids were accepted without fitting. Only **1,807 completed later** and
  1,667 failed at their own TTL.
- **The other ~42,800 (about 92%, by difference) failed with their parent order:** the order's TTL fired before the
  claim's next cycle. The producer cannot see the order's age, so A1 assumes one future cycle. The assumption
  **binds**.
- **The cost of a dead claim:** its tokens are burnt at acceptance, and its quantity, being accepted, is not
  re-routed while it waits. So it is lost, where today's auction would have declined it and let the orderer
  re-route.
- **This shows in the classes:** late inputs rise from 153.7k to 172.3k units and total lost units from 399.2k to
  417.5k. `e_declined` is 0, as designed: A1 declines only on capacity.
- **Fund gaming:**
  - 18,363 of A1's 29,251 declines (63%) were bids priced below the cycle's cut-off;
  - the fund share moved against the default is 3.3% (MWWk) / 3.2% (RR) of the fund per episode at 12, and
    5.5% / 3.6% at 42.

## Conclusions
- **A1 is `harmful`.** Accepting work that cannot run now mostly creates claims that die with their order, and
  locks quantity the orderer would otherwise re-route.
- **The binding constraint is information.** The policy needs the sub-order's remaining order TTL (own-order
  information the producer could be given) to accept-later only when a later cycle exists. That is the
  supervisor's anticipated variant.
- **A2 needs the observer fix and a rerun** (≈ 0.4 vCPU-h) before it can be read.
- **Spend:** ≈ 0.7 vCPU-h billed (stop 4).
