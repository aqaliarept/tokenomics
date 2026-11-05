# EXP-0074: Supply trace: why consumer units are lost on the producer side

**Status: complete, valid.** Task 7.15 of `add-m4-orderer-redesign` (producer evaluation, step 2;
notes/producer-eval.md). Not an acceptance read.

## In plain English

**The question.** On the loaded atlas worlds the strongest hand-written stack (`entity_planner`
producer + MWW orderer) delivers only about half of what the tightest atlas bound allows, and
every simple orderer sits in the same place. So the gap is probably on the supply side. When a
consumer order is lost, which producer failure lost it?
- the producer lacked capacity;
- it lacked an input it never ordered;
- it ordered the input but the input did not arrive (and why not);
- nothing upstream could make the input;
- the order was outbid;
- a shock.

**What was run.**
- EXP-0057's 71 world-episodes (23 worlds at 12 producers, 14 at 42), `entity_planner` + MWWk,
  2 replicates, horizon 100, TTL 2.
- A read-only observer traced every lost consumer unit back to its last auction and classified
  it. Each class is defined in `eval/supply_trace.py`.

**Validity.**
- **Neutral:** every one of the 142 rows equals EXP-0057's on all compared cells, so the observer
  changed nothing.
- **Consistent:** in every row the classes add up exactly to the routing tracer's lost units,
  reason by reason. There are 0 contract failures.

**What happened.** About three quarters of ordered consumer units are lost (534k lost vs 180k
delivered at 12 producers). Their causes, at 12 producers (42 in brackets):

| cause | share of lost units |
|---|---|
| **input ordered but not delivered in time (b2)** | **74.6%** (79.5%) |
| — the input arrived, but after the waiting reservation had expired (`late`) | 32.6% (31.7%) |
| — the input order itself died upstream, because the supplier's own reservation expired waiting for *its* inputs (`lost_ttl`) | 29.1% (30.6%) |
| — the input order was outbid / did not fit upstream | 9.6% (11.7%) |
| — the input order was never funded | 0.8% (2.2%) |
| inputs in stock, but another assembly used them first (b4) | 10.2% (7.0%) |
| outbid at the auction (c) | 8.8% (3.6%) |
| capacity-short (a): degraded 2.6%, held by reservations 1.5%, too big 0.9%, dropped 0.3% | 5.2% (6.9%) |
| input never ordered (b1) | 0.5% (2.0%) |
| outages (d) | 0.5% (1.0%) |
| nothing upstream could make the input (b3) | 0.1% (0.04%) |

**The registered reading:** `b2_not_delivered` at 12 producers, 64 points ahead of the runner-up
(b4), so no tie. The same holds at 42.

**What it means.**
- **The producer side loses consumer orders by timing its inputs.** Producers do order their
  inputs (b1 is 0.5%), upstream can make them (b3 is 0.1%), and capacity and restoration are
  minor. Restorations almost always complete: 37,571 of 37,741 at 12 producers.
- **The inputs arrive too late.** Either they come after the 2-cycle reservation window, or the
  input order dies one level up for the same reason.
- **It compounds.** About 29% of all losses are a supplier's reservation expiring while it waits
  for its own inputs. Of the replenishment units producers ordered, 47% were rejected (12
  producers) and 61% (42).
- **The decision to target is input buying: lead time and timing.** That means ordering inputs
  earlier or holding stock ahead of need, not buying more of them.
- **Whether a longer reservation lifetime would let the inputs arrive is task 7.17 (EXP-0075).**
  It re-runs the TTL sweep with the reservation lifetime following the TTL (7.16) and this
  observer on. This run is at TTL 2, where 7.16 changes nothing.

## Details
- **Driver:** `python -m tokenomics.eval.supply_trace` (task 7.14), commit `d0171680ec9b…`, from a
  clean worktree, so the uncommitted 7.16 change was off the pod.
- **Inputs:** atlas v2 `7517314f…` (content `220886a7cf12…`); EXP-0012 `orderer_scorer.pt`
  `d25388bf…` (pinned only); EXP-0057 `episodes.csv` `69656f9e…` (the neutrality reference).
- **Rows:** 142 (71 world-episodes × MWWk r0 / r1). `contract_failures.csv` is header-only.
  Neutrality: 142 / 142 equal (`neutrality.csv`). Consistency holds in every row, and the check
  counters (`history_missing`, `unregistered_orders`, `qty_mismatch`) are 0.
- **Reading** (`reading.json`, rule registered in task 7.15):
  - 12 producers: b2 0.746, b4 0.102, c 0.088, a 0.052, d 0.006, b1 0.005, b3 0.001 → largest
    b2, nothing within 5 points;
  - 42 producers: b2 0.795, a 0.069, b4 0.070, c 0.036, b1 0.020, d 0.010, b3 0.0004.
- **Lost / delivered consumer units:** 12: 534,218 / 180,255; 42: 715,962 / 331,788.
- **Producer-cycles** (`artifacts/producer_cycles.csv`; 103,200 at 12, 235,200 at 42):

  | measure | 12 producers | 42 producers |
  |---|---|---|
  | mean capacity ratio | 0.725 | 0.698 |
  | restoration placed / completed / rejected | 37,741 / 37,571 / 170 | 94,402 / 89,696 / 4,706 |
  | upgrade placed / completed / rejected | 12,092 / 8,947 / 3,145 | 47,149 / 25,668 / 21,481 |
  | replenishment units placed / delivered / rejected | 732,179 / 382,904 / 340,702 | 2,038,445 / 767,891 / 1,250,091 |

- **Class definitions and precedence** (b3 > b1 > b2 for several missing inputs): the
  `eval/supply_trace.py` docstring and notes/7.14.md. `b4_contention` was added in 7.14.
- **Cost:**
  - One fresh secure 16 vCPU pod, `63qn7q50chzele`. It was created at ≈ 09:55:43 UTC and
    terminated by 09:58:05; the jobs ran 09:57:14–09:57:50.
  - **≈ 0.64 vCPU-h billed** (estimate 1.6, stop 4, watchdog). A 32 vCPU secure request failed
    first (no capacity; no pod).
  - Launch-fit 90.2 + 4 = 94.2 ≤ 125. **Tranche A ≈ 90.8 of 125.**
- **Files:**
  - `spec.json`, `reading.json`, `neutrality.csv`, `supply_trace.csv` (per episode × class),
    `contract_failures.csv`;
  - `summarize_supply.py` → `summary.json` (class shares, producer-cycle totals);
  - `artifacts/episodes.csv` (git-lfs, `ba580f0a…`) and `artifacts/producer_cycles.csv` (git-lfs,
    `14886f3d…`).

## Conclusions
- **Reading `b2_not_delivered`** (12 producers: 74.6%; 42: 79.5%). The producer decision to
  target in step 3 is **input buying: its lead time and timing**.
- **Late inputs (33%) and input orders lost to an upstream reservation expiring (29%)** dominate.
  Capacity / restoration (5%), un-ordered inputs (0.5%) and structural upstream inability (0.1%)
  are minor.
- **Next:** 7.17 (EXP-0075) tests whether a longer reservation lifetime turns `late` /
  `lost_ttl` into deliveries. That separates "inputs need more time" from "inputs must be bought
  earlier".
