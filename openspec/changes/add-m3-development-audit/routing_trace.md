# Routing trace (task 2.1, design D2 check (c))

**What this is.** A single-episode, unit-level look at how nine ordering routers route orders on
two registry v2 development worlds, before EXP-0048 runs. It describes what the trace shows. It
does **not** name the cause of `random`'s edge: EXP-0048 decides that (design D2), and one
episode per world cannot.

**What was run.** One episode per (world, router): the first registry episode seed, horizon 100,
the `entity_planner` producer, each router under its `routing_audit` contract (all nine are
joint_stack) and action seed (`random` and `random_whole` use replicate 0,
`replicate_seed(world, episode, 0)`), `--workers 1`, one process at a time.

- `dev-sparse_wide-42-shocked-00`, episode 81752346: `random` wins 40 of 40 episodes on this
  world in EXP-0047.
- `dev-sparse_wide-12-calm-03`, episode 1142209014: the 12-calm world where a deterministic
  rule beats `random` by the largest margin in EXP-0047 `episodes.csv` (`proportional`, 572.0
  against `random`'s replicate mean 517.7 over the world's three episodes: +54.3 units per
  episode; the other 12-calm worlds range from −31.1 to +38.8).
- Routers: `even_split`, `proportional`, `blend_0.5`, `random` (r0), `round_robin`,
  `even_split_rotating`, `proportional_load_aware`, `proportional_whole`, `random_whole` (r0).

Scripts (kept in this change dir): `/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/trace_21.py`, one command per world × router, 4–25 s
each, run from `/srv/research/tokenomics/tokenomics` as
`../.venv/bin/python /srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/trace_21.py --world <world> --router <router> --out /tmp/m3a12/trace`;
tables rendered by `../.venv/bin/python /srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/report_21.py /tmp/m3a12/trace`. The raw output
(`/tmp/m3a12/trace/<world>__<router>.json` and the full per-order log `.decisions.jsonl`) is
**ephemeral** (not committed; rerun the commands to regenerate it; the domain is deterministic and
the reviewer's reruns reproduced the JSON exactly). Each command runs the episode three ways: with the
tracer plus a per-order decision log, plain, and with the tracer only. All three gave identical
episode statistics for all 18 commands, and the four roster routers reproduce EXP-0047's units
on these two world-episodes exactly (8 of 8: e.g. 42-shocked `random` r0 1545, `even_split`
1366; 12-calm `proportional` 572, `random` r0 515).

Terms. A *decision* is one order routed by one ordering agent in one cycle (an order that is
re-routed after a rejection counts again). *Shares* are the decode's softmax over the order's
eligible producers; the *integer split* is what the largest-remainder step makes of them.
*Wasted completed units* are units delivered for an order that was then rejected because another
of its sub-orders failed (whole-order completion).

## What the trace shows, per D2 rule defect

Units on these two episodes, for orientation only (single episodes, not a comparison result):
42-shocked: `round_robin` 1601, `random_whole` 1569, `even_split_rotating` 1554, `random` 1545,
`blend_0.5` 1441, `proportional_whole` 1399, `even_split` 1366, `proportional_load_aware` 1365,
`proportional` 1337. 12-calm: `proportional_load_aware` 574, `proportional` 572, `blend_0.5`
569, `round_robin` 555, `even_split_rotating` 547, `random_whole` 532, `random` 515,
`even_split` 493, `proportional_whole` 462. No episode ended in system death.

Consumer orders are small on both worlds: every consumer order has quantity 1–3 (42-shocked
`even_split`: 536 / 462 / 110 decisions of quantity 1 / 2 / 3; 12-calm: 100 / 115 / 85) and
exactly two eligible producers (fanout 2). Producer orders are mostly quantity 1–3 too, up to 10.

1. **Lowest-index tie-break: occurs, and is total for `even_split`.** With equal shares the
   decode hands the leftover unit to the lowest-index eligible producer.
   - `even_split` sends 100% of quantity-1 orders to the lower-index producer
     (`q1_lowest_index_share` 1.0 on both worlds, 2,425 and 671 quantity-1 orders). Replaying
     the decode, every tied remainder went to the lowest index: consumer 646 of 646 tied
     decisions (42-shocked), 185 of 185 (12-calm); producer 2,234 of 2,234 and 681 of 681.
     That is every quantity-1 and quantity-3 order.
   - `proportional`, `blend_0.5`, `proportional_load_aware` tie rarely (consumer 35 / 28 / 41 on
     42-shocked, 0 on 12-calm; producer 19–104), and then also to the lowest index. Their
     quantity-1 lowest-index share is 0.52 / 0.53 / 0.50 on 42-shocked and 0.70 / 0.69 / 0.64
     on 12-calm, where it comes from the throughput shares, not from ties.
   - `random`, `round_robin`, `even_split_rotating`, `random_whole` have no tied decisions and a
     quantity-1 lowest-index share of 0.48–0.51 on both worlds; `proportional_whole` (argmax,
     ties to the lowest index by design) 0.51 / 0.69.
2. **Quantity rounding: occurs for the share-based rules.** Rounding moved at least a quarter
   of a unit away from the shares on 646 of 1,108 consumer decisions for `even_split`, 633 of
   1,101 for `proportional`, 717 of 1,104 for `blend_0.5` and 637 of 1,101 for
   `proportional_load_aware` on 42-shocked (12-calm: 185, 199, 159, 208 of 300); on average
   0.27–0.31 units per decision. For `random` it is 140 of 1,109 (46 of 301), 0.07–0.09 units,
   because its shares are usually near 0 or 1. The rotating and whole routers emit exact
   integer splits, so 0.
3. **Zero-throughput routing: essentially absent.** `zero_throughput_route_share` is at most
   0.001 for every router on 42-shocked (routes to a producer that cannot make one unit at that
   moment) and 0 on 12-calm.
4. **Slot overflow: absent.** `overflow_orders` is 0 for every router on both worlds (no ordering
   agent held more orders than its `k_stream` slots).
5. **Split orders sunk by a failed sub-order (wasted completed units): occurs, for the splitting
   routers.** Share of consumer orders split over both producers: `even_split` 0.52,
   `even_split_rotating` 0.52, `blend_0.5` 0.50, `proportional` 0.41,
   `proportional_load_aware` 0.41, `random` 0.06, the whole routers 0 (42-shocked; 12-calm
   0.67 / 0.67 / 0.66 / 0.54 / 0.53 / 0.08 / 0). Wasted completed consumer units on 42-shocked:
   `even_split` 120 (117 orders), `proportional_load_aware` 114, `blend_0.5` 104,
   `proportional` 102, `even_split_rotating` 64, `random` 10, whole routers 0; producer orders
   105 / 32 / 80 / 55 / 72 / 7 / 0. On 12-calm they are small except `even_split` (consumer 34,
   producer 15) and `even_split_rotating` (15 / 7); `proportional` 2 / 3, `random` 4 / 2.

Other observations from the same aggregates, reported without a reading:
- The main reason units are lost is a reservation expiring while it waits for inputs (`ttl`):
  42-shocked consumer `lost_units_consumer_ttl` from 146 (`even_split_rotating`) to 358
  (`proportional_whole`), `random` 202; 12-calm from 7 (`proportional`, `blend_0.5`,
  `proportional_load_aware`) to 121 (`proportional_whole`), `random` 63. Auction rejections
  (`outage`, `no_fit`, `displaced`) lose at most a few consumer units. Unfunded producer orders
  (`lost_units_producer_never_bid`) are the largest producer-side loss (42-shocked 460–782).
- Routed-unit concentration (`herfindahl_mean`, 1.0 = one producer per product and cycle) is
  0.75–0.84 for the splitting rules and 0.90–1.0 for `random` and the whole routers.
- `proportional_whole` on 12-calm has an investment rejection rate of 0.83 (others 0–0.05) and
  542 never-bid producer units (others 90–154).

**Measurement defect for D2 (d), the fund channel: the scalar `fund_share_herfindahl` is
broken.** The tracer's `fund_share_herfindahl` is about 0 for every router (3e-5 on 42-shocked, 1.4e-4 to 1.9e-4 on 12-calm) and cannot separate
routers. It squares each producer's share *of the whole investment fund*, and only about 3% of
the fund is granted per cycle (tracer: 2.6–2.8% on 42-shocked, 2.9–3.4% on 12-calm; checked
independently on 12-calm `even_split` by summing every `Order.fund` grant: 6,987 tokens of
100 × 2,200 = 3.18%, equal to the tracer). Normalised over the granted amount, the concentration
is 0.039–0.047 on 42-shocked (1/N = 0.024) and 0.13–0.22 on 12-calm (1/N = 0.083). The
per-producer shares in `episodes.csv`'s `trace_fund_share_by_producer` column allow that
normalisation. Not changed here (task 1.1 code, outside this task's scope): the supervisor
scheduled the fix as task 2.2, before EXP-0048 runs (normalise over the fund granted in each
cycle, skip cycles without grants, keep the raw per-producer series).

## Per-router aggregates and per-order excerpts

### dev-sparse_wide-42-shocked-00 (episode 81752346)

Tracer aggregates (`RoutingTracer.summary()`, task 1.1), one column per router:

| metric | even_split | proportional | blend_0.5 | random | round_robin | even_split_rotating | proportional_load_aware | proportional_whole | random_whole |
|---|---|---|---|---|---|---|---|---|---|
| units | 1366 | 1337 | 1441 | 1545 | 1601 | 1554 | 1365 | 1399 | 1569 |
| dead | False | False | False | False | False | False | False | False | False |
| rejected_bid_units_consumer_outage | 9 | 1 | 4 | 11 | 4 | 6 | 1 | 0 | 8 |
| rejected_bid_units_consumer_no_fit | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 5 |
| rejected_bid_units_consumer_displaced | 1 | 0 | 0 | 7 | 7 | 1 | 0 | 0 | 4 |
| failed_bid_units_consumer_dropped | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| failed_bid_units_consumer_ttl | 276 | 313 | 214 | 202 | 169 | 146 | 281 | 358 | 194 |
| failed_bid_units_consumer_ttl_outage | 1 | 0 | 1 | 4 | 0 | 1 | 0 | 0 | 2 |
| lost_units_consumer_outage | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_no_fit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_displaced | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_dropped | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_ttl | 276 | 313 | 214 | 202 | 169 | 146 | 281 | 358 | 194 |
| lost_units_consumer_ttl_outage | 1 | 0 | 1 | 4 | 0 | 1 | 0 | 0 | 2 |
| lost_units_consumer_never_bid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| split_share_consumer | 0.516 | 0.411 | 0.501 | 0.059 | 0 | 0.517 | 0.407 | 0 | 0 |
| split_degree_consumer | 1.516 | 1.411 | 1.501 | 1.059 | 1 | 1.517 | 1.407 | 1 | 1 |
| wasted_orders_consumer | 117 | 97 | 97 | 7 | 0 | 60 | 106 | 0 | 0 |
| wasted_units_consumer | 120 | 102 | 104 | 10 | 0 | 64 | 114 | 0 | 0 |
| rejected_bid_units_producer_outage | 62 | 31 | 31 | 39 | 21 | 31 | 16 | 3 | 32 |
| rejected_bid_units_producer_no_fit | 3 | 0 | 0 | 57 | 51 | 10 | 0 | 5 | 87 |
| rejected_bid_units_producer_displaced | 21 | 0 | 4 | 17 | 17 | 11 | 8 | 0 | 25 |
| failed_bid_units_producer_dropped | 4 | 0 | 2 | 4 | 7 | 3 | 0 | 0 | 4 |
| failed_bid_units_producer_ttl | 220 | 373 | 179 | 175 | 225 | 113 | 328 | 260 | 199 |
| failed_bid_units_producer_ttl_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_outage | 13 | 5 | 4 | 4 | 0 | 1 | 0 | 0 | 3 |
| lost_units_producer_no_fit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| lost_units_producer_displaced | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_dropped | 5 | 0 | 2 | 4 | 7 | 3 | 0 | 0 | 4 |
| lost_units_producer_ttl | 219 | 373 | 179 | 175 | 225 | 113 | 328 | 260 | 199 |
| lost_units_producer_ttl_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_never_bid | 655 | 782 | 621 | 460 | 479 | 595 | 694 | 511 | 465 |
| split_share_producer | 0.329 | 0.224 | 0.309 | 0.053 | 0 | 0.327 | 0.215 | 0 | 0 |
| split_degree_producer | 1.329 | 1.224 | 1.309 | 1.053 | 1 | 1.327 | 1.215 | 1 | 1 |
| wasted_orders_producer | 87 | 40 | 67 | 6 | 0 | 62 | 29 | 0 | 0 |
| wasted_units_producer | 105 | 55 | 80 | 7 | 0 | 72 | 32 | 0 | 0 |
| herfindahl_mean | 0.782 | 0.844 | 0.796 | 0.902 | 0.914 | 0.783 | 0.839 | 1 | 0.931 |
| q1_lowest_index_share | 1 | 0.521 | 0.527 | 0.497 | 0.502 | 0.514 | 0.501 | 0.508 | 0.49 |
| q1_orders | 2425 | 2212 | 2493 | 2053 | 1980 | 2523 | 2233 | 1692 | 1947 |
| zero_throughput_route_share | 0.001 | 0 | 0 | 0.001 | 0.001 | 0.001 | 0 | 0 | 0.001 |
| overflow_orders | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fund_share_herfindahl | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| investment_rejection_rate | 0.018 | 0.098 | 0.029 | 0.004 | 0.006 | 0.011 | 0.092 | 0.047 | 0.011 |

Decision-level checks (every routed order of the episode, consumer class; replayed from the router's action):

| router | q histogram | decisions | multi_eligible | split_decisions | tie_decisions | tie_lowest_index_won | small_q_decisions | small_q_rounded | rounding_units_moved_mean | overflow_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| even_split | {'1': 536, '2': 462, '3': 110} | 1108 | 1108 | 572 | 646 | 646 | 1108 | 646 | 0.292 | 0 |
| proportional | {'1': 529, '2': 462, '3': 110} | 1101 | 1101 | 452 | 35 | 35 | 1101 | 633 | 0.284 | 0 |
| blend_0.5 | {'1': 532, '2': 462, '3': 110} | 1104 | 1104 | 553 | 28 | 28 | 1104 | 717 | 0.308 | 0 |
| random | {'1': 530, '2': 469, '3': 110} | 1109 | 1109 | 65 | 0 | 0 | 1109 | 140 | 0.072 | 0 |
| round_robin | {'1': 528, '2': 468, '3': 111} | 1107 | 1107 | 0 | 0 | 0 | 1107 | 0 | 0 | 0 |
| even_split_rotating | {'1': 535, '2': 462, '3': 110} | 1107 | 1107 | 572 | 0 | 0 | 1107 | 0 | 0 | 0 |
| proportional_load_aware | {'1': 529, '2': 462, '3': 110} | 1101 | 1101 | 448 | 41 | 41 | 1101 | 637 | 0.287 | 0 |
| proportional_whole | {'1': 528, '2': 462, '3': 110} | 1100 | 1100 | 0 | 0 | 0 | 1100 | 0 | 0 | 0 |
| random_whole | {'1': 529, '2': 470, '3': 110} | 1109 | 1109 | 0 | 0 | 0 | 1109 | 0 | 0 | 0 |

Same, producer orders (replenishment + investment):

| router | q histogram | decisions | multi_eligible | split_decisions | tie_decisions | tie_lowest_index_won | small_q_decisions | small_q_rounded | rounding_units_moved_mean | overflow_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| even_split | {'1': 1912, '2': 568, '3': 307, '4': 45, '5': 15, '6': 1} | 2848 | 2848 | 936 | 2234 | 2234 | 2787 | 2219 | 0.392 | 0 |
| proportional | {'1': 1686, '2': 566, '3': 297, '4': 57, '5': 11, '6': 10, '7': 5, '8': 5, '9': 1, '10': 1} | 2639 | 2639 | 592 | 86 | 86 | 2549 | 1498 | 0.293 | 0 |
| blend_0.5 | {'1': 1977, '2': 585, '3': 297, '4': 24, '5': 12, '6': 4} | 2899 | 2899 | 896 | 85 | 85 | 2859 | 2205 | 0.357 | 0 |
| random | {'1': 1525, '2': 575, '3': 415, '4': 37, '5': 11, '6': 6} | 2569 | 2569 | 136 | 0 | 0 | 2515 | 321 | 0.074 | 0 |
| round_robin | {'1': 1452, '2': 580, '3': 433, '4': 47, '5': 11, '6': 14, '7': 2} | 2539 | 2539 | 0 | 0 | 0 | 2465 | 0 | 0 | 0 |
| even_split_rotating | {'1': 2017, '2': 695, '3': 263, '4': 15, '5': 7} | 2997 | 2997 | 980 | 0 | 0 | 2975 | 0 | 0 | 0 |
| proportional_load_aware | {'1': 1712, '2': 522, '3': 320, '4': 48, '5': 14, '6': 10, '7': 4, '8': 4, '9': 3} | 2637 | 2637 | 568 | 104 | 104 | 2554 | 1529 | 0.299 | 0 |
| proportional_whole | {'1': 1164, '2': 439, '3': 452, '4': 69, '5': 21, '6': 7, '7': 3, '8': 4, '10': 1} | 2160 | 2160 | 0 | 0 | 0 | 2055 | 0 | 0 | 0 |
| random_whole | {'1': 1418, '2': 566, '3': 455, '4': 48, '5': 7, '6': 10, '7': 1, '8': 3} | 2508 | 2508 | 0 | 0 | 0 | 2439 | 0 | 0 | 0 |

Per-order excerpt: first 3 consumer orders of each of the first 3 cycles with consumer demand. `split` = integer split decode applied; outcomes = this cycle's bids (producer, units, outcome, reason); `resolved` = (completed, completed units) if the order resolved this cycle.

<details><summary>even_split</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[0.5, 0.5] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_3 1u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[0.5, 0.5] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_3 1u accepted] resolved=None
c3 order_c0__200 o96 q=2 elig=['p2_0', 'p2_13'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_13': 1} -> [p2_0 1u accepted; p2_13 1u accepted; p2_13 1u completed] resolved=None
c3 order_c1__201 o97 q=1 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1} -> [p2_0 1u accepted; p2_0 1u completed] resolved=[True, 1]
c3 order_c2__203 o98 q=2 elig=['p2_2', 'p2_3'] shares=[0.5, 0.5] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_2 1u completed; p2_3 1u accepted] resolved=None
```

</details>

<details><summary>proportional</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[0.588, 0.412] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[0.4, 0.6] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[0.872, 0.128] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.621, 0.379] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[0.394, 0.606] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[0.85, 0.15] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__200 o97 q=2 elig=['p2_0', 'p2_13'] shares=[0.905, 0.095] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c1__201 o98 q=1 elig=['p2_0', 'p2_1'] shares=[0.667, 0.333] split={'p2_0': 1} -> [p2_0 1u accepted; p2_0 1u completed] resolved=[True, 1]
c3 order_c2__203 o99 q=2 elig=['p2_2', 'p2_3'] shares=[0.848, 0.152] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
```

</details>

<details><summary>blend_0.5</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[0.544, 0.456] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[0.449, 0.551] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[0.723, 0.277] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_3 1u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.561, 0.439] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[0.44, 0.56] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[0.747, 0.253] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_3 1u accepted] resolved=None
c3 order_c0__200 o98 q=2 elig=['p2_0', 'p2_13'] shares=[0.755, 0.245] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c1__201 o99 q=1 elig=['p2_0', 'p2_1'] shares=[0.586, 0.414] split={'p2_0': 1} -> [p2_0 1u accepted; p2_0 1u completed] resolved=[True, 1]
c3 order_c2__203 o100 q=2 elig=['p2_2', 'p2_3'] shares=[0.846, 0.154] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
```

</details>

<details><summary>random</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[0.057, 0.943] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[0.249, 0.751] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[1.0, 0.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[0.001, 0.999] split={'p2_3': 2} -> [p2_3 2u accepted] resolved=None
c3 order_c0__200 o96 q=2 elig=['p2_0', 'p2_13'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c1__201 o97 q=1 elig=['p2_0', 'p2_1'] shares=[0.992, 0.008] split={'p2_0': 1} -> [p2_0 1u accepted; p2_0 1u completed] resolved=[True, 1]
c3 order_c2__203 o98 q=2 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
```

</details>

<details><summary>round_robin</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[1.0, 0.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[0.0, 1.0] split={'p2_3': 2} -> [p2_3 2u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__200 o93 q=2 elig=['p2_0', 'p2_13'] shares=[0.0, 1.0] split={'p2_13': 2} -> [p2_13 2u rejected(no_fit)] resolved=None
c3 order_c1__201 o94 q=1 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c3 order_c2__203 o95 q=2 elig=['p2_2', 'p2_3'] shares=[0.0, 1.0] split={'p2_3': 2} -> [p2_3 2u accepted] resolved=None
```

</details>

<details><summary>even_split_rotating</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[0.5, 0.5] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_3 1u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[0.5, 0.5] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_3 1u accepted] resolved=None
c3 order_c0__200 o95 q=2 elig=['p2_0', 'p2_13'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_13': 1} -> [p2_0 1u accepted; p2_13 1u rejected(displaced)] resolved=None
c3 order_c1__201 o96 q=1 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c3 order_c2__203 o97 q=2 elig=['p2_2', 'p2_3'] shares=[0.5, 0.5] split={'p2_2': 1, 'p2_3': 1} -> [p2_2 1u accepted; p2_3 1u accepted] resolved=None
```

</details>

<details><summary>proportional_load_aware</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[0.588, 0.412] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[0.385, 0.615] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[0.867, 0.133] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.621, 0.379] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[0.375, 0.625] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[0.846, 0.154] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__200 o95 q=2 elig=['p2_0', 'p2_13'] shares=[0.905, 0.095] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c1__201 o96 q=1 elig=['p2_0', 'p2_1'] shares=[0.636, 0.364] split={'p2_0': 1} -> [p2_0 1u accepted; p2_0 1u completed] resolved=[True, 1]
c3 order_c2__203 o97 q=2 elig=['p2_2', 'p2_3'] shares=[0.848, 0.152] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
```

</details>

<details><summary>proportional_whole</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__200 o99 q=2 elig=['p2_0', 'p2_13'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c1__201 o100 q=1 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 1} -> [p2_0 1u accepted] resolved=None
c3 order_c2__203 o101 q=2 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
```

</details>

<details><summary>random_whole</summary>

```
c1 order_c0__201 o1 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c1 order_c1__202 o2 q=2 elig=['p2_1', 'p2_2'] shares=[1.0, 0.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c1 order_c2__203 o3 q=2 elig=['p2_2', 'p2_3'] shares=[0.0, 1.0] split={'p2_3': 2} -> [p2_3 2u accepted] resolved=None
c2 order_c0__201 o68 q=2 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c1__202 o69 q=2 elig=['p2_1', 'p2_2'] shares=[1.0, 0.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c2__203 o70 q=2 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__200 o96 q=2 elig=['p2_0', 'p2_13'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted; p2_0 2u completed] resolved=[True, 2]
c3 order_c1__201 o97 q=1 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c3 order_c2__203 o98 q=2 elig=['p2_2', 'p2_3'] shares=[0.0, 1.0] split={'p2_3': 2} -> [p2_3 2u accepted] resolved=None
```

</details>


### dev-sparse_wide-12-calm-03 (episode 1142209014)

Tracer aggregates (`RoutingTracer.summary()`, task 1.1), one column per router:

| metric | even_split | proportional | blend_0.5 | random | round_robin | even_split_rotating | proportional_load_aware | proportional_whole | random_whole |
|---|---|---|---|---|---|---|---|---|---|
| units | 493 | 572 | 569 | 515 | 555 | 547 | 574 | 462 | 532 |
| dead | False | False | False | False | False | False | False | False | False |
| rejected_bid_units_consumer_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rejected_bid_units_consumer_no_fit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| rejected_bid_units_consumer_displaced | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 3 |
| failed_bid_units_consumer_dropped | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| failed_bid_units_consumer_ttl | 54 | 7 | 7 | 63 | 28 | 23 | 7 | 121 | 51 |
| failed_bid_units_consumer_ttl_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_no_fit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_displaced | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_dropped | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_ttl | 54 | 7 | 7 | 63 | 28 | 23 | 7 | 121 | 51 |
| lost_units_consumer_ttl_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_consumer_never_bid | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| split_share_consumer | 0.667 | 0.543 | 0.663 | 0.076 | 0 | 0.667 | 0.533 | 0 | 0 |
| split_degree_consumer | 1.667 | 1.543 | 1.663 | 1.076 | 1 | 1.667 | 1.533 | 1 | 1 |
| wasted_orders_consumer | 32 | 1 | 4 | 4 | 0 | 13 | 0 | 0 | 0 |
| wasted_units_consumer | 34 | 2 | 5 | 4 | 0 | 15 | 0 | 0 | 0 |
| rejected_bid_units_producer_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rejected_bid_units_producer_no_fit | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 6 |
| rejected_bid_units_producer_displaced | 0 | 0 | 0 | 0 | 7 | 5 | 0 | 0 | 12 |
| failed_bid_units_producer_dropped | 0 | 0 | 0 | 6 | 3 | 0 | 0 | 0 | 0 |
| failed_bid_units_producer_ttl | 31 | 5 | 2 | 33 | 8 | 8 | 3 | 55 | 43 |
| failed_bid_units_producer_ttl_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_no_fit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_displaced | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_dropped | 0 | 0 | 0 | 6 | 3 | 0 | 0 | 0 | 0 |
| lost_units_producer_ttl | 31 | 5 | 2 | 33 | 8 | 8 | 3 | 55 | 43 |
| lost_units_producer_ttl_outage | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lost_units_producer_never_bid | 154 | 90 | 94 | 109 | 107 | 134 | 90 | 542 | 97 |
| split_share_producer | 0.346 | 0.337 | 0.332 | 0.068 | 0 | 0.363 | 0.326 | 0 | 0 |
| split_degree_producer | 1.346 | 1.337 | 1.332 | 1.068 | 1 | 1.363 | 1.326 | 1 | 1 |
| wasted_orders_producer | 14 | 2 | 1 | 1 | 0 | 5 | 1 | 0 | 0 |
| wasted_units_producer | 15 | 3 | 1 | 2 | 0 | 7 | 2 | 0 | 0 |
| herfindahl_mean | 0.76 | 0.785 | 0.764 | 0.905 | 0.979 | 0.746 | 0.78 | 1 | 0.951 |
| q1_lowest_index_share | 1 | 0.696 | 0.691 | 0.506 | 0.481 | 0.512 | 0.641 | 0.685 | 0.508 |
| q1_orders | 671 | 642 | 713 | 490 | 401 | 692 | 666 | 321 | 431 |
| zero_throughput_route_share | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| overflow_orders | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fund_share_herfindahl | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| investment_rejection_rate | 0.006 | 0.019 | 0.025 | 0 | 0.013 | 0.052 | 0.05 | 0.832 | 0.019 |

Decision-level checks (every routed order of the episode, consumer class; replayed from the router's action):

| router | q histogram | decisions | multi_eligible | split_decisions | tie_decisions | tie_lowest_index_won | small_q_decisions | small_q_rounded | rounding_units_moved_mean | overflow_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| even_split | {'1': 100, '2': 115, '3': 85} | 300 | 300 | 200 | 185 | 185 | 300 | 185 | 0.308 | 0 |
| proportional | {'1': 100, '2': 115, '3': 85} | 300 | 300 | 163 | 0 | 0 | 300 | 199 | 0.294 | 0 |
| blend_0.5 | {'1': 100, '2': 115, '3': 85} | 300 | 300 | 199 | 0 | 0 | 300 | 159 | 0.27 | 0 |
| random | {'1': 100, '2': 115, '3': 86} | 301 | 301 | 23 | 0 | 0 | 301 | 46 | 0.088 | 0 |
| round_robin | {'1': 100, '2': 115, '3': 85} | 300 | 300 | 0 | 0 | 0 | 300 | 0 | 0 | 0 |
| even_split_rotating | {'1': 100, '2': 115, '3': 85} | 300 | 300 | 200 | 0 | 0 | 300 | 0 | 0 | 0 |
| proportional_load_aware | {'1': 100, '2': 115, '3': 85} | 300 | 300 | 160 | 0 | 0 | 300 | 208 | 0.301 | 0 |
| proportional_whole | {'1': 100, '2': 115, '3': 85} | 300 | 300 | 0 | 0 | 0 | 300 | 0 | 0 | 0 |
| random_whole | {'1': 100, '2': 115, '3': 88} | 303 | 303 | 0 | 0 | 0 | 303 | 0 | 0 | 0 |

Same, producer orders (replenishment + investment):

| router | q histogram | decisions | multi_eligible | split_decisions | tie_decisions | tie_lowest_index_won | small_q_decisions | small_q_rounded | rounding_units_moved_mean | overflow_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| even_split | {'1': 571, '2': 171, '3': 110, '4': 21} | 873 | 873 | 302 | 681 | 681 | 852 | 681 | 0.39 | 0 |
| proportional | {'1': 542, '2': 212, '3': 108, '4': 19, '5': 1} | 882 | 882 | 297 | 19 | 19 | 862 | 559 | 0.314 | 0 |
| blend_0.5 | {'1': 613, '2': 210, '3': 94, '4': 13} | 930 | 930 | 309 | 23 | 23 | 917 | 697 | 0.351 | 0 |
| random | {'1': 390, '2': 184, '3': 157, '4': 13, '5': 2} | 746 | 746 | 51 | 0 | 0 | 731 | 99 | 0.077 | 0 |
| round_robin | {'1': 301, '2': 162, '3': 220, '4': 14, '7': 1} | 698 | 698 | 0 | 0 | 0 | 683 | 0 | 0 | 0 |
| even_split_rotating | {'1': 593, '2': 255, '3': 80, '4': 3} | 931 | 931 | 338 | 0 | 0 | 928 | 0 | 0 | 0 |
| proportional_load_aware | {'1': 566, '2': 225, '3': 98, '4': 13, '5': 1} | 903 | 903 | 294 | 31 | 31 | 889 | 575 | 0.317 | 0 |
| proportional_whole | {'1': 221, '2': 150, '3': 169, '4': 23, '5': 2} | 565 | 565 | 0 | 0 | 0 | 540 | 0 | 0 | 0 |
| random_whole | {'1': 331, '2': 165, '3': 197, '4': 18, '5': 1, '6': 1} | 713 | 713 | 0 | 0 | 0 | 693 | 0 | 0 | 0 |

Per-order excerpt: first 3 consumer orders of each of the first 3 cycles with consumer demand. `split` = integer split decode applied; outcomes = this cycle's bids (producer, units, outcome, reason); `resolved` = (completed, completed units) if the order resolved this cycle.

<details><summary>even_split</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1} -> [p2_0 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[0.5, 0.5] split={'p2_2': 2, 'p2_3': 1} -> [p2_2 2u accepted; p2_3 1u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c3 order_c0__21 o33 q=3 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 2, 'p2_1': 1} -> [p2_0 2u accepted; p2_1 1u accepted] resolved=None
c3 order_c1__21 o34 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_0 1u completed; p2_1 1u accepted] resolved=None
c3 order_c2__22 o35 q=3 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 2, 'p2_2': 1} -> [p2_1 2u accepted; p2_2 1u accepted; p2_2 1u completed] resolved=None
```

</details>

<details><summary>proportional</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.9, 0.1] split={'p2_0': 1} -> [p2_0 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.116, 0.884] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[0.917, 0.083] split={'p2_2': 3} -> [p2_2 3u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[0.891, 0.109] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.143, 0.857] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.143, 0.857] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__21 o33 q=3 elig=['p2_0', 'p2_1'] shares=[0.878, 0.122] split={'p2_0': 3} -> [p2_0 3u accepted] resolved=None
c3 order_c1__21 o34 q=2 elig=['p2_0', 'p2_1'] shares=[0.878, 0.122] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c2__22 o35 q=3 elig=['p2_1', 'p2_2'] shares=[0.172, 0.828] split={'p2_1': 1, 'p2_2': 2} -> [p2_1 1u accepted; p2_2 2u accepted] resolved=None
```

</details>

<details><summary>blend_0.5</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.75, 0.25] split={'p2_0': 1} -> [p2_0 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.266, 0.734] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[0.768, 0.232] split={'p2_2': 2, 'p2_3': 1} -> [p2_2 2u accepted; p2_3 1u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[0.741, 0.259] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.287, 0.713] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.287, 0.713] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c3 order_c0__21 o33 q=3 elig=['p2_0', 'p2_1'] shares=[0.811, 0.189] split={'p2_0': 2, 'p2_1': 1} -> [p2_0 2u accepted; p2_1 1u accepted] resolved=None
c3 order_c1__21 o34 q=2 elig=['p2_0', 'p2_1'] shares=[0.811, 0.189] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c2__22 o35 q=3 elig=['p2_1', 'p2_2'] shares=[0.214, 0.786] split={'p2_1': 1, 'p2_2': 2} -> [p2_1 1u accepted; p2_2 2u accepted; p2_2 2u completed] resolved=None
```

</details>

<details><summary>random</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.046, 0.954] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.099, 0.901] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[0.866, 0.134] split={'p2_2': 3} -> [p2_2 3u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.003, 0.997] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__21 o32 q=3 elig=['p2_0', 'p2_1'] shares=[0.417, 0.583] split={'p2_0': 1, 'p2_1': 2} -> [p2_0 1u accepted; p2_0 1u completed; p2_1 2u accepted] resolved=None
c3 order_c1__21 o33 q=2 elig=['p2_0', 'p2_1'] shares=[0.102, 0.898] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c3 order_c2__22 o34 q=3 elig=['p2_1', 'p2_2'] shares=[0.007, 0.993] split={'p2_2': 3} -> [p2_2 3u accepted] resolved=None
```

</details>

<details><summary>round_robin</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 3} -> [p2_2 3u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[1.0, 0.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[1.0, 0.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c3 order_c0__21 o33 q=3 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 3} -> [p2_1 3u accepted] resolved=None
c3 order_c1__21 o34 q=2 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c3 order_c2__22 o35 q=3 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 3} -> [p2_2 3u accepted; p2_2 3u completed] resolved=[True, 3]
```

</details>

<details><summary>even_split_rotating</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[0.333, 0.667] split={'p2_2': 1, 'p2_3': 2} -> [p2_2 1u accepted; p2_3 2u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_1 1u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.5, 0.5] split={'p2_1': 1, 'p2_2': 1} -> [p2_1 1u accepted; p2_2 1u accepted] resolved=None
c3 order_c0__21 o33 q=3 elig=['p2_0', 'p2_1'] shares=[0.333, 0.667] split={'p2_0': 1, 'p2_1': 2} -> [p2_0 1u accepted; p2_0 1u completed; p2_1 2u accepted] resolved=None
c3 order_c1__21 o34 q=2 elig=['p2_0', 'p2_1'] shares=[0.5, 0.5] split={'p2_0': 1, 'p2_1': 1} -> [p2_0 1u accepted; p2_0 1u completed; p2_1 1u accepted] resolved=None
c3 order_c2__22 o35 q=3 elig=['p2_1', 'p2_2'] shares=[0.333, 0.667] split={'p2_1': 1, 'p2_2': 2} -> [p2_1 1u accepted; p2_2 2u accepted; p2_2 2u completed] resolved=None
```

</details>

<details><summary>proportional_load_aware</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.9, 0.1] split={'p2_0': 1} -> [p2_0 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.116, 0.884] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[0.917, 0.083] split={'p2_2': 3} -> [p2_2 3u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[0.891, 0.109] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.143, 0.857] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.152, 0.848] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__21 o32 q=3 elig=['p2_0', 'p2_1'] shares=[0.878, 0.122] split={'p2_0': 3} -> [p2_0 3u accepted] resolved=None
c3 order_c1__21 o33 q=2 elig=['p2_0', 'p2_1'] shares=[0.868, 0.132] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c2__22 o34 q=3 elig=['p2_1', 'p2_2'] shares=[0.172, 0.828] split={'p2_1': 1, 'p2_2': 2} -> [p2_1 1u accepted; p2_1 1u completed; p2_2 2u accepted] resolved=None
```

</details>

<details><summary>proportional_whole</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 1} -> [p2_0 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[1.0, 0.0] split={'p2_2': 3} -> [p2_2 3u accepted] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__21 o33 q=3 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 3} -> [p2_0 3u accepted] resolved=None
c3 order_c1__21 o34 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted] resolved=None
c3 order_c2__22 o35 q=3 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 3} -> [p2_2 3u accepted] resolved=None
```

</details>

<details><summary>random_whole</summary>

```
c1 order_c0__21 o1 q=1 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 1} -> [p2_1 1u accepted] resolved=None
c1 order_c1__22 o2 q=1 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 1} -> [p2_2 1u accepted] resolved=None
c1 order_c2__23 o3 q=3 elig=['p2_2', 'p2_3'] shares=[0.0, 1.0] split={'p2_3': 3} -> [p2_3 3u rejected(no_fit)] resolved=None
c2 order_c0__21 o22 q=2 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 2} -> [p2_1 2u accepted] resolved=None
c2 order_c1__22 o23 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c2 order_c2__22 o24 q=2 elig=['p2_1', 'p2_2'] shares=[0.0, 1.0] split={'p2_2': 2} -> [p2_2 2u accepted] resolved=None
c3 order_c0__21 o33 q=3 elig=['p2_0', 'p2_1'] shares=[0.0, 1.0] split={'p2_1': 3} -> [p2_1 3u accepted] resolved=None
c3 order_c1__21 o34 q=2 elig=['p2_0', 'p2_1'] shares=[1.0, 0.0] split={'p2_0': 2} -> [p2_0 2u accepted; p2_0 2u completed] resolved=[True, 2]
c3 order_c2__22 o35 q=3 elig=['p2_1', 'p2_2'] shares=[1.0, 0.0] split={'p2_1': 3} -> [p2_1 3u accepted] resolved=None
```

</details>


