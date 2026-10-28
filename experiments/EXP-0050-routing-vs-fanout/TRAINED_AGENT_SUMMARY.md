# How the trained agents perform (EXP-0050 summary)

Plain-English summary of EXP-0050 for sharing. Full record: `RESULT.md` in this directory.

## Bottom line

The trained ordering agent is the best router once each product has **4 or more possible
makers**. With only 2 makers (the setting the current gate worlds use) it loses slightly to the
simplest blind rule.

## What was measured

- **Worlds.** 64 worlds, each a three-layer supply chain (raw materials → parts → final
  products), at 12 or 42 producers. **Fanout** = how many factories may make each product (2,
  4, 7 or 14). In **dense** worlds every factory of a layer can make every product of that layer.
  Each world was played calm and with shocks (capacity hits and producer outages).
- **Score.** Consumer units delivered per episode (100 cycles), averaged over the 4 worlds of
  each cell. Higher is better. **Fill** = delivered ÷ ordered units (1.0 = every order filled).
- **Trained agents.**
  - **trained**: the trained ordering agent with the hand-written producer.
  - **portable**: the full trained stack (trained producer + trained orderer). It scores almost
    identically to *trained*, so the orderer does nearly all the work.
- **Opponents.**
  - **round_robin**: sends every order whole to a maker that rotates each cycle. Blind, but strong.
  - **random**: random split over the makers; the rule that sets the current gate's bar.
  - **best hand-written**: the best of the 5 deterministic rules on the gate's roster, chosen per
    episode.

## Units delivered per episode (calm / shocked)

| world | trained | round_robin | random | best hand-written | trained's fill |
|---|---|---|---|---|---|
| 12 producers, fanout 2 | 500 / 458 | **520 / 484** | 491 / 446 | 478 / 428 | 0.89 / 0.88 |
| 12, fanout 4 | **556 / 554** | 534 / 550 | 532 / 524 | 544 / 522 | 0.99 / 0.98 |
| 42, fanout 2 | 1796 / 1681 | 1778 / **1703** | 1704 / 1635 | 1662 / 1553 | 0.93 / 0.86 |
| 42, fanout 4 | **1950 / 1852** | 1882 / 1826 | 1857 / 1751 | 1767 / 1606 | 0.98 / 0.98 |
| 42, fanout 7 | 1834 / 1816 | 1830 / 1814 | 1761 / 1737 | 1495 / 1393 | 0.97 / 0.97 |
| 42, fanout 14 | **1838 / 1874** | 1686 / 1723 | 1418 / 1428 | 1412 / 1335 | 0.95 / 0.94 |
| dense 12 | **494 / 514** | 482 / 500 | 470 / 480 | 454 / 461 | 0.99 / 0.98 |
| dense 42 | 1754 / 1782 | **1824 / 1790** | 1722 / 1709 | 1616 / 1539 | 0.93 / 0.96 |

## What the numbers mean

1. **It beats every hand-written rule and random in all 16 cells on average.** The difference is
   statistically clear in 12–13 of the 16. The margin over the best hand-written rule at 42
   producers grows from about +130 units at fanout 2 to +430–540 at fanout 14.
2. **It clearly beats round_robin at 42 producers with fanout 4 (+25 to +68 units) and fanout 14
   (+151 to +153).** At fanout 14 it fills about 95% of orders, round_robin 87%.
3. **It ties round_robin at fanout 7.** Both fill about 97%.
4. **It is behind round_robin at fanout 2,** by 21–26 units at 12 producers and 22 in the
   42-producer shocked cell (clear in 2 of the 4 cells). It may also be behind in the dense-42
   calm cell (−69 units), but that difference is not statistically clear.
5. **Its 12-producer wins are close to the ceiling.** At 12 producers with fanout 4, and on
   dense-12, fill is already 0.97–0.99, so these cells cannot separate good routers from great
   ones.

Each cell rests on 4 worlds; a difference counts as clear at t ≥ 3.2 (paired, world-clustered
standard error).

## Why it wins

The trained orderer splits orders across makers about as often as the hand-written rules that
fail (about 1.7 makers per order). The difference is that its split pieces almost always finish:

| 42 producers, fanout 14 | units wasted on failed split orders | units lost to orders expiring |
|---|---|---|
| trained | 15 | 90–98 |
| round_robin (never splits) | 0 | 258–264 |
| proportional | 160–202 | 467–571 |

It was trained to imitate a teacher that weights each maker by how much it can finish right now:
spare capacity **and** the input parts it already holds in stock. The hand-written rules look at
capacity only. Almost every lost unit in these worlds is an order that expired while a factory
waited for input parts, not one turned away for lack of room. That points to input availability
as the reason it wins, but it is a likely explanation, not a proven one.

## What this means

- The trained agent has real skill, and that skill shows only when there is a real choice of maker.
- The current gate worlds use fanout 2, the one setting where it does not shine. That is a
  further reason for the multi-dimension world atlas now being planned.
- **Not yet tested:** the check that would confirm the mechanism. It would (a) run the teacher
  rule directly as a router, (b) run a whole-order version of it, and (c) trace every expired
  order against its factory's input stock at routing time. It is part of the atlas plan.
