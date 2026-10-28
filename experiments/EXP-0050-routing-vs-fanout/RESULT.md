# EXP-0050 — Routing versus producers per product

**Status: complete.** 2,976 episode rows over 96 world-episodes (64 worlds in 16 cells), 15
subjects (R = 5 for the stochastic ones), horizon 100. Run commit
`2ef010ff9e860b46e1c23f21b5d71d925d33299b` (clean tree). Produced by
`python -m tokenomics.eval.routing_audit --plan exp0050` (see `spec.json`). Change
`add-m3-development-audit`, task 3.5, design D2 "Routing versus producers per product".

## The worlds, in one paragraph

Every world here is a three-layer supply chain: raw-material producers, then producers of parts
made from them, then producers of the final products consumers buy. Think of each product as
something a set of factories is allowed to make. In the **sparse_wide** worlds each layer has
many products (one per producer in the layer) and each product can be made by `fanout`
factories: fanout 2 means every product has exactly two possible makers (this is all the gate's
registry v2 uses, and all EXP-0048/0049 tested), fanout 14 at 42 producers means all 14 factories
of a layer can make every product of that layer. In the **dense** worlds each layer has only 4
products, and every factory of the layer can make all of them (fanout = the whole layer: 4 at 12
producers, 14 at 42). An ordering agent's job is to choose, for each order, which of the
possible makers gets it (whole or split). With two makers that is nearly a coin flip; with 14 it
is a real choice.

**The question** (the user's, after EXP-0049: "we have only 2 factories to choose from —
increase the number"): does routing matter more when each product has more possible makers? And
do the routers that look at the producers' spare capacity, which lost in EXP-0049, start to win
when they have more makers to choose from?

## Verdict

1. **Yes, routing matters much more with more makers, mostly because the hand-written rules
   break down.** At 42 producers, `round_robin` (every order whole, to a maker that rotates with
   the cycle) beats the best hand-written roster rule by +116 / +150 units per episode at fanout
   2 (calm / shocked), +116 / +220 at fanout 4, +335 / +421 at fanout 7 and +274 / +388 at fanout
   14 (each 4 of 4 worlds, t ≥ 3.6). Rules that split orders over the makers lose more as the makers
   multiply. The clean case is `random`: at 42 producers it splits an order over 1.07 makers on
   average at fanout 2 and 1.42 at fanout 14, its wasted units rise from 12–16 to 77–86 per
   episode, and sending the same random choice whole (`random_whole`) recovers +254 / +261
   units at fanout 14 (t ≥ 6.4). `round_robin` delivers 87–97% of what consumers order in every
   cell.
2. **The spare-capacity routers still do not beat blind choice, even with 4–14 makers.**
   `two_choice_spare` now really picks 2 of k, and still loses to blind `random_whole` in 14 of
   16 cells (by 5–171 units); it is ahead only at sparse_wide fanout 14, by 4–7 units (0.2–0.4%).
   `spare_weighted_whole` loses in 14 of 16. `spare_rotating_whole` is `round_robin` again.
3. **The trained portable orderer is the strongest router at wider fanout.** By cell mean it
   beats `random` and the best hand-written rule in all 16 cells; clearly (t ≥ 3.2) in 13 and 12
   of them. It clearly beats `round_robin` at 42 producers with fanout 4 (+68 / +25, t ≥ 5.2) and
   14 (+153 / +151, t ≥ 8.9). Its leads at 12 producers with fanout 4 and on dense-12 are in
   near-saturated cells (fill 0.97–0.99) that cannot separate routers well. It ties at fanout 7
   and is behind at fanout 2, clearly in 2 of the 4 fanout-2 cells (12 and 42 shocked, −26 and
   −22), the setting of EXP-0049.
4. **Splitting alone does not explain the losses; where an order's parts go matters too**
   (partly diagnosed). Lost units are almost all reservations that expire waiting for inputs. At fanout ≥ 4 the trained
   orderer splits orders into as many parts as `even_split` and `proportional` (1.6–1.8 producers
   per consumer order) yet wastes 10–15 delivered units per episode against their 126–271 at 42
   producers and fanout ≥ 4, and on the 42-producer sparse_wide worlds it loses the fewest units
   to expiry. Why its parts complete and
   theirs do not is **not diagnosed**; the deciding check is named below.

## How to read the numbers

- *Units*: consumer units delivered per episode (higher is better), as an equal-weight mean over
  the 4 worlds of a cell. *Fill*: delivered ÷ ordered consumer units (1.0 = every order filled).
- *Best det.*: the best of the five deterministic roster v2 rules on each episode (`even_split`,
  `proportional`, `obs_price`, `blend_0.5`, `shock_reactive`). *Gain*: units minus that.
- *Contrasts* are paired (same world and episode) and come with a world-clustered standard error
  (SE) and t = mean ÷ SE over the 4 worlds. With 4 worlds a difference needs t of about 3 to be
  clear; design D2 lists the detectable differences (for example 45 / 18 / 38 / 30 units for
  `round_robin` − `random` on 12-calm / 12-shocked / 42-calm / 42-shocked at fanout 2). Calm
  cells are the weak ones (4 worlds, 1 episode each).
- *Split degree*: producers per consumer order (1.0 = never split). *Wasted units*: units
  delivered as part of a split order that then failed. *TTL losses*: units of accepted orders
  that expired waiting for inputs. *Receiver change*: share of a stream's consecutive orders
  sent to a different producer (EXP-0049).
- Each cell has its own worlds. Comparisons **across** fanouts are not paired: the cells differ
  in their mix of strong and weak producers and in topology (design D2), so a trend over fanout
  is read as a pattern, not a measured effect of fanout alone. Comparisons **within** a cell are
  paired.

## Result

### Units per episode

| cell | best det. | even_split | proportional | random | random_whole | round_robin | two_choice | spare_weighted | spare_rotating | trained | portable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sparse_wide 12 f2 calm | 478 | 429 | 464 | 491 | 498 | 520 | 459 | 490 | 520 | 500 | 500 |
| sparse_wide 12 f2 shocked | 428 | 395 | 419 | 446 | 453 | 484 | 424 | 451 | 485 | 458 | 458 |
| sparse_wide 12 f4 calm | 544 | 350 | 532 | 532 | 530 | 534 | 522 | 531 | 538 | 556 | 556 |
| sparse_wide 12 f4 shocked | 522 | 339 | 505 | 524 | 526 | 550 | 518 | 524 | 552 | 554 | 554 |
| sparse_wide 42 f2 calm | 1662 | 1560 | 1518 | 1704 | 1713 | 1778 | 1542 | 1673 | 1764 | 1796 | 1799 |
| sparse_wide 42 f2 shocked | 1553 | 1481 | 1471 | 1635 | 1665 | 1703 | 1554 | 1643 | 1708 | 1681 | 1679 |
| sparse_wide 42 f4 calm | 1767 | 1331 | 1419 | 1857 | 1854 | 1882 | 1796 | 1837 | 1885 | 1950 | 1950 |
| sparse_wide 42 f4 shocked | 1606 | 1131 | 1368 | 1751 | 1764 | 1826 | 1708 | 1752 | 1824 | 1852 | 1853 |
| sparse_wide 42 f7 calm | 1495 | 254 | 1134 | 1761 | 1816 | 1830 | 1777 | 1791 | 1834 | 1834 | 1830 |
| sparse_wide 42 f7 shocked | 1393 | 320 | 910 | 1737 | 1784 | 1814 | 1746 | 1767 | 1814 | 1816 | 1814 |
| sparse_wide 42 f14 calm | 1412 | 918 | 1303 | 1418 | 1672 | 1686 | 1679 | 1657 | 1678 | 1838 | 1858 |
| sparse_wide 42 f14 shocked | 1335 | 834 | 1199 | 1428 | 1688 | 1723 | 1692 | 1677 | 1726 | 1874 | 1890 |
| dense 12 f4 calm | 454 | 338 | 416 | 470 | 476 | 482 | 458 | 469 | 481 | 494 | 494 |
| dense 12 f4 shocked | 461 | 335 | 432 | 480 | 482 | 500 | 471 | 479 | 499 | 514 | 514 |
| dense 42 f14 calm | 1616 | 1349 | 1528 | 1722 | 1810 | 1824 | 1805 | 1814 | 1816 | 1754 | 1700 |
| dense 42 f14 shocked | 1539 | 1072 | 1513 | 1709 | 1787 | 1790 | 1782 | 1785 | 1793 | 1782 | 1771 |

### Fill: how close to saturation

| cell | round_robin | trained | random | even_split |
|---|---|---|---|---|
| sparse_wide 12 f2 calm / shocked | 0.92 / 0.93 | 0.89 / 0.88 | 0.88 / 0.86 | 0.77 / 0.76 |
| sparse_wide 12 f4 calm / shocked | 0.95 / 0.97 | 0.99 / 0.98 | 0.95 / 0.93 | 0.60 / 0.59 |
| sparse_wide 42 f2 calm / shocked | 0.92 / 0.88 | 0.93 / 0.86 | 0.88 / 0.84 | 0.81 / 0.76 |
| sparse_wide 42 f4 calm / shocked | 0.95 / 0.97 | 0.98 / 0.98 | 0.94 / 0.93 | 0.67 / 0.60 |
| sparse_wide 42 f7 calm / shocked | 0.97 / 0.97 | 0.97 / 0.97 | 0.93 / 0.93 | 0.14 / 0.17 |
| sparse_wide 42 f14 calm / shocked | 0.87 / 0.87 | 0.95 / 0.94 | 0.73 / 0.72 | 0.47 / 0.42 |
| dense 12 calm / shocked | 0.97 / 0.96 | 0.99 / 0.98 | 0.95 / 0.92 | 0.69 / 0.65 |
| dense 42 calm / shocked | 0.97 / 0.96 | 0.93 / 0.96 | 0.91 / 0.92 | 0.72 / 0.58 |

The 12-producer cells at fanout 4 (sparse_wide and dense) are near saturation: the best routers
fill 97–99% of orders, so little room is left to separate them there (EXP-0044 saw the same for
dense-12). The 42-producer cells at fanout 4 and 7 are close to it for the whole-order and trained
routers (0.95–0.98).

### Question 1: does routing matter more with more makers?

`round_robin`'s paired gain over the best hand-written rule, and over `random` (units; SE; worlds
better of 4):

| cell | over best det. | over `random` |
|---|---|---|
| sparse_wide 12 f2 calm / shocked | +41.8 (25.2; 3) / +56.4 (16.4; 4) | +28.9 (9.5; 4) / +38.3 (7.4; 4) |
| sparse_wide 12 f4 calm / shocked | −9.0 (10.9; 2) / +28.9 (7.7; 4) | +2.2 (14.5; 2) / +26.2 (3.9; 4) |
| sparse_wide 42 f2 calm / shocked | +116.2 (29.0; 4) / +150.0 (23.2; 4) | +74.2 (19.2; 4) / +68.1 (7.9; 4) |
| sparse_wide 42 f4 calm / shocked | +115.5 (28.6; 4) / +220.4 (32.4; 4) | +25.1 (15.1; 3) / +75.0 (13.8; 4) |
| sparse_wide 42 f7 calm / shocked | +335.0 (91.9; 4) / +420.9 (92.5; 4) | +69.2 (14.6; 4) / +77.4 (15.9; 4) |
| sparse_wide 42 f14 calm / shocked | +273.5 (38.5; 4) / +388.4 (82.0; 4) | +267.6 (38.5; 4) / +295.8 (47.4; 4) |
| dense 12 calm / shocked | +27.0 (17.6; 2) / +39.2 (17.4; 4) | +11.5 (4.0; 4) / +20.1 (7.1; 4) |
| dense 42 calm / shocked | +207.2 (20.5; 4) / +250.9 (38.1; 4) | +101.1 (14.7; 4) / +80.7 (13.2; 4) |

At 42 producers the gap between a good and a bad router grows from about 150 units at fanout 2
to 300–400 at fanout 7–14. It grows because the splitting rules fall away, not because
`round_robin` gets better: the best hand-written rule delivers 1,662 / 1,553 units at fanout 2
and 1,412 / 1,335 at fanout 14, while `round_robin` stays at 1,686–1,882. At 12 producers the
cells near saturation (fanout 4) show small gains, and the calm one none (−9.0, t −0.8).

`even_split`'s collapse at fanout 7 (fill 0.14–0.17, against 0.60–0.67 at fanout 4 and 0.42–0.47
at 14; 4 of 4 worlds, no deaths) is not mainly a splitting effect: its wasted units there
(126–143) are no higher than at fanout 4, its loss is expiring reservations (1,250–1,650 units
per episode), and it sends every quantity-1 order to the lowest-index maker (EXP-0048's tie-break
defect, `q1_lowest_index_share` 1.0), which most likely concentrates orders on a few makers in
that topology. It is not used as evidence here.

`random`, the rule that set the bar on registry v2, also falls behind at fanout 14 (1,418 /
1,428). Uniform random scores split an order over more makers as the number of makers grows
(split degree at 42 producers 1.07 at fanout 2, 1.14–1.15 at 4, 1.22–1.23 at 7, 1.41–1.43 at
14), and its wasted
units rise with it (12–16 → 77–86 per episode at 42). Sending the same random choice whole
(`random_whole`) recovers +254 / +261 units at fanout 14 (t 10.9 / 6.5), +48–56 at fanout 7, and
+7–30 at fanout 2.

### Question 1b: do the spare-capacity routers pull ahead?

Paired difference to the blind counterpart (units; SE; worlds better of 4):

| cell | `two_choice_spare` − `random_whole` | `spare_weighted_whole` − `random_whole` | `spare_rotating_whole` − `round_robin` |
|---|---|---|---|
| sparse_wide 12 f2 calm / shocked | −38.9 (8.3; 0) / −28.6 (7.6; 0) | −8.3 (4.1; 1) / −2.1 (3.1; 2) | +0.0 (0.8) / +0.5 (0.9) |
| sparse_wide 12 f4 calm / shocked | −8.9 (5.6; 1) / −7.3 (3.1; 0) | +0.8 (2.0; 2) / −1.1 (4.5; 3) | +3.2 (1.3; 4) / +1.9 (0.7; 4) |
| sparse_wide 42 f2 calm / shocked | −171.2 (6.3; 0) / −110.9 (18.5; 0) | −40.0 (11.4; 0) / −21.8 (14.0; 1) | −14.8 (11.8) / +5.1 (8.3) |
| sparse_wide 42 f4 calm / shocked | −58.4 (11.3; 0) / −56.5 (7.8; 0) | −17.2 (5.7; 0) / −12.3 (3.0; 0) | +2.5 (5.0) / −2.8 (3.0) |
| sparse_wide 42 f7 calm / shocked | −39.4 (11.3; 0) / −38.5 (9.8; 0) | −25.9 (7.7; 0) / −17.2 (4.8; 0) | +4.2 (2.6) / −0.4 (4.3) |
| sparse_wide 42 f14 calm / shocked | +7.4 (8.4; 2) / +4.1 (1.1; 4) | −14.4 (6.0; 1) / −11.0 (5.1; 1) | −8.0 (5.6) / +3.0 (6.6) |
| dense 12 calm / shocked | −17.7 (8.0; 1) / −10.5 (2.1; 0) | −7.1 (2.3; 0) / −2.2 (4.0; 2) | −0.5 (0.3) / −1.1 (1.5) |
| dense 42 calm / shocked | −5.4 (1.8; 0) / −4.6 (2.4; 1) | +3.6 (1.9; 3) / −2.2 (2.7; 1) | −7.2 (3.1) / +3.1 (2.8) |

With more makers, the informed routers lose less, but they do not win: `two_choice_spare`'s
deficit at 42 producers shrinks from −111 to −171 at fanout 2 (an argmax there, EXP-0049) to −39
at fanout 7 and
turns into +4 to +7 at fanout 14 (shocked t 3.8, 0.2% of units). `spare_rotating_whole` differs
from `round_robin` by at most 15 units, and by at most 0.6% where |t| > 2 (12-f4, dense-42
calm). Spare capacity
still almost never limits an order: units lost because a producer had no room
(`lost_units_consumer_no_fit`) are 0–0.15 per episode for every subject in every cell except
`even_split` at fanout 14 shocked (2.4).

### Question 2: the trained orderer

Paired difference (units; SE; for `round_robin` also worlds better of 4). With 4 worlds a
difference is clear at t = mean ÷ SE ≥ 3.2:

| cell | trained − `round_robin` | trained − `random` | trained − best det. | `portable` − trained |
|---|---|---|---|---|
| sparse_wide 12 f2 calm / shocked | −20.8 (10.9; 0) / −25.8 (8.0; 0) | +8.1 (13.4) / +12.6 (2.9) | +21.0 (21.2) / +30.6 (8.8) | 0.0 / 0.0 |
| sparse_wide 12 f4 calm / shocked | +21.8 (12.7; 3) / +3.5 (4.3; 3) | +24.0 (10.1) / +29.7 (5.1) | +12.8 (3.8) / +32.4 (10.6) | 0.0 / +0.1 |
| sparse_wide 42 f2 calm / shocked | +18.2 (24.0; 3) / −21.9 (7.0; 0) | +92.5 (12.9) / +46.2 (2.8) | +134.5 (14.7) / +128.1 (18.6) | +2.8 / −2.0 |
| sparse_wide 42 f4 calm / shocked | +67.8 (9.8; 4) / +25.4 (4.8; 4) | +92.9 (10.9) / +100.4 (13.1) | +183.2 (25.3) / +245.8 (33.5) | +0.2 / +1.6 |
| sparse_wide 42 f7 calm / shocked | +4.0 (16.4; 3) / +1.4 (4.1; 3) | +73.2 (19.8) / +78.7 (16.4) | +339.0 (98.3) / +422.2 (92.9) | −4.5 / −1.8 |
| sparse_wide 42 f14 calm / shocked | +153.0 (12.0; 4) / +151.1 (16.9; 4) | +420.6 (44.3) / +446.9 (60.8) | +426.5 (49.3) / +539.5 (91.2) | +20.0 / +15.8 |
| dense 12 calm / shocked | +13.0 (3.5; 4) / +14.5 (5.5; 4) | +24.5 (6.5) / +34.6 (10.5) | +40.0 (18.2) / +53.8 (20.1) | −0.5 / −0.2 |
| dense 42 calm / shocked | −69.2 (41.5; 1) / −7.1 (15.9; 2) | +31.9 (30.9) / +73.5 (16.0) | +138.0 (36.2) / +243.8 (36.9) | −53.8 / −11.8 |

The trained orderer (EXP-0012's portable scorer with the `entity_planner` producer; distilled
from an input-aware target, log of `makeable_qty_now`, see the next section) beats
`random` and the best hand-written rule in every cell by mean, clearly in 13 and 12 of 16
(not clear: 12-f2 calm, 12-f4 calm, dense-42 calm against `random`; 12-f2 calm, 12-f4 shocked
and both dense-12 cells against the best rule). Against `round_robin` it is clearly ahead at 42
producers with fanout 4 and 14, ahead but not clearly (or in saturated cells) at 12-f4 and on
dense-12, even at fanout 7, clearly behind at fanout 2 in 12-f2 and 42-f2 shocked, and behind
but not clearly in 12-f2 calm (−21, t −1.9) and dense-42 calm (−69, t −1.7). The full portable stack
(its own trained producer) matches it except on dense-42 (−54 calm, t −1.8) and at sparse_wide
fanout 14 (+16 to +20).

### Why splitting hurts some routers and not others

Consumer-order routing at 42 producers, sparse_wide (ranges over the calm and shocked cells; full
per-cell values in `routing_metrics.csv`):

| fanout | subject | split degree | wasted units | TTL losses | receiver change |
|---|---|---|---|---|---|
| 2 | `even_split` | 1.59 | 111–146 | 254–295 | 0.17–0.48 |
| 2 | `proportional` | 1.48–1.50 | 120–131 | 281–317 | 0.21–0.46 |
| 2 | `round_robin` | 1.00 | 0 | 150–229 | 0.66–0.90 |
| 2 | trained | 1.30–1.32 | 32–41 | 105–207 | 0.63–0.69 |
| 4 | `even_split` | 1.72–1.80 | 156–178 | 475–559 | 0.20–0.54 |
| 4 | `proportional` | 1.71–1.79 | 169–171 | 339–379 | 0.49–0.73 |
| 4 | `round_robin` | 1.00 | 0 | 62–96 | 0.95–0.96 |
| 4 | trained | 1.66–1.77 | 10–11 | 22–28 | 0.91–0.92 |
| 14 | `even_split` | 1.77–1.79 | 264–271 | 727–839 | 0.22–0.55 |
| 14 | `proportional` | 1.77–1.79 | 160–202 | 467–571 | 0.88–0.94 |
| 14 | `random` | 1.41–1.43 | 77–86 | 448–472 | 0.97 |
| 14 | `round_robin` | 1.00 | 0 | 258–264 | 0.996–1.00 |
| 14 | trained | 1.69–1.70 | 15 | 90–98 | 0.88–0.90 |

The trained orderer splits about as much as the hand-written rules (1.66–1.77 producers per
consumer order at fanout ≥ 4) and changes receivers about as often as `round_robin`, yet its
split orders almost always complete (10–15 wasted units against 126–271) and it loses far fewer
units to expiring reservations than any other router (90–98 at fanout 14, against
`round_robin`'s 258–264). So splitting alone does not explain the losses: a router can split as much as the failing
rules and lose little, as long as its parts go to producers that finish them before the order
expires. Whether splitting still costs something even with well-placed parts is open (branch
(b) of the deciding check below). The trained
orderer was taught to avoid exactly that: its distillation target (`train/orderer_scorer.py`
`teacher_logits`) is log(units each maker can finish now), `makeable_qty_now` = min(units its
spare capacity allows, units its inputs in stock allow). `proportional`, by contrast, weighs
makers by capacity only (net of reservations), so despite the similar split it is not the rule
the scorer imitates; the teacher is an input-aware `proportional`. That is direct evidence for
input availability as the lever, but not a test of it: this run did not include the teacher
itself as a router. Rejections for lack of room stay near zero
for every router, so the failing producers had room; they lacked something else, most likely
inputs.

## Conclusions

- **Question 1, diagnosed in part.** Routing matters far more with more possible makers: the
  gap between `round_robin` and the best hand-written rule grows from about 150 to 300–400 units
  per episode at 42 producers (fanout 2 → 7–14). Cause, with evidence: routers that split orders lose more as the
  number of makers grows; the matched contrast is `random` against `random_whole` (same random
  choice, split or whole): +7 to +30 units for routing whole at fanout 2, +48–56 at 7, +254 /
  +261 at 14, with `random`'s split degree and wasted units rising with fanout. (`even_split`'s
  fanout-7 collapse has another likely cause, its lowest-index tie-break, and is not counted.) Lack of room is not the cause (no-fit losses ≈ 0). Read across fanouts as a
  pattern only (cells differ in producer mix); every within-cell contrast quoted has its SE.
- **The spare-capacity routers: diagnosed as not useful at any fanout tested.** Spare capacity
  does not limit orders here, so steering by it adds nothing; with more makers it costs less but
  never gains more than 0.4%.
- **Question 2, the trained orderer: measured, mechanism undiagnosed.** It is clearly the best
  router at 42 producers with sparse_wide fanout 4 and 14 (its leads at 12-f4 and dense-12 sit in
  near-saturated cells), ties `round_robin` at fanout 7, and loses to it at fanout 2 (clearly in
  the two shocked cells) and possibly on dense-42. It splits as much as the failing rules but its split orders
  complete. **Undiagnosed:** why. Its distillation target is input-aware (log of
  `makeable_qty_now`: capacity and inputs in stock), unlike `proportional` (capacity only), which
  points at input availability but does not test it. The deciding check has three arms on this
  grid: (a) the teacher as a router (split each order in proportion to `makeable_qty_now`); (b)
  its whole-order version (each order whole to a maker by `makeable_qty_now`); (c) a trace of
  every expired reservation by its producer's input stock at routing time. Teacher ≈ trained:
  input-aware weighting is the lever and splitting is not the cost. Whole input-aware router well
  above the teacher: splitting still costs even with input-aware weights. Teacher well below
  trained: the scorer learned something beyond its target.
- **This revises EXP-0049's M4 reading.** EXP-0049 concluded that M4 needs whole-order routing.
  At wider fanout the trained orderer wins while splitting, so "route whole" is not a hard
  requirement for a learned router: where the parts go matters at least as much. Whether routing
  whole would add to it is the deciding check's branch (b).
- **What changes for roster v3 (design D3, the user's decision).** `round_robin` remains the
  candidate and its value grows with fanout (+116 to +421 units over the best v2 rule at 42
  producers). The spare-capacity routers are not candidates. On wider-fanout worlds `random` is
  no longer near the top (fanout 14: 268–296 units below `round_robin`), so a bar without
  `round_robin` would be weak there.
- **What changes for the gate's world families.** Registry v2's fanout 2 is the setting where
  routing matters least among those tested and where `random` looks best. A registry v3 could add
  sparse_wide worlds with fanout 4–14 at 42 producers, where routers separate clearly; the
  12-producer fanout-4 and dense-12 worlds are near saturation (fill 0.97–0.99) and would add
  little. That is a registry decision (a re-screen, D3), for the user.
- **What changes for M4's starting point.** The portable orderer is already ahead of every
  hand-written rule on every tested world by mean (clearly in 12 of 16 cells), and clearly ahead
  of `round_robin` at 42 producers with fanout 4 and 14; the
  remaining gaps are fanout 2 (−21 to −26 units) and dense-42 (unclear). M4 should be trained and
  read on wider-fanout worlds too, not only on registry v2.
- **Power and limits** (design D2): 4 worlds per cell; calm cells rest on 4 world draws with one
  episode each and are underpowered for differences of 20–40 units (for example 12-f4 calm
  `round_robin` − `random` +2.2, SE 14.5). The fanout worlds are unscreened; fill is the only
  saturation read.

## Details

**Command** (repo root; checkpoints staged under `tokenomics/_pod_ckpts/`, deleted after):

```
python3 run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 \
  --vcpu 16 --mem 32 --disk 30 --pod-name exp0050-routing-vs-fanout --verify-source-commit \
  --train "-u -m tokenomics.eval.routing_audit --plan exp0050 \
    --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt \
    --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt \
    --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/exp0050" \
  --pull "/workspace/tokenomics/runs/exp0050:./experiments/EXP-0050-routing-vs-fanout"
```

`--pull` placed the run under `exp0050/`; its files were moved up unchanged and `episodes.csv`
into `artifacts/` (git-lfs). A resume in this directory would need it moved back.

**Grid.** 8 cell types (sparse_wide 12 × fanout 2 / 4; sparse_wide 42 × 2 / 4 / 7 / 14; dense 12
and 42) × calm / shocked (`shocks-i`), 4 worlds each, 1 calm and 2 shocked episodes per world;
world seeds from sha256 of (plan, family, size, fanout, index), so a cell's calm and shocked
worlds share their topology. Every world is pinned in `spec.json` with its measured declarers per
product and its config sha256. 31 runs per world-episode: the six roster v2 members (`random`
R = 5), `round_robin`, `random_whole` (R = 5), `proportional_whole`, `even_split_rotating`,
`two_choice_spare` (R = 5), `spare_weighted_whole` (R = 5), `spare_rotating_whole`,
`entity_planner/trained_orderer`, `portable`.

**Verification** (`/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/exp0050_check.py`, run from the repo root as
`python3 openspec/changes/add-m3-development-audit/probes/exp0050_check.py experiments/EXP-0050-routing-vs-fanout`;
its own arithmetic, not `routing_audit`'s):
- `contract_failures.csv` header-only; `spec.json` status `complete`, commit as above, plan
  `exp0050`.
- 2,976 rows, every planned (world, episode, subject, replicate) once, none extra; one demand and
  shock digest pair per world-episode (96).
- 64 of 64 worlds: measured declarers per product = the cell's fanout.
- Checkpoint pins producer `8b51b77cb5ac…`, orderer `d25388bfa33e…` = sha256 of the staged files
  (and EXP-0046's / EXP-0049's pins).
- `fanout_summary.csv` recomputes from `artifacts/episodes.csv` (units, every gain column,
  `beats_counterpart`): 240 rows, 0 mismatches.
- No row reproduces an earlier run (new worlds); the fanout-2 sparse_wide cells repeat
  EXP-0049's pattern (`round_robin` best whole-order router, `two_choice_spare` an argmax that
  loses, the trained orderer below `round_robin` on shocked worlds).

**Compute.** Runpod secure cloud, cpu5c, 16 vCPU / 32 GB, `runpod/base:0.7.0-ubuntu2004`,
python 3.12, torch 2.15.0+cpu. Pod `lgo5oai0z91pa7`, created and terminated by the launcher.
- Pod life: created after 17:30:17 UTC and terminated before 17:44:52 UTC (local clock): ≤ 14.6
  min ≈ **3.9 vCPU-h billed**.
- Execution (pod clock): `spec.json` `started` 17:31:26 to the last row at 17:44:32 ≈ 13.1 min ≈
  3.5 vCPU-h. `sizing.md` §EXP-0050 expected 3.6 vCPU-h and 13.5 min.
- M3 pod budget: ≈ 10.8 of 40 vCPU-h used (EXP-0048 4.4, EXP-0049 2.5, this 3.9).
  `runpodctl pod list` returned `[]` afterwards.

**Files.** `spec.json` (pins incl. plan, checkpoints and every world), `artifacts/episodes.csv`
(gate row format plus `trace_*` columns; git-lfs), `routing_metrics.csv` (per family × size ×
fanout × regime and subject: units, fill, death rate, every tracer scalar), `fanout_summary.csv`
(units, fill, gains over the best deterministic member, `random`, `round_robin`,
`random_whole`, and over the blind counterpart), `contract_failures.csv`.
