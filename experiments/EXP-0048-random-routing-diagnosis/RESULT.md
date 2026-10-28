# EXP-0048 — Random-routing diagnosis

**Status: complete.** 8,832 episode rows over 128 world-episodes, 21 routers (R = 5), horizon
100; registry v2 `7b09f536e988`, roster v2 `5050508ecb25`. Run commit
`291b2229ef92f0b5447ec48b64e11587e5c96686` (clean tree). Produced by
`python -m tokenomics.eval.routing_audit` (see `spec.json`). Change
`add-m3-development-audit`, task 3.1, design D2.

**Verdict: diagnosed, two factors.** `random` beats the hand-written routers for two reasons,
and both are things a deterministic rule can do. First, it varies which eligible producer gets
an order from one cycle to the next (explanations (b)/(c)). Second, it almost never splits an
order, and a split order completes only if every part does (explanation (f)). Randomness itself
is not needed. A deterministic router that does both, `round_robin` (whole order to a producer
that rotates with the cycle), recovers the whole edge in all three random-dominated cells: at
least 1.6× of it. Removing either factor loses. A fixed assignment of whole orders
(`random_fixed_episode`, fixed random weights; `proportional_whole`, which also sends every
orderer to the same producer) recovers nothing. Splitting the orders (`even_split_rotating`,
rotating) recovers less: 0.69–0.84. Varying the receiver at random also helps, but a rotation
helps somewhat more. The rotation factor includes a defect in
`even_split`: it always gives the odd unit to the lowest-index producer, and rotating that unit
lifts it from 0 to 2 of the 3 cells. Both findings name hand-written rules, so design D3
applies: whether to open roster v3 is the user's decision.

## Question

On the registry v2 development worlds, the `random` ordering baseline (uniform random logits,
mean of 5 seeded replicates) sets the hindsight baseline on most episodes. EXP-0047 measured
this: 95 of 128 episodes, and all 40 on 42-producer shocked worlds. Why does random routing beat
the hand-written routers (`even_split`, `proportional`, `blend_0.5`, `obs_price`,
`shock_reactive`)? Design D2 lists six competing explanations. Each one has a discriminating
check, built from diagnostic routers that change one thing at a time:

- **(a) Packing.** The rules pile sub-orders onto the same producers, and atomic auctions
  reject them. Check: `proportional_load_aware` (`proportional`, with throughput net of bids
  already placed this cycle) and the rejection reasons.
- **(b) Temporal variation, not randomness.** Spreading orders over time is what helps.
  Checks: `round_robin` (whole order to a producer that rotates with the cycle) against
  `proportional_whole` (whole order to the largest-share producer, no rotation);
  `random_fixed_episode` (random weights drawn once per episode, then held); the scale sweep
  `random_scaled_s` (uniform logits in [−s, s]; `random` is s = 10); `blend_noise_σ`
  (`blend_0.5` plus Gaussian noise).
- **(c) Rule defects.** Lowest-index tie-break, rounding of 1–3-unit orders, routes to producers
  that cannot make a unit, slot overflow. Checks: the task 2.1 trace, plus
  `even_split_rotating` (`even_split` with the leftover unit rotated by cycle instead of always
  going to the lowest-index producer).
- **(d) Investment-fund channel.** Random routing of producers' own purchases changes who gets
  investment. Checks: `random_consumer_only` and `random_producer_only` (random on one order
  class, `blend_0.5` on the other), plus the fund concentration.
- **(e) Scoring artifact.** `random` is a mean of 5 replicates and the rules are single runs.
  Checks: R = 1 against R = 5, and how often random's worst replicate still wins.
- **(f) Split penalty.** A split order completes only if all its sub-orders complete; the rules
  split 2–3-unit orders and `random` rarely does. Check: a 2×2 of whole/split ×
  deterministic/random at matched split degree. Whole orders: `proportional_whole`,
  `round_robin` / `random_whole`. Split orders: `even_split`, `even_split_rotating` /
  `even_split_random_leftover`.

Every router runs with the `entity_planner` producer, the producer of every roster v2 member.
The run covers all 128 world-episodes (24 worlds: sparse-wide family, 12 and 42 producers,
calm with 3 episodes per world, shocked with 10).

**Terms.** The *edge* of a world-episode is random's units minus the best deterministic roster
member's units on it. A router's *recovered fraction* in a size × regime cell is its mean gain
over that best member divided by random's mean edge, both as equal-weight means over worlds.
1.0 means it does exactly as well as `random`; above 1.0, better. Pre-registered rule (D2): a
router *recovers the edge* in a cell at a recovered fraction of 0.8 or more. It recovers it in
*most cells* at 2 of the 3 random-dominated cells: 12-shocked, 42-calm, 42-shocked. *Split
degree* is the mean number of producers an order is sent to (1.0 = never split).

## Result

### The edge per cell

| cell | random units | best deterministic member | edge | edge with R = 1 | episodes where random's worst replicate beats the best member |
|---|---|---|---|---|---|
| 12-calm | 513.4 | 529.1 | −15.7 | −16.4 | 3 of 24 |
| 12-shocked | 469.5 | 458.5 | 11.1 | 13.3 | 18 of 40 |
| 42-calm | 1723.3 | 1702.4 | 20.9 | 17.8 | 13 of 24 |
| 42-shocked | 1625.3 | 1548.3 | 77.0 | 75.6 | 38 of 40 |

Units are equal-weight world means of per-episode consumer units. "Best deterministic member" is
the per-episode best of the five deterministic roster members; on shocked cells that is mostly
`shock_reactive`. Random has no edge on 12-calm. There the recovered fraction is undefined and
the table below gives units.

### Recovered fraction per router

Recovered fraction, with the gain over the best deterministic member in units in brackets.
Fractions above 1 on a small edge (12-shocked: 11.1 units) look larger than they are.

| router | kind | 12-calm units | 12-shocked | 42-calm | 42-shocked | dominated cells ≥ 0.8 | consumer split degree (42-shocked) |
|---|---|---|---|---|---|---|---|
| `random` | stochastic | 513.4 | 1.00 (+11.1) | 1.00 (+20.9) | 1.00 (+77.0) | 3 of 3 | 1.07 |
| `blend_0.5` | deterministic | 522.8 | -1.32 (-14.6) | -3.41 (-71.2) | -0.52 (-40.2) | 0 of 3 | 1.57 |
| `even_split` | deterministic | 485.6 | -3.74 (-41.4) | -4.65 (-97.2) | -1.07 (-82.7) | 0 of 3 | 1.58 |
| `obs_price` | deterministic | 491.0 | -2.04 (-22.6) | -3.33 (-69.6) | -0.84 (-64.9) | 0 of 3 | 1.40 |
| `proportional` | deterministic | 513.0 | -1.39 (-15.4) | -6.34 (-132.5) | -1.43 (-110.3) | 0 of 3 | 1.45 |
| `shock_reactive` | deterministic | 519.0 | -0.43 (-4.7) | -1.05 (-21.9) | -0.08 (-5.9) | 0 of 3 | 1.47 |
| `round_robin` | deterministic | 539.5 | 4.72 (+52.3) | 2.51 (+52.4) | 1.62 (+124.8) | 3 of 3 | 1.00 |
| `even_split_rotating` | deterministic | 522.5 | 0.69 (+7.6) | 0.81 (+17.0) | 0.84 (+64.6) | 2 of 3 | 1.58 |
| `proportional_load_aware` | deterministic | 512.6 | -1.26 (-13.9) | -5.31 (-111.0) | -1.30 (-100.5) | 0 of 3 | 1.45 |
| `proportional_whole` | deterministic | 473.5 | 0.46 (+5.1) | -3.34 (-69.8) | -0.73 (-55.9) | 0 of 3 | 1.00 |
| `random_scaled_0.5` | stochastic | 519.2 | -0.08 (-0.9) | 0.14 (+2.9) | 0.24 (+18.1) | 0 of 3 | 1.58 |
| `random_scaled_1` | stochastic | 521.8 | -0.04 (-0.4) | 0.45 (+9.4) | 0.33 (+25.6) | 0 of 3 | 1.49 |
| `random_scaled_2` | stochastic | 518.2 | 0.17 (+1.9) | 0.66 (+13.7) | 0.53 (+40.9) | 0 of 3 | 1.30 |
| `random_scaled_5` | stochastic | 513.0 | 0.65 (+7.2) | 0.80 (+16.7) | 0.82 (+63.2) | 1 of 3 | 1.14 |
| `random_fixed_episode` | stochastic | 466.2 | -3.29 (-36.4) | -6.62 (-138.3) | -1.50 (-115.8) | 0 of 3 | 1.07 |
| `blend_noise_0.15` | stochastic | 529.1 | -0.19 (-2.1) | -2.02 (-42.3) | -0.38 (-29.4) | 0 of 3 | 1.57 |
| `blend_noise_0.6` | stochastic | 524.3 | 0.01 (+0.1) | -0.81 (-16.9) | 0.03 (+2.6) | 0 of 3 | 1.46 |
| `random_consumer_only` | stochastic | 530.0 | 0.65 (+7.2) | -0.50 (-10.5) | 0.37 (+28.4) | 0 of 3 | 1.07 |
| `random_producer_only` | stochastic | 512.5 | -0.05 (-0.6) | -0.21 (-4.3) | 0.48 (+36.7) | 0 of 3 | 1.57 |
| `random_whole` | stochastic | 514.9 | 1.79 (+19.8) | 1.15 (+24.0) | 1.28 (+98.3) | 3 of 3 | 1.00 |
| `even_split_random_leftover` | stochastic | 518.9 | 0.07 (+0.7) | 0.11 (+2.3) | 0.31 (+23.6) | 0 of 3 | 1.58 |

`random_scaled_5` on 42-calm is 0.799, just under the threshold.

### Reading each explanation

**(b) Temporal variation: confirmed, and it is the main lever.** `round_robin` sends every
order whole to a producer that rotates with the cycle. It recovers 4.72 / 2.51 / 1.62 of the edge
(+52.3 / +52.4 / +124.8 units). It beats `random`'s world mean on every world of the three
dominated cells (16 of 16) and on every one of their 104 episodes, and on 7 of the 8 12-calm
worlds, where random has no edge.

The pre-registered control, `proportional_whole`, recovers 0.46 / −3.34 / −0.73. It is not a
rotation-only control, though. It sends every orderer's order for a product to the same argmax
producer, so it also herds: routed-unit concentration per product and cycle
(`herfindahl_mean`) is 1.000 in every dominated cell, against `round_robin`'s 0.975 / 0.932 /
0.925. It also loses the most units to expired reservations (consumer `ttl` 92 / 296 / 397,
`round_robin` 47 / 179 / 229). `round_robin` against `proportional_whole` therefore mixes
rotation over time with cross-orderer herding.

The cleaner evidence for time variation is `random_fixed_episode`: random weights drawn once
per episode and then held. It is decorrelated across orderers (`herfindahl_mean` 0.906 / 0.914 /
0.912), mostly whole (split degree 1.07–1.08) and fixed over time. It recovers −3.29 / −6.62 /
−1.50 and is the worst router in three of the four cells (on 12-shocked only `even_split` is
worse). It loses 88.8 / 190.6 / 240.6 units to `round_robin` and 56.2 / 162.3 / 214.1 units to
`random_whole` (random, whole, redrawn every decision; 1.79 / 1.15 / 1.28). So the variation over
time matters, whether it comes from a rotation or from fresh random draws, and the rotation is
worth somewhat more (`round_robin` beats `random_whole` by 32.5 / 28.3 / 26.5 units). One
alternative is not separated: no balanced fixed assignment was run. `random_fixed_episode`'s
fixed assignment may simply be unbalanced, so "alternation over time" and "long-run balance of a
fixed assignment" remain mixed.

D2's reading applies: a deterministic rotating rule is a roster v3 candidate (D3), and M4 does
not need stochastic deployment for this. The scale sweep rises with s (42-shocked: 0.24, 0.33,
0.53, 0.82 for s = 0.5, 1, 2, 5) while the split degree falls (1.58 → 1.14). Part of its gain is
therefore the split control of (f), see below. Gaussian noise on `blend_0.5` (σ = 0.15, 0.6)
does not recover the edge anywhere.

**(c) Rule defect (lowest-index tie-break in `even_split`): confirmed.** `even_split` sends
every quantity-1 order to the lower-index eligible producer (`q1_lowest_index_share` 1.000 in
every cell) and the odd unit of every quantity-3 order too. The task 2.1 trace counted 646 of
646 tied consumer decisions on one 42-shocked episode. `even_split_rotating` changes only where
that leftover unit goes, by a cycle-indexed rotation; it has the same split degree (1.58). That
lifts `even_split` from 0 to 2 of the 3 dominated cells (0.69 / 0.81 / 0.84, against −3.74 /
−4.65 / −1.07). It also lifts its 12-calm units from 485.6 to 522.5. Placing the leftover unit at
random (`even_split_random_leftover`) gains most of that too: +42.1 / +99.5 / +106.3 units over
`even_split`, against +49.0 / +114.2 / +147.2 for the rotation. It stays below the 0.8 bar (0.07
/ 0.11 / 0.31) because the order is still split, see (f). As in (b), varying the receiver helps
whether the variation is random or rotating, and the rotation helps somewhat more. The other D2
rule defects do not occur at a measurable
rate. No router overflowed its slots (`overflow_orders` 0 everywhere). Routes to producers that
cannot make a unit are at most 0.5% of routes (`zero_throughput_route_share` ≤ 0.0047).
Rounding does move units in the share-based rules (task 2.1: about 0.3 units per order), but no
check here isolates it.

**(a) Packing: not supported.** `proportional_load_aware` does not recover (−1.26 / −5.31 /
−1.30) and is no better than `proportional`. Auction rejections (outage, no-fit, displaced) lose
only a few consumer units per episode for every router. On 42-shocked, `proportional` loses 2.5
units to outage and 0 to no-fit or displaced; `random` loses 1.1 to outage and none to the other
two. Lost consumer units are almost all reservations that expire waiting for inputs (`ttl`):
`proportional` 318.7, `even_split` 279.4, `random` 257.5, `round_robin` 228.7,
`even_split_rotating` 186.6 and `proportional_whole` 397.2 units per episode on 42-shocked.
Routed-unit concentration per product and cycle (`herfindahl_mean`) is higher for `random`
(0.90) than for the rules (0.77–0.84). Random does not win by spreading load within a cycle.

**(d) Investment-fund channel: not the lever.** Neither one-class router recovers the edge:
`random_consumer_only` 0.65 / −0.50 / 0.37, `random_producer_only` −0.05 / −0.21 / 0.48. Fund
concentration over the episode (`fund_share_herfindahl_episode`; reference 1/N = 0.083 at 12
producers and 0.024 at 42) barely differs between routers: 0.127–0.161 at 12 producers and
0.038–0.047 at 42. The random and rotating routers are at the low end of these ranges. The
investment rejection rate is low for every router except `obs_price` (0.38–0.45) and, at 12-calm,
`proportional_whole` (0.29).

**(e) Scoring artifact: no.** The edge with one replicate equals the edge with five within a
few units (12-shocked 13.3 vs 11.1, 42-calm 17.8 vs 20.9, 42-shocked 75.6 vs 77.0). On
42-shocked, random's worst replicate still beats the best deterministic member on 38 of 40
episodes. On 12-shocked and 42-calm it does on about half (18 of 40, 13 of 24): there the edge is
small and partly the mean of the replicates. That does not change the reading, because the
deterministic `round_robin` beats random there too.

**(f) Split penalty: confirmed, as a second factor that combines with rotation.** The
pre-registered single-factor pattern, "whole routers recover, split routers do not", does not
hold. Among whole routers, the rotating ones recover (`round_robin`, `random_whole`) and the
fixed one does not (`proportional_whole`). Among split routers, the rotating one recovers in 2
of 3 cells (`even_split_rotating`). Read as two crossed factors, though, at matched split degree
(whole orders 1.00, split orders 1.58 on 42-shocked), routing whole is worth most of the edge on
its own (matched contrasts below). The recovered fractions:

| | deterministic | random |
|---|---|---|
| **whole** (split degree 1.00) | `round_robin` 4.72 / 2.51 / 1.62; `proportional_whole` 0.46 / −3.34 / −0.73 | `random_whole` 1.79 / 1.15 / 1.28 |
| **split** (1.58–1.69) | `even_split` −3.74 / −4.65 / −1.07; `even_split_rotating` 0.69 / 0.81 / 0.84 | `even_split_random_leftover` 0.07 / 0.11 / 0.31 |

The split degrees match within each row: the gap between the deterministic and the random
routers is 0 for whole orders and at most 0.0004 for split orders in every cell (`edge.csv` level
`split_2x2`). Splitting does cost units. Split
orders that fail after part of them was delivered waste completed units: on 42-shocked,
consumer orders waste 148.1 units per episode for `even_split`, 135.4 for `proportional`, 104.3
for `even_split_rotating` and 15.8 for `random`. Whole routers waste none. `random` itself is mostly
whole (split degree 1.07), so its edge carries both factors.

### Each explanation's share of the edge (matched contrasts)

Design D2 asks for each explanation's share of the edge per cell. Each contrast below changes one
factor and holds the other fixed. Values are the difference in mean units (equal-weight world
means), with the fraction of random's edge in brackets. Columns: 12-shocked (edge 11.1),
42-calm (20.9), 42-shocked (77.0).

| factor changed | contrast | 12-shocked | 42-calm | 42-shocked |
|---|---|---|---|---|
| time variation vs fixed random weights (b) | `round_robin` − `random_fixed_episode` | +88.8 (8.02) | +190.6 (9.12) | +240.6 (3.12) |
| time variation vs fixed random weights (b) | `random_whole` − `random_fixed_episode` | +56.2 (5.07) | +162.3 (7.77) | +214.1 (2.78) |
| rotation vs fixed argmax, whole (b, mixed with herding) | `round_robin` − `proportional_whole` | +47.2 (4.27) | +122.1 (5.85) | +180.7 (2.35) |
| rotating leftover vs lowest-index leftover (c) | `even_split_rotating` − `even_split` | +49.0 (4.42) | +114.2 (5.47) | +147.2 (1.91) |
| random leftover vs lowest-index leftover (c) | `even_split_random_leftover` − `even_split` | +42.1 (3.80) | +99.5 (4.76) | +106.3 (1.38) |
| whole vs split, rotating (f) | `round_robin` − `even_split_rotating` | +44.7 (4.04) | +35.4 (1.69) | +60.2 (0.78) |
| whole vs split, random (f) | `random_whole` − `even_split_random_leftover` | +19.1 (1.72) | +21.8 (1.04) | +74.7 (0.97) |
| random vs rotation, whole | `random_whole` − `round_robin` | −32.5 (−2.94) | −28.3 (−1.36) | −26.5 (−0.34) |
| random vs rotation, split | `even_split_random_leftover` − `even_split_rotating` | −6.9 (−0.62) | −14.7 (−0.71) | −41.0 (−0.53) |

Varying the receiver over time, by rotation or by fresh random draws, is worth 1.4–9.1 times
the edge against a fixed assignment at either split degree. Routing whole is worth 0.8 of the
edge or more in 5 of 6 contrasts (0.78 in the sixth). Rotation beats random variation in every
cell, by 6.9–41.0 units. That is less than the time-variation contrasts in the same cell (at
least 42.1 units). It is not always less than routing whole: among the random routers, the
whole-vs-split contrast is smaller on 12-shocked (19.1 against 32.5) and 42-calm (21.8 against
28.3). The two factors combine: rotating whole orders is best, and
fixed or split variants lose. On 12-calm, where random has no edge, the same contrasts give +66.0 and
+36.9 units for rotation and +17.0 for routing whole when rotating.

### Task 2.1 trace, folded in

The single-episode trace (`openspec/changes/add-m3-development-audit/routing_trace.md`) showed
the same mechanics on one world per size. `even_split` breaks every tie toward the lowest index
(646 of 646 consumer ties on `dev-sparse_wide-42-shocked-00`). It found no slot overflow and
almost no zero-throughput routing. Splitting routers waste completed units on 42-shocked
(`even_split` 120, whole routers 0). Losses are mostly expired reservations (`ttl`). The trace
also found that the 1.1 tracer's fund concentration was broken: it squared shares of the whole
fund. Task 2.2 fixed that before this run, and the fund numbers above use the fixed episode
metric.

## Conclusions

- **Diagnosed cause, with evidence: two factors that combine.**
  - (b) Temporal variation of the receiving producer. A deterministic rotation (`round_robin`)
    recovers the edge in 3 of 3 dominated cells and beats `random` on all 104 of their episodes.
    Fixed random weights (`random_fixed_episode`), decorrelated but held for the episode, are
    among the worst routers in every cell and lose 89–241 units to `round_robin`. The
    pre-registered whole-order control (`proportional_whole`) recovers in none but also herds,
    so it mixes rotation with herding. Random variation helps too; rotation helps somewhat more.
    Not separated: alternation over time against the long-run balance of a fixed assignment
    (no balanced fixed assignment was run).
  - (c) A rule defect inside it: `even_split`'s lowest-index tie-break. Its rotated-leftover fix
    recovers in 2 of 3 cells. A random leftover gains most of the same units but stays under the
    bar because the order is still split.
  - (f) Whole-order routing under whole-order completion. At the same kind of variation, routing
    whole instead of split is worth 0.78–4.04 of the edge (matched contrasts). Split routers
    waste completed units that whole routers never do.

  Rotating whole orders is best; rotation with splitting, or whole orders without rotation, lose.
- **Ruled out by their checks:** (a) packing (load-aware netting does not help; auction
  rejections are negligible), (d) the fund channel (neither one-class router recovers; fund
  concentration barely differs), (e) the scoring artifact (R = 1 gives the same edge).
- **Not explained by this run:** how rotation over time avoids the expiring reservations. Lost
  consumer units are almost all `ttl`, and rotation lowers them (42-shocked: `round_robin`
  228.7, `proportional_whole` 397.2). Which producers' inputs starve, and why, needs a
  producer-level trace of the `ttl` failures. That would explain the mechanism; it would not
  change the diagnosis.
- **What changes (design D2/D3).** A hand-written rule is defective (`even_split`'s
  tie-break), and a deterministic rule that rotates and routes whole beats every roster
  member, `random` included. D3 therefore applies. The roster v3 candidates are a rotating rule
  and a whole-order rule. `round_robin` is both; `even_split_rotating` is the rotating fix of the
  defective rule. They would enter `ORDERING_BASELINES` under new names, through a roster v3
  draft and a baseline-only re-screen (≈ 4 vCPU-h). That raises the hindsight baseline, so it is
  the user's decision. M3 stops here until the user decides. M4 does not need stochastic
  deployment to get this edge: a deterministic rule gets it. M4's residual context must be able
  to route an order whole and to rotate the receiving producer over time.
- **For PoC-A (EXP-0050, D5; renumbered from EXP-0049 when the EXP-0049 follow-up was inserted).** On v2 the hindsight baseline is mostly `random`, which a rotating
  rule beats. Read the teacher against v2 and, if the user opens it, against v3.

## Details

**Command** (repo root):

```
python3 run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 \
  --vcpu 16 --mem 32 --disk 30 --pod-name exp0048-random-routing-diagnosis --verify-source-commit \
  --train "-u -m tokenomics.eval.routing_audit --workers 16 --source-commit {SOURCE_COMMIT} \
    --out-dir /workspace/tokenomics/runs/exp0048" \
  --pull "/workspace/tokenomics/runs/exp0048:./experiments/EXP-0048-random-routing-diagnosis"
```

`--pull` placed the run under `exp0048/`; its files were moved up into this directory unchanged,
and `episodes.csv` then into `artifacts/` (git-lfs, `.gitattributes` `experiments/**/artifacts/**`)
because of its size. The driver writes it at the top level, so a resume in this directory would
need it moved back.

**Verification** (`openspec/changes/add-m3-development-audit/probes/exp0048_check.py`, run from
`tokenomics/` as `../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/exp0048_check.py
../experiments/EXP-0048-random-routing-diagnosis ../experiments/EXP-0047-registry-v2-calibration`).
It does its own arithmetic and does not call `routing_audit`'s analysis code.
- `contract_failures.csv` is header-only; `spec.json` status `complete`, commit as above.
- `artifacts/episodes.csv` holds 8,832 rows, every planned (world, episode, subject, replicate) row
  exactly once, none extra. Every one of the 128 world-episodes has a single demand and shock
  digest pair across all 69 runs.
- Tracer neutrality at scale: all 1,280 rows of the six roster v2 members (random's 5
  replicates included) equal EXP-0047's units for the same world-episode, member and replicate.
- `edge.csv` recomputes from `artifacts/episodes.csv` with equal-weight world means and the
  0.8 / 2-of-3 rule: 84 cell rows and 21 rule rows, 0 mismatches.

**Compute.** Runpod secure cloud, cpu5c, 16 vCPU / 32 GB, `runpod/base:0.7.0-ubuntu2004`,
python 3.12, torch 2.15.0+cpu. Pod `9ergohtro859lj`, created and terminated by the launcher.
- Pod life: created after 12:40:35 UTC and terminated before 12:57:15 UTC (local clock, stamps
  taken around the launcher): ≤ 16.7 min ≈ **4.4 vCPU-h billed**.
- Execution (pod clock): `spec.json` `started` 12:41:40 to the last row at 12:57:02 ≈ 15.4 min
  ≈ 4.1 vCPU-h. `sizing.md` §EXP-0048 expected 4.2 vCPU-h and 15.8 min.
- M3 pod budget: 4.4 of 40 vCPU-h used. `runpodctl pod list` returned `[]` afterwards.

**Files.** `spec.json` (pins, provenance, the pre-registered rule and the driver's choices),
`artifacts/episodes.csv` (gate row format plus `trace_*` columns, 33.8 MB; git-lfs), `routing_metrics.csv` (per
size × regime and router: units, death rate and every tracer scalar, equal-weight world means),
`edge.csv` (levels `episode`, `cell`, `rule`, `split_2x2`), `contract_failures.csv`.
