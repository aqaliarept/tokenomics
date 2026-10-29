# EXP-0052 — Headroom atlas, confirm

**Status: complete, valid.** `python -m tokenomics.eval.atlas --mode confirm --screen-dir <EXP-0051>`,
run commit `9d442202d2184837d3f4c5408d194a6102b37042` (task 5.2 of `add-m3-development-audit`,
design D12 + task 4.7, ADR-0057); screen = EXP-0051, run commit
`92f317725ca4fe02bd703b950008ee2a3cb64335`. 2,775 episode rows over 185 world-episodes (the 96
selected worlds: 64 at 12 producers, 32 at 42; 2 fresh episodes each, 1 on the 7 worlds with
cyclic demand and no shocks; 15 rows per world-episode), horizon 100. 0 contract failures. Frozen
**atlas v1**: `atlas_v1.json`, content hash `f241464c4bc4…`. Pod shape and spend in
[Details](#details): fresh **8 vCPU** secure pod (the user's fallback after 16 vCPU pods failed),
≈ 3.8 vCPU-h billed.

## In plain English

> **Revised by task 5.3** (addendum below Conclusions): a robust (median-based) noise floor and
> a dead-arm collapse rule, applied offline to these rows, give hard 11 / open 49 / collapse 4 at
> 12 producers and hard 13 / open 17 / collapse 2 at 42 (atlas v2, `reclassified/`). The counts
> below are the pre-registered rule's.

**The question.** The screen (EXP-0051) played 640 worlds once each and labelled most of them
*hard* (a cheap better policy beats the best simple rule by more than chance) or *open* (the
world's ceiling leaves room nobody cheap reached), but its "more than chance" test borrowed a
measure of chance variation from easy worlds and was far too lenient (DIAGNOSTICS 21). This
confirm stage replays the 96 worlds the screen selected (72 labelled hard, 24 open) on **fresh
seeds**, two episodes each and two replicates of every random policy, and measures chance
variation **from its own runs**. It keeps the screen's choice of which simple rule is the *bar*
(the best blind or hand-written orderer) and which cheap policy is the *reach arm* (the best of
`makeable_split`, `makeable_weighted_whole` and the trained orderer), so a world cannot look hard
just because one lucky policy was picked after the fact.

**Terms.** *Gain*: reach-arm units minus bar units, averaged over paired episodes. *Noise floor*:
how large a gain chance alone could produce on this kind of world: 3 standard errors (or 1% of
the bar, if larger), where the standard error comes from two spreads pooled over the confirm's
own worlds of the same size and choice: σ_ep (how much the gain moves from one episode to the
next) and σ_rep (how much a random policy's units move between two replicate runs). *Ceiling C*:
the most consumer units the world can deliver by the input-output bound of task 4.7 (B_lp, and
B_lp_base without upgrades), or the best played arm if higher. *Room*: (C − bar) / C. Classes are
as pre-registered: collapse, saturated (bar fills ≥ 95% of orders or ≥ 97% of C), hard (gain >
floor and positive in every episode and replicate), open (room ≥ 0.10), flat.

**What we found.** Most of the screen's hard labels do not survive. At 12 producers **4 of 64**
selected worlds are confirmed hard (screen: 48), the other 60 are open; at 42 producers **10 of
32** are hard (screen: 24), 22 open. No selected world is saturated, flat or collapsed. The
reach arm still beats the bar on most worlds (44 of 64 at 12, 24 of 32 at 42), by a median
5.7% of the bar at 12 (screen: 9.4%; the screen's gains shrank on fresh seeds on 45 of 64 worlds,
the winner's curse the confirm exists to catch) and 11.1% at 42 (screen 11.6%; held up). But
the confirm's own noise floor is large: a median of **397 units at 12 producers** (33% of a
typical bar) and **577 at 42** (15%), against the screen's 14 and 76.

**Why the floor is so large.** Both spreads measured on atlas worlds are big: σ_ep = 164 units at
12 and 156 at 42 (EXP-0050's, which the screen borrowed: 6–14 and 6–37), σ_rep = 90–131 at 12 and
315 at 42 (EXP-0050: 2.6 and 10–21; the screen's own `random_whole` estimate was 46–77 and
156–343). Each stratum had fewer than 30 degrees of freedom, so as pre-registered every stratum
takes its size's pool. This pooled estimator is the one design D12 prescribes, so 4 hard at 12
and 10 at 42 is the pre-registered result. But both spreads are concentrated in a few worlds. The
episode spread comes mostly from worlds whose gain swings by hundreds of units between the two
episodes: the 5 largest give 89% of its sum of squares at 12 and 72% at 42 (e.g. 12-0033, dense,
seasonal, shocked: the bar arm `random_whole` delivers ≈ 2,300 units in one episode and ≈ 620 in
the other); these swings are not deaths (pooling only worlds where neither fixed arm ever died
gives the same σ_ep). The replicate spread is concentrated too: 5 worlds give 69% of its sum of
squares at 12 and 87% at 42, and at 42 its largest term is a death: on 42-0045 one of the two
`random` replicates (the bar arm) dies at cycle 23 or 17 with 163 or 39 units while the other
delivers ≈ 1,900. At 42 producers the floor is mostly this replicate term (σ_rep²/2 ≈ 49,600
against σ_ep² ≈ 24,200 units²). The spread is also pooled in absolute units over worlds whose
bars range from a few hundred to 16,400 units, so high-volume worlds weigh most. The typical world
is much calmer: the median episode-to-episode change of the gain is 52 units at 12 and 134 at 42
(3.7% and 2.6% of the bar). To see how much the class counts hang on this, the check script
recounts hard worlds with other spread estimates (none pre-registered): pooled without the 5
largest σ_ep worlds, without the 5 largest σ_rep worlds, or without both; pooled without worlds
where a fixed arm died; relative to the bar; median-based (robust to the few wild worlds); and
with the bar re-chosen on the confirm's own runs. At 3 standard errors the hard count ranges
**2–12 of 64 at 12** and **5–17 of 32 at 42** (pre-registered 4 and 10; removing both sets of 5
worlds from the pools gives 12 and 17); at 2 standard errors 3–20 and 7–20. 33 worlds at 12 and
22 at 42 have a gain that is positive in every episode and replicate pairing, a weak test with only
2–4 pairings per world. So: measured headroom exists, it is thin at 12 producers under the
pre-registered floor (and up to about a fifth of the selected worlds with a robust one), and it is
clear at 42 producers with wide choice (5–17 of the 32 selected worlds by estimator, 10
pre-registered).

**Where.** At 42 producers the hard worlds are almost all **wide-choice** worlds: 9 of the 14
selected worlds with 14 makers per product are hard, plus 1 dense world; none of the 15 with 2, 4
or 7 makers. Their gains are large (median +16% of the bar; 352–1,758 units), and the winning
cheap policy is the input-aware one: `makeable_weighted_whole` (6), `makeable_split` (3), the
trained orderer (1). Demand pattern and shocks do not separate them (burst 3, iid 4, cyclic,
hit, switch 1 each; 4 shocked, 6 calm). At 12 producers the 4 hard worlds are 2 cyclic worlds
without shocks, where every episode replays exactly and any gain is real (12-0464: `makeable_split`
+163 over `round_robin`, 9%; 12-0348: the trained orderer +79 over `round_robin`, 9%); 12-0273
(sparse f4, drift, shocked), which clears its floor of 399 by 31 units; and 12-0435, which is an
artifact: the screen's bar arm `random_whole` dies at cycle 4 in all four confirm runs, so the bar
reads 0, while other blind arms (`random`, `proportional`, `shock_reactive`, `round_robin`) mostly
survive with 470–790 units; against the confirm's own best blind arm (`shock_reactive`, 656) the
gain is +218, below the floor, so the world is not hard. 12-0273 also drops (free bar
`random_whole`, gain +340 < 399); with the bar re-chosen only the two exact cyclic worlds stay. The 64 at 12 by choice: dense 1 hard / 27
open, f2 0 / 16, f4 3 / 17.

**Open worlds.** 60 at 12 and 22 at 42. The bar fills a median 47% / 45% of consumer orders and
delivers about half of the ceiling (median room 0.48 / 0.53, never below 0.14). The ceiling is
the task-4.7 input-output bound: B_lp ≈ B_lp_base (median ratio 1.00) ≈ scheduled demand. As
EXP-0051's addendum says, this bound ignores input lead time, the investment fund and inventory
limits, so this room is room against those simplifications; how much of it is reachable is
unknown until a teacher (PoC-A) or a tighter bound tries. Arms above B_lp_base (which is not a
guaranteed bound): 16 arm × world-episode cases, on two 12-producer f4 worlds (12-0435 both
episodes, 12-0350 one); B_lp itself is never exceeded.

**The deciding check (EXP-0050's, pre-registered).** EXP-0050 asked *why* the trained orderer's
split orders complete when other splitting rules' do not; its distillation target is input-aware
(how many units each producer can finish now, given capacity and inputs in stock). On the 14 hard
worlds with at least 4 makers per product (4 at 12, 10 at 42) three readings were pre-registered,
each needing 2/3 of the worlds:
- `makeable_split` (the teacher target as a router) within the floor of the trained orderer →
  input-aware weighting is the lever, splitting is not the cost: **4 of 14** — not met;
- `makeable_weighted_whole` above `makeable_split` by more than the floor → splitting still costs
  even with input-aware weights: **8 of 14** (57%) — not met (it is 7 of 10 at 42 producers, a
  split by size that was not pre-registered);
- `makeable_split` below the trained orderer by more than the floor → the scorer learned more
  than its target: **2 of 14** — not met.

No branch holds. The nearest is "splitting still costs": on 8 worlds, routing each order whole
by the same input-aware weights beats splitting it, by up to 8,700 units at 42 producers. And
`makeable_split` beats the trained orderer by more than the floor on 7 of the 10 at 42 (by
409–3,286 units), the opposite direction from the third branch. With floors of 330–580 units,
several "within the floor" calls are low-powered. Without 12-0435 (the dead-bar artifact below,
which counts as "within the floor" for the first branch) the counts are 3 / 8 / 2 of 13: still no
branch.

**Expired reservations by input stock.** The tracer splits the units lost to an expired
reservation (TTL) by whether the receiving producer's input stock could cover the order at routing
time. The driver's `deciding_check.csv` sums consumer orders and producers' own investment orders
(D12 names the field per arm without saying which order side); split by side (from
`episodes.csv`, all 96 worlds, sums over replicates):
- **consumer orders** (the deciding check's question): 88–91% expired with inputs short for the
  blind rules that set the bar (`round_robin` 0.886, `random_whole` 0.883, `random` 0.908; the
  screen's addendum found 87% for the same two bar arms, also consumer-side), 87–89% for the other
  roster rules, against **0.705 `makeable_split`, 0.728 trained orderer, 0.732 `portable`, 0.772
  `makeable_weighted_whole`**; on the 14 hard worlds of the deciding check 0.76–0.81 for the three
  split-capable input-aware arms and 0.90 for `makeable_weighted_whole`, against 0.95–0.98 for the
  rest;
- **investment orders**: 0.87–0.91 for the blind bar rules, 0.58–0.68 for the other roster rules,
  0.365 `makeable_split`, 0.50–0.52 trained / `portable`, 0.815 `makeable_weighted_whole`;
- both together (the driver's column): 0.89 for the bar rules, 0.59 `makeable_split`, 0.65–0.66
  trained / `portable`, 0.79 `makeable_weighted_whole`.

So on consumer orders input-aware routing lowers the inputs-short share by about 16–18 points
(0.89 → 0.71–0.73), a clear but partial removal of the input-shortage cause of expiries; a larger
part of the combined drop is on investment orders. The consumer expiries that remain with inputs in
stock grow (337k units for `makeable_split` against 128k for `round_robin`, both R = 1), which fits
orders concentrating on the producers that can make them now.

**`portable`** (the trained producer plus the trained orderer, reported only) is below the bar
on most worlds: median −10% at 12 (above the bar on 16 of the 63 worlds with a non-zero bar) and −31% at 42 (7 of 32); against
the reach arm −15% / −40%. The trained orderer with `entity_planner` as producer does better
than with the trained producer on these worlds.

**Evidence for the user's gate decisions (not decisions).** Hard regions confirmed on fresh seeds:
42 producers with wide choice (f14: 9 of 14 selected; dense: 1 of 3), and at 12 producers only
isolated worlds (2 exact cyclic, 1 marginal, 1 artifact). Saturated regions: none among the
selected worlds; the screen located saturation only in the low-tightness, small-order corner
(registry v2 / EXP-0050 territory) and a few over-demanded worlds, and none of those were
selected. Open (room against the input-output ceiling, unreached): the majority at both sizes.
Whether any of this enters a gate registry (a v3 re-screen) or roster v3 is the user's call; the
atlas changes no gate file.

**What M4 should train and read on.**
- Train where headroom is measured: the 42-producer hard worlds, 9 f14 + 1 dense (train split:
  0031, 0054, 0073, 0076, 0126 and dense 0119; validation 0066; held-out 0008, 0125;
  out-of-region 0098). The lever
  already visible is input-aware routing that can send an order whole (`makeable_weighted_whole`
  wins 6 of 10; it beats `makeable_split` on 7 of 10).
- Read with more episodes. Two episodes per world give a floor of 15–33% of the bar; to resolve
  a 5% gain at 3 SE needs n ≈ (3σ / 0.05·bar)² (episode term only): with the pooled σ_ep ≈ 160 that is about 4 episodes
  on a 5,000-unit 42-producer world but about 40 on a 1,500-unit 12-producer world (about 5 with
  the median-based σ_ep ≈ 55). Report floors pooled robustly or per world class, and flag worlds
  whose fixed bar arm dies (12-0435; DIAGNOSTICS 20).
- PoC-A's pre-registered sample (design D5: atlas v1 train split, 12 producers, 6 hard + 6 open by
  confirm gain / room, the other class fills) has only **2 hard train worlds** at 12 producers:
  12-0435 (gain +874, the top one, the dead-bar artifact) and 12-0348 (cyclic, +79); 12-0273 is
  held-out and 12-0464 validation. So its hard class would be 2 worlds, one an artifact, and 10
  open. D5 re-chooses the bar as the best blind arm on the same world-episode, so 12-0435 would not
  read as a 5-fold gain there, but it would not be a measured-hard world either. Evidence for
  re-planning tasks 6.2 / 7.1 (not a decision).
- Open worlds (train split) are the teacher's sample for PoC-A (tasks 6.x); their room is
  unproven until a teacher or a lead-time-aware bound shows it.
- Do not use `portable` as the student baseline on these worlds without retraining: its producer
  loses 10–31% against `entity_planner`-based arms.

## Verification

Independent check (stdlib + `world_space.episode_seeds`; D12 + task 4.7 rules re-implemented):
```
.venv/bin/python openspec/changes/add-m3-development-audit/probes/exp0052_check.py \
  experiments/EXP-0052-headroom-atlas-confirm --screen experiments/EXP-0051-headroom-atlas-screen \
  --ckpt producer=experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt \
  --ckpt orderer=experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt
```
→ **ALL CHECKS PASS**:

| check | result |
|---|---|
| `contract_failures.csv` | header-only (0 rows); `spec.json` kind confirm, status complete |
| planned (world-episode, arm, replicate) rows | 2,775 of 2,775 (185 world-episodes × 15), no duplicates, no unplanned subject; all 96 selected worlds, no others |
| digest pairs | exactly one (demand, shock) digest pair per world-episode |
| episode seeds | = `episode_seeds(world, n, "confirm")` on all 96 worlds (n = 2; 1 for the 7 cyclic worlds without shocks); 0 worlds share a seed with the screen |
| bounds | 0 rows above B2 + 0.01; 0 above B_lp + 0.01; exactly one LP row per world-episode (185) |
| fixed argmax | bar / reach arms = EXP-0051 `selection.csv` (sha256 `84505523d573…` = the pin) |
| pooled spreads | recomputed (per size × choice, merged within size below 30 df), equal to `spec.json` |
| `worlds.csv` | bar, reach, gain, fill, B2, B_lp, B_lp_base, ceiling, room, SE, floor, sign, class, screen class, split, arms above B_lp_base: equal for all 96 |
| `deciding_check.csv` | all 14 world rows, 3 branch rows, 24 TTL rows equal |
| `atlas_v1.json` | content hash = sha256 of its canonical body; classes, screen classes, gain, room, split equal; screen commit `92f3177`, confirm commit `9d44220` |
| split | = EXP-0051's (train 60, validation 16, held-out 16, held-out-region 4) |
| checkpoints | producer `8b51b77cb5ac…`, orderer `d25388bfa33e…`: staged files = `spec.json` pins |

## Tables

**Classes, screen → confirm.**

| size | selected | screen | confirm | transitions |
|---|---|---|---|---|
| 12 | 64 | hard 48, open 16 | hard 4, open 60 | hard→hard 4, hard→open 44, open→open 16 |
| 42 | 32 | hard 24, open 8 | hard 10, open 22 | hard→hard 10, hard→open 14, open→open 8 |

**Classes by region (confirm).**

| size | axis | hard / open |
|---|---|---|
| 12 | choice | dense 1 / 27; f2 0 / 16; f4 3 / 17 |
| 12 | pattern | cyclic 2 / 10; drift 1 / 4; iid 1 / 4; burst 0 / 19; hit 0 / 7; seasonal 0 / 9; switch 0 / 7 |
| 12 | shocks | no 2 / 24; yes 2 / 36 |
| 42 | choice | f14 9 / 5; dense 1 / 2; f2 0 / 5; f4 0 / 6; f7 0 / 4 |
| 42 | pattern | iid 4 / 3; burst 3 / 5; cyclic 1 / 2; hit 1 / 4; switch 1 / 3; drift 0 / 2; seasonal 0 / 3 |
| 42 | shocks | no 6 / 9; yes 4 / 13 |

**Spreads (pooled, confirm's own, vs the screen's provisional EXP-0050 ones).**

| stratum | σ_ep (df) | screen σ_ep | σ_rep (df) | screen σ_rep |
|---|---|---|---|---|
| 12 / dense, f2, f4 | 163.9 (58, size pool) | 5.9 / 14.2 / 6.3 | 131.2 (65) / 90.1 (46) / 92.2 (43) | 2.6 |
| 42 / all five | 155.7 (31, size pool) | 5.8–36.6 | 315.0 (73, size pool) | 9.7–21.3 |

Median floor: 397 at 12 (screen 14), 577 at 42 (screen 76).

**Sensitivity of the hard count (not pre-registered; check script "sensitivity").**

| spread estimate | σ_ep / σ_rep at 12 | σ_ep / σ_rep at 42 | hard at 12, 3 SE (2 SE) | hard at 42, 3 SE (2 SE) |
|---|---|---|---|---|
| pre-registered pool | 164 / 90–131 | 156 / 315 | 4 (5) | 10 (14) |
| pool without worlds where a fixed arm died | 168 / 109 | 159 / 234 | 4 (5) | 12 (14) |
| relative to the bar | 11.5% / 14.6% | 4.3% / 22.9% | 4 (7) | 5 (7) |
| median-based | 55 / 52 | 141 / 122 | 12 (19) | 13 (18) |
| bar re-chosen on confirm runs | pre-registered pool | same | 2 (3) | 11 (13) |
| pool without the 5 largest σ_ep worlds per size | 57 / 86–94 | 91 / 333 | 6 (12) | 11 (15) |
| pool without the 5 largest σ_rep worlds per size | 157 / 44–72 | 156 / 127 | 5 (7) | 13 (17) |
| pool without both sets | 52 / 44–72 | 84 / 121 | 12 (20) | 17 (20) |

Share of the sum of squares from the 5 largest worlds: σ_ep 89% at 12, 72% at 42 (12-0033, 0408, 0352, 0449, 0121; 42-0098, 0066, 0076, 0125, 0119); σ_rep 69% / 87% (12-0408, 0278, 0273, 0423, 0392; 42-0045, 0126, 0119, 0013, 0102; per world, summed over its replicate pairs). Excluded worlds leave only the pools; they are still classified. Gain positive
in every episode and replicate: 33 of 64 at 12, 22 of 32 at 42. Worlds where the fixed bar or
reach arm died at least once: 4 at 12, 2 at 42.

**Deciding check (hard worlds with E ≥ 4; floors at 3 SE).**

| world | split − trained (floor) | whole − split (floor) |
|---|---|---|
| 12-0273 | −234 (348) | **+439** (374) |
| 12-0348 | **−17** (9) | −31 (278) |
| 12-0435 | +13 (348) | −678 (374) |
| 12-0464 | **+190** (17) | −69 (196) |
| 42-0008 | +160 (330) | −140 (577) |
| 42-0031 | **+409** (330) | **+2,928** (577) |
| 42-0054 | **−1,455** (330) | **+1,390** (577) |
| 42-0066 | **+3,286** (330) | −281 (577) |
| 42-0073 | +180 (330) | −365 (577) |
| 42-0076 | **+593** (330) | **+2,317** (577) |
| 42-0098 | **+900** (330) | **+5,498** (577) |
| 42-0119 | **+1,128** (330) | **+8,565** (577) |
| 42-0125 | **+2,044** (330) | **+8,717** (577) |
| 42-0126 | **+1,041** (330) | **+1,985** (577) |

Branches: input-aware weighting (split within floor of trained) 4 / 14; splitting costs (whole
above split) 8 / 14; scorer beyond target (split below trained) 2 / 14; none reaches 2/3.

**Expired-reservation units by input stock at routing time, consumer + investment orders
combined as in `deciding_check.csv` (all 96 worlds; sums over replicates).** Consumer-only and
investment-only shares in the last columns (recomputed from `episodes.csv`'s
`trace_ttl_lost_units_{consumer,producer}_inputs_{short,ok}`).

| arm | inputs short | inputs ok | share short | share short, 14 hard E ≥ 4 | consumer only | investment only | consumer only, 14 hard |
|---|---|---|---|---|---|---|---|
| `round_robin` | 1,593,356 | 191,244 | 0.893 | 0.964 | 0.886 | 0.904 | 0.975 |
| `random_whole` (R 2) | 3,178,719 | 378,021 | 0.894 | 0.962 | 0.883 | 0.912 | 0.970 |
| `random` (R 2) | 3,643,454 | 427,593 | 0.895 | 0.935 | 0.908 | 0.874 | 0.977 |
| `proportional` | 1,651,594 | 388,061 | 0.810 | 0.859 | 0.880 | 0.671 | 0.967 |
| `even_split` / `obs_price` / `blend_0.5` / `shock_reactive` | 1.49–1.65 M | 0.37–0.40 M | 0.795–0.806 | 0.847–0.863 | 0.871–0.894 | 0.580–0.675 | 0.951–0.979 |
| `makeable_split` | 1,029,286 | 728,796 | 0.585 | 0.537 | 0.705 | 0.365 | 0.764 |
| `makeable_weighted_whole` (R 2) | 2,715,882 | 727,645 | 0.789 | 0.883 | 0.772 | 0.815 | 0.904 |
| `trained_orderer` | 1,238,731 | 651,152 | 0.655 | 0.659 | 0.728 | 0.520 | 0.803 |
| `portable` | 1,230,299 | 667,983 | 0.648 | 0.643 | 0.732 | 0.501 | 0.809 |

**Per world** (sorted by size, class, gain; units are means over the confirm's episodes; screen
values in brackets; m_split = `makeable_split`, mww = `makeable_weighted_whole`, rr =
`round_robin`).

| world | choice | pattern | shocks | split | bar arm | bar | reach arm | reach | gain (screen) | floor (screen) | room | bar fill | screen → confirm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-0435 | f4 | iid | y | train | rnd_whole | 0 | m_split | 874 | +874 (+148) | 374 (13) | 1.00 | 0.00 | hard |
| 12-0273 | f4 | drift | y | held-out | random | 516 | mww | 947 | +430 (+651) | 399 (15) | 0.76 | 0.15 | hard |
| 12-0464 | f4 | cyclic | n | validation | rr | 1727 | m_split | 1890 | +163 (+163) | 17 (17) | 0.29 | 0.71 | hard |
| 12-0348 | dense | cyclic | n | train | rr | 861 | trained | 940 | +79 (+79) | 9 (9) | 0.12 | 0.88 | hard |
| 12-0033 | dense | seasonal | y | held-out | rnd_whole | 1461 | mww | 1919 | +458 (+1323) | 445 (13) | 0.75 | 0.14 | hard → **open** |
| 12-0092 | dense | hit | y | validation | rr | 3473 | mww | 3660 | +188 (+304) | 400 (34) | 0.14 | 0.17 | hard → **open** |
| 12-0140 | f4 | hit | y | train | rnd_whole | 538 | mww | 723 | +185 (+108) | 399 (14) | 0.73 | 0.21 | hard → **open** |
| 12-0060 | f4 | burst | y | train | rnd_whole | 2450 | mww | 2635 | +184 (+228) | 399 (24) | 0.47 | 0.53 | hard → **open** |
| 12-0356 | dense | burst | y | validation | rr | 1202 | m_split | 1368 | +167 (+134) | 348 (12) | 0.35 | 0.65 | hard → **open** |
| 12-0330 | dense | cyclic | n | validation | rnd_whole | 2873 | mww | 3032 | +158 (+268) | 393 (29) | 0.49 | 0.51 | hard → **open** |
| 12-0176 | f4 | cyclic | n | train | rr | 1814 | mww | 1970 | +156 (+240) | 196 (18) | 0.44 | 0.56 | hard → **open** |
| 12-0350 | f4 | burst | n | validation | rr | 679 | mww | 828 | +150 (+82) | 374 (14) | 0.23 | 0.23 | hard → **open** |
| 12-0449 | dense | burst | n | train | rnd_whole | 1508 | m_split | 1656 | +148 (+141) | 400 (14) | 0.48 | 0.52 | hard → **open** |
| 12-0504 | f4 | cyclic | y | held-out | rnd_whole | 1176 | mww | 1299 | +123 (+110) | 399 (14) | 0.31 | 0.69 | hard → **open** |
| 12-0091 | dense | burst | y | held-out | blend_0.5 | 644 | m_split | 764 | +120 (+66) | 348 (12) | 0.48 | 0.53 | hard → **open** |
| 12-0361 | dense | iid | n | train | random | 914 | mww | 1028 | +114 (+103) | 445 (14) | 0.33 | 0.67 | hard → **open** |
| 12-0302 | dense | burst | y | train | rnd_whole | 3430 | mww | 3543 | +112 (+406) | 445 (33) | 0.28 | 0.52 | hard → **open** |
| 12-0340 | dense | seasonal | y | train | rr | 1696 | mww | 1799 | +104 (+188) | 400 (17) | 0.51 | 0.49 | hard → **open** |
| 12-0005 | dense | cyclic | n | train | rr | 834 | mww | 932 | +98 (+106) | 278 (8) | 0.29 | 0.71 | hard → **open** |
| 12-0203 | dense | seasonal | n | train | rnd_whole | 1093 | m_split | 1190 | +96 (+112) | 400 (12) | 0.44 | 0.56 | hard → **open** |
| 12-0384 | f4 | switch | n | validation | rr | 635 | mww | 730 | +94 (+56) | 374 (14) | 0.65 | 0.35 | hard → **open** |
| 12-0054 | f4 | burst | n | train | rnd_whole | 1506 | mww | 1599 | +93 (+146) | 399 (15) | 0.51 | 0.49 | hard → **open** |
| 12-0365 | dense | switch | n | train | rr | 1226 | mww | 1317 | +90 (+167) | 400 (13) | 0.44 | 0.46 | hard → **open** |
| 12-0413 | dense | cyclic | y | held-out | blend_0.5 | 1118 | trained | 1206 | +88 (+99) | 348 (12) | 0.21 | 0.79 | hard → **open** |
| 12-0466 | f4 | seasonal | n | train | rr | 1510 | mww | 1596 | +86 (+134) | 374 (15) | 0.35 | 0.65 | hard → **open** |
| 12-0172 | f4 | burst | y | held-out | rnd_whole | 1729 | mww | 1810 | +82 (+161) | 399 (17) | 0.55 | 0.44 | hard → **open** |
| 12-0459 | f4 | hit | n | held-out | rnd_whole | 1349 | trained | 1430 | +81 (+150) | 374 (13) | 0.36 | 0.64 | hard → **open** |
| 12-0151 | dense | burst | y | train | rnd_whole | 1034 | mww | 1111 | +77 (+166) | 445 (13) | 0.55 | 0.45 | hard → **open** |
| 12-0016 | dense | iid | n | held-out | rnd_whole | 628 | mww | 696 | +67 (+62) | 445 (13) | 0.27 | 0.63 | hard → **open** |
| 12-0401 | f4 | hit | n | train | rr | 970 | mww | 1036 | +67 (+83) | 374 (14) | 0.42 | 0.58 | hard → **open** |
| 12-0211 | dense | seasonal | y | validation | rnd_whole | 866 | mww | 927 | +60 (+76) | 445 (13) | 0.26 | 0.74 | hard → **open** |
| 12-0446 | dense | burst | y | train | rr | 517 | m_split | 577 | +60 (+50) | 348 (12) | 0.44 | 0.56 | hard → **open** |
| 12-0094 | f4 | cyclic | y | train | blend_0.5 | 740 | m_split | 793 | +52 (+67) | 348 (13) | 0.15 | 0.85 | hard → **open** |
| 12-0239 | f2 | burst | y | held-out | rnd_whole | 1046 | mww | 1096 | +51 (+102) | 397 (29) | 0.75 | 0.24 | hard → **open** |
| 12-0470 | dense | hit | y | train | rr | 1124 | mww | 1175 | +51 (+114) | 400 (13) | 0.46 | 0.54 | hard → **open** |
| 12-0235 | f4 | switch | y | train | rr | 565 | trained | 611 | +46 (+62) | 348 (13) | 0.22 | 0.78 | hard → **open** |
| 12-0032 | dense | seasonal | y | train | rnd_whole | 558 | mww | 600 | +42 (+61) | 445 (13) | 0.23 | 0.77 | hard → **open** |
| 12-0155 | f2 | cyclic | n | train | rnd_whole | 695 | mww | 735 | +40 (-40) | 270 (8) | 0.64 | 0.29 | open |
| 12-0490 | dense | switch | y | held-out | rr | 1422 | mww | 1460 | +38 (+134) | 400 (14) | 0.38 | 0.62 | hard → **open** |
| 12-0049 | dense | seasonal | n | validation | rr | 566 | mww | 601 | +36 (+63) | 400 (13) | 0.20 | 0.80 | hard → **open** |
| 12-0139 | f4 | burst | n | train | rr | 680 | m_split | 714 | +34 (+65) | 348 (13) | 0.31 | 0.69 | hard → **open** |
| 12-0089 | f2 | burst | y | validation | rnd_whole | 781 | mww | 812 | +32 (+6) | 397 (29) | 0.62 | 0.25 | open |
| 12-0474 | f2 | burst | y | train | rr | 667 | mww | 695 | +28 (+68) | 373 (29) | 0.51 | 0.49 | hard → **open** |
| 12-0483 | dense | iid | y | train | rnd_whole | 1308 | mww | 1323 | +15 (+120) | 445 (14) | 0.45 | 0.22 | hard → **open** |
| 12-0312 | f2 | hit | y | validation | rnd_whole | 669 | mww | 669 | -0 (+61) | 397 (29) | 0.67 | 0.31 | hard → **open** |
| 12-0494 | f4 | iid | y | validation | rr | 1034 | mww | 1032 | -2 (+104) | 374 (14) | 0.36 | 0.64 | hard → **open** |
| 12-0050 | f2 | cyclic | y | validation | rnd_whole | 3506 | mww | 3490 | -16 (-21) | 397 (35) | 0.73 | 0.27 | open |
| 12-0308 | f2 | seasonal | n | train | rr | 2746 | mww | 2727 | -19 (-144) | 373 (29) | 0.66 | 0.29 | open |
| 12-0394 | dense | drift | n | train | rr | 742 | trained | 696 | -46 (+100) | 348 (12) | 0.29 | 0.71 | hard → **open** |
| 12-0109 | f2 | switch | y | validation | rr | 1774 | mww | 1723 | -52 (-30) | 373 (29) | 0.72 | 0.25 | open |
| 12-0368 | f2 | cyclic | y | held-out | rnd_whole | 2634 | mww | 2566 | -68 (+215) | 397 (29) | 0.52 | 0.39 | hard → **open** |
| 12-0344 | f2 | burst | y | train | rr | 3240 | mww | 3151 | -89 (-26) | 373 (32) | 0.71 | 0.28 | open |
| 12-0321 | f2 | hit | n | train | rnd_whole | 789 | mww | 690 | -99 (-26) | 397 (29) | 0.65 | 0.22 | open |
| 12-0391 | f2 | switch | y | train | rr | 2496 | mww | 2397 | -99 (-85) | 373 (29) | 0.73 | 0.13 | open |
| 12-0230 | dense | burst | y | train | rr | 978 | mww | 874 | -104 (-74) | 400 (13) | 0.67 | 0.32 | open |
| 12-0423 | f2 | burst | n | train | rr | 556 | mww | 442 | -114 (-1) | 373 (29) | 0.64 | 0.34 | open |
| 12-0121 | f4 | drift | n | train | rr | 4320 | mww | 4174 | -146 (+367) | 374 (44) | 0.48 | 0.22 | hard → **open** |
| 12-0408 | dense | seasonal | n | train | rnd_whole | 4004 | mww | 3843 | -160 (+546) | 445 (41) | 0.55 | 0.28 | hard → **open** |
| 12-0436 | f2 | drift | y | train | rnd_whole | 1325 | mww | 1161 | -164 (-5) | 397 (29) | 0.57 | 0.25 | open |
| 12-0111 | f2 | burst | y | train | rr | 3217 | mww | 3044 | -173 (-232) | 373 (33) | 0.77 | 0.22 | open |
| 12-0254 | f2 | cyclic | y | train | rr | 3696 | mww | 3483 | -213 (-55) | 373 (37) | 0.65 | 0.17 | open |
| 12-0278 | dense | burst | y | held-out | rnd_whole | 4157 | mww | 3929 | -228 (-144) | 445 (40) | 0.78 | 0.10 | open |
| 12-0352 | dense | switch | y | train | rnd_whole | 1549 | mww | 1131 | -418 (-396) | 445 (15) | 0.62 | 0.10 | open |
| 12-0392 | f4 | drift | n | train | rr | 4138 | mww | 3684 | -454 (-630) | 374 (42) | 0.66 | 0.34 | open |
| 42-0119 | dense | burst | n | train | rnd_whole | 11646 | mww | 13404 | +1758 (+1548) | 745 (120) | 0.45 | 0.55 | hard |
| 42-0098 | f14 | switch | n | held-out-region | rnd_whole | 9806 | mww | 11354 | +1547 (+1582) | 745 (90) | 0.47 | 0.53 | hard |
| 42-0066 | f14 | iid | n | validation | rnd_whole | 5130 | m_split | 6276 | +1146 (+1179) | 576 (79) | 0.40 | 0.60 | hard |
| 42-0126 | f14 | iid | n | train | rnd_whole | 2028 | mww | 3095 | +1067 (+1238) | 745 (90) | 0.76 | 0.24 | hard |
| 42-0031 | f14 | hit | y | train | rr | 4784 | mww | 5782 | +998 (+753) | 576 (85) | 0.51 | 0.49 | hard |
| 42-0125 | f14 | burst | n | held-out | rr | 11793 | mww | 12638 | +844 (+1192) | 576 (113) | 0.77 | 0.17 | hard |
| 42-0076 | f14 | burst | y | train | rr | 2370 | mww | 2980 | +610 (+827) | 576 (85) | 0.75 | 0.16 | hard |
| 42-0073 | f14 | iid | n | train | rr | 3456 | m_split | 4046 | +590 (+535) | 330 (73) | 0.19 | 0.81 | hard |
| 42-0054 | f14 | cyclic | y | train | rr | 5160 | trained | 5742 | +582 (+534) | 330 (73) | 0.26 | 0.74 | hard |
| 42-0008 | f14 | iid | y | held-out | rr | 2563 | m_split | 2916 | +352 (+356) | 330 (73) | 0.20 | 0.80 | hard |
| 42-0041 | f7 | cyclic | n | train | rr | 5910 | mww | 6558 | +648 (+684) | 668 (59) | 0.35 | 0.65 | hard → **open** |
| 42-0045 | f2 | drift | y | train | random | 1017 | m_split | 1548 | +532 (-346) | 576 (68) | 0.85 | 0.13 | open |
| 42-0093 | f14 | hit | y | train | rnd_whole | 2254 | m_split | 2780 | +526 (+629) | 576 (79) | 0.40 | 0.60 | hard → **open** |
| 42-0005 | f14 | hit | n | held-out | rr | 2392 | mww | 2841 | +449 (+243) | 576 (85) | 0.60 | 0.40 | hard → **open** |
| 42-0000 | f7 | hit | y | train | rnd_whole | 3187 | mww | 3574 | +388 (+409) | 745 (41) | 0.55 | 0.45 | hard → **open** |
| 42-0127 | dense | switch | n | held-out-region | rnd_whole | 4343 | mww | 4698 | +355 (+466) | 745 (51) | 0.32 | 0.68 | hard → **open** |
| 42-0072 | f14 | hit | n | validation | rnd_whole | 2312 | mww | 2662 | +351 (+410) | 745 (90) | 0.55 | 0.45 | hard → **open** |
| 42-0122 | f14 | seasonal | n | train | rr | 1637 | m_split | 1882 | +246 (+281) | 330 (73) | 0.17 | 0.83 | hard → **open** |
| 42-0034 | f7 | burst | y | train | rr | 3058 | mww | 3302 | +244 (+299) | 576 (38) | 0.51 | 0.49 | hard → **open** |
| 42-0113 | f14 | burst | y | train | rr | 1398 | mww | 1607 | +208 (+274) | 576 (85) | 0.40 | 0.60 | hard → **open** |
| 42-0037 | f4 | iid | n | train | rr | 2118 | mww | 2288 | +169 (+242) | 576 (23) | 0.27 | 0.73 | hard → **open** |
| 42-0007 | f4 | iid | n | held-out | rr | 1944 | trained | 2113 | +168 (+221) | 330 (19) | 0.16 | 0.84 | hard → **open** |
| 42-0102 | dense | burst | y | validation | rnd_whole | 14067 | mww | 14174 | +107 (-85) | 745 (137) | 0.64 | 0.36 | open |
| 42-0052 | f2 | cyclic | y | train | rnd_whole | 1952 | mww | 2033 | +80 (+108) | 745 (70) | 0.15 | 0.85 | hard → **open** |
| 42-0087 | f2 | switch | n | held-out-region | rnd_whole | 9052 | mww | 9050 | -2 (-51) | 745 (93) | 0.65 | 0.24 | open |
| 42-0040 | f2 | burst | y | train | rr | 6314 | mww | 6261 | -54 (-97) | 576 (68) | 0.64 | 0.35 | open |
| 42-0071 | f2 | switch | y | held-out-region | rnd_whole | 8716 | mww | 8634 | -81 (-218) | 745 (89) | 0.64 | 0.12 | open |
| 42-0043 | f4 | seasonal | y | train | rr | 2670 | mww | 2564 | -106 (-186) | 576 (26) | 0.64 | 0.36 | open |
| 42-0081 | f4 | drift | y | train | rnd_whole | 5358 | mww | 5216 | -143 (+824) | 745 (37) | 0.33 | 0.16 | hard → **open** |
| 42-0013 | f4 | burst | y | train | rr | 14510 | mww | 14226 | -284 (-808) | 576 (146) | 0.72 | 0.23 | open |
| 42-0039 | f4 | iid | n | train | rr | 16422 | mww | 15919 | -503 (-474) | 576 (162) | 0.71 | 0.24 | open |
| 42-0051 | f7 | seasonal | y | train | rr | 1425 | m_split | 481 | -944 (+204) | 330 (32) | 0.26 | 0.74 | hard → **open** |

## Conclusions

- Most of the screen's hard labels do not survive the pre-registered confirm: on fresh seeds with
  its own pooled spreads (D12's estimator), 4 of 48 selected hard worlds stay hard at 12 producers
  and 10 of 24 at 42; every other selected world is open. Both pooled spreads are concentrated in
  a few worlds (σ_ep 89% / 72%, σ_rep 69% / 87% of the sum of squares from 5 worlds; at 42 the
  floor is mostly σ_rep, led by a `random` replicate that dies), so the count depends on the
  estimator: 2–12 at 12 and 5–17 at 42 at 3 SE across eight estimates. DIAGNOSTICS 21's "a stage
  with its own spread decides" branch applied, and the confirm's own pooled floor shows a second
  form of the problem (absolute pooling over worlds of very different volume and variance).
- Confirmed headroom is a 42-producer, wide-choice (f14) phenomenon, with input-aware whole-order
  routing as the visible lever (median +16% of the bar). At 12 producers it is isolated.
- EXP-0050's deciding check: no pre-registered branch holds (4 / 8 / 2 of 14 against 10 needed);
  splitting still costs on 8 of 14 (7 of 10 at 42). Input-aware routers cut the share of expired
  reservations whose receiver lacked inputs: on consumer orders from ≈ 0.89 to 0.71–0.73, on
  investment orders further (0.37–0.52 for the split-capable ones).
- Open worlds dominate both sizes (bar ≈ 46% fill, room ≈ 0.5 under the input-output ceiling);
  their reachable share is unknown until PoC-A or a lead-time-aware bound.
- Atlas v1 is frozen (`atlas_v1.json`, `f241464c4bc4…`; split train 60 / validation 16 /
  held-out 16 / held-out-region 4). Gate registry, roster and parameters are untouched; adding
  atlas worlds to a gate is the user's decision.

## Addendum (task 5.3, 2026-10-29): robust noise floor, dead-arm rule, atlas v2

**What was asked.** The user's decision after this run was "fix the floor, then sample". Two
defects were diagnosed above (DIAGNOSTICS 20 and 21): the noise floor was set by a handful of
worlds, and one "hard" world (12-0435) was hard only because the arm it was compared against
died. Task 5.3 changes the two classification rules and classifies this run's recorded rows again,
offline: no episode was replayed, and the raw files (`artifacts/episodes.csv`, `worlds.csv`,
`deciding_check.csv`, `atlas_v1.json`) are unchanged. The derived files are in
`reclassified/` (`worlds.csv` with the old class in `class_v1`, `deciding_check.csv`,
`atlas_v2.json`, `reclassify.json`).

**The two rule changes, in plain words.**
- *Noise floor.* The floor says how much of a gain could be luck. It is built from how much the
  gain moves between a world's two episodes (σ_ep) and between two runs of a random arm on the
  same episode (σ_rep), pooled over many worlds. The pre-registered rule pooled them as a
  standard deviation, which squares every swing, so a few worlds with huge swings set the floor
  for everyone: 5 worlds carried 89% (12 producers) and 72% (42) of σ_ep's sum of squares. The
  new rule takes the **median** swing instead (the median absolute residual, scaled so that it
  equals the standard deviation when the swings are normal). A median ignores how extreme the
  wildest minority is, which is the property needed here; it costs precision (it behaves like a
  standard deviation from about 37% as many worlds), which the design's false-alarm estimate
  accounts for (≈ 0.3–0.6% per world instead of ≈ 0.2%). Excluding only worlds with a death was
  checked first and does not help: σ_ep without them is 168 / 159 units against 164 / 156, because
  the big swings are whole-episode differences, not deaths.
- *Dead-arm rule.* Confirm compares each world against two arms fixed from the screen: the best
  blind arm (the bar) and the best reach arm. If either of them reaches system death in any
  episode or replicate, the world is now **collapse**, never hard or open: the dead episode's
  units measure the death, not the routing, and a dying bar also inflates the room under the
  ceiling. The rule is wider than the bar-death case: a dying *reach* arm does not bias the room,
  so three worlds that read open (12-0312, 12-0423, 42-0051) are excluded on purpose (they would
  otherwise feed the teacher sample with worlds where a cheap input-aware policy can kill the
  economy), and a single late death counts too (12-0140, one replicate at cycle 84). "Collapse"
  now has two meanings: every arm fails (none here), or a fixed arm dies while others may deliver
  normally (all 6 here; `argmax_died` = true). Design D12 "Classes" states the trade-off.

**Honesty note.** The robust estimator was chosen after the sensitivity counts of this run were
known (the "median-based" row gave 12 / 13 hard worlds). It is justified in design D12 by its
statistical properties (a 50% breakdown point against a 0% one, no tuning constant, no scaling
model), not by the count, and both counts are shown side by side below. The pre-registered count
stays the run's recorded result; the new one is the revised rule's reading of the same rows.

**What changed.**
- The pooled spreads shrink where outliers carried them: σ_ep 164 → 55 units at 12 producers,
  156 → 141 at 42; σ_rep 131 / 90 / 92 → 44 / 73 / 41 at 12 (dense / f2 / f4) and 315 → 122 at 42.
  The median floor falls from 397 to 145 units at 12 producers and from 577 to 349 at 42.
- Counts per size, pre-registered → new:

  | size | hard | open | collapse | saturated / flat |
  |---|---|---|---|---|
  | 12 producers | 4 → **11** | 60 → **49** | 0 → **4** | 0 / 0 |
  | 42 producers | 10 → **13** | 22 → **17** | 0 → **2** | 0 / 0 |

- **Flips at 12 producers (12 worlds).** Eight open worlds become hard because their gain now
  clears the smaller floor: 0005 (+98 against a floor of 93; was 278), 0060 (+185 / 145; was
  399), 0091 (+121 / 116), 0092 (+188 / 134), 0176 (+156 / 87), 0330 (+159 / 132), 0350
  (+150 / 131), 0356 (+167 / 116); two of these are marginal (0005, 0091 within 5% of the
  floor). Four worlds become collapse because a fixed arm died: 0435 (was hard; its bar arm
  `random_whole` dies at cycle 4 in all 4 runs, bar 0, while `shock_reactive` delivers 656), 0140
  (bar `random_whole` dies at cycle 84 in one replicate), 0312 (reach `makeable_weighted_whole`
  dies at cycle 77 in one replicate) and 0423 (reach `makeable_weighted_whole` dies at cycle 13 in
  one replicate).
- **Flips at 42 producers (5 worlds).** Three open worlds become hard (0005 +449 against 349,
  0041 +648 against 258, 0093 +526 against 349); two become collapse: 0045 (bar `random` dies at
  cycles 17–23 in one replicate per episode, the death that led σ_rep) and 0051 (reach
  `makeable_split` dies at cycle 29 in both episodes, gain −944).
- **Where headroom is.** 12 producers: 6 dense and 5 f4 worlds, none at f2; 42 producers: 11 f14,
  1 f7, 1 dense. The 42-producer wide-choice reading holds; at 12 producers the earlier "isolated"
  reading came from the floor, and hard worlds now appear at dense and f4 choice with gains of
  5–22% of the bar (12-0273, held-out, 83%).
- **Per-world check of the new floor's assumption.** A pooled median assumes the stratum shares
  one noise scale. On every 12-producer hard world the world's own episode-to-episode swing is at
  most 1.1× the pooled σ_ep; at 42 producers three hard worlds swing more than 2× (0066 2.1×,
  0076 2.1×, 0098 3.4×; all hard under the pre-registered rule too), so their margin is thinner
  than the floor says.
- **EXP-0050's deciding check** under the new classes (24 hard worlds with E ≥ 4): input-aware
  weighting 5 / 24, splitting costs 15 / 24, scorer beyond target 7 / 24; still no branch at 2/3.
- **Atlas v2** (`reclassified/atlas_v2.json`, content hash `220886a7cf12…`; `reclassified_from`
  pins `episodes.csv` `0e22ddd0…`, `worlds.csv` `14e9e132…` and atlas v1 `f241464c…`). Split,
  fixed arms and world records are atlas v1's. PoC-A and PoC-C read atlas v2.

**PoC-A candidates** (design D5: train split, 12 producers, patterns cyclic / iid / drift / burst
/ seasonal; 6 hard by gain + 6 open by room, the other class filling a shortfall; half of each
class shocked when available; * = shocked):

| rule | hard (by gain) | open (by room) |
|---|---|---|
| pre-registered (atlas v1) | 0435* (artifact), 0348 | 0111*, 0344*, 0230*, 0254*, 0436*, 0308, 0392, 0423, 0155, 0408 |
| new (atlas v2) | 0060* (+185), 0176 (+156), 0005 (+98), 0348 (+79) | 0111* (0.77), 0344* (0.72), 0230* (0.67), 0254* (0.65), 0308 (0.66), 0392 (0.66), 0155 (0.64), 0408 (0.55) |

With 4 hard train worlds at 12 producers, open worlds fill to 12 (4 hard + 8 open); only one hard
train world has shocks, so the shocked half of the sample comes from the open class (5 of 12
shocked). The pre-registered sample would have carried 12-0435 and 12-0423, both worlds whose
fixed arm dies.

**Verification.** `python3 openspec/changes/add-m3-development-audit/probes/exp0052_check.py
experiments/EXP-0052-headroom-atlas-confirm --screen experiments/EXP-0051-headroom-atlas-screen
--reclassified experiments/EXP-0052-headroom-atlas-confirm/reclassified` recomputes the spreads,
floors, classes, flips, the atlas v2 hash and both PoC-A samples independently of
`tokenomics.eval.atlas` and prints ALL CHECKS PASS. Command that wrote `reclassified/`:
`python -m tokenomics.eval.atlas --mode reclassify --confirm-dir
experiments/EXP-0052-headroom-atlas-confirm --out-dir
experiments/EXP-0052-headroom-atlas-confirm/reclassified` (local, about 1 s).

## Details

- Command (from the repo root): `python3 run_pod_training.py --create-pod --image
  runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 --vcpu 8 --mem 16 --disk 30 --cpu-flavor
  cpu5c --pod-name exp0052-headroom-atlas-confirm --verify-source-commit --train "-u -m
  tokenomics.eval.atlas --mode confirm --screen-dir /workspace/tokenomics/_pod_ckpts/exp0051
  --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt --orderer-ckpt
  /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --workers 8 --source-commit {SOURCE_COMMIT}
  --out-dir /workspace/tokenomics/runs/exp0052" --pull
  "/workspace/tokenomics/runs/exp0052:./experiments/EXP-0052-headroom-atlas-confirm"`. Inputs
  staged under the git-ignored `tokenomics/_pod_ckpts/` (both checkpoints; EXP-0051 `spec.json`,
  `worlds.csv`, `selection.csv` under `exp0051/`), deleted after the run. The `lp` extra (scipy)
  was installed on the pod (`.[train,lp]`); B_lp ran on every world-episode.
- **Pod shape and why.** Planned: fresh secure 16 vCPU (`cpu5c`). Secure 16 vCPU creation
  returned "no instances available" for `cpu5c`, `cpu3c`, `cpu5g`, `cpu3g` (07:13Z, no pod
  created). A retry created pod `jjiy4yl9uj74jf` (16 vCPU `cpu3c`, 07:13:41Z) that never got a
  container (runtime `initializing / awaiting_container`, SSH timeout at 600 s); the launcher
  stopped and terminated it at ≈ 07:24:05Z: ≈ 10.4 min, **≈ 2.8 vCPU-h-equivalent (≈ $0.08) if
  RunPod bills initialization**, no output; counted in M3 spend, not against this run's stop. The
  user then chose a smaller secure pod (no community cloud): pod `90kwaumzj6zf8x`, 8 vCPU `cpu5c`,
  16 GB, `--workers 8`.
- **Spend.** Pod life 07:24:24Z (create) → ≈ 07:52:45Z (terminated) = 28.4 min × 8 vCPU = **≈ 3.8
  vCPU-h billed**; the driver ran 07:25:22 → 07:52:37 (27.3 min, 3.6 pod vCPU-h; recheck estimate
  4.03). Hard stop: planned 6 vCPU-h (45 min on 8 vCPU); at 07:38Z the supervisor extended it to
  7.4 vCPU-h (08:20Z) because the 8 vCPU run projected to finish at ≈ 08:06Z (42-producer rows at
  ≈ 24 rows / min, ~10% slower than sized); not needed, the run ended at 07:52Z. Pod list after
  the run: `[]`.
- **M3 spend** (budget 40 vCPU-h): EXP-0048–0050 ≈ 10.8 + EXP-0051 ≈ 7.3 + EXP-0052 ≈ 3.8 =
  **≈ 21.9 vCPU-h billed**; ≈ 24.7 if the failed pod's initialization was billed (not confirmed:
  an upper estimate from its pod life); ≈ 15–18 remain.
- **Driver choices behind the numbers** (not in `spec.json`'s `driver_choices`): (1) the deciding
  check's floors reuse the world's stratum σ_ep (pooled from the bar − reach gain, not from the
  split − trained or whole − split differences) plus σ_rep² / R for each stochastic arm of the
  compared pair, with the 1% term on the world's bar, at 3 SE (e.g. 347.7 = 3 · 163.9 / √2;
  374.2 = 3 · √(163.9² + 92.2² / 2) / √2; 8.6 = 0.01 · 861); (2) the ceiling's "best played arm" is
  the largest single replicate, not the replicate mean (only matters on 12-0350: C 883.9 against
  855.1, room 0.232 against 0.206; class unchanged); (3) the TTL input-stock columns of
  `deciding_check.csv` sum consumer and investment orders.
- Arms (15 rows per world-episode): roster v2 (`entity_planner` with `even_split`,
  `proportional`, `obs_price`, `blend_0.5`, `shock_reactive`, `random` R = 2), `round_robin`,
  `random_whole` R = 2 (blind); `makeable_split`, `makeable_weighted_whole` R = 2,
  `trained_orderer` (reach); `portable` (reported). Bar / reach argmax fixed from EXP-0051.
- Files: `spec.json` (pins, provenance, result summary), `worlds.csv`, `deciding_check.csv`,
  `atlas_v1.json`, `contract_failures.csv`, `artifacts/episodes.csv` (git-lfs).
