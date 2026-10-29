# EXP-0051 — Headroom atlas, screen

**Status: complete, valid.** `python -m tokenomics.eval.atlas --mode screen`, run commit
`92f317725ca4fe02bd703b950008ee2a3cb64335` (task 5.1 of `add-m3-development-audit`, design D12,
ADR-0057). Fresh `cpu16` pod (`--create-pod`, `runpod/base:0.7.0-ubuntu2004`, 16 vCPU, `--workers
16`, BLAS 1 thread), `--verify-source-commit`. 7,776 episode rows over 648 world-episodes (512
worlds at 12 producers, 128 at 42, 8 EXP-0050 anchors; 1 episode each; 12 rows per world-episode),
horizon 100. Pod life 05:10:13 → 05:37:31 UTC = 27.3 min = **≈ 7.3 vCPU-h billed**; the driver
ran 05:11:21 → 05:37:10 (25.8 min, 6.9 pod vCPU-h against the 8.26 sized in `sizing.md`
§EXP-0051; hard stop 10). Screen classes are provisional: they drive the selection only, and
the confirm stage (EXP-0052) decides.

## In plain English

**The question.** Earlier M3 runs (EXP-0048–0050) found that the development worlds sit in an
"easy corner": on most of them the best simple routing rule already fills 97–99% of consumer
orders, so no learner could show an improvement there. This screen asks where, in a much wider
space of worlds, a better policy than the best blind or hand-written rule can be *reached*, and
how much room the world leaves above that rule.

**The world space.** Each world is a point on seven axes (design D10). *Size*: 12 or 42
producers. *Choice*: how many producers can make each product (sparse with 2, 4, 7 or 14 makers,
or dense, where every producer of a layer can make every product of that layer). *Tightness ρ*:
expected demand per cycle against the top layer's starting throughput (ρ = 1 means demand equals
what the makers can deliver at the start). *Order size σ*: a typical order against a maker's
capacity. *Heterogeneity*: how unequal producers are in capacity, degradation and cost (three
coefficients of variation). *Shocks*: off, or capacity hits and outages at a sampled rate.
*Demand pattern*: cyclic, independent random, drift, bursts, a moving "hit" product, seasonal
waves, or regime switches. 512 worlds at 12 producers and 128 at 42 were drawn to cover these
axes evenly (a Latin hypercube), each played for one 100-cycle episode.

**What was played.** Every world was played by two groups of ordering policies, all with the
same hand-written producer (`entity_planner`). The **atlas bar** is the best of eight blind and
hand-written orderers (the six roster v2 rules, `round_robin`, `random_whole`). The **reach
arms** are three cheaper-than-search policies that might do better: `makeable_split` and
`makeable_weighted_whole` (route by how many units each maker can finish now, split or whole)
and the trained orderer. For each world-episode an analytic bound gives a **ceiling** C (the most
consumer units the world could deliver, from the demand schedule and base capacity).

**The classes** (pre-registered, checked in this order):
- *collapse*: every policy kills the economy before cycle 50;
- *saturated*: the bar already fills ≥ 95% of orders, or reaches ≥ 97% of the ceiling — nothing
  left to gain;
- *hard*: a reach arm beats the bar by more than the **noise floor** (how much two policies differ
  by chance alone on this kind of world, estimated from EXP-0050's repeated episodes; the screen
  uses 2 standard errors, or 1% of the bar if larger) and in every replicate — a better policy is
  reachable and measured;
- *open*: not hard, but the ceiling is ≥ 10% above the bar (**room** ≥ 0.10) — room nobody cheap
  reached, candidates for an expensive teacher (PoC-A);
- *flat*: the rest.

**What we found.** The easy corner is small: only **21 of 512** 12-producer worlds and **0 of
128** 42-producer worlds are saturated. Far fewer worlds are *measurably* hard than the screen's
own labels say. The screen labels **213 hard and 274 open** at 12 producers and **68 hard and 58
open** at 42, but its noise floor is too low on these worlds (next paragraph). With the spread the
screen itself measured, about **63–71 worlds at 12 (12–14%) and 34 at 42 (27%)** keep a gain above
the floor; the 213 / 68 are an upper bound. On a typical labelled-hard world the best reach arm
beats the bar by about 5–6% (median 64 units at 12, 297 at 42), and on a typical open world the
bar fills only 39–44% of consumer orders and delivers 45–52% less than the ceiling allows. The reach arm
that wins is almost always `makeable_weighted_whole` (226 of 281 hard worlds): sending each order
whole to a maker drawn in proportion to what it can finish now beats both the rotating
`round_robin` and the blind `random_whole`, which set the bar on almost every world.

**The noise floor is borrowed from easy worlds.** A screen world is played once, so it cannot
measure its own chance variation; the pre-registered floor took it from EXP-0050's repeated
episodes (design D12). Those are easy-corner worlds that fill 97%+ of orders, and the spreads
have only 4–40 degrees of freedom. But the screen does measure one spread directly:
`random_whole` is played twice on every world with different random seeds. On the atlas worlds
two such replicates differ by a pooled SD of **46–77 units at 12 producers and 156–343 at 42**
(about 2–4% of the units, more where replicates die), against **2.6 and 10–21** in the EXP-0050
spreads the floor used: 18–35 times larger. 178 of the 213 hard worlds at 12 (58 of 68 at 42)
have a stochastic winning or bar arm, so their floor is mostly this replicate term. Using the
screen's own replicate spread (absolute, or relative to the bar) and keeping EXP-0050's
episode spread, the hard count at the screen's 2 × SE becomes 63 / 71 at 12 and 34 / 34 at 42;
at confirm's 3 × SE, 41 / 47 and 25 / 23. The episode-to-episode spread on atlas worlds is not
measured at all (one episode each) and is probably also larger than EXP-0050's. The classes were
computed exactly as pre-registered; it is the pre-registered input that does not transfer. The
selection still mostly holds: 41 of its 48 hard worlds at 12 and 21 of 24 at 42 clear the floor
built on the relative replicate spread. Confirm (EXP-0052) pools its own spreads from its own
replicated episodes, so it does not inherit the problem.

**Where.** Saturation lives in two thin places: the low-tightness, small-order corner (10 worlds
with ρ ≤ 0.23 and σ ≤ 0.48 fill ≥ 95%; it contains the EXP-0050 family worlds), and 11
over-demanded worlds (ρ up to 14) where demand is so far above base capacity that the bar already
delivers ≥ 97% of what the ceiling allows. Hard worlds concentrate at lower tightness (ρ in the
first three quartiles: 188 of 213 at 12) and, at 42 producers, at wide choice: 22 of 26 worlds
with 14 makers per product are hard, against 2 of 26 with 2 makers (24 of those are open). Open
worlds concentrate at high tightness (ρ's top quartile: 92 of 128 at 12, 24 of 32 at 42): demand
the bar cannot serve, where routing alone does not close the gap. Shocks and the demand pattern
change the mix less: every pattern has 25–35 hard worlds at 12 producers.

**The anchors.** Eight EXP-0050 world-episodes were replayed inside the screen; all 80 rows that
the two runs share reproduce EXP-0050's units exactly, so the two drivers measure the same thing.
EXP-0050's easy corner shows up as such: the 12-producer fanout-4 anchor is saturated (fill
0.976), as are three of the five 42-producer anchors (f4, f7, dense). The dense-12 anchor sits at
fill 0.946, just under the 0.95 cut, and reads hard by a small margin (the trained orderer +19
units against a floor of 11.8). The 42-producer fanout-14 anchor is hard (+222 units by
`makeable_split`), matching EXP-0050's finding that routing matters most at wide fanout.

**Selection and split.** The pre-registered selection took 64 worlds at 12 producers (48 hard, 16
open) and 32 at 42 (24 hard, 8 open), first one world for every uncovered axis level and
coordinate quartile, then the best scores. One coverage bin per size could not be covered
because it is empty (the second quartile of shock events per producer: half the worlds have no
shocks, so its cut points coincide at 0). The split puts every 42-producer regime-switch world (4)
in an out-of-region held-out part, then 20% held-out / 20% validation / 60% train within each
size × choice × pattern stratum: 60 train, 16 validation, 16 held-out, 4 held-out-region.

**What this means.** For the confirm stage (EXP-0052): the 96 selected worlds are replayed on
fresh seeds with two replicates of the stochastic arms and `portable`, so the screen's
single-episode gains, and its strong lean on the stochastic `makeable_weighted_whole` played once,
get tested against a winner's-curse bias and against a noise floor measured on atlas worlds;
the recheck extrapolation for it is 3.91 pod vCPU-h, under its 6 vCPU-h stop. For M4: saturation
is rare outside the easy corner, headroom measured against the screen's own spread exists on
roughly an eighth of the 12-producer worlds and a quarter of the 42-producer ones, and the
cheapest lever already visible is input-aware whole-order routing; the open
worlds (high tightness) are where a teacher would have to find something routing cannot. Whether
any of this enters a gate registry or roster v3 is the user's decision after confirm.

## Verification

Independent check: `python3 /srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/exp0051_check.py
experiments/EXP-0051-headroom-atlas-screen --exp0050 <EXP-0050 artifacts/episodes.csv> --ckpt
producer=… --ckpt orderer=…` (stdlib only; D12's measures, spreads, floors, classes, bins,
selection and split re-implemented; arm table from the roster v2 file) → **ALL CHECKS PASS**:

| check | result |
|---|---|
| `contract_failures.csv` | header-only (0 rows) |
| planned (world-episode, arm, replicate) rows | 7,776 of 7,776, no duplicates, no unplanned subject |
| digest pairs | exactly one (demand, shock) digest pair per world-episode (648) |
| B2 violations (units > B2 + 0.01) | 0 |
| anchors vs EXP-0050 `episodes.csv` | 80 of 80 shared rows equal; `anchors.csv` all `true` |
| provisional spreads | recomputed from EXP-0050, equal to `spec.json` (and `sizing.md` §EXP-0051) |
| screen's own σ_rep (`random_whole` R = 2, pooled per size × choice; review finding) | 12: dense 63.5, f2 46.4, f4 76.7 (df 170–171); 42: dense 156, f2 188, f4 343, f7 183, f14 157 (df 25–26); EXP-0050's: 2.6 at 12, 9.7–21.3 at 42 |
| hard kept with the screen's σ_rep (absolute / relative × bar; EXP-0050 σ_ep kept) | 2 × SE: 63 / 71 at 12, 34 / 34 at 42; 3 × SE: 41 / 47, 25 / 23; selected hard: 28 / 41 of 48 at 12, 17 / 21 of 24 at 42 (relative, 2 × SE) |
| `worlds.csv` | bar/reach arms and units, gain, fill, ceiling, room, SE, floor, sign, class, arms above B_base: equal for all 648 worlds |
| `selection.csv` | 96 rows equal (rank, world, class, pass, split); `worlds.csv` split column equal |
| checkpoints | producer `8b51b77cb5ac…`, orderer `d25388bfa33e…`: staged files = `spec.json` pins |
| EXP-0050 input | `exp0050_episodes_sha256` `187a5800a21c…` |

Staging note: the EXP-0050 `episodes.csv` was staged at `tokenomics/_pod_ckpts/exp0050_episodes.csv`
(git-ignored), not `tokenomics/_pod_inputs/` (not ignored: an untracked file there fails
`--verify-source-commit`). Staged files were deleted after the run.

## Tables

Classes per size (anchors apart):

| size | worlds | collapse | saturated | hard | open | flat |
|---|---|---|---|---|---|---|
| 12 | 512 | 0 | 21 | 213 | 274 | 4 |
| 42 | 128 | 1 | 0 | 68 | 58 | 1 |
| anchors | 8 | 0 | 4 | 2 | 1 | 1 |

Per class, quartiles q25 / median / q75:

| size, class | bar fill | room | gain (units) | gain / bar | floor | ρ | σ |
|---|---|---|---|---|---|---|---|
| 12 saturated (21) | 0.32 / 0.63 / 0.98 | 0.003 / 0.015 / 0.022 | −40 / 0 / 6 | | 12.6 / 21.7 / 28.4 | 0.079 / 0.60 / 2.34 | 0.091 / 0.30 / 0.65 |
| 12 hard (213) | 0.48 / 0.59 / 0.71 | 0.26 / 0.39 / 0.49 | 39 / 64 / 109 | 0.034 / 0.049 / 0.075 | 12.9 / 14.8 / 24.9 | 0.18 / 0.34 / 0.70 | 0.15 / 0.30 / 0.56 |
| 12 open (274) | 0.30 / 0.44 / 0.59 | 0.34 / 0.45 / 0.56 | −118 / −48 / −4 | | 20.9 / 28.9 / 29.1 | 0.29 / 0.63 / 1.45 | 0.18 / 0.38 / 0.78 |
| 42 hard (68) | 0.41 / 0.61 / 0.80 | 0.21 / 0.36 / 0.54 | 159 / 297 / 535 | 0.045 / 0.061 / 0.116 | 41 / 71 / 90 | 0.19 / 0.34 / 0.69 | 0.17 / 0.29 / 0.56 |
| 42 open (58) | 0.25 / 0.39 / 0.55 | 0.37 / 0.52 / 0.58 | −523 / −210 / −50 | | 68 / 69 / 92 | 0.32 / 0.79 / 1.60 | 0.25 / 0.47 / 1.06 |

Saturated at 12: 10 by fill (all ρ ≤ 0.23, σ ≤ 0.48), 11 by the ceiling only (fill 0.15–0.63;
ρ 0.6–14.2, B_base caps them). On 126 of 274 open worlds at 12 (28 of 58 at 42) the
base-capacity bound B_base is below the demand cap B2 and sets C; B_base ignores degradation,
so part of the open room is optimistic. Dropping B_base (C = max(B2, best arm)) only raises C,
so it removes no open world; the informative check is a degradation-aware B_base: scaled by the best arm's end-of-episode
capacity ratio r_end, C = max(min(B2, B_base · r_end), best arm), **202 of 274** open worlds at 12
and **37 of 58** at 42 stay open (at (1 + r_end) / 2: 249 and 55; the reviewer's own recount,
best-arm r_end taken differently, gave 195 and 36). On open worlds the best played arm sits at
only 1.006× the bar at 12 and 1.000× at 42 (median), so "open" is room against an optimistic bound
that no played arm approaches, not demonstrated headroom; that is what the teacher (PoC-A) is for.

**Zero-unit rows.** 96 rows on 53 worlds deliver 0 units without system death. All come from
splitting roster rules (`obs_price` 27, `even_split` 27, `proportional` 18, `blend_0.5` 16,
`shock_reactive` 8), never a whole-order or reach arm. They split each consumer order across
4.4 producers (median `trace_split_degree_consumer`); a split order completes only if every part
does, the parts expire on their TTL, mostly with the receiving producer's inputs short, and the
parts that did complete are counted as wasted. These worlds are the tight, large-order corner
(median σ 0.56 against 0.35 overall, ρ 1.45 against 0.47); their classes are open 28, hard 22,
saturated 2, flat 1. This is EXP-0048's whole-order factor at its extreme, not a defect.

By region (hard / open counts; the rest saturated, flat or collapse):

| size | choice | n | hard | open | sat. | hard reach arm (mww / split / trained) | median gain/bar of hard |
|---|---|---|---|---|---|---|---|
| 12 | sparse_f2 | 171 | 32 | 134 | 2 | 30 / 0 / 2 | 0.045 |
| 12 | sparse_f4 | 171 | 85 | 75 | 10 | 65 / 14 / 6 | 0.049 |
| 12 | dense | 170 | 96 | 65 | 9 | 76 / 10 / 10 | 0.053 |
| 42 | sparse_f2 | 26 | 2 | 24 | 0 | 2 / 0 / 0 | 0.036 |
| 42 | sparse_f4 | 26 | 12 | 13 | 0 (1 collapse) | 8 / 2 / 2 | 0.045 |
| 42 | sparse_f7 | 25 | 17 | 8 | 0 | 16 / 1 / 0 | 0.051 |
| 42 | sparse_f14 | 26 | 22 | 4 | 0 | 14 / 7 / 1 | 0.123 |
| 42 | dense | 25 | 15 | 9 | 0 (1 flat) | 15 / 0 / 0 | 0.055 |

| size | tightness ρ quartile (cuts) | Q1 hard / open / sat | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| 12 | 0.202 / 0.471 / 1.12 (range 0.027–14.2) | 68 / 49 / 9 | 66 / 60 / 1 | 54 / 73 / 1 | 25 / 92 / 10 |
| 42 | 0.213 / 0.444 / 1.27 (range 0.028–13.0) | 21 / 9 / 0 | 19 / 13 / 0 | 20 / 12 / 0 | 8 / 24 / 0 |

Patterns at 12 (hard / open): cyclic 35/35, iid 27/39, drift 25/42, burst 35/37, hit 31/39,
seasonal 32/39, switch 28/43; shocks off 113/128, on 100/146. At 42: cyclic 13/6, iid 13/5,
drift 7/10, burst 9/10, hit 11/7, seasonal 6/11, switch 9/9; shocks off 40/23, on 28/35. The
other coordinates (σ, the three heterogeneity CVs, shock events, demand concentration and
volatility) move the hard / open mix by at most about 2 : 1 between quartiles; full quartile
tables in the check script's output and `coverage.csv`.

Bar argmax arm: `round_robin` 276 / 86, `random_whole` 211 / 41 (12 / 42), others 25 / 1. Arms
above B_base (world-episodes, all at 12 producers): `makeable_weighted_whole` 14, `round_robin`
13, `random_whole` 11, six others 1–2 each; none at 42.

Anchors (screen class, bar arm units, reach arm units, floor):

| anchor | class | bar | reach | gain | floor | fill | room |
|---|---|---|---|---|---|---|---|
| dense-12-f4 | hard | 548 (`round_robin`) | 567 (trained) | +19 | 11.8 | 0.946 | 0.054 |
| sparse_wide-12-f2 | flat | 474 (`round_robin`) | 464 (`makeable_split`) | −10 | 28.4 | 0.926 | 0.074 |
| sparse_wide-12-f4 | saturated | 540 (`round_robin`) | 550 (`makeable_split`) | +10 | 12.6 | 0.976 | 0.024 |
| sparse_wide-42-f2 | open | 1,673 (`round_robin`) | 1,720 (`makeable_split`) | +47 | 61.6 | 0.890 | 0.110 |
| sparse_wide-42-f4 | saturated | 1,943 (`round_robin`) | 1,971 (mww) | +28 | 22.5 | 0.958 | 0.042 |
| sparse_wide-42-f7 | saturated | 1,889 (`round_robin`) | 1,904 (mww) | +15 | 38.2 | 0.971 | 0.029 |
| sparse_wide-42-f14 | hard | 1,876 (`round_robin`) | 2,098 (`makeable_split`) | +222 | 73.2 | 0.845 | 0.155 |
| dense-42-f14 | saturated | 1,882 (`random_whole`) | 1,912 (mww) | +30 | 50.5 | 0.956 | 0.044 |

Selection (`selection.csv`): 12 producers 64 (hard 48, open 16; choice f2 16 / f4 20 / dense 28;
shocks on 38; 14 by coverage pass), 42 producers 32 (hard 24, open 8; f2 5 / f4 6 / f7 4 / f14 14 /
dense 3; shocks on 17; 9 by coverage pass). Split: 12 → train 39, validation 13, held-out 12;
42 → train 21, validation 3, held-out 4, held-out-region 4. Uncovered bins: only the empty
`shock_events_per_producer` Q2 at both sizes (0 worlds).

## Conclusions

- Saturation is rare outside the low-tightness, small-order corner where EXP-0047–0050's worlds
  live (21 of 640 worlds). Measured headroom is narrower than the screen's labels: 281 hard by the
  pre-registered floor is an upper bound; with the screen's own replicate spread about 97–105
  (63–71 at 12, 34 at 42) keep their gain at 2 × SE. DIAGNOSTICS entry 20's "saturated" branch
  applies to the easy corner only; entry 21 records the borrowed-floor cause.
- The visible cheap lever is input-aware whole-order routing (`makeable_weighted_whole` wins 226
  of 281 hard worlds), strongest at 42 producers with 14 makers per product (median +12% over the
  bar). This sharpens EXP-0050's deciding check, which confirm reads on its hard worlds with
  E ≥ 4.
- Caveats for confirm: the screen's hard calls rest on one episode, EXP-0050's spreads (18–35×
  too small for σ_rep on atlas worlds; σ_ep unmeasured) and a 2 × SE floor, and the winning arm is stochastic with one replicate; open room partly rests on B_base,
  which ignores degradation. EXP-0052 replays the 96 selected worlds on fresh seeds (recheck
  extrapolation 3.91 pod vCPU-h, stop 6) and decides.

## Addendum (task 4.7, 2026-10-29): the input-output ceiling

**In plain English.** The screen judged room against two simple bounds: B2, which counts
upgrades as free and so is really just the scheduled demand, and B_base, which looks at capacity
but pools it across producers, ignores degradation and is not guaranteed. Task 4.7 replaced them
with one linear program, an input-output (Leontief-style) model of the whole economy: each
producer can only make the products it declares, with its own capacity; one product mix is shared
across all capacity types; recipes are followed exactly, so inputs, restorations and upgrades all
have to be made by someone; capacity wears down every cycle and only comes back through
restorations; upgrades are allowed, at most one per producer per cycle, and each costs a real
product; and an order can only be served in the three cycles before it expires. The guaranteed
version (with upgrades) is **B_lp**; the same program with upgrades switched off is
**B_lp_base**, the new base-capacity ceiling.

The question was whether this sharper ceiling changes where the screen sees room. It barely does.
B_lp never falls below what any policy actually delivered (0 of 7,776 rows), so it is sound. But on
601 of 648 world-episodes it still equals the scheduled demand, and it is never below 91% of it:
one paid-for upgrade per cycle is cheap enough on these worlds (e.g. +23 capacity on a base of 67
for one unit of a 4-capacity product) that capacity could grow until demand binds. The ceiling
without upgrades, B_lp_base, is never above B_base and is lower on 202 world-episodes, but by a
median factor of 1.000, so room hardly moves. Read with C = max(min(B2, B_lp, B_lp_base), best
arm), one world changes class (12-0369, open → saturated: the bar fills 11% of orders and is
already at what base capacity allows) and none of the 96 selected worlds changes class. The
selection rule would pick the same 32 worlds at 42 producers and 60 of the same 64 at 12; the
other 4 swaps are only a re-ranking among worlds that stay open (room 0.59–0.64 instead of
0.65–0.69).

What this means: the bar's low fill on open worlds (about 40–45% of orders) is not explained by
the limits the program models. Capacity, its upkeep, who may make what, the recipes and the
consumer order deadline all allow far more. That does not yet say what does cause it, because the
program leaves three real limits out:
- **input lead time**: the program lets an input be used before it is made, while the simulation
  needs it in stock before assembly, bought through a replenishment order with its own TTL, across
  multi-level recipes. The screen's traces fit this: for the two arms that set the bar on almost
  every world (`round_robin`, `random_whole`, 1,944 rows), TTL expiries are 84% of the consumer
  units lost (8.17 M of 9.77 M), and 87% of those expired with the receiving producer's inputs
  short (7.09 M against 1.07 M with inputs in stock);
- **the investment fund**: it is fixed at half the emission per cycle and pays for every input
  purchase, restoration and upgrade. A funded order needs at least one token, so the fund caps the
  number of such orders per cycle; it does not cap units per order, so the program cannot use it
  as a sound limit, but it shapes which orders get through;
- **inventory limits** on input stock.

So the gap is either a limit of this kind or a matter of how the economy is run (how inputs are
stocked, how the fund is spent, how orders are routed, how little producers upgrade), and this
bound cannot tell the two apart. A time-expanded input flow in the program, or the teacher (PoC-A)
finding policies that close the gap, would discriminate. Until then the open worlds' room is room
against a bound that ignores these limits, and the bound cannot say how much of it is reachable.
EXP-0052 keeps this screen's selection and split and reads its ceiling with B_lp and B_lp_base on
its own episodes (design D12).

| check (648 world-episodes, `probes/rebound_47.py`) | result |
|---|---|
| rows above B_lp + 0.01 | 0 of 7,776 |
| B_lp above B2 / B_lp_base above B_base | 0 / 0 |
| B_lp = B2 (demand) | 601; below demand on 47 (min B_lp / demand 0.913) |
| B_lp_base < B_base | 202 (median ratio 1.000) |
| classes with the LP ceiling, 12 producers | saturated 22, hard 213, open 273, flat 4 (one open → saturated: 12-0369, not selected) |
| classes with the LP ceiling, 42 producers | unchanged (hard 68, open 58, flat 1, collapse 1) |
| selection rerun with the LP ceiling | 42: 32 of 32 the same; 12: 60 of 64 the same (dropped 12-0089, 0155, 0352, 0436, all still open, room ≥ 0.59; added 12-0012, 0025, 0196, 0336) |
| ceiling against B_lp alone (guaranteed, for comparison) | 12: 12 classes change upward (8 saturated → open, 3 saturated → hard, 1 flat → open); 42: none |
| solve time, both programs | 0.59 s mean per world-episode at 12 producers, 7.7 s at 42 (max 23 s); 1,335 s in all |

Reproduce: `../.venv/bin/python ../openspec/changes/add-m3-development-audit/probes/rebound_47.py
compute --out <jsonl>` (chunked with `--start/--count`), then `... analyse --out <jsonl>` (from
`/srv/research/tokenomics/tokenomics`, with the `lp` extra); the per-world-episode values are in
`/srv/research/tokenomics/openspec/changes/add-m3-development-audit/probes/rebound_47.jsonl`. The
screen's own files are unchanged.
