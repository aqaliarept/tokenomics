# EXP-0057 — Decoder check: is the trained orderer's split decode the bottleneck?

**Status: complete, valid. Not an acceptance read. Readings: primary `decoder_partial`
(licence: step 2, the unexplained share recorded), fidelity `faithful`, dispersion
`history_flat` (borderline: one world short, and the ±0.02 band changes the reading).**
`python -m tokenomics.eval.decoder_check`, run commit `5814dfa5e856d55657277f408516c6f611bc31e7`
(task 3.1 of `add-m4-orderer-redesign`, design D3), on atlas v2 = EXP-0052
`reclassified/atlas_v2.json` (content hash `220886a7cf12…`, file SHA-256 `7517314f297a…`),
trained orderer EXP-0012 `orderer_scorer.pt` (SHA-256 `d25388bfa33e…`), references EXP-0052
`episodes.csv` (`0e22ddd0546d…`) and EXP-0056 `episodes.csv` (`6963ee42af61…`). 639 episode rows
= 71 world-episodes (43 at 12 producers, 28 at 42) × 9 arm-replicates, horizon 100,
`entity_planner` producer, routing tracer on. 0 contract failures; 293 / 293 reference rows
reproduced; 4 deaths, all in the TW arm. Fresh 8 vCPU secure pod, ≈ 1.4 vCPU-h billed
([Details](#details)).

## In plain English

**The question.** The trained orderer (the EXP-0012 scorer, played here with the hand-written
producer `entity_planner`) delivers fewer consumer units than the plain rule `round_robin`
(RR), which sends every order whole to the next eligible producer in turn. The trained orderer
turns its scores into a *split*: every order is divided across producers in proportion to
softmax(score). EXP-0056 showed on three worlds that removing splitting alone, or changing the
receivers alone, does not close the gap; only both together do. This experiment asks whether
the loss sits in that *decode* (how scores become routing) or in the *scores* themselves: if
we keep the scores but decode them differently — draw one producer per order from
softmax(score) and send the order whole — how much of the gap to `round_robin` comes back?

**The arms** (all with `entity_planner`; "whole" = the order goes to one producer, unsplit):
- **TR** — the trained orderer as deployed (softmax split).
- **TS** (`trained_sampled_whole`) — the same scores, one producer drawn per order from
  softmax(score), order routed whole. Two replicates (R = 2).
- **MWWk** (`mww_keyed`) — the scorer's *target* as a router: one producer drawn per order with
  weights ∝ `makeable_qty_now` (how many units the producer could make right now from capacity
  and inputs in stock), order whole. It uses the same random draws as TS ("keyed": the draw is a
  hash of world, episode, replicate, agent, cycle and slot), so TS − MWWk compares scores
  against target with the dice held fixed. R = 2.
- **MWW** — `makeable_weighted_whole`, the same rule as MWWk with the atlas's own random stream;
  it is here to reproduce EXP-0052. R = 2.
- **TW** (`trained_whole`) — the same scores, each order whole to the top-scored producer
  (no sampling).
- **RR** — `round_robin`.

**The worlds.** Every atlas v2 *train* world with tightness ρ > 0.28 (ρ = demand against
starting throughput) whose class is not `collapse`: 23 worlds at 12 producers and 14 at 42,
on the atlas confirm's episode seeds (2 per world, 1 on three cyclic shock-free worlds). Only
the 12-producer worlds decide the readings; the 42-producer table is reported.

**Terms.**
- **Gap G_w** on a world = RR_w − TR_w, the units `round_robin` gains over the trained orderer
  (world means: per episode, the mean over replicates; then the mean over episodes).
- **Floor** of a comparison on a world = the larger of the atlas's noise floor for that world
  and 3 × the standard error of this run's own paired per-episode differences (the own SE only
  when there are ≥ 2 paired observations). A difference counts only above its floor; each
  comparison has its own floor.
- **Gap world** = a world with G_w above its floor.
- **Recovered share rec_w** = (TS_w − TR_w) / G_w, the fraction of the gap that the whole-order
  sampled decode wins back (1 = all of it; above 1 = more than all).

**The pre-registered readings (design D3, fixed before the run).**
- *Primary*, first match wins: fewer than 5 gap worlds → `no_gap`; rec_w ≥ 2/3 on at least
  2/3 of gap worlds → `decoder`; median rec_w ≥ 1/3 → `decoder_partial`; else `not_decoder`.
  `decoder` and `decoder_partial` license step 2 (the choice decode, group 4); the other two
  mean "diagnose first".
- *Fidelity* — do the scores carry their target? TS and MWWk are *equivalent* on a world when
  |TS_w − MWWk_w| is within the floor. Equivalent on ≥ 2/3 of worlds → `faithful`; else
  `fidelity_loss` / `beyond_target` when TS is below / above MWWk beyond the floor on ≥ 1/3
  (both or neither → `mixed`). Never grants or withholds the licence.
- *Dispersion* — tests design D5's premise that each ordering agent's own history differs
  from the others' enough to be worth a per-agent feature. Under TS, per world:
  `agent_history_dispersion` (the across-agent spread of own completion rates) must reach 0.05
  and `routed_share_dispersion` (how differently agents spread their units over the same
  product's producers, as a total-variation distance) must reach 0.10. Both on ≥ 2/3 of the
  worlds → `history_differs`; else `history_flat`, and the supervisor asks the user before
  group 4.

**What happened.**

1. **Primary: `decoder_partial` — the whole-order sampled decode wins back about 70% of the
   gap, short of the 2/3-of-worlds bar by one world.** 17 of the 23 worlds are gap worlds
   (the same 17 the prediction named). TS recovers rec_w ≥ 2/3 on 11 of them (12 needed for
   `decoder`); the median rec_w is 0.704, and pooled over the gap worlds TS wins back 11,158 of
   15,870 units (0.70). The six worlds under 2/3 sit at 0.48–0.63 (12-0408 0.48, 12-0254 0.55,
   12-0111 0.56, 12-0392 0.60, 12-0352 0.62, 12-0230 0.63), so none is a failure, they are
   partial recoveries. The licence is step 2, with the unexplained ≈ 30% recorded (below).
2. **Prediction against outcome.** Task 1.2 predicted `decoder` (17 gap worlds, 16 with
   rec_w ≥ 2/3, median 0.908) by putting MWW in TS's place, and said it flips if TS gets less
   than about 0.78 of MWW's gain over TR. MWW in this run reproduces that prediction exactly
   (16 / 17, median 0.908). TS got 0.77 of MWW's gain pooled (median per world 0.81), right on
   the flip line, and the reading came out one step lower, `decoder_partial`.
3. **Fidelity: `faithful`, leaning low.** TS is within the floor of MWWk on 17 of 23 worlds,
   below it beyond the floor on 6 (12-0060, 0121, 0254, 0302, 0344, 0391; three of them by
   less than 20 units over the floor), above it on none. On average TS trails MWWk by 136
   units, and it is above MWWk on only 5 worlds. So the scores carry their target well enough
   for the rule, but where they differ, they are worse than the target, not better. The D3
   interpretation cell: *"decode is the bottleneck; scores carry their target"*.
   Score fidelity on the states the arms visited (medians; KL target defined below): TS's
   scores rank producers like log `makeable_qty_now` (Spearman 0.95), pick the same top
   producer on 78% of decisions, KL 0.10 (TR 0.95 / 83% / 0.08; TW 0.95 / 87% / 0.08).
4. **Where the rest of the gap goes (the unexplained share).** On the 12-producer gap worlds,
   relative to the gap: TS recovers 0.70, MWWk 0.89, MWW 0.91, TW 0.29. So of the ≈ 0.30 TS
   leaves, ≈ 0.19 is the scores falling short of their own target (MWWk − TS) and ≈ 0.11 is
   what `round_robin` still gains over the target rule (RR − MWWk; MWWk is at or above RR on 7
   of the 17 gap worlds).
5. **Sampling, not just going whole, is what helps.** TW (whole, argmax) recovers only 0.29 of
   the gap and is below TS on every world (TS − TW +334 units on average, beyond its floor on
   13 of 23, never below). TW also causes all 4 deaths of the run (12-0436, 42-0000, 42-0031,
   42-0043). With argmax, every concurrent order of a product goes to one producer (concurrent
   top share 0.99, the share of a product's concurrent orders whose lead receiver is the
   single most-used one); TS lowers it to 0.84, MWWk 0.73, RR 0.63, TR 0.98.
6. **Dispersion: `history_flat`, borderline.** Agents' completion histories differ everywhere
   (history ≥ 0.05 on 23 / 23). The routed-share measure passes on 15 of 23 worlds; 16 were
   needed. Five worlds sit within ±0.02 of a line (12-0005 and 12-0060 just pass, 12-0111,
   12-0155 and 12-0344 just fail), and moving them changes the reading. Under MWW, MWWk or RR
   the same rule passes 16 / 23 and would read `history_differs`. The three alternative
   share measures named in the 1.3 review do *not* rescue it: producer streams 14 / 23, each
   agent vs the pooled rest 11 / 23, unit-weighted 10 / 23 (all `history_flat`). Pre-registered
   rule stands: `history_flat`, borderline. Table in [Dispersion](#dispersion).
7. **42 producers (reported, never a reading).** Primary `decoder_partial` too (11 gap
   worlds of 14, rec_w ≥ 2/3 on 7, median 0.82), but fidelity `fidelity_loss`: TS is below
   MWWk beyond the floor on 8 of 14 worlds (by 529 to 5,544 units), and pooled TS gets only
   0.59 of MWW's gain over TR, while MWWk recovers 1.08 of the gap (more than `round_robin`).
   Score fidelity is worse there (TS Spearman 0.80, KL 0.43). Dispersion would read
   `history_differs` (11 / 14).

**What changes because of it.** The licence holds: step 2 (group 4, the choice decode) is
licensed by `decoder_partial`, and the recorded unexplained share is ≈ 0.30 of the gap at 12
producers (0.19 score shortfall against the target, 0.11 target rule against `round_robin`).
Two things are flagged for the supervisor, not decided here:
- the dispersion reading is `history_flat` (borderline), which by D3 means the user is asked
  before group 4, because agent-local history would carry little contrast on about a third of
  the worlds;
- at 42 producers the scores fall clearly short of their own target, so at that size a better
  decode alone leaves a large share on the table (TS − MWWk −1,327 units on average).

## Details

### Method

- Sample: atlas v2 train split, ρ > 0.28, `confirm_class` ≠ `collapse`, sizes 12 and 42: 23 +
  14 worlds, EXP-0056's 12-0060 / 12-0111 / 12-0344 included; episodes = the confirm's seeds in
  generation order (12-0005, 12-0155, 12-0176 have one): 43 + 28 = 71 world-episodes.
- Arms and replicates: TR 1, TS 2, MWWk 2, MWW 2 (the atlas's replicate seeds), TW 1, RR 1 →
  9 jobs per world-episode, 639 jobs. TS and MWWk set the per-job draw key (`set_draw_key`).
  TR / TS / TW run the scorer inside `ScoreFidelity`, which records the score-fidelity cells.
- Floors: per comparison max(atlas `noise_floor` of the world, 3·SE of the run's own paired
  differences with ≥ 2 pairs). RR − TR pairs by episode (1–2 pairs); TS − MWWk by episode ×
  replicate (2–4 pairs); TS − TW every TS replicate against TW's single row. On 12 of 23
  12-producer worlds the own 3·SE of RR − TR exceeded the atlas floor and set it.
- **KL fidelity target**: softmax(`makeable_logits`(m)) — log `makeable_qty_now` clipped to
  ±10 (`env.spaces.ACTION_LOGIT_BOUND`), a −10 logit for a producer with nothing makeable,
  uniform when nothing is makeable (`routing_trace.makeable_logits`, the EXP-0012 scorer's
  distillation target). Not D3's plain makeable share, whose KL is infinite wherever a producer
  has m = 0 (carry-forward from the 2.1 review). Spearman is against log `makeable_qty_now`;
  top-1 ties use the first argmax.

### Primary and fidelity, per world

Units are world means (rounded); G_w = RR − TR; floor = the comparison's floor; TS − TW is
the secondary read.

#### 12 producers (the readings)

| world | cell | ρ | eps | TR | TS | MWWk | MWW | TW | RR | G_w | floor | gap | rec_w | TS − MWWk | floor | relation | TS − TW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-0005 | dense | 0.51 | 1 | 894 | 935 | 929 | 932 | 644 | 834 | -60 | 93 | no | -0.68 | 6 | 93 | equivalent | 291 |
| 12-0060 | sparse_f4 | 0.33 | 2 | 2,120 | 2,443 | 2,600 | 2,635 | 2,046 | 2,434 | 314 | 145 | yes | 1.03 | -158 | 145 | below | 396 |
| 12-0094 | sparse_f4 | 0.34 | 2 | 777 | 740 | 724 | 730 | 670 | 736 | -42 | 116 | no | 0.90 | 16 | 116 | equivalent | 70 |
| 12-0111 | sparse_f2 | 0.40 | 2 | 2,442 | 2,878 | 2,989 | 3,044 | 2,874 | 3,217 | 774 | 350 | yes | 0.56 | -112 | 243 | equivalent | 4 |
| 12-0121 | dense | 1.16 | 2 | 1,767 | 3,564 | 4,407 | 4,174 | 2,530 | 4,320 | 2,554 | 730 | yes | 0.70 | -843 | 448 | below | 1,034 |
| 12-0139 | sparse_f4 | 0.29 | 2 | 711 | 697 | 697 | 706 | 680 | 680 | -30 | 116 | no | 0.46 | 0 | 116 | equivalent | 18 |
| 12-0155 | dense | 0.37 | 1 | 545 | 652 | 718 | 735 | 515 | 710 | 165 | 219 | no | 0.65 | -65 | 219 | equivalent | 138 |
| 12-0176 | sparse_f4 | 0.30 | 1 | 1,452 | 1,900 | 1,900 | 1,970 | 1,509 | 1,814 | 362 | 87 | yes | 1.24 | -0 | 112 | equivalent | 390 |
| 12-0230 | sparse_f2 | 0.57 | 2 | 573 | 828 | 864 | 874 | 748 | 978 | 404 | 134 | yes | 0.63 | -35 | 134 | equivalent | 80 |
| 12-0254 | dense | 0.74 | 2 | 2,412 | 3,115 | 3,528 | 3,483 | 2,596 | 3,696 | 1,284 | 255 | yes | 0.55 | -414 | 352 | below | 520 |
| 12-0302 | dense | 1.66 | 2 | 2,582 | 3,116 | 3,482 | 3,543 | 1,674 | 3,348 | 766 | 214 | yes | 0.70 | -366 | 283 | below | 1,442 |
| 12-0308 | sparse_f2 | 0.82 | 2 | 2,071 | 2,638 | 2,742 | 2,727 | 2,412 | 2,746 | 676 | 160 | yes | 0.84 | -104 | 171 | equivalent | 225 |
| 12-0321 | sparse_f2 | 0.46 | 2 | 510 | 656 | 598 | 690 | 550 | 716 | 207 | 193 | yes | 0.71 | 58 | 193 | equivalent | 106 |
| 12-0340 | dense | 0.38 | 2 | 1,331 | 1,740 | 1,782 | 1,799 | 1,488 | 1,696 | 364 | 134 | yes | 1.12 | -42 | 134 | equivalent | 252 |
| 12-0344 | sparse_f2 | 0.56 | 2 | 2,364 | 2,977 | 3,153 | 3,151 | 2,848 | 3,240 | 876 | 860 | yes | 0.70 | -176 | 160 | below | 129 |
| 12-0352 | dense | 30.82 | 2 | 274 | 1,100 | 1,182 | 1,131 | 570 | 1,610 | 1,336 | 474 | yes | 0.62 | -81 | 228 | equivalent | 530 |
| 12-0361 | sparse_f4 | 0.32 | 2 | 890 | 1,054 | 1,039 | 1,028 | 822 | 952 | 62 | 212 | no | 2.66 | 14 | 149 | equivalent | 232 |
| 12-0365 | dense | 0.35 | 2 | 932 | 1,289 | 1,353 | 1,317 | 890 | 1,226 | 294 | 328 | no | 1.21 | -64 | 134 | equivalent | 400 |
| 12-0391 | sparse_f2 | 0.42 | 2 | 1,420 | 2,294 | 2,457 | 2,397 | 1,886 | 2,496 | 1,076 | 195 | yes | 0.81 | -163 | 160 | below | 408 |
| 12-0392 | sparse_f2 | 0.62 | 2 | 2,077 | 3,320 | 3,482 | 3,684 | 2,930 | 4,138 | 2,062 | 310 | yes | 0.60 | -162 | 296 | equivalent | 390 |
| 12-0408 | dense | 0.83 | 2 | 2,526 | 3,342 | 3,688 | 3,843 | 2,966 | 4,230 | 1,704 | 454 | yes | 0.48 | -346 | 462 | equivalent | 376 |
| 12-0436 | sparse_f4 | 0.31 | 2 | 779 | 1,176 | 1,260 | 1,161 | 932 | 1,126 | 348 | 193 | yes | 1.14 | -84 | 193 | equivalent | 243 |
| 12-0483 | dense | 0.37 | 2 | 482 | 1,253 | 1,262 | 1,323 | 1,244 | 1,248 | 766 | 236 | yes | 1.01 | -9 | 149 | equivalent | 9 |

Primary: 17 gap worlds; rec_w ≥ 2/3 on 11 (0.647 < 2/3); median rec_w 0.704 ≥ 1/3 →
**`decoder_partial`**. Fidelity: equivalent 17, below 6, above 0 of 23 → **`faithful`**
(loss rule needs 8 below).

#### 42 producers (reported only)

| world | cell | ρ | eps | TR | TS | MWWk | MWW | TW | RR | G_w | floor | gap | rec_w | TS − MWWk | floor | relation | TS − TW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42-0000 | sparse_f7 | 0.57 | 2 | 1,610 | 3,099 | 3,628 | 3,574 | 1,103 | 3,417 | 1,806 | 394 | yes | 0.82 | -529 | 394 | below | 1,996 |
| 42-0013 | sparse_f4 | 0.57 | 2 | 7,718 | 11,833 | 14,340 | 14,226 | 10,787 | 14,510 | 6,792 | 742 | yes | 0.61 | -2,507 | 396 | below | 1,046 |
| 42-0031 | sparse_f14 | 0.45 | 2 | 2,446 | 4,691 | 5,696 | 5,782 | 556 | 4,784 | 2,339 | 867 | yes | 0.96 | -1,005 | 350 | below | 4,136 |
| 42-0034 | sparse_f7 | 0.33 | 2 | 2,416 | 3,144 | 3,330 | 3,302 | 2,072 | 3,058 | 642 | 349 | yes | 1.13 | -187 | 349 | equivalent | 1,071 |
| 42-0037 | sparse_f4 | 0.29 | 2 | 2,262 | 2,272 | 2,253 | 2,288 | 1,786 | 2,118 | -144 | 349 | no | -0.07 | 18 | 349 | equivalent | 486 |
| 42-0039 | sparse_f4 | 1.38 | 2 | 8,456 | 13,341 | 16,149 | 15,919 | 12,221 | 16,422 | 7,966 | 2,598 | yes | 0.61 | -2,808 | 524 | below | 1,120 |
| 42-0040 | sparse_f2 | 0.39 | 2 | 5,300 | 6,179 | 6,299 | 6,261 | 6,079 | 6,314 | 1,014 | 349 | yes | 0.87 | -120 | 349 | equivalent | 100 |
| 42-0043 | sparse_f4 | 0.64 | 2 | 1,736 | 2,533 | 2,576 | 2,564 | 1,000 | 2,670 | 934 | 463 | yes | 0.85 | -44 | 349 | equivalent | 1,533 |
| 42-0076 | sparse_f14 | 1.24 | 2 | 70 | 499 | 3,190 | 2,980 | 446 | 2,370 | 2,300 | 538 | yes | 0.19 | -2,692 | 870 | below | 53 |
| 42-0081 | sparse_f4 | 5.42 | 2 | 2,344 | 4,749 | 5,070 | 5,216 | 3,282 | 5,592 | 3,249 | 1,890 | yes | 0.74 | -321 | 601 | equivalent | 1,468 |
| 42-0093 | sparse_f14 | 0.34 | 2 | 2,232 | 2,147 | 2,714 | 2,695 | 851 | 2,287 | 56 | 349 | no | -1.52 | -566 | 349 | below | 1,296 |
| 42-0113 | sparse_f14 | 0.29 | 2 | 688 | 1,392 | 1,600 | 1,607 | 886 | 1,398 | 710 | 349 | yes | 0.99 | -208 | 349 | equivalent | 505 |
| 42-0119 | dense | 0.48 | 2 | 3,712 | 7,690 | 13,234 | 13,404 | 1,452 | 11,860 | 8,148 | 1,467 | yes | 0.49 | -5,544 | 792 | below | 6,238 |
| 42-0126 | sparse_f14 | 0.80 | 2 | 70 | 798 | 2,867 | 3,095 | 1,012 | 1,684 | 1,614 | 2,168 | no | 0.45 | -2,069 | 1,025 | below | -214 |

42: 11 gap worlds, rec_w ≥ 2/3 on 7 (0.636), median 0.824 → `decoder_partial`; fidelity
equivalent 6, below 8, above 0 of 14 → `fidelity_loss` (interpretation cell: "decode is the
bottleneck; MWWk is the stronger reference (noted)"); dispersion 11 / 14 → `history_differs`
(borderline worlds 42-0039, 42-0081). Pooled over the gap worlds: TS 0.63 of the gap, MWWk
1.08, MWW 1.07, TW 0.09.

### Prediction (task 1.2) against outcome

| | prediction (MWW as TS, EXP-0052 rows) | this run, MWW | this run, TS |
|---|---|---|---|
| gap worlds | 17 / 23 | 17 / 23 | 17 / 23 |
| rec_w ≥ 2/3 | 16 / 17 (0.941) | 16 / 17 | **11 / 17 (0.647)** |
| median rec_w | 0.908 | 0.908 | **0.704** |
| reading | `decoder` | `decoder` | **`decoder_partial`** |

Flip condition stated in advance: TS below ≈ 0.78 of MWW's gain over TR. Observed: 0.772
pooled over the gap worlds, 0.811 median per world.

### Score fidelity (medians over episode rows; KL target above)

| arm | size | Spearman vs log makeable | top-1 agreement | KL to softmax(makeable_logits) | decisions |
|---|---|---|---|---|---|
| TR | 12 | 0.949 | 0.828 | 0.081 | 150,894 |
| TS | 12 | 0.949 | 0.779 | 0.104 | 277,309 |
| TW | 12 | 0.949 | 0.867 | 0.083 | 121,915 |
| TR | 42 | 0.865 | 0.788 | 0.437 | 483,367 |
| TS | 42 | 0.800 | 0.723 | 0.431 | 789,543 |
| TW | 42 | 0.949 | 0.849 | 0.074 | 200,561 |

The numbers differ by arm because each arm visits different states.

### Dispersion

Under TS (the reading's arm), 12 producers. Lines: history ≥ 0.05, share ≥ 0.10. "✗" marks a
share-type value below 0.10. The three alternatives replace the share measure (1.3 review):
*producer streams* = only the producers' own purchase streams; *vs rest* = each agent's
routed-share vector against the pooled vector of all other agents; *unit-weighted* = agent
pairs weighted by routed units. MWW and RR share values alongside.

| world | history (≥ 0.05) | share (≥ 0.10) | passes | in band | producer streams | vs rest | unit-weighted | MWW share | RR share |
|---|---|---|---|---|---|---|---|---|---|
| 12-0005 | 0.148 | 0.114 | yes | yes | 0.103 | 0.100 | 0.135 | 0.135 | 0.137 |
| 12-0060 | 0.165 | 0.107 | yes | yes | 0.095 ✗ | 0.090 ✗ | 0.119 | 0.119 | 0.191 |
| 12-0094 | 0.126 | 0.224 | yes |  | 0.226 | 0.186 | 0.140 | 0.224 | 0.341 |
| 12-0111 | 0.063 | 0.087 ✗ | no | yes | 0.082 ✗ | 0.066 ✗ | 0.086 ✗ | 0.087 | 0.104 |
| 12-0121 | 0.122 | 0.127 | yes |  | 0.142 | 0.097 ✗ | 0.108 | 0.130 | 0.097 |
| 12-0139 | 0.072 | 0.224 | yes |  | 0.227 | 0.183 | 0.194 | 0.231 | 0.343 |
| 12-0155 | 0.130 | 0.088 ✗ | no | yes | 0.091 ✗ | 0.075 ✗ | 0.026 ✗ | 0.100 | 0.093 |
| 12-0176 | 0.153 | 0.217 | yes |  | 0.217 | 0.151 | 0.099 ✗ | 0.191 | 0.305 |
| 12-0230 | 0.085 | 0.240 | yes |  | 0.238 | 0.156 | 0.123 | 0.245 | 0.229 |
| 12-0254 | 0.080 | 0.063 ✗ | no |  | 0.082 ✗ | 0.048 ✗ | 0.053 ✗ | 0.069 | 0.024 |
| 12-0302 | 0.187 | 0.188 | yes |  | 0.132 | 0.125 | 0.099 ✗ | 0.204 | 0.199 |
| 12-0308 | 0.094 | 0.073 ✗ | no |  | 0.093 ✗ | 0.062 ✗ | 0.054 ✗ | 0.082 | 0.100 |
| 12-0321 | 0.098 | 0.066 ✗ | no |  | 0.072 ✗ | 0.055 ✗ | 0.039 ✗ | 0.087 | 0.065 |
| 12-0340 | 0.107 | 0.186 | yes |  | 0.186 | 0.138 | 0.135 | 0.196 | 0.208 |
| 12-0344 | 0.103 | 0.086 ✗ | no | yes | 0.083 ✗ | 0.063 ✗ | 0.077 ✗ | 0.097 | 0.083 |
| 12-0352 | 0.094 | 0.138 | yes |  | 0.120 | 0.095 ✗ | 0.110 | 0.135 | 0.102 |
| 12-0361 | 0.145 | 0.160 | yes |  | 0.129 | 0.111 | 0.108 | 0.170 | 0.189 |
| 12-0365 | 0.126 | 0.167 | yes |  | 0.165 | 0.123 | 0.100 ✗ | 0.189 | 0.177 |
| 12-0391 | 0.070 | 0.072 ✗ | no |  | 0.081 ✗ | 0.049 ✗ | 0.067 ✗ | 0.077 | 0.064 |
| 12-0392 | 0.106 | 0.135 | yes |  | 0.149 | 0.098 ✗ | 0.082 ✗ | 0.151 | 0.138 |
| 12-0408 | 0.148 | 0.170 | yes |  | 0.128 | 0.115 | 0.113 | 0.183 | 0.179 |
| 12-0436 | 0.074 | 0.071 ✗ | no |  | 0.064 ✗ | 0.050 ✗ | 0.041 ✗ | 0.062 | 0.055 |
| 12-0483 | 0.110 | 0.166 | yes |  | 0.133 | 0.118 | 0.059 ✗ | 0.168 | 0.163 |

Side by side (worlds passing both lines, of 23; 16 needed):

| definition | passing | reading |
|---|---|---|
| **pre-registered** (history + routed share, TS) | **15** | **`history_flat`** (borderline: 5 worlds in band, the band changes the reading) |
| alternative: producer streams | 14 | `history_flat` |
| alternative: vs rest | 11 | `history_flat` |
| alternative: unit-weighted | 10 | `history_flat` |
| pre-registered rule, MWW / MWWk / RR rows (context) | 16 / 16 / 16 | would be `history_differs` |
| pre-registered rule, TR / TW rows (context) | 10 / 6 | `history_flat` |

No world was undefined (count columns > 0 everywhere; 0 skipped). The history measure passes
on all 23 worlds; the reading turns entirely on the share measure. The 1.3 trace expected the
alternatives to fire on ≥ 2 of its 3 worlds under MWW (one episode); under TS over 2 episodes
on those worlds they read 0.095 / 0.082 / 0.083 (producer streams) and 0.090 / 0.066 / 0.063
(vs rest), all below 0.10. D5's own EMA history features (1.3: per-cycle TV 0.23–0.49) are not
measured here.

### Concentration, herding and composition (`routing_metrics.csv`, episode-weighted)

| size | arm | units | split share (consumer) | wasted units (consumer) | receiver change (consumer) | lead concurrent top share (consumer) | share swing (consumer) | concurrent cross-agent share | orders / turn (consumer / producer) | displaced bid units (consumer) | herding HHI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | TR | 1,394 | 0.51 | 389 | 0.61 | 0.98 | 0.41 | 0.975 | 1.10 / 1.23 | 2,691 | 0.76 |
| 12 | TS | 1,906 | 0 | 0 | 0.61 | 0.84 | 0.57 | 0.966 | 1.09 / 1.21 | 2,512 | 0.75 |
| 12 | MWWk | 2,042 | 0 | 0 | 0.62 | 0.73 | 0.55 | 0.968 | 1.09 / 1.20 | 2,426 | 0.66 |
| 12 | MWW | 2,051 | 0 | 0 | 0.62 | 0.73 | 0.55 | 0.967 | 1.09 / 1.19 | 2,350 | 0.66 |
| 12 | TW | 1,571 | 0 | 0 | 0.35 | 0.99 | 0.34 | 0.910 | 1.12 / 1.20 | 4,221 | 0.98 |
| 12 | RR | 2,100 | 0 | 0 | 0.82 | 0.63 | 0.67 | 0.970 | 1.07 / 1.19 | 1,602 | 0.66 |
| 42 | TR | 2,933 | 0.53 | 959 | 0.74 | 0.98 | 0.48 | 0.974 | 1.15 / 1.27 | 3,825 | 0.75 |
| 42 | TS | 4,598 | 0 | 0 | 0.74 | 0.77 | 0.72 | 0.966 | 1.11 / 1.23 | 3,250 | 0.72 |
| 42 | MWWk | 5,925 | 0 | 0 | 0.78 | 0.62 | 0.71 | 0.962 | 1.09 / 1.18 | 2,255 | 0.50 |
| 42 | MWW | 5,922 | 0 | 0 | 0.78 | 0.62 | 0.70 | 0.962 | 1.09 / 1.18 | 2,291 | 0.50 |
| 42 | TW | 3,109 | 0 | 0 | 0.24 | 0.99 | 0.25 | 0.858 | 1.22 / 1.18 | 8,956 | 0.98 |
| 42 | RR | 5,606 | 0 | 0 | 0.93 | 0.50 | 0.78 | 0.959 | 1.10 / 1.18 | 1,463 | 0.43 |

Concurrent same-product orders come from different agents in 96–97% of cases (cross-agent
share) under every arm but TW, as task 1.3 found: one agent's turn rarely holds more than one
order (≈ 1.1 per turn). Sampling from the scores spreads concurrent orders less than sampling
from the target (TS 0.84 vs MWWk 0.73 at 12; 0.77 vs 0.62 at 42): the scores are sharper than
their target where it matters. Descriptive, not a causal test.

### Secondary reads (never gating)

TS − TW per size × choice cell (mean units, worlds beyond the floor / worlds; gap-world median
rec_w): 12-dense +401 (6 / 9; 0.66), 12-f2 +222 (3 / 8; 0.71), 12-f4 +383 (4 / 6; 0.87); 42-f14
+1,155 (3 / 5; 0.96), 42-f4 +1,130 (5 / 5; 0.68), 42-f7 +1,534 (2 / 2; 0.98), 42-f2 +100 (0 / 1;
0.87), 42-dense +6,238 (1 / 1; 0.49). Per ρ quartile (12): q1 +188 (rec 1.08), q2 +210 (0.63),
q3 +465 (0.71), q4 +500 (0.70). Full rows in `decoder_check.csv` (`scope` = `cell` /
`rho_quartile`).

### Validity and verification

- `contract_failures.csv` header-only (0 rows); `spec.json` `status: complete`, `result.rows`
  639, `reproduced` 293 / `compared` 293.
- Independent check `openspec/changes/add-m4-orderer-redesign/probes/exp0057_check.py`
  (stdlib only, imports nothing from the package; re-derives the sample from the atlas, the
  confirm seeds from EXP-0052's rows, the job grid, every comparison, floor and reading from
  raw rows): **ALL PASS** — 23 + 14 worlds, 43 + 28 world-episodes; 639 rows = the planned grid,
  each once; demand + shock digests equal across the 9 arm-replicates on all 71
  world-episodes and equal to EXP-0052's; `reproduction.csv` 293 rows (EXP-0052 TR 71, RR 71,
  MWW 142; EXP-0056 TR / TW / RR 3 each), all equal on units, death and both digests; its own
  comparison of the 284 TR / RR / MWW rows with EXP-0052 equal; the primary, fidelity and
  dispersion readings (and the three alternatives, the borderline worlds and
  `borderline_changes_reading`) recomputed from raw rows equal `reading.json` at 12 and 42;
  `reading.json` recomputed from `decoder_check.csv`, `fidelity.csv` and `dispersion.csv` alone
  equals it; per-world G_w, floors, rec_w and dispersion values in the CSVs equal the raw
  recomputation.
- Staged inputs: SHA-256 `orderer_scorer.pt` `d25388bf…a8c`, `atlas_v2.json` `7517314f…568`
  (content hash `220886a7…a23`), `exp0052_episodes.csv` `0e22ddd0…b12`, `exp0056_episodes.csv`
  `6963ee42…62a` — equal to the pins in `decoder_check.py` and to `spec.json` `pins`.
- Source: commit `5814dfa5e856d55657277f408516c6f611bc31e7`, tree SHA-256 `bdc3616b…`
  (`spec.json` `source`; `--verify-source-commit`).
- Deaths (not contract failures): 4, all TW: 12-0436, 42-0000, 42-0031, 42-0043 (one episode
  each).

### Compute

`python3 run_pod_training.py --create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python
python3.12 --vcpu 8 --mem 16 --disk 30 --pod-name exp0057-decoder-check --verify-source-commit
--train "-u -m tokenomics.eval.decoder_check --atlas …/atlas_v2.json --orderer-ckpt
…/orderer_scorer.pt --exp0052-episodes …/exp0052_episodes.csv --exp0056-episodes
…/exp0056_episodes.csv --workers 8 --source-commit {SOURCE_COMMIT} --out-dir
/workspace/tokenomics/runs/exp0057" --pull …`. Pod `m047272s6plghz`, secure cloud, cpu5c,
8 vCPU, 16 GB; created 2026-10-30 ≈ 15:01:07 UTC, terminated ≈ 15:11:40 UTC (pod life ≈ 10.6
min); driver ≈ 15:01:49–15:11:33 (pod clock), 639 jobs in ≈ 8.7 min. **Billed ≈ 1.4 vCPU-h**
(8 × 10.6 min), against the estimate 3.5 and the hard stop 7.0 (52 min). Tranche A: ≈ 1.4 of
100 vCPU-h. Faster than sized (sizing.md §EXP-0057: ≈ 16 min driver wall): setup took under a
minute, not the assumed 10.

### Files

`spec.json`, `reading.json`, `decoder_check.csv`, `fidelity.csv`, `dispersion.csv`,
`reproduction.csv`, `routing_metrics.csv`, `contract_failures.csv`; `artifacts/episodes.csv`
(git-lfs; 639 rows, gate columns + `trace_*` + `arm` + `score_*`).

## Conclusions

- **Is the fixed decoder the bottleneck? Mostly.** Keeping the scores and only changing how
  they become routing (one producer sampled per order, order whole) wins back ≈ 70% of the
  trained orderer's loss to `round_robin` on the 12-producer worlds: `decoder_partial`, one world
  short of `decoder`. Step 2 (the choice decode) is licensed; the unexplained ≈ 30% splits into
  ≈ 19 points of scores falling short of their target and ≈ 11 points of `round_robin` beating
  the target rule.
- **Do the scores carry their target? Yes at 12 producers (`faithful`), with a one-sided
  shortfall**: TS is never above MWWk beyond the floor and is below it on 6 of 23. At 42
  producers they do not (`fidelity_loss`, 8 of 14 below, reported only).
- **Sampling matters, not only going whole**: the argmax whole decode (TW) recovers 0.29 and
  causes every death in the run.
- **The task 1.2 prediction (`decoder`) missed by one step**, exactly at its stated flip
  condition (TS at 0.77 of MWW's gain against the 0.78 line).
- **Dispersion: `history_flat`, borderline** under the pre-registered definition (15 / 23,
  16 needed; the ±0.02 band changes it) and under all three alternatives (14 / 11 / 10). Per
  D3, the supervisor asks the user before group 4.
