# EXP-0066 — Split-need re-test: do whole-routing `no_fit` losses need a split?

**Status: complete, valid. Not an acceptance read. Reading: `whole_suffices`** (deciding arm
MWWk: completion-corrected split-needed units = 0.55% of filled consumer units at 12 producers,
1.07% at 42; threshold 5%). Next, as pre-registered: step 2 stays whole-only, and design D5
records why. `python -m tokenomics.eval.split_need`, run commit
`e28f1438bcb16cff49b8719511e94d4419c0847f` (task 4.6 of `add-m4-orderer-redesign`; driver task
4.5). Inputs: atlas v2 = EXP-0052 `reclassified/atlas_v2.json` (content hash `220886a7cf12…`,
file SHA-256 `7517314f297a…`), trained orderer EXP-0012 `orderer_scorer.pt` (SHA-256
`d25388bfa33e…`), reference EXP-0057 `artifacts/episodes.csv` (SHA-256 `69656f9e2034…`). 284
episode rows = 71 world-episodes (43 at 12 producers over 23 worlds, 28 at 42 over 14 worlds) ×
TS and MWWk × 2 replicates, horizon 100, `entity_planner` producer, routing tracer plus the
split-need observer. 0 contract failures; 284 / 284 rows identical to their EXP-0057 rows; in
every row the classes add up to the traced `no_fit` loss. Fresh 8 vCPU secure pod, ≈ 1.7
vCPU-h billed ([Details](#details)).

## In plain English

**The question (from the user).** Step 2 of the M4 ladder builds an orderer that sends every
order *whole* to one producer, drawn from outcome-trained scores. It can never split an order
across producers. The user asked: can a whole-only orderer cope with orders that are too big for
any single producer? If many orders need a split to be filled, a whole-only step 2 has a ceiling
built in, and a split rule would have to be designed before training (group 6).

**What gets lost.** Under whole routing an order goes to one producer as one sub-order, and that
producer's auction takes it only if the whole bundle fits its remaining capacity. When it does
not, the bid is rejected with reason **`no_fit`** (the bid was larger than the chosen producer's
remaining capacity). If the order then ends unfulfilled and its last bid was rejected `no_fit`,
its units count as **lost to `no_fit`**. In EXP-0057, per episode, TS lost about 252 (12
producers) / 901 (42) consumer units this way and MWWk 306 / 847, against 88 / 98 for the
splitting trained orderer.

**Three classes.** Each lost unit is classified by the state of the producers right after the
auction where its order's last bid was rejected (what capacity was left over once every
producer's auction had run):
- **choice-fixable**: another eligible producer (one that declares the product and is not in an
  outage) had enough leftover capacity of every type in the recipe to take the whole sub-order.
  A better *whole* choice would have saved it.
- **split-needed**: no single eligible producer could take it whole, but all eligible producers'
  leftovers together could, type by type. Only a split would have saved it.
- **unfillable**: not even all eligible producers together had the capacity. Neither a better
  choice nor a split would have saved it in that cycle.

**Two ways to count.** Several orders can be lost in the same cycle, and they compete for the
same leftover capacity.
- The **joint count** (the yardstick, used by the rule) hands the leftovers out once: the
  cycle's lost bids go in descending price order, each takes the first eligible producer it fits
  whole (choice-fixable), else a split of the remaining pool if that holds it (split-needed),
  else it is unfillable, and whatever it takes is gone for the bids after it.
- The **per-order count** (L, reported only) lets every lost order see the full leftovers, as if
  it were the only one.

The joint count is much stricter, and on overloaded worlds the gap is large. On 12-0352 (the
most overloaded 12-producer world, ρ = 4.49), MWWk replicate 0, first episode: joint
**choice-fixable 398 / split-needed 103 / unfillable 3,072** units against per-order **1,935 /
865 / 773**. Per order, the same few spare units are counted again for every order that could
have used them; jointly they go to one order, so most of the loss becomes unfillable. Pooled,
joint split-needed units are about half the per-order ones (12: 3,515 vs 6,260; 42: 11,353 vs
18,987, MWWk). The joint count is conservative for choice-fixable and split-needed *together*,
not for split-needed alone: an order the per-order count calls choice-fixable can turn
split-needed once a dearer order has taken that producer's leftover.

**Completion correction.** A bid that is accepted is not yet a filled order: a composite product
still needs its input products, and many accepted bids later fail (inputs not delivered, TTL
expiry). So a rescued bid would only sometimes have become filled units. The rule scales each
run's joint split-needed units by that run's **completion** = completed / accepted consumer bid
units = (units + wasted) / (units + wasted + failed dropped / ttl / ttl_outage). Here completion
is low: pooled 0.28 at 12 and 0.34 at 42 for MWWk. This is a crude scaling, not a counterfactual
(a rescued bid would also change what happens afterwards).

**Why a re-test.** Task 4.4 ran the same classification on a small pre-registered sample (three
shocked 12-producer worlds and one 42-producer world, one episode each). Its rule fired, but
that sample held only 13.7% (12) / 0.1% (42) of TS's `no_fit` loss in EXP-0057: the loss sits
on a few heavy worlds outside it, and the 42-producer world lost almost nothing. The answer
depended on which worlds were picked. The user chose a fair re-test: every EXP-0057 world-episode,
both whole arms, both replicates, with the reading fixed before the run.

**What was measured.** All 71 world-episodes of EXP-0057 were replayed with the two whole
arms, with the same random draws as EXP-0057, plus a read-only observer:
- **MWWk** (`mww_keyed`): one producer drawn per order with weights ∝ `makeable_qty_now`
  (units the producer could make now from capacity and inputs in stock), order whole. The
  better whole chooser in EXP-0057, so it decides.
- **TS** (`trained_sampled_whole`): one producer drawn per order from softmax(EXP-0012 score),
  order whole. Reported beside MWWk.

**Pre-registered reading (task 4.5, fixed before the run).** For each size, pool the deciding
arm's completion-corrected split-needed units over all its world-episodes and replicates and
divide by its pooled filled consumer units. `split_needed` if that share is ≥ 5% at 12 or at 42
(the supervisor then brings a rule-gated split design to the user before group 6); else
`whole_suffices` (step 2 stays whole-only; design D5 records why). Per-world shares, TS, the
per-order count and the uncorrected shares are reported and never decide.

**What happened.**

1. **`whole_suffices`, by a wide margin.** MWWk's completion-corrected split-needed share is
   **0.55%** at 12 producers (1,000 of 180,255 filled units) and **1.07%** at 42 (3,546 of
   331,788). TS reads 0.35% / 0.65%. Both sizes are far below 5%.
2. **The reading does not depend on the crude correction.** Without it, MWWk's joint
   split-needed units are 1.95% (12) / 3.42% (42) of filled units, still below 5%. That
   uncorrected share is the ceiling under the yardstick: completion is at most 1, so even if
   every rescued unit had completed, the corrected share could not exceed it. Only the
   loosest combination, the per-order count with no correction, crosses 5%, at 42 producers
   (MWWk 5.72%, TS 7.68%; at 12: 3.47% / 2.84%). With the correction the per-order count
   reads 0.80% / 1.65%.
3. **Most of the `no_fit` loss needs neither a split nor a better choice in that cycle.**
   MWWk loses 14.6% (12) / 14.3% (42) of its filled units to `no_fit`. Joint classes of those
   lost units: at 12, choice-fixable 26%, split-needed 13%, unfillable 60% (6,910 / 3,515 /
   15,870 of 26,295); at 42, 32% / 24% / 44% (15,092 / 11,353 / 20,985 of 47,430). Most of
   the capacity the lost orders needed was not there at all. Where something could have been
   saved, a better whole choice could have saved more units than a split (choice-fixable >
   split-needed at both sizes, both arms).
4. **Per world (reported, never deciding)** the picture is concentrated: one world per size is
   above 5% after the correction, under both arms: **12-0230** (dense, ρ 0.60: MWWk 16.0%, TS
   13.9%) and **42-0076** (`sparse_f14`, ρ 1.24: MWWk 13.4%, TS 15.7%). Uncorrected, the joint
   count is ≥ 5% for MWWk on 4 of 23 worlds at 12 (12-0111 10.0%, 12-0230 43.2%, 12-0321 5.1%,
   12-0352 6.8%) and 3 of 14 at 42 (42-0040 7.3%, 42-0076 43.0%, 42-0081 5.5%). The two
   outliers are the same two worlds that stayed ≥ 5% after the correction in task 4.4's context
   replays (TS replicate 0, first episode: 12-0230 18.7%, 42-0076 14.5%). Full table in
   [Per-world shares](#per-world-shares).
5. **TS beside MWWk.** The arms agree on the reading and on the two outlier worlds. They differ
   on single worlds: on 12-0111 MWWk's loss is mostly split-needed (1,197 of 2,296) while TS
   has none split-needed (1,452 choice-fixable, 981 unfillable); on 42-0076 TS starves (1,995
   filled units against MWWk's 12,761, completion 0.05), so its uncorrected shares there exceed
   100% (small denominator, not double counting).
6. **Producer stream (reported).** The producers' own purchases (input products and
   investments) also lose units to `no_fit`; they are not consumer units and the rule ignores
   them. MWWk joint: 12 producers 3,755 / 4,241 / 20,175 (choice / split / unfillable), 42
   producers 12,655 / 17,739 / 89,837; TS 4,991 / 3,866 / 16,035 and 24,101 / 21,588 /
   193,413. As for consumers, most is unfillable.

**What changes because of it.** The pre-registered consequence applies: **step 2 stays
whole-only**, and design D5 records why (pooled over every EXP-0057 world-episode, orders that
only a split could have saved are about 1% of filled units or less at both sizes, and under 3.5%
even without the completion correction). The result does not say splitting is never useful: on
two worlds (12-0230, 42-0076) split-needed losses are a large share of that world's fills under
both arms, and at 42 producers the loosest count (per order, uncorrected) crosses 5%. Those are
recorded here; under the rule they do not decide.

## Details

### Method

- Sample: `decoder_check.select_views`, the EXP-0057 sample — atlas v2 train split, ρ > 0.28,
  `confirm_class` ≠ `collapse`: 23 worlds at 12 producers, 14 at 42; every confirm episode seed
  (2 per world; 12-0005, 12-0155, 12-0176 have 1): 43 + 28 = 71 world-episodes.
- Arms: TS and MWWk, R = 2 each (EXP-0057's subjects and replicate seeds); each job sets its
  draw key (`set_draw_key`), so both arms use EXP-0057's exact draws. 284 jobs, run longest-first
  on 8 workers.
- Job path: EXP-0057's (`joint_stack_env` → draw key → `RoutingTracer` → `ContractMonitor` →
  policies) plus `SplitNeedObserver`, which wraps each producer's `produce` and
  `System.complete_cycle` read-only (in `eval/`, never in `domain/`). The observer replays every
  auction from the bids and capacity it saw and must agree with the domain
  (`sn_replay_mismatch` = 0).
- Classification: each consumer / producer unit lost to `no_fit` goes to the class of its
  order's last `no_fit` bid at that bid's auction. Eligible receivers = producers declaring the
  product, outage producers excluded. Counts: **joint** (the yardstick: one cycle's lost bids
  by descending price claim the leftovers in turn; choice = the first other eligible producer,
  in declaration order, whose remaining leftover holds the whole sub-order; a split takes each
  capacity type greedily), **L** (per order, full leftovers pooled type by type), **P** (per
  order, price insertion: at each producer the capacity held by accepted bids cheaper than the
  lost bid counts as available), **units** (per order, split-needed only if the whole units
  each producer could make from its leftover add up to the quantity). All four are reported in
  `split_need.csv` / `reading.json`; only joint enters the rule.
- Reading: deciding arm MWWk; per size Σ (joint consumer split-needed units × the row's
  completion) / Σ filled consumer units over the size's MWWk rows; ≥ 5% at either size →
  `split_needed`. Exact fractions (`Fraction`), so 5% is an exact edge.
- Validity: contract failures, missing rows, digest differences, any row whose classes do not
  sum to the traced `no_fit` loss or whose self-check counters are non-zero
  (`split_need_consistency`), or any row that differs from its EXP-0057 row on the 79 compared
  cells (`exp0057_neutrality`) would make the run invalid (exit 2). None occurred.

### Readings (pooled, consumer stream)

| size | arm | rows | filled units | lost `no_fit` | joint choice / split / unfillable | completion (pooled) | **corrected split share** | joint split share | L split share | L × completion |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | **MWWk** | 86 | 180,255 | 26,295 (14.6%) | 6,910 / 3,515 / 15,870 | 0.282 | **0.55%** | 1.95% | 3.47% | 0.80% |
| 12 | TS | 86 | 167,854 | 21,712 (12.9%) | 4,702 / 1,911 / 15,099 | 0.259 | 0.35% | 1.14% | 2.84% | 0.57% |
| 42 | **MWWk** | 56 | 331,788 | 47,430 (14.3%) | 15,092 / 11,353 / 20,985 | 0.342 | **1.07%** | 3.42% | 5.72% | 1.65% |
| 42 | TS | 56 | 257,460 | 50,446 (19.6%) | 16,027 / 10,935 / 23,484 | 0.269 | 0.65% | 4.25% | 7.68% | 1.16% |

Corrected split share = Σ (joint split-needed × the row's completion) / Σ filled; the
correction is per run, never a pooled completion. P and units counts (split share): MWWk 12
3.02% / 3.16%, 42 5.49% / 5.39%; TS 12 2.58% / 2.34%, 42 7.40% / 7.31% (`reading.json`).
L × completion is a sensitivity computed for this record from the rows (`artifacts/episodes.csv`),
not a driver output.

### Per-world shares

Per world × arm, pooled over its episodes and replicates (`split_need_worlds.csv`). Reported,
never deciding. "Corrected" = Σ joint split × completion / Σ filled; "joint" and "L" =
uncorrected split-needed shares.

| world | choice | ρ | MWWk filled | MWWk lost `no_fit` | MWWk joint c / s / u | MWWk completion | MWWk corrected | MWWk joint | MWWk L | TS filled | TS corrected | TS joint | TS L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12-0005 | dense | 0.51 | 1858 | 0 | 0 / 0 / 0 | 79.9% | 0.0% | 0.0% | 0.0% | 1870 | 0.0% | 0.0% | 0.0% |
| 12-0060 | sparse_f4 | 0.33 | 10402 | 144 | 144 / 0 / 0 | 57.4% | 0.0% | 0.0% | 0.0% | 9772 | 0.0% | 0.0% | 0.0% |
| 12-0094 | sparse_f4 | 0.34 | 2897 | 7 | 7 / 0 / 0 | 83.9% | 0.0% | 0.0% | 0.0% | 2959 | 0.0% | 0.0% | 0.0% |
| 12-0111 | sparse_f2 | 0.47 | 11956 | 2296 | 511 / 1197 / 588 | 23.3% | 2.3% | 10.0% | 10.0% | 11510 | 0.0% | 0.0% | 0.0% |
| 12-0121 | sparse_f4 | 2.24 | 17628 | 1153 | 1119 / 17 / 17 | 24.1% | 0.0% | 0.1% | 0.2% | 14257 | 0.0% | 0.0% | 0.1% |
| 12-0139 | sparse_f4 | 0.46 | 2787 | 0 | 0 / 0 / 0 | 70.5% | 0.0% | 0.0% | 0.0% | 2788 | 0.0% | 0.0% | 0.0% |
| 12-0155 | sparse_f2 | 0.80 | 1435 | 395 | 100 / 70 / 225 | 35.7% | 1.7% | 4.9% | 6.3% | 1305 | 1.9% | 5.4% | 6.9% |
| 12-0176 | sparse_f4 | 0.30 | 3800 | 0 | 0 / 0 / 0 | 58.8% | 0.0% | 0.0% | 0.0% | 3799 | 0.0% | 0.0% | 0.0% |
| **12-0230** | dense | 0.60 | 3454 | 2549 | 238 / 1492 / 819 | 36.9% | **16.0%** | 43.2% | 42.3% | 3313 | **13.9%** | 38.4% | 41.6% |
| 12-0254 | sparse_f2 | 2.27 | 14114 | 147 | 22 / 0 / 125 | 17.1% | 0.0% | 0.0% | 0.1% | 12460 | 0.0% | 0.0% | 0.0% |
| 12-0302 | dense | 1.12 | 13928 | 297 | 207 / 0 / 90 | 55.3% | 0.0% | 0.0% | 0.3% | 12466 | 0.0% | 0.0% | 0.0% |
| 12-0308 | sparse_f2 | 1.05 | 10967 | 443 | 47 / 49 / 347 | 31.2% | 0.1% | 0.4% | 1.4% | 10551 | 0.2% | 0.6% | 0.9% |
| 12-0321 | sparse_f2 | 1.52 | 2393 | 1504 | 555 / 122 / 827 | 21.7% | 1.1% | 5.1% | 8.5% | 2624 | 0.6% | 2.9% | 4.7% |
| 12-0340 | dense | 0.32 | 7128 | 37 | 37 / 0 / 0 | 52.7% | 0.0% | 0.0% | 0.0% | 6960 | 0.0% | 0.0% | 0.0% |
| 12-0344 | sparse_f2 | 0.60 | 12613 | 469 | 291 / 40 / 138 | 29.0% | 0.1% | 0.3% | 0.3% | 11909 | 0.0% | 0.0% | 0.0% |
| 12-0352 | dense | 4.49 | 4726 | 11139 | 1653 / 322 / 9164 | 13.4% | 0.9% | 6.8% | 53.9% | 4401 | 0.7% | 6.7% | 62.6% |
| 12-0361 | dense | 0.49 | 4157 | 0 | 0 / 0 / 0 | 77.0% | 0.0% | 0.0% | 0.0% | 4214 | 0.0% | 0.0% | 0.0% |
| 12-0365 | dense | 1.20 | 5412 | 62 | 21 / 0 / 41 | 52.5% | 0.0% | 0.0% | 0.6% | 5156 | 0.1% | 0.1% | 0.6% |
| 12-0391 | sparse_f2 | 2.38 | 9829 | 4600 | 1624 / 59 / 2917 | 16.1% | 0.1% | 0.6% | 1.9% | 9177 | 0.0% | 0.3% | 1.5% |
| 12-0392 | sparse_f4 | 1.01 | 13929 | 31 | 0 / 0 / 31 | 28.9% | 0.0% | 0.0% | 0.1% | 13281 | 0.0% | 0.0% | 0.0% |
| 12-0408 | dense | 1.10 | 14753 | 69 | 25 / 17 / 27 | 26.8% | 0.0% | 0.1% | 0.2% | 13367 | 0.0% | 0.1% | 0.0% |
| 12-0436 | sparse_f2 | 1.67 | 5040 | 848 | 309 / 88 / 451 | 26.9% | 0.5% | 1.7% | 3.1% | 4702 | 0.3% | 1.3% | 2.3% |
| 12-0483 | dense | 1.83 | 5049 | 105 | 0 / 42 / 63 | 22.5% | 0.2% | 0.8% | 1.1% | 5013 | 0.1% | 0.5% | 0.7% |
| 42-0000 | sparse_f7 | 0.57 | 14513 | 53 | 53 / 0 / 0 | 51.5% | 0.0% | 0.0% | 0.0% | 12396 | 0.1% | 0.1% | 0.0% |
| 42-0013 | sparse_f4 | 0.57 | 57360 | 7423 | 3679 / 2172 / 1572 | 24.3% | 0.9% | 3.8% | 4.6% | 47331 | 0.7% | 3.2% | 4.1% |
| 42-0031 | sparse_f14 | 0.45 | 22786 | 18 | 18 / 0 / 0 | 58.5% | 0.0% | 0.0% | 0.0% | 18765 | 0.0% | 0.0% | 0.0% |
| 42-0034 | sparse_f7 | 0.33 | 13321 | 155 | 95 / 60 / 0 | 54.4% | 0.2% | 0.5% | 0.3% | 12574 | 0.2% | 0.5% | 0.2% |
| 42-0037 | sparse_f4 | 0.29 | 9013 | 20 | 16 / 4 / 0 | 78.8% | 0.0% | 0.0% | 0.0% | 9087 | 0.0% | 0.0% | 0.0% |
| 42-0039 | sparse_f4 | 1.38 | 64595 | 0 | 0 / 0 / 0 | 24.2% | 0.0% | 0.0% | 0.0% | 53363 | 0.0% | 0.0% | 0.0% |
| 42-0040 | sparse_f2 | 0.39 | 25197 | 5046 | 1412 / 1847 / 1787 | 40.8% | 3.0% | 7.3% | 7.3% | 24715 | 2.4% | 6.0% | 7.1% |
| 42-0043 | sparse_f4 | 0.64 | 10306 | 2296 | 777 / 409 / 1110 | 39.6% | 1.6% | 4.0% | 8.1% | 10131 | 1.3% | 3.2% | 7.1% |
| **42-0076** | sparse_f14 | 1.24 | 12761 | 14085 | 3449 / 5484 / 5152 | 31.2% | **13.4%** | 43.0% | 65.0% | 1995 | **15.7%** | 307.6% | 501.2% |
| 42-0081 | sparse_f4 | 5.42 | 20279 | 16676 | 4250 / 1125 / 11301 | 21.0% | 1.2% | 5.5% | 25.8% | 18996 | 1.2% | 5.8% | 27.4% |
| 42-0093 | sparse_f14 | 0.34 | 10855 | 8 | 8 / 0 / 0 | 72.8% | 0.0% | 0.0% | 0.0% | 8589 | 0.0% | 0.0% | 0.0% |
| 42-0113 | sparse_f14 | 0.29 | 6399 | 237 | 147 / 90 / 0 | 70.9% | 1.0% | 1.4% | 1.3% | 5567 | 1.0% | 1.5% | 1.5% |
| 42-0119 | dense | 0.48 | 52934 | 167 | 167 / 0 / 0 | 63.3% | 0.0% | 0.0% | 0.0% | 30759 | 0.0% | 0.0% | 0.0% |
| 42-0126 | sparse_f14 | 0.80 | 11469 | 1246 | 1021 / 162 / 63 | 36.5% | 0.5% | 1.4% | 0.1% | 3192 | 0.5% | 5.5% | 1.9% |

World ids are `ws-atlas-v1-<id>`. Corrected share ≥ 5%: 1 of 23 worlds at 12 and 1 of 14 at 42,
under both arms. Shares above 100% (TS on 42-0076) come from a small denominator.

### Joint vs per-order on one world-episode

12-0352, episode 547143209, MWWk r0 (874 filled units; consumer `no_fit` loss 3,573):

| count | choice-fixable | split-needed | unfillable |
|---|---|---|---|
| joint | 398 | 103 | 3,072 |
| per order (L) | 1,935 | 865 | 773 |

### Validity

- `contract_failures.csv`: header only (0 rows). 284 / 284 rows, one per planned (world,
  episode, subject, replicate) key; none missing, none extra.
- Neutrality: `neutrality.csv` 284 / 284 rows `equal` on the driver's 79 compared cells (units,
  dead, digests, `shock_events`, `fulfilled_trace`, ordered / fulfilled units, every scalar
  `trace_*` cell). The observer changes nothing.
- Consistency: in every row and both streams, each count's three classes sum exactly to the
  traced `trace_lost_units_*_no_fit` and `sn_lost_total_*` equals it; `sn_replay_mismatch`,
  `sn_dup_bids`, `sn_qty_mismatch` are 0 everywhere. Traced `no_fit` totals over the run:
  consumer 145,883, producer 412,396 units.
- Independent check: `openspec/changes/add-m4-orderer-redesign/probes/exp0066_check.py`
  (stdlib only, imports nothing from the package) → ALL PASS (14 checks): the above, plus 99
  columns shared with EXP-0057's file (including the JSON series) equal on every row, the four
  per-size pooled shares and the reading recomputed from the rows equal `reading.json`, and the
  74 per-world corrected shares equal `split_need_worlds.csv` (max |Δ| 0).

### Run

- Pod `st3fj5qx2e25ut`: fresh, `--create-pod`, secure, cpu5c, 8 vCPU / 16 GB / 30 GB disk, image
  `runpod/base:0.7.0-ubuntu2004`, python3.12. Created ≈ 19:54:37 UTC 2026-10-30, SSH up
  ≈ 19:57:20, driver started 19:58:37 (`spec.json`), 284 jobs done ≈ 20:05:35, pulled, stopped
  and terminated ≈ 20:05:45. Pod life ≈ 11.1 min.
- A first attempt (pod `o07eoa1u8jaty8`, created ≈ 19:52:31) failed right after creation:
  `runpodctl pod get` did not yet find the just-created pod, and the launcher stopped and
  terminated it ≈ 19:53:54 (life ≈ 1.4 min, nothing ran). Relaunched unchanged.
- **Billed ≈ 1.7 vCPU-h** = 8 × (11.1 + 1.4) / 60 (estimate 2.47, hard stop 4.9). Tranche A
  ≈ 1.41 + 1.67 = 3.1 / 100. `runpodctl pod list` → `[]` after the run.
- Inputs staged in the git-ignored `tokenomics/_pod_ckpts/exp0066/` (SHA-256s equal the
  registered pins) and deleted after the run.

### Files

- `spec.json` (pins, provenance, the pre-registered block), `reading.json` (the reading, per
  size × arm pooled blocks, all four counts), `split_need.csv` (per row), `split_need_worlds.csv`
  (per world × arm), `neutrality.csv`, `contract_failures.csv`.
- `artifacts/episodes.csv` (git-lfs): gate rows + `trace_*` + `arm` + 57 `sn_*` cells.

### Limits

- Classes count *acceptance* at the auction, not fills; the completion correction is a scaling
  by the run's own completed / accepted ratio, not a counterfactual replay.
- The classification is per cycle and myopic: it asks whether that cycle's leftovers could have
  taken the order, not whether a different routing earlier would have left more capacity.
- The joint count's price-order greedy assignment is one tie-break among many; it is
  conservative for choice-fixable + split-needed together, not for split-needed alone.
- The sample is EXP-0057's (loaded train worlds, ρ > 0.28); worlds outside it are not covered.
