# EXP-0115: why A1 acceptance is harmful (add-m6-joint-mpg-gnn task 4.1)

**Question.** EXP-0098 found A1 (`ttl_feasible`: also accept a bid that fits the next cycle after the queued
claims) `harmful` at 12 producers. Its offered mechanism, claims dying with their order, was refuted. What costs
the units? All arms are diagnostics, never candidates.

**Setup.**
- Pre-registration `notes/exp0115-prereg.md` (SHA `7a8ecbb4…`, frozen before data); code 5d6187b; read with
  `read_exp0115.py` → `reading_exp0115.json`.
- `supply_trace --order-ttl 2 --producer entity_planner_buyahead_k5_rf5`, MWWk + RR, 12 and 42 producers,
  horizon 100. The arms:
  - `auction`: neutrality against EXP-0088;
  - `ttl_feasible` (A1): reproduction against EXP-0098;
  - `ttl_feasible_life` (A1-life): accept-later only when the order has a later cycle;
  - `ttl_feasible_new_first` (A1-new-first): this cycle's accepted bids run before carried pending claims.
- **Runs:** pod hk7lwtdbnyr7qg (secure 8 vCPU + 20 GB volume, EUR-IS-1), 08:19–08:29 UTC.
  Attempt 1 (u97t6mh3oxrmbv) ran every arm, but the pull hit a full local disk and the pod had no volume, so its
  rows were lost.
- **Spend:** ≈ 3.4 vCPU-h in total (stop 6), tranche M6-A.

**Validity: valid.**
- No contract failures; the row sets and digests equal EXP-0088's.
- The `auction` rows equal EXP-0088's cell for cell (the default path is untouched).
- The A1 rows equal EXP-0098's (columns added since read zero).

## Reading: **`neither:b2_not_delivered/b2_lost_ttl`**

A1's loss is reproduced, but neither arm recovers it. The extra loss sits in consumer units lost because an
input order failed at its TTL.

**Recovery** (R = the share of A1's loss an arm recovers; 0.5 is the rule's line):

| 12 producers | default | A1 | A1-life | A1-new-first |
|---|---|---|---|---|
| MWWk units / episode (deciding) | 3,703.2 | 3,486.1 (L = **208.0**, floor 146.5) | 3,497.3 (R **0.05**) | 3,515.9 (R **0.15**) |
| RR units / episode | 3,841.9 | 3,717.7 (L 146.5) | 3,735.0 (R 0.32) | 3,715.2 (R 0.18) |

At 42 the pattern is the same: MWWk L 643.3, R 0.03 / 0.16; RR L 517.2, R 0.02 / 0.08.

**Where A1's units go** (12, MWWk, units summed over episodes, A1 − default):
- **Displacement losses vanish:**
  - consumer bids outbid (`c_displaced`): 38,686 → 1,484 (−37,202);
  - input bids outbid (`b2_lost_displaced`): 35,128 → 3,229 (−31,899).
- **Input losses at TTL take their place:**
  - inputs whose order failed at TTL (`b2_lost_ttl`): 63,222 → 112,510 (+49,288);
  - inputs that arrived late (`b2_late`): +18,620;
  - capacity held by waiting reservations (`a_held`): +9,693;
  - inputs that did not fit (`b2_lost_no_fit`): +6,944.
- **Net:** +18,279 lost units. At 42 it is +34,868, with `b2_lost_ttl` +73,650 against
  `b2_lost_displaced` −52,915.

**What the arms rule out:**
- **The order's age (A1-life):**
  - claims dying with their order drop from 4,010 to 1,820;
  - units barely move (R 0.05 at 12, 0.03 at 42);
  - the claims themselves mostly run: 40,108 of 46,235 accepted-later claims run in a later cycle.
- **The step order (A1-new-first):**
  - no claim is displaced by a carried one (A1 displaces 28,251 new bids, 62% of them pricier than the cheapest
    claim that ran);
  - the failures move to the carried claims instead: claims failing at TTL rise from 1,667 to 8,543;
  - R is 0.15.
- **The fund:**
  - the share moved is 3.2–3.3% at 12 and 5.0–5.5% at 42 in every arm;
  - the mean cut-off is within 1% of the default's.
  - It does not track the arms' losses.
- **Lag:** the replenishment lag (Little's law) rises from 2.69 to 2.89 cycles at 12 (3.69 → 4.05 at 42). Inputs
  arrive later.

## Conclusions
- **Cause (DIAGNOSTICS 36): accepted-later work is late work.**
  - Under today's auction, a bid that does not fit is declined, and its order is re-routed next cycle, often to a
    producer with free capacity. Part of that work is lost to displacement.
  - A1 keeps the work on the busy producer instead. The input orders then complete a cycle later, or fail at TTL
    while queued. On TTL-2 supply chains the downstream composite's reservation then times out
    (`b2_lost_ttl`, `b2_late`).
  - The loss is larger than the displacement A1 saves.
  - Neither the order's age (A1-life) nor the order of execution (A1-new-first) is the cause.
- **For learned acceptance (4.2–4.5):**
  - The accept head must learn when declining (so the order is re-routed) beats accepting for later. The
    information that decides it is own-order:
    - the bid's remaining lifetime (already a bid feature);
    - whether the work can run now (the `fits` feature);
    - whether the order is an input to someone else's composite, i.e. the order kind.
  - **Proposal for the 4.2 review:** carry the parent order's kind (consumer / replenishment / investment) on the
    bid as own-order information, as ADR-0061's addendum did for the remaining lifetime.
- **No more hand-written acceptance variants** (user rule 2026-11-06). This closes the A1 diagnosis.
