# Producer finalization (supervisor plan, 2026-11-05)

**User decision:** finalize the hand-written producer before any more M5 learning. Producer questions go to the
supervisor. v5 is frozen (`06160cac…`, after EXP-0085) and EXP-0086 is not launched; M5 re-bases later if the
producer changes.

## Step 1: EXP-0087, rf4's loss classes (offline read of EXP-0084's rows)
- At 12 producers, MWWk: 4,767 units lost per episode.
- By class: input arrived late (after the order's TTL) 37%; input order died upstream 18%; contention between
  reservations 18%; outbid 10%; displaced upstream 9%; capacity 2.5% (rf3 → rf4: 259 → 118).
- Same order at 42.

## Step 2: 12-0391 (rf4 −402 / −482 against rf3 in EXP-0084), local trace
Script: `probes/upgrade_cap_0391.py` (`--exact` for the counterfactual).

**Cause: the view estimate errs at a tiny max.**
- p1_1 starts at max `capacity-1` = 1. The view's degradation is `ceil(6% · 1) = 1`, so the estimate for
  max 1 → 66 is `ceil(1 · 66 / 1) = 66` > restore 53, and the upgrade is blocked.
- The exact value is `ceil(6% · 66) = 4` ≤ 53.
- p1_1 stays at capacity 0–1 for the whole episode under rf4; rf3 grows it to 391.

**The other blocks are correct:**
- p2_2 at max 693: exact 61 > 57.
- p2_1 at 846: exact 73 > 69.
- These are real traps: rf3 drains p2_2 to 57 and p2_1 to 69 by the end.

**Episode 1977552299, MWWk r0:**

| arm | horizon 100 | horizon 200 |
|---|---|---|
| rf3 | 4,795 | 8,821 |
| rf4 (view estimate) | 4,339 | 7,538 |
| rf4 with the exact degradation % (diagnostic only) | **4,855** | **9,717** |

**So the cap is right and its estimator is wrong at small max.** At max < 100 / d%, the ceil hides the rate.
A view-only estimate cannot recover d% at max = 1: the view's degradation of 1 fits any rate from 1% to 100%.

**Fix options (the supervisor's decision):**
1. Expose the degradation % in the view: an additive `CapacityRow` field, a domain change. This makes the cap
   exact.
2. View-only: apply the cap only when the rate is pinned to within 1 point, i.e. `max ≥ 100` (with
   `deg / max` then within 1% of d%), and otherwise allow the upgrade.

## Step 3: rf5 and the buffer re-sweep
- **PF-3 / PF-4, rf5 (exact cap, EXP-0089):** `no_gain` by the rule.
  - 12: +135.3 (MWWk) / +155.1 (RR) vs floor 146.5, 0 worlds below, 0 deaths.
  - +18.5 / +44.4 over rf4.
  - It holds, so PF-5 ran on rf5.
- **PF-5, EXP-0088: k* = 5** at 12 (both orderers), +106 / +106 over rf5. k = 6 is best, at the edge of the
  range; the gains diminish (+145, +106, +55 per step).
- **What the buffer moves:** the gain comes from fewer input orders dying upstream (MWWk 78k → 61k from k = 3 to
  6). Late inputs at the assembler stay flat (157k → 153k, −2.6%).

## Step 4: a rule for "input arrived late" (37% of losses)? None proposed
The trace shows that this class does not respond to the producer's levers:
- **Buffer size:** late inputs fall only 2.6% from k = 3 to 6 (EXP-0088), while the buffer's whole gain comes
  through the died-upstream class.
- **Inventory room:** at 12, only 6% (MWWk) / 7% (RR) of rf5's late units are headroom-blocked (EXP-0089's
  `st_b2_late_headroom_blocked`). EXP-0078 showed that re-allocating room only moves the loss (rf1 / rf12
  harmful), and that more room is what helps; rf3 / rf5 already have it.
- **Timing is structural:** the input for a reservation accepted at cycle t is ordered by the next cycle's
  command (t + 1). It misses TTL 2 whenever the supplier needs more than that one cycle. The producer's command
  (replenish / restore / upgrade / expand) cannot order before the acceptance. Declining or deferring such
  acceptances is the auction's job, in the domain, outside the producer's command.

So I see **no hand-written producer rule with a plausible gain above the floor** on this class. The second
class (died upstream, 18%) is the one the buffer moves, and k* = 5 already takes what the rule licenses.

**The producer is final:** `entity_planner_buyahead_k5_rf5`, i.e. buy-ahead k = 5, the early inventory
expansion, and the exact sustainable-upgrade cap.
- Possible follow-ups (not proposed as producer rules):
  - an auction-side "accept only if the inputs can arrive within TTL" rule (a domain change);
  - learning (the M5 re-base, now on k5_rf5).
