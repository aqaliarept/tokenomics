# EXP-0123: does EXP-0120's imperfect turn-head imitation cost units in closed loop?

**Setup.**
- User decision 2026-11-07: closed-loop diagnostic first.
- Pre-registration note `notes/exp0123-prereg.md` (SHA `799963da…`, committed 96f393d before data). Driver
  `train/m6_turn_closed_loop.py`.
- **Arms:**
  - `candidate`: F40 + EXP-0120's turn heads (held-out accept agreement 0.962), greedy;
  - `reference`: F40 + the default turns.
- 16 monitor worlds × 2 fresh episodes (stage `m6-turn-closed-loop`, never played) × r0 / r1 = 128 episodes, paired.
- Pod 65bih6r0l19dzj (secure 8 vCPU + 20 GB volume, EUR-IS-1), 15:56–16:03 UTC ≈ **1.0 vCPU-h** (stop 3, M6-A).
- The 4.3 offline gate is unchanged: EXP-0120 still reads `offline_fail`.
- `tests/test_m6_pilot.py` ran on the same pod: **8 passed**.

## Reading: **`cost`**
v11's metric and floors (units per episode):

| | candidate (heads) | reference (default turns) |
|---|---|---|
| units / episode | 4,129.2 | 4,185.9 |
| g / F | **−56.7 / 43.4** | |
| deaths | 0 | 0 |
| run-down share | **0.44** | 0.33 |
| starvation share | **0.625** | 0.594 |

Both clauses hold: g < −F, and the side effects are worse.

## Where the decisions differ (the candidate's accept turns, against the default on the same state)
- **Closed-loop accept agreement:** **0.956** (238,928 of 249,942 bits; offline held-out was 0.962).
- **Direction:** 6,598 extra accepts (the heads accept what the auction declines), 4,416 extra declines.
- **Slack**, all differing bits of bids that fit alone:

  | slack (own capacities) | differing bits |
  |---|---|
  | [−2, −0.5) | 2,773 |
  | [−0.5, 0) | 3,825 |
  | [0, 0.5) | 2,998 |
  | [0.5, 2) | 1,322 |
  | ≥ 2 | 96 |

  No `no_fit` bid differs.
- **Pending claims created** by the heads' extra accepts: 13,360 (the default creates none).
  - 6,093 (46%) ran in a later cycle;
  - 733 (5.5%) failed at TTL;
  - the rest were still pending at the episode end, or left by re-validation.
- Nothing past the slots (`beyond_k` 0).

## Conclusions
- **The imperfect imitation costs units:** −1.4% per episode, beyond the floor, with more run-down and starvation.
  A 4.4% disagreement at the threshold is not harmless.
- **Mechanism (DIAGNOSTICS 36):**
  - the extra accepts of bids that do not fit become pending claims, i.e. late work, which is what made A1 harmful;
  - the extra declines lose work the auction would have run.
- **The 0.99 gate is justified.** The heads should not start 4.5 as they are. The decision goes to the user.
