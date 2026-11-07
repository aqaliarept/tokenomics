# EXP-0124: the EXP-0121 probe with the order-aware accept pass

**Setup.**
- Pre-registration note `notes/exp0124-prereg.md` (SHA `7e7ecb4f…`, unchanged). Driver
  `train/m6_turn_probe.py --linear --ordered`.
- EXP-0122's records (EXP-0120's held-out turns + linear fractions), the same halves.
- **Attempt 1** (code 264c857, pod 8tcohug99rbfsy, 16:35–16:58 UTC, ≈ 3.1 vCPU-h): hard-stopped with no data. The
  stop (3 vCPU-h = 23 min) came from a timing on a short local sample. The slowest setting needed ≈ 24 min, and
  results were written only at the end (my error).
- **Supervisor rule since:** size stops from real-data timings + 50%; save partial results.
- **Attempt 2** (code df880e1, which appends each setting as it finishes; supervisor: same grid, stop 5): pod
  nv3l2cockbiz53, 17:00–17:23 UTC ≈ **3.1 vCPU-h**.
  - A mis-pathed launch before it (pod xpeoeosi2jh7fk) was removed within ≈ 1 min, ≈ 0.1 vCPU-h.
- **Total ≈ 6.3 vCPU-h**, M6-A.

## Reading: **no winner**
Each pair is accept agreement / accept-turn rank. The slack columns give the share of bids that fit alone but get
the wrong accept bit.

| setting | train | probe held-out | slack [−0.5, 0) | slack [0, 0.5) | slack [0.5, 2) | accepted wrong | s / 1k turn-epochs |
|---|---|---|---|---|---|---|---|
| w32 L2 e30 | 0.977 / 0.987 | 0.965 / 0.986 | 13.0% | 17.1% | 5.8% | 2.5% | 0.34 |
| w32 L2 e90 | 0.986 / 0.989 | 0.965 / 0.987 | 14.0% | 17.1% | 5.3% | 2.4% | 0.27 |
| w32 L3 e30 | 0.976 / 0.988 | 0.965 / 0.987 | 13.8% | 18.7% | 4.5% | 2.4% | 0.42 |
| w32 L3 e90 | 0.989 / 0.990 | **0.969** / 0.986 | 12.0% | 14.4% | 4.8% | 2.1% | 0.33 |
| w64 L2 e30 | 0.981 / 0.989 | 0.966 / 0.986 | 13.4% | 16.2% | 5.1% | 2.3% | 0.59 |
| w64 L2 e90 | 0.994 / 0.991 | 0.967 / 0.986 | 13.8% | 12.9% | 4.9% | 2.0% | 0.39 |
| w64 L3 e30 | 0.982 / 0.989 | 0.965 / 0.987 | 13.6% | 17.0% | 5.1% | 2.3% | 0.70 |
| w64 L3 e90 | 0.996 / 0.992 | 0.965 / 0.986 | 14.8% | 14.0% | 5.0% | 2.1% | 0.44 |

Fund-turn rank 1.000.

**Against the set encoder alone (EXP-0122):**
- held-out accept 0.965–0.969 (from 0.942–0.950);
- displaced errors 8–10% (from 16–19%);
- boundary errors 12–19% (from 20–28%).

## Diagnosis: the default labels carry a hidden tie order
- **Ranks:** 4.4% of the scored pairs (claims + accepted bids, after v13's exclusions) have equal price and the same
  kind: 3,963 bid–bid and 2,866 claim–claim. The default orders them by list position (routing order). No feature
  carries that, the same kind of unlearnable label v13 excluded for replenishment pairs. Rank sits at 0.986 in
  every setting.
- **Accept bits:** 36% of held-out bids share their price with another bid of the turn. For **5.9%** of all bids
  (3,727 of 62,806), the default bit flips when equal-priced bids are taken in the reverse order: the auction
  replayed from the rows, one capacity type per producer. Those bits are not determined by any input.
- **Confirmation fit** (local, single thread, low priority, no pod; w32 L2 e30 ordered + linear, same halves, same
  code; overall 0.9649 = the probe's 0.9651):

  | probe held-out bits | count | accept agreement |
  |---|---|---|
  | tie-dependent (the default bit flips with the tie order) | 3,727 | 0.640 |
  | determinable | 59,079 | **0.985** |

## Conclusions
- **The ordered pass works:** displaced errors halved, accept 0.969 at best. The synthetic core claim held (0.992).
- **Two causes remain:**
  - **The hidden tie order:** 5.9% of bits, 64% right; ≈ 2.1 points of the gap. The default's tie-break is the
    arrival order, which the heads do not see.
  - **Genuine misses on determinable bits** at this setting: 1.5% (0.985). Larger settings were not split.
- **Supervisor (2026-11-07):** check the routing-position feature only if the determinable bits clear 0.99. They do
  not at the cheapest setting, so this is reported back. The arrival-position feature is written (uncommitted, off
  by default) and has not been run.
