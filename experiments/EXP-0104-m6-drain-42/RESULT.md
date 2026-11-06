# EXP-0104: why the imitated M6 stack drains capacity at 42 producers

add-m6-joint-mpg-gnn, supervisor decision 2026-11-06 (root-cause rule, before the RL pre-registration). Code
e368e08 (`train/m6_drain.py`); EXP-0100 models. Pod o0vjweodztdyvt (secure 16 vCPU, EUR-IS-1), 12:23–12:30 UTC
≈ **1.9 vCPU-h** (stop 6). Local one-episode checks: `artifacts/drain_0125.py` (DIAGNOSTICS 29's checks).

## Question
In EXP-0103 the GNN stack drained capacity on held-out 42-0125: run-down 16 / 16, against 0 / 16 for the base.
Is it unfunded restorations (DIAGNOSTICS 30), restorations never ordered, restorations rejected, or something
else? Does the pattern appear outside the held-out set?

## Pod run: horizon 200
42-0125 (EXP-0103's keys) and the 8 loaded 42-producer atlas worlds outside the held-out set (rho > 0.5,
fresh seeds). A read-only tracer records every investment order's funding and fate, and each needy
producer-cycle (ratio < X): running, asked, or never ordered.

**On 42-0125 (16 episodes):**
| arm | units 0–99 | units 100–199 | capacity at cycles 50 / 99 / 149 / 199 | restorations ordered / unfunded / rejected / completed |
|---|---|---|---|---|
| GNN + GNN | 28,043 | **24,801** | 616 / 550 / 488 / 462 | 4,063 / 0 / 0 / 4,063 |
| k5_rf5 + MWWk | 26,663 | 34,032 | 632 / 688 / 706 / 714 | 4,096 / 0 / 0 / 4,096 |

- **The drain bites:** over 200 cycles the GNN stack makes 52,843 units against the base's 60,694 (−13%). It
  leads in the first 100 cycles and loses 9,231 units in the second 100.
- **Not unfunded, rejected or never ordered:** 0 restorations unfunded and 0 rejected. The GNN asks for
  restorations in more of its needy producer-cycles than the base (2,819 vs 1,436) and completes about as
  many restorations (4,063 vs 4,096). Every producer gets funding.
- **More upgrades complete:** 652 vs 572.

**Elsewhere at 42 (4 episodes each, h200):**
- GNN capacity falls in the second half on 5 of 8 worlds: 0013, 0066, 0076, 0126, 0081 (slightly).
- Restorations are unfunded or rejected about as often as the base, or less (0081: 1,742 vs 1,282 unfunded,
  on a world with rho 5.4 where both are fund-limited).

## Discriminating checks (DIAGNOSTICS 29, local, one episode, horizon 200)
| world | arm | units | 0–99 / 100–199 | end capacity | producers past the sustainable ceiling |
|---|---|---|---|---|---|
| 42-0125 (held-out) | GNN + GNN | 51,458 | 27,117 / 24,341 | 464 | **12** |
| 42-0125 | GNN + GNN, upgrades past the ceiling vetoed | **60,754** | 27,802 / 32,952 | 834 | 0 |
| 42-0125 | k5_rf5 + MWWk | 60,618 | 26,497 / 34,121 | 713 | 0 |
| 42-0076 (train) | GNN + GNN | 14,677 | 6,151 / 8,526 | 90 | **8** |
| 42-0076 | k5_rf5 + MWWk | 12,022 | 4,617 / 7,405 | 148 | 0 |
| 42-0126 (train) | GNN + GNN | 10,751 | 4,313 / 6,438 | 130 | **9** |
| 42-0126 | k5_rf5 + MWWk | 11,228 | 4,142 / 7,086 | 136 | 0 |

"Past the ceiling" means `degradation_for(t) > restoration.restores` on the restoration type: even a
restoration every cycle loses to degradation.

## Diagnosis: the upgrade trap (DIAGNOSTICS 29), reached by the imitated producer at 42 producers
- k5_rf5 has the exact sustainable-upgrade cap and never passes the ceiling. The GNN producer imitates that
  cap but does not reproduce it at 42-producer scale: maxes reach 743–2,633, far beyond the 12-producer
  training worlds.
- It upgrades past the ceiling on 8–12 producers per world. Capacity then drains from about cycle 50,
  although every restoration is funded and completed.
- **The veto counterfactual removes the drain:** units rise from 51,458 to 60,754, above the base, and end
  capacity from 464 to 834.
- The same signature (8 and 9 producers past the ceiling, 0 for k5_rf5) appears on two non-held-out training
  worlds, so the diagnosis does not rest on the held-out world. On 42-0076 the GNN stack still out-produces
  the base over 200 cycles despite the trap.
- **Why the horizon-100 reads hid it:** the drain bites in the second 100 cycles. MAPPO with γ = 1 and a
  finite horizon would not see it either: drain at the end of an episode is free under the current potential.
- **EXP-0103's 42-producer read** (reported, never deciding) is now explained by this cause. The 12-producer
  read stays clean (EXP-0101: no drain at 12 up to horizon 200).

## Consequences for MAPPO (to the supervisor; no change made)
- The potential's run-down term charges ρ < X per cycle, but a trap that bites after the horizon is not
  charged. Candidate responses:
  - **(a)** a horizon or bootstrap that sees the drain: train at horizon 200, or bootstrap V at truncation;
  - **(b)** a learned-feature fix: give the producer the post-upgrade degradation − restore effect as a signed
    row feature;
  - **(c)** 42-producer worlds in the training set.
- **(d)** A veto mask would be a hand-written rule. It was used here only as a counterfactual.
