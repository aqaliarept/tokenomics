# EXP-0123 pre-registration: does EXP-0120's imperfect turn-head imitation cost units in closed loop?

Frozen before any data (user decision 2026-11-07: closed-loop diagnostic first). Driver
`tokenomics/train/m6_turn_closed_loop.py`. Never an acceptance read: the 4.3 offline gate is unchanged, and EXP-0120
still reads `offline_fail`.

## Arms
Paired, on the same keyed world-episodes.
- **`candidate`:** F40 (EXP-0114 candidate_it40's trunk and investment gates + the EXP-0110 start's replenish) with
  EXP-0120's turn heads (`models.pt`, held-out accept agreement 0.962), greedy, sharing one trunk pass per
  producer-cycle (TurnCache).
- **`reference`:** F40 with the default turns (today's auction, claim order and funding order).
- The orderer is the same frozen one in both.

## Episodes
- The 16 monitor worlds × 2 fresh episodes (stage `m6-turn-closed-loop`, never played; disjoint from every earlier
  episode, asserted) × r0 / r1.
- Horizon 100, 12 producers: 128 episodes.

## Reading
v11's metric and floors (M5's g / F over world-episodes):
- **`gain`:** g > F.
- **`cost`:** g < −F, or deaths, run-down share or starvation share worse.
- **`no_cost`:** otherwise (|g| ≤ F and no worse side effects).

## Reported
From the candidate's episodes, at each accept turn, the heads' decision against the default decision on the same
state:
- the closed-loop accept-bit agreement;
- the differing bits by direction (extra accept / extra decline) and by slack bin for bids that fit alone
  (DIAGNOSTICS 38), and `no_fit`;
- the pending claims created and their fate: ran in a later cycle, or failed at TTL (DIAGNOSTICS 36);
- bids, claims or requests past the slots (`beyond_k`).

## Sizing
- About 128 episodes at ≈ 1.4× the cost of the no-turn stack ≈ 0.5 vCPU-h, plus pod setup.
- `tests/test_m6_pilot.py` runs on the same pod after the main job (supervisor; records staged under
  `/workspace/experiments`).
- Pod 8 vCPU with a persistent volume, EUR-IS-1, **stop 3**, M6-A.

## Consequence
Reported to the user, who decides whether 4.5 starts from these heads.
