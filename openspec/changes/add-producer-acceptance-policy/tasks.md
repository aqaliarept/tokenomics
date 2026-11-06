# Tasks

Local budget, remote profiles and the hard rules as in add-m5-potential-game-gnn. No experiment before group 2's
goldens are green.

## 1. Decisions (user)
- [x] 1.1 The user decides D1–D5 (defaults in design.md). ADR-0061 moves from proposed to accepted.

## 2. Domain and env (code)
- [x] 2.1 [code][local] `domain/`: acceptance as an input to `produce()` (accepted / declined per sub-order);
  reservations as multi-cycle capacity claims (D1); per-sub-order outcome records (D2). Verify: unit tests for
  accept, decline, execute-later and fail-at-TTL; token burn and cut-off; no same-cycle re-route.
- [x] 2.2 [code][local] `eval/baselines.py`: `acceptance_auction` (D3) and `acceptance_ttl_feasible` (A1),
  `acceptance_ttl_inputs` (A2). Verify: the default reproduces today (goldens; EXP-0088 k5_rf5 replay on 2
  world-episodes).
- [x] 2.3 [code][local] `env/`: the acceptance phase, K-slot action, additive orderer-view outcome fields (D2, D4).
  Verify: `pettingzoo` api_test; existing policies unchanged.
  - Done 2026-11-06 (notes/build.md): D4 implemented as an env-level hook (no AEC turn), flagged; default byte-identical (goldens 336; auction-as-policy = default rows on 12-0111 / 12-0391; EXP-0088 k5_rf5 replay 2 / 2); tests/test_acceptance.py 8.

## 3. Experiments
- [ ] 3.1 [experiment][remote:cpu16] EXP-0098: A1 (`ttl_feasible`) and A2 (`ttl_inputs`) vs the k5_rf5 default
  (EXP-0088's k5_rf5 rows, the default acceptance; digests must match), × {MWWk R = 2, RR}, TTL 2, EXP-0057's
  worlds, observer on. ≈ 2 vCPU-h, stop 4; pinned EUR-IS-1.
  - **Pre-registered reading (2026-11-06, before data; as EXP-0084's; 12 producers decide, 42 reported), per arm
    A ∈ {A1, A2}:** per orderer, diff_w = mean paired (episode, replicate) units(A) − units(default) per world,
    floor_w = the atlas v2 `noise_floor`.
    - `A_helps` if mean_w(diff_w) > mean_w(floor_w) for BOTH orderers, with no extra deaths and run-down check
      failures (r_end < 0.7516 r_start) up by ≤ 5 (absolute, per orderer).
    - `harmful` if mean_w(diff_w) < −mean_w(floor_w), or extra deaths, for EITHER orderer.
    - `no_gain` otherwise.
  - **Reported only:**
    - fund gaming: declined bids priced below the cycle's cut-off (`acceptance_stats`), and the fund share moved
      against the default (half the L1 distance between the arms' per-producer fund-granted shares, per
      episode);
    - pending claims completed late vs failed at TTL;
    - consumer units lost to `e_declined` vs the late-input class (`b2_late`);
    - whether A1's one-future-cycle assumption binds (pending claims failing at TTL; if so, the sub-order could
      carry its order's remaining TTL as own-order information);
    - deaths, units, held stock, run-down.
  - Ran 2026-11-06 (EXP-0098): **A1 `harmful`** (12: −208.0 / −146.5 + 1 death; accepted-later claims mostly die with their parent order, so the one-cycle assumption binds); **A2 invalid** (observer: `b2_lost_declined` missing from `B2_SUBS`), fix and rerun pending. ≈ 0.7 vCPU-h.

## 4. Records
- [ ] 4.1 [code][local] ADR-0061 accepted; README / HISTORY; DIAGNOSTICS if a new cause appears.
