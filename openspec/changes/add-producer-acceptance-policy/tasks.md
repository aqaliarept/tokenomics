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
- [ ] 3.1 [experiment][remote:cpu16] EXP-NNNN: A1 and A2 vs k5_rf5 default (D5). ≈ 2 vCPU-h, stop 4. Pinned
  EUR-IS-1.

## 4. Records
- [ ] 4.1 [code][local] ADR-0061 accepted; README / HISTORY; DIAGNOSTICS if a new cause appears.
