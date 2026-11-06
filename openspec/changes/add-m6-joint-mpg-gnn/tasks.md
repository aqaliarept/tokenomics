# Tasks

The hard rules as in add-m5-potential-game-gnn (cloud runs, pinned data center, watchdog, ledger, root-cause
rule).

## 1. Decisions (user)
- [ ] 1.1 The user decides D1–D7 (defaults in design.md) and the tranche (sizing.md).

## 2. Code (local)
- [ ] 2.1 `domain/` graph view (ADR-0058; includes remaining lifetime and last-cycle outcomes) + tests.
- [ ] 2.2 `env/` potential Φ (D1) + graph encoder; the producer / orderer residual heads at zero = k5_rf5 + MWWk
  (byte for byte).
- [ ] 2.3 `train/` GNN trunk, MAPPO (CTDE critic), checkpoints and pins; `train/` reads (as M5).
- [ ] 2.4 L0: the decoding check at 12 / 42 / unseen; ADR-0058 tests; the timing probe re-sizes sizing.md.
- [ ] 2.5 The pre-registration v1 frozen before L2.

## 3. Ladder (experiments, remote)
- [ ] 3.1 L1 → 3.2 L2 → 3.3 L3 → 3.4 L4 → 3.5 L5 → 3.6 L6, each gated by the previous reading (D6). Stop
  branches → DIAGNOSTICS + the user.

## 4. Gated: learned acceptance
- [ ] 4.1 Diagnostic first: A1's harm (design Risks), local offline replays.
- [ ] 4.2 Only if 4.1 licenses it: the accept head as an AEC turn (ADR-0061 slot action), then L3–L5.

## 5. Records
- [ ] 5.1 HISTORY / README / ledger; ADRs for the graph view and the acceptance turn.
