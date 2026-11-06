# Tasks

The hard rules as in add-m5-potential-game-gnn (cloud runs, pinned data center, watchdog, ledger, root-cause
rule).

## 1. Decisions (user)
- [ ] 1.1 The user decides D1–D7 (defaults in design.md) and the tranche (sizing.md).

## 2. Code (local)
- [x] 2.1 Actor graphs = the agents' own ADR-0058 views (producer entity view; product-stream view with remaining lifetime and last-cycle outcomes) + leak tests. A global graph view (domain) is deferred to the critic (task 2.3).
- [x] 2.2a `train/m6_gnn.py`: producer / orderer GNNs (fully learned, D2) + deployed policies; `train/m6_imitation.py`: Stage I collect / fit (Hogwild) / pre-registered imitation check / timing.
- [ ] 2.2b `env/` potential Φ (D1) for the RL stage.
- [ ] 2.3 `train/` GNN trunk, MAPPO (CTDE critic), checkpoints and pins; `train/` reads (as M5).
- [x] 2.4 L0: the decoding check at 12 / 42 / unseen; ADR-0058 tests; the timing probe re-sizes sizing.md.
  - Done 2026-11-06: tests/test_m6_gnn.py (7); notes/timing.md; sizing re-sized (L0b 8, success path ≈ 86).
- [x] 2.5a The Stage I pre-registration `train/registry/m6_prereg_v1.json` (SHA `38db5c0c…`): the imitation
  rule, the floor rule and its source, the monitor set and replicates, the fit's stopping rule (fixed 20 epochs,
  Hogwild × 16, seed 0) and the reported fidelity metrics. Frozen before EXP-0099; real runs refuse without
  the pin.
- [ ] 2.5b The RL pre-registration, frozen before L2.
- [x] 2.6 [experiment][remote:cpu16] EXP-0099: Stage I (collect → fit → check), pinned EUR-IS-1, stop 16.
  - Done 2026-11-06: **`short`** (3,360.8 vs 3,871.4 units, g_imit −510.6, floor 158.0, deaths 0 / 0); orderer BC
    diverged (unscaled features); ≈ 5.0 vCPU-h. Stop before RL; user decision pending.
  - Rerun 2026-11-06 as EXP-0100 (prereg v2, normaliser + clip + sync): **`faithful`** (+196.1 vs floor 90.5, 0 deaths);
    role swap: the GNN producer carries the gain (+207), the GNN orderer is neutral (+5.7); ≈ 7.2 vCPU-h.
- [x] 2.7 [experiment][remote:cpu16] EXP-0101: side effects of the imitated stack (deciding check before
  MAPPO). Pre-registered 2026-11-06, before any data:
  - **Arms**, the EXP-0100 models (SHA `5acf8b82…`), on the 16 monitor world-episodes × replicates 0 / 1, at
    the same DrawKeys:
    - horizon 100: GNN + GNN, GNN producer + MWWk, k5_rf5 + MWWk (base);
    - horizon 200, reported only: GNN + GNN and the base.
  - **Clauses** (L5's side-effect clauses as in M5, `producer_read._guard_ok`; X / Y = the registered v2
    side-effect params). Each is per arm, against the base at the same horizon:
    - (a) deaths ≤ the base's;
    - (b) run-down share (rows with r_end < X·r_start) ≤ the base's;
    - (c) starvation share (rows with any consumer fill < Y) ≤ the base's.
  - **Reading** (deciding arm GNN + GNN at horizon 100): **`clean`** if no clause is worse than the base, else
    **`side_effect`**, naming the clause(s). GNN producer + MWWk (the swap) and the horizon-200 pair take the
    same clauses, reported only.
  - **Reported, never deciding:** per arm, mean r_end / r_start, min capacity ratio, holding units per cycle
    (M5's stock-heavy flag: > 1.5× the base), investment rejection rate, per-producer fund shares (the
    routing tracer), per-head toggle firing rates; per world, 12-0121 in detail (units, fills, investments,
    capacity, fund).
  - Pinned EUR-IS-1, 16 vCPU, watchdog, stop ≤ 8 vCPU-h.
  - Done 2026-11-06: **`clean`** (run-down 0.563 vs 0.781, starvation 0.656 vs 0.750, deaths 0 / 0; h200 clean); ≈ 0.6 vCPU-h.

## 3. Ladder (experiments, remote)
- [ ] 3.1 L1 → 3.2 L2 → 3.3 L3 → 3.4 L4 → 3.5 L5 → 3.6 L6, each gated by the previous reading (D6). Stop
  branches → DIAGNOSTICS + the user.

## 4. Gated: learned acceptance
- [ ] 4.1 Diagnostic first: A1's harm (design Risks), local offline replays.
- [ ] 4.2 Only if 4.1 licenses it: the accept head as an AEC turn (ADR-0061 slot action), then L3–L5.

## 5. Records
- [ ] 5.1 HISTORY / README / ledger; ADRs for the graph view and the acceptance turn.
