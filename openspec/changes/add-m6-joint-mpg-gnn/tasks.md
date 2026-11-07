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
- [x] 2.8 [experiment][remote:cpu16] EXP-0102: validation of the imitated stack on unseen worlds (supervisor
  decision 2026-11-06), before MAPPO. Pre-registered in `train/registry/m6_prereg_v3.json` (SHA `409643f2…`; v1 / v2
  frozen). The EXP-0100 models; arms GNN + GNN, GNN producer + MWWk (reported), k5_rf5 + MWWk (base); L5's loaded
  atlas v2 validation worlds (9 at 12, deciding; 3 at 42, reported), 8 episodes (stage m4-es-validate) × replicates
  0 / 1, horizon 100, L5's floors. `parity` if g ≥ −floor with no extra deaths and run-down / starvation shares ≤ the
  base's; `better` if parity and g > floor; else `short`. Held-out set untouched. Pinned EUR-IS-1, stop ≤ 8.
  - Done 2026-11-06: **`better`** (12: +95.9 vs floor 35.0; 42: +196.1 vs 160.0; deaths 0; run-down ≤ base); ≈ 1.9 vCPU-h.
- [x] 2.9 [experiment][remote:cpu16] EXP-0103: held-out read of the imitated stack (user decision 2026-11-06: held-out
  now, then MAPPO). Pre-registered in `train/registry/m6_prereg_v4.json` (SHA `f06fdf0c…`), one shot with a pending
  entry in `experiments/m6_heldout_ledger.json` committed before the launch. M5's L6 set: every atlas v2 held-out
  world played (12 at 12, 4 at 42), the loaded ones gating (8 / 2), 8 episodes (stage m5-heldout) × replicates 0 / 1,
  horizon 100; EXP-0102's rule (`parity` / `better` / `short`, side-effect clauses), 12 deciding, 42 reported.
  MAPPO's later held-out read is a second comparison on this set. Pinned EUR-IS-1, stop ≤ 8.
  - Done 2026-11-06: **`better` at 12** (+184.4 vs floor 82.2, deaths 1 vs 2, run-down 0.758 vs 0.867); 42 reported `short`
    on run-down (1.000 vs 0.500; capacity drain on 42-0125, undiagnosed); ≈ 2.2 vCPU-h.
- [x] 2.10 [experiment][remote:cpu16 + local probe] EXP-0104: the 42-0125 drain, diagnosed (supervisor decision 2026-11-06):
  the upgrade trap by imitation (DIAGNOSTICS 32); veto counterfactual recovers 60,754 vs 51,458 units at h200; same
  signature on train 42-0076 / 42-0126; ≈ 1.9 vCPU-h.
- [x] 2.11 [experiment][remote:cpu16] EXP-0105: Stage I v3 (supervisor decision 2026-11-06, DIAGNOSTICS 32): the bounded
  ceiling-margin feature, collect at 12 and 42 (training split only), refit, the three-clause check (`m6_prereg_v5.json`
  `4fdf45ef…`; the pre-data `63d85a94…` superseded). Pinned EUR-IS-1, stop 24 (raised from 16 by the supervisor to keep the
  42 check at replicates 0 / 1).
  - Done 2026-11-06: **`short` on (a), (b), (c)**: the v3 producer passes the ceiling more (381 vs 0 at 42) and loses units at 12
    (−274 vs base, −470 vs EXP-0100); cause undiagnosed; ≈ 16.2 vCPU-h over 3 attempts.
- [x] 2.12 [experiment][remote:cpu8] EXP-0106: the v3 upgrade gate, diagnosed offline: the margin feature is used but the
  boundary is learned smoothly under joint-scale training (DIAGNOSTICS 33); ≈ 0.6 vCPU-h.
- [x] 2.13 [experiment][remote:cpu16] EXP-0107: Stage I v4 (prereg v6): offline gate `gate_pass` (0 past-ceiling firing at
  both sizes); check `short` (a, b units, c; 0 past the ceiling at 42) with the orderer diverged at epoch 17 (confound,
  undiagnosed); ≈ 18.7 vCPU-h.
- [x] 2.14 [experiment][remote:cpu16] EXP-0108: attribution (prereg v7): candidate C (v4 producer + EXP-0100 orderer) `short`, ≈ the
  EXP-0107 stack → the v4 producer carries the shortfall; reported arms lost to the hard stop; ≈ 12.0 vCPU-h.
- [x] 2.15 [experiment][remote:cpu16] EXP-0109: closed-loop diagnosis of the v4 shortfall (supervisor decision 2026-11-06).
  Pre-registered before any data (`train/m6_restore_probe.py`, RULE):
  - **Arms:** v4c (the v4 producer + the EXP-0100 orderer), v2 (the EXP-0100 stack), base (k5_rf5 + MWWk),
    and v4c_bias (v4c with the restore-gate bias shifted by b = +7.234, so its restore firing on the 12 monitor
    samples equals k5_rf5's label rate, 0.373; offline calibration, `calibration.json`).
  - **Sets:** the 12 monitor world-episodes × r0 / r1, and the four non-held-out 42 worlds with EXP-0107's
    largest shortfalls (0000, 0043, 0076, 0039; 4 episodes, stage m6-drain, × r0 / r1), all at horizon 200.
  - **Reading:** recovered share r = (v4c_bias − v4c) / (base − v4c) in mean units per set. `under_restoration` if
    r ≥ 0.5 at both sets; `partial` if r ≥ 0.2 at both or ≥ 0.5 at one; else `other`.
  - **Descriptive:** per-cycle restore / upgrade / expand asks and orders, capacity, losses by class.
  - **Run:** pinned EUR-IS-1, persistent `/workspace` volume, rows streamed as they finish, stop 16.
  - Done 2026-11-06: **`other`** (r = −0.02 / −0.04): not under-restoration; v4 grows less early (fewer upgrades and expansions,
    capacity −7% / −13% in cycles 0–49) and loses more inputs to TTL; DIAGNOSTICS 34; ≈ 2.8 vCPU-h.
- [x] 2.16 [experiment][remote:cpu16] EXP-0110: critic-free Stage 0 (prereg v8): **`no_signal`** (no D0.3 cosine beats its null);
  D0.1 0.98 / 0.97; D0.2 γ 0.99 / 0.90; D0.5 ≈ 110 s per 42 episode; absent vs underpowered open (DIAGNOSTICS 35); ≈ 2.9 vCPU-h.
- [x] 2.17 [experiment][remote:cpu8] EXP-0111: power calibration (DIAGNOSTICS 35): **`absent`**: the producer's score is 99.4% replenish
  exploration, the gates explore about 1%, the orderer moves no return; about 700 seeds per half needed; ≈ 2.1 vCPU-h. Options to the user.
- [x] 2.18 [experiment][remote:cpu16] EXP-0112: Stage 0 with reshaped exploration (prereg v9): **`absent` under the rule**, but a real producer
  gradient at 12 (CI > 0), E3 n80 = 48 seeds per half; E4 no help; orderer and 42 no gradient; ≈ 7.7 vCPU-h. Decision to the supervisor / user.
- [x] 2.19 [experiment][remote:cpu16] EXP-0113: capped Stage 1 pilot (prereg v10): **`hold`, rising** (greedy +95 → +107 vs the EXP-0100 stack,
  floor ≈ 150; side effects better; training return −38.8 → −20.1); ≈ 11.2 vCPU-h. Continuation is the user's call.
- [x] 2.20 [code][local:light] Pilot continuation (prereg v11 `f55aea40…`): `m6_pilot --prereg v11 --resume-from <EXP-0113 artifacts>`
  resumes the SHA-pinned iteration-10 checkpoint (producer + Adam), runs 11-40, stop-only gates at 15 / 20 / 30,
  the deciding reading at 40; resume pins and EXP-0100 reference-row reproduction checked.
- [ ] 2.21 [experiment][remote:cpu16] EXP-0114: the pilot continued to iteration 40 (prereg v11); stop ≤ 40 vCPU-h.

## 3. Ladder (experiments, remote)
- [ ] 3.1 L1 → 3.2 L2 → 3.3 L3 → 3.4 L4 → 3.5 L5 → 3.6 L6, each gated by the previous reading (D6). Stop
  branches → DIAGNOSTICS + the user.

## 4. Gated: learned acceptance
- [ ] 4.1 Diagnostic first: A1's harm (design Risks), local offline replays.
- [ ] 4.2 Only if 4.1 licenses it: the accept head as an AEC turn (ADR-0061 slot action), then L3–L5.

## 5. Records
- [ ] 5.1 HISTORY / README / ledger; ADRs for the graph view and the acceptance turn.
