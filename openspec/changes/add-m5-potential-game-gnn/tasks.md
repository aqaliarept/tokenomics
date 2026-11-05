# Tasks

Local budget: 2 cores, ≈ 3.8 GB RAM, shared by supervisor + implementer + reviewer. Classes:
local | local-exclusive | remote:<profile>.
- **Paths** are under /srv/research/tokenomics/tokenomics/ unless absolute.
- **Goldens** = `tests/test_legacy_invariance.py`, green and unedited in every code task.
- **Targeted tests:** one file at a time, `nice -n 19`, BLAS 1 thread, load < 4.

Remote profiles (secure only):
- `cpu16` / `cpu8` (`--create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12
  --vcpu N --data-center <pinned>`), never unpinned (memory note, 2026-11-05);
- `--workers` = vCPU;
- staging under the git-ignored `tokenomics/_pod_ckpts/` only; `--verify-source-commit`;
- a watchdog at the stop; nothing local while a pod bills.

Budget (design D9): tranche **M5-P, ≈ 80 vCPU-h, requested (user approval pending)**.
- Per run: hard stop = 2 × estimate, ≤ 40; launch-fit on billed spend.
- Ladder gates (design D6): a **Gated** group starts only when the named reading is in its EXP
  `RESULT.md`. Any other reading stops the ladder for a diagnosis task (root-cause rule) and the
  user.

## 1. Pre-step: "expand earlier" alone (design D8)

- [x] 1.1 [code][local] Add `entity_planner_buyahead_k4_rf3` (k4 + part 3 of 7.22 only) to
  `eval/baselines.py` `EXTRA_PRODUCING`. Verify: all parts off == k4 (existing test pattern);
  test_buyahead; goldens.
  - Done 2026-11-05: `entity_planner_buyahead_k4_rf3` = `producing_entity_planner_buyahead_rf(4,
    blocked_first=False, keep_room=False)`. With parts 1 / 2 off, `plan_reservations_first` now
    keeps k = 4's replenishment exactly (it staged them before), so rf3 = k4 + earlier expansion only:
    replenishment, restoration and upgrade equal k4's on views (test). rf / rf1 / rf12 unchanged
    (EXP-0078 rf rows reproduced 6 / 6 on 12-0111 at horizon 100). test_buyahead 15,
    test_supply_trace 28, goldens 336, test_roster 36.
- [x] 1.2 [experiment][remote:cpu16] EXP-0079: rf3 vs k4 (EXP-0077 rows; digests must match),
  × {MWWk, RR}, TTL 2, EXP-0057's worlds, observer on. **Pre-registered reading (2026-11-05,
  before data; as EXP-0078 / 7.22b; 12 producers decide, 42 reported):** per orderer, diff_w =
  mean paired (episode, replicate) units(rf3) − units(k4) per world, floor_w = atlas v2
  `noise_floor`; `rf_helps` if mean_w(diff_w) > mean_w(floor_w) for BOTH orderers with no extra
  deaths and run-down check failures (r_end < 0.7516 r_start) up by ≤ 5 (absolute, per orderer);
  `harmful` if mean_w(diff_w) < −mean_w(floor_w) or extra deaths for EITHER orderer; `no_gain`
  otherwise. Report late / headroom-blocked late units, expansions requested / completed,
  investment rejections, held stock. On `rf_helps` rf3 becomes the base for group 2 (design D8:
  plan text updated); otherwise k4 stays the base. ≈ 0.7 vCPU-h (EXP-0078: 1.9 for 3 × 213 jobs),
  stop 3; cpu16 pinned EUR-IS-1 (fallback 8, pinned), charged to M5-P. After 1.1.
  - Done 2026-11-05 (EXP-0079): `rf_helps` (12: +326 / +345 vs floor 146.5, 0 deaths, run-down 0 / −1; 42: +868 / +940). **The ladder's base is now `entity_planner_buyahead_k4_rf3`** (design D8): every "base" below means rf3; θ = 0 equivalence and the D5 guard are to rf3. ≈ 0.9 vCPU-h; M5-P ≈ 0.9 / 80 (notes/tranche-ledger.md).

## 2. Residual policy and learner (code)

- [x] 2.1 [code][local] `train/producer_residual.py`: the D2 residual (per-input k_g, extra buying,
  expansion trigger) over shared per-row features. The features are agent-local and past-only,
  with episode state weak-keyed by the system (7.22's fix). Verify:
  - θ = 0 equals the base action for action over an episode at 12 and at 42;
  - the parameter count is identical at 12 and 42;
  - an ADR-0058 check: features read only the producer's own view and history;
  - goldens.
  - Done 2026-11-05 (notes/2.1.md): θ 23 values (θ_k 9, θ_b 9, θ_e 5) on rf3; buffer k_g = 4·exp(clip(θ_k·φ_g, ±2)), extra units on top of the target, expansion when θ_e·ψ > 0; `plan_reservations_first(plan_view=)` added (None = unchanged); θ = 0 == rf3 at 12 and 42; tests 6.
- [x] 2.2 [code][local] `train/producer_es.py`: ES on 2.1 reusing `order_es`:
  - bar precompute on the base, keyed draws, antithetic pairs, rank fitness, Adam, minibatch 24;
  - modes `probe` / `train` / `noise-floor`;
  - checkpoints and resume with pins.
  Verify: a horizon-12 dry run; resume refuses other pins; noise-floor mode reproduces the bar's
  base rows.
  - Done 2026-11-05 (notes/2.2.md): modes bar / noise-floor / probe / train on M4's pinned training set; units-only fitness, rf3-relative guard with the v4 margin formula; holding / investment-rejection diagnostics; EXP-0079 cross-check in the bar; tests 7 (dry chain, resume, pin refusal, CRN, cross-check); constants pending 2.3.
- [x] 2.3 [code][local] Pre-registration `train/registry/es_prereg_m5_v1.json`: the D6 readings,
  thresholds, the D5 margin formula, the monitor set, σ, learning rate. Frozen by `--mode
  freeze`, pinned by SHA-256; real runs refuse a mismatch. Verify: freeze gives MATCH; a changed
  section is refused.
  - Done 2026-11-05 (notes/2.3.md): frozen `67721b38…` (pinned `PREREG_SHA256`); supervisor decisions: M4's normaliser fitted on rf3's bar rows, probe at σ 0.05 and 2σ with M4's step reading, M4 monitor (replicates 0 / 1, generation 0); flags: Adam lr / wd and L5 8 episodes carried from M4. Tests: producer_es 13, producer_residual 9.
- [x] 2.4 [code][local] Validation and held-out reads (L5 / L6) reusing `order_es_validate` and the
  M4 ledger design (D9 C0 / C1), with key `held-out/m5-producer`. Verify: the reading table per
  size; `transfer_failure`; ledger refusals; `--recompute` MATCH on a dry run.
  - Done 2026-11-05 (notes/2.4.md): `train/producer_read.py` (pilot / validate / heldout / recompute / ledger); supervisor decisions: L6 candidate = the seed with the larger 12-producer candidate − rf3 (tie → seed 0), in pre-registration v2 `5887b2b2…` (re-frozen after review; v1 untouched); the L3 pilot read in this module. Tests: producer_read 11.

## 3. Sizing (code)

- [x] 3.1 [code][local] Timing probe: horizon-12 episodes of 2.1 at θ ≠ 0 at 12 and 42 against
  buy-ahead k4. Re-size design D9 and the stops. Verify: sizing.md lists the measured seconds,
  the estimates, the stops and a launch-fit table against M5-P.
  - Done 2026-11-05 (sizing.md): 12-producer episode 1.65 s, 42-producer 28.7 s at horizon 100 (local; pods ≈ 2× faster); success path ≈ 38 vCPU-h (D9's 60 re-sized); stops L1 3, L2 4, L3 14, L4 14 / 27, L5 7, L6 7; launch-fit ≤ 42.4 of 80.

## 4. Base and probe (experiment). Gated: tranche M5-P approved by the user

- [x] 4.1 [experiment][remote:cpu16] EXP-0080: L1 bar + noise floor on the base (rf3, design D5).
  On M4's pinned training set (EXP-0058 `bar/`, 128 world-episodes; design D7). Verify: r0
  reproduces the bar's rf3 rows; rf3 replayed on 2 EXP-0057 world-episodes equals EXP-0079's rows;
  self-flags ≤ 5%
  with the margin; 0 contract failures. ≈ 1.5, stop 3 (sizing.md).
  - Done 2026-11-05 (EXP-0080): r0 128 / 128 equal; EXP-0079 cross-check 2 / 2; self-flags 24 / 512 = 4.69% (margins run-down 0.0966, starvation 0.1180); 0 contract failures; bar mean 3,196.5 units; ≈ 0.7 vCPU-h, M5-P ≈ 1.6 / 80.
- [ ] 4.2 [experiment][remote:cpu16] EXP-0081: L2 gradient probe. Reading `signal` / `weak` /
  `no_signal` (cosine 0.3 / 0.1); `no_signal` → DIAGNOSTICS entry, stop. ≈ 2, stop 4 (sizing.md).
  - Ran 2026-11-05 (EXP-0081): **stop branch.** Cosine undefined (minibatch B 24 / 24 disqualified), step `no_signal`. DIAGNOSTICS 25 disqualification-driven: rule off 0.840 / 0.902; rf3 re-keys 4 / 4 disqualified on B. Rule redesign goes to the user (new prereg version). ≈ 1.1 vCPU-h, M5-P ≈ 2.7 / 80.
  - 4.2b (2026-11-05): prereg v3 `97c3332c…` (flag-count guard, T = 5/3); EXP-0082 (offline re-score of EXP-0081's rows) → **`no_signal`** (−0.670 / −0.748), caused by the absolute death clause (flags-only 0.840); stop, to the user.
  - 4.2c (2026-11-05): prereg v4 `4a99c824…` (death scores units / B_we − 1); EXP-0083 (offline re-score) → **`no_signal`** (−0.239 / −0.161): the penalty (0.042 per death on the mean) is ≈ 2 SD of B's member spread; stop, to the user.

## 5. Pilot and main (experiment). Gated: EXP-0081 `signal` or `weak`

- [x] 4.2f [code][local] `entity_planner_buyahead_k4_rf4` = rf3 + never upgrade past the sustainable
  max (DIAGNOSTICS 29; supervisor decisions in notes/4.2f.md: view-only estimate, no cap when
  restore_effect = 0). Verify: the cap blocks over-ceiling / allows under-ceiling upgrades; rf4 ==
  rf3 where no over-ceiling upgrade happens; rf3 / k4 / entity_planner unchanged; goldens.
- [x] 4.2g [experiment][remote:cpu16] EXP-0084: rf4 vs rf3 (EXP-0079 rows; digests must match),
  × {MWWk R = 2, RR}, TTL 2, EXP-0057's worlds, observer on. **Pre-registered reading (2026-11-05,
  before data; as EXP-0079's; 12 producers decide, 42 reported):** per orderer, diff_w = mean
  paired (episode, replicate) units(rf4) − units(rf3) per world, floor_w = atlas v2 `noise_floor`;
  `rf4_helps` if mean_w(diff_w) > mean_w(floor_w) for BOTH orderers with no extra deaths and
  run-down check failures (r_end < 0.7516 r_start) up by ≤ 5 (absolute, per orderer); `harmful` if
  mean_w(diff_w) < −mean_w(floor_w) or extra deaths for EITHER orderer; `no_gain` otherwise.
  Report deaths, units at horizon 100, upgrades blocked, held stock, run-down, upgrades placed /
  completed. Whether rf4 becomes the base (and the L1 / probe redo) follows from the reading and
  goes to the supervisor / user. **Reported only, never deciding (supervisor decision 2026-11-05,
  before launch):** rf4 and rf3 also at horizon 200 on the same 23 12-producer worlds and keys,
  MWWk only (a `--dry-run` supply-trace run, horizon ≠ 100): deaths and units. ≈ 1.2 vCPU-h, stop 6;
  cpu16 pinned EUR-IS-1 (fallback 8, pinned), charged to M5-P. After 4.2f.
  - Done 2026-11-05 (EXP-0084): **`no_gain`** (12: +116.8 / +113.7 vs floor 146.5; 0 deaths; run-down down 5 / 3); 42 +326 / +367 vs 362; horizon 200 (reported) +8.3%, 0 deaths both; rf4 == rf3 where no block. The base decision goes to the supervisor / user. ≈ 0.9 vCPU-h, M5-P ≈ 3.6 / 80.
- [ ] 4.2h [code][local] Re-base on rf4 (user decision 2026-11-05 "Adopt rf4 as base"): the residual /
  ES / reads take rf4 (`upgrade_cap`); θ = 0 == rf4; rf3 paths kept (v1..v4 rebuild unchanged).
  Pre-registration v5 = v4's rules on base rf4, its flag threshold / sources from the rf4 L1 run;
  frozen after EXP-0085, before any probe data. Verify: θ = 0 == rf4 (incl. a capped episode);
  v1..v4 unchanged; v5 refused until calibrated; goldens.
- [ ] 4.2i [experiment][remote:cpu16] EXP-0085: L1 on rf4 (bar + noise floor, M4's pinned training
  set). Verify: r0 reproduces the bar; rf4 on 2 EXP-0057 world-episodes equals EXP-0084's rows;
  self-flags ≤ 5%; 0 contract failures; training-set deaths reported (12-0016 / 12-0004 expected
  none). ≈ 1.5, stop 3. Then (local, no pod up) `--mode flag-calibrate` → v5 constants → freeze v5.
  - Ran 2026-11-05 (EXP-0085): **L1 stop branch**: self-flags 28 / 512 = 5.47% > 5% (all other checks pass; 0 training deaths; flag-count calibration T = 5/3 at 4.27%). v5 not frozen; to the supervisor / user. ≈ 0.6 vCPU-h, M5-P ≈ 4.2 / 80.
- [ ] 4.2j [experiment][remote:cpu16] EXP-0086: a fresh L2 probe on rf4 under v5 (new rows). Reading
  as 4.2; `signal` / `weak` → 5.1 (L3 pilot, stop 14); else stop. ≈ 2, stop 4.

### Producer finalization (supervisor plan 2026-11-05; notes/producer-finalization.md)

- [x] PF-1 [experiment][local-read] EXP-0087: rf4's loss classes (offline read of EXP-0084's rows).
- [x] PF-2 [diagnostic][local] 12-0391: the cap's view estimate errs at a tiny max (DIAGNOSTICS 29).
- [x] PF-3 [code][local] rf5 = rf3 + the exact sustainable-upgrade cap. The producer's own degradation rate
  is an additive `CapacityRow.degradation_rate` that no existing encoder or policy reads; there is no cap when
  restore_effect = 0; rf4 is unchanged (supervisor decision). Verify: the exact rule allows 12-0391's p1_1 and
  blocks p2_2 / p2_1; existing policies act identically with the rate perturbed; goldens.
- [x] PF-4 [experiment][remote:cpu16] EXP-0089: rf5 vs rf3, using EXP-0079's rows (digests must match).
  × {MWWk R = 2, RR}, TTL 2, EXP-0057's worlds, observer on.
  - **Pre-registered reading (2026-11-05, before data; as EXP-0084's; 12 producers decide, 42 reported):**
    per orderer, diff_w = mean paired (episode, replicate) units(rf5) − units(rf3) per world, and
    floor_w = the atlas v2 `noise_floor`.
    - `rf5_helps` if mean_w(diff_w) > mean_w(floor_w) for BOTH orderers, with no extra deaths and run-down
      check failures (r_end < 0.7516 r_start) up by ≤ 5 (absolute, per orderer).
    - `harmful` if mean_w(diff_w) < −mean_w(floor_w), or extra deaths, for EITHER orderer.
    - `no_gain` otherwise.
  - **Reported only:** rf5 vs rf4 (EXP-0084's rows), the same statistics; horizon 200 rf5 vs EXP-0084's
    h200 rf3 / rf4 (MWWk, 12, dry run); upgrades blocked; deaths; held stock; run-down.
  - ≈ 1 vCPU-h, stop 6; cpu16 pinned EUR-IS-1 (fallback 8, pinned), charged to M5-P.
  - Done 2026-11-05 (EXP-0089): **`no_gain`** (12: +135.3 / +155.1 vs floor 146.5, 0 below, 0 deaths); vs rf4 +18.5 / +44.4; horizon 200 +9.1%. Holds, so PF-5 runs on rf5.
- [x] PF-5 [experiment][remote:cpu16] EXP-0088: buffer re-sweep on rf5 (it holds: EXP-0089), adding
  `entity_planner_buyahead_k{3,5,6}_rf5` (registrations only); k = 4 is rf5 (EXP-0089's rows, digests must
  match). × {MWWk R = 2, RR}, TTL 2, EXP-0057's worlds, observer on.
  - **Pre-registered k* rule (2026-11-05, before data; as EXP-0077's, with rf5 = k 4 as the reference):**
    per orderer o at 12 producers (42 reported):
    - eligible(k) = deaths(k) ≤ deaths(rf5), and run-down check failures (r_end < 0.7516 r_start) of k
      minus those of rf5 ≤ 5 (absolute);
    - best = the eligible k with the largest mean units per episode;
    - within(k) = mean_w(units_best − units_k) ≤ mean_w(floor_w), paired per world;
    - k*_o = the smallest eligible k with within(k); k* = max over the two orderers; none eligible → no k*.
  - **Reported:** units, held stock, run-down, deaths, investment rejection, upgrades blocked, and the late /
    died-upstream / contention classes per k.
  - **Stop rule (supervisor):** if k* = 4, the buffer step's gain is within the floor and the step stops there.
  - ≈ 2 vCPU-h, stop 4; cpu16 pinned EUR-IS-1 (fallback 8, pinned), charged to M5-P.
  - Done 2026-11-05 (EXP-0088): **k* = 5** at 12 (both orderers; k5 +106 / +106 over rf5; k6 best, at the edge, gains diminishing); 42 k* = 4; late inputs flat across k. ≈ 1.2 vCPU-h, M5-P ≈ 6.1 / 80.

### M5 on the final producer (supervisor 2026-11-05: `entity_planner_buyahead_k5_rf5`)

- [x] R-1 [code][local] Re-base M5 on k5_rf5. θ = 0 == k5_rf5 (buffer k = 5, so θ_k's offsets are around 5;
  the exact cap); the rf3 / rf4 paths are kept (v1..v5 rebuild unchanged); the reads compare against k5_rf5.
  Pre-registration v6 skeleton = v5's rules on k5_rf5, refused until its L1 constants are filled.
- [x] R-2 [experiment][remote:cpu16] EXP-0090: L1 on k5_rf5, as EXP-0085: bar + noise floor; r0 reproduces the
  bar; k5_rf5 on 2 EXP-0057 world-episodes equals EXP-0088's rows; training-set deaths; per-episode self-flags
  (reported); flag-count calibration (L1's criterion: ≤ 5% of re-keyed minibatches). ≈ 1.5, stop 3. Then,
  locally with no pod up, fill and freeze v6.
  - Done 2026-11-05 (EXP-0090): L1 passes (128 / 128, 2 / 2, 0 deaths, T = 5/3 at 3.35%); v6 frozen `5cd0e302…`; residual-base guard added (R-1 review). ≈ 0.9 vCPU-h, M5-P ≈ 7.0 / 80.
- [x] R-3 [experiment][remote:cpu16] EXP-0086: the L2 probe on k5_rf5 under v6 (fresh rows). `signal` /
  `weak` → the L3 pilot (stop 14); else stop. ≈ 2, stop 4.
  - Done 2026-11-05 (EXP-0086): **`signal`** (0.967 / 0.947), step `heterogeneity` (pilot at σ); 0 member deaths. Pilot (5.1) licensed. ≈ 1.3 vCPU-h, M5-P ≈ 8.3 / 80.
- [x] 5.1 [experiment][remote:cpu16] EXP-0082: L3 pilot, 20 generations, seed 0. `feasible` if
  g_tr > F_tr; else `unresolved` → DIAGNOSTICS 27's checks first. ≈ 7, stop 14 (sizing.md).
  - Done 2026-11-05 (EXP-0091, on k5_rf5 under v6): **`feasible`** (g_tr +467.5 vs F_tr 125.6); held stock 1.77× (stock_heavy), investment rejection 0.10 → 0.33; centre − k6_rf5 +402 (reported). ≈ 5.0 vCPU-h, M5-P ≈ 13.3 / 80.
- [x] 5.2 [experiment][remote:cpu16] EXP-0082 (resume): seed 0 to 40 generations; then seed 1 to
  40 generations (≈ 7 + 13.5, stops 14 / 27; sizing.md). Verify: g_tr recomputes; checkpoints and pins
  recorded.
  - Done 2026-11-05 (EXP-0092, k5_rf5, v6): both seeds to 40; g_tr +569.2 / +573.7 vs F_tr ≈ 223; 0 monitor deaths; stock ≈ 1.8×, rejection ≈ 3×. ≈ 14.0 vCPU-h, M5-P ≈ 27.3 / 80.
- [ ] 5.3 [experiment][remote:cpu16] EXP-0083: L5 validation at 12 and 42 (`success` / `plateau` /
  `kill` / `transfer_failure`). Stop branches → DIAGNOSTICS + user. ≈ 3.5, stop 7 (sizing.md).
  - Ran 2026-11-05 (EXP-0093, k5_rf5, v6): **`kill`** (1 extra death: cand-s0 on 12-0109); gains +413 / +383 at 12, +579 / +557 at 42; stock_heavy 1.58×. DIAGNOSTICS 30 (expansion orders dilute the fund; sole maker's restoration unfunded). Stop branch, to the supervisor / user. ≈ 1.5 vCPU-h, M5-P ≈ 28.8 / 80.

## 6. Held-out confirmation (experiment). Gated: EXP-0083 `success`

- [ ] 6.1 [experiment][remote:cpu16] EXP-0084: L6 held-out read (ledger pending entry committed
  before launch, closed after). `confirmed` → the user decides the roster and phase 2;
  `not_confirmed` → DIAGNOSTICS, user. ≈ 3.5, stop 7 (sizing.md).

## 7. Records (code)

- [ ] 7.1 [code][local] HISTORY §15 (M5 producer), README rows, spend against M5-P. Verify: every
  EXP-0079+ record has a README row and a §15 paragraph; spend totals equal the records' billed
  vCPU-h.

## Workflow follow-up

- **AskUserQuestion points:**
  - the design's open questions 1–7 (before group 2);
  - M5-P approval (before group 4);
  - stop branches;
  - after `confirmed`: roster and phase 2 (joint).
- **Phase 2 (joint potential game / GNN)** is a separate `/opsx:update` of this change after
  `confirmed`, citing the L6 result and an orderer-headroom diagnostic.
- **Pushes, champion promotion and gate / roster / registry changes** stay human gates.
