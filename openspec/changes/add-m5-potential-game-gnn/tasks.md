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
- [ ] 1.2 [experiment][remote:cpu16] EXP-0079: rf3 vs k4 (EXP-0077 rows; digests must match),
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

## 2. Residual policy and learner (code)

- [ ] 2.1 [code][local] `train/producer_residual.py`: the D2 residual (per-input k_g, extra buying,
  expansion trigger) over shared per-row features. The features are agent-local and past-only,
  with episode state weak-keyed by the system (7.22's fix). Verify:
  - θ = 0 equals the base action for action over an episode at 12 and at 42;
  - the parameter count is identical at 12 and 42;
  - an ADR-0058 check: features read only the producer's own view and history;
  - goldens.
- [ ] 2.2 [code][local] `train/producer_es.py`: ES on 2.1 reusing `order_es`:
  - bar precompute on the base, keyed draws, antithetic pairs, rank fitness, Adam, minibatch 24;
  - modes `probe` / `train` / `noise-floor`;
  - checkpoints and resume with pins.
  Verify: a horizon-12 dry run; resume refuses other pins; noise-floor mode reproduces the bar's
  base rows.
- [ ] 2.3 [code][local] Pre-registration `train/registry/es_prereg_m5_v1.json`: the D6 readings,
  thresholds, the D5 margin formula, the monitor set, σ, learning rate. Frozen by `--mode
  freeze`, pinned by SHA-256; real runs refuse a mismatch. Verify: freeze gives MATCH; a changed
  section is refused.
- [ ] 2.4 [code][local] Validation and held-out reads (L5 / L6) reusing `order_es_validate` and the
  M4 ledger design (D9 C0 / C1), with key `held-out/m5-producer`. Verify: the reading table per
  size; `transfer_failure`; ledger refusals; `--recompute` MATCH on a dry run.

## 3. Sizing (code)

- [ ] 3.1 [code][local] Timing probe: horizon-12 episodes of 2.1 at θ ≠ 0 at 12 and 42 against
  buy-ahead k4. Re-size design D9 and the stops. Verify: sizing.md lists the measured seconds,
  the estimates, the stops and a launch-fit table against M5-P.

## 4. Base and probe (experiment). Gated: tranche M5-P approved by the user

- [ ] 4.1 [experiment][remote:cpu16] EXP-0080: L1 bar + noise floor on the base (design D5).
  Verify: r0 reproduces EXP-0077 / EXP-0079 base rows on shared world-episodes; self-flags ≤ 5%
  with the margin; 0 contract failures. ≈ 3, stop 6.
- [ ] 4.2 [experiment][remote:cpu16] EXP-0081: L2 gradient probe. Reading `signal` / `weak` /
  `no_signal` (cosine 0.3 / 0.1); `no_signal` → DIAGNOSTICS entry, stop. ≈ 4, stop 8.

## 5. Pilot and main (experiment). Gated: EXP-0081 `signal` or `weak`

- [ ] 5.1 [experiment][remote:cpu16] EXP-0082: L3 pilot, 20 generations, seed 0. `feasible` if
  g_tr > F_tr; else `unresolved` → DIAGNOSTICS 27's checks first. ≈ 10, stop 20.
- [ ] 5.2 [experiment][remote:cpu16] EXP-0082 (resume): seed 0 to 40 generations; then seed 1 to
  40 generations (≈ 10 + 20, stops 20 / 40). Verify: g_tr recomputes; checkpoints and pins
  recorded.
- [ ] 5.3 [experiment][remote:cpu16] EXP-0083: L5 validation at 12 and 42 (`success` / `plateau` /
  `kill` / `transfer_failure`). Stop branches → DIAGNOSTICS + user. ≈ 6, stop 12.

## 6. Held-out confirmation (experiment). Gated: EXP-0083 `success`

- [ ] 6.1 [experiment][remote:cpu16] EXP-0084: L6 held-out read (ledger pending entry committed
  before launch, closed after). `confirmed` → the user decides the roster and phase 2;
  `not_confirmed` → DIAGNOSTICS, user. ≈ 6, stop 12.

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
