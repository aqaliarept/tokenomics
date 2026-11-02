# Tasks

Local budget (probed 2026-10-30): 2 cores, 3814/1137 MB RAM (total/available), 23 GB free disk, load 0.65.
Shared by supervisor + implementer + reviewer. Classes: local | local-exclusive | remote:<profile>.
Paths are under /srv/research/tokenomics/tokenomics/ unless absolute. "Full suite" = `cd /srv/research/tokenomics/tokenomics && ../.venv/bin/python -m pytest -q` (~2.5 min, local-exclusive). "Goldens" = `/srv/research/tokenomics/tokenomics/tests/test_legacy_invariance.py`, green and unedited in every code task. "Frozen hashes" = the v1 / v2 registry, roster and parameter files under `/srv/research/tokenomics/tokenomics/tokenomics/eval/registry/` keep their recorded content hashes (9.1 and 11.1 only add v3 files there). "Defaults stay v2" = `DEFAULT_REGISTRY`, `DEFAULT_ROSTER` and the default parameter file are the v2 files in every task of this change (design D8). "Atlas" = `/srv/research/tokenomics/experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json` (content hash `220886a7…`). "Checkpoints" = `/srv/research/tokenomics/experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt` (`d25388bf…`) and `/srv/research/tokenomics/experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt` (`8b51b77c…`). "Drafts" = `/srv/research/tokenomics/openspec/changes/add-gate-v3/drafts/` (draft registry v3, draft roster v3 and the pilot roster, written by 1.1 / 2.2). "Sizing" = `/srv/research/tokenomics/openspec/changes/add-gate-v3/sizing.md`. "v3 paths" = the packaged `worlds_v3.json`, `roster_v3.json` and `side_effect_params_v3.json`, always passed explicitly as `--registry / --roster / --params`.
Remote profiles (secure cloud only, never community): `cpu4` / `cpu8` / `cpu16` / `cpu32` = `--create-pod --image runpod/base:0.7.0-ubuntu2004 --cpu --python python3.12 --vcpu N --mem 2N --disk 30`. Largest shape first whose workers stay busy (jobs ≥ 2 × workers), `--workers` = vCPU, longest-first submission; a run bounded by a few long sequential jobs takes the smallest shape that holds them (sizing.md §EXP-0053 lesson). `--pull` nests the run in a subdirectory: move files up, `episodes.csv` into `artifacts/` (git-lfs). Stage inputs under the git-ignored `tokenomics/_pod_ckpts/` (never `_pod_inputs/`, which is tracked) and delete them after each run. `run_pod_training.py` has no hard-stop flag: the experiment implementer enforces each stop by pod-life wall clock.
Budget (design D10): every run has a hard stop = 2 × its probe estimate, ≤ 40 vCPU-h (auto-approved per CLAUDE.md). Ungated runs ≈ 39.5 billed vCPU-h (stops 79); gated reads ≈ 18.5 (stops 37). Spend counts against M4's budget under the supervisor's authority (user decision), but not inside tranche A, which has only 8.6 margin: gate v3 draws from tranche **G1** (ungated runs; proposed 65 vCPU-h) and **G2** (gated reads; sized after 11.2), released by the supervisor. The supervisor records the G1 decision in `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/notes/user-decisions.md` before 6.1. **Launch-fit rule per run:** billed so far in its tranche + its hard stop ≤ the tranche, recorded in the EXP record before launch; a miss goes to the supervisor (resize within the grant), and to the user only if the supervisor declines. A probe above its stop after the pre-registered reductions goes to the supervisor and the user; nothing launches above a stop. Nothing above 42 producers.
Human gates: the v1 / v2 gate files are never edited; v3 files enter `eval/registry/` only in 9.1 / 11.1, after the user approves, and the defaults stay v2 (gate-harness "Packaging a gate version does not move driver defaults", "Raising the bar is a user decision"). Groups marked **Gated** start only when the named condition is recorded.
Coordination with M4: every gate-v3 code commit changes M4's package source hash and `teacher_path_fingerprint` (design Risks). The supervisor lands gate-v3 code between M4 pod runs, and no M4 out-dir is resumed across a gate-v3 commit. Other implementers may have uncommitted files in the tree (e.g. `/srv/research/tokenomics/tokenomics/tokenomics/train/order_es_worlds.py`, `/srv/research/tokenomics/tokenomics/tokenomics/train/order_es.py` and their tests): never edit, stage or commit them. `--verify-source-commit` needs a clean synced tree, so pod launches wait until such files are committed by their owner or moved out of the synced tree by them.

## 1. Roster v3 draft (code)

- [x] 1.1 [code][local] Promote `round_robin`, `even_split_rotating`, `random_whole`, `makeable_weighted_whole` and `makeable_split` (design D1).
  - **Structure, no import cycle.** `routing_diagnostics` imports `baselines`, so move the five routers and their private helpers unchanged to a new module that imports neither (e.g. `/srv/research/tokenomics/tokenomics/tokenomics/eval/rotation_routers.py`). Register them in `ORDERING_BASELINES` in `/srv/research/tokenomics/tokenomics/tokenomics/eval/baselines.py`, and have `/srv/research/tokenomics/tokenomics/tokenomics/eval/routing_diagnostics.py` re-export them in `DIAGNOSTIC_ROUTERS` under the same names.
  - Add `STOCHASTIC_BASELINES = {random, random_whole, makeable_weighted_whole}`.
  - In `/srv/research/tokenomics/tokenomics/tokenomics/eval/roster.py` add `ROSTER_V3_ORDERING` and `build_roster_v3()` (88 members, stochastic = either policy in `STOCHASTIC_BASELINES`, R = 5). The stored `random` flags of v1 / v2 stay unchanged.
  - Write `roster_v3_draft.json` and the pilot roster `roster_v3_pilot_ep.json` (`entity_planner` × the 11 ordering baselines, unfrozen, named "pilot, not a gate roster") to Drafts.
  - Add one `COMBOS` row per promoted router to `/srv/research/tokenomics/tokenomics/tokenomics/eval/signal_check.py` (with `stock_aware`, like the v1 / v2 routers).
  - Update the `routing_diagnostics.py` docstring (promoted routers).
  - Tests in `/srv/research/tokenomics/tokenomics/tests/test_roster.py` and `/srv/research/tokenomics/tokenomics/tests/test_routing_diagnostics.py`:
    - the draft has 88 members, 32 of them stochastic, 216 runs per world-episode;
    - each promoted router's actions are byte-identical as a baseline and as a diagnostic router on a horizon-12 `shocks-i-smoke` episode;
    - v1 / v2 draft rosters and the Frozen hashes are unchanged;
    - each promoted router under `joint_stack` and `LEGACY` (`obs_stock_aware`) at horizon 12 gives zero contract failures;
    - stochastic members get `replicate_seed` action seeds;
    - a fresh interpreter imports each of `baselines`, `routing_diagnostics`, `roster` and `gate_cli` first without error.

  Verify: those tests and the Goldens pass; `python -m tokenomics.eval.signal_check` prints a row for each of the five routers.
- [x] 1.2 [code][local] Write the ADR (next free number after 0059) "Gate v3: promoted routers, region families, per-family side-effect parameters, per-order gate subject" in `/srv/research/tokenomics/docs/adr/`, from design D1–D8. Rejected alternatives: atlas worlds, unselected screen worlds, pooled X / Y, trained roster members, `ub`, moving the defaults during M4. After 1.1. Verify: the ADR cites EXP-0047 / 0048 / 0051 / 0052 / 0053 and add-m4-orderer-redesign D6 / D9; `grep -c` of "Status" = 1; no edits outside `docs/adr/`.

## 2. Registry v3 draft and per-family parameters (code)

- [x] 2.1 [code][local] Region samples in `/srv/research/tokenomics/tokenomics/tokenomics/eval/world_space.py`: a function that takes (stream, size, n, fixed categorical + shock levers) and returns `WorldPoint`s. Their continuous levers come from `lhs_columns(stream, size, n)`; choice / shocks / pattern / shock levers are the fixed ones (world-space "Region samples"). The `world_space` reproducer in `/srv/research/tokenomics/tokenomics/tokenomics/eval/task_archive.py` gains an optional `shock_profile` param that accepts only `"shocks-i"` with the neutral shock levers. Tests in `/srv/research/tokenomics/tokenomics/tests/test_world_space.py`:
  - fixed choice overrides the hypercube (f14 → 14 declarers per product);
  - the same inputs give the same config hashes;
  - `gate-v3-development`, `gate-v3-sealed`, `atlas-v1` and `m4-es-train-v1` share no seed over the first 4,096 indices at 12 and 42;
  - a shocked neutral-lever world's shock kinds equal the registered `shocks-i` profile's;
  - `shock_profile` with any other value is refused.

  After 1.1. Verify: tests pass in < 2 min; Goldens green.
- [x] 2.2 [code][local-exclusive] `build_worlds_v3()` in `/srv/research/tokenomics/tokenomics/tokenomics/eval/registry.py` (design D2).
  - **Contents.**
    - Every v2 entry is copied unchanged.
    - `ws_loaded` (12 producers; dense / sparse_f4 alternating by slot) and `ws_wide` (42; sparse_f14) per split, with the D2 counts and 4 reserves per regime.
    - Calm = shocks off + cyclic. Shocked = neutral shock levers + drift, generator param `shock_profile: "shocks-i"`.
    - Draws come from the 2.1 sampler on `gate-v3-<split>`, n = 128, accepted in index order when measured ρ ∈ (0.28, 3.1]. Rejections are recorded; an exhausted pool is refused.
    - Ids and episode seeds as in v2. Each region entry carries its world record.
  - **Registry changes the review listed** (`registry.py` 239–268, 307–320):
    - The seed-range check applies to legacy families only. Region seeds are checked by explicit set disjointness: development vs sealed, and against every v1 / v2 world and episode seed, every atlas-v1 sample / anchor seed and the first 4,096 `m4-es-train-v1` seeds.
    - `_check_generator` accepts `world_space` for region families, with params {producers, levers, shock_window, shock_profile?}.
    - New entry status `excluded` (a family × size that failed headroom, design D3 step 4). It applies to both splits, gate worlds and reserves; the screening record lists each exclusion with its headroom result. A v1 / v2 registry holding an `excluded` entry or a recorded exclusion is refused on load (version guard).
    - `admission_problems` / `coverage_problems` treat a recorded exclusion as a family × size the registry does not declare: no "headroom failed" problem and no coverage requirement. A failed headroom record without a matching exclusion is still refused.
  - **Version-aware validation.**
    - v1 / v2 admit `sparse_wide` only; v3 admits `ws_loaded` / `ws_wide` as well. The legacy `dense` family is refused.
    - `ws_wide` only at 42, `ws_loaded` only at 12.
    - Shocked region kinds equal `shocks-i`'s; the record hash must replay.
    - `coverage_problems` checks each family at its declared sizes from v3 on.
  - **Draft output.** `--write-draft --draft-version 3` writes `worlds_v3_draft.json` and `acceptance.csv` (draw index, slot, ρ, accepted) to Drafts.
  - **Tests** in a new `/srv/research/tokenomics/tokenomics/tests/test_registry_v3.py`, including:
    - the "World registry" scenarios: hashed region seeds load; a region seed under both splits is refused;
    - a synthetic frozen set excluding `sparse_wide`/12 passes development and sealed admission, with its sealed 12 entries marked `excluded`;
    - the same set with the failed record but no exclusion is refused;
    - an `excluded` entry or recorded exclusion in a v1 / v2 registry is refused;
    - `admission_problems` and `coverage_problems` give the same output as at the base commit on the packaged `worlds_v1.json` and `worlds_v2.json` (both splits).

  After 2.1. Verify: tests pass; the draft builds and reloads (hash check) in < 5 min; Frozen hashes unchanged; `acceptance.csv` shows every group filled.
- [x] 2.3 [code][local] Side-effect parameters per family and replicated-candidate checks in `/srv/research/tokenomics/tokenomics/tokenomics/eval/gate.py` (design D4, D6).
  - Parameter files from version 3 carry `{family: {x, y}}` with per-family evidence: value count, distinct-value count, quantile = minimum flag, and `small_sample` when there are < 20 distinct values.
  - `calibrate_x_y` runs per family with v2's nearest-rank q rule; sparse_wide is pooled over sizes and recomputed with the v3 roster.
  - The run-down and starvation checks use the world's family.
  - **Replicated candidates (v3+ parameters).** Run-down and starvation are judged on the calibration's replicate means; per-replicate violations go to the monitored metrics. Deaths are checked per replicate.
  - v1 / v2 scalar files, and their per-replicate checks, are read unchanged.
  - Tests in `/srv/research/tokenomics/tokenomics/tests/test_gate.py` and `/srv/research/tokenomics/tokenomics/tests/test_gate_versions.py`:
    - family values are applied;
    - a v3 file missing a family present in the registry is refused;
    - the "One replicate starves a consumer" and "One replicate dies" scenarios hold;
    - distinct counts are correct on triplicated calm rows;
    - `--mode recompute` of EXP-0045's and EXP-0047's recorded dirs still gives MATCH.

  After 2.2. Verify: tests pass; Goldens green.

## 3. Screen modes and size-scoped reports (code)

- [ ] 3.1 [code][local] Two-wave v3 screen in `/srv/research/tokenomics/tokenomics/tokenomics/eval/gate_cli.py` and `/srv/research/tokenomics/tokenomics/tokenomics/eval/roster.py` (design D3), selected by `--member-waves v3` and recorded in `spec.json`.
  - **Member plan.** Wave 1 = all members on every development gate world except `ws_wide`. Wave 2 = `ws_wide`, with the wave-1 winners ∪ `entity_planner` × every ordering baseline. Reserves on demand per wave.
  - **Replacement.** Bar-death replacement for region families.
  - **Staged headroom with per-family admission.** Wave-1 families are checked before wave 2. A failing family × size is marked `excluded` in both splits (design D3 step 4). Exit 3 only when a size has no admitted family, before wave 2 if wave 1 decides it.
  - **Excluded rows are reported only.** Fixed before W2 is built (wave 1) or before pruning (wave 2), they are not in the W2 winner set, the never-argmax evidence or any X / Y.
  - **Pruning.** Rule `never-argmax-two-wave-v1`; the record holds waves, members, episodes, digests and the not-carried members.
  - **Reported, never gating:**
    - attainability per (size, regime) and per family, with the `unattainable` flag;
    - the per-world routing reading: `reach_gain_w` against 0.05 and the world's floor, the family rule `routing_matters` iff ≥ 1/3 of admitted worlds count, and `router_spread_w` as a column only;
    - v2 against v3 X / Y per family, and the v2-roster bar beside the v3 bar on v2 worlds.
  - **Output.** Per-family X / Y (2.3); `frozen/*_v3.json` with `screened_on` = the draft hash.
  - **Reductions.** The D3 reductions are flags recorded in `spec.json`: `--region-shocked-episodes 5`, and `--reuse-rows <EXP-0047 dir>` with its reproduction check.
  - **Tests** in `/srv/research/tokenomics/tokenomics/tests/test_gate_screen.py` (pure, on synthetic rows): wave-2 membership, carried winners, bar-death replacement order, per-family exclusion vs exit 3, the three "excluded" scenarios of the gate-harness delta (both splits, admission, rows), attainability flag, routing reading (including "Poor routers do not make routing matter"), pruning record.
  - **Smoke.** `--dry-run --horizon 12` with `shocks-i-smoke`, 1 world per family, `--max-members 6`, `--workers 1`.

  After 2.3. Verify: tests pass; the smoke finishes in < 2 min and writes `screening.json` naming both waves and the attainability and routing tables; v3 frozen outputs from the smoke carry one `screened_on` and are marked "not an acceptance read".
- [ ] 3.2 [code][local] Sealed screen by group and size-scoped real reports (design D5).
  - `--mode screen-sealed --sizes 12` runs the frozen roster alone on the sealed gate worlds of the named sizes: zero-baseline + bar-death replacement only, no pruning or calibration. It keeps `screened_on`, records the screened (split, size) groups (v1 / v2 records read as whole-split screens) and appends to the access log.
  - `--scope-sizes` on a real report: `gate_sizes` = the scope, so no margin failure for an unscoped size. The scope is recorded in `spec.json` and read back by `--mode recompute`; a sealed read needs a manifest binding the same scope. Without it, the report covers every gate size.
  - A scoped report needs its (split, size) groups screened and admitted; the real-report filter refusal (`gate_cli.py` 723–735) stays for `--sizes`.
  - The sealed screen and sealed reports skip `excluded` entries.
  - Tests in `/srv/research/tokenomics/tokenomics/tests/test_gate_screen.py`, `/srv/research/tokenomics/tokenomics/tests/test_gate_versions.py` and `/srv/research/tokenomics/tokenomics/tests/test_gate.py`, covering the "Size-scoped real report" and "Sealed split screened by size" scenarios. They also cover the synthetic one-exclusion set from 2.2 through the sealed screen and a 12-producer scoped sealed dry-run report: excluded entries are never run, and admission passes.

  After 3.1. Verify: tests pass; a horizon-12 dry run on 1 sealed world finishes in < 2 min; roster / parameter hashes unchanged by it.

## 4. Per-order gate subject (code)

- [ ] 4.1 [code][local] `--subject per-order` in `/srv/research/tokenomics/tokenomics/tokenomics/eval/gate_cli.py` (design D6).
  - **Inputs.** `--orderer-ckpt` composite (`per-order-composite-v1`) and `--deciding-producer` (a white-box producing baseline, default `entity_planner`).
  - **Env and replicates.** Candidate envs come from `per_order_scorer.per_order_env` (choice-v1 + agent-local history). R = the roster's R replicates, with `replicate_seed` + `set_draw_key(DrawKey(world, episode, r))`. Deciding replicates get twins on shocked worlds.
  - **Pins.** `per_order_scorer.per_order_pins(policy)` (decode pin included, per the binding carry-forward item in `/srv/research/tokenomics/openspec/changes/add-m4-orderer-redesign/notes/carry-forward.md`), plus the gate pins: composite SHA-256 / format, M4 pre-registration hash, deciding producer, R, scope. They go into `spec.json` and the resume check.
  - **Contract failures.** `NonFiniteChoice` and `validate_choice` violations are `job` rows.
  - **Refusals.** A composite under `logits-v1`; a legacy scorer as `choice-v1`; R ≠ the roster's R or a non-white-box deciding producer in a real report.
  - **Tests** in `/srv/research/tokenomics/tokenomics/tests/test_gate_cli.py` and `/srv/research/tokenomics/tokenomics/tests/test_gate_contract.py`, using a θ = 0 (A0) composite built in the test:
    - resume refuses a changed base hash;
    - a choice-v1 report never resumes onto a logits-v1 out-dir, nor the reverse;
    - every `per_order_pins` key is in the report's pins;
    - a forced non-finite score is one contract failure and the run is invalid;
    - a keyed policy without a key is a contract failure;
    - one dying replicate fails zero-death, naming the replicate.

  After 3.2. Verify: tests pass; a `--dry-run --horizon 12` report with the A0 composite on 1 world per family (`--replicates 2`, `--workers 1`) finishes in < 2 min, and `--mode recompute` gives MATCH.
- [ ] 4.2 [code][local-exclusive] Reported column and sealed manifest (design D6).
  - `--reported-prod-ckpt` adds subject `candidate:reported` rows (same orderer, seeds and keys; no twins), split off before the verdict.
  - Output: `reported.csv` (per world and regime: ratio against the same hindsight baseline, deaths, hard checks) and a "reported, not gating" block in `verdict.json` / `RESULT.md`.
  - `--mode recompute` gives MATCH only if the verdict, resilience and reported tables all match.
  - The sealed pre-registration loader requires the same pin set as 4.1, plus the reported producer SHA-256 and the scope.
  - Update the gate bullets of `/srv/research/tokenomics/ARCHITECTURE.md` §7.2 for the new subject (v3 itself is documented in 9.1).
  - Tests in `/srv/research/tokenomics/tokenomics/tests/test_gate_cli.py` and `/srv/research/tokenomics/tokenomics/tests/test_gate.py`: a reported miss leaves the verdict unchanged; a manifest missing the normaliser hash or the scope is refused; recompute gives MISMATCH when `reported.csv` is edited.

  After 4.1. Verify: tests pass; the 4.1 dry run with `--reported-prod-ckpt` (the EXP-0035 Checkpoint) recomputes to MATCH; Full suite green.

## 5. Analysis script, timing probes and teacher options (code)

- [ ] 5.1 [code][local] Offline analysis script `/srv/research/tokenomics/openspec/changes/add-gate-v3/probes/attainability_v3.py` (stdlib only, runs in seconds). From gate-format `episodes.csv` rows it computes:
  - headroom per family × size under a given member set;
  - perfect-fill ratios per (size, regime, family);
  - the per-world routing reading and family rule (design D3 step 4: `reach_gain_w` ≥ 0.05 and above the floor, family `routing_matters` at ≥ 1/3). The floor follows D3's three pre-registered cases: shocked = 3 · sd(paired episode differences) / √n_ep; calm with a stochastic side = 3 · √(Σ s_side² / n_side) over episode × replicate runs; calm with both sides deterministic = 0. Unit tests on synthetic rows cover each case.

  Validate it on EXP-0047 + EXP-0048 rows, and the routing rule on EXP-0051's screen rows: it must reproduce the D3 table of `/srv/research/tokenomics/openspec/changes/add-gate-v3/probes/routing_rule_check.py` (42 sparse_f14 0.75 matters, 12 sparse_f2 0.05 flat, 42 sparse_f2 0.00 flat). On the gate rows it must reproduce the review's 0.9414 / 0.8921 median fill and the 1.052 / 1.099 / 1.097 / 1.137 perfect-fill table (design Context), and v2's 0.9037 / 0.8749. After 3.1 (same rules as the screen; a test asserts the script and `gate_cli`'s tables agree on the 3.1 smoke rows). Verify: the reproduced numbers are written to Sizing §attainability.
- [ ] 5.2 [code][local] Screen and pilot timing probe `/srv/research/tokenomics/openspec/changes/add-gate-v3/probes/timing_screen_v3.py` (`--workers 1`, < 2 min per command).
  - Episodes: one horizon-100 episode each on one calm and one shocked development world of `ws_loaded` (one dense, one f4) and of `ws_wide` from Drafts, plus one v2 world as the 0.646 check.
  - Members: `entity_planner` × {`round_robin`, `proportional`, `makeable_weighted_whole`, `makeable_split`}, `noop` × `even_split`, `random` × `random`.
  - Extrapolate the pilot (design D3 step 0) and the screen's wave 1 and wave 2 into Sizing §pilot / §screen: pod vCPU-h, billed on cpu32 / cpu16, the longest job, stop = 2 × billed, and which D3 reduction (if any) applies so the stop is ≤ 40.

  After 2.2. Verify: raw timings in `probes/timing_screen_v3.jsonl`; Sizing states estimate, shape, wall and stop for both runs; the shocked `ws_wide` world's units equal a rerun's (determinism).
- [ ] 5.3 [code][local] 42-producer teacher options in `/srv/research/tokenomics/tokenomics/tokenomics/eval/teacher_audit.py` (design D7).
  - `--size 42` registers the sample: 42-0054, 42-0076, 42-0073, 42-0126, 42-0113, 42-0122, each at the confirm's first episode seed.
  - Arms: `floor`, `d2`, the bar / reach reruns, and PoC-C S / A / B on the shocked worlds (T = the run's `d2` row). No `ub`.
  - `--reduction 42-L0…L5` applies the ladder. `--candidate-cap 4` (trained row + the top-3 `makeable_qty_now` eligible producers, ties by id, the floor's receiver excluded) and `--stream-cap 8` (largest consumer order quantity first, ties by order id) are accepted in a real run only as the registered values of the chosen step.
  - Readings:
    - `above_bar` / `within_floor` / `below_bar` per world against the bar, MWW, RR and TR;
    - `teacher_headroom_42`;
    - c_O / c_P with the 0.15 band, "to the user" when c_P ≥ c_O or tied;
    - the atlas ceilings beside `d2`, with `d2` labelled a lower bound;
    - the applied steps in every reading.
  - Existing refusals unchanged (> 42, non-train, hit / switch).
  - Tests in `/srv/research/tokenomics/tokenomics/tests/test_teacher_audit.py`:
    - sample membership; a validation world is refused;
    - the caps select the documented candidates / streams on a fixture state;
    - the 12-producer defaults are byte-identical.
  - Dry-run smoke on 42-0054 at `--horizon 4 --reduction 42-L3 --max-search-actions 2`, `--workers 1`.

  After 1.1. Verify: tests pass; the smoke finishes in < 2 min; Goldens green.
- [ ] 5.4 [code][local] Teacher timing probe `/srv/research/tokenomics/openspec/changes/add-gate-v3/probes/timing_teacher42.py`, adapted from `/srv/research/tokenomics/openspec/changes/archive/2026-10-30-add-m3-development-audit/probes/timing_63.py` / `timing_82.py`. It runs `--workers 1`, with a per-command time budget so each command stays < 2 min; remaining streams are counted, not timed.
  - **Cells.** 42-0054 (posterior, shocked) and 42-0126 (posterior, iid), cycles 30 / 60 / 90, all candidates and K = 4, for T (`entity_planner`) and A (portable producer).
  - **Measured:**
    - s per searched stream and search s per cycle;
    - orderer turns and streams per cycle;
    - the scenario mix;
    - the posterior acceptance rate and the **unresolved fraction** of searched decisions;
    - floor-speed and S / B episode times.
  - **Extrapolation.** Every ladder step (D7) goes into Sizing §teacher42: per-episode pod minutes, the longest job, billed vCPU-h and wall per shape (cpu4 / cpu8 / cpu16), and stop = 2 × billed. Pick the first step that fits (2 × billed ≤ 40 vCPU-h and longest job × 2 ≤ the stop wall of the smallest shape holding the long jobs).
  - **No launch** is recorded if L5 does not fit, or if ≥ 0.5 of the searched decisions on either probe world are unresolved.

  After 5.3. Verify: raw timings in `probes/timing_teacher42.jsonl`; every sample world is priced from its own cells or a stated, measured driver (no pooled pricing, sizing.md §EXP-0053 concern #346); Sizing names the chosen step and shape, or "no launch". On "no launch" the task reports to the supervisor and the user, and 8.1 does not start.

## 6. Pre-screen pilot (experiment)

- [ ] 6.1 [experiment][remote:cpu16] Stage Drafts' `worlds_v3_draft.json` and `roster_v3_pilot_ep.json` under `/srv/research/tokenomics/tokenomics/_pod_ckpts/`. Run `python -m tokenomics.eval.gate --mode screen --dry-run --registry /workspace/tokenomics/_pod_ckpts/worlds_v3_draft.json --roster /workspace/tokenomics/_pod_ckpts/roster_v3_pilot_ep.json --reserves none --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/pilot_v3` (every development gate world, no filters) into `/srv/research/tokenomics/experiments/EXP-NNNN-gate-v3-pilot/`. Then run `probes/attainability_v3.py` on its rows. After 5.1 and 5.2. Verify:
  - `contract_failures.csv` is header-only;
  - the `entity_planner` rows of the v2 routers and of RR / ESR / RW on v2 worlds reproduce EXP-0047's / EXP-0048's rows exactly;
  - the attainability, headroom and routing tables are written, also when the dry-run screen exits 3 on a headroom failure (the rows are the result);
  - the record says "not an acceptance read; prunes and calibrates nothing";
  - the launch-fit check against tranche G1 is recorded before launch (0 + 14 ≤ G1).
  - Resources: cpu16 fresh pod (--create-pod), ≈ 20 min driver wall, ≈ 7 billed vCPU-h (≈ 5.1 pod vCPU-h of compute; design D3 pilot estimate, sized by task 5.2), hard stop 14 vCPU-h (≈ 52 min of cpu16 pod life); runs commit after 5.1 and 5.2; stage the Drafts' draft registry and pilot roster; pull /workspace/tokenomics/runs/pilot_v3 → experiments/EXP-NNNN-gate-v3-pilot/; --verify-source-commit yes.

  Records: `RESULT.md`, plain English first. It says what the pilot asked (does the v3 bar leave headroom and attainable margins on each family, and do routers differ on the drafted region worlds), the numbers per family, size and regime, and what they mean for the region weight. Also a README row and an opening HISTORY §15 paragraph. Then the supervisor asks the user (AskUserQuestion) design Open Questions 1–3 with these numbers. Group 7 starts only after the answers are recorded; an answer that changes D2 / D3 goes through `/opsx:update` first.

## 7. v3 development screen (experiment)

- [ ] 7.1 [experiment][remote:cpu32] Stage Drafts' `worlds_v3_draft.json` + `roster_v3_draft.json` (and, only if Sizing §screen chose reduction 2, EXP-0047's `episodes.csv`) under `/srv/research/tokenomics/tokenomics/_pod_ckpts/`. Run `python -m tokenomics.eval.gate --mode screen --registry /workspace/tokenomics/_pod_ckpts/worlds_v3_draft.json --roster /workspace/tokenomics/_pod_ckpts/roster_v3_draft.json --member-waves v3 --prune --q 0.05 --workers 32 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/screen_v3` (plus the Sizing-chosen reduction flags) into `/srv/research/tokenomics/experiments/EXP-NNNN-registry-v3-screen/`. After 6.1 and the user's answers. Verify:
  - `contract_failures.csv` is header-only;
  - every planned row is present once, with matching digests;
  - headroom per family × size was decided after its wave; any `excluded` family is listed, and exit 3 (no admitted family at a size) is itself the result, recorded and diagnosed before anything else;
  - the three `frozen/*_v3.json` share `screened_on` = the draft hash and reload;
  - the pruning record lists both waves;
  - X / Y per family recompute from the rows, with distinct counts;
  - the attainability and routing tables match `probes/attainability_v3.py` on the same rows;
  - v2-world bests are reported against the v2 roster's on the same rows;
  - the launch-fit check against tranche G1 is recorded before launch.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 35 min driver wall, ≈ 18 billed vCPU-h (16.4 compute; design D3, sized by task 5.2, grounded on EXP-0047's 1.125 pod-s per run and sizing.md §EXP-0051), hard stop 36 vCPU-h (≈ 67 min of cpu32 pod life); runs commit after 5.4; stage the Drafts (+ EXP-0047 episodes if reduction 2); pull /workspace/tokenomics/runs/screen_v3 → experiments/EXP-NNNN-registry-v3-screen/; --verify-source-commit yes.

  Records: `RESULT.md`, plain English first. It covers:
  - what the new roster and worlds are;
  - which promoted routers survived pruning and on which worlds they set the bar;
  - how much the bar rose on v2 worlds;
  - per-family headroom, admission, attainability and routing reading;
  - v2 against v3 X / Y per family, with sample-size flags;
  - the replaced and excluded worlds;
  - the limits of two-wave pruning and of the region definition.

  Also a README row and a HISTORY §15 paragraph. Then stop for the user's approval of v3 (AskUserQuestion), with the bar, attainability and routing evidence.

## 8. 42-producer teacher read (experiment)

- [ ] 8.1 [experiment][remote:cpu4] Stage the Atlas, both Checkpoints, the packaged `roster_v2.json` and EXP-0052 / EXP-0057 `episodes.csv` (pairing references) under `/srv/research/tokenomics/tokenomics/_pod_ckpts/`. Run `python -m tokenomics.eval.teacher_audit --size 42 --reduction <step from Sizing §teacher42> --atlas /workspace/tokenomics/_pod_ckpts/atlas_v2.json --roster /workspace/tokenomics/_pod_ckpts/roster_v2.json --orderer-ckpt /workspace/tokenomics/_pod_ckpts/orderer_scorer.pt --prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt --workers <vCPU> --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/teacher42` into `/srv/research/tokenomics/experiments/EXP-NNNN-teacher-42/`. After 5.4, with Sizing naming a launchable step. Independent of groups 6–7. Verify:
  - `contract_failures.csv` is header-only;
  - digests are equal across arms per world-episode;
  - the bar / reach / RR / TR rows reproduce EXP-0052 / EXP-0057 rows for the same world-episodes;
  - readings recompute from `episodes.csv`;
  - every reading names the applied ladder step;
  - there are no `ub` rows;
  - the launch-fit check against tranche G1 is recorded before launch, with the probe-set stop.
  - Resources: shape chosen by task 5.4 (prior: cpu4 fresh pod (--create-pod), ≈ 3 h wall, ≈ 12 billed vCPU-h at step L4, extrapolated from sizing.md §EXP-0053 × §EXP-0051 in design D7 and replaced by the probe), hard stop = 2 × the probe's billed estimate, ≤ 40 (prior 24); runs commit after 5.4; stage Atlas + Checkpoints + roster v2 + EXP-0052 / EXP-0057 episodes; pull /workspace/tokenomics/runs/teacher42 → experiments/EXP-NNNN-teacher-42/; --verify-source-commit yes.

  Records: `RESULT.md`, plain English first. It covers:
  - what `d2` is: a lower bound and a reference, never a label source;
  - per world, `d2` against bar / MWW / RR / TR, relative to the floor;
  - the room closed against the atlas ceilings;
  - the posterior's unresolved share;
  - the PoC-C closures and what they say about the role at 42;
  - every applied reduction step and what it weakens.

  Also a README row and a HISTORY §15 paragraph. If c_P ≥ c_O or a tie: AskUserQuestion (M4's role is unchanged meanwhile).

## 9. Package v3 (code). Gated: the user approves v3 after 7.1

- [ ] 9.1 [code][local-exclusive] Copy the screen's `frozen/{worlds,roster,side_effect_params}_v3.json` to `/srv/research/tokenomics/tokenomics/tokenomics/eval/registry/` unchanged.
  - Add the constants `REGISTRY_V3`, `ROSTER_V3` and `PARAMS_V3`. **The defaults stay v2** (design D8).
  - Document the explicit v3 paths in the CLAUDE.md Gate harness bullet (`/srv/research/tokenomics/CLAUDE.md`), `/srv/research/tokenomics/ARCHITECTURE.md` §7.2 and the ADR's status.
  - Tests in `/srv/research/tokenomics/tokenomics/tests/test_gate_versions.py`:
    - the packaged v3 hashes equal the EXP dir's;
    - `version_set_problem` admits the v3 set and refuses v3 mixed with v1 / v2 files;
    - `DEFAULT_REGISTRY` / `DEFAULT_ROSTER` / the default parameter file are still v2;
    - the import-time and real-run checks of `train/order_es_worlds.py`, `train/order_es.py` (its scalar X / Y disqualification guard), `eval/split_need.py`, `eval/teacher_audit.py`, `eval/decoder_check.py`, `eval/atlas.py` and `eval/routing_audit.py` are unchanged (they still load the v2 parameters and accept / refuse the same rosters), tested without editing those modules;
    - EXP-0045 / EXP-0047 recompute gives MATCH.

  After 7.1 and the user's approval. Verify: Full suite green; Frozen hashes of v1 / v2 unchanged.

## 10. Sealed 12-producer screen (experiment). Gated: 9.1

- [ ] 10.1 [experiment][remote:cpu16] Run `python -m tokenomics.eval.gate --mode screen-sealed --sizes 12 --registry <v3 paths> --roster <v3 paths> --params <v3 paths> --workers 16 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/sealed12_v3` into `/srv/research/tokenomics/experiments/EXP-NNNN-sealed-12-screen-v3/`. After 9.1. Verify:
  - `contract_failures.csv` is header-only;
  - the screened registry keeps `screened_on`, and the roster / parameter hashes are unchanged;
  - every 12-producer sealed (split, size) group is marked screened;
  - replacements (if any) follow the reserve order;
  - the access log has the entry;
  - the launch-fit check against tranche G1 is recorded before launch.
  - Resources: cpu16 fresh pod (--create-pod), ≈ 15 min wall, ≈ 2.5 billed vCPU-h (192 world-episodes × ≈ 25 runs at the 12-producer per-run cost measured by task 5.2 and the pilot, plus setup), hard stop 5 vCPU-h; runs commit after 9.1; stage nothing (packaged files); pull /workspace/tokenomics/runs/sealed12_v3 → experiments/EXP-NNNN-sealed-12-screen-v3/; --verify-source-commit yes.

  Records: `RESULT.md` plain English first; README row; HISTORY §15 line. The screened registry is packaged by task 11.1 (a gate file: human gate).

## 11. Sealed packaging and gate-read preparation (code)

- [ ] 11.1 [code][local-exclusive] Package the sealed-screened registry from 10.1's EXP dir as `worlds_v3.json` in `/srv/research/tokenomics/tokenomics/tokenomics/eval/registry/`, with the same `screened_on`. Record the new registry version in the ADR and `/srv/research/tokenomics/CLAUDE.md`. Defaults stay v2. Human gate: the user approves the replacement of a gate file. After 10.1 and that approval. Verify: Full suite green; `version_set_problem` admits the set; the 12-producer sealed groups read as screened; v1 / v2 Frozen hashes unchanged.
- [ ] 11.2 [code][local] Timing probe of the real candidate. **Gated:** add-m4-orderer-redesign EXP-0061 reads `confirmed`, and 9.1 is done (the exit read is the sealed 12-producer read, user decision). One horizon-100 episode per (size, family, regime) with the M4 composite at R = 1 under `entity_planner` and under the EXP-0035 producer (`--workers 1`, < 2 min per command). Extrapolate the development report (design D9.1) and the sealed-12 read (D9.2) into Sizing §gate-read with shapes and stops. After the gate condition. Verify: raw timings recorded; Sizing gives both estimates and stops ≤ 40.
- [ ] 11.3 [code][local] Write the sealed pre-registration manifest for the M4 candidate under `/srv/research/tokenomics/tokenomics/tokenomics/eval/manifests/`, with its loader test. It binds the full 4.1 / 4.2 pin set, the packaged v3 versions from 11.1, split sealed and scope `--scope-sizes 12`. After 11.1 and 11.2. Verify: the loader test passes; the manifest's hashes equal the files to be staged.

## 12. Gated reads of the M4 candidate (experiment). Gated: as 11.2

- [ ] 12.1 [experiment][remote:cpu32] Development report on v3: `python -m tokenomics.eval.gate --mode report --subject per-order --registry <v3 paths> --roster <v3 paths> --params <v3 paths> --orderer-ckpt /workspace/tokenomics/_pod_ckpts/<composite>.pt --deciding-producer entity_planner --reported-prod-ckpt /workspace/tokenomics/_pod_ckpts/producer_candidate.pt --split development --workers 32 --source-commit {SOURCE_COMMIT} --out-dir /workspace/tokenomics/runs/gate_v3_dev` into `/srv/research/tokenomics/experiments/EXP-NNNN-m4-gate-dev-v3/`. After 11.2. Verify: valid run (no contract failures, no missing rows); `--mode recompute` gives MATCH (verdict, resilience, reported); the pins equal the M4 candidate's recorded `per_order_pins` and hashes; the launch-fit check against tranche G2 is recorded before launch.
  - Resources: cpu32 fresh pod (--create-pod), ≈ 30 min wall, ≈ 14 billed vCPU-h (design D10, from EXP-0057's per-job cost and sizing.md §EXP-0051; re-sized by task 11.2), hard stop 28 vCPU-h; runs commit after 11.2; stage the composite + its base / normaliser + the EXP-0035 Checkpoint; pull /workspace/tokenomics/runs/gate_v3_dev → experiments/EXP-NNNN-m4-gate-dev-v3/; --verify-source-commit yes.

  Records: `RESULT.md` plain English first (the deciding verdict at 12 and 42, the reported EXP-0035 column, per-family tables, resilience); README row; HISTORY §15 paragraph.
- [ ] 12.2 [experiment][remote:cpu16] Sealed 12-producer exit read: the 12.1 command with `--split sealed --scope-sizes 12 --preregistration <11.3 manifest> --out-dir /workspace/tokenomics/runs/gate_sealed12`, into `/srv/research/tokenomics/experiments/EXP-NNNN-m4-gate-sealed-12/`. After 11.3 and 12.1. Verify: valid run; recompute gives MATCH with the recorded scope; an access-log entry exists; the manifest's pins and scope equal `spec.json`'s; the verdict covers 12 producers only; the launch-fit check against tranche G2 is recorded before launch.
  - Resources: cpu16 fresh pod (--create-pod), ≈ 20 min wall, ≈ 4.5 billed vCPU-h (design D10, re-sized by task 11.2), hard stop 9 vCPU-h; runs commit after 11.3; stage the composite + components + the EXP-0035 Checkpoint; pull /workspace/tokenomics/runs/gate_sealed12 → experiments/EXP-NNNN-m4-gate-sealed-12/; --verify-source-commit yes.

  Records: `RESULT.md` plain English first (the exit verdict per regime, the reported column, what passing or failing means for M4); README row; HISTORY §15 and §14 lines. The verdict goes to the user (human gate: promotion).

## 13. Close (code)

- [ ] 13.1 [code][local] Write the HISTORY §15 "Gate v3" summary and update §0: what changed in the bar and the worlds, the pilot and the teacher read at 42, the gated reads' status, and spend per run against each stop. After the last executed task. Verify: every EXP of this change has a README row and a §15 paragraph; spend totals equal the sum of the records' billed vCPU-h.

## Workflow follow-up

- Each task goes through the cross-harness review (`supervise` / `review-orchestrator`) before it is ticked.
- AskUserQuestion points:
  - design Open Questions 1–3 (with the pilot's numbers) after 6.1, before group 7;
  - v3 approval after 7.1 (before 9.1);
  - a 5.4 "no launch";
  - an 8.1 c_P ≥ c_O or tie;
  - the sealed packaging before 11.1;
  - the exit read's verdict after 12.2.
- Moving the packaged defaults to v3 is a follow-up after M4 closes (design D8): it pins every driver that reads a default first, with tests, then needs the user's approval.
- Sync the `gate-harness`, `development-audit` and `world-space` specs, and archive the change once 13.1 is written. Tasks 11.2–12.2 may close as "not run" if M4 does not reach `confirmed`.
- Pushing `new-model`, packaging v3 (9.1 / 11.1) and champion promotion stay human gates.
