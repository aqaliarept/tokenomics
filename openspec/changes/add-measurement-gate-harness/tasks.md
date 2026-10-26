# Tasks

Local budget (probed 2026-10-26): 2 cores, 3814/2548 MB RAM (total/available), 26 GB free disk, load 0.38.
Shared by supervisor + implementer + reviewer. Classes: local | local-exclusive | remote:<profile>.
Paths are under /srv/research/tokenomics/tokenomics/ unless absolute. "Full suite" = `cd /srv/research/tokenomics/tokenomics && ../.venv/bin/python -m pytest -q` (~2.5 min on this box).

## 1. Legacy-invariance safety net

- [x] 1.1 [code][local] Add `/srv/research/tokenomics/tokenomics/tests/test_legacy_invariance.py` and `/srv/research/tokenomics/tokenomics/tests/data/legacy_goldens.json`, captured on the unmodified tree before any other change. The goldens cover every existing `episode_stats` key and `capacity_trace`, for a fixed set of (fixture world, baseline pair, seed) cells at horizon ≤ 30 on the legacy and joint env contracts, plus `build_scale_config(seed)` output for seeds 0–3. Verify: the new test passes on the unmodified tree and runs in ≤ 60 s.

## 2. Domain: outcome records and cyclic demand

- [x] 2.1 [code][local-exclusive] Tag each consumer order with its placement cycle. Add a tuple of per-order consumer outcomes (consumer, product, quantity, tokens at placement, placement cycle, completed/rejected) to `CycleResult` in `/srv/research/tokenomics/tokenomics/tokenomics/domain/system.py` (design D2). After 1.1. Verify: new domain tests on a hand-built economy where one order completes and one is rejected; the full suite, including 1.1, is green.
- [x] 2.2 [code][local-exclusive] Add `ConsumerConfig.schedule` in `/srv/research/tokenomics/tokenomics/tokenomics/domain/config.py`, with validation (products exist, quantities ≥ 1), `load_config` JSON support and a `config_to_dict` round-trip. When it is set, the env builds a `ScheduledConsumer` from it (design D8). Verify: tests for the three "Deterministic cyclic demand" scenarios; the full suite, including 1.1, is green.

## 3. Env: accounting and side-effect metrics

- [x] 3.1 [code][local-exclusive] Add per-consumer `ordered_units` / `fulfilled_units` with the played-cycle cohort in `/srv/research/tokenomics/tokenomics/tokenomics/env/aec_env.py` (design D2). After 2.1. Verify: fixture tests for every "Per-consumer accounting" scenario (horizon 1, the unplayed next-cycle orders, an order open at the horizon, early death, a zero-order consumer); the full suite is green.
- [x] 3.2 [code][local-exclusive] Add the side-effect metrics (design D3). The hard ones are run-down (`r_start`, `r_end`) and per-consumer fill rate. The monitored ones are token-priority inversion rate, minimum capacity ratio, minimum spare fraction, investment and consumer rejection rates, and held units. Verify: hand-built fixtures with a falling run-down ratio, a starved consumer, a known inversion pair and a known spare fraction; the full suite is green.
- [x] 3.3 [code][local-exclusive] Add a per-episode demand-schedule digest to episode stats (spec "Identical demand across observation contracts"). After 2.2. Verify: same config and seed give the same digest under the legacy and joint contracts, and a different digest under another seed; the full suite is green.

## 4. Seeding and roster

- [x] 4.1 [code][local-exclusive] Add `action_seed` to `run_episode` in `/srv/research/tokenomics/tokenomics/tokenomics/eval/runner.py`, with today's seeding as the default, and the hash-derived replicate seed helper (design D5). Add `blend_0.5` to `ORDERING_BASELINES` in `/srv/research/tokenomics/tokenomics/tokenomics/eval/baselines.py`. Verify: tests for both "Random-baseline replicates" scenarios; `python -m tokenomics.eval.signal_check --seeds 1 --horizon 12` shows the new rows with the existing rows unchanged; the full suite, including 1.1, is green.
- [x] 4.2 [code][local] Add `/srv/research/tokenomics/tokenomics/tokenomics/eval/roster.py`. It defines the roster file format (members, env contract, random flag, R, pruning record), uses the content hash as the version, refuses a mismatched hash, and implements the development-only pruning rule. Add draft roster v1 = producing × (ordering + `blend_0.5`). Verify: tests for the "Frozen baseline roster" and "Roster pruning uses development evidence only" scenarios.

## 5. Hindsight-baseline and verdict arithmetic

- [x] 5.1 [code][local] Add pure functions in `/srv/research/tokenomics/tokenomics/tokenomics/eval/gate.py`: per-episode max, replicate mean, world hindsight baseline, world ratio, equal-weight regime ratio, and a verdict per acceptance clause (calm ≥ 1.05, shocked ≥ 1.10, no losing world, zero death, run-down, starvation; monitored metrics pass through). Document them in `/srv/research/tokenomics/ARCHITECTURE.md` §7.2 (design D6). Verify: tests for both "Hindsight-baseline arithmetic" scenarios, "One clause fails" and "Recompute from the CSV"; the §7.2 worked example matches the test numbers.

## 6. World families

- [ ] 6.1 [code][local-exclusive] Add `/srv/research/tokenomics/tokenomics/tokenomics/eval/families.py`: dense and sparse-wide layered builders at 12 and 120 producers with an explicit fanout, plus the calm variant whose schedules are drawn at generation time from the world seed (design D8, D9). Register both in `task_archive.REPRODUCERS`. After 2.2. Verify: tests for both "Scale families" scenarios, both ratios within tolerance between 12 and 120, builder determinism, and identical calm demand under episode seeds 0 and 9; 1.1's `build_scale_config` goldens unchanged; the full suite is green.

## 7. World registry

- [ ] 7.1 [code][local] Add `/srv/research/tokenomics/tokenomics/tokenomics/eval/registry.py`. It loads, validates and versions the registry (unique ids, disjoint split seeds, calm ≥ 2× shocked per size). It also provides the zero-baseline reserve replacement, sealed gating by pre-registration manifest (hashes checked through `eval/provenance.py`), and the append-only access log `/srv/research/tokenomics/experiments/SEALED_ACCESS.log`. Add `eval/registry/*.json` to `/srv/research/tokenomics/tokenomics/pyproject.toml` package data. Generate the draft `eval/registry/worlds_v1.json` (development, sealed and reserves; design D7 defaults) with a deterministic script. After 6.1. Verify: tests for every "World registry", "Sealed worlds need a pre-registration", "Zero-baseline screening" and "Regimes" scenario; rerunning the generator reproduces the draft byte for byte, and the draft validates.

## 8. Gate driver

- [ ] 8.1 [code][local] Add the `eval/gate.py` CLI with `--mode report` and `--mode recompute` (design D11). Requirements: sharded `ProcessPoolExecutor` workers with BLAS pinned to 1 thread; per-row append and resume; the schedule-digest assertion; provenance from `eval/provenance.py`; EXP-dir output (`spec.json`, `episodes.csv`, `worlds.csv`, `regimes.csv`, `verdict.json`, `RESULT.md` skeleton). Document the CLI and the new metrics in the `/srv/research/tokenomics/CLAUDE.md` Evaluation section. After 3.3, 4.2, 5.1, 7.1. Verify: an end-to-end test on a 2-world fixture registry (horizon ≤ 12, 1 episode, 2 workers) in which report then recompute give identical verdicts and an injected digest mismatch raises; the documented `--mode recompute` command runs as written.
- [ ] 8.2 [code][local] Add `--mode screen`: roster only, with zero-baseline replacement, the headroom check, roster pruning and q-quantile X/Y calibration (design D4, D10). It writes candidate frozen files (`worlds_v1.json`, `roster_v1.json`, `side_effect_params_v1.json`, each with its content hash) into the EXP dir's `frozen/` folder. Verify: tests on synthetic rows with known quantiles, a zero-baseline world replaced by its reserve, and a saturated family rejected; changing X or Y creates a new version.
- [ ] 8.3 [code][local] Do a local dry-run of `--mode screen` and `--mode report` on 1 development world per size, at horizon ≤ 12, with 1 episode, `--max-members 3` and 1 replicate. Time one episode per family × size at horizon 12. Record the per-episode times and the extrapolated wall time and vCPU-hours for tasks 10.1 and 12.1 in `/srv/research/tokenomics/openspec/changes/add-measurement-gate-harness/sizing.md`. After 8.1, 8.2. Verify: both dry-runs finish in under 2 min total and write the EXP layout to a temp dir; `sizing.md` lists all four family × size timings.

## 9. Integration check

- [ ] 9.1 [code][local-exclusive] Run the full suite and `openspec validate add-measurement-gate-harness --strict`, then commit the tree so the pod runs a recorded commit. After 8.3. Verify: both are green, 1.1 passes, and `git status` is clean.

## 10. EXP-0044: baseline-only calibration

- [ ] 10.1 [experiment][remote:cpu16] Run `python -m tokenomics.eval.gate --mode screen` on development worlds and their reserves into `/srv/research/tokenomics/experiments/EXP-0044-<title>/`. Then write `RESULT.md` (plain English first: admitted families, replacements, pruned members, X and Y), the `/srv/research/tokenomics/experiments/README.md` index row, and a HISTORY paragraph. If a family fails the headroom check, stop and report; retuning is a new code task. After 9.1. Verify: `episodes.csv` covers every development and reserve world × episode × roster member × replicate; `frozen/` holds the three files, each with a recorded content hash; the records cite the run commit.
  - Resources: cpu16 fresh pod (--create-pod), wall time and vCPU-h sized by task 8.3 (switch to cpu32 if > 4 h on cpu16), about 22.5k member-episodes (256 development world-episodes × (28 + 12·R) members, R = 5; design D5 and D7 defaults) plus reserves; runs commit after 9.1; stage nothing (baselines only); pull /workspace/tokenomics/runs/exp0044 → experiments/EXP-0044-<title>/; --verify-source-commit yes.

## 11. Install frozen versions

- [ ] 11.1 [code][local] Copy the three frozen files from `/srv/research/tokenomics/experiments/EXP-0044-<title>/frozen/` into `/srv/research/tokenomics/tokenomics/tokenomics/eval/registry/`, replacing the drafts. Add a test asserting their content hashes equal those recorded in the EXP-0044 `spec.json`. Commit. After 10.1. Verify: the new test passes, registry validation passes, and `git status` is clean.

## 12. EXP-0045: portable champion development report

- [ ] 12.1 [experiment][remote:cpu16] Pull the EXP-0012 `orderer_scorer.pt` and EXP-0035 `producer_candidate.pt` with git-lfs and stage them under `tokenomics/_pod_ckpts/`. Run `python -m tokenomics.eval.gate --mode report` for the portable stack at 12 and 120 producers on development worlds into `/srv/research/tokenomics/experiments/EXP-0045-<title>/`. Then write `RESULT.md` (plain English first: per-regime ratios, losing worlds, deaths, side effects, contract failures), the README index row, and a HISTORY paragraph opening the measurement arc. Update HISTORY §0 to name the gate subject (design D1). Delete `_pod_ckpts/`. After 11.1. Verify: the staged checkpoints' SHA-256 match the pins in the report spec; `verdict.json` exists for both sizes; `--mode recompute` reproduces it locally from the CSV; the records cite the frozen versions and the run commit.
  - Resources: cpu16 fresh pod (--create-pod), wall time and vCPU-h sized by task 8.3 (switch to cpu32 if > 4 h on cpu16), about 22.5k frozen-roster member-episodes plus 256 subject world-episodes (the report reruns the frozen roster, design D11); runs commit after 11.1; stage orderer_scorer.pt + producer_candidate.pt under tokenomics/_pod_ckpts/; pull /workspace/tokenomics/runs/exp0045 → experiments/EXP-0045-<title>/; --verify-source-commit yes.

## Workflow follow-up

- Propose the separate deployed-path cost refactor change (eligible-only rows, incremental per-producer telemetry, O(E) remainder; `establish-project-baseline` design D2, last bullet).
- Run a cross-harness review of this change before archive.
- Sync specs and archive the change after review.
