# Tasks

## 1. Legacy-invariance safety net

- [ ] 1.1 Add `tests/test_legacy_invariance.py`. Before any other change, record golden values of every existing `episode_stats` key and `capacity_trace`, for a fixed set of (fixture world, baseline pair, seed) cells on legacy and joint env contracts, plus `build_scale_config(seed)` output for seeds 0–3. Store them under `tests/data/`. Verify the test passes on the unmodified tree.

## 2. Domain: outcome records and cyclic demand

- [ ] 2.1 Tag each consumer order with its placement cycle. Add a tuple of per-order consumer outcomes (consumer, product, quantity, tokens at placement, placement cycle, completed/rejected) to `CycleResult`. Verify with new domain tests on a hand-built economy where one order completes and one is rejected, and verify that 1.1 still passes.
- [ ] 2.2 Add `ConsumerConfig.schedule` with validation (products exist, quantities ≥ 1), `load_config` JSON support and `config_to_dict` round-trip. When it is set, the env builds a `ScheduledConsumer` from it. Verify with tests for the three "Deterministic cyclic demand" scenarios, and verify that 1.1 still passes.

## 3. Env: per-consumer stats and side-effect metrics

- [ ] 3.1 Add per-consumer `ordered_units` / `fulfilled_units` using the played-cycle cohort (design D2). Verify with fixture tests for every "Per-consumer accounting" scenario: horizon 1, the unplayed next-cycle orders, an order open at the horizon, early death, and a zero-order consumer.
- [ ] 3.2 Add run-down (`r_start`, `r_end`) and per-consumer fill rate (design D3). Verify with tests on a fixture where restoration stops, so the ratio falls, and one where a consumer is starved.
- [ ] 3.3 Add the monitored metrics: token-priority inversion rate, minimum capacity ratio, minimum spare fraction, investment and consumer rejection rates, and held units. Verify with hand-built fixtures that produce a known inversion pair and a known spare fraction.
- [ ] 3.4 Add a per-episode demand-schedule digest to episode stats. Verify with a test that the same config and seed give the same digest under the legacy and joint contracts, and a different digest under another seed.

## 4. Seeding and roster

- [ ] 4.1 Add `action_seed` to `run_episode`; the default keeps today's seeding. Add the hash-derived replicate seed helper. Verify with tests for both "Random-baseline replicates" scenarios, and verify that 1.1 still passes.
- [ ] 4.2 Add `blend_0.5` to `ORDERING_BASELINES`. Verify that `test_eval` passes and that `python -m tokenomics.eval.signal_check --seeds 1 --horizon 12` prints the new rows with unchanged values in the existing rows.
- [ ] 4.3 Add `eval/roster.py`. It defines the roster file format (members, env contract, random flag, R, pruning record), uses the content hash as the version, refuses a mismatched hash, and implements the development-only pruning rule. Add draft roster v1 = producing × (ordering + `blend_0.5`). Verify with tests for the "Frozen baseline roster" and "Roster pruning uses development evidence only" scenarios.

## 5. Hindsight baseline and verdict arithmetic

- [ ] 5.1 Add pure functions in `eval/gate.py`: per-episode max, replicate mean, world hindsight baseline, world ratio and equal-weight regime ratio. Verify with tests for both "Hindsight-baseline arithmetic" scenarios.
- [ ] 5.2 Add a pure verdict function per acceptance clause: calm ≥ 1.05, shocked ≥ 1.10, no losing world, zero death, run-down, starvation. Monitored metrics pass through without gating. Verify with tests for "One clause fails" and "Recompute from the CSV".
- [ ] 5.3 Document the hindsight-baseline arithmetic, roster and replicate rule in `/srv/research/tokenomics/ARCHITECTURE.md` §7.2. Verify that the worked example there matches the 5.1 test numbers.

## 6. World families

- [ ] 6.1 Add `eval/families.py` with dense and sparse-wide layered builders at 12 and 120 producers and an explicit fanout (design D9), and register them in `task_archive.REPRODUCERS`. Verify with tests for both "Scale families" scenarios, both ratios within tolerance between 12 and 120, builder determinism, and 1.1's `build_scale_config` goldens.
- [ ] 6.2 Add the calm variant: schedules are drawn at generation time from the world seed. Verify that calm worlds of both families give identical demand under episode seeds 0 and 9.

## 7. World registry

- [ ] 7.1 Add `eval/registry.py`, which loads, validates and versions the registry (unique ids, disjoint split seeds, calm ≥ 2× shocked per size). It also provides the zero-baseline reserve replacement, sealed gating by pre-registration manifest (hashes checked through `eval/provenance.py`), and the append-only access log. Add `eval/registry/*.json` to pyproject package data. Verify with tests for every "World registry", "Sealed worlds need a pre-registration", "Zero-baseline screening" and "Regimes" scenario.
- [ ] 7.2 Generate the draft `eval/registry/worlds_v1.json` (development, sealed and reserves; design D7 defaults) with a deterministic script. Verify that the generator output is reproducible and the file passes validation. It stays a draft until 10.3.

## 8. Gate driver

- [ ] 8.1 Add the `eval/gate.py` CLI with `--mode screen|report|recompute`. Requirements: sharded `ProcessPoolExecutor` workers with BLAS pinned to 1 thread; per-row append and resume; schedule-digest assertion; provenance from `eval/provenance.py`; EXP-dir output (`spec.json`, `episodes.csv`, `worlds.csv`, `regimes.csv`, `verdict.json`, `RESULT.md` skeleton). Verify with an end-to-end test on a 2-world fixture registry (horizon ≤ 12, 1 episode): report then recompute give identical verdicts, and an injected digest mismatch raises.
- [ ] 8.2 Add the side-effect parameter file format and the q-quantile calibration over hindsight-baseline-attaining development episodes (design D4). Verify with a test on synthetic rows with known quantiles, and that a parameter change creates a new version.
- [ ] 8.3 Document the gate CLI and the new metrics in the `/srv/research/tokenomics/CLAUDE.md` Evaluation section. Verify that the documented `--mode recompute` command runs as written on the 8.1 fixture output.
- [ ] 8.4 Measure single-episode wall time locally at horizon ≤ 12 for each family × size, as a quick one-shot diagnostic. Record the extrapolated pod budget in the EXP-0044 `spec.json` draft.

## 9. Integration check

- [ ] 9.1 Run the full suite (`../.venv/bin/python -m pytest -q`) and `openspec validate add-measurement-gate-harness --strict`. Verify that both are green and that 1.1 still passes.

## 10. EXP-0044: baseline-only calibration (CPU pod)

- [ ] 10.1 Do a local `--dry-run` of `--mode screen` on 1–2 registry worlds at horizon ≤ 12. Verify that it finishes in seconds and writes the EXP layout.
- [ ] 10.2 Run `--mode screen` on development worlds via `run_pod_training.py --cpu --pull` into `experiments/EXP-0044-<title>/`. Verify that `episodes.csv` covers every development world × episode × roster member × replicate.
- [ ] 10.3 Apply zero-baseline replacement, the headroom check (retuning recorded if needed), roster pruning and X/Y calibration. Freeze roster v1, registry v1 and parameters v1 by content hash. Verify that each frozen file validates, and that its hash is recorded in the EXP-0044 `spec.json`.
- [ ] 10.4 Write the EXP-0044 `RESULT.md` (plain English first) and the `/srv/research/tokenomics/experiments/README.md` index row. Verify that `RESULT.md` states the frozen versions, the admitted families and the X/Y values.

## 11. EXP-0045: portable champion development report (CPU pod)

- [ ] 11.1 Pull the EXP-0012 `orderer_scorer.pt` and EXP-0035 `producer_candidate.pt` with git-lfs and stage them under `tokenomics/_pod_ckpts/`. Verify that their SHA-256 match the pins recorded in the report spec.
- [ ] 11.2 Run `--mode report` for the portable stack at 12 and 120 producers on development worlds via `run_pod_training.py --cpu --pull` into `experiments/EXP-0045-<title>/`. Verify that `verdict.json` exists for both sizes and that `--mode recompute` reproduces it locally from the CSV.
- [ ] 11.3 Write the EXP-0045 `RESULT.md` (plain English first: per-regime ratios, losing worlds, deaths, side effects, contract failures), add the README index row, and add a HISTORY paragraph opening the measurement arc. Update HISTORY §0 to name the gate subject (design D1). Delete `tokenomics/_pod_ckpts/`. Verify that the records cite the frozen versions and the commit.

## Workflow follow-up

- Propose the separate deployed-path cost refactor change (eligible-only rows, incremental per-producer telemetry, O(E) remainder; `establish-project-baseline` design D2, last bullet).
- Run a cross-harness review of this change before archive.
- Sync specs and archive the change after review.
