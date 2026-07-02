# EXP-0025 — Stage 3 Random Hard-Case Mining

**Stage:** CURRICULUM_GAME_MASTER_APPROACH.md Stage 3 (Random Hard-Case Mining).
**Version:** curriculum-mining-v1. **Promote recommended:** NO (no learner change).

## Question

EXP-0024 gave the curriculum its first reference bar strong enough to expose in-claim
`policy_weakness` (`entity_planner+mpc_search`), but only on a hand-picked 18-task stress
suite, and it flagged that the gaps are narrow (≤0.5%, ~2 units, ~half the seeds) — so a
single-seed gap could be noise. Stage 3 asks: mine a *large random* legal pool with that
bar, and does the paired-seed noise floor cleanly separate reproduced in-claim weaknesses
from coin-flip noise, leaving an archive worth training on?

## What was built

- `tokenomics/eval/curriculum_mining.py` — the Stage 3 driver, torch-free at its core via
  an injectable `LearnerRun`:
  - `mining_pool(generator, seeds, params, boundary)` → legal, replayable, uniquely-ided
    `TaskRecord`s (validated against `REPRODUCERS`; world_seed == episode_seed == seed).
  - `normalized_gap` = `(reference_units - learner_units) / max(1, denom)` — regret scaled
    by requested/feasible units, floored so a tiny-demand world can't blow the ratio up.
  - `novelty(desc, archive)` = L1-distance falloff over topology descriptors
    `(depth, products, producers, caps)`; empty archive → fully novel.
  - `passes_gates` — **taxonomy gate** (only `policy_weakness`) **+ paired-seed noise
    floor**: `gap_units >= min_gap` AND `win_frac >= min_win_frac` (fraction of paired eval
    seeds where the reference beats the learner). Contract/implementation/wall/distribution
    never mined.
  - `evaluate_pool` + `rank_and_archive` (greedy-diverse by `alpha*normalized_gap +
    beta*novelty`) — split so a **parallel** sharded eval + one global rank is byte-identical
    to a sequential mine (regression-tested).
  - `_mine_parallel` — subprocess fan-out, one shard per worker (`--emit-rows` jsonl
    children, BLAS pinned to 1 thread), parent merges + ranks. Mining per task is
    single-threaded; throughput comes from process-per-core (CLAUDE.md "use the cores").
- 10 mining tests (`test_curriculum_mining.py`), TDD. Full suite **513 pass**.
- **Off-box enabler:** `run_pod_training.py --create-pod` now creates the pod via RunPod
  **REST** `POST /v1/pods` (with `vcpuCount` + `cpuFlavorIds`), not `runpodctl pod create`
  (which has no vCPU flag → a CPU pod came up at 2 vCPU). Verified 16 vCPU / cpu5c comes up
  as asked; `--clone-from` reads the template via REST `GET` (only REST returns
  `cpuFlavorId`). See `tests/test_run_pod_clone.py`.

## Results

Run: fresh Runpod CPU pod (16 vCPU / cpu5c, created + terminated via `--create-pod`),
full seed suite `--generator build_scale_config --depth 3 --seeds 0-199` (200 tasks),
`--reference-bar entity_planner+mpc_search`, `--eval-seeds 3 --horizon 100`, `--workers
16`, gates `min_gap 2.0 / min_win_frac 0.6`. (A 30-seed pilot, seeds 20-49, preceded this
and agreed on the noise-floor behaviour; the full 200-seed run supersedes it.)

**Failure-class histogram:** `{none: 148, policy_weakness: 52}` — 0 contract, 0
implementation, 0 distribution_mismatch, 0 feasibility_wall. The strong bar beats the
learner in-claim on **52/200** random depth-3 tasks; on the other 148 the learner is
already at/above the bar.

**Gated archive — 24 tasks** (of the 52 `policy_weakness`). Gaps 2.0–7.67 units (mean
3.24, median 3.0); **8/24 win all 3 paired seeds** (`win_frac = 1.0`) — robust reproduced
weaknesses, not seen in the tiny pilot pool. Top of the archive:

| task (seed) | gap (units) | win_frac | note |
|---|---|---|---|
| s136 | 7.67 | 1.00 | biggest gap in the pool, **all 3 seeds** |
| s174 | 6.67 | 0.67 | |
| s191 | 4.33 | 1.00 | all 3 seeds |
| s27  | 4.33 | 0.67 | (also archived in the pilot) |
| s80  | 3.67 | 1.00 | all 3 seeds |
| s28  | 3.67 | 0.67 | |

**The noise floor did its job at scale.** Of the 52 `policy_weakness` rows, 28 were
dropped — including **6 with a real mean gap (≥2u) but only a 1-of-3-seed win**:

| dropped (seed) | gap | win_frac |
|---|---|---|
| s34  | 4.67 | 0.33 |
| s50  | 3.00 | 0.33 |
| s186 | 3.00 | 0.33 |
| s21  | 2.00 | 0.33 |
| s161 | 2.00 | 0.33 |

s34 recurs as the headline false positive: a 4.67u mean gap that is a **1-of-3-seed**
artifact. Ranking by mean gap alone would seed the curriculum with it; the paired-seed
floor rejects it — and now correctly *keeps* s136 (7.67u across all 3 seeds) above it.

## Conclusions

- **Stage 3 machinery works end-to-end on the cloud, at full scale.** 200-task random pool
  → strong-bar eval → taxonomy + paired-seed gating → ranked, diverse archive, materialized
  as a replayable `task_archive.jsonl` + `curriculum_matrix.csv` + `summary.csv`, all
  parallel across 16 cores with parallel == sequential guaranteed.
- **The paired-seed noise floor is load-bearing, not decorative.** At 200 tasks it flips
  the ranking in the same way the pilot showed: the largest *robust* gap (s136, 7.67u, 3/3
  seeds) tops the archive; the largest *noisy* gap (s34, 4.67u, 1/3 seeds) is dropped.
  Confirms the EXP-0024 caveat quantitatively, and the full run adds 8 all-seed-win tasks
  the pilot's 30-seed pool could not surface.
- **Yield is real (24/200 = 12%) but all on one topology cell.** Every task is depth-3
  `build_scale_config` (pool topology cells = 1); the novelty term has no distinct cells to
  reward, so ranking is effectively regret-only here. Gaps stay narrow (≤1.5% normalized),
  consistent with a strong learner.
- No learner change, no promotion. Portable stack unchanged
  (`producer_scorer_broad.pt` + `orderer_scorer.pt`); legacy `champion.json` untouched.

## Next

- **Stage 3 (broaden the pool):** mine across a topology grid (multiple depths + quantity
  rungs + out-of-family recipe shapes), not a single-cell seed sweep, so novelty selection
  and the archive span cells — this run proves the pipeline + regret ranking at scale; it
  does not yet produce a topology-diverse curriculum.
- **Stage 4:** feed the 24 archived weakness tasks into expert-iteration / targeted
  training and re-mine to confirm the gaps close (curriculum loop closes).
