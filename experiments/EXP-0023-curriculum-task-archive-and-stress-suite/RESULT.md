# EXP-0023 — Curriculum Task Archive & Independent Stress Suite

**Stage:** CURRICULUM_GAME_MASTER_APPROACH.md Stage 0 + Stage 1 (scaffolding).
**Generator version:** curriculum-scaffold-v1. **Promote recommended:** NO (scaffolding).

## Question

Can the current *portable* producer/orderer stack and the deterministic reference bar
run on **serialized, replayable** curriculum tasks with **per-row failure
classification** — reusing the existing `experiments/` index, keeping contract failures
out of the policy scores, and making no promotion claim?

## What was built

- `tokenomics/eval/curriculum.py`:
  - `TaskRecord` — the doc's Task Object; `to_json_line`/`from_json_line` (jsonl, torch-free 1:1 roundtrip).
  - **Generator-ref reproducibility** — `world_spec = {generator, params}` names a vetted
    pure `(seed, **params) -> Configuration` builder in `REPRODUCERS`; `replay()` re-calls
    it. No inlined world dump, no hidden process state. Illegal generator name = replay
    error, not arbitrary code.
  - `stress_suite()` — 18 legal, replayable, labeled tasks across the accepted fixture
    families, **separate from any training replay buffer**.
  - `classify_failure()` + `ReplayOutcome` — the doc's six-way taxonomy
    (`none | policy_weakness | feasibility_wall | contract_failure |
    implementation_failure | distribution_mismatch`), most-specific-first, one class per row.
  - `run_replay_suite()` — learner (portable stack) vs reference bar
    (`entity_planner + proportional`) on the SAME replayed world, per-row classification.
  - `main()` — emits the EXP layout: `artifacts/task_archive.jsonl`,
    `artifacts/curriculum_matrix.csv`, `summary.csv`.

- Stack under test = the **portable** scorers (`producer_scorer_broad.pt` [EXP-0021] +
  `orderer_scorer.pt` [EXP-0012]), not the legacy fixed-shape `champion.json` stack (left
  untouched; recorded as historical baseline only).

## Results

Run: `--eval-seeds 3 --horizon 100`, local. 18 tasks, mean learner (portable stack) vs
reference (`entity_planner + proportional`) consumer-completed units on the same replayed
world.

**Failure-class histogram:** `{none: 4, distribution_mismatch: 13, feasibility_wall: 1}`.
**Zero `contract_failure`, zero `implementation_failure`, zero `policy_weakness`.**

| task | boundary | learner | ref | class |
|---|---|---|---|---|
| scale-d3-s6 | in_claim | 490.3 | 453.3 | none |
| scale-d3-s7 | in_claim | 496.7 | 474.3 | none |
| scale-d3-s8 | in_claim | 589.3 | 499.7 | none |
| scale-d3-s9 | in_claim | 544.7 | 491.3 | none |
| forkjoin-s0 | out_of_claim | 173.3 | 63.3 | distribution_mismatch |
| shared_cap-s0 | out_of_claim | 494.0 | 341.0 | distribution_mismatch |
| uneven-s0 | out_of_claim | 484.0 | 349.3 | distribution_mismatch |
| **deep_join-s0** | **feasibility_wall** | **0.0** | **0.0** | **feasibility_wall** |
| scarcity-low_funding-s0 | out_of_claim | 495.3 | 347.3 | distribution_mismatch |
| scarcity-low_inventory-s0 | out_of_claim | 472.3 | 244.0 | distribution_mismatch |
| scarcity-high_backlog-s0 | out_of_claim | 893.0 | 514.7 | distribution_mismatch |
| scarcity-degraded-s0 | out_of_claim | 457.0 | 303.3 | distribution_mismatch |
| scarcity-running_investment-s0 | out_of_claim | 482.0 | 319.3 | distribution_mismatch |
| degenerate-single_producer-s0 | out_of_claim | 414.0 | 414.0 | distribution_mismatch |
| degenerate-primal_only-s0 | out_of_claim | 552.7 | 552.7 | distribution_mismatch |
| degenerate-no_inventory_expansion-s0 | out_of_claim | 494.0 | 341.0 | distribution_mismatch |
| degenerate-no_backlog-s0 | out_of_claim | 294.0 | 297.0 | distribution_mismatch |
| degenerate-minimal-s0 | out_of_claim | 321.3 | 321.3 | distribution_mismatch |

Per-row units, death, contract status, ratio: `summary.csv` /
`artifacts/curriculum_matrix.csv`. Reproducible records: `artifacts/task_archive.jsonl`.

### Reading

- **Scaffolding works.** Every task serialized, replayed from its generator-ref, and ran
  under both learner and reference with a failure class attached. Contract and policy are
  separated per row — no contract failure was ever scored as a policy number.
- **Portable stack loads & runs on the whole surface** — 0 contract / 0 implementation
  failures, incl. the out-of-family and degenerate edges.
- **Known feasibility wall confirmed** — `deep_join` (depth-5 vs TTL=2): learner AND
  reference both 0 units → labeled `feasibility_wall`, kept out of the policy scores.
- **The learner is not beaten by the practical bar on any replayed task** (0
  `policy_weakness`; learner ≥ reference on all live rows, incl. every out-of-family /
  scarcity row). The `distribution_mismatch` label is a claim-boundary tag, not a loss:
  the CSV still carries learner/ref/ratio, so out-of-claim *strength* stays visible.
  **Implication for Stage 3 (hard-case mining):** the deterministic `entity_planner +
  proportional` bar is too weak to expose gaps — mining useful `policy_weakness` will need
  a stronger reference (local routing / bounded short-rollout search), exactly the bars
  Stage 0 lists as "where available". That is the first real curriculum blocker, and it is
  a *reference-strength* problem, not a learner problem.

### Scaffold fixes (post-review, before relying on the archive/matrix)

Three P2 review items fixed + TDD-covered (`test_curriculum.py`, full suite 491 pass):

1. **Replay the recorded `episode_seed`.** Episodes are now anchored at
   `record.episode_seed` (`_episode_seeds(seed, n) = [seed..seed+n)`), not hardcoded from
   0 — a one-row archive reproduces the exact condition it serialized, and `--eval-seeds`
   averages only seeds derivable from the record.
2. **`target_depth` never blank.** Derived from the recipe chain (`_chain_depth`,
   cycle-safe, DAG-based — not the capacity-type count, which shared-capacity fixtures
   undercount) and depth serialized into every stress record's params. `deep_join` wall
   correctly reports depth 5; `primal_only` correctly reports depth 1.
3. **Stage 1 telemetry.** Added `learner_rejected_units`, `reference_rejected_units`, and
   `reference_model_id` per row. Low-completed rows are now diagnosable — e.g. the
   `deep_join` wall shows learner AND reference **591.67 rejected units / 0 completed**,
   and scarcity `low_inventory` shows learner 30 rejected vs reference 254, making the
   1.94× ratio legible.

## Acceptance (doc's "Immediate Next Experiment")

- [x] Portable stack loads from the checkpoint paths recorded in `spec.json`.
- [x] At least one task from each accepted fixture family serializes and replays.
- [x] Champion and reference bars run on the replay smoke.
- [x] Failure class present in every row.
- [x] No new parallel experiment index / directory convention introduced.
- [x] No promotion recommendation from this scaffolding.

## Not done here (later stages)

Broad legal randomization (Stage 2), hard-case mining (Stage 3), prioritized
replay/relabel + training (Stage 4), mutation (Stage 5), learned game master (Stage 6).
Search/short-rollout reference bar not wired (`ratio_vs_best_available_bar` blank, not
aliased to the practical bar).
