# EXP-0024 — Curriculum Reference-Strength (stronger reference bar)

**Stage:** CURRICULUM_GAME_MASTER_APPROACH.md Stage 0 reference bars (enabler for Stage 3).
**Version:** curriculum-refbar-v1. **Promote recommended:** NO (no learner change).

## Question

EXP-0023 found the first curriculum blocker is *reference strength*, not the learner:
the fixed `entity_planner + proportional` bar is beaten by the portable stack on every
live task (0 `policy_weakness` across 18), so it can't surface a gap for Stage 3 mining.
Does a stronger, checkpoint-free reference bar — per-cycle **directed short-rollout
orderer search** (producer fixed at `entity_planner`) — expose in-claim `policy_weakness`
the fixed bar could not?

## What was built

- `REFERENCE_BARS` registry in `tokenomics/eval/curriculum.py`:
  - `entity_planner+proportional` — the fixed practical bar (EXP-0023).
  - `entity_planner+mpc_search` — **bounded short-rollout / directed routing search**:
    producer held at `entity_planner`; each cycle the orderer routing is re-chosen by a
    `lookahead=6` rollout over `{even_split, proportional}` + 8 sampled routing vectors
    (`search_orderer_episode`, EXP-0005). Deterministic given seed, no checkpoint — the
    doc's "local routing / bounded short-rollout search where available".
- `run_replay_task` gains an opt-in `reference_bars` list; the reference is the
  **strongest** of the enabled bars per task, and the winning bar id is retained
  (`reference_model_id`) so every ratio is attributable. **Default = single practical
  bar → EXP-0023 behaviour and artifacts unchanged.**
- CLI `--reference-bars`. Three new tests (registry, bars-run, best-of selection); 494 pass.

## Results

Run: pod (CPU), `--eval-seeds 3 --horizon 100`, `--reference-bars
entity_planner+proportional,entity_planner+mpc_search`. Same 18-task stress suite and
same portable stack as EXP-0023; only the reference changed.

**Failure-class histogram:** `{policy_weakness: 2, none: 2, distribution_mismatch: 13,
feasibility_wall: 1}` — vs EXP-0023's `{none: 4, distribution_mismatch: 13,
feasibility_wall: 1}`. The stronger bar wins best-reference on **13/18** rows; the
practical bar owns the other 5 — the `deep_join` wall (both 0), three exact-tie degenerate
edges (`single_producer`, `primal_only`, `minimal`, where search buys nothing), and
`degenerate-no_backlog` (practical 297 > search).

| task | boundary | learner | ref | bar | ratio | class |
|---|---|---|---|---|---|---|
| scale-d3-s6 | in_claim | 491.3 | **493.3** | mpc_search | **0.996** | **policy_weakness** |
| scale-d3-s7 | in_claim | 496.7 | **499.0** | mpc_search | **0.995** | **policy_weakness** |
| scale-d3-s8 | in_claim | 589.3 | 586.0 | mpc_search | 1.006 | none |
| scale-d3-s9 | in_claim | 544.7 | 539.0 | mpc_search | 1.010 | none |
| forkjoin-s0 | out_of_claim | 173.3 | 110.3 | mpc_search | 1.571 | distribution_mismatch |
| shared_cap-s0 | out_of_claim | 494.0 | 482.7 | mpc_search | 1.024 | distribution_mismatch |
| uneven-s0 | out_of_claim | 484.0 | 475.0 | mpc_search | 1.019 | distribution_mismatch |
| **deep_join-s0** | **feasibility_wall** | **0.0** | **0.0** | proportional | — | **feasibility_wall** |
| scarcity-low_funding-s0 | out_of_claim | 495.3 | 481.7 | mpc_search | 1.028 | distribution_mismatch |
| scarcity-low_inventory-s0 | out_of_claim | 471.0 | 453.3 | mpc_search | 1.039 | distribution_mismatch |
| scarcity-high_backlog-s0 | out_of_claim | 893.0 | 700.7 | mpc_search | 1.275 | distribution_mismatch |
| scarcity-degraded-s0 | out_of_claim | 457.0 | 459.0 | mpc_search | 0.996 | distribution_mismatch |
| scarcity-running_investment-s0 | out_of_claim | 482.0 | 415.3 | mpc_search | 1.161 | distribution_mismatch |
| degenerate-single_producer-s0 | out_of_claim | 414.0 | 414.0 | proportional | 1.000 | distribution_mismatch |
| degenerate-primal_only-s0 | out_of_claim | 552.7 | 552.7 | proportional | 1.000 | distribution_mismatch |
| degenerate-no_inventory_expansion-s0 | out_of_claim | 494.0 | 482.7 | mpc_search | 1.024 | distribution_mismatch |
| degenerate-no_backlog-s0 | out_of_claim | 294.0 | 297.0 | proportional | 0.990 | distribution_mismatch |
| degenerate-minimal-s0 | out_of_claim | 321.3 | 321.3 | proportional | 1.000 | distribution_mismatch |

Per-row units / rejected units / bar id: `artifacts/curriculum_matrix.csv` /
`summary.csv`. Reproducible records: `artifacts/task_archive.jsonl` (identical suite to
EXP-0023).

### Reading

- **Blocker resolved.** The stronger bar surfaces the **first in-claim `policy_weakness`
  in the arc** — `scale-d3-s6` (493.3 > 491.3) and `scale-d3-s7` (499.0 > 496.7), where
  EXP-0023's fixed bar scored `none`. Stage 3 mining now has a reference that can expose a
  gap instead of always losing to the learner.
- **Taxonomy holds under the stronger bar.** `scarcity-degraded` has search 459 > learner
  457 but is `out_of_claim`, so it is correctly `distribution_mismatch`, **not** miscounted
  as a promoted-claim `policy_weakness`. Out-of-claim *strength* stays visible in the ratio
  (learner still ≥ search on 11/13 out-of-claim rows, up to 1.57× on forkjoin).
- **Wall + ties unchanged.** `deep_join` still both-0 (`feasibility_wall`); the exact-tie
  degenerate edges pick the practical bar (search buys nothing), so the label is stable.

### Caveats (before Stage 3 trains on these)

- The two in-claim gaps are **narrow**: ≤0.5% (~2 units on ~490), and only 2/4 held-out
  scale seeds (s8/s9 the learner still wins). This is a *reference that can now rank the
  learner below itself on some tasks*, not a proven large policy hole.
- Needs a **paired-seed / trace check** before mining treats a 2-unit gap as a real
  weakness rather than search-sampling noise — exactly the close-win rule in the doc's
  promotion gates. Higher `n_samples` / `lookahead` would sharpen the bar (more compute).

## Acceptance (enabler)

- [x] Stronger reference bar wired, checkpoint-free, deterministic-given-seed.
- [x] Best-of-bars reference selection, winning bar attributed per row.
- [x] Default single-bar behaviour (EXP-0023) unchanged; existing tests green.
- [x] In-claim `policy_weakness` surfaced where the fixed bar found none.
- [x] Failure taxonomy still separates out-of-claim gaps from policy weakness.
- [x] No learner change, no promotion.

## Not done here (later stages)

Stage 2 broad legal randomization + inline `world_spec`; Stage 3 mining a large random
pool against this bar; paired-seed confirmation of the narrow in-claim gaps; sharper bar
(more samples/lookahead) if the narrow gaps prove to be noise.
