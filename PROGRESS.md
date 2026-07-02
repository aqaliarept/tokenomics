# Progress

Use this file for ongoing implementation progress notes.

Keep roadmap/design changes in the appropriate design document, and keep empirical
results in the existing `experiments/EXP-NNNN-*` records. This file is only for
work-in-progress status, handoff notes, blockers, and the next concrete actions.

Suggested entry format:

```text
YYYY-MM-DD - short topic
Status:
Artifacts:
Decisions:
Blockers:
Next:
```

---

2026-07-01 - Curriculum game-master Stage 0 + Stage 1 scaffold (EXP-0023)
Status: DONE + reviewed. CURRICULUM_GAME_MASTER_APPROACH.md "Immediate Next Experiment"
  implemented via TDD. New `tokenomics/eval/curriculum.py`: TaskRecord (jsonl 1:1
  roundtrip), generator-ref replay (world_spec={generator,params} -> vetted pure
  REPRODUCERS builder, no hidden state), stress_suite() (18 labeled tasks, separate from
  any train replay), classify_failure() (6-way taxonomy, one class/row), run_replay_suite()
  (learner vs entity_planner+proportional on same replayed world), main() emitter.
  Three P2 review fixes landed: (A) replay anchored at record.episode_seed; (B) target_depth
  from recipe chain (_chain_depth), never blank; (C) Stage 1 telemetry cols
  (learner/reference rejected_units + reference_model_id).
Artifacts: experiments/EXP-0023-curriculum-task-archive-and-stress-suite/ (spec.json,
  RESULT.md, summary.csv, artifacts/{task_archive.jsonl, curriculum_matrix.csv}).
  tests/test_curriculum.py (16 pass). Full suite 491 passed, 0 regressions.
  Stack under test = portable producer_scorer_broad.pt (EXP-0021) + orderer_scorer.pt
  (EXP-0012); legacy champion.json untouched (historical baseline only).
Decisions: generator-ref reproducibility (not inline world_spec) for Stage 0/1 — fixtures
  are already pure(seed,**params); inline dump deferred to Stage 2's random generator.
  No champion change, no promotion (scaffolding). CLAUDE.md HARD RULE tightened: every EXP
  run -> Runpod; only smoke/quick diagnostics local.
Blockers: hard-case mining (Stage 3) is blocked on REFERENCE STRENGTH, not the learner —
  the deterministic entity_planner+proportional bar is beaten by the learner on every live
  task (0 policy_weakness across 18), so it can't surface gaps. Need a stronger reference
  (local routing / bounded short-rollout search) before mining is meaningful.
Next: (1) wire the stronger reference bar (search) into run_replay_task; then Stage 2 broad
  legal randomization + inline world_spec, or Stage 3 mining on the stronger bar.
  (2) Re-run EXP-0023 full replay on a Runpod pod via run_pod_training.py --pull to make the
  canonical artifacts cloud-produced per the tightened rule (was run local pre-tightening;
  deterministic, so identical output expected).

2026-07-01 - Stronger reference bar wired (EXP-0024) — blocker (1) above DONE
Status: DONE. `REFERENCE_BARS` registry in eval/curriculum.py: added `entity_planner+mpc_search`
  (producer fixed at entity_planner; orderer routing re-chosen per cycle by lookahead=6 rollout
  over {even_split, proportional} + 8 sampled routings via search_orderer_episode; deterministic
  given seed, no ckpt) beside the fixed practical bar. `run_replay_task` gains opt-in
  `reference_bars` → reference = STRONGEST of the set, winning bar id attributed per row
  (`reference_model_id`); DEFAULT single practical bar keeps EXP-0023 + old tests byte-identical.
  CLI `--reference-bars`. TDD: 3 new tests (registry, bars-run, best-of selection), full suite 494 pass.
Artifacts: experiments/EXP-0024-curriculum-reference-strength/ (spec.json, RESULT.md, summary.csv,
  artifacts/{curriculum_matrix.csv, task_archive.jsonl}). Ran on Runpod CPU pod (--cpu --python
  python3.12), pulled back. Same 18-task suite + same portable stack as EXP-0023; only ref changed.
Result: histogram {policy_weakness: 2, none: 2, distribution_mismatch: 13, feasibility_wall: 1}
  (was {none: 4, ...}). FIRST in-claim policy_weakness in the arc: search bar beats learner on
  scale-d3-s6 (493.3>491.3) and s7 (499>496.7) — `none` under the fixed bar. Taxonomy holds
  (scarcity-degraded search>learner but out_of_claim → distribution_mismatch, not miscounted).
Decisions: reference = best-of-available-bars, per-row attributable. No learner change, no promotion.
  CLAUDE.md got a concrete 7-step cloud-run recipe (ckpt-staging gotcha: driver syncs inner
  tokenomics/ pkg dir, NOT repo root → stage input ckpts under tokenomics/_pod_ckpts/; CPU pod
  needs --cpu --python python3.12; detached driver, --attach — old foreground caveat was stale).
Blockers: the 2 in-claim gaps are NARROW (≤0.5%, ~2 units, only 2/4 held-out seeds). Before Stage 3
  trains on them, need a paired-seed/trace check (doc close-win rule) to rule out search-sampling
  noise; or sharpen the bar (more samples/lookahead, more compute).
Next: Stage 2 broad legal randomization + inline world_spec, OR Stage 3 mining a large random pool
  against the mpc_search bar (with the paired-seed guard on narrow gaps).

2026-07-02 - Stage 3 Random Hard-Case Mining (EXP-0025) — blocker above resolved
Status: DONE. New `tokenomics/eval/curriculum_mining.py` (TDD, torch-free core via injectable
  LearnerRun): mining_pool (legal/replayable/uniquely-ided), normalized_gap, topology novelty
  (L1 over (depth,products,producers,caps)), passes_gates (taxonomy=policy_weakness only +
  PAIRED-SEED noise floor min_gap/min_win_frac — the EXP-0024 narrow-gap guard), evaluate_pool +
  rank_and_archive (greedy-diverse), _mine_parallel subprocess shard fan-out (--workers, one proc
  per core, BLAS pinned to 1; parallel == sequential regression-tested). 10 mining tests, full
  suite 513 pass.
Artifacts: experiments/EXP-0025-stage3-random-mining/ (spec.json, RESULT.md, summary.csv,
  artifacts/{task_archive.jsonl, curriculum_matrix.csv}). Full 200-seed suite (build_scale_config
  d3, seeds 0-199) vs entity_planner+mpc_search, eval-seeds 3, workers 16. (A 30-seed pilot 20-49
  preceded + agreed; full run supersedes it — dropped, this dir IS the full run.)
Result: histogram {none: 148, policy_weakness: 52}; 24 archived (gaps 2.0–7.67u, mean 3.24;
  8/24 win all 3 seeds; top s136 7.67u/3-of-3). Noise floor load-bearing: robust s136 tops the
  archive while the largest NOISY gap s34 (4.67u, 1/3 seeds) is dropped — 6 gap≥2 tasks rejected
  on the win floor (exactly the EXP-0024 false-positive). Yield real 24/200=12%.
Decisions: no learner change, no promotion. Ran on a FRESH 16-vCPU Runpod pod via
  run_pod_training.py --create-pod, terminated after --pull. Driver create/clone rewired to RunPod
  REST (POST/GET /v1/pods) — runpodctl pod create has no vCPU flag (CPU pod came up at 2 vCPU);
  REST honours vcpuCount+cpuFlavorIds (verified 16 vCPU / cpu5c). tests/test_run_pod_clone.py.
Blockers: yield is all ONE topology cell (depth-3 seed sweep) → novelty term is idle, ranking is
  regret-only. A useful curriculum needs topology diversity in the pool.
Next: (1) Stage 3 broaden — mine across a topology GRID (depths × quantity rungs × out-of-family
  recipe shapes), not a seed sweep, so novelty has distinct cells. (2) Stage 4 — feed the 24
  archived weaknesses into expert-iteration / targeted training, re-mine to confirm gaps close.

2026-07-02 - Review fixes: 3 correctness bugs in the new pod-lifecycle + parallel mining
Status: DONE. Addressed 3 review comments on the newly-added code.
  P1 (run_pod_training.py) — a created pod (--create-pod/--clone-from) holds the ONLY copy of
    artifacts; a failed --pull previously fell through to remove_pod and destroyed results.
    `_do_pulls` now RETURNS the failed specs; driver tracks a `pulls_ok` invariant (True only
    when every requested pull succeeded, or none requested) — any pull failure OR error before
    pulling keeps the created pod (stopped, not terminated) with recovery instructions. A pod
    that never came up (no artifacts) still terminates.
  P2 (run_pod_training.py build_create_body) — a cloned RunPod REST payload has ports as a JSON
    LIST (['8888/http','22/tcp']); re.split on a list raised TypeError before the create request.
    Now normalizes list OR string, de-dupes, always exposes 22/tcp first. PodSpec.ports type
    widened to list|str|None.
  P3 (curriculum_mining.py _mine_parallel) — with --workers>1, one shard failing while others
    produced rows only warned, then ranked the survivors → a BIASED archive written under the
    full-pool name. Now ABORTS (RuntimeError) if ANY shard fails; points at the shard logs.
Artifacts: tests added — test_run_pod_clone.py (cloned port-list + port-string normalization,
  _do_pulls returns failed specs for bad-spec + rsync-failure), test_curriculum_mining.py
  (_mine_parallel aborts when a shard fails, via bogus ckpts → nonzero children). Affected files
  34 pass; full suite green.
Decisions: bias-free-or-abort for mining (a partial archive is worse than no archive); never
  auto-terminate a created pod with un-pulled artifacts (cost of a lingering pod << lost results).
Blockers: none.
Next: same as EXP-0025 next (topology-grid mining, then Stage 4).

