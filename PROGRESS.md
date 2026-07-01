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

