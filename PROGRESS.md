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


---

## Stage 4 driver + observability (in progress)
Status: Stage 4 (prioritized replay + relabeling) built + TDD'd; cloud run live on pod
  hd8oel7ei7to1c (blind-replay phase at time of writing) → will pull EXP-0026. Observability
  improvements to run_pod_training.py DONE + green.
Artifacts:
  - eval/curriculum_replay.py — pure half (split_archive / score_from_notes / replay_priorities /
    sample_replay, priority ∝ (eps+score)^eta) + torch-guarded run_stage4_replay driver +
    main()/--dry-run. Candidate trained in CHAMPION score-model format (producer:
    distill_producer_scorer, entity_planner teacher == mining bar's producer; orderer:
    distill_orderer_scorer, proportional teacher) so transfer_matrix/eval_joint_stack compare
    apples-to-apples. Evals candidate-vs-champion on mined-val + blind (evaluate_pool, mpc_search
    bar), transfer+stress (transfer_matrix ×2), producer-alone/joint (eval_joint_stack). Emits EXP
    layout + advisory gate; promote_recommended ALWAYS False (single-cell + human-approved).
  - tests/test_curriculum_replay.py — 18 (split/priority/sample/score pure + end-to-end torch
    smoke + phase-marker assertion). Full suite 545 green.
  - Observability (user-requested): run_pod_training.py — (1) init_local_mirror writes a run
    header + truncates at LAUNCH START (before the ~5 min install), make_content_sink gains
    truncate=False so the stream appends → local mirror never masquerades as a prior run during
    setup (the stale-file gotcha we hit); (2) parse_run_status/write_run_status + a throttled
    heartbeat in the sink → compact <remote-stem>.status (phase + last line + state) on disk
    without an active tail; (3) curriculum_replay emits ordered `[phase i/7] name` markers the
    heartbeat parses. `.gitignore` += *.status. tests/test_run_pod_detached.py +9.
Decisions:
  - Stage 4 on the CURRENT single-cell EXP-0025 archive (user); best case = "promising, not
    promoted". Lever = BOTH, but the portable orderer SCORE model distills only from the
    proportional teacher (teacher_logits), so it structurally can't absorb the search-oracle
    headroom the mined gap used — retraining ≈ reproduces champion; that limit is the Stage-4
    orderer finding (reported), and the shared transfer/stress harness only scores the
    score-model orderer (not the GaussianActor search orderer) anyway.
  - Pre-flight local smoke (mpc_search + fixtures, tiny) before burning a pod; real run on cloud
    per HARD RULE (fresh 16-vCPU cpu5c pod, auto-terminated).
Blockers: none.
Next: on pull, fill EXP-0026 RESULT.md conclusions + README index row + PROGRESS; interpret the
  gate (expect producer ≈/↑ champion, orderer ≈ champion, no transfer/stress regression).

## Stage 4 result (EXP-0026) — REJECT, pipeline validated
Status: cloud run done (fresh 16-vCPU pod, auto-terminated), pulled to
  experiments/EXP-0026-stage4-prioritized-replay (nesting flattened). Full suite 545 green.
Result: candidate ≈ champion on the mined family (val 511.39 vs 511.33, blind tail 444.33 tie,
  in-family joint 532.7 vs 530.5) AND 4 out-of-family regressions, all uneven-strength
  (465–495 vs 487–584, −22..−92u), producer-side. 0 contract failures. promote_recommended=false.
Why: (1) portable orderer score model distills only from proportional → can't absorb the
  mpc_search headroom the 2–7u mined gap used (architecture limit, not data). (2) producer trained
  FROM SCRATCH on 12 narrow depth-3 seeds REPLACED EXP-0021's broad coverage → lost uneven-strength.
Decisions: single-cell mined weakness too small/narrow to train on; replace-not-augment replay
  trades robustness for nothing. Fixes for next: broaden archive across topologies (EXP-0025
  blocker) THEN augment the champion's broad train set with mined replay (not mined-only distill).
Blockers: none.
Next: (1) topology-grid mining (EXP-0025 blocker); (2) Stage 4 v2 = augment broad train set with
  prioritized mined replay, re-eval transfer/stress. No promotion; champion untouched.

## Stage 4 review fixes (curriculum_replay + run_pod heartbeat)
Status: three code-review items on the Stage 4 driver + heartbeat fixed + TDD'd. Full suite 552
  green (was 545; +7 tests). Forward-looking — do NOT change the concluded EXP-0026 verdict.
Fixes:
  - [P2] Default replay sampling now honors priorities. `sample_replay(..., replace=True)` added;
    driver default (sample_k=None) draws a full-budget batch WITH replacement so high-regret tasks
    are DUPLICATED (real prioritized replay). Old default (no-replacement, k==len(train)) selected
    every task once → priorities only reordered. Explicit --sample-k still = distinct subset.
  - [P2] Orderer training is now config-aware. `collect_scorer_samples(..., configs=)` +
    `distill_orderer_scorer(..., train_configs=)`; driver passes the replayed archive configs so a
    non-scale / mixed-depth archive trains the orderer on the mined topologies, not a
    build_scale_config(seed, depth) stand-in. Misaligned configs→seeds raises ValueError.
  - [P3] Heartbeat keeps the last phase when a long phase emits >8 KB before the next marker.
    Sink tracks a STICKY phase off the full incoming chunk (not the bounded 8192-char tail);
    `write_run_status(..., phase=)` override wins over the parsed tail.
Note: none of these alter EXP-0026 (single-cell archive: replayed config == build_scale_config,
  and replace-vs-not is near-identical over one topology cell) — the REJECT stands, no pod re-run.
Blockers: none.
Next: unchanged — topology-grid mining, then Stage 4 v2 (augment, not replace).

## Stage 4 bug-fixed rerun (EXP-0027) — REJECT stands, now scientifically valid
Status: cloud run done (fresh 16-vCPU pod, terminated), pulled to
  experiments/EXP-0027-stage4-replay-fixed. Suite 552 green. Controlled rerun of EXP-0026 with the
  three review fixes applied, defaults otherwise identical (single-cell archive, eta=1.0, ε=0.01,
  epochs/iters=400, mpc_search bar, fixtures).
Result: candidate now BEATS champion on mined-val (512.998 vs 511.333, +1.67u; EXP-0026 was +0.06u
  wash), blind tail (445.67 vs 444.33), and every scale/sharedcap/scarcity/degenerate cell. Lift is
  the ORDERER (orderer-alone 497.7/594.0 > champ on s6/s8) — bug-fix #1 (with-replacement duplication)
  reweighted the fit; updates EXP-0026's "orderer can't move". BUT same 4 uneven-strength regressions
  (456–528 vs 487–584, −16..−128u), producer-side (producer-alone ≈ champ on scale). 0 contract fails.
Why it matters: reproduces the EXP-0026 REJECT for the SAME structural reason on the CORRECT effective
  data → the rejection was never a data-bug artifact. The two prescribed fixes (broaden archive across
  topologies; AUGMENT not replace the broad train set) are confirmed as the real next steps.
Ops bug filed (not fixed): run_pod_training detached-launch SSH hung — remote launch shell stuck in
  do_wait, never exits → client blocks in ssh_run(launch), never reaches monitor_remote, no streaming.
  Root-caused with ps/wchan evidence. Run itself fine (setsid job independent); recovered via manual
  reconnecting `tail -F` + rsync pull, pod terminated. Proposed fix: `… & ; exit 0` + `ssh -n` on the
  remote launch, and a bounded timeout on ssh_run(launch). Observability heartbeat worked up to launch.
Blockers: none (Stage 4 concluded on this archive).
Next: topology-grid mining (EXP-0025 blocker), then Stage 4 v2 = augment broad train set with
  prioritized mined replay. Optionally apply the run_pod launch-hang fix. No promotion; champion untouched.

## run_pod launch-hang fix (item 1 of the post-EXP-0027 queue)
Status: DONE. Applied the three-part fix to run_pod_training.py; suite 555 green (+3 tests).
Artifacts: run_pod_training.py — build_launch_cmd now appends `exit 0` after the backgrounded
  `setsid … &` (outer sshd shell exits instead of sitting in do_wait); Ssh.base_cmd(no_stdin=True)
  adds `ssh -n`; ssh_run gains `timeout`/`no_stdin`, launch call bounded by LAUNCH_TIMEOUT_S=60 and
  on TimeoutExpired logs + falls through to monitor_remote (liveness via pid/exit files either way).
  tests/test_run_pod_detached.py +3 (outer-shell-exit, base_cmd -n, ssh_run timeout threads -n).
Decisions: timeout path is non-fatal — the detached job runs in its own session, so a slow/hung
  launch shell no longer blocks the client; monitor picks the job up from the pod files regardless.
Blockers: none. Not committed (per rule).
Next: item 2 — topology-grid mining (vary depth/products/producers so the archive has distinct
  novelty cells), then item 3 — Stage 4 v2 (augment, not replace).

## Topology-grid mining driver + launch (item 2 of the post-EXP-0027 queue)
Status: IN PROGRESS (cloud run launched). Driver made grid-capable; suite 560 green (+5 tests).
Artifacts (code): tokenomics/eval/curriculum_mining.py — parse_grid (JSON list of param cells,
  falls back to single default cell = old behaviour), grid_pool (concat mining_pool over cells;
  distinct params ⇒ distinct task_ids), grid_shards (distribute workers across cell×seed grid);
  main() gains --grid/--params, _mine_parallel shards by (cell,seed-range) passing --params JSON;
  tests/test_curriculum_mining.py +5. Quoting round-trip through build_launch_cmd verified (grid
  JSON survives as one argv token).
Run: EXP-0028 topology-grid mine — 9 IN-CLAIM depth-3 scale cells (products_per_layer∈{3,4,5} ×
  producers_per_layer∈{4,5,6} → products 9-15, producers 12-18, all inside the stage-5 envelope
  8-16/10-20/3), 30 seeds/cell = 270 tasks, eval-seeds 3, mpc_search bar, min-gap 2.0/min-win 0.6.
  9 distinct descriptor cells (was 1 in EXP-0025). Fresh 16-vCPU pod via --create-pod (auto-terminate).
Decisions: kept the grid honestly in-claim (fixed depth=3, the claimed wall axis; varied width +
  producer density only) so gaps score policy_weakness, not distribution_mismatch. Depth-4+ excluded
  (out-of-claim / feasibility wall). This also live-tests the fix-#1 launch path.
Blockers: none yet. Next: pull EXP-0028, report histogram + archived topology-cell spread, then
  item 3 = Stage 4 v2 (augment broad train set with prioritized mined replay).

## Topology-grid mining (EXP-0028) — DONE, single-cell blocker FIXED
Status: cloud run done (fresh 16-vCPU pod, auto-terminated), pulled to
  experiments/EXP-0028-topology-grid-mining. Suite 560. First live run on the FIXED
  detached-launch path (item 1) — client reached monitor_remote, streamed, no hang.
Result: 270 sampled over 9 in-claim depth-3 cells (width×producer-density); {none:221,
  policy_weakness:49}; 26 archived across 8 of 9 TOPOLOGY cells (EXP-0025 was 1), gaps
  2.0-7.33u mean 3.67, 12/26 robust 3-of-3 wins, even spread, 0 contract fails / 0 deaths.
Finding: weakness concentrates at LOW producer density (producers=12 → pw 8/8/9; producers=18
  → 2/5/1), ~independent of width — mineable gap lives where capacity is tight vs demand (the
  capacity-asymmetry axis the single-cell EXP-0025 couldn't see; exactly the coverage Stage 4
  needs to avoid the uneven-strength regression).
Decisions: grid kept honestly in-claim (depth fixed at claimed 3; only width + producer density
  varied, all inside stage-5 envelope) so gaps are policy_weakness not distribution_mismatch.
  EXP-0028 artifacts are jsonl/logs only (no .pt → no git-lfs).
Blockers: none. Next: item 3 = Stage 4 v2 — augment champion's broad producer/orderer train set
  with THIS prioritized mined replay (not mined-only distill), re-eval transfer/stress.

## Stage 4 v2 — augment-not-replace (item 3) code + EXP-0029 launch
Status: IN PROGRESS (cloud run launched, pod 76ycecm9w2shrj). Suite 563 green (+3 tests).
Artifacts (code): tokenomics/eval/curriculum_replay.py — new augmented_train_configs (champions'
  broad sets + mined replay: producer = broadened_producer_train_configs(0-5,d3,w4) = 54 economies
  incl. the UNEVEN family the mined-only run dropped; orderer = build_scale_config(0-5,d3) = 6) +
  mined replay, per-distiller aligned seeds; run_stage4_replay gains augment/broad_seeds/broad_width;
  LEAKAGE GUARD drops replay tasks whose world_seed ∈ held_out (mining seeds 0-29 overlap held-out
  6-9 → a mined w4p4 s6 cell could reconstruct held-out scale-d3-s6); spec/RESULT record augment
  mode (stage4-v2). CLI --augment/--broad-seeds/--broad-width; dry-run shows mode + dropped seeds.
  tests/test_curriculum_replay.py +3.
Run: EXP-0029 = controlled flip of EXP-0027 to augment=True on the EXP-0028 diverse archive.
  Hyperparams matched to the CHAMPIONS (producer epochs 300 lr 0.01 = EXP-0021; orderer iters 400
  = EXP-0012) so the broad half reproduces the champion → any residual uneven gap is about
  data/augment, not under-training. eta 1.0, ε 0.01, eval-seeds 3, held 6,7,8,9, fixtures on.
  Dry-run on real archive: 26→train14/val6/blind6, leakage-drop seeds [7,9], AUGMENT mode. Fresh
  16-vCPU pod (--create-pod, auto-terminate).
Decisions: held everything constant vs EXP-0027 except the augment flip (+ champion-matched
  hyperparams) → isolates the augment effect. Success criterion: candidate keeps/beats champion on
  mined-val + blind AND sheds the 4 uneven-strength regressions (456-528 vs 487-584 in EXP-0027).
Blockers: none yet. Next: pull EXP-0029, check the 4 uneven cells + gate, write RESULT, then STOP.

## Stage 4 v2 augment (EXP-0029) — DONE, producer regression CLOSED (item 3, final)
Status: cloud run done (fresh 16-vCPU pod, auto-terminated), pulled to
  experiments/EXP-0029-stage4-augment. Suite 563. Controlled flip of EXP-0027 to augment=True on
  the EXP-0028 diverse archive; champion-matched hyperparams; leakage guard dropped seeds 7,9.
Result: augmented PRODUCER reproduces champion EXACTLY on held-out scale (producer_alone ==
  champion 491.3/496.7/589.3/544.7 every seed) AND joint candidate BEATS champion on all 4
  uneven-strength cells (+1.67/+2/+4/+4.67) — the 4 EXP-0027 regressions (−16..−128u) are GONE.
  First Stage-4 candidate with no producer-side transfer regression. Mined-val tie (−0.11u), blind
  tail +2. Residual: 2 regressions → 1 locus, ORDERER-side (scale-d3-s9 + sharedcap twin, −13u
  −2.4%, carried by candidate_orderer_alone) from the novelty/seed-14-heavy mined orderer replay.
  0 contract fails, 0 deaths. Gate promote_recommended=false (val tie + 2 regressions).
Why it matters: DECISIVELY confirms the EXP-0026/0027 diagnosis+fix — the uneven regressions were a
  replace-not-augment coverage hole; augment closes them. Producer half solved. Next lever = ORDERER
  (down-weight or drop mined replay into the orderer; its scorer distills only from proportional and
  can't absorb mined search-oracle headroom anyway).
Artifacts: augmented_train_configs + --augment/--broad-seeds/--broad-width + leakage guard in
  curriculum_replay.py; +3 tests. EXP-0029 candidate .pt files need git-lfs on commit. Not committed.
Blockers: none. Items 1,2,3 all DONE. (Stopping here per user.)

## Stage 4 v4 orderer oracle-relabel — collector BUILT (TDD, green), verification LAUNCHED on pod
Status: TDD build done + suite green (571). v4 verification run in flight on a fresh 16-vCPU CPU
  pod (t3rg8k2avnyhao, --create-pod, auto-terminate), pulling to EXP-0032. Follows EXP-0031 verdict
  (producer replay dead as a lever: closes 5.02u/5.27% of the 95.34u archived gap, train-split 0.0%,
  21/26 not_producer) → pivot to teaching the orderer from the mining bar's own search.
Artifacts (train/orderer_scorer.py): `collect_oracle_orderer_samples(configs, seeds, *, prod, ...)` —
  drives the product-stream v4 deployment env (entity_producer_obs + corrected streams) with the
  champion producer + per-cycle `_search_best_pol` (reused from orderer_oracle), snapshots each
  backlog-holding ProductStreamView, and relabels the target from the COMMITTED search routing:
  `_reduce_committed` (mean over active backlog slots) → `_center_clip_eligible` (center eligible
  logits, softmax shift-invariance, clip ±BOUND, floor ineligible). Same ScorerSample shape → existing
  masked `fit_scorer` consumes it unchanged. Defaults reproduce the mpc_search bar (even_split+
  proportional, n_samples=8, lookahead=6) so teacher == bar. New eval/orderer_representability.py =
  the decision gate: per task, on-task champ/search/oracle/bar units + `recovered_vs_teacher =
  (oracle−champ)/(search−champ)` (can the head reproduce the search it distilled?) and
  `recovered_vs_bar`; writes EXP layout (summary.csv/spec.json/RESULT.md) via --out-dir. Tests:
  test_orderer_scorer.py +6 (center/clip, slot-reduce, collector parity→fit_scorer, reproducibility,
  align guard, representability_row, emit layout).
Run: champion producer = EXP-0030 producer_candidate.pt (frozen, augmented — clean transfer + uneven
  wins), champion orderer = EXP-0012 orderer_scorer.pt, archive = EXP-0028; 2 high-gap train tasks
  (w4p5-s14, w4p3-s19), iters 300, horizon 100, eval-seeds 3. Staged the 3 read-inputs under
  tokenomics/_pod_ckpts/ (driver syncs the inner pkg dir, not repo-root experiments/).
Decisions: slot reduction = mean-over-active-slots (user-chosen); producer FROZEN at EXP-0030 so v4
  trains orderer only → clean attribution; verification measures the deployment-matched cross
  (champ_prod + oracle_ord) directly. Gate: mean recovered_vs_teacher ≥ ~0.5 → green-light full v4
  `run_stage4_replay` (orderer_teacher="oracle_relabel"); ~0 → architecture limit real, escalate.
Process correction: I first ran this verification LOCALLY, mislabeling it a "smoke" because it touched
  only 2 tasks — violating the CLAUDE.md HARD RULE (full-horizon per-cycle search + distill-to-
  convergence + eval_seeds>1 + multi-task = a real decision-gating run → cloud). Killed it, moved to
  the pod, and hardened CLAUDE.md: classify local-vs-cloud by COMPUTE PROFILE + PURPOSE not task
  count, with a "naming trap" note (calling it a smoke doesn't make it one). Renamed the function
  run_representability_smoke → run_representability_verification.
Blockers: none. Next: on pod completion, read EXP-0032 RESULT gate; if green-light, wire
  orderer_teacher="oracle_relabel" into run_stage4_replay (step 2) and launch fresh EXP as a pod run.

## Stage 4 v4 representability (EXP-0032) — DONE, INCONCLUSIVE (invalid probe) → do NOT proceed
Status: cloud run done (pod t3rg8k2avnyhao, auto-terminated), pulled to
  experiments/EXP-0032-orderer-oracle-representability. Collector worked (1267/1150 oracle samples;
  masked-MSE converged 0.79 s14, plateaued 2.05 s19).
Result: the 2 "high-gap" probe tasks have ~0 on-task orderer headroom under the FROZEN EXP-0030
  producer. teacher_headroom (search−champ) = 3.33u (0.7%) / 0.33u (0.07%): champ orderer already ≈
  search ≈ bar. The archived gaps (5.67u/~7u) were vs a WEAKER champion (producer_scorer_broad); the
  augmented EXP-0030 producer already ate them. recovered_vs_teacher = −1.4 / −31 is noise
  (denominator 0.33u). Two real signals: (1) oracle relabel is WORSE than champion orderer (−4.67 /
  −10.33u) even where MSE converged; (2) matches [[exp0006b-stochastic]] — the search's lever is
  temporal per-cycle re-routing, and a deterministic scorer distill collapses that diversity.
Decision: do NOT green-light the full v4 run_stage4_replay. Corrected the auto-verdict in RESULT.md
  (it wrongly read "architecture limit real"; actually confounded by task selection).
Next (open, needs user call): (a) re-select probe tasks by MEASURED headroom under EXP-0030 producer
  (keep teacher_headroom ≥ ~2% of champ_cu); if few clear the floor → augmented producer already
  closed the mined gap → orderer lever spent → STOP. (b) explain oracle<champion (lossy mean-slot/
  center-clip target vs deterministic-distill-loses-switching). Gate fix: require headroom floor +
  report absolute recovered units (ratio unstable as headroom→0).
Blockers: none. Code (collector + verification) is sound and reusable for the re-probe.

## Stage 4 v4 headroom scan (EXP-0032b) — DONE, DECISIVE: orderer lever spent → STOP v4
Status: cloud run done (pod ru1uervyedyhy8, auto-terminated $0.56/hr), pulled to
  experiments/EXP-0032b-orderer-headroom-scan. Re-probe after EXP-0032's invalid (near-zero-headroom)
  probe: measured teacher_headroom = champ_prod+search − champ_prod+champ_ord over ALL 26 archived
  tasks under the frozen EXP-0030 producer, floor 2% of champ_cu.
Result: 0/26 clear the 2% floor. Max headroom +9.33u (+1.91%, still under floor); distribution a
  tight band [−4.83%, +1.91%]; 8/26 NEGATIVE (champion orderer BEATS the myopic mpc_search, down to
  −26.33u). The mpc_search teacher v4 wanted to distill is not even a ceiling for the orderer under
  this producer.
Decision: STOP the v4 orderer oracle-relabel lever — no recoverable orderer headroom exists archive-
  wide; nothing to teach. EXP-0032's negative read was task selection (0 headroom), NOT an
  architecture limit. Closes the Stage-4 improvement thread: producer solved (EXP-0029), producer
  replay can't close residual (EXP-0031), orderer has no residual either (EXP-0032/0032b). Remaining
  mined gap vs entity_planner+mpc_search bar = producer/bar mismatch (bar's entity_planner producer),
  not a champion-orderer deficiency.
Artifacts: headroom_row + _qualifies + scan_headroom + --scan mode + _champ_search_units refactor in
  eval/orderer_representability.py; tests +2 (headroom_row, _qualifies). Suite 574.
Blockers: none. Next (only if further improvement wanted, different lever): stronger/less-myopic
  reference than mpc_search, or stochastic/elite orderer forms ([[exp0006b-stochastic]]) — but no
  evidence the current bar leaves recoverable orderer headroom to justify it.

## HANDOVER — EXP-0033 offline achievable-units upper bound (ceiling audit). NOT STARTED
Status: handover for a fresh agent. Read this entry + the three below it (EXP-0031/0032/0032b) first;
  full arc context in experiments/README.md index rows EXP-0023..0032b.
Question: is the champion NEAR-IDEAL, or does the weak searcher merely hide headroom? The Stage-4
  improvement loop is expert iteration and it is STALLED at a fixed point of its search operator
  (EXP-0032b: champion orderer >= mpc_search on 26/26 archived tasks, 8 negative; EXP-0031: producer
  channel closes 5.27%). All bars so far are RELATIVE (baselines/search/champion). Nothing measures
  distance to the true ceiling. One number decides the roadmap:
  - champion at ~>=90-95% of an offline optimum bound -> agent effectively ideal on this
    distribution; residual = feasibility, not policy. Close the arc (promotion decision, next axis).
  - champion at ~<=80% -> real headroom hidden by the myopic searcher -> build a stronger improvement
    operator (deeper lookahead / joint producer+orderer search / CEM over routing sequences) and
    expert iteration resumes with a live teacher.
Method (tiered relaxations; every tier MUST be a provable UPPER bound — relax, never approximate):
  domain is deterministic w/ scripted consumers, so the offline optimum is well-defined per economy.
  - B0: total demanded consumer units over horizon 100 (trivial demand cap).
  - B1: capacity-throughput cap — per capacity type, aggregate capacity-cycles over the horizon
    (optimistic: capacity pinned at max every cycle = restoration free & instant, a valid
    relaxation) divided by per-unit capacity cost; min over binding types, then min w/ B0.
  - B2: B1 + input-product coupling — expand each consumer product's recipe TRANSITIVELY (composite
    products consume primal outputs), charge the full capacity bundle per delivered unit, re-take
    the min over types. Analytic, numpy-only.
  - B3 (optional, only if B2-vs-champion is ambiguous): temporal LP adding TTL=2 windows, inventory
    limits, real degradation/restoration dynamics. NOTE: an LP solver (scipy) is a NEW dep — needs
    justification per CLAUDE.md; try B0-B2 first, they may already separate the two hypotheses.
Inputs (all exist, no training, no ckpts to stage):
  - Tasks: experiments/EXP-0028-topology-grid-mining/artifacts/task_archive.jsonl (26 records);
    rebuild each Configuration via eval/curriculum.py replay()/TaskRecord (world_spec generator-ref,
    generator build_scale_config). demand_trace fields are empty — derive demand from the replayed
    config's scripted consumers, horizon 100.
  - Champion + bar units per task: same dir, curriculum_matrix.csv (learner_units/reference_units).
    Optionally also score the EXP-0030 candidate stack (ckpts in EXP-0030 artifacts, git-lfs).
  - Recipes/capacities: read from the replayed Configuration (process sheets, producer configs);
    keep the bound code in eval/ (new module, e.g. eval/ceiling_bound.py) — domain/ stays pure.
Deliverables: experiments/EXP-0033-<title>/ (spec.json, summary.csv one row per task x tier w/
  champion ratio, RESULT.md w/ the decision call), index row in experiments/README.md, PROGRESS.md
  entry. Per-task output: B0/B1/B2(/B3), champion_units, bar_units, champion/bound ratio, binding
  constraint (which tier/type binds).
Build/run: TDD the bound locally against 1-2 archived tasks + a hand-checkable toy config (assert
  bound >= best observed units on EVERY task — an invariant test; any violation = bound bug). Full
  26-task sweep is pure arithmetic — borderline local-eval, but per the hard rule if it materializes
  the EXP dir, wrap it: run_pod_training.py --create-pod (16 vCPU default) --train "-u -m
  tokenomics.eval.ceiling_bound --archive ... --out-dir /workspace/tokenomics/exp0033" --pull
  "...:./experiments/EXP-0033-<title>". No GPU, no staged ckpts needed.
Pitfalls: (1) optimism direction — degradation/restoration/TTL/inventory must only ever be relaxed
  UP, else the "ceiling" undercuts real policies and the ratio lies; the invariant test catches
  this (matrix has 26 champion + 26 bar observations to check against). (2) fulfilled units metric
  = consumer units (cons✓ convention), NOT return. (3) suite must stay green (574); style rules:
  from __future__ import annotations, full type annotations, docstrings say why.
Parked (do NOT fold into EXP-0033, separate decisions): (a) promotion of the EXP-0030 candidate —
  ledger: 0 transfer regressions + uneven/scarcity/degen wins vs one mined-blind cell w4p3-s8
  -11.33u traced to seed-14 replay concentration; option = one hygiene rerun (replay priority = gap
  not novelty score, dup-cap <=2, orderer frozen) then promote if blind clean; val tie is the
  proven ceiling (EXP-0031/0032b), not a miss. (b) recoverability gate for Stage-3 mining
  (passes_gates + cheap cross-probes from gap_attribution/orderer_representability) so future
  archives only contain component-recoverable gaps. (c) GNN: no evidence it helps now; only
  defensible probe = GNN value model on the EXP-0009 dataset, gate = held-out R^2 > 0.
Blockers: none. Everything needed is committed-or-staged on branch new-model (staged, not committed).

## EXP-0033 offline achievable-units upper bound — DONE, NEAR-BOUND
Status: implemented + cloud run done (fresh 16-vCPU CPU pod, auto-terminated), pulled to
  experiments/EXP-0033-offline-ceiling-bound. Focused tests green.
Artifacts: new `tokenomics/eval/ceiling_bound.py` computes B0/B1/B2 analytic upper bounds from
  replayed `Configuration`s: exact seeded consumer demand over the observed matrix episode seeds,
  direct recipe capacity cap, and transitive input-product capacity cap. Capacity tiers are
  optimistic (aggregate producers, pinned capacity, free/instant upgrade slack, no TTL/inventory/
  funding/routing), and the invariant is every tier >= max(champion, bar) with 0.01 tolerance for
  the two-decimal observed matrix. New tests in `tests/test_ceiling_bound.py`: hand-checkable toy
  config, transitive bundle, EXP layout, and all-26 archived invariant.
Result: B0=B1=B2 are identical and **demand-bound on 26/26 EXP-0028 archived tasks**. Champion/B2
  mean 0.9830, min 0.9755, 26/26 >=95%; bar/B2 mean 0.9902, min 0.9819; 0 invariant failures.
Decision: current Stage-4 archive is effectively ceiling-saturated. The remaining mined gap is the
  final 1-2% of exact demanded units, not evidence for hidden headroom behind a weak `mpc_search`
  teacher. B3 temporal LP is unnecessary on this archive because exact demand already binds every
  row. Close the current Stage-4 improvement loop; next work is either promotion hygiene for the
  EXP-0030-style candidate or a new distribution/claim axis, not more expert-iteration on EXP-0028.
Blockers: none.
Next: superseded by the EXP-0030 promotion override below; keep the gap-priority/duplicate-cap
  amendment for future replay runs.

## Roadmap update — CURRICULUM_GAME_MASTER_APPROACH.md post-EXP-0033 revision
Status: DONE (doc-only, no code).
Artifacts: CURRICULUM_GAME_MASTER_APPROACH.md — (1) new "Executed History And Current
  Position (EXP-0023..0033)" section: stages 0-4 closed on the depth-3 scale family,
  fixed point real, family demand-saturated, EXP-0030 promotion was parked as human call
  at the time;
  (2) binding Amendments appended to Stage 3 (ceiling pre-flight via eval/ceiling_bound,
  recoverability gate via cross-stack probes, mining seeds disjoint from held-out) and
  Stage 4 (augment-not-replace, replay priority = gap not mining score + dup-cap <=2,
  val tie at zero recoverable headroom = ceiling w/ noise-aware pairing); (3) stale
  "Immediate Next Experiment" (EXP-0023 scaffold, long done) replaced by "Next
  Improvement Tracks" in priority order: Track A capacity-binding mining (no domain
  change, scarcity_variant reproducers mineable, pre-flight-gated grid), Track B
  scripted demand/shock traces (reserved TaskRecord trace slots, closed-loop
  adaptivity, Stage-5 legal primitives), Track C per-order TTL (Order.ttl default 2
  byte-identical, sub-orders inherit REMAINING ttl, producer orders stay constant,
  schedule = new orders only, consumers-block config, versioned obs bump, score
  unchanged; probe depth{4,5} x TTL{2,4,6,8} x {cold,primed}; anneal TTL~2*depth->tight;
  flat TTL 10-20 rejected as saturation re-creation).
Decisions: track order = cost-to-signal (A no-domain-change first); TTL design made
  per-order (user proposal) instead of global per-family constant — urgency becomes a
  learnable signal and a game-master dial; ADR required before Track C code.
Blockers: none. Next: superseded by the EXP-0030 promotion override below, then Track A grid spec
  + ceiling pre-flight wiring or Track C ADR.

## Roadmap restructure — Stage 4b inserted (tracks moved into stage order)
Status: DONE (doc-only). Follow-up to the post-EXP-0033 revision: the three improvement
  tracks moved from a trailing "Next Improvement Tracks" section into a formal
  "Stage 4b: New-Distribution Curriculum To A New Champion" between Stage 4 and
  Stage 5 — because Stage 4 had reached a fixed point on a saturated family and the next real
  improvement must come from a new distribution, the stage order must carry the implementation order.
Artifacts: CURRICULUM_GAME_MASTER_APPROACH.md — Stage 4b (purpose, Tracks A/B/C
  unchanged in content, stage-exit acceptance = promoted champion via full gates incl.
  mined-val improvement since real headroom exists, reject = all tracks saturate ->
  claim-boundary finding); Stage 5 gains entry criteria (Stage 4b exit); Status header
  updated (active stage = 4b, Stage 5 waits).
Blockers: none. Next: superseded by the EXP-0030 promotion override below, then start Stage 4b
  Track A (grid spec + ceiling pre-flight wiring) or Track C ADR.

## EXP-0030 promotion override — DONE
Status: PROMOTED by human approval on 2026-07-03; no rerun.
Artifacts: experiments/EXP-0030-stage4-freeze-orderer/RESULT.md now carries the promotion
  override note; spec.json has a top-level `promotion` object. CURRICULUM_GAME_MASTER_APPROACH.md
  now names the EXP-0030 producer candidate as the active producer checkpoint while retaining the
  EXP-0012 frozen orderer.
Reasoning: EXP-0030's advisory gate failed only on mined-val -0.165u and the blind w4p3-s8
  tail -11.33u, with 0 transfer/stress regressions. EXP-0031/EXP-0032b show there is no
  meaningful recoverable producer/orderer headroom left on the mined archive under the frozen
  stack, and EXP-0033 proves the archive is exact-demand saturated (champion/B2 min 0.9755,
  bar/B2 min 0.9819). The failing rows are therefore noise on a saturated family, while the wins
  are exactly where Track A should mine next: scarcity high-backlog, degenerate, and uneven cells.
Decision: promote as-is and skip the hygiene rerun for EXP-0028. Future Stage-4 replay still must
  use the sampling amendment (gap priority plus duplicate cap) so Track A does not repeat the
  seed-14 replay concentration failure mode.
Next: commit this as the EXP-0030 promotion boundary, then implement the replay sampling amendment
  before Stage 4b Track A.

## Stage-4 replay sampling amendment — DONE
Status: implemented locally; no cloud run required.
Artifacts: tokenomics/eval/curriculum_replay.py now separates `score_from_notes` (mining/archive
  score) from `priority_from_notes` (replay signal), and Stage-4 replay defaults to raw archived
  `gap=` priority instead of novelty-weighted mining score. Replacement replay now accepts a
  positive `max_per_record` cap; the Stage-4 driver and CLI default `--replay-duplicate-cap 2`
  and emit `replay_priority_signal=gap` plus `replay_duplicate_cap` into spec.json.
Tests: tokenomics/tests/test_curriculum_replay.py adds the gap-priority parser behavior, capped
  replacement sampling, oversized capped budgets, and Stage-4 spec metadata assertions.
Decision: this is the binding pre-4b fix for the EXP-0030 seed-14 concentration failure mode.
  Track A replay should use the default capped gap-priority path unless a run explicitly records
  a reason to override it.

2026-07-03 - Stage 4b Track A capacity-binding mining — gates + grid built (TDD)
Status: DONE (code, local). CURRICULUM_GAME_MASTER_APPROACH.md Stage 4b Track A implemented
  as the two NEW gates + capacity-tight grid, wired into the EXISTING Stage 3/4 machinery
  (curriculum_mining.grid_pool/mine_hard_cases + gap_attribution unchanged). Vertical TDD, one
  test->impl per slice, full suite 595 passed / 0 regressions.
Artifacts:
  - eval/ceiling_bound.py: `ceiling_bounds(..., include_upgrade_slack=False)` plumbed (default
    True = EXP-0033 byte-identical). KEY FINDING: the slack tier relaxes upgrades to free/instant
    (increases*horizon^2), so the ceiling is STRUCTURALLY demand-bound — capacity never binds and a
    literal "B2<B0" pre-flight would mine nothing. The no-slack (base-capacity) tier is the correct
    capacity-binding signal: a cell whose base throughput can't meet demand is one where investment
    skill separates strong from weak.
  - eval/scale.py: build_scale_config gains opt-in `quantity_max` / `capacity_scale` / `degradation`
    axes; every rng draw still happens when overridden so defaults are fingerprint-identical.
  - eval/curriculum_capacity.py (NEW, torch-free): `preflight_cells` (keep cells whose no-slack
    B2 binds below demand at EVERY world seed of the mine's range — see review-fix below),
    `capacity_tight_cells()` (shipped grid: ppl 2-3 below envelope, raised quantity,
    capacity_scale<1, one harsh-decay cell — all bind at every seed 20-119),
    `is_recoverable`/`recoverable_task_ids`/`filter_archive_recoverable` (recoverability gate over a
    gap_attribution matrix: keep only dominant∈{producer,producer_partial,orderer,mixed} with
    component gain ≥0.5u; drops joint-mismatch = the EXP-0031/0032b dead ends), CLI `preflight`
    (emits a curriculum_mining --grid) + `filter-recoverable`.
  - tests/test_curriculum_capacity.py (8), + no-slack tier test in test_ceiling_bound.py,
    + 3 capacity-axis tests in test_scale.py.
Smoke: preflight emits the 4-cell grid (4/4 capacity-bound); curriculum_mining --dry-run accepts it
  (params flow into the pool). Tiny torch smoke (1 cell ppl3/qmax16/capscale0.30, 1 seed, horizon 12)
  vs mpc_search: learner 46 < ref 60, gap 14, class=policy_weakness — a capacity-bound cell surfaces
  a real weakness the saturated depth-3 family never did. Premise validated.
Decisions: no domain change (all eval-layer); reuse mining + gap_attribution as-is; no ADR needed
  (Track A is spec+run per the roadmap). No promotion, no champion change — this only curates tasks.

Review fix (blocking, resolved): the first cut certified a cell at world seed 0 ONLY, but
  build_scale_config draws capacity per WORLD seed (strong/weak rng split), so a cell capacity-bound
  at seed 0 can be demand-bound (saturated) at seed 17 — mining sweeps the whole range, so saturated
  tasks would leak into the archive (the exact failure pre-flight exists to stop). Fix: `cell_ceiling`
  now bounds EVERY world seed (each with its own `_episode_seeds(ws, eval_seeds)` demand, mirroring
  curriculum_mining._evaluate) and keeps a cell only if the LOOSEST seed's capacity/demand ratio still
  binds (`b2_over_b0_worst < 1-margin`); `binds_frac` reports partial-bind cells instead of dropping
  silently. CLI split `--seeds` into `--world-seeds` (default 20-119, match the mine) + `--eval-seeds`.
  Regression test added (a cell that binds at seed 0 but not across 0-9 is now dropped). Shipped grid
  retuned so all 4 cells bind at every seed 20-119. Note (documented): no-slack B2 is a MINING SELECTOR,
  not a ceiling certificate — real policies CAN upgrade past it; never reuse it as a promotion bound.
  Minors also fixed: is_recoverable docstring warns to run gap_attribution WITH --orderer-cross (else
  not_producer silently drops orderer-recoverable tasks); scale.py qd loop de-obscured; quoted
  annotations dropped; flip-test comment aligned with its assert.
Blockers: none.
Next: launch the FULL Track A run on Runpod per CLAUDE.md HARD RULE — preflight (local) -> stage the
  EXP-0030 producer + EXP-0012 orderer ckpts under tokenomics/_pod_ckpts/ -> curriculum_mining --grid
  on the pod (wide seed sweep, eval-seeds 3, horizon 100, --workers) -> gap_attribution on the mined
  archive -> filter-recoverable -> Stage-4 augment replay (capped gap-priority) -> full promotion
  gates. Materialize as experiments/EXP-0034-stage4b-track-a-*. Then evaluate stage-exit (mined-val
  improvement now that headroom exists) vs claim-boundary reject.

## Stage 4b Track A FULL cloud run (EXP-0034) — DONE, premise VALIDATED, candidate advisory-fail
Status: full pipeline run on Runpod (3 fresh 16-vCPU CPU pods, --create-pod --cpu --python python3.12,
  each auto-terminated), pulled to experiments/EXP-0034-stage4b-track-a. Pre-flight + filter local.
Pipeline (each stage gated on prior):
  1. preflight (local, torch-free, 1.5s): 4/4 vetted cells capacity-bound across world seeds 20-119
     (b2_over_b0_worst<1, binds_frac 1.0 each) -> artifacts/vetted_grid.json. No-slack B2 = SELECTOR.
  2. mining (curriculum_mining --grid, 400 tasks = 4 cells x seeds 20-119, bar entity_planner+mpc_search,
     min_gap 2.0 min_win 0.6): 146/400 policy_weakness -> 119 archived (all 4 config cells).
  3. gap_attribution (orderer-cross ON): producer_closure_ratio 1.026, orderer 0.386; channels
     producer 83 / orderer 16 / mixed 14 / interaction_or_noise 6.
  4. filter-recoverable (local): 113/119 kept (dropped 6 interaction_or_noise).
  5. curriculum_replay --augment --freeze-orderer --depth 3 (champion EXP-0030 producer_candidate,
     orderer frozen EXP-0012; 57 train / 28 val / 28 blind; champion-matched defaults).
Result: capacity binding REVIVES producer headroom + the producer lever, closing the Stages-0-4
  demand-saturated fixed point. Candidate producer: val +14.80u mean (16/28 better), blind +8.33u mean
  (18/28 better), broad transfer (seeds 6-9) BYTE-IDENTICAL (0 forgetting, 0 transfer regressions,
  0 contract fails, 0 deaths). NOT Pareto: worst blind task -107u (capacity_scale0.35 ppl2 qmax14 s54)
  trips the conservative TAIL gate (blind_tail 721.67 < champion 772.33) -> objective_gate_pass=false,
  promote_recommended=false. FIRST Stage-4 producer replay to move val at all (vs EXP-0031's dead 5.27%
  closure / 21-of-26 not_producer on the saturated family).
Fix (blocking, in-run): gap_attribution ProcessPoolExecutor workers had NO thread pinning -> 16 workers
  x default torch intra-op threads oversubscribed the box (load 157 on 16 vCPU), 0/119 tasks done in
  62 min, log() never fired. Added to _init_worker: OMP/MKL/OPENBLAS_NUM_THREADS=1 + torch.set_num_
  threads(1) per worker (mining already pins the same on its subprocess env, curriculum_mining.py:432).
  Also added a startup banner + per-completion (k/total done) counter that flush under -u. Verified by
  local 2-task smoke, then relaunched: load 157 -> ~19, full run finished in ~15 min. Stale first pod
  (wrong ckpt path: staged _pod_ckpts one dir too deep INSIDE the package vs the inner root; driver
  syncs the inner pkg dir) terminated; corrected path landed ckpts at /workspace/tokenomics/_pod_ckpts.
Artifacts: experiments/EXP-0034-stage4b-track-a/{RESULT.md, artifacts/(preflight.csv, vetted_grid.json,
  task_archive.jsonl[119], recoverable_archive.jsonl[113]), summary.csv[400], gap_attribution/, replay/}.
  gap_attribution.py thread-pin + logging change (no test asserts the log string; curriculum_capacity
  tests still 9/9 green). .pt candidates need git-lfs on commit. NOT committed.
Decisions: mirror EXP-0030 recipe (--augment --freeze-orderer, champion-matched defaults); promotion is
  a HUMAN call (mean-improving tail-trading producer, same posture as the EXP-0030 incumbent override).
Blockers: none.
Next (open, human call): (a) promote the Track-A producer candidate (mean up on both held-out
  capacity-binding splits, broad family untouched) accepting the -107u blind-tail task, OR keep EXP-0030
  incumbent; (b) if promoting, a tail-robust variant (down-weight/exclude the s54-class outlier, or
  priority reweighting) before promotion; (c) Track B (harsher-degradation / deeper-chain capacity
  regimes) if more headroom wanted. Commit EXP-0034 (+ gap_attribution fix) with git-lfs for the .pt.

## 2026-07-04 - EXP-0034 blind-tail task diagnosed (s54, -107u) — real skill trade, not noise
Status: DONE (local single-task diagnostic; quick-eval class: 1 task, 2 ckpts, 3 paired seeds + 1 per-cycle trace).
What: reproduced the blind-tail regression exactly (champion 878.0 / candidate 771.33, seeds 54-56) and traced
  the mechanism. Behavioral, not instability: candidate loses on ALL 3 seeds (-115/-165/-40), 0 deaths,
  cap_min identical. Mechanism = upgrade misallocation at the layer-0 primal bottleneck: s54's layer-0 is
  dominated by p0_0 (initial max 48 vs sibling 6; one +31 upgrade costs 35). Champion buys it by cycle 10
  (max 48->79, held near max); candidate NEVER upgrades p0_0 in 100 cycles (runs 28-46/48) and spends
  upgrades mid/late-chain (p1_1 193->234; fewer p2 upgrades). Layer-0 starves the chain: repl_rej 135->171,
  units even to cycle 30 then steady compounding divergence. Track-A replay traded away the champion's
  "upgrade the tight primal source first" behavior; augment protected the broad family, not this in-family skill.
Artifacts: EXP-0034 RESULT.md "Diagnosis" section (numbers + reading). Diag script was throwaway (deleted).
Decisions: none taken - promotion still the open human call, now with the trade-off named: promoting accepts
  losing small-primal-bottleneck economies (~1/3 regressing tasks likely share the signature - unverified),
  OR next replay round mixes in primal-tight cells (grid axis: scale down ONE layer-0 producer) to keep both skills.
Blockers: none.
Next: human promotion call on the Track-A candidate; if "train past it" chosen, add primal-tight cells to the
  Track A grid and rerun augment replay on the pod.

## 2026-07-04 - EXP-0034 train-closure diagnostic — harvest bottleneck = DISTILL FIDELITY (new tool + CLAUDE.md rule)
Status: DONE. New HARD RULE in CLAUDE.md ("Root cause before next strategy"): no new strategy/axis/arc
  until the failure cause is diagnosed with evidence and written down; discriminating diagnostic first.
What: built tokenomics/eval/train_closure.py (reusable closure-by-split diagnostic: candidate-only rollouts,
  champion+bar joined from gap_attribution.csv, worker pool with thread pinning; tests 4/4, local smoke) and
  ran it on a fresh 16-vCPU pod (image runpod/base:0.7.0-ubuntu2004, --cpu --python python3.12; pod created,
  ran, pulled, TERMINATED - no idle billing). 57 train tasks x 3 paired seeds, horizon 100.
Result: train closure 0.159 aggregate / 0.098 median (vs val 0.19, blind 0.12) -> train ~ val ~ blind =
  NO generalization gap. Sharper: candidate reaches 92.7% of its OWN distill teacher on its OWN train split
  (teacher stack 1093.3u vs candidate 1013.0u/task; 13% of teacher-champion gap captured; 26/57 train tasks
  regress vs champion; closure quartiles -0.51/0.10/0.57). EXP-0004 pattern: CLONING-FIDELITY gap.
Decisions: (a) more Track-A mining rounds ON HOLD - the pre-staged EXP-0035 primal-tight round-2
  record dir was removed (user call; nothing had launched); the primal_capacity_scale axis, primal_tight_cells()
  grid and tests stay in the tree, valid + ready if a distill fix revives the round; (b) fidelity wall is on Track B's critical path too (same distill
  recipe harvests any mined archive); (c) EXP-0034 candidate promotion question unchanged (means real).
Blockers: none.
Next (root-cause layer 2, pick one): single-task overfit discriminator - distill on ONE mined task with
  generous epochs; teacher matched -> optimization/capacity at scale (bigger scorer, staged DAgger a la
  EXP-0004); not matched -> representability (scorer can't express planner lookahead). THEN decide distill
  fix vs accept ceiling; only after that Track B / EXP-0035 relaunch.

## 2026-07-04 - Single-task overfit probe: fidelity wall = MULTI-TASK OPTIMIZATION (representability CLEARED)
Status: DONE. Root-cause chain closed per the CLAUDE.md rule: low harvest -> not generalization (train
  closure ~ val ~ blind) -> not representability, not single-task capacity (this probe) -> multi-task
  optimization/interference at scale.
What: new tokenomics/eval/overfit_probe.py (+4 tests; suite green) - distills the producer scorer on ONE
  mined task with the exact replay recipe (entity_planner teacher, proportional-orderer collect at seed 0),
  2000 epochs, arms hidden {64=champion arch, 256}, deploys with frozen champion orderer on paired seeds,
  verdict vs bar_producer_champ_orderer_units from gap_attribution. Fresh 16-vCPU pod, created+terminated.
Result: 4/6 arms MATCH the teacher (fidelity >= 0.98). Closures solo vs full-replay: s42 0.97 vs 0.10,
  s96 0.90 (256: 1.04) vs -0.16, s118 0.66/0.77 vs 0.10. The 64-wide champion architecture CAN express the
  planner behavior it fails to learn at scale; 256 ~ 64 solo. Cause = one scorer fitting 121 topologies
  (57 mined + 64 broad augment) in 400 epochs averages away per-task investment behavior. Recipe's
  collect-vs-deploy covariate shift (proportional/seed-0 collect) did NOT block solo fidelity - secondary.
Artifacts: EXP-0034 overfit_probe/ (CSV+summary), RESULT.md probe section, eval/overfit_probe.py.
Decisions: mining stays ON HOLD; fix trials now licensed (cause named): at-scale epochs >> 400, at-scale
  width >= 256 (multi-task capacity untested), sample balance, staged/rung distillation (EXP-0004 cure).
  Each = one pod replay rerun judged by eval.train_closure.
Blockers: none.
Next (human pick): cheapest fix trial = augment-replay rerun with --epochs 2000 (+optional --hidden 256 arm),
  then train_closure; if closure jumps, EXP-0035-style round 2 + promotion path reopen.

## 2026-07-04 - Diagnostics registry: docs/DIAGNOSTICS.md (known problems -> verification checks)
Status: DONE. New registry docs/DIAGNOSTICS.md, wired into the CLAUDE.md root-cause rule ("check the
  registry first") + Conventions ("new diagnosed cause -> add entry").
What: 9 evidence-backed entries, each = plain-English problem statement + discriminating check (tool +
  command) + outcome branches + EXP citation: (1) low harvest -> train_closure (STANDING GATE for every
  replay/distill experiment; cheapest detector), (2) fidelity wall -> overfit_probe (TRIGGERED only - answer
  is a property of arch/obs/recipe/family, re-run only when one changes), (3) weak-teacher vs weak-cloning
  (EXP-0004), (4) single-task regression trace: skill trade vs noise (s54), (5) producer/orderer channel
  attribution (gap_attribution/swap), (6) memorization vs transfer (EXP-0009 held-out R^2), (7) return vs
  fulfilled-units objective (ADR-0040), (8) signal_check learnability, (9) pod thread-oversubscription.
  Plus "Unverified problems" inbox: humans drop plain-English problem statements, agent attaches checks.
Decisions: train_closure = standing output of every replay experiment; overfit_probe = on-trigger
  discriminator, not a per-experiment gate (answered user question: don't re-verify representability when
  nothing it depends on changed).
Blockers: none.
Next: unchanged - human picks distill fix trial (augment-replay rerun --epochs 2000, optional --hidden 256),
  judged by train_closure per registry entry 1.

## 2026-07-04 - EXP-0035 distill fix trial: epochs 2000 BREAKS the fidelity wall (all gates pass)
Status: DONE. Cheapest licensed fix (per the EXP-0034 root-cause chain: multi-task interference at
  400 epochs) tried first: EXP-0034 augment-replay rerun byte-identical except --epochs 2000, hidden 64
  unchanged. Two fresh 16-vCPU pods (replay, then train_closure standing gate), both auto-terminated.
Result: closure of mined (bar-champion) gap: train 0.159 -> 1.096 (median/task 1.04), val 0.19 -> 0.860
  (+68.3u/task), blind 0.12 -> 1.000 (+68.4u/task; candidate mean = bar mean). Worse-than-champion:
  train 26/57 -> 3/57, val 12/28 -> 1/28, blind 10/28 -> 1/28. s54 (EXP-0034's -107u tail-gate task,
  upgrade-misallocation) heals to +61.7u, above its own bar. Objective gate PASSES (first Stage-4 replay
  ever): val 1106.7>1038.4, blind tail 813.3>772.3, 0 contract/transfer/death failures, broad seeds 6-9
  clean. Fit MSE was still falling at the old 400 cutoff (0.045@400 -> 0.027@600) - under-training
  signature confirmed live.
Artifacts: experiments/EXP-0035-distill-epochs/{RESULT.md, replay/ (spec, summary, candidate .pt via lfs,
  val/blind/transfer CSVs), train_closure/}; README index row.
Decisions: (a) --epochs 2000 = new replay posture; train_closure stays the standing gate per DIAGNOSTICS
  entry 1; (b) mining OFF HOLD - Track-A round 2 (primal-tight cells, code ready in tree) + Track B now
  licensed; (c) EXP-0034 candidate is obsolete - EXP-0035 candidate strictly dominates it.
Blockers: none.
Next (human): PROMOTE replay/artifacts/producer_candidate.pt over the EXP-0030 incumbent (passes every
  gate incl. the tail gate the EXP-0034 candidate failed; promotion is human-approved by design). Then
  pick: Track-A round 2 (primal-tight) or Track B (scripted demand/shock traces) for the next mine.

## 2026-07-04 - EXP-0035 candidate PROMOTED (human approval) — new champion producer
Status: DONE. Human approved promotion of the EXP-0035 candidate over the EXP-0030 incumbent.
What: active stack is now `experiments/EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt`
  (producer) + EXP-0012 `orderer_scorer.pt` (orderer, unchanged). Recorded in EXP-0035 RESULT.md and
  the experiments/README.md index (EXP-0030 row marked superseded). Legacy champion.json untouched
  (historical baseline only).
Basis: first Stage-4 replay to pass EVERY gate — val 1106.7>1038.4, blind tail 813.3>772.3, 0
  contract/transfer/death failures, broad seeds 6-9 clean; beats the incumbent on train/val/blind
  of the capacity-binding family (closure 1.096/0.860/1.000) at zero broad-family cost.
Decisions: future replay rounds train against THIS champion (champion ckpt arg in curriculum_replay /
  gap_attribution / train_closure now points at the EXP-0035 .pt); mined bars must be re-attributed
  vs the new champion before reuse (old gap_attribution.csv references the EXP-0030 stack).
Blockers: none.
Next: pick the next mine — Track-A round 2 (primal-tight cells, code ready) or Track B (scripted
  demand/shock traces).

## 2026-07-04 - EXP-0036 Track-A round 2 (primal-tight): mined 81 tasks, headroom is ORDERER-owned; probe seed bug fixed
Status: IN PROGRESS (representability probe pod running). Pipeline so far all on fresh CPU pods, each terminated.
What: preflight 4/4 primal_tight_cells() bind at capacity-0 (every world seed 20-119) -> mine 400 vs
  entity_planner+mpc_search with the EXP-0035 champion (400->96 policy_weakness->81 archived, cells
  26/35/2/18; gaps min 2.7/med 53/mean 99.9/MAX 973u - order of magnitude beyond round 1) ->
  gap_attribution over combined archive (113 round-1 retention + 81 new).
Result (attribution): round-2 headroom is ORDERER-channel - 45 orderer/15 mixed/11 producer/10 noise;
  orderer-only cross recovers 7172u of 8088u (89%), producer-only 14% (closure ratios 0.928 vs 0.310);
  6/81 tasks are champion SYSTEM-DEATH that routing alone prevents (monster gaps 200-973u). Round 1 was
  the inverse (70% producer). Round-1 sanity: 54/113 old tasks have no positive gap left - the promoted
  champion absorbed round 1. Decision (user): orderer-probe-first (freeze-orderer replay would strand
  ~86% of the mined headroom).
Bug found + fixed (DIAGNOSTICS entry 10): orderer_representability evaluated archived tasks at episode
  seeds range(eval_seeds)=0-2 instead of _episode_seeds(record.episode_seed,...) - a different demand
  trace (champ "alive 957u" vs archived full-death 973u gap on s77). First probe pod killed mid-run.
  Fix: all evals + oracle collection anchored at the archive convention (collector now covers all eval
  seeds); regression test test_headroom_row_anchors_at_record_episode_seed; suite 608 passed. Caveat
  propagated: EXP-0032b scan_headroom shared the bug - its "0/26 orderer headroom" survives only via
  EXP-0033's exact-seed ceiling, not its own measurement.
Artifacts: experiments/EXP-0036-track-a-round2-primal-tight/{spec.json, RESULT.md (in progress),
  artifacts/(preflight.csv, vetted_grid.json), mine/, gap_attribution/}; README index row.
Blockers: none. Probe ETA hours (sequential search collection ~30 min/task x 10).
Next: probe verdict -> recovered_vs_teacher high = full orderer oracle-relabel replay on the 81-task
  archive (plan to user first); low = architecture discussion, no new strategy without root-cause.

## 2026-07-04 - Repo-root docs: ARCHITECTURE.md + GLOSSARY.md (plain-English training architecture)
Status: DONE.
Task: user asked for a comprehensive plain-English document of the existing training architecture
  (components + roles) for a non-technical reader, with full experiment-history narrative, plus a
  glossary of the training/eval terms used. No invented terms - repo vocabulary only.
Artifacts: ARCHITECTURE.md (repo root) - what the project is; how training works (imitation/
  distillation pipeline, champion/candidate/gates, game-master loop); 12 component sections
  (domain, env, baselines/planners, search bar, the two score models, distillation methods,
  champion+gates, curriculum pipeline, eval harness, cloud infra, records system, legacy);
  one-round walkthrough; 6-arc history EXP-0001..0036; current state. GLOSSARY.md (repo root) -
  training/eval/curriculum/infra terms grouped in 5 sections; domain terms deliberately NOT
  duplicated, deferred to CONTEXT.md (user call). Cross-links to docs/rl-training-architecture.md
  (technical RL internals), CURRICULUM_GAME_MASTER_APPROACH.md, experiments/README.md.
Decisions (user): scope = active + legacy + history narrative; glossary = training terms only,
  link CONTEXT.md; audience = non-technical stakeholder.
Blockers: none. EXP-0036 probe still running in parallel (separate entry above).
Next: none for docs; keep ARCHITECTURE.md section 6 (current state) fresh on promotions.

## 2026-07-04 - ARCHITECTURE.md rewritten DEEP (rl-training-architecture.md style) + GLOSSARY extended
Status: DONE.
Task: user feedback on the first ARCHITECTURE.md draft - components not explained deeply enough
  (how the searcher acts, how teachers behave, mining/eval examples). Rewrote in the style of
  docs/rl-training-architecture.md: execution-order narrative, worked numeric examples, tables,
  alternatives-considered tables.
Artifacts: ARCHITECTURE.md v2 - 12 sections. New depth (all verified against code, not memory):
  planner score formulas from domain/producer_planner.py (restore threshold 0.60, stock target 3,
  tie-group replenishment); orderer model shape + 9/31 feature split + softmax/largest-remainder
  worked routing example (train/orderer_scorer.py, env/allocator); mpc_search mechanics
  (10 candidates = even_split+proportional+8 sampled, seeded by (seed,cycle), deepcopy ~3ms,
  lookahead 6, 60x cost arithmetic) from eval/orderer_oracle.py; teacher targets (log makeable
  qty; oracle relabel reduce-over-backlog + center-clip); gate heads story (11442/24000 FPs);
  mining gates/ranking math (min_gap 2.0, win_frac 0.6, score = 1.0*norm_gap + 0.5*novelty,
  novelty d/(1+d) on descriptor) from eval/curriculum_mining.py; failure-class precedence table;
  ceiling bound B0/B1/B2; gap-attribution 4-stack table; replay priorities (0.01+gap)^eta +
  gap-not-score lesson; closure formula (sum-based); EXP-0036 as the running worked example
  (s77: champ 0 dead / bar 973.3 / cross 997.3), one-round summary table. GLOSSARY.md: added
  entity view, product stream, logit/softmax, gate head, masked loss, lookahead/rollout,
  win fraction, normalized gap, novelty; mpc_search entry expanded.
Blockers: none. EXP-0036 probe still running (separate thread).
Next: keep ARCHITECTURE.md section 12 (current state) fresh on promotions.

## 2026-07-04 - EXP-0036 orderer representability probe ABORTED at 6/10 (verdict forced) + partial results archived
Status: DONE (probe stopped early on purpose; diagnosis OPEN).
Task: user flagged bad probe results mid-run; assessed and agreed to stop. 5/6 completed rows
  show zero-or-negative recovery, so the "high recovery on most tasks" green-light branch was
  already mathematically dead - the remaining 4 rows could not change the verdict. Killed the
  local monitor task, stopped + terminated pod pjw7prk43xbq47. Probe writes summary.csv only
  after ALL tasks, so the 6 [repr] log rows were hand-assembled into
  experiments/EXP-0036-track-a-round2-primal-tight/orderer_probe/ (summary.csv + NOTE.md +
  probe.log). No distilled ckpts/samples existed on the pod (module never saves them) - nothing
  else was lost by aborting.
Artifacts: orderer_probe/summary.csv (6 rows), NOTE.md (read + diagnosis status), probe.log.
Key numbers: recovered_vs_teacher = 0.0 (s77 death), 0.571 (s45 death - the ONE success, 560 of
  982u), 0.0 (deg9-s45 death), -0.002 (deg9-s62), -0.769 (s68 alive, oracle 0 vs champ 253),
  -8.073 (s29 alive, oracle 0 vs champ 953). Pattern: distilled orderer delivers EXACTLY 0 units
  on 5/6 tasks incl. alive worlds; final masked-mse anti-correlates with deploy outcome (best
  fit 0.224 -> worst deploy).
Decisions: verdict = NOT a green light for full oracle-relabel replay; per DIAGNOSTICS entry 2
  the failure sits in the "representability or recipe" branch - lead hypothesis collect-vs-deploy
  covariate shift (trained only on search-trajectory states; deploy drift + capacity-tight worlds
  -> system death -> 0 units), NOT yet proven. Root-cause rule: no next strategy until the
  discriminating check runs.
Blockers: probe saves no artifacts; rerun needs module change (save samples + distilled ckpt,
  report in-sample routing agreement, deployed-episode death/capacity trace).
Next: proposed discriminating diagnostic (needs user go, pod ~1.5-2h): instrumented rerun on 3
  tasks (s29 alive-catastrophic, s77 death-0.0, p0.22-s45 death-0.571) to split covariate shift
  vs representability; then pick fix (DAgger-style deploy-state collection vs obs/arch change).

## 2026-07-05 - EXP-0036 orderer representability DIAGNOSED: oracle-relabel targets are mostly noise (target/recipe failure)
Status: DONE (diagnosis complete, evidence-backed). Arc decision OPEN (user).
Task: instrumented probe retry (pod 842xx0vutkdhln, 3 tasks s29/s77/s45, saved samples +
  distilled ckpts + deploy traces) came back failing the gate (mean recovered_vs_teacher
  -0.159). Analyzed the saved samples locally to settle DIAGNOSTICS entry-2's branch.
Evidence (new DIAGNOSTICS entry 11):
  - 42-50% of oracle targets are FLAT (search committed even_split -> centered target = zeros).
  - Variance explained vs a zero-predictor (= route uniformly) on TRAIN states: 25% (s29),
    10% (s77), 64% (s45) - the converged student barely beats uniform routing in-sample.
  - Top-1 teacher agreement 0.50-0.60 vs chance 0.33-0.50 (2-3 eligible producers/stream).
  - Identical recipe, different torch init: deploy swung 0 -> 800 units (s29) between probe 1
    and retry - fitting noise, unstable by construction.
Diagnosis: the search commits ONE global routing per cycle among near-tied candidates (incl.
  sampled Gaussian vectors); sliced into per-stream local targets, the informative part is
  global + tie-arbitrary -> per-stream MSE regression has little learnable signal. NOT a
  representability wall (s45 recovered 370-560u twice), NOT plain covariate shift.
Artifacts: orderer_probe_retry/ (summary.csv, per-task debug/), orderer_probe/NOTE.md updated
  with resolution, docs/DIAGNOSTICS.md entry 11 added.
Decisions: full orderer oracle-relabel replay on the 81-task archive = REJECTED as designed
  (would distill noise). Next lever = user choice: (a) denoise teacher signal (average committed
  routings over repeated searches / margin-filter / drop Gaussian candidates), (b) global
  context in orderer obs, (c) deploy-time light search (known ~1000u headroom, cost knob).
Blockers: none.
Next: user picks harvest lever; then wrap EXP-0036 (RESULT.md probe section, README index row,
  memory update, delete tokenomics/_pod_ckpts, commit).

## 2026-07-05 - New project skill: /challenge (adversarial review of proposals + EXP results)
Status: DONE.
Task: created `.claude/skills/challenge/SKILL.md`. Puts the agent in hostile-reviewer mode
  over a proposal or EXP record: trust hierarchy (raw artifacts > producing code > RESULT.md
  prose > chat claims; only raw evidence admissible without reverification), reverify every
  number from summary.csv/logs, audit the driver/script for correctness bugs (leakage, seed
  handling, config mismatch, metric bugs, selection bias), attack the inference (alternative
  explanations, DIAGNOSTICS.md match, falsifiability), verdict per claim
  (UPHELD/OVERTURNED/UNVERIFIED with cited evidence). Cheap checks run now; anything needing
  a real run stays UNVERIFIED with the exact pod command (pod-only rule respected).
Artifacts: .claude/skills/challenge/SKILL.md
Decisions: review-only skill - delivers verdict, does not fix; fixes offered only after.
Blockers: none.
Next: use as `/challenge <EXP dir | proposal | last result>`; natural first target = the
  EXP-0036 probe diagnosis or the next harvest-lever proposal before it goes to the pod.

## 2026-07-05 - /challenge on the "search-as-judge / actor-critic" harvest proposal + learnability-gated plan into Stage 4b Track A
Status: DONE (review delivered; plan recorded). Arc decision still OPEN (user picks whether to run Stage A0).
Task: user ran /challenge on a proposed orderer-harvest fix (soft-label candidate tables,
  a learned critic, advantage-weighted actor, global context, critic-gated deploy-time search).
  Reviewed adversarially against raw evidence, then wrote my own plan and inserted it as a
  binding amendment to Stage 4b Track A.
Review verdicts (all cited to raw artifacts, not prose):
  - Recomputed the EXP-0036 diagnosis from raw samples.pkl.gz: variance-explained over a
    route-uniformly baseline = 25.4%/9.8%/64.1% (s29/s77/s45) — reproduces DIAGNOSTICS entry 11
    (25/10/64) exactly; flat-target frac 43/50/29%, chance-top1 0.50/0.33/0.33. Diagnosis UPHELD.
  - Learned critic (proposal steps 3,6) OVERTURNED: EXP-0009 RESULT shows units-to-go from
    global_state has NEGATIVE held-out R^2 across topologies (ridge -0.10..-6.6, MLP -2.5..-3.4;
    best held-out corr ~0.15) and explicitly says "do not rescue the value model; expect the same
    wall elsewhere in the arc." Critic-picks-best / critic-gated search depend on the one
    generalization the arc disproved.
  - Candidate soft-labels over routes A/B/C (steps 1-2) OVERTURNED: eval/orderer_oracle.py
    _candidate_list emits 2 named + 8 per-cycle Gaussian vectors re-drawn each cycle (seeded by
    (seed,cycle)); no stable route to label; deploy action is a continuous per-stream simplex.
  - Global context (step 5) OVERTURNED as motivated: its rationale is the covariate-shift story
    the diagnosis ruled out; real mechanism is global-commit-sliced-to-per-stream (coupling), and
    per-stream deciders stay independent, so global features raise an upper bound but don't
    reconstruct a joint routing.
  - Advantage-weighting from OFFLINE search scores (step 4) = the one salvageable idea (no critic,
    inside entry-11's denoise branch); folded into my plan as the loss form.
  - No falsifiable bar in the proposal (7-step roadmap, no gate) — flagged.
Plan recorded (CURRICULUM_GAME_MASTER_APPROACH.md, Stage 4b Track A amendment "learnability-gated
  orderer harvest"): three competing hypotheses (H1 high-variance labels / H2 global coupling /
  H3 no headroom), each a different fix; Stage A0 = one instrumented pod run emitting a decision
  table (headroom evaluated FIRST as a STOP gate: oracle~=even_split -> done; then single-vs-
  averaged-vs-+global variance-explained); Stage A1 = distill denoised target with advantage-
  weighted regression, gate on deployed units >= champion; Stage A2 = deploy-time light search on
  a CHEAP heuristic trigger (never a learned critic). Binding constraints: no learned value model
  (EXP-0009), no route-classification (Gaussian candidates). The cheapest run (A0) can end the
  orderer track — discriminating-diagnostic-first per the HARD RULE.
Artifacts: CURRICULUM_GAME_MASTER_APPROACH.md (Track A amendment); this entry. No code, no runs.
Decisions: proposal's critic + soft-label mechanics rejected on evidence; harvest gated on the A0
  learnability probe before any cloud training on architecture/global-context.
Blockers: none.
Next (user go needed): build the Stage A0 probe driver (extend the instrumented probe to emit the
  headroom row + single/averaged/+global VE table), TDD'd locally on 1 task via --dry-run, then
  launch full on the pod.

---

## 2026-07-05 — Stage A0 orderer-harvest probe: built (TDD) + launched on pod

Status: DONE (build) / RUNNING (pod). Implemented the Stage A0 learnability gate from the
CURRICULUM_GAME_MASTER_APPROACH.md Track A amendment, TDD'd locally, full suite green (617
passed), and launched the decision-gating run on a fresh CPU pod (user approved pod start).

What A0 answers (plain English): the EXP-0036 orderer harvest failed at the SIGNAL — distilling
the search's single per-cycle committed routing gave a per-stream target that is mostly noise
(variance-explained 25/10/64% on the 3 probe tasks). A0 emits one decision table per task to pick
the branch: (H3) is there any orderer headroom at all — does the search-oracle beat even_split? if
not, STOP; (H1) does AVERAGING K=4 independent per-cycle searches (committing the mean vector to
advance a single trajectory) raise variance-explained over the single search? a jump => the single
target was tie-arbitrary noise the search can denoise => proceed to Stage A1. H2 (global-context
obs) deferred: it is an obs-contract change and the amendment makes it conditional on H1 staying
flat.

New code (all TDD, one test -> one impl):
- `train/orderer_scorer.py::masked_variance_explained` — VE = 1 - model_mse/zero_mse over eligible
  entries (the metric the diagnosis recomputed by hand from samples.pkl.gz); now first-class.
- `scorer_sample_metrics` now emits `sample_variance_explained`.
- `train/orderer_scorer.py::collect_averaged_oracle_orderer_samples` — the H1 denoiser: K
  independent searches/cycle (RNG stride 1_000_003), commit the MEAN routing to advance + as
  target. Invariant test: k_searches=1 is byte-identical to the single-search collector.
- `eval/orderer_representability.py::headroom_row` gains `even_split_cu` / `proportional_cu` (H3
  STOP-gate table); `a0_probe_row` (headroom + VE_single vs VE_averaged); `run_a0_probe` + `_emit_a0`
  (pullable EXP layout, headroom-first RESULT.md); CLI `--a0 --k-searches`.

Artifacts: 5 new tests in tokenomics/tests/test_orderer_scorer.py (29 in file, all pass; full
suite 617 passed). Staged pod inputs under tokenomics/_pod_ckpts/ (producer_champion_exp0035.pt =
EXP-0035 replay/producer_candidate.pt; orderer_scorer.pt = EXP-0012; task_archive.jsonl = EXP-0036
mine). Pod x1vvuz1zg3lekp (fresh, 16 vCPU cpu5c), --a0 --k-searches 4 --iters 1000 --horizon 100
--eval-seeds 3 on the 3 probe tasks (s29/s77/s45); pulls to
experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0. Launch log /tmp/a0_pod_launch.log.
Decisions: kept iters=1000 + the 3 probe tasks to match the EXP-0036 retry probe fidelity so VE is
comparable; H2 not built (conditional lever). No champion ckpt touched (producer frozen throughout).
Blockers: none.
Next: when the pod pulls back, READ HEADROOM FIRST (search_cu vs even_split_cu per task) then the
VE jump (ve_averaged vs ve_single). Branch per the amendment: H1 jump -> Stage A1 (advantage-
weighted distill of the denoised target, gate deployed units >= champion); no jump + headroom ->
Stage A2 (deploy-time search); no jump + no headroom -> orderer track done (claim-boundary). Then
delete tokenomics/_pod_ckpts/.

---

## 2026-07-05 — A0 probe: added named-policy representation comparison; relaunched

Status: DONE (build) / RUNNING (pod ua7llllthh9kol). User asked whether the teacher could return a
NAMED policy (don't-care / even_split / proportional / blend) instead of a single continuous
per-producer routing. Assessed: it attacks the single-choice noise from a cleaner angle (named
policies are state-reproducible, so the RNG-tie-arbitrary noise vanishes; small categorical output
is more learnable; "don't-care" names the no-lesson rows) and it reverses my earlier soft-label
rejection (that assumed Gaussian candidates — named candidates make soft labels well-defined). The
one thing that decides viability: does the search headroom live INSIDE the named vocabulary
(per-cycle even_split/proportional switching) or in the continuous Gaussian routings? The A0
`even_split_cu`/`proportional_cu` columns DON'T answer it — those are FIXED policies (~810-840 on
s29); a per-cycle CHOICE among named policies is strictly stronger and sits between that and the
full search (1070).

Added the discriminating column to the A0 probe (TDD, +1 test, file now 30 pass):
- `eval/orderer_representability.py::_named_search_units` — per-cycle search with `n_samples=0`
  (named candidates only, still switching per cycle, no Gaussian samples).
- `a0_probe_row` now emits `search_named_cu` + `named_capture_frac = (search_named - even_split) /
  (search_full - even_split)`. ~1 => named-policy teacher captures the lever (switch to a
  categorical/blend head, your idea); ~0 => continuous routings matter (keep continuous target, H1).
- `_emit_a0` RESULT.md read order now: headroom -> representation -> denoise. CLI print + means updated.

Ops: terminated the first in-flight pod x1vvuz1zg3lekp (only s29 done) so one clean pull carries all
three answers per task; first relaunch hit a transient RunPod `pod get` race (driver auto-cleaned the
pod ob2iu0u6bkaykw and bailed); retry succeeded on pod ua7llllthh9kol (same --a0 --k-searches 4
--iters 1000 command). Champion untouched, producer frozen.
Decisions: relaunch (not a cheap 2nd run) chosen for one authoritative artifact; ~1h of s29 recompute
accepted (cheap CPU). Blend variant (alpha*even_split + (1-alpha)*proportional, 1 coeff/stream) noted
as the middle-ground representation if named_capture is close-but-not-1.
Blockers: none.
Next: on pull, read summary.csv per task in order headroom -> named_capture_frac -> ve jump. If
named_capture ~1 => spec the categorical/blend orderer head (new mini-arc); elif ve_averaged jumps
=> Stage A1 continuous-denoise distill; elif headroom but neither => Stage A2; else track done. Then
delete tokenomics/_pod_ckpts/.

---

## 2026-07-05 — Docs: policy-choice (named-teacher) proposal written into the roadmap + DIAGNOSTICS

Status: DONE (docs). Wrote the named-policy teacher-representation proposal into the two owning docs
(was only in chat + code columns before).
- CURRICULUM_GAME_MASTER_APPROACH.md Stage 4b Track A amendment: A0 decision table gains step "2.
  Representation (H4)" (named_capture_frac via n_samples=0 search; explicit warning not to read it off
  the fixed-policy columns); Branches now read representation-first; new "Stage A1b (categorical/blend
  orderer head)" describes the fix (per-stream classify {even_split, proportional, don't-care} with
  soft-labelled near-ties + explicit don't-care class, OR a per-stream blend coefficient alpha;
  decode by executing the named policy); Accept/Reject updated to cover A1b + named_capture~0.
- docs/DIAGNOSTICS.md entry 11: added the H4 representation discriminator to the low-VE branch
  (named_capture~1 => output-space fix / named teacher; ~0 => continuous denoise), with the
  fixed-policy read-off warning.
Artifacts: CURRICULUM_GAME_MASTER_APPROACH.md, docs/DIAGNOSTICS.md. No code, no runs.
Blockers: none. Next: unchanged — A0 pod run (ua7llllthh9kol) still in flight; on pull, read
headroom -> named_capture_frac -> ve jump and pick A1b / A1 / A2 / done. Update entry 11 "Seen in"
with the named_capture numbers once they land.

---

## 2026-07-05 — Stage A0 orderer-harvest gate: COMPLETE (pulled). Verdict = named/blend teacher (A1b)

Status: DONE. Pod ua7llllthh9kol run finished + pulled to
experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0/{a0,debug} (flattened a double-nested
rsync). Pod terminated, tokenomics/_pod_ckpts/ deleted.

Authoritative summary.csv (frozen champion producer, 3 mined tight probe tasks, eval_seeds=3, K=4):
  task  champ  search  even_split  proportional  search_named  named_capture  ve_single->averaged
  s29   952.7  1070.7  809         837.7         1070.7         1.00           0.27 -> 0.52
  s77   0      997.3   513.7       681.3         847.7          0.69           0.26 -> 0.58
  s45   0      981.7   199.7       367           716.7          0.66           0.64 -> 0.75
  means: teacher_headroom=699, named_capture=0.784, ve_single=0.389, ve_averaged=0.618

Reads (headroom -> representation -> denoise):
1. Headroom: LARGE on all three. On s77 and s45 the deployed CHAMPION orderer delivers 0 units
   (< even_split's 514 / 200) — the memorized champion collapses on the mined capacity-binding
   regime (operating-point failure, [[generalization-overfit]]). ~1000u to harvest on 2/3 tasks.
2. Representation (H4, user's proposal): named_capture 1.00 / 0.69 / 0.66 (mean 0.78). A per-cycle
   NAMED-only search (even_split/proportional switching, no Gaussian samples) captures 66-100% of
   the search headroom. The single-choice noise is largely an OUTPUT-SPACE problem, not label noise.
3. Denoise (H1): averaging lifts VE on all three, but averaged VE tops out ~0.5-0.75 — the raw
   continuous target stays only partly learnable.

Decision: primary harvest = Stage A1b, the BLEND orderer head (per-stream alpha between even_split
and proportional). Rationale: captures the per-cycle named switching that IS 66-100% of the lever,
with a state-reproducible target (kills the RNG tie-noise that made the continuous vector
unlearnable), and the continuous alpha also grabs part of s77/s45's ~1/3 continuous residual. A1
continuous-denoise = fallback for the residual only. Near-free sanity win to bank first: even_split
already beats the champion by 500-1000u on s77/s45 — the champion is worse than a naive baseline on
the mined tasks.
Artifacts: orderer_a0/ (summary.csv, RESULT.md, spec.json, debug/*/samples_{single,averaged}.pkl.gz);
docs/DIAGNOSTICS.md entry 11 "Seen in" updated with A0 numbers.
Blockers: none.
Next: (a) index EXP-0036 orderer_a0 result + fill RESULT.md Conclusions (human read first); (b) spec
the blend orderer head (Stage A1b) — new mini-arc, TDD the head + advantage-weighted fit, gate
deployed units >= champion on the held-out mined suite; (c) flag separately that the champion orderer
collapses (0u) on mined tight tasks — the harvest bar is ~even_split, not the champion.

## 2026-07-05 — Stage A0b built + launched: the two pre-A1b checks (blend value + choice learnability)
Status: A0b diagnostic built via TDD (5 new tests, file 35, full suite 623 passed) and launched on a
fresh pod. Motivation: A0's named_capture_frac only proved the headroom is REACHABLE inside a named
vocabulary via the lookahead oracle — it did NOT prove (a) the alpha-blend family adds anything over
hard even_split/proportional selection (the search never blended), nor (b) the committed choice is
PREDICTABLE from the stream observation (oracle uses 6-cycle rollouts the student won't have). A1b
rests on both; A0b tests both before spending the training arc.
- `eval/baselines.py::ordering_blend(alpha)` — logit-space blend `a*even_split+(1-a)*proportional`;
  even_split = zero logits, so the family is a temperature on proportional logits (shares ~
  throughput^(1-a)) — exactly what a per-stream blend head can deploy (vocab == deployable head).
- `eval/orderer_oracle.py::blend_vocabulary(alphas)` — named+blend candidate vocabulary, insertion
  order = categorical label order; `_named_search_units` gains `alphas` (byte-identical when empty).
- `train/orderer_scorer.py::collect_choice_samples` (per-stream view + committed vocab choice +
  decisiveness margin, ties kept + split in metrics), `OrdererChoiceModel` (same encoders, mean-
  pooled rows -> vocab logits), `fit_choice_model` (CE), `choice_metrics` (held-out top-1 vs
  majority floor, overall + decisive-only).
- `eval/orderer_representability.py::a0b_diag_row` / `run_a0b_diag` / `_emit_a0b`; CLI `--a0b
  --alphas` (needs no orderer ckpt — no champion involved). D2 = blend_capture_frac vs
  named_capture_frac; D1 = held-out choice_top1 vs choice_majority (train seeds[:-1], hold out last).
Artifacts: staged _pod_ckpts (producer_champion_exp0035.pt = EXP-0035 replay candidate;
task_archive.jsonl = EXP-0036 mine); pod mwiodimxu9oo2d (fresh 16 vCPU cpu5c), `--a0b --iters 1000
--horizon 100 --eval-seeds 3 --alphas 0.75,0.5,0.25` on the 3 probe tasks (s29/s77/s45); pulls to
experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0b. Launch log /tmp/a0b_pod_launch2.log.
Decisions: blend in LOGIT space (not probability space) so search vocabulary == what the head emits;
choice samples keep tie rows (margin=0) and metrics split decisive vs all — accuracy on ties is
noise, not learnability evidence.
Blockers: none.
Next: read blend value first (blend_capture >> named_capture => keep alpha head; ~equal => plain
categorical), then learnability (choice_top1 ~ majority on held-out => A1b inherits the wall, fall
back to A1/A2; well above => A1b green-lit). Then delete tokenomics/_pod_ckpts/.

## 2026-07-05 — Stage A0b RESULT: blend vocabulary confirmed, choice NOT learnable from stream view — A1b not green-lit
Status: A0b pod run complete (pod mwiodimxu9oo2d, stopped+terminated; pulled to
experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0b/). Both questions answered:
- **D2 blend value: YES, keep the alpha-blend vocabulary.** Mean capture 0.784 (named) -> 0.901
  (blend). On the tasks with a residual: s77 0.69->0.95, s45 0.66->0.78; s29 (named already 1.0)
  blend 0.97 = trajectory noise. Caveat: s45 leaves ~21% of headroom outside the blend family.
- **D1 choice learnability: FAIL 3/3.** Held-out top-1 BELOW the majority floor on every task
  (0.27/0.53 s29, 0.38/0.56 s77, 0.34/0.47 s45); decisive-only split no better. Train CE ends
  0.157/0.171/0.097 -> fits training episodes near-perfectly, anti-transfers to the held-out seed.
  Wall = transfer, not fit.
Decisions: Stage A1b as a pure per-stream obs->choice head is NOT green-lit (would imitate an
unpredictable target). Representation is right; the mapping from LOCAL obs is not learnable.
Two competing causes (root-cause rule, discriminate before next arc): (a) missing global context
(H2) vs (b) choice noise (lookahead margins within rollout noise). Discriminating check = fit the
SAME classifier on the FULL GLOBAL STATE on the same samples: beats majority -> (a) enrich obs;
still fails -> (b) soft labels / deploy-time light search (A2), enrichment useless.
Artifacts: orderer_a0b/{RESULT.md (Conclusions filled), summary.csv, spec.json}; DIAGNOSTICS.md
entry 11 amended with the A0b gate. _pod_ckpts/ deleted.
Blockers: none.
Next: global-state probe (cheap; reuse collect_choice_samples with a global-state encoder) to pick
branch (a) vs (b). Deferred: index EXP-0036 orderer_a0 + orderer_a0b in experiments/README.md;
champion-collapses-on-mined-tasks (0 units on s77/s45) still flagged separately.

## 2026-07-05 — Stage A0c built + launched: global-state probe (missing-context vs choice-noise discriminator)
Status: A0c built via TDD (4 cycles, +2 tests -> file 37, full suite 625 passed) and launched on a
fresh pod. Motivation: A0b's D1 failed 3/3 (held-out choice_top1 < majority floor while train CE
fits to 0.10-0.17) — two competing causes demand different fixes: (a) missing global context
(enrich obs) vs (b) rollout-noise choice (soft labels / deploy-time search). A0c discriminates:
fit the SAME classification protocol from the FULL global state (env.global_state(), ADR-0004).
- `train/orderer_scorer.py::GlobalChoiceSample` — one per searched cycle (only cycles that yielded
  stream samples, so both probes share the cycle population); `collect_choice_samples` now returns
  `(samples, global_samples, names)` from the SAME search pass (no extra search cost).
- `train/orderer_scorer.py::GlobalChoiceModel(n_choices, state_dim)` — plain MLP + set_norm
  (train-set mean/std standardization; EXP-0009 lesson: raw global_state magnitudes wreck
  optimization). `fit_choice_model`/`choice_metrics` reused via duck-typed `.view/.choice/.margin`.
- `eval/orderer_representability.py::a0b_diag_row` gains gchoice_top1/gchoice_majority (+decisive
  split, counts); `_emit_a0b` read order now has step 3 = global probe branch logic.
Artifacts: restaged _pod_ckpts (same producer champion + archive); fresh create-pod launch,
same --a0b CLI, --out-dir /workspace/tokenomics/orderer_a0c; pulls to
experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0c. Launch log /tmp/a0c_pod_launch.log.
Decisions: global sample label/margin identical to the cycle's stream samples (same committed
choice); probe n is small (~1 per cycle, ~200 train / ~100 heldout per task) — read it as a coarse
beats-majority-or-not gate, not a precise accuracy estimate.
Blockers: none.
Next: read gchoice_top1 vs gchoice_majority per task. Beats floor => D1 wall = missing context ->
spec obs enrichment (compact global features). Fails => choice is rollout noise -> A1 soft-label /
margin-filter or A2 deploy-time light search. Then delete tokenomics/_pod_ckpts/.

## 2026-07-05 — Stage A0c RESULT: global state fails too — obs enrichment is dead, branch = target/deployment
Status: A0c pod run complete (fresh pod 71jew6yf5pr0vb, stopped+terminated; pulled to
experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0c/). The discriminator answered:
- Held-out gchoice_top1 vs majority floor: s29 0.31/0.53, s77 0.29/0.54 (both clear FAIL),
  s45 0.40/0.42 (tie within n=100 noise). Nowhere does the FULL global state beat its floor.
- D2/D1 reproduce A0b exactly (deterministic domain; search_cu identical).
Decisions: the D1 wall is NOT missing context — no orderer obs enrichment can fix it (even the
ADR-0004 centralized-critic input fails). The committed choice is a function of state in
principle (n_samples=0 search is deterministic) but hinges on 6-cycle-lookahead consequences
that don't compress into a learnable pattern at this sample size. Branch (b):
- A2 deploy-time light search = primary lever (5-candidate vocabulary search realizes 0.90 mean
  blend capture by construction; EXP-0009 showed short rollouts suffice).
- A1 soft/advantage labels = secondary; gate it on a critic's held-out candidate-RANKING check
  first (decisive-cycle fails suggest ranking from state is hard too).
Artifacts: orderer_a0c/{RESULT.md Conclusions filled, summary.csv, spec.json}; DIAGNOSTICS.md
entry 11 A0c gate appended. _pod_ckpts/ deleted.
Blockers: none.
Next (user decision): spec Stage A2 (deploy-time vocabulary search as the orderer, cost profile +
acceptance vs champion on mined tight tasks) vs A1 critic-ranking gate. Deferred: index EXP-0036
orderer_a0/a0b/a0c in experiments/README.md; champion-collapses-on-mined-tasks flag still open.

## 2026-07-05 — A0c verdict CORRECTED on user review: "obs enrichment dead" overclaimed; A0d complete-state probe is the discriminator
Status: user challenged the A0c conclusion on two fronts: (1) no trace-level teacher-vs-probe
comparison was ever done (only aggregate top-1 vs majority; per-sample margins not persisted);
(2) "even if the orderer sees the whole world, can it extract the signal? maybe the gap is
aggregated/historical observations, not emulation". Code check confirmed the challenge:
- `encode_global` (env/encode.py:455, ADR-0004) is NOT the full state — per-producer capacity is
  SUMMED ACROSS TYPES (bottlenecks/death on primal-tight tasks are per-type), no order-queue/TTL
  state, inventory as unit count not composition, no history. The "FULL global state" claim in
  the A0c write-up was false.
- Sample starvation confound: probe trained on ~200 cycles/task — failing a ~50% majority floor
  from 200 samples of a high-dim 5-way problem is weak evidence of impossibility.
- Domain+search deterministic => the committed choice IS an exact function of the FULL env
  state; the target function provably exists. A0c only killed enrichment with the EXISTING
  global features.
Decisions: withdrew "branch (b) picked / obs enrichment DEAD / A2 primary". Three causes now
standing (root-cause rule): lossy snapshot | missing temporal/aggregated features (history
recovers hidden state given a lossy snapshot; trend features = backward proxies for what the
lookahead unrolls) | genuine incompressibility. Discriminator = A0d complete-state probe: same
protocol, LOSSLESS env-state serialization, ± k-cycle history stack, ~10x mined cycles. Fails =>
incompressibility diagnosed with evidence, A2/A1 branch confirmed. Passes => enrichment alive,
find compact features (temporal aggregates first).
Artifacts: orderer_a0c/RESULT.md Conclusions rewritten (correction dated, original verdict kept
struck in prose); orderer_a0b/RESULT.md discriminating-check para post-hoc note;
eval/orderer_representability.py emitter template + A0c code comment fixed (future runs won't
emit the overclaim); DIAGNOSTICS.md entry 11 A0c gate rewritten; experiments/README.md EXP-0036
row updated with the full orderer probe ladder (A0/A0b/A0c + correction) — clears the deferred
index item.
Blockers: none.
Next: spec + TDD A0d complete-state probe (lossless serialization, history stack, 10x cycles) ->
pod. Trace-level teacher-vs-probe read + margin persistence fold into A0d instrumentation.
Deferred: champion-collapses-on-mined-tasks flag still open.

## 2026-07-13 — Stage A0t built + launched: trace-regret analysis (which decisions go wrong vs the teacher, and what they cost)
Status: user direction: before adding observation metrics blindly, analyze TRACES — find the
decisions the student gets wrong vs the search teacher, read what the teacher saw, and infer the
missing features from evidence. Built TDD (4 red-green cycles, suite 629 passed; +4 tests):
- `train/orderer_scorer.py::CycleTrace` — per searched cycle: the FULL candidate score table the
  argmax threw away, decisiveness margin, `champion_units` (champion orderer scored at the SAME
  state by the same lookahead-6 rollout), and `_cycle_state_snapshot(env)` — JSON-readable
  decision-time state with the per-type detail the encoders roll up (per-type capacities/reserved/
  requested, inventory + inbound by product, investment flags, live order book with ages, open
  demand). `collect_choice_samples` now returns `(samples, gsamples, traces, names)` + `champion=`
  kwarg; determinism test: champion==proportional reproduces the proportional table column exactly.
- `eval/orderer_representability.py::a0t_diag_row/run_a0t_diag/_emit_a0t`; CLI `--a0t` (needs
  --orderer-ckpt). Per held-out cycle: probe_regret = units[teacher argmax] − units[global-probe
  pick] (≥0 by construction); champ_regret = units[teacher argmax] − champion rollout (<0 =
  champion beats the whole vocabulary there); local_headroom = units[teacher] − units[even_split].
  regret_frac = Σregret/Σheadroom — cost-weighted misses, ties stop polluting the read. RESULT.md
  dumps top-5 regret cycles per task WITH candidate tables + state snapshots; full traces →
  artifacts/traces.json.gz.
Artifacts: local tiny smoke passed (1 task, horizon 12, iters 3 — CLI+layout verified; smoke rows
already show real cost shape: cycle-2 even_split=0 vs blend_0.5=21, champion regret 9). Staged
_pod_ckpts (producer_champion_exp0035.pt, orderer_scorer.pt = EXP-0012, task_archive.jsonl);
fresh cpu5c pod via --create-pod; `--a0t --iters 1000 --horizon 100 --eval-seeds 3 --alphas
0.75,0.5,0.25` on s29/s77/s45; pulls to experiments/EXP-0036-.../orderer_a0t. Launch log
/tmp/a0t_pod_launch.log. NOT committed (user hold: fix observation + rerun before committing).
Decisions: per-cycle units are lookahead-window units (local proxy, not episode units) — RESULT
read order states this; champion scored at TEACHER-visited states (aligned comparison; its own-
trajectory drift is a separate, later read).
Blockers: none.
Next: read probe_regret_frac first (≈0 despite low top-1 => D1 wall costs nothing); then read the
top-regret cycle dumps and NAME the missing feature from what the teacher saw (inbound arrivals /
per-type bottleneck / TTL cliffs); champ_regret_frac + n_champ_better situate the champion and the
vocabulary ceiling. Then decide: obs enrichment (A0d candidates) vs deploy-time search (A2).

## 2026-07-13 — Stage A0t RESULT: misses are expensive, no system-level feature separates the classes, champion beats the vocab at 40% of states
Status: A0t pod run complete (pod b6y3ke45teehw1 created+terminated; pulled to
experiments/EXP-0036-track-a-round2-primal-tight/orderer_a0t/, Conclusions filled). Reads:
- probe_regret_frac 0.98/0.86/0.56 (s29/s77/s45) — the D1 wall costs real units, not tie noise.
  Dominant miss: picking even_split when it's not in the tied-winner set (154–166u per task).
  margin==0 cycles can carry 52u regret → "decisive=margin>0" was the wrong split; tie cycles
  have a winner SET target.
- champ_regret_frac 0.52/0.20/0.17 at teacher states; n_champ_better 40/47/41 of 100 — champion
  per-decision is decent where the teacher goes and BEATS the whole 5-entry vocabulary at ~40% of
  states → its 0-unit episodes are trajectory-level compounding, and the vocabulary is not a
  behavior cover.
- Feature inference from LOSSLESS snapshots: system-level aggregates (min per-type ratio, cap
  imbalance, inbound, inventory, order count/age, open demand, phase) are FLAT between even-wins
  and concentration-wins cycles — the asked-for "infer the metric from wrong decisions" search
  found none at this granularity.
- Temporal: choice persistence 59–63%, sticky repeat-last-pick still loses 24–50% of edge, edge
  lag-1 autocorr 0.35–0.44 with 13–23 large sign flips per 100 cycles → history features capture
  only the slow component.
Decisions: evidence weakens both enrichment sub-hypotheses (lossy-snapshot AND temporal) and
strengthens A2 deploy-time light search with candidate set = {blend vocabulary + CHAMPION} (union
dominates both; covers the champion-better 40%). A0d complete-state probe now low-prior (would
need relational features, not more inputs). Analysis scripts were ad-hoc (in-chat python over
traces.json.gz); traces persisted for re-analysis.
Blockers: none. Working tree intentionally UNCOMMITTED (user hold from 2026-07-05 continues).
Next (user decision): spec Stage A2 with the champion-augmented candidate set (cost profile +
full-episode acceptance vs champion on mined tight tasks + broad-family no-regression) — or a
relational-feature probe first if the enrichment path should be exhausted.

## 2026-07-14 — Stage A0f (HANDOVER Task 1) DONE: missing-observation probe — no TESTED block family helps; per-cycle choice reads as seed-idiosyncratic label noise
Status: executed HANDOVER.md Task 1 (+ the flat-relational slice of Task 2 and the soft-target
slice of Task 4 in the same evidence set). Built TDD (6 new tests, suite 638 passed):
- `train/orderer_scorer.py`: `cycle_feature_blocks(env)` (named decision-time blocks: gstate /
  pooled stream / per-type capacity / order-book+TTL / inventory / inbound / relational recipe
  ratios / compact), `history_features` (k-cycle stack, seed-boundary safe),
  `ValueTableModel`+`fit_value_table`+`value_table_metrics` (predict the FULL candidate table,
  regret-first metrics — HANDOVER: accuracy is the wrong primary metric),
  `collect_choice_samples(..., with_features=True)`, `CycleTrace.features`.
- `eval/orderer_representability.py`: `a0f_feature_row`/`run_a0f_probe`/`_emit_a0f`, CLI `--a0f
  --k-history`; local tiny smoke, then full run on fresh cpu5c pod (3f6606gset773e,
  created+terminated), pulled to experiments/EXP-0036-.../orderer_a0f/ (traces+features persisted
  to artifacts/feature_traces.json.gz; local re-fit script analysis_followup.py + .log).
Reads (RESULT.md Conclusions filled; wording tightened on user review 2026-07-14 — original
"no missing observation exists / train 0.0 everywhere / constants beat every probe" overclaimed):
- NO tested block set gives a STABLE held-out regret reduction: per task 0.90–1.14 (s29),
  0.76–1.26 (s77), 0.365–0.89 (s45 — best = its two SMALLEST sets, stream 0.365/gstate 0.407;
  every added block hurts s45: small-sample overfitting signature). train_regret_frac 0.0 for
  every set with dim >= 109; only the 81-dim stream block underfits (0.53/0.49/0.81) →
  memorizes train, transfers nothing.
- Constant mid-blend picks match or beat the fitted probes ALMOST everywhere (always-blend_0.5 →
  0.90/0.97/0.29 s29/s77/s45); sole exception s77 +order_queue 0.7555 vs constant blend_0.75
  0.7821 — a 0.027 one-task/one-held-out-seed edge, not a stable reduction.
  train-majority=even_split → 1.0.
- Learning curve wrong-signed: 1→2 train seeds RAISES held-out regret 3/3 (0.88→0.97, 1.06→1.13,
  0.62→0.80) → anti-transfer, not sample starvation.
- Task 4 soft targets (winner-set BCE, margin-weighted CE) don't rescue (best 0.65 vs 0.29
  constant floor).
Decisions (root-cause rule satisfied, claim SCOPED): for the TESTED families / probe models /
data scale the A0c discriminator lands on (3): the per-cycle lookahead-6 winner on a NEW demand
seed behaves as noise around a stable "mid-blend > even_split" mean preference. Closes HANDOVER
Task 1 (the tested ladder), NOT all possible observations/models (e.g. a graph/relational
encoder is untested — but it has no positive signal here to build on). A0d complete-state probe
superseded (A0f is that check, stronger: value targets + block ladder + constant floors +
learning curve). HANDOVER Task 3 (learner-trajectory DAgger) moot for this wall — train and
held-out are both teacher trajectories, only the demand seed differs; deployment always faces
new seeds. DIAGNOSTICS entry 11 extended (A0t + A0f gates); experiments/README.md EXP-0036 row
updated.
Artifacts: orderer_a0f/{RESULT.md, summary.csv, spec.json, analysis_followup.py+.log,
artifacts/feature_traces.json.gz}; tests in tests/test_orderer_scorer.py. Working tree still
UNCOMMITTED (user hold from 2026-07-05 continues). _pod_ckpts/ deleted.
Blockers: none.
Next (user decision): spec Stage A2 deploy-time light search — candidate set = blend vocabulary
+ champion (covers the champion-better 40% of states, A0t) — cost profile + full-episode
acceptance vs champion on mined tight tasks + broad-family no-regression; include FIXED
blend_0.5/blend_0.75 policies in the comparison as the zero-cost floor A0f measured.

## 2026-07-14 — Stage A0g: producer-count-agnostic value table (HANDOVER "Proposed Model") implemented; set bias does NOT change the A0f verdict
Status: DONE (user-directed: implement the HANDOVER Proposed Model).
Artifacts:
- `train/orderer_scorer.py`: `ProducerSetValueModel` (shared per-producer MLP → masked
  mean+max pool → head on [context ‖ pooled] → fixed candidate table; HANDOVER invariants:
  producer-permutation invariant, count never changes model shape, mask-0 rows exactly
  absent, output fixed by vocabulary), `fit_set_value_table` (same per-row-centered
  value-table MSE as the flat probe — deltas read as inductive bias, not objective),
  `set_value_table_metrics` (+ shared `_table_pick_metrics` refactor),
  `set_features_from_blocks` (lossless reshape of the A0f blocks: producer row = per-type
  capacity ‖ inventory ‖ inbound ‖ relational column, 108 dims; context = stream ‖
  order_queue ‖ compact, 230 dims), `cycle_set_features` (live path, unit-tested identical
  to trace reconstruction). 3 new tests; module 53 passed, full suite green.
- `experiments/EXP-0036-.../orderer_a0g/`: probe_set_value.py + .log, summary.csv,
  spec.json, RESULT.md. Local run = saved-artifact recompute over
  orderer_a0f/artifacts/feature_traces.json.gz (HANDOVER-allowed local; seconds-scale fits).
Reads (same 3 tasks, same traces, same target/iters/init as A0f):
- Train regret_frac ~0.0 every variant; held-out at/near the constant mid-blend floors.
  s45 decisive: EVERY set variant 0.51–0.75 vs constant blend_0.5 0.2926 (and flat stream
  0.365) — more structure hurt where less already hurt.
- Only sub-floor readings are ~0.03 edges in DIFFERENT variants on different tasks (s29
  set+history 0.8652 vs floor 0.8965; s77 set+history 1-seed 0.7476 vs 0.7821) — same
  noise magnitude as A0f's s77 exception, not a stable reduction.
- Learning curve still wrong-signed: 1-seed beats 2-seed held-out on s77 (0.7476 vs
  0.9922) and s45 (0.6008 vs 0.6429).
- Cross-task pooled fit (600 rows; possible ONLY in the set view — identical row/context
  dims across tasks, P=6 vs 9 handled by mask) does not rescue: 0.9141/1.0815/0.5147
  (s77 WORSE than even_split).
Decisions (scoped): the HANDOVER "Proposed Model" as specified (mean+max pool,
centered-table MSE, features reconstructible from saved traces) is implemented, invariant-
tested, and does not beat the constant mid-blend floor on these traces — consistent with
the A0f diagnosis (per-cycle lookahead winner on a new demand seed ≈ seed-idiosyncratic
noise around a stable mid-blend preference). Untested: attention pooling, ranking/winner-set
losses on the set encoder, per-producer routed/completed/rejected history (not in saved
traces). The model stays as the portable distillation head if a future arc (e.g. A2
deploy-time search traces) finds learnable signal. DIAGNOSTICS entry 11 A0g gate added;
experiments/README.md EXP-0036 row extended.
Blockers: none. Working tree still UNCOMMITTED (user hold from 2026-07-05 continues).
Next (user decision, unchanged): spec Stage A2 deploy-time light search (blend vocabulary
+ champion candidates, fixed mid-blends as the zero-cost acceptance floor).

---

## 2026-07-15 — Stage A0w: window/episode-level blend-policy distillation (HANDOVER Task 5) — driver implemented + TDD + local smoke; full run is a pod job

Status: DONE (code + tests + local smoke). The real multi-seed / multi-task / full-horizon
run is a CLOUD job (CLAUDE.md HARD RULE) and has NOT been run — the driver is built, TDD'd,
and smoke-verified locally, and the pod command is staged below.

What Task 5 changes vs A0f/A0g: A0f/A0g predicted a PER-CYCLE candidate value table whose
winner ordering does not transfer across demand seeds. A0w changes the target TIME SCALE —
the value of each CONSTANT blend policy over an episode / rolling window (5/10/20 cycles),
one row per window instead of per searched cycle. Deployment picks ONE constant blend via a
conservative switch rule (default blend, switch only when the predicted advantage clears a
threshold), and is scored by DEPLOYED ROLLOUT against the best constant blend — not by table
error alone.

Artifacts (code):
- `tokenomics/train/orderer_scorer.py`: `aggregate_set_window` (window feature summary:
  context = concat[mean, last]; producer rows = mean-over-window — the same summary the
  deployed policy builds from its rolling buffer), `blend_decision` (conservative default-
  sticky switch rule, threshold in the model's centered/std-scaled output units),
  `set_table_ordering_policy` (deploy the window head as a meta-orderer: per-cycle rolling
  window → predict table → pick one constant blend → act). Reuses the A0g
  `ProducerSetValueModel` / `fit_set_value_table` / `set_value_table_metrics` unchanged.
- `tokenomics/eval/orderer_representability.py`: `--a0w` mode — `_fixed_blend_episode`
  (full-episode fixed-policy per-cycle unit deltas + reference-trajectory set features),
  `episode_value_table` / `window_value_rows` (constant-policy targets; dead policies padded
  to the reference length with 0), `_const_regret_frac` / `_decision_regret_frac` (best-
  constant-blend baseline + deploy-rule regret), `a0w_task5_row` / `run_a0w_task5` /
  `_emit_a0w` (per-granularity fit + deployed rollout + closure vs search; EXP layout).
- Tests: `tests/test_a0w_task5.py` (4: episode/window target construction, dead-policy
  padding, const & decision regret math) + 2 in `tests/test_orderer_scorer.py`
  (`aggregate_set_window`, `blend_decision`). All pass.
- Local smoke (1 task s29, eval-seeds 2, horizon 12, window 5, iters 20) emits the full EXP
  layout to /tmp/a0w_smoke (episode + window5 + deploy rows, all metrics populated; deploy
  showed default blend_0.5 = 26u vs the toy-trained meta-orderer switching to even_split = 0u,
  champ 2u, search 69u) — pipeline verified end-to-end, numbers are smoke-garbage by design.

Reads: none yet — no real result. The definition of done (does the learned window/episode
model beat the best CONSTANT blend by more than seed noise, on table regret AND deployed
units) is answered only by the pod run.

Decisions: build/TDD/smoke local per the HANDOVER local-allowance; the decision-gating run
goes to Runpod. No DIAGNOSTICS entry or RESULT conclusion yet — RESULT.md ships with a
human-TODO conclusion, filled after the pod run produces a verdict.

Blockers: none. Working tree still UNCOMMITTED (user hold from 2026-07-05 continues).

Next: launch the full A0w pod run (stage producer champion + orderer champion ckpts under
`tokenomics/_pod_ckpts/`, wrap with run_pod_training.py, `--pull` the EXP dir back to
`experiments/EXP-0036-.../orderer_a0w/`), over the three A0t tasks first (s29/s77/s45 — the
same decisive trio A0f/A0g used) at `--iters 1000 --horizon 100 --eval-seeds 3 --windows
5,10,20`; then fill RESULT.md's Conclusions and (if it settles the verdict) a DIAGNOSTICS
entry. Owner decision whether to also run the full 81-row archive.

---

## 2026-07-15 — Stage A0w RESULT: trained-orderer path STOPPED — learned window/episode meta-orderer loses to a fixed blend_0.5 on all 3 tasks

Status: DONE. Narrow 3-task A0w pod run complete (fresh 16-vCPU CPU pod xt414xetp10l4f,
runpod/base:0.7.0-ubuntu2004, --cpu python3.12; created → ran → pulled → terminated).
`--a0w` over s29/s77/s45 at --iters 1000 --horizon 100 --eval-seeds 3 --windows 5,10,20
--deploy-window 10 --threshold 0.25. Artifacts: `experiments/EXP-0036-.../orderer_a0w/`
(RESULT.md, summary.csv, spec.json).

Verdict (owner's rule = deployed rollout, NOT table regret): learned_cu vs default(blend_0.5)
vs best_const —
- s29: 810 vs 960 vs 768(blend_0.25) — loses to default.
- s77: 666 vs 749 vs 893(blend_0.25) — loses to BOTH.
- s45: 588 vs 621 vs 621(blend_0.5) — loses to BOTH.
Learned < default 3/3, < best_const 2/3. Rule (stop if loses to default/best-constant on
2 of 3) → **STOP the trained-orderer path.**

Reads:
- The default fixed blend_0.5 is strong and is the deployable policy: 960/749/621 units,
  near champion on s29 (1004) and FAR above the *dead* memorized champion on tight s77/s45
  (champ = 0, system death — the A0 collapse). No learned switcher needed.
- Switching destroys value: deployed switch rate 0.40–0.88; s29 drops 960→810 purely from
  switching. threshold 0.25 too eager, but direction is wrong not just magnitude (best-const
  beats learned on s77, 893 vs 666).
- Table regret does NOT predict deploy value (the trap the rule warns of): longer windows
  lower held-out regret_frac (s77 0.54→0.30; s77 episode 0.0) yet deploy still loses; where
  table looked best (s77) the best CONSTANT blend's regret was lower still (0.34/0.20/0.10 <
  learned 0.54/0.46/0.30).
- closure_vs_search 0.65/0.60 on s77/s45 is a red herring — recovery over a DEAD champion;
  honest baselines beat learned outright.

Decisions: A0w closes the last untested lever (target TIME SCALE) from the A0f/A0g arc. Across
the whole ladder (A0b/A0c/A0f/A0g/A0w) the deployable learned orderer never beats a fixed
mid-blend on held-out demand → the per-cycle/window search winner is seed-idiosyncratic noise
around a stable mid-blend preference; no stable supervised target for a learned orderer here.
Ship/compare against fixed blend_0.5 (or per-task best-constant). DIAGNOSTICS entry 11 A0w gate
added; experiments/README.md EXP-0036 row extended.

Blockers: none. Working tree still UNCOMMITTED (user hold from 2026-07-05 continues).

Next (owner decision): NOT the full 81-row archive on this path, and NOT a threshold sweep
(only warranted if A0w looked promising — it didn't). The remaining EXP-0036 headroom, if
pursued, is spec Stage **A2 deploy-time light search** over the blend vocabulary (a COMPUTE
lever, not a learned policy), with the fixed mid-blend as the zero-cost acceptance floor A0w
just established. Owner call whether to open A2 or close EXP-0036's orderer arc here.

---

## 2026-07-15 — EXP-0037 opened: value-routed orderer branch

Status: SCAFFOLD ONLY. EXP-0036 closure committed as `65898d9` (`Close EXP-0036 orderer
probe arc`). New experiment directory:
`experiments/EXP-0037-value-routed-orderer/`.

Question: did EXP-0036 solve the wrong trained-orderer problem by selecting among
predefined capacity-shaped policies? New branch tests direct value-based producer routing:

```text
(order, eligible producer, context) -> marginal value score
softmax over eligible producers -> routing weights
```

Proposed safe policy shape:

```text
base_logits = blend_0.5
final_logits = base_logits + clipped learned residual logits
```

The acceptance floor is fixed `blend_0.5`, not the old champion and not
`closure_vs_search`. First gate remains the decisive EXP-0036 trio s29/s77/s45. Stop if the
deployed residual router does not beat fixed `blend_0.5` on at least 2 of 3 tasks.

Next implementation task: build an offline residual-router probe that perturbs routing
weights around `blend_0.5` and learns per-order/per-producer value scores, instead of
training another selector over `even_split`/blend/proportional policies.

---

## 2026-07-16 — EXP-0037 residual-router probe INFRA built (TDD)

Status: probe infrastructure implemented, unit-tested, tiny local smoke green. NO real gate
run yet (that is a pod job). Built the value-routed orderer per HANDOVER "Recommended First
Task": a residual score over fixed `blend_0.5`, trained on marginal-value-of-logit-bump
labels, judged by deployed rollout — differing from EXP-0012 (throughput teacher) and
EXP-0032 (noisy per-cycle search vector) in all three required ways.

Artifacts:
- `tokenomics/tokenomics/eval/value_router.py` (torch-free): `residual_router`,
  `marginal_value_labels` (teacher), `evaluate_router_gate` (DoD instrument, deployed rollout
  vs blend_0.5 at anchored seeds).
- `tokenomics/tokenomics/train/value_router.py` (torch): `producer_row_features` (white-box
  value features), `ResidualScoreModel` (producer-count-agnostic shared row scorer),
  `fit_residual_scorer` (masked MSE on centered marginal labels), `to_score_fn` deploy
  adapter, `collect_router_samples`, `run_value_router` driver + CLI.
- `tests/test_value_router.py` (5) + `tests/test_value_router_train.py` (3).
- EXP-0037 `spec.json` (driver + pod reproduce cmd) and `RESULT.md` (Status/What-was-built)
  updated.

Decisions:
- Label objective = marginal value of a `+delta` logit bump (owner pick over pairwise / PG).
- Session scope = probe infra + tiny smoke; the real s29/s77/s45 gate runs on Runpod.
- `residual_router` zero-score == blend_0.5 byte-identical is the safety floor (pinned).

Tests: full suite **655 passed** (+8). Local CLI smoke on real s29 archive task ran end to
end (190 labeled decisions, tiny knobs) — pipeline only, not model evidence.

Blockers: none. Worktree still UNCOMMITTED (user hold continues).

Next: launch the real 3-task gate on a pod (`spec.json` reproduce: `--delta 1.0 --lookahead 8
--eval-seeds 3 --horizon 100 --epochs 300 --clip 2.0`), stage `task_archive.jsonl` +
`producer_champion_exp0035.pt` under `tokenomics/_pod_ckpts/`, pass full s29/s77/s45 task ids,
pull into `experiments/EXP-0037-.../gate`. Stop rule: beat fixed `blend_0.5` on >=2 of 3 by
deployed units, else close the branch.

---

## 2026-07-16 — EXP-0037 review: three P1 correctness fixes (pre-pod)

Status: review of the probe infra found three P1s that would have made any gate result
untrustworthy; all fixed test-first, full suite green.

1. **Confounded labels** — `marginal_value_labels`/`_one_cycle_bump` gated the `+delta` bump on
   `env.current_cycle` only, so in the joint-stack env (many product-stream ordering agents per
   cycle) the nudge leaked into every other agent sharing that stream/producer index. Now gated
   on the labeled `agent` too (regression `test_bump_perturbs_only_the_labeled_agent`).
2. **Wrong env contract** — gate + collector built plain `TokenomicsAECEnv`; EXP-0036 measured
   its `blend_0.5` floor in `joint_stack_env` (entity producer obs + corrected v4 product-stream
   ordering). Both now use `joint_stack_env` + `run_episode` → `blend_cu` comparable to the floor.
3. **Wrong producer loader** — `--prod-ckpt` used `_load_prod_closure` (legacy GaussianActor); the
   EXP-0035 champion is a producer *scorer* ckpt. Now `scorer_producing_policy(
   load_producer_score_model(...))`, matching EXP-0036. Verified against
   `EXP-0035-distill-epochs/replay/artifacts/producer_candidate.pt`.

Artifacts: `eval/value_router.py`, `train/value_router.py`, `tests/test_value_router.py` (+1),
EXP-0037 `RESULT.md` (Review corrections) + `spec.json` reproduce updated to `--prod-ckpt` the
scorer champion.

Tests: full suite **656 passed**. CLI smoke on real s29 with the frozen scorer producer on
joint_stack_env ran end to end (202 labeled decisions) — pipeline only.

Blockers: none. Worktree still UNCOMMITTED. Next unchanged: launch the real s29/s77/s45 gate on
a pod (stage `task_archive.jsonl` + `producer_champion_exp0035.pt` under `_pod_ckpts/`), stop
rule = beat fixed `blend_0.5` on >=2 of 3 by deployed units.

---

## 2026-07-16 — EXP-0037 gate RUN on pod: FAIL, branch CLOSED

Status: full 3-task gate ran on Runpod CPU pod `ymys3x82siomyk` (16 vCPU, base:0.7.0, --cpu
python3.12; created → smoked → ran → pulled → TERMINATED). ~40 min. Verdict **1/3 beats fixed
blend_0.5** → predeclared ≥2/3 stop rule fired → value-routed-orderer branch CLOSED.

Numbers (learned vs blend_0.5, deployed rollout, in-sample s29/s77/s45 × 3 anchored seeds):
- s29: 856.3 vs 891.0 (Δ −34.7, −3.9%) — loses
- s77: 696.0 vs 692.3 (Δ +3.7, +0.5%) — wins (seed noise)
- s45: 175.0 vs 400.0 (Δ −225, −56%), death 0.67 vs 0.33 — loses hard, residual KILLS economy

Root cause (evidence, not hypothesis):
1. Target low-signal — training loss flat/rising (ep0 13.16 → ep299 12.97 centered MSE); the
   shared-row scorer cannot fit the marginal-value labels. Same failure family as EXP-0036's
   noisy per-cycle routing targets, reproduced from the new value-residual action space.
2. Learned residual adds harmful noise not stable improvement — lone win is +0.5% in-sample
   noise; on s45 it drives more system deaths than the floor. blend_0.5 base + ±2.0 clip bounded
   but did not prevent damage.

Did NOT run the 81-row archive (handover Non-Goal: only if 3-task gate passes). blend_0.5 stays
the zero-cost orderer floor; no learned orderer (EXP-0036 blend-selector or EXP-0037
value-residual) beats it on primal-tight worlds.

Two P1 correctness fixes landed pre-run (label agent-isolation, joint_stack env + scorer
producer loader), plus a pod-surfaced batching bug fixed test-first: samples from different-
topology tasks have different producer counts P → `fit_residual_scorer` np.stack failed;
fixed by pad-to-max-P + mask (regression `test_fit_handles_samples_with_different_producer_counts`);
added per-(task,seed) collection progress logging (the run was previously blind).

Artifacts: `experiments/EXP-0037-value-routed-orderer/gate/{summary.csv,spec.json,residual_scorer.pt}`,
RESULT.md Conclusions + spec.json compute filled. Pod terminated, `_pod_ckpts/` cleaned.

Tests: 11 new value_router tests; full suite green (run post-changes).

Blockers: none. Worktree still UNCOMMITTED. Next (owner call): EXP-0037 is closed. Open orderer
question stands — search headroom on primal-tight worlds does not compress into any learned
orderer tried; either accept blend_0.5 as the deployed floor or pursue a deploy-time light
search (EXP-0036 Stage A2, a compute lever not a learned policy).

---

## 2026-07-18 — EXP-0037 P2 auditability review: code fixed, verdict unchanged

Status: a second review (4× P2) said the headline gate numbers are consistent with summary.csv
but the *supporting* evidence wasn't reproducible/auditable from the pulled dir. Addressed all
four; the deployed-rollout verdict (1/3, branch CLOSED) is unchanged — it never depended on the
loss trajectory.

- P2 seed: `run_value_router`/CLI now take `--seed` (default 0), seed torch model init, record it
  in `spec.json` → fit + `learned_cu` + 2-of-3 count reproducible. Test
  `test_run_value_router_is_reproducible_under_a_fixed_seed`.
- P2 loss history: driver writes per-epoch `loss_history.csv` (was: only `final_loss`); spec now
  keeps loss_first/last/min. The cited ep0=13.16/ep299=12.97 are from the ORIGINAL pod stdout
  (pre-fix), flagged as such in RESULT.
- P2 per-seed rows: `router_gate` returns `(aggregate, per_seed)` in one rollout pass; driver
  writes `gate_rows.csv` (task,seed,policy,cu,dead) so the s77 "seed noise" read + s45 death
  counts are checkable from raw rows. `evaluate_router_gate` kept as aggregate-only wrapper.
  Test `test_run_value_router_persists_audit_artifacts`.
- P2 feature scope: RESULT conclusion 2 narrowed — the flat fit is over the producer-only feature
  row (no order-age/TTL/backlog), so it is CONSISTENT WITH but does not PROVE an unlearnable
  marginal-value target (feature aliasing not excluded). Verdict rests on deployed rollout instead.

Note: the existing `gate/` artifacts are from the original pre-fix pod run, so they don't contain
`loss_history.csv`/`gate_rows.csv`; those are produced by re-running the reproduce command. Did
NOT re-run the pod (closed negative branch; a re-run only regenerates auditable supporting
artifacts, doesn't change the verdict) — left as an owner call.

Artifacts: `tokenomics/eval/value_router.py` (router_gate + per-seed), `tokenomics/train/value_router.py`
(--seed, loss_history.csv, gate_rows.csv), 2 new tests in `tests/test_value_router_train.py`,
RESULT.md auditability-corrections section.

Tests: value_router suites green (confirmed). Blockers: none. Worktree still UNCOMMITTED.

---

## 2026-07-19 — EXP-0037 finalized; EXP-0038 opened for actor/teacher error analysis

Status: documentation finalized and next diagnostic scaffolded.

EXP-0037 final read: branch remains CLOSED. The deployed gate failed 1/3 against fixed
`blend_0.5` (s29 856.3 < 891, s77 696 > 692.3 by only +0.5%, s45 175 < 400 with higher death).
The result now explicitly says the verdict rests on deployed rollout, while the flat-loss read is
scoped to the tried producer-only feature row and does not prove all possible observations fail.
`spec.json` reproduce now includes `--seed 0` for the post-P2 deterministic fit path. The pulled
`gate/` artifact remains the original pre-P2 pod output (`summary.csv`, `spec.json`,
`residual_scorer.pt` only); the code emits `loss_history.csv` and `gate_rows.csv` on rerun.

EXP-0038 created: `experiments/EXP-0038-actor-teacher-error-microscope/`.

Purpose: before adding features or another actor, compare teacher marginal values and actor
residual scores on the same routing decisions. The diagnostic should dump decision rows,
producer rows, top mistakes, alias groups, and a summary. Read: if high-regret mistakes separate
by order/backlog/TTL fields, observation enrichment is justified; if current features alias
conflicting labels, name the missing fields; if labels remain inconsistent, target noise remains
the blocker.

Handoff prep: EXP-0038 `HANDOVER.md` now has a concrete first implementation pass, expected CSV
schemas, local-smoke size, and non-goals. No EXP-0038 code, smoke, pod run, or commit was made in
this prep step. A targeted value-router pytest command was started during finalization and then
intentionally interrupted by the user; treat the last completed targeted result as the prior
12-pass run and rerun the targeted suite before behavior changes.

Next implementation task: build the EXP-0038 dump generator on a tiny local smoke, then run the
s29/s77/s45 diagnostic on a pod only if the local output has all required audit tables.

---

## 2026-07-19 — EXP-0038 dump generator built (TDD), local smoke green

Status: diagnostic driver implemented and smoke-proven locally. No pod run yet.

Built `tokenomics.eval.value_router_errors` (measurement only, torch-free core; torch used
only to hold/reproduce the actor). Reuses the EXP-0037 teacher (`marginal_value_labels`) and the
EXP-0037 actor (`ResidualScoreModel` via `train.value_router`), so labels + scores match the gate.

Public surface (TDD, one behavior per test):
- `decision_metrics(labels, scores, eligible_cols)` — teacher-best vs actor-best over the eligible
  set, `decision_regret` (teacher value the actor pick forgoes, >= 0), `top1_match`,
  `rank_correlation` (spearman, 0.0 when < 2 eligible / constant side).
- `summarize` (top1-match rate + mean/max regret), `top_mistakes` (sort by regret),
  `alias_groups` (identical actor feature_hash -> conflicting teacher `teacher_pos`, label_variance).
- `collect_error_records` — drive fixed blend_0.5 on each task/seed; at every decision with >= 2
  eligible producers record teacher labels, actor scores, actor-visible feature rows, and order
  identity (cycle, order, product, qty, order_age/ttl from `env.system.orders[oid].cycle_counter`,
  backlog + same-product backlog).
- `decision_row` / `producer_rows` render the two base CSVs; `fit_reproducible_actor` reproduces the
  gate actor via the seeded collect+fit (HANDOVER "seeded rerun path", pre-P2 gate/ artifact avoided).
- `run_value_router_errors` + `main()` emit the full dump layout: `decisions.csv`, `producers.csv`,
  `top_mistakes.csv`, `alias_groups.csv`, `summary.csv`, `spec.json`. CLI: `--actor-ckpt` to
  microscope the pulled gate scorer, else seed-rerun the fit.

Design correction during build: the driver microscopes the ALREADY-TRAINED actor (required `model`
arg / seeded rerun), not a random-init model — a fresh net would say nothing about why the trained
actor fits the teacher badly.

Tests: `tests/test_value_router_errors.py` (8, torch-free) + `tests/test_value_router_errors_run.py`
(3, torch-guarded) green; full suite 670 passed, 1 deselected. Local smoke dump (2 depth-3 scale
tasks, horizon 16) writes all 6 files; schema + mistake/alias extraction confirmed (byte-identical
producer feature rows get identical actor scores while the teacher splits them -> 121 alias groups).
These toy-task numbers are SCHEMA PROOF only, not decision evidence.

Blockers: none. Worktree UNCOMMITTED.
Next: run the real s29/s77/s45 dump on a pod (CPU, `--python python3.12`) via
`run_pod_training.py`, pull into EXP-0038, then read top_mistakes/alias_groups for the DoD verdict
(enrich obs / change teacher / stop). Pod run only on user go — it's a decision-gating diagnostic.

---

## 2026-07-19 — EXP-0038 review fixes (P1 CLI import, P2 rank-corr + alias precision)

Status: three review findings on the dump generator fixed, TDD, suites green.

- P1 (blocker): `main()` imported the producer scorer from `.producer_scorer` (tokenomics.eval),
  but the EXP-0035 scorer loader lives in `tokenomics.train.producer_scorer` — the pod CLI would
  have crashed after arg parsing, before writing any CSV. Fixed the import; also moved
  `_load_archive` to `.curriculum_replay`. New test `test_cli_resolves_the_producer_scorer_and_
  archive_loaders` pins the exact deferred imports the CLI performs.
- P2a: `_spearman` judged constancy on argsort ranks, but argsort of a constant vector still yields
  distinct ordinal ranks, so flat all-zero teacher labels (a core failure mode this microscope
  diagnoses) got a spurious +1 correlation. Now judges constancy on the raw values (`np.ptp`) and
  ranks tie-aware (average ranks) via new `_rankdata`. Verified in-pipeline: all 448 flat-label
  decisions now report rank_correlation 0.0 (was ±0.x).
- P2b: `_alias_item` rounded feature rows to 2 decimals before hashing, so rows differing in the
  3rd/4th decimal collapsed into false aliases — manufacturing the "missing observation" signal.
  Now hashes the EXACT float32 rows; only byte-identical actor features share a feature_hash.

Tests: `test_value_router_errors*` 15 passed; value_router+errors suites 27 passed. CLI loads via
`python -m tokenomics.eval.value_router_errors --help`. Blockers: none. Worktree UNCOMMITTED.
Next unchanged: real s29/s77/s45 dump on a CPU pod on user go.

---

## 2026-07-19 — EXP-0038 pod run DONE: current actor-row aliases marginal-value target

Status: diagnostic complete for the current 7-feature actor row. Later review narrowed the
conclusion on 2026-07-20; see EXP-0038A.

Ran `tokenomics.eval.value_router_errors` on a fresh CPU pod (`--create-pod`,
runpod/base:0.7.0-ubuntu2004, `--cpu python3.12`, 16 vCPU; stopped+terminated after pull) over the
three EXP-0036/0037 gate tasks. Exact task_ids recovered from EXP-0037 `gate/summary.csv` (the
"s29/s77/s45" nicknames are long ids; `-s45` collided). Seeded actor rerun (seed 0, delta 1.0,
lookahead 8, eval-seeds 3, horizon 100, epochs 300) reproduces the EXP-0037 gate actor. 13,940
labeled decisions. Dump pulled to `experiments/EXP-0038-actor-teacher-error-microscope/dump/`.

Scoped verdict: **the current actor observation is insufficient.** Three-step read (see RESULT.md):
1. 69.4% of decisions have a flat teacher (blend_0.5 already optimal; nothing to learn). Only 30.6%
   are real choices; top-1 there 0.451, regret concentrated (top 5% of decisions = 65% of regret).
2. Of 1,975 real mistakes, only 3.9% are between identical current-row producers; 96.1% differ on a
   current feature (makeable/headroom/reserved). This rules out pure local producer-row aliasing,
   but does not rule out richer order/system context.
3. Decisive: exact feature rows seen ≥5× receive teacher labels ranging median 10 / mean 14.3 / max
   90 units (66.9% > 1u) — the per-decision marginal value of a logit bump is a global
   path-dependent quantity, not a function of the current 7-feature row. Explains the flat ~12.9
   fit loss for the current actor. Consistent with EXP-0009 (value-from-global_state didn't
   generalize), but not final proof that all deployable observations fail.

Concrete next action after review: run EXP-0038A's deployable observation ladder before training any
new actor. `blend_0.5` remains the orderer floor. Registry entry: DIAGNOSTICS.md #12 (actor fits
teacher badly — missing obs vs unlearnable target). Records later narrowed in RESULT.md,
spec.json, experiments/README.md row.

Artifacts: `dump/{summary,top_mistakes,alias_groups}.csv` + `spec.json` small (track directly);
`dump/{decisions.csv 2.5M, producers.csv 6.5M}` heavy (git-lfs if committed). Staged `_pod_ckpts/`
removed. Tests green (670). Blockers: none. Worktree UNCOMMITTED.

---

## 2026-07-20 — EXP-0038 narrowed; EXP-0038A opened for missing-observation audit

Status: EXP-0038 is no longer treated as final proof that observation enrichment is dead.

Review of the pulled dump upheld the current-row failure but overturned the overclaim. The
EXP-0037 actor only sees a 7-feature producer row. Repeated current rows have large teacher-label
spread (median 10, mean 14.3, worst 90), and the residual actor explains only ~1-2% of centered
label variance, so the current observation cannot fit the marginal-value teacher. But adding logged
decision context (especially product/quantity and other context fields) reduces fixed-feature label
conflict, so richer deployable observations remain a live explanation.

Docs updated:
- `EXP-0038/RESULT.md`, `HANDOVER.md`, `spec.json` narrowed to "current-row aliasing" rather than
  "all observations fail."
- `docs/DIAGNOSTICS.md` entry 12 now says to repeat the label≈f(obs) test over candidate
  deployable observation blocks before stopping a teacher.
- New continuation folder:
  `experiments/EXP-0038A-missing-observation-audit/` with `HANDOVER.md`, `RESULT.md`, and
  `spec.json`.

EXP-0038A goal: identify which teacher information is missing from actor input, and separate
deployable observations (order/system/history/global context) from oracle-only rollout/future
facts. No new actor training is licensed until a deployable observation block materially reduces
held-out label conflict and real-choice regret on at least 2/3 gate tasks.

---

## 2026-07-27 — EXP-0038A missing-observation audit DONE: deployable dump blocks do NOT license enrichment

Status: audit closed for the blocks the EXP-0038 dump already logs (current_row, +order,
+local_queue). Verdict: observation enrichment NOT licensed. New driver, TDD, suites green.

Built `tokenomics.eval.observation_ladder` (torch-free; reuses EXP-0038 `dump/`, no re-sim). Reads
the 13,940 decisions / 37,804 eligible-producer rows over s29/s77/s45 and walks a cumulative
deployable-observation ladder with two metrics per block:
- **Exact-alias label spread** (`alias_spread`): group producer rows by the block's feature tuple;
  for groups seen >=5x, teacher-label range. current_row median 10 / frac>1 0.669 → **+order median
  0 / frac>1 0.224** (collapses; +local_queue identical).
- **Held-out routing probe** (`heldout_probe`, leave-one-seed-out per task): two families, per the
  user's cross-check choice — **ridge** (standardized block feats + current×order-context
  interactions) AND **kNN** — scored by within-decision real-choice regret + top1 + centered r2 vs
  two floors (`floor_metrics`): random and the DEPLOYED EXP-0037 actor (`actor_score` argmax).

Decision gate (both conditions; probes must agree >=2/3):
1. Label conflict collapses — PASS for +order.
2. Held-out routing beats floor on >=2/3, both probes — **FAIL for every block.** Deployed
   current_row-only actor floor (regret s45 1.313 / s77 1.378 / s29 1.248) already beats the +order
   ridge probe 3/3 and kNN 2/3; best deployable +order-kNN beats actor on only 1/3, ridge 0/3;
   probes never agree; **held-out r2 negative everywhere**. Numbers materialized in
   `audit/{block_summary,alias_by_block,probe_by_block,top_mistakes_by_block}.csv` (all <10KB,
   tracked directly).

Mechanism (why alias-collapse ≠ routing): order/queue fields are CONSTANT across a decision's
eligible producers → they shift the between-decision LEVEL (collapsing the alias spread) but carry
zero within-decision contrast to pick WHICH producer is best. Corroboration: 96.7% of the top-60
deployed-actor mistakes have teacher-best ≠ actor-best DIFFERING on the current row — producers are
already distinguishable, the actor just ranks them wrong. Not a missing per-producer observation;
the teacher's within-decision preference is not a learnable function of these features (global
path-dependent rollout quantity; consistent with EXP-0009 + EXP-0038).

DIAGNOSTICS #12 refined: added the "between-decision level vs within-decision contrast" trap +
discriminating check (score blocks by held-out routing regret vs the incumbent actor floor, both
probe families, not alias spread alone). RESULT.md / spec.json / README row updated to CLOSED.

Still open (HANDOVER non-goal upheld — ladder not exhausted): per-producer blocks
full_producer_state (4) + history (7) are the only remaining candidates that could re-rank within a
decision; they need a live re-dump (extend `value_router_errors`). system_demand (5) /
recipe_bottleneck (6) are decision-constant like +order. Not computed: near-neighbor label spread
(supplementary; exact-alias coverage adequate, routing probe is the decisive metric — cannot change
verdict). `blend_0.5` stays the orderer floor; no actor trained.

Tests: `tokenomics/tests/test_observation_ladder.py` (8, torch-free) green; full suite green.
Blockers: none. Worktree UNCOMMITTED.
Next (superseded by EXP-0038B): the per-producer/product re-dump happened and Phase 1 failed. The
remaining continuation is Phase 2 system-wide interaction metrics; retiring the target requires that
Phase 2 fail too.

---

## 2026-07-27 — EXP-0038A probe metric CORRECTED (review P1/P2/P3); verdict stronger (0/3)

Status: review of the EXP-0038A intermediate tables found the probe metric did not match the stated
gate. Fixed all three, regenerated `audit/`, updated RESULT/spec/README/DIAGNOSTICS. Verdict
direction unchanged but cleaner and stronger.

Bugs (in `tokenomics.eval.observation_ladder`):
- **P1 metric ≠ gate.** Probe fit RAW `teacher_label` and scored EVERY contested decision (incl.
  flat-teacher rows), while RESULT claimed "centered r2 / real-choice regret". Fix: fit
  decision-centered labels (`_decision_centered`); score only real-choice decisions (≥2 eligible AND
  non-flat) in `_routing_metrics`; centered within-decision r2.
- **P2 inconsistent aggregation.** Probe averaged unweighted FOLD-MEANS; actor floor was
  decision-weighted over the whole task → not comparable (this manufactured the spurious "+order kNN
  beats actor 1/3"). Fix: `heldout_probe` now POOLS held-out folds (each decision once) →
  decision-weighted, matching floors; actor floor now runs through the same `_routing_metrics`.
- **P3 top1 tie handling.** Probe required pick==argmax(labels); floor accepted any tie at best. Fix:
  tie-aware on both sides.

Regenerated numbers (real-choice-only, decision-weighted, per-task regret; lower better):
actor floor 3.93/4.75/4.12 (s45/s77/s29), random floor 5.24/6.24/5.38. **No deployable block, either
probe, beats the actor floor on ANY task → 0/3 everywhere** (+order kNN closest 4.78/6.15/4.68).
Probes DO beat the random floor (not noise). Centered within-decision r2 ≈ 0 (ridge) / slightly neg
(kNN) for all blocks → deployable obs explain ~none of the within-decision contrast. Alias collapse
(median 10→0) upheld = between-decision level artifact. Attribution unchanged (96.7% of top mistakes
differ on current row).

Tests: `test_observation_ladder.py` now 12 (added flat-exclusion/tie-aware + floor-consistency
regressions; relaxed the additive-r2 threshold to the honest centered-with-shrinkage value 0.7).
All green. DIAGNOSTICS #12 gained a "score the check correctly or it lies" subsection.

Blocks 4+7 driver (value_router_errors `--full`) + ladder support (built earlier today) unchanged and
still green. Pod re-dump for blocks 4+7 is HELD pending user approval (pod 629uddqwwrh3d9 created then
STOPPED before training on user request; not terminated). Blockers: none. Worktree UNCOMMITTED.
Next: on approval, resume the blocks 4+7 pod re-dump and run the (now-correct) gate on them.

---

## 2026-07-27 — EXP-0038B opened: producer/product observation ladder

Status: new diagnostic experiment created; no run launched and no actor trained.

User pointed out the old `current_row` is too compressed for the real routing question: products
can require several capacity types, producers can use the same capacity for competing products, raw
inventory can satisfy multiple products, and historical fulfilment speed may matter. EXP-0038B
created to test that richer class of deployable per-producer/product metrics before retiring the
marginal-value teacher.

New folder: `experiments/EXP-0038B-producer-product-observation-ladder/` with `HANDOVER.md`,
`RESULT.md`, and `spec.json`. The handover prioritizes per-producer blocks first: dedicated
target-product throughput, per-capacity dispersion, capacity opportunity cost, dependency input
cover and fulfilment speed, base/cutoff margins, and last-5-cycle producer/product history. Only
after those are measured should system-wide demand/capacity/recipe/other-actor context be added,
and then only through producer interactions because most system fields are decision-constant.

Gate unchanged from corrected EXP-0038A: decision-centered labels, real-choice decisions only,
decision-weighted held-out folds, tie-aware top1, ridge+interactions and kNN, compared against the
deployed actor floor. No new actor training unless a deployable block beats the actor floor on at
least 2/3 tasks for both probes.

---

## 2026-07-27 — EXP-0038B instrumentation built (Phase-1 per-producer/product ladder), local smoke green

Status: DONE (local build + TDD + schema smoke). No actor trained (HANDOVER non-goal upheld). Pod
re-dump = user's next step (all real runs on pod).

Built the deployable Phase-1 per-producer/product observation blocks the EXP-0038B handover
prioritized — the missing class that can re-rank producers *within* one decision (EXP-0038A proved
order/queue context only shifts the between-decision level).

Artifacts:
- `tokenomics/eval/producer_product_features.py` (NEW, torch-free, pure): P1 `pp_throughput_`
  (dedicated throughput dispersion), P2 `pp_opp_` (opportunity cost vs competing products), P3
  `pp_dep_` (input cover in target units + primal sentinel), P4 `pp_base_` (base-policy routing
  margin), P6 `pp_fulfill_` (TTL feasibility). Each block = one pure fn on a `ProducingAgent` + sheets.
- `tokenomics/eval/value_router_errors.py`: `producer_product_rows` orchestrator + `--feature-set
  producer_product_v1` (preserves `--full`); rides pp_* + reused `h_*` (P5) into producers.csv.
  Factored `history_block` out of `producer_fullstate_row` (existing behavior unchanged).
- `tokenomics/eval/observation_ladder.py`: `PP_BLOCKS`; `ladder_for`/`load_dump` discover the pp_
  prefixes and score `+pp_throughput/+pp_opportunity/+pp_dependency/+pp_base/+pp_fulfilment/+history`
  cumulatively with the corrected EXP-0038A gate metric.
- Tests: `test_producer_product_features.py` (7, torch-free — multi-cap dispersion, target-unit cover
  vs raw inventory, primal sentinel, capacity-shared opportunity cost, base rank/margin, TTL
  feasibility, history-past-cycles-only), +2 integration in `test_value_router_errors_run.py`, +1
  ladder-discovery in `test_observation_ladder.py`. **Full suite: 699 passed.**

Local schema smoke (2-task depth-3, horizon 14): dump emits 60 pp_ columns; ladder discovers all six
blocks and scores each vs the actor floor end-to-end. Pipeline proven; NOT a gate read.

Deliberate deployability deviations from the handover ladder (documented in RESULT):
- P3 transitive/upstream-speed cols OMITTED — need recipe graph + other producers' state (Phase-2
  system concern, not a per-producer deployable fact).
- P4 price cols OMITTED — this env has no decision-time cut-off/market price; base logit is the proxy.
- P5 uses folded telemetry EMAs (no exact 5-cycle window exists); handover-sanctioned fallback.

Blockers: none. Worktree UNCOMMITTED.
Next (user): stage `_pod_ckpts/{gate_archive.jsonl,producer_candidate.pt}`, run the CPU-pod re-dump
with `--feature-set producer_product_v1` (command in RESULT.md), pull `dump/`, run
`observation_ladder` audit, then read the gate and fill in RESULT Conclusions.

## 2026-07-28 — EXP-0038B Phase 1 pod re-dump + audit: gate FAILED; Phase 2 remains open

Status: DONE for Phase 1 per-producer/product metrics. Diagnostic-only; no enriched actor trained
because the gate did not pass. Do NOT mark the marginal-value teacher retired from this result alone.

Ran the s29/s77/s45 re-dump on a fresh CPU pod `s8spjt6z0xs5s6` (create-pod,
runpod/base:0.7.0-ubuntu2004, --cpu python3.12, 16 vCPU; stopped+terminated after pull) with
`--feature-set producer_product_v1`, same params as EXP-0038 (delta 1.0, lookahead 8, eval-seeds 3,
horizon 100, epochs 300, seed 0). 13,940 decisions (identical to EXP-0038 — provenance holds),
37,804 candidate rows, 60 `pp_*` + 8 `h_*` columns. Then `eval.observation_ladder` scored the
cumulative Phase-1 ladder locally with the corrected EXP-0038A gate. A local rerun of the audit to
`/tmp/exp0038b-review-audit` reproduced the recorded audit CSVs byte-for-byte.

Result: every tested Phase-1 block fails held-out within-decision routing regret vs the deployed
actor floor (s45/s77/s29 = 3.93/4.75/4.12): 0/3 beats-actor for BOTH ridge+interactions and kNN,
centered within-decision r² ≈ 0 (ridge best 0.0027, mostly negative; kNN always negative), and adding
pp_ blocks raises regret (ridge 5.31→5.66, kNN 5.20→5.52 — overfit noise). Exact-alias conflict falls
on repeated rows (median range 10.0→0.0; frac>1 0.669→0.034 at +pp_throughput→0.012 at +pp_base), but
this is secondary evidence only: repeated-row coverage collapses as continuous pp_/h_ features are
added (75.1% current_row, 26.2% +pp_throughput, 8.7% +pp_base, 4.6% +history). Do not claim the label
is deterministic over the full observation space from exact-alias collapse alone.

Corrected conclusion after review: Phase 1 per-producer/product enrichment is NOT licensed and no
enriched actor should be trained from it. However, EXP-0038B has NOT tested Phase 2 system-wide
demand/capacity/recipe/other-actor metrics (`sys_*`), so retiring the marginal-value teacher as a
target is still premature. `blend_0.5` remains the orderer floor while Phase 2 is evaluated.

Artifacts: dump/ + audit/ under experiments/EXP-0038B-*; RESULT.md/spec.json/DIAGNOSTICS/README
updated to the narrowed conclusion. Worktree UNCOMMITTED. `tokenomics/_pod_ckpts/` staged
(gate_archive.jsonl + producer_candidate.pt) — safe to delete when no longer needed.

Next: implement Phase 2 system-wide interaction ladder. Add a feature set such as
`producer_product_system_v1`; append `sys_*` columns; update `observation_ladder` so `sys_*` is
treated as context for ridge interactions; keep the same held-out gate. Retire the marginal-value
teacher only if the Phase 2 interaction blocks also fail.

---

2026-07-28 - EXP-0038B Phase 2 system-wide interaction ladder (DONE — both phases failed, target retired)

Status: COMPLETE. Implemented the Phase-2 system-wide (`sys_*`) observation ladder per the
HANDOVER and ran it to a gate read on a fresh CPU pod. BOTH EXP-0038B phases now fail the same
corrected held-out routing gate → the EXP-0037 marginal-value teacher is retired as an actor
training target. No actor was trained (gate never passed). `blend_0.5` remains the orderer floor.

What was built (all green, full suite 708 passed):
- `eval/producer_product_features.py`: 5 torch-free `sys_*` builders — sys_demand (S1), sys_capacity
  (S2, incl. free-throughput HHI + required-type global scarcity), sys_recipe (S3, depth /
  downstream-demand-unlocked / transitive inputs / upstream cover), sys_cutoff (S4, urgency proxies —
  env has NO decision-time price, explicit absence flag, no faked price), sys_actor (S5). Plus
  helpers system_open_qty / transitive_inputs / _global_free_units. Most fields decision-constant;
  the producer-varying ones are `*_competing_for_p_capacity_qty`.
- `eval/value_router_errors.py`: new `--feature-set producer_product_system_v1` (preserves
  producer_product_v1); `producer_product_rows(system=True)` appends sys_ cols; `_system_context`
  precomputes per-decision shared facts once (all orders / all producers / folded telemetry).
- `eval/observation_ladder.py`: `SYS_BLOCKS`; `ladder_for` appends sys blocks after +history;
  `_split_cols` routes `sys_*` into ridge interaction CONTEXT (they cannot re-rank additively — kNN
  is the additive cross-check); `load_dump` discovers `sys_` prefix.
- Tests: +5 sys builder unit tests, +2 dump integration tests, +3 ladder tests (split-cols routing,
  interaction-recovery mechanism, sys ladder discovery).

Run: pod `5gi98ku7gfnf2e` (create-pod, runpod/base:0.7.0-ubuntu2004, --cpu python3.12, 16 vCPU;
stopped+terminated after pull). `--feature-set producer_product_system_v1`, same params as EXP-0038
(delta 1.0, lookahead 8, eval-seeds 3, horizon 100, epochs 300, seed 0). 13,940 decisions / 37,804
rows / 92 pp_+h_ + 32 sys_ cols; top1 0.49, mean regret 1.322 — byte-identical trajectory to
EXP-0038. Then `eval.observation_ladder` scored the full 14-block ladder locally.

Result: every `sys_*` block is 0/3 beats-actor for BOTH ridge+interactions and kNN. Actor floor
regret s45/s77/s29 = 3.93/4.75/4.12; sys blocks ridge 5.6-5.9 / kNN 5.5-5.7 (at/above the random
floor 5.24/6.24/5.38). Centered within-decision r² <= 0 and worsening as blocks accumulate (ridge
-0.02→-0.18→-0.22→-0.21→-0.25; kNN ~-0.05). Alias collapse is not a pass: repeated-row coverage
falls to 543 rows / 93 groups at +sys_demand (from 28,383 / 902 at current_row).

Decisions: Retire the marginal-value teacher as an actor target (HANDOVER planned read: license
retirement iff Phase-2 interaction blocks also fail — they do). The teacher's within-decision
producer preference is not a function of ANY tested deployable observation (per-producer/product OR
system-wide interaction context) — it is produced by the 8-cycle rollout. A learned orderer must
either emulate the rollout (a compute lever, cf. EXP-0009 value-guided planner) or optimise a
different, obs-measurable objective. This confirms + extends memory exp0038-marginal-value-target-
unlearnable and DIAGNOSTICS entry 12.

Artifacts: dump_sys/ + audit_sys/ under experiments/EXP-0038B-*; RESULT.md / spec.json / DIAGNOSTICS
entry 12 / experiments/README.md all updated to the both-phases-failed / target-retired conclusion.
Worktree UNCOMMITTED (commit only on request). `tokenomics/_pod_ckpts/` + `tokenomics/_pod_inputs/`
staged (producer_candidate.pt + gate_archive.jsonl) — safe to delete when no longer needed.

Blockers: none.

Next: marginal-value-target arc is closed. If the orderer lever is revisited, either (a) pursue the
rollout-emulation / compute lever (short-horizon planner, EXP-0009 family) rather than obs
enrichment, or (b) define a different, obs-measurable orderer objective. Do NOT train an actor on the
marginal-value target.

---

## 2026-07-29 — EXP-0039 GNN observation probe: local build + TDD complete (graph_primitives_v1)

Status: Built the full EXP-0039 supervised representability probe scaffolding, TDD, all green
locally. This is the narrower follow-up to EXP-0038B: the hand-metric ladder failed, but that
only proved the *compressed* metrics were not learnable — EXP-0039 asks whether a GNN over
*primitive relational* deployable state recovers the marginal-value teacher's within-decision
producer preference. Probe only; no orderer trained, no deployed gate.

Design decisions (accepted before coding, ARCH.md Phase 0): (1) commitment visibility = VISIBLE —
other orderers' current-cycle pending bids + live reservations are part of the deployable contract,
exposed through a domain-owned graph view (causal pre-action state, not future leakage), recorded
in graph_schema.json; (2) model = minimal pure-Torch relational message passing (no PyG dep).

Artifacts (new, all with tests):
- `tokenomics/domain/graph_view.py` — pure EconomyGraphView (mirrors OrdererEconomyView): all
  node types + recipe/eligibility/capacity/inventory/telemetry/commitment relations, structural
  zeros, no numpy/torch/IO/labels. `tests/test_graph_view.py` (7). Caught + fixed a real bug:
  UNDEFINED_PRICE is NaN, so the cutoff mask must use isnan, not `!= UNDEFINED_PRICE`.
- `tokenomics/eval/graph_value_router_dump.py` — frozen graph_primitives_v1 schema (ordered
  feature names + log1p/binary transforms + reverse relations + schema hash), view→id-free numeric
  arrays (local indices; raw ids only in a separate audit map), blend_0.5 trajectory dump (one
  graph per contested decision, extracted BEFORE teacher/step), pickle-free NPZ shards + loader,
  decisions.csv (byte-identical target) / candidates.csv / topologies.jsonl. `tests/test_graph_dump.py` (7).
- `tokenomics/train/graph_router_probe.py` — pre-registered 4-layer hidden-64 relational GNN
  (schema-role embeddings only, sum agg, residual+LN, shared candidate head), decision-weighted
  MSE on decision-centered labels, leave-one-seed-out gate reusing observation_ladder metrics +
  floors. `tests/test_graph_probe.py` (5) + end-to-end smoke `tests/test_graph_probe_smoke.py` (1).
- `experiments/EXP-0039-gnn-observation-probe/` — spec.json + RESULT.md skeleton with the exact
  staged pod commands + predeclared gate + baselines-to-report.

Decisions: predeclared gate is fixed (beat deployed actor floor on >=2/3 tasks, centered r²>0 on
>=2/3, >=10% avg relative regret reduction, no task worse than actor by >5%, stable across >=3
training seeds). Beating only random = fail. Report the EXP-0038B tabular numbers copied, not
recomputed; do NOT copy EXP-0038B's stale `source_dump` provenance bug.

Blockers: none. Local build follows the HARD RULE — real s29/s77/s45 dump + probe run on a Runpod
CPU pod (Phases 7-8, commands in RESULT.md). Worktree UNCOMMITTED (commit only on request).

Next: launch the pod dump; verify n_decisions=13940 / n_candidates=37804 / decisions.csv
byte-identical; run the probe across 3 training seeds; apply the predeclared gate without
reinterpretation and fill RESULT.md. Pass → license a deployed-GNN orderer experiment; fail →
strengthens exp0038-marginal-value-target-unlearnable (target produced by the rollout, not
decision-time state even represented relationally).

### 2026-07-29 — EXP-0039 pre-run review corrections (before any pod spend)

Status: DONE (local, TDD). A pre-run review of the EXP-0039 build found three input defects that
each bias a supervised probe toward a FALSE FAIL. Verified every claim against the normative docs
(ARCH.md steps 4–6/290–296, HANDOVER.md 204–207) — all real — and fixed test-first. A negative
result from the pre-fix build would have been invalid, so this gates the pod run.

Artifacts (edits, all with new tests):
- `eval/graph_value_router_dump.py` — added `signed_log1p` transform; `free`+`headroom` now
  `signed_log1p` (were `log1p` → `NaN/−inf` when reserved>current). `_assert_finite(graph)` in the
  dump loop rejects any non-finite tensor before serialization. Schema hash changed
  76bf1d42a75cd622 → 9918b12f69760bcd. Tests: `test_free_capacity_uses_signed_transform_and_stays_finite`,
  `test_dump_tensors_are_all_finite`.
- `train/graph_router_probe.py`:
  - `_flat_decisions` + gate now trains PRIMARILY on non-flat decisions (flats dropped per fold,
    `n_train_flat_excluded`/`n_train_decisions` reported). EXP-0038 dump is ~69% flat (9,675/13,940).
    Test: `test_flat_decisions_are_excluded_from_training_and_counted`.
  - `FoldNormalizer` (fit train-fold continuous cols, apply to train+held-out; binary untouched;
    `.groups()` audit) wired into `heldout_gate`; stats persisted to probe `spec.json`. ARCH.md 290–291.
    Test: `test_fold_normalizer_standardizes_continuous_only_and_is_train_fitted`.
  - `_verdict` computes the five pre-registered pass conditions mechanically (no reinterpretation);
    `run_probe` emits it + `schema_hash`. `_load_dump` validates schema hash + finiteness + candidate
    alignment. Device-aware (`cuda` if available else `cpu`). Tests:
    `test_gate_verdict_applies_preregistered_pass_conditions`; smoke asserts verdict+schema_hash+normalization.

Decisions: right-sized finding #4 (scale) — kept full-batch fit (it is the pre-registered fold fit)
but made it device-aware so a GPU pod accelerates the 27 fits; did NOT rewrite to mini-batching
(would change the pre-registered fit). Flats excluded (not downweighted) and the count reported, per
HANDOVER's "report whether they were included".

DIAGNOSTICS: added #13 "A supervised probe FAILS — is the input even valid, or is the fit real?"
(finite inputs + non-flat training composition + fold standardization must all hold before trusting
a FAIL).

Blockers: none. 25 graph tests + end-to-end smoke green. Full suite re-run in progress.
Worktree UNCOMMITTED (commit only on request).

Next (unchanged): launch the s29/s77/s45 dump on a Runpod pod; verify n_decisions=13940 /
n_candidates=37804 / decisions.csv byte-identical to the EXP-0038 dump; run the probe across 3
training seeds; the probe now emits the verdict automatically — record it in RESULT.md. Pass →
license a deployed-GNN orderer experiment; fail → strengthens exp0038-marginal-value-target-unlearnable.

### 2026-07-29 — EXP-0039 second review round (5 more corrections, still pre-pod)

Status: DONE (local, TDD). A second review of the corrected build found five issues, two of which
(1,2) could invalidate the "mechanical" read. Verified each against HANDOVER/ARCH — all real — fixed
test-first.

1. Verdict false-PASS on instability. `_verdict` averaged r2 across seeds while stability only
   checked regret-vs-floor → r2 −100/−100/201 passed. Now the four numeric conditions must hold for
   the seed-mean AND independently for each of ≥3 seeds (any seed fails ⇒ no-pass); dispersion
   (regret/r2 std, r2 min/max) reported. Test: `test_gate_verdict_no_pass_when_a_seed_is_unstable`.
   HANDOVER 241.
2. Shallow dump validation. `_load_dump` now REQUIRES graph_schema.json and validates one graph per
   decision: exact decision coverage, `candidate_index` 0..k-1, count vs graph and `n_eligible`,
   producer-id alignment via topologies.jsonl, task/seed consistency, optional expected counts. Test:
   `test_load_dump_requires_schema_and_validates_per_decision_alignment` (missing schema + collapsed
   candidates both raise). ARCH 343.
3. Persisted normalization ≠ trained normalization. Dropped the all-graph refit (leaked held-out +
   flats); `spec.json` `folds[]` now stores the ACTUAL per-fold train stats.
4. Missing audit artifacts. `run_probe` writes `predictions.csv` (gate recomputable without a
   checkpoint or the sim — smoke asserts a per-task regret recomputed from it matches the reported
   metric to 1e-6) + per-fold `checkpoints/*.pt` + loss histories. HANDOVER 379/405/408.
5. Device correction bypassed. Probe step re-staged on a GPU pod (dump stays CPU) in RESULT.md.

Artifacts: `train/graph_router_probe.py` (`_verdict` rewrite, `_load_dump` full validation,
`heldout_gate` per-fold capture, `run_probe` predictions/checkpoints/folds). Tests: +3 in
test_graph_probe.py, smoke extended. 27 graph tests + smoke green; full suite re-run in progress.
Worktree UNCOMMITTED.

Next (unchanged): launch s29/s77/s45 dump (CPU pod) → verify counts + decisions.csv byte-identical →
probe (GPU pod, 3 seeds) → the verdict is emitted + recomputable from predictions.csv; record in
RESULT.md.

### 2026-07-29 — EXP-0039 third review round (4 more corrections, still pre-pod)

Status: DONE (local, TDD). Third review found 4 issues (2 blocking); verified vs HANDOVER/ARCH — all
real — fixed test-first.

1. Metadata alignment. `_load_dump` compared candidate seed only within a group, not against
   decisions.csv → a candidate seed flipped to 999999 was accepted. Now each candidate's
   task/seed/cycle/order_id must equal the decisions.csv row. Test: `_load_dump` case (c) raises on a
   mismatched seed.
2. Pre-registered shape not enforced. `_verdict` passed with 2 tasks (≥2 thresholds) and heldout_gate
   accepted <3 seeds; run_probe never passed expected_counts; spec key alias mismatch. Now
   `run_probe(require_shape=True)` requires 3 tasks × ≥3 traj seeds × ≥3 model seeds; `_verdict` has a
   `three_tasks` condition; `_load_dump` accepts expected counts (both key spellings) and CLI has
   `--expect-decisions/--expect-candidates`. Tests: `test_gate_verdict_requires_the_three_task_shape`,
   smoke asserts a partial dump raises.
3. GPU determinism. `train_gnn` now pins `torch.set_num_threads(1)` → CPU bitwise-exact (was ~5e-4
   multithread jitter); `_determinism_preflight` trains a subset twice and aborts if max abs pred diff
   > tol (0 CPU / 1e-4 CUDA); device/torch/tol recorded in spec.json. Deterministic gate read staged on
   a CPU pod (reverses round-2 #8 GPU default — determinism > speed for a licensing gate). HANDOVER 381.
4. Actor-floor fallback. `--actor-floor-json {task: mean_regret}` now REPLACES the derived floor in the
   computed verdict (both recorded); smoke asserts `verdict_floor=="override"`. RESULT.md 109.

Artifacts: `train/graph_router_probe.py` (`_load_dump` metadata+counts, `run_probe` shape guard +
device/determinism preflight + override + spec fields, `_verdict` three_tasks, `train_gnn` 1-thread,
`heldout_gate` device param, CLI flags). Tests: +1 in test_graph_probe.py, `_load_dump` test extended,
smoke extended. 28 graph tests + smoke green; full suite re-run in progress. Worktree UNCOMMITTED.

Next (unchanged): launch s29/s77/s45 dump (CPU) with count stop-checks → deterministic CPU probe
(3 seeds, `--expect-decisions 13940 --expect-candidates 37804`) → verdict emitted + recomputable from
predictions.csv; record in RESULT.md.

### 2026-07-30 — EXP-0039 fourth review round (4 more corrections, still pre-pod)

Status: DONE (local, TDD). Fourth review found 4 issues (2 blocking) + 1 budget note; verified vs
ARCH/HANDOVER — all real — fixed test-first. Schema hash UNCHANGED (audit-binding/validation, not
observation changes).

1. Graph↔decision binding (blocker). Nothing bound `graphs[di]` to `decisions.csv` row di — swapping
   two structurally-identical graph payloads (same candidate set, different live features) passed every
   count/producer-alignment check. Now the dump stamps each graph with a `decision_digest(task_id,seed,
   cycle,order_id)` as a SEPARATE shard array (never a model tensor); `_load_dump` recomputes it from
   decisions.csv and rejects a mismatch. Tests: `test_shard_binds_each_graph_to_its_decision_row`
   (dump), `test_load_dump_rejects_a_swapped_graph_payload` (probe).
2. Pre-registered dump config not enforced (blocker). `_load_dump` never read the dump's own spec.json;
   changed task-ids/delta/lookahead/horizon/has_actor were accepted (same decision count, different
   labels). Now `_validate_dump_config` rejects any deviation from `GATE_CONFIG`/`GATE_TASK_IDS`
   (task-ids set, delta, lookahead, horizon, eval_seeds, floor, has_actor); `require_shape` requires
   EXACTLY 3 anchored trajectory seeds (not ≥3) and the exact s29/s77/s45 ids. Test:
   `test_load_dump_validates_preregistered_dump_config`.
3. Effective-floor consistency. metrics_by_task.csv reported the derived actor floor while the verdict
   used the override → contradiction. Now the table carries `verdict_floor_regret`+`floor_source`, and a
   new `floor_by_task.csv` records derived+override+source (verdict reproducible from artifacts). Smoke
   extended to assert both under a 999.0 override.
4. Determinism cert too weak. Preflight was 40 decisions × 10 epochs. Now `_determinism_preflight`
   trains ONE real leave-one-seed-out fold at the run's FULL epochs twice and compares held-out preds
   (the literal HANDOVER 381 identical-full-run condition); epochs/n_train_decisions/fold recorded.
   Smoke asserts `determinism["epochs"]==2` and a real `fold`.
5. Compute budget (not an invalidation). Single-thread-BLAS fits are single-core → 27 fits × 200 epochs
   ≈ ~4 core-hours on the CPU pod; documented in RESULT.md + spec.json, launch detached.

Artifacts: `eval/graph_value_router_dump.py` (`decision_digest`, GraphArrays.decision_digest, shard
write/load, run_graph_dump stamps it); `train/graph_router_probe.py` (`GATE_CONFIG`/`GATE_TASK_IDS`,
`_validate_dump_config`, `_load_dump` digest+config checks, `run_probe` exact-seed+task-id guard +
floor_by_task.csv + verdict-floor columns, `_determinism_preflight` full-epoch real fold). Tests: +1
dump, +2 probe, smoke extended → 31 graph tests green. DIAGNOSTICS #13 extended (f)+(g). RESULT.md/
spec.json updated; probe read confirmed CPU-only (fixed a stale "Step 2 → GPU pod" line). Worktree
UNCOMMITTED.

Next (unchanged): launch s29/s77/s45 dump (CPU) with count stop-checks + byte-identical decisions.csv →
deterministic CPU probe (3 model seeds, `--expect-decisions 13940 --expect-candidates 37804`) → verdict
emitted + recomputable from predictions.csv; fill RESULT.md Results table.

### 2026-07-30 — EXP-0039 fifth review round (4 more corrections: input provenance, still pre-pod)

Status: DONE (local, TDD). Fifth review found 4 issues (2 blocking, both input-provenance); verified vs
spec.json/HANDOVER — all real — fixed test-first. Schema hash STILL unchanged.

1. Input files fingerprinted + validated (blocker). Dump recorded only `has_actor: true` → a wrong
   producer (changes labels) or wrong actor (weakens floor), same shape, passed. Now `sha256_file`
   fingerprints archive+producer+actor ckpt into the dump `inputs` manifest; probe validates against
   pinned `GATE_INPUT_SHA256` (frozen pre-registered hashes: archive bc6ca4eb…, producer 8b51b77c…,
   actor 666533e4…) and rejects a wrong/unfingerprinted input. Tests:
   `test_dump_records_task_seed_sets_and_input_manifest` (dump),
   `test_load_dump_validates_task_seed_sets_and_input_fingerprints` (probe).
2. Exact task→seed sets enforced (blocker). "Exactly 3 seeds" checked only count. Dump records anchored
   `task_seeds`; probe enforces exact s29={29,30,31}/s77={77,78,79}/s45={45,46,47} (`GATE_TASK_SEEDS`)
   against BOTH the dump manifest and the scored candidates.csv seeds.
3. floor_by_task.csv reproduces the verdict independently. Was 4-dp rounded (1.000049→1.0 could flip
   PASS↔FAIL). Now derived+verdict floors at FULL precision (`repr(float)`) + the override JSON's
   source path+sha256 (`override_source` column, `actor_floor_override_source` in spec.json). Display
   metrics_by_task.csv stays rounded. Smoke exercises a 1.000049 override.
4. Determinism claim scoped. The preflight is a REPRESENTATIVE full-epoch fold (CPU bitwise by
   construction), not a whole-27-fit re-run; RESULT/spec/DIAGNOSTICS wording corrected (no code change).

Artifacts: `eval/graph_value_router_dump.py` (`sha256_file`, `run_graph_dump(input_manifest=)` records
`inputs`+`task_seeds`, `main()` fingerprints the 3 inputs); `train/graph_router_probe.py`
(`GATE_TASK_SEEDS`, `GATE_INPUT_SHA256`, `_validate_dump_provenance`, `_load_dump` seed+input checks,
`run_probe` default provenance enforcement + scored-seed check + full-precision `floor_by_task.csv` +
`override_source`, CLI records override source). Tests: +1 dump, +1 probe, smoke extended → 33 graph
tests green. DIAGNOSTICS #13 extended (h)–(k). RESULT.md/spec.json updated (pinned hashes + task_seeds in
params). Worktree UNCOMMITTED. Full suite re-run pending.

Next (unchanged): launch s29/s77/s45 dump (CPU, real archive+ckpts so recorded SHA-256 match the pins) →
deterministic CPU probe (3 model seeds, `--expect-decisions 13940 --expect-candidates 37804`, default
input/seed provenance validation) → verdict emitted + recomputable from predictions.csv + floor_by_task;
fill RESULT.md Results table.

---

## 2026-07-30 — EXP-0039 GNN probe: sixth review round (source revision + authoritative floor + pre-spend input check)

Status: DONE (code + TDD). Sixth review found three provenance/licensing gaps beyond input fingerprints,
all verified against ARCH.md/HANDOVER.md and fixed test-first. Schema hash UNCHANGED (audit/validation,
not observation).

1. Immutable source revision. The observation-schema hash is blind to training/metric/verdict code
   changes, and the pod rsync excludes `.git/` — nothing bound artifacts to the code that produced them
   (HANDOVER.md:413 requires the commit in RESULT.md). Added `source_fingerprint(commit)` = SHA-256 over
   the bytes of the 4 producing modules (git-independent, reproducible on the pod) + optional git commit.
   Stamped into dump spec (`run_graph_dump(source_commit=)`) and probe result (`run_probe(source_commit=)`);
   CLIs take `--source-commit`. COMMIT the source + pass its commit before launch.
2. Authoritative fallback floor. `--actor-floor-json` accepted any `{task: regret}` — an inflated floor
   silently weakens the licensing bar (verdict wants GNN regret < floor). Pinned `GATE_ACTOR_FLOOR_0038B`
   (EXP-0038B audit_sys deployed-actor mean_regret) + `GATE_FLOOR_SHA256`; `canonical_actor_floor()`
   derives it mechanically from the SHA-gated artifact; `assert_override_is_canonical` rejects any other
   value on a `require_shape` run; new `--actor-floor-exp0038b` produces a correct override by derivation.
3. Input check before spend. The dump recorded input SHA-256 but only the probe (after the finished dump)
   compared them — a wrong staged ckpt burned the ~4-core-hour dump first. Dump CLI `--require-gate-inputs`
   calls `assert_inputs_match` against the pins BEFORE `run_graph_dump`, failing at launch.

Artifacts: `eval/graph_value_router_dump.py` (`source_fingerprint`, `assert_inputs_match`,
`run_graph_dump(source_commit=)` stamps `source`, `main()` `--source-commit`/`--require-gate-inputs`);
`train/graph_router_probe.py` (`GATE_FLOOR_ARTIFACT`/`GATE_FLOOR_SHA256`/`GATE_ACTOR_FLOOR_0038B`,
`canonical_actor_floor`, `assert_override_is_canonical`, `run_probe` override-validation + `source` stamp,
CLI `--actor-floor-exp0038b`/`--source-commit`). Tests: +2 dump, +2 probe, smoke extended (source stamp) →
37 graph tests green (was 33); full repo suite green. DIAGNOSTICS #13 extended (l)–(n). RESULT.md
corrections 21–23 + updated pod commands (`--require-gate-inputs`, `--source-commit`, `--actor-floor-exp0038b`)
+ test-count table (dump 14 / probe 17). spec.json corrections + status. Worktree UNCOMMITTED.

Next (unchanged): commit the EXP-0039 source, then launch s29/s77/s45 dump (CPU, `--require-gate-inputs
--source-commit <c>`, real archive+ckpts so recorded SHA-256 match the pins) → deterministic CPU probe
(3 model seeds, `--expect-decisions 13940 --expect-candidates 37804 --source-commit <c>`) → verdict
emitted + recomputable from predictions.csv + floor_by_task; fill RESULT.md Results table. Cloud spend
awaits user go-ahead.

---

## 2026-07-31 — EXP-0039 GNN probe: seventh review round (source identity ENFORCED + exact floor + pod-safe fallback)

Status: DONE (code + TDD). Seventh review found round 6's provenance was audit-only (recorded, not
enforced) plus two fallback defects; all verified against the code + HANDOVER.md and fixed test-first.
Schema hash UNCHANGED.

1. Source identity is now a REQUIRED cross-stage invariant (was audit-only). `_load_dump` never validated
   the dump's `source` — a zeroed hash + bogus commit was accepted (dump from source A → verdict under
   source B). Added `_validate_commit` (rejects None/`not-a-git-object` by git-hex shape) + `assert_source_matches`
   (dump `tree_sha256` must equal the LIVE probe fingerprint; commit must be valid AND equal the probe's
   `--source-commit`), called by `_load_dump(require_source=)` BEFORE any training; `run_probe(require_shape)`
   requires a non-null commit. Also WIDENED `source_fingerprint` from 4 hand-picked modules to the whole
   synced package tree (every `.py`), covering transitive producers (teacher/env/curriculum/actor/domain).
2. Override validation is EXACT + verdict uses pinned values. `assert_override_is_canonical` used `tol=1e-6`
   — a `+5e-7` inflation passed and would become the verdict floor (strict `<` can flip inside the
   interval). Now exact equality; `run_probe` additionally SUBSTITUTES `GATE_ACTOR_FLOOR_0038B` into the
   verdict so no drift survives.
3. Pod-safe canonical fallback. `--actor-floor-exp0038b`'s default artifact is under `experiments/…`,
   outside the synced pod tree (`/workspace/tokenomics`) — the documented fallback would fail on a clean
   pod. New `--actor-floor-canonical` uses the in-code pinned floor (identical values, no file).

Artifacts: `eval/graph_value_router_dump.py` (`source_fingerprint` now whole-tree `tree_sha256`+`n_files`);
`train/graph_router_probe.py` (`_validate_commit`, `assert_source_matches`, `_load_dump(expected_source_commit=,
require_source=)` cross-checks source before training, `run_probe` requires+passes commit and substitutes
pinned floor, `assert_override_is_canonical` exact, CLI `--actor-floor-canonical`). Tests: +1 probe
(`test_source_identity_is_a_required_cross_stage_invariant`), override test extended with the 5e-7 case,
dump/smoke source-key updated → 38 graph tests green (dump+probe 30, smoke+view 8); full repo suite pending.
DIAGNOSTICS #13 extended (o)–(q). RESULT.md corrections 24–26 + pod commands (`--source-commit` both stages,
`--actor-floor-canonical`) + test-count (probe 18). spec.json corrections + status. Worktree UNCOMMITTED.

Next: COMMIT the EXP-0039 source, then launch s29/s77/s45 dump (CPU, `--require-gate-inputs --source-commit
<c>`) → deterministic CPU probe (3 model seeds, `--expect-decisions 13940 --expect-candidates 37804
--source-commit <c>`, same commit both stages) → verdict recomputable from predictions.csv + floor_by_task;
fill RESULT.md Results table. Cloud spend awaits user go-ahead.

---

## 2026-07-31 — EXP-0039 GNN probe: eighth review round (commit bound to a real clean tree + runtime env manifest)

Status: DONE (code + TDD). Eighth review found the recorded commit was still forgeable (shape-only) and
the runtime env was uncaptured; both verified against the code + HANDOVER.md:413 and fixed test-first.
Schema hash UNCHANGED.

1. Commit bound to a real, clean revision (launcher-side). `_validate_commit` accepted any 7-40 hex
   (`a1b2c3d`/`deadbee`/40 zeros passed) — a dirty tree could be stamped with a bogus id whose tree hash
   still matched. Now `_validate_commit` requires a FULL 40-char sha; new `resolve_verified_commit`
   (needs git, launcher-side) resolves HEAD, verifies the object EXISTS (`git cat-file -e`) and the
   producing tree is CLEAN (`git status --porcelain` — so the live fingerprint corresponds to that
   commit), wired as `--verify-source-commit` on both CLIs. Dump CLI validates the commit shape BEFORE
   `run_graph_dump` (under `--require-gate-inputs`), so a missing/malformed commit fails at launch not
   after the dump budget. The pod has no `.git`, so existence/clean-tree is verified LOCALLY and the
   resolved sha passed via `--source-commit`.
2. Runtime environment captured + compared. Deps are open lower bounds (`torch>=2.2`) installed fresh on
   the pod; the probe recorded only `torch_version`. New `environment_manifest` (python/platform/
   implementation + numpy/torch/pettingzoo/gymnasium) is stamped into both specs; `_load_dump` compares
   the gated version keys exactly under `require_shape` — a probe cannot score a dump built under a
   different runtime.

Artifacts: `eval/graph_value_router_dump.py` (`resolve_verified_commit`, `environment_manifest`, dump
spec stamps `env`, dump CLI `--verify-source-commit` + pre-spend `_validate_commit`); `train/graph_router_probe.py`
(`_validate_commit` full-40, `assert_env_matches` + `_ENV_GATED_KEYS`, `_load_dump` compares env under
require_source, `run_probe` result stamps `env`, probe CLI `--verify-source-commit`). Tests: +2 probe
(`test_resolve_verified_commit_binds_a_clean_tree_to_a_real_commit`, `test_dump_and_probe_environment_manifest_is_captured_and_compared`),
Round-7 source test updated to full-40 shas + rejects abbreviations → 42 graph tests green (dump+probe 32,
smoke+view 8, +2 net). DIAGNOSTICS #13 extended (r)-(s). RESULT.md corrections 27-28 + pod guidance
(resolve_verified_commit one-liner, same-image env) + test-count (probe 20). spec.json corrections +
status. Worktree UNCOMMITTED. Full repo suite pending.

Next: resolve the clean-tree commit LOCALLY (`resolve_verified_commit` after committing the EXP-0039
source), then launch s29/s77/s45 dump (CPU, `--require-gate-inputs --source-commit <sha40>`) →
deterministic CPU probe on the SAME image (3 model seeds, `--expect-decisions 13940 --expect-candidates
37804 --source-commit <sha40>`) → verdict recomputable from predictions.csv + floor_by_task; fill
RESULT.md Results table. Cloud spend awaits user go-ahead.

---

## 2026-07-31 — EXP-0039 GNN probe: ninth review round (launcher-enforced commit binding + arch/BLAS env gate + count fix)

Status: DONE (code + TDD). Ninth review found the clean-commit binding was still manual (helper never
called by the launcher), the env gate ignored arch/backend, and recorded test counts were wrong. All
verified against the code and fixed test-first. Schema hash UNCHANGED.

1. Launcher-enforced clean-commit binding (was manual). `resolve_verified_commit` existed but
   `run_pod_training.py` never called it; the pod (no `.git`) only checks the 40-char shape, so forty
   zeros passed end-to-end. Added `--verify-source-commit` to the launcher: BEFORE rsync (the only place
   with `.git`) it resolves+verifies the clean 40-char HEAD via inline `resolve_clean_commit` (no
   numpy/torch import) and SUBSTITUTES it for a `{SOURCE_COMMIT}` placeholder in `--train`; dirty/non-repo
   aborts the launch. Binding no longer depends on prose.
2. Env gate covers arch + numeric backend. `_ENV_GATED_KEYS` compared only versions — a
   `machine="other-arch"`/`blas="different"` manifest was accepted. `environment_manifest` now records
   `machine`, `numpy_blas`, `torch_build` (hash of `torch.__config__`), and `image` (EXP0039_POD_IMAGE,
   audit); the first three are gated exactly.
3. Test counts corrected. Recorded 14/20/42 were wrong; actual collection is dump 13 + probe 19 + view 7
   + smoke 1 = 40 graph tests, plus 1 launcher binding test. Repo suite is the source of truth.

Artifacts: `run_pod_training.py` (`resolve_clean_commit` + `--verify-source-commit` injecting
`{SOURCE_COMMIT}` before rsync); `eval/graph_value_router_dump.py` (`environment_manifest` adds
machine/numpy_blas/torch_build/image via `_blas_identity`); `train/graph_router_probe.py`
(`_ENV_GATED_KEYS` += machine/numpy_blas/torch_build). Tests: +1 launcher
(`test_resolve_clean_commit_binds_a_clean_tree_and_injects_into_train` in test_run_pod_detached.py), env
test extended (arch/blas/build adversaries); dump+probe+launcher 57 passed. DIAGNOSTICS #13 extended
(t)-(u). RESULT.md corrections 29-31 + pod commands now `--verify-source-commit`/`{SOURCE_COMMIT}` + test
table fixed (13/19, total 40). spec.json corrections + status. Worktree UNCOMMITTED. Full suite pending.

Next: commit the EXP-0039 source, then launch both stages with `run_pod_training.py --verify-source-commit`
(injects the clean commit) — dump (CPU, `--require-gate-inputs`) → deterministic CPU probe on the SAME
image (3 model seeds, `--expect-decisions 13940 --expect-candidates 37804`) → verdict recomputable from
predictions.csv + floor_by_task; fill RESULT.md Results table. Cloud spend awaits user go-ahead.

---

## 2026-07-31 — EXP-0039 GNN probe: tenth review round (launcher itself in the verified set + pod image actually recorded)

Status: DONE (code + TDD). Tenth review closed two launcher-integration gaps; both verified against the
code and fixed test-first. Schema hash UNCHANGED.

1. Cleanliness now covers the launcher itself. `resolve_clean_commit` checked only the synced
   `tokenomics/` subtree — but `run_pod_training.py` lives outside it and IS producing code (resolves the
   commit, rewrites the command, launches). A launcher modified after HEAD (subtree clean) still returned
   the "clean" commit. Now checks the synced tree AND the launcher file (`Path(__file__)`) against HEAD,
   aborting if either is dirty — NOT the whole repo (Phase-7 pulls intentionally dirty experiments/).
   Placeholder injection extracted to tested helper `apply_source_commit`.
2. Pod image actually recorded. `environment_manifest` read `EXP0039_POD_IMAGE` but nothing set it →
   artifact recorded image: null. Launcher now `resolve_pod_image` (REST GET /v1/pods/<id> imageName) and
   exports `EXP0039_POD_IMAGE=<img>` into the remote run (detached + foreground, via `_env_prefix` on the
   python invocation), so env.image is populated. Best-effort (unresolved -> null, never fatal).

Artifacts: `run_pod_training.py` (`resolve_clean_commit(*also_paths)` incl. launcher, `apply_source_commit`,
`resolve_pod_image`, `_env_prefix`, `build_launch_cmd`/`train` take `env_exports`, main resolves+exports
the image). Tests: replaced the round-9 launcher test with `test_resolve_clean_commit_covers_synced_tree_and_launcher`
(dirty launcher outside subtree aborts) + `test_apply_source_commit_requires_and_fills_the_placeholder` +
`test_launch_cmd_exports_pod_image_into_the_run`; launcher suite 27 passed, probe 19 passed. DIAGNOSTICS
#13 extended (v)-(w). RESULT.md corrections 32-33 + pod-run notes. spec.json corrections + status.
Worktree UNCOMMITTED. Full repo suite pending.

Next: commit the EXP-0039 source (incl. run_pod_training.py so the launcher is clean), then launch both
stages with `run_pod_training.py --verify-source-commit` (verifies clean tree+launcher, injects the
commit, exports the pod image) — dump (CPU, `--require-gate-inputs`) → deterministic CPU probe on the SAME
image (3 model seeds, `--expect-decisions 13940 --expect-candidates 37804`) → verdict recomputable from
predictions.csv + floor_by_task; fill RESULT.md Results table. Cloud spend awaits user go-ahead.

---

### EXP-0039 pod run started; `load_shard` OOM found + fixed (2026-07-31)

Status: committed EXP-0039 source (`d6ad733`), launched the staged CPU pod run. **Stage 1 (dump)** on a
fresh kept pod (`6eb2gyf3vjo4cs`, `runpod/base:0.7.0-ubuntu2004`, 16 vCPU/cpu5c/32 GB): clean-commit
verified+injected, all three input SHA-256 matched the pins, produced **13940 decisions / 37804
candidates**. Stop-checks: decisions.csv **teacher/context columns byte-identical to the EXP-0038 dump
across all 13940 rows** (0 diffs); the only divergence is the deployed-actor floor (6715 rows, actor-
derived cols) = the anticipated pre-P2 actor-scoring drift → handled with the pre-registered
`--actor-floor-canonical` (pin the bar to the audited EXP-0038B floor).

**Stage 2 (probe) hit `Killed`/exit 137 (OOM) during dump LOAD** — a 32 GB pod died on a 124 MB dump.
Root-caused (not a pod-size workaround, DIAGNOSTICS #14): `load_shard` indexed the lazy `NpzFile` inside
the per-graph loop, and `NpzFile.__getitem__` re-decompresses the whole member every access →
O(n_graphs × shard_bytes). Fix (TDD): read each member once into a dict, slice views. Guard:
`test_load_shard_reads_each_npz_member_at_most_once` (RED before, GREEN after). Payoff: all 13940 graphs
load in **0.45 s / 246 MB** (was OOM). `write_shard` untouched → dump bytes identical; training is per-fold
(~1k graphs) so the 32 GB pod is sufficient. Also diagnosed the container-disk gotcha: a created pod's
`/workspace` resets on stop/start, so a kept-pod dump does not persist for a later probe stage.

Artifacts: `tokenomics/tokenomics/eval/graph_value_router_dump.py` (`load_shard` cache-once),
`tokenomics/tests/test_graph_dump.py` (+1 guard, 14 passed), DIAGNOSTICS #14, dump pulled to
`experiments/EXP-0039-gnn-observation-probe/dump_graph/`. Full repo suite **752 passed**.

Decisions: the load fix touches a hashed source file → both stages must **re-run under the new commit**
(dump bytes identical, only the provenance stamp changes; teacher parity preserved).

Blockers: none — root cause fixed and green.

Next: commit the fix; re-run dump (new commit, `--no-stop` to keep the pod up) → verify counts + teacher
parity → run the probe against the on-pod dump (same running pod) → pull `probe/`, read the verdict vs the
canonical floor, fill RESULT.md Results table; then terminate `6eb2gyf3vjo4cs` and delete
`tokenomics/_pod_ckpts/`.

---

### EXP-0039 CLOSED — probe FAILS; GNN does not recover the marginal-value target (2026-07-31)

Status: **DONE — negative result.** Committed the `load_shard` fix (`7c80a88`), re-ran both stages under
it on pod `6eb2gyf3vjo4cs` (same image, image-matched), then terminated the pod and removed
`tokenomics/_pod_ckpts/`. Dump re-run verified: 13940/37804, stamped `7c80a88` + new `tree_sha256`,
**0 teacher/context diffs** vs EXP-0038, decisions.csv byte-identical to the prior dump (determinism).
Probe determinism preflight **bitwise-exact** (`max_abs_diff 0.0 ≤ tol 0.0`).

**Verdict: FAIL (0/6 gate conditions, only `three_tasks`).** Held-out (leave-one-seed-out, mean of 3
model seeds): GNN mean regret **5.29 / 6.23 / 5.25** (s29/s77/s45) — sits **at the random floor**
(5.38/6.24/5.24); centered r² **−0.76 / −0.81 / −0.72** (negative → worse than predicting the mean);
never beats the pinned EXP-0038B actor floor (4.12/4.75/3.93); avg rel-reduction **−31%**. Train loss fell
≈49→1.5 per fold → the GNN **fits the training seeds but does not generalize** to a held-out seed of the
same topology (within-topology generalization gap, not optimization). Floor = `--actor-floor-canonical`
(the checked-in `residual_scorer.pt` is pre-P2, so its recomputed actor picks drift on 6715/13940 rows
while the teacher target is unchanged; pinned floor bypasses the drift).

Interpretation (plain English): a graph neural net given the full relational graph of deployable
decision-time state cannot predict which producer the 8-cycle marginal-value teacher prefers, on unseen
trajectory seeds — it does no better than routing at random. Richer relational observation did **not**
beat the EXP-0038B hand-metric wall. The teacher preference is not a stable function of
`graph_primitives_v1` decision-time state (consistent with DIAGNOSTICS #12 / EXP-0038). **Licenses no
deployed-GNN orderer experiment.** Scoped claim only (held-out trajectory, fixed topology).

Artifacts (git-lfs): `experiments/EXP-0039-gnn-observation-probe/artifacts/{dump_graph,probe}/` (probe: metrics_by_task.csv,
floor_by_task.csv, predictions.csv, checkpoints/, spec.json), RESULT.md Results filled, spec.json `result`
block, README index row, DIAGNOSTICS #14.

Decisions: STOP the "richer orderer observation" line for the marginal-value target. Next lever =
rollout/search as a compute lever, or a directly-observable held-out-learnable objective — not a bigger
observation.

Blockers: none. Pod terminated (`runpodctl pod delete 6eb2gyf3vjo4cs`), `pod list` empty, `_pod_ckpts`
removed.

Next: (housekeeping) decide whether to commit the large `dump_graph/`+`probe/` artifacts via git-lfs per
experiments/README convention, or keep them local — awaiting user preference on LFS.
