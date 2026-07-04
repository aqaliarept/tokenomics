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
